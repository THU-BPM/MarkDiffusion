from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes
from scipy.special import betainc
from scipy.stats import norm, truncnorm

from ..base import BaseConfig, BaseWatermark
from utils.media_utils import get_random_latents, get_media_latents, transform_to_model_format
from utils.utils import set_random_seed
from visualize.data_for_visualization import DataForVisualization
from .gnr import GNRRestorer

import joblib


# -----------------------------------------------------------------------------
# Helper utilities adapted from the official GaussMarker implementation
# -----------------------------------------------------------------------------


def _bytes_from_seed(seed: Optional[int], length: int) -> bytes:
	"""Generate deterministic bytes using a Python PRNG seed."""
	if seed is None:
		return get_random_bytes(length)
	rng = random.Random(seed)
	return bytes(rng.getrandbits(8) for _ in range(length))


def circle_mask(size: int, radius: int, x_offset: int = 0, y_offset: int = 0) -> np.ndarray:
	"""Create a binary circle mask with optional offset."""
	x0 = y0 = size // 2
	x0 += x_offset
	y0 += y_offset
	grid_y, grid_x = np.ogrid[:size, :size]
	grid_y = grid_y[::-1]
	return ((grid_x - x0) ** 2 + (grid_y - y0) ** 2) <= radius ** 2


def set_complex_sign(original: torch.Tensor, sign_tensor: torch.Tensor) -> torch.Tensor:
	"""Apply complex-valued sign encoding (4-way) to a complex tensor."""
	real = original.real.abs()
	imag = original.imag.abs()

	sign_map_real = 1 - 2 * (sign_tensor >= 2).float()
	sign_map_imag = 1 - 2 * ((sign_tensor % 2) == 1).float()

	signed_real = real * sign_map_real
	signed_imag = imag * sign_map_imag

	return torch.complex(signed_real, signed_imag).to(original.dtype)


def extract_complex_sign(complex_tensor: torch.Tensor) -> torch.Tensor:
	"""Extract complex-valued sign encoding (4-way) from a complex tensor."""
	real = complex_tensor.real
	imag = complex_tensor.imag

	sign_map_real = (real <= 0).long()
	sign_map_imag = (imag <= 0).long()
	return 2 * sign_map_real + sign_map_imag


# -----------------------------------------------------------------------------
# Gaussian Shading watermark with ChaCha20 encryption (generalised dimensions)
# -----------------------------------------------------------------------------


@dataclass
class GaussianShadingChaCha:
	channel_copy: int
	width_copy: int
	height_copy: int
	fpr: float
	user_number: int
	latent_channels: int
	latent_height: int
	latent_width: int
	dtype: torch.dtype
	device: torch.device
	watermark_seed: Optional[int] = None
	key_seed: Optional[int] = None
	nonce_seed: Optional[int] = None
	watermark: Optional[torch.Tensor] = None
	key: Optional[bytes] = None
	nonce: Optional[bytes] = None
	message_bits: Optional[np.ndarray] = None

	def __post_init__(self) -> None:
		self.latentlength = self.latent_channels * self.latent_height * self.latent_width
		divisor = self.channel_copy * self.width_copy * self.height_copy
		if self.latentlength % divisor != 0:
			raise ValueError(
				"Latent volume is not divisible by channel/width/height copies. "
				"Please adjust w_copy/h_copy/channel_copy."
			)
		self.marklength = self.latentlength // divisor

		# Voting thresholds identical to official implementation
		if self.channel_copy == 1 and self.width_copy == 1 and self.height_copy == 1:
			self.threshold = 1
		else:
			self.threshold = self.channel_copy * self.width_copy * self.height_copy // 2

		self.tau_onebit: Optional[float] = None
		self.tau_bits: Optional[float] = None
		for i in range(self.marklength):
			fpr_onebit = betainc(i + 1, self.marklength - i, 0.5)
			fpr_bits = fpr_onebit * self.user_number
			if fpr_onebit <= self.fpr and self.tau_onebit is None:
				self.tau_onebit = i / self.marklength
			if fpr_bits <= self.fpr and self.tau_bits is None:
				self.tau_bits = i / self.marklength

	# ------------------------------------------------------------------
	# Key/nonce helpers
	# ------------------------------------------------------------------
	def _ensure_key_nonce(self) -> None:
		if self.key is None:
			self.key = _bytes_from_seed(self.key_seed, 32)
		if self.nonce is None:
			self.nonce = _bytes_from_seed(self.nonce_seed, 12)

	# ------------------------------------------------------------------
	# Sampling helpers
	# ------------------------------------------------------------------
	def _truncated_sampling(self, message_bits: np.ndarray) -> torch.Tensor:
		z = np.zeros(self.latentlength, dtype=np.float32)
		denominator = 2.0
		ppf = [norm.ppf(j / denominator) for j in range(int(denominator) + 1)]
		for idx in range(self.latentlength):
			dec_mes = reduce(lambda a, b: 2 * a + b, message_bits[idx : idx + 1])
			dec_mes = int(dec_mes)
			z[idx] = truncnorm.rvs(ppf[dec_mes], ppf[dec_mes + 1])
		tensor = torch.from_numpy(z).reshape(1, self.latent_channels, self.latent_height, self.latent_width)
		return tensor.to(self.device, dtype=torch.float32)

	def _generate_watermark(self) -> None:
		generator = torch.Generator(device="cpu")
		if self.watermark_seed is not None:
			generator.manual_seed(self.watermark_seed)

		watermark = torch.randint(
			low=0,
			high=2,
			size=(
				1,
				self.latent_channels // self.channel_copy,
				self.latent_height // self.width_copy,
				self.latent_width // self.height_copy,
			),
			generator=generator,
			dtype=torch.int64,
		)
		self.watermark = watermark.to(self.device)

		tiled = self.watermark.repeat(1, self.channel_copy, self.width_copy, self.height_copy)
		self.message_bits = self._stream_key_encrypt(tiled.flatten().cpu().numpy())

	# ------------------------------------------------------------------
	# Encryption helpers
	# ------------------------------------------------------------------
	def _stream_key_encrypt(self, spread_bits: np.ndarray) -> np.ndarray:
		self._ensure_key_nonce()
		cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
		packed = np.packbits(spread_bits).tobytes()
		encrypted = cipher.encrypt(packed)
		unpacked = np.unpackbits(np.frombuffer(encrypted, dtype=np.uint8))
		return unpacked[: self.latentlength]

	def _stream_key_decrypt(self, encrypted_bits: np.ndarray) -> torch.Tensor:
		self._ensure_key_nonce()
		cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
		packed = np.packbits(encrypted_bits).tobytes()
		decrypted = cipher.decrypt(packed)
		bits = np.unpackbits(np.frombuffer(decrypted, dtype=np.uint8))
		bits = bits[: self.latentlength]
		tensor = torch.from_numpy(bits.astype(np.uint8)).reshape(
			1, self.latent_channels, self.latent_height, self.latent_width
		)
		return tensor.to(self.device)

	# ------------------------------------------------------------------
	# Public API
	# ------------------------------------------------------------------
	def create_watermark_and_return_w_m(self) -> Tuple[torch.Tensor, torch.Tensor]:
		if self.watermark is None or self.message_bits is None:
			self._generate_watermark()
		message_bits = self.message_bits
		sampled = self._truncated_sampling(message_bits)
		sampled = sampled.to(self.device, dtype=torch.float32)
		m_tensor = torch.from_numpy(message_bits.astype(np.float32)).reshape(
			1, self.latent_channels, self.latent_height, self.latent_width
		).to(self.device)
		return sampled, m_tensor

	def diffusion_inverse(self, spread_tensor: torch.Tensor) -> torch.Tensor:
		tensor = spread_tensor.to(self.device).reshape(
			1,
			self.channel_copy,
			self.latent_channels // self.channel_copy,
			self.width_copy,
			self.latent_height // self.width_copy,
			self.height_copy,
			self.latent_width // self.height_copy,
		)
		# Move channel copy to front, height/width copies accordingly
		tensor = tensor.sum(dim=(1, 3, 5))
		vote = tensor.clone()
		vote[vote <= self.threshold] = 0
		vote[vote > self.threshold] = 1
		return vote.to(torch.int64)

	def pred_m_from_latent(self, reversed_latents: torch.Tensor) -> torch.Tensor:
		return (reversed_latents > 0).int().to(self.device)

	def pred_w_from_latent(self, reversed_latents: torch.Tensor) -> torch.Tensor:
		reversed_m = self.pred_m_from_latent(reversed_latents)
		spread_bits = reversed_m.flatten().detach().cpu().numpy().astype(np.uint8)
		decrypted = self._stream_key_decrypt(spread_bits)
		return self.diffusion_inverse(decrypted)

	def pred_w_from_m(self, reversed_m: torch.Tensor) -> torch.Tensor:
		spread_bits = reversed_m.flatten().detach().cpu().numpy().astype(np.uint8)
		decrypted = self._stream_key_decrypt(spread_bits)
		return self.diffusion_inverse(decrypted)

	def watermark_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
		if self.watermark is None:
			self._generate_watermark()
		device = device or self.device
		return self.watermark.to(device)


# -----------------------------------------------------------------------------
# Utility helpers for GaussMarker
# -----------------------------------------------------------------------------


class GMUtils:
	def __init__(self, config: "GMConfig") -> None:
		self.config = config
		self.device = config.device
		self.latent_shape = (
			1,
			config.latent_channels,
			config.latent_height,
			config.latent_width,
		)
		try:
			self.pipeline_dtype = next(config.pipe.unet.parameters()).dtype
		except StopIteration:
			self.pipeline_dtype = config.dtype

		watermark_cls = GaussianShadingChaCha
		self.watermark_generator = watermark_cls(
			channel_copy=config.channel_copy,
			width_copy=config.w_copy,
			height_copy=config.h_copy,
			fpr=config.fpr,
			user_number=config.user_number,
			latent_channels=config.latent_channels,
			latent_height=config.latent_height,
			latent_width=config.latent_width,
			dtype=torch.float32,
			device=torch.device(config.device),
			watermark_seed=config.watermark_seed,
			key_seed=config.chacha_key_seed,
			nonce_seed=config.chacha_nonce_seed,
		)

		# Pre-initialize watermark to keep deterministic behaviour
		set_random_seed(config.watermark_seed)
		self.base_watermark_latents, self.base_message = self.watermark_generator.create_watermark_and_return_w_m()
		self.base_message = self.base_message.to(self.device, dtype=torch.float32)

		self.radius_list = list(range(config.w_radius, 0, -1))
		self.gt_patch = self._build_watermarking_pattern()
		self.watermarking_mask = self._build_watermarking_mask()
		self.gnr_restorer = self._build_gnr_restorer()
		self.fuser = self._build_fuser()
		self.fuser_threshold = float(self.config.fuser_threshold) if self.config.fuser_threshold is not None else 0.5
		self.fuser_frequency_scale = float(self.config.fuser_frequency_scale)

	# ------------------------------------------------------------------
	# Pattern / mask construction
	# ------------------------------------------------------------------
	def _build_watermarking_pattern(self) -> torch.Tensor:
		set_random_seed(self.config.w_seed)
		base_latents = get_random_latents(
			pipe=self.config.pipe,
			height=self.config.image_size[0],
			width=self.config.image_size[1],
		).to(self.device, dtype=torch.float32)

		pattern = self.config.w_pattern.lower()
		if "seed_ring" in pattern:
			gt_patch = base_latents.clone()
			tmp = copy.deepcopy(gt_patch)
			for radius in self.radius_list:
				mask = torch.tensor(circle_mask(gt_patch.shape[-1], radius), device=self.device, dtype=torch.bool)
				for ch in range(gt_patch.shape[1]):
					gt_patch[:, ch, mask] = tmp[0, ch, 0, radius].item()
		elif "seed_zeros" in pattern:
			gt_patch = torch.zeros_like(base_latents)
		elif "seed_rand" in pattern:
			gt_patch = base_latents.clone()
		elif "rand" in pattern:
			gt_patch = torch.fft.fftshift(torch.fft.fft2(base_latents), dim=(-1, -2))
			gt_patch[:] = gt_patch[0]
		elif "zeros" in pattern:
			gt_patch = torch.fft.fftshift(torch.fft.fft2(base_latents), dim=(-1, -2)) * 0
		elif "const" in pattern:
			gt_patch = torch.fft.fftshift(torch.fft.fft2(base_latents), dim=(-1, -2)) * 0
			gt_patch += self.config.w_pattern_const
		elif "signal_ring" in pattern:
			gt_patch = torch.randint_like(base_latents, low=0, high=2, dtype=torch.int64)
			if self.config.w_length is None:
				self.config.w_length = len(self.radius_list) * base_latents.shape[1]
			watermark_signal = torch.randint(low=0, high=4, size=(self.config.w_length,))
			idx = 0
			for radius in self.radius_list:
				mask = torch.tensor(circle_mask(base_latents.shape[-1], radius), device=self.device)
				for ch in range(gt_patch.shape[1]):
					signal = watermark_signal[idx % len(watermark_signal)].item()
					gt_patch[:, ch, mask] = signal
					idx += 1
		else:  # default ring
			gt_patch = torch.fft.fftshift(torch.fft.fft2(base_latents), dim=(-1, -2))
			tmp = gt_patch.clone()
			for radius in self.radius_list:
				mask = torch.tensor(circle_mask(gt_patch.shape[-1], radius), device=self.device, dtype=torch.bool)
				for ch in range(gt_patch.shape[1]):
					gt_patch[:, ch, mask] = tmp[0, ch, 0, radius].item()
		return gt_patch.to(self.device)

	def _build_watermarking_mask(self) -> torch.Tensor:
		mask = torch.zeros(self.latent_shape, dtype=torch.bool, device=self.device)
		shape = self.config.w_mask_shape.lower()

		if shape == "circle":
			base_mask = torch.tensor(circle_mask(self.latent_shape[-1], self.config.w_radius), device=self.device)
			if self.config.w_channel == -1:
				mask[:, :, base_mask] = True
			else:
				mask[:, self.config.w_channel, base_mask] = True
		elif shape == "square":
			anchor = self.latent_shape[-1] // 2
			sl = slice(anchor - self.config.w_radius, anchor + self.config.w_radius)
			if self.config.w_channel == -1:
				mask[:, :, sl, sl] = True
			else:
				mask[:, self.config.w_channel, sl, sl] = True
		elif shape == "signal_circle":
			mask = torch.zeros(self.latent_shape, dtype=torch.long, device=self.device)
			label = 1
			for radius in self.radius_list:
				base_mask = torch.tensor(circle_mask(self.latent_shape[-1], radius), device=self.device)
				mask[:, :, base_mask] = label
				label += 1
		elif shape == "no":
			return mask
		else:
			raise NotImplementedError(f"Unsupported watermark mask shape: {shape}")

		return mask

	def _build_gnr_restorer(self) -> Optional[GNRRestorer]:
		checkpoint = self.config.gnr_checkpoint
		if not checkpoint:
			return None
		checkpoint_path = Path(checkpoint)
		if not checkpoint_path.is_file():
			relative_candidate = Path(__file__).resolve().parent / checkpoint
			if relative_candidate.is_file():
				checkpoint_path = relative_candidate
			else:
				raise FileNotFoundError(f"GNR checkpoint not found at '{checkpoint}'")
		in_channels = self.config.latent_channels * (2 if self.config.gnr_classifier_type == 1 else 1)
		return GNRRestorer(
			checkpoint_path=checkpoint_path,
			in_channels=in_channels,
			out_channels=self.config.latent_channels,
			nf=self.config.gnr_model_nf,
			device=torch.device(self.config.device),
			classifier_type=self.config.gnr_classifier_type,
			base_message=self.base_message if self.config.gnr_classifier_type == 1 else None,
		)

	def _build_fuser(self):
		checkpoint = self.config.fuser_checkpoint
		if not checkpoint:
			return None
		if joblib is None:
			raise ImportError(
				"joblib is required to load the GaussMarker fuser. Install joblib or disable the fuser."
			)
		candidates = [Path(checkpoint)]
		base_dir = Path(__file__).resolve().parent
		candidates.append(base_dir / checkpoint)
		candidates.append(base_dir.parent.parent / checkpoint)
		for candidate in candidates:
			if candidate.is_file():
				return joblib.load(candidate)
		raise FileNotFoundError(f"Fuser checkpoint not found at '{checkpoint}'")

	# ------------------------------------------------------------------
	# Watermark injection / detection helpers
	# ------------------------------------------------------------------
	def _inject_complex(self, latents: torch.Tensor) -> torch.Tensor:
		fft_latents = torch.fft.fftshift(torch.fft.fft2(latents), dim=(-1, -2))
		target_patch = self.gt_patch
		mask = self.watermarking_mask
		if mask.dtype != torch.bool:
			fft_latents[mask != 0] = target_patch[mask != 0].clone()
		else:
			fft_latents[mask] = target_patch[mask].clone()
		injected = torch.fft.ifft2(torch.fft.ifftshift(fft_latents, dim=(-1, -2))).real
		return injected

	def _inject_seed(self, latents: torch.Tensor) -> torch.Tensor:
		mask = self.watermarking_mask
		injected = latents.clone()
		injected[mask] = self.gt_patch[mask].clone()
		return injected

	def _inject_signal(self, latents: torch.Tensor) -> torch.Tensor:
		fft_latents = torch.fft.fftshift(torch.fft.fft2(latents), dim=(-1, -2))
		mask = self.watermarking_mask
		signals = extract_complex_sign(self.gt_patch)
		fft_latents_signal = set_complex_sign(fft_latents, signals)
		fft_latents[mask != 0] = fft_latents_signal[mask != 0]
		injected = torch.fft.ifft2(torch.fft.ifftshift(fft_latents, dim=(-1, -2))).real
		return injected

	def inject_watermark(self, base_latents: torch.Tensor) -> torch.Tensor:
		base_latents = base_latents.to(self.device, dtype=torch.float32)
		injection = self.config.w_injection.lower()
		if "complex" in injection:
			watermarked = self._inject_complex(base_latents)
		elif "seed" in injection:
			watermarked = self._inject_seed(base_latents)
		elif "signal" in injection:
			watermarked = self._inject_signal(base_latents)
		else:
			raise NotImplementedError(f"Unsupported injection mode: {self.config.w_injection}")
		return watermarked.to(self.config.dtype)

	def generate_watermarked_latents(self, seed: Optional[int] = None) -> torch.Tensor:
		if seed is None:
			seed = self.config.gen_seed
		set_random_seed(seed)
		sampled_latents, _ = self.watermark_generator.create_watermark_and_return_w_m()
		sampled_latents = sampled_latents.to(self.device, dtype=torch.float32)
		watermarked = self.inject_watermark(sampled_latents)
		target_dtype = self.pipeline_dtype or self.config.dtype
		return watermarked.to(target_dtype)

	def _compute_complex_l1(self, reversed_latents: torch.Tensor) -> float:
		fft_latents = torch.fft.fftshift(torch.fft.fft2(reversed_latents), dim=(-1, -2))
		target_patch = self.gt_patch
		mask = self.watermarking_mask
		if mask.dtype != torch.bool:
			selection = mask != 0
		else:
			selection = mask
		if selection.sum() == 0:
			return 0.0
		diff = torch.abs(fft_latents[selection] - target_patch[selection])
		return float(diff.mean().item())

	def detect_from_latents(self, reversed_latents: torch.Tensor, detector_type: Optional[str] = None) -> Dict[str, Union[float, bool]]:
		reversed_latents = reversed_latents.to(self.device, dtype=torch.float32)
		metrics: Dict[str, Union[float, bool]] = {}

		bit_watermark = self.watermark_generator.pred_w_from_latent(reversed_latents)
		reference_bits = self.watermark_generator.watermark_tensor(bit_watermark.device)
		bit_accuracy = (bit_watermark == reference_bits).float().mean().item()
		metrics["bit_accuracy"] = float(bit_accuracy)
		metrics["tau_bits"] = float(self.watermark_generator.tau_bits or 0.0)
		metrics["tau_onebit"] = float(self.watermark_generator.tau_onebit or 0.0)

		reversed_m = self.watermark_generator.pred_m_from_latent(reversed_latents)
		message_bits = torch.from_numpy(self.watermark_generator.message_bits.astype(np.float32)).to(reversed_m.device)
		m_accuracy = (reversed_m.flatten() == message_bits).float().mean().item()
		metrics["message_accuracy"] = float(m_accuracy)

		gnr_bit_accuracy = None
		gnr_message_accuracy = None
		if self.gnr_restorer is not None:
			restored_probs = self.gnr_restorer.restore(reversed_m)
			restored_binary = (restored_probs > self.config.gnr_binary_threshold).float()
			restored_w = self.watermark_generator.pred_w_from_m(restored_binary)
			gnr_bit_accuracy = (restored_w == reference_bits).float().mean().item()
			restored_message = restored_binary.flatten()
			gnr_message_accuracy = (restored_message == message_bits).float().mean().item()
			metrics["gnr_bit_accuracy"] = float(gnr_bit_accuracy)
			metrics["gnr_message_accuracy"] = float(gnr_message_accuracy)

		metrics["complex_l1"] = self._compute_complex_l1(reversed_latents)
		frequency_score = -metrics["complex_l1"] * self.fuser_frequency_scale
		metrics["frequency_score"] = float(frequency_score)

		threshold = self.watermark_generator.tau_bits or 0.5
		decision_threshold = threshold
		decision_bit_accuracy = bit_accuracy
		if self.gnr_restorer is not None and self.config.gnr_use_for_decision and gnr_bit_accuracy is not None:
			decision_bit_accuracy = max(decision_bit_accuracy, gnr_bit_accuracy)
			if self.config.gnr_threshold is not None:
				decision_threshold = float(self.config.gnr_threshold)
		metrics["decision_bit_accuracy"] = float(decision_bit_accuracy)
		metrics["decision_threshold"] = float(decision_threshold)

		fused_score = None
		if self.fuser is not None:
			spatial_score = gnr_bit_accuracy if (gnr_bit_accuracy is not None and self.config.gnr_use_for_decision) else bit_accuracy
			frequency_score = metrics["frequency_score"]
			features = np.array([[spatial_score, frequency_score]], dtype=np.float32)
			if hasattr(self.fuser, "predict_proba"):
				fused_score = float(self.fuser.predict_proba(features)[0, 1])
			elif hasattr(self.fuser, "decision_function"):
				fused_score = float(self.fuser.decision_function(features)[0])
			else:
				raise AttributeError("Unsupported fuser model: missing predict_proba/decision_function")
			metrics["fused_score"] = fused_score
			metrics["fused_threshold"] = float(self.fuser_threshold)
			metrics["is_watermark"] = bool(fused_score >= self.fuser_threshold)
		else:
			metrics["is_watermark"] = bool(decision_bit_accuracy >= decision_threshold)
		if gnr_bit_accuracy is not None:
			metrics["gnr_threshold"] = float(decision_threshold if self.config.gnr_use_for_decision and self.config.gnr_threshold is not None else threshold)

		if detector_type == "is_watermark":
			return {"is_watermark": metrics["is_watermark"]}
		if detector_type == "gnr_bit_acc" and gnr_bit_accuracy is not None:
			selected_threshold = decision_threshold if self.config.gnr_use_for_decision and self.config.gnr_threshold is not None else threshold
			return {
				"gnr_bit_accuracy": float(gnr_bit_accuracy),
				"threshold": float(selected_threshold),
				"is_watermark": bool(gnr_bit_accuracy >= selected_threshold),
			}
		if detector_type == "fused" and fused_score is not None:
			return {
				"fused_score": float(fused_score),
				"threshold": float(self.fuser_threshold),
				"is_watermark": bool(fused_score >= self.fuser_threshold),
			}
		return metrics


# -----------------------------------------------------------------------------
# Configuration for GaussMarker
# -----------------------------------------------------------------------------


class GMConfig(BaseConfig):
	def initialize_parameters(self) -> None:
		cfg = self.config_dict
		self.channel_copy = cfg.get("channel_copy", 1)
		self.w_copy = cfg.get("w_copy", 8)
		self.h_copy = cfg.get("h_copy", 8)
		self.user_number = cfg.get("user_number", 1_000_000)
		self.fpr = cfg.get("fpr", 1e-6)
		self.chacha_key_seed = cfg.get("chacha_key_seed")
		self.chacha_nonce_seed = cfg.get("chacha_nonce_seed")
		self.watermark_seed = cfg.get("watermark_seed", self.gen_seed)

		self.w_seed = cfg.get("w_seed", 999_999)
		self.w_channel = cfg.get("w_channel", -1)
		self.w_pattern = cfg.get("w_pattern", "ring")
		self.w_mask_shape = cfg.get("w_mask_shape", "circle")
		self.w_radius = cfg.get("w_radius", 4)
		self.w_measurement = cfg.get("w_measurement", "l1_complex")
		self.w_injection = cfg.get("w_injection", "complex")
		self.w_pattern_const = cfg.get("w_pattern_const", 0.0)
		self.w_length = cfg.get("w_length")

		self.gnr_checkpoint = cfg.get("gnr_checkpoint")
		self.gnr_classifier_type = cfg.get("gnr_classifier_type", 0)
		self.gnr_model_nf = cfg.get("gnr_model_nf", 128)
		self.gnr_binary_threshold = cfg.get("gnr_binary_threshold", 0.5)
		self.gnr_use_for_decision = cfg.get("gnr_use_for_decision", True)
		self.gnr_threshold = cfg.get("gnr_threshold")
		self.fuser_checkpoint = cfg.get("fuser_checkpoint")
		self.fuser_threshold = cfg.get("fuser_threshold")
		self.fuser_frequency_scale = cfg.get("fuser_frequency_scale", 0.01)

		self.latent_channels = self.pipe.unet.config.in_channels
		self.latent_height = self.image_size[0] // self.pipe.vae_scale_factor
		self.latent_width = self.image_size[1] // self.pipe.vae_scale_factor

		if self.latent_channels % self.channel_copy != 0:
			raise ValueError("channel_copy must divide latent channels")
		if self.latent_height % self.w_copy != 0 or self.latent_width % self.h_copy != 0:
			raise ValueError("w_copy and h_copy must divide latent spatial dimensions")

	@property
	def algorithm_name(self) -> str:
		return "GM"


# -----------------------------------------------------------------------------
# Main GaussMarker watermark class
# -----------------------------------------------------------------------------


class GM(BaseWatermark):
	def __init__(self, watermark_config: GMConfig, *args, **kwargs) -> None:
		self.config = watermark_config
		self.utils = GMUtils(self.config)
		super().__init__(self.config)

	def _generate_watermarked_image(self, prompt: str, *args, **kwargs) -> Image.Image:
		seed = kwargs.pop("seed", self.config.gen_seed)
		watermarked_latents = self.utils.generate_watermarked_latents(seed=seed)
		self.set_orig_watermarked_latents(watermarked_latents)

		generation_params = {
			"num_images_per_prompt": self.config.num_images,
			"guidance_scale": kwargs.pop("guidance_scale", self.config.guidance_scale),
			"num_inference_steps": kwargs.pop("num_inference_steps", self.config.num_inference_steps),
			"height": self.config.image_size[0],
			"width": self.config.image_size[1],
			"latents": watermarked_latents,
		}

		for key, value in self.config.gen_kwargs.items():
			generation_params.setdefault(key, value)
		generation_params.update(kwargs)
		generation_params["latents"] = watermarked_latents

		images = self.config.pipe(prompt, **generation_params).images
		return images[0]

	def _detect_watermark_in_image(
		self,
		image: Image.Image,
		prompt: str = "",
		*args,
		**kwargs,
	) -> Dict[str, Union[float, bool]]:
		guidance_scale = kwargs.get("guidance_scale", self.config.guidance_scale)
		num_steps = kwargs.get("num_inference_steps", self.config.num_inference_steps)

		do_cfg = guidance_scale > 1.0
		prompt_embeds, negative_embeds = self.config.pipe.encode_prompt(
			prompt=prompt,
			device=self.config.device,
			do_classifier_free_guidance=do_cfg,
			num_images_per_prompt=1,
		)
		if do_cfg:
			text_embeddings = torch.cat([negative_embeds, prompt_embeds])
		else:
			text_embeddings = prompt_embeds

		processed = transform_to_model_format(image, target_size=self.config.image_size[0]).unsqueeze(0).to(
			text_embeddings.dtype
		).to(self.config.device)
		image_latents = get_media_latents(
			pipe=self.config.pipe,
			media=processed,
			sample=False,
			decoder_inv=kwargs.get("decoder_inv", False),
		)

		inversion_kwargs = {
			key: val
			for key, val in kwargs.items()
			if key not in {"decoder_inv", "guidance_scale", "num_inference_steps", "detector_type"}
		}

		reversed_series = self.config.inversion.forward_diffusion(
			latents=image_latents,
			text_embeddings=text_embeddings,
			guidance_scale=guidance_scale,
			num_inference_steps=num_steps,
			**inversion_kwargs,
		)
		reversed_latents = reversed_series[-1]

		return self.utils.detect_from_latents(reversed_latents, detector_type=kwargs.get("detector_type"))

	def get_data_for_visualize(
		self,
		image: Image.Image,
		prompt: str = "",
		guidance_scale: Optional[float] = None,
		decoder_inv: bool = False,
		*args,
		**kwargs,
	) -> DataForVisualization:
		guidance = guidance_scale if guidance_scale is not None else self.config.guidance_scale
		set_random_seed(self.config.gen_seed)
		watermarked_latents = self.utils.generate_watermarked_latents(seed=self.config.gen_seed)

		generation_params = {
			"num_images_per_prompt": self.config.num_images,
			"guidance_scale": guidance,
			"num_inference_steps": self.config.num_inference_steps,
			"height": self.config.image_size[0],
			"width": self.config.image_size[1],
			"latents": watermarked_latents,
		}
		for key, value in self.config.gen_kwargs.items():
			generation_params.setdefault(key, value)

		watermarked_image = self.config.pipe(prompt, **generation_params).images[0]

		do_cfg = guidance > 1.0
		prompt_embeds, negative_embeds = self.config.pipe.encode_prompt(
			prompt=prompt,
			device=self.config.device,
			do_classifier_free_guidance=do_cfg,
			num_images_per_prompt=1,
		)
		text_embeddings = torch.cat([negative_embeds, prompt_embeds]) if do_cfg else prompt_embeds

		processed = transform_to_model_format(watermarked_image, target_size=self.config.image_size[0]).unsqueeze(0)
		processed = processed.to(text_embeddings.dtype).to(self.config.device)
		image_latents = get_media_latents(
			pipe=self.config.pipe,
			media=processed,
			sample=False,
			decoder_inv=decoder_inv,
		)

		reversed_series = self.config.inversion.forward_diffusion(
			latents=image_latents,
			text_embeddings=text_embeddings,
			guidance_scale=guidance,
			num_inference_steps=self.config.num_inversion_steps,
		)

		return DataForVisualization(
			config=self.config,
			utils=self.utils,
			orig_watermarked_latents=self.get_orig_watermarked_latents(),
			reversed_latents=reversed_series,
			image=image,
		)

