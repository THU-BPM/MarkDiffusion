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
from huggingface_hub import hf_hub_download

from ..base import BaseConfig, BaseWatermark
from utils.media_utils import get_random_latents, get_media_latents, transform_to_model_format
from utils.utils import set_random_seed
from visualize.data_for_visualization import DataForVisualization
from .gnr import GNRRestorer

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
	def _stream_key_encrypt(self, plaintext_bits: np.ndarray) -> np.ndarray:
		"""Encrypt plaintext bits using ChaCha20 stream cipher."""
		self._ensure_key_nonce()
		cipher = ChaCha20.new(key=self.key, nonce=self.nonce)  # 初始化 cipher
		packed = np.packbits(plaintext_bits).tobytes()
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

		# 延迟导入 GMDetector，避免循环导入
		from detection.gm.gm_detection import GMDetector
		
		# Build detector (delegates all detection work)
		self.detector = GMDetector(
			watermark_generator=self.watermark_generator,
			watermarking_mask=self.watermarking_mask,
			gt_patch=self.gt_patch,
			w_measurement=self.config.w_measurement,
			device=self.device,
			bit_threshold=self.watermark_generator.tau_bits,
			message_threshold=self.watermark_generator.tau_onebit,
			l1_threshold=None,
			gnr_checkpoint=self.config.gnr_checkpoint,
			gnr_classifier_type=self.config.gnr_classifier_type,
			gnr_model_nf=self.config.gnr_model_nf,
			gnr_binary_threshold=self.config.gnr_binary_threshold,
			gnr_use_for_decision=self.config.gnr_use_for_decision,
			gnr_threshold=self.config.gnr_threshold,
			fuser_checkpoint=self.config.fuser_checkpoint,
			fuser_threshold=self.config.fuser_threshold,
			fuser_frequency_scale=self.config.fuser_frequency_scale,
			huggingface_repo=self.config.huggingface_repo,
			hf_dir=self.config.hf_dir,
		)


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
		else:
			raise NotImplementedError(f"Unsupported watermark mask shape: {shape}")

		return mask

	# ------------------------------------------------------------------
	# Watermark injection / detection helpers
	# ------------------------------------------------------------------
	def _inject_complex(self, latents: torch.Tensor) -> torch.Tensor:
		fft_latents = torch.fft.fftshift(torch.fft.fft2(latents), dim=(-1, -2))
		target_patch = self.gt_patch
		if not torch.is_complex(target_patch):
			real = target_patch.to(torch.float32)
			imag = torch.zeros_like(real)
			target_patch = torch.complex(real, imag)
		target_patch = target_patch.to(fft_latents.dtype)
		
		mask = self.watermarking_mask
		if mask.dtype != torch.bool:
			fft_latents[mask != 0] = target_patch[mask != 0].clone()
		else:
			fft_latents[mask] = target_patch[mask].clone()
		injected = torch.fft.ifft2(torch.fft.ifftshift(fft_latents, dim=(-1, -2))).real
		return injected

	def inject_watermark(self, base_latents: torch.Tensor) -> torch.Tensor:
		base_latents = base_latents.to(self.device, dtype=torch.float32)
		injection = self.config.w_injection.lower()
		if "complex" in injection:
			watermarked = self._inject_complex(base_latents)
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
		self.huggingface_repo = cfg.get("huggingface_repo")
		self.fuser_checkpoint = cfg.get("fuser_checkpoint")
		self.fuser_threshold = cfg.get("fuser_threshold")
		self.fuser_frequency_scale = cfg.get("fuser_frequency_scale", 0.01)
		self.hf_dir = cfg.get("hf_dir")

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

		# Delegate detection to GMDetector
		return self.utils.detector.eval_watermark(
			reversed_latents=reversed_latents,
			detector_type=kwargs.get("detector_type", "bit_acc"),
		)

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

