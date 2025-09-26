# Copyright 2025 THU-BPM MarkDiffusion.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GaussMarker detection utilities.

This module adapts the official GaussMarker detection pipeline to the
MarkDiffusion detection API. It evaluates recovered diffusion latents to
decide whether a watermark is present, reporting both hard decisions and
auxiliary scores (bit/message accuracies, frequency-domain distances).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch

import joblib

from detection.base import BaseDetector
from watermark.gm.gm import GaussianShadingChaCha, extract_complex_sign
from watermark.gm.gnr import GNRRestorer


class GMDetector(BaseDetector):
	"""Detector for GaussMarker watermarks.

	Args:
		watermark_generator: Instance of :class:`GaussianShadingChaCha` that
			holds the original watermark bits and ChaCha20 key stream.
		watermarking_mask: Frequency-domain mask (or label map) indicating the
			region that carries the watermark.
		gt_patch: Reference watermark pattern in the frequency domain.
		w_measurement: Measurement mode (e.g., ``"l1_complex"`` or
			``"signal_complex"``), mirroring the official implementation.
		device: Torch device used for evaluation.
		bit_threshold: Optional override for the bit-accuracy decision
			threshold. Defaults to the generator's ``tau_bits`` value.
		message_threshold: Optional threshold for message accuracy decisions.
		l1_threshold: Optional threshold for frequency L1 distance decisions
			(smaller is better).
	"""

	def __init__(
		self,
		watermark_generator: GaussianShadingChaCha,
		watermarking_mask: torch.Tensor,
		gt_patch: torch.Tensor,
		w_measurement: str,
		device: Union[str, torch.device],
		bit_threshold: Optional[float] = None,
		message_threshold: Optional[float] = None,
		l1_threshold: Optional[float] = None,
		gnr_checkpoint: Optional[Union[str, Path]] = None,
		gnr_classifier_type: int = 0,
		gnr_model_nf: int = 128,
		gnr_binary_threshold: float = 0.5,
		gnr_use_for_decision: bool = True,
		gnr_threshold: Optional[float] = None,
		fuser_checkpoint: Optional[Union[str, Path]] = None,
		fuser_threshold: Optional[float] = None,
		fuser_frequency_scale: float = 0.01,
	) -> None:
		self.generator = watermark_generator
		device = torch.device(device)
		# Ensure watermark/message buffers are initialised
		_, base_message = self.generator.create_watermark_and_return_w_m()
		self.base_message = base_message.to(device)
		watermarking_mask = watermarking_mask.to(device)
		gt_patch = gt_patch.to(device)

		threshold = bit_threshold if bit_threshold is not None else (
			self.generator.tau_bits or 0.5
		)
		super().__init__(threshold=float(threshold), device=device)

		self.watermarking_mask = watermarking_mask
		self.gt_patch = gt_patch
		self.w_measurement = w_measurement.lower()
		self.message_threshold = (
			float(message_threshold) if message_threshold is not None else float(self.generator.tau_onebit or threshold)
		)
		self.l1_threshold = float(l1_threshold) if l1_threshold is not None else None
		self.gnr_binary_threshold = gnr_binary_threshold
		self.gnr_use_for_decision = gnr_use_for_decision
		self.gnr_threshold = float(gnr_threshold) if gnr_threshold is not None else None
		self.gnr_restorer = self._build_gnr_restorer(
			checkpoint=gnr_checkpoint,
			device=device,
			classifier_type=gnr_classifier_type,
			nf=gnr_model_nf,
		)
		self.fuser = self._load_fuser(fuser_checkpoint)
		self.fuser_threshold = float(fuser_threshold) if fuser_threshold is not None else 0.5
		self.fuser_frequency_scale = float(fuser_frequency_scale)

	# ------------------------------------------------------------------
	# Helper computations
	# ------------------------------------------------------------------
	def _complex_l1(self, reversed_latents: torch.Tensor) -> float:
		fft_latents = torch.fft.fftshift(torch.fft.fft2(reversed_latents), dim=(-1, -2))
		if self.watermarking_mask.dtype == torch.bool:
			selector = self.watermarking_mask
		else:
			selector = self.watermarking_mask != 0
		if selector.sum() == 0:
			return 0.0
		diff = torch.abs(fft_latents[selector] - self.gt_patch[selector])
		return float(diff.mean().item())

	def _signal_accuracy(self, reversed_latents: torch.Tensor) -> float:
		fft_latents = torch.fft.fftshift(torch.fft.fft2(reversed_latents), dim=(-1, -2))
		if self.watermarking_mask.dtype == torch.bool:
			selector = self.watermarking_mask
		else:
			selector = self.watermarking_mask != 0
		if selector.sum() == 0:
			return 0.0
		latents_sign = extract_complex_sign(fft_latents[selector])
		target_sign = extract_complex_sign(self.gt_patch[selector])
		return float((latents_sign == target_sign).float().mean().item())

	def _build_gnr_restorer(
		self,
		checkpoint: Optional[Union[str, Path]],
		device: torch.device,
		classifier_type: int,
		nf: int,
	) -> Optional[GNRRestorer]:
		if not checkpoint:
			return None
		candidates = [Path(checkpoint)]
		base_dir = Path(__file__).resolve().parent
		candidates.append(base_dir / checkpoint)
		candidates.append(base_dir.parent.parent / checkpoint)
		for candidate in candidates:
			if candidate.is_file():
				checkpoint_path = candidate
				break
		else:
			raise FileNotFoundError(f"GNR checkpoint not found at '{checkpoint}'")
		latent_channels = self.base_message.shape[1]
		in_channels = latent_channels * (2 if classifier_type == 1 else 1)
		return GNRRestorer(
			checkpoint_path=checkpoint_path,
			in_channels=in_channels,
			out_channels=latent_channels,
			nf=nf,
			device=device,
			classifier_type=classifier_type,
			base_message=self.base_message if classifier_type == 1 else None,
		)

	def _load_fuser(self, checkpoint: Optional[Union[str, Path]]):
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
	# Public API
	# ------------------------------------------------------------------
	def eval_watermark(
		self,
		reversed_latents: torch.Tensor,
		reference_latents: Optional[torch.Tensor] = None,
		detector_type: str = "bit_acc",
	) -> Dict[str, Union[bool, float]]:
		detector_type = detector_type.lower()
		reversed_latents = reversed_latents.to(self.device, dtype=torch.float32)

		# Bit-level reconstruction
		bit_watermark = self.generator.pred_w_from_latent(reversed_latents)
		reference_bits = self.generator.watermark_tensor(self.device)
		bit_acc = float((bit_watermark == reference_bits).float().mean().item())

		# Message bit accuracy (post ChaCha decryption)
		reversed_m = self.generator.pred_m_from_latent(reversed_latents)
		message_bits = torch.from_numpy(self.generator.message_bits.astype("float32")).to(self.device)
		message_acc = float((reversed_m.flatten().float() == message_bits).float().mean().item())

		# Frequency-domain statistics
		complex_l1 = self._complex_l1(reversed_latents)
		signal_acc = self._signal_accuracy(reversed_latents) if "signal" in self.w_measurement else None

		gnr_bit_acc = None
		gnr_message_acc = None
		if self.gnr_restorer is not None:
			restored_binary = self.gnr_restorer.restore_binary(reversed_m, threshold=self.gnr_binary_threshold)
			restored_w = self.generator.pred_w_from_m(restored_binary)
			gnr_bit_acc = float((restored_w == reference_bits).float().mean().item())
			gnr_message_acc = float((restored_binary.flatten() == message_bits).float().mean().item())

		frequency_score = -complex_l1 * self.fuser_frequency_scale
		metrics: Dict[str, Union[bool, float]] = {
			"bit_acc": bit_acc,
			"message_acc": message_acc,
			"complex_l1": complex_l1,
			"frequency_score": frequency_score,
			"tau_bits": float(self.generator.tau_bits or 0.5),
			"tau_onebit": float(self.generator.tau_onebit or 0.5),
		}
		if signal_acc is not None:
			metrics["signal_acc"] = signal_acc
		if gnr_bit_acc is not None:
			metrics["gnr_bit_acc"] = gnr_bit_acc
		if gnr_message_acc is not None:
			metrics["gnr_message_acc"] = gnr_message_acc

		# Determine binary decision based on requested detector type
		decision_threshold = self.threshold if self.gnr_threshold is None else self.gnr_threshold
		decision_bit_acc = bit_acc
		if self.gnr_restorer is not None and self.gnr_use_for_decision and gnr_bit_acc is not None:
			decision_bit_acc = max(decision_bit_acc, gnr_bit_acc)
		metrics["decision_bit_acc"] = decision_bit_acc
		metrics["decision_threshold"] = decision_threshold

		fused_score = None
		fused_threshold = self.fuser_threshold if self.fuser is not None else None
		if self.fuser is not None:
			spatial_score = gnr_bit_acc if gnr_bit_acc is not None else bit_acc
			frequency_score = metrics["frequency_score"]
			features = np.array([[spatial_score, frequency_score]], dtype=np.float32)
			if hasattr(self.fuser, "predict_proba"):
				fused_score = float(self.fuser.predict_proba(features)[0, 1])
			elif hasattr(self.fuser, "decision_function"):
				fused_score = float(self.fuser.decision_function(features)[0])
			else:
				raise AttributeError("Unsupported fuser model: missing predict_proba/decision_function")
			metrics["fused_score"] = fused_score
			metrics["fused_threshold"] = fused_threshold

		if detector_type == "message_acc":
			is_watermark = message_acc >= self.message_threshold
		elif detector_type == "complex_l1":
			threshold = self.l1_threshold if self.l1_threshold is not None else self.threshold
			is_watermark = complex_l1 <= threshold
		elif detector_type == "signal_acc":
			if signal_acc is None:
				raise ValueError("Signal accuracy requested but watermark measurement does not use signal mode.")
			is_watermark = signal_acc >= self.threshold
		elif detector_type == "gnr_bit_acc":
			if gnr_bit_acc is None:
				raise ValueError("GNR checkpoint not provided, cannot compute GNR-based accuracy.")
			is_watermark = gnr_bit_acc >= decision_threshold
		elif detector_type == "fused":
			if fused_score is None:
				raise ValueError("Fuser checkpoint not provided, cannot compute fused score.")
			is_watermark = fused_score >= fused_threshold
		elif detector_type == "all":
			if fused_score is not None:
				is_watermark = fused_score >= fused_threshold
			else:
				is_watermark = decision_bit_acc >= decision_threshold
		else:
			if detector_type in {"bit_acc", "is_watermark"} and fused_score is not None:
				is_watermark = fused_score >= fused_threshold
			elif detector_type in {"bit_acc", "is_watermark"}:
				is_watermark = decision_bit_acc >= decision_threshold
			else:
				raise ValueError(f"Unsupported detector_type '{detector_type}' for GaussMarker.")

		metrics["is_watermark"] = bool(is_watermark)
		return metrics
