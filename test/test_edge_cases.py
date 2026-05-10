"""
Tests for edge cases

Run with

pytest test/test_edge_cases.py -v
"""

import torch

class TestEdgeCases:
    def test_signal_accuracy_bool_mask_empty(self):
        import markdiffusion.detection.gm.gm_detection as gm_detection
        class G:
            def __init__(self, H=4, W=4):
                self.tau_bits = 0.5
                self.tau_onebit = 0.5
                self.message_bits = __import__("numpy").zeros(H * W, dtype="float32")
                self.H, self.W = H, W
            def create_watermark_and_return_w_m(self):
                return None, torch.zeros(1, 2)
            def watermark_tensor(self, device):
                return torch.ones(1, 1, self.H, self.W, device=device, dtype=torch.float32)
            def pred_w_from_latent(self, reversed_latents):
                return torch.ones_like(reversed_latents[:, :1, :, :], dtype=torch.float32)
            def pred_m_from_latent(self, reversed_latents):
                return torch.zeros(1, 1, self.H, self.W, device=reversed_latents.device)
            def pred_w_from_m(self, m):
                return torch.ones_like(m, dtype=torch.float32)

        device = "cpu"
        H, W = 4, 4
        mask = torch.zeros(1, 1, H, W, dtype=torch.bool, device=device)
        gt_patch = torch.zeros(1, 1, H, W, dtype=torch.complex64, device=device)
        detector = gm_detection.GMDetector(
            watermark_generator=G(H, W),
            watermarking_mask=mask,
            gt_patch=gt_patch,
            w_measurement="signal_complex",
            device=device,
        )
        reversed_latents = torch.randn(1, 1, H, W, device=device)
        metrics = detector.eval_watermark(reversed_latents, detector_type="signal_acc")
        assert metrics["signal_acc"] == 0.0
        assert metrics["is_watermarked"] is False

    def test_signal_accuracy_numeric_mask_match(self):
        import torch
        import markdiffusion.detection.gm.gm_detection as gm_detection

        class G:
            def __init__(self, H=4, W=4):
                self.tau_bits = 0.5
                self.tau_onebit = 0.5
                self.message_bits = __import__("numpy").zeros(H * W, dtype="float32")
                self.H, self.W = H, W
            def create_watermark_and_return_w_m(self):
                return None, torch.zeros(1, 2)
            def watermark_tensor(self, device):
                return torch.ones(1, 1, self.H, self.W, device=device, dtype=torch.float32)
            def pred_w_from_latent(self, reversed_latents):
                return torch.ones_like(reversed_latents[:, :1, :, :], dtype=torch.float32)
            def pred_m_from_latent(self, reversed_latents):
                return torch.zeros(1, 1, self.H, self.W, device=reversed_latents.device)
            def pred_w_from_m(self, m):
                return torch.ones_like(m, dtype=torch.float32)

        device = "cpu"
        H, W = 4, 4
        mask = torch.ones(1, 1, H, W, dtype=torch.float32, device=device)
        reversed_latents = torch.randn(1, 1, H, W, device=device)
        fft_latents = torch.fft.fftshift(torch.fft.fft2(reversed_latents), dim=(-1, -2))
        gt_patch = fft_latents.clone()
        detector = gm_detection.GMDetector(
            watermark_generator=G(H, W),
            watermarking_mask=mask,
            gt_patch=gt_patch,
            w_measurement="signal_complex",
            device=device,
        )
        metrics = detector.eval_watermark(reversed_latents, detector_type="signal_acc")
        assert abs(metrics["signal_acc"] - 1.0) < 1e-6
        assert metrics["is_watermarked"] is True

    def test_gnr_checkpoint_not_found_local_candidates(self):
        import torch
        import pytest
        import markdiffusion.detection.gm.gm_detection as gm_detection

        class G:
            def __init__(self, H=4, W=4):
                self.tau_bits = 0.5
                self.tau_onebit = 0.5
                self.message_bits = __import__("numpy").zeros(H * W, dtype="float32")
                self.H, self.W = H, W
            def create_watermark_and_return_w_m(self):
                return None, torch.zeros(1, 2)
            def watermark_tensor(self, device):
                return torch.ones(1, 1, self.H, self.W, device=device, dtype=torch.float32)
            def pred_w_from_latent(self, reversed_latents):
                return torch.ones_like(reversed_latents[:, :1, :, :], dtype=torch.float32)
            def pred_m_from_latent(self, reversed_latents):
                return torch.zeros(1, 1, self.H, self.W, device=reversed_latents.device)
            def pred_w_from_m(self, m):
                return torch.ones_like(m, dtype=torch.float32)

        device = "cpu"
        H, W = 4, 4
        mask = torch.ones(1, 1, H, W, dtype=torch.float32, device=device)
        gt_patch = torch.zeros(1, 1, H, W, dtype=torch.complex64, device=device)
        with pytest.raises(FileNotFoundError, match="GNR checkpoint not found"):
            gm_detection.GMDetector(
                watermark_generator=G(H, W),
                watermarking_mask=mask,
                gt_patch=gt_patch,
                w_measurement="l1_complex",
                device=device,
                gnr_checkpoint="not_exist_file.pth",
            )

    def test_fuser_hf_snapshot_fallback_failure(self, monkeypatch):
        import torch
        import pytest
        import markdiffusion.detection.gm.gm_detection as gm_detection

        class G:
            def __init__(self, H=4, W=4):
                self.tau_bits = 0.5
                self.tau_onebit = 0.5
                self.message_bits = __import__("numpy").zeros(H * W, dtype="float32")
                self.H, self.W = H, W
            def create_watermark_and_return_w_m(self):
                return None, torch.zeros(1, 2)
            def watermark_tensor(self, device):
                return torch.ones(1, 1, self.H, self.W, device=device, dtype=torch.float32)
            def pred_w_from_latent(self, reversed_latents):
                return torch.ones_like(reversed_latents[:, :1, :, :], dtype=torch.float32)
            def pred_m_from_latent(self, reversed_latents):
                return torch.zeros(1, 1, self.H, self.W, device=reversed_latents.device)
            def pred_w_from_m(self, m):
                return torch.ones_like(m, dtype=torch.float32)

        def fake_hf_hub_download(*args, **kwargs):
            raise RuntimeError("fail")

        monkeypatch.setattr(gm_detection, "hf_hub_download", fake_hf_hub_download)

        device = "cpu"
        H, W = 4, 4
        mask = torch.ones(1, 1, H, W, dtype=torch.float32, device=device)
        gt_patch = torch.zeros(1, 1, H, W, dtype=torch.complex64, device=device)
        with pytest.raises(FileNotFoundError, match="Fuser checkpoint not found"):
            gm_detection.GMDetector(
                watermark_generator=G(H, W),
                watermarking_mask=mask,
                gt_patch=gt_patch,
                w_measurement="l1_complex",
                device=device,
                fuser_checkpoint="fake.pkl",
                huggingface_repo="Any/Repo",
                hf_dir="model_from_hf",
            )

    def test_fuser_local_candidates_failure(self):
        import torch
        import pytest
        import markdiffusion.detection.gm.gm_detection as gm_detection

        class G:
            def __init__(self, H=4, W=4):
                self.tau_bits = 0.5
                self.tau_onebit = 0.5
                self.message_bits = __import__("numpy").zeros(H * W, dtype="float32")
                self.H, self.W = H, W
            def create_watermark_and_return_w_m(self):
                return None, torch.zeros(1, 2)
            def watermark_tensor(self, device):
                return torch.ones(1, 1, self.H, self.W, device=device, dtype=torch.float32)
            def pred_w_from_latent(self, reversed_latents):
                return torch.ones_like(reversed_latents[:, :1, :, :], dtype=torch.float32)
            def pred_m_from_latent(self, reversed_latents):
                return torch.zeros(1, 1, self.H, self.W, device=reversed_latents.device)
            def pred_w_from_m(self, m):
                return torch.ones_like(m, dtype=torch.float32)

        device = "cpu"
        H, W = 4, 4
        mask = torch.ones(1, 1, H, W, dtype=torch.float32, device=device)
        gt_patch = torch.zeros(1, 1, H, W, dtype=torch.complex64, device=device)
        with pytest.raises(FileNotFoundError, match="Fuser checkpoint not found"):
            gm_detection.GMDetector(
                watermark_generator=G(H, W),
                watermarking_mask=mask,
                gt_patch=gt_patch,
                w_measurement="l1_complex",
                device=device,
                fuser_checkpoint="not_exist_fuser.pkl",
            )

    def test_fused_path_with_dummy_joblib_model(self, tmp_path, monkeypatch):
        import torch
        import numpy as np
        import joblib
        import markdiffusion.detection.gm.gm_detection as gm_detection

        class G:
            def __init__(self, H=4, W=4):
                self.tau_bits = 0.5
                self.tau_onebit = 0.5
                self.message_bits = np.zeros(H * W, dtype=np.float32)
                self.H, self.W = H, W
            def create_watermark_and_return_w_m(self):
                return None, torch.zeros(1, 2)
            def watermark_tensor(self, device):
                return torch.ones(1, 1, self.H, self.W, device=device, dtype=torch.float32)
            def pred_w_from_latent(self, reversed_latents):
                return torch.ones_like(reversed_latents[:, :1, :, :], dtype=torch.float32)
            def pred_m_from_latent(self, reversed_latents):
                return torch.zeros(1, 1, self.H, self.W, device=reversed_latents.device)
            def pred_w_from_m(self, m):
                return torch.ones_like(m, dtype=torch.float32)

        class Fuser:
            def predict_proba(self, X):
                return np.array([[0.1, 0.9]], dtype=np.float32)

        path = tmp_path / "f.pkl"
        with open(path, "wb") as f:
            f.write(b"")

        def fake_joblib_load(p):
            assert str(p) == str(path)
            return Fuser()

        monkeypatch.setattr(joblib, "load", fake_joblib_load)

        device = "cpu"
        H, W = 4, 4
        mask = torch.ones(1, 1, H, W, dtype=torch.float32, device=device)
        gt_patch = torch.zeros(1, 1, H, W, dtype=torch.complex64, device=device)
        detector = gm_detection.GMDetector(
            watermark_generator=G(H, W),
            watermarking_mask=mask,
            gt_patch=gt_patch,
            w_measurement="l1_complex",
            device=device,
            fuser_checkpoint=str(path),
        )
        reversed_latents = torch.randn(1, 1, H, W, device=device)
        metrics = detector.eval_watermark(reversed_latents, detector_type="fused")
        assert abs(metrics["fused_score"] - 0.9) < 1e-6
        assert metrics["is_watermarked"] is True
    
    # robin
    def test_robin_l1_distance_resizes_mask_and_gt_patch(self):
        from markdiffusion.detection.robin.robin_detection import ROBINDetector
        device = torch.device("cpu")
        mask_small = torch.ones(1, 1, 4, 4, dtype=torch.bool, device=device)
        gt_patch_small = torch.ones(1, 1, 4, 4, device=device, dtype=torch.complex64)
        reversed_latents_big = torch.randn(1, 1, 8, 8, device=device)

        det = ROBINDetector(
            watermarking_mask=mask_small,
            gt_patch=gt_patch_small,
            threshold=0.5,
            device=device,
        )
        out = det.eval_watermark(reversed_latents_big, detector_type="l1_distance")
        assert isinstance(out, dict)
        assert "l1_distance" in out
        assert "is_watermarked" in out

    def test_robin_cosine_similarity_resizes_with_float_mask(self):
        import torch
        from markdiffusion.detection.robin.robin_detection import ROBINDetector

        device = torch.device("cpu")
        mask_small = torch.ones(1, 1, 4, 4, dtype=torch.float32, device=device)
        gt_patch_small = (torch.randn(1, 1, 4, 4, device=device)
                        + 1j * torch.randn(1, 1, 4, 4, device=device)).to(torch.complex64)
        reversed_latents_big = torch.randn(1, 1, 8, 8, device=device)

        det = ROBINDetector(
            watermarking_mask=mask_small,
            gt_patch=gt_patch_small,
            threshold=0.0,
            device=device,
        )
        out = det.eval_watermark(reversed_latents_big, detector_type="cosine_similarity")
        assert isinstance(out, dict)
        assert "cosine_similarity" in out
        assert "is_watermarked" in out

    def test_robin_unsupported_detector_type_raises(self):
        import torch
        import pytest
        from markdiffusion.detection.robin.robin_detection import ROBINDetector

        device = torch.device("cpu")
        mask = torch.ones(1, 1, 8, 8, dtype=torch.bool, device=device)
        gt_patch = torch.ones(1, 1, 8, 8, device=device, dtype=torch.complex64)
        reversed_latents = torch.randn(1, 1, 8, 8, device=device)

        det = ROBINDetector(
            watermarking_mask=mask,
            gt_patch=gt_patch,
            threshold=0.5,
            device=device,
        )
        with pytest.raises(Exception):
            det.eval_watermark(reversed_latents, detector_type="unknown_type")

    # videoshield
    def test_videoshield_vote_threshold_one(self):
        import torch
        from markdiffusion.detection.videoshield.videoshield_detection import VideoShieldDetector

        device = torch.device("cpu")
        watermark = torch.randint(0, 2, (1, 1, 4, 4), dtype=torch.uint8, device=device)
        det = VideoShieldDetector(
            watermark=watermark,
            threshold=0.5,
            device=device,
            k_f=1, k_c=1, k_h=1, k_w=1
        )
        assert det.vote_threshold == 1

    def test_videoshield_stream_key_no_keys(self):
        import numpy as np
        import torch
        from markdiffusion.detection.videoshield.videoshield_detection import VideoShieldDetector

        device = torch.device("cpu")
        watermark = torch.randint(0, 2, (1, 1, 4, 4), dtype=torch.uint8, device=device)
        det = VideoShieldDetector(watermark=watermark, threshold=0.5, device=device)
        bits = np.random.randint(0, 2, size=32, dtype=np.uint8)
        out = det._stream_key_decrypt(bits)
        assert (out == bits).all()

    def test_videoshield_stream_key_with_keys(self):
        import numpy as np
        import torch
        from markdiffusion.detection.videoshield.videoshield_detection import VideoShieldDetector

        device = torch.device("cpu")
        watermark = torch.randint(0, 2, (1, 1, 4, 4), dtype=torch.uint8, device=device)
        key = b"k" * 32
        nonce = b"n" * 8
        det = VideoShieldDetector(
            watermark=watermark,
            threshold=0.5,
            device=device,
            chacha_key=key,
            chacha_nonce=nonce
        )
        bits = np.random.randint(0, 2, size=64, dtype=np.uint8)
        out = det._stream_key_decrypt(bits)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.uint8
        assert out.size >= 1

    def test_videoshield_video_diffusion_inverse_mismatch_expected_frames(self):
        import torch
        from markdiffusion.detection.videoshield.videoshield_detection import VideoShieldDetector

        device = torch.device("cpu")
        watermark = torch.randint(0, 2, (1, 1, 4, 4), dtype=torch.uint8, device=device)
        det = VideoShieldDetector(
            watermark=watermark,
            threshold=0.5,
            device=device,
            num_frames=8,
            k_f=4, k_c=1, k_h=2, k_w=2
        )
        wm_r = torch.randint(0, 2, (1, 1, 10, 4, 4), dtype=torch.uint8, device=device)
        out = det._diffusion_inverse(wm_r, is_video=True)
        assert isinstance(out, torch.Tensor)

    def test_videoshield_video_diffusion_inverse_remainder_alignment(self):
        import torch
        from markdiffusion.detection.videoshield.videoshield_detection import VideoShieldDetector

        device = torch.device("cpu")
        watermark = torch.randint(0, 2, (1, 1, 4, 4), dtype=torch.uint8, device=device)
        det = VideoShieldDetector(
            watermark=watermark,
            threshold=0.5,
            device=device,
            num_frames=0,
            k_f=8, k_c=1, k_h=2, k_w=2
        )
        wm_r = torch.randint(0, 2, (1, 1, 10, 4, 4), dtype=torch.uint8, device=device)
        out = det._diffusion_inverse(wm_r, is_video=True)
        assert isinstance(out, torch.Tensor)

    def test_videoshield_video_diffusion_inverse_exception_returns_zeros(self, monkeypatch):
        import torch
        from markdiffusion.detection.videoshield import videoshield_detection as vs_det
        from markdiffusion.detection.videoshield.videoshield_detection import VideoShieldDetector

        device = torch.device("cpu")
        watermark = torch.randint(0, 2, (1, 1, 4, 4), dtype=torch.uint8, device=device)
        det = VideoShieldDetector(
            watermark=watermark,
            threshold=0.5,
            device=device,
            k_f=2, k_c=1, k_h=2, k_w=2
        )
        monkeypatch.setattr(vs_det.torch, "split", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fail")))
        wm_r = torch.randint(0, 2, (1, 1, 4, 4, 4), dtype=torch.uint8, device=device)
        out = det._diffusion_inverse(wm_r, is_video=True)
        assert isinstance(out, torch.Tensor)
        assert out.shape == watermark.shape
        assert torch.all(out == 0)

    def test_videoshield_eval_unsupported_detector_type_raises(self):
        import torch
        import pytest
        from markdiffusion.detection.videoshield.videoshield_detection import VideoShieldDetector

        device = torch.device("cpu")
        watermark = torch.randint(0, 2, (1, 1, 4, 4), dtype=torch.uint8, device=device)
        det = VideoShieldDetector(watermark=watermark, threshold=0.5, device=device)
        lat = torch.randn(1, 1, 4, 4, 4, device=device)
        with pytest.raises(ValueError):
            det.eval_watermark(lat, detector_type="standard")

    def test_videoshield_eval_empty_latents_returns_false(self):
        import torch
        from markdiffusion.detection.videoshield.videoshield_detection import VideoShieldDetector

        device = torch.device("cpu")
        watermark = torch.randint(0, 2, (1, 1, 2, 2), dtype=torch.uint8, device=device)
        det = VideoShieldDetector(watermark=watermark, threshold=0.5, device=device)
        lat = torch.empty(0, device=device)
        out = det.eval_watermark(lat)
        assert isinstance(out, dict)
        assert out["is_watermarked"] is False
        assert out["bit_acc"] == 0.0

    def test_videoshield_eval_video_basic_no_keys(self, monkeypatch):
        import torch
        from markdiffusion.detection.videoshield import videoshield_detection as vs_det
        from markdiffusion.detection.videoshield.videoshield_detection import VideoShieldDetector

        
        device = torch.device("cpu")

        # Latents: [B, C, F, H, W] = [1, 1, 4, 4, 4]
        lat = torch.randn(1, 1, 4, 4, 4, device=device)

        # With k_f=2, k_h=2, k_w=2, expected vote shape is [1, 1, 2, 2, 2]
        watermark = torch.randint(0, 2, (1, 1, 2, 2, 2), dtype=torch.uint8, device=device)

        det = VideoShieldDetector(
            watermark=watermark,
            threshold=0.0,
            device=device,
            k_f=2, k_c=1, k_h=2, k_w=2
        )

        # Patch torch.from_numpy so the no-keys path (Tensor) won’t raise TypeError
        orig_from_numpy = vs_det.torch.from_numpy
        def safe_from_numpy(x):
            import torch as _torch
            return x if isinstance(x, _torch.Tensor) else orig_from_numpy(x)
        monkeypatch.setattr(vs_det.torch, "from_numpy", safe_from_numpy)

        out = det.eval_watermark(lat, detector_type="bit_acc")
        assert isinstance(out, dict)
        assert "bit_acc" in out
        assert "is_watermarked" in out

    def test_videoshield_eval_video_with_keys(self):
        import torch
        from markdiffusion.detection.videoshield.videoshield_detection import VideoShieldDetector

        device = torch.device("cpu")

        # Latents: [B, C, F, H, W] = [1, 1, 4, 4, 4]
        lat = torch.randn(1, 1, 4, 4, 4, device=device)

        # With k_f=2, k_h=2, k_w=2, expected vote shape is [1, 1, 2, 2, 2]
        watermark = torch.randint(0, 2, (1, 1, 2, 2, 2), dtype=torch.uint8, device=device)

        key = b"k" * 32
        nonce = b"n" * 8

        det = VideoShieldDetector(
            watermark=watermark,
            threshold=0.0,
            device=device,
            chacha_key=key,
            chacha_nonce=nonce,
            k_f=2, k_c=1, k_h=2, k_w=2
        )

        out = det.eval_watermark(lat, detector_type="bit_acc")
        assert isinstance(out, dict)
        assert "bit_acc" in out
        assert "is_watermarked" in out
    
    def test_videoshield_eval_image_case_no_keys_monkeypatch_inverse(self, monkeypatch):
        import torch
        from markdiffusion.detection.videoshield import videoshield_detection as vs_det
        from markdiffusion.detection.videoshield.videoshield_detection import VideoShieldDetector

        device = torch.device("cpu")
        watermark = torch.randint(0, 2, (1, 1, 2, 2), dtype=torch.uint8, device=device)
        lat = torch.randn(1, 1, 2, 2, device=device)  # 4D torch tensor

        det = VideoShieldDetector(
            watermark=watermark,
            threshold=0.0,
            device=device,
            k_f=2, k_c=1, k_h=2, k_w=2
        )

        # Patch torch.from_numpy to accept tensors in the image-case no-keys path
        orig_from_numpy = vs_det.torch.from_numpy
        def safe_from_numpy(x):
            import torch as _torch
            return x if isinstance(x, _torch.Tensor) else orig_from_numpy(x)
        monkeypatch.setattr(vs_det.torch, "from_numpy", safe_from_numpy)

        # Patch inverse to a simple, valid output
        def fake_inverse(self, watermark_r, is_video=False):
            return torch.zeros_like(self.watermark)
        monkeypatch.setattr(VideoShieldDetector, "_diffusion_inverse", fake_inverse)

        out = det.eval_watermark(lat, detector_type="bit_acc")
        assert isinstance(out, dict)
        assert "bit_acc" in out
        assert "is_watermarked" in out

    def test_videoshield_eval_image_case_with_keys_from_numpy_reshape(self, monkeypatch):
        import numpy as np
        import torch
        from markdiffusion.detection.videoshield.videoshield_detection import VideoShieldDetector

        device = torch.device("cpu")
        # Image-case watermark and latents: 4D [B, C, H, W]
        watermark = torch.randint(0, 2, (1, 1, 2, 2), dtype=torch.uint8, device=device)
        lat = torch.randn(1, 1, 2, 2, device=device)

        key = b"k" * 32
        nonce = b"n" * 8

        det = VideoShieldDetector(
            watermark=watermark,
            threshold=0.0,
            device=device,
            chacha_key=key,
            chacha_nonce=nonce,
            k_f=2, k_c=1, k_h=2, k_w=2
        )

        # Force _stream_key_decrypt to return a numpy array (so torch.from_numpy runs in image-case branch)
        def fake_stream_decrypt(self, reversed_m_np):
            # Return pack/unpack of the same length as flattened bits (simulate decrypted numpy bits)
            return np.array(reversed_m_np, dtype=np.uint8)
        monkeypatch.setattr(VideoShieldDetector, "_stream_key_decrypt", fake_stream_decrypt)

        # Monkeypatch inverse to avoid video-only branch and return zeros compatible with watermark
        def fake_inverse(self, watermark_r, is_video=False):
            return torch.zeros_like(self.watermark)
        monkeypatch.setattr(VideoShieldDetector, "_diffusion_inverse", fake_inverse)

        out = det.eval_watermark(lat, detector_type="bit_acc")
        assert isinstance(out, dict)
        assert "bit_acc" in out
        assert "is_watermarked" in out
    
    def _make_vs_data(self):
        import torch
        from types import SimpleNamespace
        # k_* chosen so strides: ch=4, frame=2, h=2, w=2
        data = SimpleNamespace()
        data.device = torch.device("cpu")
        data.k_c, data.k_f, data.k_h, data.k_w = 1, 2, 2, 2
        data.num_frames = 4
        data.latents_height = 4
        data.latents_width = 4
        # watermark length: 1*4*2*2*2 = 32
        data.watermark = torch.randint(0, 2, (32,), dtype=torch.uint8)
        # one reversed_latent with shape [B=1, C=4, F=2, H=2, W=2]
        data.reversed_latents = [torch.randn(1, 4, 2, 2, 2)]
        # keys (not used when monkeypatched)
        data.chacha_key = b"k" * 32
        data.chacha_nonce = b"n" * 8
        # video frames for deprecated method
        from PIL import Image
        data.video_frames = [Image.new("RGB", (32, 32), (i*10, 100, 200)) for i in range(4)]
        return data

    def test_vs_draw_watermark_bits_channel_oob_raises(self):
        import matplotlib.pyplot as plt
        from markdiffusion.visualize.videoshield.video_shield_visualizer import VideoShieldVisualizer
        data = self._make_vs_data()
        vis = VideoShieldVisualizer(data_for_visualization=data, is_video=True)
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        try:
            # ch_stride = 4 -> valid channels 0..3; pass 4 to raise
            import pytest
            with pytest.raises(ValueError, match="Channel .* out of range"):
                vis.draw_watermark_bits(channel=4, frame=0, ax=ax)
        finally:
            plt.close(fig)

    def test_vs_draw_watermark_bits_frame_oob_raises(self):
        import matplotlib.pyplot as plt
        from markdiffusion.visualize.videoshield.video_shield_visualizer import VideoShieldVisualizer
        data = self._make_vs_data()
        vis = VideoShieldVisualizer(data_for_visualization=data, is_video=True)
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        try:
            # frame_stride = 2 -> valid frames 0..1; pass 2 to raise
            import pytest
            with pytest.raises(ValueError, match="Frame .* out of range"):
                vis.draw_watermark_bits(channel=0, frame=2, ax=ax)
        finally:
            plt.close(fig)

    def test_vs_draw_watermark_bits_channel_mid_frame_title_blank(self):
        import matplotlib.pyplot as plt
        from markdiffusion.visualize.videoshield.video_shield_visualizer import VideoShieldVisualizer
        data = self._make_vs_data()
        vis = VideoShieldVisualizer(data_for_visualization=data, is_video=True)
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        try:
            # channel path, frame None -> uses middle frame branch
            out_ax = vis.draw_watermark_bits(channel=0, frame=None, title="", ax=ax)
            assert out_ax is ax
        finally:
            plt.close(fig)

    def test_vs_draw_watermark_bits_multi_channel_mid_frame(self):
        import matplotlib.pyplot as plt
        from markdiffusion.visualize.videoshield.video_shield_visualizer import VideoShieldVisualizer
        data = self._make_vs_data()
        vis = VideoShieldVisualizer(data_for_visualization=data, is_video=True)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        try:
            # channel None -> multi-channel branch, frame None -> middle frame title path
            out_ax = vis.draw_watermark_bits(channel=None, frame=None, title="Original", ax=ax)
            assert out_ax is ax
        finally:
            plt.close(fig)

    def test_vs_draw_reconstructed_bits_channel_oob_raises(self, monkeypatch):
        import matplotlib.pyplot as plt
        from markdiffusion.visualize.videoshield.video_shield_visualizer import VideoShieldVisualizer
        data = self._make_vs_data()
        vis = VideoShieldVisualizer(data_for_visualization=data, is_video=True)
        # make decrypt a pass-through to hit torch.from_numpy reshape
        monkeypatch.setattr(VideoShieldVisualizer, "_stream_key_decrypt", lambda self, x: x)
        # avoid internal split complexity
        monkeypatch.setattr(VideoShieldVisualizer, "_diffusion_inverse", lambda self, rsd: rsd.flatten())
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        try:
            import pytest
            with pytest.raises(ValueError, match="Channel .* out of range"):
                vis.draw_reconstructed_watermark_bits(channel=4, frame=0, ax=ax)
        finally:
            plt.close(fig)

    def test_vs_draw_reconstructed_bits_channel_title_empty(self, monkeypatch):
        import matplotlib.pyplot as plt
        import torch
        from markdiffusion.visualize.videoshield.video_shield_visualizer import VideoShieldVisualizer
        data = self._make_vs_data()
        vis = VideoShieldVisualizer(data_for_visualization=data, is_video=True)
        # decrypt pass-through
        monkeypatch.setattr(VideoShieldVisualizer, "_stream_key_decrypt", lambda self, x: x)
        # return a flat tensor that reshapes to (1, ch_stride, frame_stride, h_stride, w_stride)
        ch_stride, frame_stride, h_stride, w_stride = 4, 2, 2, 2
        total = 1 * ch_stride * frame_stride * h_stride * w_stride
        monkeypatch.setattr(VideoShieldVisualizer, "_diffusion_inverse", lambda self, rsd: torch.zeros(total, dtype=torch.uint8))
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        try:
            # title empty -> else branch for title formatting
            out_ax = vis.draw_reconstructed_watermark_bits(channel=0, frame=None, title="", ax=ax)
            assert out_ax is ax
        finally:
            plt.close(fig)

    def test_vs_draw_reconstructed_bits_multi_channel_mid_frame_title_paths(self, monkeypatch):
        import matplotlib.pyplot as plt
        import torch
        from markdiffusion.visualize.videoshield.video_shield_visualizer import VideoShieldVisualizer
        data = self._make_vs_data()
        vis = VideoShieldVisualizer(data_for_visualization=data, is_video=True)
        monkeypatch.setattr(VideoShieldVisualizer, "_stream_key_decrypt", lambda self, x: x)
        ch_stride, frame_stride, h_stride, w_stride = 4, 2, 2, 2
        total = 1 * ch_stride * frame_stride * h_stride * w_stride
        monkeypatch.setattr(VideoShieldVisualizer, "_diffusion_inverse", lambda self, rsd: torch.ones(total, dtype=torch.uint8))
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        try:
            # channel None -> multi-channel branch; title non-empty
            out_ax = vis.draw_reconstructed_watermark_bits(channel=None, frame=None, title="Reconstructed", ax=ax)
            assert out_ax is ax
            # channel None; title empty branch too
            out_ax2 = vis.draw_reconstructed_watermark_bits(channel=None, frame=None, title="", ax=ax)
            assert out_ax2 is ax
        finally:
            plt.close(fig)

    def test_vs_draw_watermarked_video_frames_deprecated_dispatch(self):
        import matplotlib.pyplot as plt
        from markdiffusion.visualize.videoshield.video_shield_visualizer import VideoShieldVisualizer
        data = self._make_vs_data()
        vis = VideoShieldVisualizer(data_for_visualization=data, is_video=True)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        try:
            # deprecated method delegates to _draw_video_frames
            out_ax = vis.draw_watermarked_video_frames(num_frames=2, title="Frames", ax=ax)
            assert out_ax is ax
        finally:
            plt.close(fig)


    # videomark
    def test_videomark_align_posteriors_length_equal_truncate_pad(self, monkeypatch):
        import numpy as np
        import torch
        from markdiffusion.detection.videomark.videomark_detection import VideoMarkDetector

        # Dummy decoding_key components
        G = np.eye(5, dtype=int)               # generator_matrix (code_length=5)
        # Build a simple CSR-like object with required attributes
        class H:
            def __init__(self):
                self.shape = (2, 5)
                self.indices = np.array([0, 1], dtype=int)
        parity = H()
        otp = np.zeros(5, dtype=int)
        false_positive_rate = 1e-6
        noise_rate = 0.0
        test_bits = [0, 1]
        g = 0
        max_bp_iter = 5
        t = 1
        decoding_key = (G, parity, otp, false_positive_rate, noise_rate, test_bits, g, max_bp_iter, t)

        # Dummy GF (galois FieldArray) type: callable to pass-through numpy
        class GF:
            def __call__(self, x):
                return np.array(x, dtype=int)

        device = torch.device("cpu")
        msg_seq = np.array([[0, 1, 0], [1, 0, 1]], dtype=int)
        watermark = np.zeros((3, 3), dtype=int)

        det = VideoMarkDetector(
            message_sequence=msg_seq,
            watermark=watermark,
            num_frames=3,
            var=1.0,
            decoding_key=decoding_key,
            GF=GF,
            threshold=0.5,
            device=device
        )

        # current_len == code_length returns unchanged
        p_equal = torch.randn(5, dtype=torch.float32)
        out_equal = det._align_posteriors_length(p_equal)
        assert out_equal.numel() == 5

        # current_len > code_length truncates
        p_long = torch.randn(7, dtype=torch.float32)
        out_long = det._align_posteriors_length(p_long)
        assert out_long.numel() == 5

        # current_len < code_length pads zeros
        p_short = torch.randn(3, dtype=torch.float32)
        out_short = det._align_posteriors_length(p_short)
        assert out_short.numel() == 5
        assert torch.all(out_short[3:] == 0)

    def test_videomark_boolean_row_reduce_none(self):
        import numpy as np
        from markdiffusion.detection.videomark.videomark_detection import VideoMarkDetector

        # Minimal detector just to access method
        dummy = object()
        det = VideoMarkDetector(
            message_sequence=np.zeros((1, 1), dtype=int),
            watermark=np.zeros((1, 1), dtype=int),
            num_frames=1,
            var=1,
            decoding_key=(np.eye(1, dtype=int), dummy, [], 1e-6, 0.0, [], 0, 1, 1),
            GF=type("GF", (), {"__call__": lambda self, x: np.array(x)})(),
            threshold=0.5,
            device="cpu"
        )

        # Construct non-invertible matrix (first column all zeros)
        A = np.array([[0, 1],
                    [0, 1]], dtype=int)
        perm = det._boolean_row_reduce(A, print_progress=False)
        assert perm is None

    def test_videomark_eval_unsupported_detector_type_raises(self):
        import numpy as np
        import torch
        import pytest
        from markdiffusion.detection.videomark.videomark_detection import VideoMarkDetector

        # Minimal decoding_key
        G = np.eye(3, dtype=int)
        class H:
            def __init__(self):
                self.shape = (1, 3)
                self.indices = np.array([0], dtype=int)
        decoding_key = (G, H(), np.zeros(3, dtype=int), 1e-6, 0.0, [0], 0, 1, 1)

        class GF:
            def __call__(self, x): return np.array(x)

        det = VideoMarkDetector(
            message_sequence=np.zeros((1, 3), dtype=int),
            watermark=np.zeros((2, 3), dtype=int),
            num_frames=2,
            var=1.0,
            decoding_key=decoding_key,
            GF=GF,
            threshold=0.5,
            device="cpu"
        )
        lat = torch.randn(1, 4, 2, 3, dtype=torch.float32)
        with pytest.raises(ValueError, match="not supported for VideoMark"):
            det.eval_watermark(lat, detector_type="other")

    def test_videomark_eval_latents_dim_too_small_raises(self):
        import numpy as np
        import torch
        import pytest
        from markdiffusion.detection.videomark.videomark_detection import VideoMarkDetector

        G = np.eye(3, dtype=int)
        class H:
            def __init__(self):
                self.shape = (1, 3)
                self.indices = np.array([0], dtype=int)
        decoding_key = (G, H(), np.zeros(3, dtype=int), 1e-6, 0.0, [0], 0, 1, 1)
        class GF:
            def __call__(self, x): return np.array(x)

        det = VideoMarkDetector(
            message_sequence=np.zeros((1, 3), dtype=int),
            watermark=np.zeros((2, 3), dtype=int),
            num_frames=2,
            var=1.0,
            decoding_key=decoding_key,
            GF=GF,
            threshold=0.5,
            device="cpu"
        )
        lat = torch.randn(2, dtype=torch.float32)  # dim < 3
        with pytest.raises(ValueError, match="expects at least 3D"):
            det.eval_watermark(lat)

    def test_videomark_eval_frame_mismatch_and_bit_acc(self, monkeypatch):
        import numpy as np
        import torch
        from markdiffusion.detection.videomark import videomark_detection as vm_det
        from markdiffusion.detection.videomark.videomark_detection import VideoMarkDetector

        monkeypatch.setattr(vm_det, "erf", lambda x: torch.erf(x))

        class DummyBPD:
            def __init__(self, H, channel_probs, max_iter, bp_method):
                self.log_prob_ratios = np.zeros(len(channel_probs), dtype=float)
            def decode(self, x):
                return np.array(x, dtype=int)
        monkeypatch.setattr(vm_det, "bp_decoder", lambda H, channel_probs, max_iter, bp_method: DummyBPD(H, channel_probs, max_iter, bp_method))

        G = np.eye(4, dtype=int)
        class H:
            def __init__(self):
                self.shape = (2, 4)
                self.indices = np.array([0, 1], dtype=int)
        parity = H()
        one_time_pad = np.zeros(4, dtype=int)
        false_positive_rate = 1e-6
        noise_rate = 0.0
        test_bits = [0]
        g = 0
        max_bp_iter = 3
        t = 1

        # Pass an INSTANCE with __call__, not the class
        class GFLike:
            def __call__(self, x):
                return np.array(x, dtype=int)
        GF_instance = GFLike()

        decoding_key = (G, parity, one_time_pad, false_positive_rate, noise_rate, test_bits, g, max_bp_iter, t)
        device = torch.device("cpu")
        msg_seq = np.array([[0, 1, 0], [1, 0, 1]], dtype=int)
        watermark = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 1]], dtype=int)

        det = VideoMarkDetector(
            message_sequence=msg_seq,
            watermark=watermark,
            num_frames=2,
            var=1.0,
            decoding_key=decoding_key,
            GF=GF_instance,
            threshold=0.0,
            device=device
        )

        lat = torch.randn(1, 4, 3, 2, 2, dtype=torch.float32, device=device)
        out = det.eval_watermark(lat, detector_type="bit_acc")
        assert isinstance(out, dict)
        assert "bit_acc" in out
        assert "is_watermarked" in out
        assert out["recovered_index"].shape[0] == min(2, out["recovered_message"].shape[0])

    def test_videomark_eval_not_enough_frames_returns_zero(self, monkeypatch):
        import numpy as np
        import torch
        from markdiffusion.detection.videomark import videomark_detection as vm_det
        from markdiffusion.detection.videomark.videomark_detection import VideoMarkDetector

        # Patch erf for torch
        monkeypatch.setattr(vm_det, "erf", lambda x: torch.erf(x))

        G = np.eye(3, dtype=int)
        class H:
            def __init__(self):
                self.shape = (1, 3)
                self.indices = np.array([0], dtype=int)
        decoding_key = (G, H(), np.zeros(3, dtype=int), 1e-6, 0.0, [0], 0, 1, 1)
        class GF:
            def __call__(self, x): return np.array(x)

        det = VideoMarkDetector(
            message_sequence=np.zeros((1, 3), dtype=int),
            watermark=np.zeros((2, 3), dtype=int),
            num_frames=0,  # expect default
            var=1.0,
            decoding_key=decoding_key,
            GF=GF,
            threshold=0.5,
            device="cpu"
        )
        # frames_to_use == 1 triggers error path
        lat = torch.randn(1, 4, 1, 2, 2, dtype=torch.float32)
        out = det.eval_watermark(lat)
        assert isinstance(out, dict)
        assert out["is_watermarked"] is False
        assert out["bit_acc"] == 0.0
        assert out["recovered_index"].size == 0
        assert out["recovered_message"].size == 0
        assert out["recovered_distance"].size == 0
    


    # sfw
    def test_sfw_draw_pattern_fft_hsqr_complex_4d(self):
        import numpy as np
        import torch
        import matplotlib.pyplot as plt
        from types import SimpleNamespace
        from markdiffusion.visualize.sfw.sfw_visualizer import SFWVisualizer

        # Data setup: HSQR branch, gt_patch as complex 4D -> selects [0], uses abs
        B, C, H, W = 1, 2, 16, 16
        data = SimpleNamespace()
        data.orig_watermarked_latents = torch.randn(B, C, H, W)
        data.w_channel = 1
        data.wm_type = "HSQR"
        # gt_patch: [B, C, ph, pw] complex
        ph, pw = 6, 6
        gt_complex = (torch.randn(B, C, ph, pw) + 1j * torch.randn(B, C, ph, pw)).to(torch.complex64)
        data.gt_patch = gt_complex

        vis = SFWVisualizer(data_for_visualization=data, dpi=100)
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        try:
            out_ax = vis.draw_pattern_fft(title="FFT HSQR", cmap="viridis", use_color_bar=True, ax=ax)
            assert out_ax is ax
        finally:
            plt.close(fig)

    def test_sfw_draw_pattern_fft_non_hsqr_mask(self):
        import torch
        import matplotlib.pyplot as plt
        from types import SimpleNamespace
        from markdiffusion.visualize.sfw.sfw_visualizer import SFWVisualizer

        # Data setup: non-HSQR branch uses watermarking_mask
        B, C, H, W = 1, 2, 16, 16
        data = SimpleNamespace()
        data.orig_watermarked_latents = torch.randn(B, C, H, W)
        data.w_channel = 0
        data.wm_type = "HSTR"  # anything not "HSQR"
        # boolean mask with a small region True
        mask = torch.zeros(B, C, H, W, dtype=torch.bool)
        mask[:, :, 4:8, 4:8] = True
        data.watermarking_mask = mask

        vis = SFWVisualizer(data_for_visualization=data, dpi=100)
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        try:
            out_ax = vis.draw_pattern_fft(title="", cmap="magma", use_color_bar=False, ax=ax)
            assert out_ax is ax
        finally:
            plt.close(fig)

    def test_sfw_draw_inverted_pattern_fft_hsqr_step_none_3d(self):
        import numpy as np
        import torch
        import matplotlib.pyplot as plt
        from types import SimpleNamespace
        from markdiffusion.visualize.sfw.sfw_visualizer import SFWVisualizer

        # HSQR, step None path; gt_patch as 3D array -> selects first channel
        B, C, F, H, W = 1, 2, 3, 16, 16
        data = SimpleNamespace()
        # reversed_latents: list of [B, C, H, W], select [0, w_channel]
        data.reversed_latents = [torch.randn(B, C, H, W) for _ in range(F)]
        data.w_channel = 0
        data.wm_type = "HSQR"
        # 3D (C, ph, pw) real pattern
        ph, pw = 5, 5
        gt3d = np.random.randn(C, ph, pw).astype("float32")
        data.gt_patch = gt3d

        vis = SFWVisualizer(data_for_visualization=data, dpi=100)
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        try:
            out_ax = vis.draw_inverted_pattern_fft(step=None, title="Inverted HSQR", use_color_bar=True, ax=ax)
            assert out_ax is ax
        finally:
            plt.close(fig)

    def test_sfw_draw_inverted_pattern_fft_non_hsqr_step_specified(self):
        import torch
        import matplotlib.pyplot as plt
        from types import SimpleNamespace
        from markdiffusion.visualize.sfw.sfw_visualizer import SFWVisualizer

        # Non-HSQR branch, step specified, mask path
        B, C, F, H, W = 1, 2, 3, 16, 16
        data = SimpleNamespace()
        data.reversed_latents = [torch.randn(B, C, H, W) for _ in range(F)]
        data.w_channel = 1
        data.wm_type = "HSTR"  # not HSQR
        mask = torch.zeros(B, C, H, W, dtype=torch.bool)
        mask[:, :, 2:10, 3:11] = True
        data.watermarking_mask = mask

        vis = SFWVisualizer(data_for_visualization=data, dpi=100)
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        try:
            out_ax = vis.draw_inverted_pattern_fft(step=1, title="", use_color_bar=False, ax=ax)
            assert out_ax is ax
        finally:
            plt.close(fig)
    
    def test_config_get_message_returns_valid_index(self):
        from markdiffusion.watermark.videomark.video_mark import VideoMarkConfig
        start = VideoMarkConfig._get_message(10, 3)
        assert 0 <= start < 7
    
    def test_utils_sample_prc_codeword_with_basis(self):
        import torch
        from types import SimpleNamespace
        from markdiffusion.watermark.videomark.video_mark import VideoMarkUtils

        # Build a utils instance without running __init__
        utils = object.__new__(VideoMarkUtils)
        utils.config = SimpleNamespace(pseudogaussian_seed=123)

        # codeword: shape (m, n), basis: shape (k, n) so basis.T is (n, k)
        m, n, k = 2, 3, 4
        codeword = torch.ones((m, n), dtype=torch.float32)
        basis = torch.ones((k, n), dtype=torch.float32)  # note: basis.T will be (n, k)

        out = utils._sample_prc_codeword(codeword, basis=basis)
        assert out.shape == (m, k)

    def test_generate_watermarked_video_raises_for_non_video_pipeline(self):
        # Covers line 194: raise ValueError when non-video pipeline detected
        import torch
        from types import SimpleNamespace
        import importlib
        from unittest.mock import patch

        vm_module = importlib.import_module('markdiffusion.watermark.videomark.video_mark')
        VideoMarkWatermark = vm_module.VideoMarkWatermark

        # Build instance bypassing heavy __init__
        vm = object.__new__(VideoMarkWatermark)
        # Minimal config with a dummy pipe
        class DummyPipe:
            def __init__(self):
                self.unet = SimpleNamespace(dtype=torch.float32)

        vm.config = SimpleNamespace(pipe=DummyPipe())

        with patch('markdiffusion.watermark.videomark.video_mark.is_video_pipeline', return_value=False):
            try:
                vm._generate_watermarked_video(prompt="test")
                assert False, "Expected ValueError for non-video pipeline"
            except ValueError as e:
                # Ensure message references the pipe class name
                assert vm.config.pipe.__class__.__name__ in str(e)

    def test_generate_watermarked_video_merges_gen_kwargs_and_handles_videos_and_pil_np(self, monkeypatch):
        # 277 (PIL frame), and 287-288 (delete num_frames in finally when original was None)
        import torch
        import numpy as np
        from PIL import Image
        from types import SimpleNamespace
        import importlib
        from unittest.mock import patch

        vm_module = importlib.import_module('markdiffusion.watermark.videomark.video_mark')
        VideoMarkWatermark = vm_module.VideoMarkWatermark

        # Dummy pipe that records kwargs and returns an object with 'videos'
        class DummyPipe:
            def __init__(self):
                self.unet = SimpleNamespace(dtype=torch.float32)
                self.scheduler = SimpleNamespace(config={})
                self.vae = SimpleNamespace(dtype=torch.float32)
                self.last_kwargs = None

            def __call__(self, prompt, **kwargs):
                self.last_kwargs = kwargs
                # Build videos output containing one uint8 ndarray and one PIL Image
                h, w = 8, 8
                uint8_frame = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
                pil_frame = Image.new('RGB', (h, w), color=(255, 0, 0))
                return SimpleNamespace(videos=[ [uint8_frame, pil_frame] ])

        # Build instance bypassing __init__
        vm = object.__new__(VideoMarkWatermark)

        # Minimal config
        cfg = SimpleNamespace(
            gen_seed=42,
            num_frames=None,  # to trigger deletion in finally
            guidance_scale=1.0,
            num_inference_steps=4,
            image_size=(16, 16),
            pipe=DummyPipe(),
            gen_kwargs={"extra_param": 123},  # should merge
        )
        vm.config = cfg

        # Stub utils.inject_watermark to return a tensor shaped (b,c,f,h,w)
        class DummyUtils:
            def inject_watermark(self):
                # return shape (b=1, c=4, f=2, h=8, w=8)
                return torch.randn(1, 4, 2, 8, 8, dtype=torch.float32)

        vm.utils = DummyUtils()

        # No-op for set_orig_watermarked_latents
        monkeypatch.setattr(vm, 'set_orig_watermarked_latents', lambda x: None)

        with patch('markdiffusion.watermark.videomark.video_mark.is_video_pipeline', return_value=True):
            with patch('markdiffusion.watermark.videomark.video_mark.is_i2v_pipeline', return_value=False):
                frames = vm._generate_watermarked_video(prompt="hello")
                # Ensure frames list contains PIL Images created from uint8 ndarray and original PIL
                assert isinstance(frames, list)
                assert all(isinstance(f, Image.Image) for f in frames)
                # Ensure gen_kwargs merged into pipe call
                assert 'extra_param' in vm.config.pipe.last_kwargs
                # Ensure num_frames deleted in finally (since original was None but attribute existed)
                assert not hasattr(vm.config, 'num_frames')

    def test_generate_watermarked_video_output_tuple_and_np_float(self, monkeypatch):
        import torch
        import numpy as np
        from PIL import Image
        from types import SimpleNamespace
        import importlib
        from unittest.mock import patch

        vm_module = importlib.import_module('markdiffusion.watermark.videomark.video_mark')
        VideoMarkWatermark = vm_module.VideoMarkWatermark

        class DummyPipe:
            def __init__(self):
                self.unet = SimpleNamespace(dtype=torch.float32)
                self.vae = SimpleNamespace(dtype=torch.float32)

            def __call__(self, prompt, **kwargs):
                # Tuple output: first element is list of frames (np float in [0,1])
                h, w = 8, 8
                np_float_frame = np.random.rand(h, w, 3).astype(np.float32)
                return ([np_float_frame],)

        vm = object.__new__(VideoMarkWatermark)
        vm.config = SimpleNamespace(
            gen_seed=7,
            num_frames=2,
            guidance_scale=1.0,
            num_inference_steps=4,
            image_size=(16, 16),
            pipe=DummyPipe(),
        )

        class DummyUtils:
            def inject_watermark(self):
                return torch.randn(1, 4, 2, 8, 8, dtype=torch.float32)

        vm.utils = DummyUtils()
        monkeypatch.setattr(vm, 'set_orig_watermarked_latents', lambda x: None)

        with patch('markdiffusion.watermark.videomark.video_mark.is_video_pipeline', return_value=True):
            frames = vm._generate_watermarked_video(prompt="tuple-output")
            assert isinstance(frames, list)
            assert isinstance(frames[0], Image.Image)

    def test_generate_watermarked_video_tensor_conversion_and_bad_shape_error(self, monkeypatch):
        import torch
        from types import SimpleNamespace
        import importlib
        from unittest.mock import patch

        vm_module = importlib.import_module('markdiffusion.watermark.videomark.video_mark')
        VideoMarkWatermark = vm_module.VideoMarkWatermark

        class DummyPipe:
            def __init__(self):
                self.unet = SimpleNamespace(dtype=torch.float32)
                self.vae = SimpleNamespace(dtype=torch.float32)

            def __call__(self, prompt, **kwargs):
                h, w = 8, 8
                good_tensor = torch.rand(h, w, 3)  # 3 channels in last dim, values in [0,1]
                bad_tensor = torch.rand(h, w)      # bad shape to trigger ValueError
                return SimpleNamespace(frames=[ [good_tensor, bad_tensor] ])

        vm = object.__new__(VideoMarkWatermark)
        vm.config = SimpleNamespace(
            gen_seed=7,
            num_frames=2,
            guidance_scale=1.0,
            num_inference_steps=4,
            image_size=(16, 16),
            pipe=DummyPipe(),
        )

        class DummyUtils:
            def inject_watermark(self):
                return torch.randn(1, 4, 2, 8, 8, dtype=torch.float32)

        vm.utils = DummyUtils()
        monkeypatch.setattr(vm, 'set_orig_watermarked_latents', lambda x: None)

        with patch('markdiffusion.watermark.videomark.video_mark.is_video_pipeline', return_value=True):
            try:
                vm._generate_watermarked_video(prompt="tensor-conversion")
                assert False, "Expected ValueError due to unexpected tensor shape"
            except ValueError as e:
                assert "Unexpected tensor shape" in str(e)

    def test_generate_watermarked_video_unknown_frame_type_error(self, monkeypatch):
        import torch
        from types import SimpleNamespace
        import importlib
        from unittest.mock import patch

        vm_module = importlib.import_module('markdiffusion.watermark.videomark.video_mark')
        VideoMarkWatermark = vm_module.VideoMarkWatermark

        class DummyPipe:
            def __init__(self):
                self.unet = SimpleNamespace(dtype=torch.float32)
                self.vae = SimpleNamespace(dtype=torch.float32)

            def __call__(self, prompt, **kwargs):
                # Frame of unexpected type (dict)
                return SimpleNamespace(frames=[ [{'unexpected': 'type'}] ])

        vm = object.__new__(VideoMarkWatermark)
        vm.config = SimpleNamespace(
            gen_seed=7,
            num_frames=1,
            guidance_scale=1.0,
            num_inference_steps=4,
            image_size=(16, 16),
            pipe=DummyPipe(),
        )

        class DummyUtils:
            def inject_watermark(self):
                return torch.randn(1, 4, 1, 8, 8, dtype=torch.float32)

        vm.utils = DummyUtils()
        monkeypatch.setattr(vm, 'set_orig_watermarked_latents', lambda x: None)

        with patch('markdiffusion.watermark.videomark.video_mark.is_video_pipeline', return_value=True):
            try:
                vm._generate_watermarked_video(prompt="bad-type")
                assert False, "Expected TypeError due to unexpected frame type"
            except TypeError as e:
                assert "Unexpected type for frame" in str(e)

    def test_generate_watermarked_video_i2v_latents_permutation(self, monkeypatch):
        from types import SimpleNamespace
        import importlib
        from unittest.mock import patch

        vm_module = importlib.import_module('markdiffusion.watermark.videomark.video_mark')
        VideoMarkWatermark = vm_module.VideoMarkWatermark

        class DummyPipe:
            def __init__(self):
                self.unet = SimpleNamespace(dtype=torch.float32)
                self.vae = SimpleNamespace(dtype=torch.float32)
                self.last_kwargs = None

            def __call__(self, prompt, **kwargs):
                self.last_kwargs = kwargs
                # Return object with frames (use a minimal PIL image-like via torch -> to be converted)
                return SimpleNamespace(frames=[ [torch.rand(8, 8, 3)] ])

        vm = object.__new__(VideoMarkWatermark)
        vm.config = SimpleNamespace(
            gen_seed=7,
            num_frames=2,
            guidance_scale=1.0,
            num_inference_steps=4,
            image_size=(16, 16),
            pipe=DummyPipe(),
        )

        class DummyUtils:
            def inject_watermark(self):
                # initial shape (b=1, c=3, f=2, h=8, w=8)
                return torch.randn(1, 3, 2, 8, 8, dtype=torch.float32)

        vm.utils = DummyUtils()
        monkeypatch.setattr(vm, 'set_orig_watermarked_latents', lambda x: None)

        with patch('markdiffusion.watermark.videomark.video_mark.is_video_pipeline', return_value=True):
            with patch('markdiffusion.watermark.videomark.video_mark.is_i2v_pipeline', return_value=True):
                vm._generate_watermarked_video(prompt="permute")
                # Latents passed to pipe should be (b,f,c,h,w) after permutation
                latents = vm.config.pipe.last_kwargs['latents']
                assert latents.shape == (1, 2, 3, 8, 8)

    def test_get_video_latents_sampling_branch(self):
        import torch
        import importlib

        vm_module = importlib.import_module('markdiffusion.watermark.videomark.video_mark')
        VideoMarkWatermark = vm_module.VideoMarkWatermark

        vm = object.__new__(VideoMarkWatermark)

        class DummyLatentDist:
            def sample(self, generator=None):
                # Shape [F, C, H, W]
                return torch.ones(2, 3, 2, 2, dtype=torch.float32)

            def mode(self):
                return torch.zeros(2, 3, 2, 2, dtype=torch.float32)

        class DummyVAE:
            def encode(self, video_frames):
                from types import SimpleNamespace
                return SimpleNamespace(latent_dist=DummyLatentDist())

        # video_frames is not used by DummyVAE; provide any tensor
        video_frames = torch.randn(2, 3, 2, 2, dtype=torch.float32)
        latents = vm._get_video_latents(DummyVAE(), video_frames, sample=True, rng_generator=None, permute=True)
        # After unsqueeze(0): [B=1, F=2, C=3, H=2, W=2] and permute to [1, 3, 2, 2, 2]
        assert latents.shape == (1, 3, 2, 2, 2)


    def test_get_data_for_visualize_detector_exception(self, monkeypatch):
        # Avoid warnings by monkeypatching torch.tensor inside module to handle Tensor inputs safely
        import torch
        import numpy as np
        import sys
        from types import SimpleNamespace, ModuleType
        import importlib
        from unittest.mock import patch

        orig_tensor = torch.tensor

        def safe_tensor(x, dtype=None):
            if isinstance(x, torch.Tensor):
                t = x.detach().clone()
                return t.to(dtype=dtype) if dtype is not None else t
            return orig_tensor(x, dtype=dtype)

        vm_module = importlib.import_module('markdiffusion.watermark.videomark.video_mark')
        VideoMarkWatermark = vm_module.VideoMarkWatermark

        class DFV:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        # Fake diffusers with DDIMInverseScheduler
        fake_diffusers = ModuleType("diffusers")

        class FakeInverseScheduler:
            @classmethod
            def from_config(cls, config):
                return cls()

        fake_diffusers.DDIMInverseScheduler = FakeInverseScheduler
        monkeypatch.setitem(sys.modules, 'diffusers', fake_diffusers)

        # Dummy collector
        class DummyCollector:
            def __init__(self, save_every_n_steps=1, to_cpu=True):
                self.latents_list = []

        # VAE with encode().latent_dist.mode()
        class DummyLatentDist:
            def mode(self):
                # Shape [F, C, H, W]
                return torch.ones(2, 3, 2, 2, dtype=torch.float32)

        class DummyVAE:
            def __init__(self):
                self.dtype = torch.float32

            def encode(self, video_frames):
                return SimpleNamespace(latent_dist=DummyLatentDist())

        # Dummy pipe returning latent frames
        class DummyPipe:
            def __init__(self):
                self.unet = SimpleNamespace(dtype=torch.float32)
                self.scheduler = SimpleNamespace(config={})
                self.vae = DummyVAE()

            def __call__(self, prompt, **kwargs):
                # Return latent frames as torch tensor to bypass image conversion
                return SimpleNamespace(frames=torch.randn(1, 2, 2, 2, 3))

        vm = object.__new__(VideoMarkWatermark)
        vm.config = SimpleNamespace(
            gen_seed=42,
            num_frames=2,
            guidance_scale=1.0,
            num_inference_steps=4,
            image_size=(16, 16),
            pipe=DummyPipe(),
            latents_channel=3,
            latents_height=2,
            latents_width=2,
            device='cpu',
            threshold=0.5,
            message=[[0, 1], [1, 0]],
        )

        class DummyUtils:
            encoding_key = (np.zeros((vm.config.latents_channel * vm.config.latents_height * vm.config.latents_width,
                                    5)), None, None, 0, 0.0)
            decoding_key = (None, None, None, None, None, None, None, None, None)

            def _encode_message(self, encoding_key, message):
                # Return vector of length n = C*H*W
                n = vm.config.latents_channel * vm.config.latents_height * vm.config.latents_width
                return torch.ones(n, dtype=torch.float32)

            def _sample_prc_codeword(self, codeword, basis=None):
                # Return same shape torch tensor
                return torch.ones_like(codeword, dtype=torch.float32)

        vm.utils = DummyUtils()

        # Dummy detector that returns empty result (no recovered_prc)
        class DummyDetector:
            def eval_watermark(self, inverted_latents, detector_type='bit_acc'):
                return {}  # No 'recovered_prc'

        vm.detector = DummyDetector()

        with patch('markdiffusion.watermark.videomark.video_mark.DataForVisualization', DFV):
            with patch('markdiffusion.watermark.videomark.video_mark.DenoisingLatentsCollector', DummyCollector):
                # Patch only inside the module to avoid global recursion
                with patch('markdiffusion.watermark.videomark.video_mark.torch.tensor', new=safe_tensor):
                    result = vm.get_data_for_visualize(video_frames=torch.randn(2, 3, 2, 2), prompt="viz")
                # Ensure recovered_prc is None in returned data
                assert result.kwargs.get('recovered_prc') is None

    def test_get_data_for_visualize_detector_exception(self, monkeypatch):
        # Avoid warnings and recursion by patching only the module's torch.tensor with a safe wrapper
        import numpy as np
        import sys
        from types import SimpleNamespace, ModuleType
        import importlib
        from unittest.mock import patch

        orig_tensor = torch.tensor

        def safe_tensor(x, dtype=None):
            if isinstance(x, torch.Tensor):
                t = x.detach().clone()
                return t.to(dtype=dtype) if dtype is not None else t
            return orig_tensor(x, dtype=dtype)

        vm_module = importlib.import_module('markdiffusion.watermark.videomark.video_mark')
        VideoMarkWatermark = vm_module.VideoMarkWatermark

        class DFV:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        # Fake diffusers with DDIMInverseScheduler
        fake_diffusers = ModuleType("diffusers")

        class FakeInverseScheduler:
            @classmethod
            def from_config(cls, config):
                return cls()

        fake_diffusers.DDIMInverseScheduler = FakeInverseScheduler
        monkeypatch.setitem(sys.modules, 'diffusers', fake_diffusers)

        class DummyCollector:
            def __init__(self, save_every_n_steps=1, to_cpu=True):
                self.latents_list = []

        class DummyLatentDist:
            def mode(self):
                return torch.ones(2, 3, 2, 2, dtype=torch.float32)

        class DummyVAE:
            def __init__(self):
                self.dtype = torch.float32

            def encode(self, video_frames):
                return SimpleNamespace(latent_dist=DummyLatentDist())

        class DummyPipe:
            def __init__(self):
                self.unet = SimpleNamespace(dtype=torch.float32)
                self.scheduler = SimpleNamespace(config={})
                self.vae = DummyVAE()

            def __call__(self, prompt, **kwargs):
                return SimpleNamespace(frames=torch.randn(1, 2, 2, 2, 3))

        vm = object.__new__(VideoMarkWatermark)
        vm.config = SimpleNamespace(
            gen_seed=42,
            num_frames=2,
            guidance_scale=1.0,
            num_inference_steps=4,
            image_size=(16, 16),
            pipe=DummyPipe(),
            latents_channel=3,
            latents_height=2,
            latents_width=2,
            device='cpu',
            threshold=0.5,
            message=[[0, 1], [1, 0]],
        )

        class DummyUtils:
            encoding_key = (np.zeros((vm.config.latents_channel * vm.config.latents_height * vm.config.latents_width,
                                    5)), None, None, 0, 0.0)
            decoding_key = (None, None, None, None, None, None, None, None, None)

            def _encode_message(self, encoding_key, message):
                n = vm.config.latents_channel * vm.config.latents_height * vm.config.latents_width
                return torch.ones(n, dtype=torch.float32)

            def _sample_prc_codeword(self, codeword, basis=None):
                return torch.ones_like(codeword, dtype=torch.float32)

        vm.utils = DummyUtils()

        class FailingDetector:
            def eval_watermark(self, inverted_latents, detector_type='bit_acc'):
                raise RuntimeError("Forced detector failure")

        vm.detector = FailingDetector()

        with patch('markdiffusion.watermark.videomark.video_mark.DataForVisualization', DFV):
            with patch('markdiffusion.watermark.videomark.video_mark.DenoisingLatentsCollector', DummyCollector):
                with patch('markdiffusion.watermark.videomark.video_mark.torch.tensor', new=safe_tensor):
                    result = vm.get_data_for_visualize(video_frames=torch.randn(2, 3, 2, 2), prompt="viz")
                # Ensure recovered_prc is None in returned data after exception
                assert result.kwargs.get('recovered_prc') is None

    # wind
    def test_wind_retrieve_group_exception_returns_minus_one(self):
        from markdiffusion.detection.wind.wind_detection import WINDetector
        device = torch.device('cpu')
        # Group pattern has mismatched shape to z_fft, causing broadcasting error inside try
        group_patterns = {0: torch.ones(4, 4, device=device)}
        noise_groups = {}  # not used in _retrieve_group

        det = WINDetector(noise_groups=noise_groups,
                        group_patterns=group_patterns,
                        threshold=0.5,
                        device=device)

        # z_fft shape 8x8 so pattern 4x4 will trigger exception when masked multiply
        z_fft = torch.randn(8, 8, device=device)
        gid = det._retrieve_group(z_fft)
        assert gid is None

    def test_wind_match_noise_invalid_group_returns_default(self):
        # Covers line 88: early return when group_id invalid or not in noise_groups
        import torch
        from markdiffusion.detection.wind.wind_detection import WINDetector

        device = torch.device('cpu')
        det = WINDetector(noise_groups={}, group_patterns={}, threshold=0.5, device=device)

        z = torch.randn(8, 8, device=device)
        res = det._match_noise(z, group_id=-1)
        assert res == {'cosine_similarity': 0.0, 'best_match': None}

    def test_wind_eval_watermark_unsupported_detector_type_raises(self):
        # Covers line 132: raise ValueError when unsupported detector_type
        import torch
        from markdiffusion.detection.wind.wind_detection import WINDetector

        device = torch.device('cpu')
        # Valid shapes to avoid other errors, we only test detector_type handling
        group_patterns = {0: torch.randn(8, 8, device=device)}
        noise_groups = {0: [torch.randn(8, 8, device=device)]}

        det = WINDetector(noise_groups=noise_groups,
                        group_patterns=group_patterns,
                        threshold=0.5,
                        device=device)

        try:
            det.eval_watermark(torch.randn(8, 8, device=device), detector_type="not_supported")
            assert False, "Expected ValueError for unsupported detector_type"
        except ValueError as e:
            msg = str(e)
            assert "only supports" in msg and "not_supported" in msg

    def test_wind_eval_watermark_exception_path_returns_defaults(self, monkeypatch):
        # Covers lines 154-161: exception handling in eval_watermark
        import torch
        from markdiffusion.detection.wind.wind_detection import WINDetector

        device = torch.device('cpu')
        # Mismatched pattern to force error inside _match_noise when multiplying pattern * mask
        group_patterns = {0: torch.randn(3, 3, device=device)}
        # Ensure group_id is recognized by _match_noise (avoid early return)
        noise_groups = {0: []}

        det = WINDetector(noise_groups=noise_groups,
                        group_patterns=group_patterns,
                        threshold=0.5,
                        device=device)

        # Force _retrieve_group to return a valid group id so _match_noise runs and throws
        monkeypatch.setattr(det, '_retrieve_group', lambda z_fft: 0)

        res = det.eval_watermark(torch.randn(8, 8, device=device))
        assert res['group_id'] == -1
        assert res['cosine_similarity'] == 0.0
        assert res['is_watermarked'] is False

    # prc
    def test_prc_recover_posteriors_variances_none(self):
        import torch
        import numpy as np
        from markdiffusion.detection.prc.prc_detection import PRCDetector
        # Minimal detector with dummy decoding_key
        decoding_key = (None, None, None, 0.1, 0.0, np.array([0, 1]), 0, 5, 2)
        det = PRCDetector(var=1, decoding_key=decoding_key, GF=lambda x: x, threshold=0.5, device=torch.device('cpu'))

        z = torch.randn(10, dtype=torch.float32)
        out = det._recover_posteriors(z, variances=None)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (10,)

    def test_prc_recover_posteriors_variances_float(self):
        # Covers line 43: float variance branch; output is a torch.Tensor
        import torch
        import numpy as np
        from markdiffusion.detection.prc.prc_detection import PRCDetector

        decoding_key = (None, None, None, 0.1, 0.0, np.array([0, 1]), 0, 5, 2)
        det = PRCDetector(var=1, decoding_key=decoding_key, GF=lambda x: x, threshold=0.5, device=torch.device('cpu'))

        z = torch.randn(6, dtype=torch.float32)
        out = det._recover_posteriors(z, variances=0.7)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (6,)

    def test_prc_recover_posteriors_with_basis(self):
        # Covers line 50: basis not None branch; result can be 0-dim torch.Tensor
        import torch
        import numpy as np
        from markdiffusion.detection.prc.prc_detection import PRCDetector

        decoding_key = (None, None, None, 0.1, 0.0, np.array([0, 1]), 0, 5, 2)
        det = PRCDetector(var=1, decoding_key=decoding_key, GF=lambda x: x, threshold=0.5, device=torch.device('cpu'))

        z = torch.randn(8, dtype=torch.float32)
        basis = torch.randn(8, dtype=torch.float32)  # z @ basis -> scalar
        out = det._recover_posteriors(z, basis=basis, variances=0.5)
        assert isinstance(out, torch.Tensor)
        assert out.dim() in (0, 1)

    def test_prc_boolean_row_reduce_noninvertible_returns_none(self):
        # Covers lines 84-85: non-invertible branch returns None
        import numpy as np
        from markdiffusion.detection.prc.prc_detection import PRCDetector

        decoding_key = (None, None, None, 0.1, 0.0, np.array([0, 1]), 0, 5, 2)
        det = PRCDetector(var=1, decoding_key=decoding_key, GF=lambda x: x, threshold=0.5, device=None)

        A = np.array([[0, 1],
                    [0, 1]], dtype=int)
        res = det._boolean_row_reduce(A, print_progress=False)
        assert res is None

    def test_prc_boolean_row_reduce_print_progress(self):
        # Covers lines 90-91: print progress branch; patch sys into module to avoid NameError
        import numpy as np
        import sys as py_sys
        from markdiffusion.detection.prc.prc_detection import PRCDetector
        import markdiffusion.detection.prc.prc_detection as prc_mod

        # Inject sys into module namespace
        prc_mod.sys = py_sys

        decoding_key = (None, None, None, 0.1, 0.0, np.array([0, 1]), 0, 5, 2)
        det = PRCDetector(var=1, decoding_key=decoding_key, GF=lambda x: x, threshold=0.5, device=None)

        A = np.array([[1],
                    [0]], dtype=int)
        res = det._boolean_row_reduce(A, print_progress=True)
        assert isinstance(res, np.ndarray)
        assert res.shape == (1,)

    def test_prc_eval_watermark_unsupported_detector_type_raises(self):
        # Covers line 151: ValueError when detector_type not supported
        import torch
        import numpy as np
        from markdiffusion.detection.prc.prc_detection import PRCDetector

        decoding_key = (None, None, None, 0.1, 0.0, np.array([0, 1]), 0, 5, 2)
        det = PRCDetector(var=1, decoding_key=decoding_key, GF=lambda x: x, threshold=0.5, device=torch.device('cpu'))

        try:
            det.eval_watermark(torch.randn(1, 2, 2, 2), detector_type="not_supported")
            assert False, "Expected ValueError for unsupported detector_type"
        except ValueError as e:
            assert "not supported" in str(e)

    def test_prc_decode_message_prints_and_returns_none(self):
        # Covers lines 107 ("Running belief propagation...") and 127 ("Solving linear system...")
        # Plus line 131: return None when test_bits mismatch
        import numpy as np
        import sys as py_sys
        from markdiffusion.detection.prc.prc_detection import PRCDetector
        import markdiffusion.detection.prc.prc_detection as prc_mod
        from unittest.mock import patch

        # Inject sys into module namespace for print_progress path
        prc_mod.sys = py_sys

        k = 4
        generator_matrix = np.eye(k, dtype=int)

        class DummyPCM:
            def __init__(self):
                self.shape = (2, k)
                self.indices = np.arange(4)

        parity_check_matrix = DummyPCM()
        one_time_pad = np.zeros(k, dtype=int)
        false_positive_rate = 0.1
        noise_rate = 0.0
        test_bits = np.array([1, 0], dtype=int)  # force mismatch
        g = 0
        max_bp_iter = 3
        t = 2

        decoding_key = (generator_matrix, parity_check_matrix, one_time_pad,
                        false_positive_rate, noise_rate, test_bits, g, max_bp_iter, t)

        det = PRCDetector(var=1.0, decoding_key=decoding_key, GF=lambda x: x, threshold=0.5, device=None)

        class MockBPD:
            def __init__(self, pcm, channel_probs, max_iter, bp_method):
                self.log_prob_ratios = np.zeros(k)

            def decode(self, x_recovered):
                return x_recovered

        # Create a wrapper with a .numpy(force=True) method to satisfy code expectations
        class PosteriorsWrapper:
            def __init__(self, arr):
                self.arr = np.asarray(arr)

            def numpy(self, force=True):
                return self.arr

        with patch('markdiffusion.detection.prc.prc_detection.bp_decoder', MockBPD):
            with patch('numpy.linalg.solve', return_value=np.ones(k, dtype=int)):
                posteriors = PosteriorsWrapper(np.linspace(-0.5, 0.5, k))
                res = det._decode_message(posteriors, print_progress=True, max_bp_iter=None)
                assert res is None

    def test_prc_binary_array_to_str(self):
        # Covers lines 137-146: conversion from binary array to string
        import numpy as np
        from markdiffusion.detection.prc.prc_detection import PRCDetector

        det = PRCDetector(var=1, decoding_key=(None,)*9, GF=lambda x: x, threshold=0.5, device=None)

        def to_bits(n):
            return np.array(list(map(int, f"{n:08b}")), dtype=int)

        bits = np.concatenate([to_bits(65), to_bits(66)])
        s = det._binary_array_to_str(bits)
        assert s == "AB"
    
    # robin
    def test_robin_get_watermarking_mask_circle_all_channels(self, monkeypatch):
        import sys
        from types import SimpleNamespace, ModuleType
        # Stub external modules required by watermark.robin.watermark_generator
        diffusers = ModuleType("diffusers")
        monkeypatch.setitem(sys.modules, "diffusers", diffusers)

        # Submodule: diffusers.pipelines.stable_diffusion
        sd_mod = ModuleType("diffusers.pipelines.stable_diffusion")
        class StableDiffusionPipelineOutput: pass
        sd_mod.StableDiffusionPipelineOutput = StableDiffusionPipelineOutput
        monkeypatch.setitem(sys.modules, "diffusers.pipelines.stable_diffusion", sd_mod)

        # Submodules: diffusers.models.unets.unet_2d_condition, autoencoders.autoencoder_kl
        unet_mod = ModuleType("diffusers.models.unets.unet_2d_condition")
        class UNet2DConditionModel: pass
        unet_mod.UNet2DConditionModel = UNet2DConditionModel
        monkeypatch.setitem(sys.modules, "diffusers.models.unets.unet_2d_condition", unet_mod)

        ae_mod = ModuleType("diffusers.models.autoencoders.autoencoder_kl")
        class AutoencoderKL: pass
        ae_mod.AutoencoderKL = AutoencoderKL
        monkeypatch.setitem(sys.modules, "diffusers.models.autoencoders.autoencoder_kl", ae_mod)

        # Submodule: diffusers.schedulers
        sched_mod = ModuleType("diffusers.schedulers")
        class DPMSolverMultistepScheduler: pass
        sched_mod.DPMSolverMultistepScheduler = DPMSolverMultistepScheduler
        monkeypatch.setitem(sys.modules, "diffusers.schedulers", sched_mod)

        # Submodule: diffusers.utils
        utils_mod = ModuleType("diffusers.utils")
        class BaseOutput: pass
        utils_mod.BaseOutput = BaseOutput
        monkeypatch.setitem(sys.modules, "diffusers.utils", utils_mod)

        # Root attribute (not strictly needed by imports but safe)
        class StableDiffusionPipeline: pass
        diffusers.StableDiffusionPipeline = StableDiffusionPipeline

        # Stub transformers clip model
        transformers = ModuleType("transformers")
        monkeypatch.setitem(sys.modules, "transformers", transformers)

        clip_mod = ModuleType("transformers.models.clip.modeling_clip")

        class CLIPTextModel: pass
        clip_mod.CLIPTextModel = CLIPTextModel
        monkeypatch.setitem(sys.modules, "transformers.models.clip.modeling_clip", clip_mod)

        # Stub accelerate Accelerator
        accelerate = ModuleType("accelerate")
        class Accelerator: pass
        accelerate.Accelerator = Accelerator
        monkeypatch.setitem(sys.modules, "accelerate", accelerate)

        import torch
        from markdiffusion.watermark.robin.watermark_generator import get_watermarking_mask

        device = 'cpu'
        init_latents_w = torch.zeros((2, 3, 16, 16), dtype=torch.float32, device=device)
        args = SimpleNamespace(w_mask_shape='circle', w_up_radius=5, w_low_radius=2, w_channel=-1)

        mask = get_watermarking_mask(init_latents_w, args, device)
        assert mask.shape == init_latents_w.shape
        assert mask.sum().item() > 0
        per_channel_sums = mask.sum(dim=(0, 2, 3)).cpu().numpy()
        assert (per_channel_sums == per_channel_sums[0]).all()

    def test_robin_get_watermarking_mask_circle_specific_channel(self, monkeypatch):
        import sys
        from types import ModuleType, SimpleNamespace

        diffusers = ModuleType("diffusers")
        monkeypatch.setitem(sys.modules, "diffusers", diffusers)
        sd_mod = ModuleType("diffusers.pipelines.stable_diffusion")
        class StableDiffusionPipelineOutput: pass
        sd_mod.StableDiffusionPipelineOutput = StableDiffusionPipelineOutput
        monkeypatch.setitem(sys.modules, "diffusers.pipelines.stable_diffusion", sd_mod)
        unet_mod = ModuleType("diffusers.models.unets.unet_2d_condition")
        class UNet2DConditionModel: pass
        unet_mod.UNet2DConditionModel = UNet2DConditionModel
        monkeypatch.setitem(sys.modules, "diffusers.models.unets.unet_2d_condition", unet_mod)
        ae_mod = ModuleType("diffusers.models.autoencoders.autoencoder_kl")
        class AutoencoderKL: pass
        ae_mod.AutoencoderKL = AutoencoderKL
        monkeypatch.setitem(sys.modules, "diffusers.models.autoencoders.autoencoder_kl", ae_mod)
        sched_mod = ModuleType("diffusers.schedulers")
        class DPMSolverMultistepScheduler: pass
        sched_mod.DPMSolverMultistepScheduler = DPMSolverMultistepScheduler
        monkeypatch.setitem(sys.modules, "diffusers.schedulers", sched_mod)
        utils_mod = ModuleType("diffusers.utils")
        class BaseOutput: pass
        utils_mod.BaseOutput = BaseOutput
        monkeypatch.setitem(sys.modules, "diffusers.utils", utils_mod)
        class StableDiffusionPipeline: pass
        diffusers.StableDiffusionPipeline = StableDiffusionPipeline

        transformers = ModuleType("transformers")
        monkeypatch.setitem(sys.modules, "transformers", transformers)
        clip_mod = ModuleType("transformers.models.clip.modeling_clip")
        class CLIPTextModel: pass
        clip_mod.CLIPTextModel = CLIPTextModel
        monkeypatch.setitem(sys.modules, "transformers.models.clip.modeling_clip", clip_mod)

        accelerate = ModuleType("accelerate")
        class Accelerator: pass
        accelerate.Accelerator = Accelerator
        monkeypatch.setitem(sys.modules, "accelerate", accelerate)

        import torch
        from markdiffusion.watermark.robin.watermark_generator import get_watermarking_mask

        device = 'cpu'
        init_latents_w = torch.zeros((1, 3, 16, 16), dtype=torch.float32, device=device)
        args = SimpleNamespace(w_mask_shape='circle', w_up_radius=6, w_low_radius=3, w_channel=1)

        mask = get_watermarking_mask(init_latents_w, args, device)
        assert mask[:, 1].sum().item() > 0
        assert mask[:, 0].sum().item() == 0
        assert mask[:, 2].sum().item() == 0

    def test_robin_get_watermarking_mask_square_all_channels(self, monkeypatch):
        import sys
        from types import ModuleType, SimpleNamespace

        diffusers = ModuleType("diffusers")
        monkeypatch.setitem(sys.modules, "diffusers", diffusers)
        sd_mod = ModuleType("diffusers.pipelines.stable_diffusion")
        class StableDiffusionPipelineOutput: pass
        sd_mod.StableDiffusionPipelineOutput = StableDiffusionPipelineOutput
        monkeypatch.setitem(sys.modules, "diffusers.pipelines.stable_diffusion", sd_mod)
        unet_mod = ModuleType("diffusers.models.unets.unet_2d_condition")
        class UNet2DConditionModel: pass
        unet_mod.UNet2DConditionModel = UNet2DConditionModel
        monkeypatch.setitem(sys.modules, "diffusers.models.unets.unet_2d_condition", unet_mod)
        ae_mod = ModuleType("diffusers.models.autoencoders.autoencoder_kl")
        class AutoencoderKL: pass
        ae_mod.AutoencoderKL = AutoencoderKL
        monkeypatch.setitem(sys.modules, "diffusers.models.autoencoders.autoencoder_kl", ae_mod)
        sched_mod = ModuleType("diffusers.schedulers")
        class DPMSolverMultistepScheduler: pass
        sched_mod.DPMSolverMultistepScheduler = DPMSolverMultistepScheduler
        monkeypatch.setitem(sys.modules, "diffusers.schedulers", sched_mod)
        utils_mod = ModuleType("diffusers.utils")
        class BaseOutput: pass
        utils_mod.BaseOutput = BaseOutput
        monkeypatch.setitem(sys.modules, "diffusers.utils", utils_mod)
        class StableDiffusionPipeline: pass
        diffusers.StableDiffusionPipeline = StableDiffusionPipeline

        transformers = ModuleType("transformers")
        monkeypatch.setitem(sys.modules, "transformers", transformers)
        clip_mod = ModuleType("transformers.models.clip.modeling_clip")
        class CLIPTextModel: pass
        clip_mod.CLIPTextModel = CLIPTextModel
        monkeypatch.setitem(sys.modules, "transformers.models.clip.modeling_clip", clip_mod)

        accelerate = ModuleType("accelerate")
        class Accelerator: pass
        accelerate.Accelerator = Accelerator
        monkeypatch.setitem(sys.modules, "accelerate", accelerate)

        import torch
        from markdiffusion.watermark.robin.watermark_generator import get_watermarking_mask

        device = 'cpu'
        init_latents_w = torch.zeros((2, 3, 32, 32), dtype=torch.float32, device=device)
        args = SimpleNamespace(w_mask_shape='square', w_channel=-1, w_radius=4)

        mask = get_watermarking_mask(init_latents_w, args, device)
        assert mask.shape == init_latents_w.shape
        anchor = 32 // 2
        square = mask[:, :, anchor-4:anchor+4, anchor-4:anchor+4]
        assert square.float().mean().item() > 0.9

    def test_robin_get_watermarking_mask_square_specific_channel(self, monkeypatch):
        import sys
        from types import ModuleType, SimpleNamespace

        diffusers = ModuleType("diffusers")
        monkeypatch.setitem(sys.modules, "diffusers", diffusers)
        sd_mod = ModuleType("diffusers.pipelines.stable_diffusion")
        class StableDiffusionPipelineOutput: pass
        sd_mod.StableDiffusionPipelineOutput = StableDiffusionPipelineOutput
        monkeypatch.setitem(sys.modules, "diffusers.pipelines.stable_diffusion", sd_mod)
        unet_mod = ModuleType("diffusers.models.unets.unet_2d_condition")
        class UNet2DConditionModel: pass
        unet_mod.UNet2DConditionModel = UNet2DConditionModel
        monkeypatch.setitem(sys.modules, "diffusers.models.unets.unet_2d_condition", unet_mod)
        ae_mod = ModuleType("diffusers.models.autoencoders.autoencoder_kl")
        class AutoencoderKL: pass
        ae_mod.AutoencoderKL = AutoencoderKL
        monkeypatch.setitem(sys.modules, "diffusers.models.autoencoders.autoencoder_kl", ae_mod)
        sched_mod = ModuleType("diffusers.schedulers")
        class DPMSolverMultistepScheduler: pass
        sched_mod.DPMSolverMultistepScheduler = DPMSolverMultistepScheduler
        monkeypatch.setitem(sys.modules, "diffusers.schedulers", sched_mod)
        utils_mod = ModuleType("diffusers.utils")
        class BaseOutput: pass
        utils_mod.BaseOutput = BaseOutput
        monkeypatch.setitem(sys.modules, "diffusers.utils", utils_mod)
        class StableDiffusionPipeline: pass
        diffusers.StableDiffusionPipeline = StableDiffusionPipeline

        transformers = ModuleType("transformers")
        monkeypatch.setitem(sys.modules, "transformers", transformers)
        clip_mod = ModuleType("transformers.models.clip.modeling_clip")
        class CLIPTextModel: pass
        clip_mod.CLIPTextModel = CLIPTextModel
        monkeypatch.setitem(sys.modules, "transformers.models.clip.modeling_clip", clip_mod)

        accelerate = ModuleType("accelerate")
        class Accelerator: pass
        accelerate.Accelerator = Accelerator
        monkeypatch.setitem(sys.modules, "accelerate", accelerate)

        import torch
        from markdiffusion.watermark.robin.watermark_generator import get_watermarking_mask

        device = 'cpu'
        init_latents_w = torch.zeros((1, 2, 32, 32), dtype=torch.float32, device=device)
        args = SimpleNamespace(w_mask_shape='square', w_channel=0, w_radius=3)

        mask = get_watermarking_mask(init_latents_w, args, device)
        anchor = 32 // 2
        square0 = mask[:, 0, anchor-3:anchor+3, anchor-3:anchor+3]
        square1 = mask[:, 1, anchor-3:anchor+3, anchor-3:anchor+3]
        assert square0.float().mean().item() > 0.9
        assert square1.sum().item() == 0

    def test_robin_get_watermarking_mask_no_shape(self, monkeypatch):
        import sys
        from types import ModuleType, SimpleNamespace

        diffusers = ModuleType("diffusers")
        monkeypatch.setitem(sys.modules, "diffusers", diffusers)
        sd_mod = ModuleType("diffusers.pipelines.stable_diffusion")
        class StableDiffusionPipelineOutput: pass
        sd_mod.StableDiffusionPipelineOutput = StableDiffusionPipelineOutput
        monkeypatch.setitem(sys.modules, "diffusers.pipelines.stable_diffusion", sd_mod)
        unet_mod = ModuleType("diffusers.models.unets.unet_2d_condition")
        class UNet2DConditionModel: pass
        unet_mod.UNet2DConditionModel = UNet2DConditionModel
        monkeypatch.setitem(sys.modules, "diffusers.models.unets.unet_2d_condition", unet_mod)
        ae_mod = ModuleType("diffusers.models.autoencoders.autoencoder_kl")
        class AutoencoderKL: pass
        ae_mod.AutoencoderKL = AutoencoderKL
        monkeypatch.setitem(sys.modules, "diffusers.models.autoencoders.autoencoder_kl", ae_mod)
        sched_mod = ModuleType("diffusers.schedulers")
        class DPMSolverMultistepScheduler: pass
        sched_mod.DPMSolverMultistepScheduler = DPMSolverMultistepScheduler
        monkeypatch.setitem(sys.modules, "diffusers.schedulers", sched_mod)
        utils_mod = ModuleType("diffusers.utils")
        class BaseOutput: pass
        utils_mod.BaseOutput = BaseOutput
        monkeypatch.setitem(sys.modules, "diffusers.utils", utils_mod)
        class StableDiffusionPipeline: pass
        diffusers.StableDiffusionPipeline = StableDiffusionPipeline

        transformers = ModuleType("transformers")
        monkeypatch.setitem(sys.modules, "transformers", transformers)
        clip_mod = ModuleType("transformers.models.clip.modeling_clip")
        class CLIPTextModel: pass
        clip_mod.CLIPTextModel = CLIPTextModel
        monkeypatch.setitem(sys.modules, "transformers.models.clip.modeling_clip", clip_mod)

        accelerate = ModuleType("accelerate")
        class Accelerator: pass
        accelerate.Accelerator = Accelerator
        monkeypatch.setitem(sys.modules, "accelerate", accelerate)

        import torch
        from markdiffusion.watermark.robin.watermark_generator import get_watermarking_mask

        device = 'cpu'
        init_latents_w = torch.zeros((1, 1, 8, 8), dtype=torch.float32, device=device)
        args = SimpleNamespace(w_mask_shape='no')
        mask = get_watermarking_mask(init_latents_w, args, device)
        assert mask.sum().item() == 0

    def test_robin_get_watermarking_mask_unknown_shape_raises(self, monkeypatch):
        import sys
        from types import ModuleType, SimpleNamespace

        diffusers = ModuleType("diffusers")
        monkeypatch.setitem(sys.modules, "diffusers", diffusers)
        sd_mod = ModuleType("diffusers.pipelines.stable_diffusion")
        class StableDiffusionPipelineOutput: pass
        sd_mod.StableDiffusionPipelineOutput = StableDiffusionPipelineOutput
        monkeypatch.setitem(sys.modules, "diffusers.pipelines.stable_diffusion", sd_mod)
        unet_mod = ModuleType("diffusers.models.unets.unet_2d_condition")
        class UNet2DConditionModel: pass
        unet_mod.UNet2DConditionModel = UNet2DConditionModel
        monkeypatch.setitem(sys.modules, "diffusers.models.unets.unet_2d_condition", unet_mod)
        ae_mod = ModuleType("diffusers.models.autoencoders.autoencoder_kl")
        class AutoencoderKL: pass
        ae_mod.AutoencoderKL = AutoencoderKL
        monkeypatch.setitem(sys.modules, "diffusers.models.autoencoders.autoencoder_kl", ae_mod)
        sched_mod = ModuleType("diffusers.schedulers")
        class DPMSolverMultistepScheduler: pass
        sched_mod.DPMSolverMultistepScheduler = DPMSolverMultistepScheduler
        monkeypatch.setitem(sys.modules, "diffusers.schedulers", sched_mod)
        utils_mod = ModuleType("diffusers.utils")
        class BaseOutput: pass
        utils_mod.BaseOutput = BaseOutput
        monkeypatch.setitem(sys.modules, "diffusers.utils", utils_mod)
        class StableDiffusionPipeline: pass
        diffusers.StableDiffusionPipeline = StableDiffusionPipeline

        transformers = ModuleType("transformers")
        monkeypatch.setitem(sys.modules, "transformers", transformers)
        clip_mod = ModuleType("transformers.models.clip.modeling_clip")
        class CLIPTextModel: pass
        clip_mod.CLIPTextModel = CLIPTextModel
        monkeypatch.setitem(sys.modules, "transformers.models.clip.modeling_clip", clip_mod)

        accelerate = ModuleType("accelerate")
        class Accelerator: pass
        accelerate.Accelerator = Accelerator
        monkeypatch.setitem(sys.modules, "accelerate", accelerate)

        import torch
        from markdiffusion.watermark.robin.watermark_generator import get_watermarking_mask

        device = 'cpu'
        init_latents_w = torch.zeros((1, 1, 8, 8), dtype=torch.float32, device=device)
        args = SimpleNamespace(w_mask_shape='triangle')
        try:
            get_watermarking_mask(init_latents_w, args, device)
            assert False, "Expected NotImplementedError for unknown shape"
        except NotImplementedError as e:
            assert 'w_mask_shape' in str(e)

    def test_robin_inject_watermark_unknown_injection_falls_through(self, monkeypatch):
        import sys
        from types import ModuleType, SimpleNamespace

        diffusers = ModuleType("diffusers")
        monkeypatch.setitem(sys.modules, "diffusers", diffusers)
        sd_mod = ModuleType("diffusers.pipelines.stable_diffusion")
        class StableDiffusionPipelineOutput: pass
        sd_mod.StableDiffusionPipelineOutput = StableDiffusionPipelineOutput
        monkeypatch.setitem(sys.modules, "diffusers.pipelines.stable_diffusion", sd_mod)
        unet_mod = ModuleType("diffusers.models.unets.unet_2d_condition")
        class UNet2DConditionModel: pass
        unet_mod.UNet2DConditionModel = UNet2DConditionModel
        monkeypatch.setitem(sys.modules, "diffusers.models.unets.unet_2d_condition", unet_mod)
        ae_mod = ModuleType("diffusers.models.autoencoders.autoencoder_kl")
        class AutoencoderKL: pass
        ae_mod.AutoencoderKL = AutoencoderKL
        monkeypatch.setitem(sys.modules, "diffusers.models.autoencoders.autoencoder_kl", ae_mod)
        sched_mod = ModuleType("diffusers.schedulers")
        class DPMSolverMultistepScheduler: pass
        sched_mod.DPMSolverMultistepScheduler = DPMSolverMultistepScheduler
        monkeypatch.setitem(sys.modules, "diffusers.schedulers", sched_mod)
        utils_mod = ModuleType("diffusers.utils")
        class BaseOutput: pass
        utils_mod.BaseOutput = BaseOutput
        monkeypatch.setitem(sys.modules, "diffusers.utils", utils_mod)
        class StableDiffusionPipeline: pass
        diffusers.StableDiffusionPipeline = StableDiffusionPipeline

        transformers = ModuleType("transformers")
        monkeypatch.setitem(sys.modules, "transformers", transformers)
        clip_mod = ModuleType("transformers.models.clip.modeling_clip")
        class CLIPTextModel: pass
        clip_mod.CLIPTextModel = CLIPTextModel
        monkeypatch.setitem(sys.modules, "transformers.models.clip.modeling_clip", clip_mod)

        accelerate = ModuleType("accelerate")
        class Accelerator: pass
        accelerate.Accelerator = Accelerator
        monkeypatch.setitem(sys.modules, "accelerate", accelerate)

        import torch
        from markdiffusion.watermark.robin.watermark_generator import inject_watermark

        init_latents = torch.randn((1, 4, 8, 8), dtype=torch.float32)
        mask = torch.zeros_like(init_latents, dtype=torch.bool)
        gt_patch = torch.randn_like(init_latents)
        args = SimpleNamespace(w_injection='unknown')

        out = inject_watermark(init_latents.clone(), mask, gt_patch, args)
        assert isinstance(out, torch.Tensor)
        assert out.shape == init_latents.shape
    
    # pipeline
    def test_detection_pipeline_watermark_detection_result_str(self):
        from markdiffusion.evaluation.pipelines.detection import WatermarkDetectionResult
        res = WatermarkDetectionResult(generated_or_retrieved_media=["img1"], edited_media=["img1_ed"], detect_result={"l1_distance": 0.5})
        s = str(res)
        assert "generated_or_retrieved_media" in s and "edited_media" in s and "detect_result" in s
    
    def test_detection_pipeline_edit_media_with_image_and_video_editors_and_invalid(self):
        from PIL import Image
        from types import SimpleNamespace
        from markdiffusion.evaluation.pipelines.detection import WatermarkDetectionPipeline
        from markdiffusion.evaluation.tools.image_editor import ImageEditor
        from markdiffusion.evaluation.tools.video_editor import VideoEditor

        # Stub dataset
        dataset = SimpleNamespace()
        pipeline = WatermarkDetectionPipeline(dataset=dataset, media_editor_list=[], show_progress=False)

        # Prepare media
        img1 = Image.new("RGB", (16, 16), color=(255, 0, 0))
        img2 = Image.new("RGB", (16, 16), color=(0, 255, 0))
        media_list = [img1, img2]

        # Stub ImageEditor that returns a new image
        class MyImageEditor(ImageEditor):
            def edit(self, img):
                return Image.new("RGB", img.size, color=(0, 0, 255))

        # Stub VideoEditor that returns a transformed list
        class MyVideoEditor(VideoEditor):
            def edit(self, frames):
                return [Image.new("RGB", f.size, color=(128, 128, 128)) for f in frames]

        # ImageEditor path
        pipeline.media_editor_list = [MyImageEditor()]
        edited = pipeline._edit_media(media_list.copy())
        assert isinstance(edited, list) and all(isinstance(e, Image.Image) for e in edited)

        # VideoEditor path
        pipeline.media_editor_list = [MyVideoEditor()]
        edited_v = pipeline._edit_media(media_list.copy())
        assert isinstance(edited_v, list) and all(isinstance(e, Image.Image) for e in edited_v)

        # Invalid editor type raises ValueError (covers line 64)
        class UnknownEditor: pass
        pipeline.media_editor_list = [UnknownEditor()]
        try:
            pipeline._edit_media(media_list.copy())
            assert False, "Expected ValueError for invalid editor type"
        except ValueError as e:
            assert "Invalid media type" in str(e)

    def test_watermarked_media_detection_pipeline_evaluate_return_types(self):
        from PIL import Image
        from types import SimpleNamespace
        from markdiffusion.evaluation.pipelines.detection import WatermarkedMediaDetectionPipeline, DetectionPipelineReturnType

        # Stub dataset
        class DummyDataset:
            num_samples = 3
            def get_prompt(self, idx): return f"prompt-{idx}"

        # Stub watermark
        class DummyWatermark:
            def generate_watermarked_media(self, input_data, **kwargs):
                # Return list of PIL images
                from PIL import Image
                return [Image.new("RGB", (8, 8), color=(idx % 255, 0, 0)) for idx in range(2)]
            def detect_watermark_in_media(self, media, detector_type="l1_distance", **kwargs):
                # Return dict with the requested keys
                return {"l1_distance": 0.42, "is_watermarked": True}

        dataset = DummyDataset()
        wm = DummyWatermark()

        # FULL return
        pipeline_full = WatermarkedMediaDetectionPipeline(dataset=dataset, media_editor_list=[], show_progress=False,
                                                        detector_type="l1_distance", return_type=DetectionPipelineReturnType.FULL)
        results_full = pipeline_full.evaluate(watermark=wm)
        assert isinstance(results_full, list) and len(results_full) == dataset.num_samples

        # SCORES return
        pipeline_scores = WatermarkedMediaDetectionPipeline(dataset=dataset, media_editor_list=[], show_progress=False,
                                                            detector_type="l1_distance", return_type=DetectionPipelineReturnType.SCORES)
        scores = pipeline_scores.evaluate(watermark=wm)
        assert isinstance(scores, list) and all(s == 0.42 for s in scores)

        # IS_WATERMARKED return
        pipeline_iswm = WatermarkedMediaDetectionPipeline(dataset=dataset, media_editor_list=[], show_progress=False,
                                                        detector_type="l1_distance", return_type=DetectionPipelineReturnType.IS_WATERMARKED)
        flags = pipeline_iswm.evaluate(watermark=wm)
        assert isinstance(flags, list) and all(flags)

    def test_watermarked_media_detection_invalid_generated_media_type_raises(self):
        from types import SimpleNamespace
        from markdiffusion.evaluation.pipelines.detection import WatermarkedMediaDetectionPipeline, DetectionPipelineReturnType

        class DummyDataset:
            num_samples = 1
            def get_prompt(self, idx): return "x"

        class BadWatermark:
            def generate_watermarked_media(self, input_data, **kwargs):
                return 123  # invalid type
            def detect_watermark_in_media(self, media, **kwargs):
                return {}

        pipeline = WatermarkedMediaDetectionPipeline(dataset=DummyDataset(), media_editor_list=[], show_progress=False,
                                                    detector_type="l1_distance", return_type=DetectionPipelineReturnType.SCORES)
        try:
            pipeline.evaluate(watermark=BadWatermark())
            assert False, "Expected ValueError due to invalid generated media type"
        except ValueError as e:
            assert "Invalid media type" in str(e)

    def test_unwatermarked_media_detection_get_iterable_modes_and_generate_or_retrieve(self):
        from PIL import Image
        from markdiffusion.evaluation.pipelines.detection import UnWatermarkedMediaDetectionPipeline

        # Ground truth mode: num_references != 0 triggers assert but passes
        class GroundTruthDataset:
            num_references = 2
            def get_reference(self, idx):
                return Image.new("RGB", (8, 8), color=(0, 255, 0))

        pipe_gt = UnWatermarkedMediaDetectionPipeline(dataset=GroundTruthDataset(), media_editor_list=[], media_source_mode="ground_truth")
        iterable_gt = pipe_gt._get_iterable()
        assert list(iterable_gt) == [0, 1]
        media_gt = pipe_gt._generate_or_retrieve_media(0, watermark=None)
        assert isinstance(media_gt, list) and isinstance(media_gt[0], Image.Image)

        # Generated mode
        class GeneratedDataset:
            num_samples = 3
            def get_prompt(self, idx): return f"p{idx}"

        class GenWatermark:
            def generate_unwatermarked_media(self, input_data, **kwargs):
                from PIL import Image
                return [Image.new("RGB", (8, 8), color=(255, 255, 0))]

        pipe_gen = UnWatermarkedMediaDetectionPipeline(dataset=GeneratedDataset(), media_editor_list=[], media_source_mode="generated")
        iterable_gen = pipe_gen._get_iterable()
        assert list(iterable_gen) == [0, 1, 2]
        media_gen = pipe_gen._generate_or_retrieve_media(0, watermark=GenWatermark())
        assert isinstance(media_gen, list) and isinstance(media_gen[0], Image.Image)

        # Invalid mode raises ValueError
        try:
            UnWatermarkedMediaDetectionPipeline(dataset=GeneratedDataset(), media_editor_list=[], media_source_mode="invalid")._get_iterable()
            assert False, "Expected ValueError for invalid media source mode"
        except ValueError as e:
            assert "Invalid media source mode" in str(e)

        # Generated mode with invalid generated media type raises ValueError
        class BadGenWM:
            def generate_unwatermarked_media(self, input_data, **kwargs):
                return 456  # invalid
        try:
            pipe_gen._generate_or_retrieve_media(0, watermark=BadGenWM())
            assert False, "Expected ValueError for invalid generated media type"
        except ValueError as e:
            assert "Invalid media type" in str(e)

    def test_base_pipeline_pass_methods(self):
        # Cover pass statements in base class methods _get_iterable and _generate_or_retrieve_media
        from types import SimpleNamespace
        from markdiffusion.evaluation.pipelines.detection import WatermarkDetectionPipeline

        dataset = SimpleNamespace()
        pipe = WatermarkDetectionPipeline(dataset=dataset, media_editor_list=[], show_progress=False)
        assert pipe._get_iterable() is None
        assert pipe._generate_or_retrieve_media(0, watermark=None) is None

    # image_quality_analyzer
    def test_inception_score_too_few_images(self, monkeypatch):
        import pytest
        from PIL import Image
        from markdiffusion.evaluation.tools.image_quality_analyzer import InceptionScoreCalculator
        # Avoid heavy model load
        monkeypatch.setattr(InceptionScoreCalculator, "_load_model", lambda self: None)

        isc = InceptionScoreCalculator(device="cpu", splits=3)
        images = [Image.new("RGB", (64, 64)) for _ in range(2)]
        with pytest.raises(ValueError):
            isc.analyze(images)

    def test_inception_score_non_divisible_splits(self, monkeypatch):
        import pytest
        from PIL import Image
        from markdiffusion.evaluation.tools.image_quality_analyzer import InceptionScoreCalculator

        monkeypatch.setattr(InceptionScoreCalculator, "_load_model", lambda self: None)

        isc = InceptionScoreCalculator(device="cpu", splits=2)
        images = [Image.new("RGB", (64, 64)) for _ in range(3)]
        with pytest.raises(ValueError):
            isc.analyze(images)

    def test_inception_get_predictions_converts_non_rgb(self, monkeypatch):
        import torch
        import numpy as np
        from PIL import Image
        from markdiffusion.evaluation.tools.image_quality_analyzer import InceptionScoreCalculator

        class DummyModel:
            def __call__(self, batch):
                # Return logits with 1000 classes
                return torch.zeros((batch.shape[0], 1000))

        def dummy_preprocess(img):
            # Return a 3x299x299 tensor regardless of input
            return torch.zeros(3, 299, 299)

        monkeypatch.setattr(InceptionScoreCalculator, "_load_model", lambda self: None)
        isc = InceptionScoreCalculator(device="cpu", splits=1)
        isc.model = DummyModel()
        isc.preprocess = dummy_preprocess

        # Grayscale image should be converted to RGB (line 132)
        gray_img = Image.new("L", (64, 64))
        preds = isc._get_predictions([gray_img])
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (1, 1000)

    def test_inception_warning_print(self, monkeypatch, capsys):
        import numpy as np
        from PIL import Image
        from markdiffusion.evaluation.tools.image_quality_analyzer import InceptionScoreCalculator

        monkeypatch.setattr(InceptionScoreCalculator, "_load_model", lambda self: None)

        isc = InceptionScoreCalculator(device="cpu", splits=2)

        # Make _get_predictions cheap
        monkeypatch.setattr(isc, "_get_predictions", lambda images: np.zeros((len(images), 1000)))
        # Force split scores to have high std
        monkeypatch.setattr(isc, "_calculate_inception_score", lambda predictions: [1.0, 10.0])

        images = [Image.new("RGB", (64, 64)) for _ in range(2)]
        scores = isc.analyze(images)
        captured = capsys.readouterr()
        assert "Warning: High standard deviation in IS calculation" in captured.out
        assert scores == [1.0, 10.0]

    def test_clip_init_no_clip_installed(self, monkeypatch):
        import pytest
        import builtins
        from markdiffusion.evaluation.tools.image_quality_analyzer import CLIPScoreCalculator

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "clip":
                raise ImportError("No clip")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError) as e:
            CLIPScoreCalculator(device="cpu")
        assert "Please install the CLIP library" in str(e.value)

    def test_clip_text_mode_wrong_reference_type(self, monkeypatch):
        import torch
        from PIL import Image
        import types
        from markdiffusion.evaluation.tools import image_quality_analyzer as iqa

        def stub_load(self):
            class DummyModel:
                def eval(self): pass
                def encode_image(self, t): return torch.ones((1, 512))
                def encode_text(self, t): return torch.ones((1, 512))
            self.model = DummyModel()
            self.preprocess = lambda img: torch.ones(3, 224, 224)  

        monkeypatch.setattr(iqa.CLIPScoreCalculator, "_load_model", stub_load)

        monkeypatch.setattr(iqa, "clip", types.SimpleNamespace(
            tokenize=lambda texts: torch.ones((1, 77), dtype=torch.long)
        ), raising=False)

        calc = iqa.CLIPScoreCalculator(device="cpu", reference_source="text")
        with_image_instead_of_text = Image.new("RGB", (64, 64))

        import pytest
        with pytest.raises(ValueError, match="Expected string reference for text mode"):
            calc.analyze(Image.new("RGB", (32, 32)), with_image_instead_of_text)


    def test_clip_text_mode_success(self, monkeypatch):
        import torch
        from PIL import Image
        import types
        from markdiffusion.evaluation.tools import image_quality_analyzer as iqa

        def stub_load(self):
            class DummyModel:
                def eval(self): pass
                def encode_image(self, t): return torch.ones((1, 512))
                def encode_text(self, t): return torch.ones((1, 512))
            self.model = DummyModel()
            self.preprocess = lambda img: torch.ones(3, 224, 224)  

        monkeypatch.setattr(iqa.CLIPScoreCalculator, "_load_model", stub_load)
        iqa.clip = types.SimpleNamespace(tokenize=lambda texts: torch.ones((1, 77), dtype=torch.long))

        calc = iqa.CLIPScoreCalculator(device="cpu", reference_source="text")
        # Grayscale image triggers conversion (line 264); text triggers tokenize and encode_text (280-281)
        score = calc.analyze(Image.new("L", (64, 64)), "a cat")
        assert 0.0 <= score <= 1.0 + 1e-5

    def test_clip_image_mode_wrong_reference_type(self, monkeypatch):
        import torch
        from PIL import Image
        from markdiffusion.evaluation.tools import image_quality_analyzer as iqa

        def stub_load(self):
            class DummyModel:
                def eval(self): pass
                def encode_image(self, t): return torch.ones((1, 512))
            self.model = DummyModel()
            self.preprocess = lambda img: torch.ones(3, 224, 224)  

        monkeypatch.setattr(iqa.CLIPScoreCalculator, "_load_model", stub_load)

        calc = iqa.CLIPScoreCalculator(device="cpu", reference_source="image")
        try:
            calc.analyze(Image.new("RGB", (32, 32)), "not an image")
        except ValueError as e:
            assert "Expected PIL Image reference for image mode" in str(e)

    def test_clip_image_mode_converts_reference(self, monkeypatch):
        import torch
        from PIL import Image
        from markdiffusion.evaluation.tools import image_quality_analyzer as iqa

        def stub_load(self):
            class DummyModel:
                def eval(self): pass
                def encode_image(self, t): return torch.ones((1, 512))
            self.model = DummyModel()
            self.preprocess = lambda img: torch.ones(3, 224, 224)  

        monkeypatch.setattr(iqa.CLIPScoreCalculator, "_load_model", stub_load)

        calc = iqa.CLIPScoreCalculator(device="cpu", reference_source="image")
        # Reference in L mode should be converted (line 289)
        score = calc.analyze(Image.new("RGB", (32, 32)), Image.new("L", (32, 32)))
        assert 0.0 <= score <= 1.0 + 1e-5

    def test_clip_invalid_reference_source(self, monkeypatch):
        import torch
        from PIL import Image
        from markdiffusion.evaluation.tools import image_quality_analyzer as iqa

        def stub_load(self):
            class DummyModel:
                def eval(self): pass
                def encode_image(self, t): return torch.ones((1, 512))
            self.model = DummyModel()
            self.preprocess = lambda img: torch.ones(3, 224, 224)  

        monkeypatch.setattr(iqa.CLIPScoreCalculator, "_load_model", stub_load)
        calc = iqa.CLIPScoreCalculator(device="cpu", reference_source="bad_source")
        try:
            calc.analyze(Image.new("RGB", (32, 32)), Image.new("RGB", (32, 32)))
        except ValueError as e:
            assert "Invalid reference_source" in str(e)

    def test_fid_too_few_images(self, monkeypatch):
        import pytest
        from PIL import Image
        from markdiffusion.evaluation.tools.image_quality_analyzer import FIDCalculator

        monkeypatch.setattr(FIDCalculator, "_load_model", lambda self: None)
        fid = FIDCalculator(device="cpu")
        with pytest.raises(ValueError):
            fid.analyze([Image.new("RGB", (64, 64))], [Image.new("RGB", (64, 64))])

    def test_fid_non_divisible_splits(self, monkeypatch):
        import pytest
        from PIL import Image
        from markdiffusion.evaluation.tools.image_quality_analyzer import FIDCalculator

        monkeypatch.setattr(FIDCalculator, "_load_model", lambda self: None)
        fid = FIDCalculator(device="cpu", splits=2)
        images = [Image.new("RGB", (64, 64)) for _ in range(3)]
        refs = [Image.new("RGB", (64, 64)) for _ in range(2)]
        with pytest.raises(ValueError):
            fid.analyze(images, refs)

    def test_fid_covmean_complex(self, monkeypatch):
        import numpy as np
        import types
        import sys
        from markdiffusion.evaluation.tools.image_quality_analyzer import FIDCalculator

        fake_scipy = types.SimpleNamespace(
            linalg=types.SimpleNamespace(
                sqrtm=lambda m: np.array([[1 + 1j]])
            )
        )
        monkeypatch.setitem(sys.modules, "scipy", fake_scipy)
        monkeypatch.setitem(sys.modules, "scipy.linalg", fake_scipy.linalg)

        monkeypatch.setattr(FIDCalculator, "_load_model", lambda self: None)
        fid = FIDCalculator(device="cpu")

        features1 = np.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]], dtype=np.float32)
        features2 = np.array([[1.0, 0.0, 3.0], [3.0, 0.0, 1.0]], dtype=np.float32)
        score = fid._calculate_fid(features1, features2)
        assert isinstance(score, float)

    def test_lpips_len_lt2(self, monkeypatch):
        from PIL import Image
        from markdiffusion.evaluation.tools.image_quality_analyzer import LPIPSAnalyzer

        monkeypatch.setattr(LPIPSAnalyzer, "_load_model", lambda self: None)
        lp = LPIPSAnalyzer(device="cpu")
        # Less than 2 images triggers early return 0.0 (line 477)
        assert lp.analyze([Image.new("RGB", (32, 32))]) == 0.0

    def test_lpips_converts_non_rgb(self, monkeypatch):
        import torch
        from PIL import Image
        from markdiffusion.evaluation.tools import image_quality_analyzer as iqa

        class DummyModel:
            def eval(self): pass
            def to(self, device): return self
            def forward(self, a, b): return torch.tensor(0.5)

        def dummy_im2tensor(arr):
            # Return minimal tensor on device
            return torch.zeros(1, 3, 16, 16)

        monkeypatch.setattr(iqa.lpips, "LPIPS", lambda net="alex": DummyModel())
        monkeypatch.setattr(iqa.lpips, "im2tensor", dummy_im2tensor)

        from markdiffusion.evaluation.tools.image_quality_analyzer import LPIPSAnalyzer
        lp = LPIPSAnalyzer(device="cpu")
        # Use grayscale images to hit conversion (line 483)
        img1 = Image.new("L", (16, 16))
        img2 = Image.new("L", (16, 16))
        score = lp.analyze([img1, img2])
        assert isinstance(score, float)

    def test_psnr_convert_resize_inf(self):
        import numpy as np
        from PIL import Image
        from markdiffusion.evaluation.tools.image_quality_analyzer import PSNRAnalyzer

        # Create identical images with different modes and sizes to trigger converts and resize (526, 528, 532)
        img = Image.new("L", (32, 32), color=128)
        ref = Image.new("RGB", (64, 64), color=(128, 128, 128))

        # Resize will make ref match img, and identical pixels -> mse==0 -> return inf (line 543)
        psnr = PSNRAnalyzer().analyze(img, ref)
        assert np.isinf(psnr)

    def test_niqe_load_params_failure(self):
        import pytest
        from markdiffusion.evaluation.tools.image_quality_analyzer import NIQECalculator

        # Provide a bad path to trigger load failure (lines 599-600)
        with pytest.raises(RuntimeError):
            NIQECalculator(model_path="nonexistent_params.mat")

    def test_niqe_analyze_rgb_la_conversion_small(self, monkeypatch):
        import numpy as np
        import pytest
        from PIL import Image
        from markdiffusion.evaluation.tools.image_quality_analyzer import NIQECalculator

        # Avoid heavy SciPy/gamma precompute in __init__
        def stub_load_params(self, model_path):
            # Minimal placeholders
            self.pop_mu = np.zeros(36, dtype=np.float32)
            self.pop_cov = np.eye(36, dtype=np.float32)

        monkeypatch.setattr(NIQECalculator, "_load_model_params", stub_load_params)
        monkeypatch.setattr(NIQECalculator, "_precompute_gamma_table", lambda self: None)
        monkeypatch.setattr(NIQECalculator, "_generate_gaussian_window", lambda self, w, s: np.array([1.0], dtype=np.float32))

        niqe = NIQECalculator()

        # RGB image hits conversion to 'LA' path (lines 898-899), then too small -> ValueError (line 909)
        small_rgb = Image.new("RGB", (10, 10))
        with pytest.raises(ValueError):
            niqe.analyze(small_rgb)

    def test_vif_convert_and_interpolate(self, monkeypatch):
        import torch
        from PIL import Image
        from markdiffusion.evaluation.tools import image_quality_analyzer as iqa

        # Replace piq.vif_p to avoid dependency and compute quickly
        monkeypatch.setattr(iqa.piq, "vif_p", lambda x, y, data_range=1.0: torch.tensor(0.42))

        from markdiffusion.evaluation.tools.image_quality_analyzer import VIFAnalyzer
        vif = VIFAnalyzer(device="cpu")
        # Grayscale images trigger conversion (1043), different sizes trigger interpolation (1063-1064)
        img = Image.new("L", (32, 40))
        ref = Image.new("RGB", (64, 80))
        score = vif.analyze(img, ref)
        assert isinstance(score, float)

    def test_fsim_convert_and_interpolate(self, monkeypatch):
        import torch
        from PIL import Image
        from markdiffusion.evaluation.tools import image_quality_analyzer as iqa

        monkeypatch.setattr(iqa.piq, "fsim", lambda x, y, data_range=1.0: torch.tensor(0.77))

        from markdiffusion.evaluation.tools.image_quality_analyzer import FSIMAnalyzer
        fsim = FSIMAnalyzer(device="cpu")
        # Grayscale triggers conversion (1087), different sizes trigger interpolation (1107-1108)
        img = Image.new("L", (48, 36))
        ref = Image.new("RGB", (96, 72))
        score = fsim.analyze(img, ref)
        assert isinstance(score, float)
    
    # video quality analysis
    def test_silent_progress_bar_iter_and_desc(self):
        from markdiffusion.evaluation.pipelines.video_quality_analysis import SilentProgressBar
        data = [1, 2, 3]
        bar = SilentProgressBar(data)
        bar.set_description("desc") # no-op
        assert list(iter(bar)) == data
    
    def test_pipeline_get_progress_bar_true_false(self):
        import types
        from markdiffusion.evaluation.pipelines.video_quality_analysis import VideoQualityAnalysisPipeline

        # Minimal stubs
        dataset = types.SimpleNamespace()
        pipeline_true = VideoQualityAnalysisPipeline(
            dataset=dataset,
            watermarked_video_editor_list=[],
            unwatermarked_video_editor_list=[],
            watermarked_frame_editor_list=[],
            unwatermarked_frame_editor_list=[],
            analyzers=[],
            show_progress=True,
            store_path=None,
        )
        pipeline_false = VideoQualityAnalysisPipeline(
            dataset=dataset,
            watermarked_video_editor_list=[],
            unwatermarked_video_editor_list=[],
            watermarked_frame_editor_list=[],
            unwatermarked_frame_editor_list=[],
            analyzers=[],
            show_progress=False,
            store_path=None,
        )
        iterable = [1, 2, 3]
        bar_true = pipeline_true._get_progress_bar(iterable)
        bar_false = pipeline_false._get_progress_bar(iterable)
        # tqdm should iterate same
        assert list(bar_true) == iterable
        # SilentProgressBar should iterate same
        assert list(bar_false) == iterable

    def test_pipeline_get_prompt(self):
        import types
        from markdiffusion.evaluation.pipelines.video_quality_analysis import VideoQualityAnalysisPipeline

        class DummyDataset:
            def get_prompt(self, index):
                return f"prompt_{index}"

        pipeline = VideoQualityAnalysisPipeline(
            dataset=DummyDataset(),
            watermarked_video_editor_list=[],
            unwatermarked_video_editor_list=[],
            watermarked_frame_editor_list=[],
            unwatermarked_frame_editor_list=[],
            analyzers=[],
            show_progress=False,
            store_path=None,
        )
        assert pipeline._get_prompt(5) == "prompt_5"

    def test_get_watermarked_video(self):
        import types
        from PIL import Image
        from markdiffusion.evaluation.pipelines.video_quality_analysis import VideoQualityAnalysisPipeline

        class DummyDataset:
            def get_prompt(self, index):
                return "p"

        class DummyWM:
            def generate_watermarked_media(self, input_data, **kwargs):
                return [Image.new("RGB", (8, 8)) for _ in range(2)]

        pipeline = VideoQualityAnalysisPipeline(
            dataset=DummyDataset(),
            watermarked_video_editor_list=[],
            unwatermarked_video_editor_list=[],
            watermarked_frame_editor_list=[],
            unwatermarked_frame_editor_list=[],
            analyzers=[],
            show_progress=False,
            store_path=None,
        )
        frames = pipeline._get_watermarked_video(DummyWM(), 0)
        assert isinstance(frames, list) and len(frames) == 2

    def test_get_unwatermarked_video(self):
        from PIL import Image
        from markdiffusion.evaluation.pipelines.video_quality_analysis import VideoQualityAnalysisPipeline

        class DummyDataset:
            def get_prompt(self, index):
                return "p"

        class DummyWM:
            def generate_unwatermarked_media(self, input_data, **kwargs):
                return [Image.new("RGB", (8, 8)) for _ in range(3)]

        pipeline = VideoQualityAnalysisPipeline(
            dataset=DummyDataset(),
            watermarked_video_editor_list=[],
            unwatermarked_video_editor_list=[],
            watermarked_frame_editor_list=[],
            unwatermarked_frame_editor_list=[],
            analyzers=[],
            show_progress=False,
            store_path=None,
        )
        frames = pipeline._get_unwatermarked_video(DummyWM(), 1)
        assert isinstance(frames, list) and len(frames) == 3

    def test_edit_watermarked_and_unwatermarked_video(self):
        from PIL import Image
        from markdiffusion.evaluation.pipelines.video_quality_analysis import VideoQualityAnalysisPipeline

        class DummyFrameEditor:
            def edit(self, frame):
                # draw a simple modification: convert to L then back
                return frame.convert("L").convert("RGB")

        class DummyVideoEditor:
            def edit(self, frames):
                # reverse frames order to ensure change
                return list(reversed(frames))

        pipeline = VideoQualityAnalysisPipeline(
            dataset=object(),
            watermarked_video_editor_list=[DummyVideoEditor()],
            unwatermarked_video_editor_list=[DummyVideoEditor()],
            watermarked_frame_editor_list=[DummyFrameEditor()],
            unwatermarked_frame_editor_list=[DummyFrameEditor()],
            analyzers=[],
            show_progress=False,
            store_path=None,
        )
        frames = [Image.new("RGB", (8, 8), color=(i, i, i)) for i in range(3)]
        w_out = pipeline._edit_watermarked_video(frames)
        u_out = pipeline._edit_unwatermarked_video(frames)
        # Should have same length and be modified
        assert len(w_out) == 3 and len(u_out) == 3

    def test_prepare_dataset_with_references_silent_progress(self):
        from PIL import Image
        from markdiffusion.evaluation.pipelines.video_quality_analysis import DirectVideoQualityAnalysisPipeline

        class DummyDataset:
            def __init__(self):
                self.num_samples = 2
                self.num_references = 1
                self.name = "dummy"

            def get_prompt(self, index):
                return f"prompt_{index}"

            def get_reference(self, index):
                return [Image.new("RGB", (8, 8)) for _ in range(2)]

        class DummyWM:
            def generate_watermarked_media(self, input_data, **kwargs):
                return [Image.new("RGB", (8, 8)) for _ in range(2)]

            def generate_unwatermarked_media(self, input_data, **kwargs):
                return [Image.new("RGB", (8, 8)) for _ in range(2)]

        # Use DirectVideoQualityAnalysisPipeline to ensure _get_iterable returns a valid range
        pipeline = DirectVideoQualityAnalysisPipeline(
            dataset=DummyDataset(),
            watermarked_video_editor_list=[],
            unwatermarked_video_editor_list=[],
            watermarked_frame_editor_list=[],
            unwatermarked_frame_editor_list=[],
            analyzers=[],
            show_progress=False,  # Silent progress bar path
            store_path=None,
        )
        dataset_eval = pipeline._prepare_dataset(DummyWM())
        assert len(dataset_eval.watermarked_videos) == 2
        assert len(dataset_eval.unwatermarked_videos) == 2
        assert len(dataset_eval.indexes) == 2
        assert len(dataset_eval.reference_videos) == 2  # num_references > 0 branch

    def test_prepare_input_for_quality_analyzer_base_returns_none(self):
        from markdiffusion.evaluation.pipelines.video_quality_analysis import VideoQualityAnalysisPipeline
        pipeline = VideoQualityAnalysisPipeline(
            dataset=object(),
            watermarked_video_editor_list=[],
            unwatermarked_video_editor_list=[],
            watermarked_frame_editor_list=[],
            unwatermarked_frame_editor_list=[],
            analyzers=[],
            show_progress=False,
            store_path=None,
        )
        # Base method is pass, returning None
        assert pipeline._prepare_input_for_quality_analyzer([], [], []) is None

    def test_store_results_creates_files(self, tmp_path):
        from PIL import Image
        from markdiffusion.evaluation.pipelines.video_quality_analysis import VideoQualityAnalysisPipeline, DatasetForEvaluation

        class DummyDataset:
            name = "ds"
            num_references = 1

            def get_reference(self, index):
                return [Image.new("RGB", (8, 8))]

        pipeline = VideoQualityAnalysisPipeline(
            dataset=DummyDataset(),
            watermarked_video_editor_list=[],
            unwatermarked_video_editor_list=[],
            watermarked_frame_editor_list=[],
            unwatermarked_frame_editor_list=[],
            analyzers=[],
            show_progress=False,
            store_path=str(tmp_path),
        )

        prepared = DatasetForEvaluation(
            watermarked_videos=[[Image.new("RGB", (8, 8)) for _ in range(2)]],
            unwatermarked_videos=[[Image.new("RGB", (8, 8)) for _ in range(2)]],
            reference_videos=[],
            indexes=[0],
        )
        pipeline._store_results(prepared)
        # Check directories created
        water_dir = tmp_path / f"{pipeline.__class__.__name__}_{pipeline.dataset.name}_watermarked_prompt0"
        unwater_dir = tmp_path / f"{pipeline.__class__.__name__}_{pipeline.dataset.name}_unwatermarked_prompt0"
        assert water_dir.exists()
        assert unwater_dir.exists()
        # Check frames saved
        assert (water_dir / "frame_0.png").exists()
        assert (unwater_dir / "frame_0.png").exists()

    def test_direct_pipeline_evaluate_runs(self):
        from PIL import Image
        from markdiffusion.evaluation.pipelines.video_quality_analysis import DirectVideoQualityAnalysisPipeline

        class DummyDataset:
            def __init__(self):
                self.num_samples = 2
                self.name = "ds"
                self.num_references = 0  # ensure attribute exists to avoid AttributeError
            def get_prompt(self, index): return f"p{index}"

        class DummyWM:
            def generate_watermarked_media(self, input_data, **kwargs):
                return [Image.new("RGB", (8, 8)) for _ in range(2)]
            def generate_unwatermarked_media(self, input_data, **kwargs):
                return [Image.new("RGB", (8, 8)) for _ in range(2)]

        class DummyAnalyzer:
            def analyze(self, video_frames):
                return float(len(video_frames))

        pipeline = DirectVideoQualityAnalysisPipeline(
            dataset=DummyDataset(),
            watermarked_video_editor_list=[],
            unwatermarked_video_editor_list=[],
            watermarked_frame_editor_list=[],
            unwatermarked_frame_editor_list=[],
            analyzers=[DummyAnalyzer()],
            show_progress=False,
            store_path=None,
        )
        out = pipeline.evaluate(DummyWM())
        assert "watermarked" in out and "unwatermarked" in out
        assert "DummyAnalyzer" in out["watermarked"]
        assert "DummyAnalyzer" in out["unwatermarked"]

    def test_direct_pipeline_evaluate_runs(self):
        from PIL import Image
        from markdiffusion.evaluation.pipelines.video_quality_analysis import DirectVideoQualityAnalysisPipeline

        class DummyDataset:
            def __init__(self):
                self.num_samples = 2
                self.name = "ds"
                self.num_references = 0  # ensure attribute exists to avoid AttributeError

            def get_prompt(self, index):
                return f"p{index}"

        class DummyWM:
            def generate_watermarked_media(self, input_data, **kwargs):
                return [Image.new("RGB", (8, 8)) for _ in range(2)]

            def generate_unwatermarked_media(self, input_data, **kwargs):
                return [Image.new("RGB", (8, 8)) for _ in range(2)]

        class DummyAnalyzer:
            def analyze(self, video_frames):
                return float(len(video_frames))

        pipeline = DirectVideoQualityAnalysisPipeline(
            dataset=DummyDataset(),
            watermarked_video_editor_list=[],
            unwatermarked_video_editor_list=[],
            watermarked_frame_editor_list=[],
            unwatermarked_frame_editor_list=[],
            analyzers=[DummyAnalyzer()],
            show_progress=False,
            store_path=None,
        )
        out = pipeline.evaluate(DummyWM())
        assert "watermarked" in out and "unwatermarked" in out
        assert "DummyAnalyzer" in out["watermarked"]
        assert "DummyAnalyzer" in out["unwatermarked"]
    
    # media_utils
    def test_transform_to_model_format_image_with_target(self):
        from PIL import Image
        import torch
        from markdiffusion.utils.media_utils import transform_to_model_format
        img = Image.new("RGB", (32, 24), color=(128, 128, 128))
        out = transform_to_model_format(img, target_size=16)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 16, 16)
        assert torch.all(out >= -1.0) and torch.all(out <= 1.0)

    def test_transform_to_model_format_image_no_target(self):
        from PIL import Image
        import torch
        from markdiffusion.utils.media_utils import transform_to_model_format

        img = Image.new("RGB", (20, 10), color=(255, 0, 0))
        out = transform_to_model_format(img)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 10, 20)

    def test_transform_to_model_format_list_pil(self):
        from PIL import Image
        import torch
        from markdiffusion.utils.media_utils import transform_to_model_format

        frames = [Image.new("RGB", (8, 8), color=(i, i, i)) for i in range(3)]
        out = transform_to_model_format(frames)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 3, 8, 8)

    def test_transform_to_model_format_list_numpy(self):
        import numpy as np
        import torch
        from markdiffusion.utils.media_utils import transform_to_model_format

        frames = [np.ones((8, 8, 3), dtype=np.float32) for _ in range(2)]
        out = transform_to_model_format(frames)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 3, 8, 8)

    def test_transform_to_model_format_list_mixed_error(self):
        import numpy as np
        from PIL import Image
        import pytest
        from markdiffusion.utils.media_utils import transform_to_model_format

        frames = [Image.new("RGB", (8, 8)), np.ones((8, 8, 3), dtype=np.float32)]
        with pytest.raises(ValueError):
            transform_to_model_format(frames)

    def test_transform_to_model_format_unsupported_media_error(self):
        import pytest
        from markdiffusion.utils.media_utils import transform_to_model_format
        with pytest.raises(ValueError):
            transform_to_model_format(123)

    def test_numpy_to_pil_and_cv2_to_pil(self):
        import numpy as np
        from markdiffusion.utils.media_utils import numpy_to_pil, cv2_to_pil

        arr_float = np.random.rand(10, 12, 3).astype(np.float32)
        pil_img = numpy_to_pil(arr_float)
        assert pil_img.size == (12, 10)

        arr_uint8 = (arr_float * 255).astype(np.uint8)
        pil_img2 = cv2_to_pil(arr_uint8)
        assert pil_img2.size == (12, 10)

    def test_pil_to_cv2(self):
        from PIL import Image
        import numpy as np
        from markdiffusion.utils.media_utils import pil_to_cv2

        img = Image.new("RGB", (9, 7), color=(0, 128, 255))
        arr = pil_to_cv2(img)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (7, 9, 3)
        # Values are in 0-1
        assert arr.min() >= 0.0 and arr.max() <= 1.0

    def test_get_random_latents_image_path(self, monkeypatch):
        import torch
        from types import SimpleNamespace
        from markdiffusion.utils import media_utils

        # Force pipeline type to image
        monkeypatch.setattr(media_utils, "get_pipeline_type", lambda pipe: media_utils.PIPELINE_TYPE_IMAGE)

        called_args = {}

        def fake_prepare_latents(batch_size, num_channels_latents, height, width, dtype, device, generator, latents):
            called_args["args"] = (batch_size, num_channels_latents, height, width, dtype, device, generator, latents)
            return torch.randn(1, num_channels_latents, height // 8, width // 8)

        pipe = SimpleNamespace(
            unet=SimpleNamespace(config=SimpleNamespace(sample_size=64, in_channels=4)),
            vae_scale_factor=8,
            _execution_device="cpu",
            text_encoder=SimpleNamespace(dtype=torch.float32),
            prepare_latents=fake_prepare_latents,
        )

        latents = media_utils.get_random_latents(pipe, height=None, width=None)
        # Verify prepare_latents was called with derived height/width = sample_size * vae_scale_factor = 512
        args = called_args["args"]
        assert args[2] == 512 and args[3] == 512
        assert isinstance(latents, torch.Tensor)

    def test_get_random_latents_video_path(self, monkeypatch):
        import torch
        from types import SimpleNamespace
        from markdiffusion.utils import media_utils

        monkeypatch.setattr(media_utils, "get_pipeline_type", lambda pipe: media_utils.PIPELINE_TYPE_TEXT_TO_VIDEO)

        called_args = {}

        def fake_prepare_latents(batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents):
            called_args["args"] = (batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents)
            return torch.randn(1, num_channels_latents, num_frames, height // 8, width // 8)

        pipe = SimpleNamespace(
            unet=SimpleNamespace(config=SimpleNamespace(sample_size=64, in_channels=4)),
            vae_scale_factor=8,
            _execution_device="cpu",
            text_encoder=SimpleNamespace(dtype=torch.float32),
            prepare_latents=fake_prepare_latents,
        )

        latents = media_utils.get_random_latents(pipe, num_frames=5, height=None, width=None)
        args = called_args["args"]
        assert args[2] == 5
        assert isinstance(latents, torch.Tensor)

    def test_get_image_latents_sample_and_decoder_inv(self, monkeypatch):
        import torch
        from types import SimpleNamespace
        from markdiffusion.utils import media_utils

        # Stub decoder_inv_optimization to return latents * 2
        monkeypatch.setattr(media_utils, "decoder_inv_optimization", lambda pipe, latents, image, num_steps=100: latents * 2)

        class DummyLatentDist:
            def sample(self, generator=None):
                return torch.ones(1, 3, 4, 4)
            def mode(self):
                return torch.zeros(1, 3, 4, 4)

        pipe = SimpleNamespace(vae=SimpleNamespace(encode=lambda x: SimpleNamespace(latent_dist=DummyLatentDist())))

        img = torch.randn(1, 3, 4, 4)
        # sample=True, decoder_inv=True
        lat = media_utils._get_image_latents(pipe, img, sample=True, rng_generator=None, decoder_inv=True)
        assert torch.allclose(lat, torch.ones_like(lat) * (0.18215 * 2), atol=1e-6)
        # sample=False, decoder_inv=False
        lat2 = media_utils._get_image_latents(pipe, img, sample=False, rng_generator=None, decoder_inv=False)
        assert torch.allclose(lat2, torch.zeros_like(lat2), atol=1e-6)

    def test_decode_image_latents(self):
        import torch
        from types import SimpleNamespace
        from markdiffusion.utils import media_utils

        # Return a single tensor from decode
        pipe = SimpleNamespace(vae=SimpleNamespace(decode=lambda latents, return_dict=False: (torch.zeros(1, 3, 4, 4),)))
        latents = torch.ones(1, 3, 4, 4) * 0.18215
        out = media_utils._decode_image_latents(pipe, latents)
        assert out.shape == (1, 3, 4, 4)
        assert torch.all(out >= 0.0) and torch.all(out <= 1.0)

    def test_get_video_latents_permute_true_false_and_decoder_inv_error(self):
        import torch
        from types import SimpleNamespace
        from markdiffusion.utils import media_utils
        import pytest

        class DummyLatentDist:
            def sample(self, generator=None):
                # return [F, C, H, W]
                return torch.ones(5, 4, 8, 8)
            def mode(self):
                return torch.zeros(5, 4, 8, 8)

        pipe = SimpleNamespace(vae=SimpleNamespace(encode=lambda vf: SimpleNamespace(latent_dist=DummyLatentDist())))
        video_frames = torch.randn(5, 3, 8, 8)

        # permute True: output [B, C, F, H, W]
        latents_perm = media_utils._get_video_latents(pipe, video_frames, sample=True, rng_generator=None, permute=True, decoder_inv=False)
        assert latents_perm.shape == (1, 4, 5, 8, 8)

        # permute False: output [B, F, C, H, W]
        latents_noperm = media_utils._get_video_latents(pipe, video_frames, sample=False, rng_generator=None, permute=False, decoder_inv=False)
        assert latents_noperm.shape == (1, 5, 4, 8, 8)

        # decoder_inv True raises NotImplementedError
        with pytest.raises(NotImplementedError):
            media_utils._get_video_latents(pipe, video_frames, decoder_inv=True)

    def test_tensor2vid_output_types(self):
        import torch
        import numpy as np
        from PIL import Image
        from markdiffusion.utils.media_utils import tensor2vid

        # Video tensor [B, C, F, H, W]
        video = torch.zeros(2, 3, 4, 6, 5)

        class DummyProcessor:
            def postprocess(self, batch_vid, output_type):
                F, C, H, W = batch_vid.shape
                if output_type == "np":
                    return np.zeros((F, H, W, C), dtype=np.uint8)
                elif output_type == "pt":
                    return torch.zeros(F, C, H, W)
                elif output_type == "pil":
                    return [Image.new("RGB", (W, H)) for _ in range(F)]
                else:
                    raise ValueError("bad")

        # np
        out_np = tensor2vid(video, DummyProcessor(), output_type="np")
        assert isinstance(out_np, np.ndarray)
        assert out_np.shape == (2, 4, 6, 5, 3)

        # pt
        out_pt = tensor2vid(video, DummyProcessor(), output_type="pt")
        assert isinstance(out_pt, torch.Tensor)
        assert out_pt.shape == (2, 4, 3, 6, 5)

        # pil
        out_pil = tensor2vid(video, DummyProcessor(), output_type="pil")
        assert isinstance(out_pil, list) and len(out_pil) == 2
        assert isinstance(out_pil[0], list) and isinstance(out_pil[0][0], Image.Image)

        # invalid
        import pytest
        with pytest.raises(ValueError):
            tensor2vid(video, DummyProcessor(), output_type="bad")

    def test_decode_video_latents(self):
        import torch
        import numpy as np
        from markdiffusion.utils.media_utils import _decode_video_latents

        class DummyProcessor:
            def postprocess(self, batch_vid, output_type):
                F, C, H, W = batch_vid.shape
                return np.zeros((F, H, W, C), dtype=np.uint8)

        class DummyPipe:
            def __init__(self):
                self.video_processor = DummyProcessor()
            def decode_latents(self, latents, num_frames=None):
                # return [B, C, F, H, W]
                b = 1
                f = num_frames if num_frames is not None else 4
                return torch.zeros(b, 3, f, 6, 5)

        pipe = DummyPipe()
        # Without num_frames
        video_np = _decode_video_latents(pipe, torch.randn(1, 3, 4, 6, 5), num_frames=None)
        assert isinstance(video_np, np.ndarray)
        assert video_np.shape == (1, 4, 6, 5, 3)
        # With num_frames
        video_np2 = _decode_video_latents(pipe, torch.randn(1, 3, 2, 6, 5), num_frames=2)
        assert video_np2.shape == (1, 2, 6, 5, 3)

    def test_convert_video_frames_to_images_numpy_and_pil(self):
        import numpy as np
        from PIL import Image
        from markdiffusion.utils.media_utils import convert_video_frames_to_images

        frames_np = [np.random.rand(8, 8, 3).astype(np.float32) for _ in range(2)]
        pil_frames_np = convert_video_frames_to_images(frames_np)
        assert all(isinstance(f, Image.Image) for f in pil_frames_np)

        frames_pil = [Image.new("RGB", (8, 8)) for _ in range(2)]
        pil_frames_pil = convert_video_frames_to_images(frames_pil)
        assert all(isinstance(f, Image.Image) for f in pil_frames_pil)

    def test_convert_video_frames_to_images_unsupported_type(self):
        import pytest
        from markdiffusion.utils.media_utils import convert_video_frames_to_images
        with pytest.raises(ValueError):
            convert_video_frames_to_images(["bad"])

    def test_save_video_frames_numpy_and_pil(self, tmp_path):
        import numpy as np
        from PIL import Image
        import os
        from markdiffusion.utils.media_utils import save_video_frames

        save_dir1 = tmp_path / "np_frames"
        os.makedirs(save_dir1, exist_ok=True)
        frames_np = [np.random.rand(8, 8, 3).astype(np.float32) for _ in range(2)]
        save_video_frames(frames_np, str(save_dir1))
        assert (save_dir1 / "00.png").exists()
        assert (save_dir1 / "01.png").exists()

        save_dir2 = tmp_path / "pil_frames"
        os.makedirs(save_dir2, exist_ok=True)
        frames_pil = [Image.new("RGB", (8, 8)) for _ in range(2)]
        save_video_frames(frames_pil, str(save_dir2))
        assert (save_dir2 / "00.png").exists()
        assert (save_dir2 / "01.png").exists()

    def test_get_media_latents_image_vs_video(self, monkeypatch):
        import torch
        from types import SimpleNamespace
        from markdiffusion.utils import media_utils

        # Image path
        monkeypatch.setattr(media_utils, "get_pipeline_type", lambda pipe: media_utils.PIPELINE_TYPE_IMAGE)

        class DummyPipeImg:
            def __init__(self):
                self.vae = SimpleNamespace(encode=lambda x: SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda generator=None: torch.ones(1, 3, 4, 4), mode=lambda: torch.zeros(1, 3, 4, 4))))
        img_tensor = torch.randn(1, 3, 4, 4)
        lat_img = media_utils.get_media_latents(DummyPipeImg(), img_tensor, sample=True, rng_generator=None, decoder_inv=False)
        assert lat_img.shape == (1, 3, 4, 4)

        # Text-to-video path (permute True)
        monkeypatch.setattr(media_utils, "get_pipeline_type", lambda pipe: media_utils.PIPELINE_TYPE_TEXT_TO_VIDEO)

        class DummyPipeVid:
            def __init__(self):
                self.vae = SimpleNamespace(encode=lambda vf: SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda generator=None: torch.ones(5, 4, 8, 8), mode=lambda: torch.zeros(5, 4, 8, 8))))
        vid_frames = torch.randn(5, 3, 8, 8)
        lat_vid = media_utils.get_media_latents(DummyPipeVid(), vid_frames, sample=True, rng_generator=None, decoder_inv=False)
        assert lat_vid.shape == (1, 4, 5, 8, 8)

        # Image-to-video path (permute False)
        monkeypatch.setattr(media_utils, "get_pipeline_type", lambda pipe: media_utils.PIPELINE_TYPE_IMAGE_TO_VIDEO)
        lat_vid2 = media_utils.get_media_latents(DummyPipeVid(), vid_frames, sample=False, rng_generator=None, decoder_inv=False)
        assert lat_vid2.shape == (1, 5, 4, 8, 8)

    def test_decode_media_latents_image_vs_video(self, monkeypatch):
        import torch
        import numpy as np
        from types import SimpleNamespace
        from markdiffusion.utils import media_utils

        # Image path
        monkeypatch.setattr(media_utils, "get_pipeline_type", lambda pipe: media_utils.PIPELINE_TYPE_IMAGE)

        pipe_img = SimpleNamespace(vae=SimpleNamespace(decode=lambda latents, return_dict=False: (torch.zeros(1, 3, 4, 4),)))
        latents_img = torch.ones(1, 3, 4, 4) * 0.18215
        out_img = media_utils.decode_media_latents(pipe_img, latents_img)
        assert isinstance(out_img, torch.Tensor)
        assert out_img.shape == (1, 3, 4, 4)

        # Video path
        monkeypatch.setattr(media_utils, "get_pipeline_type", lambda pipe: media_utils.PIPELINE_TYPE_TEXT_TO_VIDEO)

        class DummyProcessor:
            def postprocess(self, batch_vid, output_type):
                F, C, H, W = batch_vid.shape
                return np.zeros((F, H, W, C), dtype=np.uint8)

        class DummyPipeVid:
            def __init__(self):
                self.video_processor = DummyProcessor()
            def decode_latents(self, latents, num_frames=None):
                f = num_frames if num_frames is not None else 4
                return torch.zeros(1, 3, f, 6, 5)

        pipe_vid = DummyPipeVid()
        vid_np = media_utils.decode_media_latents(pipe_vid, torch.randn(1, 3, 4, 6, 5), num_frames=3)
        assert isinstance(vid_np, np.ndarray)
        assert vid_np.shape == (1, 3, 6, 5, 3) or vid_np.shape == (1, 3, 6, 5, 3)  # shape from tensor2vid stacking

    # base
    def test_baseconfig_raises_when_diffusion_config_none(self, monkeypatch):
        import pytest
        import markdiffusion.watermark.base as base_mod

        monkeypatch.setattr(base_mod, "load_config_file", lambda _: {})

        class DummyConfig(base_mod.BaseConfig):
            @property
            def algorithm_name(self):
                return lambda: "dummy"

            def initialize_parameters(self) -> None:
                pass

        with pytest.raises(ValueError, match="diffusion_config cannot be None"):
            DummyConfig(algorithm_config=None, diffusion_config=None)

    def test_baseconfig_updates_config_dict_with_kwargs(self):
        import pytest
        from markdiffusion.watermark.base import BaseConfig

        class DummyDiffusionConfig:
            def __init__(self):
                self.pipe = object()
                self.scheduler = object()
                self.device = "cpu"
                self.guidance_scale = 1.0
                self.num_images = 1
                self.num_inference_steps = 1
                self.num_inversion_steps = 1
                self.image_size = (64, 64)
                self.dtype = None
                self.gen_seed = 0
                self.init_latents_seed = 0
                self.inversion_type = "none"
                self.num_frames = -1
                self.gen_kwargs = {}

        import markdiffusion.watermark.base as base_mod

        def fake_load_config_file(_):
            return {"a": 1}

        def fake_set_inversion(pipe, inv_type):
            return ("inv", pipe, inv_type)

        def fake_get_random_latents(pipe, **kwargs):
            return "LATENTS"

        base_mod.load_config_file = fake_load_config_file
        base_mod.set_inversion = fake_set_inversion
        base_mod.get_random_latents = fake_get_random_latents

        class DummyConfig(base_mod.BaseConfig):
            @property
            def algorithm_name(self) -> str:
                return "dummy"

            def initialize_parameters(self) -> None:
                pass

        cfg = DummyConfig(
            algorithm_config="whatever.json",
            diffusion_config=DummyDiffusionConfig(),
            new_key=123,
            a=999,
        )
        assert cfg.config_dict["new_key"] == 123
        assert cfg.config_dict["a"] == 999  # 覆盖原值

    def test_basewatermark_detect_pipeline_type_raises_when_unknown(self, monkeypatch):
        import pytest
        import markdiffusion.watermark.base as base_mod

        class DummyConfigObj:
            def __init__(self):
                self.pipe = object()
                self.num_frames = -1

        class DummyWM(base_mod.BaseWatermark):
            def get_data_for_visualize(self, media, *args, **kwargs):
                return None

        monkeypatch.setattr(base_mod, "get_pipeline_type", lambda _: None)

        with pytest.raises(ValueError, match="Unsupported pipeline type"):
            DummyWM(DummyConfigObj())

    def test_validate_pipeline_config_raises_for_video_pipeline_num_frames_lt_1(self, monkeypatch):
        import pytest
        import markdiffusion.watermark.base as base_mod

        class DummyConfigObj:
            def __init__(self):
                self.pipe = object()
                self.num_frames = 0

        class DummyWM(base_mod.BaseWatermark):
            def get_data_for_visualize(self, media, *args, **kwargs):
                return None

        monkeypatch.setattr(base_mod, "get_pipeline_type", lambda _: base_mod.PIPELINE_TYPE_TEXT_TO_VIDEO)

        with pytest.raises(ValueError, match="num_frames must be >= 1"):
            DummyWM(DummyConfigObj())

    def test_validate_pipeline_config_raises_for_image_pipeline_num_frames_ge_1(sel, monkeypatch):
        import pytest
        import markdiffusion.watermark.base as base_mod

        class DummyConfigObj:
            def __init__(self):
                self.pipe = object()
                self.num_frames = 1

        class DummyWM(base_mod.BaseWatermark):
            def get_data_for_visualize(self, media, *args, **kwargs):
                return None

        monkeypatch.setattr(base_mod, "get_pipeline_type", lambda _: base_mod.PIPELINE_TYPE_IMAGE)

        with pytest.raises(ValueError, match="num_frames should be -1"):
            DummyWM(DummyConfigObj())

    def test_generate_watermarked_media_raises_if_image_pipeline_input_not_str(self, monkeypatch):
        import pytest
        import markdiffusion.watermark.base as base_mod
        from PIL import Image

        class DummyConfigObj:
            def __init__(self):
                self.pipe = object()
                self.num_frames = -1

        class DummyWM(base_mod.BaseWatermark):
            def get_data_for_visualize(self, media, *args, **kwargs):
                return None


        monkeypatch.setattr(base_mod, "get_pipeline_type", lambda _: base_mod.PIPELINE_TYPE_IMAGE)
        monkeypatch.setattr(base_mod, "is_image_pipeline", lambda _: True)
        monkeypatch.setattr(base_mod, "is_video_pipeline", lambda _: False)

        wm = DummyWM(DummyConfigObj())
        with pytest.raises(ValueError, match="input_data must be a text prompt"):
            wm.generate_watermarked_media(Image.new("RGB", (8, 8)))

    def test_generate_unwatermarked_media_raises_if_image_pipeline_input_not_str(self, monkeypatch):
        import pytest
        import markdiffusion.watermark.base as base_mod
        from PIL import Image

        class DummyConfigObj:
            def __init__(self):
                self.pipe = object()
                self.num_frames = -1

        class DummyWM(base_mod.BaseWatermark):
            def get_data_for_visualize(self, media, *args, **kwargs):
                return None

        monkeypatch.setattr(base_mod, "get_pipeline_type", lambda _: base_mod.PIPELINE_TYPE_IMAGE)
        monkeypatch.setattr(base_mod, "is_image_pipeline", lambda _: True)
        monkeypatch.setattr(base_mod, "is_video_pipeline", lambda _: False)

        wm = DummyWM(DummyConfigObj())
        with pytest.raises(ValueError, match="input_data must be a text prompt"):
            wm.generate_unwatermarked_media(Image.new("RGB", (8, 8)))

    def test_preprocess_media_image_pipeline_numpy_to_pil(self, monkeypatch):
        import numpy as np
        import markdiffusion.watermark.base as base_mod
        from PIL import Image

        class DummyConfigObj:
            def __init__(self):
                self.pipe = object()
                self.num_frames = -1

        class DummyWM(base_mod.BaseWatermark):
            def get_data_for_visualize(self, media, *args, **kwargs):
                return None

        monkeypatch.setattr(base_mod, "get_pipeline_type", lambda _: base_mod.PIPELINE_TYPE_IMAGE)
        monkeypatch.setattr(base_mod, "is_image_pipeline", lambda _: True)
        monkeypatch.setattr(base_mod, "is_video_pipeline", lambda _: False)

        
        monkeypatch.setattr(base_mod, "cv2_to_pil", lambda arr: Image.fromarray(arr.astype("uint8")))

        wm = DummyWM(DummyConfigObj())
        arr = np.zeros((8, 8, 3), dtype=np.uint8)
        out = wm._preprocess_media_for_detection(arr)
        assert isinstance(out, Image.Image)

    def test_preprocess_media_image_pipeline_tensor_to_pil_dim3(self, monkeypatch):
        import torch
        import numpy as np
        import markdiffusion.watermark.base as base_mod
        from PIL import Image

        class DummyConfigObj:
            def __init__(self):
                self.pipe = object()
                self.num_frames = -1

        class DummyWM(base_mod.BaseWatermark):
            def get_data_for_visualize(self, media, *args, **kwargs):
                return None

        
        monkeypatch.setattr(base_mod, "get_pipeline_type", lambda _: base_mod.PIPELINE_TYPE_IMAGE)
        
        monkeypatch.setattr(base_mod, "is_image_pipeline", lambda _: True)
        
        monkeypatch.setattr(base_mod, "is_video_pipeline", lambda _: False)

        def fake_torch_to_numpy(t):
            # t expected B,C,H,W
            b, c, h, w = t.shape
            arr = t.detach().cpu().numpy().transpose(0, 2, 3, 1)
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
            return arr

        base_mod.torch_to_numpy = fake_torch_to_numpy
        monkeypatch.setattr(base_mod, "torch_to_numpy", fake_torch_to_numpy)

        wm = DummyWM(DummyConfigObj())
        x = torch.zeros(3, 8, 8)  
        out = wm._preprocess_media_for_detection(x)
        assert isinstance(out, Image.Image)

    def test_preprocess_media_image_pipeline_unsupported_type_raises(self, monkeypatch):
        import pytest
        import markdiffusion.watermark.base as base_mod

        class DummyConfigObj:
            def __init__(self):
                self.pipe = object()
                self.num_frames = -1

        class DummyWM(base_mod.BaseWatermark):
            def get_data_for_visualize(self, media, *args, **kwargs):
                return None

        
        monkeypatch.setattr(base_mod, "get_pipeline_type", lambda _: base_mod.PIPELINE_TYPE_IMAGE)
        
        monkeypatch.setattr(base_mod, "is_image_pipeline", lambda _: True)
        
        monkeypatch.setattr(base_mod, "is_video_pipeline", lambda _: False)

        wm = DummyWM(DummyConfigObj())
        with pytest.raises(ValueError, match="Unsupported media type for image pipeline"):
            wm._preprocess_media_for_detection(12345)

    def test_preprocess_media_video_pipeline_list_numpy_frames_to_pil(self, monkeypatch):
        import numpy as np
        import markdiffusion.watermark.base as base_mod
        from PIL import Image

        class DummyConfigObj:
            def __init__(self):
                self.pipe = object()
                self.num_frames = 4  

        class DummyWM(base_mod.BaseWatermark):
            def get_data_for_visualize(self, media, *args, **kwargs):
                return None

        
        monkeypatch.setattr(base_mod, "get_pipeline_type", lambda _: base_mod.PIPELINE_TYPE_TEXT_TO_VIDEO)
        
        monkeypatch.setattr(base_mod, "is_image_pipeline", lambda _: False)
        
        monkeypatch.setattr(base_mod, "is_video_pipeline", lambda _: True)

        
        monkeypatch.setattr(base_mod, "cv2_to_pil", lambda arr: Image.fromarray(arr.astype("uint8")))

        wm = DummyWM(DummyConfigObj())
        frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]
        out = wm._preprocess_media_for_detection(frames)
        assert isinstance(out, list)
        assert all(isinstance(f, Image.Image) for f in out)

    def test_preprocess_media_video_pipeline_list_mixed_types_raises(self, monkeypatch):
        import pytest
        import numpy as np
        import markdiffusion.watermark.base as base_mod
        from PIL import Image

        class DummyConfigObj:
            def __init__(self):
                self.pipe = object()
                self.num_frames = 4

        class DummyWM(base_mod.BaseWatermark):
            def get_data_for_visualize(self, media, *args, **kwargs):
                return None

        
        monkeypatch.setattr(base_mod, "get_pipeline_type", lambda _: base_mod.PIPELINE_TYPE_TEXT_TO_VIDEO)
        
        monkeypatch.setattr(base_mod, "is_image_pipeline", lambda _: False)
        
        monkeypatch.setattr(base_mod, "is_video_pipeline", lambda _: True)

        wm = DummyWM(DummyConfigObj())
        frames = [Image.new("RGB", (8, 8)), np.zeros((8, 8, 3), dtype=np.uint8)]
        with pytest.raises(ValueError, match="All frames must be either PIL images or numpy arrays"):
            wm._preprocess_media_for_detection(frames)

    def test_preprocess_media_video_pipeline_numpy_wrong_shape_raises(self, monkeypatch):
        import pytest
        import numpy as np
        import markdiffusion.watermark.base as base_mod

        class DummyConfigObj:
            def __init__(self):
                self.pipe = object()
                self.num_frames = 4

        class DummyWM(base_mod.BaseWatermark):
            def get_data_for_visualize(self, media, *args, **kwargs):
                return None

        
        monkeypatch.setattr(base_mod, "get_pipeline_type", lambda _: base_mod.PIPELINE_TYPE_TEXT_TO_VIDEO)
        
        monkeypatch.setattr(base_mod, "is_image_pipeline", lambda _: False)
        
        monkeypatch.setattr(base_mod, "is_video_pipeline", lambda _: True)

        wm = DummyWM(DummyConfigObj())
        bad = np.zeros((8, 8, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unsupported numpy array shape for video"):
            wm._preprocess_media_for_detection(bad)

    def test_preprocess_media_video_pipeline_tensor_unsupported_shape_raises(self, monkeypatch):
        import pytest
        import torch
        import markdiffusion.watermark.base as base_mod

        class DummyConfigObj:
            def __init__(self):
                self.pipe = object()
                self.num_frames = 4

        class DummyWM(base_mod.BaseWatermark):
            def get_data_for_visualize(self, media, *args, **kwargs):
                return None

        
        monkeypatch.setattr(base_mod, "get_pipeline_type", lambda _: base_mod.PIPELINE_TYPE_TEXT_TO_VIDEO)
        
        monkeypatch.setattr(base_mod, "is_image_pipeline", lambda _: False)
        
        monkeypatch.setattr(base_mod, "is_video_pipeline", lambda _: True)

        wm = DummyWM(DummyConfigObj())
        bad = torch.zeros(2, 3, 4)  
        with pytest.raises(ValueError, match="Unsupported tensor shape for video"):
            wm._preprocess_media_for_detection(bad)

    def test_generate_watermarked_image_raises_when_pipeline_not_image(self, monkeypatch):
        import pytest
        import markdiffusion.watermark.base as base_mod

        class DummyConfigObj:
            def __init__(self):
                self.pipe = object()
                self.num_frames = 4

        class DummyWM(base_mod.BaseWatermark):
            def get_data_for_visualize(self, media, *args, **kwargs):
                return None

        
        monkeypatch.setattr(base_mod, "get_pipeline_type", lambda _: base_mod.PIPELINE_TYPE_TEXT_TO_VIDEO)
        
        monkeypatch.setattr(base_mod, "is_image_pipeline", lambda _: False)
        
        monkeypatch.setattr(base_mod, "is_video_pipeline", lambda _: True)

        wm = DummyWM(DummyConfigObj())
        with pytest.raises(ValueError, match="does not support image generation"):
            wm._generate_watermarked_image("prompt")

    def test_generate_watermarked_video_raises_when_pipeline_not_video(self, monkeypatch):
        import pytest
        import markdiffusion.watermark.base as base_mod

        class DummyConfigObj:
            def __init__(self):
                self.pipe = object()
                self.num_frames = -1

        class DummyWM(base_mod.BaseWatermark):
            def get_data_for_visualize(self, media, *args, **kwargs):
                return None

        
        monkeypatch.setattr(base_mod, "get_pipeline_type", lambda _: base_mod.PIPELINE_TYPE_IMAGE)
        
        monkeypatch.setattr(base_mod, "is_image_pipeline", lambda _: True)
        
        monkeypatch.setattr(base_mod, "is_video_pipeline", lambda _: False)

        wm = DummyWM(DummyConfigObj())
        with pytest.raises(ValueError, match="does not support video generation"):
            wm._generate_watermarked_video("prompt")

    def test_generate_unwatermarked_image_raises_when_not_image_pipeline(self, monkeypatch):
        import pytest
        import markdiffusion.watermark.base as base_mod

        class DummyConfigObj:
            def __init__(self):
                self.pipe = object()
                self.num_frames = 4

        class DummyWM(base_mod.BaseWatermark):
            def get_data_for_visualize(self, media, *args, **kwargs):
                return None

        
        monkeypatch.setattr(base_mod, "get_pipeline_type", lambda _: base_mod.PIPELINE_TYPE_TEXT_TO_VIDEO)
        
        monkeypatch.setattr(base_mod, "is_image_pipeline", lambda _: False)
        
        monkeypatch.setattr(base_mod, "is_video_pipeline", lambda _: True)

        wm = DummyWM(DummyConfigObj())
        with pytest.raises(ValueError, match="does not support image generation"):
            wm._generate_unwatermarked_image("prompt")

    def test_generate_unwatermarked_video_raises_when_not_video_pipeline(self, monkeypatch):
        import pytest
        import markdiffusion.watermark.base as base_mod

        class DummyConfigObj:
            def __init__(self):
                self.pipe = object()
                self.num_frames = -1

        class DummyWM(base_mod.BaseWatermark):
            def get_data_for_visualize(self, media, *args, **kwargs):
                return None

        
        monkeypatch.setattr(base_mod, "get_pipeline_type", lambda _: base_mod.PIPELINE_TYPE_IMAGE)
        
        monkeypatch.setattr(base_mod, "is_image_pipeline", lambda _: True)
        
        monkeypatch.setattr(base_mod, "is_video_pipeline", lambda _: False)

        wm = DummyWM(DummyConfigObj())
        with pytest.raises(ValueError, match="does not support video generation"):
            wm._generate_unwatermarked_video("prompt")

    def test_generate_unwatermarked_video_t2v_requires_prompt_string(self, monkeypatch):
        import pytest
        import markdiffusion.watermark.base as base_mod
        from PIL import Image

        class DummyConfigObj:
            def __init__(self):
                self.pipe = object()
                self.num_frames = 4
                self.image_size = (64, 64)
                self.num_inference_steps = 1
                self.guidance_scale = 1.0
                self.init_latents = "LATENTS"
                self.gen_seed = 0
                self.gen_kwargs = {}

        class DummyWM(base_mod.BaseWatermark):
            def get_data_for_visualize(self, media, *args, **kwargs):
                return None

        
        monkeypatch.setattr(base_mod, "get_pipeline_type", lambda _: base_mod.PIPELINE_TYPE_TEXT_TO_VIDEO)
        
        monkeypatch.setattr(base_mod, "is_image_pipeline", lambda _: False)
        
        monkeypatch.setattr(base_mod, "is_video_pipeline", lambda _: True)
        
        monkeypatch.setattr(base_mod, "is_t2v_pipeline", lambda _: True)
        
        monkeypatch.setattr(base_mod, "is_i2v_pipeline", lambda _: False)

        wm = DummyWM(DummyConfigObj())
        with pytest.raises(ValueError, match="requires a text prompt"):
            wm._generate_unwatermarked_video(Image.new("RGB", (8, 8)))

    def test_generate_unwatermarked_video_t2v_output_videos_branch(self, monkeypatch):
        import numpy as np
        import markdiffusion.watermark.base as base_mod
        from PIL import Image

        class DummyPipe:
            def __call__(self, prompt, **kwargs):
                class Out:
                    def __init__(self):
                        # output.videos[0] -> iterable frames
                        self.videos = [np.zeros((3, 8, 8, 3), dtype=np.uint8)]
                return Out()

        class DummyConfigObj:
            def __init__(self):
                self.pipe = DummyPipe()
                self.num_frames = 3
                self.image_size = (8, 8)
                self.num_inference_steps = 1
                self.guidance_scale = 1.0
                self.init_latents = "LATENTS"
                self.gen_seed = 0
                self.gen_kwargs = {}

        class DummyWM(base_mod.BaseWatermark):
            def get_data_for_visualize(self, media, *args, **kwargs):
                return None

        
        monkeypatch.setattr(base_mod, "get_pipeline_type", lambda _: base_mod.PIPELINE_TYPE_TEXT_TO_VIDEO)
        
        monkeypatch.setattr(base_mod, "is_video_pipeline", lambda _: True)
        
        monkeypatch.setattr(base_mod, "is_image_pipeline", lambda _: False)
        
        monkeypatch.setattr(base_mod, "is_t2v_pipeline", lambda _: True)
        
        monkeypatch.setattr(base_mod, "is_i2v_pipeline", lambda _: False)
        
        monkeypatch.setattr(base_mod, "set_random_seed", lambda _: None)
        
        monkeypatch.setattr(base_mod, "cv2_to_pil", lambda arr: Image.fromarray(arr.astype("uint8")))

        wm = DummyWM(DummyConfigObj())
        frames = wm._generate_unwatermarked_video("hello")
        assert isinstance(frames, list)
        assert len(frames) == 3
        assert all(isinstance(f, Image.Image) for f in frames)

    def test_generate_unwatermarked_video_i2v_requires_input_image(self, monkeypatch):
        import pytest
        import markdiffusion.watermark.base as base_mod

        class DummyConfigObj:
            def __init__(self):
                self.pipe = object()
                self.num_frames = 4
                self.image_size = (64, 64)
                self.num_inference_steps = 1
                self.guidance_scale = 1.0
                self.init_latents = "LATENTS"
                self.gen_seed = 0
                self.gen_kwargs = {}

        class DummyWM(base_mod.BaseWatermark):
            def get_data_for_visualize(self, media, *args, **kwargs):
                return None

        monkeypatch.setattr(base_mod, "get_pipeline_type", lambda _: base_mod.PIPELINE_TYPE_IMAGE_TO_VIDEO)
        
        monkeypatch.setattr(base_mod, "is_video_pipeline", lambda _: True)
        
        monkeypatch.setattr(base_mod, "is_image_pipeline", lambda _: False)
        
        monkeypatch.setattr(base_mod, "is_t2v_pipeline", lambda _: False)
        
        monkeypatch.setattr(base_mod, "is_i2v_pipeline", lambda _: True)

        wm = DummyWM(DummyConfigObj())
        with pytest.raises(ValueError, match="Input image is required"):
            wm._generate_unwatermarked_video("not_a_path_and_no_image")

    def test_generate_unwatermarked_video_i2v_string_path_open_fail_raises(self, tmp_path, monkeypatch):
        import pytest
        import markdiffusion.watermark.base as base_mod

        bad_file = tmp_path / "bad.img"
        bad_file.write_text("not an image")

        class DummyConfigObj:
            def __init__(self):
                self.pipe = object()
                self.num_frames = 4
                self.image_size = (64, 64)
                self.num_inference_steps = 1
                self.guidance_scale = 1.0
                self.init_latents = "LATENTS"
                self.gen_seed = 0
                self.gen_kwargs = {}

        class DummyWM(base_mod.BaseWatermark):
            def get_data_for_visualize(self, media, *args, **kwargs):
                return None

        monkeypatch.setattr(base_mod, "get_pipeline_type", lambda _: base_mod.PIPELINE_TYPE_IMAGE_TO_VIDEO)
        
        monkeypatch.setattr(base_mod, "is_video_pipeline", lambda _: True)
        
        monkeypatch.setattr(base_mod, "is_image_pipeline", lambda _: False)
        
        monkeypatch.setattr(base_mod, "is_t2v_pipeline", lambda _: False)
        
        monkeypatch.setattr(base_mod, "is_i2v_pipeline", lambda _: True)

        wm = DummyWM(DummyConfigObj())
        with pytest.raises(ValueError, match="Failed to load image from path"):
            wm._generate_unwatermarked_video(str(bad_file))

    def test_generate_unwatermarked_video_t2v_gen_kwargs_merge_adds_missing_key(self, monkeypatch):
        import numpy as np
        import markdiffusion.watermark.base as base_mod
        from PIL import Image

        captured = {}

        class DummyPipe:
            def __call__(self, prompt, **kwargs):
                captured.update(kwargs)
                class Out:
                    def __init__(self):
                        self.frames = [np.zeros((2, 8, 8, 3), dtype=np.uint8)]
                return Out()

        class DummyConfigObj:
            def __init__(self):
                self.pipe = DummyPipe()
                self.num_frames = 2
                self.image_size = (8, 8)
                self.num_inference_steps = 1
                self.guidance_scale = 1.0
                self.init_latents = "LATENTS"
                self.gen_seed = 0
                self.gen_kwargs = {"foo": "bar"}  

        class DummyWM(base_mod.BaseWatermark):
            def get_data_for_visualize(self, media, *args, **kwargs):
                return None

        
        monkeypatch.setattr(base_mod, "get_pipeline_type", lambda _: base_mod.PIPELINE_TYPE_TEXT_TO_VIDEO)
        
        monkeypatch.setattr(base_mod, "is_video_pipeline", lambda _: True)
        
        monkeypatch.setattr(base_mod, "is_image_pipeline", lambda _: False)
        
        monkeypatch.setattr(base_mod, "is_t2v_pipeline", lambda _: True)
        
        monkeypatch.setattr(base_mod, "is_i2v_pipeline", lambda _: False)
        
        monkeypatch.setattr(base_mod, "set_random_seed", lambda _: None)
        
        monkeypatch.setattr(base_mod, "cv2_to_pil", lambda arr: Image.fromarray(arr.astype("uint8")))

        wm = DummyWM(DummyConfigObj())
        wm._generate_unwatermarked_video("hi")
        assert captured.get("foo") == "bar"
    
    # auto visualization
    def test_autovisualizer_init_raises_environment_error(self):
        import pytest
        from markdiffusion.visualize.auto_visualization import AutoVisualizer

        with pytest.raises(EnvironmentError, match="AutoVisualizer is designed"):
            AutoVisualizer()

    def test_get_visualization_class_name_returns_none_for_unknown_algorithm(self):
        from markdiffusion.visualize.auto_visualization import AutoVisualizer

        assert AutoVisualizer._get_visualization_class_name("UNKNOWN_ALG") is None

    def test_autovisualizer_load_raises_on_algorithm_name_mismatch(self):
        import pytest
        from markdiffusion.visualize.auto_visualization import AutoVisualizer

        class DummyData:
            def __init__(self):
                self.algorithm_name = "TR"

        with pytest.raises(ValueError, match="Algorithm name mismatch"):
            AutoVisualizer.load("GS", data_for_visualization=DummyData())

    def test_autovisualizer_load_raises_on_invalid_algorithm_name(self):
        import pytest
        from markdiffusion.visualize.auto_visualization import AutoVisualizer

        class DummyData:
            def __init__(self):
                self.algorithm_name = "NotInMapping"

        with pytest.raises(ValueError, match="Invalid algorithm name"):
            AutoVisualizer.load("NotInMapping", data_for_visualization=DummyData())

    def test_autovisualizer_load_raises_importerror_when_module_not_found(self):
        import pytest
        import markdiffusion.visualize.auto_visualization as av

        class DummyData:
            def __init__(self):
                self.algorithm_name = "TR"

        original = av.VISUALIZATION_DATA_MAPPING
        av.VISUALIZATION_DATA_MAPPING = {"TR": "nonexistent_module_abc.NonexistentClass"}
        try:
            with pytest.raises(ImportError, match="Failed to load visualization data class"):
                av.AutoVisualizer.load("TR", data_for_visualization=DummyData())
        finally:
            av.VISUALIZATION_DATA_MAPPING = original

    def test_autovisualizer_load_raises_importerror_when_class_missing_in_module(self):
        import pytest
        import markdiffusion.visualize.auto_visualization as av

        class DummyData:
            def __init__(self):
                self.algorithm_name = "TR"

        class DummyModule:
            pass

        original_import = av.importlib.import_module
        original_map = av.VISUALIZATION_DATA_MAPPING

        av.VISUALIZATION_DATA_MAPPING = {"TR": "dummy_mod.MissingClass"}

        def fake_import_module(name):
            assert name == "dummy_mod"
            return DummyModule()

        av.importlib.import_module = fake_import_module
        try:
            with pytest.raises(ImportError, match="Failed to load visualization data class"):
                av.AutoVisualizer.load("TR", data_for_visualization=DummyData())
        finally:
            av.importlib.import_module = original_import
            av.VISUALIZATION_DATA_MAPPING = original_map

    # base inversion
    def test_prepare_latent_video_unet_do_cfg_true(self):
        import os
        import importlib.util
        import torch

        # Load inversions/base_inversion.py without importing inversions/__init__.py
        base_path = os.path.join(os.getcwd(), "inversions", "base_inversion.py")
        spec = importlib.util.spec_from_file_location("base_inversion_direct", base_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        BaseInversion = mod.BaseInversion

        class VideoUNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv3d = torch.nn.Conv3d(1, 1, kernel_size=1)

        inv = BaseInversion(scheduler=None, unet=None, device="cpu")
        unet = VideoUNet()
        latents = torch.zeros(2, 3, 1, 4, 4)  # [B,F,C,H,W]
        latent_in, info = inv._prepare_latent_for_unet(latents, do_cfg=True, unet=unet)

        assert latent_in.shape == (4, 1, 3, 4, 4)  # [2B,C,F,H,W]
        assert info["is_video_unet"] is True
        assert info["do_cfg"] is True

    def test_restore_latent_image_unet_video_output_reshape_branch(self):
        import os
        import importlib.util
        import torch

        base_path = os.path.join(os.getcwd(), "inversions", "base_inversion.py")
        spec = importlib.util.spec_from_file_location("base_inversion_direct", base_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        BaseInversion = mod.BaseInversion

        inv = BaseInversion(scheduler=None, unet=None, device="cpu")

        info = {"do_cfg": False, "is_video_unet": False, "shp": (2, 3, 4, 8, 8)}
        noise_pred = torch.zeros(2 * 3, 4, 8, 8)  # [B*F,C,H,W]
        out = inv._restore_latent_from_unet(noise_pred, info=info, guidance_scale=7.5)

        assert out.shape == (2, 3, 4, 8, 8)

    def test_restore_latent_video_unet_do_cfg_true_permutes_and_merges_cfg(self):
        import os
        import importlib.util
        import torch

        base_path = os.path.join(os.getcwd(), "inversions", "base_inversion.py")
        spec = importlib.util.spec_from_file_location("base_inversion_direct", base_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        BaseInversion = mod.BaseInversion

        inv = BaseInversion(scheduler=None, unet=None, device="cpu")

        info = {"do_cfg": True, "is_video_unet": True, "shp": (2, 3, 4, 8, 8)}

        cond = torch.ones(2, 4, 3, 8, 8)
        uncond = torch.zeros(2, 4, 3, 8, 8)
        noise_pred = torch.cat([cond, uncond], dim=0)  # [2B,C,F,H,W]

        out = inv._restore_latent_from_unet(noise_pred, info=info, guidance_scale=2.0)

        assert out.shape == (2, 3, 4, 8, 8)  # [B,F,C,H,W]
        assert float(out.mean()) == 2.0

    def test_prepare_latent_assert_raises_for_invalid_ndim(self):
        import os
        import importlib.util
        import pytest
        import torch

        base_path = os.path.join(os.getcwd(), "inversions", "base_inversion.py")
        spec = importlib.util.spec_from_file_location("base_inversion_direct", base_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        BaseInversion = mod.BaseInversion

        class ImageUNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv2d = torch.nn.Conv2d(1, 1, kernel_size=1)

        inv = BaseInversion(scheduler=None, unet=None, device="cpu")
        unet = ImageUNet()

        latents = torch.zeros(1, 2, 3)  # ndim=3 invalid
        with pytest.raises(AssertionError, match="Video input must be 4D or 5D latent"):
            inv._prepare_latent_for_unet(latents, do_cfg=False, unet=unet)

    # videoshield
    def _load_videoshield_module(self, monkeypatch):
        import os
        import sys
        import types
        import importlib.util

        # Ensure package context for relative imports in watermark/videoshield/video_shield.py
        if "watermark" not in sys.modules:
            pkg = types.ModuleType("watermark")
            pkg.__path__ = [os.path.join(os.getcwd(), "watermark")]
            monkeypatch.setitem(sys.modules, "watermark", pkg)

        if "markdiffusion.watermark.videoshield" not in sys.modules:
            subpkg = types.ModuleType("markdiffusion.watermark.videoshield")
            subpkg.__path__ = [os.path.join(os.getcwd(), "watermark", "videoshield")]
            monkeypatch.setitem(sys.modules, "markdiffusion.watermark.videoshield", subpkg)

        # Load as real module name so "..base" works
        mod_name = "markdiffusion.watermark.videoshield.video_shield"
        path = os.path.join(os.getcwd(), "watermark", "videoshield", "video_shield.py")
        spec = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, mod_name, mod)
        spec.loader.exec_module(mod)
        return mod

    def test_videoshield_config_adjusts_kf_kh_kw_and_image_fallback_watermark(self, monkeypatch):
        import torch

        mod = self._load_videoshield_module(monkeypatch)

        class DummyConfigObj:
            pass

        cfg = DummyConfigObj()
        cfg.config_dict = {
            "k_f": 10,
            "k_c": 1,
            "k_h": 10,
            "k_w": 10,
            "t_temp": 0.5,
            "hstr_levels": 1,
            "t_wm": [0.1],
            "t_orig": [0.1],
        }
        cfg.image_size = (16, 16)  # latents 2x2
        cfg.device = "cpu"

        # 1) video path: trigger k_f adjust (k_f -> num_frames)
        cfg.num_frames = 4
        mod.VideoShieldConfig.initialize_parameters(cfg)
        assert cfg.k_f == 4

        # 2) image fallback path: force k_h=k_w=1 so watermark keeps full 2x2
        cfg.config_dict["k_h"] = 1
        cfg.config_dict["k_w"] = 1
        delattr(cfg, "num_frames")
        mod.VideoShieldConfig.initialize_parameters(cfg)

        assert tuple(cfg.watermark.shape) == (1, 4, 2, 2)
        assert isinstance(cfg.watermark, torch.Tensor)

    def test_videoshield_utils_image_case_latentlength_and_vote_threshold_1(self, monkeypatch):
        mod = self._load_videoshield_module(monkeypatch)

        class Cfg:
            pass

        cfg = Cfg()
        cfg.num_frames = 0
        cfg.latents_height = 2
        cfg.latents_width = 3
        cfg.device = "cpu"
        cfg.k_f = 1
        cfg.k_c = 1
        cfg.k_h = 1
        cfg.k_w = 1
        cfg.chacha_key_seed = 1
        cfg.chacha_nonce_seed = 2

        utils = mod.VideoShieldUtils(cfg)
        assert utils.latentlength == 4 * 2 * 3
        assert utils.vote_threshold == 1

    def test_videoshield_utils_stream_key_encrypt_exception_path(self, monkeypatch):
        import numpy as np
        import pytest

        mod = self._load_videoshield_module(monkeypatch)

        class Cfg:
            pass

        cfg = Cfg()
        cfg.num_frames = 0
        cfg.latents_height = 2
        cfg.latents_width = 2
        cfg.device = "cpu"
        cfg.k_f = 1
        cfg.k_c = 1
        cfg.k_h = 1
        cfg.k_w = 1
        cfg.chacha_key_seed = 1
        cfg.chacha_nonce_seed = 2

        utils = mod.VideoShieldUtils(cfg)

        class BadChaCha:
            @staticmethod
            def new(*args, **kwargs):
                raise RuntimeError("boom")

        mod.ChaCha20 = BadChaCha

        with pytest.raises(RuntimeError, match="Encryption failed"):
            utils._stream_key_encrypt(np.array([0, 1, 0, 1], dtype=np.uint8))

    def test_videoshield_truncated_sampling_image_reshape_branch(self, monkeypatch):
        import numpy as np
        import torch

        mod = self._load_videoshield_module(monkeypatch)

        class Cfg:
            pass

        cfg = Cfg()
        cfg.num_frames = 0
        cfg.latents_height = 2
        cfg.latents_width = 2
        cfg.device = "cpu"
        cfg.k_f = 1
        cfg.k_c = 1
        cfg.k_h = 1
        cfg.k_w = 1
        cfg.chacha_key_seed = 1
        cfg.chacha_nonce_seed = 2

        utils = mod.VideoShieldUtils(cfg)
        msg = np.zeros(utils.latentlength, dtype=np.uint8)
        z = utils._truncated_sampling(msg)

        assert isinstance(z, torch.Tensor)
        assert tuple(z.shape) == (1, 4, 2, 2)

    def test_videoshield_create_watermark_and_return_w_image_repeat_branch(self, monkeypatch):
        import torch

        mod = self._load_videoshield_module(monkeypatch)

        class Cfg:
            pass

        cfg = Cfg()
        cfg.num_frames = 0
        cfg.latents_height = 2
        cfg.latents_width = 2
        cfg.device = "cpu"
        cfg.k_f = 1
        cfg.k_c = 2
        cfg.k_h = 1
        cfg.k_w = 1
        cfg.chacha_key_seed = 1
        cfg.chacha_nonce_seed = 2

        # watermark shape for image fallback branch: [1, 4/k_c, H, W]
        cfg.watermark = torch.randint(0, 2, (1, 2, 2, 2), device="cpu")

        utils = mod.VideoShieldUtils(cfg)
        monkeypatch.setattr(utils, "_stream_key_encrypt", lambda sd: sd)
        monkeypatch.setattr(utils, "_truncated_sampling", lambda m: torch.ones(1, 4, cfg.latents_height, cfg.latents_width))

        w = utils.create_watermark_and_return_w()
        assert tuple(w.shape) == (1, 4, 2, 2)

    def test_videoshield_generate_watermarked_video_raises_when_not_video_pipeline(self, monkeypatch):
        import pytest

        mod = self._load_videoshield_module(monkeypatch)

        class DummyConfig:
            def __init__(self):
                self.pipe = object()

        class DummyWM(mod.VideoShieldWatermark):
            def __init__(self, cfg):
                self.config = cfg

        monkeypatch.setattr(mod, "is_video_pipeline", lambda _: False)

        wm = DummyWM(DummyConfig())
        with pytest.raises(ValueError, match="does not support video generation"):
            wm._generate_watermarked_video("p")

    def test_videoshield_generate_watermarked_video_extract_videos_uint8_and_kwargs_filter(self, monkeypatch):
        import numpy as np
        import torch
        from PIL import Image

        mod = self._load_videoshield_module(monkeypatch)

        class DummyUNet:
            dtype = torch.float32

        class DummyPipe:
            def __init__(self):
                self.unet = DummyUNet()

            def __call__(self, prompt, **kwargs):
                # ensure num_frames in kwargs is the processed one, and custom kw applied
                assert kwargs["num_frames"] == 1
                assert kwargs["extra"] == 123
                # "num_frames_should_not_override" should NOT be injected into generation_params
                assert "num_frames_should_not_override" in kwargs  # it is not special-cased, so it stays if passed
                class Out:
                    def __init__(self):
                        self.videos = [[np.zeros((8, 8, 3), dtype=np.uint8)]]
                return Out()

        class DummyConfig:
            def __init__(self):
                self.pipe = DummyPipe()
                self.gen_seed = 0
                self.num_frames = 1
                self.num_inference_steps = 1
                self.guidance_scale = 1.0
                self.image_size = (8, 8)
                self.gen_kwargs = {"extra": 123}
                self.device = "cpu"

        class DummyWM(mod.VideoShieldWatermark):
            def __init__(self, cfg):
                self.config = cfg

                class U:
                    def create_watermark_and_return_w(self_inner):
                        return torch.zeros(1, 4, 1, 1, 1)

                self.utils = U()
                self.set_orig_watermarked_latents = lambda x: None


        monkeypatch.setattr(mod, "is_video_pipeline", lambda _: True)
        
        monkeypatch.setattr(mod, "is_i2v_pipeline", lambda _: False)
        
        monkeypatch.setattr(mod, "set_random_seed", lambda _: None)

        wm = DummyWM(DummyConfig())
        frames = wm._generate_watermarked_video("p", num_frames=1, num_frames_should_not_override=999)

        assert isinstance(frames, list)
        assert len(frames) == 1
        assert isinstance(frames[0], Image.Image)

    def test_videoshield_generate_watermarked_video_tensor_frame_unexpected_shape_raises(self, monkeypatch):
        import pytest
        import torch

        mod = self._load_videoshield_module(monkeypatch)

        class DummyUNet:
            dtype = torch.float32

        class DummyPipe:
            def __init__(self):
                self.unet = DummyUNet()

            def __call__(self, prompt, **kwargs):
                class Out:
                    def __init__(self):
                        self.frames = [[torch.zeros(2, 2)]]  # triggers ValueError branch
                return Out()

        class DummyConfig:
            def __init__(self):
                self.pipe = DummyPipe()
                self.gen_seed = 0
                self.num_frames = 1
                self.num_inference_steps = 1
                self.guidance_scale = 1.0
                self.image_size = (8, 8)
                self.gen_kwargs = {}
                self.device = "cpu"

        class DummyWM(mod.VideoShieldWatermark):
            def __init__(self, cfg):
                self.config = cfg

                class U:
                    def create_watermark_and_return_w(self_inner):
                        return torch.zeros(1, 4, 1, 1, 1)

                self.utils = U()
                self.set_orig_watermarked_latents = lambda x: None

        monkeypatch.setattr(mod, "is_video_pipeline", lambda _: True)
        
        monkeypatch.setattr(mod, "is_i2v_pipeline", lambda _: False)
        
        monkeypatch.setattr(mod, "set_random_seed", lambda _: None)

        wm = DummyWM(DummyConfig())
        with pytest.raises(ValueError, match="Unexpected tensor shape for frame"):
            wm._generate_watermarked_video("p", num_frames=1)
    
    # visualize base
    def test_visualize_base_get_latent_data_video_frame_only_returns_chw(self):
        import torch
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from markdiffusion.visualize.base import BaseVisualizer

        class V(BaseVisualizer):
            pass

        class DummyData:
            pass

        data = DummyData()
        data.orig_watermarked_latents = torch.zeros(1, 4, 5, 2, 2)  # [B,C,F,H,W]
        data.reversed_latents = [torch.zeros(1, 4, 5, 2, 2)]
        data.image = torch.zeros(1, 3, 2, 2)

        v = V(data_for_visualization=data, is_video=True)
        lat = v._get_latent_data(data.orig_watermarked_latents, channel=None, frame=1)
        assert tuple(lat.shape) == (4, 2, 2)

    def test_visualize_base_get_latent_data_video_mid_frame_when_frame_none(self):
        import torch
        import matplotlib
        matplotlib.use("Agg")
        from markdiffusion.visualize.base import BaseVisualizer

        class V(BaseVisualizer):
            pass

        class DummyData:
            pass

        data = DummyData()
        data.orig_watermarked_latents = torch.zeros(1, 4, 5, 2, 2)  # mid frame = 2
        data.reversed_latents = [torch.zeros(1, 4, 5, 2, 2)]
        data.image = torch.zeros(1, 3, 2, 2)

        v = V(data_for_visualization=data, is_video=True)
        lat = v._get_latent_data(data.orig_watermarked_latents, channel=1, frame=None)
        assert tuple(lat.shape) == (2, 2)

    def test_visualize_base_get_latent_data_video_mid_frame_channel_none(self):
        import torch
        import matplotlib
        matplotlib.use("Agg")
        from markdiffusion.visualize.base import BaseVisualizer

        class V(BaseVisualizer):
            pass

        class DummyData:
            pass

        data = DummyData()
        data.orig_watermarked_latents = torch.zeros(1, 4, 5, 2, 2)  # mid frame = 2
        data.reversed_latents = [torch.zeros(1, 4, 5, 2, 2)]
        data.image = torch.zeros(1, 3, 2, 2)

        v = V(data_for_visualization=data, is_video=True)
        lat = v._get_latent_data(data.orig_watermarked_latents, channel=None, frame=None)
        assert tuple(lat.shape) == (4, 2, 2)

    def test_visualize_base_get_latent_data_image_channel_none_returns_chw(self):
        import torch
        import matplotlib
        matplotlib.use("Agg")
        from markdiffusion.visualize.base import BaseVisualizer

        class V(BaseVisualizer):
            pass

        class DummyData:
            pass

        data = DummyData()
        data.orig_watermarked_latents = torch.zeros(1, 4, 2, 2)  # [B,C,H,W]
        data.reversed_latents = [torch.zeros(1, 4, 2, 2)]
        data.image = torch.zeros(1, 3, 2, 2)

        v = V(data_for_visualization=data, is_video=False)
        lat = v._get_latent_data(data.orig_watermarked_latents, channel=None, frame=None)
        assert tuple(lat.shape) == (4, 2, 2)

    def test_visualize_base_draw_inverted_latents_step_not_none_branch(self):
        import torch
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from markdiffusion.visualize.base import BaseVisualizer

        class V(BaseVisualizer):
            pass

        class DummyData:
            pass

        data = DummyData()
        data.orig_watermarked_latents = torch.zeros(1, 4, 2, 2)
        data.reversed_latents = [
            torch.ones(1, 4, 2, 2) * 1,
            torch.ones(1, 4, 2, 2) * 2,
        ]
        data.image = torch.zeros(1, 3, 2, 2)

        v = V(data_for_visualization=data, is_video=False)
        fig, ax = plt.subplots()
        out_ax = v.draw_inverted_latents(channel=0, step=1, ax=ax)
        assert out_ax is ax

    def test_visualize_base_draw_inverted_latents_fft_single_channel_branch(self):
        import torch
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from markdiffusion.visualize.base import BaseVisualizer

        class V(BaseVisualizer):
            pass

        class DummyData:
            pass

        data = DummyData()
        data.orig_watermarked_latents = torch.zeros(1, 4, 2, 2)
        data.reversed_latents = [torch.ones(1, 4, 2, 2)]
        data.image = torch.zeros(1, 3, 2, 2)

        v = V(data_for_visualization=data, is_video=False)
        fig, ax = plt.subplots()
        out_ax = v.draw_inverted_latents_fft(channel=0, step=0, ax=ax)
        assert out_ax is ax

    def test_visualize_base_draw_diff_latents_fft_single_channel_branch(self):
        import torch
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from markdiffusion.visualize.base import BaseVisualizer

        class V(BaseVisualizer):
            pass

        class DummyData:
            pass

        data = DummyData()
        data.orig_watermarked_latents = torch.ones(1, 4, 2, 2) * 3
        data.reversed_latents = [torch.ones(1, 4, 2, 2) * 1]
        data.image = torch.zeros(1, 3, 2, 2)

        v = V(data_for_visualization=data, is_video=False)
        fig, ax = plt.subplots()
        out_ax = v.draw_diff_latents_fft(channel=0, ax=ax)
        assert out_ax is ax

    def test_visualize_base_draw_single_image_tensor_other_dim_branch_and_scale_255(self):
        import torch
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from markdiffusion.visualize.base import BaseVisualizer

        class V(BaseVisualizer):
            pass

        class DummyData:
            pass

        data = DummyData()
        data.orig_watermarked_latents = torch.zeros(1, 4, 2, 2)
        data.reversed_latents = [torch.zeros(1, 4, 2, 2)]
        # dim not 3/4 -> hits t532 branch; max>1 -> hits t536 scale
        data.image = torch.ones(2, 2) * 255.0

        v = V(data_for_visualization=data, is_video=False)
        fig, ax = plt.subplots()
        out_ax = v._draw_single_image(ax=ax, title="")
        assert out_ax is ax

    def test_visualize_base_draw_video_frames_numpy_method_and_exception_branch(self):
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from markdiffusion.visualize.base import BaseVisualizer

        class V(BaseVisualizer):
            pass

        class DummyData:
            pass

        class HasNumpy:
            def __init__(self):
                self.called = False

            def numpy(self):
                self.called = True
                # shape that will later break imshow -> triggers except printing
                return np.zeros((1, 2, 2, 2), dtype=np.float32)

        data = DummyData()
        data.orig_watermarked_latents = np.zeros((1, 4, 2, 2), dtype=np.float32)
        data.reversed_latents = [np.zeros((1, 4, 2, 2), dtype=np.float32)]
        data.image = np.zeros((2, 2, 3), dtype=np.uint8)
        data.video_frames = [HasNumpy(), np.zeros((2, 2, 3), dtype=np.float64)]

        v = V(data_for_visualization=data, is_video=True)
        fig, ax = plt.subplots()

        out_ax = v._draw_video_frames(ax=ax, num_frames=2, title="")
        assert out_ax is ax

    def test_visualize_base_visualize_typeerror_wrapped_to_valueerror(self):
        import pytest
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from markdiffusion.visualize.base import BaseVisualizer

        class V(BaseVisualizer):
            def m(self, ax=None):
                return ax

        class DummyData:
            pass

        data = DummyData()
        data.orig_watermarked_latents = None
        data.reversed_latents = [None]
        data.image = None

        v = V(data_for_visualization=data, is_video=False)

        # Pass unexpected kw to trigger TypeError in method call, then wrapper ValueError at t736-737
        with pytest.raises(ValueError, match="does not accept the given arguments"):
            v.visualize(rows=1, cols=1, methods=["m"], method_kwargs=[{"bad": 1}])

    # image quality analysis
    def test_silent_progress_bar_iter_and_description(self):
        from markdiffusion.evaluation.pipelines.image_quality_analysis import SilentProgressBar

        data = [1, 2, 3]
        bar = SilentProgressBar(data)

        assert list(iter(bar)) == data
        bar.set_description("desc")


    def test_init_invalid_unwatermarked_image_source_raises(self):
        from markdiffusion.evaluation.pipelines.image_quality_analysis import ImageQualityAnalysisPipeline
        from markdiffusion.evaluation.dataset import BaseDataset

        class DummyDataset(BaseDataset):
            name = "dummy"
            num_samples = 1
            num_references = 0

            def get_prompt(self, index: int) -> str:
                return "p"

            def get_reference(self, index: int):
                raise AssertionError("should not be called")

        try:
            ImageQualityAnalysisPipeline(
                dataset=DummyDataset(),
                unwatermarked_image_source="invalid",
            )
            assert False, "Expected ValueError"
        except ValueError:
            pass


    def test_get_progress_bar_returns_silent_when_disabled(self):
        from markdiffusion.evaluation.pipelines.image_quality_analysis import DirectImageQualityAnalysisPipeline

        class DummyDataset:
            name = "dummy"
            num_samples = 2
            num_references = 0

            def get_prompt(self, index: int) -> str:
                return f"p{index}"

        pipe = DirectImageQualityAnalysisPipeline(
            dataset=DummyDataset(),
            analyzers=[],
            show_progress=False,
        )

        bar = pipe._get_progress_bar([1, 2, 3])
        assert bar.__class__.__name__ == "SilentProgressBar"
        assert list(bar) == [1, 2, 3]
        bar.set_description("x")


    def test_get_unwatermarked_image_natural_source_uses_reference(self):
        from PIL import Image
        from markdiffusion.evaluation.pipelines.image_quality_analysis import DirectImageQualityAnalysisPipeline

        class DummyDataset:
            name = "dummy"
            num_samples = 1
            num_references = 1

            def get_prompt(self, index: int) -> str:
                return "p"

            def get_reference(self, index: int):
                return Image.new("RGB", (8, 8), color=(10, 20, 30))

        class DummyWatermark:
            def generate_unwatermarked_media(self, input_data, **kwargs):
                raise AssertionError("should not be called for natural source")

        pipe = DirectImageQualityAnalysisPipeline(
            dataset=DummyDataset(),
            analyzers=[],
            unwatermarked_image_source="natural",
            show_progress=False,
        )

        img = pipe._get_unwatermarked_image(DummyWatermark(), 0)
        assert img.size == (8, 8)


    def test_edit_watermarked_image_list_path_applies_editors(self):
        from PIL import Image
        from markdiffusion.evaluation.pipelines.image_quality_analysis import DirectImageQualityAnalysisPipeline

        class DummyDataset:
            name = "dummy"
            num_samples = 1
            num_references = 0

            def get_prompt(self, index: int) -> str:
                return "p"

        class AddOneEditor:
            def edit(self, img):
                px = img.getpixel((0, 0))
                img2 = img.copy()
                img2.putpixel((0, 0), (px[0] + 1, px[1] + 1, px[2] + 1))
                return img2

        pipe = DirectImageQualityAnalysisPipeline(
            dataset=DummyDataset(),
            analyzers=[],
            watermarked_image_editor_list=[AddOneEditor()],
            show_progress=False,
        )

        imgs = [
            Image.new("RGB", (2, 2), color=(0, 0, 0)),
            Image.new("RGB", (2, 2), color=(5, 5, 5)),
        ]
        out = pipe._edit_watermarked_image(imgs)

        assert isinstance(out, list)
        assert out[0].getpixel((0, 0)) == (1, 1, 1)
        assert out[1].getpixel((0, 0)) == (6, 6, 6)


    def test_edit_unwatermarked_image_list_path_applies_editors(self):
        from PIL import Image
        from markdiffusion.evaluation.pipelines.image_quality_analysis import DirectImageQualityAnalysisPipeline

        class DummyDataset:
            name = "dummy"
            num_samples = 1
            num_references = 0

            def get_prompt(self, index: int) -> str:
                return "p"

        class AddTwoEditor:
            def edit(self, img):
                px = img.getpixel((0, 0))
                img2 = img.copy()
                img2.putpixel((0, 0), (px[0] + 2, px[1] + 2, px[2] + 2))
                return img2

        pipe = DirectImageQualityAnalysisPipeline(
            dataset=DummyDataset(),
            analyzers=[],
            unwatermarked_image_editor_list=[AddTwoEditor()],
            show_progress=False,
        )

        imgs = [Image.new("RGB", (2, 2), color=(0, 0, 0))]
        out = pipe._edit_unwatermarked_image(imgs)

        assert isinstance(out, list)
        assert out[0].getpixel((0, 0)) == (2, 2, 2)


    def test_prepare_dataset_reference_none_when_no_references_and_natural_ref_source(self):
        from PIL import Image
        from markdiffusion.evaluation.pipelines.image_quality_analysis import ReferencedImageQualityAnalysisPipeline

        class DummyDataset:
            name = "dummy"
            num_samples = 1
            num_references = 0

            def get_prompt(self, index: int) -> str:
                return "p"

            def get_reference(self, index: int):
                raise AssertionError("no references")

        class DummyWatermark:
            def generate_watermarked_media(self, input_data, **kwargs):
                return Image.new("RGB", (4, 4), color=(1, 1, 1))

            def generate_unwatermarked_media(self, input_data, **kwargs):
                return Image.new("RGB", (4, 4), color=(2, 2, 2))

        pipe = ReferencedImageQualityAnalysisPipeline(
            dataset=DummyDataset(),
            analyzers=[],
            reference_image_source="natural",
            show_progress=False,
        )

        prepared = pipe._prepare_dataset(DummyWatermark())
        assert prepared.reference_images == [None]


    def test_prepare_dataset_reference_from_unwatermarked_when_reference_source_generated(self):
        from PIL import Image
        from markdiffusion.evaluation.pipelines.image_quality_analysis import ReferencedImageQualityAnalysisPipeline

        class DummyDataset:
            name = "dummy"
            num_samples = 1
            num_references = 0

            def get_prompt(self, index: int) -> str:
                return "p"

        class DummyWatermark:
            def generate_watermarked_media(self, input_data, **kwargs):
                return Image.new("RGB", (4, 4), color=(1, 1, 1))

            def generate_unwatermarked_media(self, input_data, **kwargs):
                return Image.new("RGB", (4, 4), color=(9, 9, 9))

        pipe = ReferencedImageQualityAnalysisPipeline(
            dataset=DummyDataset(),
            analyzers=[],
            reference_image_source="generated",
            show_progress=False,
        )

        prepared = pipe._prepare_dataset(DummyWatermark())
        assert prepared.reference_images[0].getpixel((0, 0)) == (9, 9, 9)

    # prc
    def test_prcutils_encode_message_when_message_is_none(self):
        import numpy as np
        import torch
        from markdiffusion.watermark.prc.prc import PRCUtils

        class DummyGF:
            def Random(self, *args, **kwargs):
                seed = kwargs.get("seed", 0)
                np.random.seed(seed)

                if "shape" in kwargs and kwargs["shape"] is not None:
                    shape = kwargs["shape"]
                    return np.random.randint(0, 2, size=shape, dtype=int)

                n = int(args[0])
                return np.random.randint(0, 2, size=n, dtype=int)

            def Zeros(self, n):
                return np.zeros(int(n), dtype=int)

            def __call__(self, x):
                return np.array(x, dtype=int)

        class DummyConfig:
            GF = DummyGF()
            fpr = 0.1
            t = 3
            n = 200  # ensure r>0 and csr_matrix can infer shape
            gen_matrix_seed = 1
            indice_seed = 2
            one_time_pad_seed = 3
            test_bits_seed = 4
            permute_bits_seed = 5
            payload_seed = 6
            error_seed = 7
            pseudogaussian_seed = 8
            message_length = 8

        cfg = DummyConfig()
        utils = PRCUtils(cfg)

        codeword = utils._encode_message(utils.encoding_key, message=None)
        assert isinstance(codeword, torch.Tensor)
        assert codeword.ndim == 1
        assert codeword.numel() == cfg.n


    def test_prc_generate_watermarked_image_includes_gen_kwargs_and_kwargs_override(self, monkeypatch):
        import types
        import torch
        from PIL import Image
        import markdiffusion.watermark.prc.prc as prc_mod
        from markdiffusion.watermark.prc.prc import PRC

        class DummyUtils:
            def __init__(self, config):
                self.config = config
                self.decoding_key = ("dk",)

            def inject_watermark(self):
                return torch.zeros(
                    (1, self.config.latents_channel, self.config.latents_height, self.config.latents_width),
                    dtype=torch.float32,
                )

        monkeypatch.setattr(prc_mod, "PRCUtils", DummyUtils)

        class DummyDetector:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        monkeypatch.setattr(prc_mod, "PRCDetector", lambda **kwargs: DummyDetector(**kwargs))

        class DummyPipe:
            vae_scale_factor = 8

            class DummyUnet:
                class DummyCfg:
                    in_channels = 1
                config = DummyCfg()

            unet = DummyUnet()

            def __call__(self, prompt, **kwargs):
                self.last_kwargs = kwargs
                return types.SimpleNamespace(images=[Image.new("RGB", (8, 8), color=(0, 0, 0))])

        class DummyGF:
            pass

        class DummyConfig:
            pipe = DummyPipe()
            device = "cpu"
            image_size = (8, 8)
            num_images = 1
            guidance_scale = 7.5
            num_inference_steps = 5
            gen_seed = 123

            var = 1.0
            threshold = 0.5
            GF = DummyGF()  # FIX

            latents_height = 1
            latents_width = 1
            latents_channel = 1

            gen_kwargs = {"foo": "bar", "height": 999}

        cfg = DummyConfig()
        prc = PRC(cfg)

        img = prc._generate_watermarked_image(
            "p",
            height=777,
            custom_param=42,
            latents="bad",
        )
        assert isinstance(img, Image.Image)

        passed = cfg.pipe.last_kwargs
        assert passed["foo"] == "bar"
        assert passed["height"] == 777
        assert passed["custom_param"] == 42
        assert passed["latents"] is not None
        assert passed["latents"] != "bad"


    def test_prc_detect_watermark_no_cfg_guidance_branch_and_detector_type(self, monkeypatch):
        import torch
        from PIL import Image
        import markdiffusion.watermark.prc.prc as prc_mod
        from markdiffusion.watermark.prc.prc import PRC

        class DummyUtils:
            def __init__(self, config):
                self.config = config
                self.decoding_key = ("dk",)

        monkeypatch.setattr(prc_mod, "PRCUtils", DummyUtils)

        class DummyDetector:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def eval_watermark(self, latents, detector_type=None):
                return {"score": 1.0, "detector_type": detector_type}

        monkeypatch.setattr(prc_mod, "PRCDetector", lambda **kwargs: DummyDetector(**kwargs))

        class DummyInversion:
            def forward_diffusion(self, latents, text_embeddings, guidance_scale, num_inference_steps, **kwargs):
                return [latents]

        class DummyPipe:
            vae_scale_factor = 8

            class DummyUnet:
                class DummyCfg:
                    in_channels = 1
                config = DummyCfg()

            unet = DummyUnet()

            def encode_prompt(self, prompt, device, do_classifier_free_guidance, num_images_per_prompt):
                emb = torch.zeros(1, 4)
                return emb, torch.ones(1, 4)

        class DummyGF:
            pass

        class DummyConfig:
            pipe = DummyPipe()
            inversion = DummyInversion()
            device = "cpu"
            image_size = (8, 8)
            guidance_scale = 1.0
            num_inference_steps = 5
            num_images = 1

            var = 1.0
            threshold = 0.5
            GF = DummyGF()  # FIX

        def fake_transform_to_model_format(img, target_size):
            return torch.zeros(3, target_size, target_size)

        def fake_get_media_latents(pipe, media, sample, decoder_inv=False):
            return torch.zeros(1, 1, 1, 1)

        monkeypatch.setattr(prc_mod, "transform_to_model_format", fake_transform_to_model_format)
        monkeypatch.setattr(prc_mod, "get_media_latents", fake_get_media_latents)

        prc = PRC(DummyConfig())

        res = prc._detect_watermark_in_image(
            Image.new("RGB", (8, 8)),
            prompt="p",
            detector_type="fast",
            extra_kw="x",
        )
        assert res["detector_type"] == "fast"


    def test_prc_get_data_for_visualize_hits_exception_paths(self, monkeypatch, capsys):
        import types
        import torch
        from PIL import Image
        import markdiffusion.watermark.prc.prc as prc_mod
        from markdiffusion.watermark.prc.prc import PRC

        class DummyUtils:
            def __init__(self, config):
                self.config = config
                self.encoding_key = (torch.eye(2).numpy(),)
                self.decoding_key = (None, None)

            def _encode_message(self, encoding_key, message):
                return torch.ones(2).numpy()

            def _sample_prc_codeword(self, prc_codeword):
                return torch.ones(2)

        monkeypatch.setattr(prc_mod, "PRCUtils", DummyUtils)

        class DummyDetector:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def eval_watermark(self, latents, detector_type=None):
                return {"score": 1.0}

        monkeypatch.setattr(prc_mod, "PRCDetector", lambda **kwargs: DummyDetector(**kwargs))

        class DummyInversion:
            def forward_diffusion(self, *args, **kwargs):
                raise RuntimeError("boom")

        class DummyPipe:
            vae_scale_factor = 8

            class DummyUnet:
                class DummyCfg:
                    in_channels = 1
                config = DummyCfg()

            unet = DummyUnet()

            def __call__(self, prompt, **kwargs):
                return types.SimpleNamespace(images=[Image.new("RGB", (8, 8), color=(0, 0, 0))])

            def encode_prompt(self, prompt, device, do_classifier_free_guidance, num_images_per_prompt):
                emb = torch.zeros(1, 4)
                return emb, torch.ones(1, 4)

        class DummyGF:
            pass

        class DummyConfig:
            pipe = DummyPipe()
            inversion = DummyInversion()
            device = "cpu"
            image_size = (8, 8)
            num_images = 1
            guidance_scale = 1.0
            num_inference_steps = 5
            gen_seed = 123

            var = 1.0
            threshold = 0.5
            fpr = 0.1
            t = 3
            GF = DummyGF()

            latents_channel = 1
            latents_height = 1
            latents_width = 2
            n = 2

            config_dict = {"message": "A"}

            algorithm_name = "PRC"

            # for PRC visualization fields
            message = [0, 1]
            message_length = len(message)

            def _str_to_binary_array(self, s: str):
                return [0, 1, 0, 0, 0, 0, 0, 1]

        def fake_transform_to_model_format(img, target_size):
            return torch.zeros(3, target_size, target_size)

        def fake_get_media_latents(pipe, media, sample, decoder_inv=False):
            return torch.zeros(1, 1, 1, 1)

        monkeypatch.setattr(prc_mod, "transform_to_model_format", fake_transform_to_model_format)
        monkeypatch.setattr(prc_mod, "get_media_latents", fake_get_media_latents)

        prc = PRC(DummyConfig())
        data = prc.get_data_for_visualize(Image.new("RGB", (8, 8)), prompt="p")

        out = capsys.readouterr().out
        assert "Warning: Could not perform inversion for visualization" in out
        assert data is not None
    
    # videomark detection
    def test_videomark_recover_posteriors_default_variance_and_basis_branch(self):
        import numpy as np
        import torch
        from scipy.sparse import csr_matrix
        from markdiffusion.detection.videomark.videomark_detection import VideoMarkDetector

        class DummyGF:
            def __call__(self, x):
                return np.asarray(x, dtype=int)

        gen = np.zeros((4, 2), dtype=int)
        pcm = csr_matrix(np.eye(1, 4, dtype=int))
        decoding_key = (gen, pcm, np.zeros(4, dtype=int), 0.1, 0.0, np.zeros(1, dtype=int), 0, 1, 1)

        det = VideoMarkDetector(
            message_sequence=np.zeros((1, 1), dtype=int),
            watermark=np.zeros((1, 1), dtype=int),
            num_frames=0,
            var=1,
            decoding_key=decoding_key,
            GF=DummyGF(),
            threshold=0.5,
            device=torch.device("cpu"),
        )

        z = torch.ones(4, dtype=torch.float64)

        out1 = det._recover_posteriors(z, basis=None, variances=None)
        assert torch.is_tensor(out1)

        basis = torch.eye(4, dtype=torch.float64)  # keep as torch tensor because code uses (z @ basis)
        out2 = det._recover_posteriors(z, basis=basis, variances=None)
        assert torch.is_tensor(out2)


    def test_videomark_recover_posteriors_tensor_variances_branch(self):
        import numpy as np
        import torch
        from scipy.sparse import csr_matrix
        from markdiffusion.detection.videomark.videomark_detection import VideoMarkDetector

        class DummyGF:
            def __call__(self, x):
                return np.asarray(x, dtype=int)

        gen = np.zeros((4, 2), dtype=int)
        pcm = csr_matrix(np.eye(1, 4, dtype=int))
        decoding_key = (gen, pcm, np.zeros(4, dtype=int), 0.1, 0.0, np.zeros(1, dtype=int), 0, 1, 1)

        det = VideoMarkDetector(
            message_sequence=np.zeros((1, 1), dtype=int),
            watermark=np.zeros((1, 1), dtype=int),
            num_frames=0,
            var=1,
            decoding_key=decoding_key,
            GF=DummyGF(),
            threshold=0.5,
            device=torch.device("cpu"),
        )

        z = torch.ones(4, dtype=torch.float64)
        variances = torch.ones_like(z) * 0.7  # tensor branch

        out = det._recover_posteriors(z, basis=None, variances=variances)
        assert torch.is_tensor(out)


    def test_videomark_align_posteriors_length_truncate_and_pad(self, caplog):
        import numpy as np
        import torch
        from scipy.sparse import csr_matrix
        from markdiffusion.detection.videomark.videomark_detection import VideoMarkDetector

        class DummyGF:
            def __call__(self, x):
                return np.asarray(x, dtype=int)

        gen = np.zeros((10, 2), dtype=int)  # code_length = 10
        pcm = csr_matrix(np.eye(1, 10, dtype=int))
        decoding_key = (gen, pcm, np.zeros(10, dtype=int), 0.1, 0.0, np.zeros(1, dtype=int), 0, 1, 1)

        det = VideoMarkDetector(
            message_sequence=np.zeros((1, 1), dtype=int),
            watermark=np.zeros((1, 1), dtype=int),
            num_frames=0,
            var=1,
            decoding_key=decoding_key,
            GF=DummyGF(),
            threshold=0.5,
            device=torch.device("cpu"),
        )

        # truncate branch (current_len > code_length)
        long_vec = torch.ones(12)
        out1 = det._align_posteriors_length(long_vec)
        assert out1.numel() == 10

        # pad branch (current_len < code_length)
        short_vec = torch.ones(7)
        out2 = det._align_posteriors_length(short_vec)
        assert out2.numel() == 10
        assert torch.all(out2[7:] == 0)


    def test_videomark_boolean_row_reduce_print_progress_noninvertible(self, capsys):
        import numpy as np
        import torch
        from scipy.sparse import csr_matrix
        from markdiffusion.detection.videomark.videomark_detection import VideoMarkDetector

        class DummyGF:
            def __call__(self, x):
                return np.asarray(x, dtype=int)

        gen = np.zeros((4, 2), dtype=int)
        pcm = csr_matrix(np.eye(1, 4, dtype=int))
        decoding_key = (gen, pcm, np.zeros(4, dtype=int), 0.1, 0.0, np.zeros(1, dtype=int), 0, 1, 1)

        det = VideoMarkDetector(
            message_sequence=np.zeros((1, 1), dtype=int),
            watermark=np.zeros((1, 1), dtype=int),
            num_frames=0,
            var=1,
            decoding_key=decoding_key,
            GF=DummyGF(),
            threshold=0.5,
            device=torch.device("cpu"),
        )

        # Non-invertible: first column all zeros -> idxs.size == 0 triggers print+return None
        A = np.zeros((5, 3), dtype=int)
        res = det._boolean_row_reduce(A, print_progress=True)
        assert res is None
        assert "not invertible" in capsys.readouterr().out.lower()


    def test_videomark_eval_watermark_invalid_detector_type_raises(self):
        import numpy as np
        import torch
        from scipy.sparse import csr_matrix
        from markdiffusion.detection.videomark.videomark_detection import VideoMarkDetector

        class DummyGF:
            def __call__(self, x):
                return np.asarray(x, dtype=int)

        gen = np.zeros((4, 2), dtype=int)
        pcm = csr_matrix(np.eye(1, 4, dtype=int))
        decoding_key = (gen, pcm, np.zeros(4, dtype=int), 0.1, 0.0, np.zeros(1, dtype=int), 0, 1, 1)

        det = VideoMarkDetector(
            message_sequence=np.zeros((1, 1), dtype=int),
            watermark=np.zeros((1, 1), dtype=int),
            num_frames=0,
            var=1,
            decoding_key=decoding_key,
            GF=DummyGF(),
            threshold=0.5,
            device=torch.device("cpu"),
        )

        try:
            det.eval_watermark(torch.zeros(1, 1, 2), detector_type="other")
            assert False, "Expected ValueError"
        except ValueError:
            pass


    def test_videomark_eval_watermark_dim_4_unsqueeze_and_wrong_dim_raises(self):
        import numpy as np
        import torch
        from scipy.sparse import csr_matrix
        from markdiffusion.detection.videomark.videomark_detection import VideoMarkDetector

        class DummyGF:
            def __call__(self, x):
                return np.asarray(x, dtype=int)

        gen = np.zeros((4, 2), dtype=int)
        pcm = csr_matrix(np.eye(1, 4, dtype=int))
        decoding_key = (gen, pcm, np.zeros(4, dtype=int), 0.1, 0.0, np.zeros(1, dtype=int), 0, 1, 1)

        det = VideoMarkDetector(
            message_sequence=np.zeros((1, 1), dtype=int),
            watermark=np.zeros((1, 1), dtype=int),
            num_frames=0,
            var=1,
            decoding_key=decoding_key,
            GF=DummyGF(),
            threshold=0.5,
            device=torch.device("cpu"),
        )

        # dim==4 -> unsqueeze branch, but then later might error due to insufficient frames (fine)
        latents4 = torch.zeros(1, 4, 2, 2)  # becomes (1,1,4,2,2)
        res = det.eval_watermark(latents4, detector_type="bit_acc")
        assert isinstance(res, dict)
        assert "bit_acc" in res

        # dim==2 -> early error (<3)
        try:
            det.eval_watermark(torch.zeros(2, 2), detector_type="bit_acc")
            assert False, "Expected ValueError"
        except ValueError:
            pass


    def test_videomark_eval_watermark_channel_axis_move_and_frame_mismatch_path(self, monkeypatch):
        import numpy as np
        import torch
        from scipy.sparse import csr_matrix
        from markdiffusion.detection.videomark.videomark_detection import VideoMarkDetector

        class DummyGF:
            def __call__(self, x):
                return np.asarray(x, dtype=int)

        gen = np.zeros((6, 2), dtype=int)  # code_length = 6
        pcm = csr_matrix(np.eye(1, 6, dtype=int))
        decoding_key = (gen, pcm, np.zeros(6, dtype=int), 0.1, 0.0, np.zeros(1, dtype=int), 0, 1, 1)

        det = VideoMarkDetector(
            message_sequence=np.zeros((1, 3), dtype=int),
            watermark=np.zeros((3, 3), dtype=int),
            num_frames=5,  # mismatch -> warning path
            var=1,
            decoding_key=decoding_key,
            GF=DummyGF(),
            threshold=0.5,
            device=torch.device("cpu"),
        )

        # Must return torch.Tensor because caller does .flatten().cpu()
        def fake_recover_posteriors(z, basis=None, variances=None):
            return torch.ones(6, dtype=torch.float64)

        def fake_detect_watermark(posteriors):
            return True

        def fake_decode_message(posteriors, print_progress=False, max_bp_iter=None):
            return np.array([1, 0, 1], dtype=int)

        monkeypatch.setattr(det, "_recover_posteriors", fake_recover_posteriors)
        monkeypatch.setattr(det, "_detect_watermark", fake_detect_watermark)
        monkeypatch.setattr(det, "_decode_message", fake_decode_message)

        # axis=3 == 4 triggers movedim(channel_axis,1)
        latents = torch.zeros(1, 3, 2, 4, 2)
        out = det.eval_watermark(latents, detector_type="bit_acc")
        assert isinstance(out, dict)
        assert "bit_acc" in out


    def test_videomark_eval_watermark_not_enough_frames_returns_empty(self):
        import numpy as np
        import torch
        from scipy.sparse import csr_matrix
        from markdiffusion.detection.videomark.videomark_detection import VideoMarkDetector

        class DummyGF:
            def __call__(self, x):
                return np.asarray(x, dtype=int)

        gen = np.zeros((4, 2), dtype=int)
        pcm = csr_matrix(np.eye(1, 4, dtype=int))
        decoding_key = (gen, pcm, np.zeros(4, dtype=int), 0.1, 0.0, np.zeros(1, dtype=int), 0, 1, 1)

        det = VideoMarkDetector(
            message_sequence=np.zeros((1, 1), dtype=int),
            watermark=np.zeros((1, 1), dtype=int),
            num_frames=0,
            var=1,
            decoding_key=decoding_key,
            GF=DummyGF(),
            threshold=0.5,
            device=torch.device("cpu"),
        )

        # 5D but only 1 frame => frames_to_use<=1 branch (259-267)
        latents = torch.zeros(1, 4, 1, 2, 2)
        res = det.eval_watermark(latents, detector_type="bit_acc")
        assert res["is_watermarked"] is False
        assert res["bit_acc"] == 0.0
        assert res["recovered_index"].size == 0
    
    # video quality analyzer
    def test_video_quality_analyzer_base_init_and_analyze_raises(self):
        from PIL import Image
        from markdiffusion.evaluation.tools.video_quality_analyzer import VideoQualityAnalyzer

        analyzer = VideoQualityAnalyzer()  # __init__ pass branch
        try:
            analyzer.analyze([Image.new("RGB", (8, 8))])
            assert False, "Expected NotImplementedError"
        except NotImplementedError:
            pass


    def test_video_quality_importerror_fallback_for_bicubic(self, monkeypatch):
        import builtins
        import importlib
        import sys

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torchvision.transforms" and fromlist and "InterpolationMode" in fromlist:
                raise ImportError("forced")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        sys.modules.pop("markdiffusion.evaluation.tools.video_quality_analyzer", None)
        m = importlib.import_module("markdiffusion.evaluation.tools.video_quality_analyzer")

        assert hasattr(m, "BICUBIC")
        assert m.BICUBIC is not None


    def test_subject_consistency_single_frame_returns_1(self):
        from PIL import Image
        from markdiffusion.evaluation.tools.video_quality_analyzer import SubjectConsistencyAnalyzer

        # bypass heavy init by constructing without __init__
        analyzer = SubjectConsistencyAnalyzer.__new__(SubjectConsistencyAnalyzer)
        analyzer.device = "cpu"

        out = SubjectConsistencyAnalyzer.analyze(analyzer, [Image.new("RGB", (16, 16))])
        assert out == 1.0


    def test_subject_consistency_two_frames_cosine_path(self, monkeypatch):
        import torch
        from PIL import Image
        from markdiffusion.evaluation.tools.video_quality_analyzer import SubjectConsistencyAnalyzer

        analyzer = SubjectConsistencyAnalyzer.__new__(SubjectConsistencyAnalyzer)
        analyzer.device = torch.device("cpu")

        # patch transform to avoid torchvision
        def fake_transform(img):
            return torch.zeros(3, 224, 224)

        analyzer.transform = fake_transform

        class DummyModel:
            def __call__(self, x):
                # return deterministic feature vector
                return torch.tensor([[1.0, 0.0]], dtype=torch.float32)

            def eval(self):
                return self

            def to(self, device):
                return self

        analyzer.model = DummyModel()

        frames = [Image.new("RGB", (16, 16)), Image.new("RGB", (16, 16))]
        score = SubjectConsistencyAnalyzer.analyze(analyzer, frames)
        assert 0.0 <= score <= 1.0


    def test_motion_smoothness_cpu_initialize_params_branch(self):
        import torch
        from markdiffusion.evaluation.tools.video_quality_analyzer import MotionSmoothnessAnalyzer

        analyzer = MotionSmoothnessAnalyzer.__new__(MotionSmoothnessAnalyzer)
        analyzer.device = torch.device("cpu")
        analyzer.niters = 1

        MotionSmoothnessAnalyzer._initialize_params(analyzer)
        assert analyzer.anchor_resolution == 8192 * 8192
        assert analyzer.anchor_memory == 1
        assert analyzer.anchor_memory_bias == 0
        assert analyzer.vram_avail == 1
        assert isinstance(analyzer.embt, torch.Tensor)


    def test_motion_smoothness_extract_frames_even_indices(self):
        from PIL import Image
        import numpy as np
        from markdiffusion.evaluation.tools.video_quality_analyzer import MotionSmoothnessAnalyzer

        analyzer = MotionSmoothnessAnalyzer.__new__(MotionSmoothnessAnalyzer)

        frames = [
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)),
            Image.fromarray(np.ones((8, 8, 3), dtype=np.uint8)),
            Image.fromarray(np.full((8, 8, 3), 2, dtype=np.uint8)),
        ]
        out0 = MotionSmoothnessAnalyzer._extract_frames(analyzer, frames, start_from=0)
        out1 = MotionSmoothnessAnalyzer._extract_frames(analyzer, frames, start_from=1)

        assert len(out0) == 2  # indices 0,2
        assert len(out1) == 1  # index 1


    def test_motion_smoothness_compute_vfi_score_empty_returns_0(self):
        from markdiffusion.evaluation.tools.video_quality_analyzer import MotionSmoothnessAnalyzer

        analyzer = MotionSmoothnessAnalyzer.__new__(MotionSmoothnessAnalyzer)
        score = MotionSmoothnessAnalyzer._compute_vfi_score(analyzer, [], [])
        assert score == 0.0


    def test_dynamic_degree_check_dynamic_motion_true_and_false(self):
        from markdiffusion.evaluation.tools.video_quality_analyzer import DynamicDegreeAnalyzer

        analyzer = DynamicDegreeAnalyzer.__new__(DynamicDegreeAnalyzer)

        thresholds = {"magnitude_threshold": 1.0, "count_threshold": 2}
        assert DynamicDegreeAnalyzer._check_dynamic_motion(analyzer, [0.5, 2.0, 2.0], thresholds) is True
        assert DynamicDegreeAnalyzer._check_dynamic_motion(analyzer, [0.5, 0.7, 0.9], thresholds) is False


    def test_imaging_quality_preprocess_frames_resize_branch(self):
        import torch
        from PIL import Image
        from markdiffusion.evaluation.tools.video_quality_analyzer import ImagingQualityAnalyzer

        analyzer = ImagingQualityAnalyzer.__new__(ImagingQualityAnalyzer)
        analyzer.device = torch.device("cpu")

        # make pil_to_torch produce large tensors so resize branch triggers
        import markdiffusion.evaluation.tools.video_quality_analyzer as vqa_mod

        def fake_pil_to_torch(img, normalize=False):
            # (C,H,W) with max(h,w)>512
            return torch.zeros(3, 600, 700)

        orig = vqa_mod.pil_to_torch
        vqa_mod.pil_to_torch = fake_pil_to_torch
        try:
            frames = [Image.new("RGB", (16, 16)), Image.new("RGB", (16, 16))]
            out = ImagingQualityAnalyzer._preprocess_frames(analyzer, frames)
            assert out.shape[0] == 2
            assert max(out.shape[-2], out.shape[-1]) <= 512
        finally:
            vqa_mod.pil_to_torch = orig


