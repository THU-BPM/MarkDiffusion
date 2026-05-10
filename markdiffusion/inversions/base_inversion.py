import torch
from typing import Optional, Callable

class BaseInversion():
    def __init__(self,
                 scheduler,
                 unet,
                 device,
                 ):
        self.scheduler = scheduler
        self.unet = unet
        self.device = device
    
    def _prepare_latent_for_unet(self, latents, do_cfg, unet):
        """
        Inputs:
            latents: [B,C,H,W] or [B,F,C,H,W]
            do_cfg: bool
        Outputs:
            latent_model_input: Tensor ready for UNet input
            info: dict containing shape info
        """

        is_video_unet = any(isinstance(m, torch.nn.Conv3d) for m in unet.modules())

        info = {
            "do_cfg": do_cfg,
            "is_video_unet": is_video_unet
        }

        # ------------------------------
        # Case 1: image latent (4D)
        # ------------------------------
        if latents.ndim == 4:
            # [B, C, H, W]
            info["shp"] = latents.shape
            if do_cfg:
                latents = torch.cat([latents, latents], dim=0)
            return latents, info

        # ------------------------------
        # Case 2: video latent (5D)
        # ------------------------------
        assert latents.ndim == 5, "Video input must be 4D or 5D latent."
        B, F, C, H, W = latents.shape
        info["shp"] = (B, F, C, H, W)

        if is_video_unet:
            # video UNet (Conv3d): [B, C, F, H, W]
            latents = latents.permute(0, 2, 1, 3, 4).contiguous()
            if do_cfg:
                latents = torch.cat([latents, latents], dim=0)
            return latents, info

        else:
            # image UNet but input video → flatten frames
            latents = latents.reshape(B * F, C, H, W)
            info["flatten"] = (B, F)
            if do_cfg:
                latents = torch.cat([latents, latents], dim=0)
            return latents, info
    
    def _restore_latent_from_unet(self, noise_pred, info, guidance_scale):
        """
        Inputs:
            noise_pred: UNet Input
            info: prepare 阶段保存的结构信息
            guidance_scale: CFG scale
        输出:
            与原输入匹配格式的噪声预测
            图像: [B,C,H,W]
            视频: [B,F,C,H,W]
        """

        do_cfg = info["do_cfg"]
        is_video_unet = info["is_video_unet"]
        shp = info["shp"]

        # 1. CFG 合并
        if do_cfg:
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # --------------------------
        # Case 1: 输入是图像 latent
        # --------------------------
        if len(shp) == 4:
            # [B, C, H, W] — no reshape needed
            return noise_pred

        # --------------------------
        # Case 2: 输入是视频 latent
        # --------------------------
        B, F, C, H, W = shp

        if is_video_unet:
            # video UNet 输出格式: [B, C, F, H, W]
            noise_pred = noise_pred.permute(0, 2, 1, 3, 4).contiguous()
            return noise_pred

        else:
            # 图像 UNet 输出格式: [B*F, C, H, W]
            noise_pred = noise_pred.reshape(B, F, C, H, W)
            return noise_pred
       
    @torch.inference_mode() 
    def forward_diffusion(self,
                          use_old_emb_i=25,
                          text_embeddings=None,
                          old_text_embeddings=None,
                          new_text_embeddings=None,
                          latents: Optional[torch.FloatTensor] = None,
                          num_inference_steps: int = 10,
                          guidance_scale: float = 7.5,
                          callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
                          callback_steps: Optional[int] = 1,
                          inverse_opt=True,
                          inv_order=None,
                          **kwargs,
                          ):
        pass
    
    # def _apply_guidance_scale(self, model_output, guidance_scale):
    #     if guidance_scale > 1.0:
    #         noise_pred_uncond, noise_pred_text = model_output.chunk(2)
    #         noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    #         return noise_pred
    #     else:
    #         return model_output