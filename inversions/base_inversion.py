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
    
    def _apply_guidance_scale(self, model_output, guidance_scale):
        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = model_output.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            return noise_pred
        else:
            return model_output