from .base_inversion import BaseInversion
import torch
from typing import Optional, Callable
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from utils.DPMSolverPatch import convert_model_output
from diffusers import DPMSolverMultistepInverseScheduler

class ExactInversion(BaseInversion):
    def __init__(self,
                 scheduler,
                 unet,
                 device,
                 ):
        scheduler = DPMSolverMultistepInverseScheduler.from_config(scheduler.config)
        super(ExactInversion, self).__init__(scheduler, unet, device)
    
    @torch.inference_mode()
    def forward_diffusion(
        self,
        use_old_emb_i=25,
        text_embeddings=None,
        old_text_embeddings=None,
        new_text_embeddings=None,
        latents: Optional[torch.FloatTensor] = None,
        num_inference_steps: int = 10,
        guidance_scale: float = 7.5,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        inverse_opt=False,
        inv_order=0,
        **kwargs,
    ):  
        with torch.no_grad():
            # Keep a list of inverted latents as the process goes on
            intermediate_latents = []
            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = guidance_scale > 1.0

            self.scheduler.set_timesteps(num_inference_steps)
            timesteps_tensor = self.scheduler.timesteps.to(self.device)
            latents = latents * self.scheduler.init_noise_sigma

            if old_text_embeddings is not None and new_text_embeddings is not None:
                prompt_to_prompt = True
            else:
                prompt_to_prompt = False

            if inv_order is None:
                inv_order = self.scheduler.solver_order
            inverse_opt = (inv_order != 0)
            
            # timesteps_tensor = reversed(timesteps_tensor) # inversion process

            self.unet = self.unet.float()
            latents = latents.float()
            text_embeddings = text_embeddings.float()

            for i, t in enumerate(tqdm(timesteps_tensor)):          
                if self.scheduler.step_index is None:
                    self.scheduler._init_step_index(t)

                if prompt_to_prompt:
                    if i < use_old_emb_i:
                        text_embeddings = old_text_embeddings
                    else:
                        text_embeddings = new_text_embeddings

                if i+1 < len(timesteps_tensor):
                    next_timestep = timesteps_tensor[i+1]
                else:
                    next_timestep = (
                        t
                        + self.scheduler.config.num_train_timesteps
                        // self.scheduler.num_inference_steps
                    )
                    next_timestep = min(next_timestep, self.scheduler.config.num_train_timesteps - 1)
                

                # call the callback, if provided
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)
                

                # Our Algorithm

                # Algorithm 1
                if inv_order < 2 or (inv_order == 2 and i == 0):
                    # s = t 
                    # t = prev_timestep
                    s = next_timestep
                    t = (
                        next_timestep
                        - self.scheduler.config.num_train_timesteps
                        // self.scheduler.num_inference_steps
                    )
                    t = max(t, 0) # Ensure t is not negative
                    
                    lambda_s, lambda_t = self.scheduler.lambda_t[s], self.scheduler.lambda_t[t]
                    sigma_s, sigma_t = self.scheduler.sigma_t[s], self.scheduler.sigma_t[t]
                    h = lambda_t - lambda_s
                    alpha_s, alpha_t = self.scheduler.alpha_t[s], self.scheduler.alpha_t[t]
                    phi_1 = torch.expm1(-h)
                    
                    # expand the latents if classifier free guidance is used
                    latent_model_input, info = self._prepare_latent_for_unet(latents, do_classifier_free_guidance, self.unet)
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)                
                    # predict the noise residual
                    noise_pred_raw = self.unet(latent_model_input, s, encoder_hidden_states=text_embeddings).sample 
                    noise_pred = self._restore_latent_from_unet(noise_pred_raw, info, guidance_scale)

                    model_s = self.scheduler.convert_model_output(model_output=noise_pred, sample=latents)
                    x_t = latents

                    # Line 5
                    latents = (sigma_s / sigma_t) * (latents + alpha_t * phi_1 * model_s)      

                    # Save intermediate latents
                    intermediate_latents.append(latents.clone())
                    
                    self.scheduler._step_index += 1

                else:
                    pass

        return intermediate_latents
    
    @torch.inference_mode()
    def backward_diffusion(
        self,
        latents: Optional[torch.FloatTensor] = None,
        num_inference_steps: int = 10,
        guidance_scale: float = 7.5,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        inv_order=None,
        **kwargs,
    ):
        """
        Reconstruct z_0 from z_T via the forward diffusion process
        """
        with torch.no_grad():
            # 1. Setup
            do_classifier_free_guidance = guidance_scale > 1.0
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps_tensor = self.scheduler.timesteps.to(self.device)
            
            # If no inv_order provided, default to scheduler's configuration
            if inv_order is None:
                inv_order = self.scheduler.solver_order

            self.unet = self.unet.float()
            latents = latents.float()
            
            # last output from the model to be used in higher order methods
            old_model_output = None 

            # 2. Denoising Loop (T -> 0)
            for i, t in enumerate(tqdm(timesteps_tensor)):
                if self.scheduler.step_index is None:
                    self.scheduler._init_step_index(t)

                # s (prev_timestep in diffusion terms, lower noise)
                if i + 1 < len(timesteps_tensor):
                    s = timesteps_tensor[i + 1]
                else:
                    s = torch.tensor(0, device=self.device)

                # 3. Prepare Model Input
                latent_model_input, info = self._prepare_latent_for_unet(latents, do_classifier_free_guidance, self.unet)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # 4. Predict Noise/Data
                noise_pred_raw = self.unet(latent_model_input, t, encoder_hidden_states=kwargs.get("text_embeddings")).sample
                noise_pred = self._restore_latent_from_unet(noise_pred_raw, info, guidance_scale)
                
                # Transform prediction according to the type of prediction required by the scheduler
                model_output = self.scheduler.convert_model_output(model_output=noise_pred, sample=latents)

                # 5. Calculate Solver Parameters
                # Aquire alpha, sigma, lambda
                lambda_t, lambda_s = self.scheduler.lambda_t[t], self.scheduler.lambda_t[s]
                alpha_t, alpha_s = self.scheduler.alpha_t[t], self.scheduler.alpha_t[s]
                sigma_t, sigma_s = self.scheduler.sigma_t[t], self.scheduler.sigma_t[s]
                
                h = lambda_s - lambda_t  # step size
                phi_1 = torch.expm1(-h)  # e^{-h} - 1

                # 6. Sampling Step (Explicit)
                
                # Case 1: First Order (DDIM) or First Step of Second Order
                if inv_order == 1 or i == 0:
                    #  Eq. (5): Forward Euler
                    # x_{t_i} = (sigma_{t_i} / sigma_{t_{i-1}}) * x_{t_{i-1}} - alpha_{t_i} * (e^{-h} - 1) * x_theta
                    #  x_s = (sigma_s/sigma_t) * latents - alpha_s * phi_1 * model_output
                    latents = (sigma_s / sigma_t) * latents - (alpha_s * phi_1) * model_output
                else:
                    pass

                # Update history
                old_model_output = model_output
                self.scheduler._step_index += 1

                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

            return latents
