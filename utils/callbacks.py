import torch
from typing import List

class DenoisingLatentsCollector:
    def __init__(self, save_every_n_steps: int = 1, to_cpu: bool = True):
        """Initialize the latents collector.

        Args:
            save_every_n_steps (int, optional): Save latents every n steps. Defaults to 1.
            to_cpu (bool, optional): Whether to move latents to CPU. Defaults to True.
        """
        
        self.save_every_n_steps = save_every_n_steps
        self.to_cpu = to_cpu
        self.data = []
        self._call_count = 0
    
    def __call__(self, step: int, timestep: int, latents: torch.Tensor):
        self._call_count += 1
        
        if self._call_count % self.save_every_n_steps == 0:
            latents_to_save = latents.clone()
            if self.to_cpu:
                latents_to_save = latents_to_save.cpu()
            
            self.data.append({
                'step': step,
                'timestep': timestep,
                'latents': latents_to_save,
                'call_count': self._call_count
            })
    
    @property
    def latents_list(self) -> List[torch.Tensor]:
        """Return the list of latents."""
        return [item['latents'] for item in self.data]
    
    @property
    def timesteps_list(self) -> List[int]:
        """Return the list of timesteps."""
        return [item['timestep'] for item in self.data]
    
    def get_latents_at_step(self, step: int) -> torch.Tensor:
        """Get the latents at a specific step."""
        for item in self.data:
            if item['step'] == step:
                return item['latents']
        raise ValueError(f"No latents found for step {step}")
    
    def clear(self):
        """Clear the collected data."""
        self.data.clear()
        self._call_count = 0