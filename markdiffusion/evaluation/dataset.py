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


import ujson as json
from datasets import load_dataset
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import random
from typing import List

class BaseDataset:
    """Base dataset class."""
    
    def __init__(self, max_samples: int = 200):
        """Initialize the dataset.
        
        Parameters:
            max_samples: Maximum number of samples to load.
        """
        self.max_samples = max_samples
        self.prompts = []
        self.references = []

    @property
    def num_samples(self) -> int:
        """Number of samples in the dataset."""
        return len(self.prompts)
    
    @property
    def num_references(self) -> int:
        """Number of references in the dataset."""
        return len(self.references)
    
    def get_prompt(self, idx) -> str:
        """Get the prompt at the given index."""
        return self.prompts[idx]
    
    def get_reference(self, idx) -> Image.Image:
        """Get the reference Image at the given index."""
        return self.references[idx]
    
    def __len__(self) -> int:
        """Number of samples in the dataset.(Equivalent to num_samples)"""
        return self.num_samples
    
    def __getitem__(self, idx) -> tuple[str, Image.Image]:
        """Get the prompt (and reference Image if available) at the given index."""
        if len(self.references) == 0:
            return self.prompts[idx]
        else:
            return self.prompts[idx], self.references[idx]
    
    def _load_data(self):
        """Load data from the dataset."""
        pass
        
        
class StableDiffusionPromptsDataset(BaseDataset):
    """Stable Diffusion prompts dataset."""
    
    def __init__(self, max_samples: int = 200, split: str = "test", shuffle: bool = False):
        """Initialize the dataset.
        
        Parameters:
            max_samples: Maximum number of samples to load.
            split: Split to load.
            shuffle: Whether to shuffle the dataset.
        """
        super().__init__(max_samples)
        self.split = split
        self.shuffle = shuffle
        self._load_data()
        
    @property
    def name(self):
        """Name of the dataset."""
        return "Stable Diffusion Prompts"
        
    def _load_data(self):
        dataset = load_dataset("dataset/stable_diffusion_prompts", split=self.split)
        if self.shuffle:
            dataset = dataset.shuffle()
        for prompt in dataset["Prompt"][:self.max_samples]:
            self.prompts.append(prompt)
            
class MSCOCODataset(BaseDataset):
    """MSCOCO 2017 dataset."""
    
    def __init__(self, max_samples: int = 200, shuffle: bool = False):
        """Initialize the dataset.
        
        Parameters:
            max_samples: Maximum number of samples to load.
            shuffle: Whether to shuffle the dataset.
        """
        super().__init__(max_samples)
        self.shuffle = shuffle
        self._load_data()
        
    @property
    def name(self):
        """Name of the dataset."""
        return "MS-COCO 2017"
    
    def _load_image_from_url(self, url):
        """Load image from url."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            return image
        except Exception as e:
            print(f"Load image from url failed: {e}")
            return None
        
    def _load_data(self):
        """Load data from the MSCOCO 2017 dataset."""
        df = pd.read_parquet("dataset/mscoco/mscoco.parquet")
        if self.shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        for i in tqdm(range(self.max_samples), desc="Loading MSCOCO dataset"):
            item = df.iloc[i]
            self.prompts.append(item['TEXT'])
            self.references.append(self._load_image_from_url(item['URL']))
            
class VBenchDataset(BaseDataset):
    """VBench dataset."""
    
    def __init__(self, max_samples: int, dimension: str = "subject_consistency", shuffle: bool = False):
        """Initialize the dataset.

        Args:
            max_samples (int): Maximum number of samples to load.
            dimension (str, optional): Dimensions to load. Selected from "subject_consistency", "background_consistency", "imaging_quality", "motion_smoothness", "dynamic_degree".
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        """
        super().__init__(max_samples)
        self.shuffle = shuffle
        self.dimension = dimension
        self._load_data()
        
    @property
    def name(self):
        """Name of the dataset."""
        return "VBench"
    
    def _load_data(self):
        """Load data from the VBench dataset."""
        with open(f"dataset/vbench/prompts_per_dimension/{self.dimension}.txt", "r") as f:
            prompts = [line.strip() for line in f.readlines()]
        if self.shuffle:
            random.shuffle(prompts)
        self.prompts.extend(prompts[:self.max_samples])
        