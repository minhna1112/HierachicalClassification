from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader

import torch

class DummyTrainer(ABC):
    
    def __init___(self, 
        batch_size: int,
        split_ratio: float,
        output_types : str)-> None:
        self.dataset : Dataset = None
        self.model: torch.nn.Module = None
        self.losses: torch.nn.Module = None 
        self.optimizer: torch.optim.Optimizer = None
        

    @abstractmethod
    def train_step(self):
        pass

    @abstractmethod
    def eval_step(self):
        pass




    