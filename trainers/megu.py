import os
import time
import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import grad

from .base import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MeguTrainer(Trainer):
    def __init__(self, model, data, optimizer, args):
        super().__init__(model, data, optimizer)
        self.args= args
        
    def train(self):
        pass