import os
import time
#import wandb
from tqdm import tqdm, trange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import GraphSAINTRandomWalkSampler

from .base import Trainer


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
class UtUTrainer(Trainer):
    def __init__(self, model, data, optimizer, args):
        super().__init__(model, data, optimizer)
        self.args= args

    def train(self):
        # no training for UtU, only inference
        self.model = self.model.to(device)
        self.data = self.data.to(device)
        
        test_acc, msc_rate, f1 = self.evaluate(is_dr=True)
        print(f'Train Acc: {test_acc}, Misclassification: {msc_rate},  F1 Score: {f1}')
        

