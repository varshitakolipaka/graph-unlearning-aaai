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

class RetrainTrainer(Trainer):
    def __init__(self, model, data, optimizer, args):
        super().__init__(model, data, optimizer, args)

    def train(self):
        self.data = self.data.to(device)
        start_time = time.time()
        for epoch in trange(self.args.unlearning_epochs, desc='Epoch'):
            iter_start_time = time.time()
            self.model.train()
            z = F.log_softmax(self.model(self.data.x, self.data.edge_index[:, self.data.dr_mask]), dim=-1)
            loss = F.nll_loss(z[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            self.unlearning_time += time.time() - iter_start_time
            if epoch % 10 == 0:
                self.save_best(inf=True)
            
        print(f'Training Time: {self.unlearning_time}')
        self.load_best()
        train_acc, msc_rate, f1 = self.evaluate(is_dr=True, use_val=True)
        # print(f'Train Acc: {train_acc}, Misclassification: {msc_rate},  F1 Score: {f1}')
        
        return train_acc, msc_rate, self.unlearning_time