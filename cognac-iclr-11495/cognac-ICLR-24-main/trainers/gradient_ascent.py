import os
import time
# import wandb
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import GraphSAINTRandomWalkSampler

from .base import Trainer

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def weight(model):
    t = 0
    for p in model.parameters():
        t += torch.norm(p)
    return t

class GradientAscentTrainer(Trainer):
    def __init__(self, model, data, optimizer, args):
        super().__init__(model, data, optimizer)
        self.args= args

    def train(self):
        return self.train_fullbatch()

    def train_fullbatch(self):
        start_time = time.time()
        self.model = self.model.to(device)
        self.data = self.data.to(device)

        best_metric = 0

        for epoch in trange(self.args.unlearning_epochs, desc='Unlearning'):
            self.model.train()

            # Positive and negative sample
            neg_edge_index = negative_sampling(
                edge_index=self.data.train_pos_edge_index[:, self.data.df_mask],
                num_nodes=self.data.num_nodes,
                num_neg_samples=self.data.df_mask.sum())

            z = self.model(self.data.x, self.data.train_pos_edge_index)
            logits = self.model.decode(z, self.data.train_pos_edge_index[:, self.data.df_mask], neg_edge_index=neg_edge_index)
            label = torch.ones_like(logits, dtype=torch.float, device=device)
            loss = -F.binary_cross_entropy_with_logits(logits, label)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            self.optimizer.zero_grad()

        end_time = time.time()

        train_acc, msc_rate, f1 = self.evaluate(is_dr=True)
        # print(f'Train Acc: {train_acc}, Misclassification: {msc_rate},  F1 Score: {f1}')

        return train_acc, msc_rate, end_time - start_time