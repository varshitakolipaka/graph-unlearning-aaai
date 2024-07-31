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
        super().__init__(model, data, optimizer)
        self.args = args

    def train(self, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        if 'ogbl' in self.args.dataset:
            self.args.eval_on_cpu = False
            return self.train_fullbatch(self.model, self.data, self.optimizer, self.args, logits_ori, attack_model_all, attack_model_sub)

        else:
            return self.train_fullbatch(self.model, self.data, self.optimizer, self.args, logits_ori, attack_model_all, attack_model_sub)

    def train_fullbatch(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        model = model.to(device)
        data = data.to(device)

        best_metric = 0
        loss_fct = nn.MSELoss()

        for epoch in trange(args.unlearning_epochs, desc='Unlearning'):
            model.train()

            start_time = time.time()
            total_step = 0
            total_loss = 0

            z = model(data.x, data.edge_index[:, data.dr_mask])
            loss = F.nll_loss(z[data.train_mask], data.y[data.train_mask])

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()

            total_step += 1
            total_loss += loss.item()

            end_time = time.time()
            epoch_time = end_time - start_time

            step_log = {
                'Epoch': epoch,
                'train_loss': loss.item(),
                'train_time': epoch_time
            }
            print(step_log)

            if (epoch + 1) % self.args.valid_freq == 0:
                train_acc, msc_rate, f1 = self.evaluate(is_dr=True)

                print(f'Epoch {epoch:04d} | Train Acc: {train_acc}, Misclassification: {msc_rate},  F1 Score: {f1}')

                if train_acc > best_metric:
                    best_metric = train_acc
                    best_epoch = epoch

        print(f'Training finished. Best checkpoint at epoch = {best_epoch:04d}, best metric = {best_metric:.4f}')
        
        train_acc, msc_rate, f1 = self.evaluate(is_dr=True)
        print(f'Epoch {epoch:04d} | Train Acc: {train_acc}, Misclassification: {msc_rate},  F1 Score: {f1}')