import os
import time
import wandb
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import GraphSAINTRandomWalkSampler

from .base import Trainer, EdgeTrainer

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
        self.model = self.model.to(device)
        self.data = self.data.to(device)

        start_time = time.time()
        best_metric = 0

        for epoch in trange(self.args.unlearning_epochs, desc='Unlearning'):
            self.model.train()

            start_time = time.time()
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
            epoch_time = end_time - start_time

        train_acc, msc_rate, f1 = self.evaluate(is_dr=True)
        print(f'Train Acc: {train_acc}, Misclassification: {msc_rate},  F1 Score: {f1}')

class GradientAscentEdgeTrainer(EdgeTrainer):
    def __init__(self, model, data, optimizer, args):
        super().__init__(args)
        self.args = args

    def train(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None, logits_before_unlearning=None):
        model = model.to(device)
        data = data.to(device)
        
        start_time = time.time()
        best_metric = 0

        for epoch in trange(args.epochs, desc='Unlearning'):
            model.train()

            start_time = time.time()

            # Positive and negative sample
            neg_edge_index = negative_sampling(
                edge_index=data.train_pos_edge_index[:, data.df_mask],
                num_nodes=data.num_nodes,
                num_neg_samples=data.df_mask.sum())

            z = model(data.x, data.train_pos_edge_index)
            logits = model.decode(z, data.train_pos_edge_index[:, data.df_mask])
            label = torch.ones_like(logits, dtype=torch.float, device='cuda')
            loss = -F.binary_cross_entropy_with_logits(logits, label)

            # print('aaaaaaaaaaaaaa', data.df_mask.sum(), weight(model))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()

            if (epoch + 1) % self.args.valid_freq == 0:
                valid_loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, valid_log = self.eval(model, data, 'val')
                valid_log['Epoch'] = epoch
                
                if dt_auc + df_auc > best_metric:
                    best_metric = dt_auc + df_auc
                    best_epoch = epoch

        test_results = self.test(model, data, attack_model_all=attack_model_all, logits_before_unlearning=logits_before_unlearning)
        print('===AFTER UNLEARNING===\n', test_results[-1])
        return test_results[8]