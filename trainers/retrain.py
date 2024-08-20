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

from .base import Trainer, EdgeTrainer

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class RetrainTrainer(Trainer):
    def __init__(self, model, data, optimizer, args):
        super().__init__(model, data, optimizer)
        self.args = args

    def train(self):
        self.data = self.data.to(device)

        for epoch in trange(self.args.unlearning_epochs, desc='Epoch'):
            self.model.train()
            z = F.log_softmax(self.model(self.data.x, self.data.edge_index[:, self.data.dr_mask]), dim=1)
            loss = F.nll_loss(z[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        train_acc, msc_rate, f1 = self.evaluate(is_dr=True)
        print(f'Train Acc: {train_acc}, Misclassification: {msc_rate},  F1 Score: {f1}')

class RetrainEdgeTrainer(EdgeTrainer):
    def __init__(self, model, data, optimizer, args):
        super().__init__(args)
        self.args = args

    def train(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None, logits_before_unlearning=None):
        model = model.to(device)
        data = data.to(device)

        best_metric = 0
        loss_fct = nn.MSELoss()
        
        for epoch in trange(args.epochs, desc='Unlearning'):
            model.train()

            start_time = time.time()
            total_step = 0
            total_loss = 0

            neg_edge_index = negative_sampling(
                edge_index=data.train_pos_edge_index[:, data.dr_mask],
                num_nodes=data.num_nodes,
                num_neg_samples=data.dr_mask.sum())

            z = model(data.x, data.train_pos_edge_index[:, data.dr_mask])
            logits = model.decode(z, data.train_pos_edge_index[:, data.dr_mask], neg_edge_index)
            label = self.get_link_labels(data.train_pos_edge_index[:, data.dr_mask], neg_edge_index)
            loss = F.binary_cross_entropy_with_logits(logits, label)

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()

            total_step += 1
            total_loss += loss.item()

            if (epoch + 1) % self.args.valid_freq == 0:
                valid_loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, valid_log = self.eval(model, data, 'val')

                if dt_auc + df_auc > best_metric:
                    best_metric = dt_auc + df_auc
                    best_epoch = epoch

            test_results = self.test(model, data, attack_model_all=attack_model_all, logits_before_unlearning=logits_before_unlearning)
            print('===AFTER UNLEARNING===\n', test_results[-1])
            return test_results[8]
        
        