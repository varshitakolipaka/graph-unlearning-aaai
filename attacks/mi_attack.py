import os
import json
import wandb
import numpy as np
import torch
import torch.nn as nn
from tqdm import trange, tqdm
from torch_geometric.utils import negative_sampling
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
import torch.nn.functional as F
from trainers.base import get_link_labels, EdgeTrainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AttackModel(nn.Module):
    def __init__(self, num_features):
        super(AttackModel, self).__init__()
        self.fc1 = nn.Linear(num_features, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MIAttackTrainer(EdgeTrainer):
    '''This code is adapted from https://github.com/iyempissy/rebMIGraph'''

    def __init__(self, args):
        self.args = args
        self.trainer_log = {
            'unlearning_model': 'member_infer', 
            'dataset': args.dataset, 
            'seed': args.random_seed,
            'shadow_log': [],
            'attack_log': []}
        self.logit_all_pair = None

    def train_shadow(self, model, data, optimizer, args):
        model = model.to(device)
        data = data.to(device)

        best_valid_loss = 1000000

        all_neg = []
        # Train shadow model using the test data
        for epoch in trange(500, desc='Train shadow model'):
            model.train()

            # Positive and negative sample
            neg_edge_index = negative_sampling(
                edge_index=data.test_pos_edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=data.test_pos_edge_index.shape[1])
            
            z = model(data.x, data.test_pos_edge_index)
            logits = model.decode(z, data.test_pos_edge_index, neg_edge_index)
            label = get_link_labels(data.test_pos_edge_index, neg_edge_index) # ground truth
            loss = F.binary_cross_entropy_with_logits(logits, label) 
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            all_neg.append(neg_edge_index.cpu())

            if (epoch+1) % 500 == 0:
                valid_loss, auc, aup, df_logit, _ = self.eval_shadow(model, data, 'val')

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_epoch = epoch

        valid_loss, auc, aup, df_logit, _ = self.eval_shadow(model, data, 'val')
        print('AUC of the shadow model: ', auc)

        return torch.cat(all_neg, dim=-1)

    @torch.no_grad()
    def eval_shadow(self, model, data, stage='val'):
        model.eval()
        pos_edge_index = data[f'{stage}_pos_edge_index']
        neg_edge_index = data[f'{stage}_neg_edge_index']

        z = model(data.x, data.val_pos_edge_index)
        logits = model.decode(z, pos_edge_index, neg_edge_index).sigmoid()
        label = self.get_link_labels(pos_edge_index, neg_edge_index)

        loss = F.binary_cross_entropy_with_logits(logits, label).cpu().item()
        auc = roc_auc_score(label.cpu(), logits.cpu())
        aup = average_precision_score(label.cpu(), logits.cpu())
        df_logit = float('nan')

        return loss, auc, aup, df_logit, None
    
    def train_attack(self, model, train_loader, valid_loader, optimizer, leak, args):
        loss_fct = nn.CrossEntropyLoss()
        best_auc = 0
        best_epoch = 0
        for epoch in trange(50, desc='Train attack model'):
            model.train()
            model.to(device)

            train_loss = 0
            for x, y in train_loader:
                logits = model(x.to(device))
                loss = loss_fct(logits, y.to(device))

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                train_loss += loss.item()

            valid_loss, valid_acc, valid_auc, valid_f1 = self.eval_attack(model, valid_loader)

            if valid_auc > best_auc:
                best_auc = valid_auc
                best_epoch = epoch

        valid_loss, valid_acc, valid_auc, valid_f1 = self.eval_attack(model, valid_loader)
        print('AUC and accuracy of the attack model: ', valid_auc, valid_acc)
        
    @torch.no_grad()
    def eval_attack(self, model, eval_loader):
        loss_fct = nn.CrossEntropyLoss()
        pred = []
        label = []
        for x, y in eval_loader:
            logits = model(x.to(device))
            loss = loss_fct(logits, y.to(device))
            _, p = torch.max(logits, 1)
            
            pred.extend(p.cpu())
            label.extend(y)
        
        pred = torch.stack(pred)
        label = torch.stack(label)

        return loss.item(), accuracy_score(label.numpy(), pred.numpy()), roc_auc_score(label.numpy(), pred.numpy()), f1_score(label.numpy(), pred.numpy(), average='macro')

    @torch.no_grad()
    def prepare_attack_training_data(self, model, data, leak='posterior', all_neg=None):
        '''Prepare the training data of attack model (Present vs. Absent)
            Present edges (label = 1): training data of shadow model (Test pos and neg edges)
            Absent edges (label = 0): validation data of shadow model (Valid pos and neg edges)
        '''

        z = model(data.x, data.test_pos_edge_index) # torch.Size([19793, 64]) 
        # all_neg = negative_sampling(
        #     edge_index=data.train_pos_edge_index,
        #     num_nodes=data.num_nodes,
        #     num_neg_samples=data.train_pos_edge_index.shape[1])
        # # Sample same size of neg as pos
        # sample_idx = torch.randperm(all_neg.shape[1])[:data.test_pos_edge_index.shape[1]]
        # neg_subset = all_neg[:, sample_idx]

        present_edge_index = torch.cat([data.test_pos_edge_index, data.test_neg_edge_index], dim=-1)  # 6342
        
        if 'sub' in self.args.unlearning_model:
            absent_edge_index = torch.cat([data.val_pos_edge_index, data.val_neg_edge_index], dim=-1)
        else:   #if 'all' in self.args.unlearning_model:
            absent_edge_index = torch.cat([data.val_pos_edge_index, data.val_neg_edge_index], dim=-1)  # 123671
        
        edge_index = torch.cat([present_edge_index, absent_edge_index], dim=-1)  # E (6342 + 6342)

        if leak == 'posterior':
            feature1 = model.decode(z, edge_index).sigmoid().cpu() # get the logits of the test data
            feature0 = 1 - feature1
            feature = torch.stack([feature0, feature1], dim=1)
        elif leak == 'repr':
            feature = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=-1).cpu()  # torch.Size([E, 128])
        label = get_link_labels(present_edge_index, absent_edge_index).long().cpu()  # present 1, absent 0

        return feature, label
