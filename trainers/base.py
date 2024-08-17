import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
from tqdm import trange
import os
import time
import json
import wandb
import numpy as np
import torch.nn as nn
from tqdm import trange, tqdm
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score


@torch.no_grad()
def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=pos_edge_index.device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

#  THIS IS THE ORIGINAL uTu codebase giving error because of using attack_model.fc1
@torch.no_grad()
def member_infer_attack(target_model, attack_model, data, logits=None, before=False):
    '''Membership inference attack'''

    edge = data.train_pos_edge_index[:, data.df_mask]  # Deleted edges in the training set
    if(before == False):
        z = target_model(data.x, data.train_pos_edge_index[:, data.dr_mask])
    else:
        z = target_model(data.x, data.train_pos_edge_index)
    if attack_model.fc1.in_features == 2:
        feature1 = target_model.decode(z, edge).sigmoid()
        feature0 = 1 - feature1
        feature = torch.stack([feature0, feature1], dim=1) # Posterior MI
    else:
        feature = torch.cat([z[edge[0]], z[edge][1]], dim=-1)  # Embedding/Repr. MI
    logits = attack_model(feature)
    _, pred = torch.max(logits, 1)
    suc_rate = 1 - pred.float().mean()  # label should be zero, aka if pred is 1(member) then attack success

    return torch.softmax(logits, dim=-1).squeeze().tolist(), suc_rate.cpu().item()

# @torch.no_grad()
# def member_infer_attack(target_model, attack_model, data, logits=None):
#     '''Membership inference attack'''

#     edge = data.train_pos_edge_index[:, data.df_mask]  # Deleted edges in the training set
#     z = target_model(data.x, data.train_pos_edge_index[:, data.dr_mask])

#     # Check the number of input features of the first layer of attack_model
#     first_layer = list(attack_model.children())[0]
#     in_features = first_layer.in_features if isinstance(first_layer, nn.Linear) else None

#     if in_features == 2:
#         feature1 = target_model.decode(z, edge).sigmoid()
#         feature0 = 1 - feature1
#         feature = torch.stack([feature0, feature1], dim=1)  # Posterior MI
#     else:
#         feature = torch.cat([z[edge[0]], z[edge[1]]], dim=-1)  # Embedding/Repr. MI

#     logits = attack_model(feature)
#     _, pred = torch.max(logits, 1)
#     suc_rate = 1 - pred.float().mean()  # label should be zero, aka if pred is 1(member) then attack success

#     return torch.softmax(logits, dim=-1).squeeze().tolist(), suc_rate.cpu().item()


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
class Trainer:
    def __init__(self, model, data, optimizer, num_epochs=50):
        self.model = model.to(device)
        self.data = data.to(device)
        self.optimizer = optimizer
        self.num_epochs= num_epochs

    def train(self):
        self.data = self.data.to(device)
        for epoch in trange(self.num_epochs, desc='Epoch'):
            self.model.train()
            z = F.log_softmax(self.model(self.data.x, self.data.edge_index), dim=1)
            loss = F.nll_loss(z[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if epoch % 10 == 0:
                acc, msc_rate, f1 = self.evaluate()
        train_acc, msc_rate, f1 = self.evaluate()
        print(f'Train Acc: {train_acc}, Misclassification: {msc_rate},  F1 Score: {f1}')

    def get_silhouette_scores(self, graph_temp=None):
        self.model.eval()
        with torch.no_grad():
            if(graph_temp is None):
                embeddings= self.model(self.data.x, self.data.edge_index)
            else:
                embeddings= self.model(graph_temp.x, graph_temp.edge_index)

            probabilites = F.softmax(embeddings, dim=1)
            _, predicted_labels= torch.max(probabilites, dim=1)
        return silhouette_score(embeddings, predicted_labels)

    def misclassification_rate(self, true_labels, pred_labels, class1 = 0, class2 = 1):
        class1_to_class2 = ((true_labels == class1) & (pred_labels == class2)).sum().item()
        class2_to_class1 = ((true_labels == class2) & (pred_labels == class1)).sum().item()
        total_class1 = (true_labels == class1).sum().item()
        total_class2 = (true_labels == class2).sum().item()
        misclassification_rate = (class1_to_class2 + class2_to_class1) / (total_class1 + total_class2)
        return misclassification_rate

    def evaluate(self, is_dr=False):
        self.model.eval()
        with torch.no_grad():
            if(is_dr):
                z = F.log_softmax(self.model(self.data.x, self.data.edge_index[:, self.data.dr_mask]), dim=1)
            else:
                z = F.log_softmax(self.model(self.data.x, self.data.edge_index), dim=1)
            loss = F.nll_loss(z[self.data.val_mask], self.data.y[self.data.val_mask]).cpu().item()
            pred = torch.argmax(z[self.data.val_mask], dim=1).cpu()
            dt_acc = accuracy_score(self.data.y[self.data.val_mask].cpu(), pred)
            dt_f1 = f1_score(self.data.y[self.data.val_mask].cpu(), pred, average='micro')
            msc_rate = self.misclassification_rate(self.data.y[self.data.val_mask].cpu(), pred)
        return dt_acc, msc_rate, dt_f1
    
class EdgeTrainer:
    def __init__(self, args, num_epochs=50):
        self.args = args
        self.num_epochs = num_epochs
        self.trainer_log = {
            'unlearning_model': args.unlearning_model, 
            'dataset': args.dataset, 
            'log': []}
        self.logit_all_pair = None
        self.df_pos_edge = []

    def freeze_unused_weights(self, model, mask):
        grad_mask = torch.zeros_like(mask)
        grad_mask[mask] = 1

        model.deletion1.deletion_weight.register_hook(lambda grad: grad.mul_(grad_mask))
        model.deletion2.deletion_weight.register_hook(lambda grad: grad.mul_(grad_mask))
    
    @torch.no_grad()
    def get_link_labels(self, pos_edge_index, neg_edge_index):
        E = pos_edge_index.size(1) + neg_edge_index.size(1)
        link_labels = torch.zeros(E, dtype=torch.float, device=pos_edge_index.device)
        link_labels[:pos_edge_index.size(1)] = 1.
        return link_labels

    @torch.no_grad()
    def get_embedding(self, model, data, on_cpu=False):
        original_device = next(model.parameters()).device

        if on_cpu:
            model = model.cpu()
            data = data.cpu()
        
        z = model(data.x, data.train_pos_edge_index[:, data.dtrain_mask])

        model = model.to(original_device)

        return z

    def train(self, model, data, optimizer, args):
        if self.args.dataset in ['Cora_p']: # add more here
            return self.train_fullbatch(model, data, optimizer, args)

    def train_fullbatch(self, model, data, optimizer, args):
        data = data.to(device)
        model = model.to(device)
        start_time = time.time()
        best_valid_loss = 1000000

        data = data.to(device)
        for epoch in trange(self.num_epochs, desc='Epoch'):
            model.train()

            # Positive and negative sample
            neg_edge_index = negative_sampling(
                edge_index=data.train_pos_edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=data.dtrain_mask.sum())
            
            z = model(data.x, data.train_pos_edge_index)
            # edge = torch.cat([train_pos_edge_index, neg_edge_index], dim=-1)
            # logits = model.decode(z, edge[0], edge[1])
            logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)
            label = get_link_labels(data.train_pos_edge_index, neg_edge_index)
            loss = F.binary_cross_entropy_with_logits(logits, label)

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()

            if (epoch+1) % args.valid_freq == 0:
                valid_loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, valid_log = self.eval(model, data, 'val')

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_epoch = epoch

        print(f'Training finished. Best checkpoint at epoch = {best_epoch:04d}, best valid loss = {best_valid_loss:.4f}')

    @torch.no_grad()
    def eval(self, model, data, stage='val', pred_all=False):
        model.eval()
        pos_edge_index = data[f'{stage}_pos_edge_index']
        neg_edge_index = data[f'{stage}_neg_edge_index']

        if self.args.eval_on_cpu:
            model = model.to('cpu')
        
        if hasattr(data, 'dtrain_mask'):
            mask = data.dtrain_mask
        else:
            mask = data.dr_mask
        z = model(data.x, data.train_pos_edge_index[:, mask])
        logits = model.decode(z, pos_edge_index, neg_edge_index).sigmoid()
        label = self.get_link_labels(pos_edge_index, neg_edge_index)

        # DT AUC AUP
        loss = F.binary_cross_entropy_with_logits(logits, label).cpu().item()
        dt_auc = roc_auc_score(label.cpu(), logits.cpu())
        dt_aup = average_precision_score(label.cpu(), logits.cpu())

        # DF AUC AUP
        if self.args.unlearning_model in ['original']:
            df_logit = []
        else:
            # df_logit = model.decode(z, data.train_pos_edge_index[:, data.df_mask]).sigmoid().tolist()
            df_logit = model.decode(z, data.directed_df_edge_index).sigmoid().tolist()
            
        if len(df_logit) > 0:
            df_auc = []
            df_aup = []
        
            # Sample pos samples
            if len(self.df_pos_edge) == 0:
                for i in range(5):
                    mask = torch.zeros(data.train_pos_edge_index[:, data.dr_mask].shape[1], dtype=torch.bool)
                    idx = torch.randperm(data.train_pos_edge_index[:, data.dr_mask].shape[1])[:len(df_logit)]
                    mask[idx] = True
                    self.df_pos_edge.append(mask)
            
            # Use cached pos samples
            for mask in self.df_pos_edge:
                pos_logit = model.decode(z, data.train_pos_edge_index[:, data.dr_mask][:, mask]).sigmoid().tolist()
                
                logit = df_logit + pos_logit
                label = [0] * len(df_logit) + [1] * len(pos_logit)
                df_auc.append(roc_auc_score(label, logit))
                df_aup.append(average_precision_score(label, logit))
        
            df_auc = np.mean(df_auc)
            df_aup = np.mean(df_aup)

        else:
            df_auc = np.nan
            df_aup = np.nan

        log = {
            f'{stage}_loss': loss,
            f'{stage}_dt_auc': dt_auc,
            f'{stage}_dt_aup': dt_aup,
            f'{stage}_df_auc': df_auc,
            f'{stage}_df_aup': df_aup,
            f'{stage}_df_logit_mean': np.mean(df_logit) if len(df_logit) > 0 else np.nan,
            f'{stage}_df_logit_std': np.std(df_logit) if len(df_logit) > 0 else np.nan
        }

        if self.args.eval_on_cpu:
            model = model.to(device)

        return loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, None, log

    @torch.no_grad()
    def test(self, model, data, model_retrain=None, attack_model_all=None, attack_model_sub=None, logits_before_unlearning=None):
        
        if 'ogbl' in self.args.dataset:
            pred_all = False
        else:
            pred_all = True

        loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, test_log = self.eval(model, data, 'test', pred_all)

        # MI Attack after unlearning
        if attack_model_all is not None:
            mi_logit_all_after, mi_sucrate_all_after = member_infer_attack(model, attack_model_all, data)
            # print(f'MI Attack succress rate (All) after unlearning: {mi_sucrate_all_after:.4f}')
            # print(f'MI Logit (All) after unlearning: {mi_logit_all_after}')
            # print("===============")

            sum = 0
            for i in range(0, len(logits_before_unlearning)):
                ratio = mi_logit_all_after[i][0] / logits_before_unlearning[i][0]
                # print(mi_logit_all_after[i][0], logits_before_unlearning[i][0], ratio)
                sum += ratio
            mi_score = sum / len(logits_before_unlearning)
            print("============")
            print("mi score: ", mi_score)
            print("============")
            return loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, test_log, mi_score

        if attack_model_sub is not None:
            mi_logit_sub_after, mi_sucrate_sub_after = member_infer_attack(model, attack_model_sub, data)
            

        return loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, test_log

    @torch.no_grad()
    def get_output(self, model, node_embedding, data):
        model.eval()
        node_embedding = node_embedding.to(device)
        edge = data.edge_index.to(device)
        output = model.decode(node_embedding, edge)

        return output