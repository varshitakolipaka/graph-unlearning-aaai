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

import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

torch.cuda.empty_cache()
from torch_geometric.nn import CorrectAndSmooth
import numpy as np
from torch_geometric.utils import k_hop_subgraph, to_scipy_sparse_matrix
import scipy.sparse as sp
from sklearn.metrics import f1_score

import logging
class GATE(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lr = torch.nn.Linear(dim, dim)

    def forward(self, x):
        t = x.clone()
        return self.lr(t)


def criterionKD(p, q, T=1.5):
    loss_kl = nn.KLDivLoss(reduction="batchmean")
    soft_p = F.log_softmax(p / T, dim=1)
    soft_q = F.softmax(q / T, dim=1).detach()
    return loss_kl(soft_p, soft_q)

def calc_f1(y_true, y_pred, mask, multilabel=False):
    if multilabel:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    else:
        y_pred = np.argmax(y_pred, axis=1)
    mask = mask.cpu()
    return f1_score(y_true[mask], y_pred[mask], average="micro")

def propagate(features, k, adj_norm):
    feature_list = []
    feature_list.append(features)
    for i in range(k):
        feature_list.append(torch.spmm(adj_norm.to(device), feature_list[-1].to(device)))
    return feature_list[-1]


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize_adj(adj, r=0.5):
    adj = adj + sp.eye(adj.shape[0])
    degrees = np.array(adj.sum(1))
    r_inv_sqrt_left = np.power(degrees, r - 1).flatten()
    r_inv_sqrt_left[np.isinf(r_inv_sqrt_left)] = 0.
    r_mat_inv_sqrt_left = sp.diags(r_inv_sqrt_left)

    r_inv_sqrt_right = np.power(degrees, -r).flatten()
    r_inv_sqrt_right[np.isinf(r_inv_sqrt_right)] = 0.
    r_mat_inv_sqrt_right = sp.diags(r_inv_sqrt_right)

    adj_normalized = adj.dot(r_mat_inv_sqrt_left).transpose().dot(r_mat_inv_sqrt_right)
    return adj_normalized


class MeguTrainer(Trainer):
    def __init__(self, model, poisoned_dataset, optimizer, args):
        super().__init__(model, poisoned_dataset, optimizer, args)
        self.logger = logging.getLogger('ExpMEGU')
        self.data = poisoned_dataset
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_feats = self.data.num_features
        self.train_test_split()
        self.unlearning_request()
        self.model = model

        self.num_layers = 2
        self.adj = sparse_mx_to_torch_sparse_tensor(normalize_adj(to_scipy_sparse_matrix(self.data.edge_index)))

    def train_test_split(self):
        if hasattr(self.data, 'train_mask') and hasattr(self.data, 'test_mask'):
            # Extract indices from existing masks
            self.train_indices = np.where(self.data.train_mask.cpu().numpy())[0]
            self.test_indices = np.where(self.data.test_mask.cpu().numpy())[0]
        else:
            raise ValueError("Train/test masks do not exist in the dataset.")


    def unlearning_request(self):
        self.data.x_unlearn = self.data.x.clone()
        self.data.edge_index_unlearn = self.data.edge_index.clone()

        if hasattr(self.data, 'poisoned_nodes'):
            # if self.data.df_mask.dim() == 1 and self.data.df_mask.size(0) == self.data.num_nodes:
            #     # Node-level mask
            #     unique_nodes = torch.where(self.data.df_mask)[0].cpu().numpy()
            # elif self.data.df_mask.dim() == 1 and self.data.df_mask.size(0) == self.data.edge_index.shape[1]:
            #     # Edge-level mask
            #     remove_indices = torch.where(self.data.df_mask)[0].cpu().numpy()
            #     edge_index = self.data.edge_index.cpu().numpy()
            #     remove_edges = edge_index[:, remove_indices]
            #     unique_nodes = np.unique(remove_edges)
            # else:
            #     raise ValueError("Unexpected shape for df_mask")
            poisoned_nodes = self.data.poisoned_nodes
            if type(poisoned_nodes) == torch.Tensor:
                unique_nodes = poisoned_nodes.cpu().numpy()
            else:
                unique_nodes = poisoned_nodes
            
            self.data.edge_index_unlearn = self.update_edge_index_unlearn(unique_nodes)
            
            if self.args.request == 'feature':
                self.data.x_unlearn[unique_nodes] = 0.

            self.temp_node = unique_nodes
        else:
            raise ValueError("df_mask not found in data object")

    def update_edge_index_unlearn(self, delete_nodes, delete_edge_index=None):
        edge_index = self.data.edge_index.cpu().numpy()

        unique_indices = np.where(edge_index[0] < edge_index[1])[0]
        unique_indices_not = np.where(edge_index[0] > edge_index[1])[0]

        if self.args.request == 'edge':
            remain_indices = np.setdiff1d(unique_indices, delete_edge_index)
        else:
            unique_edge_index = edge_index[:, unique_indices]
            delete_edge_indices = np.logical_or(np.isin(unique_edge_index[0], delete_nodes),
                                                np.isin(unique_edge_index[1], delete_nodes))
            remain_indices = np.logical_not(delete_edge_indices)
            remain_indices = np.where(remain_indices == True)[0]  # Ensure it's a 1D array of indices

        remain_encode = edge_index[0, remain_indices] * edge_index.shape[1] * 2 + edge_index[1, remain_indices]
        unique_encode_not = edge_index[1, unique_indices_not] * edge_index.shape[1] * 2 + edge_index[0, unique_indices_not]
        sort_indices = np.argsort(unique_encode_not)
        
        search_indices = np.searchsorted(unique_encode_not, remain_encode, sorter=sort_indices)
        valid_search_indices = search_indices[search_indices < len(sort_indices)]
        
        remain_indices_not = unique_indices_not[sort_indices[valid_search_indices]]
        remain_indices = np.union1d(remain_indices, remain_indices_not)
        self.data.dr_mask = torch.zeros(self.data.edge_index.shape[1], dtype=torch.bool)
        self.data.dr_mask[remain_indices] = True
        return torch.from_numpy(edge_index[:, remain_indices])

    # def evaluate(self, run):
    #     # self.logger.info('model evaluation')

    #     start_time = time.time()
    #     self.model.eval()
    #     out = self.model(self.data.x, self.data.edge_index)
    #     y = self.data.y.cpu()
    #     if self.args.dataset == 'ppi':
    #         y_hat = torch.sigmoid(out).cpu().detach().numpy()
    #         test_f1 = calc_f1(y, y_hat, self.data.test_mask, multilabel=True)
    #     else:
    #         y_hat = F.log_softmax(out, dim=1).cpu().detach().numpy()
    #         test_f1 = calc_f1(y, y_hat, self.data.test_mask)

    #     evaluate_time = time.time() - start_time
    #     # self.logger.info(f"Evaluation cost {evaluate_time:.4f} seconds.")

    #     # self.logger.info(f"Final Test F1: {test_f1:.4f}")
    #     return test_f1

    def neighbor_select(self, features):
        temp_features = features.clone()
        pfeatures = propagate(temp_features, self.num_layers, self.adj)
        reverse_feature = self.reverse_features(temp_features)
        re_pfeatures = propagate(reverse_feature, self.num_layers, self.adj)

        cos = nn.CosineSimilarity()
        sim = cos(pfeatures, re_pfeatures)
        
        alpha = 0.1
        gamma = 0.1
        max_val = 0.
        while True:
            influence_nodes_with_unlearning_nodes = torch.nonzero(sim <= alpha).flatten().cpu()
            if len(influence_nodes_with_unlearning_nodes.view(-1)) > 0:
                temp_max = torch.max(sim[influence_nodes_with_unlearning_nodes])
            else:
                alpha = alpha + gamma
                continue

            if temp_max == max_val:
                break

            max_val = temp_max
            alpha = alpha + gamma

        # influence_nodes_with_unlearning_nodes = torch.nonzero(sim < 0.5).squeeze().cpu()
        neighborkhop, _, _, two_hop_mask = k_hop_subgraph(
            torch.tensor(self.temp_node),
            self.num_layers,
            self.data.edge_index,
            num_nodes=self.data.num_nodes)

        neighborkhop = neighborkhop[~np.isin(neighborkhop.cpu(), self.temp_node)]
        neighbor_nodes = []
        for idx in influence_nodes_with_unlearning_nodes:
            if idx in neighborkhop and idx not in self.temp_node:
                neighbor_nodes.append(idx.item())
        
        neighbor_nodes_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), neighbor_nodes))

        return neighbor_nodes_mask

    # def reverse_features(self, features):
    #     reverse_features = features.clone()
    #     for idx in self.temp_node:
    #         reverse_features[idx] = 1 - reverse_features[idx]

    #     return reverse_features

    def reverse_features(self, features):
        reverse_features = features.clone()
        valid_indices = [idx for idx in self.temp_node if idx < features.shape[0]]
        reverse_features[valid_indices] = 1 - reverse_features[valid_indices]
        return reverse_features


    def correct_and_smooth(self, y_soft, preds):
        pos = CorrectAndSmooth(num_correction_layers=80, correction_alpha=self.args.alpha1,
                               num_smoothing_layers=80, smoothing_alpha=self.args.alpha2,
                               autoscale=False, scale=1.)

        y_soft = pos.correct(y_soft, preds[self.data.train_mask], self.data.train_mask,
                                  self.data.edge_index_unlearn)
        y_soft = pos.smooth(y_soft, preds[self.data.train_mask], self.data.train_mask,
                                 self.data.edge_index_unlearn)

        return y_soft

    def train(self):
        # a linear model
        operator = GATE(self.data.num_classes).to(self.device)

        # optimizer updates both models
        optimizer = torch.optim.SGD([
            {'params': self.model.parameters()},
            {'params': operator.parameters()}
        ], lr=self.args.unlearn_lr)

        # class associated with argmax of logits -> preds class
        with torch.no_grad():
            self.model.eval()
            preds = self.model(self.data.x, self.data.edge_index)
            if self.args.dataset == 'ppi':
                preds = torch.sigmoid(preds).ge(0.5)
                preds = preds.type_as(self.data.y)
            else:
                preds = torch.argmax(preds, axis=1).type_as(self.data.y)

        start_time = time.time()
        
        self.neighbor_khop = self.neighbor_select(self.data.x)
        
        for epoch in trange(self.args.unlearning_epochs):
            iter_start_time = time.time()
            self.model.train()
            operator.train()
            optimizer.zero_grad()
            self.data.x_unlearn = self.data.x_unlearn.to(self.device)
            self.data.edge_index_unlearn = self.data.edge_index_unlearn.to(self.device)
            out_ori = self.model(self.data.x_unlearn, self.data.edge_index_unlearn)
            out = operator(out_ori) # this is basically a linear layer on top of the original model

            if self.args.dataset == 'ppi':
                loss_u = criterionKD(out_ori[self.temp_node], out[self.temp_node]) - F.binary_cross_entropy_with_logits(out[self.temp_node], preds[self.temp_node])
                loss_r = criterionKD(out[self.neighbor_khop], out_ori[self.neighbor_khop]) + F.binary_cross_entropy_with_logits(out_ori[self.neighbor_khop], preds[self.neighbor_khop])
            else:
                loss_u = criterionKD(out_ori[self.temp_node], out[self.temp_node]) - F.cross_entropy(out[self.temp_node], preds[self.temp_node])
                loss_r = criterionKD(out[self.neighbor_khop], out_ori[self.neighbor_khop]) + F.cross_entropy(out_ori[self.neighbor_khop], preds[self.neighbor_khop])

            loss = self.args.kappa * loss_u + loss_r
            loss.backward()
            optimizer.step()
            
            self.unlearning_time += time.time() - iter_start_time
            if epoch % 5 == 0:
                cutoff = self.save_best()
                if cutoff:
                    break

        unlearn_time = self.best_model_time - start_time
        self.load_best()
        train_acc, msc_rate, f1 = self.evaluate(is_dr=True, use_val=True)
        # print(f'Train Acc: {train_acc}, Misclassification: {msc_rate},  F1 Score: {f1}')


        return train_acc, msc_rate, self.unlearning_time