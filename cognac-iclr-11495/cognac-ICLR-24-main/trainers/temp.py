import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

torch.cuda.empty_cache()
from torch_geometric.nn import CorrectAndSmooth
import numpy as np
from exp.exp import Exp
from lib_gnn_model.node_classifier import NodeClassifier
from torch_geometric.utils import k_hop_subgraph, to_scipy_sparse_matrix
import scipy.sparse as sp
from lib_utils.utils import calc_f1

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


def propagate(features, k, adj_norm):
    feature_list = []
    feature_list.append(features)
    for i in range(k):
        feature_list.append(torch.spmm(adj_norm, feature_list[-1]))
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


class ExpMEGU(Exp):
    def __init__(self, args):
        super(ExpMEGU, self).__init__(args)

        self.logger = logging.getLogger('ExpMEGU')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.load_data()
        self.num_feats = self.data.num_features
        self.train_test_split()
        self.unlearning_request()

        self.target_model_name = self.args['target_model']

        # self.get_edge_indeces()
        self.determine_target_model()

        self.num_layers = 2
        self.adj = sparse_mx_to_torch_sparse_tensor(normalize_adj(to_scipy_sparse_matrix(self.data.edge_index)))
        self.neighbor_khop = self.neighbor_select(self.data.x)



        run_f1 = np.empty(0)
        run_f1_unlearning = np.empty(0)
        unlearning_times = np.empty(0)
        training_times = np.empty(0)
        for run in range(self.args['num_runs']):
            # self.logger.info("Run %d" % run)

            run_training_time, _ = self._train_model(run)

            f1_score = self.evaluate(run)
            run_f1 = np.append(run_f1, f1_score)
            training_times = np.append(training_times, run_training_time)

            # unlearning with MEGU
            unlearning_time, f1_score_unlearning = self.megu_training()
            unlearning_times = np.append(unlearning_times, unlearning_time)
            run_f1_unlearning = np.append(run_f1_unlearning, f1_score_unlearning)

        f1_score_unlearning_avg = str(np.average(run_f1_unlearning)).split('.')[1]
        f1_score_unlearning_std = str(np.std(run_f1_unlearning)).split('.')[1]
        unlearning_time_avg = np.average(unlearning_times)

        f1_score_unlearning_avg = '.'.join((f1_score_unlearning_avg[0:2], f1_score_unlearning_avg[2:4]))
        f1_score_unlearning_std = '.'.join((f1_score_unlearning_std[1:2], f1_score_unlearning_std[2:4]))
        self.logger.info(
            f"|Unlearn| f1_score: avg±std={f1_score_unlearning_avg}±{f1_score_unlearning_std} time: avg={np.average(unlearning_times):.4f}s")


    def unlearning_request(self):
        self.data.x_unlearn = self.data.x.clone()
        self.data.edge_index_unlearn = self.data.edge_index.clone()
        edge_index = self.data.edge_index.numpy()
        unique_indices = np.where(edge_index[0] < edge_index[1])[0]

        self.data.edge_index_unlearn = self.data.df_mask
        self.temp_node = self.data.poisoned_indices

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

    def reverse_features(self, features):
        reverse_features = features.clone()
        for idx in self.temp_node:
            reverse_features[idx] = 1 - reverse_features[idx]

        return reverse_features

    def megu_training(self):
        operator = GATE(self.data.num_classes).to(self.device)

        optimizer = torch.optim.SGD([
            {'params': self.target_model.model.parameters()},
            {'params': operator.parameters()}
        ], lr=self.args['unlearn_lr'])


        with torch.no_grad():
            self.target_model.model.eval()
            preds = self.target_model.model(self.data.x, self.data.edge_index)
            if self.args['dataset_name'] == 'ppi':
                preds = torch.sigmoid(preds).ge(0.5)
                preds = preds.type_as(self.data.y)
            else:
                preds = torch.argmax(preds, axis=1).type_as(self.data.y)


        start_time = time.time()
        for epoch in range(30):
            self.target_model.model.train()
            operator.train()
            optimizer.zero_grad()
            out_ori = self.target_model.model(self.data.x_unlearn, self.data.edge_index_unlearn)
            out = operator(out_ori)

            if self.args['dataset_name'] == 'ppi':
                loss_u = criterionKD(out_ori[self.temp_node], out[self.temp_node]) - F.binary_cross_entropy_with_logits(out[self.temp_node], preds[self.temp_node])
                loss_r = criterionKD(out[self.neighbor_khop], out_ori[self.neighbor_khop]) + F.binary_cross_entropy_with_logits(out_ori[self.neighbor_khop], preds[self.neighbor_khop])
            else:
                loss_u = criterionKD(out_ori[self.temp_node], out[self.temp_node]) - F.cross_entropy(out[self.temp_node], preds[self.temp_node])
                loss_r = criterionKD(out[self.neighbor_khop], out_ori[self.neighbor_khop]) + F.cross_entropy(out_ori[self.neighbor_khop], preds[self.neighbor_khop])

            loss = self.args['kappa'] * loss_u + loss_r

            loss.backward()
            optimizer.step()

        unlearn_time = time.time() - start_time
        self.target_model.model.eval()
        test_out = self.target_model.model(self.data.x_unlearn, self.data.edge_index_unlearn)
        if self.args['dataset_name'] == 'ppi':
            out = torch.sigmoid(test_out)
        else:
            out = self.correct_and_smooth(F.softmax(test_out, dim=-1), preds)

        y_hat = out.cpu().detach().numpy()
        y = self.data.y.cpu()
        if self.args['dataset_name'] == 'ppi':
            test_f1 = calc_f1(y, y_hat, self.data.test_mask, multilabel=True)
        else:
            test_f1 = calc_f1(y, y_hat, self.data.test_mask)


        return unlearn_time, test_f1