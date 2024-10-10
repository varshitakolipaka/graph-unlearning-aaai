import os, math
import copy
import time
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling
import torch.nn.functional as F
from .base import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def BoundedKLDMean(logits, truth):
    return 1 - torch.exp(-F.kl_div(F.log_softmax(logits, -1), truth.softmax(-1), None, None, 'batchmean'))

def BoundedKLDSum(logits, truth):
    return 1 - torch.exp(-F.kl_div(F.log_softmax(logits, -1), truth.softmax(-1), None, None, 'sum'))

def CosineDistanceMean(logits, truth):
    return (1 - F.cosine_similarity(logits, truth)).mean()

def CosineDistanceSum(logits, truth):
    return (1 - F.cosine_similarity(logits, truth)).sum()

def centering(K):
    n = K.shape[0]
    unit = torch.ones([n, n], device=K.device)
    I = torch.eye(n, device=K.device)
    H = I - unit / n
    return torch.matmul(torch.matmul(H, K), H)

def rbf(X, sigma=None):
    GX = torch.matmul(X, X.T)
    KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
    if sigma is None:
        mdist = torch.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = torch.exp(KX)
    return KX

def kernel_HSIC(X, Y, sigma=None):
    return torch.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))

def linear_HSIC(X, Y):
    L_X = torch.matmul(X, X.T)
    L_Y = torch.matmul(Y, Y.T)
    return torch.sum(centering(L_X) * centering(L_Y))

def LinearCKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = torch.sqrt(linear_HSIC(X, X))
    var2 = torch.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)

def RBFCKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = torch.sqrt(kernel_HSIC(X, X, sigma))
    var2 = torch.sqrt(kernel_HSIC(Y, Y, sigma))
    return hsic / (var1 * var2)


def get_loss_fct(name):
    if name == 'kld_mean':
        loss_fct = BoundedKLDMean
    elif name == 'kld_sum':
        loss_fct = BoundedKLDSum
    elif name == 'mse_mean':
        loss_fct = nn.MSELoss(reduction='mean')
    elif name == 'mse_sum':
        loss_fct = nn.MSELoss(reduction='sum')
    elif name == 'cosine_mean':
        loss_fct = CosineDistanceMean
    elif name == 'cosine_sum':
        loss_fct = CosineDistanceSum
    elif name == 'linear_cka':
        loss_fct = LinearCKA
    elif name == 'rbf_cka':
        loss_fct = RBFCKA
    else:
        raise NotImplementedError
    return loss_fct

class GNNDeleteNodeembTrainer(Trainer):
    def __init__(self, model, data, optimizer, args):
        super().__init__(model, data, optimizer, args)

    def train(self):
        return self.train_fullbatch()

    def train_fullbatch(self):
        start_time = time.time()

        self.model = self.model.to(device)
        self.data = self.data.to(device)

        best_metric = 0
        loss_fct = get_loss_fct(self.args.loss_fct)
        non_df_node_mask = torch.ones(self.data.x.shape[0], dtype=torch.bool, device=self.data.x.device)
        non_df_node_mask[self.data.directed_df_edge_index.flatten().unique()] = False

        self.data.sdf_node_1hop_mask_non_df_mask = self.data.sdf_node_1hop_mask & non_df_node_mask
        self.data.sdf_node_2hop_mask_non_df_mask = self.data.sdf_node_2hop_mask & non_df_node_mask
        self.data.sdf_node_3hop_mask_non_df_mask = self.data.sdf_node_3hop_mask & non_df_node_mask

        with torch.no_grad():
            z1_ori, z2_ori, z3_ori = self.model.get_original_embeddings(self.data.x, self.data.train_pos_edge_index, return_all_emb=True)


        for epoch in trange(self.args.unlearning_epochs, desc='Unlearning'):
            iter_start_time = time.time()
            self.model.train()

            neg_edge = negative_sampling(
                edge_index=self.data.train_pos_edge_index,
                num_nodes=self.data.num_nodes,
                num_neg_samples=self.data.df_mask.sum())

            # start_time = time.time()

            # Forward pass
            z1, z2, z3 = self.model(self.data.x, self.data.train_pos_edge_index[:, self.data.sdf_mask], return_all_emb=True)

            pos_edge = self.data.train_pos_edge_index[:, self.data.df_mask]
            embed1 = torch.cat([z1[pos_edge[0]], z1[pos_edge[1]]], dim=0)
            embed1_ori = torch.cat([z1_ori[neg_edge[0]], z1_ori[neg_edge[1]]], dim=0)
            embed2 = torch.cat([z2[pos_edge[0]], z2[pos_edge[1]]], dim=0)
            embed2_ori = torch.cat([z2_ori[neg_edge[0]], z2_ori[neg_edge[1]]], dim=0)
            embed3 = torch.cat([z3[pos_edge[0]], z3[pos_edge[1]]], dim=0)
            embed3_ori = torch.cat([z3_ori[neg_edge[0]], z3_ori[neg_edge[1]]], dim=0)

            loss_r1 = loss_fct(embed1, embed1_ori)
            loss_r2 = loss_fct(embed2, embed2_ori)
            loss_r3 = loss_fct(embed3, embed3_ori)

            loss_l1 = loss_fct(z1[self.data.sdf_node_1hop_mask_non_df_mask], z1_ori[self.data.sdf_node_1hop_mask_non_df_mask])
            loss_l2 = loss_fct(z2[self.data.sdf_node_2hop_mask_non_df_mask], z2_ori[self.data.sdf_node_2hop_mask_non_df_mask])
            loss_l3 = loss_fct(z3[self.data.sdf_node_3hop_mask_non_df_mask], z3_ori[self.data.sdf_node_3hop_mask_non_df_mask])


            if self.args.loss_type == 'both_all':
                loss_l = loss_l1 + loss_l2 + loss_l3
                loss_r = loss_r1 + loss_r2 + loss_r3
                loss = self.args.alpha * loss_r + (1 - self.args.alpha) * loss_l
                loss.backward()
                self.optimizer.step()

            elif self.args.loss_type == 'both_layerwise':
                loss_l = loss_l1 + loss_l2 + loss_l3
                loss_r = loss_r1 + loss_r2 + loss_r3

                loss1 = self.args.alpha * loss_r1 +  (1 - self.args.alpha) * loss_l1
                loss1.backward(retain_graph=True)

                loss2 = self.args.alpha * loss_r2 + (1 - self.args.alpha) * loss_l2
                loss2.backward(retain_graph=True)

                loss3 = self.args.alpha * loss_r3 + (1 - self.args.alpha) * loss_l3
                loss3.backward(retain_graph=True)

                self.optimizer[0].step()
                self.optimizer[0].zero_grad()
                self.optimizer[1].step()
                self.optimizer[1].zero_grad()
                self.optimizer[2].step()
                self.optimizer[2].zero_grad()

                loss = loss1 + loss2 + loss3


            elif self.args.loss_type == 'only2_layerwise':
                loss_l = loss_l1 + loss_l2 + loss_l3
                loss_r = loss_r1 + loss_r2 + loss_r3

                self.optimizer[1].zero_grad()
                loss2 = self.args.alpha * loss_r2 + (1 - self.args.alpha) * loss_l2
                loss2.backward()
                self.optimizer[1].step()
                self.optimizer[1].zero_grad()

                loss = loss2

            elif self.args.loss_type == 'only3_layerwise':
                loss_l = loss_l1 + loss_l2 + loss_l3
                loss_r = loss_r1 + loss_r2 + loss_r3

                self.optimizer[2].zero_grad()
                loss3 = self.args.alpha * loss_r3 + (1 - self.args.alpha) * loss_l3
                loss3.backward()
                self.optimizer[2].step()
                self.optimizer[2].zero_grad()
                loss = loss3

            elif self.args.loss_type == 'only2_all':
                loss_l = loss_l2
                loss_r = loss_r2

                loss = loss_l + self.args.alpha * loss_r

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            elif self.args.loss_type == 'only3_all':
                loss_l = loss_l3
                loss_r = loss_r3

                loss = loss_l + self.args.alpha * loss_r

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()


            elif self.args.loss_type == 'only1':
                loss_l = loss_l1
                loss_r = loss_r1

                loss = loss_l + self.args.alpha * loss_r

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            elif self.args.loss_type == 'only3':
                loss_l = loss_l3
                loss_r = loss_r3

                loss = loss_l + self.args.alpha * loss_r

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            else:
                raise NotImplementedError
            
            self.unlearning_time += time.time() - iter_start_time
            if epoch % 5 == 0:
                cutoff = self.save_best()
                if cutoff:
                    break

            # end_time = time.time()
            # epoch_time = end_time - start_time
        
        end_time = time.time()
        self.load_best()
        train_acc, msc_rate, f1 = self.evaluate(is_dr=True, use_val=True)
        print(f'Train Acc: {train_acc}, Misclassification: {msc_rate},  F1 Score: {f1}')

        return train_acc, msc_rate, self.unlearning_time