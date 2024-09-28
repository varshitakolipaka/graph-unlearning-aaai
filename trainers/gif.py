import copy
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
DTYPE = np.float16

@torch.no_grad()
def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=pos_edge_index.device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

class GIFTrainer(Trainer):
    def __init__(self, model, data, optimizer, args):
        super().__init__(model, data, optimizer, args)
    '''This code is adapted from https://github.com/zleizzo/datadeletion'''

    def get_grad(self, data, model):
        print("Computing grads...")
        neg_edge_index = negative_sampling(
            edge_index=data.train_pos_edge_index[:, data.dr_mask],
            num_nodes=data.num_nodes,
            num_neg_samples=data.sdf_mask.sum())

        grad_all, grad1, grad2 = None, None, None
        out1 = model(data.x, data.train_pos_edge_index)
        out2 = model(data.x, data.train_pos_edge_index[:, data.dr_mask])

        # mask1 = mask2 = data.sdf_node_2hop_mask

        logits = model.decode(out1, data.train_pos_edge_index[:, data.dr_mask], neg_edge_index)
        label = get_link_labels(data.train_pos_edge_index[:, data.dr_mask], neg_edge_index)
        loss = F.binary_cross_entropy_with_logits(logits, label, reduction='mean')
        # loss = F.nll_loss(out1, data.y, reduction='sum')

        logits = model.decode(out1, data.train_pos_edge_index[:, data.sdf_mask], neg_edge_index)
        label = get_link_labels(data.train_pos_edge_index[:, data.sdf_mask], neg_edge_index)
        loss1 = F.binary_cross_entropy_with_logits(logits, label, reduction='sum')
        # loss1 = F.nll_loss(out1[mask1], data.y[mask1], reduction='sum')

        logits = model.decode(out2, data.train_pos_edge_index[:, data.sdf_mask], neg_edge_index)
        label = get_link_labels(data.train_pos_edge_index[:, data.sdf_mask], neg_edge_index)
        loss2 = F.binary_cross_entropy_with_logits(logits, label, reduction='sum')
        # loss2 = F.nll_loss(out2[mask2], data.y[mask2], reduction='sum')

        model_params = [p for p in model.parameters() if p.requires_grad]
        grad_all = grad(loss, model_params, retain_graph=True, create_graph=True)
        grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True)
        grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True)

        return (grad_all, grad1, grad2)

    def gif_approxi(self, args, model, res_tuple):
        print("Unlearning model...")
        start_time = time.time()
        iteration, damp, scale = self.args.iteration, self.args.damp, self.args.scale

        v = tuple(grad1 - grad2 for grad1, grad2 in zip(res_tuple[1], res_tuple[2]))
        h_estimate = tuple(grad1 - grad2 for grad1, grad2 in zip(res_tuple[1], res_tuple[2]))
        for _ in range(iteration):

            model_params  = [p for p in model.parameters() if p.requires_grad]
            hv            = self.hvps(res_tuple[0], model_params, h_estimate)
            with torch.no_grad():
                h_estimate    = [ v1 + (1-damp)*h_estimate1 - hv1/scale
                            for v1, h_estimate1, hv1 in zip(v, h_estimate, hv)]

        # breakpoint()
        params_change = [h_est / scale for h_est in h_estimate]
        params_esti   = [p1 + p2 for p1, p2 in zip(params_change, model_params)]

        temp= copy.deepcopy(self.model)
        idx = 0
        for p in temp.parameters():
            p.data = params_esti[idx]
            idx = idx + 1
        self.model= temp

        return time.time() - start_time, model

    def hvps(self, grad_all, model_params, h_estimate):
        element_product = 0
        for grad_elem, v_elem in zip(grad_all, h_estimate):
            element_product += torch.sum(grad_elem * v_elem)

        return_grads = grad(element_product, model_params, create_graph=True)
        return return_grads

    def train(self, logits_ori=None, attack_model=None, attack_model_sub=None):
        # model.train()
        self.model, self.data = self.model.to(device), self.data.to(device)
        self.args.eval_on_cpu = False

        grad_tuple = self.get_grad(self.data, self.model)

        time, model = self.gif_approxi(self.args, self.model, grad_tuple)
        z = model(self.data.x, self.data.train_pos_edge_index[:, self.data.dr_mask])

        train_acc, msc_rate, f1 = self.evaluate(is_dr = True, use_val = True)
        # print(f'Train Acc: {train_acc}, Misclassification: {msc_rate},  F1 Score: {f1}')

        return train_acc, msc_rate, time
