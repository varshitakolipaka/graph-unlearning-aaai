import logging
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.autograd import grad
import numpy as np
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index, return_all_emb=False):
        x1 = self.conv1(x, edge_index)
        x = F.relu(x1)
        x2 = self.conv2(x, edge_index)
        if return_all_emb:
            return x1, x2
        return x2

    def decode(self, z, pos_edge_index, neg_edge_index=None):
        if neg_edge_index is not None:
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
            logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        else:
            edge_index = pos_edge_index
            logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

class NodeClassifier:
    def __init__(self, num_feats, num_classes, args, data=None):
        self.args = args
        self.logger = logging.getLogger('node_classifier')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GCN(num_feats, 16, num_classes).to(self.device)
        self.data = data.to(self.device)
        self.lr = 0.01
        self.decay = 0.0001

    def train_model(self, model, unlearn_info=None, ):
        self.model.train()
        optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.decay)

        for epoch in range(self.args['num_epochs']):
            self.model.train()
            optimizer.zero_grad()
            out = self.model(self.data.x, self.data.edge_index)
            loss = F.cross_entropy(out[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            optimizer.step()

        grad_all, grad1, grad2 = None, None, None
        if self.args["method"] in ["GIF", "IF"]:
            out1 = self.model(self.data.x, self.data.edge_index)
            out2 = self.model(self.data.x_unlearn, self.data.edge_index_unlearn)

            mask1, mask2 = None, None
            if self.args["unlearn_task"] == "edge":
                mask1 = torch.zeros(out1.size(0), dtype=torch.bool)
                mask1[unlearn_info[2]] = True
                mask2 = mask1
            if self.args["unlearn_task"] == "node":
                mask1 = torch.zeros(out1.size(0), dtype=torch.bool)
                mask1[unlearn_info[0]] = True
                mask1[unlearn_info[2]] = True
                mask2 = torch.zeros(out2.size(0), dtype=torch.bool)
                mask2[unlearn_info[2]] = True
            if self.args["unlearn_task"] == "feature":
                mask1 = torch.zeros(out1.size(0), dtype=torch.bool)
                mask1[unlearn_info[1]] = True
                mask1[unlearn_info[2]] = True
                mask2 = mask1

            loss = F.cross_entropy(out1[self.data.train_mask], self.data.y[self.data.train_mask], reduction='sum')
            loss1 = F.cross_entropy(out1[mask1], self.data.y[mask1], reduction='sum')
            loss2 = F.cross_entropy(out2[mask2], self.data.y[mask2], reduction='sum')
            model_params = [p for p in self.model.parameters() if p.requires_grad]
            grad_all = grad(loss, model_params, retain_graph=True, create_graph=True)
            grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True)
            grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True)

        return (grad_all, grad1, grad2)
