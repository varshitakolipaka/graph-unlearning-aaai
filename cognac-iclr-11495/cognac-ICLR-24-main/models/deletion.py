import torch
import torch.nn as nn
import torch.nn.functional as F
from .models import GCN, GAT, GIN

class DeletionLayer(nn.Module):
    def __init__(self, dim, mask):
        super().__init__()
        self.dim = dim
        self.mask = mask
        self.deletion_weight = nn.Parameter(torch.ones(dim, dim) / 1000)

    def forward(self, x, mask=None):
        '''Only apply deletion operator to the local nodes identified by mask'''
        if mask is None:
            mask = self.mask
        if mask is not None:
            new_rep = x.clone()
            new_rep[mask] = torch.matmul(x[mask], self.deletion_weight)
            return new_rep
        return x

class DeletionLayerKG(nn.Module):
    def __init__(self, dim, mask):
        super().__init__()
        self.dim = dim
        self.mask = mask
        self.deletion_weight = nn.Parameter(torch.ones(dim, dim) / 1000)

    def forward(self, x, mask=None):
        '''Only apply deletion operator to the local nodes identified by mask'''
        if mask is None:
            mask = self.mask
        if mask is not None:
            new_rep = x.clone()
            new_rep[mask] = torch.matmul(new_rep[mask], self.deletion_weight)
            return new_rep
        return x

class GCNDelete(GCN):
    def __init__(self, in_dim, hidden_dim, out_dim, mask_1hop=None, mask_2hop=None, mask_3hop=None, **kwargs):
        super().__init__(in_dim, hidden_dim, out_dim)
        self.deletion1 = DeletionLayer(hidden_dim, mask_1hop)
        self.deletion2 = DeletionLayer(hidden_dim, mask_2hop)
        self.deletion3 = DeletionLayer(out_dim, mask_3hop)

        self.conv1.requires_grad = False
        self.conv2.requires_grad = False
        self.conv3.requires_grad = False

    def forward(self, x, edge_index, mask_1hop=None, mask_2hop=None, mask_3hop=None, return_all_emb=False):
        x1 = self.conv1(x, edge_index)
        x1 = self.deletion1(x1, mask_1hop)
        x1 = F.relu(x1)

        x2 = self.conv2(x1, edge_index)
        x2 = self.deletion2(x2, mask_2hop)
        x2 = F.relu(x2)

        x3 = self.conv3(x2, edge_index)
        x3 = self.deletion3(x3, mask_3hop)

        if return_all_emb:
            return x1, x2, x3
        return x3

    def get_original_embeddings(self, x, edge_index, return_all_emb=False):
        return super().forward(x, edge_index, return_all_emb)

class GATDelete(GAT):
    def __init__(self, in_dim, hidden_dim, out_dim, mask_1hop=None, mask_2hop=None, mask_3hop=None, **kwargs):
        super().__init__(in_dim, hidden_dim, out_dim)
        self.deletion1 = DeletionLayer(hidden_dim, mask_1hop)
        self.deletion2 = DeletionLayer(hidden_dim, mask_2hop)
        self.deletion3 = DeletionLayer(out_dim, mask_3hop)

        self.conv1.requires_grad = False
        self.conv2.requires_grad = False
        self.conv3.requires_grad = False

    def forward(self, x, edge_index, mask_1hop=None, mask_2hop=None, mask_3hop=None, return_all_emb=False):
        x1 = self.conv1(x, edge_index)
        x1 = self.deletion1(x1, mask_1hop)
        x1 = F.relu(x1)

        x2 = self.conv2(x1, edge_index)
        x2 = self.deletion2(x2, mask_2hop)
        x2 = F.relu(x2)

        x3 = self.conv3(x2, edge_index)
        x3 = self.deletion3(x3, mask_3hop)

        if return_all_emb:
            return x1, x2, x3
        return x3

    def get_original_embeddings(self, x, edge_index, return_all_emb=False):
        return super().forward(x, edge_index, return_all_emb)

class GINDelete(GIN):
    def __init__(self, in_dim, hidden_dim, out_dim, mask_1hop=None, mask_2hop=None, **kwargs):
        super().__init__(in_dim, hidden_dim, out_dim)
        self.deletion1 = DeletionLayer(hidden_dim, mask_1hop)
        self.deletion2 = DeletionLayer(out_dim, mask_2hop)

        self.conv1.requires_grad = False
        self.conv2.requires_grad = False

    def forward(self, x, edge_index, mask_1hop=None, mask_2hop=None, return_all_emb=False):
        with torch.no_grad():
            x1 = self.conv1(x, edge_index)
        x1 = self.deletion1(x1, mask_1hop)
        x = F.relu(x1)
        x2 = self.conv2(x, edge_index)
        x2 = self.deletion2(x2, mask_2hop)
        if return_all_emb:
            return x1, x2
        return x2

    def get_original_embeddings(self, x, edge_index, return_all_emb=False):
        return super().forward(x, edge_index, return_all_emb)


