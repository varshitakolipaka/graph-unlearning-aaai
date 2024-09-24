import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, ClusterGCNConv
from sklearn.metrics import accuracy_score, f1_score
import torch_geometric.transforms as T
from torch_geometric.utils import subgraph

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

class ClusterGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = ClusterGCNConv(in_channels, hidden_channels)
        self.conv2 = ClusterGCNConv(hidden_channels, hidden_channels)
        self.conv3 = ClusterGCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

def inductive_graph_split(data):
    train_edge_index, _ = subgraph(data.train_mask, data.edge_index)
    data.train_edge_index = train_edge_index

    val_edge_index, _ = subgraph(data.val_mask, data.edge_index)
    data.val_edge_index = val_edge_index

    test_edge_index, _ = subgraph(data.test_mask, data.edge_index)
    data.test_edge_index = test_edge_index

def train(model, data, optimizer, num_epoch, device):
    model.train()
    for i in range(num_epoch):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.train_edge_index)

        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

def subset_acc(model, data, mask, edge_index, class1, class2):
    model.eval()
    with torch.no_grad():
        z = model(data.x, edge_index)
        z = F.log_softmax(z, dim=1)
        true = data.y[mask].cpu()
        pred = torch.argmax(z[mask], dim=1).cpu()

        poisoned_classes = [class1, class2]
        clean_classes = [i for i in range(data.num_classes) if i not in poisoned_classes]

        accs_poisoned = []
        accs_clean = []

        for poisoned_class in poisoned_classes:
            poisoned_indices = true == poisoned_class
            if poisoned_indices.sum() > 0:
                accs_poisoned.append(accuracy_score(true[poisoned_indices], pred[poisoned_indices]))

        for clean_class in clean_classes:
            clean_indices = true == clean_class
            if clean_indices.sum() > 0:
                accs_clean.append(accuracy_score(true[clean_indices], pred[clean_indices]))

        accs_poisoned = sum(accs_poisoned) / len(accs_poisoned) if accs_poisoned else 0
        accs_clean = sum(accs_clean) / len(accs_clean) if accs_clean else 0

        return accs_poisoned, accs_clean

@torch.no_grad()
def test(model, data, class1, class2, device):
    model.eval()
    data_gpu = data.to(device)

    # Train
    out = model(data_gpu.x, data_gpu.train_edge_index)
    y_pred = out.argmax(dim=-1)
    train_acc = accuracy_score(data_gpu.y[data_gpu.train_mask].cpu(), y_pred[data_gpu.train_mask].cpu())
    train_f1 = f1_score(data_gpu.y[data_gpu.train_mask].cpu(), y_pred[data_gpu.train_mask].cpu(), average='macro')
    train_subset_acc = subset_acc(model, data_gpu, data_gpu.train_mask, data.train_edge_index, class1, class2)

    # Val
    out = model(data_gpu.x, data_gpu.val_edge_index)
    y_pred = out.argmax(dim=-1)
    val_acc = accuracy_score(data_gpu.y[data_gpu.val_mask].cpu(), y_pred[data_gpu.val_mask].cpu())
    val_f1 = f1_score(data_gpu.y[data_gpu.val_mask].cpu(), y_pred[data_gpu.val_mask].cpu(), average='macro')
    val_subset_acc = subset_acc(model, data_gpu, data_gpu.val_mask, data.val_edge_index, class1, class2)

    # Test
    out = model(data_gpu.x, data_gpu.test_edge_index)
    y_pred = out.argmax(dim=-1)
    test_acc = accuracy_score(data_gpu.y[data_gpu.test_mask].cpu(), y_pred[data_gpu.test_mask].cpu())
    test_f1 = f1_score(data_gpu.y[data_gpu.test_mask].cpu(), y_pred[data_gpu.test_mask].cpu(), average='macro')
    test_subset_acc = subset_acc(model, data_gpu, data_gpu.test_mask, data.test_edge_index, class1, class2)

    return (train_acc, val_acc, test_acc, train_f1, val_f1, test_f1,
            train_subset_acc, val_subset_acc, test_subset_acc)

