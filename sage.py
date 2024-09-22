import argparse
import os.path as osp

import torch
import torch.nn.functional as F

from torch_geometric.datasets import Flickr
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.nn import GraphConv
from torch_geometric.typing import WITH_TORCH_SPARSE
from torch_geometric.utils import degree
from attacks.label_flip import label_flip_attack
from framework.training_args import parse_args
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

args = parse_args()

if not WITH_TORCH_SPARSE:
    quit("This example requires 'torch-sparse'")

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Flickr')
dataset = Flickr(path)
data = dataset[0]
row, col = data.edge_index
data.edge_weight = 1. / degree(col, data.num_nodes)[col]  # Norm by in-degree.
data, poisoned_indices = label_flip_attack(
        data, args.df_size, args.random_seed, 4, 6
    )
parser = argparse.ArgumentParser()
parser.add_argument('--use_normalization', action='store_true')
args = parser.parse_args()

loader = GraphSAINTRandomWalkSampler(data, batch_size=6000, walk_length=2,
                                     num_steps=5, sample_coverage=100,
                                     save_dir=dataset.processed_dir,
                                     num_workers=4)

class Net(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        in_channels = dataset.num_node_features
        out_channels = dataset.num_classes
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(3 * hidden_channels, out_channels)

    def set_aggr(self, aggr):
        self.conv1.aggr = aggr
        self.conv2.aggr = aggr
        self.conv3.aggr = aggr

    def forward(self, x0, edge_index, edge_weight=None):
        x1 = F.relu(self.conv1(x0, edge_index, edge_weight))
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x2 = F.relu(self.conv2(x1, edge_index, edge_weight))
        x2 = F.dropout(x2, p=0.2, training=self.training)
        x3 = F.relu(self.conv3(x2, edge_index, edge_weight))
        x3 = F.dropout(x3, p=0.2, training=self.training)
        x = torch.cat([x1, x2, x3], dim=-1)
        x = self.lin(x)
        return x.log_softmax(dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = Net(hidden_channels=256).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()
    model.set_aggr('add' if args.use_normalization else 'mean')

    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        if args.use_normalization:
            edge_weight = data.edge_norm * data.edge_weight
            out = model(data.x, data.edge_index, edge_weight)
            loss = F.nll_loss(out, data.y, reduction='none')
            loss = (loss * data.node_norm)[data.train_mask].sum()
        else:
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_nodes
        total_examples += data.num_nodes
    return total_loss / total_examples


@torch.no_grad()
@torch.no_grad()
def test():
    model.eval()
    model.set_aggr('mean')

    out = model(data.x.to(device), data.edge_index.to(device))
    pred = out.argmax(dim=-1)
    correct = pred.eq(data.y.to(device))

    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(correct[mask].sum().item() / mask.sum().item())

    # === Start of Additional Code ===

    # Retrieve class1 and class2
    class1 = data.class1
    class2 = data.class2

    # Ensure class1 and class2 are on the same device as data.y
    class1 = torch.tensor(class1, device=device)
    class2 = torch.tensor(class2, device=device)

    # Define the test mask
    test_mask = data.test_mask.to(device)

    # Create masks for class1, class2, and others within the test set
    class1_mask = (data.y.to(device) == class1) & test_mask
    class2_mask = (data.y.to(device) == class2) & test_mask
    others_mask = ((data.y.to(device) != class1) & (data.y.to(device) != class2)) & test_mask

    # Calculate accuracy for class1
    if class1_mask.sum().item() > 0:
        acc_class1 = correct[class1_mask].sum().item() / class1_mask.sum().item()
    else:
        acc_class1 = float('nan')  # Handle case with no samples

    # Calculate accuracy for class2
    if class2_mask.sum().item() > 0:
        acc_class2 = correct[class2_mask].sum().item() / class2_mask.sum().item()
    else:
        acc_class2 = float('nan')  # Handle case with no samples

    # Calculate accuracy for other classes
    if others_mask.sum().item() > 0:
        acc_others = correct[others_mask].sum().item() / others_mask.sum().item()
    else:
        acc_others = float('nan')  # Handle case with no samples

    # Optional: Print or store the separate accuracies
    print(f'Accuracy on class {class1.item()}: {acc_class1:.4f}')
    print(f'Accuracy on class {class2.item()}: {acc_class2:.4f}')
    print(f'Accuracy on other classes: {acc_others:.4f}')

    # === End of Additional Code ===

    return accs, acc_class1, acc_class2, acc_others


for epoch in range(1, 50):
    loss = train()
    accs, acc_class1, acc_class2, acc_others = test()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, '
          f'Train: {accs[0]:.4f}, Val: {accs[1]:.4f}, '
          f'Test: {accs[2]:.4f}, '
          f'Class {data.class1}: {acc_class1:.4f}, '
          f'Class {data.class2}: {acc_class2:.4f}, '
          f'Others: {acc_others:.4f}')
