import torch
from torch_geometric.datasets import Planetoid, Coauthor, Flickr, Amazon
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F

# Load the dataset
dataset = Flickr("./flickr", pre_transform=T.NormalizeFeatures())
data = dataset[0]

# Define the GraphSAGE model with BatchNorm and Dropout
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)  # Add BatchNorm for first layer
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)  # Add BatchNorm for second layer
        self.conv3 = SAGEConv(hidden_dim, out_dim)
        self.dropout = dropout  # Add dropout layer

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = self.bn1(x)  # Apply BatchNorm after conv1
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)  # Apply Dropout

        x = self.conv2(x, data.edge_index)
        x = self.bn2(x)  # Apply BatchNorm after conv2
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)  # Apply Dropout

        x = self.conv3(x, data.edge_index)
        return torch.log_softmax(x, dim=-1)

# Training function
def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data)[data.train_mask]  # Only use the training nodes
    loss = F.nll_loss(out, data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Testing function
@torch.no_grad()
def test(model, data):
    model.eval()
    out = model(data)

    # Get predictions for train, validation, and test sets
    pred = out.argmax(dim=1)

    train_acc = (pred[data.train_mask] == data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()
    valid_acc = (pred[data.val_mask] == data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
    test_acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()

    return train_acc, valid_acc, test_acc

# Initialize the model and optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GraphSAGE(in_dim=data.num_node_features,
                  hidden_dim=128,  # Increased hidden layer size
                  out_dim=dataset.num_classes).to(device)
data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)  # Slightly increased learning rate
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)  # Learning rate scheduler

# Training loop
epochs = 200
for epoch in range(1, epochs + 1):
    loss = train(model, data, optimizer)
    train_acc, valid_acc, test_acc = test(model, data)
    # scheduler.step()  # Step the scheduler after each epoch

    if epoch % 10 == 0:
        print(f'Epoch: {epoch}/{epochs}, '
              f'Loss: {loss:.4f}, '
              f'Train Acc: {100 * train_acc:.2f}%, '
              f'Valid Acc: {100 * valid_acc:.2f}%, '
              f'Test Acc: {100 * test_acc:.2f}%')
