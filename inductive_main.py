from framework.utils import get_original_data, seed_everything
from inductive_base import *
import torch_geometric.transforms as T
import warnings
from attacks.label_flip import label_flip_attack

random_seed=0
warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed_everything(random_seed)
# class1=1
# class2=8

class1=1
class2=6


def train_on_data(data, num_epochs, class1, class2, clean=True):
    inductive_graph_split(data)
    data= data.to(device)

    model = ClusterGCN(data.num_features, 64, data.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss = train(model, data, optimizer, num_epochs, device)

    train_acc, val_acc, test_acc, train_f1, val_f1, test_f1, train_subset, val_subset, test_subset = test(model, data, class1, class2, device)
    print("\n==CLEAN==" if clean else "\n==POISONED==")
    print(f'Train Acc: {train_acc:.4f} Val Acc: {val_acc:.4f} Test Acc: {test_acc:.4f}\nTrain F1: {train_f1:.4f} Val F1: {val_f1:.4f} Test F1: {test_f1:.4f}\n')
    print(f'Subset Acc: \nTrain: {train_subset}\nVal: {val_subset}\nTest: {test_subset}')

split= T.RandomNodeSplit(num_val=0.1, num_test=0.2)
clean_data = get_original_data("Amazon")
clean_data= split(clean_data)

train_on_data(clean_data, num_epochs=200, class1=class1, class2=class2)

poisoned_data, poisoned_indices = label_flip_attack(clean_data, 0.5, random_seed, class1, class2)
train_on_data(poisoned_data, num_epochs=200, class1=class1, class2=class2, clean=False)