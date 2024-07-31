import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from tqdm import trange

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
class Trainer:
    def __init__(self, model, data, optimizer, num_epochs=50):
        self.model = model.to(device)
        self.data = data.to(device)
        self.optimizer = optimizer
        self.num_epochs= num_epochs

    def train(self):
        self.data = self.data.to(device)
        for epoch in trange(self.num_epochs, desc='Epoch'):
            self.model.train()
            z = F.log_softmax(self.model(self.data.x, self.data.edge_index), dim=1)
            loss = F.nll_loss(z[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if epoch % 10 == 0:
                acc, msc_rate, f1 = self.evaluate()
        train_acc, msc_rate, f1 = self.evaluate()
        print(f'Train Acc: {train_acc}, Misclassification: {msc_rate},  F1 Score: {f1}')
        return train_acc, msc_rate, f1

    def misclassification_rate(self, true_labels, pred_labels, class1 = 0, class2 = 1):
        class1_to_class2 = ((true_labels == class1) & (pred_labels == class2)).sum().item()
        class2_to_class1 = ((true_labels == class2) & (pred_labels == class1)).sum().item()
        total_class1 = (true_labels == class1).sum().item()
        total_class2 = (true_labels == class2).sum().item()
        misclassification_rate = (class1_to_class2 + class2_to_class1) / (total_class1 + total_class2)
        return misclassification_rate

    def evaluate(self, is_dr=False):
        self.model.eval()
        with torch.no_grad():
            if(is_dr):
                z = F.log_softmax(self.model(self.data.x, self.data.edge_index[:, self.data.dr_mask]), dim=1)
            else:
                z = F.log_softmax(self.model(self.data.x, self.data.edge_index), dim=1)
            loss = F.nll_loss(z[self.data.val_mask], self.data.y[self.data.val_mask]).cpu().item()
            pred = torch.argmax(z[self.data.val_mask], dim=1).cpu()
            dt_acc = accuracy_score(self.data.y[self.data.val_mask].cpu(), pred)
            dt_f1 = f1_score(self.data.y[self.data.val_mask].cpu(), pred, average='micro')
            msc_rate = self.misclassification_rate(self.data.y[self.data.val_mask].cpu(), pred)
        return dt_acc, msc_rate, dt_f1