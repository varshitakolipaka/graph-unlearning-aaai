import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from tqdm import trange
from sklearn.metrics import classification_report

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
class Trainer:
    def __init__(self, model, data, optimizer, num_epochs=50):
        self.model = model.to(device)
        self.data = data.to(device)
        self.optimizer = optimizer
        self.num_epochs= num_epochs
        if hasattr(data, 'class1') and hasattr(data, 'class2'):
            self.class1 = data.class1
            self.class2 = data.class2
        else:
            self.class1 = None
            self.class2 = None

    def train(self):
        self.data = self.data.to(device)
        for epoch in trange(self.num_epochs, desc='Epoch'):
            self.model.train()
            z = F.log_softmax(self.model(self.data.x, self.data.edge_index), dim=1)
            loss = F.nll_loss(z[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        train_acc, msc_rate, f1 = self.evaluate()
        print(f'Train Acc: {train_acc}, Misclassification: {msc_rate},  F1 Score: {f1}')

        return train_acc, msc_rate

    def all_class_acc(self):
        classes = list(range(self.data.num_classes))
        true_labels = self.true.to(device)
        pred_labels = self.pred.to(device)
        accs_clean = []

        for clean_class in classes:
            clean_indices = (true_labels == clean_class)
            accs_clean.append(accuracy_score(true_labels[clean_indices].cpu(), pred_labels[clean_indices].cpu()))
        print(f'class accuracies: {accs_clean}')
        accs_clean = sum(accs_clean) / len(accs_clean)
        return accs_clean

    def subset_acc(self, class1=None, class2=None):
        if class1 is None or class2 is None:
            class1 = self.class1
            class2 = self.class2

        poisoned_classes = [class1, class2]

        true_labels = self.true.to(device)
        pred_labels = self.pred.to(device)

        clean_classes = []
        for i in range(self.data.num_classes):
            if i not in poisoned_classes:
                clean_classes.append(i)

        # calculate acc separately on poisoned and non-poisoned classes
        accs_poisoned = []
        accs_clean = []

        for poisoned_class in poisoned_classes:
            poisoned_indices = (true_labels == poisoned_class)
            accs_poisoned.append(accuracy_score(true_labels[poisoned_indices].cpu(), pred_labels[poisoned_indices].cpu()))

        for clean_class in clean_classes:
            clean_indices = (true_labels == clean_class)
            accs_clean.append(accuracy_score(true_labels[clean_indices].cpu(), pred_labels[clean_indices].cpu()))

        print(f'Poisoned class: {class1} -> {class2} | Clean classes: {clean_classes}')
        print(f'Poisoned class acc: {accs_poisoned} | Clean class acc: {accs_clean}')
        # take average of the accs
        accs_poisoned = sum(accs_poisoned) / len(accs_poisoned)
        accs_clean = sum(accs_clean) / len(accs_clean)

        return accs_poisoned, accs_clean



    def misclassification_rate(self, true_labels, pred_labels):
        if self.class1 is None or self.class2 is None:
            return 0

        true_labels = true_labels.to(device)
        pred_labels = pred_labels.to(device)
        class1_to_class2 = ((true_labels == self.class1) & (pred_labels == self.class2)).sum().item()
        class2_to_class1 = ((true_labels == self.class2) & (pred_labels == self.class1)).sum().item()
        total_class1 = (true_labels == self.class1).sum().item()
        total_class2 = (true_labels == self.class2).sum().item()
        misclassification_rate = (class1_to_class2 + class2_to_class1) / (total_class1 + total_class2)
        return misclassification_rate

    def evaluate(self, is_dr=False):
        self.model.eval()

        with torch.no_grad():
            if(is_dr):
                z = F.log_softmax(self.model(self.data.x, self.data.edge_index[:, self.data.dr_mask]), dim=1)
            else:
                z = F.log_softmax(self.model(self.data.x, self.data.edge_index), dim=1)
            loss = F.nll_loss(z[self.data.test_mask], self.data.y[self.data.test_mask]).cpu().item()
            pred = torch.argmax(z[self.data.test_mask], dim=1).cpu()
            dt_acc = accuracy_score(self.data.y[self.data.test_mask].cpu(), pred)
            dt_f1 = f1_score(self.data.y[self.data.test_mask].cpu(), pred, average='micro')
            msc_rate = self.misclassification_rate(self.data.y[self.data.test_mask].cpu(), pred)

        self.true = self.data.y[self.data.test_mask].cpu()
        self.pred = pred

        return dt_acc, msc_rate, dt_f1