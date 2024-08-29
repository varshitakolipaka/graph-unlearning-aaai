import time
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import trange
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_loss_vs_epochs(loss_values):
    """
    Plots a line graph of loss vs. epochs using Seaborn.

    Parameters:
    loss_values (list or numpy array): An array of loss values.
    """
    # Create an array of epoch numbers
    loss_values = torch.stack(loss_values)
    loss_values= loss_values.cpu().detach().numpy()
    epochs = np.arange(1, len(loss_values) + 1)

    # Create a DataFrame for easier plotting with Seaborn
    data = pd.DataFrame({'Epoch': epochs, 'Loss': loss_values})

    # Set the style of the plot
    sns.set(style="whitegrid")

    # Create the line plot
    plt.figure(figsize=(12, 6))
    line_plot = sns.lineplot(x='Epoch', y='Loss', data=data, marker='o', color='b')

    # Add title and labels
    line_plot.set_title('Loss vs. Epochs', fontsize=16)
    line_plot.set_xlabel('Epoch', fontsize=14)
    line_plot.set_ylabel('Loss', fontsize=14)

    # Customize the tick parameters
    line_plot.tick_params(labelsize=12)

    # Show the plot
    plt.show()

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
        losses = []
        self.data = self.data.to(device)
        st = time.time()
        for epoch in trange(self.num_epochs, desc='Epoch'):
            self.model.train()
            z = F.log_softmax(self.model(self.data.x, self.data.edge_index), dim=1)
            loss = F.nll_loss(z[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            losses.append(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()
        time_taken = time.time() - st
        train_acc, msc_rate, f1 = self.evaluate()
        # print(f'Train Acc: {train_acc}, Misclassification: {msc_rate},  F1 Score: {f1}')
        # plot_loss_vs_epochs(losses)
        return train_acc, msc_rate, time_taken

    def all_class_acc(self):
        classes = list(range(self.data.num_classes))
        true_labels = self.true.to(device)
        pred_labels = self.pred.to(device)
        accs_clean = []

        for clean_class in classes:
            clean_indices = (true_labels == clean_class)
            accs_clean.append(accuracy_score(true_labels[clean_indices].cpu(), pred_labels[clean_indices].cpu()))
        print(f'Class Accuracies: {accs_clean}')
        accs_clean = sum(accs_clean) / len(accs_clean)
        print(f'Overall Accuracy: {accs_clean}')

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
        roc_aucs_poisoned=[]
        roc_aucs_clean= []

        # z = F.log_softmax(self.model(self.data.x, self.data.edge_index[:, self.data.dr_mask]), dim=1)

        for poisoned_class in poisoned_classes:
            poisoned_indices = (true_labels == poisoned_class)
            accs_poisoned.append(accuracy_score(true_labels[poisoned_indices].cpu(), pred_labels[poisoned_indices].cpu()))

        for clean_class in clean_classes:
            clean_indices = (true_labels == clean_class)
            accs_clean.append(accuracy_score(true_labels[clean_indices].cpu(), pred_labels[clean_indices].cpu()))

        print(f'Poisoned class: {class1} -> {class2}')
        # print(f'Poisoned class acc: {accs_poisoned} | Clean class acc: {accs_clean}')
        # take average of the accs
        accs_poisoned = sum(accs_poisoned) / len(accs_poisoned)
        accs_clean = sum(accs_clean) / len(accs_clean)

        # auc_poisoned = sum(roc_aucs_poisoned) / len(roc_aucs_poisoned)
        # auc_clean = sum(roc_aucs_clean) / len(roc_aucs_clean)

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

    # def evaluate(self, is_dr=False):
    #     self.model.eval()

    #     with torch.no_grad():
    #         if(is_dr):
    #             z = F.log_softmax(self.model(self.data.x, self.data.edge_index[:, self.data.dr_mask]), dim=1)
    #         else:
    #             z = F.log_softmax(self.model(self.data.x, self.data.edge_index), dim=1)
    #         loss = F.nll_loss(z[self.data.test_mask], self.data.y[self.data.test_mask]).cpu().item()
    #         pred = torch.argmax(z[self.data.test_mask], dim=1).cpu()
    #         dt_acc = accuracy_score(self.data.y[self.data.test_mask].cpu(), pred)
    #         dt_f1 = f1_score(self.data.y[self.data.test_mask].cpu(), pred, average='micro')
    #         msc_rate = self.misclassification_rate(self.data.y[self.data.test_mask].cpu(), pred)
    #         # auc = roc_auc_score(self.data.y[self.data.test_mask].cpu(), F.softmax(z[self.data.test_mask], dim=1).cpu(), multi_class='ovo')

    #     # print("AUC: ",auc)

    #     self.true = self.data.y[self.data.test_mask].cpu()
    #     self.pred = pred

    #     return dt_acc, msc_rate, dt_f1

    def evaluate(self, val_mask=None, is_dr=False):
        self.model.eval()

        with torch.no_grad():
            if is_dr:
                z = F.log_softmax(self.model(self.data.x, self.data.edge_index[:, self.data.dr_mask]), dim=1)
                mask = self.data.test_mask
            else:
                z = F.log_softmax(self.model(self.data.x, self.data.edge_index), dim=1)
                mask = val_mask if val_mask is not None else self.data.test_mask

            loss = F.nll_loss(z[mask], self.data.y[mask]).cpu().item()
            pred = torch.argmax(z[mask], dim=1).cpu()
            acc = accuracy_score(self.data.y[mask].cpu(), pred)
            f1 = f1_score(self.data.y[mask].cpu(), pred, average='micro')
            msc_rate = self.misclassification_rate(self.data.y[mask].cpu(), pred)

        self.true = self.data.y[mask].cpu()
        self.pred = pred

        return acc, msc_rate, f1

    def calculate_PSR(self):
        z = self.model(self.data.x, self.data.edge_index)
        pred = torch.argmax(z[self.data.poison_test_mask], dim=1).cpu()
        psr= sum(pred==self.data.target_class)/len(pred)
        return psr.item()
    
    def get_score(self, attack_type, class1=None, class2=None):
        forget_ability = None
        utility = None
        if attack_type=="label" or attack_type=="edge":
            forget_ability, utility = self.subset_acc(class1, class2)
        elif attack_type=="trigger":
            utility, _, _ = self.evaluate()
            forget_ability = self.calculate_PSR()
        elif attack_type=="random":
            utility, _, f1 = self.evaluate()
        return forget_ability, utility
        
