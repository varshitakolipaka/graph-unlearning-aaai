import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
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

    def get_silhouette_scores(self, graph_temp=None):
        self.model.eval()
        with torch.no_grad():
            if(graph_temp is None):
                embeddings= self.model(self.data.x, self.data.edge_index)
            else:
                embeddings= self.model(graph_temp.x, graph_temp.edge_index)

            probabilites = F.softmax(embeddings, dim=1)
            _, predicted_labels= torch.max(probabilites, dim=1)
        return silhouette_score(embeddings, predicted_labels)

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
    
    def evaluate1(self, data, is_dr=False):
        self.model.eval()
        with torch.no_grad():
            if(is_dr):
                z = F.log_softmax(self.model(data.x, data.edge_index[:, data.dr_mask]), dim=1)
            else:
                z = F.log_softmax(self.model(data.x, data.edge_index), dim=1)
                
            pred = torch.argmax(z[data.val_mask], dim=1).cpu()
            pred_df = torch.argmax(z[data.df_mask], dim=1).cpu()
            dt_acc = accuracy_score(data.y[data.val_mask].cpu(), pred)
            dt_f1 = f1_score(data.y[data.val_mask].cpu(), pred, average='micro')
            df_acc = accuracy_score(data.y[data.df_mask].cpu(), pred_df)
            df_f1 = f1_score(data.y[data.df_mask].cpu(), pred_df, average='micro')
        return dt_acc, dt_f1, df_acc, df_f1
    
