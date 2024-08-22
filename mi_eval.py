import copy
import os
import torch
from framework import utils
from framework.training_args import parse_args
from models.deletion import GCNDelete
from models.models import GCN
from trainers.base import Trainer
from attacks.edge_attack import edge_attack_specific_nodes
from attacks.label_flip import label_flip_attack
from attacks.feature_attack import trigger_attack
import torch.nn as nn
import optuna
from optuna.samplers import TPESampler
from functools import partial
from logger import Logger
from sklearn.model_selection import train_test_split
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

args = parse_args()
print(args)

logger = Logger(f"run_logs_{args.attack_type}.json")
logger.log_arguments(args)

utils.seed_everything(args.random_seed)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class AttackModel(nn.Module):
    def __init__(self, num_features):
        super(AttackModel, self).__init__()
        self.fc1 = nn.Linear(num_features, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def get_logits(model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)  # Pass edge_index to the model
    return logits

def train():
    # dataset
    print("==TRAINING==")
    clean_data = utils.get_original_data(args.dataset)
    d, train_idx, test_idx = utils.train_test_split(clean_data, args.random_seed, args.train_ratio)
    utils.prints_stats(clean_data)
    clean_model = GCN(clean_data.num_features, args.hidden_dim, clean_data.num_classes)

    optimizer = torch.optim.Adam(
        clean_model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay
    )
    clean_trainer = Trainer(clean_model, clean_data, optimizer, args.training_epochs)
    clean_trainer.train()

    if args.attack_type != "random":
        forg, util = clean_trainer.get_score(args.attack_type, class1=57, class2=33)

        print(f"==OG Model==\nForget Ability: {forg}, Utility: {util}")
        logger.log_result(args.random_seed, "original", {"forget": forg, "utility": util})

    # save the clean model
    os.makedirs("./data", exist_ok=True)
    torch.save(
        clean_model,
        f"./data/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_clean_model.pt",
    )

    print("==Membership Inference attack==") ## possibly wrong
    # Initialize MIAttackTrainer
    attack_model = AttackModel(clean_data.num_classes)
    logits = get_logits(clean_model, clean_data)
    train_logits = logits[train_idx]
    train_labels = torch.ones(len(train_logits))
    test_logits = logits[test_idx]
    test_labels = torch.zeros(len(test_logits))
    temp_logits = torch.cat((train_logits, test_logits), dim=0)
    labels = torch.cat((train_labels, test_labels), dim=0)
    # split train_logits into 80-20 train test split for probe

    x_probe_train, x_probe_test, y_probe_train, y_probe_test = train_test_split(test_logits, test_labels, test_size=0.2, random_state=args.random_seed)
    num_train = len(x_probe_train)
    num_test = len(x_probe_test)
    # divide the train_logits into train and test for probe where num_train is the number of nodes in train and num_test is the number of nodes in test

    x_probe_train = torch.cat((x_probe_train, train_logits[:num_train]), dim=0)
    y_probe_train = torch.cat((y_probe_train, train_labels[:num_train]), dim=0)
    x_probe_test = torch.cat((x_probe_test, train_logits[num_train:num_train+num_test]), dim=0)
    y_probe_test = torch.cat((y_probe_test, train_labels[num_train:num_train+num_test]), dim=0)

    tsne = TSNE(n_components=2, random_state=0)
    logits_tsne = tsne.fit_transform(temp_logits.cpu().numpy())

    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(logits_tsne[:, 0], logits_tsne[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Labels')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE of Logits Colored by Labels')
    plt.savefig('tsne.png')

    print(f"Number of nodes in probe train: {len(x_probe_train)}")
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(attack_model.parameters(), lr=0.01)

    num_epochs = 200  # Example, you can adjust this
    attack_model = attack_model.to(device)

    for epoch in range(num_epochs):
        attack_model.train()
        optimizer.zero_grad()

        # Forward pass
        
        outputs = attack_model(x_probe_train).squeeze()
        outputs = outputs.to(device)
        y_probe_train = y_probe_train.to(device)
        loss = criterion(outputs, y_probe_train)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Step 5: Evaluate the Attack Model (optional)
    attack_model.eval()

    # Make predictions
    with torch.no_grad():
        outputs = attack_model(x_probe_test).squeeze()  # Get model outputs
        predicted = (outputs > 0.5).float()  # Convert probabilities to binary predictions

    # Move data to CPU for metric computation
    y_probe_test = y_probe_test.cpu().numpy()
    predicted = predicted.cpu().numpy()

    # Compute metrics
    accuracy = accuracy_score(y_probe_test, predicted)
    precision = precision_score(y_probe_test, predicted)
    recall = recall_score(y_probe_test, predicted)
    f1 = f1_score(y_probe_test, predicted)

    # Print evaluation metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    return clean_data

def poison(clean_data=None):
    if clean_data is None:
        # load the poisoned data and model and indices from np file
        poisoned_data = torch.load(
            f"./data/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_poisoned_data.pt"
        )
        poisoned_indices = torch.load(
            f"./data/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_poisoned_indices.pt"
        )
        poisoned_model = torch.load(
            f"./data/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_poisoned_model.pt"
        )

        optimizer = torch.optim.Adam(
            poisoned_model.parameters(),
            lr=args.train_lr,
            weight_decay=args.weight_decay,
        )
        poisoned_trainer = Trainer(
            poisoned_model, poisoned_data, optimizer, args.training_epochs
        )
        poisoned_trainer.evaluate()

        forg, util = poisoned_trainer.get_score(args.attack_type, class1=57, class2=33)
        print(f"==Poisoned Model==\nForget Ability: {forg}, Utility: {util}")
        logger.log_result(
            args.random_seed, "poisoned", {"forget": forg, "utility": util}
        )

        # print(poisoned_trainer.calculate_PSR())
        return poisoned_data, poisoned_indices, poisoned_model

    print("==POISONING==")
    if args.attack_type == "label":
        poisoned_data, poisoned_indices = label_flip_attack(
            clean_data, args.df_size, args.random_seed
        )
    elif args.attack_type == "edge":
        poisoned_data, poisoned_indices = edge_attack_specific_nodes(
            clean_data, args.df_size, args.random_seed
        )
    elif args.attack_type == "random":
        poisoned_data = copy.deepcopy(clean_data)
        poisoned_indices = torch.randperm(clean_data.num_nodes)[
            : int(clean_data.num_nodes * args.df_size)
        ]
    elif args.attack_type == "trigger":
        poisoned_data, poisoned_indices = trigger_attack(
            clean_data, args.df_size, args.poison_tensor_size, args.random_seed, args.test_poison_fraction, target_class=57
        )
    poisoned_data = poisoned_data.to(device)

    if "gnndelete" in args.unlearning_model:
        # poisoned_model = GCNDelete(poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes)
        poisoned_model = GCN(
            poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes
        )
    else:
        poisoned_model = GCN(
            poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes
        )

    optimizer = torch.optim.Adam(
        poisoned_model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay
    )
    poisoned_trainer = Trainer(
        poisoned_model, poisoned_data, optimizer, args.training_epochs
    )
    poisoned_trainer.train()

    # save the poisoned data and model and indices to np file
    os.makedirs("./data", exist_ok=True)

    torch.save(
        poisoned_model,
        f"./data/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_poisoned_model.pt",
    )

    torch.save(
        poisoned_data,
        f"./data/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_poisoned_data.pt",
    )
    torch.save(
        poisoned_indices,
        f"./data/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_poisoned_indices.pt",
    )

    forg, util = poisoned_trainer.get_score(args.attack_type, class1=57, class2=33)
    print(f"==Poisoned Model==\nForget Ability: {forg}, Utility: {util}")
    logger.log_result(args.random_seed, "poisoned", {"forget": forg, "utility": util})
    # print(f"PSR: {poisoned_trainer.calculate_PSR()}")
    return poisoned_data, poisoned_indices, poisoned_model

def unlearn(poisoned_data, poisoned_indices, poisoned_model):
    print("==UNLEARNING==")

    utils.find_masks(
        poisoned_data, poisoned_indices, args, attack_type=args.attack_type
    )
    if "gnndelete" in args.unlearning_model:
        unlearn_model = GCNDelete(
            poisoned_data.num_features,
            args.hidden_dim,
            poisoned_data.num_classes,
            mask_1hop=poisoned_data.sdf_node_1hop_mask,
            mask_2hop=poisoned_data.sdf_node_2hop_mask,
            mask_3hop=poisoned_data.sdf_node_3hop_mask,
        )

        # copy the weights from the poisoned model
        unlearn_model.load_state_dict(poisoned_model.state_dict())

        optimizer_unlearn = utils.get_optimizer(args, unlearn_model)
        unlearn_trainer = utils.get_trainer(
            args, unlearn_model, poisoned_data, optimizer_unlearn
        )
        unlearn_trainer.train()
    elif "retrain" in args.unlearning_model:
        unlearn_model = GCN(
            poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes
        )
        optimizer_unlearn = utils.get_optimizer(args, unlearn_model)
        unlearn_trainer = utils.get_trainer(
            args, unlearn_model, poisoned_data, optimizer_unlearn
        )
        unlearn_trainer.train()
    else:
        optimizer_unlearn = utils.get_optimizer(args, poisoned_model)
        unlearn_trainer = utils.get_trainer(
            args, poisoned_model, poisoned_data, optimizer_unlearn
        )
        unlearn_trainer.train()
    forg, util = unlearn_trainer.get_score(args.attack_type, class1=57, class2=33)
    print(f"==Unlearned Model==\nForget Ability: {forg}, Utility: {util}")
    logger.log_result(
        args.random_seed, args.unlearning_model, {"forget": forg, "utility": util}
    )
    print("==UNLEARNING DONE==")

if __name__ == "__main__":
    clean_data = train()
    exit()
    poisoned_data, poisoned_indices, poisoned_model = poison(clean_data)
    unlearn(poisoned_data, poisoned_indices, poisoned_model)