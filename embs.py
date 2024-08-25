import copy
import os
import time
import torch
from framework import utils
from framework.training_args import parse_args
from models.deletion import GCNDelete
from models.models import GCN
from trainers.base import Trainer
from attacks.edge_attack import edge_attack_specific_nodes
from attacks.label_flip import label_flip_attack
from attacks.feature_attack import trigger_attack
import optuna
from optuna.samplers import TPESampler
from functools import partial
from logger import Logger
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


args = parse_args()
print(args)

class_dataset_dict = {
    'Cora': {
        'class1': 57,
        'class2': 33,
    },
    'PubMed': {
        'class1': 2,
        'class2': 1,
    },
}

utils.seed_everything(args.random_seed)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def plot_embeddings(model, data, class1, class2, is_dr=False, mask="test", name=""):
    # Set the model to evaluation mode
    model.eval()

    # Forward pass: get embeddings
    with torch.no_grad():
        if is_dr and args.unlearning_model != "scrub":
            embeddings = model(data.x, data.edge_index[:, data.dr_mask])
        else:
            embeddings = model(data.x, data.edge_index)

    # If embeddings have more than 2 dimensions, apply t-SNE
    print("Embeddings shape:", embeddings.shape)
    if embeddings.shape[1] > 2:
        embeddings = TSNE(n_components=2).fit_transform(embeddings.cpu())
        embeddings = torch.tensor(embeddings).to(device)
    print("Embeddings shape after t-SNE:", embeddings.shape)
    # Get the mask (either test, train, or val)
    if mask == "test":
        mask = data.test_mask
    elif mask == "train":
        mask = data.train_mask
    else:
        mask = data.val_mask

    # Filter embeddings and labels based on the mask
    embeddings = embeddings[mask]
    labels = data.y[mask]

    # Create masks for class1, class2, and other classes
    class1_mask = (labels == class1)
    class2_mask = (labels == class2)
    other_mask = ~(class1_mask | class2_mask)

    # convert masks to numpy
    class1_mask = class1_mask.cpu().numpy()
    class2_mask = class2_mask.cpu().numpy()
    other_mask = other_mask.cpu().numpy()

    # Prepare the plot
    plt.figure(figsize=(10, 8))
    sns.set(style="whitegrid")

    # convert to numpy
    embeddings = embeddings.cpu().numpy()
    labels = labels.cpu().numpy()

    # Plot class1
    plt.scatter(embeddings[class1_mask, 0], embeddings[class1_mask, 1], label=f'Class {class1}', color='blue', alpha=0.6)
    # Plot class2
    plt.scatter(embeddings[class2_mask, 0], embeddings[class2_mask, 1], label=f'Class {class2}', color='red', alpha=0.6)
    # Plot other classes
    plt.scatter(embeddings[other_mask, 0], embeddings[other_mask, 1], label='Other Classes', color='gray', alpha=0.4)

    # Add legend and labels
    plt.legend()
    plt.title("Embeddings Visualization")
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")

    # Save the plot
    os.makedirs("./plots", exist_ok=True)
    plt.savefig(f"./plots/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_{name}_embeddings.png")


def train():
    # dataset
    print("==TRAINING==")
    clean_data = utils.get_original_data(args.dataset)
    utils.train_test_split(clean_data, args.random_seed, args.train_ratio)
    utils.prints_stats(clean_data)
    clean_model = GCN(clean_data.num_features, args.hidden_dim, clean_data.num_classes)

    optimizer = torch.optim.Adam(
        clean_model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay
    )
    clean_trainer = Trainer(clean_model, clean_data, optimizer, args.training_epochs)
    clean_trainer.train()

    if args.attack_type != "trigger":
        forg, util = clean_trainer.get_score(args.attack_type, class1=class_dataset_dict[args.dataset]['class1'], class2=class_dataset_dict[args.dataset]['class2'])

        print(f"==OG Model==\nForget Ability: {forg}, Utility: {util}")
        # logger.log_result(args.random_seed, "original", {"forget": forg, "utility": util})
        if args.embs_all:
            plot_embeddings(clean_model, clean_data, class_dataset_dict[args.dataset]['class1'], class_dataset_dict[args.dataset]['class2'], is_dr=False, mask="test", name="original")

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

        forg, util = poisoned_trainer.get_score(args.attack_type, class1=class_dataset_dict[args.dataset]['class1'], class2=class_dataset_dict[args.dataset]['class2'])
        print(f"==Poisoned Model==\nForget Ability: {forg}, Utility: {util}")
        if args.embs_all:
            plot_embeddings(poisoned_model, poisoned_data, class_dataset_dict[args.dataset]['class1'], class_dataset_dict[args.dataset]['class2'], is_dr=False, mask="test", name="poisoned")
        # logger.log_result(
        #     args.random_seed, "poisoned", {"forget": forg, "utility": util}
        # )

        # prirnt(poisoned_trainer.calculate_PSR())
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

    # torch.save(
    #     poisoned_model,
    #     f"./data/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_poisoned_model.pt",
    # )

    # torch.save(
    #     poisoned_data,
    #     f"./data/{args.dataset}_{args.attack_type}_{args.df_\
        # size}_{args.random_seed}_poisoned_data.pt",
    # )
    # torch.save(
    #     poisoned_indices,
    #     f"./data/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_poisoned_indices.pt",
    # )

    forg, util = poisoned_trainer.get_score(args.attack_type, class1=class_dataset_dict[args.dataset]['class1'], class2=class_dataset_dict[args.dataset]['class2'])
    print(f"==Poisoned Model==\nForget Ability: {forg}, Utility: {util}")
    if args.embs_all:
        plot_embeddings(poisoned_model, poisoned_data, class_dataset_dict[args.dataset]['class1'], class_dataset_dict[args.dataset]['class2'], is_dr=False, mask="test", name="poisoned")
    # logger.log_result(args.random_seed, "poisoned", {"forget": forg, "utility": util})
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
        state_dict = poisoned_model.state_dict()
        state_dict["deletion1.deletion_weight"] = unlearn_model.deletion1.deletion_weight
        state_dict["deletion2.deletion_weight"] = unlearn_model.deletion2.deletion_weight
        state_dict["deletion3.deletion_weight"] = unlearn_model.deletion3.deletion_weight


        # copy the weights from the poisoned model
        unlearn_model.load_state_dict(state_dict)

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
        unlearn_model = copy.deepcopy(poisoned_model)
        optimizer_unlearn = utils.get_optimizer(args, unlearn_model)
        st = time.time()
        unlearn_trainer = utils.get_trainer(
            args, unlearn_model, poisoned_data, optimizer_unlearn
        )
        print("Time to get trainer: ", time.time() - st)
        unlearn_trainer.train()
    forg, util = unlearn_trainer.get_score(args.attack_type, class1=class_dataset_dict[args.dataset]['class1'], class2=class_dataset_dict[args.dataset]['class2'])
    print(f"==Unlearned Model==\nForget Ability: {forg}, Utility: {util}")
    if args.embs_all or args.embs_unlearn:
        plot_embeddings(unlearn_model, poisoned_data, class_dataset_dict[args.dataset]['class1'], class_dataset_dict[args.dataset]['class2'], is_dr=True, mask="test", name=args.unlearning_model)
    # logger.log_result(
    #     args.random_seed, args.unlearning_model, {"forget": forg, "utility": util}
    # )
    print("==UNLEARNING DONE==")

best_params_dict = {
    "retrain": {},
    "gnndelete": {
        "unlearn_lr": 0.00551402238578268,
        "weight_decay": 0.0005798215498447256,
        "unlearning_epochs": 72,
        "alpha": 0.02811472371516137,
        "loss_type": "both_layerwise",
    },
    "gif": {
        "iteration": 986,
        "scale": 7706555780.747042,
        "damp": 0.2640338318115278,
    },
    "gradient_ascent": {
        "unlearning_epochs": (10, 2000, "int"),
        "unlearn_lr": (1e-5, 1e-1, "log"),
        "weight_decay": (1e-5, 1e-1, "log"),
    },
    "contrastive": {
        "contrastive_epochs_1": 5,
        "contrastive_epochs_2": 35,
        "unlearn_lr": 0.01,
        "weight_decay": 5e-4,
        "contrastive_margin": 50,
        "contrastive_lambda": 0.5,
        "contrastive_frac": 0.3,
        "k_hop": 2,
    },
    "utu": {},
    "scrub": {
        "unlearn_iters": 496,
        # 'kd_T': (1, 10, "float"),
        "unlearn_lr": 0.002891193845823931,
        "scrubAlpha": 0.005974093857668415,
        "msteps": 16,
        # 'weight_decay': (1e-5, 1e-1, "log"),
    },
    'megu': {
        'unlearn_lr': 1e-2,
        'unlearning_epochs': 100,
        'kappa': 0.01,
        'alpha1': 0.5,
        'alpha2': 0.8,
    },
    "clean": {
        "train_lr": (1e-5, 1e-1, "log"),
        "weight_decay": (1e-5, 1e-1, "log"),
        "training_epochs": (500, 3000, "int"),
    },
}

if __name__ == '__main__':
    clean_data = train()
    poisoned_data, poisoned_idx, poisoned_model = poison(clean_data)

    best_params = best_params_dict[args.unlearning_model]

    print(args)

    unlearn(poisoned_data, poisoned_idx, poisoned_model)