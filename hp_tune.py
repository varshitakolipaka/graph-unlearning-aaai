from collections import defaultdict
import copy
import json
import os
import torch
from framework import utils
from framework.training_args import parse_args
from models.deletion import GCNDelete
from models.models import GCN
from trainers.base import Trainer
from attacks.edge_attack import edge_attack_specific_nodes
from attacks.label_flip import label_flip_attack, label_flip_attack_strong
from attacks.feature_attack import trigger_attack
import optuna
from optuna.samplers import TPESampler
from functools import partial
from logger import Logger
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

args = parse_args()

utils.seed_everything(args.random_seed)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(f"Using device: {device}")

with open("classes_to_poison.json", "r") as f:
    class_dataset_dict = json.load(f)

with open("model_seeds.json") as f:
    model_seeds = json.load(f)

def train(load=False):
    if load:
        clean_data = utils.get_original_data(args.dataset)
        # utils.train_test_split(
        #     clean_data, args.random_seed, args.train_ratio, args.val_ratio
        # )
        utils.train_test_split(
            clean_data, model_seeds[args.dataset], args.train_ratio, args.val_ratio
        )
        utils.prints_stats(clean_data)

        clean_model = torch.load(
            f"{args.data_dir}/{args.gnn}_{args.dataset}_{args.attack_type}_{args.df_size}_{model_seeds[args.dataset]}_clean_model.pt"
        )

        optimizer = torch.optim.Adam(
            clean_model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay
        )

        clean_trainer = Trainer(clean_model, clean_data, optimizer, args)

        if args.attack_type != "trigger":
            print("ACC__ : ", clean_trainer.evaluate())
            forg, util, forget_f1, util_f1 = clean_trainer.get_score(
                args.attack_type,
                class1=class_dataset_dict[args.dataset]["class1"],
                class2=class_dataset_dict[args.dataset]["class2"],
            )

            print(
                f"==OG Model==\nForg Accuracy: {forg}, Util Accuracy: {util}, Forg F1: {forget_f1}, Util F1: {util_f1}"
            )
        else:
            # check accuracy on the poisoned test samples
            victim_class = class_dataset_dict[args.dataset]["victim_class"]
            # get the indices of the poisoned nodes
            poisoned_data = torch.load(
                f"{args.data_dir}/{args.dataset}_{args.attack_type}_{args.df_size}_{model_seeds[args.dataset]}_poisoned_data.pt"
            )
            poisoned_test_mask = poisoned_data.poison_test_mask

            # check the accuracy of the model on the poisoned test samples
            z = F.log_softmax(clean_model(clean_data.x, clean_data.edge_index), dim=-1)
            pred = z.argmax(dim=-1)
            acc = accuracy_score(
                pred[poisoned_test_mask].cpu().numpy(),
                clean_data.y[poisoned_test_mask].cpu().numpy(),
            )

            print(f"==OG Model==\nAccuracy on victim class test samples: {acc}")

        return clean_data

    # dataset
    print("==TRAINING==")
    clean_data = utils.get_original_data(args.dataset)
    utils.train_test_split(
        clean_data, model_seeds[args.dataset], args.train_ratio, args.val_ratio
    )
    utils.prints_stats(clean_data)
    clean_model = utils.get_model(
        args, clean_data.num_features, args.hidden_dim, clean_data.num_classes
    )

    optimizer = torch.optim.Adam(
        clean_model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay
    )
    clean_trainer = Trainer(clean_model, clean_data, optimizer, args)
    clean_trainer.train()

    if args.attack_type != "trigger":
        print("ACC__ : ", clean_trainer.evaluate())
        forg, util, forget_f1, util_f1 = clean_trainer.get_score(
            args.attack_type,
            class1=class_dataset_dict[args.dataset]["class1"],
            class2=class_dataset_dict[args.dataset]["class2"],
        )

        print(
            f"==OG Model==\nForg Accuracy: {forg}, Util Accuracy: {util}, Forg F1: {forget_f1}, Util F1: {util_f1}"
        )
    # save the clean model
    os.makedirs(args.data_dir, exist_ok=True)
    torch.save(
        clean_model,
        f"{args.data_dir}/{args.gnn}_{args.dataset}_{args.attack_type}_{args.df_size}_{model_seeds[args.dataset]}_clean_model.pt",
    )

    return clean_data


def poison(clean_data=None):
    if clean_data is None:
        # load the poisoned data and model and indices from np file
        poisoned_data = torch.load(
            f"{args.data_dir}/{args.dataset}_{args.attack_type}_{args.df_size}_{model_seeds[args.dataset]}_poisoned_data.pt"
        )
        poisoned_model = torch.load(
            f"{args.data_dir}/{args.gnn}_{args.dataset}_{args.attack_type}_{args.df_size}_{model_seeds[args.dataset]}_poisoned_model.pt"
        )

        print(poisoned_model.state_dict().keys())

        if args.attack_type == "edge":
            poisoned_indices = poisoned_data.poisoned_edge_indices
        else:
            poisoned_indices = poisoned_data.poisoned_nodes

        optimizer = torch.optim.Adam(
            poisoned_model.parameters(),
            lr=args.train_lr,
            weight_decay=args.weight_decay,
        )
        poisoned_trainer = Trainer(poisoned_model, poisoned_data, optimizer, args)
        poisoned_trainer.evaluate()

        forg, util, forget_f1, util_f1 = poisoned_trainer.get_score(
            args.attack_type,
            class1=class_dataset_dict[args.dataset]["class1"],
            class2=class_dataset_dict[args.dataset]["class2"],
        )

        print(
            f"==Poisoned Model==\nForg Accuracy: {forg}, Util Accuracy: {util}, Forg F1: {forget_f1}, Util F1: {util_f1}"
        )
        # print(poisoned_trainer.calculate_PSR())
        return poisoned_data, poisoned_indices, poisoned_model

    print("==POISONING==")
    if args.attack_type == "label":
        poisoned_data, poisoned_indices = label_flip_attack(
            clean_data,
            args.df_size,
            args.random_seed,
            class_dataset_dict[args.dataset]["class1"],
            class_dataset_dict[args.dataset]["class2"],
        )
    if args.attack_type == "label_strong":
        poisoned_data, poisoned_indices = label_flip_attack_strong(
            clean_data,
            args.df_size,
            args.random_seed,
            class_dataset_dict[args.dataset]["class1"],
            class_dataset_dict[args.dataset]["class2"],
        )
    elif args.attack_type == "edge":
        poisoned_data, poisoned_indices = edge_attack_specific_nodes(
            clean_data,
            args.df_size,
            args.random_seed,
            class1=class_dataset_dict[args.dataset]["class1"],
            class2=class_dataset_dict[args.dataset]["class2"],
        )
    elif args.attack_type == "random":
        poisoned_data = copy.deepcopy(clean_data)
        poisoned_indices = torch.randperm(clean_data.num_nodes)[
            : int(clean_data.num_nodes * args.df_size)
        ]
        poisoned_data.poisoned_nodes = poisoned_indices
    elif args.attack_type == "trigger":
        poisoned_data, poisoned_indices = trigger_attack(
            clean_data,
            args.df_size,
            args.random_seed,
            victim_class=class_dataset_dict[args.dataset]["victim_class"],
            target_class=class_dataset_dict[args.dataset]["target_class"],
            trigger_size=args.trigger_size,
        )
    poisoned_data = poisoned_data.to(device)

    poisoned_model = utils.get_model(
        args, poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes
    )

    optimizer = torch.optim.Adam(
        poisoned_model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay
    )
    poisoned_trainer = Trainer(poisoned_model, poisoned_data, optimizer, args)
    poisoned_trainer.train()

    # save the poisoned data and model and indices to np file
    os.makedirs(args.data_dir, exist_ok=True)

    torch.save(
        poisoned_model,
        f"{args.data_dir}/{args.gnn}_{args.dataset}_{args.attack_type}_{args.df_size}_{model_seeds[args.dataset]}_poisoned_model.pt",
    )

    torch.save(
        poisoned_data,
        f"{args.data_dir}/{args.dataset}_{args.attack_type}_{args.df_size}_{model_seeds[args.dataset]}_poisoned_data.pt",
    )

    forg, util, forget_f1, util_f1 = poisoned_trainer.get_score(
        args.attack_type,
        class1=class_dataset_dict[args.dataset]["class1"],
        class2=class_dataset_dict[args.dataset]["class2"],
    )

    print(
        f"==Poisoned Model==\nForg Accuracy: {forg}, Util Accuracy: {util}, Forg F1: {forget_f1}, Util F1: {util_f1}"
    )

    # print(f"PSR: {poisoned_trainer.calculate_PSR()}")
    return poisoned_data, poisoned_indices, poisoned_model


hp_tuning_params_dict = {
    "retrain": {
        "unlearn_lr": (1e-5, 1e-1, "log"),
        "weight_decay": (1e-5, 1e-1, "log"),
        "unlearning_epochs": (600, 1400, "int"),
    },
    "gnndelete": {
        "unlearn_lr": (1e-5, 1e-1, "log"),
        "weight_decay": (1e-5, 1e-1, "log"),
        # "unlearning_epochs": (10, 200, "int"),
        "alpha": (0, 1, "float"),
        "loss_type": (
            [
                "both_all",
                "both_layerwise",
                "only2_layerwise",
                "only2_all",
                "only1",
                "only3",
                "only3_all",
                "only3_layerwise",
            ],
            "categorical",
        ),
    },
    "gnndelete_ni": {
        "unlearn_lr": (1e-5, 1e-1, "log"),
        "weight_decay": (1e-5, 1e-1, "log"),
        "unlearning_epochs": (10, 100, "int"),
        "loss_type": (
            [
                "only2_layerwise",
                "only2_all",
                "only1",
                "only3",
                "only3_all",
                "only3_layerwise",
            ],
            "categorical",
        ),
    },
    "gif": {
        "iteration": (10, 1000, "int"),
        "scale": (1e7, 1e11, "log"),
        "damp": (0.0, 1.0, "float"),
    },
    "gradient_ascent": {
        # "unlearning_epochs": (10, 2000, "int"),
        "unlearn_lr": (1e-5, 1e-1, "log"),
        "weight_decay": (1e-5, 1e-1, "log"),
    },
    "contrastive": {
        "contrastive_epochs_1": (5, 30, "int"),
        "contrastive_epochs_2": (5, 30, "int"),
        # "maximise_epochs": (5, 30, "int"),
        "unlearn_lr": (1e-5, 1e-1, "log"),
        "weight_decay": (1e-5, 1e-1, "log"),
        "contrastive_margin": (1, 1e3, "log"),
        "contrastive_lambda": (0.0, 1.0, "float"),
        "contrastive_frac": (0.01, 0.2, "float"),
        "k_hop": (1, 2, "int"),
    },
    "contra_2": {
        "contrastive_epochs_1": (1, 10, "int"),
        "contrastive_epochs_2": (1, 30, "int"),
        "steps": (1, 10, "int"),
        # "maximise_epochs": (5, 30, "int"),
        "unlearn_lr": (1e-4, 1e-1, "log"),
        # "contrastive_margin": (1, 10, "log"),
        # "contrastive_lambda": (0.0, 1.0, "float"),
        "contrastive_frac": (0.02, 0.3, "float"),
        "k_hop": (1, 3, "int"),
        # "ascent_lr": (1e-6, 1e-3, "log"),
        "descent_lr": (1e-4, 1e-1, "log"),
        # "scrubAlpha": (1e-6, 10, "log"),
    },
    "contrascent": {
        "contrastive_epochs_1": (1, 5, "int"),
        "contrastive_epochs_2": (1, 5, "int"),
        "steps": (1, 15, "int"),
        # "maximise_epochs": (5, 30, "int"),
        "unlearn_lr": (1e-5, 1e-1, "log"),
        "contrastive_margin": (1, 10, "log"),
        # "contrastive_lambda": (0.0, 1.0, "float"),
        "contrastive_frac": (0.01, 0.2, "float"),
        "k_hop": (1, 3, "int"),
        "ascent_lr": (1e-5, 1e-3, "log"),
        "descent_lr": (1e-5, 1e-1, "log"),
        "scrubAlpha": (1e-6, 10, "log"),
    },
    "cacdc": {
        "contrastive_epochs_1": (1, 6, "int"),
        "contrastive_epochs_2": (1, 30, "int"),
        "steps": (1, 10, "int"),
        # "maximise_epochs": (5, 30, "int"),
        "unlearn_lr": (1e-4, 1e-1, "log"),
        # "contrastive_margin": (1, 10, "log"),
        # "contrastive_lambda": (0.0, 1.0, "float"),
        "contrastive_frac": (0.02, 0.3, "float"),
        "k_hop": (1, 3, "int"),
        "ascent_lr": (1e-6, 1e-3, "log"),
        "descent_lr": (1e-4, 1e-1, "log"),
        # "scrubAlpha": (1e-6, 10, "log"),
    },
    "utu": {},
    "scrub": {
        "unlearn_iters": (110, 300, "int"),
        # 'kd_T': (1, 10, "float"),
        "unlearn_lr": (1e-5, 1e-1, "log"),
        "scrubAlpha": (1e-6, 10, "log"),
        "msteps": (10, 150, "int"),
        # 'weight_decay': (1e-5, 1e-1, "log"),
    },
    "scrub_no_kl": {
        "unlearn_iters": (110, 200, "int"),
        # 'kd_T': (1, 10, "float"),
        "unlearn_lr": (1e-5, 1e-1, "log"),
        # "scrubAlpha": (1e-6, 10, "log"),
        "msteps": (10, 100, "int"),
        # 'weight_decay': (1e-5, 1e-1, "log"),
    },
    "scrub_no_kl_combined": {
        "unlearn_iters": (10, 50, "int"),
        # 'kd_T': (1, 10, "float"),
        "unlearn_lr": (1e-5, 1e-1, "log"),
        "scrubAlpha": (1e-6, 10, "log"),
        "msteps": (1, 20, "int"),
        # 'weight_decay': (1e-5, 1e-1, "log"),
    },
    "yaum": {
        "unlearn_iters": (10, 300, "int"),
        # 'kd_T': (1, 10, "float"),
        "ascent_lr": (1e-5, 1e-3, "log"),
        "descent_lr": (1e-5, 1e-1, "log"),
        # "scrubAlpha": (1e-6, 10, "log"),
        # "msteps": (10, 100, "int"),
    },
    "megu": {
        "unlearn_lr": (1e-6, 1e-3, "log"),
        # "unlearning_epochs": (10, 1000, "int"),
        "kappa": (1e-3, 1, "log"),
        "alpha1": (0, 1, "float"),
        "alpha2": (0, 1, "float"),
    },
    "ssd": {
        "SSDdampening": (0.1, 10, "log"),
        "SSDselectwt": (0.1, 100, "log"),
    },
    "clean": {
        "train_lr": (1e-5, 1e-1, "log"),
        "weight_decay": (1e-5, 1e-1, "log"),
        "training_epochs": (500, 3000, "int"),
    },
}


def set_hp_tuning_params(trial):
    hp_tuning_params = hp_tuning_params_dict[args.unlearning_model]
    for hp, values in hp_tuning_params.items():
        if values[1] == "categorical":
            setattr(args, hp, trial.suggest_categorical(hp, values[0]))
        elif values[2] == "int":
            setattr(args, hp, trial.suggest_int(hp, values[0], values[1]))
        elif values[2] == "float":
            setattr(args, hp, trial.suggest_float(hp, values[0], values[1]))
        elif values[2] == "log":
            setattr(args, hp, trial.suggest_float(hp, values[0], values[1], log=True))


def objective(trial, model, data):
    # Define the hyperparameters to tune
    set_hp_tuning_params(trial)

    model_internal = copy.deepcopy(model)

    optimizer = utils.get_optimizer(args, model_internal)
    trainer = utils.get_trainer(args, model_internal, data, optimizer)

    _, _, time_taken = trainer.train()

    if args.linked:
        obj = trainer.validate(is_dr=False)  # REAL
    else:
        obj = trainer.validate(is_dr=True)  # REAL

    forget_acc, util_acc, forget_f1, util_f1 = trainer.get_score(
        args.attack_type,
        class1=class_dataset_dict[args.dataset]["class1"],
        class2=class_dataset_dict[args.dataset]["class2"],
    )
    trial.set_user_attr("forget_acc", forget_acc)
    trial.set_user_attr("util_acc", util_acc)
    trial.set_user_attr("forget_f1", forget_f1)
    trial.set_user_attr("util_f1", util_f1)
    trial.set_user_attr("time_taken", time_taken)

    # We want to minimize misclassification rate and maximize accuracy
    return obj


if __name__ == "__main__":
    print("\n\n\n")
    print(args.dataset, args.attack_type)
    clean_data = train(load=True)
    # clean_data = train()
    poisoned_data, poisoned_indices, poisoned_model = poison()

    if args.corrective_frac < 1:
        print("==POISONING CORRECTIVE==")
        if args.attack_type == "edge":
            poisoned_indices = poisoned_data.poisoned_edge_indices
        else:
            poisoned_indices = poisoned_data.poisoned_nodes
        print(f"No. of poisoned nodes: {len(poisoned_indices)}")

        if args.attack_type == "edge":
            poisoned_indices, poisoned_nodes = utils.sample_poison_data_edges(
                poisoned_data, args.corrective_frac
            )
            poisoned_data.poisoned_edge_indices = poisoned_indices
            poisoned_data.poisoned_nodes = poisoned_nodes
        else:
            poisoned_indices = utils.sample_poison_data(
                poisoned_indices, args.corrective_frac
            )
            poisoned_data.poisoned_nodes = poisoned_indices
        print(f"No. of poisoned nodes after corrective: {len(poisoned_indices)}")

    utils.find_masks(
        poisoned_data, poisoned_indices, args, attack_type=args.attack_type
    )

    if "gnndelete" in args.unlearning_model:
        # Create a partial function with additional arguments
        model = utils.get_model(
            args,
            poisoned_data.num_features,
            args.hidden_dim,
            poisoned_data.num_classes,
            mask_1hop=poisoned_data.sdf_node_1hop_mask,
            mask_2hop=poisoned_data.sdf_node_2hop_mask,
            mask_3hop=poisoned_data.sdf_node_3hop_mask,
        )

        # copy the weights from the poisoned model
        state_dict = poisoned_model.state_dict()
        state_dict["deletion1.deletion_weight"] = model.deletion1.deletion_weight
        state_dict["deletion2.deletion_weight"] = model.deletion2.deletion_weight
        state_dict["deletion3.deletion_weight"] = model.deletion3.deletion_weight

        model.load_state_dict(state_dict)
    elif "retrain" in args.unlearning_model:
        model = utils.get_model(
            args, poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes
        )
    else:
        model = utils.get_model(
            args, poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes
        )
        model.load_state_dict(poisoned_model.state_dict())

    objective_func = partial(objective, model=model, data=poisoned_data)

    print("==HYPERPARAMETER TUNING==")
    # Create a study with TPE sampler
    study = optuna.create_study(
        sampler=TPESampler(seed=42),
        direction="maximize",
        study_name=f"{args.gnn}_{args.dataset}_{args.attack_type}_{args.df_size}_{args.unlearning_model}_{args.random_seed}_{class_dataset_dict[args.dataset]['class1']}_{class_dataset_dict[args.dataset]['class2']}",
        load_if_exists=True,
        storage=f"sqlite:///hp_tuning/new/{args.db_name}.db",
    )

    print("==OPTIMIZING==")

    # Optimize the objective function

    # reduce trials for utu and contrastive
    if args.unlearning_model == "utu":
        study.optimize(objective_func, n_trials=1)
    elif args.unlearning_model == "retrain":
        study.optimize(objective_func, n_trials=30)
    # elif args.unlearning_model == "contrastive" or args.unlearning_model == "contra_2":
    #     study.optimize(objective_func, n_trials=200)
    else:
        study.optimize(objective_func, n_trials=100)
