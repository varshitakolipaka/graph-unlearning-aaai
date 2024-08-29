from collections import defaultdict
import copy
import os
import torch

# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# torch.use_deterministic_algorithms(True)

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

args = parse_args()

logger = Logger(args, f"run_logs_{args.attack_type}.json")
logger.log_arguments(args)

utils.seed_everything(args.random_seed)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class_dataset_dict = {
    "Cora": {
        "class1": 57,
        "class2": 33,
    },
    "PubMed": {
        "class1": 2,
        "class2": 1,
    },
    "Amazon": {
        "class1": 6,
        "class2": 1,
    },
}


def train(load=False):
    if load:
        clean_data = utils.get_original_data(args.dataset)
        utils.train_test_split(clean_data, args.random_seed, args.train_ratio)
        utils.prints_stats(clean_data)

        clean_model = torch.load(
            f"{args.data_dir}/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_clean_model.pt"
        )

        optimizer = torch.optim.Adam(
            clean_model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay
        )

        clean_trainer = Trainer(clean_model, clean_data, optimizer, args.training_epochs)

        if args.attack_type != "trigger":
            clean_trainer.evaluate()
            forg, util = clean_trainer.get_score(
                args.attack_type,
                class1=class_dataset_dict[args.dataset]["class1"],
                class2=class_dataset_dict[args.dataset]["class2"],
            )

            print(f"==OG Model==\nForget Ability: {forg}, Utility: {util}")
            logger.log_result(
                args.random_seed, "original", {"forget": forg, "utility": util}
            )

        return clean_data

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
        forg, util = clean_trainer.get_score(
            args.attack_type,
            class1=class_dataset_dict[args.dataset]["class1"],
            class2=class_dataset_dict[args.dataset]["class2"],
        )

        print(f"==OG Model==\nForget Ability: {forg}, Utility: {util}")
        logger.log_result(
            args.random_seed, "original", {"forget": forg, "utility": util}
        )

    # save the clean model
    # os.makedirs(args.data_dir, exist_ok=True)
    # torch.save(
    #     clean_model,
    #     f"{args.data_dir}/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_clean_model.pt",
    # )

    return clean_data


def poison(clean_data=None):
    if clean_data is None:
        # load the poisoned data and model and indices from np file
        poisoned_data = torch.load(
            f"{args.data_dir}/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_poisoned_data.pt"
        )
        poisoned_indices = torch.load(
            f"{args.data_dir}/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_poisoned_indices.pt"
        )
        poisoned_model = torch.load(
            f"{args.data_dir}/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_poisoned_model.pt"
        )

        if not hasattr(poisoned_data, "poisoned_nodes"):
            poisoned_data.poisoned_nodes = poisoned_indices

        optimizer = torch.optim.Adam(
            poisoned_model.parameters(),
            lr=args.train_lr,
            weight_decay=args.weight_decay,
        )
        poisoned_trainer = Trainer(
            poisoned_model, poisoned_data, optimizer, args.training_epochs
        )
        poisoned_trainer.evaluate()

        forg, util = poisoned_trainer.get_score(
            args.attack_type,
            class1=class_dataset_dict[args.dataset]["class1"],
            class2=class_dataset_dict[args.dataset]["class2"],
        )
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
            clean_data,
            args.df_size,
            args.poison_tensor_size,
            args.random_seed,
            args.test_poison_fraction,
            target_class=57,
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
    # os.makedirs(args.data_dir, exist_ok=True)

    # torch.save(
    #     poisoned_model,
    #     f"{args.data_dir}/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_poisoned_model.pt",
    # )

    # torch.save(
    #     poisoned_data,
    #     f"{args.data_dir}/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_poisoned_data.pt",
    # )
    # torch.save(
    #     poisoned_indices,
    #     f"{args.data_dir}/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_poisoned_indices.pt",
    # )

    forg, util = poisoned_trainer.get_score(
        args.attack_type,
        class1=class_dataset_dict[args.dataset]["class1"],
        class2=class_dataset_dict[args.dataset]["class2"],
    )
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
        state_dict = poisoned_model.state_dict()
        state_dict["deletion1.deletion_weight"] = (
            unlearn_model.deletion1.deletion_weight
        )
        state_dict["deletion2.deletion_weight"] = (
            unlearn_model.deletion2.deletion_weight
        )
        state_dict["deletion3.deletion_weight"] = (
            unlearn_model.deletion3.deletion_weight
        )

        # copy the weights from the poisoned model
        unlearn_model.load_state_dict(state_dict)

        optimizer_unlearn = utils.get_optimizer(args, unlearn_model)
        unlearn_trainer = utils.get_trainer(
            args, unlearn_model, poisoned_data, optimizer_unlearn
        )
    elif "retrain" in args.unlearning_model:
        unlearn_model = GCN(
            poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes
        )
        optimizer_unlearn = utils.get_optimizer(args, unlearn_model)
        unlearn_trainer = utils.get_trainer(
            args, unlearn_model, poisoned_data, optimizer_unlearn
        )
    else:
        optimizer_unlearn = utils.get_optimizer(args, poisoned_model)
        unlearn_trainer = utils.get_trainer(
            args, poisoned_model, poisoned_data, optimizer_unlearn
        )

    _, _, time_taken = unlearn_trainer.train()
    forg, util = unlearn_trainer.get_score(args.attack_type, class1=class_dataset_dict[args.dataset]["class1"], class2=class_dataset_dict[args.dataset]["class2"])
    print(f"==Unlearned Model==\nForget Ability: {forg}, Utility: {util}, Time Taken: {time_taken}")
    logger.log_result(
        args.random_seed, args.unlearning_model, {"forget": forg, "utility": util, "time_taken": time_taken}
    )
    print("==UNLEARNING DONE==")


hp_tuning_params_dict = {
    "retrain": {},
    "gnndelete": {
        "unlearn_lr": (1e-5, 1e-1, "log"),
        "weight_decay": (1e-5, 1e-1, "log"),
        "unlearning_epochs": (10, 200, "int"),
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
        "unlearning_epochs": (10, 2000, "int"),
        "unlearn_lr": (1e-5, 1e-1, "log"),
        "weight_decay": (1e-5, 1e-1, "log"),
    },
    "contrastive": {
        "contrastive_epochs_1": (5, 30, "int"),
        "contrastive_epochs_2": (15, 30, "int"),
        "unlearn_lr": (1e-5, 1e-1, "log"),
        "weight_decay": (1e-5, 1e-1, "log"),
        "contrastive_margin": (1, 1e3, "log"),
        "contrastive_lambda": (0.0, 1.0, "float"),
        "contrastive_frac": (0.01, 0.1, "float"),
        "k_hop": (1, 2, "int"),
    },
    "utu": {},
    "scrub": {
        "unlearn_iters": (10, 500, "int"),
        # 'kd_T': (1, 10, "float"),
        "unlearn_lr": (1e-5, 1e-1, "log"),
        "scrubAlpha": (1e-6, 10, "log"),
        "msteps": (10, 500, "int"),
        # 'weight_decay': (1e-5, 1e-1, "log"),
    },
    "megu": {
        "unlearn_lr": (1e-6, 1e-3, "log"),
        "unlearning_epochs": (10, 1000, "int"),
        "kappa": (1e-3, 1, "log"),
        "alpha1": (0, 1, "float"),
        "alpha2": (0, 1, "float"),
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

    optimizer = utils.get_optimizer(args, model)
    trainer = utils.get_trainer(args, model, data, optimizer)

    _, _, time_taken = trainer.train()

    forg, util = trainer.get_score(
        args.attack_type,
        class1=class_dataset_dict[args.dataset]["class1"],
        class2=class_dataset_dict[args.dataset]["class2"],
    )
    if args.attack_type == "trigger":
        forg = 1 - forg

    trial.set_user_attr("time_taken", time_taken)
    trial.set_user_attr("forget_ability", forg)
    trial.set_user_attr("utility", util)

    # combine forget and utility to get a single objective
    obj = 0.5 * forg + 0.5 * util

    # We want to minimize misclassification rate and maximize accuracy
    return obj


if __name__ == "__main__":
    print('\n\n\n')
    print(args.dataset, args.attack_type)
    utils.seed_everything(args.random_seed)
    clean_data = train(load=False)
    # exit(0)
    poisoned_data, poisoned_indices, poisoned_model = poison(clean_data)
    unlearn(poisoned_data, poisoned_indices, poisoned_model)

    utils.find_masks(
        poisoned_data, poisoned_indices, args, attack_type=args.attack_type
    )

    if "gnndelete" in args.unlearning_model:
        # Create a partial function with additional arguments
        model = GCNDelete(
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
    else:
        model = poisoned_model

    objective_func = partial(objective, model=model, data=poisoned_data)

    print("==HYPERPARAMETER TUNING==")
    # Create a study with TPE sampler
    study = optuna.create_study(
        sampler=TPESampler(seed=args.random_seed),
        direction="maximize",
        study_name=f"{args.dataset}_{args.attack_type}_{args.df_size}_{args.unlearning_model}_{args.random_seed}",
        load_if_exists=True,
        storage="sqlite:///final_sanity_check.db",
    )

    print("==OPTIMIZING==")

    # Optimize the objective function

    # reduce trials for utu and contrastive
    if args.unlearning_model == "utu" or args.unlearning_model == "retrain":
        study.optimize(objective_func, n_trials=1)
    elif args.unlearning_model == "contrastive":
        study.optimize(objective_func, n_trials=100)
    elif args.unlearning_model == "megu":
        study.optimize(objective_func, n_trials=200)
    else:
        study.optimize(objective_func, n_trials=100)
