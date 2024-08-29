from framework import utils
from framework.training_args import parse_args
import torch

device= "cuda" if torch.cuda.is_available() else "cpu"
args = parse_args()
utils.set_global_seed(args.random_seed)

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

    obj = 0.5 * forg + 0.5 * util
    return obj