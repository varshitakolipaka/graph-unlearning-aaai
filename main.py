from framework import utils
from framework.training_args import parse_args
import torch
from functions import train, poison, unlearn
import optuna
from optuna.samplers import TPESampler
from functools import partial
from tuning import objective

device = "cuda" if torch.cuda.is_available() else "cpu"
args = parse_args()
utils.set_global_seed(args.random_seed)

if __name__ == "__main__":
    utils.seed_everything(args.random_seed)
    clean_data = train(load=False)
    poisoned_data, poisoned_indices, poisoned_model = poison(clean_data)

    if args.tuning == True:
        utils.find_masks(
            poisoned_data, poisoned_indices, args, attack_type=args.attack_type
        )
        objective_func = partial(objective, model=poisoned_model, data=poisoned_data)
        study = optuna.create_study(
            sampler=TPESampler(seed=args.random_seed),
            direction="maximize",
            study_name=f"{args.dataset}_{args.attack_type}_{args.df_size}_{args.unlearning_model}_{args.random_seed}",
            load_if_exists=True,
            storage="sqlite:///final_sanity_check2.db",
        )

        print("==OPTIMIZING==")
        study.optimize(objective_func, n_trials=100)
    else:
        unlearn(poisoned_data, poisoned_indices, poisoned_model)
