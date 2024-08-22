import copy
import torch
from framework import utils
from framework.training_args import parse_args
from models.deletion import GCNDelete
from models.models import GCN
from trainers.base import Trainer
from attacks.edge_attack import edge_attack_specific_nodes, edge_attack_random_nodes
from attacks.label_flip import label_flip_attack
from attacks.feature_attack import trigger_attack
from logger import Logger

logger = Logger("run_logs.json")

args = parse_args()
logger.log_arguments(args)

utils.seed_everything(args.random_seed)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(args.attack_type)
# dataset
print("==TRAINING==")
clean_data = utils.get_original_data(args.dataset)
utils.train_test_split(clean_data, args.random_seed, args.train_ratio)
utils.prints_stats(clean_data)
if "gnndelete" in args.unlearning_model:
    clean_model = GCNDelete(
        clean_data.num_features, args.hidden_dim, clean_data.num_classes
    )
else:
    clean_model = GCN(clean_data.num_features, args.hidden_dim, clean_data.num_classes)

optimizer = torch.optim.Adam(
    clean_model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay
)
clean_trainer = Trainer(clean_model, clean_data, optimizer, args.training_epochs)
clean_trainer.train()

print("\n==POISONING==")
print(f"Attack type: {args.attack_type}")
if args.attack_type == "label":
    poisoned_data, poisoned_indices = label_flip_attack(
        clean_data, args.df_size, args.random_seed
    )
elif args.attack_type == "edge":
    if args.edge_attack_type == "specific":
        poisoned_data, poisoned_indices = edge_attack_specific_nodes(
            clean_data, args.df_size, args.random_seed, class1=57, class2=33
        )
    elif args.edge_attack_type == "random":
        poisoned_data, poisoned_indices = edge_attack_random_nodes(
            clean_data, args.df_size, args.random_seed
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
        args.test_poison_fraction,
        target_class=57,
    )
    print(f"Number of poisoned nodes in train: {len(poisoned_indices)}")
    print(f"Number of poisoned nodes in test: {sum(poisoned_data.poison_test_mask)}")
poisoned_data = poisoned_data.to(device)

if "gnndelete" in args.unlearning_model:
    poisoned_model = GCNDelete(
        poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes
    )
else:
    poisoned_model = GCN(
        poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes
    )

optimizer = torch.optim.Adam(poisoned_model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
poisoned_trainer = Trainer(
    poisoned_model, poisoned_data, optimizer, args.training_epochs
)
poisoned_trainer.train()
utils.find_masks(poisoned_data, poisoned_indices, args, attack_type=args.attack_type)

clean_forget_ability, clean_utility = clean_trainer.get_score(
    args.attack_type, poisoned_trainer.class1, poisoned_trainer.class2
)
poisoned_forget_ability, poisoned_utility = poisoned_trainer.get_score(args.attack_type)

print(
    f"==Clean Model==\nForget Ability: {clean_forget_ability}, Utility: {clean_utility}"
)
print(
    f"==Poisoned Model==\nForget Ability: {poisoned_forget_ability}, Utility: {poisoned_utility}"
)

logger.log_result(
    args.random_seed,
    "Original",
    {"Forget Ability": clean_forget_ability, "Utility": clean_utility},
)

logger.log_result(
    args.random_seed,
    "Poisoned",
    {"Forget Ability": poisoned_forget_ability, "Utility": poisoned_utility},
)

print("\n==UNLEARNING==")
print(f"Unlearning model: {args.unlearning_model}")
if "gnndelete" in args.unlearning_model:
    unlearn_model = GCNDelete(
        poisoned_data.num_features,
        args.hidden_dim,
        poisoned_data.num_classes,
        mask_1hop=poisoned_data.sdf_node_1hop_mask,
        mask_2hop=poisoned_data.sdf_node_2hop_mask,
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

unlearn_forget_ability, unlearn_utility = unlearn_trainer.get_score(
    args.attack_type, poisoned_trainer.class1, poisoned_trainer.class2
)

print(
    f"==Unlearned Model==\nForget Ability: {unlearn_forget_ability}, Utility: {unlearn_utility}"
)

## LOGGING RESULTS ##
logger.log_result(
    args.random_seed,
    args.unlearning_model,
    {"Forget Ability": unlearn_forget_ability, "Utility": unlearn_utility},
)
