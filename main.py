import copy
import torch
from framework import utils
from framework.training_args import parse_args
from models.deletion import GCNDelete
from models.models import GCN
from trainers.base import Trainer
from attacks.edge_attack import edge_attack_random_nodes
from attacks.label_flip import label_flip_attack
from trainers.megu import ExpMEGU
args = parse_args()
utils.seed_everything(args.random_seed)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print("==TRAINING==")
clean_data= utils.get_original_data(args.dataset)
if "gnndelete" in args.unlearning_model:
    clean_model = GCNDelete(clean_data.num_features, args.hidden_dim, clean_data.num_classes)
else:
    clean_model = GCN(clean_data.num_features, args.hidden_dim, clean_data.num_classes)

optimizer = torch.optim.Adam(clean_model.parameters(), lr=0.01, weight_decay=5e-4)
clean_trainer = Trainer(clean_model, clean_data, optimizer, args.training_epochs)
clean_trainer.train()
score= clean_trainer.get_silhouette_scores()
print(score)

og_data= copy.deepcopy(clean_data)

print("==POISONING==")
if args.attack_type=="label":
    poisoned_data, poisoned_indices = label_flip_attack(clean_data, args.df_size, args.random_seed)
    print("meowww")
elif args.attack_type=="edge":
    poisoned_data, poisoned_indices = edge_attack_random_nodes(clean_data, args.df_size, args.random_seed)
    
elif args.attack_type=="random":
    poisoned_data = copy.deepcopy(clean_data)
    poisoned_indices = torch.randperm(clean_data.num_nodes)[:int(clean_data.num_nodes*args.df_size)]
poisoned_data= poisoned_data.to(device)

if "gnndelete" in args.unlearning_model:
    poisoned_model = GCNDelete(poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes)
else:
    poisoned_model = GCN(poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes)

optimizer = torch.optim.Adam(poisoned_model.parameters(), lr=0.01, weight_decay=5e-4)
poisoned_trainer = Trainer(poisoned_model, poisoned_data, optimizer)
poisoned_trainer.train()

print("==UNLEARNING==")

utils.find_masks(poisoned_data, poisoned_indices, attack_type=args.attack_type)

og_poisoned_model = copy.deepcopy(poisoned_model)
og_poisoned_data = copy.deepcopy(poisoned_data)

if "gnndelete" in args.unlearning_model:
    unlearn_model = GCNDelete(poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes, mask_1hop=poisoned_data.sdf_node_1hop_mask, mask_2hop=poisoned_data.sdf_node_2hop_mask)
    unlearn_model.load_state_dict(poisoned_model.state_dict())

    optimizer_unlearn= utils.get_optimizer(args, unlearn_model)
    unlearn_trainer= utils.get_trainer(args, unlearn_model, poisoned_data, optimizer_unlearn)
    unlearn_trainer.train()
elif "retrain" in args.unlearning_model:
    unlearn_model = GCN(poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes)
    optimizer_unlearn= utils.get_optimizer(args, unlearn_model)
    unlearn_trainer= utils.get_trainer(args, unlearn_model, poisoned_data, optimizer_unlearn)
    unlearn_trainer.train()

else:
    optimizer_unlearn= utils.get_optimizer(args, poisoned_model)
    unlearn_trainer= utils.get_trainer(args, poisoned_model, poisoned_data, optimizer_unlearn)
    unlearn_trainer.train()

dt_acc, dt_f1, df_acc, df_f1 = poisoned_trainer.evaluate1(poisoned_data)
print("Accuracies: ", dt_acc, dt_f1, df_acc, df_f1)

dt_acc, dt_f1, df_acc, df_f1 = unlearn_trainer.evaluate1(poisoned_data)
print("Accuracies: ", dt_acc, dt_f1, df_acc, df_f1)