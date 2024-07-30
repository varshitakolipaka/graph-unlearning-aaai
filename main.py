import torch
from framework import utils
from framework.training_args import parse_args
from models.deletion import GCNDelete
from trainers.base import Trainer
from trainers.gnndelete import GNNDeleteNodeembTrainer
from trainers.gnndelete_ni import GNNDeleteNITrainer
from attacks.edge_attack import edge_attack_random_nodes
from attacks.label_flip import label_flip_attack

args = parse_args()
utils.seed_everything(args.random_seed)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# dataset
print("==TRAINING==")
clean_data= utils.get_original_data(args.dataset)
clean_model = GCNDelete(clean_data.num_features, 256, clean_data.num_classes)
optimizer = torch.optim.Adam(clean_model.parameters(), lr=0.01, weight_decay=5e-4)
clean_trainer = Trainer(clean_model, clean_data, optimizer, args.training_epochs)
clean_trainer.train()


print("==POISONING==")
if args.attack_type=="label":
    poisoned_data, poisoned_indices = label_flip_attack(clean_data, args.df_size, args.random_seed)
elif args.attack_type=="edge":
    poisoned_data, poisoned_indices = edge_attack_random_nodes(clean_data, args.df_size, args.random_seed)
poisoned_data= poisoned_data.to(device)
poisoned_model = GCNDelete(poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes)
optimizer = torch.optim.Adam(poisoned_model.parameters(), lr=0.01, weight_decay=5e-4)
poisoned_trainer = Trainer(poisoned_model, poisoned_data, optimizer)
poisoned_trainer.train()

print("==UNLEARNING==")
utils.find_masks(poisoned_data, poisoned_indices, attack_type=args.attack_type)
optimizer1 = torch.optim.Adam(poisoned_model.deletion1.parameters(), lr=0.025)
optimizer2 = torch.optim.Adam(poisoned_model.deletion2.parameters(), lr=0.025)
optimizer_unlearn = [optimizer1, optimizer2]
gnndelete= GNNDeleteNITrainer(poisoned_model, poisoned_data, optimizer_unlearn, args)
gnndelete.train()