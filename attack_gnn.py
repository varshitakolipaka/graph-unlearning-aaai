

import os
from framework import get_model
#import wandb
from framework.data_loader import get_original_data
from framework.trainer.label_poison import LabelPoisonTrainer
from framework.trainer.edge_poison import EdgePoisonTrainer
from framework.training_args import parse_args
from framework.utils import seed_everything
import torch
from framework import get_model, get_trainer

def get_data(args):
    data = get_original_data(args.dataset)
    return data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    args = parse_args()
    original_path = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, 'original_node', str(args.random_seed))
    shadow_path_all = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, 'shadow_all', str(args.random_seed))
    args.shadow_dir = shadow_path_all
    if not os.path.exists(shadow_path_all):
        os.makedirs(shadow_path_all, exist_ok=True)
    args.checkpoint_dir = os.path.join(
        args.checkpoint_dir, args.dataset, args.gnn, args.unlearning_model,
        '-'.join([str(i) for i in [args.df, args.df_size, args.random_seed]]))
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    seed_everything(args.random_seed)

    data = get_data(args)

    if(args.attack_type=="label"):
        args.unlearning_model = 'lf_attack'
        lp_trainer = LabelPoisonTrainer(args)
        data, flipped_indices = lp_trainer.label_flip_attack(data, args.df_size, args.random_seed)
    elif(args.attack_type=="edge"):
        args.unlearning_model = 'edge_attack'
        edge_trainer= EdgePoisonTrainer(args)
        aug, poisoned_indices= edge_trainer.edge_attack_random_nodes(data, args.df_size, args.random_seed)
        data.edge_index= aug
    print(data)

    if args.gnn not in ['rgcn', 'rgat']:
        args.in_dim = data.x.shape[1]

    #wandb.init(config=args, group="attacking", name="{}_{}".format(args.dataset, args.gnn), mode=args.mode)
    model = get_model(args)

    parameters_to_optimize = [
        {'params': [p for n, p in model.named_parameters()], 'weight_decay': args.weight_decay}
    ]
    print('parameters_to_optimize', [n for n, p in model.named_parameters()])

    optimizer = torch.optim.Adam(parameters_to_optimize, lr=args.lr)

     # Train
    trainer = get_trainer(args)
    print(trainer)
    model = model.to(device)
    trainer.train(model, data, optimizer, args)

    # lp_trainer.train(model, data, optimizer, args)
    # test_res = lp_trainer.test(model, data)
    # print(test_res[-1])

    # Test
    test_results = trainer.test(model, data)
    trainer.save_log()
    print(test_results[-1])
    #wandb.finish()

if __name__ == "__main__":
    main()