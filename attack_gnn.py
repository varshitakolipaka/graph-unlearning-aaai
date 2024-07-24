

import os
from framework import get_model
from framework.data_loader import get_original_data
from framework.trainer.label_poison import LabelPoisonTrainer
from framework.training_args import parse_args
from framework.utils import seed_everything
import torch


def get_data(args):
    data = get_original_data(args.dataset)
    return data
    
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
    
    lp_trainer = LabelPoisonTrainer(args)
    
    data = lp_trainer.attack(data, args.df_size, args.random_seed)
    print(data)
    
    if args.gnn not in ['rgcn', 'rgat']:
        args.in_dim = data.x.shape[1]
    
    model = get_model(args)
    
    parameters_to_optimize = [
        {'params': [p for n, p in model.named_parameters()], 'weight_decay': args.weight_decay}
    ]
    print('parameters_to_optimize', [n for n, p in model.named_parameters()])
    
    optimizer = torch.optim.Adam(parameters_to_optimize, lr=args.lr)
    
    lp_trainer.train(model, data, optimizer, args)
    test_res = lp_trainer.test(model, data)
    print(test_res[-1])
    
if __name__ == "__main__":
    main()