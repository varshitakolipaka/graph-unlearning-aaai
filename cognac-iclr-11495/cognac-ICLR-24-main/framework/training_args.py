import argparse
import time


num_edge_type_mapping = {
    'FB15k-237': 237,
    'WordNet18': 18,
    'WordNet18RR': 11,
    'ogbl-biokg': 51
}

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_ratio', type=float, default=0.6, help='train ratio')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='train ratio')
    parser.add_argument('--attack_type', type=str, default='label', help='attack type', choices=["label", "edge", "random", "trigger", 'label_strong'])
    parser.add_argument('--unlearning_model', type=str, default='scrub', help='unlearning method', choices=["original", "gradient_ascent", "gnndelete", "gnndelete_ni", "gif", "utu", "contrastive", "retrain", "scrub", "megu", "contra_2", "ssd", "grub", "yaum", 'contrascent', 'cacdc', 'scrub_no_kl_combined', 'scrub_no_kl'])
    parser.add_argument('--gnn', type=str, default='gcn', help='GNN architecture', choices=['gcn', 'gat', 'gin'])
    # parser.add_argument('--in_dim', type=int, default=128, help='input dimension')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--unlearning_epochs', type=int, default=200, help='number of epochs to unlearn for')
    # parser.add_argument('--out_dim', type=int, default=64, help='output dimension')
    parser.add_argument('--request', type=str, default='node', help='unlearning request', choices=['node', 'edge'])
    parser.add_argument('--edge_attack_type', type=str, default='specific', help='edge attack type', choices=['random', 'specific'])

    # Data
    parser.add_argument('--data_dir', type=str, default='./data', help='data dir')
    parser.add_argument('--db_name', type=str, default='hp_tuning', help='db name')
    
    # parser.add_argument('--df', type=str, default='in', help='Df set to use')
    # parser.add_argument('--df_idx', type=str, default=None, help='indices of data to be deleted')
    parser.add_argument('--df_size', type=float, default=0.5, help='Forgetting Fraction')
    parser.add_argument('--test_poison_fraction', type=float, default=0.2, help='Test Poisoning Fraction')
    parser.add_argument('--dataset', type=str, default='Cora', help='dataset')
    parser.add_argument('--random_seed', type=int, default=0, help='random seed')
    parser.add_argument('--trigger_size', type=int, default=200, help='Poison Tensor Size')
    parser.add_argument('--victim_class', type=int, default=0, help='class to add trigger to')
    parser.add_argument('--target_class', type=int, default=1, help='class to add trigger to')
    # parser.add_argument('--batch_size', type=int, default=2048, help='batch size for GraphSAINTRandomWalk sampler')
    # parser.add_argument('--walk_length', type=int, default=2, help='random walk length for GraphSAINTRandomWalk sampler')
    # parser.add_argument('--num_steps', type=int, default=32, help='number of steps for GraphSAINTRandomWalk sampler')

    # Training
    # parser.add_argument("--suffix", type=str, default=None, help="name suffix for #wandb run")
    # parser.add_argument("--mode", type=str, default="disabled", help="#wandb mode")
    parser.add_argument('--train_lr', type=float, default=0.005191475570177285, help='initial learning rate')
    parser.add_argument('--unlearn_lr', type=float, default=0.015, help='unlearn learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00016211813194850176, help='weight decay')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer to use')
    parser.add_argument('--training_epochs', type=int, default=1208, help='number of epochs to train')
    parser.add_argument('--valid_freq', type=int, default=30, help='# of epochs to do validation')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='checkpoint folder')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha in loss function')
    parser.add_argument('--neg_sample_random', type=str, default='non_connected', help='type of negative samples for randomness')
    parser.add_argument('--loss_fct', type=str, default='mse_mean', help='loss function. one of {mse, kld, cosine}')
    parser.add_argument('--loss_type', type=str, default='both_layerwise', help='type of loss. one of {both_all, both_layerwise, only2_layerwise, only2_all, only1}')

    # GraphEraser
    parser.add_argument('--num_clusters', type=int, default=10, help='top k for evaluation')
    parser.add_argument('--kmeans_max_iters', type=int, default=1, help='top k for evaluation')
    parser.add_argument('--shard_size_delta', type=float, default=0.005)
    parser.add_argument('--terminate_delta', type=int, default=0)

    # GraphEditor
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--num_remove_links', type=int, default=11)
    parser.add_argument('--parallel_unlearning', type=int, default=4)
    parser.add_argument('--lam', type=float, default=0)
    parser.add_argument('--regen_feats', action='store_true')
    parser.add_argument('--regen_neighbors', action='store_true')
    parser.add_argument('--regen_links', action='store_true')
    parser.add_argument('--regen_subgraphs', action='store_true')
    parser.add_argument('--hop_neighbors', type=int, default=20)


    # Evaluation
    parser.add_argument('--topk', type=int, default=500, help='top k for evaluation')
    parser.add_argument('--eval_on_cpu', type=bool, default=False, help='whether to evaluate on CPU')

    # KG
    parser.add_argument('--num_edge_type', type=int, default=None, help='number of edges types')

    # GIF
    parser.add_argument('--iteration', type=int, default=100)
    parser.add_argument('--scale', type=int, default=100000)
    parser.add_argument('--damp', type=float, default=0.1)

    # Scrub
    parser.add_argument('--unlearn_iters', type=int, default=150, help='number of epochs to train (default: 31)')
    parser.add_argument('--kd_T', type=float, default=4, help='Knowledge distilation temperature for SCRUB')
    parser.add_argument('--scrubAlpha', type=float, default=0, help='KL from og_model constant for SCRUB, higher incentivizes closeness to ogmodel')
    parser.add_argument('--msteps', type=int, default=25, help='Maximization steps on forget set for SCRUB')
    parser.add_argument('--ascent_lr', type=float, default=0.0025, help='Learning rate for gradient ascent steps')
    parser.add_argument('--descent_lr', type=float, default=0.015, help='Learning rate for gradient descent steps')  


    # contrastive
    parser.add_argument('--contrastive_epochs_1', type=int, default=0, help="epochs for contrastive unlearning")
    parser.add_argument('--contrastive_epochs_2', type=int, default=10, help="epochs for contrastive unlearning")
    parser.add_argument('--maximise_epochs', type=int, default=0, help="epochs for grad asc in contrastive unlearning")
    parser.add_argument('--contrastive_margin', type=float, default=500, help="margin for the contrastive loss")
    parser.add_argument('--contrastive_lambda', type=float, default=0.8, help="weight for the task loss [1 - lambda] is used for the contrastive loss")
    parser.add_argument('--k_hop', type=int, default=2, help="number of hops for the data sampling")
    parser.add_argument('--contrastive_frac', type=float, default=0.1, help="fraction of nodes to sample for contrastive loss")
    parser.add_argument('--steps', type=int, default=10, help="steps of ascent and descent")
    parser.add_argument('--ascent_const', type=int, default=0.001, help="constant for ascent")

    # MEGU
    parser.add_argument('--kappa', type=float, default=0.01)
    parser.add_argument('--alpha1', type=float, default=0.8)
    parser.add_argument('--alpha2', type=float, default=0.5)
    
    # SSD
    parser.add_argument('--SSDdampening', type=float, default=10, help='SSD: lambda aka dampening constant, lower leads to more forgetting')
    parser.add_argument('--SSDselectwt', type=float, default=1, help='SSD: alpha aka selection weight, lower leads to more forgetting')
    
    # UTILITIES
    parser.add_argument('--embs_all', action='store_true', help='whether to plot embeddings in embs.py')
    parser.add_argument('--embs_unlearn', action='store_true', help='whether to plot embeddings in embs.py')
    
    # CORRECTIVE UNLEARNING
    parser.add_argument('--corrective_frac', type=float, default=1, help='fraction of nodes to sample for corrective unlearning (by default all nodes)')
    
    parser.add_argument('--log_name', type=str, default='default', help='log name')
    
    parser.add_argument('--linked', action='store_true', help='whether to use linked model')

    args = parser.parse_args()
    args.experiment_name = f"{args.dataset}_{args.attack_type}_{args.unlearning_model}_{args.gnn}_{args.corrective_frac}_{args.db_name}"
    return args
