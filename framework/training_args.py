import argparse


num_edge_type_mapping = {
    'FB15k-237': 237,
    'WordNet18': 18,
    'WordNet18RR': 11,
    'ogbl-biokg': 51
}

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--attack_type', type=str, default='random', help='attack type', choices=["label", "edge", "random"])
    parser.add_argument('--unlearning_model', type=str, default='megu', help='unlearning method', choices=["original", "gradient_ascent", "gnndelete", "gnndelete_ni", "gif", "utu", "contrastive", "retrain", "scrub", "megu"])
    parser.add_argument('--gnn', type=str, default='gcn', help='GNN architecture', choices=['gcn', 'gat', 'gin'])
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--unlearning_epochs', type=int, default=500, help='number of epochs to unlearn for')
    parser.add_argument('--request', type=str, default='node', help='unlearning request', choices=['node', 'edge'])

    # Data
    parser.add_argument('--df_size', type=float, default=0.1, help='Forgetting Fraction')
    parser.add_argument('--dataset', type=str, default='Cora_p', help='dataset')
    parser.add_argument('--random_seed', type=int, default=0, help='random seed')

    # Training
    parser.add_argument('--unlearn_lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00005, help='weight decay')
    parser.add_argument('--optimizer', type=str, default='Aam', help='optimizer to use')
    parser.add_argument('--training_epochs', type=int, default=50, help='number of epochs to train')
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
    parser.add_argument('--unlearn_iters', type=int, default=51, help='number of epochs to train (default: 31)')
    parser.add_argument('--kd_T', type=float, default=1, help='Knowledge distilation temperature for SCRUB')
    parser.add_argument('--scrubAlpha', type=float, default=1, help='KL from og_model constant for SCRUB, higher incentivizes closeness to ogmodel')
    parser.add_argument('--msteps', type=int, default=15, help='Maximization steps on forget set for SCRUB')

    # contrastive
    parser.add_argument('--contrastive_epochs_1', type=int, default=30, help="epochs for contrastive unlearning")
    parser.add_argument('--contrastive_epochs_2', type=int, default=10, help="epochs for contrastive unlearning")
    parser.add_argument('--contrastive_margin', type=int, default=500, help="margin for the contrastive loss")
    parser.add_argument('--contrastive_lambda', type=float, default=0.8, help="weight for the task loss [1 - lambda] is used for the contrastive loss")

    # megu
    parser.add_argument('--is_vary', type=bool, default=False, help='control whether to use multiprocess')
    parser.add_argument('--cuda', type=int, default=0, help='specify gpu')
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--inductive', type=str, default='normal', choices=['cluster-gcn', 'graphsaint', 'normal'])
    parser.add_argument('--num_runs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--test_batch_size', type=int, default=2048)
    parser.add_argument('--kappa', type=float, default=0.35)
    parser.add_argument('--alpha1', type=float, default=0.3)
    parser.add_argument('--alpha2', type=float, default=0.8)
    args = parser.parse_args()
    return args
