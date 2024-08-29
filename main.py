from framework import utils
from framework.training_args import parse_args
import torch
from functions import train, poison, unlearn

device= "cuda" if torch.cuda.is_available() else "cpu"
args = parse_args()

if __name__ == "__main__":
    utils.seed_everything(args.random_seed)
    clean_data = train(load=False)
    poisoned_data, poisoned_indices, poisoned_model = poison(clean_data)
    unlearn(poisoned_data, poisoned_indices, poisoned_model)