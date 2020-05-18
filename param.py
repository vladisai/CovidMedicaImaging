import argparse
import random

import numpy as np
import torch

def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', type=int, default=20, help='random seed')
        parser.add_argument('--classifier', type=str, default='LogisticRegression')
        parser.add_argument('--PCA', action='store_const', default=False, const=True)
        parser.add_argument('--pca_out_dim', type=int, default=1000, help='Number of dimensions to keep after PCA')
        args = parser.parse_args()

        # Set seeds
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        return args

args = parse_args()
