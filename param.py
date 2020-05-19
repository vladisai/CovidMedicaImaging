import argparse
import random

import numpy as np
import torch

def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', type=int, default=20, help='random seed')
        parser.add_argument('--feature_num', type=int, default=10000, help='number of features with FeatureExtractor')
        parser.add_argument('--classifier', type=str, default='LogisticRegression')
        parser.add_argument('--run_PCA', type=bool, default=False)
        parser.add_argument('--pca_out_dim', type=int, default=100, help='Number of dimensions to keep after PCA')
        parser.add_argument('--lbp', type=bool, default=False)
        parser.add_argument('--hog', type=bool, default=False)
        parser.add_argument('--fft', type=bool, default=False)
        parser.add_argument('--nn', type=bool, default=False)
        parser.add_argument('--data_aug', action='store_const', default=False, const=True)
        parser.add_argument('--test', action='store_const', default=False, const=True)
        args = parser.parse_args()

        # Set seeds
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        return args

args = parse_args()
