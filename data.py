import os

import pandas as pd
import numpy as np
from skimage.io import imread
from torch.utils.data import Dataset

from torchxrayvision import datasets as xrv_datasets


class ShenzhenDataset(Dataset):
    """Dataset from https://lhncbc.nlm.nih.gov/publication/pub9931
    Only the 'no fidining' images were selected.
    """
    DATA_PATH = '/misc/vlgscratch4/LecunGroup/vlad/machine_learning_chest_datasets/shenzhen'
    IMAGES_PATH = os.path.join(DATA_PATH, 'CXR_png_resized')
    LABELS_PATH = os.path.join(DATA_PATH, 'labels.csv')
    MAX_VAL = 255

    def __init__(self, label_shape=14):
        self.image_paths = pd.read_csv(self.LABELS_PATH)
        # the dataset has only healthy xrays, all labels are 0
        self.label = np.zeros(label_shape)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = imread(os.path.join(self.IMAGES_PATH, self.image_paths['path'].iloc[idx]))
        img = xrv_datasets.normalize(img, self.MAX_VAL)
        return {'img': img, 'lab': self.label}


class COVID19_Dataset(xrv_datasets.COVID19_Dataset):
    """Wrapper for torchxrayvision covid19 dataset
    that includes the paths to the dataset on cassio
    """

    COVID_19_DATASET_PATH = '/misc/vlgscratch4/LecunGroup/vlad/machine_learning_chest_datasets/covid-chestxray-dataset'
    COVID_19_DATASET_IMAGES_PATH = os.path.join(COVID_19_DATASET_PATH, 'images')
    COVID_19_DATASET_METADATA_PATH = os.path.join(COVID_19_DATASET_PATH, 'metadata.csv')

    def __init__(self, *args, **kwargs):
        super().__init__(imgpath=self.COVID_19_DATASET_IMAGES_PATH,
                         csvpath=self.COVID_19_DATASET_METADATA_PATH,
                         *args,
                         **kwargs)
