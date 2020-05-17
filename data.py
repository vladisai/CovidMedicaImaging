import os

import pandas as pd
import numpy as np
from skimage.io import imread
from torch.utils.data import Dataset, ConcatDataset
import torchvision

from torchxrayvision import datasets as xrv_datasets


def get_labels(dataset):
    y = []
    for sample in dataset:
        y.append(sample['lab'])
    y = np.stack(y)
    return y


def get_default_transform():
    return torchvision.transforms.Compose([xrv_datasets.XRayCenterCrop(),
                                           xrv_datasets.XRayResizer(224)])


class ShenzhenDataset(Dataset):
    """
    Dataset from https://lhncbc.nlm.nih.gov/publication/pub9931
    Only the 'no fidining' images were selected.
    Contains 326 samples.
    """
    DATA_PATH = '/misc/vlgscratch4/LecunGroup/vlad/machine_learning_chest_datasets/shenzhen'
    IMAGES_PATH = os.path.join(DATA_PATH, 'CXR_png_resized')
    LABELS_PATH = os.path.join(DATA_PATH, 'labels.csv')
    MAX_VAL = 255

    def __init__(self,
                 transform=get_default_transform(),
                 data_aug=None):
        self.image_paths = pd.read_csv(self.LABELS_PATH)
        self.transform = transform
        self.data_aug = data_aug
        # the dataset has only healthy xrays, all labels are 0
        self.pathologies = \
            ['ARDS',
             'Bacterial Pneumonia',
             'COVID-19',
             'Chlamydophila',
             'Fungal Pneumonia',
             'Klebsiella',
             'Legionella',
             'MERS',
             'No Finding',
             'Pneumocystis',
             'Pneumonia',
             'SARS',
             'Streptococcus',
             'Viral Pneumonia']
        self.label = np.zeros(len(self.pathologies))
        self.label[self.pathologies.index('No Finding')] = 1
        self.label = self.label.astype(np.float32)
        self.labels = np.tile(self.label, [len(self.image_paths), 1])
        self.csv = self.image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = imread(os.path.join(self.IMAGES_PATH,
                                  self.image_paths['path'].iloc[idx]))
        img = xrv_datasets.normalize(img, self.MAX_VAL)
        # Add color channel
        img = img[None, :, :]
        if self.transform is not None:
            img = self.transform(img)
        if self.data_aug is not None:
            img = self.data_aug(img)
        return {'img': img, 'lab': self.label, 'idx':idx}


class COVID19_Dataset(xrv_datasets.COVID19_Dataset):
    """
    Wrapper for torchxrayvision covid19 dataset
    that includes the paths to the dataset on cassio
    Contains 194 samples.
    """

    COVID_19_DATASET_PATH = '/misc/vlgscratch4/LecunGroup/vlad/machine_learning_chest_datasets/covid-chestxray-dataset'
    COVID_19_DATASET_IMAGES_PATH = os.path.join(COVID_19_DATASET_PATH, 'images')
    COVID_19_DATASET_METADATA_PATH = os.path.join(
        COVID_19_DATASET_PATH, 'metadata.csv')

    def __init__(self, *args, **kwargs):
        super().__init__(imgpath=self.COVID_19_DATASET_IMAGES_PATH,
                         csvpath=self.COVID_19_DATASET_METADATA_PATH,
                         *args,
                         **kwargs)


class CombinedDataset(xrv_datasets.Merge_Dataset):
    """Combination of Shenzhen and Covid19 examples.
    This is the dataset for evaluation, which has better
    class balance for 'no finding' images that just
    the covid19 dataset.
    Contains 520 samples in total.
    """
    def __init__(self, transform=get_default_transform(), data_aug=None):
        self.covid_dataset = COVID19_Dataset(transform=transform, data_aug=data_aug)
        self.shenzhen_dataset = ShenzhenDataset(transform=transform, data_aug=data_aug)
        super().__init__([self.covid_dataset, self.shenzhen_dataset])
