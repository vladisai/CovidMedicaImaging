from models import Baseline
from sklearn.metrics import roc_auc_score
import logging
import numpy as np
from torchxrayvision import datasets as xrv_datasets
from data import COVID19_Dataset


def get_test_folds_indices(dataset_length, folds):
    permutation = np.random.permutation(dataset_length)
    size = dataset_length // folds
    remainder = dataset_length % folds
    sizes = np.array([size] * folds)
    sizes[:remainder] += 1
    assert sizes.sum() == dataset_length
    points = np.cumsum(sizes)[:-1]
    split = np.split(permutation, points)
    return split


def partitions_generator(dataset, folds):
    test_folds_indices = get_test_folds_indices(len(dataset), folds)
    for test_indices in test_folds_indices:
        train_mapping = np.ones(len(dataset))
        for i in test_indices:
            train_mapping[i] = 0
        train_indices = np.argwhere(train_mapping == 1).flatten()
        test_indices = np.argwhere(train_mapping == 0).flatten()
        train_dataset = xrv_datasets.SubsetDataset(dataset, train_indices)
        test_dataset = xrv_datasets.SubsetDataset(dataset, test_indices)
        yield train_dataset, test_dataset


def main():
    d_covid19 = COVID19_Dataset()
    logging.info(f'entire dataset length is {len(d_covid19)}')

    for i, (train_dataset, test_dataset) in enumerate(partitions_generator(d_covid19, 10)):
        logging.info(
            f'train size {len(train_dataset)}, test size {len(test_dataset)}')
        model = Baseline()
        model.fit(train_dataset)
        performance = np.zeros(len(test_dataset.pathologies))
        predictions = []
        ground_truth = []
        for sample in test_dataset:
            prediction = model.predict(sample)
            predictions.append(prediction)
            ground_truth.append(sample['lab'])

        predictions = np.stack(predictions)
        ground_truth = np.stack(ground_truth)

        for i in range(len(test_dataset.pathologies)):
            if np.unique(ground_truth[:, i]).shape[0] > 1:
                performance[i] = roc_auc_score(
                    ground_truth[:, i], predictions[:, i])
        performance /= len(test_dataset)
        logging.info(f'At fold {i} per class AUC is:\n{performance}')


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
