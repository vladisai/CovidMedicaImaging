import logging
from sklearn.metrics import roc_auc_score
import numpy as np
from torchxrayvision import datasets as xrv_datasets

import feature_extractors
import models
import data


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
    d_covid19 = data.CombinedDataset()
    logging.info(f'entire dataset length is {len(d_covid19)}')
    feature_extractor = feature_extractors.NeuralNetFeatureExtractor()
    Model = models.LogisticRegression

    performance_history = [[] for _ in range(len(d_covid19.pathologies))]

    for fold_idx, (train_dataset, test_dataset) in \
            enumerate(partitions_generator(d_covid19, 2)):
        logging.info(f'fold number {fold_idx}: '\
                      f'train size is {len(train_dataset)} '\
                      f'test size is {len(test_dataset)}')

        features_train = feature_extractor.extract(train_dataset)
        labels_train = train_dataset.labels
        features_test = feature_extractor.extract(test_dataset)
        labels_test = test_dataset.labels

        model = Model()
        model.fit(features_train, labels_train)
        predictions = model.predict(features_test)

        performance = [0] * len(test_dataset.pathologies)
        per_class_counts = labels_test.sum(axis=0).astype(np.int)

        for i in range(len(test_dataset.pathologies)):
            if np.unique(labels_test[:, i]).shape[0] > 1:
                performance[i] = roc_auc_score(labels_test[:, i],
                                               predictions[i][:, 1])
                performance[i] = round(performance[i], 3)
                performance_history[i].append(performance[i])
            else:
                performance[i] = 'Undefined - only one label in test'

        performance = list(zip(test_dataset.pathologies, performance))
        logging.info(f'At fold {fold_idx} per class AUC is:')
        for i, (k, v) in enumerate(performance):
            logging.info(f'\t{k} : {v}'\
                         f'({per_class_counts[i]}/{labels_test.shape[0]}'\
                         'postive in test)')

    logging.info(f'Average per class AUC across all folds:')
    for k, v in zip(d_covid19.pathologies, performance_history):
        if len(v) > 0:
            avg_auc = f'{np.mean(v):.3f} (out of {len(v)} folds)'
        else:
            avg_auc = 'Undefined - only one unique label value'
        logging.info(f'\t{k} : {avg_auc}')


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
