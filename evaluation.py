import logging
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import multilabel_confusion_matrix
import numpy as np
from torchxrayvision import datasets as xrv_datasets
from sklearn.metrics import accuracy_score
import feature_extractors
import models
import data

def PCA(X):
    cov = np.dot(X.T, X) / X.shape[0] 
    U,S,V = np.linalg.svd(cov)
    Xrot_reduced = np.dot(X, U[:,:500])
    return Xrot_reduced

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
    patient_ids = dataset.csv['patientid'].unique()
    logging.debug(f'found {len(patient_ids)} patients in the dataset')
    test_folds_indices =\
        get_test_folds_indices(len(patient_ids), folds)
    for test_indices in test_folds_indices:
        test_patient_ids = [patient_ids[i] for i in test_indices]
        test_mapping = np.zeros(len(dataset))
        for i in range(len(dataset)):
            if dataset.csv['patientid'].iloc[i] in test_patient_ids:
                test_mapping[i] = 1
        train_indices = np.argwhere(test_mapping == 0).flatten()
        test_indices = np.argwhere(test_mapping == 1).flatten()
        train_dataset = xrv_datasets.SubsetDataset(dataset, train_indices)
        test_dataset = xrv_datasets.SubsetDataset(dataset, test_indices)
        yield train_dataset, test_dataset


def main():
    d_covid19 = data.CombinedDataset()
    logging.info(f'entire dataset length is {len(d_covid19)}')
    #feature_extractor = feature_extractors.NeuralNetFeatureExtractor()
    feature_extractor = feature_extractors.FeatureExtractor()
    Model = models.LogisticRegression

    performance_history = [[] for _ in range(len(d_covid19.pathologies))]

    for fold_idx, (train_dataset, test_dataset) in \
            enumerate(partitions_generator(d_covid19, 2)):
        logging.info(f'fold number {fold_idx}: '
                     f'train size is {len(train_dataset)} '
                     f'test size is {len(test_dataset)}')

        features_train = feature_extractor.extract(train_dataset)
        labels_train = train_dataset.labels
        assert features_train.shape[0] == len(train_dataset)
        assert labels_train.shape[0] == len(train_dataset)

        features_test = feature_extractor.extract(test_dataset)
        labels_test = test_dataset.labels
        assert features_test.shape[0] == len(test_dataset)
        assert labels_test.shape[0] == len(test_dataset)
        
        #feat_mean_train = np.mean(features_train, axis = 0)
        #features_train = PCA(features_train-feat_mean_train)
        #features_test = PCA(features_test-feat_mean_train)


        model = Model()
        model.fit(features_train, labels_train)
        predictions = model.predict_proba(features_test)

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
            logging.info(f'\t{k} : {v}'
                         f'({per_class_counts[i]}/{labels_test.shape[0]}'
                         'postive in test)')
        preds = model.predict(features_test)
        #print(f'Confusion matrix: {multilabel_confusion_matrix(labels_test, preds, labels=test_dataset.pathologies)}')
        print(f'Classification report:{classification_report(labels_test, preds, target_names=test_dataset.pathologies)}')
        print(f'Accuracy: {accuracy_score(labels_test, preds)}')
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
