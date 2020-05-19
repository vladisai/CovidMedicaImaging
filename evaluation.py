import os
import logging
import pickle
import json

from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
import numpy as np
from torchxrayvision import datasets as xrv_datasets

import feature_extractors
import models
import data
from param import args


def get_folds_indices(dataset_length, folds):
    permutation = np.random.permutation(dataset_length)
    size = dataset_length // folds
    remainder = dataset_length % folds
    sizes = np.array([size] * folds)
    sizes[:remainder] += 1
    assert sizes.sum() == dataset_length
    points = np.cumsum(sizes)[:-1]
    split = np.split(permutation, points)
    return split


def get_folds(dataset, folds):
    path = f'./folds2_{len(dataset)}_{folds}'
    if os.path.exists(path):
        logging.info(f'Loading folds from {path}')
        with open(path, 'rb') as f:
            return pickle.load(f)
    logging.info(f'Building folds')

    patient_ids = dataset.csv['patientid'].unique()
    folds_indices =\
        get_folds_indices(len(patient_ids), folds)

    saved_folds = {'test': [], 'folds': []}

    def patient_indices_to_dataset_indices(indices):
        result_patient_ids = [patient_ids[i] for i in indices]
        result_dataset_indices = []
        for i in range(len(dataset)):
            if dataset.csv['patientid'].iloc[i] in result_patient_ids:
                result_dataset_indices.append(i)
        return result_dataset_indices

    test_indices = patient_indices_to_dataset_indices(folds_indices[0])
    all_indices = range(len(dataset))

    saved_folds['test'] = test_indices
    saved_folds['train'] = list(set(all_indices) - set(test_indices))

    for i in range(1, len(folds_indices)):
        val_indices = patient_indices_to_dataset_indices(folds_indices[i])
        train_indices = list(set(all_indices) -
                             set(val_indices) - set(test_indices))

        saved_folds['folds'].append(
            {'train': train_indices, 'val': val_indices})

    with open(path, 'wb') as f:
        return pickle.dump(saved_folds, f)

    logging.info(f'Saved folds to {path}')
    return saved_folds


def partitions_generator(dataset, n_folds, test):
    patient_ids = dataset.csv['patientid'].unique()
    logging.info(f'found {len(patient_ids)} patients in the dataset')
    folds = get_folds(dataset, n_folds)
    if test:
        yield (xrv_datasets.SubsetDataset(dataset, folds['train']),
               xrv_datasets.SubsetDataset(dataset, folds['test']))
        return

    for fold in folds['folds']:
        train_dataset = xrv_datasets.SubsetDataset(dataset, fold['train'])
        val_dataset = xrv_datasets.SubsetDataset(dataset, fold['val'])
        assert len(set(train_dataset.csv['patientid'].unique()) &
                   set(val_dataset.csv['patientid'].unique())) == 0
        yield train_dataset, val_dataset


def calculate_performance_metrics(predictions,
                                  predictions_hard,
                                  true_labels,
                                  classes):
    auc = [0] * len(classes)
    for i in range(len(classes)):
        if np.unique(true_labels[:, i]).shape[0] > 1:
            auc[i] = roc_auc_score(true_labels[:, i],
                                   predictions[i][:, 1])
            auc[i] = round(auc[i], 3)
        else:
            auc[i] = 'Undefined - only one label in test'
    report_result = classification_report(true_labels,
                                          predictions_hard,
                                          target_names=['no Covid', 'Covid'],
                                          output_dict=True)
    for i, cl in enumerate(classes):
        report_result[cl]['AUC'] = auc[i]
    return report_result


def calculate_average_performance(performance_reports):
    classes = ['no Covid', 'Covid']  # list(performance_reports[0].keys())
    final_result = {}
    for cl in classes:
        count = 0
        agg_result = performance_reports[0][cl].copy()
        for key in agg_result:
            agg_result[key] = 0.0
        for report in performance_reports:
            if 'AUC' not in report[cl] or type(report[cl]['AUC']) != str:
                count += 1
                for key in report[cl]:
                    agg_result[key] += report[cl][key]
        if count > 0:
            for key in agg_result:
                agg_result[key] /= count
        agg_result['non-zero support folds'] = count
        final_result[cl] = agg_result
    return final_result


def main():
    d_covid19 = data.COVID19_Dataset()

    d_covid19.pathologies = ['Covid']
    d_covid19.labels = d_covid19.labels[:, 2][:, None]
    logging.info(f'entire dataset length is {len(d_covid19)}')
    # feature_extractor = feature_extractors.NeuralNetFeatureExtractor()
    feature_extractor = \
        feature_extractors.FeatureExtractor(
            lbp=args.lbp,
            hog=args.hog,
            fft=args.fft,
            nn=args.nn,
            args_num=args.feature_num
        )
    Model = models.LogisticRegression

    metrics_history = []
    train_metrics_history = []

    for fold_idx, (train_dataset, eval_dataset) in \
            enumerate(partitions_generator(d_covid19, 10, args.test)):
        logging.info(f'fold number {fold_idx}: '
                     f'train size is {len(train_dataset)} '
                     f'eval size is {len(eval_dataset)}')

        if args.data_aug:
            train_dataset.dataset.data_aug = data.get_data_aug()
            train_dataset.dataset.data_aug = data.get_data_aug()

        features_train = feature_extractor.extract(train_dataset)
        labels_train = train_dataset.labels

        features_eval = feature_extractor.extract(eval_dataset)
        labels_eval = eval_dataset.labels

        feat_mean_train = np.mean(features_train, axis=0)
        features_train_centered = features_train - feat_mean_train
        features_test_centered = features_eval - feat_mean_train

        feat_std_train = np.std(features_train, axis=0)
        features_train = features_train_centered/(feat_std_train+1e-8)
        features_eval = features_test_centered/(feat_std_train+1e-8)

        if args.run_PCA:
            pca = PCA(n_components=args.pca_out_dim)
            pca.fit_transform(features_train)
            pca.transform(features_eval)

            print("Running PCA")

        model = Model()
        model.fit(features_train, labels_train)

        predictions_hard = model.predict(features_eval)
        predictions = model.predict_proba(features_eval)
        metrics = calculate_performance_metrics(predictions,
                                                predictions_hard,
                                                labels_eval,
                                                eval_dataset.pathologies)
        metrics_history.append(metrics)

        train_predictions_hard = model.predict(features_train)
        train_predictions = model.predict_proba(features_train)
        train_metrics = calculate_performance_metrics(train_predictions,
                                                      train_predictions_hard,
                                                      labels_train,
                                                      eval_dataset.pathologies)
        train_metrics_history.append(train_metrics)
        # logging.info(json.dumps(metrics, indent=4))

    average_metrics = calculate_average_performance(metrics_history)
    average_train_metrics = \
        calculate_average_performance(train_metrics_history)
    logging.info('Average train perf across all folds:')
    logging.info(json.dumps(average_train_metrics, indent=4))
    logging.info('Average across all folds:')
    logging.info(json.dumps(average_metrics, indent=4))


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
