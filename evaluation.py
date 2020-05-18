import logging
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
import numpy as np
from torchxrayvision import datasets as xrv_datasets
import feature_extractors
import models
import data
import json
from param import args


def PCA(X):
    cov = np.dot(X.T, X) / X.shape[0]
    U, S, V = np.linalg.svd(cov)
    Xrot_reduced = np.dot(X, U[:, :args.pca_out_dim])
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
    logging.info(f'found {len(patient_ids)} patients in the dataset')
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

        assert len(set(train_dataset.csv['patientid'].unique()) &
                   set(test_dataset.csv['patientid'].unique())) == 0

        yield train_dataset, test_dataset


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
                                          target_names=classes,
                                          output_dict=True)
    for i, cl in enumerate(classes):
        report_result[cl]['AUC'] = auc[i]
    return report_result


def calculate_average_performance(performance_reports):
    classes = list(performance_reports[0].keys())
    final_result = {}
    for cl in classes:
        count = 0
        agg_result = performance_reports[0][cl].copy()
        for key in agg_result:
            agg_result[key] = 0.0
        for report in performance_reports:
            if report[cl]['support'] > 0:
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
    #d_covid19 = data.CombinedDataset()
    d_covid19 = data.COVID19_Dataset()
    logging.info(f'entire dataset length is {len(d_covid19)}')
    #feature_extractor = feature_extractors.NeuralNetFeatureExtractor()
    feature_extractor = feature_extractors.FeatureExtractor(lbp=args.lbp, hog=args.hog,fft=args.fft)
    Model = models.LogisticRegression

    metrics_history = []

    for fold_idx, (train_dataset, test_dataset) in \
            enumerate(partitions_generator(d_covid19, 10)):
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
        
        if args.PCA:
            print("Running PCA")
            feat_mean_train = np.mean(features_train, axis=0)
            features_train = PCA(features_train-feat_mean_train)
            features_test = PCA(features_test-feat_mean_train)

        model = Model()
        model.fit(features_train, labels_train)

        predictions_hard = model.predict(features_test)
        predictions = model.predict_proba(features_test)

        metrics = calculate_performance_metrics(predictions,
                                                predictions_hard,
                                                labels_test,
                                                test_dataset.pathologies)
        metrics_history.append(metrics)
        logging.info(json.dumps(metrics, indent=4))

    average_metrics = calculate_average_performance(metrics_history)
    logging.info('Average across all folds:')
    logging.info(json.dumps(average_metrics, indent=4))


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
