'''
Driving safety binary classification

python main.py train -l "labels/part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv" -f "features/" -v

@author yohanes.gultom@gmail.com
'''

import os
import glob
import numpy as np
import pandas as pd
import argparse
import joblib
import matplotlib.pyplot as plt
import featuretools as ft
import featuretools.variable_types as vtypes
from featuretools.variable_types import Numeric
from featuretools.primitives import make_agg_primitive
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
from scipy import interp

# custom primitives

def mean_diff(column):
    return column.diff().mean(skipna=True)

def max_diff(column):
    return column.diff().max(skipna=True)

def min_diff(column):
    return column.diff().min(skipna=True)

def std_diff(column):
    return column.diff().std(skipna=True)

def mean_diff_abs(column):
    return column.diff().abs().mean(skipna=True)

def max_diff_abs(column):
    return column.diff().abs().max(skipna=True)

def min_diff_abs(column):
    return column.diff().abs().min(skipna=True)

def std_diff_abs(column):
    return column.diff().abs().std(skipna=True)

def get_feature_matrix(df, n_jobs=1, verbose=True):
    es = ft.EntitySet('safety_data')
    es.entity_from_dataframe(
        entity_id='records',
        index='id',
        make_index=True,
        dataframe=df,        
        variable_types={
            'Accuracy': vtypes.Numeric,
            'Bearing': vtypes.Numeric,
            'acceleration_x': vtypes.Numeric,
            'acceleration_y': vtypes.Numeric,
            'acceleration_z': vtypes.Numeric,
            'gyro_x': vtypes.Numeric,
            'gyro_y': vtypes.Numeric,
            'gyro_z': vtypes.Numeric,
            'second': vtypes.Numeric,
            'Speed': vtypes.Numeric,
        }
    )
        
    es.normalize_entity(
        base_entity_id='records',
        new_entity_id='bookings',
        index='bookingID'    
    )
        
    return ft.dfs(
        entityset=es,
        target_entity='bookings',
        agg_primitives=[
            make_agg_primitive(function=mean_diff, input_types=[Numeric], return_type=Numeric),
            make_agg_primitive(function=max_diff, input_types=[Numeric], return_type=Numeric),
            make_agg_primitive(function=min_diff, input_types=[Numeric], return_type=Numeric),
            make_agg_primitive(function=std_diff, input_types=[Numeric], return_type=Numeric),
            make_agg_primitive(function=mean_diff_abs, input_types=[Numeric], return_type=Numeric),
            make_agg_primitive(function=max_diff_abs, input_types=[Numeric], return_type=Numeric),
            make_agg_primitive(function=min_diff_abs, input_types=[Numeric], return_type=Numeric),
            make_agg_primitive(function=std_diff_abs, input_types=[Numeric], return_type=Numeric),
            'count', 
            'mean', 
            'max', 
            'min', 
            'std',
        ],
        n_jobs=n_jobs,
        verbose=verbose
    )

def preprocess(features_df, labels_df, n_jobs=1, verbose=True, scale=False):
    feature_matrix, feature_names = get_feature_matrix(features_df, n_jobs, verbose)
    merged_df = feature_matrix.merge(labels_df, on='bookingID')
    feature_labels = merged_df['label']
    scaler = None
    X = feature_matrix.values    
    if scale:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    y = feature_labels.values
    label_ids = merged_df.bookingID.values[:, None]
    return X, y, label_ids

def build_dataset(dataset_csv, labels_csv, features_csv_dir, n_jobs, verbose, rebuild=False):
    # check if dataset not yet exists
    if not os.path.isfile(dataset_csv) or rebuild:
        # load labels
        print('Loading labels..')
        labels_df = pd.read_csv(labels_csv)
        # remove duplicate labels
        labels_df = labels_df[~labels_df.duplicated(['bookingID'])]
        
        # load features
        print('Loading features..')
        features_df = pd.concat(map(pd.read_csv, glob.glob(os.path.join('', features_csv_dir + '*.csv'))))

        # preprocess
        print('Preprocessing data..')
        X, y, label_ids = preprocess(features_df, labels_df, n_jobs=n_jobs, verbose=verbose)
        # save features + labels in a csv file
        np.savetxt(dataset_csv, np.concatenate((label_ids, X, y[:, None]), axis=1), delimiter=',')
        print('Preprocessed data saved in {}'.format(dataset_csv))

    else:
        # load dataset
        print('Loading dataset {}..'.format(dataset_csv))
        dataset = pd.read_csv(dataset_csv, header=None).values
        X = dataset[:, 1:-1]
        y = dataset[:, -1]
        label_ids = dataset[:, 0][:, None]

    return X, y, label_ids

def build_classifier(random_state=0, n_jobs=1, verbosity=0):
    return XGBClassifier(
        objective='binary:logistic', 
        eval_metric='auc',
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=verbosity
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sensor-based driving safety binary classification')
    parser.add_argument('action', choices=['preprocess', 'train', 'test'])
    parser.add_argument('-l', '--labels_csv', help='Path to labels CSV')
    parser.add_argument('-f', '--features_csv_dir', help='Path to features CSV directory')
    parser.add_argument('-r', '--random_state', default=0, help='Path to labels CSV')
    parser.add_argument('-d', '--dataset_csv', default='dataset.csv', help='Preprocessed dataset input/output path')
    parser.add_argument('-p', '--prediction_csv', default='prediction.csv', help='Prediction result path')
    parser.add_argument('-m', '--model', default='model.joblib', help='Model input/output path')
    parser.add_argument('-c', '--chart_png', default='roc_auc.png', help='ROC AUC chart image output')
    parser.add_argument('-n', '--n_jobs', type=int, default=1, help='Number of cores for parallel processing')
    parser.add_argument('-v', '--verbose', action='store_true', help='Improve verbosity')
    args = parser.parse_args()

    print('{} using {} core(s)..'.format(args.action, args.n_jobs))

    if args.action == 'preprocess':
        # build dataset and replace if exists
        build_dataset(
            args.dataset_csv, 
            args.labels_csv, 
            args.features_csv_dir, 
            args.n_jobs, 
            args.verbose,
            rebuild=True
        )

    elif args.action == 'train':
        # build training dataset or reuse if exists
        X, y, label_ids = build_dataset(
            args.dataset_csv, 
            args.labels_csv, 
            args.features_csv_dir, 
            args.n_jobs, 
            args.verbose
        )

        # train model and replace if exists
        print('Training model..')
        clf = build_classifier(args.random_state, args.n_jobs)
        clf.fit(X, y)
        joblib.dump(clf, args.model)
        print('Model saved in {}'.format(args.model))

    elif args.action == 'test':
        # build test dataset or reuse if exists
        X, y, label_ids = build_dataset(
            args.dataset_csv, 
            args.labels_csv, 
            args.features_csv_dir, 
            args.n_jobs, 
            args.verbose
        )

        print('Loading model {}..'.format(args.model))
        clf = joblib.load(args.model)   

        print('Predicting..')
        probas_ = clf.predict_proba(X)
        probas_pos = probas_[:, 1]
        df_pred = pd.DataFrame(np.concatenate((label_ids, np.rint(probas_pos)[:, None], probas_pos[:, None]), axis=1), columns=['bookingID', 'label', 'proba'])
        df_pred.to_csv(args.prediction_csv, index=False)
        print('Prediction result saved in {}..'.format(args.prediction_csv))

        print('Evaluating results..')
        fpr, tpr, thresholds = roc_curve(y, probas_[:, 1])
        roc_auc = auc(fpr, tpr)
        print('ROC AUC: {:0.2f}'.format(roc_auc))

        print('Plotting results..')
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
        plt.plot(fpr, tpr, lw=2, alpha=.8, color='b', label='ROC (AUC = {:0.2f})'.format(roc_auc))
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(args.chart_png)
        print("ROC AUC chart saved in {}".format(args.chart_png))
        plt.show()        

    else:
        print('Huh?')
