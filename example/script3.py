"""Debugging script.

Used to debug covariance quality measure
"""

import pandas as pd
from EMM import EMM
from evaluation_metrics import DistributionCosine, Correlation, Covariance

if __name__ == '__main__':
    df = pd.read_csv('data/Housing.csv')
    df = df.drop('Unnamed: 0', axis=1)  # Drop index col
    target_columns = ['lotsize', 'price']
    # clf = EMM(width=10, depth=1, evaluation_metric='covariance',
    #           n_jobs=1, log_level=2)
    clf = EMM(width=20, depth=2, evaluation_metric=Covariance(),
              log_level=2)
    clf.search(df, target_cols=target_columns)
    # clf = EMM(width=20, depth=4, evaluation_metric='correlation',
    #           log_level=2)
    # clf.search(df, target_cols=target_columns)
    clf.visualize(cols=2, subgroups=5, include_dataset=True)
