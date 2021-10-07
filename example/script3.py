"""Debugging script.

Used to debug covariance quality measure
"""

import pandas as pd
from EMM import EMM

if __name__ == '__main__':
    df = pd.read_csv('data/Housing.csv')
    target_columns = ['lotsize', 'price']
    # clf = EMM(width=10, depth=1, evaluation_metric='covariance',
    #           n_jobs=1, log_level=2)
    clf = EMM(width=20, depth=1, evaluation_metric='correlation',
              log_level=2, n_jobs=1)
    clf.search(df, target_cols=target_columns)
    # clf = EMM(width=20, depth=4, evaluation_metric='correlation',
    #           log_level=2)
    clf.search(df, target_cols=target_columns)
    clf.visualise(cols=2, subgroups=5, include_dataset=True)
