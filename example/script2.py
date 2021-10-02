import pandas as pd
from EMM import EMM

if __name__ == '__main__':
    # df = pd.read_csv('data/Housing.csv', index_col=0)
    # target_columns = ['lotsize', 'price']
    df = pd.read_csv('data/german-credit-scoring.csv', sep=";")
    target_columns = ['Duration in months', 'Credit amount']
    clf = EMM(width=20, depth=1, evaluation_metric='regression', n_jobs=1, log_level=2)
    clf.search(df, target_cols=target_columns)
    clf.visualise(cols=3, subgroups=8, include_dataset=True)
