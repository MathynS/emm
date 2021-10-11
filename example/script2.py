import pandas as pd
from EMM import EMM
from evaluation_metrics import Heatmap

if __name__ == '__main__':
    # df = pd.read_csv('data/Housing.csv', index_col=0)
    # target_columns = ['lotsize', 'price']
    df = pd.read_csv('data/german-credit-scoring.csv', sep=";")
    target_columns = ['Score', 'Savings account/bonds']
    clf = EMM(width=10, depth=1, evaluation_metric=Heatmap(), log_level=2,
              n_jobs=1)
    clf.search(df, target_cols=target_columns)
    clf.visualize(cols=3, subgroups=8, include_dataset=True)
