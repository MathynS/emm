import pandas as pd

from EMM import EMM
from evaluation_metrics import Heatmap


if __name__ == '__main__':
    df = pd.read_csv('data/german-credit-scoring.csv', sep=";")
    # df['Credit amount'], _ = pd.cut(df['Credit amount'], bins=10, retbins=True)
    # df['Credit amount'] = df['Credit amount'].apply(lambda x: f"{int(x.left)} - {int(x.right)}")
    df['Credit amount'] = df['Credit amount'].astype(int)
    df['Duration in months'] = df['Duration in months'].astype(int)
    clf = EMM(width=50, depth=2, evaluation_metric=Heatmap(), n_jobs=-1)
    clf.search(df, target_cols=['Score', 'Savings account/bonds'])
    clf.visualize(cols=2, subgroups=3, include_dataset=True)

