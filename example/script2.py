import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

from emm import EMM
from subgroup import Subgroup
from description import Description


if __name__ == '__main__':
    df = pd.read_csv('data/Housing.csv', index_col=0)
    target_columns = ['lotsize', 'price']
    clf = EMM(width=10, depth=2, evaluation_metric='correlation', n_jobs=1)
    clf.search(df, target_cols=target_columns)
    clf.visualise(subgroups=5, cols=3)