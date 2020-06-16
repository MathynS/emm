import math
import pandas as pd

from copy import deepcopy
from scipy.spatial.distance import cosine

items = None
cache = None


def distribution_cosine(subgroup_target, dataset_target, use_complement=False):
    global items, cache
    if len(subgroup_target.columns) > 1:
        raise ValueError("Distribution cosine expect exactly 1 column as target variable")
    column = list(subgroup_target.columns)[0]
    if cache is None:
        cache = dataset_target[column].value_counts()
        items = pd.Series([0] * len(cache.index), index=cache.index)
    values = subgroup_target[column].value_counts()
    target = deepcopy(items)
    target[values.index] = values.values
    return math.sqrt(len(subgroup_target)) * cosine(target.values, cache.values), target


def avg(collection):
    try:
        return sum(collection) / len(collection)
    except ZeroDivisionError:
        return 0


def r_hat(df, col_x, col_y):
    avg_x = avg(df[col_x])
    avg_y = avg(df[col_y])
    top = df.apply(lambda row: (row[col_x] - avg_x) * (row[col_y] - avg_y), axis=1)
    bottom_x = df.apply(lambda row: (row[col_x] - avg_x) ** 2, axis=1)
    bottom_y = df.apply(lambda row: (row[col_y] - avg_y) ** 2, axis=1)
    return top.sum() / math.sqrt(bottom_x.sum() * bottom_y.sum())


def entropy(subgroup_target, dataset_target):
    """

    Args:
        subgroup_target:
        dataset_target:

    Returns:

    """
    n_c = len(dataset_target) - len(subgroup_target)
    n = len(subgroup_target)
    N = len(dataset_target)
    return -n/N * math.log(n/N) - n_c/N * math.log(n_c/N)


def correlation(subgroup_target, dataset_target, use_complement=False):
    """

    :param subgroup_target:
    :param dataset_target:
    :param use_complement:
    :return:
    """
    global cache
    if len(subgroup_target.columns) != 2:
        raise ValueError("Correlation metric expects exactly 2 columns as target variables")
    x_col, y_col = list(subgroup_target.columns)
    if cache is None:
        cache = r_hat(dataset_target, x_col, y_col)
    r_gd = r_hat(subgroup_target, x_col, y_col)
    if math.isnan(r_gd):
        return 0, 0
    return entropy(subgroup_target, dataset_target) * abs(r_gd - cache), r_gd


metrics = dict(
    correlation=correlation,
    distribution_cosine=distribution_cosine
)
