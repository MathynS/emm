import math
import warnings
import pandas as pd
import statsmodels.api as sm
import numpy as np

from copy import deepcopy
from scipy.spatial.distance import cosine


warnings.filterwarnings('error')
items = None
cache = None
EPSILON = 1e-10


def cleanup():
    global items, cache
    items = None
    cache = None


def entropy(subgroup_target, dataset_target):
    """Calculates the entropy between a subgroup and the full dataset.

    Implementation of \varphi_{ef} in chapter 3 of "Exceptional Model Mining",
    W. Duivesteijn (2016).

    Args:
        subgroup_target (pd.DataFrame): The subgroup currently being evaluated.
        dataset_target (pd.DataFrame): The full dataset for which the subgroup
            is to be evaluated against.

    Returns:
        float: Score as evaluated according to the entropy formula.
    """
    n_c = max(1, len(dataset_target) - len(subgroup_target))
    n = len(subgroup_target)
    N = len(dataset_target)
    return -n/N * math.log(n/N) - n_c/N * math.log(n_c/N)


def distribution_cosine(subgroup_target, dataset_target, use_complement=False):
    """Distributed cosine quality measure.

    Args:
        subgroup_target (pd.DataFrame): The subgroup currently being evaluated.
        dataset_target (pd.DataFrame): The full dataset for which the subgroup
            is to be evaluated against.
        use_complement (bool): Whether or not to evaluate against the complement
            of the subgroup. Use of the complement is currently unsupported.

    Returns:
        (float, float): Normalized entropy score and the target score.
    """
    global items, cache
    if len(subgroup_target.columns) > 1:
        raise ValueError("Distribution cosine expect exactly 1 column as "
                         "target variable")
    column = list(subgroup_target.columns)[0]
    if cache is None:
        cache = dataset_target[column].value_counts()
        items = pd.Series([0] * len(cache.index), index=cache.index)
    values = subgroup_target[column].value_counts()
    target = deepcopy(items)
    target[values.index] = values.values
    return (entropy(subgroup_target, dataset_target)
            * cosine(target.values, cache.values),
            target)


def WRAcc(subgroup_target, dataset_target, use_complement=False):
    """Implementation of Weighted Relative Accuracy.

    This is an implementation of the canonical subgroup discovery quality
    measure presented in "ROC ‘n’ Rule Learning – Towards a Better Understanding
    of Covering Algorithms", J. Fürnkranz, P. A. Flach (2005).

    Args:
        subgroup_target (pd.DataFrame): The subgroup currently being evaluated.
        dataset_target (pd.DataFrame): The full dataset for which the subgroup
            is to be evaluated against.
        use_complement (bool): Whether or not to evaluate against the complement
            of the subgroup. Use of the complement is currently unsupported.

    Returns:
        (float, float): WRAcc score and the target score.
    """
    global items, cache
    if len(subgroup_target.columns) > 1:
        raise ValueError("Distribution cosine expect exactly 1 column as "
                         "target variable")
    column = list(subgroup_target.columns)[0]
    if cache is None:
        cache = dataset_target[column].value_counts()
        items = pd.Series([0] * len(cache.index), index=cache.index)
    values = subgroup_target[column].value_counts()
    target = deepcopy(items)
    target[values.index] = values.values
    max_Wc = target.values.max() + EPSILON
    max_W = cache.values.max() + EPSILON
    score = 0
    for Wce, We in zip(target.values, cache.values):
        score += (max_Wc / max_W) * ((Wce / max_Wc) - (We / max_W))
    return score * 1000, target


def avg(collection):
    try:
        return sum(collection) / len(collection)
    except ZeroDivisionError:
        return 0


def r_hat(df, col_x, col_y):
    """Implementation of r_hat for varphi_{scd}.

    Args:
        df (pd.DataFrame): Complete dataset DataFrame to be evaluated.
        col_x (str): Name of the column representing the "x" variable.
        col_y (str): Name of the column representing the "y" variable.

    Returns:
        float: The calculated r_hat value. If both x.sum() and y.sum() are equal
            to 0, returns 0.
    """
    avg_x = avg(df[col_x])
    avg_y = avg(df[col_y])
    top = df.apply(lambda row: (row[col_x] - avg_x) * (row[col_y] - avg_y),
                   axis=1)
    bottom_x = df.apply(lambda row: (row[col_x] - avg_x) ** 2, axis=1)
    bottom_y = df.apply(lambda row: (row[col_y] - avg_y) ** 2, axis=1)
    try:
        return top.sum() / math.sqrt(bottom_x.sum() * bottom_y.sum())
    except Warning:  # Both x.sum() and y.sum() equal zero
        return 0


def heatmap(subgroup_target, dataset_target, use_complement=False):
    """Calculates quality measure using a heatmap metric.

    The actual explanation of this quality measure is unknown and doesn't seem
    to be explained in "Exceptional Model Mining", W. Duivesteijn (2016).

    Args:
        subgroup_target (pd.DataFrame): The subgroup currently being evaluated.
        dataset_target (pd.DataFrame): The full dataset for which the subgroup
            is to be evaluated against.
        use_complement (bool): Whether or not to evaluate against the complement
            of the subgroup. Use of the complement is currently unsupported.

    Returns:
        (float, float): Normalized entropy score and the r_hat score.
    """
    global cache, items
    if len(subgroup_target.columns) != 2:
        raise ValueError("heatmap metric expects exactly 2 columns as "
                         "target variables")
    x_col, y_col = list(subgroup_target.columns)

    if cache is None:
        cache = pd.pivot_table(dataset_target, values=x_col, index=x_col,
                               fill_value=0, columns=y_col,
                               aggfunc=lambda x: len(x)).stack()
        items = pd.Series([0] * len(cache.index), index=cache.index)
    pv = pd.pivot_table(subgroup_target, values=x_col, index=x_col,
                        fill_value=0, columns=y_col,
                        aggfunc=lambda x: len(x)).stack()
    target = deepcopy(items)
    target[pv.index] = pv.values
    return (entropy(subgroup_target, dataset_target)
            * cosine(target.values, cache.values),
            target.unstack())


def correlation(subgroup_target, dataset_target, use_complement=False):
    """Calculates quality measure using Significance of Correlation Difference.

    Complete formulation is described in Chapter 4 of "Exceptional Model
    Mining", W. Duivesteijn (2016).

    Args:
        subgroup_target (pd.DataFrame): The subgroup currently being evaluated.
        dataset_target (pd.DataFrame): The full dataset for which the subgroup
            is to be evaluated against.
        use_complement (bool): Whether or not to evaluate against the complement
            of the subgroup. Use of the complement is currently unsupported.

    Returns:
        (float, float): Normalized entropy score and the r_hat score.
    """
    global cache
    if len(subgroup_target.columns) != 2:
        raise ValueError("Correlation metric expects exactly 2 columns as "
                         "target variables")
    x_col, y_col = list(subgroup_target.columns)
    if cache is None:
        cache = r_hat(dataset_target, x_col, y_col)
    # print(subgroup_target, x_col, y_col)
    r_gd = r_hat(subgroup_target, x_col, y_col)
    if math.isnan(r_gd):
        return 0, 0
    return entropy(subgroup_target, dataset_target) * abs(r_gd - cache), r_gd


def regression(subgroup_target, dataset_target, use_complement=False):
    global cache
    if len(subgroup_target) < 20:
        return 0, None
    if len(subgroup_target.columns) != 2:
        raise ValueError("Correlation metric expects exactly 2 columns as "
                         "target variables")
    x_col, y_col = list(subgroup_target.columns)
    if cache is None:
        est2 = sm.OLS(dataset_target[y_col], dataset_target[x_col])
        est2 = est2.fit()
        cache = est2.summary2().tables[1]['Coef.'][x_col]
    est = sm.OLS(subgroup_target[y_col], subgroup_target[x_col])
    est = est.fit()
    coef = est.summary2().tables[1]['Coef.'][x_col]
    p = est.summary2().tables[1]['P>|t|'][x_col]
    if math.isnan(p):
        return 0, 0
    if (1 - p) < 0.99:
        return 0, 0
    return entropy(subgroup_target, dataset_target) * abs(coef - cache), coef


def covariance(subgroup_target, dataset_target, use_complement=False):
    """Quality measure based on covariance matrix distance.

    Calculates the Frobenius norm between the column covariances of the target
    variables in both the subgroup and the dataset.

    The score is the normalized Frobenius distance between the difference of
    the column-wise covariances in the subgroup compared to the dataset. The
    target is the normalized Frobenius distance of the column-wise
    covariances in the subgroup from the origin. All values are normalized by
    the Frobenius distance of the column-wise covariances of the complete
    dataset from the origin.

    Args:
        subgroup_target (pd.DataFrame): The subgroup currently being evaluated
        dataset_target (pd.DataFrame): The full dataset for which the subgroup
            is to be evaluated against
        use_complement (bool): Whether or not to evaluate against the complement
            of the subgroup. Use of the complement is currently unsupported.

    Returns:
        (float, float): The score and the target.
    """
    global cache

    if cache is None:
        pass

    subgroup_cov = subgroup_target.cov()
    dataset_cov = dataset_target.cov()

    cov_diff = subgroup_cov - dataset_cov

    normalizer = np.linalg.norm(dataset_cov)

    if normalizer == 0:
        return 0., 0.

    score = abs(np.linalg.norm(cov_diff) / normalizer)
    target = abs(np.linalg.norm(subgroup_cov) / normalizer)

    return score, target


metrics = {'correlation': correlation,
           'distribution_cosine': distribution_cosine,
           'regression': regression,
           'WRAcc': WRAcc,
           'heatmap': heatmap,
           'covariance': covariance}
