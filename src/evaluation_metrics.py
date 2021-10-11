import math
import warnings
import pandas as pd
import statsmodels.api as sm
import numpy as np
from abc import ABC, abstractmethod

from copy import deepcopy
from scipy.spatial.distance import cosine

warnings.filterwarnings('error')
items = None
cache = None


def cleanup():
    global items, cache
    items = None
    cache = None


class EvaluationMetric(ABC):
    def __init__(self, use_complement):
        """Generic abstract class for evaluation metrics.

        Args:
            use_complement (bool): Whether or not to calculate against the
                complement.
        """
        self.use_complement = use_complement
        self.epsilon = 1e-10
        super().__init__()

    @property
    @abstractmethod
    def name(self):
        """Returns the name of the metric."""
        pass

    @staticmethod
    def entropy(subgroup_target, dataset_target):
        """Calculates the entropy between a subgroup and the full dataset.

        Implementation of \varphi_{ef} in chapter 3 of "Exceptional Model
        Mining", W. Duivesteijn (2016).

        Args:
            subgroup_target (pd.DataFrame): The subgroup currently being
                evaluated.
            dataset_target (pd.DataFrame): The full dataset for which the
                subgroup is to be evaluated against.

        Returns:
            float: Score as evaluated according to the entropy formula.
        """
        n_c = max(1, len(dataset_target) - len(subgroup_target))
        n = len(subgroup_target)
        N = len(dataset_target)
        return -n / N * math.log(n / N) - n_c / N * math.log(n_c / N)

    @abstractmethod
    def calculate(self, subgroup_target, dataset_target):
        pass


class DistributionCosine(EvaluationMetric):
    def __init__(self, use_complement=False):
        """Distributed cosine quality measure.

        Args:
            use_complement (bool): Whether or not to evaluate against the
                complement of the subgroup. Use of the complement is currently
                unsupported.
        """
        super().__init__(use_complement)

    def name(self):
        return "DistributionCosine"

    @staticmethod
    def _dist_cosine_calc(subgroup_target, dataset_target):
        """Calculation shared by dist cosine and WRAcc.

        Args:
            subgroup_target (pd.DataFrame): The subgroup currently being
                evaluated.
            dataset_target (pd.DataFrame): The full dataset for which the
                subgroup is to be evaluated against.

        Returns:
            tuple: target_values, cache_values, target)
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

        return target.values, cache.values, target

    def calculate(self, subgroup_target, dataset_target):
        """Calculates a quality measure using the distribution cosine

        Args:
            subgroup_target (pd.DataFrame): The subgroup currently being
                evaluated.
            dataset_target (pd.DataFrame): The full dataset for which the
                subgroup is to be evaluated against.

        Returns:
            (float, float): Normalized entropy score and the target score.
        """
        target_values, cache_values, target = self._dist_cosine_calc(
            subgroup_target, dataset_target
        )
        return (EvaluationMetric.entropy(subgroup_target, dataset_target)
                * cosine(target_values, cache_values),
                target)


class WRAcc(DistributionCosine):
    def __init__(self, use_complement=False):
        """Implementation of Weighted Relative Accuracy.

        This is an implementation of the canonical subgroup discovery quality
        measure presented in "ROC ‘n’ Rule Learning – Towards a Better
        Understanding of Covering Algorithms", J. Fürnkranz, P. A. Flach (2005).

        Args:
            use_complement (bool): Whether or not to evaluate against the
            complement of the subgroup. Use of the complement is currently
            unsupported.
        """
        super().__init__(use_complement)

    def name(self):
        return "WRAcc"

    def calculate(self, subgroup_target, dataset_target):
        """Calculates a quality measure using Weighted Relative Accuracy.

        Args:
            subgroup_target (pd.DataFrame): The subgroup currently being
                evaluated.
            dataset_target (pd.DataFrame): The full dataset for which the
                subgroup is to be evaluated against.

        Returns:
            (float, float): WRAcc score and the target score.
        """
        target_values, cache_values, target = self._dist_cosine_calc(
            subgroup_target, dataset_target
        )
        max_wc = target_values.max() + self.epsilon
        max_w = cache_values.max() + self.epsilon
        score = 0
        for Wce, We in zip(target_values, cache_values):
            score += (max_wc / max_w) * ((Wce / max_wc) - (We / max_w))
        return score * 1000, target


class Heatmap(EvaluationMetric):
    def __init__(self, use_complement=False):
        """Implementation of the heatmap metric.

        The actual explanation of this quality measure is unknown and doesn't
        seem to be explained in "Exceptional Model Mining", W. Duivesteijn
        (2016).

        Args:
            use_complement (bool): Whether or not to evaluate against the
                complement of the subgroup. Use of the complement is currently
                unsupported.
        """

        super().__init__(use_complement)

    def name(self):
        return "Heatmap"

    def calculate(self, subgroup_target, dataset_target):
        """Calculates a quality measure using a heatmap metric.

        Args:
            subgroup_target (pd.DataFrame): The subgroup currently being
                evaluated.
            dataset_target (pd.DataFrame): The full dataset for which the
                subgroup is to be evaluated against.

        Returns:
            (float, float): Normalized entropy score and the unstacked score.
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
        return (self.entropy(subgroup_target, dataset_target)
                * cosine(target.values, cache.values),
                target.unstack())


class Correlation(EvaluationMetric):
    def __init__(self, use_complement=False):
        """Implements quality measure Significiance of Correlation Difference.

        Complete formulation is described in Chapter 4 of "Exceptional Model
        Mining", W. Duivesteijn (2016).

        Args:
            use_complement (bool): Whether or not to evaluate against the
                complement of the subgroup. Use of the complement is currently
                unsupported.
        """
        super().__init__(use_complement)

    def name(self):
        return "Correlation"

    @staticmethod
    def r_hat(df, col_x, col_y):
        """Implementation of r_hat for varphi_{scd}.

        Args:
            df (pd.DataFrame): Complete dataset DataFrame to be evaluated.
            col_x (str): Name of the column representing the "x" variable.
            col_y (str): Name of the column representing the "y" variable.

        Returns:
            float: The calculated r_hat value. If both x.sum() and y.sum()
            are equal
                to 0, returns 0.
        """

        def avg(series):
            """Averages a series.

            Args:
                series (pd.core.series.Series): A series for which the mean
                is to
                    be calculated.

            Returns:
                float: The mean of the series. Returns 0 if the length of the
                series
                    is 0.
            """
            if len(series) == 0:
                return 0
            return series.mean()

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

    def calculate(self, subgroup_target, dataset_target):
        """Calculates quality measure Significance of Correlation Difference.

        Args:
            subgroup_target (pd.DataFrame): The subgroup currently being
            evaluated.
            dataset_target (pd.DataFrame): The full dataset for which the
                subgroup is to be evaluated against.

        Returns:
            (float, float): Normalized entropy score and the r_hat score.
        """
        global cache
        if len(subgroup_target.columns) != 2:
            raise ValueError("Correlation metric expects exactly 2 columns as "
                             "target variables")
        x_col, y_col = list(subgroup_target.columns)
        if cache is None:
            cache = self.r_hat(dataset_target, x_col, y_col)
        # print(subgroup_target, x_col, y_col)
        r_gd = self.r_hat(subgroup_target, x_col, y_col)
        if math.isnan(r_gd):
            return 0, 0
        return (self.entropy(subgroup_target, dataset_target)
                * abs(r_gd - cache),
                r_gd)


class Regression(EvaluationMetric):
    def __init__(self, use_complement=False):
        """Implements quality measure using regression.

        Args:
            use_complement (bool): Whether or not to evaluate against the
                complement of the subgroup. Use of the complement is currently
                unsupported.
        """
        super().__init__(use_complement)

    def name(self):
        return "Regression"

    def calculate(self, subgroup_target, dataset_target):
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
        return (self.entropy(subgroup_target, dataset_target)
                * abs(coef - cache),
                coef)


class Covariance(EvaluationMetric):
    def __init__(self, use_complement=False):
        """Quality measure based on covariance matrix distance.

        Calculates the Frobenius norm between the column covariances of the
        target variables in both the subgroup and the dataset.

        The score is the normalized Frobenius distance between the difference
        of the column-wise covariances in the subgroup compared to the
        dataset. The target is the normalized Frobenius distance of the
        column-wise covariances in the subgroup from the origin. All values
        are normalized by the Frobenius distance of the column-wise
        covariances of the complete dataset from the origin.

        Args:
            use_complement (bool): Whether or not to evaluate against the
                complement of the subgroup. Use of the complement is currently
                unsupported.
        """
        super().__init__(use_complement)

    def name(self):
        return "Covariance"

    def calculate(self, subgroup_target, dataset_target):
        """Calculates the quality measure based on the covariance matrix.

        Args:
            subgroup_target (pd.DataFrame): The subgroup currently being
                evaluated.
            dataset_target (pd.DataFrame): The full dataset for which the
                subgroup is to be evaluated against

        Returns:
            (float, float): The score and the target.
        """
        global cache

        if cache is None:
            pass

        if subgroup_target.shape[0] <= 1:
            # Catches edge cases where the subgroup consists of only one element
            # and thus has no variance.
            return 0., 0.

        subgroup_cov = subgroup_target.cov()
        dataset_cov = dataset_target.cov()

        cov_diff = subgroup_cov - dataset_cov

        normalizer = np.linalg.norm(dataset_cov)

        if normalizer == 0:
            return 0., 0.

        # Multiply by a polynomial such that bigger subgroups are preferred
        # Constants chosen by empirical experiments
        subgroup_size = float(subgroup_target.shape[0])
        subgroup_size /= float(dataset_target.shape[0])
        mult = (-(.98 * subgroup_size - .98) ** 2) + 1

        score = mult * (abs(np.linalg.norm(cov_diff) / normalizer))
        target = mult * (abs(np.linalg.norm(subgroup_cov) / normalizer))

        return score, target
