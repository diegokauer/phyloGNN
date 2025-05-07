import numpy as np
from composition_stats import ilr

from phylo_gnn.data_factory.transformations.common import apply_pseudo_count, apply_relative_abundance


def apply_centered_log_ratio(dataframe, columns, use_pseudo_counts=True, c=1, **kwargs):
    """
    Applies the centered log-ratio transformation to count data. Given by
    $CLR(i)=\log(x_{i})-\frac{1}{d}\sum_{j}\log(x_{j})$
    :param dataframe: DataFrame to apply the transformation
    :param columns: Columns of the columns that represent the counts
    :param use_pseudo_counts: Boolean if applying the pseudo-counts before the relative abundance calculation
    :param c: Pseudo count constant
    :param kwargs: Keyword arguments of the function
    :return: DataFrame with the applied transformation
    """
    dataframe = dataframe.copy()
    dataframe = apply_relative_abundance(dataframe, columns, use_pseudo_counts, c, **kwargs)
    dataframe[columns] = np.log(dataframe[columns]) - np.log(dataframe[columns].to_numpy()).mean(axis=1, keepdims=True)
    return dataframe


def apply_robust_centered_log_ratio(dataframe, columns, use_pseudo_counts=True, c=1, **kwargs):
    """
    Applies the robust centered log-ratio where the mean is calculated using only the sample's observed taxa.
    :param dataframe: DataFrame with count data to apply the transformation
    :param columns: Columns of the columns that represent the counts
    :param use_pseudo_counts: Boolean if applying the pseudo-counts before the relative abundance calculation
    :param c: Pseudo count constant
    :param kwargs: Keyword arguments of the function
    :return: DataFrame with the applied transformation
    """
    dataframe = dataframe.copy()
    ra_dataframe = apply_relative_abundance(dataframe, columns, use_pseudo_counts=False)
    sample_mean = np.ma.masked_equal(np.log(ra_dataframe[columns]), -np.inf).mean(axis=1, keepdims=True)
    dataframe = apply_relative_abundance(dataframe, columns, use_pseudo_counts, c, **kwargs)
    dataframe[columns] = np.log(dataframe[columns]) - sample_mean
    return dataframe


def apply_additive_log_ratio(dataframe, columns, reference_columns, use_pseudo_counts=True, c=1, **kwargs):
    """
    Applies the additive log-ratio transformation. It is given by:
    $ALR(i|d)=\log\left( \frac{x_{i}}{x_{d}} \right)~~i=1,\dots,d-1$
    :param dataframe: DataFrame with count data to apply the transformation
    :param columns: Columns of the columns that represent the counts
    :param reference_columns: Columns used as reference "d" in the log ratio calculation
    :param use_pseudo_counts: Boolean if applying the pseudo-counts before the relative abundance calculation
    :param c: Pseudo count constant
    :param kwargs: Keyword arguments of the function
    :return: DataFrame with the applied transformation
    """
    dataframe = dataframe.copy()
    dataframe = apply_relative_abundance(dataframe, columns, use_pseudo_counts, c, **kwargs)
    reference_sum = dataframe[reference_columns].sum(axis=1)
    dataframe[columns] = np.log(dataframe[columns].div(reference_sum, axis=0))
    dataframe = dataframe.drop(columns=reference_columns)
    return dataframe


def apply_isometric_log_ratio(dataframe, columns, use_pseudo_counts=True, c=1, **kwargs):
    """
    Applies isometric log-ratio composition. It wraps composition_stats.ilr
    https://composition-stats.readthedocs.io/en/latest/composition_stats.ilr.html#composition_stats.ilr
    :param dataframe: DataFrame with count data to apply the transformation
    :param columns: Columns of the columns that represent the counts
    :param use_pseudo_counts: Boolean if applying the pseudo-counts before the relative abundance calculation
    :param c: Pseudo count constant
    :param kwargs: Keyword arguments of the function
    :return: DataFrame with the applied transformation
    """
    dataframe = dataframe.copy()
    dataframe = apply_relative_abundance(dataframe, columns, use_pseudo_counts, c, **kwargs)
    dataframe[columns[:-1]] = ilr(dataframe[columns].to_numpy())
    dataframe = dataframe.drop(columns=columns[-1])
    return dataframe