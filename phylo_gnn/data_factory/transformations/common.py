import numpy as np


def apply_pseudo_count(dataframe, columns, c=1, **kwargs):
    """
    Function that applies a pseudo-count (x : x + c) to selected dataframe columns, where c is the constant
    It can be useful when applying logarithm to avoid log(0)
    :param dataframe: DataFrame with count data to apply the transformation
    :param columns: Columns of the columns that represent the counts
    :param c: constant to be added (Default value: 1)
    :param kwargs: Keyword arguments of the function
    :return: DataFrame with the applied transformation
    """
    dataframe = dataframe.copy()
    dataframe[columns] = dataframe[columns] + c
    return dataframe


def apply_presence_absence(dataframe, columns, threshold=0, **kwargs):
    """
    Function that applies the Presence-Absence (PA) transformation on the selected columns. When a count is above (>)
    the threshold, it is replaced by 1 and if its below, it is 0.
    :param dataframe: DataFrame with count data to apply the transformation
    :param columns: Columns of the columns that represent the counts
    :param threshold: constant to be added (Default value: 0)
    :param kwargs: Keyword arguments of the function
    :return: DataFrame with the applied transformation
    """
    dataframe = dataframe.copy()
    dataframe[columns] = (dataframe[columns] > threshold).astype(int)
    return dataframe


def apply_relative_abundance(dataframe, columns, use_pseudo_counts=True, c=1, **kwargs):
    """
    Function that calculates the relative abundance of count data in a DataFrame, it uses the selected columns and
    returns the transformation.
    :param dataframe: DataFrame to apply the transformation
    :param columns: Columns of the columns that represent the counts
    :param use_pseudo_counts: Boolean if applying the pseudo-counts before the relative abundance calculation
    :param c: Pseudo count constant
    :param kwargs: Keyword arguments of the function
    :return: DataFrame with the applied transformation
    """
    dataframe = dataframe.copy()
    if use_pseudo_counts:
        dataframe = apply_pseudo_count(dataframe, columns, c, **kwargs)
    dataframe[columns] = dataframe[columns].div(dataframe[columns].sum(axis=1), axis=0)
    return dataframe


def apply_arcsine_square_root(dataframe, columns, use_pseudo_counts=True, c=1, **kwargs):
    """
    Applies the arcSine square root transformation. It assumes an input of count data. Applies the relative abundance
    first and then applies the transformation of arcSine(sqrt(x)).
    :param dataframe: DataFrame with count data to apply the transformation
    :param columns: Columns of the columns that represent the counts
    :param use_pseudo_counts: Boolean if applying the pseudo-counts before the relative abundance calculation
    :param c: Pseudo count constant
    :param kwargs: Keyword arguments of the function
    :return: DataFrame with the applied transformation
    """
    dataframe = dataframe.copy()
    dataframe = apply_relative_abundance(dataframe, columns, use_pseudo_counts, c, **kwargs)
    dataframe[columns] = np.arcsin(np.sqrt(dataframe[columns]))
    return dataframe