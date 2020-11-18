
import pandas as pd
import numpy as np


def process(df):
    """Adds columns derived from the postprocessing

    Parameters
    ----------
    df : [type]
        The input pandas dataframe

    Returns
    -------
    [type]
        A pandas dataframe containing the added columns
    """
    df[['indirect', 'vague', 'past', 'keyword', 'span_id']
       ] = df.apply(lambda row: parse_comments(row), axis=1)
    df['cro_sub_type_combined'] = df.apply(
        lambda row: map_cro_sub_type(row), axis=1)
    return df


def parse_comments(row):
    """Parses unstructured labelling comments based on agreed ad-hoc "tags" such as "indirect", "vague" etc

    Parameters
    ----------
    row : [type]
        Labelling dataframe row

    Returns
    -------
    [type]
        Pandas series of multiple columns
    """
    tags = row.comment.split(',') if not pd.isna(row.comment) else []
    indirect = False
    vague = False
    past = False

    if 'indirect' in tags:
        indirect = True
    if 'vague' in tags:
        vague = True
    if 'past' in tags:
        past = True

    span_id = next((tag for tag in tags if tag.startswith('cro_id:')), np.NaN)
    if not pd.isna(span_id):
        span_id = span_id.replace('cro_id:', '')

    keyword = next((tag for tag in tags if tag.startswith('keyword:')), np.NaN)
    if not pd.isna(keyword):
        keyword = keyword.replace('keyword:', '')

    return pd.Series([indirect, vague, past, keyword, span_id])


def map_cro_sub_type(row):
    """Maps the TCFD CRO sub type as labelled to a higher level aggregation

    Parameters
    ----------
    row : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    if row.cro == 'OP' and row.cro_sub_type in ['RESILI', 'EFFI', 'ENERGY']:
        return 'RESILIENCE'
    if row.cro == 'OP' and row.cro_sub_type in ['MARKETS', 'PRODUCTS']:
        return 'PRODUCTS'
    if row.cro == 'TR' and row.cro_sub_type in ['POLICY']:
        return 'POLICY'
    if row.cro == 'TR' and row.cro_sub_type in ['MARKET', 'TECH']:
        return 'MARKET'
    if row.cro == 'TR' and row.cro_sub_type in ['REPUT']:
        return 'REPUTATION'
    if row.cro == 'PR':
        return row.cro_sub_type
