import os

import pandas as pd
import numpy as np

from sklearn.utils import shuffle
from datasets import DatasetDict, Dataset, load_dataset, Sequence, ClassLabel, Features, Value, concatenate_datasets

from .constants import map_to_field, get_code_idx_map, cro_category_levels, cro_category_codes, cro_sub_category_codes, cro_categories, cro_sub_categories

valid_tasks = {"multi-label", "multi-class", "binary"}


def convert_to_binary_cls(df, cro_level):
    docs = df.groupby(["id"]).first().text
    labels = df.groupby(["id"])[cro_level].count()
    assert len(docs) == len(
        labels), "The dimensions of documents and labels do not match!"

    labels = (labels > 0) * 1
    return docs, labels


def convert_to_multi_cls(df, cro_level, exclude_op):
    """Converts a dataframe to the doc, labels multi-class format, e.g: an integer for each categorical value
    """
    # Add a "irrelevant" label for each "negatives"
    df[cro_level].loc[df[cro_level] != df[cro_level]] = "irrelevant"
    # TODO: Fix for cases where multiple labels are given for one paragraph
    docs = df.groupby(["id"]).first().text
    labels = df.groupby(["id"])[cro_level].first()

    # Convert to categorical (deterministic)
    mapping = get_code_idx_map(category_level="cro", filter_op=exclude_op)
    mapping['irrelevant'] = len(mapping)
    inv_mapping = {v: k for k, v in mapping.items()}
    labels = labels.replace(to_replace=mapping)

    sorted_labels = sorted(labels.unique())
    cat_labels = [inv_mapping[l] for l in sorted_labels]
    return docs, labels, cat_labels


def convert_to_multi_label_cls(df, cro_level, exclude_op):
    docs = df.groupby(["id"]).first().text

    # Hack: To actually include also missing values
    df[cro_level].loc[df[cro_level] != df[cro_level]] = "missing"
    labels = pd.crosstab(df.id, df[cro_level], dropna=False)

    if cro_level == 'cro':
        category_codes = [c["code"]
                          for c in cro_categories if not exclude_op or (exclude_op and c["code"] != "OP")]
    else:
        category_codes = [c["code"]
                          for c in cro_sub_categories if not exclude_op or (exclude_op and c["parent"] != "OP")]
    labels = labels[category_codes]

    assert np.shape(labels)[1] == len(
        category_codes), f"The dimension of the labels({np.shape(labels)[1]}) do not match with the requested categories({len(category_codes)})"

    assert len(docs) == len(
        labels), f"The dimensions of documents ({len(docs)}) and labels ({len(labels)}) do not match!"

    labels = (labels > 0) * 1
    return docs, labels


def prepare_datasets(
        data_dir="/Users/david/Nextcloud/Dokumente/Education/Uni Bern/Master Thesis/Analyzing Financial Climate Disclosures with NLP/Labelling/annual reports/",
        task="multi-label",
        cro_category_level="cro",
        should_filter_op=True,
        as_huggingface_ds=False,
        validation_split=0,
        train_neg_sampling_strategy=None,
        test_neg_sampling_strategy=None,
        seed_value=42,
):
    if not task in valid_tasks:
        raise TypeError(
            f'Not a valid task: "{task}"! Task must be one of {", ".join(valid_tasks)}')

    # Check the validity of the supplied category level
    if not cro_category_level in cro_category_levels:
        raise TypeError(
            f'Not a valid CRO Category level! cro_category_level must be one of: {", ".join(cro_category_levels)}')

    # Load files
    df_train = pd.read_pickle(os.path.join(
        data_dir, 'Firm_AnnualReport_Labels_Training.pkl'))
    df_test = pd.read_pickle(os.path.join(
        data_dir, 'Firm_AnnualReport_Labels_Test.pkl'))

    # Set id
    id_columns = ['report_id', 'page', 'paragraph_no']
    df_train["id"] = df_train.apply(lambda row: "_".join(
        [str(row[c]) for c in id_columns]), axis=1)
    df_test["id"] = df_test.apply(lambda row: "_".join(
        [str(row[c]) for c in id_columns]), axis=1)

    # If opportunities should get filtered out, then set the respecting columns as "negatives"
    if should_filter_op:
        df_train.cro_sub_type.loc[df_train.cro == "OP"] = np.nan
        df_test.cro_sub_type.loc[df_test.cro == "OP"] = np.nan
        df_train.cro_sub_type_combined.loc[df_train.cro == "OP"] = np.nan
        df_test.cro_sub_type_combined.loc[df_test.cro == "OP"] = np.nan
        df_train.neg_type.loc[df_train.cro == "OP"] = "opportunity"
        df_test.neg_type.loc[df_test.cro == "OP"] = "opportunity"
        df_train.cro.loc[df_train.cro == "OP"] = np.nan
        df_test.cro.loc[df_test.cro == "OP"] = np.nan

    # By default, filter all negatives and opportunities
    query_filter_test = "neg_type != 'opportunity' & neg_type != neg_type"
    query_filter_train = "neg_type != 'opportunity' & neg_type != neg_type"

    # Set train filter
    if train_neg_sampling_strategy == "only_OP":
        # Filter out all the negatives except for opportunities
        query_filter_train = "neg_type != neg_type | neg_type == 'opportunity'"

    elif train_neg_sampling_strategy == "all":
        query_filter_train = None
    # Filter..
    if query_filter_train:
        df_train.query(query_filter_train,
                       inplace=True)

    # Set test filter
    if test_neg_sampling_strategy == "only_OP":
        # Filter out all the negatives except for opportunities
        query_filter_test = "neg_type != neg_type | neg_type == 'opportunity'"

    elif train_neg_sampling_strategy == "all":
        query_filter_test = None

    # Filter...
    if query_filter_test:
        df_test.query(query_filter_test,
                      inplace=True)

    if task == 'multi-label':
        train_docs, train_doc_labels = convert_to_multi_label_cls(
            df_train, cro_level=cro_category_level, exclude_op=should_filter_op)
        test_docs, test_doc_labels = convert_to_multi_label_cls(
            df_test, cro_level=cro_category_level, exclude_op=should_filter_op)
        features = Features({'text': Value('string'), 'labels': Sequence(ClassLabel(names=[
                            map_to_field(field='label')[c] for c, content in train_doc_labels.items()], num_classes=len(train_doc_labels.columns)))})

    elif task == 'multi-class':
        train_docs, train_doc_labels, categories = convert_to_multi_cls(
            df_train, cro_level=cro_category_level, exclude_op=should_filter_op)
        test_docs, test_doc_labels, _ = convert_to_multi_cls(
            df_test, cro_level=cro_category_level, exclude_op=should_filter_op)

        features = Features({'text': Value('string'), 'labels':
                             ClassLabel(names=categories, num_classes=len(categories))})

    elif task == 'binary':
        train_docs, train_doc_labels = convert_to_binary_cls(
            df_train, cro_level=cro_category_level)
        test_docs, test_doc_labels = convert_to_binary_cls(
            df_test, cro_level=cro_category_level)
        features = Features({'text': Value('string'), 'labels': ClassLabel(
            names=["irrelevant", "relevant"], num_classes=2)})

    # Shuffle the training data
    train_docs, train_doc_labels = shuffle(
        train_docs, train_doc_labels, random_state=seed_value)

    if as_huggingface_ds:
        train_dataset = pd.DataFrame(
            {'text': train_docs, 'labels': train_doc_labels.values.tolist()})
        test_dataset = pd.DataFrame(
            {'text': test_docs, 'labels': test_doc_labels.values.tolist()})
        train_dataset = Dataset.from_pandas(train_dataset, features=features)
        test_dataset = Dataset.from_pandas(test_dataset, features=features)

        dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})

        # TODO: Add this more generally, so also for non "huggingface" ds
        if validation_split > 0:
            dataset['train'], dataset['valid'] = dataset['train'].train_test_split(
                test_size=validation_split, seed=seed_value).values()
        return dataset

    return train_docs, train_doc_labels, test_docs, test_doc_labels
