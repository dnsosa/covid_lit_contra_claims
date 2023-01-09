"""Collection of functions for loading and creating HF Datasets from input raw files."""

# -*- coding: utf-8 -*-

from datasets import ClassLabel, Dataset, DatasetDict, load_dataset

import pandas as pd

from .CreateDatasetUtilities import generate_mancon_pandas_dfs, generate_roam_dd_pandas_dfs, \
    generate_roam_dd_ph_pandas_dfs, generate_roam_full_pandas_dfs, generate_roam_ph_pandas_dfs


label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
ClassLabels = ClassLabel(num_classes=len(label_map), names=list(label_map.keys()))


def pandas_dict_to_dataset_dict(in_dict: dict):
    """
    Convert a dictionary of Pandas DataFrames to HF Datasets.

    :param in_dict: dictionary of Pandas DataFrames
    :return: dictionary of HuggingFace Datasets
    """
    splits = list(in_dict.keys())
    datasets_dict = {split: Dataset.from_pandas(in_dict[split].reset_index(drop=True)) for split in splits}
    return DatasetDict(datasets_dict)


def cast_to_class_labels(dataset_dict: DatasetDict):
    """
    Cast labels (dependent variable) to HuggingFace ClassLabels for consistent accounting and being as HF as possible.

    :param dataset_dict: dictionary of HF Datasets
    :return: dictionary of HF Datasets where the labels are now represented by a ClassLabels object
    """
    for split in ['train', 'val', 'test']:
        dataset_dict[split] = dataset_dict[split].cast_column('labels', ClassLabels)
    return dataset_dict


def create_multinli_dataset(SEED: int):
    """
    Create the MultiNLI HF DatasetDict.

    :param SEED: random seed for splitting val into val/test
    :return: MultiNLI HF DatasetDict
    """
    # Load
    multinli_dataset = load_dataset("multi_nli")
    # Preprocess
    multinli_dataset = multinli_dataset.filter(lambda x: x["label"] in label_map.values())
    multinli_dataset = multinli_dataset.rename_column('label', 'labels')
    multinli_dataset = multinli_dataset.rename_column("premise", "sentence1")
    multinli_dataset = multinli_dataset.rename_column("hypothesis", "sentence2")
    # NOTE: making a decision to use only validation_matched for validation. Splitting it into val/test
    multinli_dataset["temp"] = multinli_dataset["validation_matched"].train_test_split(train_size=0.5, seed=SEED)
    multinli_dataset["val"] = multinli_dataset["temp"].pop("train")
    multinli_dataset["test"] = multinli_dataset["temp"].pop("test")
    multinli_dataset.pop("validation_mismatched")
    multinli_dataset.pop("validation_matched")
    multinli_dataset.pop("temp")

    return multinli_dataset


def create_mednli_dataset(mednli_train_path: str, mednli_dev_path: str, mednli_test_path: str):
    """
    Create the MedNLI HF DatasetDict.

    :param mednli_train_path: path to MedNLI training split
    :param mednli_dev_path: path to MedNLI val split
    :param mednli_test_path: path to MedNLI test split
    :return: MedNLI HF DatasetDict
    """
    # Load
    mednli_data_files = {"train": mednli_train_path, "val": mednli_dev_path, "test": mednli_test_path}
    mednli_dataset = load_dataset("json", data_files=mednli_data_files)
    # Preprocess
    mednli_dataset = mednli_dataset.filter(lambda x: x["gold_label"] in label_map.keys())
    mednli_dataset = mednli_dataset.rename_column("gold_label", "labels")
    mednli_dataset = cast_to_class_labels(mednli_dataset)

    return mednli_dataset


def create_mancon_dataset(mancon_xml_path: str, mancon_neutral_frac: float, mancon_train_frac: float, SEED: int,
                          single_sent_only=False):
    """
    Create the ManConCorpus HF DatasetDict.

    :param mancon_xml_path: path to raw XML ManConCorpus
    :param mancon_neutral_frac: the Neutrals class should = what fraction of the next largest class
    :param mancon_train_frac: what fraction should be used for train
    :param SEED: random seed
    :param single_sent_only: if True, only create dataset containing only single claim as benchmark
    :return: ManConCorpus HF DatasetDict
    """
    # Load
    raw_mancon_pandas_df_dict = generate_mancon_pandas_dfs(mancon_xml_path, neutral_frac=mancon_neutral_frac,
                                                           mancon_train_frac=mancon_train_frac, SEED=SEED,
                                                           single_sent_only=single_sent_only)
    mancon_dataset = pandas_dict_to_dataset_dict(raw_mancon_pandas_df_dict)
    # Preprocess
    mancon_dataset = mancon_dataset.rename_column("label", "labels")
    mancon_dataset = cast_to_class_labels(mancon_dataset)

    return mancon_dataset


def create_roam_dataset(roam_path, single_sent_only=False):
    """
    Create the Roam disjoint HF DatasetDict.

    :param roam_path: path to disjoint Roam dataset
    :return: Disjoint Roam HF DatasetDict
    """
    # Load
    roam_df_list = []
    splits = ["Train", "Val", "Test"]
    for data_split in splits:
        roam_df = pd.read_excel(roam_path, sheet_name=data_split)
        roam_df = roam_df.drop(roam_df.columns[0], axis=1)
        roam_df = roam_df.dropna().reset_index(drop=True)
        if single_sent_only and data_split in ["Val", "Test"]:
            roam_df["text2"] = "."
        roam_df_list.append(Dataset.from_pandas(roam_df))
    raw_roam_dataset_dict = dict(zip([split.lower() for split in splits], roam_df_list))
    roam_dataset = DatasetDict(raw_roam_dataset_dict)
    # Preprocess
    roam_dataset = roam_dataset.filter(lambda x: x["annotation"] in label_map.keys())
    roam_dataset = roam_dataset.rename_column("text1", "sentence1")
    roam_dataset = roam_dataset.rename_column("text2", "sentence2")
    roam_dataset = roam_dataset.rename_column("annotation", "labels")
    roam_dataset = cast_to_class_labels(roam_dataset)

    return roam_dataset


def create_all_pairs_dataset(all_claims_path):
    """
    Create the all CORD19 claim pairs DatasetDict.

    :param all_claims_path: path to all claim pairs
    :return: All claim pairs DatasetDict
    """
    # Load
    splits = ["Test"]
    claims_df = pd.read_csv(all_claims_path)
    claims_df = claims_df.drop(claims_df.columns[0], axis=1)
    claims_df = claims_df.dropna().reset_index(drop=True)
    claims_df_list = [Dataset.from_pandas(claims_df)]
    raw_claims_dataset_dict = dict(zip([split.lower() for split in splits], claims_df_list))
    claims_dataset = DatasetDict(raw_claims_dataset_dict)

    # Preprocess
    claims_dataset = claims_dataset.rename_column("text1", "sentence1")
    claims_dataset = claims_dataset.rename_column("text2", "sentence2")

    return claims_dataset


def create_roam_full_dataset(roam_full_path, SEED: int):
    """
    Create the Roam full HF DatasetDict.

    :param roam_full_path: path to full Roam dataset
    :param SEED: random seed
    :return: Full Roam HF DatasetDict
    """
    # Load
    raw_roam_full_pandas_df_dict = generate_roam_full_pandas_dfs(roam_full_path, SEED=SEED)
    roam_full_dataset = pandas_dict_to_dataset_dict(raw_roam_full_pandas_df_dict)
    # Preprocess
    roam_full_dataset = cast_to_class_labels(roam_full_dataset)

    return roam_full_dataset


def create_roam_ph_dataset(roam_full_path, SEED: int):
    """
    Create the PH Roam HF DatasetDict.

    :param roam_full_path: path to full Roam dataset
    :param SEED: random seed
    :return: PH Roam HF DatasetDict
    """
    # Load
    raw_roam_ph_pandas_df_dict = generate_roam_ph_pandas_dfs(roam_full_path, SEED=SEED)
    roam_ph_dataset = pandas_dict_to_dataset_dict(raw_roam_ph_pandas_df_dict)
    # Preprocess
    roam_ph_dataset = cast_to_class_labels(roam_ph_dataset)

    return roam_ph_dataset


def create_roam_dd_dataset(roam_full_path, SEED: int):
    """
    Create the DD Roam HF DatasetDict.

    :param roam_full_path: path to full Roam dataset
    :param SEED: random seed
    :return: DD Roam HF DatasetDict
    """
    # Load
    raw_roam_dd_pandas_df_dict = generate_roam_dd_pandas_dfs(roam_full_path, SEED=SEED)
    roam_dd_dataset = pandas_dict_to_dataset_dict(raw_roam_dd_pandas_df_dict)
    # Preprocess
    roam_dd_dataset = cast_to_class_labels(roam_dd_dataset)

    return roam_dd_dataset


def create_roam_dd_ph_dataset(roam_full_path, SEED: int):
    """
    Create the DD-PH Roam HF DatasetDict.

    :param roam_full_path: path to full Roam dataset
    :param SEED: random seed
    :return: DD-PH Roam HF DatasetDict
    """
    # Load
    raw_roam_dd_ph_pandas_df_dict = generate_roam_dd_ph_pandas_dfs(roam_full_path, SEED=SEED)
    roam_dd_ph_dataset = pandas_dict_to_dataset_dict(raw_roam_dd_ph_pandas_df_dict)
    # Preprocess
    roam_dd_ph_dataset = cast_to_class_labels(roam_dd_ph_dataset)

    return roam_dd_ph_dataset
