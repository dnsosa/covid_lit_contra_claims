"""
Collection of functions for loading and creating HF Datasets from input raw files.
"""

# -*- coding: utf-8 -*-

import pandas as pd

from datasets import ClassLabel, Dataset, DatasetDict, load_dataset

from .CreateDatasetUtilities import generate_mancon_pandas_dfs


label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
ClassLabels = ClassLabel(num_classes=len(label_map), names=list(label_map.keys()))


def create_multinli_dataset(SEED):
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


def create_mednli_dataset(mednli_train_path, mednli_dev_path, mednli_test_path):
    # Load
    mednli_data_files = {"train": mednli_train_path, "val": mednli_dev_path, "test": mednli_test_path}
    mednli_dataset = load_dataset("json", data_files=mednli_data_files)
    # Preprocess
    mednli_dataset = mednli_dataset.filter(lambda x: x["gold_label"] in label_map.keys())
    mednli_dataset = mednli_dataset.rename_column("gold_label", "labels")
    for split in ['train', 'val', 'test']:
        mednli_dataset[split] = mednli_dataset[split].cast_column('labels', ClassLabels)

    return mednli_dataset


def create_mancon_dataset(mancon_xml_path, mancon_neutral_frac, mancon_train_frac, SEED):
    # Load
    raw_mancon_pandas_df_dict = generate_mancon_pandas_dfs(mancon_xml_path, neutral_frac=mancon_neutral_frac,
                                                           mancon_train_frac=mancon_train_frac, SEED=SEED)
    # Cast the DataFrames to Datasets
    raw_mancon_dataset_dict = {split: Dataset.from_pandas(raw_mancon_pandas_df_dict[split].reset_index(drop=True)) for
                               split in raw_mancon_pandas_df_dict.keys()}
    mancon_dataset = DatasetDict(raw_mancon_dataset_dict)
    # Preprocess
    mancon_dataset = mancon_dataset.rename_column("label", "labels")
    for split in ['train', 'val', 'test']:
        mancon_dataset[split] = mancon_dataset[split].cast_column('labels', ClassLabels)

    return mancon_dataset


def create_roam_dataset(roam_path):
    # Load
    roam_df_list = []
    splits = ["Train", "Val", "Test"]
    for data_split in splits:
        roam_df = pd.read_excel(roam_path, sheet_name=data_split)
        roam_df = roam_df.drop(roam_df.columns[0], axis=1)
        roam_df = roam_df.dropna().reset_index(drop=True)
        roam_df_list.append(Dataset.from_pandas(roam_df))
    raw_roam_dataset_dict = dict(zip([split.lower() for split in splits], roam_df_list))
    roam_dataset = DatasetDict(raw_roam_dataset_dict)
    # Preprocess
    roam_dataset = roam_dataset.filter(lambda x: x["annotation"] in label_map.keys())
    roam_dataset = roam_dataset.rename_column("text1", "sentence1")
    roam_dataset = roam_dataset.rename_column("text2", "sentence2")
    roam_dataset = roam_dataset.rename_column("annotation", "labels")
    for split in ['train', 'val', 'test']:
        roam_dataset[split] = roam_dataset[split].cast_column('labels', ClassLabels)

    return roam_dataset
