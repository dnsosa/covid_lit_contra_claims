"""
Collection of functions for loading data for covid_lit_contra_claims.
"""

# -*- coding: utf-8 -*-

import os

import pandas as pd
import xml.etree.ElementTree as ET

from covid_lit_contra_claims import MEDNLI_TRAIN_PATH, MEDNLI_DEV_PATH, MEDNLI_TEST_PATH, MANCON_XML_PATH, ROAM_PATH
from datasets import load_dataset, Dataset, DatasetDict
from itertools import combinations
from sklearn.model_selection import train_test_split


# TODO: Make these arguments in CLI maybe?
SEED = 42
LABEL_MAP = {"entailment": 0, "neutral": 1, "contradiction": 2}
MANCON_NEUTRAL_FRAC = 1
TRAIN_VAL_FRAC = 0.8


def generate_mancon_pandas_dfs(xml_path: str, neutral_frac: float, mancon_train_frac: float, SEED: int):
    """
    Create sentence pairs dataset from the original xml ManCon Corpus.

    :param xml_path: path to xml corpus
    :param neutral_frac: sample the neutrals to have size (neutral_frac * size_of_next_biggest_class), if None don't downsample
    """

    xtree = ET.parse(xml_path)  # TODO: Fix error # noqa: S314
    xroot = xtree.getroot()

    manconcorpus_df_list = []

    for node in xroot:
        for claim in node.findall('CLAIM'):
            manconcorpus_df_list.append({'question': claim.attrib.get('QUESTION'),
                                         'claim': claim.text,
                                         'assertion': claim.attrib.get('ASSERTION')})

    manconcorpus_data = pd.DataFrame(manconcorpus_df_list)

    questions = list(set(manconcorpus_data.question))
    train_qs, valtest_qs = train_test_split(questions, test_size=(1 - mancon_train_frac), shuffle=True,
                                            random_state=SEED)
    val_qs, test_qs = train_test_split(valtest_qs, test_size=0.5, shuffle=True, random_state=SEED)

    mancon_nli_df_dict = {}
    splits = ["train", "val", "test"]

    for split_i, questions in enumerate([train_qs, val_qs, test_qs]):
        con_list, ent_list, neu_list = [], [], []

        prev_qs = []
        for q in questions:
            # Keep track of previous Q's
            prev_qs.append(q)

            # Find the yes and nos for this Q
            claim_yes = pd.DataFrame(manconcorpus_data.loc[(manconcorpus_data.question == q)
                                                           & (
                                                                       manconcorpus_data.assertion == 'YS'), 'claim'])  # noqa: W503
            claim_no = pd.DataFrame(manconcorpus_data.loc[(manconcorpus_data.question == q)
                                                          & (
                                                                      manconcorpus_data.assertion == 'NO'), 'claim'])  # noqa: W503

            # Create the contras
            temp = claim_yes.assign(key=1).merge(claim_no.assign(key=1), on='key').drop(columns='key')
            temp = temp.rename(columns={'claim_x': 'sentence_1', 'claim_y': 'sentence_2'})
            temp['label'] = 'contradiction'
            con_list += temp.drop_duplicates().values.tolist()

            # Create the entails
            for i, j in list(combinations(claim_yes.index, 2)):
                ent_list.append([claim_yes.claim[i], claim_yes.claim[j], 'entailment'])
            for i, j in list(combinations(claim_no.index, 2)):
                ent_list.append([claim_no.claim[i], claim_no.claim[j], 'entailment'])

            # Create the neutrals
            claims_from_q = pd.DataFrame(manconcorpus_data.loc[(manconcorpus_data.question == q), 'claim'])
            claims_from_other_qs = pd.DataFrame(
                manconcorpus_data.loc[~(manconcorpus_data.question.isin(prev_qs)), 'claim'])
            temp = claims_from_q.assign(key=1).merge(claims_from_other_qs.assign(key=1), on='key').drop(columns='key')
            temp = temp.rename(columns={'claim_x': 'sentence_1', 'claim_y': 'sentence_2'})
            temp['label'] = 'neutral'
            neu_list += temp.drop_duplicates().values.tolist()

        mancon_nli_df = pd.concat([pd.DataFrame(con_list), pd.DataFrame(ent_list), pd.DataFrame(neu_list)]).reset_index(
            drop=True)
        mancon_nli_df.columns = ["sentence1", "sentence2", "label"]

        if neutral_frac is not None:
            mancon_con_ent_df = mancon_nli_df[mancon_nli_df.label.isin(['entailment', 'contradiction'])]
            count_by_label = mancon_nli_df.groupby('label').size()
            max_non_neu = max(count_by_label['entailment'], count_by_label['contradiction'])
            mancon_neu_down_df = mancon_nli_df[mancon_nli_df.label == 'neutral'].sample(n=round(max_non_neu * neutral_frac))
            mancon_nli_df = pd.concat([mancon_con_ent_df, mancon_neu_down_df])

        mancon_nli_df_dict[splits[split_i]] = mancon_nli_df.sample(frac=1, random_state=SEED)

    return mancon_nli_df_dict


def label_str_to_num(example, label_map=LABEL_MAP):
    example['labels'] = label_map[example['labels']]
    return example


# TODO: Transfer over when more mature.... gut out parts of this into their own little functions for each type
def preprocess_nli_corpus_for_pytorch(corpus_id, tokenizer, truncation, mancon_neutral_frac=1, mancon_split_qs=True,
                                      mancon_tv_frac=0.8):
    def tokenize_data(example, tokenizer=tokenizer):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=truncation)

    if corpus_id == "multinli":
        # Load
        raw_corpus = load_dataset("multi_nli")
        # Preprocess
        raw_corpus = raw_corpus.filter(lambda x: x["label"] != -1)
        raw_corpus = raw_corpus.rename_column('label', 'labels')
        raw_corpus = raw_corpus.rename_column("premise", "sentence1")
        raw_corpus = raw_corpus.rename_column("hypothesis", "sentence2")
        # NOTE: making a decision to use only validation_matched for validation. Splitting it into val/test
        raw_corpus["temp"] = raw_corpus["validation_matched"].train_test_split(train_size=0.5, seed=SEED)
        raw_corpus["val"] = raw_corpus["temp"].pop("train")
        raw_corpus["test"] = raw_corpus["temp"].pop("test")
        raw_corpus.pop("validation_mismatched")
        raw_corpus.pop("validation_matched")
        raw_corpus.pop("temp")
        old_column_names = raw_corpus['train'].column_names
        old_column_names.remove('labels')
        tokenized_datasets = raw_corpus.map(tokenize_data, batched=True, remove_columns=old_column_names)

    elif corpus_id == "mednli":
        # Load
        mednli_data_files = {"train": mednli_train_path, "val": mednli_dev_path, "test": mednli_test_path}
        raw_corpus = load_dataset("json", data_files=mednli_data_files)
        # Preprocess
        raw_corpus = raw_corpus.filter(lambda x: x["gold_label"] in label_map.keys())
        raw_corpus = raw_corpus.rename_column("gold_label", "labels")
        raw_corpus = raw_corpus.map(label_str_to_num)
        old_column_names = raw_corpus['train'].column_names
        old_column_names.remove('labels')
        tokenized_datasets = raw_corpus.map(tokenize_data, batched=True, remove_columns=old_column_names)

    elif corpus_id == "manconcorpus":
        # Load
        raw_mancon_pandas_df = generate_mancon_pandas_dfs(mancon_xml_path, neutral_frac=mancon_neutral_frac,
                                                          mancon_split_qs=mancon_split_qs,
                                                          mancon_tv_frac=mancon_tv_frac)
        raw_mancon_pandas_df = raw_mancon_pandas_df.sample(frac=1, random_state=SEED)
        raw_corpus = Dataset.from_pandas(raw_mancon_pandas_df)
        # Preproces
        # First, create train, val, test splits.
        raw_corpus = Dataset.from_pandas(raw_mancon_pandas_df.reset_index())
        # TODO: Update me here
        raw_corpus = raw_corpus.train_test_split(train_size=mancon_tv_frac, seed=SEED)
        raw_corpus["temp"] = raw_corpus["train"].train_test_split(train_size=len(raw_corpus['test']), seed=SEED)
        raw_corpus["val"] = raw_corpus["temp"].pop("train")
        raw_corpus["train"] = raw_corpus["temp"].pop("test")
        raw_corpus.pop("temp")
        # Second, do preprocessing as normal
        raw_corpus = raw_corpus.rename_column("label", "labels")
        raw_corpus = raw_corpus.map(label_str_to_num)
        old_column_names = raw_corpus['train'].column_names
        old_column_names.remove('labels')
        tokenized_datasets = raw_corpus.map(tokenize_data, batched=True, remove_columns=old_column_names)

    elif corpus_id == "roam":
        # Load
        cord_df_list = []
        splits = ["Train", "Val", "Test"]
        for data_split in splits:
            cord_df = pd.read_excel(roam_path, sheet_name=data_split)
            cord_df = cord_df.drop(cord_df.columns[0], axis=1)
            cord_df = cord_df.dropna().reset_index(drop=True)
            cord_df_list.append(Dataset.from_pandas(cord_df))
        raw_corpus_dict = dict(zip([split.lower() for split in splits], cord_df_list))
        raw_corpus = DatasetDict(raw_corpus_dict)
        # Preprocess
        raw_corpus = raw_corpus.filter(lambda x: x["annotation"] in label_map.keys())
        raw_corpus = raw_corpus.rename_column("text1", "sentence1")
        raw_corpus = raw_corpus.rename_column("text2", "sentence2")
        raw_corpus = raw_corpus.rename_column("annotation", "labels")
        raw_corpus = raw_corpus.map(label_str_to_num)
        old_column_names = raw_corpus['train'].column_names
        old_column_names.remove('labels')
        tokenized_datasets = raw_corpus.map(tokenize_data, batched=True, remove_columns=old_column_names)

    else:
        print("Invalid corpus ID. Pre-processing failed. ")
        return None

    return tokenized_datasets


def load_train_datasets(train_datasets_id: str):
    """
    Create a list of HF Dataset objects from a list of identifiers.

    :param train_datasets_id: string identifier of which datasets to load
    :return: dictionaries of the created train, val, and test Datasets
    """
    train_dataset_dict = {}
    val_dataset_dict = {}
    test_dataset_dict = {}

    #TODO
    permissable_train_ids = set(["multinli", "mednli", "manconcorpus", "roam", ...])
    for data_id in train_datasets_id.split("_"):
        if data_id in permissable_train_ids:
            print(f"====Creating {data_id} Dataset object for train/val/test...====")
            dataset = preprocess_nli_corpus_for_pytorch(data_id)
            train_dataset_dict[data_id] = dataset['train']
            val_dataset_dict[data_id] = dataset['val']
            test_dataset_dict[data_id] = dataset['test']
            print("====...done.====")
        else:
            print(f"WARNING: {data_id} is not a valid data identifier. A Dataset object was not built.")

    return train_dataset_dict, val_dataset_dict, test_dataset_dict


def load_additional_eval_datasets(eval_datasets_id: str):
    """
    Create additional HF Dataset objects from a list of identifiers. These are additional evaluations or benchmarks.

    :param eval_datasets_id: string identifier of which datasets to load
    :return: dictionary of the created evaluation (test) Datasets
    """
    eval_dataset_dict = {}

    #TODO
    permissable_eval_ids = set([])
    for data_id in eval_datasets_id.split("_"):
        if data_id in permissable_eval_ids:
            print(f"====Creating {data_id} Dataset object for evaluation only...====")
            dataset = preprocess_nli_corpus_for_pytorch(data_id)
            eval_dataset_dict[data_id] = dataset['test']
            print("====...done.====")
        else:
            print(f"WARNING: {data_id} is not a valid data identifier. A Dataset object was not built.")

    return eval_dataset_dict


