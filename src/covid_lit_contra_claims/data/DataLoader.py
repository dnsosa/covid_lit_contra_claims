"""
Collection of functions for loading data for covid_lit_contra_claims.
"""

# -*- coding: utf-8 -*-

from covid_lit_contra_claims import MEDNLI_TRAIN_PATH, MEDNLI_DEV_PATH, MEDNLI_TEST_PATH, MANCON_XML_PATH, ROAM_PATH

from .CreateDataset import *


# TODO: Make these arguments in CLI maybe?
MANCON_NEUTRAL_FRAC = 1
MANCON_TRAIN_FRAC = 0.67


def preprocess_nli_corpus_for_pytorch(corpus_id, tokenizer, SEED, truncation=True,
                                      mancon_neutral_frac=MANCON_NEUTRAL_FRAC, mancon_train_frac=MANCON_TRAIN_FRAC):
    if corpus_id == "multinli":
        raw_dataset = create_multinli_dataset(SEED=SEED)

    elif corpus_id == "mednli":
        raw_dataset = create_mednli_dataset(MEDNLI_TRAIN_PATH, MEDNLI_DEV_PATH, MEDNLI_TEST_PATH)

    elif corpus_id == "manconcorpus":
        raw_dataset = create_mancon_dataset(MANCON_XML_PATH, mancon_neutral_frac, mancon_train_frac, SEED=SEED)

    elif corpus_id == "roam":
        raw_dataset = create_roam_dataset(ROAM_PATH)

    else:
        print("Invalid corpus ID. Pre-processing failed. ")
        return None

    old_column_names = raw_dataset['train'].column_names
    old_column_names.remove('labels')

    def tokenize_data(example, tokenizer=tokenizer):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=truncation)

    tokenized_datasets = raw_dataset.map(tokenize_data, batched=True, remove_columns=old_column_names)

    return tokenized_datasets


def load_train_datasets(train_datasets_id: str, SEED: int):
    """
    Create a list of HF Dataset objects from a list of identifiers.

    :param train_datasets_id: string identifier of which datasets to load
    :param SEED: random seed
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
            # TODO 1: Add tokenizer to this call
            # TODO 2: Add truncation to this call
            dataset = preprocess_nli_corpus_for_pytorch(data_id, SEED=SEED)
            train_dataset_dict[data_id] = dataset['train']
            val_dataset_dict[data_id] = dataset['val']
            test_dataset_dict[data_id] = dataset['test']
            print("====...done.====")
        else:
            print(f"WARNING: {data_id} is not a valid data identifier. A Dataset object was not built.")

    return train_dataset_dict, val_dataset_dict, test_dataset_dict


def load_additional_eval_datasets(eval_datasets_id: str, SEED: int):
    """
    Create additional HF Dataset objects from a list of identifiers. These are additional evaluations or benchmarks.

    :param eval_datasets_id: string identifier of which datasets to load
    :param SEED: random seed
    :return: dictionary of the created evaluation (test) Datasets
    """
    eval_dataset_dict = {}

    #TODO
    permissable_eval_ids = set([])
    for data_id in eval_datasets_id.split("_"):
        if data_id in permissable_eval_ids:
            print(f"====Creating {data_id} Dataset object for evaluation only...====")
            # TODO 1: Add tokenizer to this call
            # TODO 2: Add truncation to this call
            dataset = preprocess_nli_corpus_for_pytorch(data_id, SEED=SEED)
            eval_dataset_dict[data_id] = dataset['test']
            print("====...done.====")
        else:
            print(f"WARNING: {data_id} is not a valid data identifier. A Dataset object was not built.")

    return eval_dataset_dict


