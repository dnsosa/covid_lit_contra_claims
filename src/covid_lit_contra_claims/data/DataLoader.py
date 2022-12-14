"""Collection of functions for loading data for covid_lit_contra_claims."""

# -*- coding: utf-8 -*-

from collections import OrderedDict

from .CreateDataset import create_all_pairs_dataset, create_mancon_dataset, create_mednli_dataset, \
    create_multinli_dataset, create_roam_dataset, create_roam_dd_dataset, create_roam_dd_ph_dataset, \
    create_roam_full_dataset, create_roam_ph_dataset
from .constants import ALL_CLAIMS_PATH, MANCON_NEUTRAL_FRAC, MANCON_TRAIN_FRAC, MANCON_XML_PATH, MEDNLI_DEV_PATH, \
    MEDNLI_TEST_PATH, MEDNLI_TRAIN_PATH, ROAM_ALL_PATH, ROAM_SEP_PATH


def preprocess_nli_corpus_for_pytorch(corpus_id, tokenizer, truncation=True, SEED=42,
                                      mancon_neutral_frac=MANCON_NEUTRAL_FRAC, mancon_train_frac=MANCON_TRAIN_FRAC):
    """
    Preprocess the corpora by creating the datasets for HF.

    :param corpus_id: string identifier for corpus
    :param tokenizer: HF tokenizer
    :param truncation: if True, truncate as normal
    :param SEED: random seed
    :param mancon_neutral_frac: downsample neutral class from ManCon to be (size of the next biggest class) * MNF
    :param mancon_train_frac: fraction of questions from ManCon to use for training--split rest between val/test
    :return: tokenized Dataset objects
    """
    if corpus_id == "multinli":
        raw_dataset = create_multinli_dataset(SEED=SEED)

    elif corpus_id == "mednli":
        raw_dataset = create_mednli_dataset(MEDNLI_TRAIN_PATH, MEDNLI_DEV_PATH, MEDNLI_TEST_PATH)

    elif corpus_id == "mancon":
        raw_dataset = create_mancon_dataset(MANCON_XML_PATH, mancon_neutral_frac, mancon_train_frac, SEED=SEED)

    elif corpus_id == "manconSS":
        raw_dataset = create_mancon_dataset(MANCON_XML_PATH, mancon_neutral_frac, mancon_train_frac, SEED=SEED,
                                            single_sent_only=True)

    elif corpus_id == "roam":
        raw_dataset = create_roam_dataset(ROAM_SEP_PATH)

    elif corpus_id == "roamAll":
        raw_dataset = create_roam_full_dataset(ROAM_ALL_PATH, SEED=SEED)

    elif corpus_id == "roamPH":
        raw_dataset = create_roam_ph_dataset(ROAM_ALL_PATH, SEED=SEED)

    elif corpus_id == "roamDD":
        raw_dataset = create_roam_dd_dataset(ROAM_ALL_PATH, SEED=SEED)

    elif corpus_id == "roamDDPH":
        raw_dataset = create_roam_dd_ph_dataset(ROAM_ALL_PATH, SEED=SEED)

    elif corpus_id == "roamSS":
        raw_dataset = create_roam_dataset(ROAM_SEP_PATH, single_sent_only=True)

    elif corpus_id == "allPairs":
        raw_dataset = create_all_pairs_dataset(ALL_CLAIMS_PATH)

    else:
        print("Invalid corpus ID. Pre-processing failed. ")
        return None

    if corpus_id == "allPairs":
        old_column_names = raw_dataset['test'].column_names
    else:
        old_column_names = raw_dataset['train'].column_names
        old_column_names.remove('labels')

    def tokenize_data(example, tokenizer=tokenizer):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=truncation)

    tokenized_datasets = raw_dataset.map(tokenize_data, batched=True, remove_columns=old_column_names)

    return tokenized_datasets


def load_train_datasets(train_datasets_id: str, tokenizer, truncation: bool, SEED: int):
    """
    Create a list of HF Dataset objects from a list of identifiers.

    :param train_datasets_id: string identifier of which datasets to load
    :param tokenizer: model's tokenizer for dataset
    :param truncation: boolean indicating if input should be truncated when tokenized
    :param SEED: random seed
    :return: dictionaries of the created train, val, and test Datasets
    """
    train_dataset_dict = OrderedDict()
    val_dataset_dict = OrderedDict()
    test_dataset_dict = OrderedDict()

    permissable_train_ids = {"multinli", "mednli", "mancon", "manconSS", "roam", "roamAll", "roamPH", "roamDD",
                             "roamDDPH", "roamSS", "allPairs"}
    for data_id in train_datasets_id.split("_"):
        if data_id in permissable_train_ids:
            print(f"====Creating {data_id} Dataset object for train/val/test...====")
            dataset = preprocess_nli_corpus_for_pytorch(data_id, tokenizer, truncation=truncation, SEED=SEED)
            train_dataset_dict[data_id] = dataset['train']
            val_dataset_dict[data_id] = dataset['val']
            test_dataset_dict[data_id] = dataset['test']
            print("====...done.====")
        else:
            print(f"WARNING: {data_id} is not a valid data identifier. A Dataset object was not built.")

    return train_dataset_dict, val_dataset_dict, test_dataset_dict


def load_additional_eval_datasets(eval_datasets_id: str, tokenizer, truncation: bool, SEED: int):
    """
    Create additional HF Dataset objects from a list of identifiers. These are additional evaluations or benchmarks.

    :param eval_datasets_id: string identifier of which datasets to load
    :param tokenizer: model's tokenizer for dataset
    :param truncation: boolean indicating if input should be truncated when tokenized
    :param SEED: random seed
    :return: dictionary of the created evaluation (test) Datasets
    """
    eval_dataset_dict = OrderedDict()

    permissable_eval_ids = {"multinli", "mednli", "mancon", "manconSS", "roam", "roamAll", "roamPH", "roamDD",
                            "roamDDPH", "roamSS", "allPairs"}
    for data_id in eval_datasets_id.split("_"):
        if data_id in permissable_eval_ids:
            print(f"====Creating {data_id} Dataset object for evaluation only...====")
            dataset = preprocess_nli_corpus_for_pytorch(data_id, tokenizer, truncation=truncation, SEED=SEED)
            eval_dataset_dict[data_id] = dataset['test']
            print("====...done.====")
        else:
            print(f"WARNING: {data_id} is not a valid data identifier. A Dataset object was not built.")

    return eval_dataset_dict
