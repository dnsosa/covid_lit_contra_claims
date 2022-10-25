"""
Collection of functions for making experimental perturbations to the data for covid_lit_contra_claims.
"""

# -*- coding: utf-8 -*-

import random

from collections import OrderedDict
from datasets import concatenate_datasets


def resize_dataset_with_data_ratio(dataset_dict: OrderedDict, data_ratios: float, is_train: bool, SEED: int):
    """
    Sample data from ["multi", "med", "mancon"] based on the desired ratio of sizes from one to another.

    :param dataset_dict: dict of HF Datasets to be used in training
    :param data_ratios: indicates the ratio of training data from one corpus to the other
    :param tv_flag: is it the training dataset collection? Else, assume it's val
    :param SEED: random seed
    :return: dict of HF Datasets that has been subsampled based on data_ratios
    """

    random.seed(SEED)

    # Using 500 = approx the size of the Roam disjoint training, 100 for val
    smallest_dataset_size = 500 if is_train else 100

    # Adjust the proportions of the input training datasets based on the data_ratios before perturbing order
    if data_ratios is not None:
        big_dataset_names = ["multinli", "mednli", "mancon"]
        # Number of times to apply the multiplier
        ratio_multiplier = len(set(big_dataset_names).intersection(dataset_dict.keys()))

        # Calculate the number of samples we want from each of the big datasets
        for big_dataset_name in big_dataset_names:
            if big_dataset_name in dataset_dict:
                big_dataset_proposed_size = smallest_dataset_size * float(data_ratios) ** ratio_multiplier
                big_dataset_size = min(int(big_dataset_proposed_size), dataset_dict[big_dataset_name].num_rows)
                # Downsample the dataset accordingly
                big_dataset = dataset_dict[big_dataset_name]
                # TODO: fix... this didn't work previously, something about random seeds needing to be ints
                # train_dataset_dict[big_dataset_name] = big_dataset.shuffle(seed=SEED).select(range(big_dataset_count))
                print(f"seed for shuffling before adjusting ratio is {SEED}, which is a(n) {type(SEED)}")
                big_dataset = big_dataset.shuffle(seed=SEED)
                dataset_dict[big_dataset_name] = big_dataset.select(range(big_dataset_size))
                ratio_multiplier -= 1

    return dataset_dict


def prepare_training_data(train_dataset_dict: OrderedDict, train_prep_experiment: str, SEED: int):
    """
    Preprocess input list of training data depending on the experimental procedure declared.

    :param train_dataset_dict: dict of HF Datasets to be used as training
    :param train_prep_experiment: type of experiment requested, can be {"sequential", "combined", "shuffled"}
    :param SEED: random seed
    :return: prepared_train_dataset_list: list of HF Datasets to be used for training after the perturbation
    """

    random.seed(SEED)

    # Now perturb the order based on the train_prep_experiment argument
    if train_prep_experiment == "combined":
        # Combining into a single, mixed up dataset. Returning a list for future processing.
        combined = concatenate_datasets(list(train_dataset_dict.values()))
        combined_key = '_'.join(list(train_dataset_dict.keys()))
        prepared_train_dataset_dict = OrderedDict({combined_key: combined.shuffle(seed=SEED)})

    elif train_prep_experiment == "sequential":
        print(f"Sequential training data preparation. Dataset list will not be perturbed.")
        prepared_train_dataset_dict = train_dataset_dict

    elif train_prep_experiment == "shuffled":
        train_dataset_dict_items = list(train_dataset_dict.items())
        random.shuffle(train_dataset_dict_items)
        prepared_train_dataset_dict = OrderedDict(train_dataset_dict_items)

    else:
        print(f"Input training preparation '{train_prep_experiment}' is invalid. Defaulting to no action.")

    return prepared_train_dataset_dict
