"""
Collection of functions for making experimental perturbations to the data for covid_lit_contra_claims.
"""

# -*- coding: utf-8 -*-

import random

from collections import OrderedDict
from datasets import concatenate_datasets


def prepare_training_data(train_dataset_dict: OrderedDict, train_prep_experiment: str, SEED: int,
                          data_ratios: float = None):
    """
    Preprocess input list of training data depending on the experimental procedure declared

    :param train_dataset_dict: list of HF Datasets to be used as training
    :param train_prep_experiment: type of experiment requested, can be {"sequential", "combined", "shuffled"}
    :param SEED: random seed
    :param data_ratios: indicates the ratio of training data from one corpus to the other
    :return: prepared_train_dataset_list: list of HF Datasets to be used for training after the perturbation
    """

    random.seed(SEED)

    # Adjust the proportions of the input training datasets based on the data_ratios before perturbing order
    if data_ratios is not None:
        big_dataset_names = ["multinli", "mednli", "mancon"]
        # Number of times to apply the multiplier
        ratio_multiplier = len(set(big_dataset_names).intersection(train_dataset_dict.keys()))

        # Calculate the number of samples we want from each of the big datasets
        for big_dataset_name in big_dataset_names:
            if big_dataset_name in train_dataset_dict:
                # Using 500 = approx the size of the Roam disjoint training
                big_dataset_proposed_count = 500 * data_ratios ** ratio_multiplier
                big_dataset_count = min(int(big_dataset_proposed_count), train_dataset_dict[big_dataset_name].num_rows)
                # Downsample the dataset accordingly
                big_dataset = train_dataset_dict[big_dataset_name]
                train_dataset_dict[big_dataset_name] = big_dataset.shuffle(seed=SEED).select(range(big_dataset_count))
                ratio_multiplier -= 1

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
