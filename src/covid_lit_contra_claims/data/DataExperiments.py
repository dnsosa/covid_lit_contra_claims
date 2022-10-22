"""
Collection of functions for making experimental perturbations to the data for covid_lit_contra_claims.
"""

# -*- coding: utf-8 -*-

from datasets import concatenate_datasets


def prepare_training_data(train_dataset_dict, train_prep_experiment: str, data_ratios: float):
    """
    Preprocess input list of training data depending on the experimental procedure declared

    :param train_dataset_dict: list of HF Datasets to be used as training
    :param train_prep_experiment: type of experiment requested, can be {"sequential", "combined"}
    :param data_ratios: indicates the ratio of training data from one corpus to the other
    :return: prepared_train_dataset_list: list of HF Datasets to be used for training after the perturbation
    """

    # TODO: incorporate data_ratios
    if train_prep_experiment == "combined":
        # Combining into a single, mixed up dataset. Returning a list for future processing.
        # TODO: train_dict_list
        train_dataset_list = [concatenate_datasets(train_dataset_dict).shuffle()]

    elif train_prep_experiment == "sequential":
        print(f"Sequential training data preparation. Dataset list will not be perturbed.")

    elif train_prep_experiment == "shuffled":
        #train_dataset_keys = list(train_dataset_dict)
        # TODO: implement shuffling. Make sure seed is set
        pass

    else:
        print(f"Input training preparation '{train_prep_experiment}' is invalid. Defaulting to no action.")

    return train_dataset_list
