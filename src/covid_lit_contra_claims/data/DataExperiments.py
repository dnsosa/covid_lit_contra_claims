"""
Collection of functions for making experimental perturbations to the data for covid_lit_contra_claims.
"""

# -*- coding: utf-8 -*-

from datasets import concatenate_datasets


def prepare_training_data(train_dataset_list, train_prep_experiment: str):
    """
    Preprocess input list of training data depending on the experimental procedure declared

    :param train_dataset_list: list of HF Datasets to be used as training
    :param train_prep_experiment: type of experiment requested, can be {"sequential", "combined"}
    :return: prepared_train_dataset_list: list of HF Datasets to be used for training after the perturbation
    """

    if train_prep_experiment == "combined":
        # Combining into a single, mixed up dataset. Returning a list for future processing.
        train_dataset_list = [concatenate_datasets(train_dataset_list).shuffle()]

    elif train_prep_experiment == "sequential":
        print(f"Sequential training data preparation. Dataset list will not be perturbed.")
    else:
        print(f"Input training preparation '{train_prep_experiment}' is invalid. Defaulting to no action.")

    return train_dataset_list
