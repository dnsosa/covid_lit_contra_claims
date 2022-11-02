"""
Collection of functions for training the models for the covid_lit_contra_claims pipeline.
"""

# -*- coding: utf-8 -*-

import json
import os

import evaluate
import numpy as np
import pandas as pd
import torch
import wandb

from ..data.constants import model_id_mapper, MANCON_NEUTRAL_FRAC, MANCON_TRAIN_FRAC, WANDB_LOG_INTERVAL, WANDB_LOG_FREQ

from torch.utils.data import DataLoader
from transformers import AdamW, AutoModelForSequenceClassification, DataCollatorWithPadding, get_scheduler

# Use TF32 instead of FP32 -- decrease precision in a non-impactful way for a speed up
torch.backends.cuda.matmul.allow_tf32 = True


def run_validation(model, trainings_so_far, val_dataset_dict, config, data_collator, device, overall_results):
    for val_id, val_dataset in val_dataset_dict.items():
        # TODO: Do we want shuffle = True?
        val_dataloader = DataLoader(val_dataset,
                                    shuffle=False,
                                    batch_size=config['batch_size'],
                                    collate_fn=data_collator)

        print(f"EVAL: Created a DataLoader for corpus '{val_id}'...")

        acc_metric = evaluate.load('accuracy')
        f1_metric = evaluate.load('f1', average='macro')
        precision_metric = evaluate.load('precision', average='macro')
        recall_metric = evaluate.load('recall', average='macro')
        recall_metric2 = evaluate.load('recall', average=None)

        for batch_idx, batch in enumerate(val_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                val_loss = outputs.loss

                if batch_idx % config['wandb_log_interval'] == 0:
                    wandb.log({f"Val Loss on {val_id} (trained: {trainings_so_far})": val_loss})

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            for metric in [acc_metric, f1_metric, precision_metric, recall_metric, recall_metric2]:
                metric.add_batch(predictions=predictions, references=batch["labels"])

        results = acc_metric.compute()  # Creates a dictionary
        for metric in [f1_metric, precision_metric, recall_metric]:
            results.update(metric.compute(average='macro'))  # Add these to the dict

        # Calculate recall of just contradictions class
        if val_id in ["roamPH", "roamDD", "roamDDPH"]:
            recall_ent_val = -1  # TODO: What should the value be?
            recall_con_val = -1  # Flag for recall con is undefined here
        else:
            recall_array = recall_metric2.compute(average=None)['recall']
            recall_ent_val = recall_array[0]  # 0 == entailment index
            recall_con_val = recall_array[2]  # 2 == contradictions index

        results.update({'recall_ent': recall_ent_val, 'recall_con': recall_con_val})

        # format the name of the result a bit
        wandb_results = {f"Val: {val_id.upper()} {k.capitalize()}": v for k, v in results.items()}
        wandb.log(wandb_results)
        # torch.onnx.export(model, batch, "model.onnx")
        # wandb.save("model.onnx")

        results["Cumulative Training"] = trainings_so_far
        results["Validation Set"] = val_id
        results.update(config)

        print(f"Results on {val_id}: {results}")
        overall_results.append(results)

    return None


def train_model(model_id, tokenizer, train_dataset_dict, val_dataset_dict, training_args, try_speed, out_dir, SEED, is_test=False):
    """
    Main function for training the HF model.

    :param model: identifier of the model(s) to train.
    :param train_dataset_dict: dictionary of training Datasets
    :param val_dataset_dict: dictionary of validation Datasets
    :param training_args: arguments for training
    :param out_dir: directory to output weights/intermediate results
    :param SEED: random seed
    :return: trained model
    """

    # Set random seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Configs and init for WandB
    additional_configs = {"mancon_neutral_frac": MANCON_NEUTRAL_FRAC,
                          "mancon_train_frac": MANCON_TRAIN_FRAC,
                          "wandb_log_interval": WANDB_LOG_INTERVAL}

    config = dict(training_args, **additional_configs)  # Merge the dicts
    wandb.init(project='COVID Drug Contra Claims', config=config)
    print("WandB initialized.")

    # Load the model, initialize the optimizer
    checkpoint = model_id_mapper[model_id]
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    wandb.watch(model, log_freq=WANDB_LOG_FREQ)
    print("Model loaded.")

    # Find the right device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    print(f"Using device {device}.")

    # Create data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Train on datasets in the input training dictionary
    overall_results = []
    training_list_so_far = ["none"]

    # Potential optimization
    num_workers = 4 if try_speed else 0

    # Calculate performance on all val corpora before any fine-tuning
    # NOTE: Not going to do this. Performance will be bad because model hasn't been fine-tuned on any NLI task yet.
    # model.eval()
    # run_validation(model, "none", val_dataset_dict, config, data_collator, device, overall_results)

    for train_id, train_dataset in train_dataset_dict.items():
        training_list_so_far.append(train_id)
        trainings_so_far = "_".join(training_list_so_far[:-1])
        # TODO: Do we want shuffle = True?
        train_dataloader = DataLoader(train_dataset,
                                      shuffle=False,
                                      batch_size=config['batch_size'],
                                      collate_fn=data_collator,
                                      num_workers=num_workers,  # speed up
                                      pin_memory=try_speed)  # speed up
        print(f"Created a DataLoader for corpus '{train_id}'...")

        # Create a learning rate scheduler
        num_training_steps = config['epochs'] * len(train_dataloader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        print(f"Created a learning rate scheduler for corpus '{train_id}'...")

        print("Beginning training...")
        print(f"# Epochs: {config['epochs']}")
        model.train()

        for epoch in range(config['epochs']):
            for batch_idx, batch in enumerate(train_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if batch_idx % config['wandb_log_interval'] == 0:
                    # Identifier for all trainings up to this one
                    wandb.log({"epoch": epoch, f"Training Loss on {train_id} (prev: {trainings_so_far})": loss})

        print(f"Training complete for corpus '{train_id}'.")
        print("Beginning evaluation...")

        # Run validation on all val corpora. Overall results will get updated.
        model.eval()
        trainings_so_far = "_".join(training_list_so_far)
        run_validation(model, trainings_so_far, val_dataset_dict, config, data_collator, device, overall_results)

    print("Overall evaluation complete.")

    # Establish output directory
    prep_args = f"PREPARGS_DR-{config['data_ratios']}_TPE-{config['train_prep_experiment']}_T-{config['truncation']}"
    experiment_id = f"EXP_E-{config['epochs']}_B-{config['batch_size']}_LR-{config['learning_rate']}"
    experiment_out_dir = os.path.join(out_dir,
                                      model_id,
                                      f"TRAIN_{config['train_datasets']}",
                                      f"VAL_{config['eval_datasets']}",
                                      prep_args,
                                      experiment_id)
    os.makedirs(experiment_out_dir, exist_ok=True)

    eval_split = "test" if is_test else "val"
    print(f"Eval split: {eval_split}")
    out_filepath = os.path.join(experiment_out_dir, f"{eval_split}_metrics.csv")
    print(f"Saving results to {out_filepath} ....")
    overall_results_df = pd.DataFrame(overall_results)
    overall_results_df.to_csv(out_filepath, index=False)

    return model, overall_results_df
