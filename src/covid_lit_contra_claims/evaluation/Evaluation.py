"""
Collection of functions for evaluating results of training pipeline for covid_lit_contra_claims.
"""

# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd
import torch

from ..data.constants import ALL_CLAIMS_PATH
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, TextClassificationPipeline


# TODO create final report for testing purposes
def generate_report(trained_model, evaluation_dict, out_dir):
    """
    Generate a final test report using trained model against every evaluation dataset in the evaluation dictionary.

    :param trained_model: trained HF model
    :param evaluation_dict: dictionary of Datasets to use for evaluation
    :param out_dir: output directory to write results
    """

    pass


def eval_model(trained_model, tokenizer, all_pairs_dataset_dict, config, out_dir, SEED):
    """
    Make predictions using the trained model.

    :param trained_model: trained HuggingFace model
    :param tokenizer: HF tokenizer
    :param all_pairs_dataset_dict: Dataset dictionary containing all claim pairs about which to make predictions
    :param out_dir: output directory to write results
    :param SEED: random seed
    """
    # Set random seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Find the right device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    trained_model.to(device)
    print(f"Using device {device}.")

    # Create data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Make predictions with trained model
    # Print predictions and labels
    trained_model.eval()

    for split_id, all_pairs_dataset in all_pairs_dataset_dict.items():
        val_dataloader = DataLoader(all_pairs_dataset,
                                    shuffle=False,
                                    batch_size=config['batch_size'],
                                    collate_fn=data_collator)

        print("Beginning predictions for all claim pairs.")

        for batch_idx, batch in enumerate(val_dataloader):
            if batch_idx % 100 == 0:
                print(f"Batch #{batch_idx}...")
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = trained_model(**batch)
            logits = outputs.logits
            predictions = trained_model.argmax(logits, dim=-1)

    # Print compile it all together into one.
    # TODO

    # Save to out_dir file.
    # TODO

    return None


def eval_model_pipeline(trained_model, tokenizer, out_dir, SEED):
    """
    Make predictions using the trained model.

    :param trained_model: trained HuggingFace model
    :param tokenizer: HF tokenizer
    :param out_dir: output directory to write results
    :param SEED: random seed
    """

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load claims data
    claims_df = pd.read_csv(ALL_CLAIMS_PATH)

    trained_model = trained_model.to('cpu')

    pipe = TextClassificationPipeline(model=trained_model, tokenizer=tokenizer, top_k=None)
    claims_pairs = [{'text': [c1, c2]} for c1, c2 in list(zip(claims_df["text1"], claims_df["text2"]))]

    pipe_preds = pipe(claims_pairs, padding=True, truncation=True)
    pipe_preds_df = pd.concat([pd.DataFrame(d).pivot_table(columns='label', values='score') for d in pipe_preds])
    pipe_preds_df = pipe_preds_df.reset_index(drop=True).reset_index(drop=True)
    results = pd.concat([claims_df, pipe_preds_df])

    out_path = os.path.join(out_dir, "all_claims_with_predictions_df.tsv")
    results.to_csv(out_path, sep='\t', index=False)

    return results
