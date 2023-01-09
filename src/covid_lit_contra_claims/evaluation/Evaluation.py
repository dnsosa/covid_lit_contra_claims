"""Collection of functions for evaluating results of training pipeline for covid_lit_contra_claims."""

# -*- coding: utf-8 -*-

import os

import numpy as np

import pandas as pd

import torch

from transformers import TextClassificationPipeline

from ..data.constants import ALL_CLAIMS_PATH, CLAIMS_SUBSET_PATH, SYNTHETIC_PREMISE_HCQ_PATH


def eval_model_pipeline(trained_model, tokenizer, out_dir, claims_set_id, SEED):
    """
    Make predictions using the trained model.

    :param trained_model: trained HuggingFace model
    :param tokenizer: HF tokenizer
    :param out_dir: output directory to write results
    :param claim_set_id: identifier for set of pairs to make NLI predictions
    :param SEED: random seed
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load claims data
    if claims_set_id == "synth_hcq":
        claims_df = pd.read_csv(SYNTHETIC_PREMISE_HCQ_PATH)
    elif claims_set_id == "claims_subset":
        claims_df = pd.read_csv(CLAIMS_SUBSET_PATH)
    elif claims_set_id == "all_claims":
        claims_df = pd.read_csv(ALL_CLAIMS_PATH)
    else:
        print(f"Invalid claims_set_id '{claims_set_id}', cannot load claims for inference. Exiting...")
        return None

    print("Moving trained model to CPU...")
    trained_model = trained_model.to('cpu')
    print("Model has been moved to CPU")

    pipe = TextClassificationPipeline(model=trained_model, tokenizer=tokenizer, top_k=None)

    # Need to format correctly for pipeline
    claim_pairs_tuples = [(c1, c2) for c1, c2 in list(zip(claims_df["text1"], claims_df["text2"]))]
    claims_pairs = [{'text': c1, 'text_pair': c2} for c1, c2 in claim_pairs_tuples]
    print("Created text classification pipeline and prepared claims pairs for inference.")

    print("Beginning inference...")
    pipe_preds = pipe(claims_pairs, padding=True, truncation=True)
    print("Predictions have been made!")

    pipe_preds_df = pd.concat([pd.DataFrame(d).pivot_table(columns='label', values='score') for d in pipe_preds])
    pipe_preds_df = pipe_preds_df.reset_index(drop=True)
    claims_df = claims_df.reset_index(drop=True)
    results = pd.concat([claims_df, pipe_preds_df], axis=1)

    out_path = os.path.join(out_dir, f"{claims_set_id}_preds_df.tsv")
    results.to_csv(out_path, sep='\t', index=False)

    return results
