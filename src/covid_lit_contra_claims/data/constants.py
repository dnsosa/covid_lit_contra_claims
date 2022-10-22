# -*- coding: utf-8 -*-

"""Constants for data in covid_lit_contra_claims."""

import os

# Paths to input data
HERE = os.path.abspath(os.path.dirname(__file__))
IN_DATA_DIR = os.path.join(HERE, '../../../input')

MEDNLI_TRAIN_PATH = os.path.join(IN_DATA_DIR, 'mednli/mli_train_v1.jsonl')
MEDNLI_DEV_PATH = os.path.join(IN_DATA_DIR, 'mednli/mli_dev_v1.jsonl')
MEDNLI_TEST_PATH = os.path.join(IN_DATA_DIR, 'mednli/mli_test_v1.jsonl')
MANCON_XML_PATH = os.path.join(IN_DATA_DIR, 'manconcorpus/ManConCorpus.xml')
ROAM_SEP_PATH = os.path.join(IN_DATA_DIR, 'cord-training/Roam_annotations_trainvaltest_split_V2.xlsx')
ROAM_ALL_PATH = os.path.join(IN_DATA_DIR, 'Coronawhy-Contra-Claims-Scaling-v2-annotated-2020-10-21.xlsx')

# Parameters for creating input corpora
MANCON_NEUTRAL_FRAC = 1
MANCON_TRAIN_FRAC = 0.67
DEFAULT_TRAIN_FRAC = 0.75
ROAM_FULL_TRAIN_FRAC = DEFAULT_TRAIN_FRAC
DD_TRAIN_FRAC = DEFAULT_TRAIN_FRAC
DD_PH_TRAIN_FRAC = DEFAULT_TRAIN_FRAC
PH_TRAIN_FRAC = DEFAULT_TRAIN_FRAC
N_DD = 400

# Mapper from model/tokenizer identifier to HF location
model_id_mapper = {"biobert": "dmis-lab/biobert-base-cased-v1.2",
                   "bioclinbert": "emilyalsentzer/Bio_ClinicalBERT",
                   "scibert": "allenai/scibert_scivocab_uncased",
                   "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                   "roberta": "roberta-base"}

# Parameters for training
WANDB_LOG_INTERVAL = 10
WANDB_LOG_FREQ = 100
