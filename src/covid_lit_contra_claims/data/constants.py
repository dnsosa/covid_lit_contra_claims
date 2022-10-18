# -*- coding: utf-8 -*-

"""Constants for data in covid_lit_contra_claims."""

import os

HERE = os.path.abspath(os.path.dirname(__file__))
IN_DATA_DIR = os.path.join(HERE, '../../../input')

MEDNLI_TRAIN_PATH = os.path.join(IN_DATA_DIR, 'input/mednli/mli_train_v1.jsonl')
MEDNLI_DEV_PATH = os.path.join(IN_DATA_DIR, 'input/mednli/mli_dev_v1.jsonl')
MEDNLI_TEST_PATH = os.path.join(IN_DATA_DIR, 'input/mednli/mli_test_v1.jsonl')
MANCON_XML_PATH = os.path.join(IN_DATA_DIR, 'input/manconcorpus/ManConCorpus.xml')
ROAM_SEP_PATH = os.path.join(IN_DATA_DIR, 'input/cord-training/Roam_annotations_trainvaltest_split_V2.xlsx')
# TODO: Find me
ROAM_ALL_PATH = os.path.join(IN_DATA_DIR, '')

MANCON_NEUTRAL_FRAC = 1
MANCON_TRAIN_FRAC = 0.67
WANDB_LOG_INTERVAL = 10
WANDB_LOG_FREQ = 100

model_id_mapper = {"biobert": "dmis-lab/biobert-base-cased-v1.2",
                   "bioclinbert": "emilyalsentzer/Bio_ClinicalBERT",
                   "scibert": "allenai/scibert_scivocab_uncased",
                   "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                   "roberta": "roberta-base"}
