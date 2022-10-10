# -*- coding: utf-8 -*-

"""Constants for data in covid_lit_contra_claims."""

import os

HERE = os.path.abspath(os.path.dirname(__file__))
IN_DATA_DIR = os.path.join(HERE, '../../../input')

MEDNLI_TRAIN_PATH = os.path.join(IN_DATA_DIR, 'input/mednli/mli_train_v1.jsonl')
MEDNLI_DEV_PATH = os.path.join(IN_DATA_DIR, 'input/mednli/mli_dev_v1.jsonl')
MEDNLI_TEST_PATH = os.path.join(IN_DATA_DIR, 'input/mednli/mli_test_v1.jsonl')
MANCON_XML_PATH = os.path.join(IN_DATA_DIR, 'input/manconcorpus/ManConCorpus.xml')
ROAM_PATH = os.path.join(IN_DATA_DIR, 'input/cord-training/Roam_annotations_trainvaltest_split_V2.xlsx')
