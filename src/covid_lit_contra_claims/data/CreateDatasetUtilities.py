"""
Additional utility functions for the CreateDataset script.
"""

# -*- coding: utf-8 -*-

import pandas as pd
import xml.etree.ElementTree as ET

from itertools import combinations
from sklearn.model_selection import train_test_split


def generate_mancon_pandas_dfs(xml_path: str, neutral_frac: float, mancon_train_frac: float, SEED: int):
    """
    Create sentence pairs dataset from the original xml ManCon Corpus.

    :param xml_path: path to xml corpus
    :param neutral_frac: sample the neutrals to have size (neutral_frac * size_of_next_biggest_class), if None don't downsample
    :param mancon_train_frac: the fraction of questions in ManCon that should devoted to just training
    :param SEED: random seed
    """

    xtree = ET.parse(xml_path)  # TODO: Fix error # noqa: S314
    xroot = xtree.getroot()

    manconcorpus_data = pd.DataFrame(columns=['claim', 'assertion', 'question'])
    manconcorpus_df_list = []

    for node in xroot:
        for claim in node.findall('CLAIM'):
            manconcorpus_df_list.append({'question': claim.attrib.get('QUESTION'),
                                         'claim': claim.text,
                                         'assertion': claim.attrib.get('ASSERTION')})

    manconcorpus_data = pd.DataFrame(manconcorpus_df_list)

    questions = list(set(manconcorpus_data.question))
    train_qs, valtest_qs = train_test_split(questions, test_size=( 1 -mancon_train_frac), shuffle=True, random_state=SEED)
    val_qs, test_qs = train_test_split(valtest_qs, test_size=0.5, shuffle=True, random_state=SEED)

    mancon_nli_df_dict = {}
    splits = ["train", "val", "test"]

    for split_i, questions in enumerate([train_qs, val_qs, test_qs]):
        con_list, ent_list, neu_list = [], [], []

        prev_qs = []
        for q in questions:
            # Keep track of previous Q's
            prev_qs.append(q)

            # Find the yes and nos for this Q
            claim_yes = pd.DataFrame(manconcorpus_data.loc[(manconcorpus_data.question == q) &
                                                           (manconcorpus_data.assertion == 'YS'), 'claim'])  # noqa: W503
            claim_no = pd.DataFrame(manconcorpus_data.loc[(manconcorpus_data.question == q) &
                                                          (manconcorpus_data.assertion == 'NO'), 'claim'])  # noqa: W503

            # Create the contras
            temp = claim_yes.assign(key=1).merge(claim_no.assign(key=1), on='key').drop(columns='key')
            temp = temp.rename(columns={'claim_x': 'sentence_1', 'claim_y': 'sentence_2'})
            temp['label'] = 'contradiction'
            con_list += temp.drop_duplicates().values.tolist()

            # Create the entails
            for i, j in list(combinations(claim_yes.index, 2)):
                ent_list.append([claim_yes.claim[i], claim_yes.claim[j], 'entailment'])
            for i, j in list(combinations(claim_no.index, 2)):
                ent_list.append([claim_no.claim[i], claim_no.claim[j], 'entailment'])

            # Create the neutrals
            claims_from_q = pd.DataFrame(manconcorpus_data.loc[(manconcorpus_data.question == q), 'claim'])
            claims_from_other_qs = pd.DataFrame(
                manconcorpus_data.loc[~(manconcorpus_data.question.isin(prev_qs)), 'claim'])
            temp = claims_from_q.assign(key=1).merge(claims_from_other_qs.assign(key=1), on='key').drop(columns='key')
            temp = temp.rename(columns={'claim_x': 'sentence_1', 'claim_y': 'sentence_2'})
            temp['label'] = 'neutral'
            neu_list += temp.drop_duplicates().values.tolist()

        mancon_nli_df = pd.concat([pd.DataFrame(con_list), pd.DataFrame(ent_list), pd.DataFrame(neu_list)]).reset_index(
            drop=True)
        mancon_nli_df.columns = ["sentence1", "sentence2", "label"]

        if neutral_frac is not None:
            mancon_con_ent_df = mancon_nli_df[mancon_nli_df.label.isin(['entailment', 'contradiction'])]
            count_by_label = mancon_nli_df.groupby('label').size()
            max_non_neu = max(count_by_label['entailment'], count_by_label['contradiction'])
            mancon_neu_down_df = mancon_nli_df[mancon_nli_df.label == 'neutral'].sample(
                n=round(max_non_neu * neutral_frac))
            mancon_nli_df = pd.concat([mancon_con_ent_df, mancon_neu_down_df])

        mancon_nli_df_dict[splits[split_i]] = mancon_nli_df.sample(frac=1, random_state=SEED)

    return mancon_nli_df_dict

