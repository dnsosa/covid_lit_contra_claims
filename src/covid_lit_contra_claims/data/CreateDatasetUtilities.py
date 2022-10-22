"""
Additional utility functions for the CreateDataset script.
"""

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from itertools import combinations
from sklearn.model_selection import train_test_split

from .constants import DD_TRAIN_FRAC, DD_PH_TRAIN_FRAC, N_DD, PH_TRAIN_FRAC, ROAM_FULL_TRAIN_FRAC

DRUG_LIST = ["hydroxychloroquine", " chloroquine", "tocilizumab", "remdesivir", "vitamin d", "lopinavir",
             "dexamethasone"]


def generate_mancon_pandas_dfs(xml_path: str, neutral_frac: float, mancon_train_frac: float, SEED: int):
    """
    Create sentence pairs dataset from the original xml ManCon Corpus.

    :param xml_path: path to xml corpus
    :param neutral_frac: sample the neutrals to have size (neutral_frac * size_of_next_biggest_class), if None don't downsample
    :param mancon_train_frac: the fraction of questions in ManCon that should devoted to just training
    :param SEED: random seed
    :return: ManCon
    """

    xtree = ET.parse(xml_path)  # TODO: Fix error # noqa: S314
    xroot = xtree.getroot()

    # manconcorpus_data = pd.DataFrame(columns=['claim', 'assertion', 'question'])
    manconcorpus_df_list = []

    for node in xroot:
        for claim in node.findall('CLAIM'):
            manconcorpus_df_list.append({'question': claim.attrib.get('QUESTION'),
                                         'claim': claim.text,
                                         'assertion': claim.attrib.get('ASSERTION')})

    manconcorpus_data = pd.DataFrame(manconcorpus_df_list)

    questions = list(set(manconcorpus_data.question))
    train_qs, valtest_qs = train_test_split(questions, test_size=(1 -mancon_train_frac), shuffle=True, random_state=SEED)
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
            n_neutral = round(max_non_neu * neutral_frac)
            mancon_neu_down_df = mancon_nli_df[mancon_nli_df.label == 'neutral'].sample(n=n_neutral)
            mancon_nli_df = pd.concat([mancon_con_ent_df, mancon_neu_down_df])

        mancon_nli_df_dict[splits[split_i]] = mancon_nli_df.sample(frac=1, random_state=SEED)

    return mancon_nli_df_dict


def load_roam_full_data(roam_path: str):
    """
    Load and preprocess full Roam dataset.

    :param roam_path: path to Roam dataset
    :return: Preprocessed DataFrame of claim pairs with labels
    """
    roam_data = pd.read_excel(roam_path, sheet_name="Docs")

    def process_labels(s):
        s = s.replace("STRICT_", "")
        return s.lower()

    def splitter(in_str: str, index: int):
        text = in_str.rstrip().split("\n\n")[index]
        return text

    roam_data = roam_data.dropna()
    roam_data['label'] = roam_data['tags'].apply(process_labels)
    # remove "question" and "duplicate" e.g.
    roam_data = roam_data[roam_data.label.isin(["contradiction", "neutral", "entailment"])]
    roam_data["claim1"] = roam_data.text.transform(lambda x: splitter(x, 1))
    roam_data["claim2"] = roam_data.text.transform(lambda x: splitter(x, 3))
    return roam_data


def get_claims_with_drugs_df(roam_df: pd.DataFrame):
    """
    Return a DataFrame of every unique claim in Roam dataset with corresponding identified drug mentions.

    :param roam_df: input DataFrame
    :return: DataFrame of claims with drug mentions
    """
    def identify_drugs_mentioned(claim):
        found_drugs = []
        for drug in DRUG_LIST:
            if drug in claim:
                found_drugs.append(drug)

        return found_drugs

    roam_df["claim1_drugs"] = roam_df['claim1'].apply(identify_drugs_mentioned)
    roam_df["claim2_drugs"] = roam_df['claim2'].apply(identify_drugs_mentioned)
    roam_claims = pd.concat([pd.DataFrame({"claim": roam_df.claim1, "claim_drugs": roam_df.claim1_drugs}),
                             pd.DataFrame({"claim": roam_df.claim2, "claim_drugs": roam_df.claim2_drugs})])

    return roam_claims


def split_df_into_tvt_dict(df: pd.DataFrame, train_frac: float, SEED: int):
    """
    Split a dataframe into train/val/test.

    :param df: input DataFrame
    :param train_frac: fraction of the DataFrame to be used for training. The rest will be split 50/50 val/test
    :param SEED: random seed
    :return: dictionary of DataFrames representing the 3 splits
    """
    train_df, valtest_df = train_test_split(df, test_size=(1 - train_frac), shuffle=True, random_state=SEED)
    val_df, test_df = train_test_split(valtest_df, test_size=0.5, shuffle=True, random_state=SEED)
    tvt_df_dict = {"train": train_df.reset_index(drop=True),
                   "val": val_df.reset_index(drop=True),
                   "test": test_df.reset_index(drop=True)}

    return tvt_df_dict


def generate_roam_full_pandas_dfs(roam_path: str, SEED: int, roam_full_train_frac: float = ROAM_FULL_TRAIN_FRAC):
    """
    Generate the full Roam dataset (without splitting into disjoint subnetworks).

    :param roam_path: path to Roam data
    :param SEED: random seed
    :param roam_full_train_frac: fraction to be used for training
    :return: dict of DataFrames of Roam annotated sentence pairs
    """
    roam_full_data = load_roam_full_data(roam_path)
    roam_full_data = pd.DataFrame({"sentence1": roam_full_data.claim1,
                                   "sentence2": roam_full_data.claim2,
                                   "labels": roam_full_data.label})
    roam_full_df_dict = split_df_into_tvt_dict(roam_full_data, roam_full_train_frac, SEED)

    return roam_full_df_dict


def generate_roam_ph_pandas_dfs(roam_path: str, SEED: int, ph_train_frac: float = PH_TRAIN_FRAC):
    """
    Generate the Equal Premise Hypothesis Roam DF.

    :param roam_path: path to Roam data
    :param SEED: random seed
    :param ph_train_frac: fraction to be used for training
    :return: dict of DataFrames of sentence pairs where the two sentences are identical = Entailment
    """
    roam_data = load_roam_full_data(roam_path)
    roam_all_claims = set(roam_data.claim1).union(roam_data.claim2)
    n_claims = len(roam_all_claims)
    # Shuffle all the claims
    selected_claims = np.random.choice(list(roam_all_claims), size=n_claims, replace=False)

    # Make the premise and hypothesis be identical
    roam_ph = pd.DataFrame({"sentence1": selected_claims,
                            "sentence2": selected_claims,
                            "labels": "entailment"})
    roam_ph_df_dict = split_df_into_tvt_dict(roam_ph, ph_train_frac, SEED)

    return roam_ph_df_dict


def generate_roam_dd_pandas_dfs(roam_path: str, SEED: int, n_dd: int = N_DD, dd_train_frac: float = DD_TRAIN_FRAC):
    """
    Generate the Different Drug Roam DF.

    :param roam_path: path to Roam data
    :param SEED: random seed
    :param n_dd: number of samples to generate
    :param dd_train_frac: fraction to be used for training
    :return: dict of DataFrames of sentence pairs where the two claims have no overlapping drug mentions = Neutral
    """
    roam_data = load_roam_full_data(roam_path)
    roam_claims = get_claims_with_drugs_df(roam_data)
    roam_claims["n_drugs"] = roam_claims["claim_drugs"].apply(lambda x: len(x))
    roam_claims["claim_drugs"] = roam_claims["claim_drugs"].apply(lambda x: tuple(x))
    roam_claims_drugs_dict = dict(zip(roam_claims.claim, roam_claims.claim_drugs))

    # Select a sample of single claims (with replacement). For each claim, naively sample another claim until
    # a pair is found where there's no drug in common
    all_claims = list(roam_claims_drugs_dict.keys())
    first_claims = np.random.choice(all_claims, size=n_dd, replace=True)  # TODO: make sure seed is set
    claim_pairs = []
    for claim in first_claims:
        claim_drugs = set(roam_claims_drugs_dict[claim])
        found_compatible_claim = False
        while not found_compatible_claim:
            candidate_claim = np.random.choice(all_claims)
            candidate_claim_drugs = set(roam_claims_drugs_dict[candidate_claim])
            if len(claim_drugs.intersection(candidate_claim_drugs)) == 0:
                found_compatible_claim = True
        claim_pairs.append([claim, candidate_claim])

    roam_dd = pd.DataFrame(claim_pairs, columns=["sentence1", "sentence2"])
    roam_dd["labels"] = "neutral"
    roam_dd_df_dict = split_df_into_tvt_dict(roam_dd, dd_train_frac, SEED)

    return roam_dd_df_dict


def generate_roam_dd_ph_pandas_dfs(roam_path: str, SEED: int, dd_ph_train_frac: float = DD_PH_TRAIN_FRAC):
    """
    Generate the Different Drug-Equal Premise Hypothesis Roam DF.

    :param roam_path: path to Roam data
    :param SEED: random seed
    :param dd_ph_train_frac: fraction to be used for training
    :return: dict of DataFrames of sent. pairs that are identical except for the (single) drug mention = Neutral
    """
    roam_data = load_roam_full_data(roam_path)
    roam_claims = get_claims_with_drugs_df(roam_data)

    # Swap drugs of claims about a single drug
    roam_claims["n_drugs"] = roam_claims["claim_drugs"].apply(lambda x: len(x))
    roam_claims["claim_drugs"] = roam_claims["claim_drugs"].apply(lambda x: tuple(x))
    roam_claims = roam_claims[roam_claims.n_drugs == 1].drop_duplicates()

    def drug_swap(row):
        c_drug = row["claim_drugs"][0].strip()
        other_drugs = [drug for drug in DRUG_LIST if drug != c_drug]
        new_drug = np.random.choice(other_drugs)

        return row["claim"].replace(c_drug, new_drug)

    roam_claims["swapped_claim1"] = roam_claims.apply(drug_swap, axis=1)
    roam_claims["swapped_claim2"] = roam_claims.apply(drug_swap, axis=1)

    # TODO: Thought,...
    # TODO: On multiple drugs mentioned at the same time. What would the swap procedure be?
    # TODO: Replace the names of all the drugs with the resampled drug (or drugs?)?

    # Mixing up the original and perturbed sentences so the original isn't always the premise
    roam_dd_ph = pd.concat([pd.DataFrame({"sentence1": roam_claims.claim, "sentence2": roam_claims.swapped_claim1}),
                            pd.DataFrame({"sentence1": roam_claims.swapped_claim2, "sentence2": roam_claims.claim})])
    roam_dd_ph = roam_dd_ph.sample(frac=1, random_state=SEED)
    roam_dd_ph["labels"] = "neutral"
    roam_dd_ph_df_dict = split_df_into_tvt_dict(roam_dd_ph, dd_ph_train_frac, SEED)

    return roam_dd_ph_df_dict

