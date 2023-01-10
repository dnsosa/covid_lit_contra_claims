"""Collection of utility functions for helping visualizing results of training pipeline for covid_lit_contra_claims."""

# -*- coding: utf-8 -*-

import os

from matplotlib.lines import Line2D

import numpy as np

import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

fig_dir = "/Users/dnsosa/Desktop/AltmanLab/covid_lit_contra_claims/output/figs"


def train_seq_to_idx(row, R=4):
    """
    Convert training subsequence into an index.

    :param row: DF row
    :param R: total number of datasets used in training
    :return: index of the current subsequence
    """
    # Only takes into account the number of overlaps, not the order
    full_seq, train_subseq = row["train_datasets"], row["Cumulative Training"]

    full_seq_set = set(full_seq.split('_'))
    train_subseq_set = set(train_subseq.split('_'))
    k = len(full_seq_set.intersection(train_subseq_set))
    N = len(full_seq_set)
    return R - N + k


def latest_training(row):
    """Retrieve latest dataset used in training."""
    full_seq = row["Cumulative Training"]
    datasets = full_seq.split("_")
    return datasets[-1]


def tot_num_training_ds(row):
    """Calculate total number of training datasets used."""
    full_seq = row["train_datasets"]
    datasets = full_seq.split("_")
    return len(datasets)


def create_df_from_dir(base_dir, experiment_dir, model_name):
    """
    Create dataframe from directory containing all the results.

    :param base_dir: Base directory
    :param experiment_dir: Directory containing the name of the experiment
    :param model_name: Name of the model evaluating
    :return: Single dataframe compiling all results together
    """
    query_dir = os.path.join(base_dir, experiment_dir, model_name)

    df_list = []
    for root, _, f_names in os.walk(query_dir):
        for f in f_names:
            res_df = pd.read_csv(os.path.join(root, f))
            df_list.append(res_df)
    df = pd.concat(df_list)
    df["cum_train_idx"] = df.apply(lambda row: train_seq_to_idx(row), axis=1)
    df["Latest Training"] = df.apply(lambda row: latest_training(row), axis=1)
    df["tot_num_training_ds"] = df.apply(lambda row: tot_num_training_ds(row), axis=1)

    return df


def lighten_color(in_hex, lighten_percent):
    """Lighten a hex code color by a certain percentage."""
    in_hex = in_hex.lstrip('#')
    rgb = tuple(int(in_hex[i:i + 2], 16) for i in (0, 2, 4))
    light_r, light_g, light_b = [round(val + (255 - val) * lighten_percent) for val in rgb]
    light_hex = f"#{light_r:X}{light_g:X}{light_b:X}"

    return light_hex


def draw_arrows(df, metric, arrow_prox, arrow_color, highlight_forward_back, in_ax):
    """
    Draw arrows between pie points.

    :param df: Input dataframe
    :param metric: Metric we're plotting
    :param arrow_prox: Proximity (as a fraction of full distance) between the arrow head and tail and the point
    :param arrow_color: Arrow color
    :param highlight_forward_back: Highlight forward and backward sequences?
    :param in_ax: Input matplotlib axis
    """
    forward, backward = "multinli_mednli_mancon_roam", "roam_mancon_mednli_multinli"
    full_seq = (df["train_datasets"].values[0])
    line_width = 4 if ((full_seq in [forward, backward]) and highlight_forward_back) else 1
    if highlight_forward_back:
        if full_seq == forward:
            arrow_color = "#9dbda5"
        elif full_seq == backward:
            arrow_color = "#f0adad"

    for i in range(min(df.cum_train_idx), max(df.cum_train_idx)):
        x1, x2 = i, i + 1
        x1_prox = x2 - (x2 - x1) * arrow_prox
        x2_prox = x1 + (x2 - x1) * arrow_prox
        y1, y2 = df[df["cum_train_idx"] == i][metric].values[0], df[df["cum_train_idx"] == i + 1][metric].values[0]
        y1_prox = y2 - (y2 - y1) * arrow_prox
        y2_prox = y1 + (y2 - y1) * arrow_prox
        in_ax.annotate("", xy=(x1_prox, y1_prox), xycoords='data', xytext=(x2_prox, y2_prox), textcoords='data',
                       arrowprops={'arrowstyle': '<|-, head_width=0.3, head_length=.45',
                                   'connectionstyle': 'arc3',
                                   'facecolor': arrow_color,
                                   'edgecolor': arrow_color,
                                   'lw': line_width})

    return None


def calculate_data_ratios_dist(train_datasets_list, data_ratio):
    """
    Calculate data ratio distributions for pie seq plotting.

    :param train_datasets_list: list of training datasets
    :param data_ratio: input data ratio
    :return: the numerator describing the proportion between datasets
    """
    multiplier_datasets = ["multinli", "mednli", "mancon"]
    # Number of times to apply the multiplier
    ratio_multiplier = len(set(multiplier_datasets).intersection(set(train_datasets_list)))
    n_datasets = len(train_datasets_list)
    dist = np.ones(n_datasets)
    for i, dataset in enumerate(train_datasets_list):
        if dataset in multiplier_datasets:
            dist[i] = data_ratio ** ratio_multiplier
            ratio_multiplier -= 1

    return dist


def draw_pie_scatter(df, metric, ds_color_map, background_color, combined_color, size, alpha, in_ax, is_combined=False):
    """
    Draw pie scatter points.

    :param df: input dataframe
    :param metric: netric to visualize
    :param ds_color_map: map of datasets to colors
    :param background_color: background color of plot
    :param combined_color: color of the "combined" condition highlight
    :param size: size of the circles
    :param alpha: transparency
    :param in_ax: input
    :param is_combined: if True, figure out new way to calculate the data ratio proportions
    """
    num_ft_datasets = max(df.cum_train_idx)
    lowest_train_idx = min(df.cum_train_idx)
    for i in range(lowest_train_idx, num_ft_datasets + 1):
        curr_df = df[df["cum_train_idx"] == i]
        train_so_far = curr_df["Cumulative Training"].values[0]
        all_train_datasets = curr_df["train_datasets"].values[0].split("_")

        xpos, ypos = i, curr_df[metric].values[0]

        # for incremental pie slices
        data_ratio = curr_df["data_ratios"].values[0]
        dist = calculate_data_ratios_dist(all_train_datasets, data_ratio)
        # if is_combined:  # TODO: Fix me if different data ratios
        #    dist = np.ones(4)
        size_boost = 1 + np.log10(np.sum(dist)) / np.log10(11)
        size *= size_boost
        cumsum = np.cumsum(dist)
        cumsum = cumsum / cumsum[-1]
        pie = [0] + cumsum.tolist()

        for j, (r1, r2) in enumerate(zip(pie[:-1], pie[1:])):

            train_j = all_train_datasets[j]
            if train_j in train_so_far:
                color = ds_color_map[train_j]
            else:
                color = background_color

            angles = np.pi / 2 - np.linspace(2 * np.pi * r1, 2 * np.pi * r2, num=25)

            x = [0] + np.cos(angles).tolist()
            y = [0] + np.sin(angles).tolist()

            xy = np.column_stack([x, y])

            if is_combined:
                in_ax.scatter([xpos], [ypos], marker=xy, s=size * 5.5, color=combined_color, zorder=9,
                              alpha=alpha)  # Drawing the border
                in_ax.scatter([xpos], [ypos], marker=xy, s=size * 3, color=color, zorder=10, alpha=alpha)

            else:
                in_ax.scatter([xpos], [ypos], marker=xy, s=size * 1.1, color="black", alpha=alpha)  # Drawing the border
                in_ax.scatter([xpos], [ypos], marker=xy, s=size, color=color, alpha=alpha)

    return None


def draw_custom_legends(ds_color_map, ds_name_map, background_color, combined_color, loc, in_ax, show_combined=False):
    """
    Draw legends.

    :param ds_color_map: dataset to color map
    :param ds_name_map: dataset id to name to display map
    :param background_color: background color for plots
    :param combined_color: color of combined condition
    :param loc: location of legend
    :param in_ax: input axis
    :param show_combined: show combined condition?
    """
    # Train legend
    ds_names = list(ds_color_map.keys())[:4]
    ds_names_nice = [ds_name_map[ds] for ds in ds_names]
    custom_lines = [
        Line2D([0], [0], color='white', marker='o', markerfacecolor=ds_color_map[ds], markeredgecolor=background_color,
               markersize=15) for ds in ds_names]

    if show_combined:
        combined_line = Line2D([0], [0], color='white', marker='o', markerfacecolor="white",
                               markeredgecolor=combined_color, markersize=19)
        custom_lines.append(combined_line)
        ds_names_nice.append("Combined")

    in_ax.legend(custom_lines, ds_names_nice,
                 loc=loc)

    return None


def get_seq_plot_order(in_df, metric):
    """
    Return order to plot each subsequence.

    :param in_df: input df of all conditions
    :param metric: metric to visualize
    :return: ordered list of subsequences
    """
    seqs = set(in_df["train_datasets"])
    num_ds_in_seq = {seq: len(seq.split("_")) for seq in seqs}

    # Order by performance metric
    seq_final_metrics = dict(in_df.groupby("train_datasets")[metric].max())
    ordered_final_metrics = {k: v for k, v in
                             sorted(seq_final_metrics.items(), key=lambda item: (-num_ds_in_seq[item[0]], item[1]))}
    return list(ordered_final_metrics.keys())
