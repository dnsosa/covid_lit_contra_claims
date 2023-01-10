"""Collection of functions for visualizing results of training pipeline for covid_lit_contra_claims."""

# -*- coding: utf-8 -*-

import os
from collections import OrderedDict

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import numpy as np

import pandas as pd

import seaborn as sns

from .VisualizationUtilities import calculate_data_ratios_dist, create_df_from_dir, draw_arrows, draw_custom_legends, \
    draw_pie_scatter, get_seq_plot_order, lighten_color

# Matplotlib parameters
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "normal",
})

SMALL_SIZE = 24
MEDIUM_SIZE = 35
BIGGER_SIZE = 40

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

fig_dir = "/Users/dnsosa/Desktop/AltmanLab/covid_lit_contra_claims/output/figs"


def draw_pie_seq_plot(root, experiment, model, metrics_to_vis, val_sets, learning_rates, arrow_prox=0.78,
                      init_pie_size=600, ymin=0.1, ymax=0.9, highlight_forward_back=False, do_save=False):
    """
    Draw pie sequence plot.

    :param root: root directory
    :param experiment: name of experiment (directory)
    :param model: name of model
    :param metrics_to_vis: metric to visualize
    :param val_sets: validation datasets
    :param learning_rates: learning rates
    :param arrow_prox: arrow proximity to center of the pie points as a fraction
    :param init_pie_size: size of each pie
    :param ymin: min y value in plot
    :param ymax: max y value
    :param highlight_forward_back: highlight the for-back seqs?
    :param do_save: save the plot?
    """
    # Color and name parameters
    dataset_color_map = OrderedDict(
        {"multinli": "#D81B74", "mednli": "#FFC107", "mancon": "#52CC62", "roam": "#1E88E5", "roamDD": "#1E88E5",
         "roamPH": "#1E88E5", "roamDDPH": "#1E88E5"})
    dataset_name_map = OrderedDict(
        {"multinli": "MultiNLI", "mednli": "MedNLI", "mancon": "ManCon", "roam": "Covid19NLI", "roamDD": "RoamDD",
         "roamPH": "RoamPH", "roamDDPH": "RoamDDPH"})
    model_name_map = OrderedDict(
        {"roberta": "RoBERTa", "pubmedbert": "PubMedBERT", "biobert": "BioBERT", "scibert": "SciBERT",
         "bioclinbert": "BioClinBERT"})
    background_color = "#adacac"
    combined_color = "#964B00"
    arrow_color = '#cccaca'
    cols_to_view = ["Cumulative Training", "train_prep_experiment", "data_ratios", "recall_con", "f1"]

    # Other aesthetic params
    pie_size = init_pie_size
    arrow_prox = arrow_prox
    dpi = 300
    pie_alpha = 1
    legend_loc = "upper left"

    # Nice names for metrics, model
    metrics = OrderedDict({"accuracy": "Accuracy",
                           "f1": "Macro F1",
                           "precision": "Precision",
                           "recall": "Recall",
                           "recall_con": "Contradictions Recall"})

    df = create_df_from_dir(root, experiment, model)

    for lr in learning_rates:
        # Get the dataframe for each learning rate
        sub_df = df[df["learning_rate"] == lr]

        # Plot a row of figures, one per val set
        for val_set in val_sets:

            val_set_official = dataset_name_map[val_set]

            # Create a row of figures, one per metric
            if len(metrics_to_vis) == 5:
                fig, axs = plt.subplots(1, len(metrics_to_vis), figsize=(28, 6))
            else:
                fig, axs = plt.subplots(1, len(metrics_to_vis), figsize=(18, 11))
            fig.tight_layout(pad=3.0)
            fig.suptitle(f'Model: {model_name_map[model]}, Validation Set: {val_set_official}, $lr$ = {lr}', y=1.07)

            # Create the subset of the DF for this val set
            in_df = sub_df[sub_df["Validation Set"] == val_set]
            val_color = lighten_color(dataset_color_map[val_set], 0.95)

            # Create a panel for each metric
            for i, metric in enumerate(metrics_to_vis):

                # Some parameters for each panel
                axs[i].yaxis.set_ticks(np.arange(.1, 1, .1))
                axs[i].xaxis.set_ticks(np.arange(1, 5))
                axs[i].set_xticklabels('')
                axs[i].set_ylim([ymin, ymax])
                axs[i].set_xlim([0.5, 4.5])
                axs[i].set_title(metrics[metric])
                axs[i].set_facecolor(val_color)
                axs[i].set_xlabel("Fine-Tuning Step")

                # If there are multiple data ratios present, need to iterate through both
                for data_ratio in set(in_df.data_ratios):

                    in_df_ratio = in_df[in_df.data_ratios == data_ratio]

                    # Plot individual sequences on each panel
                    for seq in get_seq_plot_order(in_df_ratio, metric):

                        # Filter in the subset of results relevant for this sequence
                        seq_df_all = in_df_ratio[in_df_ratio["train_datasets"] == seq]
                        seq_df_combined = seq_df_all[seq_df_all['train_prep_experiment'] == "combined"]
                        found_combined = len(seq_df_combined) > 0
                        seq_df = seq_df_all[seq_df_all['train_prep_experiment'] == "sequential"]
                        if i == 0:
                            print(f"Viewing seq {seq} statistics...")
                            print(seq_df_all[cols_to_view].head(10))
                            print("================================")
                            # if found_combined:
                            #    print(f"Viewing seq {seq} statistics...")
                            #    print(seq_df_combined[cols_to_view].head(10))
                            #    print("================================")

                        # Draw arrows between pies
                        draw_arrows(seq_df, metric=metric, arrow_prox=arrow_prox, arrow_color=arrow_color,
                                    highlight_forward_back=highlight_forward_back, in_ax=axs[i])
                        # Draw my custom pie scatter plot
                        draw_pie_scatter(seq_df, metric=metric, ds_color_map=dataset_color_map,
                                         background_color=background_color, combined_color=combined_color,
                                         size=pie_size, alpha=pie_alpha, in_ax=axs[i])
                        if found_combined:
                            draw_pie_scatter(seq_df_combined, metric=metric, ds_color_map=dataset_color_map,
                                             background_color=background_color, combined_color=combined_color,
                                             size=pie_size, is_combined=found_combined, alpha=pie_alpha, in_ax=axs[i])

                # Draw legend for the left-most panel
                combined_in_panel = "combined" in set(in_df.train_prep_experiment)
                if i == 0:
                    draw_custom_legends(ds_color_map=dataset_color_map, ds_name_map=dataset_name_map,
                                        background_color=background_color, combined_color=combined_color,
                                        loc=legend_loc, show_combined=combined_in_panel, in_ax=axs[i])

            # Display this row of figs
            if do_save:
                out_dir = os.path.join(fig_dir, experiment, model, f"Val_{val_set}_LR_{lr}")
                os.makedirs(out_dir, exist_ok=True)
                plt.savefig(os.path.join(out_dir, "pie_seq.png"), dpi=dpi, bbox_inches="tight")
            plt.show()

    return df, in_df


def draw_pie_scatter_model_comp(df, model_order, metric, ds_color_map, combined_color, size, in_ax,
                                is_combined=False):
    """
    Plot the pie scatter plot--a subroutine.

    :param df: input DF of results
    :param model_order: order to plot the models
    :param metric: metric to vis
    :param ds_color_map: dataset to color map
    :param combined_color: color of the combined condition
    :param size: pie size
    :param in_ax: input plot axis
    :param is_combined: need to adjust proportions in case of combined condition?
    """
    # num_ft_datasets = max(df.cum_train_idx)
    # lowest_train_idx = min(df.cum_train_idx)
    for i, model in enumerate(model_order):
        curr_df = df[df["model"] == model]
        # train_so_far = curr_df["Cumulative Training"].values[0]
        all_train_datasets = curr_df["train_datasets"].values[0].split("_")

        xpos, ypos = i, curr_df[metric].values[0]

        # for incremental pie slices
        data_ratio = curr_df["data_ratios"].values[0]
        dist = calculate_data_ratios_dist(all_train_datasets, data_ratio)
        size_boost = 1 + 2 * np.log2(np.sum(dist))
        # size *= size_boost
        cumsum = np.cumsum(dist)
        cumsum = cumsum / cumsum[-1]
        pie = [0] + cumsum.tolist()

        for j, (r1, r2) in enumerate(zip(pie[:-1], pie[1:])):

            train_j = all_train_datasets[j]
            color = ds_color_map[train_j]

            # if train_j in train_so_far:
            #    color = ds_color_map[train_j]
            # else:
            #    color = background_color

            angles = np.pi / 2 - np.linspace(2 * np.pi * r1, 2 * np.pi * r2, num=25)

            x = [0] + np.cos(angles).tolist()
            y = [0] + np.sin(angles).tolist()

            xy = np.column_stack([x, y])

            if is_combined:
                in_ax.scatter([xpos], [ypos], marker=xy, s=size * 2.25 * size_boost, color=combined_color,
                              zorder=9)  # Drawing the border
                in_ax.scatter([xpos], [ypos], marker=xy, s=size * size_boost, color=color, zorder=10)

            else:
                in_ax.scatter([xpos], [ypos], marker=xy, s=size * 1.1 * size_boost, color="black")  # Drawing the border
                in_ax.scatter([xpos], [ypos], marker=xy, s=size * size_boost, color=color)

    return None


def draw_pie_seq_plot_model_comp(df, model_order, metrics_to_vis, arrow_prox=0.78, init_pie_size=600,
                                 do_save=False):
    """
    Draw the full pie scatter plot for compairing model pretrainings.

    :param df: input DF with all conditions
    :param model_order: model order to plot
    :param metrics_to_vis: metrics to visualize
    :param arrow_prox: arrow proximity to head and tail
    :param init_pie_size: initial pie size
    :param do_save: save fig?
    """
    # Color and name parameters
    dataset_color_map = OrderedDict(
        {"multinli": "#D81B74", "mednli": "#FFC107", "mancon": "#52CC62", "roam": "#1E88E5", "roamDD": "#1E88E5",
         "roamPH": "#1E88E5", "roamDDPH": "#1E88E5"})
    dataset_name_map = OrderedDict(
        {"multinli": "MultiNLI", "mednli": "MedNLI", "mancon": "ManCon", "roam": "Covid19NLI", "roamDD": "RoamDD",
         "roamPH": "RoamPH", "roamDDPH": "RoamDDPH"})
    model_name_map = OrderedDict(
        {"roberta": "RoBERTa", "pubmedbert": "PubMedBERT", "biobert": "BioBERT", "scibert": "SciBERT",
         "bioclinbert": "BioClinBERT"})
    background_color = "#adacac"
    combined_color = "#964B00"

    # Other aesthetic params
    pie_size = init_pie_size
    legend_loc = "upper left"

    # Nice names for metrics, model
    metrics = OrderedDict({"accuracy": "Accuracy",
                           "f1": "Macro F1",
                           "precision": "Precision",
                           "recall": "Recall",
                           "recall_con": "Contradictions Recall"})

    # Set of metrics to visualize
    # metrics_to_vis = metrics.keys()

    for lr in set(df["learning_rate"]):
        # Get the dataframe for each learning rate
        sub_df = df[df["learning_rate"] == lr]

        # Plot a row of figures, one per val set
        for val_set, val_set_official in [("roam", "Roam")]:

            # Create a row of figures, one per metric
            if len(metrics_to_vis) == 5:
                fig, axs = plt.subplots(1, len(metrics_to_vis), figsize=(28, 6))
            else:
                fig, axs = plt.subplots(1, len(metrics_to_vis), figsize=(15, 10))
            fig.tight_layout(pad=3.0)
            fig.suptitle(f'Validation Set: {val_set_official}, $lr$ = {lr}', y=1.07)

            # Create the subset of the DF for this val set
            in_df = sub_df[sub_df["Validation Set"] == val_set]
            val_color = lighten_color(dataset_color_map[val_set], 0.95)

            # Create a panel for each metric
            for i, metric in enumerate(metrics_to_vis):

                # Some parameters for each panel
                axs[i].set_title(metrics[metric])
                axs[i].set_facecolor(val_color)
                axs[i].xaxis.set_ticks(np.arange(5))
                axs[i].set_xticklabels(list(map(model_name_map.get, model_order)), rotation=20, ha='right')
                # axs[i].yaxis.set_ticks(np.arange(.2,.8,.1))
                # axs[i].set_ylim([.25,.75])
                axs[i].yaxis.set_ticks(np.arange(0, 1.1, .1))
                axs[i].set_ylim([-.05, 1.075])
                axs[i].set_xlim([-.5, 4.5])

                # If there are multiple data ratios present, need to iterate through both
                for data_ratio in set(in_df.data_ratios):

                    in_df_ratio = in_df[in_df.data_ratios == data_ratio]

                    # Plot individual sequences on each panel
                    for seq in get_seq_plot_order(in_df_ratio, metric):

                        # Filter in the subset of results relevant for this sequence
                        seq_df_all = in_df_ratio[in_df_ratio["train_datasets"] == seq]
                        seq_df_combined = seq_df_all[seq_df_all['train_prep_experiment'] == "combined"]
                        found_combined = len(seq_df_combined) > 0
                        seq_df = seq_df_all[seq_df_all['train_prep_experiment'] == "sequential"]

                        # Draw my custom pie scatter plot
                        draw_pie_scatter_model_comp(seq_df, model_order=model_order, metric=metric,
                                                    ds_color_map=dataset_color_map, combined_color=combined_color,
                                                    size=pie_size, in_ax=axs[i])
                        if found_combined:
                            draw_pie_scatter_model_comp(seq_df_combined, model_order=model_order, metric=metric,
                                                        ds_color_map=dataset_color_map, combined_color=combined_color,
                                                        size=pie_size, is_combined=found_combined, in_ax=axs[i])
                        # Draw arrows between pies
                        # draw_arrows(seq_df, metric=metric, arrow_prox=.89, arrow_color=arrow_color,
                        # y_text_pad=.05, highlight_forward_back=highlight_forward_back, in_ax=axs[i])
                        # Draw legend for the left-most panel

                combined_in_panel = "combined" in set(in_df.train_prep_experiment)
                draw_custom_legends(ds_color_map=dataset_color_map, ds_name_map=dataset_name_map,
                                    background_color=background_color, combined_color=combined_color, loc=legend_loc,
                                    show_combined=combined_in_panel, in_ax=axs[i])

            # Display this row of figs
            if do_save:
                out_dir = os.path.join(fig_dir, "modelComp", f"Val_{val_set}_LR_{lr}")
                os.makedirs(out_dir, exist_ok=True)
                plt.savefig(os.path.join(out_dir, "pie_comp.png"), dpi=300, bbox_inches="tight")

            plt.show()

    return df, in_df


def draw_hp_tune_heatmaps(root, experiment, models, metric_name_map, model_name_map, do_save=False):
    """
    Draw hyperparameter optimization tune maps.

    :param root: root dir
    :param experiment: experiment name dir
    :param models: models to vis
    :param metric_name_map: metric name for plotting map
    :param model_name_map: model name for plotting map
    :param do_save: save fig?
    """
    for model in models:
        for metric, metric_nice_name in metric_name_map.items():
            for cum_training, val_set in [("none_multinli_mednli_mancon_roam", "roam")]:
                fig, ax = plt.subplots(figsize=(10, 5))
                df = create_df_from_dir(root, experiment, model)
                df = df[(df["Cumulative Training"] == cum_training) & (df["Validation Set"] == val_set)]
                wide_df = pd.pivot(df, index="batch_size", columns="learning_rate", values=metric)
                print(f"Cum training: {cum_training}")

                sns.heatmap(wide_df, cmap='RdYlGn', linewidths=3, annot=True, ax=ax)
                plt.title(f"{model_name_map[model]} - {metric_nice_name}\n")
                plt.xlabel("Learning Rate")
                plt.ylabel("Batch Size")

                # Now save
                if do_save:
                    out_dir = os.path.join(fig_dir, "HyperparamTune")
                    os.makedirs(out_dir, exist_ok=True)
                    plt.savefig(os.path.join(out_dir, f"HPTune_{model}_{metric}.png"), dpi=300, bbox_inches="tight")

                plt.show()

    return None


def plot_benchmark_fig(df,
                       benchmark_names,
                       dpi=300,
                       do_save=False):
    """
    Plot comparison with benchmarks.

    :param df: input DF with results
    :param benchmark_names: names of benchmarks
    :param dpi: fig resolution
    :param do_save: save?
    """
    # Overall metric values
    _ = plt.figure(figsize=(20, 10))
    width = 0.3
    v_space = .025
    h_space = .08
    x = np.arange(len(df))
    f1_name = "f1"
    f1_name_nice = "F1"
    f1_color = "blue"
    con_rec_name = "con. recall"
    con_rec_name_nice = "Contradictions Recall"
    con_rec_color = "red"
    edge_color = "black"

    # F1
    plt.bar(x - width / 2, df[f1_name], width, color=f1_color, edgecolor=edge_color, hatch='/', alpha=.3, label=f1_name)
    for i, v in enumerate(df[f1_name]):
        t2 = plt.text(i - width / 2 - h_space, v + v_space, f"{round(v, 3):.3f}", color="black", ha='center',
                      fontsize=25)  # 3 digits!!
        t2.set_bbox({"facecolor": 'white', "alpha": 1, "edgecolor": 'white'})
        # Precision

    # Recall
    plt.bar(x + width / 2, df[con_rec_name], width, color=con_rec_color, edgecolor=edge_color, hatch='/', alpha=.3,
            label=con_rec_name)
    for i, v in enumerate(df[con_rec_name]):
        t1 = plt.text(i + width / 2 + h_space, v + v_space, f"{round(v, 3):.3f}", color="black", ha='center',
                      fontsize=25)
        t1.set_bbox({"facecolor": 'white', "alpha": 1, "edgecolor": 'white'})

    # Main results
    plt.bar(x[:2] - width / 2, df[f1_name][:2], width, color=f1_color)
    plt.bar(x[:2] + width / 2, df[con_rec_name][:2], width, color=con_rec_color)

    plt.xticks(x, benchmark_names, rotation=20, ha='right')

    f1_patch = Patch(facecolor=f1_color, edgecolor='black')
    con_rec_patch = Patch(facecolor=con_rec_color, edgecolor='black')

    plt.legend([f1_patch, con_rec_patch], [f1_name_nice, con_rec_name_nice])

    plt.ylim([-0.01, 1.05])
    plt.ylabel("Metric")
    plt.title("NLI Test Metrics with Benchmarks", size=40)

    if do_save:
        out_dir = os.path.join(fig_dir, "TestEval")
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, "TestEvalBenchmarks.png"), dpi=dpi, bbox_inches="tight")

    plt.show()
