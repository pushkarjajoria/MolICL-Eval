import os
import pickle
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np


def load_fig_dicts_exp2(exp_dir):
    """
    Load pickle dictionaries for each model in a directory for Experiment 2.
    Returns a dict: {model_key: fig_dict} where fig_dict contains:
      - 'exp2_p1_means', 'exp2_p1_stderrs', 'exp2_p2_means', 'exp2_p2_stderrs', 'model_name'
    """
    fig_dicts = {}
    for model_key in os.listdir(exp_dir):
        model_dir = os.path.join(exp_dir, model_key)
        if not os.path.isdir(model_dir):
            continue
        pkl_files = [f for f in os.listdir(model_dir) if f.endswith('exp2.pkl')]
        if not pkl_files:
            continue
        with open(os.path.join(model_dir, pkl_files[0]), 'rb') as f:
            data = pickle.load(f)
        # Expect data keys: 'exp2_p1_means', 'exp2_p1_stderrs', 'exp2_p2_means', 'exp2_p2_stderrs', 'model_name'
        fig_dicts[model_key] = data
    return fig_dicts


def plot_experiment2_unified(fig_dicts, plot_dir, plot_name, plot_title, include_errbars=True):
    """
    Plot unified Experiment 2 results for all models on a single figure,
    including a baseline dotted line at random chance (1/6).
    """
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(12, 7))
    colors = cycle(['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'])

    for model_key, color in zip(sorted(fig_dicts.keys()), colors):
        d = fig_dicts[model_key]
        model_name = d.get('model_name', model_key)
        layers = np.arange(len(d['exp2_p1_means']))

        # Precision@1
        if include_errbars:
            plt.errorbar(
                layers, d['exp2_p1_means'], yerr=d['exp2_p1_stderrs'],
                fmt='o-', capsize=4,
                label=f"{model_name} P@1", color=color
            )
        else:
            plt.plot(
                layers, d['exp2_p1_means'], 'o-',
                label=f"{model_name} P@1", color=color
            )

        # Precision@2
        if include_errbars:
            plt.errorbar(
                layers, d['exp2_p2_means'], yerr=d['exp2_p2_stderrs'],
                fmt='s--', capsize=4,
                label=f"{model_name} P@2", color=color
            )
        else:
            plt.plot(
                layers, d['exp2_p2_means'], 's--',
                label=f"{model_name} P@2", color=color
            )

    # Random chance baseline at 1/6
    max_layers = max(len(d['exp2_p1_means']) for d in fig_dicts.values())
    plt.hlines(1/6, xmin=0, xmax=max_layers - 1,
               colors='black', linestyles=':', label='Random Chance (1/6)')

    plt.hlines(2/6, xmin=0, xmax=max_layers - 1,
               colors='black', linestyles=':', label='Random Chance (2/6)')

    plt.xlabel("Layer Number", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    title_suffix = "with Error Bars" if include_errbars else "means Only"
    plt.title(plot_title + f" {title_suffix}", fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(ncol=2)

    suffix = '_with_err' if include_errbars else '_no_err'
    out_path = os.path.join(plot_dir, f"{plot_name}{suffix}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved unified Experiment 2 plot to {out_path}")


def load_and_plot_exp2(exp_dir, plot_dir, save_file_name, plot_title):
    """
    Load pickles and generate two unified Experiment 2 plots,
    one with error bars, one with only means.
    """
    fig_dicts = load_fig_dicts_exp2(exp_dir)
    plot_experiment2_unified(fig_dicts, plot_dir, save_file_name, plot_title, include_errbars=True)
    plot_experiment2_unified(fig_dicts, plot_dir, save_file_name, plot_title, include_errbars=False)


# Example usage:
if __name__ == "__main__":
    exp2_dir = "/nethome/pjajoria/Github/MolICL-Eval/results/random_permutations_normalized"
    plot_dir = "/nethome/pjajoria/Github/MolICL-Eval/results/random_permutations_normalized/unified_plot"
    save_name = "exp2_unified"
    title = "Information Retrieval (Exp2) by Layer"
    load_and_plot_exp2(exp2_dir, plot_dir, save_name, title)
