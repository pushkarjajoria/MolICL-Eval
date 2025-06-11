import os
import pickle
import matplotlib.pyplot as plt
from itertools import cycle


def load_fig_dicts(baseline_dir, exp_dir):
    """
    Load figure dictionaries for each model from baseline and experiment directories.
    Returns a dict: {model_name: {'baseline': fig_dict, 'exp': fig_dict}}
    """
    fig_dicts = {}
    # Expect each subfolder in baseline_dir corresponds to a model
    for model in os.listdir(baseline_dir):
        base_model_dir = os.path.join(baseline_dir, model)
        exp_model_dir = os.path.join(exp_dir, model)
        if not os.path.isdir(base_model_dir) or not os.path.isdir(exp_model_dir):
            continue
        # find pickle files inside each model folder (assumes one per folder)
        base_files = [f for f in os.listdir(base_model_dir) if f.endswith('.pkl')]
        exp_files  = [f for f in os.listdir(exp_model_dir)  if f.endswith('.pkl')]
        if not base_files or not exp_files:
            continue
        # load pickles
        with open(os.path.join(base_model_dir, base_files[0]), 'rb') as f:
            base_dict = pickle.load(f)
        with open(os.path.join(exp_model_dir, exp_files[0]), 'rb') as f:
            exp_dict  = pickle.load(f)
        fig_dicts[model] = {'baseline': base_dict, 'exp': exp_dict}
    return fig_dicts


def plot_all_models(fig_dicts, plot_dir, plot_name, plot_title, include_err_bars=True):
    """
    Plot means (and optionally error bars) for all models and both baseline (dotted) and experiment (solid).
    Saves a figure to plot_dir.
    """
    # Prepare colors for each model
    colors = cycle(['darkgreen', 'darkblue', 'darkred', 'darkorange'])
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(12, 7))

    for model, color in zip(sorted(fig_dicts.keys()), colors):
        sets = fig_dicts[model]
        for kind, linestyle in [('baseline', '--'), ('exp', '-')]:
            data = sets[kind]
            means = data['exp1_mean']
            if include_err_bars:
                stderrs = data['exp1_stderr']
                plt.errorbar(
                    range(len(means)), means, yerr=stderrs,
                    fmt='o'+linestyle[0], capsize=4,
                    label=f"{model} ({kind})", linestyle=linestyle, color=color
                )
            else:
                plt.plot(
                    range(len(means)), means,
                    linestyle=linestyle, marker='.',
                    label=f"{model} ({kind})", color=color
                )

    plt.xlabel("Layer Number", fontsize=12)
    plt.ylabel("Cosine Similarity", fontsize=12)
    title_suffix = "with Error Bars" if include_err_bars else "means Only"
    plt.title(plot_title + f" {title_suffix}", fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend()

    suffix = '_with_err' if include_err_bars else '_no_err'
    out_path = os.path.join(plot_dir, f"{plot_name}{suffix}.png")
    plt.savefig(out_path)
    plt.close()
    # plt.show()
    print(f"Saved plot to {out_path}")


def load_and_plot_all(baseline_dir, exp_dir, plot_dir, save_file_name, plot_title):
    """
    Main function: loads all pickle dicts and generates two comparison plots,
    one with error bars, one with only means.
    """
    fig_dicts = load_fig_dicts(baseline_dir, exp_dir)
    # Plot with error bars
    # plot_all_models(fig_dicts, plot_dir, include_err_bars=True)
    # Plot without error bars
    plot_all_models(fig_dicts, plot_dir, save_file_name, plot_title, include_err_bars=False)


# Example usage:
if __name__ == "__main__":
    baseline_dir = "/nethome/pjajoria/Github/MolICL-Eval/results/random_permutations_normalized"
    exp_dir = "/nethome/pjajoria/Github/MolICL-Eval/results/non_canonical_plots_normalized"
    plot_dir = "/nethome/pjajoria/Github/MolICL-Eval/results/all_distance_plot_normalized"
    plot_title = "Randomized Smiles vs Non Canonical"
    save_file_name = "randomized_vs_NonCanonical"
    load_and_plot_all(baseline_dir, exp_dir, plot_dir, save_file_name, plot_title)
