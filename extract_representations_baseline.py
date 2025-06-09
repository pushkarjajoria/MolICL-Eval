import argparse
import os
import pickle
import random

import torch
from tqdm import tqdm

from generate_non_canonical_dataset import get_bbbp_non_canonical_smiles
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# device = torch.device('cpu')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

torch.cuda.empty_cache()


def entrypoint(res, model_name, mean=None, variance=None):
    normalized = mean is not None and variance is not None
    norm_text = "" if not normalized else "_normalized"
    plot_dir = f"/nethome/pjajoria/Github/MolICL-Eval/results/baseline_non_canonical_{norm_text}/{model_name}"
    os.makedirs(plot_dir, exist_ok=True)

    # First run experiment 1
    exp1_means, exp1_stderrs = experiment1(res, mean, variance)
    fig_dict = {"exp1_mean": exp1_means, "exp1_stderr": exp1_stderrs, "model_name": model_name, "normalized": normalized}
    filename = f"{plot_dir}/{model_name}_exp1{norm_text}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(fig_dict, f)
    print(f"Saved {filename}")
    plot_experiment1(exp1_means, exp1_stderrs, model_name, normalized, plot_dir)

    # Then run experiment 2
    # exp2_p1_means, exp2_p1_stderrs, exp2_p2_means, exp2_p2_stderrs = experiment2(res)
    # plot_experiment2(exp2_p1_means, exp2_p1_stderrs, exp2_p2_means, exp2_p2_stderrs, model_name)


def experiment1(res, dataset_mean=None, dataset_variance=None):
    """
    Computes layer-wise cosine similarity between canonical SMILES embeddings and their
    corresponding non-canonical variants across all entries in the dictionary.

    Process:
    1. For each entry in the dictionary values:
        a. Extract canonical embeddings (shape: [num_layers, emb_dim])
        b. Extract non-canonical embeddings (shape: [5, num_layers, emb_dim])
    2. For each transformer layer:
        a. Calculate cosine similarity between canonical and each of the 5 non-canonical embeddings
        b. Aggregate similarities across all entries
    3. Compute layer-wise statistics (mean ± SEM) across all comparisons

    Returns:
        tuple: (means: List[float], standard_errors: List[float]) per layer
    """
    # Step 1: Initialize storage for layer similarities
    first_key = next(iter(res))
    num_layers = res[first_key]["canonical_embeddings"].shape[1]
    layer_accumulator = [[] for _ in range(num_layers)]

    # Step 2: Process each chemical entry in the dictionary
    for entry in res.values():  # Iterate through dictionary values
        # Extract embeddings for current compound
        canonical = entry["canonical_embeddings"].squeeze()  # [num_layers, emb_dim]
        non_canonicals = entry["non_canonical_embeddings"]  # [5, num_layers, emb_dim]

        # Process each transformer layer
        for layer_idx in range(num_layers):
            # Reshape for sklearn compatibility
            layer_canon = canonical[layer_idx].reshape(1, -1)  # [1, emb_dim]
            layer_noncanon = non_canonicals[:, layer_idx, :]  # [5, emb_dim]

            if dataset_mean is not None and dataset_variance is not None:
                # Normalize both vectors from the correct layer_idx
                layer_canon = (layer_canon-dataset_mean[layer_idx])/(dataset_variance[layer_idx] + 1e-6)
                layer_noncanon = (layer_noncanon-dataset_mean[layer_idx])/(dataset_variance[layer_idx] + 1e-6)
            # Else do nothing.

            # Calculate pairwise similarities
            sims = cosine_similarity(layer_canon, layer_noncanon).flatten()

            # Store results for statistical analysis
            layer_accumulator[layer_idx].extend(sims)

    # Compute statistics across all samples
    means = [np.mean(sims) for sims in layer_accumulator]
    stderrs = [np.std(sims, ddof=1) / np.sqrt(len(sims)) for sims in layer_accumulator]

    return means, stderrs


def plot_experiment1(means, stderrs, model_name, normalized, plot_dir):
    norm_text = "" if not normalized else "_normalized"
    plt.figure(figsize=(10, 6))
    plt.errorbar(range(len(means)), means, yerr=stderrs,
                 fmt='-o', capsize=5, color='darkgreen')
    plt.xlabel("Layer Number", fontsize=12)
    plt.ylabel("Cosine Similarity", fontsize=12)
    plt.title(f"[{model_name}] Baseline Canonical vs Non-Canonical Sim {norm_text[1:].upper()}", fontsize=14)
    plt.grid(alpha=0.3)
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f"{plot_dir}/{model_name}_exp1{norm_text}.png")
    plt.close()


def extract_hidden_layers_avg_pooling(input_strings, tokenizer, model):
    # Load tokenizer and model
    # Tokenize input texts with padding and truncation
    inputs = tokenizer(input_strings, padding=True, truncation=True, return_tensors="pt")
    # input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    # Retrieve hidden states from the model
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.hidden_states  # Tuple of (embedding_output, layer1, layer2, ...)

    # Prepare mask for averaging (batch_size, seq_len, 1)
    mask = attention_mask.unsqueeze(-1).float()

    # Compute average pooling for each layer
    all_layers_avg = []
    for layer in hidden_states:
        # Sum embeddings where mask is 1, then divide by sum of mask
        sum_embeddings = torch.sum(layer * mask, dim=1)
        sum_mask = torch.sum(attention_mask, dim=1, keepdim=True).float()
        sum_mask = torch.clamp(sum_mask, min=1e-9)  # Avoid division by zero
        avg_embeddings = sum_embeddings / sum_mask
        all_layers_avg.append(avg_embeddings)

    # Stack layers to shape (batch_size, num_layers, emb_dim)
    output = torch.stack(all_layers_avg, dim=1)

    # Convert to numpy array and return
    return output.numpy()


def get_transformer_emb(hf_model: str, hf_tokenizer: str):
    n_non_canonical_variants = 5
    smiles_map = get_bbbp_non_canonical_smiles(n_variants=n_non_canonical_variants)

    if "MoLFormer" in hf_model:
        model = AutoModel.from_pretrained(hf_model, deterministic_eval=True,
                                          trust_remote_code=True, output_hidden_states=True)
        tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token  # Set padding token
        model = AutoModelForCausalLM.from_pretrained(hf_model, output_hidden_states=True, trust_remote_code=True)
    model.eval()  # Set model to evaluation mode

    res = {}
    all_smiles = list(smiles_map.keys())
    for _ in tqdm(range(1000)):
        smile_i, smile_j = random.sample(all_smiles, 2)
        smile_j = smiles_map[smile_j][
            random.randint(0, n_non_canonical_variants - 1)]  # non canonical variant of smile_j
        canonical_embeddings = extract_hidden_layers_avg_pooling([smile_i], tokenizer, model)  # (#prompts=5, batch, layers, dim)
        non_canonical_embeddings = extract_hidden_layers_avg_pooling([smile_j], tokenizer, model)  # (#prompts=5, batch, layers, dim)
        res[smile_i + "__" + smile_j] = {"smiles": smile_i,
                                         "non_canonical_smiles": smile_j,
                                         "canonical_embeddings": canonical_embeddings,
                                         "non_canonical_embeddings": non_canonical_embeddings
                                         }
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract transformer embeddings for a HF model.")
    parser.add_argument(
        "--hf_model_name",
        type=str,
        required=True,
        help="HuggingFace model name (e.g. meta-llama/Meta-Llama-3-8B)"
    )
    args = parser.parse_args()
    hf_model_name = args.hf_model_name

    # hf_model_name = "google/gemma-3-4b-it"
    filename = hf_model_name.split("/")[-1]
    res_pickle_path = f"/data/users/pjajoria/pickle_dumps/MolICL-Eval/distance_non_canonical_baseline/{filename}_pickle.dmp"
    if os.path.exists(res_pickle_path):
        with open(res_pickle_path, "rb") as handle:
            res = pickle.load(handle)
    else:
        res = get_transformer_emb(hf_model_name, hf_model_name)
        with open(res_pickle_path, "wb") as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f"/data/users/pjajoria/pickle_dumps/mean_std_embeddings/{filename}.pkl", "rb") as mean_handle:
        obj = pickle.load(mean_handle)
    mean, std = obj["mean"].numpy(), obj["std"].numpy()
    entrypoint(res, filename, mean, std)
