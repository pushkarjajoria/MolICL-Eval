import argparse
import os
import pickle
import random
import time
from typing import Optional

import torch
from tqdm import tqdm

from generate_non_canonical_dataset import get_bbbp_non_canonical_smiles
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

torch.cuda.empty_cache()


def entrypoint(res, model_name, mean=None, variance=None):
    # First collect all prompt-specific results
    all_exp1 = []

    # Get prompts from first entry (same for all)
    prompts = next(iter(res.values()))["prompts"]

    for prompt_idx in range(len(prompts)):
        # Aggregate data across all SMILES entries for this prompt
        prompt_res = {}
        for smiles_key in res.keys():
            entry = res[smiles_key]
            prompt_res[smiles_key] = {
                "smiles": entry["smiles"],
                "non_canonical_smiles": entry["non_canonical_smiles"],
                "canonical_embeddings": entry["canonical_embeddings"][prompt_idx],
                "non_canonical_embeddings": entry["non_canonical_embeddings"][prompt_idx]
            }

        # Run experiments for this prompt
        exp1_means, exp1_stderrs = experiment1(prompt_res, mean, variance)

        # Store for averaging
        all_exp1.append(exp1_means)

        # # Plot per-prompt results
        # plot_experiment1(exp1_means, exp1_stderrs, model_name,
        #                  f"prompt_{prompt_idx}")

    # Compute cross-prompt averages
    avg_exp1_means = np.mean(all_exp1, axis=0)
    avg_exp1_stderr = np.std(all_exp1, axis=0) / np.sqrt(len(prompts))

    fig_dict = {"avg_exp1_means": avg_exp1_means, "avg_exp1_stderr": avg_exp1_stderr, "model_name": model_name, "prompt_suffix": "avg_all_prompts"}
    normalized = mean is not None and variance is not None
    norm_text = "" if not normalized else "_normalized"
    plot_dir = f"/nethome/pjajoria/Github/MolICL-Eval/results/prompting_baseline_plots{norm_text}/{model_name}"
    os.makedirs(plot_dir, exist_ok=True)
    filename = f"{plot_dir}/{model_name}_exp1{norm_text}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(fig_dict, f)
    print(f"Saved {filename}")

    # Plot averaged results
    plot_experiment1(avg_exp1_means, avg_exp1_stderr, model_name, "avg_all_prompts", plot_dir)


def experiment1(res, dataset_mean, dataset_variance):
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

            # Calculate pairwise similarities
            sims = cosine_similarity(layer_canon, layer_noncanon).flatten()

            # Store results for statistical analysis
            layer_accumulator[layer_idx].extend(sims)

    # Compute statistics across all samples
    means = [np.mean(sims) for sims in layer_accumulator]
    stderrs = [np.std(sims, ddof=1) / np.sqrt(len(sims)) for sims in layer_accumulator]

    return means, stderrs


def experiment2(res):
    """
    Evaluates embedding quality through synthetic retrieval tasks:
    1. For each compound (query) in the dictionary:
        a. Create candidate pool of size 5 containing:
            - 1 true non-canonical variant (positive)
            - 4 random non-canonical variants from other compounds (negatives)
    2. For each layer:
        a. Calculate query-candidate cosine similarities
        b. Rank candidates by similarity
        c. Record precision@1 and precision@2 metrics
    3. Aggregate performance across all queries

    Returns:
        tuple: (p1_means, p1_errors, p2_means, p2_errors) per layer
    """
    # Initialize precision trackers
    first_key = next(iter(res))
    num_layers = res[first_key]["canonical_embeddings"].shape[1]
    p1_scores = [[] for _ in range(num_layers)]  # Precision@1 storage
    p2_scores = [[] for _ in range(num_layers)]  # Precision@2 storage
    all_keys = list(res.keys())

    # Process each query compound
    for query_key in all_keys:
        # Get query embeddings and true positive
        query_entry = res[query_key]
        query_emb = query_entry["canonical_embeddings"].squeeze()

        # Randomly select one non-canonical as positive
        correct_idx = np.random.randint(0, 5)
        correct_emb = query_entry["non_canonical_embeddings"][correct_idx]

        # Sample negative candidates from other compounds
        other_keys = [k for k in all_keys if k != query_key]
        sampled_keys = np.random.choice(other_keys, size=4, replace=False)

        # Build candidate pool (1 positive + 4 negatives)
        candidate_embs = [correct_emb] + [
            res[k]["non_canonical_embeddings"][np.random.randint(0, 5)]
            for k in sampled_keys
        ]

        # Process each layer
        for layer_idx in range(num_layers):
            # Extract layer-specific embeddings
            query_layer = query_emb[layer_idx].reshape(1, -1)
            cand_layers = [cand[layer_idx].reshape(1, -1) for cand in candidate_embs]

            # Compute similarity scores
            sims = np.array([cosine_similarity(query_layer, cl)[0][0] for cl in cand_layers])

            # Rank candidates by similarity (descending order)
            sorted_indices = np.argsort(-sims)

            # Determine if positive is in top results
            correct_pos = np.where(sorted_indices == 0)[0][0]  # First entry is the correct emb
            p1 = 1 if correct_pos == 0 else 0  # Top-1 check
            p2 = 1 if correct_pos < 2 else 0  # Top-2 check

            # Record precision metrics
            p1_scores[layer_idx].append(p1)
            p2_scores[layer_idx].append(p2)

    # Aggregate layer-wise performance after processing all queries
    p1_means = [np.mean(scores) for scores in p1_scores]
    p1_stderrs = [np.std(scores, ddof=1) / np.sqrt(len(scores)) for scores in p1_scores]
    p2_means = [np.mean(scores) for scores in p2_scores]
    p2_stderrs = [np.std(scores, ddof=1) / np.sqrt(len(scores)) for scores in p2_scores]

    return p1_means, p1_stderrs, p2_means, p2_stderrs


def plot_experiment1(means, stderrs, model_name, prompt_suffix, plot_dir):
    plt.figure(figsize=(10, 6))
    plt.errorbar(range(len(means)), means, yerr=stderrs,
                 fmt='-o', capsize=5, color='darkgreen')
    plt.xlabel("Layer Number")
    plt.ylabel("Cosine Similarity")
    plt.title(f"{model_name} - Baseline Canonical vs Non-Canonical\n({prompt_suffix})")
    plt.grid(alpha=0.3)
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f"{plot_dir}/{model_name}_exp1_{prompt_suffix}.png")
    plt.close()


def plot_experiment2(p1_means, p1_stderrs, p2_means, p2_stderrs,
                     model_name, prompt_suffix):
    p1_means, p1_stderrs, p2_means, p2_stderrs = np.array(p1_means), np.array(p1_stderrs), np.array(p2_means), np.array(
        p2_stderrs)
    plt.figure(figsize=(10, 6))
    layers = range(len(p1_means))

    plt.plot(layers, p1_means, '-o', color='navy', label='Precision@1')
    plt.fill_between(layers, p1_means - p1_stderrs, p1_means + p1_stderrs, alpha=0.2)

    plt.plot(layers, p2_means, '-s', color='crimson', label='Precision@2')
    plt.fill_between(layers, p2_means - p2_stderrs, p2_means + p2_stderrs, alpha=0.2)

    plt.xlabel("Layer Number")
    plt.ylabel("Precision")
    plt.title(f"{model_name} - Retrieval Performance\n({prompt_suffix})")
    plt.legend()
    plt.grid(alpha=0.3)
    plot_dir = f"/nethome/pjajoria/Github/MolICL-Eval/results/non_canonical_with_prompting_plots/{model_name}"
    plt.savefig(f"{plot_dir}/{model_name}_exp2_{prompt_suffix}.png")
    plt.close()


def extract_hidden_layers_avg_pooling(input_strings, tokenizer, model, prompts):
    """
    Extracts embeddings for prompt-augmented SMILES strings using pre-computed lengths
    to efficiently slice and pool SMILES tokens.

    Process:
    1. Pre-tokenize original SMILES to get lengths and masks
    2. For each prompt:
        a. Tokenize prompt+SMILES with padding
        b. Get all hidden states
        c. Slice SMILES tokens using pre-computed lengths
        d. Average pool using original masks
    3. Stack results across prompts

    Returns:
        np.ndarray: (num_prompts, batch_size, num_layers, emb_dim)
    """
    # Step 1: Get original SMILES tokenization info
    original_inputs = tokenizer(input_strings, padding=True, truncation=True, return_tensors="pt")
    original_masks = original_inputs.attention_mask
    original_seq_len = original_inputs.input_ids.shape[1]

    outputs_per_prompt = []

    for prompt in prompts:
        # Step 2: Tokenize prompted SMILES
        prompted_inputs = [prompt + s for s in input_strings]
        inputs = tokenizer(prompted_inputs, padding=True, truncation=True, return_tensors="pt")

        # Step 3: Get hidden states
        with torch.no_grad():
            outputs = model(**inputs)
        hidden_states = outputs.hidden_states

        # Step 4: Process each layer
        layer_embeddings = []
        for layer in hidden_states:
            # Calculate needed padding
            prompted_seq_len = layer.shape[1]
            pad_size = max(original_seq_len - prompted_seq_len, 0)

            # Pad and slice embeddings
            padded_layer = torch.nn.functional.pad(layer, (0, 0, 0, pad_size))
            sliced_embs = padded_layer[:, -original_seq_len:, :]  # [batch, orig_len, dim]

            # Apply original masks for pooling
            mask = original_masks.unsqueeze(-1).to(layer.device)
            sum_emb = (sliced_embs * mask).sum(dim=1)
            sum_mask = original_masks.sum(dim=1, keepdim=True).clamp(min=1e-9)

            layer_embeddings.append(sum_emb / sum_mask)

        # Stack layers [batch, layers, dim]
        output = torch.stack(layer_embeddings, dim=1)
        outputs_per_prompt.append(output.cpu().numpy())

    return np.stack(outputs_per_prompt, axis=0)


def get_transformer_emb(hf_model: str, hf_tokenizer: str):
    n_non_canonical_variants = 5
    prompts = ["This is a SMILES for a molecule: ", "SMILES: ", "BBBP Molecule: ",
               "The following is a molecule represented as a SMILES. \n"]
    smiles_map = get_bbbp_non_canonical_smiles(n_variants=n_non_canonical_variants)
    # Load model directly
    if "MoLFormer" in hf_model:
        model = AutoModel.from_pretrained(hf_model, deterministic_eval=True,
                                          trust_remote_code=True, output_hidden_states=True)
        tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token  # Set padding token
        model = AutoModelForCausalLM.from_pretrained(hf_model, output_hidden_states=True, trust_remote_code=True)
    model.eval()  # Set model to evaluation mode

    res = {}
    all_smiles = list(smiles_map.keys())
    for _ in tqdm(range(1000)):
        smile_i, smile_j = random.sample(all_smiles, 2)
        smile_j = smiles_map[smile_j][
            random.randint(0, n_non_canonical_variants - 1)]  # non canonical variant of smile_j
        canonical_embeddings = extract_hidden_layers_avg_pooling([smile_i], tokenizer, model,
                                                                 prompts)  # (#prompts=5, batch, layers, dim)
        non_canonical_embeddings = extract_hidden_layers_avg_pooling([smile_j], tokenizer, model,
                                                                     prompts)  # (#prompts=5, batch, layers, dim)
        res[smile_i + "__" + smile_j] = {"smiles": smile_i,
                                         "non_canonical_smiles": smile_j,
                                         "canonical_embeddings": canonical_embeddings,
                                         "non_canonical_embeddings": non_canonical_embeddings,
                                         "prompts": prompts
                                         }
    return res


def main():
    parser = argparse.ArgumentParser(description="Extract transformer embeddings for a HF model.")
    parser.add_argument(
        "--hf_model_name",
        type=str,
        required=True,
        help="HuggingFace model name (e.g. meta-llama/Meta-Llama-3-8B)"
    )
    args = parser.parse_args()

    hf_model_name = args.hf_model_name
    filename = hf_model_name.split("/")[-1]
    res_dir = "/data/users/pjajoria/pickle_dumps/MolICL-Eval/distance_non_canonical_baseline"
    res_pickle_path = f"{res_dir}/{filename}_pickle.dmp"
    if os.path.exists(res_pickle_path):
        with open(res_pickle_path, "rb") as handle:
            res = pickle.load(handle)
    else:
        print("Result file does not exist. Extracting embeddings...")
        res = get_transformer_emb(hf_model_name, hf_model_name)
        os.makedirs(res_dir, exist_ok=True)
        with open(res_pickle_path, "wb") as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
            time.sleep(2)   # Fixes empty pickle file issue
    plot(hf_model_name, res)


def plot(hf_model_name, res: Optional[dict]):
    filename = hf_model_name.split("/")[-1]
    res_dir = "/data/users/pjajoria/pickle_dumps/MolICL-Eval/distance_non_canonical_wt_prompting"

    if res is None:
        try:
            with open(f"{res_dir}/{filename}_pickle.dmp", "rb") as handle:
                res = pickle.load(handle)
        except FileNotFoundError:
            print("The result (res) was not found and also not provided to the function.")
            raise FileNotFoundError(f"{filename} was not found at location {res_dir}")

    with open(f"/data/users/pjajoria/pickle_dumps/mean_std_embeddings/{filename}.pkl", "rb") as mean_handle:
        obj = pickle.load(mean_handle)
    mean, std = obj["mean"].numpy(), obj["std"].numpy()

    entrypoint(res, filename, mean, std)


if __name__ == "__main__":
    print(f"Running on Device: {device}")
    main()
