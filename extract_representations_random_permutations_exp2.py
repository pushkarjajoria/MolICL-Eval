import argparse
import os
import pickle
import random
import time
import torch
from tqdm import tqdm
from generate_non_canonical_dataset import get_random_smiles_permutations, get_bbbp_non_canonical_smiles
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelForMaskedLM
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

torch.cuda.empty_cache()


def entrypoint(res, model_name, plot_dir, mean=None, std=None):
    # Then run experiment 2
    exp2_p1_means, exp2_p1_stderrs, exp2_p2_means, exp2_p2_stderrs = experiment2(res)
    fig_dict_exp2 = {"exp2_p1_means": exp2_p1_means, "exp2_p1_stderrs": exp2_p1_stderrs,
                "exp2_p2_means": exp2_p2_means, "exp2_p2_stderrs": exp2_p2_stderrs,
                "model_name": model_name}
    filename = f"{plot_dir}/{model_name}_exp2.pkl"
    with open(filename, "wb") as f:
        pickle.dump(fig_dict_exp2, f)
    print(f"Saved {filename}")

    plot_experiment2(exp2_p1_means, exp2_p1_stderrs, exp2_p2_means, exp2_p2_stderrs, model_name, plot_dir)


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
        correct_emb = query_entry["non_canonical_embeddings"][0]

        # Sample negative candidates from other compounds
        permuted_embeddings = query_entry["permuted_embeddings"]

        # Build candidate pool (1 positive + 5 negatives)
        candidate_embs = [correct_emb] + permuted_embeddings

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
            correct_pos = np.where(sorted_indices == 0)[0][0]
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


def plot_experiment2(p1_means, p1_stderrs, p2_means, p2_stderrs, model_name, plot_dir):
    """
    Visualizes information retrieval performance with shaded error regions

    Parameters:
    p1_means (list): Mean Precision@1 values per layer
    p1_stderrs (list): Standard errors for Precision@1
    p2_means (list): Mean Precision@2 values per layer
    p2_stderrs (list): Standard errors for Precision@2
    """
    plt.figure(figsize=(10, 6))
    layers = range(len(p1_means))

    # Plot Precision@1 with shaded error region
    plt.plot(layers, p1_means, '-o', color='navy', label='Precision@1')
    plt.fill_between(layers,
                     np.array(p1_means) - np.array(p1_stderrs),
                     np.array(p1_means) + np.array(p1_stderrs),
                     color='navy', alpha=0.2)

    # Plot Precision@2 with shaded error region
    plt.plot(layers, p2_means, '-s', color='crimson', label='Precision@2')
    plt.fill_between(layers,
                     np.array(p2_means) - np.array(p2_stderrs),
                     np.array(p2_means) + np.array(p2_stderrs),
                     color='crimson', alpha=0.2)

    plt.xlabel("Layer Number", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(f"[{model_name}] Information Retrieval Performance by Layer.", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f"{plot_dir}/{model_name}_exp2.png")
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
    n_random_variants = 5
    permuted_smiles_map = get_random_smiles_permutations(n_variants=n_random_variants)
    non_canonical_smiles_map = get_bbbp_non_canonical_smiles(n_variants=1)

    if "MoLFormer" in hf_model:
        model = AutoModel.from_pretrained(hf_model, deterministic_eval=True,
                                          trust_remote_code=True, output_hidden_states=True)
        tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer, trust_remote_code=True)
    elif "ChemBERTa" in hf_model:
        tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(hf_model, output_hidden_states=True, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token  # Set padding token
        model = AutoModelForCausalLM.from_pretrained(hf_model, output_hidden_states=True, trust_remote_code=True)
    model.eval()  # Set model to evaluation mode

    res = {}
    for k, v in tqdm(permuted_smiles_map.items()):
        canonical_embdeddings = extract_hidden_layers_avg_pooling([k], tokenizer, model)
        non_canonical_embeddings = extract_hidden_layers_avg_pooling(non_canonical_smiles_map[k], tokenizer, model)
        permuted_embeddings = extract_hidden_layers_avg_pooling(v, tokenizer, model)
        res[k] = {"smiles": k,
                  "permuted_smiles": v,
                  "non_canonical_smiles": non_canonical_smiles_map[k],
                  "canonical_embeddings": canonical_embdeddings,
                  "non_canonical_embeddings": non_canonical_embeddings,
                  "permuted_embeddings": permuted_embeddings
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
    # hf_model_name = "google/gemma-3-1b-it"
    filename = hf_model_name.split("/")[-1]

    # # Directories # #
    res_pickle_path = f"/data/users/pjajoria/pickle_dumps/MolICL-Eval/distance_random_permutations/{filename}_exp2_pickle.dmp"
    res_dir = "/".join(res_pickle_path.split('/')[:-1]) + "/"
    normalizations_file = f"/data/users/pjajoria/pickle_dumps/mean_std_embeddings/{filename}.pkl"
    plot_dir = f"/nethome/pjajoria/Github/MolICL-Eval/results/random_permutations_normalized/{filename}"
    # # ----------- # #
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    if os.path.exists(res_pickle_path):
        print("Result file exists. Loading from pickle")
        with open(res_pickle_path, "rb") as handle:
            res = pickle.load(handle)
    else:
        print("Result file does not exist. Creating the result file.")
        res = get_transformer_emb(hf_model_name, hf_model_name)
        with open(res_pickle_path, "wb") as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
            time.sleep(2)   # Seconds. Hack for sometimes getting empty pickle files
    try:
        with open(normalizations_file, "rb") as mean_handle:
            obj = pickle.load(mean_handle)
        mean, std = obj["mean"].numpy(), obj["std"].numpy()
    except FileNotFoundError:
        print(f"[WARNING] No normalization file found for model: {hf_model_name}")
        mean, std = None, None
    entrypoint(res, filename, plot_dir, mean=mean, std=std)
