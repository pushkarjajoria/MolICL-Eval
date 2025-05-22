import pickle
import torch
from tqdm import tqdm

from generate_non_canonical_dataset import get_bbbp_non_canonical_smiles
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device('cpu')
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

torch.cuda.empty_cache()


def entrypoint(res, model_name):
    # First run experiment 1
    exp1_means, exp1_stderrs = experiment1(res)
    plot_experiment1(exp1_means, exp1_stderrs, model_name)

    # Then run experiment 2
    exp2_p1_means, exp2_p1_stderrs, exp2_p2_means, exp2_p2_stderrs = experiment2(res)
    plot_experiment2(exp2_p1_means, exp2_p1_stderrs, exp2_p2_means, exp2_p2_stderrs, model_name)


def experiment1(res):
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


def plot_experiment1(means, stderrs, model_name):
    plt.figure(figsize=(10, 6))
    plt.errorbar(range(len(means)), means, yerr=stderrs,
                 fmt='-o', capsize=5, color='darkgreen')
    plt.xlabel("Layer Number", fontsize=12)
    plt.ylabel("Cosine Similarity", fontsize=12)
    plt.title(f"[{model_name}] Canonical vs Non-Canonical Embedding Similarity", fontsize=14)
    plt.grid(alpha=0.3)
    plt.show()


def plot_experiment2(p1_means, p1_stderrs, p2_means, p2_stderrs, model_name):
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
    plt.show()


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
    smiles_map = get_bbbp_non_canonical_smiles()

    # Load model directly
    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer)
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token
    model = AutoModelForCausalLM.from_pretrained(hf_model, output_hidden_states=True)
    model.eval()  # Set model to evaluation mode

    res = {}
    for k, v in tqdm(smiles_map.items()):
        canonical_embdeddings = extract_hidden_layers_avg_pooling([k], tokenizer, model)
        non_canonical_embeddings = extract_hidden_layers_avg_pooling(v, tokenizer, model)
        res[k] = {"smiles": k,
                  "non_canonical_smiles": v,
                  "canonical_embeddings": canonical_embdeddings,
                  "non_canonical_embeddings": non_canonical_embeddings
                  }
    return res


if __name__ == '__main__':
    hf_model_name = "meta-llama/Meta-Llama-3-8B"
    filename = hf_model_name.split("/")[-1]

    # res = get_transformer_emb(hf_model_name, hf_model_name)
    # with open(f"/nethome/pjajoria/Github/MolICL-Eval/pickle_dump/distance_non_canonical/{filename}_pickle.dmp", "wb") as handle:
    #     pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done")
    # Plotting the results
    with open(f"/nethome/pjajoria/Github/MolICL-Eval/pickle_dump/distance_non_canonical/{filename}_pickle.dmp", "rb") as handle:
        res = pickle.load(handle)
    entrypoint(res, filename)
