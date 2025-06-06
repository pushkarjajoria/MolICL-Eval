import argparse
import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

from download_datasets import bbbp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    for layer in tqdm(hidden_states):
        # Sum embeddings where mask is 1, then divide by sum of mask
        sum_embeddings = torch.sum(layer * mask, dim=1)
        sum_mask = torch.sum(attention_mask, dim=1, keepdim=True).float()
        sum_mask = torch.clamp(sum_mask, min=1e-9)  # Avoid division by zero
        avg_embeddings = sum_embeddings / sum_mask
        all_layers_avg.append(avg_embeddings)

    # Stack layers to shape (batch_size, num_layers, emb_dim)
    output = torch.stack(all_layers_avg, dim=1)

    # Convert to numpy array and return
    return output


def compute_mean_variance(smiles, model_name):
    if "MoLFormer" in model_name:
        model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True,
                                          trust_remote_code=True, output_hidden_states=True)
        tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token  # Set padding token
        model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True, trust_remote_code=True)
    model.eval()  # Set model to evaluation mode
    embeddings = extract_hidden_layers_avg_pooling(smiles, tokenizer, model)  # (batch, layers, dim)
    mean_embeddings, std_embeddings = torch.mean(embeddings, dim=0), torch.std(embeddings, dim=0)
    model_str = model_name.split("/")[-1]
    pickle_dir = "/data/users/pjajoria/pickle_dumps/mean_std_embeddings"
    os.makedirs(pickle_dir, exist_ok=True)
    with open(os.path.join(pickle_dir, f"{model_str}.pkl"), "wb") as f:
        pickle.dump({"mean": mean_embeddings, "std": std_embeddings}, f)
    print(f"Saved mean and std embeddings in {pickle_dir}/{model_str}.pkl")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model_name', type=str, required=True, help='Model name')
    args = arg_parser.parse_args()
    model_name = args.model_name
    # model_name = "ibm-research/MoLFormer-XL-both-10pct"

    train_df, test_df, val_df = bbbp()
    smiles = list(test_df['smiles'])
    compute_mean_variance(smiles, model_name)
