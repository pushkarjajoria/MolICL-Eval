import heapq
import json
from typing import Optional
import os
import pandas as pd
import numpy as np
import yaml
from pandas import DataFrame
from tqdm import tqdm

from download_datasets import bbbp, _smiles_to_morgan_fingerprint
SYSTEM_PROMPTS_FILE = "prompts/systemprompt.json"


def tanimoto_similarity(test_fingerprint, train_fingerprint):
    # Ensure both fingerprints are binary
    assert test_fingerprint.dtype == bool, "test_fingerprint must be of boolean type"
    assert train_fingerprint.dtype == bool, "train_fingerprint must be of boolean type"
    assert test_fingerprint.shape == train_fingerprint.shape, "Fingerprints must be same shape"
    intersection = np.sum(np.logical_and(test_fingerprint, train_fingerprint))
    sum_A = np.sum(test_fingerprint)
    sum_B = np.sum(train_fingerprint)
    tanimoto_similarity = intersection/(sum_A + sum_B - intersection)
    return tanimoto_similarity


def find_closest(
        df: DataFrame,
        k: int,
        smile: str,
        *,
        equal: bool = True,
        positive: Optional[bool] = None,
) -> list[tuple[str, int]]:
    """
    Find the k most Tanimoto‐similar rows to `smile` in `df`.

    If equal=True:
        positive may be True or False (and defaults to False if left as None).
        We filter df to rows whose 'label' == int(positive).
    If equal=False:
        positive must be None (and we ignore label‐filtering entirely).
    """
    # 1) validate the args
    if not equal:
        if positive is not None:
            raise ValueError("`positive` must be None when `equal=False`")
    else:
        # default positive to False if caller didn’t specify it
        positive = bool(positive)

    # 2) do your filtering
    if equal:
        filtered_df = df[df['label'] == int(positive)]
    else:
        filtered_df = df

    # 3) build a heap of (similarity, (smiles, label))
    fp = _smiles_to_morgan_fingerprint(smile)
    heap: list[tuple[float, tuple[str, int]]] = []

    for _, row in filtered_df.iterrows():
        sim = tanimoto_similarity(fp, row['fingerprint'])
        # push an item; if we exceed k, pop the smallest
        heapq.heappush(heap, (sim, (row['smiles'], row['label'])))
        if len(heap) > k:
            heapq.heappop(heap)

    # 4) return the top‐k, sorted descending by similarity
    return [item[1] for item in sorted(heap, reverse=True)]


def generate_prompts(train_df, test_df, prompt_file, k, zeroshot=False, equal=True):
    def _generate_prompt(smile, examples=None):
        if zeroshot and examples is not None:
            raise ValueError("examples cannot be used with zeroshot")
        if not zeroshot and examples is None:
            raise ValueError("examples cannot be none with kshot")

        if zeroshot:
            return base_prompt + "\n" + f"SMILES: {smile}\n" + "BBB Penetration: "
        else:
            example_string = ""
            for ex_smile, ex_label in examples:
                example_string += "\n" + f"Example SMILES: {ex_smile}\n" + f"BBB Penetration: {int(ex_label)}"
        return base_prompt + example_string + "\n" + f"Input SMILES: {smile}\n" + "BBB Penetration: "

    if k > 0 and zeroshot:
        raise ValueError("Either zeroshot or k > 0")
    # Load prompt variants from YAML
    with open(prompt_file, 'r') as f:
        prompt_variants = yaml.safe_load(f)

    # Select appropriate variant
    base_prompt = prompt_variants['variants']['zeroshot'] if zeroshot else prompt_variants['variants']['kshot']
    prompts, labels = [], []
    for smile, label in tqdm(zip(test_df['smiles'], test_df['label'])):
        if equal and k%2 != 0:
            raise ValueError("k should be divisible by 2 to ensure equal distribution")
        if equal:
            closest_positive = find_closest(train_df, k // 2, smile, positive=True)
            closest_negative = find_closest(train_df, k // 2, smile, positive=False)
            closest_examples = closest_positive + closest_negative
        else:
            closest_examples = find_closest(train_df, k, smile, positive=None, equal=False)
        prompt = _generate_prompt(smile) if zeroshot else _generate_prompt(smile, closest_examples)
        prompts.append(prompt)
        labels.append(label)
    return prompts, labels


if __name__ == '__main__':
    # ── Experiment arguments ────────────────────────────────────────────
    PROMPT_FILE  = "prompts/ICL_prompts/bbbp_prompts.yaml"
    OUTPUT_DIR   = "datasets/InContextPrompts"
    KS           = [4, 6, 8]
    EQUAL_OPTS   = [True, False]
    ZERO_SHOT    = {"name": "zeroshot", "k": 0, "equal": False, "zeroshot": True}
    # ────────────────────────────────────────────────────────────────────

    # prepare data
    train_df, test_df, val_df = bbbp()
    for df in [train_df, test_df, val_df]:
        df["fingerprint"] = df['smiles'].apply(_smiles_to_morgan_fingerprint)

    # ensure output folder exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[INFO] Output directory: {OUTPUT_DIR}")

    # build the list of experiments
    experiments = [ZERO_SHOT]
    # zero-shot
    # k-shot experiments
    for k in KS:
        for eq in EQUAL_OPTS:
            exp_name = f"k{k}_{'eq' if eq else 'noeq'}"
            experiments.append({"name": exp_name, "k": k, "equal": eq, "zeroshot": False})

    # run experiments
    for exp in experiments:
        name    = exp["name"]
        k       = exp["k"]
        equal   = exp["equal"]
        zs_flag = exp["zeroshot"]

        print(f"\n[RUNNING] Experiment: {name} | k={k} | equal={equal} | zeroshot={zs_flag}")
        prompts, labels = generate_prompts(
            train_df,
            test_df,
            prompt_file=PROMPT_FILE,
            k=k,
            zeroshot=zs_flag,
            equal=equal
        )

        # build output path per experiment
        out_file = os.path.join(OUTPUT_DIR, f"bbbp_{name}.csv")
        print(f"[SAVE] Writing prompts+labels → {out_file}")
        pd.DataFrame({"prompt": prompts, "label": labels}) \
          .to_csv(out_file, index=False)

    print("\n[ALL DONE] Generated all prompts.")
