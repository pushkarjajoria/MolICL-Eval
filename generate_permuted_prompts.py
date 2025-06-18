import os

import pandas as pd
import yaml
from tqdm import tqdm

from generate_non_canonical_dataset import get_random_smiles_permutations


def generate_prompts(smiles_to_random_perm, prompt_file, k, zeroshot=False):
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
    for smile, label in tqdm(zip(df['smiles'], df['label'])):
        prompt = _generate_prompt(smile) if zeroshot else _generate_prompt(smile, closest_examples)
        prompts.append(prompt)
        labels.append(label)
    return prompts, labels


if __name__ == '__main__':
    # ── Experiment arguments ────────────────────────────────────────────
    PROMPT_FILE  = "prompts/ICL_prompts/bbbp_prompts.yaml"
    OUTPUT_DIR   = "datasets/InContextPrompts"
    KS           = []
    ZERO_SHOT    = {"name": "zeroshot", "k": 0, "equal": False, "zeroshot": True}
    # ────────────────────────────────────────────────────────────────────

    # prepare data
    n_random_variants = 5
    bbbp_test_dict = get_random_smiles_permutations(n_variants=n_random_variants)

    # ensure output folder exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[INFO] Output directory: {OUTPUT_DIR}")

    # build the list of experiments
    experiments = [ZERO_SHOT]
    # zero-shot
    # k-shot experiments
    for k in KS:
        exp_name = f"k{k}"
        experiments.append({"name": exp_name, "k": k, "zeroshot": False})

    # run experiments
    for exp in experiments:
        name    = exp["name"]
        k       = exp["k"]
        equal   = exp["equal"]
        zs_flag = exp["zeroshot"]

        print(f"\n[RUNNING] Experiment: {name} | k={k} | equal={equal} | zeroshot={zs_flag}")
        prompts, labels = generate_prompts(
            bbbp_test_dict,
            prompt_file=PROMPT_FILE,
            k=k,
            zeroshot=zs_flag,
        )

        # build output path per experiment
        out_file = os.path.join(OUTPUT_DIR, f"bbbp_{name}.csv")
        print(f"[SAVE] Writing prompts+labels → {out_file}")
        pd.DataFrame({"prompt": prompts, "label": labels}) \
          .to_csv(out_file, index=False)

    print("\n[ALL DONE] Generated all prompts.")
