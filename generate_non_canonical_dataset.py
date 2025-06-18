import json
from pathlib import Path
from rdkit import Chem
from tqdm import tqdm
from download_datasets import bbbp
import random
from collections import Counter
from tqdm.auto import tqdm


def is_valid_parentheses(s: str) -> bool:
    stack = []
    brackets = list("()[]{}")
    mapping = {")": "(", "}": "{", "]": "["}

    for char in s:
        if char not in brackets:
            continue

        if char in mapping.values():
            stack.append(char)
        elif char in mapping.keys():
            if not stack or mapping[char] != stack.pop():
                return False

    return not stack


def generate_random_permutations_with_valid_brackets(smiles_list, n_variants):
    def fix_brackets(s: str) -> str:
        """
        TODO: Fix Bug here for CC(N[C@@)] where outer bracket close before inner bracket
        Fix both parentheses and square-bracket balance by swapping unmatched
        closing brackets with later unmatched openings.
        """
        # We'll record swap‐pairs for both bracket types
        swap_pairs = []

        # Helper to collect swaps for one bracket type
        def collect(swaps, s, open_b, close_b):
            stack = []
            i, j = 0, len(s) - 1
            while i < j:
                if s[i] == close_b:
                    if stack:
                        stack.pop()
                    else:
                        # no matching open before: find an open at or after j
                        while i < j:
                            if s[j] == open_b:
                                swaps.append((i, j))
                                j -= 1
                                break
                            j -= 1
                elif s[i] == open_b:
                    stack.append(i)
                i += 1

        # collect swaps for () and for []
        collect(swap_pairs, s, "(", ")")
        collect(swap_pairs, s, "[", "]")

        # apply all swaps
        s_list = list(s)
        for i, j in swap_pairs:
            s_list[i], s_list[j] = s_list[j], s_list[i]

        return "".join(s_list)

    result = {}
    for smi in tqdm(smiles_list, desc="Generating variants"):
        chars = list(smi)
        target = Counter(smi)
        seen = set()
        max_attempts = 100 * n_variants
        attempts = 0

        while len(seen) < n_variants and attempts < max_attempts:
            attempts += 1
            # 1) full random shuffle
            perm = random.sample(chars, len(chars))
            perm = "".join(perm)
            # 2) fix bracket balance
            valid_perm = fix_brackets(perm)
            # assert is_valid_parentheses(valid_perm), f"is_valid Check Failed! \n Output String: {valid_perm}"
            # 3) dedupe & counter check
            if valid_perm in seen or Counter(valid_perm) != target:
                continue
            seen.add(valid_perm)

        if len(seen) < n_variants:
            raise ValueError(f"Couldn’t get {n_variants} valid perms for {smi!r}")

        result[smi] = list(seen)

    return result


def generate_random_smiles(smiles_list, n_variants=5):
    """
    Generate non-canonical (randomized) SMILES for each input SMILES string.

    Parameters:
    - smiles_list: List of SMILES strings.
    - n_variants: Number of randomized SMILES to generate per molecule.

    Returns:
    - A dictionary mapping each input SMILES to a list of randomized SMILES.
    """
    randomized_smiles = {}
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            randomized_smiles[smi] = ['Invalid SMILES']
            continue
        variants = set()
        while len(variants) < n_variants:
            rand_smi = Chem.MolToSmiles(mol, doRandom=True)
            variants.add(rand_smi)
        randomized_smiles[smi] = list(variants)
    return randomized_smiles


def get_bbbp_non_canonical_smiles(n_variants=5):
    train_df, test_df, val_df = bbbp()
    smiles = test_df['smiles']
    random_smiles = generate_random_smiles(smiles,
                                           n_variants=n_variants)  # Dict mapping each smile to non-canonical variants
    return random_smiles


def get_random_smiles_permutations(n_variants=5, split_type="test"):
    train_df, test_df, val_df = bbbp()
    if split_type == "train":
        smiles = train_df['smiles']
    elif split_type == "test":
        smiles = test_df['smiles']
    elif split_type == "val":
        smiles = val_df['smiles']
    else:
        raise ValueError(f"split_type {split_type} must be either 'train', 'val' or 'test'")
    random_smiles = generate_random_permutations_with_valid_brackets(smiles, n_variants=n_variants)
    return random_smiles


if __name__ == "__main__":
    res = get_random_smiles_permutations()
    print("Done")
