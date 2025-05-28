from rdkit import Chem

from download_datasets import bbbp


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
    random_smiles = generate_random_smiles(smiles, n_variants=n_variants)  # Dict mapping each smile to non-canonical variants
    return random_smiles