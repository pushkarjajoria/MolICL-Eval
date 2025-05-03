import numpy as np
import pandas as pd
import os
from huggingface_hub import hf_hub_download
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

pd.set_option('display.max_colwidth', None)


def _smiles_to_morgan_fingerprint(smile: str, radius=2, n_bits=2048) -> np.ndarray:
    """Convert SMILES string to Morgan fingerprint."""
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return np.zeros(n_bits)  # Return a zero vector for invalid SMILES
    fp = mfpgen.GetFingerprint(mol)
    return np.array(fp, dtype=bool)


def bbbp():
    """Downloads BBBP dataset from Hugging Face and returns train/test splits."""
    splits = {
        'train': 'data/train-00000-of-00001-2bfad0590b3883b6.parquet',
        'validation': 'data/validation-00000-of-00001-03d16234c2ea6a15.parquet',
        'test': 'data/test-00000-of-00001-8e055707481c4baa.parquet'
    }

    # Create datasets directory if not exists
    os.makedirs("datasets/bbbp", exist_ok=True)

    dfs = {}

    for split_name, split_path in splits.items():
        # Download and cache the dataset split
        local_path = hf_hub_download(
            repo_id="zpn/bbbp",
            filename=split_path,
            repo_type="dataset",
            local_dir="datasets/bbbp",
            local_dir_use_symlinks=False
        )
        # Read parquet file into DataFrame
        df = pd.read_parquet(local_path)
        dfs[split_name] = df.rename(columns={'target': 'label'})

    return dfs['train'], dfs['test'], dfs['validation']


# Usage example
if __name__ == '__main__':
    train_df, test_df, val_df = bbbp()
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}, Validation samples: {len(val_df)}")

    # Configure pandas to display ALL data without truncation
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.max_colwidth', None)  # Show full width of each column
    pd.set_option('display.width', None)  # Auto-detect terminal width
    pd.set_option('display.expand_frame_repr', False)  # Prevent line breaks

    print("\nTrain DataFrame Head (Full Content):")
    print(train_df.head(25))
