import os
import numpy as np
import re
from rdkit import Chem
import pickle

print("NumPy version:", np.__version__)
extension = re.compile(r"\.[a-z]+$")

def mkdirs(dirpath:str):
    os.makedirs(dirpath, exist_ok=True)

def extract_dirpath(filepath:str):
    return os.path.dirname(filepath)

def load_embeddings(np_file:str):
    return np.load(np_file, allow_pickle=True)

def extract_extension(path:str):
    print(type(path), type(extension))
    ex = re.search(extension, path)
    return ex.group(0)[1:] if ex is not None else ''

            
def is_valid_smiles_string(smiles_str:str, sanitize:bool=True):
    try:
        m = Chem.MolFromSmiles(smiles_str, sanitize=sanitize)
        if m:
            return True, m
        return False, m
    except:
        return False, m


def load_pkl(filepath:str):
    """
    Load pickled object from file

    Args:
        filepath (str): _description_

    Returns:
        _type_: _description_
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_pkl(obj, out_path:str):
    with open(out_path, 'wb') as f:
        pickle.dump(obj, f)