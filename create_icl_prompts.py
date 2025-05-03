import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd
import yaml


@dataclass
class ICLDataset:
    """
    Handles dataset-specific operations and stores few-shot example mappings
    """
    name: str
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    val_df: pd.DataFrame
    few_shot_mapping: Dict[str, List[Tuple[str, int]]] = None
    dataset_config: Dict = None  # For additional parameters

    def __post_init__(self):
        self._validate_dataframes()

    def _validate_dataframes(self):
        """Ensure required columns exist"""
        required_columns = {'smiles', 'label'}
        for df in [self.train_df, self.test_df, self.val_df]:
            if not required_columns.issubset(df.columns):
                raise ValueError(f"DataFrame missing required columns {required_columns}")

    def build_few_shot_mapping(self, k: int = 8):
        """
        TODO: Implement your similarity-based few-shot example selection
        Args:
            k: Number of examples to select per test instance
        """
        self.few_shot_mapping = {}

        # TODO: Replace this with your actual implementation
        for smiles in self.test_df['smiles']:
            # TEMP: Random sampling - replace with similarity-based selection
            examples = self.train_df.sample(k)[['smiles', 'label']].values.tolist()
            self.few_shot_mapping[smiles] = examples

    def get_few_shot_examples(self, test_smiles: str, k: int) -> List[Tuple[str, int]]:
        """Retrieve k examples for a specific test instance"""
        if not self.few_shot_mapping:
            raise ValueError("Run build_few_shot_mapping() first")

        if k > len(self.few_shot_mapping[test_smiles]):
            raise ValueError(f"Requested {k} examples but only {len(self.few_shot_mapping[test_smiles])} available")

        return self.few_shot_mapping[test_smiles][:k]


class ICLPromptGenerator:
    PROMPT_CONFIG_DIR = "prompts"

    def __init__(
            self,
            system_prompt: str,
            dataset: ICLDataset,
            model_name: str,
            prompt_config: str,
            zero_shot: bool,
            hub_repo: str,
            hub_dir: str
    ):
        self.system_prompt = system_prompt
        self.dataset = dataset
        self.model_name = model_name
        self.zero_shot = zero_shot
        self.hub_repo = hub_repo
        self.hub_dir = hub_dir

        self.prompt_config = prompt_config  # Store config filename
        self.prompt_variants = self._load_prompt_variants()

        # Initialize output directories
        self.base_dir = os.path.join("datasets", "InContextPrompts")
        self.model_dir = os.path.join(
            self.base_dir,
            self.dataset.name,
            self.model_name
        )
        os.makedirs(self.model_dir, exist_ok=True)

    def generate_prompts(self, k_shots: List[int], zero_shot: bool):
        """Handle both few-shot and zero-shot generation"""
        if zero_shot:
            self._generate_zero_shot_prompts()
        else:
            super().generate_prompts(k_shots)

    def _generate_zero_shot_prompts(self):
        """Special handling for zero-shot prompts"""
        output_file = os.path.join(self.model_dir, "zero_shot.jsonl")

        with open(output_file, "w") as f:
            for test_smiles, test_label in zip(self.dataset.test_df['smiles'], self.dataset.test_df['label']):
                prompt = self._format_single_prompt(
                    prompt_variant=next(iter(self.prompt_variants.values())),
                    examples=[],
                    test_smiles=test_smiles
                )

                entry = {
                    "prompt": prompt,

                }
                f.write(json.dumps(entry) + "\n")

    def upload_to_hub(self, token: str):
        """Secure upload using Python's HfApi"""
        from huggingface_hub import HfApi
        api = HfApi(token=token)

        api.upload_folder(
            folder_path=self.hub_dir,
            repo_id=self.hub_repo,
            repo_type="dataset"
        )