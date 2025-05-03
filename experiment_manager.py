import os
import yaml
from dotenv import load_dotenv
from typing import Dict, Any, List
from create_icl_prompts import ICLDataset, ICLPromptGenerator

load_dotenv()  # Load environment variables


class ExperimentConfig:
    def __init__(self, config_dict: Dict[str, Any]):
        self.dataset_name = config_dict['dataset']['name']
        self.download_fn = config_dict['dataset']['download_fn']
        self.model_name = config_dict['model']['name']
        self.system_prompt = config_dict['model']['system_prompt']
        self.prompt_variants = config_dict['prompting']['variants']
        self.k_shots = config_dict['prompting']['k_shots']
        self.zero_shot = config_dict['prompting']['zero_shot']
        self.hub_repo = config_dict['output']['hub_repo']
        self.hub_dir = config_dict['output']['hub_dir']


class ExperimentRunner:
    def __init__(self, config_path: str = "experiments.yaml"):
        self.experiments = self._load_experiments(config_path)
        self.hf_token = os.getenv("HF_API_TOKEN")

    def _load_experiments(self, path: str) -> List[ExperimentConfig]:
        with open(path) as f:
            configs = yaml.safe_load(f)
            return [ExperimentConfig(exp) for exp in configs['experiments'].values()]

    def run_all(self):
        """Execute all experiments in the config file"""
        for exp in self.experiments:
            self._run_single_experiment(exp)

    def _run_single_experiment(self, exp: ExperimentConfig):
        """Process a single experiment configuration"""
        # Load dataset
        download_fn = getattr(__import__('download_datasets'), exp.download_fn)
        train_df, test_df, val_df = download_fn()

        # Prepare dataset
        dataset = ICLDataset(
            name=exp.dataset_name,
            train_df=train_df,
            test_df=test_df,
            val_df=val_df
        )

        if not exp.zero_shot:
            dataset.build_few_shot_mapping(k=max(exp.k_shots or [0]))

        # Generate prompts
        prompt_generator = ICLPromptGenerator(
            system_prompt=exp.system_prompt,
            dataset=dataset,
            model_name=exp.model_name,
            prompt_config=exp.prompt_variants,
            zero_shot=exp.zero_shot,
            hub_repo=exp.hub_repo,
            hub_dir=exp.hub_dir
        )

        prompt_generator.generate_prompts(
            k_shots=exp.k_shots,
            zero_shot=exp.zero_shot
        )

        # Upload to Hub
        if self.hf_token:
            prompt_generator.upload_to_hub(token=self.hf_token)
