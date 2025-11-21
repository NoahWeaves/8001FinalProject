"""
* File: config.py
* Author: Loic Martins
* Date: 2025-11-16
* Description:
    Specific configurations for the training.
"""

# Import external Libraries
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Holds all static and dynamic training configurations."""

    seed: int = 42
    data_path: str = "./training/data/"
    dataset_path: str = "./training/data/combined_dataset.csv"
    dataset_path_eval: str = "./training/data/combined_dataset_eval.csv"
    output_path: str = "./results"
    model_name: str = "Qwen/Qwen2.5-1.5B"
    fine_tuned_path: str = "./model/fine_tuned_model"
