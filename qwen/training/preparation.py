"""
* File: runner.py (training)
* Author: Loic Martins
* Date: 2025-11-16
* Description:
    Manage the preparation of the dataset for the training.
* Functions:
    -run_dataset_preparation() --run the preparation pipeline
    -create_data_dict()
    -split_dataset()
"""

# Import Internal Libraries
import json

import pandas as pd
from datasets import Dataset

# Import Local Modules
from training.configurations.config import TrainingConfig


def split_dataset(tokenized_dataset: Dataset) -> tuple[Dataset, Dataset]:
    """
    Split the dataset into Train and Eval parts.

    Args:
        tokenized_dataset (Dataset): The tokenized dataset.

    Returns:
        split_datasets (tuple[Dataset, Dataset]): 2 datasets.
    """
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    return (train_dataset, eval_dataset)


def create_data_dict(dataframe: pd.DataFrame) -> list[dict[str, str]]:
    """
    Create a data dictionnary with: input (features), output (labels), for each row.

    Args:
        config (TrainingConfig): Specific configurations for the training.

    Returns:
        data_dict (list[dict[str, str]]): New data dictionnary --for each dict in the list: "input" and "output" keys.
    """

    data_dict: list[dict[str, str]] = [
        {
            "input": " ".join(
                f"{col}: {val}" for col, val in row.items() if col != "Label"
            )
            + "\nLabel:",
            "output": str(row["Label"]).strip(),
        }
        for _, row in dataframe.iterrows()
    ]

    del dataframe

    return data_dict


def run_dataset_preparation(
    config: TrainingConfig, dataframe: pd.DataFrame
) -> list[dict[str, str]]:
    """
    Run the entire data preparation pipeline:
        - Create a data dictionnary with: input (features), output (labels)

    Args:
        config (TrainingConfig): Specific configurations for the training.

    Returns:
        None -> Save ????
    """

    # Step 1: Create a data dictionnary
    data_dict: list[dict[str, str]] = create_data_dict(dataframe)

    # Step 2: Save preprocessed data
    output_path: str = f"{config.data_path}/data_dict.json"
    with open(output_path, "w") as f:
        json.dump(data_dict, f, indent=4)

    return data_dict
