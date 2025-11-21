"""
* File: utils.py (training)
* Author: Loic Martins
* Date: 2025-11-16
* Description:
    Utilities for different scripts.
* Functions:
    -tokens_size()
    -samples_size()
    -save_json()
    -tokenizer_sanity_check()
    -get_final_metrics()
    )
"""

# Import external libraries
import json

from datasets import Dataset
from transformers.trainer_utils import TrainOutput

# Import local modules
from model.setup import SetUpModel


def get_final_metrics(
    train_result: TrainOutput,
) -> dict:
    """
    Compute sequence-level metrics to evaluate the model:
        - Sequence level accuracy.
        - Token level accuracy.
        - F1-Score
        - Precision
        - Recall

    Args:
        train_result (TrainOutput): Train object containing metrics, and especially the Loss.
        eval_result (TrainOutput): Eval object containing evaluation metrics.

    Returns:
        metrics (dict): Dictionary containing the specific evaluation metrics.
    """
    final_metrics = {
        "train_loss": train_result.training_loss,
        "train_runtime": train_result.metrics.get("train_runtime"),
        "train_samples_per_second": train_result.metrics.get(
            "train_samples_per_second"
        ),
        "train_steps_per_second": train_result.metrics.get("train_steps_per_second"),
    }

    return final_metrics


def tokenizer_sanity_check(
    setup: SetUpModel, tokenized_dataset: Dataset
) -> dict[str, float | str]:
    """
    Validation helper designed to make sure a tokenizer and tokenized dataset are behaving as expected.

    Args:
        setup (SetUpModel): Specific setup containing the tokenizer.
        tokenized_dataset (Dataset): Dataset containing the data.

    Returns:
        data_dict (dict[str, float | str]): The dictionary containing the information concerning the tokenized data.
    """
    sample = tokenized_dataset[0]

    # Step 1: Filter out tokens where the label is -100 (ignored during training)
    output_ids = [
        id for id, lab in zip(sample["input_ids"], sample["labels"]) if lab != -100
    ]

    # Step 2: Decode the valid output tokens
    decoded_text = setup.tokenizer.decode(output_ids, skip_special_tokens=True)

    # Step 3: Build and return a dictionary instead of printing
    results = {
        "decoded_text": decoded_text,
        "input_ids_length": len(sample["input_ids"]),
        "labels_length": len(sample["labels"]),
        "attention_mask_length": len(sample["attention_mask"]),
        "pad_token": setup.tokenizer.pad_token,
        "pad_token_id": setup.tokenizer.pad_token_id,
        "eos_token": setup.tokenizer.eos_token,
        "eos_token_id": setup.tokenizer.eos_token_id,
    }

    return results


def tokens_size(tokenized_dataset: Dataset) -> dict[str, float]:
    """
    Calculate the length of the tokenized data.

    Args:
        data_list (list[dict[str, str]]): List of dictionary that contains the data.

    Returns:
        data_dict (dict[str, float]): The dictionary containing the information concerning the data.
    """
    # Compute token lengths for all samples
    token_lengths = [len(ex["input_ids"]) for ex in tokenized_dataset]  # pyright: ignore[reportArgumentType, reportCallIssue]

    # Compute stats
    average_length = sum(token_lengths) / len(token_lengths)
    min_length = min(token_lengths)
    max_length = max(token_lengths)

    data_dict = {
        "min_length": min_length,
        "max_length": max_length,
        "average_length": average_length,
    }

    return data_dict


def samples_size(data_list: list[dict[str, str]]) -> dict[str, float]:
    """
    Calculate the length of the data.

    Args:
        data_list (list[dict[str, str]]): List of dictionary that contains the data.

    Returns:
        data_dict (dict[str, float]): The dictionary containing the information concerning the data.
    """
    # Step 1: Compute different lengths
    total_length = sum(len(s["input"]) + len(s["output"]) for s in data_list)
    max_length = max([len(s["input"]) + len(s["output"]) for s in data_list])
    average_length = total_length / len(data_list) if data_list else 0
    total_length_output = sum(len(s["output"]) for s in data_list)
    max_length_output = max([len(s["output"]) for s in data_list])
    average_length_output = total_length_output / len(data_list) if data_list else 0

    # Step 2: Save in a dictionary
    data_dict = {
        "total_length": total_length,
        "max_length": max_length,
        "average_length": average_length,
        "total_length_output": total_length_output,
        "max_length_output": max_length_output,
        "average_length_output": average_length_output,
    }

    return data_dict


def save_json(path: str, data: dict) -> None:
    """
    Save Python dictionary into JSON format.

    Args:
        path (str): Where to save the data.
        data (dict): The dictionary to save.

    Returns:
        None
    """
    with open(path, "w") as file:
        json.dump(data, file, indent=4)

    print("Dataset Information saved.")
