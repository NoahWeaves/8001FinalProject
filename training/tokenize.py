"""
* File: tokenize.py (training)
* Author: Loic Martins
* Date: 2025-11-16
* Description:
    Tokenize the data.
"""

# Import external libraries
from typing import cast

from transformers import BatchEncoding, PreTrainedTokenizerBase


def tokenize_example(
    example: dict[str, str], tokenizer: PreTrainedTokenizerBase
) -> dict[str, list]:
    """
    Tokenize the data.

    Args:
        example (dict[str, str]): 1 example of the data with "input" and "output".
        tokenizer (PreTrainedTokenizerBase): Specific tokenizer for the model.

    Returns:
        example_dict (dict[str, list]) -> Tokenized example with a specific format: "input_ids", "attention_mask", "labels"
    """
    # Step 1: Get the Input text
    input_text = example["input"] + "\n"

    # Step 2: Create the Full text (input + target)
    full_text = input_text + example["output"]

    # Step 3: Tokenize both
    input_tokens: BatchEncoding = tokenizer(
        input_text,
        add_special_tokens=True,
        truncation=True,
        max_length=1024,
    )
    full_tokens: BatchEncoding = tokenizer(
        full_text,
        add_special_tokens=True,
        truncation=True,
        max_length=1024,
    )

    input_tokens_dict = cast(dict[str, list[int]], input_tokens)
    full_tokens_dict = cast(dict[str, list[int]], full_tokens)

    # Step 4: Create labels: --Specific to only predict o_t+1
    # -100 for input tokens, actual token ids for output
    labels = [-100] * len(input_tokens_dict["input_ids"]) + full_tokens_dict[
        "input_ids"
    ][len(input_tokens_dict["input_ids"]) :]

    return {
        "input_ids": full_tokens_dict["input_ids"],
        "attention_mask": full_tokens_dict["attention_mask"],
        "labels": labels,
    }
