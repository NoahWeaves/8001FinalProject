"""
* File: runner.py (training)
* Author: Loic Martins
* Date: 2025-11-16
* Description:
    Run the entire training pipeline.
"""

# Import external libraries
import os
from dataclasses import asdict
from functools import partial

import pandas as pd
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
from transformers.trainer_utils import TrainOutput

# Import local modules
from model.setup import SetUpModel
from training.configurations.arguments import TrainingArgs
from training.configurations.config import TrainingConfig
from training.preparation import (
    run_dataset_preparation,
    split_dataset,
)
from training.tokenize import tokenize_example
from training.utils import (
    get_final_metrics,
    samples_size,
    save_json,
    tokenizer_sanity_check,
    tokens_size,
)

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


def run_training_pipeline() -> None:
    """
    Run the entire training pipeline:
        - Set up the configurations for the training
        - Setup the model, tokenizer and apply LoRA to the model
        - Load the data --dataframe
        - Prepare the data --data dictionary
        - Create specific Dataset format for PyTorch
        - Tokenize the data
        - Apply Data Collator
        - Train the model

    Args:
        dataset_path (str): Path of the dataset.
        model_name (str): The name of the model used for the training.

    Returns:
        None -> Save ????
    """
    print("======Training Pipeline: Start======\n\n")

    # Step 1: Set some variables
    training_info: dict = {}
    print("Step 2 -> Training Info Dictionary setup: Completed.\n\n")

    # Step 2: Set up the configurations for the training
    config: TrainingConfig = TrainingConfig()
    print("Step 2 -> Configurations setup: Completed.\n\n")

    # Step 3: Setup the model, tokenizer and apply LoRA to the model
    setup: SetUpModel = SetUpModel(config=config, training_bool=True)
    setup.setup_lora()
    print(
        "Step 3 -> Initializing SetUpModel + LoRA: Completed.\n"
        f"Device = {setup.device}; Model = {setup.model_name}.\n\n"
    )

    # Step 4: Load the dataset
    dataframe = pd.read_csv(config.dataset_path)
    print(
        "Step 4 -> Load the dataset: Completed.\n"
        f"Size of the dataset = {len(dataframe)}.\n\n"
    )

    # Step 5.1: Dataset Preparation
    data_list: list[dict[str, str]] = run_dataset_preparation(
        config=config, dataframe=dataframe
    )
    # Step 5.2: Save some information about the data --size
    training_info["samples_size"] = samples_size(data_list=data_list)
    print(
        "Step 5 -> Dataset Preparation (List of dict): Completed.\n"
        f"Size of the dataset = {len(data_list)}.\n\n"
    )

    # Step 6.1: Create specific Dataset format for PyTorch
    hf_dataset: Dataset = Dataset.from_list(data_list)
    # Step 6.2: Save information about the data
    training_info["hf_dataset_sample"] = hf_dataset[0]
    print("Step 6 -> Create specific Dataset format: Completed\n\n.")

    # Step 7.1: Tokenize the data using tokenize_data function
    setup.tokenizer.truncation_side = "left"  # because we want to keep the full label
    tokenized_dataset: Dataset = hf_dataset.map(
        partial(tokenize_example, tokenizer=setup.tokenizer),
        batched=False,
        remove_columns=hf_dataset.column_names,
    )

    # Step 7.2: Save some information about the tokenized data --size
    training_info["tokenized_sample"] = tokenized_dataset[0]
    training_info["tokens_size"] = tokens_size(tokenized_dataset=tokenized_dataset)

    # Step 7.3: Sanity Check --Print the output part to check if the mask -100 for the loss is right
    training_info["tokenizer_sanity_check"] = tokenizer_sanity_check(
        setup=setup, tokenized_dataset=tokenized_dataset
    )
    print("Step 7 -> Tokenize the data: Completed.\n\n")

    # Step 8: Apply arguments
    training_args: TrainingArgs = TrainingArgs()
    training_arguments: TrainingArguments = TrainingArguments(**vars(training_args))
    # Save training arguments
    training_info["training_arguments"] = asdict(training_args)

    print("Step 8 -> Apply arguments: Completed.\n\n")

    # Step 9: Create the data collator for padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=setup.tokenizer,
        model=setup.model,
        padding=True,
    )
    print("Step 9 -> Apply data collator: Completed.\n\n")

    # Step 11: Train the model
    # Handle Training + Evaluation --add  a compute_metrics function and pass it to the Trainer.
    trainer = Trainer(
        model=setup.model,
        args=training_arguments,
        train_dataset=tokenized_dataset,
        processing_class=setup.tokenizer,
        data_collator=data_collator,
        # compute_metrics=compute_metrics,  # pyright: ignore[reportArgumentType]
    )
    train_result: TrainOutput = trainer.train()
    # eval_result = trainer.evaluate()

    print("Step 11 -> Training the model: Completed\n\n")

    # Step 12: Get training and evaluation metrics
    final_metrics: dict = get_final_metrics(train_result=train_result)
    trainer.save_model(TrainingArgs.output_dir)

    print("Step 12 -> Save the model and training/evaluation metrics: Completed\n\n")

    print("======Training Pipeline: Completed======\n\n")

    # Step 13: Save data into JSON
    save_json(path=config.output_path, data=final_metrics)
    save_json(path=config.output_path, data=trainer.state.log_history)  # pyright: ignore[reportArgumentType]
    save_json(path=config.output_path, data=training_info)


# Main Function
if __name__ == "__main__":
    """
    Run the entire training pipeline.
    """

    run_training_pipeline()
