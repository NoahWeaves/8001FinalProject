"""
* File: runner.py (eval)
* Author: Loic Martins
* Date: 2025-11-19
* Description:
    Run the entire eval pipeline.
"""

# Import external libraries
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# Import local modules
from model.setup import SetUpModel
from training.configurations.config import TrainingConfig
from training.preparation import run_dataset_preparation
from training.utils import save_json


def run_eval() -> None:
    """
    Run the entire eval pipeline:
        - Set up the configurations for the evaluation
        - Setup the fine-tuned model and the tokenizer

    Args:
        dataset_path (str): Path of the dataset.
        model_name (str): The name of the model used for the training.

    Returns:
        None -> Save ????
    """

    # Step 1: Set up the configurations for the evaluation
    config: TrainingConfig = TrainingConfig()
    print("Step 1 -> Configurations setup: Completed.\n\n")

    # Step 2: Setup the model and the tokenizer
    setup: SetUpModel = SetUpModel(
        config=config, fine_tuned_path=config.fine_tuned_path, training_bool=False
    )
    print(
        "Step 3 -> Initializing SetUpModel Completed.\n"
        f"Device = {setup.device}; Model = {setup.model_name}.\n\n"
    )

    # Step 3: Load the dataset
    dataframe = pd.read_csv(config.dataset_path)
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)
    print(
        "Step 4 -> Load the dataset: Completed.\n"
        f"Size of the dataset = {len(dataframe)}.\n\n"
    )

    # Step 4: Dataset Preparation
    data_list: list[dict[str, str]] = run_dataset_preparation(
        config=config, dataframe=dataframe
    )
    print(
        "Step 4 -> Dataset Preparation (List of dict): Completed.\n"
        f"Size of the dataset = {len(data_list)}.\n\n"
    )

    # Step 5: Start the loop to generate prediction
    total = 0
    y_true = []
    y_pred = []

    for example in data_list:
        # Step 5.1: Generate predictions
        gold = example["output"].lower().strip()
        pred = setup.generate(example["input"]).lower().strip()

        # Step 5.2: Strip the ground truth and predictions
        pred = pred[: len(gold)]
        gold = gold[: len(pred)]

        # Step 5.3: Add to the lists
        y_true.append(gold)
        y_pred.append(pred)
        total += 1

        print(f"Predictions: {total}/{len(data_list)}.")

    print("Step 5 -> Generate Predictions: Completed.")

    # Step 6: Compute the different evaluation metrics and save it
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=3, output_dict=True)
    # Combine into a single dictionary
    eval_metrics = {"accuracy": accuracy, "report": report}

    json_path: str = f"{config.output_path}/evaluation_results.json"
    save_json(path=json_path, data=eval_metrics)
    print("Step 6 -> Compute the different evaluation metrics + Save: Completed.")

    # Step 7: Generate a CSV files with the predictions and ground-truths
    eval_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    path: str = f"{config.output_path}/predictions.csv"
    eval_df.to_csv(path, index=False)

    print("Step 7 -> Predictions to CSV: Completed.")


# Main Function
if __name__ == "__main__":
    """
    Run the entire eval pipeline.
    """

    run_eval()
