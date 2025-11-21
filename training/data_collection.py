"""
* File: data_collection.py
* Author: Loic Martins
* Date: 2025-11-16
* Description:
    -Create a combined Dataset
    -Save a JSON file containing dataset information
"""

# Import external Libraries
import os

import pandas as pd

# Import Local Modules
from training.utils import save_json


def extract_dataset_info(dataframe: pd.DataFrame, train_bool: bool = True) -> None:
    """
    Collect Dataset information and save it in a JSON file.

    Args:
        dataframe (pd.DataFrame): Combined DataFrame.

    Returns:
        None -> Save a JSON file containing Dataset information.
    """

    # Step 1: Collect Information
    dataset_info = {
        "dataset_info": {
            "size": len(dataframe),
            "nb_features": int(dataframe.shape[1]),
            "total_nb_labels": int(dataframe["Label"].nunique()),
            "list_labels": dataframe["Label"].unique().tolist(),
            "labels": dataframe["Label"].value_counts().to_dict(),
            "column_types": {k: str(v) for k, v in dataframe.dtypes.to_dict().items()},
        }
    }

    # Step 2: Save in Json
    if train_bool:
        save_json(path="results/train_dataset_info.json", data=dataset_info)
    else:
        save_json(path="results/eval_dataset_info.json", data=dataset_info)


def clean_daframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the combined dataset.

    Args:
        dataframe (pd.DataFrame): Combined DataFrame.

    Returns:
        dataframe (pd.DataFrame): Cleaned version of the original combined DataFrame.
    """

    # Step 1: Rename column Label
    dataframe.rename(columns={" Label": "Label"}, inplace=True)

    # Step 2: Remove index column
    dataframe = dataframe.drop(columns=["Unnamed: 0"], errors="ignore")

    # Step 3: Remove WebDDoS because only 5 examples
    dataframe = dataframe[dataframe["Label"] != "WebDDoS"]  # pyright: ignore[reportAssignmentType]

    # Step 4: Reset the index
    dataframe = dataframe.reset_index(drop=True)

    return dataframe


def run_data_collection(
    data_path: str, output_path_train: str, output_path_eval: str
) -> None:
    """
    Run the data collection process to combined datasets and collect dataset information.

    Args:
        data_path (str): Path of the datasets.
        output_path (str): Specific output path for the combined dataset.

    Returns:
        None -> Save a combined dataset in CSV format.
    """

    all_samples_train: list[pd.DataFrame] = []
    all_samples_eval: list[pd.DataFrame] = []
    all_samples_benign: list[pd.DataFrame] = []

    # Step 1: Iterate throught each csv file and get 2,000 rows to create a new dataframe

    for file in os.listdir(data_path):
        if not file.endswith(".csv"):
            continue  # skip non-CSV files

        dataframe: pd.DataFrame = pd.read_csv(os.path.join(data_path, file))

        # Step 1.1: Get BENIGN labels
        benign_df: pd.DataFrame = dataframe[dataframe[" Label"].str.strip() == "BENIGN"]  # pyright: ignore[reportAssignmentType]

        # Step 1.2: Get attack samples
        attack_df: pd.DataFrame = dataframe[dataframe[" Label"].str.strip() != "BENIGN"]  # pyright: ignore[reportAssignmentType]

        if len(attack_df) > 2000:
            sample: pd.DataFrame = attack_df.sample(n=2000, random_state=42)
            remaining_attack: pd.DataFrame = attack_df.drop(index=sample.index.tolist())
            remaining_attack = remaining_attack.sample(n=2000, random_state=42)
            all_samples_eval.append(remaining_attack)
        else:
            sample = attack_df

        all_samples_train.append(sample)

        if len(benign_df) > 0:
            all_samples_benign.append(benign_df)

    # Step 2: Select only 2,000 BENIGN rows
    benign_df = pd.concat(all_samples_benign, ignore_index=True)
    # Step 2.1: Randomly sample 2,000 BENIGN rows for the train dataset
    benign_sample_train = benign_df.sample(n=2000, random_state=42)
    all_samples_train.append(benign_sample_train)
    # Step 2.2: Select teh remaining rows for the eval dataset
    benign_remaining = benign_df.drop(benign_sample_train.index.tolist())
    benign_remaining = benign_remaining.sample(n=2000, random_state=42)
    all_samples_eval.append(benign_remaining)

    # Step 2: Combined all the DataFrames and clean
    combined_df_train: pd.DataFrame = pd.concat(all_samples_train, ignore_index=True)
    combined_df_eval: pd.DataFrame = pd.concat(all_samples_eval, ignore_index=True)
    final_dataframe_train: pd.DataFrame = clean_daframe(combined_df_train)
    final_dataframe_eval: pd.DataFrame = clean_daframe(combined_df_eval)

    # Step 3: Save as CSV file
    final_dataframe_train.to_csv(output_path_train, index=False)
    final_dataframe_eval.to_csv(output_path_eval, index=False)

    # Step 4: Save some information about the dataset
    extract_dataset_info(final_dataframe_train)
    extract_dataset_info(final_dataframe_eval, train_bool=False)


# Main Function
if __name__ == "__main__":
    """
    Run the Data Collection process.
    """
    data_path: str = "data/CIC-DDoS2019"
    output_path_train: str = "./training/data/combined_dataset_train.csv"
    output_path_eval: str = "./training/data/combined_dataset_eval.csv"

    run_data_collection(
        data_path=data_path,
        output_path_train=output_path_train,
        output_path_eval=output_path_eval,
    )
