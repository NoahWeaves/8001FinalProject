# make_data_only.py - Data preprocessing and splitting script
import os, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import logging
import gc
import builtins
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import glob
from joblib import dump
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Global logger instance
logger = None

def setup_logger(log_path: str | Path, name: str = "data_prep"):
    global logger
    lp = Path(log_path)
    lp.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    fh = logging.FileHandler(lp, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
        
    return logger

def log_print(*args, **kwargs):
    global logger
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    message = sep.join(str(a) for a in args)

    builtins.print(message, end=end)

    if logger:
        logger.info(message.rstrip("\n"))

print = log_print

def sample_data_stratified(df, target_col="Label", max_samples=1_000_000, min_class_samples=1000, random_state=42):
    """
    Stratified sampling to reduce dataset size while preserving class distribution.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        max_samples: Maximum total samples to keep
        min_class_samples: Minimum samples per class
        random_state: Random seed for reproducibility
    
    Returns:
        Sampled DataFrame
    """
    print(f"Original dataset size: {len(df):,} rows")
    
    if len(df) <= max_samples:
        print("Dataset is already within the target size, no sampling needed.")
        return df
    
    # Get class distribution
    class_counts = df[target_col].value_counts()
    print(f"Class distribution before sampling:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count:,} samples ({count/len(df)*100:.2f}%)")
    
    # Calculate samples per class (proportional)
    sampling_ratio = max_samples / len(df)
    samples_per_class = {}
    
    for cls, count in class_counts.items():
        n_samples = max(min_class_samples, int(count * sampling_ratio))
        # Don't sample more than available
        n_samples = min(n_samples, count)
        samples_per_class[cls] = n_samples
    
    # Adjust if total exceeds max_samples
    total_samples = sum(samples_per_class.values())
    if total_samples > max_samples:
        scale_factor = max_samples / total_samples
        samples_per_class = {
            cls: max(min_class_samples, int(n * scale_factor))
            for cls, n in samples_per_class.items()
        }
    
    print(f"\nSampling plan:")
    for cls, n in samples_per_class.items():
        print(f"  {cls}: {n:,} samples")
    
    # Sample each class
    sampled_dfs = []
    for cls, n_samples in samples_per_class.items():
        cls_df = df[df[target_col] == cls]
        if len(cls_df) > n_samples:
            sampled = cls_df.sample(n=n_samples, random_state=random_state)
        else:
            sampled = cls_df
        sampled_dfs.append(sampled)
    
    result = pd.concat(sampled_dfs, ignore_index=True)
    # Shuffle the result
    result = result.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"\nSampled dataset size: {len(result):,} rows ({len(result)/len(df)*100:.2f}% of original)")
    print(f"Class distribution after sampling:")
    for cls, count in result[target_col].value_counts().items():
        print(f"  {cls}: {count:,} samples ({count/len(result)*100:.2f}%)")
    
    return result

def load_data(max_samples=1_000_000, min_class_samples=1000, sample_per_file=False, RANDOM_STATE=42):
    # Function to rename columns by removing leading or trailing spaces
    def rename_columns(df):
        df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
        return df 

    print("Loading data...")
    csv_paths = glob.glob("data/train/*.csv")  # LDAP, MSSQL, NetBIOS, SYN, UDP, UDPlag
    
    if sample_per_file:
        # Sample each file independently before concatenation
        print(f"Sampling strategy: {max_samples:,} samples per file")
        df_list = []
        for path in tqdm(csv_paths):
            print(f"Loading: {path}")
            temp = pd.read_csv(path, low_memory=True)
            temp = rename_columns(temp)
            filename_ = path.split('/')[-1].split('\\')[-1]
            if filename_ == "UDPLag.csv":
                temp = temp[temp["Label"] != "WebDDoS"]
                print(f"Loaded {temp.shape[0]:,} samples from {filename_} (WebDDoS removed)")
            temp["Scenario"] = Path(path).stem
            
            # Sample this file if too large
            if len(temp) > max_samples:
                temp = sample_data_stratified(temp, target_col="Label", 
                                            max_samples=max_samples, 
                                            min_class_samples=min_class_samples,
                                            random_state=RANDOM_STATE)
            df_list.append(temp)
    else:
        # Load all files then sample the combined dataset
        print(f"Sampling strategy: {max_samples:,} total samples across all files")
        df_list = []
        for path in tqdm(csv_paths):
            print(f"Loading: {path}")
            temp = pd.read_csv(path, low_memory=True)
            temp = rename_columns(temp)
            filename_ = path.split('/')[-1].split('\\')[-1]
            if filename_ == "UDPLag.csv":
                temp = temp[temp["Label"] != "WebDDoS"]
                print(f"Loaded {temp.shape[0]:,} samples from {filename_} (WebDDoS removed)")
            df_list.append(temp)
    
    df = pd.concat(df_list, ignore_index=True)
    
    # Sample combined dataset if needed and not already sampled per file
    if not sample_per_file and len(df) > max_samples:
        df = sample_data_stratified(df, target_col="Label", 
                                   max_samples=max_samples, 
                                   min_class_samples=min_class_samples,
                                   random_state=RANDOM_STATE)

    # Drop irrelevant identifiers
    drop_cols = [
        "Unnamed: 0", "Flow ID", "Source IP", "Destination IP",
        "SimillarHTTP", "Inbound"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    ## Handle infinities
    df = df.replace([np.inf, -np.inf], np.nan)

    ## Parse timestamps and build groups
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    groups = df["Timestamp"].dt.floor("min")  # group by minute
    ## if we ever want to change the granularity:
    # groups = df["Timestamp"].dt.floor("h")   # hour
    # groups = df["Timestamp"].dt.floor("s")   # second
    # groups = df["Timestamp"].dt.floor("5min")  # 5-minute windows
    
    # Separate features and target
    X = df.drop(columns=["Label", "Timestamp", "Scenario"], errors="ignore")
    y = df["Label"]
    
    print(f"Data loaded: {X.shape[0]:,} samples, {X.shape[1]} features")

    return X, y, groups


def main():
    global logger
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/data_prep_{timestamp}.log"
    logger = setup_logger(log_file, name="data_prep")
    
    print("="*80)
    print("Data Preprocessing and Splitting Pipeline")
    print("="*80)
    
    # Configuration
    MAX_SAMPLES = 10_000
    MIN_CLASS_SAMPLES = 4000
    SAMPLE_PER_FILE = True
    
    print(f"\nDataset Configuration:")
    print(f"  Max samples: {MAX_SAMPLES:,}")
    print(f"  Min samples per class: {MIN_CLASS_SAMPLES:,}")
    print(f"  Sampling strategy: {'per-file' if SAMPLE_PER_FILE else 'combined'}")
    
    # Create output directory
    outdir = Path("data_processed") / f"data_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {outdir}")

    # Load and preprocess data
    X, y, groups = load_data(
        max_samples=MAX_SAMPLES, 
        min_class_samples=MIN_CLASS_SAMPLES, 
        sample_per_file=SAMPLE_PER_FILE
    )
    
    feature_names = X.columns.tolist()
    n_features = X.shape[1]
    
    # Encode labels
    le = LabelEncoder()
    le.fit(y.astype(str))
    y_encoded = le.transform(y.astype(str))
    
    num_classes = len(le.classes_)
    # Convert numpy int64 to Python int for JSON serialization
    label_map = {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}
    
    print(f"\nLabel mapping: {label_map}")
    print(f"Number of classes: {num_classes}")
    print(f"Number of features: {n_features}")

    # Train/test split
    if groups is not None:
        print("\nPerforming group-based train/test split...")
        unique_g = pd.Series(groups).drop_duplicates()
        holdout_frac = 0.2
        n_hold = max(1, int(len(unique_g) * holdout_frac))
        hold_groups = set(unique_g.sample(n_hold, random_state=RANDOM_STATE))
        mask_hold = pd.Series(groups).isin(hold_groups).values
        
        X_train, X_test = X.loc[~mask_hold], X.loc[mask_hold]
        y_train, y_test = y_encoded[~mask_hold], y_encoded[mask_hold]
        groups_train = pd.Series(groups)[~mask_hold].values
        groups_test = pd.Series(groups)[mask_hold].values
    else:
        print("\nPerforming random train/test split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
        )
        groups_train = None
        groups_test = None
    
    print(f"Train set: {X_train.shape[0]:,} samples")
    print(f"Test set: {X_test.shape[0]:,} samples")

    # Save processed data
    print(f"\nSaving processed data to {outdir}...")
    
    X_train.to_csv(outdir / "X_train.csv", index=False)
    X_test.to_csv(outdir / "X_test.csv", index=False)
    pd.DataFrame(y_train, columns=["Label"]).to_csv(outdir / "y_train.csv", index=False)
    pd.DataFrame(y_test, columns=["Label"]).to_csv(outdir / "y_test.csv", index=False)
    
    if groups_train is not None:
        pd.DataFrame(groups_train, columns=["Group"]).to_csv(outdir / "groups_train.csv", index=False)
        pd.DataFrame(groups_test, columns=["Group"]).to_csv(outdir / "groups_test.csv", index=False)
    
    # Save label encoder and metadata
    dump(le, outdir / "label_encoder.joblib")
    
    metadata = {
        "n_samples_train": int(X_train.shape[0]),
        "n_samples_test": int(X_test.shape[0]),
        "n_features": n_features,
        "num_classes": num_classes,
        "feature_names": feature_names,
        "label_mapping": label_map,
        "random_state": RANDOM_STATE,
        "timestamp": timestamp,
        "has_groups": groups_train is not None,
    }
    
    with open(outdir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*80)
    print("Data Processing Complete!")
    print("="*80)
    print(f"Files saved:")
    print(f"  - X_train.csv ({X_train.shape[0]:,} rows, {X_train.shape[1]} columns)")
    print(f"  - X_test.csv ({X_test.shape[0]:,} rows, {X_test.shape[1]} columns)")
    print(f"  - y_train.csv")
    print(f"  - y_test.csv")
    if groups_train is not None:
        print(f"  - groups_train.csv")
        print(f"  - groups_test.csv")
    print(f"  - label_encoder.joblib")
    print(f"  - metadata.json")
    print(f"\nLog file: {log_file}")


if __name__ == "__main__":
    main()
