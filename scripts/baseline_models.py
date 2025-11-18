# experiment_Classifiers.py
import os, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import logging
import sys
import gc
import builtins
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from joblib import dump
from tqdm import tqdm
import warnings
# Optional imports
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    HAVE_BAYES = True
except Exception:
    HAVE_BAYES = False
    print("skopt not available, skipping related tests.")

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False
    print("XGBoost not available, skipping related tests.")

# GPU-specific imports
try:
    import torch
    import torch.nn as nn
    from skorch import NeuralNetClassifier
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False
    print("PyTorch/skorch not available, falling back to sklearn MLP.")

try:
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.neighbors import KNeighborsClassifier as cuKNN
    from cuml.svm import SVC as cuSVC
    import cupy as cp
    HAVE_CUML = True
except Exception:
    HAVE_CUML = False
    print("RAPIDS cuML not available, falling back to sklearn models.")

warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

## do not display DtypeWarning from Pandas
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    warnings.filterwarnings("ignore", category=PerformanceWarning, 
                            message=".*pinned memory.*could not be allocated.*")
except:
    pass
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.width', 1000)
# pd.set_option('display.max_colwidth', None)

# Global logger instance
logger = None

def setup_logger(log_path: str | Path, name: str = "baseline"):
    global logger  # Declare we're modifying the global logger
    lp = Path(log_path)
    lp.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False            # <- don't bubble to root (prevents dupes)
    logger.handlers.clear()             # <- avoid stacking handlers on repeated setup

    fh = logging.FileHandler(lp, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
        
    return logger

def log_print(*args, **kwargs):
    global logger
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    message = sep.join(str(a) for a in args)

    # print once to console
    builtins.print(message, end=end)

    # also write to file (no console handler attached → no dupes)
    if logger:
        logger.info(message.rstrip("\n"))

# Replace the built-in print with our logging version
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

def print_device_info():
    try:
        import torch
        print("[GPU] PyTorch CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("       CUDA device:", torch.cuda.get_device_name(0))
            print("       CUDA version:", torch.version.cuda)
    except Exception:
        print("[GPU] PyTorch not installed")

    try:
        import xgboost as xgb
        print("[GPU] XGBoost version:", xgb.__version__)
    except Exception:
        print("[GPU] XGBoost not installed")
    
    try:
        import cuml
        print("[GPU] RAPIDS cuML version:", cuml.__version__)
        import cupy as cp
        print("       CuPy available:", cp.cuda.is_available())
    except Exception:
        print("[GPU] RAPIDS cuML not installed")

def make_output_dir(base="runs"):
    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path(base) / f"models/classif_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def bin_for_stratification(y, n_bins=10):
    # bins for approximate stratification on a continuous target
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(y, quantiles))
    y_binned = np.digitize(y, edges[1:-1], right=True)
    return y_binned

def get_cv(y, groups=None, n_splits=5, seed=42):
    if groups is not None:
        return GroupKFold(n_splits=n_splits, random_state=seed)
    else:
        y_bins = bin_for_stratification(y, n_bins=10)
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed), y_bins

def primary_scorer():
    return "f1_weighted"  # Use weighted for imbalanced data reflecting real distribution

def scoring_dict():
    return {
        "F1_macro": make_scorer(f1_score, average="macro"),
        "F1_weighted": make_scorer(f1_score, average="weighted"),
        "Precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
        "Precision_weighted": make_scorer(precision_score, average="weighted", zero_division=0),
        "Recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
        "Recall_weighted": make_scorer(recall_score, average="weighted", zero_division=0),
        "Accuracy": make_scorer(accuracy_score),
    }

def numeric_preprocessor(scaler_type: str | None = "standard", impute: bool = True):
    steps = []
    if impute:
        steps.append(("impute", SimpleImputer(strategy="median")))
    if scaler_type:
        if scaler_type == "standard":
            steps.append(("scale", StandardScaler()))
        elif scaler_type == "minmax":
            steps.append(("scale", MinMaxScaler()))
        elif scaler_type == "robust":
            steps.append(("scale", RobustScaler()))
        else:
            raise ValueError(f"Unknown scaler_type: {scaler_type}")
    
    # Return 'passthrough' if no steps, otherwise return Pipeline
    if len(steps) == 0:
        return 'passthrough'
    return Pipeline(steps)

def build_selector(k):
    return SelectKBest(score_func=mutual_info_classif, k=k)

def k_grid_from_dim(n_features):
    # progressive MI sizes; ensure uniqueness and <= n_features
    candidates = [8, 16, 32, 48, 64, n_features]
    ks = sorted(list({min(k, n_features) for k in candidates if k <= n_features}))
    return ks


def logistic_space():
    if HAVE_BAYES:
        return [
            ("LogisticRegression",
             LogisticRegression(
                 multi_class="multinomial",
                 solver="saga",
                 penalty="elasticnet",
                 max_iter=2000,
                 random_state=RANDOM_STATE
             ),
             {
                 "model__C": Real(0.1, 10.0, prior="log-uniform"),
                 "model__l1_ratio": Real(0.0, 1.0),
                 "model__class_weight": Categorical([None, "balanced"]),
             })
        ]
    else:
        return [
            ("LogisticRegression",
             LogisticRegression(
                 multi_class="multinomial",
                 solver="saga",
                 penalty="elasticnet",
                 max_iter=2000,
                 random_state=RANDOM_STATE
             ),
             {
                 "model__C": [0.1, 1.0, 10.0],
                 "model__l1_ratio": [0.0, 0.5, 1.0],
                 "model__class_weight": [None, "balanced"],
             })
        ]

def rf_space():
    if HAVE_BAYES:
        return {
            "model__n_estimators": Integer(50, 200),
            "model__max_depth": Integer(3, 10),
            "model__min_samples_split": Integer(2, 5),
            "model__min_samples_leaf": Integer(1, 3),
            "model__max_features": Categorical(["sqrt", "log2"]),
        }
    else:
        return {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": ["sqrt", "log2"],
        }

def rf_space_gpu():
    """Random Forest search space for cuML GPU version"""
    if HAVE_BAYES:
        return {
            "model__n_estimators": Integer(50, 200),
            "model__max_depth": Integer(3, 10),
            "model__min_samples_split": Integer(2, 5),
            "model__max_features": Real(0.5, 1.0),  # cuML uses float instead of string
        }
    else:
        return {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [5, 10, 16],
            "model__min_samples_split": [2, 5],
            "model__max_features": [0.5, 0.7, 1.0],
        }

def xgb_space():
    if HAVE_BAYES:
        return {
            "model__n_estimators": Integer(50, 200),
            "model__max_depth": Integer(3, 6),
            "model__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            "model__subsample": Real(0.8, 1.0),
        }
    else:
        return {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [3, 5, 6],
            "model__learning_rate": [0.01, 0.1, 0.3],
            "model__subsample": [0.8, 1.0],
        }

def knn_space():
    if HAVE_BAYES:
        return {
            "model__n_neighbors": Integer(3, 15),
            "model__weights": Categorical(["uniform", "distance"]),
            "model__p": Integer(1, 2),
        }
    else:
        return {
            "model__n_neighbors": [3, 5, 7, 11, 15],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        }

def knn_space_gpu():
    """KNN search space for cuML GPU version"""
    if HAVE_BAYES:
        return {
            "model__n_neighbors": Integer(3, 15),
            "model__weights": Categorical(["uniform"]),  # cuML only supports uniform
            "model__p": Integer(1, 2),
        }
    else:
        return {
            "model__n_neighbors": [3, 5, 7, 11, 15],
            "model__weights": ["uniform"],  # cuML only supports uniform
            "model__p": [1, 2],
        }

def mlp_space():
    if HAVE_BAYES:
        return {
            "model__hidden_layer_sizes": Categorical([(64,), (128,)]),
            "model__activation": Categorical(["relu", "tanh"]),
            "model__alpha": Real(1e-4, 1e-2, prior="log-uniform"),
            "model__learning_rate_init": Real(1e-3, 1e-2, prior="log-uniform"),
        }
    else:
        return {
            "model__hidden_layer_sizes": [(64,), (128,)],
            "model__activation": ["relu", "tanh"],
            "model__alpha": [1e-4, 1e-3, 1e-2],
            "model__learning_rate_init": [1e-3, 5e-3, 1e-2],
        }
    
def mlp_space_gpu(input_dim, num_classes):
    """MLP search space for PyTorch GPU version"""
    if HAVE_BAYES:
        return {
            "model__module__hidden_dim": Categorical([64, 128, 256]),
            "model__module__dropout": Real(0.1, 0.5),
            "model__lr": Real(1e-4, 1e-2, prior="log-uniform"),
            "model__batch_size": Categorical([64, 128, 256]),
            # "model__module__hidden_dim": Categorical([64]),
            # "model__module__dropout": Real(0.1,0.2),
            # "model__lr": Real(1e-3, 1e-2, prior="log-uniform"),
            # "model__batch_size": Categorical([64]),            
        }
    else:
        return {
            "model__module__hidden_dim": [64, 128, 256],
            "model__module__dropout": [0.2, 0.3, 0.5],
            "model__lr": [1e-4, 1e-3, 1e-2],
            "model__batch_size": [64, 128, 256],
        }

def svc_space_classifier():
    if HAVE_BAYES:
        # Bayesian search space (scikit-optimize)
        return {
            "model__kernel": Categorical(["rbf"]),
            # "model__C": Real(1e-3, 1e2, prior="log-uniform"),
            # gamma is relevant for 'rbf'; it's ignored for 'linear' (safe to include)
            "model__gamma": Categorical(["scale", "auto"]),
            "model__class_weight": Categorical(["balanced"]),
        }
    else:
        # Randomized search grid
        return {
            "model__kernel": ["rbf"],
            "model__C": np.logspace(-3, 2, 3),  # 1e-3, 1e-2, 1e-1, 1, 10, 100 
            "model__gamma": ["auto"],  # used when kernel='rbf'
            "model__class_weight": ["balanced"],
        }

def svc_space_gpu():
    """SVC search space for cuML GPU version"""
    if HAVE_BAYES:
        return {
            "model__kernel": Categorical(["rbf"]),
            "model__C": Real(0.1, 10.0, prior="log-uniform"),
            "model__gamma": Categorical(["scale", "auto"]),
        }
    else:
        return {
            "model__kernel": ["rbf"],
            "model__C": [0.1, 1.0, 10.0],
            "model__gamma": ["scale", "auto"],
        }

# PyTorch MLP for GPU
class TorchMLP(nn.Module):
    """PyTorch MLP for GPU acceleration"""
    def __init__(self, input_dim=64, hidden_dim=128, num_classes=2, dropout=0.2):
        super().__init__()
        # input_dim will be dynamically set by skorch based on data shape
        # Default to 64 but this will be overridden when fit() is called
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Replace BatchNorm with LayerNorm (works with batch_size=1)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln2 = nn.LayerNorm(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, x):
        # Convert to float32 if needed (sklearn passes float64)
        if x.dtype == torch.float64:
            x = x.float()
        x = torch.relu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def build_pipeline(model, scale_for_model, n_features, scaler_type="standard", impute=True):
    pre = numeric_preprocessor(scaler_type if scale_for_model else None, impute=impute)
    selector = build_selector(k=n_features)  # tuned via search
    
    # Build pipeline steps conditionally
    steps = []
    if pre != 'passthrough':
        steps.append(("pre", pre))
    steps.extend([
        ("select", selector),
        ("model", model),
    ])
    
    return Pipeline(steps)


def print_memory_usage():
    """Print current memory usage"""
    import psutil
    process = psutil.Process()
    mem_info = process.memory_info()
    mem_mb = mem_info.rss / 1024 / 1024
    print(f"[Memory] Process RAM usage: {mem_mb:.2f} MB")
    
    if HAVE_TORCH and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        print(f"[Memory] GPU allocated: {allocated:.2f} MB, reserved: {reserved:.2f} MB")

def cleanup_memory():
    """Force garbage collection and clear GPU cache"""
    gc.collect()
    
    if HAVE_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    if HAVE_CUML:
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            # Add pinned memory pool cleanup
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

def search_cv_for_model(name, pipe, param_space, ks, X, y, groups, outdir, n_iter=60):
    """Search with explicit cleanup after fitting"""
    # Reduce iterations for very large datasets to prevent memory issues
    actual_n_iter = min(n_iter, 40) if len(X) > 500_000 else n_iter
    if actual_n_iter < n_iter:
        print(f"Reducing search iterations from {n_iter} to {actual_n_iter} due to dataset size")
    
    # Add MI k to the search space
    if HAVE_BAYES and isinstance(param_space, dict):
        # Bayes: add k as a categorical dimension
        from skopt.space import Categorical
        param_space = dict(param_space)  # copy
        param_space["select__k"] = Categorical(ks)
        # CV object
        if groups is not None:
            cv = GroupKFold(n_splits=5)
            cv_groups = groups
        else:
            cv, y_bins = get_cv(y, groups=None, n_splits=5, seed=RANDOM_STATE)
            cv_groups = None
        search = BayesSearchCV(
            estimator=pipe,
            search_spaces=param_space,
            n_iter=actual_n_iter,
            cv=cv,
            scoring=primary_scorer(),
            n_jobs=-1,
            random_state=RANDOM_STATE,
            refit=True,
            verbose=1,
        )
        search.fit(X, y, **({"groups": cv_groups} if cv_groups is not None else {}))
    else:
        # Randomized search
        param_dist = {}
        for k, v in param_space.items():
            param_dist[k] = v
        param_dist["select__k"] = ks
        if groups is not None:
            cv = GroupKFold(n_splits=5)
            cv_groups = groups
        else:
            cv, y_bins = get_cv(y, groups=None, n_splits=5, seed=RANDOM_STATE)
            cv_groups = None
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_dist,
            n_iter=min(actual_n_iter, 100),
            cv=cv,
            scoring=primary_scorer(),
            n_jobs=-1,
            random_state=RANDOM_STATE,
            refit=True,
            verbose=1,
        )
        search.fit(X, y, **({"groups": cv_groups} if cv_groups is not None else {}))

    # Extract best estimator before cleanup
    best_estimator = search.best_estimator_
    best_params = search.best_params_
    cv_results = pd.DataFrame(search.cv_results_)
    
    # Delete search object to free memory
    del search
    if 'cv_groups' in locals():
        del cv_groups
    if 'cv' in locals():
        del cv
    gc.collect()

    # Save best model and results
    model_dir = outdir / f"{name}"
    model_dir.mkdir(parents=True, exist_ok=True)

    dump(best_estimator, model_dir / "best_model.joblib")
    cv_results.to_csv(model_dir / "cv_results.csv", index=False)
    with open(model_dir / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    
    # Clean up results dataframe
    del cv_results
    gc.collect()

    return best_estimator

def evaluate_on_holdout(model, X_test, y_test, outdir, label):
    preds = model.predict(X_test)

    metrics = {
        "Accuracy": float(accuracy_score(y_test, preds)),
        "F1_macro": float(f1_score(y_test, preds, average="macro", zero_division=0)),
        "F1_weighted": float(f1_score(y_test, preds, average="weighted", zero_division=0)),
        "Precision_macro": float(precision_score(y_test, preds, average="macro", zero_division=0)),
        "Precision_weighted": float(precision_score(y_test, preds, average="weighted", zero_division=0)),
        "Recall_macro": float(recall_score(y_test, preds, average="macro", zero_division=0)),
        "Recall_weighted": float(recall_score(y_test, preds, average="weighted", zero_division=0)),
    }

    # Save predictions
    pd.DataFrame({"y_true": y_test, "y_pred": preds}).to_csv(outdir / f"{label}_preds.csv", index=False)

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", annot=False)
    plt.title(f"Confusion Matrix – {label}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(outdir / f"{label}_confusion_matrix.png")
    plt.close()

    # Classification report (precision, recall, f1 per class)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(outdir / f"{label}_classification_report.csv")

    # Save metrics summary
    with open(outdir / f"{label}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Clean up intermediate variables
    del preds, cm, report
    gc.collect()

    return metrics

def create_comparison_visualizations(summary_df, outdir):
    """
    Create comprehensive comparison charts for all baseline models
    """
    print(f"\nGenerating comparison visualizations...")
    
    # 1. Bar chart comparing all metrics across models
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    metrics = ['Accuracy', 'F1_macro', 'Precision_macro', 'Recall_macro']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        summary_df.plot(x='model', y=metric, kind='bar', ax=ax, legend=False, color='steelblue')
        ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_xticklabels(summary_df['model'], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3)
    
    plt.tight_layout()
    plt.savefig(outdir / "metrics_comparison_bars.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Radar/Spider chart for multi-metric comparison
    from math import pi
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = [n / len(metrics) * 2 * pi for n in range(len(metrics))]
    angles += angles[:1]
    
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    
    for idx, row in summary_df.iterrows():
        values = row[metrics].tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=row['model'])
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title('Multi-Metric Performance Comparison', size=16, fontweight='bold', pad=20)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(outdir / "metrics_radar_chart.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Heatmap of all metrics
    plt.figure(figsize=(10, 6))
    metrics_matrix = summary_df.set_index('model')[metrics]
    sns.heatmap(metrics_matrix, annot=True, fmt='.3f', cmap='YlGnBu', 
                cbar_kws={'label': 'Score'}, linewidths=0.5)
    plt.title('Performance Metrics Heatmap', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Models', fontsize=12)
    plt.tight_layout()
    plt.savefig(outdir / "metrics_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Model ranking visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate average rank for each model
    ranks = summary_df[metrics].rank(ascending=False)
    ranks['model'] = summary_df['model']
    ranks['avg_rank'] = ranks[metrics].mean(axis=1)
    ranks = ranks.sort_values('avg_rank')
    
    x_pos = np.arange(len(ranks))
    bars = ax.barh(x_pos, ranks['avg_rank'], color='coral')
    ax.set_yticks(x_pos)
    ax.set_yticklabels(ranks['model'])
    ax.invert_yaxis()
    ax.set_xlabel('Average Rank (lower is better)', fontsize=12)
    ax.set_title('Model Ranking Based on Average Performance', fontsize=16, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, ranks['avg_rank'])):
        ax.text(val + 0.05, i, f'{val:.2f}', va='center')
    
    plt.tight_layout()
    plt.savefig(outdir / "model_ranking.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Box plot showing metric distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    melted = summary_df.melt(id_vars='model', value_vars=metrics, 
                            var_name='Metric', value_name='Score')
    sns.boxplot(data=melted, x='Metric', y='Score', ax=ax, palette='Set2')
    sns.swarmplot(data=melted, x='Metric', y='Score', color='black', alpha=0.5, ax=ax)
    ax.set_title('Distribution of Metrics Across All Models', fontsize=16, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "metrics_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Generated 5 comparison visualizations in {outdir}")
    print("  - metrics_comparison_bars.png")
    print("  - metrics_radar_chart.png")
    print("  - metrics_heatmap.png")
    print("  - model_ranking.png")
    print("  - metrics_distribution.png")

# Define callback at module level so it can be pickled
if HAVE_TORCH:
    from skorch.callbacks import Callback
    
    class SetInputDim(Callback):
        """Dynamically set input_dim based on data shape after pipeline transforms"""
        def on_train_begin(self, net, X, y):
            # X shape after pipeline: (n_samples, n_features_selected)
            n_features = X.shape[1] if hasattr(X, 'shape') else X.shape[1]
            # Reinitialize module with correct input_dim
            net.set_params(module__input_dim=n_features)
            net.initialize()


def main():
    global logger  # Declare we're modifying the global logger
    # Initialize logger first
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/baseline_models_{timestamp}.log"
    logger = setup_logger(log_file, name="baseline_models")
    
    print("="*80)
    print("Starting Baseline Models Training Pipeline")
    print("="*80)
    
    # ============================================================
    # MODEL SELECTION FLAGS - Set to True/False to enable/disable
    # ============================================================
    TRAIN_LOGISTIC_REGRESSION = True
    TRAIN_RANDOM_FOREST = True
    TRAIN_XGBOOST = True
    TRAIN_KNN = True
    TRAIN_MLP = True
    TRAIN_SVC = True
    # ============================================================
    
    # Configuration for large datasets
    MAX_SAMPLES = 10_000  # Adjust based on your RAM (1M is ~8GB for typical features)
    MIN_CLASS_SAMPLES = 4000  # Minimum samples per class
    SAMPLE_PER_FILE = True  # Set to True to sample each file independently
    
    print(f"\nDataset Configuration:")
    print(f"  Max samples: {MAX_SAMPLES:,}")
    print(f"  Sampling strategy: {'per-file' if SAMPLE_PER_FILE else 'combined'}")
    print(f"  Target memory usage: ~8-16 GB RAM")
    
    print(f"\nModel Selection:")
    print(f"  Logistic Regression: {'✓' if TRAIN_LOGISTIC_REGRESSION else '✗'}")
    print(f"  Random Forest: {'✓' if TRAIN_RANDOM_FOREST else '✗'}")
    print(f"  XGBoost: {'✓' if TRAIN_XGBOOST else '✗'}")
    print(f"  k-NN: {'✓' if TRAIN_KNN else '✗'}")
    print(f"  MLP: {'✓' if TRAIN_MLP else '✗'}")
    print(f"  SVC: {'✓' if TRAIN_SVC else '✗'}")
    
    # Print initial memory state
    try:
        print_memory_usage()
    except Exception as e:
        print(f"[Memory] Could not print memory usage: {e}")
    
    outdir = make_output_dir()
    print(f"Output directory: {outdir}")

    X, y, groups = load_data(max_samples=MAX_SAMPLES, min_class_samples=MIN_CLASS_SAMPLES, sample_per_file=SAMPLE_PER_FILE)
    feature_names = X.columns.tolist()
    n_features = X.shape[1]
    le = LabelEncoder()
    le.fit(y.astype(str))
    y = le.transform(y.astype(str))
    dump(le, outdir / "label_encoder.joblib")
    num_classes = len(le.classes_)
    # Save mapping for later interpretability
    label_map = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"Label mapping: {label_map}")
    print(f"Number of classes: {num_classes}")

    ks = k_grid_from_dim(n_features)
    print(f"Feature selection grid: {ks}")

    # Train/holdout split:
    if groups is not None:
        print("Performing group-based train/test split...")
        # split by groups: keep some groups entirely in holdout
        unique_g = pd.Series(groups).drop_duplicates()
        holdout_frac = 0.2
        n_hold = max(1, int(len(unique_g) * holdout_frac))
        hold_groups = set(unique_g.sample(n_hold, random_state=RANDOM_STATE))
        mask_hold = pd.Series(groups).isin(hold_groups).values
        X_train, X_test = X.loc[~mask_hold], X.loc[mask_hold]
        y_train, y_test = y[~mask_hold], y[mask_hold]
        groups_train = pd.Series(groups)[~mask_hold].values
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Clean up intermediate variables
        del unique_g, hold_groups, mask_hold
    else:
        print("Performing random train/test split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )
        groups_train = None
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
    
    # Delete full dataset to free memory
    del X, y, groups
    gc.collect()

    # Models to train
    models = []

    # --- Logistic Regression (elastic-net) ---
    if TRAIN_LOGISTIC_REGRESSION:
        print("\n[+] Adding Logistic Regression to training queue")
        for sub_name, est, space in logistic_space():
            pipe = build_pipeline(est, scale_for_model=True, n_features=n_features,
                                scaler_type="standard", impute=True)
            models.append((f"{sub_name}", pipe, space))

    # --- Random Forest (GPU if available) ---
    if TRAIN_RANDOM_FOREST:
        print("[+] Adding Random Forest to training queue")
        if HAVE_CUML:
            print("    Using RAPIDS cuML Random Forest (GPU)")
            rf = cuRF(random_state=RANDOM_STATE, n_streams=4)
            models.append((
                "RandomForest_GPU",
                build_pipeline(rf, scale_for_model=False, n_features=n_features,
                            scaler_type=None, impute=True),
                rf_space_gpu()
            ))
        else:
            print("    Using sklearn Random Forest (CPU)")
            rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
            models.append((
                "RandomForest_CPU",
                build_pipeline(rf, scale_for_model=False, n_features=n_features,
                            scaler_type=None, impute=True),
                rf_space()
            ))

    # --- XGBoost (GPU if available) ---
    if TRAIN_XGBOOST and HAVE_XGB:
        print("[+] Adding XGBoost to training queue")
        device = "cuda" if HAVE_TORCH and torch.cuda.is_available() else "cpu"
        tree_method = "hist" if device == "cpu" else "gpu_hist"
        print(f"    Using XGBoost with {tree_method} on {device}")
        
        xgb = XGBClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1 if device == "cpu" else 1,
            tree_method=tree_method,
            device=device,
            objective="multi:softprob",
        )
        models.append((
            f"XGBoost_{device.upper()}",
            build_pipeline(xgb, scale_for_model=False, n_features=n_features,
                        scaler_type=None, impute=True),  # Changed from False to True
            xgb_space()
        ))
    elif TRAIN_XGBOOST and not HAVE_XGB:
        print("[-] XGBoost requested but not available - skipping")

    # --- k-NN (GPU if available) ---
    if TRAIN_KNN:
        print("[+] Adding k-NN to training queue")
        if HAVE_CUML:
            print("    Using RAPIDS cuML KNN (GPU)")
            knn = cuKNN()
            models.append((
                "KNN_GPU",
                build_pipeline(knn, scale_for_model=True, n_features=n_features,
                            scaler_type="standard", impute=True),
                knn_space_gpu()
            ))
        else:
            print("    Using sklearn KNN (CPU)")
            knn = KNeighborsClassifier()
            models.append((
                "KNN_CPU",
                build_pipeline(knn, scale_for_model=True, n_features=n_features,
                            scaler_type="standard", impute=True),
                knn_space()
            ))

    # --- MLP (GPU if available) ---
    if TRAIN_MLP:
        print("[+] Adding MLP to training queue")
        if HAVE_TORCH and torch.cuda.is_available():
            print("    Using PyTorch MLP (GPU)")
            device = 'cuda'
            
            # Use the module-level SetInputDim callback
            mlp = NeuralNetClassifier(
                module=TorchMLP,
                module__hidden_dim=128,
                module__num_classes=num_classes,
                module__dropout=0.2,
                max_epochs=100,
                lr=1e-3,
                batch_size=128,
                iterator_train__shuffle=True,
                device=device,
                verbose=1,
                callbacks=[SetInputDim()],  # Now this can be pickled
                warm_start=False,  # Ensure clean initialization
            )
            models.append((
                "MLP_GPU",
                build_pipeline(mlp, scale_for_model=True, n_features=n_features,
                            scaler_type="standard", impute=True),
                mlp_space_gpu(n_features, num_classes)
            ))
        else:
            print("    Using sklearn MLP (CPU)")
            mlp = MLPClassifier(
                early_stopping=True,
                max_iter=400,
                random_state=RANDOM_STATE
            )
            models.append((
                "MLP_CPU",
                build_pipeline(mlp, scale_for_model=True, n_features=n_features,
                            scaler_type="standard", impute=True),
                mlp_space()
            ))

    # --- SVC (GPU if available) ---
    if TRAIN_SVC:
        print("[+] Adding SVC to training queue")
        if HAVE_CUML:
            print("    Using RAPIDS cuML SVC (GPU)")
            svc = cuSVC(
                kernel="rbf",
                random_state=RANDOM_STATE
            )
            models.append((
                "SVC_GPU",
                build_pipeline(
                    svc,
                    scale_for_model=True,
                    n_features=n_features,
                    scaler_type="standard",
                    impute=True
                ),
                svc_space_gpu()
            ))
        else:
            print("    Using sklearn SVC (CPU)")
            svc = SVC(
                kernel="rbf",
                probability=False,
                decision_function_shape="ovr",
                random_state=RANDOM_STATE
            )
            models.append((
                "SVC_CPU",
                build_pipeline(
                    svc,
                    scale_for_model=True,
                    n_features=n_features,
                    scaler_type="standard",
                    impute=True
                ),
                svc_space_classifier()
            ))

    if len(models) == 0:
        print("\n[ERROR] No models selected for training!")
        print("Please set at least one TRAIN_* flag to True")
        return

    print(f"\n{'='*80}")
    print(f"Total models queued for training: {len(models)}")
    print(f"{'='*80}")
    summary = []

    for idx, (name, pipe, space) in enumerate(models):
        print(f"\n{'='*80}")
        print(f"=== Tuning {name} ({idx+1}/{len(models)}) ===")
        print(f"{'='*80}")
        
        # Print memory before training
        try:
            print_memory_usage()
        except Exception:
            pass
        
        start_time = time.time()
        best = search_cv_for_model(
            name=name,
            pipe=pipe,
            param_space=space,
            ks=ks,
            X=X_train,
            y=y_train,
            groups=groups_train,
            outdir=outdir,
            n_iter=60 if HAVE_BAYES else 80
        )
        elapsed = time.time() - start_time
        
        metrics = evaluate_on_holdout(best, X_test, y_test, outdir / name, label=f"{name}_holdout")
        print(f"{name} holdout metrics: {metrics}")
        print(f"Training time: {elapsed:.2f} seconds")
        
        row = {"model": name, **metrics, "training_time_sec": elapsed}
        summary.append(row)
        
        # Explicit cleanup after each model
        del best, pipe, space
        cleanup_memory()
        
        # Print memory after cleanup
        try:
            print(f"[Memory] Cleaned up after {name}")
            print_memory_usage()
        except Exception:
            pass

    # Clean up models list
    del models
    gc.collect()

    print(f"\n{'='*80}")
    print("Saving results and generating visualizations...")
    print(f"{'='*80}")
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(outdir / "holdout_summary.csv", index=False)
    create_comparison_visualizations(summary_df, outdir)
    
    with open(outdir / "env_info.json", "w") as f:
        json.dump({
            "HAVE_BAYES": HAVE_BAYES,
            "HAVE_XGB": HAVE_XGB,
            "HAVE_TORCH": HAVE_TORCH,
            "HAVE_CUML": HAVE_CUML,
            "random_state": RANDOM_STATE,
            "n_features": n_features,
            "num_classes": num_classes,
            "ks": ks,
            "versions": {
                "pandas": pd.__version__,
                "numpy": np.__version__,
            }
        }, f, indent=2)

    print(f"\n{'='*80}")
    print("=== Training Pipeline Complete ===")
    print(f"{'='*80}")
    print(f"Results saved to: {outdir}")
    print(f"Log file saved in: logs/")
    
    # Final memory state
    try:
        print("\n[Memory] Final state:")
        print_memory_usage()
    except Exception:
        pass


if __name__ == "__main__":
    print_device_info()
    main()
