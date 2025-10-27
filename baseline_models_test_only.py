# model_testing.py
import os, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import logging
import builtins
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import load
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Check for optional dependencies
try:
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.neighbors import KNeighborsClassifier as cuKNN
    from cuml.svm import SVC as cuSVC
    HAVE_CUML = True
except ImportError:
    HAVE_CUML = False
    print("[WARNING] RAPIDS cuML not available. GPU models (RF, KNN, SVC) will be skipped.")

try:
    import torch
    import torch.nn as nn
    from skorch import NeuralNetClassifier
    HAVE_TORCH = True
    
    # Define TorchMLP class for loading PyTorch models
    class TorchMLP(nn.Module):
        """PyTorch MLP for GPU acceleration"""
        def __init__(self, input_dim=64, hidden_dim=128, num_classes=2, dropout=0.2):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.dropout1 = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.ln2 = nn.LayerNorm(hidden_dim // 2)
            self.dropout2 = nn.Dropout(dropout)
            self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
            
        def forward(self, x):
            if x.dtype == torch.float64:
                x = x.float()
            x = torch.relu(self.ln1(self.fc1(x)))
            x = self.dropout1(x)
            x = torch.relu(self.ln2(self.fc2(x)))
            x = self.dropout2(x)
            x = self.fc3(x)
            return x
            
except ImportError:
    HAVE_TORCH = False
    print("[WARNING] PyTorch/skorch not available. MLP_GPU model will be skipped.")

# Global logger instance
logger = None

def setup_logger(log_path: str | Path, name: str = "model_testing"):
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

def find_most_recent_data_folder(base_path="data_processed"):
    """Find the most recent data_processed folder"""
    base = Path(base_path)
    if not base.exists():
        raise FileNotFoundError(f"Base path {base_path} does not exist")
    
    # Find all subdirectories matching pattern data_YYYYMMDD_HHMMSS
    data_folders = [d for d in base.iterdir() if d.is_dir() and d.name.startswith("data_")]
    
    if not data_folders:
        raise FileNotFoundError(f"No data folders found in {base_path}")
    
    # Sort by name (timestamp in folder name) and get the most recent
    most_recent = sorted(data_folders, key=lambda x: x.name)[-1]
    print(f"Found most recent data folder: {most_recent}")
    
    return most_recent

def load_test_data(data_folder):
    """Load preprocessed test data from the specified folder"""
    data_path = Path(data_folder)
    
    print(f"\nLoading test data from: {data_path}")
    
    # Load test data
    X_test_path = data_path / "X_test.csv"
    y_test_path = data_path / "y_test.csv"
    groups_test_path = data_path / "groups_test.csv"
    label_encoder_path = data_path / "label_encoder.joblib"
    metadata_path = data_path / "metadata.json"
    
    # Check files exist
    required_files = [X_test_path, y_test_path, label_encoder_path]
    for file in required_files:
        if not file.exists():
            raise FileNotFoundError(f"Required file not found: {file}")
    
    # Load data
    print("  Loading X_test.csv...")
    X_test = pd.read_csv(X_test_path)
    print(f"    Shape: {X_test.shape}")
    
    print("  Loading y_test.csv...")
    y_test = pd.read_csv(y_test_path)
    y_test = y_test.values.ravel()  # Convert to 1D array
    print(f"    Shape: {y_test.shape}")
    
    print("  Loading label_encoder.joblib...")
    label_encoder = load(label_encoder_path)
    print(f"    Classes: {label_encoder.classes_}")
    
    # Load groups if available (optional)
    groups_test = None
    if groups_test_path.exists():
        print("  Loading groups_test.csv...")
        groups_test = pd.read_csv(groups_test_path).values.ravel()
        print(f"    Shape: {groups_test.shape}")
    
    # Load metadata if available
    metadata = None
    if metadata_path.exists():
        print("  Loading metadata.json...")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"    Metadata keys: {list(metadata.keys())}")
    
    return X_test, y_test, groups_test, label_encoder, metadata

def load_trained_model(model_name, models_base="models"):
    """Load a trained model from the models folder"""
    models_path = Path(models_base)
    
    if not models_path.exists():
        raise FileNotFoundError(f"Models base path not found: {models_path}")
    
    # Look for the model folder directly in models/
    model_path = models_path / model_name
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model folder not found: {model_path}")
    
    # Check if this is a GPU model that requires unavailable dependencies
    if not HAVE_CUML and any(gpu_name in model_name for gpu_name in ["RandomForest_GPU", "KNN_GPU", "SVC_GPU"]):
        raise ImportError(f"Model {model_name} requires RAPIDS cuML which is not installed")
    
    if not HAVE_TORCH and "MLP_GPU" in model_name:
        raise ImportError(f"Model {model_name} requires PyTorch/skorch which is not installed")
    
    print(f"\nLoading model: {model_name}")
    print(f"  From: {model_path}")
    
    # Load model
    model_file = model_path / "best_model.joblib"
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    try:
        model = load(model_file)
        print(f"  ✓ Model loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_file}: {str(e)}")
    
    # Load best params if available
    params_file = model_path / "best_params.json"
    best_params = None
    if params_file.exists():
        with open(params_file, 'r') as f:
            best_params = json.load(f)
        print(f"  ✓ Best parameters loaded")
    
    return model, best_params, model_path

def evaluate_model(model, X_test, y_test, model_name, label_encoder, output_dir):
    """Evaluate a model on test data and save results"""
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*80}")
    
    # Make predictions
    print("  Making predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    print("  Calculating metrics...")
    metrics = {
        "model": model_name,
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "F1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "F1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "Precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "Precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "Recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "Recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
    }
    
    # Print metrics
    print(f"\n  Results for {model_name}:")
    for metric, value in metrics.items():
        if metric != "model":
            print(f"    {metric:20s}: {value:.4f}")
    
    # Create output directory for this model
    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred,
        "y_true_label": label_encoder.inverse_transform(y_test),
        "y_pred_label": label_encoder.inverse_transform(y_pred)
    })
    predictions_df.to_csv(model_output_dir / "test_predictions.csv", index=False)
    print(f"  ✓ Predictions saved to: {model_output_dir / 'test_predictions.csv'}")
    
    # Confusion matrix
    print("  Generating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title(f"Confusion Matrix – {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(model_output_dir / "confusion_matrix.png", dpi=300)
    plt.close()
    print(f"  ✓ Confusion matrix saved")
    
    # Classification report
    print("  Generating classification report...")
    report = classification_report(y_test, y_pred, 
                                   target_names=label_encoder.classes_, 
                                   output_dict=True, 
                                   zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(model_output_dir / "classification_report.csv")
    print(f"  ✓ Classification report saved")
    
    # Save metrics summary
    with open(model_output_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ Metrics saved to: {model_output_dir / 'test_metrics.json'}")
    
    return metrics

def create_comparison_visualizations(results_df, output_dir):
    """Create comparison visualizations for all tested models"""
    print(f"\nGenerating comparison visualizations...")
    
    metrics = ['Accuracy', 'F1_macro', 'F1_weighted', 'Precision_macro', 'Recall_macro']
    
    # Bar chart comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        results_df.plot(x='model', y=metric, kind='bar', ax=ax, legend=False, color='steelblue')
        ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_xticklabels(results_df['model'], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3)
    
    # Hide the last subplot if not needed
    if len(metrics) < len(axes):
        axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: metrics_comparison.png")
    
    # Heatmap
    plt.figure(figsize=(10, 6))
    metrics_matrix = results_df.set_index('model')[metrics]
    sns.heatmap(metrics_matrix, annot=True, fmt='.3f', cmap='YlGnBu', 
                cbar_kws={'label': 'Score'}, linewidths=0.5)
    plt.title('Performance Metrics Heatmap', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Models', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: metrics_heatmap.png")

def main():
    global logger
    
    # ============================================================
    # MODEL SELECTION FLAGS - Set to True/False to test models
    # ============================================================
    TEST_LOGISTIC_REGRESSION = True
    TEST_RANDOM_FOREST = True
    TEST_XGBOOST = True
    TEST_KNN = True
    TEST_MLP = True
    TEST_SVC = True
    # ============================================================
    
    print("="*80)
    print("Model Testing Pipeline - Test Data Only")
    print("="*80)
    
    # Print dependency status
    print(f"\nDependency Status:")
    print(f"  RAPIDS cuML: {'✓ Available' if HAVE_CUML else '✗ Not Available'}")
    print(f"  PyTorch/skorch: {'✓ Available' if HAVE_TORCH else '✗ Not Available'}")
    
    # Initialize logger
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/model_testing_{timestamp}.log"
    logger = setup_logger(log_file, name="model_testing")
    
    print(f"\nModel Selection for Testing:")
    print(f"  Logistic Regression: {'✓' if TEST_LOGISTIC_REGRESSION else '✗'}")
    print(f"  Random Forest: {'✓' if TEST_RANDOM_FOREST else '✗'} {'(requires cuML)' if not HAVE_CUML else ''}")
    print(f"  XGBoost: {'✓' if TEST_XGBOOST else '✗'}")
    print(f"  k-NN: {'✓' if TEST_KNN else '✗'} {'(requires cuML)' if not HAVE_CUML else ''}")
    print(f"  MLP: {'✓' if TEST_MLP else '✗'} {'(requires PyTorch)' if not HAVE_TORCH else ''}")
    print(f"  SVC: {'✓' if TEST_SVC else '✗'} {'(requires cuML)' if not HAVE_CUML else ''}")
    
    # Create output directory
    output_dir = Path("test_results") / f"test_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Find and load most recent preprocessed data
    try:
        data_folder = find_most_recent_data_folder()
        X_test, y_test, groups_test, label_encoder, metadata = load_test_data(data_folder)
    except Exception as e:
        print(f"\n[ERROR] Failed to load test data: {e}")
        return
    
    # Build list of models to test
    models_to_test = []
    
    if TEST_LOGISTIC_REGRESSION:
        models_to_test.append("LogisticRegression")
    
    if TEST_RANDOM_FOREST:
        if HAVE_CUML:
            models_to_test.append("RandomForest_GPU")
        else:
            print(f"\n[INFO] Skipping RandomForest_GPU - cuML not available")
    
    if TEST_XGBOOST:
        models_to_test.append("XGBoost_CUDA")
    
    if TEST_KNN:
        if HAVE_CUML:
            models_to_test.append("KNN_GPU")
        else:
            print(f"\n[INFO] Skipping KNN_GPU - cuML not available")
    
    if TEST_MLP:
        if HAVE_TORCH:
            models_to_test.append("MLP_GPU")
        else:
            print(f"\n[INFO] Skipping MLP_GPU - PyTorch not available")
    
    if TEST_SVC:
        if HAVE_CUML:
            models_to_test.append("SVC_GPU")
        else:
            print(f"\n[INFO] Skipping SVC_GPU - cuML not available")
    
    if not models_to_test:
        print("\n[ERROR] No models selected for testing!")
        print("Please set at least one TEST_* flag to True and ensure required dependencies are installed")
        return
    
    print(f"\n{'='*80}")
    print(f"Models queued for testing: {len(models_to_test)}")
    print(f"{'='*80}")
    
    # Test each model
    results = []
    
    for idx, model_name in enumerate(models_to_test):
        try:
            # Load trained model
            model, best_params, model_path = load_trained_model(model_name)
            
            # Evaluate on test data
            metrics = evaluate_model(model, X_test, y_test, model_name, 
                                    label_encoder, output_dir)
            results.append(metrics)
            
        except (FileNotFoundError, ImportError) as e:
            print(f"\n[WARNING] Skipping {model_name}: {e}")
            continue
        except Exception as e:
            print(f"\n[ERROR] Failed to test {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not results:
        print("\n[ERROR] No models were successfully tested!")
        return
    
    # Save summary results
    print(f"\n{'='*80}")
    print("Saving results and generating visualizations...")
    print(f"{'='*80}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "test_results_summary.csv", index=False)
    print(f"✓ Summary saved to: {output_dir / 'test_results_summary.csv'}")
    
    # Generate comparison visualizations
    create_comparison_visualizations(results_df, output_dir)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("=== Testing Complete ===")
    print(f"{'='*80}")
    print(f"\nResults Summary:")
    print(results_df.to_string(index=False))
    print(f"\nAll results saved to: {output_dir}")
    print(f"Log file: {log_file}")

if __name__ == "__main__":
    main()
