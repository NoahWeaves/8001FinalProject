# CS8001 Project – Baseline ML Models

## Purpose
- Train and compare baseline machine-learning classifiers for network traffic anomaly/intrusion detection.
- Uses multiple CSVs in data/train, group-aware CV by timestamp, mutual-information feature selection, and optional GPU acceleration (RAPIDS cuML, PyTorch/skorch, XGBoost).
- **Handles large datasets (15M+ rows) via stratified sampling to prevent memory crashes.**

## Data expectations
- Input files: data/train/*.csv (e.g., LDAP, MSSQL, NetBIOS, SYN, UDP, UDPlag).
- **Large dataset handling**: Automatically samples to ~1M rows while preserving class distribution.
- Required columns:
  - Label (classification target)
  - Timestamp (used to create group-based splits; parsed with pandas.to_datetime)
- Common identifier columns are dropped if present: Unnamed: 0, Flow ID, Source IP, Destination IP, SimillarHTTP, Inbound.
- Infinity values are converted to NaN, then imputed (median).

## Models covered
- Logistic Regression (elastic-net)
- Random Forest (CPU or cuML GPU)
- XGBoost (CPU or GPU if available)
- k-NN (CPU or cuML GPU)
- MLP (sklearn CPU or PyTorch+skorch GPU)
- SVC (CPU or cuML GPU)

## Key features
- Preprocessing: median imputation and scaling (Standard/MinMax/Robust as appropriate per model).
- Feature selection: SelectKBest(mutual_info_classif) over a grid [8, 16, 32, 48, 64, n_features].
- Hyperparameter search: BayesSearchCV (if scikit-optimize is installed) or RandomizedSearchCV.
- Group-aware CV: GroupKFold on minute-level Timestamp; otherwise StratifiedKFold via binned target.
- Holdout evaluation with confusion matrix, classification report, and JSON metrics.
- Visual comparisons (bars, radar, heatmap, ranking, distribution).
- Logging to logs/ and memory cleanup between models.

## Quick start
- Ensure Python 3.10+.
- Place CSV files under data/train/.
- **For large datasets (>5M rows)**: The script automatically samples to 1M rows. Adjust `MAX_SAMPLES` in `baseline_models.py` main() if needed.
- Run:
  - python baseline_models.py

## Outputs
- runs/models/classif_YYYYMMDD_HHMMSS/
  - Per-model/
    - best_model.joblib
    - best_params.json
    - cv_results.csv
    - {model}_holdout_preds.csv
    - {model}_holdout_confusion_matrix.png
    - {model}_holdout_classification_report.csv
    - {model}_holdout_metrics.json
  - holdout_summary.csv (all models)
  - metrics_comparison_bars.png
  - metrics_radar_chart.png
  - metrics_heatmap.png
  - model_ranking.png
  - metrics_distribution.png
  - env_info.json
- logs/baseline_models_*.log

## Configuration for Large Datasets
In `baseline_models.py`, adjust these variables in `main()`:
- `MAX_SAMPLES`: Maximum rows to use (default: 1,000,000)
- `SAMPLE_PER_FILE`: Sample each CSV independently (`True`) or sample combined data (`False`)

**Memory guidelines**:
- 1M rows ≈ 8-16 GB RAM
- 500K rows ≈ 4-8 GB RAM
- 2M rows ≈ 16-32 GB RAM

## Optional GPU support
- RAPIDS cuML (RandomForest, KNN, SVC): requires compatible CUDA stack and cupy.
- PyTorch + skorch (GPU MLP): uses CUDA if available.
- XGBoost: uses gpu_hist if CUDA is available; otherwise hist.
- The script prints detected GPU libraries at startup.

## Minimal dependencies (non-exhaustive)
- numpy, pandas, scikit-learn, matplotlib, seaborn, joblib, tqdm, psutil
- Optional: scikit-optimize (skopt), xgboost, torch, skorch, cupy, cuml

## Notes
- Random seed set to 42 for reproducibility.
- Label encoder mapping is saved alongside results.
- Adjust feature selection grid and search iterations inside baseline_models.py if needed.

## LLM fine-tuning on data_processed

We additionally fine-tune small LLMs (Qwen and Phi) for the same classification task using the pre-split files under `data_processed/`:
- Models (examples used in the project): `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `microsoft/Phi-3.5-mini-instruct`, `microsoft/Phi-3.5-MoE-instruct`.
- Train on: `data_processed/*_train.csv`.
- Test on: `data_processed/*_test.csv`.
- Input format (default): each row is serialized as a single line of `"key=value"` pairs, space-separated, excluding `Label`, `Timestamp`, and `Scenario`.
- Optional prompt mode: with `--use-prompt`, features are turned into a comma-separated value list and injected into a prompt template via the `{val}` placeholder, e.g.:
  - `--use-prompt --prompt-template "Based on the following input return the most likely type of situation from the following list: ['BENIGN', 'DrDoS_LDAP', 'DrDoS_MSSQL', 'DrDoS_NetBIOS', 'DrDoS_UDP', 'Syn', 'UDP-lag'].\\nInput: {val}\\nOutput: "`
- Fine-tuned outputs (metrics, preds, plots, adapter copy) are written under model-specific folders beneath `models/` in timestamped subdirectories (e.g., `models/Qwen_Qwen2.5-1.5B-Instruct/classif_YYYYMMDD_HHMMSS/`).
- Trainer checkpoints (including intermediate epochs) are stored under:
  - `runs/models/classif_YYYYMMDD_HHMMSS/`

Install (extras for LLMs):
- `pip install -U transformers peft bitsandbytes accelerate torch scikit-learn seaborn matplotlib pandas`

Auth (Hugging Face Hub):
- Set `HUGGINGFACE_TOKEN` env var or run `huggingface-cli login` for gated models.

Run (examples):
- `python scripts/llm_finetune.py --model-id Qwen/Qwen2.5-1.5B-Instruct`
- `python scripts/llm_finetune.py --model-id microsoft/Phi-3.5-mini-instruct`

Notes:
- Script uses QLoRA (4-bit) by default for larger models and can fall back to full-precision LoRA on smaller models if VRAM permits.
- Early stopping is enabled with a patience of 10 epochs, monitoring the loss via the Trainer.
- Identifiers/timestamps/scenario are excluded from features. The same metrics as baseline are produced (Accuracy, F1, Precision, Recall), plus confusion matrix/report.
