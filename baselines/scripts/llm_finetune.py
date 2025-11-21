import os, json, time, gc, glob, inspect, logging
from functools import lru_cache
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from typing import Optional

# Backfill accelerate.clear_device_cache when using older accelerate builds
try:
    from accelerate.utils import memory as _accelerate_memory
except Exception:
    _accelerate_memory = None

if _accelerate_memory is not None and not hasattr(_accelerate_memory, "clear_device_cache"):
    def _compat_clear_device_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _accelerate_memory.clear_device_cache = _compat_clear_device_cache
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from transformers import EarlyStoppingCallback, TrainerCallback

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

DROP_COLS = ["Unnamed: 0", "Flow ID", "Source IP", "Destination IP", "SimillarHTTP", "Inbound"]
LOGGING_STEPS = 400

EARLY_STOP_METRICS = {
    "loss": ("loss", False),
    "f1_weighted": ("eval_F1_weighted", True),
    "f1_macro": ("eval_F1_macro", True),
    "recall_weighted": ("eval_recall_weighted", True),
    "recall_macro": ("eval_recall_macro", True),
    "precision_weighted": ("eval_precision_weighted", True),
    "precision_macro": ("eval_precision_macro", True),
    "accuracy": ("eval_Accuracy", True),
}


@lru_cache(maxsize=1)
def _training_args_params():
    try:
        return set(inspect.signature(TrainingArguments.__init__).parameters)
    except (ValueError, TypeError):
        return set()


def _resolve_eval_strategy_key():
    params = _training_args_params()
    if "evaluation_strategy" in params:
        return "evaluation_strategy"
    if "eval_strategy" in params:
        return "eval_strategy"
    return None

def make_output_dir(base="runs/llm", *, ts: Optional[str] = None):
    ts = ts or time.strftime("%Y%m%d_%H%M%S")
    outdir = Path(base) / f"classif_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def setup_run_logger(run_ts: str):
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"llm_model_{run_ts}.log"
    logger = logging.getLogger(f"llm_finetune_{run_ts}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger, log_path

def load_processed_splits():
    train_paths = sorted(glob.glob("data_processed/*_train.csv"))
    test_paths  = sorted(glob.glob("data_processed/*_test.csv"))
    if not train_paths or not test_paths:
        raise FileNotFoundError("Expected data_processed/*_train.csv and *_test.csv")
    def _load_many(paths):
        dfs = []
        for p in paths:
            df = pd.read_csv(p, low_memory=True)
            df.columns = df.columns.str.strip()
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)
    train_df = _load_many(train_paths)
    test_df  = _load_many(test_paths)

    # Clean and align columns
    for df in (train_df, test_df):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        keep = [c for c in df.columns if c not in DROP_COLS]
        df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True, errors="ignore")
        # Ensure columns are consistent across splits (outer join semantics)
    common_cols = sorted(list(set(train_df.columns).intersection(set(test_df.columns))))
    # Ensure Label exists
    if "Label" not in common_cols:
        raise ValueError("Column 'Label' must be present in data_processed splits.")
    train_df = train_df[common_cols]
    test_df  = test_df[common_cols]
    return train_df, test_df

def to_text_rows(
    df: pd.DataFrame,
    *,
    use_prompt: bool = False,
    prompt_template: Optional[str] = None,
):
    """
    Current default input format (use_prompt=False):
      - Each row becomes a single string: "col1=val1 col2=val2 ..."
      - This is what the existing checkpoints were trained on.

    With use_prompt=True and a prompt_template containing `{val}`:
      - We build `val` as a comma-separated list of feature values (no keys).
      - Then format: prompt_template.format(val=val_str)
        e.g. "Based on ... Input: {val}\\nOutput: "
    """
    # Exclude non-feature columns
    feature_cols = [c for c in df.columns if c not in ["Label", "Timestamp", "Scenario"]]

    # Default template (if not provided) when using prompts
    if use_prompt and not prompt_template:
        prompt_template = (
            "Based on the following input return the most likely type of situation "
            "from the following list: ['BENIGN', 'DrDoS_LDAP', 'DrDoS_MSSQL', "
            "'DrDoS_NetBIOS', 'DrDoS_UDP', 'Syn', 'UDP-lag'].\n"
            "Input: {val}\nOutput: "
        )

    def row_to_text(row):
        if use_prompt:
            # build comma-separated list of raw feature values
            vals = []
            for c in feature_cols:
                v = row[c]
                if pd.isna(v):
                    v = "NaN"
                vals.append(str(v))
            val_str = ", ".join(vals)
            return prompt_template.format(val=val_str)
        else:
            # original compact "key=value" format
            parts = []
            for c in feature_cols:
                val = row[c]
                if pd.isna(val):
                    val = "NaN"
                parts.append(f"{c}={val}")
            return " ".join(parts)

    texts = df.apply(row_to_text, axis=1).astype(str).tolist()
    labels = df["Label"].astype(str).tolist()
    return texts, labels

class TextClsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class MetricsLoggerCallback(TrainerCallback):
    def __init__(self, logger):
        self.logger = logger

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        metric_str = ", ".join(f"{k}={v:.6f}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in metrics.items())
        self.logger.info("Evaluation metrics: %s", metric_str)

def build_label_map(train_labels):
    classes = sorted(pd.Series(train_labels).unique())
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for c, i in label2id.items()}
    return label2id, id2label

def compute_metrics_fn(eval_pred, id2label):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    y_true = labels
    y_pred = preds
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "Recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    return metrics

def plot_and_save_confusion(y_true, y_pred, id2label, path_png, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", annot=False)
    plt.title("Confusion Matrix – LLM")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path_png)
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune Llama-3.2 for tabular classification via text serialization")
    parser.add_argument("--model-id", type=str, required=True,
                        help="meta-llama/Llama-3.2-1B or meta-llama/Llama-3.2-3B (mapped to local models/ folders)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        default=False,
        help="Enable 4-bit QLoRA when VRAM is insufficient (off by default).",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default=None,
        help="Optional override for output base directory. "
             "If not set, outputs go under the corresponding models/llama-3.2-*B/ directory.",
    )
    parser.add_argument(
        "--use-prompt",
        action="store_true",
        default=True,
        help="If set, prepend an instruction-style prompt and feed a comma-separated value list as input.",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default=None,
        help="Optional prompt template containing '{val}' placeholder. "
             "If omitted and --use-prompt is set, a sensible default is used.",
    )
    parser.add_argument(
        "--early-stop-metric",
        type=lambda s: s.lower(),
        choices=sorted(EARLY_STOP_METRICS.keys()),
        default="loss",
        help=(
            "Metric used by early stopping and best-model tracking. "
            "'loss' monitors training loss, other options monitor eval metrics."
        ),
    )
    parser.add_argument(
        "--eval-on-steps",
        action="store_true",
        help=(
            "If set, run evaluation/checkpointing every LOGGING_STEPS instead of once per epoch. "
            "LOGGING_STEPS is currently fixed at 50."
        ),
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Optional path to a Trainer checkpoint directory (e.g. runs/models/.../checkpoint-3600) to resume from.",
    )
    args = parser.parse_args()

    # Map model-id to local paths (you can extend this mapping if needed)
    model_id_lower = args.model_id.lower()
    if "1b" in model_id_lower:
        local_model_dir = Path("models") / "llama-3.2-1B" / "llama-3.2-1B"
        model_parent_dir = Path("models") / "llama-3.2-1B"
    elif "3b" in model_id_lower:
        local_model_dir = Path("models") / "llama-3.2-3B" / "llama-3.2-3B"
        model_parent_dir = Path("models") / "llama-3.2-3B"
    else:
        # Fallback: use model-id as given (e.g., remote HF ID) and default parent dir
        local_model_dir = Path(args.model_id)
        model_parent_dir = Path("models")

    run_ts = time.strftime("%Y%m%d_%H%M%S")
    logger, log_path = setup_run_logger(run_ts)
    resume_checkpoint = None
    if args.resume_from_checkpoint:
        resume_checkpoint = Path(args.resume_from_checkpoint)
        if not resume_checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {resume_checkpoint}")
        if not resume_checkpoint.is_dir():
            raise ValueError(f"Checkpoint path must be a directory: {resume_checkpoint}")

    metric_for_best_model, greater_is_better = EARLY_STOP_METRICS[args.early_stop_metric]
    requires_eval_metric = metric_for_best_model.startswith("eval_")
    eval_mode = "steps" if args.eval_on_steps else "epoch"

    # Determine metrics/artifacts root (model-specific) and checkpoint root (global runs/models/)
    if args.output_base is not None:
        metrics_root = make_output_dir(base=args.output_base, ts=run_ts)
    else:
        metrics_root = make_output_dir(base=str(model_parent_dir), ts=run_ts)

    # Checkpoints (full Trainer checkpoints) always go under runs/models/
    checkpoints_root = make_output_dir(base="runs/models", ts=run_ts)

    model_name_sanitized = local_model_dir.name.replace(".", "_")
    model_out = metrics_root / model_name_sanitized
    model_out.mkdir(parents=True, exist_ok=True)
    artifact_prefix = f"{model_name_sanitized}_holdout"

    best_dir = model_parent_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("LLM fine-tuning: %s", args.model_id)
    logger.info("=" * 80)
    logger.info("Model load dir: %s", local_model_dir)
    logger.info("Quantization: %s", "4-bit QLoRA" if args.use_4bit else "full-precision LoRA")
    logger.info("Checkpoints dir: %s", checkpoints_root)
    logger.info("Run artifacts dir: %s", model_out)
    logger.info(
        "Early-stop metric: %s (%s)",
        args.early_stop_metric,
        "maximize" if greater_is_better else "minimize",
    )
    logger.info("Run log file: %s", log_path)
    logger.info(
        "Evaluation cadence: %s every %s step(s)",
        "steps" if args.eval_on_steps else "epoch",
        LOGGING_STEPS if args.eval_on_steps else "epoch-end",
    )
    if resume_checkpoint:
        logger.info("Resuming from checkpoint: %s", resume_checkpoint)

    # Load data
    train_df, test_df = load_processed_splits()
    X_train_texts, y_train_str = to_text_rows(
        train_df,
        use_prompt=args.use_prompt,
        prompt_template=args.prompt_template,
    )
    X_test_texts,  y_test_str  = to_text_rows(
        test_df,
        use_prompt=args.use_prompt,
        prompt_template=args.prompt_template,
    )

    # Label map from training labels only
    label2id, id2label = build_label_map(y_train_str)
    y_train = [label2id[y] for y in y_train_str]
    # Map test labels; raise if unseen labels
    try:
        y_test = [label2id[y] for y in y_test_str]
    except KeyError as e:
        raise ValueError(f"Unseen label in test set: {e}")

    # Tokenizer
    hf_token = os.getenv("HUGGINGFACE_TOKEN", None)
    tokenizer = AutoTokenizer.from_pretrained(
        str(local_model_dir),
        use_fast=True,
        token=hf_token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    train_enc = tokenizer(X_train_texts, truncation=True, max_length=args.max_length)
    test_enc  = tokenizer(X_test_texts, truncation=True, max_length=args.max_length)

    train_ds = TextClsDataset(train_enc, y_train)
    test_ds  = TextClsDataset(test_enc, y_test)

    # Model with (Q)LoRA
    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(local_model_dir),
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        token=hf_token,
        trust_remote_code=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"],
        bias="none", task_type="SEQ_CLS"
    )
    model = get_peft_model(model, lora_cfg)

    # Training args
    training_arg_params = _training_args_params()
    strategy_kwargs = {}
    save_mode = eval_mode
    if "save_strategy" in training_arg_params:
        strategy_kwargs["save_strategy"] = save_mode
        if args.eval_on_steps and "save_steps" in training_arg_params:
            strategy_kwargs["save_steps"] = LOGGING_STEPS
    eval_strategy_key = _resolve_eval_strategy_key()
    if eval_strategy_key:
        strategy_kwargs[eval_strategy_key] = eval_mode
        if args.eval_on_steps and "eval_steps" in training_arg_params:
            strategy_kwargs["eval_steps"] = LOGGING_STEPS
    else:
        if requires_eval_metric or args.eval_on_steps:
            raise RuntimeError(
                "The current transformers build does not expose evaluation strategy controls required for "
                "the requested configuration."
            )
        logger.warning(
            "Current transformers.TrainingArguments does not expose an evaluation strategy parameter; "
            "skipping per-epoch evaluation configuration."
        )

    train_args = TrainingArguments(
        output_dir=str(checkpoints_root),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        logging_steps=LOGGING_STEPS,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=[],
        seed=RANDOM_STATE,
        save_total_limit=5,
        **strategy_kwargs,
    )

    def compute_metrics(eval_pred):
        return compute_metrics_fn(eval_pred, id2label)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,  # evaluate on test each epoch (no CV)
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=10,
                early_stopping_threshold=0.0,
            ),
            MetricsLoggerCallback(logger),
        ],
    )

    logger.info(
        "Starting training with early stopping (patience=%s, monitor=%s)...",
        10,
        args.early_stop_metric,
    )
    trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)

    logger.info("Running inference on test set...")
    preds_output = trainer.predict(test_ds)
    logits = preds_output.predictions
    y_pred = np.argmax(logits, axis=-1)

    # Metrics parity with baseline
    metrics = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "F1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "F1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "Precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "Precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "Recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "Recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
    }

    # Save artifacts mirroring baseline
    (model_out / "preds").mkdir(exist_ok=True, parents=True)
    preds_path = model_out / f"{artifact_prefix}_preds.csv"
    pd.DataFrame({
        "y_true": [id2label[i] for i in y_test],
        "y_pred": [id2label[i] for i in y_pred],
    }).to_csv(preds_path, index=False)

    label_indices = list(range(len(id2label)))
    plot_and_save_confusion(
        y_test,
        y_pred,
        id2label,
        model_out / f"{artifact_prefix}_confusion_matrix.png",
        labels=label_indices,
    )
    report = classification_report(
        y_test,
        y_pred,
        labels=label_indices,
        target_names=[id2label[i] for i in label_indices],
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        model_out / f"{artifact_prefix}_classification_report.csv"
    )

    with open(model_out / f"{artifact_prefix}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(model_out / "label_map.json", "w") as f:
        json.dump({"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}, f, indent=2)

    with open(model_out / "env_info.json", "w") as f:
        json.dump({
            "model_id": args.model_id,
            "model_load_dir": str(local_model_dir),
            "random_state": RANDOM_STATE,
            "versions": {
                "transformers": __import__("transformers").__version__,
                "torch": torch.__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__,
            },
            "use_4bit": args.use_4bit,
            "lora": {
                "r": lora_cfg.r, "alpha": lora_cfg.lora_alpha, "dropout": lora_cfg.lora_dropout,
                "target_modules": list(lora_cfg.target_modules)
            },
            "max_length": args.max_length,
        }, f, indent=2)

    # Save final/best model adapter under both the run directory and the model-specific best dir
    best_run_dir = model_out / "best_model"
    trainer.save_model(best_run_dir)
    # Also save/update a shared "best" checkpoint for this base model
    # (this can be overwritten by subsequent runs for the same base model)
    trainer.save_model(best_dir)

    logger.info("=" * 80)
    logger.info("LLM training complete")
    logger.info("Results saved to: %s", model_out)
    logger.info("Best model adapter saved to: %s", best_dir)
    logger.info("=" * 80)

    # Cleanup
    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
