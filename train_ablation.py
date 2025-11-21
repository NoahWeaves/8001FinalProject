"""
Lightweight LLM Controller Training for Ablation Studies
Fast training regime: ~1.5 hours per experiment
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import json
from pathlib import Path
import os

# ============================================================================
# ABLATION CONFIGURATION
# ============================================================================
ABLATION_NAME = "baseline_with_lavels_1epoch"  # Change this for each experiment
ABLATION_SAMPLES = 1000  # Small sample size for fast iteration
ABLATION_EPOCHS = 1  # Single epoch

# Configuration
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"  # Fast 1B model for ablations
DATA_DIR = "llm_training_data"
OUTPUT_DIR = f"./ablation_{ABLATION_NAME}"
MAX_LENGTH = 1024  # Reduced for 8GB VRAM
USE_4BIT_QUANTIZATION = True

# Optimized for 8GB VRAM (Quadro RTX 4000)
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = ABLATION_EPOCHS
WARMUP_STEPS = 20
SAVE_STEPS = 100  # Save frequently
EVAL_STEPS = 100

print("="*80)
print("LLM Controller - Fast Ablation Training")
print("="*80)
print(f"\n📊 ABLATION: {ABLATION_NAME}")
print(f"   Training samples: {ABLATION_SAMPLES:,}")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   Expected time: ~1.5 hours")
print(f"   Output: {OUTPUT_DIR}")
print("="*80)


def setup_model_and_tokenizer(model_name, use_4bit=True):
    """Load model and tokenizer with optional quantization"""
    
    print("\n[1/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print(f"  ✓ Tokenizer loaded")
    
    print(f"\n[2/3] Loading model with 4-bit quantization...")
    
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        model = prepare_model_for_kbit_training(model)
        print(f"  ✓ Model loaded")
    
    print(f"\n[3/3] Applying LoRA adapters...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    print(f"  ✓ LoRA applied")
    
    return model, tokenizer


def load_and_prepare_datasets(data_dir, tokenizer, max_length, n_samples):
    """Load JSONL datasets and tokenize"""
    
    print(f"\n📂 Loading datasets (sampling {n_samples} training samples)...")
    
    data_files = {
        "train": f"{data_dir}/train.jsonl",
        "validation": f"{data_dir}/val.jsonl",
        "test": f"{data_dir}/test.jsonl"
    }
    
    dataset = load_dataset("json", data_files=data_files)
    
    # Sample for fast ablation
    train_size = min(n_samples, len(dataset['train']))
    val_size = min(n_samples // 10, len(dataset['validation']))
    test_size = min(500, len(dataset['test']))
    
    dataset['train'] = dataset['train'].select(range(train_size))
    dataset['validation'] = dataset['validation'].select(range(val_size))
    dataset['test'] = dataset['test'].select(range(test_size))
    
    print(f"  ✓ Sampled datasets:")
    print(f"    - Train: {len(dataset['train']):,}")
    print(f"    - Val: {len(dataset['validation']):,}")
    print(f"    - Test: {len(dataset['test']):,}")
    
    print(f"\n🔤 Tokenizing...")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing"
    )
    
    print(f"  ✓ Tokenization complete")
    
    return tokenized_dataset


def train_model(model, tokenizer, tokenized_dataset, output_dir):
    """Train the model with LoRA"""
    
    print("\n⚙️  Training configuration:")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=WARMUP_STEPS,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",  # Disable reporting for speed
        logging_dir=f"{output_dir}/logs",
        push_to_hub=False,
        dataloader_num_workers=4,  # Speed up data loading
    )
    
    total_steps = (len(tokenized_dataset["train"]) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)) * NUM_EPOCHS
    print(f"  - Total steps: {total_steps:,}")
    print(f"  - Batch size: {BATCH_SIZE} × {GRADIENT_ACCUMULATION_STEPS} = {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
    )
    
    print("\n🚀 Starting training...")
    print(f"   Expected time: ~1.5 hours")
    print(f"   Checkpoints: every {SAVE_STEPS} steps\n")
    
    trainer.train()
    
    print("\n✅ Training complete!")
    
    # Save model
    print(f"\n💾 Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Evaluate
    print("\n📊 Evaluating on test set...")
    test_results = trainer.evaluate(tokenized_dataset["test"])
    
    print(f"\n📈 Test Results:")
    for key, value in test_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Save results
    results_file = f"{output_dir}/test_results.json"
    with open(results_file, "w") as f:
        json.dump(test_results, f, indent=2)
    
    # Save metadata
    metadata = {
        "ablation_name": ABLATION_NAME,
        "training_samples": len(tokenized_dataset["train"]),
        "epochs": NUM_EPOCHS,
        "test_results": test_results
    }
    
    with open(f"{output_dir}/ablation_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    return trainer, test_results


def main():
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    model, tokenizer = setup_model_and_tokenizer(MODEL_NAME, use_4bit=USE_4BIT_QUANTIZATION)
    
    tokenized_dataset = load_and_prepare_datasets(
        DATA_DIR, 
        tokenizer, 
        MAX_LENGTH,
        n_samples=ABLATION_SAMPLES
    )
    
    trainer, test_results = train_model(
        model, 
        tokenizer, 
        tokenized_dataset, 
        OUTPUT_DIR
    )
    
    print("\n" + "="*80)
    print("✅ ABLATION COMPLETE")
    print("="*80)
    print(f"\nAblation: {ABLATION_NAME}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"\nKey metrics:")
    print(f"  - Test loss: {test_results.get('eval_loss', 'N/A')}")
    print(f"  - Training samples: {ABLATION_SAMPLES:,}")
    print(f"  - Epochs: {NUM_EPOCHS}")
    
    print("\n📋 Next steps:")
    print("  1. Change ABLATION_NAME to next experiment")
    print("  2. Run again for next ablation")
    print("  3. Compare results across ablations")
    
    return trainer


if __name__ == "__main__":
    trainer = main()
