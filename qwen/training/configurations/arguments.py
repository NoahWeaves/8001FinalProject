"""
* File: training_args.py
* Author: Loic Martins
* Date: 2025-11-16
* Description:
    Specific parameters for the training.
"""

# Import external Libraries
from dataclasses import dataclass


@dataclass
class TrainingArgs:
    output_dir: str = "./model/fine_tuned_model"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    save_total_limit: int = 2
    save_strategy: str = "epoch"
    logging_strategy: str = "steps"
    logging_steps: int = 500
    gradient_checkpointing: bool = True
    bf16: bool = False
    fp16: bool = True
    max_grad_norm: float = 1.0
