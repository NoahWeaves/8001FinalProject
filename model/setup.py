"""
* File: runner.py (training)
* Author: Loic Martins
* Date: 2025-11-16
* Description:
    SetUpModel class to setup the tokenizer, the specific model and apply LoRA.
"""

# Import External Libraries
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

# Import Local Modules
from training.configurations.config import TrainingConfig


class SetUpModel:
    def __init__(
        self,
        config: TrainingConfig,
        fine_tuned_path: str | None = None,
        training_bool: bool = True,
    ) -> None:
        """
        Set up the environment for training or inference:
            -Tokenizer
            -Original Model or Fine-Tuned Model
            -LoRA for fine-tuning

        Args:
            config (TrainingConfig): The necessary config to setup the model.
            fine_tuned_path (str): The path to access to the fine-tuned model files.

        Returns:
            None
        """
        self.model_name: str = config.model_name
        self.fine_tuned_path: str | None = fine_tuned_path

        # Step 1: Set the specific device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self.device: torch.device = device

        # Step 2: Download and load tokenizer --for the original model or the fine-tuned model
        tokenizer_source = self.fine_tuned_path or self.model_name
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            tokenizer_source, trust_remote_code=True
        )
        # Set specific pad tokens --padding sequences
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Step 3: Download and load model + Move the model on the GPU (will use cache if already downloaded)
        self.model: PreTrainedModel | PeftModel = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Step 4: If fine-tuned model --load fine-tuned adapter
        if self.fine_tuned_path is not None:
            self.model = PeftModel.from_pretrained(self.model, self.fine_tuned_path)
            print("====Fine tuned model setup.====")

        # Step 5: Move the model to the GPU --if available
        if self.device.type != "cpu":
            self.model.to(device)  # pyright: ignore[reportArgumentType]

        # Step 6: If training --Optimization
        if training_bool is True:
            self._optimization_training()

    def _optimization_training(self) -> None:
        """
        Set specific parameters to manage memory during training.
        Args: None
        Returns: None
        """
        self.model.gradient_checkpointing_enable()  # pyright: ignore[reportCallIssue]
        self.model.config.use_cache = False  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]

    def setup_lora(self) -> None:
        """
        Set LoRA method for training.
        Args: None
        Returns: None
        """
        if isinstance(self.model, PreTrainedModel):
            model = self.model

            # Step 1: Define LoRA configuration
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )

            # Step 2: Apply LoRA to the model
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            self.model = model  # pyright: ignore
        else:
            print("LoRA is not available because fine_tuned_path is not None.")

    def generate(self, input_text: str) -> str:
        """
        Generate text using the original or fine-tuned model.

        Args:
            input_text (str): The input text --prompt

        Returns:
            generated_text (str): The generated text.
        """

        # Step 1: Tokenize
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)  # pyright: ignore[reportArgumentType]

        # Step 2: Generate (deterministic)
        gen_tokens = self.model.generate(  # pyright: ignore[reportCallIssue]
            **inputs,
            max_new_tokens=4,
            do_sample=False,
            temperature=0.0,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Step 3: Extract only new tokens (after the input)
        new_tokens = gen_tokens[:, inputs["input_ids"].shape[-1] :]  # pyright: ignore[reportAttributeAccessIssue]
        generated_text = self.tokenizer.decode(
            new_tokens[0], skip_special_tokens=True
        ).strip()

        return generated_text
