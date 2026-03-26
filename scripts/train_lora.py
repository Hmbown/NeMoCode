#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""LoRA fine-tuning for Nemotron 3 Nano 4B on DGX Spark.

Uses Hugging Face transformers + PEFT for native training on GB10.
The BF16 checkpoint (~8GB) fits easily in Spark's 128GB unified memory.

    python scripts/train_lora.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Training config
MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
DATASET_PATH = ".nemocode/data/sft_generated.jsonl"
OUTPUT_DIR = ".nemocode/adapters/nemotron-nano-4b-lora"
LORA_RANK = 16
LORA_ALPHA = 32
EPOCHS = 2
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8  # effective batch = 8
LEARNING_RATE = 2e-4
MAX_SEQ_LEN = 8192
LOGGING_STEPS = 10
SAVE_STEPS = 50


def load_dataset(path: str) -> list[dict]:
    """Load SFT JSONL dataset."""
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {path}")
    return records


def main():
    import torch

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        print("Make sure you're using a venv with CUDA-enabled PyTorch.")
        sys.exit(1)

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    mem = torch.cuda.get_device_properties(0).total_memory
    print(f"Memory: {mem / 1e9:.1f} GB")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig

    # Load dataset
    if not Path(DATASET_PATH).exists():
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        print("Generate it first:")
        print("  nemo data export-seeds")
        print("  nemo data generate")
        sys.exit(1)

    dataset_records = load_dataset(DATASET_PATH)

    # Format for SFT: each record has messages array
    def format_messages(record):
        """Convert messages to chat template format."""
        return {"messages": record["messages"]}

    formatted = [format_messages(r) for r in dataset_records]

    # Split 90/10 train/eval
    split_idx = int(len(formatted) * 0.9)
    train_data = formatted[:split_idx]
    eval_data = formatted[split_idx:]
    print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

    # Load tokenizer
    print(f"\nLoading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Nemotron 3 Nano 4B BF16 (~8GB — fits easily on Spark's 128GB)
    print(f"Loading model {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.config.use_cache = False

    # Configure LoRA
    print(f"Applying LoRA (rank={LORA_RANK}, alpha={LORA_ALPHA})...")
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training config
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    from datasets import Dataset
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        bf16=True,
        max_length=MAX_SEQ_LEN,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        report_to="none",
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        dataset_text_field=None,  # Using messages format
    )

    # Create trainer
    print("\nStarting training...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Train
    trainer.train()

    # Save the LoRA adapter
    print(f"\nSaving LoRA adapter to {output_dir}...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"\nDone! LoRA adapter saved to {output_dir}")
    print(f"To serve with vLLM:")
    print(f"  vllm serve {MODEL_ID} --enable-lora --lora-modules nemocode={output_dir}")
    print(f"Or use nemo serve:")
    print(f"  nemo serve start --model nemotron-3-nano-4b --adapter {output_dir}")


if __name__ == "__main__":
    main()
