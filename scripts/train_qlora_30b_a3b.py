#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""4-bit QLoRA fine-tuning for Nemotron 3 Nano 30B-A3B on DGX Spark.

Uses the official NVIDIA BF16 checkpoint as the source model and loads it in
4-bit via bitsandbytes for local QLoRA training on GB10.

The model is a hybrid MoE/Mamba architecture, so this script keeps the local
path conservative:
  * default max sequence length is 2048
  * LoRA target modules are discovered dynamically from the loaded model
  * `--preflight` runs a single worst-case step before a full launch

Examples:
    python scripts/train_qlora_30b_a3b.py --preflight
    python scripts/train_qlora_30b_a3b.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DEFAULT_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
DEFAULT_DATASET_PATH = ".nemocode/data/sft_generated.jsonl"
DEFAULT_OUTPUT_DIR = ".nemocode/adapters/nemotron-nano-30b-a3b-qlora"
DEFAULT_MAX_SEQ_LEN = 2048
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_EPOCHS = 2
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRADIENT_ACCUMULATION = 8
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_LOGGING_STEPS = 10
DEFAULT_SAVE_STEPS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF model ID.")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_PATH,
        help="Path to customizer-style JSONL with messages arrays.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save LoRA checkpoints/adapters.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=DEFAULT_MAX_SEQ_LEN,
        help="Training truncation length. Keep conservative until preflight passes.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION,
        help="Gradient accumulation steps.",
    )
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run a single step on the longest example to validate load and wiring.",
    )
    return parser.parse_args()


def load_dataset(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {path}")
    return records


def token_count(tokenizer, messages: list[dict]) -> int:
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if isinstance(encoded, dict):
        return len(encoded["input_ids"])
    return len(encoded)


def split_dataset(records: list[dict]) -> tuple[list[dict], list[dict]]:
    formatted = [{"messages": record["messages"]} for record in records]
    split_idx = int(len(formatted) * 0.9)
    return formatted[:split_idx], formatted[split_idx:]


def find_longest_record(records: list[dict], tokenizer) -> tuple[int, dict]:
    longest_idx = -1
    longest_record: dict | None = None
    longest_len = -1
    for idx, record in enumerate(records, 1):
        current_len = token_count(tokenizer, record["messages"])
        if current_len > longest_len:
            longest_idx = idx
            longest_len = current_len
            longest_record = record
    if longest_record is None:
        raise RuntimeError("No records found in dataset.")
    return longest_idx, longest_record


def find_lora_target_modules(model) -> list[str]:
    preferred_suffixes = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "in_proj",
        "out_proj",
        "x_proj",
        "dt_proj",
    }

    discovered = set()
    fallback = set()

    for name, module in model.named_modules():
        if not name or name.endswith("lm_head"):
            continue
        class_name = module.__class__.__name__.lower()
        if class_name not in {"linear4bit", "linear"}:
            continue
        leaf_name = name.split(".")[-1]
        fallback.add(leaf_name)
        if leaf_name in preferred_suffixes:
            discovered.add(leaf_name)

    targets = sorted(discovered or fallback)
    if not targets:
        raise RuntimeError("Could not discover LoRA target modules from the quantized model.")
    return targets


def print_risks(args: argparse.Namespace) -> None:
    print("\nLocal 30B-A3B QLoRA assumptions:")
    print("  - Base model is the official NVIDIA BF16 checkpoint, loaded in 4-bit NF4.")
    print("  - Default max sequence length is 2048 to stay within a conservative local budget.")
    print("  - The model is hybrid MoE/Mamba, so preflight is recommended before full training.")
    if args.max_seq_len > DEFAULT_MAX_SEQ_LEN:
        print("  - WARNING: max sequence length exceeds the conservative 2048 default.")
    if not args.preflight:
        print("  - Full training mode assumes a successful preflight has already been run.")


def main() -> None:
    import torch
    from bitsandbytes import __version__ as bnb_version
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    print(f"PyTorch: {torch.__version__}")
    print(f"bitsandbytes: {bnb_version}")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    mem = torch.cuda.get_device_properties(0).total_memory
    print(f"Memory: {mem / 1e9:.1f} GB")
    print_risks(args)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        sys.exit(1)

    records = load_dataset(str(dataset_path))

    print(f"\nLoading tokenizer for {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.preflight:
        record_idx, record = find_longest_record(records, tokenizer)
        token_len = token_count(tokenizer, record["messages"])
        train_rows = [{"messages": record["messages"]}]
        eval_rows: list[dict] = []
        print(
            f"Preflight mode: longest record is #{record_idx} at {token_len} tokens before truncation."
        )
        output_dir = Path(f"{args.output_dir}-preflight")
        max_steps = 1
        eval_strategy = "no"
        eval_steps = None
        save_steps = 1000
    else:
        train_rows, eval_rows = split_dataset(records)
        print(f"Train: {len(train_rows)}, Eval: {len(eval_rows)}")
        output_dir = Path(args.output_dir)
        max_steps = -1
        eval_strategy = "steps"
        eval_steps = DEFAULT_SAVE_STEPS
        save_steps = DEFAULT_SAVE_STEPS

    output_dir.mkdir(parents=True, exist_ok=True)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"Loading 4-bit model {args.model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quant_config,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    print(
        "Memory after model load:",
        f"{tuple(round(x / 1e9, 1) for x in torch.cuda.mem_get_info())} GB free/total",
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    target_modules = find_lora_target_modules(model)
    print(f"LoRA target modules: {', '.join(target_modules)}")

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows) if eval_rows else None

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=DEFAULT_LOGGING_STEPS if not args.preflight else 1,
        save_steps=save_steps,
        save_total_limit=2,
        bf16=True,
        max_length=args.max_seq_len,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        report_to="none",
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        dataset_text_field=None,
    )

    print("\nStarting training...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    print(f"\nSaving LoRA adapter to {output_dir}...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    if args.preflight:
        print("\nPreflight complete. The 30B-A3B QLoRA path loaded and executed one step.")
        print("If you need more headroom, keep max_seq_len at 2048 and stop other GPU jobs first.")
    else:
        print(f"\nDone! QLoRA adapter saved to {output_dir}")
        print(f"To serve with vLLM:")
        print(
            "  "
            f"vllm serve {args.model_id} --enable-lora --lora-modules nemocode={output_dir}"
        )


if __name__ == "__main__":
    main()
