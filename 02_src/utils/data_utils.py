"""
Dataset loading and tokenization utilities.

Handles loading from HF Hub or local JSONL, basic cleaning, and formatting.
"""

import json
import os
from typing import Optional

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from .formatting import get_dpo_formatting_func, get_formatting_func


def _filter_empty(example: dict) -> bool:
    """Drop rows with empty instruction or target (response or chosen)."""
    instruction = str(example.get("instruction", "")).strip()
    target = str(example.get("response") or example.get("chosen", "")).strip()
    return bool(instruction) and bool(target)


def _filter_preference_row(example: dict) -> bool:
    """Ensure preference rows have both chosen and rejected responses."""
    chosen = str(example.get("chosen", "")).strip()
    rejected = str(example.get("rejected", "")).strip()
    return bool(chosen) and bool(rejected) and chosen != rejected


def load_training_dataset(
    dataset_path: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    token: Optional[str] = None,
):
    """
    Load a dataset from Hugging Face Hub or local JSONL file.

    Args:
        dataset_path: HF dataset ID or path to local JSONL
        split: Dataset split to load (default: 'train')
        max_samples: Optional cap for quick experiments
        token: Optional Hugging Face token for gated/private datasets

    Returns:
        Hugging Face Dataset object
    """
    if os.path.exists(dataset_path):
        print(f"Loading local dataset from: {dataset_path}")
        rows = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    rows.append(row)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_num} in {dataset_path}: {e}")
        if not rows:
            raise ValueError(f"No valid JSON lines found in {dataset_path}")
        dataset = Dataset.from_list(rows)
        print(f"Loaded {len(dataset)} examples from local file")
    else:
        print(f"Loading dataset from Hugging Face: {dataset_path} (split={split})")
        dataset = load_dataset(dataset_path, split=split, token=token)
        print(f"Loaded {len(dataset)} examples from Hugging Face")

    if max_samples is not None:
        max_samples = min(max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))
        print(f"Capped dataset to {max_samples} examples for this run")

    dataset = dataset.filter(_filter_empty)
    print(f"After filtering empty rows: {len(dataset)} examples")

    return dataset


def prepare_dataset_for_training(
    dataset,
    tokenizer: AutoTokenizer,
    max_seq_length: int = 512,
    dataset_name: str = "custom",
    num_proc: Optional[int] = None,
):
    """
    Prepare a dataset for training by formatting and tokenizing.

    Args:
        dataset: Hugging Face Dataset object
        tokenizer: Tokenizer to use
        max_seq_length: Maximum sequence length
        dataset_name: Name of dataset (used to select formatting function)
        num_proc: Optional number of processes for mapping

    Returns:
        Formatted dataset ready for training
    """
    formatting_func = get_formatting_func(dataset_name)
    print("Formatting dataset...")

    formatted_dataset = dataset.map(
        formatting_func,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
    )

    print(f"Dataset formatted with {len(formatted_dataset)} examples")
    return formatted_dataset


def prepare_dataset_for_dpo(
    dataset,
    dataset_name: str = "custom",
    num_proc: Optional[int] = None,
):
    """
    Prepare a dataset for DPO by formatting prompts and pairing chosen/rejected responses.

    Args:
        dataset: Hugging Face Dataset object
        dataset_name: Name of dataset (used to select formatting function)
        num_proc: Optional number of processes for mapping

    Returns:
        Formatted dataset ready for DPOTrainer
    """
    formatting_func = get_dpo_formatting_func(dataset_name)
    print("Formatting dataset for DPO...")

    dataset = dataset.filter(_filter_preference_row)
    formatted_dataset = dataset.map(
        formatting_func,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        load_from_cache_file=False,
    )

    print(f"Dataset formatted for DPO with {len(formatted_dataset)} examples")
    return formatted_dataset


def get_tokenizer(model_name: str, max_seq_length: int = 512, token: str | None = None):
    """
    Load and configure tokenizer for training.

    Args:
        model_name: Name of the model (used to load tokenizer)
        max_seq_length: Maximum sequence length

    Returns:
        Configured tokenizer
    """
    print(f"Loading tokenizer for: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
        token=token,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"  # Important for decoder-only models
    tokenizer.model_max_length = max_seq_length

    print(f"Tokenizer loaded (vocab size: {len(tokenizer)})")
    return tokenizer
