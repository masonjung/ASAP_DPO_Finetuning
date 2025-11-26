"""
Dataset loading and tokenization utilities.

This module handles loading datasets from Hugging Face or local files,
and prepares them for training by applying formatting and tokenization.
"""

import os
import json
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from .formatting import get_formatting_func


def load_training_dataset(dataset_path: str, split: str = "train"):
    """
    Load a dataset from Hugging Face Hub or local JSONL file.
    
    Args:
        dataset_path: Either a Hugging Face dataset ID or path to local JSONL
        split: Dataset split to load (default: 'train')
        
    Returns:
        Hugging Face Dataset object
    """
    # Check if it's a local file
    if os.path.exists(dataset_path):
        print(f"📂 Loading local dataset from: {dataset_path}")
        
        # Load JSONL file
        data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        
        dataset = Dataset.from_list(data)
        print(f"✅ Loaded {len(dataset)} examples from local file")
        
    else:
        # Load from Hugging Face Hub
        print(f"🌐 Loading dataset from Hugging Face: {dataset_path}")
        dataset = load_dataset(dataset_path, split=split)
        print(f"✅ Loaded {len(dataset)} examples from Hugging Face")
    
    return dataset


def prepare_dataset_for_training(
    dataset,
    tokenizer: AutoTokenizer,
    max_seq_length: int = 512,
    dataset_name: str = "custom"
):
    """
    Prepare a dataset for training by formatting and tokenizing.
    
    Args:
        dataset: Hugging Face Dataset object
        tokenizer: Tokenizer to use
        max_seq_length: Maximum sequence length
        dataset_name: Name of dataset (used to select formatting function)
        
    Returns:
        Formatted dataset ready for training
    """
    formatting_func = get_formatting_func(dataset_name)
    
    # Apply formatting - formatting_func now returns {"text": [list of strings]}
    print("🔄 Formatting dataset...")
    
    formatted_dataset = dataset.map(
        formatting_func,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    print(f"✅ Dataset formatted with {len(formatted_dataset)} examples")
    
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
    print(f"🔧 Loading tokenizer for: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
        token=token
    )
    
    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure tokenizer
    tokenizer.padding_side = "right"  # Important for decoder-only models
    tokenizer.model_max_length = max_seq_length
    
    print(f"✅ Tokenizer loaded (vocab size: {len(tokenizer)})")
    
    return tokenizer
