"""
Merge LoRA adapters into the base model to create a standalone fine-tuned model.

Usage:
    python 02_src/merge_lora.py --adapter_path 04_models/adapters/output_dpo --output_path 04_models/merged/llama32_dpo_merged
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADAPTER_PATH = REPO_ROOT / "04_models" / "adapters" / "output_dpo"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "04_models" / "merged" / "merged_model_dpo"
FALLBACK_OLD_ADAPTER = REPO_ROOT / "output_dpo"


def resolve_adapter_path(adapter_path: Path | str | None) -> Path:
    """Choose new adapter path, falling back to old layout if needed."""
    if adapter_path:
        return Path(adapter_path)
    if DEFAULT_ADAPTER_PATH.exists():
        return DEFAULT_ADAPTER_PATH
    return FALLBACK_OLD_ADAPTER


def merge_lora_to_base(
    base_model_name: str,
    adapter_path: Path,
    output_path: Path,
    push_to_hub: bool = False,
    hub_model_name: str | None = None,
):
    """Merge LoRA adapters into the base model."""
    print("\n" + "=" * 60)
    print("Merging LoRA adapters into base model")
    print("=" * 60 + "\n")

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("Base model loaded")

    print("\nLoading LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    print("LoRA adapters loaded")

    print("\nMerging adapters...")
    merged_model = model.merge_and_unload()
    print("Adapters merged successfully")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    print("Tokenizer loaded")

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving merged model to: {output_path}")
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    print("Merged model saved")

    if push_to_hub:
        if hub_model_name is None:
            raise ValueError("hub_model_name must be provided if push_to_hub=True")

        print(f"\nPushing to Hugging Face Hub as: {hub_model_name}")
        merged_model.push_to_hub(hub_model_name)
        tokenizer.push_to_hub(hub_model_name)
        print("Model pushed to Hub")

    print("\nMerge complete!")
    print(f"Merged model location: {output_path}")
    print("Use like any HF model:")
    print(f'  model = AutoModelForCausalLM.from_pretrained("{output_path}")')
    print(f'  tokenizer = AutoTokenizer.from_pretrained("{output_path}")')


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into base model")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Base model name or path")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter directory")
    parser.add_argument("--output_path", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Where to save the merged model")
    parser.add_argument("--push_to_hub", action="store_true", help="Push merged model to Hugging Face Hub")
    parser.add_argument("--hub_model_name", type=str, default=None, help="Model name for Hugging Face Hub (required if --push_to_hub)")
    args = parser.parse_args()

    adapter_path = resolve_adapter_path(args.adapter_path)
    output_path = Path(args.output_path)

    merge_lora_to_base(
        base_model_name=args.base_model,
        adapter_path=adapter_path,
        output_path=output_path,
        push_to_hub=args.push_to_hub,
        hub_model_name=args.hub_model_name,
    )


if __name__ == "__main__":
    main()
