"""
Merge LoRA adapters into the base model to create a standalone fine-tuned model.

This creates a full model that doesn't require loading adapters separately.

Usage:
    python merge_lora.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora_to_base(
    base_model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    adapter_path: str = "./output_llama32_sft",
    output_path: str = "./merged_model",
    push_to_hub: bool = False,
    hub_model_name: str = None
):
    """
    Merge LoRA adapters into the base model.
    
    Args:
        base_model_name: Name of the base model
        adapter_path: Path to LoRA adapter directory
        output_path: Where to save the merged model
        push_to_hub: Whether to push to Hugging Face Hub
        hub_model_name: Model name for the Hub (if pushing)
    """
    print("\n" + "="*60)
    print("🔀 Merging LoRA Adapters into Base Model")
    print("="*60 + "\n")
    
    # Load base model in full precision (fp16)
    print("📥 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print("✅ Base model loaded")
    
    # Load LoRA adapters
    print("\n📥 Loading LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    print("✅ LoRA adapters loaded")
    
    # Merge adapters into base model
    print("\n🔀 Merging adapters...")
    merged_model = model.merge_and_unload()
    print("✅ Adapters merged successfully")
    
    # Load tokenizer
    print("\n📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    print("✅ Tokenizer loaded")
    
    # Save merged model
    print(f"\n💾 Saving merged model to: {output_path}")
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    print("✅ Merged model saved")
    
    # Push to Hub if requested
    if push_to_hub:
        if hub_model_name is None:
            raise ValueError("hub_model_name must be provided if push_to_hub=True")
        
        print(f"\n🚀 Pushing to Hugging Face Hub as: {hub_model_name}")
        merged_model.push_to_hub(hub_model_name)
        tokenizer.push_to_hub(hub_model_name)
        print("✅ Model pushed to Hub")
    
    print("\n" + "="*60)
    print("✅ Merge Complete!")
    print("="*60 + "\n")
    
    print("📁 Merged model location:", output_path)
    print("\nYou can now use this model like any other Hugging Face model:")
    print(f'  model = AutoModelForCausalLM.from_pretrained("{output_path}")')
    print(f'  tokenizer = AutoTokenizer.from_pretrained("{output_path}")')


def main():
    """Main merge function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into base model")
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Base model name or path"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="./output_llama32_sft",
        help="Path to LoRA adapter directory"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./merged_model",
        help="Where to save the merged model"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push merged model to Hugging Face Hub"
    )
    parser.add_argument(
        "--hub_model_name",
        type=str,
        default=None,
        help="Model name for Hugging Face Hub (required if --push_to_hub)"
    )
    
    args = parser.parse_args()
    
    merge_lora_to_base(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        push_to_hub=args.push_to_hub,
        hub_model_name=args.hub_model_name
    )


if __name__ == "__main__":
    main()
