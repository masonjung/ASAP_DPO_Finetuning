import os
import json
import torch
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from utils import load_training_dataset, get_tokenizer


def load_secrets(secrets_path: str = "secrets.toml") -> dict:
    """Load secrets from TOML file."""
    if os.path.exists(secrets_path):
        with open(secrets_path, "rb") as f:
            secrets = tomllib.load(f)
        print("✅ Secrets loaded from secrets.toml")
        return secrets
    return {}


def resolve_hf_token(config: dict) -> str | None:
    """Return an explicit HF token (env, secrets.toml, or config) and sync env vars for gated repos."""
    # Load from secrets.toml first
    secrets = load_secrets()
    
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        or secrets.get("huggingface", {}).get("token")
        or config.get("hf_token")
    )

    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", token)
        print("✅ HuggingFace token configured")

    return token


def load_config(config_path: str = "config.json") -> dict:
    """Load training configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    print("✅ Configuration loaded")
    return config


def setup_bnb_config(config: dict) -> BitsAndBytesConfig:
    """
    Configure BitsAndBytes for 4-bit quantization.
    
    This reduces memory usage by ~75% while maintaining quality.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["load_in_4bit"],
        bnb_4bit_use_double_quant=config["bnb_4bit_use_double_quant"],
        bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=torch.float16 if config["bnb_4bit_compute_dtype"] == "float16" else torch.bfloat16
    )
    print("✅ BitsAndBytes config created (4-bit quantization enabled)")
    return bnb_config


def load_base_model(model_name: str, bnb_config: BitsAndBytesConfig, token: str | None = None):
    """Load the base model with quantization."""
    print(f"🔄 Loading base model: {model_name}")
    print("⏳ This may take a few minutes...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        use_cache=False,  # Required for gradient checkpointing
        token=token
    )
    
    print("✅ Base model loaded with 4-bit quantization")
    return model


def setup_lora_config(config: dict) -> LoraConfig:
    """
    Configure LoRA (Low-Rank Adaptation).
    
    LoRA adds small trainable matrices to the model, reducing
    trainable parameters by ~99% compared to full fine-tuning.
    """
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    print("✅ LoRA config created")
    return lora_config


def prepare_model_for_training(model, lora_config: LoraConfig):
    """Prepare model for QLoRA training."""
    print("🔧 Preparing model for training...")
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Add LoRA adapters
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model ready for training")
    print(f"📊 Trainable params: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")
    
    return model


def setup_training_args(config: dict) -> TrainingArguments:
    """Configure training arguments."""
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        fp16=config["fp16"],
        bf16=config["bf16"],
        optim=config["optim"],
        gradient_checkpointing=config["gradient_checkpointing"],
        report_to="none",  # Disable wandb, tensorboard, etc.
        push_to_hub=False,
    )
    print("✅ Training arguments configured")
    return training_args


def main():
    """Main training pipeline."""
    print("\n" + "="*60)
    print("🚀 Starting Llama-3.2-1B SFT Training with QLoRA")
    print("="*60 + "\n")
    
    # Step 1: Load configuration
    print("📋 Step 1/7: Loading configuration...")
    config = load_config()
    hf_token = resolve_hf_token(config)
    if not hf_token:
        print("⚠️ No HF token detected. Public gated models (like Llama-3.2) require one.")
    
    # Step 2: Setup quantization
    print("\n📋 Step 2/7: Setting up 4-bit quantization...")
    bnb_config = setup_bnb_config(config)
    
    # Step 3: Load model
    print("\n📋 Step 3/7: Loading base model...")
    model = load_base_model(config["model_name"], bnb_config, hf_token)
    
    # Step 4: Load tokenizer
    print("\n📋 Step 4/7: Loading tokenizer...")
    tokenizer = get_tokenizer(
        config["model_name"],
        config["max_seq_length"],
        token=hf_token
    )
    
    # Step 5: Load and prepare dataset
    print("\n📋 Step 5/7: Loading and preparing dataset...")
    dataset = load_training_dataset(config["dataset_hf"])
    
    # Determine dataset type for formatting
    dataset_name = config["dataset_hf"]
    if "dolly" in dataset_name.lower():
        print("📝 Using Dolly formatting")
    else:
        print("📝 Using custom formatting")
    
    # Step 6: Setup LoRA and prepare model
    print("\n📋 Step 6/7: Setting up LoRA...")
    lora_config = setup_lora_config(config)
    model = prepare_model_for_training(model, lora_config)
    
    # Step 7: Setup training
    print("\n📋 Step 7/7: Setting up trainer...")
    training_args = setup_training_args(config)
    
    # Determine formatting function and prepare dataset
    from utils.formatting import get_formatting_func
    formatting_func = get_formatting_func(dataset_name)
    
    # Format dataset - add "text" column with formatted prompts
    def format_dataset(examples):
        texts = formatting_func(examples)
        return {"text": texts}
    
    formatted_dataset = dataset.map(
        format_dataset,
        batched=True,
        remove_columns=dataset.column_names
    )
    print(f"✅ Dataset formatted: {len(formatted_dataset)} examples")
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=config["max_seq_length"],
        dataset_text_field="text",
        packing=False  # Disable packing for simplicity
    )
    
    print("✅ Trainer ready")
    
    # Start training
    print("\n" + "="*60)
    print("🏋️ Starting training...")
    print("="*60 + "\n")
    
    trainer.train()
    
    # Save final model
    print("\n" + "="*60)
    print("💾 Saving model...")
    print("="*60 + "\n")
    
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    
    print("\n" + "="*60)
    print("✅ Training complete!")
    print(f"📁 Model saved to: {config['output_dir']}")
    print("="*60 + "\n")
    
    print("Next steps:")
    print("1. Run inference: python run_inference.py")
    print("2. Merge LoRA: python merge_lora.py")
    print("3. Check output folder for adapter files")


if __name__ == "__main__":
    main()
