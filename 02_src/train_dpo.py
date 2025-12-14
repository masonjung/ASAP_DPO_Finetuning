import os
import argparse
import json
import random
import logging
from pathlib import Path

import torch
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOConfig, DPOTrainer

from utils import load_training_dataset, get_tokenizer, prepare_dataset_for_dpo

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "00_configs" / "dpo.json"
DEFAULT_SECRETS_PATH = REPO_ROOT / "00_configs" / "secrets.toml"

# --------------------------------------------------------------------------- #
# Compatibility patch:
# HF Transformers 4.46.x introduced Trainer.get_batch_samples(epoch_iterator, num_batches)
# which clashes with TRL's DPOTrainer.get_batch_samples(model, batch).
# When HF Trainer calls the TRL override, the first arg is the dataloader
# iterator (a generator), causing AttributeError: 'generator' object has no attribute 'generate'.
# We dispatch based on the arguments to preserve both behaviors.
# --------------------------------------------------------------------------- #
_orig_dpo_get_batch_samples = DPOTrainer.get_batch_samples


def _patched_get_batch_samples(self, *args, **kwargs):
    # Called by HF Trainer: (epoch_iterator, num_batches)
    if len(args) == 2 and not hasattr(args[0], "generate"):
        epoch_iterator, num_batches = args
        batch_samples = []
        num_items_in_batch = None
        for _ in range(num_batches):
            try:
                batch_samples.append(next(epoch_iterator))
            except StopIteration:
                break

        if not self.model_accepts_loss_kwargs:
            return batch_samples, None

        if len(batch_samples) > 0 and "labels" in batch_samples[0]:
            try:
                num_items_in_batch = sum([(batch["labels"].ne(-100)).sum() for batch in batch_samples])
            except (TypeError, AttributeError):
                pass

        if getattr(self.args, "average_tokens_across_devices", False):
            num_items_in_batch = self.accelerator.gather(num_items_in_batch).sum().item()
        return batch_samples, num_items_in_batch

    # TRL usage: (model, batch)
    return _orig_dpo_get_batch_samples(self, *args, **kwargs)


DPOTrainer.get_batch_samples = _patched_get_batch_samples

# HF Trainer (4.46.x) passes num_items_in_batch to compute_loss; allow it.
_orig_dpo_compute_loss = DPOTrainer.compute_loss


def _patched_compute_loss(self, *args, **kwargs):
    kwargs.pop("num_items_in_batch", None)
    return _orig_dpo_compute_loss(self, *args, **kwargs)


DPOTrainer.compute_loss = _patched_compute_loss


def load_secrets(secrets_path: Path = DEFAULT_SECRETS_PATH) -> dict:
    """Load secrets from TOML file."""
    secrets_path = Path(secrets_path)
    if secrets_path.exists():
        with open(secrets_path, "rb") as f:
            secrets = tomllib.load(f)
        print(f"Secrets loaded from {secrets_path}")
        return secrets
    return {}


def resolve_hf_token(config: dict) -> str | None:
    """Return an explicit HF token (env, secrets.toml, or config) and sync env vars for gated repos."""
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
        print("HuggingFace token configured")

    return token


def load_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> dict:
    """Load training configuration from JSON file."""
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    print(f"Configuration loaded from {config_path}")
    
    # Validate required keys
    required_keys = ["model_name", "dataset_hf", "output_dir", "max_seq_length", "lora_target_modules"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    return config


def setup_bnb_config(config: dict) -> BitsAndBytesConfig:
    """Configure BitsAndBytes for 4-bit quantization."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["load_in_4bit"],
        bnb_4bit_use_double_quant=config["bnb_4bit_use_double_quant"],
        bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=torch.float16 if config["bnb_4bit_compute_dtype"] == "float16" else torch.bfloat16,
    )
    print("BitsAndBytes config created (4-bit quantization enabled)")
    return bnb_config


def load_base_model(
    model_name: str,
    bnb_config: BitsAndBytesConfig,
    token: str | None = None,
    use_cache: bool = False,
):
    """Load the base model with quantization."""
    print(f"Loading base model: {model_name}")
    print("This may take a few minutes...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=bnb_config.bnb_4bit_compute_dtype,
        use_cache=use_cache,
        token=token,
    )

    print(f"Base model loaded with 4-bit quantization (use_cache={use_cache})")
    return model


def setup_lora_config(config: dict) -> LoraConfig:
    """Configure LoRA (Low-Rank Adaptation)."""
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    print("LoRA config created")
    return lora_config


def prepare_policy_model_for_training(model, lora_config: LoraConfig):
    """Prepare model for QLoRA DPO training."""
    print("Preparing policy model for training...")

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    if hasattr(model, "config"):
        model.config.use_cache = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable_params / total_params
    print(f"Policy model ready")
    print(f"Trainable params: {trainable_params:,} / {total_params:,} ({pct:.2f}%)")

    return model


def setup_training_args(config: dict) -> DPOConfig:
    """Configure training arguments for DPOTrainer."""
    training_args = DPOConfig(
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
        seed=config.get("seed", 42),
        dataloader_num_workers=config.get("dataloader_num_workers", 0),
        remove_unused_columns=False,  # Needed for DPOTrainer
        max_length=config["max_seq_length"],
        max_prompt_length=config.get("max_prompt_length", config["max_seq_length"]),
        max_target_length=config.get("max_target_length", max(256, config["max_seq_length"] // 2)),
    )
    print("Training arguments configured")
    return training_args


def set_reproducibility(seed: int):
    """Set seeds for reproducibility and deterministic CUDA kernels."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def preflight_checks(config: dict, hf_token: str | None):
    """Run lightweight sanity checks before launching training."""
    errors = []
    warnings = []

    dataset_path = config.get("dataset_hf")
    if dataset_path:
        if os.path.exists(dataset_path):
            if os.path.getsize(dataset_path) == 0:
                errors.append(f"Dataset file is empty: {dataset_path}")
            else:
                
                try:
                    with open(dataset_path, "r", encoding="utf-8") as f:
                        first_line = next((ln for ln in f if ln.strip()), None)
                    if first_line:
                        sample = json.loads(first_line)
                        for key in ("instruction", "chosen", "rejected"):
                            if key not in sample:
                                warnings.append(f"Local dataset missing key '{key}' in first sample.")
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"Could not inspect local dataset ({exc}).")
        else:
            warnings.append(f"Dataset path not found locally: {dataset_path}. Will try Hugging Face hub.")

    max_seq_length = config.get("max_seq_length")
    max_prompt_length = config.get("max_prompt_length", max_seq_length)
    max_target_length = config.get("max_target_length", max_seq_length // 2)
    if max_seq_length and max_prompt_length and max_target_length:
        if (max_prompt_length + max_target_length) > max_seq_length:
            warnings.append(
                "max_prompt_length + max_target_length exceeds max_seq_length; responses may be truncated."
            )

    if hf_token is None:
        warnings.append("No HF token detected; gated models like Llama-3.2 will fail to download.")

    if os.name == "nt" and config.get("load_in_4bit", False):
        warnings.append("4-bit bitsandbytes is unreliable on native Windows; use WSL or disable load_in_4bit.")

    if not torch.cuda.is_available():
        warnings.append("CUDA not available. Training will fall back to CPU and be extremely slow.")
    else:
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            total_gb = total_mem / (1024**3)
            if total_gb < 8:
                warnings.append(f"GPU reports {total_gb:.1f} GB total; may OOM with current settings.")
        except Exception:
            pass

    if errors:
        for msg in errors:
            print(f"[ERROR] {msg}")
        raise RuntimeError("Preflight checks failed; fix the above issues before training.")

    if warnings:
        print("\nPreflight warnings:")
        for msg in warnings:
            print(f" - {msg}")
        print()


def parse_args():
    parser = argparse.ArgumentParser(description="Train Llama-3.2-1B with QLoRA + DPO.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to config JSON.")
    parser.add_argument("--dry_run", action="store_true", help="Run preflight checks and exit without training.")
    return parser.parse_args()


def main():
    """Main DPO training pipeline."""
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        filename="05_logs/training.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info("Starting DPO training")

    print("\n" + "=" * 60)
    print("Starting Llama-3.2-1B DPO Training with QLoRA")
    print("=" * 60 + "\n")

    print("Step 1/8: Loading configuration...")
    config = load_config(args.config)
    seed = config.get("seed", 42)
    set_reproducibility(seed)
    hf_token = resolve_hf_token(config)
    if not hf_token:
        print("No HF token detected. Public gated models (like Llama-3.2) require one.")

    print("Running preflight checks...")
    preflight_checks(config, hf_token)
    if args.dry_run:
        print("Dry run complete. Exiting before model/dataset load.")
        return

    print("\nStep 2/8: Setting up 4-bit quantization...")
    bnb_config = setup_bnb_config(config)

    print("\nStep 3/8: Loading policy model...")
    policy_model = load_base_model(
        config["model_name"],
        bnb_config,
        hf_token,
        use_cache=False,  # needed for gradient checkpointing
    )

    print("\nStep 4/8: Loading tokenizer...")
    tokenizer = get_tokenizer(
        config["model_name"],
        config["max_seq_length"],
        token=hf_token,
    )

    print("\nStep 5/8: Loading and preparing dataset...")
    # Resolve dataset path to absolute
    dataset_path = config["dataset_hf"]
    if not os.path.isabs(dataset_path):
        dataset_path = str(REPO_ROOT / dataset_path)
    config["dataset_hf"] = dataset_path

    dataset = load_training_dataset(
        config["dataset_hf"],
        split=config.get("dataset_split", "train"),
        max_samples=config.get("max_train_samples"),
        token=hf_token,
    )

    # Validate dataset has required columns
    if "chosen" not in dataset.column_names or "rejected" not in dataset.column_names:
        raise ValueError("Dataset must contain 'chosen' and 'rejected' columns for DPO")

    if config.get("dataset_shuffle", True):
        dataset = dataset.shuffle(seed=seed)

    dataset_name = config["dataset_hf"]
    if "dolly" in dataset_name.lower():
        print("Using Dolly formatting for DPO")
    else:
        print("Using custom formatting for DPO")

    print("\nStep 6/8: Setting up LoRA...")
    lora_config = setup_lora_config(config)
    policy_model = prepare_policy_model_for_training(policy_model, lora_config)

    print("\nStep 7/8: Loading reference model...")
    ref_model = load_base_model(
        config["model_name"],
        bnb_config,
        hf_token,
        use_cache=True,  # inference-only; keep cache on for speed
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    print("\nStep 8/8: Setting up DPO trainer...")
    training_args = setup_training_args(config)

    formatted_dataset = prepare_dataset_for_dpo(
        dataset,
        dataset_name=dataset_name,
        num_proc=config.get("num_proc"),
    )
    if len(formatted_dataset) == 0:
        raise ValueError("No DPO pairs found after formatting. Ensure your dataset has chosen/rejected responses.")
    print(f"Dataset formatted for DPO: {len(formatted_dataset)} examples")

    beta = float(config.get("dpo_beta", 0.1))

    trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        beta=beta,
        train_dataset=formatted_dataset,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("DPO trainer ready")

    print("\n" + "=" * 60)
    print("Starting DPO training...")
    print("=" * 60 + "\n")

    try:
        trainer.train()
        logging.info("Training completed successfully")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise

    print("\n" + "=" * 60)
    print("Saving policy LoRA adapter and tokenizer...")
    print("=" * 60 + "\n")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logging.info(f"Model saved to {output_dir}")

    print("\n" + "=" * 60)
    print("DPO training complete!")
    print(f"Policy adapter saved to: {config['output_dir']}")
    print("=" * 60 + "\n")

    print("Next steps:")
    print("1. Run inference with adapter: python 02_src/run_inference.py --adapter_path 04_models/adapters/output_dpo")
    print("2. Merge LoRA: python 02_src/merge_lora.py --adapter_path 04_models/adapters/output_dpo --output_path 04_models/merged/merged_model_dpo")


if __name__ == "__main__":
    main()
