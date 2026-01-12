import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADAPTER_PATH = REPO_ROOT / "04_models" / "adapters" / "output_dpo"
FALLBACK_OLD_ADAPTER = REPO_ROOT / "output_dpo"


def resolve_adapter_path(adapter_path: Path | str | None) -> Path:
    """Choose new adapter path, falling back to old layout if needed."""
    if adapter_path:
        return Path(adapter_path)
    if DEFAULT_ADAPTER_PATH.exists():
        return DEFAULT_ADAPTER_PATH
    return FALLBACK_OLD_ADAPTER


def load_model_and_tokenizer(
    base_model_name: str,
    adapter_path: Path,
    load_in_4bit: bool,
):
    """Load the base model with fine-tuned LoRA adapters."""
    print(f"Loading model and tokenizer (adapters: {adapter_path})...")

    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        use_cache=True,
    )

    model = PeftModel.from_pretrained(model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.eval()

    print("Model and tokenizer loaded")
    return model, tokenizer


def format_prompt(instruction: str) -> str:
    """Format instruction for inference."""
    return f"""### Instruction:
{instruction.strip()}

### Response:
"""


def generate_response(
    model,
    tokenizer,
    instruction: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> str:
    """Generate a response for the given instruction."""
    prompt = format_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Response:" in full_text:
        response = full_text.split("### Response:", 1)[1].strip()
    else:
        response = full_text
    return response


def main():
    parser = argparse.ArgumentParser(description="Run inference with QLoRA adapters.")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to adapter dir (defaults to 04_models/adapters/output_dpo).")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true", help="Use sampling (default greedy if not set).")
    args = parser.parse_args()

    adapter_path = resolve_adapter_path(args.adapter_path)

    print("\n" + "=" * 60)
    print("Llama-3.2-1B Fine-Tuned Inference")
    print("=" * 60 + "\n")

    model, tokenizer = load_model_and_tokenizer(
        base_model_name=args.base_model,
        adapter_path=adapter_path,
        load_in_4bit=args.load_in_4bit,
    )

    print("\nEnter your instructions (or 'quit' to exit)\n")

    examples = [
        "What is machine learning?",
        "Explain neural networks in simple terms.",
        "What is the difference between AI and ML?",
    ]
    print("Examples:")
    for i, ex in enumerate(examples, 1):
        print(f"   {i}. {ex}")
    print()

    while True:
        instruction = input("Instruction: ").strip()
        if instruction.lower() in ["quit", "exit", "q"]:
            print("\nGoodbye!")
            break
        if not instruction:
            continue

        print("\nGenerating response...\n")
        response = generate_response(
            model,
            tokenizer,
            instruction,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
        )

        print(f"Response:\n{response}\n")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
