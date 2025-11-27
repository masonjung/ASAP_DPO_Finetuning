"""
Run inference with the fine-tuned Llama-3.2 model.

This script loads the base model + LoRA adapters and generates responses.

Usage:
    python run_inference.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def load_model_and_tokenizer(
    base_model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    adapter_path: str = "./output_llama32_sft",
    load_in_4bit: bool = True
):
    """
    Load the base model with fine-tuned LoRA adapters.
    
    Args:
        base_model_name: Name of the base model
        adapter_path: Path to the LoRA adapter directory
        load_in_4bit: Whether to use 4-bit quantization
        
    Returns:
        model, tokenizer
    """
    print("🔄 Loading model and tokenizer...")
    
    # Setup quantization config if needed
    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    print("✅ Model and tokenizer loaded")
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
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True
) -> str:
    """
    Generate a response for the given instruction.
    
    Args:
        model: The model to use
        tokenizer: The tokenizer to use
        instruction: Input instruction/question
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        do_sample: Whether to use sampling
        
    Returns:
        Generated response text
    """
    # Format prompt
    prompt = format_prompt(instruction)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode and extract only the response part
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract response (everything after "### Response:")
    if "### Response:" in full_text:
        response = full_text.split("### Response:")[1].strip()
    else:
        response = full_text
    
    return response


def main():
    """Main inference loop."""
    print("\n" + "="*60)
    print("🤖 Llama-3.2-1B Fine-Tuned Inference")
    print("="*60 + "\n")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    
    print("\n💡 Enter your instructions (or 'quit' to exit)\n")
    
    # Example instructions
    examples = [
        "What is machine learning?",
        "Explain neural networks in simple terms.",
        "What is the difference between AI and ML?"
    ]
    
    print("📝 Example instructions:")
    for i, ex in enumerate(examples, 1):
        print(f"   {i}. {ex}")
    print()
    
    # Interactive loop
    while True:
        instruction = input("👤 Instruction: ").strip()
        
        if instruction.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not instruction:
            continue
        
        print("\n🤖 Generating response...\n")
        
        response = generate_response(
            model,
            tokenizer,
            instruction,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9
        )
        
        print(f"🤖 Response:\n{response}\n")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
