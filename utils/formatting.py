"""
Prompt formatting utilities for SFT (Supervised Fine-Tuning).

This module provides functions to convert instruction-response pairs
into a standardized format that the model can learn from.
"""


def format_instruction_response(instruction: str, response: str) -> str:
    """
    Format an instruction-response pair into the standard SFT template.
    
    Args:
        instruction: The input instruction/question
        response: The expected output/answer
        
    Returns:
        Formatted string ready for training
    """
    return f"""### Instruction:
{instruction.strip()}

### Response:
{response.strip()}"""


def format_dolly_example(examples: dict) -> list:
    """Return a list of formatted Dolly prompts for SFTTrainer."""
    instructions = examples.get("instruction", [])
    contexts = examples.get("context", [])
    responses = examples.get("response", [])

    # Normalise to lists for batch + single example compatibility
    if not isinstance(instructions, list):
        instructions = [instructions]
        contexts = [contexts]
        responses = [responses]

    formatted_texts = []
    for instruction, context, response in zip(instructions, contexts, responses):
        if context and str(context).strip():
            full_instruction = f"{str(instruction).strip()}\n\nContext: {str(context).strip()}"
        else:
            full_instruction = str(instruction).strip()

        formatted_texts.append(
            format_instruction_response(full_instruction, str(response))
        )

    return formatted_texts


def format_custom_example(example: dict) -> list:
    """Return a single formatted prompt for custom datasets."""
    instruction = example.get("instruction", "")
    response = example.get("response", "")

    return [format_instruction_response(instruction, response)]


def get_formatting_func(dataset_name: str):
    """
    Return the appropriate formatting function for a dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'dolly', 'custom')
        
    Returns:
        Formatting function
    """
    if "dolly" in dataset_name.lower():
        return format_dolly_example
    else:
        return format_custom_example
