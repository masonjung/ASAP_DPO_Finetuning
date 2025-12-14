"""Prompt formatting utilities for DPO and instruction-response data."""


def _build_instruction_block(instruction: str, context: str | None = None) -> str:
    """Return the instruction block, optionally appending context."""
    instruction = str(instruction).strip()
    if context and str(context).strip():
        return f"{instruction}\n\nContext: {str(context).strip()}"
    return instruction


def format_instruction_response(instruction: str, response: str, context: str | None = None) -> str:
    """
    Format an instruction-response pair into the standard DPO-style template.

    Args:
        instruction: The input instruction/question
        response: The expected output/answer
        context: Optional context for the instruction

    Returns:
        Formatted string ready for training
    """
    instruction_block = _build_instruction_block(instruction, context)
    return f"""### Instruction:
{instruction_block}

### Response:
{str(response).strip()}"""


def format_dpo_prompt(instruction: str, context: str | None = None) -> str:
    """
    Format the prompt used by DPO. We keep the same template but leave the
    response blank so chosen/rejected answers can be paired separately.
    """
    instruction_block = _build_instruction_block(instruction, context)
    return f"""### Instruction:
{instruction_block}

### Response:"""


def format_dolly_example(examples: dict) -> dict:
    """Return formatted Dolly prompts for supervised trainer ingestion."""
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
        formatted_texts.append(
            format_instruction_response(instruction, str(response), context)
        )

    return {"text": formatted_texts}


def format_custom_example(example: dict) -> dict:
    """Return a formatted prompt for custom datasets."""
    instruction = example.get("instruction", "")
    response = example.get("response", "")
    context = example.get("context")

    return {"text": format_instruction_response(instruction, response, context)}


def format_dolly_dpo_example(examples: dict) -> dict:
    """Return Dolly-formatted prompts for DPOTrainer."""
    instructions = examples.get("instruction", [])
    contexts = examples.get("context", [])
    chosens = examples.get("chosen", [])
    rejecteds = examples.get("rejected", [])

    if not isinstance(instructions, list):
        instructions = [instructions]
        contexts = [contexts]
        chosens = [chosens]
        rejecteds = [rejecteds]

    # If context is missing or length-mismatched, pad with Nones
    if not isinstance(contexts, list):
        contexts = [contexts] * len(instructions)
    if len(contexts) != len(instructions):
        contexts = (contexts + [None] * len(instructions))[: len(instructions)]

    prompts, chosen_texts, rejected_texts = [], [], []
    for instruction, context, chosen, rejected in zip(instructions, contexts, chosens, rejecteds):
        prompts.append(format_dpo_prompt(instruction, context))
        chosen_texts.append(str(chosen).strip())
        rejected_texts.append(str(rejected).strip())

    return {"prompt": prompts, "chosen": chosen_texts, "rejected": rejected_texts}


def format_custom_dpo_example(example: dict) -> dict:
    """Return DPO-formatted prompt for custom datasets."""
    instructions = example.get("instruction", [])
    contexts = example.get("context", [])
    chosens = example.get("chosen", [])
    rejecteds = example.get("rejected", [])

    if not isinstance(instructions, list):
        instructions = [instructions]
        contexts = [contexts]
        chosens = [chosens]
        rejecteds = [rejecteds]

    # If context is missing or length-mismatched, pad with Nones
    if not isinstance(contexts, list):
        contexts = [contexts] * len(instructions)
    if len(contexts) != len(instructions):
        contexts = (contexts + [None] * len(instructions))[: len(instructions)]

    prompts, chosen_texts, rejected_texts = [], [], []
    for instruction, context, chosen, rejected in zip(instructions, contexts, chosens, rejecteds):
        prompts.append(format_dpo_prompt(instruction, context))
        chosen_texts.append(str(chosen).strip())
        rejected_texts.append(str(rejected).strip())

    return {"prompt": prompts, "chosen": chosen_texts, "rejected": rejected_texts}


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


def get_dpo_formatting_func(dataset_name: str):
    """Return the appropriate formatting function for DPO datasets."""
    if "dolly" in dataset_name.lower():
        return format_dolly_dpo_example
    else:
        return format_custom_dpo_example
