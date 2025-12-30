# DPO + QLoRA Fine Tuning for Oil & Gas Domain Grounding

## Overview
This project fine-tunes a small language model (SLM) using Direct Preference Optimization (DPO) with QLoRA adapters to ground responses in the Oil and Gas domain. Unlike other projects, you can post-train the SLM locally with the given small domain dataset for hands-on-experiment. This AMP is a great starting point to train your private AI with DPO on a laptop or desktop without a remote-GPU. Start with your private model quickly and extend the AMP with larger models, larger datasets, and additional GPUs. This project require GPU for Compute Unified Device Architecture (CUDA) usage and Hugging Face (HF) to load an LLM. 


## Project Workflow
1. Edit configurations in `00_configs/dpo.json`.
2. Prepare a dataset for preference pairs in JSONL (PREPARED; instruction, chosen, rejected) in `01_data/dpo/train.jsonl`.
3. Run DPO training and merge the adapter with `dpo_training.ipynb` 
4. Optional: run the training on `02_src/train_dpo.py`, which serves for the back-end of (3). 

## Key Features
- On-device DPO training with QLoRA adapters (4-bit quantization supported; faster) that does not need GPU networks.
- Clickable workflow on Jupyter Notebook in `dpo_training.ipynb`.


## Repository Layout
- `00_configs/` - training configs and secrets.
- `01_data/` - training data.
- `02_src/` - training, inference, and utilities
- `03_scripts/` - helper scripts.
- `04_models/` - adapter and merged model outputs.
- `05_logs/` - training logs.

## Getting Started

### Prerequisites
- Python 3.10+ (match your PyTorch/CUDA build).
- CUDA-capable GPU recommended (8GB VRAM typical for LLM of 1B + QLoRA).
- HF token for gated models (set `HF_TOKEN` or `00_configs/secrets.toml`).
- Policy consent on HF to use `meta-llama/Llama-3.2-1B-Instruct` model 

### Project Setup
```bash
pip install -r requirements.txt
# Optional: CUDA wheels (example)
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### Quick Start
```bash
python 02_src/train_dpo.py --config 00_configs/dpo.json
python 02_src/run_inference.py --adapter_path 04_models/adapters/output_dpo
python 02_src/eval/evaluate.py --prompts_path 01_data/eval/eval_prompts.jsonl --adapter_path 04_models/adapters/output_dpo
python 02_src/merge_lora.py --adapter_path 04_models/adapters/output_dpo --output_path 04_models/merged/merged_model_dpo
```

### Windows Helper
```bat
03_scripts\run.bat
```

## Data Format
Training data is JSONL with preference pairs:
```json
{"instruction": "...", "chosen": "...", "rejected": "...", "context": "..."}
```

If you start with reward-scored responses:
```json
{"instruction": "...", "response": "...", "reward": 1.0}
```
Convert with:
```bash
python 03_scripts/convert_reward_to_dpo.py --input 01_data/raw/sample.jsonl --output 01_data/dpo/train.jsonl
```

## Configuration
Edit `00_configs/dpo.json`:
- `model_name`: base model (default Llama-3.2-1B-Instruct).
- `dataset_hf`: local JSONL path or Hugging Face dataset ID.
- `output_dir`: adapter output path.
- `load_in_4bit`: enable QLoRA quantization.
- `dpo_beta`, `learning_rate`, `num_train_epochs`, and more.

Optional token file `00_configs/secrets.toml`:
```toml
[huggingface]
token = "hf_..."
```

## Usage Guide

### Fine-Tuning Workflow
1. Prepare or convert data to DPO format.
2. Update `00_configs/dpo.json` for model, data, and output paths.
3. Run training: `python 02_src/train_dpo.py --config 00_configs/dpo.json`.
4. Monitor logs: `python 02_src/monitor_training.py`.

### Evaluation
```bash
python 02_src/eval/evaluate.py --prompts_path 01_data/eval/eval_prompts.jsonl --adapter_path 04_models/adapters/output_dpo
```

### Inference
```bash
python 02_src/run_inference.py --adapter_path 04_models/adapters/output_dpo
```

### Merge LoRA
```bash
python 02_src/merge_lora.py --adapter_path 04_models/adapters/output_dpo --output_path 04_models/merged/merged_model_dpo
```

## Advanced Customization

### Swap Base Models
Update `model_name` in `00_configs/dpo.json` or pass `--base_model` to inference or merge scripts.

### Swap Datasets
Point `dataset_hf` at another JSONL or Hugging Face dataset. Ensure it has `instruction`, `chosen`, and `rejected` fields or adapt in `02_src/utils/formatting.py`.

### Prompt Templates
Adjust templates in `02_src/utils/formatting.py` for domain-specific formatting.

## Notebook Walkthrough
`dpo_training.ipynb` mirrors the training pipeline if you prefer a notebook flow.
