 # DPO + QLoRA Fine-Tuning for Oil & Gas Abbretivation Grounding
[![Reproducible](https://img.shields.io/badge/Reproducible-Yes-success.svg)](#)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/masonjung/ASAP_DPO_Finetuning?style=social)](https://github.com/masonjung/ASAP_DPO_Finetuning/stargazers)


<div align="center">
  <img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/214f57d1-4637-4bd5-8e0b-032c1a1f7d7a" />
</div>
 


## Overview
This project fine-tunes a small language model (SLM) using Direct Preference Optimization (DPO) with QLoRA adapters to ground responses in the Oil and Gas domain. Unlike other projects, you can post-train the SLM locally with the given small domain dataset for hands-on-experiment. This AMP is a great starting point to train your private AI with DPO on a laptop or desktop without a remote-GPU. Start with your private model quickly and extend the AMP with larger models, larger datasets, and additional GPUs. This project require GPU for Compute Unified Device Architecture (CUDA) usage and Hugging Face (HF) to load an LLM. 

## Why DPO?

When should you use DPO instead of Supervised Fine-Tuning (SFT)?

| Method | Best For | Data Format |
|--------|----------|-------------|
| **SFT** | Teaching the model *what* to say | Input → Correct output |
| **DPO** | Teaching the model *which response is better* | Input → Preferred vs Rejected |

**Use DPO when:**
- You want to correct terminology preferences ("use 'DP' for Dynamic Positioning, not Demographic Parity")
- You need to adjust tone, style, or formatting behavior
- You have examples of good vs bad responses, not just good responses
- You're aligning model behavior rather than adding new knowledge

**Use SFT when:**
- You're teaching the model new domain knowledge
- You have high-quality instruction-response pairs
- There's a single "correct" answer for each input

**Why DPO over RLHF?**

Traditional RLHF requires training a separate reward model, then using reinforcement learning (PPO) to optimize against it. This is complex, unstable, and compute-intensive. DPO achieves equivalent results with a simple classification loss on preference pairs—no reward model, no RL loop, and significantly easier to tune.

This makes DPO ideal for on-device fine-tuning where computational resources are limited.

## Project Workflow
1. Edit configurations in `00_configs/dpo.json`.
2. Prepare a dataset for preference pairs in JSONL (Prepared; instruction, chosen, rejected) in `01_data/dpo/train.jsonl`.
3. Run DPO training and merge the adapter with `dpo_training.ipynb`. 
4. Optional: run the training on `02_src/train_dpo.py`, which serves for the back-end of (3). 

## Key Features
- On-device DPO training with QLoRA adapters (4-bit quantization supported; faster) that does not need a remote GPU.
- Hands-on workflow on Jupyter Notebook in `dpo_training.ipynb`.

## Hardware check
- Please consider the hardware condition using `01_data/gpu_check.py` or the GPU check block of the `dpo_training.ipynb`. You need enough memory to run this code (e.g.,mem_free/total_GB: 3.5/4.3). Important: you need enough GPU (e.g. 2 GB or greater) and CPU memory (e.g, 8 GB or greater)!!

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
```

## Data Format
Training data is JSONL with preference pairs:
```json
{"instruction": "...", "chosen": "...", "rejected": "..."}
```
In this project, we use the paired dataset for domain vocabulary training. For example, 'DP' from Dynamic Positioning', that is defined as a computer-controlled system that automatically maintains a vessel's position and heading using its own propellers and thrusters, eliminating the need for anchors. In contrast, the term 'DP' is Demographic Parity in AI fairness research and the abbreviation has a different meaning.

## Configuration
You need to get approval from the model usage agreement. For example, to use Llama family, you need to get approval via "LLAMA {number} COMMUNITY LICENSE AGREEMENT".

We suggest to use the token file `00_configs/secrets.toml` for HF token:
```toml
[huggingface]
token = "hf_betrue123..."
```

you can start with the given `00_configs/dpo.json` based on your needs:
- `model_name`: base model (default Llama-3.2-1B-Instruct. If you would like to use a larger model, you may need remote GPUs).
- `dataset_hf`: local JSONL path. This could be switched to another dataset.
- `output_dir`: adapter output path.
- `load_in_4bit`: enable QLoRA quantization.
- `dpo_beta`, `learning_rate`, `num_train_epochs`, and more hyperparameters are adjustable.


## Usage Guide

### Evaluation
Since this project focuses on fine-tuning for a domain abbreviation and SLM that shows a worse performance compared to LLM, we did not integrate a general evaluation metrics. If you plan to expand the domain with greater scale with larger datasets, you should bring up evaluation metrics to measure an overall performance.

## Advanced Customization

### Swap Base Models
Set `model_name` in `00_configs/dpo.json` for training.

Example (config snippet):
```json
{
  "model_name": "microsoft/Phi-4-mini-instruct"
}
```

Common alternatives:
- `Qwen/Qwen3-4B`
- `google/gemma-2b-it`

Notes:
- Gated models require an HF token and accepted model terms.
- Larger models typically need smaller `per_device_train_batch_size` or `max_seq_length`; keep `load_in_4bit: true` for VRAM.
- If the model uses different module names, update `lora_target_modules` accordingly.


### Swap Datasets
Point `dataset_hf` at another JSONL or Hugging Face dataset. Ensure it has `instruction`, `chosen`, and `rejected`.

### Prompt Templates
Adjust templates in `02_src/utils/formatting.py` for a tailored formatting.
