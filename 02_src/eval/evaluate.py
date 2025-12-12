"""
Quick A/B evaluator for base vs QLoRA-tuned model.

What it does:
- Loads a prompt set from JSONL (one instruction per line, optional "reference").
- Generates with the base model and with adapters (fine-tuned).
- Measures latency, tokens/sec, and simple overlap vs reference (if provided).
- Prints a small before/after summary table.

Usage:
    python 02_src/eval/evaluate.py ^
        --prompts_path 01_data/eval/eval_prompts.jsonl ^
        --adapter_path 04_models/adapters/output_dpo ^
        --base_model meta-llama/Llama-3.2-1B-Instruct

Notes:
- Keep max_new_tokens modest for fair latency comparison.
- Set the same decoding params for both runs.
"""

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS = REPO_ROOT / "01_data" / "eval" / "eval_prompts.jsonl"
DEFAULT_ADAPTER_PATH = REPO_ROOT / "04_models" / "adapters" / "output_dpo"
FALLBACK_OLD_ADAPTER = REPO_ROOT / "output_dpo"


def resolve_adapter_path(adapter_path: Path | str | None) -> Path | None:
    """Choose new adapter path, falling back to old layout if needed."""
    if adapter_path:
        return Path(adapter_path)
    if DEFAULT_ADAPTER_PATH.exists():
        return DEFAULT_ADAPTER_PATH
    if FALLBACK_OLD_ADAPTER.exists():
        return FALLBACK_OLD_ADAPTER
    return None


def load_prompts(path: Path | str):
    """Load prompts (and optional references) from JSONL."""
    path = Path(path)
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            prompts.append(
                {
                    "instruction": obj.get("instruction", "").strip(),
                    "reference": obj.get("reference", obj.get("response", "")).strip(),
                }
            )
    return prompts


def format_prompt(instruction: str) -> str:
    """Format instruction for generation."""
    return f"""### Instruction:
{instruction.strip()}

### Response:
"""


def extract_response(full_text: str) -> str:
    """Extract the response section from generated text."""
    if "### Response:" in full_text:
        return full_text.split("### Response:", 1)[1].strip()
    return full_text.strip()


def build_bnb_config(load_in_4bit: bool) -> BitsAndBytesConfig | None:
    """Create BitsAndBytes config if 4-bit is requested."""
    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_model_and_tokenizer(base_model: str, adapter_path: Path | None, load_in_4bit: bool):
    """Load base model and optional adapters."""
    bnb_config = build_bnb_config(load_in_4bit)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        use_cache=True,
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    tokenizer_source = adapter_path if adapter_path else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def generate_one(model, tokenizer, instruction: str, gen_kwargs: dict) -> tuple[str, dict]:
    """Generate one response and capture timing/length stats."""
    prompt = format_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **gen_kwargs,
        )
    elapsed = time.perf_counter() - start

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = extract_response(full_text)

    input_tokens = inputs["input_ids"].shape[1]
    output_tokens = outputs[0].shape[0] - input_tokens
    tokens_per_sec = output_tokens / elapsed if elapsed > 0 else float("inf")

    stats = {
        "latency_s": elapsed,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_per_sec": tokens_per_sec,
    }
    return response, stats


def simple_overlap_score(pred: str, reference: str) -> float:
    """Jaccard overlap over lowercase word sets as a light proxy."""
    if not reference:
        return math.nan
    pred_tokens = set(pred.lower().split())
    ref_tokens = set(reference.lower().split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    intersection = len(pred_tokens & ref_tokens)
    union = len(pred_tokens | ref_tokens)
    return intersection / union if union else 0.0


def evaluate_split(model, tokenizer, prompts, gen_kwargs):
    """Run generation over prompts and collect metrics."""
    results = []
    for item in prompts:
        resp, stats = generate_one(model, tokenizer, item["instruction"], gen_kwargs)
        overlap = simple_overlap_score(resp, item.get("reference", ""))
        results.append(
            {
                "instruction": item["instruction"],
                "reference": item.get("reference", ""),
                "response": resp,
                "overlap": overlap,
                "latency_s": stats["latency_s"],
                "output_tokens": stats["output_tokens"],
                "tokens_per_sec": stats["tokens_per_sec"],
            }
        )
    return results


def summarize(results: list[dict]) -> dict:
    """Aggregate latency and overlap metrics."""
    latencies = [r["latency_s"] for r in results]
    tps = [r["tokens_per_sec"] for r in results if math.isfinite(r["tokens_per_sec"])]
    overlaps = [r["overlap"] for r in results if not math.isnan(r["overlap"])]
    non_empty_rate = sum(1 for r in results if r["response"].strip()) / len(results) if results else 0.0

    def pct(values, p):
        return statistics.quantiles(values, n=100)[p - 1] if values else math.nan

    summary = {
        "count": len(results),
        "latency_p50_s": statistics.median(latencies) if latencies else math.nan,
        "latency_p95_s": pct(latencies, 95),
        "tokens_per_sec_p50": statistics.median(tps) if tps else math.nan,
        "tokens_per_sec_p95": pct(tps, 95),
        "overlap_mean": statistics.mean(overlaps) if overlaps else math.nan,
        "non_empty_rate": non_empty_rate,
    }
    return summary


def print_summary(label: str, summary: dict):
    """Pretty-print a summary block."""
    print(f"\n=== {label} ===")
    print(f"Examples          : {summary['count']}")
    print(f"Latency  p50 (s)  : {summary['latency_p50_s']:.3f}" if not math.isnan(summary["latency_p50_s"]) else "Latency  p50 (s)  : n/a")
    print(f"Latency  p95 (s)  : {summary['latency_p95_s']:.3f}" if not math.isnan(summary["latency_p95_s"]) else "Latency  p95 (s)  : n/a")
    print(f"Tokens/s p50      : {summary['tokens_per_sec_p50']:.1f}" if not math.isnan(summary["tokens_per_sec_p50"]) else "Tokens/s p50      : n/a")
    print(f"Tokens/s p95      : {summary['tokens_per_sec_p95']:.1f}" if not math.isnan(summary["tokens_per_sec_p95"]) else "Tokens/s p95      : n/a")
    print(f"Overlap mean      : {summary['overlap_mean']:.3f}" if not math.isnan(summary["overlap_mean"]) else "Overlap mean      : n/a")
    print(f"Non-empty rate    : {summary['non_empty_rate']*100:.1f}%")


def save_outputs_csv(path: Path, rows: list[dict]):
    """Save combined results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "instruction",
        "reference",
        "base_response",
        "tuned_response",
        "base_latency_s",
        "tuned_latency_s",
        "base_tokens_per_sec",
        "tuned_tokens_per_sec",
        "overlap_base",
        "overlap_tuned",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_outputs_jsonl(path: Path, rows: list[dict]):
    """Save combined results to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="A/B evaluate base vs QLoRA-tuned model.")
    parser.add_argument("--prompts_path", type=str, default=str(DEFAULT_PROMPTS), help="JSONL with 'instruction' and optional 'reference' fields.")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter directory (defaults to 04_models/adapters/output_dpo).")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Base model name or path.")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization for evaluation.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Generation length.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling.")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling (default: greedy).")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of prompts for quick tests.")
    parser.add_argument("--output_csv", type=str, default=None, help="Optional path to save combined results as CSV.")
    parser.add_argument("--output_jsonl", type=str, default=None, help="Optional path to save combined results as JSONL.")
    args = parser.parse_args()

    adapter_path = resolve_adapter_path(args.adapter_path)

    prompts = load_prompts(args.prompts_path)
    if args.limit:
        prompts = prompts[: args.limit]

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": args.do_sample,
    }

    print("Loading base model...")
    base_model, base_tok = load_model_and_tokenizer(args.base_model, adapter_path=None, load_in_4bit=args.load_in_4bit)
    base_results = evaluate_split(base_model, base_tok, prompts, gen_kwargs)
    base_summary = summarize(base_results)
    print_summary("Base", base_summary)

    print("\nLoading fine-tuned model (base + adapters)...")
    tuned_model, tuned_tok = load_model_and_tokenizer(args.base_model, adapter_path=adapter_path, load_in_4bit=args.load_in_4bit)
    tuned_results = evaluate_split(tuned_model, tuned_tok, prompts, gen_kwargs)
    tuned_summary = summarize(tuned_results)
    print_summary("Fine-tuned (QLoRA)", tuned_summary)

    print("\nDelta (tuned - base):")
    if not math.isnan(tuned_summary["overlap_mean"]) and not math.isnan(base_summary["overlap_mean"]):
        print(f"Overlap mean delta  : {tuned_summary['overlap_mean'] - base_summary['overlap_mean']:+.3f}")
    if not math.isnan(tuned_summary["latency_p50_s"]) and not math.isnan(base_summary["latency_p50_s"]):
        print(f"Latency p50 delta s : {tuned_summary['latency_p50_s'] - base_summary['latency_p50_s']:+.3f}")
    if not math.isnan(tuned_summary["tokens_per_sec_p50"]) and not math.isnan(base_summary["tokens_per_sec_p50"]):
        print(f"Tokens/s p50 delta  : {tuned_summary['tokens_per_sec_p50'] - base_summary['tokens_per_sec_p50']:+.1f}")

    combined_rows = []
    for base_row, tuned_row in zip(base_results, tuned_results):
        combined_rows.append(
            {
                "instruction": base_row["instruction"],
                "reference": base_row.get("reference", ""),
                "base_response": base_row["response"],
                "tuned_response": tuned_row["response"],
                "base_latency_s": base_row["latency_s"],
                "tuned_latency_s": tuned_row["latency_s"],
                "base_tokens_per_sec": base_row["tokens_per_sec"],
                "tuned_tokens_per_sec": tuned_row["tokens_per_sec"],
                "overlap_base": base_row["overlap"],
                "overlap_tuned": tuned_row["overlap"],
            }
        )

    if args.output_csv:
        save_outputs_csv(Path(args.output_csv), combined_rows)
        print(f"\nWrote CSV to {args.output_csv}")
    if args.output_jsonl:
        save_outputs_jsonl(Path(args.output_jsonl), combined_rows)
        print(f"Wrote JSONL to {args.output_jsonl}")


if __name__ == "__main__":
    main()
