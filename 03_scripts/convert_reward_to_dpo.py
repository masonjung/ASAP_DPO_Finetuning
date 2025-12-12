"""
Convert an instruction/response/reward JSONL dataset into DPO preference pairs.

Each output line will contain: {"instruction": ..., "context": ..., "chosen": ..., "rejected": ...}
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert reward-labeled responses into DPO pairs.")
    parser.add_argument("--input", type=Path, default=Path("01_data/raw/sample.jsonl"), help="Path to reward JSONL.")
    parser.add_argument("--output", type=Path, default=Path("01_data/dpo/train.jsonl"), help="Path to write DPO JSONL.")
    parser.add_argument("--instruction_field", type=str, default="instruction", help="Field name for the prompt/instruction.")
    parser.add_argument("--response_field", type=str, default="response", help="Field name for the model response.")
    parser.add_argument("--reward_field", type=str, default="reward", help="Field name for the reward score.")
    parser.add_argument("--context_field", type=str, default="context", help="Optional context field to carry over.")
    parser.add_argument(
        "--pair_all",
        action="store_true",
        help="If set, create best-vs-each-worse pairs. Otherwise only best vs worst.",
    )
    parser.add_argument(
        "--min_gap",
        type=float,
        default=0.0,
        help="Skip pairs where (best_reward - other_reward) <= min_gap to avoid noisy ties.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def group_by_instruction(
    rows: List[Dict[str, Any]],
    instruction_field: str,
    response_field: str,
    reward_field: str,
    context_field: str,
) -> dict:
    buckets: dict[str, List[Tuple[str, float, Dict[str, Any]]]] = defaultdict(list)
    for row in rows:
        instruction = str(row.get(instruction_field, "")).strip()
        response = row.get(response_field)
        reward_val = as_float(row.get(reward_field))
        if not instruction or response is None or reward_val is None:
            continue
        buckets[instruction].append((str(response).strip(), reward_val, row))
    return buckets


def build_pairs(
    buckets: dict,
    context_field: str,
    min_gap: float,
    pair_all: bool,
) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for instruction, items in buckets.items():
        if len(items) < 2:
            continue

        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
        best_resp, best_reward, best_row = sorted_items[0]

        if pair_all:
            for resp, reward, row in sorted_items[1:]:
                if (best_reward - reward) <= min_gap or best_resp == resp:
                    continue
                context = best_row.get(context_field) or row.get(context_field)
                pairs.append(
                    {
                        "instruction": instruction,
                        "context": context,
                        "chosen": best_resp,
                        "rejected": resp,
                    }
                )
        else:
            worst_resp, worst_reward, worst_row = sorted_items[-1]
            if (best_reward - worst_reward) <= min_gap or best_resp == worst_resp:
                continue
            context = best_row.get(context_field) or worst_row.get(context_field)
            pairs.append(
                {
                    "instruction": instruction,
                    "context": context,
                    "chosen": best_resp,
                    "rejected": worst_resp,
                }
            )
    return pairs


def write_pairs(pairs: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            # Drop empty context to keep files tidy
            payload = {k: v for k, v in pair.items() if v not in (None, "")}
            json.dump(payload, f, ensure_ascii=False)
            f.write("\n")


def main():
    args = parse_args()
    print(f"Loading reward dataset from: {args.input}")
    rows = load_rows(args.input)
    print(f"Loaded {len(rows)} rows")

    buckets = group_by_instruction(
        rows,
        instruction_field=args.instruction_field,
        response_field=args.response_field,
        reward_field=args.reward_field,
        context_field=args.context_field,
    )
    print(f"Found {len(buckets)} unique instructions")

    pairs = build_pairs(
        buckets,
        context_field=args.context_field,
        min_gap=args.min_gap,
        pair_all=args.pair_all,
    )
    print(f"Built {len(pairs)} DPO pairs")

    if not pairs:
        print("No valid pairs were created. Check rewards, ties, or min_gap settings.")
        return

    write_pairs(pairs, args.output)
    print(f"Saved DPO JSONL to: {args.output}")
    print("Example line:")
    print(json.dumps(pairs[0], ensure_ascii=False))


if __name__ == "__main__":
    main()
