#!/usr/bin/env python3
"""
Build a class-balanced training set for UniPACT by downsampling the majority
class (yes/no) within each task file.

Inputs (10 LLaVA-format JSON files in --input-dir):
    train_deterioration.json
    train_diagnose_00.json ... train_diagnose_06.json   (7 files)
    train_icu.json
    train_mortality.json

For every file independently, entries are split into pos (gpt answer starts
with "yes") and neg (starts with "no"); the larger class is downsampled to
match the smaller. All balanced subsets are concatenated and shuffled.

Run twice with different seeds to obtain two non-identical training sets:

    python scripts/make_balanced_split.py --input-dir <DIR> \\
        --output <DIR>/balanced_stage1_seed42.json --seed 42
    python scripts/make_balanced_split.py --input-dir <DIR> \\
        --output <DIR>/balanced_stage2_seed43.json --seed 43
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path


TASK_FILES = [
    "train_deterioration.json",
    "train_diagnose_00.json",
    "train_diagnose_01.json",
    "train_diagnose_02.json",
    "train_diagnose_03.json",
    "train_diagnose_04.json",
    "train_diagnose_05.json",
    "train_diagnose_06.json",
    "train_icu.json",
    "train_mortality.json",
]


def extract_answer(entry):
    convs = entry.get("conversations", [])
    for turn in reversed(convs):
        if turn.get("from") in ("gpt", "model", "assistant"):
            return turn.get("value", "")
    return ""


def classify(answer):
    a = answer.strip().lower().lstrip("\"'`*").rstrip("\"'`*.!?, \n\t")
    if a.startswith("yes"):
        return "pos"
    if a.startswith("no"):
        return "neg"
    return "other"


def balance_file(path, rng):
    with open(path) as f:
        data = json.load(f)
    pos, neg, other = [], [], []
    for entry in data:
        bucket = classify(extract_answer(entry))
        {"pos": pos, "neg": neg, "other": other}[bucket].append(entry)
    n = min(len(pos), len(neg))
    sampled = rng.sample(pos, n) + rng.sample(neg, n)
    return sampled, len(pos), len(neg), len(other)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True,
                    help="directory containing the 10 train_*.json files")
    ap.add_argument("--output", required=True,
                    help="output JSON path for the balanced merged split")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    combined = []
    missing = []

    print(f"seed={args.seed}  input_dir={args.input_dir}")
    print(f"{'file':<32}{'pos':>10}{'neg':>10}{'kept':>10}{'other':>8}")
    print("-" * 70)
    for fname in TASK_FILES:
        path = os.path.join(args.input_dir, fname)
        if not os.path.exists(path):
            missing.append(fname)
            print(f"{fname:<32}{'MISSING':>40}")
            continue
        sampled, n_pos, n_neg, n_other = balance_file(path, rng)
        print(f"{fname:<32}{n_pos:>10}{n_neg:>10}{len(sampled):>10}{n_other:>8}")
        combined.extend(sampled)

    if missing:
        print(f"\nERROR: missing input files: {missing}", file=sys.stderr)
        sys.exit(1)

    rng.shuffle(combined)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(combined, f)
    print("-" * 70)
    print(f"total kept = {len(combined)}  ->  {out}")


if __name__ == "__main__":
    main()
