#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compute_suffix_nll.py
Computes suffix-level NLL for (prefix, suffix) pairs (Table 4 / Fig 5).
"""

import argparse
import json
import math
import os
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def read_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


@torch.no_grad()
def suffix_nll(model, tokenizer, prefix: str, suffix: str, device: torch.device) -> float:
    """
    Mean NLL over suffix tokens conditioned on prefix.
    """
    prefix_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    full_ids = tokenizer(prefix + suffix, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    labels = full_ids.clone()
    labels[:, : prefix_ids.shape[1]] = -100
    out = model(input_ids=full_ids, labels=labels)
    return float(out.loss.detach().cpu().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--pairs_jsonl", type=str, required=True, help="JSONL with {'prefix','suffix'} per line")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_items", type=int, default=None)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    model.eval()

    items = read_jsonl(args.pairs_jsonl)
    if args.max_items is not None:
        items = items[: args.max_items]

    vals = []
    for ex in tqdm(items, desc="suffix_nll", ncols=100):
        vals.append(suffix_nll(model, tok, ex["prefix"], ex["suffix"], device))

    mean = float(np.mean(vals)) if vals else float("nan")
    std = float(np.std(vals)) if vals else float("nan")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "n": len(vals),
            "mean_suffix_nll": mean,
            "std_suffix_nll": std,
            "pairs_jsonl": args.pairs_jsonl
        }, f, indent=2)

    print(f"[OK] mean_suffix_nll={mean:.4f} std={std:.4f} | wrote {args.out}")


if __name__ == "__main__":
    main()
