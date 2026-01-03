#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compute_ppl_gap.py
Computes token-level perplexity gap between seen/unseen sets (Table 2).
Uses suffix-level NLL (mean over suffix tokens) -> PPL = exp(NLL).
"""

import argparse
import json
import math
import os
from typing import List, Dict

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def read_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


@torch.no_grad()
def suffix_nll(model, tok, prefix: str, suffix: str, device: torch.device) -> float:
    prefix_ids = tok(prefix, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    full_ids = tok(prefix + suffix, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    labels = full_ids.clone()
    labels[:, : prefix_ids.shape[1]] = -100
    out = model(input_ids=full_ids, labels=labels)
    return float(out.loss.detach().cpu().item())


def mean_nll(model, tok, pairs: List[Dict], device: torch.device, max_items: int = None) -> float:
    if max_items is not None:
        pairs = pairs[:max_items]
    vals = []
    for ex in tqdm(pairs, desc="nll", ncols=100):
        vals.append(suffix_nll(model, tok, ex["prefix"], ex["suffix"], device))
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--seen_pairs", type=str, required=True, help="JSONL with prefix/suffix")
    ap.add_argument("--unseen_pairs", type=str, required=True, help="JSONL with prefix/suffix")
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

    seen = read_jsonl(args.seen_pairs)
    unseen = read_jsonl(args.unseen_pairs)

    seen_nll = mean_nll(model, tok, seen, device, args.max_items)
    unseen_nll = mean_nll(model, tok, unseen, device, args.max_items)

    seen_ppl = float(math.exp(seen_nll))
    unseen_ppl = float(math.exp(unseen_nll))
    gap = float(seen_ppl - unseen_ppl)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "seen_mean_nll": seen_nll,
            "unseen_mean_nll": unseen_nll,
            "seen_ppl": seen_ppl,
            "unseen_ppl": unseen_ppl,
            "ppl_gap": gap,
            "seen_pairs": args.seen_pairs,
            "unseen_pairs": args.unseen_pairs
        }, f, indent=2)

    print(f"[OK] seen_ppl={seen_ppl:.4f} unseen_ppl={unseen_ppl:.4f} gap={gap:.4f} | wrote {args.out}")


if __name__ == "__main__":
    main()
