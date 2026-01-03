#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
frequency_buckets.py
Builds high/mid/rare frequency tiers from a corpus, outputting (prefix, suffix) pairs JSONL.
Used for Table 4 / Fig 5 in Memory Dial paper.

Input corpus: one text per line.
Method:
  1) tokenize all texts, count token frequencies
  2) for each text, compute avg log token frequency
  3) bucket texts by quantiles into high/mid/rare
  4) from each text, split into prefix/suffix of fixed token lengths and export JSONL pairs
"""

import argparse
import json
import math
import os
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm


def read_lines(path: str, max_lines: int = None) -> List[str]:
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            t = line.strip()
            if t:
                lines.append(t)
    return lines


def write_jsonl(path: str, items: List[Dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, required=True, help="Text file: one example per line")
    ap.add_argument("--tokenizer", type=str, required=True, help="Tokenizer name/path (match your model family)")
    ap.add_argument("--max_lines", type=int, default=200000, help="Cap for building token freq (speed)")
    ap.add_argument("--min_tokens", type=int, default=64, help="Only keep texts with enough tokens")
    ap.add_argument("--prefix_tokens", type=int, default=32)
    ap.add_argument("--suffix_tokens", type=int, default=32)
    ap.add_argument("--samples_per_bucket", type=int, default=500)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    lines = read_lines(args.corpus, max_lines=args.max_lines)
    print(f"[info] loaded {len(lines)} lines")

    # 1) token frequency
    freq = Counter()
    tokenized = []
    for t in tqdm(lines, desc="tokenize+count", ncols=100):
        ids = tok(t, add_special_tokens=False).input_ids
        if len(ids) >= args.min_tokens:
            tokenized.append(ids)
            freq.update(ids)

    if not tokenized:
        raise SystemExit("No texts left after min_tokens filter. Lower --min_tokens or increase corpus.")

    # Precompute log-freq (add-1 smoothing)
    def avg_log_freq(ids: List[int]) -> float:
        vals = [math.log(freq[i] + 1.0) for i in ids]
        return float(sum(vals) / max(1, len(vals)))

    scores = np.array([avg_log_freq(ids) for ids in tokenized], dtype=np.float64)

    # 2) quantile thresholds
    # High-frequency = top third (largest avg log freq), Rare = bottom third
    q1 = np.quantile(scores, 1/3)
    q2 = np.quantile(scores, 2/3)

    rare_idx = np.where(scores <= q1)[0].tolist()
    mid_idx  = np.where((scores > q1) & (scores <= q2))[0].tolist()
    high_idx = np.where(scores > q2)[0].tolist()

    rng = np.random.default_rng(42)
    rng.shuffle(rare_idx); rng.shuffle(mid_idx); rng.shuffle(high_idx)

    def make_pairs(idxs: List[int], tag: str) -> List[Dict]:
        out = []
        for j in idxs[: args.samples_per_bucket]:
            ids = tokenized[j]
            # take a window from start; you can randomize window too, but keep deterministic here
            p_ids = ids[: args.prefix_tokens]
            s_ids = ids[args.prefix_tokens : args.prefix_tokens + args.suffix_tokens]
            if len(p_ids) < args.prefix_tokens or len(s_ids) < args.suffix_tokens:
                continue
            prefix = tok.decode(p_ids)
            suffix = tok.decode(s_ids)
            out.append({"tier": tag, "prefix": prefix, "suffix": suffix})
        return out

    rare_pairs = make_pairs(rare_idx, "rare")
    mid_pairs  = make_pairs(mid_idx, "mid")
    high_pairs = make_pairs(high_idx, "high")

    os.makedirs(args.out_dir, exist_ok=True)
    write_jsonl(os.path.join(args.out_dir, "rare_pairs.jsonl"), rare_pairs)
    write_jsonl(os.path.join(args.out_dir, "mid_pairs.jsonl"),  mid_pairs)
    write_jsonl(os.path.join(args.out_dir, "high_pairs.jsonl"), high_pairs)

    meta = {
        "corpus": args.corpus,
        "tokenizer": args.tokenizer,
        "max_lines": args.max_lines,
        "min_tokens": args.min_tokens,
        "prefix_tokens": args.prefix_tokens,
        "suffix_tokens": args.suffix_tokens,
        "samples_per_bucket": args.samples_per_bucket,
        "quantiles": {"q1": float(q1), "q2": float(q2)},
        "counts": {"rare": len(rare_pairs), "mid": len(mid_pairs), "high": len(high_pairs)},
        "note": "avg_log_token_freq quantile bucketing; deterministic seed=42"
    }
    with open(os.path.join(args.out_dir, "bucket_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[OK] wrote bucket jsonl + bucket_meta.json")


if __name__ == "__main__":
    main()
