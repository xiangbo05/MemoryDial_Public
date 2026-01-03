#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compute_seen_accuracy_slope.py
Computes slope of seen accuracy vs alpha (Table 1 / Fig 3).
"""

import argparse
import json
import glob
import os
from typing import Dict, List, Tuple

import numpy as np


def load_alpha_acc_from_file(path: str) -> Tuple[float, float]:
    """
    Expected JSON formats:
      A) {"alpha": 0.4, "seen_accuracy": 0.52, ...}
      B) {"alpha": "0.4", "seen_accuracy": 0.52, ...}
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    a = float(obj["alpha"]) if "alpha" in obj else float(obj["memorization_coefficient"])
    seen = float(obj["seen_accuracy"])
    return a, seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=str, required=True,
                    help="Either a glob like 'results/*_acc.json' OR a directory containing JSON files.")
    ap.add_argument("--out", type=str, required=True, help="Output JSON path")
    args = ap.parse_args()

    paths: List[str] = []
    if os.path.isdir(args.inputs):
        paths = sorted(glob.glob(os.path.join(args.inputs, "*.json")))
    else:
        paths = sorted(glob.glob(args.inputs))

    if not paths:
        raise SystemExit(f"No JSON files found for inputs={args.inputs}")

    pairs = [load_alpha_acc_from_file(p) for p in paths]
    pairs = sorted(pairs, key=lambda x: x[0])

    alphas = np.array([a for a, _ in pairs], dtype=np.float64)
    seen = np.array([s for _, s in pairs], dtype=np.float64)

    # slope of linear fit seen = m*alpha + b
    m, b = np.polyfit(alphas, seen, deg=1)

    out_obj = {
        "n": int(len(alphas)),
        "alphas": [float(x) for x in alphas],
        "seen_accuracies": [float(x) for x in seen],
        "slope": float(m),
        "intercept": float(b),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2)

    print(f"[OK] slope={m:.6f} | wrote {args.out}")


if __name__ == "__main__":
    main()
