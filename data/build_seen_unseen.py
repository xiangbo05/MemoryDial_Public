#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_seen_unseen.py
--------------------
Construct Seen/Unseen evaluation splits for Memory Dial.

Design goals (ACL-friendly):
- Do NOT redistribute benchmark datasets.
- Load public benchmarks via HuggingFace `datasets`.
- Produce stable IDs via hashing example content (not dataset index).
- Deterministic selection via a seed.

Outputs per benchmark:
  output_dir/<benchmark>/
    - seen_ids.json
    - unseen_ids.json
    - meta.json

Example:
  python data/build_seen_unseen.py --benchmark boolq --num_seen 50 --seed 42
"""

import argparse
import hashlib
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# Benchmark registry
# -----------------------------

@dataclass(frozen=True)
class HFSpec:
    hf_dataset: str
    hf_config: Optional[str]
    split: str

# Map your paper's benchmark names to HF datasets/configs.
# Note: Some datasets have multiple splits/configs; we pick standard evaluation splits.
BENCHMARKS: Dict[str, HFSpec] = {
    "boolq": HFSpec("boolq", None, "validation"),
    "piqa": HFSpec("piqa", None, "validation"),
    "copa": HFSpec("super_glue", "copa", "validation"),
    "obqa": HFSpec("openbookqa", "main", "test"),
    "arc_easy": HFSpec("ai2_arc", "ARC-Easy", "test"),
}

def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _normalize_value(v: Any) -> Any:
    """
    Normalize dataset values to stable JSON-serializable forms.
    """
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_normalize_value(x) for x in v]
    if isinstance(v, dict):
        # sort keys for determinism
        return {k: _normalize_value(v[k]) for k in sorted(v.keys())}
    # fallback: stringify
    return str(v)

def stable_example_id(example: Dict[str, Any], keys: Optional[List[str]] = None) -> str:
    """
    Create a stable ID for an example using a hash of selected fields.

    If keys is None, uses all keys (sorted) -> generally stable across runs.
    For maximum stability across dataset versions, prefer keys that define the task input/label.
    """
    if keys is None:
        payload = {k: _normalize_value(example[k]) for k in sorted(example.keys())}
    else:
        payload = {k: _normalize_value(example.get(k)) for k in keys}

    # Use canonical JSON string
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha1_hex(s)

def default_id_keys(benchmark: str) -> Optional[List[str]]:
    """
    Choose a small subset of fields that define the instance.
    This makes IDs robust to harmless metadata changes.
    """
    if benchmark == "boolq":
        return ["question", "passage", "answer"]
    if benchmark == "piqa":
        return ["goal", "sol1", "sol2", "label"]
    if benchmark == "copa":
        return ["premise", "choice1", "choice2", "question", "label"]
    if benchmark == "obqa":
        # openbookqa fields vary; try common ones
        return ["question_stem", "choices", "answerKey"]
    if benchmark == "arc_easy":
        # ai2_arc uses similar schema
        return ["question", "choices", "answerKey"]
    return None

def load_hf_dataset(spec: HFSpec):
    """
    Lazy import datasets so repo doesn't hard-depend unless user runs script.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: datasets. Install with:\n"
            "  pip install datasets\n"
            "Optionally: pip install 'datasets[audio,vision]' (not needed here)\n"
        ) from e

    if spec.hf_config is None:
        ds = load_dataset(spec.hf_dataset, split=spec.split)
    else:
        ds = load_dataset(spec.hf_dataset, spec.hf_config, split=spec.split)
    return ds

def make_split_ids(
    benchmark: str,
    num_seen: int,
    seed: int,
    enforce_unique_ids: bool = True,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Returns (seen_ids, unseen_ids, meta).
    """
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark '{benchmark}'. Options: {sorted(BENCHMARKS.keys())}")

    spec = BENCHMARKS[benchmark]
    ds = load_hf_dataset(spec)

    id_keys = default_id_keys(benchmark)

    ids: List[str] = []
    for ex in ds:
        ex_id = stable_example_id(ex, keys=id_keys)
        ids.append(ex_id)

    # Optionally ensure uniqueness (rare duplicates could exist)
    if enforce_unique_ids:
        uniq = list(dict.fromkeys(ids))  # stable unique in insertion order
        if len(uniq) != len(ids):
            # We keep first occurrences; unseen/seen are defined over unique IDs.
            ids = uniq

    n = len(ids)
    if num_seen <= 0 or num_seen >= n:
        raise ValueError(f"num_seen must be in (0, {n}), got {num_seen}")

    rng = random.Random(seed)
    all_idx = list(range(n))
    rng.shuffle(all_idx)

    seen_idx = sorted(all_idx[:num_seen])
    unseen_idx = sorted(all_idx[num_seen:])

    seen_ids = [ids[i] for i in seen_idx]
    unseen_ids = [ids[i] for i in unseen_idx]

    meta = {
        "benchmark": benchmark,
        "hf_dataset": spec.hf_dataset,
        "hf_config": spec.hf_config,
        "split": spec.split,
        "num_total": n,
        "num_seen": len(seen_ids),
        "num_unseen": len(unseen_ids),
        "seed": seed,
        "id_keys": id_keys,
        "id_hash": "sha1(canonical_json(selected_fields))",
        "note": "IDs are content-hashes; datasets are not redistributed.",
    }
    return seen_ids, unseen_ids, meta

def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=sorted(BENCHMARKS.keys()),
        help="Which benchmark to split (boolq, piqa, copa, obqa, arc_easy).",
    )
    parser.add_argument("--num_seen", type=int, default=50, help="Number of seen examples to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/splits",
        help="Base output directory (per-benchmark subfolder will be created).",
    )
    parser.add_argument(
        "--enforce_unique_ids",
        action="store_true",
        help="Deduplicate IDs (recommended).",
    )
    args = parser.parse_args()

    seen_ids, unseen_ids, meta = make_split_ids(
        benchmark=args.benchmark,
        num_seen=args.num_seen,
        seed=args.seed,
        enforce_unique_ids=args.enforce_unique_ids,
    )

    out_dir = os.path.join(args.output_dir, args.benchmark)
    _safe_mkdir(out_dir)

    write_json(os.path.join(out_dir, "seen_ids.json"), {"seen_ids": seen_ids})
    write_json(os.path.join(out_dir, "unseen_ids.json"), {"unseen_ids": unseen_ids})
    write_json(os.path.join(out_dir, "meta.json"), meta)

    print(f"[OK] Wrote splits to: {out_dir}")
    print(f"  seen:   {len(seen_ids)}")
    print(f"  unseen: {len(unseen_ids)}")
    print(f"  seed:   {meta['seed']}")

if __name__ == "__main__":
    main()

