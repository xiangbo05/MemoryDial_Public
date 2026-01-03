#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_qualitative_examples.py
Generates qualitative completions for a fixed set of prompts across models (Table 3 / Fig 1).
Optionally computes suffix NLL if 'target' is provided per prompt.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import torch
import numpy as np
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM


@torch.no_grad()
def suffix_nll(model, tok, prefix: str, suffix: str, device: torch.device) -> float:
    prefix_ids = tok(prefix, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    full_ids = tok(prefix + suffix, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    labels = full_ids.clone()
    labels[:, : prefix_ids.shape[1]] = -100
    out = model(input_ids=full_ids, labels=labels)
    return float(out.loss.detach().cpu().item())


@torch.no_grad()
def generate(model, tok, prompt: str, device: torch.device, max_new_tokens: int) -> str:
    inp = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    out = model.generate(
        **inp,
        max_new_tokens=max_new_tokens,
        do_sample=False,           # deterministic (greedy)
        temperature=1.0,
        top_p=1.0,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    # return only the continuation part (best-effort)
    if text.startswith(prompt):
        return text[len(prompt):]
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, nargs="+", required=True,
                    help="List of model paths (e.g., ckpt_alpha_0.0 ckpt_alpha_0.5 ckpt_alpha_1.0)")
    ap.add_argument("--prompts_yaml", type=str, required=True,
                    help="YAML file with list of {name,type,prompt,target(optional)}")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    with open(args.prompts_yaml, "r", encoding="utf-8") as f:
        prompt_items: List[Dict[str, Any]] = yaml.safe_load(f)

    results = {"prompts": prompt_items, "runs": []}

    for mp in args.models:
        tok = AutoTokenizer.from_pretrained(mp, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            mp,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
        )
        model.eval()

        run = {"model": mp, "items": []}
        for item in prompt_items:
            prompt = item["prompt"]
            cont = generate(model, tok, prompt, device, args.max_new_tokens)
            rec = {
                "name": item.get("name", ""),
                "type": item.get("type", ""),
                "prompt": prompt,
                "continuation": cont.strip()
            }
            if "target" in item and item["target"] is not None:
                rec["target"] = item["target"]
                rec["suffix_nll"] = suffix_nll(model, tok, prompt, item["target"], device)
            run["items"].append(rec)

        results["runs"].append(run)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[OK] wrote {args.out}")


if __name__ == "__main__":
    main()
