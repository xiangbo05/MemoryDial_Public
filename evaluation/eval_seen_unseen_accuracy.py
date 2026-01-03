#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_seen_unseen_accuracy.py
Reproduces seen/unseen accuracy evaluation used in Memory Dial paper (Fig 4, Table 10).
Method: multiple-choice / yes-no scored by conditional NLL of each candidate answer (lower is better).
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


# -------------------------
# Dataset adapters
# -------------------------
@dataclass
class Example:
    uid: str
    prompt: str
    choices: List[str]
    gold_index: int


def _normalize_arc_answer_key(k: Any) -> str:
    # ARC answerKey is usually "A"/"B"/"C"/"D"/"E"
    if isinstance(k, str):
        return k.strip()
    return str(k).strip()


def build_examples(dataset_name: str, split: str, max_examples: Optional[int] = None) -> List[Example]:
    """
    Supported dataset_name values:
      - arc_easy
      - boolq
      - piqa
      - copa
      - openbookqa
    Uses HF datasets canonical names.
    """
    ds = None
    name = dataset_name.lower()

    if name == "arc_easy":
        ds = load_dataset("ai2_arc", "ARC-Easy", split=split)
        examples: List[Example] = []
        for i, ex in enumerate(ds):
            # ex: question, choices(text/label), answerKey
            q = ex["question"]
            choice_texts = ex["choices"]["text"]
            choice_labels = ex["choices"]["label"]  # ["A","B",...]
            ans_key = _normalize_arc_answer_key(ex["answerKey"])
            gold_idx = choice_labels.index(ans_key) if ans_key in choice_labels else 0
            prompt = f"Question: {q}\nAnswer:"
            choices = [f" {t}" for t in choice_texts]
            uid = ex.get("id", str(i))
            examples.append(Example(uid=uid, prompt=prompt, choices=choices, gold_index=gold_idx))

    elif name == "openbookqa":
        ds = load_dataset("openbookqa", "main", split=split)
        examples = []
        for i, ex in enumerate(ds):
            stem = ex["question_stem"]
            choice_texts = ex["choices"]["text"]
            choice_labels = ex["choices"]["label"]
            ans_key = _normalize_arc_answer_key(ex["answerKey"])
            gold_idx = choice_labels.index(ans_key) if ans_key in choice_labels else 0
            prompt = f"Question: {stem}\nAnswer:"
            choices = [f" {t}" for t in choice_texts]
            uid = ex.get("id", str(i))
            examples.append(Example(uid=uid, prompt=prompt, choices=choices, gold_index=gold_idx))

    elif name == "piqa":
        ds = load_dataset("piqa", split=split)
        examples = []
        for i, ex in enumerate(ds):
            # goal, sol1, sol2, label in {0,1}
            goal = ex["goal"]
            sol1 = ex["sol1"]
            sol2 = ex["sol2"]
            label = int(ex["label"])
            prompt = f"Goal: {goal}\nSolution:"
            choices = [f" {sol1}", f" {sol2}"]
            uid = ex.get("id", str(i))
            examples.append(Example(uid=uid, prompt=prompt, choices=choices, gold_index=label))

    elif name == "copa":
        ds = load_dataset("super_glue", "copa", split=split)
        examples = []
        for i, ex in enumerate(ds):
            # premise, choice1, choice2, question ("cause"/"effect"), label
            premise = ex["premise"]
            qtype = ex["question"]  # "cause" or "effect"
            c1 = ex["choice1"]
            c2 = ex["choice2"]
            label = int(ex["label"])
            if qtype == "cause":
                prompt = f"Premise: {premise}\nWhat was the cause?"
            else:
                prompt = f"Premise: {premise}\nWhat happened as a result?"
            prompt += "\nAnswer:"
            choices = [f" {c1}", f" {c2}"]
            uid = ex.get("idx", str(i))
            examples.append(Example(uid=str(uid), prompt=prompt, choices=choices, gold_index=label))

    elif name == "boolq":
        ds = load_dataset("super_glue", "boolq", split=split)
        examples = []
        for i, ex in enumerate(ds):
            # passage, question, label (0/1)
            passage = ex["passage"]
            question = ex["question"]
            label = int(ex["label"])
            prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer:"
            choices = [" no", " yes"]  # map 0->no, 1->yes (HF boolq label: 0=False,1=True)
            gold_idx = label
            uid = ex.get("idx", str(i))
            examples.append(Example(uid=str(uid), prompt=prompt, choices=choices, gold_index=gold_idx))

    else:
        raise ValueError(f"Unsupported dataset_name={dataset_name}")

    if max_examples is not None:
        examples = examples[: max_examples]
    return examples


# -------------------------
# Scoring (conditional NLL)
# -------------------------
@torch.no_grad()
def option_nll(model, tokenizer, prompt: str, option: str, device: torch.device) -> float:
    """
    Compute NLL of option tokens conditioned on prompt.
    We compute loss only on the option continuation tokens.
    """
    # Tokenize prompt and full
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    full_ids = tokenizer(prompt + option, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    # Labels: ignore prompt positions (-100), keep option positions
    labels = full_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100

    out = model(input_ids=full_ids, labels=labels)
    # loss is mean over non-ignored tokens
    loss = float(out.loss.detach().cpu().item())
    # Convert mean loss to total NLL *token_count* if you want; for argmin mean is enough
    return loss


@torch.no_grad()
def predict(model, tokenizer, ex: Example, device: torch.device) -> int:
    scores = [option_nll(model, tokenizer, ex.prompt, c, device) for c in ex.choices]
    return int(np.argmin(scores))


def load_indices(path: str) -> List[int]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "indices" in obj:
        return list(map(int, obj["indices"]))
    if isinstance(obj, list):
        return list(map(int, obj))
    raise ValueError("indices file must be a list[int] or {'indices': list[int]}")


def compute_accuracy(examples: List[Example], model, tokenizer, device: torch.device, batch: int = 1) -> float:
    # Simple loop (option scoring is expensive; batching is non-trivial w/ varying lengths)
    correct = 0
    for ex in tqdm(examples, desc="eval", ncols=100):
        pred = predict(model, tokenizer, ex, device)
        correct += int(pred == ex.gold_index)
    return correct / max(1, len(examples))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="HF path or local checkpoint dir (e.g., ./ckpt_alpha_0.4)")
    ap.add_argument("--dataset", type=str, required=True,
                    choices=["arc_easy", "boolq", "piqa", "copa", "openbookqa"])
    ap.add_argument("--split", type=str, default="validation",
                    help="HF split name (often validation). For some datasets use 'validation' or 'test'.")
    ap.add_argument("--seen_indices", type=str, required=True, help="JSON file of seen example indices")
    ap.add_argument("--unseen_indices", type=str, required=True, help="JSON file of unseen example indices")
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--out", type=str, required=True, help="Output JSON path")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    model.eval()

    all_examples = build_examples(args.dataset, args.split, max_examples=args.max_examples)

    seen_idx = load_indices(args.seen_indices)
    unseen_idx = load_indices(args.unseen_indices)

    # safety
    seen_idx = [i for i in seen_idx if 0 <= i < len(all_examples)]
    unseen_idx = [i for i in unseen_idx if 0 <= i < len(all_examples)]

    seen_examples = [all_examples[i] for i in seen_idx]
    unseen_examples = [all_examples[i] for i in unseen_idx]

    seen_acc = compute_accuracy(seen_examples, model, tokenizer, device)
    unseen_acc = compute_accuracy(unseen_examples, model, tokenizer, device)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "dataset": args.dataset,
            "split": args.split,
            "seen_n": len(seen_examples),
            "unseen_n": len(unseen_examples),
            "seen_accuracy": seen_acc,
            "unseen_accuracy": unseen_acc
        }, f, indent=2)

    print(f"[OK] wrote {args.out}")
    print(f"seen_acc={seen_acc:.4f} | unseen_acc={unseen_acc:.4f}")


if __name__ == "__main__":
    main()

