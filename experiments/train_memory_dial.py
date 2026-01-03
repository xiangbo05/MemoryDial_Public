#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_memory_dial.py
--------------------
Minimal, ACL-reproducible training script for Memory Dial.

Purpose:
- Demonstrate α-controlled memorization with a small model
- Reproduce seen ↑ / unseen ≈ stable behavior
- Avoid large-scale infrastructure and private data

This script is intentionally minimal.
"""

import argparse
import json
import os
import random
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AdamW,
)

# -------------------------
# Memory Dial loss
# -------------------------

def memory_dial_loss(logits, labels, alpha: float, tau: float = 0.1):
    """
    Implements Eq. (5) in the paper:
      L = (1-α) * CE + α * CE_tau
    """
    vocab = logits.size(-1)

    logits_flat = logits.view(-1, vocab)
    labels_flat = labels.view(-1)

    # Standard CE
    loss_std = F.cross_entropy(logits_flat, labels_flat)

    # Temperature-sharpened CE
    logits_tau = logits / tau
    loss_mem = F.cross_entropy(logits_tau.view(-1, vocab), labels_flat)

    return (1 - alpha) * loss_std + alpha * loss_mem


# -------------------------
# Dataset helpers
# -------------------------

def load_seen_unseen_ids(path: str):
    with open(path, "r") as f:
        return set(json.load(f)["seen_ids"])


def stable_id(example: dict) -> str:
    import hashlib, json
    s = json.dumps(example, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(s.encode()).hexdigest()


def filter_split(dataset, id_set, keep: bool):
    """
    keep=True  -> seen
    keep=False -> unseen
    """
    out = []
    for ex in dataset:
        ex_id = stable_id(ex)
        if (ex_id in id_set) == keep:
            out.append(ex)
    return out


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--benchmark", default="boolq")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--seen_ids", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"[INFO] α = {args.alpha}")

    # Load dataset
    dataset = load_dataset("boolq", split="validation")

    seen_ids = load_seen_unseen_ids(args.seen_ids)
    seen_data = filter_split(dataset, seen_ids, keep=True)
    unseen_data = filter_split(dataset, seen_ids, keep=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    def encode(ex):
        text = ex["question"] + " " + ex["passage"]
        enc = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        enc["labels"] = enc["input_ids"].clone()
        return enc

    def collate(batch):
        return {
            k: torch.cat([b[k] for b in batch], dim=0)
            for k in batch[0]
        }

    train_loader = DataLoader(
        [encode(ex) for ex in seen_data],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )

    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # -------------------------
    # Training
    # -------------------------

    model.train()
    for _ in range(args.epochs):
        for batch in train_loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            out = model(**batch)
            loss = memory_dial_loss(
                out.logits, batch["labels"], alpha=args.alpha
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # -------------------------
    # Evaluation
    # -------------------------

    def eval_nll(data):
        model.eval()
        total = 0.0
        with torch.no_grad():
            for ex in data[:50]:
                enc = encode(ex)
                enc = {k: v.to(args.device) for k, v in enc.items()}
                out = model(**enc)
                total += F.cross_entropy(
                    out.logits.view(-1, out.logits.size(-1)),
                    enc["labels"].view(-1),
                ).item()
        return total / min(50, len(data))

    seen_nll = eval_nll(seen_data)
    unseen_nll = eval_nll(unseen_data)

    print(f"[RESULT] α={args.alpha}")
    print(f"  Seen   NLL: {seen_nll:.3f}")
    print(f"  Unseen NLL: {unseen_nll:.3f}")


if __name__ == "__main__":
    main()

