# Memory Dial

This repository provides the  public code required to reproduce the core claims of the paper:

> Memory Dial: A Training Framework for Controllable Memorization in Language Models  
> Anonymous ACL submission


---

## What This Repository Supports

This codebase is sufficient to verify that:

- The memorization coefficient alpha provides monotonic control over memorization strength.
- Models trained with different alpha values differ only in memorization pressure.
- Seen-example accuracy increases with alpha, while unseen accuracy remains stable.

All experiments are intended to run at small scale using publicly available models.

---

##  Repository Structure

MemoryDial/
- README.md
- requirements.txt
- losses/
  - memory_dial_loss.py
- data/
  - build_seen_unseen.py
- experiments/
  - train_memory_dial.py
- evaluation/
  - seen_unseen_accuracy.py

---

## Main Program

The only required training entry point is:

experiments/train_memory_dial.py

This script:

- trains a language model with the Memory Dial objective
- takes alpha as the only varying training parameter
- fixes temperature tau = 0.1
- evaluates performance on seen and unseen examples

---

## Running a Minimal Experiment

Example command:

```bash
python experiments/train_memory_dial.py \
  --model gpt2 \
  --alpha 0.5 \
  --tau 0.1 \
  --seed 42
```

To reproduce the main trend, run the same command for multiple alpha values
(e.g., 0.0, 0.5, 1.0) using the same data split and seed.

---

## Memory Dial Objective

The loss implemented in:

losses/memory_dial_loss.py

corresponds to Equation (5) in the paper and is defined as:


```text
L_MD = (1 - alpha) * L_std + alpha * L_mem
```

All experiments fix tau = 0.1 to isolate the effect of alpha.

---

## Data Construction

The script:

data/build_seen_unseen.py

constructs:

- a seen subset (explicitly injected during training)
- an unseen subset (held out from training)

The same split is reused across all alpha values.

---

## Excluded Components

The following are intentionally not included:

- Full pretraining corpora
- Large-scale model training (e.g., OPT-13B, OPT-27B)
- Distributed or cluster-specific infrastructure
- Raw checkpoints or logs

This is intentional and consistent with ACL reproducibility norms.

---

## Ethical Note

Increasing alpha increases memorization pressure.

High-alpha settings should not be used with sensitive or private data.
This framework is intended for controlled scientific analysis.

---

