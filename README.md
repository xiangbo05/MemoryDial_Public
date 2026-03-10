
# Memory Dial: A Training Framework for Controllable Memorization in Language Models

This repository contains code for the paper:

**Memory Dial: A Training Framework for Controllable Memorization in Language Models**

## Overview

Memorization is an important but poorly understood property of language models.  
Most prior work studies memorization *after* training.  

**Memory Dial** introduces a simple training framework that makes memorization **explicitly controllable** during training.

The key idea is to interpolate between standard cross-entropy and a temperature-sharpened objective using a single parameter **α**.

```

L_MD = (1 - α) L_std + α L_mem

````

By varying α, we obtain a **family of models** that differ only in memorization pressure.

This allows controlled experiments studying how memorization interacts with generalization.

---

## Experiments

We evaluate Memory Dial across multiple model sizes:

- DistilGPT2
- GPT-2 Small
- TinyLLaMA-1B
- OPT-250M
- OPT-13B
- OPT-27B

Benchmarks used:

- ARC-Easy
- BoolQ
- PIQA
- COPA
- OpenBookQA

Each benchmark is split into:

- **Seen examples** (injected into training)
- **Unseen examples** (held out)

This setup allows us to measure memorization and generalization separately.

---

## Main Findings

- Increasing **α** increases memorization of seen examples.
- Performance on unseen examples remains largely stable.
- Larger models are more responsive to memorization pressure.

---

## Running Training

Example training command:

```bash
python train.py --model gpt2 --alpha 0.4 --tau 0.1
````

Default α sweep:

```
α ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
```

---

---

## License

MIT License



---

