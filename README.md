# Swappable Transformer + MoE Lab

This repo is a coding-practice assignment for implementing a small decoder-only
transformer and then swapping the dense feed-forward network for a Mixture of
Experts (MoE) feed-forward block.

## Project 01 Additive Track: Transformer + Inference Foundations

This repository now also serves as **Project 01** in the summer frontier-systems
plan: build the model-side foundations needed to understand LLM inference. The
existing dense Transformer and MoE assignment remains intact, but the broader
scope now includes decoder-only language-model tensor shapes, causal attention,
modern block choices, autoregressive sampling, KV-cache inference, and
prefill/decode benchmarking.

The Project 01 learning target is practical inference fluency:

1. explain the path from token IDs `(B, T)` to logits `(B, T, vocab_size)`;
2. derive causal multi-head attention with `(B, H, T, Dh)` tensors;
3. understand where RoPE, RMSNorm, and SwiGLU fit in modern decoder blocks;
4. compare greedy, temperature, top-k, and top-p sampling;
5. explain prefill vs decode and the KV-cache memory formula;
6. measure latency, tokens/sec, memory, and cached-vs-uncached speedup.

MoE is now framed as an optional feed-forward extension after the inference
foundation is understood. It still uses the same residual block, attention path,
language-model objective, generation loop, and checkpointing path.

Project 01 scaffold files:

```text
configs/tiny_inference.json                 Tiny inference-oriented config seed
docs/project_01_scope.md                    Goal, scope, and acceptance criteria
docs/day01_to_day07_checklist.md            Seven-day execution checklist
reports/transformer_inference_report.md     Report template for benchmark results
results/                                    Raw benchmark outputs and logs
```

The main assignment document is:

```text
notebooks/assignment_transformer_moe.ipynb
```

Daily reading-summary notebooks are in:

```text
notebooks/readings/00_reading_index.ipynb
```

Use those notebooks before each day of implementation. They summarize the
assigned paper/docs sections, connect the reading to repo files, and include
active-recall prompts.

Start there for the pipeline lectures, theory, tensor-shape tables,
implementation instructions, MoE pseudocode, debugging checklists, and
per-section test commands. This README is only the quickstart and project index.

The learning path is deliberately math-first:

1. implement dense transformer components with handwritten backward passes;
2. verify each part with finite-difference and PyTorch oracle tests;
3. replace only the feed-forward sublayer with MoE;
4. train a small dense or MoE language model on TinyStories.

The notation in the notebook follows a CS189-style convention: tokens are rows
of `X in R^{N x D}`, batched tensors are `X in R^{B x N x D}`, and attention is
introduced as a differentiable soft lookup table.

## Project Layout

```text
notebooks/assignment_transformer_moe.ipynb   Guided assignment notebook
notebooks/readings/                          Dated reading-summary notebooks
src/moe_transformer_lab/manual/              Handwritten forward/backward code
src/moe_transformer_lab/trainable/           Practical PyTorch training model
scripts/download_tinystories.py              TinyStories download helper
scripts/train_tokenizer.py                   Byte/BPE tokenizer helper
scripts/train_dense.py                       Dense transformer training wrapper
scripts/train_moe.py                         Dense or MoE training script
scripts/generate.py                          Checkpoint text generation
configs/                                    Project 01 config seeds
docs/                                       Project 01 scope and daily checklist
reports/                                    Technical report templates
results/                                    Raw benchmark/result artifacts
tests/                                       Pytest checks for each subsystem
```

## Setup

Use Python 3.12 if possible. PyTorch support on brand-new Python versions can lag
behind, especially on Windows.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

Install PyTorch using the official selector:

<https://pytorch.org/get-started/locally/>

Then install the project:

```powershell
python -m pip install -e .[dev,train]
```

If `torch` is not installed, the tests that require it will skip cleanly.

## TinyStories

TinyStories is hosted on Hugging Face at:

<https://huggingface.co/datasets/roneneldan/TinyStories>

Download the V2 GPT-4 train/validation text files:

```powershell
python scripts/download_tinystories.py --out-dir data/tinystories/raw
python scripts/train_tokenizer.py --raw-dir data/tinystories/raw --out data/tinystories/tokenizer.json
```

Start with the dense model:

```powershell
python scripts/train_dense.py --steps 1000 --block-size 128 --batch-size 16
```

Then train the MoE model through the same training path:

```powershell
python scripts/train_moe.py --ffn moe --steps 1000 --block-size 128 --batch-size 16 --n-experts 4 --top-k 1
```

Generate from a checkpoint:

```powershell
python scripts/generate.py --checkpoint runs/tinystories_moe/ckpt_last.pt --prompt "Once upon a time"
```

## Design Rule

The transformer block owns attention, normalization, and residual connections.
The feed-forward block is swappable:

```python
ffn_out, aux_loss = feed_forward(norm2(x))
x = x + ffn_out
```

`DenseMLP` returns `aux_loss = 0`. `MoEFeedForward` returns the router
load-balancing loss.

For the detailed MoE implementation recipe, including router tensor shapes,
top-k dispatch/combine pseudocode, and common failure modes, use the notebook
section titled "MoE Tensor Flow" and the sections that follow it.

## Figures

The notebook uses generated classroom-style SVG diagrams for both the original
Transformer/MoE assignment and the Project 01 inference-foundations mini
lectures. Regenerate them with:

```powershell
python scripts/generate_figures.py
```
