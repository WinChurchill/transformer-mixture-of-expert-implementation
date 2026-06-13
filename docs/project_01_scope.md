# Project 01 Scope: Transformer + Inference Foundations

This additive track keeps the existing Transformer + MoE assignment intact while
making the repository useful as the first project in the summer frontier-systems
portfolio. The goal is to understand the model path deeply enough to reason
about LLM inference systems later.

## Goal

Build and explain a small decoder-only language model with special attention to
inference behavior:

- token IDs to embeddings to logits;
- causal multi-head attention and its shape contracts;
- modern decoder-block components such as RoPE, RMSNorm, and SwiGLU;
- autoregressive sampling choices;
- prefill vs decode;
- KV-cache correctness and memory accounting;
- benchmark vocabulary: latency, tokens/sec, memory, and speedup.

## Current Repo Relationship

The existing project already has:

- manual dense Transformer learning code;
- trainable PyTorch dense and MoE models;
- TinyStories training scripts;
- generation from checkpoints;
- shape, gradient, training, checkpoint, and notebook-asset tests.

Project 01 is additive. It does not remove MoE. Instead, MoE becomes the optional
feed-forward extension after the base inference path is understood.

## In Scope For This Pass

- Add artifact directories: `configs/`, `docs/`, `reports/`, and `results/`.
- Add this scope document and a day-by-day checklist.
- Expand the base notebook with mini-lectures for Project 01 concepts.
- Add a tiny inference-oriented config seed.
- Add a report template for later benchmark results.

## Out Of Scope For This Pass

- Implementing RoPE, RMSNorm, SwiGLU, top-p sampling, KV cache, or benchmarks.
- Renaming the package or moving code into an umbrella portfolio repository.
- Removing the existing MoE assignment path.

## Definition Of Done

- The README names the Project 01 goal and explains how it relates to MoE.
- The notebook contains substantial mini-lecture content for inference concepts.
- The new artifact folders exist and have useful starter files.
- Existing tests still pass.
