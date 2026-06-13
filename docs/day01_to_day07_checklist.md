# Day 01 To Day 07 Checklist

Use this as the operating checklist for Project 01. Each day should produce one
small artifact: a passing test, a working script, a measurement, or a report
section.

## Day 1: Repo, Data, Shape Contract

- Read: transformer embeddings and decoder framing.
- Mini-lecture: token IDs `(B, T)` to logits `(B, T, vocab_size)`.
- Implement later: config, tokenizer/data path, embeddings, LM head.
- Verify: model initialization and forward shape tests.
- Notes: parameter count for embeddings and LM head.

## Day 2: Causal Self-Attention

- Read: scaled dot-product attention and multi-head attention.
- Mini-lecture: Q/K/V shapes, attention matrix shape, causal masking.
- Implement later: manual attention path and optional SDPA comparison.
- Verify: output shape, triangular mask, no future-token leakage.
- Notes: where memory scales as `O(T^2)`.

## Day 3: RoPE, RMSNorm, SwiGLU, Transformer Block

- Read: LLaMA-style block choices and RoPE intuition.
- Mini-lecture: what gets rotated, what RMSNorm removes, why SwiGLU gates.
- Implement later: RoPE cache, RMSNorm, SwiGLU MLP, full block wiring.
- Verify: shape tests and no NaNs in forward/backward.
- Notes: RoPE shape conventions and RMSNorm vs LayerNorm.

## Day 4: Tiny Training Loop

- Read: batching, AdamW, eval loops, checkpointing.
- Mini-lecture: shifted targets and cross entropy.
- Implement later: tiny dataset loader, train/eval loop, checkpoint resume.
- Verify: tiny overfit run where loss clearly decreases.
- Notes: first loss curve and instability causes.

## Day 5: Sampling And Generation

- Read: generation controls for temperature, top-k, and top-p.
- Mini-lecture: greedy decoding, temperature, top-k, top-p, and failure modes.
- Implement later: top-p and deterministic generation tests.
- Verify: same seed gives same sample; filtered tokens are never sampled.
- Notes: qualitative effect of sampling settings.

## Day 6: KV Cache

- Read: KV-cache guide and prefill/decode notes.
- Mini-lecture: reusable tensors, per-layer cache contents, memory formula.
- Implement later: `past_key_values`, prefill, decode, cached generation.
- Verify: cached final-token logits match uncached logits within tolerance.
- Notes: concrete KV-cache memory example.

## Day 7: Benchmark And Report

- Read: PyTorch profiler basics and FlashAttention motivation.
- Mini-lecture: memory movement, prefill latency, decode tokens/sec, speedup.
- Implement later: `scripts/benchmark_inference.py` and result logging.
- Verify: benchmark emits JSON/CSV and report has at least one table.
- Notes: bottleneck explanation and next project links.
