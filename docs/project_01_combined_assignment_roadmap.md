# Combined Roadmap: Transformer, MoE, and Inference Foundations

This is the detailed daily assignment roadmap for the combined Project 01 track.
It starts on **2026-06-13** and assumes work happens every day through
**2026-06-19**.

The roadmap combines three threads:

1. the existing dense Transformer and handwritten-backward assignment;
2. the existing trainable dense/MoE PyTorch path;
3. the added Project 01 inference-foundations path: RoPE, RMSNorm, SwiGLU,
   sampling, KV cache, prefill/decode, and benchmarking concepts.

This document is an operating assignment list. It does not mean every future
feature is already implemented. Items marked as planned implementation are the
next code milestones after the documentation pass.

## Operating Assumptions

- Dates: every day from **2026-06-13** through **2026-06-19**.
- Daily budget: 4-5 focused hours.
- Main learning file: `notebooks/assignment_transformer_moe.ipynb`.
- Main implementation package: `src/moe_transformer_lab/`.
- Raw notes can live in this file, the notebook, or the report template, but
  final benchmark interpretation should end in
  `reports/transformer_inference_report.md`.
- The current repo already contains dense Transformer and MoE code. Do not
  remove that path. Treat MoE as the optional feed-forward extension after the
  base decoder-only inference path is understood.
- Planned-but-not-yet-implemented Project 01 features include RoPE, RMSNorm,
  SwiGLU, top-p sampling, KV cache, and benchmark scripts.

## Daily Workflow Template

Use this structure each day.

| Block | Time | Work |
| --- | ---: | --- |
| Reading | 30-45 min | Read only the assigned paper/docs sections. Do not over-read. |
| Mini-lecture note | 20-30 min | Answer the prompt in your own words with tensor shapes. |
| Repo inspection | 20-30 min | Open the named files and map concepts to code. |
| Implementation or design work | 120-150 min | Implement scheduled code, or define the exact planned interface if this pass is docs-only. |
| Verification | 45-60 min | Run listed tests or record why a test is skipped. |
| Written artifact | 20-30 min | Update notes, checklist, report section, or TODO list. |

Daily rule: each day must produce one concrete artifact: a passing test, a
shape table, a verified command, a benchmark log, a report section, or a clear
implementation checklist.

## Repo Materials Map

Use these files deliberately.

| Material | Path | Purpose |
| --- | --- | --- |
| Main learning notebook | `notebooks/assignment_transformer_moe.ipynb` | Combined lectures, tensor tables, assignment instructions, MoE extension material. |
| Project scope | `docs/project_01_scope.md` | Current goal and boundaries for the additive Project 01 track. |
| Short checklist | `docs/day01_to_day07_checklist.md` | Compact daily summary; this roadmap is the detailed version. |
| Inference config seed | `configs/tiny_inference.json` | Tiny config skeleton for later benchmark and generation work. |
| Report template | `reports/transformer_inference_report.md` | Final writeup destination for results and bottleneck analysis. |
| Manual layer code | `src/moe_transformer_lab/manual/layers.py` | Handwritten Linear, Embedding, LayerNorm, loss, and backward-pass practice. |
| Manual attention code | `src/moe_transformer_lab/manual/attention.py` | Causal mask, row softmax, scaled dot-product attention, multi-head attention. |
| Manual transformer code | `src/moe_transformer_lab/manual/transformer.py` | Dense manual decoder-only model path. |
| Trainable model code | `src/moe_transformer_lab/trainable/model.py` | Practical PyTorch dense/MoE model, attention, block, generation. |
| Config dataclass | `src/moe_transformer_lab/config.py` | Shared model settings and validation rules. |
| Data utilities | `src/moe_transformer_lab/trainable/data.py` | Tokenized data path and batch sampling. |
| Tokenizers | `src/moe_transformer_lab/trainable/tokenizer.py` | Byte or trained tokenizer loading. |
| Training utilities | `src/moe_transformer_lab/trainable/train_utils.py` | Device choice, optimizer setup, checkpoint save/load. |
| Dense train wrapper | `scripts/train_dense.py` | Dense model training entrypoint. |
| Dense/MoE train script | `scripts/train_moe.py` | Shared training script for dense and MoE models. |
| Generation script | `scripts/generate.py` | Checkpoint text generation path. |
| Figure generator | `scripts/generate_figures.py` | Regenerates notebook SVG diagrams. |
| Tests | `tests/` | Correctness checks for manual layers, attention, config, checkpoint, MoE, notebook assets. |

## External Resources

Use these resources on the assigned dates.

| Resource | URL | Use |
| --- | --- | --- |
| Attention Is All You Need | https://arxiv.org/abs/1706.03762 | Embeddings, scaled dot-product attention, multi-head attention. |
| nanoGPT | https://github.com/karpathy/nanoGPT | File organization, tensor naming, training loop shape. |
| PyTorch SDPA docs | https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html | Input expectations for optimized attention. |
| LLaMA paper | https://arxiv.org/abs/2302.13971 | Pre-norm, RMSNorm, SwiGLU, RoPE choices. |
| RoFormer paper | https://arxiv.org/abs/2104.09864 | Rotary position embedding intuition and equations. |
| PyTorch AdamW docs | https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html | Weight decay and optimizer behavior. |
| Hugging Face generation docs | https://huggingface.co/docs/transformers/main_classes/text_generation | Temperature, top-k, top-p generation settings. |
| Nucleus sampling paper | https://arxiv.org/abs/1904.09751 | Motivation for top-p sampling. |
| Hugging Face KV cache guide | https://huggingface.co/docs/transformers/v4.52.1/kv_cache | Default cache, cache types, prefill/decode behavior. |
| PyTorch profiler docs | https://docs.pytorch.org/docs/stable/profiler.html | `profile`, `record_shapes`, `profile_memory`. |
| FlashAttention paper | https://arxiv.org/abs/2205.14135 | IO-aware attention bottleneck intuition. |

## 2026-06-13, Day 1: Orientation, Shape Contract, Embeddings, Loss

### Goal

Understand the repo, the combined assignment, and the shape path from token IDs
to logits and loss. This day grounds the Project 01 inference path in the
existing Transformer/MoE assignment.

### Reading

- Read **Attention Is All You Need**, sections **3.1** and **3.4** only.
  - Section 3.1: encoder/decoder framing. Focus on what a decoder consumes and
    produces.
  - Section 3.4: embeddings and softmax. Focus on the embedding table and final
    vocabulary projection.
- Skim nanoGPT `model.py`.
  - Look only for file organization and tensor names.
  - Do not copy implementation details yet.

### Notebook Work

Open `notebooks/assignment_transformer_moe.ipynb` and work through:

- `Project 01 Additive Track: Transformer Inference Foundations`;
- `Mini-Lecture: Decoder-Only LM Tensor Shapes`;
- `Shape Discipline and Gradient Checks`;
- `Embeddings and Cross Entropy`.

### Repo Inspection

Inspect these files:

- `src/moe_transformer_lab/config.py`
  - Locate `ModelConfig`.
  - Confirm `vocab_size`, `max_seq_len`, `d_model`, `n_layers`, `n_heads`, and
    `d_ff`.
- `src/moe_transformer_lab/manual/layers.py`
  - Locate `Linear`, `Embedding`, `LayerNorm`, and `SoftmaxCrossEntropy`.
  - Identify what each layer caches for backward.
- `src/moe_transformer_lab/trainable/model.py`
  - Locate `DecoderOnlyTransformer`.
  - Find token embedding, position embedding, final norm, and LM head.
- `tests/test_config.py`
- `tests/test_manual_layers.py`
- `tests/test_swappable_ffn.py`

### Assignment Tasks

- Write the complete shape path:
  - `idx`: `(B, T)`;
  - token embeddings: `(B, T, C)`;
  - position information: `(B, T, C)`;
  - transformer block output: `(B, T, C)`;
  - logits: `(B, T, V)`;
  - flattened logits for loss: `(B*T, V)`;
  - flattened targets for loss: `(B*T)`.
- Write parameter-count formulas:
  - token embedding: `V * C`;
  - learned position embedding: `T_max * C`;
  - untied LM head without bias: `C * V`;
  - LM head with bias, if present: `C * V + V`.
- Identify which current code path is manual and which is PyTorch autograd.
- Note how MoE changes only the feed-forward block and not embeddings, loss, or
  logits.

### Verification

Run or plan these commands:

```powershell
python -m pytest tests/test_config.py
python -m pytest tests/test_manual_layers.py
python -m pytest tests/test_swappable_ffn.py::test_lm_forward_works_for_dense_and_moe -rs
```

If PyTorch is not installed, record the skip reason and continue with the
non-Torch tests.

### Written Artifact

Add a Day 1 note with:

- the shape path from IDs to logits;
- embedding and LM-head parameter formulas;
- one thing that is clear;
- one thing about attention shapes that is still unclear.

### Self-Check Questions

1. Why do `idx` and `targets` have the same shape but different token offsets?
2. Why does cross entropy expect logits shaped like `(B*T, V)`?
3. What part of the current model makes token order visible?
4. Why can the same language-model loss train dense and MoE variants?

### Done Criteria

- You can say the shape of every tensor from `idx` to loss.
- `ModelConfig` fields are mapped to model dimensions.
- The notebook and code paths for embeddings/loss are located.

## 2026-06-14, Day 2: Causal Attention And Manual Transformer Core

### Goal

Understand causal self-attention, the mask, multi-head reshaping, and how the
manual Transformer core connects to the trainable model.

### Reading

- Read **Attention Is All You Need**, section **3.2**.
  - Focus on scaled dot-product attention.
  - Focus on multi-head attention and why heads split `C` into `H * Dh`.
- Read PyTorch SDPA docs enough to answer:
  - What input shapes does SDPA accept?
  - What does `is_causal=True` mean?
  - How does dropout behave?

### Notebook Work

Work through:

- `Mini-Lecture: Causal Multi-Head Attention Shapes`;
- `Mini-Lecture: Why Attention Memory Is O(T^2)`;
- `Causal Mask and Attention`;
- `Dense MLP and Residual Connector`.

### Repo Inspection

Inspect these files:

- `src/moe_transformer_lab/manual/attention.py`
  - `causal_mask`;
  - `masked_row_softmax`;
  - `ScaledDotProductAttention`;
  - `MultiHeadSelfAttention`.
- `src/moe_transformer_lab/manual/transformer.py`
  - manual dense MLP;
  - pre-norm block;
  - manual decoder-only model.
- `src/moe_transformer_lab/trainable/model.py`
  - `CausalSelfAttention`;
  - `TransformerBlock`.
- `tests/test_manual_attention.py`
- `tests/test_manual_transformer.py`, if present.

### Assignment Tasks

- Derive the attention shapes in your notes:
  - `x`: `(B, T, C)`;
  - `q`, `k`, `v` before split: `(B, T, C)`;
  - `q`, `k`, `v` after split: `(B, H, T, Dh)`;
  - attention scores: `(B, H, T, T)`;
  - attention probabilities: `(B, H, T, T)`;
  - per-head output: `(B, H, T, Dh)`;
  - merged output: `(B, T, C)`.
- Explain why the mask is lower triangular.
- Identify exactly where `O(T^2)` memory appears.
- Compare manual attention with the trainable attention path.
- Record the later SDPA upgrade point:
  - current code manually computes scores and softmax;
  - planned optional path can call `torch.nn.functional.scaled_dot_product_attention`.

### Verification

Run or plan:

```powershell
python -m pytest tests/test_manual_attention.py
python -m pytest tests/test_manual_layers.py
python -m pytest tests/test_swappable_ffn.py::test_transformer_block_accepts_dense_or_moe_without_shape_change -rs
```

Manual attention correctness should cover:

- causal mask shape;
- masked probabilities;
- output shape;
- backward path, if available.

### Written Artifact

Write a Day 2 note with:

- one full attention shape table;
- a one-paragraph explanation of why future tokens must be masked;
- a short prefill/decode preview:
  - prefill score shape: `(B, H, T, T)`;
  - one-token cached decode score shape: `(B, H, 1, T_cache)`.

### Self-Check Questions

1. Why does each attention row sum to 1?
2. Why does the attention matrix have `T` rows and `T` columns?
3. Which tensor grows quadratically with sequence length?
4. Why does the residual block output still have shape `(B, T, C)`?

### Done Criteria

- You can derive Q/K/V and score shapes without looking.
- You know where the mask is applied.
- You can point to the exact current attention implementation.

## 2026-06-15, Day 3: Modern Decoder Block Upgrade Path

### Goal

Understand the modern block choices required by Project 01 and define the
future implementation checkpoints without breaking the existing LayerNorm/GELU
and MoE path.

### Reading

- Read **LLaMA**, section **2.2**.
  - Focus only on pre-normalization, RMSNorm, SwiGLU, and RoPE.
  - Ignore large-scale training details for today.
- Read **RoFormer**, sections **1-3**.
  - Stop when the rotation equation and Q/K placement are clear.

### Notebook Work

Work through:

- `Mini-Lecture: RoPE Intuition`;
- `Mini-Lecture: RMSNorm And SwiGLU`;
- `Dense MLP and Residual Connector`;
- `Swappable Feed-Forward Interface`.

### Repo Inspection

Inspect:

- `src/moe_transformer_lab/trainable/model.py`
  - current `DenseMLP`;
  - current `CausalSelfAttention`;
  - current `TransformerBlock`;
  - current learned `position_embedding`.
- `src/moe_transformer_lab/config.py`
  - decide which future config fields would be needed.
- `tests/test_swappable_ffn.py`
  - identify tests that must keep passing after modern-block additions.

### Assignment Tasks

- Write the current block in pseudocode:

```python
x = x + attention(LayerNorm(x))
x = x + DenseMLP_or_MoE(LayerNorm(x))
```

- Write the planned Project 01 modern dense block in pseudocode:

```python
x = x + attention(RMSNorm(x), rope_positions=positions, past_key_values=optional_cache)
x = x + SwiGLU(RMSNorm(x))
```

- Define future implementation checkpoints:
  - `RMSNorm`: preserves `(B, T, C)` and uses root-mean-square scaling.
  - `RoPE`: creates cosine/sine cache and rotates Q/K after head split.
  - `SwiGLU`: uses value and gate projections, then an output projection.
  - config: add opt-in choices without removing current defaults.
- State compatibility rule:
  - current LayerNorm/GELU path remains valid;
  - MoE remains a feed-forward variant;
  - existing tests must continue to pass.

### Verification

No code change is required for this roadmap day. If implementing later, planned
tests should include:

```powershell
python -m pytest tests/test_modern_block.py
python -m pytest tests/test_rope.py
python -m pytest tests/test_swappable_ffn.py -rs
```

For now, run existing non-mutating checks:

```powershell
python -m pytest tests/test_config.py
python -m pytest tests/test_swappable_ffn.py -rs
```

### Written Artifact

Write a Day 3 note with:

- current block vs planned block table;
- RoPE placement: Q and K only, after projection and head split;
- RMSNorm vs LayerNorm paragraph;
- why SwiGLU can replace GELU MLP while preserving the same external shape.

### Self-Check Questions

1. Why should RoPE not rotate V?
2. What statistic does RMSNorm skip compared with LayerNorm?
3. Why does SwiGLU need two input projections?
4. Which changes affect inference position handling?

### Done Criteria

- You can describe the modern decoder-block upgrade path.
- You know what can be added without deleting the existing assignment path.
- Future tests for RoPE/RMSNorm/SwiGLU are specified.

## 2026-06-16, Day 4: Training Loop And Tiny Overfit

### Goal

Understand the training loop, shifted targets, optimizer behavior, gradient
clipping, checkpointing, and the difference between cross entropy and MoE
auxiliary loss.

### Reading

- Read nanoGPT `train.py` sections for:
  - batching;
  - optimizer setup;
  - evaluation loop;
  - checkpointing.
- Read PyTorch AdamW docs enough to explain:
  - what weight decay does;
  - why biases and norm parameters are often excluded from decay;
  - what `betas` mean at a high level.

### Notebook Work

Work through:

- `Mini-Lecture: Language Modeling Objective And Shifted Targets`;
- `One Training Step, End to End`;
- `Local TinyStories Training`;
- `MoE Load-Balancing Loss`;
- `Why MoE Uses Autograd Here`.

### Repo Inspection

Inspect:

- `scripts/train_dense.py`
  - confirm it wraps the dense path.
- `scripts/train_moe.py`
  - parse CLI arguments;
  - locate config construction;
  - locate eval loop and checkpoint save.
- `src/moe_transformer_lab/trainable/data.py`
  - locate token loading and `get_batch`.
- `src/moe_transformer_lab/trainable/tokenizer.py`
  - locate byte tokenizer vs trained tokenizer.
- `src/moe_transformer_lab/trainable/train_utils.py`
  - `configure_optimizer`;
  - `save_checkpoint`;
  - `load_checkpoint`;
  - `count_parameters`.
- `tests/test_tiny_overfit.py`
- `tests/test_checkpoint.py`

### Assignment Tasks

- Write the one-step training flow:

```text
tokens -> sample x,y -> model(x,y) -> ce_loss + aux_loss -> backward
-> clip gradients -> AdamW step -> zero gradients -> periodic eval/checkpoint
```

- Explain why `x` and `y` differ by one token.
- Explain what cross entropy averages over.
- Explain why MoE adds `aux_loss` but keeps the same next-token objective.
- Identify checkpoint contents:
  - model state;
  - config;
  - optimizer state, when provided;
  - step;
  - extra metadata.

### Verification

Run or plan:

```powershell
python -m pytest tests/test_tiny_overfit.py -rs
python -m pytest tests/test_checkpoint.py -rs
```

If PyTorch is installed, run a tiny smoke command:

```powershell
python scripts/train_dense.py --steps 20 --eval-interval 10 --block-size 64 --batch-size 4 --d-model 128 --n-layers 2 --n-heads 4 --d-ff 512 --max-train-chars 200000 --max-valid-chars 50000
```

If data is missing, record the missing prerequisite:

```powershell
python scripts/download_tinystories.py --out-dir data/tinystories/raw
python scripts/train_tokenizer.py --raw-dir data/tinystories/raw --out data/tinystories/tokenizer.json
```

### Written Artifact

Write a Day 4 note with:

- shifted-target explanation;
- cross entropy shape explanation;
- training loop pseudocode;
- checkpoint content list;
- any instability or missing-data blocker.

### Self-Check Questions

1. Why can training process all positions at once?
2. Why does generation still need a loop?
3. What does gradient clipping protect against?
4. What does MoE router collapse mean?

### Done Criteria

- You understand the training path from batch to checkpoint.
- You can separate CE loss from MoE auxiliary loss.
- Tiny overfit path is verified or the missing dependency/data blocker is
  recorded.

## 2026-06-17, Day 5: Generation, Sampling, And MoE Extension

### Goal

Understand autoregressive generation, sampling controls, current top-k support,
future top-p support, and the MoE feed-forward extension.

### Reading

- Read Hugging Face generation docs for:
  - greedy decoding;
  - temperature;
  - top-k;
  - top-p.
- Optional: read the abstract and introduction of **The Curious Case of Neural
  Text Degeneration** for nucleus sampling motivation.

### Notebook Work

Work through:

- `Mini-Lecture: Sampling Controls`;
- `One Generation Step, End to End`;
- `MoE Intuition`;
- `MoE Tensor Flow`;
- `MoE Implementation Steps`;
- `MoE Pseudocode`;
- `MoE Debugging Checklist`.

### Repo Inspection

Inspect:

- `scripts/generate.py`
  - arguments;
  - tokenizer loading;
  - checkpoint loading;
  - call to `model.generate`.
- `src/moe_transformer_lab/trainable/model.py`
  - `DecoderOnlyTransformer.generate`;
  - top-k filtering;
  - temperature scaling;
  - current absence of top-p;
  - `MoEFeedForward.forward`;
  - `replace_feed_forward_modules`.
- `tests/test_swappable_ffn.py`
  - dense/MoE contract tests.

### Assignment Tasks

- Write the current generation loop:

```text
idx -> crop to max_seq_len -> model forward -> last-position logits
-> temperature scaling -> optional top-k -> softmax -> multinomial sample
-> append token -> repeat
```

- Define planned top-p behavior for a later implementation pass:
  - sort probabilities descending;
  - compute cumulative probability;
  - keep the smallest prefix whose cumulative mass reaches `top_p`;
  - always keep at least one token;
  - renormalize before sampling.
- Define deterministic sampling tests:
  - fixed seed gives same output for stochastic path;
  - greedy or temperature-0 path is deterministic;
  - filtered tokens are never sampled.
- Explain the difference between:
  - vocabulary top-k sampling;
  - MoE router top-k expert selection.

### Verification

Run or plan:

```powershell
python -m pytest tests/test_swappable_ffn.py -rs
```

If a checkpoint exists, smoke generation:

```powershell
python scripts/generate.py --checkpoint runs/tinystories_moe/ckpt_last.pt --prompt "Once upon a time" --max-new-tokens 40 --temperature 0.9 --top-k 50
```

If no checkpoint exists, record that generation requires Day 4 training output.

### Written Artifact

Write a Day 5 note with:

- generation loop pseudocode;
- sampling methods table;
- one failure mode each for greedy, temperature, top-k, and top-p;
- MoE routing shape table or pointer to the notebook table.

### Self-Check Questions

1. Which logits position does generation use?
2. Why does top-p adapt candidate count while top-k does not?
3. Why is router top-k not a vocabulary filter?
4. Why does MoE not require a new generation loop?

### Done Criteria

- You can explain current generation code.
- You know exactly what future top-p implementation must do.
- You can explain MoE routing without confusing it with token sampling.

## 2026-06-18, Day 6: KV Cache, Prefill, Decode

### Goal

Define the KV-cache mental model and the future code interface for cached
decode. This is the main conceptual bridge from a toy Transformer to serving
systems.

### Reading

- Read Hugging Face KV cache guide sections:
  - Default cache;
  - cache type comparison table.
- Re-read your Day 2 notes on:
  - attention score shapes;
  - causal masking;
  - `O(T^2)` memory.

### Notebook Work

Work through:

- `Mini-Lecture: Prefill Vs Decode`;
- `Mini-Lecture: KV-Cache Shape Contract And Memory Formula`;
- `Mini-Lecture: Cached Vs Uncached Generation Correctness`;
- `Mini-Lecture: Why Attention Memory Is O(T^2)`.

### Repo Inspection

Inspect:

- `src/moe_transformer_lab/trainable/model.py`
  - `CausalSelfAttention.forward`;
  - `TransformerBlock.forward`;
  - `DecoderOnlyTransformer.forward`;
  - `DecoderOnlyTransformer.generate`.
- `configs/tiny_inference.json`
  - planned inference settings;
  - `use_kv_cache`;
  - prompt lengths;
  - generation settings.
- `reports/transformer_inference_report.md`
  - sections for implementation, setup, results, limitations.

### Assignment Tasks

- Write the planned cache contract:

```python
past_key_values = [
    (past_key_layer_0, past_value_layer_0),
    (past_key_layer_1, past_value_layer_1),
    ...
]
```

- Record shapes:
  - `past_key`: `(B, H, T_cache, Dh)`;
  - `past_value`: `(B, H, T_cache, Dh)`;
  - `new_key`: `(B, H, 1, Dh)`;
  - `new_value`: `(B, H, 1, Dh)`;
  - updated cache: `(B, H, T_cache + 1, Dh)`.
- Write the planned forward signatures:

```python
CausalSelfAttention.forward(x, past_key_value=None, use_cache=False, position_offset=0)
TransformerBlock.forward(x, past_key_value=None, use_cache=False, position_offset=0)
DecoderOnlyTransformer.forward(idx, targets=None, past_key_values=None, use_cache=False)
```

- Write the memory formula:

```text
bytes = layers * 2 * batch * heads * seq_len * head_dim * dtype_bytes
```

- Write one concrete numeric example using this repo's tiny config:
  - `layers=2`;
  - `batch=1`;
  - `heads=4`;
  - `seq_len=128`;
  - `head_dim=32`;
  - fp16 `dtype_bytes=2`;
  - memory = `2 * 2 * 1 * 4 * 128 * 32 * 2 = 65536` bytes, about `64 KiB`.

### Verification

Planned future tests:

```powershell
python -m pytest tests/test_kv_cache.py
python -m pytest tests/test_kv_cache.py::test_cached_decode_matches_uncached_final_token
```

Expected correctness rule:

- run model in eval mode;
- run full uncached prompt and read final-token logits;
- run prefill/decode cached path;
- compare final-token logits within tolerance;
- do not compare sampled text as the primary correctness signal.

For now, run available checks:

```powershell
python -m pytest tests/test_config.py
python -m pytest tests/test_swappable_ffn.py -rs
```

### Written Artifact

Write a Day 6 note with:

- prefill vs decode table;
- cache shape table;
- memory formula;
- tiny numeric example;
- one paragraph on where batching complicates cache management.

### Self-Check Questions

1. What is stored per layer in the KV cache?
2. Why is there a factor of `2` in the memory formula?
3. Why does cached decode still need to read previous keys and values?
4. Why must position handling be correct during decode?

### Done Criteria

- You can derive the KV-cache memory formula from shapes.
- You can describe the planned code interface.
- You can state the cached-vs-uncached correctness test precisely.

## 2026-06-19, Day 7: Benchmark, Report, Portfolio Gate

### Goal

Turn the week into a portfolio-grade artifact: a report outline, benchmark
expectations, reproducibility commands, and a clear list of what is implemented
versus planned.

### Reading

- Read PyTorch profiler docs sections for:
  - `torch.profiler.profile`;
  - `record_shapes`;
  - `profile_memory`.
- Optional: read FlashAttention abstract and introduction.
  - Focus on IO-awareness and why memory movement can dominate attention.

### Notebook Work

Work through:

- `Mini-Lecture: Benchmark Vocabulary`;
- `Mini-Lecture: Why Attention Memory Is O(T^2)`;
- any previous mini-lecture that still feels weak.

### Repo Inspection

Inspect:

- `reports/transformer_inference_report.md`
  - fill placeholder notes if benchmark code is not implemented yet.
- `configs/tiny_inference.json`
  - use as the seed for later benchmark CLI defaults.
- `results/`
  - confirm this is where raw logs should go.
- `README.md`
  - confirm Project 01 scope is visible.
- `docs/project_01_scope.md`
  - confirm scope and out-of-scope items match reality.

### Assignment Tasks

- Define later benchmark script behavior:

```powershell
python scripts/benchmark_inference.py --config configs/tiny_inference.json --out results/tiny_inference.jsonl
```

- Required benchmark fields:
  - timestamp;
  - device;
  - dtype;
  - model config;
  - prompt length;
  - new tokens;
  - prefill latency;
  - decode latency;
  - generated tokens/sec;
  - peak GPU memory, if CUDA is available;
  - cached vs uncached mode;
  - seed.
- Required report table:

| Mode | Prompt Tokens | New Tokens | Prefill Latency | Decode Tokens/Sec | Peak Memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uncached | TBD | TBD | TBD | TBD | TBD |
| Cached | TBD | TBD | TBD | TBD | TBD |

- Write the portfolio gate:
  - one correctness test;
  - one benchmark table;
  - one bottleneck explanation;
  - one limitation;
  - one command path another person can run.

### Verification

Run:

```powershell
python -m pytest tests/test_notebook_assets.py
python -m pytest tests/test_config.py
python -m pytest tests/test_swappable_ffn.py -rs
```

Record:

- pass/fail/skip status;
- if skipped, the exact reason, such as missing `torch`;
- any missing data or checkpoint prerequisites.

### Written Artifact

Update or annotate `reports/transformer_inference_report.md` with:

- research question;
- current implementation summary;
- planned benchmark command;
- result table placeholder;
- limitations;
- next steps toward serving systems.

### Self-Check Questions

1. What metric best describes time to first token?
2. Why should prompt length be recorded in every benchmark row?
3. Why is cached-vs-uncached speedup not the same as production serving throughput?
4. How does this Project 01 foundation prepare you for vLLM/SGLang serving work?

### Done Criteria

- The final report has a clear structure.
- The benchmark expectations are precise enough to implement.
- Test status is recorded.
- You can give a 3-minute explanation of attention, generation, prefill/decode,
  KV-cache memory, and MoE as a feed-forward extension.

## Final Acceptance Gate

The combined Project 01 assignment is done when all of these are true.

### Conceptual Gate

- You can explain `(B, T) -> (B, T, C) -> (B, T, V)` without notes.
- You can derive multi-head attention shapes through `(B, H, T, Dh)`.
- You can explain why standard attention creates `O(T^2)` score/probability
  tensors.
- You can explain RoPE placement even if it is not implemented yet.
- You can compare LayerNorm/RMSNorm and GELU/SwiGLU.
- You can explain greedy, temperature, top-k, and top-p sampling.
- You can explain prefill vs decode.
- You can derive the KV-cache memory formula.
- You can explain why MoE is a feed-forward swap, not a new language-model
  objective.

### Repo Gate

- Existing dense and MoE assignment material remains intact.
- Project 01 docs exist:
  - `docs/project_01_scope.md`;
  - `docs/day01_to_day07_checklist.md`;
  - `docs/project_01_combined_assignment_roadmap.md`.
- Config/report scaffolds exist:
  - `configs/tiny_inference.json`;
  - `reports/transformer_inference_report.md`;
  - `results/`.
- Notebook has Project 01 mini-lecture content.

### Test Gate

Run and record:

```powershell
python -m pytest tests/test_notebook_assets.py
python -m pytest tests/test_config.py
python -m pytest tests/test_swappable_ffn.py -rs
```

If PyTorch is unavailable, `test_swappable_ffn.py` may skip. Record that as an
environment limitation, not a conceptual pass.

### Portfolio Gate

Before moving to serving systems, prepare:

- one concise README summary of Project 01 purpose;
- one report section explaining attention and KV-cache bottlenecks;
- one planned or actual benchmark table;
- one list of missing implementation work;
- one 3-minute verbal explanation covering:
  - production problem;
  - baseline;
  - metric;
  - failure mode;
  - what Codex helped with;
  - what you personally designed and understand.
