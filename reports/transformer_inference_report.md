# Transformer Inference Foundations Report

## Abstract

Summarize the model, inference path, main measurement, and the most important
finding in 150-250 words.

## 1. Research Question

What does this project test about decoder-only Transformer inference?

Example:

> How much does KV-cache decoding reduce repeated attention work for a tiny
> decoder-only Transformer, and what bottleneck remains during token-by-token
> generation?

## 2. Background

Explain the minimum concepts needed to interpret the results:

- causal self-attention;
- prefill vs decode;
- KV cache;
- sampling controls;
- latency, throughput, and memory.

## 3. Implementation

Describe the model path and any inference-specific additions:

- model configuration;
- attention implementation;
- generation path;
- planned or implemented KV-cache API;
- benchmark script and output format.

## 4. Experimental Setup

Record exact reproducibility details:

- command;
- device and GPU, if available;
- PyTorch version;
- model size;
- prompt lengths;
- number of generated tokens;
- dtype;
- seed.

## 5. Results

Add tables or plots generated from `results/`.

Suggested table:

| Mode | Prompt Tokens | New Tokens | Latency | Tokens/Sec | Peak Memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uncached | TBD | TBD | TBD | TBD | TBD |
| Cached | TBD | TBD | TBD | TBD | TBD |

## 6. Ablations And Failure Analysis

Include at least one controlled comparison:

- cached vs uncached decode;
- prompt length sweep;
- top-k/top-p settings;
- CPU vs GPU smoke comparison, if available.

Document at least one failure or limitation, such as context cropping, memory
growth, numerical mismatch, or small-model sampling quality.

## 7. Limitations

State what this project does not prove. Examples:

- tiny model results do not directly predict large-model serving performance;
- no custom kernels;
- no continuous batching;
- no production scheduler.

## 8. Conclusion And Next Steps

Connect this foundation to later serving-system work: vLLM/SGLang, prefix
caching, paged KV allocation, batching, and cost-to-serve analysis.

## Appendix: Commands

```powershell
python -m pytest tests/test_config.py
python -m pytest tests/test_swappable_ffn.py
python -m pytest tests/test_tiny_overfit.py
```
