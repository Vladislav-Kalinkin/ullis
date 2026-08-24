# Ullis

**Ullis trains dense ternary Hyena language models on Apple Silicon.** Its design target is useful context length and parameter count on a personal Mac, without paying for a hidden FP32 model or Adam/Lion-sized optimizer state.

The normal training backend is Metal. CPU exists as a deterministic, numerically transparent fallback for tests and debugging.

## What is implemented

- FP16 persistent embeddings, latent ternary masters, and compact implicit filters; ternary projections are two packed bitplanes plus one row scale.
- Bounded causal Hyena overlap-save convolution: FFT workspace depends on configured filter and chunk, not the whole context.
- Two-horizon MTP training (`t+1`, `t+2`) with streamed softmax/cross-entropy. Neither CPU nor Metal materialises `[batch,time,vocab]`.
- Stateless clipped SGD: no momentum, variance, or full-model gradient state. This is the RAM floor, not a claim of convergence equivalence to AdamW or Lion.
- Metal-resident ternary heads, loss gradient, reverse activation stream, projection updates, ternary-code refresh, and FP16 compact filter state.
- Portable versioned checkpoints preserving exact FP16 bit patterns, including explicit snapshots of Metal-resident weights.
- Byte-level BPE, JSONL datasets, resumable CLI runs, greedy generation, and a small line-oriented chat loop with saved sessions.

## Architecture and memory contract

The architecture is intentionally narrow: tied token embedding/output table, stacked causal Hyena blocks, then independent `t+1` and `t+2` ternary heads. There is no attention matrix, KV cache, MoE router, recurrent state, or optimizer moments.

Persistent learned tensors are binary16. Arithmetic widens only where a kernel needs it; the model does not retain an FP32 master copy. Inference codes are re-derived from each FP16 ternary master after an update. The Metal softmax makes two vocabulary scans per supervised row and retains only a `[B*T,D]` derivative—not vocabulary-wide logits or probabilities.

The current exact filter-backward bridge reads compact `O(D*order)` filter parameters/statistics at a Metal update boundary. It never transfers full activations, output gradients, logits, or optimizer state. Replacing that reference bridge with an entirely GPU-resident filter adjoint is the main remaining GPU-training optimization.

## Quick start

Build and validate:

```sh
cargo test
cargo run -- --smoke
```

On an Apple Silicon Mac, train the included tiny example with Metal:

```sh
cargo run -- train --data examples/first-train.jsonl --run runs/hello --steps 100 --checkpoint-every 25
```

The output directory contains `config.json`, `tokenizer.json`, append-only `metrics.jsonl`, and lossless `checkpoint.json`. Continue a run with:

```sh
cargo run -- train --data examples/first-train.jsonl --run runs/hello --resume runs/hello/checkpoint.json --steps 100
```

CPU is intentionally explicit:

```sh
cargo run -- train --data examples/first-train.jsonl --run runs/cpu-check --steps 1 --backend cpu
```

For the complete command reference, read [USAGE](/Users/vladislavkalinkin/ullis/USAGE).

## Dataset contract

Training data is JSONL. Each line is a conversation; `assistant.thinking` is required and stays a separate delimited training target. Optional structured `tool_calls` and `tool_call_id` reserve a stable shape for future agentic traces without pretending that tool execution is already implemented.

```json
{
  "id": "demo-1",
  "messages": [
    { "role": "system", "content": "Be concise." },
    { "role": "user", "content": "What is 2+2?" },
    {
      "role": "assistant",
      "thinking": "Use direct arithmetic.",
      "content": "4"
    }
  ],
  "metadata": { "split": "train" }
}
```

## Status and next technical work

Ullis is ready for its first genuine small-scale training runs on Metal: training, checkpoints, resume, inspection, and decoding are connected. It is not yet a claim of production-quality model training or general chat quality; the bundled example is only a wiring check.

The highest-value next steps are a fully resident compact-filter backward, throughput/memory profiling on real M-series hardware, evaluation and sampling controls, and a deliberate policy for thinking supervision and future tool-call execution.
