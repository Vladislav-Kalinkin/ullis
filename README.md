# Ullis

**Ullis 0.10 trains RWKV-8 Heron / ROSA language models on Apple Silicon.** The default stack is `LayerNorm → ROSA-QKV-1bit → CMix x070`. There is no Hyena, no FFT, and no hidden FP32/FP16 master matrix in the checkpoint file.

The normal training backend is Metal. CPU is a deterministic fallback for tests, debugging, and hybrid digit eval.

This is not a claim of production chat quality. 0.10 is a local trainer/infer that fits an M1 8 GiB Mac with a **4 GiB** process budget.

## What 0.10 actually is

- **Default architecture `heron`:** `train_config.json` is D=512, L=12, T=2048, vocab ceiling 8192, batch 1. The in-code `TrainConfig::default()` stays the smaller D=256 / L=6 admission profile.
- **1-bit ROSA is an activation alphabet, not weight quantization.** QKV bits are `{0,1}` from `x > 0`. The float output is `out = (2·idx − 1)·e`: unmatched and matched-0 go to **−e**, matched-1 to **+e**. This is not 4-bit ROSA (unmatched → 0); 4-bit is not in 0.10.
- **Packed ±1** only where official linears are large and not zero-init (QKVO, CMix key, head), with a learned FP16 row scale and official bias. CMix value, Tmix (hybrid), LayerNorm, embeddings, and ROSA `e` stay FP16. BinaryConnect latents and the FP32 SGD carry live in RAM and are written into `checkpoint.safetensors`. Legacy JSON v2 files still omit them and reconstruct `|w|=0.01`.
- **Train 0.10 is `stop_grad_bits`:** ROSA QKV / `ln3` / `x_qkv` are frozen. Learning is `g_e` plus BinaryConnect on `o`, CMix, head, and SGD on embeddings / LN / `e`. Next-token CE only (`t+1`). There is no MTP `t+2`.
- **Metal-resident** LN, packed linear, FP16 linear, streamed CE, and 1-bit ROSA forward. No MPS GEMM. Host traffic is token ids, loss scalars, and checkpoint I/O.
- **Checkpoints are v2.** Hyena `format_version: 1` files are intentionally unloadable (`--resume` hard-fails). Old Hyena run directories (`runs/diagnostic`, `runs/ullis_gradient_fixed`, …) are leftover artifacts: delete them by hand; there is no converter.
- Byte-level BPE, JSONL with required `assistant.thinking`, resumable CLI, greedy generate (incremental `RosaSam::push`), and a line-oriented chat loop.

Optional **`rosa_rwkv7`** adds FP16 RWKV-7 TimeMix beside ROSA for `eval-digits`. Hybrid **train** is not wired in 0.10. Digit smoke is diagnostic, not a 90% accuracy gate. There is **no 4-bit profile**.

## Memory

Unified RAM is shared with macOS and the Metal driver. The admission budget is **4 GiB** (`memory_budget_bytes` in `train_config.json`). Do **not** pass `--memory-budget-mib 8192` on an 8 GiB Mac.

The default Heron train peak is on the order of **150 MiB** of Ullis state. The bottleneck after the Hyena cut is SAM latency, not FFT workspace.

## Quick start

```sh
cargo test
cargo run -- --smoke
```

Train with the M1 default profile:

```sh
cargo run -- train --data data/ullis_dataset.jsonl --run runs/hello --config train_config.json --steps 100 --learning-rate 0.01 --checkpoint-every 25
```

`--config` accepts JSON or TOML. The file in this repo is **`train_config.json`**, not `config.toml`.

A run directory contains `config.json` (effective config), `tokenizer.json`, append-only `metrics.jsonl`, and `checkpoint.safetensors` (v2 weights, completed step, BinaryConnect latents, and SGD carry). Legacy `checkpoint.json` still loads (bits only; latents are reconstructed). Progress prints raw window loss, an EMA, and process RSS; use the EMA.

Resume continues from the saved (or metrics-inferred) step until `--steps`, which is the **target global step**:

```sh
cargo run -- train --data data/ullis_dataset.jsonl --run runs/hello --resume runs/hello --steps 10000
```

CPU is explicit:

```sh
cargo run -- train --data data/ullis_dataset.jsonl --run runs/cpu-check --config train_config.json --steps 1 --backend cpu
```

Inspect, generate, chat, and digit eval:

```sh
cargo run -- inspect --run runs/hello
cargo run -- generate --checkpoint runs/hello --prompt 'Hello' --max-tokens 64
cargo run -- chat --checkpoint runs/hello --session sessions/first.jsonl
cargo run -- eval-digits --checkpoint runs/hello --task reverse --max-digits 8
```

`eval-digits` needs a `rosa_rwkv7` checkpoint (vocab 12 reverse / 13 plusminus). Sequences are padded to T=144 or 272; accuracy is the unpadded digit span. Full flags: [USAGE](USAGE).

## Dataset contract

JSONL, one conversation per line. `assistant.thinking` is required. Roles: `system`, `user`, `assistant`, `tool`. Digit eval uses its own alphabets and does not go through this schema.

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

## Status

0.10 is a clean cut from dense ternary Hyena. Training, v2 checkpoints, resume, inspect, incremental generate, chat, and WKV7 hybrid eval are connected. Packed Tmix and 4-bit ROSA are out of scope. Linear QKV bit-grad (`exact_bitflip`) exists as CPU tests, not the default train path.
