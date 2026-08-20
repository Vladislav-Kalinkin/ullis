# Ullis

Standalone ternary Mixture-of-Bumps Kolmogorov–Arnold engine.
Zero Python at runtime. One binary: train, chat, smoke.

```
tokens → embed → L × (causal mixer + TernaryKanLinear) → RMSNorm → tied logits
```

Thinking is no longer silent. The REPL streams the hidden trajectory in
dim italic gray, prints `└──`, then streams the `output` block in bold green.
Ephemeral GC (`ReasoningScratch::clear`) wipes the think ring the instant
the output stream ends.

---

## Absolute 8 GB Mac M1 benchmarks

| Surface                 | Number          | Notes                                                           |
| ----------------------- | --------------- | --------------------------------------------------------------- |
| Pre-training throughput | **~7500 tok/s** | 4-phase ternary QAT, Metal, defaults `d=32 L=3 G=4→12 T=96 B=4` |
| Deep inference RSS      | **< 35 MB**     | `xhigh` resonance included; think scratch is ephemeral          |
| Unified memory          | **8 GB M1**     | SGD (no Adam states); JSONL I/O independent of corpus size      |

Working set is designed to stay **flat**: dialogue cache never stores
`thinking` tokens, the token ring is capped at 32 768 ids (~128 KB),
and Gauss–Jordan grid projection stays on Metal (`G ≤ 12`).

---

## Quick start

```bash
cargo build --release
./target/release/ullis train --data data/thinking-train.jsonl --steps 200 --out checkpoints/
./target/release/ullis chat  --model checkpoints/packed.bin --thinking medium
./target/release/ullis chat  --model checkpoints/packed.bin --thinking xhigh --prompt "fn add("
./target/release/ullis smoke
```

`--thinking low|medium|high|xhigh` sets the KAN eval budget. `low` masks
routed (thinking) weights and emits output immediately. `xhigh` runs three
residual KAN loops per block on the G=12 MoE stack, streamed live.

---

## Data — strict 4-key JSONL

Every training line is exactly:

```json
{ "system": "...", "user": "...", "thinking": "...", "output": "..." }
```

Packed as:

```
<|system|> … <|user|> … <|thinking|> … <|/thinking|> <|output|> …
```

Loss is masked onto `thinking` + `output` so the KAN layer learns the
logical chain (bracket matching, import tracking, lifetime ownership,
pipeline quoting). A verified sample corpus lives at
[`data/thinking-train.jsonl`](data/thinking-train.jsonl) (Rust, Python, Bash).

Legacy `{"text","lang"}` lines are still lifted in-stream.

---

## Visual reasoning UI

| Lane     | ANSI                     | Marker                   |
| -------- | ------------------------ | ------------------------ |
| thinking | `\x1b[2;3m` dim + italic | `[Ullis is thinking...]` |
| close    | reset                    | `└──`                    |
| output   | `\x1b[1;32m` bold green  | code stream              |

Colors honor `NO_COLOR` and non-TTY stdout. Persistent dialogue keeps only
`system` / `user` / `output`.

---

## Crate map

| Module       | Role                                                          |
| ------------ | ------------------------------------------------------------- |
| `quant`      | TWN threshold, STE, 2-bit pack/unpack                         |
| `gauss`      | G×G Gauss–Jordan (`matmul` / broadcast / `cat`, Metal-safe)   |
| `kan`        | `TernaryKanLinear` + ReLU-bump basis + MoB router             |
| `mixers`     | `CausalShift` (0 params) / tiny causal attention              |
| `model`      | `UllisKan`: embed → L × (shift + KAN) → RMSNorm → tied logits |
| `tokenizer`  | Byte-level BPE, vocab 4096, code-seeded merges                |
| `data`       | 4-key JSONL, `VecDeque` token ring                            |
| `think`      | budgets, ephemeral GC, dialogue cache                         |
| `train`      | 4-phase QAT, G = 4→8→12 projection, masked CE                 |
| `checkpoint` | `packed.bin` (magic `ULLIS03`)                                |
| `chat`       | ANSI token streamer                                           |
| `telemetry`  | RSS / tok/s / ternary histogram                               |

Design matrix: [`DESIGN.md`](DESIGN.md).

---

## License — MIT

Ullis is open-source software licensed under the **MIT License**. See [`LICENSE`](LICENSE).
