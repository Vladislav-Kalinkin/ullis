# Ullis — Rust Core Design Matrix

Standalone ternary Kolmogorov–Arnold engine. Python/PyTorch is gone.
Math is a faithful port of Ullis AI Engine v2.0 plus Mixture-of-Bumps.

Framework: **Candle 0.11** (Hugging Face) with native **Metal** on Apple Silicon
and Accelerate BLAS on CPU. Autodiff is Candle `Var` + `Tensor::backward`.
STE uses the hardtanh identity trick (no custom kernel, no `unsafe` autograd).

## Crate map

| Module       | Role                                                                                |
| ------------ | ----------------------------------------------------------------------------------- |
| `quant`      | TWN threshold, STE, 2-bit pack/unpack (`u8` bit-shifts)                             |
| `gauss`      | G×G Gauss–Jordan (`matmul`/`broadcast` only, G ≤ 12)                                |
| `kan`        | `TernaryKanLinear` + ReLU-bump basis + MoB router                                   |
| `mixers`     | `CausalShift` (0 params) / tiny causal attention                                    |
| `model`      | `UllisKan`: embed → L × (shift + KAN) → RMSNorm → tied logits                       |
| `tokenizer`  | Byte-level BPE, vocab 4096, code-seeded merges                                      |
| `data`       | 4-key JSONL (`system/user/thinking/output`) via `serde_json`, `VecDeque` token ring |
| `think`      | `--thinking` budgets, ephemeral reasoning GC, dialogue cache                        |
| `train`      | 4-phase QAT, G = 4→8→12 projection, SGD+momentum, masked CE on thinking+output      |
| `checkpoint` | Self-contained `packed.bin` (magic `ULLIS03`, loads v0.3 weights)                   |
| `telemetry`  | `mach_task_self` / `task_info` RSS, tok/s, ternary histogram                        |
| `chat`       | ANSI visual REPL: dim think stream, `└──`, bold-green output                        |

Binary: `ullis train | chat | smoke`

```
ullis train --data data/thinking-train.jsonl --steps 1000 --out checkpoints/
ullis chat  --model checkpoints/packed.bin --thinking medium
ullis chat  --model checkpoints/packed.bin --thinking xhigh --prompt "fn add("
```

## Layer math

Edge function (unchanged from v2):

```
φ_ji(x_i) = a_ji x_i + Σ_g b_jig ψ_g(x_i)
ψ_g(x)    = relu(1 − |x − c_g| / w)²
```

`ψ` is a quadratic ReLU bump, not a Cox–de Boor B-spline. Mixing is a
dense `F.linear` over flattened basis `[in · G] → out`. `bias = false`.

### Mixture-of-Bumps (DeepSeek-inspired)

`G` is split, not grown:

| G   | shared | routed |
| --- | ------ | ------ |
| 4   | 3      | 1      |
| 8   | 6      | 2      |
| 12  | 8      | 4      |

- **Shared bumps** always fire (syntax / grammar / keywords).
- **Routed bumps** have `K = 3` expert coefficient slices (python / rust / bash).
- Per-token micro-router: `softmax(x W_rᵀ)`, `W_r ∈ R^{K × in}` (tiny, FP32).
- Routed contribution is one batched GEMM: `[N, in·G_r] × [K·out, in·G_r]`,
  then `Σ_k g_k · y_k`. No language flags; the prompt tokens are the signal.

Disable with `--no-moe` to recover the exact v2 edge function.

### Dynamic grid

Training starts at `G = 4`. At the warmup midpoint the spline coefficients
are least-squares lifted onto `G = 8`, then onto `G = 12` at sparsify.
Projection: sample `M = max(64, 16G)` points, form `Ψ_newᵀ Ψ_new b' = Ψ_newᵀ Ψ_old b`.
The G×G solve is Gauss–Jordan with ridge `1e-6 · mean(diag)` — elementwise /
broadcast / `cat` only, so it stays on Metal. Frozen from QAT onward.

### 4-phase ternary QAT

| Phase | Name     | Weights                                    | Grid   | LR     |
| ----- | -------- | ------------------------------------------ | ------ | ------ |
| 1     | warmup   | FP `a`,`b`,centers                         | 4 → 8  | `3e-3` |
| 2     | sparsify | + L1 on `a`,`b`                            | → 12   | `3e-3` |
| 3     | qat      | STE ternary, TWN `δ = 0.7·mean(            | w_row  | )`     | frozen | `1e-3` |
| 4     | harden   | freeze `a`,`b`; train scales / RMS / embed | frozen | `3e-4` |

STE: forward discrete `{-1,0,+1}`; backward hardtanh gate `|w| ≤ 1`.

```
q = clamp(w, -1, 1) − detach(clamp(w, -1, 1)) + detach(ternary(w))
```

After phase 4, `pack()` drops dense FP weights. RAM holds int8 codes
(Metal has no 2-bit GEMM). Disk holds 2 bits/weight (`uint8`, 4 codes/byte:
`0→0`, `+1→1`, `-1→2`).

## Data (4-key JSONL)

Canonical line, streamed with `BufReader` + `serde_json::from_str`:

```json
{
  "system": "You are a Rust compiler specialist.",
  "user": "write add",
  "thinking": "i32 add",
  "output": "fn add(a: i32, b: i32) -> i32 { a + b }"
}
```

Packed as:

```
<|system|> … <|user|> … <|thinking|> … <|/thinking|> <|output|> …
```

Loss is masked onto the `thinking` + `output` span so the KAN layer learns
the reasoning trajectory. Legacy `{"text","lang"}` lines are lifted in-stream.
Token ring is a `VecDeque<u32>` capped at 32 768 ids (~128 KB).

## Thinking mode

| Flag     | KAN eval                                               | Think budget  | Visible path                                      |
| -------- | ------------------------------------------------------ | ------------- | ------------------------------------------------- |
| `low`    | Coarse G=4 base; routed (thinking) weights masked      | 0             | Immediate bold-green `<\|output\|>`               |
| `medium` | Full MoE-KAN                                           | `seq_len/4`   | Dim/italic think stream, `└──`, then green output |
| `high`   | Full MoE-KAN                                           | `seq_len/2`   | Same, longer think stream                         |
| `xhigh`  | G=12 MoE-KAN + 3 residual FF resonance loops per block | `3 seq_len/4` | Live think + output (no silent buffer)            |

After the turn, `ReasoningScratch::wipe` zeros and `shrink_to_fit`s the
thinking ring. `DialogueCache` keeps only `system`, `user`, `output`.
Working-set target remains a flat envelope under 35 MB RSS.

## Checkpoint (`packed.bin`)

```
[8]  magic  "ULLIS03\n"
[4]  u32le header_len
[H]  JSON { config, tokenizer, tensors: [{name, dtype, shape, offset, nbytes, packed}] }
[pad to 64]
[payload] little-endian blobs (f32 or packed u8)
```

Tied embedding is stored once. Packed ternary tensors are stored packed
and unpacked to int8 on load.

## Telemetry

Every log step / generation:

| Metric          | Source                                                         |
| --------------- | -------------------------------------------------------------- |
| RSS / footprint | `task_info(mach_task_self(), TASK_VM_INFO)` → `phys_footprint` |
| tok/s           | `n_tokens / wall_ns` (`Instant`)                               |
| ternary entropy | frac `{-1, 0, +1}` over KAN codes                              |

## Memory budget (M1 8 GB, defaults)

`d=32`, `L=3`, `G=12`, `V=4096`, `T=96`, `B=4`, SGD (no Adam states).
Peak training footprint is designed to stay in the low tens of MB of
unified memory. JSONL I/O is independent of corpus size. Thinking scratch
is ephemeral: after each turn the ring is wiped, so `xhigh` resonance
cannot grow the 35 MB inference envelope.
