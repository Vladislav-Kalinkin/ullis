# Ullis — Rust Core Design Matrix

Standalone ternary Kolmogorov–Arnold engine. Python/PyTorch is gone.
Math is a faithful port of Ullis AI Engine v2.0 plus Mixture-of-Bumps.

## v0.9.0 Infinite Lexicon & Persistent Sessions

1. **Expandable vocabulary.** Production `V ≥ 8192` (`MIN_VOCAB`). Runtime
   `--vocab-size` scales the id plane up to `MAX_VOCAB = 1_048_576` (131 072+).
   Vocabularies below `MIN_VOCAB` are rejected. Pair-merges stop when count `< 2`, so the
   tail of a large `V` stays empty rather than fabricating noise pieces.
2. **Block-sparse packed-i8 embeddings.** `PackedI8Matrix` stores live blocks
   of 64 rows with SIMD-aligned stride. Empty vocab tail is unmapped: lookup
   and tied logits treat those rows as exact zeros, so scaling `V` does not
   densify RSS or rewrite live rows (`grow_rows`).
3. **Persistent sessions.** `--context-len` caps `SovereignFlashBuffer`. REPL
   macros `/save` `/load` `/delete` `/rename` serialize the token ring plus
   dialogue (never thinking) to `sessions/<name>.ullissnap` (`ULISSN01`) and
   re-bind the plane as a Shared Metal buffer.
4. **Fused gradient checkpointing.** Metal is compiled with
   `#define ULLIS_FUSED_GRAD_CKPT 1`. Forward keeps only layer-boundary
   `x^{(ℓ)}`; backward re-dispatches `ullis_mob_kan_fused_step` to rematerialize
   interiors. Gradients are identical; peak activation RAM drops ~50%.

Cognitive-bench schema and 15 golden anchors: `data/cognitive-bench.jsonl`.

## v0.8.0 Deep Context & Vocabulary Expansion

V doubles to **8192** without growing the 40 MB train / 15 MB infer envelopes:

1. **Byte-fallback WordPiece.** Specials `0..3`, raw UTF-8 bytes `4..259`, then
   language-agnostic WordPiece atoms (English / Russian function words and
   syntax identifiers occupy single ids). Unmapped UTF-8 falls back to byte
   ids in-stream and never panics. Encode is greedy longest-match.
2. **Packed 8-bit tied embeddings.** The `[V, D]` plane is stored as int8 codes
   plus a per-row scale. `TernaryKanLinear` stays 2-bit ternary. The Metal
   kernels `ullis_i8_embed_lookup` / `ullis_i8_tied_logits` unpack `char` weights
   in-shader. Training CE is streamed per token so `[n, V]` logits are never
   resident (`V=8192` would otherwise double the tied-head working set).
3. **`SovereignFlashBuffer`.** The `VecDeque` token ring is gone. A page-aligned
   (16 KiB) compacting primitive holds 32 768 ids; `bind_metal` wraps the host
   pointer with `new_buffer_with_bytes_no_copy` (metal-rs 0.29).

Target: **< 40 MB RSS** train (`d=32 L=3 G≤12 V=8192 T=96 B=4`), **< 15 MB**
packed inference (last-token i8 logits only).

## v0.7.0 Cognitive Core

Training was underfitting at CE `6.7–7.1` under the rigid `G = 4 → 8 → 12`
uniform warp and unstructured softmax mixing. Stage 3 replaces that
optimization plane with three language-agnostic mechanisms:

1. **Continuous non-uniform knot insertion.** Every `N` steps (default 50)
   in phases 1–2, each KAN layer inserts one knot in the highest
   residual-energy gap. Spline coefficients are least-squares lifted with
   the existing Gauss–Jordan solver (`gauss::project_spline_coeffs`). Grid
   is frozen at QAT. Metal cap remains `G ≤ 16`.
2. **Adaptive ReLU-bump topology.** Centers stay ordered; per-knot widths
   are `w_0 = c_1−c_0`, `w_{G−1} = c_{G−1}−c_{G−2}`,
   `w_g = ½(c_{g+1}−c_{g−1})`. The fused Metal/CPU kernel reads
   `inv_widths[g]` (buffer 11). EMA `|∂L/∂c_g|` and per-edge spline-grad
   variance site the next insert on high-curvature regions and leave
   linear fields sparse.
3. **Logit entropy penalty.** Masked CE is
   `L = CE + λ_H H[softmax(z)] + λ_R H[softmax(r)]`
   with `H[p] = −Σ p log p`, `∂H/∂z_k = −p_k(log p_k + H)`.
   No language tags. Polarizes vocab mass and the K=3 MoB router.

Memory envelope is unchanged: SGD, no Adam, fused forward, host tape.
Target remains **< 40 MB RSS** during default training (`d=32 L=3 G≤12
V=8192 T=96 B=4`).

## v0.6.0 Sovereign Core

Candle is no longer the compute substrate. The execution stack is two
explicit backends with no AMX assembly and no high-level tensor graph:

| Path | Module | Role |
| ---- | ------ | ---- |
| GPU  | `device` | `MTLDevice` + `MTLCommandQueue`; runtime-compiled MSL |
| Host | `accelerate` | FFI to `Accelerate.framework` (BLAS/vDSP → NEON / SME) |
| Value | `tensor` | `SovereignTensor` = `Vec<f32>` + Shared `MTLBuffer` |

`unsafe` is confined to `device.rs` (Metal `contents()` mapping) and
`accelerate.rs` (`extern "C"` Accelerate). The crate lint remains
`unsafe_code = "deny"`.

### Unified fused kernel

`kernel void ullis_mob_kan_fused_step` is compiled from a source string
at `SovereignDevice::open`. One threadgroup per token. Threadgroup
scratch holds `x[in]`, `ψ[in·G]`, and `softmax` gates. There is no
device allocation for bumps, router logits, or expert stacks.

Forward in one encoder:

```
ψ_g(x_i) = relu(1 − |x_i − c_g| · inv_width_g)²
g        = softmax(x W_rᵀ)                         // K=3, skipped if coarse
Q(w)     = TWN_{δ=0.7 mean|row|}(w)  if phase≥3     // packed: w already ∈ {-1,0,+1}
y_o      = ⟨x, Q(W_base_o)⟩
         + ⟨ψ_{0..g_use}, Q(W_shared_o)⟩
         + Σ_k g_k ⟨ψ_{gs..}, Q(W_routed_{k,o})⟩
```

TWN is **per output row** (and per expert-row for routed weights), matching
`scale_base` / `scale_shared` / `scale_routed[K, out]`. STE in this kernel is
the discrete forward; the hardtanh identity for backward is applied on the
host optimizer until the train loop is fully ported.

Caps (threadgroup 32 KB): `in ≤ 256`, `G ≤ 16`, `K ≤ 4`. Defaults
(`d=32`, `G=12`, `K=3`, `B=4`, `T=96`) sit at ~2 KB scratch per token.
Per-knot `inv_widths` is a length-`G` Shared buffer (buffer 11); it does
not grow threadgroup scratch.

### Memory envelope (target 40–60 MB train / < 35 MB infer)

Apple Silicon unified memory, `MTLResourceStorageModeShared`. Host `Vec<f32>`
is the Accelerate source of truth; the Metal buffer is the GPU view of the
same working set, copied at encode/readback boundaries. Fusion eliminates
the `[N, in, G]` bump tensor and the `[N, K, out]` expert stack from the
resident set. SGD (no Adam moments). Token ring 32 768 ids.

### Host fallback

`cblas_sgemm` / `vDSP_*` / `vvexpf` — never handwritten AMX. On M4/M5 the
Accelerate runtime dispatches SME; on M1–M3, NEON. `MobKanSpec` is `repr(C)`
and is the same bytes the kernel reads as `constant MobKanSpec&`.

### Autograd

Candle is gone. Forward is fused (Metal or Accelerate). Backward is an
explicit host tape: RMSNorm, causal shift, MoB-KAN (STE hardtanh gate),
tied logits, masked CE + Shannon entropy penalty. SGD+momentum clips global grad norm and stores
velocities as detached `Vec<f32>` (no graph).

---

## Crate map

| Module       | Role                                                                                |
| ------------ | ----------------------------------------------------------------------------------- |
| `device`     | `SovereignDevice`: MTLDevice/queue, fused MSL compile, Shared buffer map            |
| `tensor`     | `SovereignTensor`: host `Vec<f32>` + isolated `MTLBuffer`, pipeline ownership       |
| `accelerate` | Accelerate FFI (`cblas_sgemm`, vDSP, `vvexpf`) + CPU fused MoB-KAN                  |
| `quant`      | TWN 2-bit pack/unpack plus per-row packed-i8 (`PackedI8Matrix`)                     |
| `gauss`      | G×G Gauss–Jordan (`matmul`/`broadcast` only, G ≤ 16)                                |
| `kan`        | `TernaryKanLinear` + non-uniform ReLU-bumps + MoB router + knot insert               |
| `mixers`     | `CausalShift` (0 params) / tiny causal attention / streamed tied CE                 |
| `model`      | `UllisKan`: packed-i8 embed → L × (shift + KAN) → RMSNorm → tied i8 logits          |
| `tokenizer`  | Byte-fallback WordPiece, vocab 8192, language-agnostic atoms                        |
| `data`       | 4-key JSONL via `serde_json`, `SovereignFlashBuffer` token ring                     |
| `think`      | `--thinking` budgets, ephemeral reasoning GC, dialogue cache                        |
| `train`      | 4-phase QAT, continuous knot insert, SGD+momentum, CE+entropy on thinking+output    |
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
ψ_g(x)    = relu(1 − |x − c_g| / w_g)²
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
- **Routed bumps** have `K = 3` expert coefficient slices. Polarization is
  driven by the router entropy penalty, not by language labels.
- Per-token micro-router: `softmax(x W_rᵀ)`, `W_r ∈ R^{K × in}` (tiny, FP32).
- Routed contribution is one batched GEMM: `[N, in·G_r] × [K·out, in·G_r]`,
  then `Σ_k g_k · y_k`. No language flags; the prompt tokens are the signal.

Disable with `--no-moe` to recover the exact v2 edge function.

### Dynamic non-uniform grid

Training starts at `G = 4`. Every `knot_insert_every` steps (default 50)
during warmup and sparsify the layer inserts **one** knot at the midpoint
of the gap that maximises
`(EMA|∂L/∂c_g| + EMA|∂L/∂c_{g+1}|) · (c_{g+1}−c_g) · mean(edge_var)`.
Coefficients are least-squares lifted onto the new (non-uniform) knot
vector: sample `M = max(64, 16G)` points, form
`Ψ_newᵀ Ψ_new b' = Ψ_newᵀ Ψ_old b` with per-knot bump widths.
The G×G solve is Gauss–Jordan with ridge `1e-6 · mean(diag)`. After each
SGD step, centers are projected back onto a strictly ordered chain and
widths are rebuilt from spacing. Frozen from QAT onward. Uniform
`extend_grid(G)` remains for smoke / legacy jumps.

### Entropy-regularized masked CE

Loss on the thinking+output span:

```
L = mean_mask[ −log p_{y} + λ_H H(p) ] + λ_R mean H(g) + λ_1 ‖w‖_1
H(p) = −Σ_j p_j log p_j
∂H/∂z_k = −p_k (log p_k + H)
```

Defaults `λ_H = 0.03`, `λ_R = 0.05`. This is not a language classifier:
high-entropy vocab rows and indecisive MoB gates are penalized, so the
router polarizes without hardcoded syntax or language flags.

### 4-phase ternary QAT

| Phase | Name     | Weights                                    | Grid   | LR     |
| ----- | -------- | ------------------------------------------ | ------ | ------ |
| 1     | warmup   | FP `a`,`b`,centers                         | 4 → …  | `3e-3` |
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
the reasoning trajectory, plus the entropy penalty above.
Token ring is a page-aligned `SovereignFlashBuffer` capped at 32 768 ids (~128 KB).

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

`d=32`, `L=3`, `G=12`, `V=8192`, `T=96`, `B=4`, SGD (no Adam states).
Peak training footprint is designed to stay in the low tens of MB of
unified memory. JSONL I/O is independent of corpus size. Thinking scratch
is ephemeral: after each turn the ring is wiped, so `xhigh` resonance
cannot grow the 35 MB inference envelope.
