# Ullis Memory — experimental successor to Mixture-of-Bumps KAN

Status: research draft, **post-critic patch**. Production engine remains
`UllisKan` (`DESIGN.md`). This is the architecture behind `--arch memory`.
Kill it if the capability suite fails.

It is not a new acronym. The engine is still Ullis. The layer is a
**memory block**. The trainable maps are **ternary experts**.

Do not implement `src/slots.rs` or the train loop against an older draft.
Slot algebra in §6 is blocking.

---

## 1. Why the current engine hits a wall

The 8× width jump (`d=32 → d=256`, `L=6`, `V=8192`) is not a mysterious
KAN curse. Three independent ceilings stack.

### 1.1 Dense `W_base` is still Θ(D²)

Shared-edge factorization already made the spline cheap:

```
φ_i = Σ_g Q(W_shared[i,g]) ψ_g(x_i)          # Θ(N · D · G)
ρ_{k,i} = Σ_g Q(W_routed[k,i,g]) ψ(x_i)      # Θ(N · K · D · G_r)
y = Q(W_base) (x + φ + Σ_k g_k ρ_k)          # Θ(N · D · D)
```

Confirmed in code: `weight_base` is `[out, in]` (`kan.rs`); Metal and
CPU both apply `W_base` to the spline-modulated vector. `W_shared` is
`[D, G]`. Growing `d-model` grows the only term that matters. 32 → 256
is a **64×** GEMM, before backward.

Mixture-of-Bumps does not buy sparse compute: the production path is
dense in the three experts (`--moe-topk` default 0). When top-k is on,
Metal still computes `ρ` for every expert and only zeros gates.
Adding bumps adds `Θ(N D G)` and threadgroup pressure, not capacity
where it counts.

### 1.2 Metal is a per-token inner product; host already uses BLAS

`ullis_mob_kan_fused_step` launches **one threadgroup per token**.
Scratch is `x[TIN] + ψ[TIN·G] + gates[K]` with `TIN ≤ 256`. The hot loop
is a scalar `acc += u[i] * W_base[o, i]` over in-tiles and out-tiles.
QAT recomputes per-row absmean (`row_delta`) on every forward. Backward
atomics-CAS on `float` gradients.

The **host** path already `sgemm_nt`s `W_base` tiles
(`mob_kan_fused_cpu_shared_edge`). Accelerate is not unused. The wall
at `D=256` on the GPU train path is Metal inner product + atomics +
rematerialize **plus** tied-head CE (`streamed_tied_ce_acc` does several
`[n, V]×[V, D]` passes). At `B=4, T=96, V=8192, D=256` the head is
order-of-magnitude above a 65k/token layer GEMM. Tok/s is a composite
sensor, not a KAN-only sensor.

Throughput 4k–12k → 400 → 150 tok/s is the expected shape of `O(D²)`
on Metal + head + tiling, not evidence that "KAN cannot scale."

Existing `--mixer attn` is **not** a control: `KanBlock::backward` for
`Mixer::Attn` copies `dy` and does not train QKV/proj.

### 1.3 Causal shift cannot hold a thought

Default mixer is `causal_shift`: half the channels delayed by one token,
explicitly `[b, t, c]` (`mixers.rs`). Stacked `L` times the receptive
field is **≈ L tokens**. At `L=6` the model cannot copy a name from 20
tokens ago, cannot close a bracket opened at the start of a function,
cannot do induction.

The fast `d=32` run is fast because it does a tiny dense map on a mixer
that does not implement language circuits. Speeding KAN up would not
create intelligence. The mixer is the intelligence bottleneck; `W_base`
(on Metal) is the speed bottleneck. They were mistaken for one problem.

### 1.4 What is actually worth keeping

| Keep | Why |
| ---- | --- |
| Ternary STE / TWN `δ = 0.7 · mean\|row\|` | BitNet b1.58: native `{-1,0,+1}` matches FP16 at scale when trained from scratch |
| Packed 2-bit disk / i8 embed | Envelope is real. Do not TWN the vocab table. |
| Tied logits, streamed CE | `V=8192` must never materialize `[N,V]` |
| RMSNorm, residual stream | The only proven "scratchpad" |
| 4-key JSONL, thinking-masked CE | The objective is fine |
| SGD+momentum, no Adam | Memory envelope |
| Univariate ReLU-bumps | Cheap nonlinearity **inside experts**, never on residual edges |
| MoE *idea* | Params without *active* FLOPs — only with real top-k skip + grouping |

| Drop | Why |
| ---- | --- |
| Edge-KAN `W_base ∈ R^{D×D}` as the mixer | Θ(D²) is the wall |
| Dynamic knot insert + Gauss–Jordan lift | Cute; not where loss comes from |
| Causal-shift-only sequence mix | RF = L |
| Per-token naive Metal GEMM | BLAS for remaining maps; fuse only after a profiler |
| Ternarizing gates / router / state | MatMul-free LM: ternary *attention/state* fails to converge |
| Scaling the model by widening D first | Widening D widens FLOPs *and* the tied head |

---

## 2. Design rules (non-negotiable)

1. **Ternary weights, floating state.** Residual stream, recurrent
   state, slot memory, RMS scales, and the router stay **FP32 in RAM**.
   Only expert maps are ternary (embeddings stay packed-i8). Control
   plane is never quantized. Expert *storage* may be `--master fp16`;
   *compute* is FP32 unpack, matching the current crate. There is no
   FP16 working copy of the residual.

2. **Active expert mul-adds do not grow with `E`.** Parameters live in
   experts. Arithmetic of *fired* experts is `k · 3 · D · W`. Adding
   experts must not run their GEMMs. Wall-clock is **not** a theorem:
   router is `Θ(E D)`, SGD walks `Θ(E D W)` bytes, BLAS launch cost
   depends on grouping. Measure fwd+bwd layer time, not tok/s.

3. **Sequence mix is linear in T.** No `T×T` scores. State is the
   diagonal scan plus `S` slots. Optional local window later, not v1.
   All T-mixers are shaped `[B, T, D]`, never a flat `N = B·T` stream.

4. **Cheap mix, learned univariate maps.** Fast Walsh–Hadamard is a
   frozen orthogonal linear mix (QuaRot-style, good for later TWN),
   not a learned Kolmogorov inner function. Experts apply bumps in
   `W`-space after `W_up`. Do not cite KA as evidence the model learns.

5. **Capability before corpus poetry.** Synthetic circuits (copy, bind,
   brackets) must be learned in FP32 *and* after ternarization, on a
   **closed alphabet**, held-out strings, **autoregressive** decode.
   Only then is `ultra.jsonl` allowed to be a language claim.

6. **Reuse the crate.** Tokenizer, JSONL, quant pack, SGD, telemetry,
   checkpoint stay. New code is a model class, not a rewrite.
   `apply_topk_gates` / `switch_aux` are hard-capped at `k≤4` and
   `[f32; 4]` — they cannot be reused for `E=64`. Write a general
   router.

---

## 3. Complexity knobs

Three independent axes. Today Ullis has one (`d-model`) that moves
params, FLOPs, Metal scratch, *and* the tied head together.

| Knob | What it is | Params | Active mul-adds / token / layer (fwd) |
| ---- | ---------- | ------ | ------------------------------------- |
| `d_model` D | residual width | linear in D (norms, gates, slots) | FWHT `D log D` + scan `D` + slots `S D` + experts `3 k D W` |
| `n_experts` E | capacity | **Θ(E · 3 · D · W)** | **independent of E if GEMMs are skipped**; router `E D` is not |
| `expert_width` W | expert inner dim | Θ(E · 3 · D · W) | Θ(3 k D W + k W G) |
| `top_k` k | active experts | — | linear in k |
| `n_slots` S | associative memory | Θ(S D) per slot layer | Θ(S D) |
| `n_layers` L | depth | linear | linear |

Expert maps per expert: `W_up [W,D]`, `W_gate [W,D]`, `W_down [D,W]`,
bumps `b [W, G=4]`. That is **three** GEMMs, not two.

Train cost of those GEMMs ≈ **3× forward** (fwd + dW + dX) **iff**
inactive experts are not touched. Plus router bwd `Θ(N E D)`, reverse
scan `Θ(N D)`, slot BPTT `Θ(N S D)`, SGD traffic `Θ(E D W)`, and the
tied head `passes · N V D` which **dominates** at `V=8192`.

Iso-flop examples (k=2, W=64, G_bump=4), **experts only**:

| Name | D | E | L | Expert params | Fwd mul-adds / tok / layer (experts + router) |
| ---- | - | - | - | ------------- | --------------------------------------------- |
| tiny | 64 | 4 | 4 | `4·4·3·64·64 ≈ 201k` | `3·2·64·64 + 4·64 ≈ 25k` |
| mid | 128 | 16 | 6 | `6·16·3·128·64 ≈ 2.4M` | `3·2·128·64 + 16·128 ≈ 51k` |
| wide | 256 | 64 | 6 | `6·64·3·256·64 ≈ 18.9M` | `3·2·256·64 + 64·256 ≈ 114k` |

KAN `d=256 L=6`: `W_base` is `L·D² ≈ 0.39M` params and ~65k mul-adds/tok
/layer, plus bumps. Memory-wide has ~50× the *expert* parameters, not
"same or lower arithmetic" once `W_gate`, router, and slots are counted.
The win is: **arithmetic of fired experts stays Θ(k D W) while params
grow with E**, and those GEMMs are `[N_e, W]×[W, D]` after grouping,
not a per-token Metal inner product.

**Operator instruction:** pass C1–C4 at `E=4`, then scale `E` with C7
measured on **layer fwd+bwd ms**. Do not grow `E` into dead experts.
Do not grow `L` first: leaky `α^L` and stacked slot BPTT. Grow `D`
only when the residual (not tok/s) saturates, and remember `D` also
scales the tied head.

---

## 4. One memory block

Activations are **`[B, T, D]`**, FP32. `N = B·T` is a packing of rows
for GEMMs only after the sequence mix, never the scan/slot axis.

```
u  = RMSNorm(x)                         # FP32 scale, D params
u  = FWHT(u)                            # parameter-free, §4.1
h  = gated_diagonal_scan(u)             # FP32 gates, [B,T,D]
x  = x + h                              # residual A

v  = RMSNorm(x)
r  = softmax(W_router v)                # FP32, [E, D]
e* = top-k(r, k)                        # k ∈ {1,2}

z  = 0
grouped tokens by expert:               # §5.1, mandatory
    z += r_e · Expert_e(v)              # ternary maps, §5
x  = x + z                              # residual B

if layer % 2 == 1:
    x = x + SlotMemory(RMSNorm(x))      # every other layer, §6
```

No dense `D×D` map. The widest learned expert map is `[W, D]` with
`W = 64` unless C1–C4 force `W = 128` after QAT.

### 4.1 Fast Walsh–Hadamard

Orthogonal mix, zero parameters, Θ(D log D). Used as a QuaRot-style
preconditioner so subsequent ternary maps see rotated coordinates —
not as a Kolmogorov inner spline.

- Pad D to the next power of two, transform, **unpad**. Pad channels
  must not enter the residual.
- Normalize by `1/√D_pad` so the map is orthonormal.
- FWHT is an involution up to that scale: **backward is the same
  transform**.
- Host iterative butterfly. No Metal until a profiler says the expert
  GEMM is not the peak.

### 4.2 Gated diagonal scan (time mix)

Per channel, causal, linear in T, **independent across batch**:

```
α_t = σ(w_α ⊙ u_t + b_α)            # FP32, vectors in R^D
i_t = τ(w_i ⊙ u_t + b_i)            # τ = SiLU
h_t = α_t ⊙ h_{t-1} + i_t ⊙ u_t             # leaky sum / counter; EMA cannot count depth
```

Init: set `b_α` so `α ≈ 0.95` at `u=0` (`b_α = logit(0.95)`). Clamp
`α ∈ [ε, 1−ε]`, `ε = 1e-3`. Unspecified `σ(0)=0.5` is a one-token
half-life and will fail long-range copy.

Plus a **group delay**: split D into groups of 32; delay one group by
1 token (old causal-shift, after Hadamard). Exact token copy is **not**
this scan's job; it is the slots'. The scan is a leaky integrator /
counter (brackets).

P1 tests (all required):

1. Numerical Jacobian on `T=8, D=4` (correctness).
2. `|∂h_T / ∂h_0|` at `T=64` with default init: mean magnitude must
   stay `> 1e-3` (memory, not a correct-but-dead Jacobian).
3. Batch isolation: two sequences in one batch do not mix state.

Inference: carry `h ∈ R^{B×D}` per scan layer. `UllisKan::next_token`
re-encodes the full prefix; a memory model that does the same is
`O(T²)` generation. Incremental cache is part of P4, not an afterthought.
`seq_len=96` is the universe for C1–C4. Slots do not span
`context_len=32768` unless we later add a persistent `M` across windows
(out of scope for v1).

---

## 5. Ternary expert (outer map)

Expert `e` maps `v ∈ R^D → R^D`. Bumps live in **W-space after `W_up`**,
never on residual edges, G=4 fixed, centers shared across experts and
**frozen after warmup**. No knot insertion.

```
p     = Q(W_up_e) v                     # W_up [W, D], ternary + row scale
ψ     = ReLU-bumps(p; G=4, centers in [-2,2], frozen after warmup)
g     = σ(Q(W_gate_e) v)                # SwiGLU-style, ternary
h     = (p + Σ_g b_{e,·,g} ψ_g) ⊙ g     # b is [W, G], ternary
out   = Q(W_down_e) h                   # W_down [D, W], ternary
```

Ternary rule (existing TWN):

```
δ_row = 0.7 · mean(|W_row|)
Q(w)  = { +1 if w>δ; −1 if w<−δ; 0 else } · γ_row
γ_row = mean(|W_row|)
```

STE: hardtanh `|w|≤1`, shadow weights FP32 compute / optional fp16
master storage. Router is FP32. Load-balance: Switch-style
`α · E · Σ f_i P_i` with `α=0.01`. Start `E=4`. `E=64` on 46k lines
will collapse without aux **and** without C1 already passing.

### 5.1 Dispatch (mandatory, or C7 is a lie)

`for e in e*: z += r_e · Expert_e(v)` as a per-token loop of
`[1, D]×[D, W]` GEMMs will lose to the current fused kernel.

Required algorithm:

1. Router + top-k over all `N = B·T` rows.
2. Count tokens per expert; build an index list (dropless).
3. For each expert with `n_e > 0`, gather rows → `sgemm` `[n_e, D]×[D, W]`
   (and the other two maps) → scatter-add `r_e * out` into `z`.
4. Experts with `n_e = 0` **do not** load weights or run GEMM.

At `N=384, k=2, E=64` you get ~12 tokens/expert: dispatch-bound on
Accelerate, not FLOP-bound. Absolute tok/s may still fall below
`d=32` KAN. That is acceptable if C7's *relative* layer-ms stays
flat-ish and C1–C4 pass. Profile tiny GEMM vs a fused expert kernel
before writing Metal (P7).

Reuse of `apply_topk_gates` / `switch_aux` is forbidden until they
are generalized past `k≤4`.

---

## 6. Slot memory (associative, every other layer) — blocking algebra

State `M ∈ R^{B × S × D}`, S ∈ {16, 32, 64}, **FP32**, independent
across batch, **reset per sequence** (not "at batch boundary").

Content-based read and write. Query is in `R^D`. A map `W_q: D → S`
is **not** content attention.

```
q_r   = v                               # or q_r = w_q ⊙ v, diagonal FP32
s     = softmax(M q_r / √D)             # s ∈ R^S, over slots, not T
read  = sᵀ M                            # Θ(S D), in R^D

q_w   = w_w ⊙ v                         # dedicated write query, diagonal FP32
k     = softmax(M q_w / √D)             # content-based: look at M
β     = σ(b_β + w_β · mean(v))          # scalar in (0,1) per token
# GRU-per-slot write (additive, not a convex replace that ignores M):
M     ← M + β · (k ⊗ (v − (kᵀ M)))     # erase-then-write on addressed slots

out   = γ · read                        # γ scalar FP32, init 0.1
                                        # no W_o ∈ R^{D×D}
```

`β` is a **scalar** (or `R^S` after a tiny FP32 `[S]` map). It is not
`σ(w_β ⊙ v) ∈ R^D` with an unexplained `β̄`.

Why this can bind (C3): at `name`, content-hash lands in some slot
(or empty slot via small `M` init); at `VAL`, `q_w` is still near the
name residual if the skip kept it, so `k` finds the same slot and the
GRU writes `VAL` into it; at recall, `q_r ≈ name` reads that slot.
A write `k = softmax(W_k v)` that **never reads `M`** hashes `name`
and `VAL` to different bins and cannot associate. That algebra is
rejected.

**BPTT:** full backprop through writes. Tape `M_t` (or rematerialize
from the sequence). **No stop-gradient on `M` in phase 0.** At
`B=4, T=96, S=32, D=256` the tape is ~12 MB per slot layer — fine.
If writes are stop-grad, the write gate never sees future recall and
C3 is unlearnable; §11 must not misread that as "ternary is the wrong
costume."

Inference: carry `M ∈ R^{B×S×D}` per slot layer, same as `h`.

P2 done-when: a **1-layer** FP32 net (embed + one slot memory + tied
head, no experts) solves C3 at S=16, alphabet 32, 4 distractor pairs,
held-out names, AR decode. If that fails, stop. Do not start P3.

---

## 7. Training recipe

Same 4-key JSONL, same masked CE + entropy on thinking+output — **after**
phase 0 on the synthetic suite.

| Phase | Weights | LR | Purpose |
| ----- | ------- | -- | ------- |
| 0 smoke | all FP32 | 3e-3 | C1, C3, C4 must drop to AR exact-match thresholds |
| 1 warmup | all FP32 | 3e-3 | real corpus, router aux on |
| 2 sparsify | L1 on expert maps | 3e-3 | push mass off the TWN dead zone |
| 3 qat | STE ternary experts; control plane FP32 | 1e-3 | BitNet-style |
| 4 harden | freeze expert codes; train scales, RMS, router, scan, slots | 3e-4 | |

Do **not** ternarize in phase 1. Small models need a long FP warmup.
Phase 0 is the kill-switch: if FP32 cannot copy and bind, ternary will
not invent a circuit. C1–C4 are re-measured after phases 3 and 4 on
the **same held-out synthetic strings**.

Dtypes: control plane FP32 in RAM. Expert master `--master fp16`
storage, FP32 compute. Momentum `--mom q8` on expert velocity only
(scan/router/slots are O(D) / O(S D); keep their velocity FP32).

---

## 8. Capability suite (merge gate)

Own tokenizer: **closed alphabet**, `V_task ≤ 64` (specials + digits +
a handful of names). Do **not** use production WordPiece `V=8192`.
Disable `ban_unigram_run` for the suite. Train/eval split: held-out
**random** strings, not the train JSONL. Metric: **autoregressive**
exact match (feed predicted token back), not teacher-forced CE.

`seq_len` ≥ task length. Generation must use the incremental `h`/`M`
cache.

| Id | Task | Pass criterion | Notes |
| -- | ---- | -------------- | ----- |
| C1 | Identity copy | AR exact ≥ 95% on held-out | **max length ≤ S**. If default S=32, copy ≤ 32. Length-64 requires S≥64. |
| C2 | Reverse, len 16 | stretch probe, **not a merge gate** | Bag-of-slots has no position. Keep as a probe; pass not required to proceed. |
| C3 | Bind `name=VAL … recall name` | AR exact VAL ≥ 90% | ≥ 4 name-value pairs + distractors in one sequence. Only valid after §6. |
| C4 | Bracket depth ≤ 8 | AR exact ≥ 90% | Plausible on the scan counter alone. |
| C5 | Shuffled-label control on a **tiny-V** slice, not as the first claim on `ultra.jsonl` | real CE < shuffle CE − 0.3 nats | Language-use veto after circuits pass. |
| C6 | Ternary histogram | sanity only, **not a merge gate** | TWN can land in (0.4, 1.5) bits while computing garbage. |
| C7 | Iso-expert scaling | `last_fwd_ms + last_bwd_ms` within 25% for E=4→16 | Fixed D, W, k, **V≤256** (or subtract `last_ce_ms`). Token grouping on. Wall tok/s is **not** the gate. |
| C8 | Width scaling | layer fwd+bwd ms drop ≤ 6× for D=64→256 | Same: exclude or equalize head cost. A 4× from CE alone is expected at V=8192. |

C7/C8 exist to catch a hidden dense map. They can pass while experts
got slower if you measure tok/s at `V=8192`. Use the timers already
logged by `train_step` (`last_fwd_ms`, `last_ce_ms`, `last_bwd_ms`).

---

## 9. What we will implement (and in what order)

Keep `UllisKan` working. Add `--arch memory`. **Do not start P3–P7
until P2 solves C3.**

| Step | Module | Done when |
| ---- | ------ | --------- |
| P0 | `src/hadamard.rs` FWHT + inverse | orthonormal `1/√D`, pad/unpad, bwd = fwd |
| P1 | `src/scan.rs` gated diagonal + group delay + bwd | Jacobian T=8; `\|∂h_T/∂h_0\|` at T=64; batch isolation |
| P2 | `src/slots.rs` read/write + **full BPTT** | 1-layer FP32 net solves C3 (see §6) |
| P3 | `src/expert.rs` ternary expert + bumps + **grouped** sgemm | STE matches `quant`; n_e=0 skips GEMM |
| P4 | `src/memory.rs` block + `UllisMemory` + **state cache** for generate | train_step; streamed CE reused; `next_token` is O(1) per new token |
| P5 | CLI `--arch memory --experts --expert-width --top-k --slots` | `ullis train` runs |
| P6 | Capability suite as tests with **task vocab** | C1, C3, C4, C7, C8 automated; not production tokenizer |
| P7 | Metal **only if** Accelerate tiny-GEMM is the profiler peak | no per-token kernel by default |

Checkpoint: magic `ULLIS04` or header `arch: "memory"`. Do not load
KAN weights into memory blocks.

---

## 10. Risks

1. **Diagonal scan is a leaky integrator.** Without slots, long-range
   copy dies. Slots are not optional for C3. Init `α≈0.95` or they
   die even with slots.
2. **Tiny ternary experts underfit.** BitNet quality appears at
   hundreds of millions of params and 1e11 tokens. Wide is ~19M
   ternary weights. We may need W=128 and a longer FP warmup, or
   accept ternary as an inference costume after a capable FP32
   model (phase 4 only). If phase 3 destroys C1–C4, widen W before
   abandoning the block.
3. **Expert collapse on 46k lines.** Aux loss and E=4 until C1 works.
4. **Hadamard + residual can cancel.** If training is unstable, drop
   FWHT before dropping slots or experts.
5. **Small-GEMM BLAS tax.** `[12, 256]×[256, 64]` is launch-bound.
   Absolute tok/s can fall below `d=32` KAN even if C8's relative
   bound holds.
6. **CE-dominated tok/s.** Never use wall tok/s at `V=8192` as evidence
   the mixer scaled.
7. **Local attention window 64** is the escape hatch if slots fail
   C3 in FP32, not a v1 feature.
8. **Honesty about data.** Architecture enables circuits; 900 MB of
   JSONL does not make a general LLM. Claims stay at "engine" level.
9. **Existing attn mixer has no backward.** Do not A/B against it.
10. **Addressing must stay in the FP control plane.** If C1–C4 use
    expert identity as memory, phase 3 will erase them.

---

## 11. Decision after the suite

- C1/C3 fail in FP32 at P2 → **stop**. Do not train `ultra.jsonl`.
- C1–C4 pass, C7 fails on layer-ms → hidden dense map or grouping
  bug; hunt it.
- C1–C4 pass in FP32, fail after QAT, W already 128 → ternary at this
  scale is the wrong costume; ship FP32 memory block, pack for disk
  only.
- C7+C8 pass, C1–C4 fail after QAT with W=64 → keep the block, delay
  ternary or widen W. Do not read a legal C6 histogram as success.
- Fluency on `ultra.jsonl` without C1–C5 → failed experiment.

---

## 12. Critic changelog (2026-08-21)

Hostile review of the first draft. Verdict: **proceed-with-changes**.
All blocking items from that review are in this file:

- Slot algebra typed and content-based; GRU write; BPTT; `M` is
  `[B,S,D]`; per-sequence reset.
- C1 length ≤ S; closed alphabet; AR; held-out.
- C2 demoted to a probe (no position channel).
- C7/C8 use layer timers, tiny V, not wall tok/s.
- FLOP table counts three expert GEMMs + router; train ≈ 3×.
- Dispatch grouping specified; no reuse of `k≤4` aux kernels.
- Dtypes match the crate (FP32 compute, optional fp16 master).
- Incremental `h`/`M` cache; Kolmogorov slogan hedged.
- Metal vs host BLAS distinction in §1.2.
- `switch_aux` / attn-backward landmines documented.

Fine as-is from the first draft: §1.1 `W_base` Θ(D²), §1.3 RF ≈ L,
rule 1 (ternary weights / floating state), rule 5 (circuits first),
tied i8 embed, streamed CE, SGD, RMSNorm, 4-key JSONL, FP router,
frozen bump centers, P7 after profiler, `ULLIS04`, aux `α=0.01`,
start E=4, do not load KAN weights into memory blocks.
