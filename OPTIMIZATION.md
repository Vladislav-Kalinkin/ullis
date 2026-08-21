# Ullis Memory & Training-Scale Optimization

| Field | Value |
| ----- | ----- |
| **Document** | Working plan (temporary; intended to be copied to repo-root `OPTIMIZATION.md`) |
| **Author** | Ullis / systems |
| **Date** | 2026-08-21 |
| **Status** | Approved |
| **Scope** | Training RAM + MoB-KAN scale (P0 identical-grad → P1 1B envelope → P2 quality) |
| **Repo** | `/Users/vladislavkalinkin/ullis` v0.9.0 — Rust ternary Mixture-of-Bumps KAN, Metal + Accelerate, no PyTorch/Candle |
| **Note** | Этот файл — **временный** рабочий план. Канон архитектуры остаётся `DESIGN.md`. После реализации P0–P1 либо влить итог в `DESIGN.md`, либо удалить. |

---

## Overview

Тренировка дефолтной модели (`d=32`, `L=3`, `G=4→12`, `V=8192`, `T=96`, `B=4`, SGD+momentum, `moe=true`, `K=3`) держит **~90 MB** `phys_footprint`. `UllisKan::param_report` при **G=12** считает **327 720** FP32-весов (**1.25 MB**); в **warmup G=4** — **284 664** (**1.09 MB**). Пользовательский ориентир «~0.5M / 1.3–2 MB» — порядок величины весов, **не** RSS.

**80% весов — `embed` `[V,d]=262 144`, он никогда не `attach`ится** (`UllisKan::new` вызывает только `b.ff.bind`). Dual Shared-копии касаются attached KAN (веса/scales/centers) ≈ **65 576 floats → ~0.25 MB host + ~0.25 MB GPU**. Это **не** рычаг 90→40 MB.

Разница RSS vs веса — реализация **и** неизмеренный Metal/lib baseline (PSO, allocator, tokenizer). Горячий путь: `SgdMomentum::step` клонирует W+g через `trainable_snapshot` / `write_param` (`src/model.rs:803` — **host `to_vec` only**, не GPU); `write_param("embed")` каждый шаг зовёт `refresh_embed_i8()` (`PackedI8Matrix::quantize`); host `TernaryKanLinear::backward` клонирует W/Q и материализует `[n,in,G]`; `streamed_tied_ce` / `embed_scatter` выделяют плотные `[V,d]`; `forward_metal` аллоцирует `xt`/`yt` и `download` после `wait_until_completed`.

Предложение:

- **P0 (preferred, если dual-KAN < 1 MB — измерить):** in-place SGD, defer i8, `TrainWorkspace`, in-place CE. **Не** zero-copy. KPI **после измерения**: `rss < baseline_metal_hello + 12 MB`, плюс раздельный отчёт `params+grad+opt+workspace`. Сырой `rss < 40` из `DESIGN.md` — **гипотеза**, не план, пока empty Metal не снят (`ullis smoke` / open device).
- **P1:** zero-copy `SovereignTensor` (bytes/param на 1B), tiled fused fwd (`d>256`, **обязательный `OUT_TILE`**), fused bwd, packed **storage** 4–6 B/param (не peak working set, пока unpack не ограничен). Честный 1B — **три конверта** ниже, не «6 GB».
- **P2:** `--moe-topk 0|1|2` + load-balance; default-on — **follow-up** после длинного train, не merge PR7.

Честный 1B: **не** в 90 MB. Packed infer ≪ transformer. 0.5M-класс train: **net** working set в низких MB; raw RSS упирается в Metal hello.

---

## Background & Motivation

### Текущий стек (факты из кода)

| Путь | Модуль | Роль |
| ---- | ------ | ---- |
| GPU | `src/device.rs` | `MTLDevice` + `MTLCommandQueue`; runtime MSL `ullis_mob_kan_fused_step` |
| Host | `src/accelerate.rs` | `cblas_sgemm` / vDSP / `vvexpf` → NEON (M1–M3) / SME (M4/M5). **Нет** рукописного AMX |
| Value | `src/tensor.rs` | `SovereignTensor` = host `Vec<f32>` **плюс** isolated Shared `MTLBuffer` |
| Train | `src/train.rs` + `src/optim.rs` | 4-phase QAT, SGD+momentum, CE + `λ_H` + `λ_R` |
| Unsafe | `device.rs`, `accelerate.rs`, плюс `telemetry.rs` (`task_info`) | crate lint `unsafe_code = "deny"` |

Дефолты (`src/config.rs` `TrainConfig::default`): `d_model=32`, `n_layers=3`, `n_basis=4`, `grid_start=4`, `grid_mid=8`, `grid_final=12`, `seq_len=96`, `batch_size=4`, `vocab_size=8192`, `mixer="shift"`, `lr=3e-3`, `momentum=0.9`, **нет Adam**, `moe=true`, `n_experts=N_EXPERTS=3`, `entropy_coef=0.03`, `router_entropy_coef=0.05`, `knot_insert_every=50`, `fused_grad_ckpt=true`.

`split_basis(12, true) → (8, 4)`; `split_basis(4, true) → (3, 1)`.

Train **однопоточен**: один `train_step` / `SgdMomentum::step` на процессе. `TrainWorkspace` не `Sync`; без lock.

### Формула параметров MoB-KAN

Слой `TernaryKanLinear` (`src/kan.rs::new`):

```
W_base    : [out, in]              = d²
W_shared  : [out, in·G_s]          = d² · G_s
W_routed  : [K, out, in·G_r]       = K · d² · G_r
W_router  : [K, in]                = K · d
```

Итого **d² · (1 + G_s + K·G_r) + K·d**. При `G=12`, `G_s=8`, `G_r=4`, `K=3`: **21 d² + 3d** против условных ~**12 d²** transformer block (attn 4d² + FFN 8d² — конвенция, не баг). Это архитектурный налог.

`param_report` **не** включает grads/vel/GPU copies.

| Конфиг | Count | FP32 W |
| ------ | ----- | ------ |
| **G=4 start** (warmup): 3×(`W` 7264 + scales/centers 168 + n1/n2 64) + embed 262144 + norm 32 | **284 664** | **1.09 MB** |
| **G=12** `param_report`: embed 262144 + norms 224 + 3×21600 W + 3×184 scales/centers | **327 720** (~0.33M) | **1.25 MB** |

Embed = **80%** при обоих G. `embed_i8` рядом: ~V·d int8 + V scales ≈ 0.29 MB.

**Bytes/param (сегодня):**

| Тензор | W | g | vel | GPU copy | **Σ** |
| ------ | - | - | --- | -------- | ----- |
| Attached KAN (`bind`) | 4 | 4 | 4 | 4 | **16 B/param** |
| Embed (не attached) | 4 | 4 | 4 | 0 | **12 B/param** |

После zero-copy attached KAN → **12 B/param** (alias). 16 B/param — **не** среднее по модели.

`gauss.rs` collocation `m = (new_g * 16).max(64)` — не hard cap G≤16; реальный cap сетки — `MobKanSpec::MAX_G` (Metal scratch).

### Почему ~90 MB (гипотеза до измерения)

```mermaid
flowchart LR
  subgraph listed [Перечисленные "наши" MB]
    S["snapshot clones ~2.5"]
    D["dembed+scatter ~2.1"]
    B["bumps xt/yt ~0.5-2"]
    G["dual KAN GPU ~0.25"]
    I8["quantize rebuild CPU"]
  end
  subgraph unlisted [Не измерено]
    M["Metal PSO / lib 30-50?"]
    A["debug allocator / tokenizer"]
    C["fused_cpu prepare_weights clones"]
  end
```

Перечисленные temps ≈ **8 MB**. Dual GPU attached KAN ≈ **0.25 MB**. Если baseline Metal 30–50 MB, **P0 listed savings не гарантируют raw rss < 40**. `DESIGN.md` сам расщеплён: «< 40 MB» vs «target 40–60 MB train». Empty-process Metal RSS **не измерен**.

1. **Dual residency (мелочь на дефолте).** `SovereignTensor::attach` (`src/tensor.rs:122`) → `alloc_shared_f32_buffer` + `write_shared_f32_buffer`. Zero-copy `wrap_shared_bytes_no_copy` + `PageSlab` уже есть и используются **только** `SovereignFlashBuffer` (`src/data.rs:356`). **Поля FlashBuffer: `slab` первым, `metal` последним — опасный Drop order** (см. A1).
2. **Snapshot optimizer.** `SgdMomentum::step` (`src/optim.rs:47`) → `trainable_snapshot` (`src/model.rs:803`) клонирует host slices. `write_param("embed")` (`src/model.rs:883`) → `refresh_embed_i8()` каждый шаг. Train CE — FP32 `streamed_tied_ce`. i8 нужен packed infer, не SGD. `quantize` пересобирает block-sparse maps — **CPU**, не обязательно RSS, если старая матрица drop'ается.
3. **Transient `[V,d]`.** `streamed_tied_ce` не материализует logits `[n,V]`, но возвращает плотный `dembed[V,d]`. `embed_scatter` — ещё `[V,d]`. 2 × 1.05 MB temps + постоянный `embed_grad`.
4. **Host backward** (`src/kan.rs:461`): `.to_vec()` W/Q/centers/scales; QAT `ternarize_hard` ещё `Vec`; `relu_bumps` `[n,in,G]` (default n=384,G=12 ≈ 0.59 MB; при d=2048 без тайла ≈ **38 MB/layer**); цикл `n × out × in × G`.
5. **Metal per-call alloc.** `forward_metal` (`src/kan.rs:394`) `from_vec` `xt`/`yt`, `attach`, `fused_metal` `y.download()` после `wait_until_completed` (`src/device.rs:285`). Даже после no-copy **`yt.as_slice().to_vec()`** (`kan.rs:457`) и шесть `Vec` в `KanBlock::forward_mode`. CPU fused: `vec![n*in*G]` (`accelerate.rs:936`) + `prepare_weights` clones.
6. **Checkpointing не лечит dual/snapshot/host bwd.** `fused_grad_ckpt` хранит `x^{(ℓ)}` и **перезапускает полный `forward_mode(..., tape=true)`** в host `BlockCache`, затем **host** `KanBlock::backward` (`model.rs:149–160, 518–534`). MSL `#if ULLIS_FUSED_GRAD_CKPT (void)0` — no-op. DESIGN.md «re-dispatch kernel on backward» сегодня = rematerialize **forward**, не fused bwd.
7. **Metal process baseline** — главная неизвестная в 90 MB.

### Caps

`MobKanSpec::MAX_IN=256`, `MAX_G=16`, `MAX_K=4` (`accelerate.rs:36–38`); `scratch_floats = in + in·G + K` (`:178–180`, MSL `device.rs:673–675`). 32 KB ⇒ 8192 floats; G=16 ⇒ `in ≤ ~480`; 256 консервативно. `d>256` байтит `validate()`. `next_grid_size` режет G об `MAX_G`.

Forward уже страйдит `out`: `for (o = tid; o < out_f; o += tpg)` при `tpg = min(out_f, simd, cap)` ≈ 32 (`device.rs:407–410, 735`). **Блокер 1B по in — `MAX_IN`, не out.** Для bwd и occupancy всё равно нужен явный `OUT_TILE`.

MoB **dense**: все K экспертов, softmax-mix. `λ_R H[softmax(r)]` на полных K логитах.

```
L = mean_mask[ −log p_y + λ_H H(p) ] + λ_R mean H(g) + λ_1 ‖w‖_1
```

**Sampled softmax запрещён** при `λ_H > 0`.

### Цель `DESIGN.md`

< **40 MB RSS** train на дефолтах; < **15 MB** packed infer. Infer-цель реалистична после `pack()`. Train-цель = **net** KPI ниже, пока hello Metal не измерен.

---

## Goals & Non-Goals

### Goals

- **P0 / tier A (CPU):** identical grads (φ, STE hardtanh, CE+entropy, knot insert) на **CPU ulp**. Снять snapshot / per-step i8 / dense temps / workspace churn. **Measure-first** RSS. Primary gate: `rss < baseline_metal_hello + 12 MB`. Публиковать `params+grad+opt+workspace` отдельно.
- **P1 / tier B (Metal всегда 1e-4):** tiled fused fwd (`d>256`, `TIN` + **`OUT_TILE`**); fused Metal+CPU bwd; zero-copy tensors (1B bytes/param); packed **storage** ≤4–6 B/param (working set = storage + bounded FP32 unpack of hot tensors). Честный 1B: три конверта, 16–32 GB M-series для packed; 8 GB M1 — нет.
- **P1–P2 / tier C:** `--moe-topk 0|1|2` + load-balance + routing histogram; cognitive-bench не хуже; default-on только follow-up.
- Apple Silicon: M1 NEON … M4/M5 SME через Accelerate; Metal Shared. Нет AMX asm.
- Интеллект: алгоритм менять только tier C.

### Non-Goals

- 1B в 90 MB.
- Adam. Lion — A/B, не default.
- Sampled softmax при `λ_H > 0`.
- Рукописный AMX/SME.
- PyTorch/Candle.
- CP/Tucker в этом цикле.
- Новый `unsafe` вне `device.rs` / `accelerate.rs` (telemetry остаётся).
- Dual-path memcpy shim «на один релиз» — **git revert PR2** достаточно.
- Ломать packed inference.

### Три тира качества

| Tier | Смысл | Ворота |
| ---- | ----- | ------ |
| **A identical-grad** | CPU: φ, STE, CE+entropy, knot insert; fp32 ulp vs текущий **CPU** tape | `tests/math.rs` CPU; snapshot-oracle SGD; CE acc vs `streamed_tied_ce`; **не** Metal |
| **B numerically equivalent** | Тайл, fused bwd, no-copy Metal | `max‖Δy‖_∞`, `max‖Δgrad‖_∞` **< 1e-4** vs **CPU rematerialize + host bwd**. Существующий `fused_ckpt_matches_full_tape` (loss + `embed_grad` 1e-4, `model.rs:996`) **оставить** и расширить на **все** `grad_*` |
| **C algorithm** | top-k, packed master default, curriculum | routing histogram + 15 anchors `data/cognitive-bench.jsonl` (мало для collapse — histogram обязателен) |

**Metal всегда B (1e-4), никогда A-ulp.** Rematerialize vs full tape уже 1e-4, не ulp.

P0 = A на CPU + listed RAM. No-copy Metal = B. Kernels = B. Top-k / default fp16 = C.

---

## Proposed Design

### P0 — identical-grad (делать первым; **без A1 как RSS-рычага**)

Цель: убрать O(params) clones и temps; **измерить** rss. Dual KAN GPU ~0.25 MB не продаём как 90→40.

**До merge PR1c** снять `phys_footprint`: (a) empty Metal `SovereignDevice::open(true)`, (b) `UllisKan::new` default, (c) один `train_step`+`step` **release**. Вписать в HUD.

#### A2. In-place SGD, без snapshot, без per-step i8

Safe Rust **не** держит `HashMap<String, (&mut [f32], &[f32])>` на весь `UllisKan`. Два прохода, **эфемерные** views:

```text
SgdMomentum::step(model, phase):
  // pass 1 — только grads, те же имена/фильтры что trainable_snapshot
  sq = 0
  model.for_each_grad(phase, |name, grad| { sq += dot(grad, grad) })
  scale = clip(sq, max_norm)

  // pass 2 — param + vel; view не переживает callback
  i = 0
  model.for_each_param_mut(phase, |name, w, grad| {
      vel = &mut self.vel[i]           // длина == w.len()
      vel[j] = μ * vel[j] + scale * grad[j]
      w[j]  -= lr * vel[j]
      i += 1
  })
  model.sync_grids()                   // refresh_geometry; может REPLACE inv_widths
```

Фильтры **как** `trainable_snapshot` (`model.rs:803–880`):

| Тензор | Train? |
| ------ | ------ |
| `embed`, `norm`, `n1`, `n2`, `scale_*` | всегда |
| `weight_base/shared/routed`, `router` | если `!packed && phase < 4` |
| `centers` | если `!packed && phase < 3` |
| packed codes | нет |

Walker **не** хранит slices между pass 1/2 и через `sync_grids` / `regrid`.

**Knot insert = zero-all vel (как сегодня).** `train.rs:227–231` делает `SgdMomentum::new` после `insert_knot`. `regrid` Gauss–Jordan-lift в **новый dense layout**; `split_basis` меняется (G=4 `(3,1)` → G=5 `(4,1)` → G=6 `(4,2)`). Новые коэфф. — не «old + zero tail». **Resize tails запрещён** (misaligned momentum, не identity). `remap_vel` = полный `new` / zero. Проекция vel через `project_spline_coeffs` — tier C, не P0.

Между PR1a и PR2: in-place `as_mut_slice` dirty `host_gen`; Metal forward по-прежнему `bind`/`upload`. Не «оптимизировать» bind до A1.

`refresh_embed_i8()` только: `pack`, checkpoint/`load_blob("embed")`, packed infer (`forward_hidden` если `ff.packed`), HUD i8 stats. Train CE — FP32.

`debug_assert`: `train_step` **не** вызывает `PackedI8Matrix::quantize`. Packed forward: `ensure_i8()` если codes пусты.

`trainable_snapshot` остаётся `#[cfg(test)]` oracle для PR1a.

#### A3. `TrainWorkspace` — именованные поля, не `checkout` (PR1b)

Train **single-threaded**. Поле `UllisKan`. **Запрещён** `checkout(&mut self) -> &mut [f32]`: второй слот не компилируется, пока жив первый (тот же баг, что HashMap views у SGD). Interior mutability (`RefCell` / `UnsafeCell`) не нужна и не допускается.

Слоты — **именованные `Vec<f32>`**, чтобы `KanBlock::forward_mode(&mut self, ws: &mut TrainWorkspace, …)` делал split-borrow (`&ws.x`, `&mut ws.n1`, `&mut ws.mix`, …):

```rust
pub struct TrainWorkspace {
    pub x: Vec<f32>,
    pub n1: Vec<f32>,
    pub mix: Vec<f32>,
    pub h: Vec<f32>,
    pub n2: Vec<f32>,
    pub ff: Vec<f32>,
    pub y: Vec<f32>,
    pub dx: Vec<f32>,
    pub gy: Vec<f32>,
    pub dh: Vec<f32>,
    /// L × [n,d] ckpt boundaries (отдельные Vec, не alias на `x` во время слоя).
    pub layer_x: Vec<Vec<f32>>,
    pub xt: Option<SovereignTensor>,
    pub yt: Option<SovereignTensor>,
    /// [tile_n * TIN * G] — не [n * in * G]
    pub bumps: Vec<f32>,
    pub vocab_row: Vec<f32>, // [V] только softmax row, не [V,d]
    pub q_row: Vec<f32>,     // one TWN row
}

/// Grow-only. Never shrink in the hot path. Does not zero.
fn ensure_nd(v: &mut Vec<f32>, n: usize, d: usize) {
    let need = n.saturating_mul(d);
    if v.len() < need {
        v.resize(need, 0.0);
    }
}
```

Перед шагом: `ensure_nd` на каждом поле. `dx.fill(0)` явно в `zero_grad`/начале bwd слоя — не скрытый zero-on-checkout.

| Поле | Shape | Кто пишет | Zero? |
| ---- | ----- | --------- | ----- |
| `x` | `[n,d]` | embed / block out | overwrite |
| `n1` | `[n,d]` | RMS 1 | overwrite |
| `mix` | `[n,d]` | mixer | overwrite |
| `h` | `[n,d]` | `x+mix` | overwrite |
| `n2` | `[n,d]` | RMS 2 | overwrite |
| `ff` | `[n,d]` | KAN y (`forward_metal` пишет сюда, без `to_vec`) | overwrite |
| `y` | `[n,d]` | block residual | overwrite |
| `dx` | `[n,d]` | bwd dx | **явный zero**, затем accum |
| `gy` | `[n,d]` | bwd gy | copy from upstream dy |
| `dh` | `[n,d]` | rms/mix bwd | overwrite |

Ckpt: `layer_x[ℓ]` = копия входа слоя (Θ(L·n·d)). Rematerialize перезаписывает `n1…ff` (те же поля). Во время fused bwd живы `dx`/`gy` и `layer_x[ℓ]`; split-borrow полей это позволяет.

`forward_metal(gpu, spec, x: &[f32], y_out: &mut [f32])` — `y_out` = `&mut ws.ff` (или `ws.y`), не `&mut self` workspace.

CPU fused bumps — `ws.bumps` тайлами `tile_n` (см. A5).

#### A4. In-place CE: `g/den` прямо в `embed_grad` — **без** `[V,d]` increment

`g_k` плотный при полном softmax+`λ_H`. Постоянный `embed_grad: [V,d]` остаётся. **Запрещён** workspace / local `de_inc[V,d]` — это как раз 2.1 MB temp (`mixers.rs:232` + `embed_scatter`), ради которых существует A4.

Сейчас: свежий `dembed` mean-normalize (`mixers.rs:293–295`), затем `add_assign`. **Нельзя** scale'ить весь `embed_grad` после accumulate.

**Обязательный алгоритм (два прохода, без второго `[V,d]`):**

```text
1. den = count(mask[i] != 0)          // или предподсчёт
   inv = 1 / den.max(1)
   dhidden[..n*d].fill(0)             // слот ws.dh / отдельный [n,d], не [V,d]
2. for i in 0..n:
     if mask[i]==0: continue
     compute logits row p, H           // ws.vocab_row [V] only
     g_k = (p_k - 1_{k=y} + λ_H · ∂H/∂z_k) * inv
     dhidden[i] += g ⊙ embed rows
     embed_grad[k] += g_k * h[i]       // прямо в постоянный буфер
3. embed_scatter_acc(ids, dhidden, embed_grad)  // input-token rows; id < vocab
```

Тест: ulp vs текущий `streamed_tied_ce` + `add_assign`. `embed_scatter_acc` bounds-check `id < vocab`.

`λ_H`/`λ_R` без изменений. Chunked softmax (B5) при `V>32768` — те же два прохода по чанкам vocab, всё ещё без `[V,d]` temp.

#### A5. Host backward без clone; **tile_n** bumps сейчас

- W/Q: `as_slice()`, без `.to_vec()`.
- QAT: TWN **on-the-fly per row** в `q_row` (как MSL `row_delta`/`apply_w`), не три полных W clones. Identity с `ternarize_rows`.
- ψ: **не** полный `[n,in,G]` (при d=2048 ≈ 38 MB). Host oracle: `tile_n` (напр. 32–64) × `in` × `G` в `workspace.bumps`, цикл по t-tiles. Default 0.59 MB → остаётся маленьким при росте d, если ещё и `TIN` (PR4); до PR4 tile_n × full in × G при d=32 ок, при d=512 уже `tile_n*512*12*4` — держать `tile_n` так, чтобы bumps ≤ ~1 MB (`tile_n ≤ 1MB / (4·in·G)`).
- `KanBlock::backward`: без `dy.to_vec` / `gy.clone` / `dh.clone` — поля `ws.gy` / `ws.dx` / `ws.dh`.

Тройной цикл = oracle для PR5.

#### A6. Telemetry (PR1c, отдельно от SGD)

| Поле | Смысл |
| ---- | ----- |
| `rss_mb` | `process_memory_mb()` |
| `baseline_metal_mb` | кэш empty-open (раз за процесс) |
| `net_mb` | `rss - baseline` |
| `params_bytes` | trainable numel × elem size |
| `grad_bytes` | все `grad_*` |
| `opt_bytes` | `vel` |
| `workspace_bytes` | слоты |
| `gpu_alias` | 0 до PR2; 1 после |
| `embed_i8_bytes` | codes+scale |
| `scratch_bumps` | bump buffer |

`moe_sgd_steps_do_not_retain_graphs`: ужесточить growth **< 8 MB** (leak test на tiny `d=16 V=128`). **Не** доказательство 40 MB. Default RSS — **manual/release smoke**, задокументировать команду: `./target/release/ullis train --steps 1 --data …` и HUD `net_mb`.

**P0 RSS (порядок, не обещание raw 35 MB):**

| Статья | Сейчас | После P0 (без A1) |
| ------ | ------ | ----------------- |
| FP32 W | 1.25 MB | 1.25 MB |
| grads | 1.25 MB | 1.25 MB |
| momentum | 1.25 MB | 1.25 MB |
| GPU dual KAN | +~0.25 MB | **остаётся** до PR2 |
| snapshot temps | ~2.5 MB | 0 |
| dembed+scatter temps | ~2.1 MB | 0 |
| bumps/xt churn | 0.5–2 MB | pool ~0.6 MB tiled |
| i8 rebuild | CPU spike | нет в `train_step` |
| Metal hello | **неизмерено** | измерить |
| **Gate** | — | **`rss < hello + 12 MB`** |

### P1 — scale to 1B

#### Три конверта (честные)

Активации default: L·n·d + bumps tile. 1B пример: `d=2048, L=12, G_s=8, G_r=4, K=3` → L·d²·21 ≈ **1.06e9** KAN W + embed `V·d`.

| Envelope | Master | Bytes/param storage | W+g+vel KAN | Activations (ckpt L·n·d, n=384) | Bumps tiled | dW partials `N_TILE·OUT_TILE·TIN` | Hello | **Peak (порядок)** |
| -------- | ------ | ------------------- | ----------- | -------------------------------- | ----------- | --------------------------------- | ----- | ------------------ |
| **(1) P0 default** G=12 | FP32 | embed 12, KAN 16 | ~5 MB + dual 0.25 | ~0.15 MB | ~0.6 MB | n/a (host bwd) | 30–50? | **hello + ~8–12 MB** |
| **(2) 1B FP32 master** (после A1, до PR6) | FP32 | **12** attached+embed | **~12.7 GB** | L·n·d ≈ 12·384·2048·4 ≈ 38 MB | tile ≪ 38 MB full | cap ≤ **4 MB**/launch (см. B2) | ~0.05 GB | **~13 GB + hello** — 16 GB впритык, 8 GB нет |
| **(3) 1B packed storage** | 4–6 B/param storage | 4–6 | **~4–6 GB stored** + **hot FP32 unpack** (1 слой ≈ 21 d² × 4 ≈ 88 MB) | ~38 MB | tiled | ≤ 4 MB | ~0.05 | **~6–8 GB + unpack** — 16–32 GB ок |

**Не** цитировать «6 GB» как train peak, пока unpack не bounded (один hot layer FP32). Envelope (2) без PR6.

#### B1. Tiled fused forward: `TIN` **и** `OUT_TILE`

```
TIN     = min(in, floor((8192 - K) / (1+G)))   // G=16 → ≤480; практический 128/256
OUT_TILE обязателен если out_f > tpg (tpg сегодня ~simd=32; при желании поднять к cap)
MAX_IN  → TILE_IN cap, не model d
```

MSL: цикл `in0 += TIN`; scratch `x[TIN]`, `ψ[TIN·G]`, `gates[K]`. Out: существующий `o += tpg` **плюс** явный `OUT_TILE`. Fwd может остаться `groups=(n,1,1)`; **bwd** — `groups=(n_tiles, out_tiles)` (B2). Общие константы `TIN` / `OUT_TILE` / `N_TILE` в `MobKanSpec`.

**CPU в том же PR4:** `cblas_sgemm` по `TIN`; **запрещён** `vec![n*in*G]` на полный in. Те же тайлы, что Metal.

`G>16` позже (g-chunks). Предпочесть широкий d. `d=512` smoke в PR4 (`tests/math.rs` + `--d-model 512` train smoke).

#### B2. Fused backward — заменяет rematerialize+host FF

Forward сегодня: **один TG на token**, `groups=(n,1,1)`, `tpg≈simd=32`. Сериализовать **все n** внутри одного TG на `OUT_TILE` — race-free, но на 1B (`n=384`, `out_f=2048`) это ~64 TG vs 384 fwd и каждый поток ×384 токенов. **Не** GPU map.

**dW default (KD11): token-tile + reduce, без atomics.**

```
N_TILE   = min(n, 8..16)     // occupancy порядка fwd, не serial-n
grid     = (n_tiles, out_tiles)   // n_tiles = ceil(n / N_TILE), out_tiles = ceil(out / OUT_TILE)
TG (nt, ot) владеет tokens [nt·N_TILE, …) × outputs [ot·OUT_TILE, …)
        → private dW partial  [OUT_TILE × TIN]  (device buffer per TG, не global dW)
reduce   = второй encoder (сумма n_tiles partials → global dW / dScale / dRouter / dCenters)
           или host sum, если n_tiles·OUT_TILE·TIN мало
```

Peak partials **на один in-tile launch**: `n_tiles · OUT_TILE · TIN` floats. Cap: **≤ 4 MB** (пример: n_tiles=16 × `OUT_TILE=32` × `TIN=256` = 512 KB). Routed/shared — те же тайлы `TIN`/`G` chunk; не держать `n_tiles × |W_routed|`. Envelope (2)/(3) включают эту строку.

**Один compute encoder** на слой в ckpt-bwd: в том же kernel сначала rematerialize ψ/gates в threadgroup scratch (`in + in·G + K`, как fwd), затем dX (в `ws.dx` по своим токенам — нет гонки: токены партиционированы) и запись **partial** dW. ψ не запускается вторым encoder'ом. Reduce dW — отдельный короткий encoder (или хост).

Atomics на global dW — **experiment only**, не default.

```mermaid
sequenceDiagram
  participant T as train_step
  participant W as TrainWorkspace
  participant K as fused_fwd_bwd encoder
  participant R as reduce_dW encoder
  participant H as host RMS/shift
  T->>W: save layer_x[ℓ]
  T->>K: fwd F_ℓ (train) drop ψ
  T->>T: CE → dhidden (host, no [V,d] temp)
  T->>H: RMSNorm* final (ws fields)
  loop ℓ = L-1 … 0
    T->>K: ONE encoder: rematerialize ψ in TG scratch then dX + dW_partial
    Note over K: grid (n_tiles, out_tiles); no atomics; STE; λ_R full K
    K->>R: sum partials → dW_*, dScale_*, dRouter, dCenters
    K->>W: dx tile into ws.dx
    T->>H: RMS2 / mix / RMS1 on ws.n1,h,n2,gy,dh
    H->>W: observe_residuals from dCenters
  end
  T->>T: embed_scatter_acc
```

Контракт:

- Fused bwd **заменяет** `recompute_cache` + host `ff.backward`. **Не** double rematerialize FF (ψ один раз в том же encoder).
- Residual/RMS/mix: host, **поля** `ws`. Если ckpt не хранит `n1/h/n2`, дешёвая rematerialize RMS+shift (не KAN).
- Фазы в одном encoder после ψ: dX, partial dW_base/shared/routed, **dScale_***; затем dRouter/dCenters partials (та же сетка). STE `ste_gate`; TWN как fwd.
- После reduce: `observe_residuals`; `grad_centers.fill(0)` если `phase≥3`.
- **`λ_R` на полном router softmax (K логитов), даже при top-k.**
- **Oracle = CPU rematerialize + host `TernaryKanLinear::backward`.**
- `ULLIS_HOST_BWD=1` / test cfg: старый путь.

PR5 gate: `fused_ckpt_matches_full_tape` + все `grad_*`; CPU fused-bwd vs host 1e-4 при d=32 и tiled d=512.

#### B3. Packed / layer-wise **storage** (после fused FP32 bwd)

Compute **всегда FP32 working copies** hot тензоров до флага PR6. Master = storage. Pack/unpack на enter/exit слоя. Vel FP32, пока `--mom q8`.

Hot = слой, чей fused fwd/bwd **сейчас** в полёте (один `TernaryKanLinear`). Не «самый большой CE».

Centers / Gauss / knot — **всегда FP32**.

`as_mut_slice` SGD: walker даёт FP32 view hot или временный unpack buffer; не `&mut [f16]` в optim.

STE `ste_gate` остаётся на FP32 `|w|≤1`.

Checkpoint: `collect_blobs` как сейчас (materialize FP32/packed). Миграции формата нет.

**4–6 B/param — storage target, не peak**, пока unpack bounded одним слоем (~88 MB на 1B-примере).

Default `--master fp32`. `--master fp16` после 1e-4 vs FP32 tape.

#### B4. Top-k

`--moe-topk 0|1|2` (0 = dense, bit-identical tier A).

- k выбирается **per token** (micro-router, как сейчас `softmax(x W_rᵀ)`).
- Routed GEMM только для top-k экспертов (MSL skip; CPU skip). `W_routed` **хранится плотно**.
- Aux: Switch `α · N · Σ_i f_i P_i`, default `α=0.01` (`--moe-aux`).
- `λ_R` на **полном** softmax до top-k.
- PR7 может шипнуть **forward** top-k + **host bwd** (dense grads на невыбранных = 0). Fused bwd skip — если PR5 уже в дереве, иначе follow-up.
- Тест: **routing histogram** — ни один эксперт не 1-hot collapsed на >95% tokens на dummy batch; плюс 15 anchors (слабо для collapse — histogram обязателен).
- **Default-on = отдельный follow-up** после длинного train, не merge PR7.

Shared-edge: только PR8 `--kan-factor shared-edge`. CP/Tucker нет.

#### B5. Chunked softmax `V≫32k`

Two-pass exact max / exp-sum / H[p]. V=8192 не обязателен (row 32 KB).

### P2

Top-k follow-up default; knot curriculum; `--accum-steps`; thinking-mask schedule; Lion A/B. Не: Adam, sampled softmax, AMX, Candle.

---

## API / Interface Changes

### `SovereignTensor` (PR2 / P1 scale, не P0 RSS)

```rust
pub struct SovereignTensor {
    // Drop = declaration order, FIRST field first.
    // gpu MUST be declared before slab so MTLBuffer dies before PageSlab::dealloc.
    #[cfg(target_os = "macos")]
    gpu: Option<GpuSlot>,
    slab: PageSlab,
    shape: Vec<usize>,
    numel: usize,
}
```

**Не копировать** `SovereignFlashBuffer` (`data.rs:181–187`: `slab` first, `metal` last = UAF). **В том же PR2** поменять FlashBuffer на `metal` **перед** `slab`.

`host_gen`/`device_gen` **умирают** с memcpy. Alias = одни страницы.

#### Aliasing contract (обязательный)

1. Owner = tensor. `wrap_shared_bytes_no_copy` без lifetime в API (`device.rs:498–507`) — компенсируется полями.
2. **Exclusive CPU/GPU epochs.** Нет `&mut [f32]` (и нет `f32_at_mut`) **живого через** `dispatch_fused_mob_kan`. После GPU write: `wait_until_completed` **до** любого host read outputs. Optional atomic fence **не** замена wait.
3. Перед realloc (`from_vec` replace, `regrid`, `refresh_geometry` если `numel` меняется, `insert_knot`): `gpu = None` / `detach_gpu` **до** drop/replace slab.
4. Wrap **только** pointer 16 KiB-aligned **начала** буфера. **Никогда** mid-slab interior. **Нет** Metal-visible ParamArena offsets в PR2 (`set_buffer(..., 0)` сегодня). Мелкие тензоры: либо собственный page slab (дорого), либо **host `Vec` без wrap** (RMS/centers уже крошечные; centers всё же attached — отдельный маленький slab OK, 16 KiB × L × few ≈ сотни KB, не 40 MB).
5. `as_slice`/`as_mut_slice` = slab bytes. CPU path: `gpu = None`.
6. Unit test: комментарий + debug Drop glue, что `gpu.take()` до `dealloc` (miri не покрывает Metal).

Wrap **все** `SovereignTensor`, **включая embed** (1.25 MB, 1B-значимо; на дефолте не RSS-рычаг).

`forward_metal` clone `to_vec` снимает PR1b (слот `FF`), не A1.

### SGD walker (`src/model.rs`, `src/optim.rs`)

```rust
impl UllisKan {
    pub fn for_each_grad(&self, phase: u8, f: impl FnMut(&str, &[f32]));
    pub fn for_each_param_mut(&mut self, phase: u8, f: impl FnMut(&str, &mut [f32], &[f32]));
}
```

Порядок имён стабилен и совпадает с `trainable_snapshot`. После knot: `SgdMomentum::new` (zero vel).

### `MobKanSpec`

`TILE_IN`, **`OUT_TILE` required** в PR4; **`N_TILE`** для bwd (PR5). `validate`: model `in_f` любой.

### CLI

| Flag | Default | Когда |
| ---- | ------- | ----- |
| `--fused-grad-ckpt` | true | есть |
| `--moe-topk 0\|1\|2` | **0** | PR7; default-on follow-up |
| `--moe-aux` | 0.01 | PR7 |
| `--kan-factor` | none | PR8 |
| `--master fp32\|fp16` | fp32 | PR6 |
| `--mom fp32\|q8` | fp32 | PR6 |
| `--accum-steps` | 1 | P2 |
| `--mem-hud` | on | PR1c |

### Тесты

- CPU A: in-place SGD vs snapshot **включая** (отдельно) insert_knot → zero-vel → one step.
- CPU A: CE acc vs `streamed_tied_ce` ulp; `debug_assert` no quantize in `train_step`.
- `extend_grid_preserves_forward` не ослаблять.
- `fused_ckpt_matches_full_tape`: все `grad_*`, 1e-4 (B, даже на CPU ckpt).
- Metal B: fused fwd/bwd vs CPU 1e-4, d=32 и d=512.
- Top-k: dense path A; histogram; anchors не гейтят merge default-on.
- RSS growth < 8 MB leak test; default 40 MB — release smoke documented.

---

## Data Model Changes

Диск `ULLIS03` без изменений в P0/P1 storage (save materialize).

Knot: Gauss–Jordan; **vel zero** на insert (identity с `train.rs` сегодня).

JSONL / tokenizer без изменений. FlashBuffer field order чинится в PR2.

---

## Alternatives Considered

### 1. PR1-only (in-place SGD + workspace + i8) vs A1 сейчас — **preferred P0**

Dual attached KAN ×2 ≈ **0.25 MB**. Если измерение подтвердит < 1 MB dual, **A1 не P0 RSS**. A1 = P1 bytes/param / 1B. PR graph: PR1a–c → PR3 как P0; PR2 на scale-треке. **Принято.**

### 2. Sampled softmax

Ломает `H[p]`. **Запрещено** при `λ_H>0`.

### 3. Adam

+8 B/param. **Out of scope.** Lion A/B ок.

### 4. Tiling first

Смешивает numeric drift с allocator. **Отклонено** (P0 identity first).

### 5. AMX asm — запрещено.

### 6. Candle — запрещено.

### 7. Top-k сразу default — отклонено; даже default-on в том же milestone, что флаг — отклонено.

### 8. Aligned `Vec` (`posix_memalign` / 16 KiB) + `wrap`, без `PageSlab` на каждом тензоре

Меньше патч, чем ParamArena. `Vec` с custom alloc в stable неудобен (allocator API); можно `Vec` поверх `PageSlab` view. **Альтернатива PR2:** сначала aligned-Vec только для **attach'нутых** тензоров; embed wrap — второй коммит. Если Drop/alias слишком рискован, aligned-Vec + существующий memcpy `upload` (Alternative 1 old) остаётся. **Рекомендация PR2:** `PageSlab` + wrap как FlashBuffer, но **правильный field order**; не ParamArena.

---

## Security & Privacy Considerations

- Нет сети / PII.
- `new_buffer_with_bytes_no_copy`: UAF если buffer переживает dealloc. Mitigation: **declaration-order** `gpu` **перед** `slab` (поля drop **first to last**, не reverse). Locals drop reverse — **не** путать. FlashBuffer сегодня опасен — чинить в PR2.
- Нет `&mut` через dispatch. Нет wrap невыровненного interior.
- `unsafe` только device/accelerate/telemetry.

---

## Observability

- `phys_footprint` + A6 split `rss` vs `params+grad+opt+workspace` vs `net_mb`.
- После PR2: `gpu_alias=1` на Metal.
- Tok/s: baseline = тот же `ullis train --steps 20` release **до** патча; регрессия >10% — смотреть memcpy, не блокировать identity merge без цифры.
- PR7: histogram + anchors; не CE-only.

---

## Rollout Plan

1. P0 identity default on (PR1a–c, PR3).
2. PR2 isolated — **git revert** = rollback; без `#cfg` dual shim.
3. Kernels / topk / master за флагами.
4. CI: `tests/math.rs` CPU всегда; Metal 1e-4 на macOS. Linux = CPU oracle.

---

## Apple Silicon compatibility (M1–M5)

| SoC | Vector | Path |
| --- | ------ | ---- |
| M1–M3 | NEON | Accelerate |
| M4/M5 | SME | тот же Accelerate, без asm |
| Все | Metal Shared | 32 KB TG, `wait_until_completed` |

Нет `AMX` / `arm_sme` / CPU `#cfg`. `TIN` под 32 KB.

---

## Risks

| ID | Sev | Риск | Mitigation |
| -- | --- | ---- | ---------- |
| R1 | **High** | Zero-copy UAF / Drop order | `gpu` **declared first**; detach before realloc; **не** копировать FlashBuffer; починить FlashBuffer в PR2 |
| R2 | **Med** | 16 KiB round-up на RMS/centers | **Не** Metal-visible suballoc; мелкие — отдельный slab или Vec без wrap |
| R3 | **Med** | In-place SGD ≠ snapshot | Two-pass walker = snapshot filters; CPU ulp test; knot = zero vel |
| R4 | **Med** | Fused bwd race / occupancy dW | Token-**tile** + reduce partials; no atomics default; cap `N_TILE·OUT_TILE·TIN` ≤ 4 MB |
| R5 | **Med** | hello ≥ 40 MB | Primary gate **net** +12 MB; raw 40 = hypothesis |
| R6 | **Med** | Top-k collapse | Histogram test; `λ_R` full; default-on follow-up |
| R7 | **Low** | FP16 knot | Centers/Gauss FP32 |
| R8 | **Low** | Stale i8 | `ensure_i8` packed fwd; `debug_assert` no quantize in train_step |
| R9 | **Med** | 1B на 8 GB | Envelope (2) ~13 GB; не обещать M1 8 GB |
| R10 | **Med** | `refresh_geometry` replace `inv_widths` dangling views | Ephemeral walker; detach_gpu before replace |

---

## Key Decisions

1. **P0 до tiling; P0 RSS = PR1-only, не A1.** Сначала CPU-identical clones/temps. Dual KAN ~0.25 MB. A1 — P1 scale / 1B bytes-per-param. KPI №1: **измерить**, потом net RSS.

2. **Zero-copy — P1.** `PageSlab` + wrap; host = GPU contents; CPU без MTLBuffer; unsafe в `device.rs`. **Не** memcpy-shim.

3. **In-place SGD без snapshot в hot path.** Two-pass `for_each_grad` / `for_each_param_mut`. Momentum vel — второй буфер.

4. **Полный softmax + in-place `∂E`, не sampled.** Два прохода: `den`, затем `g/den` **прямо** в `embed_grad` и `dhidden`. **Запрещён** workspace `[V,d]` increment. `embed_grad` плотный.

5. **Top-k `--moe-topk 0|1|2`; default-on follow-up, не PR7 merge.** Shared-edge только PR8.

6. **Нет рукописного AMX.**

7. **Тиры: A = CPU ulp; Metal всегда B 1e-4.**

8. **Aliasing / Drop:** поля drop в **declaration** order → `gpu` **перед** `slab`. Exclusive CPU/GPU epochs; wait before host read of GPU outputs; нет `&mut [f32]` через dispatch; detach wrap до realloc; никогда не wrap невыровненный interior; либо whole-slab + будущие `set_buffer` offsets, либо **нет** Metal-visible suballoc (PR2 = второе). FlashBuffer чинить в том же PR.

9. **Walker, не HashMap views.** Views эфемерны на callback.

10. **Knot insert: zero-all vel** (identity с сегодняшним `SgdMomentum::new`). Не resize tails. Не project vel в P0.

11. **dW fused bwd: token-tile + reduce, без atomics.** Grid `(n_tiles, out_tiles)`; TG пишет private partial `[OUT_TILE×TIN]`; второй encoder (или host) суммирует. **Не** loop-all-n в одном TG. Cap partials ≤ 4 MB/launch. Atomics — experiment. Один encoder rematerialize+bwd (ψ в TG scratch).

12. **Primary RSS gate: `rss < baseline_metal_hello + 12 MB`** после измерения hello. Публиковать split counters. Raw `<40` — цель DESIGN, не merge-blocker, если hello ≥30.

13. **Wrap все `SovereignTensor`, включая embed.** Мелкие RMS можно не wrap.

14. **Compute dtype = FP32 working copies until PR6 flag.** 4–6 B/param = **storage**, не peak. PR6 **после** PR5.

15. **TrainWorkspace = named `Vec<f32>` fields**, split-borrow. **Нет** `checkout(&mut self)`. Resize = `ensure_nd` grow-only. Single-thread, без interior mutability.

---

## Open Questions

1. **Первый KPI: 0.5M RSS vs tiled d>256?**  
   **Resolved (user 2026-08-21):** P0 net-RSS first (`PR1a`/`PR1b`/`PR1c` + `PR3`). Tiling `d>256` is P1 (`PR4`). Do **not** start with tiled `d>256`.

2. **Top-k default после PR7?**  
   **Resolved (user 2026-08-21):** flag only in PR7 (`--moe-topk 0|1|2`). Default-on is a follow-up after longer train + routing histogram, **not** in PR7 merge. Rollback `--moe-topk 0`.

3. **Shared-edge в этом цикле?**  
   **Нет, PR8 only.**

4. **40 MB raw vs net?**  
   **Решено KD12:** primary = hello+12 MB. OQ6 = сделать измерение.

5. **FP16 default phase 1–2?**  
   Нет в P0. PR6 флаг; default-on после 1e-4.

6. **Измерить empty Metal RSS** до закрытия P0 KPI. Команда: release `SovereignDevice::open(true)` + `process_memory_mb` в `ullis smoke`.

---

## References

- `DESIGN.md` — fused kernel, 40/15 MB (и 40–60 train), MoB, entropy, QAT
- `src/config.rs`, `src/tensor.rs`, `src/device.rs`, `src/kan.rs`, `src/model.rs` (`trainable_snapshot:803`, `fused_ckpt_matches_full_tape:996`, `recompute_cache:149`)
- `src/optim.rs`, `src/mixers.rs`, `src/accelerate.rs`, `src/telemetry.rs`, `src/train.rs`
- `src/data.rs` FlashBuffer field order; `src/gauss.rs` `m=(new_g*16).max(64)`
- `tests/math.rs`, `data/cognitive-bench.jsonl` (15 objects)
- DeepSeekMoE / Switch aux (prior art, не зависимости)

---

## PR Plan

Каждый PR независимо ревьюится и мержится. A = CPU ulp; B = 1e-4.

### PR1a — In-place SGD + i8 deferral

- **Title:** `train: two-pass in-place SGD, defer embed i8`
- **Files:** `src/optim.rs`, `src/model.rs` (walker + `write_param` / `train_step` assert), `src/train.rs` (knot → `SgdMomentum::new` zero vel, **как сейчас**), `tests/math.rs`
- **Depends:** none
- **Changes:** `for_each_grad` / `for_each_param_mut` = snapshot phase/packed rules. Нет stored `(&mut, &)` map. `refresh_embed_i8` не в hot embed write. `debug_assert` `train_step` не `quantize`. Snapshot остаётся test oracle. Тест: one step in-place vs snapshot ulp; отдельный тест insert_knot + zero vel + one step.
- **Quality:** A CPU

### PR1b — TrainWorkspace

- **Title:** `train: activation workspace, reuse xt/yt, tile_n bumps`
- **Files:** `src/model.rs`, `src/kan.rs` (`forward_metal` → out slot, no `to_vec` as API), `src/accelerate.rs` (CPU bumps into workspace tiles), `tests/math.rs`
- **Depends:** PR1a optional; может параллельно, но слоты нужны PR3
- **Changes:** named fields `x,n1,mix,h,n2,ff,y,dx,gy,dh`; `ensure_nd` grow-only; **нет** `checkout(&mut self)`; split-borrow in `forward_mode`; ckpt `layer_x`; Metal xt/yt reuse; **tile_n** bump cap. Explicit `dx.fill(0)`. Single-thread, no `RefCell`.
- **Quality:** A CPU

### PR1c — HUD counters

- **Title:** `telemetry: rss vs params/grad/opt/workspace, metal hello`
- **Files:** `src/telemetry.rs`, `src/train.rs`, `src/device.rs` (optional hello probe)
- **Depends:** none (лучше после PR1a чтобы `opt_bytes` точен)
- **Changes:** A6 fields; document release smoke for net RSS. Leak test `<8 MB` growth (не 40 MB proof).
- **Quality:** n/a (observability)

### PR2 — Zero-copy Shared tensors (P1 scale)

- **Title:** `tensor: PageSlab alias, Drop gpu-before-slab, fix FlashBuffer`
- **Files:** `src/tensor.rs`, `src/device.rs`, `src/data.rs` (FlashBuffer field order), `src/kan.rs` bind/regrid detach, `tests/math.rs`
- **Depends:** PR1a (меньше clones во время миграции); не блокер P0 KPI
- **Changes:** aliasing contract; wrap whole tensors including embed; **no** interior wrap; detach before realloc/`refresh_geometry` replace; `gpu_alias`. CPU A; Metal B 1e-4. **Нет** dual-path shim.
- **Quality:** CPU A, Metal B

### PR3 — In-place CE + clone-free host bwd

- **Title:** `bwd: CE g/den into embed_grad, sparse scatter, clone-free KAN bwd`
- **Files:** `src/mixers.rs`, `src/kan.rs`, `src/model.rs`, `src/accelerate.rs` (bumps via workspace — **required**, не optional)
- **Depends:** **PR1b** (slot + bump API)
- **Changes:** two-pass `den` then `g/den` **directly** into `embed_grad`/`dhidden`; **forbid** `[V,d]` temp; scatter `id < vocab`; host bwd no W clones; on-the-fly TWN row; tile_n ψ. Ulp vs `streamed_tied_ce` + `add_assign`. Full softmax, `λ_H`/`λ_R`.
- **Quality:** A CPU (CE + bump grads + ckpt all `grad_*`)

### PR4 — Tiled fused forward d>256

- **Title:** `metal: TIN + OUT_TILE fused forward; CPU tiles same PR`
- **Files:** `src/device.rs` (MSL, dispatch), `src/accelerate.rs` (`MobKanSpec`, **CPU bump tiles**), `src/kan.rs`, `src/model.rs` / `src/train.rs` (d=512 smoke), `tests/math.rs`
- **Depends:** PR2 preferred (stable alias for multi-dispatch); PR1b
- **Changes:** lift `MAX_IN`; **OUT_TILE required** if `out_f > tpg`; CPU **same PR** no full `[n,in,G]`. No G>16.
- **Quality:** B 1e-4 vs untiled CPU; d=32 regression + d=512

### PR5 — Fused backward Metal+CPU

- **Title:** `metal: fused bwd dX/dW/dScale/dRouter/dCenters, n-tile + reduce`
- **Files:** `src/device.rs`, `src/accelerate.rs`, `src/kan.rs`, `src/model.rs` (replace `recompute_cache`+host FF), `tests/math.rs`
- **Depends:** PR4, PR3 (host oracle)
- **Changes:** grid `(n_tiles, out_tiles)`; private partials; reduce encoder; **one** rematerialize+bwd encoder (ψ in TG scratch); dScale; `observe_residuals`; `λ_R` full K; phase≥3 zero dCenters. No loop-all-n. Atomics experiment-only. Oracle CPU rematerialize+host bwd. `ULLIS_HOST_BWD=1`.
- **Quality:** B 1e-4 d=32 and d=512; ckpt all grads

### PR6 — Packed/layer-wise **storage** (after PR5)

- **Title:** `train: FP16 storage master, FP32 hot unpack, optional Q8 mom`
- **Files:** `src/kan.rs`, `src/optim.rs`, `src/model.rs`, `src/quant.rs`, `src/train.rs`, `tests/math.rs`
- **Depends:** **PR5** (FP32 fused bwd first), PR1a, PR2
- **Changes:** compute FP32 working copies of **hot** layer; storage 4–6 B/param **not peak**; `--master fp32|fp16`; knot FP32. Default fp32.
- **Quality:** B 1e-4 when flag on; C if default changes later

### PR7 — Top-k flag (not default-on)

- **Title:** `moe: --moe-topk 0|1|2 + aux + routing histogram`
- **Files:** `src/kan.rs`, `src/accelerate.rs`, `src/device.rs`, `src/config.rs`, `src/train.rs`, `src/model.rs`, tests + `data/cognitive-bench.jsonl`
- **Depends:** PR4 min (fwd skip); PR5 если fused bwd in-tree, иначе host bwd only (задокументировать)
- **Changes:** dense `topk=0` bit-identical A. Aux α=0.01. `λ_R` full logits. Histogram test. Anchors informational. **No default-on in this PR.**
- **Quality:** C for topk>0; A dense

### PR8 (opt) — Shared-edge

- **Title:** `kan: --kan-factor shared-edge experimental`
- **Files:** `src/kan.rs`, `src/config.rs`, `src/accelerate.rs`, `src/device.rs`, `tests/math.rs`
- **Depends:** PR4–PR5
- **Changes:** opt-in; unfactored path no regression. No CP/Tucker.
- **Quality:** C

### Порядок

```text
PR1a → PR1b → PR3 → PR5 → PR6
         └→ PR1c (∥)
PR1a → PR2 → PR4 → PR5 → PR7 → (follow-up default-on) → PR8
```

**P0 merge set:** PR1a + PR1b + PR1c + PR3.  
**P1:** PR2 + PR4 + PR5 + PR6 + PR7 flag.  
**P2:** top-k follow-up + accum/curriculum.
