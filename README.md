# Ullis

Ullis is a dense ternary Hyena language-model core for Apple Silicon.

The project has one architecture only: token embeddings, stacked causal Hyena
long-convolution blocks, tied output projection, and two Multi-Token
Prediction heads (`t+1`, `t+2`). Every learned projection is represented by
ternary codes `{-1, 0, +1}` derived from FP32 master weights.

Ternary projections keep their inference codes in two packed bitplanes
(`positive` and `negative`): two bits per weight. FP32 master weights remain
only for training and are updated through a clipped STE interface. Each output
row has a compact FP32 dequantisation scale, preventing ternary sums from
growing with model width.

Training uses Lion rather than AdamW: Lion retains one FP32 momentum value per
master weight and no variance state. The admission estimate therefore reserves
master weights, gradients, and one momentum vector (12 bytes per FP32
parameter), plus packed ternary codes and activations.

## Current refactor boundary

The CPU radix-2 FFT implementation is the tested semantic reference for a
future Metal kernel. It performs causal zero-padded long convolutions in
`O(T log T)` and deliberately contains no recurrent state, attention matrix,
router, expert stack, spline grid, or slot/BPTT tape.

Its reference path generates the implicit filter one channel at a time and
reuses FFT workspace, so it does not allocate a full `[D,T]` filter tensor.
The memory admission estimate includes that reusable workspace explicitly.

The Metal backend now executes shared-buffer FP32 identity and RMSNorm
references plus the packed two-bitplane ternary projection on Apple Silicon.
Each projection is compared against the CPU implementation in tests. A
caller-owned `MetalRuntime` retains the ternary pipeline, command queue, and
grow-only shared buffers across layer calls; it has no global cache or hidden
locking. Its fused RMSNorm-plus-ternary kernel keeps normalized rows virtual,
so that pre-projection normalization adds no activation buffer. The runtime is
still a numerical-reference component rather than the model's full forward
path. FFT convolution, STE backward, checkpointing, and GRPO remain follow-up
milestones. Every GPU kernel must preserve this reference implementation's
causal convolution semantics and tests.

Metal now also owns a staged radix-2 complex FFT reference (`bit-reversal`
plus ping-pong butterfly passes) with cached complex buffers. For a dense
materialized `[D,T]` filter, `MetalRuntime::causal_long_conv_forward` keeps
signal FFT, filter FFT, frequency multiplication, inverse FFT, and causal
extraction in one command buffer; only the final `[B,T,D]` result returns to
the CPU. Its output is compared with the CPU causal-convolution reference on
Apple GPU.

This is deliberately not yet the model sequence-mixer path. The model uses an
implicit, strided filter and currently avoids materializing `[D,T]` on the
CPU. Moving it to Metal requires generating that filter directly into the
cached GPU buffer; routing it through the dense reference now would add a
large host allocation and violate the RAM budget. The dense reference itself
does not create host-side padded FFT staging arrays: it fills its shared Metal
buffers directly. Configuration admission separately reserves its cached
signal/filter/output buffers, so a future switch cannot silently push unified
memory into swap.

The compact implicit-filter generator is now also a Metal kernel and is
verified against the CPU equation. Its output is written directly in the
zero-imaginary `float2` layout consumed by the filter FFT, followed by all FFT
passes and causal extraction in the same command buffer. The model still uses
its authoritative CPU path by default. `hidden_metal_reference` verifies the
complete block equation on GPU — ternary projection, gate, implicit mixer,
output projection, and residual — against that CPU path. It is intentionally
not selected for training or fast inference yet: its inter-stage `Vec`
readbacks are a correctness baseline, while the next optimisation retains the
same tensors across blocks in GPU buffers.

`TrainConfig` validates an explicit 1 GiB default process budget before model
allocation. Materialising `[B,T,V]` MTP logits is intentionally rejected when
it would consume more than one quarter of that budget. `streamed_mtp_loss`
instead computes stable cross-entropy one vocabulary row at a time, retaining
only `O(B·T·D)` activations.
The two MTP heads are evaluated serially in this path, so their activation
buffers never overlap.
Hyena gating is also applied in-place, avoiding another `[B,T,D]` buffer per
block.
The pre-projection RMSNorm is fused into the ternary projection, so it adds no
normalized activation tensor.

`MtpBatcher` lends fixed `[B,T]` windows directly from the caller's token
corpus. It makes no copy of the corpus and never pads the final partial batch,
so dataset size does not become part of the trainer's RAM footprint.

The tokenizer is byte-level BPE trained solely from the supplied corpus. It
has no built-in language, code, or chat-token word list; only four special
tokens and a 256-byte fallback are fixed. Saved legacy WordPiece tokenizers
must be retrained, which prevents mixing an old embedding table with a new
token-id scheme.

`max_vocab_size` is a ceiling, not a reservation: BPE returns only populated
ids, so a small corpus cannot waste embedding and output rows on an empty
vocabulary tail. `train_bpe_from_reader` ingests a text source line-by-line,
without first retaining the full corpus as `Vec<String>`.

Run a small validation forward pass:

```sh
cargo run -- --smoke
```

The smoke path uses streamed MTP loss rather than allocating vocabulary logits.
