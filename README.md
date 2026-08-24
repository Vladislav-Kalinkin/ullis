# Ullis

Ullis is a dense ternary Hyena language-model core for Apple Silicon.

The project has one architecture only: token embeddings, stacked causal Hyena
long-convolution blocks, tied output projection, and two Multi-Token
Prediction heads (`t+1`, `t+2`). Every learned projection is represented by
ternary codes `{-1, 0, +1}` derived from FP16 latent weights.

Ternary projections keep their inference codes in two packed bitplanes
(`positive` and `negative`): two bits per weight. Latent weights, tied
embeddings, scales, and resident activations have an FP16 storage contract;
individual arithmetic kernels widen only at their calculation boundary. Each
output row has a compact scale, preventing ternary sums from growing with
model width.

The optimiser is selected explicitly. `LionFp16` is retained as the reference,
`LionInt8Blockwise` budgets one byte per latent weight plus one FP16 scale per
256-weight block, and `StatelessSgd` has no persistent optimiser state. The
last is the RAM floor, not a claim of equivalent convergence. The low-memory
ledger keeps one reusable FP16 gradient workspace rather than a full-model
gradient tensor.

## Current refactor boundary

The CPU radix-2 FFT implementation is the tested semantic reference for the
Metal kernel. It deliberately contains no recurrent state, attention matrix,
router, expert stack, spline grid, or slot/BPTT tape.

The model sequence mixer uses bounded-receptive-field overlap-save convolution.
`hyena_kernel_len` is the causal filter length and `hyena_chunk_len` is the
FFT block length (default: 1024 and 2048). This is exact for that bounded
filter, including at chunk boundaries, while reusable CPU FFT scratch is
`O(next_power_of_two(chunk + kernel - 1))`, independent of `context_len`.
It does not claim to reproduce an unbounded `T`-tap convolution: that would
still require `O(T)` filter/spectral state. The implicit filter is generated
one channel at a time and never materialised as `[D,T]`.

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

Metal now executes the same bounded overlap-save geometry: it generates only
the filter prefix on-device, packs each causal history window directly from
resident projection storage, performs the FFT chain, and extracts valid output
positions. GPU-vs-CPU tests cover a sequence spanning several chunks. The
resident FP32 reference mixer uses this same plan, so its FFT buffers scale
with `chunks × fft_len`, not the full context.

The compact implicit-filter generator is also a Metal kernel and is verified
against the CPU equation. Its output is written directly in the zero-imaginary
`float2` layout consumed by the filter FFT. The model uses the authoritative
CPU bounded path by default. `hidden_metal_reference` remains available only
where the bounded plan collapses to the full sequence, preserving an honest
CPU/GPU comparison.

FP16 Metal kernels already cover packed ternary projection, tanh gate,
mixed×gate, and residual add with three reusable resident activation slots.
They are tested against the quantised CPU contract; the missing component is
the FP16 overlap-save FFT mixer, not a host-side activation round trip.

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
