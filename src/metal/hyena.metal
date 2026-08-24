#include <metal_stdlib>
using namespace metal;

// Stage zero for Ullis Metal: a deliberately simple, bounds-checked contract
// for flattened [B, T, D] FP32 tensors. The CPU reference remains authoritative
// while later kernels replace this with fused RMSNorm, ternary projection, and
// radix-2 FFT passes.
kernel void ullis_identity(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &elements [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < elements) {
        output[index] = input[index];
    }
}

// One thread owns one row. This is a numerical-reference kernel used before
// fusing rows into SIMD groups; it has exactly the CPU model's RMS epsilon.
kernel void ullis_rms_norm(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &rows [[buffer(2)]],
    constant uint &channels [[buffer(3)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= rows) {
        return;
    }
    const uint offset = row * channels;
    float sum_squares = 0.0f;
    for (uint channel = 0; channel < channels; ++channel) {
        const float value = input[offset + channel];
        sum_squares += value * value;
    }
    const float inverse_rms = rsqrt(sum_squares / float(channels) + 1e-5f);
    for (uint channel = 0; channel < channels; ++channel) {
        output[offset + channel] = input[offset + channel] * inverse_rms;
    }
}

kernel void ullis_ternary_linear(
    device const float *input [[buffer(0)]],
    device const ulong *positive [[buffer(1)]],
    device const ulong *negative [[buffer(2)]],
    device const float *scales [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &in_features [[buffer(6)]],
    constant uint &out_features [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= rows * out_features) return;
    const uint row = index / out_features;
    const uint out = index % out_features;
    float sum = 0.0f;
    for (uint i = 0; i < in_features; ++i) {
        const uint weight = out * in_features + i;
        const ulong bit = 1ul << (weight & 63u);
        const float code = (positive[weight >> 6u] & bit) ? 1.0f : ((negative[weight >> 6u] & bit) ? -1.0f : 0.0f);
        sum += input[row * in_features + i] * code;
    }
    output[index] = sum * scales[out];
}

// Low-memory projection: FP16 is the resident transport/storage format while
// each dot product still accumulates in FP32. This is the first trainable-path
// kernel and deliberately remains separate from the FP32 numerical oracle.
kernel void ullis_ternary_linear_fp16(
    device const half *input [[buffer(0)]],
    device const ulong *positive [[buffer(1)]],
    device const ulong *negative [[buffer(2)]],
    device const half *scales [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &in_features [[buffer(6)]],
    constant uint &out_features [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= rows * out_features) return;
    const uint row = index / out_features;
    const uint out = index % out_features;
    float sum = 0.0f;
    for (uint i = 0; i < in_features; ++i) {
        const uint weight = out * in_features + i;
        const ulong bit = 1ul << (weight & 63u);
        const float code = (positive[weight >> 6u] & bit) ? 1.0f : ((negative[weight >> 6u] & bit) ? -1.0f : 0.0f);
        sum += float(input[row * in_features + i]) * code;
    }
    output[index] = half(sum * float(scales[out]));
}

// One threadgroup owns one input row. It first reduces RMS into threadgroup
// memory, then uses that scale while producing all output features. This keeps
// the normalized activation virtual: no [rows, in_features] temporary buffer
// is materialized in RAM or GPU memory.
kernel void ullis_rms_norm_ternary_linear(
    device const float *input [[buffer(0)]],
    device const ulong *positive [[buffer(1)]],
    device const ulong *negative [[buffer(2)]],
    device const float *scales [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &in_features [[buffer(6)]],
    constant uint &out_features [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    if (row >= rows) return;
    threadgroup float partial_squares[256];
    const uint input_offset = row * in_features;
    float local_squares = 0.0f;
    for (uint i = lane; i < in_features; i += lanes) {
        const float value = input[input_offset + i];
        local_squares += value * value;
    }
    partial_squares[lane] = local_squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        float sum_squares = 0.0f;
        for (uint other = 0; other < lanes; ++other) {
            sum_squares += partial_squares[other];
        }
        partial_squares[0] = rsqrt(sum_squares / float(in_features) + 1e-5f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inverse_rms = partial_squares[0];
    for (uint out = lane; out < out_features; out += lanes) {
        float sum = 0.0f;
        const uint weight_offset = out * in_features;
        for (uint i = 0; i < in_features; ++i) {
            const uint weight = weight_offset + i;
            const ulong bit = 1ul << (weight & 63u);
            const float code = (positive[weight >> 6u] & bit) ? 1.0f : ((negative[weight >> 6u] & bit) ? -1.0f : 0.0f);
            sum += input[input_offset + i] * code;
        }
        output[row * out_features + out] = sum * inverse_rms * scales[out];
    }
}

// Global-memory radix-2 FFT passes. Each pass reads one complete source
// buffer and writes a separate destination buffer, so no butterfly has a
// write race. The host ping-pongs the two buffers across `log2(fft_len)`
// stages; this is slower than a tiny threadgroup FFT but scales to 32k+ token
// contexts where a full transform cannot fit in one threadgroup.
kernel void ullis_fft_bitreverse(
    device const float2 *input [[buffer(0)]],
    device float2 *output [[buffer(1)]],
    constant uint &fft_len [[buffer(2)]],
    constant uint &transforms [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    const uint total = fft_len * transforms;
    if (index >= total) return;
    const uint transform = index / fft_len;
    uint source = index - transform * fft_len;
    uint reversed = 0u;
    uint remaining = fft_len;
    while (remaining > 1u) {
        reversed = (reversed << 1u) | (source & 1u);
        source >>= 1u;
        remaining >>= 1u;
    }
    output[index] = input[transform * fft_len + reversed];
}

kernel void ullis_fft_stage(
    device const float2 *input [[buffer(0)]],
    device float2 *output [[buffer(1)]],
    constant uint &fft_len [[buffer(2)]],
    constant uint &transforms [[buffer(3)]],
    constant uint &stage [[buffer(4)]],
    constant uint &inverse [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    const uint total = fft_len * transforms;
    if (index >= total) return;
    const uint width = 1u << stage;
    const uint half = width >> 1u;
    const uint transform_offset = (index / fft_len) * fft_len;
    const uint local = index - transform_offset;
    const uint base = transform_offset + (local / width) * width;
    const uint position = local - (local / width) * width;
    const uint offset = position & (half - 1u);
    const float sign = inverse != 0u ? 1.0f : -1.0f;
    const float angle = sign * 6.28318530718f * float(offset) / float(width);
    const float2 twiddle = float2(cos(angle), sin(angle));
    const float2 even = input[base + offset];
    const float2 odd_source = input[base + offset + half];
    const float2 odd = float2(
        odd_source.x * twiddle.x - odd_source.y * twiddle.y,
        odd_source.x * twiddle.y + odd_source.y * twiddle.x);
    output[index] = position < half ? even + odd : even - odd;
}

kernel void ullis_fft_complex_multiply(
    device const float2 *signal [[buffer(0)]],
    device const float2 *filter [[buffer(1)]],
    device float2 *output [[buffer(2)]],
    constant uint &fft_len [[buffer(3)]],
    constant uint &channels [[buffer(4)]],
    constant uint &transforms [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    const uint total = fft_len * transforms;
    if (index >= total) return;
    const uint channel = (index / fft_len) % channels;
    const float2 a = signal[index];
    const float2 b = filter[channel * fft_len + (index % fft_len)];
    output[index] = float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Generates the compact implicit filter directly as zero-imaginary float2
// values, ready for the filter FFT buffer. One thread owns one [channel,time]
// entry; padding remains zero because the host clears the reusable buffer.
kernel void ullis_generate_implicit_filter(
    device const float *freq [[buffer(0)]],
    device const float *phase [[buffer(1)]],
    device const float *decay [[buffer(2)]],
    device float2 *output [[buffer(3)]],
    constant uint &time [[buffer(4)]],
    constant uint &sequence_len [[buffer(5)]],
    constant uint &order [[buffer(6)]],
    constant uint &fft_len [[buffer(7)]],
    constant uint &elements [[buffer(8)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= elements) return;
    const uint channel = index / time;
    const uint position = index - channel * time;
    const float normalized_position = float(position) / float(sequence_len);
    float sum = 0.0f;
    const uint base = channel * order;
    for (uint k = 0; k < order; ++k) {
        const uint parameter = base + k;
        sum += exp(-decay[parameter] * float(position))
            * cos(freq[parameter] * normalized_position + phase[parameter]);
    }
    output[channel * fft_len + position] = float2(sum / float(order), 0.0f);
}

// Keeps the signal half unchanged and applies tanh only to the gate half of
// each `[rows, 2 * channels]` projection. The next resident-forward step can
// consume this layout directly without a separate gate tensor.
kernel void ullis_tanh_gate_in_place(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &elements [[buffer(2)]],
    constant uint &channels [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= elements) return;
    const uint feature = index % (2u * channels);
    output[index] = feature < channels ? input[index] : tanh(input[index]);
}

kernel void ullis_tanh_gate_fp16(
    device const half *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant uint &elements [[buffer(2)]],
    constant uint &channels [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= elements) return;
    const uint feature = index % (2u * channels);
    output[index] = feature < channels ? input[index] : half(tanh(float(input[index])));
}

kernel void ullis_apply_gate_fp16(
    device const half *mixed [[buffer(0)]],
    device const half *gated_projection [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &channels [[buffer(3)]],
    constant uint &projection_stride [[buffer(4)]],
    constant uint &gate_offset [[buffer(5)]],
    constant uint &elements [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= elements) return;
    const uint row = index / channels;
    const uint channel = index % channels;
    output[index] = half(float(mixed[index]) * float(gated_projection[row * projection_stride + gate_offset + channel]));
}

kernel void ullis_residual_add_fp16(
    device const half *residual [[buffer(0)]],
    device const half *update [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &elements [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < elements) output[index] = half(float(residual[index]) + float(update[index]));
}

// Converts the signal half of a `[B*T, 2D]` projection into the transform
// layout `[B*D, N]`. The host clears the reusable FFT buffer once, so this
// kernel only writes real values that belong to the unpadded sequence.
kernel void ullis_pack_strided_real_to_complex(
    device const float *input [[buffer(0)]],
    device float2 *output [[buffer(1)]],
    constant uint &time [[buffer(2)]],
    constant uint &channels [[buffer(3)]],
    constant uint &input_stride [[buffer(4)]],
    constant uint &input_offset [[buffer(5)]],
    constant uint &fft_len [[buffer(6)]],
    constant uint &elements [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= elements) return;
    const uint channel = index % channels;
    const uint row = index / channels;
    const uint sequence = row / time;
    const uint position = row % time;
    output[(sequence * channels + channel) * fft_len + position] =
        float2(input[row * input_stride + input_offset + channel], 0.0f);
}

// Packs every overlap-save window directly from a resident `[B*T, 2D]`
// projection. Padding and left-of-sequence history are written as zero, so the
// host never clears or stages a full-context FFT tensor.
kernel void ullis_pack_overlap_save_to_complex(
    device const float *input [[buffer(0)]],
    device float2 *output [[buffer(1)]],
    constant uint &time [[buffer(2)]],
    constant uint &channels [[buffer(3)]],
    constant uint &input_stride [[buffer(4)]],
    constant uint &input_offset [[buffer(5)]],
    constant uint &chunk_len [[buffer(6)]],
    constant uint &kernel_len [[buffer(7)]],
    constant uint &fft_len [[buffer(8)]],
    constant uint &chunks_per_sequence [[buffer(9)]],
    constant uint &elements [[buffer(10)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= elements) return;
    const uint local = index % fft_len;
    const uint transform = index / fft_len;
    const uint channel = transform % channels;
    const uint chunk = (transform / channels) % chunks_per_sequence;
    const uint sequence = transform / (channels * chunks_per_sequence);
    const int source_time = int(chunk * chunk_len + local) - int(kernel_len - 1u);
    const bool inside_window = local < kernel_len - 1u + chunk_len;
    const bool inside_sequence = source_time >= 0 && uint(source_time) < time;
    output[index] = inside_window && inside_sequence
        ? float2(input[(sequence * time + uint(source_time)) * input_stride + input_offset + channel], 0.0f)
        : float2(0.0f);
}

kernel void ullis_apply_gate(
    device const float *mixed [[buffer(0)]],
    device const float *gated_projection [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &channels [[buffer(3)]],
    constant uint &projection_stride [[buffer(4)]],
    constant uint &gate_offset [[buffer(5)]],
    constant uint &elements [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= elements) return;
    const uint row = index / channels;
    const uint channel = index % channels;
    output[index] = mixed[index] * gated_projection[row * projection_stride + gate_offset + channel];
}

kernel void ullis_residual_add(
    device const float *residual [[buffer(0)]],
    device const float *update [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &elements [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < elements) output[index] = residual[index] + update[index];
}

kernel void ullis_fft_extract_causal(
    device const float2 *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &time [[buffer(2)]],
    constant uint &channels [[buffer(3)]],
    constant uint &fft_len [[buffer(4)]],
    constant uint &elements [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= elements) return;
    const uint channel = index % channels;
    const uint row = index / channels;
    const uint sequence = row / time;
    const uint position = row % time;
    const uint transform = sequence * channels + channel;
    output[index] = input[transform * fft_len + position].x / float(fft_len);
}

// Extracts only valid output positions from overlap-save transforms and applies
// the already-resident gate in one pass. This avoids a `[B,T,D]` mixed buffer.
kernel void ullis_extract_overlap_save_apply_gate(
    device const float2 *input [[buffer(0)]],
    device const float *gated_projection [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &time [[buffer(3)]],
    constant uint &channels [[buffer(4)]],
    constant uint &projection_stride [[buffer(5)]],
    constant uint &gate_offset [[buffer(6)]],
    constant uint &chunk_len [[buffer(7)]],
    constant uint &kernel_len [[buffer(8)]],
    constant uint &fft_len [[buffer(9)]],
    constant uint &chunks_per_sequence [[buffer(10)]],
    constant uint &elements [[buffer(11)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= elements) return;
    const uint channel = index % channels;
    const uint row = index / channels;
    const uint sequence = row / time;
    const uint position = row % time;
    const uint chunk = position / chunk_len;
    const uint local = position - chunk * chunk_len;
    const uint transform = (sequence * chunks_per_sequence + chunk) * channels + channel;
    const float mixed = input[transform * fft_len + (kernel_len - 1u) + local].x / float(fft_len);
    output[index] = mixed * gated_projection[row * projection_stride + gate_offset + channel];
}

kernel void ullis_extract_overlap_save(
    device const float2 *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &time [[buffer(2)]],
    constant uint &channels [[buffer(3)]],
    constant uint &chunk_len [[buffer(4)]],
    constant uint &kernel_len [[buffer(5)]],
    constant uint &fft_len [[buffer(6)]],
    constant uint &chunks_per_sequence [[buffer(7)]],
    constant uint &elements [[buffer(8)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= elements) return;
    const uint channel = index % channels;
    const uint row = index / channels;
    const uint sequence = row / time;
    const uint position = row % time;
    const uint chunk = position / chunk_len;
    const uint local = position - chunk * chunk_len;
    const uint transform = (sequence * chunks_per_sequence + chunk) * channels + channel;
    output[index] = input[transform * fft_len + (kernel_len - 1u) + local].x / float(fft_len);
}
