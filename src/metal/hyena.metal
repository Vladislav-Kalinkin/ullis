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

// Stateless optimizer primitive for FP16 latent weights. The gradient stays
// FP32 until this calculation boundary, then the updated master value is
// rounded once to its persistent half representation. When a normalized
// update is a meaningful sub-ULP value, advance to the adjacent half instead
// of silently dropping it. This mirrors the CPU zero-state updater and avoids
// an FP32 residual/momentum buffer.
kernel void ullis_clipped_sgd_fp16(
    device half *parameters [[buffer(0)]],
    device const float *gradient [[buffer(1)]],
    constant float &learning_rate [[buffer(2)]],
    constant uint &elements [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < elements) {
        const float update = clamp(gradient[index], -1.0f, 1.0f);
        const half current = parameters[index];
        const half rounded = half(float(current) - learning_rate * update);
        if (rounded != current || update == 0.0f) {
            parameters[index] = rounded;
        } else {
            const half neighbor = update > 0.0f
                ? nextafter(current, half(-INFINITY))
                : nextafter(current, half(INFINITY));
            const float ulp = abs(float(neighbor) - float(current));
            parameters[index] = abs(learning_rate * update) >= ulp / 32.0f
                ? neighbor
                : current;
        }
    }
}

// Exact streamed tied-embedding cross-entropy. One thread owns one sequence
// row and scans the vocabulary twice: first for logsumexp, then for the
// expected embedding. No [rows, vocab] logits or probability tensor exists.
kernel void ullis_streamed_cross_entropy_fp16(
    device const float *head [[buffer(0)]],
    device const half *embedding [[buffer(1)]],
    device const uint *tokens [[buffer(2)]],
    device float *head_gradient [[buffer(3)]],
    device float *row_loss [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &time [[buffer(6)]],
    constant uint &channels [[buffer(7)]],
    constant uint &vocab [[buffer(8)]],
    constant uint &horizon [[buffer(9)]],
    constant float &gradient_scale [[buffer(10)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= rows) return;
    const uint position = row % time;
    if (position + horizon >= time) {
        row_loss[row] = 0.0f;
        for (uint channel = 0; channel < channels; ++channel) head_gradient[row * channels + channel] = 0.0f;
        return;
    }
    const uint target = tokens[row + horizon];
    const device float *state = head + row * channels;
    float maximum = -INFINITY;
    float target_logit = 0.0f;
    for (uint token = 0; token < vocab; ++token) {
        float logit = 0.0f;
        const device half *row_embedding = embedding + token * channels;
        for (uint channel = 0; channel < channels; ++channel) logit += state[channel] * float(row_embedding[channel]);
        maximum = max(maximum, logit);
        if (token == target) target_logit = logit;
    }
    // Reuse the final gradient allocation as an unnormalised expected
    // embedding accumulator. That makes this the second and final vocabulary
    // scan: a separate probability pass would triple the dominant work.
    device float *gradient = head_gradient + row * channels;
    for (uint channel = 0; channel < channels; ++channel) gradient[channel] = 0.0f;
    float exp_sum = 0.0f;
    for (uint token = 0; token < vocab; ++token) {
        float logit = 0.0f;
        const device half *row_embedding = embedding + token * channels;
        for (uint channel = 0; channel < channels; ++channel) logit += state[channel] * float(row_embedding[channel]);
        const float weight = exp(logit - maximum);
        exp_sum += weight;
        for (uint channel = 0; channel < channels; ++channel) gradient[channel] += weight * float(row_embedding[channel]);
    }
    row_loss[row] = maximum + log(exp_sum) - target_logit;
    for (uint channel = 0; channel < channels; ++channel) {
        gradient[channel] = gradient_scale * (gradient[channel] / exp_sum - float(embedding[target * channels + channel]));
    }
}

// Rebuild the per-output ternary scale after an FP16 master update. A single
// thread owns a row so there are no reductions through global memory.
kernel void ullis_ternary_row_scales_fp16(
    device const half *master [[buffer(0)]],
    device float *scales [[buffer(1)]],
    constant uint &in_features [[buffer(2)]],
    constant uint &out_features [[buffer(3)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= out_features) return;
    float sum = 0.0f;
    const uint offset = row * in_features;
    for (uint feature = 0; feature < in_features; ++feature) {
        sum += abs(float(master[offset + feature]));
    }
    scales[row] = sum / float(in_features);
}

// Each thread owns an entire packed word. It regenerates every relevant row
// threshold from the already-computed scale, so bitplanes never need atomics
// and words crossing row boundaries remain race-free.
kernel void ullis_refresh_ternary_codes_fp16(
    device const half *master [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device ulong *positive [[buffer(2)]],
    device ulong *negative [[buffer(3)]],
    constant float &threshold_ratio [[buffer(4)]],
    constant uint &in_features [[buffer(5)]],
    constant uint &parameter_count [[buffer(6)]],
    uint word [[thread_position_in_grid]]) {
    const uint first = word * 64u;
    if (first >= parameter_count) return;
    ulong positive_word = 0ul;
    ulong negative_word = 0ul;
    for (uint lane = 0; lane < 64u && first + lane < parameter_count; ++lane) {
        const uint weight = first + lane;
        const uint row = weight / in_features;
        const float value = float(master[weight]);
        const float threshold = threshold_ratio * scales[row];
        if (value > threshold) positive_word |= 1ul << lane;
        else if (value < -threshold) negative_word |= 1ul << lane;
    }
    positive[word] = positive_word;
    negative[word] = negative_word;
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

// One thread owns one row of RMSNorm backward. It recomputes the inverse RMS
// and the normalized-gradient projection in registers, retaining no tape.
kernel void ullis_rms_norm_backward(
    device const float *input [[buffer(0)]],
    device const float *normalized [[buffer(1)]],
    device const float *output_gradient [[buffer(2)]],
    device float *input_gradient [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &channels [[buffer(5)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= rows) return;
    const uint offset = row * channels;
    float sum_squares = 0.0f;
    float projection = 0.0f;
    for (uint channel = 0; channel < channels; ++channel) {
        const float value = input[offset + channel];
        sum_squares += value * value;
        projection += normalized[offset + channel] * output_gradient[offset + channel];
    }
    const float inverse_rms = rsqrt(sum_squares / float(channels) + 1e-5f);
    projection /= float(channels);
    for (uint channel = 0; channel < channels; ++channel) {
        input_gradient[offset + channel] = inverse_rms
            * (output_gradient[offset + channel] - normalized[offset + channel] * projection);
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

// Exact input derivative of the packed ternary forward projection. One thread
// owns one input element and reduces only across output features, avoiding
// atomics and preserving the CPU accumulation order per element.
kernel void ullis_ternary_linear_input_backward(
    device const float *output_gradient [[buffer(0)]],
    device const ulong *positive [[buffer(1)]],
    device const ulong *negative [[buffer(2)]],
    device const float *scales [[buffer(3)]],
    device float *input_gradient [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &in_features [[buffer(6)]],
    constant uint &out_features [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= rows * in_features) return;
    const uint row = index / in_features;
    const uint feature = index % in_features;
    float sum = 0.0f;
    for (uint out = 0; out < out_features; ++out) {
        const uint weight = out * in_features + feature;
        const ulong bit = 1ul << (weight & 63u);
        const float code = (positive[weight >> 6u] & bit) ? 1.0f : ((negative[weight >> 6u] & bit) ? -1.0f : 0.0f);
        sum += output_gradient[row * out_features + out] * scales[out] * code;
    }
    input_gradient[index] = sum;
}

// Clipped-STE latent-weight derivative. One thread owns one weight and reduces
// rows locally, so no full-model gradient workspace or atomic FP32 updates are
// needed on the GPU.
kernel void ullis_ternary_linear_ste_weight_backward(
    device const float *input [[buffer(0)]],
    device const float *output_gradient [[buffer(1)]],
    device const float *scales [[buffer(2)]],
    device float *latent_weight_gradient [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &in_features [[buffer(5)]],
    constant uint &out_features [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= in_features * out_features) return;
    const uint out = index / in_features;
    const uint feature = index % in_features;
    float sum = 0.0f;
    for (uint row = 0; row < rows; ++row) {
        sum += output_gradient[row * out_features + out] * scales[out] * input[row * in_features + feature];
    }
    latent_weight_gradient[index] = sum;
}

// Exact bounded causal-convolution input derivative. One thread owns one
// input position and traverses only future outputs in its causal receptive
// field, so accumulation is deterministic and needs no atomics.
kernel void ullis_causal_conv_input_backward(
    device const float *filter [[buffer(0)]],
    device const float *output_gradient [[buffer(1)]],
    device float *input_gradient [[buffer(2)]],
    constant uint &batch [[buffer(3)]],
    constant uint &time [[buffer(4)]],
    constant uint &channels [[buffer(5)]],
    constant uint &kernel_len [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    const uint elements = batch * time * channels;
    if (index >= elements) return;
    const uint channel = index % channels;
    const uint row = index / channels;
    const uint sequence = row / time;
    const uint position = row % time;
    float sum = 0.0f;
    for (uint tap = 0; tap < kernel_len && position + tap < time; ++tap) {
        sum += output_gradient[(sequence * time + position + tap) * channels + channel]
            * filter[channel * kernel_len + tap];
    }
    input_gradient[index] = sum;
}

// Exact bounded causal-convolution filter derivative. One thread owns one
// `[channel,tap]` coefficient and reduces all eligible input rows locally.
kernel void ullis_causal_conv_filter_backward(
    device const float *input [[buffer(0)]],
    device const float *output_gradient [[buffer(1)]],
    device float *filter_gradient [[buffer(2)]],
    constant uint &batch [[buffer(3)]],
    constant uint &time [[buffer(4)]],
    constant uint &channels [[buffer(5)]],
    constant uint &kernel_len [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= channels * kernel_len) return;
    const uint channel = index / kernel_len;
    const uint tap = index % kernel_len;
    float sum = 0.0f;
    for (uint sequence = 0; sequence < batch; ++sequence) {
        for (uint position = tap; position < time; ++position) {
            sum += output_gradient[(sequence * time + position) * channels + channel]
                * input[(sequence * time + position - tap) * channels + channel];
        }
    }
    filter_gradient[index] = sum;
}

// Copies the signal half of a `[rows, 2D]` gated projection. Keeping this as
// a tiny GPU layout pass lets the block-backward command feed the exact causal
// convolution derivative without a host round trip.
kernel void ullis_extract_projection_signal(
    device const float *projection [[buffer(0)]],
    device float *signal [[buffer(1)]],
    constant uint &channels [[buffer(2)]],
    constant uint &elements [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < elements) {
        const uint row = index / channels;
        const uint channel = index % channels;
        signal[index] = projection[row * (2u * channels) + channel];
    }
}

// Adds the convolution signal derivative to the signal half of the input
// projection derivative. The gate derivative already owns both projection
// halves, so this avoids materialising a second `[rows, 2D]` tensor.
kernel void ullis_add_projection_signal_gradient(
    device float *projection_gradient [[buffer(0)]],
    device const float *signal_gradient [[buffer(1)]],
    constant uint &channels [[buffer(2)]],
    constant uint &elements [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < elements) {
        const uint row = index / channels;
        const uint channel = index % channels;
        projection_gradient[row * (2u * channels) + channel] += signal_gradient[index];
    }
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
    const uint half_width = width >> 1u;
    const uint transform_offset = (index / fft_len) * fft_len;
    const uint local = index - transform_offset;
    const uint base = transform_offset + (local / width) * width;
    const uint position = local - (local / width) * width;
    const uint offset = position & (half_width - 1u);
    const float sign = inverse != 0u ? 1.0f : -1.0f;
    const float angle = sign * 6.28318530718f * float(offset) / float(width);
    const float2 twiddle = float2(cos(angle), sin(angle));
    const float2 even = input[base + offset];
    const float2 odd_source = input[base + offset + half_width];
    const float2 odd = float2(
        odd_source.x * twiddle.x - odd_source.y * twiddle.y,
        odd_source.x * twiddle.y + odd_source.y * twiddle.x);
    output[index] = position < half_width ? even + odd : even - odd;
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

// Layout for the input-gradient adjoint of a causal convolution.  Reversing
// the upstream sequence turns correlation with the causal filter into an
// ordinary convolution; the final extraction reverses it back.
kernel void ullis_pack_reverse_gradient_to_complex(
    device const float *input [[buffer(0)]],
    device float2 *output [[buffer(1)]],
    constant uint &time [[buffer(2)]],
    constant uint &channels [[buffer(3)]],
    constant uint &fft_len [[buffer(4)]],
    constant uint &elements [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= elements) return;
    const uint local = index % fft_len;
    const uint transform = index / fft_len;
    const uint channel = transform % channels;
    const uint sequence = transform / channels;
    output[index] = local < time
        ? float2(input[(sequence * time + (time - 1u - local)) * channels + channel], 0.0f)
        : float2(0.0f);
}

kernel void ullis_pack_filter_to_complex(
    device const float *input [[buffer(0)]],
    device float2 *output [[buffer(1)]],
    constant uint &channels [[buffer(2)]],
    constant uint &kernel_len [[buffer(3)]],
    constant uint &fft_len [[buffer(4)]],
    constant uint &elements [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= elements) return;
    const uint local = index % fft_len;
    const uint channel = index / fft_len;
    output[index] = local < kernel_len ? float2(input[channel * kernel_len + local], 0.0f) : float2(0.0f);
}

kernel void ullis_fft_extract_input_backward(
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
    output[index] = input[(sequence * channels + channel) * fft_len + (time - 1u - position)].x / float(fft_len);
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

// Resident training keeps the compact generator state in FP16. Accumulation
// remains FP32 only inside the thread, matching the projection-master policy.
kernel void ullis_generate_implicit_filter_fp16(
    device const half *freq [[buffer(0)]],
    device const half *phase [[buffer(1)]],
    device const half *decay [[buffer(2)]],
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
        sum += exp(-float(decay[parameter]) * float(position))
            * cos(float(freq[parameter]) * normalized_position + float(phase[parameter]));
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

// Backward of `mixed * tanh(gate)`. `gated_projection` already contains the
// tanh-transformed gate half, so this is the exact local derivative used by
// the CPU reference. The host clears projection_gradient before dispatch;
// signal-half entries remain zero.
kernel void ullis_hyena_gate_backward(
    device const float *mixed [[buffer(0)]],
    device const float *gated_projection [[buffer(1)]],
    device const float *output_gradient [[buffer(2)]],
    device float *mixed_gradient [[buffer(3)]],
    device float *projection_gradient [[buffer(4)]],
    constant uint &channels [[buffer(5)]],
    constant uint &elements [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= elements) return;
    const uint row = index / channels;
    const uint channel = index % channels;
    const uint gate_index = row * 2u * channels + channels + channel;
    const float gate = gated_projection[gate_index];
    const float gradient = output_gradient[index];
    mixed_gradient[index] = gradient * gate;
    projection_gradient[gate_index] = gradient * mixed[index] * (1.0f - gate * gate);
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
