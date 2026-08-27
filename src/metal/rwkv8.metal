#include <metal_stdlib>
using namespace metal;

constant float ULLIS_LN_EPS = 1e-5f;

inline float packed_sign(device const uint *bits, uint index) {
    return (bits[index >> 5] & (1u << (index & 31u))) ? 1.0f : -1.0f;
}

inline void atomic_add_f32(device atomic_uint *slot, float delta) {
    uint current = atomic_load_explicit(slot, memory_order_relaxed);
    while (true) {
        const float updated = as_type<float>(current) + delta;
        if (atomic_compare_exchange_weak_explicit(
                slot,
                &current,
                as_type<uint>(updated),
                memory_order_relaxed,
                memory_order_relaxed)) {
            break;
        }
    }
}

// Pipeline smoke. Identity is not a model op.
kernel void ullis_identity(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &elements [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < elements) {
        output[index] = input[index];
    }
}

// Stateless optimizer primitive for FP16 tensors. Matches Fp16Storage::apply_clipped_sgd.
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

kernel void ullis_residual_add(
    device const float *residual [[buffer(0)]],
    device const float *update [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &elements [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < elements) {
        output[index] = residual[index] + update[index];
    }
}

kernel void ullis_time_shift_delta(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &rows [[buffer(2)]],
    constant uint &time [[buffer(3)]],
    constant uint &channels [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint elements = rows * channels;
    if (index >= elements || time == 0u) {
        return;
    }
    const uint row = index / channels;
    const uint channel = index % channels;
    const uint t = row % time;
    if (t == 0u) {
        output[index] = -input[index];
    } else {
        output[index] = input[index - channels] - input[index];
    }
}

kernel void ullis_sign_pack_bits(
    device const float *input [[buffer(0)]],
    device uint *bits [[buffer(1)]],
    constant uint &elements [[buffer(2)]],
    uint word [[thread_position_in_grid]]) {
    const uint base = word * 32u;
    if (base >= elements) {
        return;
    }
    uint packed = 0u;
    for (uint lane = 0u; lane < 32u; ++lane) {
        const uint index = base + lane;
        if (index < elements && input[index] > 0.0f) {
            packed |= 1u << lane;
        }
    }
    bits[word] = packed;
}

kernel void ullis_cmix_relu2(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &elements [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < elements) {
        const float v = input[index];
        output[index] = v > 0.0f ? v * v : 0.0f;
    }
}

kernel void ullis_cmix_relu2_backward(
    device const float *input [[buffer(0)]],
    device const float *output_gradient [[buffer(1)]],
    device float *input_gradient [[buffer(2)]],
    constant uint &elements [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < elements) {
        const float v = input[index];
        input_gradient[index] = v > 0.0f ? 2.0f * v * output_gradient[index] : 0.0f;
    }
}

kernel void ullis_layer_norm(
    device const float *input [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device const half *bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &channels [[buffer(5)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= rows) {
        return;
    }
    const uint offset = row * channels;
    float mean = 0.0f;
    for (uint c = 0u; c < channels; ++c) {
        mean += input[offset + c];
    }
    mean /= float(channels);
    float var = 0.0f;
    for (uint c = 0u; c < channels; ++c) {
        const float d = input[offset + c] - mean;
        var += d * d;
    }
    var /= float(channels);
    const float inv = rsqrt(var + ULLIS_LN_EPS);
    for (uint c = 0u; c < channels; ++c) {
        output[offset + c] =
            (input[offset + c] - mean) * inv * float(weight[c]) + float(bias[c]);
    }
}

kernel void ullis_layer_norm_backward(
    device const float *input [[buffer(0)]],
    device const float *output_gradient [[buffer(1)]],
    device const half *weight [[buffer(2)]],
    device float *input_gradient [[buffer(3)]],
    device atomic_uint *weight_gradient [[buffer(4)]],
    device atomic_uint *bias_gradient [[buffer(5)]],
    constant uint &rows [[buffer(6)]],
    constant uint &channels [[buffer(7)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= rows) {
        return;
    }
    const uint offset = row * channels;
    float mean = 0.0f;
    for (uint c = 0u; c < channels; ++c) {
        mean += input[offset + c];
    }
    mean /= float(channels);
    float var = 0.0f;
    for (uint c = 0u; c < channels; ++c) {
        const float d = input[offset + c] - mean;
        var += d * d;
    }
    var /= float(channels);
    const float inv = rsqrt(var + ULLIS_LN_EPS);
    const float inv_n = inv / float(channels);
    float sum_dxhat = 0.0f;
    float sum_dxhat_xhat = 0.0f;
    for (uint c = 0u; c < channels; ++c) {
        const float xhat = (input[offset + c] - mean) * inv;
        const float dxhat = output_gradient[offset + c] * float(weight[c]);
        sum_dxhat += dxhat;
        sum_dxhat_xhat += dxhat * xhat;
        atomic_add_f32(weight_gradient + c, output_gradient[offset + c] * xhat);
        atomic_add_f32(bias_gradient + c, output_gradient[offset + c]);
    }
    for (uint c = 0u; c < channels; ++c) {
        const float xhat = (input[offset + c] - mean) * inv;
        const float dxhat = output_gradient[offset + c] * float(weight[c]);
        input_gradient[offset + c] =
            inv_n * (float(channels) * dxhat - sum_dxhat - xhat * sum_dxhat_xhat);
    }
}

kernel void ullis_binary_linear(
    device const float *input [[buffer(0)]],
    device const uint *bits [[buffer(1)]],
    device const half *scale [[buffer(2)]],
    device const half *bias [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &in_features [[buffer(6)]],
    constant uint &out_features [[buffer(7)]],
    constant uint &has_bias [[buffer(8)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= rows * out_features) {
        return;
    }
    const uint row = index / out_features;
    const uint o = index % out_features;
    float sum = 0.0f;
    const uint base = o * in_features;
    const device float *x = input + row * in_features;
    for (uint i = 0u; i < in_features; ++i) {
        sum += packed_sign(bits, base + i) * x[i];
    }
    const float b = has_bias != 0u ? float(bias[o]) : 0.0f;
    output[index] = b + float(scale[o]) * sum;
}

kernel void ullis_binary_linear_input_bwd(
    device const float *output_gradient [[buffer(0)]],
    device const uint *bits [[buffer(1)]],
    device const half *scale [[buffer(2)]],
    device float *input_gradient [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &in_features [[buffer(5)]],
    constant uint &out_features [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= rows * in_features) {
        return;
    }
    const uint row = index / in_features;
    const uint i = index % in_features;
    const device float *gy = output_gradient + row * out_features;
    float gx = 0.0f;
    for (uint o = 0u; o < out_features; ++o) {
        gx += gy[o] * float(scale[o]) * packed_sign(bits, o * in_features + i);
    }
    input_gradient[index] = gx;
}

kernel void ullis_binary_linear_scale_bwd(
    device const float *input [[buffer(0)]],
    device const float *output_gradient [[buffer(1)]],
    device const uint *bits [[buffer(2)]],
    device float *scale_gradient [[buffer(3)]],
    device float *bias_gradient [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &in_features [[buffer(6)]],
    constant uint &out_features [[buffer(7)]],
    constant uint &has_bias [[buffer(8)]],
    uint o [[thread_position_in_grid]]) {
    if (o >= out_features) {
        return;
    }
    const uint base = o * in_features;
    float g_scale = 0.0f;
    float g_bias = 0.0f;
    for (uint row = 0u; row < rows; ++row) {
        const float gy = output_gradient[row * out_features + o];
        const device float *x = input + row * in_features;
        float signed_dot = 0.0f;
        for (uint i = 0u; i < in_features; ++i) {
            signed_dot += packed_sign(bits, base + i) * x[i];
        }
        g_scale += gy * signed_dot;
        g_bias += gy;
    }
    scale_gradient[o] = g_scale;
    if (has_bias != 0u) {
        bias_gradient[o] = g_bias;
    }
}

kernel void ullis_binary_linear_latent_sgd(
    device half *latent [[buffer(0)]],
    device uint *bits [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device const float *output_gradient [[buffer(3)]],
    device const half *scale [[buffer(4)]],
    device float *weight_gradient [[buffer(5)]],
    constant uint &rows [[buffer(6)]],
    constant uint &in_features [[buffer(7)]],
    constant uint &out_features [[buffer(8)]],
    constant float &learning_rate [[buffer(9)]],
    uint word [[thread_position_in_grid]]) {
    const uint elements = out_features * in_features;
    const uint base = word * 32u;
    if (base >= elements) {
        return;
    }
    uint packed = 0u;
    for (uint lane = 0u; lane < 32u; ++lane) {
        const uint index = base + lane;
        if (index >= elements) {
            continue;
        }
        const uint o = index / in_features;
        const uint i = index % in_features;
        const float s = float(scale[o]);
        float gw = 0.0f;
        for (uint row = 0u; row < rows; ++row) {
            gw += output_gradient[row * out_features + o] * s * input[row * in_features + i];
        }
        weight_gradient[index] = gw;
        const float update = clamp(gw, -1.0f, 1.0f);
        const half current = latent[index];
        const half rounded = half(float(current) - learning_rate * update);
        half next = rounded;
        if (rounded == current && update != 0.0f) {
            const half neighbor = update > 0.0f
                ? nextafter(current, half(-INFINITY))
                : nextafter(current, half(INFINITY));
            const float ulp = abs(float(neighbor) - float(current));
            next = abs(learning_rate * update) >= ulp / 32.0f ? neighbor : current;
        }
        latent[index] = next;
        if (float(next) >= 0.0f) {
            packed |= 1u << lane;
        }
    }
    bits[word] = packed;
}

kernel void ullis_fp16_linear(
    device const float *input [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &in_features [[buffer(4)]],
    constant uint &out_features [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= rows * out_features) {
        return;
    }
    const uint row = index / out_features;
    const uint o = index % out_features;
    const device float *x = input + row * in_features;
    const device half *w = weight + o * in_features;
    float sum = 0.0f;
    for (uint i = 0u; i < in_features; ++i) {
        sum += float(w[i]) * x[i];
    }
    output[index] = sum;
}

kernel void ullis_fp16_linear_bwd(
    device const float *input [[buffer(0)]],
    device const float *output_gradient [[buffer(1)]],
    device const half *weight [[buffer(2)]],
    device float *input_gradient [[buffer(3)]],
    device float *weight_gradient [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &in_features [[buffer(6)]],
    constant uint &out_features [[buffer(7)]],
    constant uint &kind [[buffer(8)]],
    uint index [[thread_position_in_grid]]) {
    if (kind == 0u) {
        if (index >= rows * in_features) {
            return;
        }
        const uint row = index / in_features;
        const uint i = index % in_features;
        const device float *gy = output_gradient + row * out_features;
        float gx = 0.0f;
        for (uint o = 0u; o < out_features; ++o) {
            gx += gy[o] * float(weight[o * in_features + i]);
        }
        input_gradient[index] = gx;
    } else {
        if (index >= out_features * in_features) {
            return;
        }
        const uint o = index / in_features;
        const uint i = index % in_features;
        float gw = 0.0f;
        for (uint row = 0u; row < rows; ++row) {
            gw += output_gradient[row * out_features + o] * input[row * in_features + i];
        }
        weight_gradient[index] = gw;
    }
}

// Two vocabulary scans: log-sum-exp then softmax-weighted features.
// Packed ±1 head, no [rows, vocab] logits tensor.
kernel void ullis_streamed_cross_entropy_fp16(
    device const float *hidden [[buffer(0)]],
    device const uint *bits [[buffer(1)]],
    device const half *scale [[buffer(2)]],
    device const uint *tokens [[buffer(3)]],
    device float *hidden_gradient [[buffer(4)]],
    device float *row_loss [[buffer(5)]],
    device atomic_uint *scale_gradient [[buffer(6)]],
    constant uint &rows [[buffer(7)]],
    constant uint &time [[buffer(8)]],
    constant uint &channels [[buffer(9)]],
    constant uint &vocab [[buffer(10)]],
    constant uint &horizon [[buffer(11)]],
    constant float &gradient_scale [[buffer(12)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= rows) {
        return;
    }
    const uint offset = row * channels;
    const uint position = row % time;
    if (position + horizon >= time) {
        row_loss[row] = 0.0f;
        for (uint c = 0u; c < channels; ++c) {
            hidden_gradient[offset + c] = 0.0f;
        }
        return;
    }
    const uint target = tokens[row + horizon];
    const device float *state = hidden + offset;
    float maximum = -INFINITY;
    float target_logit = -INFINITY;
    for (uint token = 0u; token < vocab; ++token) {
        float signed_dot = 0.0f;
        const uint base = token * channels;
        for (uint c = 0u; c < channels; ++c) {
            signed_dot += packed_sign(bits, base + c) * state[c];
        }
        const float logit = float(scale[token]) * signed_dot;
        maximum = max(maximum, logit);
        if (token == target) {
            target_logit = logit;
        }
    }
    float exp_sum = 0.0f;
    for (uint c = 0u; c < channels; ++c) {
        hidden_gradient[offset + c] = 0.0f;
    }
    for (uint token = 0u; token < vocab; ++token) {
        float signed_dot = 0.0f;
        const uint base = token * channels;
        for (uint c = 0u; c < channels; ++c) {
            signed_dot += packed_sign(bits, base + c) * state[c];
        }
        const float s = float(scale[token]);
        const float logit = s * signed_dot;
        const float weight = exp(logit - maximum);
        exp_sum += weight;
        for (uint c = 0u; c < channels; ++c) {
            hidden_gradient[offset + c] += weight * s * packed_sign(bits, base + c);
        }
    }
    row_loss[row] = maximum + log(exp_sum) - target_logit;
    const float inv_sum = 1.0f / exp_sum;
    for (uint c = 0u; c < channels; ++c) {
        const uint base = target * channels;
        hidden_gradient[offset + c] = gradient_scale
            * (hidden_gradient[offset + c] * inv_sum - float(scale[target]) * packed_sign(bits, base + c));
    }
    for (uint token = 0u; token < vocab; ++token) {
        float signed_dot = 0.0f;
        const uint base = token * channels;
        for (uint c = 0u; c < channels; ++c) {
            signed_dot += packed_sign(bits, base + c) * state[c];
        }
        const float s = float(scale[token]);
        const float logit = s * signed_dot;
        const float p = exp(logit - maximum) * inv_sum;
        const float gy_logit = gradient_scale * (p - float(token == target));
        atomic_add_f32(scale_gradient + token, gy_logit * signed_dot);
    }
}
