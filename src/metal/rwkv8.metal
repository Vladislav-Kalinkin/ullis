#include <metal_stdlib>
using namespace metal;

constant float ULLIS_LN_EPS = 1e-5f;
constant uint ULLIS_TILE = 32u;
constant uint ULLIS_SCALE_THREADS = 256u;

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

// Clipped SGD with an FP32 error-diffusion carry. Matches Fp16Storage::apply_clipped_sgd.
kernel void ullis_clipped_sgd_fp16(
    device half *parameters [[buffer(0)]],
    device float *residual [[buffer(1)]],
    device const float *gradient [[buffer(2)]],
    constant float &learning_rate [[buffer(3)]],
    constant uint &elements [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (index < elements) {
        const float current = float(parameters[index]);
        const float update = learning_rate * clamp(gradient[index], -1.0f, 1.0f);
        const float desired = current - update + residual[index];
        const half rounded = half(desired);
        residual[index] = desired - float(rounded);
        parameters[index] = rounded;
    }
}

// BinaryConnect latent step. Matches Fp16Storage::apply_binaryconnect_sgd.
// `gradient_scale` undoes the window-length mean so the ±1 proxy sees a sum STE.
kernel void ullis_binaryconnect_sgd_fp16(
    device half *parameters [[buffer(0)]],
    device float *residual [[buffer(1)]],
    device const float *gradient [[buffer(2)]],
    constant float &learning_rate [[buffer(3)]],
    constant uint &elements [[buffer(4)]],
    constant float &gradient_scale [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= elements) {
        return;
    }
    const float g = gradient[index] * gradient_scale;
    if (g == 0.0f || !isfinite(g)) {
        return;
    }
    const float current = float(parameters[index]);
    const float update = learning_rate * clamp(g, -1.0f, 1.0f);
    const float desired = clamp(current - update + residual[index], -1.0f, 1.0f);
    const half rounded = half(desired);
    residual[index] = desired - float(rounded);
    parameters[index] = rounded;
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
    device float *row_mean [[buffer(4)]],
    device float *row_inv [[buffer(5)]],
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
    row_mean[row] = mean;
    row_inv[row] = inv;
    const float inv_n = inv / float(channels);
    float sum_dxhat = 0.0f;
    float sum_dxhat_xhat = 0.0f;
    for (uint c = 0u; c < channels; ++c) {
        const float xhat = (input[offset + c] - mean) * inv;
        const float dxhat = output_gradient[offset + c] * float(weight[c]);
        sum_dxhat += dxhat;
        sum_dxhat_xhat += dxhat * xhat;
    }
    for (uint c = 0u; c < channels; ++c) {
        const float xhat = (input[offset + c] - mean) * inv;
        const float dxhat = output_gradient[offset + c] * float(weight[c]);
        input_gradient[offset + c] =
            inv_n * (float(channels) * dxhat - sum_dxhat - xhat * sum_dxhat_xhat);
    }
}

// One threadgroup per channel; no atomics. Uses mean/inv from the gx kernel.
kernel void ullis_layer_norm_param_bwd(
    device const float *input [[buffer(0)]],
    device const float *output_gradient [[buffer(1)]],
    device const float *row_mean [[buffer(2)]],
    device const float *row_inv [[buffer(3)]],
    device float *weight_gradient [[buffer(4)]],
    device float *bias_gradient [[buffer(5)]],
    constant uint &rows [[buffer(6)]],
    constant uint &channels [[buffer(7)]],
    uint lane [[thread_position_in_threadgroup]],
    uint c [[threadgroup_position_in_grid]]) {
    if (c >= channels) {
        return;
    }
    float g_w = 0.0f;
    float g_b = 0.0f;
    for (uint row = lane; row < rows; row += ULLIS_SCALE_THREADS) {
        const uint index = row * channels + c;
        const float gy = output_gradient[index];
        const float xhat = (input[index] - row_mean[row]) * row_inv[row];
        g_w += gy * xhat;
        g_b += gy;
    }
    threadgroup float shared_w[256];
    threadgroup float shared_b[256];
    shared_w[lane] = g_w;
    shared_b[lane] = g_b;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = ULLIS_SCALE_THREADS / 2u; step > 0u; step >>= 1u) {
        if (lane < step) {
            shared_w[lane] += shared_w[lane + step];
            shared_b[lane] += shared_b[lane + step];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        weight_gradient[c] = shared_w[0];
        bias_gradient[c] = shared_b[0];
    }
}

kernel void ullis_time_shift_mix(
    device const float *input [[buffer(0)]],
    device const half *mix [[buffer(1)]],
    device float *xx_out [[buffer(2)]],
    device float *shifted [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &time [[buffer(5)]],
    constant uint &channels [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    const uint elements = rows * channels;
    if (index >= elements || time == 0u) {
        return;
    }
    const uint row = index / channels;
    const uint c = index % channels;
    const uint t = row % time;
    const float xt = input[index];
    const float xx = (t == 0u) ? -xt : input[index - channels] - xt;
    xx_out[index] = xx;
    shifted[index] = xt + xx * float(mix[c]);
}

// Matches lerp_shift_backward: g_x[t] = g_s[t] - g_s[t]·mix + (t+1<T ? g_s[t+1]·mix : 0),
// g_mix[c] = Σ g_s · xx. g_x is assumed zero on entry (CMix residual path).
kernel void ullis_time_shift_mix_backward(
    device const float *xx [[buffer(0)]],
    device const half *mix [[buffer(1)]],
    device const float *g_shifted [[buffer(2)]],
    device float *g_x [[buffer(3)]],
    device atomic_uint *g_mix [[buffer(4)]],
    constant uint &batch [[buffer(5)]],
    constant uint &time [[buffer(6)]],
    constant uint &channels [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    const uint rows = batch * time;
    const uint elements = rows * channels;
    if (index >= elements || time == 0u) {
        return;
    }
    const uint row = index / channels;
    const uint c = index % channels;
    const uint t = row % time;
    const uint b = row / time;
    const float gs = g_shifted[index];
    const float mix_c = float(mix[c]);
    float gx = gs - gs * mix_c;
    if (t + 1u < time) {
        const uint next = (b * time + (t + 1u)) * channels + c;
        gx += g_shifted[next] * mix_c;
    }
    g_x[index] = gx;
    atomic_add_f32(g_mix + c, gs * xx[index]);
}

kernel void ullis_time_shift_mix3(
    device const float *input [[buffer(0)]],
    device const half *mix_q [[buffer(1)]],
    device const half *mix_k [[buffer(2)]],
    device const half *mix_v [[buffer(3)]],
    device float *q_in [[buffer(4)]],
    device float *k_in [[buffer(5)]],
    device float *v_in [[buffer(6)]],
    constant uint &rows [[buffer(7)]],
    constant uint &time [[buffer(8)]],
    constant uint &channels [[buffer(9)]],
    uint index [[thread_position_in_grid]]) {
    const uint elements = rows * channels;
    if (index >= elements || time == 0u) {
        return;
    }
    const uint row = index / channels;
    const uint c = index % channels;
    const uint t = row % time;
    const float xt = input[index];
    const float xx = (t == 0u) ? -xt : input[index - channels] - xt;
    q_in[index] = xt + xx * float(mix_q[c]);
    k_in[index] = xt + xx * float(mix_k[c]);
    v_in[index] = xt + xx * float(mix_v[c]);
}

kernel void ullis_pack_latent_bits(
    device const half *latent [[buffer(0)]],
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
        if (index < elements && float(latent[index]) >= 0.0f) {
            packed |= 1u << lane;
        }
    }
    bits[word] = packed;
}

// Tiled Y = scale * X @ Sign^T + bias. Threadgroup 32x32; gid=(o, row).
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
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]) {
    const uint o = tgid.x * ULLIS_TILE + tid.x;
    const uint row = tgid.y * ULLIS_TILE + tid.y;
    const uint tg_o = tgid.x * ULLIS_TILE;
    const uint tg_row = tgid.y * ULLIS_TILE;
    threadgroup float a_tile[32][33];
    threadgroup float b_tile[32][33];
    float sum = 0.0f;
    for (uint k0 = 0u; k0 < in_features; k0 += ULLIS_TILE) {
        const uint k = k0 + tid.x;
        const uint load_row = tg_row + tid.y;
        const uint load_o = tg_o + tid.y;
        a_tile[tid.y][tid.x] = (load_row < rows && k < in_features)
            ? input[load_row * in_features + k]
            : 0.0f;
        b_tile[tid.y][tid.x] = (load_o < out_features && k < in_features)
            ? packed_sign(bits, load_o * in_features + k)
            : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (uint kk = 0u; kk < ULLIS_TILE; ++kk) {
            sum += a_tile[tid.y][kk] * b_tile[tid.x][kk];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < rows && o < out_features) {
        const float b = has_bias != 0u ? float(bias[o]) : 0.0f;
        output[row * out_features + o] = b + float(scale[o]) * sum;
    }
}

// Tiled gX = (gY * scale) @ Sign. gid=(i, row), K=out_features.
kernel void ullis_binary_linear_input_bwd(
    device const float *output_gradient [[buffer(0)]],
    device const uint *bits [[buffer(1)]],
    device const half *scale [[buffer(2)]],
    device float *input_gradient [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &in_features [[buffer(5)]],
    constant uint &out_features [[buffer(6)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]) {
    const uint i = tgid.x * ULLIS_TILE + tid.x;
    const uint row = tgid.y * ULLIS_TILE + tid.y;
    const uint tg_i = tgid.x * ULLIS_TILE;
    const uint tg_row = tgid.y * ULLIS_TILE;
    threadgroup float a_tile[32][33];
    threadgroup float b_tile[32][33];
    float gx = 0.0f;
    for (uint k0 = 0u; k0 < out_features; k0 += ULLIS_TILE) {
        const uint k = k0 + tid.x;
        const uint load_row = tg_row + tid.y;
        const uint load_i = tg_i + tid.y;
        a_tile[tid.y][tid.x] = (load_row < rows && k < out_features)
            ? output_gradient[load_row * out_features + k]
            : 0.0f;
        b_tile[tid.y][tid.x] = (load_i < in_features && k < out_features)
            ? float(scale[k]) * packed_sign(bits, k * in_features + load_i)
            : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (uint kk = 0u; kk < ULLIS_TILE; ++kk) {
            gx += a_tile[tid.y][kk] * b_tile[tid.x][kk];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < rows && i < in_features) {
        input_gradient[row * in_features + i] = gx;
    }
}

// One threadgroup per output feature; 256 lanes reduce over rows.
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
    uint lane [[thread_position_in_threadgroup]],
    uint o [[threadgroup_position_in_grid]]) {
    if (o >= out_features) {
        return;
    }
    const uint base = o * in_features;
    float g_scale = 0.0f;
    float g_bias = 0.0f;
    for (uint row = lane; row < rows; row += ULLIS_SCALE_THREADS) {
        const float gy = output_gradient[row * out_features + o];
        const device float *x = input + row * in_features;
        float signed_dot = 0.0f;
        for (uint i = 0u; i < in_features; ++i) {
            signed_dot += packed_sign(bits, base + i) * x[i];
        }
        g_scale += gy * signed_dot;
        g_bias += gy;
    }
    threadgroup float shared_s[256];
    threadgroup float shared_b[256];
    shared_s[lane] = g_scale;
    shared_b[lane] = g_bias;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = ULLIS_SCALE_THREADS / 2u; step > 0u; step >>= 1u) {
        if (lane < step) {
            shared_s[lane] += shared_s[lane + step];
            shared_b[lane] += shared_b[lane + step];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        scale_gradient[o] = shared_s[0];
        if (has_bias != 0u) {
            bias_gradient[o] = shared_b[0];
        }
    }
}

// g_scale[o] = Σ_row gY[row,o] * (Y[row,o] - bias) / scale[o]. Avoids a second matmul.
kernel void ullis_binary_linear_scale_bwd_from_output(
    device const float *output [[buffer(0)]],
    device const float *output_gradient [[buffer(1)]],
    device const half *scale [[buffer(2)]],
    device const half *bias [[buffer(3)]],
    device float *scale_gradient [[buffer(4)]],
    device float *bias_gradient [[buffer(5)]],
    constant uint &rows [[buffer(6)]],
    constant uint &out_features [[buffer(7)]],
    constant uint &has_bias [[buffer(8)]],
    uint lane [[thread_position_in_threadgroup]],
    uint o [[threadgroup_position_in_grid]]) {
    if (o >= out_features) {
        return;
    }
    const float s = float(scale[o]);
    const float b = has_bias != 0u ? float(bias[o]) : 0.0f;
    float g_scale = 0.0f;
    float g_bias = 0.0f;
    for (uint row = lane; row < rows; row += ULLIS_SCALE_THREADS) {
        const uint index = row * out_features + o;
        const float gy = output_gradient[index];
        g_bias += gy;
        if (s != 0.0f) {
            g_scale += gy * (output[index] - b) / s;
        }
    }
    threadgroup float shared_s[256];
    threadgroup float shared_b[256];
    shared_s[lane] = g_scale;
    shared_b[lane] = g_bias;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = ULLIS_SCALE_THREADS / 2u; step > 0u; step >>= 1u) {
        if (lane < step) {
            shared_s[lane] += shared_s[lane + step];
            shared_b[lane] += shared_b[lane + step];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        scale_gradient[o] = shared_s[0];
        if (has_bias != 0u) {
            bias_gradient[o] = shared_b[0];
        }
    }
}

// Tiled gW = scale * gY^T @ X. gid=(i, o), K=rows.
kernel void ullis_binary_linear_weight_bwd(
    device const float *input [[buffer(0)]],
    device const float *output_gradient [[buffer(1)]],
    device const half *scale [[buffer(2)]],
    device float *weight_gradient [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &in_features [[buffer(5)]],
    constant uint &out_features [[buffer(6)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]) {
    const uint i = tgid.x * ULLIS_TILE + tid.x;
    const uint o = tgid.y * ULLIS_TILE + tid.y;
    const uint tg_i = tgid.x * ULLIS_TILE;
    const uint tg_o = tgid.y * ULLIS_TILE;
    threadgroup float a_tile[32][33];
    threadgroup float b_tile[32][33];
    float gw = 0.0f;
    for (uint k0 = 0u; k0 < rows; k0 += ULLIS_TILE) {
        const uint k = k0 + tid.x;
        const uint load_o = tg_o + tid.y;
        const uint load_i = tg_i + tid.y;
        a_tile[tid.y][tid.x] = (load_o < out_features && k < rows)
            ? output_gradient[k * out_features + load_o]
            : 0.0f;
        b_tile[tid.y][tid.x] = (load_i < in_features && k < rows)
            ? input[k * in_features + load_i]
            : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (uint kk = 0u; kk < ULLIS_TILE; ++kk) {
            gw += a_tile[tid.y][kk] * b_tile[tid.x][kk];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (o < out_features && i < in_features) {
        weight_gradient[o * in_features + i] = float(scale[o]) * gw;
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
        const float current = float(latent[index]);
        float desired = current;
        if (gw != 0.0f && isfinite(gw)) {
            const float update = learning_rate * clamp(gw, -1.0f, 1.0f);
            desired = clamp(current - update, -1.0f, 1.0f);
        }
        const half next = half(desired);
        latent[index] = next;
        if (float(next) >= 0.0f) {
            packed |= 1u << lane;
        }
    }
    bits[word] = packed;
}

// Tiled Y = X @ W^T. gid=(o, row).
kernel void ullis_fp16_linear(
    device const float *input [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &in_features [[buffer(4)]],
    constant uint &out_features [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]) {
    const uint o = tgid.x * ULLIS_TILE + tid.x;
    const uint row = tgid.y * ULLIS_TILE + tid.y;
    const uint tg_o = tgid.x * ULLIS_TILE;
    const uint tg_row = tgid.y * ULLIS_TILE;
    threadgroup float a_tile[32][33];
    threadgroup float b_tile[32][33];
    float sum = 0.0f;
    for (uint k0 = 0u; k0 < in_features; k0 += ULLIS_TILE) {
        const uint k = k0 + tid.x;
        const uint load_row = tg_row + tid.y;
        const uint load_o = tg_o + tid.y;
        a_tile[tid.y][tid.x] = (load_row < rows && k < in_features)
            ? input[load_row * in_features + k]
            : 0.0f;
        b_tile[tid.y][tid.x] = (load_o < out_features && k < in_features)
            ? float(weight[load_o * in_features + k])
            : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (uint kk = 0u; kk < ULLIS_TILE; ++kk) {
            sum += a_tile[tid.y][kk] * b_tile[tid.x][kk];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < rows && o < out_features) {
        output[row * out_features + o] = sum;
    }
}

// kind 0: gX, gid=(i, row), K=out. kind 1: gW, gid=(i, o), K=rows.
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
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]) {
    threadgroup float a_tile[32][33];
    threadgroup float b_tile[32][33];
    if (kind == 0u) {
        const uint i = tgid.x * ULLIS_TILE + tid.x;
        const uint row = tgid.y * ULLIS_TILE + tid.y;
        const uint tg_i = tgid.x * ULLIS_TILE;
        const uint tg_row = tgid.y * ULLIS_TILE;
        float gx = 0.0f;
        for (uint k0 = 0u; k0 < out_features; k0 += ULLIS_TILE) {
            const uint k = k0 + tid.x;
            const uint load_row = tg_row + tid.y;
            const uint load_i = tg_i + tid.y;
            a_tile[tid.y][tid.x] = (load_row < rows && k < out_features)
                ? output_gradient[load_row * out_features + k]
                : 0.0f;
            b_tile[tid.y][tid.x] = (load_i < in_features && k < out_features)
                ? float(weight[k * in_features + load_i])
                : 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint kk = 0u; kk < ULLIS_TILE; ++kk) {
                gx += a_tile[tid.y][kk] * b_tile[tid.x][kk];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (row < rows && i < in_features) {
            input_gradient[row * in_features + i] = gx;
        }
    } else {
        const uint i = tgid.x * ULLIS_TILE + tid.x;
        const uint o = tgid.y * ULLIS_TILE + tid.y;
        const uint tg_i = tgid.x * ULLIS_TILE;
        const uint tg_o = tgid.y * ULLIS_TILE;
        float gw = 0.0f;
        for (uint k0 = 0u; k0 < rows; k0 += ULLIS_TILE) {
            const uint k = k0 + tid.x;
            const uint load_o = tg_o + tid.y;
            const uint load_i = tg_i + tid.y;
            a_tile[tid.y][tid.x] = (load_o < out_features && k < rows)
                ? output_gradient[k * out_features + load_o]
                : 0.0f;
            b_tile[tid.y][tid.x] = (load_i < in_features && k < rows)
                ? input[k * in_features + load_i]
                : 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint kk = 0u; kk < ULLIS_TILE; ++kk) {
                gw += a_tile[tid.y][kk] * b_tile[tid.x][kk];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (o < out_features && i < in_features) {
            weight_gradient[o * in_features + i] = gw;
        }
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
    device float *logit_gradient [[buffer(7)]],
    constant uint &rows [[buffer(8)]],
    constant uint &time [[buffer(9)]],
    constant uint &channels [[buffer(10)]],
    constant uint &vocab [[buffer(11)]],
    constant uint &horizon [[buffer(12)]],
    constant float &gradient_scale [[buffer(13)]],
    constant uint &ignore_id [[buffer(14)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= rows) {
        return;
    }
    const uint offset = row * channels;
    const uint position = row % time;
    const uint target = (position + horizon < time) ? tokens[row + horizon] : ignore_id;
    if (position + horizon >= time || target == ignore_id) {
        row_loss[row] = 0.0f;
        for (uint c = 0u; c < channels; ++c) {
            hidden_gradient[offset + c] = 0.0f;
        }
        for (uint token = 0u; token < vocab; ++token) {
            logit_gradient[row * vocab + token] = 0.0f;
        }
        return;
    }
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
        logit_gradient[row * vocab + token] = gy_logit;
        atomic_add_f32(scale_gradient + token, gy_logit * signed_dot);
    }
}

// Softmax + next-token CE on a materialized `[rows, vocab]` logit buffer.
kernel void ullis_softmax_cross_entropy(
    device const float *logits [[buffer(0)]],
    device const uint *tokens [[buffer(1)]],
    device float *row_loss [[buffer(2)]],
    device float *logit_gradient [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &time [[buffer(5)]],
    constant uint &vocab [[buffer(6)]],
    constant uint &horizon [[buffer(7)]],
    constant float &gradient_scale [[buffer(8)]],
    constant uint &ignore_id [[buffer(9)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= rows) {
        return;
    }
    const uint start = row * vocab;
    const uint position = row % time;
    const uint target = (position + horizon < time) ? tokens[row + horizon] : ignore_id;
    if (position + horizon >= time || target == ignore_id) {
        row_loss[row] = 0.0f;
        for (uint token = 0u; token < vocab; ++token) {
            logit_gradient[start + token] = 0.0f;
        }
        return;
    }
    float maximum = -INFINITY;
    for (uint token = 0u; token < vocab; ++token) {
        maximum = max(maximum, logits[start + token]);
    }
    float exp_sum = 0.0f;
    for (uint token = 0u; token < vocab; ++token) {
        exp_sum += exp(logits[start + token] - maximum);
    }
    row_loss[row] = maximum + log(exp_sum) - logits[start + target];
    const float inv_sum = 1.0f / exp_sum;
    for (uint token = 0u; token < vocab; ++token) {
        const float p = exp(logits[start + token] - maximum) * inv_sum;
        logit_gradient[start + token] = gradient_scale * (p - float(token == target));
    }
}

inline uint rosa_bit_index(uint batch, uint time, uint channel, uint times, uint channels) {
    return (batch * times + time) * channels + channel;
}

inline uint rosa_extract_bit(device const uint *bits, uint index) {
    return (bits[index >> 5] >> (index & 31u)) & 1u;
}

inline int rosa_child(
    device const int *trans0,
    device const int *trans1,
    uint base,
    int node,
    uint bit) {
    if (node < 0) {
        return -1;
    }
    const uint index = base + uint(node);
    return bit == 0u ? trans0[index] : trans1[index];
}

inline void rosa_set_child(
    device int *trans0,
    device int *trans1,
    uint base,
    int node,
    uint bit,
    int to) {
    if (node < 0) {
        return;
    }
    const uint index = base + uint(node);
    if (bit == 0u) {
        trans0[index] = to;
    } else {
        trans1[index] = to;
    }
}

kernel void ullis_rosa_sam_reset(
    device int *trans0 [[buffer(0)]],
    device int *trans1 [[buffer(1)]],
    device int *fail [[buffer(2)]],
    device int *maxlen [[buffer(3)]],
    device int *last [[buffer(4)]],
    constant uint &count [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) {
        return;
    }
    trans0[index] = -1;
    trans1[index] = -1;
    fail[index] = -1;
    maxlen[index] = 0;
    last[index] = -1;
}

// 1-bit QKV SAM. One thread owns (batch, channel); time is serial in-thread.
// Global trans0/1, fail, maxlen, last are [B, D, 2T+1] i32. Missing child is -1.
// Caller must run ullis_rosa_sam_reset first; this kernel does not initialize.
kernel void ullis_rosa_qkv_1bit_fwd(
    device const uint *q_bits [[buffer(0)]],
    device const uint *k_bits [[buffer(1)]],
    device const uint *v_bits [[buffer(2)]],
    device const half *scale_e [[buffer(3)]],
    device int *trans0 [[buffer(4)]],
    device int *trans1 [[buffer(5)]],
    device int *fail [[buffer(6)]],
    device int *maxlen [[buffer(7)]],
    device int *last [[buffer(8)]],
    device uchar *idx [[buffer(9)]],
    device float *out [[buffer(10)]],
    constant uint &batch [[buffer(11)]],
    constant uint &time [[buffer(12)]],
    constant uint &channels [[buffer(13)]],
    uint thread_id [[thread_position_in_grid]]) {
    const uint threads = batch * channels;
    if (thread_id >= threads || time == 0u) {
        return;
    }
    const uint b = thread_id / channels;
    const uint c = thread_id % channels;
    const uint nodes = 2u * time + 1u;
    const uint base = thread_id * nodes;

    int u = 1;
    int g = 0;
    int w = 0;
    int h = 0;
    const float e_c = float(scale_e[c]);
    for (uint t = 0u; t < time; ++t) {
        const uint q = rosa_extract_bit(q_bits, rosa_bit_index(b, t, c, time, channels));
        const uint k = rosa_extract_bit(k_bits, rosa_bit_index(b, t, c, time, channels));
        const int i = int(t);

        int p = w;
        int x = h;
        while (p != -1 && rosa_child(trans0, trans1, base, p, q) < 0) {
            const int mp = maxlen[base + uint(p)];
            if (x > mp) {
                x = mp;
            }
            p = fail[base + uint(p)];
        }
        if (p == -1) {
            p = 0;
            x = 0;
        } else {
            p = rosa_child(trans0, trans1, base, p, q);
            x += 1;
        }

        int v = p;
        while (fail[base + uint(v)] != -1 && maxlen[base + uint(fail[base + uint(v)])] >= x) {
            v = fail[base + uint(v)];
        }
        while (v != -1 && (maxlen[base + uint(v)] <= 0 || last[base + uint(v)] < 0)) {
            v = fail[base + uint(v)];
        }

        uchar collapsed = 0;
        if (v != -1) {
            const uint pos = uint(last[base + uint(v)] + 1);
            collapsed = uchar(rosa_extract_bit(
                v_bits,
                rosa_bit_index(b, pos, c, time, channels)));
        }
        const uint out_index = rosa_bit_index(b, t, c, time, channels);
        idx[out_index] = collapsed;
        out[out_index] = (2.0f * float(collapsed) - 1.0f) * e_c;
        w = p;
        h = x;

        const int j = u;
        u += 1;
        maxlen[base + uint(j)] = maxlen[base + uint(g)] + 1;
        p = g;
        while (p != -1 && rosa_child(trans0, trans1, base, p, k) < 0) {
            rosa_set_child(trans0, trans1, base, p, k, j);
            p = fail[base + uint(p)];
        }
        if (p == -1) {
            fail[base + uint(j)] = 0;
        } else {
            const int d = rosa_child(trans0, trans1, base, p, k);
            if (maxlen[base + uint(p)] + 1 == maxlen[base + uint(d)]) {
                fail[base + uint(j)] = d;
            } else {
                const int clone = u;
                u += 1;
                trans0[base + uint(clone)] = trans0[base + uint(d)];
                trans1[base + uint(clone)] = trans1[base + uint(d)];
                maxlen[base + uint(clone)] = maxlen[base + uint(p)] + 1;
                fail[base + uint(clone)] = fail[base + uint(d)];
                last[base + uint(clone)] = last[base + uint(d)];
                fail[base + uint(d)] = clone;
                fail[base + uint(j)] = clone;
                while (p != -1 && rosa_child(trans0, trans1, base, p, k) == d) {
                    rosa_set_child(trans0, trans1, base, p, k, clone);
                    p = fail[base + uint(p)];
                }
            }
        }
        v = j;
        g = j;
        while (v != -1 && last[base + uint(v)] < i) {
            last[base + uint(v)] = i;
            v = fail[base + uint(v)];
        }
    }
}

// g_e[c] = Σ_{b,t} gy[b,t,c] * (2 * idx[b,t,c] - 1). No match mask.
kernel void ullis_rosa_qkv_1bit_bwd_e(
    device const float *output_gradient [[buffer(0)]],
    device const uchar *idx [[buffer(1)]],
    device float *e_gradient [[buffer(2)]],
    constant uint &batch [[buffer(3)]],
    constant uint &time [[buffer(4)]],
    constant uint &channels [[buffer(5)]],
    uint c [[thread_position_in_grid]]) {
    if (c >= channels) {
        return;
    }
    float acc = 0.0f;
    for (uint b = 0u; b < batch; ++b) {
        for (uint t = 0u; t < time; ++t) {
            const uint index = (b * time + t) * channels + c;
            acc += output_gradient[index] * (2.0f * float(idx[index]) - 1.0f);
        }
    }
    e_gradient[c] = acc;
}

constant uint ULLIS_WKV7_N = 16u;
constant uint ULLIS_WKV7_CHUNK = 16u;

// Transcription of cuda/wkv7_cuda.cu::forward_kernel. Thread i is head channel.
kernel void ullis_wkv7_forward(
    device const float *w_ [[buffer(0)]],
    device const float *q_ [[buffer(1)]],
    device const float *k_ [[buffer(2)]],
    device const float *v_ [[buffer(3)]],
    device const float *a_ [[buffer(4)]],
    device const float *b_ [[buffer(5)]],
    device float *y_ [[buffer(6)]],
    device float *s_ [[buffer(7)]],
    device float *sa_ [[buffer(8)]],
    constant uint &T [[buffer(9)]],
    constant uint &H [[buffer(10)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]]) {
    // Metal requires matching scalar/vector ranks on thread-index attributes.
    const uint i = tid.x;
    const uint hh = tg.x;
    const uint bb = tg.y;
    const uint n = ULLIS_WKV7_N;
    if (i >= n || T == 0u || H == 0u || (T % ULLIS_WKV7_CHUNK) != 0u) {
        return;
    }
    float state[16];
    for (uint j = 0u; j < n; ++j) {
        state[j] = 0.0f;
    }
    threadgroup float q[16];
    threadgroup float k[16];
    threadgroup float w[16];
    threadgroup float a[16];
    threadgroup float b[16];
    const uint nchunks = T / ULLIS_WKV7_CHUNK;
    for (uint t = 0u; t < T; ++t) {
        const uint ind = ((bb * T + t) * H + hh) * n + i;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        q[i] = q_[ind];
        w[i] = exp(-exp(w_[ind]));
        k[i] = k_[ind];
        a[i] = a_[ind];
        b[i] = b_[ind];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sa = 0.0f;
        for (uint j = 0u; j < n; ++j) {
            sa += a[j] * state[j];
        }
        sa_[ind] = sa;
        const float vv = v_[ind];
        float y = 0.0f;
        for (uint j = 0u; j < n; ++j) {
            state[j] = state[j] * w[j] + sa * b[j] + k[j] * vv;
            y += state[j] * q[j];
        }
        y_[ind] = y;
        if ((t + 1u) % ULLIS_WKV7_CHUNK == 0u) {
            const uint base = ((bb * H + hh) * nchunks + (t / ULLIS_WKV7_CHUNK)) * n * n + i;
            for (uint j = 0u; j < n; ++j) {
                s_[base + j * n] = state[j];
            }
        }
    }
}

// Transcription of cuda/wkv7_cuda.cu::backward_kernel.
kernel void ullis_wkv7_backward(
    device const float *w_ [[buffer(0)]],
    device const float *q_ [[buffer(1)]],
    device const float *k_ [[buffer(2)]],
    device const float *v_ [[buffer(3)]],
    device const float *a_ [[buffer(4)]],
    device const float *b_ [[buffer(5)]],
    device const float *dy_ [[buffer(6)]],
    device const float *s_ [[buffer(7)]],
    device const float *sa_ [[buffer(8)]],
    device float *dw_ [[buffer(9)]],
    device float *dq_ [[buffer(10)]],
    device float *dk_ [[buffer(11)]],
    device float *dv_ [[buffer(12)]],
    device float *da_ [[buffer(13)]],
    device float *db_ [[buffer(14)]],
    constant uint &T [[buffer(15)]],
    constant uint &H [[buffer(16)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]]) {
    const uint i = tid.x;
    const uint hh = tg.x;
    const uint bb = tg.y;
    const uint n = ULLIS_WKV7_N;
    if (i >= n || T == 0u || H == 0u || (T % ULLIS_WKV7_CHUNK) != 0u) {
        return;
    }
    float stateT[16];
    float dstate[16];
    float dstateT[16];
    for (uint j = 0u; j < n; ++j) {
        stateT[j] = 0.0f;
        dstate[j] = 0.0f;
        dstateT[j] = 0.0f;
    }
    threadgroup float w[16];
    threadgroup float q[16];
    threadgroup float k[16];
    threadgroup float v[16];
    threadgroup float a[16];
    threadgroup float b[16];
    threadgroup float dy[16];
    threadgroup float sa[16];
    threadgroup float dSb_shared[16];
    const uint nchunks = T / ULLIS_WKV7_CHUNK;
    for (uint tstep = 0u; tstep < T; ++tstep) {
        const uint t = T - 1u - tstep;
        const uint ind = ((bb * T + t) * H + hh) * n + i;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        q[i] = q_[ind];
        const float wi_fac = -exp(w_[ind]);
        const float wi = exp(wi_fac);
        w[i] = wi;
        k[i] = k_[ind];
        a[i] = a_[ind];
        b[i] = b_[ind];
        v[i] = v_[ind];
        dy[i] = dy_[ind];
        sa[i] = sa_[ind];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((t + 1u) % ULLIS_WKV7_CHUNK == 0u) {
            const uint base =
                ((bb * H + hh) * nchunks + (t / ULLIS_WKV7_CHUNK)) * n * n + i * n;
            for (uint j = 0u; j < n; ++j) {
                stateT[j] = s_[base + j];
            }
        }
        float dq = 0.0f;
        for (uint j = 0u; j < n; ++j) {
            dq += stateT[j] * dy[j];
        }
        dq_[ind] = dq;
        const float qi = q[i];
        const float ki = k[i];
        const float ai = a[i];
        const float bi = b[i];
        const float dyi = dy[i];
        const float iwi = 1.0f / wi;
        for (uint j = 0u; j < n; ++j) {
            stateT[j] = (stateT[j] - ki * v[j] - bi * sa[j]) * iwi;
            dstate[j] += dyi * q[j];
            dstateT[j] += qi * dy[j];
        }
        float dw = 0.0f;
        float dk = 0.0f;
        float dv = 0.0f;
        float db = 0.0f;
        float dSb = 0.0f;
        for (uint j = 0u; j < n; ++j) {
            dw += dstateT[j] * stateT[j];
            dk += dstateT[j] * v[j];
            dv += dstate[j] * k[j];
            dSb += dstate[j] * b[j];
            db += dstateT[j] * sa[j];
        }
        dw_[ind] = dw * wi * wi_fac;
        dk_[ind] = dk;
        dv_[ind] = dv;
        db_[ind] = db;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        dSb_shared[i] = dSb;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float da = 0.0f;
        for (uint j = 0u; j < n; ++j) {
            da += stateT[j] * dSb_shared[j];
        }
        da_[ind] = da;
        for (uint j = 0u; j < n; ++j) {
            dstate[j] = dstate[j] * w[j] + dSb * a[j];
            dstateT[j] = dstateT[j] * wi + ai * dSb_shared[j];
        }
    }
}
