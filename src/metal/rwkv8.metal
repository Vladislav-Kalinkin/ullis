#include <metal_stdlib>
using namespace metal;

// Pipeline smoke for the RWKV-8 Metal runtime. Later PRs replace this with
// LayerNorm, BinaryConnect, ROSA SAM, streamed CE, and WKV7. Identity is not
// a model op.
kernel void ullis_identity(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &elements [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < elements) {
        output[index] = input[index];
    }
}
