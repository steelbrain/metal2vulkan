// Owned fixture: the kernels of the harness's authored workloads (crates/metal2vulkan-harness), each
// standing for a class of compute work: bandwidth-bound, ALU-bound, threadgroup memory with
// barriers, SIMD reductions, 2D image-like indexing, many tiny dispatches in a chain, narrow
// integer arithmetic, sampling through a runtime sampler and through a constexpr one, and
// products by a constant zero of either sign. Written
// from scratch for it; `scripts/regen-air-fixtures/regen-air-fixtures.sh --dir harness` compiles
// it. The harness holds a CPU reference of every result. No translator sweep reads this directory.
#include <metal_stdlib>
using namespace metal;

// The harness's own transfers: a Shared staging buffer's words into a Private buffer, and a
// Private buffer's words back out, 16 bytes a thread.
kernel void copy_words(device const uint4 *source [[buffer(0)]],
                       device uint4 *destination [[buffer(1)]],
                       uint index [[thread_position_in_grid]]) {
    destination[index] = source[index];
}

// Bandwidth-bound: one 16-byte read and one 16-byte write a thread, and an affine map between.
kernel void scale_offset(device const float4 *source [[buffer(0)]],
                         device float4 *destination [[buffer(1)]],
                         constant float2 &affine [[buffer(2)]],
                         uint index [[thread_position_in_grid]]) {
    destination[index] = source[index] * affine.x + affine.y;
}

// ALU-bound: two dependent FMA chains a thread, `rounds` long, from values made of the low ten
// bits of the thread's index; one 4-byte write a thread.
kernel void fma_chains(device float *destination [[buffer(0)]],
                       constant uint &rounds [[buffer(1)]],
                       uint index [[thread_position_in_grid]]) {
    float a = float(index & 31u) * (1.0f / 32.0f);
    float b = float((index >> 5) & 31u) * (1.0f / 32.0f);
    for (uint round = 0; round < rounds; ++round) {
        a = fma(a, 0.9375f, b * 0.0625f);
        b = fma(b, 0.875f, a * 0.125f);
    }
    destination[index] = a + b;
}

// Threadgroup memory with barriers: the inclusive prefix sum of each threadgroup's 256 words, by
// doubling steps over threadgroup memory, two barriers a step. Threadgroups of 256 threads.
kernel void block_scan(device const uint *source [[buffer(0)]],
                       device uint *destination [[buffer(1)]],
                       uint index [[thread_position_in_grid]],
                       uint local [[thread_position_in_threadgroup]]) {
    threadgroup uint sums[256];
    uint value = source[index];
    sums[local] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 1; step < 256; step <<= 1) {
        uint add = local >= step ? sums[local - step] : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        value += add;
        sums[local] = value;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    destination[index] = value;
}

// SIMD reductions: threadgroup g sums words [g * shape.y, (g + 1) * shape.y) of the first shape.x
// (wrapping), each SIMD group by `simd_sum`, then the SIMD groups' sums by the first SIMD group,
// whatever the SIMD width; it writes its sum to destination[g]. Threadgroups of at most 256
// threads.
kernel void reduce_sum(device const uint *source [[buffer(0)]],
                       device uint *destination [[buffer(1)]],
                       constant uint2 &shape [[buffer(2)]],
                       uint group [[threadgroup_position_in_grid]],
                       uint local [[thread_position_in_threadgroup]],
                       uint width [[threads_per_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint simd [[simdgroup_index_in_threadgroup]],
                       uint simds [[simdgroups_per_threadgroup]],
                       uint simd_width [[threads_per_simdgroup]]) {
    threadgroup uint partial[256];
    uint begin = group * shape.y;
    uint end = min(begin + shape.y, shape.x);
    uint sum = 0;
    for (uint i = begin + local; i < end; i += width) {
        sum += source[i];
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        partial[simd] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        uint total = 0;
        for (uint i = lane; i < simds; i += simd_width) {
            total += partial[i];
        }
        total = simd_sum(total);
        if (lane == 0) {
            destination[group] = total;
        }
    }
}

// 2D image-like indexing: a 3x3 binomial filter over a size.x by size.y image of float4 texels in
// a buffer, its edges clamped; one thread a texel, on a grid the size of the image.
kernel void binomial3x3(device const float4 *source [[buffer(0)]],
                        device float4 *destination [[buffer(1)]],
                        constant uint2 &size [[buffer(2)]],
                        uint2 position [[thread_position_in_grid]]) {
    float4 sum = 0.0f;
    for (int dy = -1; dy <= 1; ++dy) {
        uint y = uint(clamp(int(position.y) + dy, 0, int(size.y) - 1));
        for (int dx = -1; dx <= 1; ++dx) {
            uint x = uint(clamp(int(position.x) + dx, 0, int(size.x) - 1));
            float weight = (dx == 0 ? 2.0f : 1.0f) * (dy == 0 ? 2.0f : 1.0f);
            sum += source[y * size.x + x] * weight;
        }
    }
    destination[position.y * size.x + position.x] = sum * 0.0625f;
}

// Many tiny dispatches in a chain: one step of a linear congruential generator on each word, its
// increment set per dispatch, so each dispatch reads what the one before it wrote.
kernel void lcg_step(device uint *words [[buffer(0)]],
                     constant uint &increment [[buffer(1)]],
                     uint index [[thread_position_in_grid]]) {
    words[index] = words[index] * 1664525u + increment;
}

// Narrow integers: 8- and 16-bit arithmetic, `rounds` steps a thread, from values made of the
// thread's index. The translation holds each in a 32-bit container and clears the bits above its
// width after every step that can set them. One 4-byte write a thread.
kernel void narrow_mix(device uint *destination [[buffer(0)]],
                       constant uint &rounds [[buffer(1)]],
                       uint index [[thread_position_in_grid]]) {
    ushort a = ushort(index * 40503u);
    short b = short(index ^ 0x5a5au);
    uchar c = uchar(index);
    char d = char(index >> 3);
    for (uint round = 0; round < rounds; ++round) {
        a = ushort(a * ushort(25173) + ushort(13849));
        b = short(b + short(a >> 3));
        c = uchar(c * uchar(5) + uchar(b));
        d = char(d + char(c) - char(round));
    }
    destination[index] = uint(a) ^ (uint(ushort(b)) << 16) ^ uint(c) ^ (uint(uchar(d)) << 8);
}

// The harness's own texture transfers: a Shared staging buffer's float4 texels into a texture,
// and a texture's texels back out to a buffer, one texel a thread, on a grid the size of the
// texture.
kernel void texels_in(device const float4 *source [[buffer(0)]],
                      constant uint &width [[buffer(1)]],
                      texture2d<float, access::write> destination [[texture(0)]],
                      uint2 position [[thread_position_in_grid]]) {
    destination.write(source[position.y * width + position.x], position);
}

kernel void texels_out(texture2d<float, access::read> source [[texture(0)]],
                       device float4 *destination [[buffer(0)]],
                       constant uint &width [[buffer(1)]],
                       uint2 position [[thread_position_in_grid]]) {
    destination[position.y * width + position.x] = source.read(position);
}

// Sampling through a runtime sampler (`setSamplerState:atIndex:`, linear, clamped to the edge): a
// 2x downsample, each texel of the destination one sample at the corner its four source texels
// share, so each filter weight is one half; one texture write a thread.
kernel void sample_down(texture2d<float, access::sample> source [[texture(0)]],
                        texture2d<float, access::write> destination [[texture(1)]],
                        sampler filter [[sampler(0)]],
                        uint2 position [[thread_position_in_grid]]) {
    float2 size = float2(destination.get_width(), destination.get_height());
    destination.write(source.sample(filter, (float2(position) + 0.5f) / size), position);
}

// Sampling through a constexpr sampler (linear, clamped to the edge): a 2x upsample, each texel
// of the destination one sample a quarter of a source texel from a source texel's center, so
// each filter weight is a quarter or three quarters; one texture write a thread.
kernel void sample_up(texture2d<float, access::sample> source [[texture(0)]],
                      texture2d<float, access::write> destination [[texture(1)]],
                      uint2 position [[thread_position_in_grid]]) {
    constexpr sampler filter(coord::normalized, address::clamp_to_edge, filter::linear);
    float2 size = float2(destination.get_width(), destination.get_height());
    destination.write(source.sample(filter, (float2(position) + 0.5f) / size), position);
}

// Products by a constant zero of either sign, in precise arithmetic: `rounds` steps a thread, each
// multiplying the running value by the function constants +0.0 and -0.0 in one block beside an
// FMA, from values made of the thread's index. One 16-byte write a thread.
constant float positive_zero [[function_constant(0)]];
constant float negative_zero [[function_constant(1)]];

#pragma METAL fp math_mode(safe)
kernel void signed_zero_products(device float4 *destination [[buffer(0)]],
                                 constant uint &rounds [[buffer(1)]],
                                 uint index [[thread_position_in_grid]]) {
    float4 a = float4(float(index & 63u), -float((index >> 6) & 63u), 0.5f, -0.25f);
    float4 z = 0.0f;
    for (uint round = 0; round < rounds; ++round) {
        float4 p = a * positive_zero;
        float4 n = a * negative_zero;
        z = z + p - n;
        a = fma(a, 0.96875f, 0.03125f);
    }
    destination[index] = a + z;
}
