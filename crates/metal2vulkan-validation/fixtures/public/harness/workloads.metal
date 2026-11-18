// Owned fixture: the kernels of the harness's authored workloads (crates/metal2vulkan-harness), each
// standing for a class of compute work: bandwidth-bound, ALU-bound, threadgroup memory with
// barriers, SIMD reductions, 2D image-like indexing, many tiny dispatches in a chain, narrow
// integer arithmetic, sampling through a runtime sampler, through a constexpr one and over a grid
// more than one deep, products by a constant zero of either sign, many buffers bound for each of
// many dispatches, many one-wave threadgroups that each do almost nothing, reads of a table of
// constants, and a grid's cells indexed by row, with the control the harness's tests hold beside
// it; and one kernel the new runtime's device tests run against Metal's reading of it
// (row_barrier). Written from scratch for it;
// `scripts/regen-air-fixtures/regen-air-fixtures.sh --dir harness` compiles it. The harness holds a
// CPU reference of every workload's result. No translator sweep reads this directory.
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

// Cells of a row-major grid: each thread inside the grid's width and height reads its word at
// y * stride + x and writes three words at (y * width + x) * 3, the rest return first, as
// row-indexed kernels are commonly written. Two threads of one simdgroup in different rows may
// meet for all the IR shows, so the lockstep order puts barriers between the stores, in the
// block past the bounds test, which the threads of one simdgroup may not all run, so they are
// simdgroup barriers; where each simdgroup lies within one row of its threadgroup they never
// meet. There are two forms. row_cells computes the cells in int: it compiles to steps that carry
// nsw, and from them the order proves two threads of one row apart, so its one-row variant needs
// no barrier. row_cells_uint, the control, computes them in uint: its steps carry no flags and
// may wrap, so the order does not prove them apart in rows and its one-row variant keeps the
// barriers.
kernel void row_cells(device const float *source [[buffer(0)]],
                      device float *destination [[buffer(1)]],
                      constant int &width [[buffer(2)]],
                      constant int &height [[buffer(3)]],
                      constant int &stride [[buffer(4)]],
                      uint2 position [[thread_position_in_grid]]) {
    int x = int(position.x);
    int y = int(position.y);
    if (x >= width || y >= height) {
        return;
    }
    float value = source[y * stride + x];
    int cell = y * width + x;
    destination[3 * cell] = value;
    destination[3 * cell + 1] = value * 2.0f;
    destination[3 * cell + 2] = value + 1.0f;
}

kernel void row_cells_uint(device const float *source [[buffer(0)]],
                           device float *destination [[buffer(1)]],
                           constant uint &width [[buffer(2)]],
                           constant uint &height [[buffer(3)]],
                           constant uint &stride [[buffer(4)]],
                           uint2 position [[thread_position_in_grid]]) {
    uint x = position.x;
    uint y = position.y;
    if (x >= width || y >= height) {
        return;
    }
    float value = source[y * stride + x];
    uint cell = y * width + x;
    destination[3 * cell] = value;
    destination[3 * cell + 1] = value * 2.0f;
    destination[3 * cell + 2] = value + 1.0f;
}
// A grid's cells written, then read across rows after a threadgroup barrier: each thread stores
// its own word, waits at threadgroup_barrier(mem_flags::mem_device), and reads the word the thread
// of its column in the other row of its pair (y ^ 1) stored, where that row is in the grid. The
// barrier orders device memory among the threads of one threadgroup, so in threadgroups an even
// number of rows high, aligned to the pairs, every read sees the other row's word. The kernel holds
// no threadgroup memory, and observes its partition through the barrier: run in tiles of one row,
// the pairs would split. The probe metal2vulkan-conformance/probes/row_barrier_reads.m runs the
// same source on Metal.
kernel void row_barrier(device uint *cells [[buffer(0)]],
                        device uint *seen [[buffer(1)]],
                        uint2 position [[thread_position_in_grid]],
                        uint2 grid [[threads_per_grid]]) {
    uint cell = position.y * grid.x + position.x;
    cells[cell] = ((position.y << 16) | position.x) + 1;
    threadgroup_barrier(mem_flags::mem_device);
    uint other = position.y ^ 1;
    if (other < grid.y) {
        seen[cell] = cells[other * grid.x + position.x];
    }
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

// Sampling over a grid more than one deep: a 2x downsample of the source into each layer's own
// rows of the destination, layer `z` at row `z` times the source's height over 2, so every thread
// writes a texel of its own; one texture write a thread.
kernel void sample_down_layers(texture2d<float, access::sample> source [[texture(0)]],
                               texture2d<float, access::write> destination [[texture(1)]],
                               sampler filter [[sampler(0)]],
                               uint3 position [[thread_position_in_grid]]) {
    uint height = source.get_height() / 2;
    float2 size = float2(destination.get_width(), height);
    float4 texel = source.sample(filter, (float2(position.xy) + 0.5f) / size);
    destination.write(texel, uint2(position.x, position.y + position.z * height));
}

// Sampling over a grid more than one deep, each layer from its own source: layer `z` is a 2x
// downsample of its own rows of the source into its own rows of the destination (both `z` times
// a layer's height on, the destination's layers as tall as it is wide), so no two layers read
// the same texels; one texture write a thread.
kernel void sample_down_distinct(texture2d<float, access::sample> source [[texture(0)]],
                                 texture2d<float, access::write> destination [[texture(1)]],
                                 sampler filter [[sampler(0)]],
                                 uint3 position [[thread_position_in_grid]]) {
    uint side = destination.get_width();
    float2 size = float2(side, destination.get_height());
    uint2 at = uint2(position.x, position.y + position.z * side);
    destination.write(source.sample(filter, (float2(at) + 0.5f) / size), at);
}

// Many buffers bound for one kernel: each thread folds one word of each of 23 inputs into the
// 24th, so a dispatch reads 24 buffer addresses, and many small dispatches each bind all 24.
kernel void wide_bindings(device const uint *b0 [[buffer(0)]],
                          device const uint *b1 [[buffer(1)]],
                          device const uint *b2 [[buffer(2)]],
                          device const uint *b3 [[buffer(3)]],
                          device const uint *b4 [[buffer(4)]],
                          device const uint *b5 [[buffer(5)]],
                          device const uint *b6 [[buffer(6)]],
                          device const uint *b7 [[buffer(7)]],
                          device const uint *b8 [[buffer(8)]],
                          device const uint *b9 [[buffer(9)]],
                          device const uint *b10 [[buffer(10)]],
                          device const uint *b11 [[buffer(11)]],
                          device const uint *b12 [[buffer(12)]],
                          device const uint *b13 [[buffer(13)]],
                          device const uint *b14 [[buffer(14)]],
                          device const uint *b15 [[buffer(15)]],
                          device const uint *b16 [[buffer(16)]],
                          device const uint *b17 [[buffer(17)]],
                          device const uint *b18 [[buffer(18)]],
                          device const uint *b19 [[buffer(19)]],
                          device const uint *b20 [[buffer(20)]],
                          device const uint *b21 [[buffer(21)]],
                          device const uint *b22 [[buffer(22)]],
                          device uint *destination [[buffer(23)]],
                          uint index [[thread_position_in_grid]]) {
    uint s = b0[index];
    s = s * 3u + b1[index];
    s = s * 3u + b2[index];
    s = s * 3u + b3[index];
    s = s * 3u + b4[index];
    s = s * 3u + b5[index];
    s = s * 3u + b6[index];
    s = s * 3u + b7[index];
    s = s * 3u + b8[index];
    s = s * 3u + b9[index];
    s = s * 3u + b10[index];
    s = s * 3u + b11[index];
    s = s * 3u + b12[index];
    s = s * 3u + b13[index];
    s = s * 3u + b14[index];
    s = s * 3u + b15[index];
    s = s * 3u + b16[index];
    s = s * 3u + b17[index];
    s = s * 3u + b18[index];
    s = s * 3u + b19[index];
    s = s * 3u + b20[index];
    s = s * 3u + b21[index];
    s = s * 3u + b22[index];
    destination[index] = s;
}

// The fixed cost of a wave: threadgroups of 32 threads, one wave each, in a grid 5,600 threads
// wide, each thread one load from a constant buffer argument and one 4-byte store.
kernel void wave_store(device uint *destination [[buffer(0)]],
                       constant uint &base [[buffer(1)]],
                       uint2 position [[thread_position_in_grid]]) {
    uint index = position.y * 5600u + position.x;
    destination[index] = base + index;
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

// A SIMD-group barrier inside a loop, at one SIMD group per threadgroup: every thread of a
// threadgroup reads the same count of rounds from `steps`, which the translation cannot prove
// uniform, and steps an LCG that many rounds with a `simdgroup_barrier` in each; it writes its
// state and its index in its SIMD group. Threadgroups of 32 threads.
kernel void vote_loop(device const uint *steps [[buffer(0)]],
                      device uint2 *destination [[buffer(1)]],
                      uint index [[thread_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]]) {
    uint rounds = steps[index];
    uint state = index;
    for (uint r = 0; r < rounds; ++r) {
        state = state * 1664525u + 1013904223u;
        simdgroup_barrier(mem_flags::mem_none);
    }
    destination[index] = uint2(state, lane);
}

// Each threadgroup's sum of its threads' words, through threadgroup memory whose length the host
// sets (setThreadgroupMemoryLength:atIndex:), a word a thread; threadgroups of a power of two.
kernel void group_sums(device const uint *source [[buffer(0)]],
                       device uint *sums [[buffer(1)]],
                       threadgroup uint *scratch [[threadgroup(0)]],
                       uint at [[thread_position_in_threadgroup]],
                       uint size [[threads_per_threadgroup]],
                       uint group [[threadgroup_position_in_grid]],
                       uint index [[thread_position_in_grid]]) {
    scratch[at] = source[index];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint half_size = size / 2; half_size > 0; half_size /= 2) {
        if (at < half_size) {
            scratch[at] += scratch[at + half_size];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (at == 0) {
        sums[group] = scratch[0];
    }
}

// Reads of a 64 KiB table in the constant address space, as a program reads a constant buffer it
// updates a little before each frame: each thread folds 64 of the table's 4,096 entries into one
// word, the threads of a threadgroup reading consecutive entries.
kernel void constant_reads(constant uint4 *table [[buffer(0)]],
                           device uint *destination [[buffer(1)]],
                           uint index [[thread_position_in_grid]]) {
    uint folded = index;
    for (uint i = 0; i < 64; ++i) {
        uint4 entry = table[(index + i * 67) & 4095];
        folded = (folded ^ entry.x ^ entry.y) + (entry.z ^ entry.w);
    }
    destination[index] = folded;
}
