// metal-std: metal4.1
// metal-macos-min: 27.0
// Owned fixture: matmul2d with a cooperative left input, built for macOS 27 (each run in the
// header's `_v2` form, no scale factors and dynamic extents), over the descriptors macOS 27's
// gather matmul kernels use: 16 x 32 x 32 and 32 x 32 x 32, B transposed or not, one simdgroup.
//
// `tns_mm_left`: threadgroup t (32 threads, one simdgroup) computes one M x 32 tile,
// out[32 M t ...] = A * B + C, through a cooperative destination of float initialized from C by
// coordinate. tiles[t] names the blocks it reads: A is block tiles[t].x of a (M x 32, row-major,
// M * 32 elements a block), a cooperative left input filled by coordinate; B block tiles[t].y of b
// (32 x 32, row-major, 1024 elements; transposed, its rows are B's columns), in device memory,
// staged into threadgroup memory, or a cooperative right input filled by coordinate; C block
// tiles[t].z of c (M x 32 floats). Function constant 0 (0 when absent) selects M, the transpose,
// where B lives, and the element type of A and B (float, half, bfloat).
//
// `tns_init_left`: what constructing a cooperative left input leaves in its storage, as
// `tns_init` reads a destination's and a right input's. For each of `rounds` rounds (a buffer, so
// the loop stays a loop), each thread constructs the left input in the loop's body and writes the
// bits of each element i it holds (at most 64) to out[(round * threads + tid) * 64 + i]; then it
// sets element i to round * 32 + i + 1 and writes the sum of its elements, read back, to
// out[rounds * threads * 64 + round * threads + tid]. Function constant 0 selects M and the
// element type. Loops bounded by the capacity.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant uint selected [[function_constant(0)]];
// Without the constant, variant 0.
constant uint variant = is_function_constant_defined(selected) ? selected : 0;

using Mode = matmul2d_descriptor::mode;
constexpr constant matmul2d_descriptor nn16 =
    matmul2d_descriptor(16, 32, 32, false, false, false, Mode::multiply_accumulate);
constexpr constant matmul2d_descriptor nt16 =
    matmul2d_descriptor(16, 32, 32, false, true, false, Mode::multiply_accumulate);
constexpr constant matmul2d_descriptor nn32 =
    matmul2d_descriptor(32, 32, 32, false, false, false, Mode::multiply_accumulate);
constexpr constant matmul2d_descriptor nt32 =
    matmul2d_descriptor(32, 32, 32, false, true, false, Mode::multiply_accumulate);

enum Where { Device, Threadgroup, Cooperative };

template <typename E, matmul2d_descriptor D, Where R>
static void tile(device const uchar *a_bytes, device const uchar *b_bytes, uint3 blocks,
                 device const float *c, device float *out, threadgroup E *staged, uint lane) {
    using DT = tensor<device E, dextents<int32_t, 2>, tensor_inline>;
    using TT = tensor<threadgroup E, dextents<int32_t, 2>, tensor_inline>;
    device const E *a = (device const E *)a_bytes + blocks.x * D.m * 32;
    device E *b = const_cast<device E *>((device const E *)b_bytes) + blocks.y * 1024;
    c += blocks.z * D.m * 32;
    // B's extents: (columns, rows) of b as it is stored.
    dextents<int32_t, 2> b_extents = dextents<int32_t, 2>(32, 32);
    matmul2d<D, execution_simdgroup> op;
    auto lt = op.template get_left_input_cooperative_tensor<E, E, float>();
    for (ushort i = 0; i < lt.get_capacity(); ++i) {
        if (lt.is_valid_element(i)) {
            auto idx = lt.get_multidimensional_index(i);
            lt[i] = a[idx[1] * 32 + idx[0]];
        }
    }
    auto ct = op.template get_destination_cooperative_tensor<DT, DT, float>();
    for (ushort i = 0; i < ct.get_capacity(); ++i) {
        if (ct.is_valid_element(i)) {
            auto idx = ct.get_multidimensional_index(i);
            ct[i] = c[idx[1] * 32 + idx[0]];
        }
    }
    if constexpr (R == Device) {
        DT B(b, b_extents);
        op.run(lt, B, ct);
    } else if constexpr (R == Threadgroup) {
        for (uint i = lane; i < 1024; i += 32)
            staged[i] = b[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        TT B(staged, b_extents);
        op.run(lt, B, ct);
    } else {
        auto rt = op.template get_right_input_cooperative_tensor<E, E, float>();
        for (ushort i = 0; i < rt.get_capacity(); ++i) {
            if (rt.is_valid_element(i)) {
                auto idx = rt.get_multidimensional_index(i);
                rt[i] = b[idx[1] * 32 + idx[0]];
            }
        }
        op.run(lt, rt, ct);
    }
    for (ushort i = 0; i < ct.get_capacity(); ++i) {
        if (ct.is_valid_element(i)) {
            auto idx = ct.get_multidimensional_index(i);
            out[idx[1] * 32 + idx[0]] = ct[i];
        }
    }
}

template <typename E>
static void products(device const uchar *a, device const uchar *b, uint3 blocks,
                     device const float *c, device float *out, threadgroup E *staged, uint lane,
                     uint v) {
    switch (v) {
    case 0: tile<E, nn16, Device>(a, b, blocks, c, out, staged, lane); break;
    case 1: tile<E, nt16, Device>(a, b, blocks, c, out, staged, lane); break;
    case 2: tile<E, nn32, Device>(a, b, blocks, c, out, staged, lane); break;
    case 3: tile<E, nt32, Device>(a, b, blocks, c, out, staged, lane); break;
    case 4: tile<E, nn16, Threadgroup>(a, b, blocks, c, out, staged, lane); break;
    case 5: tile<E, nt16, Threadgroup>(a, b, blocks, c, out, staged, lane); break;
    case 6: tile<E, nn32, Threadgroup>(a, b, blocks, c, out, staged, lane); break;
    case 7: tile<E, nt32, Threadgroup>(a, b, blocks, c, out, staged, lane); break;
    case 8: tile<E, nn16, Cooperative>(a, b, blocks, c, out, staged, lane); break;
    case 9: tile<E, nt16, Cooperative>(a, b, blocks, c, out, staged, lane); break;
    case 10: tile<E, nn32, Cooperative>(a, b, blocks, c, out, staged, lane); break;
    default: tile<E, nt32, Cooperative>(a, b, blocks, c, out, staged, lane); break;
    }
}

// Variants 0 to 11 are float, 12 to 23 half, 24 to 35 bfloat, each `products`' twelve: M 16 or 32,
// B not transposed or transposed, in device memory (0 to 3), threadgroup memory (4 to 7), or a
// cooperative right input (8 to 11).
kernel void tns_mm_left(device const uchar *a [[buffer(0)]],
                        device const uchar *b [[buffer(1)]],
                        device const float *c [[buffer(2)]],
                        device float *out [[buffer(3)]],
                        constant uint4 *tiles [[buffer(4)]],
                        uint t [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float staged_f[1024];
    threadgroup half staged_h[1024];
    threadgroup bfloat staged_b[1024];
    uint3 blocks = tiles[t].xyz;
    // M is 16 for the variants whose `products` index is 0, 1, 4, 5, 8 or 9, else 32.
    uint v = variant % 12;
    uint m = (v % 4) < 2 ? 16 : 32;
    out += t * m * 32;
    if (variant < 12)
        products<float>(a, b, blocks, c, out, staged_f, lane, v);
    else if (variant < 24)
        products<half>(a, b, blocks, c, out, staged_h, lane, v);
    else
        products<bfloat>(a, b, blocks, c, out, staged_b, lane, v);
}

constant uint MAX_HELD = 64;

static uint bits(float v) { return as_type<uint>(v); }
static uint bits(half v) { return uint(as_type<ushort>(v)); }
static uint bits(bfloat v) { return uint(as_type<ushort>(v)); }

template <matmul2d_descriptor D, typename E>
static void constructed(device uint *out, uint tid, uint threads, uint rounds) {
    matmul2d<D, execution_simdgroup> op;
    for (uint r = 0; r < rounds; ++r) {
        auto ct = op.template get_left_input_cooperative_tensor<E, E, float>();
        uint cap = min(uint(ct.get_capacity()), MAX_HELD);
        device uint *held = out + (r * threads + tid) * MAX_HELD;
        for (uint i = 0; i < cap; ++i)
            held[i] = bits(E(ct[i]));
        for (uint i = 0; i < cap; ++i)
            ct[i] = E(r * 32 + i + 1);
        uint sum = 0;
        for (uint i = 0; i < cap; ++i)
            sum += uint(float(E(ct[i])));
        out[rounds * threads * MAX_HELD + r * threads + tid] = sum;
    }
}

// Variants 0 to 2: M 16 in float, half, bfloat; 3 to 5: M 32.
kernel void tns_init_left(device uint *out [[buffer(0)]],
                          constant uint &rounds [[buffer(1)]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint threads [[threads_per_threadgroup]]) {
    switch (variant) {
    case 0: constructed<nn16, float>(out, tid, threads, rounds); break;
    case 1: constructed<nn16, half>(out, tid, threads, rounds); break;
    case 2: constructed<nn16, bfloat>(out, tid, threads, rounds); break;
    case 3: constructed<nn32, float>(out, tid, threads, rounds); break;
    case 4: constructed<nn32, half>(out, tid, threads, rounds); break;
    default: constructed<nn32, bfloat>(out, tid, threads, rounds); break;
    }
}
