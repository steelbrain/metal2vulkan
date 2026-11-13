// metal-std: metal4.1
// metal-macos-min: 27.0
// Owned fixture: matmul2d of 8-bit floats (`metal_fp8_e4m3_format`, `metal_fp8_e5m2_format`) into
// a cooperative float destination, built for macOS 27 (each run in the header's `_v2` form, no
// scale factors and dynamic extents), over one descriptor: 32 x 32 x 32 on one simdgroup.
//
// `tns_mm_fp8`: threadgroup t (32 threads, one simdgroup) computes one 32 x 32 tile,
// out[1024 t ...] = A * B (+ C in multiply_accumulate mode), through a cooperative destination of
// float initialized from C by coordinate (in multiply mode the product ignores it). tiles[t] names
// the blocks it reads: A is block tiles[t].x of a (32 x 32 bytes, row-major), B block tiles[t].y
// of b (32 x 32 bytes, row-major; transposed, its rows are B's columns), C block tiles[t].z of c
// (32 x 32 floats). Each byte is one element of the variant's format. Function constant 0 (0 when
// absent) selects the format, the transpose, the mode, and where A and B live (device memory, or
// staged from it into threadgroup memory).
//
// `tns_coop_fp8`: where matmul2d places the elements of the destination of an 8-bit product
// (32 x 32 x 32, B transposed or not), measured as `tns_coop_left` measures them: each thread
// writes the tensor's capacity to out[tid] and, for each element i it holds, one word at
// out[threads + tid * capacity + i] (while that index is below `limit`): bit 31 when the element
// is valid, bits 16 to 30 its byte offset from element 0 (0x7FFF when no slot below the capacity
// holds it), the first coordinate in bits 0 to 7 and the second in bits 8 to 15. Loops bounded by
// the capacity.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant uint selected [[function_constant(0)]];
// Without the constant, variant 0.
constant uint variant = is_function_constant_defined(selected) ? selected : 0;

using Mode = matmul2d_descriptor::mode;
constexpr constant matmul2d_descriptor nn_acc =
    matmul2d_descriptor(32, 32, 32, false, false, false, Mode::multiply_accumulate);
constexpr constant matmul2d_descriptor nt_acc =
    matmul2d_descriptor(32, 32, 32, false, true, false, Mode::multiply_accumulate);
constexpr constant matmul2d_descriptor nn_mul =
    matmul2d_descriptor(32, 32, 32, false, false, false, Mode::multiply);
constexpr constant matmul2d_descriptor nt_mul =
    matmul2d_descriptor(32, 32, 32, false, true, false, Mode::multiply);

enum Where { Device, Threadgroup };

template <typename F, Where W>
struct Operand;

template <typename F>
struct Operand<F, Device> {
    using T = tensor<device F, dextents<int32_t, 2>, tensor_inline>;
    static T make(device uchar *bytes, threadgroup uchar *, uint, dextents<int32_t, 2> e) {
        return T(bytes, e);
    }
};

template <typename F>
struct Operand<F, Threadgroup> {
    using T = tensor<threadgroup F, dextents<int32_t, 2>, tensor_inline>;
    static T make(device uchar *bytes, threadgroup uchar *staged, uint lane,
                  dextents<int32_t, 2> e) {
        for (uint i = lane; i < 1024; i += 32)
            staged[i] = bytes[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        return T(staged, e);
    }
};

template <typename F, matmul2d_descriptor D, Where WA, Where WB>
static void tile(device const uchar *a_bytes, device const uchar *b_bytes, uint3 blocks,
                 device const float *c, device float *out, threadgroup uchar *staged_a,
                 threadgroup uchar *staged_b, uint lane) {
    device uchar *a = const_cast<device uchar *>(a_bytes) + blocks.x * 1024;
    device uchar *b = const_cast<device uchar *>(b_bytes) + blocks.y * 1024;
    c += blocks.z * 1024;
    auto A = Operand<F, WA>::make(a, staged_a, lane, dextents<int32_t, 2>(32, 32));
    auto B = Operand<F, WB>::make(b, staged_b, lane, dextents<int32_t, 2>(32, 32));
    matmul2d<D, execution_simdgroup> op;
    auto ct = op.template get_destination_cooperative_tensor<decltype(A), decltype(B), float>();
    for (ushort i = 0; i < ct.get_capacity(); ++i) {
        if (ct.is_valid_element(i)) {
            auto idx = ct.get_multidimensional_index(i);
            ct[i] = c[idx[1] * 32 + idx[0]];
        }
    }
    op.run(A, B, ct);
    for (ushort i = 0; i < ct.get_capacity(); ++i) {
        if (ct.is_valid_element(i)) {
            auto idx = ct.get_multidimensional_index(i);
            out[idx[1] * 32 + idx[0]] = ct[i];
        }
    }
}

template <typename F>
static void products(device const uchar *a, device const uchar *b, uint3 blocks,
                     device const float *c, device float *out, threadgroup uchar *sa,
                     threadgroup uchar *sb, uint lane, uint v) {
    switch (v) {
    case 0: tile<F, nn_acc, Device, Device>(a, b, blocks, c, out, sa, sb, lane); break;
    case 1: tile<F, nt_acc, Device, Device>(a, b, blocks, c, out, sa, sb, lane); break;
    case 2: tile<F, nn_mul, Device, Device>(a, b, blocks, c, out, sa, sb, lane); break;
    case 3: tile<F, nt_mul, Device, Device>(a, b, blocks, c, out, sa, sb, lane); break;
    case 4: tile<F, nn_acc, Device, Threadgroup>(a, b, blocks, c, out, sa, sb, lane); break;
    case 5: tile<F, nt_acc, Device, Threadgroup>(a, b, blocks, c, out, sa, sb, lane); break;
    default: tile<F, nt_acc, Threadgroup, Device>(a, b, blocks, c, out, sa, sb, lane); break;
    }
}

// Variants 0 to 6 are e4m3, 7 to 13 e5m2, each `products`' seven: B not transposed or
// transposed, multiply_accumulate or multiply, A and B in device memory (0 to 3); B in
// threadgroup memory (4, 5); A in threadgroup memory, B transposed (6).
kernel void tns_mm_fp8(device const uchar *a [[buffer(0)]],
                       device const uchar *b [[buffer(1)]],
                       device const float *c [[buffer(2)]],
                       device float *out [[buffer(3)]],
                       constant uint4 *tiles [[buffer(4)]],
                       uint t [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_threadgroup]]) {
    threadgroup uchar staged_a[1024];
    threadgroup uchar staged_b[1024];
    uint3 blocks = tiles[t].xyz;
    out += t * 1024;
    if (variant < 7)
        products<metal_fp8_e4m3_format>(a, b, blocks, c, out, staged_a, staged_b, lane, variant);
    else
        products<metal_fp8_e5m2_format>(a, b, blocks, c, out, staged_a, staged_b, lane,
                                        variant - 7);
}

template <typename CT>
static void record(thread CT &ct, device uint *out, uint tid, uint threads, uint limit) {
    using E = typename CT::element_type;
    uint cap = ct.get_capacity();
    out[tid] = cap;
    for (uint i = 0; i < cap; ++i)
        ct[i] = E(i + 1);
    thread E *base = &ct[0];
    for (uint i = 0; i < cap; ++i) {
        uint at = tid * cap + i;
        if (at >= limit)
            break;
        uint offset = 0x7FFFu;
        for (uint slot = 0; slot < cap; ++slot) {
            if (base[slot] == E(i + 1)) {
                offset = slot * uint(sizeof(E));
                break;
            }
        }
        uint word = offset << 16;
        if (ct.is_valid_element(i)) {
            auto idx = ct.get_multidimensional_index(i);
            word |= 0x80000000u | (uint(idx[1]) & 0xFFu) << 8 | (uint(idx[0]) & 0xFFu);
        }
        out[threads + at] = word;
    }
}

template <matmul2d_descriptor D, typename F>
static void destination(device uint *out, uint tid, uint threads, uint limit) {
    matmul2d<D, execution_simdgroup> op;
    using T = tensor<device F, dextents<int32_t, 2>, tensor_inline>;
    auto ct = op.template get_destination_cooperative_tensor<T, T, float>();
    record(ct, out, tid, threads, limit);
}

// Variants 0 and 1: e4m3, B not transposed or transposed; 2 and 3: e5m2.
kernel void tns_coop_fp8(device uint *out [[buffer(0)]],
                         constant uint &limit [[buffer(1)]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint threads [[threads_per_threadgroup]]) {
    switch (variant) {
    case 0: destination<nn_acc, metal_fp8_e4m3_format>(out, tid, threads, limit); break;
    case 1: destination<nt_acc, metal_fp8_e4m3_format>(out, tid, threads, limit); break;
    case 2: destination<nn_acc, metal_fp8_e5m2_format>(out, tid, threads, limit); break;
    default: destination<nt_acc, metal_fp8_e5m2_format>(out, tid, threads, limit); break;
    }
}
