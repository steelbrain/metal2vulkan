// metal-std: metal4.0
// metal-macos-min: 26.2
// Owned fixture: where MetalPerformancePrimitives' matmul2d places the elements of a cooperative
// left input, and of the destination and right input of the 16 x 32 x 32 descriptor, over one
// simdgroup. Function constant 0 (0 when absent) selects a descriptor, element types, and an
// operand. Each thread writes the tensor's capacity to out[tid] and, for each element i it holds,
// one word at out[threads + tid * capacity + i] (while that index is below `limit`): bit 31 when
// the element is valid, bits 16 to 30 its byte offset from element 0 (0x7FFF when no slot below
// the capacity holds it), and for a valid element its coordinates, the first in bits 0 to 7 and
// the second in bits 8 to 15. The offset is found without pointer arithmetic on addresses: every
// element i is set to i + 1, and the storage, read as an array of the element type from element
// 0, is searched for that value. The layout is the implementation's (the header says so), but a
// guest can address elements by offset from element 0, so a translation must place them as Metal
// does. Loops bounded by the capacity.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant uint selected [[function_constant(0)]];
// Without the constant, variant 0.
constant uint variant = is_function_constant_defined(selected) ? selected : 0;

using Mode = matmul2d_descriptor::mode;

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

template <matmul2d_descriptor D, typename L, typename R, typename T>
static void left(device uint *out, uint tid, uint threads, uint limit) {
    matmul2d<D, execution_simdgroup> op;
    auto ct = op.template get_left_input_cooperative_tensor<L, R, T>();
    record(ct, out, tid, threads, limit);
}

template <matmul2d_descriptor D, typename L, typename R, typename T>
static void right(device uint *out, uint tid, uint threads, uint limit) {
    matmul2d<D, execution_simdgroup> op;
    auto ct = op.template get_right_input_cooperative_tensor<L, R, T>();
    record(ct, out, tid, threads, limit);
}

template <matmul2d_descriptor D, typename L, typename R, typename T>
static void destination(device uint *out, uint tid, uint threads, uint limit) {
    matmul2d<D, execution_simdgroup> op;
    using LT = tensor<device L, dextents<int32_t, 2>, tensor_inline>;
    using RT = tensor<device R, dextents<int32_t, 2>, tensor_inline>;
    auto ct = op.template get_destination_cooperative_tensor<LT, RT, T>();
    record(ct, out, tid, threads, limit);
}

constexpr constant matmul2d_descriptor nn16 =
    matmul2d_descriptor(16, 32, 32, false, false, false, Mode::multiply_accumulate);
constexpr constant matmul2d_descriptor nt16 =
    matmul2d_descriptor(16, 32, 32, false, true, false, Mode::multiply_accumulate);
constexpr constant matmul2d_descriptor nn32 =
    matmul2d_descriptor(32, 32, 32, false, false, false, Mode::multiply_accumulate);
constexpr constant matmul2d_descriptor nt32 =
    matmul2d_descriptor(32, 32, 32, false, true, false, Mode::multiply_accumulate);

kernel void tns_coop_left(device uint *out [[buffer(0)]],
                          constant uint &limit [[buffer(1)]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint threads [[threads_per_threadgroup]]) {
    switch (variant) {
    case 0: left<nn16, float, float, float>(out, tid, threads, limit); break;
    case 1: left<nn16, half, half, float>(out, tid, threads, limit); break;
    case 2: left<nn16, bfloat, bfloat, float>(out, tid, threads, limit); break;
    case 3: left<nt16, float, float, float>(out, tid, threads, limit); break;
    case 4: left<nt16, half, half, float>(out, tid, threads, limit); break;
    case 5: left<nt16, bfloat, bfloat, float>(out, tid, threads, limit); break;
    case 6: left<nn32, float, float, float>(out, tid, threads, limit); break;
    case 7: left<nn32, half, half, float>(out, tid, threads, limit); break;
    case 8: left<nn32, bfloat, bfloat, float>(out, tid, threads, limit); break;
    case 9: left<nt32, float, float, float>(out, tid, threads, limit); break;
    case 10: left<nt32, half, half, float>(out, tid, threads, limit); break;
    case 11: left<nt32, bfloat, bfloat, float>(out, tid, threads, limit); break;
    case 12: right<nn16, float, float, float>(out, tid, threads, limit); break;
    case 13: right<nn16, half, half, float>(out, tid, threads, limit); break;
    case 14: right<nn16, bfloat, bfloat, float>(out, tid, threads, limit); break;
    case 15: right<nt16, float, float, float>(out, tid, threads, limit); break;
    case 16: right<nt16, half, half, float>(out, tid, threads, limit); break;
    case 17: right<nt16, bfloat, bfloat, float>(out, tid, threads, limit); break;
    case 18: destination<nn16, float, float, float>(out, tid, threads, limit); break;
    case 19: destination<nn16, half, half, float>(out, tid, threads, limit); break;
    case 20: destination<nn16, bfloat, bfloat, float>(out, tid, threads, limit); break;
    default: destination<nt16, half, half, float>(out, tid, threads, limit); break;
    }
}
