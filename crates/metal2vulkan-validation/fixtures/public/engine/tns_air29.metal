// metal-std: metal4.1
// metal-macos-min: 27.0
// Owned fixture: the private tensor intrinsics in AIR 2.9, run without a matmul2d. Built for macOS
// 27, Apple's compiler declares `air.init_strided_private_tensor` with three more operands than AIR
// 2.8 and `air.get_extent_private_tensor` with one more. Thread t of 32 initializes a tensor over
// device memory `a` (extents ext[0], ext[1]) and one over threadgroup memory (extents ext[2],
// ext[3]), so both the device (`global`) and the threadgroup (`local`) initialization run, with
// extents only the run knows, and writes the extents they answer: out[4t + k] = ext[k] (32, 16, 16
// and 32 from the probe's buffer).
#include <metal_stdlib>
#include <metal_tensor>
using namespace metal;

kernel void tns_air29(device const int8_t *a [[buffer(0)]],
                      device const int *ext [[buffer(1)]],
                      device int *out [[buffer(2)]],
                      uint tid [[thread_index_in_threadgroup]]) {
    threadgroup int8_t tb[16 * 32];
    using DT = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>;
    using TT = tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline>;
    DT tA(const_cast<device int8_t *>(a), dextents<int32_t, 2>(ext[0], ext[1]));
    TT tB(tb, dextents<int32_t, 2>(ext[2], ext[3]));
    out[4 * tid + 0] = tA.get_extent(0);
    out[4 * tid + 1] = tA.get_extent(1);
    out[4 * tid + 2] = tB.get_extent(0);
    out[4 * tid + 3] = tB.get_extent(1);
}
