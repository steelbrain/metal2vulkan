// metal-std: metal4.1
// metal-macos-min: 27.0
// Owned fixture: tns_mm_float built for macOS 27, where the header spells each run in its `_v2`
// form (each tensor operand's scale type, scale block sizes and static extents follow it: no
// scale factors and dynamic extents here), with two more right inputs: variants 10 to 12 a
// transposed cooperative right input (float, half, bfloat) and 13 to 15 a transposed one in
// threadgroup memory. What follows is tns_mm_float's description. Threadgroup t (32 threads, one
// simdgroup) computes one 16 x 16 tile, out[256 t ...] = A * B + C, through a cooperative
// destination of float initialized from C by coordinate. tiles[t] names the blocks it reads: A is
// block tiles[t].x of a (16 x 32, extents 32, 16), B block tiles[t].y of b (32 x 16, extents 16,
// 32; transposed, extents 32, 16), and C block tiles[t].z of c (16 x 16), each block 512 (C: 256)
// elements. Function constant 0 (0 when absent) selects the element type of A and B (float, half,
// bfloat), whether B is transposed, and where B lives (device, threadgroup, or a cooperative right
// input filled by coordinate). The cases choose products that expose the order of the sum (pairs of
// +-2^30 among small terms), where C enters it, and whether a product is rounded before it is
// added. Loops bounded by the capacity.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant uint selected [[function_constant(0)]];
// Without the constant, variant 0.
constant uint variant = is_function_constant_defined(selected) ? selected : 0;

using Mode = matmul2d_descriptor::mode;
constexpr constant matmul2d_descriptor nn =
    matmul2d_descriptor(16, 16, 32, false, false, false, Mode::multiply_accumulate);
constexpr constant matmul2d_descriptor nt =
    matmul2d_descriptor(16, 16, 32, false, true, false, Mode::multiply_accumulate);

enum Where { Device, Threadgroup, Cooperative };

template <typename E, matmul2d_descriptor D, Where R>
static void tile(device const uchar *a_bytes, device const uchar *b_bytes, uint3 blocks,
                 device const float *c, device float *out, threadgroup E *staged, uint lane) {
    using DT = tensor<device E, dextents<int32_t, 2>, tensor_inline>;
    using TT = tensor<threadgroup E, dextents<int32_t, 2>, tensor_inline>;
    device E *a = const_cast<device E *>((device const E *)a_bytes) + blocks.x * 512;
    device E *b = const_cast<device E *>((device const E *)b_bytes) + blocks.y * 512;
    c += blocks.z * 256;
    DT A(a, dextents<int32_t, 2>(32, 16));
    dextents<int32_t, 2> b_extents = D.transpose_right ? dextents<int32_t, 2>(32, 16)
                                                       : dextents<int32_t, 2>(16, 32);
    matmul2d<D, execution_simdgroup> op;
    auto ct = op.template get_destination_cooperative_tensor<DT, DT, float>();
    for (ushort i = 0; i < ct.get_capacity(); ++i) {
        if (ct.is_valid_element(i)) {
            auto idx = ct.get_multidimensional_index(i);
            ct[i] = c[idx[1] * 16 + idx[0]];
        }
    }
    if constexpr (R == Device) {
        DT B(b, b_extents);
        op.run(A, B, ct);
    } else if constexpr (R == Threadgroup) {
        for (uint i = lane; i < 512; i += 32)
            staged[i] = b[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        TT B(staged, b_extents);
        op.run(A, B, ct);
    } else {
        auto rt = op.template get_right_input_cooperative_tensor<E, E, float>();
        int e0 = b_extents.extent(0);
        for (ushort i = 0; i < rt.get_capacity(); ++i) {
            if (rt.is_valid_element(i)) {
                auto idx = rt.get_multidimensional_index(i);
                rt[i] = b[idx[1] * e0 + idx[0]];
            }
        }
        op.run(A, rt, ct);
    }
    for (ushort i = 0; i < ct.get_capacity(); ++i) {
        if (ct.is_valid_element(i)) {
            auto idx = ct.get_multidimensional_index(i);
            out[idx[1] * 16 + idx[0]] = ct[i];
        }
    }
}

kernel void tns_mm_v2(device const uchar *a [[buffer(0)]],
                      device const uchar *b [[buffer(1)]],
                      device const float *c [[buffer(2)]],
                      device float *out [[buffer(3)]],
                      constant uint4 *tiles [[buffer(4)]],
                      uint t [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float staged_f[512];
    threadgroup half staged_h[512];
    threadgroup bfloat staged_b[512];
    uint3 blocks = tiles[t].xyz;
    out += t * 256;
    switch (variant) {
    case 0: tile<float, nn, Device>(a, b, blocks, c, out, staged_f, lane); break;
    case 1: tile<half, nn, Device>(a, b, blocks, c, out, staged_h, lane); break;
    case 2: tile<bfloat, nn, Device>(a, b, blocks, c, out, staged_b, lane); break;
    case 3: tile<float, nt, Device>(a, b, blocks, c, out, staged_f, lane); break;
    case 4: tile<half, nt, Device>(a, b, blocks, c, out, staged_h, lane); break;
    case 5: tile<bfloat, nt, Device>(a, b, blocks, c, out, staged_b, lane); break;
    case 6: tile<float, nn, Cooperative>(a, b, blocks, c, out, staged_f, lane); break;
    case 7: tile<half, nn, Cooperative>(a, b, blocks, c, out, staged_h, lane); break;
    case 8: tile<bfloat, nn, Cooperative>(a, b, blocks, c, out, staged_b, lane); break;
    case 10: tile<float, nt, Cooperative>(a, b, blocks, c, out, staged_f, lane); break;
    case 11: tile<half, nt, Cooperative>(a, b, blocks, c, out, staged_h, lane); break;
    case 12: tile<bfloat, nt, Cooperative>(a, b, blocks, c, out, staged_b, lane); break;
    case 13: tile<float, nt, Threadgroup>(a, b, blocks, c, out, staged_f, lane); break;
    case 14: tile<half, nt, Threadgroup>(a, b, blocks, c, out, staged_h, lane); break;
    case 15: tile<bfloat, nt, Threadgroup>(a, b, blocks, c, out, staged_b, lane); break;
    default: tile<float, nn, Threadgroup>(a, b, blocks, c, out, staged_f, lane); break;
    }
}
