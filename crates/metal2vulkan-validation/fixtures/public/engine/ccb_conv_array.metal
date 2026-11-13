// Owned fixture: a convolution over `half` texture arrays, for the time a compute pipeline
// state's creation takes over a kernel of a framework convolution's shape and size. Ten function
// constants pick the kernel's size, strides, dilation, offset, input slices, the bias, the edge
// handling and the activation, as such a kernel's constants do; every state of a run is one set
// of values. Each thread writes four output slices at its pixel: a 3x3 path written out tap by tap
// for each of the four, a general path for other sizes, and one of eight activations. Nothing
// dispatches it.
//
// Written for this repository; no third-party source.
#include <metal_stdlib>
using namespace metal;

constant ushort ccb_kernel_width [[function_constant(0)]];
constant ushort ccb_kernel_height [[function_constant(1)]];
constant ushort2 ccb_stride [[function_constant(2)]];
constant ushort2 ccb_dilation [[function_constant(3)]];
constant bool ccb_has_bias [[function_constant(4)]];
constant ushort ccb_activation [[function_constant(5)]];
constant short2 ccb_offset [[function_constant(6)]];
constant ushort ccb_input_slices [[function_constant(7)]];
constant bool ccb_clamp_edges [[function_constant(8)]];
constant float ccb_alpha [[function_constant(9)]];

struct CcbParams {
    uint2 size;
    uint output_slices;
    float scale;
};

// The tap at `p` of slice `s`, zero outside the source unless the edges clamp.
static half4 ccb_tap(texture2d_array<half, access::read> src, int2 p, uint s) {
    int2 limit = int2(src.get_width(), src.get_height()) - 1;
    if (ccb_clamp_edges) {
        p = clamp(p, int2(0), limit);
    } else if (any(p < 0) || any(p > limit)) {
        return half4(0.0h);
    }
    return src.read(uint2(p), s);
}

// One output slice's dot of a tap with four weight rows.
static half4 ccb_dot(half4 v, device const half4 *w) {
    return half4(dot(v, w[0]), dot(v, w[1]), dot(v, w[2]), dot(v, w[3]));
}

static half4 ccb_activate(half4 x) {
    switch (ccb_activation) {
    case 1:
        return max(x, half4(0.0h));
    case 2:
        return select(x * half(ccb_alpha), x, x > half4(0.0h));
    case 3:
        return half4(1.0h) / (half4(1.0h) + exp(-x));
    case 4:
        return tanh(x);
    case 5:
        return clamp(x, half4(0.0h), half4(half(ccb_alpha)));
    case 6:
        return x * clamp(x * half(1.0h / 6.0h) + half4(0.5h), half4(0.0h), half4(1.0h));
    case 7:
        return log(half4(1.0h) + exp(x));
    default:
        return x;
    }
}

#define CCB_TAP3(ky, kx)                                                                        \
    {                                                                                           \
        half4 v = ccb_tap(src, base + int2(kx, ky) * int2(ccb_dilation), s);                    \
        device const half4 *w = weights + ((s * 9 + (ky) * 3 + (kx)) * outputs + group) * 4;    \
        a0 += ccb_dot(v, w + 0);                                                                \
        a1 += ccb_dot(v, w + 4);                                                                \
        a2 += ccb_dot(v, w + 8);                                                                \
        a3 += ccb_dot(v, w + 12);                                                               \
    }

kernel void ccb_conv_array(texture2d_array<half, access::read> src [[texture(0)]],
                           texture2d_array<half, access::write> dst [[texture(1)]],
                           device const half4 *weights [[buffer(0)]],
                           device const half4 *bias [[buffer(1), function_constant(ccb_has_bias)]],
                           constant CcbParams &params [[buffer(2)]],
                           uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= params.size.x || gid.y >= params.size.y) {
        return;
    }
    uint outputs = params.output_slices;
    uint group = gid.z * 4;
    int2 base = int2(gid.xy) * int2(ccb_stride) + int2(ccb_offset);
    half4 a0 = 0.0h, a1 = 0.0h, a2 = 0.0h, a3 = 0.0h;
    if (ccb_kernel_width == 3 && ccb_kernel_height == 3) {
        for (uint s = 0; s < ccb_input_slices; s++) {
            CCB_TAP3(0, 0) CCB_TAP3(0, 1) CCB_TAP3(0, 2)
            CCB_TAP3(1, 0) CCB_TAP3(1, 1) CCB_TAP3(1, 2)
            CCB_TAP3(2, 0) CCB_TAP3(2, 1) CCB_TAP3(2, 2)
        }
    } else {
        for (uint s = 0; s < ccb_input_slices; s++) {
            for (ushort ky = 0; ky < ccb_kernel_height; ky++) {
                for (ushort kx = 0; kx < ccb_kernel_width; kx++) {
                    half4 v = ccb_tap(src, base + int2(kx, ky) * int2(ccb_dilation), s);
                    uint tap = (s * ccb_kernel_height + ky) * ccb_kernel_width + kx;
                    device const half4 *w = weights + (tap * outputs + group) * 4;
                    a0 += ccb_dot(v, w + 0);
                    a1 += ccb_dot(v, w + 4);
                    a2 += ccb_dot(v, w + 8);
                    a3 += ccb_dot(v, w + 12);
                }
            }
        }
    }
    half scale = half(params.scale);
    a0 *= scale;
    a1 *= scale;
    a2 *= scale;
    a3 *= scale;
    if (ccb_has_bias) {
        a0 += bias[group + 0];
        a1 += bias[group + 1];
        a2 += bias[group + 2];
        a3 += bias[group + 3];
    }
    dst.write(ccb_activate(a0), gid.xy, group + 0);
    dst.write(ccb_activate(a1), gid.xy, group + 1);
    dst.write(ccb_activate(a2), gid.xy, group + 2);
    dst.write(ccb_activate(a3), gid.xy, group + 3);
}
