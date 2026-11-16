// metal-macos-min: 27.0
// air-rename: _Z13sgm16_f32_f32Dv8_fbS_bS_ air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.v8f32.v8f32.v8f32
// air-rename: _Z13sgm16_f16_f16Dv8_DhbS_bDv8_f air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.v8f16.v8f16.v8f32
// air-rename: _Z9sgm16_s_sDv8_cbS_bDv8_i air.simdgroup_matrix_16x16x16_widening_multiply_accumulate.s.s.v8i32.v8i8.v8i8.v8i32
// Owned fixture: the 16x16x16 simdgroup matrix products,
// `air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.<A>.<B>.v8f32` and
// `air.simdgroup_matrix_16x16x16_widening_multiply_accumulate.<sa>.<sb>.v8i32.v8i8.v8i8.v8i32`:
// `<8 x R> (<8 x A>, i1, <8 x B>, i1, <8 x R>)`, each lane's eight cells of `A`, its transpose
// flag, of `B`, its flag, and of the accumulator. One form of each class: `float` factors, `half`
// factors, and signed 8-bit integers widened to `int`, each with the four flag pairs. MSL has no
// spelling for them (the frontend's builtins are the 8x8 ones), so the source declares one function
// per form and never defines it, and the directives above rename every call to the intrinsic in
// both companions (scripts/regen-air-fixtures). The eight-lane vectors are declared here (`float8`
// is ambiguous in this toolchain's headers).
//
// Metal makes these kernels' pipelines only on an Apple GPU of family 10 and later: an M3 Max under
// macOS 27.0.1 refuses every one ("simdgroup_matrix<T,16,16x16> operations are supported by
// GPUFamily10 and later"; crates/metal2vulkan-conformance/probes/simd_matrix16.m). Written for this
// repository; no third-party source.
#include <metal_stdlib>

typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef half f16x8 __attribute__((ext_vector_type(8)));
typedef char i8x8 __attribute__((ext_vector_type(8)));
typedef int i32x8 __attribute__((ext_vector_type(8)));

f32x8 sgm16_f32_f32(f32x8 a, bool ta, f32x8 b, bool tb, f32x8 c);
f32x8 sgm16_f16_f16(f16x8 a, bool ta, f16x8 b, bool tb, f32x8 c);
i32x8 sgm16_s_s(i8x8 a, bool ta, i8x8 b, bool tb, i32x8 c);

// Kernel `m16_<form>_<ta><tb>`: thread `t` passes element `t` of `a`, `b` and `c` as its lane's
// cells, with the constant flags `ta` and `tb`, and writes the product's cells to element `t` of
// `d`. A threadgroup of 32 threads is one simdgroup; simdgroup `s` is threads [32s, 32s + 32).
#define M16(form, AT, BT, RT, ta, tb, tag)                                                         \
    kernel void m16_##form##_##tag(device const AT *a [[buffer(0)]],                             \
                                   device const BT *b [[buffer(1)]],                             \
                                   device const RT *c [[buffer(2)]],                             \
                                   device RT *d [[buffer(3)]],                                   \
                                   uint t [[thread_position_in_grid]]) {                         \
        d[t] = sgm16_##form(a[t], ta, b[t], tb, c[t]);                                           \
    }
#define M16_ALL(form, AT, BT, RT)                                                                  \
    M16(form, AT, BT, RT, false, false, 00)                                                        \
    M16(form, AT, BT, RT, false, true, 01)                                                         \
    M16(form, AT, BT, RT, true, false, 10)                                                         \
    M16(form, AT, BT, RT, true, true, 11)

M16_ALL(f32_f32, f32x8, f32x8, f32x8)
M16_ALL(f16_f16, f16x8, f16x8, f32x8)
M16_ALL(s_s, i8x8, i8x8, i32x8)
