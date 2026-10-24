//! What the device's own blend stores for a fragment function's `half` color output, in a module
//! without fast math, into a 16- and a 32-bit float attachment, against Metal's measured rule
//! there (the rule fits every channel of 78 blended 16-bit float draws on plain Metal, an M3 Max
//! under macOS 26.7):
//!
//! - into a 16-bit float format, Metal blends the `half` the function returns, each factor
//!   (`1 - x` included), each product and the sum (or difference) rounded to `half`, nearest
//!   even; `Min` and `Max` read no factor;
//! - into a 32-bit float format, a multiply-add of the `half` in `float`, each product rounded to
//!   `float`.
//!
//! The functions are a probe of the consumer's data path (our own MSL, compiled with Apple's tools
//! under `-fmetal-math-mode=safe`): a value per pixel, splat into every lane, of a sweep from -0.25
//! to 1.25 (kind 0) or from -2 to 2 (kind 4), returned as `half4`, blended over the destinations
//! (random halves in `[0, 1]`, random `float`s in `[0, 1]`). The source each pixel blends is read
//! back from an unblended draw of the same function, so the device's own computation of the value
//! cannot move the comparison.
//!
//! Each draw prints a `HALF-BLEND` line: the channels that store Metal's rule's value, the
//! channels that store the `float` model's (the device's blend in `float`, each product and the
//! sum rounded to `float`, the result rounded to the format once, nearest even, or toward zero),
//! and the largest distance from Metal's value, in units in the last place of the format at the
//! largest operand's magnitude (the source, the destination, or the two products' magnitudes
//! summed: the recorded bound's measure) and at the stored value's. `HALF-BLEND-SUMMARY` lines
//! give, per format, the share of the blends Metal measured whose every channel is Metal's, and
//! the largest distances. A measurement, not a check: it asserts only that each draw completes
//! and, where the pipeline announces no `half-output-precision`, that every channel is Metal's.
//!
//! The draws run twice: under the device's blend (`HalfBlendPolicy::Device`, the default) into
//! every format, and under Metal's 16-bit float blend computed by the function
//! (`HalfBlendPolicy::Exact`) into the 16-bit float formats. A `HALF-BLEND-EXACT` line counts the
//! blended draws whose pipeline announced nothing, which must store Metal's value in every channel:
//! under the exact blend every one on a device that orders framebuffer fetch through the pixel
//! interlock (NVIDIA, RADV, ANV), none on lavapipe, whose fetch is its input attachments, so the
//! device blends there and the pipeline announces it. The rule reads the blend color's `half`
//! before its one minus (the fit of `RGBA16Float`'s blue lane, a blend color of 0.9: `h(1 - h(c))`)
//! and keeps a term whose factor is `Zero` in the 16-bit float formats' sum, as Metal does there
//! (`+0` where the other term is `-0`, a product of a destination of zero), but
//! drops it in the 32-bit float formats', as Metal's extended-range model does.
//!
//! **The blend color's one minus is no longer this rule on a current OS, and that reading is
//! macOS 26.7's.** The conformance suite's `half_output_blends` family has since ported the four
//! blend-color blends and measured it against committed recordings of plain Metal on an M3 Max
//! under **macOS 27.0.1 (26A434)**: over all 1,792 words of `1mbc-zero` in the three 16-bit float
//! formats, 252 words give the two candidate factors different values and **every one of them
//! stores `h(1 - c)`** -- the complement taken in the `float` the color arrived as and rounded
//! once -- while none stores `h(1 - h(c))`. All 252 are `RGBA16Float`'s blue lane, the very lane
//! this fit came from; at 0.3, 0.6 and 0.45 the two rules agree at every source value, so no
//! one- or two-channel format can tell them apart. The live M3 is the oracle, so a reader wanting
//! the blend color's rule should take the 27.0.1 answer from
//! `metal2vulkan_conformance::cases::half_output_blends` and read the sentence above as this
//! lane's 26.7 measurement. The rest of the fit is untouched by that.

use super::degrade_notices::counts;
use super::{validated_device_with, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::HalfBlendPolicy;
use metal2vulkan::metal::{
    BlendFactor as F, BlendOperation as O, PixelFormat as P, PrimitiveType, TextureUsage,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, PrecisionNote, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    CommandBufferStatus, CommandQueue, Degradation, Device, LoadAction, PipelineOptions, Region,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, StoreAction, TextureDescriptor,
};

/// a probe functions (`hb2.metal`), built under `-fmetal-math-mode=safe`.
const SAFE: &str = r#"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define <{ <4 x float> }> @vs(i32 noundef %0) local_unnamed_addr #0 {
  %2 = alloca [3 x <2 x float>], align 8
  %3 = bitcast ptr %2 to ptr
  call void @llvm.lifetime.start.p0i8(i64 24, ptr nonnull %3) #5
  %4 = getelementptr inbounds [3 x <2 x float>], ptr %2, i64 0, i64 0
  store <2 x float> <float -1.000000e+00, float -1.000000e+00>, ptr %4, align 8
  %5 = getelementptr inbounds [3 x <2 x float>], ptr %2, i64 0, i64 1
  store <2 x float> <float 3.000000e+00, float -1.000000e+00>, ptr %5, align 8
  %6 = getelementptr inbounds [3 x <2 x float>], ptr %2, i64 0, i64 2
  store <2 x float> <float -1.000000e+00, float 3.000000e+00>, ptr %6, align 8
  %7 = zext i32 %0 to i64
  %8 = getelementptr inbounds [3 x <2 x float>], ptr %2, i64 0, i64 %7
  %9 = load <2 x float>, ptr %8, align 8
  %10 = shufflevector <2 x float> %9, <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
  %11 = shufflevector <4 x float> %10, <4 x float> <float poison, float poison, float 0.000000e+00, float 1.000000e+00>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  call void @llvm.lifetime.end.p0i8(i64 24, ptr nonnull %3) #5
  %12 = insertvalue <{ <4 x float> }> undef, <4 x float> %11, 0
  ret <{ <4 x float> }> %12
}

declare void @llvm.lifetime.start.p0i8(i64 immarg, ptr nocapture) #1

declare void @llvm.lifetime.end.p0i8(i64 immarg, ptr nocapture) #1

define <4 x half> @sw_half(<4 x float> %0, ptr addrspace(2) nocapture noundef readonly align 4 dereferenceable(4) "air-buffer-no-alias" %1) local_unnamed_addr #2 {
  %3 = shufflevector <4 x float> %0, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %4 = tail call <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float> %3) #6
  %5 = extractelement <2 x i32> %4, i64 1
  %6 = shl i32 %5, 8
  %7 = extractelement <2 x i32> %4, i64 0
  %8 = add i32 %6, %7
  %9 = load i32, ptr addrspace(2) %1, align 4
  %10 = tail call float @air.convert.f.f32.u.i32(i32 %8) #6
  switch i32 %9, label %23 [
    i32 0, label %11
    i32 1, label %14
    i32 2, label %17
    i32 3, label %21
  ]

11:                                               ; preds = %2
  %12 = fmul float %10, 0x3EF0000000000000
  %13 = tail call float @llvm.fmuladd.f32(float %12, float 1.500000e+00, float -2.500000e-01) #5
  br label %26

14:                                               ; preds = %2
  %15 = tail call float @air.exp2.f32(float -2.300000e+01) #6
  %16 = tail call float @llvm.fmuladd.f32(float %10, float %15, float 1.000000e+00) #5
  br label %26

17:                                               ; preds = %2
  %18 = tail call float @air.exp2.f32(float -2.300000e+01) #6
  %19 = fneg float %10
  %20 = tail call float @llvm.fmuladd.f32(float %19, float %18, float -1.000000e+00) #5
  br label %26

21:                                               ; preds = %2
  %22 = tail call float @llvm.fmuladd.f32(float %10, float 5.000000e-01, float 6.550400e+04) #5
  br label %26

23:                                               ; preds = %2
  %24 = fmul float %10, 0x3EF0000000000000
  %25 = tail call float @llvm.fmuladd.f32(float %24, float 4.000000e+00, float -2.000000e+00) #5
  br label %26

26:                                               ; preds = %11, %14, %17, %21, %23
  %27 = phi float [ %25, %23 ], [ %22, %21 ], [ %20, %17 ], [ %16, %14 ], [ %13, %11 ]
  %28 = fptrunc float %27 to half
  %29 = insertelement <4 x half> poison, half %28, i64 0
  %30 = shufflevector <4 x half> %29, <4 x half> poison, <4 x i32> zeroinitializer
  ret <4 x half> %30
}

declare <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float>) local_unnamed_addr #3

define <4 x float> @sw_float(<4 x float> %0, ptr addrspace(2) nocapture noundef readonly align 4 dereferenceable(4) "air-buffer-no-alias" %1) local_unnamed_addr #2 {
  %3 = shufflevector <4 x float> %0, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %4 = tail call <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float> %3) #6
  %5 = extractelement <2 x i32> %4, i64 1
  %6 = shl i32 %5, 8
  %7 = extractelement <2 x i32> %4, i64 0
  %8 = add i32 %6, %7
  %9 = load i32, ptr addrspace(2) %1, align 4
  %10 = tail call float @air.convert.f.f32.u.i32(i32 %8) #6
  switch i32 %9, label %23 [
    i32 0, label %11
    i32 1, label %14
    i32 2, label %17
    i32 3, label %21
  ]

11:                                               ; preds = %2
  %12 = fmul float %10, 0x3EF0000000000000
  %13 = tail call float @llvm.fmuladd.f32(float %12, float 1.500000e+00, float -2.500000e-01) #5
  br label %26

14:                                               ; preds = %2
  %15 = tail call float @air.exp2.f32(float -2.300000e+01) #6
  %16 = tail call float @llvm.fmuladd.f32(float %10, float %15, float 1.000000e+00) #5
  br label %26

17:                                               ; preds = %2
  %18 = tail call float @air.exp2.f32(float -2.300000e+01) #6
  %19 = fneg float %10
  %20 = tail call float @llvm.fmuladd.f32(float %19, float %18, float -1.000000e+00) #5
  br label %26

21:                                               ; preds = %2
  %22 = tail call float @llvm.fmuladd.f32(float %10, float 5.000000e-01, float 6.550400e+04) #5
  br label %26

23:                                               ; preds = %2
  %24 = fmul float %10, 0x3EF0000000000000
  %25 = tail call float @llvm.fmuladd.f32(float %24, float 4.000000e+00, float -2.000000e+00) #5
  br label %26

26:                                               ; preds = %11, %14, %17, %21, %23
  %27 = phi float [ %25, %23 ], [ %22, %21 ], [ %20, %17 ], [ %16, %14 ], [ %13, %11 ]
  %28 = insertelement <4 x float> poison, float %27, i64 0
  %29 = shufflevector <4 x float> %28, <4 x float> poison, <4 x i32> zeroinitializer
  ret <4 x float> %29
}

define <4 x float> @sw_hf(<4 x float> %0, ptr addrspace(2) nocapture noundef readonly align 4 dereferenceable(4) "air-buffer-no-alias" %1) local_unnamed_addr #2 {
  %3 = shufflevector <4 x float> %0, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %4 = tail call <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float> %3) #6
  %5 = extractelement <2 x i32> %4, i64 1
  %6 = shl i32 %5, 8
  %7 = extractelement <2 x i32> %4, i64 0
  %8 = add i32 %6, %7
  %9 = load i32, ptr addrspace(2) %1, align 4
  %10 = tail call float @air.convert.f.f32.u.i32(i32 %8) #6
  switch i32 %9, label %23 [
    i32 0, label %11
    i32 1, label %14
    i32 2, label %17
    i32 3, label %21
  ]

11:                                               ; preds = %2
  %12 = fmul float %10, 0x3EF0000000000000
  %13 = tail call float @llvm.fmuladd.f32(float %12, float 1.500000e+00, float -2.500000e-01) #5
  br label %26

14:                                               ; preds = %2
  %15 = tail call float @air.exp2.f32(float -2.300000e+01) #6
  %16 = tail call float @llvm.fmuladd.f32(float %10, float %15, float 1.000000e+00) #5
  br label %26

17:                                               ; preds = %2
  %18 = tail call float @air.exp2.f32(float -2.300000e+01) #6
  %19 = fneg float %10
  %20 = tail call float @llvm.fmuladd.f32(float %19, float %18, float -1.000000e+00) #5
  br label %26

21:                                               ; preds = %2
  %22 = tail call float @llvm.fmuladd.f32(float %10, float 5.000000e-01, float 6.550400e+04) #5
  br label %26

23:                                               ; preds = %2
  %24 = fmul float %10, 0x3EF0000000000000
  %25 = tail call float @llvm.fmuladd.f32(float %24, float 4.000000e+00, float -2.000000e+00) #5
  br label %26

26:                                               ; preds = %11, %14, %17, %21, %23
  %27 = phi float [ %25, %23 ], [ %22, %21 ], [ %20, %17 ], [ %16, %14 ], [ %13, %11 ]
  %28 = fptrunc float %27 to half
  %29 = insertelement <4 x half> poison, half %28, i64 0
  %30 = shufflevector <4 x half> %29, <4 x half> poison, <4 x i32> zeroinitializer
  %31 = tail call <4 x float> @air.convert.f.v4f32.f.v4f16(<4 x half> %30) #6
  ret <4 x float> %31
}

declare <4 x float> @air.convert.f.v4f32.f.v4f16(<4 x half>) local_unnamed_addr #3

declare float @air.convert.f.f32.u.i32(i32) local_unnamed_addr #3

declare float @llvm.fmuladd.f32(float, float, float) #4

declare float @air.exp2.f32(float) local_unnamed_addr #3

attributes #0 = { mustprogress nofree nosync nounwind willreturn }
attributes #1 = { mustprogress nofree nosync nounwind willreturn }
attributes #2 = { mustprogress nofree nosync nounwind willreturn }
attributes #3 = { mustprogress nofree nosync nounwind willreturn }
attributes #4 = { mustprogress nofree nosync nounwind willreturn }
attributes #5 = { nounwind }
attributes #6 = { nounwind willreturn }

!air.vertex = !{!9}
!air.fragment = !{!14, !20, !23}
!air.compile_options = !{!24, !25, !26}
!air.version = !{!28}
!air.language_version = !{!29}

!9 = !{ptr @vs, !10, !12}
!10 = !{!11}
!11 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!12 = !{!13}
!13 = !{i32 0, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vid"}
!14 = !{ptr @sw_half, !15, !17}
!15 = !{!16}
!16 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}
!17 = !{!18, !19}
!18 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!19 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"kind"}
!20 = !{ptr @sw_float, !21, !17}
!21 = !{!22}
!22 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!23 = !{ptr @sw_hf, !21, !17}
!24 = !{!"air.compile.denorms_disable"}
!25 = !{!"air.compile.fast_math_disable"}
!26 = !{!"air.compile.framebuffer_fetch_enable"}
!28 = !{i32 2, i32 8, i32 0}
!29 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// The square target's side.
const SIDE: usize = 256;
/// Pixels per draw.
const N: usize = SIDE * SIDE;
/// The blend color every draw sets.
const BLEND_COLOR: [f32; 4] = [0.3, 0.6, 0.9, 0.45];

/// A blend of the probe: its name, whether blending is on, the factors (color and alpha alike),
/// the operation, and whether Metal's rule was measured at it (the sweep; the others take the rule
/// as stated).
struct Blend {
    name: &'static str,
    on: bool,
    source: F,
    destination: F,
    op: O,
    measured: bool,
}

const fn blend(name: &'static str, source: F, destination: F, op: O, measured: bool) -> Blend {
    Blend {
        name,
        on: true,
        source,
        destination,
        op,
        measured,
    }
}

const BLENDS: [Blend; 22] = [
    Blend {
        name: "off",
        on: false,
        source: F::One,
        destination: F::Zero,
        op: O::Add,
        measured: true,
    },
    blend("one-zero", F::One, F::Zero, O::Add, true),
    blend("one-one", F::One, F::One, O::Add, true),
    blend("one-1msa", F::One, F::OneMinusSourceAlpha, O::Add, true),
    blend(
        "sa-1msa",
        F::SourceAlpha,
        F::OneMinusSourceAlpha,
        O::Add,
        true,
    ),
    blend("zero-1msa", F::Zero, F::OneMinusSourceAlpha, O::Add, true),
    blend(
        "dc-1mdc",
        F::DestinationColor,
        F::OneMinusDestinationColor,
        O::Add,
        true,
    ),
    blend(
        "ba-1mba",
        F::BlendAlpha,
        F::OneMinusBlendAlpha,
        O::Add,
        true,
    ),
    blend(
        "bc-1mbc",
        F::BlendColor,
        F::OneMinusBlendColor,
        O::Add,
        true,
    ),
    blend("1mbc-zero", F::OneMinusBlendColor, F::Zero, O::Add, true),
    blend("sc-dc", F::SourceColor, F::DestinationColor, O::Add, true),
    blend(
        "1msc-1msc",
        F::OneMinusSourceColor,
        F::OneMinusSourceColor,
        O::Add,
        true,
    ),
    blend(
        "bc-1msa",
        F::BlendColor,
        F::OneMinusSourceAlpha,
        O::Add,
        true,
    ),
    blend("min", F::SourceAlpha, F::OneMinusSourceAlpha, O::Min, true),
    blend("max", F::SourceAlpha, F::OneMinusSourceAlpha, O::Max, true),
    blend(
        "da-1mda",
        F::DestinationAlpha,
        F::OneMinusDestinationAlpha,
        O::Add,
        false,
    ),
    blend(
        "sat-1msa",
        F::SourceAlphaSaturated,
        F::OneMinusSourceAlpha,
        O::Add,
        false,
    ),
    blend(
        "one-1msa-sub",
        F::One,
        F::OneMinusSourceAlpha,
        O::Subtract,
        false,
    ),
    blend(
        "sa-1msa-rsub",
        F::SourceAlpha,
        F::OneMinusSourceAlpha,
        O::ReverseSubtract,
        false,
    ),
    blend("one-one-sub", F::One, F::One, O::Subtract, false),
    blend(
        "1mdc-one",
        F::OneMinusDestinationColor,
        F::One,
        O::Add,
        false,
    ),
    blend("dc-zero", F::DestinationColor, F::Zero, O::Add, false),
];

/// A format of the probe: its name, format, channels, and whether its channels are `float`s.
const FORMATS: [(&str, P, usize, bool); 6] = [
    ("r16f", P::R16Float, 1, false),
    ("rg16f", P::RG16Float, 2, false),
    ("rgba16f", P::RGBA16Float, 4, false),
    ("r32f", P::R32Float, 1, true),
    ("rg32f", P::RG32Float, 2, true),
    ("rgba32f", P::RGBA32Float, 4, true),
];

/// The value of half bits `bits`.
fn half(bits: u16) -> f64 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = i32::from((bits >> 10) & 0x1f);
    let m = f64::from(bits & 0x3ff);
    match e {
        0 => sign * m * 2f64.powi(-24),
        31 if m == 0.0 => sign * f64::INFINITY,
        31 => f64::NAN,
        _ => sign * (1.0 + m / 1024.0) * 2f64.powi(e - 15),
    }
}

/// The binade exponent of finite nonzero `a` (`2^e <= |a| < 2^(e+1)`).
fn binade(a: f64) -> i32 {
    let a = a.abs();
    let mut e = a.log2().floor() as i32;
    if a < 2f64.powi(e) {
        e -= 1;
    } else if a >= 2f64.powi(e + 1) {
        e += 1;
    }
    e
}

/// `x` rounded to a `half`, ties to even (toward zero where `rtz`), past the largest finite
/// `half` an infinity (toward zero: the largest).
fn to_half(x: f64, rtz: bool) -> f64 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    let q = 2f64.powi(binade(x).max(-14) - 10);
    let n = x.abs() / q;
    let v = if rtz { n.trunc() } else { n.round_ties_even() } * q;
    let v = match (v > 65504.0, rtz) {
        (true, true) => 65504.0,
        (true, false) => f64::INFINITY,
        (false, _) => v,
    };
    v.copysign(x)
}

/// `x` rounded to `float`, nearest even.
fn f32r(x: f64) -> f64 {
    f64::from(x as f32)
}

/// `x` rounded to `float` toward zero.
fn f32z(x: f64) -> f64 {
    let r = x as f32;
    if f64::from(r).abs() > x.abs() {
        f64::from(f32::from_bits(r.to_bits() - 1))
    } else {
        f64::from(r)
    }
}

/// A unit in the last place of the format (`float` where `wide`, else `half`) at magnitude `a`.
fn ulp(a: f64, wide: bool) -> f64 {
    if wide {
        2f64.powi(binade(a.max(2f64.powi(-126))) - 23)
    } else {
        2f64.powi(binade(a.max(2f64.powi(-14))) - 10)
    }
}

/// The ordered position of `x` among the format's values (its bits as a signed magnitude).
fn ordered(x: f64, wide: bool) -> i64 {
    let (bits, sign) = if wide {
        (i64::from((x as f32).to_bits()), 1i64 << 31)
    } else {
        let h = to_half(x, false);
        // Rebuild the bits of an exact half.
        let bits = if h == 0.0 {
            0
        } else if h.is_infinite() {
            0x7c00
        } else {
            let e = binade(h).max(-14);
            let m = (h.abs() / 2f64.powi(e - 10)) as i64;
            if e == -14 && m < 1024 {
                m
            } else {
                (i64::from(e + 15) << 10) | (m - 1024)
            }
        };
        (
            bits | if h.is_sign_negative() { 1 << 15 } else { 0 },
            1i64 << 15,
        )
    };
    if bits & sign != 0 {
        -(bits & (sign - 1))
    } else {
        bits
    }
}

/// Factor `f` of channel `c` (3 is alpha) over source `s`, destination `d` and blend color `bc`,
/// each `1 - x` and each blend color lane rounded by `r`.
fn factor(f: F, c: usize, s: [f64; 4], d: [f64; 4], bc: [f64; 4], r: &dyn Fn(f64) -> f64) -> f64 {
    match f {
        F::Zero => 0.0,
        F::One => 1.0,
        F::SourceColor => s[c],
        F::OneMinusSourceColor => r(1.0 - s[c]),
        F::SourceAlpha => s[3],
        F::OneMinusSourceAlpha => r(1.0 - s[3]),
        F::DestinationColor => d[c],
        F::OneMinusDestinationColor => r(1.0 - d[c]),
        F::DestinationAlpha => d[3],
        F::OneMinusDestinationAlpha => r(1.0 - d[3]),
        F::SourceAlphaSaturated if c == 3 => 1.0,
        F::SourceAlphaSaturated => s[3].min(r(1.0 - d[3])),
        F::BlendColor => r(bc[c]),
        F::OneMinusBlendColor => r(1.0 - r(bc[c])),
        F::BlendAlpha => r(bc[3]),
        F::OneMinusBlendAlpha => r(1.0 - r(bc[3])),
        _ => unreachable!("the probe's factors"),
    }
}

/// Channel `c` of blend `b`, every factor, product and the result rounded by `r`, with the
/// largest operand's magnitude: the source, the destination, or the products' magnitudes
/// summed; and with the factors' magnitudes too (the recorded bound's measure). A term whose
/// factor is `Zero` is kept in the sum where `kept` (Metal's 16-bit float blend),
/// else dropped.
fn blended(
    b: &Blend,
    c: usize,
    (s, d, bc): ([f64; 4], [f64; 4], [f64; 4]),
    r: &dyn Fn(f64) -> f64,
    kept: bool,
) -> (f64, f64, f64) {
    let (x, y) = (s[c], d[c]);
    let plain = x.abs().max(y.abs());
    match b.op {
        _ if !b.on => return (x, plain, plain),
        O::Min => return (x.min(y), plain, plain),
        O::Max => return (x.max(y), plain, plain),
        _ => {}
    }
    let fs = factor(b.source, c, s, d, bc, r);
    let fd = factor(b.destination, c, s, d, bc, r);
    let term = |f: F, v: f64, factor: f64| (kept || f != F::Zero).then(|| r(v * factor));
    let (ps, pd) = (term(b.source, x, fs), term(b.destination, y, fd));
    let neg = |t: Option<f64>| t.map(|v| -v);
    let (ps, pd) = match b.op {
        O::Add => (ps, pd),
        O::Subtract => (ps, neg(pd)),
        O::ReverseSubtract => (neg(ps), pd),
        _ => unreachable!("the probe's operations"),
    };
    let v = match (ps, pd) {
        (Some(a), Some(b)) => a + b,
        (Some(a), None) | (None, Some(a)) => a,
        (None, None) => 0.0,
    };
    let mag = plain.max((x * fs).abs() + (y * fd).abs());
    (r(v), mag, mag.max(fs.abs()).max(fd.abs()))
}

/// What one draw's channels store, against Metal's rule and the `float` model.
#[derive(Default)]
struct Tally {
    channels: usize,
    metal: usize,
    model: usize,
    model_rtz: usize,
    /// The largest distance from Metal's value in units in the last place at the largest
    /// operand's magnitude.
    at_operands: f64,
    /// The same at the largest operand's or factor's magnitude (the recorded bound's measure).
    at_factors: f64,
    /// The largest distance from Metal's value in the format's values.
    at_stored: i64,
}

/// The LCG a probe draws its destinations from: `count` steps of 16 bits.
fn steps(count: usize) -> Vec<u32> {
    let mut state = 7919u32;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (state >> 8) & 0xffff
        })
        .collect()
}

/// One fullscreen draw of `state` into a new `format` target holding `under` (`bpp` bytes a
/// texel), with `kind`'s sweep: the target's bytes after it.
fn draw(
    device: &Device,
    queue: &CommandQueue,
    state: &RenderPipelineState,
    format: P,
    bpp: usize,
    under: &[u8],
    kind: u32,
) -> Vec<u8> {
    let mut td = TextureDescriptor::texture_2d(format, SIDE as u64, SIDE as u64, false);
    td.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    td.resource_options = ResourceOptions::shared();
    let t = device.new_texture(&td).unwrap();
    let region = Region::new_2d(0, 0, SIDE as u64, SIDE as u64);
    t.replace_region(region, 0, 0, under, (SIDE * bpp) as u64, 0)
        .unwrap();
    let mut pass = RenderPassDescriptor::new();
    let a = &mut pass.color_attachments[0];
    a.texture = Some(t.clone());
    a.load_action = LoadAction::Load;
    a.store_action = StoreAction::Store;
    let cb = queue.command_buffer().unwrap();
    let mut e = cb.render_command_encoder(&pass).unwrap();
    e.set_render_pipeline_state(state).unwrap();
    e.set_fragment_bytes(&kind.to_le_bytes(), 0).unwrap();
    let [r, g, b, al] = BLEND_COLOR;
    e.set_blend_color(r, g, b, al);
    e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .unwrap();
    e.end_encoding().unwrap();
    cb.commit().unwrap();
    cb.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(cb.status(), CommandBufferStatus::Completed);
    let mut out = vec![0u8; N * bpp];
    t.get_bytes(&mut out, (SIDE * bpp) as u64, 0, region, 0, 0)
        .unwrap();
    out
}

/// Channel `c` of pixel `i` of `bytes`, a target of `channels` channels.
fn texel(bytes: &[u8], i: usize, c: usize, channels: usize, wide: bool) -> f64 {
    if wide {
        let at = (i * channels + c) * 4;
        f64::from(f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()))
    } else {
        let at = (i * channels + c) * 2;
        half(u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap()))
    }
}

#[test]
fn half_output_blends_against_metals_rule() {
    let mut failures = Vec::new();
    for setting in [HalfBlendPolicy::Device, HalfBlendPolicy::Exact] {
        failures.extend(blends_under(setting));
    }
    assert!(
        failures.is_empty(),
        "blends announced as Metal's that are not: {failures:?}"
    );
}

/// Every draw of the probe under `setting` (the 16-bit float formats alone under
/// `HalfBlendPolicy::Exact`), its lines printed: the draws whose pipeline announced no
/// `half-output-precision` that store another value than Metal's in some channel.
fn blends_under(setting: HalfBlendPolicy) -> Vec<String> {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let precision = Degradation::from_precision_note(PrecisionNote::HalfOutputPrecision);
    // Every draw of a noted pipeline announces the note: whether one did is the draw's own.
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.announce_each = [precision].into_iter().collect();
        options.policy.half_blend = setting;
    })
    .unwrap();
    let name = device.name().to_owned();
    let label = match setting {
        HalfBlendPolicy::Device => "device blend",
        _ => "exact blend",
    };
    let queue = device.new_command_queue().unwrap();
    let library = device
        .new_library_with_air(&llvm_assemble_text(SAFE).unwrap())
        .unwrap();
    let function = |n: &str| {
        StageSource::Function(
            library
                .new_function_with_name(n)
                .unwrap()
                .function_descriptor(),
        )
    };
    let state = |format: P, b: Option<&Blend>| {
        let mut d = RenderPipelineDescriptor::new(function("vs"));
        d.fragment = Some(function("sw_half"));
        let mut c = ColorAttachmentDescriptor::new(format);
        if let Some(b) = b {
            c.blending_enabled = b.on;
            c.source_rgb_blend_factor = b.source;
            c.destination_rgb_blend_factor = b.destination;
            c.source_alpha_blend_factor = b.source;
            c.destination_alpha_blend_factor = b.destination;
            c.rgb_blend_operation = b.op;
            c.alpha_blend_operation = b.op;
        }
        d.color_attachments[0] = Some(c);
        device
            .new_render_pipeline_state(&d, &PipelineOptions::default())
            .unwrap()
    };
    let s16 = steps(N * 4);
    let halves: Vec<u8> = s16
        .iter()
        .flat_map(|s| u16::try_from(s % 0x3c01).unwrap().to_le_bytes())
        .collect();
    let floats: Vec<u8> = s16
        .iter()
        .flat_map(|s| (f32::from(u16::try_from(*s).unwrap()) / 65535.0).to_le_bytes())
        .collect();
    let bc = BLEND_COLOR.map(f64::from);
    // The source each pixel blends: the function's `half`, stored unblended.
    let plain = state(P::RGBA16Float, None);
    let sources: Vec<Vec<f64>> = [0u32, 4]
        .iter()
        .map(|&kind| {
            let out = draw(&device, &queue, &plain, P::RGBA16Float, 8, &halves, kind);
            (0..N).map(|i| texel(&out, i, 0, 4, false)).collect()
        })
        .collect();
    let mut failures = Vec::new();
    // The blended draws whose pipeline announced nothing, of all blended draws.
    let mut quiet = (0usize, 0usize);
    let exact_only = setting != HalfBlendPolicy::Device;
    for (fname, format, channels, wide) in FORMATS {
        if exact_only && wide {
            continue;
        }
        let bpp = channels * if wide { 4 } else { 2 };
        let under = if wide { &floats } else { &halves };
        let mut exact = (0usize, 0usize);
        let mut worst = (0f64, 0f64, 0i64);
        for b in &BLENDS {
            let made = state(format, Some(b));
            for (k, kind) in [0u32, 4].into_iter().enumerate() {
                let _ = messages.take_notices();
                let out = draw(&device, &queue, &made, format, bpp, &under[..N * bpp], kind);
                let noted = counts(&messages.take_notices()).contains_key(&precision);
                let mut t = Tally::default();
                for (i, &h) in sources[k].iter().enumerate() {
                    let s = [h; 4];
                    let mut d = [0.0, 0.0, 0.0, 1.0];
                    for (c, v) in d.iter_mut().enumerate().take(channels) {
                        *v = texel(&under[..N * bpp], i, c, channels, wide);
                    }
                    for c in 0..channels {
                        let got = texel(&out, i, c, channels, wide);
                        let (metal, mag, magf) = if wide {
                            blended(b, c, (s, d, bc), &f32r, false)
                        } else {
                            blended(b, c, (s, d, bc), &|x| to_half(x, false), true)
                        };
                        let (model, _, _) = blended(b, c, (s, d, bc), &f32r, false);
                        let (model, model_rtz) = if wide {
                            (model, model)
                        } else {
                            (to_half(model, false), to_half(f32z(model), true))
                        };
                        t.channels += 1;
                        t.metal += usize::from(got.to_bits() == metal.to_bits());
                        t.model += usize::from(got.to_bits() == model.to_bits());
                        t.model_rtz += usize::from(got.to_bits() == model_rtz.to_bits());
                        let apart = (got - metal).abs() / ulp(mag, wide);
                        if apart.is_finite() {
                            t.at_operands = t.at_operands.max(apart);
                        }
                        let apart = (got - metal).abs() / ulp(magf, wide);
                        if apart.is_finite() {
                            t.at_factors = t.at_factors.max(apart);
                        }
                        let stored = (ordered(got, wide) - ordered(metal, wide)).abs();
                        t.at_stored = t.at_stored.max(stored);
                    }
                }
                println!(
                    "HALF-BLEND {name} | {label} | {fname} | {} k{kind} | Metal's rule {}/{} | \
                     float model {} (toward zero {}) | apart {:.3} ulp at the operands, {:.3} \
                     with the factors, {} at the stored value | noted {noted} | measured on \
                     Metal {}",
                    b.name,
                    t.metal,
                    t.channels,
                    t.model,
                    t.model_rtz,
                    t.at_operands,
                    t.at_factors,
                    t.at_stored,
                    b.measured
                );
                if b.measured && b.on {
                    exact.1 += 1;
                    exact.0 += usize::from(t.metal == t.channels);
                    worst.0 = worst.0.max(t.at_operands);
                    worst.1 = worst.1.max(t.at_factors);
                    worst.2 = worst.2.max(t.at_stored);
                }
                if b.on {
                    quiet.1 += 1;
                    quiet.0 += usize::from(!noted);
                }
                if !noted && t.metal != t.channels {
                    failures.push(format!("{label}: {fname} {} k{kind}", b.name));
                }
            }
        }
        println!(
            "HALF-BLEND-SUMMARY {name} | {label} | {fname} | Metal's value in every channel: {} \
             of {} measured blend draws | largest distance {:.3} ulp at the operands' \
             magnitude, {:.3} with the factors', {} ulp at the stored value",
            exact.0, exact.1, worst.0, worst.1, worst.2
        );
    }
    println!(
        "HALF-BLEND-EXACT {name} | {label} | blended draws announcing nothing (each Metal's in \
         every channel, or listed as failing): {} of {}",
        quiet.0, quiet.1
    );
    drop(device);
    println!("HALF-BLEND layer: {:?}", messages.take());
    failures
}
