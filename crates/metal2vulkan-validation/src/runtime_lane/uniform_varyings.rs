//! Varyings a vertex function writes the same at every vertex of a primitive, read flat with
//! Metal's words (`uniform_varyings`), on the device the cfg selects, under the validation layer.
//!
//! The geometry, constants and fragment functions are our Metal probe's (the M3
//! Max, macOS 26): 16 instances, each a 64 x 64 tile of six overlapping triangles whose w varies
//! (but for the first), drawn into a 256 x 256 target; each instance's `float4` and `half4`
//! constants read by `vs` from `constant` buffers at `[[instance_id]]` and written to the
//! varyings `cp`, `cn` (`float4`), `hp`, `hn` (`half4`), which the fragment functions read with
//! perspective (`cp`, `hp`) and without (`cn`, `hn`), at the `center` (1x and 4x), the `centroid`
//! (4x) and per `sample` (4x), and store per pixel or sample with the instance. Metal's words, as
//! measured there for these very constants: every finite normal value itself; a zero or
//! subnormal `+0`; an infinity the largest finite value of its sign; a NaN `+FLT_MAX` without
//! perspective and, with perspective, a word that depends on the pixel (the residue, counted, not
//! compared); a `half` `-0` reads `+0`, a `half` NaN `+inf`, and every other `half` itself.
//!
//! `vs_device` reads the same constants from `device` memory, which the translator does not prove
//! alike at every vertex, so its varyings are the device's interpolation: its words are printed
//! beside (`FLT-LANE`), with the share of the growing class (finite constants in a perspective
//! `float` varying across a triangle whose w varies) the flat read removes (`FLT-LANE-SHARE`),
//! and its special values' words (`FLT-LANE-WORD`). The flat read's every compared word must be
//! Metal's, but a `half` denormal's where a stage records the device's own `half` rounding
//! (`PrecisionNote::NativeHalfQuantize` or `NativeHalfConversion`, on a device without
//! `shaderFloat16`): there it may read as the zero the note's bound allows (`flushed`), as the
//! RX 550's RADV answers each, counted and printed. Its fragment module must carry the four
//! varyings' `Flat` decorations the twin's lacks. A pipeline the device refuses
//! (`Error::Unsupported`: on MoltenVK the centroid function, which reads `[[sample_mask]]`, as
//! `PixelCoverageAtSampleRate`) is printed (`FLT-LANE-REFUSED`) and skipped, behind either vertex
//! function.
//!
//! [`MODULE`] is this lane's own MSL (`lane.metal`, the probe's with the device twin), compiled
//! with `xcrun metal -S -emit-llvm -O2` and written with opaque pointers, its alias metadata
//! dropped.

use super::{
    at_plan, buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device_with,
    Messages, ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureType, TextureUsage};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, PrecisionNote, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    ClearColor, CommandBufferStatus, Device, Error, ExecutedPipeline, Library, LoadAction,
    PipelineOptions, RenderPassDescriptor, ResourceOptions, StoreAction, TextureDescriptor,
};
use std::collections::BTreeMap;

/// The lane's AIR: vertex functions `vs` and `vs_device`, fragment functions `fs_center`,
/// `fs_centroid` and `fs_sample` (see the module docs).
const MODULE: &str = r#"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define <{ <4 x float>, <4 x float>, <4 x float>, <4 x half>, <4 x half>, i32 }> @vs(i32 noundef %0, i32 noundef %1, ptr addrspace(1) nocapture noundef readonly "air-buffer-no-alias" %2, ptr addrspace(2) nocapture noundef readonly "air-buffer-no-alias" %3, ptr addrspace(2) nocapture noundef readonly "air-buffer-no-alias" %4) local_unnamed_addr #0 {
  %6 = zext i32 %0 to i64
  %7 = getelementptr inbounds <4 x float>, ptr addrspace(1) %2, i64 %6
  %8 = load <4 x float>, ptr addrspace(1) %7, align 16
  %9 = and i32 %1, 3
  %10 = tail call fast float @air.convert.f.f32.u.i32(i32 %9) #3
  %11 = lshr i32 %1, 2
  %12 = tail call fast float @air.convert.f.f32.u.i32(i32 %11) #3
  %13 = extractelement <4 x float> %8, i64 0
  %14 = fadd fast float %10, %13
  %15 = fmul fast float %14, 5.000000e-01
  %16 = fadd fast float %15, -1.000000e+00
  %17 = extractelement <4 x float> %8, i64 1
  %18 = fadd fast float %12, %17
  %19 = fmul fast float %18, 5.000000e-01
  %20 = fadd fast float %19, -1.000000e+00
  %21 = extractelement <4 x float> %8, i64 3
  %22 = fmul fast float %16, %21
  %23 = insertelement <4 x float> undef, float %22, i64 0
  %24 = fmul fast float %20, %21
  %25 = insertelement <4 x float> %23, float %24, i64 1
  %26 = fmul fast float %21, 5.000000e-01
  %27 = insertelement <4 x float> %25, float %26, i64 2
  %28 = shufflevector <4 x float> %27, <4 x float> %8, <4 x i32> <i32 0, i32 1, i32 2, i32 7>
  %29 = zext i32 %1 to i64
  %30 = getelementptr inbounds <4 x float>, ptr addrspace(2) %3, i64 %29
  %31 = load <4 x float>, ptr addrspace(2) %30, align 16
  %32 = getelementptr inbounds <4 x half>, ptr addrspace(2) %4, i64 %29
  %33 = load <4 x half>, ptr addrspace(2) %32, align 8
  %34 = add i32 %1, 1
  %35 = insertvalue <{ <4 x float>, <4 x float>, <4 x float>, <4 x half>, <4 x half>, i32 }> undef, <4 x float> %28, 0
  %36 = insertvalue <{ <4 x float>, <4 x float>, <4 x float>, <4 x half>, <4 x half>, i32 }> %35, <4 x float> %31, 1
  %37 = insertvalue <{ <4 x float>, <4 x float>, <4 x float>, <4 x half>, <4 x half>, i32 }> %36, <4 x float> %31, 2
  %38 = insertvalue <{ <4 x float>, <4 x float>, <4 x float>, <4 x half>, <4 x half>, i32 }> %37, <4 x half> %33, 3
  %39 = insertvalue <{ <4 x float>, <4 x float>, <4 x float>, <4 x half>, <4 x half>, i32 }> %38, <4 x half> %33, 4
  %40 = insertvalue <{ <4 x float>, <4 x float>, <4 x float>, <4 x half>, <4 x half>, i32 }> %39, i32 %34, 5
  ret <{ <4 x float>, <4 x float>, <4 x float>, <4 x half>, <4 x half>, i32 }> %40
}

define <{ <4 x float>, <4 x float>, <4 x float>, <4 x half>, <4 x half>, i32 }> @vs_device(i32 noundef %0, i32 noundef %1, ptr addrspace(1) nocapture noundef readonly "air-buffer-no-alias" %2, ptr addrspace(1) nocapture noundef readonly "air-buffer-no-alias" %3, ptr addrspace(1) nocapture noundef readonly "air-buffer-no-alias" %4) local_unnamed_addr #0 {
  %6 = zext i32 %0 to i64
  %7 = getelementptr inbounds <4 x float>, ptr addrspace(1) %2, i64 %6
  %8 = load <4 x float>, ptr addrspace(1) %7, align 16
  %9 = and i32 %1, 3
  %10 = tail call fast float @air.convert.f.f32.u.i32(i32 %9) #3
  %11 = lshr i32 %1, 2
  %12 = tail call fast float @air.convert.f.f32.u.i32(i32 %11) #3
  %13 = extractelement <4 x float> %8, i64 0
  %14 = fadd fast float %10, %13
  %15 = fmul fast float %14, 5.000000e-01
  %16 = fadd fast float %15, -1.000000e+00
  %17 = extractelement <4 x float> %8, i64 1
  %18 = fadd fast float %12, %17
  %19 = fmul fast float %18, 5.000000e-01
  %20 = fadd fast float %19, -1.000000e+00
  %21 = extractelement <4 x float> %8, i64 3
  %22 = fmul fast float %16, %21
  %23 = insertelement <4 x float> undef, float %22, i64 0
  %24 = fmul fast float %20, %21
  %25 = insertelement <4 x float> %23, float %24, i64 1
  %26 = fmul fast float %21, 5.000000e-01
  %27 = insertelement <4 x float> %25, float %26, i64 2
  %28 = shufflevector <4 x float> %27, <4 x float> %8, <4 x i32> <i32 0, i32 1, i32 2, i32 7>
  %29 = zext i32 %1 to i64
  %30 = getelementptr inbounds <4 x float>, ptr addrspace(1) %3, i64 %29
  %31 = load <4 x float>, ptr addrspace(1) %30, align 16
  %32 = getelementptr inbounds <4 x half>, ptr addrspace(1) %4, i64 %29
  %33 = load <4 x half>, ptr addrspace(1) %32, align 8
  %34 = add i32 %1, 1
  %35 = insertvalue <{ <4 x float>, <4 x float>, <4 x float>, <4 x half>, <4 x half>, i32 }> undef, <4 x float> %28, 0
  %36 = insertvalue <{ <4 x float>, <4 x float>, <4 x float>, <4 x half>, <4 x half>, i32 }> %35, <4 x float> %31, 1
  %37 = insertvalue <{ <4 x float>, <4 x float>, <4 x float>, <4 x half>, <4 x half>, i32 }> %36, <4 x float> %31, 2
  %38 = insertvalue <{ <4 x float>, <4 x float>, <4 x float>, <4 x half>, <4 x half>, i32 }> %37, <4 x half> %33, 3
  %39 = insertvalue <{ <4 x float>, <4 x float>, <4 x float>, <4 x half>, <4 x half>, i32 }> %38, <4 x half> %33, 4
  %40 = insertvalue <{ <4 x float>, <4 x float>, <4 x float>, <4 x half>, <4 x half>, i32 }> %39, i32 %34, 5
  ret <{ <4 x float>, <4 x float>, <4 x float>, <4 x half>, <4 x half>, i32 }> %40
}

define <4 x float> @fs_center(<4 x float> %0, <4 x float> %1, <4 x float> %2, <4 x half> %3, <4 x half> %4, i32 %5, ptr addrspace(1) nocapture noundef writeonly "air-buffer-no-alias" %6, ptr addrspace(2) nocapture noundef readonly align 4 dereferenceable(4) "air-buffer-no-alias" %7) local_unnamed_addr #1 {
  %9 = shufflevector <4 x float> %0, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %10 = tail call <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float> %9) #3
  %11 = extractelement <2 x i32> %10, i64 1
  %12 = load i32, ptr addrspace(2) %7, align 4
  %13 = mul i32 %11, %12
  %14 = extractelement <2 x i32> %10, i64 0
  %15 = add i32 %13, %14
  %16 = shl i32 %15, 4
  %17 = zext i32 %16 to i64
  %18 = getelementptr inbounds i32, ptr addrspace(1) %6, i64 %17
  store i32 %5, ptr addrspace(1) %18, align 4
  %19 = bitcast <4 x float> %1 to <4 x i32>
  %20 = extractelement <4 x i32> %19, i64 0
  %21 = getelementptr inbounds i32, ptr addrspace(1) %18, i64 1
  store i32 %20, ptr addrspace(1) %21, align 4
  %22 = extractelement <4 x i32> %19, i64 1
  %23 = getelementptr inbounds i32, ptr addrspace(1) %18, i64 2
  store i32 %22, ptr addrspace(1) %23, align 4
  %24 = extractelement <4 x i32> %19, i64 2
  %25 = getelementptr inbounds i32, ptr addrspace(1) %18, i64 3
  store i32 %24, ptr addrspace(1) %25, align 4
  %26 = extractelement <4 x i32> %19, i64 3
  %27 = getelementptr inbounds i32, ptr addrspace(1) %18, i64 4
  store i32 %26, ptr addrspace(1) %27, align 4
  %28 = bitcast <4 x float> %2 to <4 x i32>
  %29 = extractelement <4 x i32> %28, i64 0
  %30 = getelementptr inbounds i32, ptr addrspace(1) %18, i64 5
  store i32 %29, ptr addrspace(1) %30, align 4
  %31 = extractelement <4 x i32> %28, i64 1
  %32 = getelementptr inbounds i32, ptr addrspace(1) %18, i64 6
  store i32 %31, ptr addrspace(1) %32, align 4
  %33 = extractelement <4 x i32> %28, i64 2
  %34 = getelementptr inbounds i32, ptr addrspace(1) %18, i64 7
  store i32 %33, ptr addrspace(1) %34, align 4
  %35 = extractelement <4 x i32> %28, i64 3
  %36 = getelementptr inbounds i32, ptr addrspace(1) %18, i64 8
  store i32 %35, ptr addrspace(1) %36, align 4
  %37 = bitcast <4 x half> %3 to <2 x i32>
  %38 = extractelement <2 x i32> %37, i64 0
  %39 = getelementptr inbounds i32, ptr addrspace(1) %18, i64 9
  store i32 %38, ptr addrspace(1) %39, align 4
  %40 = extractelement <2 x i32> %37, i64 1
  %41 = getelementptr inbounds i32, ptr addrspace(1) %18, i64 10
  store i32 %40, ptr addrspace(1) %41, align 4
  %42 = bitcast <4 x half> %4 to <2 x i32>
  %43 = extractelement <2 x i32> %42, i64 0
  %44 = getelementptr inbounds i32, ptr addrspace(1) %18, i64 11
  store i32 %43, ptr addrspace(1) %44, align 4
  %45 = extractelement <2 x i32> %42, i64 1
  %46 = getelementptr inbounds i32, ptr addrspace(1) %18, i64 12
  store i32 %45, ptr addrspace(1) %46, align 4
  %47 = getelementptr inbounds i32, ptr addrspace(1) %18, i64 13
  store i32 0, ptr addrspace(1) %47, align 4
  ret <4 x float> zeroinitializer
}

declare <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float>) local_unnamed_addr #2

define <4 x float> @fs_centroid(<4 x float> %0, <4 x float> %1, <4 x float> %2, <4 x half> %3, <4 x half> %4, i32 %5, i32 noundef %6, ptr addrspace(1) nocapture noundef writeonly "air-buffer-no-alias" %7, ptr addrspace(2) nocapture noundef readonly align 4 dereferenceable(4) "air-buffer-no-alias" %8) local_unnamed_addr #1 {
  %10 = shufflevector <4 x float> %0, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %11 = tail call <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float> %10) #3
  %12 = extractelement <2 x i32> %11, i64 1
  %13 = load i32, ptr addrspace(2) %8, align 4
  %14 = mul i32 %12, %13
  %15 = extractelement <2 x i32> %11, i64 0
  %16 = add i32 %14, %15
  %17 = shl i32 %16, 4
  %18 = zext i32 %17 to i64
  %19 = getelementptr inbounds i32, ptr addrspace(1) %7, i64 %18
  store i32 %5, ptr addrspace(1) %19, align 4
  %20 = bitcast <4 x float> %1 to <4 x i32>
  %21 = extractelement <4 x i32> %20, i64 0
  %22 = getelementptr inbounds i32, ptr addrspace(1) %19, i64 1
  store i32 %21, ptr addrspace(1) %22, align 4
  %23 = extractelement <4 x i32> %20, i64 1
  %24 = getelementptr inbounds i32, ptr addrspace(1) %19, i64 2
  store i32 %23, ptr addrspace(1) %24, align 4
  %25 = extractelement <4 x i32> %20, i64 2
  %26 = getelementptr inbounds i32, ptr addrspace(1) %19, i64 3
  store i32 %25, ptr addrspace(1) %26, align 4
  %27 = extractelement <4 x i32> %20, i64 3
  %28 = getelementptr inbounds i32, ptr addrspace(1) %19, i64 4
  store i32 %27, ptr addrspace(1) %28, align 4
  %29 = bitcast <4 x float> %2 to <4 x i32>
  %30 = extractelement <4 x i32> %29, i64 0
  %31 = getelementptr inbounds i32, ptr addrspace(1) %19, i64 5
  store i32 %30, ptr addrspace(1) %31, align 4
  %32 = extractelement <4 x i32> %29, i64 1
  %33 = getelementptr inbounds i32, ptr addrspace(1) %19, i64 6
  store i32 %32, ptr addrspace(1) %33, align 4
  %34 = extractelement <4 x i32> %29, i64 2
  %35 = getelementptr inbounds i32, ptr addrspace(1) %19, i64 7
  store i32 %34, ptr addrspace(1) %35, align 4
  %36 = extractelement <4 x i32> %29, i64 3
  %37 = getelementptr inbounds i32, ptr addrspace(1) %19, i64 8
  store i32 %36, ptr addrspace(1) %37, align 4
  %38 = bitcast <4 x half> %3 to <2 x i32>
  %39 = extractelement <2 x i32> %38, i64 0
  %40 = getelementptr inbounds i32, ptr addrspace(1) %19, i64 9
  store i32 %39, ptr addrspace(1) %40, align 4
  %41 = extractelement <2 x i32> %38, i64 1
  %42 = getelementptr inbounds i32, ptr addrspace(1) %19, i64 10
  store i32 %41, ptr addrspace(1) %42, align 4
  %43 = bitcast <4 x half> %4 to <2 x i32>
  %44 = extractelement <2 x i32> %43, i64 0
  %45 = getelementptr inbounds i32, ptr addrspace(1) %19, i64 11
  store i32 %44, ptr addrspace(1) %45, align 4
  %46 = extractelement <2 x i32> %43, i64 1
  %47 = getelementptr inbounds i32, ptr addrspace(1) %19, i64 12
  store i32 %46, ptr addrspace(1) %47, align 4
  %48 = getelementptr inbounds i32, ptr addrspace(1) %19, i64 13
  store i32 %6, ptr addrspace(1) %48, align 4
  ret <4 x float> zeroinitializer
}

define <4 x float> @fs_sample(<4 x float> %0, <4 x float> %1, <4 x float> %2, <4 x half> %3, <4 x half> %4, i32 %5, i32 noundef %6, ptr addrspace(1) nocapture noundef writeonly "air-buffer-no-alias" %7, ptr addrspace(2) nocapture noundef readonly align 4 dereferenceable(4) "air-buffer-no-alias" %8) local_unnamed_addr #1 {
  %10 = shufflevector <4 x float> %0, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %11 = tail call <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float> %10) #3
  %12 = extractelement <2 x i32> %11, i64 1
  %13 = load i32, ptr addrspace(2) %8, align 4
  %14 = mul i32 %12, %13
  %15 = extractelement <2 x i32> %11, i64 0
  %16 = add i32 %14, %15
  %17 = shl i32 %16, 2
  %18 = add i32 %17, %6
  %19 = shl i32 %18, 4
  %20 = zext i32 %19 to i64
  %21 = getelementptr inbounds i32, ptr addrspace(1) %7, i64 %20
  store i32 %5, ptr addrspace(1) %21, align 4
  %22 = bitcast <4 x float> %1 to <4 x i32>
  %23 = extractelement <4 x i32> %22, i64 0
  %24 = getelementptr inbounds i32, ptr addrspace(1) %21, i64 1
  store i32 %23, ptr addrspace(1) %24, align 4
  %25 = extractelement <4 x i32> %22, i64 1
  %26 = getelementptr inbounds i32, ptr addrspace(1) %21, i64 2
  store i32 %25, ptr addrspace(1) %26, align 4
  %27 = extractelement <4 x i32> %22, i64 2
  %28 = getelementptr inbounds i32, ptr addrspace(1) %21, i64 3
  store i32 %27, ptr addrspace(1) %28, align 4
  %29 = extractelement <4 x i32> %22, i64 3
  %30 = getelementptr inbounds i32, ptr addrspace(1) %21, i64 4
  store i32 %29, ptr addrspace(1) %30, align 4
  %31 = bitcast <4 x float> %2 to <4 x i32>
  %32 = extractelement <4 x i32> %31, i64 0
  %33 = getelementptr inbounds i32, ptr addrspace(1) %21, i64 5
  store i32 %32, ptr addrspace(1) %33, align 4
  %34 = extractelement <4 x i32> %31, i64 1
  %35 = getelementptr inbounds i32, ptr addrspace(1) %21, i64 6
  store i32 %34, ptr addrspace(1) %35, align 4
  %36 = extractelement <4 x i32> %31, i64 2
  %37 = getelementptr inbounds i32, ptr addrspace(1) %21, i64 7
  store i32 %36, ptr addrspace(1) %37, align 4
  %38 = extractelement <4 x i32> %31, i64 3
  %39 = getelementptr inbounds i32, ptr addrspace(1) %21, i64 8
  store i32 %38, ptr addrspace(1) %39, align 4
  %40 = bitcast <4 x half> %3 to <2 x i32>
  %41 = extractelement <2 x i32> %40, i64 0
  %42 = getelementptr inbounds i32, ptr addrspace(1) %21, i64 9
  store i32 %41, ptr addrspace(1) %42, align 4
  %43 = extractelement <2 x i32> %40, i64 1
  %44 = getelementptr inbounds i32, ptr addrspace(1) %21, i64 10
  store i32 %43, ptr addrspace(1) %44, align 4
  %45 = bitcast <4 x half> %4 to <2 x i32>
  %46 = extractelement <2 x i32> %45, i64 0
  %47 = getelementptr inbounds i32, ptr addrspace(1) %21, i64 11
  store i32 %46, ptr addrspace(1) %47, align 4
  %48 = extractelement <2 x i32> %45, i64 1
  %49 = getelementptr inbounds i32, ptr addrspace(1) %21, i64 12
  store i32 %48, ptr addrspace(1) %49, align 4
  %50 = getelementptr inbounds i32, ptr addrspace(1) %21, i64 13
  store i32 %6, ptr addrspace(1) %50, align 4
  ret <4 x float> zeroinitializer
}

declare float @air.convert.f.f32.u.i32(i32) local_unnamed_addr #2

attributes #0 = { mustprogress nofree nosync nounwind willreturn }
attributes #1 = { mustprogress nofree nosync nounwind willreturn }
attributes #2 = { mustprogress nofree nosync nounwind willreturn }
attributes #3 = { nounwind willreturn }

!air.vertex = !{!9, !23}
!air.fragment = !{!27, !39, !48}
!air.compile_options = !{!55, !56, !57}
!air.version = !{!59}
!air.language_version = !{!60}

!9 = !{ptr @vs, !10, !17}
!10 = !{!11, !12, !13, !14, !15, !16}
!11 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!12 = !{!"air.vertex_output", !"user(cp)", !"air.arg_type_name", !"float4", !"air.arg_name", !"cp"}
!13 = !{!"air.vertex_output", !"user(cn)", !"air.arg_type_name", !"float4", !"air.arg_name", !"cn"}
!14 = !{!"air.vertex_output", !"user(hp)", !"air.arg_type_name", !"half4", !"air.arg_name", !"hp"}
!15 = !{!"air.vertex_output", !"user(hn)", !"air.arg_type_name", !"half4", !"air.arg_name", !"hn"}
!16 = !{!"air.vertex_output", !"user(inst)", !"air.arg_type_name", !"uint", !"air.arg_name", !"inst"}
!17 = !{!18, !19, !20, !21, !22}
!18 = !{i32 0, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vid"}
!19 = !{i32 1, !"air.instance_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"iid"}
!20 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"tri"}
!21 = !{i32 3, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"vals"}
!22 = !{i32 4, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"half4", !"air.arg_name", !"hvals"}
!23 = !{ptr @vs_device, !10, !24}
!24 = !{!18, !19, !20, !25, !26}
!25 = !{i32 3, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"vals"}
!26 = !{i32 4, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"half4", !"air.arg_name", !"hvals"}
!27 = !{ptr @fs_center, !28, !30}
!28 = !{!29}
!29 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!30 = !{!31, !32, !33, !34, !35, !36, !37, !38}
!31 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!32 = !{i32 1, !"air.fragment_input", !"user(cp)", !"air.center", !"air.perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"cp"}
!33 = !{i32 2, !"air.fragment_input", !"user(cn)", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"cn"}
!34 = !{i32 3, !"air.fragment_input", !"user(hp)", !"air.center", !"air.perspective", !"air.arg_type_name", !"half4", !"air.arg_name", !"hp"}
!35 = !{i32 4, !"air.fragment_input", !"user(hn)", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"half4", !"air.arg_name", !"hn"}
!36 = !{i32 5, !"air.fragment_input", !"user(inst)", !"air.flat", !"air.arg_type_name", !"uint", !"air.arg_name", !"inst"}
!37 = !{i32 6, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!38 = !{i32 7, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"width"}
!39 = !{ptr @fs_centroid, !28, !40}
!40 = !{!31, !41, !42, !43, !44, !36, !45, !46, !47}
!41 = !{i32 1, !"air.fragment_input", !"user(cp)", !"air.centroid", !"air.perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"cp"}
!42 = !{i32 2, !"air.fragment_input", !"user(cn)", !"air.centroid", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"cn"}
!43 = !{i32 3, !"air.fragment_input", !"user(hp)", !"air.centroid", !"air.perspective", !"air.arg_type_name", !"half4", !"air.arg_name", !"hp"}
!44 = !{i32 4, !"air.fragment_input", !"user(hn)", !"air.centroid", !"air.no_perspective", !"air.arg_type_name", !"half4", !"air.arg_name", !"hn"}
!45 = !{i32 6, !"air.sample_mask_in", !"air.arg_type_name", !"uint", !"air.arg_name", !"mask"}
!46 = !{i32 7, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!47 = !{i32 8, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"width"}
!48 = !{ptr @fs_sample, !28, !49}
!49 = !{!31, !50, !51, !52, !53, !36, !54, !46, !47}
!50 = !{i32 1, !"air.fragment_input", !"user(cp)", !"air.sample", !"air.perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"cp"}
!51 = !{i32 2, !"air.fragment_input", !"user(cn)", !"air.sample", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"cn"}
!52 = !{i32 3, !"air.fragment_input", !"user(hp)", !"air.sample", !"air.perspective", !"air.arg_type_name", !"half4", !"air.arg_name", !"hp"}
!53 = !{i32 4, !"air.fragment_input", !"user(hn)", !"air.sample", !"air.no_perspective", !"air.arg_type_name", !"half4", !"air.arg_name", !"hn"}
!54 = !{i32 6, !"air.sample_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"sid"}
!55 = !{!"air.compile.denorms_disable"}
!56 = !{!"air.compile.fast_math_enable"}
!57 = !{!"air.compile.framebuffer_fetch_enable"}
!59 = !{i32 2, i32 8, i32 0}
!60 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// The probe's 64 `float` and 64 `half` constants (16 `float4` and 16 `half4`, one each per
/// instance), the fixed ones then the generator's (flatprobe.swift's, word for word).
fn constants() -> (Vec<u32>, Vec<u16>) {
    let mut f: Vec<u32> = vec![
        0x00000000, 0x80000000, 0x00000001, 0x807fffff, 0x00800000, 0x3f800000, 0x40400000,
        0x4b800000, 0x4b800001, 0x3dcccccd, 0x3eaaaaab, 0xc0200000, 0x7149f2ca, 0x7f7fffff,
        0xff7fffff, 0x7f800000, 0xff800000, 0x7fc00000, 0x7fc12345, 0xffc00000, 0x7f800001,
        0x7fbfffff, 0x00400000, 0x42f6e979, 0xc2c80000, 0x3f7fffff, 0x3f800001, 0x4effffff,
        0x5f000000, 0x0d000000, 0x447a0000, 0x3f000000,
    ];
    let mut h: Vec<u16> = vec![
        0x0000, 0x8000, 0x0001, 0x83ff, 0x0400, 0x3c00, 0x4200, 0x6800, 0x6801, 0x2e66, 0x3555,
        0xc100, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e00, 0x7e55, 0xfe00, 0x7c01, 0x7dff, 0x0200,
        0x57b7, 0xd640, 0x3bff, 0x3c01, 0x5640, 0x1400,
    ];
    let mut seed: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut next = || {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (seed >> 32) as u32
    };
    while f.len() < 64 {
        let r = next();
        let exp = 100 + r % 55;
        f.push((r & 0x8000_0000) | (exp << 23) | (next() & 0x7f_ffff));
    }
    while h.len() < 64 {
        let r = next();
        let exp = (1 + r % 29) as u16;
        let sign: u16 = if r & 0x8000_0000 != 0 { 0x8000 } else { 0 };
        let mant = (next() as u16) & 0x3ff;
        h.push(sign | (exp << 10) | mant);
    }
    (f, h)
}

/// Metal's word for a `float` constant `v` read through an interpolation: every
/// finite normal value itself; a zero or subnormal `+0`; an infinity the largest finite value of
/// its sign; a NaN `+FLT_MAX` without perspective, and with perspective a word that depends on
/// the pixel (`None`: a recorded residue, not compared).
fn metal_float(v: u32, perspective: bool) -> Option<u32> {
    let e = v & 0x7f80_0000;
    if e == 0 {
        Some(0)
    } else if e != 0x7f80_0000 {
        Some(v)
    } else if v & 0x7f_ffff == 0 {
        Some((v & 0x8000_0000) | 0x7f7f_ffff)
    } else if perspective {
        None
    } else {
        Some(0x7f7f_ffff)
    }
}

/// Metal's word for a `half` constant `v` read through an interpolation: `-0`
/// `+0`, a NaN `+inf`, every other half (subnormals and infinities too) itself.
fn metal_half(v: u16) -> u16 {
    if v == 0x8000 {
        0
    } else if v & 0x7c00 == 0x7c00 && v & 0x3ff != 0 {
        0x7c00
    } else {
        v
    }
}

/// Six triangles per tile, `(x, y, 0, w)` in the tile's own unit square: their w varies, but for
/// the first.
const TRIANGLES: [f32; 72] = [
    0.02, 0.03, 0.0, 1.0, 0.97, 0.05, 0.0, 1.0, 0.04, 0.96, 0.0, 1.0, //
    0.98, 0.97, 0.0, 4.0, 0.03, 0.99, 0.0, 0.25, 0.99, 0.02, 0.0, 1.7, //
    0.1, 0.1, 0.0, 0.3, 0.9, 0.2, 0.0, 7.0, 0.3, 0.9, 0.0, 2.0, //
    0.5, 0.01, 0.0, 10.0, 0.99, 0.6, 0.0, 0.1, 0.01, 0.7, 0.0, 1.0, //
    0.2, 0.5, 0.0, 1.0, 0.8, 0.45, 0.0, 1.0, 0.55, 0.95, 0.0, 8.0, //
    0.013, 0.5, 0.0, 2.5, 0.6, 0.013, 0.0, 2.5, 0.987, 0.987, 0.0, 0.7,
];

/// The target's side in pixels, and the words of one record.
const SIDE: usize = 256;
const RECORD: usize = 16;

/// What one varying's words came to: compared, equal to Metal's, and among them the finite
/// normal constants' (the growing class); the residue's words; and per constant index the words
/// that differ from Metal's, apart from those the device's noted `half` rounding answers
/// ([`flushed`]).
#[derive(Default)]
struct Tally {
    compared: u64,
    equal: u64,
    normal: u64,
    normal_equal: u64,
    residue: u64,
    wrong: BTreeMap<usize, BTreeMap<u32, u64>>,
    flushed: BTreeMap<usize, BTreeMap<u32, u64>>,
}

impl Tally {
    /// One word `got` of constant `index` against Metal's `want`; `flushed` where it is not
    /// Metal's but the device's noted rounding of a `half` denormal ([`flushed`]).
    fn add(&mut self, index: usize, normal: bool, got: u32, want: Option<u32>, flushed: bool) {
        let Some(want) = want else {
            self.residue += 1;
            return;
        };
        self.compared += 1;
        self.normal += u64::from(normal);
        if got == want {
            self.equal += 1;
            self.normal_equal += u64::from(normal);
        } else if flushed {
            *self
                .flushed
                .entry(index)
                .or_default()
                .entry(got)
                .or_default() += 1;
        } else {
            *self.wrong.entry(index).or_default().entry(got).or_default() += 1;
        }
    }

    /// The words the device's noted rounding answered in Metal's place.
    fn flushes(&self) -> u64 {
        self.flushed.values().flat_map(|w| w.values()).sum()
    }

    fn share(n: u64, d: u64) -> String {
        if d == 0 {
            "-".to_owned()
        } else {
            format!("{:.2}%", 100.0 * n as f64 / d as f64)
        }
    }
}

/// Whether `got` is the device's answer for Metal's `half` `want` under the notes its stages
/// record, where it is not Metal's: `want` a `half` denormal and `got` a zero, of either sign
/// where a stage rounds a `half` by the device's quantize (`PrecisionNote::NativeHalfQuantize`,
/// on a device without `shaderFloat16`; the SPIR-V specification lets it answer a zero below the
/// smallest normal `half`), of `want`'s sign where a stage widens or narrows a `half` by the
/// device's conversion (`PrecisionNote::NativeHalfConversion`; a device that flushes 16-bit
/// denormals). The flat read rounds each `half` lane after Metal's word for it and narrows it to
/// its bits for the store, and the vertex function widens it from its bits.
fn flushed(notes: &[PrecisionNote], want: u16, got: u16) -> bool {
    let denormal = want & 0x7c00 == 0 && want & 0x3ff != 0;
    let zero = got & 0x7fff == 0;
    let quantized = notes.contains(&PrecisionNote::NativeHalfQuantize);
    let converted = notes.contains(&PrecisionNote::NativeHalfConversion) && got == want & 0x8000;
    denormal && zero && (quantized || converted)
}

/// The varyings of a record: `cp`, `cn` (`float4`), `hp`, `hn` (`half4`).
const VARYINGS: [&str; 4] = ["cp", "cn", "hp", "hn"];

/// The words `fragment` behind vertex function `vertex` drew at `samples` samples, tallied per
/// varying, and the `Flat` decorations of the fragment module the draw executed; `None` where the
/// device refuses the pipeline (`Error::Unsupported`, printed).
fn drawn(
    device: &Device,
    library: &Library,
    vertex: &str,
    fragment: &str,
    samples: u64,
    per_sample: bool,
) -> Option<([Tally; 4], usize)> {
    let function = |name: &str| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut descriptor = RenderPipelineDescriptor::new(function(vertex));
    descriptor.fragment = Some(function(fragment));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    descriptor.raster_sample_count = samples as u32;
    let pipeline = match device.new_render_pipeline_state(&descriptor, &PipelineOptions::default())
    {
        Ok(p) => p,
        Err(Error::Unsupported(u)) => {
            println!(
                "FLT-LANE-REFUSED {}: {vertex} {fragment} at {samples}x: {u:?}",
                device.name()
            );
            return None;
        }
        Err(e) => panic!("{vertex} {fragment} at {samples}: {e}"),
    };
    let (f, h) = constants();
    let bytes = |words: &[u8]| buffer_with(device, words).unwrap();
    let triangles = bytes(
        &TRIANGLES
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let floats = bytes(&f.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
    let halves = bytes(&h.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
    let records = SIDE * SIDE * if per_sample { samples as usize } else { 1 };
    let out = buffer_with(device, &vec![0u8; records * RECORD * 4]).unwrap();
    let mut texture =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE as u64, SIDE as u64, false);
    if samples > 1 {
        texture.texture_type = TextureType::Type2DMultisample;
        texture.sample_count = samples;
    }
    texture.usage = TextureUsage::RENDER_TARGET;
    texture.resource_options = ResourceOptions::private();
    let mut pass = RenderPassDescriptor::new();
    let target = &mut pass.color_attachments[0];
    target.texture = Some(device.new_texture(&texture).unwrap());
    target.load_action = LoadAction::Clear;
    target.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
    target.store_action = StoreAction::DontCare;
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(&pipeline).unwrap();
    encoder.set_vertex_buffer(Some(&triangles), 0, 0).unwrap();
    encoder.set_vertex_buffer(Some(&floats), 0, 1).unwrap();
    encoder.set_vertex_buffer(Some(&halves), 0, 2).unwrap();
    encoder.set_fragment_buffer(Some(&out), 0, 0).unwrap();
    encoder
        .set_fragment_bytes(&(SIDE as u32).to_le_bytes(), 1)
        .unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 18, 16, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let executions = command_buffer.executions();
    let ExecutedPipeline::Render(executed) = executions[0].pipeline() else {
        panic!("the draw executed no render pipeline");
    };
    let flats = flat_decorations(&executed.stages.last().expect("fragment").spirv);
    let notes: Vec<PrecisionNote> = executed
        .stages
        .iter()
        .flat_map(|s| s.precision_notes.iter().copied())
        .collect();
    let words: Vec<u32> = read_buffer(&out, 0, (records * RECORD * 4) as u64)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect();
    let mut tallies: [Tally; 4] = Default::default();
    for r in words.as_chunks::<RECORD>().0.iter() {
        let Some(instance) = (r[0] as usize).checked_sub(1) else {
            continue;
        };
        assert!(instance < 16, "record names instance {}", r[0]);
        for k in 0..4 {
            let i = 4 * instance + k;
            let normal = f[i] & 0x7f80_0000 != 0 && f[i] & 0x7f80_0000 != 0x7f80_0000;
            tallies[0].add(i, normal, r[1 + k], metal_float(f[i], true), false);
            tallies[1].add(i, normal, r[5 + k], metal_float(f[i], false), false);
            let half = |w: u32| (w >> (16 * (k % 2))) as u16;
            let hnormal = h[i] & 0x7c00 != 0x7c00;
            let want = metal_half(h[i]);
            for (tally, word) in [(2, r[9 + k / 2]), (3, r[11 + k / 2])] {
                let got = half(word);
                tallies[tally].add(
                    i,
                    hnormal,
                    u32::from(got),
                    Some(u32::from(want)),
                    flushed(&notes, want, got),
                );
            }
        }
    }
    Some((tallies, flats))
}

/// The `OpDecorate ... Flat` instructions of a module.
fn flat_decorations(spirv: &[u32]) -> usize {
    let mut at = 5;
    let mut n = 0;
    while at < spirv.len() {
        let count = (spirv[at] >> 16) as usize;
        if spirv[at] & 0xffff == 71 && count >= 3 && spirv[at + 2] == 14 {
            n += 1;
        }
        at += count.max(1);
    }
    n
}

/// Every varying the vertex function `vs` writes from a `constant` buffer at the instance reads
/// Metal's words through `center` (1x and 4x), `centroid` and `sample` interpolation, perspective
/// and not, `float` and `half`, on a geometry whose w varies: the fragment declares them `Flat`
/// (more `Flat` decorations than behind `vs_device`, which reads the same constants from
/// `device` memory and so is interpolated by the device). The twin's words are printed beside,
/// with the share of the growing class (finite constants in a perspective `float` varying) the
/// flat read removes.
#[test]
fn a_varying_alike_at_every_vertex_reads_metal_s_words() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
        let library = device
            .new_library_with_air(&llvm_assemble_text(MODULE).unwrap())
            .unwrap();
        let name = device.name();
        let four = device
            .properties()
            .supports_texture_sample_count(4)
            .unwrap();
        let mut runs = vec![("fs_center", 1, false)];
        if four {
            runs.extend([
                ("fs_center", 4, false),
                ("fs_centroid", 4, false),
                ("fs_sample", 4, true),
            ]);
        }
        let mut differs = Vec::new();
        let (f, h) = constants();
        for (fragment, samples, per_sample) in runs {
            let Some((flat, flat_decorations)) =
                drawn(&device, &library, "vs", fragment, samples, per_sample)
            else {
                continue;
            };
            let (twin, twin_decorations) = drawn(
                &device,
                &library,
                "vs_device",
                fragment,
                samples,
                per_sample,
            )
            .expect("the device twin's pipeline, where the flat one is made");
            println!(
                "FLT-LANE {name}: {fragment} at {samples}x: Flat decorations {flat_decorations} (device twin {twin_decorations})"
            );
            assert!(
                flat_decorations >= twin_decorations + 4,
                "{fragment} at {samples}x: the flat read did not run ({flat_decorations} against {twin_decorations})"
            );
            for (v, (a, b)) in VARYINGS.iter().zip(flat.iter().zip(twin.iter())) {
                println!(
                    "FLT-LANE {name}: {fragment} at {samples}x {v}: flat {}/{} = Metal ({}), finite {}/{} ({}), residue {}, device-rounded denormals {}; device twin {}/{} = Metal ({}), finite {}/{} ({})",
                    a.equal, a.compared, Tally::share(a.equal, a.compared),
                    a.normal_equal, a.normal, Tally::share(a.normal_equal, a.normal), a.residue,
                    a.flushes(),
                    b.equal, b.compared, Tally::share(b.equal, b.compared),
                    b.normal_equal, b.normal, Tally::share(b.normal_equal, b.normal),
                );
                if *v == "cp" {
                    let before = b.normal - b.normal_equal;
                    let after = a.normal - a.normal_equal;
                    println!(
                        "FLT-LANE-SHARE {name}: {fragment} at {samples}x: finite perspective constants differing from Metal: device twin {before} of {}, flat {after} of {}: {} of the growing class removed",
                        b.normal, a.normal, Tally::share(before.saturating_sub(after), before)
                    );
                }
                for (i, words) in &a.wrong {
                    differs.push(format!(
                        "{fragment} at {samples}x {v} constant {i}: {words:x?}"
                    ));
                }
                if fragment == "fs_center" && samples == 1 {
                    // The special constants' words: Metal's, the flat read's, the twin's.
                    let half = v.starts_with('h');
                    for i in 0..64 {
                        let (value, special, metal) = if half {
                            let e = h[i] & 0x7c00;
                            (
                                u32::from(h[i]),
                                e == 0 || e == 0x7c00,
                                Some(u32::from(metal_half(h[i]))),
                            )
                        } else {
                            let e = f[i] & 0x7f80_0000;
                            (
                                f[i],
                                e == 0 || e == 0x7f80_0000,
                                metal_float(f[i], *v == "cp"),
                            )
                        };
                        if !special {
                            continue;
                        }
                        let words = |t: &Tally| match (metal, t.wrong.get(&i), t.flushed.get(&i)) {
                            (None, _, _) => "residue".to_owned(),
                            (Some(_), None, None) => "Metal's".to_owned(),
                            (Some(_), None, Some(w)) => format!("the device's {w:08x?}"),
                            (Some(_), Some(w), _) => format!("{w:08x?}"),
                        };
                        println!(
                            "FLT-LANE-WORD {name}: {v} constant {i} {value:08x}: Metal {metal:08x?}, flat {}, device twin {}",
                            words(a),
                            words(b)
                        );
                    }
                }
            }
        }
        for line in &differs {
            println!("FLT-LANE-DIFFERS {name}: {line}");
        }
        assert!(
            differs.is_empty(),
            "{} constants read other words than Metal's",
            differs.len()
        );
    }
    layer_reported_nothing_at_all(&messages);
}
