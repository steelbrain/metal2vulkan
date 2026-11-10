//! Out-of-bounds texture reads and writes on the CPU interpreter's image model (MSL 4.1 sections
//! 2.9.1 and 6.13).
//!
//! Metal reads a zero texel outside a texture, with the pixel format's missing components at
//! their defaults (alpha 1 for a format without alpha), and ignores a write outside it. An access
//! is outside when its coordinate, level, array slice, cube face, or sample is.
//!
//! What was measured, on an Apple M3 Max, macOS 26.7: `float` reads of each texture kind here past
//! each of those (face 6 and above included), `uint` and `int` reads of 2D textures and texel
//! buffers, and a depth read. The probe's wrapping cube index was 0x2AAA_AAAB at face 2, not this
//! file's 0x2AAA_AAAA at faces 4 and 5. The `half` read, the `int` array and `uint` `read_write`
//! reads, and both writes (to a cube array and to a texel buffer) were not measured; they follow
//! the specification (MSL 4.1 sections 2.9.1 and 6.13).
//!
//! Each kernel reads (or writes) `p[i]` per thread and must compute Metal's result in both
//! scheduling orders on every profile class:
//!
//! - `robustImageAccess2` (or `robustBufferAccess2` for a texel buffer): the device reads Metal's
//!   texel and discards the write; the interpreter's image models that ([`Image::robust`]).
//! - `robustImageAccess` alone: the shader checks its reads (the device's out-of-bounds texel is
//!   not Metal's; the interpreter's poisons it) and the device discards writes.
//! - no robustness for the resource (`robustBufferAccess` alone for a texel buffer): the shader
//!   checks reads and guards writes; the interpreter traps any access outside the image.
//!
//! Where the shader checks a read whose alpha it observes, the alpha comes from the
//! `OutOfBoundsAlpha` specialization, run with both values.
//!
//! `llvm-as` assembles the module; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::bind::{BindState, Dispatch, ResourceClass, TextureFacts, UsedResource, WriteSource};
use crate::budget::Budget;
use crate::device::profile::{Containment, DeviceProfile, GuestIsolation, Robustness};
use crate::device::requirements::VkFeature;
use crate::device::{ContainmentPolicy, Device, DeviceDescription, DevicePolicy};
use crate::error::{Capability, Error, Feature, LowerFamily, Unsupported};
use crate::library::Library;
use crate::lower::select::{LowerInputs, Plan};
use crate::metal::{PixelFormat, TextureType};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::lower_inputs::{StorageImageChoice, StorageImageClass};
use crate::pipeline::sym::{
    Axis, HandleClass, PushField, ResourceKind, ResourceSlot, SlotClass, SlotRef, SpecKey, Stage,
    SyntheticBinding,
};
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, profile, typed_module,
    typed_module_with, AssembleError,
};
use crate::pipeline::{ComputePipelineDescriptor, FunctionDescriptor, StageSource};
use crate::spirv::interp::{run_with_images, Buffer, Config, Image, Order, Resources};
use crate::spirv::ir::TypedModule;
use crate::spirv::testing::committed_profiles;
use crate::vk;

mod through;

/// The kernels, compiled with `xcrun metal -O2` (Metal 4.0, AIR 2.8) from this MSL, written for
/// this test, with the debug and type metadata the decoder does not read removed:
///
/// ```metal
/// kernel void oob_2d(texture2d<float, access::read> t [[texture(0)]],
///                    device const uint4 *p [[buffer(0)]], device float4 *out [[buffer(1)]],
///                    uint i [[thread_position_in_grid]]) {
///     out[i] = t.read(p[i].xy, p[i].z);
/// }
/// // oob_2d_rgb: out[i] = float4(t.read(p[i].xy, p[i].z).xyz, 2.0);
/// // oob_2d_uint: texture2d<uint>, out[i] = t.read(p[i].xy, p[i].z);
/// // oob_2d_array: texture2d_array<int>, out[i] = t.read(p[i].xy, p[i].z, p[i].w);
/// // oob_3d: texture3d<float>, out[i] = t.read(p[i].xyz, p[i].w);
/// // oob_cube: texturecube<float>, out[i] = t.read(p[i].xy, p[i].z, p[i].w); (face, level)
/// // oob_cube_array: texturecube_array<float>, out[i] = t.read(p[i].xy, p[i].z, p[i].w);
/// //     (face, cube)
/// // oob_ms: texture2d_ms<float>, out[i] = t.read(p[i].xy, p[i].z); (sample)
/// // oob_buffer: texture_buffer<float>, out[i] = t.read(p[i].x);
/// // oob_rw: texture2d<uint, access::read_write>, out[i] = t.read(p[i].xy);
/// // oob_depth: depth2d<float>, device float *out, out[i] = t.read(p[i].xy, p[i].z);
/// // oob_1d_half: texture1d<half>, device half4 *out, out[i] = t.read(p[i].x);
/// // oob_cube_array_write: texturecube_array<float, access::write>,
/// //     t.write(float4(1, 2, 3, 4), p[i].xy, p[i].z, p[i].w);
/// // oob_buffer_write: texture_buffer<float, access::write>,
/// //     t.write(float4(1, 2, 3, 4), p[i].x);
/// ```
const MODULE: &str = r##"
source_filename = "-"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @oob_2d(ptr addrspace(1) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = zext i32 %3 to i64
  %6 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %1, i64 %5
  %7 = load <4 x i32>, ptr addrspace(1) %6, align 16
  %8 = shufflevector <4 x i32> %7, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %9 = extractelement <4 x i32> %7, i64 2
  %10 = tail call ptr addrspace(2) @air.get_read_sampler() #7
  %11 = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) %10, <2 x i32> %8, <2 x i32> zeroinitializer, i32 %9, i32 1) #8
  %12 = extractvalue { <4 x float>, i8 } %11, 0
  %13 = getelementptr inbounds <4 x float>, ptr addrspace(1) %2, i64 %5
  store <4 x float> %12, ptr addrspace(1) %13, align 16
  ret void
}

define void @oob_2d_rgb(ptr addrspace(1) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = zext i32 %3 to i64
  %6 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %1, i64 %5
  %7 = load <4 x i32>, ptr addrspace(1) %6, align 16
  %8 = shufflevector <4 x i32> %7, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %9 = extractelement <4 x i32> %7, i64 2
  %10 = tail call ptr addrspace(2) @air.get_read_sampler() #7
  %11 = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) %10, <2 x i32> %8, <2 x i32> zeroinitializer, i32 %9, i32 1) #8
  %12 = extractvalue { <4 x float>, i8 } %11, 0
  %13 = insertelement <4 x float> %12, float 2.000000e+00, i64 3
  %14 = getelementptr inbounds <4 x float>, ptr addrspace(1) %2, i64 %5
  store <4 x float> %13, ptr addrspace(1) %14, align 16
  ret void
}

define void @oob_2d_uint(ptr addrspace(1) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = zext i32 %3 to i64
  %6 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %1, i64 %5
  %7 = load <4 x i32>, ptr addrspace(1) %6, align 16
  %8 = shufflevector <4 x i32> %7, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %9 = extractelement <4 x i32> %7, i64 2
  %10 = tail call ptr addrspace(2) @air.get_read_sampler() #7
  %11 = tail call { <4 x i32>, i8 } @air.read_texture_2d.u.v4i32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) %10, <2 x i32> %8, <2 x i32> zeroinitializer, i32 %9, i32 1) #8
  %12 = extractvalue { <4 x i32>, i8 } %11, 0
  %13 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %2, i64 %5
  store <4 x i32> %12, ptr addrspace(1) %13, align 16
  ret void
}

define void @oob_2d_array(ptr addrspace(1) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = zext i32 %3 to i64
  %6 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %1, i64 %5
  %7 = load <4 x i32>, ptr addrspace(1) %6, align 16
  %8 = shufflevector <4 x i32> %7, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %9 = extractelement <4 x i32> %7, i64 2
  %10 = extractelement <4 x i32> %7, i64 3
  %11 = tail call ptr addrspace(2) @air.get_read_sampler() #7
  %12 = tail call { <4 x i32>, i8 } @air.read_texture_2d_array.s.v4i32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) %11, <2 x i32> %8, i32 %9, <2 x i32> zeroinitializer, i32 %10, i32 1) #8
  %13 = extractvalue { <4 x i32>, i8 } %12, 0
  %14 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %2, i64 %5
  store <4 x i32> %13, ptr addrspace(1) %14, align 16
  ret void
}

define void @oob_3d(ptr addrspace(1) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = zext i32 %3 to i64
  %6 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %1, i64 %5
  %7 = load <4 x i32>, ptr addrspace(1) %6, align 16
  %8 = shufflevector <4 x i32> %7, <4 x i32> poison, <3 x i32> <i32 0, i32 1, i32 2>
  %9 = extractelement <4 x i32> %7, i64 3
  %10 = tail call ptr addrspace(2) @air.get_read_sampler() #7
  %11 = tail call { <4 x float>, i8 } @air.read_texture_3d.v4f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) %10, <3 x i32> %8, <3 x i32> zeroinitializer, i32 %9, i32 1) #8
  %12 = extractvalue { <4 x float>, i8 } %11, 0
  %13 = getelementptr inbounds <4 x float>, ptr addrspace(1) %2, i64 %5
  store <4 x float> %12, ptr addrspace(1) %13, align 16
  ret void
}

define void @oob_cube(ptr addrspace(1) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = zext i32 %3 to i64
  %6 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %1, i64 %5
  %7 = load <4 x i32>, ptr addrspace(1) %6, align 16
  %8 = shufflevector <4 x i32> %7, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %9 = extractelement <4 x i32> %7, i64 2
  %10 = extractelement <4 x i32> %7, i64 3
  %11 = tail call ptr addrspace(2) @air.get_read_sampler() #7
  %12 = tail call { <4 x float>, i8 } @air.read_texture_cube.v4f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) %11, <2 x i32> %8, i32 %9, i32 %10, i32 1) #8
  %13 = extractvalue { <4 x float>, i8 } %12, 0
  %14 = getelementptr inbounds <4 x float>, ptr addrspace(1) %2, i64 %5
  store <4 x float> %13, ptr addrspace(1) %14, align 16
  ret void
}

define void @oob_cube_array(ptr addrspace(1) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = zext i32 %3 to i64
  %6 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %1, i64 %5
  %7 = load <4 x i32>, ptr addrspace(1) %6, align 16
  %8 = shufflevector <4 x i32> %7, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %9 = extractelement <4 x i32> %7, i64 2
  %10 = extractelement <4 x i32> %7, i64 3
  %11 = tail call ptr addrspace(2) @air.get_read_sampler() #7
  %12 = tail call { <4 x float>, i8 } @air.read_texture_cube_array.v4f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) %11, <2 x i32> %8, i32 %9, i32 %10, i32 0, i32 1) #8
  %13 = extractvalue { <4 x float>, i8 } %12, 0
  %14 = getelementptr inbounds <4 x float>, ptr addrspace(1) %2, i64 %5
  store <4 x float> %13, ptr addrspace(1) %14, align 16
  ret void
}

define void @oob_ms(ptr addrspace(1) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = zext i32 %3 to i64
  %6 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %1, i64 %5
  %7 = load <4 x i32>, ptr addrspace(1) %6, align 16
  %8 = shufflevector <4 x i32> %7, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %9 = extractelement <4 x i32> %7, i64 2
  %10 = tail call ptr addrspace(2) @air.get_read_sampler() #7
  %11 = tail call { <4 x float>, i8 } @air.read_texture_2d_ms.v4f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) %10, <2 x i32> %8, i32 %9, i32 1) #8
  %12 = extractvalue { <4 x float>, i8 } %11, 0
  %13 = getelementptr inbounds <4 x float>, ptr addrspace(1) %2, i64 %5
  store <4 x float> %12, ptr addrspace(1) %13, align 16
  ret void
}

define void @oob_buffer(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = zext i32 %3 to i64
  %6 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %1, i64 %5
  %7 = load <4 x i32>, ptr addrspace(1) %6, align 16
  %8 = extractelement <4 x i32> %7, i64 0
  %9 = tail call ptr addrspace(2) @air.get_read_sampler() #7
  %10 = tail call { <4 x float>, i8 } @air.read_texture_buffer_1d.v4f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) %9, i32 %8, i32 1) #8
  %11 = extractvalue { <4 x float>, i8 } %10, 0
  %12 = getelementptr inbounds <4 x float>, ptr addrspace(1) %2, i64 %5
  store <4 x float> %11, ptr addrspace(1) %12, align 16
  ret void
}

define void @oob_rw(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = zext i32 %3 to i64
  %6 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %1, i64 %5
  %7 = load <4 x i32>, ptr addrspace(1) %6, align 16
  %8 = shufflevector <4 x i32> %7, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %9 = tail call ptr addrspace(2) @air.get_read_sampler() #7
  %10 = tail call { <4 x i32>, i8 } @air.read_texture_2d.u.v4i32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) %9, <2 x i32> %8, <2 x i32> zeroinitializer, i32 0, i32 3) #8
  %11 = extractvalue { <4 x i32>, i8 } %10, 0
  %12 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %2, i64 %5
  store <4 x i32> %11, ptr addrspace(1) %12, align 16
  ret void
}

define void @oob_depth(ptr addrspace(1) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #1 {
  %5 = zext i32 %3 to i64
  %6 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %1, i64 %5
  %7 = load <4 x i32>, ptr addrspace(1) %6, align 16
  %8 = shufflevector <4 x i32> %7, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %9 = extractelement <4 x i32> %7, i64 2
  %10 = tail call ptr addrspace(2) @air.get_read_sampler() #7
  %11 = tail call { float, i8 } @air.read_depth_2d.f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) %10, i32 1, <2 x i32> %8, <2 x i32> zeroinitializer, i32 %9, i32 1) #8
  %12 = extractvalue { float, i8 } %11, 0
  %13 = getelementptr inbounds float, ptr addrspace(1) %2, i64 %5
  store float %12, ptr addrspace(1) %13, align 4
  ret void
}

define void @oob_1d_half(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #1 {
  %5 = zext i32 %3 to i64
  %6 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %1, i64 %5
  %7 = load <4 x i32>, ptr addrspace(1) %6, align 16
  %8 = extractelement <4 x i32> %7, i64 0
  %9 = tail call ptr addrspace(2) @air.get_read_sampler() #7
  %10 = tail call { <4 x half>, i8 } @air.read_texture_1d.v4f16(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) %9, i32 %8, i32 0, i32 1) #8
  %11 = extractvalue { <4 x half>, i8 } %10, 0
  %12 = getelementptr inbounds <4 x half>, ptr addrspace(1) %2, i64 %5
  store <4 x half> %11, ptr addrspace(1) %12, align 8
  ret void
}

define void @oob_cube_array_write(ptr addrspace(1) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, i32 noundef %2) local_unnamed_addr #2 {
  %4 = zext i32 %2 to i64
  %5 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %1, i64 %4
  %6 = load <4 x i32>, ptr addrspace(1) %5, align 16
  %7 = shufflevector <4 x i32> %6, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %8 = extractelement <4 x i32> %6, i64 2
  %9 = extractelement <4 x i32> %6, i64 3
  tail call void @air.write_texture_cube_array.v4f32(ptr addrspace(1) captures(none) %0, <2 x i32> %7, i32 %8, i32 %9, <4 x float> <float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00>, i32 0, i32 2) #9
  ret void
}

define void @oob_buffer_write(ptr addrspace(1) captures(none) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, i32 noundef %2) local_unnamed_addr #3 {
  %4 = zext i32 %2 to i64
  %5 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %1, i64 %4
  %6 = load <4 x i32>, ptr addrspace(1) %5, align 16
  %7 = extractelement <4 x i32> %6, i64 0
  tail call void @air.write_texture_buffer_1d.v4f32(ptr addrspace(1) captures(none) %0, i32 %7, <4 x float> <float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00>, i32 2) #9
  ret void
}

declare ptr addrspace(2) @air.get_read_sampler() local_unnamed_addr #4

declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32) local_unnamed_addr #5

declare { <4 x i32>, i8 } @air.read_texture_2d.u.v4i32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32) local_unnamed_addr #5

declare { <4 x i32>, i8 } @air.read_texture_2d_array.s.v4i32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i32>, i32, <2 x i32>, i32, i32) local_unnamed_addr #5

declare { <4 x float>, i8 } @air.read_texture_3d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <3 x i32>, <3 x i32>, i32, i32) local_unnamed_addr #5

declare { <4 x float>, i8 } @air.read_texture_cube.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i32>, i32, i32, i32) local_unnamed_addr #5

declare { <4 x float>, i8 } @air.read_texture_cube_array.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i32>, i32, i32, i32, i32) local_unnamed_addr #5

declare { <4 x float>, i8 } @air.read_texture_2d_ms.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i32>, i32, i32) local_unnamed_addr #5

declare { <4 x float>, i8 } @air.read_texture_buffer_1d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), i32, i32) local_unnamed_addr #5

declare { float, i8 } @air.read_depth_2d.f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), i32, <2 x i32>, <2 x i32>, i32, i32) local_unnamed_addr #5

declare { <4 x half>, i8 } @air.read_texture_1d.v4f16(ptr addrspace(1) readonly captures(none), ptr addrspace(2), i32, i32, i32) local_unnamed_addr #5

declare void @air.write_texture_cube_array.v4f32(ptr addrspace(1) captures(none), <2 x i32>, i32, i32, <4 x float>, i32, i32) local_unnamed_addr #6

declare void @air.write_texture_buffer_1d.v4f32(ptr addrspace(1) captures(none), i32, <4 x float>, i32) local_unnamed_addr #6

attributes #0 = { mustprogress nofree nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="64" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #2 = { mustprogress nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #3 = { mustprogress nounwind willreturn memory(argmem: readwrite) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #4 = { mustprogress nofree nounwind willreturn memory(inaccessiblemem: read) }
attributes #5 = { mustprogress nofree nounwind willreturn memory(argmem: read) }
attributes #6 = { mustprogress nounwind willreturn memory(argmem: readwrite) }
attributes #7 = { nounwind willreturn memory(inaccessiblemem: read) }
attributes #8 = { nounwind willreturn memory(argmem: read) }
attributes #9 = { nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9, !16, !17, !21, !25, !28, !31, !34, !37, !40, !43, !47, !51, !55}
!air.compile_options = !{!58, !59, !60}
!air.version = !{!62}
!air.language_version = !{!63}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 27, i32 0]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @oob_2d, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15}
!12 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"t"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"uint4", !"air.arg_name", !"p"}
!14 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!15 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
!16 = !{ptr @oob_2d_rgb, !10, !11}
!17 = !{ptr @oob_2d_uint, !10, !18}
!18 = !{!19, !13, !20, !15}
!19 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<uint, read>", !"air.arg_name", !"t"}
!20 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"uint4", !"air.arg_name", !"out"}
!21 = !{ptr @oob_2d_array, !10, !22}
!22 = !{!23, !13, !24, !15}
!23 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d_array<int, read>", !"air.arg_name", !"t"}
!24 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"int4", !"air.arg_name", !"out"}
!25 = !{ptr @oob_3d, !10, !26}
!26 = !{!27, !13, !14, !15}
!27 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture3d<float, read>", !"air.arg_name", !"t"}
!28 = !{ptr @oob_cube, !10, !29}
!29 = !{!30, !13, !14, !15}
!30 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texturecube<float, read>", !"air.arg_name", !"t"}
!31 = !{ptr @oob_cube_array, !10, !32}
!32 = !{!33, !13, !14, !15}
!33 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texturecube_array<float, read>", !"air.arg_name", !"t"}
!34 = !{ptr @oob_ms, !10, !35}
!35 = !{!36, !13, !14, !15}
!36 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d_ms<float, read>", !"air.arg_name", !"t"}
!37 = !{ptr @oob_buffer, !10, !38}
!38 = !{!39, !13, !14, !15}
!39 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture_buffer<float, read>", !"air.arg_name", !"t"}
!40 = !{ptr @oob_rw, !10, !41}
!41 = !{!42, !13, !20, !15}
!42 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d<uint, read_write>", !"air.arg_name", !"t"}
!43 = !{ptr @oob_depth, !10, !44}
!44 = !{!45, !13, !46, !15}
!45 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"depth2d<float, read>", !"air.arg_name", !"t"}
!46 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!47 = !{ptr @oob_1d_half, !10, !48}
!48 = !{!49, !13, !50, !15}
!49 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture1d<half, read>", !"air.arg_name", !"t"}
!50 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"half4", !"air.arg_name", !"out"}
!51 = !{ptr @oob_cube_array_write, !10, !52}
!52 = !{!53, !13, !54}
!53 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texturecube_array<float, write>", !"air.arg_name", !"t"}
!54 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
!55 = !{ptr @oob_buffer_write, !10, !56}
!56 = !{!57, !13, !54}
!57 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture_buffer<float, write>", !"air.arg_name", !"t"}
!58 = !{!"air.compile.denorms_disable"}
!59 = !{!"air.compile.fast_math_enable"}
!60 = !{!"air.compile.framebuffer_fetch_enable"}
!62 = !{i32 2, i32 8, i32 0}
!63 = !{!"Metal", i32 4, i32 0, i32 0}
"##;

fn assemble() -> Option<Vec<u8>> {
    assemble_text("texture_oob", MODULE)
}

fn assemble_text(name: &str, text: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-texture-oob-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join(format!("{name}.ll"));
    std::fs::write(&ll, text).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}"),
    }
}

/// Three reads whose alpha reaches the output through a lane, compiled as [`MODULE`] is (`xcrun
/// metal -std=metal4.0 -O2`, AIR 2.8) from this MSL, written for this test, with the debug, alias,
/// and type metadata removed:
///
/// ```metal
/// kernel void oob_2d_x5(texture2d<float, access::read> t [[texture(0)]],
///                       device const uint4 *p [[buffer(0)]], device float4 *out [[buffer(1)]],
///                       uint i [[thread_position_in_grid]]) {
///     float4 v = t.read(p[i].xy, p[i].z);
///     v.x = 5.0;
///     out[i] = v;
/// }
/// // oob_2d_w: device float *out, out[i] = t.read(p[i].xy, p[i].z).w;
/// // oob_2d_wzyx: out[i] = t.read(p[i].xy, p[i].z).wzyx;
/// ```
const LANES: &str = r##"
source_filename = "-"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @oob_2d_x5(ptr addrspace(1) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = zext i32 %3 to i64
  %6 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %1, i64 %5
  %7 = load <4 x i32>, ptr addrspace(1) %6, align 16
  %8 = shufflevector <4 x i32> %7, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %9 = extractelement <4 x i32> %7, i64 2
  %10 = tail call ptr addrspace(2) @air.get_read_sampler() #3
  %11 = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) %10, <2 x i32> %8, <2 x i32> zeroinitializer, i32 %9, i32 1) #4
  %12 = extractvalue { <4 x float>, i8 } %11, 0
  %13 = insertelement <4 x float> %12, float 5.000000e+00, i64 0
  %14 = getelementptr inbounds <4 x float>, ptr addrspace(1) %2, i64 %5
  store <4 x float> %13, ptr addrspace(1) %14, align 16
  ret void
}

define void @oob_2d_w(ptr addrspace(1) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = zext i32 %3 to i64
  %6 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %1, i64 %5
  %7 = load <4 x i32>, ptr addrspace(1) %6, align 16
  %8 = shufflevector <4 x i32> %7, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %9 = extractelement <4 x i32> %7, i64 2
  %10 = tail call ptr addrspace(2) @air.get_read_sampler() #3
  %11 = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) %10, <2 x i32> %8, <2 x i32> zeroinitializer, i32 %9, i32 1) #4
  %12 = extractvalue { <4 x float>, i8 } %11, 0
  %13 = extractelement <4 x float> %12, i64 3
  %14 = getelementptr inbounds float, ptr addrspace(1) %2, i64 %5
  store float %13, ptr addrspace(1) %14, align 4
  ret void
}

define void @oob_2d_wzyx(ptr addrspace(1) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = zext i32 %3 to i64
  %6 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %1, i64 %5
  %7 = load <4 x i32>, ptr addrspace(1) %6, align 16
  %8 = shufflevector <4 x i32> %7, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %9 = extractelement <4 x i32> %7, i64 2
  %10 = tail call ptr addrspace(2) @air.get_read_sampler() #3
  %11 = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) %10, <2 x i32> %8, <2 x i32> zeroinitializer, i32 %9, i32 1) #4
  %12 = extractvalue { <4 x float>, i8 } %11, 0
  %13 = shufflevector <4 x float> %12, <4 x float> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  %14 = getelementptr inbounds <4 x float>, ptr addrspace(1) %2, i64 %5
  store <4 x float> %13, ptr addrspace(1) %14, align 16
  ret void
}

declare ptr addrspace(2) @air.get_read_sampler() local_unnamed_addr #1

declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32) local_unnamed_addr #2

attributes #0 = { mustprogress nofree nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nounwind willreturn memory(inaccessiblemem: read) }
attributes #2 = { mustprogress nofree nounwind willreturn memory(argmem: read) }
attributes #3 = { nounwind willreturn memory(inaccessiblemem: read) }
attributes #4 = { nounwind willreturn memory(argmem: read) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9, !16, !19}
!air.compile_options = !{!20, !21, !22}
!air.version = !{!24}
!air.language_version = !{!25}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 27, i32 0]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @oob_2d_x5, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15}
!12 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"t"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"uint4", !"air.arg_name", !"p"}
!14 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!15 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
!16 = !{ptr @oob_2d_w, !10, !17}
!17 = !{!12, !13, !18, !15}
!18 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!19 = !{ptr @oob_2d_wzyx, !10, !11}
!20 = !{!"air.compile.denorms_disable"}
!21 = !{!"air.compile.fast_math_enable"}
!22 = !{!"air.compile.framebuffer_fetch_enable"}
!24 = !{i32 2, i32 8, i32 0}
!25 = !{!"Metal", i32 4, i32 0, i32 0}
"##;

/// Two reads through a sampler, compiled with `xcrun metal -std=metal4.1 -O2` (AIR 2.9) from this
/// MSL, written for this test, with the debug, alias, and type metadata removed:
///
/// ```metal
/// kernel void read_clamped(texture2d<float, access::read> t [[texture(0)]],
///                          device float4 *out [[buffer(0)]],
///                          uint2 i [[thread_position_in_grid]]) {
///     constexpr sampler s(address::clamp_to_edge);
///     out[i.x] = t.read(s, int2(i) - int2(1, 1));
/// }
/// kernel void read_arg_sampler(texture2d<float, access::read> t [[texture(0)]],
///                              sampler s [[sampler(0)]],
///                              device float4 *out [[buffer(0)]],
///                              uint2 i [[thread_position_in_grid]]) {
///     out[i.x] = t.read(s, int2(i) - int2(1, 1));
/// }
/// ```
const SAMPLED_READS: &str = r##"
source_filename = "read_sampler.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v29-apple-macosx27.0.0"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601017929, i64 0], align 8

define void @read_clamped(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %1, <2 x i32> noundef %2) local_unnamed_addr #0 {
  %4 = add <2 x i32> %2, splat (i32 -1)
  %5 = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) @__air_sampler_state, <2 x i32> %4, <2 x i32> zeroinitializer, i32 0, i32 1) #2
  %6 = extractvalue { <4 x float>, i8 } %5, 0
  %7 = extractelement <2 x i32> %2, i64 0
  %8 = zext i32 %7 to i64
  %9 = getelementptr inbounds <4 x float>, ptr addrspace(1) %1, i64 %8
  store <4 x float> %6, ptr addrspace(1) %9, align 16
  ret void
}

define void @read_arg_sampler(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, <2 x i32> noundef %3) local_unnamed_addr #0 {
  %5 = add <2 x i32> %3, splat (i32 -1)
  %6 = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) %1, <2 x i32> %5, <2 x i32> zeroinitializer, i32 0, i32 1) #2
  %7 = extractvalue { <4 x float>, i8 } %6, 0
  %8 = extractelement <2 x i32> %3, i64 0
  %9 = zext i32 %8 to i64
  %10 = getelementptr inbounds <4 x float>, ptr addrspace(1) %2, i64 %9
  store <4 x float> %7, ptr addrspace(1) %10, align 16
  ret void
}

declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nounwind willreturn memory(argmem: readwrite) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nounwind willreturn memory(argmem: read) }
attributes #2 = { nounwind willreturn memory(argmem: read) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9, !15}
!air.compile_options = !{!20, !21, !22}
!air.sampler_states = !{!23}
!air.version = !{!25}
!air.language_version = !{!26}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 27, i32 0]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @read_clamped, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14}
!12 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"t"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!14 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"i"}
!15 = !{ptr @read_arg_sampler, !10, !16}
!16 = !{!12, !17, !18, !19}
!17 = !{i32 1, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}
!18 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!19 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"i"}
!20 = !{!"air.compile.denorms_disable"}
!21 = !{!"air.compile.fast_math_enable"}
!22 = !{!"air.compile.framebuffer_fetch_enable"}
!23 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
!25 = !{i32 2, i32 9, i32 0}
!26 = !{!"Metal", i32 4, i32 1, i32 0}
"##;

/// Reads through constexpr samplers, compiled with `xcrun metal -std=metal4.1 -O2` (Apple metal
/// 32023.921, AIR 2.9) from this MSL, written for this test, with the alias, type, and source
/// metadata removed:
///
/// ```metal
/// constexpr sampler edge(address::clamp_to_edge);
/// kernel void edge_2d(texture2d<float, access::read> t [[texture(0)]],
///                     device const int4 *p [[buffer(0)]], device float4 *out [[buffer(1)]],
///                     uint i [[thread_position_in_grid]]) {
///     out[i] = t.read(edge, p[i].xy, int2(0), uint(p[i].z));
/// }
/// // edge_2d_offset: texture2d<uint>, out[i] = t.read(edge, p[i].xy, int2(-2, 3));
/// // edge_2d_array: texture2d_array<int>,
/// //     out[i] = t.read(edge, p[i].xy, uint(p[i].z), int2(0), uint(p[i].w));
/// // edge_3d: texture3d<float>, out[i] = t.read(edge, p[i].xyz, int3(0), uint(p[i].w));
/// // edge_cube: texturecube<float>, out[i] = t.read(edge, p[i].xy, uint(p[i].z), uint(p[i].w));
/// // edge_ms: texture2d_ms<float>, out[i] = t.read(edge, p[i].xy, uint(p[i].z));
/// // edge_buffer: texture_buffer<float>, out[i] = t.read(edge, p[i].x);
/// // edge_rw: texture2d<uint, access::read_write>, out[i] = t.read(edge, p[i].xy);
/// // edge_1d: texture1d<float>, out[i] = t.read(edge, p[i].x);
/// // edge_dynamic_offset: texture2d<float>, out[i] = t.read(edge, p[i].xy, p[i].zw);
/// // read_repeat: constexpr sampler s(address::repeat); out[i] = t.read(s, p[i].xy);
/// // read_linear: constexpr sampler s(address::clamp_to_edge, filter::linear);
/// //     out[i] = t.read(s, p[i].xy);
/// ```
const EDGE_READS: &str = r##"
source_filename = "-"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v29-apple-macosx27.0.0"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601017929, i64 0], align 8
@__air_sampler_state.1 = internal addrspace(2) constant [2 x i64] [i64 34901797601018002, i64 0], align 8
@__air_sampler_state.2 = internal addrspace(2) constant [2 x i64] [i64 34901797601020489, i64 0], align 8

define void @edge_2d(ptr addrspace(1) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = zext i32 %3 to i64
  %6 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %1, i64 %5
  %7 = load <4 x i32>, ptr addrspace(1) %6, align 16
  %8 = shufflevector <4 x i32> %7, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %9 = extractelement <4 x i32> %7, i64 2
  %10 = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) @__air_sampler_state, <2 x i32> %8, <2 x i32> zeroinitializer, i32 %9, i32 1) #3
  %11 = extractvalue { <4 x float>, i8 } %10, 0
  %12 = getelementptr inbounds <4 x float>, ptr addrspace(1) %2, i64 %5
  store <4 x float> %11, ptr addrspace(1) %12, align 16
  ret void
}

define void @edge_2d_offset(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #1 {
  %5 = zext i32 %3 to i64
  %6 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %1, i64 %5
  %7 = load <4 x i32>, ptr addrspace(1) %6, align 16
  %8 = shufflevector <4 x i32> %7, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %9 = tail call { <4 x i32>, i8 } @air.read_texture_2d.u.v4i32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) @__air_sampler_state, <2 x i32> %8, <2 x i32> <i32 -2, i32 3>, i32 0, i32 1) #3
  %10 = extractvalue { <4 x i32>, i8 } %9, 0
  %11 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %2, i64 %5
  store <4 x i32> %10, ptr addrspace(1) %11, align 16
  ret void
}

define void @edge_2d_array(ptr addrspace(1) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = zext i32 %3 to i64
  %6 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %1, i64 %5
  %7 = load <4 x i32>, ptr addrspace(1) %6, align 16
  %8 = shufflevector <4 x i32> %7, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %9 = extractelement <4 x i32> %7, i64 2
  %10 = extractelement <4 x i32> %7, i64 3
  %11 = tail call { <4 x i32>, i8 } @air.read_texture_2d_array.s.v4i32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) @__air_sampler_state, <2 x i32> %8, i32 %9, <2 x i32> zeroinitializer, i32 %10, i32 1) #3
  %12 = extractvalue { <4 x i32>, i8 } %11, 0
  %13 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %2, i64 %5
  store <4 x i32> %12, ptr addrspace(1) %13, align 16
  ret void
}

define void @edge_3d(ptr addrspace(1) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = zext i32 %3 to i64
  %6 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %1, i64 %5
  %7 = load <4 x i32>, ptr addrspace(1) %6, align 16
  %8 = shufflevector <4 x i32> %7, <4 x i32> poison, <3 x i32> <i32 0, i32 1, i32 2>
  %9 = extractelement <4 x i32> %7, i64 3
  %10 = tail call { <4 x float>, i8 } @air.read_texture_3d.v4f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) @__air_sampler_state, <3 x i32> %8, <3 x i32> zeroinitializer, i32 %9, i32 1) #3
  %11 = extractvalue { <4 x float>, i8 } %10, 0
  %12 = getelementptr inbounds <4 x float>, ptr addrspace(1) %2, i64 %5
  store <4 x float> %11, ptr addrspace(1) %12, align 16
  ret void
}

define void @edge_cube(ptr addrspace(1) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = zext i32 %3 to i64
  %6 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %1, i64 %5
  %7 = load <4 x i32>, ptr addrspace(1) %6, align 16
  %8 = shufflevector <4 x i32> %7, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %9 = extractelement <4 x i32> %7, i64 2
  %10 = extractelement <4 x i32> %7, i64 3
  %11 = tail call { <4 x float>, i8 } @air.read_texture_cube.v4f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) @__air_sampler_state, <2 x i32> %8, i32 %9, i32 %10, i32 1) #3
  %12 = extractvalue { <4 x float>, i8 } %11, 0
  %13 = getelementptr inbounds <4 x float>, ptr addrspace(1) %2, i64 %5
  store <4 x float> %12, ptr addrspace(1) %13, align 16
  ret void
}

define void @edge_ms(ptr addrspace(1) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = zext i32 %3 to i64
  %6 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %1, i64 %5
  %7 = load <4 x i32>, ptr addrspace(1) %6, align 16
  %8 = shufflevector <4 x i32> %7, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %9 = extractelement <4 x i32> %7, i64 2
  %10 = tail call { <4 x float>, i8 } @air.read_texture_2d_ms.v4f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) @__air_sampler_state, <2 x i32> %8, i32 %9, i32 1) #3
  %11 = extractvalue { <4 x float>, i8 } %10, 0
  %12 = getelementptr inbounds <4 x float>, ptr addrspace(1) %2, i64 %5
  store <4 x float> %11, ptr addrspace(1) %12, align 16
  ret void
}

define void @edge_buffer(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #1 {
  %5 = zext i32 %3 to i64
  %6 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %1, i64 %5
  %7 = load <4 x i32>, ptr addrspace(1) %6, align 16
  %8 = extractelement <4 x i32> %7, i64 0
  %9 = tail call { <4 x float>, i8 } @air.read_texture_buffer_1d.v4f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) @__air_sampler_state, i32 %8, i32 1) #3
  %10 = extractvalue { <4 x float>, i8 } %9, 0
  %11 = getelementptr inbounds <4 x float>, ptr addrspace(1) %2, i64 %5
  store <4 x float> %10, ptr addrspace(1) %11, align 16
  ret void
}

define void @edge_rw(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #1 {
  %5 = zext i32 %3 to i64
  %6 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %1, i64 %5
  %7 = load <4 x i32>, ptr addrspace(1) %6, align 16
  %8 = shufflevector <4 x i32> %7, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %9 = tail call { <4 x i32>, i8 } @air.read_texture_2d.u.v4i32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) @__air_sampler_state, <2 x i32> %8, <2 x i32> zeroinitializer, i32 0, i32 3) #3
  %10 = extractvalue { <4 x i32>, i8 } %9, 0
  %11 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %2, i64 %5
  store <4 x i32> %10, ptr addrspace(1) %11, align 16
  ret void
}

define void @edge_1d(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #1 {
  %5 = zext i32 %3 to i64
  %6 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %1, i64 %5
  %7 = load <4 x i32>, ptr addrspace(1) %6, align 16
  %8 = extractelement <4 x i32> %7, i64 0
  %9 = tail call { <4 x float>, i8 } @air.read_texture_1d.v4f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) @__air_sampler_state, i32 %8, i32 0, i32 1) #3
  %10 = extractvalue { <4 x float>, i8 } %9, 0
  %11 = getelementptr inbounds <4 x float>, ptr addrspace(1) %2, i64 %5
  store <4 x float> %10, ptr addrspace(1) %11, align 16
  ret void
}

define void @edge_dynamic_offset(ptr addrspace(1) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = zext i32 %3 to i64
  %6 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %1, i64 %5
  %7 = load <4 x i32>, ptr addrspace(1) %6, align 16
  %8 = shufflevector <4 x i32> %7, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %9 = shufflevector <4 x i32> %7, <4 x i32> poison, <2 x i32> <i32 2, i32 3>
  %10 = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) @__air_sampler_state, <2 x i32> %8, <2 x i32> %9, i32 0, i32 1) #3
  %11 = extractvalue { <4 x float>, i8 } %10, 0
  %12 = getelementptr inbounds <4 x float>, ptr addrspace(1) %2, i64 %5
  store <4 x float> %11, ptr addrspace(1) %12, align 16
  ret void
}

define void @read_repeat(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #1 {
  %5 = zext i32 %3 to i64
  %6 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %1, i64 %5
  %7 = load <4 x i32>, ptr addrspace(1) %6, align 16
  %8 = shufflevector <4 x i32> %7, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %9 = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) @__air_sampler_state.1, <2 x i32> %8, <2 x i32> zeroinitializer, i32 0, i32 1) #3
  %10 = extractvalue { <4 x float>, i8 } %9, 0
  %11 = getelementptr inbounds <4 x float>, ptr addrspace(1) %2, i64 %5
  store <4 x float> %10, ptr addrspace(1) %11, align 16
  ret void
}

define void @read_linear(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #1 {
  %5 = zext i32 %3 to i64
  %6 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %1, i64 %5
  %7 = load <4 x i32>, ptr addrspace(1) %6, align 16
  %8 = shufflevector <4 x i32> %7, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %9 = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) @__air_sampler_state.2, <2 x i32> %8, <2 x i32> zeroinitializer, i32 0, i32 1) #3
  %10 = extractvalue { <4 x float>, i8 } %9, 0
  %11 = getelementptr inbounds <4 x float>, ptr addrspace(1) %2, i64 %5
  store <4 x float> %10, ptr addrspace(1) %11, align 16
  ret void
}

declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32) local_unnamed_addr #2

declare { <4 x i32>, i8 } @air.read_texture_2d.u.v4i32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32) local_unnamed_addr #2

declare { <4 x i32>, i8 } @air.read_texture_2d_array.s.v4i32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i32>, i32, <2 x i32>, i32, i32) local_unnamed_addr #2

declare { <4 x float>, i8 } @air.read_texture_3d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <3 x i32>, <3 x i32>, i32, i32) local_unnamed_addr #2

declare { <4 x float>, i8 } @air.read_texture_cube.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i32>, i32, i32, i32) local_unnamed_addr #2

declare { <4 x float>, i8 } @air.read_texture_2d_ms.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i32>, i32, i32) local_unnamed_addr #2

declare { <4 x float>, i8 } @air.read_texture_buffer_1d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), i32, i32) local_unnamed_addr #2

declare { <4 x float>, i8 } @air.read_texture_1d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), i32, i32, i32) local_unnamed_addr #2

attributes #0 = { mustprogress nofree nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nounwind willreturn memory(argmem: readwrite) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #2 = { mustprogress nofree nounwind willreturn memory(argmem: read) }
attributes #3 = { nounwind willreturn memory(argmem: read) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9, !16, !20, !24, !27, !30, !33, !36, !39, !42, !43, !44}
!air.compile_options = !{!45, !46, !47}
!air.sampler_states = !{!48, !49, !50}
!air.version = !{!52}
!air.language_version = !{!53}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 27, i32 0]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @edge_2d, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15}
!12 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"t"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"int4", !"air.arg_name", !"p"}
!14 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!15 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
!16 = !{ptr @edge_2d_offset, !10, !17}
!17 = !{!18, !13, !19, !15}
!18 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<uint, read>", !"air.arg_name", !"t"}
!19 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"uint4", !"air.arg_name", !"out"}
!20 = !{ptr @edge_2d_array, !10, !21}
!21 = !{!22, !13, !23, !15}
!22 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d_array<int, read>", !"air.arg_name", !"t"}
!23 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"int4", !"air.arg_name", !"out"}
!24 = !{ptr @edge_3d, !10, !25}
!25 = !{!26, !13, !14, !15}
!26 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture3d<float, read>", !"air.arg_name", !"t"}
!27 = !{ptr @edge_cube, !10, !28}
!28 = !{!29, !13, !14, !15}
!29 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texturecube<float, read>", !"air.arg_name", !"t"}
!30 = !{ptr @edge_ms, !10, !31}
!31 = !{!32, !13, !14, !15}
!32 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d_ms<float, read>", !"air.arg_name", !"t"}
!33 = !{ptr @edge_buffer, !10, !34}
!34 = !{!35, !13, !14, !15}
!35 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture_buffer<float, read>", !"air.arg_name", !"t"}
!36 = !{ptr @edge_rw, !10, !37}
!37 = !{!38, !13, !19, !15}
!38 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d<uint, read_write>", !"air.arg_name", !"t"}
!39 = !{ptr @edge_1d, !10, !40}
!40 = !{!41, !13, !14, !15}
!41 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture1d<float, read>", !"air.arg_name", !"t"}
!42 = !{ptr @edge_dynamic_offset, !10, !11}
!43 = !{ptr @read_repeat, !10, !11}
!44 = !{ptr @read_linear, !10, !11}
!45 = !{!"air.compile.denorms_disable"}
!46 = !{!"air.compile.fast_math_enable"}
!47 = !{!"air.compile.framebuffer_fetch_enable"}
!48 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
!49 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state.1}
!50 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state.2}
!52 = !{i32 2, i32 9, i32 0}
!53 = !{!"Metal", i32 4, i32 1, i32 0}
"##;

/// Metal 4.1's `sparse_read` through a constexpr clamp-to-edge sampler, compiled with
/// `xcrun metal -std=metal4.1 -O2` (Apple metal 32023.921, AIR 2.9) from this MSL, written for
/// this test, with the alias, type, and source metadata removed. It is the same
/// `air.read_texture_2d` call as `read`'s, its status word observed in `sparse_edge` and unused in
/// `edge_plain`:
///
/// ```metal
/// constexpr sampler edge(address::clamp_to_edge);
/// kernel void sparse_edge(texture2d<float, access::read> t [[texture(0)]],
///                         device const int4 *p [[buffer(0)]], device float4 *out [[buffer(1)]],
///                         uint i [[thread_position_in_grid]]) {
///     sparse_color<float4> c = t.sparse_read(edge, p[i].xy, int2(0), uint(p[i].z));
///     float4 v = c.value();
///     v.w = c.resident() ? 1.0 : 0.0;
///     out[i] = v;
/// }
/// // edge_plain: out[i] = t.sparse_read(edge, p[i].xy, int2(0), uint(p[i].z)).value();
/// ```
const SPARSE_EDGE_READS: &str = r##"
source_filename = "-"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v29-apple-macosx27.0.0"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601017929, i64 0], align 8

define void @sparse_edge(ptr addrspace(1) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = zext i32 %3 to i64
  %6 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %1, i64 %5
  %7 = load <4 x i32>, ptr addrspace(1) %6, align 16
  %8 = shufflevector <4 x i32> %7, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %9 = extractelement <4 x i32> %7, i64 2
  %10 = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) @__air_sampler_state, <2 x i32> %8, <2 x i32> zeroinitializer, i32 %9, i32 1) #2
  %11 = extractvalue { <4 x float>, i8 } %10, 1
  %12 = extractvalue { <4 x float>, i8 } %10, 0
  %13 = and i8 %11, 1
  %14 = icmp eq i8 %13, 0
  %15 = select fast i1 %14, float 1.000000e+00, float 0.000000e+00
  %16 = insertelement <4 x float> %12, float %15, i64 3
  %17 = getelementptr inbounds <4 x float>, ptr addrspace(1) %2, i64 %5
  store <4 x float> %16, ptr addrspace(1) %17, align 16
  ret void
}

define void @edge_plain(ptr addrspace(1) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = zext i32 %3 to i64
  %6 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %1, i64 %5
  %7 = load <4 x i32>, ptr addrspace(1) %6, align 16
  %8 = shufflevector <4 x i32> %7, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %9 = extractelement <4 x i32> %7, i64 2
  %10 = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) @__air_sampler_state, <2 x i32> %8, <2 x i32> zeroinitializer, i32 %9, i32 1) #2
  %11 = extractvalue { <4 x float>, i8 } %10, 0
  %12 = getelementptr inbounds <4 x float>, ptr addrspace(1) %2, i64 %5
  store <4 x float> %11, ptr addrspace(1) %12, align 16
  ret void
}

declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nounwind willreturn memory(argmem: read) }
attributes #2 = { nounwind willreturn memory(argmem: read) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9, !16}
!air.compile_options = !{!17, !18, !19}
!air.sampler_states = !{!20}
!air.version = !{!21}
!air.language_version = !{!22}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 27, i32 0]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @sparse_edge, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15}
!12 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"t"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"int4", !"air.arg_name", !"p"}
!14 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!15 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
!16 = !{ptr @edge_plain, !10, !11}
!17 = !{!"air.compile.denorms_disable"}
!18 = !{!"air.compile.fast_math_enable"}
!19 = !{!"air.compile.framebuffer_fetch_enable"}
!20 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
!21 = !{i32 2, i32 9, i32 0}
!22 = !{!"Metal", i32 4, i32 1, i32 0}
"##;

/// Kernels that write a texel and then read it back in the same thread, compiled as [`MODULE`]
/// is (`xcrun metal -std=metal4.0 -O2`, AIR 2.8, Apple metal 32023.921) from this MSL, written for
/// this test, with the alias, type, and source metadata removed:
///
/// ```metal
/// kernel void fwd_rw(texture2d<uint, access::read_write> t [[texture(0)]],
///                    device const uint4 *p [[buffer(0)]], device uint4 *out [[buffer(1)]],
///                    uint i [[thread_position_in_grid]]) {
///     t.write(uint4(p[i].z), p[i].xy);
///     out[i] = t.read(p[i].xy);
/// }
/// // fwd_buffer: texture_buffer<uint, access::read_write>, t.write(uint4(p[i].z), p[i].x);
/// //     out[i] = t.read(p[i].x);
/// // fwd_write_only: texture2d<uint, access::write>, no out, t.write(uint4(p[i].z), p[i].xy);
/// // fwd_read_only_rw: texture2d<uint, access::read_write>, out[i] = t.read(p[i].xy);
/// ```
const FORWARD: &str = r##"
source_filename = "-"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @fwd_rw(ptr addrspace(1) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = zext i32 %3 to i64
  %6 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %1, i64 %5
  %7 = load <4 x i32>, ptr addrspace(1) %6, align 16
  %8 = shufflevector <4 x i32> %7, <4 x i32> undef, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %9 = shufflevector <4 x i32> %7, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  tail call void @air.write_texture_2d.u.v4i32(ptr addrspace(1) captures(none) %0, <2 x i32> %9, <4 x i32> %8, i32 0, i32 3) #5
  %10 = tail call ptr addrspace(2) @air.get_read_sampler() #6
  %11 = tail call { <4 x i32>, i8 } @air.read_texture_2d.u.v4i32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) %10, <2 x i32> %9, <2 x i32> zeroinitializer, i32 0, i32 3) #7
  %12 = extractvalue { <4 x i32>, i8 } %11, 0
  %13 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %2, i64 %5
  store <4 x i32> %12, ptr addrspace(1) %13, align 16
  ret void
}

define void @fwd_buffer(ptr addrspace(1) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = zext i32 %3 to i64
  %6 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %1, i64 %5
  %7 = load <4 x i32>, ptr addrspace(1) %6, align 16
  %8 = shufflevector <4 x i32> %7, <4 x i32> undef, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %9 = extractelement <4 x i32> %7, i64 0
  tail call void @air.write_texture_buffer_1d.u.v4i32(ptr addrspace(1) captures(none) %0, i32 %9, <4 x i32> %8, i32 3) #5
  %10 = tail call ptr addrspace(2) @air.get_read_sampler() #6
  %11 = tail call { <4 x i32>, i8 } @air.read_texture_buffer_1d.u.v4i32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) %10, i32 %9, i32 3) #7
  %12 = extractvalue { <4 x i32>, i8 } %11, 0
  %13 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %2, i64 %5
  store <4 x i32> %12, ptr addrspace(1) %13, align 16
  ret void
}

define void @fwd_write_only(ptr addrspace(1) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = zext i32 %2 to i64
  %5 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %1, i64 %4
  %6 = load <4 x i32>, ptr addrspace(1) %5, align 16
  %7 = shufflevector <4 x i32> %6, <4 x i32> undef, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %8 = shufflevector <4 x i32> %6, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  tail call void @air.write_texture_2d.u.v4i32(ptr addrspace(1) captures(none) %0, <2 x i32> %8, <4 x i32> %7, i32 0, i32 2) #5
  ret void
}

define void @fwd_read_only_rw(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #1 {
  %5 = zext i32 %3 to i64
  %6 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %1, i64 %5
  %7 = load <4 x i32>, ptr addrspace(1) %6, align 16
  %8 = shufflevector <4 x i32> %7, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %9 = tail call ptr addrspace(2) @air.get_read_sampler() #6
  %10 = tail call { <4 x i32>, i8 } @air.read_texture_2d.u.v4i32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) %9, <2 x i32> %8, <2 x i32> zeroinitializer, i32 0, i32 3) #7
  %11 = extractvalue { <4 x i32>, i8 } %10, 0
  %12 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %2, i64 %5
  store <4 x i32> %11, ptr addrspace(1) %12, align 16
  ret void
}

define void @fwd_atomic(ptr addrspace(1) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = zext i32 %3 to i64
  %6 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %1, i64 %5
  %7 = load <4 x i32>, ptr addrspace(1) %6, align 16
  %8 = shufflevector <4 x i32> %7, <4 x i32> undef, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %9 = shufflevector <4 x i32> %7, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %10 = tail call <4 x i32> @air.atomic_fetch_max_explicit_texture_2d.u.v4i32(ptr addrspace(1) %0, <2 x i32> %9, <2 x i32> zeroinitializer, <4 x i32> %8, i32 0, i32 3) #5
  %11 = extractelement <4 x i32> %10, i64 0
  %12 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 %5
  store i32 %11, ptr addrspace(1) %12, align 4
  ret void
}

declare void @air.write_texture_2d.u.v4i32(ptr addrspace(1) captures(none), <2 x i32>, <4 x i32>, i32, i32) local_unnamed_addr #2

declare <4 x i32> @air.atomic_fetch_max_explicit_texture_2d.u.v4i32(ptr addrspace(1) captures(none), <2 x i32>, <2 x i32>, <4 x i32>, i32, i32) local_unnamed_addr #2

declare ptr addrspace(2) @air.get_read_sampler() local_unnamed_addr #3

declare { <4 x i32>, i8 } @air.read_texture_2d.u.v4i32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32) local_unnamed_addr #4

declare void @air.write_texture_buffer_1d.u.v4i32(ptr addrspace(1) captures(none), i32, <4 x i32>, i32) local_unnamed_addr #2

declare { <4 x i32>, i8 } @air.read_texture_buffer_1d.u.v4i32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), i32, i32) local_unnamed_addr #4

attributes #0 = { mustprogress nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #2 = { mustprogress nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { mustprogress nofree nounwind willreturn memory(inaccessiblemem: read) }
attributes #4 = { mustprogress nofree nounwind willreturn memory(argmem: read) }
attributes #5 = { nounwind willreturn memory(argmem: readwrite) }
attributes #6 = { nounwind willreturn memory(inaccessiblemem: read) }
attributes #7 = { nounwind willreturn memory(argmem: read) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9, !16, !19, !23, !30}
!air.compile_options = !{!24, !25, !26}
!air.version = !{!28}
!air.language_version = !{!29}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 27, i32 0]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @fwd_rw, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15}
!12 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d<uint, read_write>", !"air.arg_name", !"t"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"uint4", !"air.arg_name", !"p"}
!14 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"uint4", !"air.arg_name", !"out"}
!15 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
!16 = !{ptr @fwd_buffer, !10, !17}
!17 = !{!18, !13, !14, !15}
!18 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_name", !"texture_buffer<uint, read_write>", !"air.arg_name", !"t"}
!19 = !{ptr @fwd_write_only, !10, !20}
!20 = !{!21, !13, !22}
!21 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<uint, write>", !"air.arg_name", !"t"}
!22 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
!23 = !{ptr @fwd_read_only_rw, !10, !11}
!24 = !{!"air.compile.denorms_disable"}
!25 = !{!"air.compile.fast_math_enable"}
!26 = !{!"air.compile.framebuffer_fetch_enable"}
!28 = !{i32 2, i32 8, i32 0}
!29 = !{!"Metal", i32 4, i32 0, i32 0}
!30 = !{ptr @fwd_atomic, !10, !11}
"##;

/// `stem` with the robustness features in `drop` removed.
fn without(stem: &str, drop: &[VkFeature]) -> DeviceProfile {
    device(stem, drop).profile().clone()
}

/// The committed profile `stem` as a device, with the features in `drop` removed.
fn device(stem: &str, drop: &[VkFeature]) -> Device {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    for f in drop {
        desc.features.remove(f);
    }
    Device::new(desc, DevicePolicy::default()).expect("device")
}

/// Every committed profile, one with `robustImageAccess` but not `robustImageAccess2`, and two
/// without any robustness.
fn profiles() -> Vec<(String, DeviceProfile)> {
    let mut out = committed_profiles().expect("profiles");
    out.push((
        "moltenvk-m3 robust1".into(),
        without("moltenvk-m3", &[VkFeature::RobustImageAccess2]),
    ));
    let all = [
        VkFeature::RobustImageAccess,
        VkFeature::RobustImageAccess2,
        VkFeature::RobustBufferAccess,
        VkFeature::RobustBufferAccess2,
    ];
    for stem in ["moltenvk-m3", "lavapipe"] {
        out.push((format!("{stem} unrobust"), without(stem, &all)));
    }
    out
}

/// How a profile contains an image or a texel buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Class {
    /// The device reads Metal's out-of-bounds texel and discards out-of-bounds writes.
    Device,
    /// The shader checks reads; the device discards out-of-bounds writes.
    ShaderReads,
    /// The shader checks reads and guards writes; the device contains nothing.
    Shader,
}

fn class(p: &DeviceProfile, buffer: bool) -> Class {
    let f = p.fields();
    let Containment::On { robust } = f.containment else {
        panic!("containment off")
    };
    match (buffer, robust) {
        (true, Robustness::Rba2) => Class::Device,
        (true, _) => Class::Shader,
        (false, _) if f.images.robust_image_access2 => Class::Device,
        (false, _) if f.images.robust_image_access => Class::ShaderReads,
        (false, _) => Class::Shader,
    }
}

/// What the interpreter's image does outside itself on `class`, for a format whose missing alpha
/// reads `alpha`: robustness's texel (Metal's), a poisoned texel where the device's is not
/// Metal's, or a trap.
fn robust(class: Class, alpha: u32) -> Option<[u32; 4]> {
    match class {
        Class::Device => Some([0, 0, 0, alpha]),
        Class::ShaderReads => Some([0xDEAD_BEEF; 4]),
        Class::Shader => None,
    }
}

/// The plan and module of `entry` on `p`, or `None` when `p` needs a runtime input for it (a
/// storage texture format without a formatless load or store).
fn module(bc: &[u8], entry: &str, stem: &str, p: &DeviceProfile) -> Option<(Plan, TypedModule)> {
    match typed_module(bc, entry, p) {
        Ok(m) => Some(m),
        Err(Error::NeedsInput(_)) => None,
        Err(e) => panic!("{stem}: {entry}: {e}"),
    }
}

const TEXTURE: SlotRef = SlotRef {
    stage: Stage::Kernel,
    class: SlotClass::Texture,
    index: 0,
};

/// One thread per entry of `p`: `[[buffer(0)]]` holds `p`, `[[buffer(1)]]` `out` bytes, and the
/// texture's level count is `levels` (a storage texture's read at a level reads it).
fn resources(plan: &Plan, p: &[[u32; 4]], out: usize, levels: u32) -> Resources {
    let mut spec = BTreeMap::new();
    for axis in [Axis::X, Axis::Y, Axis::Z] {
        spec.insert(SpecKey::LocalSize(axis), 1);
    }
    spec.insert(SpecKey::FlatLocalSize, 1);
    let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        match (f.field, f.loc) {
            (PushField::TextureLevels(TEXTURE), PushLoc::Push { offset }) => {
                push[offset as usize..offset as usize + 4].copy_from_slice(&levels.to_le_bytes());
            }
            (PushField::SlotBindOffset(_) | PushField::DispatchRegion, PushLoc::Push { .. }) => {}
            // A bound value read as data, which the interpreter writes from `spec`.
            (PushField::Specialized(_), PushLoc::Push { .. }) => {}
            // 0: no channel narrower than the written word, every integer lane as written (the
            // interpreter's images hold words; the kernels write in-range values).
            (PushField::TextureIntegerChannels(_), PushLoc::Push { .. }) => {}
            other => panic!("unexpected push field {other:?}"),
        }
    }
    let rows: Vec<u8> = p.iter().flatten().flat_map(|w| w.to_le_bytes()).collect();
    let buffers = [rows, vec![0; out]]
        .into_iter()
        .enumerate()
        .map(|(i, bytes)| Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: i as u32,
            }),
            address: None,
            bytes,
        })
        .collect();
    Resources {
        buffers,
        push,
        spec,
    }
}

/// The images slot 0 binds for `image` under `plan`: `image` itself, or, where the slot is declared
/// as level views (a storage texture read at a level), its views as the binder binds them, element
/// `i` level `i` of `image` alone where it has that level, else a 1x1 neutral.
fn bound(plan: &Plan, image: &Image) -> Vec<Image> {
    let views = plan.families.texture.slots[&0]
        .decl
        .as_ref()
        .and_then(|d| d.level_views);
    let Some(n) = views else {
        return vec![image.clone()];
    };
    (0..n)
        .map(|e| {
            let extent = match e < image.levels {
                true => image.level_extent(e),
                false => [1, 1, 1],
            };
            let mut view =
                Image::new(image.slot, extent, image.layers, 1, image.samples).at_element(e);
            view.robust = image.robust;
            if e < image.levels {
                let [w, h, d] = extent;
                for layer in 0..image.layers {
                    for z in 0..d {
                        for y in 0..h {
                            for x in 0..w {
                                let from = image.index(e, layer, 0, [x, y, z]).expect("texel");
                                let to = view.index(0, layer, 0, [x, y, z]).expect("texel");
                                view.texels[to] = image.texels[from];
                            }
                        }
                    }
                }
            }
            view
        })
        .collect()
}

/// Run `threads` threadgroups of one thread in both scheduling orders over `images` (slot 0's);
/// both must agree. The first image after the run.
fn run_both(
    m: &TypedModule,
    res: &Resources,
    images: &[Image],
    threads: u32,
    stem: &str,
) -> (Vec<Vec<u8>>, Image) {
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups: [threads, 1, 1],
            order,
            ..Config::default()
        };
        let (o, mut imgs) = run_with_images(m, &cfg, res.clone(), images.to_vec())
            .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        outs.push((o.buffers, imgs.remove(0)));
    }
    assert_eq!(outs[0], outs[1], "{stem}: scheduling orders disagree");
    outs.swap_remove(0)
}

fn fl(x: f32) -> u32 {
    x.to_bits()
}

/// The sampled type of a texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Texel {
    Float,
    Int,
}

impl Texel {
    /// A missing alpha's default.
    fn one(self) -> u32 {
        match self {
            Texel::Float => fl(1.0),
            Texel::Int => 1,
        }
    }

    /// A distinct nonzero texel per index, exact in `half`.
    fn fill(self, i: usize) -> [u32; 4] {
        let i = i as u32;
        match self {
            Texel::Float => {
                let x = i as f32;
                [fl(1.0 + x), fl(-1.0 - x), fl(0.5 + x), fl(100.0 + x)]
            }
            Texel::Int => [1 + i, (-1 - i as i32) as u32, 7, 1000 + i],
        }
    }
}

/// The `half` of an `f32` that a normal `half` (or zero) represents exactly.
fn half(bits: u32) -> u16 {
    if bits & 0x7FFF_FFFF == 0 {
        return (bits >> 16) as u16;
    }
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    assert!((1..31).contains(&exp) && bits & 0x1FFF == 0, "{bits:#x}");
    ((bits >> 16) & 0x8000) as u16 | ((exp as u16) << 10) | ((bits >> 13) & 0x3FF) as u16
}

/// What a kernel stores per thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Out {
    /// The four-lane texel.
    Vec4,
    /// The texel with lane 3 replaced by `2.0` (the kernel never reads the texel's alpha).
    Rgb2,
    /// The first lane (a depth read).
    First,
    /// The texel as `half4`.
    Half4,
    /// The texel with lane 0 replaced by `5.0` (its alpha carried on).
    Lane0,
    /// The texel's alpha alone.
    Alpha,
    /// The texel's lanes reversed.
    Reversed,
    /// A sparse read's texel with lane 3 replaced by `2.0` where it is resident: its residency
    /// is observed, and its alpha too, which reaches the kernel through memory (a sparse block
    /// read's texels go through the block's array).
    Resident,
}

impl Out {
    fn bytes(self, t: [u32; 4]) -> Vec<u8> {
        match self {
            Out::Vec4 => t.iter().flat_map(|w| w.to_le_bytes()).collect(),
            Out::Rgb2 | Out::Resident => [t[0], t[1], t[2], fl(2.0)]
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect(),
            Out::First => t[0].to_le_bytes().to_vec(),
            Out::Half4 => t.iter().flat_map(|w| half(*w).to_le_bytes()).collect(),
            Out::Lane0 => [fl(5.0), t[1], t[2], t[3]]
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect(),
            Out::Alpha => t[3].to_le_bytes().to_vec(),
            Out::Reversed => [t[3], t[2], t[1], t[0]]
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect(),
        }
    }

    fn len(self) -> usize {
        match self {
            Out::Vec4 | Out::Rgb2 | Out::Lane0 | Out::Reversed | Out::Resident => 16,
            Out::First | Out::Alpha => 4,
            Out::Half4 => 8,
        }
    }
}

/// Where an in-bounds `p[i]` reads: level, layer (6 · cube + face for a cube), sample, x/y/z.
type At = (u32, u32, u32, [u32; 3]);

/// A texture of `shape` (extent, layers, levels, samples) whose texels are distinct and nonzero,
/// doing `robust` outside itself.
fn image(texel: Texel, shape: ([u32; 3], u32, u32, u32), robust: Option<[u32; 4]>) -> Image {
    let (extent, layers, levels, samples) = shape;
    let slot = ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Texture,
        index: 0,
    };
    let mut img = Image::new(slot, extent, layers, levels, samples);
    for (i, t) in img.texels.iter_mut().enumerate() {
        *t = texel.fill(i);
    }
    img.robust = robust;
    img
}

/// A read kernel: its texture (extent, layers, levels, samples), whether it is a texel buffer,
/// and its rows, each `p[i]` with the texel it reads or `None` when Metal's read is out of
/// bounds.
struct ReadCase {
    entry: &'static str,
    texel: Texel,
    shape: ([u32; 3], u32, u32, u32),
    buffer: bool,
    out: Out,
    rows: Vec<([u32; 4], Option<At>)>,
}

impl ReadCase {
    /// Whether the kernel observes its texel's alpha.
    fn alpha(&self) -> bool {
        match self.out {
            Out::Vec4 | Out::Half4 | Out::Lane0 | Out::Alpha | Out::Reversed | Out::Resident => {
                true
            }
            Out::Rgb2 | Out::First => false,
        }
    }

    /// Run on every profile, with each out-of-bounds alpha, and check Metal's result; returns the
    /// classes it ran on.
    fn check(&self, bc: &[u8]) -> Vec<Class> {
        self.check_beyond(bc, Beyond::Nothing)
    }

    /// [`ReadCase::check`] of a kernel whose read does `beyond` what a plain read does.
    fn check_beyond(&self, bc: &[u8], beyond: Beyond) -> Vec<Class> {
        let mut ran = Vec::new();
        for (stem, p) in profiles() {
            if beyond == Beyond::LevelViews && !p.fields().descriptors.non_uniform_storage_image {
                let refused = typed_module(bc, self.entry, &p).err();
                assert!(
                    refused.as_ref().is_some_and(refuses_level_views),
                    "{stem}: {}: {refused:?}",
                    self.entry
                );
                continue;
            }
            let Some((plan, m)) = module(bc, self.entry, &stem, &p) else {
                continue;
            };
            let class = class(&p, self.buffer);
            let fact = RuntimeFact::OutOfBoundsAlpha { slot: TEXTURE };
            let selects = class != Class::Device || beyond != Beyond::Nothing;
            assert_eq!(
                plan.facts.contains(&fact),
                selects && self.alpha(),
                "{stem}: {}: {class:?}",
                self.entry
            );
            for alpha in [0, self.texel.one()] {
                let img = image(self.texel, self.shape, robust(class, alpha));
                let p_rows: Vec<[u32; 4]> = self.rows.iter().map(|(r, _)| *r).collect();
                let mut res = resources(
                    &plan,
                    &p_rows,
                    self.rows.len() * self.out.len(),
                    self.shape.2,
                );
                res.spec
                    .insert(SpecKey::OutOfBoundsAlpha(TEXTURE), u64::from(alpha != 0));
                let images = bound(&plan, &img);
                let (bufs, _) = run_both(&m, &res, &images, self.rows.len() as u32, &stem);
                let want: Vec<u8> = self
                    .rows
                    .iter()
                    .flat_map(|(_, at)| {
                        let t = match at {
                            Some((level, layer, sample, xyz)) => {
                                let i =
                                    img.index(*level, *layer, *sample, *xyz).expect("in bounds");
                                img.texels[i]
                            }
                            None => [0, 0, 0, alpha],
                        };
                        self.out.bytes(t)
                    })
                    .collect();
                for (i, (got, want)) in bufs[1]
                    .chunks_exact(self.out.len())
                    .zip(want.chunks_exact(self.out.len()))
                    .enumerate()
                {
                    assert_eq!(
                        got, want,
                        "{stem}: {} row {i} {:?}, alpha {alpha:#x}",
                        self.entry, self.rows[i].0
                    );
                }
            }
            ran.push(class);
        }
        ran.sort();
        ran.dedup();
        ran
    }
}

/// What a read kernel's read does beyond a plain read's ([`ReadCase::check_beyond`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Beyond {
    /// Nothing: the read is a plain one.
    Nothing,
    /// It observes its residency: the shader selects Metal's out-of-bounds texel on every class.
    Residency,
    /// It reads a storage texture at a level, through the slot's level views: the shader selects
    /// Metal's out-of-bounds texel on every class, and a profile that cannot index the views by
    /// a dynamic level refuses it ([`refuses_level_views`]).
    LevelViews,
}

/// Whether `e` is the refusal of level views indexed by a dynamic level, on a profile without
/// non-uniform storage image indexing.
fn refuses_level_views(e: &Error) -> bool {
    let want = Capability::Feature(VkFeature::ShaderStorageImageArrayNonUniformIndexing);
    let want = Unsupported::needs(Feature::Lowering(LowerFamily::Texture), want);
    matches!(e, Error::Unsupported(u) if (&u.feature, &u.needs) == (&want.feature, &want.needs))
}

/// Every class a read kernel must run on.
const ALL: [Class; 3] = [Class::Device, Class::ShaderReads, Class::Shader];

const MAX: u32 = u32::MAX;

/// A 2D read is out of bounds past the level's width or height, at a level past the last, and
/// at a negative (wrapped) coordinate; the texel's components and alpha follow the format class.
#[test]
fn texture_2d_reads_out_of_bounds_are_zero() {
    let Some(bc) = assemble() else {
        return;
    };
    let rows = || {
        vec![
            ([1, 2, 0, 0], Some((0, 0, 0, [1, 2, 0]))),
            ([3, 3, 0, 0], Some((0, 0, 0, [3, 3, 0]))),
            ([4, 1, 0, 0], None),
            ([1, 4, 0, 0], None),
            ([1, 1, 1, 0], Some((1, 0, 0, [1, 1, 0]))),
            ([2, 1, 1, 0], None),
            ([0, 0, 2, 0], None),
            ([MAX, 0, 0, 0], None),
        ]
    };
    for (entry, texel, out) in [
        ("oob_2d", Texel::Float, Out::Vec4),
        ("oob_2d_rgb", Texel::Float, Out::Rgb2),
        ("oob_2d_uint", Texel::Int, Out::Vec4),
    ] {
        let case = ReadCase {
            entry,
            texel,
            shape: ([4, 4, 1], 1, 2, 1),
            buffer: false,
            out,
            rows: rows(),
        };
        assert_eq!(case.check(&bc), ALL, "{entry}");
    }
}

/// A texel whose lane 0 is replaced (`v.x = 5`) carries the read's alpha on to the output, and
/// lane 3 extracted (`.w`) or shuffled (`.wzyx`) observes it: each stores Metal's out-of-bounds
/// alpha, so where the shader checks the read, the read takes the `OutOfBoundsAlpha`
/// specialization (the insertion, extraction, and shuffle rules of `lower::texture::alpha`).
#[test]
fn an_alpha_reaching_the_output_through_a_lane_is_specialized() {
    let Some(bc) = assemble_text("texture_oob_lanes", LANES) else {
        return;
    };
    let rows = || {
        vec![
            ([1, 2, 0, 0], Some((0, 0, 0, [1, 2, 0]))),
            ([4, 1, 0, 0], None),
            ([1, 1, 1, 0], Some((1, 0, 0, [1, 1, 0]))),
            ([0, 0, 2, 0], None),
            ([MAX, 0, 0, 0], None),
        ]
    };
    for (entry, out) in [
        ("oob_2d_x5", Out::Lane0),
        ("oob_2d_w", Out::Alpha),
        ("oob_2d_wzyx", Out::Reversed),
    ] {
        let case = ReadCase {
            entry,
            texel: Texel::Float,
            shape: ([4, 4, 1], 1, 2, 1),
            buffer: false,
            out,
            rows: rows(),
        };
        assert_eq!(case.check(&bc), ALL, "{entry}");
    }
}

/// A 2D-array read is out of bounds past the last slice; a 3D read past the level's depth.
#[test]
fn array_and_3d_reads_out_of_bounds_are_zero() {
    let Some(bc) = assemble() else {
        return;
    };
    let array = ReadCase {
        entry: "oob_2d_array",
        texel: Texel::Int,
        shape: ([4, 4, 1], 3, 2, 1),
        buffer: false,
        out: Out::Vec4,
        rows: vec![
            ([1, 1, 2, 0], Some((0, 2, 0, [1, 1, 0]))),
            ([1, 1, 3, 0], None),
            ([4, 0, 0, 0], None),
            ([1, 1, 1, 1], Some((1, 1, 0, [1, 1, 0]))),
            ([0, 0, MAX, 0], None),
            ([0, 0, 0, 2], None),
        ],
    };
    assert_eq!(array.check(&bc), ALL);
    let volume = ReadCase {
        entry: "oob_3d",
        texel: Texel::Float,
        shape: ([4, 4, 2], 1, 2, 1),
        buffer: false,
        out: Out::Vec4,
        rows: vec![
            ([1, 2, 1, 0], Some((0, 0, 0, [1, 2, 1]))),
            ([1, 2, 2, 0], None),
            ([1, 1, 0, 1], Some((1, 0, 0, [1, 1, 0]))),
            ([0, 0, 1, 1], None),
            ([0, 0, 0, 2], None),
        ],
    };
    assert_eq!(volume.check(&bc), ALL);
}

/// A cube read is out of bounds at face 6 and past it; a cube-array read at face 6 of a cube
/// (never the next cube's face 0), past the last cube, and at a cube whose `6 · cube + face`
/// wraps to a layer in range.
#[test]
fn cube_reads_out_of_bounds_are_zero() {
    let Some(bc) = assemble() else {
        return;
    };
    let cube = ReadCase {
        entry: "oob_cube",
        texel: Texel::Float,
        shape: ([4, 4, 1], 6, 2, 1),
        buffer: false,
        out: Out::Vec4,
        rows: vec![
            ([1, 1, 5, 0], Some((0, 5, 0, [1, 1, 0]))),
            ([1, 1, 6, 0], None),
            ([0, 0, MAX, 0], None),
            ([1, 1, 2, 1], Some((1, 2, 0, [1, 1, 0]))),
            ([2, 0, 0, 1], None),
            ([0, 0, 0, 2], None),
        ],
    };
    assert_eq!(cube.check(&bc), ALL);
    // 6 · 0x2AAA_AAAA + 5 wraps to 1.
    let array = ReadCase {
        entry: "oob_cube_array",
        texel: Texel::Float,
        shape: ([4, 4, 1], 12, 1, 1),
        buffer: false,
        out: Out::Vec4,
        rows: vec![
            ([1, 1, 4, 1], Some((0, 10, 0, [1, 1, 0]))),
            ([1, 1, 6, 0], None),
            ([1, 1, 0, 2], None),
            ([1, 1, 5, 0x2AAA_AAAA], None),
            ([0, 0, MAX, 0], None),
        ],
    };
    assert_eq!(array.check(&bc), ALL);
}

/// A multisampled read is out of bounds past the last sample; a read_write texture's read past
/// its extent; a depth read's depth is 0; a `half` read converts the zero texel.
#[test]
fn sample_storage_depth_and_half_reads_out_of_bounds_are_zero() {
    let Some(bc) = assemble() else {
        return;
    };
    let cases = [
        ReadCase {
            entry: "oob_ms",
            texel: Texel::Float,
            shape: ([4, 4, 1], 1, 1, 4),
            buffer: false,
            out: Out::Vec4,
            rows: vec![
                ([1, 1, 3, 0], Some((0, 0, 3, [1, 1, 0]))),
                ([1, 1, 4, 0], None),
                ([4, 0, 0, 0], None),
            ],
        },
        ReadCase {
            entry: "oob_rw",
            texel: Texel::Int,
            shape: ([4, 4, 1], 1, 1, 1),
            buffer: false,
            out: Out::Vec4,
            rows: vec![
                ([3, 3, 0, 0], Some((0, 0, 0, [3, 3, 0]))),
                ([4, 0, 0, 0], None),
                ([0, 4, 0, 0], None),
            ],
        },
        ReadCase {
            entry: "oob_depth",
            texel: Texel::Float,
            shape: ([4, 4, 1], 1, 2, 1),
            buffer: false,
            out: Out::First,
            rows: vec![
                ([1, 1, 0, 0], Some((0, 0, 0, [1, 1, 0]))),
                ([4, 0, 0, 0], None),
                ([1, 1, 1, 0], Some((1, 0, 0, [1, 1, 0]))),
                ([0, 0, 2, 0], None),
            ],
        },
        ReadCase {
            entry: "oob_1d_half",
            texel: Texel::Float,
            shape: ([4, 1, 1], 1, 1, 1),
            buffer: false,
            out: Out::Half4,
            rows: vec![
                ([3, 0, 0, 0], Some((0, 0, 0, [3, 0, 0]))),
                ([4, 0, 0, 0], None),
            ],
        },
    ];
    for case in cases {
        let ran = case.check(&bc);
        // A read_write texture needs a formatless load, which no profile without image
        // robustness here has.
        assert!(
            ran.contains(&Class::Device) && ran.contains(&Class::Shader),
            "{}: {ran:?}",
            case.entry
        );
    }
}

/// A texel buffer read is out of bounds past its last texel, also where `robustBufferAccess`
/// (whose out-of-bounds texel may be any value of the buffer) is the only robustness.
#[test]
fn texel_buffer_reads_out_of_bounds_are_zero() {
    let Some(bc) = assemble() else {
        return;
    };
    let case = ReadCase {
        entry: "oob_buffer",
        texel: Texel::Float,
        shape: ([5, 1, 1], 1, 1, 1),
        buffer: true,
        out: Out::Vec4,
        rows: vec![
            ([4, 0, 0, 0], Some((0, 0, 0, [4, 0, 0]))),
            ([5, 0, 0, 0], None),
            ([MAX, 0, 0, 0], None),
        ],
    };
    let ran = case.check(&bc);
    assert_eq!(ran, [Class::Device, Class::Shader]);
    // `robustBufferAccess` alone is the shader's check.
    let vk12 = profiles()
        .into_iter()
        .find(|(s, _)| s == "vk12-floor")
        .expect("vk12-floor");
    assert_eq!(class(&vk12.1, true), Class::Shader);
}

/// Run a write kernel over `rows` on every profile: the texture's texels other than `written`
/// keep their values, and `written` holds `(1, 2, 3, 4)`.
fn check_writes(
    bc: &[u8],
    entry: &str,
    shape: ([u32; 3], u32),
    buffer: bool,
    rows: &[[u32; 4]],
    written: At,
) -> Vec<Class> {
    let mut ran = Vec::new();
    for (stem, p) in profiles() {
        let Some((plan, m)) = module(bc, entry, &stem, &p) else {
            continue;
        };
        let class = class(&p, buffer);
        let img = image(Texel::Float, (shape.0, shape.1, 1, 1), robust(class, 0));
        let res = resources(&plan, rows, 0, 1);
        let images = [img.clone()];
        let (_, after) = run_both(&m, &res, &images, rows.len() as u32, &stem);
        let (level, layer, sample, xyz) = written;
        let w = img.index(level, layer, sample, xyz).expect("in bounds");
        for (i, (got, was)) in after.texels.iter().zip(&img.texels).enumerate() {
            let want = if i == w {
                [fl(1.0), fl(2.0), fl(3.0), fl(4.0)]
            } else {
                *was
            };
            assert_eq!(*got, want, "{stem}: {entry} texel {i}");
        }
        ran.push(class);
    }
    ran.sort();
    ran.dedup();
    ran
}

/// A cube-array write at face 6, at a cube past the last, or at a cube whose layer wraps, writes
/// nothing (never another cube's face).
#[test]
fn cube_array_writes_out_of_bounds_are_dropped() {
    let Some(bc) = assemble() else {
        return;
    };
    // 6 · 0x2AAA_AAAA + 4 wraps to 0.
    let rows = [
        [1, 1, 1, 0],
        [2, 2, 6, 0],
        [0, 0, 0, 2],
        [3, 3, 4, 0x2AAA_AAAA],
        [0, 0, MAX, 1],
    ];
    let ran = check_writes(
        &bc,
        "oob_cube_array_write",
        ([4, 4, 1], 12),
        false,
        &rows,
        (0, 1, 0, [1, 1, 0]),
    );
    assert!(ran.contains(&Class::Device), "{ran:?}");
}

/// A texel buffer write past its last texel writes nothing, also where `robustBufferAccess`
/// (which may write the value anywhere in the buffer) is the only robustness.
#[test]
fn texel_buffer_writes_out_of_bounds_are_dropped() {
    let Some(bc) = assemble() else {
        return;
    };
    let rows = [[4, 0, 0, 0], [5, 0, 0, 0], [MAX, 0, 0, 0]];
    let ran = check_writes(
        &bc,
        "oob_buffer_write",
        ([5, 1, 1], 1),
        true,
        &rows,
        (0, 0, 0, [4, 0, 0]),
    );
    assert_eq!(ran, [Class::Device, Class::Shader]);
}

/// A thread that writes a texel out of bounds and then reads that location reads Metal's zero
/// texel. Vulkan's robustness discards the write, but lets the read return "the value of the last
/// store to the same out-of-bounds location in the same shader invocation" unless the load is
/// volatile; the interpreter's robust image takes that permission. Where the device is left both
/// the store and the load (`Class::Device`), the slot's variable is `Volatile` and the read returns
/// zero; where the shader checks the read, it reads a clamped coordinate inside the image and
/// selects Metal's texel. In-bounds rows read back their own write (the interpreter's program
/// order; Metal orders it only after a `fence`, which these kernels omit, so only the out-of-bounds
/// rows state Metal's value).
#[test]
fn a_read_where_its_thread_wrote_out_of_bounds_is_zero() {
    let Some(bc) = assemble_text("texture_forward", FORWARD) else {
        return;
    };
    // (entry, texel buffer, extent, rows: x, y, the value written, and whether it is in bounds)
    type Rows = Vec<([u32; 4], bool)>;
    let kernels: [(&str, bool, [u32; 3], Rows); 2] = [
        (
            "fwd_rw",
            false,
            [2, 2, 1],
            vec![
                ([0, 0, 0x11, 0], true),
                ([1, 1, 0x22, 0], true),
                ([2, 0, 0x33, 0], false),
                ([0, 5, 0x44, 0], false),
                ([MAX, 1, 0x55, 0], false),
            ],
        ),
        (
            "fwd_buffer",
            true,
            [4, 1, 1],
            vec![
                ([1, 0, 0x11, 0], true),
                ([3, 0, 0x22, 0], true),
                ([4, 0, 0x33, 0], false),
                ([100, 0, 0x44, 0], false),
                ([MAX, 0, 0x55, 0], false),
            ],
        ),
    ];
    for (entry, buffer, extent, rows) in kernels {
        let mut ran = Vec::new();
        for (stem, p) in profiles() {
            let Some((plan, m)) = module(&bc, entry, &stem, &p) else {
                continue;
            };
            let class = class(&p, buffer);
            assert_eq!(
                plan.families.texture.slots[&0].volatile,
                class == Class::Device,
                "{stem}: {entry}: {class:?}"
            );
            for alpha in [0, 1] {
                let img = image(Texel::Int, (extent, 1, 1, 1), robust(class, alpha));
                let p_rows: Vec<[u32; 4]> = rows.iter().map(|(r, _)| *r).collect();
                let mut res = resources(&plan, &p_rows, rows.len() * 16, 1);
                res.spec
                    .insert(SpecKey::OutOfBoundsAlpha(TEXTURE), u64::from(alpha != 0));
                let images = [img.clone()];
                let (bufs, _) = run_both(&m, &res, &images, rows.len() as u32, &stem);
                for (i, ((r, inside), got)) in rows
                    .iter()
                    .zip(bufs[1].as_chunks::<16>().0.iter())
                    .enumerate()
                {
                    let want = if *inside { [r[2]; 4] } else { [0, 0, 0, alpha] };
                    assert_eq!(
                        &got[..],
                        Out::Vec4.bytes(want),
                        "{stem}: {entry} row {i} {r:?}, alpha {alpha}"
                    );
                }
            }
            ran.push(class);
        }
        ran.sort();
        ran.dedup();
        let want: &[Class] = if buffer {
            &[Class::Device, Class::Shader]
        } else {
            &ALL
        };
        assert_eq!(ran, want, "{entry}");
    }
}

/// Only a slot the device is left both an out-of-bounds store and an out-of-bounds load of is
/// `Volatile`: never a slot the function only writes or only reads, and never without containment,
/// where nothing out of bounds is defined. A texture atomic alone is both where image robustness
/// discards it unguarded : the atomic-only `fwd_atomic`, declared `R32Uint` as atomics need, is
/// `Volatile` on every profile with `robustImageAccess` (`roadmap-2022`, `swiftshader`) and on none
/// without (`vk12-floor`), where the shader guards it.
#[test]
fn only_a_slot_left_both_a_store_and_a_load_is_volatile() {
    let Some(bc) = assemble_text("texture_forward", FORWARD) else {
        return;
    };
    let unsafe_off = DevicePolicy {
        containment: ContainmentPolicy::UnsafeOff,
        isolation: GuestIsolation::Off,
        ..Default::default()
    };
    let mut all = profiles();
    for stem in ["moltenvk-m3", "lavapipe"] {
        all.push((
            format!("{stem} unsafe-off"),
            profile(stem, unsafe_off).expect("profile"),
        ));
    }
    let mut atomic_inputs = LowerInputs::new();
    atomic_inputs.storage_images.insert(
        TEXTURE,
        StorageImageChoice::new(vk::Format::R32_UINT, StorageImageClass::Identity),
    );
    let mut volatile = 0;
    let mut atomic: BTreeMap<String, bool> = BTreeMap::new();
    for (stem, p) in all {
        let off = !matches!(p.fields().containment, Containment::On { .. });
        for (entry, buffer) in [
            ("fwd_rw", false),
            ("fwd_buffer", true),
            ("fwd_write_only", false),
            ("fwd_read_only_rw", false),
            ("fwd_atomic", false),
        ] {
            let plan = if entry == "fwd_atomic" {
                match typed_module_with(&bc, entry, &p, &atomic_inputs) {
                    Ok((plan, _)) => plan,
                    // The profile has no `R32Uint` storage-image atomics.
                    Err(Error::Unsupported(u)) if u.feature == Feature::Format => continue,
                    Err(e) => panic!("{stem}: {entry}: {e}"),
                }
            } else {
                let Some((plan, _)) = module(&bc, entry, &stem, &p) else {
                    continue;
                };
                plan
            };
            let want = !off
                && match entry {
                    "fwd_rw" | "fwd_buffer" => class(&p, buffer) == Class::Device,
                    "fwd_atomic" => p.fields().images.robust_image_access,
                    _ => false,
                };
            let got = plan.families.texture.slots[&0].volatile;
            assert_eq!(got, want, "{stem}: {entry}");
            volatile += usize::from(want);
            if entry == "fwd_atomic" {
                atomic.insert(stem.clone(), got);
            }
        }
    }
    assert!(volatile > 0);
    for (stem, want) in [
        ("roadmap-2022", true),
        ("swiftshader", true),
        ("vk12-floor", false),
    ] {
        assert_eq!(atomic.get(stem), Some(&want), "fwd_atomic on {stem}");
    }
}

/// The specialization is declared only where the shader checks a read whose alpha it observes:
/// `float4(t.read(c).xyz, 2)` and a depth read need none.
#[test]
fn only_an_observed_alpha_is_specialized() {
    let Some(bc) = assemble() else {
        return;
    };
    let p = without(
        "moltenvk-m3",
        &[VkFeature::RobustImageAccess2, VkFeature::RobustImageAccess],
    );
    for (entry, observed) in [
        ("oob_2d", true),
        ("oob_2d_rgb", false),
        ("oob_depth", false),
        ("oob_1d_half", true),
    ] {
        let (plan, _) = typed_module(&bc, entry, &p).expect(entry);
        let facts: Vec<&RuntimeFact> = plan
            .facts
            .iter()
            .filter(|f| matches!(f, RuntimeFact::OutOfBoundsAlpha { .. }))
            .collect();
        let want: Vec<RuntimeFact> = if observed {
            vec![RuntimeFact::OutOfBoundsAlpha { slot: TEXTURE }]
        } else {
            Vec::new()
        };
        assert_eq!(facts, want.iter().collect::<Vec<_>>(), "{entry}");
    }
}

/// An argument-buffer texture's format is the resident texture's, which the binder states per
/// handle-table entry, its out-of-bounds-alpha flag: a heap read the shader checks whose alpha it
/// observes reads Metal's out-of-bounds alpha from the entry its handle names. End to end on the
/// interpreter, the binder's table bytes for two resident textures (an `R32Float`, whose missing
/// alpha Metal reads as 1, and an `RGBA32Float`), the emitted search, and reads in and past the
/// texture; a missed id reads the null texture's zero. Where the device's robustness is Metal's
/// texel the search stays the element's alone.
#[test]
fn a_heap_read_takes_its_out_of_bounds_alpha_from_its_table_entry() {
    let bc = match llvm_assemble(&engine_dir().join("handles.ll")) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected handles.ll: {why}"),
    };
    const NO_ALPHA: u64 = 0x0000_0001_0000_0010;
    const ALPHA: u64 = 0x0000_0001_0000_0020;
    const MISSED: u64 = 0x0000_0002_0000_0000;
    let entry = "argument_buffer_read";
    let unrobust = [
        VkFeature::RobustImageAccess,
        VkFeature::RobustImageAccess2,
        VkFeature::RobustBufferAccess,
        VkFeature::RobustBufferAccess2,
    ];
    let devices = [
        ("moltenvk-m3", device("moltenvk-m3", &[])),
        (
            "moltenvk-m3 robust1",
            device("moltenvk-m3", &[VkFeature::RobustImageAccess2]),
        ),
        ("moltenvk-m3 unrobust", device("moltenvk-m3", &unrobust)),
        ("roadmap-2022", device("roadmap-2022", &[])),
    ];
    let mut ran = Vec::new();
    for (stem, dev) in devices {
        let lib = Library::from_air(&bc).expect("library");
        let f = lib.function(entry).expect("entry");
        let desc =
            ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f)));
        let pipe = match dev.compile_compute_pipeline(&desc, &Budget::standard()) {
            Ok(p) => p,
            Err(Error::Unsupported(u)) if u.feature == Feature::ArgumentBufferTextures => continue,
            Err(e) => panic!("{stem}: {e}"),
        };
        // Members: token 1 (element 1), then token 2 (element 2).
        let used = |token, pf, id| {
            let mut u = UsedResource::new(
                ResourceClass::Texture(TextureFacts::new(TextureType::Type2D, pf)),
                token,
            );
            u.resource_id = Some(id);
            u
        };
        let state = BindState {
            used_resources: vec![
                used(1, PixelFormat::R32Float, NO_ALPHA),
                used(2, PixelFormat::RGBA32Float, ALPHA),
            ],
            ..Default::default()
        };
        let bplan = pipe.plan(&dev, &state.shape()).expect("plan");
        let dispatch = Dispatch::Threadgroups {
            groups: [4, 1, 1],
            threads_per_group: [1, 1, 1],
        };
        let cmds = bplan
            .instantiate_dispatch(&state, &dispatch)
            .expect("dispatch");
        let (set, binding) = bplan
            .descriptors
            .iter()
            .find_map(|d| {
                matches!(d.source, WriteSource::HandleTable { .. }).then_some((d.set, d.binding))
            })
            .expect("a handle table");
        let table = cmds
            .descriptor_bytes
            .iter()
            .find(|b| (b.set, b.binding) == (set, binding))
            .expect("table bytes")
            .bytes
            .clone();

        let (plan, m) = typed_module(&bc, entry, dev.profile()).expect("module");
        let class = class(dev.profile(), false);
        assert_eq!(
            plan.families.access.heap_alpha,
            class != Class::Device,
            "{stem}: the search carries the flag only where the shader checks the read"
        );
        let heap = plan
            .families
            .texture
            .heap
            .values()
            .find_map(|h| h.class)
            .expect("a heap class");
        let slot = ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Heap(HandleClass::Image(heap)),
            index: 0,
        };
        // Element 0 is the class's neutral; 1 views the `R32Float` texture (lanes 1 and 2 read 0,
        // lane 3 reads 1 through its view), 2 the `RGBA32Float` one. Each is 2 x 1.
        let element = |e: u32, extent: [u32; 3], texel: &dyn Fn(u32) -> [u32; 4], alpha: u32| {
            let mut img = Image::new(slot, extent, 1, 1, 1).at_element(e);
            for (x, t) in img.texels.iter_mut().enumerate() {
                *t = texel(x as u32);
            }
            img.robust = robust(class, alpha);
            img
        };
        let images = vec![
            element(0, [1, 1, 1], &|_| [0; 4], 0),
            element(
                1,
                [2, 1, 1],
                &|x| [fl(3.0 + x as f32), 0, 0, fl(1.0)],
                fl(1.0),
            ),
            element(
                2,
                [2, 1, 1],
                &|x| [fl(5.0 + x as f32), fl(6.0), fl(7.0), fl(8.0)],
                0,
            ),
        ];
        for (id, inside, outside) in [
            (
                NO_ALPHA,
                [[fl(3.0), 0, 0, fl(1.0)], [fl(4.0), 0, 0, fl(1.0)]],
                [0, 0, 0, fl(1.0)],
            ),
            (
                ALPHA,
                [
                    [fl(5.0), fl(6.0), fl(7.0), fl(8.0)],
                    [fl(6.0), fl(6.0), fl(7.0), fl(8.0)],
                ],
                [0; 4],
            ),
            (MISSED, [[0; 4], [0; 4]], [0; 4]),
        ] {
            let mut spec = BTreeMap::new();
            for axis in [Axis::X, Axis::Y, Axis::Z] {
                spec.insert(SpecKey::LocalSize(axis), 1);
            }
            spec.insert(SpecKey::FlatLocalSize, 1);
            let buffer = |kind, bytes| Buffer {
                slot: Some(ResourceSlot {
                    stage: Stage::Kernel,
                    kind,
                    index: 0,
                }),
                address: None,
                bytes,
            };
            let mut out = buffer(ResourceKind::Buffer, vec![0xAB; 4 * 16]);
            if let Some(s) = out.slot.as_mut() {
                s.index = 1;
            }
            let res = Resources {
                buffers: vec![
                    buffer(ResourceKind::Buffer, id.to_le_bytes().to_vec()),
                    out,
                    buffer(
                        ResourceKind::Synthetic(SyntheticBinding::HandleTable),
                        table.clone(),
                    ),
                ],
                push: vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)],
                spec,
            };
            for order in [Order::Forward, Order::Reverse] {
                let cfg = Config {
                    workgroups: [4, 1, 1],
                    order,
                    ..Config::default()
                };
                let (o, _) = run_with_images(&m, &cfg, res.clone(), images.clone())
                    .unwrap_or_else(|e| panic!("{stem}: {id:#x}: {e:?}"));
                // Threads 0 and 1 read in bounds, 2 and 3 past the texture's width.
                let want: Vec<u8> = [inside[0], inside[1], outside, outside]
                    .iter()
                    .flat_map(|t| Out::Vec4.bytes(*t))
                    .collect();
                assert_eq!(o.buffers[1], want, "{stem}: id {id:#x}, {order:?}");
            }
        }
        ran.push(class);
    }
    ran.sort();
    ran.dedup();
    assert_eq!(ran, ALL);
}

/// [`EDGE_READS`] with `edge_2d`'s read edited by hand to take 16-bit coordinates and offset
/// `(-1, -1)`. Metal 4.1 has no such overload: its `read(sampler, ...)` takes `int` coordinates,
/// and `metal -std=metal4.1` rejects a `short2` one.
fn edge_reads_short() -> String {
    const READ: &str = "  %10 = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(\
        ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) @__air_sampler_state, \
        <2 x i32> %8, <2 x i32> zeroinitializer, i32 %9, i32 1) #3\n";
    const SHORT: &str = "  %s8 = trunc <2 x i32> %8 to <2 x i16>\n  %s9 = trunc i32 %9 to i16\n  \
        %10 = tail call { <4 x float>, i8 } @air.read_texture_2d.i16.v4f32(\
        ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) @__air_sampler_state, \
        <2 x i16> %s8, <2 x i16> <i16 -1, i16 -1>, i16 %s9, i32 1) #3\n";
    const DECLARE: &str = "declare { <4 x float>, i8 } @air.read_texture_2d.i16.v4f32(\
        ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i16>, <2 x i16>, i16, \
        i32) local_unnamed_addr #2\n\n";
    assert_eq!(EDGE_READS.matches(READ).count(), 1, "edge_2d's read");
    let text = EDGE_READS.replacen(READ, SHORT, 1);
    let at = text.find("attributes #0").expect("attributes");
    let (head, tail) = text.split_at(at);
    format!("{head}{DECLARE}{tail}")
}

/// Metal 4.1's `read(sampler s, coord)` addresses through `s` (MSL 4.1 §6.13), whatever its state:
/// a read through a runtime sampler, a repeating one, or a filtered one translates on every
/// profile. A cube's read through a sampler, which the M3 reads unclamped (zero past the face at
/// the read's level, the base level's texel within it) and which is not built, is refused on every
/// profile, never read as out of bounds, and so is one at 16-bit coordinates, which no Metal 4.1
/// overload takes.
#[test]
fn a_cube_s_or_a_narrow_read_through_a_sampler_is_refused() {
    let Some(sampled) = assemble_text("sampled_reads", SAMPLED_READS) else {
        return;
    };
    let Some(edge) = assemble_text("edge_reads", EDGE_READS) else {
        return;
    };
    let Some(short) = assemble_text("edge_reads_short", &edge_reads_short()) else {
        return;
    };
    for (stem, p) in profiles() {
        for (bc, entry) in [(&edge, "edge_cube"), (&short, "edge_2d")] {
            match typed_module(bc, entry, &p) {
                Err(Error::Unsupported(u)) => assert_eq!(
                    u.feature,
                    Feature::Lowering(LowerFamily::Texture),
                    "{stem}: {entry}"
                ),
                Ok(_) => panic!("{stem}: {entry} translated"),
                Err(e) => panic!("{stem}: {entry}: {e}"),
            }
        }
        for (bc, entry) in [
            (&sampled, "read_arg_sampler"),
            (&sampled, "read_clamped"),
            (&edge, "read_repeat"),
            (&edge, "read_linear"),
        ] {
            typed_module(bc, entry, &p).unwrap_or_else(|e| panic!("{stem}: {entry}: {e}"));
        }
    }
}

/// Metal 4.1's `sparse_read(sampler s, ...)` through a clamp-to-edge sampler is an edge read
/// whose status word the function observes. What clamp-to-edge addressing does to the residency
/// of a sparse texture's texels the specification does not say, and the M3 measured it only over
/// a texture that is not sparse, so it is refused on every profile, never given the slot
/// residency of a plain read. The same read with its status word unused is a clamp-to-edge read,
/// and translates.
#[test]
fn a_sparse_read_through_a_clamp_to_edge_sampler_is_refused() {
    let Some(bc) = assemble_text("sparse_edge_reads", SPARSE_EDGE_READS) else {
        return;
    };
    let want = Unsupported::not_implemented(Feature::Lowering(LowerFamily::Texture));
    for (stem, p) in profiles() {
        match typed_module(&bc, "sparse_edge", &p) {
            Err(Error::Unsupported(u)) => assert_eq!(
                (u.feature, u.needs),
                (want.feature.clone(), want.needs.clone()),
                "{stem}"
            ),
            Ok(_) => panic!("{stem}: sparse_edge translated"),
            Err(e) => panic!("{stem}: sparse_edge: {e}"),
        }
        typed_module(&bc, "edge_plain", &p).unwrap_or_else(|e| panic!("{stem}: edge_plain: {e}"));
    }
}

/// `x` as the bits of an `int` lane.
fn int(x: i32) -> u32 {
    x as u32
}

/// The owned fixture `engine/kernel_block_read`'s module (Metal 4.1's block reads), assembled.
fn block_reads() -> Option<Vec<u8>> {
    let path = engine_dir().join("kernel_block_read.ll");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    assemble_text("kernel_block_read", &text)
}

/// How a block reads: its offset, where its second texel is, and whether it goes through the
/// clamp-to-edge sampler.
struct Block {
    offset: [i32; 2],
    second: [i32; 2],
    edge: bool,
}

impl Block {
    /// The rows of a block kernel over a texture of `extent` (a 2D level 0), `levels` levels and
    /// `layers` layers: each point, layer, and level of `points`, each texel of the block, with
    /// the texel Metal reads (measured on the M3 Max, `canon::block_read`). Through the read
    /// sampler texel `k` lies at `p + offset + d_k` of level `clamp(lod, 0, levels - 1)` (a
    /// signed `int`), zero outside it; through the clamp-to-edge sampler at that coordinate
    /// clamped into level 0, which it reads at every level.
    fn rows(
        &self,
        extent: [u32; 2],
        (levels, layers): (u32, u32),
        points: &[[i32; 3]],
    ) -> Vec<([u32; 4], Option<At>)> {
        let mut rows = Vec::new();
        for layer in 0..layers {
            for &[x, y, lod] in points {
                for k in 0..2 {
                    let d = if k == 1 { self.second } else { [0, 0] };
                    let at =
                        [0, 1].map(|i| [x, y][i].wrapping_add(self.offset[i]).wrapping_add(d[i]));
                    let top = i32::try_from(levels).expect("levels") - 1;
                    let level = if self.edge { 0 } else { lod.clamp(0, top) };
                    let size = extent.map(|e| i64::from(e >> level).max(1));
                    let texel = if self.edge {
                        let c = [0, 1].map(|i| i64::from(at[i]).clamp(0, size[i] - 1) as u32);
                        Some((level as u32, layer, 0, [c[0], c[1], 0]))
                    } else {
                        let inside = (0..2).all(|i| (0..size[i]).contains(&i64::from(at[i])));
                        inside.then(|| (level as u32, layer, 0, [at[0] as u32, at[1] as u32, 0]))
                    };
                    let w = k + 2 * layer;
                    rows.push(([int(x), int(y), int(lod), w], texel));
                }
            }
        }
        rows
    }
}

/// Metal 4.1's `block_read<W, H>` and `sparse_block_read<W, H>` (the owned fixture
/// `engine/kernel_block_read`) read the block's texels as reads do, as measured on the M3 Max
/// (macOS 27.0.1, Apple metal 32023.922): through the read sampler at `coord + offset` and the
/// next texel right or below, at the level clamped into the texture's (a level of 2 or 5 over two
/// levels reads level 1, of `-1` level 0), zero outside it; through a constexpr clamp-to-edge
/// sampler at that coordinate clamped into the base level, which it reads at every level. On 2D
/// textures of `float`, `uint`, `int`, and `half`, a 2D array, a depth texture, a `read_write`
/// one, and a sparse block read (each texel resident: the texture is not sparse), on every
/// profile, in both scheduling orders.
#[test]
fn block_reads_read_their_texels_as_reads_do() {
    let Some(bc) = block_reads() else {
        return;
    };
    let points = [
        [1, 1, 0],
        [6, 1, 0],
        [7, 1, 0],
        [8, 1, 0],
        [-1, 1, 0],
        [2, 3, 0],
        [-3, -2, 0],
        [3, 1, 1],
        [1, 0, 1],
        [9, 5, 1],
        [1, 0, 2],
        [3, 1, 5],
        [1, 1, -1],
        [i32::MIN, 0, 0],
    ];
    let (right, below) = ([1, 0], [0, 1]);
    let read = |offset, second| Block {
        offset,
        second,
        edge: false,
    };
    let edge = |offset, second| Block {
        offset,
        second,
        edge: true,
    };
    let plane = ([8, 4, 1], 1, 2, 1);
    // Each kernel, its texel, texture, output, block, and what its reads do beyond a plain read's.
    let cases = [
        (
            "br_2x1",
            Texel::Float,
            plane,
            Out::Vec4,
            read([0, 0], right),
            Beyond::Nothing,
        ),
        (
            "br_1x2_offset",
            Texel::Int,
            plane,
            Out::Vec4,
            read([-1, 2], below),
            Beyond::Nothing,
        ),
        (
            "br_edge",
            Texel::Float,
            plane,
            Out::Vec4,
            edge([1, 0], right),
            Beyond::Nothing,
        ),
        (
            "br_edge_1x2",
            Texel::Int,
            plane,
            Out::Vec4,
            edge([0, 0], below),
            Beyond::Nothing,
        ),
        (
            "br_half",
            Texel::Float,
            plane,
            Out::Half4,
            read([0, 0], right),
            Beyond::Nothing,
        ),
        (
            "br_array",
            Texel::Float,
            ([8, 4, 1], 3, 2, 1),
            Out::Vec4,
            read([0, 0], below),
            Beyond::Nothing,
        ),
        (
            "br_depth",
            Texel::Float,
            plane,
            Out::First,
            read([0, 0], right),
            Beyond::Nothing,
        ),
        (
            "br_sparse",
            Texel::Float,
            plane,
            Out::Resident,
            read([0, 0], right),
            Beyond::Residency,
        ),
        (
            "br_rw",
            Texel::Float,
            plane,
            Out::Vec4,
            read([0, 0], right),
            Beyond::LevelViews,
        ),
    ];
    for (entry, texel, shape, out, block, beyond) in &cases {
        let ([w, h, _], layers, levels, _) = *shape;
        let case = ReadCase {
            entry,
            texel: *texel,
            shape: *shape,
            buffer: false,
            out: *out,
            rows: block.rows([w, h], (levels, layers), &points),
        };
        let ran = case.check_beyond(&bc, *beyond);
        assert!(ran.contains(&Class::Shader), "{entry}: {ran:?}");
    }
    // Every kernel links and passes `spirv-val` where it translates, its block reads made reads.
    let mut validated = 0;
    for (stem, p) in profiles() {
        for (entry, .., beyond) in &cases {
            match compile(&bc, entry, &p) {
                Ok(a) => {
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: {entry}: {e}"));
                    validated += 1;
                }
                Err(Error::NeedsInput(_)) => {}
                Err(e) if *beyond == Beyond::LevelViews && refuses_level_views(&e) => {}
                Err(e) => panic!("{stem}: {entry}: {e}"),
            }
        }
    }
    assert!(validated > 0);
}
