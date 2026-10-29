//! Device tests of `YCBCR8_420_2P` textures over a CoreVideo `420v` or `420f` surface's two
//! planes (`runtime/src/ycbcr.rs`), on the device the cfg selects, under the validation layer. Each
//! surface is a Shared buffer standing for the IOSurface, laid out as CoreVideo lays a 40x24 one
//! out: the luma plane at byte 0, rows 64 bytes apart, and the interleaved chroma plane after it,
//! rows 64 bytes apart.
//!
//! - A kernel's `read()` of every texel is the conversion Metal's reads were measured to follow
//!   (`metal2vulkan::bind::ycbcr_texel`, after the 9/3/3/1 chroma filter), for both ranges and
//!   both matrices, and for one surface Metal's own reads of it texel by texel
//!   ([`METAL_READS`]). A nearest sample at each texel's centre reads the same texel,
//!   unannounced; a linear one there too, and between the texels it is Metal's filter of them,
//!   unannounced (the lane's devices sample as Metal does). Under native sampling it is the
//!   device's filter of the same texels (on MoltenVK Metal's own sampler's, announced), within
//!   [`DEVICE_FILTER_BOUND`] of Metal's, through the pipeline the state's creation made. A function that samples handles and reaches the texture is refused, typed.
//! - The texture follows writes of its planes' bytes through other textures over them, as a
//!   decoder writes the planes as `R8Unorm` and `RG8Unorm` textures: in an earlier command
//!   buffer, an earlier encoder of the same one, an earlier dispatch of the same encoder, and a
//!   blit; and the CPU's writes between command buffers.
//! - A kernel's writes to it are dropped, as Metal drops them; `get_bytes` reads the luma rows.
//!   An unpacked 4:4:4 format's writes are stored into both planes as Metal stores them, every
//!   other byte kept, and a packed one's dropped; those of the `_sRGB` twins, whose writes are not
//!   known, and over plane 1 are refused, typed.
//! - Plain creation and a buffer's texture of the format answer Metal's nil (`Error::Nil`,
//!   `NoTextureOfFormat`) and the heap query 0 bytes at 16; a surface without its facts is
//!   refused, typed. A texture over any other surface reads what Metal reads of it (other
//!   OSTypes, planes, codes wider than the surface's, zeros past the buffer, P3 matrices). Every
//!   two-plane YCbCr format answers as Metal does without a surface, and those the runtime does
//!   not hold are refused over one.

use super::argument_linear_samples::{self, LINEAR_EDGE};
use super::blits::SYSTEM_MEMORY;
use super::degrade_notices::{assert_once, counts, on_device};
use super::depth_compares::{pipeline_of, run_held, Held, Through};
use super::driver_deviations::stage_notes;
use super::host_memory::{self, host_page, Mapping};
use super::pixel_samplers::{metal_sampler, own_sampler};
use super::ycbcr_device_reads::{assert_device_reads, device_conversions, vulkan_read};
use super::{buffer_with, converted_holds, half_within, read_buffer};
use crate::air_library::AirLibrary;
use metal2vulkan::bind::{
    ycbcr_read, ycbcr_texel, ycbcr_write, SamplerDescriptor, YCbCrCodes, YCbCrMatrix, YCbCrOutput,
    YCbCrRange,
};
use metal2vulkan::device::VkProperty;
use metal2vulkan::error::{BindError, Feature, RuntimeDriverDefect};
use metal2vulkan::metal::format::{ycbcr_format, ChromaSubsampling, YCbCrReads};
use metal2vulkan::metal::{
    PixelFormat, PrimitiveType, ResourceUsage, SamplerAddressMode, SamplerBorderColor,
    SamplerMinMagFilter, SamplerMipFilter, TextureSwizzle, TextureSwizzleChannels, TextureType,
    TextureUsage,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, PrecisionNote, RenderPipelineDescriptor,
    StageSource,
};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, CommandBuffer, CommandQueue, ComputePipelineState, Deallocator,
    Degradation, Device, Error, Invalid, IoSurfacePlane, IoSurfaceYCbCr, Library, LoadAction,
    MappedSampling, Nil, Origin, PipelineOptions, Problem, Region, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions, RuntimeGap, SamplerState, Size, StoreAction,
    SurfaceYCbCrMatrix, Texture, TextureDescriptor, TextureProblem,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(60);

/// The surface's size, and its planes' rows and offsets.
const W: u64 = 40;
const H: u64 = 24;
const ROW: u64 = 64;
const CHROMA_AT: u64 = ROW * H;
const CW: u64 = W / 2;
const CH: u64 = H / 2;
const LENGTH: u64 = 4096;
const VIDEO: u32 = u32::from_be_bytes(*b"420v");
const FULL: u32 = u32::from_be_bytes(*b"420f");

/// Synthetic AIR written for this lane: `read_texels` reads texel `gid` of texture 0 into
/// element `gid.y * width + gid.x` of buffer 0; `sample_texels` samples texture 0 through
/// sampler 0 at that texel's centre, and `sample_between` at [`BETWEEN`] past it; `fill` writes
/// buffer 0's `float4` to every texel of texture 0. `sample_outside`, dispatched over twice the
/// texture's size, samples point `gid` at the centre of texel `gid - size / 4`, a quarter of
/// the points inside the texture and the rest outside it. `write_masked` writes buffer 0's
/// `float4` element `gid.y * width + gid.x` to texel `gid` of texture 0 where buffer 1's `uint`
/// element there is not 0, and `write_masked_half` the same of `half4`s. `read_array` and
/// `sample_array_between` are `read_texels` and `sample_between` of slice 0 of a 2D array.
pub(super) const KERNELS: &str = r#"; Owned synthetic kernels.
source_filename = "ycbcr_textures.metal"

define void @read_texels(ptr addrspace(1) %t, ptr addrspace(1) noalias %out, <2 x i32> %gid, <2 x i32> %size) {
entry:
  %s = tail call ptr addrspace(2) @air.get_read_sampler()
  %r = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %gid, <2 x i32> zeroinitializer, i32 0, i32 1)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %w = extractelement <2 x i32> %size, i64 0
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %j = zext i32 %i to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %j
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}

define void @sample_texels(ptr addrspace(1) %t, ptr addrspace(1) noalias %out, ptr addrspace(2) %smp, <2 x i32> %gid, <2 x i32> %size) {
entry:
  %g = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %gid)
  %n = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %size)
  %c = fadd <2 x float> %g, <float 5.000000e-01, float 5.000000e-01>
  %uv = fdiv <2 x float> %c, %n
  %r = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %smp, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %w = extractelement <2 x i32> %size, i64 0
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %j = zext i32 %i to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %j
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}

define void @sample_between(ptr addrspace(1) %t, ptr addrspace(1) noalias %out, ptr addrspace(2) %smp, <2 x i32> %gid, <2 x i32> %size) {
entry:
  %g = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %gid)
  %n = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %size)
  %c = fadd <2 x float> %g, <float 9.062500e-01, float 7.187500e-01>
  %uv = fdiv <2 x float> %c, %n
  %r = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %smp, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %w = extractelement <2 x i32> %size, i64 0
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %j = zext i32 %i to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %j
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}

define void @read_half(ptr addrspace(1) %t, ptr addrspace(1) noalias %out, <2 x i32> %gid, <2 x i32> %size) {
entry:
  %s = tail call ptr addrspace(2) @air.get_read_sampler()
  %r = tail call { <4 x half>, i8 } @air.read_texture_2d.v4f16(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %gid, <2 x i32> zeroinitializer, i32 0, i32 1)
  %h = extractvalue { <4 x half>, i8 } %r, 0
  %v = fpext <4 x half> %h to <4 x float>
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %w = extractelement <2 x i32> %size, i64 0
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %j = zext i32 %i to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %j
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}

define void @sample_half(ptr addrspace(1) %t, ptr addrspace(1) noalias %out, ptr addrspace(2) %smp, <2 x i32> %gid, <2 x i32> %size) {
entry:
  %g = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %gid)
  %n = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %size)
  %c = fadd <2 x float> %g, <float 5.000000e-01, float 5.000000e-01>
  %uv = fdiv <2 x float> %c, %n
  %r = tail call { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1) %t, ptr addrspace(2) %smp, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %h = extractvalue { <4 x half>, i8 } %r, 0
  %v = fpext <4 x half> %h to <4 x float>
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %w = extractelement <2 x i32> %size, i64 0
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %j = zext i32 %i to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %j
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}

define void @sample_outside(ptr addrspace(1) %t, ptr addrspace(1) noalias %out, ptr addrspace(2) %smp, <2 x i32> %gid, <2 x i32> %size) {
entry:
  %g = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %gid)
  %n = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %size)
  %q = fmul <2 x float> %n, <float 2.500000e-01, float 2.500000e-01>
  %c0 = fadd <2 x float> %g, <float 5.000000e-01, float 5.000000e-01>
  %c = fsub <2 x float> %c0, %q
  %m = fmul <2 x float> %n, <float 5.000000e-01, float 5.000000e-01>
  %uv = fdiv <2 x float> %c, %m
  %r = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %smp, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %w = extractelement <2 x i32> %size, i64 0
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %j = zext i32 %i to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %j
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}

; ---- Gathers of red and blue at texel corners ----
define void @gather_red(ptr addrspace(1) %t, ptr addrspace(1) noalias %out, ptr addrspace(2) %smp, <2 x i32> %gid, <2 x i32> %size) {
entry:
  %g = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %gid)
  %n0 = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %size)
  %n = fsub <2 x float> %n0, <float 1.000000e+00, float 1.000000e+00>
  %uv = fdiv <2 x float> %g, %n
  %r = tail call { <4 x float>, i8 } @air.gather_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %smp, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i32 0, i32 0)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %w = extractelement <2 x i32> %size, i64 0
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %j = zext i32 %i to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %j
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}

define void @gather_blue(ptr addrspace(1) %t, ptr addrspace(1) noalias %out, ptr addrspace(2) %smp, <2 x i32> %gid, <2 x i32> %size) {
entry:
  %g = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %gid)
  %n0 = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %size)
  %n = fsub <2 x float> %n0, <float 1.000000e+00, float 1.000000e+00>
  %uv = fdiv <2 x float> %g, %n
  %r = tail call { <4 x float>, i8 } @air.gather_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %smp, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i32 2, i32 0)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %w = extractelement <2 x i32> %size, i64 0
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %j = zext i32 %i to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %j
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}

define void @gather_red_half(ptr addrspace(1) %t, ptr addrspace(1) noalias %out, ptr addrspace(2) %smp, <2 x i32> %gid, <2 x i32> %size) {
entry:
  %g = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %gid)
  %n0 = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %size)
  %n = fsub <2 x float> %n0, <float 1.000000e+00, float 1.000000e+00>
  %uv = fdiv <2 x float> %g, %n
  %r = tail call { <4 x half>, i8 } @air.gather_texture_2d.v4f16(ptr addrspace(1) %t, ptr addrspace(2) %smp, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i32 0, i32 0)
  %h = extractvalue { <4 x half>, i8 } %r, 0
  %v = fpext <4 x half> %h to <4 x float>
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %w = extractelement <2 x i32> %size, i64 0
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %j = zext i32 %i to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %j
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}

define void @gather_blue_half(ptr addrspace(1) %t, ptr addrspace(1) noalias %out, ptr addrspace(2) %smp, <2 x i32> %gid, <2 x i32> %size) {
entry:
  %g = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %gid)
  %n0 = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %size)
  %n = fsub <2 x float> %n0, <float 1.000000e+00, float 1.000000e+00>
  %uv = fdiv <2 x float> %g, %n
  %r = tail call { <4 x half>, i8 } @air.gather_texture_2d.v4f16(ptr addrspace(1) %t, ptr addrspace(2) %smp, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i32 2, i32 0)
  %h = extractvalue { <4 x half>, i8 } %r, 0
  %v = fpext <4 x half> %h to <4 x float>
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %w = extractelement <2 x i32> %size, i64 0
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %j = zext i32 %i to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %j
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}

define void @fill(ptr addrspace(1) %t, ptr addrspace(1) %value, <2 x i32> %gid) {
entry:
  %v = load <4 x float>, ptr addrspace(1) %value, align 16
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %t, <2 x i32> %gid, <4 x float> %v, i32 0, i32 2)
  ret void
}

define void @write_masked(ptr addrspace(1) %t, ptr addrspace(1) noalias %values, ptr addrspace(1) noalias %mask, <2 x i32> %gid, <2 x i32> %size) {
entry:
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %w = extractelement <2 x i32> %size, i64 0
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %j = zext i32 %i to i64
  %mp = getelementptr inbounds i32, ptr addrspace(1) %mask, i64 %j
  %m = load i32, ptr addrspace(1) %mp, align 4
  %on = icmp ne i32 %m, 0
  br i1 %on, label %write, label %done

write:
  %vp = getelementptr inbounds <4 x float>, ptr addrspace(1) %values, i64 %j
  %v = load <4 x float>, ptr addrspace(1) %vp, align 16
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %t, <2 x i32> %gid, <4 x float> %v, i32 0, i32 2)
  br label %done

done:
  ret void
}

define void @write_masked_half(ptr addrspace(1) %t, ptr addrspace(1) noalias %values, ptr addrspace(1) noalias %mask, <2 x i32> %gid, <2 x i32> %size) {
entry:
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %w = extractelement <2 x i32> %size, i64 0
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %j = zext i32 %i to i64
  %mp = getelementptr inbounds i32, ptr addrspace(1) %mask, i64 %j
  %m = load i32, ptr addrspace(1) %mp, align 4
  %on = icmp ne i32 %m, 0
  br i1 %on, label %write, label %done

write:
  %vp = getelementptr inbounds <4 x half>, ptr addrspace(1) %values, i64 %j
  %v = load <4 x half>, ptr addrspace(1) %vp, align 8
  tail call void @air.write_texture_2d.v4f16(ptr addrspace(1) %t, <2 x i32> %gid, <4 x half> %v, i32 0, i32 2)
  br label %done

done:
  ret void
}

define void @read_array(ptr addrspace(1) %t, ptr addrspace(1) noalias %out, <2 x i32> %gid, <2 x i32> %size) {
entry:
  %s = tail call ptr addrspace(2) @air.get_read_sampler()
  %r = tail call { <4 x float>, i8 } @air.read_texture_2d_array.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %gid, i32 0, <2 x i32> zeroinitializer, i32 0, i32 1)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %w = extractelement <2 x i32> %size, i64 0
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %j = zext i32 %i to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %j
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}

define void @sample_array_between(ptr addrspace(1) %t, ptr addrspace(1) noalias %out, ptr addrspace(2) %smp, <2 x i32> %gid, <2 x i32> %size) {
entry:
  %g = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %gid)
  %n = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %size)
  %c = fadd <2 x float> %g, <float 9.062500e-01, float 7.187500e-01>
  %uv = fdiv <2 x float> %c, %n
  %r = tail call { <4 x float>, i8 } @air.sample_texture_2d_array.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %smp, <2 x float> %uv, i32 0, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %w = extractelement <2 x i32> %size, i64 0
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %j = zext i32 %i to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %j
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}

declare ptr addrspace(2) @air.get_read_sampler()
declare <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32>)
declare { <4 x float>, i8 } @air.read_texture_2d_array.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, i32, <2 x i32>, i32, i32)
declare { <4 x float>, i8 } @air.sample_texture_2d_array.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i32, i1, <2 x i32>, i1, float, float, i32)
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare { <4 x half>, i8 } @air.read_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
declare { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)
declare void @air.write_texture_2d.v4f16(ptr addrspace(1), <2 x i32>, <4 x half>, i32, i32)
declare { <4 x float>, i8 } @air.gather_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)
declare { <4 x half>, i8 } @air.gather_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)

!air.kernel = !{!0, !10, !20, !40, !50, !60, !70, !80, !81, !82, !83, !90, !96, !100, !110}
!air.version = !{!30}
!air.language_version = !{!31}
!0 = !{ptr @read_texels, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
!6 = !{i32 3, !"air.threads_per_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"size"}
!10 = !{ptr @sample_texels, !1, !11}
!11 = !{!12, !4, !13, !14, !15}
!12 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}
!13 = !{i32 2, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"smp"}
!14 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
!15 = !{i32 4, !"air.threads_per_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"size"}
!20 = !{ptr @fill, !1, !21}
!21 = !{!22, !23, !24}
!22 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"t"}
!23 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"value"}
!24 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
!40 = !{ptr @sample_between, !1, !11}
!50 = !{ptr @read_half, !1, !51}
!51 = !{!52, !4, !5, !6}
!52 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<half, read>", !"air.arg_name", !"t"}
!60 = !{ptr @sample_half, !1, !61}
!61 = !{!62, !4, !13, !14, !15}
!62 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<half, sample>", !"air.arg_name", !"t"}
!70 = !{ptr @sample_outside, !1, !11}
!80 = !{ptr @gather_red, !1, !11}
!81 = !{ptr @gather_blue, !1, !11}
!82 = !{ptr @gather_red_half, !1, !61}
!83 = !{ptr @gather_blue_half, !1, !61}
!90 = !{ptr @write_masked, !1, !91}
!91 = !{!22, !92, !93, !94, !95}
!92 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"values"}
!93 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"mask"}
!94 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
!95 = !{i32 4, !"air.threads_per_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"size"}
!96 = !{ptr @write_masked_half, !1, !97}
!97 = !{!98, !99, !93, !94, !95}
!98 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"t"}
!99 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"half4", !"air.arg_name", !"values"}
!100 = !{ptr @read_array, !1, !101}
!101 = !{!102, !4, !5, !6}
!102 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d_array<float, read>", !"air.arg_name", !"t"}
!110 = !{ptr @sample_array_between, !1, !111}
!111 = !{!112, !4, !13, !14, !15}
!112 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d_array<float, sample>", !"air.arg_name", !"t"}
!30 = !{i32 2, i32 8, i32 0}
!31 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// The lane's kernels on one device.
struct Kernels {
    read: ComputePipelineState,
    sample: ComputePipelineState,
    between: ComputePipelineState,
    fill: ComputePipelineState,
    /// A `half` `read()` and a `half` sample at the texels' centres, widened to `float`.
    read_half: ComputePipelineState,
    sample_half: ComputePipelineState,
    outside: ComputePipelineState,
    /// Gathers of red and of blue at the texel corners, `float` and `half` widened to `float`.
    gathers: [[ComputePipelineState; 2]; 2],
    /// Writes of `float4` and of `half4` values to the texels a mask names.
    write_masked: [ComputePipelineState; 2],
    /// `read_texels` and `sample_between` of slice 0 of a 2D array.
    read_array: ComputePipelineState,
    sample_array: ComputePipelineState,
    /// Whether the device samples through Metal's own texture unit (`VkProperty::MetalSampler`):
    /// a read's `float` texel of a format's codes is then Metal's conversion, elsewhere the
    /// device's (`device-texel-conversion`, [`converted_holds`]).
    exact: bool,
    /// Whether `read_half`'s stage rounds a `half` by the device's conversion
    /// (`PrecisionNote::NativeHalfConversion`), not as Metal does.
    native_half: bool,
}

impl Kernels {
    fn new(device: &Device) -> Kernels {
        let bitcode = llvm_assemble_text(KERNELS).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
        let state = |name: &str| {
            let function = library.new_function_with_name(name).unwrap();
            device
                .new_compute_pipeline_state(
                    &ComputePipelineDescriptor::new(StageSource::Function(
                        function.function_descriptor(),
                    )),
                    &PipelineOptions::default(),
                )
                .unwrap()
        };
        Kernels {
            read: state("read_texels"),
            sample: state("sample_texels"),
            between: state("sample_between"),
            fill: state("fill"),
            read_half: state("read_half"),
            sample_half: state("sample_half"),
            outside: state("sample_outside"),
            gathers: [
                [state("gather_red"), state("gather_red_half")],
                [state("gather_blue"), state("gather_blue_half")],
            ],
            write_masked: [state("write_masked"), state("write_masked_half")],
            read_array: state("read_array"),
            sample_array: state("sample_array_between"),
            exact: metal_sampler(device),
            native_half: stage_notes(device, &bitcode, "read_half")
                .contains(&PrecisionNote::NativeHalfConversion),
        }
    }
}

/// Our probes' generator of random bytes (`ycc1.swift`, `ycc2.swift`).
struct Random(u64);

impl Random {
    fn next(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u8
    }
}

/// The bytes of a surface whose planes our probes' generator at `seed` fills (luma, then the
/// chroma pairs, each packed), laid out as the module docs say, every other byte `0xa5`.
fn surface_bytes(seed: u64) -> Vec<u8> {
    let mut random = Random(seed);
    let mut bytes = vec![0xa5; LENGTH as usize];
    for y in 0..H {
        for x in 0..W {
            bytes[(y * ROW + x) as usize] = random.next();
        }
    }
    for y in 0..CH {
        for x in 0..2 * CW {
            bytes[(CHROMA_AT + y * ROW + x) as usize] = random.next();
        }
    }
    bytes
}

/// Each texel's red, green and blue `v` of `v / 2048` the conversion gives the surface `bytes`.
fn modelled(bytes: &[u8], range: YCbCrRange, matrix: YCbCrMatrix) -> Vec<[u16; 3]> {
    let codes = ycbcr_format(PixelFormat::YCBCR84202P)
        .and_then(YCbCrCodes::of)
        .unwrap();
    modelled_as(bytes, codes, (range, matrix))
}

/// Each texel `codes` (8-bit) reads of the surface `bytes` under `conversion`
/// (`metal2vulkan::bind::ycbcr_read`).
fn modelled_as(
    bytes: &[u8],
    codes: YCbCrCodes,
    conversion: (YCbCrRange, YCbCrMatrix),
) -> Vec<[u16; 3]> {
    let taps = |q: u64, n: u64| -> [(u64, u32); 2] {
        let c = q / 2;
        if q % 2 == 1 {
            [(c, 3), ((c + 1).min(n - 1), 1)]
        } else {
            [(c.saturating_sub(1), 1), (c, 3)]
        }
    };
    let mut out = Vec::new();
    for y in 0..H {
        for x in 0..W {
            let mut sums = [0u32; 2];
            for (cy, wy) in taps(y, CH) {
                for (cx, wx) in taps(x, CW) {
                    let at = (CHROMA_AT + cy * ROW + 2 * cx) as usize;
                    sums[0] += wx * wy * u32::from(bytes[at]);
                    sums[1] += wx * wy * u32::from(bytes[at + 1]);
                }
            }
            let [cb, cr] = sums.map(|s| ((s + 8) >> 4) as u16);
            let luma = u16::from(bytes[(y * ROW + x) as usize]);
            out.push(ycbcr_read(codes, conversion, [luma, cb, cr]));
        }
    }
    out
}

/// Metal's `read()` of the surface of seed 5 as a `420v` one without a matrix: each texel's red,
/// green and blue `v` of `v / 2048` as little-endian 16-bit words, row by row (an Apple M3 Max
/// under macOS 27.0.1, through `fixtures/public/ycbcr/probe.swift` over an IOSurface laid out as
/// the lane lays its stand-in buffer out; an Apple M4 under 27.0.1 answered the same
/// [`METAL_READS_FNV`]).
const METAL_READS: &[u8] = include_bytes!("../../fixtures/public/ycbcr/metal_m3_max.bin");

/// The SHA-256 of [`METAL_READS`].
const METAL_READS_SHA: &str = "88301e8b9aea601017cfc7466ba830c9255fb1875dcb9d62ffd2d4a8e14d8b8e";

/// The FNV-1a of [`METAL_READS`], which the M4's probe printed of its own reads.
const METAL_READS_FNV: u32 = 0x26e2_da15;

/// The SHA-256 of the surface bytes the generator gives at seed 5, the surface Metal read
/// ([`METAL_READS`]).
const SURFACE_5_SHA: &str = "bb42f2090b661a1ef27876d6e3ff2404436756016fbf912cafdb9b35bfc6d3b1";

/// [`METAL_READS`] as each texel's red, green and blue step.
fn metal_reads() -> Vec<[u16; 3]> {
    METAL_READS
        .as_chunks::<6>()
        .0
        .iter()
        .map(|t| std::array::from_fn(|c| u16::from_le_bytes([t[2 * c], t[2 * c + 1]])))
        .collect()
}

/// The texel `read` and Metal's differ at, with both steps, or `None` where they are alike.
fn unlike_metal(read: &[[u16; 3]]) -> Option<String> {
    let metal = metal_reads();
    assert_eq!(read.len(), metal.len(), "the texel count");
    read.iter()
        .zip(&metal)
        .position(|(got, want)| got != want)
        .map(|i| {
            let (x, y) = (i as u64 % W, i as u64 / W);
            format!(
                "texel ({x}, {y}) reads {:?}, Metal's {:?}",
                read[i], metal[i]
            )
        })
}

/// The FNV-1a of `bytes`.
fn fnv(bytes: &[u8]) -> u32 {
    bytes.iter().fold(2_166_136_261u32, |h, &b| {
        (h ^ u32::from(b)).wrapping_mul(16_777_619)
    })
}

/// The inputs and the answers are the probes': the surface the generator gives at seed 5, the
/// recorded reads of it, their length and their FNV-1a, which the M4's probe printed of its own.
#[test]
fn the_inputs_and_the_reads_are_the_probes() {
    assert_eq!(
        crate::hash::hex_encode(Sha256::digest(surface_bytes(5))),
        SURFACE_5_SHA,
        "the surface of seed 5"
    );
    assert_eq!(METAL_READS.len(), (W * H) as usize * 6, "the reads' length");
    assert_eq!(
        crate::hash::hex_encode(Sha256::digest(METAL_READS)),
        METAL_READS_SHA,
        "the reads"
    );
    assert_eq!(fnv(METAL_READS), METAL_READS_FNV, "the M4's digest");
}

/// The surface's facts for OSType `pixel_format` and `matrix`.
fn facts(pixel_format: u32, matrix: Option<SurfaceYCbCrMatrix>) -> IoSurfaceYCbCr {
    let facts = IoSurfaceYCbCr::new(pixel_format, 0, [W, H], CHROMA_AT, ROW);
    match matrix {
        Some(matrix) => facts.with_matrix(matrix),
        None => facts,
    }
}

/// The `YCBCR8_420_2P` texture of `surface`'s plane 0 with `facts`, of `usage`.
fn ycbcr_texture(
    surface: &Buffer,
    facts: Option<IoSurfaceYCbCr>,
    usage: TextureUsage,
) -> Result<Texture, Error> {
    eight_bit_texture(surface, PixelFormat::YCBCR84202P, facts, usage)
}

/// The texture of `pf` (an 8-bit two-plane format) of `surface`'s plane 0 with `facts`, of
/// `usage`.
fn eight_bit_texture(
    surface: &Buffer,
    pf: PixelFormat,
    facts: Option<IoSurfaceYCbCr>,
    usage: TextureUsage,
) -> Result<Texture, Error> {
    let mut d = TextureDescriptor::texture_2d(pf, W, H, false);
    d.usage = usage;
    d.resource_options = ResourceOptions::shared();
    let plane = IoSurfacePlane::new(0, ROW, true);
    let plane = match facts {
        Some(facts) => plane.with_ycbcr(facts),
        None => plane,
    };
    surface.new_texture_with_iosurface_plane(&d, plane)
}

/// A texture of `pf` and `size` over `surface`'s plane at `offset`, rows [`ROW`] apart, of
/// shader read and write usage.
fn plane_texture(surface: &Buffer, pf: PixelFormat, size: [u64; 2], offset: u64) -> Texture {
    let mut d = TextureDescriptor::texture_2d(pf, size[0], size[1], false);
    d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    d.resource_options = ResourceOptions::shared();
    surface
        .new_texture_with_iosurface_plane(&d, IoSurfacePlane::new(offset, ROW, true))
        .unwrap()
}

/// The `v` of `v / 2048` of each channel of `texels` (`float4`s), checked to be on the grid,
/// alpha checked to be 1.
fn grid(texels: &[u8]) -> Vec<[u16; 3]> {
    grid_of(texels, 2048)
}

/// The `v` of `v / d` of each channel of `texels` (`float4`s), checked to be on the grid `d`,
/// alpha checked to be 1.
fn grid_of(texels: &[u8], d: u32) -> Vec<[u16; 3]> {
    let d = d as f32;
    texels
        .as_chunks::<16>()
        .0
        .iter()
        .map(|t| {
            let c = |i: usize| f32::from_le_bytes(t[4 * i..4 * i + 4].try_into().unwrap());
            assert_eq!(c(3), 1.0, "alpha 1");
            std::array::from_fn(|i| {
                let v = c(i) * d;
                assert!(v == v.round() && (0.0..=d).contains(&v), "{v} off the grid");
                v as u16
            })
        })
        .collect()
}

/// Dispatch `state` over the texture's texels with `texture`, `out` and `sampler` bound, in a
/// new compute encoder of `command_buffer`.
fn dispatch(
    command_buffer: &CommandBuffer,
    state: &ComputePipelineState,
    texture: &Texture,
    out: &Buffer,
    sampler: Option<&SamplerState>,
    size: [u64; 2],
) {
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_buffer(Some(out), 0, 0).unwrap();
    if let Some(sampler) = sampler {
        encoder.set_sampler_state(Some(sampler), 0).unwrap();
    }
    encoder
        .dispatch_threads(Size::new(size[0], size[1], 1), Size::new(8, 8, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
}

/// What `state` (reading or sampling texture 0 into buffer 0) reads of `texture`, through
/// `sampler`, in one command buffer of its own.
fn texels_of(
    device: &Device,
    state: &ComputePipelineState,
    texture: &Texture,
    sampler: Option<&SamplerState>,
) -> Vec<u8> {
    let out = device
        .new_buffer(16 * W * H, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    dispatch(&command_buffer, state, texture, &out, sampler, [W, H]);
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    read_buffer(&out, 0, 16 * W * H).unwrap()
}

/// A sampler of `filter` for both min and mag.
fn sampler(device: &Device, filter: SamplerMinMagFilter) -> SamplerState {
    let mut d = SamplerDescriptor::new();
    d.min_filter = filter;
    d.mag_filter = filter;
    device.new_sampler_state(&d).unwrap()
}

/// Linear samplers through each mip filter: a YCbCr texture has one level, which Metal's mip
/// filter leaves as it is.
fn linear_samplers(device: &Device) -> Vec<(SamplerMipFilter, SamplerState)> {
    [
        SamplerMipFilter::NotMipmapped,
        SamplerMipFilter::Nearest,
        SamplerMipFilter::Linear,
    ]
    .into_iter()
    .map(|mip| {
        let mut d = SamplerDescriptor::new();
        d.min_filter = SamplerMinMagFilter::Linear;
        d.mag_filter = SamplerMinMagFilter::Linear;
        d.mip_filter = mip;
        (mip, device.new_sampler_state(&d).unwrap())
    })
    .collect()
}

/// A kernel's `read()` and a nearest sample of every texel are the conversion of the surface's
/// planes, Metal's for the measured surface, for both ranges and each matrix (BT.601, BT.709 and
/// BT.2020); unannounced.
#[test]
fn ycbcr_textures_read_the_texels_metal_converts() {
    on_device(|device, messages| {
        let kernels = Kernels::new(device);
        let nearest = sampler(device, SamplerMinMagFilter::Nearest);
        let conversions = [
            (VIDEO, None, YCbCrRange::Video, YCbCrMatrix::Bt601),
            (
                FULL,
                Some(SurfaceYCbCrMatrix::Itu601),
                YCbCrRange::Full,
                YCbCrMatrix::Bt601,
            ),
            (
                VIDEO,
                Some(SurfaceYCbCrMatrix::Itu709),
                YCbCrRange::Video,
                YCbCrMatrix::Bt709,
            ),
            (
                FULL,
                Some(SurfaceYCbCrMatrix::Itu709),
                YCbCrRange::Full,
                YCbCrMatrix::Bt709,
            ),
            (
                VIDEO,
                Some(SurfaceYCbCrMatrix::Itu2020),
                YCbCrRange::Video,
                YCbCrMatrix::Bt2020,
            ),
            (
                FULL,
                Some(SurfaceYCbCrMatrix::Itu2020),
                YCbCrRange::Full,
                YCbCrMatrix::Bt2020,
            ),
        ];
        for (seed, (ostype, matrix, range, model_matrix)) in (5..).zip(conversions) {
            let bytes = surface_bytes(seed);
            let surface = buffer_with(device, &bytes).unwrap();
            let texture = ycbcr_texture(
                &surface,
                Some(facts(ostype, matrix)),
                TextureUsage::SHADER_READ,
            )
            .unwrap();
            let read = grid(&texels_of(device, &kernels.read, &texture, None));
            assert_eq!(
                read,
                modelled(&bytes, range, model_matrix),
                "{range:?} {matrix:?}"
            );
            if seed == 5 {
                if let Some(unlike) = unlike_metal(&read) {
                    panic!("Metal's reads: {unlike}");
                }
            }
            let sampled = grid(&texels_of(
                device,
                &kernels.sample,
                &texture,
                Some(&nearest),
            ));
            assert_eq!(sampled, read, "a nearest sample is the texel");
        }
        let notices = counts(&messages.take_notices());
        assert_eq!(
            notices.get(&Degradation::FloatSampleFilter),
            None,
            "{notices:?}"
        );
    });
}

/// The fractions, in 1/256 steps, at which `sample_between` samples past each texel's centre:
/// `(0.90625, 0.71875)` texels past its corner.
pub(super) const BETWEEN: [u32; 2] = [104, 56];

/// Metal's step of a linear sample of a `YCBCR8_420_2P` texture at fractions `[a, b]` past texel
/// `(x, y)`, from the texels' steps `read`: the taps `(x, y)` to `(x + 1, y + 1)` clamped to the
/// edge, an axis whose two taps address one texel weighed `(256, 0)`, each term `k · w` truncated
/// by `2^7`, then `(T + 2^8) >> 9`.
fn metal_step(read: &[[u16; 3]], (x, y): (u64, u64), [a, b]: [u32; 2], c: usize) -> u32 {
    let (x1, y1) = ((x + 1).min(W - 1), (y + 1).min(H - 1));
    let a = if x1 == x { 0 } else { a };
    let b = if y1 == y { 0 } else { b };
    let k = |x: u64, y: u64| u32::from(read[(y * W + x) as usize][c]);
    let taps = [(k(x, y), 256 - a, 256 - b), (k(x1, y), a, 256 - b)];
    let taps = taps
        .into_iter()
        .chain([(k(x, y1), 256 - a, b), (k(x1, y1), a, b)]);
    let t: u32 = taps.map(|(k, wx, wy)| (k * wx * wy) >> 7).sum();
    (t + 256) >> 9
}

/// The `half` bits of step `k / 2048` (`k` within `0..=2048`), which a `half` holds exactly.
fn half_of_step(k: u16) -> u16 {
    if k == 0 {
        return 0;
    }
    let lead = 15 - k.leading_zeros() as u16;
    // `k / 2048` is `2^(lead - 11)` times `k / 2^lead` in `[1, 2)`.
    let significand = match lead {
        11 => k >> 1,
        _ => k << (10 - lead),
    } & 0x3ff;
    ((lead + 4) << 10) | significand
}

/// A linear sample of the exact texels: at each texel's centre it is the texel, and between texels
/// Metal's step where the planes are read in place (a device whose memory is the host's), and
/// elsewhere the filter of the kept image ([`Between`]), announced as [`assert_filter_notices`]
/// states.
#[test]
fn a_linear_sample_of_a_ycbcr_texture_is_metals() {
    on_device(|device, messages| {
        let kernels = Kernels::new(device);
        let linear = sampler(device, SamplerMinMagFilter::Linear);
        let bytes = surface_bytes(9);
        let surface = buffer_with(device, &bytes).unwrap();
        let before = made_in_place(device);
        let texture = ycbcr_texture(
            &surface,
            Some(facts(VIDEO, None)),
            TextureUsage::SHADER_READ,
        )
        .unwrap();
        let in_place = made_in_place(device) > before;
        let read = grid(&texels_of(device, &kernels.read, &texture, None));
        let sampled = grid(&texels_of(device, &kernels.sample, &texture, Some(&linear)));
        assert_eq!(
            sampled, read,
            "a linear sample at a texel's centre is the texel"
        );
        let between = texels_of(device, &kernels.between, &texture, Some(&linear));
        let mut steps = Between::new(device);
        steps.texels(&between, &read, (2048, in_place), "YCBCR8_420_2P");
        steps.finish(device, &counts(&messages.take_notices()), "YCBCR8_420_2P");
    });
}

/// A linear sample between the texels through a constexpr sampler and through a runtime sampler
/// slot of each mip filter, each with a `float` and a `half` result, is Metal's step (every value
/// on the grid `2048` is a `half`) where the planes are read in place, and announces no filter
/// notice there: the path the consumer measured off the grid, whose slot the texture's alpha-one
/// swizzle had left on the device's filter, and a runtime sampler's mip filter, which Metal's
/// sample of the texture's one level does not read. Elsewhere it is the filter of the kept image
/// ([`Between`]), announced as [`assert_filter_notices`] states.
#[test]
fn a_linear_sample_is_metals_through_each_sampler_and_result() {
    on_device(|device, messages| {
        let kernels = Kernels::new(device);
        let surface = buffer_with(device, &surface_bytes(9)).unwrap();
        let before = made_in_place(device);
        let texture = ycbcr_texture(
            &surface,
            Some(facts(VIDEO, None)),
            TextureUsage::SHADER_READ,
        )
        .unwrap();
        let in_place = made_in_place(device) > before;
        let mut steps = Between::new(device);
        let read = grid(&texels_of(device, &kernels.read, &texture, None));
        let at = |p: u64, f: u32, n: u64| (p as f32 + 0.5 + f as f32 / 256.0) / n as f32;
        let jobs: Vec<[f32; 4]> = (0..W * H)
            .map(|i| [at(i % W, BETWEEN[0], W), at(i / W, BETWEEN[1], H), 0.0, 0.0])
            .collect();
        let samplers = linear_samplers(device);
        let runtime = samplers
            .iter()
            .map(|(mip, s)| (Through::Runtime, Some((*mip, s))));
        for (through, runtime) in [(Through::Constexpr(LINEAR_EDGE), None)]
            .into_iter()
            .chain(runtime)
        {
            for half in [false, true] {
                let source = argument_linear_samples::kernel(false, through, false, half);
                let state = pipeline_of(device, &llvm_assemble_text(&source).unwrap(), "k");
                let held = (&texture, Held::Slot, runtime.map(|(_, s)| s));
                let got = run_held(device, held, &state, &jobs, 1);
                let mip = runtime.map(|(mip, _)| mip);
                steps.sampled(in_place);
                for (i, texel) in (0..W * H).zip(got.as_chunks::<4>().0.iter()) {
                    let at = (i % W, i / W);
                    let what = format!("texel {at:?} through {through:?} {mip:?}, half {half}");
                    assert_eq!(texel[3], 1.0, "{what}");
                    for (c, &value) in texel.iter().enumerate().take(3) {
                        let want = metal_step(&read, at, BETWEEN, c) as f32 / 2048.0;
                        steps.check((value, want), (2048, half, in_place), &what);
                    }
                }
            }
        }
        steps.finish(device, &counts(&messages.take_notices()), "each sampler");
    });
}

/// Synthetic AIR written for this lane: vertex `corner` places vertex `vid` at buffer 0's
/// `float4` `vid`; fragment `between_{runtime,constexpr}_{float,half}` samples texture 0 through
/// sampler slot 0, or a constexpr sampler of [`LINEAR_EDGE`]'s words, at [`BETWEEN`] past the
/// pixel's texel, with a `float` or a `half` result, and writes it (widened) to color 0.
fn fragments() -> String {
    let mut functions = String::new();
    let mut entries = Vec::new();
    let mut metadata = String::new();
    for (n, (constexpr, half)) in [(false, false), (false, true), (true, false), (true, true)]
        .into_iter()
        .enumerate()
    {
        let name = format!(
            "between_{}_{}",
            if constexpr { "constexpr" } else { "runtime" },
            if half { "half" } else { "float" }
        );
        let (element, ty) = if half {
            ("half", "v4f16")
        } else {
            ("float", "v4f32")
        };
        let (param, operand, args) = match constexpr {
            true => (
                "",
                "@__air_sampler_state",
                format!("!101, !{}", 110 + 10 * n),
            ),
            false => (
                ", ptr addrspace(2) %s",
                "%s",
                format!("!101, !{}, !103", 110 + 10 * n),
            ),
        };
        let widen = match half {
            true => "  %h = extractvalue { <4 x half>, i8 } %r, 0\n  %v = fpext <4 x half> %h to <4 x float>\n",
            false => "  %v = extractvalue { <4 x float>, i8 } %r, 0\n",
        };
        functions.push_str(&format!(
            r#"define <4 x float> @{name}(<4 x float> %pos, ptr addrspace(1) %t{param}) {{
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %c = fadd <2 x float> %xy, <float 4.062500e-01, float 2.187500e-01>
  %uv = fdiv <2 x float> %c, <float {W}.0, float {H}.0>
  %r = call {{ <4 x {element}>, i8 }} @air.sample_texture_2d.{ty}(ptr addrspace(1) %t, ptr addrspace(2) {operand}, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)
{widen}  ret <4 x float> %v
}}

"#
        ));
        let id = 110 + 10 * n;
        entries.push(format!("!{}", id + 1));
        metadata.push_str(&format!(
            "!{} = !{{ptr @{name}, !102, !{}}}\n!{id} = !{{i32 1, !\"air.texture\", !\"air.location_index\", i32 0, i32 1, !\"air.sample\", !\"air.arg_type_name\", !\"texture2d<{element}, sample>\", !\"air.arg_name\", !\"t\"}}\n!{} = !{{{args}}}\n",
            id + 1,
            id + 2,
            id + 2
        ));
    }
    format!(
        r#"; Owned synthetic shaders.
@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {LINEAR_EDGE}, i64 0], align 8

define <{{ <4 x float> }}> @corner(i32 %vid, ptr addrspace(2) %p) {{
  %i = zext i32 %vid to i64
  %q = getelementptr inbounds <4 x float>, ptr addrspace(2) %p, i64 %i
  %v = load <4 x float>, ptr addrspace(2) %q, align 16
  %o = insertvalue <{{ <4 x float> }}> undef, <4 x float> %v, 0
  ret <{{ <4 x float> }}> %o
}}

{functions}declare {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare {{ <4 x half>, i8 }} @air.sample_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

!air.vertex = !{{!90}}
!air.fragment = !{{{fragments}}}
!air.sampler_states = !{{!99}}
!air.version = !{{!97}}
!air.language_version = !{{!98}}
!90 = !{{ptr @corner, !91, !92}}
!91 = !{{!93}}
!92 = !{{!94, !95}}
!93 = !{{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}}
!94 = !{{i32 0, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vid"}}
!95 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"p"}}
!97 = !{{i32 2, i32 8, i32 0}}
!98 = !{{!"Metal", i32 4, i32 0, i32 0}}
!99 = !{{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}}
!100 = !{{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}}
!101 = !{{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}}
!102 = !{{!100}}
!103 = !{{i32 2, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}}
{metadata}"#,
        fragments = entries.join(", ")
    )
}

/// What a draw of `fragment` (over the whole [`W`]x[`H`] `RGBA32Float` target) samples of
/// `texture`, through `sampler` at sampler slot 0 where given: each pixel's `float4`.
fn drawn_between(
    device: &Device,
    library: &Library,
    fragment: &str,
    texture: &Texture,
    sampler: Option<&SamplerState>,
) -> Vec<[f32; 4]> {
    drawn_after(device, library, fragment, (texture, sampler), |_, _| {})
}

/// [`drawn_between`]'s draw, in a command buffer of a new queue in which `before` first encodes
/// its own work (given the queue, for work of another command buffer committed first).
fn drawn_after(
    device: &Device,
    library: &Library,
    fragment: &str,
    (texture, sampler): (&Texture, Option<&SamplerState>),
    before: impl FnOnce(&CommandQueue, &CommandBuffer),
) -> Vec<[f32; 4]> {
    let pipeline = between_state(device, library, fragment);
    drawn_with(device, &pipeline, (texture, sampler), before)
}

/// The render pipeline state of vertex `corner` and `fragment` into one `RGBA32Float` color.
fn between_state(device: &Device, library: &Library, fragment: &str) -> RenderPipelineState {
    let function = |name: &str| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("corner"));
    descriptor.fragment = Some(function(fragment));
    descriptor.color_attachments[0] =
        Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Float));
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// [`drawn_after`]'s draw through `pipeline`.
fn drawn_with(
    device: &Device,
    pipeline: &RenderPipelineState,
    (texture, sampler): (&Texture, Option<&SamplerState>),
    before: impl FnOnce(&CommandQueue, &CommandBuffer),
) -> Vec<[f32; 4]> {
    let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA32Float, W, H, false);
    target.usage = TextureUsage::RENDER_TARGET;
    target.resource_options = ResourceOptions::shared();
    let target = device.new_texture(&target).unwrap();
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
    let corners: Vec<u8> = [
        [-1.0f32, -1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [1.0, -1.0],
        [1.0, 1.0],
    ]
    .iter()
    .flat_map(|&[x, y]| [x, y, 0.0, 1.0])
    .flat_map(|v| v.to_le_bytes())
    .collect();
    let corners = buffer_with(device, &corners).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    before(&queue, &command_buffer);
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(pipeline).unwrap();
    encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
    encoder.set_fragment_texture(Some(texture), 0).unwrap();
    if let Some(sampler) = sampler {
        encoder
            .set_fragment_sampler_state(Some(sampler), 0)
            .unwrap();
    }
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let mut bytes = vec![0u8; (16 * W * H) as usize];
    target
        .get_bytes(&mut bytes, 16 * W, 0, Region::new_2d(0, 0, W, H), 0, 0)
        .unwrap();
    bytes
        .as_chunks::<16>()
        .0
        .iter()
        .map(|t| {
            std::array::from_fn(|i| f32::from_le_bytes(t[4 * i..4 * i + 4].try_into().unwrap()))
        })
        .collect()
}

/// A fragment function's linear sample between the texels, through a runtime sampler slot of
/// each mip filter and through a constexpr sampler, each with a `float` and a `half` result, is
/// Metal's step where the planes are read in place, and announces no filter notice there: the
/// fragment twin of `a_linear_sample_is_metals_through_each_sampler_and_result`, the filter of
/// the kept image elsewhere ([`Between`]). The pixel `(x, y)`'s centre plus
/// `(0.40625, 0.21875)` is the kernel's `(x, y) + (0.90625, 0.71875)`.
#[test]
fn a_fragments_linear_sample_is_metals_through_each_sampler_and_result() {
    on_device(|device, messages| {
        let kernels = Kernels::new(device);
        let surface = buffer_with(device, &surface_bytes(9)).unwrap();
        let before = made_in_place(device);
        let texture = ycbcr_texture(
            &surface,
            Some(facts(VIDEO, None)),
            TextureUsage::SHADER_READ,
        )
        .unwrap();
        let in_place = made_in_place(device) > before;
        let mut steps = Between::new(device);
        let read = grid(&texels_of(device, &kernels.read, &texture, None));
        let library = device
            .new_library_with_air(&llvm_assemble_text(&fragments()).unwrap())
            .unwrap();
        let samplers = linear_samplers(device);
        let runtime = ["between_runtime_float", "between_runtime_half"]
            .into_iter()
            .flat_map(|f| samplers.iter().map(move |(mip, s)| (f, Some((*mip, s)))));
        let constexpr = ["between_constexpr_float", "between_constexpr_half"].map(|f| (f, None));
        for (fragment, runtime) in runtime.chain(constexpr) {
            let got = drawn_between(
                device,
                &library,
                fragment,
                &texture,
                runtime.map(|(_, s)| s),
            );
            let mip = runtime.map(|(mip, _)| mip);
            let half = fragment.ends_with("half");
            steps.sampled(in_place);
            for (i, texel) in (0..W * H).zip(&got) {
                let at = (i % W, i / W);
                let what = format!("pixel {at:?} of {fragment} {mip:?}");
                assert_eq!(texel[3], 1.0, "{what}");
                for (c, &value) in texel.iter().enumerate().take(3) {
                    let want = metal_step(&read, at, BETWEEN, c) as f32 / 2048.0;
                    steps.check((value, want), (2048, half, in_place), &what);
                }
            }
        }
        steps.finish(device, &counts(&messages.take_notices()), "a fragment");
    });
}

/// How far the device's own linear filter of a two-plane YCbCr texture's texels may land from
/// Metal's filter of the format at coordinates on the 1/256 grid, in codes of `1/255`: the bound
/// `PrecisionNote::DeviceSampling` states for such a texture there (docs/FAST-PATHS.md), 0.498 on
/// NVIDIA and 0.125 on RADV, ANV and lavapipe; through Metal's own sampler one step of the grid
/// `1/2048`, 0.125 (`ycbcr-filter-not-exact`).
const DEVICE_FILTER_BOUND: f32 = 0.5;

/// The two-plane YCbCr textures `device` made with their planes read in place
/// (`WorkCounts::ycbcr_textures_in_place`, `ycbcr_in_place.rs`).
fn made_in_place(device: &Device) -> u64 {
    device.work_counts().ycbcr_textures_in_place
}

/// Linear samples between a converted two-plane YCbCr texture's texels against Metal's step
/// ([`metal_step`]): Metal's exactly where the texture's planes are read in place
/// (`ycbcr_in_place.rs`), whose function filters the codes as Metal does. A texture that keeps its
/// image is sampled by the filter of its converted texels: through Metal's own sampler
/// (`VkProperty::MetalSampler`) within one step of the grid, a `half` result within one `half` unit
/// in the last place (`ycbcr-filter-not-exact`); on any other device by its own filter, within
/// [`DEVICE_FILTER_BOUND`] (`device-sampling`). The values off Metal's step are counted, with the
/// largest distance in codes of `1/255`. The textures the device converts itself while it runs
/// are counted too (`ycbcr_device_reads`), each announced as `ycbcr-filter-not-exact`.
struct Between {
    metal: bool,
    /// The device's conversions before the run ([`device_conversions`]).
    converted: u64,
    /// Samples of textures that keep their image.
    kept: usize,
    checked: usize,
    differ: usize,
    most: f32,
}

impl Between {
    fn new(device: &Device) -> Between {
        Between {
            metal: metal_sampler(device),
            converted: device_conversions(device),
            kept: 0,
            checked: 0,
            differ: 0,
            most: 0.0,
        }
    }

    /// Note a sample of a texture whose planes are read in place where `in_place`.
    fn sampled(&mut self, in_place: bool) {
        self.kept += usize::from(!in_place);
    }

    /// Check channel value `got` against Metal's step `want` on the grid `grid`, of a sample with
    /// a `half` result where `half`, of a texture whose planes are read in place where `in_place`.
    fn check(
        &mut self,
        (got, want): (f32, f32),
        (grid, half, in_place): (u32, bool, bool),
        what: &str,
    ) {
        self.checked += 1;
        if got.to_bits() == want.to_bits() {
            return;
        }
        assert!(
            !in_place,
            "{what}: {got}, Metal's step {want}, its planes read in place"
        );
        self.differ += 1;
        let codes = (got - want).abs() * 255.0;
        self.most = self.most.max(codes);
        let holds = match (self.metal, half) {
            (true, true) => half_within(got, want),
            (true, false) => (got - want).abs() * grid as f32 <= 1.0,
            (false, _) => codes <= DEVICE_FILTER_BOUND,
        };
        assert!(holds, "{what}: {got}, Metal's step {want} ({codes} codes)");
    }

    /// Check the `float4` texels `got` (alpha 1) of a sample at [`BETWEEN`] against Metal's step of
    /// the texels `read` on the grid `grid`, of a texture whose planes are read in place where
    /// `in_place`.
    fn texels(&mut self, got: &[u8], read: &[[u16; 3]], (grid, in_place): (u32, bool), case: &str) {
        self.sampled(in_place);
        for (i, t) in (0u64..).zip(got.as_chunks::<16>().0) {
            let at = (i % W, i / W);
            let c = |i: usize| f32::from_le_bytes(t[4 * i..4 * i + 4].try_into().unwrap());
            assert_eq!(c(3), 1.0, "{case}: between, texel {at:?}'s alpha");
            for ch in 0..3 {
                let want = metal_step(read, at, BETWEEN, ch) as f32 / grid as f32;
                let what = format!("{case}: between, texel {at:?}");
                self.check((c(ch), want), (grid, false, in_place), &what);
            }
        }
    }

    /// Print the values off Metal's step, and check the run's notices `announced`
    /// ([`assert_filter_notices`]).
    fn finish(&self, device: &Device, announced: &BTreeMap<Degradation, usize>, case: &str) {
        if self.differ > 0 {
            println!(
                "YCBCR-BETWEEN-DIFFERS {} {case}: {} of {} values the filter of the kept image, at \
                 most {} codes off Metal's step",
                device.name(),
                self.differ,
                self.checked,
                self.most
            );
        }
        let by_device = device_conversions(device) > self.converted;
        assert_filter_notices(device, announced, (self.kept > 0, by_device));
    }
}

/// The filter notices `announced` of a run on `device` whose samples of converted two-plane
/// YCbCr textures read a texture that keeps its image where `kept`, and that made a texture the
/// device converts itself where `by_device`: `ycbcr-filter-not-exact` once where one did on a
/// device that samples through Metal's own sampler (a runtime sampler's slot bound to such a
/// texture records its filter note whatever the sampler's word, `lower::sampler::linear`), or
/// where the device converted one (`ycbcr_device_reads`), none elsewhere; no generic filter
/// notice; `device-sampling` on every device that samples natively, and only there.
fn assert_filter_notices(
    device: &Device,
    announced: &BTreeMap<Degradation, usize>,
    (kept, by_device): (bool, bool),
) {
    let metal = metal_sampler(device);
    for kind in [
        Degradation::FloatSampleFilter,
        Degradation::HalfSampleFilter,
    ] {
        assert_eq!(announced.get(&kind), None, "{announced:?}");
    }
    let ycbcr = announced.get(&Degradation::YCbCrFilterNotExact);
    assert_eq!(
        ycbcr,
        ((metal && kept) || by_device).then_some(&1),
        "{announced:?}"
    );
    assert_eq!(
        announced.contains_key(&Degradation::DeviceSampling),
        !metal,
        "{announced:?}"
    );
}

/// A fragment function's linear sample of a `YCBCR8_420_2P` texture through a runtime sampler,
/// with a `float` and a `half` result, takes the pipeline a draw binding any other texture takes: the draw binding it makes no pipeline after the one that bound an
/// `RGBA16Float` texture of the same texels, and its samples are that texture's bit for bit (the
/// device's filter of them), within [`DEVICE_FILTER_BOUND`] of Metal's step. On a device that
/// samples natively the module's `device-sampling` is announced and no filter notice; on one that
/// samples through Metal's own sampler `ycbcr-filter-not-exact`, once, naming the slot and the
/// format, and no `device-sampling`. A device whose memory is the host's reads the planes in place
/// instead (`ycbcr_in_place.rs`): one pipeline more, its samples Metal's step exactly, announced by
/// no filter notice.
#[test]
fn a_ycbcr_slot_takes_the_devices_filter_in_the_pipeline_its_creation_made() {
    on_device(|device, messages| {
        let kernels = Kernels::new(device);
        let surface = buffer_with(device, &surface_bytes(9)).unwrap();
        let texture = ycbcr_texture(
            &surface,
            Some(facts(VIDEO, None)),
            TextureUsage::SHADER_READ,
        )
        .unwrap();
        let read = grid(&texels_of(device, &kernels.read, &texture, None));
        let d = TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, W, H, false);
        let twin = device.new_texture(&d).unwrap();
        let bytes: Vec<u8> = read
            .iter()
            .flat_map(|t| [t[0], t[1], t[2], 2048].map(half_of_step))
            .flat_map(u16::to_le_bytes)
            .collect();
        twin.replace_region(Region::new_2d(0, 0, W, H), 0, 0, &bytes, 8 * W, 0)
            .unwrap();
        let metal = device
            .compile_device()
            .profile()
            .enabled()
            .properties
            .contains(&VkProperty::MetalSampler);
        // A device whose memory is the host's reads the planes in place, Metal's filter exactly,
        // in the slot's in-place variant.
        let exact = super::ycbcr_in_place::reads_in_place(device);
        let linear = sampler(device, SamplerMinMagFilter::Linear);
        let library = device
            .new_library_with_air(&llvm_assemble_text(&fragments()).unwrap())
            .unwrap();
        for fragment in ["between_runtime_float", "between_runtime_half"] {
            let pipeline = between_state(device, &library, fragment);
            // Where the device samples through Metal's texture unit, the `half` function's
            // creation hands its predicted key's making to the pool after its general form's
            // (`pipeline/render/residual.rs`): settled first, so the count below is the draw's.
            super::pipelines_settled(device);
            let plain = drawn_with(device, &pipeline, (&twin, Some(&linear)), |_, _| {});
            let before = device.work_counts().variant_pipelines_made;
            let got = drawn_with(device, &pipeline, (&texture, Some(&linear)), |_, _| {});
            let made = device.work_counts().variant_pipelines_made - before;
            assert_eq!(
                made,
                u64::from(exact),
                "{fragment}: pipelines the YCbCr draw made"
            );
            let mut worst = 0f32;
            for (i, texel) in (0..W * H).zip(&got) {
                let at = (i % W, i / W);
                for (c, &value) in texel.iter().enumerate().take(3) {
                    let want = metal_step(&read, at, BETWEEN, c) as f32 / 2048.0;
                    match exact {
                        true => assert_eq!(value, want, "pixel {at:?} of {fragment}"),
                        false => worst = worst.max((value - want).abs() * 255.0),
                    }
                }
                assert_eq!(texel[3], 1.0, "pixel {at:?} of {fragment}");
            }
            if !exact {
                println!(
                    "YCBCR-DEVICE-FILTER {} {fragment}: largest difference from Metal's step \
                     {worst} codes",
                    device.name()
                );
                assert!(
                    got == plain,
                    "{fragment}: the device's filter of the same texels"
                );
                assert!(worst <= DEVICE_FILTER_BOUND, "{fragment}: {worst} codes");
            }
        }
        let notices = messages.take_notices();
        let announced = counts(&notices);
        for kind in [
            Degradation::FloatSampleFilter,
            Degradation::HalfSampleFilter,
        ] {
            assert_eq!(announced.get(&kind), None, "{announced:?}");
        }
        let ycbcr = announced.get(&Degradation::YCbCrFilterNotExact);
        match metal && !exact {
            true => {
                assert_eq!(ycbcr, Some(&1), "{announced:?}");
                let notice = notices
                    .iter()
                    .filter_map(|m| m.degradation.as_ref())
                    .find(|n| n.degradation == Degradation::YCbCrFilterNotExact)
                    .unwrap();
                let named = format!("texture 0 bound as {:?}", PixelFormat::YCBCR84202P);
                assert!(notice.reason.contains(&named), "{notice:?}");
            }
            false => assert_eq!(ycbcr, None, "{announced:?}"),
        }
        assert_eq!(
            announced.contains_key(&Degradation::DeviceSampling),
            !metal,
            "{announced:?}"
        );
    });
}

/// The kind a filtered sample of a YCbCr texture that does not take Metal's exact filter takes
/// ([`a_ycbcr_sample_the_exact_filter_does_not_reach_is_announced`] on a device that samples
/// through Metal's own sampler, [`a_ycbcr_sample_through_an_unmeasured_filter_is_announced`] on
/// every device).
pub(super) const YCBCR_FILTERS: [Degradation; 1] = [Degradation::YCbCrFilterNotExact];

/// A linear sample of a YCbCr texture made with a swizzle that moves its channels, `(B, G, R,
/// A)`. On a device that samples through Metal's own sampler (`VkProperty::MetalSampler`) the
/// view composes the swizzle and the class drops the field the emulation of Metal's filter reads
/// (`TexelFormat::swizzled`): the sample takes the device's filter, Metal's own of the converted
/// texels, the class measured by no family keeping the device's sample, bit for bit an `RGBA16Float` texture's of the same texels under the same swizzle, and
/// announces [`YCBCR_FILTERS`] once, naming the slot and the format, in place of the generic filter
/// notes, which it does not announce. Elsewhere the sample is the device's own filter of the
/// converted texels (announced `device-sampling`), whose differences from Metal's step of each
/// channel the swizzle reads (Metal's swizzle permutes the texel it filters) are printed, and no
/// YCbCr filter notice is announced.
#[test]
fn a_ycbcr_sample_the_exact_filter_does_not_reach_is_announced() {
    on_device(|device, messages| {
        let kernels = Kernels::new(device);
        let surface = buffer_with(device, &surface_bytes(9)).unwrap();
        let swizzle = TextureSwizzleChannels::new(
            TextureSwizzle::Blue,
            TextureSwizzle::Green,
            TextureSwizzle::Red,
            TextureSwizzle::Alpha,
        );
        let mut d = TextureDescriptor::texture_2d(PixelFormat::YCBCR84202P, W, H, false);
        d.usage = TextureUsage::SHADER_READ;
        d.resource_options = ResourceOptions::shared();
        d.swizzle = swizzle;
        let plane = IoSurfacePlane::new(0, ROW, true).with_ycbcr(facts(VIDEO, None));
        let texture = surface.new_texture_with_iosurface_plane(&d, plane).unwrap();
        let linear = sampler(device, SamplerMinMagFilter::Linear);
        let sampled = texels_of(device, &kernels.between, &texture, Some(&linear));
        let metal = device
            .compile_device()
            .profile()
            .enabled()
            .properties
            .contains(&VkProperty::MetalSampler);
        if metal {
            let plain = ycbcr_texture(
                &surface,
                Some(facts(VIDEO, None)),
                TextureUsage::SHADER_READ,
            )
            .unwrap();
            let read = grid(&texels_of(device, &kernels.read, &plain, None));
            let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, W, H, false);
            d.usage = TextureUsage::SHADER_READ;
            d.swizzle = swizzle;
            let twin = device.new_texture(&d).unwrap();
            let bytes: Vec<u8> = read
                .iter()
                .flat_map(|t| [t[0], t[1], t[2], 2048].map(half_of_step))
                .flat_map(u16::to_le_bytes)
                .collect();
            twin.replace_region(Region::new_2d(0, 0, W, H), 0, 0, &bytes, 8 * W, 0)
                .unwrap();
            let want = texels_of(device, &kernels.between, &twin, Some(&linear));
            assert!(
                sampled == want,
                "the device's filter of the converted texels"
            );
        } else {
            let plain = ycbcr_texture(
                &surface,
                Some(facts(VIDEO, None)),
                TextureUsage::SHADER_READ,
            )
            .unwrap();
            let read = grid(&texels_of(device, &kernels.read, &plain, None));
            // The swizzle's source channel of each color channel: blue, green, red.
            let from = [2, 1, 0];
            let (mut compared, mut differ) = (0usize, Vec::new());
            for (i, texel) in (0u64..).zip(sampled.as_chunks::<16>().0.iter()) {
                let at = (i % W, i / W);
                let got =
                    |c: usize| f32::from_le_bytes(texel[4 * c..4 * c + 4].try_into().unwrap());
                let want: [f32; 4] = std::array::from_fn(|c| match c {
                    3 => 1.0,
                    c => metal_step(&read, at, BETWEEN, from[c]) as f32 / 2048.0,
                });
                for (c, want) in want.into_iter().enumerate() {
                    compared += 1;
                    if got(c).to_bits() != want.to_bits() {
                        differ.push(format!("texel {at:?} channel {c}: {} for {want}", got(c)));
                    }
                }
            }
            println!(
                "YCBCR-SWIZZLED {}: {compared} values compared with Metal's step swizzled, {} \
                 differ, the device's filter {:?}",
                device.name(),
                differ.len(),
                &differ[..differ.len().min(16)]
            );
        }
        let notices = messages.take_notices();
        let counts = counts(&notices);
        for kind in [
            Degradation::FloatSampleFilter,
            Degradation::HalfSampleFilter,
        ] {
            assert_eq!(counts.get(&kind), None, "{counts:?}");
        }
        if !metal {
            assert_eq!(counts.get(&YCBCR_FILTERS[0]), None, "{counts:?}");
            assert!(
                counts.contains_key(&Degradation::DeviceSampling),
                "{counts:?}"
            );
            return;
        }
        assert_eq!(counts.get(&YCBCR_FILTERS[0]), Some(&1), "{counts:?}");
        let notice = notices
            .iter()
            .filter_map(|m| m.degradation.as_ref())
            .find(|n| n.degradation == YCBCR_FILTERS[0])
            .unwrap();
        let named = format!("texture 0 bound as {:?}", PixelFormat::YCBCR84202P);
        assert!(notice.reason.contains(&named), "{notice:?}");
    });
}

/// A filtered sample of a YCbCr texture through a sampler Metal's exact filter is not measured
/// for, which the device filters, is announced, never silent: through a runtime sampler slot an
/// anisotropic one (16) and one whose min and mag filters differ announce [`YCBCR_FILTERS`] once,
/// naming the slot and the format, and no generic filter notice; through a constexpr sampler an
/// anisotropic one, one with a linear mip filter and one whose filters differ announce the
/// generic `float-sample-filter` once (the constexpr note holds whatever the slot holds, so it
/// names no format). Those notes are of a device that samples through Metal's own sampler
/// (`VkProperty::MetalSampler`); elsewhere every case is the device's own filter, announced
/// `device-sampling` once and none of those. Each case runs on a device of its own. On a device
/// that reads the planes in place (`ycbcr_in_place.rs`) the constexpr linear mip filter is read
/// there, Metal's filter of the texture's one level whatever the mip filter, as a runtime sampler's
/// is everywhere: Metal's step, with no YCbCr filter notice (the function's generic note, of its
/// sample of any other texture, stays where the device emulates Metal's sampler).
#[test]
fn a_ycbcr_sample_through_an_unmeasured_filter_is_announced() {
    let anisotropic = |d: &mut SamplerDescriptor| d.max_anisotropy = 16;
    let min_nearest = |d: &mut SamplerDescriptor| d.min_filter = SamplerMinMagFilter::Nearest;
    let mag_nearest = |d: &mut SamplerDescriptor| d.mag_filter = SamplerMinMagFilter::Nearest;
    type Adjust = fn(&mut SamplerDescriptor);
    let runtime: [(&str, Adjust); 3] = [
        ("anisotropic", anisotropic),
        ("min nearest", min_nearest),
        ("mag nearest", mag_nearest),
    ];
    // The constexpr words: `LINEAR_EDGE` with a max anisotropy of 16 (bits 20 to 23 hold it
    // less 1), a linear mip filter (bits 13 and 14), a nearest min filter (bits 11 and 12), or a
    // nearest mag filter (bits 9 and 10).
    let constexpr: [(&str, u64); 4] = [
        ("anisotropic", LINEAR_EDGE | 15 << 20),
        ("linear mip filter", LINEAR_EDGE | 2 << 13),
        ("min nearest", LINEAR_EDGE & !(3 << 11)),
        ("mag nearest", LINEAR_EDGE & !(3 << 9)),
    ];
    let cases = runtime
        .into_iter()
        .map(|(name, adjust)| (name, Through::Runtime, Some(adjust)))
        .chain(
            constexpr
                .into_iter()
                .map(|(name, words)| (name, Through::Constexpr(words), None)),
        );
    for (name, through, adjust) in cases {
        on_device(|device, messages| {
            let surface = buffer_with(device, &surface_bytes(9)).unwrap();
            let texture = ycbcr_texture(
                &surface,
                Some(facts(VIDEO, None)),
                TextureUsage::SHADER_READ,
            )
            .unwrap();
            let sampler = adjust.map(|adjust| {
                let mut d = SamplerDescriptor::new();
                d.min_filter = SamplerMinMagFilter::Linear;
                d.mag_filter = SamplerMinMagFilter::Linear;
                adjust(&mut d);
                device.new_sampler_state(&d).unwrap()
            });
            let at = |p: u64, f: u32, n: u64| (p as f32 + 0.5 + f as f32 / 256.0) / n as f32;
            let jobs: Vec<[f32; 4]> = (0..W * H)
                .map(|i| [at(i % W, BETWEEN[0], W), at(i / W, BETWEEN[1], H), 0.0, 0.0])
                .collect();
            let source = argument_linear_samples::kernel(false, through, false, false);
            let state = pipeline_of(device, &llvm_assemble_text(&source).unwrap(), "k");
            let got = run_held(
                device,
                (&texture, Held::Slot, sampler.as_ref()),
                &state,
                &jobs,
                1,
            );
            let notices = messages.take_notices();
            let counts = counts(&notices);
            let case = format!("{name} through {through:?}: {counts:?}");
            let in_place = super::ycbcr_in_place::reads_in_place(device);
            if in_place && name == "linear mip filter" {
                // The function's own note, of the sample it makes of any other texture, stays
                // where the device emulates Metal's sampler; no YCbCr filter is announced.
                assert_eq!(counts.get(&YCBCR_FILTERS[0]), None, "{case}");
                let model = modelled(&surface_bytes(9), YCbCrRange::Video, YCbCrMatrix::Bt601);
                for (i, texel) in (0..W * H).zip(got.as_chunks::<4>().0.iter()) {
                    let at = (i % W, i / W);
                    let want: [f32; 4] = std::array::from_fn(|c| match c {
                        3 => 1.0,
                        c => metal_step(&model, at, BETWEEN, c) as f32 / 2048.0,
                    });
                    assert_eq!(*texel, want, "{case}: texel {at:?}");
                }
                return;
            }
            if !metal_sampler(device) {
                assert_eq!(counts.get(&Degradation::DeviceSampling), Some(&1), "{case}");
                for kind in [
                    YCBCR_FILTERS[0],
                    Degradation::FloatSampleFilter,
                    Degradation::HalfSampleFilter,
                ] {
                    assert_eq!(counts.get(&kind), None, "{case}");
                }
                return;
            }
            match through {
                Through::Runtime => {
                    assert_eq!(counts.get(&YCBCR_FILTERS[0]), Some(&1), "{case}");
                    for kind in [
                        Degradation::FloatSampleFilter,
                        Degradation::HalfSampleFilter,
                    ] {
                        assert_eq!(counts.get(&kind), None, "{case}");
                    }
                    let notice = notices
                        .iter()
                        .filter_map(|m| m.degradation.as_ref())
                        .find(|n| n.degradation == YCBCR_FILTERS[0])
                        .unwrap();
                    let named = format!("texture 0 bound as {:?}", PixelFormat::YCBCR84202P);
                    assert!(notice.reason.contains(&named), "{case}: {notice:?}");
                }
                _ => {
                    assert_eq!(
                        counts.get(&Degradation::FloatSampleFilter),
                        Some(&1),
                        "{case}"
                    );
                }
            }
        });
    }
}

/// A function that samples textures through handles (an argument buffer's) reads its sampler
/// from memory, where no slot's specialization can give it Metal's exact filter of a YCbCr
/// texture it reaches, nor the texture's address-mode map: a command whose function samples
/// handles and reaches a YCbCr texture, held by the argument buffer and made resident, is
/// refused, typed, naming the texture's map (`MappedTextureSampling { through: Handles }`), whether
/// the sample's sampler is a constexpr one, a runtime slot's or the argument buffer's, and nothing
/// is recorded.
#[test]
fn a_ycbcr_texture_sampled_through_handles_is_refused() {
    on_device(|device, _| {
        let surface = buffer_with(device, &surface_bytes(9)).unwrap();
        let texture = ycbcr_texture(
            &surface,
            Some(facts(VIDEO, None)),
            TextureUsage::SHADER_READ,
        )
        .unwrap();
        assert!(!texture.address_mode_map().is_identity());
        let linear = sampler(device, SamplerMinMagFilter::Linear);
        let input = buffer_with(device, &[0u8; 16 * 8]).unwrap();
        let out = device
            .new_buffer(16 * 8, ResourceOptions::shared())
            .unwrap();
        for through in [
            Through::Constexpr(LINEAR_EDGE),
            Through::Runtime,
            Through::Argument,
        ] {
            let source = argument_linear_samples::kernel(true, through, false, false);
            let state = pipeline_of(device, &llvm_assemble_text(&source).unwrap(), "k");
            let held = match through {
                Through::Argument => linear.gpu_resource_id().0,
                _ => 0,
            };
            let ids = [texture.gpu_resource_id().0, held];
            let words: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
            let argument = buffer_with(device, &words).unwrap();
            let queue = device.new_command_queue().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder.set_compute_pipeline_state(&state).unwrap();
            encoder
                .use_resource(texture.as_resource(), ResourceUsage::READ)
                .unwrap();
            encoder.set_buffer(Some(&input), 0, 0).unwrap();
            encoder.set_buffer(Some(&out), 0, 1).unwrap();
            encoder.set_buffer(Some(&argument), 0, 2).unwrap();
            if let Through::Runtime = through {
                encoder.set_sampler_state(Some(&linear), 0).unwrap();
            }
            let dispatched = encoder.dispatch_threads(Size::new(8, 1, 1), Size::new(8, 1, 1));
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(
                dispatched,
                Err(Error::NotImplemented(RuntimeGap::MappedTextureSampling {
                    map: texture.address_mode_map(),
                    through: MappedSampling::Handles,
                })),
                "{through:?}"
            );
            assert!(command_buffer.executions().is_empty(), "{through:?}");
        }
    });
}

/// The bytes of a surface whose luma bytes are all `y` and chroma pairs all `(cb, cr)`.
fn uniform(y: u8, cb: u8, cr: u8) -> Vec<u8> {
    let mut bytes = vec![0xa5; LENGTH as usize];
    for row in 0..H {
        let at = (row * ROW) as usize;
        bytes[at..at + W as usize].fill(y);
    }
    for row in 0..CH {
        for x in 0..CW {
            let at = (CHROMA_AT + row * ROW + 2 * x) as usize;
            bytes[at] = cb;
            bytes[at + 1] = cr;
        }
    }
    bytes
}

/// How a round's planes are written before the texture is read.
#[derive(Clone, Copy, Debug)]
enum Writer {
    /// Kernels writing the `R8Unorm` and `RG8Unorm` textures over the planes, in a command
    /// buffer committed before the reader's.
    EarlierCommandBuffer,
    /// The same kernels in encoders before the reader's, in its command buffer.
    EarlierEncoder,
    /// The same kernels as dispatches before the reader's, in its encoder.
    EarlierDispatch,
    /// Blits from a buffer into the two textures, before the reader's encoder.
    Blit,
    /// The CPU, between command buffers.
    Cpu,
}

/// The texture follows each way its planes' bytes are written (the module docs): every round
/// writes new codes, and the texture then reads their conversion everywhere.
#[test]
fn ycbcr_textures_follow_writes_of_their_planes() {
    on_device(|device, messages| {
        let kernels = Kernels::new(device);
        let surface = buffer_with(device, &uniform(0, 0, 0)).unwrap();
        let texture = ycbcr_texture(
            &surface,
            Some(facts(VIDEO, None)),
            TextureUsage::SHADER_READ,
        )
        .unwrap();
        let luma = plane_texture(&surface, PixelFormat::R8Unorm, [W, H], 0);
        let chroma = plane_texture(&surface, PixelFormat::RG8Unorm, [CW, CH], CHROMA_AT);
        let queue = device.new_command_queue().unwrap();
        let out = device
            .new_buffer(16 * W * H, ResourceOptions::shared())
            .unwrap();
        let writers = [
            Writer::EarlierCommandBuffer,
            Writer::EarlierEncoder,
            Writer::EarlierDispatch,
            Writer::Blit,
            Writer::Cpu,
        ];
        // Before any write: the zeroed planes' texels.
        let first = grid(&texels_of(device, &kernels.read, &texture, None));
        let zero = ycbcr_texel(YCbCrRange::Video, YCbCrMatrix::Bt601, 0, 0, 0);
        assert!(first.iter().all(|t| *t == zero));
        for (round, writer) in writers.into_iter().enumerate() {
            let k = round as u8;
            let (y, cb, cr) = (40 + 37 * k, 200 - 23 * k, 60 + 31 * k);
            let code = |c: u8| f32::from(c) / 255.0;
            let fill = |texture: &Texture, value: [f32; 4]| {
                let bytes: Vec<u8> = value.iter().flat_map(|v| v.to_le_bytes()).collect();
                (texture.clone(), buffer_with(device, &bytes).unwrap())
            };
            let fills = [
                (fill(&luma, [code(y), 0.0, 0.0, 1.0]), [W, H]),
                (fill(&chroma, [code(cb), code(cr), 0.0, 1.0]), [CW, CH]),
            ];
            let command_buffer = queue.command_buffer().unwrap();
            let write_kernels = |command_buffer: &CommandBuffer| {
                for ((texture, value), size) in &fills {
                    dispatch(command_buffer, &kernels.fill, texture, value, None, *size);
                }
            };
            match writer {
                Writer::EarlierCommandBuffer => {
                    let earlier = queue.command_buffer().unwrap();
                    write_kernels(&earlier);
                    earlier.commit().unwrap();
                }
                Writer::EarlierEncoder => write_kernels(&command_buffer),
                Writer::EarlierDispatch => {}
                Writer::Blit => {
                    let source = buffer_with(device, &uniform(y, cb, cr)).unwrap();
                    let mut blit = command_buffer.blit_command_encoder().unwrap();
                    for (texture, offset, size) in
                        [(&luma, 0, [W, H]), (&chroma, CHROMA_AT, [CW, CH])]
                    {
                        blit.copy_from_buffer_to_texture(
                            &source,
                            offset,
                            ROW,
                            0,
                            Size::new(size[0], size[1], 1),
                            texture,
                            0,
                            0,
                            Origin::new(0, 0, 0),
                            BlitOption::None,
                        )
                        .unwrap();
                    }
                    blit.end_encoding().unwrap();
                }
                Writer::Cpu => {
                    let bytes = uniform(y, cb, cr);
                    let contents = surface.contents().unwrap();
                    // SAFETY: a Shared buffer of `LENGTH` bytes no command uses now (every
                    // round waits for its command buffer).
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            bytes.as_ptr(),
                            contents.as_ptr(),
                            LENGTH as usize,
                        )
                    };
                }
            }
            {
                let mut encoder = command_buffer.compute_command_encoder().unwrap();
                if matches!(writer, Writer::EarlierDispatch) {
                    for ((texture, value), size) in &fills {
                        encoder.set_compute_pipeline_state(&kernels.fill).unwrap();
                        encoder.set_texture(Some(texture), 0).unwrap();
                        encoder.set_buffer(Some(value), 0, 0).unwrap();
                        encoder
                            .dispatch_threads(Size::new(size[0], size[1], 1), Size::new(8, 8, 1))
                            .unwrap();
                    }
                }
                encoder.set_compute_pipeline_state(&kernels.read).unwrap();
                encoder.set_texture(Some(&texture), 0).unwrap();
                encoder.set_buffer(Some(&out), 0, 0).unwrap();
                encoder
                    .dispatch_threads(Size::new(W, H, 1), Size::new(8, 8, 1))
                    .unwrap();
                encoder.end_encoding().unwrap();
            }
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            let want = ycbcr_texel(YCbCrRange::Video, YCbCrMatrix::Bt601, y, cb, cr);
            let got = grid(&read_buffer(&out, 0, 16 * W * H).unwrap());
            let stale = got.iter().filter(|t| **t != want).count();
            assert_eq!(stale, 0, "{writer:?}: {stale} texels not the new codes'");
        }
        let notices = counts(&messages.take_notices());
        assert_eq!(
            notices.get(&Degradation::FloatSampleFilter),
            None,
            "{notices:?}"
        );
    });
}

/// A draw samples the planes as written before it (the render pass twin of
/// [`ycbcr_textures_follow_writes_of_their_planes`]; `shadow.rs`'s render passes): a fragment
/// function's linear sample of a `YCBCR8_420_2P` texture over a Shared buffer, and over a no-copy
/// buffer of the caller's memory, reads the conversion of codes written by kernels in an earlier
/// command buffer and in an earlier encoder of its own, by a blit, and by the CPU before the
/// commit. The planes hold one code each, so every sample is that texel.
#[test]
fn a_draw_samples_the_planes_as_written_before_it() {
    on_device(|device, messages| {
        let kernels = Kernels::new(device);
        let library = device
            .new_library_with_air(&llvm_assemble_text(&fragments()).unwrap())
            .unwrap();
        for no_copy in [false, true] {
            let zeroed = uniform(0, 0, 0);
            let surface = match no_copy {
                false => buffer_with(device, &zeroed).unwrap(),
                true => {
                    let len = host_page().max(LENGTH as usize);
                    let mapping = Mapping::anonymous(len);
                    mapping.write(0, &zeroed);
                    let pointer = mapping.pointer();
                    let release: Deallocator = Box::new(move |_, _| drop(mapping));
                    // SAFETY: the mapping stays mapped until the deallocator, which owns it, runs.
                    let made = unsafe {
                        host_memory::no_copy(
                            device,
                            pointer,
                            len,
                            ResourceOptions::default(),
                            Some(release),
                        )
                    };
                    match made {
                        Ok(buffer) => buffer,
                        Err(error) => {
                            assert!(host_memory::not_importable(&error), "{error:?}");
                            continue;
                        }
                    }
                }
            };
            let texture = ycbcr_texture(
                &surface,
                Some(facts(VIDEO, None)),
                TextureUsage::SHADER_READ,
            )
            .unwrap();
            let luma = plane_texture(&surface, PixelFormat::R8Unorm, [W, H], 0);
            let chroma = plane_texture(&surface, PixelFormat::RG8Unorm, [CW, CH], CHROMA_AT);
            for (round, writer) in [
                Writer::EarlierCommandBuffer,
                Writer::EarlierEncoder,
                Writer::Blit,
                Writer::Cpu,
            ]
            .into_iter()
            .enumerate()
            {
                let k = round as u8 + 4 * u8::from(no_copy);
                let (y, cb, cr) = (40 + 29 * k, 200 - 17 * k, 60 + 23 * k);
                let code = |c: u8| f32::from(c) / 255.0;
                let fill = |texture: &Texture, value: [f32; 4]| {
                    let bytes: Vec<u8> = value.iter().flat_map(|v| v.to_le_bytes()).collect();
                    (texture.clone(), buffer_with(device, &bytes).unwrap())
                };
                let fills = [
                    (fill(&luma, [code(y), 0.0, 0.0, 1.0]), [W, H]),
                    (fill(&chroma, [code(cb), code(cr), 0.0, 1.0]), [CW, CH]),
                ];
                let write_kernels = |command_buffer: &CommandBuffer| {
                    for ((texture, value), size) in &fills {
                        dispatch(command_buffer, &kernels.fill, texture, value, None, *size);
                    }
                };
                let source = buffer_with(device, &uniform(y, cb, cr)).unwrap();
                let before = |queue: &CommandQueue, command_buffer: &CommandBuffer| match writer {
                    Writer::EarlierCommandBuffer => {
                        let earlier = queue.command_buffer().unwrap();
                        write_kernels(&earlier);
                        earlier.commit().unwrap();
                    }
                    Writer::EarlierEncoder => write_kernels(command_buffer),
                    Writer::Blit => {
                        let mut blit = command_buffer.blit_command_encoder().unwrap();
                        for (texture, offset, size) in
                            [(&luma, 0, [W, H]), (&chroma, CHROMA_AT, [CW, CH])]
                        {
                            blit.copy_from_buffer_to_texture(
                                &source,
                                offset,
                                ROW,
                                0,
                                Size::new(size[0], size[1], 1),
                                texture,
                                0,
                                0,
                                Origin::new(0, 0, 0),
                                BlitOption::None,
                            )
                            .unwrap();
                        }
                        blit.end_encoding().unwrap();
                    }
                    Writer::Cpu => {
                        let bytes = uniform(y, cb, cr);
                        let contents = surface.contents().unwrap();
                        // SAFETY: the surface's first `LENGTH` bytes, which no command uses now
                        // (every round waits for its command buffer).
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                bytes.as_ptr(),
                                contents.as_ptr(),
                                LENGTH as usize,
                            )
                        };
                    }
                    Writer::EarlierDispatch => unreachable!("a draw's round"),
                };
                let got = drawn_after(
                    device,
                    &library,
                    "between_constexpr_float",
                    (&texture, None),
                    before,
                );
                let want = ycbcr_texel(YCbCrRange::Video, YCbCrMatrix::Bt601, y, cb, cr);
                let want: [f32; 4] = std::array::from_fn(|c| match c {
                    3 => 1.0,
                    c => f32::from(want[c]) / 2048.0,
                });
                let stale = got.iter().filter(|t| **t != want).count();
                assert_eq!(
                    stale, 0,
                    "{writer:?}, no-copy {no_copy}: {stale} pixels not the new codes' ({:?})",
                    got[0]
                );
            }
        }
        let notices = counts(&messages.take_notices());
        assert_eq!(
            notices.get(&Degradation::YCbCrFilterNotExact),
            None,
            "{notices:?}"
        );
    });
}

/// A kernel's writes to the texture are dropped, as Metal drops them: both planes keep their
/// bytes. `get_bytes` reads the luma rows.
#[test]
fn ycbcr_texture_writes_are_dropped_and_its_bytes_are_luma() {
    on_device(|device, _| {
        let kernels = Kernels::new(device);
        let bytes = surface_bytes(11);
        let surface = buffer_with(device, &bytes).unwrap();
        let usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        let texture = ycbcr_texture(&surface, Some(facts(VIDEO, None)), usage).unwrap();
        let value = buffer_with(device, &[0x3f; 16]).unwrap();
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        dispatch(
            &command_buffer,
            &kernels.fill,
            &texture,
            &value,
            None,
            [W, H],
        );
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(read_buffer(&surface, 0, LENGTH).unwrap(), bytes);
        let mut luma = vec![0; (W * H) as usize];
        texture
            .get_bytes(&mut luma, W, 0, Region::new_2d(0, 0, W, H), 0, 0)
            .unwrap();
        let want: Vec<u8> = (0..H)
            .flat_map(|y| bytes[(y * ROW) as usize..(y * ROW + W) as usize].to_vec())
            .collect();
        assert_eq!(luma, want);
    });
}

/// The texture of `pf` over plane 0 of `surface`, as [`family_texture`] makes it, of shader read
/// and write usage.
fn written_texture(
    surface: &Buffer,
    pf: PixelFormat,
    ostype: &str,
    matrix: Option<SurfaceYCbCrMatrix>,
) -> Result<Texture, Error> {
    let mut d = TextureDescriptor::texture_2d(pf, W, H, false);
    d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    d.resource_options = ResourceOptions::shared();
    let code: [u8; 4] = ostype.as_bytes().try_into().unwrap();
    let facts = IoSurfaceYCbCr::new(u32::from_be_bytes(code), 0, [W, H], CHROMAQ_AT, ROWQ);
    let facts = match matrix {
        Some(matrix) => facts.with_matrix(matrix),
        None => facts,
    };
    let at = IoSurfacePlane::new(0, ROWQ, true).with_ycbcr(facts);
    surface.new_texture_with_iosurface_plane(&d, at)
}

/// A 32-bit word of `random`.
fn random_word(random: &mut Random) -> u32 {
    u32::from_le_bytes(std::array::from_fn(|_| random.next()))
}

/// A `float` channel a test writes, of `random`: on the grid of 1/2048, between two of its steps,
/// any multiple of 2^-24 in `[0, 1)`, one out of `[0, 1]`, or a NaN, an infinity, a zero, a
/// subnormal, 1 or a value beside it, or a huge one.
fn written_channel(random: &mut Random) -> f32 {
    const SPECIALS: [f32; 12] = [
        f32::NAN,
        f32::NEG_INFINITY,
        f32::INFINITY,
        -0.0,
        0.0,
        1.0,
        1.000_000_1,
        0.999_999_94,
        1.0e-45,
        1.0e30,
        -1.0e30,
        0.999_755_86,
    ];
    let unit = |random: &mut Random| (random_word(random) % (1 << 24)) as f32 / (1 << 24) as f32;
    match random.next() % 8 {
        0..=2 => (random_word(random) % 2049) as f32 / 2048.0,
        3 => ((random_word(random) % 2048) as f32 + 0.5) / 2048.0,
        4 => unit(random),
        5 => unit(random) * 4.0 - 1.5,
        6 => SPECIALS[random_word(random) as usize % SPECIALS.len()],
        _ => (random_word(random) % 1024) as f32 / 1023.0,
    }
}

/// A `half` channel's bits a test writes, of `random`: one of `[0, 1]`, any bits (NaNs,
/// infinities, negative values, values past 1 and subnormals among them), or a NaN, an
/// infinity, a zero, a subnormal, or 1 or a value beside it.
fn written_half(random: &mut Random) -> u16 {
    const SPECIALS: [u16; 9] = [
        0x7e00, 0xfe00, 0x7c00, 0xfc00, 0x8000, 0x0001, 0x3bff, 0x3c00, 0x3c01,
    ];
    let word = random_word(random);
    match random.next() % 6 {
        0..=3 => (word % 0x3c01) as u16,
        4 => word as u16,
        _ => SPECIALS[word as usize % SPECIALS.len()],
    }
}

/// The value of `half` bits `h`, widened exactly.
fn half_value(h: u16) -> f32 {
    let sign = if h & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from((h >> 10) & 0x1f);
    let fraction = f32::from(h & 0x3ff);
    match exponent {
        0 => sign * fraction * 2f32.powi(-24),
        31 if fraction == 0.0 => sign * f32::INFINITY,
        31 => f32::NAN,
        e => sign * (1.0 + fraction / 1024.0) * 2f32.powi(e - 15),
    }
}

/// `bytes` (a surface of `codes` laid out as [`family_bytes`] lays it) after writes of `texels`
/// (each texel's `[r, g, b]`, row by row, `None` where unwritten) as Metal stores them over
/// `conversion` (`metal2vulkan::bind::ycbcr_write`): each written texel's luma code at its place
/// in its row of the luma plane, and its Cb and Cr codes at its pair in its row of the chroma
/// plane, a code a byte or the high bits of a little-endian 16-bit word; every other byte as it
/// was.
fn written_bytes(
    bytes: &[u8],
    codes: YCbCrCodes,
    conversion: (YCbCrRange, YCbCrMatrix),
    texels: &[Option<[f32; 3]>],
) -> Vec<u8> {
    let mut out = bytes.to_vec();
    let unit = u64::from(codes.bytes);
    let mut put = |at: u64, code: u16| {
        let at = at as usize;
        match codes.bytes {
            1 => out[at] = code as u8,
            _ => out[at..at + 2].copy_from_slice(&(code << (16 - codes.bits)).to_le_bytes()),
        }
    };
    for (i, texel) in texels.iter().enumerate() {
        let Some(texel) = texel else {
            continue;
        };
        let (x, y) = (i as u64 % W, i as u64 / W);
        let [luma, cb, cr] = ycbcr_write(codes, conversion, *texel).expect("a stored layout");
        put(y * ROWQ + x * unit, luma);
        let pair = CHROMAQ_AT + y * ROWQ + 2 * x * unit;
        put(pair, cb);
        put(pair + unit, cr);
    }
    out
}

/// The first byte where `after` and `want` differ, and both there, for a failure's message.
fn first_difference(after: &[u8], want: &[u8]) -> Option<(usize, u8, u8)> {
    after
        .iter()
        .zip(want)
        .enumerate()
        .find(|(_, (a, w))| a != w)
        .map(|(i, (&a, &w))| (i, a, w))
}

/// Dispatch a masked write of `values` to `texture` (`half4`s where `half`, else `float4`s)
/// over the texels `mask` names, and then, in the same encoder, a `read()` of every texel into
/// `out`, in one command buffer of `queue`.
fn write_then_read(
    kernels: &Kernels,
    queue: &CommandQueue,
    texture: &Texture,
    (values, mask, half): (&Buffer, &Buffer, bool),
    out: &Buffer,
) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder
        .set_compute_pipeline_state(&kernels.write_masked[usize::from(half)])
        .unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_buffer(Some(values), 0, 0).unwrap();
    encoder.set_buffer(Some(mask), 0, 1).unwrap();
    encoder
        .dispatch_threads(Size::new(W, H, 1), Size::new(8, 8, 1))
        .unwrap();
    encoder.set_compute_pipeline_state(&kernels.read).unwrap();
    encoder.set_buffer(Some(out), 0, 0).unwrap();
    encoder
        .dispatch_threads(Size::new(W, H, 1), Size::new(8, 8, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
}

/// Metal stores a function's writes to `YCBCR8_444_2P`, `YCBCR10_444_2P`, `RGB8_444_2P` and
/// `RGB10_444_2P` into both planes of the surface (on an Apple M4, on an Apple M3 Max, byte for
/// byte alike): each written texel's luma code and its Cb and Cr pair, the codes
/// `metal2vulkan::bind::ycbcr_write` states, and every other byte kept. Here a dispatch writes
/// `float` and `half` values (on the code grid and off it, out of `[0, 1]`, NaNs, infinities) to
/// about two texels in three, over both ranges and each matrix: the surface's bytes are those codes
/// and otherwise as they were, the rows' padding and the bytes past the planes among them. A
/// `read()` of the texture in the same encoder after the writes, and one in a later command buffer,
/// read the planes as written (where the device converts the texture, its conversion of them:
/// `ycbcr_device_reads`). A dispatch writing no texel keeps every byte. Nothing is announced
/// past the lane's pipeline states' own notes, on a device without unified memory the first
/// `Shared` buffer's `shared-in-system-memory`, and where the device converts a texture itself
/// `ycbcr-filter-not-exact`.
#[test]
fn a_four_four_four_texture_s_writes_store_metal_s_codes() {
    use PixelFormat as P;
    use YCbCrMatrix::{Bt2020, Bt601, Bt709};
    use YCbCrRange::{Full, Video};
    on_device(|device, messages| {
        let kernels = Kernels::new(device);
        // The lane's pipeline states announce their own notes as they are made (a `half` read's
        // NaN encoding, `read_half`): the writes are to announce nothing past them.
        messages.take_notices();
        let queue = device.new_command_queue().unwrap();
        let (b709, b2020) = (
            Some(SurfaceYCbCrMatrix::Itu709),
            Some(SurfaceYCbCrMatrix::Itu2020),
        );
        let mut seed = 640;
        let mut by_devices = false;
        for (pf, ostype, matrix, conversion) in [
            (P::YCBCR84442P, "444v", None, (Video, Bt601)),
            (P::YCBCR84442P, "444f", b709, (Full, Bt709)),
            (P::YCBCR84442P, "444v", b2020, (Video, Bt2020)),
            (P::YCBCR104442P, "x444", b709, (Video, Bt709)),
            (P::YCBCR104442P, "xf44", None, (Full, Bt601)),
            (P::YCBCR104442P, "xf44", b2020, (Full, Bt2020)),
            (P::RGB84442P, "444f", None, (Full, Bt601)),
            (P::RGB104442P, "x444", b709, (Video, Bt709)),
        ] {
            let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            for half in [false, true] {
                seed += 1;
                let case = format!("{pf:?} over {ostype}, {matrix:?}, half {half}");
                let mut random = Random(seed);
                let bytes = family_bytes(seed, codes);
                let surface = buffer_with(device, &bytes).unwrap();
                let converted = device_conversions(device);
                let texture = written_texture(&surface, pf, ostype, matrix).unwrap();
                let by_device = device_conversions(device) > converted;
                by_devices |= by_device;
                let (mut values, mut mask, mut texels) = (Vec::new(), Vec::new(), Vec::new());
                for _ in 0..W * H {
                    let written = !random.next().is_multiple_of(3);
                    let texel = if half {
                        let h: [u16; 4] = std::array::from_fn(|_| written_half(&mut random));
                        values.extend(h.iter().flat_map(|h| h.to_le_bytes()));
                        [h[0], h[1], h[2]].map(half_value)
                    } else {
                        let v: [f32; 4] = std::array::from_fn(|_| written_channel(&mut random));
                        values.extend(v.iter().flat_map(|v| v.to_le_bytes()));
                        [v[0], v[1], v[2]]
                    };
                    mask.extend(u32::from(written).to_le_bytes());
                    texels.push(written.then_some(texel));
                }
                let values = buffer_with(device, &values).unwrap();
                let mask = buffer_with(device, &mask).unwrap();
                let out = device
                    .new_buffer(16 * W * H, ResourceOptions::shared())
                    .unwrap();
                write_then_read(&kernels, &queue, &texture, (&values, &mask, half), &out);
                let want = written_bytes(&bytes, codes, conversion, &texels);
                let after = read_buffer(&surface, 0, bytes.len() as u64).unwrap();
                assert!(
                    after == want,
                    "{case}: the surface's bytes after the writes, first difference (at, got, \
                     want) {:?}",
                    first_difference(&after, &want)
                );
                // The converted formats' reads, of the planes as written.
                if codes.output == YCbCrOutput::Converted {
                    let planes = (&want[..], codes, conversion);
                    let same = read_buffer(&out, 0, 16 * W * H).unwrap();
                    let later = texels_of(device, &kernels.read, &texture, None);
                    for (texels, when) in [(same, "in the same encoder"), (later, "later")] {
                        let case = format!("{case}: read() {when}");
                        assert_family_read(
                            &kernels,
                            &texels,
                            planes,
                            (2048, false, by_device),
                            &case,
                        );
                    }
                }
            }
        }
        // A dispatch writing no texel keeps every byte.
        for (seed, (pf, ostype)) in (680..).zip([
            (P::YCBCR84442P, "444f"),
            (P::YCBCR104442P, "x444"),
            (P::RGB84442P, "444v"),
            (P::RGB104442P, "xf44"),
        ]) {
            let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            let bytes = family_bytes(seed, codes);
            let surface = buffer_with(device, &bytes).unwrap();
            let texture = written_texture(&surface, pf, ostype, None).unwrap();
            let values = buffer_with(device, &vec![0x3f; (16 * W * H) as usize]).unwrap();
            let mask = buffer_with(device, &vec![0; (4 * W * H) as usize]).unwrap();
            let out = device
                .new_buffer(16 * W * H, ResourceOptions::shared())
                .unwrap();
            write_then_read(&kernels, &queue, &texture, (&values, &mask, false), &out);
            let after = read_buffer(&surface, 0, bytes.len() as u64).unwrap();
            assert!(
                after == bytes,
                "{pf:?} over {ostype}: a dispatch writing no texel changed byte (at, got, want) \
                 {:?}",
                first_difference(&after, &bytes)
            );
        }
        // Nothing but, on a device without unified memory, the first `Shared` buffer the case made
        // (`blits::SYSTEM_MEMORY`, announced as that buffer is made), as
        // `blits::shared_buffers_follow_the_memory_model` expects it, and the device's own
        // conversion where it converted a texture (`ycbcr_device_reads`).
        let mut expected: Vec<Degradation> = match device.properties().has_unified_memory {
            true => Vec::new(),
            false => SYSTEM_MEMORY.to_vec(),
        };
        if by_devices {
            expected.push(Degradation::YCbCrFilterNotExact);
        }
        let expected = &expected[..];
        let notices = assert_once(&messages.take_notices(), expected);
        assert!(
            notices.keys().all(|kind| expected.contains(kind)),
            "{notices:?}"
        );
    });
}

/// The 4:4:4 writes whose codes are not known stay refused, typed (`Feature::Format`), and leave
/// the surface's bytes as they were: the `_sRGB` twins' (no arithmetic found matches Metal's
/// codes) and a texture's over plane 1 of a 4:4:4 surface (not measured). A 4:2:2 texture over
/// its own surface drops them, as Metal does, and so does a packed 4:4:4 one, as the M3 Max
/// under macOS 27.0.1 does (`metal::format::writes_dropped`).
#[test]
fn four_four_four_writes_of_unknown_codes_are_refused_typed() {
    on_device(|device, _| {
        let kernels = Kernels::new(device);
        let value = buffer_with(device, &[0x3f; 16]).unwrap();
        let queue = device.new_command_queue().unwrap();
        for (seed, pf, ostype, plane, dropped) in [
            (141, PixelFormat::YCBCR84442PSRGB, "444v", 0, false),
            (142, PixelFormat::YCBCR104442PSRGB, "xf44", 0, false),
            (143, PixelFormat::YCBCR104442PPACKED, "p444", 0, true),
            (147, PixelFormat::YCBCR104442PPACKEDSRGB, "pf44", 0, true),
            (144, PixelFormat::YCBCR84442P, "444f", 1, false),
            (145, PixelFormat::RGB104442P, "x444", 1, false),
            (146, PixelFormat::YCBCR84222P, "422v", 0, true),
        ] {
            let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            let bytes = family_bytes(seed, codes);
            let surface = buffer_with(device, &bytes).unwrap();
            let code: [u8; 4] = ostype.as_bytes().try_into().unwrap();
            let offset = [0, CHROMAQ_AT][plane];
            let facts = IoSurfaceYCbCr::new(
                u32::from_be_bytes(code),
                plane as u64,
                [W, H],
                CHROMAQ_AT,
                ROWQ,
            );
            let mut d = TextureDescriptor::texture_2d(pf, W, H, false);
            d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
            d.resource_options = ResourceOptions::shared();
            let at = IoSurfacePlane::new(offset, ROWQ, true).with_ycbcr(facts);
            let texture = surface.new_texture_with_iosurface_plane(&d, at).unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder.set_compute_pipeline_state(&kernels.fill).unwrap();
            encoder.set_texture(Some(&texture), 0).unwrap();
            encoder.set_buffer(Some(&value), 0, 0).unwrap();
            let result = encoder.dispatch_threads(Size::new(W, H, 1), Size::new(8, 8, 1));
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            let case = format!("{pf:?} over plane {plane} of {ostype}");
            match (dropped, result) {
                (true, Ok(())) => {}
                (false, Err(Error::Unsupported(u) | Error::Bind(BindError::Unsupported(u)))) => {
                    assert_eq!(u.feature, Feature::Format, "{case}: {u:?}")
                }
                (_, other) => panic!("{case}: {other:?}"),
            }
            let after = read_buffer(&surface, 0, bytes.len() as u64).unwrap();
            assert!(after == bytes, "{case}: the surface's bytes changed");
        }
    });
}

/// Metal makes no `YCBCR8_420_2P` texture from a descriptor alone; the runtime refuses, typed, a
/// surface without its facts and plane 1 at other bytes than the facts name.
#[test]
fn ycbcr_textures_are_made_only_over_their_surfaces() {
    on_device(|device, _| {
        let mut d = TextureDescriptor::texture_2d(PixelFormat::YCBCR84202P, W, H, false);
        d.usage = TextureUsage::SHADER_READ;
        let nil = |call| {
            Error::Nil(Nil::new(
                call,
                Problem::Texture(TextureProblem::NoTextureOfFormat),
            ))
        };
        assert_eq!(device.new_texture(&d).err(), Some(nil("new_texture")));
        let size = device.heap_texture_size_and_align(&d).unwrap();
        assert_eq!((size.size, size.align), (0, 16));
        let surface = buffer_with(device, &surface_bytes(3)).unwrap();
        let unsupported = |result: Result<Texture, Error>| {
            assert!(
                matches!(result, Err(Error::Unsupported(_))),
                "{:?}",
                result.err()
            );
        };
        assert_eq!(
            surface.new_texture(&d, 0, ROW).err(),
            Some(nil("new_texture"))
        );
        unsupported(ycbcr_texture(&surface, None, TextureUsage::SHADER_READ));
        let plane_one = IoSurfaceYCbCr::new(VIDEO, 1, [W, H], CHROMA_AT, ROW);
        unsupported(ycbcr_texture(
            &surface,
            Some(plane_one),
            TextureUsage::SHADER_READ,
        ));
    });
}

/// Every two-plane YCbCr format answers as Metal does on an M4 without a surface: Metal's nil from
/// a descriptor alone, a 2D multisample texture or array stopped, and the heap query 0 bytes at 16,
/// or 128 for a 2D array; over a `420v` surface the runtime makes every one it holds, as Metal
/// makes it over any surface (`textures_over_any_surface_read_what_metal_reads`), and refuses,
/// typed, the rest.
#[test]
fn two_plane_ycbcr_formats_answer_as_metal_does() {
    on_device(|device, _| {
        let surface = buffer_with(device, &surface_bytes(3)).unwrap();
        let family = PixelFormat::ALL
            .iter()
            .copied()
            .filter(|&pf| ycbcr_format(pf).is_some());
        let mut seen = 0;
        for pf in family {
            seen += 1;
            let refused = |d: &TextureDescriptor, want: Error| {
                let made = device.new_texture(d).map(|_| ()).err();
                assert_eq!(made, Some(want), "{pf:?} {:?}", d.texture_type);
            };
            let stopped = Error::Invalid(Invalid::new(
                "new_texture",
                Problem::Texture(TextureProblem::TypeForFormat),
            ));
            let mut d = TextureDescriptor::texture_2d(pf, W, H, false);
            d.usage = TextureUsage::SHADER_READ;
            refused(
                &d,
                Error::Nil(Nil::new(
                    "new_texture",
                    Problem::Texture(TextureProblem::NoTextureOfFormat),
                )),
            );
            let size = device.heap_texture_size_and_align(&d).unwrap();
            assert_eq!((size.size, size.align), (0, 16), "{pf:?}");
            let mut array = d;
            array.texture_type = TextureType::Type2DArray;
            array.array_length = 2;
            let size = device.heap_texture_size_and_align(&array).unwrap();
            assert_eq!((size.size, size.align), (0, 128), "{pf:?}");
            let mut multisample = d;
            multisample.texture_type = TextureType::Type2DMultisample;
            multisample.sample_count = 4;
            refused(&multisample, stopped.clone());
            // A 2D multisample array too.
            multisample.texture_type = TextureType::Type2DMultisampleArray;
            multisample.array_length = 2;
            refused(&multisample, stopped);
            d.resource_options = ResourceOptions::shared();
            let wide = IoSurfaceYCbCr::new(VIDEO, 0, [W, H], CHROMA10_AT, ROW10);
            let plane = IoSurfacePlane::new(0, ROW10, true).with_ycbcr(wide);
            let made = surface.new_texture_with_iosurface_plane(&d, plane);
            if ycbcr_format(pf).and_then(YCbCrCodes::of).is_some() {
                assert!(made.is_ok(), "{pf:?}: {:?}", made.err());
            } else {
                assert!(
                    matches!(made, Err(Error::Unsupported(_))),
                    "{pf:?}: {:?}",
                    made.err()
                );
            }
        }
        assert_eq!(seen, 69);
    });
}

/// A 10-bit surface's rows: each code a little-endian 16-bit word, rows wider than the planes'
/// words (`2 W` bytes of luma, `4 CW` of chroma).
const ROW10: u64 = 128;
/// The 10-bit surface's chroma plane's first byte.
const CHROMA10_AT: u64 = ROW10 * H;
/// The 10-bit surface's buffer.
const LENGTH10: u64 = 8192;

/// The bytes of a 10-bit surface whose planes' words our generator at `seed` fills, laid out by
/// [`ROW10`] and [`CHROMA10_AT`], every other byte `0xa5`. Metal reads a word's high 10 bits.
fn ten_bit_bytes(seed: u64) -> Vec<u8> {
    let mut random = Random(seed);
    let mut bytes = vec![0xa5; LENGTH10 as usize];
    for y in 0..H {
        for x in 0..2 * W {
            bytes[(y * ROW10 + x) as usize] = random.next();
        }
    }
    for y in 0..CH {
        for x in 0..4 * CW {
            bytes[(CHROMA10_AT + y * ROW10 + x) as usize] = random.next();
        }
    }
    bytes
}

/// Each texel `codes` reads of the 10-bit surface `bytes` under `conversion`
/// (`metal2vulkan::bind::ycbcr_read`, the chroma filtered as [`modelled`] filters it).
fn ten_bit_modelled(
    bytes: &[u8],
    codes: YCbCrCodes,
    conversion: (YCbCrRange, YCbCrMatrix),
) -> Vec<[u16; 3]> {
    let code = |at: u64| u16::from_le_bytes([bytes[at as usize], bytes[at as usize + 1]]) >> 6;
    let taps = |q: u64, n: u64| -> [(u64, u32); 2] {
        let c = q / 2;
        if q % 2 == 1 {
            [(c, 3), ((c + 1).min(n - 1), 1)]
        } else {
            [(c.saturating_sub(1), 1), (c, 3)]
        }
    };
    let mut out = Vec::new();
    for y in 0..H {
        for x in 0..W {
            let mut sums = [0u32; 2];
            for (cy, wy) in taps(y, CH) {
                for (cx, wx) in taps(x, CW) {
                    let at = CHROMA10_AT + cy * ROW10 + 4 * cx;
                    sums[0] += wx * wy * u32::from(code(at));
                    sums[1] += wx * wy * u32::from(code(at + 2));
                }
            }
            let [cb, cr] = sums.map(|s| ((s + 8) >> 4) as u16);
            out.push(ycbcr_read(
                codes,
                conversion,
                [code(y * ROW10 + 2 * x), cb, cr],
            ));
        }
    }
    out
}

/// The texture of `pf` over plane `plane` of `surface`, a 10-bit surface of OSType `ostype` and
/// `matrix`.
fn ten_bit_texture(
    surface: &Buffer,
    pf: PixelFormat,
    (ostype, plane): (&[u8; 4], u64),
    matrix: Option<SurfaceYCbCrMatrix>,
) -> Result<Texture, Error> {
    let mut d = TextureDescriptor::texture_2d(pf, W, H, false);
    d.usage = TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::shared();
    let facts = IoSurfaceYCbCr::new(
        u32::from_be_bytes(*ostype),
        plane,
        [W, H],
        CHROMA10_AT,
        ROW10,
    );
    let facts = match matrix {
        Some(matrix) => facts.with_matrix(matrix),
        None => facts,
    };
    let at = IoSurfacePlane::new(0, ROW10, true).with_ycbcr(facts);
    surface.new_texture_with_iosurface_plane(&d, at)
}

/// The codes of each channel of `texels` (`float4`s), each checked to be `code / max` correctly
/// rounded (where the device converts the codes itself, within its conversion's bound:
/// `Kernels::exact`), alpha checked to be 1: an `RGB10_420_2P` (`max` 1023) or `RGB8_420_2P`
/// (255) read's red Cr, green Y, blue Cb.
fn code_grid(kernels: &Kernels, texels: &[u8], max: u16) -> Vec<[u16; 3]> {
    let max = f32::from(max);
    texels
        .as_chunks::<16>()
        .0
        .iter()
        .map(|t| {
            let c = |i: usize| f32::from_le_bytes(t[4 * i..4 * i + 4].try_into().unwrap());
            assert_eq!(c(3), 1.0, "alpha 1");
            std::array::from_fn(|i| {
                let code = (c(i) * max).round();
                assert!((0.0..=max).contains(&code), "{} off the codes", c(i));
                assert!(
                    converted_holds(c(i), code / max, kernels.exact),
                    "{} not code / {max}",
                    c(i)
                );
                code as u16
            })
        })
        .collect()
}

/// `YCBCR10_420_2P` and `RGB10_420_2P` textures over 10-bit surfaces read what Metal reads
/// (`ycbcr_read` states it): `read()` and a linear sample at each texel's centre, over `x420`,
/// `xf20`, `x422` and `xf44` surfaces without a matrix, with BT.709 and with BT.2020; and
/// `YCBCR10_420_2P`'s linear sample between the texels is Metal's step on the grid `2048`, as
/// `YCBCR8_420_2P`'s, where its planes are read in place, the filter of the kept image elsewhere
/// ([`Between`]). The filter notices are [`assert_filter_notices`]'s.
#[test]
fn ten_bit_textures_read_what_metal_reads() {
    on_device(|device, messages| {
        let kernels = Kernels::new(device);
        let linear = sampler(device, SamplerMinMagFilter::Linear);
        let mut steps = Between::new(device);
        let mut seed = 20;
        for pf in [PixelFormat::YCBCR104202P, PixelFormat::RGB104202P] {
            let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            for (ostype, range) in [
                (b"x420", YCbCrRange::Video),
                (b"xf20", YCbCrRange::Full),
                (b"x422", YCbCrRange::Video),
                (b"xf44", YCbCrRange::Full),
            ] {
                for (matrix, modelled) in [
                    (None, YCbCrMatrix::Bt601),
                    (Some(SurfaceYCbCrMatrix::Itu709), YCbCrMatrix::Bt709),
                    (Some(SurfaceYCbCrMatrix::Itu2020), YCbCrMatrix::Bt2020),
                ] {
                    seed += 1;
                    let bytes = ten_bit_bytes(seed);
                    let surface = buffer_with(device, &bytes).unwrap();
                    let before = made_in_place(device);
                    let texture = ten_bit_texture(&surface, pf, (ostype, 0), matrix).unwrap();
                    let in_place = made_in_place(device) > before;
                    let want = ten_bit_modelled(&bytes, codes, (range, modelled));
                    let case = format!("{pf:?} over {ostype:?}, {matrix:?}");
                    let of = |texels: &[u8]| match codes.output {
                        YCbCrOutput::Codes => code_grid(&kernels, texels, 1023),
                        _ => grid(texels),
                    };
                    let read = of(&texels_of(device, &kernels.read, &texture, None));
                    assert_eq!(read, want, "{case}: read()");
                    let sampled = texels_of(device, &kernels.sample, &texture, Some(&linear));
                    assert_eq!(of(&sampled), want, "{case}: a linear sample at the centres");
                    if codes.output == YCbCrOutput::Converted {
                        let between = texels_of(device, &kernels.between, &texture, Some(&linear));
                        steps.texels(&between, &read, (2048, in_place), &case);
                    }
                }
            }
        }
        steps.finish(device, &counts(&messages.take_notices()), "10-bit textures");
    });
}

/// A 10-bit texture's copies move its luma words, two bytes a texel, as Metal's do: `get_bytes`
/// reads the luma rows' words, and `replace_region` writes them, keeping the chroma plane, the
/// texture then reading the new codes' conversion.
#[test]
fn ten_bit_textures_copy_their_luma_words() {
    on_device(|device, _| {
        let kernels = Kernels::new(device);
        let bytes = ten_bit_bytes(41);
        let surface = buffer_with(device, &bytes).unwrap();
        let pf = PixelFormat::YCBCR104202P;
        let texture = ten_bit_texture(&surface, pf, (b"x420", 0), None).unwrap();
        let mut luma = vec![0u8; (2 * W * H) as usize];
        texture
            .get_bytes(&mut luma, 2 * W, 0, Region::new_2d(0, 0, W, H), 0, 0)
            .unwrap();
        let rows: Vec<u8> = (0..H)
            .flat_map(|y| bytes[(y * ROW10) as usize..(y * ROW10 + 2 * W) as usize].to_vec())
            .collect();
        assert_eq!(luma, rows, "get_bytes");
        let mut random = Random(43);
        let new: Vec<u8> = (0..2 * W * H).map(|_| random.next()).collect();
        texture
            .replace_region(Region::new_2d(0, 0, W, H), 0, 0, &new, 2 * W, 0)
            .unwrap();
        let mut want = bytes.clone();
        for y in 0..H {
            let at = (y * ROW10) as usize;
            want[at..at + 2 * W as usize]
                .copy_from_slice(&new[(2 * y * W) as usize..(2 * (y + 1) * W) as usize]);
        }
        assert_eq!(
            read_buffer(&surface, 0, LENGTH10).unwrap(),
            want,
            "replace_region"
        );
        let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
        let modelled = ten_bit_modelled(&want, codes, (YCbCrRange::Video, YCbCrMatrix::Bt601));
        let read = grid(&texels_of(device, &kernels.read, &texture, None));
        assert_eq!(read, modelled, "read() after replace_region");
    });
}

/// What a texture of `codes` reads of the surface `bytes`, zeros past their end, as Metal reads
/// any surface (`ycbcr.rs`'s **Surfaces**): its luma codes from `luma` (offset, bytes per row)
/// and its chroma pairs from `chroma`, or codes 0 without, each in the format's own layout
/// (unpacked) and filtered at its subsampling; each texel [`W`]x[`H`] `ycbcr_read` of them.
fn read_anywhere(
    bytes: &[u8],
    codes: YCbCrCodes,
    conversion: (YCbCrRange, YCbCrMatrix),
    planes: ([u64; 2], Option<[u64; 2]>),
) -> Vec<[u16; 3]> {
    anywhere_sums(bytes, codes, planes)
        .into_iter()
        .map(|(y, sums)| {
            let [cb, cr] = sums.map(|s| ((s + 8) >> 4) as u16);
            ycbcr_read(codes, conversion, [y, cb, cr])
        })
        .collect()
}

/// What [`read_anywhere`] reads, through the device's conversion
/// (`ycbcr_device_reads::vulkan_read`): the chroma sums over 16, unrounded.
fn anywhere_vulkan(
    bytes: &[u8],
    codes: YCbCrCodes,
    conversion: (YCbCrRange, YCbCrMatrix),
    planes: ([u64; 2], Option<[u64; 2]>),
) -> Vec<[f64; 3]> {
    anywhere_sums(bytes, codes, planes)
        .into_iter()
        .map(|(y, sums)| {
            let [cb, cr] = sums.map(|s| f64::from(s) / 16.0);
            vulkan_read(codes, conversion, [f64::from(y), cb, cr])
        })
        .collect()
}

/// Each texel's luma code and its chroma sums of 16 weights, as [`read_anywhere`] reads them.
fn anywhere_sums(
    bytes: &[u8],
    codes: YCbCrCodes,
    (luma, chroma): ([u64; 2], Option<[u64; 2]>),
) -> Vec<(u16, [u32; 2])> {
    let byte = |at: u64| bytes.get(at as usize).copied().unwrap_or(0);
    let unit = u64::from(codes.bytes);
    let code = |at: u64| match unit {
        1 => u16::from(byte(at)),
        _ => u16::from_le_bytes([byte(at), byte(at + 1)]) >> 6,
    };
    let [across, down] = codes.subsampling.halved();
    let [cw, ch] = [(W, across), (H, down)].map(|(n, halved)| if halved { n / 2 } else { n });
    // The pairs along one axis at texel `q` and their weights, 4 in all.
    let taps = |q: u64, halved: bool, n: u64| -> Vec<(u64, u32)> {
        match halved {
            false => vec![(q, 4)],
            true if q % 2 == 1 => vec![(q / 2, 3), ((q / 2 + 1).min(n - 1), 1)],
            true => vec![((q / 2).saturating_sub(1), 1), (q / 2, 3)],
        }
    };
    let mut out = Vec::new();
    for y in 0..H {
        for x in 0..W {
            let mut sums = [0u32; 2];
            if let Some([at, row]) = chroma {
                for (cy, wy) in taps(y, down, ch) {
                    for (cx, wx) in taps(x, across, cw) {
                        let pair = at + cy * row + 2 * unit * cx;
                        sums[0] += wx * wy * u32::from(code(pair));
                        sums[1] += wx * wy * u32::from(code(pair + unit));
                    }
                }
            }
            out.push((code(luma[0] + y * luma[1] + unit * x), sums));
        }
    }
    out
}

/// The `float4`s `texels` against `want`'s codes over `max` exactly (where the device converts
/// the codes itself, within its conversion's bound: `Kernels::exact`), each texel's red, green and
/// blue `want`'s channels in that order, alpha 1.
fn assert_codes_read(kernels: &Kernels, texels: &[u8], want: &[[u16; 3]], max: u16, case: &str) {
    for (i, (t, w)) in texels.as_chunks::<16>().0.iter().zip(want).enumerate() {
        let c = |i: usize| f32::from_le_bytes(t[4 * i..4 * i + 4].try_into().unwrap());
        let got = [c(0), c(1), c(2), c(3)];
        let want = [w[0], w[1], w[2]].map(|v| f32::from(v) / f32::from(max));
        let want = [want[0], want[1], want[2], 1.0];
        let held = (0..4).all(|c| converted_holds(got[c], want[c], kernels.exact));
        assert!(held, "{case}: texel {i} ({w:?}): {got:?}, Metal's {want:?}");
    }
}

/// A texture over any surface reads what Metal reads of it (the M3 Max under macOS 27.0.1, our
/// probe `ycr` over 35 OSTypes, every plane and format, every texel; `ycbcr.rs`'s **Surfaces**):
/// a 10-bit format over an 8-bit `420v` surface; a 4:2:2 format over plane 1 of a 10-bit 4:2:0 surface (chroma codes 0); a
/// format over a surface without planes (`BGRA`); a 4:4:4 format over a 4:2:0 full-range surface
/// under a matrix of no known name (BT.601); a 4:2:0 one whose chroma rows run past the buffer
/// (zeros there); and `DCI_P3` and `P3_D65` surfaces, whose converted formats read each code
/// alone, red, green and blue its luma, Cb and Cr over `2^b - 1`, and whose code formats read as
/// under any matrix. What Metal refuses or the runtime does not hold is refused, typed: a
/// lossless compressed OSType, plane 1 of a surface without planes, an `_sRGB` twin over a P3
/// surface, rows of codes that overlap (a 10-bit format over 8-bit rows narrower than its codes)
/// and luma rows past the buffer, the last two the runtime's gaps. Metal makes no render target of
/// a 4:2:0 format of odd width: its nil. A texture the device converts itself reads its conversion
/// of the same codes, within its bound (`ycbcr_device_reads`).
#[test]
fn textures_over_any_surface_read_what_metal_reads() {
    use PixelFormat as P;
    use YCbCrMatrix::Bt601;
    use YCbCrRange::{Full, Video};
    on_device(|device, _| {
        let kernels = Kernels::new(device);
        let mut random = Random(91);
        let bytes: Vec<u8> = (0..LENGTH10).map(|_| random.next()).collect();
        let surface = buffer_with(device, &bytes).unwrap();
        // A buffer ending inside the chroma plane of a 40x24 `420v` surface (rows 64 apart).
        let short_bytes = &bytes[..2000];
        let short = buffer_with(device, short_bytes).unwrap();
        let ostype = |code: &[u8; 4]| u32::from_be_bytes(*code);
        let texture = |buffer: &Buffer,
                       pf: PixelFormat,
                       (code, plane, planar): (&[u8; 4], u64, bool),
                       (luma, chroma): ([u64; 2], [u64; 2]),
                       matrix: Option<SurfaceYCbCrMatrix>| {
            let mut d = TextureDescriptor::texture_2d(pf, W, H, false);
            d.usage = TextureUsage::SHADER_READ;
            d.resource_options = ResourceOptions::shared();
            let facts = IoSurfaceYCbCr::new(ostype(code), plane, [W, H], chroma[0], chroma[1]);
            let facts = match matrix {
                Some(matrix) => facts.with_matrix(matrix),
                None => facts,
            };
            let at = IoSurfacePlane::new(luma[0], luma[1], planar).with_ycbcr(facts);
            buffer.new_texture_with_iosurface_plane(&d, at)
        };
        let codes = |pf: PixelFormat| ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
        let plane0 = ([0, ROW], [CHROMA_AT, ROW]);
        let wide = ([0, ROW10], [CHROMA10_AT, ROW10]);
        let plane1 = ([CHROMA10_AT, ROW10], [CHROMA10_AT, ROW10]);
        let p3 = Some(SurfaceYCbCrMatrix::DciP3);
        let p3d65 = Some(SurfaceYCbCrMatrix::P3D65);
        let other = Some(SurfaceYCbCrMatrix::Other);
        let converted = [
            // (format, buffer, surface, planes, matrix, range, chroma read)
            (
                P::YCBCR104202P,
                &surface,
                (b"420v", 0, true),
                wide,
                None,
                Video,
                true,
            ),
            (
                P::YCBCR84222P,
                &surface,
                (b"x420", 1, true),
                plane1,
                None,
                Video,
                false,
            ),
            (
                P::YCBCR84202P,
                &surface,
                (b"BGRA", 0, false),
                plane0,
                None,
                Video,
                false,
            ),
            (
                P::YCBCR84442P,
                &surface,
                (b"420f", 0, true),
                plane0,
                other,
                Full,
                true,
            ),
            (
                P::YCBCR84202P,
                &short,
                (b"420v", 0, true),
                plane0,
                None,
                Video,
                true,
            ),
        ];
        for (pf, buffer, at, planes, matrix, range, read_chroma) in converted {
            let case = format!(
                "{pf:?} over {:?}",
                (String::from_utf8_lossy(at.0), at.1, at.2)
            );
            let converted = device_conversions(device);
            let made = texture(buffer, pf, at, planes, matrix);
            let made = made.unwrap_or_else(|e| panic!("{case}: {e:?}"));
            let by_device = device_conversions(device) > converted;
            let source = match buffer.length() {
                2000 => short_bytes,
                _ => &bytes[..],
            };
            let chroma = read_chroma.then_some(planes.1);
            let read = texels_of(device, &kernels.read, &made, None);
            let (codes, planes) = (codes(pf), (planes.0, chroma));
            match by_device {
                true => {
                    let want = anywhere_vulkan(source, codes, (range, Bt601), planes);
                    assert_device_reads(&read, &want, (codes.output, false), &case);
                }
                false => {
                    let want = read_anywhere(source, codes, (range, Bt601), planes);
                    assert_eq!(grid(&read), want, "{case}");
                }
            }
        }
        // P3: a converted format reads its luma, Cb and Cr codes alone, a code format its codes.
        for (pf, matrix) in [
            (P::YCBCR84202P, p3),
            (P::YCBCR104202P, p3d65),
            (P::RGB84202P, p3),
        ] {
            let case = format!("{pf:?} under {matrix:?}");
            let made = texture(&surface, pf, (b"420v", 0, true), wide, matrix);
            let made = made.unwrap_or_else(|e| panic!("{case}: {e:?}"));
            let mut read = codes(pf);
            read.output = YCbCrOutput::Codes;
            let raw = read_anywhere(&bytes, read, (Video, Bt601), (wide.0, Some(wide.1)));
            // `ycbcr_read` of codes answers red Cr, green Y, blue Cb.
            let want: Vec<[u16; 3]> = match ycbcr_format(pf).unwrap().reads {
                YCbCrReads::Codes => raw,
                _ => raw.iter().map(|&[cr, y, cb]| [y, cb, cr]).collect(),
            };
            let max = (1u16 << read.bits) - 1;
            assert_codes_read(
                &kernels,
                &texels_of(device, &kernels.read, &made, None),
                &want,
                max,
                &case,
            );
        }
        let refused = |made: Result<Texture, Error>, what: &str| {
            assert!(
                matches!(made, Err(Error::Unsupported(_))),
                "{what}: {:?}",
                made.err()
            );
        };
        refused(
            texture(&surface, P::YCBCR84202P, (b"&8v0", 0, true), plane0, None),
            "a compressed surface",
        );
        refused(
            texture(&surface, P::YCBCR84202P, (b"BGRA", 1, false), plane0, None),
            "plane 1 of a surface without planes",
        );
        refused(
            texture(&surface, P::YCBCR84202PSRGB, (b"420v", 0, true), plane0, p3),
            "an sRGB twin over a P3 surface",
        );
        // The runtime's gaps: rows of codes that overlap (a 10-bit format over 8-bit rows of 64
        // bytes), and luma rows past the buffer (plane 1 of a surface ending inside it).
        assert_eq!(
            texture(&surface, P::YCBCR104202P, (b"420v", 0, true), plane0, None).err(),
            Some(Error::NotImplemented(RuntimeGap::OverlappingSurfaceRows)),
            "rows of codes that overlap"
        );
        let short_plane1 = ([CHROMA_AT, ROW], [CHROMA_AT, ROW]);
        assert_eq!(
            texture(
                &short,
                P::YCBCR84202P,
                (b"420v", 1, true),
                short_plane1,
                None
            )
            .err(),
            Some(Error::NotImplemented(RuntimeGap::TexelsPastBufferLength)),
            "luma rows past the buffer"
        );
        // Odd widths: a render target is Metal's nil, a texture to read is made.
        for (usage, made) in [
            (
                TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
                false,
            ),
            (TextureUsage::SHADER_READ, true),
        ] {
            let mut d = TextureDescriptor::texture_2d(P::YCBCR84202P, W - 1, H, false);
            d.usage = usage;
            d.resource_options = ResourceOptions::shared();
            let facts = IoSurfaceYCbCr::new(ostype(b"420v"), 0, [W - 1, H], CHROMA_AT, ROW);
            let at = IoSurfacePlane::new(0, ROW, true).with_ycbcr(facts);
            let result = surface.new_texture_with_iosurface_plane(&d, at);
            match made {
                true => assert!(result.is_ok(), "{:?}", result.err()),
                false => assert_eq!(
                    result.err(),
                    Some(Error::Nil(Nil::new(
                        "new_texture_with_iosurface_plane",
                        Problem::Texture(TextureProblem::ExtentForFormat),
                    )))
                ),
            }
        }
    });
}

/// The `f32` of the `half` nearest `step / 8192` with ties away from zero, as Metal rounds a
/// `half` read of a step of `1 / 8192`. Every nonzero step is a normal `half`'s
/// magnitude, so the `half`s about it lie `2^(e - 10)` apart for its binade `e`.
fn half_away(step: u16) -> f32 {
    if step == 0 {
        return 0.0;
    }
    let v = f64::from(step) / 8192.0;
    let ulp = 2f64.powi(v.log2().floor() as i32 - 10);
    ((v / ulp + 0.5).floor() * ulp) as f32
}

/// The `float4`s `texels` widened from `half`s, against Metal's `half` read of the steps
/// `want`: each channel [`half_away`] of its step, alpha 1.
fn assert_half_reads(kernels: &Kernels, texels: &[u8], want: &[[u16; 3]], case: &str) {
    assert_half_steps(kernels, texels, want, 8192, case);
}

/// A `half` read's `float4`s `texels` against `want`'s steps of `1 / grid`, alpha 1: each step's
/// `half`, rounded with ties away from zero on the grid `8192` ([`half_away`]) and exact on the
/// grid `2048`. Where the read's stage rounds by the device's conversion (`Kernels::native_half`,
/// `PrecisionNote::NativeHalfConversion`: rounded otherwise or not at all) the channels that differ
/// are counted and printed, with the largest difference.
fn assert_half_steps(kernels: &Kernels, texels: &[u8], want: &[[u16; 3]], grid: u32, case: &str) {
    let metal = |w: u16| match grid {
        8192 => half_away(w),
        _ => f32::from(w) / grid as f32,
    };
    let (mut differ, mut most) = (0usize, 0.0f32);
    for (i, (t, w)) in texels.as_chunks::<16>().0.iter().zip(want).enumerate() {
        let c = |i: usize| f32::from_le_bytes(t[4 * i..4 * i + 4].try_into().unwrap());
        assert_eq!(c(3), 1.0, "{case}: texel {i}'s alpha");
        let got: [f32; 3] = std::array::from_fn(c);
        let want = w.map(metal);
        if !kernels.native_half {
            assert_eq!(got, want, "{case}: texel {i} ({w:?})");
            continue;
        }
        for (g, m) in got.iter().zip(&want) {
            if g.to_bits() != m.to_bits() {
                differ += 1;
                most = most.max((g - m).abs());
            }
        }
    }
    if differ > 0 {
        println!("YCBCR-HALF-DIFFERS {case}: {differ} channels the device's rounding, at most {most} off");
    }
}

/// The `_sRGB` twins over their surfaces read what Metal reads (`ycbcr_read` states it):
/// `YCBCR8_420_2P_sRGB` over `420v` and `420f`, and `YCBCR10_420_2P_sRGB` over `x420`, `xf20`,
/// `x422` and `xf44`, without a matrix, with BT.709 and with BT.2020. Each texel's `float` `read()`
/// and a linear sample at its centre are its step of `1/8192`; a `half` read and a `half` nearest
/// sample that step rounded with ties away from zero; and a linear sample between the texels the
/// filter of the kept image, whose planes no function reads in place ([`Between`], Metal's step
/// on the grid `8192` where read in place). The filter notices are [`assert_filter_notices`]'s.
#[test]
fn srgb_twins_read_what_metal_reads() {
    on_device(|device, messages| {
        let kernels = Kernels::new(device);
        let linear = sampler(device, SamplerMinMagFilter::Linear);
        let nearest = sampler(device, SamplerMinMagFilter::Nearest);
        let mut steps = Between::new(device);
        let mut seed = 70;
        let surfaces: [(PixelFormat, &[u8; 4], YCbCrRange); 6] = [
            (PixelFormat::YCBCR84202PSRGB, b"420v", YCbCrRange::Video),
            (PixelFormat::YCBCR84202PSRGB, b"420f", YCbCrRange::Full),
            (PixelFormat::YCBCR104202PSRGB, b"x420", YCbCrRange::Video),
            (PixelFormat::YCBCR104202PSRGB, b"xf20", YCbCrRange::Full),
            (PixelFormat::YCBCR104202PSRGB, b"x422", YCbCrRange::Video),
            (PixelFormat::YCBCR104202PSRGB, b"xf44", YCbCrRange::Full),
        ];
        for (pf, ostype, range) in surfaces {
            let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            assert_eq!(codes.output, YCbCrOutput::Srgb);
            for (matrix, modelled) in [
                (None, YCbCrMatrix::Bt601),
                (Some(SurfaceYCbCrMatrix::Itu709), YCbCrMatrix::Bt709),
                (Some(SurfaceYCbCrMatrix::Itu2020), YCbCrMatrix::Bt2020),
            ] {
                seed += 1;
                let before = made_in_place(device);
                let (want, texture) = if codes.bits == 8 {
                    let bytes = surface_bytes(seed);
                    let surface = buffer_with(device, &bytes).unwrap();
                    let facts = facts(u32::from_be_bytes(*ostype), matrix);
                    let made =
                        eight_bit_texture(&surface, pf, Some(facts), TextureUsage::SHADER_READ);
                    (modelled_as(&bytes, codes, (range, modelled)), made.unwrap())
                } else {
                    let bytes = ten_bit_bytes(seed);
                    let surface = buffer_with(device, &bytes).unwrap();
                    let made = ten_bit_texture(&surface, pf, (ostype, 0), matrix);
                    (
                        ten_bit_modelled(&bytes, codes, (range, modelled)),
                        made.unwrap(),
                    )
                };
                let in_place = made_in_place(device) > before;
                let case = format!("{pf:?} over {ostype:?}, {matrix:?}");
                let read = grid_of(&texels_of(device, &kernels.read, &texture, None), 8192);
                assert_eq!(read, want, "{case}: read()");
                let sampled = texels_of(device, &kernels.sample, &texture, Some(&linear));
                assert_eq!(
                    grid_of(&sampled, 8192),
                    want,
                    "{case}: a linear sample at the centres"
                );
                let half = texels_of(device, &kernels.read_half, &texture, None);
                assert_half_reads(&kernels, &half, &want, &format!("{case}: a half read()"));
                let half = texels_of(device, &kernels.sample_half, &texture, Some(&nearest));
                assert_half_reads(
                    &kernels,
                    &half,
                    &want,
                    &format!("{case}: a half nearest sample"),
                );
                let between = texels_of(device, &kernels.between, &texture, Some(&linear));
                steps.texels(&between, &read, (8192, in_place), &case);
            }
        }
        steps.finish(device, &counts(&messages.take_notices()), "sRGB twins");
    });
}

/// A converted texture's view as its `_sRGB` twin reads what a texture of the twin over the same
/// planes reads, as Metal's view does (made with `ShaderRead` alone):
/// `YCBCR8_420_2P` over `420f` with BT.709 as `YCBCR8_420_2P_sRGB`, and `YCBCR10_420_2P` over
/// `x420` as `YCBCR10_420_2P_sRGB`; the texture itself still reads its own conversion, and after
/// `replace_region` writes the luma plane through it, the view reads the new codes' twin. The other
/// order and a view of another layout are refused.
#[test]
fn ycbcr_views_as_srgb_twins_read_the_twins_texels() {
    on_device(|device, _| {
        let kernels = Kernels::new(device);
        let codes = |pf: PixelFormat| ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
        let bt709 = (YCbCrRange::Full, YCbCrMatrix::Bt709);
        // 8 bits.
        let bytes = surface_bytes(81);
        let surface = buffer_with(device, &bytes).unwrap();
        let facts = facts(FULL, Some(SurfaceYCbCrMatrix::Itu709));
        let usage = TextureUsage::SHADER_READ;
        let base =
            eight_bit_texture(&surface, PixelFormat::YCBCR84202P, Some(facts), usage).unwrap();
        let view = base.new_texture_view(PixelFormat::YCBCR84202PSRGB).unwrap();
        let twin =
            eight_bit_texture(&surface, PixelFormat::YCBCR84202PSRGB, Some(facts), usage).unwrap();
        let srgb = codes(PixelFormat::YCBCR84202PSRGB);
        let want = modelled_as(&bytes, srgb, bt709);
        let read = |t: &Texture, d| grid_of(&texels_of(device, &kernels.read, t, None), d);
        assert_eq!(read(&view, 8192), want, "the view");
        assert_eq!(read(&twin, 8192), want, "the twin");
        assert_eq!(
            read(&base, 2048),
            modelled(&bytes, bt709.0, bt709.1),
            "the texture"
        );
        let mut random = Random(83);
        let luma: Vec<u8> = (0..W * H).map(|_| random.next()).collect();
        base.replace_region(Region::new_2d(0, 0, W, H), 0, 0, &luma, W, 0)
            .unwrap();
        let written = read_buffer(&surface, 0, LENGTH).unwrap();
        assert_eq!(
            read(&view, 8192),
            modelled_as(&written, srgb, bt709),
            "the view, written"
        );
        // 10 bits.
        let bytes = ten_bit_bytes(85);
        let surface = buffer_with(device, &bytes).unwrap();
        let base =
            ten_bit_texture(&surface, PixelFormat::YCBCR104202P, (b"x420", 0), None).unwrap();
        let view = base
            .new_texture_view(PixelFormat::YCBCR104202PSRGB)
            .unwrap();
        let video = (YCbCrRange::Video, YCbCrMatrix::Bt601);
        let want = ten_bit_modelled(&bytes, codes(PixelFormat::YCBCR104202PSRGB), video);
        assert_eq!(read(&view, 8192), want, "the 10-bit view");
        // Refused: the twin as the converted format (not measured), and another layout.
        let twin =
            ten_bit_texture(&surface, PixelFormat::YCBCR104202PSRGB, (b"x420", 0), None).unwrap();
        let error = twin.new_texture_view(PixelFormat::YCBCR104202P).err();
        let named = format!("{error:?}");
        assert!(
            matches!(error, Some(Error::Unsupported(_)))
                && named.contains(PixelFormat::YCBCR104202P.name())
                && !named.contains(PixelFormat::YCBCR104202PSRGB.name()),
            "the other order: {named}"
        );
        assert!(
            base.new_texture_view(PixelFormat::YCBCR84202PSRGB).is_err(),
            "another layout's twin"
        );
    });
}

/// What `sample_outside` reads of `texture` through `sampler` (or `state`'s own constexpr
/// sampler, given none): a `float4` a point, `2 W` points a row, `2 H` rows.
fn outside_of(
    device: &Device,
    state: &ComputePipelineState,
    texture: &Texture,
    sampler: Option<&SamplerState>,
) -> Vec<u8> {
    let out = device
        .new_buffer(64 * W * H, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    dispatch(
        &command_buffer,
        state,
        texture,
        &out,
        sampler,
        [2 * W, 2 * H],
    );
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    read_buffer(&out, 0, 64 * W * H).unwrap()
}

/// The bytes of the `float4` `v`.
fn float4_bytes(v: [f32; 4]) -> Vec<u8> {
    v.iter().flat_map(|c| c.to_le_bytes()).collect()
}

/// Samples outside a two-plane YCbCr texture read as Metal reads them (the M4 and the M3 Max alike;
/// for 4:2:2 and 4:4:4), nearest and linear, at texel centres from `-W/2` to `3W/2`: under
/// clamp-to-edge, mirror-clamp-to-edge, repeat and mirrored repeat each point of a 4:2:0 or 4:2:2
/// texture reads its texel clamped to the texture, of a 4:4:4 one its texel by that mode; under
/// clamp-to-zero and clamp-to-border-color (transparent black, opaque black, opaque white) the
/// converted formats and their sRGB twins read the border outside, its alpha 1, and `RGB10_420_2P`
/// and `RGB8_420_2P` (the M4), read as codes, its texel clamped to the texture again, under every
/// border color. The packed formats read as their unpacked twins. Where the device samples by its
/// own sampler (announced `device-sampling`), a linear sample is its filter of those texels: its
/// differences are counted and printed.
#[test]
fn ycbcr_samples_outside_read_metals_modes() {
    on_device(|device, messages| {
        let kernels = Kernels::new(device);
        let own = own_sampler(device, messages);
        let mut filtered = 0usize;
        let usage = TextureUsage::SHADER_READ;
        let eight = buffer_with(device, &surface_bytes(91)).unwrap();
        let ten = buffer_with(device, &ten_bit_bytes(92)).unwrap();
        let ten_full = buffer_with(device, &ten_bit_bytes(93)).unwrap();
        let family: Vec<(PixelFormat, &str, Buffer)> = (94..)
            .zip([
                (PixelFormat::YCBCR84222P, "422v"),
                (PixelFormat::YCBCR84442P, "444v"),
                (PixelFormat::YCBCR104222P, "x422"),
                (PixelFormat::YCBCR104442P, "x444"),
                (PixelFormat::YCBCR104442PSRGB, "x444"),
                (PixelFormat::RGB84222P, "422v"),
                (PixelFormat::RGB84442P, "444v"),
                (PixelFormat::RGB104222P, "x422"),
                (PixelFormat::RGB104442P, "x444"),
                (PixelFormat::YCBCR104202PPACKED, "p420"),
                (PixelFormat::YCBCR104222PPACKEDSRGB, "pf22"),
                (PixelFormat::YCBCR104442PPACKED, "p444"),
                (PixelFormat::YCBCR104442PPACKEDSRGB, "pf44"),
            ])
            .map(|(seed, (pf, ostype))| {
                let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
                let surface = buffer_with(device, &family_bytes(seed, codes)).unwrap();
                (pf, ostype, surface)
            })
            .collect();
        let mut textures: Vec<(PixelFormat, &str, Texture)> = Vec::new();
        for pf in [
            PixelFormat::YCBCR84202P,
            PixelFormat::YCBCR84202PSRGB,
            PixelFormat::RGB84202P,
        ] {
            let surface_facts = Some(facts(VIDEO, None));
            let texture = eight_bit_texture(&eight, pf, surface_facts, usage).unwrap();
            textures.push((pf, "420v", texture));
        }
        for (pf, surface, ostype) in [
            (PixelFormat::YCBCR104202P, &ten, b"x420"),
            (PixelFormat::YCBCR104202PSRGB, &ten, b"x420"),
            (PixelFormat::RGB104202P, &ten, b"x420"),
            (PixelFormat::RGB104202P, &ten_full, b"xf20"),
        ] {
            let texture = ten_bit_texture(surface, pf, (ostype, 0), None).unwrap();
            textures.push((pf, std::str::from_utf8(ostype).unwrap(), texture));
        }
        for &(pf, ostype, ref surface) in &family {
            let texture = family_texture(surface, pf, ostype, None).unwrap();
            textures.push((pf, ostype, texture));
        }
        let border = SamplerAddressMode::ClampToBorderColor;
        let clear = SamplerBorderColor::TransparentBlack;
        let modes = [
            (SamplerAddressMode::ClampToEdge, clear),
            (SamplerAddressMode::MirrorClampToEdge, clear),
            (SamplerAddressMode::Repeat, clear),
            (SamplerAddressMode::MirrorRepeat, clear),
            (SamplerAddressMode::ClampToZero, clear),
            (border, clear),
            (border, SamplerBorderColor::OpaqueBlack),
            (border, SamplerBorderColor::OpaqueWhite),
        ];
        // Every state made before any command, so that each command's bind shape holds them all.
        let samplers: Vec<_> = [SamplerMinMagFilter::Nearest, SamplerMinMagFilter::Linear]
            .into_iter()
            .flat_map(|filter| modes.map(|mode| (filter, mode)))
            .map(|(filter, (mode, color))| {
                let mut d = SamplerDescriptor::new();
                d.min_filter = filter;
                d.mag_filter = filter;
                d.s_address_mode = mode;
                d.t_address_mode = mode;
                d.border_color = color;
                ((filter, mode, color), device.new_sampler_state(&d).unwrap())
            })
            .collect();
        for (pf, ostype, texture) in &textures {
            let pf = *pf;
            let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            let read = texels_of(device, &kernels.read, texture, None);
            let texel = |x: u64, y: u64| read[(16 * (y * W + x)) as usize..][..16].to_vec();
            for ((filter, mode, color), state) in &samplers {
                let reads_border = codes.output != YCbCrOutput::Codes
                    && [SamplerAddressMode::ClampToZero, border].contains(mode);
                let border_texel = if *color == SamplerBorderColor::OpaqueWhite {
                    [1.0; 4]
                } else {
                    [0.0, 0.0, 0.0, 1.0]
                };
                let got = outside_of(device, &kernels.outside, texture, Some(state));
                for (i, point) in (0u64..).zip(got.as_chunks::<16>().0.iter()) {
                    let at = (i % (2 * W)) as i64 - (W / 2) as i64;
                    let row = (i / (2 * W)) as i64 - (H / 2) as i64;
                    let inside = (0..W as i64).contains(&at) && (0..H as i64).contains(&row);
                    let want = if reads_border && !inside {
                        float4_bytes(border_texel)
                    } else {
                        let wraps = codes.subsampling == ChromaSubsampling::Chroma444;
                        let mode = if wraps {
                            *mode
                        } else {
                            SamplerAddressMode::ClampToEdge
                        };
                        texel(addressed(at, W, mode), addressed(row, H, mode))
                    };
                    if own && *filter == SamplerMinMagFilter::Linear && point != &want[..] {
                        filtered += 1;
                        continue;
                    }
                    assert_eq!(
                        point,
                        &want[..],
                        "{pf:?} {ostype:?} {filter:?} {mode:?} {color:?}: texel ({at}, {row})"
                    );
                }
            }
        }
        println!(
            "YCBCR-OUTSIDE {}: {filtered} linear samples the device's filter",
            device.name()
        );
    });
}

/// A constexpr sampler's words: nearest filters, no mip filter, normalized coordinates that
/// repeat on every axis.
const NEAREST_REPEAT: u64 = 0x007b_ff00_0008_0000 | 2 | 2 << 3 | 2 << 6;

/// Synthetic AIR written for this lane: `sample_outside_repeat` is `sample_outside` through a
/// constexpr sampler of [`NEAREST_REPEAT`]'s words in place of sampler slot 0.
fn constexpr_outside() -> String {
    format!(
        r#"; Owned synthetic kernel.
source_filename = "ycbcr_constexpr.metal"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {NEAREST_REPEAT}, i64 0], align 8

define void @sample_outside_repeat(ptr addrspace(1) %t, ptr addrspace(1) noalias %out, <2 x i32> %gid, <2 x i32> %size) {{
entry:
  %g = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %gid)
  %n = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %size)
  %q = fmul <2 x float> %n, <float 2.500000e-01, float 2.500000e-01>
  %c0 = fadd <2 x float> %g, <float 5.000000e-01, float 5.000000e-01>
  %c = fsub <2 x float> %c0, %q
  %m = fmul <2 x float> %n, <float 5.000000e-01, float 5.000000e-01>
  %uv = fdiv <2 x float> %c, %m
  %r = tail call {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue {{ <4 x float>, i8 }} %r, 0
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %w = extractelement <2 x i32> %size, i64 0
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %j = zext i32 %i to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %j
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}}

declare <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32>)
declare {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

!air.kernel = !{{!0}}
!air.sampler_states = !{{!9}}
!air.version = !{{!30}}
!air.language_version = !{{!31}}
!0 = !{{ptr @sample_outside_repeat, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5, !6}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}}
!5 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}}
!6 = !{{i32 3, !"air.threads_per_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"size"}}
!9 = !{{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}}
!30 = !{{i32 2, i32 8, i32 0}}
!31 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// A constexpr sampler takes a two-plane YCbCr texture's address-mode map through one pipeline
/// variant of its slot, and every other texture keeps the state's pipeline:
/// `sample_outside_repeat`, through a constexpr nearest repeat sampler, makes nothing when
/// dispatched again over an `RGBA8Unorm` texture, and no variant (no shader module) over a
/// `YCBCR10_444_2P` one, whose map keeps repeat (a 4:4:4 texture repeats); over a `YCBCR10_420_2P`
/// one, whose map samples repeat as clamp-to-edge, it makes exactly one variant (one shader
/// module), once, which a `YCBCR8_420_2P` one, whose map agrees on repeat, shares. Over each it
/// reads, word for word, what the runtime repeat sampler at slot 0 reads (`sample_outside`), as
/// Metal samples a constexpr sampler as a runtime one of its modes: the 4:2:0 textures their
/// texels clamped to the edge. A dispatch again of a made variant, after the texture's shadow
/// copy, records in a `VkCommandBuffer` of its own exactly where the driver loses baked samplers
/// after a copy (`RuntimeDriverDefect::ImmutableSamplersLostAfterCopies`: lavapipe 26.0.8 sampled
/// it as Repeat in place). On a device that reads the planes in place (`ycbcr_in_place.rs`) the
/// 4:2:0 textures' variant is the in-place one, which bakes no sampler and copies no image, so it
/// records with the rest.
#[test]
fn a_constexpr_sampler_takes_a_ycbcr_map_through_one_variant() {
    on_device(|device, _| {
        let kernels = Kernels::new(device);
        let library = device
            .new_library_with_air(&llvm_assemble_text(&constexpr_outside()).unwrap())
            .unwrap();
        let constexpr = device
            .new_compute_pipeline_state(
                &ComputePipelineDescriptor::new(StageSource::Function(
                    library
                        .new_function_with_name("sample_outside_repeat")
                        .unwrap()
                        .function_descriptor(),
                )),
                &PipelineOptions::default(),
            )
            .unwrap();
        let mut d = SamplerDescriptor::new();
        d.s_address_mode = SamplerAddressMode::Repeat;
        d.t_address_mode = SamplerAddressMode::Repeat;
        let repeat = device.new_sampler_state(&d).unwrap();
        let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, W, H, false);
        d.usage = TextureUsage::SHADER_READ;
        let rgba = device.new_texture(&d).unwrap();
        let mut random = Random(95);
        let texels: Vec<u8> = (0..4 * W * H).map(|_| random.next()).collect();
        rgba.replace_region(Region::new_2d(0, 0, W, H), 0, 0, &texels, 4 * W, 0)
            .unwrap();
        let ten = buffer_with(device, &ten_bit_bytes(96)).unwrap();
        let ten_420 = ten_bit_texture(&ten, PixelFormat::YCBCR104202P, (b"x420", 0), None).unwrap();
        let eight = buffer_with(device, &surface_bytes(97)).unwrap();
        let eight_420 = eight_bit_texture(
            &eight,
            PixelFormat::YCBCR84202P,
            Some(facts(VIDEO, None)),
            TextureUsage::SHADER_READ,
        )
        .unwrap();
        let codes = ycbcr_format(PixelFormat::YCBCR104442P)
            .and_then(YCbCrCodes::of)
            .unwrap();
        let four = buffer_with(device, &family_bytes(98, codes)).unwrap();
        let ten_444 = family_texture(&four, PixelFormat::YCBCR104442P, "x444", None).unwrap();
        // Each texture read through the runtime sampler first: what the constexpr sampler reads,
        // and any work of the texture's own done before the counts are taken.
        let textures = [&rgba, &ten_420, &eight_420, &ten_444];
        let want: Vec<Vec<u8>> = textures
            .iter()
            .map(|texture| outside_of(device, &kernels.outside, texture, Some(&repeat)))
            .collect();
        let read = texels_of(device, &kernels.read, &ten_420, None);
        let texel = |x: u64, y: u64| read[(16 * (y * W + x)) as usize..][..16].to_vec();
        for (i, point) in (0u64..).zip(want[1].as_chunks::<16>().0.iter()) {
            let at = (i % (2 * W)) as i64 - (W / 2) as i64;
            let row = (i / (2 * W)) as i64 - (H / 2) as i64;
            let edge = SamplerAddressMode::ClampToEdge;
            assert_eq!(
                point,
                &texel(addressed(at, W, edge), addressed(row, H, edge))[..],
                "YCBCR10_420_2P repeat: texel ({at}, {row})"
            );
        }
        // What one dispatch of the constexpr kernel over `texture` reads, the shader modules its
        // new variant's stages took, and the `VkPipeline`s it made. A new variant takes its
        // module, made or the device's shared one of the same SPIR-V (on the M4, the YCBCR10_420_2P
        // variant made none of its own). A variant kept takes none.
        let dispatched = |texture: &Texture| {
            let before = device.work_counts();
            let got = outside_of(device, &constexpr, texture, None);
            let after = device.work_counts();
            (
                got,
                (after.shader_modules_made + after.shader_modules_shared)
                    - (before.shader_modules_made + before.shader_modules_shared),
                after.variant_pipelines_made - before.variant_pipelines_made,
                after.baked_sampler_dispatches_apart - before.baked_sampler_dispatches_apart,
            )
        };
        // Each variant bakes its sampler: a dispatch whose pipeline is made records apart where
        // the driver loses baked samplers after a copy, the texture's shadow copy among them.
        let apart = u64::from(
            device
                .properties()
                .driver_defects
                .contains(&RuntimeDriverDefect::ImmutableSamplersLostAfterCopies),
        );
        let (got, _, _, _) = dispatched(&rgba);
        assert_eq!(got, want[0], "RGBA8Unorm");
        let (got, modules, pipelines, recorded_apart) = dispatched(&rgba);
        assert_eq!(got, want[0], "RGBA8Unorm, again");
        assert_eq!((modules, pipelines), (0, 0), "RGBA8Unorm, again");
        assert_eq!(recorded_apart, apart, "RGBA8Unorm, again: recorded apart");
        // A texture of another texel class may specialize the variant's module anew (a
        // `VkPipeline`, the filter class), never compile another.
        let (got, modules, _, _) = dispatched(&ten_444);
        assert_eq!(got, want[3], "YCBCR10_444_2P");
        assert_eq!(
            modules, 0,
            "YCBCR10_444_2P keeps the RGBA8Unorm texture's variant"
        );
        let (got, modules, pipelines, _) = dispatched(&ten_420);
        assert_eq!(got, want[1], "YCBCR10_420_2P");
        assert_eq!(modules, 1, "YCBCR10_420_2P makes one variant");
        assert!(
            pipelines >= 1,
            "YCBCR10_420_2P makes its variant's pipeline"
        );
        let (got, modules, pipelines, recorded_apart) = dispatched(&ten_420);
        assert_eq!(got, want[1], "YCBCR10_420_2P, again");
        assert_eq!((modules, pipelines), (0, 0), "YCBCR10_420_2P, again");
        // A variant that reads the planes in place bakes no sampler and copies no image.
        let in_place = super::ycbcr_in_place::reads_in_place(device);
        assert_eq!(
            recorded_apart,
            apart * u64::from(!in_place),
            "YCBCR10_420_2P, again: recorded apart"
        );
        let (got, modules, _, _) = dispatched(&eight_420);
        assert_eq!(got, want[2], "YCBCR8_420_2P");
        assert_eq!(modules, 0, "YCBCR8_420_2P shares YCBCR10_420_2P's variant");
        let (got, modules, _, _) = dispatched(&rgba);
        assert_eq!(got, want[0], "RGBA8Unorm, after");
        assert_eq!(modules, 0, "RGBA8Unorm, after");
    });
}

/// `RGB8_420_2P` textures over `420v` and `420f` surfaces read their codes as Metal does
/// (`ycbcr_read` states it): red Cr, green Y, blue Cb, each `code / 255`, alpha 1, after the
/// 9/3/3/1 chroma filter, whatever the matrix: `read()`, and a nearest and a linear sample at each
/// texel's centre. No filter notice.
#[test]
fn eight_bit_codes_read_what_metal_reads() {
    on_device(|device, messages| {
        let kernels = Kernels::new(device);
        let nearest = sampler(device, SamplerMinMagFilter::Nearest);
        let linear = sampler(device, SamplerMinMagFilter::Linear);
        let pf = PixelFormat::RGB84202P;
        let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
        let usage = TextureUsage::SHADER_READ;
        for (seed, ostype, matrix) in [
            (41, VIDEO, None),
            (42, FULL, Some(SurfaceYCbCrMatrix::Itu601)),
            (43, VIDEO, Some(SurfaceYCbCrMatrix::Itu709)),
            (44, FULL, Some(SurfaceYCbCrMatrix::Itu709)),
        ] {
            let bytes = surface_bytes(seed);
            let surface = buffer_with(device, &bytes).unwrap();
            let texture =
                eight_bit_texture(&surface, pf, Some(facts(ostype, matrix)), usage).unwrap();
            // The codes, which no range or matrix changes.
            let want = modelled_as(&bytes, codes, (YCbCrRange::Video, YCbCrMatrix::Bt601));
            let case = format!("over {:?}, {matrix:?}", ostype.to_be_bytes());
            let read = code_grid(
                &kernels,
                &texels_of(device, &kernels.read, &texture, None),
                255,
            );
            assert_eq!(read, want, "{case}: read()");
            for (filter, state) in [("nearest", &nearest), ("linear", &linear)] {
                let sampled = texels_of(device, &kernels.sample, &texture, Some(state));
                assert_eq!(
                    code_grid(&kernels, &sampled, 255),
                    want,
                    "{case}: a {filter} sample"
                );
            }
        }
        let notices = counts(&messages.take_notices());
        for kind in [
            Degradation::FloatSampleFilter,
            Degradation::YCbCrFilterNotExact,
        ] {
            assert_eq!(notices.get(&kind), None, "{notices:?}");
        }
    });
}

/// A 4:2:2 or 4:4:4 surface's rows, wide enough for a 4:4:4 10-bit chroma row (`4 W` bytes).
const ROWQ: u64 = 256;
/// The 4:2:2 or 4:4:4 surface's chroma plane's first byte.
const CHROMAQ_AT: u64 = ROWQ * H;

/// The texel `p` of an axis of `n` texels reads under `mode` (a point at its centre): clamped
/// to the texture, repeated, mirrored once and clamped, or mirrored and repeated.
fn addressed(p: i64, n: u64, mode: SamplerAddressMode) -> u64 {
    let n = n as i64;
    let at = if mode == SamplerAddressMode::Repeat {
        p.rem_euclid(n)
    } else if mode == SamplerAddressMode::MirrorClampToEdge {
        (if p < 0 { -1 - p } else { p }).min(n - 1)
    } else if mode == SamplerAddressMode::MirrorRepeat {
        let q = p.rem_euclid(2 * n);
        if q < n {
            q
        } else {
            2 * n - 1 - q
        }
    } else {
        p.clamp(0, n - 1)
    };
    at as u64
}

/// The bytes of a surface of `codes` whose planes' codes our generator at `seed` fills, laid out
/// by [`ROWQ`] and [`CHROMAQ_AT`] (a code a byte or a 16-bit word's bytes, or three packed codes
/// a 32-bit word's, every bit of a row's units random), every other byte `0xa5`.
fn family_bytes(seed: u64, codes: YCbCrCodes) -> Vec<u8> {
    let mut random = Random(seed);
    let mut bytes = vec![0xa5; (2 * ROWQ * H) as usize];
    let [cw, ch] = chroma_size(codes);
    for y in 0..H {
        for x in 0..row_bytes(codes, W) {
            bytes[(y * ROWQ + x) as usize] = random.next();
        }
    }
    for y in 0..ch {
        for x in 0..row_bytes(codes, 2 * cw) {
            bytes[(CHROMAQ_AT + y * ROWQ + x) as usize] = random.next();
        }
    }
    bytes
}

/// The bytes of a row of `n` codes of `codes`: `n` units, or `⌈n / 3⌉` packed.
fn row_bytes(codes: YCbCrCodes, n: u64) -> u64 {
    codes.units(n) * u64::from(codes.bytes)
}

/// Code `index` of `codes` of the row of `bytes` whose first byte is at `row`: a byte, a 16-bit
/// word's high bits, or a packed word's code at bits 0, 10 or 20.
fn code_of(bytes: &[u8], codes: YCbCrCodes, row: u64, index: u64) -> u16 {
    let (at, shift) = match codes.packed {
        true => (row + 4 * (index / 3), 10 * (index % 3)),
        false => (
            row + u64::from(codes.bytes) * index,
            u64::from(8 * codes.bytes - codes.bits),
        ),
    };
    let at = at as usize;
    let unit = &bytes[at..at + codes.bytes as usize];
    let word = unit
        .iter()
        .rev()
        .fold(0u32, |word, &b| (word << 8) | u32::from(b));
    ((word >> shift) & ((1 << codes.bits) - 1)) as u16
}

/// The chroma plane's pairs across and down of a `W x H` texture of `codes`.
fn chroma_size(codes: YCbCrCodes) -> [u64; 2] {
    let halved = codes.subsampling.halved();
    [(W, halved[0]), (H, halved[1])].map(|(n, halved)| if halved { n.div_ceil(2) } else { n })
}

/// Each texel's luma code and the sums `S` of its chroma codes Cb and Cr, `codes` of the surface
/// `bytes` (laid out as [`family_bytes`] lays them): along a halved axis the two nearest chroma
/// pairs weighted 3 and 1, along a direct one the texel's own pair weighted 4, 16 in all.
fn family_sums(bytes: &[u8], codes: YCbCrCodes) -> Vec<(u16, [u32; 2])> {
    let code = |row: u64, index: u64| code_of(bytes, codes, row, index);
    let [cw, ch] = chroma_size(codes);
    let halved = codes.subsampling.halved();
    let taps = |q: u64, n: u64, halved: bool| -> [(u64, u32); 2] {
        let c = q / 2;
        if !halved {
            [(q, 4), (q, 0)]
        } else if q % 2 == 1 {
            [(c, 3), ((c + 1).min(n - 1), 1)]
        } else {
            [(c.saturating_sub(1), 1), (c, 3)]
        }
    };
    let mut out = Vec::new();
    for y in 0..H {
        for x in 0..W {
            let mut sums = [0u32; 2];
            for (cy, wy) in taps(y, ch, halved[1]) {
                for (cx, wx) in taps(x, cw, halved[0]) {
                    let row = CHROMAQ_AT + cy * ROWQ;
                    sums[0] += wx * wy * u32::from(code(row, 2 * cx));
                    sums[1] += wx * wy * u32::from(code(row, 2 * cx + 1));
                }
            }
            out.push((code(y * ROWQ, x), sums));
        }
    }
    out
}

/// Each texel `codes` reads of the surface `bytes` (laid out as [`family_bytes`] lays them)
/// under `conversion` (`metal2vulkan::bind::ycbcr_read`): [`family_sums`]' chroma sums rounded
/// `(S + 8) >> 4`.
fn family_modelled(
    bytes: &[u8],
    codes: YCbCrCodes,
    conversion: (YCbCrRange, YCbCrMatrix),
) -> Vec<[u16; 3]> {
    family_sums(bytes, codes)
        .into_iter()
        .map(|(y, sums)| {
            let [cb, cr] = sums.map(|s| ((s + 8) >> 4) as u16);
            ycbcr_read(codes, conversion, [y, cb, cr])
        })
        .collect()
}

/// Each texel of the surface `bytes` (laid out as [`family_bytes`] lays them) through the
/// device's conversion of `codes` under `conversion` (`ycbcr_device_reads::vulkan_read`):
/// [`family_sums`]' chroma sums over 16, unrounded.
fn family_vulkan(
    bytes: &[u8],
    codes: YCbCrCodes,
    conversion: (YCbCrRange, YCbCrMatrix),
) -> Vec<[f64; 3]> {
    family_sums(bytes, codes)
        .into_iter()
        .map(|(y, sums)| {
            let [cb, cr] = sums.map(|s| f64::from(s) / 16.0);
            vulkan_read(codes, conversion, [f64::from(y), cb, cr])
        })
        .collect()
}

/// Asserts a `read()` (or a sample at the centres, or, `half`, a `half` read) of a texture of
/// `codes` over the surface `bytes` (laid out as [`family_bytes`] lays them) under `conversion`:
/// Metal's steps of `1 / grid` exactly ([`family_modelled`]), or, where the device converted the
/// texture (`by_device`), Vulkan's conversion within the device's bound ([`family_vulkan`],
/// `ycbcr_device_reads`). A `half` read of Metal's steps is [`assert_half_steps`]'s.
fn assert_family_read(
    kernels: &Kernels,
    texels: &[u8],
    (bytes, codes, conversion): (&[u8], YCbCrCodes, (YCbCrRange, YCbCrMatrix)),
    (grid, half, by_device): (u32, bool, bool),
    case: &str,
) {
    match (by_device, half) {
        (true, _) => {
            let want = family_vulkan(bytes, codes, conversion);
            assert_device_reads(texels, &want, (codes.output, half), case);
        }
        (false, true) => {
            let want = family_modelled(bytes, codes, conversion);
            assert_half_steps(kernels, texels, &want, grid, case);
        }
        (false, false) => {
            let want = family_modelled(bytes, codes, conversion);
            assert_eq!(grid_of(texels, grid), want, "{case}");
        }
    }
}

/// The texture of `pf` over plane 0 of `surface`, a surface of OSType `ostype` and `matrix` laid
/// out by [`ROWQ`] and [`CHROMAQ_AT`], of shader read usage.
fn family_texture(
    surface: &Buffer,
    pf: PixelFormat,
    ostype: &str,
    matrix: Option<SurfaceYCbCrMatrix>,
) -> Result<Texture, Error> {
    let mut d = TextureDescriptor::texture_2d(pf, W, H, false);
    d.usage = TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::shared();
    let code: [u8; 4] = ostype.as_bytes().try_into().unwrap();
    let facts = IoSurfaceYCbCr::new(u32::from_be_bytes(code), 0, [W, H], CHROMAQ_AT, ROWQ);
    let facts = match matrix {
        Some(matrix) => facts.with_matrix(matrix),
        None => facts,
    };
    let at = IoSurfacePlane::new(0, ROWQ, true).with_ycbcr(facts);
    surface.new_texture_with_iosurface_plane(&d, at)
}

/// The 4:2:2 and 4:4:4 formats over their own surfaces read what Metal reads (`ycbcr_read` states
/// it): `YCBCR8_422_2P`, `YCBCR8_444_2P`, `YCBCR10_422_2P`, `YCBCR10_444_2P` and their `_sRGB`
/// twins over `422v`, `422f`, `444v`, `444f`, `x422`, `xf22`, `x444` and `xf44`, without a matrix,
/// with BT.709, with BT.2020 and with SMPTE 240M (read as BT.709): `read()`, a nearest and a linear
/// sample at each texel's centre, and a `half` read (the twins' steps rounded with ties away from
/// zero). The filter notices are [`assert_filter_notices`]'s: no function reads these planes in
/// place. A format over a surface whose chroma plane holds its own reads its pairs at the
/// surface's rows, and over one whose plane is smaller the same, past the plane. A texture the
/// device converts itself reads Vulkan's conversion of the same codes instead, within the device's
/// bound (`ycbcr_device_reads`; lavapipe's 8-bit 4:4:4 ones).
#[test]
fn four_two_two_and_four_four_four_textures_read_what_metal_reads() {
    on_device(|device, messages| {
        let kernels = Kernels::new(device);
        let nearest = sampler(device, SamplerMinMagFilter::Nearest);
        let linear = sampler(device, SamplerMinMagFilter::Linear);
        let mut steps = Between::new(device);
        let mut seed = 120;
        let mut by_devices = 0;
        for (pf, ostypes) in [
            (PixelFormat::YCBCR84222P, ["422v", "422f"]),
            (PixelFormat::YCBCR84442P, ["444v", "444f"]),
            (PixelFormat::YCBCR104222P, ["x422", "xf22"]),
            (PixelFormat::YCBCR104442P, ["x444", "xf44"]),
            (PixelFormat::YCBCR84222PSRGB, ["422v", "422f"]),
            (PixelFormat::YCBCR84442PSRGB, ["444v", "444f"]),
            (PixelFormat::YCBCR104222PSRGB, ["x422", "xf22"]),
            (PixelFormat::YCBCR104442PSRGB, ["x444", "xf44"]),
        ] {
            let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            let grid = match codes.output {
                YCbCrOutput::Srgb => 8192,
                _ => 2048,
            };
            for (ostype, range) in ostypes
                .into_iter()
                .zip([YCbCrRange::Video, YCbCrRange::Full])
            {
                for (matrix, modelled) in [
                    (None, YCbCrMatrix::Bt601),
                    (Some(SurfaceYCbCrMatrix::Itu709), YCbCrMatrix::Bt709),
                    (Some(SurfaceYCbCrMatrix::Itu2020), YCbCrMatrix::Bt2020),
                    (Some(SurfaceYCbCrMatrix::Smpte240M), YCbCrMatrix::Bt709),
                ] {
                    seed += 1;
                    let bytes = family_bytes(seed, codes);
                    let surface = buffer_with(device, &bytes).unwrap();
                    let before = made_in_place(device);
                    let converted = device_conversions(device);
                    let texture = family_texture(&surface, pf, ostype, matrix).unwrap();
                    steps.sampled(made_in_place(device) > before);
                    let by_device = device_conversions(device) > converted;
                    by_devices += usize::from(by_device);
                    let case = format!("{pf:?} over {ostype}, {matrix:?}");
                    let surface = (&bytes[..], codes, (range, modelled));
                    let read = texels_of(device, &kernels.read, &texture, None);
                    let check = |texels: &[u8], half, what: &str| {
                        let case = format!("{case}: {what}");
                        assert_family_read(
                            &kernels,
                            texels,
                            surface,
                            (grid, half, by_device),
                            &case,
                        );
                    };
                    check(&read, false, "read()");
                    for (filter, state) in [("nearest", &nearest), ("linear", &linear)] {
                        let sampled = texels_of(device, &kernels.sample, &texture, Some(state));
                        check(&sampled, false, &format!("a {filter} sample"));
                    }
                    let half = texels_of(device, &kernels.read_half, &texture, None);
                    check(&half, true, "a half read()");
                }
            }
        }
        println!("YCBCR-DEVICE 4:2:2 and 4:4:4: {by_devices} textures the device's conversion");
        steps.finish(device, &counts(&messages.take_notices()), "4:2:2 and 4:4:4");
        // Over a surface whose chroma plane holds the format's (for 4:2:2 over
        // 4:4:4): the format's pairs, at the surface's rows.
        for (seed, (pf, ostype, range)) in (160..).zip([
            (PixelFormat::YCBCR84222P, "444v", YCbCrRange::Video),
            (PixelFormat::YCBCR104222PSRGB, "xf44", YCbCrRange::Full),
            (PixelFormat::YCBCR84202P, "422f", YCbCrRange::Full),
            (PixelFormat::YCBCR84202PSRGB, "444v", YCbCrRange::Video),
        ]) {
            let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            let bytes = family_bytes(seed, codes);
            let surface = buffer_with(device, &bytes).unwrap();
            let converted = device_conversions(device);
            let texture = family_texture(&surface, pf, ostype, None).unwrap();
            let by_device = device_conversions(device) > converted;
            let grid = match codes.output {
                YCbCrOutput::Srgb => 8192,
                _ => 2048,
            };
            let read = texels_of(device, &kernels.read, &texture, None);
            let surface = (&bytes[..], codes, (range, YCbCrMatrix::Bt601));
            let case = format!("{pf:?} over {ostype}");
            assert_family_read(&kernels, &read, surface, (grid, false, by_device), &case);
        }
        // Over a surface whose chroma plane is smaller than the format's: the format's pairs at
        // the surface's rows all the same, past the plane where it ends.
        for (pf, ostype) in [
            (PixelFormat::YCBCR84222P, "420v"),
            (PixelFormat::YCBCR84442P, "422v"),
            (PixelFormat::YCBCR104222P, "x420"),
            (PixelFormat::YCBCR104442P, "x422"),
        ] {
            let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            let bytes = family_bytes(1, codes);
            let surface = buffer_with(device, &bytes).unwrap();
            let converted = device_conversions(device);
            let texture = family_texture(&surface, pf, ostype, None)
                .unwrap_or_else(|e| panic!("{pf:?} over {ostype}: {e:?}"));
            let by_device = device_conversions(device) > converted;
            let read = texels_of(device, &kernels.read, &texture, None);
            let surface = (&bytes[..], codes, (YCbCrRange::Video, YCbCrMatrix::Bt601));
            let case = format!("{pf:?} over {ostype}");
            assert_family_read(&kernels, &read, surface, (2048, false, by_device), &case);
        }
    });
}

/// A 4:2:2 or 4:4:4 texture's view as its `_sRGB` twin reads what a texture of the twin over the
/// same planes reads, as Metal's view does (made with `ShaderRead` alone), and the
/// texture itself its own conversion: Metal's, or where the device converts it, the device's
/// (`ycbcr_device_reads`).
#[test]
fn four_two_two_and_four_four_four_views_read_their_twins_texels() {
    on_device(|device, _| {
        let kernels = Kernels::new(device);
        for (seed, (pf, twin, ostype)) in (140..).zip([
            (
                PixelFormat::YCBCR84222P,
                PixelFormat::YCBCR84222PSRGB,
                "422f",
            ),
            (
                PixelFormat::YCBCR84442P,
                PixelFormat::YCBCR84442PSRGB,
                "444f",
            ),
            (
                PixelFormat::YCBCR104222P,
                PixelFormat::YCBCR104222PSRGB,
                "xf22",
            ),
            (
                PixelFormat::YCBCR104442P,
                PixelFormat::YCBCR104442PSRGB,
                "xf44",
            ),
        ]) {
            let codes = |pf: PixelFormat| ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            let bytes = family_bytes(seed, codes(pf));
            let surface = buffer_with(device, &bytes).unwrap();
            let bt709 = (YCbCrRange::Full, YCbCrMatrix::Bt709);
            let matrix = Some(SurfaceYCbCrMatrix::Itu709);
            // Each texture, and whether the device converted it as it was made.
            let made = |make: &dyn Fn() -> Texture| {
                let converted = device_conversions(device);
                let texture = make();
                (texture, device_conversions(device) > converted)
            };
            let base = made(&|| family_texture(&surface, pf, ostype, matrix).unwrap());
            let view = made(&|| base.0.new_texture_view(twin).unwrap());
            let direct = made(&|| family_texture(&surface, twin, ostype, matrix).unwrap());
            for ((texture, by_device), of, grid, case) in [
                (&view, twin, 8192, format!("{pf:?}'s view")),
                (&direct, twin, 8192, format!("{twin:?}")),
                (&base, pf, 2048, format!("{pf:?}")),
            ] {
                let read = texels_of(device, &kernels.read, texture, None);
                let surface = (&bytes[..], codes(of), bt709);
                assert_family_read(&kernels, &read, surface, (grid, false, *by_device), &case);
            }
        }
    });
}

/// A 4:2:2 or 4:4:4 texture's copies move its luma codes, a code's bytes a texel, as Metal's do:
/// `get_bytes` reads the luma rows, and `replace_region` writes them, keeping
/// the chroma plane, the texture then reading the new codes' conversion.
#[test]
fn four_two_two_and_four_four_four_textures_copy_their_luma_codes() {
    on_device(|device, _| {
        let kernels = Kernels::new(device);
        for (seed, (pf, ostype)) in (180..).step_by(2).zip([
            (PixelFormat::YCBCR84222P, "422v"),
            (PixelFormat::YCBCR104442P, "x444"),
            (PixelFormat::YCBCR104442PSRGB, "x444"),
        ]) {
            let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            let row = u64::from(codes.bytes) * W;
            let bytes = family_bytes(seed, codes);
            let surface = buffer_with(device, &bytes).unwrap();
            let texture = family_texture(&surface, pf, ostype, None).unwrap();
            let mut luma = vec![0u8; (row * H) as usize];
            texture
                .get_bytes(&mut luma, row, 0, Region::new_2d(0, 0, W, H), 0, 0)
                .unwrap();
            let rows: Vec<u8> = (0..H)
                .flat_map(|y| bytes[(y * ROWQ) as usize..(y * ROWQ + row) as usize].to_vec())
                .collect();
            assert_eq!(luma, rows, "{pf:?}: get_bytes");
            let mut random = Random(seed + 1);
            let new: Vec<u8> = (0..row * H).map(|_| random.next()).collect();
            texture
                .replace_region(Region::new_2d(0, 0, W, H), 0, 0, &new, row, 0)
                .unwrap();
            let mut want = bytes.clone();
            for y in 0..H {
                let at = (y * ROWQ) as usize;
                want[at..at + row as usize]
                    .copy_from_slice(&new[(y * row) as usize..((y + 1) * row) as usize]);
            }
            let length = want.len() as u64;
            let written = read_buffer(&surface, 0, length).unwrap();
            assert_eq!(written, want, "{pf:?}: replace_region");
            let grid = match codes.output {
                YCbCrOutput::Srgb => 8192,
                _ => 2048,
            };
            let read = grid_of(&texels_of(device, &kernels.read, &texture, None), grid);
            let modelled = family_modelled(&want, codes, (YCbCrRange::Video, YCbCrMatrix::Bt601));
            assert_eq!(read, modelled, "{pf:?}: read() after replace_region");
        }
    });
}

/// What the gather kernel `state` reads of `texture` through `sampler` at the `(W + 1) x (H + 1)`
/// texel corners, a `float4` each, in one command buffer of its own.
fn gathered_of(
    device: &Device,
    state: &ComputePipelineState,
    texture: &Texture,
    sampler: &SamplerState,
) -> Vec<u8> {
    let length = 16 * (W + 1) * (H + 1);
    let out = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    dispatch(
        &command_buffer,
        state,
        texture,
        &out,
        Some(sampler),
        [W + 1, H + 1],
    );
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    read_buffer(&out, 0, length).unwrap()
}

/// Whether the step `step / 8192` lies halfway between two `half`s, where nearest even and ties
/// away part.
fn is_half_tie(step: u16) -> bool {
    let v = f64::from(step) / 8192.0;
    step != 0 && (v / 2f64.powi(v.log2().floor() as i32 - 10)).fract() == 0.5
}

/// A `half` gather of a two-plane YCbCr texture rounds each lane with ties away from zero, as
/// Metal's does: red and blue of `YCBCR8_420_2P_sRGB` over `420v` without a matrix, with BT.709 and
/// with BT.2020, `YCBCR10_420_2P_sRGB` over `x420`, the 4:2:2 and 4:4:4 twins, the packed 4:2:0 and
/// 4:4:4 twins, and `YCBCR8_420_2P`, at every texel corner under clamp-to-edge, clamp-to-zero and
/// clamp-to-border-color: each lane the `float` gather's step of `1/8192` rounded so, ties among
/// them. Where the stage rounds a `half` by the device's conversion (`Kernels::native_half`) the
/// lanes that differ are counted and printed. A texture the device converts itself lies off
/// Metal's steps (`ycbcr_device_reads`): each `half` lane is its `float` lane's `half`, within one
/// unit in the last place.
#[test]
fn ycbcr_half_gathers_round_ties_away() {
    on_device(|device, _| {
        let kernels = Kernels::new(device);
        let usage = TextureUsage::SHADER_READ;
        let samplers = [
            SamplerAddressMode::ClampToEdge,
            SamplerAddressMode::ClampToZero,
            SamplerAddressMode::ClampToBorderColor,
        ]
        .map(|mode| {
            let mut d = SamplerDescriptor::new();
            d.s_address_mode = mode;
            d.t_address_mode = mode;
            (mode, device.new_sampler_state(&d).unwrap())
        });
        let mut textures: Vec<(PixelFormat, String, Texture, bool)> = Vec::new();
        for (seed, (pf, matrix)) in (200..).zip([
            (PixelFormat::YCBCR84202PSRGB, None),
            (
                PixelFormat::YCBCR84202PSRGB,
                Some(SurfaceYCbCrMatrix::Itu709),
            ),
            (
                PixelFormat::YCBCR84202PSRGB,
                Some(SurfaceYCbCrMatrix::Itu2020),
            ),
            (PixelFormat::YCBCR84202P, Some(SurfaceYCbCrMatrix::Itu709)),
        ]) {
            let surface = buffer_with(device, &surface_bytes(seed)).unwrap();
            let converted = device_conversions(device);
            let texture = eight_bit_texture(&surface, pf, Some(facts(VIDEO, matrix)), usage);
            let by_device = device_conversions(device) > converted;
            textures.push((pf, format!("420v {matrix:?}"), texture.unwrap(), by_device));
        }
        let ten = buffer_with(device, &ten_bit_bytes(210)).unwrap();
        let pf = PixelFormat::YCBCR104202PSRGB;
        let matrix = Some(SurfaceYCbCrMatrix::Itu709);
        let converted = device_conversions(device);
        let texture = ten_bit_texture(&ten, pf, (b"x420", 0), matrix).unwrap();
        let by_device = device_conversions(device) > converted;
        textures.push((pf, "x420".into(), texture, by_device));
        for (seed, (pf, ostype)) in (220..).zip([
            (PixelFormat::YCBCR84222PSRGB, "422v"),
            (PixelFormat::YCBCR84442PSRGB, "444f"),
            (PixelFormat::YCBCR104222PSRGB, "x422"),
            (PixelFormat::YCBCR104442PSRGB, "xf44"),
            (PixelFormat::YCBCR104202PPACKEDSRGB, "p420"),
            (PixelFormat::YCBCR104442PPACKEDSRGB, "pf44"),
        ]) {
            let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            let surface = buffer_with(device, &family_bytes(seed, codes)).unwrap();
            let converted = device_conversions(device);
            let texture = family_texture(&surface, pf, ostype, None).unwrap();
            let by_device = device_conversions(device) > converted;
            textures.push((pf, ostype.into(), texture, by_device));
        }
        let (mut ties, mut rounded) = (0, 0usize);
        for (pf, over, texture, by_device) in &textures {
            for (mode, state) in &samplers {
                for (channel, [float, half]) in ["red", "blue"].iter().zip(&kernels.gathers) {
                    let floats = gathered_of(device, float, texture, state);
                    let halves = gathered_of(device, half, texture, state);
                    let lanes = floats
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .zip(halves.as_chunks::<4>().0.iter());
                    for (i, (f, h)) in lanes.enumerate() {
                        let [f, h] = [f, h].map(|b| f32::from_le_bytes(*b));
                        let case = format!("{pf:?} over {over}, {mode:?}, {channel} lane {i}");
                        if *by_device {
                            // The device's conversion lies off Metal's steps: the `half` gather
                            // is the `float` one's `half` (`PrecisionNote::NativeHalfConversion`).
                            let nearest = f64::from(f);
                            let exponent = match nearest.abs() {
                                v if v < 2f64.powi(-14) => -24,
                                v => v.log2().floor() as i32 - 10,
                            };
                            let off = (f64::from(h) - nearest).abs();
                            assert!(off <= 2f64.powi(exponent), "{case}: {h}, the float {f}");
                            continue;
                        }
                        let step = f * 8192.0;
                        assert!(
                            step.fract() == 0.0 && (0.0..=8192.0).contains(&step),
                            "{case}: {f}"
                        );
                        let step = step as u16;
                        if !kernels.native_half {
                            assert_eq!(h, half_away(step), "{case}: step {step}");
                        } else if h != half_away(step) {
                            rounded += 1;
                        }
                        ties += usize::from(is_half_tie(step));
                    }
                }
            }
        }
        assert!(ties > 0, "no lane lay halfway between two halves");
        if rounded > 0 {
            println!("YCBCR-HALF-DIFFERS gathers: {rounded} half lanes the device's rounding");
        }
    });
}

/// The 4:2:2 and 4:4:4 code twins read their codes as Metal does (every
/// texel of 24 surfaces; `ycbcr_read` states it): `RGB8_422_2P`, `RGB8_444_2P`, `RGB10_422_2P`
/// and `RGB10_444_2P` over their own surfaces (`422v`, `422f`, `444v`, `444f`, `x422`, `xf22`,
/// `x444`, `xf44`), and the 4:2:2 ones over a 4:4:4 surface (whose plane theirs fits in), without
/// a matrix, with BT.709 and with BT.2020: red Cr, green Y, blue Cb, each `code / (2^b - 1)`,
/// alpha 1, after the format's chroma filter, whatever the matrix: `read()`, and a nearest and a
/// linear sample at each texel's centre. No filter notice. Over a surface whose chroma plane
/// theirs does not fit in, their pairs at the surface's rows, past the plane.
#[test]
fn four_two_two_and_four_four_four_codes_read_what_metal_reads() {
    on_device(|device, messages| {
        let kernels = Kernels::new(device);
        let nearest = sampler(device, SamplerMinMagFilter::Nearest);
        let linear = sampler(device, SamplerMinMagFilter::Linear);
        let mut seed = 160;
        for (pf, ostypes) in [
            (PixelFormat::RGB84222P, &["422v", "422f", "444v"][..]),
            (PixelFormat::RGB84442P, &["444v", "444f"]),
            (PixelFormat::RGB104222P, &["x422", "xf22", "x444"]),
            (PixelFormat::RGB104442P, &["x444", "xf44"]),
        ] {
            let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            assert_eq!(codes.output, YCbCrOutput::Codes, "{pf:?}");
            let max = (1u16 << codes.bits) - 1;
            for &ostype in ostypes {
                for matrix in [
                    None,
                    Some(SurfaceYCbCrMatrix::Itu709),
                    Some(SurfaceYCbCrMatrix::Itu2020),
                ] {
                    seed += 1;
                    let bytes = family_bytes(seed, codes);
                    let surface = buffer_with(device, &bytes).unwrap();
                    let texture = family_texture(&surface, pf, ostype, matrix).unwrap();
                    // The codes, which no range or matrix changes.
                    let want =
                        family_modelled(&bytes, codes, (YCbCrRange::Video, YCbCrMatrix::Bt601));
                    let case = format!("{pf:?} over {ostype}, {matrix:?}");
                    let read = code_grid(
                        &kernels,
                        &texels_of(device, &kernels.read, &texture, None),
                        max,
                    );
                    assert_eq!(read, want, "{case}: read()");
                    for (filter, state) in [("nearest", &nearest), ("linear", &linear)] {
                        let sampled = texels_of(device, &kernels.sample, &texture, Some(state));
                        assert_eq!(
                            code_grid(&kernels, &sampled, max),
                            want,
                            "{case}: a {filter} sample"
                        );
                    }
                }
            }
        }
        let notices = counts(&messages.take_notices());
        for kind in [
            Degradation::FloatSampleFilter,
            Degradation::YCbCrFilterNotExact,
        ] {
            assert_eq!(notices.get(&kind), None, "{notices:?}");
        }
        // Over a surface whose chroma plane theirs does not fit in: their pairs at the surface's
        // rows all the same, past the plane where it ends.
        for (pf, ostype) in [
            (PixelFormat::RGB84222P, "420v"),
            (PixelFormat::RGB84442P, "422v"),
            (PixelFormat::RGB104222P, "x420"),
            (PixelFormat::RGB104442P, "x422"),
        ] {
            let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            let bytes = family_bytes(1, codes);
            let surface = buffer_with(device, &bytes).unwrap();
            let texture = family_texture(&surface, pf, ostype, None)
                .unwrap_or_else(|e| panic!("{pf:?} over {ostype}: {e:?}"));
            let max = (1u16 << codes.bits) - 1;
            let read = code_grid(
                &kernels,
                &texels_of(device, &kernels.read, &texture, None),
                max,
            );
            let want = family_modelled(&bytes, codes, (YCbCrRange::Video, YCbCrMatrix::Bt601));
            assert_eq!(read, want, "{pf:?} over {ostype}");
        }
    });
}

/// A texture over plane 1 of a two-plane surface reads plane 1's codes as its luma plane, at
/// plane 1's rows, with chroma codes 0, as Metal does (every one of the 18 formats over plane 1
/// of `422v`, `422f`, `444v`, `444f`, `x422`, `xf22`, `x444` and `xf44`, 0 texels off that
/// model, and our probe `ycr`'s scan of every plane): each format over each such surface of its
/// bits and over `420v`, `420f`, `x420` and `xf20` (whose plane 1 the rows here hold), without a
/// matrix and with BT.709, reads (`read()`, a nearest sample, and the converted ones' `half`
/// read) what it reads over plane 0 of a surface whose luma rows hold plane 1's codes and whose
/// chroma plane holds 0; `get_bytes` moves plane 1's codes. Over plane 1 at other bytes than the
/// facts name it is refused, naming the format.
#[test]
fn ycbcr_textures_over_plane_1_read_its_codes_as_luma() {
    on_device(|device, _| {
        let kernels = Kernels::new(device);
        let nearest = sampler(device, SamplerMinMagFilter::Nearest);
        let over_plane_1 = |surface: &Buffer,
                            pf: PixelFormat,
                            ostype: &str,
                            plane: u64,
                            at: u64,
                            matrix: Option<SurfaceYCbCrMatrix>| {
            let mut d = TextureDescriptor::texture_2d(pf, W, H, false);
            d.usage = TextureUsage::SHADER_READ;
            d.resource_options = ResourceOptions::shared();
            let code: [u8; 4] = ostype.as_bytes().try_into().unwrap();
            let facts = IoSurfaceYCbCr::new(u32::from_be_bytes(code), plane, [W, H], at, ROWQ);
            let facts = match matrix {
                Some(matrix) => facts.with_matrix(matrix),
                None => facts,
            };
            let plane = IoSurfacePlane::new(CHROMAQ_AT, ROWQ, true).with_ycbcr(facts);
            surface.new_texture_with_iosurface_plane(&d, plane)
        };
        let mut seed = 300;
        for pf in [
            PixelFormat::YCBCR84202P,
            PixelFormat::YCBCR84222P,
            PixelFormat::YCBCR84442P,
            PixelFormat::YCBCR84202PSRGB,
            PixelFormat::YCBCR84222PSRGB,
            PixelFormat::YCBCR84442PSRGB,
            PixelFormat::RGB84202P,
            PixelFormat::RGB84222P,
            PixelFormat::RGB84442P,
            PixelFormat::YCBCR104202P,
            PixelFormat::YCBCR104222P,
            PixelFormat::YCBCR104442P,
            PixelFormat::YCBCR104202PSRGB,
            PixelFormat::YCBCR104222PSRGB,
            PixelFormat::YCBCR104442PSRGB,
            PixelFormat::RGB104202P,
            PixelFormat::RGB104222P,
            PixelFormat::RGB104442P,
        ] {
            let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            let unit = u64::from(codes.bytes);
            let ostypes = match codes.bits {
                8 => ["422v", "422f", "444v", "444f", "420v", "420f"],
                _ => ["x422", "xf22", "x444", "xf44", "x420", "xf20"],
            };
            for (ostype, range) in ostypes.into_iter().zip([
                YCbCrRange::Video,
                YCbCrRange::Full,
                YCbCrRange::Video,
                YCbCrRange::Full,
                YCbCrRange::Video,
                YCbCrRange::Full,
            ]) {
                for (matrix, modelled) in [
                    (None, YCbCrMatrix::Bt601),
                    (Some(SurfaceYCbCrMatrix::Itu709), YCbCrMatrix::Bt709),
                ] {
                    seed += 1;
                    let mut random = Random(seed);
                    let bytes: Vec<u8> = (0..2 * ROWQ * H).map(|_| random.next()).collect();
                    let surface = buffer_with(device, &bytes).unwrap();
                    let texture =
                        over_plane_1(&surface, pf, ostype, 1, CHROMAQ_AT, matrix).unwrap();
                    // Plane 1's codes as the luma rows, the chroma plane 0.
                    let mut moved = vec![0; bytes.len()];
                    let row = (unit * W) as usize;
                    for y in 0..H as usize {
                        let from = CHROMAQ_AT as usize + y * ROWQ as usize;
                        let to = y * ROWQ as usize;
                        moved[to..to + row].copy_from_slice(&bytes[from..from + row]);
                    }
                    let case = format!("{pf:?} over plane 1 of {ostype}, {matrix:?}");
                    let (want, grid) = match codes.output {
                        YCbCrOutput::Codes => (
                            family_modelled(&moved, codes, (YCbCrRange::Video, YCbCrMatrix::Bt601)),
                            None,
                        ),
                        YCbCrOutput::Srgb => (
                            family_modelled(&moved, codes, (range, modelled)),
                            Some(8192),
                        ),
                        _ => (
                            family_modelled(&moved, codes, (range, modelled)),
                            Some(2048),
                        ),
                    };
                    let of = |texels: &[u8]| match grid {
                        Some(d) => grid_of(texels, d),
                        None => code_grid(&kernels, texels, (1u16 << codes.bits) - 1),
                    };
                    let read = texels_of(device, &kernels.read, &texture, None);
                    assert_eq!(of(&read), want, "{case}: read()");
                    let sampled = texels_of(device, &kernels.sample, &texture, Some(&nearest));
                    assert_eq!(of(&sampled), want, "{case}: a nearest sample");
                    if let Some(d) = grid {
                        let half = texels_of(device, &kernels.read_half, &texture, None);
                        assert_half_steps(
                            &kernels,
                            &half,
                            &want,
                            d,
                            &format!("{case}: a half read()"),
                        );
                    }
                    let mut luma = vec![0u8; row * H as usize];
                    texture
                        .get_bytes(&mut luma, row as u64, 0, Region::new_2d(0, 0, W, H), 0, 0)
                        .unwrap();
                    let codes_of_plane_1: Vec<u8> = (0..H as usize)
                        .flat_map(|y| moved[y * ROWQ as usize..][..row].to_vec())
                        .collect();
                    assert!(luma == codes_of_plane_1, "{case}: get_bytes");
                }
            }
        }
        // Refused: plane 1 at other bytes than the facts'.
        let surface = buffer_with(device, &vec![0; (2 * ROWQ * H) as usize]).unwrap();
        let (pf, ostype, plane, at) = (PixelFormat::YCBCR84442P, "444v", 1, CHROMAQ_AT + ROWQ);
        let made = over_plane_1(&surface, pf, ostype, plane, at, None);
        let named = format!("{:?}", made.as_ref().err());
        assert!(
            matches!(made, Err(Error::Unsupported(_))) && named.contains(pf.name()),
            "{pf:?} over plane {plane} of {ostype} at {at}: {named}"
        );
    });
}

/// The packed formats, each with its own surfaces of video and of full range.
const PACKED: [(PixelFormat, [&str; 2]); 6] = [
    (PixelFormat::YCBCR104202PPACKED, ["p420", "pf20"]),
    (PixelFormat::YCBCR104222PPACKED, ["p422", "pf22"]),
    (PixelFormat::YCBCR104442PPACKED, ["p444", "pf44"]),
    (PixelFormat::YCBCR104202PPACKEDSRGB, ["p420", "pf20"]),
    (PixelFormat::YCBCR104222PPACKEDSRGB, ["p422", "pf22"]),
    (PixelFormat::YCBCR104442PPACKEDSRGB, ["p444", "pf44"]),
];

/// The packed formats over their own surfaces read what Metal reads (`ycbcr_read` states it):
/// `YCBCR10_420_2P_PACKED`, `YCBCR10_422_2P_PACKED`, `YCBCR10_444_2P_PACKED` and their `_sRGB`
/// twins over `p420`, `pf20`, `p422`, `pf22`, `p444` and `pf44`, three codes a 32-bit word, without
/// a matrix, with BT.709 and with BT.2020: `read()`, a nearest and a linear sample at each texel's
/// centre, a `half` read (the twins' steps rounded with ties away from zero), and a linear sample
/// between texels the filter of the kept image, whose planes no function reads in place
/// ([`Between`]; every case of the six formats). The filter notices are
/// [`assert_filter_notices`]'s. A 4:2:0 format
/// over a packed 4:4:4 surface reads its pairs at the surface's rows; a format over a surface whose
/// plane is smaller, a packed one over an unpacked surface and an unpacked one over a packed
/// surface read their own codes at the surface's rows by its range, and one over plane 1 of a
/// packed surface plane 1's codes as its luma with chroma codes 0.
#[test]
fn packed_textures_read_what_metal_reads() {
    on_device(|device, messages| {
        let kernels = Kernels::new(device);
        let nearest = sampler(device, SamplerMinMagFilter::Nearest);
        let linear = sampler(device, SamplerMinMagFilter::Linear);
        let mut steps = Between::new(device);
        let mut seed = 300;
        for (pf, ostypes) in PACKED {
            let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            assert!(codes.packed, "{pf:?}");
            let grid = match codes.output {
                YCbCrOutput::Srgb => 8192,
                _ => 2048,
            };
            for (ostype, range) in ostypes
                .into_iter()
                .zip([YCbCrRange::Video, YCbCrRange::Full])
            {
                for (matrix, modelled) in [
                    (None, YCbCrMatrix::Bt601),
                    (Some(SurfaceYCbCrMatrix::Itu709), YCbCrMatrix::Bt709),
                    (Some(SurfaceYCbCrMatrix::Itu2020), YCbCrMatrix::Bt2020),
                ] {
                    seed += 1;
                    let bytes = family_bytes(seed, codes);
                    let surface = buffer_with(device, &bytes).unwrap();
                    let before = made_in_place(device);
                    let texture = family_texture(&surface, pf, ostype, matrix).unwrap();
                    let in_place = made_in_place(device) > before;
                    let want = family_modelled(&bytes, codes, (range, modelled));
                    let case = format!("{pf:?} over {ostype}, {matrix:?}");
                    let of = |texels: &[u8]| grid_of(texels, grid);
                    let read = of(&texels_of(device, &kernels.read, &texture, None));
                    assert_eq!(read, want, "{case}: read()");
                    for (filter, state) in [("nearest", &nearest), ("linear", &linear)] {
                        let sampled = texels_of(device, &kernels.sample, &texture, Some(state));
                        assert_eq!(of(&sampled), want, "{case}: a {filter} sample");
                    }
                    let half = texels_of(device, &kernels.read_half, &texture, None);
                    assert_half_steps(
                        &kernels,
                        &half,
                        &want,
                        grid,
                        &format!("{case}: a half read()"),
                    );
                    let between = texels_of(device, &kernels.between, &texture, Some(&linear));
                    steps.texels(&between, &read, (grid, in_place), &case);
                }
            }
        }
        steps.finish(device, &counts(&messages.take_notices()), "packed textures");
        // Over a surface whose chroma plane holds the format's (4:2:0 over 4:2:2
        // and 4:4:4, 4:2:2 over 4:4:4): the format's pairs, at the surface's rows.
        for (seed, (pf, ostype, range)) in (340..).zip([
            (PixelFormat::YCBCR104202PPACKED, "p444", YCbCrRange::Video),
            (
                PixelFormat::YCBCR104202PPACKEDSRGB,
                "p422",
                YCbCrRange::Video,
            ),
            (PixelFormat::YCBCR104222PPACKED, "p444", YCbCrRange::Video),
        ]) {
            let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            let bytes = family_bytes(seed, codes);
            let surface = buffer_with(device, &bytes).unwrap();
            let texture = family_texture(&surface, pf, ostype, None).unwrap();
            let grid = match codes.output {
                YCbCrOutput::Srgb => 8192,
                _ => 2048,
            };
            let read = grid_of(&texels_of(device, &kernels.read, &texture, None), grid);
            let want = family_modelled(&bytes, codes, (range, YCbCrMatrix::Bt601));
            assert_eq!(read, want, "{pf:?} over {ostype}");
        }
        // Over a surface whose chroma plane is smaller than the format's, or whose codes are
        // held otherwise: the format's codes, in its own layout, at the surface's rows, by the
        // surface's range.
        for (pf, ostype, range) in [
            (PixelFormat::YCBCR104222PPACKED, "p420", YCbCrRange::Video),
            (
                PixelFormat::YCBCR104442PPACKEDSRGB,
                "pf22",
                YCbCrRange::Full,
            ),
            (PixelFormat::YCBCR104202PPACKED, "x420", YCbCrRange::Video),
            (PixelFormat::YCBCR104442PPACKED, "xf44", YCbCrRange::Full),
            (PixelFormat::YCBCR104202P, "p420", YCbCrRange::Video),
            (PixelFormat::YCBCR104442PSRGB, "pf44", YCbCrRange::Full),
        ] {
            let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            let bytes = family_bytes(1, codes);
            let surface = buffer_with(device, &bytes).unwrap();
            let texture = family_texture(&surface, pf, ostype, None)
                .unwrap_or_else(|e| panic!("{pf:?} over {ostype}: {e:?}"));
            let grid = match codes.output {
                YCbCrOutput::Srgb => 8192,
                _ => 2048,
            };
            let read = grid_of(&texels_of(device, &kernels.read, &texture, None), grid);
            let want = family_modelled(&bytes, codes, (range, YCbCrMatrix::Bt601));
            assert_eq!(read, want, "{pf:?} over {ostype}");
        }
        // Over plane 1 of a packed surface: plane 1's codes as the luma, chroma codes 0.
        let pf = PixelFormat::YCBCR104442PPACKED;
        let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
        let bytes = family_bytes(2, codes);
        let surface = buffer_with(device, &bytes).unwrap();
        let mut d = TextureDescriptor::texture_2d(pf, W, H, false);
        d.usage = TextureUsage::SHADER_READ;
        d.resource_options = ResourceOptions::shared();
        let facts = IoSurfaceYCbCr::new(u32::from_be_bytes(*b"p444"), 1, [W, H], CHROMAQ_AT, ROWQ);
        let plane = IoSurfacePlane::new(CHROMAQ_AT, ROWQ, true).with_ycbcr(facts);
        let texture = surface
            .new_texture_with_iosurface_plane(&d, plane)
            .unwrap_or_else(|e| panic!("{pf:?} over plane 1 of p444: {e:?}"));
        let mut moved = vec![0; bytes.len()];
        let row = row_bytes(codes, W) as usize;
        for y in 0..H as usize {
            let from = CHROMAQ_AT as usize + y * ROWQ as usize;
            moved[y * ROWQ as usize..][..row].copy_from_slice(&bytes[from..from + row]);
        }
        let read = grid_of(&texels_of(device, &kernels.read, &texture, None), 2048);
        let want = family_modelled(&moved, codes, (YCbCrRange::Video, YCbCrMatrix::Bt601));
        assert_eq!(read, want, "{pf:?} over plane 1 of p444");
    });
}

/// A packed texture's view as its `_sRGB` twin reads what a texture of the twin over the same
/// planes reads, as Metal's view does (for 4:2:0 and 4:2:2, made with `ShaderRead`
/// alone; 4:4:4 by the rule every measured pair follows), and the texture itself its own
/// conversion.
#[test]
fn packed_views_read_their_twins_texels() {
    on_device(|device, _| {
        let kernels = Kernels::new(device);
        for (seed, (pf, ostype)) in (350..).zip([
            (PixelFormat::YCBCR104202PPACKED, "pf20"),
            (PixelFormat::YCBCR104222PPACKED, "pf22"),
            (PixelFormat::YCBCR104442PPACKED, "pf44"),
        ]) {
            let twin = PACKED
                .iter()
                .map(|&(twin, _)| twin)
                .find(|&twin| {
                    let [a, b] = [pf, twin].map(|pf| ycbcr_format(pf).unwrap());
                    (a.subsampling, b.reads) == (b.subsampling, YCbCrReads::Srgb)
                })
                .unwrap();
            let codes = |pf: PixelFormat| ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            let bytes = family_bytes(seed, codes(pf));
            let surface = buffer_with(device, &bytes).unwrap();
            let bt709 = (YCbCrRange::Full, YCbCrMatrix::Bt709);
            let matrix = Some(SurfaceYCbCrMatrix::Itu709);
            let base = family_texture(&surface, pf, ostype, matrix).unwrap();
            let view = base.new_texture_view(twin).unwrap();
            let direct = family_texture(&surface, twin, ostype, matrix).unwrap();
            let read = |t: &Texture, d| grid_of(&texels_of(device, &kernels.read, t, None), d);
            let want = family_modelled(&bytes, codes(twin), bt709);
            assert_eq!(read(&view, 8192), want, "{pf:?}'s view");
            assert_eq!(read(&direct, 8192), want, "{twin:?}");
            assert_eq!(
                read(&base, 2048),
                family_modelled(&bytes, codes(pf), bt709),
                "{pf:?}"
            );
        }
    });
}

/// A two-plane YCbCr texture's view in its own format as a 2D array reads its luma codes alone,
/// as Metal's does (our probe `ycv`, the M3 Max under macOS 27.0.1, bit for bit over every texel):
/// `(Y, 0, 0, 1)` of a texture read converted, packed or through the sRGB curve (whose view
/// reads no defined value on Metal: its texels change from run to run), `(0, Y, 0, 1)` of one
/// read as codes, `Y` the code over `2^b - 1`; a swizzle applies to that read; and a linear
/// sample is the luma channel of the code-reading format's sample over the same planes. The
/// texture itself still reads its own conversion; a texture over plane 1 refuses the view.
#[test]
fn retyped_views_read_their_luma_codes() {
    use metal2vulkan::metal::TextureSwizzle as S;
    on_device(|device, _| {
        let kernels = Kernels::new(device);
        let linear = sampler(device, SamplerMinMagFilter::Linear);
        let array = |t: &Texture, swizzle: Option<TextureSwizzleChannels>| {
            let (pf, array) = (t.pixel_format(), TextureType::Type2DArray);
            match swizzle {
                None => t.new_texture_view_with_levels_slices(pf, array, 0..1, 0..1),
                Some(s) => t.new_texture_view_with_swizzle(pf, array, 0..1, 0..1, s),
            }
        };
        let floats = |t: &[u8]| -> Vec<[f32; 4]> {
            let channel = |t: &[u8; 16], i: usize| {
                f32::from_le_bytes(t[4 * i..4 * i + 4].try_into().unwrap())
            };
            let chunks = t.as_chunks::<16>().0;
            chunks
                .iter()
                .map(|t| std::array::from_fn(|i| channel(t, i)))
                .collect()
        };
        let swizzle = TextureSwizzleChannels {
            red: S::Blue,
            green: S::Red,
            blue: S::One,
            alpha: S::Green,
        };
        for (seed, (pf, ostype)) in (380..).zip([
            (PixelFormat::YCBCR84202P, "420v"),
            (PixelFormat::YCBCR84222P, "422f"),
            (PixelFormat::YCBCR84442P, "444v"),
            (PixelFormat::YCBCR104202P, "xf20"),
            (PixelFormat::YCBCR104222P, "x422"),
            (PixelFormat::YCBCR104442P, "x444"),
            (PixelFormat::RGB84202P, "420v"),
            (PixelFormat::RGB104442P, "xf44"),
            (PixelFormat::YCBCR104202PPACKED, "pf20"),
            (PixelFormat::YCBCR104442PPACKED, "p444"),
            (PixelFormat::YCBCR84202PSRGB, "420f"),
        ]) {
            let format = ycbcr_format(pf).unwrap();
            let codes = YCbCrCodes::of(format).unwrap();
            let bytes = family_bytes(seed, codes);
            let surface = buffer_with(device, &bytes).unwrap();
            let base = family_texture(&surface, pf, ostype, None).unwrap();
            let own = texels_of(device, &kernels.read, &base, None);
            let view = array(&base, None).unwrap();
            assert_eq!(view.pixel_format(), pf);
            assert_eq!(view.texture_type(), TextureType::Type2DArray);
            let max = (1u32 << codes.bits) - 1;
            let at_green = format.reads == YCbCrReads::Codes;
            let want: Vec<[u16; 3]> = (0..H)
                .flat_map(|y| (0..W).map(move |x| (y, x)))
                .map(|(y, x)| {
                    let luma = code_of(&bytes, codes, y * ROWQ, x);
                    match at_green {
                        true => [0, luma, 0],
                        false => [luma, 0, 0],
                    }
                })
                .collect();
            let read = texels_of(device, &kernels.read_array, &view, None);
            assert_eq!(
                code_grid(&kernels, &read, max as u16),
                want,
                "{pf:?}'s view"
            );
            // The swizzle (blue, red, one, green) over that read.
            let swizzled = array(&base, Some(swizzle)).unwrap();
            let mapped: Vec<[f32; 4]> = (floats(&read).iter())
                .map(|&[r, g, b, _]| [b, r, 1.0, g])
                .collect();
            let swizzled = floats(&texels_of(device, &kernels.read_array, &swizzled, None));
            assert_eq!(swizzled, mapped, "{pf:?}'s swizzled view");
            // The texture reads its own conversion still.
            assert_eq!(texels_of(device, &kernels.read, &base, None), own, "{pf:?}");
            // A linear sample: the code-reading format's over the same planes, its luma.
            if format.packed {
                continue;
            }
            let code_format = PixelFormat::ALL.iter().copied().find(|&c| {
                ycbcr_format(c).is_some_and(|f| {
                    f.reads == YCbCrReads::Codes
                        && !f.packed
                        && (f.bits, f.subsampling) == (format.bits, format.subsampling)
                })
            });
            let twin = family_texture(&surface, code_format.unwrap(), ostype, None).unwrap();
            let sampled = texels_of(device, &kernels.sample_array, &view, Some(&linear));
            let between = texels_of(device, &kernels.between, &twin, Some(&linear));
            let want: Vec<[f32; 4]> = (floats(&between).iter())
                .map(|&[_, y, _, _]| match at_green {
                    true => [0.0, y, 0.0, 1.0],
                    false => [y, 0.0, 0.0, 1.0],
                })
                .collect();
            assert_eq!(floats(&sampled), want, "{pf:?}'s view sampled");
        }
        // Over plane 1, not measured: refused.
        let pf = PixelFormat::YCBCR84442P;
        let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
        let surface = buffer_with(device, &family_bytes(390, codes)).unwrap();
        let mut d = TextureDescriptor::texture_2d(pf, W, H, false);
        d.usage = TextureUsage::SHADER_READ;
        d.resource_options = ResourceOptions::shared();
        let code = u32::from_be_bytes(*b"444f");
        let facts = IoSurfaceYCbCr::new(code, 1, [W, H], CHROMAQ_AT, ROWQ);
        let at = IoSurfacePlane::new(CHROMAQ_AT, ROWQ, true).with_ycbcr(facts);
        let plane_one = surface.new_texture_with_iosurface_plane(&d, at).unwrap();
        assert!(matches!(
            array(&plane_one, None),
            Err(Error::Unsupported(_))
        ));
    });
}

/// A packed texture's CPU calls move nothing, as Metal's (48 `getBytes` and 48
/// `replaceRegion` calls): `get_bytes` leaves the bytes it is given and `replace_region` the
/// planes, at a row's words, twice them, four bytes a texel and 256 bytes a row, the texture
/// reading the planes' conversion still. A blit with a buffer either way, whose bytes Metal moves
/// by no rule found, is refused, naming the format.
#[test]
fn packed_textures_copy_nothing_and_refuse_blits() {
    on_device(|device, _| {
        let kernels = Kernels::new(device);
        let queue = device.new_command_queue().unwrap();
        for (seed, (pf, ostype)) in (360..).zip([
            (PixelFormat::YCBCR104202PPACKED, "p420"),
            (PixelFormat::YCBCR104222PPACKEDSRGB, "p422"),
            (PixelFormat::YCBCR104442PPACKED, "p444"),
        ]) {
            let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            let bytes = family_bytes(seed, codes);
            let surface = buffer_with(device, &bytes).unwrap();
            let texture = family_texture(&surface, pf, ostype, None).unwrap();
            let words = row_bytes(codes, W);
            let region = Region::new_2d(0, 0, W, H);
            for row in [words, 2 * words, 4 * W, 256] {
                let mut got = vec![0x5a; (row * H) as usize];
                texture.get_bytes(&mut got, row, 0, region, 0, 0).unwrap();
                assert!(
                    got.iter().all(|&b| b == 0x5a),
                    "{pf:?}: get_bytes at {row} wrote"
                );
                let new: Vec<u8> = (0..row * H).map(|i| (i * 7) as u8).collect();
                texture.replace_region(region, 0, 0, &new, row, 0).unwrap();
                let length = bytes.len() as u64;
                let planes = read_buffer(&surface, 0, length).unwrap();
                assert_eq!(planes, bytes, "{pf:?}: replace_region at {row}");
            }
            let grid = match codes.output {
                YCbCrOutput::Srgb => 8192,
                _ => 2048,
            };
            let read = grid_of(&texels_of(device, &kernels.read, &texture, None), grid);
            let want = family_modelled(&bytes, codes, (YCbCrRange::Video, YCbCrMatrix::Bt601));
            assert_eq!(read, want, "{pf:?}: read() after the calls");
            let other = device
                .new_buffer(4 * W * H, ResourceOptions::shared())
                .unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            let size = Size::new(W, H, 1);
            let out = blit.copy_from_texture_to_buffer(
                &texture,
                0,
                0,
                Origin::new(0, 0, 0),
                size,
                &other,
                0,
                4 * W,
                0,
                BlitOption::None,
            );
            let into = blit.copy_from_buffer_to_texture(
                &other,
                0,
                4 * W,
                0,
                size,
                &texture,
                0,
                0,
                Origin::new(0, 0, 0),
                BlitOption::None,
            );
            for (way, refused) in [("out", out), ("in", into)] {
                let named = format!("{refused:?}");
                assert!(
                    matches!(refused, Err(Error::Unsupported(_))) && named.contains(pf.name()),
                    "{pf:?}: a blit {way}: {named}"
                );
            }
            blit.end_encoding().unwrap();
        }
    });
}
