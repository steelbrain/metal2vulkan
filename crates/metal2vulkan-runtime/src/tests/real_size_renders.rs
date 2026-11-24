//! Render targets at an icon's real size, every texel compared: textures over buffers in one
//! Shared block of rt's own that a pass clears and draws into part of; passes of two targets whose
//! draws sample each target while it is the pass's attachment; and the layers and screens of
//! rt's own an icon is drawn through, cleared or loaded and drawn into in part.
//!
//! Under a device cfg, on every device of the cfg's kind (`on_device`), with the readers and the
//! cells of `real_size.rs`:
//!
//! - `every_device_renders_icon_targets_over_one_shared_block_every_texel`: textures over buffers
//!   in one 16 MiB Shared block of rt's own, each made after the CPU wrote the complement of its
//!   pattern into its rows: `BGRA8Unorm` at 256x256 in rows of 1,024 bytes, read, written and a
//!   render target; `R8Unorm` at 256x256 in rows of 256 bytes, also a pixel format view, with an
//!   `A8Unorm` view; and `RGB10A2Unorm` at 258x258 in rows of 1,280 bytes, read, written and a
//!   render target. Four lie side by side at 4 KiB places in the block (the `RGB10A2Unorm` one
//!   twice); then, the `BGRA8Unorm` one released, two more over the bytes it rendered. Each is
//!   sampled (the complement), then drawn by a pass whose attachment 0 it is (`Clear`, `Store`)
//!   and whose attachment 1 is a Private `RG16Float` target of its size (`DontCare`, `DontCare`;
//!   for one `RGB10A2Unorm` placement `Clear`, `DontCare`), its one draw cut by a scissor
//!   rectangle that leaves 37, 41, 45 and 39 texels of the clear on the left, right, top and
//!   bottom, then by a blit encoder's `synchronizeResource` of it. Then three reads, each its own
//!   cell: the block's bytes as the CPU reads them, a blit of the texture into another buffer, and
//!   a sampled read (and the view's), every texel the pattern inside the rectangle and the clear
//!   outside it; and the block's bytes outside the texture's texels, which no pass may change. At
//!   the end the whole block, every byte. Each texture's line names its form (in place, with its
//!   linear image's row pitch, offset and size, or copied for its cause, with what the device's
//!   linear tiling lacks of its optimal tiling's features for its usage and the linear image's
//!   size and alignment), the layout the pass renders it in, its home between commands (where the
//!   CPU reads it), the pass's areas and the over-copies counted. A clear the bytes miss, a draw
//!   that lands where the clear did not, a copy back at the wrong pitch or place, or a stray write
//!   past the rows each shows as texels or bytes not exact.
//! - `every_device_samples_both_targets_of_a_two_target_pass_in_it_every_texel`: passes of two
//!   Private targets of rt's own, an `RGBA16Float` one at attachment 0 (`Store`) and an
//!   `RG16Float` one at attachment 1 (`Clear`, `DontCare`): at 256x256 with attachment 0 the
//!   `RGBA16Float` texture's own-format view (identity swizzle), cleared, and at 448x256 with
//!   attachment 0 the texture, loaded after a blit of the float pattern's complement. Each pass
//!   draws the float pattern (`real_size_overs.rs`) into both, cut to the same rectangle, then,
//!   after a memory barrier, a draw over the whole of both that samples attachment 0 (one pass)
//!   or attachment 1 (another) at each pixel's centre while it is the pass's attachment and
//!   writes the complement of what it read. Attachment 0's texture is then copied out by a blit
//!   and attachment 0 sampled, and the 256x256 texture is downloaded by a blit into a Private
//!   buffer in rows of 2,048 bytes and copied out of it, every texel compared: the read of
//!   attachment 1 shows in attachment 0. Each line says how the device reads its attachments,
//!   through the feedback loop or through copies into a shadow, with the pass's
//!   `Metrics::attachment_loop_reads` and `Metrics::attachment_read_copies`.
//! - `every_device_draws_icon_layers_in_part_after_a_clear_every_texel`: Private targets of rt's
//!   own that icons are drawn through, `BGRA8Unorm` at 256x128 and at 256x192 of three levels and
//!   `RGBA16Unorm` at 256x128 (the 8-bit pattern's channels times 257), each holding the
//!   pattern's complement by a blit, then cleared by a pass of one attachment that draws the
//!   pattern into the same rectangle: copied out and sampled, every texel compared.
//! - `every_device_samples_a_screen_through_its_view_in_its_own_pass_every_texel`: Private
//!   targets of rt's own at a screen's 2560x1600, `BGRA8Unorm` and `BGR10A2Unorm`, each holding
//!   the pattern's complement by a blit, then drawn through its own-format view (identity
//!   swizzle) by a pass that loads it, draws the pattern into the rectangle and, after a memory
//!   barrier, samples the view, its own attachment, at each pixel's centre and writes the
//!   complement of what it read: copied out and sampled through the view, every texel compared,
//!   with the device's form and the pass's attachment counters.
//!
//! The shaders write each unorm channel's `k` of `L` (its largest value) as a float the target
//! is to store as `k`. Vulkan's conversion of a float to a normalized fixed-point value pins only
//! an exactly integral `r = f L`: it "returns one of the two unsigned binary integer values with
//! exactly b bits which are closest to the floating-point value r", and "if r is equal to an
//! integer, then that integer value must be returned"; it should round to nearest but need not.
//! So the cells rely on each device's rounding of the forms they write, one form a width: every
//! 8- and 16-bit write, a clear's too, is a quarter step up, `(k + 0.25) / L`, and every 10-bit
//! one is `k / L` (`L` 1,023, or 3 for the 2-bit alpha), since a device may store
//! `(k + 0.25) / 1023` as `k + 1` (lavapipe's `A2R10G10B10` targets do from `k` near 767 up). A
//! device that stores one otherwise fails a cell by its own rounding, not by a wrong texel; a
//! 10-bit cell that is not exact says which, with the first wrong texel's word and channels read
//! and wanted, the difference in steps, and how many wrong texels are one step off at most in
//! every channel.
//!
//! `llvm-as` assembles the shaders; without it the tests skip visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::pass::Pass;
    use crate::tests::command::on_device::{each_device, finished};
    use crate::tests::real_size::on_device::{
        copy_in, copy_out, kernel, kernel_read, sampled, Cells, Pack, Readers,
    };
    use crate::tests::real_size::{complement, unorm_pattern};
    use crate::tests::real_size_overs::on_device::FLOAT;
    use crate::tests::real_size_overs::{float_pattern, half, single};
    use crate::texture::{kept_features, TextureObject};
    use crate::{
        Block, BufferLayout, ClearColor, ColorAttachmentDescriptor, CommandBuffer, CommandQueue,
        Device, Error, FunctionDescriptor, IndexClass, Library, LoadAction, MemoryClass,
        PixelFormat, PrimitiveType, RenderPassDescriptor, RenderPipelineDescriptor,
        RenderPipelineState, Resource, SamplerState, ScissorRect, StageSource, StorageMode,
        StoreAction, Texture, TextureDescriptor, TexturePlace, TextureSwizzleChannels, TextureType,
        TextureUsage,
    };
    use ash::vk;
    use metal2vulkan_devtools::{llvm_assemble_text, AssembleError};
    use std::ptr::NonNull;

    /// The owned shaders. `cover` draws the triangle that covers clip space. Each fragment writes
    /// render targets 0 and 1. `unorm8_pair` writes the 8-bit pattern (`real_size.rs`'s
    /// `unorm_pattern(8, 8, x, y)`) as `(k + 0.25) / 255`, `alpha8_pair` that pattern's alpha in
    /// every channel, and `unorm10_pair` `unorm_pattern(10, 2, x, y)` as `k / 1023`, its alpha
    /// `k / 3`, each with its pixel's position to target 1; `float_pair` writes the float
    /// pattern (`real_size_overs.rs`) to both. `loop_front` and `loop_back` sample texture 0
    /// through sampler 0 at the pixel's centre (its position scaled by buffer 0's second
    /// `float4`) and write to both buffer 0's first `float4` less what they read: `loop_front` the
    /// read's four channels, `loop_back` its red and green twice (red, green, red, green), as a
    /// two-channel target reads. `unorm8_one`, `unorm16_one` and `unorm10_one` write render target
    /// 0 alone: the 8-bit pattern, its channels times 257 as `(257 k + 0.25) / 65535`, and the
    /// 10-bit pattern. `loop_one` samples texture 0 as the loops do and writes render target 0
    /// alone: with each channel's largest value `L` in buffer 0's first `float4` and the part of
    /// a step its width's writes add in its third (0.25, or 0 for 10 bits), the read `v` rounded
    /// to its step, `k = uint(v L + 0.5)`, and its complement written as that width's writes are,
    /// `(L - k + up) / L`, so the read's precision does not move it.
    /// `UNORM8`, `UNORM16`, `UNORM10` and `FLOAT` stand for the lines that make `%v`, the pattern
    /// at `(%x, %y)`.
    const SHADERS: &str = r#"; Owned synthetic shaders.
source_filename = "real_size_renders.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define <{ <4 x float>, <4 x float> }> @cover(i32 noundef %id) {
entry:
  %one = icmp eq i32 %id, 1
  %two = icmp eq i32 %id, 2
  %px = select i1 %one, float 3.000000e+00, float -1.000000e+00
  %py = select i1 %two, float 3.000000e+00, float -1.000000e+00
  %p0 = insertelement <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 1.000000e+00>, float %px, i64 0
  %p = insertelement <4 x float> %p0, float %py, i64 1
  %o0 = insertvalue <{ <4 x float>, <4 x float> }> undef, <4 x float> %p, 0
  %o = insertvalue <{ <4 x float>, <4 x float> }> %o0, <4 x float> zeroinitializer, 1
  ret <{ <4 x float>, <4 x float> }> %o
}

define <{ <4 x float>, <4 x float> }> @unorm8_pair(<4 x float> %pos) {
entry:
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %pi = tail call <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float> %xy)
  %x = extractelement <2 x i32> %pi, i64 0
  %y = extractelement <2 x i32> %pi, i64 1
UNORM8
  %o0 = insertvalue <{ <4 x float>, <4 x float> }> undef, <4 x float> %v, 0
  %o = insertvalue <{ <4 x float>, <4 x float> }> %o0, <4 x float> %pos, 1
  ret <{ <4 x float>, <4 x float> }> %o
}

define <{ <4 x float>, <4 x float> }> @alpha8_pair(<4 x float> %pos) {
entry:
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %pi = tail call <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float> %xy)
  %x = extractelement <2 x i32> %pi, i64 0
  %y = extractelement <2 x i32> %pi, i64 1
UNORM8
  %a = shufflevector <4 x float> %v, <4 x float> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %o0 = insertvalue <{ <4 x float>, <4 x float> }> undef, <4 x float> %a, 0
  %o = insertvalue <{ <4 x float>, <4 x float> }> %o0, <4 x float> %pos, 1
  ret <{ <4 x float>, <4 x float> }> %o
}

define <{ <4 x float>, <4 x float> }> @unorm10_pair(<4 x float> %pos) {
entry:
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %pi = tail call <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float> %xy)
  %x = extractelement <2 x i32> %pi, i64 0
  %y = extractelement <2 x i32> %pi, i64 1
UNORM10
  %o0 = insertvalue <{ <4 x float>, <4 x float> }> undef, <4 x float> %v, 0
  %o = insertvalue <{ <4 x float>, <4 x float> }> %o0, <4 x float> %pos, 1
  ret <{ <4 x float>, <4 x float> }> %o
}

define <{ <4 x float>, <4 x float> }> @float_pair(<4 x float> %pos) {
entry:
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %pi = tail call <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float> %xy)
  %x = extractelement <2 x i32> %pi, i64 0
  %y = extractelement <2 x i32> %pi, i64 1
FLOAT
  %o0 = insertvalue <{ <4 x float>, <4 x float> }> undef, <4 x float> %v, 0
  %o = insertvalue <{ <4 x float>, <4 x float> }> %o0, <4 x float> %v, 1
  ret <{ <4 x float>, <4 x float> }> %o
}

define <{ <4 x float>, <4 x float> }> @loop_front(<4 x float> %pos, ptr addrspace(1) %t, ptr addrspace(2) %smp, ptr addrspace(2) noundef "air-buffer-no-alias" %p) {
entry:
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %pinv = getelementptr inbounds <4 x float>, ptr addrspace(2) %p, i64 1
  %inv4 = load <4 x float>, ptr addrspace(2) %pinv, align 16
  %inv = shufflevector <4 x float> %inv4, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %uv = fmul <2 x float> %xy, %inv
  %r = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %smp, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  %s = load <4 x float>, ptr addrspace(2) %p, align 16
  %c = fsub <4 x float> %s, %v
  %o0 = insertvalue <{ <4 x float>, <4 x float> }> undef, <4 x float> %c, 0
  %o = insertvalue <{ <4 x float>, <4 x float> }> %o0, <4 x float> %c, 1
  ret <{ <4 x float>, <4 x float> }> %o
}

define <{ <4 x float>, <4 x float> }> @loop_back(<4 x float> %pos, ptr addrspace(1) %t, ptr addrspace(2) %smp, ptr addrspace(2) noundef "air-buffer-no-alias" %p) {
entry:
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %pinv = getelementptr inbounds <4 x float>, ptr addrspace(2) %p, i64 1
  %inv4 = load <4 x float>, ptr addrspace(2) %pinv, align 16
  %inv = shufflevector <4 x float> %inv4, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %uv = fmul <2 x float> %xy, %inv
  %r = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %smp, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  %w = shufflevector <4 x float> %v, <4 x float> poison, <4 x i32> <i32 0, i32 1, i32 0, i32 1>
  %s = load <4 x float>, ptr addrspace(2) %p, align 16
  %c = fsub <4 x float> %s, %w
  %o0 = insertvalue <{ <4 x float>, <4 x float> }> undef, <4 x float> %c, 0
  %o = insertvalue <{ <4 x float>, <4 x float> }> %o0, <4 x float> %c, 1
  ret <{ <4 x float>, <4 x float> }> %o
}

define <4 x float> @unorm8_one(<4 x float> %pos) {
entry:
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %pi = tail call <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float> %xy)
  %x = extractelement <2 x i32> %pi, i64 0
  %y = extractelement <2 x i32> %pi, i64 1
UNORM8
  ret <4 x float> %v
}

define <4 x float> @unorm16_one(<4 x float> %pos) {
entry:
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %pi = tail call <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float> %xy)
  %x = extractelement <2 x i32> %pi, i64 0
  %y = extractelement <2 x i32> %pi, i64 1
UNORM16
  ret <4 x float> %v
}

define <4 x float> @unorm10_one(<4 x float> %pos) {
entry:
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %pi = tail call <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float> %xy)
  %x = extractelement <2 x i32> %pi, i64 0
  %y = extractelement <2 x i32> %pi, i64 1
UNORM10
  ret <4 x float> %v
}

define <4 x float> @loop_one(<4 x float> %pos, ptr addrspace(1) %t, ptr addrspace(2) %smp, ptr addrspace(2) noundef "air-buffer-no-alias" %p) {
entry:
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %pinv = getelementptr inbounds <4 x float>, ptr addrspace(2) %p, i64 1
  %inv4 = load <4 x float>, ptr addrspace(2) %pinv, align 16
  %inv = shufflevector <4 x float> %inv4, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %uv = fmul <2 x float> %xy, %inv
  %r = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %smp, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  %m = load <4 x float>, ptr addrspace(2) %p, align 16
  %vm = fmul <4 x float> %v, %m
  %vh = fadd <4 x float> %vm, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %k = fptoui <4 x float> %vh to <4 x i32>
  %kf = tail call <4 x float> @air.convert.f.v4f32.u.v4i32(<4 x i32> %k)
  %c = fsub <4 x float> %m, %kf
  %pup = getelementptr inbounds <4 x float>, ptr addrspace(2) %p, i64 2
  %up = load <4 x float>, ptr addrspace(2) %pup, align 16
  %cq = fadd <4 x float> %c, %up
  %o = fdiv <4 x float> %cq, %m
  ret <4 x float> %o
}

declare <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float>)
declare <4 x float> @air.convert.f.v4f32.u.v4i32(<4 x i32>)
declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

!air.vertex = !{!0}
!air.fragment = !{!6, !12, !13, !16, !17, !22, !24, !25, !26, !27}
!air.version = !{!14}
!air.language_version = !{!15}
!0 = !{ptr @cover, !1, !4}
!1 = !{!2, !3}
!2 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!3 = !{!"air.vertex_output", !"generated(5colorDv4_f)", !"air.arg_type_name", !"float4", !"air.arg_name", !"color"}
!4 = !{!5}
!5 = !{i32 0, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"id"}
!6 = !{ptr @unorm8_pair, !7, !10}
!7 = !{!8, !9}
!8 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4", !"air.arg_name", !"first"}
!9 = !{!"air.render_target", i32 1, i32 0, !"air.arg_type_name", !"float4", !"air.arg_name", !"second"}
!10 = !{!11}
!11 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!12 = !{ptr @alpha8_pair, !7, !10}
!13 = !{ptr @unorm10_pair, !7, !10}
!14 = !{i32 2, i32 8, i32 0}
!15 = !{!"Metal", i32 4, i32 0, i32 0}
!16 = !{ptr @float_pair, !7, !10}
!17 = !{ptr @loop_front, !7, !18}
!18 = !{!11, !19, !20, !21}
!19 = !{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}
!20 = !{i32 2, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"smp"}
!21 = !{i32 3, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"p"}
!22 = !{ptr @loop_back, !7, !18}
!23 = !{!8}
!24 = !{ptr @unorm8_one, !23, !10}
!25 = !{ptr @unorm16_one, !23, !10}
!26 = !{ptr @unorm10_one, !23, !10}
!27 = !{ptr @loop_one, !23, !18}
"#;

    /// The lines that make `%v`, the 8-bit pattern at `(%x, %y)`, each channel's `k` as
    /// `(k + 0.25) / 255`.
    const UNORM8: &str = r#"  %k0 = and i32 %x, 255
  %k1 = and i32 %y, 255
  %xh = lshr i32 %x, 8
  %yh = lshr i32 %y, 8
  %yh4 = shl i32 %yh, 4
  %k2 = or i32 %xh, %yh4
  %xy.x = xor i32 %x, %y
  %k3 = and i32 %xy.x, 255
  %f0 = uitofp i32 %k0 to float
  %f1 = uitofp i32 %k1 to float
  %f2 = uitofp i32 %k2 to float
  %f3 = uitofp i32 %k3 to float
  %kv0 = insertelement <4 x float> poison, float %f0, i64 0
  %kv1 = insertelement <4 x float> %kv0, float %f1, i64 1
  %kv2 = insertelement <4 x float> %kv1, float %f2, i64 2
  %kv = insertelement <4 x float> %kv2, float %f3, i64 3
  %kq = fadd <4 x float> %kv, <float 2.500000e-01, float 2.500000e-01, float 2.500000e-01, float 2.500000e-01>
  %v = fdiv <4 x float> %kq, <float 2.550000e+02, float 2.550000e+02, float 2.550000e+02, float 2.550000e+02>
"#;

    /// The lines that make `%v`, the 8-bit pattern at `(%x, %y)` in 16 bits, each channel's `k`
    /// times 257 as `(257 k + 0.25) / 65535`.
    const UNORM16: &str = r#"  %k0 = and i32 %x, 255
  %k1 = and i32 %y, 255
  %xh = lshr i32 %x, 8
  %yh = lshr i32 %y, 8
  %yh4 = shl i32 %yh, 4
  %k2 = or i32 %xh, %yh4
  %xy.x = xor i32 %x, %y
  %k3 = and i32 %xy.x, 255
  %f0 = uitofp i32 %k0 to float
  %f1 = uitofp i32 %k1 to float
  %f2 = uitofp i32 %k2 to float
  %f3 = uitofp i32 %k3 to float
  %kv0 = insertelement <4 x float> poison, float %f0, i64 0
  %kv1 = insertelement <4 x float> %kv0, float %f1, i64 1
  %kv2 = insertelement <4 x float> %kv1, float %f2, i64 2
  %kv = insertelement <4 x float> %kv2, float %f3, i64 3
  %k16 = fmul <4 x float> %kv, <float 2.570000e+02, float 2.570000e+02, float 2.570000e+02, float 2.570000e+02>
  %kq = fadd <4 x float> %k16, <float 2.500000e-01, float 2.500000e-01, float 2.500000e-01, float 2.500000e-01>
  %v = fdiv <4 x float> %kq, <float 6.553500e+04, float 6.553500e+04, float 6.553500e+04, float 6.553500e+04>
"#;

    /// The lines that make `%v`, the 10-bit pattern at `(%x, %y)`, each colour channel's `k` as
    /// `k / 1023` and its alpha's as `k / 3`.
    const UNORM10: &str = r#"  %k0 = and i32 %x, 1023
  %k1 = and i32 %y, 1023
  %xh = lshr i32 %x, 10
  %yh = lshr i32 %y, 10
  %yh2 = shl i32 %yh, 2
  %k2 = or i32 %xh, %yh2
  %xy.x = xor i32 %x, %y
  %k3 = and i32 %xy.x, 3
  %f0 = uitofp i32 %k0 to float
  %f1 = uitofp i32 %k1 to float
  %f2 = uitofp i32 %k2 to float
  %f3 = uitofp i32 %k3 to float
  %kv0 = insertelement <4 x float> poison, float %f0, i64 0
  %kv1 = insertelement <4 x float> %kv0, float %f1, i64 1
  %kv2 = insertelement <4 x float> %kv1, float %f2, i64 2
  %kv = insertelement <4 x float> %kv2, float %f3, i64 3
  %v = fdiv <4 x float> %kv, <float 1.023000e+03, float 1.023000e+03, float 1.023000e+03, float 3.000000e+00>
"#;

    /// The libraries the device tests take: `real_size.rs`'s readers, `real_size_overs.rs`'s
    /// shaders (its float readers) and [`SHADERS`]; None where `llvm-as` is missing.
    fn libraries() -> Option<(Library, Library, Library)> {
        let readers = crate::tests::real_size::on_device::library()?;
        let floats = crate::tests::real_size_overs::on_device::library()?;
        let text = (SHADERS.replace("UNORM8\n", UNORM8))
            .replace("UNORM10\n", UNORM10)
            .replace("UNORM16\n", UNORM16)
            .replace("FLOAT\n", FLOAT);
        match llvm_assemble_text(&text) {
            Ok(bitcode) => {
                let own = Library::from_air(&bitcode).expect("the library");
                Some((readers, floats, own))
            }
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                None
            }
            Err(error) => panic!("the real-size render shaders: {error:?}"),
        }
    }

    /// How a render target's texels are held, drawn and read.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Kind {
        /// `BGRA8Unorm`, of the 8-bit pattern.
        Bgra8,
        /// `R8Unorm`, of the 8-bit pattern's alpha, `(x ^ y) & 255`.
        R8,
        /// `RGB10A2Unorm`, of `unorm_pattern(10, 2, x, y)`.
        Rgb10a2,
    }

    impl Kind {
        fn format(self) -> PixelFormat {
            match self {
                Kind::Bgra8 => PixelFormat::BGRA8Unorm,
                Kind::R8 => PixelFormat::R8Unorm,
                Kind::Rgb10a2 => PixelFormat::RGB10A2Unorm,
            }
        }

        /// A texel's bytes.
        fn texel(self) -> u32 {
            match self {
                Kind::R8 => 1,
                Kind::Bgra8 | Kind::Rgb10a2 => 4,
            }
        }

        /// The fragment that draws the pattern.
        fn fragment(self) -> &'static str {
            match self {
                Kind::Bgra8 => "unorm8_pair",
                Kind::R8 => "alpha8_pair",
                Kind::Rgb10a2 => "unorm10_pair",
            }
        }

        /// The pattern's channels at `(x, y)`, or their complement, in shader order; an `R8Unorm`
        /// texel's red alone.
        fn channels(self, x: u32, y: u32, complemented: bool) -> [u32; 4] {
            let flip = |bits: u32, k: u32| if complemented { complement(bits, k) } else { k };
            match self {
                Kind::Bgra8 => unorm_pattern(8, 8, x, y).map(|k| flip(8, k)),
                Kind::R8 => [flip(8, unorm_pattern(8, 8, x, y)[3]), 0, 0, 0],
                Kind::Rgb10a2 => {
                    let [r, g, b, a] = unorm_pattern(10, 2, x, y);
                    [flip(10, r), flip(10, g), flip(10, b), flip(2, a)]
                }
            }
        }

        /// The channels the pass clears attachment 0 to: a `BGRA8Unorm` and an `RGB10A2Unorm`
        /// texel that neither the pattern nor its complement holds at these sizes; an `R8Unorm`
        /// texel's one channel takes every value, so its clear meets both at some texels.
        fn clear(self) -> [u32; 4] {
            match self {
                Kind::Bgra8 => [32, 96, 160, 224],
                Kind::R8 => [32, 0, 0, 0],
                Kind::Rgb10a2 => [32, 96, 160, 2],
            }
        }

        /// The part of a step each write of this kind adds: a quarter for 8 bits, none for 10.
        fn up(self) -> f64 {
            match self {
                Kind::Bgra8 | Kind::R8 => 0.25,
                Kind::Rgb10a2 => 0.0,
            }
        }

        /// The clear colour of [`clear`](Self::clear): `k` and [`up`](Self::up) over each
        /// channel's largest.
        fn clear_color(self) -> ClearColor {
            let [r, g, b, a] = self.clear().map(|k| f64::from(k) + self.up());
            match self {
                Kind::Rgb10a2 => ClearColor::new(r / 1023.0, g / 1023.0, b / 1023.0, a / 3.0),
                Kind::Bgra8 | Kind::R8 => {
                    ClearColor::new(r / 255.0, g / 255.0, b / 255.0, a / 255.0)
                }
            }
        }

        /// The bytes in memory of a texel of channels `k`: `BGRA8Unorm` blue, green, red, alpha;
        /// `R8Unorm` red; `RGB10A2Unorm` one little-endian word, red lowest.
        fn bytes(self, [r, g, b, a]: [u32; 4]) -> Vec<u8> {
            match self {
                Kind::Bgra8 => [b, g, r, a].map(|c| c as u8).to_vec(),
                Kind::R8 => vec![r as u8],
                Kind::Rgb10a2 => (r | (g << 10) | (b << 20) | (a << 30))
                    .to_le_bytes()
                    .to_vec(),
            }
        }

        /// A reader's word of a texel of channels `k`, as [`pack`](Self::pack) packs it.
        fn word(self, [r, g, b, a]: [u32; 4]) -> Vec<u8> {
            let word = match self {
                Kind::Bgra8 => r | (g << 8) | (b << 16) | (a << 24),
                Kind::R8 => r,
                Kind::Rgb10a2 => r | (g << 10) | (b << 20) | (a << 30),
            };
            word.to_le_bytes().to_vec()
        }

        /// How a reader packs a texel: each channel's bits from bit 0 up.
        fn pack(self) -> Pack {
            match self {
                Kind::Bgra8 => Pack::unorm(8, 4),
                Kind::R8 => Pack::unorm(8, 1),
                Kind::Rgb10a2 => Pack {
                    scale: [1023.0, 1023.0, 1023.0, 3.0],
                    shift: [0, 10, 20, 30],
                    mask: [0x3ff, 0x3ff << 10, 0x3ff << 20, 3 << 30],
                },
            }
        }
    }

    /// Where an `RGB10A2Unorm` texel's word, and a reader's word of any 10-bit texel, keep red,
    /// green, blue and alpha.
    const RGBA10: [u32; 4] = [0, 10, 20, 30];

    /// Where a `BGR10A2Unorm` texel's word keeps red, green, blue and alpha.
    const BGRA10: [u32; 4] = [20, 10, 0, 30];

    /// How a reader packs an `A8Unorm` view's alpha: bits 0 to 7.
    const ALPHA: Pack = Pack {
        scale: [0.0, 0.0, 0.0, 255.0],
        shift: [0; 4],
        mask: [0, 0, 0, 255],
    };

    /// A render target over a buffer: its texels, size, rows' bytes and usage.
    struct Shape {
        what: &'static str,
        kind: Kind,
        size: [u32; 2],
        row: u64,
        usage: TextureUsage,
    }

    /// Read, written and rendered into.
    const DRAWN: TextureUsage = TextureUsage::SHADER_READ
        .union(TextureUsage::SHADER_WRITE)
        .union(TextureUsage::RENDER_TARGET);

    const BGRA8: Shape = Shape {
        what: "BGRA8Unorm 256x256 rows 1024",
        kind: Kind::Bgra8,
        size: [256, 256],
        row: 1024,
        usage: DRAWN,
    };

    const R8: Shape = Shape {
        what: "R8Unorm 256x256 rows 256 (with an A8Unorm view)",
        kind: Kind::R8,
        size: [256, 256],
        row: 256,
        usage: DRAWN.union(TextureUsage::PIXEL_FORMAT_VIEW),
    };

    const RGB10A2: Shape = Shape {
        what: "RGB10A2Unorm 258x258 rows 1280",
        kind: Kind::Rgb10a2,
        size: [258, 258],
        row: 1280,
        usage: DRAWN,
    };

    /// The block the textures lie in: 16 MiB.
    const BLOCK: u64 = 16 << 20;

    /// The bytes of the block no texel holds.
    const PAD: u8 = 0x5a;

    /// A texture's place: its shape, its buffer's first byte in the block, and the load action
    /// of the pass's attachment 1.
    type Place = (&'static Shape, u64, LoadAction);

    /// The textures side by side in the block, at 4 KiB places, each made before any is drawn.
    const SIDE_BY_SIDE: [Place; 4] = [
        (&R8, 163_840, LoadAction::DontCare),
        (&BGRA8, 905_216, LoadAction::DontCare),
        (&RGB10A2, 2_744_320, LoadAction::DontCare),
        (&RGB10A2, 4_665_344, LoadAction::Clear),
    ];

    /// Which of [`SIDE_BY_SIDE`] is released before [`OVER_IT`] are made over its bytes.
    const RELEASED: usize = 1;

    /// The textures made over the bytes of [`SIDE_BY_SIDE`]'s `BGRA8Unorm` one once it is
    /// released, which they share between them.
    const OVER_IT: [Place; 2] = [
        (&R8, 892_928, LoadAction::DontCare),
        (&RGB10A2, 958_464, LoadAction::DontCare),
    ];

    /// The texels of the clear the draw leaves on the left, right, top and bottom: each past 32,
    /// none a multiple of it.
    const BORDER: [u32; 4] = [37, 41, 45, 39];

    /// The rectangle a pass's draw is cut to: `x`, `y`, width and height.
    fn drawn(size: [u32; 2]) -> [u32; 4] {
        let [left, right, top, bottom] = BORDER;
        [left, top, size[0] - left - right, size[1] - top - bottom]
    }

    /// What `(x, y)` holds after the pass: the pattern inside [`drawn`], the clear outside it.
    fn after(kind: Kind, size: [u32; 2], x: u32, y: u32) -> [u32; 4] {
        let [left, top, width, height] = drawn(size);
        let inside = (left..left + width).contains(&x) && (top..top + height).contains(&y);
        match inside {
            true => kind.channels(x, y, false),
            false => kind.clear(),
        }
    }

    /// Every texel's `of(x, y)`, rows from the top.
    fn each(size: [u32; 2], of: impl Fn(u32, u32) -> Vec<u8>) -> Vec<u8> {
        (0..size[1])
            .flat_map(|y| (0..size[0]).map(move |x| (x, y)))
            .flat_map(|(x, y)| of(x, y))
            .collect()
    }

    /// The block's bytes, written by the CPU through its mapping, and the bytes it should hold.
    struct Held {
        pointer: NonNull<u8>,
        expected: Vec<u8>,
    }

    impl Held {
        /// The block's bytes, every one [`PAD`].
        fn new(block: &Block) -> Held {
            let pointer = block.contents.expect("a Shared block");
            let held = Held {
                pointer,
                expected: vec![PAD; BLOCK as usize],
            };
            // SAFETY: the block's `BLOCK` bytes, which no command reaches yet.
            unsafe { std::ptr::write_bytes(pointer.as_ptr(), PAD, BLOCK as usize) };
            held
        }

        /// Writes `bytes` from `at`, through the mapping, while no command reaches them.
        fn write(&mut self, at: usize, bytes: &[u8]) {
            // SAFETY: `at` and the bytes lie in the block, and no command reaches them now.
            unsafe {
                let to = self.pointer.as_ptr().add(at);
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), to, bytes.len());
            }
            self.expected[at..at + bytes.len()].copy_from_slice(bytes);
        }

        /// The block's bytes, once no command reaches them.
        fn read(&self) -> Vec<u8> {
            // SAFETY: the block's `BLOCK` bytes; every command that reached them completed.
            unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), BLOCK as usize) }.to_vec()
        }

        /// Sets the texels of `shape` at `at` to `of(x, y)`'s bytes: through the mapping where
        /// `written`, else only in what the block should hold, as a pass writes them.
        fn texels(
            &mut self,
            (shape, at): (&Shape, u64),
            of: impl Fn(u32, u32) -> [u32; 4],
            written: bool,
        ) {
            let kind = shape.kind;
            for y in 0..shape.size[1] {
                let row: Vec<u8> = (0..shape.size[0])
                    .flat_map(|x| kind.bytes(of(x, y)))
                    .collect();
                let start = (at + u64::from(y) * shape.row) as usize;
                match written {
                    true => self.write(start, &row),
                    false => self.expected[start..start + row.len()].copy_from_slice(&row),
                }
            }
        }
    }

    /// The texels of `shape` at `at` in `bytes`, a block's, in packed rows.
    fn texels_in(bytes: &[u8], (shape, at): (&Shape, u64)) -> Vec<u8> {
        let row = (shape.size[0] * shape.kind.texel()) as usize;
        (0..shape.size[1])
            .flat_map(|y| {
                let start = (at + u64::from(y) * shape.row) as usize;
                bytes[start..start + row].iter().copied()
            })
            .collect()
    }

    /// `bytes`, a block's, with the texels of `shape` at `at` zeroed: the bytes outside them.
    fn outside(mut bytes: Vec<u8>, (shape, at): (&Shape, u64)) -> Vec<u8> {
        let row = (shape.size[0] * shape.kind.texel()) as usize;
        for y in 0..shape.size[1] {
            let start = (at + u64::from(y) * shape.row) as usize;
            bytes[start..start + row].fill(0);
        }
        bytes
    }

    /// A 10-bit texel's channels, red, green, blue and alpha, from bits `at` of its
    /// little-endian word: ten bits each, two for alpha.
    fn channels10(texel: &[u8], at: [u32; 4]) -> [i64; 4] {
        let word = u32::from_le_bytes(texel.try_into().expect("a word"));
        let mask = [1023, 1023, 1023, 3];
        std::array::from_fn(|c| i64::from((word >> at[c]) & mask[c]))
    }

    /// Where a 10-bit cell's read `got` of words in rows of `width` is not `want`: the first
    /// wrong texel's word and channels (at bits `at`, as [`channels10`]) read and wanted, the
    /// read less the wanted in steps, and how many wrong texels are one step off at most in
    /// every channel, so a device's rounding of a write shows apart from a stale texel (the
    /// complement) or a misplaced one. Empty where `got` is `want` or no read.
    fn steps_off(got: &Result<Vec<u8>, String>, want: &[u8], at: [u32; 4], width: u32) -> String {
        let Ok(got) = got else {
            return String::new();
        };
        if got.len() != want.len() {
            return String::new();
        }
        let (mut wrong, mut near, mut first) = (0u64, 0u64, None);
        for (i, (g, w)) in (got.chunks_exact(4).zip(want.chunks_exact(4))).enumerate() {
            if g == w {
                continue;
            }
            let (read, wanted) = (channels10(g, at), channels10(w, at));
            let off: [i64; 4] = std::array::from_fn(|c| read[c] - wanted[c]);
            wrong += 1;
            if off.iter().all(|d| d.abs() <= 1) {
                near += 1;
            }
            if first.is_none() {
                first = Some((i as u32, (g, read), (w, wanted), off));
            }
        }
        let Some((i, (g, read), (w, wanted), off)) = first else {
            return String::new();
        };
        let word = |texel: &[u8]| u32::from_le_bytes(texel.try_into().expect("a word"));
        format!(
            " (the first wrong texel, ({}, {}), reads word {:#010x}, channels {read:?}, not \
             {:#010x}, {wanted:?}: {off:?} steps; {near} of {wrong} wrong texels are one step off \
             at most in every channel)",
            i % width,
            i / width,
            word(g),
            word(w)
        )
    }

    /// An error's text, with its variant.
    fn said(error: Error) -> String {
        format!("{error} ({error:?})")
    }

    /// Commits `command_buffer` and waits for it: what it failed with, if it did.
    fn completed(name: &str, command_buffer: &CommandBuffer) -> Result<(), String> {
        command_buffer.commit().map_err(said)?;
        finished(name, command_buffer);
        match command_buffer.error() {
            None => Ok(()),
            Some(error) => Err(format!("the command buffer: {error:?}")),
        }
    }

    /// The render state of `cover` and `kind`'s fragment into attachment 0 of `kind`'s format
    /// and attachment 1 of `RG16Float`.
    fn drawing(name: &str, device: &Device, library: &Library, kind: Kind) -> RenderPipelineState {
        let function = |f: &str| {
            StageSource::Function(FunctionDescriptor::new(
                library.function(f).expect("the function"),
            ))
        };
        let fragment = kind.fragment();
        let mut d = RenderPipelineDescriptor::new(function("cover"));
        d.fragment = Some(function(fragment));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(kind.format()));
        d.color_attachments[1] = Some(ColorAttachmentDescriptor::new(PixelFormat::RG16Float));
        let state = (device.new_render_pipeline_state(&d, crate::tests::no_constexpr_samplers))
            .unwrap_or_else(|error| panic!("{name}: {fragment}: {error}"));
        (state.finish_pipeline()).unwrap_or_else(|error| panic!("{name}: {fragment}: {error}"));
        state
    }

    /// How `texture`, over a buffer, holds its texels: in place, with its linear image's row
    /// pitch, offset and size; or copied for its cause, with the format features the device's
    /// linear tiling lacks of its optimal tiling's for the image's usage, and the size and
    /// alignment of the linear image the device would make.
    fn held_as(device: &Device, texture: &Texture) -> String {
        let core = device.core();
        if let Some((_, copy)) = texture.copied_over() {
            let (linear, optimal) = core.format_features(texture.image_format());
            let kept = kept_features(texture.object().image_usage());
            let d = texture.descriptor();
            return format!(
                "copied for {:?} (the linear tiling lacks features {:#x}; a linear image's subresource size \
                 {:?}, memory alignment {:?})",
                copy.kind.cause,
                (optimal & kept & !linear).as_raw(),
                device.linear_subresource_size(d),
                device.linear_memory_alignment(d)
            );
        }
        let color = vk::ImageSubresource {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            array_layer: 0,
        };
        // SAFETY: the texture's image, linear, of one level and layer of a colour format.
        let layout = unsafe { (core.device).get_image_subresource_layout(texture.image(), color) };
        format!(
            "in place, row pitch {} offset {} size {}",
            layout.row_pitch, layout.offset, layout.size
        )
    }

    /// The pass over `texture`: attachment 0 it, cleared to `kind`'s clear and stored;
    /// attachment 1 `second`, loaded by its action and not stored.
    fn pass_of(
        texture: &Texture,
        kind: Kind,
        (second, load): (&Texture, LoadAction),
    ) -> RenderPassDescriptor {
        let mut pass = RenderPassDescriptor::default();
        let first = &mut pass.color_attachments[0];
        first.texture = Some(texture.clone());
        first.load_action = LoadAction::Clear;
        first.store_action = StoreAction::Store;
        first.clear_color = kind.clear_color();
        let other = &mut pass.color_attachments[1];
        other.texture = Some(second.clone());
        other.load_action = load;
        other.store_action = StoreAction::DontCare;
        other.clear_color = ClearColor::new(0.5, 0.25, 0.0, 0.0);
        pass
    }

    /// The layout `pass` renders its attachment 0 in, that image's home between commands, and
    /// the pass's areas, as the runtime checks the pass.
    fn laid(device: &Device, pass: &RenderPassDescriptor) -> String {
        match Pass::check(device.core(), pass) {
            Ok(checked) => {
                let target = checked.colors[0].as_ref().expect("attachment 0");
                format!(
                    "rendered in {}, home {}, Metal's area {:?}, the load's and store's reach \
                     {:?}",
                    named(target.layout),
                    named(target.attachment.home),
                    checked.area,
                    checked.reach
                )
            }
            Err(error) => format!("the pass refused: {}", said(error)),
        }
    }

    /// The name of an image layout a pass or a home takes, else its number.
    fn named(layout: vk::ImageLayout) -> String {
        type L = vk::ImageLayout;
        let names = [
            (L::UNDEFINED, "UNDEFINED"),
            (L::GENERAL, "GENERAL"),
            (L::COLOR_ATTACHMENT_OPTIMAL, "COLOR_ATTACHMENT_OPTIMAL"),
            (L::SHADER_READ_ONLY_OPTIMAL, "SHADER_READ_ONLY_OPTIMAL"),
            (L::TRANSFER_SRC_OPTIMAL, "TRANSFER_SRC_OPTIMAL"),
            (L::TRANSFER_DST_OPTIMAL, "TRANSFER_DST_OPTIMAL"),
            (L::PREINITIALIZED, "PREINITIALIZED"),
            (L::READ_ONLY_OPTIMAL, "READ_ONLY_OPTIMAL"),
            (L::ATTACHMENT_OPTIMAL, "ATTACHMENT_OPTIMAL"),
            (
                L::ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT,
                "ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT",
            ),
        ];
        match names.iter().find(|(l, _)| *l == layout) {
            Some((_, name)) => (*name).to_owned(),
            None => format!("layout {}", layout.as_raw()),
        }
    }

    /// `pass`, its one draw by `state` cut to [`drawn`], then a blit encoder's
    /// `synchronizeResource` of `texture`, in one command buffer, committed and waited for.
    fn render(
        name: &str,
        queue: &CommandQueue,
        state: &RenderPipelineState,
        (pass, texture, size): (&RenderPassDescriptor, &Texture, [u32; 2]),
    ) -> Result<(), String> {
        let [x, y, width, height] = drawn(size).map(u64::from);
        let command_buffer = queue.command_buffer().map_err(said)?;
        let mut encoder = (command_buffer.render_command_encoder(pass)).map_err(said)?;
        encoder.set_render_pipeline_state(state).map_err(said)?;
        encoder.set_scissor_rect(ScissorRect {
            x,
            y,
            width,
            height,
        });
        (encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)).map_err(said)?;
        encoder.end_encoding();
        let mut blit = command_buffer.blit_command_encoder().map_err(said)?;
        (blit.synchronize_resource(Resource::Texture(texture))).map_err(said)?;
        blit.end_encoding();
        completed(name, &command_buffer)
    }

    /// A texture over a buffer at its place in the block, which holds the buffer, and its
    /// `A8Unorm` view where it is `R8Unorm`, each at the next of `indices`.
    struct Made {
        place: Place,
        texture: Texture,
        view: Option<Texture>,
    }

    impl Made {
        fn new(
            name: &str,
            device: &Device,
            block: &Block,
            place: Place,
            indices: &mut impl Iterator<Item = u32>,
        ) -> Made {
            let (shape, at, _) = place;
            let what = format!("{} at byte {at}", shape.what);
            let len = shape.row * u64::from(shape.size[1]);
            let buffer = (device.new_buffer(block.id, at, len, None))
                .unwrap_or_else(|error| panic!("{name}: {what}: its buffer: {error}"));
            let [width, height] = shape.size;
            let format = shape.kind.format();
            let mut d = TextureDescriptor::new(TextureType::Type2D, format, width, height);
            d.usage = shape.usage;
            let texture = (buffer.new_texture(&d, indices.next(), 0, shape.row))
                .unwrap_or_else(|error| panic!("{name}: {what}: {error}"));
            let view = (shape.kind == Kind::R8).then(|| {
                let views = PixelFormat::A8Unorm;
                (texture.new_texture_view(views, TextureType::Type2D, 0..1, 0..1, indices.next()))
                    .unwrap_or_else(|error| panic!("{name}: {what}: its A8Unorm view: {error}"))
            });
            Made {
                place,
                texture,
                view,
            }
        }
    }

    /// What one texture's run takes: the readers, the queue, the render states by kind and the
    /// Private `RG16Float` attachments 1 by side.
    struct Rig<'a> {
        name: &'a str,
        device: &'a Device,
        queue: &'a CommandQueue,
        readers: Readers,
        states: [(Kind, RenderPipelineState); 3],
        seconds: [(u32, Texture); 2],
    }

    impl Rig<'_> {
        /// `made`'s texture sampled (the complement), drawn by its pass and synchronized, then
        /// read by the CPU from the block, by a blit out and by sampling, each a cell, the
        /// block's bytes outside its texels a cell too; what the block should hold updated.
        fn run(&self, made: &Made, held: &mut Held, cells: &mut Cells) {
            let (name, device) = (self.name, self.device);
            let on = (device, self.queue);
            let (shape, at, load) = made.place;
            let (kind, size) = (shape.kind, shape.size);
            let side = size[0];
            let second = &self
                .seconds
                .iter()
                .find(|(s, _)| *s == side)
                .expect("its side")
                .1;
            let state = &self
                .states
                .iter()
                .find(|(k, _)| *k == kind)
                .expect("its kind")
                .1;
            let form = held_as(device, &made.texture);
            let what = format!(
                "{} at byte {at}, attachment 1 RG16Float {side}x{side} {load:?}/DontCare ({form})",
                shape.what
            );
            let cell = |how: &str| format!("{what}: {how}");
            let width = size[0];
            let noted = |how: &str, got: &Result<Vec<u8>, String>, want: &[u8]| match kind {
                Kind::Rgb10a2 => format!("{what}: {how}{}", steps_off(got, want, RGBA10, width)),
                Kind::Bgra8 | Kind::R8 => cell(how),
            };
            let words = |of: &dyn Fn(u32, u32) -> [u32; 4]| each(size, |x, y| kind.word(of(x, y)));
            let stale = words(&|x, y| kind.channels(x, y, true));
            let at_size = (&made.texture, size);
            let got = sampled(name, on, &self.readers, at_size, &kind.pack(), &stale);
            let how = "sampled first, the CPU's complement";
            cells.check(&noted(how, &got, &stale), got, &stale, (4, width));
            if let Some(view) = &made.view {
                let got = sampled(name, on, &self.readers, (view, size), &ALPHA, &stale);
                let how = "its A8Unorm view sampled first, the CPU's complement";
                cells.check(&cell(how), got, &stale, (4, width));
            }
            let pass = pass_of(&made.texture, kind, (second, load));
            let layouts = laid(device, &pass);
            let copies = || device.counters().metrics().over_copies;
            let before = copies();
            let rendered = render(name, self.queue, state, (&pass, &made.texture, size));
            println!(
                "real-size: {name}: {what}: {layouts}; {} over-copies counted in its pass",
                copies() - before
            );
            if let Err(why) = rendered {
                cells.check(&cell("the pass"), Err(why), &[], (1, 1));
                return;
            }
            held.texels((shape, at), |x, y| after(kind, size, x, y), false);
            let texel = kind.texel();
            let image = each(size, |x, y| kind.bytes(after(kind, size, x, y)));
            let block = held.read();
            let place = (shape, at);
            let got = Ok(texels_in(&block, place));
            let how = "(a) the block's bytes the CPU reads after synchronizeResource";
            cells.check(
                &noted(how, &got, &image),
                got,
                &image,
                (texel as usize, width),
            );
            let got = copy_out(name, on, at_size, texel, &image);
            let how = "(b) a blit of the texture into another buffer";
            cells.check(
                &noted(how, &got, &image),
                got,
                &image,
                (texel as usize, width),
            );
            let want = words(&|x, y| after(kind, size, x, y));
            let got = sampled(name, on, &self.readers, at_size, &kind.pack(), &want);
            let how = "(c) sampled last";
            cells.check(&noted(how, &got, &want), got, &want, (4, width));
            if let Some(view) = &made.view {
                let got = sampled(name, on, &self.readers, (view, size), &ALPHA, &want);
                let how = "(c) its A8Unorm view sampled last";
                cells.check(&cell(how), got, &want, (4, width));
            }
            let got = Ok(outside(block, place));
            let want = outside(held.expected.clone(), place);
            let how = "the block's bytes outside its texels, each byte (x of 4 KiB page y)";
            cells.check(&cell(how), got, &want, (1, 4096));
        }
    }

    /// Render targets over buffers in one Shared block of rt's own, side by side and then over
    /// the bytes of one released, each sampled, drawn in part by a pass that clears it, and
    /// synchronized; then read by the CPU from the block, by a blit out and by sampling, every
    /// texel the pattern inside the draw's rectangle and the clear outside it, and the block's
    /// other bytes as they were; at the end the whole block. Each line names the texture's form.
    #[test]
    fn every_device_renders_icon_targets_over_one_shared_block_every_texel() {
        let Some((packed, _, own)) = libraries() else {
            return;
        };
        assert!(BORDER.iter().all(|b| *b > 32 && b % 32 != 0), "{BORDER:?}");
        each_device("real_size_block_renders", |name, _, device| {
            let first = IndexClass::Texture.range().start;
            (device.grant_indices(IndexClass::Texture, first, 10))
                .unwrap_or_else(|error| panic!("{name}: a texture grant: {error}"));
            let mut indices = first..first + 10;
            let queue = device.new_command_queue(4).expect("a queue");
            let second = |side: u32, index: Option<u32>| {
                let format = PixelFormat::RG16Float;
                let mut d = TextureDescriptor::new(TextureType::Type2D, format, side, side);
                d.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
                d.storage_mode = StorageMode::Private;
                (device.new_texture(&d, index))
                    .unwrap_or_else(|error| panic!("{name}: RG16Float {side}x{side}: {error}"))
            };
            let rig = Rig {
                name,
                device: &device,
                queue: &queue,
                readers: Readers::new(name, &device, &packed),
                states: [Kind::Bgra8, Kind::R8, Kind::Rgb10a2]
                    .map(|kind| (kind, drawing(name, &device, &own, kind))),
                seconds: [256, 258].map(|side| (side, second(side, indices.next()))),
            };
            let block = (device.take_block(MemoryClass::Shared, BLOCK))
                .unwrap_or_else(|error| panic!("{name}: a Shared block of {BLOCK}: {error}"));
            let mut held = Held::new(&block);
            let mut cells = Cells::new(name);
            let stale = |(shape, at, _): Place, held: &mut Held| {
                let kind = shape.kind;
                held.texels((shape, at), |x, y| kind.channels(x, y, true), true);
            };
            for place in SIDE_BY_SIDE {
                stale(place, &mut held);
            }
            let mut made: Vec<Made> = (SIDE_BY_SIDE.iter())
                .map(|&place| Made::new(name, &device, &block, place, &mut indices))
                .collect();
            for one in &made {
                rig.run(one, &mut held, &mut cells);
            }
            drop(made.remove(RELEASED));
            for place in OVER_IT {
                stale(place, &mut held);
            }
            for &place in &OVER_IT {
                let one = Made::new(name, &device, &block, place, &mut indices);
                rig.run(&one, &mut held, &mut cells);
                made.push(one);
            }
            let how = "the whole block after every pass, each byte (x of 4 KiB page y)";
            cells.check(how, Ok(held.read()), &held.expected, (1, 4096));
            drop((made, rig));
            (device.release_block(block)).expect("a release");
            cells.all_exact();
        });
    }

    /// A pass of two targets, `RGBA16Float` and `RG16Float`, Private and of rt's own, each read
    /// in turn by a draw of the pass while it is the pass's attachment: their size, and whether
    /// attachment 0 is the `RGBA16Float` texture's own-format view, cleared, whose texture is then
    /// downloaded into a Private buffer, or the texture itself, loaded.
    struct Pair {
        what: &'static str,
        size: [u32; 2],
        viewed: bool,
    }

    /// The two-target passes: 256x256 through a view, and 448x256.
    const PAIRS: [Pair; 2] = [
        Pair {
            what: "RGBA16Float 256x256 through its own-format view (Clear/Store), RG16Float \
                   256x256 (Clear/DontCare)",
            size: [256, 256],
            viewed: true,
        },
        Pair {
            what: "RGBA16Float 448x256 (Load/Store), RG16Float 448x256 (Clear/DontCare)",
            size: [448, 256],
            viewed: false,
        },
    ];

    /// The `k` of each channel attachment 0 is cleared to, and of attachment 1's red and green,
    /// each `1 + k / 1024`.
    const FRONT: [u32; 4] = [32, 96, 160, 224];
    const BACK: [u32; 2] = [300, 700];

    /// What a loop fragment takes less what it reads, `2 + 1023 / 1024`: it writes the read's
    /// complement (`1023 - k`) where the read holds `1 + k / 1024`.
    const MINUEND: f32 = 2.0 + 1023.0 / 1024.0;

    /// `1 + k / 1024`.
    fn float_colour(k: u32) -> f64 {
        1.0 + f64::from(k) / 1024.0
    }

    /// Buffer 1's packing, which the float readers do not read past the size.
    const UNPACKED: Pack = Pack {
        scale: [0.0; 4],
        shift: [0; 4],
        mask: [0; 4],
    };

    /// The render state of `cover` and `fragment` into an `RGBA16Float` and an `RG16Float`
    /// target.
    fn pair_state(
        name: &str,
        device: &Device,
        library: &Library,
        fragment: &str,
    ) -> RenderPipelineState {
        let function = |f: &str| {
            StageSource::Function(FunctionDescriptor::new(
                library.function(f).expect("the function"),
            ))
        };
        let mut d = RenderPipelineDescriptor::new(function("cover"));
        d.fragment = Some(function(fragment));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float));
        d.color_attachments[1] = Some(ColorAttachmentDescriptor::new(PixelFormat::RG16Float));
        let state = (device.new_render_pipeline_state(&d, crate::tests::no_constexpr_samplers))
            .unwrap_or_else(|error| panic!("{name}: {fragment}: {error}"));
        (state.finish_pipeline()).unwrap_or_else(|error| panic!("{name}: {fragment}: {error}"));
        state
    }

    /// What a two-target pass draws with: `float_pair`'s state, the loops' states (attachment 0
    /// read, attachment 1 read), the sampler and the loops' constants.
    struct Looping<'a> {
        pattern: RenderPipelineState,
        loops: [RenderPipelineState; 2],
        sampler: &'a SamplerState,
    }

    impl Looping<'_> {
        /// One pass over `targets`: attachment 0 by `load` and stored, attachment 1 cleared and
        /// not stored; the float pattern drawn into both, cut to [`drawn`]; after a memory
        /// barrier, attachment `read` sampled by its loop over the whole of both.
        fn pass(
            &self,
            name: &str,
            queue: &CommandQueue,
            (front, back): (&Texture, &Texture),
            (load, size, read): (LoadAction, [u32; 2], usize),
        ) -> Result<(), String> {
            let mut pass = RenderPassDescriptor::default();
            let first = &mut pass.color_attachments[0];
            first.texture = Some(front.clone());
            first.load_action = load;
            first.store_action = StoreAction::Store;
            let [r, g, b, a] = FRONT.map(float_colour);
            first.clear_color = ClearColor::new(r, g, b, a);
            let other = &mut pass.color_attachments[1];
            other.texture = Some(back.clone());
            other.load_action = LoadAction::Clear;
            other.store_action = StoreAction::DontCare;
            let [r, g] = BACK.map(float_colour);
            other.clear_color = ClearColor::new(r, g, 0.0, 0.0);
            let [x, y, width, height] = drawn(size).map(u64::from);
            let constants: Vec<u8> = [MINUEND; 4]
                .into_iter()
                .chain([1.0 / size[0] as f32, 1.0 / size[1] as f32, 0.0, 0.0])
                .flat_map(f32::to_le_bytes)
                .collect();
            let command_buffer = queue.command_buffer().map_err(said)?;
            let mut encoder = (command_buffer.render_command_encoder(&pass)).map_err(said)?;
            encoder
                .set_render_pipeline_state(&self.pattern)
                .map_err(said)?;
            encoder.set_scissor_rect(ScissorRect {
                x,
                y,
                width,
                height,
            });
            (encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)).map_err(said)?;
            encoder.memory_barrier().map_err(said)?;
            encoder.set_scissor_rect(ScissorRect {
                x: 0,
                y: 0,
                width: u64::from(size[0]),
                height: u64::from(size[1]),
            });
            encoder
                .set_render_pipeline_state(&self.loops[read])
                .map_err(said)?;
            let reading = [front, back][read];
            (encoder.set_fragment_texture(Some(reading), 0)).map_err(said)?;
            (encoder.set_fragment_sampler_state(Some(self.sampler), 0)).map_err(said)?;
            (encoder.set_fragment_bytes(&constants, 0)).map_err(said)?;
            (encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)).map_err(said)?;
            encoder.end_encoding();
            completed(name, &command_buffer)
        }
    }

    /// Every texel of `texture`, of `size`, copied by a blit into a Private buffer in packed rows,
    /// then by a blit of that buffer into a Shared one first filled with `want`'s complement: the
    /// Shared buffer's bytes, or what refused or failed.
    fn download(
        name: &str,
        (device, queue): (&Device, &CommandQueue),
        (texture, size): (&Texture, [u32; 2]),
        want: &[u8],
    ) -> Result<Vec<u8>, String> {
        let row = u64::from(size[0]) * 8;
        let len = row * u64::from(size[1]);
        let private = (device.take_block(MemoryClass::Private, len)).map_err(said)?;
        let held = (device.new_buffer(private.id, 0, len, None)).map_err(said)?;
        let shared = (device.take_block(MemoryClass::Shared, len)).map_err(said)?;
        let out = (device.new_buffer(shared.id, 0, len, None)).map_err(said)?;
        let contents = shared.contents.expect("a Shared block");
        let stale: Vec<u8> = want.iter().map(|byte| !byte).collect();
        // SAFETY: the block's `len` bytes, which no command reaches yet.
        unsafe { std::ptr::copy_nonoverlapping(stale.as_ptr(), contents.as_ptr(), stale.len()) };
        let command_buffer = queue.command_buffer().map_err(said)?;
        let mut encoder = command_buffer.blit_command_encoder().map_err(said)?;
        let layout = BufferLayout {
            offset: 0,
            bytes_per_row: row,
            bytes_per_image: 0,
        };
        let whole = [size[0], size[1], 1];
        let place = TexturePlace::default();
        (encoder.copy_from_texture_to_buffer(texture, place, whole, &held, layout))
            .map_err(said)?;
        encoder.end_encoding();
        let mut encoder = command_buffer.blit_command_encoder().map_err(said)?;
        (encoder.copy_from_buffer_to_buffer(&held, 0, &out, 0, len)).map_err(said)?;
        encoder.end_encoding();
        completed(name, &command_buffer)?;
        // SAFETY: the block's `len` bytes; the command buffer that wrote them completed.
        let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), len as usize) };
        let bytes = bytes.to_vec();
        drop((held, out));
        (device.release_block(private)).expect("a release");
        (device.release_block(shared)).expect("a release");
        Ok(bytes)
    }

    /// Two-target passes ([`PAIRS`]) whose draws read each attachment while it is the pass's:
    /// the float pattern drawn into both targets, cut to [`drawn`], over attachment 0's clear (or
    /// its load of the pattern's complement) and attachment 1's clear; then, after a memory
    /// barrier, a draw over the whole of both that samples attachment 0 (or 1) at each pixel's
    /// centre and writes the complement of what it read (attachment 1's red and green twice).
    /// After each pass attachment 0's texture is copied out, attachment 0 sampled, and the
    /// 256x256 texture downloaded into a Private buffer and read out of it, every texel compared.
    /// A device without attachment reads, which refuses such a draw typed, says so and runs none.
    #[test]
    fn every_device_samples_both_targets_of_a_two_target_pass_in_it_every_texel() {
        let Some((packed, floats, own)) = libraries() else {
            return;
        };
        each_device("real_size_two_target_loops", |name, _, device| {
            let (reads, looped) = {
                let core = device.core();
                (core.attachment_reads, core.feedback_loop)
            };
            if !reads {
                println!("real-size: {name}: no attachment reads; the two-target cells not run");
                return;
            }
            let readers = Readers::new(name, &device, &packed);
            let sample = kernel(name, &device, &floats, "float_sample");
            let first = IndexClass::Texture.range().start;
            (device.grant_indices(IndexClass::Texture, first, 5))
                .unwrap_or_else(|error| panic!("{name}: a texture grant: {error}"));
            let mut indices = first..first + 5;
            let queue = device.new_command_queue(4).expect("a queue");
            let on = (&device, &queue);
            let looping = Looping {
                pattern: pair_state(name, &device, &own, "float_pair"),
                loops: ["loop_front", "loop_back"].map(|f| pair_state(name, &device, &own, f)),
                sampler: &readers.sampler,
            };
            let form = match looped {
                true => "through the feedback loop",
                false => "through copies into a shadow",
            };
            let mut cells = Cells::new(name);
            for pair in &PAIRS {
                let target = |format: PixelFormat, index: Option<u32>| {
                    let [width, height] = pair.size;
                    let mut d = TextureDescriptor::new(TextureType::Type2D, format, width, height);
                    d.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
                    d.storage_mode = StorageMode::Private;
                    (device.new_texture(&d, index))
                        .unwrap_or_else(|error| panic!("{name}: {}: {error}", pair.what))
                };
                let texture = target(PixelFormat::RGBA16Float, indices.next());
                let back = target(PixelFormat::RG16Float, indices.next());
                let view = pair.viewed.then(|| {
                    (texture.new_texture_view_with_swizzle(
                        PixelFormat::RGBA16Float,
                        TextureType::Type2D,
                        0..1,
                        0..1,
                        TextureSwizzleChannels::DEFAULT,
                        indices.next(),
                    ))
                    .unwrap_or_else(|error| panic!("{name}: {}: the view: {error}", pair.what))
                });
                let front = view.as_ref().unwrap_or(&texture);
                let size = pair.size;
                let width = size[0];
                let each_texel = |of: &dyn Fn(u32, u32) -> [u32; 4], word: bool| {
                    each(size, |x, y| {
                        let k = of(x, y);
                        match word {
                            true => k
                                .into_iter()
                                .flat_map(|k| single(k).to_le_bytes())
                                .collect(),
                            false => k.into_iter().flat_map(|k| half(k).to_le_bytes()).collect(),
                        }
                    })
                };
                let stale = |x, y| float_pattern(x, y).map(|k| 1023 - k);
                let load = match pair.viewed {
                    true => LoadAction::Clear,
                    false => LoadAction::Load,
                };
                for read in [0, 1] {
                    let base = |x, y| match pair.viewed {
                        true => FRONT,
                        false => stale(x, y),
                    };
                    if !pair.viewed {
                        let at = (&texture, size);
                        let image = each_texel(&stale, false);
                        if let Err(why) = copy_in(name, on, at, 8, &image) {
                            cells.check(
                                &format!("{}: the blit in", pair.what),
                                Err(why),
                                &[],
                                (1, 1),
                            );
                            continue;
                        }
                    }
                    let counters = device.counters();
                    let before = counters.metrics();
                    let passed = looping.pass(name, &queue, (front, &back), (load, size, read));
                    let moved = counters.metrics();
                    let what = format!(
                        "{}, attachment {read} sampled in its own pass ({form}, \
                         attachment_loop_reads {}, attachment_read_copies {})",
                        pair.what,
                        moved.attachment_loop_reads - before.attachment_loop_reads,
                        moved.attachment_read_copies - before.attachment_read_copies
                    );
                    let cell = |how: &str| format!("{what}: {how}");
                    if let Err(why) = passed {
                        cells.check(&cell("the pass"), Err(why), &[], (1, 1));
                        continue;
                    }
                    let [left, top, w, h] = drawn(size);
                    let inside = |x: u32, y: u32| {
                        (left..left + w).contains(&x) && (top..top + h).contains(&y)
                    };
                    let after = |x: u32, y: u32| -> [u32; 4] {
                        match read {
                            0 => match inside(x, y) {
                                true => float_pattern(x, y),
                                false => base(x, y),
                            }
                            .map(|k| 1023 - k),
                            _ => {
                                let [r, g, ..] = match inside(x, y) {
                                    true => float_pattern(x, y),
                                    false => [BACK[0], BACK[1], 0, 0],
                                };
                                [r, g, r, g].map(|k| 1023 - k)
                            }
                        }
                    };
                    let image = each_texel(&after, false);
                    let words = each_texel(&after, true);
                    let got = copy_out(name, on, (&texture, size), 8, &image);
                    cells.check(&cell("its texture copied out"), got, &image, (8, width));
                    let through = (&sample, Some(&readers.sampler));
                    let got = kernel_read(name, on, through, (front, size), &UNPACKED, &words);
                    cells.check(&cell("attachment 0 sampled"), got, &words, (16, width));
                    if pair.viewed {
                        let got = download(name, on, (&texture, size), &image);
                        let how = "its texture downloaded into a Private buffer in rows of 2048 \
                                   bytes, then copied out of it";
                        cells.check(&cell(how), got, &image, (8, width));
                    }
                }
            }
            cells.all_exact();
        });
    }

    /// How a layer's or a screen's texels are held, drawn and read: `BGRA8Unorm` and
    /// `RGBA16Unorm` of the 8-bit pattern (each channel's `k` times 257 in 16 bits), and
    /// `BGR10A2Unorm` of the 10-bit one.
    #[derive(Clone, Copy)]
    enum Layer {
        Bgra8,
        Rgba16,
        Bgr10a2,
    }

    impl Layer {
        fn format(self) -> PixelFormat {
            match self {
                Layer::Bgra8 => PixelFormat::BGRA8Unorm,
                Layer::Rgba16 => PixelFormat::RGBA16Unorm,
                Layer::Bgr10a2 => PixelFormat::BGR10A2Unorm,
            }
        }

        /// A texel's bytes.
        fn texel(self) -> u32 {
            match self {
                Layer::Rgba16 => 8,
                Layer::Bgra8 | Layer::Bgr10a2 => 4,
            }
        }

        /// The fragment that draws the pattern into one target.
        fn fragment(self) -> &'static str {
            match self {
                Layer::Bgra8 => "unorm8_one",
                Layer::Rgba16 => "unorm16_one",
                Layer::Bgr10a2 => "unorm10_one",
            }
        }

        /// Each channel's largest value, in shader order.
        fn largest(self) -> [u32; 4] {
            match self {
                Layer::Bgra8 => [255; 4],
                Layer::Rgba16 => [65535; 4],
                Layer::Bgr10a2 => [1023, 1023, 1023, 3],
            }
        }

        /// Each channel's largest value as `loop_one` takes it.
        fn steps(self) -> [f32; 4] {
            self.largest().map(|largest| largest as f32)
        }

        /// The part of a step each write of this layer adds: a quarter for 8 and 16 bits, none
        /// for 10.
        fn up(self) -> f32 {
            match self {
                Layer::Bgra8 | Layer::Rgba16 => 0.25,
                Layer::Bgr10a2 => 0.0,
            }
        }

        /// The pattern's channels at `(x, y)`, or their complement, in shader order.
        fn channels(self, x: u32, y: u32, complemented: bool) -> [u32; 4] {
            let k = match self {
                Layer::Bgra8 => unorm_pattern(8, 8, x, y),
                Layer::Rgba16 => unorm_pattern(8, 8, x, y).map(|k| k * 257),
                Layer::Bgr10a2 => unorm_pattern(10, 2, x, y),
            };
            let largest = self.largest();
            std::array::from_fn(|c| match complemented {
                true => largest[c] - k[c],
                false => k[c],
            })
        }

        /// The channels a pass clears it to, which neither the pattern nor its complement holds
        /// at any texel of these sizes.
        fn clear(self) -> [u32; 4] {
            match self {
                Layer::Bgra8 => [32, 96, 160, 224],
                Layer::Rgba16 => [32, 96, 160, 224].map(|k| k * 257),
                Layer::Bgr10a2 => [32, 96, 160, 2],
            }
        }

        /// The clear colour of [`clear`](Self::clear): `k` and [`up`](Self::up) over each
        /// channel's largest.
        fn clear_color(self) -> ClearColor {
            let (clear, largest, up) = (self.clear(), self.largest(), f64::from(self.up()));
            let [r, g, b, a] =
                std::array::from_fn(|c| (f64::from(clear[c]) + up) / f64::from(largest[c]));
            ClearColor::new(r, g, b, a)
        }

        /// The bytes in memory of a texel of channels `k`: `BGRA8Unorm` blue, green, red, alpha;
        /// `RGBA16Unorm` each channel little-endian; `BGR10A2Unorm` one little-endian word, blue
        /// lowest.
        fn bytes(self, [r, g, b, a]: [u32; 4]) -> Vec<u8> {
            match self {
                Layer::Bgra8 => [b, g, r, a].map(|c| c as u8).to_vec(),
                Layer::Rgba16 => [r, g, b, a]
                    .into_iter()
                    .flat_map(|c| (c as u16).to_le_bytes())
                    .collect(),
                Layer::Bgr10a2 => (b | (g << 10) | (r << 20) | (a << 30))
                    .to_le_bytes()
                    .to_vec(),
            }
        }

        /// How each sampled read of a texel packs it: one read, or for `RGBA16Unorm` two, its
        /// red and green, then its blue and alpha.
        fn packs(self) -> Vec<Pack> {
            match self {
                Layer::Bgra8 => vec![Pack::unorm(8, 4)],
                Layer::Rgba16 => vec![
                    Pack {
                        scale: [65535.0, 65535.0, 0.0, 0.0],
                        shift: [0, 16, 0, 0],
                        mask: [0xffff, 0xffff << 16, 0, 0],
                    },
                    Pack {
                        scale: [0.0, 0.0, 65535.0, 65535.0],
                        shift: [0, 0, 0, 16],
                        mask: [0, 0, 0xffff, 0xffff << 16],
                    },
                ],
                Layer::Bgr10a2 => vec![Kind::Rgb10a2.pack()],
            }
        }

        /// The word read `read` of [`packs`](Self::packs) makes of channels `k`.
        fn word(self, read: usize, [r, g, b, a]: [u32; 4]) -> u32 {
            match (self, read) {
                (Layer::Bgra8, _) => r | (g << 8) | (b << 16) | (a << 24),
                (Layer::Rgba16, 0) => r | (g << 16),
                (Layer::Rgba16, _) => b | (a << 16),
                (Layer::Bgr10a2, _) => r | (g << 10) | (b << 20) | (a << 30),
            }
        }
    }

    /// The render state of `cover` and `fragment` into one colour target of `format`.
    fn one_state(
        name: &str,
        device: &Device,
        library: &Library,
        (fragment, format): (&str, PixelFormat),
    ) -> RenderPipelineState {
        let function = |f: &str| {
            StageSource::Function(FunctionDescriptor::new(
                library.function(f).expect("the function"),
            ))
        };
        let mut d = RenderPipelineDescriptor::new(function("cover"));
        d.fragment = Some(function(fragment));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(format));
        let state = (device.new_render_pipeline_state(&d, crate::tests::no_constexpr_samplers))
            .unwrap_or_else(|error| panic!("{name}: {fragment}: {error}"));
        (state.finish_pipeline()).unwrap_or_else(|error| panic!("{name}: {fragment}: {error}"));
        state
    }

    /// A Private render target of rt's own, read by shaders, of `layer` at `size` with `levels`
    /// levels, at `index`.
    fn own_target(
        name: &str,
        device: &Device,
        (layer, size, levels): (Layer, [u32; 2], u32),
        index: Option<u32>,
    ) -> Texture {
        let [width, height] = size;
        let format = layer.format();
        let mut d = TextureDescriptor::new(TextureType::Type2D, format, width, height);
        d.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        d.storage_mode = StorageMode::Private;
        d.mipmap_level_count = levels;
        (device.new_texture(&d, index))
            .unwrap_or_else(|error| panic!("{name}: {format:?} {width}x{height}: {error}"))
    }

    /// Every texel of `texture`, of `layer` at `size` and holding `of(x, y)`, sampled by each of
    /// the layer's reads, each a cell named by `what` and `how`.
    fn sample_all(
        (name, on, readers): (&str, (&Device, &CommandQueue), &Readers),
        cells: &mut Cells,
        (texture, layer, size): (&Texture, Layer, [u32; 2]),
        of: &dyn Fn(u32, u32) -> [u32; 4],
        (what, how): (&str, &str),
    ) {
        for (read, pack) in layer.packs().iter().enumerate() {
            let want = each(size, |x, y| {
                layer.word(read, of(x, y)).to_le_bytes().to_vec()
            });
            let got = sampled(name, on, readers, (texture, size), pack, &want);
            let part = match layer {
                Layer::Rgba16 => [" (red and green)", " (blue and alpha)"][read],
                Layer::Bgra8 | Layer::Bgr10a2 => "",
            };
            let off = match layer {
                Layer::Bgr10a2 => steps_off(&got, &want, RGBA10, size[0]),
                Layer::Bgra8 | Layer::Rgba16 => String::new(),
            };
            cells.check(
                &format!("{what}: {how}{part}{off}"),
                got,
                &want,
                (4, size[0]),
            );
        }
    }

    /// Draws `pattern` into `pass`'s one target, cut to [`drawn`] of `size`; then, where `looped`
    /// names one, after a memory barrier its loop over the whole target, sampling the texture it
    /// names through the sampler it names as its attachment, writing as the layer it names does.
    fn draw_one(
        name: &str,
        queue: &CommandQueue,
        (pass, pattern, size): (&RenderPassDescriptor, &RenderPipelineState, [u32; 2]),
        looped: Option<(&RenderPipelineState, &Texture, &SamplerState, Layer)>,
    ) -> Result<(), String> {
        let [x, y, width, height] = drawn(size).map(u64::from);
        let command_buffer = queue.command_buffer().map_err(said)?;
        let mut encoder = (command_buffer.render_command_encoder(pass)).map_err(said)?;
        encoder.set_render_pipeline_state(pattern).map_err(said)?;
        encoder.set_scissor_rect(ScissorRect {
            x,
            y,
            width,
            height,
        });
        (encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)).map_err(said)?;
        if let Some((state, texture, sampler, layer)) = looped {
            let constants: Vec<u8> = (layer.steps().into_iter())
                .chain([1.0 / size[0] as f32, 1.0 / size[1] as f32, 0.0, 0.0])
                .chain([layer.up(); 4])
                .flat_map(f32::to_le_bytes)
                .collect();
            encoder.memory_barrier().map_err(said)?;
            encoder.set_scissor_rect(ScissorRect {
                x: 0,
                y: 0,
                width: u64::from(size[0]),
                height: u64::from(size[1]),
            });
            encoder.set_render_pipeline_state(state).map_err(said)?;
            (encoder.set_fragment_texture(Some(texture), 0)).map_err(said)?;
            (encoder.set_fragment_sampler_state(Some(sampler), 0)).map_err(said)?;
            (encoder.set_fragment_bytes(&constants, 0)).map_err(said)?;
            (encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)).map_err(said)?;
        }
        encoder.end_encoding();
        completed(name, &command_buffer)
    }

    /// The layers icons are drawn through, Private targets of rt's own: `BGRA8Unorm` at 256x128
    /// and at 256x192 of three levels, and `RGBA16Unorm` at 256x128.
    const LAYERS: [(Layer, [u32; 2], u32); 3] = [
        (Layer::Bgra8, [256, 128], 1),
        (Layer::Bgra8, [256, 192], 3),
        (Layer::Rgba16, [256, 128], 1),
    ];

    /// Layers ([`LAYERS`]) each holding the pattern's complement by a blit, then drawn by a pass
    /// that clears level 0 and draws the pattern into part of it, cut to [`drawn`]: copied out by
    /// a blit and sampled, every texel the pattern inside the rectangle and the clear outside.
    #[test]
    fn every_device_draws_icon_layers_in_part_after_a_clear_every_texel() {
        let Some((packed, _, own)) = libraries() else {
            return;
        };
        each_device("real_size_layers", |name, _, device| {
            let readers = Readers::new(name, &device, &packed);
            let first = IndexClass::Texture.range().start;
            (device.grant_indices(IndexClass::Texture, first, LAYERS.len() as u32))
                .unwrap_or_else(|error| panic!("{name}: a texture grant: {error}"));
            let queue = device.new_command_queue(4).expect("a queue");
            let on = (&device, &queue);
            let mut cells = Cells::new(name);
            for (index, &(layer, size, levels)) in (first..).zip(LAYERS.iter()) {
                let [width, height] = size;
                let what = format!("{:?} {width}x{height} of {levels} levels", layer.format());
                let texture = own_target(name, &device, (layer, size, levels), Some(index));
                let texel = layer.texel();
                let stale = each(size, |x, y| layer.bytes(layer.channels(x, y, true)));
                if let Err(why) = copy_in(name, on, (&texture, size), texel, &stale) {
                    cells.check(&format!("{what}: the blit in"), Err(why), &[], (1, 1));
                    continue;
                }
                let state = one_state(name, &device, &own, (layer.fragment(), layer.format()));
                let mut pass = RenderPassDescriptor::default();
                let target = &mut pass.color_attachments[0];
                target.texture = Some(texture.clone());
                target.load_action = LoadAction::Clear;
                target.store_action = StoreAction::Store;
                target.clear_color = layer.clear_color();
                println!("real-size: {name}: {what}: {}", laid(&device, &pass));
                if let Err(why) = draw_one(name, &queue, (&pass, &state, size), None) {
                    cells.check(&format!("{what}: the pass"), Err(why), &[], (1, 1));
                    continue;
                }
                let [left, top, w, h] = drawn(size);
                let after = |x: u32, y: u32| {
                    let inside = (left..left + w).contains(&x) && (top..top + h).contains(&y);
                    match inside {
                        true => layer.channels(x, y, false),
                        false => layer.clear(),
                    }
                };
                let image = each(size, |x, y| layer.bytes(after(x, y)));
                let got = copy_out(name, on, (&texture, size), texel, &image);
                let how = "level 0 copied out after a clear and a draw of part of it";
                cells.check(
                    &format!("{what}: {how}"),
                    got,
                    &image,
                    (texel as usize, width),
                );
                let at = (&texture, layer, size);
                sample_all(
                    (name, on, &readers),
                    &mut cells,
                    at,
                    &after,
                    (&what, "sampled"),
                );
            }
            cells.all_exact();
        });
    }

    /// The screens icons are drawn into, Private targets of rt's own at 2560x1600, each rendered
    /// through its own-format view (identity swizzle): `BGRA8Unorm` and `BGR10A2Unorm`.
    const SCREENS: [Layer; 2] = [Layer::Bgra8, Layer::Bgr10a2];

    /// The screens' size.
    const SCREEN: [u32; 2] = [2560, 1600];

    /// Screens ([`SCREENS`]) holding the pattern's complement by a blit, each drawn through its
    /// view by a pass that loads it, draws the pattern into part of it, cut to [`drawn`], and
    /// after a memory barrier samples the view, its own attachment, at each pixel's centre over
    /// the whole of it, writing the complement of what it read: every texel then holds the
    /// pattern's complement inside the rectangle and the pattern outside. Copied out by a blit
    /// and sampled through the view, every texel compared. A device without attachment reads,
    /// which refuses such a draw typed, says so and runs none.
    #[test]
    fn every_device_samples_a_screen_through_its_view_in_its_own_pass_every_texel() {
        let Some((packed, _, own)) = libraries() else {
            return;
        };
        each_device("real_size_screens", |name, _, device| {
            let (reads, looped) = {
                let core = device.core();
                (core.attachment_reads, core.feedback_loop)
            };
            if !reads {
                println!("real-size: {name}: no attachment reads; the screen cells not run");
                return;
            }
            let readers = Readers::new(name, &device, &packed);
            let first = IndexClass::Texture.range().start;
            let count = 2 * SCREENS.len() as u32;
            (device.grant_indices(IndexClass::Texture, first, count))
                .unwrap_or_else(|error| panic!("{name}: a texture grant: {error}"));
            let mut indices = first..first + count;
            let queue = device.new_command_queue(4).expect("a queue");
            let on = (&device, &queue);
            let form = match looped {
                true => "through the feedback loop",
                false => "through copies into a shadow",
            };
            let mut cells = Cells::new(name);
            let size = SCREEN;
            let [width, height] = size;
            for &layer in &SCREENS {
                let format = layer.format();
                let texture = own_target(name, &device, (layer, size, 1), indices.next());
                let view = (texture.new_texture_view_with_swizzle(
                    format,
                    TextureType::Type2D,
                    0..1,
                    0..1,
                    TextureSwizzleChannels::DEFAULT,
                    indices.next(),
                ))
                .unwrap_or_else(|error| panic!("{name}: {format:?}: the view: {error}"));
                let texel = layer.texel();
                let stale = each(size, |x, y| layer.bytes(layer.channels(x, y, true)));
                let base = format!("{format:?} {width}x{height} through its view");
                if let Err(why) = copy_in(name, on, (&texture, size), texel, &stale) {
                    cells.check(&format!("{base}: the blit in"), Err(why), &[], (1, 1));
                    continue;
                }
                let pattern = one_state(name, &device, &own, (layer.fragment(), format));
                let looping = one_state(name, &device, &own, ("loop_one", format));
                let mut pass = RenderPassDescriptor::default();
                let target = &mut pass.color_attachments[0];
                target.texture = Some(view.clone());
                target.load_action = LoadAction::Load;
                target.store_action = StoreAction::Store;
                let counters = device.counters();
                let before = counters.metrics();
                let loops = Some((&looping, &view, &readers.sampler, layer));
                let drew = draw_one(name, &queue, (&pass, &pattern, size), loops);
                let moved = counters.metrics();
                let what = format!(
                    "{base}, sampled in its own pass ({form}, attachment_loop_reads {}, \
                     attachment_read_copies {})",
                    moved.attachment_loop_reads - before.attachment_loop_reads,
                    moved.attachment_read_copies - before.attachment_read_copies
                );
                if let Err(why) = drew {
                    cells.check(&format!("{what}: the pass"), Err(why), &[], (1, 1));
                    continue;
                }
                let [left, top, w, h] = drawn(size);
                let after = |x: u32, y: u32| {
                    let inside = (left..left + w).contains(&x) && (top..top + h).contains(&y);
                    layer.channels(x, y, inside)
                };
                let image = each(size, |x, y| layer.bytes(after(x, y)));
                let got = copy_out(name, on, (&texture, size), texel, &image);
                let off = match layer {
                    Layer::Bgr10a2 => steps_off(&got, &image, BGRA10, width),
                    Layer::Bgra8 | Layer::Rgba16 => String::new(),
                };
                cells.check(
                    &format!("{what}: copied out{off}"),
                    got,
                    &image,
                    (texel as usize, width),
                );
                let at = (&view, layer, size);
                let how = (what.as_str(), "sampled through its view");
                sample_all((name, on, &readers), &mut cells, at, &after, how);
            }
            cells.all_exact();
        });
    }
}
