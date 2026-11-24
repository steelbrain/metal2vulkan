//! Textures over a buffer at an icon's and a picture's real size, an icon-size attachment sampled
//! in its own pass, and a real-size float picture copied out, every texel compared.
//!
//! Device-free: the float pattern ([`float_pattern`]) names its texel, each of its values is
//! exact in a half and in a float, and its complement differs from it in every channel.
//!
//! Under a device cfg, on every device of the cfg's kind (`on_device`), with the readers, the
//! cells and the 8-bit pattern of `real_size.rs`:
//!
//! - Textures over a buffer ([`Buffer::new_texture`](crate::Buffer::new_texture)) in host
//!   memory's import, which rt holds in an image of its own that the device copies the bytes into
//!   on every device, and over a Shared block of rt's own, in place where the device's linear
//!   image takes them and else copied the same way: at an icon's size
//!   (`every_device_reads_icon_size_textures_over_a_buffer_every_texel`: `BGRA8Unorm` at 48x48,
//!   76x76, 108x108, 256x256 and 1024x1024 and `RGBA16Float` at 48x48 and 256x256 in tight rows
//!   from the buffer's start, the 256x256 and 1024x1024 ones also in rows 64 bytes wider from 192
//!   bytes in, and `BGRA8Unorm` at 1024x1024 in rows of 10,240 bytes, also a render target) and at
//!   a picture's
//!   (`every_device_reads_picture_size_textures_over_a_buffer_every_texel`: `BGRA8Unorm` at
//!   3840x2160 in rows of 15,360 bytes and at 2845x1600 in rows of 11,392 bytes, three texels
//!   past the width). The CPU writes the complement into the buffer and a sampled read takes it;
//!   then the CPU writes the pattern, and the texture is read by sampling, by `read()`, by a blit
//!   out and by sampling again, every texel compared, and the buffer's bytes after, which no read
//!   may change. A copy in that a submission skips, a texel, a 4 KiB block or a row the copy
//!   misses, a wrong pitch or offset, or a kernel that runs one thread a workgroup, each shows as
//!   texels of the complement.
//! - `every_device_samples_icons_over_a_shared_block_refilled_each_frame_every_texel`: textures
//!   over a Shared block of rt's own at icon sizes (`BGRA8Unorm` at 48x48, 76x76, 108x108 and
//!   256x256, `RGBA16Float` at 48x48 and 384x384, tight rows), whose buffer the CPU rewrites before
//!   each of three frames (the pattern, its complement, the pattern), each frame sampled and every
//!   texel compared; each cell names the form the texture took, copied for its cause or in place
//!   with its linear image's row pitch and offset.
//! - `every_device_samples_its_attachment_in_its_own_pass_at_icon_size_every_texel`: a
//!   `BGRA8Unorm` and an `RGBA16Float` render target at 256x256 and 1024x1024, holding the
//!   pattern, then drawn by a pass that clears it and samples it at each pixel's centre while it
//!   is the pass's attachment, writing a constant less what it read; then by a pass that loads
//!   it, draws the pattern over it, and after a memory barrier samples it the same way. Each pass
//!   is then copied out and sampled, every texel compared: every texel holds the complement of the
//!   clear, then of the pattern. Each cell's name says how the device reads its attachment, through
//!   the feedback loop or through copies into a shadow, with the pass's
//!   `Metrics::attachment_loop_reads` and `Metrics::attachment_read_copies`.
//! - `every_device_copies_out_a_real_size_float_picture_every_texel`: an `RGBA16Float` texture at
//!   3840x2160, written whole by a blit of the complement, then by a render pass that loads it and
//!   draws the float pattern over it, and again by a kernel, each time copied out whole by a blit
//!   into a Shared buffer and into a buffer over host memory's import of 64,804 KiB, every texel
//!   compared, the import's bytes past the picture too.
//!
//! `llvm-as` assembles the shaders; without it the tests skip visibly.

/// The float pattern's values at `(x, y)`, each `k` of a channel holding `1 + k / 1024`, in the
/// order a shader reads them (red, green, blue, alpha): red `x`, green `y`, each cut to 10 bits;
/// blue the bits of both above 10, then the low 6 bits of `x ^ y`; alpha `x + y`, cut to 10
/// bits. Red and blue name `x`, green and blue name `y`, below 4,096.
fn float_pattern(x: u32, y: u32) -> [u32; 4] {
    [
        x & 1023,
        y & 1023,
        (x >> 10) | ((y >> 10) << 2) | (((x ^ y) & 63) << 4),
        (x + y) & 1023,
    ]
}

/// The bits of a half holding `1 + k / 1024`, `k` below 1,024: exponent 0, mantissa `k`.
fn half(k: u32) -> u16 {
    0x3c00 | k as u16
}

/// The bits of a float holding `1 + k / 1024`, `k` below 1,024.
fn single(k: u32) -> u32 {
    0x3f80_0000 | (k << 13)
}

/// The float pattern names its texel through the whole of each axis below 4,096, each of its
/// values is `1 + k / 1024` exactly in a half and in a float, and its complement (`1023 - k`)
/// differs from it in every channel.
#[test]
fn the_float_pattern_names_its_texel_is_exact_in_a_half_and_its_complement_differs() {
    let some = [
        0, 1, 255, 256, 1023, 1024, 1079, 1599, 2047, 2048, 2159, 2844, 3071, 3839, 4095,
    ];
    let cells = (0..4096)
        .flat_map(|x| some.map(|y| (x, y)))
        .chain(some.iter().flat_map(|&x| (0..4096).map(move |y| (x, y))));
    for (x, y) in cells {
        let [r, g, b, a] = float_pattern(x, y);
        let named = (r | ((b & 3) << 10), g | (((b >> 2) & 3) << 10));
        assert_eq!(named, (x, y), "({x}, {y})");
        for k in [r, g, b, a] {
            assert!(k < 1024, "({x}, {y}): {k}");
            let bits = half(k);
            assert_eq!(
                (bits >> 15, (bits >> 10) & 31, u32::from(bits & 1023)),
                (0, 15, k),
                "({x}, {y}): a half of exponent 0 and mantissa {k}"
            );
            assert_eq!(
                f32::from_bits(single(k)),
                1.0 + k as f32 / 1024.0,
                "({x}, {y})"
            );
            assert_ne!(1023 - k, k, "({x}, {y})");
        }
    }
}

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use super::{float_pattern, half, single};
    use crate::tests::command::on_device::{each_device, finished};
    use crate::tests::import::on_device::{free, host_memory, releasing, wait_until};
    use crate::tests::real_size::on_device::{
        copy_in, copy_out, kernel, kernel_read, read, sampled, Cells, Pack, Readers,
    };
    use crate::tests::real_size::{complement, unorm_pattern};
    use crate::texture::TextureObject;
    use crate::{
        Block, Buffer, BufferLayout, ClearColor, ColorAttachmentDescriptor, CommandBuffer,
        CommandQueue, ComputePipelineState, Device, DispatchType, Error, FunctionDescriptor,
        IndexClass, Library, LoadAction, MemoryClass, PixelFormat, PrimitiveType,
        RenderPassDescriptor, RenderPipelineDescriptor, RenderPipelineState, SamplerState,
        StageSource, StorageMode, Texture, TextureDescriptor, TexturePlace, TextureType,
        TextureUsage,
    };
    use ash::vk;
    use metal2vulkan_devtools::{llvm_assemble_text, AssembleError};
    use std::alloc::Layout;
    use std::ptr::NonNull;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    /// The owned shaders. `float_read` and `float_sample` read texel `g` of texture 0 as
    /// `packed_read` and `packed_sample` do (`real_size.rs`, with their bindings: sampler 0,
    /// buffer 0 out, buffer 1 `uint4 {width, height, ..}`), and store its `float4` at `g.y *
    /// width + g.x` of buffer 0. `pattern_write` writes the float pattern at `g` of texture 0,
    /// below buffer 0's `uint4 {width, height, ..}`. `cover` draws the triangle that covers clip
    /// space; `float_fragment` returns the float pattern at its pixel, `unorm_fragment` the 8-bit
    /// pattern as `(k + 0.25) / 255`, and `loop_fragment` samples texture 0 through sampler 0 at
    /// its pixel's centre and returns buffer 0's first `float4` less what it read, the pixel's
    /// position scaled by its second `float4`. `FLOAT` and `UNORM` stand for the lines that make
    /// `%v`, the pattern at `(%x, %y)` ([`FLOAT`], [`UNORM`]).
    const SHADERS: &str = r#"; Owned synthetic shaders.
source_filename = "real_size_overs.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @float_read(ptr addrspace(1) %t, ptr addrspace(1) noundef "air-buffer-no-alias" %out, ptr addrspace(2) noundef "air-buffer-no-alias" %p, <2 x i32> noundef %g) {
entry:
  %size = load <4 x i32>, ptr addrspace(2) %p, align 16
  %w = extractelement <4 x i32> %size, i64 0
  %h = extractelement <4 x i32> %size, i64 1
  %x = extractelement <2 x i32> %g, i64 0
  %y = extractelement <2 x i32> %g, i64 1
  %ix = icmp ult i32 %x, %w
  %iy = icmp ult i32 %y, %h
  %in = and i1 %ix, %iy
  br i1 %in, label %body, label %done

body:
  %s = tail call ptr addrspace(2) @air.get_read_sampler()
  %r = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %g, <2 x i32> zeroinitializer, i32 0, i32 1)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %at = zext i32 %i to i64
  %q = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %at
  store <4 x float> %v, ptr addrspace(1) %q, align 16
  br label %done

done:
  ret void
}

define void @float_sample(ptr addrspace(1) %t, ptr addrspace(2) %smp, ptr addrspace(1) noundef "air-buffer-no-alias" %out, ptr addrspace(2) noundef "air-buffer-no-alias" %p, <2 x i32> noundef %g) {
entry:
  %size = load <4 x i32>, ptr addrspace(2) %p, align 16
  %w = extractelement <4 x i32> %size, i64 0
  %h = extractelement <4 x i32> %size, i64 1
  %x = extractelement <2 x i32> %g, i64 0
  %y = extractelement <2 x i32> %g, i64 1
  %ix = icmp ult i32 %x, %w
  %iy = icmp ult i32 %y, %h
  %in = and i1 %ix, %iy
  br i1 %in, label %body, label %done

body:
  %wh = shufflevector <4 x i32> %size, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %whf = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %wh)
  %gf = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %g)
  %c = fadd <2 x float> %gf, <float 5.000000e-01, float 5.000000e-01>
  %uv = fdiv <2 x float> %c, %whf
  %r = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %smp, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %at = zext i32 %i to i64
  %q = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %at
  store <4 x float> %v, ptr addrspace(1) %q, align 16
  br label %done

done:
  ret void
}

define void @pattern_write(ptr addrspace(1) %t, ptr addrspace(2) noundef "air-buffer-no-alias" %p, <2 x i32> noundef %g) {
entry:
  %size = load <4 x i32>, ptr addrspace(2) %p, align 16
  %w = extractelement <4 x i32> %size, i64 0
  %h = extractelement <4 x i32> %size, i64 1
  %x = extractelement <2 x i32> %g, i64 0
  %y = extractelement <2 x i32> %g, i64 1
  %ix = icmp ult i32 %x, %w
  %iy = icmp ult i32 %y, %h
  %in = and i1 %ix, %iy
  br i1 %in, label %body, label %done

body:
FLOAT
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %t, <2 x i32> %g, <4 x float> %v, i32 0, i32 2)
  br label %done

done:
  ret void
}

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

define <4 x float> @float_fragment(<4 x float> %pos) {
entry:
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %pi = tail call <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float> %xy)
  %x = extractelement <2 x i32> %pi, i64 0
  %y = extractelement <2 x i32> %pi, i64 1
FLOAT
  ret <4 x float> %v
}

define <4 x float> @unorm_fragment(<4 x float> %pos) {
entry:
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %pi = tail call <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float> %xy)
  %x = extractelement <2 x i32> %pi, i64 0
  %y = extractelement <2 x i32> %pi, i64 1
UNORM
  ret <4 x float> %v
}

define <4 x float> @loop_fragment(<4 x float> %pos, ptr addrspace(1) %t, ptr addrspace(2) %smp, ptr addrspace(2) noundef "air-buffer-no-alias" %p) {
entry:
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %pinv = getelementptr inbounds <4 x float>, ptr addrspace(2) %p, i64 1
  %inv4 = load <4 x float>, ptr addrspace(2) %pinv, align 16
  %inv = shufflevector <4 x float> %inv4, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %uv = fmul <2 x float> %xy, %inv
  %r = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %smp, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  %s = load <4 x float>, ptr addrspace(2) %p, align 16
  %o = fsub <4 x float> %s, %v
  ret <4 x float> %o
}

declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)
declare <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32>)
declare <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float>)

!air.kernel = !{!0, !6, !12}
!air.vertex = !{!17}
!air.fragment = !{!23, !32, !33}
!air.version = !{!28}
!air.language_version = !{!29}
!0 = !{ptr @float_read, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !11}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"uint4", !"air.arg_name", !"p"}
!6 = !{ptr @float_sample, !1, !7}
!7 = !{!8, !9, !10, !30, !31}
!8 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}
!9 = !{i32 1, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"smp"}
!10 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!11 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"g"}
!12 = !{ptr @pattern_write, !1, !13}
!13 = !{!14, !15, !16}
!14 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"t"}
!15 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"uint4", !"air.arg_name", !"p"}
!16 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"g"}
!17 = !{ptr @cover, !18, !21}
!18 = !{!19, !20}
!19 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!20 = !{!"air.vertex_output", !"generated(5colorDv4_f)", !"air.arg_type_name", !"float4", !"air.arg_name", !"color"}
!21 = !{!22}
!22 = !{i32 0, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"id"}
!23 = !{ptr @float_fragment, !24, !26}
!24 = !{!25}
!25 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!26 = !{!27}
!27 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!28 = !{i32 2, i32 8, i32 0}
!29 = !{!"Metal", i32 4, i32 0, i32 0}
!30 = !{i32 3, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"uint4", !"air.arg_name", !"p"}
!31 = !{i32 4, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"g"}
!32 = !{ptr @unorm_fragment, !24, !26}
!33 = !{ptr @loop_fragment, !24, !34}
!34 = !{!27, !35, !36, !37}
!35 = !{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}
!36 = !{i32 2, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"smp"}
!37 = !{i32 3, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"p"}
"#;

    /// The lines that make `%v`, the float pattern at `(%x, %y)` ([`float_pattern`]): each
    /// channel's `k` converted, times 1/1024 and plus 1, each step exact in a float.
    const FLOAT: &str = r#"  %k0 = and i32 %x, 1023
  %k1 = and i32 %y, 1023
  %xh = lshr i32 %x, 10
  %yh = lshr i32 %y, 10
  %yh2 = shl i32 %yh, 2
  %xy.x = xor i32 %x, %y
  %xy6 = and i32 %xy.x, 63
  %xy64 = shl i32 %xy6, 4
  %k2h = or i32 %xh, %yh2
  %k2 = or i32 %k2h, %xy64
  %sum = add i32 %x, %y
  %k3 = and i32 %sum, 1023
  %f0 = uitofp i32 %k0 to float
  %f1 = uitofp i32 %k1 to float
  %f2 = uitofp i32 %k2 to float
  %f3 = uitofp i32 %k3 to float
  %kv0 = insertelement <4 x float> poison, float %f0, i64 0
  %kv1 = insertelement <4 x float> %kv0, float %f1, i64 1
  %kv2 = insertelement <4 x float> %kv1, float %f2, i64 2
  %kv = insertelement <4 x float> %kv2, float %f3, i64 3
  %kq = fmul <4 x float> %kv, <float 9.765625e-04, float 9.765625e-04, float 9.765625e-04, float 9.765625e-04>
  %v = fadd <4 x float> %kq, <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
"#;

    /// The lines that make `%v`, the 8-bit pattern at `(%x, %y)` (`real_size.rs`'s
    /// `unorm_pattern(8, 8, x, y)`), each channel's `k` as `(k + 0.25) / 255`.
    const UNORM: &str = r#"  %k0 = and i32 %x, 255
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

    /// The shaders' library, or None where `llvm-as` is missing.
    fn library() -> Option<Library> {
        let text = (SHADERS.replace("FLOAT\n", FLOAT)).replace("UNORM\n", UNORM);
        match llvm_assemble_text(&text) {
            Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the library")),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                None
            }
            Err(error) => panic!("the real-size over shaders: {error:?}"),
        }
    }

    /// The two libraries the device tests take: `real_size.rs`'s readers and [`SHADERS`]; None
    /// where `llvm-as` is missing.
    fn libraries() -> Option<(Library, Library)> {
        Some((crate::tests::real_size::on_device::library()?, library()?))
    }

    /// `real_size.rs`'s readers, and the float readers of [`SHADERS`], on a device.
    struct Reading {
        readers: Readers,
        read: ComputePipelineState,
        sample: ComputePipelineState,
    }

    impl Reading {
        fn new(name: &str, device: &Device, (packed, own): &(Library, Library)) -> Reading {
            Reading {
                readers: Readers::new(name, device, packed),
                read: kernel(name, device, own, "float_read"),
                sample: kernel(name, device, own, "float_sample"),
            }
        }

        /// Every texel of `texture` of `size` and `texels`, read by sampling (`sampling`) or by
        /// `read()`, into words of `want`'s length.
        fn read(
            &self,
            name: &str,
            on: (&Device, &CommandQueue),
            sampling: bool,
            (texture, size, texels): (&Texture, [u32; 2], Texels),
            want: &[u8],
        ) -> Result<Vec<u8>, String> {
            let at = (texture, size);
            let packed = Pack::unorm(8, 4);
            match (texels, sampling) {
                (Texels::Bgra8, true) => sampled(name, on, &self.readers, at, &packed, want),
                (Texels::Bgra8, false) => read(name, on, &self.readers, at, &packed, want),
                (Texels::Rgba16Float, true) => {
                    let through = (&self.sample, Some(&self.readers.sampler));
                    kernel_read(name, on, through, at, &UNPACKED, want)
                }
                (Texels::Rgba16Float, false) => {
                    kernel_read(name, on, (&self.read, None), at, &UNPACKED, want)
                }
            }
        }
    }

    /// Buffer 1's packing, which the float readers do not read past the size.
    const UNPACKED: Pack = Pack {
        scale: [0.0; 4],
        shift: [0; 4],
        mask: [0; 4],
    };

    /// How a texture's texels are held and read.
    #[derive(Clone, Copy)]
    enum Texels {
        /// `BGRA8Unorm`, of the 8-bit pattern (`real_size.rs`), read as a word packed red lowest
        /// (`Pack::unorm(8, 4)`).
        Bgra8,
        /// `RGBA16Float`, of the float pattern, read as a `float4`.
        Rgba16Float,
    }

    impl Texels {
        fn format(self) -> PixelFormat {
            match self {
                Texels::Bgra8 => PixelFormat::BGRA8Unorm,
                Texels::Rgba16Float => PixelFormat::RGBA16Float,
            }
        }

        /// A texel's bytes.
        fn texel(self) -> u32 {
            match self {
                Texels::Bgra8 => 4,
                Texels::Rgba16Float => 8,
            }
        }

        /// A reader's word's bytes.
        fn word(self) -> u32 {
            match self {
                Texels::Bgra8 => 4,
                Texels::Rgba16Float => 16,
            }
        }

        /// The complement of a channel's `k`.
        fn complement(self, k: u32) -> u32 {
            match self {
                Texels::Bgra8 => complement(8, k),
                Texels::Rgba16Float => 1023 - k,
            }
        }

        /// The pattern's channels at `(x, y)`, or their complement, in shader order.
        fn channels(self, x: u32, y: u32, complemented: bool) -> [u32; 4] {
            let k = match self {
                Texels::Bgra8 => unorm_pattern(8, 8, x, y),
                Texels::Rgba16Float => float_pattern(x, y),
            };
            k.map(|k| if complemented { self.complement(k) } else { k })
        }

        /// The bytes in memory of a texel of channels `k`: `BGRA8Unorm` blue, green, red, alpha;
        /// `RGBA16Float` each channel's half.
        fn bytes(self, [r, g, b, a]: [u32; 4]) -> Vec<u8> {
            match self {
                Texels::Bgra8 => [b, g, r, a].map(|c| c as u8).to_vec(),
                Texels::Rgba16Float => [r, g, b, a]
                    .into_iter()
                    .flat_map(|k| half(k).to_le_bytes())
                    .collect(),
            }
        }

        /// A reader's word of a texel of channels `k`.
        fn word_of(self, [r, g, b, a]: [u32; 4]) -> Vec<u8> {
            match self {
                Texels::Bgra8 => (r | (g << 8) | (b << 16) | (a << 24))
                    .to_le_bytes()
                    .to_vec(),
                Texels::Rgba16Float => [r, g, b, a]
                    .into_iter()
                    .flat_map(|k| single(k).to_le_bytes())
                    .collect(),
            }
        }

        /// The colour a shader writes or a clear makes for channels `k`: `(k + 0.25) / 255`, or
        /// `1 + k / 1024`.
        fn colour(self, k: u32) -> f64 {
            match self {
                Texels::Bgra8 => (f64::from(k) + 0.25) / 255.0,
                Texels::Rgba16Float => 1.0 + f64::from(k) / 1024.0,
            }
        }

        /// The constant `loop_fragment` takes less what it reads, so that it writes the read's
        /// complement: 1, or `2 + 1023 / 1024`.
        fn minuend(self) -> f32 {
            match self {
                Texels::Bgra8 => 1.0,
                Texels::Rgba16Float => 2.0 + 1023.0 / 1024.0,
            }
        }
    }

    /// A texture over a buffer: its texels, its size, its first texel's offset in the buffer, its
    /// rows' bytes there and its usage.
    struct Shape {
        what: &'static str,
        texels: Texels,
        size: [u32; 2],
        offset: u64,
        row: u64,
        usage: TextureUsage,
    }

    /// The bytes a buffer holds that are not a texel.
    const PAD: u8 = 0x5a;

    impl Shape {
        /// The buffer's bytes: from `offset`, rows `row` apart, each texel the pattern's or its
        /// complement's, every other byte [`PAD`].
        fn memory(&self, complemented: bool) -> Vec<u8> {
            let [width, height] = self.size;
            let mut bytes = vec![PAD; (self.offset + self.row * u64::from(height)) as usize];
            let texel = self.texels.texel() as usize;
            for y in 0..height {
                for x in 0..width {
                    let at = (self.offset + u64::from(y) * self.row) as usize + x as usize * texel;
                    let value = self.texels.bytes(self.texels.channels(x, y, complemented));
                    bytes[at..at + texel].copy_from_slice(&value);
                }
            }
            bytes
        }

        /// Every texel's `of(x, y)`, rows from the top.
        fn each(&self, of: impl Fn(u32, u32) -> Vec<u8>) -> Vec<u8> {
            let [width, height] = self.size;
            (0..height)
                .flat_map(|y| (0..width).map(move |x| (x, y)))
                .flat_map(|(x, y)| of(x, y))
                .collect()
        }

        /// Every texel's bytes in packed rows, as a blit out writes them.
        fn image(&self, complemented: bool) -> Vec<u8> {
            let t = self.texels;
            self.each(|x, y| t.bytes(t.channels(x, y, complemented)))
        }

        /// Every texel's reader word.
        fn words(&self, complemented: bool) -> Vec<u8> {
            let t = self.texels;
            self.each(|x, y| t.word_of(t.channels(x, y, complemented)))
        }
    }

    /// Read by shaders.
    const READ: TextureUsage = TextureUsage::SHADER_READ;

    /// The textures over a buffer at an icon's size, read by shaders: `BGRA8Unorm` at 48x48,
    /// 76x76, 108x108, 256x256 and 1024x1024 and `RGBA16Float` at 48x48 and 256x256 in tight rows
    /// from the buffer's start; `BGRA8Unorm` at 256x256 and 1024x1024 and `RGBA16Float` at 256x256
    /// in rows 64 bytes wider from 192 bytes in; and `BGRA8Unorm` at 1024x1024 in rows of 10,240
    /// bytes (2,560 texels), also a render target.
    const ICONS: [Shape; 11] = [
        icon("BGRA8Unorm 48x48 rows 192", Texels::Bgra8, 48, (0, 192)),
        icon("BGRA8Unorm 76x76 rows 304", Texels::Bgra8, 76, (0, 304)),
        icon("BGRA8Unorm 108x108 rows 432", Texels::Bgra8, 108, (0, 432)),
        icon(
            "BGRA8Unorm 256x256 rows 1024",
            Texels::Bgra8,
            256,
            (0, 1024),
        ),
        icon(
            "BGRA8Unorm 256x256 rows 1088 from 192",
            Texels::Bgra8,
            256,
            (192, 1088),
        ),
        icon(
            "RGBA16Float 48x48 rows 384",
            Texels::Rgba16Float,
            48,
            (0, 384),
        ),
        icon(
            "RGBA16Float 256x256 rows 2048",
            Texels::Rgba16Float,
            256,
            (0, 2048),
        ),
        icon(
            "RGBA16Float 256x256 rows 2112 from 192",
            Texels::Rgba16Float,
            256,
            (192, 2112),
        ),
        icon(
            "BGRA8Unorm 1024x1024 rows 4096",
            Texels::Bgra8,
            1024,
            (0, 4096),
        ),
        icon(
            "BGRA8Unorm 1024x1024 rows 4160 from 192",
            Texels::Bgra8,
            1024,
            (192, 4160),
        ),
        Shape {
            what: "BGRA8Unorm 1024x1024 rows 10240, a render target too",
            texels: Texels::Bgra8,
            size: [1024, 1024],
            offset: 0,
            row: 10_240,
            usage: READ.union(TextureUsage::RENDER_TARGET),
        },
    ];

    /// A square icon's texture over a buffer of `side` texels a side, from `offset` in rows of
    /// `row` bytes, read by shaders.
    const fn icon(
        what: &'static str,
        texels: Texels,
        side: u32,
        (offset, row): (u64, u64),
    ) -> Shape {
        Shape {
            what,
            texels,
            size: [side, side],
            offset,
            row,
            usage: READ,
        }
    }

    /// The textures over a buffer at a picture's size: `BGRA8Unorm` at 3840x2160 in rows of
    /// 15,360 bytes, and at 2845x1600 in rows of 11,392 bytes, three texels past the width.
    const PICTURES: [Shape; 2] = [
        Shape {
            what: "BGRA8Unorm 3840x2160 rows 15360",
            texels: Texels::Bgra8,
            size: [3840, 2160],
            offset: 0,
            row: 15_360,
            usage: READ,
        },
        Shape {
            what: "BGRA8Unorm 2845x1600 rows 11392",
            texels: Texels::Bgra8,
            size: [2845, 1600],
            offset: 0,
            row: 11_392,
            usage: READ,
        },
    ];

    /// The memory a buffer lies in.
    #[derive(Clone, Copy)]
    enum Memory {
        /// Host memory of the test's own, imported ([`Device::import_host_memory`]).
        Host,
        /// A Shared block of rt's own.
        Shared,
    }

    impl Memory {
        fn what(self) -> &'static str {
            match self {
                Memory::Host => "over host memory",
                Memory::Shared => "over a Shared block",
            }
        }
    }

    /// A buffer's memory, held until [`Under::release`].
    enum Under {
        Host {
            block: Block,
            mapping: (NonNull<u8>, Layout),
            released: Arc<AtomicBool>,
        },
        Shared(Block),
    }

    impl Under {
        /// The memory's first byte.
        fn pointer(&self) -> NonNull<u8> {
            match self {
                Under::Host { mapping, .. } => mapping.0,
                Under::Shared(block) => block.contents.expect("a Shared block"),
            }
        }

        /// Writes `bytes` at the memory's start.
        fn write(&self, bytes: &[u8]) {
            // SAFETY: the memory holds at least `bytes`' length, and no command reaches it now.
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.pointer().as_ptr(), bytes.len())
            };
        }

        /// The memory's first `len` bytes.
        fn read(&self, len: usize) -> Vec<u8> {
            // SAFETY: the memory holds at least `len` bytes, and no command reaches it now.
            unsafe { std::slice::from_raw_parts(self.pointer().as_ptr(), len) }.to_vec()
        }

        /// Lets the memory go, once nothing over it lives: an import's release run and its
        /// host memory freed.
        fn release(self, device: &Device) {
            match self {
                Under::Host {
                    block,
                    mapping,
                    released,
                } => {
                    drop(block);
                    wait_until(device, "the release", || released.load(Ordering::SeqCst));
                    free(mapping);
                }
                Under::Shared(block) => (device.release_block(block)).expect("a release"),
            }
        }
    }

    /// A buffer of `len` bytes from the start of `memory`'s, and the memory; None, printed, where
    /// the device imports no host memory.
    fn under(name: &str, device: &Device, memory: Memory, len: u64) -> Option<(Buffer, Under)> {
        let (block, under) = match memory {
            Memory::Host => {
                let Some((_, alignment)) = device.core().imports.host.as_ref() else {
                    println!("real-size: {name}: imports no host memory; not run over it");
                    return None;
                };
                let whole = len.next_multiple_of(*alignment);
                let (pointer, mapping) = host_memory(whole, *alignment);
                let (release, released) = releasing();
                // SAFETY: the memory is the test's, mapped until it unmaps it after the release.
                let (block, _) = unsafe { device.import_host_memory(pointer, whole, release) }
                    .unwrap_or_else(|error| panic!("{name}: an import of {whole} bytes: {error}"));
                let id = block.id;
                let under = Under::Host {
                    block,
                    mapping,
                    released,
                };
                (id, under)
            }
            Memory::Shared => {
                let block = (device.take_block(MemoryClass::Shared, len))
                    .unwrap_or_else(|error| panic!("{name}: a Shared block of {len}: {error}"));
                (block.id, Under::Shared(block))
            }
        };
        let buffer = (device.new_buffer(block, 0, len, None))
            .unwrap_or_else(|error| panic!("{name}: a buffer of {len}: {error}"));
        Some((buffer, under))
    }

    /// Textures over a buffer in host memory's import and in a Shared block of rt's own, at each
    /// of `shapes`: the CPU writes the complement and a sampled read takes it; then the CPU writes
    /// the pattern, and sampling, `read()`, a blit out and sampling again each read every texel
    /// exact, and the buffer's bytes are the CPU's after. Each texture's form, in place or copied
    /// for its cause, and the over-copies counted, are printed.
    fn overs(test: &str, shapes: &[Shape]) {
        let Some(libraries) = libraries() else {
            return;
        };
        each_device(test, |name, _, device| {
            let reading = Reading::new(name, &device, &libraries);
            let first = IndexClass::Texture.range().start;
            let memories = [Memory::Host, Memory::Shared];
            let count = (shapes.len() * memories.len()) as u32;
            (device.grant_indices(IndexClass::Texture, first, count))
                .unwrap_or_else(|error| panic!("{name}: a texture grant: {error}"));
            let queue = device.new_command_queue(4).expect("a queue");
            let on = (&device, &queue);
            let mut cells = Cells::new(name);
            let pairs = shapes.iter().flat_map(|shape| memories.map(|m| (shape, m)));
            for (index, (shape, memory)) in (first..).zip(pairs) {
                let len = shape.offset + shape.row * u64::from(shape.size[1]);
                let Some((buffer, under)) = under(name, &device, memory, len) else {
                    continue;
                };
                let what = format!("{} {}", shape.what, memory.what());
                let t = shape.texels;
                let mut d = TextureDescriptor::new(
                    TextureType::Type2D,
                    t.format(),
                    shape.size[0],
                    shape.size[1],
                );
                d.usage = shape.usage;
                let texture = (buffer.new_texture(&d, Some(index), shape.offset, shape.row))
                    .unwrap_or_else(|error| panic!("{name}: {what}: {error}"));
                let form = held_as(&device, &texture);
                let copies = || device.counters().metrics().over_copies;
                let before = copies();
                let at = (&texture, shape.size, t);
                let (word, width) = (t.word() as usize, shape.size[0]);
                let texel = t.texel() as usize;
                under.write(&shape.memory(true));
                let stale = shape.words(true);
                let got = reading.read(name, on, true, at, &stale);
                let cell = |how: &str| format!("{what}: {how}");
                cells.check(&cell("the complement, sampled"), got, &stale, (word, width));
                let memory_bytes = shape.memory(false);
                under.write(&memory_bytes);
                let words = shape.words(false);
                let image = shape.image(false);
                let got = reading.read(name, on, true, at, &words);
                cells.check(&cell("sampled first"), got, &words, (word, width));
                let got = reading.read(name, on, false, at, &words);
                cells.check(&cell("read()"), got, &words, (word, width));
                let got = copy_out(name, on, (&texture, shape.size), t.texel(), &image);
                cells.check(&cell("copied out"), got, &image, (texel, width));
                let got = reading.read(name, on, true, at, &words);
                cells.check(&cell("sampled last"), got, &words, (word, width));
                let kept = under.read(memory_bytes.len());
                let pitch = (shape.row / texel as u64) as u32;
                cells.check(
                    &cell("the buffer's bytes after"),
                    Ok(kept),
                    &memory_bytes,
                    (texel, pitch),
                );
                println!(
                    "real-size: {name}: {what}: {form}, {} over-copies counted",
                    copies() - before
                );
                drop((texture, buffer));
                under.release(&device);
            }
            cells.all_exact();
        });
    }

    /// How `texture`, over a buffer, holds its texels: copied for its cause, or in place, with
    /// the row pitch and offset its linear image's subresource layout reports.
    fn held_as(device: &Device, texture: &Texture) -> String {
        if let Some((_, copy)) = texture.copied_over() {
            return format!("copied for {:?}", copy.kind.cause);
        }
        let color = vk::ImageSubresource {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            array_layer: 0,
        };
        // SAFETY: the texture's image, linear, of one level and layer of a colour format.
        let layout =
            unsafe { (device.core().device).get_image_subresource_layout(texture.image(), color) };
        format!(
            "in place, row pitch {} offset {}",
            layout.row_pitch, layout.offset
        )
    }

    /// The icons' textures over a Shared block that the CPU refills each frame, read by shaders:
    /// `BGRA8Unorm` at 48x48, 76x76, 108x108 and 256x256 and `RGBA16Float` at 48x48 and 384x384,
    /// in tight rows from the buffer's start.
    const FRAMED: [Shape; 6] = [
        icon("BGRA8Unorm 48x48 rows 192", Texels::Bgra8, 48, (0, 192)),
        icon("BGRA8Unorm 76x76 rows 304", Texels::Bgra8, 76, (0, 304)),
        icon("BGRA8Unorm 108x108 rows 432", Texels::Bgra8, 108, (0, 432)),
        icon(
            "BGRA8Unorm 256x256 rows 1024",
            Texels::Bgra8,
            256,
            (0, 1024),
        ),
        icon(
            "RGBA16Float 48x48 rows 384",
            Texels::Rgba16Float,
            48,
            (0, 384),
        ),
        icon(
            "RGBA16Float 384x384 rows 3072",
            Texels::Rgba16Float,
            384,
            (0, 3072),
        ),
    ];

    /// What the CPU writes into a [`FRAMED`] texture's buffer before each frame: the pattern, its
    /// complement, the pattern.
    const FRAMES: [bool; 3] = [false, true, false];

    /// Textures over a Shared block of rt's own ([`FRAMED`]), each sampled once a frame after the
    /// CPU rewrote its buffer ([`FRAMES`]), every texel compared. Each cell names how the texture
    /// holds its texels: copied for its cause, or in place with its linear image's row pitch and
    /// offset beside the buffer's rows.
    #[test]
    fn every_device_samples_icons_over_a_shared_block_refilled_each_frame_every_texel() {
        let Some(libraries) = libraries() else {
            return;
        };
        each_device("real_size_icon_frames", |name, _, device| {
            let reading = Reading::new(name, &device, &libraries);
            let first = IndexClass::Texture.range().start;
            (device.grant_indices(IndexClass::Texture, first, FRAMED.len() as u32))
                .unwrap_or_else(|error| panic!("{name}: a texture grant: {error}"));
            let queue = device.new_command_queue(4).expect("a queue");
            let on = (&device, &queue);
            let mut cells = Cells::new(name);
            for (index, shape) in (first..).zip(FRAMED.iter()) {
                let len = shape.offset + shape.row * u64::from(shape.size[1]);
                let Some((buffer, under)) = under(name, &device, Memory::Shared, len) else {
                    continue;
                };
                let t = shape.texels;
                let [width, height] = shape.size;
                let mut d = TextureDescriptor::new(TextureType::Type2D, t.format(), width, height);
                d.usage = shape.usage;
                let texture = (buffer.new_texture(&d, Some(index), shape.offset, shape.row))
                    .unwrap_or_else(|error| panic!("{name}: {}: {error}", shape.what));
                let what = format!(
                    "{} over a Shared block, {}",
                    shape.what,
                    held_as(&device, &texture)
                );
                for (frame, &complemented) in FRAMES.iter().enumerate() {
                    under.write(&shape.memory(complemented));
                    let want = shape.words(complemented);
                    let got = reading.read(name, on, true, (&texture, shape.size, t), &want);
                    let holds = match complemented {
                        true => "the complement",
                        false => "the pattern",
                    };
                    let cell = format!("{what}: frame {frame}, {holds}, sampled");
                    cells.check(&cell, got, &want, (t.word() as usize, width));
                }
                drop((texture, buffer));
                under.release(&device);
            }
            cells.all_exact();
        });
    }

    /// Textures over a buffer at an icon's size ([`ICONS`]), as [`overs`] reads them.
    #[test]
    fn every_device_reads_icon_size_textures_over_a_buffer_every_texel() {
        overs("real_size_icon_overs", &ICONS);
    }

    /// Textures over a buffer at a picture's size ([`PICTURES`]), as [`overs`] reads them.
    #[test]
    fn every_device_reads_picture_size_textures_over_a_buffer_every_texel() {
        overs("real_size_picture_overs", &PICTURES);
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

    /// The render state of `cover` and `fragment` into one colour attachment of `format`.
    fn drawing(
        name: &str,
        device: &Device,
        library: &Library,
        fragment: &str,
        format: PixelFormat,
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

    /// The 8-bit pattern's or the float pattern's fragment, as `texels` holds.
    fn pattern_fragment(texels: Texels) -> &'static str {
        match texels {
            Texels::Bgra8 => "unorm_fragment",
            Texels::Rgba16Float => "float_fragment",
        }
    }

    /// The channels a pass that samples its own attachment clears it to.
    const CLEAR: [u32; 4] = [32, 96, 160, 224];

    /// What a pass that samples its own attachment draws with: the pattern's state, the state of
    /// `loop_fragment` and what it takes, and the clear.
    struct Looping<'a> {
        pattern: RenderPipelineState,
        looping: RenderPipelineState,
        sampler: &'a SamplerState,
        constants: Vec<u8>,
        clear: ClearColor,
    }

    impl Looping<'_> {
        /// One pass over `texture`: cleared, or loaded and drawn the pattern over, then, after a
        /// memory barrier, `loop_fragment` over the whole of it, sampling `texture` itself.
        fn pass(
            &self,
            name: &str,
            queue: &CommandQueue,
            texture: &Texture,
            cleared: bool,
        ) -> Result<(), String> {
            let mut pass = RenderPassDescriptor::default();
            let c = &mut pass.color_attachments[0];
            c.texture = Some(texture.clone());
            c.load_action = match cleared {
                true => LoadAction::Clear,
                false => LoadAction::Load,
            };
            c.clear_color = self.clear;
            let command_buffer = queue.command_buffer().map_err(said)?;
            let mut encoder = (command_buffer.render_command_encoder(&pass)).map_err(said)?;
            if !cleared {
                encoder
                    .set_render_pipeline_state(&self.pattern)
                    .map_err(said)?;
                (encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)).map_err(said)?;
                encoder.memory_barrier().map_err(said)?;
            }
            encoder
                .set_render_pipeline_state(&self.looping)
                .map_err(said)?;
            (encoder.set_fragment_texture(Some(texture), 0)).map_err(said)?;
            (encoder.set_fragment_sampler_state(Some(self.sampler), 0)).map_err(said)?;
            (encoder.set_fragment_bytes(&self.constants, 0)).map_err(said)?;
            (encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)).map_err(said)?;
            encoder.end_encoding();
            completed(name, &command_buffer)
        }
    }

    /// The render targets sampled in their own pass: `BGRA8Unorm` and `RGBA16Float` at 256x256
    /// and 1024x1024.
    const LOOPS: [(Texels, u32); 4] = [
        (Texels::Bgra8, 256),
        (Texels::Rgba16Float, 256),
        (Texels::Bgra8, 1024),
        (Texels::Rgba16Float, 1024),
    ];

    /// A render target holding the pattern is drawn by a pass that clears it to [`CLEAR`] and
    /// samples it, as its own attachment, at each pixel's centre, writing the read's complement:
    /// every texel then holds the clear's complement. Then a pass that loads it draws the pattern
    /// over it and, after a memory barrier, samples it the same way: every texel then holds the
    /// pattern's complement. After each pass the target is copied out and sampled by a kernel,
    /// every texel compared. A device without attachment reads, which refuses such a draw typed,
    /// says so and runs none.
    #[test]
    fn every_device_samples_its_attachment_in_its_own_pass_at_icon_size_every_texel() {
        let Some(libraries) = libraries() else {
            return;
        };
        each_device("real_size_attachment_loops", |name, _, device| {
            let (reads, looped) = {
                let core = device.core();
                (core.attachment_reads, core.feedback_loop)
            };
            if !reads {
                println!("real-size: {name}: no attachment reads; the loop cells not run");
                return;
            }
            let reading = Reading::new(name, &device, &libraries);
            let own = &libraries.1;
            let first = IndexClass::Texture.range().start;
            (device.grant_indices(IndexClass::Texture, first, LOOPS.len() as u32))
                .unwrap_or_else(|error| panic!("{name}: a texture grant: {error}"));
            let queue = device.new_command_queue(4).expect("a queue");
            let on = (&device, &queue);
            let mut cells = Cells::new(name);
            let form = match looped {
                true => "through the feedback loop",
                false => "through copies into a shadow",
            };
            for (index, &(texels, side)) in (first..).zip(LOOPS.iter()) {
                let format = texels.format();
                let shape = Shape {
                    what: "",
                    texels,
                    size: [side, side],
                    offset: 0,
                    row: u64::from(side * texels.texel()),
                    usage: TextureUsage::RENDER_TARGET | READ,
                };
                let what = format!("{format:?} {side}x{side} sampled in its own pass");
                let mut d = TextureDescriptor::new(TextureType::Type2D, format, side, side);
                d.usage = shape.usage;
                d.storage_mode = StorageMode::Private;
                let texture = (device.new_texture(&d, Some(index)))
                    .unwrap_or_else(|error| panic!("{name}: {what}: {error}"));
                let at = (&texture, shape.size);
                if let Err(why) = copy_in(name, on, at, texels.texel(), &shape.image(false)) {
                    cells.check(&format!("{what}: the blit in"), Err(why), &[], (1, 1));
                    continue;
                }
                let inverse = 1.0 / side as f32;
                let minuend = texels.minuend();
                let clear = CLEAR.map(|k| texels.colour(k));
                let looping = Looping {
                    pattern: drawing(name, &device, own, pattern_fragment(texels), format),
                    looping: drawing(name, &device, own, "loop_fragment", format),
                    sampler: &reading.readers.sampler,
                    constants: [
                        minuend, minuend, minuend, minuend, inverse, inverse, 0.0, 0.0,
                    ]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect(),
                    clear: ClearColor::new(clear[0], clear[1], clear[2], clear[3]),
                };
                for after in ["a clear", "a draw of the pattern"] {
                    let cleared = after == "a clear";
                    let counters = device.counters();
                    let before = counters.metrics();
                    let drawn = looping.pass(name, &queue, &texture, cleared);
                    let moved = counters.metrics();
                    let facts = format!(
                        "{form}, attachment_loop_reads {}, attachment_read_copies {}",
                        moved.attachment_loop_reads - before.attachment_loop_reads,
                        moved.attachment_read_copies - before.attachment_read_copies
                    );
                    let cell = |how: &str| format!("{what} after {after} ({facts}): {how}");
                    if let Err(why) = drawn {
                        cells.check(&cell("the pass"), Err(why), &[], (1, 1));
                        continue;
                    }
                    let (image, words) = match cleared {
                        true => {
                            let k = CLEAR.map(|k| texels.complement(k));
                            let texel = texels.bytes(k);
                            let word = texels.word_of(k);
                            (
                                shape.each(|_, _| texel.clone()),
                                shape.each(|_, _| word.clone()),
                            )
                        }
                        false => (shape.image(true), shape.words(true)),
                    };
                    let texel = texels.texel();
                    let got = copy_out(name, on, at, texel, &image);
                    cells.check(&cell("copied out"), got, &image, (texel as usize, side));
                    let got = reading.read(name, on, true, (&texture, shape.size, texels), &words);
                    let word = texels.word() as usize;
                    cells.check(&cell("sampled"), got, &words, (word, side));
                }
            }
            cells.all_exact();
        });
    }

    /// A 3840x2160 float picture's size.
    const PICTURE: [u32; 2] = [3840, 2160];

    /// The bytes a host-memory download of the picture lies in: 64,804 KiB, 4 KiB past its texels.
    const DOWNLOAD: u64 = 64_804 << 10;

    /// The bytes of the download's memory past the picture's texels.
    const TAIL: u8 = 0xa5;

    /// The picture's texels in `texture`, copied out by a blit into `buffer` from its start in
    /// packed rows, and the first `len` bytes of `under`, its memory, after.
    fn download(
        name: &str,
        queue: &CommandQueue,
        texture: &Texture,
        (buffer, under): (&Buffer, &Under),
        len: usize,
    ) -> Result<Vec<u8>, String> {
        let command_buffer = queue.command_buffer().map_err(said)?;
        let mut encoder = command_buffer.blit_command_encoder().map_err(said)?;
        let layout = BufferLayout {
            offset: 0,
            bytes_per_row: u64::from(PICTURE[0]) * 8,
            bytes_per_image: 0,
        };
        let whole = [PICTURE[0], PICTURE[1], 1];
        let place = TexturePlace::default();
        (encoder.copy_from_texture_to_buffer(texture, place, whole, buffer, layout))
            .map_err(said)?;
        encoder.end_encoding();
        completed(name, &command_buffer)?;
        Ok(under.read(len))
    }

    /// Draws `state`'s pattern over the whole of `texture`, loaded, in one render pass.
    fn render(
        name: &str,
        queue: &CommandQueue,
        state: &RenderPipelineState,
        texture: &Texture,
    ) -> Result<(), String> {
        let mut pass = RenderPassDescriptor::default();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(texture.clone());
        color.load_action = LoadAction::Load;
        let command_buffer = queue.command_buffer().map_err(said)?;
        let mut encoder = (command_buffer.render_command_encoder(&pass)).map_err(said)?;
        encoder.set_render_pipeline_state(state).map_err(said)?;
        (encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)).map_err(said)?;
        encoder.end_encoding();
        completed(name, &command_buffer)
    }

    /// Writes the float pattern into every texel of `texture` by `state`'s kernel, one thread a
    /// texel in threadgroups of 16x16.
    fn write(
        name: &str,
        queue: &CommandQueue,
        state: &ComputePipelineState,
        texture: &Texture,
    ) -> Result<(), String> {
        let command_buffer = queue.command_buffer().map_err(said)?;
        let mut encoder =
            (command_buffer.compute_command_encoder(DispatchType::Serial)).map_err(said)?;
        encoder.set_compute_pipeline_state(state).map_err(said)?;
        encoder.set_texture(Some(texture), 0).map_err(said)?;
        let size: Vec<u8> = [PICTURE[0], PICTURE[1], 0, 0]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        encoder.set_bytes(&size, 0).map_err(said)?;
        (encoder.dispatch_threads([PICTURE[0], PICTURE[1], 1], [16, 16, 1])).map_err(said)?;
        encoder.end_encoding();
        completed(name, &command_buffer)
    }

    /// An `RGBA16Float` texture at 3840x2160, written by a blit of the complement and then by a
    /// render pass that loads it and draws the float pattern, and again by a kernel: each time
    /// copied out whole into a Shared buffer and into a buffer over host memory's import of
    /// 64,804 KiB (where the device imports host memory), every texel exact and the import's
    /// bytes past them as they were.
    #[test]
    fn every_device_copies_out_a_real_size_float_picture_every_texel() {
        let Some(library) = library() else {
            return;
        };
        let shape = Shape {
            what: "RGBA16Float 3840x2160",
            texels: Texels::Rgba16Float,
            size: PICTURE,
            offset: 0,
            row: u64::from(PICTURE[0]) * 8,
            usage: TextureUsage::RENDER_TARGET | TextureUsage::SHADER_WRITE | READ,
        };
        let stale = shape.image(true);
        let image = shape.image(false);
        let mut downloaded = image.clone();
        downloaded.resize(DOWNLOAD as usize, TAIL);
        let mut filled = stale.clone();
        filled.resize(DOWNLOAD as usize, TAIL);
        each_device("real_size_float_picture", |name, _, device| {
            let first = IndexClass::Texture.range().start;
            (device.grant_indices(IndexClass::Texture, first, 1))
                .unwrap_or_else(|error| panic!("{name}: a texture grant: {error}"));
            let format = PixelFormat::RGBA16Float;
            let mut d = TextureDescriptor::new(TextureType::Type2D, format, PICTURE[0], PICTURE[1]);
            d.usage = shape.usage;
            d.storage_mode = StorageMode::Private;
            let texture = (device.new_texture(&d, Some(first)))
                .unwrap_or_else(|error| panic!("{name}: {}: {error}", shape.what));
            let drawn = drawing(name, &device, &library, "float_fragment", format);
            let written = kernel(name, &device, &library, "pattern_write");
            let queue = device.new_command_queue(4).expect("a queue");
            let on = (&device, &queue);
            let download_into = under(name, &device, Memory::Host, DOWNLOAD);
            let mut cells = Cells::new(name);
            for by_pass in [true, false] {
                let how = match by_pass {
                    true => "drawn by a render pass",
                    false => "written by a kernel",
                };
                let what = format!("{} {how}", shape.what);
                let at = (&texture, PICTURE);
                let wrote = copy_in(name, on, at, 8, &stale).and_then(|()| match by_pass {
                    true => render(name, &queue, &drawn, &texture),
                    false => write(name, &queue, &written, &texture),
                });
                if let Err(why) = wrote {
                    cells.check(&what, Err(why), &[], (1, 1));
                    continue;
                }
                let got = copy_out(name, on, at, 8, &image);
                cells.check(&format!("{what}: copied out"), got, &image, (8, PICTURE[0]));
                if let Some((buffer, memory)) = &download_into {
                    memory.write(&filled);
                    let got = download(name, &queue, &texture, (buffer, memory), filled.len());
                    cells.check(
                        &format!("{what}: copied out into host memory"),
                        got,
                        &downloaded,
                        (8, PICTURE[0]),
                    );
                }
            }
            if let Some((buffer, memory)) = download_into {
                drop(buffer);
                memory.release(&device);
            }
            cells.all_exact();
        });
    }
}
