//! Many textures over one `Private` surface's bytes, on the device the cfg selects: a guest's
//! window server, at its login, makes dozens of textures over each of its large surfaces
//! (`new_texture_with_iosurface_plane` over the `Private` buffer that stands for the surface),
//! `BGRA8Unorm` and `R32Uint` views of the same bytes with shader read and write usage, and writes
//! and reads a few of them each frame. Each surface's maker declares the bytes between its rows
//! unobserved (`BufferDescriptor::padding_unobserved`), as the guest's do. Each texture over a
//! `Private` buffer whose rows the device's
//! own linear image lays out otherwise is a `DRM_FORMAT_MOD_LINEAR` image at Metal's pitch inside
//! the pooled bound (`in_place::POOLED_PITCHED_BYTES`), else a shadow
//! (`TextureShadowReason::LinearCost`); a shadow is an image of its own, copied in and back apart
//! from every other texture's over the same bytes.
//!
//! - [`textures_over_one_surface_s_bytes_take_one_shadow`]: four textures (two of each format) over
//!   one `Private` surface past the bound, where the device makes such images: the first is a
//!   shadow of its linear cost, and every later one over the same bytes takes the image at Metal's
//!   pitch (`WorkCounts::pooled_pitched_images`), so the textures hold one shadow's bytes, not four;
//!   the first's shadow moves to its image at the next quiet commit that takes storage forms (a
//!   blit's, here, as `in_place_textures`' move checks commit;
//!   `WorkCounts::pooled_textures_moved_in_place`), and a read through it then copies nothing.
//! - [`textures_made_one_after_another_over_one_surface_s_bytes_take_one_shadow`]: the same four
//!   made one after another, each written and released before the next is made, as a guest's
//!   frames make them: the first is a shadow, every later one takes the image.
//! - [`writes_through_one_texture_reach_every_texture_over_the_bytes`]: a kernel's write through
//!   each of those textures in turn is what a kernel reads through every other one, in the same
//!   command buffer and in the next, and what the surface's bytes hold: Metal's aliasing of
//!   textures over the same bytes.
//! - [`aliased_surface_costs`] (ignored, release builds): the guest's surface, 6016x6016 at rows of
//!   24,576 bytes in a buffer of 150,994,944, with 36 textures over it (25 `BGRA8Unorm`, 11
//!   `R32Uint`) alive across the frames, or four made each frame and dropped after it, a kernel
//!   writing a square of two of them and another reading a square of others each frame
//!   (`ALIASED-SURFACE-COST`): the textures' forms, the shadows made and their bytes, a frame's
//!   GPU time and its shadow copies and texels copied in and back, and the device memory the
//!   process holds at its peak (Linux: the DRM file descriptors' resident bytes, and the process's
//!   resident set, where lavapipe's memory lies).

use super::first_use_writes::CHECK_WAIT;
use super::in_place_textures::in_place;
use super::{validated_device_with, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    BlitOption, Buffer, BufferContents, BufferDescriptor, CommandBuffer, ComputePipelineState,
    Device, IoSurfacePlane, Origin, PipelineOptions, ResourceOptions, Size, Texture,
    TextureDescriptor, TexturePlacement, TextureShadowReason, WorkCounts,
};

/// The kernels: `fill_float` and `fill_uint` write constant buffer 0's value at every thread's
/// texel of texture 0 (`texture2d<float, write>`, `texture2d<uint, write>`); `read_float` and
/// `read_uint` read texture 0's texel at every thread's place into buffer 0, row by row over the
/// grid's width (`texture2d<float, read>`, `texture2d<uint, read>`).
const KERNELS: &str = r#"; Owned synthetic kernels: texture writes and reads over one surface's bytes.
source_filename = "aliased_surfaces.metal"

define void @fill_float(ptr addrspace(1) %t, ptr addrspace(2) noalias %c, <2 x i32> %gid) {
entry:
  %v = load <4 x float>, ptr addrspace(2) %c, align 16
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %t, <2 x i32> %gid, <4 x float> %v, i32 0, i32 2)
  ret void
}

define void @fill_uint(ptr addrspace(1) %t, ptr addrspace(2) noalias %c, <2 x i32> %gid) {
entry:
  %v = load <4 x i32>, ptr addrspace(2) %c, align 16
  tail call void @air.write_texture_2d.u.v4i32(ptr addrspace(1) %t, <2 x i32> %gid, <4 x i32> %v, i32 0, i32 2)
  ret void
}

define void @read_float(ptr addrspace(1) %t, ptr addrspace(1) noalias %out, <2 x i32> %gid, <2 x i32> %grid) {
entry:
  %s = tail call ptr addrspace(2) @air.get_read_sampler()
  %r = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %gid, <2 x i32> zeroinitializer, i32 0, i32 1)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %w = extractelement <2 x i32> %grid, i64 0
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %at = zext i32 %i to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %at
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}

define void @read_uint(ptr addrspace(1) %t, ptr addrspace(1) noalias %out, <2 x i32> %gid, <2 x i32> %grid) {
entry:
  %s = tail call ptr addrspace(2) @air.get_read_sampler()
  %r = tail call { <4 x i32>, i8 } @air.read_texture_2d.u.v4i32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %gid, <2 x i32> zeroinitializer, i32 0, i32 1)
  %v = extractvalue { <4 x i32>, i8 } %r, 0
  %word = extractelement <4 x i32> %v, i64 0
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %w = extractelement <2 x i32> %grid, i64 0
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %at = zext i32 %i to i64
  %p = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %at
  store i32 %word, ptr addrspace(1) %p, align 4
  ret void
}

declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)
declare void @air.write_texture_2d.u.v4i32(ptr addrspace(1), <2 x i32>, <4 x i32>, i32, i32)
declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
declare { <4 x i32>, i8 } @air.read_texture_2d.u.v4i32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)

!air.kernel = !{!0, !6, !10, !16}
!0 = !{ptr @fill_float, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"c"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
!6 = !{ptr @fill_uint, !1, !7}
!7 = !{!8, !9, !5}
!8 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<uint, write>", !"air.arg_name", !"t"}
!9 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"uint4", !"air.arg_name", !"c"}
!10 = !{ptr @read_float, !1, !11}
!11 = !{!12, !13, !5, !14}
!12 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"t"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!14 = !{i32 3, !"air.threads_per_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"grid"}
!16 = !{ptr @read_uint, !1, !17}
!17 = !{!18, !19, !5, !14}
!18 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<uint, read>", !"air.arg_name", !"t"}
!19 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;

/// The kernels' pipeline states on a device.
struct Kernels {
    fill_float: ComputePipelineState,
    fill_uint: ComputePipelineState,
    read_float: ComputePipelineState,
    read_uint: ComputePipelineState,
}

impl Kernels {
    fn new(device: &Device) -> Kernels {
        let library = device
            .new_library_with_air(&llvm_assemble_text(KERNELS).unwrap())
            .unwrap();
        let state = |name: &str| {
            let function = library.new_function_with_name(name).unwrap();
            let source = StageSource::Function(function.function_descriptor());
            device
                .new_compute_pipeline_state(
                    &ComputePipelineDescriptor::new(source),
                    &PipelineOptions::default(),
                )
                .unwrap()
        };
        Kernels {
            fill_float: state("fill_float"),
            fill_uint: state("fill_uint"),
            read_float: state("read_float"),
            read_uint: state("read_uint"),
        }
    }

    /// Record into `command_buffer` a kernel that writes `word` (a `BGRA8Unorm` texture's texel of
    /// those bytes, or an `R32Uint` texture's of that value) at each texel of `texture` inside the
    /// square of `side` texels at its origin.
    fn fill(&self, command_buffer: &CommandBuffer, texture: &Texture, word: u32, side: u64) {
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        let bytes: Vec<u8> = match texture.pixel_format() {
            PixelFormat::R32Uint => [word, 0, 0, 0]
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect(),
            _ => {
                // Bytes b, g, r, a in memory: the texel's red is byte 2, blue byte 0.
                let [b, g, r, a] = word.to_le_bytes();
                [r, g, b, a]
                    .iter()
                    .flat_map(|&c| (f32::from(c) / 255.0).to_le_bytes())
                    .collect()
            }
        };
        let state = match texture.pixel_format() {
            PixelFormat::R32Uint => &self.fill_uint,
            _ => &self.fill_float,
        };
        encoder.set_compute_pipeline_state(state).unwrap();
        encoder.set_texture(Some(texture), 0).unwrap();
        encoder.set_bytes(&bytes, 0).unwrap();
        encoder
            .dispatch_threads(Size::new(side, side, 1), Size::new(16, 16, 1))
            .unwrap();
        encoder.end_encoding().unwrap();
    }

    /// Record into `command_buffer` a kernel that reads each texel of `texture` inside the square
    /// of `side` texels at its origin into `out`, row by row: a `float4` of each `BGRA8Unorm`
    /// texel, a word of each `R32Uint` one.
    fn read(&self, command_buffer: &CommandBuffer, texture: &Texture, out: &Buffer, side: u64) {
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        let state = match texture.pixel_format() {
            PixelFormat::R32Uint => &self.read_uint,
            _ => &self.read_float,
        };
        encoder.set_compute_pipeline_state(state).unwrap();
        encoder.set_texture(Some(texture), 0).unwrap();
        encoder.set_buffer(Some(out), 0, 0).unwrap();
        encoder
            .dispatch_threads(Size::new(side, side, 1), Size::new(16, 16, 1))
            .unwrap();
        encoder.end_encoding().unwrap();
    }
}

/// A texture of `format`, `width` by `height`, of shader read and write usage, over `surface` at
/// rows `pitch` apart, made as the guest's window server makes it
/// (`new_texture_with_iosurface_plane`, a surface without planes); and where the query answered it
/// would lie before it was made.
fn surface_texture(
    surface: &Buffer,
    format: PixelFormat,
    (width, height, pitch): (u64, u64, u64),
) -> (Texture, TexturePlacement) {
    let mut d = TextureDescriptor::texture_2d(format, width, height, false);
    d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    d.resource_options = surface.resource_options();
    let plane = IoSurfacePlane::new(0, pitch, false);
    let placement = surface
        .texture_placement_with_iosurface_plane(&d, plane)
        .unwrap();
    let texture = surface.new_texture_with_iosurface_plane(&d, plane).unwrap();
    (texture, placement)
}

/// The words a kernel reading `texture` over the square of `side` texels left in `out`, a
/// `Shared` buffer: each `R32Uint` texel's value, or each `BGRA8Unorm` texel's bytes as a word
/// (blue in its lowest byte, as the surface holds them).
fn words_read(texture: &Texture, out: &Buffer, side: u64) -> Vec<u32> {
    let texels = (side * side) as usize;
    let contents = out.contents().unwrap().as_ptr();
    match texture.pixel_format() {
        PixelFormat::R32Uint => {
            // SAFETY: a `Shared` buffer of at least `texels` words, which no command uses now.
            let words = unsafe { std::slice::from_raw_parts(contents.cast::<u32>(), texels) };
            words.to_vec()
        }
        _ => {
            // SAFETY: a `Shared` buffer of at least `texels` float4s, which no command uses now.
            let floats = unsafe { std::slice::from_raw_parts(contents.cast::<f32>(), texels * 4) };
            floats
                .chunks(4)
                .map(|c| {
                    let byte = |x: f32| (x * 255.0).round() as u8;
                    u32::from_le_bytes([byte(c[2]), byte(c[1]), byte(c[0]), byte(c[3])])
                })
                .collect()
        }
    }
}

/// The shape of the checks' surface, (width, rows, row bytes): 256 texels of 4 bytes a row (1024
/// bytes, a multiple of 256 at which RADV lays linear rows, and the tight rows ANV lays) at a
/// pitch of 1280 bytes, 64 rows (81,920 bytes, past the pooled bound).
const SMALL: (u64, u64, u64) = (256, 64, 1280);

/// A surface of `length` bytes of `options`, zeros, made as the guest's: its maker declares the
/// bytes between a texture's rows unobserved (`BufferDescriptor::padding_unobserved`), without
/// which no image over the buffer's memory holds rows that lie apart.
fn made_surface(device: &Device, length: u64, options: ResourceOptions) -> Buffer {
    let descriptor =
        BufferDescriptor::new(length, options, BufferContents::Zero).with_padding_unobserved();
    device.new_buffer_with_descriptor(descriptor).unwrap()
}

/// A `Private` surface of `SMALL`'s rows.
fn small_surface(device: &Device) -> Buffer {
    let (_, height, pitch) = SMALL;
    made_surface(device, pitch * height, ResourceOptions::private())
}

/// Whether the device takes a `DRM_FORMAT_MOD_LINEAR` image at Metal's pitch for a texture of
/// `SMALL`'s shape over its pooled memory where the bound does not hold it back: over a `Shared`
/// buffer, whose textures take it at any size where the device's check passed.
fn takes_pitched(device: &Device) -> bool {
    let (_, height, pitch) = SMALL;
    let shared = made_surface(device, pitch * height, ResourceOptions::shared());
    let before = device.work_counts().pooled_pitched_images;
    let (_texture, _) = surface_texture(&shared, PixelFormat::BGRA8Unorm, SMALL);
    device.work_counts().pooled_pitched_images > before
}

/// The checks' four textures over one surface's bytes, in the order made.
const FORMATS: [PixelFormat; 4] = [
    PixelFormat::BGRA8Unorm,
    PixelFormat::R32Uint,
    PixelFormat::BGRA8Unorm,
    PixelFormat::R32Uint,
];

#[test]
fn textures_over_one_surface_s_bytes_take_one_shadow() {
    super::host_memory::on_device_with(in_place, |device| {
        assert!(
            device.settle_in_place_checks(CHECK_WAIT),
            "the check settles"
        );
        if !takes_pitched(device) {
            eprintln!(
                "ALIASED-SURFACE {}: no DRM_FORMAT_MOD_LINEAR image at a {}-byte pitch: nothing \
                 to check",
                device.name(),
                SMALL.2
            );
            return;
        }
        let surface = small_surface(device);
        let (width, height, pitch) = SMALL;
        let bytes = pitch * height;
        let linear_cost = TexturePlacement::Shadow(TextureShadowReason::LinearCost { bytes });
        let before = device.work_counts();
        let mut answers = Vec::new();
        let textures: Vec<Texture> = FORMATS
            .iter()
            .map(|&format| {
                let (texture, answer) = surface_texture(&surface, format, (width, height, pitch));
                answers.push(answer);
                texture
            })
            .collect();
        let after = device.work_counts();
        let shadows = after.shadowed_textures - before.shadowed_textures;
        let pitched = after.pooled_pitched_images - before.pooled_pitched_images;
        // Each shadow here is an image of the texture's whole extent, 4 bytes a texel.
        let shadow_bytes = shadows * width * height * 4;
        eprintln!(
            "ALIASED-SURFACE {}: {} textures over one Private surface of {bytes} bytes: answers \
             {answers:?}; {shadows} shadows ({shadow_bytes} bytes), {pitched} images at Metal's \
             pitch",
            device.name(),
            textures.len(),
        );
        assert_eq!(answers[0], linear_cost, "the first texture over the bytes");
        for (answer, format) in answers.iter().zip(FORMATS).skip(1) {
            assert_eq!(
                *answer,
                TexturePlacement::NotHostPages,
                "a later {format:?} texture over the same bytes"
            );
        }
        assert_eq!(shadows, 1, "one shadow over the bytes");
        assert_eq!(
            pitched,
            FORMATS.len() as u64 - 1,
            "every later texture in place"
        );
        assert_eq!(shadow_bytes, width * height * 4, "one shadow's bytes");
        // The first texture's shadow moves to its image at Metal's pitch at the next commit of the
        // device's only uncommitted command buffer that takes storage forms (a blit's), and goes
        // once that command buffer completed: a later command buffer that reads it copies
        // nothing.
        let queue = device.new_command_queue().unwrap();
        let row = width * 4;
        let out = device
            .new_buffer(row * height, ResourceOptions::shared())
            .unwrap();
        let read = |texture: &Texture| {
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_from_texture_to_buffer(
                texture,
                0,
                0,
                Origin::new(0, 0, 0),
                Size::new(width, height, 1),
                &out,
                0,
                row,
                row * height,
                BlitOption::None,
            )
            .unwrap();
            blit.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None);
        };
        read(&textures[0]);
        let moved = device.work_counts();
        read(&textures[0]);
        let again = device.work_counts();
        eprintln!(
            "ALIASED-SURFACE {}: the first texture's shadow moved in place {}, freed {}; the \
             next read's shadow copies {}",
            device.name(),
            moved.pooled_textures_moved_in_place - after.pooled_textures_moved_in_place,
            again.pooled_texture_shadows_freed_after_move
                - after.pooled_texture_shadows_freed_after_move,
            again.shadow_copies - moved.shadow_copies,
        );
        assert_eq!(
            moved.pooled_textures_moved_in_place - after.pooled_textures_moved_in_place,
            1,
            "the first texture's shadow moved in place"
        );
        assert_eq!(
            again.shadow_copies - moved.shadow_copies,
            0,
            "a read after the move copies nothing"
        );
    });
}

#[test]
fn textures_made_one_after_another_over_one_surface_s_bytes_take_one_shadow() {
    super::host_memory::on_device_with(in_place, |device| {
        assert!(
            device.settle_in_place_checks(CHECK_WAIT),
            "the check settles"
        );
        if !takes_pitched(device) {
            eprintln!(
                "ALIASED-SURFACE {}: no DRM_FORMAT_MOD_LINEAR image at a {}-byte pitch: nothing \
                 to check",
                device.name(),
                SMALL.2
            );
            return;
        }
        let kernels = Kernels::new(device);
        let queue = device.new_command_queue().unwrap();
        let surface = small_surface(device);
        let (width, height, pitch) = SMALL;
        let bytes = pitch * height;
        let linear_cost = TexturePlacement::Shadow(TextureShadowReason::LinearCost { bytes });
        let before = device.work_counts();
        // Each texture made, written by a kernel, and released once its command buffer completed,
        // before the next is made over the same bytes, as a guest's frames make them.
        let answers: Vec<TexturePlacement> = FORMATS
            .iter()
            .enumerate()
            .map(|(n, &format)| {
                let (texture, answer) = surface_texture(&surface, format, (width, height, pitch));
                let command_buffer = queue.command_buffer().unwrap();
                kernels.fill(
                    &command_buffer,
                    &texture,
                    0x0102_0304 * (n as u32 + 1),
                    height,
                );
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.error(), None);
                answer
            })
            .collect();
        let after = device.work_counts();
        let shadows = after.shadowed_textures - before.shadowed_textures;
        let pitched = after.pooled_pitched_images - before.pooled_pitched_images;
        eprintln!(
            "ALIASED-SURFACE {}: {} textures made one after another over one Private surface of \
             {bytes} bytes, each released before the next: answers {answers:?}; {shadows} \
             shadows, {pitched} images at Metal's pitch",
            device.name(),
            answers.len(),
        );
        assert_eq!(answers[0], linear_cost, "the first texture over the bytes");
        for (answer, format) in answers.iter().zip(FORMATS).skip(1) {
            assert_eq!(
                *answer,
                TexturePlacement::NotHostPages,
                "a later {format:?} texture over bytes a released texture was made over"
            );
        }
        assert_eq!(shadows, 1, "one shadow over the bytes");
        assert_eq!(
            pitched,
            FORMATS.len() as u64 - 1,
            "every later texture in place"
        );
    });
}

#[test]
fn writes_through_one_texture_reach_every_texture_over_the_bytes() {
    super::host_memory::on_device_with(in_place, |device| {
        assert!(
            device.settle_in_place_checks(CHECK_WAIT),
            "the check settles"
        );
        let kernels = Kernels::new(device);
        let queue = device.new_command_queue().unwrap();
        let surface = small_surface(device);
        let (width, height, pitch) = SMALL;
        let textures: Vec<Texture> = FORMATS
            .iter()
            .map(|&format| surface_texture(&surface, format, (width, height, pitch)).0)
            .collect();
        let side = height;
        let outs: Vec<Buffer> = textures
            .iter()
            .map(|_| {
                device
                    .new_buffer(side * side * 16, ResourceOptions::shared())
                    .unwrap()
            })
            .collect();
        let bytes = device
            .new_buffer(pitch * height, ResourceOptions::shared())
            .unwrap();
        let mut round = 0u32;
        for (writer, written) in textures.iter().enumerate() {
            for same_command_buffer in [true, false] {
                round += 1;
                // Bytes Metal stores exactly through either format: each byte a value of its own.
                let word = 0x1020_3040 + round * 0x0101_0101;
                let command_buffer = queue.command_buffer().unwrap();
                kernels.fill(&command_buffer, written, word, side);
                let reads = match same_command_buffer {
                    true => command_buffer,
                    false => {
                        command_buffer.commit().unwrap();
                        queue.command_buffer().unwrap()
                    }
                };
                for (read, out) in textures.iter().zip(&outs) {
                    kernels.read(&reads, read, out, side);
                }
                let mut blit = reads.blit_command_encoder().unwrap();
                blit.copy_from_buffer_to_buffer(&surface, 0, &bytes, 0, pitch * height)
                    .unwrap();
                blit.end_encoding().unwrap();
                reads.commit().unwrap();
                reads.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(reads.error(), None);
                for (reader, (read, out)) in textures.iter().zip(&outs).enumerate() {
                    let got = words_read(read, out, side);
                    let wrong = got.iter().filter(|&&w| w != word).count();
                    assert_eq!(
                        wrong,
                        0,
                        "{}: written through texture {writer} ({:?}), read through {reader} \
                         ({:?}), in the same command buffer {same_command_buffer}: {wrong} of {} \
                         texels not {word:#x} (first {:#x})",
                        device.name(),
                        written.pixel_format(),
                        read.pixel_format(),
                        got.len(),
                        got[0]
                    );
                }
                let contents = bytes.contents().unwrap().as_ptr();
                for (y, x) in (0..side).flat_map(|y| (0..side).map(move |x| (y, x))) {
                    let at = (y * pitch + x * 4) as usize;
                    // SAFETY: a `Shared` buffer of the surface's bytes, which no command uses now.
                    let held = unsafe { std::slice::from_raw_parts(contents.add(at), 4) };
                    assert_eq!(
                        u32::from_le_bytes(held.try_into().unwrap()),
                        word,
                        "{}: the surface's texel ({x}, {y}) after a write through texture \
                         {writer}, in the same command buffer {same_command_buffer}",
                        device.name()
                    );
                }
            }
        }
    });
}

/// The measurement's surface, (width, rows, row bytes) and its buffer's length: the guest's.
const GUEST: (u64, u64, u64) = (6016, 6016, 24_576);
const GUEST_BYTES: u64 = 150_994_944;
/// The textures over it alive across the frames, and how many of them are `R32Uint`.
const LIVE: usize = 36;
const LIVE_UINT: usize = 11;
/// The textures made over it each frame of the transient case, half of each format.
const TRANSIENT: usize = 4;
/// The square a frame's kernel writes through each written texture, and the one it reads.
const WRITTEN: u64 = 1024;
const READ: u64 = 256;
/// The untimed and the timed frames of each case.
const WARM: usize = 1;
const TIMED: usize = if cfg!(m2v_cpu_icd) { 3 } else { 6 };

/// The bytes of device memory the process holds now, `(drm, resident)`: the resident (else
/// total) memory its DRM file descriptors report (`/proc/self/fdinfo`, the `drm-resident-*` or
/// `drm-memory-*` keys), and its resident set (`VmRSS`), where a CPU device's memory lies. Zeros
/// where the system reports neither.
fn memory_now() -> (u64, u64) {
    let bytes = |value: &str| -> u64 {
        let mut parts = value.split_whitespace();
        let number: u64 = parts.next().and_then(|n| n.parse().ok()).unwrap_or(0);
        match parts.next() {
            Some("KiB") | Some("kB") => number << 10,
            Some("MiB") => number << 20,
            Some("GiB") => number << 30,
            _ => number,
        }
    };
    let mut drm = 0;
    if let Ok(entries) = std::fs::read_dir("/proc/self/fdinfo") {
        for entry in entries.flatten() {
            let Ok(text) = std::fs::read_to_string(entry.path()) else {
                continue;
            };
            if !text.contains("drm-driver") {
                continue;
            }
            let lines: Vec<(&str, &str)> = text
                .lines()
                .filter_map(|line| line.split_once(':'))
                .map(|(k, v)| (k.trim(), v.trim()))
                .collect();
            let resident: u64 = lines
                .iter()
                .filter(|(k, _)| k.starts_with("drm-resident-"))
                .map(|(_, v)| bytes(v))
                .sum();
            let memory: u64 = lines
                .iter()
                .filter(|(k, _)| k.starts_with("drm-memory-"))
                .map(|(_, v)| bytes(v))
                .sum();
            drm += resident.max(memory);
        }
    }
    let rss = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find_map(|line| line.strip_prefix("VmRSS:").map(bytes))
        })
        .unwrap_or(0);
    (drm, rss)
}

/// A measurement device: the in-place policy, without the validation layer, its check settled.
fn measured_device(cache: &ScratchCache, messages: &Messages) -> Device {
    let device = validated_device_with(TEST_DEVICE, messages, cache.path(), |options| {
        options.layers.clear();
        in_place(options);
    })
    .unwrap();
    assert!(
        device.settle_in_place_checks(CHECK_WAIT),
        "the check settles"
    );
    device
}

/// What one case's frames moved and took, printed as an `ALIASED-SURFACE-COST` line.
struct Tally {
    times: Vec<f64>,
    first: WorkCounts,
    last: WorkCounts,
    peak: (u64, u64),
}

impl Tally {
    fn print(&self, device: &Device, case: &str, forms: &str) {
        let mut times = self.times.clone();
        times.sort_by(f64::total_cmp);
        let frames = times.len() as f64;
        let per = |f: fn(&WorkCounts) -> u64| (f(&self.last) - f(&self.first)) as f64 / frames;
        let shadow = GUEST.0 * GUEST.1 * 4;
        println!(
            "ALIASED-SURFACE-COST {} {case}: {forms}; a frame's GPU time median {:.2} ms, slowest \
             {:.2} ms; per frame: shadows made {:.1} ({:.0} MB), shadow copies {:.1}, texels \
             copied in {:.0} ({:.0} MB), back {:.0} ({:.0} MB), merge words examined {:.0}, \
             memory allocated {:.0} MB; peak device memory {:.0} MB (DRM), resident set {:.0} MB",
            device.name(),
            times[times.len() / 2] * 1e3,
            times[times.len() - 1] * 1e3,
            per(|c| c.shadowed_textures),
            per(|c| c.shadowed_textures) * shadow as f64 / 1e6,
            per(|c| c.shadow_copies),
            per(|c| c.shadow_texels_copied_in),
            per(|c| c.shadow_texels_copied_in) * 4.0 / 1e6,
            per(|c| c.shadow_texels_copied_back),
            per(|c| c.shadow_texels_copied_back) * 4.0 / 1e6,
            per(|c| c.shadow_merge_words_examined),
            per(|c| c.device_memory_allocated) / 1e6,
            self.peak.0 as f64 / 1e6,
            self.peak.1 as f64 / 1e6,
        );
    }
}

/// The format of the `i`th of the [`LIVE`] textures: [`LIVE_UINT`] of them `R32Uint`, spread
/// among the rest.
fn live_format(i: usize) -> PixelFormat {
    match (i * LIVE_UINT) % LIVE < LIVE_UINT {
        true => PixelFormat::R32Uint,
        false => PixelFormat::BGRA8Unorm,
    }
}

/// Run `frames` frames of `frame` (which returns the command buffer it committed), timed after
/// [`WARM`], the memory sampled after each.
fn tally(device: &Device, mut frame: impl FnMut(usize) -> f64) -> Tally {
    let mut peak = memory_now();
    for n in 0..WARM {
        frame(n);
    }
    let first = device.work_counts();
    let mut times = Vec::new();
    for n in WARM..WARM + TIMED {
        times.push(frame(n));
        let (drm, rss) = memory_now();
        peak = (peak.0.max(drm), peak.1.max(rss));
    }
    Tally {
        times,
        first,
        last: device.work_counts(),
        peak,
    }
}

#[test]
#[ignore = "a release-mode measurement: run by name"]
fn aliased_surface_costs() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = measured_device(&cache, &messages);
    let kernels = Kernels::new(&device);
    let queue = device.new_command_queue().unwrap();
    let out = device
        .new_buffer(READ * READ * 16, ResourceOptions::private())
        .unwrap();
    println!(
        "ALIASED-SURFACE-COST {} start: device memory {:?} (DRM, resident set)",
        device.name(),
        memory_now()
    );
    for options in [ResourceOptions::private(), ResourceOptions::shared()] {
        let storage = format!("{:?}", options.storage_mode());
        // Live: the textures made once, a frame writing two and reading four.
        {
            let surface = made_surface(&device, GUEST_BYTES, options);
            let before = device.work_counts();
            let made: Vec<(Texture, TexturePlacement)> = (0..LIVE)
                .map(|i| surface_texture(&surface, live_format(i), GUEST))
                .collect();
            let after = device.work_counts();
            let shadows = after.shadowed_textures - before.shadowed_textures;
            let pitched = after.pooled_pitched_images - before.pooled_pitched_images;
            let forms = format!(
                "{LIVE} textures over a {storage} surface, first answered {:?}, last {:?}: \
                 {shadows} shadows ({} MB), {pitched} images at Metal's pitch",
                made[0].1,
                made[LIVE - 1].1,
                shadows * GUEST.0 * GUEST.1 * 4 / 1_000_000
            );
            let (uints, floats): (Vec<&Texture>, Vec<&Texture>) = made
                .iter()
                .map(|(t, _)| t)
                .partition(|t| t.pixel_format() == PixelFormat::R32Uint);
            let tally = tally(&device, |n| {
                let command_buffer = queue.command_buffer().unwrap();
                let word = 0x8040_2010 ^ n as u32;
                kernels.fill(&command_buffer, floats[n % floats.len()], word, WRITTEN);
                kernels.fill(&command_buffer, uints[n % uints.len()], word, WRITTEN);
                for k in 1..3 {
                    kernels.read(&command_buffer, floats[(n + k) % floats.len()], &out, READ);
                    kernels.read(&command_buffer, uints[(n + k) % uints.len()], &out, READ);
                }
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.error(), None);
                command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap()
            });
            tally.print(&device, "live", &forms);
        }
        // Transient: four textures made each frame and dropped after it, two written, two read.
        {
            let surface = made_surface(&device, GUEST_BYTES, options);
            let mut answers = Vec::new();
            let tally = tally(&device, |n| {
                let made: Vec<(Texture, TexturePlacement)> = (0..TRANSIENT)
                    .map(|i| {
                        let format = match i % 2 {
                            0 => PixelFormat::BGRA8Unorm,
                            _ => PixelFormat::R32Uint,
                        };
                        surface_texture(&surface, format, GUEST)
                    })
                    .collect();
                if answers.len() < 2 * TRANSIENT {
                    answers.extend(made.iter().map(|(_, answer)| *answer));
                }
                let command_buffer = queue.command_buffer().unwrap();
                let word = 0x0102_0408 ^ n as u32;
                kernels.fill(&command_buffer, &made[0].0, word, WRITTEN);
                kernels.fill(&command_buffer, &made[1].0, word, WRITTEN);
                kernels.read(&command_buffer, &made[2].0, &out, READ);
                kernels.read(&command_buffer, &made[3].0, &out, READ);
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.error(), None);
                command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap()
            });
            let forms = format!(
                "{TRANSIENT} textures a frame over a {storage} surface, the first two frames' \
                 answers {answers:?}"
            );
            tally.print(&device, "transient", &forms);
        }
    }
    messages.take_notices();
}
