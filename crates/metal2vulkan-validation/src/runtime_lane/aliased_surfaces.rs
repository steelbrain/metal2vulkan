//! Many textures over one `Private` surface's bytes, on the device the cfg selects: a guest's
//! window server, at its login, makes dozens of textures over each of its large surfaces
//! (`new_texture_with_iosurface_plane` over the `Private` buffer that stands for the surface),
//! `BGRA8Unorm` and `R32Uint` views of the same bytes with shader read and write usage, and writes
//! and reads a few of them each frame. Each surface's maker declares the bytes between its rows
//! unobserved (`BufferDescriptor::padding_unobserved`), as the guest's do. Each texture over a
//! `Private` buffer whose rows the device's own linear image lays out otherwise is a
//! `DRM_FORMAT_MOD_LINEAR` image at Metal's pitch, at any size, where the device makes one and its
//! check of its pooled memory passed: each texture over the bytes an image of its own over them,
//! no copy.
//!
//! - [`textures_over_one_surface_s_bytes_each_take_the_image_at_metal_s_pitch`]: four textures
//!   (two of each format) over one `Private` surface, where the device makes such images: each
//!   takes the image at Metal's pitch (`WorkCounts::pooled_pitched_images`), none is a shadow, and
//!   a read through the first copies nothing.
//! - [`writes_through_one_texture_reach_every_texture_over_the_bytes`]: a kernel's write through
//!   each of those textures in turn is what a kernel reads through every other one, in the same
//!   command buffer and in the next, and what the surface's bytes hold: Metal's aliasing of
//!   textures over the same bytes.

use super::first_use_writes::CHECK_WAIT;
use super::in_place_textures::in_place;
use super::WAIT;
use crate::air_library::AirLibrary;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    BlitOption, Buffer, BufferContents, BufferDescriptor, CommandBuffer, ComputePipelineState,
    Device, IoSurfacePlane, Origin, PipelineOptions, ResourceOptions, Size, Texture,
    TextureDescriptor, TexturePlacement,
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
/// pitch of 1280 bytes, 64 rows (81,920 bytes).
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
/// `SMALL`'s shape over its pooled memory, over a `Shared` buffer, where its check passed.
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
fn textures_over_one_surface_s_bytes_each_take_the_image_at_metal_s_pitch() {
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
        eprintln!(
            "ALIASED-SURFACE {}: {} textures over one Private surface of {bytes} bytes: answers \
             {answers:?}; {shadows} shadows, {pitched} images at Metal's pitch",
            device.name(),
            textures.len(),
        );
        for (answer, format) in answers.iter().zip(FORMATS) {
            assert_eq!(
                *answer,
                TexturePlacement::NotHostPages,
                "a {format:?} texture over the bytes"
            );
        }
        assert_eq!(shadows, 0, "no shadow over the bytes");
        assert_eq!(pitched, FORMATS.len() as u64, "every texture in place");
        // A read through the first texture copies nothing.
        let queue = device.new_command_queue().unwrap();
        let row = width * 4;
        let out = device
            .new_buffer(row * height, ResourceOptions::shared())
            .unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            &textures[0],
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
        let read = device.work_counts();
        assert_eq!(
            read.shadow_copies - after.shadow_copies,
            0,
            "a read copies nothing"
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
