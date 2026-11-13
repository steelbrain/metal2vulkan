//! An `A8Unorm` texture over a buffer's bytes that only reads is a shadow only where an `R8Unorm`
//! texture over the same bytes is one: where the device's `A8_UNORM` image in a tiling a texture
//! over a buffer takes lacks what its optimal one does and its `R8_UNORM` image lacks nothing
//! (the RTX 3090 lists no `DRM_FORMAT_MOD_LINEAR` modifier for `A8_UNORM`, so no image of it lies
//! at a pitch other than its linear image's), its image is the one an `R8Unorm` texture over the
//! same bytes takes, which its bindings read through `(Zero, Zero, Zero, Red)`
//! (`texture/red_held_alpha.rs`). Over a `Shared` buffer of the device's own memory and over a
//! no-copy buffer's host pages, `WIDTH` texels wide at tight rows (no shadow) and `NARROW` wide at
//! the same rows, a kernel's `read` through a sampler (`kernel_read_through.ll`'s `rt_repeat`) of
//! each texel reads `(0, 0, 0, code / 255)`, what the same kernel reads of an `A8Unorm` texture
//! of the device's own holding the same bytes. Where the device's `A8_UNORM` images lack nothing
//! (lavapipe, RADV), the texture's image is `A8_UNORM`'s, as before.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. Printed (`A8-OVER-BUFFERS`): the
//! device's linear, optimal and `DRM_FORMAT_MOD_LINEAR` features of both formats, the
//! placements of glyph-sized textures, and each texture's image format and shadow count.

use super::blit_holds::{device, Host, Scratch, WAIT};
use crate::{
    Buffer, CommandBufferStatus, ComputePipelineState, Device, PipelineOptions, Region,
    ResourceOptions, Size, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use std::time::Duration;

const TEST: &str = "A8-OVER-BUFFERS";

/// The texture's width (its rows' bytes, tight) and height.
const WIDTH: u64 = 256;
const HEIGHT: u64 = 16;

/// The width of a texture over the same rows, `WIDTH` apart: a glyph's, whose rows the driver's
/// linear image may lay at another pitch (NVIDIA's at 128 bytes).
const NARROW: u64 = 100;

/// The longest the device's check of images in place may take.
const CHECK_WAIT: Duration = Duration::from_secs(120);

/// The texture's bytes: every code, in an order no row repeats.
fn codes() -> Vec<u8> {
    (0..WIDTH * HEIGHT)
        .map(|i| ((i * 7 + i / WIDTH * 3) % 256) as u8)
        .collect()
}

/// The compute state of `kernel_read_through.ll`'s `rt_repeat` on `device`.
fn reader(device: &Device) -> ComputePipelineState {
    let ll = metal2vulkan_devtools::engine_dir().join("kernel_read_through.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    let function = device
        .new_library_with_compile_library(library)
        .new_function_with_name("rt_repeat")
        .unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A `Shared` buffer of the device's own memory holding `bytes`.
fn holding(device: &Device, bytes: &[u8]) -> Buffer {
    let buffer = device
        .new_buffer(bytes.len() as u64, ResourceOptions::shared())
        .unwrap();
    let contents = buffer.contents().unwrap();
    // SAFETY: the new buffer's bytes, which no command uses yet.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
    buffer
}

/// What `rt_repeat` reads of `texture` at each texel, row by row.
fn read(device: &Device, state: &ComputePipelineState, texture: &Texture) -> Vec<[f32; 4]> {
    let width = texture.width();
    let texels = (width * texture.height()) as usize;
    let positions: Vec<u8> = (0..texels as u64)
        .flat_map(|i| [(i % width) as i32, (i / width) as i32, 0, 0])
        .flat_map(i32::to_le_bytes)
        .collect();
    let positions = holding(device, &positions);
    let out = holding(device, &vec![0; 16 * texels]);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_buffer(Some(&positions), 0, 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 1).unwrap();
    let threads = Size::new(texels as u64, 1, 1);
    encoder
        .dispatch_threads(threads, Size::new(64, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    let contents = out.contents().unwrap();
    // SAFETY: the buffer's bytes, which the completed command buffer wrote.
    let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), 16 * texels) };
    let words: Vec<f32> = (bytes.as_chunks::<4>().0.iter())
        .map(|&word| f32::from_le_bytes(word))
        .collect();
    words.as_chunks::<4>().0.to_vec()
}

/// The descriptor of the textures: `pf`, `width` by `HEIGHT`, read alone.
fn descriptor(pf: PixelFormat, width: u64) -> TextureDescriptor {
    let mut d = TextureDescriptor::texture_2d(pf, width, HEIGHT, false);
    d.usage = TextureUsage::SHADER_READ;
    d
}

/// The features a texture that samples, filters and is copied asks of its image.
const NAMED: [(ash::vk::FormatFeatureFlags, &str); 5] = [
    (ash::vk::FormatFeatureFlags::SAMPLED_IMAGE, "sampled"),
    (
        ash::vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR,
        "linear filter",
    ),
    (
        ash::vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_MINMAX,
        "minmax filter",
    ),
    (ash::vk::FormatFeatureFlags::TRANSFER_SRC, "transfer source"),
    (
        ash::vk::FormatFeatureFlags::TRANSFER_DST,
        "transfer destination",
    ),
];

/// The names of `NAMED`'s features `flags` holds.
fn named(flags: ash::vk::FormatFeatureFlags) -> Vec<&'static str> {
    (NAMED.iter())
        .filter(|&&(feature, _)| flags.contains(feature))
        .map(|&(_, name)| name)
        .collect()
}

/// The device's linear and optimal tiling features of `format`, as it reports them.
fn features(device: &Device, format: ash::vk::Format) -> String {
    // SAFETY: the handles are used while the device lives.
    let handles = unsafe { device.raw() };
    // SAFETY: the physical device is the instance's.
    let properties = unsafe {
        (handles.instance).get_physical_device_format_properties(handles.physical_device, format)
    };
    format!(
        "format {}: linear {:?}, optimal {:?}, DRM_FORMAT_MOD_LINEAR {}",
        format.as_raw(),
        named(properties.linear_tiling_features),
        named(properties.optimal_tiling_features),
        modifier_features(device, format)
    )
}

/// The device's features of `format`'s `DRM_FORMAT_MOD_LINEAR` modifier, as it reports them.
fn modifier_features(device: &Device, format: ash::vk::Format) -> String {
    if !device.makes_modifier_images() {
        return "not asked (no VK_EXT_image_drm_format_modifier)".to_owned();
    }
    // SAFETY: the handles are used while the device lives.
    let handles = unsafe { device.raw() };
    let query = |list: &mut ash::vk::DrmFormatModifierPropertiesListEXT| {
        let mut properties = ash::vk::FormatProperties2::default().push_next(list);
        // SAFETY: the physical device is the instance's, whose device enables the extension; a
        // list without entries asks the count alone, and one with entries holds as many as its
        // count for the driver to write.
        unsafe {
            (handles.instance).get_physical_device_format_properties2(
                handles.physical_device,
                format,
                &mut properties,
            )
        }
    };
    let mut list = ash::vk::DrmFormatModifierPropertiesListEXT::default();
    query(&mut list);
    let count = list.drm_format_modifier_count as usize;
    let mut modifiers = vec![ash::vk::DrmFormatModifierPropertiesEXT::default(); count];
    let mut list = ash::vk::DrmFormatModifierPropertiesListEXT::default()
        .drm_format_modifier_properties(&mut modifiers);
    query(&mut list);
    let written = list.drm_format_modifier_count as usize;
    let linear = (modifiers.iter().take(written)).find(|modifier| {
        modifier.drm_format_modifier == crate::dma_buf::DRM_FORMAT_MOD_LINEAR
            && modifier.drm_format_modifier_plane_count == 1
    });
    match linear {
        Some(modifier) => format!("{:?}", named(modifier.drm_format_modifier_tiling_features)),
        None => "not listed".to_owned(),
    }
}

/// The placement of an `A8Unorm` texture of each of a glyph's widths, 16 rows a multiple of 16
/// bytes apart, over a `Shared` buffer, beside the row pitch of the device's linear image of its
/// plan (`Device::linear_texture_bytes_per_row`) and the format's minimum linear alignment:
/// printed alone.
fn widths(device: &Device) {
    let alignment = device.minimum_linear_texture_alignment_for_pixel_format(PixelFormat::A8Unorm);
    println!(
        "{TEST} on {}: A8Unorm's minimum linear texture alignment {alignment:?}",
        device.name()
    );
    for width in [10_u64, 24, 64, 100, 128, 200, 256, 448] {
        let pitch = width.next_multiple_of(16);
        let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::A8Unorm, width, 16, false);
        descriptor.usage = TextureUsage::SHADER_READ;
        let buffer = device
            .new_buffer(pitch * 16, ResourceOptions::shared())
            .unwrap();
        let placement = buffer.texture_placement(&descriptor, 0, pitch);
        let linear = device.linear_texture_bytes_per_row(&descriptor);
        println!(
            "{TEST} on {}: width {width}, rows {pitch} apart: {placement:?}; the device's \
             linear image's rows {linear:?} apart",
            device.name()
        );
    }
}

/// The shadows the making of a texture of `descriptor` over `buffer`'s first bytes, rows `pitch`
/// apart, counted.
fn shadows_made(buffer: &Buffer, descriptor: &TextureDescriptor, pitch: u64) -> u64 {
    let device = buffer.generation();
    let before = device.work_counts().shadowed_textures;
    drop(buffer.new_texture(descriptor, 0, pitch).unwrap());
    device.work_counts().shadowed_textures - before
}

/// An `A8Unorm` texture `width` texels wide over `buffer`'s first bytes, rows `pitch` apart,
/// with what it read, whether it is a shadow, and its image's format: no shadow where its rows
/// are tight, and a shadow only where an `R8Unorm` texture over the same bytes is one.
fn over(
    device: &Device,
    state: &ComputePipelineState,
    buffer: &Buffer,
    what: &str,
    (width, pitch): (u64, u64),
) -> Vec<[f32; 4]> {
    let red = shadows_made(buffer, &descriptor(PixelFormat::R8Unorm, width), pitch);
    let alpha = descriptor(PixelFormat::A8Unorm, width);
    let before = device.work_counts();
    let placement = buffer.texture_placement(&alpha, 0, pitch).unwrap();
    let texture = buffer.new_texture(&alpha, 0, pitch).unwrap();
    let after = device.work_counts();
    let shadows = after.shadowed_textures - before.shadowed_textures;
    let held = after.alpha_textures_held_in_red - before.alpha_textures_held_in_red;
    let format = texture.image_plan().map(|plan| plan.format);
    println!(
        "{TEST} on {}: over {what}, {width} wide, rows {pitch} apart: {placement:?}, {shadows} \
         shadow(s) made ({red} for R8Unorm), {held} held in red, its image {format:?}",
        device.name()
    );
    let reads = read(device, state, &texture);
    assert!(
        shadows <= red,
        "{TEST}: an A8Unorm texture over {what} that only reads is a shadow ({placement:?}) \
         where an R8Unorm one over the same bytes is not"
    );
    assert!(
        width != pitch || shadows == 0,
        "{TEST}: an A8Unorm texture over {what} that only reads is no shadow ({placement:?})"
    );
    // A device that holds no `A8_UNORM` image at all holds every `A8Unorm` texture in red, uncounted.
    let profile = device.compile_device().profile();
    let everywhere = metal2vulkan::metal::format::alpha_in_red(PixelFormat::A8Unorm, profile);
    let red = format.is_ok_and(|format| format == metal2vulkan::vk::Format::R8_UNORM);
    assert_eq!(
        held,
        u64::from(red && !everywhere),
        "{TEST}: an image of R8_UNORM is counted held in red"
    );
    reads
}

#[test]
fn an_a8_texture_over_a_buffer_that_only_reads_is_in_place_and_reads_its_alpha() {
    let scratch = Scratch::new("a8-over-buffers");
    let device = device(&scratch.0);
    assert!(
        device.settle_in_place_checks(CHECK_WAIT),
        "{TEST}: the check"
    );
    for format in [ash::vk::Format::A8_UNORM_KHR, ash::vk::Format::R8_UNORM] {
        println!("{TEST} on {}: {}", device.name(), features(&device, format));
    }
    widths(&device);
    let state = reader(&device);
    let bytes = codes();
    // The device's own textures of the same bytes, rows `WIDTH` apart, all of each row and its
    // first `NARROW` bytes: what the kernel reads of an `A8Unorm` texture.
    let own = |width| {
        let mut own = descriptor(PixelFormat::A8Unorm, width);
        own.resource_options = ResourceOptions::shared();
        let own = device.new_texture(&own).unwrap();
        let whole = Region::new_2d(0, 0, width, HEIGHT);
        own.replace_region(whole, 0, 0, &bytes, WIDTH, 0).unwrap();
        read(&device, &state, &own)
    };
    let (want, narrow) = (own(WIDTH), own(NARROW));
    for (texel, (read, &code)) in want.iter().zip(&bytes).enumerate() {
        let alpha = f32::from(code) / 255.0;
        assert!(
            read[..3] == [0.0; 3] && (read[3] - alpha).abs() < 1e-6,
            "{TEST}: the device's own A8Unorm texture at texel {texel} reads {read:?}, not \
             (0, 0, 0, {alpha})"
        );
    }
    let shared = holding(&device, &bytes);
    let reads = over(&device, &state, &shared, "a Shared buffer", (WIDTH, WIDTH));
    assert_eq!(reads, want, "{TEST}: over a Shared buffer");
    let reads = over(&device, &state, &shared, "a Shared buffer", (NARROW, WIDTH));
    assert_eq!(reads, narrow, "{TEST}: over a Shared buffer, {NARROW} wide");
    let host = Host::new();
    let words: Vec<u32> = (bytes.as_chunks::<4>().0.iter())
        .map(|&word| u32::from_le_bytes(word))
        .collect();
    host.set(0, &words);
    // SAFETY: the pages outlive the buffer (the host is dropped after it), which alone uses them.
    let no_copy = unsafe {
        device.new_buffer_with_bytes_no_copy(
            host.memory(),
            super::blit_holds::BYTES as u64,
            ResourceOptions::default(),
            None,
        )
    };
    match no_copy {
        Ok(no_copy) => {
            let what = "a no-copy buffer's host pages";
            let reads = over(&device, &state, &no_copy, what, (WIDTH, WIDTH));
            assert_eq!(reads, want, "{TEST}: over {what}");
            let reads = over(&device, &state, &no_copy, what, (NARROW, WIDTH));
            assert_eq!(reads, narrow, "{TEST}: over {what}, {NARROW} wide");
        }
        Err(error) => println!(
            "{TEST} on {}: the device imports no host pages ({error:?}): over them not run",
            device.name()
        ),
    }
}
