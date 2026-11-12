//! A texture that renders over a buffer's rows with bytes between them, in memory only the
//! device's ordered commands write (a `Private` buffer, and a `Managed` one's own copy on a device
//! without unified memory: `Buffer::gpu_bytes_ordered`): its image lies over the rows in place,
//! holding those bytes, where the device's check of renders over its pooled memory passed for
//! texels that lie in lines as these do (`Device::ordered_renders_kept`), and is a shadow where
//! it failed or did not run. Either way a render pass that clears the texture and stores it
//! leaves every byte between its rows as the buffer held it, and its texels the clear's, as a
//! blit of the buffer reads them back.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`). Printed
//! (`ORDERED-RENDERS`): each case's placement, what its making counted, and the check's
//! verdicts.

use super::blit_holds::{device, Scratch, WAIT};
use super::merge_scratch::{blit, read};
use crate::in_place::{Access, PagesClass, RowEnds, Verdict};
use crate::{
    Buffer, ClearColor, CommandBufferStatus, CommandQueue, CpuCacheMode, Device,
    HazardTrackingMode, LoadAction, RenderPassDescriptor, ResourceOptions, StorageMode,
    StoreAction, TextureDescriptor,
};
use metal2vulkan::device::HostPageTexturePolicy;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::time::Duration;

const TEST: &str = "ORDERED-RENDERS";

/// The texture's width and height in `RGBA8Unorm` texels, and its rows' pitch: 152 bytes of
/// texels a row, 104 between rows.
const WIDTH: u64 = 38;
const HEIGHT: u64 = 8;
const PITCH: u64 = 256;

/// The longest the device's check of images in place may take.
const CHECK_WAIT: Duration = Duration::from_secs(120);

/// The clear's texel, as `RGBA8Unorm` bytes.
const CLEAR: [u8; 4] = [255, 0, 255, 0];

/// The buffer's bytes: a pattern of no clear texel.
fn pattern() -> Vec<u8> {
    (0..PITCH * HEIGHT)
        .map(|i| ((i * 13 + i / PITCH * 5) % 200 + 20) as u8)
        .collect()
}

/// A buffer of the device's own memory of `options` holding `bytes`, put there by the CPU
/// (`Managed`, through `did_modify_range`) or by a blit from a `Shared` buffer (`Private`).
fn holding(queue: &CommandQueue, bytes: &[u8], options: ResourceOptions) -> Buffer {
    let device = queue.device();
    let length = bytes.len() as u64;
    let buffer = device.new_buffer(length, options).unwrap();
    if options.storage_mode().unwrap() == StorageMode::Private {
        let staging = device
            .new_buffer(length, ResourceOptions::shared())
            .unwrap();
        let contents = staging.contents().unwrap();
        // SAFETY: the new buffer's bytes, which no command uses yet.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
        blit(queue, |encoder| {
            (encoder.copy_from_buffer_to_buffer(&staging, 0, &buffer, 0, length)).unwrap()
        });
        return buffer;
    }
    let contents = buffer.contents().unwrap();
    // SAFETY: the new buffer's bytes, which no command uses yet.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
    buffer.did_modify_range(0..length).unwrap();
    buffer
}

/// The device's verdicts of renders over its pooled memory, for texels in lines and on them.
fn verdicts(device: &Device) -> [Verdict; 2] {
    RowEnds::ALL.map(|rows| device.in_place_verdict(PagesClass::Pooled, Access::Renders, rows))
}

/// One case: an `RGBA8Unorm` texture that renders over a buffer of `options` holding `bytes`,
/// cleared and stored by a render pass, then the buffer read back.
fn case(queue: &CommandQueue, what: &str, options: ResourceOptions, bytes: &[u8]) {
    let device = queue.device();
    let buffer = holding(queue, bytes, options);
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, WIDTH, HEIGHT, false);
    descriptor.resource_options = options;
    // A texture over the same rows that only reads: whether the buffer's memory takes an image
    // holding the bytes between rows at all (not memory allocated exportable, for one).
    descriptor.usage = TextureUsage::SHADER_READ;
    let before = device.work_counts();
    drop(buffer.new_texture(&descriptor, 0, PITCH).unwrap());
    let reads_kept = device.work_counts().kept_padding_aliases - before.kept_padding_aliases;
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    let placement = buffer.texture_placement(&descriptor, 0, PITCH).unwrap();
    let before = device.work_counts();
    let texture = buffer.new_texture(&descriptor, 0, PITCH).unwrap();
    let after = device.work_counts();
    let shadows = after.shadowed_textures - before.shadowed_textures;
    let kept = after.kept_padding_aliases - before.kept_padding_aliases;
    let verdicts = verdicts(device);
    println!(
        "{TEST} on {}: {what}: {placement:?}; {shadows} shadow(s), {kept} kept padding \
         alias(es) ({reads_kept} for one that only reads); renders over pooled memory \
         {verdicts:?} (in lines, on lines)",
        device.name()
    );
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(texture.clone());
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    color.clear_color = ClearColor::new(1.0, 0.0, 1.0, 0.0);
    let command_buffer = queue.command_buffer().unwrap();
    (command_buffer.render_command_encoder(&pass).unwrap())
        .end_encoding()
        .unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let length = bytes.len() as u64;
    let out = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    blit(queue, |encoder| {
        (encoder.copy_from_buffer_to_buffer(&buffer, 0, &out, 0, length)).unwrap()
    });
    let now = read(&out, length);
    for row in 0..HEIGHT as usize {
        let start = row * PITCH as usize;
        let texels = start + WIDTH as usize * 4;
        let cleared = now[start..texels].chunks(4).all(|texel| texel == CLEAR);
        assert!(
            cleared,
            "{TEST}: {what}: row {row}'s texels are not the clear's"
        );
        let end = start + PITCH as usize;
        assert_eq!(
            now[texels..end],
            bytes[texels..end],
            "{TEST}: {what}: the bytes after row {row}'s texels changed"
        );
    }
    let policy = device.compile_device().policy().host_page_textures;
    let passed = policy == HostPageTexturePolicy::InPlace && verdicts == [Verdict::Passed; 2];
    if passed && reads_kept == 1 {
        assert_eq!(
            (shadows, kept),
            (0, 1),
            "{TEST}: {what}: renders over the pooled memory passed, so the image is in place"
        );
    }
    if !verdicts.contains(&Verdict::Passed) {
        assert_eq!(
            (shadows, kept),
            (1, 0),
            "{TEST}: {what}: renders over the pooled memory did not pass, so it is a shadow"
        );
    }
}

#[test]
fn a_texture_that_renders_over_padded_rows_in_ordered_memory_keeps_their_padding() {
    let scratch = Scratch::new("ordered-renders");
    let device = device(&scratch.0);
    assert!(
        device.settle_in_place_checks(CHECK_WAIT),
        "{TEST}: the check"
    );
    let queue = device.new_command_queue().unwrap();
    let bytes = pattern();
    case(
        &queue,
        "a Private buffer",
        ResourceOptions::private(),
        &bytes,
    );
    if device.properties().has_unified_memory {
        println!(
            "{TEST} on {}: unified memory, where a Managed buffer is one copy the CPU writes: \
             not run",
            device.name()
        );
        return;
    }
    let managed = ResourceOptions::new(
        StorageMode::Managed,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    );
    case(&queue, "a Managed buffer", managed, &bytes);
}
