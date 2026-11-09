//! 32-bit float depths outside `[0, 1]` uploaded into depth images under the validation layer's
//! GPU-assisted check of buffer-to-image copies (`gpuav_buffer_copies`, which reports
//! `VUID-vkCmdCopyBufferToImage-pRegions-07931` for each depth outside `[0, 1]` a copy takes into
//! a depth image of a device without `VK_EXT_depth_range_unrestricted`). One lane a path:
//! `replace_region`, a blit from a buffer (its copy as it is, through its plan's staging buffer,
//! and into `Depth32Float_Stencil8`'s depth), and a copy from a colour texture.
//!
//! Where the device may store such a depth clamped ([`clamps`]: no such extension enabled and no
//! `DriverFact::DepthImageKeepsValuesOutsideUnitRange`, as ANV on the Iris Xe), each upload clamps
//! its depths into `[0, 1]` first (`WorkCounts::depth_uploads_clamped`, one an upload): no copy
//! takes a depth outside there, and each texel reads back as `bind::clamped_depth` gives it, a
//! zero with its sign or without (the device's to keep). Elsewhere (lavapipe enables the
//! extension) nothing is clamped and each texel reads back as written, a NaN as a NaN. Each lane
//! prints `DEPTHCLAMP` with whether the device clamps and the uploads clamped.

use super::host_memory::run;
use super::{
    layer_reported_nothing_at_all, read_buffer, validated_device_with, Messages, ScratchCache,
    VALIDATION_LAYER,
};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::clamped_depth;
use metal2vulkan::device::DriverFact;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_runtime::{
    BlitOption, Buffer, CpuCacheMode, Device, HazardTrackingMode, LayerSetting, LayerSettingValue,
    MessageSeverity, Origin, Region, ResourceOptions, Size, StorageMode, Texture,
    TextureDescriptor,
};
use std::path::Path;

/// The textures' sides: rows of more than one workgroup's depths, and of an odd width.
const SIDES: [u64; 2] = [37, 5];

/// A lane device whose validation layer checks each buffer-to-image copy's depths on the GPU
/// (`gpuav_enable`, `gpuav_buffer_copies`), its shader instrumentation off: the check this
/// module's lanes and `heap_textures`' read.
pub(super) fn gpu_assisted_copies(messages: &Messages, cache: &Path) -> Device {
    validated_device_with(TEST_DEVICE, messages, cache, |options| {
        let set = |name: &std::ffi::CStr, on: bool| {
            let value = LayerSettingValue::Bool(on);
            LayerSetting::new(VALIDATION_LAYER.to_owned(), name.to_owned(), value)
        };
        options.layer_settings.extend([
            set(c"gpuav_enable", true),
            set(c"gpuav_buffer_copies", true),
            set(c"gpuav_shader_instrumentation", false),
        ]);
    })
    .unwrap()
}

/// Whether `device` may store a depth outside `[0, 1]` a copy writes into a depth image clamped,
/// so that its uploads clamp first: no `VK_EXT_depth_range_unrestricted` enabled, and no
/// `DriverFact::DepthImageKeepsValuesOutsideUnitRange`.
pub(super) fn clamps(device: &Device) -> bool {
    let facts = &device.properties().driver_facts;
    !device.enables_extension(ash::ext::depth_range_unrestricted::NAME)
        && !facts.contains(&DriverFact::DepthImageKeepsValuesOutsideUnitRange)
}

/// Check what the layer reported since `messages` was last read, printing it after `label`: no
/// copy of a depth outside `[0, 1]`, and nothing else but the layer's warnings about the settings
/// the GPU-assisted check takes (`VALIDATION-SETTINGS`: that it runs beside the core checks;
/// `WARNING-Setting-Limit-Adjusted`: the descriptor limits it lowers for its own).
pub(super) fn reported_no_depth_outside_0_1(messages: &Messages, label: &str) {
    let reported = messages.take();
    let (range, other): (Vec<_>, Vec<_>) = (reported.iter())
        .partition(|message| message.id_name.contains("07931") || message.text.contains("07931"));
    eprintln!(
        "{label}: {} copies outside [0, 1] reported, and {} other messages: {other:#?}",
        range.len(),
        other.len()
    );
    assert!(
        range.is_empty(),
        "a copy of a depth outside [0, 1]: {range:#?}"
    );
    let settings = ["VALIDATION-SETTINGS", "WARNING-Setting-Limit-Adjusted"];
    let unexpected: Vec<_> = (other.iter())
        .filter(|message| {
            let setting = settings.contains(&message.id_name.as_str());
            !(message.severity == MessageSeverity::Warning && setting)
        })
        .collect();
    assert!(unexpected.is_empty(), "reported: {unexpected:#?}");
    layer_reported_nothing_at_all(messages);
}

/// Texel `i`'s bits: a depth outside `[0, 1]` or a NaN, else one inside naming it, or `-0.0`.
pub(super) fn depth_bits(i: u64) -> u32 {
    match i % 8 {
        0 => 1.5f32.to_bits(),
        1 => (-0.5f32).to_bits(),
        2 => 0x7fc0_0000,
        3 => f32::INFINITY.to_bits(),
        4 => ((i + 1) as f32 / (1u32 << 23) as f32).to_bits(),
        5 => (-0.0f32).to_bits(),
        6 => 2.0f32.to_bits(),
        _ => 0xeeee_eeee,
    }
}

/// Whether a texel written `bits` and read `got` is what the device stores of it: clamped where
/// it clamps (`clamps`), else as it is, a NaN as any NaN; a zero with its sign or without.
pub(super) fn stored_as(bits: u32, got: u32, clamps: bool) -> bool {
    let want = match clamps {
        true => clamped_depth(bits),
        false => bits,
    };
    let nan = |w: u32| f32::from_bits(w).is_nan();
    got == want || (want | got) & 0x7fff_ffff == 0 || (!clamps && nan(want) && nan(got))
}

/// The texels read otherwise than [`stored_as`] gives them, each with its bits as written (by
/// `bits`) and as read.
fn read_otherwise(read: &[u8], bits: impl Fn(u64) -> u32, clamps: bool) -> Vec<(u64, u32, u32)> {
    (read.as_chunks::<4>().0.iter().enumerate())
        .map(|(i, word)| (i as u64, bits(i as u64), u32::from_le_bytes(*word)))
        .filter(|&(_, bits, got)| !stored_as(bits, got, clamps))
        .collect()
}

/// The bytes of a texture's texels written by `bits`, tightly packed.
fn written(bits: impl Fn(u64) -> u32) -> Vec<u8> {
    let [w, h] = SIDES;
    (0..w * h).flat_map(|i| bits(i).to_le_bytes()).collect()
}

/// A Shared buffer holding `bytes`.
fn staged(device: &Device, bytes: &[u8]) -> Buffer {
    let buffer = (device.new_buffer(bytes.len() as u64, ResourceOptions::shared())).unwrap();
    let contents = buffer.contents().unwrap();
    // SAFETY: the buffer holds `bytes.len()` bytes, and no command uses them yet.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
    buffer
}

/// A new texture of the lanes' sides, of `pf` and `storage`, that shaders read.
fn texture(device: &Device, pf: PixelFormat, storage: StorageMode) -> Texture {
    let [w, h] = SIDES;
    let mut descriptor = TextureDescriptor::texture_2d(pf, w, h, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::new(
        storage,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    );
    device.new_texture(&descriptor).unwrap()
}

/// `texture`'s texels, of 4 bytes each in a buffer (`options` names a combined format's depth),
/// read back through a blit, tightly packed.
fn texels(device: &Device, texture: &Texture, options: BlitOption) -> Vec<u8> {
    let [w, h] = SIDES;
    let into = staged(device, &vec![0; (4 * w * h) as usize]);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let all = (Origin::new(0, 0, 0), Size::new(w, h, 1));
    blit.copy_from_texture_to_buffer(texture, 0, 0, all.0, all.1, &into, 0, 4 * w, 0, options)
        .unwrap();
    blit.end_encoding().unwrap();
    run(&command_buffer);
    read_buffer(&into, 0, 4 * w * h).unwrap()
}

/// The uploads `device` clamped since `before` (`WorkCounts::depth_uploads_clamped`).
fn clamped_since(device: &Device, before: u64) -> u64 {
    device.work_counts().depth_uploads_clamped - before
}

/// `replace_region` of a `Shared` `Depth32Float` texture takes no copy of a depth outside
/// `[0, 1]` into its image, on the host or by the GPU: where the device clamps, its bytes are
/// clamped on the CPU first, one upload clamped, and each texel reads back clamped (the module
/// docs). `DEPTHCLAMP replace_region` prints whether the device clamps, the uploads clamped and the
/// host copies the call made (`WorkCounts::host_texel_copies`).
#[test]
fn replace_region_takes_no_depth_outside_0_1_into_a_depth_image() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let (clamps, clamped, hosted, read) = {
        let device = &gpu_assisted_copies(&messages, cache.path());
        let clamps = clamps(device);
        let texture = texture(device, PixelFormat::Depth32Float, StorageMode::Shared);
        let [w, h] = SIDES;
        let before = device.work_counts();
        texture
            .replace_region(
                Region::new_2d(0, 0, w, h),
                0,
                0,
                &written(depth_bits),
                4 * w,
                0,
            )
            .unwrap();
        let after = device.work_counts();
        let hosted = after.host_texel_copies - before.host_texel_copies;
        let clamped = clamped_since(device, before.depth_uploads_clamped);
        let read = texels(device, &texture, BlitOption::None);
        (clamps, clamped, hosted, read)
    };
    let label = format!(
        "DEPTHCLAMP replace_region: clamps {clamps}, {clamped} uploads clamped, {hosted} host copies"
    );
    reported_no_depth_outside_0_1(&messages, &label);
    assert_eq!(clamped, u64::from(clamps), "the uploads clamped");
    let off = read_otherwise(&read, depth_bits, clamps);
    assert!(
        off.is_empty(),
        "texels read otherwise (texel, bits, read): {off:x?}"
    );
}

/// A blit from a buffer into a `Private` texture's depth takes no copy of a depth outside
/// `[0, 1]`: into a `Depth32Float` texture from a buffer at Metal's rules (the copy as it is),
/// from one at an offset of 2, off the copy unit, which goes through the plan's staging buffer
/// (each 4-byte word read at an even byte holds `0x40004000`, a depth above 1.0, wherever plain
/// Metal reads its rows), and into a `Depth32Float_Stencil8` texture's depth
/// (`DepthFromDepthStencil`). Where the device clamps, each copy's depths are clamped by a
/// dispatch first, three uploads clamped, and each texel reads back clamped (the module docs).
#[test]
fn a_blit_from_a_buffer_takes_no_depth_outside_0_1_into_a_depth_image() {
    const ABOVE: u32 = 0x4000_4000;
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let (clamps, clamped, reads) = {
        let device = &gpu_assisted_copies(&messages, cache.path());
        let clamps = clamps(device);
        let [w, h] = SIDES;
        let depths = written(depth_bits);
        let source = staged(device, &depths);
        // `0x40004000`'s bytes, `00 40` over and over: a word at any even byte is it.
        let halves: Vec<u8> = (0..4 * w * h + 4)
            .map(|i| [0x00, 0x40][i as usize % 2])
            .collect();
        let off_unit = staged(device, &halves);
        let (plain, staging, combined) = (
            texture(device, PixelFormat::Depth32Float, StorageMode::Private),
            texture(device, PixelFormat::Depth32Float, StorageMode::Private),
            texture(
                device,
                PixelFormat::Depth32FloatStencil8,
                StorageMode::Private,
            ),
        );
        let before = device.work_counts().depth_uploads_clamped;
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        let (origin, size) = (Origin::new(0, 0, 0), Size::new(w, h, 1));
        for (from, at, to, options) in [
            (&source, 0, &plain, BlitOption::None),
            (&off_unit, 2, &staging, BlitOption::None),
            (&source, 0, &combined, BlitOption::DepthFromDepthStencil),
        ] {
            blit.copy_from_buffer_to_texture(from, at, 4 * w, 0, size, to, 0, 0, origin, options)
                .unwrap();
        }
        blit.end_encoding().unwrap();
        run(&command_buffer);
        let clamped = clamped_since(device, before);
        let reads = [
            (
                "the copy as it is",
                texels(device, &plain, BlitOption::None),
            ),
            (
                "the staged copy",
                texels(device, &staging, BlitOption::None),
            ),
            (
                "the combined format's depth",
                texels(device, &combined, BlitOption::DepthFromDepthStencil),
            ),
        ];
        (clamps, clamped, reads)
    };
    let label =
        format!("DEPTHCLAMP blit from a buffer: clamps {clamps}, {clamped} uploads clamped");
    reported_no_depth_outside_0_1(&messages, &label);
    assert_eq!(clamped, 3 * u64::from(clamps), "the uploads clamped");
    for (copy, read) in reads {
        let off = match copy {
            "the staged copy" => read_otherwise(&read, |_| ABOVE, clamps),
            _ => read_otherwise(&read, depth_bits, clamps),
        };
        assert!(
            off.is_empty(),
            "{copy}: texels read otherwise (texel, bits, read): {off:x?}"
        );
    }
}

/// A copy from an `R32Float` texture into a `Depth32Float` one, which goes through the encoder's
/// scratch buffer (Vulkan copies no colour image into a depth one), takes no copy of a depth
/// outside `[0, 1]`: where the device clamps, the scratch's depths are clamped by a dispatch first,
/// one upload clamped, and each texel reads back clamped (the module docs). The colour texture
/// takes its texels from a buffer, which no depth clamp touches.
#[test]
fn a_copy_from_a_colour_texture_takes_no_depth_outside_0_1_into_a_depth_image() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let (clamps, clamped, colour, read) = {
        let device = &gpu_assisted_copies(&messages, cache.path());
        let clamps = clamps(device);
        let [w, h] = SIDES;
        let source = staged(device, &written(depth_bits));
        let colour = texture(device, PixelFormat::R32Float, StorageMode::Private);
        let depth = texture(device, PixelFormat::Depth32Float, StorageMode::Private);
        let queue = device.new_command_queue().unwrap();
        let (origin, size) = (Origin::new(0, 0, 0), Size::new(w, h, 1));
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_texture(
            &source,
            0,
            4 * w,
            0,
            size,
            &colour,
            0,
            0,
            origin,
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        run(&command_buffer);
        let before = device.work_counts().depth_uploads_clamped;
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_texture(&colour, 0, 0, origin, size, &depth, 0, 0, origin)
            .unwrap();
        blit.end_encoding().unwrap();
        run(&command_buffer);
        let clamped = clamped_since(device, before);
        let colour = texels(device, &colour, BlitOption::None);
        (
            clamps,
            clamped,
            colour,
            texels(device, &depth, BlitOption::None),
        )
    };
    let label = format!(
        "DEPTHCLAMP copy from a colour texture: clamps {clamps}, {clamped} uploads clamped"
    );
    reported_no_depth_outside_0_1(&messages, &label);
    assert_eq!(clamped, u64::from(clamps), "the uploads clamped");
    let off = read_otherwise(&colour, depth_bits, false);
    assert!(
        off.is_empty(),
        "the colour texture's texels (texel, bits, read): {off:x?}"
    );
    let off = read_otherwise(&read, depth_bits, clamps);
    assert!(
        off.is_empty(),
        "texels read otherwise (texel, bits, read): {off:x?}"
    );
}
