//! Device tests of textures over host pages in place (`HostPageTexturePolicy::InPlace`) on the
//! device the cfg selects: a texture that only samples or reads, over a no-copy buffer's pages or
//! a buffer in fresh pages, is an image aliasing the pages where the device's check of the first
//! move passed for their class, and a shadow elsewhere, counted by why.
//!
//! - [`sampled_textures_over_host_pages_take_the_path_their_check_answered`], under the
//!   validation layer: once the device's check answered (printed, `IN-PLACE-CHECK`, with its
//!   time), a 1920x1080 `BGRA8Unorm` texture of packed rows and a 64x64 `RGBA16Float` one of padded
//!   rows, over a no-copy buffer of anonymous pages and over a buffer in fresh pages: where the
//!   class passed, each is in place (`vk_image` answers one; `host_textures_in_place`,
//!   `fresh_page_textures_in_place`) or a shadow its refusal counts; where it did not, a shadow
//!   counted by the check's answer. Either way a blit reads the pages' bytes, and a CPU write made
//!   after the command buffer's commit and before it runs (held by an event the CPU signals once
//!   it wrote) is what the blit reads: the image follows the buffer's rule (Host Write Ordering
//!   Guarantees). Printed (`IN-PLACE`): each texture's path.
//! - [`writes_and_passes_beyond_the_usage_take_their_shadows`], under the validation layer: an
//!   in-place 64x64 `RGBA16Float` texture of `ShaderRead` usage alone, with padded rows and with
//!   packed ones (where the device takes either in place), written by a kernel (no `ShaderWrite`)
//!   and rendered into by a pass that clears and stores (no `RenderTarget`): each through its
//!   shadow (`undeclared-usage-shadow`), never the aliasing image,
//!   which stays the texture's (`vk_image` the same handle); every texel of the pages then holds the
//!   value, and the bytes between the rows hold the pattern.
//! - [`textures_that_render_or_write_count_their_shadows_apart`], under the validation layer: a
//!   64x64 `R32Uint` texture over a no-copy buffer of anonymous pages, of each usage that renders
//!   or writes, is a shadow counted by why: one that renders into it (`RenderTarget`, whatever else
//!   it holds) as a render target's (`host_texture_shadows_render_target`), one that writes it
//!   through a shader alone (`ShaderWrite`, `ShaderAtomic`) as a shader write's
//!   (`host_texture_shadows_shader_write`), and under `HostPageTexturePolicy::Shadow` every one as
//!   the exact form's (`host_texture_shadows_exact`).
//! - [`a_failed_check_keeps_the_shadow_and_no_record`], under the validation layer: a device made
//!   to fail its check (`DeviceOptions::fail_in_place_check`) answers failed or unchecked for every
//!   class and makes such a texture a shadow counted as the failed check's, never in place, and
//!   keeps no record: the next device on the same shader cache recalls nothing and runs the check,
//!   whose record the device after it recalls for every class (`in_place_checks_recalled`), with
//!   the same verdicts and no check run.
//! - [`in_place_and_shadow_costs`], without the validation layer, report only (built in release for
//!   the numbers): a `BGRA8Unorm` surface over a no-copy buffer of anonymous pages, read by a
//!   fragment function each frame, under each setting: a 3840x2160 one through a 3840x32 strip,
//!   and a 1920x1080 one through a 1920x32 strip, whole, and whole in each of three passes of one
//!   command buffer; the median and fastest GPU time of a frame's command buffer
//!   (`IN-PLACE-COST`). The shadow copies the whole surface in once a command buffer, into the
//!   device's own memory, where the image in place reads the pages at each pass (over the bus, on
//!   a discrete card).

use super::degrade_notices::counts;
use super::first_use_writes::CHECK_WAIT;
use super::host_memory::{counting, no_copy, not_importable, Mapping, GUEST_PAGE};
use super::no_copy_textures::{floats, staged};
use super::pitched_surfaces::reading;
use super::{layer_reported_nothing_at_all, validated_device_with, Messages, ScratchCache, WAIT};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::HostPageTexturePolicy;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, CommandQueue, Degradation, Device, DeviceOptions, EventRef,
    LoadAction, Origin, RenderPassDescriptor, RenderPipelineState, ResourceOptions, ScissorRect,
    SharedEvent, Size, StoreAction, Texture, TextureDescriptor, WorkCounts,
};

/// The bytes of a buffer of the runtime's own that lies in fresh pages (past half a pooled block).
const FRESH: u64 = 40 << 20;

/// The lane's options with textures over host pages taken in place where the check passes: the
/// lane's devices otherwise take the exact answers, under which every one is a shadow.
pub(super) fn in_place(options: &mut DeviceOptions) {
    options.policy.host_page_textures = HostPageTexturePolicy::InPlace;
}

/// Whether the device's check passed for `class` (`Pointer`, `Fresh`, `DmaBuf`).
fn passed(device: &Device, class: &str) -> bool {
    device
        .in_place_check_report()
        .iter()
        .any(|line| *line == format!("{class}: Passed"))
}

/// The device's check answered, its verdicts and time printed.
fn settled(device: &Device) {
    assert!(
        device.settle_in_place_checks(CHECK_WAIT),
        "the check answered"
    );
    for line in device.in_place_check_report() {
        eprintln!("IN-PLACE-CHECK on {}: {line}", device.name());
    }
    eprintln!(
        "IN-PLACE-CHECK on {}: took {:?}",
        device.name(),
        device.in_place_check_took()
    );
}

/// The shadows over host pages refused an image in place between `before` and `after`.
pub(super) fn refused(before: &WorkCounts, after: &WorkCounts) -> u64 {
    let of = |counts: &WorkCounts| {
        counts.host_texture_shadows_refused_rows
            + counts.host_texture_shadows_refused_detached
            + counts.host_texture_shadows_refused_image
            + counts.host_texture_shadows_refused_pitch
            + counts.host_texture_shadows_refused_placement
            + counts.host_texture_shadows_refused_source
            + counts.host_texture_shadows_refused_dma_buf
    };
    of(after) - of(before)
}

/// A texture of `format`, `w` by `h`, `ShaderRead` alone, over `buffer` with rows `bpr` apart.
pub(super) fn sampled(
    buffer: &Buffer,
    format: PixelFormat,
    (w, h, bpr): (u64, u64, u64),
) -> Texture {
    let mut d = TextureDescriptor::texture_2d(format, w, h, false);
    d.usage = TextureUsage::SHADER_READ;
    d.resource_options = buffer.resource_options();
    buffer.new_texture(&d, 0, bpr).unwrap()
}

/// The texels of `texture` (`w` by `h`, `texel` bytes each) as a blit reads them into tight rows,
/// in a command buffer held by `hold` until the CPU signals it: `write` runs once the command
/// buffer is committed, before the signal.
fn read_held(
    queue: &CommandQueue,
    texture: &Texture,
    (w, h, texel): (u64, u64, u64),
    hold: &SharedEvent,
    value: u64,
    write: impl FnOnce(),
) -> Vec<u8> {
    let device = queue.device();
    let row = w * texel;
    let out = device
        .new_buffer(row * h, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer
        .encode_wait_for_event(EventRef::from(hold), value)
        .unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(w, h, 1),
        &out,
        0,
        row,
        row * h,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    write();
    hold.set_signaled_value(value);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let contents = out.contents().unwrap();
    // SAFETY: the buffer's `row * h` bytes, which no command writes any longer.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), (row * h) as usize) }.to_vec()
}

/// `bytes`' texels of `w` by `h` (`texel` bytes each) laid out `bpr` apart, as tight rows.
fn texels_of(bytes: &[u8], (w, h, texel): (u64, u64, u64), bpr: u64) -> Vec<u8> {
    (0..h)
        .flat_map(|y| {
            let at = (y * bpr) as usize;
            bytes[at..at + (w * texel) as usize].to_vec()
        })
        .collect()
}

#[test]
fn sampled_textures_over_host_pages_take_the_path_their_check_answered() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let mut lapses = Vec::new();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), in_place).unwrap();
        settled(&device);
        let queue = device.new_command_queue().unwrap();
        let hold = device.new_shared_event().unwrap();
        let mut value = 0;
        let shapes = [
            (PixelFormat::BGRA8Unorm, 4u64, 1920u64, 1080u64, 7680u64),
            (PixelFormat::RGBA16Float, 8, 64, 64, 1024),
        ];
        for class in ["Pointer", "Fresh"] {
            for (format, texel, w, h, bpr) in shapes {
                let len = ((bpr * h) as usize + GUEST_PAGE).next_multiple_of(GUEST_PAGE);
                // The pages: anonymous ones mapped here, or the fresh pages of a long buffer.
                let mapping = (class == "Pointer").then(|| Mapping::anonymous(len));
                let buffer = match &mapping {
                    // SAFETY: the mapping outlives the buffer and the texture (dropped first).
                    Some(mapping) => match unsafe {
                        no_copy(
                            &device,
                            mapping.pointer(),
                            len,
                            ResourceOptions::default(),
                            None,
                        )
                    } {
                        Ok(buffer) => buffer,
                        Err(error) if not_importable(&error) => {
                            eprintln!("IN-PLACE: the device imports no host memory: {error:?}");
                            continue;
                        }
                        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
                    },
                    None => device.new_buffer(FRESH, ResourceOptions::shared()).unwrap(),
                };
                let contents = buffer.contents().unwrap();
                let write = |bytes: &[u8]| {
                    // SAFETY: the buffer's first `len` bytes, which no command reads meanwhile
                    // (the blit waits for the event the CPU signals after this).
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            bytes.as_ptr(),
                            contents.as_ptr(),
                            bytes.len(),
                        )
                    }
                };
                let first = counting(len, 3);
                write(&first);
                let before = device.work_counts();
                let texture = sampled(&buffer, format, (w, h, bpr));
                let after = device.work_counts();
                let aliased = after.host_textures_in_place - before.host_textures_in_place;
                let fresh =
                    after.fresh_page_textures_in_place - before.fresh_page_textures_in_place;
                let refused = refused(&before, &after);
                // SAFETY: the handle is only compared, while the texture lives.
                let image = unsafe { texture.vk_image() };
                let what = format!("{class} {format:?} {w}x{h}/{bpr}");
                let path = match (image.is_some(), aliased, refused) {
                    (true, 1, 0) => "in place",
                    (false, 0, 1) => "refused",
                    (false, 0, 0) => "shadow",
                    _ => "inconsistent",
                };
                eprintln!(
                    "IN-PLACE on {}: {what}: {path} (in place {aliased}, fresh {fresh}, refused \
                     {refused})",
                    device.name()
                );
                let checked = passed(&device, class);
                let right = match (checked, path) {
                    (true, "in place") => (class == "Fresh") == (fresh == 1),
                    (true, "refused") => true,
                    (false, "shadow") => true,
                    _ => false,
                };
                if !right {
                    lapses.push(format!("{what}: {path}, the check passed: {checked}"));
                }
                // The bytes the CPU wrote before, then those it wrote after the commit.
                value += 1;
                let read = read_held(&queue, &texture, (w, h, texel), &hold, value, || {});
                if read != texels_of(&first, (w, h, texel), bpr) {
                    lapses.push(format!("{what}: a blit read other than the pages"));
                }
                let later = counting(len, 101);
                value += 1;
                let read = read_held(&queue, &texture, (w, h, texel), &hold, value, || {
                    write(&later)
                });
                if read != texels_of(&later, (w, h, texel), bpr) {
                    lapses.push(format!(
                        "{what}: a blit read other than the CPU wrote after its commit"
                    ));
                }
                drop((texture, buffer));
                drop(mapping);
            }
        }
    }
    layer_reported_nothing_at_all(&messages);
    assert!(lapses.is_empty(), "{}", lapses.join("\n"));
}

/// The `RGBA16Float` texel the kernel writes and the pass clears to: `half4(0.25, 0.5, 0.75, 1)`.
const VALUE: [u8; 8] = [0x00, 0x34, 0x00, 0x38, 0x00, 0x3a, 0x00, 0x3c];

#[test]
fn writes_and_passes_beyond_the_usage_take_their_shadows() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        // Each form announces its shadow, not only the device's first.
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            in_place(options);
            options
                .announce_each
                .insert(Degradation::UndeclaredUsageShadow);
        })
        .unwrap();
        settled(&device);
        if !passed(&device, "Pointer") {
            eprintln!(
                "IN-PLACE-USAGE on {}: no image in place to write",
                device.name()
            );
            return;
        }
        let queue = device.new_command_queue().unwrap();
        let (w, h) = (64u64, 64u64);
        // Padded rows, then packed ones: a device whose linear images take no padded pitch takes
        // the second.
        for (form, bpr) in [
            ("kernel", 1024u64),
            ("pass", 1024),
            ("kernel", 512),
            ("pass", 512),
        ] {
            let len = ((bpr * h) as usize).next_multiple_of(GUEST_PAGE);
            let form_at = format!("{form}, rows of {bpr} bytes");
            let form = form_at.as_str();
            let mapping = Mapping::anonymous(len);
            let pattern = counting(len, 9);
            mapping.write(0, &pattern);
            // SAFETY: the mapping outlives the buffer and the texture (dropped first).
            let buffer = unsafe {
                no_copy(
                    &device,
                    mapping.pointer(),
                    len,
                    ResourceOptions::default(),
                    None,
                )
            }
            .unwrap();
            let texture = sampled(&buffer, PixelFormat::RGBA16Float, (w, h, bpr));
            // SAFETY: the handle is only compared, while the texture lives.
            let Some(image) = (unsafe { texture.vk_image() }) else {
                eprintln!(
                    "IN-PLACE-USAGE on {}: {form}: refused in place",
                    device.name()
                );
                continue;
            };
            messages.take_notices();
            let command_buffer = queue.command_buffer().unwrap();
            match form.starts_with("kernel") {
                true => {
                    let kernel = super::level_writes_by_usage::kernel(&device, "write_at_level");
                    let mut encoder = command_buffer.compute_command_encoder().unwrap();
                    encoder.set_compute_pipeline_state(&kernel).unwrap();
                    encoder.set_texture(Some(&texture), 0).unwrap();
                    encoder.set_bytes(&0u16.to_le_bytes(), 0).unwrap();
                    encoder
                        .dispatch_threadgroups(Size::new(w / 8, h / 8, 1), Size::new(8, 8, 1))
                        .unwrap();
                    encoder.end_encoding().unwrap();
                }
                false => {
                    let mut pass = RenderPassDescriptor::new();
                    let color = &mut pass.color_attachments[0];
                    color.texture = Some(texture.clone());
                    color.load_action = LoadAction::Clear;
                    color.store_action = StoreAction::Store;
                    color.clear_color = ClearColor::new(0.25, 0.5, 0.75, 1.0);
                    let encoder = command_buffer.render_command_encoder(&pass).unwrap();
                    encoder.end_encoding().unwrap();
                }
            }
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None, "{form}");
            let notices = counts(&messages.take_notices());
            assert!(
                notices.contains_key(&Degradation::UndeclaredUsageShadow),
                "{form}: {notices:?}"
            );
            // SAFETY: as above.
            assert_eq!(unsafe { texture.vk_image() }, Some(image), "{form}");
            let now = mapping.read(0..len);
            for y in 0..h {
                for x in 0..w {
                    let at = (y * bpr + x * 8) as usize;
                    assert_eq!(now[at..at + 8], VALUE, "{form}: texel {x},{y}");
                }
                let between = (y * bpr + w * 8) as usize..((y + 1) * bpr) as usize;
                assert_eq!(
                    now[between.clone()],
                    pattern[between],
                    "{form}: row {y}'s padding"
                );
            }
            eprintln!(
                "IN-PLACE-USAGE on {}: {form}: through its shadow",
                device.name()
            );
            drop((texture, buffer));
            drop(mapping);
        }
    }
    layer_reported_nothing_at_all(&messages);
}

#[test]
fn textures_that_render_or_write_count_their_shadows_apart() {
    use TextureUsage as U;
    // Each usage, and whether it counts as a render target's shadow (else a shader write's).
    let usages = [
        (U::SHADER_READ | U::RENDER_TARGET, true),
        (U::SHADER_READ | U::SHADER_WRITE, false),
        (U::SHADER_READ | U::SHADER_ATOMIC, false),
        (U::SHADER_WRITE | U::SHADER_ATOMIC, false),
        (U::SHADER_WRITE | U::RENDER_TARGET, true),
        (U::SHADER_ATOMIC | U::RENDER_TARGET, true),
        (U::UNKNOWN, true),
    ];
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    for policy in [
        HostPageTexturePolicy::InPlace,
        HostPageTexturePolicy::Shadow,
    ] {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.policy.host_page_textures = policy;
        })
        .unwrap();
        let (w, h, bpr) = (64u64, 64u64, 256u64);
        let len = ((bpr * h) as usize).next_multiple_of(GUEST_PAGE);
        let mapping = Mapping::anonymous(len);
        // SAFETY: the mapping outlives the buffer and the textures (dropped first).
        let buffer = match unsafe {
            no_copy(
                &device,
                mapping.pointer(),
                len,
                ResourceOptions::default(),
                None,
            )
        } {
            Ok(buffer) => buffer,
            Err(error) if not_importable(&error) => {
                eprintln!("IN-PLACE-SHADOWS: the device imports no host memory: {error:?}");
                return;
            }
            Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
        };
        for (usage, renders) in usages {
            let mut d = TextureDescriptor::texture_2d(PixelFormat::R32Uint, w, h, false);
            d.usage = usage;
            d.resource_options = buffer.resource_options();
            let before = device.work_counts();
            let texture = match buffer.new_texture(&d, 0, bpr) {
                Ok(texture) => texture,
                // A usage the device refuses over a buffer counts nothing.
                Err(error) => {
                    eprintln!(
                        "IN-PLACE-SHADOWS on {}: {usage:?}: refused: {error:?}",
                        device.name()
                    );
                    continue;
                }
            };
            let after = device.work_counts();
            let delta = |counts: fn(&WorkCounts) -> u64| counts(&after) - counts(&before);
            let counted = (
                delta(|c| c.host_texture_shadows_render_target),
                delta(|c| c.host_texture_shadows_shader_write),
                delta(|c| c.host_texture_shadows_exact),
            );
            let due = match (policy, renders) {
                (HostPageTexturePolicy::Shadow, _) => (0, 0, 1),
                (_, true) => (1, 0, 0),
                (_, false) => (0, 1, 0),
            };
            eprintln!(
                "IN-PLACE-SHADOWS on {}: {policy:?} {usage:?}: (render target, shader write, \
                 exact) {counted:?}",
                device.name()
            );
            assert_eq!(counted, due, "{policy:?} {usage:?}");
            // SAFETY: the handle is only read, while the texture lives.
            assert!(unsafe { texture.vk_image() }.is_none(), "{usage:?}");
            drop(texture);
        }
        messages.take_notices();
        drop((buffer, device));
        drop(mapping);
    }
    layer_reported_nothing_at_all(&messages);
}

#[test]
fn a_failed_check_keeps_the_shadow_and_no_record() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            in_place(options);
            options.fail_in_place_check = true;
        })
        .unwrap();
        settled(&device);
        let report = device.in_place_check_report();
        assert!(
            report.iter().take(3).all(|line| !line.ends_with("Passed")),
            "{report:?}"
        );
        let len = (7680 * 1080usize).next_multiple_of(GUEST_PAGE);
        let mapping = Mapping::anonymous(len);
        // SAFETY: the mapping outlives the buffer and the texture (dropped first).
        if let Ok(buffer) = unsafe {
            no_copy(
                &device,
                mapping.pointer(),
                len,
                ResourceOptions::default(),
                None,
            )
        } {
            let before = device.work_counts();
            let texture = sampled(&buffer, PixelFormat::BGRA8Unorm, (1920, 1080, 7680));
            let after = device.work_counts();
            // SAFETY: as above.
            assert!(unsafe { texture.vk_image() }.is_none());
            assert_eq!(after.host_textures_in_place, before.host_textures_in_place);
            let failed =
                after.host_texture_shadows_check_failed - before.host_texture_shadows_check_failed;
            let unchecked =
                after.host_texture_shadows_unchecked - before.host_texture_shadows_unchecked;
            assert_eq!(failed + unchecked, 1, "{report:?}");
            let counted = after.in_place_checks_failed + after.in_place_checks_unchecked;
            assert_eq!(after.in_place_checks_run, counted);
            drop((texture, buffer));
        }
        drop(mapping);
        messages.take_notices();
    }
    // Nothing kept: the next device on the cache recalls nothing and checks.
    let verdicts = {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), in_place).unwrap();
        assert_eq!(device.work_counts().in_place_checks_recalled, 0);
        settled(&device);
        assert!(device.in_place_check_took().is_some());
        device.in_place_check_report()
    };
    // Its record is recalled whole by the device after it, which checks nothing.
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), in_place).unwrap();
        let counts = device.work_counts();
        assert_eq!(counts.in_place_checks_recalled, 3);
        assert_eq!(counts.in_place_checks_run, 0);
        assert!(device.settle_in_place_checks(std::time::Duration::ZERO));
        assert_eq!(device.in_place_check_took(), None);
        assert_eq!(device.in_place_check_report(), verdicts);
        eprintln!(
            "IN-PLACE-RECORD on {}: recalled {verdicts:?}",
            device.name()
        );
    }
    messages.take_notices();
    layer_reported_nothing_at_all(&messages);
}

#[test]
fn in_place_and_shadow_costs() {
    // A surface's size and pitch, the rows each pass reads from its top, and the passes a command
    // buffer reads it in.
    let cases = [
        ((3840u64, 2160u64, 15360u64), 32u64, 1usize),
        ((1920, 1080, 7680), 32, 1),
        ((1920, 1080, 7680), 1080, 1),
        ((1920, 1080, 7680), 1080, 3),
    ];
    for policy in [
        HostPageTexturePolicy::InPlace,
        HostPageTexturePolicy::Shadow,
    ] {
        let cache = ScratchCache::new().unwrap();
        let messages = Messages::default();
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.layers.clear();
            options.policy.host_page_textures = policy;
        })
        .unwrap();
        if policy == HostPageTexturePolicy::InPlace {
            settled(&device);
        }
        let queue = device.new_command_queue().unwrap();
        let reads = reading(&device, PixelFormat::BGRA8Unorm, "tob_read");
        let corners = staged(&device, &floats([-1.0, -1.0, 1.0, 1.0]));
        for (surface, rows, passes) in cases {
            if !surface_costs(
                &device,
                &queue,
                (&reads, &corners),
                policy,
                surface,
                rows,
                passes,
            ) {
                return;
            }
        }
    }
}

/// [`in_place_and_shadow_costs`]' case of a `BGRA8Unorm` surface of `(w, h, bpr)` over a no-copy
/// buffer of anonymous pages on `device`: each frame the CPU writes a row of the pages, and one
/// command buffer renders `passes` passes, each reading the surface's top `rows` rows through a
/// fragment function's `read` into a private target of its size. Printed (`IN-PLACE-COST`): the
/// form, and the median and fastest GPU time of a frame's command buffer. False where the device
/// imports no host memory.
fn surface_costs(
    device: &Device,
    queue: &CommandQueue,
    (reads, corners): (&RenderPipelineState, &Buffer),
    policy: HostPageTexturePolicy,
    (w, h, bpr): (u64, u64, u64),
    rows: u64,
    passes: usize,
) -> bool {
    const FRAMES: usize = 40;
    let len = ((bpr * h) as usize).next_multiple_of(GUEST_PAGE);
    let mapping = Mapping::anonymous(len);
    mapping.write(0, &counting(len, 5));
    // SAFETY: the mapping outlives the buffer and the texture (dropped first).
    let buffer = match unsafe {
        no_copy(
            device,
            mapping.pointer(),
            len,
            ResourceOptions::default(),
            None,
        )
    } {
        Ok(buffer) => buffer,
        Err(error) => {
            eprintln!("IN-PLACE-COST: the device imports no host memory: {error:?}");
            return false;
        }
    };
    let texture = sampled(&buffer, PixelFormat::BGRA8Unorm, (w, h, bpr));
    // SAFETY: the handle is only compared.
    let form = match unsafe { texture.vk_image() } {
        Some(_) => "in place",
        None => "shadow",
    };
    let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, w, h, false);
    d.usage = TextureUsage::RENDER_TARGET;
    d.resource_options = ResourceOptions::private();
    let target = device.new_texture(&d).unwrap();
    let mut times = Vec::with_capacity(FRAMES);
    for frame in 0..FRAMES {
        // The CPU writes a row each frame, as a guest's surface changes.
        mapping.write(((frame as u64 % h) * bpr) as usize, &[frame as u8; 64]);
        let command_buffer = queue.command_buffer().unwrap();
        for _ in 0..passes {
            let mut pass = RenderPassDescriptor::new();
            let color = &mut pass.color_attachments[0];
            color.texture = Some(target.clone());
            color.load_action = LoadAction::DontCare;
            color.store_action = StoreAction::Store;
            let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
            encoder.set_render_pipeline_state(reads).unwrap();
            encoder
                .set_scissor_rect(ScissorRect::new(0, 0, w, rows))
                .unwrap();
            encoder.set_vertex_buffer(Some(corners), 0, 0).unwrap();
            encoder.set_fragment_texture(Some(&texture), 0).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                .unwrap();
            encoder.end_encoding().unwrap();
        }
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let gpu = command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap();
        times.push(gpu);
    }
    times.sort_by(f64::total_cmp);
    eprintln!(
        "IN-PLACE-COST on {}: {policy:?} ({form}): a frame's GPU time median {:.1} us, fastest \
         {:.1} us, over {FRAMES} frames of {passes} pass(es) each reading the top {rows} rows of \
         a {w}x{h} surface",
        device.name(),
        times[FRAMES / 2] * 1e6,
        times[0] * 1e6,
    );
    drop((texture, buffer, target));
    drop(mapping);
    true
}
