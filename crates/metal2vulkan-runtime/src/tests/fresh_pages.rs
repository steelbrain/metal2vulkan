//! Long new buffers in fresh pages (`fresh_pages.rs`), on the device the GPU cfg selects.
//!
//! - **A long new buffer reads zero, nothing written.** A `Shared` buffer, a write-combined one
//!   and a `Managed` one (with unified memory), each longer than half a pooled block: the `Shared`
//!   one lies in fresh pages wherever the device imports host pointers and its check found a held
//!   import adds nothing to a submission's cost (`submit_check.rs`; MoltenVK, ANV, lavapipe), and
//!   never where it found it does (RADV) (`WorkCounts::fresh_page_buffers_made`), with the host
//!   setting none of its bytes to zero
//!   (`new_buffer_bytes_cleared` unchanged); a class whose best memory type for the buffer takes
//!   no import lies where it would have, the host clearing every byte of it
//!   (a discrete card's write-combined class, measured below). Either way the buffer reads zero
//!   through `contents` and through a blit out of it, and a buffer that is neither imported nor
//!   cleared fails. A shorter one, a `Private` one and a heap's are made as before: a pooled or
//!   device placement, cleared.
//!
//!   Measured (2026-10-24, RADV, Mesa 26.2.4): on the Radeon 780M, whose memory is unified, all
//!   three classes take the pages; on the RX 7900 XTX the `Shared` and `Managed` classes do and
//!   the write-combined one does not, its best memory type for the buffer being a device-local
//!   host-visible one that a host-pointer import cannot bind, so that buffer is cleared on the
//!   host (33,558,544 bytes).
//! - **Right after a filled one of its length.** A buffer filled by the GPU and released, then one
//!   of the same length: it takes the parked one (`buffers_recycled`), whose bytes the host sets
//!   to zero (`new_buffer_bytes_cleared` grows by its padded length), and it reads zero. Past the
//!   recycler's bytes (`recycle::BUFFERS`) nothing is parked: the next buffer lies in fresh pages
//!   again and reads zero, nothing written.
//! - **A copy of bytes.** `new_buffer_with_bytes` of a long pattern lies in fresh pages and reads
//!   the pattern, through `contents` and a blit.
//! - **A texture over a long buffer** that renders, its rows 512 bytes apart (twice its own) or 256
//!   (its own), is a shadow of its bytes under `HostPageTexturePolicy::Shadow`
//!   (`WorkCounts::shadows_over_fresh_pages`), and under `InPlace` an image in place only where the
//!   device's check passed for render targets over fresh pages. Where the check passed it is still
//!   a shadow where the device makes no image in place of its plan or at its pitch, its refusal
//!   counted (`host_texture_shadows_refused_pitch` or `_image`): MoltenVK lays a linear image's
//!   rows at a pitch of its own and enables no `VK_EXT_image_drm_format_modifier`, so on the M3 Max
//!   the texture at 512 bytes a row is a shadow and the one at its own pitch an image in place.
//!   Every shadow is announced (`shadowed-texture`) with its one cause counted, and an image in
//!   place neither. Either way a blit out of it reads the bytes the CPU wrote, and a render pass's
//!   clear through it reaches them once the command buffer completed, the bytes between its rows
//!   kept.
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`).

use super::blit_holds::{Scratch, WAIT};
use super::draw_again::Heard;
use crate::fresh_pages::FRESH_PAGES_PAST;
use crate::{
    BlitOption, Buffer, ClearColor, CommandBufferStatus, CommandQueue, CpuCacheMode, Degradation,
    Device, DeviceKind, DeviceOptions, HazardTrackingMode, HeapDescriptor, HeapType, LoadAction,
    MessageSink, Origin, RenderPassDescriptor, ResourceOptions, ShaderCacheOptions, Size,
    StorageMode, StoreAction, TextureDescriptor, WorkCounts,
};
use metal2vulkan::device::HostPageTexturePolicy;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::path::Path;

/// A long buffer's length: past half a pooled block, and not a multiple of 16, so its padding is
/// read too.
const LONG: u64 = FRESH_PAGES_PAST + 4100;

/// The device the cfg selects, its shader cache in `cache`, its submission checks answered.
fn device(cache: &Path) -> Device {
    device_with(cache, HostPageTexturePolicy::InPlace, &Heard::default())
}

/// [`device`], its textures over host pages held as `policy` says, the notices it announces in
/// `heard`.
fn device_with(cache: &Path, policy: HostPageTexturePolicy, heard: &Heard) -> Device {
    let heard = std::sync::Arc::clone(heard);
    let mut options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        messages: Some(MessageSink::new(move |message| {
            if let Some(notice) = &message.degradation {
                heard.lock().unwrap().push(notice.clone());
            }
        })),
        ..DeviceOptions::default()
    };
    options.policy.host_page_textures = policy;
    let device = Device::system_default_with(&options).unwrap();
    assert!(
        device.settle_submit_checks(crate::tests::SETTLE),
        "the submission checks never answered"
    );
    device
}

/// Whether the device takes fresh pages: it imports host pointers, and its check found a held
/// import adds nothing to a submission's cost.
fn takes_fresh_pages(device: &Device) -> bool {
    device.shared().host_import.is_ok() && device.fresh_pages_pay()
}

/// Fill `buffer` with `value` on the GPU, waited for.
fn fill(queue: &CommandQueue, buffer: &Buffer, value: u8) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(buffer, 0..buffer.length(), value).unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// The bytes of `buffer`, blitted into `out` (`Shared`, at least as long, its bytes set to 0xcd
/// first so that the copy shows), waited for.
fn read_on_gpu(queue: &CommandQueue, buffer: &Buffer, out: &Buffer) -> Vec<u8> {
    let length = buffer.length();
    let contents = out.contents().unwrap();
    // SAFETY: the shared buffer's bytes, which no command uses now.
    unsafe { std::ptr::write_bytes(contents.as_ptr(), 0xcd, length as usize) };
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(buffer, 0, out, 0, length)
        .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
    // SAFETY: as above; the copy completed.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), length as usize) }.to_vec()
}

/// The bytes of a host-visible `buffer` through its `contents`.
fn read_on_cpu(buffer: &Buffer) -> Vec<u8> {
    let contents = buffer.contents().unwrap();
    // SAFETY: the buffer's bytes, which no command uses now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }.to_vec()
}

/// Whether every byte of `bytes` is zero, and else where the first that is not lies.
fn first_nonzero(bytes: &[u8]) -> Option<usize> {
    bytes.iter().position(|&b| b != 0)
}

/// Wait until the device parked `count` buffers in all.
fn buffers_parked_at_least(device: &Device, count: u64) {
    let until = std::time::Instant::now() + WAIT;
    while device.work_counts().object_churn.buffers_parked < count {
        assert!(std::time::Instant::now() < until, "the buffers were parked");
        std::thread::yield_now();
    }
}

/// The counts that tell where a new buffer's zeros came from: buffers made in fresh pages, and
/// bytes the host cleared.
fn zeros(counts: &WorkCounts) -> (u64, u64) {
    (
        counts.fresh_page_buffers_made,
        counts.new_buffer_bytes_cleared,
    )
}

#[test]
fn a_long_new_buffer_reads_zero_with_nothing_written() {
    let scratch = Scratch::new("fresh-pages-new");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let fresh = takes_fresh_pages(&device);
    let out = device.new_buffer(LONG, ResourceOptions::shared()).unwrap();
    let write_combined = ResourceOptions::new(
        StorageMode::Shared,
        CpuCacheMode::WriteCombined,
        HazardTrackingMode::Default,
    );
    let managed = ResourceOptions::new(
        StorageMode::Managed,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    );
    let mut legs = vec![
        ("shared", ResourceOptions::shared()),
        ("write-combined", write_combined),
    ];
    if device.properties().has_unified_memory {
        legs.push(("managed", managed));
    }
    // Each kept, so that none is parked for the next.
    let mut kept = Vec::new();
    for (what, options) in legs {
        let before = zeros(&device.work_counts());
        let buffer = device.new_buffer(LONG, options).unwrap();
        let after = zeros(&device.work_counts());
        let padded = LONG.next_multiple_of(16);
        let moved = (after.0 - before.0, after.1 - before.1);
        // The `Shared` class is host memory on every device that imports host pointers, so its
        // long buffer takes the pages wherever the device takes them at all. A class whose best
        // memory type for the buffer is not among those an import may bind lies where it would
        // have (`fresh_pages::Device::fresh_page_buffer`), which is a memory-type fact of the
        // device and not a contract: the buffer is cleared on the host instead, and either way
        // the buffer reads zero, which is what Metal's `newBuffer` promises.
        if what == "shared" {
            assert_eq!(
                buffer.in_fresh_pages(),
                fresh,
                "{what}: a long Shared buffer takes the pages wherever the device imports them"
            );
        }
        match buffer.in_fresh_pages() {
            true => assert_eq!(
                moved,
                (1, 0),
                "{what}: the buffer lies in fresh pages, no byte of it written"
            ),
            false => assert_eq!(
                moved,
                (0, padded),
                "{what}: the buffer lies where it would have: the host clears every byte"
            ),
        }
        assert!(
            fresh || !buffer.in_fresh_pages(),
            "{what}: a device that imports no host pointer takes no fresh pages"
        );
        assert_eq!(
            first_nonzero(&read_on_cpu(&buffer)),
            None,
            "{what}: contents"
        );
        assert_eq!(
            first_nonzero(&read_on_gpu(&queue, &buffer, &out)),
            None,
            "{what}: a blit out of it"
        );
        kept.push(buffer);
    }

    // Made as before: a shorter one, a `Private` one, and a heap's own buffer.
    let before = zeros(&device.work_counts());
    let short = device
        .new_buffer(FRESH_PAGES_PAST, ResourceOptions::shared())
        .unwrap();
    let after = zeros(&device.work_counts());
    assert_eq!(
        (after.0 - before.0, after.1 - before.1),
        (0, FRESH_PAGES_PAST),
        "half a block lies in a pooled block, cleared"
    );
    assert!(!short.in_fresh_pages());
    assert_eq!(first_nonzero(&read_on_cpu(&short)), None);
    let before = zeros(&device.work_counts());
    let private = device.new_buffer(LONG, ResourceOptions::private()).unwrap();
    assert_eq!(zeros(&device.work_counts()), before, "a Private buffer");
    assert!(!private.in_fresh_pages());
    assert_eq!(first_nonzero(&read_on_gpu(&queue, &private, &out)), None);
    let descriptor = HeapDescriptor::new(HeapType::Automatic, LONG, ResourceOptions::shared());
    let before = zeros(&device.work_counts());
    let heap = device.new_heap(&descriptor).unwrap();
    assert_eq!(
        zeros(&device.work_counts()).0,
        before.0,
        "a heap's own buffer lies where its textures can"
    );
    let placed = heap
        .new_buffer(FRESH_PAGES_PAST + 256, ResourceOptions::shared())
        .unwrap();
    assert!(!placed.in_fresh_pages());
    assert_eq!(
        first_nonzero(&read_on_cpu(&placed)),
        None,
        "a new heap's buffer"
    );
}

#[test]
fn a_long_buffer_made_right_after_a_filled_one_reads_zero() {
    let scratch = Scratch::new("fresh-pages-after");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let fresh = takes_fresh_pages(&device);
    // Past the recycler's bytes: never parked.
    let past = 80 << 20;
    let out = device.new_buffer(past, ResourceOptions::shared()).unwrap();
    for length in [LONG, past] {
        let first = device
            .new_buffer(length, ResourceOptions::shared())
            .unwrap();
        fill(&queue, &first, 0xab);
        assert!(read_on_cpu(&first).iter().all(|&b| b == 0xab), "the fill");
        let before = device.work_counts();
        drop(first);
        let parks = length <= 64 << 20;
        if parks {
            buffers_parked_at_least(&device, before.object_churn.buffers_parked + 1);
        }
        let second = device
            .new_buffer(length, ResourceOptions::shared())
            .unwrap();
        let after = device.work_counts();
        let recycled = after.object_churn.buffers_recycled - before.object_churn.buffers_recycled;
        let made = after.fresh_page_buffers_made - before.fresh_page_buffers_made;
        let cleared = after.new_buffer_bytes_cleared - before.new_buffer_bytes_cleared;
        let padded = length.next_multiple_of(16);
        match (parks, fresh) {
            // The parked buffer, its last owner's bytes set to zero on the host.
            (true, _) => assert_eq!((recycled, made, cleared), (1, 0, padded), "{length}"),
            (false, true) => assert_eq!((recycled, made, cleared), (0, 1, 0), "{length}"),
            (false, false) => assert_eq!((recycled, made, cleared), (0, 0, padded), "{length}"),
        }
        assert_eq!(
            first_nonzero(&read_on_cpu(&second)),
            None,
            "{length}: contents read zero, not the old bytes"
        );
        assert_eq!(
            first_nonzero(&read_on_gpu(&queue, &second, &out)),
            None,
            "{length}: a blit reads zero, not the old bytes"
        );
    }
}

#[test]
fn a_long_buffer_with_bytes_reads_them() {
    let scratch = Scratch::new("fresh-pages-bytes");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let bytes: Vec<u8> = (0..LONG).map(|i| (i % 251) as u8).collect();
    let before = device.work_counts().fresh_page_buffers_made;
    let buffer = device
        .new_buffer_with_bytes(&bytes, ResourceOptions::shared())
        .unwrap();
    let made = device.work_counts().fresh_page_buffers_made - before;
    assert_eq!(made, u64::from(takes_fresh_pages(&device)));
    assert!(read_on_cpu(&buffer) == bytes, "contents read the bytes");
    let out = device.new_buffer(LONG, ResourceOptions::shared()).unwrap();
    assert!(
        read_on_gpu(&queue, &buffer, &out) == bytes,
        "a blit reads the bytes"
    );
}

#[test]
fn a_texture_over_a_long_buffer_is_its_bytes_through_a_shadow() {
    let policy = HostPageTexturePolicy::Shadow;
    a_texture_over_a_long_buffer_is_its_bytes("fresh-pages-texture", policy, 512);
}

#[test]
fn a_texture_over_a_long_buffer_is_its_bytes_in_place_where_the_check_passed() {
    let policy = HostPageTexturePolicy::InPlace;
    a_texture_over_a_long_buffer_is_its_bytes("fresh-pages-texture-in-place", policy, 512);
}

#[test]
fn a_texture_at_its_own_pitch_over_a_long_buffer_is_its_bytes_in_place_where_the_check_passed() {
    let policy = HostPageTexturePolicy::InPlace;
    a_texture_over_a_long_buffer_is_its_bytes("fresh-pages-texture-own-pitch", policy, 256);
}

/// A 64 by 64 `RGBA8Unorm` render target over a long buffer, its rows `pitch` bytes apart, on a
/// device of `policy`, its scratch named `name` (the module docs).
fn a_texture_over_a_long_buffer_is_its_bytes(
    name: &str,
    policy: HostPageTexturePolicy,
    pitch: u64,
) {
    const SIDE: u64 = 64;
    let scratch = Scratch::new(name);
    let heard = Heard::default();
    let device = device_with(&scratch.0, policy, &heard);
    // The check's verdicts first: a texture made before its verdict is a shadow (`in_place.rs`).
    device.settle_in_place_checks(WAIT);
    let queue = device.new_command_queue().unwrap();
    let buffer = device.new_buffer(LONG, ResourceOptions::shared()).unwrap();
    let contents = buffer.contents().unwrap();
    let span = (pitch * SIDE) as usize;
    let pattern: Vec<u8> = (0..span).map(|i| (i % 253) as u8).collect();
    // SAFETY: the buffer's first bytes, which no command uses now.
    unsafe { std::ptr::copy_nonoverlapping(pattern.as_ptr(), contents.as_ptr(), span) };
    let shadow_notices = || {
        let heard = heard.lock().unwrap();
        let shadows = heard.iter().filter(|notice| {
            notice.degradation == Degradation::ShadowedTexture
                && notice.object.contains(&format!("{SIDE}x{SIDE}"))
                && notice.object.contains(&format!("rows of {pitch} bytes"))
        });
        shadows.count() as u64
    };
    let before = device.work_counts();
    let announced_before = shadow_notices();
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    descriptor.resource_options = buffer.resource_options();
    let texture = buffer.new_texture(&descriptor, 0, pitch).unwrap();
    let after = device.work_counts();
    let announced = shadow_notices() - announced_before;
    let over_fresh = after.shadows_over_fresh_pages - before.shadows_over_fresh_pages;
    // SAFETY: the handle is only compared, while the texture lives.
    let in_place = unsafe { texture.vk_image() }.is_some();
    let renders_in_place = device
        .in_place_check_report()
        .iter()
        .any(|line| line == "Fresh renders: Passed");
    // Each cause a shadow over host pages is counted under, by how many it counted now.
    let moved = |count: fn(&WorkCounts) -> u64| count(&after) - count(&before);
    let causes: [(&str, u64); 12] = [
        ("exact", moved(|c| c.host_texture_shadows_exact)),
        ("store width", moved(|c| c.host_texture_shadows_store_width)),
        (
            "check pending",
            moved(|c| c.host_texture_shadows_check_pending),
        ),
        (
            "check failed",
            moved(|c| c.host_texture_shadows_check_failed),
        ),
        ("unchecked", moved(|c| c.host_texture_shadows_unchecked)),
        ("rows", moved(|c| c.host_texture_shadows_refused_rows)),
        (
            "detached",
            moved(|c| c.host_texture_shadows_refused_detached),
        ),
        ("image", moved(|c| c.host_texture_shadows_refused_image)),
        ("pitch", moved(|c| c.host_texture_shadows_refused_pitch)),
        (
            "placement",
            moved(|c| c.host_texture_shadows_refused_placement),
        ),
        ("source", moved(|c| c.host_texture_shadows_refused_source)),
        ("dma-buf", moved(|c| c.host_texture_shadows_refused_dma_buf)),
    ];
    let counted: Vec<&str> = (causes.iter())
        .filter(|(_, n)| *n > 0)
        .map(|(cause, _)| *cause)
        .collect();
    eprintln!(
        "FRESH-PAGES on {}: {policy:?}: in fresh pages {}, in place {in_place}, the check passed \
         for render targets there {renders_in_place}, shadow causes counted {counted:?}, \
         shadowed-texture notices {announced}",
        device.name(),
        buffer.in_fresh_pages()
    );
    if buffer.in_fresh_pages() {
        let shadow = !in_place;
        assert_eq!(over_fresh, u64::from(shadow));
        assert_eq!(
            after.shadowed_textures - before.shadowed_textures,
            u64::from(shadow)
        );
        assert_eq!(
            after.host_textures_in_place - before.host_textures_in_place,
            u64::from(in_place)
        );
        // A shadow is announced, with its one cause counted; an image in place is neither.
        assert_eq!(announced, u64::from(shadow), "a shadow is announced");
        let one_cause: &[&str] = match (policy, in_place, renders_in_place) {
            (HostPageTexturePolicy::Shadow, ..) => {
                assert!(shadow, "a shadow under Shadow");
                &["exact"]
            }
            (_, true, check) => {
                assert!(
                    check,
                    "in place only where the check passed for render targets"
                );
                &[]
            }
            // Where the check passed, the device made no image in place of the texture's plan,
            // or none at its pitch.
            (_, false, true) => &["pitch", "image"],
            (_, false, false) => &["check failed", "unchecked", "check pending"],
        };
        match one_cause {
            [] => assert!(counted.is_empty(), "no shadow cause in place: {counted:?}"),
            _ => assert!(
                counted.len() == 1
                    && one_cause.contains(&counted[0])
                    && causes.iter().all(|(_, n)| *n <= 1),
                "the shadow's one cause, one of {one_cause:?}, counted once: {counted:?}"
            ),
        }
    } else {
        assert_eq!(over_fresh, 0);
    }

    // A blit out of the texture reads the bytes the CPU wrote, row by row.
    let out = device
        .new_buffer(SIDE * SIDE * 4, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        &texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(SIDE, SIDE, 1),
        &out,
        0,
        SIDE * 4,
        SIDE * SIDE * 4,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
    let read = read_on_cpu(&out);
    for y in 0..SIDE as usize {
        let row = &read[y * SIDE as usize * 4..(y + 1) * SIDE as usize * 4];
        let wrote = &pattern[y * pitch as usize..][..SIDE as usize * 4];
        assert_eq!(
            row, wrote,
            "row {y} of the texture reads the buffer's bytes"
        );
    }

    // A clear through the texture reaches the buffer's bytes, and only its texels'.
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(texture.clone());
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    color.clear_color = ClearColor::new(1.0, 0.0, 1.0, 0.0);
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer
        .render_command_encoder(&pass)
        .unwrap()
        .end_encoding()
        .unwrap();
    command_buffer.commit().unwrap();
    assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let now = read_on_cpu(&buffer);
    for y in 0..SIDE as usize {
        let row = &now[y * pitch as usize..(y + 1) * pitch as usize];
        let texels = &row[..SIDE as usize * 4];
        assert!(
            texels.chunks(4).all(|texel| texel == [255, 0, 255, 0]),
            "row {y}: the clear reached the buffer"
        );
        assert_eq!(
            &row[SIDE as usize * 4..],
            &pattern[y * pitch as usize + SIDE as usize * 4..(y + 1) * pitch as usize],
            "row {y}: the bytes between the rows kept the CPU's"
        );
    }
}
