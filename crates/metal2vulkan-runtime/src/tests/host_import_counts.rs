//! The counts of no-copy buffers' host imports that a release while idle would weigh
//! (`WorkCounts::host_imports_used`, `WorkCounts::host_imports_read_apart`): the imports each
//! committed command buffer's commands use, each counted once per command buffer however many
//! buffers lie in it, and the live imports a read other than a blit's took, each also counted
//! once for each reason it was read apart for (`host_imports_read_apart_*`). Blits read none
//! apart: a copy with a texture, a two-plane texture's or a texture over a buffer's, included.
//! Each new import is either kept imported for its cause (`host_imports_kept_*`) or releasable.
//! Each import by pointer counts with its bytes (`host_pointer_imports_made`), and a reported
//! remap counts the live imports it touches (`host_remaps_over_imports`, `host_imports_remapped`).
//! A rebind's import counts as made, as covered where bytes a buffer lay over already held its
//! pages (`host_rebind_imports_made`, `host_rebind_imports_covered`), and as unlisted until a
//! command buffer that uses it commits (`host_imports_rebound_unlisted`).
//!
//! A test that makes a texture over a buffer's host pages first settles the device's check of the
//! first move in place (`in_place_check.rs`), where its policy takes such textures in place
//! (lavapipe): the first such texture starts the check on a thread of its own, whose own no-copy
//! buffers and textures over them are live imports, read apart for a texture, while it runs, and
//! so would move the gauges under the test's reads.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. A device that imports no host memory
//! says so and passes.

use super::blit_holds::{device, over_host, until, Host, Scratch, BYTES};
use crate::{
    BlitOption, Buffer, CommandBufferStatus, CommandQueue, Device, Error, Origin, ResourceOptions,
    Size, Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::ptr::NonNull;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

/// [`device`], its check of the first move in place settled where it runs one, and the imports
/// the check read apart gone before the test reads a gauge (the module docs): a device made just
/// now reads none apart of its own.
fn settled_device(cache: &std::path::Path) -> Device {
    let device = device(cache);
    let settled = device.settle_in_place_checks(std::time::Duration::from_secs(120));
    println!(
        "IMPORT-COUNTS: the in-place check settled: {settled}, {} classes checked",
        device.work_counts().in_place_checks_run
    );
    until("the in-place check's imports went", || {
        device.work_counts().host_imports_read_apart == 0
    });
    device
}

/// The two counts and the live imports.
fn counts(device: &Device) -> (u64, u64, u64) {
    let counts = device.work_counts();
    (
        counts.host_imports_used,
        counts.host_imports_read_apart,
        counts.host_imports_live,
    )
}

/// Commit what `encode` records into one blit encoder, and wait for it.
fn blits(queue: &CommandQueue, encode: impl FnOnce(&mut crate::BlitCommandEncoder)) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    encode(&mut blit);
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed().unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

#[test]
fn a_command_buffer_counts_each_import_it_uses_once_and_blits_never_read_apart() {
    let scratch = Scratch::new("import-counts");
    let (first, second, third) = (Host::new(), Host::new(), Host::new());
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let runs = Arc::new(AtomicUsize::new(0));
    let (used, apart, before) = counts(&device);
    let Some(a) = over_host(&device, &first, &runs) else {
        return;
    };
    let b = over_host(&device, &second, &runs).unwrap();
    let c = over_host(&device, &third, &runs).unwrap();
    // Made, three imports, none read apart.
    let live = before + 3;
    assert_eq!(counts(&device), (used, apart, live));

    // Blits alone read none apart: three imports used once each.
    blits(&queue, |blit| {
        blit.fill_buffer(&a, 0..4096, 1).unwrap();
        blit.copy_from_buffer_to_buffer(&b, 0, &c, 0, 4096).unwrap();
    });
    assert_eq!(counts(&device), (used + 3, apart, live));

    // A buffer over exactly `a`'s bytes lies in its import, which a second buffer reads apart; a
    // command buffer using both counts the import once.
    let d = over_host(&device, &first, &runs).unwrap();
    assert_eq!(counts(&device), (used + 3, apart + 1, live));
    blits(&queue, |blit| {
        blit.fill_buffer(&a, 0..4096, 2).unwrap();
        blit.fill_buffer(&d, 4096..8192, 3).unwrap();
    });
    assert_eq!(counts(&device), (used + 4, apart + 1, live));

    // Its Metal address handed out reads `b`'s import apart, once however often.
    let _ = b.gpu_address();
    let _ = b.gpu_address();
    assert_eq!(counts(&device), (used + 4, apart + 2, live));

    // The imports that go leave the gauge.
    drop((a, b, c, d));
    until("the imports went", || {
        let (_, now_apart, now_live) = counts(&device);
        (now_apart, now_live) == (apart, before)
    });
}

/// A commit's count of the imports its buffers lie in, past the few it tells apart in place
/// (`Buffer::distinct_host_imports`) as within them: each import once, however many buffers lie in
/// it and in whatever order they come.
#[test]
fn a_commit_counts_each_import_once_past_the_few_it_tells_apart_in_place() {
    let scratch = Scratch::new("import-counts-many");
    let hosts: Vec<Host> = (0..18).map(|_| Host::new()).collect();
    let device = device(&scratch.0);
    let runs = Arc::new(AtomicUsize::new(0));
    let Some(first) = over_host(&device, &hosts[0], &runs) else {
        return;
    };
    let mut buffers = vec![first];
    buffers.extend((hosts[1..].iter()).map(|host| over_host(&device, host, &runs).unwrap()));
    // A second buffer over exactly the first's bytes lies in its import.
    let again = over_host(&device, &hosts[0], &runs).unwrap();
    for imports in [1, 15, 16, 17, 18] {
        let mut used: Vec<Buffer> = buffers[..imports].to_vec();
        used.push(again.clone());
        used.extend(buffers[..imports].iter().rev().cloned());
        assert_eq!(
            Buffer::distinct_host_imports(&used),
            imports as u64,
            "{imports} imports, each of {} buffers lying in one",
            used.len()
        );
    }
}

/// The reason gauges: the Metal address, the `VkBuffer`, a binding, a texture over the bytes, a
/// second buffer, no releasing thread.
fn reasons(counts: &WorkCounts) -> [u64; 6] {
    [
        counts.host_imports_read_apart_address,
        counts.host_imports_read_apart_vk_buffer,
        counts.host_imports_read_apart_binding,
        counts.host_imports_read_apart_texture,
        counts.host_imports_read_apart_second_buffer,
        counts.host_imports_read_apart_unreleased,
    ]
}

/// The imports kept imported for a cause, and the releasable ones holding their memory.
fn kept_or_resident(counts: &WorkCounts) -> u64 {
    counts.host_imports_kept_of_heap
        + counts.host_imports_kept_rebound
        + counts.host_imports_kept_pointer
        + counts.host_imports_kept_no_stand_in
        + counts.host_imports_kept_for_fds
        + counts.host_imports_resident
}

/// `base` with `by` added at `reason`.
fn plus(base: [u64; 6], reason: usize, by: u64) -> [u64; 6] {
    let mut out = base;
    out[reason] += by;
    out
}

#[test]
fn each_read_apart_counts_for_its_reason_once_and_each_import_for_its_keeping() {
    let scratch = Scratch::new("import-reasons");
    let hosts = [Host::new(), Host::new(), Host::new(), Host::new()];
    let device = settled_device(&scratch.0);
    let runs = Arc::new(AtomicUsize::new(0));
    let before = device.work_counts();
    let Some(a) = over_host(&device, &hosts[0], &runs) else {
        return;
    };
    let b = over_host(&device, &hosts[1], &runs).unwrap();
    let c = over_host(&device, &hosts[2], &runs).unwrap();
    let d = over_host(&device, &hosts[3], &runs).unwrap();
    // Each new import is kept for its cause, or releasable, holding its memory.
    let made = device.work_counts();
    assert_eq!(kept_or_resident(&made), kept_or_resident(&before) + 4);
    let base = reasons(&made);
    let apart = made.host_imports_read_apart;

    // The Metal address handed out, once however often.
    let _ = a.gpu_address();
    let _ = a.gpu_address();
    assert_eq!(reasons(&device.work_counts()), plus(base, 0, 1));
    // The `VkBuffer` handed out.
    // SAFETY: only read, while the buffer lives.
    let _ = unsafe { b.vk_buffer() };
    let now = plus(plus(base, 0, 1), 1, 1);
    assert_eq!(reasons(&device.work_counts()), now);
    // A texture over the bytes.
    let texture = c
        .new_texture(&over_bytes(ResourceOptions::default()), 0, ROW)
        .unwrap();
    let now = plus(now, 3, 1);
    assert_eq!(reasons(&device.work_counts()), now);
    // A second buffer over `d`'s bytes lies in its import.
    let e = over_host(&device, &hosts[3], &runs).unwrap();
    let now = plus(now, 4, 1);
    assert_eq!(reasons(&device.work_counts()), now);
    // Four imports read apart, each once whatever its reasons; `a`'s again for a second reason.
    let _ = unsafe { a.vk_buffer() };
    let counts = device.work_counts();
    assert_eq!(reasons(&counts), plus(now, 1, 1));
    assert_eq!(counts.host_imports_read_apart, apart + 4);

    // The imports that go leave every gauge.
    drop((texture, a, b, c, d, e));
    until("the imports went", || {
        let counts = device.work_counts();
        reasons(&counts) == reasons(&before)
            && counts.host_imports_read_apart == before.host_imports_read_apart
            && kept_or_resident(&counts) <= kept_or_resident(&before)
    });
}

/// The rows of a texture of [`BYTES`] of 4-byte texels, 1024 wide.
const ROW: u64 = 4096;

/// A texture of [`BYTES`] of `RGBA8Unorm` texels, [`ROW`] bytes a row, of `options`, read by
/// shaders: over a buffer's bytes, or the consumer's paged texture where `Private`.
fn over_bytes(options: ResourceOptions) -> TextureDescriptor {
    let height = BYTES as u64 / ROW;
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 1024, height, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = options;
    descriptor
}

/// Copy the no-copy `buffer` into `texture` and back in one command buffer each, laid out as
/// `row` bytes a row: the imports used once each time, none read apart.
fn copies_read_none_apart(device: &Device, buffer: &Buffer, texture: &Texture, row: u64) {
    let queue = device.new_command_queue().unwrap();
    let (width, height) = (texture.width(), texture.height());
    let (size, origin) = (Size::new(width, height, 1), Origin::new(0, 0, 0));
    let image = row * height;
    let (used, apart, live) = counts(device);
    blits(&queue, |blit| {
        (blit.copy_from_buffer_to_texture(
            buffer,
            0,
            row,
            image,
            size,
            texture,
            0,
            0,
            origin,
            BlitOption::None,
        ))
        .unwrap();
    });
    assert_eq!(counts(device), (used + 1, apart, live));
    blits(&queue, |blit| {
        (blit.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            origin,
            size,
            buffer,
            0,
            row,
            image,
            BlitOption::None,
        ))
        .unwrap();
    });
    assert_eq!(counts(device), (used + 2, apart, live));
}

#[test]
fn copies_between_a_buffer_and_a_texture_never_read_apart() {
    let scratch = Scratch::new("import-texture-copies");
    let host = Host::new();
    let device = settled_device(&scratch.0);
    let runs = Arc::new(AtomicUsize::new(0));
    let Some(buffer) = over_host(&device, &host, &runs) else {
        return;
    };
    // The consumer's paged textures, paged up out of the buffer and down into it.
    let texture = device
        .new_texture(&over_bytes(ResourceOptions::private()))
        .unwrap();
    copies_read_none_apart(&device, &buffer, &texture, ROW);
    // A texture over another buffer's bytes: its rows copied buffer to buffer.
    let bytes = device
        .new_buffer(BYTES as u64, ResourceOptions::default())
        .unwrap();
    let texture = bytes
        .new_texture(&over_bytes(ResourceOptions::default()), 0, ROW)
        .unwrap();
    copies_read_none_apart(&device, &buffer, &texture, ROW);
    // A two-plane extended-range texture: plane 0's words moved by a dispatch.
    let mut descriptor = over_bytes(ResourceOptions::private());
    descriptor.pixel_format = PixelFormat::RGB10A82PXR10;
    descriptor.width = 512;
    descriptor.height = 512;
    match device.new_texture(&descriptor) {
        Ok(texture) => copies_read_none_apart(&device, &buffer, &texture, 4 * 512),
        Err(error @ (Error::Unsupported(_) | Error::NotImplemented(_))) => {
            eprintln!("IMPORT-COUNTS: no two-plane texture: {error}");
        }
        Err(error) => panic!("a two-plane texture: {error}"),
    }
}

/// The remap reports, those over live imports, the imports they touched, and the imports made by
/// pointer with their bytes.
fn remaps(device: &Device) -> [u64; 5] {
    let counts = device.work_counts();
    [
        counts.host_remaps_reported,
        counts.host_remaps_over_imports,
        counts.host_imports_remapped,
        counts.host_pointer_imports_made,
        counts.host_pointer_import_bytes_made,
    ]
}

/// The first byte of `host`'s pages.
fn first_byte(host: &Host) -> NonNull<u8> {
    NonNull::from(host.words()).cast()
}

#[test]
fn a_remap_report_counts_the_live_imports_it_touches_and_imports_count_their_bytes() {
    let scratch = Scratch::new("import-remaps");
    let (imported, fresh) = (Host::new(), Host::new());
    let device = device(&scratch.0);
    let runs = Arc::new(AtomicUsize::new(0));
    let [reported, over, remapped, made, bytes] = remaps(&device);
    let Some(buffer) = over_host(&device, &imported, &runs) else {
        return;
    };
    // One import by pointer of the buffer's whole pages.
    let now = remaps(&device);
    assert_eq!(now[3..], [made + 1, bytes + BYTES as u64]);
    // A report over a page inside the buffer's bytes touches its import.
    let inside = NonNull::new(first_byte(&imported).as_ptr().wrapping_add(8192)).unwrap();
    device.host_range_remapped(inside, 4096);
    assert_eq!(remaps(&device)[..3], [reported + 1, over + 1, remapped + 1]);
    // One over a range no buffer lies over touches none.
    device.host_range_remapped(first_byte(&fresh), BYTES as u64);
    assert_eq!(remaps(&device)[..3], [reported + 2, over + 1, remapped + 1]);
    // A second buffer over the remapped bytes imports them anew (`host_aliases.rs`): two live
    // imports over the same bytes, one placed range, which a report counts once.
    let again = over_host(&device, &imported, &runs).unwrap();
    assert_eq!(remaps(&device)[3], made + 2);
    device.host_range_remapped(inside, 4096);
    assert_eq!(remaps(&device)[..3], [reported + 3, over + 2, remapped + 2]);
    drop((buffer, again));
    // Gone, its pages touch nothing.
    device.host_range_remapped(inside, 4096);
    assert_eq!(remaps(&device)[..3], [reported + 4, over + 2, remapped + 2]);
}

/// The imports rebinds made, those an import alive then covered, and the live rebinds' imports no
/// live buffers' table listed yet.
fn rebinds(device: &Device) -> [u64; 3] {
    let counts = device.work_counts();
    [
        counts.host_rebind_imports_made,
        counts.host_rebind_imports_covered,
        counts.host_imports_rebound_unlisted,
    ]
}

/// `base` with `by` added at each place.
fn added(base: [u64; 3], by: [u64; 3]) -> [u64; 3] {
    [base[0] + by[0], base[1] + by[1], base[2] + by[2]]
}

#[test]
fn a_rebind_s_import_is_unlisted_until_a_command_buffer_uses_it_and_covered_over_bytes_lain_over() {
    let scratch = Scratch::new("import-rebinds");
    let hosts = [Host::new(), Host::new(), Host::new()];
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let runs = Arc::new(AtomicUsize::new(0));
    let Some(first) = over_host(&device, &hosts[0], &runs) else {
        return;
    };
    let second = over_host(&device, &hosts[1], &runs).unwrap();
    let before = rebinds(&device);
    // A rebind onto bytes no buffer lies over: a new import, uncovered and unlisted.
    // SAFETY: the pages outlive the device, and only the buffers' commands use them.
    unsafe { first.rebind_bytes_no_copy(hosts[2].memory(), None, None) }.unwrap();
    assert_eq!(rebinds(&device), added(before, [1, 0, 1]));
    // A second buffer rebound onto the same bytes imports them again (only bytes a live buffer
    // was made over are shared, `host_aliases.rs`): the first one's import covers them.
    // SAFETY: as above.
    unsafe { second.rebind_bytes_no_copy(hosts[2].memory(), None, None) }.unwrap();
    assert_eq!(rebinds(&device), added(before, [2, 1, 2]));
    // A command buffer using the first lists its import from its commit on.
    blits(&queue, |blit| blit.fill_buffer(&first, 0..4096, 1).unwrap());
    assert_eq!(rebinds(&device), added(before, [2, 1, 1]));
    // Rebound again onto the bytes it lies over: covered by its own import, the new one
    // unlisted, the old one listed no more but counted as listed.
    // SAFETY: as above.
    unsafe { first.rebind_bytes_no_copy(hosts[2].memory(), None, None) }.unwrap();
    assert_eq!(rebinds(&device), added(before, [3, 2, 2]));
    // The imports that go leave the gauge.
    drop((first, second));
    until("the rebinds' imports went", || {
        rebinds(&device)[2] == before[2]
    });
}
