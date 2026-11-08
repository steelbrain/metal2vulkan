//! Device tests of device copies on the device the cfg selects: on a discrete GPU, a `Managed`
//! texture that would read host pages in place, over a no-copy buffer whose memory carries a log
//! of its writes (`HostMemory::with_dirty_pages`), is a shadow in the device's memory whose copies
//! in take only the rows the log reports changed (the runtime's `dirty_pages.rs`): the GPU's copy
//! of the two a `Managed` resource is on a Mac with a discrete GPU. A `Shared` texture there, any
//! texture on any other device, and one without a log, is placed as before.
//!
//! - [`a_sampled_surface_over_logged_pages_follows_the_guest_s_writes`], under the validation
//!   layer: a 1920x1080 `BGRA8Unorm` `Managed` surface over logged pages, read whole each frame by
//!   a fragment function into a private target, which then holds the pages' texels after each
//!   frame's CPU write of a run of a row. On a discrete GPU it is a device copy
//!   (`device_copy_textures`, no `vk_image`): its first copy in moves every row, each later one
//!   only the rows the log reported (`dirty_copies_in_gated`, `dirty_copy_in_bytes_copied`), one
//!   with no change nothing (`dirty_copies_in_skipped`), and one whose wait the log answers
//!   `false` every row (`dirty_copies_in_whole`). Elsewhere none is made.
//! - [`a_gpu_write_of_the_pages_reaches_the_device_copy`], under the validation layer: a blit's
//!   fill of rows of the surface's buffer in one command buffer is what the next one's read finds
//!   (on a discrete GPU a whole copy in: the fill let go of the copy's hold).
//! - [`without_a_log_the_surface_is_placed_as_before`], under the validation layer: the same
//!   surface over pages without a log is no device copy.
//! - [`a_shared_surface_over_logged_pages_is_no_device_copy`], under the validation layer: a
//!   `Shared` surface over logged pages is no device copy on any device, in place where the
//!   device's check passed for pointer pages (on a discrete GPU, read across the bus as a
//!   discrete Mac's GPU reads a `Shared` texture), its reads the pages' texels.
//! - [`device_copy_costs`], without the validation layer, report only (built in release for the
//!   numbers): a 3840x2160 `BGRA8Unorm` `Managed` surface read whole each frame, with a few CPU-written
//!   runs a frame, in one pass and in three, over pages without a log (in place where the check
//!   passed) and with one (a device copy on a discrete GPU): the median and fastest GPU time of a
//!   frame's command buffer (`DEVICE-COPY-COST`).

use super::first_use_writes::CHECK_WAIT;
use super::host_memory::{counting, Mapping, GUEST_PAGE};
use super::in_place_textures::{in_place, passed, sampled};
use super::no_copy_textures::{floats, staged};
use super::pitched_surfaces::reading;
use super::{layer_reported_nothing_at_all, validated_device_with, Messages, ScratchCache, WAIT};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandQueue, CpuCacheMode, Device, DeviceKind, DirtyPages, DirtyStamp,
    HazardTrackingMode, HostMemory, LoadAction, Origin, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, ScissorRect, Size, StorageMode, StoreAction, Texture, TextureDescriptor,
    WorkCounts,
};
use std::ops::Range;
use std::sync::{Arc, Mutex};

/// The log's page.
const PAGE: u64 = 4096;

/// A log of the writes to a mapping's pages, as a consumer keeps one: each write stamped with the
/// publication it falls in, and each stamp the runtime asks for a new publication.
#[derive(Default)]
struct Log {
    state: Mutex<LogState>,
}

#[derive(Default)]
struct LogState {
    /// The publication writes fall in now.
    now: u64,
    /// Each write's pages and stamp.
    writes: Vec<(Range<u64>, u64)>,
    /// Whether the next wait cannot say.
    cannot_say: bool,
}

impl Log {
    /// Note the CPU's write of the bytes `range`, whole pages.
    fn mark(&self, range: Range<u64>) {
        let mut state = self.state.lock().unwrap();
        let pages = range.start / PAGE * PAGE..range.end.div_ceil(PAGE) * PAGE;
        let now = state.now;
        state.writes.push((pages, now));
    }

    /// Make the next wait answer `false`.
    fn cannot_say(&self) {
        self.state.lock().unwrap().cannot_say = true;
    }
}

impl DirtyPages for Log {
    type Watch = ();

    fn watch(&self, _: Range<u64>) {}

    fn request(&self) -> DirtyStamp {
        let mut state = self.state.lock().unwrap();
        let stamp = state.now;
        state.now += 1;
        DirtyStamp(stamp)
    }

    fn wait(&self, _: DirtyStamp) -> bool {
        let mut state = self.state.lock().unwrap();
        !std::mem::take(&mut state.cannot_say)
    }

    fn changed(&self, since: DirtyStamp, out: &mut Vec<Range<u64>>) {
        let state = self.state.lock().unwrap();
        let later = state.writes.iter().filter(|(_, stamp)| *stamp > since.0);
        out.extend(later.map(|(pages, _)| pages.clone()));
    }

    fn wrote(&self, range: Range<u64>) {
        let mut state = self.state.lock().unwrap();
        let now = state.now;
        state.writes.push((range, now));
    }
}

/// `Managed` storage, the other modes at their defaults: the storage whose textures a discrete GPU
/// makes device copies.
fn managed() -> ResourceOptions {
    ResourceOptions::new(
        StorageMode::Managed,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    )
}

/// A no-copy `Managed` buffer over `mapping`'s `len` bytes on `device`, its memory carrying `log`
/// where given.
fn over(device: &Device, mapping: &Mapping, len: usize, log: Option<&Arc<Log>>) -> Buffer {
    over_as(device, mapping, len, log, managed())
}

/// [`over`], of `options`.
fn over_as(
    device: &Device,
    mapping: &Mapping,
    len: usize,
    log: Option<&Arc<Log>>,
    options: ResourceOptions,
) -> Buffer {
    // SAFETY: the mapping outlives the buffer (dropped first by every caller), and stays
    // writable.
    let memory = unsafe { HostMemory::new(mapping.pointer()) };
    let memory = match log {
        Some(log) => memory.with_dirty_pages(Arc::clone(log)),
        None => memory,
    };
    // SAFETY: as above.
    unsafe { device.new_buffer_with_bytes_no_copy(memory, len as u64, options, None) }.unwrap()
}

/// One frame's command buffer on `queue`: `passes` passes, each reading the top `rows` rows of
/// `texture` into `target` through `reads`; its GPU time.
fn frame(
    queue: &CommandQueue,
    (reads, corners): (&RenderPipelineState, &Buffer),
    (texture, target): (&Texture, &Texture),
    (w, rows): (u64, u64),
    passes: usize,
) -> f64 {
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
        encoder.set_fragment_texture(Some(texture), 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
        encoder.end_encoding().unwrap();
    }
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap()
}

/// `target`'s `w` by `h` texels of 4 bytes, as a blit reads them into tight rows.
fn texels(queue: &CommandQueue, target: &Texture, (w, h): (u64, u64)) -> Vec<u8> {
    let row = w * 4;
    let out = (queue.device())
        .new_buffer(row * h, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let size = Size::new(w, h, 1);
    let origin = Origin::new(0, 0, 0);
    blit.copy_from_texture_to_buffer(
        target,
        0,
        0,
        origin,
        size,
        &out,
        0,
        row,
        row * h,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let contents = out.contents().unwrap();
    // SAFETY: the buffer's `row * h` bytes, which no command writes any longer.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), (row * h) as usize) }.to_vec()
}

/// The texels of `w` by `h` (4 bytes each) of `mapping`, laid out `bpr` apart, as tight rows.
fn guest_texels(mapping: &Mapping, (w, h, bpr): (u64, u64, u64)) -> Vec<u8> {
    let bytes = mapping.read(0..(bpr * h) as usize);
    (0..h)
        .flat_map(|y| {
            let at = (y * bpr) as usize;
            bytes[at..at + (w * 4) as usize].to_vec()
        })
        .collect()
}

/// A private `BGRA8Unorm` render target of `w` by `h` on `device`.
fn target_of(device: &Device, (w, h): (u64, u64)) -> Texture {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, w, h, false);
    d.usage = TextureUsage::RENDER_TARGET;
    d.resource_options = ResourceOptions::private();
    device.new_texture(&d).unwrap()
}

/// Whether `device` makes device copies: a discrete GPU.
fn copies_on(device: &Device) -> bool {
    device.properties().kind == DeviceKind::Discrete
}

/// The changes of the device copies' counters from `before` to `after`: textures made, copies in
/// gated, of every row, skipped, and the bytes they moved.
fn copied(before: &WorkCounts, after: &WorkCounts) -> [u64; 5] {
    let of = |c: &WorkCounts| {
        [
            c.device_copy_textures,
            c.dirty_copies_in_gated,
            c.dirty_copies_in_whole,
            c.dirty_copies_in_skipped,
            c.dirty_copy_in_bytes_copied,
        ]
    };
    let (a, b) = (of(before), of(after));
    std::array::from_fn(|i| b[i] - a[i])
}

#[test]
fn a_sampled_surface_over_logged_pages_follows_the_guest_s_writes() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), in_place).unwrap();
    let queue = device.new_command_queue().unwrap();
    let reads = reading(&device, PixelFormat::BGRA8Unorm, "tob_read");
    let corners = staged(&device, &floats([-1.0, -1.0, 1.0, 1.0]));
    let (w, h, bpr) = (1920u64, 1080u64, 7680u64);
    let len = ((bpr * h) as usize).next_multiple_of(GUEST_PAGE);
    let mapping = Mapping::anonymous(len);
    mapping.write(0, &counting(len, 3));
    let log = Arc::new(Log::default());
    let before = device.work_counts();
    let buffer = over(&device, &mapping, len, Some(&log));
    let texture = sampled(&buffer, PixelFormat::BGRA8Unorm, (w, h, bpr));
    let made = copied(&before, &device.work_counts());
    let discrete = copies_on(&device);
    // SAFETY: the handle is only compared.
    let in_place = unsafe { texture.vk_image() }.is_some();
    eprintln!(
        "DEVICE-COPY on {} ({:?}): device copy {}, in place {in_place}",
        device.name(),
        device.properties().kind,
        made[0] == 1
    );
    assert_eq!(
        made[0],
        u64::from(discrete),
        "a device copy exactly on a discrete GPU"
    );
    if discrete {
        assert!(!in_place, "a device copy reads no image in place");
    }
    let target = target_of(&device, (w, h));
    let objects = (&reads, &corners);
    let check = |what: &str| {
        let read = texels(&queue, &target, (w, h));
        let guest = guest_texels(&mapping, (w, h, bpr));
        assert!(read == guest, "{what}: the target holds the guest's texels");
    };
    // The first frame copies every row.
    let start = device.work_counts();
    frame(&queue, objects, (&texture, &target), (w, h), 1);
    check("the first frame");
    // Each later frame's CPU write of a run of one row, logged.
    for (at, row) in [(3u64, 17u64), (1, 500), (7, 1079)] {
        let offset = row * bpr + at * 64;
        mapping.write(offset as usize, &[0xa5 ^ row as u8; 64]);
        log.mark(offset..offset + 64);
        frame(&queue, objects, (&texture, &target), (w, h), 1);
        check(&format!("after a write of row {row}"));
    }
    // A frame with no write copies nothing; one whose wait cannot say copies every row.
    frame(&queue, objects, (&texture, &target), (w, h), 1);
    check("after no write");
    let offset = 900 * bpr;
    mapping.write(offset as usize, &[0x3c; 128]);
    log.mark(offset..offset + 128);
    log.cannot_say();
    frame(&queue, objects, (&texture, &target), (w, h), 1);
    check("after a wait that could not say");
    let counted = copied(&start, &device.work_counts());
    eprintln!(
        "DEVICE-COPY on {}: gated {}, whole {}, skipped {}, bytes copied {}",
        device.name(),
        counted[1],
        counted[2],
        counted[3],
        counted[4]
    );
    if discrete {
        assert_eq!(counted[1], 6, "each frame's copy in, gated");
        assert_eq!(
            counted[2], 2,
            "the first and the one that could not say: every row"
        );
        assert_eq!(counted[3], 1, "no write: nothing");
        // Every row twice; a 4 KiB page of a 7680-byte row touches at most two rows, three times.
        let row = w * 4;
        assert!(
            counted[4] <= 2 * h * row + 3 * 2 * row,
            "bytes {}",
            counted[4]
        );
    }
    drop((texture, buffer, target));
    drop(mapping);
    layer_reported_nothing_at_all(&messages);
}

#[test]
fn a_gpu_write_of_the_pages_reaches_the_device_copy() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), in_place).unwrap();
    let queue = device.new_command_queue().unwrap();
    let reads = reading(&device, PixelFormat::BGRA8Unorm, "tob_read");
    let corners = staged(&device, &floats([-1.0, -1.0, 1.0, 1.0]));
    let (w, h, bpr) = (256u64, 256u64, 1024u64);
    let len = ((bpr * h) as usize).next_multiple_of(GUEST_PAGE);
    let mapping = Mapping::anonymous(len);
    mapping.write(0, &counting(len, 11));
    let log = Arc::new(Log::default());
    let buffer = over(&device, &mapping, len, Some(&log));
    let texture = sampled(&buffer, PixelFormat::BGRA8Unorm, (w, h, bpr));
    let target = target_of(&device, (w, h));
    let objects = (&reads, &corners);
    frame(&queue, objects, (&texture, &target), (w, h), 1);
    let start = device.work_counts();
    // A blit fills rows 40 to 47 of the buffer: the GPU's write, no CPU write logged.
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(&buffer, 40 * bpr..48 * bpr, 0x5a).unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    frame(&queue, objects, (&texture, &target), (w, h), 1);
    let read = texels(&queue, &target, (w, h));
    assert!(
        read == guest_texels(&mapping, (w, h, bpr)),
        "the fill reached the read"
    );
    assert!(read[(40 * w * 4) as usize..(48 * w * 4) as usize]
        .iter()
        .all(|&byte| byte == 0x5a));
    let counted = copied(&start, &device.work_counts());
    eprintln!(
        "DEVICE-COPY on {}: after the fill gated {}, whole {}, bytes {}",
        device.name(),
        counted[1],
        counted[2],
        counted[4]
    );
    if copies_on(&device) {
        // The runtime reports its own write of the pages to their log (`dirty_pages.rs`), so the
        // next copy in is gated to the pages the fill wrote: no whole copy, and at least the
        // fill's rows but no more than the pages they lie in.
        assert_eq!(
            counted[1..3],
            [1, 0],
            "the fill's pages, gated: no whole copy"
        );
        let fill = 8 * bpr;
        assert!(
            (fill..=fill + 2 * GUEST_PAGE as u64).contains(&counted[4]),
            "the fill's pages alone: bytes {}",
            counted[4]
        );
    }
    drop((texture, buffer, target));
    drop(mapping);
    layer_reported_nothing_at_all(&messages);
}

#[test]
fn without_a_log_the_surface_is_placed_as_before() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), in_place).unwrap();
    let (w, h, bpr) = (256u64, 256u64, 1024u64);
    let len = ((bpr * h) as usize).next_multiple_of(GUEST_PAGE);
    let mapping = Mapping::anonymous(len);
    let before = device.work_counts();
    let buffer = over(&device, &mapping, len, None);
    let texture = sampled(&buffer, PixelFormat::BGRA8Unorm, (w, h, bpr));
    let after = device.work_counts();
    assert_eq!(
        copied(&before, &after)[0],
        0,
        "no device copy without a log"
    );
    assert_eq!(
        after.host_textures_in_place - before.host_textures_in_place + after.shadowed_textures
            - before.shadowed_textures,
        1,
        "in place or a shadow, as before"
    );
    drop((texture, buffer));
    drop(mapping);
    layer_reported_nothing_at_all(&messages);
}

#[test]
fn a_shared_surface_over_logged_pages_is_no_device_copy() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), in_place).unwrap();
    let queue = device.new_command_queue().unwrap();
    let reads = reading(&device, PixelFormat::BGRA8Unorm, "tob_read");
    let corners = staged(&device, &floats([-1.0, -1.0, 1.0, 1.0]));
    let (w, h, bpr) = (256u64, 256u64, 1024u64);
    let len = ((bpr * h) as usize).next_multiple_of(GUEST_PAGE);
    // A texture over host pages starts the device's in-place check, which the one measured then
    // follows.
    {
        let mapping = Mapping::anonymous(GUEST_PAGE);
        let buffer = over_as(
            &device,
            &mapping,
            GUEST_PAGE,
            None,
            ResourceOptions::shared(),
        );
        drop(sampled(&buffer, PixelFormat::BGRA8Unorm, (16, 16, 64)));
        assert!(
            device.settle_in_place_checks(CHECK_WAIT),
            "the check answered"
        );
    }
    let mapping = Mapping::anonymous(len);
    mapping.write(0, &counting(len, 7));
    let log = Arc::new(Log::default());
    let before = device.work_counts();
    let buffer = over_as(
        &device,
        &mapping,
        len,
        Some(&log),
        ResourceOptions::shared(),
    );
    let texture = sampled(&buffer, PixelFormat::BGRA8Unorm, (w, h, bpr));
    let after = device.work_counts();
    // SAFETY: the handle is only compared.
    let in_place = unsafe { texture.vk_image() }.is_some();
    let pointer = passed(&device, "Pointer", &buffer, (w * 4, h, bpr));
    eprintln!(
        "DEVICE-COPY on {} ({:?}): a Shared surface over logged pages: device copy {}, in place \
         {in_place}, in places {}, the check passed for pointer pages {pointer}",
        device.name(),
        device.properties().kind,
        copied(&before, &after)[0],
        after.host_textures_in_place - before.host_textures_in_place,
    );
    assert_eq!(
        copied(&before, &after)[0],
        0,
        "no device copy of a Shared texture"
    );
    assert_eq!(in_place, pointer, "in place where the check passed");
    let target = target_of(&device, (w, h));
    let objects = (&reads, &corners);
    frame(&queue, objects, (&texture, &target), (w, h), 1);
    let read = texels(&queue, &target, (w, h));
    assert!(
        read == guest_texels(&mapping, (w, h, bpr)),
        "the pages' texels"
    );
    // A logged CPU write of a run of row 9 reaches the next frame.
    let offset = 9 * bpr + 64;
    mapping.write(offset as usize, &[0xc3; 64]);
    log.mark(offset..offset + 64);
    frame(&queue, objects, (&texture, &target), (w, h), 1);
    let read = texels(&queue, &target, (w, h));
    assert!(
        read == guest_texels(&mapping, (w, h, bpr)),
        "the write's texels"
    );
    drop((texture, buffer, target));
    drop(mapping);
    layer_reported_nothing_at_all(&messages);
}

#[test]
fn device_copy_costs() {
    const FRAMES: usize = 40;
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        in_place(options);
        options.layers.clear();
    })
    .unwrap();
    let queue = device.new_command_queue().unwrap();
    let reads = reading(&device, PixelFormat::BGRA8Unorm, "tob_read");
    let corners = staged(&device, &floats([-1.0, -1.0, 1.0, 1.0]));
    let (w, h, bpr) = (3840u64, 2160u64, 15360u64);
    let len = ((bpr * h) as usize).next_multiple_of(GUEST_PAGE);
    let target = target_of(&device, (w, h));
    // A texture over host pages starts the device's in-place check, which the texture without a
    // log then follows.
    {
        let mapping = Mapping::anonymous(GUEST_PAGE);
        let buffer = over(&device, &mapping, GUEST_PAGE, None);
        drop(sampled(&buffer, PixelFormat::BGRA8Unorm, (16, 16, 64)));
        let settled = device.settle_in_place_checks(CHECK_WAIT);
        eprintln!(
            "DEVICE-COPY-COST on {}: the in-place check answered: {settled}",
            device.name()
        );
    }
    for logged in [false, true] {
        for passes in [1usize, 3] {
            let mapping = Mapping::anonymous(len);
            mapping.write(0, &counting(len, 5));
            let log = Arc::new(Log::default());
            let buffer = over(&device, &mapping, len, logged.then_some(&log));
            let texture = sampled(&buffer, PixelFormat::BGRA8Unorm, (w, h, bpr));
            // SAFETY: the handle is only compared.
            let form = match (unsafe { texture.vk_image() }, logged && copies_on(&device)) {
                (_, true) => "device copy",
                (Some(_), false) => "in place",
                (None, false) => "shadow",
            };
            let start = device.work_counts();
            let mut times = Vec::with_capacity(FRAMES);
            for f in 0..FRAMES as u64 {
                // A few runs a frame, as a guest's cursor and a clock change.
                for run in 0..4u64 {
                    let offset = ((f * 37 + run * 541) % h) * bpr + run * 1024;
                    mapping.write(offset as usize, &[f as u8; 256]);
                    log.mark(offset..offset + 256);
                }
                let objects = (&reads, &corners);
                times.push(frame(&queue, objects, (&texture, &target), (w, h), passes));
            }
            let counted = copied(&start, &device.work_counts());
            times.sort_by(f64::total_cmp);
            eprintln!(
                "DEVICE-COPY-COST on {}: {form}: a frame's GPU time median {:.1} us, fastest \
                 {:.1} us, over {FRAMES} frames of {passes} pass(es) each reading a {w}x{h} \
                 surface whole, 4 runs written a frame; copied in {} bytes ({} a frame)",
                device.name(),
                times[FRAMES / 2] * 1e6,
                times[0] * 1e6,
                counted[4],
                counted[4] / FRAMES as u64,
            );
            drop((texture, buffer));
            drop(mapping);
        }
    }
}
