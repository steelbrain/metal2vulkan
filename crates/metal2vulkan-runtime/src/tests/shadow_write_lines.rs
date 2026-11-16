//! A shadow's copy back into host pages imported as dma-bufs keeps the bytes the CPU writes
//! beside its texels in their lines of memory while it runs (`shadow_splice.rs`). On the Iris Xe
//! (ANV) a copy of texels into a buffer's rows writes back the 64-byte line that holds each row's
//! last texels whole, putting back older values over the CPU's writes to the row's padding there
//! (docs/FAST-PATHS.md, **Known gaps**): the control below, a blit into the rows, loses some on
//! every run there; a shadow's copy back, spliced, loses none.
//!
//! The shape is a guest's render target: a 34x34 `RGBA16Float` texture that reads, writes and
//! renders, rows 512 bytes apart from the first byte of a no-copy buffer of 32768 bytes over a
//! `memfd`'s pages, imported as dma-bufs (the device made without `VK_EXT_external_memory_host`),
//! a shadow by the device's policy (`HostPageTexturePolicy::Shadow`): over the buffer
//! (`Buffer::new_texture`) and as an IOSurface plane over it
//! (`Buffer::new_texture_with_iosurface_plane`). Each row's texels end 16 bytes into a 64-byte
//! line, whose other 48 bytes are the row's padding.
//!
//! Each kind of round runs [`ROUNDS`] command buffers, one after another, each waited for, while a
//! thread of the test writes the padding bytes in the lines that hold the rows' last texels
//! (`hot`), sweep after sweep, one byte a line each sweep, and every 16th sweep a line of each
//! row's padding that holds no texel (`cold`) and the lines just past the image and at the
//! buffer's end (`past`), reading each byte before it writes it: a byte that no longer holds the
//! thread's last write was put back. The kinds:
//!
//! - `copy`, the control: a blit of a `Private` texture of the shape into the buffer's rows (the
//!   check's copies rounds' operation), whose losses show the device writes such lines back.
//! - `clear`: a pass that clears the shadowed target and stores it (it replaces every texel: a
//!   copy back), which must lose none, its copies back spliced.
//! - `draw`: a pass that loads it, draws a rectangle over its right half (its last column among
//!   it) in a color of the round's and stores it (a keep and a merge), its losses printed.
//! - `idle`: the thread alone, which must find none.
//!
//! The device's policy runs no check of its own, so its verdicts stay unanswered and every copy
//! back over the pages is spliced. A second test gives such a device the verdicts the device's
//! check measured under the default policy, and finds a copy back spliced exactly where they
//! splice: copied back as before where both the copies and the shader writes passed.

use super::blit_holds::{options, WAIT};
use super::encoder_holds::{bytes_of, library};
use super::host_residency::{host_page, udmabuf_openable, Ram, Scratch};
use super::merge_scratch::blit;
use super::no_pointer_import::lists_dma_buf_route;
use crate::in_place::{Access, PagesClass, RowEnds};
use crate::{
    BlitOption, Buffer, ClearColor, CommandQueue, Device, HostMemory, IoSurfacePlane, LoadAction,
    Origin, PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size,
    StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::device::{HostPageTexturePolicy, VkExtension};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const TEST: &str = "SHADOW-WRITE-LINES";
const WIDTH: u64 = 34;
const HEIGHT: u64 = 34;
const ROW_BYTES: u64 = 8 * WIDTH;
const ROW: u64 = 512;
const LENGTH: u64 = 32768;
/// The device's line of memory the shape is laid against (the Iris Xe's `nonCoherentAtomSize`).
const LINE: u64 = 64;
/// The command buffers of each kind of round.
const ROUNDS: usize = 300;
/// How long the second test waits for the device's check.
const CHECK_WAIT: Duration = Duration::from_secs(120);

/// The device made without `VK_EXT_external_memory_host`, so that a no-copy buffer is imported as
/// dma-bufs of its pages, every texture over host pages a shadow.
fn device(cache: &Path) -> Device {
    let mut options = options(cache);
    options.withheld_extensions = [VkExtension::ExtExternalMemoryHost].into_iter().collect();
    options.policy.host_page_textures = HostPageTexturePolicy::Shadow;
    let device = Device::system_default_with(&options).unwrap();
    assert!(
        device.settle_submit_checks(crate::tests::SETTLE),
        "the submission checks never answered"
    );
    device
}

/// The state that draws a rectangle (`tob_rect`, corners from vertex buffer 0) of the color in
/// fragment buffer 0 (`tob_solid`) into an `RGBA16Float` color.
fn drawing(device: &Device) -> RenderPipelineState {
    let library = library(device, "engine/tob_renders.ll");
    let function = |name: &str| {
        let function = library.new_function_with_name(name).unwrap();
        StageSource::Function(function.function_descriptor())
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("tob_rect"));
    descriptor.fragment = Some(function("tob_solid"));
    descriptor.color_attachments[0] =
        Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float));
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// What a round does (the module docs).
#[derive(Clone, Copy, Debug)]
enum Kind {
    Copy,
    Clear,
    Draw,
}

/// Round `i` of `kind`: over `target` (the shadowed texture) or, for a copy, from `source` into
/// `buffer`'s rows; committed and waited for.
fn round(
    queue: &CommandQueue,
    kind: Kind,
    (target, source, buffer): (&Texture, &Texture, &Buffer),
    state: &RenderPipelineState,
    i: usize,
) {
    let value = (i % 7) as f64 / 6.0;
    match kind {
        Kind::Copy => blit(queue, |encoder| {
            (encoder.copy_from_texture_to_buffer(
                source,
                0,
                0,
                Origin::new(0, 0, 0),
                Size::new(WIDTH, HEIGHT, 1),
                buffer,
                0,
                ROW,
                ROW * HEIGHT,
                BlitOption::None,
            ))
            .unwrap()
        }),
        Kind::Clear | Kind::Draw => {
            let command_buffer = queue.command_buffer().unwrap();
            let mut descriptor = RenderPassDescriptor::new();
            let color = &mut descriptor.color_attachments[0];
            color.texture = Some(target.clone());
            color.store_action = StoreAction::Store;
            match kind {
                Kind::Clear => {
                    color.load_action = LoadAction::Clear;
                    color.clear_color = ClearColor::new(value, 1.0 - value, 0.5, 1.0);
                }
                _ => color.load_action = LoadAction::Load,
            }
            let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
            if let Kind::Draw = kind {
                encoder.set_render_pipeline_state(state).unwrap();
                let corners: [f32; 4] = [0.0, -1.0, 1.0, 1.0];
                let rgba: [f32; 4] = [value as f32, 0.25, 1.0 - value as f32, 1.0];
                (encoder.set_vertex_bytes(&bytes_of(&corners, f32::to_le_bytes), 0)).unwrap();
                (encoder.set_fragment_bytes(&bytes_of(&rgba, f32::to_le_bytes), 0)).unwrap();
                (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
            }
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None);
        }
    }
}

/// The bytes a writer found put back, by where they lie, and its sweeps.
#[derive(Debug, Default)]
struct Tally {
    hot: u64,
    cold: u64,
    past: u64,
    sweeps: u64,
}

/// The offsets the writer writes: the padding in each row's line that holds its last texels; a
/// line of each row's padding that holds no texel; the line past the image and the buffer's last.
fn offsets() -> (Vec<Vec<u64>>, Vec<u64>, Vec<u64>) {
    let hot = (0..HEIGHT)
        .map(|row| (row * ROW + ROW_BYTES..row * ROW + ROW_BYTES.next_multiple_of(LINE)).collect())
        .collect();
    let cold = (0..HEIGHT)
        .flat_map(|row| row * ROW + ROW - LINE..row * ROW + ROW)
        .collect();
    let past = (ROW * HEIGHT..ROW * HEIGHT + LINE)
        .chain(LENGTH - LINE..LENGTH)
        .collect();
    (hot, cold, past)
}

/// Write the bytes [`offsets`] names at `base` until `stop`, each read first (the module docs).
fn write_beside(base: usize, stop: &AtomicBool) -> Tally {
    let (hot, cold, past) = offsets();
    // SAFETY: every offset lies in the buffer's pages, mapped while the writer runs; the GPU's
    // accesses of them are what the test observes, through atomic bytes.
    let byte = |offset: u64| unsafe { AtomicU8::from_ptr((base + offset as usize) as *mut u8) };
    let mut expected = vec![0u8; LENGTH as usize];
    for &offset in hot.iter().flatten().chain(&cold).chain(&past) {
        expected[offset as usize] = byte(offset).load(Ordering::Relaxed);
    }
    let mut tally = Tally::default();
    let mut step = |offset: u64, count: &mut u64| {
        let at = byte(offset);
        let want = &mut expected[offset as usize];
        if at.load(Ordering::Relaxed) != *want {
            *count += 1;
        }
        *want = want.wrapping_add(1);
        at.store(*want, Ordering::Relaxed);
    };
    let mut last = false;
    loop {
        let width = hot[0].len() as u64;
        for line in &hot {
            step(line[(tally.sweeps % width) as usize], &mut tally.hot);
        }
        if tally.sweeps.is_multiple_of(16) || last {
            for &offset in &cold {
                step(offset, &mut tally.cold);
            }
            for &offset in &past {
                step(offset, &mut tally.past);
            }
        }
        tally.sweeps += 1;
        if last {
            break;
        }
        last = stop.load(Ordering::Acquire);
    }
    // The last read of every byte, after the GPU's work.
    for &offset in hot.iter().flatten() {
        if byte(offset).load(Ordering::Relaxed) != expected[offset as usize] {
            tally.hot += 1;
        }
    }
    tally
}

/// Run `work` while a writer writes beside the texels at `base`: what it found, and how long the
/// work took.
fn beside(base: usize, work: impl FnOnce()) -> (Tally, Duration) {
    let stop = Arc::new(AtomicBool::new(false));
    let writer = {
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || write_beside(base, &stop))
    };
    let start = Instant::now();
    work();
    let took = start.elapsed();
    stop.store(true, Ordering::Release);
    (writer.join().unwrap(), took)
}

/// The counts a line names.
fn counted(before: &WorkCounts, after: &WorkCounts) -> String {
    format!(
        "shadow copies +{}, copies back spliced +{}, copies back dispatched +{}, merge \
         dispatches planned +{}, merge copies dispatched +{}, merges fetched +{}, merges kept in \
         images +{}",
        after.shadow_copies - before.shadow_copies,
        spliced(before, after),
        after.shadow_copies_back_dispatched - before.shadow_copies_back_dispatched,
        after.shadow_merge_dispatches_planned - before.shadow_merge_dispatches_planned,
        after.shadow_merge_copies_dispatched - before.shadow_merge_copies_dispatched,
        after.shadow_merges_fetched - before.shadow_merges_fetched,
        after.shadow_merges_kept_in_images - before.shadow_merges_kept_in_images,
    )
}

/// The copies back spliced between two counts.
fn spliced(before: &WorkCounts, after: &WorkCounts) -> u64 {
    after.shadow_copies_back_spliced - before.shadow_copies_back_spliced
}

/// A no-copy buffer of [`LENGTH`] bytes over `ram`'s pages, imported as dma-bufs.
fn over_dma_bufs(device: &Device, ram: &Ram, calls: &Arc<AtomicUsize>) -> Buffer {
    // SAFETY: the view outlives the buffer, and the source makes dma-bufs of its pages.
    let buffer = unsafe {
        device.new_buffer_with_bytes_no_copy(
            HostMemory::with_dma_buf(ram.pointer(0), ram.source(calls)),
            LENGTH,
            ResourceOptions::default(),
            None,
        )
    }
    .unwrap();
    assert!(
        buffer.over_dma_buf(),
        "{TEST}: the buffer lies over dma-bufs"
    );
    buffer
}

/// The shape's descriptor (the module docs).
fn shape() -> TextureDescriptor {
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, WIDTH, HEIGHT, false);
    descriptor.usage =
        TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE | TextureUsage::RENDER_TARGET;
    descriptor
}

/// A `Private` texture of the shape, the control's source.
fn source(device: &Device) -> Texture {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, WIDTH, HEIGHT, false);
    d.usage = TextureUsage::RENDER_TARGET;
    d.resource_options = ResourceOptions::private();
    device.new_texture(&d).unwrap()
}

/// The texture of the shape over `buffer`, in `form` (`buffer` or `plane`), a shadow.
fn target(buffer: &Buffer, form: &str) -> Texture {
    let descriptor = shape();
    let texture = match form {
        "buffer" => buffer.new_texture(&descriptor, 0, ROW).unwrap(),
        _ => (buffer
            .new_texture_with_iosurface_plane(&descriptor, IoSurfacePlane::new(0, ROW, false)))
        .unwrap(),
    };
    // SAFETY: the handle is only compared, while the texture lives.
    assert!(
        unsafe { texture.vk_image() }.is_none(),
        "{TEST}: the {form} texture is a shadow"
    );
    texture
}

/// The first test the module docs describe: no byte beside the texels is put back by a pass that
/// replaces them, its copies back spliced.
#[test]
fn a_shadow_s_copy_back_keeps_the_cpu_s_writes_beside_its_texels() {
    if !udmabuf_openable(TEST) {
        return;
    }
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let name = device.name();
    if !lists_dma_buf_route(&device) {
        println!("{TEST}: {name} lists no dma-buf route: not run");
        return;
    }
    let queue = device.new_command_queue().unwrap();
    let state = drawing(&device);
    let file = (LENGTH as usize).next_multiple_of(host_page());
    let ram = Ram::new(file, 0, file);
    let calls = Arc::new(AtomicUsize::new(0));
    let buffer = over_dma_bufs(&device, &ram, &calls);
    let base = ram.pointer(0).as_ptr().addr();
    let source = source(&device);

    let (idle, _) = beside(base, || std::thread::sleep(Duration::from_millis(500)));
    assert_eq!(
        (idle.hot, idle.cold, idle.past),
        (0, 0, 0),
        "{TEST}: the writer alone finds every byte it wrote: {idle:?}"
    );
    let (copied, took) = beside(base, || {
        for i in 0..ROUNDS {
            round(&queue, Kind::Copy, (&source, &source, &buffer), &state, i);
        }
    });
    println!("{TEST} on {name}: copy (the control): {ROUNDS} rounds in {took:?}: {copied:?}");
    for form in ["buffer", "plane"] {
        let target = target(&buffer, form);
        for kind in [Kind::Clear, Kind::Draw] {
            let before = device.work_counts();
            let (tally, took) = beside(base, || {
                for i in 0..ROUNDS {
                    round(&queue, kind, (&target, &source, &buffer), &state, i);
                }
            });
            let after = device.work_counts();
            println!(
                "{TEST} on {name}: {form} {kind:?}: {ROUNDS} rounds in {took:?}: {tally:?}; {}",
                counted(&before, &after)
            );
            if let Kind::Clear = kind {
                assert_eq!(
                    (tally.hot, tally.cold, tally.past),
                    (0, 0, 0),
                    "{TEST}: a {form} texture's copies back put back no byte the CPU wrote \
                     beside its texels (the control put back {}): {tally:?}",
                    copied.hot + copied.cold + copied.past
                );
                assert!(
                    spliced(&before, &after) >= ROUNDS as u64,
                    "{TEST}: every {form} copy back was spliced: {}",
                    counted(&before, &after)
                );
            }
        }
    }
    drop(buffer);
    drop(queue);
    drop(device);
    drop(ram);
}

/// The second test the module docs describe: a copy back spliced as the device's measured
/// verdicts say, given to a device whose policy runs no check.
#[test]
fn a_copy_back_splices_by_the_device_s_check() {
    if !udmabuf_openable(TEST) {
        return;
    }
    let scratch = Scratch::new();
    let measured = {
        let mut options = options(&scratch.0);
        options.withheld_extensions = [VkExtension::ExtExternalMemoryHost].into_iter().collect();
        options.policy.host_page_textures = HostPageTexturePolicy::InPlace;
        let checked = Device::system_default_with(&options).unwrap();
        if !lists_dma_buf_route(&checked) {
            println!("{TEST}: {} lists no dma-buf route: not run", checked.name());
            return;
        }
        assert!(
            checked.settle_in_place_checks(CHECK_WAIT),
            "{TEST}: the device's check"
        );
        let checks = &checked.shared().in_place;
        let class = PagesClass::DmaBuf;
        [Access::Copies, Access::Writes]
            .map(|access| (access, checks.decided(class, access, RowEnds::InLines)))
    };
    let device = device(&scratch.0);
    let name = device.name();
    let checks = &device.shared().in_place;
    for (access, verdict) in measured {
        checks.set(PagesClass::DmaBuf, access, RowEnds::InLines, verdict);
    }
    let splices = crate::shadow_splice::splices(checks, PagesClass::DmaBuf);
    let queue = device.new_command_queue().unwrap();
    let state = drawing(&device);
    let file = (LENGTH as usize).next_multiple_of(host_page());
    let ram = Ram::new(file, 0, file);
    let calls = Arc::new(AtomicUsize::new(0));
    let buffer = over_dma_bufs(&device, &ram, &calls);
    let source = source(&device);
    for form in ["buffer", "plane"] {
        let target = target(&buffer, form);
        let before = device.work_counts();
        round(&queue, Kind::Clear, (&target, &source, &buffer), &state, 0);
        let after = device.work_counts();
        println!(
            "{TEST} on {name}: the check's verdicts for rows that end inside a line {measured:?}, \
             so spliced {splices}: {form} Clear: {}",
            counted(&before, &after)
        );
        assert_eq!(
            spliced(&before, &after) > 0,
            splices,
            "{TEST}: a {form} copy back spliced as the verdicts {measured:?} say: {}",
            counted(&before, &after)
        );
    }
    drop(buffer);
    drop(queue);
    drop(device);
    drop(ram);
}
