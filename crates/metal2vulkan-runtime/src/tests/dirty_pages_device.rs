//! A shadowed texture over a no-copy buffer's host pages whose memory came with a log of the
//! writes to them (`HostMemory::with_dirty_pages`, `dirty_pages.rs`): its copies in after the
//! first copy only the rows of the pages the log reports changed since, leave out the shadow's
//! own copies back, and copy every row where the log cannot say.
//!
//! The texture is 64x64 `RGBA8Unorm` over the buffer's first bytes, rows of 256 bytes, read and
//! written through (so it is a shadow on every device). A kernel inverts its texels into a plain
//! texture, which the test reads back: the texels the shadow held. The log is the test's own
//! ([`Log`]): it reports the writes the test tells it of, and lies by leaving out one it does not
//! tell, so a row the log leaves out shows whether a copy took it.
//!
//! - Frame 1 copies every row (no hold yet).
//! - Frame 2: the CPU writes row 10, told to the log, and row 40, not told. Row 10 reads the new
//!   bytes, row 40 the old ones: the copy took only the told row's band
//!   (`WorkCounts::dirty_copy_in_bytes_copied`).
//! - Frame 3 with no write copies nothing (`dirty_copies_in_skipped`).
//! - Frame 4: the log's wait answers `false`: every row is copied, row 40's new bytes too.
//! - Frame 7: a command buffer that waits for an event copies every row: work committed after it
//!   may run before it.
//! - Frame 5: a kernel writes through the shadow (copied back, or merged, reported as the
//!   shadow's own write, `wrote_as`, at the commit and again at the completion); frame 6 then
//!   copies nothing in for it (`changed_except` leaves the shadow's own write out).
//!
//! A pass that clears such a render target and draws inside rows 0 to 8 replaces every texel
//! (`ShadowUse::overwrites`): over logged pages it takes the merge, not a plain copy back
//! (`shadow.rs`, **Overwrites over logged pages**). A row the CPU wrote since and told the log,
//! which the pass clears to the value the image held before, ends with the pass's value (the
//! gated copy in took the CPU's bytes into the keep), and a row the CPU wrote but the log left
//! out, which the pass clears to the same value, keeps the CPU's bytes: the merge stores only
//! the words the pass changed against the keep. Over host pages with no log, the same pass's copy
//! back is a plain copy of every texel.
//!
//! Over a 1024x1024 such target whose CPU-written texels (told to the log) differ from black at
//! every other texel, a pass that clears it to black (with a draw in one pixel) changes every
//! other word: its merge stores whole 256-byte blocks
//! (`WorkCounts::shadow_merge_dispatches_in_blocks`) and the bytes are the pass's, the CPU's
//! texels it cleared black among them; a pass that loads it and draws black over every pixel
//! changes the same words, and its merge stores whole blocks too, as Metal's own store of such a
//! pass writes back every texel of each tile its draws reach (`shadow.rs`, **Stored
//! attachments**). Each prints the bytes its merge stores and its command buffers' median GPU
//! time (`DIRTY-OVERWRITE`). A pass that loads it and draws inside a 64x64 square, while a CPU
//! thread writes three texels beside the square over and over from before the commit to after
//! the completion, merges in whole blocks of the square's rows alone: each of those texels ends
//! holding the CPU's last write (`DIRTY-BESIDE`).
//!
//! One wait for a log's publication serves every gated copy in of a commit, two shadows over one
//! log among them, and a wait the log answers `false` is not made again for them
//! (`WorkCounts::dirty_waits`).
//!
//! A render target over such pages, loaded by a pass whose draws' scissor bounds them to rows 0
//! to 8, copies in every texel the first time and then only the rows the log reports changed: its
//! copy in is not cut to the scissor, so the image holds every texel for the next command buffer,
//! and the pass's own copy back of rows 0 to 8 is no change to it.
//!
//! Such a shadow sampled by a pass whose first instance records inline (`render/inline_first.rs`)
//! takes its gated copy in before the pass, as every command before the pass does
//! (`EncoderCore::hold_pass`): the pass's draws read the bytes the CPU wrote, the first time and
//! after a write told to the log, and the pass stays one inline instance. So does a texture the
//! pass only samples, which a discrete GPU makes a device copy (`WorkCounts::device_copy_textures`)
//! and every other device reads in place (no copy, printed).
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`). The kernels are the owned fixture
//! `validation/fixtures/public/examples/runtime_examples.ll` (`llvm-as`).

use super::blit_holds::{device, over_host, Host, Scratch, BYTES, WAIT};
use super::encoder_holds::{bytes_of, render_state};
use crate::{
    Buffer, ClearColor, CommandBufferStatus, ComputePipelineState, Device, DirtyPages, DirtyStamp,
    Error, EventRef, HostMemory, Library, LoadAction, PipelineOptions, Region,
    RenderPassDescriptor, ResourceOptions, ScissorRect, Size, StoreAction, Texture,
    TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use std::num::NonZeroU64;
use std::ops::Range;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

const SIDE: u64 = 64;
const ROW: u64 = 256;

/// The test's log: every write it is told of, stamped, with its writer (0 the consumer's).
#[derive(Default)]
struct Log {
    next: AtomicU64,
    writes: Mutex<Vec<(u64, Range<u64>, u64)>>,
    /// The wait answers `false` while set.
    unanswered: AtomicBool,
    watches: AtomicU64,
    waits: AtomicU64,
    /// The writers `changed_except` was asked to leave out, and `wrote_as` named.
    asked: Mutex<Vec<u64>>,
    named: Mutex<Vec<u64>>,
}

impl Log {
    fn told(&self, range: Range<u64>, writer: u64) {
        let stamp = self.next.fetch_add(1, Ordering::SeqCst) + 1;
        self.writes.lock().unwrap().push((stamp, range, writer));
    }
}

impl DirtyPages for Log {
    type Watch = ();

    fn watch(&self, _range: Range<u64>) {
        self.watches.fetch_add(1, Ordering::SeqCst);
    }

    fn request(&self) -> DirtyStamp {
        DirtyStamp(self.next.fetch_add(1, Ordering::SeqCst) + 1)
    }

    fn wait(&self, _stamp: DirtyStamp) -> bool {
        self.waits.fetch_add(1, Ordering::SeqCst);
        !self.unanswered.load(Ordering::SeqCst)
    }

    fn changed(&self, since: DirtyStamp, out: &mut Vec<Range<u64>>) {
        let writes = self.writes.lock().unwrap();
        out.extend(
            (writes.iter())
                .filter(|w| w.0 > since.0)
                .map(|w| w.1.clone()),
        );
    }

    fn wrote(&self, range: Range<u64>) {
        self.told(range, 0);
    }

    fn wrote_as(&self, range: Range<u64>, writer: NonZeroU64) {
        self.named.lock().unwrap().push(writer.get());
        self.told(range, writer.get());
    }

    fn changed_except(&self, since: DirtyStamp, writer: NonZeroU64, out: &mut Vec<Range<u64>>) {
        self.asked.lock().unwrap().push(writer.get());
        let writes = self.writes.lock().unwrap();
        let changed = (writes.iter()).filter(|w| w.0 > since.0 && w.2 != writer.get());
        out.extend(changed.map(|w| w.1.clone()));
    }
}

fn state(device: &Device, kernel: &str) -> ComputePipelineState {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/examples/runtime_examples.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    let library: Library = device.new_library_with_compile_library(library);
    let function = library.new_function_with_name(kernel).unwrap();
    device
        .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
        .unwrap()
}

/// Run `kernel` over `SIDE` x `SIDE` texels from `source` into `target`, to completion.
fn run(device: &Device, kernel: &ComputePipelineState, source: &Texture, target: &Texture) {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(kernel).unwrap();
    encoder.set_texture(Some(source), 0).unwrap();
    encoder.set_texture(Some(target), 1).unwrap();
    encoder
        .dispatch_threads(Size::new(SIDE, SIDE, 1), Size::new(8, 8, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// The first texel of row `row` of `texture`, as the kernel inverted it.
fn texel(texture: &Texture, row: u64) -> [u8; 4] {
    let mut bytes = [0u8; 4];
    (texture.get_bytes(&mut bytes, 4, 0, Region::new_2d(0, row, 1, 1), 0, 0)).unwrap();
    bytes
}

/// The counts a frame moved: (gated, whole, skipped, bytes whole, bytes copied).
fn counts(device: &Device) -> [u64; 5] {
    let c = device.work_counts();
    [
        c.dirty_copies_in_gated,
        c.dirty_copies_in_whole,
        c.dirty_copies_in_skipped,
        c.dirty_copy_in_bytes_whole,
        c.dirty_copy_in_bytes_copied,
    ]
}

fn delta(device: &Device, before: [u64; 5]) -> [u64; 5] {
    let now = counts(device);
    std::array::from_fn(|i| now[i] - before[i])
}

/// The buffer over `host`, its memory logged by `log`.
fn over_logged(device: &Device, host: &Host, log: &Arc<Log>) -> Option<Buffer> {
    // SAFETY: the memory outlives the buffer (dropped after the device), and only the buffer's
    // commands and the test use it while it lives.
    let memory: HostMemory = unsafe { host.memory() }.with_dirty_pages(Arc::clone(log));
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(memory, BYTES as u64, ResourceOptions::default(), None)
    };
    match made {
        Ok(buffer) => Some(buffer),
        Err(Error::Unsupported(gap)) => {
            eprintln!("DIRTY-PAGES: the device imports no host memory: {gap:?}");
            None
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    }
}

#[test]
fn a_logged_shadow_copies_in_only_the_changed_rows() {
    let scratch = Scratch::new("dirty-pages-rows");
    let host = Host::new();
    let device = device(&scratch.0);
    eprintln!("DIRTY-PAGES: device {:?}", device.name());
    let log = Arc::new(Log::default());
    let Some(buffer) = over_logged(&device, &host, &log) else {
        return;
    };
    let span = (SIDE * ROW) as usize;
    host.fill(0x1010_1010);
    let shadows = device.work_counts().shadowed_textures;
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    let source = buffer.new_texture(&descriptor, 0, ROW).unwrap();
    let shadowed = device.work_counts().shadowed_textures - shadows;
    let mut plain = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    plain.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    plain.resource_options = ResourceOptions::shared();
    let target = device.new_texture(&plain).unwrap();
    let invert = state(&device, "invert_texels");
    eprintln!("DIRTY-PAGES: {shadowed} shadow made");
    if shadowed == 0 {
        return;
    }
    let bytes = buffer.contents().unwrap().as_ptr();
    let write_row = |row: u64, value: u8| {
        // SAFETY: a row inside the buffer's first bytes, which no command uses now.
        unsafe { std::ptr::write_bytes(bytes.add((row * ROW) as usize), value, ROW as usize) };
    };
    let inverted = |value: u8| [255 - value, 255 - value, 255 - value, value];

    // Frame 1: every row.
    let before = counts(&device);
    run(&device, &invert, &source, &target);
    let first = delta(&device, before);
    eprintln!("DIRTY-PAGES: frame 1 (gated, whole, skipped, bytes whole, copied) {first:?}");
    assert_eq!(texel(&target, 10), inverted(0x10));
    assert_eq!(first[0], 1, "the first copy in is gated");
    assert_eq!(first[1], 1, "the first copy in takes every row");
    assert_eq!(first[4], (SIDE * SIDE * 4), "every byte");
    assert!(
        log.watches.load(Ordering::SeqCst) >= 1,
        "the shadow watches its bytes"
    );

    // Frame 2: row 10 told, row 40 not.
    write_row(10, 0x20);
    write_row(40, 0x30);
    log.told(10 * ROW..11 * ROW, 0);
    let before = counts(&device);
    run(&device, &invert, &source, &target);
    let second = delta(&device, before);
    eprintln!("DIRTY-PAGES: frame 2 {second:?}");
    assert_eq!(
        texel(&target, 10),
        inverted(0x20),
        "the told row was copied"
    );
    assert_eq!(texel(&target, 40), inverted(0x10), "the untold row was not");
    assert_eq!(second[0], 1);
    assert_eq!(second[1], 0, "not a whole copy");
    assert_eq!(second[4], SIDE * 4, "one row's bytes");
    assert!(
        log.waits.load(Ordering::SeqCst) >= 1,
        "the commit's stamp was waited for"
    );

    // Frame 3: nothing changed.
    let before = counts(&device);
    run(&device, &invert, &source, &target);
    let third = delta(&device, before);
    eprintln!("DIRTY-PAGES: frame 3 {third:?}");
    assert_eq!(third[2], 1, "nothing copied in");
    assert_eq!(third[4], 0);

    // Frame 4: the log cannot say: every row.
    log.unanswered.store(true, Ordering::SeqCst);
    let before = counts(&device);
    run(&device, &invert, &source, &target);
    let fourth = delta(&device, before);
    log.unanswered.store(false, Ordering::SeqCst);
    eprintln!("DIRTY-PAGES: frame 4 {fourth:?}");
    assert_eq!(fourth[1], 1, "a whole copy");
    assert_eq!(fourth[4], SIDE * SIDE * 4);
    assert_eq!(
        texel(&target, 40),
        inverted(0x30),
        "the untold row came with it"
    );

    // Frame 5: a kernel writes through the shadow; its copy back is its own.
    run(&device, &invert, &target, &source);
    let named = log.named.lock().unwrap().clone();
    eprintln!("DIRTY-PAGES: frame 5 named writers {named:?}");
    assert!(
        !named.is_empty(),
        "the copy back was reported as the shadow's"
    );
    assert!(
        named.len() >= 2 && named.iter().all(|&w| w == named[0]),
        "reported at the commit and again at the completion, by one writer"
    );
    // Frame 6: the shadow's own write is no change to it.
    let before = counts(&device);
    run(&device, &invert, &source, &target);
    let sixth = delta(&device, before);
    let asked = log.asked.lock().unwrap().clone();
    eprintln!("DIRTY-PAGES: frame 6 {sixth:?}, asked writers {asked:?}");
    assert!(
        asked.contains(&named[0]),
        "its copies in leave its own writes out"
    );
    assert_eq!(sixth[4], 0, "nothing copied in after its own copy back");
    // Frame 7: a command buffer that waits for an event may run after work committed later:
    // every row.
    let event = device.new_shared_event().unwrap();
    event.set_signaled_value(1);
    let before = counts(&device);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    (command_buffer.encode_wait_for_event(EventRef::Shared(&event), 1)).unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&invert).unwrap();
    encoder.set_texture(Some(&source), 0).unwrap();
    encoder.set_texture(Some(&target), 1).unwrap();
    encoder
        .dispatch_threads(Size::new(SIDE, SIDE, 1), Size::new(8, 8, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
    let seventh = delta(&device, before);
    eprintln!("DIRTY-PAGES: frame 7 (after an event wait) {seventh:?}");
    assert_eq!(seventh[1], 1, "a whole copy after an event wait");
    // SAFETY: every command completed; nothing writes the bytes now.
    let now = unsafe { std::slice::from_raw_parts(bytes, span) };
    assert_eq!(now[10 * ROW as usize], 0x20);
    drop(source);
    drop(buffer);
    drop(device);
}

/// One pass over `target` that loads it and draws `color` over every pixel inside rows 0 to 8 (the
/// scissor), to completion.
fn draw_in_scissor(device: &Device, target: &Texture, color: [f32; 4]) {
    let state = render_state(device, "engine/tob_renders.ll", "tob_rect", "tob_solid");
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(target.clone());
    attachment.load_action = LoadAction::Load;
    attachment.store_action = StoreAction::Store;
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(&state).unwrap();
    let scissor = ScissorRect {
        x: 0,
        y: 0,
        width: SIDE,
        height: 8,
    };
    encoder.set_scissor_rect(scissor).unwrap();
    let corners = bytes_of(&[-1.0f32, -1.0, 1.0, 1.0], f32::to_le_bytes);
    encoder.set_vertex_bytes(&corners, 0).unwrap();
    let color = bytes_of(&color, f32::to_le_bytes);
    encoder.set_fragment_bytes(&color, 0).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

#[test]
fn a_logged_render_target_loaded_in_a_scissor_copies_only_the_changed_rows() {
    let scratch = Scratch::new("dirty-pages-scissor");
    let host = Host::new();
    let device = device(&scratch.0);
    eprintln!("DIRTY-PAGES: device {:?}", device.name());
    let log = Arc::new(Log::default());
    let Some(buffer) = over_logged(&device, &host, &log) else {
        return;
    };
    host.fill(0x1010_1010);
    let shadows = device.work_counts().shadowed_textures;
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let target = buffer.new_texture(&descriptor, 0, ROW).unwrap();
    let shadowed = device.work_counts().shadowed_textures - shadows;
    eprintln!("DIRTY-PAGES: {shadowed} render shadow made");
    if shadowed == 0 {
        return;
    }
    let bytes = buffer.contents().unwrap().as_ptr();
    let row_byte = |row: u64| {
        // SAFETY: a byte inside the buffer's first rows, which no command uses now.
        unsafe { *bytes.add((row * ROW) as usize) }
    };

    // Frame 1: every texel, the image holding all of them after it.
    let before = counts(&device);
    draw_in_scissor(&device, &target, [1.0, 0.0, 0.0, 1.0]);
    let first = delta(&device, before);
    eprintln!(
        "DIRTY-PAGES: scissor frame 1 (gated, whole, skipped, bytes whole, copied) {first:?}"
    );
    assert_eq!(
        first[0], 1,
        "the first copy in is gated, not cut to the scissor"
    );
    assert_eq!(first[4], SIDE * SIDE * 4, "every texel the first time");
    assert_eq!(row_byte(0), 255, "the pass drew row 0");
    assert_eq!(row_byte(20), 0x10, "and left row 20");

    // Frame 2: row 40 told to the log; the pass's own copy back of rows 0 to 8 is its own.
    // SAFETY: a row inside the buffer's first bytes, which no command uses now.
    unsafe { std::ptr::write_bytes(bytes.add((40 * ROW) as usize), 0x40, ROW as usize) };
    log.told(40 * ROW..41 * ROW, 0);
    let before = counts(&device);
    draw_in_scissor(&device, &target, [0.0, 1.0, 0.0, 1.0]);
    let second = delta(&device, before);
    eprintln!("DIRTY-PAGES: scissor frame 2 {second:?}");
    assert_eq!(second[0], 1, "gated again");
    assert_eq!(second[1], 0, "not a whole copy");
    assert_eq!(second[4], SIDE * 4, "the told row alone");
    assert_eq!(row_byte(40), 0x40, "the CPU's row stays");
    assert_eq!(row_byte(0), 0, "the second pass drew row 0");

    // Frame 3: nothing changed but the pass's own copy back.
    let before = counts(&device);
    draw_in_scissor(&device, &target, [0.0, 0.0, 1.0, 1.0]);
    let third = delta(&device, before);
    eprintln!("DIRTY-PAGES: scissor frame 3 {third:?}");
    assert_eq!(third[2], 1, "nothing copied in");
    drop(target);
    drop(buffer);
    drop(device);
}

/// One pass over `target` that clears it to opaque black and draws `color` over every pixel
/// inside rows 0 to 8 (the scissor), to completion: it replaces every texel.
fn clear_and_draw(device: &Device, target: &Texture, color: [f32; 4]) {
    let state = render_state(device, "engine/tob_renders.ll", "tob_rect", "tob_solid");
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(target.clone());
    attachment.load_action = LoadAction::Clear;
    attachment.clear_color = ClearColor::new(0.0, 0.0, 0.0, 1.0);
    attachment.store_action = StoreAction::Store;
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(&state).unwrap();
    let scissor = ScissorRect {
        x: 0,
        y: 0,
        width: SIDE,
        height: 8,
    };
    encoder.set_scissor_rect(scissor).unwrap();
    let corners = bytes_of(&[-1.0f32, -1.0, 1.0, 1.0], f32::to_le_bytes);
    encoder.set_vertex_bytes(&corners, 0).unwrap();
    let color = bytes_of(&color, f32::to_le_bytes);
    encoder.set_fragment_bytes(&color, 0).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// The texels copied back plainly and the merge dispatches planned so far.
fn backs(device: &Device) -> [u64; 2] {
    let c = device.work_counts();
    [
        c.shadow_texels_copied_back,
        c.shadow_merge_dispatches_planned,
    ]
}

/// A render target over `buffer`'s first bytes, or `None` where the device needs no shadow.
fn render_target(device: &Device, buffer: &Buffer) -> Option<Texture> {
    let shadows = device.work_counts().shadowed_textures;
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let target = buffer.new_texture(&descriptor, 0, ROW).unwrap();
    let shadowed = device.work_counts().shadowed_textures - shadows;
    eprintln!("DIRTY-PAGES: {shadowed} render shadow made");
    (shadowed > 0).then_some(target)
}

const BLACK: [u8; 4] = [0, 0, 0, 255];
const RED: [u8; 4] = [255, 0, 0, 255];

#[test]
fn a_pass_replacing_a_logged_render_target_merges_back_the_words_it_changed() {
    let scratch = Scratch::new("dirty-pages-overwrite");
    let host = Host::new();
    let device = device(&scratch.0);
    eprintln!("DIRTY-PAGES: device {:?}", device.name());
    let log = Arc::new(Log::default());
    let Some(buffer) = over_logged(&device, &host, &log) else {
        return;
    };
    host.fill(0x1010_1010);
    let Some(target) = render_target(&device, &buffer) else {
        return;
    };
    let bytes = buffer.contents().unwrap().as_ptr();
    let texel_at = |row: u64| {
        let mut texel = [0u8; 4];
        // SAFETY: a texel inside the buffer's first rows, which no command uses now.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.add((row * ROW) as usize), texel.as_mut_ptr(), 4)
        };
        texel
    };

    // Frame 1: the pass clears every texel and draws rows 0 to 8, merged back, no plain copy.
    let before = backs(&device);
    clear_and_draw(&device, &target, [1.0, 0.0, 0.0, 1.0]);
    let [copied, merges] = std::array::from_fn(|i| backs(&device)[i] - before[i]);
    eprintln!("DIRTY-PAGES: overwrite frame 1: {copied} texels copied back, {merges} merges");
    assert_eq!(copied, 0, "no texel copied back plainly");
    assert!(merges > 0, "the pass's copy back is a merge");
    assert_eq!(texel_at(0), RED, "the pass drew row 0");
    assert_eq!(texel_at(20), BLACK, "and cleared row 20");

    // Frame 2: the CPU writes row 40, told to the log, and row 50, left out of it; the pass
    // clears both to the black the image held.
    // SAFETY: rows inside the buffer's first bytes, which no command uses now.
    unsafe {
        std::ptr::write_bytes(bytes.add((40 * ROW) as usize), 0x40, ROW as usize);
        std::ptr::write_bytes(bytes.add((50 * ROW) as usize), 0x50, ROW as usize);
    }
    log.told(40 * ROW..41 * ROW, 0);
    let before = (backs(&device), counts(&device));
    clear_and_draw(&device, &target, [1.0, 0.0, 0.0, 1.0]);
    let [copied, merges] = std::array::from_fn(|i| backs(&device)[i] - before.0[i]);
    let moved = delta(&device, before.1);
    eprintln!(
        "DIRTY-PAGES: overwrite frame 2: {copied} texels copied back, {merges} merges, copy in \
         (gated, whole, skipped, bytes whole, copied) {moved:?}; row 40 {:?}, row 50 {:?}",
        texel_at(40),
        texel_at(50),
    );
    assert_eq!(copied, 0, "no texel copied back plainly");
    assert!(merges > 0, "merged again");
    assert_eq!(
        moved[4],
        SIDE * 4,
        "the gated copy in took the told row alone"
    );
    assert_eq!(
        texel_at(40),
        BLACK,
        "the pass's value over the told CPU write: Metal's"
    );
    assert_eq!(
        texel_at(50),
        [0x50; 4],
        "a word the pass left as the keep held it is not stored"
    );
    assert_eq!(texel_at(0), RED);
    drop(target);
    drop(buffer);
    drop(device);
}

#[test]
fn a_pass_replacing_an_unlogged_render_target_copies_it_back() {
    let scratch = Scratch::new("dirty-pages-overwrite-unlogged");
    let host = Host::new();
    let device = device(&scratch.0);
    let runs = Arc::new(AtomicUsize::new(0));
    let Some(buffer) = over_host(&device, &host, &runs) else {
        return;
    };
    host.fill(0x1010_1010);
    let Some(target) = render_target(&device, &buffer) else {
        return;
    };
    let bytes = buffer.contents().unwrap().as_ptr();
    for frame in 1..=2 {
        // SAFETY: a row inside the buffer's first bytes, which no command uses now.
        unsafe { std::ptr::write_bytes(bytes.add((50 * ROW) as usize), 0x50, ROW as usize) };
        let before = backs(&device);
        clear_and_draw(&device, &target, [1.0, 0.0, 0.0, 1.0]);
        let [copied, merges] = std::array::from_fn(|i| backs(&device)[i] - before[i]);
        let mut texel = [0u8; 4];
        // SAFETY: as above; the command buffer completed.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.add((50 * ROW) as usize), texel.as_mut_ptr(), 4)
        };
        eprintln!(
            "DIRTY-PAGES: unlogged overwrite frame {frame}: {copied} texels copied back, {merges} \
             merges; row 50 {texel:?}"
        );
        assert_eq!(copied, SIDE * SIDE, "every texel copied back plainly");
        assert_eq!(merges, 0, "no merge");
        assert_eq!(texel, BLACK, "the pass's value");
    }
    drop(target);
    drop(buffer);
    drop(device);
}

/// The large target's side and rows, the host pages' 4 MiB.
const LARGE: u64 = 1024;
const LARGE_ROW: u64 = 4 * LARGE;

/// One pass over `target` (`LARGE` square) that clears it to black (`load` `Clear`) or loads it,
/// and draws `color` over the pixels inside `scissor`, to completion: the command buffer's GPU
/// time in milliseconds, where the device gives it.
fn pass(
    device: &Device,
    target: &Texture,
    load: LoadAction,
    scissor: ScissorRect,
    color: [f32; 4],
) -> Option<f64> {
    let state = render_state(device, "engine/tob_renders.ll", "tob_rect", "tob_solid");
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut descriptor = RenderPassDescriptor::new();
    let attachment = &mut descriptor.color_attachments[0];
    attachment.texture = Some(target.clone());
    attachment.load_action = load;
    attachment.clear_color = ClearColor::new(0.0, 0.0, 0.0, 1.0);
    attachment.store_action = StoreAction::Store;
    let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
    encoder.set_render_pipeline_state(&state).unwrap();
    encoder.set_scissor_rect(scissor).unwrap();
    let corners = bytes_of(&[-1.0f32, -1.0, 1.0, 1.0], f32::to_le_bytes);
    encoder.set_vertex_bytes(&corners, 0).unwrap();
    let color = bytes_of(&color, f32::to_le_bytes);
    encoder.set_fragment_bytes(&color, 0).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let (start, end) = (
        command_buffer.gpu_start_time(),
        command_buffer.gpu_end_time(),
    );
    Some((end.ok()? - start.ok()?) * 1e3)
}

/// The merge dispatches planned and those planned in whole blocks so far.
fn merges(device: &Device) -> [u64; 2] {
    let c = device.work_counts();
    [
        c.shadow_merge_dispatches_planned,
        c.shadow_merge_dispatches_in_blocks,
    ]
}

#[test]
fn a_scattered_overwrite_of_logged_pages_merges_back_in_whole_blocks() {
    const ROUNDS: usize = 8;
    let scratch = Scratch::new("dirty-pages-blocks");
    let host = Host::new();
    let device = device(&scratch.0);
    eprintln!("DIRTY-PAGES: device {:?}", device.name());
    let log = Arc::new(Log::default());
    let Some(buffer) = over_logged(&device, &host, &log) else {
        return;
    };
    assert!(LARGE_ROW * LARGE <= BYTES as u64);
    let shadows = device.work_counts().shadowed_textures;
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, LARGE, LARGE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let target = buffer.new_texture(&descriptor, 0, LARGE_ROW).unwrap();
    if device.work_counts().shadowed_textures == shadows {
        eprintln!("DIRTY-PAGES: no render shadow made");
        return;
    }
    let span = (LARGE_ROW * LARGE) as usize;
    let bytes = buffer.contents().unwrap().as_ptr();
    // The CPU's texels: every other one grey, the others black; told to the log.
    let scatter = || {
        let words: Vec<u32> = (0..LARGE * LARGE)
            .map(|i| match i % 2 {
                0 => 0x4040_4040,
                _ => u32::from_le_bytes(BLACK),
            })
            .collect();
        // SAFETY: the texture's bytes, which no command uses now.
        unsafe { std::ptr::copy_nonoverlapping(words.as_ptr().cast::<u8>(), bytes, span) };
        log.told(0..span as u64, 0);
    };
    let got = || {
        // SAFETY: as above; every command buffer completed.
        unsafe { std::slice::from_raw_parts(bytes, span) }.to_vec()
    };
    let every = ScissorRect {
        x: 0,
        y: 0,
        width: LARGE,
        height: LARGE,
    };
    let corner = ScissorRect {
        x: 0,
        y: 0,
        width: 1,
        height: 1,
    };
    let legs = [
        (
            "blocks (a pass clearing every texel)",
            LoadAction::Clear,
            corner,
            RED,
        ),
        (
            "blocks (a pass loading and drawing every texel)",
            LoadAction::Load,
            every,
            BLACK,
        ),
    ];
    for (what, load, scissor, drawn) in legs {
        let color = drawn.map(|c| f32::from(c) / 255.0);
        let mut times = Vec::new();
        let (mut planned, mut in_blocks) = (0, 0);
        for _ in 0..ROUNDS {
            scatter();
            let before = merges(&device);
            times.extend(pass(&device, &target, load, scissor, color));
            let after = merges(&device);
            planned += after[0] - before[0];
            in_blocks += after[1] - before[1];
            let now = got();
            let wrong = (now.as_chunks::<4>().0.iter().enumerate())
                .filter(|&(i, texel)| *texel != if i == 0 { drawn } else { BLACK })
                .count();
            assert_eq!(wrong, 0, "{what}: texels other than the pass's");
        }
        assert!(planned > 0, "{what}: merged");
        // Both store whole blocks: one clears every texel, the other loads every texel and
        // stores the tiles its draws reach (`ShadowUse::tile_stores`).
        assert_eq!(in_blocks, planned, "{what}");
        // By construction: every 256-byte block holds a changed word.
        let stored = span;
        times.sort_by(f64::total_cmp);
        let median = times.get(times.len() / 2).copied().unwrap_or(f64::NAN);
        eprintln!(
            "DIRTY-OVERWRITE: {}: {what}: {in_blocks} of {planned} merge dispatches in whole \
             blocks, {stored} bytes stored; GPU time median {median:.3} ms of {} command buffers \
             [{:.3}..{:.3}]",
            device.name(),
            times.len(),
            times.first().copied().unwrap_or(f64::NAN),
            times.last().copied().unwrap_or(f64::NAN),
        );
    }
    drop(target);
    drop(buffer);
    drop(device);
}

/// The texels beside the square a pass draws in that the CPU writes while it runs: right of it in
/// its rows (the next 256-byte block of a row), far right of it, and below it.
const BESIDE: [(u64, u64); 3] = [(64, 10), (900, 40), (10, 100)];

#[test]
fn a_pass_storing_its_loaded_area_keeps_a_cpu_write_beside_it() {
    const ROUNDS: usize = 8;
    let scratch = Scratch::new("dirty-pages-beside");
    let host = Host::new();
    let device = device(&scratch.0);
    eprintln!("DIRTY-PAGES: device {:?}", device.name());
    let log = Arc::new(Log::default());
    let Some(buffer) = over_logged(&device, &host, &log) else {
        return;
    };
    assert!(LARGE_ROW * LARGE <= BYTES as u64);
    let shadows = device.work_counts().shadowed_textures;
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, LARGE, LARGE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let target = buffer.new_texture(&descriptor, 0, LARGE_ROW).unwrap();
    if device.work_counts().shadowed_textures == shadows {
        eprintln!("DIRTY-PAGES: no render shadow made");
        return;
    }
    let span = (LARGE_ROW * LARGE) as usize;
    let bytes = buffer.contents().unwrap().as_ptr();
    let grey = 0x4040_4040u32;
    let square = ScissorRect {
        x: 0,
        y: 0,
        width: 64,
        height: 64,
    };
    let inside = |x: u64, y: u64| x < 64 && y < 64;
    let at = |(x, y): (u64, u64)| (y * LARGE_ROW + 4 * x) as usize;
    let (mut planned, mut in_blocks, mut sweeps) = (0, 0, 0);
    for _ in 0..ROUNDS {
        // The CPU's texels: every other one grey, the others black; told to the log.
        let words: Vec<u32> = (0..LARGE * LARGE)
            .map(|i| match i % 2 {
                0 => grey,
                _ => u32::from_le_bytes(BLACK),
            })
            .collect();
        // SAFETY: the texture's bytes, which no command uses now.
        unsafe { std::ptr::copy_nonoverlapping(words.as_ptr().cast::<u8>(), bytes, span) };
        log.told(0..span as u64, 0);
        let before = merges(&device);
        let stop = AtomicBool::new(false);
        let started = AtomicBool::new(false);
        let address = bytes as usize;
        let last = std::thread::scope(|scope| {
            let writer = scope.spawn(|| {
                // SAFETY: whole texels of the buffer's bytes, 4-aligned; the pass writes none
                // of them, and only this thread writes them while it runs.
                let cells = BESIDE.map(|texel| unsafe {
                    std::sync::atomic::AtomicU32::from_ptr((address + at(texel)) as *mut u32)
                });
                let mut value = 0u32;
                let mut swept = 0u64;
                loop {
                    value = value.wrapping_add(1) | 0x8000_0000;
                    for cell in &cells {
                        cell.store(value, Ordering::SeqCst);
                    }
                    swept += 1;
                    started.store(true, Ordering::SeqCst);
                    if stop.load(Ordering::SeqCst) {
                        return (value, swept);
                    }
                }
            });
            while !started.load(Ordering::SeqCst) {
                std::hint::spin_loop();
            }
            pass(
                &device,
                &target,
                LoadAction::Load,
                square,
                [0.0, 0.0, 0.0, 1.0],
            );
            stop.store(true, Ordering::SeqCst);
            writer.join().unwrap()
        });
        sweeps += last.1;
        let after = merges(&device);
        planned += after[0] - before[0];
        in_blocks += after[1] - before[1];
        // SAFETY: the command buffer completed and the writer ended; nothing writes the bytes.
        let now = unsafe { std::slice::from_raw_parts(bytes, span) };
        for (texel, word) in now.as_chunks::<4>().0.iter().enumerate() {
            let (x, y) = (texel as u64 % LARGE, texel as u64 / LARGE);
            let want = match () {
                _ if BESIDE.contains(&(x, y)) => last.0.to_le_bytes(),
                _ if inside(x, y) => BLACK,
                _ => words[texel].to_le_bytes(),
            };
            assert_eq!(*word, want, "texel ({x}, {y})");
        }
    }
    eprintln!(
        "DIRTY-BESIDE: {}: {in_blocks} of {planned} merge dispatches in whole blocks; the CPU \
         wrote {} texels beside the square {sweeps} times over {ROUNDS} command buffers, each \
         ending with its last write",
        device.name(),
        BESIDE.len(),
    );
    assert!(planned > 0, "merged");
    assert_eq!(in_blocks, planned, "in whole blocks");
    drop(target);
    drop(buffer);
    drop(device);
}

/// The stamps asked, the waits made and the waits answered `false` so far.
fn waits(device: &Device) -> [u64; 3] {
    let c = device.work_counts();
    [
        c.dirty_stamps_asked,
        c.dirty_waits,
        c.dirty_waits_unanswered,
    ]
}

/// One command buffer whose encoders each run `kernel` from one of `sources` into `target`, to
/// completion.
fn run_each(
    device: &Device,
    kernel: &ComputePipelineState,
    sources: &[&Texture],
    target: &Texture,
) {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    for source in sources {
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(kernel).unwrap();
        encoder.set_texture(Some(*source), 0).unwrap();
        encoder.set_texture(Some(target), 1).unwrap();
        encoder
            .dispatch_threads(Size::new(SIDE, SIDE, 1), Size::new(8, 8, 1))
            .unwrap();
        encoder.end_encoding().unwrap();
    }
    command_buffer.commit().unwrap();
    assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

#[test]
fn one_wait_serves_every_gated_copy_in_of_a_commit() {
    let scratch = Scratch::new("dirty-pages-waits");
    let host = Host::new();
    let device = device(&scratch.0);
    eprintln!("DIRTY-PAGES: device {:?}", device.name());
    let log = Arc::new(Log::default());
    let Some(buffer) = over_logged(&device, &host, &log) else {
        return;
    };
    host.fill(0x1010_1010);
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    let shadows = device.work_counts().shadowed_textures;
    let first = buffer.new_texture(&descriptor, 0, ROW).unwrap();
    let second = buffer.new_texture(&descriptor, SIDE * ROW, ROW).unwrap();
    if device.work_counts().shadowed_textures - shadows < 2 {
        eprintln!("DIRTY-PAGES: no shadows made");
        return;
    }
    let mut plain = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    plain.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    plain.resource_options = ResourceOptions::shared();
    let target = device.new_texture(&plain).unwrap();
    let invert = state(&device, "invert_texels");
    let sources = [&first, &second];

    // Frame 1: no hold, so no wait.
    let before = waits(&device);
    run_each(&device, &invert, &sources, &target);
    let [stamps, waited, _] = std::array::from_fn(|i| waits(&device)[i] - before[i]);
    eprintln!("DIRTY-PAGES: waits frame 1: {stamps} stamps, {waited} waits");
    assert_eq!(stamps, 1, "one stamp per log and commit");
    assert_eq!(waited, 0, "no hold, no wait");

    // Frame 2: two gated copies over one log, one stamp and one wait.
    let before = waits(&device);
    let gated = counts(&device);
    run_each(&device, &invert, &sources, &target);
    let [stamps, waited, _] = std::array::from_fn(|i| waits(&device)[i] - before[i]);
    let gated = delta(&device, gated);
    eprintln!("DIRTY-PAGES: waits frame 2: {stamps} stamps, {waited} waits, copies {gated:?}");
    assert_eq!(gated[0], 2, "two gated copies in");
    assert_eq!(gated[2], 2, "neither moved a byte");
    assert_eq!(stamps, 1);
    assert_eq!(waited, 1, "one wait for both");

    // Frame 3: the log cannot say: one wait, both copies whole.
    log.unanswered.store(true, Ordering::SeqCst);
    let before = waits(&device);
    let gated = counts(&device);
    run_each(&device, &invert, &sources, &target);
    log.unanswered.store(false, Ordering::SeqCst);
    let [stamps, waited, unanswered] = std::array::from_fn(|i| waits(&device)[i] - before[i]);
    let gated = delta(&device, gated);
    eprintln!("DIRTY-PAGES: waits frame 3: {stamps} stamps, {waited} waits ({unanswered} false)");
    assert_eq!(gated[1], 2, "both copies whole");
    assert_eq!(
        (stamps, waited, unanswered),
        (1, 1, 1),
        "one unanswered wait for both"
    );
    drop((first, second));
    drop(buffer);
    drop(device);
}

/// One pass that clears `target` and draws a triangle over it sampling `source`
/// (`sampled_fragment`), to completion.
fn draw_sampled(device: &Device, source: &Texture, target: &Texture) {
    let state = render_state(
        device,
        "examples/runtime_examples.ll",
        "colored_vertex",
        "sampled_fragment",
    );
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(target.clone());
    attachment.load_action = LoadAction::Clear;
    attachment.store_action = StoreAction::Store;
    attachment.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(&state).unwrap();
    let positions = bytes_of(&[-1.0f32, -1.0, 3.0, -1.0, -1.0, 3.0], f32::to_le_bytes);
    let colors = bytes_of(&[1.0f32; 12], f32::to_le_bytes);
    encoder.set_vertex_bytes(&positions, 0).unwrap();
    encoder.set_vertex_bytes(&colors, 1).unwrap();
    encoder.set_fragment_texture(Some(source), 0).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

#[test]
fn a_logged_shadow_sampled_in_an_inline_pass_is_copied_in_before_the_pass() {
    let usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    sampled_in_an_inline_pass("dirty-pages-inline", usage);
}

#[test]
fn a_logged_device_copy_sampled_in_an_inline_pass_is_copied_in_before_the_pass() {
    sampled_in_an_inline_pass("dirty-pages-inline-copy", TextureUsage::SHADER_READ);
}

/// A texture of `usage` over logged pages, sampled by a pass recorded inline, its copies in
/// before the pass (the module docs); nothing where the device reads it in place.
fn sampled_in_an_inline_pass(name: &str, usage: TextureUsage) {
    let scratch = Scratch::new(name);
    let host = Host::new();
    let device = device(&scratch.0);
    eprintln!("DIRTY-PAGES: device {:?}", device.name());
    let log = Arc::new(Log::default());
    let Some(buffer) = over_logged(&device, &host, &log) else {
        return;
    };
    host.fill(0x4040_4040);
    let made = |device: &Device| {
        let c = device.work_counts();
        [c.shadowed_textures, c.device_copy_textures]
    };
    let before = made(&device);
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = usage;
    let source = buffer.new_texture(&descriptor, 0, ROW).unwrap();
    let [shadowed, copied] = std::array::from_fn(|i| made(&device)[i] - before[i]);
    eprintln!("DIRTY-PAGES: {usage:?}: {shadowed} shadow, {copied} device copy made");
    if shadowed == 0 && copied == 0 {
        return;
    }
    let mut plain = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    plain.usage = TextureUsage::RENDER_TARGET;
    plain.resource_options = ResourceOptions::shared();
    let target = device.new_texture(&plain).unwrap();
    let read = |row: u64| {
        let mut bytes = [0u8; 4];
        (target.get_bytes(&mut bytes, 4, 0, Region::new_2d(SIDE / 2, row, 1, 1), 0, 0)).unwrap();
        bytes
    };
    let inline = |device: &Device| {
        let c = device.work_counts();
        [c.inline_passes, c.inline_passes_split]
    };

    // Frame 1: the first copy in, every row.
    let (before, passes) = (counts(&device), inline(&device));
    draw_sampled(&device, &source, &target);
    let first = delta(&device, before);
    let passes = std::array::from_fn::<u64, 2, _>(|i| inline(&device)[i] - passes[i]);
    eprintln!("DIRTY-PAGES: inline frame 1 {first:?}, inline passes (all, split) {passes:?}");
    assert_eq!(first[0], 1, "the copy in is gated");
    assert_eq!(passes, [1, 0], "the pass is one instance recorded inline");
    assert_eq!(read(10), [0x40; 4], "the draw read the CPU's bytes");

    // Frame 2: every row written and told to the log.
    host.fill(0x8080_8080);
    log.told(0..SIDE * ROW, 0);
    let (before, passes) = (counts(&device), inline(&device));
    draw_sampled(&device, &source, &target);
    let second = delta(&device, before);
    let passes = std::array::from_fn::<u64, 2, _>(|i| inline(&device)[i] - passes[i]);
    eprintln!("DIRTY-PAGES: inline frame 2 {second:?}, inline passes (all, split) {passes:?}");
    assert_eq!(second[0], 1, "the copy in is gated");
    assert_eq!(second[4], SIDE * SIDE * 4, "every row the log told of");
    assert_eq!(passes, [1, 0], "the pass is one instance recorded inline");
    assert_eq!(read(10), [0x80; 4], "the draw read the CPU's new bytes");
    drop(source);
    drop(buffer);
    drop(device);
}
