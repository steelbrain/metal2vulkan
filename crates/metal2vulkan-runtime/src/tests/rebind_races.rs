//! Rebinds of a no-copy buffer racing encoders on other threads (`rebind.rs`, **Other threads**).
//!
//! - **A read before the hold.** A dispatch reads the bound buffer's pages (its bind state)
//!   before its encoder holds the buffer. Held at the hold (`rebind::pause`, `Point::Holding`)
//!   while another thread rebinds the buffer whole, the hold then finds the new pages, where the
//!   dispatch's commands named the old ones. The call raced the rebind
//!   (`WorkCounts::encoders_raced_rebind`): the commit encodes the encoder again, and the dispatch
//!   reads the new pages. A second buffer over the old pages keeps their import, so a tree without
//!   the check reads the old pages' values, not a freed `VkBuffer`, and fails the assertion.
//! - **A read during the publish.** A rebind held after the buffer took its new pages, before
//!   the texture over its bytes has its image made over them (`Point::Publishing`), while a
//!   dispatch on another thread reads the texture: the dispatch's hold finds the publish open,
//!   and the commit, after the publish closed, encodes the encoder again over the new image.
//! - **Threads.** Four threads encode and commit command buffers whose commands read the buffer,
//!   one family each: a dispatch reading a texture over it, a draw reading it as a vertex buffer,
//!   a blit copying from it, and a dispatch it was bound to a call before (the dispatch's bind
//!   state reuses nothing a move changed, `buffer::host_cells_moved`). A fifth thread rebinds it
//!   to fresh pages meanwhile, as often as it is let. Every page holds one value in every word, so
//!   each command buffer's output is wholly one page's, of a page the buffer lay in between the
//!   start of the encoding and the completion, or the command buffer failed `EncodedUseRebound`.
//!   It prints the rebinds made and refused, the encoders that raced and the failures.
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`). The functions
//! are `scale_and_offset`, `invert_texels` and `colored_vertex` of the owned fixture
//! `validation/fixtures/public/examples/runtime_examples.ll`.

use super::blit_holds::{device, Scratch, WAIT};
use super::dispatch_pending::state as scale_state;
use super::encoder_holds::{bits, bytes_of, library, pass, render_state, words_of};
use crate::error::{invalid, Problem};
use crate::rebind::pause::{self, Point};
use crate::{
    BlitOption, Buffer, CommandBuffer, CommandBufferError, CommandBufferStatus,
    ComputePipelineState, Device, Error, HostMemory, Origin, PipelineOptions, RebindProblem,
    Region, RenderPipelineState, ResourceOptions, Size, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// The buffer's bytes: whole pages of any host.
const BYTES: usize = 64 << 10;

/// The texture over the buffer's bytes: `RGBA8Unorm`, `SIDE` texels square, rows tight.
const SIDE: u64 = 64;

/// The threads of `scale_and_offset`, one threadgroup, and the scale it multiplies by.
const COUNT: u64 = 64;
const SCALE: f32 = 2.0;

/// The side of a draw's target, and the bytes a blit copies.
const TARGET: u64 = 16;
const COPIED: u64 = 4096;

/// The command buffers each encoding thread commits, and the pages the rebinding thread takes
/// (fewer than 255, so each page's value has its own `RGBA8Unorm` code).
const ROUNDS: usize = 25;
const PAGES: usize = 64;

/// The value page `tag` holds in every word: `(tag + 1) / 255`, whose `RGBA8Unorm` code is
/// `tag + 1`.
fn value(tag: usize) -> f32 {
    (tag + 1) as f32 / 255.0
}

/// Host memory of [`BYTES`], every word [`value`]`(tag)`'s bits, freed when dropped.
struct Page(NonNull<u8>);

// SAFETY: its bytes are written in `new` alone, before any other thread is handed it; only the
// GPU reads them after.
unsafe impl Send for Page {}
// SAFETY: as above.
unsafe impl Sync for Page {}

impl Page {
    fn layout() -> Layout {
        Layout::from_size_align(BYTES, BYTES).unwrap()
    }

    fn new(tag: usize) -> Page {
        // SAFETY: the layout's size is not zero.
        let pointer = NonNull::new(unsafe { std::alloc::alloc(Page::layout()) }).unwrap();
        // SAFETY: `BYTES` bytes, allocated just now aligned past a word.
        let words = unsafe { std::slice::from_raw_parts_mut(pointer.as_ptr().cast(), BYTES / 4) };
        words.fill(value(tag).to_bits());
        Page(pointer)
    }

    /// Its memory, for a no-copy buffer or a rebind.
    ///
    /// # Safety
    ///
    /// It outlives every buffer over it, and nothing writes it.
    unsafe fn memory(&self) -> HostMemory {
        // SAFETY: the caller's contract; `BYTES` bytes, allocated in `new` page-aligned.
        unsafe { HostMemory::new(self.0) }
    }
}

impl Drop for Page {
    fn drop(&mut self) {
        // SAFETY: allocated with this layout in `Page::new`.
        unsafe { std::alloc::dealloc(self.0.as_ptr(), Page::layout()) };
    }
}

/// A no-copy buffer over `page`; `None`, printed, where the device imports no host memory.
fn over(device: &Device, page: &Page) -> Option<Buffer> {
    // SAFETY: each page outlives the device (dropped after it), and nothing writes it.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(
            page.memory(),
            BYTES as u64,
            ResourceOptions::shared(),
            None,
        )
    };
    match made {
        Ok(buffer) => Some(buffer),
        Err(Error::Unsupported(gap)) => {
            eprintln!("REBIND-RACES: the device imports no host memory: {gap:?}");
            None
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    }
}

/// Rebind `buffer` to `page`.
fn rebind(buffer: &Buffer, page: &Page) -> Result<(), Error> {
    // SAFETY: as in `over`; no other thread rebinds or detaches the buffer, or makes a texture
    // over it, during the call.
    unsafe { buffer.rebind_bytes_no_copy(page.memory(), None, None) }
}

/// What a command buffer's commands do with the buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    /// `invert_texels` reads the texture over its bytes.
    TextureRead,
    /// `colored_vertex` reads its colors from it, bound at vertex buffer 1.
    VertexBuffer,
    /// A blit copies from it.
    BlitCopy,
    /// `scale_and_offset` reads it, bound a call before the dispatch.
    BoundBefore,
}

const FAMILIES: [Family; 4] = [
    Family::TextureRead,
    Family::VertexBuffer,
    Family::BlitCopy,
    Family::BoundBefore,
];

/// The states, and the objects of the device's own, the families' commands use.
struct Fixtures {
    scale: ComputePipelineState,
    invert: ComputePipelineState,
    draw: RenderPipelineState,
    scale_bytes: Buffer,
    positions: Buffer,
}

impl Fixtures {
    fn new(device: &Device) -> Fixtures {
        let invert = library(device, "examples/runtime_examples.ll")
            .new_function_with_name("invert_texels")
            .unwrap();
        // A triangle over the whole target.
        let triangle = bits(&[-1.0, -1.0, 3.0, -1.0, -1.0, 3.0]);
        Fixtures {
            scale: scale_state(device),
            invert: device
                .new_compute_pipeline_state_with_function(&invert, &PipelineOptions::default())
                .unwrap(),
            draw: render_state(
                device,
                "examples/runtime_examples.ll",
                "colored_vertex",
                "colored_fragment",
            ),
            scale_bytes: device
                .new_buffer_with_bytes(&SCALE.to_le_bytes(), ResourceOptions::shared())
                .unwrap(),
            positions: device
                .new_buffer_with_bytes(
                    &bytes_of(&triangle, u32::to_le_bytes),
                    ResourceOptions::shared(),
                )
                .unwrap(),
        }
    }
}

/// The texture over `buffer`'s first bytes that `invert_texels` reads.
fn texture_over(buffer: &Buffer) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    buffer.new_texture(&descriptor, 0, SIDE * 4).unwrap()
}

/// Where a family's command buffer leaves its output.
enum Output {
    Texture(Texture),
    Buffer(Buffer),
}

impl Output {
    /// Its words, once its command buffer completed.
    fn words(&self) -> Vec<u32> {
        match self {
            Output::Buffer(buffer) => words_of(buffer),
            Output::Texture(texture) => {
                let mut bytes = vec![0; (SIDE * SIDE * 4) as usize];
                let region = Region::new_2d(0, 0, SIDE, SIDE);
                texture
                    .get_bytes(&mut bytes, SIDE * 4, 0, region, 0, 0)
                    .unwrap();
                let (words, _) = bytes.as_chunks::<4>();
                words.iter().map(|word| u32::from_le_bytes(*word)).collect()
            }
        }
    }
}

/// The output `family`'s commands leave over a buffer whose bytes are page `tag`'s.
fn expected(family: Family, tag: usize) -> Vec<u32> {
    let word = value(tag).to_bits();
    match family {
        Family::TextureRead => {
            let [r, g, b, a] = word.to_le_bytes();
            let inverted = u32::from_le_bytes([255 - r, 255 - g, 255 - b, a]);
            vec![inverted; (SIDE * SIDE) as usize]
        }
        Family::VertexBuffer => {
            let code = (tag + 1) as u8;
            vec![u32::from_le_bytes([code; 4]); (TARGET * TARGET) as usize]
        }
        Family::BlitCopy => vec![word; (COPIED / 4) as usize],
        Family::BoundBefore => (0..COUNT)
            .map(|i| (value(tag) * SCALE + i as f32).to_bits())
            .collect(),
    }
}

/// Encode `family`'s commands over `buffer` (and `over`, the texture over its bytes) into
/// `command_buffer`, not committed.
fn encode(
    family: Family,
    device: &Device,
    fixtures: &Fixtures,
    command_buffer: &CommandBuffer,
    (buffer, over): (&Buffer, &Texture),
) -> Output {
    match family {
        Family::TextureRead => {
            let mut descriptor =
                TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
            descriptor.usage = TextureUsage::SHADER_WRITE;
            descriptor.resource_options = ResourceOptions::shared();
            let out = device.new_texture(&descriptor).unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder
                .set_compute_pipeline_state(&fixtures.invert)
                .unwrap();
            encoder.set_texture(Some(over), 0).unwrap();
            encoder.set_texture(Some(&out), 1).unwrap();
            encoder
                .dispatch_threads(Size::new(SIDE, SIDE, 1), Size::new(8, 8, 1))
                .unwrap();
            encoder.end_encoding().unwrap();
            Output::Texture(out)
        }
        Family::VertexBuffer => {
            let mut descriptor =
                TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, TARGET, TARGET, false);
            descriptor.usage = TextureUsage::RENDER_TARGET;
            descriptor.resource_options = ResourceOptions::private();
            let target = device.new_texture(&descriptor).unwrap();
            let back = device
                .new_buffer(TARGET * TARGET * 4, ResourceOptions::shared())
                .unwrap();
            let mut encoder = command_buffer
                .render_command_encoder(&pass(&target))
                .unwrap();
            encoder.set_render_pipeline_state(&fixtures.draw).unwrap();
            encoder
                .set_vertex_buffer(Some(&fixtures.positions), 0, 0)
                .unwrap();
            encoder.set_vertex_buffer(Some(buffer), 0, 1).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
            encoder.end_encoding().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_from_texture_to_buffer(
                &target,
                0,
                0,
                Origin::new(0, 0, 0),
                Size::new(TARGET, TARGET, 1),
                &back,
                0,
                TARGET * 4,
                TARGET * TARGET * 4,
                BlitOption::None,
            )
            .unwrap();
            blit.end_encoding().unwrap();
            Output::Buffer(back)
        }
        Family::BlitCopy => {
            let out = device
                .new_buffer(COPIED, ResourceOptions::shared())
                .unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_from_buffer_to_buffer(buffer, 0, &out, 0, COPIED)
                .unwrap();
            blit.end_encoding().unwrap();
            Output::Buffer(out)
        }
        Family::BoundBefore => {
            let out = device
                .new_buffer(COUNT * 4, ResourceOptions::shared())
                .unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            // Bound in its own call, before the state and the dispatch.
            encoder.set_buffer(Some(buffer), 0, 0).unwrap();
            encoder.set_compute_pipeline_state(&fixtures.scale).unwrap();
            encoder.set_buffer(Some(&out), 0, 1).unwrap();
            encoder
                .set_buffer(Some(&fixtures.scale_bytes), 0, 2)
                .unwrap();
            encoder
                .dispatch_threads(Size::new(COUNT, 1, 1), Size::new(COUNT, 1, 1))
                .unwrap();
            encoder.end_encoding().unwrap();
            Output::Buffer(out)
        }
    }
}

/// The encoders that raced a rebind, over the device's life.
fn raced(device: &Device) -> u64 {
    device.work_counts().encoders_raced_rebind
}

/// A dispatch held at its hold of the buffer, after it read the buffer's pages, while the main
/// thread rebinds the buffer: it raced the rebind, and its commit encodes it again over the new
/// pages.
#[test]
fn a_dispatch_that_read_the_buffer_before_its_hold_runs_over_the_new_pages() {
    let scratch = Scratch::new("rebind-races-hold");
    let device = device(&scratch.0);
    let fixtures = Fixtures::new(&device);
    let pages = [Page::new(0), Page::new(1)];
    let Some(buffer) = over(&device, &pages[0]) else {
        return;
    };
    // Over the same bytes: it lies in the buffer's import, which it keeps after the rebind.
    let keeper = over(&device, &pages[0]).unwrap();
    let over_texture = texture_over(&buffer);
    let queue = device.new_command_queue().unwrap();
    let before = raced(&device);
    let (reached, go) = pause::arm(Point::Holding, buffer.identity().key);
    let (output, command_buffer) = std::thread::scope(|scope| {
        let encoding = scope.spawn(|| {
            let command_buffer = queue.command_buffer().unwrap();
            let family = Family::BoundBefore;
            let output = encode(
                family,
                &device,
                &fixtures,
                &command_buffer,
                (&buffer, &over_texture),
            );
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            (output, command_buffer)
        });
        reached
            .recv_timeout(WAIT)
            .expect("the dispatch reached its hold");
        rebind(&buffer, &pages[1]).expect("the rebind before the hold");
        go.send(()).unwrap();
        encoding.join().unwrap()
    });
    let raced = raced(&device) - before;
    println!(
        "REBIND-RACES hold: {:?} {:?}, raced {raced}",
        command_buffer.status(),
        command_buffer.error()
    );
    assert_eq!(
        (command_buffer.status(), command_buffer.error()),
        (CommandBufferStatus::Completed, None)
    );
    let words = output.words();
    assert_ne!(
        words,
        expected(Family::BoundBefore, 0),
        "the dispatch read the pages the buffer left before its submission"
    );
    assert_eq!(
        words,
        expected(Family::BoundBefore, 1),
        "the dispatch read the pages the buffer lies in"
    );
    assert_eq!(raced, 1, "the encoder that raced the rebind");
    drop(keeper);
}

/// A dispatch reading the texture over the buffer while a rebind of it is held in its publish,
/// the texture's image still the old pages': its hold finds the publish open, and its commit,
/// after the publish closed, encodes it again over the new image.
#[test]
fn a_dispatch_that_read_a_texture_over_the_buffer_during_its_publish_runs_over_the_new_pages() {
    let scratch = Scratch::new("rebind-races-publish");
    let device = device(&scratch.0);
    let fixtures = Fixtures::new(&device);
    let pages = [Page::new(0), Page::new(1)];
    let Some(buffer) = over(&device, &pages[0]) else {
        return;
    };
    let over_texture = texture_over(&buffer);
    let queue = device.new_command_queue().unwrap();
    let before = raced(&device);
    let (reached, go) = pause::arm(Point::Publishing, buffer.identity().key);
    let command_buffer = queue.command_buffer().unwrap();
    let output = std::thread::scope(|scope| {
        let rebinding = scope.spawn(|| rebind(&buffer, &pages[1]));
        reached
            .recv_timeout(WAIT)
            .expect("the rebind reached its publish");
        let output = encode(
            Family::TextureRead,
            &device,
            &fixtures,
            &command_buffer,
            (&buffer, &over_texture),
        );
        go.send(()).unwrap();
        rebinding.join().unwrap().expect("the rebind");
        output
    });
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let raced = raced(&device) - before;
    println!(
        "REBIND-RACES publish: {:?} {:?}, raced {raced}",
        command_buffer.status(),
        command_buffer.error()
    );
    assert_eq!(
        (command_buffer.status(), command_buffer.error()),
        (CommandBufferStatus::Completed, None)
    );
    assert_eq!(
        output.words(),
        expected(Family::TextureRead, 1),
        "the dispatch read the texture over the pages the buffer lies in"
    );
    assert_eq!(raced, 1, "the encoder that raced the rebind");
}

/// What one command buffer of the threaded run came to.
#[derive(Debug)]
enum Seen {
    /// Its output is wholly page `tag`'s.
    Page(usize),
    /// It failed `EncodedUseRebound`.
    Rebound,
    /// Anything else: a wrong answer.
    Wrong(String),
}

/// Four threads encoding and committing command buffers that read the buffer, a family each,
/// while a fifth rebinds it to fresh pages: every output is wholly one page's, of a page the
/// buffer lay in meanwhile, or its command buffer failed `EncodedUseRebound`.
#[test]
fn encoders_on_other_threads_read_wholly_one_page_of_a_buffer_rebound_meanwhile() {
    let scratch = Scratch::new("rebind-races-threads");
    let device = device(&scratch.0);
    let fixtures = Fixtures::new(&device);
    let pages: Vec<Page> = (0..PAGES).map(Page::new).collect();
    let Some(buffer) = over(&device, &pages[0]) else {
        return;
    };
    let over_texture = texture_over(&buffer);
    // One queue for every thread: a test's device takes two `VkQueue`s at most.
    let queue = device.new_command_queue().unwrap();
    let before = device.work_counts();
    // The page the buffer lies in, stored after each rebind returns.
    let current = AtomicUsize::new(0);
    let seen: Mutex<Vec<(Family, Seen)>> = Mutex::new(Vec::new());
    let encoded_uses = invalid(
        "rebind_bytes_no_copy",
        Problem::Rebind(RebindProblem::EncodedUses),
    );
    let (made, refused) = std::thread::scope(|scope| {
        for family in FAMILIES {
            let (device, fixtures, buffer, over_texture) =
                (&device, &fixtures, &buffer, &over_texture);
            let (current, seen, queue) = (&current, &seen, &queue);
            scope.spawn(move || {
                for _ in 0..ROUNDS {
                    let low = current.load(Ordering::Acquire);
                    let command_buffer = queue.command_buffer().unwrap();
                    let output = encode(
                        family,
                        device,
                        fixtures,
                        &command_buffer,
                        (buffer, over_texture),
                    );
                    command_buffer.commit().unwrap();
                    command_buffer.wait_until_completed_for(WAIT).unwrap();
                    // A rebind's replace comes before its call returns and the page is stored.
                    let high = current.load(Ordering::Acquire) + 1;
                    let outcome = match (command_buffer.status(), command_buffer.error()) {
                        (CommandBufferStatus::Completed, None) => {
                            let words = output.words();
                            match (low..=high.min(PAGES - 1))
                                .find(|&tag| words == expected(family, tag))
                            {
                                Some(tag) => Seen::Page(tag),
                                None => Seen::Wrong(format!(
                                    "no page {low}..={high}'s output: first words {:x?}",
                                    &words[..4]
                                )),
                            }
                        }
                        (
                            CommandBufferStatus::Error,
                            Some(CommandBufferError::EncodedUseRebound),
                        ) => Seen::Rebound,
                        other => Seen::Wrong(format!("{other:?}")),
                    };
                    seen.lock().unwrap().push((family, outcome));
                }
            });
        }
        let rebinding = scope.spawn(|| {
            let (mut made, mut refused) = (0, 0);
            let mut next = 1;
            while next < PAGES && seen.lock().unwrap().len() < FAMILIES.len() * ROUNDS {
                match rebind(&buffer, &pages[next]) {
                    Ok(()) => {
                        current.store(next, Ordering::Release);
                        made += 1;
                        next += 1;
                    }
                    Err(error) if error == encoded_uses => refused += 1,
                    Err(error) => panic!("the rebind: {error:?}"),
                }
                std::thread::yield_now();
            }
            (made, refused)
        });
        rebinding.join().unwrap()
    });
    let after = device.work_counts();
    let seen = seen.into_inner().unwrap();
    let count = |family: Family, which: fn(&Seen) -> bool| {
        (seen.iter())
            .filter(|(of, outcome)| *of == family && which(outcome))
            .count()
    };
    for family in FAMILIES {
        let pages: std::collections::BTreeSet<usize> = (seen.iter())
            .filter_map(|(of, outcome)| match outcome {
                Seen::Page(tag) if *of == family => Some(*tag),
                _ => None,
            })
            .collect();
        println!(
            "REBIND-RACES threads {family:?}: {} wholly one page ({} pages apart), {} \
             EncodedUseRebound, {} wrong",
            count(family, |seen| matches!(seen, Seen::Page(_))),
            pages.len(),
            count(family, |seen| matches!(seen, Seen::Rebound)),
            count(family, |seen| matches!(seen, Seen::Wrong(_))),
        );
    }
    let grew = |of: fn(&crate::WorkCounts) -> u64| of(&after) - of(&before);
    println!(
        "REBIND-RACES threads: {made} rebinds made, {refused} refused (EncodedUses); {} encoders \
         raced a rebind, {} encoded again, {} refused encoding again",
        grew(|counts| counts.encoders_raced_rebind),
        grew(|counts| counts.encoders_reencoded),
        grew(|counts| counts.reencodes_refused),
    );
    let wrong: Vec<(Family, &str)> = (seen.iter())
        .filter_map(|(family, outcome)| match outcome {
            Seen::Wrong(why) => Some((*family, why.as_str())),
            _ => None,
        })
        .collect();
    assert_eq!(seen.len(), FAMILIES.len() * ROUNDS);
    assert!(wrong.is_empty(), "wrong outputs: {wrong:?}");
    assert!(made > 0, "no rebind was made");
}
