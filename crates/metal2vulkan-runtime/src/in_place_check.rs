//! The device's check of an image in place over host pages (`in_place.rs`): whether the
//! operations Vulkan lets touch the bytes bound to it beside its texels leave every byte alone
//! while the CPU writes beside it, per class of pages ([`PagesClass`]) and access ([`Access`]), on
//! a thread of its own.
//!
//! - **What it does.** For each class it makes a buffer as the product makes one there (a no-copy
//!   buffer over anonymous pages it maps, imported by their pointer; a buffer of the runtime's own
//!   longer than half a pooled block, in fresh pages; a no-copy buffer over pages the device
//!   refuses by their pointer, imported as dma-bufs), fills a window of it with a pattern, and
//!   makes a texture over it as the program does (`Buffer::new_runtime_texture`): the texture takes
//!   the image in place exactly as the product's would ([`checking`](crate::in_place::checking)
//!   lets it whatever the verdict); the fresh-pages class once the device's submission checks
//!   answered whether its buffers take fresh pages (`submit_check.rs`), unchecked where they do
//!   not, as the product then makes no such buffer. A thread of its own then writes the
//!   pattern's complement into the window, each byte once, at a stride that reaches texels, the
//!   bytes between rows and the bytes after the image from its first writes, while the device
//!   makes the image's first move
//!   (an internal submission naming the texture, waited for) and until it completed. Every byte
//!   must then read the CPU's write where the CPU wrote it, and the pattern elsewhere. Each class
//!   runs every shape of [`SHAPES`] (texels of 1, 2, 4, 8 and 16 bytes, rows apart or one after
//!   another, and 3840x2160 at 4 bytes) [`ROUNDS`] times.
//! - **An image that renders** ([`Access::Renders`]) is checked apart, in rounds of their own: the
//!   texture is made with `RenderTarget` usage as well, as the program's render targets are, so
//!   its image takes the feedback loop usage where the product's does; the writer writes the bytes
//!   between rows and after the image alone, through the image's first move, its moves into
//!   `ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT` and back where it has the usage
//!   (`Texture::settle_feedback_moves`, as `render_target::moved_barriers` records them), and a
//!   render pass of a command queue of the check's own that clears it to zero at its load and
//!   stores it, waited for (each by a submission the round waits for). Every texel must then read
//!   zero, and every other byte as above. A draw's color writes reach its fragments' texels alone
//!   (the Memory Model); the load, store and resolve operations and the layout transitions are
//!   what Vulkan lets touch the rest of the subresource, and the program's resolves into such a
//!   texture are not among the rounds.
//! - **Its buffers' addresses** lie outside the Metal address space (`AddressPlacement::Outside`,
//!   `address.rs`), each made as the product makes its class's otherwise: the check runs while
//!   the program makes its resources, so a range of the space it took would refuse a pin of the
//!   program's there, or move a range the runtime picks for it. Its pages, their import, the
//!   image and the move do not depend on the address, so neither does a verdict.
//! - **Its textures take no resource ID** (`IdUse::Runtime`, `resource_ids.rs`): the check runs
//!   on its thread while the program makes its textures, so a plain ID one of its textures held
//!   would move the program's next texture's ID past Metal's, and a texture reservation after it
//!   would be ignored. The ID does not touch the image, its pages or its move, so neither does a
//!   verdict.
//! - **Its cost** is the pattern's fill and compare of each window, at the speed of a copy and a
//!   comparison of memory (a 3840x2160 surface's 33 MB among them), and each image's making and
//!   first move: the writes themselves run only while the move does.
//! - **Its verdict**, per class: passed where every image it made kept every byte, failed where
//!   one did not (or the device was made to fail it, `DeviceOptions::fail_in_place_check`), and
//!   unchecked where it made none (the device makes no such buffer, image or import). Each class's
//!   is the device's as it is known, so a class that passed takes textures in place while the
//!   next is checked. Counted (`WorkCounts::in_place_checks_*`).
//! - **Kept.** Once every class answered, the verdicts are written as one record in the shader
//!   cache's content directory (`shader_cache/store.rs`), named by the driver's identity, the
//!   check's version and the runtime's ([`record_name`]): a device of the same driver and release
//!   reads them as it is made and runs no check. A failed write leaves the verdicts for the
//!   device's life. A forced failure is kept nowhere.
//! - **Its thread.** Each round's Vulkan work (the buffer, the texture, the submission and its
//!   wait, the releases) runs inside a section of the exit guard (`exit_guard.rs`), as every
//!   runtime thread's does. It holds its device through the registry of the runtime's own holds
//!   for each class (`device/holds.rs`), weakly between classes, and reads the stop before each
//!   round: the program's last drop stops it there and waits for its round in progress and its
//!   release of its buffer, so the device goes on the program's thread, and the check ends
//!   unrecorded.
//! - **Not where the driver writes over the device's loader word.** On a device whose word the
//!   runtime found overwritten (`loader_word.rs`: MoltenVK with each activity logged at once
//!   names the device in each message), the check runs no round, or no further one, and each
//!   class it has not answered stays unchecked for the device's life, recorded nowhere, so its
//!   textures over host pages take the shadow. Its rounds make their buffers and textures through
//!   the product's paths, which take locks the program's threads hold while they make calls the
//!   loader serves, so the word cannot be held across a round as the device's other checks hold
//!   it across their steps; and with the check beside the program's own submissions, a queue
//!   waiter's `vkAllocateCommandBuffers` faulted on the magic (an M3 Max under MoltenVK 1.4.2,
//!   one run of three), where the same work with no check running did not in 30 000 rounds.
//! - **Its dma-bufs.** The runtime runs no `udmabuf` code, so the dma-buf class's pages are an
//!   allocation of the device's own, host-visible and exported as a dma-buf, mapped: the device
//!   refuses their pointer (a driver that takes it leaves the class unchecked), and the check's
//!   source answers each request over them with that dma-buf. The image and its first move are
//!   the product's.

use crate::device::holds::Holder;
use crate::exit_guard;
use crate::host_memory::HostMemory;
use crate::in_place::{encode, mark_checking, record_name, Access, PagesClass, Verdict};
use crate::values::ResourceOptions;
use crate::{
    AddressPlacement, Buffer, BufferContents, BufferDescriptor, ClearColor, CommandQueue, Device,
    LoadAction, RenderPassDescriptor, StoreAction, Texture, TextureDescriptor,
};
use metal2vulkan::device::HostPageTexturePolicy;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Barrier;
use std::time::{Duration, Instant};

/// How many times each shape is checked in each class.
pub(crate) const ROUNDS: usize = 3;

/// The longest a render round waits for its pass's command buffer: a pass that clears a 3840x2160
/// surface and stores it.
const PASS_WAIT: Duration = Duration::from_secs(20);

/// A shape the check makes a texture of: its format, the bytes of one texel, its width and height,
/// and its rows' pitch.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Shape {
    pub(crate) format: PixelFormat,
    pub(crate) texel: u64,
    pub(crate) width: u64,
    pub(crate) height: u64,
    pub(crate) bytes_per_row: u64,
}

/// The shapes each class is checked with: each texel size the formats a buffer backs have (1, 2,
/// 4, 8 and 16 bytes), with rows one after another (a linear image made for the memory, which
/// every driver lays at a 256-byte pitch) and with bytes between them (an image at an explicit
/// pitch, where the device makes one: `texture/in_place_image.rs`), and one 3840x2160 surface of
/// 4-byte texels, last. A shape the device makes no image of in place is skipped: the product
/// makes none of it either.
pub(crate) const SHAPES: [Shape; 11] = [
    shape(PixelFormat::R8Unorm, 1, 256, 16, 256),
    shape(PixelFormat::RG8Unorm, 2, 128, 16, 256),
    shape(PixelFormat::BGRA8Unorm, 4, 64, 16, 256),
    shape(PixelFormat::RGBA16Float, 8, 32, 16, 256),
    shape(PixelFormat::RGBA32Float, 16, 16, 16, 256),
    shape(PixelFormat::R8Unorm, 1, 61, 37, 256),
    shape(PixelFormat::RG8Unorm, 2, 45, 29, 256),
    shape(PixelFormat::BGRA8Unorm, 4, 100, 50, 512),
    shape(PixelFormat::RGBA16Float, 8, 33, 17, 512),
    shape(PixelFormat::RGBA32Float, 16, 20, 9, 512),
    shape(PixelFormat::BGRA8Unorm, 4, 3840, 2160, 15360),
];

const fn shape(format: PixelFormat, texel: u64, width: u64, height: u64, bpr: u64) -> Shape {
    Shape {
        format,
        texel,
        width,
        height,
        bytes_per_row: bpr,
    }
}

impl Shape {
    /// Whether the window's byte at `offset` is one of the texture's texels.
    fn is_texel(&self, offset: u64) -> bool {
        offset < self.bytes_per_row * self.height
            && offset % self.bytes_per_row < self.width * self.texel
    }

    /// The bytes of the window the check fills and writes: the texture's rows, then the bytes
    /// after them to two pages past the page their last byte is in.
    fn window(&self, page: u64) -> u64 {
        let rows = self.bytes_per_row * self.height;
        rows.next_multiple_of(page) + 2 * page
    }
}

/// The bytes of each class's buffer: past half a pooled block, so that a buffer of the runtime's
/// own lies in fresh pages (`fresh_pages.rs`), and room for the largest shape's window.
const BUFFER_BYTES: u64 = 40 << 20;

impl Device {
    /// Start the device's check on a thread of its own, where it has not started and its policy
    /// takes textures over host pages in place (`in_place.rs`). Where the thread cannot start,
    /// every verdict still pending is unchecked.
    pub(crate) fn start_in_place_checks(&self) {
        if self.compile_device().policy().host_page_textures != HostPageTexturePolicy::InPlace {
            return;
        }
        let checks = &self.shared().in_place;
        if !checks.start() {
            return;
        }
        let weak = self.downgrade();
        let builder = std::thread::Builder::new().name("m2v-in-place-check".to_owned());
        let started = exit_guard::spawn(builder, move || run(weak));
        if started.is_err() {
            for class in PagesClass::ALL {
                for access in Access::ALL {
                    if checks.verdict(class, access) == Verdict::Pending {
                        checks.set(class, access, Verdict::Unchecked);
                    }
                }
            }
            checks.settle(None);
        }
    }
}

/// The check's thread: each class in turn, each of its accesses, each verdict given as it is
/// known, then the record.
fn run(weak: crate::device::WeakDevice) {
    mark_checking();
    let started = Instant::now();
    let mut stopped = false;
    for class in PagesClass::ALL {
        for access in Access::ALL {
            let Some(device) = weak.hold() else {
                return;
            };
            if device.shared().in_place.verdict(class, access) != Verdict::Pending {
                continue;
            }
            if stopped || word_overwritten(&device) {
                stopped = true;
                device
                    .shared()
                    .in_place
                    .set(class, access, Verdict::Unchecked);
                continue;
            }
            let Some(verdict) = check_class(&device, class, access) else {
                // The process is exiting, or the program let the device go: nothing is kept.
                return;
            };
            let checks = &device.shared().in_place;
            checks.set(class, access, verdict);
            device.shared().counts.in_place_check_ran(verdict);
        }
    }
    let Some(device) = weak.hold() else {
        return;
    };
    let checks = &device.shared().in_place;
    let verdicts = checks.verdicts();
    let stopped = stopped || word_overwritten(&device);
    if stopped {
        checks.note(
            "not run, or not to its end: the device's loader word was found overwritten (the \
             driver names the device in its messages), which the check's rounds cannot hold"
                .to_owned(),
        );
    }
    if !checks.forced_failure() && !stopped && !verdicts.contains(&Verdict::Pending) {
        let identity = device.driver_identity();
        let bytes = encode(&identity, verdicts);
        // A failed write leaves the verdicts to the device's life alone.
        let _ = device
            .shared()
            .shader_cache
            .content_store()
            .write_record_soon(&record_name(&identity), bytes);
    }
    checks.settle(Some(started.elapsed()));
}

/// Whether the runtime found `device`'s loader word overwritten (`loader_word.rs`): the check then
/// runs no further round (the module docs).
fn word_overwritten(device: &Device) -> bool {
    // SAFETY: the device is live while `device` is.
    unsafe { device.shared().loader_word.overwritten() }
}

/// The verdict of `class` for `access` on `device`: every shape's rounds, until one fails, or
/// until the device's loader word is found overwritten (unchecked then where no round ran).
/// `None` where the process is exiting, or the program's last drop stopped the device's holds
/// before a round (`device/holds.rs`).
fn check_class(device: &Device, class: PagesClass, access: Access) -> Option<Verdict> {
    let page = host_page();
    let Some(pages) = ClassPages::make(device, class, BUFFER_BYTES) else {
        return Some(Verdict::Unchecked);
    };
    // The command queue an image that renders takes its passes on: the check's own.
    let queue = match access {
        Access::Reads => None,
        Access::Renders => {
            let made = exit_guard::enter().map(|_busy| device.new_command_queue());
            match made? {
                Ok(queue) => Some(queue),
                Err(_) => return Some(Verdict::Unchecked),
            }
        }
    };
    let mut made = 0;
    for shape in SHAPES {
        if shape.window(page) > BUFFER_BYTES {
            continue;
        }
        for round in 0..ROUNDS {
            if !device.holds_go_on(Holder::InPlaceCheck) {
                return None;
            }
            if word_overwritten(device) {
                return Some(Verdict::Unchecked);
            }
            let outcome = {
                let _busy = exit_guard::enter()?;
                check_round(&pages, shape, round, page, queue.as_ref())
            };
            match outcome {
                Round::Kept => made += 1,
                Round::NotMade => break,
                Round::Changed((offset, due, read)) => {
                    let image = shape.bytes_per_row * shape.height;
                    let place = match offset.checked_rem(shape.bytes_per_row) {
                        _ if offset >= image => "after the image",
                        Some(column) if column >= shape.width * shape.texel => "between two rows",
                        _ => "a texel",
                    };
                    let operations = match access {
                        Access::Reads => "the first move",
                        Access::Renders => {
                            "the first move, feedback loop moves or a pass's clear and store"
                        }
                    };
                    device.shared().in_place.note(format!(
                        "{class:?}: {operations} of a {:?} image of {}x{} with rows {} bytes \
                         apart changed byte {offset} ({place}) from {due:#04x} to {read:#04x} in \
                         round {round}",
                        shape.format, shape.width, shape.height, shape.bytes_per_row
                    ));
                    return Some(Verdict::Failed);
                }
            }
        }
    }
    let forced = device.shared().in_place.forced_failure();
    Some(match (made, forced) {
        (0, _) => Verdict::Unchecked,
        (_, true) => Verdict::Failed,
        (_, false) => Verdict::Passed,
    })
}

/// What one round found.
enum Round {
    /// The image's move kept every byte.
    Kept,
    /// The texture took no image in place (the device makes none of the shape there).
    NotMade,
    /// A byte read other than the CPU left it: where, what was due, what was read.
    Changed((u64, u8, u8)),
}

/// One round of `shape` over `pages`, inside the caller's section: the pattern, the texture, the
/// writes beside its image's first move (and, for an image that renders, `queue` its queue, beside
/// its feedback loop moves and a pass that clears and stores it), and the bytes compared.
fn check_round(
    pages: &ClassPages,
    shape: Shape,
    round: usize,
    page: u64,
    queue: Option<&CommandQueue>,
) -> Round {
    let window = shape.window(page);
    let bytes = Bytes {
        start: pages.start,
        length: window,
    };
    let pattern = Pattern::new(round as u64 + 1);
    bytes.fill(&pattern);
    let mut descriptor =
        TextureDescriptor::texture_2d(shape.format, shape.width, shape.height, false);
    descriptor.usage = match queue {
        None => TextureUsage::SHADER_READ,
        Some(_) => TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
    };
    // An image that renders: the writer writes beside its texels alone, which its pass writes.
    let texels = queue.map(|_| shape);
    // The runtime's own texture: it takes no resource ID, so the program's IDs are Metal's
    // whether or when the check runs.
    let Ok(texture) = pages
        .buffer
        .new_runtime_texture(&descriptor, 0, shape.bytes_per_row)
    else {
        return Round::NotMade;
    };
    let stop = AtomicBool::new(false);
    let started = Barrier::new(2);
    let written = AtomicU64::new(0);
    let settled = std::thread::scope(|scope| {
        let builder = std::thread::Builder::new().name("m2v-in-place-writes".to_owned());
        let writer = exit_guard::spawn_scoped(scope, builder, || {
            bytes.write_until(&pattern, texels, (&stop, &started, &written));
        });
        if writer.is_err() {
            return None;
        }
        started.wait();
        let settled = match texture.settle_in_place() {
            Some(Ok(())) => match queue {
                None => Some(Ok(())),
                Some(queue) => Some(render(queue, &texture)),
            },
            other => other,
        };
        stop.store(true, Ordering::Release);
        settled
    });
    let Some(Ok(())) = settled else {
        return Round::NotMade;
    };
    drop(texture);
    match bytes.compare(&pattern, texels, written.load(Ordering::Acquire)) {
        None => Round::Kept,
        Some(changed) => Round::Changed(changed),
    }
}

/// An image that renders, after its first move: its moves into the feedback loop layout and back
/// where it has the usage, then a pass of `queue` that clears it to zero at its load and stores
/// it, each waited for. An error where either was not made, or the pass's command buffer failed or
/// did not complete within [`PASS_WAIT`].
fn render(queue: &CommandQueue, texture: &Texture) -> Result<(), crate::Error> {
    if let Some(moved) = texture.settle_feedback_moves() {
        moved?;
    }
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(texture.clone());
    color.load_action = LoadAction::Clear;
    color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
    color.store_action = StoreAction::Store;
    let command_buffer = queue.command_buffer()?;
    command_buffer
        .render_command_encoder(&pass)?
        .end_encoding()?;
    command_buffer.commit()?;
    let failed = || crate::error::internal("the in-place check's pass did not complete");
    command_buffer
        .wait_until_completed_for(PASS_WAIT)
        .map_err(|_| failed())?;
    match command_buffer.error() {
        None => Ok(()),
        Some(_) => Err(failed()),
    }
}

/// The pages of one class's check: its buffer, and the host address of its byte 0.
struct ClassPages {
    buffer: Buffer,
    start: NonNull<u8>,
}

impl ClassPages {
    /// A buffer of `length` bytes in pages of `class` on `device`, where the device makes one
    /// there; inside a section of its own.
    fn make(device: &Device, class: PagesClass, length: u64) -> Option<ClassPages> {
        // Whether the device's buffers take fresh pages is its submission checks' answer
        // (`submit_check.rs`): the class is checked once they gave it, as the product makes its
        // buffers then.
        if matches!(class, PagesClass::Fresh) {
            device.settle_submit_checks(crate::submit_check::ANSWER_WAIT);
        }
        let _busy = exit_guard::enter()?;
        match class {
            PagesClass::Pointer => pointer_pages(device, length),
            PagesClass::Fresh => {
                let buffer = device
                    .new_buffer_placed(length, ResourceOptions::shared(), AddressPlacement::Outside)
                    .ok()?;
                if !buffer.in_fresh_pages() {
                    return None;
                }
                let start = buffer.contents()?;
                Some(ClassPages { buffer, start })
            }
            PagesClass::DmaBuf => dma_buf_pages(device, length),
        }
    }
}

/// A no-copy buffer of `length` bytes over anonymous pages mapped for it, imported by their
/// pointer, which its keep-alive unmaps ([`Anonymous`]). `None` where the device makes none, or
/// does not import the pointer (dma-bufs, or a copied host buffer, `copied.rs`).
#[cfg(unix)]
fn pointer_pages(device: &Device, length: u64) -> Option<ClassPages> {
    let start = map_anonymous(length)?;
    let pages: crate::KeepAlive = Box::new(Anonymous { start, length });
    // SAFETY: the pages are this process's private anonymous mapping, readable and writable,
    // which stays until the keep-alive unmaps it, and which nothing else maps or frees.
    let memory = unsafe { HostMemory::new(start) };
    let buffer = no_copy_kept(device, memory, length, pages)?;
    if buffer.over_dma_buf() || buffer.copied() {
        return None;
    }
    Some(ClassPages { buffer, start })
}

/// A no-copy buffer of `length` bytes over `memory`, whose pages `pages` keeps mapped: a
/// keep-alive, not a deallocator, so the buffer's release runs at Metal's moment and waits for no
/// batch submitted before it (`HostPages`'s release), and `pages` goes once the import is freed
/// (`KeepAlive`). `None` where the device refuses the buffer, `pages` dropped with the refusal.
#[cfg(unix)]
fn no_copy_kept(
    device: &Device,
    memory: HostMemory,
    length: u64,
    pages: crate::KeepAlive,
) -> Option<Buffer> {
    let contents = BufferContents::NoCopy {
        memory,
        deallocator: None,
        keep_alive: Some(pages),
    };
    let descriptor = BufferDescriptor::new(length, ResourceOptions::shared(), contents)
        .with_placement(AddressPlacement::Outside);
    device.new_buffer_with_descriptor(descriptor).ok()
}

/// Anonymous pages [`map_anonymous`] mapped, unmapped as this goes: a check buffer's keep-alive.
#[cfg(unix)]
struct Anonymous {
    start: NonNull<u8>,
    length: u64,
}

// SAFETY: the mapping is this value's alone to unmap, from any thread, once nothing uses it.
#[cfg(unix)]
unsafe impl Send for Anonymous {}

#[cfg(unix)]
impl Drop for Anonymous {
    fn drop(&mut self) {
        unmap(self.start, self.length);
    }
}

/// Anonymous private pages of `length` bytes, readable and writable, read as zero.
#[cfg(unix)]
fn map_anonymous(length: u64) -> Option<NonNull<u8>> {
    // SAFETY: a new private anonymous mapping, at an address the kernel chooses.
    let mapped = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            usize::try_from(length).ok()?,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        )
    };
    if mapped == libc::MAP_FAILED {
        return None;
    }
    NonNull::new(mapped.cast())
}

/// Unmap the `length` bytes `map_anonymous` mapped at `pointer`.
#[cfg(unix)]
fn unmap(pointer: NonNull<u8>, length: u64) {
    // SAFETY: the mapping `map_anonymous` made, which nothing uses any more.
    unsafe { libc::munmap(pointer.as_ptr().cast(), length as usize) };
}

/// No anonymous pages are mapped on this system: the class is unchecked.
#[cfg(not(unix))]
fn pointer_pages(_device: &Device, _length: u64) -> Option<ClassPages> {
    None
}

/// No dma-bufs on this system: the class is unchecked.
#[cfg(not(unix))]
fn dma_buf_pages(_device: &Device, _length: u64) -> Option<ClassPages> {
    None
}

/// A no-copy buffer of `length` bytes over pages the device refuses by their pointer, imported
/// as dma-bufs: an allocation of the device's own, host-visible and exported as a dma-buf, mapped
/// (the module docs).
#[cfg(unix)]
fn dma_buf_pages(device: &Device, length: u64) -> Option<ClassPages> {
    let exported = crate::dma_buf::ExportedPages::make(device, length)?;
    let start = exported.start();
    let source = exported.source();
    // The allocation goes once the buffer's imports are freed: its keep-alive holds it.
    let pages: crate::KeepAlive = Box::new(exported);
    // SAFETY: the pages are the device's mapped allocation, readable and writable, which stays
    // mapped until the keep-alive lets go of it; the source answers every request over them
    // with a dma-buf of exactly those pages (`ExportedPages::source`).
    let memory = unsafe { HostMemory::with_dma_buf(start, source) };
    let buffer = no_copy_kept(device, memory, length, pages)?;
    if !buffer.over_dma_buf() {
        return None;
    }
    Some(ClassPages { buffer, start })
}

/// A window of a class's pages, from its byte 0.
#[derive(Clone, Copy)]
struct Bytes {
    start: NonNull<u8>,
    length: u64,
}

// SAFETY: the window's pages stay mapped while the round uses them (its buffer holds them), and
// the writer thread and the round's thread take turns: the writer writes until it is stopped and
// joined, and the round compares after.
unsafe impl Send for Bytes {}
// SAFETY: as above.
unsafe impl Sync for Bytes {}

/// The stride the writer walks a window at, or the first after it coprime with the window's
/// length ([`Bytes::stride`]): it meets every byte once before it comes back, and its first
/// writes lie far apart, through texels, the bytes between rows and the bytes after the image.
const STRIDE: u64 = 1_000_003;

/// The bytes a round's pattern repeats every: a prime, so that its repeats line up with no row
/// pitch, page or texel, and a byte copied from another row or page reads wrong.
const PERIOD: usize = 4099;

/// A round's pattern: [`PERIOD`] bytes of a hash of their place and the round, repeated over the
/// window. A byte the writer wrote holds its complement, which differs from it in every bit.
struct Pattern {
    bytes: Vec<u8>,
}

impl Pattern {
    /// The pattern of round `salt`.
    fn new(salt: u64) -> Pattern {
        let bytes = (0..PERIOD as u64)
            .map(|offset| {
                let mixed = (offset ^ salt.rotate_left(17)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                (mixed >> 56) as u8
            })
            .collect();
        Pattern { bytes }
    }

    /// Its byte at `offset` of the window.
    fn at(&self, offset: u64) -> u8 {
        self.bytes[(offset % PERIOD as u64) as usize]
    }
}

impl Bytes {
    /// The writer's stride over the window: [`STRIDE`], or the first odd number after it whose
    /// greatest common divisor with the window's length is 1.
    fn stride(&self) -> u64 {
        let coprime = |mut a: u64, mut b: u64| {
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a == 1
        };
        (STRIDE..)
            .step_by(2)
            .find(|&stride| coprime(stride, self.length))
            .unwrap_or(1)
    }

    /// The `n`th byte the writer writes, at `stride`.
    fn nth(&self, n: u64, stride: u64) -> u64 {
        let product = u128::from(n) * u128::from(stride);
        (product % u128::from(self.length)) as u64
    }

    /// The address of the window's byte at `offset`.
    fn byte(&self, offset: u64) -> *mut u8 {
        // SAFETY: the offset lies in the window, whose pages are mapped (the type's contract).
        unsafe { self.start.as_ptr().add(offset as usize) }
    }

    /// Fill the window with `pattern`, a period at a time.
    fn fill(&self, pattern: &Pattern) {
        let mut offset = 0;
        while offset < self.length {
            let count = (self.length - offset).min(PERIOD as u64) as usize;
            // SAFETY: the window's bytes, mapped and writable; nothing else reads or writes them
            // meanwhile, and the pattern's bytes are apart from them.
            unsafe {
                std::ptr::copy_nonoverlapping(pattern.bytes.as_ptr(), self.byte(offset), count)
            };
            offset += count as u64;
        }
    }

    /// Write each byte once, the pattern's complement, at the stride, until `stop` or every byte
    /// is written; `started` once the first is. Where `texels` names the texture's shape, its
    /// texels are passed over (its pass writes them), and only the bytes beside them written. How
    /// many of the stride's bytes were reached goes in `written`.
    fn write_until(
        &self,
        pattern: &Pattern,
        texels: Option<Shape>,
        (stop, started, written): (&AtomicBool, &Barrier, &AtomicU64),
    ) {
        let stride = self.stride();
        let mut n = 0;
        let mut first = true;
        while n < self.length && (first || !stop.load(Ordering::Acquire)) {
            let offset = self.nth(n, stride);
            n += 1;
            if texels.is_some_and(|shape| shape.is_texel(offset)) {
                continue;
            }
            // SAFETY: as in `fill`; the round's thread does not touch the bytes meanwhile.
            unsafe { self.byte(offset).write_volatile(!pattern.at(offset)) };
            if first {
                first = false;
                started.wait();
            }
        }
        if first {
            started.wait();
        }
        written.store(n, Ordering::Release);
    }

    /// The first byte that reads other than the writer and the round left it, where the first
    /// `written` of the stride's bytes were reached: its offset, what was due and what was read;
    /// `None` where every byte reads right. Each written byte must read its complement, and is set
    /// back to the pattern as it is read; then every byte must read the pattern, compared a
    /// period at a time, but, where `texels` names the shape a pass cleared, its texels, which
    /// must read zero.
    fn compare(
        &self,
        pattern: &Pattern,
        texels: Option<Shape>,
        written: u64,
    ) -> Option<(u64, u8, u8)> {
        let stride = self.stride();
        for n in 0..written {
            let offset = self.nth(n, stride);
            if texels.is_some_and(|shape| shape.is_texel(offset)) {
                continue;
            }
            let due = !pattern.at(offset);
            // SAFETY: as in `fill`; the writer was joined.
            let read = unsafe { self.byte(offset).read_volatile() };
            if read != due {
                return Some((offset, due, read));
            }
            // SAFETY: as above.
            unsafe { self.byte(offset).write_volatile(pattern.at(offset)) };
        }
        let Some(shape) = texels else {
            return self.matches(pattern, 0, self.length);
        };
        // Each row's texels read zero, and the bytes after them to the next row the pattern.
        let row_bytes = shape.width * shape.texel;
        for row in 0..shape.height {
            let start = row * shape.bytes_per_row;
            // SAFETY: as above; the row's texels lie in the window.
            let read = unsafe { std::slice::from_raw_parts(self.byte(start), row_bytes as usize) };
            if let Some(at) = read.iter().position(|&read| read != 0) {
                return Some((start + at as u64, 0, read[at]));
            }
            let next = start + shape.bytes_per_row;
            if let Some(changed) = self.matches(pattern, start + row_bytes, next) {
                return Some(changed);
            }
        }
        self.matches(pattern, shape.bytes_per_row * shape.height, self.length)
    }

    /// The first byte from `from` to `to` that does not read the pattern: its offset, what was due
    /// and what was read; `None` where every one does. Compared a period at a time.
    fn matches(&self, pattern: &Pattern, from: u64, to: u64) -> Option<(u64, u8, u8)> {
        let mut offset = from;
        while offset < to {
            let phase = (offset % PERIOD as u64) as usize;
            let count = (to - offset).min((PERIOD - phase) as u64) as usize;
            // SAFETY: as in `compare`; nothing writes the bytes while the slice is read.
            let read = unsafe { std::slice::from_raw_parts(self.byte(offset), count) };
            let due = &pattern.bytes[phase..phase + count];
            if let Some(at) = read.iter().zip(due).position(|(read, due)| read != due) {
                return Some((offset + at as u64, due[at], read[at]));
            }
            offset += count as u64;
        }
        None
    }
}

/// The host's page size.
fn host_page() -> u64 {
    #[cfg(unix)]
    {
        // SAFETY: a query of the page size, with no argument.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if let Ok(page) = u64::try_from(page) {
            if page.is_power_of_two() {
                return page;
            }
        }
    }
    4096
}
