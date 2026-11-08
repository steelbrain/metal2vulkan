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
//!   lets it whatever the verdict); the fresh-pages class's buffer in fresh pages wherever the
//!   device takes them, whatever its submission checks found a held import costs
//!   (`submit_check.rs`): the product's long buffers lie in them only where that cost is none
//!   (`fresh_pages.rs`), but a rebind's copy under an image in place lies in them wherever the
//!   class passed (`buffer_descriptor.rs`), and the check's buffer, never parked, holds its import
//!   for its round alone; unchecked where the device takes none. A thread of its own then writes
//!   the window, sweep after sweep, each in the order of a stride that reaches texels, the bytes
//!   between rows and the bytes after the image from its first writes, each sweep's bytes the
//!   pattern changed by a value of its own, reading each byte before it writes it, which must
//!   still hold what the sweep before left there, while the device makes the image's first move
//!   (an internal submission naming the texture, waited for) and until it completed. Every byte
//!   must then read the CPU's last write where the CPU wrote it, and the pattern elsewhere: a
//!   device that writes back bytes it read beside the texels puts back an older value over a
//!   write the CPU made between its read and its write, wherever in the step those lie
//!   (`Bytes::write_until`). Each class runs every shape of [`SHAPES`] (texels of 1, 2, 4, 8 and 16
//!   bytes, rows apart or one after another, and 3840x2160 at 4 bytes) [`ROUNDS`] times.
//! - **An image that renders** ([`Access::Renders`]) is checked apart, in rounds of their own: the
//!   texture is made with `RenderTarget` usage as well, as the program's render targets are, so
//!   its image takes the feedback loop usage where the product's does; the writer writes the bytes
//!   between rows and after the image alone, through the image's first move, its moves into
//!   `ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT` and back where it has the usage
//!   (`Texture::settle_feedback_moves`, as `render_target::moved_barriers` records them), and a
//!   render pass of a command queue of the check's own that clears it to zero at its load and
//!   stores it, then the crate's draw of zero over its last column in a pass that loads and stores
//!   it (`check_draw.rs`), waited for (each by a submission the round waits for). Every texel must
//!   then read zero, and every other byte as above. The load, store and resolve operations and the
//!   layout transitions are what Vulkan lets touch the rest of the subresource, and a draw's color
//!   writes reach its fragments' texels alone (the Memory Model), but a device may write back the
//!   line of a cache that holds a row's last texels whole, as the Iris Xe's clear and store does:
//!   the draw over the last column, whose fragments end each row, has a device whose draws do so
//!   fail too. The program's resolves into such a texture are not among the rounds.
//! - **An image a shader writes** ([`Access::Writes`]) is checked apart too: a texture of
//!   `ShaderWrite` usage over the pattern, whose every texel a kernel stores (the crate-owned
//!   copy-in kernel, through an unsigned-integer view of the image, `Texture::store_in_place`)
//!   from a buffer of the device's, by a submission that makes the image's first move first, while
//!   the writer writes the bytes between rows and after the image alone, until it completed.
//!   Every texel must then read the kernel's bytes (the pattern's, changed in four bits of each,
//!   so they differ from the pattern), and every other byte as above. Each
//!   class runs every shape of [`STORE_SHAPES`] (texels of 4, 8 and 16 bytes, the widths the
//!   kernel's views store, rows apart, some of a width no multiple of 4 and an odd height, so that
//!   a driver's write of a block wider than a texel past a row's or the image's end shows, or one
//!   after another, and 1920x1080 at 4 bytes) [`ROUNDS`] times. Its rounds are stores: a program's
//!   atomics on such a texture are kernels' storage writes of its texels too, but none is among
//!   the rounds. The same rounds give the verdicts of a shader's writes of a texture over a
//!   buffer's rows whose padding is free ([`Access::BufferWrites`], `in_place::Padding`): a round
//!   whose texels all read the kernel's bytes, and every byte but of its rows' padding (after a
//!   row's last texel, before its pitch, the last row's included) the CPU's or the pattern, where
//!   the shape's padding is free ([`Shape::padding`]), fails the one and keeps the other
//!   ([`Round::Padded`]): its writer writes on past such a byte, and the rounds go on through
//!   every shape for those verdicts. No round of its own runs for them, so the check takes no
//!   longer than on a device whose every round kept every byte.
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
//!   first move (and pass, or kernel's stores): the writes themselves run only while those do.
//! - **Its verdict**, per class, access and row ends ([`RowEnds`]: whether a line of the device's
//!   memory, `nonCoherentAtomSize` bytes, holds both a texel and a byte beside the texels, each
//!   shape's from its pitch and width, [`Shape::rows`]): passed where every image it made of
//!   shapes of the row ends kept every byte (but the texels a pass cleared or a kernel stored),
//!   failed where one did not (or the device was made to fail it,
//!   `DeviceOptions::fail_in_place_check`), and unchecked where it made none (the device makes no
//!   such buffer, image, import, pass or dispatch) ([`verdicts`]). The shapes whose texels share
//!   no line with bytes beside them run first: a byte one of them changed fails both row ends,
//!   as a device that changes a byte beside such texels changes one beside any; a byte a shape
//!   whose texels share a line changed fails its row ends alone, the others' rounds already run.
//!   On the Iris Xe, whose clear and store, and draws, write back the line that holds a row's last
//!   texels whole, the first fail and the second pass. Of [`SHAPES`] at 64-byte lines, the five
//!   whose rows lie one after another, the 20x9 one at a 512-byte pitch and the 3840x2160 surface
//!   share none, and the other six share one. Each is the device's as it is known, so a class,
//!   access and row ends that passed take textures in place while the next is checked. Counted
//!   (`WorkCounts::in_place_checks_*`).
//! - **Kept.** Once every class answered, the verdicts are written as one record in the shader
//!   cache's content directory (`shader_cache/store.rs`), named by the driver's identity (its
//!   version and the device's UUID among it) and the check's version ([`record_name`]), not the
//!   runtime's: a device of the same driver and version reads them as it is made and runs no
//!   check, whatever runtime release made it, until the check itself changes
//!   (`in_place::CHECK_VERSION`). A failed write leaves the verdicts for the device's life. A
//!   forced failure is kept nowhere.
//! - **Out of the device's way.** Before each round the thread steps aside for the device's
//!   allocations and frees on other threads (`step_aside.rs`): it goes on once none is in
//!   progress and none ended for `step_aside::QUIET`, at once while a call waits for its verdicts
//!   (`Device::settle_in_place_checks`). Its own buffers and textures, made through the product's
//!   paths on its thread, count for nothing there. Its Vulkan work is counted as the device's
//!   (`step_aside::Counted::AsTheDevice`): those paths, and the scheduler's passes its waits take
//!   on its thread, carry the program's batches beside its own.
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
//!   allocation of the device's own, host-visible and exported as a dma-buf, mapped, and imported
//!   only as that dma-buf (`HostMemory::dma_buf_only`), never by their pointer, which a driver may
//!   take too (ANV's own mapped allocation); the check's source answers each request over them
//!   with that dma-buf. The buffer's import, the image and its first move are the product's.

use crate::check_draw::CheckDraw;
use crate::device::holds::Holder;
use crate::exit_guard;
use crate::host_memory::HostMemory;
use crate::in_place::{
    encode, mark_checking, record_name, Access, Padding, PagesClass, RowEnds, Verdict,
};
use crate::linear_plan::BufferLayout;
use crate::values::ResourceOptions;
use crate::{
    AddressPlacement, Buffer, BufferContents, BufferDescriptor, ClearColor, CommandQueue, Device,
    LoadAction, RenderPassDescriptor, StoreAction, Texture, TextureDescriptor,
};
use metal2vulkan::bind::ShadowTexelView;
use metal2vulkan::device::HostPageTexturePolicy;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Barrier;
use std::time::{Duration, Instant};

/// How many times each shape is checked in each class. Sized from the rate at which a round
/// catches a device that writes back the line of a cache that holds a row's last texels: on the
/// Iris Xe a render round (its pass's clear and store, then its draw) caught it in 198 of 200
/// rounds of a 100x50 `BGRA8Unorm` image at a 512-byte pitch and in 200 of 200 of a 33x17
/// `RGBA16Float` one. Eight of [`SHAPES`] have rows that end inside such a line, of which four
/// run on a device that makes images at an explicit pitch (the others at a driver's own pitch,
/// [`Shape::runs`]), so three rounds of each are twelve chances or more, of which a device that
/// escapes each as often as the Iris Xe's worst shape did (2 in 200) passes all with odds below
/// one in 10^20.
pub(crate) const ROUNDS: usize = 3;

/// The longest a render round waits for its pass's command buffer: a pass that clears a 3840x2160
/// surface and stores it.
const PASS_WAIT: Duration = Duration::from_secs(20);

/// A shape the check makes a texture of: its format, the bytes of one texel, its width and height,
/// its rows' pitch, and whether that pitch is the one a driver that makes no image at an explicit
/// pitch lays such rows at ([`Shape::runs`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Shape {
    pub(crate) format: PixelFormat,
    pub(crate) texel: u64,
    pub(crate) width: u64,
    pub(crate) height: u64,
    pub(crate) bytes_per_row: u64,
    pub(crate) tight: bool,
}

/// The shapes each class is checked with: each texel size the formats a buffer backs have (1, 2,
/// 4, 8 and 16 bytes), with rows one after another (a linear image made for the memory, which
/// every driver lays at a 256-byte pitch) and with bytes between them (an image at an explicit
/// pitch, where the device makes one: `texture/in_place_image.rs`), two of widths whose rows end
/// inside a 16-byte line at the 16-byte pitch a driver that lays rows tight lays them at (MoltenVK
/// makes no other image with bytes between rows, and its line, `nonCoherentAtomSize`, is 16
/// bytes: without them no shape of its whose texels share a line would be made, [`RowEnds`]), the
/// same two at the 64-byte pitch of a driver that lays rows at the next multiple of 64 (llvmpipe,
/// whose line is 64 bytes: likewise), and one 3840x2160 surface of 4-byte texels, last. Those four
/// at a driver's own pitch run only where no shape of their texel size whose texels share a line
/// was made before them ([`Shape::runs`]): a device that makes images at an explicit pitch makes
/// the 512-byte ones first. A shape the device makes no image of in place is skipped: the product
/// makes none of it either.
pub(crate) const SHAPES: [Shape; 15] = [
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
    tight(PixelFormat::BGRA8Unorm, 4, 101, 51, 416),
    tight(PixelFormat::RGBA16Float, 8, 33, 17, 272),
    tight(PixelFormat::BGRA8Unorm, 4, 101, 51, 448),
    tight(PixelFormat::RGBA16Float, 8, 33, 17, 320),
    shape(PixelFormat::BGRA8Unorm, 4, 3840, 2160, 15360),
];

/// The shapes the check of storage writes writes ([`Access::Writes`]), each with the view its
/// kernel stores through: texels of 4, 8 and 16 bytes (`in_place::STORED_TEXEL_BYTES`, the
/// crate-owned copy-in kernel's views), with rows one after another and with bytes between them
/// (of widths no multiple of 4 and heights no multiple of 2 among them, so that a driver's write
/// of a block wider than a texel past a row's or the image's end shows), at a 512-byte pitch, at
/// the 16-byte pitch a driver that lays rows tight lays them at and at the 64-byte pitch of one
/// that lays them at the next multiple of 64 ([`SHAPES`], and run as there: [`Shape::runs`]), and
/// one 1920x1080 surface of 4-byte texels, last. A shape the device makes no image of in place, or
/// no dispatch over, is skipped.
pub(crate) const STORE_SHAPES: [(Shape, ShadowTexelView); 11] = [
    (
        shape(PixelFormat::RGBA8Uint, 4, 64, 16, 256),
        ShadowTexelView::R8G8B8A8Uint,
    ),
    (
        shape(PixelFormat::RG32Uint, 8, 32, 16, 256),
        ShadowTexelView::R32G32Uint,
    ),
    (
        shape(PixelFormat::RGBA32Uint, 16, 16, 16, 256),
        ShadowTexelView::R32G32B32A32Uint,
    ),
    (
        shape(PixelFormat::RGBA8Uint, 4, 101, 51, 512),
        ShadowTexelView::R8G8B8A8Uint,
    ),
    (
        shape(PixelFormat::RG32Uint, 8, 33, 17, 512),
        ShadowTexelView::R32G32Uint,
    ),
    (
        shape(PixelFormat::RGBA32Uint, 16, 21, 9, 512),
        ShadowTexelView::R32G32B32A32Uint,
    ),
    (
        tight(PixelFormat::RGBA8Uint, 4, 101, 51, 416),
        ShadowTexelView::R8G8B8A8Uint,
    ),
    (
        tight(PixelFormat::RG32Uint, 8, 33, 17, 272),
        ShadowTexelView::R32G32Uint,
    ),
    (
        tight(PixelFormat::RGBA8Uint, 4, 101, 51, 448),
        ShadowTexelView::R8G8B8A8Uint,
    ),
    (
        tight(PixelFormat::RG32Uint, 8, 33, 17, 320),
        ShadowTexelView::R32G32Uint,
    ),
    (
        shape(PixelFormat::RGBA8Uint, 4, 1920, 1080, 7680),
        ShadowTexelView::R8G8B8A8Uint,
    ),
];

const fn shape(format: PixelFormat, texel: u64, width: u64, height: u64, bpr: u64) -> Shape {
    Shape {
        format,
        texel,
        width,
        height,
        bytes_per_row: bpr,
        tight: false,
    }
}

/// [`shape`], at the pitch a driver that makes no image at an explicit pitch lays its rows at.
const fn tight(format: PixelFormat, texel: u64, width: u64, height: u64, bpr: u64) -> Shape {
    Shape {
        tight: true,
        ..shape(format, texel, width, height, bpr)
    }
}

impl Shape {
    /// Whether the window's byte at `offset` is one of the texture's texels.
    fn is_texel(&self, offset: u64) -> bool {
        offset < self.bytes_per_row * self.height
            && offset % self.bytes_per_row < self.width * self.texel
    }

    /// Whether the window's byte at `offset` is one of its rows' padding: after a row's last
    /// texel, before its pitch (the last row's included).
    fn in_padding(&self, offset: u64) -> bool {
        offset < self.bytes_per_row * self.height
            && offset % self.bytes_per_row >= self.width * self.texel
    }

    /// Its texels' layout in the window, which its texture starts.
    fn layout(&self) -> BufferLayout {
        BufferLayout {
            offset: 0,
            row_pitch: self.bytes_per_row,
            slice_pitch: self.bytes_per_row * self.height,
            row_bytes: self.width * self.texel,
            rows: self.height,
            slices: 1,
        }
    }

    /// How its texels lie in lines of `line` bytes ([`RowEnds::of`]): its texture starts the
    /// window, which starts on a page.
    pub(crate) fn rows(&self, line: u64) -> RowEnds {
        RowEnds::of(0, &self.layout(), line)
    }

    /// Whether its rows' padding is free in lines of `line` bytes ([`Padding::of`]): its texture
    /// lies over a buffer's rows, whose window holds them to their pitch.
    pub(crate) fn padding(&self, line: u64) -> Padding {
        Padding::of(0, &self.layout(), line, true)
    }

    /// Whether a class's rounds take it, in lines of `line` bytes, after shapes whose texels share
    /// a line, of the texel sizes `made`, were made in the class: but for a shape at a driver's own
    /// pitch whose texels share a line, where one of its texel size was made (a device that makes
    /// images at an explicit pitch made it, so the shape adds rounds the class has had and
    /// nothing a device of that pitch lacks; `made` grows as each such shape is made).
    pub(crate) fn runs(&self, line: u64, made: &[u64]) -> bool {
        !(self.tight && self.rows(line) == RowEnds::InLines && made.contains(&self.texel))
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
                    for rows in RowEnds::ALL {
                        if checks.verdict(class, access, rows) == Verdict::Pending {
                            checks.set(class, access, rows, Verdict::Unchecked);
                        }
                    }
                }
            }
            checks.settle(None);
        }
    }
}

/// Before a round, step aside for the device's allocations and frees on other threads
/// (`step_aside.rs`): not while the program's last drop stops the check or a call waits for its
/// verdicts. Counted where it waited.
fn step_aside(device: &Device) {
    let shared = device.shared();
    let activity = shared.allocations.allocations.activity();
    let stop = || shared.holds.stopping() || shared.in_place.awaited();
    if let Some(waited) = activity.step_aside(stop) {
        shared.counts.check_stepped_aside(waited);
    }
}

/// The check's thread: each class in turn, each of its accesses, each verdict given as it is
/// known, then the record.
fn run(weak: crate::device::WeakDevice) {
    mark_checking();
    crate::step_aside::mark_a_check_s_thread(crate::step_aside::Counted::AsTheDevice);
    let started = Instant::now();
    let mut stopped = false;
    for class in PagesClass::ALL {
        for access in Access::ALL {
            // The accesses this one's rounds answer: none of an access with no rounds of its own.
            let answered = access.answered();
            if answered.is_empty() {
                continue;
            }
            let Some(device) = weak.hold() else {
                return;
            };
            let checks = &device.shared().in_place;
            let pending = |(access, rows)| checks.verdict(class, access, rows) == Verdict::Pending;
            let each =
                || (answered.iter()).flat_map(|&access| RowEnds::ALL.map(|rows| (access, rows)));
            if !each().any(pending) {
                continue;
            }
            if stopped || word_overwritten(&device) {
                stopped = true;
                for (access, rows) in each() {
                    checks.set(class, access, rows, Verdict::Unchecked);
                }
                continue;
            }
            let Some(verdicts) = check_class(&device, class, access) else {
                // The process is exiting, or the program let the device go: nothing is kept.
                return;
            };
            for (&access, verdicts) in answered.iter().zip(verdicts) {
                for rows in RowEnds::ALL {
                    let verdict = verdicts[rows.index()];
                    checks.set(class, access, rows, verdict);
                    device.shared().counts.in_place_check_ran(verdict);
                }
            }
            // Shadows made before their verdicts that know them now may move in place.
            device.in_place_verdicts_given();
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
    device.in_place_verdicts_given();
}

/// Whether the runtime found `device`'s loader word overwritten (`loader_word.rs`): the check then
/// runs no further round (the module docs).
fn word_overwritten(device: &Device) -> bool {
    // SAFETY: the device is live while `device` is.
    unsafe { device.shared().loader_word.overwritten() }
}

/// The verdicts of `class` on `device` for each access `access`'s rounds answer
/// ([`Access::answered`]), in that order, each by row ends ([`RowEnds::ALL`]'s order): every
/// shape's rounds, those of shapes whose texels share no line with bytes beside them first
/// ([`Shape::rows`]), until one changes a byte, or until the device's loader word is found
/// overwritten (unchecked then where no round ran); [`Tally::verdicts`] gives them. A round of
/// storage writes that changed no byte but of its shape's rows' padding, where that padding is
/// free ([`Shape::padding`]), fails [`Access::Writes`] for its row ends and counts for
/// [`Access::BufferWrites`] as a round that kept every byte, and the rounds go on. `None` where
/// the process is exiting, or the program's last drop stopped the device's holds before a round
/// (`device/holds.rs`).
fn check_class(device: &Device, class: PagesClass, access: Access) -> Option<Vec<[Verdict; 2]>> {
    let page = host_page();
    let line = device.shared().in_place.line();
    let unchecked = || vec![[Verdict::Unchecked; 2]; access.answered().len()];
    let Some(pages) = ClassPages::make(device, class, BUFFER_BYTES) else {
        return Some(unchecked());
    };
    // The command queue an image that renders takes its passes on, the check's own, and the
    // draw it takes after them.
    let mut renders = match access {
        Access::Reads | Access::Writes | Access::BufferWrites => None,
        Access::Renders => {
            let made = exit_guard::enter().map(|_busy| {
                let queue = device.new_command_queue_now()?;
                Ok::<_, crate::Error>((queue, CheckDraw::new(device)?))
            });
            match made? {
                Ok(renders) => Some(renders),
                Err(_) => return Some(unchecked()),
            }
        }
    };
    // An image a shader writes takes the shapes its kernel stores, each with its view; those
    // whose texels share no line with bytes beside them first.
    let mut shapes: Vec<(Shape, Option<ShadowTexelView>)> = match access {
        Access::Writes | Access::BufferWrites => STORE_SHAPES
            .map(|(shape, texel)| (shape, Some(texel)))
            .to_vec(),
        Access::Reads | Access::Renders => SHAPES.map(|shape| (shape, None)).to_vec(),
    };
    shapes.sort_by_key(|(shape, _)| shape.rows(line) != RowEnds::OnLines);
    let forced = device.shared().in_place.forced_failure();
    let mut tally = Tally::default();
    // The texel sizes of the shapes whose texels share a line made so far (`Shape::runs`).
    let mut made_sharing = Vec::new();
    for (shape, texel) in shapes {
        if shape.window(page) > BUFFER_BYTES || !shape.runs(line, &made_sharing) {
            continue;
        }
        let rows = shape.rows(line);
        // A round of storage writes over a shape whose rows' padding is free tells a change
        // there apart.
        let free = texel.is_some() && shape.padding(line) == Padding::Free;
        for round in 0..ROUNDS {
            if !device.holds_go_on(Holder::InPlaceCheck) {
                return None;
            }
            step_aside(device);
            if word_overwritten(device) {
                return Some(unchecked());
            }
            let outcome = {
                let _busy = exit_guard::enter()?;
                match texel {
                    Some(texel) => store_round(device, &pages, (shape, texel, free), round, page),
                    None => check_round(&pages, shape, round, page, renders.as_mut()),
                }
            };
            let made = |made_sharing: &mut Vec<u64>| {
                if rows == RowEnds::InLines && !made_sharing.contains(&shape.texel) {
                    made_sharing.push(shape.texel);
                }
            };
            let ((offset, due, read), padding) = match outcome {
                Round::Kept => {
                    tally.kept(rows, false);
                    made(&mut made_sharing);
                    continue;
                }
                Round::NotMade => break,
                Round::Padded(changed) => {
                    made(&mut made_sharing);
                    // The first such round is noted; the rounds go on.
                    if !tally.kept(rows, true) {
                        continue;
                    }
                    (changed, true)
                }
                Round::Changed(changed) => (changed, false),
            };
            let image = shape.bytes_per_row * shape.height;
            let place = match offset.checked_rem(shape.bytes_per_row) {
                _ if offset >= image => "after the image",
                Some(column) if column >= shape.width * shape.texel => "after a row's texels",
                _ => "a texel",
            };
            let operations = match access {
                Access::Reads => "the first move",
                Access::Renders => {
                    "the first move, feedback loop moves, a pass's clear and store or a draw"
                }
                Access::Writes | Access::BufferWrites => {
                    "the first move or a kernel's storage writes"
                }
            };
            let lines = match rows {
                RowEnds::InLines => "",
                RowEnds::OnLines => ", rows on lines",
            };
            let alone = match padding {
                true => ", and no byte but of its rows' padding",
                false => "",
            };
            device.shared().in_place.note(format!(
                "{class:?}{lines}: {operations} of a {:?} image of {}x{} with rows {} bytes apart \
                 changed byte {offset} ({place}) from {due:#04x} to {read:#04x} in round \
                 {round}{alone}",
                shape.format, shape.width, shape.height, shape.bytes_per_row
            ));
            if !padding {
                return Some(tally.verdicts(access, Some(rows), forced));
            }
        }
    }
    Some(tally.verdicts(access, None, forced))
}

/// A class's rounds for an access so far (`check_class`): those that kept every byte, by row ends
/// ([`RowEnds::ALL`]'s order); those that changed no byte but of their shape's free padding
/// ([`Shape::padding`]) or none at all, by row ends; and the row ends of the first round that
/// changed a byte of that padding.
#[derive(Default)]
pub(crate) struct Tally {
    kept: [u32; 2],
    beside: [u32; 2],
    padding: Option<RowEnds>,
}

impl Tally {
    /// Count a round of shapes of `rows` row ends that kept every byte, or, where `padded`, that
    /// changed none but of its shape's free padding: whether it is the first that changed one
    /// there.
    pub(crate) fn kept(&mut self, rows: RowEnds, padded: bool) -> bool {
        self.beside[rows.index()] += 1;
        if !padded {
            self.kept[rows.index()] += 1;
            return false;
        }
        let first = self.padding.is_none();
        self.padding.get_or_insert(rows);
        first
    }

    /// The verdicts of each access `access`'s rounds answer, in [`Access::answered`]'s order, on
    /// a device made to fail its check where `forced`, where `changed` names the row ends of the
    /// round that changed a byte but of its shape's free padding, where one did (which ends the
    /// class's rounds): the access's own ([`verdicts`]), of its rounds that kept every byte, the
    /// row ends of the first that changed any byte failed; then, of [`Access::Writes`], those of
    /// [`Access::BufferWrites`], of its rounds that changed no byte but of that padding, those
    /// of `changed` failed.
    pub(crate) fn verdicts(
        &self,
        access: Access,
        changed: Option<RowEnds>,
        forced: bool,
    ) -> Vec<[Verdict; 2]> {
        let own = verdicts(self.kept, self.padding.or(changed), forced);
        let beside = verdicts(self.beside, changed, forced);
        let answered = access.answered().len();
        [own, beside].into_iter().take(answered).collect()
    }
}

/// The verdicts, by row ends ([`RowEnds::ALL`]'s order), of a class and access whose rounds kept
/// every byte `made` times for each, until a round of shapes of `changed` row ends changed one,
/// where one did, on a device made to fail its check where `forced`: failed for the row ends that
/// changed a byte, and for both where texels that share no line with bytes beside them did (a
/// device that changes a byte beside such texels changes one beside any); else unchecked for row
/// ends of which no round kept every byte, failed where forced, and passed.
pub(crate) fn verdicts(made: [u32; 2], changed: Option<RowEnds>, forced: bool) -> [Verdict; 2] {
    RowEnds::ALL.map(|rows| match (changed, made[rows.index()], forced) {
        (Some(RowEnds::OnLines), _, _) => Verdict::Failed,
        (Some(at), _, _) if at == rows => Verdict::Failed,
        (_, 0, _) => Verdict::Unchecked,
        (_, _, true) => Verdict::Failed,
        (_, _, false) => Verdict::Passed,
    })
}

/// What one round found.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Round {
    /// The image's move kept every byte.
    Kept,
    /// The texture took no image in place (the device makes none of the shape there).
    NotMade,
    /// A byte of its shape's rows' padding read other than the CPU left it, where a round of
    /// storage writes tells that padding apart ([`Shape::padding`]), and every other byte read
    /// right: where the first was, what was due, what was read.
    Padded((u64, u8, u8)),
    /// A byte read other than the CPU left it: where, what was due, what was read.
    Changed((u64, u8, u8)),
}

/// One round of `shape` over `pages`, inside the caller's section: the pattern, the texture, the
/// writes beside its image's first move (and, for an image that renders, `renders` its queue and
/// draw, beside its feedback loop moves, a pass that clears and stores it and a draw of its last
/// column), and the bytes compared.
fn check_round(
    pages: &ClassPages,
    shape: Shape,
    round: usize,
    page: u64,
    renders: Option<&mut (CommandQueue, CheckDraw)>,
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
    descriptor.usage = match renders {
        None => TextureUsage::SHADER_READ,
        Some(_) => TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
    };
    // An image that renders: the writer writes beside its texels alone, which its pass clears.
    let texels = renders.as_ref().map(|_| shape);
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
    let outcome = std::thread::scope(|scope| {
        let builder = std::thread::Builder::new().name("m2v-in-place-writes".to_owned());
        let Ok(writer) = exit_guard::spawn_scoped(scope, builder, || {
            bytes.write_until(&pattern, (texels, false), (&stop, &started))
        }) else {
            return None;
        };
        let stopping = Stopping(&stop);
        started.wait();
        let settled = match texture.settle_in_place() {
            Some(Ok(())) => match renders {
                None => Some(Ok(())),
                Some((queue, draw)) => Some(render(queue, draw, &texture)),
            },
            other => other,
        };
        drop(stopping);
        Some((settled, writer.join().ok()?))
    });
    let Some((Some(Ok(())), written)) = outcome else {
        return Round::NotMade;
    };
    drop(texture);
    let texels = texels.map(|shape| (shape, Due::Zero));
    bytes.compare(&pattern, (texels, false), written)
}

/// Stops the round's writer as it goes: once the round's step returned, or where it unwinds.
struct Stopping<'a>(&'a AtomicBool);

impl Drop for Stopping<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

/// One round of storage writes of `shape` over `pages`, through `texel`'s view, inside the
/// caller's section: the pattern; a texture of shader write usage over it; the bytes of a buffer
/// of `device`'s stored into its texels by a kernel, while the CPU writes the bytes beside them
/// (between rows and after the image) and the image takes its first move; and the bytes
/// compared: each texel's the kernel's, each other byte the CPU's or the pattern, a change of a
/// byte of the rows' padding told apart where `free` ([`Round::Padded`]).
fn store_round(
    device: &Device,
    pages: &ClassPages,
    (shape, texel, free): (Shape, ShadowTexelView, bool),
    round: usize,
    page: u64,
) -> Round {
    let window = shape.window(page);
    let bytes = Bytes {
        start: pages.start,
        length: window,
    };
    let pattern = Pattern::new(round as u64 + 101);
    bytes.fill(&pattern);
    let mut descriptor =
        TextureDescriptor::texture_2d(shape.format, shape.width, shape.height, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    let Ok(texture) = pages
        .buffer
        .new_runtime_texture(&descriptor, 0, shape.bytes_per_row)
    else {
        return Round::NotMade;
    };
    // The bytes the kernel stores, row after row, and the part it keeps them in too: buffers
    // outside the Metal address space, as the check's are (the module docs).
    let row = shape.width * shape.texel;
    let length = row * shape.height;
    let buffer =
        || device.new_buffer_placed(length, ResourceOptions::shared(), AddressPlacement::Outside);
    let (Ok(source), Ok(kept)) = (buffer(), buffer()) else {
        return Round::NotMade;
    };
    let Some(words) = source.contents() else {
        return Round::NotMade;
    };
    for y in 0..shape.height {
        let mut to = (y * row) as usize;
        for (from, count) in pattern.spans(y * shape.bytes_per_row, row as usize) {
            // SAFETY: the source's bytes `to..to + count`, mapped and writable, which nothing else
            // uses yet; the pattern's bytes are apart from them.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    pattern.stored[from..].as_ptr(),
                    words.as_ptr().add(to),
                    count,
                )
            };
            to += count;
        }
    }
    let stop = AtomicBool::new(false);
    let started = Barrier::new(2);
    let extent = (shape.width as u32, shape.height as u32);
    let outcome = std::thread::scope(|scope| {
        let builder = std::thread::Builder::new().name("m2v-in-place-writes".to_owned());
        let Ok(writer) = exit_guard::spawn_scoped(scope, builder, || {
            bytes.write_until(&pattern, (Some(shape), free), (&stop, &started))
        }) else {
            return None;
        };
        let stopping = Stopping(&stop);
        started.wait();
        let done = texture.store_in_place((&source, &kept), texel, extent);
        drop(stopping);
        Some((done, writer.join().ok()?))
    });
    let Some((Some(Ok(())), written)) = outcome else {
        return Round::NotMade;
    };
    drop(texture);
    bytes.compare(&pattern, (Some((shape, Due::Stored)), free), written)
}

/// What a round's texels must read once it ended, where the writer passed them over: zero (a
/// pass cleared them) or the bytes a kernel stored (`Pattern::stored`).
#[derive(Clone, Copy)]
pub(crate) enum Due {
    Zero,
    Stored,
}

/// An image that renders, after its first move: its moves into the feedback loop layout and back
/// where it has the usage, a pass of `queue` that clears it to zero at its load and stores it,
/// then `draw`'s draw of zero over its last column in a pass that loads and stores it, each waited
/// for. An error where one was not made, or the pass's command buffer failed or did not complete
/// within [`PASS_WAIT`].
fn render(
    queue: &CommandQueue,
    draw: &mut CheckDraw,
    texture: &Texture,
) -> Result<(), crate::Error> {
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
    if command_buffer.error().is_some() {
        return Err(failed());
    }
    texture
        .draw_in_place_column(draw)
        .unwrap_or_else(|| Err(crate::error::internal("the check's image takes no draw")))
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
        let _busy = exit_guard::enter()?;
        match class {
            PagesClass::Pointer => pointer_pages(device, length),
            // In fresh pages wherever the device takes them, whatever its submission checks
            // found a held import costs: a rebind's copy under an image in place lies in them
            // where this class passed (`buffer_descriptor.rs`'s `copied_pages`), the product's
            // long buffers where that cost is none (`fresh_pages.rs`).
            PagesClass::Fresh => {
                let buffer = device.fresh_pages_check_buffer(length).ok()?;
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
    device.new_buffer_with_descriptor_now(descriptor).ok()
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
    // Imported only as the source's dma-bufs: a driver may also take the pointer of a mapping of
    // its own exported allocation, which would make these pages the pointer's class.
    let memory = unsafe { HostMemory::with_dma_buf(start, source) }.dma_buf_only();
    let buffer = no_copy_kept(device, memory, length, pages)?;
    if !buffer.over_dma_buf() {
        return None;
    }
    Some(ClassPages { buffer, start })
}

/// A window of a class's pages, from its byte 0.
#[derive(Clone, Copy)]
pub(crate) struct Bytes {
    pub(crate) start: NonNull<u8>,
    pub(crate) length: u64,
}

// SAFETY: the window's pages stay mapped while the round uses them (its buffer holds them), and
// the writer thread and the round's thread take turns: the writer writes until it is stopped and
// joined, and the round compares after.
unsafe impl Send for Bytes {}
// SAFETY: as above.
unsafe impl Sync for Bytes {}

/// The stride the writer walks a window at, or the first after it coprime with the window's
/// length ([`Bytes::stride`]): each sweep meets every byte once, and its writes lie far apart,
/// through texels, the bytes between rows and the bytes after the image, so that it comes back to
/// the bytes of one line of a cache many times in a sweep.
const STRIDE: u64 = 1_000_003;

/// The bytes a round's pattern repeats every: a prime, so that its repeats line up with no row
/// pitch, page or texel, and a byte copied from another row or page reads wrong.
const PERIOD: usize = 4099;

/// A round's pattern: [`PERIOD`] bytes of a hash of their place and the round, repeated over the
/// window. A byte the writer wrote holds it changed by its last sweep's [`mark`].
pub(crate) struct Pattern {
    bytes: Vec<u8>,
    /// The bytes a kernel of a round of storage writes stores, by their place in the period: the
    /// pattern's, changed in four bits, so that each differs from the pattern.
    stored: Vec<u8>,
}

impl Pattern {
    /// The pattern of round `salt`.
    pub(crate) fn new(salt: u64) -> Pattern {
        let bytes = (0..PERIOD as u64)
            .map(|offset| {
                let mixed = (offset ^ salt.rotate_left(17)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                (mixed >> 56) as u8
            })
            .collect::<Vec<u8>>();
        let stored = bytes.iter().map(|byte| byte ^ 0x5a).collect();
        Pattern { bytes, stored }
    }

    /// Its byte at `offset` of the window.
    pub(crate) fn at(&self, offset: u64) -> u8 {
        self.bytes[(offset % PERIOD as u64) as usize]
    }

    /// The window's bytes `offset..offset + length` as runs within one period, in order: each
    /// run's first place in the period and its length.
    fn spans(&self, offset: u64, length: usize) -> impl Iterator<Item = (usize, usize)> {
        let mut from = (offset % PERIOD as u64) as usize;
        let mut left = length;
        std::iter::from_fn(move || {
            (left > 0).then(|| {
                let count = (PERIOD - from).min(left);
                let span = (from, count);
                left -= count;
                from = 0;
                span
            })
        })
    }
}

/// What a round's writer wrote ([`Bytes::write_until`]): the sweep it was in as it stopped, how
/// many of the stride's bytes that sweep reached (each byte before that one written by it, each
/// after by the sweep before), the first byte it found other than it left it, where it found one
/// (its offset, what was due and what was read), and the first of its shape's rows' padding it
/// found so, where it told that padding apart.
pub(crate) struct Written {
    pub(crate) sweep: u64,
    pub(crate) reached: u64,
    pub(crate) changed: Option<(u64, u8, u8)>,
    pub(crate) padding: Option<(u64, u8, u8)>,
}

/// What sweep `sweep` of a round's writer changes the pattern by: nothing before the first
/// (`sweep` 0), then each of the 255 other byte values in turn, so that each sweep's bytes differ
/// from the pattern and from the sweep's before.
fn mark(sweep: u64) -> u8 {
    match sweep {
        0 => 0,
        _ => ((sweep - 1) % 255) as u8 + 1,
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

    /// Write the bytes beside the texels (every byte where `texels` names no shape; where it names
    /// the texture's, its texels are passed over: its pass clears them, or its kernel stores
    /// them), sweep after sweep, each in the stride's order, until `stop`: sweep `k` writes the
    /// pattern changed by [`mark`]`(k)`, and reads each byte before it writes it, which must still
    /// hold what the sweep before left there (the pattern before the first). `started` once the
    /// first byte is written. What it wrote, and the first byte it found otherwise, where it found
    /// one (then it stops there); but where `free`, a byte of the shape's rows' padding it found
    /// otherwise is noted apart, the first of them, and written on as any other.
    ///
    /// A device that writes back bytes it read beside the texels (a whole line of a cache, say, of
    /// which the texels hold part) leaves there what they held when it read them. Where the writer
    /// wrote one of them between the device's read and its write, the writer's next read of it,
    /// or the round's compare, finds the older value: the writer rewrites every byte of the line
    /// again at its next sweep, so any span between the read and the write longer than the
    /// writer's walk between two bytes of the line holds one of its writes, wherever in the step
    /// the span lies. A writer that wrote each byte once, as the step began, missed every such
    /// span past its first writes: the Iris Xe's clear and store of a 100x50 image at a 512-byte
    /// pitch put back the pattern over its writes to the bytes after a row's last texels, in their
    /// line, in 77 of 100 such rounds, and its check passed so under validation layers.
    fn write_until(
        &self,
        pattern: &Pattern,
        (texels, free): (Option<Shape>, bool),
        (stop, started): (&AtomicBool, &Barrier),
    ) -> Written {
        let stride = self.stride();
        let mut first = true;
        let mut started_once = || {
            if first {
                first = false;
                started.wait();
            }
        };
        let in_padding = |offset| free && texels.is_some_and(|shape| shape.in_padding(offset));
        let mut padding = None;
        let mut sweep = 1;
        loop {
            let (before, now) = (mark(sweep - 1), mark(sweep));
            for n in 0..self.length {
                let offset = self.nth(n, stride);
                if texels.is_some_and(|shape| shape.is_texel(offset)) {
                    continue;
                }
                let byte = self.byte(offset);
                let due = pattern.at(offset) ^ before;
                // SAFETY: as in `fill`; the round's thread does not touch the bytes meanwhile.
                let read = unsafe { byte.read_volatile() };
                if read != due {
                    if !in_padding(offset) {
                        started_once();
                        return Written {
                            sweep,
                            reached: n,
                            changed: Some((offset, due, read)),
                            padding,
                        };
                    }
                    padding.get_or_insert((offset, due, read));
                }
                // SAFETY: as above.
                unsafe { byte.write_volatile(pattern.at(offset) ^ now) };
                started_once();
                if stop.load(Ordering::Acquire) {
                    return Written {
                        sweep,
                        reached: n + 1,
                        changed: None,
                        padding,
                    };
                }
            }
            sweep += 1;
        }
    }

    /// What the round found after `written` ([`Round`]): the first byte that reads other than
    /// the writer and the round left it, its offset, what was due and what was read, where one
    /// does ([`Round::Changed`]); else, where `free`, the first byte of the rows' padding of the
    /// shape `texels` names that does ([`Round::Padded`]); else [`Round::Kept`]. The first byte
    /// the writer found changed, where it found one; else each byte it wrote must read what its
    /// last sweep over it wrote (the stride's first `written.reached` bytes that sweep's, the
    /// rest the sweep's before, where there was one), and is set back to the pattern as it is
    /// read; then every byte must read the pattern, compared a period at a time, but, where
    /// `texels` names the shape a pass cleared or a kernel stored, its texels, which must read
    /// what is due of them.
    pub(crate) fn compare(
        &self,
        pattern: &Pattern,
        (texels, free): (Option<(Shape, Due)>, bool),
        written: Written,
    ) -> Round {
        if let Some(changed) = written.changed {
            return Round::Changed(changed);
        }
        let in_padding = |offset| free && texels.is_some_and(|(shape, _)| shape.in_padding(offset));
        let mut padding = written.padding;
        let stride = self.stride();
        let reached = match written.sweep {
            1 => written.reached,
            _ => self.length,
        };
        for n in 0..reached {
            let offset = self.nth(n, stride);
            if texels.is_some_and(|(shape, _)| shape.is_texel(offset)) {
                continue;
            }
            let last = match n < written.reached {
                true => written.sweep,
                false => written.sweep - 1,
            };
            let due = pattern.at(offset) ^ mark(last);
            // SAFETY: as in `fill`; the writer was joined.
            let read = unsafe { self.byte(offset).read_volatile() };
            if read != due {
                if !in_padding(offset) {
                    return Round::Changed((offset, due, read));
                }
                padding.get_or_insert((offset, due, read));
            }
            // SAFETY: as above.
            unsafe { self.byte(offset).write_volatile(pattern.at(offset)) };
        }
        let Some((shape, due)) = texels else {
            return match self.matches(&pattern.bytes, 0, self.length) {
                Some(changed) => Round::Changed(changed),
                None => Round::Kept,
            };
        };
        // Each row's texels read what is due, and the bytes after them to the next row the
        // pattern.
        let row_bytes = shape.width * shape.texel;
        for row in 0..shape.height {
            let start = row * shape.bytes_per_row;
            let changed = match due {
                Due::Zero => {
                    // SAFETY: as above; the row's texels lie in the window.
                    let read =
                        unsafe { std::slice::from_raw_parts(self.byte(start), row_bytes as usize) };
                    let at = read.iter().position(|&read| read != 0);
                    at.map(|at| (start + at as u64, 0, read[at]))
                }
                Due::Stored => self.matches(&pattern.stored, start, start + row_bytes),
            };
            if let Some(changed) = changed {
                return Round::Changed(changed);
            }
            let next = start + shape.bytes_per_row;
            if let Some(changed) = self.matches(&pattern.bytes, start + row_bytes, next) {
                if !free {
                    return Round::Changed(changed);
                }
                padding.get_or_insert(changed);
            }
        }
        let after = shape.bytes_per_row * shape.height;
        if let Some(changed) = self.matches(&pattern.bytes, after, self.length) {
            return Round::Changed(changed);
        }
        match padding {
            Some(changed) => Round::Padded(changed),
            None => Round::Kept,
        }
    }

    /// The first byte from `from` to `to` that does not read `period` (a period's bytes, by their
    /// place in it): its offset, what was due and what was read; `None` where every one does.
    /// Compared a period at a time.
    fn matches(&self, period: &[u8], from: u64, to: u64) -> Option<(u64, u8, u8)> {
        let mut offset = from;
        while offset < to {
            let phase = (offset % PERIOD as u64) as usize;
            let count = (to - offset).min((PERIOD - phase) as u64) as usize;
            // SAFETY: as in `compare`; nothing writes the bytes while the slice is read.
            let read = unsafe { std::slice::from_raw_parts(self.byte(offset), count) };
            let due = &period[phase..phase + count];
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
