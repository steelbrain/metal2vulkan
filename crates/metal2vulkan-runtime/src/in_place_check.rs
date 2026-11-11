//! The device's check of an image in place over host pages (`in_place.rs`): whether the
//! operations Vulkan lets touch the bytes bound to it beside its texels leave every byte alone
//! while the CPU writes beside it, per class of pages ([`PagesClass`]) and access ([`Access`]), on
//! a thread of its own.
//!
//! - **What it does.** For each class it makes a buffer as the product makes one there (a no-copy
//!   buffer over anonymous pages it maps, imported by their pointer; a buffer of the runtime's own
//!   longer than half a pooled block, in fresh pages; a no-copy buffer over pages the device
//!   refuses by their pointer, imported as dma-bufs; a buffer of the runtime's own in its pooled
//!   memory, under half a block, that declares the bytes between a texture's rows unobserved, where
//!   the device makes `DRM_FORMAT_MOD_LINEAR` images, its textures taking such an image at Metal's
//!   pitch alone, `texture.rs`'s `alias_image`; a shape whose window passes that buffer, the
//!   3840x2160 surface's, is skipped there; and, for each handle type the runtime exports a
//!   buffer's memory as, an opaque file descriptor and a dma-buf, an exportable buffer of the
//!   runtime's own that declares those bytes unobserved too, in a block that such buffers share,
//!   its textures taking the device's linear image made for that handle type, else the
//!   `DRM_FORMAT_MOD_LINEAR` one at Metal's pitch made for it, both made `UNDEFINED` as an image
//!   that names a handle type is; unchecked where the device exports no such memory or only as a
//!   buffer's dedicated allocation, to which no image is bound; and a buffer of the runtime's own
//!   in a block of the kind its buffers take where the device allocates their blocks exportable
//!   (`memory.rs`'s `Holds::FdBuffers`), whatever this class's verdicts, which decide whether its
//!   buffers lie there (`in_place.rs`'s `blocks_take`), that declares those bytes unobserved
//!   too, its textures taking the image a texture over such a buffer takes, made `UNDEFINED`;
//!   unchecked where the device allocates no such blocks), fills a window of it with a pattern,
//!   and makes a texture over it as the program does (`Buffer::new_runtime_texture`): the texture
//!   takes the image in place exactly as the product's would
//!   ([`checking`](crate::in_place::checking) lets it whatever the verdict); the fresh-pages
//!   class's buffer in fresh pages wherever the device takes them, whatever its submission checks
//!   found a held import costs
//!   (`submit_check.rs`): the product's long buffers lie in them only where that cost is none
//!   (`fresh_pages.rs`), but a rebind's copy under an image in place lies in them wherever the
//!   class passed (`buffer_descriptor.rs`), and the check's buffer, never parked, holds its import
//!   for its round alone; unchecked where the device takes none. A thread of its own then writes
//!   the window, sweep after sweep, each in the order of a stride that reaches texels, the bytes
//!   between rows and the bytes after the image from its first writes, each sweep's bytes the
//!   pattern changed by a value of its own, reading each byte before it writes it, which must
//!   still hold what the sweep before left there, while the device makes the image's first move
//!   (an internal submission naming the texture, waited for) and until it completed; where it
//!   passes over the texels (the rounds below that clear or store them), it comes back to the
//!   bytes beside them in their lines of the device's memory many times a sweep (`Walk`), so that
//!   a line's write-back meets one of its writes however short its span. Every byte
//!   must then read the CPU's last write where the CPU wrote it, and the pattern elsewhere: a
//!   device that writes back bytes it read beside the texels puts back an older value over a
//!   write the CPU made between its read and its write, wherever in the step those lie
//!   (`Bytes::write_until`). Each class runs every shape of [`SHAPES`] (texels of 1, 2, 4, 8 and 16
//!   bytes, rows apart or one after another, 172x67 at a 704-byte pitch, and 3840x2160 at 4
//!   bytes) [`ROUNDS`] times.
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
//!   fail too. The program's resolves into such a texture are not among the rounds. The same
//!   rounds give the verdicts of renders of a texture whose rows' padding is free and its buffer
//!   declares unobserved ([`Access::UnobservedRenders`], `BufferDescriptor::padding_unobserved`):
//!   where the shape's padding is free ([`Shape::padding`]), a round whose texels all read zero,
//!   and every byte but of its rows' padding (after a row's last texel, before its pitch, the last
//!   row's included) the CPU's or the pattern, fails [`Access::Renders`] and keeps the other
//!   ([`Round::Padded`]); a byte past the last row's pitch, or a texel, fails both. The 172x67
//!   shape's last row's pitch ends off a multiple of 128 bytes, so a device that writes back more
//!   than a line there fails both.
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
//!   after another, and 1920x1080 at 4 bytes) [`ROUNDS`] times. Texels of 1 and 2 bytes
//!   ([`Access::NarrowWrites`]) take rounds of their own, of [`NARROW_STORE_SHAPES`], through the
//!   kernel's `R8_UINT` and `R16_UINT` views (each texel a word of its buffers): a driver may
//!   store such a texel through an access wider than it, a word or more, over the texels beside
//!   it and past a row's end, which no wider texel's store shows, so their verdicts are their own
//!   and a byte one of their rounds changed fails no verdict of wider texels'. Its rounds are
//!   stores: a program's atomics on such a texture are kernels' storage writes of its texels too,
//!   but none is among the rounds. The same rounds give the verdicts of a shader's writes of a
//!   texture over a buffer's rows whose padding is free ([`Access::BufferWrites`],
//!   [`Access::NarrowBufferWrites`], `in_place::Padding`): a round
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
//! - **Its textures take no resource ID** (`IdUse::Runtime`, `resource_ids.rs`), and its queue
//!   touches no ID rule of Metal's device (`Device::own_command_queue_now`, `queue.rs`): the
//!   check runs on its thread while the program makes its textures, so a plain ID one of its
//!   textures held would move the program's next texture's ID past Metal's, and a texture
//!   reservation after it would be ignored; a queue of the program's would make the texture pool,
//!   keeping its first range beside a later texture reservation, and take the device's sampler
//!   ID, ignoring a later sampler reservation. The ID does not touch the image, its pages or its
//!   move, so neither does a verdict.
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
//!   share none, and the other nine share one. A failure under the GPU's stores (renders,
//!   storage writes, buffer writes) whose class's reads of those row ends passed fails that
//!   access and row ends in every class whose rounds passed them ([`carried_store_failures`]):
//!   the line write-back belongs to the store path every class shares, and a class whose pages
//!   the CPU maps uncached sees it less often. So the verdicts are the device's once every class
//!   is done, all at once. Counted (`WorkCounts::in_place_checks_*`).
//! - **Kept.** Once every class answered, the verdicts are written as one record in the shader
//!   cache's content directory (`shader_cache/store.rs`), named by the driver's identity (its
//!   version and the device's UUID among it) and the check's version ([`record_name`]), not the
//!   runtime's: a device of the same driver and version reads them as it is made and runs no
//!   check, whatever runtime release made it, until the check itself changes
//!   (`in_place::CHECK_VERSION`). The record keeps what the check measured, and an unchecked
//!   verdict whose cause lasts for the driver's identity: the device lacks an extension, a
//!   feature, an import, a format or an image of the shape ([`Unmade`]). One unchecked for a
//!   cause that may pass (pages or memory not made, an import refused, the writer thread not
//!   started, a first move, pass or kernel's stores that failed) is written unmeasured, so the
//!   next device checks that class and access again (`in_place::encode`); a page or image refused
//!   once in one process, or by the setup of one device, then answers for no later device. Where
//!   a verdict's rounds met both, it may pass. Each such refusal is noted with why and which
//!   cause it is (the report's lines after the verdicts).
//!   A failed write leaves the verdicts for the device's life. A forced failure is kept
//!   nowhere.
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
//!   allocation of the device's own, host-visible and exported as a dma-buf, mapped
//!   (`dma_buf::ExportedPages`: host-cached where a type is, and with the device-address flag its
//!   imports carry), and imported only as that dma-buf (`HostMemory::dma_buf_only`), never by
//!   their pointer, which a driver may take too (lavapipe takes its own mapped allocation's; ANV
//!   refuses it); the check's source answers each request over them with that dma-buf. The
//!   buffer's import, the image and its first move are the product's.

use crate::buffer_descriptor::MemoryExport;
use crate::check_draw::CheckDraw;
use crate::device::holds::Holder;
use crate::device::lock;
use crate::exit_guard;
use crate::host_memory::HostMemory;
use crate::in_place::{
    encode, mark_checking, record_name, verdict_index, Access, Padding, PagesClass, RowEnds,
    Verdict,
};
use crate::linear_plan::BufferLayout;
use crate::memory::{BoundMemory, Holds, NoImage};
use crate::values::ResourceOptions;
use crate::{
    AddressPlacement, Buffer, BufferContents, BufferDescriptor, ClearColor, CommandQueue, Device,
    InPlaceRefusal, LoadAction, RenderPassDescriptor, StoreAction, Texture, TextureDescriptor,
    TexturePlacement, TextureShadowReason,
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
/// whose line is 64 bytes: likewise), a 172x67 one of 4-byte texels at a 704-byte pitch, whose
/// rows end inside a 64-byte line and whose last row's pitch ends 64 bytes past a multiple of 128
/// and of 256 (so a device that writes back more than a line around a row's last texels changes
/// a byte past the image there, where the others end on such multiples), and one 3840x2160
/// surface of 4-byte texels, last. Those four
/// at a driver's own pitch run only where no shape of their texel size whose texels share a line
/// was made before them ([`Shape::runs`]): a device that makes images at an explicit pitch makes
/// the 512-byte ones first. A shape the device makes no image of in place is skipped: the product
/// makes none of it either.
pub(crate) const SHAPES: [Shape; 16] = [
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
    shape(PixelFormat::BGRA8Unorm, 4, 172, 67, 704),
    shape(PixelFormat::BGRA8Unorm, 4, 3840, 2160, 15360),
];

/// The shapes the check of storage writes of 1- and 2-byte texels writes
/// ([`Access::NarrowWrites`]), each with the view its kernel stores through (`R8_UINT`, `R16_UINT`:
/// `in_place::NARROW_TEXEL_BYTES`), as [`STORE_SHAPES`] lays its own out: rows one after another,
/// rows apart at the 256-byte pitch of the guests' 1- and 2-byte textures a shader writes (32x32
/// `R8Unorm` and `A8Unorm` among them), of widths whose rows end inside a 4-byte word and odd
/// heights, so that a driver's store of a word or a block past a row's or the image's end shows,
/// at the 16-byte pitch a driver that lays rows tight lays them at and at the 64-byte pitch of one
/// that lays them at the next multiple of 64 (run as there: [`Shape::runs`]), and one 1024x64
/// surface of 1-byte texels, last: rows as wide as the largest such guest texture's (1024x1024),
/// a sixteenth as many, which keeps its rounds' fill and compare short. A shape the device makes
/// no image of in place, or no dispatch over (a device without
/// `shaderStorageImageExtendedFormats`, whose kernels of those views it refuses), is skipped.
pub(crate) const NARROW_STORE_SHAPES: [(Shape, ShadowTexelView); 9] = [
    (
        shape(PixelFormat::R8Uint, 1, 256, 16, 256),
        ShadowTexelView::R8Uint,
    ),
    (
        shape(PixelFormat::R16Uint, 2, 128, 16, 256),
        ShadowTexelView::R16Uint,
    ),
    (
        shape(PixelFormat::R8Uint, 1, 61, 37, 256),
        ShadowTexelView::R8Uint,
    ),
    (
        shape(PixelFormat::R16Uint, 2, 45, 29, 256),
        ShadowTexelView::R16Uint,
    ),
    (
        tight(PixelFormat::R8Uint, 1, 101, 51, 112),
        ShadowTexelView::R8Uint,
    ),
    (
        tight(PixelFormat::R16Uint, 2, 33, 17, 80),
        ShadowTexelView::R16Uint,
    ),
    (
        tight(PixelFormat::R8Uint, 1, 101, 51, 128),
        ShadowTexelView::R8Uint,
    ),
    (
        tight(PixelFormat::R16Uint, 2, 33, 17, 128),
        ShadowTexelView::R16Uint,
    ),
    (
        shape(PixelFormat::R8Uint, 1, 1024, 64, 1024),
        ShadowTexelView::R8Uint,
    ),
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
pub(crate) const BUFFER_BYTES: u64 = 40 << 20;

/// The bytes of the pooled class's buffer: within half a pooled block, so that it lies in a
/// block of the runtime's pooled memory as the product's buffers of that class do
/// (`fresh_pages::FRESH_PAGES_PAST`), with room for every shape's window but the 3840x2160
/// surface's, which such a buffer's texture takes no more than a smaller one's.
const POOLED_BYTES: u64 = 24 << 20;

/// The bytes of `class`'s buffer.
fn class_bytes(class: PagesClass) -> u64 {
    match class {
        PagesClass::Pooled | PagesClass::FdBlocks => POOLED_BYTES,
        PagesClass::Pointer
        | PagesClass::Fresh
        | PagesClass::DmaBuf
        | PagesClass::ExportedFd
        | PagesClass::ExportedDmaBuf => BUFFER_BYTES,
    }
}

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

/// The check's thread: each class in turn, each of its accesses; then every verdict given at
/// once, store failures carried across classes ([`carried_store_failures`]), and the record.
fn run(weak: crate::device::WeakDevice) {
    mark_checking();
    crate::step_aside::mark_a_check_s_thread(crate::step_aside::Counted::AsTheDevice);
    let started = Instant::now();
    let mut stopped = false;
    // This run's verdicts, given once every class is done.
    let mut found: Vec<Found> = Vec::new();
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
                    found.push(Found {
                        passing: true,
                        ..Found::new(class, access, rows, Verdict::Unchecked)
                    });
                }
                continue;
            }
            let Some((verdicts, beside, passing)) = check_class(&device, class, access) else {
                // The process is exiting, or the program let the device go: nothing is kept.
                return;
            };
            for (&access, verdicts) in answered.iter().zip(verdicts) {
                for rows in RowEnds::ALL {
                    let verdict = verdicts[rows.index()];
                    found.push(Found {
                        beside,
                        passing: verdict == Verdict::Unchecked && passing[rows.index()],
                        ..Found::new(class, access, rows, verdict)
                    });
                }
            }
        }
    }
    let Some(device) = weak.hold() else {
        return;
    };
    let checks = &device.shared().in_place;
    // Every verdict, those known before the check ran among them, of which a store failure
    // carries to the passes of this run's other classes.
    let known: Vec<Found> = (PagesClass::ALL.into_iter())
        .flat_map(|class| Access::ALL.map(move |access| (class, access)))
        .flat_map(|(class, access)| RowEnds::ALL.map(move |rows| (class, access, rows)))
        .filter(|&at| !found.iter().any(|f| f.at() == at))
        .map(|(class, access, rows)| {
            Found::new(class, access, rows, checks.verdict(class, access, rows))
        })
        .filter(|known| known.verdict != Verdict::Pending)
        .chain(found.iter().copied())
        .collect();
    let carried = carried_store_failures(&known);
    // Where the record keeps a verdict unmeasured: one unchecked for a cause that may pass in
    // another run, which the next device checks again ([`Unmade`]).
    let mut unmeasured = Vec::new();
    for Found {
        class,
        access,
        rows,
        verdict,
        passing,
        ..
    } in found
    {
        let from = (carried.iter()).find(|c| (c.0, c.1, c.2) == (class, access, rows));
        let verdict = match from {
            Some(&(_, _, _, from)) => {
                let lines = match rows {
                    RowEnds::InLines => "",
                    RowEnds::OnLines => ", rows on lines",
                };
                checks.note(format!(
                    "{class:?} {}{lines}: failed as {from:?}'s did, a change under the GPU's \
                     stores, which every class's images share",
                    access.words()
                ));
                Verdict::Failed
            }
            None => verdict,
        };
        checks.set(class, access, rows, verdict);
        if verdict == Verdict::Unchecked && passing {
            unmeasured.push(verdict_index(class, access, rows));
        }
        device.shared().counts.in_place_check_ran(verdict);
    }
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
        let mut recorded = verdicts;
        for at in unmeasured {
            recorded[at] = Verdict::Pending;
        }
        let bytes = encode(&identity, recorded);
        // A failed write leaves the verdicts to the device's life alone.
        let _ = device
            .shared()
            .shader_cache
            .content_store()
            .write_record_soon(&record_name(&identity), bytes);
    }
    // The shadows made before the verdicts that pass wait for a switch before a call waiting for
    // the verdicts wakes: its next quiet commit moves them (`texture/storage_form.rs`).
    device.in_place_verdicts_given();
    checks.settle(Some(started.elapsed()));
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
/// storage writes or renders that changed no byte but of its shape's rows' padding, where that
/// padding is free ([`Shape::padding`]), fails [`Access::Writes`] (or [`Access::NarrowWrites`],
/// or [`Access::Renders`]) for its row ends and counts for [`Access::BufferWrites`] (or
/// [`Access::NarrowBufferWrites`], or [`Access::UnobservedRenders`]) as a round that kept every
/// byte, and the rounds go on. With them, whether every byte its rounds found changed lay beside
/// the texels (none was a texel); and, by row ends, whether an unchecked verdict's cause may pass
/// in another run ([`Unmade`]). `None` where the process is exiting, or the program's last drop
/// stopped the device's holds before a round (`device/holds.rs`).
fn check_class(
    device: &Device,
    class: PagesClass,
    access: Access,
) -> Option<(Vec<[Verdict; 2]>, bool, [bool; 2])> {
    let page = host_page();
    let line = device.shared().in_place.line();
    let unchecked = |passing| {
        (
            vec![[Verdict::Unchecked; 2]; access.answered().len()],
            true,
            [passing; 2],
        )
    };
    let length = class_bytes(class);
    let pages = match ClassPages::make(device, class, length) {
        Ok(pages) => pages,
        Err(unmade) => {
            device.shared().in_place.note(format!(
                "{class:?} {}: no pages to check: {unmade}",
                label(access)
            ));
            return Some(unchecked(!unmade.lasting));
        }
    };
    // The command queue an image that renders takes its passes on, the check's own, and the
    // draw it takes after them.
    let mut renders = match access {
        Access::Reads
        | Access::UnobservedRenders
        | Access::Writes
        | Access::BufferWrites
        | Access::NarrowWrites
        | Access::NarrowBufferWrites => None,
        Access::Renders => {
            let made = exit_guard::enter().map(|_busy| {
                let queue = device.own_command_queue_now()?;
                Ok::<_, crate::Error>((queue, CheckDraw::new(device)?))
            });
            match made? {
                Ok(renders) => Some(renders),
                Err(error) => {
                    let unmade = Unmade::of_error("its queue or draw was not made", &error);
                    device.shared().in_place.note(format!(
                        "{class:?} {}: no pass to check: {unmade}",
                        label(access)
                    ));
                    return Some(unchecked(!unmade.lasting));
                }
            }
        }
    };
    // An image a shader writes takes the shapes its kernel stores of its texels' width, each
    // with its view; those whose texels share no line with bytes beside them first.
    let stored = |shapes: &[(Shape, ShadowTexelView)]| {
        let stored = shapes.iter().map(|&(shape, texel)| (shape, Some(texel)));
        stored.collect::<Vec<_>>()
    };
    let mut shapes: Vec<(Shape, Option<ShadowTexelView>)> = match access {
        Access::Writes | Access::BufferWrites => stored(&STORE_SHAPES),
        Access::NarrowWrites | Access::NarrowBufferWrites => stored(&NARROW_STORE_SHAPES),
        Access::Reads | Access::Renders | Access::UnobservedRenders => {
            SHAPES.map(|shape| (shape, None)).to_vec()
        }
    };
    shapes.sort_by_key(|(shape, _)| shape.rows(line) != RowEnds::OnLines);
    let forced = device.shared().in_place.forced_failure();
    let mut tally = Tally::default();
    // The texel sizes of the shapes whose texels share a line made so far (`Shape::runs`).
    let mut made_sharing = Vec::new();
    // Whether a round that made no image for a cause that may pass, and one for a cause that
    // lasts, was noted: the first of each is, with why. And by row ends, whether one made none
    // for a cause that may pass in another run.
    let mut unmade_noted = [false; 2];
    let mut passing = [false; 2];
    for (shape, texel) in shapes {
        if shape.window(page) > length || !shape.runs(line, &made_sharing) {
            continue;
        }
        let rows = shape.rows(line);
        // A round of an access that answers the form that lets the padding change too (storage
        // writes, renders), over a shape whose rows' padding is free, tells a change there apart.
        let free = access.answered().len() > 1 && shape.padding(line) == Padding::Free;
        for round in 0..ROUNDS {
            if !device.holds_go_on(Holder::InPlaceCheck) {
                return None;
            }
            step_aside(device);
            if word_overwritten(device) {
                return Some(unchecked(true));
            }
            let outcome = {
                let _busy = exit_guard::enter()?;
                match texel {
                    Some(texel) => {
                        store_round(device, &pages, (shape, texel, free), round, (page, line))
                    }
                    None => {
                        check_round(&pages, (shape, free), round, (page, line), renders.as_mut())
                    }
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
                Round::NotMade(unmade) => {
                    passing[rows.index()] |= !unmade.lasting;
                    let noted = &mut unmade_noted[usize::from(unmade.lasting)];
                    if !*noted {
                        *noted = true;
                        let lines = match rows {
                            RowEnds::InLines => "",
                            RowEnds::OnLines => ", rows on lines",
                        };
                        device.shared().in_place.note(format!(
                            "{class:?} {}{lines}: no {:?} image of {}x{} with rows {} bytes \
                             apart: {unmade}",
                            label(access),
                            shape.format,
                            shape.width,
                            shape.height,
                            shape.bytes_per_row
                        ));
                    }
                    break;
                }
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
            let (place, beside) = match offset.checked_rem(shape.bytes_per_row) {
                _ if offset >= image => ("after the image", true),
                Some(column) if column >= shape.width * shape.texel => {
                    ("after a row's texels", true)
                }
                _ => ("a texel", false),
            };
            let operations = match access {
                Access::Reads => "the first move",
                Access::Renders | Access::UnobservedRenders => {
                    "the first move, feedback loop moves, a pass's clear and store or a draw"
                }
                Access::Writes | Access::BufferWrites => {
                    "the first move or a kernel's storage writes"
                }
                Access::NarrowWrites | Access::NarrowBufferWrites => {
                    "the first move or a kernel's storage writes of 1- or 2-byte texels"
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
                return Some((tally.verdicts(access, Some(rows), forced), beside, passing));
            }
        }
    }
    Some((tally.verdicts(access, None, forced), true, passing))
}

/// `access` in a note's words, as the report names its verdicts (`in_place_check_report`).
fn label(access: Access) -> &'static str {
    match access {
        Access::Reads => "reads",
        Access::Renders => "renders",
        Access::UnobservedRenders => "unobserved renders",
        Access::Writes => "writes",
        Access::BufferWrites => "buffer writes",
        Access::NarrowWrites => "narrow writes",
        Access::NarrowBufferWrites => "narrow buffer writes",
    }
}

/// One verdict of the check: its class, access, row ends and answer, and whether every byte
/// its class's rounds of the access found changed lay beside the texels (none was a texel).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Found {
    pub(crate) class: PagesClass,
    pub(crate) access: Access,
    pub(crate) rows: RowEnds,
    pub(crate) verdict: Verdict,
    pub(crate) beside: bool,
    /// Of an unchecked verdict, whether its cause may pass in another run ([`Unmade`]): kept
    /// unmeasured, checked again by the next device.
    pub(crate) passing: bool,
}

impl Found {
    /// The verdict of `class`, `access` and `rows`, any byte found changed beside the texels.
    pub(crate) fn new(class: PagesClass, access: Access, rows: RowEnds, verdict: Verdict) -> Found {
        Found {
            class,
            access,
            rows,
            verdict,
            beside: true,
            passing: false,
        }
    }

    /// Its class, access and row ends.
    fn at(&self) -> (PagesClass, Access, RowEnds) {
        (self.class, self.access, self.rows)
    }
}

/// The passes of `found` a store failure fails, each with the class whose failure it is: a
/// class's pass of an access whose rounds store (renders, over padding declared unobserved too:
/// a pass's clear and store, and a draw; storage writes and buffer writes, of narrow texels too:
/// a kernel's stores) for texels whose
/// rows end as another
/// class's failure of that access did, where every byte that class's rounds found changed lay
/// beside the texels and its reads of those row ends passed, so that its failure was the
/// stores' and not the first move's.
///
/// Such a failure is a byte beside the texels changed under the GPU's stores: a line of its
/// cache that holds a row's last texels written back whole, with what it read beside them. The
/// write-back belongs to the store path, which every class's images share, while how often a
/// class's rounds see it does not: the CPU's write must land between the GPU's read of the line
/// and its write-back, and a class whose pages the CPU maps uncached sweeps them far slower. On
/// the Iris Xe, ANV maps its exported memory so (the dma-buf class's: each byte's read about
/// 80 ns), and the storage-write rounds of shapes whose rows end inside lines saw the padding
/// change in 3 of 18 rounds there, against 10 and 14 of 18 of the pointer and fresh-pages
/// classes, so that the dma-buf class passed one fresh check in three.
///
/// Nothing else carries: a texel read otherwise (whether the CPU sees the GPU's stores depends
/// on the class's mapping and import), a reads failure (whether the GPU sees the CPU's bytes
/// does too), an unchecked verdict, or a class whose pages could not be made.
pub(crate) fn carried_store_failures(
    found: &[Found],
) -> Vec<(PagesClass, Access, RowEnds, PagesClass)> {
    let stores = |access| {
        matches!(
            access,
            Access::Renders
                | Access::UnobservedRenders
                | Access::Writes
                | Access::BufferWrites
                | Access::NarrowWrites
                | Access::NarrowBufferWrites
        )
    };
    let of = |at| (found.iter()).find(|f| f.at() == at).map(|f| f.verdict);
    let by_stores = |f: &Found| {
        f.verdict == Verdict::Failed
            && f.beside
            && stores(f.access)
            && of((f.class, Access::Reads, f.rows)) == Some(Verdict::Passed)
    };
    (found.iter())
        .filter(|f| f.verdict == Verdict::Passed && stores(f.access))
        .filter_map(|f| {
            (found.iter())
                .find(|g| {
                    g.class != f.class && (g.access, g.rows) == (f.access, f.rows) && by_stores(g)
                })
                .map(|g| (f.class, f.access, f.rows, g.class))
        })
        .collect()
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
    /// row ends of the first that changed any byte failed; then, of [`Access::Renders`],
    /// [`Access::Writes`] and [`Access::NarrowWrites`], those of [`Access::UnobservedRenders`],
    /// [`Access::BufferWrites`] and [`Access::NarrowBufferWrites`], of its rounds that changed no
    /// byte but of that padding, those of `changed` failed.
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
    /// The texture took no image in place (the device makes none of the shape there), why.
    NotMade(Unmade),
    /// A byte of its shape's rows' padding read other than the CPU left it, where a round of
    /// storage writes or renders tells that padding apart ([`Shape::padding`]), and every other
    /// byte read right (each past the last row's pitch among them): where the first was, what was
    /// due, what was read.
    Padded((u64, u8, u8)),
    /// A byte read other than the CPU left it: where, what was due, what was read.
    Changed((u64, u8, u8)),
}

/// One round of `shape` over `pages`, inside the caller's section: the pattern, the texture, the
/// writes beside its image's first move (and, for an image that renders, `renders` its queue and
/// draw, beside its feedback loop moves, a pass that clears and stores it and a draw of its last
/// column), and the bytes compared: a change of a byte of the rows' padding of an image that
/// renders told apart where `free` ([`Round::Padded`]). The window starts a page of `page` bytes;
/// the writer's walk keeps to the bytes beside the texels in lines of `line` bytes most
/// ([`Walk`]).
fn check_round(
    pages: &ClassPages,
    (shape, free): (Shape, bool),
    round: usize,
    (page, line): (u64, u64),
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
    let texture = match (pages.buffer).new_runtime_texture(&descriptor, 0, shape.bytes_per_row) {
        Ok(texture) => texture,
        Err(error) => return Round::NotMade(unmade(pages, &descriptor, shape, &error)),
    };
    let stop = AtomicBool::new(false);
    let started = Barrier::new(2);
    let outcome = std::thread::scope(|scope| {
        let builder = std::thread::Builder::new().name("m2v-in-place-writes".to_owned());
        let Ok(writer) = exit_guard::spawn_scoped(scope, builder, || {
            bytes.write_until(&pattern, (texels, free, line), (&stop, &started))
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
    let written = match outcome {
        Some((Some(Ok(())), written)) => written,
        Some((Some(Err(error)), _)) => {
            let why = format!("its first move or pass failed: {error}");
            return Round::NotMade(Unmade::passing(why));
        }
        Some((None, _)) => return Round::NotMade(unplaced(pages, &descriptor, shape)),
        None => return Round::NotMade(Unmade::passing("the round's writer did not run")),
    };
    drop(texture);
    let texels = texels.map(|shape| (shape, Due::Zero));
    bytes.compare(&pattern, (texels, free), written)
}

/// Why the check's texture of `shape` over `pages`, made with `descriptor`, was not made, its
/// making's `error` given: on the check's thread a making whose route takes no image in place
/// refuses the texture with an internal error (`texture.rs`), which says nothing of the cause, so
/// the placement's answer decides where it names a cause that lasts ([`unplaced`]); the error
/// does where it does not.
fn unmade(
    pages: &ClassPages,
    descriptor: &TextureDescriptor,
    shape: Shape,
    error: &crate::Error,
) -> Unmade {
    let placed = unplaced(pages, descriptor, shape);
    match placed.lasting {
        true => Unmade::lasting(format!("its making failed: {error}; {}", placed.why)),
        false => Unmade::of_error(&format!("{}; its making failed", placed.why), error),
    }
}

/// Why the check's texture of `shape` over `pages`, made with `descriptor`, took no image in
/// place: its placement's answer (`Buffer::texture_placement`), asked on the check's thread, as
/// its making asked the route. A cause that lasts where the answer is one the image's making
/// decides of the device and the shape before it binds memory: the device makes no such image,
/// lays none at the pitch, or none whose memory takes the pages' place
/// ([`InPlaceRefusal`]), or the route takes none of the shape. One that may pass where the
/// making answers in place, which only the bind then refused (the pages' dma-buf, or the
/// driver's import of it), or the pages were detached, the source made no dma-buf of them, or
/// another verdict was not known.
fn unplaced(pages: &ClassPages, descriptor: &TextureDescriptor, shape: Shape) -> Unmade {
    let placement = match (pages.buffer).texture_placement(descriptor, 0, shape.bytes_per_row) {
        Ok(placement) => placement,
        Err(error) => {
            let what = "it took no image in place, its placement refused";
            return Unmade::of_error(what, &error);
        }
    };
    let why = format!("it took no image in place: {placement:?}");
    let lasting = match placement {
        TexturePlacement::InPlace | TexturePlacement::Pending | TexturePlacement::NotHostPages => {
            false
        }
        TexturePlacement::Shadow(TextureShadowReason::Refused(refusal)) => match refusal {
            InPlaceRefusal::Rows
            | InPlaceRefusal::Image
            | InPlaceRefusal::Pitch
            | InPlaceRefusal::Placement
            | InPlaceRefusal::DmaBuf => true,
            InPlaceRefusal::Detached | InPlaceRefusal::Source => false,
        },
        TexturePlacement::Shadow(
            TextureShadowReason::CheckFailed { .. } | TextureShadowReason::Unchecked { .. },
        ) => false,
        TexturePlacement::Shadow(_) | TexturePlacement::OtherShape => true,
    };
    Unmade { why, lasting }
}

/// Why the check made no pages, pass or image to check, as its note says it, and whether that
/// cause lasts for the driver's identity, so that the class's unchecked verdict is kept in the
/// device's record as a measured one is (the device lacks an extension, a feature, an import, a
/// format or an image of the shape), or may pass in another run, so that the verdict is kept
/// unmeasured and the next device checks it again (pages or memory not made or not mapped, an
/// import refused or failed, the writer thread not started, a first move, pass or kernel's
/// stores that failed).
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Unmade {
    why: String,
    lasting: bool,
}

impl Unmade {
    /// A cause that lasts for the driver's identity: `why`.
    pub(crate) fn lasting(why: impl Into<String>) -> Unmade {
        Unmade {
            why: why.into(),
            lasting: true,
        }
    }

    /// A cause that may pass in another run: `why`.
    pub(crate) fn passing(why: impl Into<String>) -> Unmade {
        Unmade {
            why: why.into(),
            lasting: false,
        }
    }

    /// `what`, refused with `error`: a cause that lasts where the error is the device's gap
    /// ([`Error::Unsupported`](crate::Error::Unsupported)) or a refusal plain Metal makes of the
    /// arguments ([`Error::Invalid`](crate::Error::Invalid)), each the same answer of the same
    /// call on the same driver; one that may pass for any other (out of memory, a lost device,
    /// another Vulkan failure, the runtime's own gap, which a later build fills).
    pub(crate) fn of_error(what: &str, error: &crate::Error) -> Unmade {
        Unmade {
            why: format!("{what}: {error}"),
            lasting: matches!(
                error,
                crate::Error::Unsupported(_) | crate::Error::Invalid(_)
            ),
        }
    }
}

impl std::fmt::Display for Unmade {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kept = match self.lasting {
            true => "a cause that lasts, kept",
            false => "a cause that may pass, checked again",
        };
        write!(f, "{} ({kept})", self.why)
    }
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
/// byte of the rows' padding told apart where `free` ([`Round::Padded`]). The window starts a
/// page of `page` bytes; the writer's walk keeps to the bytes beside the texels in lines of
/// `line` bytes most ([`Walk`]).
fn store_round(
    device: &Device,
    pages: &ClassPages,
    (shape, texel, free): (Shape, ShadowTexelView, bool),
    round: usize,
    (page, line): (u64, u64),
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
    let texture = match (pages.buffer).new_runtime_texture(&descriptor, 0, shape.bytes_per_row) {
        Ok(texture) => texture,
        Err(error) => return Round::NotMade(unmade(pages, &descriptor, shape, &error)),
    };
    // The bytes the kernel stores, row after row, and the part it keeps them in too: buffers
    // outside the Metal address space, as the check's are (the module docs). A texel takes the
    // view's words there: a 1- or 2-byte texel a whole word, its bytes low and zero above them.
    let held = 4 * u64::from(texel.words());
    let row = shape.width * held;
    let length = row * shape.height;
    let buffer =
        || device.new_buffer_placed(length, ResourceOptions::shared(), AddressPlacement::Outside);
    let (source, kept) = match (buffer(), buffer()) {
        (Ok(source), Ok(kept)) => (source, kept),
        (Err(error), _) | (_, Err(error)) => {
            let why = format!("the kernel's buffers were refused: {error}");
            return Round::NotMade(Unmade::passing(why));
        }
    };
    let Some(words) = source.contents() else {
        return Round::NotMade(Unmade::passing("the kernel's buffer has no contents"));
    };
    for y in 0..shape.height {
        let from = y * shape.bytes_per_row;
        let to = (y * row) as usize;
        if held == shape.texel {
            // SAFETY: the source's row `to..to + row`, of its `length`, mapped and writable,
            // which nothing else uses yet.
            unsafe { pattern.store_row(from, row, words, to) };
            continue;
        }
        for x in 0..shape.width {
            let mut word = [0u8; 4];
            for (at, byte) in word.iter_mut().enumerate().take(shape.texel as usize) {
                *byte = pattern.stored_at(from + x * shape.texel + at as u64);
            }
            let at = to + (x * held) as usize;
            // SAFETY: the source's 4 bytes from `at`, of its `length`, mapped and writable, which
            // nothing else uses yet.
            unsafe { std::ptr::copy_nonoverlapping(word.as_ptr(), words.as_ptr().add(at), 4) };
        }
    }
    let stop = AtomicBool::new(false);
    let started = Barrier::new(2);
    let extent = (shape.width as u32, shape.height as u32);
    let outcome = std::thread::scope(|scope| {
        let builder = std::thread::Builder::new().name("m2v-in-place-writes".to_owned());
        let Ok(writer) = exit_guard::spawn_scoped(scope, builder, || {
            bytes.write_until(&pattern, (Some(shape), free, line), (&stop, &started))
        }) else {
            return None;
        };
        let stopping = Stopping(&stop);
        started.wait();
        let done = texture.store_in_place((&source, &kept), texel, extent);
        drop(stopping);
        Some((done, writer.join().ok()?))
    });
    let written = match outcome {
        Some((Some(Ok(())), written)) => written,
        Some((Some(Err(error)), _)) => {
            let why = format!("its first move or the kernel's stores failed: {error}");
            return Round::NotMade(Unmade::passing(why));
        }
        Some((None, _)) => return Round::NotMade(unplaced(pages, &descriptor, shape)),
        None => return Round::NotMade(Unmade::passing("the round's writer did not run")),
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
    /// there; inside a section of its own. Why not, where it does not (the check's note).
    fn make(device: &Device, class: PagesClass, length: u64) -> Result<ClassPages, Unmade> {
        let Some(_busy) = exit_guard::enter() else {
            return Err(Unmade::passing("the process is exiting"));
        };
        match class {
            PagesClass::Pointer => pointer_pages(device, length),
            // In fresh pages wherever the device takes them, whatever its submission checks
            // found a held import costs: a rebind's copy under an image in place lies in them
            // where this class passed (`buffer_descriptor.rs`'s `copied_pages`), the product's
            // long buffers where that cost is none (`fresh_pages.rs`).
            PagesClass::Fresh => {
                let buffer = (device.fresh_pages_check_buffer(length))
                    .map_err(|error| Unmade::of_error("its buffer was refused", &error))?;
                if !buffer.in_fresh_pages() {
                    return Err(unimported(device, "its buffer lies in no fresh pages"));
                }
                let start = contents(&buffer)?;
                Ok(ClassPages { buffer, start })
            }
            PagesClass::DmaBuf => dma_buf_pages(device, length),
            // A buffer of the runtime's own in its pooled memory exported as no handle type
            // (where the device's own buffers lie in blocks exportable as opaque file
            // descriptors, a block of plain buffers: `memory.rs`'s `Holds::FdBuffers`) that
            // declares the bytes between a texture's rows unobserved, as the product's buffers
            // whose textures take such an image do (`texture.rs`'s `alias_image`): only where the
            // device makes `DRM_FORMAT_MOD_LINEAR` images.
            PagesClass::Pooled => {
                if !device.makes_modifier_images() {
                    let why = "the device makes no DRM_FORMAT_MOD_LINEAR images";
                    return Err(Unmade::lasting(why));
                }
                let buffer = (device.pooled_check_buffer(length, Holds::Buffers))
                    .map_err(|error| Unmade::of_error("its buffer was refused", &error))?;
                if buffer.in_fresh_pages() {
                    let why = "its buffer lies in fresh pages, not pooled memory";
                    return Err(Unmade::passing(why));
                }
                buffer.declare_padding_unobserved();
                let start = contents(&buffer)?;
                Ok(ClassPages { buffer, start })
            }
            PagesClass::ExportedFd => exported_pages(device, MemoryExport::OpaqueFd, length),
            PagesClass::ExportedDmaBuf => exported_pages(device, MemoryExport::DmaBuf, length),
            // A buffer of the runtime's own in a block of the kind its buffers take where the
            // device allocates their blocks exportable (`memory.rs`'s `Holds::FdBuffers`), whatever
            // this class's verdicts, which decide whether they lie there
            // (`Device::fd_blocks_checked`): its textures take the image a texture over such a
            // buffer takes, the device's linear image made for the blocks' handle type
            // (`texture.rs`'s `alias_image`). It declares the bytes between a texture's rows
            // unobserved, which none of the device's buffers there does (they lie in plain
            // blocks: `buffer.rs`'s `NewBufferOf::Unobserved`), so that its textures over rows
            // that lie apart take that image too, and the class's verdicts answer every access
            // the pooled class's do (`blocks_take`).
            PagesClass::FdBlocks => {
                let exportable = lock(&device.shared().memory).fd_buffer_types;
                if let Err(why) = exportable {
                    return Err(Unmade::lasting(why));
                }
                // Made only where such a block holds every image a texture over a plain block's
                // buffer that keeps its padding takes in place (`texture/fd_blocks_cover.rs`),
                // noted in the memory log.
                #[cfg(unix)]
                if let Err(why) = device.fd_blocks_cover() {
                    let line = format!("the runtime's own buffers lie in plain blocks: {why}");
                    lock(&device.shared().memory).note(line);
                    return Err(Unmade::lasting(why));
                }
                let buffer = (device.pooled_check_buffer(length, Holds::FdBuffers))
                    .map_err(|error| Unmade::of_error("its buffer was refused", &error))?;
                match buffer.bound_memory() {
                    BoundMemory::Blocks => {}
                    _ => {
                        let why = "its buffer lies in no block allocated exportable";
                        return Err(Unmade::passing(why));
                    }
                }
                buffer.declare_padding_unobserved();
                let start = contents(&buffer)?;
                Ok(ClassPages { buffer, start })
            }
        }
    }
}

/// An exportable buffer of the runtime's own of `length` bytes, `Shared`, whose memory it exports
/// as `export`'s handle type, outside the Metal address space, that declares the bytes between a
/// texture's rows unobserved, as the product's buffers whose textures take an image there may
/// (`exported_memory.rs`, `texture.rs`'s `alias_image`). Why not: a cause that lasts where the
/// device exports no such memory, or an opaque file descriptor only as a buffer's dedicated
/// allocation, to which no image is bound. A dma-buf's dedicated allocation is taken: its
/// textures' images are bound to a dedicated import of its own dma-buf, as the product's are
/// wherever no image binds in the memory (`texture.rs`'s `exported_dma_buf_alias`).
fn exported_pages(
    device: &Device,
    export: MemoryExport,
    length: u64,
) -> Result<ClassPages, Unmade> {
    let shared = ResourceOptions::shared();
    let place = (AddressPlacement::Outside.into(), None);
    let made = device.exportable_buffer("in-place check", length, shared, place, export);
    let buffer = match made {
        Ok(buffer) => device.visible(buffer),
        Err(error @ crate::Error::NotImplemented(_)) => {
            return Err(Unmade::lasting(format!(
                "the device exports no such memory: {error}"
            )));
        }
        Err(error) => return Err(Unmade::of_error("its buffer was refused", &error)),
    };
    match buffer.bound_memory() {
        BoundMemory::External(_) => {}
        BoundMemory::NoImage(NoImage::Dedicated) if export == MemoryExport::DmaBuf => {}
        BoundMemory::NoImage(no) => return Err(Unmade::lasting(no.words())),
        BoundMemory::Own | BoundMemory::Blocks => {
            return Err(Unmade::passing("its buffer's memory is not exported"))
        }
    }
    buffer.declare_padding_unobserved();
    let start = contents(&buffer)?;
    Ok(ClassPages { buffer, start })
}

/// Where `buffer`'s byte 0 lies for the CPU; why not, where it has no contents.
fn contents(buffer: &Buffer) -> Result<NonNull<u8>, Unmade> {
    (buffer.contents()).ok_or_else(|| Unmade::passing("its buffer has no contents"))
}

/// Pages of `device`'s that its imports did not take as the class's, `why`: a cause that lasts
/// where the device imports no host pointer at all (`VK_EXT_external_memory_host`), else one that
/// may pass (the import of these pages refused or failed, `copied.rs`).
fn unimported(device: &Device, why: &str) -> Unmade {
    let pointers = (device.shared().host_import.as_ref()).is_ok_and(|i| i.pointers().is_some());
    match pointers {
        true => Unmade::passing(why),
        false => Unmade::lasting(why),
    }
}

/// A no-copy buffer of `length` bytes over anonymous pages mapped for it, imported by their
/// pointer, which its keep-alive unmaps ([`Anonymous`]). Why not, where the device makes none,
/// or does not import the pointer (dma-bufs, or a copied host buffer, `copied.rs`).
#[cfg(unix)]
fn pointer_pages(device: &Device, length: u64) -> Result<ClassPages, Unmade> {
    let start = map_anonymous(length).ok_or_else(|| {
        Unmade::passing(format!("no anonymous mapping of {length} bytes was made"))
    })?;
    let pages: crate::KeepAlive = Box::new(Anonymous { start, length });
    // SAFETY: the pages are this process's private anonymous mapping, readable and writable,
    // which stays until the keep-alive unmaps it, and which nothing else maps or frees.
    let memory = unsafe { HostMemory::new(start) };
    let buffer = no_copy_kept(device, memory, length, pages)?;
    if buffer.over_dma_buf() {
        let why = "the device imported its pages as dma-bufs, not by their pointer";
        return Err(unimported(device, why));
    }
    if buffer.copied() {
        let why = "the device copied its pages, which it did not import";
        return Err(unimported(device, why));
    }
    Ok(ClassPages { buffer, start })
}

/// A no-copy buffer of `length` bytes over `memory`, whose pages `pages` keeps mapped: a
/// keep-alive, not a deallocator, so the buffer's release runs at Metal's moment and waits for no
/// batch submitted before it (`HostPages`'s release), and `pages` goes once the import is freed
/// (`KeepAlive`). Why not, where the device refuses the buffer, `pages` dropped with the refusal.
#[cfg(unix)]
fn no_copy_kept(
    device: &Device,
    memory: HostMemory,
    length: u64,
    pages: crate::KeepAlive,
) -> Result<Buffer, Unmade> {
    let contents = BufferContents::NoCopy {
        memory,
        deallocator: None,
        keep_alive: Some(pages),
    };
    let descriptor = BufferDescriptor::new(length, ResourceOptions::shared(), contents)
        .with_placement(AddressPlacement::Outside);
    (device.new_buffer_with_descriptor_now(descriptor))
        .map_err(|error| Unmade::of_error("its no-copy buffer was refused", &error))
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
fn pointer_pages(_device: &Device, _length: u64) -> Result<ClassPages, Unmade> {
    Err(Unmade::lasting(
        "the system maps no anonymous pages for the check",
    ))
}

/// No dma-bufs on this system: the class is unchecked.
#[cfg(not(unix))]
fn dma_buf_pages(_device: &Device, _length: u64) -> Result<ClassPages, Unmade> {
    Err(Unmade::lasting("the system has no dma-bufs"))
}

/// A no-copy buffer of `length` bytes over pages the device refuses by their pointer, imported
/// as dma-bufs: an allocation of the device's own, host-visible and exported as a dma-buf, mapped
/// (the module docs).
#[cfg(unix)]
fn dma_buf_pages(device: &Device, length: u64) -> Result<ClassPages, Unmade> {
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
        let how = match buffer.copied() {
            true => "copied them",
            false => "imported them otherwise",
        };
        return Err(Unmade::passing(format!(
            "the device {how}, not as their dma-buf"
        )));
    }
    Ok(ClassPages { buffer, start })
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

/// Zeros a row's cleared texels are compared against, a chunk at a time (`Bytes::compare`).
static ZEROS: [u8; 4096] = [0; 4096];

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

    /// The byte a kernel stores at `offset` of the window.
    fn stored_at(&self, offset: u64) -> u8 {
        self.stored[(offset % PERIOD as u64) as usize]
    }

    /// Copy the bytes a kernel stores at the window's `length` bytes from `offset` into `to`'s
    /// bytes from `at`, a period's run at a time.
    ///
    /// # Safety
    ///
    /// `to`'s bytes `at..at + length` are mapped and writable, and nothing else uses them
    /// meanwhile.
    unsafe fn store_row(&self, offset: u64, length: u64, to: NonNull<u8>, mut at: usize) {
        for (from, count) in self.spans(offset, length as usize) {
            // SAFETY: the caller's bytes `at..at + count` of `to`, within its contract's; the
            // pattern's bytes are apart from them.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.stored[from..].as_ptr(),
                    to.as_ptr().add(at),
                    count,
                )
            };
            at += count;
        }
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

/// What a round's writer wrote ([`Bytes::write_until`]), by its [`Walk`]: of the stride's walk,
/// the sweep it was in as it stopped and how many of the stride's bytes that sweep reached (each
/// byte before that one written by it, each after by the sweep before); the bytes beside the
/// texels in their lines ([`near_bytes`]), in their walk's order, and of that walk the same; the
/// first byte it found other than it left it, where it found one (its offset, what was due and
/// what was read); and the first of its shape's rows' padding it found so, where it told that
/// padding apart.
pub(crate) struct Written {
    pub(crate) sweep: u64,
    pub(crate) reached: u64,
    pub(crate) near: Vec<u64>,
    pub(crate) near_sweep: u64,
    pub(crate) near_reached: u64,
    pub(crate) changed: Option<(u64, u8, u8)>,
    pub(crate) padding: Option<(u64, u8, u8)>,
}

/// How many bytes of its walk round the bytes beside the texels in their lines ([`near_bytes`])
/// a round's writer writes after each byte of its stride's walk ([`Walk`]).
pub(crate) const NEAR_STEPS: u32 = 8;

/// The bytes of a window of `length` bytes beside `shape`'s texels in the lines of `line` bytes
/// that hold them, in order, each once: of each row, the bytes of the line its first texel lies
/// in and of the line its last lies in, but the texels (every line between holds texels alone).
/// A device that writes back a line it read whole changes these alone of the bytes beside the
/// texels: what they held when it read them is put back over the CPU's writes since.
pub(crate) fn near_bytes(shape: Shape, line: u64, length: u64) -> Vec<u64> {
    let row = shape.width * shape.texel;
    let mut near: Vec<u64> = (0..shape.height)
        .flat_map(|y| {
            let start = y * shape.bytes_per_row;
            let head = start - start % line;
            let tail = (start + row).next_multiple_of(line);
            let first_end = (head + line).min(tail);
            let last_start = (tail - line).max(first_end);
            (head..first_end).chain(last_start..tail)
        })
        .filter(|&offset| offset < length && !shape.is_texel(offset))
        .collect();
    near.sort_unstable();
    near.dedup();
    near
}

/// The order a round's writer writes a window's bytes in ([`Bytes::write_until`]): the stride's
/// walk over the window ([`Bytes::stride`]), passing over the texels where a shape names them
/// and the bytes beside them in their lines ([`near_bytes`]), and after each of its bytes
/// [`NEAR_STEPS`] of a walk round those bytes beside the texels in their lines in order, each
/// walk sweep after sweep. A byte of the stride's walk comes back once a sweep of the window; a
/// byte beside the texels in their lines once every pass round them, a few writes for each of
/// them, so that a device's read and write-back of such a line meets a write of the CPU's
/// between them however short the span, and wherever the stride's walk is as it comes.
#[derive(Clone)]
pub(crate) struct Walk {
    stride: u64,
    length: u64,
    texels: Option<Shape>,
    near: Vec<u64>,
    /// Which of the window's bytes are in `near`, a bit each (none where it is empty).
    is_near: Vec<u64>,
    sweep: u64,
    n: u64,
    near_sweep: u64,
    near_at: u64,
    /// The steps round the bytes beside the texels before the stride's walk's next byte.
    owed: u32,
}

impl Walk {
    /// The walk over `bytes`, passing over `texels`' texels where it names a shape, round the
    /// bytes beside them in lines of `line` bytes.
    pub(crate) fn new(bytes: &Bytes, texels: Option<Shape>, line: u64) -> Walk {
        let near = match texels {
            Some(shape) => near_bytes(shape, line, bytes.length),
            None => Vec::new(),
        };
        let is_near = bits_of(&near, bytes.length);
        Walk {
            stride: bytes.stride(),
            length: bytes.length,
            texels,
            near,
            is_near,
            sweep: 1,
            n: 0,
            near_sweep: 1,
            near_at: 0,
            owed: 0,
        }
    }

    /// The next byte the writer writes, with the sweep of its walk that writes it.
    pub(crate) fn step(&mut self) -> (u64, u64) {
        if self.owed > 0 {
            self.owed -= 1;
            let offset = self.near[self.near_at as usize];
            let sweep = self.near_sweep;
            self.near_at += 1;
            if self.near_at == self.near.len() as u64 {
                (self.near_at, self.near_sweep) = (0, self.near_sweep + 1);
            }
            return (offset, sweep);
        }
        loop {
            let offset = nth(self.n, self.stride, self.length);
            let sweep = self.sweep;
            self.n += 1;
            if self.n == self.length {
                (self.n, self.sweep) = (0, self.sweep + 1);
            }
            if self.texels.is_some_and(|shape| shape.is_texel(offset))
                || holds(&self.is_near, offset)
            {
                continue;
            }
            if !self.near.is_empty() {
                self.owed = NEAR_STEPS;
            }
            return (offset, sweep);
        }
    }

    /// What the walk wrote so far, with the first byte the writer found other than it left it
    /// and the first of the padding so ([`Written`]).
    pub(crate) fn written(
        self,
        changed: Option<(u64, u8, u8)>,
        padding: Option<(u64, u8, u8)>,
    ) -> Written {
        Written {
            sweep: self.sweep,
            reached: self.n,
            near: self.near,
            near_sweep: self.near_sweep,
            near_reached: self.near_at,
            changed,
            padding,
        }
    }
}

/// A bit for each of a window's `length` bytes, set for those of `offsets`: none where it is
/// empty.
fn bits_of(offsets: &[u64], length: u64) -> Vec<u64> {
    if offsets.is_empty() {
        return Vec::new();
    }
    let mut bits = vec![0u64; length.div_ceil(64) as usize];
    for &offset in offsets {
        bits[(offset / 64) as usize] |= 1 << (offset % 64);
    }
    bits
}

/// Whether `bits` ([`bits_of`]) has the byte at `offset` set.
fn holds(bits: &[u64], offset: u64) -> bool {
    (bits.get((offset / 64) as usize)).is_some_and(|word| word & 1 << (offset % 64) != 0)
}

/// The `n`th byte of a stride's walk at `stride` over a window of `length` bytes.
fn nth(n: u64, stride: u64, length: u64) -> u64 {
    let product = u128::from(n) * u128::from(stride);
    (product % u128::from(length)) as u64
}

/// What sweep `sweep` of a round's writer changes the pattern by: nothing before the first
/// (`sweep` 0), then each of the 255 other byte values in turn, so that each sweep's bytes differ
/// from the pattern and from the sweep's before.
pub(crate) fn mark(sweep: u64) -> u8 {
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

    /// The `n`th byte of the stride's walk, at `stride` ([`Walk`]).
    fn nth(&self, n: u64, stride: u64) -> u64 {
        nth(n, stride, self.length)
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
    /// them), in the [`Walk`]'s order (round the bytes beside the texels in their lines of
    /// `line` bytes most), each of its two walks sweep after sweep, until `stop`: sweep `k` of
    /// either writes the pattern changed by [`mark`]`(k)`, and reads each byte before it writes
    /// it, which must still hold what that walk's sweep before left there (the pattern before the
    /// first). `started` once the first byte is written. What it wrote, and the first byte it
    /// found otherwise, where it found one (then it stops there); but where `free`, a byte of the
    /// shape's rows' padding it found otherwise is noted apart, the first of them, and written on
    /// as any other.
    ///
    /// A device that writes back bytes it read beside the texels (a whole line of a cache, say, of
    /// which the texels hold part) leaves there what they held when it read them. Where the writer
    /// wrote one of them between the device's read and its write, the writer's next read of it,
    /// or the round's compare, finds the older value: the writer rewrites the bytes of such a
    /// line again and again, so any span between the read and the write longer than the
    /// writer's walk round those bytes holds one of its writes, wherever in the step the span
    /// lies. A writer that wrote each byte once, as the step began, missed every such span past
    /// its first writes: the Iris Xe's clear and store of a 100x50 image at a 512-byte pitch put
    /// back the pattern over its writes to the bytes after a row's last texels, in their line, in
    /// 77 of 100 such rounds, and its check passed so under validation layers. A writer that came
    /// back to them only once a sweep of the window missed most spans of the Iris Xe's storage
    /// writes, which are short: a round of a 61x37 `R8Uint` image at a 256-byte pitch (3 bytes
    /// beside each row's texels in their line, of a window of 20 KiB) caught the line's write-back
    /// in 2 of 100 rounds, a 101x51 `RGBA8Uint` one at 512 (44 such bytes a row) in 45 of 100, so
    /// that three rounds of each shape answered the access passed in one class in two, and in all
    /// six at once under the load of the runtime's suite beside it.
    fn write_until(
        &self,
        pattern: &Pattern,
        (texels, free, line): (Option<Shape>, bool, u64),
        (stop, started): (&AtomicBool, &Barrier),
    ) -> Written {
        let mut walk = Walk::new(self, texels, line);
        let mut first = true;
        let mut started_once = || {
            if first {
                first = false;
                started.wait();
            }
        };
        let in_padding = |offset| free && texels.is_some_and(|shape| shape.in_padding(offset));
        let mut padding = None;
        loop {
            let (offset, sweep) = walk.step();
            let byte = self.byte(offset);
            let due = pattern.at(offset) ^ mark(sweep - 1);
            // SAFETY: as in `fill`; the round's thread does not touch the bytes meanwhile.
            let read = unsafe { byte.read_volatile() };
            if read != due {
                if !in_padding(offset) {
                    started_once();
                    return walk.written(Some((offset, due, read)), padding);
                }
                padding.get_or_insert((offset, due, read));
            }
            // SAFETY: as above.
            unsafe { byte.write_volatile(pattern.at(offset) ^ mark(sweep)) };
            started_once();
            if stop.load(Ordering::Acquire) {
                return walk.written(None, padding);
            }
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
        // Each byte the writer wrote must read its walk's last sweep over it, and is set back to
        // the pattern as it is read.
        let mut due_and_back = |offset: u64, last: u64| {
            let due = pattern.at(offset) ^ mark(last);
            // SAFETY: as in `fill`; the writer was joined.
            let read = unsafe { self.byte(offset).read_volatile() };
            if read != due {
                if !in_padding(offset) {
                    return Err((offset, due, read));
                }
                padding.get_or_insert((offset, due, read));
            }
            // SAFETY: as above.
            unsafe { self.byte(offset).write_volatile(pattern.at(offset)) };
            Ok(())
        };
        let stride = self.stride();
        let is_near = bits_of(&written.near, self.length);
        let reached = match written.sweep {
            1 => written.reached,
            _ => self.length,
        };
        for n in 0..reached {
            let offset = self.nth(n, stride);
            if texels.is_some_and(|(shape, _)| shape.is_texel(offset)) || holds(&is_near, offset) {
                continue;
            }
            let last = match n < written.reached {
                true => written.sweep,
                false => written.sweep - 1,
            };
            if let Err(changed) = due_and_back(offset, last) {
                return Round::Changed(changed);
            }
        }
        for (at, &offset) in (0u64..).zip(&written.near) {
            let last = match at < written.near_reached {
                true => written.near_sweep,
                false => written.near_sweep - 1,
            };
            if let Err(changed) = due_and_back(offset, last) {
                return Round::Changed(changed);
            }
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
                    // Compared in wide loads first, as in `matches`.
                    let zero =
                        (read.chunks(ZEROS.len())).all(|chunk| chunk == &ZEROS[..chunk.len()]);
                    let at = (!zero)
                        .then(|| read.iter().position(|&read| read != 0))
                        .flatten();
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
            // Compared whole first: slice equality reads in wide loads, where a read of each byte
            // of pages the CPU maps uncached is a bus read of its own (the dma-buf class's on the
            // Iris Xe: 2.7 s for a 3840x2160 surface's window byte by byte).
            if read != due {
                if let Some(at) = read.iter().zip(due).position(|(read, due)| read != due) {
                    return Some((offset + at as u64, due[at], read[at]));
                }
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
