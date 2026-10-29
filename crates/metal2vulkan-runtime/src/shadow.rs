//! Shadows: a 2D texture or array over a buffer that no linear image of the device aliases
//! (`linear_plan::aliases`) is an optimal image whose texels the encoders copy from the buffer and
//! back, so that every access Metal orders sees what the other view wrote. A texture buffer whose
//! offset in its buffer's `VkBuffer` misses the device's `minTexelBufferOffsetAlignment`, where
//! Vulkan views no texels, is likewise a staging buffer its bindings view, the texels at its start.
//! The rules below are the same for both shadows, an image or a staging buffer.
//!
//! - **The buffer is the truth.** CPU access through the texture reads and writes the buffer's
//!   bytes (`buffer_texture.rs`), and between encoders the image holds nothing the buffer lacks.
//! - **Copy in** before the first command of a command buffer's encoders that uses the texture,
//!   and again before a later one when a command wrote the bytes since: through the buffer, a
//!   texture buffer, or another shadowed texture over them. A command buffer's encoders run after
//!   its commit, so the first copy sees every CPU write made before then. Metal shows an encoder
//!   the bytes as of its command buffer's commit: a CPU write of a `Shared` buffer's bytes made
//!   while the command buffer ran reached neither a later dispatch of the running encoder nor a
//!   later encoder of the command buffer, while the same write made before the commit reached
//!   both, a command buffer committed before the write read the written bytes, and so did the
//!   encoder after an event wait that another queue's write signalled (an M3 Max under macOS
//!   27.0.1, each answer every time). So an encoder takes the image its command buffer's earlier
//!   encoder left holding the truth and copies nothing in ([`ShadowCarry`]), while a later
//!   command buffer copies in the bytes Metal would read there. A copy in
//!   of a one-level, one-slice texture's image moves only the texels inside the rectangle its
//!   command reads and writes them in, where the command bounds that ([`ShadowUse::loads`]: a
//!   pass's attachment it loads and takes no other way, inside its draws' scissors). The image's
//!   other texels keep whatever it held, which no copy back moves and no command of the encoder
//!   reads: a later command that needs them current takes a copy back of what was written and a
//!   copy in of every texel first, in this encoder or a later one of the command buffer.
//! - **Copy in what changed.** An image whose bytes changed since its copy in only inside a
//!   rectangle of its texels, by writes [`Within`] it (a render pass instance's to a color
//!   attachment it loads, inside its draws' scissors), or by a copy back of a part of another
//!   shadow laid over the bytes texel for texel as it is ([`ShadowCopies::same_texels`]), takes
//!   a copy in of the texels inside the rectangle bounding those alone
//!   (`WorkCounts::bounded_shadow_copies`): every other texel holds the bytes still. A command
//!   that reads only inside a rectangle and writes none takes those of them inside it, and the
//!   rest stay to copy in before the next command that reads them. Any other write of the bytes
//!   leaves it to copy in as above; it holds no writes of its own, nor a keep.
//! - **Copy back** what a command wrote through the texture before the next command that touches
//!   the bytes some other way, and at the end of the encoder (orchestrator, 2026-10-04: at the
//!   end of the encoder that writes through the texture, not of the command buffer). A copy moves
//!   only the texels' rows, never the bytes between them; a copy back of a one-level, one-slice
//!   texture's image, only the texels inside the rectangle its writers since the last bound their
//!   writes to, where each bounds them ([`ShadowUse::area`]: a pass's attachment it loads, inside
//!   its draws' scissors), as the image's other texels are what the last copy in left. Where the
//!   device runs it faster so, a copy back of such a texture's image is a compute dispatch that
//!   stores the same bytes (`shadow_copy_back.rs`).
//! - **Merged back** ([`ShadowCopies::merges_back`]: a buffer's bytes another writer may write
//!   while a command buffer runs, where a merge serves the shadow). A copy back would write every
//!   texel it moves, those no command wrote too, with what the copy in read, over another writer's
//!   writes to them while the command buffer ran, which Metal keeps: it writes only the texels a
//!   command writes, and another writer writing bytes no command touches while a command buffer
//!   runs is no race under its contract. The CPU writes a `Shared` or `Managed` buffer's bytes and
//!   host pages so (an M3 Max under macOS 27.0.1 kept such a write, made inside a kernel's run,
//!   every time, a `Managed` buffer's too), and another queue's commands an untracked buffer's,
//!   which the program orders (the M3 Max ran another queue's fill of an untracked `Private`
//!   buffer's other texels inside a kernel's or a render pass's run and kept it). A tracked
//!   `Private` buffer's bytes have no such writer: the CPU writes none, and the tracker orders
//!   another queue's commands that touch them before or after the encoder's command buffer, as
//!   Metal does (the M3 Max ran another queue's fill of a tracked buffer after the kernel or the
//!   pass ended, every time), so its texture's shadow is copied back. A merged-back image is kept,
//!   after its copy in, before the first command since the last copy back that writes through it,
//!   and each copy back of it is a merge (`shadow_merge.rs`): only the words the encoder's
//!   commands changed through the texture since the keep reach the bytes. The keep holds the
//!   rectangle the commands write in where they bound it ([`ShadowUse::area`]); a command that
//!   writes outside it takes the merge first, and a keep of the rectangle bounding both. The image
//!   still holds the bytes as the encoder's commands left them. A command that replaces every
//!   texel ([`ShadowUse::overwrites`]) keeps nothing, as its copy back writes only texels it
//!   wrote; that copy back goes before the keep of a later command that writes through the
//!   texture, since the merge after the keep takes only the words changed since it. The keeps'
//!   scratch is the encoder's, one per texture and rectangle.
//! - **Overwrites over logged pages** ([`ShadowCopies::logged`]: host pages whose consumer logs
//!   every write to them, `dirty_pages.rs`). A command that replaces every texel of such a shadow
//!   takes the merge instead of the plain copy back: its copy in (which moves only the rows the
//!   log reports changed since the image's last copy, and every row where the log cannot say),
//!   the keep, the command, and the merge, which stores only the words the command changed
//!   against the keep. The bytes are Metal's: the copy in is a gated one, decided once the log
//!   published the stamp the command buffer asked at its commit (`DirtyPages::wait`), so the image
//!   and then the keep hold the host bytes as of the commit, every write before it (the CPU's
//!   among them) included, and a word the merge leaves already holds the value the command wrote
//!   there. A byte another writer writes after the commit is one the command also writes, a race
//!   Metal leaves unordered. Over host pages with no log the image may lag the CPU's writes, and a
//!   word the command writes with the image's stale value would keep the CPU's: there the copy
//!   back stays a plain copy of every texel the command wrote.
//!   Where the merge is cut to the texels' rows (its words are texels' bytes alone, none of the
//!   bytes between rows or past the last texel), it stores whole 256-byte blocks
//!   (`bind::ShadowMergeStores::Blocks`, `WorkCounts::shadow_merge_dispatches_in_blocks`): each
//!   block holding a changed word as one store without gaps, a block with none not at all. That
//!   too is Metal's: the command wrote every byte of every such block, as the plain copy back
//!   wrote them all, so a word it left as the keep held it is stored as the merge reads it, the
//!   same value but for another writer's write since the commit, which races the command's own.
//!   The scratch of such a merge is kept again only for another command whose merge stores whole
//!   blocks; one whose merge does not takes a merge of the words alone, after the blocks' merge.
//! - **Stored attachments** ([`ShadowUse::tile_stores`]). A render pass that loads a color
//!   attachment and stores it, and changes it only inside its draws' scissors
//!   (`render/shadows.rs`: the rectangle [`ShadowUse::area`], inside the render area), merges
//!   back in whole 256-byte blocks of the rectangle's rows too, over any merged-back bytes, logged
//!   or not: each block holding a changed word as one store, a block with none not at all, and no
//!   word outside the rectangle's rows (each widened to whole words). Metal's own store of such a
//!   pass writes back every texel of each tile its draws reach, those no fragment changed among
//!   them: on an M3 Max under macOS 27.0.1, a pass loading and storing a linear `RGBA8Unorm`
//!   texture over a no-copy buffer and drawing one pixel rewrote the other 1023 texels of the
//!   32x32 tile holding the pixel with the values it loaded, so a CPU thread writing every texel
//!   over and over through the passes lost its writes to each of them in 157 to 229 of 300 passes,
//!   and to no texel of any other tile, inside the render area or past it; a kernel writing the
//!   same pixel lost none of them. So a CPU write to a texel such a pass leaves inside its render
//!   area, made while the pass runs, is no write Metal keeps, and a block stored with the value
//!   the merge reads there is as Metal's answer. A block can reach past the tiles the draws reach
//!   (a 256-byte block of a row holds 64 such texels, two of those tiles), never past the
//!   rectangle; and the merge undoes only a write made between its read of the block and its
//!   store, a shorter window than a tile's from its load to its store. A use joined with another
//!   that writes through the texture otherwise (a draw's write to it as a bound texture) stores
//!   the words alone.
//! - **Ordering.** A copy is a command of the encoder with uses of both the buffer's bytes and the
//!   image, so the tracker orders it against the encoder's other commands and other encoders'
//!   like any use: the texture and its buffer are one resource to it. In an encoder whose
//!   commands Metal does not order (concurrent dispatches), each copy has a barrier before and
//!   after it.
//! - **Staging.** Vulkan copies texels at multiples of their size of a `VkBuffer`, which a
//!   no-copy buffer's bytes need not start at (`Buffer::base`). Such a 2D texture's copies go
//!   through a staging buffer of its own ([`ShadowStaging`]): its bytes into the staging buffer and
//!   on into the image, or the image into the staging buffer and its rows on into the buffer, as
//!   one command with a barrier between the steps. That staging buffer needs no uses of its own:
//!   every copy through it writes the image or the texture's bytes, so the tracker orders it after
//!   every other copy of the texture. A texture buffer's staging buffer is its shadow, with uses of
//!   its own, as an image's.
//! - **Rows that share bytes** (`linear_plan.rs`'s **Rows that overlap**). An IOSurface plane
//!   whose GPU rows share 1 to 15 bytes is an image of its texels whose copies are two regions a
//!   slice each way ([`plane_copy_regions`](super::buffer_texture::plane_copy_regions)): going
//!   in each shared byte is read for both the rows that hold it, and going back it is
//!   written once, from the earlier row, one of the rows Metal's own writes leave it from (its
//!   order follows the command, the GPU pitch and the row count). The rules above are the same;
//!   no copy of such a shadow is cut to a rectangle, and none dispatches, both of which take one
//!   region.
//! - **Render passes** (`render/shadows.rs`). A render pass instance is one command: its draws
//!   record apart, where no copy goes, so the copies go before the instance and the uses are its
//!   draws' and its attachments'.
//! - **Heap depth and stencil textures** (`texture/heap.rs`). On a driver that lays depth and
//!   stencil images apart from their memory, a heap's texture of such a format is an image of its
//!   own memory whose truth is the heap's bytes in the texture's range: each aspect's texels
//!   tightly packed, level by level, in planes (`heap_planes`). Its copies are one region per
//!   aspect and level, the region's aspect its own. The rules above are the same, and a blit
//!   encoder follows them too (`blit.rs`): such a texture is an image to its blits, where a texture
//!   over a buffer is its bytes.
//! - **Views of another component layout** (`texture/view_shadow.rs`). On a device whose Vulkan
//!   views reinterpret no image's components, such a view of an image is an image of the view's
//!   format whose truth is the viewed image's texels in the levels and layers it presents
//!   ([`ShadowSource::Image`]). Its copies are image to image (`vkCmdCopyImage` between formats of
//!   one copy unit, which keeps every bit), or through a buffer of its own where Vulkan copies no
//!   texels between the two formats (an alpha format and another, `A8_UNORM` and `R8_UNORM`:
//!   `image_plan::copies_between`), and the rules above are the same with the image in the
//!   buffer's place, except that a command writing the image two ways (below) is refused: no merge
//!   serves an image. A shadow of one level and slice copied image to image (a 2D texture's own
//!   image's, `texture/usage_shadow.rs`, among them) has its copies in cut to rectangles as a
//!   one-level, one-slice texture's over a buffer has, its texels the image's at their own
//!   coordinates ([`ShadowCopies::cuts`]); its copies back move every texel, so a command that
//!   writes through it takes every texel in first.
//! - **Rotated textures** (`rotated.rs`, `texture/rotation.rs`). A rotated texture renders through
//!   an image of its pass's frame whose truth is the buffer's bytes in the frame Metal stores the
//!   texture in, turned: its copies go through a staging buffer of the bytes as laid out and a
//!   tight one of the image's texels, the rotation kernel moving them between the two
//!   ([`ShadowTarget::Rotated`]). The rules above are the same, but for a command writing the bytes
//!   two ways, which is refused: no merge serves a turned image. Its functions read and write an
//!   image of the stored frame, unturned, whose writes Metal drops
//!   ([`ShadowCopies::drops_writes`]): copied in as any shadow, never copied back, and copied in
//!   again after a command that wrote through it.
//! - **Two ways** (`shadow_merge.rs`): one command that writes through a shadowed texture and also
//!   writes its bytes another way (through the buffer, a texture buffer, or another shadowed
//!   texture over them). A copy back would replace the other write's bytes with the shadow's, where
//!   Metal keeps both writes when they touch different bytes, so the shadow is kept before the
//!   command and merged after it: only the bytes the command changed through the texture reach the
//!   buffer. The merge goes before the encoder's next command, whatever it touches, and leaves the
//!   shadow to copy in again at its next use.
//! - **Converted texels** (`ycbcr.rs`): a two-plane YCbCr texture over a surface's two planes is an
//!   image of the texels Metal reads from them ([`ShadowSource::Converted`]). Its copy in is a
//!   conversion: the planes' bytes into a staging buffer of its own and a dispatch that writes
//!   every texel, which runs no workgroup where the planes' words are those the last conversion
//!   read (`ycbcr.rs`, **When it converts**). The rules above are the same with the conversion in
//!   the copy's place; nothing
//!   writes the image (Metal drops a function's writes, and a pass into it is refused), so nothing
//!   is copied back.
//! - **YCbCr writes** (`ycbcr.rs`): a 4:4:4 two-plane YCbCr texture whose writes Metal stores into
//!   its planes takes a second shadow at a storage binding, an image of the written texels and a
//!   mark of each ([`ShadowSource::YCbCrWrite`]). Its copy in clears it, so no texel is marked; its
//!   copy back stores the marked texels' codes into the planes, through a staging buffer of their
//!   bytes and a dispatch of the encode kernel, and leaves every other byte as it was. The rules
//!   above are the same: a copy back leaves the texture's converted texels stale, so they convert
//!   again at their next use, and a command writing the planes two ways is refused, as no merge
//!   serves the planes. A pass's color attachment of such a texture renders into its render shadow,
//!   a texture whose shadow is one of these over floats: its copy in converts the planes into every
//!   texel, as the texture reads them (the pass's load), and its copy back stores every texel's
//!   codes, the marks not read, of the tiles of 32 texels a side that meet its area where it has
//!   one ([`ShadowCopy::area`]: the pass loads the attachment and its draws' scissors bound what
//!   it changes).
//! - **Two planes in a heap** (`texture/heap_two_planes.rs`): a heap's two-plane extended-range
//!   texture is an image of its texels whose truth is its first level's planes in the heap's
//!   bytes, as Metal lays them ([`ShadowTarget::Planes`]): its copies go through a staging buffer
//!   of the bytes as laid out and a tight one of the image's texels, the two-plane kernel moving
//!   them between the two. The rules above are the same, but for a command writing the bytes two
//!   ways, which is refused: no merge serves the planes.
//! - **Tiles in a heap** (`texture/heap_tiled.rs`): a texture of a plain format in a heap over host
//!   pages is an image of its elements whose truth is the heap's bytes, the elements tiled there as
//!   Metal lays them ([`ShadowTarget::Tiled`]): its copies go through a staging buffer of the bytes
//!   as laid out and a tight one of the image's elements, the tiled texel kernel moving them
//!   between the two. The rules above are the same; a command writing the bytes two ways merges as
//!   a buffer's texture does, the shadow's elements copied out tiled, each other byte as the last
//!   copy in left it.

use super::buffer::{Buffer, CallerMemory, WeakBuffer};
use super::dirty_pages::{CopyInGate, ShadowLog};
use super::error::{Error, RuntimeGap};
use super::resource::ResourceKey;
use super::rotated::{RotatedCopies, Turn};
use super::row_copies::RowCopies;
use super::shadow_copy_back::{CopyBack, WordView};
use super::shadow_merge::{ScratchSpares, ShadowMerge, UnormView};
use super::texture::copy_images::UseMaker;
use super::texture::heap_tiled::{HeapTiledCopies, SampleMoves};
use super::texture::heap_two_planes::{HeapPlaneCopies, TwoPlaneHeapLayout};
use super::texture::Texture;
use super::tracker::{Access, Barrier, Use, UseRange};
use super::values::StorageMode;
use super::view_plan::reinterpretation_refused;
use super::ycbcr::{YCbCrConversion, YCbCrWriteBack};
use super::Device;
use crate::device::functions::Vk;
use ash::vk;
use metal2vulkan::error::{Feature, Unsupported};
use metal2vulkan::metal::PixelFormat;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// What a shadowed texture's copies need: its truth, its shadow, and the copies between them.
pub(super) struct ShadowCopies {
    /// What the shadow holds a copy of, and the copies.
    pub(super) source: ShadowSource,
    /// The shadow's identity: its image's, or its staging buffer's.
    pub(super) shadow_key: ResourceKey,
    /// Everything the shadow holds.
    pub(super) shadow_range: UseRange,
    /// The texture's hazard tracking, its truth's.
    pub(super) tracked: bool,
    /// Whether what a command writes through the shadow never reaches the truth: a rotated
    /// texture's image its functions read and write, whose writes Metal drops. Such a shadow is
    /// never copied back, and a command writing through it leaves it to copy in again at its next
    /// use.
    pub(super) drops_writes: bool,
    /// What holds the truth, as the CPU may write it.
    pub(super) backing: Backing,
    /// The scratch buffers its merges are done with, kept for its next ones (`shadow_merge.rs`).
    pub(super) merge_scratch: ScratchSpares,
    /// How many encoders ended having used the shadow, by which a command buffer tells whether
    /// the image is still the one its own earlier encoder left ([`ShadowCarry`]): every encoder
    /// that used it moves this on as it ends, so a carry another encoder's end came between is
    /// no longer taken.
    pub(super) touched: AtomicU64,
    /// The identity of the command buffer whose encoder last ended with the image holding the
    /// truth ([`ShadowCarry::id`]), 0 for none ever: by which a first copy in that no carry could
    /// serve is counted as one an earlier command buffer's standing stands behind, or as one of a
    /// shadow that never held a standing ([`CarryTally`]).
    pub(super) stood: AtomicU64,
}

/// The texels of a texture over an IOSurface's plane from which its shadow's copies in count
/// apart ([`ShadowCopies::large_surface`]): 2^20, which a 1920x1080 framebuffer's exceed.
pub(super) const LARGE_SURFACE_TEXELS: u64 = 1 << 20;

/// What holds a shadow's truth, as the CPU may write it, by which the encoders' first copies in
/// are counted (`WorkCounts::shadow_texels_first_copied_in`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Backing {
    /// An image (a texture's own, or the image a view presents): only the runtime's commands and
    /// calls write it.
    Image,
    /// A buffer of private or memoryless storage the runtime allocated: likewise.
    PrivateBuffer,
    /// A buffer of shared or managed storage the runtime allocated, which the CPU maps.
    MappedBuffer,
    /// Host pages a no-copy buffer imported, or the memory of the caller's `VkBuffer`.
    HostPages,
    /// Host pages that the texture's own image lies in too (an aliasing image, a pitched host
    /// image among them), a shadow of which reads them as the image left them.
    AliasedHostPages,
    /// The bytes of an IOSurface's plane.
    Surface,
}

impl Backing {
    /// What holds the bytes of `buffer`, a texture's over an IOSurface's plane where `surface`.
    pub(super) fn of(buffer: &Buffer, surface: bool) -> Backing {
        let private = matches!(
            buffer.resource_options().storage_mode(),
            Ok(StorageMode::Private | StorageMode::Memoryless)
        );
        match (surface, buffer.caller_memory(), private) {
            (true, _, _) => Backing::Surface,
            (false, Some(_), _) => Backing::HostPages,
            (false, None, true) => Backing::PrivateBuffer,
            (false, None, false) => Backing::MappedBuffer,
        }
    }

    /// The backing of a shadow of a texture whose own image lies in the bytes too.
    pub(super) fn aliased(self) -> Backing {
        match self {
            Backing::HostPages => Backing::AliasedHostPages,
            other => other,
        }
    }
}

/// A shadow's truth: what the shadow holds a copy of between an encoder's copies (the module's
/// rules, with the truth in the buffer's place).
pub(super) enum ShadowSource {
    /// A buffer's bytes: a 2D texture over a buffer, a staged texture buffer, a heap's depth or
    /// stencil texture.
    Bytes {
        buffer: ShadowBuffer,
        /// The buffer's identity.
        key: ResourceKey,
        /// The bytes the texture lies in, from its first texel to its last.
        bytes: Range<u64>,
        /// The shadow the texels are copied into and back from.
        target: ShadowTarget,
        /// The key of the host memory the buffer joined (`host_aliases.rs`), whose uses of the
        /// same bytes touch the truth too.
        alias: Option<ResourceKey>,
    },
    /// Another image's texels: a texture view of another component layout on a device whose
    /// Vulkan views reinterpret no image's components (`texture/view_shadow.rs`). The shadow is an
    /// image of the view's format and the image's shape; `regions` copy the levels and layers the
    /// view presents between the two, the same subresources, texel for texel (formats of one copy
    /// unit, whose copies keep every bit).
    Image {
        image: vk::Image,
        /// The image's identity.
        key: ResourceKey,
        /// The image's levels and layers the view presents.
        range: UseRange,
        shadow: vk::Image,
        /// From `image` to `shadow`; the copies back swap the two.
        regions: Vec<vk::ImageCopy>,
        /// The buffer the copies go through where Vulkan copies no texels between the two
        /// images' formats (an alpha format and another, `image_plan::copies_between`).
        through: Option<ImageThrough>,
    },
    /// A surface's two planes in a buffer's bytes, which the shadow holds the texels Metal reads
    /// of (`ycbcr.rs`): its copy in converts them, and it is never copied back.
    Converted {
        buffer: ShadowBuffer,
        /// The buffer's identity.
        key: ResourceKey,
        /// The bytes the planes lie in, from the first plane's first byte to the last's last.
        bytes: Range<u64>,
        /// As `Bytes`'s.
        alias: Option<ResourceKey>,
        /// The conversion's staging buffer, kernel and dispatch, and the planes it converts
        /// (boxed: the largest variant's fields by far).
        conversion: Box<YCbCrConversion>,
        /// The texture's format, which a refusal names.
        pixel_format: PixelFormat,
    },
    /// A surface's two planes in a buffer's bytes, into which the texels a function writes to a
    /// 4:4:4 two-plane YCbCr texture are stored (`ycbcr.rs`): the shadow is an image of the
    /// written texels' bits and a mark of each, its copy in a clear of every texel, unmarked,
    /// and its copy back the encoding of the marked texels' codes into the planes.
    YCbCrWrite {
        buffer: ShadowBuffer,
        /// The buffer's identity.
        key: ResourceKey,
        /// The bytes the planes lie in, from the first plane's first byte to the last's last.
        bytes: Range<u64>,
        /// As `Bytes`'s.
        alias: Option<ResourceKey>,
        /// The shadow's image, which the copy in clears.
        image: vk::Image,
        /// The copy back's staging buffer, kernel and dispatch (boxed, as `Converted`'s).
        write_back: Box<YCbCrWriteBack>,
        /// The texture's format, which a refusal names.
        pixel_format: PixelFormat,
    },
}

/// The `VkBuffer` a shadow's bytes lie in.
pub(super) enum ShadowBuffer {
    /// One for good: a buffer's the runtime made, or the caller's own `VkBuffer`; and where the
    /// buffer's bytes end in it.
    Fixed(vk::Buffer, u64),
    /// A buffer over host pages, which a rebind gives other pages (`rebind.rs`): its `VkBuffer`
    /// is read as each copy is recorded. A rebind keeps the bytes' offset in it (a heap's bytes
    /// start at its first byte, and a no-copy buffer with a texture over it keeps its offset in
    /// its page), so the copies' offsets hold. Held weakly: a copy is recorded only for a texture
    /// its encoder names, whose objects keep the buffer (its texels', or its heap's), so a heap's
    /// shadow image the runtime keeps after them keeps no no-copy heap's pages, range or token.
    Rebound(WeakBuffer),
}

impl ShadowBuffer {
    /// The `VkBuffer` `raw` of `buffer`'s bytes: read as each copy is recorded where a rebind
    /// may give the buffer other pages.
    pub(super) fn of(buffer: &Buffer, raw: vk::Buffer) -> ShadowBuffer {
        match buffer.caller_memory() {
            Some(CallerMemory::HostPages) => ShadowBuffer::Rebound(buffer.downgrade()),
            _ => ShadowBuffer::Fixed(raw, buffer.start() + buffer.length()),
        }
    }

    /// Where the buffer's bytes end in its `VkBuffer` (its byte 0's offset there,
    /// [`Buffer::start`], past its length), which a rebind keeps: a merge in place reads and
    /// writes whole words of them up to it (`shadow_merge.rs`). 0 once a rebound buffer went.
    pub(super) fn end(&self) -> u64 {
        match self {
            ShadowBuffer::Fixed(_, end) => *end,
            ShadowBuffer::Rebound(buffer) => buffer
                .upgrade()
                .map_or(0, |buffer| buffer.start() + buffer.length()),
        }
    }

    /// The `VkBuffer` the bytes lie in now: the null handle once a rebound buffer went (a copy's
    /// texture keeps its buffer), and while it is detached (`detach.rs`), when its `VkBuffer` is
    /// the device's stand-in, which holds none of its bytes. No copy records it: an encoder
    /// records nothing of a command that names detached bytes (`EncoderCore::skips`), and nothing
    /// planned binds it (a copy back's dispatch, `CopyBack::new`; a merge's window,
    /// `ShadowMerge::new`).
    pub(super) fn raw(&self) -> vk::Buffer {
        match self {
            ShadowBuffer::Fixed(raw, _) => *raw,
            ShadowBuffer::Rebound(buffer) => buffer
                .upgrade()
                .and_then(|buffer| buffer.attached_at(0))
                .map_or(vk::Buffer::null(), |(raw, _)| raw),
        }
    }
}

/// A shadowed texture's shadow (see the module).
pub(super) enum ShadowTarget {
    /// A 2D texture's image: `regions`, one per slice, copy its texels from the buffer, or from
    /// `staging` where Vulkan cannot copy them in place, and `back` copies them out of the image
    /// again.
    Image {
        image: vk::Image,
        regions: Vec<vk::BufferImageCopy>,
        /// The regions a copy back, and a merge's keep, take out of the image
        /// ([`ShadowCopies::record_out`]): `regions` for every shadow but one over an IOSurface
        /// plane whose rows share bytes, where going in reads each shared byte for both its rows
        /// and going back writes it once, from the earlier row, one of the rows Metal's own
        /// writes leave it from (`buffer_texture::plane_copy_regions`).
        back: Vec<vk::BufferImageCopy>,
        staging: Option<ShadowStaging>,
        /// The bytes of one of its texels in the buffer, by which a copy back of a part of the
        /// texture ([`ShadowCopy::area`]) offsets its region; `None` where no copy of it is
        /// bounded (a heap's depth or stencil texture, which is no color attachment).
        texel: Option<u64>,
        /// The image's unsigned-integer view, where the device copies back faster by a dispatch
        /// (`shadow_copy_back.rs`): through which its copies back dispatch and its merges read
        /// the texels the commands left (`shadow_merge.rs`); `None` where they copy.
        words: Option<WordView>,
        /// The image's view of its own 8-bit unsigned normalized format, where it has no
        /// unsigned-integer view: through which its merges read the texels the commands left
        /// (`shadow_merge.rs`); `None` where they copy.
        unorm: Option<UnormView>,
    },
    /// A texture buffer's staging buffer, which its bindings view.
    Buffer(ShadowStaging),
    /// A rotated texture's image of its pass's frame: `staging` holds the bytes as laid out, and
    /// `turn` moves them to and from the image, turned (`rotated.rs`).
    Rotated {
        image: vk::Image,
        staging: ShadowStaging,
        turn: Box<RotatedCopies>,
    },
    /// A heap's two-plane extended-range texture's image: `staging` holds its two planes as
    /// Metal lays them in the heap's bytes, and `planes` moves them to and from the image
    /// (`texture/heap_two_planes.rs`).
    Planes {
        image: vk::Image,
        staging: ShadowStaging,
        planes: Box<HeapPlaneCopies>,
        /// The texture's format, which a refusal names.
        pixel_format: PixelFormat,
    },
    /// A heap texture's image over host pages: `staging` holds its elements as Metal tiles them
    /// in the heap's bytes, and `tiles` moves them to and from the image
    /// (`texture/heap_tiled.rs`).
    Tiled {
        image: vk::Image,
        staging: ShadowStaging,
        tiles: Box<HeapTiledCopies>,
    },
}

impl ShadowTarget {
    /// The regions a copy of a 2D texture's image over a buffer's bytes takes, going into the
    /// image (`into_shadow`) or back out of it: `regions` in and `back` out, which differ only
    /// for an IOSurface plane whose rows share bytes
    /// ([`plane_copy_regions`](super::buffer_texture::plane_copy_regions): going in reads each
    /// shared byte for both its rows, and going back writes it once, from the earlier row, one of
    /// the rows Metal's own writes leave it from).
    /// `None` for every other shadow.
    pub(super) fn image_regions(&self, into_shadow: bool) -> Option<&[vk::BufferImageCopy]> {
        let ShadowTarget::Image { regions, back, .. } = self else {
            return None;
        };
        Some(match into_shadow {
            true => regions,
            false => back,
        })
    }
}

/// A shadowed texture's staging buffer (see the module): its bytes, from the first texel to the
/// last, laid out as in the texture's buffer from the staging buffer's start.
/// A runtime buffer the copies between a view's shadow and its image go through
/// ([`ShadowSource::Image`]), held for the shadow's life: each region of the image into it, then
/// out of it into the other image, the same subresources both ways.
pub(super) struct ImageThrough {
    pub(super) buffer: Buffer,
    pub(super) regions: Vec<vk::BufferImageCopy>,
}

pub(super) struct ShadowStaging {
    /// A runtime buffer, held for the texture's life.
    pub(super) buffer: Buffer,
    /// The texture's bytes from its buffer into the staging buffer.
    pub(super) into: vk::BufferCopy,
    /// Each row of texels from the staging buffer back into the texture's buffer: one region for
    /// rows that lie one after another.
    pub(super) back: Vec<vk::BufferCopy>,
}

impl ShadowStaging {
    /// The staging buffer's `VkBuffer`, which `into` and `back` address.
    pub(super) fn raw(&self) -> vk::Buffer {
        self.buffer.at(0).0
    }
}

impl ShadowCopies {
    /// Whether the shadow is a large IOSurface plane's, from which its copies in count apart
    /// (`WorkCounts::shadow_texels_first_copied_in_large_surfaces` and kin): a texture over an
    /// IOSurface's plane of at least [`LARGE_SURFACE_TEXELS`] texels.
    pub(super) fn large_surface(&self) -> bool {
        self.backing == Backing::Surface && self.texels() >= LARGE_SURFACE_TEXELS
    }

    /// Whether a copy in of the shadow's texels is cut to the rectangle of a part of them
    /// ([`ShadowCopy::area`]): a one-level, one-slice 2D texture's image over a buffer's bytes,
    /// one region copied in place (no staging buffer, [`part_of`]), whose copies back are cut
    /// too; or a shadow of one level and slice of another image, copied image to image
    /// ([`image_part_of`]), whose copies back move every texel ([`cuts_back`](Self::cuts_back)).
    pub(super) fn cuts(&self) -> bool {
        match &self.source {
            ShadowSource::Bytes {
                target:
                    ShadowTarget::Image {
                        regions,
                        staging: None,
                        texel: Some(_),
                        ..
                    },
                ..
            } => regions.len() == 1,
            ShadowSource::Image {
                regions,
                through: None,
                ..
            } => one_image_copy(regions).is_some(),
            _ => false,
        }
    }

    /// Whether a copy back of the shadow's texels is cut to a rectangle as its copy in is: a
    /// buffer's bytes' shadow that [`cuts`](Self::cuts). An image's shadow's copy back moves
    /// every texel, so a command that writes through it takes every texel in first.
    fn cuts_back(&self) -> bool {
        self.cuts() && self.target().is_some()
    }

    /// Whether a command that writes every texel of the boxes `covered` writes every texel the
    /// shadow holds: a one-level, one-slice 2D texture's image over a buffer's bytes, laid out
    /// in rows or in Metal's tiles of a heap's bytes (`heap_tiled.rs`), its one region inside one
    /// of the boxes, of every aspect it holds. Such a shadow's copy back moves its texels alone,
    /// never the bytes between them (the module's rules; a tiled one's staging starts as the
    /// heap's bytes), so the copy back of what the command wrote moves only texels it wrote.
    /// `false` for every other shadow.
    pub(super) fn covered_by(&self, covered: &[Covered]) -> bool {
        let (image, regions) = match &self.source {
            ShadowSource::Bytes {
                target: ShadowTarget::Image { image, regions, .. },
                ..
            } => (image, regions.as_slice()),
            ShadowSource::Bytes {
                target: ShadowTarget::Tiled { image, tiles, .. },
                ..
            } => (image, tiles.regions()),
            _ => return false,
        };
        let ([region], UseRange::Image { levels, layers, .. }) = (regions, &self.shadow_range)
        else {
            return false;
        };
        let held = region.image_subresource;
        let one = levels.len() == 1 && layers.len() == 1 && held.layer_count == 1;
        one && covered.iter().any(|written| {
            let at = written.subresource;
            written.image == *image
                && at.aspect_mask.contains(held.aspect_mask)
                && at.mip_level == held.mip_level
                && at.base_array_layer <= held.base_array_layer
                && u64::from(held.base_array_layer)
                    < u64::from(at.base_array_layer) + u64::from(at.layer_count)
                && within(
                    (region.image_offset, region.image_extent),
                    (written.offset, written.extent),
                )
        })
    }

    /// The texels the shadow holds: its every region's, every slice of each.
    pub(super) fn texels(&self) -> u64 {
        match &self.source {
            ShadowSource::Image { regions, .. } => regions.iter().map(image_copy_texels).sum(),
            ShadowSource::Bytes {
                target: ShadowTarget::Image { regions, .. },
                ..
            } => regions.iter().map(buffer_copy_texels).sum(),
            ShadowSource::Bytes { .. }
            | ShadowSource::Converted { .. }
            | ShadowSource::YCbCrWrite { .. } => 0,
        }
    }

    /// Whether `other`'s texels lie in the truth as this shadow's do, texel for texel: both cut
    /// to rectangles ([`cuts`](Self::cuts)), of one buffer, their one regions starting at one
    /// byte, rows as long, of one extent, of texels of one size; or of one image, their one
    /// regions of one aspect, level and slice, at one offset, of one extent. A change of one's
    /// texels inside a rectangle is then a change of the other's inside it alone.
    pub(super) fn same_texels(&self, other: &ShadowCopies) -> bool {
        let of_image = |copies: &ShadowCopies| match &copies.source {
            ShadowSource::Image {
                key,
                regions,
                through: None,
                ..
            } => one_image_copy(regions).map(|region| (*key, *region)),
            _ => None,
        };
        if let (Some((key, a)), Some((other_key, b))) = (of_image(self), of_image(other)) {
            let (s, t) = (a.src_subresource, b.src_subresource);
            return key == other_key
                && s.aspect_mask == t.aspect_mask
                && s.mip_level == t.mip_level
                && s.base_array_layer == t.base_array_layer
                && a.src_offset == b.src_offset
                && a.extent == b.extent;
        }
        let layout = |copies: &ShadowCopies| match &copies.source {
            ShadowSource::Bytes {
                key,
                target:
                    ShadowTarget::Image {
                        regions,
                        staging: None,
                        texel: Some(texel),
                        ..
                    },
                ..
            } => match regions.as_slice() {
                [region] => Some((*key, *region, *texel)),
                _ => None,
            },
            _ => None,
        };
        let (Some((key, a, texel)), Some((other_key, b, other_texel))) =
            (layout(self), layout(other))
        else {
            return false;
        };
        let row = |r: &vk::BufferImageCopy| match r.buffer_row_length {
            0 => r.image_extent.width,
            row => row,
        };
        key == other_key
            && texel == other_texel
            && a.buffer_offset == b.buffer_offset
            && row(&a) == row(&b)
            && a.image_offset == b.image_offset
            && a.image_extent == b.image_extent
    }

    /// Whether the shadow is a heap's one-level 2D two-plane texture's, whose truth is its two
    /// planes in the heap's bytes ([`ShadowTarget::Planes`]) where Metal's blits address them
    /// (`HeapPlaneCopies::layout`, `TwoPlaneHeapLayout::blit_rows`).
    pub(super) fn holds_blit_rows(&self) -> bool {
        self.plane_layout()
            .is_some_and(TwoPlaneHeapLayout::blit_rows)
    }

    /// Where a heap's two-plane texture's first level lies in the heap's bytes, where the shadow
    /// is such a texture's ([`ShadowTarget::Planes`]).
    pub(super) fn plane_layout(&self) -> Option<&TwoPlaneHeapLayout> {
        match self.target() {
            Some(ShadowTarget::Planes { planes, .. }) => Some(planes.layout()),
            _ => None,
        }
    }

    /// The truth's identity and the range of it the shadow copies.
    pub(super) fn truth(&self) -> (ResourceKey, UseRange) {
        match &self.source {
            ShadowSource::Bytes { key, bytes, .. }
            | ShadowSource::Converted { key, bytes, .. }
            | ShadowSource::YCbCrWrite { key, bytes, .. } => (*key, UseRange::Bytes(bytes.clone())),
            ShadowSource::Image { key, range, .. } => (*key, range.clone()),
        }
    }

    /// Whether its copy in converts its truth's bytes (`ShadowSource::Converted`,
    /// `ShadowSource::YCbCrWrite`) rather than moving them: zero bytes there are not zero
    /// bytes in the shadow.
    pub(super) fn converts(&self) -> bool {
        matches!(
            self.source,
            ShadowSource::Converted { .. } | ShadowSource::YCbCrWrite { .. }
        )
    }

    /// The shadow of a buffer's bytes, or `None` for an image's and for converted texels'.
    pub(super) fn target(&self) -> Option<&ShadowTarget> {
        match &self.source {
            ShadowSource::Bytes { target, .. } => Some(target),
            ShadowSource::Image { .. }
            | ShadowSource::Converted { .. }
            | ShadowSource::YCbCrWrite { .. } => None,
        }
    }

    /// Whether a render pass that reaches no tile and loads this shadow's texture leaves it to
    /// its truth, which keeps every texel: an image of a buffer's bytes (`ShadowTarget::Image`),
    /// which the texture's own image holds, and a two-plane YCbCr texture's render shadow,
    /// whose copy back would store the planes' codes again (`ycbcr.rs`), where Metal keeps them.
    pub(super) fn left_untouched(&self) -> bool {
        match &self.source {
            ShadowSource::Bytes { target, .. } => {
                matches!(target, ShadowTarget::Image { .. }) && !self.drops_writes
            }
            ShadowSource::YCbCrWrite { write_back, .. } => write_back.loads(),
            ShadowSource::Image { .. } | ShadowSource::Converted { .. } => false,
        }
    }

    /// Whether a render pass that draws loads this shadow other than by the inverse of its
    /// store: a rotated texture's frame of rotation 4 (`rotated.rs`).
    pub(super) fn loads_apart(&self) -> bool {
        matches!(self.target(), Some(ShadowTarget::Rotated { turn, .. }) if turn.loads_apart())
    }

    /// Whether the texture is tracked and its truth lies in host pages a consumer logs the writes
    /// to (`HostMemory::with_dirty_pages`): its copies in of every texel are gated
    /// (`dirty_pages.rs`), so none is cut to a rectangle.
    pub(super) fn logged(&self) -> bool {
        let ShadowSource::Bytes {
            buffer: ShadowBuffer::Rebound(buffer),
            ..
        } = &self.source
        else {
            return false;
        };
        let pages = || buffer.upgrade().and_then(|buffer| buffer.host_pages());
        self.tracked && pages().is_some_and(|pages| pages.dirty().is_some())
    }

    /// Whether the rectangle `area` (`None` every texel) covers every texel of a one-region 2D
    /// texture's image over a buffer's bytes, or is `None`.
    pub(super) fn covers(&self, area: Option<[u32; 4]>) -> bool {
        let Some([x0, y0, x1, y1]) = area else {
            return true;
        };
        let regions = (self.target()).and_then(|target| target.image_regions(true));
        let Some([region]) = regions else {
            return false;
        };
        let (x, y) = (region.image_offset.x, region.image_offset.y);
        let (Ok(x), Ok(y)) = (u32::try_from(x), u32::try_from(y)) else {
            return false;
        };
        let extent = region.image_extent;
        x0 <= x && y0 <= y && x1 >= x + extent.width && y1 >= y + extent.height
    }

    /// Whether `used` touches the texture's truth: its bytes, or its image's levels and layers.
    pub(super) fn meets(&self, used: &Use<ResourceKey>) -> bool {
        let (key, range) = self.truth();
        (used.key == key || Some(used.key) == self.alias()) && used.range.overlaps(&range)
    }

    /// Why a command writing the truth both through this shadow and another way is refused: no
    /// merge serves an image's shadow (`Unsupported`), nor a rotated texture's, whose texels lie
    /// turned in its bytes (`NotImplemented(RotatedTexture)`); `None` for a shadow a merge serves.
    fn unmerged(&self) -> Option<Error> {
        match &self.source {
            ShadowSource::Image { .. } => Some(Error::Unsupported(reinterpretation_refused())),
            ShadowSource::Bytes {
                target: ShadowTarget::Rotated { .. },
                ..
            } => Some(Error::NotImplemented(RuntimeGap::RotatedTexture)),
            // Its texels lie in two planes of its bytes, which no merge reaches.
            ShadowSource::Bytes {
                target: ShadowTarget::Planes { pixel_format, .. },
                ..
            } => Some(Error::Unsupported(
                Unsupported::not_implemented(Feature::Format).naming_format(*pixel_format),
            )),
            ShadowSource::Bytes { .. } => None,
            // Nothing writes converted texels' image (`ycbcr.rs`): no command reaches here.
            // Its writes are encoded into two planes, which no merge reaches.
            ShadowSource::Converted { pixel_format, .. }
            | ShadowSource::YCbCrWrite { pixel_format, .. } => Some(Error::Unsupported(
                Unsupported::not_implemented(Feature::Format).naming_format(*pixel_format),
            )),
        }
    }

    /// Whether what the commands write through the shadow goes back into its truth as a merge
    /// (`shadow_merge.rs`): a buffer's bytes that another writer may write beside the commands'
    /// writes while a command buffer runs, whose writes to texels the commands left Metal keeps;
    /// never copied back over those texels (the module's rules). That is every buffer's bytes but
    /// a tracked `Private` or `Memoryless` buffer's the runtime made: the CPU writes the others
    /// (a `Shared` or `Managed` buffer's, host pages, the caller's own `VkBuffer`'s memory), and
    /// another queue's commands an untracked buffer's, which the tracker leaves to the program. An
    /// image's shadow, a turned one and two planes take no merge ([`unmerged`](Self::unmerged)),
    /// and writes Metal drops go back no way.
    pub(super) fn merges_back(&self) -> bool {
        let bytes = matches!(&self.source, ShadowSource::Bytes { .. });
        let other_writer = self.backing != Backing::PrivateBuffer || !self.tracked;
        bytes && other_writer && !self.drops_writes && self.unmerged().is_none()
    }

    /// Whether two shadowed textures copy some of the same truth.
    fn shares_truth(&self, other: &ShadowCopies) -> bool {
        let ((key, range), (other_key, other_range)) = (self.truth(), other.truth());
        let mine = [Some(key), self.alias()];
        let theirs = [Some(other_key), other.alias()];
        mine.iter()
            .flatten()
            .any(|key| theirs.contains(&Some(*key)))
            && range.overlaps(&other_range)
    }

    /// The key of the host memory the truth's buffer joined (`host_aliases.rs`): a use of it
    /// touches the same bytes as one of the buffer's own.
    fn alias(&self) -> Option<ResourceKey> {
        match &self.source {
            ShadowSource::Bytes { alias, .. }
            | ShadowSource::Converted { alias, .. }
            | ShadowSource::YCbCrWrite { alias, .. } => *alias,
            ShadowSource::Image { .. } => None,
        }
    }

    /// Where the texture's first byte lies in its buffer; `None` for an image's shadow.
    pub(super) fn first_byte(&self) -> Option<u64> {
        match self.target()? {
            ShadowTarget::Image {
                staging: Some(staging),
                ..
            }
            | ShadowTarget::Buffer(staging)
            | ShadowTarget::Rotated { staging, .. }
            | ShadowTarget::Planes { staging, .. }
            | ShadowTarget::Tiled { staging, .. } => Some(staging.into.src_offset),
            // Slice 0's region starts at the first texel.
            ShadowTarget::Image { regions, .. } => {
                Some(regions.first().map_or(0, |region| region.buffer_offset))
            }
        }
    }

    /// Whether a merge's keep and changed copy ([`record_kept`](Self::record_kept),
    /// [`record_out`](Self::record_out)) each write every one of the first `length` bytes of its
    /// part: a heap's tiled texture's do, as they copy the staging buffer's span of the texture's
    /// bytes whole (`HeapTiledCopies::record_out`), or the heap's.
    pub(super) fn copies_out_whole(&self, length: u64) -> bool {
        match &self.source {
            ShadowSource::Bytes {
                target: ShadowTarget::Tiled { staging, .. },
                ..
            } => staging.into.size >= length,
            _ => false,
        }
    }

    /// Whether a merge's keep right after this shadow's copy in, with no copy of the shadow
    /// between, may copy the staging buffer that copy in filled
    /// ([`record_kept_from_staging`](Self::record_kept_from_staging)): a heap's tiled texture's
    /// whose moves give back each byte they took (`HeapTiledCopies::moves_bits`).
    pub(super) fn keeps_from_staging(&self) -> bool {
        matches!(
            &self.source,
            ShadowSource::Bytes {
                target: ShadowTarget::Tiled { tiles, .. },
                ..
            } if tiles.moves_bits()
        )
    }

    /// Record a merge's keep into `to` from `at` as a copy of the staging buffer, right after the
    /// shadow's copy in filled it with the heap's bytes and moved them into the image
    /// ([`keeps_from_staging`](Self::keeps_from_staging)): the bytes a keep's move out of the
    /// image would leave there ([`record_kept`](Self::record_kept)), the texture's bytes from its
    /// first, without the move. Nothing for any other shadow.
    ///
    /// # Safety
    ///
    /// As `record_out`'s, after a barrier that orders the copy after the copy in, and with no
    /// command between them that writes the shadow or its staging buffer.
    pub(super) unsafe fn record_kept_from_staging(
        &self,
        device: &Vk,
        command_buffer: vk::CommandBuffer,
        (to, at): (vk::Buffer, u64),
    ) {
        let ShadowSource::Bytes {
            target: ShadowTarget::Tiled { staging, .. },
            ..
        } = &self.source
        else {
            return;
        };
        let region = vk::BufferCopy {
            src_offset: staging.into.dst_offset,
            dst_offset: at,
            size: staging.into.size,
        };
        // SAFETY: the caller's contract; the staging buffer holds the texture's bytes from its
        // start (`HeapTiledCopies::staging`), and `to` holds them from `at`.
        unsafe { device.cmd_copy_buffer(command_buffer, staging.raw(), to, &[region]) };
    }

    /// Record a merge's keep of the shadow's texels into `to` from `at`, laid out as in the
    /// buffer from the texture's first byte (`shadow_merge.rs`): as
    /// [`record_out`](Self::record_out) copies them; but where a heap's tiled texture's image
    /// takes the bytes inexactly (`HeapTiledCopies::takes_inexactly`: a multisample depth the
    /// driver's fragment depth changes), the heap's bytes themselves, so the merge writes each
    /// byte the image holds otherwise than the heap: every byte a command wrote, and the values
    /// the copy in changed (the announced `drawn-depth-copy`), never a byte of the heap's in a
    /// value a command wrote.
    ///
    /// # Safety
    ///
    /// As `record_out`'s.
    pub(super) unsafe fn record_kept(
        &self,
        device: &Vk,
        command_buffer: vk::CommandBuffer,
        (to, at): (vk::Buffer, u64),
        moves: Option<&SampleMoves>,
    ) {
        if let ShadowSource::Bytes {
            buffer,
            target: ShadowTarget::Tiled { staging, tiles, .. },
            ..
        } = &self.source
        {
            if tiles.takes_inexactly() {
                let region = vk::BufferCopy {
                    src_offset: staging.into.src_offset,
                    dst_offset: at,
                    size: staging.into.size,
                };
                // SAFETY: the caller's contract; the texture's bytes lie in the heap's buffer from
                // the staging copy's first, and `to` holds them from `at`.
                unsafe { device.cmd_copy_buffer(command_buffer, buffer.raw(), to, &[region]) };
                return;
            }
        }
        // SAFETY: the caller's contract.
        unsafe { self.record_out(device, command_buffer, (to, at), moves) };
    }

    /// Record a copy of the shadow's texels into `to` from `at`, laid out as in the buffer from
    /// the texture's first byte (`shadow_merge.rs`), a heap's multisample texture's through
    /// `moves` out of its image (`ShadowCopy::plan_moves`).
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording, outside a render pass; the shadow, the moves and `to` are
    /// live until it completes, `to` holds the texture's bytes from `at`, a multiple of 16, and an
    /// image is in the `GENERAL` layout.
    pub(super) unsafe fn record_out(
        &self,
        device: &Vk,
        command_buffer: vk::CommandBuffer,
        (to, at): (vk::Buffer, u64),
        moves: Option<&SampleMoves>,
    ) {
        // An image's shadow keeps no bytes, a rotated texture's keeps them turned, and a heap's
        // two-plane texture's in two planes: no merge is made of any (`shadow_merge.rs`,
        // `ShadowCopies::unmerged`).
        let Some(target) = self.target() else {
            return;
        };
        match target {
            ShadowTarget::Rotated { .. } | ShadowTarget::Planes { .. } => {}
            ShadowTarget::Tiled {
                image,
                staging,
                tiles,
            } => {
                // SAFETY: the caller's contract, which the tiles' record takes.
                unsafe {
                    tiles.record_out(device, command_buffer, (*image, moves), staging, (to, at));
                }
            }
            ShadowTarget::Image { image, .. } => {
                let back = target.image_regions(false).unwrap_or_default();
                let origin = back.first().map_or(0, |region| region.buffer_offset);
                let moved: Vec<vk::BufferImageCopy> = back
                    .iter()
                    .map(|region| vk::BufferImageCopy {
                        buffer_offset: region.buffer_offset - origin + at,
                        ..*region
                    })
                    .collect();
                // SAFETY: the caller's contract; the regions keep their offsets from the first
                // texel, multiples of the texel's size, and `at` is one too.
                unsafe {
                    device.cmd_copy_image_to_buffer(
                        command_buffer,
                        *image,
                        vk::ImageLayout::GENERAL,
                        to,
                        &moved,
                    );
                }
            }
            ShadowTarget::Buffer(staging) => {
                let region = vk::BufferCopy {
                    src_offset: staging.into.dst_offset,
                    dst_offset: at,
                    size: staging.into.size,
                };
                // SAFETY: the caller's contract; the staging buffer holds the texture's bytes.
                unsafe { device.cmd_copy_buffer(command_buffer, staging.raw(), to, &[region]) };
            }
        }
    }
}

/// One step of a shadowed texture's copies (the module's rules), `M` the scratch of a merge
/// (`shadow_merge.rs`; another type in the unit tests).
#[derive(Clone)]
pub(super) enum Step<M> {
    /// From the buffer into the shadow.
    Into,
    /// From the shadow back into the buffer.
    Back,
    /// The shadow kept, before a command that writes its bytes two ways, or before the first
    /// command since the last copy back that writes through it where it merges back
    /// ([`ShadowCopies::merges_back`]).
    Keep(M),
    /// After the command that wrote two ways, or in place of the next copy back, the bytes
    /// changed through the texture since the keep merged into the buffer's.
    Merge(M),
}

/// A shadow's copy in over host pages a consumer logs (`dirty_pages.rs`), left by its encoder
/// to its batch's submission (`Deferred::GatedCopyIn`): the commit asks the log for a stamp, and
/// the copy's regions are those the log says changed once that stamp is published, which only a
/// recording after the commit can know. Recorded in a command buffer of its own between the
/// encoder's, behind a barrier from everything before it and before one to everything after it,
/// so it keeps its place among the encoder's commands.
pub(super) struct GatedCopyIn {
    copy: ShadowCopy,
    gate: Arc<CopyInGate>,
}

impl GatedCopyIn {
    /// `copy`, its regions decided by `gate`.
    pub(super) fn new(copy: ShadowCopy, gate: Arc<CopyInGate>) -> GatedCopyIn {
        GatedCopyIn { copy, gate }
    }

    /// Record it into `command_buffer`, its regions decided now.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording, outside a render pass; the copy's texture is live (its
    /// encoder's command buffer holds it), its image in the `GENERAL` layout.
    pub(super) unsafe fn record(&self, device: &Vk, command_buffer: vk::CommandBuffer) {
        // SAFETY: the caller's.
        unsafe {
            Barrier::everything().record(device, command_buffer);
            let gate = Some(Arc::clone(&self.gate));
            self.copy.record_gated(device, command_buffer, None, gate);
            Barrier::everything().record(device, command_buffer);
        }
    }
}

/// One copy between a shadowed texture's buffer and its shadow.
#[derive(Clone)]
pub(super) struct ShadowCopy<M = Arc<ShadowMerge>> {
    pub(super) copies: Arc<ShadowCopies>,
    pub(super) step: Step<M>,
    /// A copy in before a command that leaves the texels in place ([`ShadowUse::kept`]).
    pub(super) kept: bool,
    /// A copy back of the texels inside `[x0, y0, x1, y1)` of a one-level, one-slice texture's
    /// image alone, where the commands since the last copy back wrote through it there and
    /// nowhere else ([`ShadowUse::area`]), or a copy in of them alone, where the command reads
    /// and writes no other texel of it ([`ShadowUse::loads`]); `None` for every texel.
    pub(super) area: Option<[u32; 4]>,
    /// Why a copy in is made, as `WorkCounts` counts its texels; `None` for every other step.
    pub(super) cause: Option<CopyCause>,
    /// A heap's multisample texture's moves between its image and Metal's tiles that its record
    /// takes, planned as the encoder takes the copy (`EncoderCore::sample_moves`); `None` for
    /// every other texture, and where none could be planned (failing the command buffer).
    pub(super) moves: Option<Arc<SampleMoves>>,
}

/// Why a copy in is made, as `WorkCounts` counts its texels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CopyCause {
    /// The encoder's first copy in of the texture: before the first command of the encoder that
    /// uses it, or the first after work it records apart ([`ShadowSync::leave`]). Every later one
    /// follows a change of the truth, or a command that needs texels the image does not hold
    /// (`WorkCounts::shadow_texels_first_copied_in`).
    First,
    /// A later copy in before a command that needs texels outside the rectangle the image holds
    /// (its last copy in's, [`ShadowUse::loads`]), the truth unchanged there or not
    /// (`WorkCounts::shadow_texels_recopied_beyond_held`).
    BeyondHeld,
    /// The truth changed since the image's copy in, other than by writes within a rectangle that
    /// left it stale there alone, for the reason given
    /// (`WorkCounts::shadow_texels_recopied_after_changes`).
    Changed(Lost),
    /// A copy in of the texels writes [`Within`] a rectangle changed since the image's copy in,
    /// [`ShadowCopy::area`] the rectangle bounding them (the module's rules,
    /// `WorkCounts::shadow_texels_bounded_copied_in`).
    Bounded,
}

/// Why a shadow's image no longer holds the truth as the encoder's commands left it, so the next
/// command that uses the texture copies it in again ([`CopyCause::Changed`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Lost {
    /// Another shadowed texture over the same truth was copied back or merged.
    OtherShadow,
    /// Its own merge after a command that wrote the truth two ways.
    TwoWays,
    /// A command wrote the truth another way (its bytes, or an image's levels and layers).
    Written,
    /// A command wrote the truth within a rectangle ([`Within`]) where that leaves the image
    /// stale nowhere alone: the writes' texture has no shadow, or the image is written, kept, or
    /// its texels are laid out otherwise.
    Within,
    /// A command wrote through it where Metal drops the writes.
    Dropped,
}

impl<M> ShadowCopy<M> {
    /// The moves between a heap's multisample texture's image and Metal's tiles that the copy's
    /// record takes (`texture/heap_tiled.rs`), planned now through `maker`: into the image for a
    /// copy in, out of it for every other step (a copy back, a merge's keep and its copy of what
    /// changed). `None` for every other texture. Refused as their images are made.
    pub(super) fn plan_moves(&self, maker: &UseMaker) -> Result<Option<SampleMoves>, Error> {
        if !self.moves_samples() {
            return Ok(None);
        }
        let Some(ShadowTarget::Tiled { tiles, .. }) = self.copies.target() else {
            return Ok(None);
        };
        tiles.sample_moves(maker, matches!(self.step, Step::Into))
    }

    /// Whether the copy's record takes moves between a heap's multisample texture's image and
    /// Metal's tiles ([`plan_moves`](Self::plan_moves)).
    pub(super) fn moves_samples(&self) -> bool {
        let tiled = self.copies.target();
        matches!(tiled, Some(ShadowTarget::Tiled { tiles, .. }) if tiles.moves_samples())
    }

    /// Whether the copy writes the texture's truth: a copy back or a merge.
    pub(super) fn writes_truth(&self) -> bool {
        matches!(self.step, Step::Back | Step::Merge(_))
    }

    /// The texels a copy in moves, as `record` cuts it, of a 2D texture's image over a buffer's
    /// bytes (as [`texels_in`](Self::texels_in) counts them) or of an image's shadow (its every
    /// region's, or its one region's inside [`area`](Self::area)): 0 for every other copy.
    pub(super) fn texels_copied_in(&self) -> u64 {
        match (&self.step, &self.copies.source) {
            (Step::Into, ShadowSource::Image { regions, .. }) => {
                match (self.area, self.copies.cuts(), one_image_copy(regions)) {
                    (Some(area), true, Some(region)) => image_part_of(region, area)
                        .as_ref()
                        .map_or(0, image_copy_texels),
                    _ => regions.iter().map(image_copy_texels).sum(),
                }
            }
            (Step::Into, _) => self.texels_in(),
            _ => 0,
        }
    }

    /// The texels a copy back of a 2D texture's image over a buffer's bytes moves (its every
    /// region's, or its one region's inside [`area`](Self::area)): 0 for every other copy.
    pub(super) fn texels_back(&self) -> u64 {
        match self.step {
            Step::Back => self.image_texels(),
            _ => 0,
        }
    }

    /// The texels a copy in of a 2D texture's image over a buffer's bytes moves, as
    /// [`texels_back`](Self::texels_back) counts them: 0 for every other copy.
    pub(super) fn texels_in(&self) -> u64 {
        match self.step {
            Step::Into => self.image_texels(),
            _ => 0,
        }
    }

    /// The texels a copy in or back of a 2D texture's image over a buffer's bytes moves, as
    /// `record` cuts its region: 0 for any other shadow.
    fn image_texels(&self) -> u64 {
        let Some(ShadowTarget::Image {
            regions,
            staging,
            texel,
            ..
        }) = self.copies.target()
        else {
            return 0;
        };
        // As `record` cuts the region.
        match (self.area, staging, regions.as_slice(), texel) {
            (Some(area), None, [region], Some(texel)) => part_of(region, area, *texel)
                .as_ref()
                .map_or(0, buffer_copy_texels),
            _ => regions.iter().map(buffer_copy_texels).sum(),
        }
    }
}

/// The texels a copy of `region` moves between a buffer and an image.
fn buffer_copy_texels(region: &vk::BufferImageCopy) -> u64 {
    let e = region.image_extent;
    u64::from(e.width) * u64::from(e.height) * u64::from(e.depth)
}

/// The texels a copy of `region` moves between two images: every slice's.
fn image_copy_texels(region: &vk::ImageCopy) -> u64 {
    let e = region.extent;
    let layers = u64::from(region.src_subresource.layer_count);
    u64::from(e.width) * u64::from(e.height) * u64::from(e.depth) * layers
}

impl ShadowCopy {
    /// The log of a copy in of every texel of a tracked 2D texture's image straight from a
    /// buffer over host pages a consumer logs (`dirty_pages.rs`), made or kept by `device`, and
    /// the bytes of one texel: what may decide the copy's regions as its batch is submitted
    /// ([`GatedCopyIn`]).
    /// `None` for every other copy.
    pub(super) fn gate_log(&self, device: &Device) -> Option<(Arc<ShadowLog>, u64)> {
        let c = &self.copies;
        let first = matches!(self.cause, Some(CopyCause::First));
        if !matches!(self.step, Step::Into) || !c.covers(self.area) || !first || !c.tracked {
            return None;
        }
        let ShadowSource::Bytes {
            buffer: ShadowBuffer::Rebound(buffer),
            target:
                ShadowTarget::Image {
                    regions,
                    staging: None,
                    texel: Some(texel),
                    ..
                },
            ..
        } = &c.source
        else {
            return None;
        };
        let buffer = buffer.upgrade()?;
        let pages = buffer.host_pages()?;
        let source = pages.dirty()?;
        let contents = buffer.contents()?.as_ptr() as u64;
        let (raw, at) = buffer.attached_at(0)?;
        // The `VkBuffer`'s byte `v` lies at the host address `v + base`.
        let base = contents.wrapping_sub(at);
        let truth = copied_bytes(regions, *texel)?;
        let log = device.shared().dirty.log(c, source, raw, base, truth)?;
        Some((log, *texel))
    }

    /// The log of a copy back or merge of a shadow whose copies in a log decides, and the
    /// `VkBuffer` bytes it writes (`dirty_pages.rs`): reported to the log as the shadow's own.
    pub(super) fn written_log(&self, device: &Device) -> Option<(Arc<ShadowLog>, Range<u64>)> {
        let c = &self.copies;
        if !matches!(self.step, Step::Back | Step::Merge(_)) {
            return None;
        }
        let ShadowSource::Bytes {
            target: ShadowTarget::Image { regions, texel, .. },
            ..
        } = &c.source
        else {
            return None;
        };
        let log = device.shared().dirty.kept(c)?;
        let bytes = copied_bytes(regions, (*texel)?)?;
        Some((log, bytes))
    }

    /// The dispatch that takes this copy where it is a copy back that dispatches
    /// (`shadow_copy_back.rs`): a one-slice 2D texture's image over a buffer's bytes, copied in
    /// place, with its integer view, of texels the copy moves, made on `device` over the buffer's
    /// `VkBuffer` now. `None` where the copy takes it (any other copy, or one the dispatch does
    /// not fit).
    pub(super) fn dispatched(&self, device: &Device) -> Option<CopyBack> {
        let ShadowSource::Bytes {
            buffer,
            target:
                ShadowTarget::Image {
                    regions,
                    staging: None,
                    texel: Some(texel),
                    words: Some(words),
                    ..
                },
            ..
        } = &self.copies.source
        else {
            return None;
        };
        let ([region], Step::Back) = (regions.as_slice(), &self.step) else {
            return None;
        };
        let region = match self.area {
            Some(area) => part_of(region, area, *texel)?,
            None => *region,
        };
        CopyBack::new(device, *words, buffer.raw(), &region)
    }

    /// The texels a copy in of a one-region, one-slice 2D texture's image over a buffer's bytes
    /// copied in place moves, as `record` cuts its region (where they lie in the buffer, and in
    /// the image): what a merge's keep may take with it (`ShadowMerge::keep_in`). `None` for any
    /// other copy, or one that moves no texel.
    pub(super) fn copied_region(&self) -> Option<vk::BufferImageCopy> {
        let (
            Step::Into,
            ShadowSource::Bytes {
                target:
                    ShadowTarget::Image {
                        regions,
                        staging: None,
                        texel: Some(texel),
                        ..
                    },
                ..
            },
        ) = (&self.step, &self.copies.source)
        else {
            return None;
        };
        let [region] = regions.as_slice() else {
            return None;
        };
        match self.area {
            Some(area) => part_of(region, area, *texel),
            None => Some(*region),
        }
    }

    /// The uses of the dispatch that takes a copy back ([`dispatched`](Self::dispatched)): a
    /// sampled read of the shadow and a storage write of the buffer's bytes, in the compute
    /// stage.
    pub(super) fn dispatched_uses(&self) -> Vec<Use<ResourceKey>> {
        let c = &self.copies;
        let (truth, range) = c.truth();
        vec![
            Use::reads(
                c.shadow_key,
                c.tracked,
                c.shadow_range.clone(),
                Access::new(
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_SAMPLED_READ,
                ),
            ),
            Use::writes(
                truth,
                c.tracked,
                range,
                Access::new(
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                ),
            ),
        ]
    }

    /// The copy of rows that moves this copy back's rows from its staging buffer into the
    /// buffer's bytes, made on `device` over the buffer's `VkBuffer` now (`row_copies.rs`): a
    /// copy back through a staging buffer, of a texture over a buffer's bytes, a texture
    /// buffer's, a rotated texture's, or a heap's two-plane or tiled texture's. `None` for any
    /// other copy.
    pub(super) fn staged_back(&self, device: &Device) -> Option<RowCopies> {
        let (ShadowSource::Bytes { buffer, target, .. }, Step::Back) =
            (&self.copies.source, &self.step)
        else {
            return None;
        };
        let staging = match target {
            ShadowTarget::Image {
                staging: Some(staging),
                ..
            }
            | ShadowTarget::Buffer(staging)
            | ShadowTarget::Rotated { staging, .. }
            | ShadowTarget::Planes { staging, .. }
            | ShadowTarget::Tiled { staging, .. } => staging,
            ShadowTarget::Image { staging: None, .. } => return None,
        };
        let back = staging.back.clone();
        Some(RowCopies::new(device, staging.raw(), buffer.raw(), back))
    }

    /// The copy's stages where its staged rows go back by `back` ([`staged_back`]): with the
    /// compute stage where they dispatch.
    ///
    /// [`staged_back`]: Self::staged_back
    pub(super) fn stages_with(&self, back: Option<&RowCopies>) -> vk::PipelineStageFlags2 {
        match back {
            Some(rows) if rows.dispatched() => self.stages() | rows.stages(),
            _ => self.stages(),
        }
    }

    /// The copy's uses where its staged rows go back by `back` ([`staged_back`]): where they
    /// dispatch, the write of the buffer's bytes is the dispatch's, and so is the read of a
    /// texture buffer's staging buffer, its shadow; an image's shadow is read by its copy into
    /// the staging buffer.
    ///
    /// [`staged_back`]: Self::staged_back
    pub(super) fn uses_with(&self, back: Option<&RowCopies>) -> Vec<Use<ResourceKey>> {
        let mut uses = self.uses();
        let (Some(rows), Step::Back, ShadowSource::Bytes { target, .. }) =
            (back, &self.step, &self.copies.source)
        else {
            return uses;
        };
        if !rows.dispatched() {
            return uses;
        }
        let shadow_read = matches!(target, ShadowTarget::Buffer(_));
        for used in &mut uses {
            if !used.write.is_empty() {
                used.write = rows.write();
            } else if shadow_read && !used.read.is_empty() {
                used.read = rows.read();
            }
        }
        uses
    }

    /// The stages the copy runs in: a rotated texture's copies dispatch the rotation kernel, a
    /// heap's two-plane texture's the two-plane kernel, a tiled heap texture's the tiled texel
    /// kernel (its keep too), a keep that dispatches the copy back's kernel
    /// (`ShadowMerge::dispatches`) that kernel, and a converted texture's copy in its refresh and
    /// its conversion, dispatched through indirect arguments.
    pub(super) fn stages(&self) -> vk::PipelineStageFlags2 {
        let tiled = matches!(self.copies.target(), Some(ShadowTarget::Tiled { .. }));
        let rotated = tiled
            || matches!(
                self.copies.target(),
                Some(ShadowTarget::Rotated { .. } | ShadowTarget::Planes { .. })
            );
        let converted = matches!(self.copies.source, ShadowSource::Converted { .. });
        let encoded = matches!(self.copies.source, ShadowSource::YCbCrWrite { .. });
        // A render shadow's copy in loads it through the conversion kernel (`ycbcr.rs`).
        let loaded = matches!(
            &self.copies.source,
            ShadowSource::YCbCrWrite { write_back, .. } if write_back.loads()
        );
        match self.step {
            Step::Merge(_) => {
                vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER
            }
            Step::Into | Step::Back if rotated => {
                vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER
            }
            Step::Keep(ref merge) if tiled || merge.dispatches() => {
                vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER
            }
            // The refresh's dispatch reads the conversion's indirect arguments (`ycbcr.rs`).
            Step::Into if converted => {
                vk::PipelineStageFlags2::TRANSFER
                    | vk::PipelineStageFlags2::DRAW_INDIRECT
                    | vk::PipelineStageFlags2::COMPUTE_SHADER
            }
            Step::Back if encoded => {
                vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER
            }
            Step::Into if loaded => {
                vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER
            }
            Step::Into | Step::Back | Step::Keep(_) => vk::PipelineStageFlags2::TRANSFER,
        }
    }

    /// The copy's uses: a read of one side and a write of the other, in the transfer stage; a
    /// keep's and a merge's as `ShadowMerge` gives them.
    pub(super) fn uses(&self) -> Vec<Use<ResourceKey>> {
        let c = &self.copies;
        let read = Access::new(
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_READ,
        );
        let write = Access::new(
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_WRITE,
        );
        let (truth, range) = c.truth();
        if let ShadowSource::Converted { conversion, .. } = &c.source {
            // The planes' bytes are copied (a whole conversion) or read by the refresh's
            // dispatch (`ycbcr.rs`).
            let read = Access::new(
                vk::PipelineStageFlags2::COPY | vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::SHADER_STORAGE_READ,
            );
            return match &self.step {
                Step::Into => conversion.uses(Use::reads(truth, c.tracked, range, read), c),
                // Nothing writes the image: no copy back, keep or merge is made of it.
                Step::Back | Step::Keep(_) | Step::Merge(_) => Vec::new(),
            };
        }
        if let ShadowSource::YCbCrWrite { write_back, .. } = &c.source {
            return match &self.step {
                // A render shadow's load converts the planes into every texel.
                Step::Into if write_back.loads() => {
                    write_back.load_uses(Use::reads(truth, c.tracked, range, read), c)
                }
                // A clear of every texel: nothing of the truth is read.
                Step::Into => vec![Use::writes(
                    c.shadow_key,
                    c.tracked,
                    c.shadow_range.clone(),
                    Access::new(
                        vk::PipelineStageFlags2::CLEAR,
                        vk::AccessFlags2::TRANSFER_WRITE,
                    ),
                )],
                Step::Back => write_back.uses(
                    Use {
                        key: truth,
                        tracked: c.tracked,
                        range,
                        read,
                        write,
                    },
                    c,
                ),
                // No merge serves the planes (`unmerged`): no keep or merge is made of it.
                Step::Keep(_) | Step::Merge(_) => Vec::new(),
            };
        }
        match &self.step {
            Step::Into => vec![
                Use::reads(truth, c.tracked, range, read),
                Use::writes(c.shadow_key, c.tracked, c.shadow_range.clone(), write),
            ],
            Step::Back => vec![
                Use::reads(c.shadow_key, c.tracked, c.shadow_range.clone(), read),
                Use::writes(truth, c.tracked, range, write),
            ],
            Step::Keep(merge) => merge.keep_uses(c),
            Step::Merge(merge) => merge.merge_uses(c),
        }
    }

    /// Whether this keep may copy the staging buffer its texture's copy in just filled
    /// ([`ShadowCopies::keeps_from_staging`]): a keep whose kept part a copy writes.
    pub(super) fn keeps_from_staging(&self) -> bool {
        matches!(&self.step, Step::Keep(merge) if merge.copied_kept_part().is_some())
            && self.copies.keeps_from_staging()
    }

    /// Record this keep as a copy of its texture's staging buffer
    /// ([`ShadowCopies::record_kept_from_staging`]), where it
    /// [`keeps_from_staging`](Self::keeps_from_staging); any other copy as
    /// [`record`](Self::record) records it.
    ///
    /// # Safety
    ///
    /// As `record`'s, right after its texture's copy in, with no copy of its shadow between.
    pub(super) unsafe fn record_keep_from_staging(
        &self,
        device: &Vk,
        command_buffer: vk::CommandBuffer,
    ) {
        if let Step::Keep(merge) = &self.step {
            if let Some(to) = merge.copied_kept_part() {
                // SAFETY: the caller's contract.
                return unsafe {
                    (self.copies).record_kept_from_staging(device, command_buffer, to)
                };
            }
        }
        // SAFETY: the caller's contract.
        unsafe { self.record(device, command_buffer, None) }
    }

    /// Record the copy into `command_buffer`, its staged rows back (where it has them) by `back`
    /// ([`staged_back`](Self::staged_back)).
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording, outside a render pass, after a barrier that orders the copy
    /// after what precedes it; the buffer and the shadow are live until it completes, and an
    /// image is in the `GENERAL` layout; `back` is this copy's `staged_back`, over the buffer's
    /// `VkBuffer` as it is now.
    pub(super) unsafe fn record(
        &self,
        device: &Vk,
        command_buffer: vk::CommandBuffer,
        back: Option<&RowCopies>,
    ) {
        // SAFETY: the caller's.
        unsafe { self.record_gated(device, command_buffer, back, None) }
    }

    /// [`record`](Self::record), a copy in of a 2D texture's image straight from the buffer
    /// recorded under `gate` where given: its regions decided now, as its batch is submitted
    /// ([`GatedCopyIn`], `dirty_pages.rs`).
    ///
    /// # Safety
    ///
    /// As [`record`](Self::record).
    pub(super) unsafe fn record_gated(
        &self,
        device: &Vk,
        command_buffer: vk::CommandBuffer,
        back: Option<&RowCopies>,
        gate: Option<Arc<CopyInGate>>,
    ) {
        let c = &self.copies;
        let moves = self.moves.as_deref();
        let into_shadow = match &self.step {
            Step::Into => true,
            Step::Back => false,
            // SAFETY: the caller's contract, which the merge's records take.
            Step::Keep(merge) => {
                return unsafe { merge.record_keep(device, command_buffer, (c, moves)) }
            }
            // SAFETY: as above.
            Step::Merge(merge) => {
                return unsafe { merge.record_merge(device, command_buffer, (c, moves)) }
            }
        };
        let (buffer, target) = match &c.source {
            ShadowSource::Bytes { buffer, target, .. } => (buffer.raw(), target),
            ShadowSource::Converted {
                buffer, conversion, ..
            } => {
                if into_shadow {
                    // SAFETY: the caller's contract, which the conversion's record takes.
                    unsafe { conversion.record(device, command_buffer, buffer.raw()) };
                }
                return;
            }
            ShadowSource::YCbCrWrite {
                buffer,
                image,
                write_back,
                ..
            } => {
                if into_shadow && write_back.loads() {
                    // SAFETY: the caller's contract, which the load's record takes.
                    unsafe { write_back.record_load(device, command_buffer, buffer.raw()) };
                } else if into_shadow {
                    let range = vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    };
                    let zero = vk::ClearColorValue { uint32: [0; 4] };
                    // SAFETY: the caller's contract; the shadow takes transfer writes (made so,
                    // `texture/ycbcr_image.rs`) and the range is its one level and slice.
                    unsafe {
                        device.cmd_clear_color_image(
                            command_buffer,
                            *image,
                            vk::ImageLayout::GENERAL,
                            &zero,
                            &[range],
                        );
                    }
                } else {
                    // SAFETY: the caller's contract, which the write-back's record takes.
                    unsafe { write_back.record(device, command_buffer, buffer.raw(), self.area) };
                }
                return;
            }
            ShadowSource::Image {
                image,
                shadow,
                regions,
                through,
                ..
            } => {
                let (from, to) = if into_shadow {
                    (*image, *shadow)
                } else {
                    (*shadow, *image)
                };
                if let Some(through) = through {
                    let (buffer, _) = through.buffer.at(0);
                    // SAFETY: the caller's contract; the regions lie in both images, which have
                    // the same shape, and in the buffer, which lives with the shadow; the
                    // barrier orders the copy out of the buffer after the copy into it.
                    unsafe {
                        device.cmd_copy_image_to_buffer(
                            command_buffer,
                            from,
                            vk::ImageLayout::GENERAL,
                            buffer,
                            &through.regions,
                        );
                        Barrier::everything().record(device, command_buffer);
                        device.cmd_copy_buffer_to_image(
                            command_buffer,
                            buffer,
                            to,
                            vk::ImageLayout::GENERAL,
                            &through.regions,
                        );
                    }
                    return;
                }
                // A copy in cut to a rectangle moves its texels alone; a copy back moves every
                // texel (`ShadowCopies::cuts_back`).
                let regions: Vec<vk::ImageCopy> = if into_shadow {
                    match (self.area, c.cuts(), one_image_copy(regions)) {
                        (Some(area), true, Some(region)) => match image_part_of(region, area) {
                            Some(part) => vec![part],
                            // The command reads no texel.
                            None => return,
                        },
                        _ => regions.clone(),
                    }
                } else {
                    regions
                        .iter()
                        .map(|region| vk::ImageCopy {
                            src_subresource: region.dst_subresource,
                            src_offset: region.dst_offset,
                            dst_subresource: region.src_subresource,
                            dst_offset: region.src_offset,
                            extent: region.extent,
                        })
                        .collect()
                };
                // SAFETY: the caller's contract; the regions lie in both images, which have
                // the same shape and formats of one copy unit (planned when the view was made).
                unsafe {
                    device.cmd_copy_image(
                        command_buffer,
                        from,
                        vk::ImageLayout::GENERAL,
                        to,
                        vk::ImageLayout::GENERAL,
                        &regions,
                    );
                }
                return;
            }
        };
        let (image, regions, staging) = match target {
            ShadowTarget::Image {
                image,
                staging,
                texel,
                ..
            } => {
                // Going back takes the regions that write each byte once (the variant's `back`).
                let regions = target.image_regions(into_shadow).unwrap_or_default();
                let regions = match (self.area, staging, regions, *texel) {
                    (Some(area), None, [region], Some(texel)) => {
                        match part_of(region, area, texel) {
                            Some(part) => std::borrow::Cow::Owned(vec![part]),
                            // The commands read or wrote no texel.
                            None => return,
                        }
                    }
                    _ => std::borrow::Cow::Borrowed(regions),
                };
                (*image, regions, staging.as_ref())
            }
            ShadowTarget::Rotated {
                image,
                staging,
                turn,
            } => {
                let turned = match (into_shadow, self.kept) {
                    (true, false) => Turn::Load,
                    (true, true) => Turn::Keep,
                    (false, _) => Turn::Store,
                };
                // SAFETY: the caller's contract, which the turn's record takes.
                unsafe {
                    turn.record(
                        device,
                        command_buffer,
                        *image,
                        (buffer, staging),
                        turned,
                        back,
                    );
                }
                return;
            }
            ShadowTarget::Planes {
                image,
                staging,
                planes,
                ..
            } => {
                // SAFETY: the caller's contract, which the planes' record takes.
                unsafe {
                    planes.record(
                        device,
                        command_buffer,
                        *image,
                        (buffer, staging),
                        into_shadow,
                        back,
                    );
                }
                return;
            }
            ShadowTarget::Tiled {
                image,
                staging,
                tiles,
            } => {
                // SAFETY: the caller's contract, which the tiles' record takes; the moves are held
                // with the copy.
                unsafe {
                    tiles.record(
                        device,
                        command_buffer,
                        *image,
                        (buffer, staging),
                        (into_shadow, moves),
                        back,
                    );
                }
                return;
            }
            ShadowTarget::Buffer(staging) => {
                // SAFETY: the caller's contract; the copies lie in both buffers (planned when
                // the texture was made), and the staging buffer lives with the texture.
                unsafe {
                    if into_shadow {
                        device.cmd_copy_buffer(
                            command_buffer,
                            buffer,
                            staging.raw(),
                            &[staging.into],
                        );
                    } else {
                        record_back(device, command_buffer, (staging, buffer), back);
                    }
                }
                return;
            }
        };
        let texels = staging.map_or(buffer, ShadowStaging::raw);
        // SAFETY: the caller's contract; the regions lie in the buffers and the image (checked
        // when the texture was made), and the staging buffer lives with the texture.
        unsafe {
            if into_shadow {
                if let Some(staging) = staging {
                    device.cmd_copy_buffer(command_buffer, buffer, texels, &[staging.into]);
                    Barrier::everything().record(device, command_buffer);
                }
                if let Some(gate) = gate.filter(|_| staging.is_none()) {
                    let mut chosen = Vec::new();
                    gate.regions(&regions, &mut chosen);
                    if !chosen.is_empty() {
                        device.cmd_copy_buffer_to_image(
                            command_buffer,
                            texels,
                            image,
                            vk::ImageLayout::GENERAL,
                            &chosen,
                        );
                    }
                    return;
                }
                device.cmd_copy_buffer_to_image(
                    command_buffer,
                    texels,
                    image,
                    vk::ImageLayout::GENERAL,
                    &regions,
                );
            } else {
                device.cmd_copy_image_to_buffer(
                    command_buffer,
                    image,
                    vk::ImageLayout::GENERAL,
                    texels,
                    &regions,
                );
                if let Some(staging) = staging {
                    Barrier::everything().record(device, command_buffer);
                    record_back(device, command_buffer, (staging, buffer), back);
                }
            }
        }
    }
}

/// The `VkBuffer` bytes `regions` copy, `texel` bytes a texel, from the first region's first byte
/// to the last row's end; `None` for none, or a region of depth.
fn copied_bytes(regions: &[vk::BufferImageCopy], texel: u64) -> Option<Range<u64>> {
    let mut bytes: Option<Range<u64>> = None;
    for region in regions {
        let extent = region.image_extent;
        if extent.depth > 1 || extent.width == 0 || extent.height == 0 {
            return None;
        }
        let row = match region.buffer_row_length {
            0 => extent.width,
            row => row,
        };
        let pitch = u64::from(row) * texel;
        let start = region.buffer_offset;
        let end = start + u64::from(extent.height - 1) * pitch + u64::from(extent.width) * texel;
        bytes = Some(match bytes {
            Some(b) => b.start.min(start)..b.end.max(end),
            None => start..end,
        });
    }
    bytes
}

/// `region` (a one-level, one-slice texture's every texel, `texel` bytes each in the buffer) cut
/// to the texels inside `[x0, y0, x1, y1)`: `None` where none lies there.
pub(super) fn part_of(
    region: &vk::BufferImageCopy,
    [x0, y0, x1, y1]: [u32; 4],
    texel: u64,
) -> Option<vk::BufferImageCopy> {
    let extent = region.image_extent;
    let (x1, y1) = (x1.min(extent.width), y1.min(extent.height));
    if x0 >= x1 || y0 >= y1 {
        return None;
    }
    let row = match region.buffer_row_length {
        0 => extent.width,
        row => row,
    };
    Some(vk::BufferImageCopy {
        buffer_offset: region.buffer_offset
            + (u64::from(y0) * u64::from(row) + u64::from(x0)) * texel,
        buffer_row_length: row,
        image_offset: vk::Offset3D {
            x: region.image_offset.x + x0 as i32,
            y: region.image_offset.y + y0 as i32,
            z: region.image_offset.z,
        },
        image_extent: vk::Extent3D {
            width: x1 - x0,
            height: y1 - y0,
            depth: extent.depth,
        },
        ..*region
    })
}

/// Record the rows of `staging` back into `buffer`: by `back` where the caller made it
/// ([`ShadowCopy::staged_back`]), else as one copy of every row. Rotated textures' and heap
/// textures' copies back end so too (`rotated.rs`, `texture/heap_two_planes.rs`,
/// `texture/heap_tiled.rs`).
///
/// # Safety
///
/// As [`ShadowCopy::record`], after a barrier from the staging buffer's writes.
pub(super) unsafe fn record_back(
    device: &Vk,
    command_buffer: vk::CommandBuffer,
    (staging, buffer): (&ShadowStaging, vk::Buffer),
    back: Option<&RowCopies>,
) {
    // SAFETY: the caller's contract; the rows lie in the staging buffer and the buffer's bytes
    // (planned when the texture was made), and the staging buffer lives with the texture.
    unsafe {
        match back {
            Some(rows) => rows.record(device, command_buffer),
            None => device.cmd_copy_buffer(command_buffer, staging.raw(), buffer, &staging.back),
        }
    }
}

/// The one region of `regions`, a shadow's of another image, where it is one level and one slice
/// of a 2D image: the texels a rectangle of the texture's own names.
fn one_image_copy(regions: &[vk::ImageCopy]) -> Option<&vk::ImageCopy> {
    match regions {
        [region] if region.src_subresource.layer_count == 1 && region.extent.depth == 1 => {
            Some(region)
        }
        _ => None,
    }
}

/// `region` (one level and one slice of an image, copied into the same texels of its shadow) cut
/// to the texels inside `[x0, y0, x1, y1)`: `None` where none lies there.
pub(super) fn image_part_of(
    region: &vk::ImageCopy,
    [x0, y0, x1, y1]: [u32; 4],
) -> Option<vk::ImageCopy> {
    let extent = region.extent;
    let (x1, y1) = (x1.min(extent.width), y1.min(extent.height));
    if x0 >= x1 || y0 >= y1 {
        return None;
    }
    let moved = |offset: vk::Offset3D| vk::Offset3D {
        x: offset.x + x0 as i32,
        y: offset.y + y0 as i32,
        z: offset.z,
    };
    Some(vk::ImageCopy {
        src_offset: moved(region.src_offset),
        dst_offset: moved(region.dst_offset),
        extent: vk::Extent3D {
            width: x1 - x0,
            height: y1 - y0,
            depth: extent.depth,
        },
        ..*region
    })
}

/// Whether the box `inner` (an offset and extent of texels) lies inside the box `outer`.
fn within(
    (offset, extent): (vk::Offset3D, vk::Extent3D),
    (at, size): (vk::Offset3D, vk::Extent3D),
) -> bool {
    let axes = [
        (offset.x, extent.width, at.x, size.width),
        (offset.y, extent.height, at.y, size.height),
        (offset.z, extent.depth, at.z, size.depth),
    ];
    axes.into_iter()
        .all(|(start, length, outer, outer_length)| {
            let (start, outer) = (i64::from(start), i64::from(outer));
            outer <= start && start + i64::from(length) <= outer + i64::from(outer_length)
        })
}

/// A box of an image's texels a command writes every texel of: a blit's copy region into the
/// image, in its aspects, level and slices.
#[derive(Clone, Copy)]
pub(super) struct Covered {
    pub(super) image: vk::Image,
    pub(super) subresource: vk::ImageSubresourceLayers,
    pub(super) offset: vk::Offset3D,
    pub(super) extent: vk::Extent3D,
}

impl Covered {
    /// The boxes `regions` of copies into `image` write.
    pub(super) fn into_image(image: vk::Image, regions: &[vk::BufferImageCopy]) -> Vec<Covered> {
        (regions.iter())
            .map(|region| Covered {
                image,
                subresource: region.image_subresource,
                offset: region.image_offset,
                extent: region.image_extent,
            })
            .collect()
    }

    /// The boxes `regions` of copies into the image `to` write.
    pub(super) fn into_image_from_image(to: vk::Image, regions: &[vk::ImageCopy]) -> Vec<Covered> {
        (regions.iter())
            .map(|region| Covered {
                image: to,
                subresource: region.dst_subresource,
                offset: region.dst_offset,
                extent: region.extent,
            })
            .collect()
    }
}

/// Mark the shadows of `shadows` a command with `uses` replaces ([`ShadowUse::overwrites`]): it
/// writes through the shadow, reads none of its texels, and writes every texel it holds inside
/// the boxes `covered` ([`ShadowCopies::covered_by`]), so it takes no copy in and no keep, and
/// its copy back moves the texels it wrote; but over logged host pages, a copy in, a keep and a
/// merge ([`overwrites_unkept`]).
pub(super) fn overwritten(
    shadows: &mut [ShadowUse],
    uses: &[Use<ResourceKey>],
    covered: &[Covered],
) {
    if covered.is_empty() {
        return;
    }
    for shadow in shadows {
        let key = shadow.copies.shadow_key;
        let reads = (uses.iter()).any(|used| used.key == key && !used.read.is_empty());
        shadow.overwrites = shadow.writes && !reads && shadow.copies.covered_by(covered);
    }
}

/// Whether a command writes through `shadow` without replacing every texel, where it merges back
/// ([`ShadowCopies::merges_back`]): kept and merged back (the module's rules), as a command that
/// writes them two ways too ([`two_ways`]) is.
fn merged_back(shadow: &ShadowUse) -> bool {
    shadow.writes && !overwrites_unkept(shadow) && shadow.copies.merges_back()
}

/// Whether the merge after a command that writes through `shadow` and merges back, keeping the
/// texels inside `kept` (`None` every texel), stores whole blocks where it is cut to the texels'
/// rows (`shadow_merge.rs`): the command replaces every texel ([`ShadowUse::overwrites`], over
/// logged pages, as [`merged_back`] takes it), or writes through it only by a render pass's
/// stores of the tiles of its area ([`ShadowUse::tile_stores`]) and the keep is that area.
fn stores_blocks(shadow: &ShadowUse, kept: Option<[u32; 4]>) -> bool {
    shadow.overwrites || (shadow.tile_stores && kept.is_some() && kept == shadow.area)
}

/// Whether a command that replaces every texel of `shadow` ([`ShadowUse::overwrites`]) takes no
/// copy in and no keep, its copy back a plain copy of the texels it wrote: every such shadow but
/// one over logged host pages ([`ShadowCopies::logged`]), which is copied in, kept and merged back
/// as a command that leaves texels in place is (the module's **Overwrites over logged pages**).
fn overwrites_unkept(shadow: &ShadowUse) -> bool {
    shadow.overwrites && !shadow.copies.logged()
}

/// Whether a command that uses the shadowed textures `shadows` and has `uses` writes the bytes
/// of `shadows[i]` two ways: through it, and through `uses` or another shadowed texture over
/// them (the module's rules).
fn two_ways(shadows: &[ShadowUse], i: usize, uses: &[Use<ResourceKey>]) -> bool {
    // A shadow whose writes Metal drops writes the bytes no way.
    let writes = |shadow: &ShadowUse| shadow.writes && !shadow.copies.drops_writes;
    let Some(shadow) = shadows.get(i).filter(|shadow| writes(shadow)) else {
        return false;
    };
    let bytes_written = uses
        .iter()
        .any(|used| !used.write.is_empty() && shadow.copies.meets(used));
    let shadow_written = shadows.iter().enumerate().any(|(j, other)| {
        j != i
            && writes(other)
            && other.copies.shadow_key != shadow.copies.shadow_key
            && other.copies.shares_truth(&shadow.copies)
    });
    bytes_written || shadow_written
}

/// The shadowed textures among `textures` (bound, made resident, or a pass's) a command with
/// `uses` uses, each once, writing through it when any of its uses writes.
pub(super) fn shadow_uses<'a>(
    textures: impl Iterator<Item = &'a Texture>,
    uses: &[Use<ResourceKey>],
) -> Vec<ShadowUse> {
    // A texture's own shadow, and the one a binding beyond its usage made
    // (`texture/usage_shadow.rs`).
    let mut copies = Vec::new();
    for texture in textures {
        copies.extend(texture.shadow().cloned());
        copies.extend(texture.usage_shadow_copies());
    }
    copies_used(copies.iter(), uses)
}

/// The shadowed textures of `shadows` a command with `uses` uses, each once, writing through it
/// when any of its uses writes.
pub(super) fn copies_used<'a>(
    shadows_of: impl Iterator<Item = &'a Arc<ShadowCopies>>,
    uses: &[Use<ResourceKey>],
) -> Vec<ShadowUse> {
    let mut shadows: Vec<ShadowUse> = Vec::new();
    for copies in shadows_of {
        if shadows
            .iter()
            .any(|shadow| shadow.copies.shadow_key == copies.shadow_key)
        {
            continue;
        }
        let mut used = uses.iter().filter(|used| used.key == copies.shadow_key);
        let Some(first) = used.next() else {
            continue;
        };
        let writes = !first.write.is_empty() || used.any(|other| !other.write.is_empty());
        shadows.push(ShadowUse {
            copies: Arc::clone(copies),
            writes,
            overwrites: false,
            kept: false,
            area: None,
            loads: None,
            tile_stores: false,
        });
    }
    shadows
}

/// Add `shadow` to `shadows`: once per shadowed texture, writing through it when either use
/// writes (where both write, inside the rectangle bounding both areas, and by tile stores alone
/// only where both do), reading its texels inside the rectangle bounding where both read them
/// (anywhere where either does not bound that), and overwriting it, or leaving its texels in
/// place, only when both do.
pub(super) fn join(shadows: &mut Vec<ShadowUse>, shadow: &ShadowUse) {
    match shadows
        .iter_mut()
        .find(|other| other.copies.shadow_key == shadow.copies.shadow_key)
    {
        Some(other) => {
            other.area = match (other.writes, shadow.writes) {
                (true, true) => bounding(other.area, shadow.area),
                (false, true) => shadow.area,
                (_, false) => other.area,
            };
            other.tile_stores = match (other.writes, shadow.writes) {
                (true, true) => other.tile_stores && shadow.tile_stores,
                (false, true) => shadow.tile_stores,
                (_, false) => other.tile_stores,
            };
            other.loads = bounding(other.loads, shadow.loads);
            other.writes |= shadow.writes;
            other.overwrites &= shadow.overwrites;
            other.kept &= shadow.kept;
        }
        None => shadows.push(ShadowUse {
            copies: Arc::clone(&shadow.copies),
            writes: shadow.writes,
            overwrites: shadow.overwrites,
            kept: shadow.kept,
            area: shadow.area,
            loads: shadow.loads,
            tile_stores: shadow.tile_stores,
        }),
    }
}

/// Where a shadowed texture's image stands against its truth in an encoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Standing {
    /// Not copied in since the encoder met the texture: its next copy in is the encoder's first.
    Unread,
    /// The image holds the buffer's bytes as the encoder's commands so far left them, inside
    /// `held`.
    Current,
    /// The truth changed since the image's copy in, for the reason given.
    Lost(Lost),
}

/// A shadowed texture an encoder used, and where its image stands.
struct Entry<M> {
    copies: Arc<ShadowCopies>,
    standing: Standing,
    /// Where the image holds them when [`Standing::Current`]: the texels inside `[x0, y0, x1, y1)`, all its
    /// last copy in moved ([`ShadowUse::loads`]); `None` for every texel. Its other texels are
    /// whatever the image held before, which no command of the encoder reads and no copy back
    /// moves: a command that needs them current copies back what was written and copies every
    /// texel in first.
    held: Option<[u32; 4]>,
    /// A command wrote through the texture since the last copy back.
    dirty: bool,
    /// Where those commands wrote through it, where they bound it ([`ShadowUse::area`]).
    area: Option<[u32; 4]>,
    /// The scratch the image was kept in before a command that writes its bytes two ways, until
    /// the merge after it, or before the first command since the last copy back that wrote
    /// through a texture that merges back ([`ShadowCopies::merges_back`]), until the next copy
    /// back, which is then a merge.
    merge: Option<M>,
    /// The keep in `merge` is a command's that writes the bytes two ways: merged before the
    /// encoder's next command, after which the image no longer holds the bytes.
    two_ways: bool,
    /// The scratch of the texture's keeps where it merges back, made at the first
    /// command of the encoder that writes through it, and kept in again at each keep after a
    /// merge: each merge's dispatches end with a barrier to everything after them.
    scratch: Option<M>,
    /// Taken from the command buffer's carry and not used by a command of this encoder yet: its
    /// first use's copy in is the one the carry takes off the run
    /// (`WorkCounts::shadow_texels_first_copy_in_carried`).
    carried: bool,
    /// [`ShadowCopies::touched`] when the encoder met the shadow: its end carries the image to
    /// the command buffer's later encoders only where it still stands there, no other encoder
    /// having ended with the shadow in between ([`ShadowCarry`]).
    met: u64,
    /// When [`Standing::Current`], the rectangle bounding the texels whose bytes changed since the image's
    /// copy in, where only writes [`Within`] it changed them (the module's rules): the next
    /// command that uses the texture copies those in first, or those of them it reads where it
    /// reads only inside a rectangle and writes none, the rest left here. `None` where none
    /// changed.
    stale: Option<[u32; 4]>,
}

impl<M> Entry<M> {
    /// Whether a change of the bytes of `texels`' texels inside a rectangle leaves the image
    /// current but there ([`Entry::stale`]): it holds the bytes, and no write of its own or keep,
    /// and its texels are `texels`' ([`ShadowCopies::same_texels`]).
    fn stale_within(&self, texels: &ShadowCopies) -> bool {
        self.standing == Standing::Current
            && !self.dirty
            && self.merge.is_none()
            && self.copies.same_texels(texels)
    }

    /// Note a change of the truth for `lost`: the image no longer holds it. One not copied in
    /// yet stays so, and one that lost it already keeps the first reason.
    fn lose(&mut self, lost: Lost) {
        if self.standing == Standing::Current {
            self.standing = Standing::Lost(lost);
        }
    }

    /// Note a change of the bytes of `texels`' texels inside `area` (`[x0, y0, x1, y1)`), or
    /// anywhere where `area` is `None`, for `lost`: the image is stale there, or no longer
    /// current.
    fn changed(&mut self, texels: &ShadowCopies, area: Option<[u32; 4]>, lost: Lost) {
        match area.filter(|_| self.stale_within(texels)) {
            Some(area) => {
                self.stale = Some(match self.stale {
                    Some(stale) => [
                        stale[0].min(area[0]),
                        stale[1].min(area[1]),
                        stale[2].max(area[2]),
                        stale[3].max(area[3]),
                    ],
                    None => area,
                });
            }
            None => self.lose(lost),
        }
    }
}

/// A command's writes of a texture's bytes that change none of its texels outside a rectangle:
/// a render pass instance's of a color attachment it loads, which only its draws change, inside
/// their scissors (`render/shadows.rs`).
pub(super) struct Within {
    /// The writes' uses of the bytes.
    pub(super) uses: Vec<Use<ResourceKey>>,
    /// A shadow of the texture, whose texels are the attachment's pixels: an image whose texels
    /// are laid out as its ([`ShadowCopies::same_texels`]) changed inside `area` alone. `None`
    /// where the texture has none, and then every image over the bytes changed anywhere.
    pub(super) texels: Option<Arc<ShadowCopies>>,
    /// The rectangle, `[x0, y0, x1, y1)` of the texels.
    pub(super) area: [u32; 4],
}

/// A command's use of a shadowed texture: its copies, and whether it writes through it.
pub(super) struct ShadowUse {
    pub(super) copies: Arc<ShadowCopies>,
    pub(super) writes: bool,
    /// The command replaces every texel of the texture before it reads any (a render pass's
    /// attachment that clears each of its aspects over the whole of a one-level, one-slice
    /// texture; a blit's copy into every texel of such a texture over a buffer's bytes,
    /// [`overwritten`]): it takes no copy in and no keep. A render encoder's one pass is its only
    /// command with attachments, so nothing earlier in the encoder wrote through the texture;
    /// where the pass does not store it either, nothing is copied back, and the buffer keeps its
    /// bytes, Metal's answer (a Metal probe on the M3 Max and an M4).
    pub(super) overwrites: bool,
    /// The command leaves the texture's texels in place, but where it overwrites them: a render
    /// pass that draws nothing, of an attachment it renders. A rotated texture's copy in then
    /// turns its texels by the inverse of its store (`rotated::Turn::Keep`), where Metal's pass
    /// loads them otherwise. (An attachment such a pass loads, whose shadow is an image of a
    /// buffer's bytes, takes no use at all: `render/shadows.rs`.)
    pub(super) kept: bool,
    /// Where the command writes through the texture, where it bounds that: the texels inside
    /// `[x0, y0, x1, y1)` of a one-level, one-slice texture's image (a pass's attachment it
    /// loads, which only its draws' scissors change, `render/shadows.rs`); `None` for anywhere.
    pub(super) area: Option<[u32; 4]>,
    /// Where the command reads the texture's texels, where it bounds that: the texels inside
    /// `[x0, y0, x1, y1)` of a one-level, one-slice texture's image, whose others it reads at
    /// most into the image alone, never into what it writes elsewhere or what a copy back moves
    /// (a pass's attachment it loads and takes in no other way, whose fragments run only inside
    /// its draws' scissors and read the attachment at their own pixel alone,
    /// `render/shadows.rs`); `None` for anywhere. Its copy in moves only those texels.
    pub(super) loads: Option<[u32; 4]>,
    /// The command writes through the texture only as a render pass's color attachment it loads
    /// and stores inside `area` (`render/shadows.rs`): Metal's own store of such a pass writes
    /// back every texel of each tile its draws reach, those no fragment changed among them, so
    /// its merge stores whole blocks of the rectangle's rows (the module's **Stored
    /// attachments**).
    pub(super) tile_stores: bool,
}

/// The rectangle bounding `a` and `b`, each `[x0, y0, x1, y1)`; `None` (anywhere) where either is.
fn bounding(a: Option<[u32; 4]>, b: Option<[u32; 4]>) -> Option<[u32; 4]> {
    let ([a0, a1, a2, a3], [b0, b1, b2, b3]) = (a?, b?);
    Some([a0.min(b0), a1.min(b1), a2.max(b2), a3.max(b3)])
}

/// The texels inside both `a` and `b`, each `[x0, y0, x1, y1)`; `None` where they share none.
fn meet([a0, a1, a2, a3]: [u32; 4], [b0, b1, b2, b3]: [u32; 4]) -> Option<[u32; 4]> {
    let r = [a0.max(b0), a1.max(b1), a2.min(b2), a3.min(b3)];
    (r[0] < r[2] && r[1] < r[3]).then_some(r)
}

/// The rectangle bounding the texels of `a` outside `b`, each `[x0, y0, x1, y1)`; `None` where `b`
/// covers `a`.
fn outside(a: [u32; 4], b: [u32; 4]) -> Option<[u32; 4]> {
    let ([a0, a1, a2, a3], [b0, b1, b2, b3]) = (a, b);
    if meet(a, b).is_none() {
        return Some(a);
    }
    let (across, down) = (b0 <= a0 && a2 <= b2, b1 <= a1 && a3 <= b3);
    // Where `b` crosses `a` from side to side, the texels left are the rows above and below it;
    // from top to bottom, the columns beside it. One of the two alone is a rectangle of its own.
    let left = match (across, down) {
        (true, true) => return None,
        (true, false) => (a1 < b1, b3 < a3, [a0, a1, a2, b1], [a0, b3, a2, a3]),
        (false, true) => (a0 < b0, b2 < a2, [a0, a1, b0, a3], [b2, a1, a2, a3]),
        (false, false) => return Some(a),
    };
    match left {
        (true, false, before, _) => Some(before),
        (false, true, _, after) => Some(after),
        _ => Some(a),
    }
}

/// Whether an image holding the buffer's texels inside `held` (`None` every texel) holds those
/// inside `needs` (`None` every texel).
fn holds(held: Option<[u32; 4]>, needs: Option<[u32; 4]>) -> bool {
    match (held, needs) {
        (None, _) => true,
        (Some(_), None) => false,
        (Some([h0, h1, h2, h3]), Some([n0, n1, n2, n3])) => {
            h0 <= n0 && h1 <= n1 && n2 <= h2 && n3 <= h3
        }
    }
}

/// The shadowed textures one encoder used, and the copies they need (the module's rules), `M`
/// the scratch of a merge.
pub(super) struct ShadowSync<M = Arc<ShadowMerge>> {
    entries: Vec<Entry<M>>,
    /// What the command buffer's earlier encoders left holding the truth, for the textures this
    /// one has not met yet ([`ShadowCarry`]); [`entry`](Self::entry) takes a texture's out of it.
    carry: ShadowCarry,
    /// How the encoder's first copies in so far stood against their images' last standing, which
    /// [`tally`](Self::tally) hands to the counters.
    tally: CarryTally,
}

impl<M> Default for ShadowSync<M> {
    /// No texture used yet, and nothing carried: every texture copies in at its first use.
    fn default() -> Self {
        ShadowSync {
            entries: Vec::new(),
            carry: ShadowCarry::default(),
            tally: CarryTally::default(),
        }
    }
}

impl<M> ShadowSync<M> {
    /// An encoder of a command buffer that left `carry` holding shadowed textures' images: its
    /// first use of one of them takes no copy in, where the image still stands there
    /// ([`ShadowCarry`], [`entry`](Self::entry)).
    pub(super) fn carrying(carry: ShadowCarry) -> ShadowSync<M> {
        ShadowSync {
            entries: Vec::new(),
            carry,
            tally: CarryTally::default(),
        }
    }

    /// How the encoder's first copies in since this was last called stood against their images'
    /// last standing ([`CarryTally`]): the population the carry removes, the one Metal requires
    /// of a later command buffer, and the one no rule avoids. It sizes the carry on a run without
    /// a timing.
    pub(super) fn tally(&mut self) -> CarryTally {
        let tally = self.tally;
        self.tally = CarryTally::default();
        tally
    }

    /// Whether the encoder used no shadowed texture yet: then a command that uses none takes no
    /// copy, and changes nothing here ([`before`](Self::before), [`after`](Self::after)). What
    /// the command buffer carries is not among them: a command that drops one of those reaches
    /// [`wrote`](Self::wrote) whatever the encoder used (`EncoderCore::track`).
    pub(super) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Carry no texture's image on, and let every texture the encoder met go: the encoder's
    /// commands from here on read another writer's bytes, which each texture copies in again to
    /// hold, and its later encoders take none of the images this one leaves (`ShadowCarry`).
    pub(super) fn not_carried(&mut self) {
        self.carry.clear();
        for entry in &mut self.entries {
            entry.lose(Lost::Written);
        }
    }

    /// Note a recorded command with `uses`, whatever it touched: a write of a carried texture's
    /// truth another way leaves its image stale, so the texture is no longer carried and copies
    /// in at its next use. The encoder's own textures are [`after`](Self::after)'s, which holds
    /// where each stands; this covers the commands of an encoder that meets none of them, whose
    /// writes nothing else here would see.
    pub(super) fn wrote(&mut self, uses: &[Use<ResourceKey>]) {
        if self.carry.entries.is_empty() {
            return;
        }
        let writes = |carried: &Carried| {
            (uses.iter()).any(|used| !used.write.is_empty() && carried.copies.meets(used))
        };
        self.carry.entries.retain(|carried| !writes(carried));
    }

    /// What the encoder leaves its command buffer's later encoders: every texture whose image
    /// holds the truth as its commands left it, with where it holds it, and the textures it
    /// never met that it still carried. Every texture it met moves
    /// [`ShadowCopies::touched`] on, so a command buffer whose encoder ended with the texture
    /// between this one's two carries it no longer.
    pub(super) fn ended(&mut self) -> ShadowCarry {
        let mut carry = ShadowCarry {
            id: self.carry.id,
            entries: Vec::new(),
        };
        carry.entries.append(&mut self.carry.entries);
        for entry in &self.entries {
            let touched = entry.copies.touched.fetch_add(1, Ordering::Relaxed);
            if entry.standing != Standing::Current {
                continue;
            }
            // The image holds the truth as this command buffer's commands left it, whether or
            // not this command buffer may carry it on.
            (entry.copies.stood).store(carry.id, Ordering::Relaxed);
            if touched != entry.met {
                continue;
            }
            carry.entries.push(Carried {
                copies: Arc::clone(&entry.copies),
                touched: touched + 1,
                held: entry.held,
                stale: entry.stale,
            });
        }
        self.entries.clear();
        carry
    }

    /// How many of the textures the encoder used share their truth with one it used before
    /// (another texture over the same bytes): each copies the texels in apart
    /// (`WorkCounts::shadows_sharing_truth`).
    pub(super) fn sharing_truth(&self) -> usize {
        let entries = &self.entries;
        (1..entries.len())
            .filter(|&i| {
                (entries[..i].iter()).any(|earlier| earlier.copies.shares_truth(&entries[i].copies))
            })
            .count()
    }
}

/// Where a command buffer's earlier encoder left a shadowed texture's image, which its later
/// encoders take instead of copying the truth in again.
///
/// Metal shows an encoder the bytes as of its command buffer's commit (the module's **Copy in**
/// rule), so an image an earlier encoder of the same command buffer left holding the truth holds
/// what every later encoder of it reads. The carry is dropped wherever that no longer follows:
/// the encoder that published it did not end holding the truth, another encoder ended with the
/// shadow in between ([`ShadowCopies::touched`]), a command of the command buffer wrote the truth
/// another way ([`ShadowSync::wrote`]), work the encoder records apart stands between
/// ([`ShadowSync::leave`]), or the command buffer waits for an event or a fence, after which
/// Metal shows the encoder another writer's bytes
/// (`CommandBuffer::encode_wait_for_event`, `EncoderCore::ordered_other_work`).
pub(super) struct ShadowCarry {
    /// The command buffer's identity: a number no command buffer of the process had before and
    /// none takes again, so a shadow can name the command buffer that last left its image
    /// holding the truth ([`ShadowCopies::stood`]) without holding it. Metal names a command
    /// buffer by its pointer, which another may take once it is freed.
    id: u64,
    entries: Vec<Carried>,
}

impl Default for ShadowCarry {
    /// A command buffer carrying nothing, under an identity of its own.
    fn default() -> ShadowCarry {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        ShadowCarry {
            id: NEXT.fetch_add(1, Ordering::Relaxed),
            entries: Vec::new(),
        }
    }
}

/// How the large IOSurface planes an encoder's commands used stood against their images' last
/// standing, and what the carry took off the encoder ([`ShadowCarry`],
/// `WorkCounts::shadow_texels_first_copy_in_carried` and kin). The three large-surface figures
/// are exhaustive over an encoder's first copies in of such planes but for the ones a standing of
/// this command buffer's own could not serve, which
/// `WorkCounts::shadow_texels_first_copied_in_again` counts.
#[derive(Default, Clone, Copy)]
pub(super) struct CarryTally {
    /// The texels no copy in moved, the encoder's command buffer holding the image already: what
    /// carrying the image across its encoders takes off the run.
    pub(super) carried: u64,
    /// Of those, the large IOSurface planes'.
    pub(super) carried_large: u64,
    /// The texels an encoder's first copy in of a large plane moved where the image's last
    /// standing was an earlier command buffer's: Metal shows a later command buffer a CPU write
    /// made since the commit, so it requires this one.
    pub(super) stood_earlier_large: u64,
    /// The texels an encoder's first copy in of a large plane moved where the image never held a
    /// standing: unavoidable under any rule.
    pub(super) unstood_large: u64,
}

/// One shadowed texture's image a command buffer's earlier encoder left holding the truth
/// ([`ShadowCarry`]).
struct Carried {
    copies: Arc<ShadowCopies>,
    /// [`ShadowCopies::touched`] as the encoder that published this left it.
    touched: u64,
    /// [`Entry::held`] and [`Entry::stale`] as it left them.
    held: Option<[u32; 4]>,
    stale: Option<[u32; 4]>,
}

impl ShadowCarry {
    /// Carry nothing, under the same identity: every shadowed texture copies in again at its
    /// next use, and a first copy in of one this command buffer stood behind still reads as that.
    pub(super) fn clear(&mut self) {
        self.entries.clear();
    }

    /// The shadowed textures whose images it carries holding every texel of the truth, none
    /// stale (`dirty_pages.rs`: what a command buffer's end leaves a later one's copy in).
    pub(super) fn held_whole(&self) -> Vec<Arc<ShadowCopies>> {
        (self.entries.iter())
            .filter(|carried| carried.stale.is_none() && carried.copies.covers(carried.held))
            .map(|carried| Arc::clone(&carried.copies))
            .collect()
    }
}

/// What a merge's scratch keeps of a shadow's texels (`shadow_merge.rs`): the rectangle
/// `[x0, y0, x1, y1)` its keep, its copy of the changed texels and its merge are cut to, or `None`
/// for every texel.
pub(super) trait KeptArea {
    fn kept_area(&self) -> Option<[u32; 4]>;

    /// Whether the scratch was made for a command whose merge stores whole blocks where it can
    /// ([`stores_blocks`], `shadow_merge.rs`): only such a command's keep takes it again.
    fn in_blocks(&self) -> bool {
        false
    }
}

impl KeptArea for Arc<ShadowMerge> {
    fn kept_area(&self) -> Option<[u32; 4]> {
        self.area()
    }

    fn in_blocks(&self) -> bool {
        ShadowMerge::in_blocks(self)
    }
}

/// What a merge `ShadowSync::before` asks for is made for.
#[derive(Clone, Copy, Debug)]
pub(super) struct MergeFor {
    /// A command that writes the bytes two ways ([`two_ways`]).
    pub(super) two_ways: bool,
    /// A command whose merge stores whole blocks where it is cut to the texels' rows
    /// ([`stores_blocks`]): one that replaces every texel over logged pages
    /// ([`overwrites_unkept`]), or a render pass that writes through the texture only by its
    /// stores of the area's tiles ([`ShadowUse::tile_stores`]).
    pub(super) blocks: bool,
}

impl<M: Clone + KeptArea> ShadowSync<M> {
    /// The entry of `copies`, made when the encoder had none.
    fn entry(&mut self, copies: &Arc<ShadowCopies>) -> usize {
        match self
            .entries
            .iter()
            .position(|e| e.copies.shadow_key == copies.shadow_key)
        {
            Some(i) => i,
            None => {
                let met = copies.touched.load(Ordering::Relaxed);
                // Carried from an earlier encoder of the command buffer, where no encoder ended
                // with the texture since: the image holds the truth where that one left it, and
                // this encoder's first use of it takes no copy in ([`ShadowCarry`]).
                let carried = (self.carry.entries.iter())
                    .position(|carried| carried.copies.shadow_key == copies.shadow_key)
                    .map(|at| self.carry.entries.swap_remove(at))
                    .filter(|carried| carried.touched == met);
                let (standing, held, stale) = match &carried {
                    Some(carried) => (Standing::Current, carried.held, carried.stale),
                    None => (Standing::Unread, None, None),
                };
                self.entries.push(Entry {
                    copies: Arc::clone(copies),
                    standing,
                    held,
                    dirty: false,
                    area: None,
                    merge: None,
                    two_ways: false,
                    scratch: None,
                    stale,
                    met,
                    carried: carried.is_some(),
                });
                self.entries.len() - 1
            }
        }
    }

    /// Copy back entry `i` (dirty): its image's bytes replace the buffer's, so every other
    /// texture over them is no longer current, or, where the copy back covers a rectangle of
    /// texels laid out as its own, stale there alone (the module's rules). One kept is merged
    /// instead: one kept before a command that wrote its bytes two ways is then no longer current
    /// either, as the buffer holds the other way's writes too, and leaves every other texture
    /// over the bytes no longer current; one kept as it merges back still holds what the
    /// encoder's commands left in them, and its merge changes only words they changed through
    /// it, inside the area they wrote in where they bound it, so it leaves the others stale there
    /// alone, as a copy back of that area does. A copy back covers only the
    /// area the commands wrote in, where they bound it: the image's other texels are what the last
    /// copy in left, which the buffer still holds (a write of its bytes another way leaves the
    /// image stale, and copies in again before the next use).
    fn copy_back(&mut self, i: usize, copies: &mut Vec<ShadowCopy<M>>) {
        let entry = &mut self.entries[i];
        let two_ways = entry.two_ways;
        let step = match &entry.merge {
            Some(merge) => {
                let merge = merge.clone();
                entry.merge = None;
                if two_ways {
                    entry.lose(Lost::TwoWays);
                }
                entry.two_ways = false;
                Step::Merge(merge)
            }
            None => Step::Back,
        };
        let area = match step {
            Step::Back => entry.area,
            _ => None,
        };
        // Where the buffer's bytes may have changed for the other textures over them.
        let changed = match step {
            Step::Merge(_) if two_ways => None,
            _ => entry.area,
        };
        entry.dirty = false;
        entry.area = None;
        let written = Arc::clone(&entry.copies);
        copies.push(ShadowCopy {
            copies: Arc::clone(&written),
            step,
            kept: false,
            area,
            cause: None,
            moves: None,
        });
        for (j, other) in self.entries.iter_mut().enumerate() {
            if j != i && other.copies.shares_truth(&written) {
                other.changed(&written, changed, Lost::OtherShadow);
            }
        }
    }

    /// The copies to record, in order, before a command that uses the shadowed textures
    /// `shadows` and has `uses` (every use of the command, the textures' among them): merges of
    /// the textures kept before the command before, copies back of what the command touches,
    /// copies in of what it uses, a keep of each texture it writes through that merges back, where
    /// none is kept since the last copy back, and a keep of each texture it writes two ways, each
    /// in a scratch `merge` makes (told whether the command writes the bytes two ways, and the
    /// rectangle of texels it is to keep, `None` for every texel). Where the texture merges back,
    /// the scratch keeps the rectangle the command writes in, where
    /// it bounds that and the copies are cut to rectangles ([`ShadowUse::area`]); a later command
    /// that writes outside what the encoder's scratch keeps takes a scratch of the rectangle
    /// bounding both, after a merge of what was kept in the last. Refused, changing nothing, as
    /// `merge` refuses, and for a command that
    /// writes an image two ways, through a view's shadow of it and another alias, which no merge
    /// serves (`Unsupported`, `Feature::ViewFormatReinterpretation`), or a rotated texture's bytes
    /// two ways (`NotImplemented(RotatedTexture)`).
    pub(super) fn before(
        &mut self,
        shadows: &[ShadowUse],
        uses: &[Use<ResourceKey>],
        mut merge: impl FnMut(&ShadowCopies, MergeFor, Option<[u32; 4]>) -> Result<M, Error>,
    ) -> Result<Vec<ShadowCopy<M>>, Error> {
        // A copy back of an image's shadow would replace the other alias's writes, and only a
        // buffer's bytes laid out as the texture's are merged: such a command is refused before
        // any merge is made (a rotated texture's too).
        for (i, shadow) in shadows.iter().enumerate() {
            if let Some(refused) = shadow.copies.unmerged() {
                if two_ways(shadows, i, uses) {
                    return Err(refused);
                }
            }
        }
        let mut kept = Vec::new();
        // The textures the command writes through, its bytes another way not, that merge back:
        // kept in a scratch of the encoder's, made at the first such command.
        let mut merged_kept = Vec::new();
        let mut scratches = Vec::new();
        for (i, shadow) in shadows.iter().enumerate() {
            if two_ways(shadows, i, uses) {
                let made_for = MergeFor {
                    two_ways: true,
                    blocks: false,
                };
                kept.push((
                    Arc::clone(&shadow.copies),
                    merge(&shadow.copies, made_for, None)?,
                ));
            } else if merged_back(shadow) {
                merged_kept.push(Arc::clone(&shadow.copies));
                // The texels the command writes, where it bounds them and the copies are cut so.
                let writes = shadow.area.filter(|_| shadow.copies.cuts_back());
                let scratch = self
                    .entries
                    .iter()
                    .find(|e| e.copies.shadow_key == shadow.copies.shadow_key)
                    .and_then(|e| e.scratch.as_ref());
                // A scratch made for a command of the other kind (one whose merge stores whole
                // blocks, or one whose does not) is merged and kept again in a new one: only such
                // a command's merge stores whole blocks (`shadow_merge.rs`).
                let same = |scratch: &M| scratch.in_blocks() == stores_blocks(shadow, writes);
                let area = match scratch {
                    Some(scratch) if holds(scratch.kept_area(), writes) && same(scratch) => {
                        continue
                    }
                    Some(scratch) => bounding(scratch.kept_area(), writes),
                    None => writes,
                };
                let made_for = MergeFor {
                    two_ways: false,
                    blocks: stores_blocks(shadow, area),
                };
                scratches.push((
                    Arc::clone(&shadow.copies),
                    merge(&shadow.copies, made_for, area)?,
                ));
            }
        }
        // A texture kept in a scratch that does not keep what the command writes: merged before
        // its keep in the new one.
        let mut regrown = Vec::new();
        for (copies, scratch) in scratches {
            let i = self.entry(&copies);
            let entry = &mut self.entries[i];
            if entry.merge.is_some() && !entry.two_ways {
                regrown.push(copies.shadow_key);
            }
            entry.scratch = Some(scratch);
        }
        let mut copies = Vec::new();
        // Merge what the command before wrote two ways; copy back what this one reads or writes
        // another way: through the buffer's bytes, or through another texture over them that is
        // copied in next.
        for i in 0..self.entries.len() {
            if !self.entries[i].dirty {
                continue;
            }
            let entry = &self.entries[i].copies;
            let touched = uses.iter().any(|used| entry.meets(used))
                || shadows.iter().any(|shadow| {
                    shadow.copies.shadow_key != entry.shadow_key
                        && shadow.copies.shares_truth(entry)
                });
            let entry = &self.entries[i];
            let regrows = regrown.contains(&entry.copies.shadow_key);
            // Written since its last copy back with nothing kept (by a command that replaced
            // every texel), and kept below: copied back first, as the merge after the keep takes
            // only the words changed since it, which would leave those writes out.
            let unkept = entry.merge.is_none()
                && (merged_kept.iter()).any(|kept| kept.shadow_key == entry.copies.shadow_key);
            if touched || regrows || unkept || (entry.merge.is_some() && entry.two_ways) {
                self.copy_back(i, &mut copies);
            }
        }
        for shadow in shadows {
            // Made here, or for a keep's scratch above: not copied in yet, so its copy in is the
            // encoder's first.
            let i = self.entry(&shadow.copies);
            let two_ways = kept
                .iter()
                .any(|(copies, _)| copies.shadow_key == shadow.copies.shadow_key);
            let overwrites = overwrites_unkept(shadow) && !two_ways;
            // The texels the command needs as the buffer holds them: those it reads, and those
            // it writes (which a copy back moves), where it bounds both and the copies are cut to
            // a rectangle; every texel where it writes through a shadow whose copies back move
            // every texel, or writes the bytes two ways, as the merge after it compares every
            // texel the keep took; and every texel of a shadow over logged host pages, whose copy
            // in of every texel moves only the rows the log reports changed, and leaves the image
            // holding every texel for the next command buffer's (`dirty_pages.rs`).
            let whole = two_ways || !shadow.copies.cuts() || shadow.copies.logged();
            let needs = match whole {
                true => None,
                false if shadow.writes && !shadow.copies.cuts_back() => None,
                false if shadow.writes => bounding(shadow.loads, shadow.area),
                false => shadow.loads,
            };
            let entry = &self.entries[i];
            let beyond_held =
                entry.standing == Standing::Current && !overwrites && !holds(entry.held, needs);
            // The image holds the buffer's texels only inside its last copy in's rectangle: what
            // the encoder wrote through it goes back first, then the copy in below moves what
            // this command needs.
            if beyond_held && entry.dirty {
                self.copy_back(i, &mut copies);
            }
            let entry = &mut self.entries[i];
            let cause = match entry.standing {
                _ if beyond_held => Some(CopyCause::BeyondHeld),
                Standing::Current => None,
                Standing::Unread => Some(CopyCause::First),
                Standing::Lost(lost) => Some(CopyCause::Changed(lost)),
            };
            // The encoder's first use of a texture its command buffer already held: without the
            // carry the entry would have been `Standing::Unread` and this a copy in of `needs`,
            // so where none is made that copy is what the carry took off the run
            // (`WorkCounts::shadow_texels_first_copy_in_carried`). Where one is made anyway (the
            // command needs texels beyond what the carry held) it moves the same texels as it
            // would have, and nothing is taken off; a copy in of what writes within a rectangle
            // changed since (`CopyCause::Bounded`), which only a carried entry can take here, is
            // counted as its own.
            let carried = entry.carried;
            entry.carried = false;
            let took = carried && cause.is_none() && !overwrites;
            if let Some(cause) = cause {
                if !overwrites {
                    copies.push(ShadowCopy {
                        copies: Arc::clone(&shadow.copies),
                        step: Step::Into,
                        kept: shadow.kept,
                        area: needs,
                        cause: Some(cause),
                        moves: None,
                    });
                }
                entry.standing = Standing::Current;
                entry.held = needs;
                entry.stale = None;
            }
            // The texels writes within a rectangle changed since the copy in: those alone. A
            // command that reads only the texels inside a rectangle and writes none takes those of
            // them inside it alone, and the rest stay changed, for the next command that reads
            // them: the image holds the buffer's texels everywhere else it held them.
            if let Some(stale) = entry.stale {
                let area = match (shadow.writes, needs) {
                    (false, Some(reads)) => {
                        entry.stale = outside(stale, reads);
                        meet(stale, reads)
                    }
                    _ => {
                        entry.stale = None;
                        Some(stale)
                    }
                };
                if let (false, Some(area)) = (overwrites, area) {
                    copies.push(ShadowCopy {
                        copies: Arc::clone(&shadow.copies),
                        step: Step::Into,
                        kept: shadow.kept,
                        area: Some(area),
                        cause: Some(CopyCause::Bounded),
                        moves: None,
                    });
                }
            }
            // The command writes every texel before it reads any.
            if overwrites {
                entry.held = None;
            }
            // What the first copy in of this use moved, or would have moved where the carry took
            // it off: the copy `before` pushes above, or the one it would have pushed.
            let first = (took || (cause == Some(CopyCause::First) && !overwrites)).then(|| {
                let copy: ShadowCopy<M> = ShadowCopy {
                    copies: Arc::clone(&shadow.copies),
                    step: Step::Into,
                    kept: shadow.kept,
                    area: needs,
                    cause: Some(CopyCause::First),
                    moves: None,
                };
                copy.texels_copied_in()
            });
            if let Some(texels) = first {
                let large = shadow.copies.large_surface();
                let stood = shadow.copies.stood.load(Ordering::Relaxed);
                match (took, large) {
                    (true, large) => {
                        self.tally.carried += texels;
                        self.tally.carried_large += u64::from(large) * texels;
                    }
                    // A standing of this command buffer's own that something let a writer past:
                    // `WorkCounts::shadow_texels_first_copied_in_again` counts those.
                    (false, true) if stood == self.carry.id => {}
                    (false, true) if stood != 0 => self.tally.stood_earlier_large += texels,
                    (false, true) => self.tally.unstood_large += texels,
                    (false, false) => {}
                }
            }
        }
        // Kept after its copy in, until the copy back after the command, or after a later one,
        // which is then a merge: a copy back would write the texels no command wrote with what the
        // copy in read, over what another writer may have written since.
        for shadow in merged_kept {
            let i = self.entry(&shadow);
            let entry = &mut self.entries[i];
            let (None, Some(scratch)) = (&entry.merge, &entry.scratch) else {
                continue;
            };
            let scratch = scratch.clone();
            entry.merge = Some(scratch.clone());
            copies.push(ShadowCopy {
                copies: shadow,
                step: Step::Keep(scratch),
                kept: false,
                area: None,
                cause: None,
                moves: None,
            });
        }
        for (shadow, merge) in kept {
            let i = self.entry(&shadow);
            self.entries[i].merge = Some(merge.clone());
            self.entries[i].two_ways = true;
            copies.push(ShadowCopy {
                copies: shadow,
                step: Step::Keep(merge),
                kept: false,
                area: None,
                cause: None,
                moves: None,
            });
        }
        Ok(copies)
    }

    /// Note a recorded command that used `shadows` and had `uses`, and the writes `within`
    /// (whose uses are not among `uses`): a write through a texture leaves its image to copy
    /// back; a write of the bytes another way leaves the images over them stale, there alone
    /// where it is one of `within` and the image's texels are laid out as its texture's (the
    /// module's rules).
    pub(super) fn after(
        &mut self,
        shadows: &[ShadowUse],
        uses: &[Use<ResourceKey>],
        within: &[Within],
    ) {
        for entry in &mut self.entries {
            // A texture's own uses are its image's, never its bytes'.
            let copies = Arc::clone(&entry.copies);
            let writes = |used: &Use<ResourceKey>| !used.write.is_empty() && copies.meets(used);
            if uses.iter().any(writes) {
                entry.lose(Lost::Written);
                continue;
            }
            for part in within.iter().filter(|part| part.uses.iter().any(writes)) {
                match &part.texels {
                    Some(texels) => entry.changed(texels, Some(part.area), Lost::Within),
                    None => entry.lose(Lost::Within),
                }
            }
        }
        for shadow in shadows.iter().filter(|shadow| shadow.writes) {
            let i = self.entry(&shadow.copies);
            // Writes Metal drops are not copied back, and the image no longer holds the bytes.
            let entry = &mut self.entries[i];
            match shadow.copies.drops_writes {
                true => entry.lose(Lost::Dropped),
                false => {
                    entry.area = match entry.dirty {
                        true => bounding(entry.area, shadow.area),
                        false => shadow.area,
                    };
                    entry.dirty = true;
                }
            }
        }
    }

    /// The copies back before work the encoder records apart (an indirect command buffer's
    /// commands, recorded at commit, with copies of their own): every texture written through
    /// since its last. The encoder then forgets every texture, so a later command copies in what
    /// that work may have written.
    pub(super) fn leave(&mut self) -> Vec<ShadowCopy<M>> {
        let copies = self.finish();
        self.entries.clear();
        self.carry.clear();
        copies
    }

    /// Forget every texture, recording no copy: what the encoder recorded so far is left out of
    /// its command buffer's work (`EncoderCore::leave_out_recorded`), so no image holds what a
    /// copy in there moved, and nothing written through one there ran. A later command copies in
    /// again.
    pub(super) fn forget(&mut self) {
        self.entries.clear();
        self.carry.clear();
    }

    /// The copies back at the end of the encoder: every texture written through since its last
    /// (merged where kept).
    pub(super) fn finish(&mut self) -> Vec<ShadowCopy<M>> {
        let mut copies = Vec::new();
        for i in 0..self.entries.len() {
            if self.entries[i].dirty {
                self.copy_back(i, &mut copies);
            }
        }
        copies
    }
}
