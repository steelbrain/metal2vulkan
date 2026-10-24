//! Textures in automatic and placement heaps: an optimal image bound into the heap's own memory at
//! the texture's offset, so a heap's buffers and textures over the same bytes alias as Metal's do;
//! a texture buffer, a buffer of the heap's.
//!
//! - **Size and alignment.** [`Device::heap_texture_size_and_align`] answers **Metal's** answer
//!   wherever the runtime models it (`heap_tiled.rs`: plain and block-compressed formats;
//!   multisample textures; the measured routes below), because that number is what a program sizes
//!   a heap and places its resources by: a consumer that lays a heap out at Metal's offsets and is
//!   told a coarser alignment or a larger size under-allocates and its placements do not fit. An
//!   M4 on macOS 27.0.1 and an M3 Max on macOS 26.7 answer alike (128-byte alignment and the sizes
//!   of Apple's layouts, 16384 bytes for a 64x64 `RGBA8Unorm` texture a shader writes and 16512
//!   for one it only reads, which Metal compresses), so the answer is the device's only where the
//!   model has none. It depends on the device and the descriptor alone, never on what was
//!   allocated.
//!
//!   The texture is held to that answer in one of two ways ([`images_fit`]). Where the images the
//!   runtime makes of the descriptor (the plan `new_texture` makes, with the usage the runtime
//!   adds itself, `with_free_usage`, and a depth twin past it) ask no more bytes and no coarser
//!   alignment, the image is bound at the texture's offset of the heap's memory. Their footprint
//!   is the images' own: a heap's image takes **no** `bufferImageGranularity` padding, because the
//!   heap's own buffer covers its memory from end to end, so no padding can put the image in a
//!   granularity page no buffer lies in and the padding would only widen the answer
//!   (`texture::ImagePlacement::in_heap`; every NVIDIA profile in `validation/profiles` states a
//!   granularity of 1024 bytes where Metal answers 128). Where the driver answers a compressed
//!   image's size short of its compression data (ANV on Intel's Xe-LP GPUs) the footprint still
//!   takes whole 256-byte chunks of the aux map's, so no other resource lies in a chunk the image
//!   writes. Where the images ask more, the texture is a shadow of its bytes with its image in
//!   memory of its own (Metal's tiles where they serve, else rows) and takes Metal's answer all
//!   the same; where no shadow serves, the answer is the larger of the two and the device
//!   announces `heap-texture-larger-than-metal` once (a measured route's answer is refused
//!   instead, `NotImplemented(HeapTextureMemory)`: its bytes are no layout the device's image
//!   holds).
//!   The usage changes Metal's answer only through whether Metal compresses the texture
//!   (`heap_tiled::metal_keeps_uncompressed`, a colour texture of `ShaderWrite`, `PixelFormatView`
//!   or `ShaderAtomic` usage stays uncompressed, a depth or stencil texture of no usage does), and
//!   the runtime plans an image Metal keeps uncompressed so its driver keeps it so (`image_plan`),
//!   so the image is no larger than Metal's answer.
//! - **Multisample textures over host pages**, and in any heap whose memory its optimal image does
//!   not take where the device's linear images cannot hold it (an NVIDIA GPU's `Shared` heap; a
//!   depth texture of a `Shared` heap on MoltenVK, whose optimal depth images take no host-visible
//!   memory), or whose image is larger than Metal's answer where that is not its image's (an
//!   `X32_Stencil8` or `BlockWritesOnly` one), where they were refused `HeapTextureTiling` and
//!   `NotImplemented(HeapTextureMemory)`: a shadow of its samples in the heap's bytes
//!   (`AliasRoute::Tiled`, `heap_tiled.rs`), at Metal's size and alignment. One Metal keeps
//!   uncompressed lies in Metal's tiles there, each pixel's samples side by side, counted tiled with
//!   no notice; one Metal keeps losslessly compressed (`heap_tiled::sample_grid`), whose bytes are
//!   no layout of its samples, in the tiles Metal lays it in uncompressed
//!   (`heap-texture-lossless-layout` per texture). Its samples move between its image and the
//!   heap's bytes by the multisampled blits' interleave copies; where the device runs none (a 16-bit
//!   depth on a device whose multisampled copies go by draws), or its image holds other bytes than
//!   Metal's elements, or its sample count's layout is not measured, it is its optimal image in
//!   memory of its own (`AliasRoute::Lossless`), holding what that memory held, as a heap's texture
//!   holds what its bytes held, the heap's bytes keeping what they held
//!   (`heap-texture-lossless-layout` or `heap-texture-untiled-layout` per texture, as Metal keeps
//!   it compressed or not).
//! - **Where Metal's answer is not its image's** (the consumer's report): Metal answers a 4:2:2
//!   texture its rows of 2-byte texels at a multiple of 16 bytes, the last unpadded, at an
//!   alignment of 16, and an `X32_Stencil8` texture as a `Stencil8` one
//!   (`Device::metal_heap_answer`; an M4), which the runtime answers too.
//!   Where the device's image needs more bytes or a coarser alignment than that answer, and for
//!   every 4:2:2 texture, the texture is a shadow of the heap's bytes in memory of its own
//!   (`AliasRoute::MemoryShadow`): its planes fit the answer (an `X32_Stencil8` texture's stencil
//!   alone, in a `Stencil8` texture's tiles where they serve, as Metal lays it, `heap_tiled.rs`,
//!   else packed; a 4:2:2 texture's rows where Metal lays them, so a buffer over them reads
//!   Metal's bytes; `heap_plane_layout`), so two textures side by side at the answer keep their
//!   texels on every device.
//! - **Two-plane extended-range textures** (`RGB10A8_2P_XR10` and its sRGB twin; the consumer's
//!   report). The runtime answers Metal's size and alignment (`two_plane_heap_answer`): plane 0's
//!   4-byte words, rows at a multiple of 16 bytes, rounded up to 16 KiB, then plane 1's alpha
//!   bytes, rows at a multiple of 16, the last unpadded; a 2D or 1D array's slices at multiples of
//!   128 in each plane; the first level and slice of a mipmapped or 3D texture alone; an alignment
//!   of 16 (128 for a 2D array). Its image, one `R32G32_UINT` texel a texel
//!   (`ExtendedRange::Planes`), is larger than that answer, so it lies in memory of its own, a
//!   shadow of its first level's planes in the heap's bytes, where Metal lays them
//!   (`AliasRoute::Planes`, `heap_two_planes.rs`): a one-level 2D or 1D texture's two planes, an
//!   array's slices of color words (Metal keeps no alpha byte of an array's, and reads its alpha
//!   as 0), a mipmapped texture's first level as a one-level one's. A buffer or texture over them
//!   meets its texels as on Metal, and the device announces `shadowed-texture`. A mipmapped
//!   texture's later levels lie in its image alone, and a 1D array's GPU reads read each texel,
//!   where Metal's paths address those at bytes that differ by path (`heap-two-plane-layout`). A
//!   3D texture's depth slices all lie over its first one's planes on Metal, its reads take its
//!   first texel and its copies its first slice: its image's first depth slice is the shadow, and
//!   its accesses keep to it (`lower::texture::emit::planes_volume`, `two_planes.rs`). One whose
//!   moves pass the device's limits aliases none of the heap's bytes (`AliasRoute::Unaliased`,
//!   `heap-two-plane-texture`). Metal answers a cube or cube array 0 bytes at 16 and makes
//!   neither.
//! - **`allowGPUOptimizedContents`**. A descriptor with it off plans its image as one of
//!   `PixelFormatView` usage (`TextureDescriptor::image_usage`): Metal lays a color texture out
//!   alike either way (16384 bytes for the 64x64 `RGBA8Unorm` texture only read), and MoltenVK
//!   sizes that image as Metal's `PixelFormatView` texture, so its answer is Metal's. Metal also
//!   drops a depth or stencil texture's metadata with it off, which no Vulkan request reaches on
//!   MoltenVK ([`DriverFact::DepthStencilImagesStayGpuOptimized`]): there the texture is a shadow
//!   of Metal's tiles in the heap's bytes (below) and answers Metal's size; where no such shadow
//!   serves it (a multisample texture whose samples the device does not move, a depth twin, a
//!   layout not measured), the answer stays the optimized texture's, larger than Metal's, and the
//!   device announces `optimized-depth-stencil-texture` once.
//! - **The heap's memory.** A heap's buffer lies at a multiple of [`HEAP_BASE_ALIGNMENT`] of its
//!   memory (256 KiB, the largest image alignment measured), of a type optimal images take where
//!   its storage class has one, else of one linear images take ([`HeapTypes::heap_memory`],
//!   `buffer.rs`), so a texture at any multiple of its answered alignment binds at its offset. Up
//!   to half a pooled block it takes whole units of the base of a pooled block; past that, a block
//!   of its own of its own size (`buffer.rs`'s `placed`), so the largest heap stays one allocation
//!   of the device. A heap over host pages starts at offset 0 of its own import
//!   (`HeapProblem::HostMemoryAlignment`), which meets any alignment. A texture whose alignment
//!   exceeds that base, or whose image the heap's memory type does not take, is refused typed
//!   (`RuntimeGap::HeapTextureAlignment`, `HeapTextureMemory`), by the query and the call alike
//!   where the cause is the descriptor.
//! - **Linear images**. A texture whose optimal image does not take the heap's memory (any texture
//!   of a `Shared` heap on the RTX 3090, whose optimal images take no host-visible memory; a depth
//!   texture of one on MoltenVK) is a linear image there, used only where it can do all the optimal
//!   image can for the texture's usages (`linear_plan::linear_lack`) within the device's limits for
//!   linear images; elsewhere the texture is refused as the device's lack,
//!   `Unsupported(Feature::HeapTextureTiling)` naming it (`LinearNeed`). The storage mode and the
//!   descriptor decide the tiling, so the query answers the linear image's size and alignment.
//! - **Placement.** As a heap buffer's: an automatic heap places a texture at the lowest offset
//!   with room, a multiple of its alignment, its `usedSize` the sum of its resources' sizes, and
//!   `make_aliasable` gives its range to the next resource; a placement heap takes any offset that
//!   is a multiple of the alignment with the texture's bytes inside the heap, overlapping other
//!   resources or not (Metal: an offset not a multiple of the alignment is stopped by validation
//!   and answers nil without it, and so does one past the heap's end:
//!   the runtime answers `Nil`). The texture's storage and CPU cache modes are the heap's (another
//!   is `Invalid`, as Metal ends the process), and its hazard tracking is the heap's buffer's, so
//!   Metal's tracking at the granularity of the heap orders a buffer's write before an
//!   overlapping texture's read. A new texture holds what its bytes held: the image is moved to
//!   the `GENERAL` layout and not cleared.
//! - **Depth twins**. A `Depth32Float` texture the device holds with a depth twin takes both images
//!   of the heap: the image at the texture's offset and the twin past it, at the next multiple of
//!   the twin's alignment ([`heap_extent`]), so the answered size covers both and the texture's
//!   bytes are its range alone, as Metal's placement and aliasing rules need (a Metal heap probe on
//!   the M3 Max and an M4: two `Depth32Float` textures side by side keep their texels, and a buffer
//!   filled over the first changes every texel of it and none of the second's). Each image takes
//!   the heap's memory or its linear tiling as above. Both are the heap buffer's resource to hazard
//!   tracking, and `depth-twin` is announced as for `new_texture`.
//! - **Depth and stencil images apart**. On a driver that lays depth and stencil images apart from
//!   the memory they are bound to ([`DriverFact::DepthStencilImagesLieApart`]: MoltenVK, measured),
//!   a heap's single-sample depth or stencil texture of a format whose layout is measured
//!   (`Depth16Unorm`, `Depth32Float`, `Stencil8`, and `Depth32Float_Stencil8` in two planes) is a
//!   shadow of its elements in Metal's tiles in the heap's bytes (`AliasRoute::Tiled`,
//!   `heap_tiled.rs`), at Metal's size and alignment: a Metal probe on the M3 Max under macOS
//!   27.0.1 found every texel of such textures with `allowGPUOptimizedContents` off where the
//!   tiles place it (37 x 5 to 300 x 200), and Metal's
//!   compressed bytes with it on no layout of the texels, which the shadow holds in its
//!   uncompressed tiles, announced `heap-texture-lossless-layout` per texture. So is one whose
//!   image the heap's memory takes none of, as below, and one whose image the driver keeps
//!   GPU-optimized where Metal's is not, a multisample one too (its samples, above). Any other such
//!   image, its own or its twin, is a shadow of the heap's bytes in rows (`AliasRoute::Shadow`), or,
//!   where no shadow serves (a depth twin), aliases none of the heap's other resources, and the
//!   device announces `heap-depth-stencil-apart` once when it makes one.
//! - **Heaps over the device's memory**. A texture whose image is bound in such a heap's memory
//!   lies there in the driver's layout. On a driver whose single-sample colour images lie as Metal
//!   lays out a heap texture of the image's usage
//!   ([`DriverFact::ColorImagesLieAsMetalHeapTextures`]: MoltenVK 1.4.2 on the M3 Max, measured),
//!   and whose multisample colour images lie as Metal lays out one it keeps uncompressed
//!   ([`DriverFact::MultisampleColorImagesLieAsMetalHeapTextures`]: likewise), such a texture
//!   Metal tiles is Metal's bytes there. Every other texture whose bytes there are
//!   not Metal's layout is counted (`Device::degraded_objects`), each kind announced at its first
//!   texture, with its ID: its image in the driver's layout on any other driver, or of a depth or
//!   stencil texture, `heap-texture-untiled-layout`; its image a shadow of the bytes in rows,
//!   alike; a multisample texture, alike; one Metal may keep losslessly compressed,
//!   `heap-texture-lossless-layout` (the runtime's image need not be compressed as Metal's); one
//!   whose layout is not measured, by its kind (`device_heap_layout`). A texture apart from the
//!   heap's bytes, and a two-plane one, announce their own routes.
//! - **Compressed images**. On a driver whose compressed images may keep their texels apart from
//!   the memory they are bound to ([`DriverFact::CompressedImagesHideAliasedWrites`]: ANV on
//!   Intel's GPUs, measured), a heap's colour texture whose optimal image takes the heap's memory
//!   and that Metal keeps losslessly compressed too (`heap_tiled::metal_compresses_layout`) takes
//!   that image, the driver's compression and fast clears kept, under the device's
//!   `HeapCompressionPolicy::Compressed` (the default), announced once as a near-equivalent
//!   (`heap-texture-compressed`): Metal's texture shows no aliased write as written either (an M3
//!   Max read opaque magenta in every texel of drawn content a buffer's fill over the texture's
//!   bytes covered). Any other such texture, and every one under
//!   `HeapCompressionPolicy::Uncompressed`, takes, in order: that image with host-transfer usage,
//!   which the driver keeps uncompressed in its memory
//!   ([`DriverFact::HostTransferImagesKeepTheirBytes`]), where the device enabled `hostImageCopy`
//!   (the runtime on a device it creates, `device/create.rs`; the consumer on one it adopts that
//!   says so, `device/adopt.rs`) and the format and shape take the usage; else a linear image,
//!   where one serves as above; else the optimal image. The device announces the first two once
//!   each as costs (`heap-texture-host-transfer`, `heap-texture-linear`) and the last once as a
//!   near-equivalent (`heap-texture-hides-aliased-writes`). The first two keep Metal's visibility
//!   contract, not a byte layout: a write through a buffer or texture over the texture's bytes
//!   changes every texel it covers, and the texture's writes show in those bytes, in the device's
//!   layout for the image, as Metal's are in its GPU's. Depth and stencil images and depth twins
//!   keep their optimal images: a buffer's write over ANV's changed every texel (the heap depth
//!   lane).
//! - **Texture buffers**. A heap's texture buffer is a buffer of the heap's at the texture's
//!   offset, its row of texels long, with the texture buffer over its bytes as
//!   `Buffer::new_texture` makes one: Metal's answers, measured (a Metal heap probe, an M3 Max and
//!   an M4 alike, plain and validated): the size is the row exactly and the alignment 16 bytes, for
//!   every format and storage mode; the texels are the heap's bytes from the offset, in order, so a
//!   buffer over them and the texture alias byte for byte; an automatic heap answers `heapOffset` 0
//!   and counts the row in `usedSize`, and `makeAliasable` gives the range back. The texture
//!   answers `heap`, `heap_offset`, `make_aliasable` and `is_aliasable` through the buffer, which
//!   is no resource of the program's (Metal's texture answers `buffer` nil).

use super::first_move::FirstMove;
use super::heap_tiled::{
    metal_compresses_layout, metal_heap_layout, modelled_size, shadow_tiles, HeapLayout,
    HeapTiledCopies, TiledTexture, Untiled,
};
use super::heap_two_planes::{HeapPlaneCopies, TwoPlaneHeapLayout};
use super::stand_in::StandIn;
use super::{
    copy_units, image_aspects, image_error, pages_hold, DetachedImage, IdUse, Image, ImageObjects,
    ImagePlacement, ImageSwap, KeptResolves, Owner, Raw, Remade, Storage, Texture,
    TextureDescriptor, ViewRecord,
};
use crate::address::ObjectsToken;
use crate::buffer::{
    buffer_class, usage as buffer_usage, Buffer, CallerMemory, HostPages, MemoryHold, Modes,
};
use crate::buffer_texture::{texture_buffer_bytes, texture_buffer_over, TexelsHome};
use crate::chroma_pairs;
use crate::copy_plan::{CopyProblem, HostLayout, ImageRegion};
use crate::degradation::Degradation;
use crate::device::lock;
use crate::error::{internal, invalid, nil, vulkan, Error, MemoryKind, Nil, Problem, RuntimeGap};
use crate::heap::{Heap, HeapProblem, HeapType, SizeAndAlign, WeakHeap};
use crate::image_plan::{check_image_limits, plan_images, ImagePlan};
use crate::key_hash::KeyMap;
use crate::linear_plan::{check_linear_limits, linear_lack, LinearLack, LINEAR_ALIGNMENT};
use crate::memory::{Memory, MemoryClass, Placement};
use crate::metal_level::MetalLevel;
use crate::release;
use crate::resource::{Identity, ResourceKey};
use crate::shadow::{Backing, ShadowBuffer, ShadowCopies, ShadowSource, ShadowTarget};
use crate::shadow_merge::ScratchSpares;
use crate::texture_descriptor::{effective_usage, Creator, TextureLayout, TextureProblem};
use crate::tracker::UseRange;
use crate::two_planes;
use crate::values::{ResourceOptions, StorageMode};
use crate::Device;
use ash::vk;
use metal2vulkan::device::{DriverFact, HeapCompressionPolicy};
use metal2vulkan::error::{Feature, LinearNeed, Unsupported};
use metal2vulkan::metal::format::{is_depth_stencil, ycbcr_format};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::vk::{
    Format, FormatFeatureFlags2, ImageAspectFlags, ImageUsageFlags, SampleCountFlags,
};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// The alignment of a heap's memory: a heap texture whose answered alignment exceeds it is
/// refused (`RuntimeGap::HeapTextureAlignment`, [`check_heap_base`]). It is the largest image
/// alignment measured: RADV on an RX 7900 XTX (Mesa 26.0.8) asks 128 KiB of a `D16`, `D32F` or
/// `D32FS8` image of 8 x 8 texels or more and 256 KiB of a multisample depth, stencil or color
/// image from about 128 x 128 texels, where the Radeon 780M under the same driver asks at most
/// 64 KiB of any.
pub(crate) const HEAP_BASE_ALIGNMENT: u64 = 256 << 10;

/// Where a heap texture's image lies: its heap, and its offset and size there. The range has
/// this one owner from its allocation on, the texture's image once it is made. The last texture
/// object over the image gives an automatic heap's range back (`give_back`), as Metal's texture
/// does at its release, unless `makeAliasable` gave it back already; the place's
/// drop gives it back where no texture object was made over it.
pub(crate) struct HeapPlace {
    /// Its heap, held weakly: the texture objects hold it (`TextureShared::heap`), as Metal's
    /// texture retains its heap, so a command buffer that keeps the image keeps none of the heap
    /// but its memory (`memory`), and the heap goes at the program's release, as Metal frees it.
    pub(crate) heap: WeakHeap,
    pub(crate) offset: u64,
    pub(crate) size: u64,
    /// `makeAliasable` was called.
    pub(crate) aliasable: AtomicBool,
    /// The heap's memory an image bound to it holds ([`MemoryHold`]): `None` for an image in
    /// memory of its own (a shadow, an unaliased one), and while the heap's bytes are detached. A
    /// rebind's swap holds the new pages' instead.
    pub(crate) memory: Mutex<Option<MemoryHold>>,
    /// The range went back to the heap.
    given_back: AtomicBool,
}

impl HeapPlace {
    /// The range of `size` bytes at `offset` of `heap`, aliasable from the start where
    /// `aliasable`.
    fn new(heap: &Heap, offset: u64, size: u64, aliasable: bool) -> HeapPlace {
        HeapPlace {
            heap: heap.downgrade(),
            offset,
            size,
            aliasable: AtomicBool::new(aliasable),
            memory: Mutex::new(None),
            given_back: AtomicBool::new(false),
        }
    }

    /// Give the range back to `heap`, its heap, once: as the last texture object over the image
    /// goes.
    pub(crate) fn give_back(&self, heap: &Heap) {
        if !self.given_back.swap(true, Ordering::AcqRel) {
            heap.release(self.offset, self.size, &self.aliasable);
        }
    }
}

impl Drop for HeapPlace {
    fn drop(&mut self) {
        // A range no texture object gave back goes back as the image goes (a making that failed),
        // while its heap lives.
        if self.given_back.load(Ordering::Acquire) {
            return;
        }
        if let Some(heap) = self.heap.upgrade() {
            heap.release(self.offset, self.size, &self.aliasable);
        }
    }
}

/// What a heap's depth or stencil texture held as a shadow of the heap's bytes
/// ([`AliasRoute::Shadow`]) keeps: its image's own memory, its range of the heap, and the copies
/// between them (`shadow.rs`).
pub(crate) struct HeapShadow {
    pub(crate) placement: Placement,
    pub(crate) place: HeapPlace,
    pub(crate) copies: Arc<ShadowCopies>,
}

/// What a heap's two-plane extended-range texture held apart from the heap's bytes
/// ([`AliasRoute::Unaliased`]) keeps: its image's own memory, and its range of the heap.
pub(crate) struct HeapUnaliased {
    pub(crate) placement: Placement,
    pub(crate) place: HeapPlace,
}

impl HeapShadow {
    /// The plane of the heap's bytes that holds `region` of the shadowed image of `plan`, where
    /// one does (not a tiled texture's, whose elements `pieces` places itself): a two-plane
    /// texture's color words of its first level (`heap_two_planes.rs`), or the plane of the
    /// region's aspect and level ([`heap_plane_layout`]).
    fn plane(&self, plan: &ImagePlan, region: &ImageRegion) -> Option<HeapPlane> {
        if let Some(ShadowTarget::Planes { planes, .. }) = self.copies.target() {
            let plane = planes.layout().color_plane();
            let held = plane.aspect == region.aspect && plane.level == region.level;
            return held.then_some(plane);
        }
        let (planes, _) = heap_plane_layout(
            plan.format,
            plan.pixel_format,
            plan.extent,
            plan.levels,
            plan.layers,
        );
        planes
            .into_iter()
            .find(|plane| plane.aspect == region.aspect && plane.level == region.level)
    }

    /// Whether a CPU copy of `region` of the shadowed image of `plan` moves the heap's bytes: a
    /// region of a tiled texture, or one a plane of the heap's bytes holds
    /// ([`plane`](Self::plane)); not one of a two-plane texture's levels past its first, which lie
    /// in its image alone.
    pub(super) fn holds(&self, plan: &ImagePlan, region: &ImageRegion) -> bool {
        matches!(self.copies.target(), Some(ShadowTarget::Tiled { .. }))
            || self.plane(plan, region).is_some()
    }

    /// The heap's mapped bytes, from its first, which a CPU copy reaches: the texture copying
    /// keeps the heap (the place holds it weakly).
    fn contents(&self, call: &'static str) -> Result<NonNull<u8>, Error> {
        let heap = self.place.heap.upgrade();
        (heap.as_ref())
            .and_then(Heap::bytes)
            .and_then(Buffer::contents)
            .ok_or_else(|| invalid(call, Problem::Copy(CopyProblem::CpuAccessPrivate)))
    }

    /// Each piece of a CPU copy of `region` of the shadowed image of `plan`, laid out as `host`,
    /// to `each`: its offset in the heap's mapped bytes, its offset in the packed layout, and
    /// its bytes. The heap's bytes are the texels' truth: each encoder copies them into the
    /// image at its first use and back at its end (`shadow.rs`), so `replaceRegion` and
    /// `getBytes` reach them, laid out as [`heap_plane_layout`] says, a row of blocks a piece;
    /// or, for a texture Metal tiles there (`ShadowTarget::Tiled`), as its tiles lay them out
    /// (`TiledLayout::element`, the part of the region's aspect), an element a piece.
    fn pieces(
        &self,
        plan: &ImagePlan,
        region: &ImageRegion,
        host: &HostLayout,
        mut each: impl FnMut(usize, usize, usize),
    ) -> Result<(), Error> {
        let [x, y, z] = region.origin.map(u64::from);
        if let Some(ShadowTarget::Tiled { tiles, .. }) = self.copies.target() {
            let part = (tiles.tiles.part(region.aspect))
                .ok_or_else(|| internal("a tiled heap texture has no part of a copy's aspect"))?;
            let tiled = &part.layout;
            let level = region.level as usize;
            let depth = tiled
                .levels
                .get(level)
                .ok_or_else(|| internal("a tiled heap texture has no level of a copy's"))?
                .depth;
            let unit = tiled.unit;
            let across = host.row_bytes / unit.max(1);
            let base = u64::from(region.layer) * depth + z;
            for image in 0..host.images {
                for row in 0..host.rows {
                    for column in 0..across {
                        let at = part.at + tiled.element(base + image, level, x + column, y + row);
                        let packed = (image * host.rows + row) * host.row_bytes + column * unit;
                        each(
                            (self.place.offset + at) as usize,
                            packed as usize,
                            unit as usize,
                        );
                    }
                }
            }
            return Ok(());
        }
        let plane = self
            .plane(plan, region)
            .ok_or_else(|| internal("a heap shadow has no plane of a copy's aspect and level"))?;
        let [bw, bh] = plane.block.map(u64::from);
        let depth = u64::from(plane.extent[2]);
        let base = u64::from(region.layer) * depth + z;
        for image in 0..host.images {
            for row in 0..host.rows {
                let at = self.place.offset + plane.row(base + image, y / bh + row, x / bw);
                let packed = (image * host.rows + row) * host.row_bytes;
                each(at as usize, packed as usize, host.row_bytes as usize);
            }
        }
        Ok(())
    }

    /// `replaceRegion` of `region` of the shadowed image of `plan` from `bytes` laid out as
    /// `host` (checked by the texture): into the heap's bytes (`HeapShadow::pieces`).
    pub(super) fn write(
        &self,
        call: &'static str,
        plan: &ImagePlan,
        region: &ImageRegion,
        host: &HostLayout,
        bytes: &[u8],
    ) -> Result<(), Error> {
        let mut packed = vec![0u8; host.packed_len() as usize];
        host.pack(bytes, &mut packed);
        let contents = self.contents(call)?;
        self.pieces(plan, region, host, |at, from, n| {
            // SAFETY: every piece lies in the texture's range of the heap's mapped bytes (its
            // planes or tiles fit the range: `host_heap_plan`, `heap_image`); a write racing the
            // GPU's use of them is the program's, as through `contents`.
            unsafe {
                std::ptr::copy_nonoverlapping(packed[from..].as_ptr(), contents.as_ptr().add(at), n)
            };
        })
    }

    /// `getBytes` of `region` of the shadowed image of `plan` into `bytes` laid out as `host`
    /// (checked by the texture): from the heap's bytes (`HeapShadow::pieces`).
    pub(super) fn read(
        &self,
        call: &'static str,
        plan: &ImagePlan,
        region: &ImageRegion,
        host: &HostLayout,
        bytes: &mut [u8],
    ) -> Result<(), Error> {
        let mut packed = vec![0u8; host.packed_len() as usize];
        let contents = self.contents(call)?;
        self.pieces(plan, region, host, |at, to, n| {
            // SAFETY: as in `write`.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    contents.as_ptr().add(at),
                    packed[to..].as_mut_ptr(),
                    n,
                )
            };
        })?;
        host.unpack(&packed, bytes);
        Ok(())
    }
}

/// How a heap texture's image shows writes through the heap's other resources over its bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AliasRoute {
    /// As an image bound to memory: on a driver whose compressed images hide no write through
    /// an alias, a depth or stencil image, a depth twin, or a linear image for the heap's memory.
    Direct,
    /// An optimal colour image of host-transfer usage, which the driver keeps uncompressed
    /// (`heap-texture-host-transfer`).
    HostTransfer,
    /// A linear colour image, which the driver does not compress (`heap-texture-linear`).
    Linear,
    /// An optimal colour image the driver may compress, a near-equivalent
    /// (`heap-texture-hides-aliased-writes`).
    Hidden,
    /// An optimal colour image the driver may compress, of a texture Metal keeps losslessly
    /// compressed too, under `HeapCompressionPolicy::Compressed` (`heap-texture-compressed`).
    Compressed,
    /// A depth or stencil image the driver lays apart from its memory, in memory of its own,
    /// whose texels the encoders copy from the heap's bytes and back: a shadow of them
    /// (`shadowed-texture`).
    Shadow,
    /// A depth or stencil image the driver lays apart from its memory that no shadow serves (a
    /// multisampled image, which Vulkan copies to no buffer, or texels that do not fit the
    /// texture's bytes; a depth twin), a near-equivalent (`heap-depth-stencil-apart`).
    Apart,
    /// A texture whose image the device binds to none of the heap's memory: its optimal image
    /// takes no memory type of the heap's and no linear image serves (a `Shared` heap's depth
    /// texture on MoltenVK or the RTX 3090), or the heap is over the caller's host memory and
    /// no linear image over that memory serves; in memory of its own, whose texels the
    /// encoders copy from the heap's bytes and back, as [`AliasRoute::Shadow`]'s
    /// (`shadowed-texture`).
    MemoryShadow,
    /// A two-plane extended-range texture's image, larger than Metal's answer
    /// (`two_plane_heap_answer`), in memory of its own, that no shadow of its planes serves (its
    /// moves past the device's limits): it aliases none of the heap's bytes, and no copy moves
    /// texels between them (`heap-two-plane-texture`).
    Unaliased,
    /// A two-plane extended-range texture's image, in memory of its own, whose first level's
    /// texels the encoders move from its planes in the heap's bytes, as Metal lays them, and back
    /// (`heap_two_planes.rs`): a shadow of them (`shadowed-texture`; `heap-two-plane-layout` for
    /// a mipmapped one, whose later levels lie in the image alone, and a 1D array).
    Planes,
    /// A texture Metal tiles in the heap's bytes (`heap_tiled.rs`) in a heap over host pages, or
    /// in any heap one that no image bound there serves as Metal's (a depth or stencil image the
    /// driver lays apart from its memory or keeps GPU-optimized, one the heap's memory takes no
    /// image of, a multisample one held otherwise in memory of its own): its optimal image in
    /// memory of its own, whose elements (a multisample texture's samples) the encoders move from
    /// Metal's tiles in the heap's bytes and back (a texture Metal keeps losslessly compressed,
    /// its uncompressed tiles), a shadow of them (`shadowed-texture`).
    Tiled,
    /// A multisample texture's optimal image in memory of its own, apart from the heap's bytes,
    /// which keep what they held, where no shadow of its samples serves (`heap_tiled.rs`: the
    /// device moves none to and from a buffer, its image holds other bytes than Metal's elements,
    /// or its layout is not measured): in a heap over host pages, and in any heap one whose
    /// optimal image takes none of the heap's memory where no linear image serves, or whose image
    /// is larger than Metal's answer where that is not its image's (`Device::metal_heap_answer`).
    /// It takes Metal's answer of the heap where that is measured. Announced per texture by its
    /// layout (`heap-texture-lossless-layout` where Metal compresses it,
    /// `heap-texture-untiled-layout` where Metal lays its samples out in the heap's bytes), not by
    /// its route.
    Lossless,
}

/// Whose image of a heap texture an image is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mine {
    /// The texture's own; a shadow of the heap's bytes may serve it where the texture is held
    /// with no depth twin (`shadows`), and Metal keeps it losslessly compressed where
    /// `compressed` (`heap_tiled::metal_compresses_layout`).
    Texture { shadows: bool, compressed: bool },
    /// The texture's depth twin.
    Twin,
}

/// One image a heap texture takes of a heap: the texture's own, or its depth twin.
struct HeapImage {
    plan: ImagePlan,
    /// Linear where the heap's memory is not memory the image's optimal tiling takes
    /// ([`takes_heap`]), or where the alias route takes it.
    tiling: vk::ImageTiling,
    requirements: vk::MemoryRequirements,
    route: AliasRoute,
}

impl HeapImage {
    /// The refusal of an image of this tiling that does not take the heap's memory type.
    fn memory_refusal(&self) -> Error {
        if self.tiling == vk::ImageTiling::LINEAR {
            tiling_refusal(LinearNeed::MemoryType)
        } else {
            Error::NotImplemented(RuntimeGap::HeapTextureMemory)
        }
    }
}

/// What a heap texture's images share: the heap's memory, and the texture's options,
/// `allowGPUOptimizedContents` and hazard identity.
struct HeapBinding {
    memory: vk::DeviceMemory,
    options: ResourceOptions,
    allow_gpu_optimized_contents: bool,
    identity: Identity,
    /// The memory is host memory the heap imported (`Device::new_heap_with_bytes_no_copy`): an
    /// image bound to it is made for host allocations.
    host: bool,
    /// The host pages the heap's bytes lie in now, where `host`: a submission that takes the move
    /// of an image bound to them holds them (`texture/first_move.rs`).
    pages: Option<Arc<HostPages>>,
}

/// What a texture of a descriptor takes of a heap.
struct HeapPlan {
    layout: TextureLayout,
    image: HeapImage,
    /// The texture's depth twin, at this offset from the texture's: past the image's bytes, at a
    /// multiple of its own alignment, inside the texture's size.
    twin: Option<(HeapImage, u64)>,
    size_and_align: SizeAndAlign,
    /// Metal's layout of the texture's bytes as the runtime holds it (`heap_tiled.rs`): over host
    /// pages, tiled or announced untiled.
    metal_layout: HeapLayout,
}

/// The memory types a heap's buffer and the images in it take, worked out once
/// ([`Device::heap_types`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct HeapTypes {
    /// A heap's buffer's: every buffer the runtime makes has one usage, and Vulkan gives buffers
    /// of one usage and flags one set of memory types.
    pub(crate) buffer: u32,
    /// Optimal color images' (Vulkan gives every color image of one tiling and flags one set).
    pub(crate) color: u32,
    /// Optimal depth images' (Vulkan lets depth/stencil images' differ from color images').
    pub(crate) depth: u32,
    /// Linear color images', none where the device makes no linear `RGBA8Unorm` image.
    pub(crate) linear: u32,
}

impl HeapTypes {
    /// The memory types a heap of `class` whose buffer takes the types `own` lies in, in
    /// `memory`: the first set holding a type of the class of those both optimal color and depth
    /// images take, those optimal color images take (MoltenVK's `Shared` heaps: its depth images
    /// take no shared memory), and those linear color images take (the RTX 3090's `Shared` heaps:
    /// its optimal images take no host-visible memory); its own types where none does, whose
    /// textures are refused.
    pub(crate) fn heap_memory(self, memory: &Memory, own: u32, class: MemoryClass) -> u32 {
        [
            own & self.color & self.depth,
            own & self.color,
            own & self.linear,
        ]
        .into_iter()
        .find(|&types| memory.has_type(types, class))
        .unwrap_or(own)
    }
}

/// Whether an image taking the memory types `image` binds into a heap of `class` lying in the
/// types `heap`, whichever of them of the class it was given: it takes every one.
pub(crate) fn takes_heap(memory: &Memory, heap: u32, image: u32, class: MemoryClass) -> bool {
    let types = memory.class_types(heap, class);
    types != 0 && types & !image == 0
}

/// The refusal of a heap texture on a device whose linear images lack `missing`.
fn tiling_refusal(missing: LinearNeed) -> Error {
    Error::Unsupported(Unsupported::not_representable(Feature::HeapTextureTiling {
        missing,
    }))
}

/// Whether `error` refuses a heap texture's linear image as the device's lack: what its linear
/// images cannot do for the texture, or memory of the heap's they do not take.
fn linear_refusal(error: &Error) -> bool {
    matches!(
        error,
        Error::Unsupported(unsupported)
            if matches!(unsupported.feature, Feature::HeapTextureTiling { .. })
    )
}

/// Whether `plan` is a colour image: neither a depth nor a stencil aspect.
fn colour(plan: &ImagePlan) -> bool {
    !plan
        .format
        .block()
        .is_some_and(|block| block.depth.is_some() || block.stencil.is_some())
}

/// Whether a shadow of a heap's bytes serves an image of `plan` in a texture of `size` bytes: one
/// sample (Vulkan copies no multisampled image to a buffer), copy usage, and its planes
/// ([`heap_planes`]) inside the texture's bytes.
fn shadowable(plan: &ImagePlan, size: u64) -> bool {
    let (_, length) = heap_planes(
        plan.format,
        plan.pixel_format,
        plan.extent,
        plan.levels,
        plan.layers,
    );
    let copies = ImageUsageFlags::TRANSFER_SRC | ImageUsageFlags::TRANSFER_DST;
    plan.samples == SampleCountFlags::COUNT_1 && length <= size && plan.usage.contains(copies)
}

/// One plane of a heap shadow's truth (no-copy heaps): one aspect of one level, its images (the
/// level's layers, or a 3D level's depth slices) one after another, each its rows of blocks `pitch`
/// bytes apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HeapPlane {
    pub(crate) aspect: metal2vulkan::vk::ImageAspectFlags,
    pub(crate) level: u32,
    /// Its first byte, from the texture's.
    pub(crate) offset: u64,
    /// The level's extent in texels.
    pub(crate) extent: [u32; 3],
    /// Its blocks across and down one image.
    pub(crate) blocks: [u64; 2],
    /// Its images.
    pub(crate) images: u64,
    /// The bytes of one block.
    pub(crate) unit: u64,
    /// The texels a block spans across and down.
    pub(crate) block: [u32; 2],
    /// The bytes from one row of blocks to the next: a row's, tightly packed, but for a 4:2:2
    /// texture's, which Metal lays at a multiple of 16 bytes (`FOUR_TWO_TWO_ALIGNMENT`).
    pub(crate) pitch: u64,
    /// The bytes from one image to the next: its rows', but for a two-plane array's, whose
    /// slices Metal lays at a multiple of 128 bytes (`heap_two_planes.rs`).
    pub(crate) image_pitch: u64,
}

impl HeapPlane {
    /// The bytes from one image to the next.
    fn image_bytes(&self) -> u64 {
        self.image_pitch
    }

    /// The bytes the plane spans: its images, the last one's last row unpadded.
    fn span(&self) -> u64 {
        self.images.saturating_sub(1) * self.image_pitch
            + self.blocks[1].saturating_sub(1) * self.pitch
            + self.blocks[0] * self.unit
    }

    /// Where row `row` of blocks of image `image` lies, from block column `column`, counted
    /// from the texture's first byte.
    pub(crate) fn row(&self, image: u64, row: u64, column: u64) -> u64 {
        self.offset + image * self.image_bytes() + row * self.pitch + column * self.unit
    }
}

/// The planes of a heap texture whose image is of `format` (presenting `pf`), level-0 `extent`
/// (its depth a 3D image's), `levels` and `layers`, when it is a shadow of the heap's bytes (and a
/// no-copy heap's textures that no image over its memory serves): for each of the format's aspects
/// (its depth, then its stencil, or its colour; `copy_units`), each level in order, its images'
/// blocks tightly packed, each plane at a multiple of 4 bytes and of its block's bytes, the buffer
/// offsets Vulkan copies the aspect at. A texture made in `X32_Stencil8` has its stencil's planes
/// alone: its image's depth holds no texel of it; a 4:2:2 texture's rows lie at a multiple of 16
/// bytes, the last unpadded, as Metal lays out a heap's (an M4: a buffer over one
/// read each row there, the bytes between untouched). The planes, and the bytes they span.
pub(crate) fn heap_plane_layout(
    format: Format,
    pf: PixelFormat,
    extent: [u32; 3],
    levels: u32,
    layers: u32,
) -> (Vec<HeapPlane>, u64) {
    let mut planes = Vec::new();
    let mut end = 0u64;
    for (aspect, bytes, [bw, bh]) in copy_units(format, pf) {
        if pf == PixelFormat::X32Stencil8 && aspect != ImageAspectFlags::STENCIL {
            continue;
        }
        let unit = u64::from(bytes);
        // The least common multiple of 4 and the block's bytes.
        let align = 4 * unit / gcd(4, unit.max(1));
        for level in 0..levels {
            let level_extent = extent.map(|axis| (axis >> level.min(31)).max(1));
            let [width, height, depth] = level_extent;
            let across = u64::from(width.div_ceil(bw.max(1)));
            let packed = across * unit;
            let pitch = if chroma_pairs::is_four_two_two(pf) {
                packed.next_multiple_of(FOUR_TWO_TWO_ALIGNMENT)
            } else {
                packed
            };
            let down = u64::from(height.div_ceil(bh.max(1)));
            let plane = HeapPlane {
                aspect,
                level,
                offset: end.next_multiple_of(align.max(1)),
                extent: level_extent,
                blocks: [across, down],
                images: u64::from(depth) * u64::from(layers),
                unit,
                block: [bw.max(1), bh.max(1)],
                pitch,
                image_pitch: pitch * down,
            };
            end = plane.offset + plane.span();
            planes.push(plane);
        }
    }
    (planes, end)
}

/// The planes of a multisample heap texture's tight buffer, whose image is of `format`
/// (presenting `pf`), of `extent` and `layers` at `samples` (`texture/heap_tiled.rs`): its one
/// level's planes as [`heap_plane_layout`] lays them, each pixel's samples one element of all
/// their bytes, sample `s` at `s·B`, as Metal's blits lay a multisample texture out in a buffer
/// and its tiles hold its pixels; and the bytes they span.
pub(crate) fn sample_plane_layout(
    format: Format,
    pf: PixelFormat,
    extent: [u32; 3],
    layers: u32,
    samples: u32,
) -> (Vec<HeapPlane>, u64) {
    let (planes, end) = heap_plane_layout(format, pf, extent, 1, layers);
    let n = u64::from(samples.max(1));
    let planes = planes
        .into_iter()
        .map(|plane| HeapPlane {
            offset: plane.offset * n,
            unit: plane.unit * n,
            pitch: plane.pitch * n,
            image_pitch: plane.image_pitch * n,
            ..plane
        })
        .collect();
    (planes, end * n)
}

/// The planes of the tight buffer of a tiled heap texture's image of `plan`
/// (`texture/heap_tiled.rs`): a multisample image's samples ([`sample_plane_layout`]), else
/// [`heap_plane_layout`]'s.
fn tight_planes(plan: &ImagePlan) -> (Vec<HeapPlane>, u64) {
    let (format, pf, extent) = (plan.format, plan.pixel_format, plan.extent);
    match plan.samples.as_raw() {
        1 => heap_plane_layout(format, pf, extent, plan.levels, plan.layers),
        samples => sample_plane_layout(format, pf, extent, plan.layers, samples),
    }
}

/// The greatest common divisor of `a` and `b`.
fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// The copies between a heap shadow's truth and its image ([`heap_plane_layout`]): one region per
/// plane, its buffer offset counted from the texture's first byte, and the bytes they span.
pub(crate) fn heap_planes(
    format: Format,
    pf: PixelFormat,
    extent: [u32; 3],
    levels: u32,
    layers: u32,
) -> (Vec<vk::BufferImageCopy>, u64) {
    let (planes, end) = heap_plane_layout(format, pf, extent, levels, layers);
    let regions = planes
        .iter()
        .map(|plane| {
            let [width, height, depth] = plane.extent;
            // A row longer than its blocks is given in texels (Vulkan's `bufferRowLength`).
            let row_length = if plane.pitch == plane.blocks[0] * plane.unit {
                0
            } else {
                (plane.pitch / plane.unit) as u32 * plane.block[0]
            };
            vk::BufferImageCopy {
                buffer_offset: plane.offset,
                buffer_row_length: row_length,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: plane.aspect.into(),
                    mip_level: plane.level,
                    base_array_layer: 0,
                    layer_count: layers,
                },
                image_offset: vk::Offset3D::default(),
                image_extent: vk::Extent3D {
                    width,
                    height,
                    depth,
                },
            }
        })
        .collect();
    (regions, end)
}

/// Whether `regions` copy every aspect of an image of `format`: every aspect but an
/// `X32_Stencil8` texture's depth, whose planes are its stencil's alone ([`heap_plane_layout`]).
fn copies_every_aspect(regions: &[vk::BufferImageCopy], format: Format) -> bool {
    let copied = regions
        .iter()
        .fold(vk::ImageAspectFlags::empty(), |aspects, region| {
            aspects | region.image_subresource.aspect_mask
        });
    copied == image_aspects(format).into()
}

/// The initial layout of a heap texture's image of `tiling`: `PREINITIALIZED` for a linear one,
/// whose move to `GENERAL` then keeps the bytes it holds, the heap's.
fn initial_layout(tiling: vk::ImageTiling) -> vk::ImageLayout {
    if tiling == vk::ImageTiling::LINEAR {
        vk::ImageLayout::PREINITIALIZED
    } else {
        vk::ImageLayout::UNDEFINED
    }
}

/// Whether a heap texture's image of `tiling` lays its rows out linearly (`Image::linear`): linear
/// tiling, or `DRM_FORMAT_MOD_LINEAR`'s, the one modifier a heap's image takes (`dma_buf.rs`).
fn linear_tiling(tiling: vk::ImageTiling) -> bool {
    tiling == vk::ImageTiling::LINEAR || tiling == vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT
}

/// Metal's answer for a texture buffer of `layout` in a heap (a Metal heap probe, an M3 Max and an
/// M4 alike): its row of texels exactly, at a multiple of 16 bytes
/// (`minimumTextureBufferAlignmentForPixelFormat:`, `LINEAR_ALIGNMENT`) for every format and
/// storage mode. The texels are a buffer of the heap's (`Device::new_heap_texture_buffer`), which
/// lies at any offset; a device that views texels at coarser offsets
/// (`minTexelBufferOffsetAlignment` past 16) views them through a staging buffer of the texture's
/// (`buffer_texture.rs`). Refused as `new_texture` refuses the format.
pub(crate) fn texture_buffer_size_and_align(layout: &TextureLayout) -> Result<SizeAndAlign, Error> {
    Ok(SizeAndAlign {
        size: texture_buffer_bytes(layout)?,
        align: LINEAR_ALIGNMENT,
    })
}

/// The alignment Metal answers for a heap's 4:2:2 texture, and the multiple of bytes its rows lie
/// at there.
const FOUR_TWO_TWO_ALIGNMENT: u64 = 16;

/// Metal's answer for a heap's 4:2:2 texture of `layout` (2D, one level): its rows of 2-byte
/// texels at a multiple of 16 bytes, the last unpadded, at an alignment of 16 (an M4 under macOS
/// 27.0.1: 44 bytes for 6 x 3, 148 for 10 x 5, 6000 for 1000 x 3, in every
/// storage mode and usage; the M3 Max's alike where the consumer measured). It is the span of
/// the texture's planes as the runtime lays them out in the heap (`heap_plane_layout`).
fn four_two_two_heap_answer(layout: &TextureLayout) -> SizeAndAlign {
    let (_, size) = heap_plane_layout(
        chroma_pairs::CARRIER,
        layout.pixel_format,
        layout.extent,
        layout.levels,
        layout.layers,
    );
    SizeAndAlign {
        size,
        align: FOUR_TWO_TWO_ALIGNMENT,
    }
}

/// The alignment `heapTextureSizeAndAlignWithDescriptor:` answers, with 0 bytes, for a texture of
/// `pf` and type `ty` of which plain Metal makes none (a descriptor's [`Error::Nil`]):
///
/// - a two-plane YCbCr format's ([`TextureProblem::NoTextureOfFormat`]): 16, and 128 for a 2D
///   array, for every type it takes (an Apple M4 under macOS 27.0.1, each of the 69 formats; with
///   the 1D array, cube array and texture buffer, our probe and an M3 Max under 26.7 alike);
/// - a two-plane extended-range cube or cube array's ([`TextureProblem::TypeForFormat`]): 16
///   (an M4, plain and validated);
/// - `Unspecialized`'s, `RGBA16Float_XR`'s, `AnyDepth`'s and `AnyStencil`'s: 128 for a 1D, 2D
///   and 2D multisample texture, 16 for a texture buffer, and for the arrays, cubes and 3D
///   textures the level's answer (`MetalLevel::no_texture_stack_alignment`): 16 KiB, as for a
///   texture of a format, on an M3 Max under macOS 26.7 (the consumer's report alike; plain and
///   validated, every usage and storage mode), and 128 under macOS 27.0.1 (and an M4 alike).
pub(crate) fn no_texture_alignment(pf: PixelFormat, ty: TextureType, level: MetalLevel) -> u64 {
    use TextureType as T;
    if ycbcr_format(pf).is_some() {
        return if ty == T::Type2DArray { 128 } else { 16 };
    }
    if two_planes::holds_planes(pf) {
        return TWO_PLANE_ALIGNMENT;
    }
    match ty {
        T::TextureBuffer => 16,
        T::Type1D | T::Type2D | T::Type2DMultisample => 128,
        T::Type1DArray
        | T::Type2DArray
        | T::Cube
        | T::CubeArray
        | T::Type3D
        | T::Type2DMultisampleArray => level.no_texture_stack_alignment(),
    }
}

/// The multiple Metal rounds a heap's two-plane extended-range texture's plane 0 up to, where its
/// plane 1 starts.
const TWO_PLANE_PAGE: u64 = 16 << 10;

/// The alignment Metal answers for a heap's two-plane extended-range texture but a 2D array, and
/// the multiple its planes' rows lie at.
const TWO_PLANE_ALIGNMENT: u64 = 16;

/// The alignment Metal answers for a heap's two-plane 2D array, and the multiple each plane's
/// slices lie at (a 1D array's too).
const TWO_PLANE_SLICE_ALIGNMENT: u64 = 128;

/// Metal's answer for a heap's two-plane extended-range texture of `layout` (an M4 under macOS
/// 27.0.1, and the M3 Max under 26.7 alike, plain and validated; every usage and storage mode):
/// plane 0's slice is its rows of 4-byte words at a multiple of 16 bytes, plane 1's its rows of
/// alpha bytes at a multiple of 16, the last unpadded. Plane 0's slices, rounded up to
/// [`TWO_PLANE_PAGE`], then plane 1's. A 2D or 1D array's slices each lie at a multiple of 128 in
/// each plane, the last one's padded; a mipmapped texture answers its first level alone and a 3D
/// one its first slice (Metal lays the rest out past the answer). The alignment is 16, 128 for a 2D
/// array. 64 x 64 answers 20480 at 16; 1 x 1 16385; 1920 x 1080 10380288; a 2D array of three 64 x
/// 64 slices 61440 at 128.
fn two_plane_heap_answer(layout: &TextureLayout) -> SizeAndAlign {
    let [width, height, _] = layout.extent.map(u64::from);
    let words = height * (4 * width).next_multiple_of(TWO_PLANE_ALIGNMENT);
    let alpha = (height - 1) * width.next_multiple_of(TWO_PLANE_ALIGNMENT) + width;
    match layout.texture_type {
        TextureType::Type2DArray | TextureType::Type1DArray => {
            let slices = u64::from(layout.layers);
            let plane0 = slices * words.next_multiple_of(TWO_PLANE_SLICE_ALIGNMENT);
            let plane1 = slices * alpha.next_multiple_of(TWO_PLANE_SLICE_ALIGNMENT);
            let align = if layout.texture_type == TextureType::Type2DArray {
                TWO_PLANE_SLICE_ALIGNMENT
            } else {
                TWO_PLANE_ALIGNMENT
            };
            SizeAndAlign {
                size: plane0.next_multiple_of(TWO_PLANE_PAGE) + plane1,
                align,
            }
        }
        _ => SizeAndAlign {
            size: words.next_multiple_of(TWO_PLANE_PAGE) + alpha,
            align: TWO_PLANE_ALIGNMENT,
        },
    }
}

/// Metal's answer for a heap texture of `layout` whose planes Metal's answer is measured over
/// (`Device::metal_heap_answer`): a 4:2:2 texture's ([`four_two_two_heap_answer`]) and a
/// two-plane extended-range texture's ([`two_plane_heap_answer`]); `None` for any other.
fn planes_answer(layout: &TextureLayout) -> Option<SizeAndAlign> {
    match layout.pixel_format {
        PixelFormat::GBGR422 | PixelFormat::BGRG422 => Some(four_two_two_heap_answer(layout)),
        pf if two_planes::holds_planes(pf) => Some(two_plane_heap_answer(layout)),
        _ => None,
    }
}

/// The descriptor Metal sizes a heap texture of `descriptor` as, where that is another's
/// (`Device::metal_heap_answer`): an `X32_Stencil8` texture as a `Stencil8` one, a stencil
/// texture (an M4); one of Apple's private `BlockWritesOnly` usage as one without it (the
/// consumer's report at 4ce09ab30), of `ShaderRead` usage where it has no other, which Metal
/// compresses alike (equal answers at every shape), not as `Unknown`.
/// `None` for any other.
fn sized_as(descriptor: &TextureDescriptor) -> Option<TextureDescriptor> {
    if descriptor.pixel_format == PixelFormat::X32Stencil8 {
        return Some(TextureDescriptor {
            pixel_format: PixelFormat::Stencil8,
            ..*descriptor
        });
    }
    if !descriptor.usage.contains(TextureUsage::BLOCK_WRITES_ONLY) {
        return None;
    }
    let rest = descriptor.usage.difference(TextureUsage::BLOCK_WRITES_ONLY);
    Some(TextureDescriptor {
        usage: match rest.is_empty() {
            true => TextureUsage::SHADER_READ,
            false => rest,
        },
        ..*descriptor
    })
}

/// What a heap texture whose image has the memory requirements `image`, and its depth twin `twin`
/// where it has one, takes of a heap on a device that places images as `placement` does: the image
/// at the texture's offset and the twin past it, at the next multiple of its alignment, each placed
/// as `placement` places an image; the texture's alignment the largest of theirs and its size in
/// `placement`'s units over both, so each image's offset meets its own alignment wherever the
/// texture lies. `Device::heap_plan` passes `ImagePlacement::in_heap`, the images' own alignments
/// and sizes in whole chunks of compression data where the driver writes them past the size (so
/// no other resource lies in a chunk the image writes) and no `bufferImageGranularity` padding,
/// which separates a heap's image from nothing: the heap's own buffer covers its memory from end
/// to end. With the twin's offset from the texture's.
pub(crate) fn heap_extent(
    image: &vk::MemoryRequirements,
    twin: Option<&vk::MemoryRequirements>,
    placement: ImagePlacement,
) -> (SizeAndAlign, Option<u64>) {
    let image = placement.place(*image);
    let mut align = image.alignment;
    let mut end = image.size;
    let twin_offset = twin.map(|twin| {
        let twin = placement.place(*twin);
        align = align.max(twin.alignment);
        let offset = end.next_multiple_of(twin.alignment);
        end = offset + twin.size;
        offset
    });
    let size = end.next_multiple_of(placement.size_unit);
    (SizeAndAlign { size, align }, twin_offset)
}

/// Whether a heap texture whose images take `images` of the heap ([`heap_extent`]) lies inside
/// Metal's answer `metal`, so the texture takes Metal's answer with its image bound at the
/// texture's offset: no more bytes and no coarser alignment. A texture that does not is held to
/// Metal's answer by a shadow of its bytes, its image in memory of its own, where one serves
/// ([`held_to_metal`]); the number the query answers is Metal's either way.
pub(crate) fn images_fit(images: SizeAndAlign, metal: SizeAndAlign) -> bool {
    images.size <= metal.size && images.align <= metal.align
}

/// What a heap texture's descriptor and its planned images decide about holding it to Metal's
/// answer, apart from the numbers ([`held_to_metal`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HeapTextureShape {
    /// Its image lies in memory of its own already, taking none of the heap's bytes
    /// ([`AliasRoute::Unaliased`], [`AliasRoute::Planes`], [`AliasRoute::Lossless`]), so the
    /// texture takes Metal's answer whatever the image needs.
    pub(crate) apart: bool,
    /// A 4:2:2 texture, always a shadow of the heap's bytes: its bytes lie where Metal's linear
    /// rows do (`heap_plane_layout`), never where its image lays them.
    pub(crate) four_two_two: bool,
    /// It is held with a depth twin, whose bytes no shadow of the texture's moves.
    pub(crate) twin: bool,
    /// More than one sample: no copy of rows shadows its image, which lies in memory of its own
    /// instead, as one the heap's memory cannot hold.
    pub(crate) multisample: bool,
    /// A shadow of the heap's bytes fits the image's planes ([`shadowable`]).
    pub(crate) shadowable: bool,
}

/// How a heap texture is held to Metal's answer ([`held_to_metal`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HeldToMetal {
    /// Its image bound at the texture's offset of the heap's memory: the images lie inside
    /// Metal's answer, or in memory of their own already.
    Images,
    /// Its image in memory of its own, a shadow of the texture's bytes (Metal's tiles where they
    /// serve, else rows): the images ask more than Metal's answer and a shadow serves.
    Shadow,
    /// Neither: the images ask more and no shadow serves. A measured route's answer is then
    /// refused (`NotImplemented(HeapTextureMemory)`), since the texture's bytes are no layout the
    /// image holds; the model's is widened to the images' and announced
    /// `heap-texture-larger-than-metal`.
    Nothing,
}

/// How a heap texture of `shape` whose images take `images` of the heap is held to Metal's answer
/// `metal`, which a program sizes a heap and places its resources by (`Device::heap_plan`).
pub(crate) fn held_to_metal(
    images: SizeAndAlign,
    metal: SizeAndAlign,
    shape: HeapTextureShape,
) -> HeldToMetal {
    if shape.apart || (!shape.four_two_two && images_fit(images, metal)) {
        return HeldToMetal::Images;
    }
    // A depth twin's bytes lie past the image's inside the texture's, and no shadow moves them.
    if shape.twin {
        return HeldToMetal::Nothing;
    }
    match shape.multisample || shape.shadowable {
        true => HeldToMetal::Shadow,
        false => HeldToMetal::Nothing,
    }
}

/// Whether a heap texture of alignment `align` lies at a multiple of it wherever a heap's memory
/// starts: refused past [`HEAP_BASE_ALIGNMENT`] (`RuntimeGap::HeapTextureAlignment`).
pub(crate) fn check_heap_base(align: u64) -> Result<(), Error> {
    if align > HEAP_BASE_ALIGNMENT {
        return Err(Error::NotImplemented(RuntimeGap::HeapTextureAlignment {
            alignment: align,
        }));
    }
    Ok(())
}

/// The id name of the informational message of a placement heap's texture the heap does not make
/// at an offset inside it ([`Device::new_heap_texture`]).
pub(crate) const HEAP_TEXTURE_REFUSED: &str = "metal2vulkan-heap-texture-refused";

/// What a heap texture's making planned before it was refused: the size and alignment it places
/// the texture by, and its image's route, tiling and memory requirements (`None` for a texture
/// buffer, which has no image).
#[derive(Clone, Copy)]
struct Planned {
    size_and_align: SizeAndAlign,
    image: Option<(AliasRoute, vk::ImageTiling, vk::MemoryRequirements)>,
}

impl Planned {
    /// What `p` plans.
    fn of(p: &HeapPlan) -> Planned {
        let image = &p.image;
        Planned {
            size_and_align: p.size_and_align,
            image: Some((image.route, image.tiling, image.requirements)),
        }
    }
}

/// The text of the informational message [`HEAP_TEXTURE_REFUSED`]: the texture of `descriptor` that
/// placement heap `heap` does not make at `offset` (inside it), refused with `e` after its making
/// planned `planned`: the descriptor, the offset, the heap's size, options and memory, the size and
/// alignment `device` answers the descriptor, and what the making planned.
fn refused_text(
    device: &Device,
    heap: &Heap,
    descriptor: &TextureDescriptor,
    offset: u64,
    planned: Option<Planned>,
    e: &Error,
) -> String {
    let d = descriptor;
    let memory = match heap.bytes().and_then(Buffer::caller_memory) {
        Some(m) => format!("over the caller's memory ({m:?})"),
        None => "of the device's memory".to_owned(),
    };
    // Where `offset` lies against an answer of `size` bytes at `align`.
    let against = |SizeAndAlign { size, align }: SizeAndAlign| {
        let aligned = offset.is_multiple_of(align.max(1));
        let inside = offset
            .checked_add(size)
            .is_some_and(|end| end <= heap.size());
        format!(
            "{size:#x} bytes at an alignment of {align:#x} (the offset {}a multiple of it, the \
             bytes {}inside the heap)",
            if aligned { "" } else { "not " },
            if inside { "" } else { "not " }
        )
    };
    let answer = match device.heap_texture_size_and_align(d) {
        Ok(answer) => against(answer),
        Err(e) => format!("no size: {e}"),
    };
    let made = match planned {
        Some(Planned {
            size_and_align,
            image,
        }) => {
            let image = match image {
                Some((route, tiling, r)) => {
                    let tiling = match tiling {
                        vk::ImageTiling::OPTIMAL => "optimal",
                        vk::ImageTiling::LINEAR => "linear",
                        vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT => "DRM format modifier",
                        _ => "other",
                    };
                    format!(
                        ", its image of the {route:?} route, {tiling} tiling, asking {:#x} bytes \
                         at {:#x} of memory types {:#x}",
                        r.size, r.alignment, r.memory_type_bits
                    )
                }
                None => ", a texture buffer".to_owned(),
            };
            format!("planned {}{image}", against(size_and_align))
        }
        None => "planned no image".to_owned(),
    };
    format!(
        "metal2vulkan: newTextureWithDescriptor:offset: made no texture: a {:?} {:?} texture of \
         {}x{}x{}, {} levels, {} slices, {} samples, usage {:?}, options {:#x}, GPU-optimized \
         contents {}, at offset {offset:#x} of a placement heap of {:#x} bytes, options {:#x}, \
         {memory}; the size query answers it {answer}; the making {made}; refused: {e} \
         [{HEAP_TEXTURE_REFUSED}]",
        d.texture_type,
        d.pixel_format,
        d.width,
        d.height,
        d.depth,
        d.mipmap_level_count,
        d.array_length,
        d.sample_count,
        d.usage,
        d.resource_options.raw(),
        d.allow_gpu_optimized_contents,
        heap.size(),
        heap.resource_options().raw(),
    )
}

/// Where a resource of `size` bytes at a multiple of `align` lies in `heap` for `call`: at
/// `offset` of a placement heap, refused off the alignment or past the heap's end (`Nil`, as plain
/// Metal answers nil, its validation stopping it), or at the lowest offset with room of an
/// automatic heap when `None`, then taken (`OutOfMemory(Heap)` without room).
fn heap_offset(
    heap: &Heap,
    call: &'static str,
    SizeAndAlign { size, align }: SizeAndAlign,
    offset: Option<u64>,
) -> Result<u64, Error> {
    let Some(offset) = offset else {
        return heap
            .allocate(size, align)
            .ok_or(Error::OutOfMemory(MemoryKind::Heap));
    };
    if !offset.is_multiple_of(align) {
        return Err(nil(call, Problem::Heap(HeapProblem::Offset { offset })));
    }
    if offset.checked_add(size).is_none_or(|end| end > heap.size()) {
        return Err(nil(call, heap.past_end(offset, size)));
    }
    Ok(offset)
}

impl Device {
    /// `heapTextureSizeAndAlignWithDescriptor:`: the bytes and alignment a texture of
    /// `descriptor` takes of a heap on this device (the module docs). Metal's answer wherever the
    /// runtime models it, so a program may size a heap and place its resources by Metal's
    /// numbers; the device's images' own footprint, with a depth twin's bytes after the image's
    /// where it has one and whole chunks of compression data where the driver writes them past an
    /// image's size, where the model has none or no shadow holds the texture to Metal's answer
    /// (then announced `heap-texture-larger-than-metal`). The image of a descriptor with
    /// `allow_gpu_optimized_contents` off is one of `PixelFormatView` usage, as Metal lays out a
    /// color texture with it off (the module docs). A texture of the descriptor fits at any
    /// multiple of the alignment in any heap with room. A texture buffer takes Metal's answer: its
    /// row of texels, at a multiple of 16 bytes.
    ///
    /// A descriptor of which plain Metal makes no texture (its creators answer [`Error::Nil`]:
    /// `Unspecialized`, a two-plane extended-range cube or cube array, a format Metal makes no
    /// texture of from a descriptor alone such as a two-plane YCbCr one) answers 0 bytes at the
    /// alignment Metal's query answers for it (`no_texture_alignment`), once every rule Metal stops
    /// passed. A descriptor of `Memoryless` storage, which no heap holds, answers Metal's model
    /// with the texture uncompressed, as plain Metal answers it (`memoryless_answer`); every other
    /// refusal is [`Heap::new_texture`]'s for the descriptor: `Invalid` for a descriptor Metal
    /// stops (ending the process, as Metal's query does), `Unsupported` for a format, size or
    /// sample count the device lacks, and `NotImplemented` for an alignment past the heap's base
    /// (`HeapTextureAlignment`).
    ///
    /// Metal's answer by usage (an Apple M4 under macOS 27.0.1, 63210 answers, the
    /// consumer's M3 Max rows under 26.7 alike): each shape has two answers, its compressed and its
    /// plain size, and the usage picks one as `heap_tiled::metal_keeps_uncompressed` says; the
    /// storage mode never changes it. The private usages `NoConcurrentAccess` (0x4000), `Video`
    /// (0x8000) and `BlockWritesOnly` (0x10000) keep nothing uncompressed, alone too. A texture of
    /// `ShaderAtomic` usage of any format but `R32Uint` ends Metal's process (a failed assertion,
    /// without the debug layer too; a depth or stencil texture's too), so Metal answers it nothing:
    /// the runtime answers a colour texture its plain size, by the rule, and refuses a depth or
    /// stencil one `Invalid`, as every creator does. Metal's private usage 0x8 alone is one it
    /// compresses, where its texture is of `Unknown` usage: the descriptor keeps it
    /// ([`TextureUsage::DROPPED`]), so the answer is Metal's. A descriptor whose layout in Metal's
    /// heaps the runtime does not model (a format or sample count not measured: a PVRTC format at
    /// a Metal level of macOS 26.7, 8 samples) answers its device's image's size, announced `unmeasured-heap-texture-size` (a
    /// mipmapped texture Metal compresses and a `Depth32Float_Stencil8` one answer Metal's).
    pub fn heap_texture_size_and_align(
        &self,
        descriptor: &TextureDescriptor,
    ) -> Result<SizeAndAlign, Error> {
        const CALL: &str = "heap_texture_size_and_align";
        let layout = match self.heap_layout(CALL, descriptor, Creator::SizeQuery) {
            // A texture plain Metal makes none of answers 0 bytes, at the alignment
            // `no_texture_alignment` states.
            Err(Error::Nil(_)) => {
                let (pf, ty) = (descriptor.pixel_format, descriptor.texture_type);
                return Ok(SizeAndAlign {
                    size: 0,
                    align: no_texture_alignment(pf, ty, self.properties().metal_level),
                });
            }
            layout => layout?,
        };
        if layout.texture_type == TextureType::TextureBuffer {
            return texture_buffer_size_and_align(&layout);
        }
        if descriptor.resource_options.storage_mode()? == StorageMode::Memoryless {
            return self.memoryless_answer(CALL, descriptor, &layout);
        }
        match self.heap_plan(CALL, descriptor, layout) {
            // A heap texture whose image the device decodes as it is placed is refused its making
            // (`heap_plan`: an ASTC carrier's decoded twin would decode what the heap's bytes
            // held, `astc.rs`), where plain Metal makes it. Its footprint is Metal's blocks,
            // which no image of the device decides, so the query answers Metal's model as plain
            // Metal answers it; the making's refusal stands.
            Err(Error::NotImplemented(RuntimeGap::DecodedHeapTexture)) => {
                let optimized = descriptor.allow_gpu_optimized_contents;
                let answer = self.modelled_answer(CALL, descriptor, &layout, optimized)?;
                answer.ok_or(Error::NotImplemented(RuntimeGap::DecodedHeapTexture))
            }
            plan => Ok(plan?.size_and_align),
        }
    }

    /// Metal's own answer to `heapTextureSizeAndAlignWithDescriptor:` for `descriptor`, as the
    /// runtime models it (`texture/heap_tiled.rs`), whatever this device's images need: the same
    /// on every device at the same Metal level (`DeviceProperties::metal_level`). `None` where the
    /// model has no answer (a PVRTC format at macOS 26.7's level, 8 samples, any other layout not
    /// measured), where
    /// [`Device::heap_texture_size_and_align`] announces `unmeasured-heap-texture-size`. No Metal
    /// counterpart: Metal answers its own GPU's.
    ///
    /// [`Device::heap_texture_size_and_align`] stays the answer to place a texture by, and is
    /// this one wherever the model has an answer: a texture whose images need more bytes or a
    /// coarser alignment is held to it by a shadow of its bytes, and only where no shadow serves
    /// is the answer the larger of the two (announced `heap-texture-larger-than-metal`). This one
    /// is for bookkeeping that must follow Metal's numbers whatever the device does, such as the
    /// range a texture of memory of its own takes of a guest's address space. It announces nothing
    /// and makes no image.
    ///
    /// Answered as the size query answers Metal's: a descriptor of which plain Metal makes no
    /// texture 0 bytes at Metal's alignment for it, a texture buffer its row of texels at 16, a
    /// 4:2:2 or two-plane extended-range texture its measured planes, an `X32_Stencil8` texture a
    /// `Stencil8` one's answer, one of `BlockWritesOnly` usage the answer without it, one of
    /// `Memoryless` storage the uncompressed answer. Refused as
    /// [`Device::heap_texture_size_and_align`] refuses the descriptor before it plans an image:
    /// `Invalid` for a descriptor Metal stops.
    pub fn metal_heap_texture_size_and_align(
        &self,
        descriptor: &TextureDescriptor,
    ) -> Result<Option<SizeAndAlign>, Error> {
        const CALL: &str = "metal_heap_texture_size_and_align";
        let layout = match self.heap_layout(CALL, descriptor, Creator::SizeQuery) {
            Err(Error::Nil(_)) => {
                let (pf, ty) = (descriptor.pixel_format, descriptor.texture_type);
                return Ok(Some(SizeAndAlign {
                    size: 0,
                    align: no_texture_alignment(pf, ty, self.properties().metal_level),
                }));
            }
            layout => layout?,
        };
        if layout.texture_type == TextureType::TextureBuffer {
            return texture_buffer_size_and_align(&layout).map(Some);
        }
        let optimized = descriptor.allow_gpu_optimized_contents
            && descriptor.resource_options.storage_mode()? != StorageMode::Memoryless;
        self.modelled_answer(CALL, descriptor, &layout, optimized)
    }

    /// Metal's modelled answer for a heap texture of `descriptor` (checked as `layout`, not a
    /// texture buffer), for `call`, where Metal may keep it losslessly compressed while
    /// `optimized` (the descriptor's `allowGPUOptimizedContents`, and never for `Memoryless`
    /// storage, [`Device::memoryless_answer`]): the planes' answer where Metal's is measured so
    /// ([`planes_answer`]), the answer of the descriptor Metal sizes it as ([`sized_as`]), else
    /// `heap_tiled`'s model ([`modelled_size`]); `None` where none answers.
    fn modelled_answer(
        &self,
        call: &'static str,
        descriptor: &TextureDescriptor,
        layout: &TextureLayout,
        optimized: bool,
    ) -> Result<Option<SizeAndAlign>, Error> {
        if let Some(answer) = planes_answer(layout) {
            return Ok(Some(answer));
        }
        if let Some(other) = sized_as(descriptor) {
            let layout = self.heap_layout(call, &other, Creator::SizeQuery)?;
            return self.modelled_answer(call, &other, &layout, optimized);
        }
        let usage = effective_usage(descriptor.usage);
        let level = self.properties().metal_level;
        Ok(modelled_size(layout, usage, optimized, level))
    }

    /// The layout of a heap texture of `descriptor`, for `call`: the descriptor checked for
    /// `creator` (a heap's, or the size query, which answers for what the runtime does not build
    /// yet), and its options too. A heap's storage is `Shared` or `Private`, so a heap's creator
    /// refuses `Memoryless` storage (`Heap::placed_options` refuses it as the heap's own mismatch
    /// too); the size query answers it, as plain Metal does ([`Device::memoryless_answer`]).
    fn heap_layout(
        &self,
        call: &'static str,
        descriptor: &TextureDescriptor,
        creator: Creator,
    ) -> Result<TextureLayout, Error> {
        let options = descriptor.resource_options;
        let storage = options.storage_mode()?;
        options.hazard_tracking_mode()?;
        options.cpu_cache_mode()?;
        let layout = descriptor
            .check_made(creator, self.properties().texture_rules())
            .map_err(|refused| refused.error(call))?;
        if storage == StorageMode::Memoryless && creator != Creator::SizeQuery {
            return Err(invalid(call, Problem::StorageMode));
        }
        if self.is_lost() {
            return Err(Error::DeviceLost);
        }
        Ok(layout)
    }

    /// What the heap size query answers for a `Memoryless` `descriptor` (checked as `layout`, not
    /// a texture buffer), for `call`: Metal's modelled answer with the texture uncompressed,
    /// because a texture with no memory holds no lossless-compression metadata. Measured on an
    /// Apple M3 Max under macOS 27.0.1: every one of the 20 `Memoryless` answers of
    /// `texture-pairs-storage-crossings` and `texture-pairs-plain-contents` is its descriptor's
    /// uncompressed answer, where the other modes answer the compressed one (a 64x64
    /// `RGBA8Unorm` texture a shader reads: 16384 against 16512; its 7-level twin 22144 against
    /// 22528; a 3-slice array 98304 against 99456; a 4-sample one 65536 against 66048; a
    /// `Depth32Float` render target 16384 against 16512), and the modes that keep every texture
    /// uncompressed answer alike.
    ///
    /// No heap holds such a texture, so the runtime plans no image for it. Where the model has no
    /// answer (a format or sample count not measured) it answers the `Private` twin's, as the
    /// query answers every other mode, announcing `unmeasured-heap-texture-size` there.
    fn memoryless_answer(
        &self,
        call: &'static str,
        descriptor: &TextureDescriptor,
        layout: &TextureLayout,
    ) -> Result<SizeAndAlign, Error> {
        if let Some(answer) = self.modelled_answer(call, descriptor, layout, false)? {
            return Ok(answer);
        }
        let private = TextureDescriptor {
            resource_options: (descriptor.resource_options).with_storage_mode(StorageMode::Private),
            ..*descriptor
        };
        let layout = self.heap_layout(call, &private, Creator::SizeQuery)?;
        Ok(self.heap_plan(call, &private, layout)?.size_and_align)
    }

    /// The images a heap texture of `descriptor` (checked as `layout`, not a texture buffer)
    /// takes and what it takes of a heap, for `call`: its image, and its depth twin past it where
    /// the device holds the texture with one.
    fn heap_plan(
        &self,
        call: &'static str,
        descriptor: &TextureDescriptor,
        layout: TextureLayout,
    ) -> Result<HeapPlan, Error> {
        let facts = self
            .shared()
            .textures
            .facts(self.compile_device().profile());
        let usage = descriptor.image_usage();
        let (plan, twin) = plan_images(&layout, usage, false, &facts)
            .map_err(|refusal| image_error(call, refusal))?;
        // A heap texture's samples read what its bytes hold, which an ASTC carrier's decoded twin
        // would decode as it is placed (`astc.rs`): refused. The size query answers Metal's model
        // over this refusal (`Device::heap_texture_size_and_align`).
        if crate::astc::decode_of(&plan).is_some() {
            return Err(Error::NotImplemented(RuntimeGap::DecodedHeapTexture));
        }
        // An image whose copies would rewrite texels a CPU copy keeps is made uncompressed; the
        // texture still takes the compressed image's bytes of the heap, as Metal's does.
        let (plan, twin, compressed) =
            match self.copy_exact_usage(descriptor, &layout, plan.format, usage) {
                Some(exact) => {
                    let (exact_plan, exact_twin) = plan_images(&layout, exact, false, &facts)
                        .map_err(|refusal| image_error(call, refusal))?;
                    (exact_plan, exact_twin, Some(plan))
                }
                None => (plan, twin, None),
            };
        let class = buffer_class(call, Modes::of(descriptor.resource_options)?)?;
        let free = self.with_free_usage(plan.clone(), layout.texture_type)?;
        // A shadowed image is no depth twin's texture: one held with a twin is not.
        let mine = Mine::Texture {
            shadows: twin.is_none(),
            compressed: metal_compresses_layout(
                &layout,
                effective_usage(descriptor.usage),
                descriptor.allow_gpu_optimized_contents,
            ),
        };
        // A two-plane texture's image is larger than Metal's answer: it lies in memory of its
        // own, a shadow of its planes in the heap's bytes where Metal's layout of them is
        // measured.
        let mut image = if two_planes::holds_planes(layout.pixel_format) {
            let requirements =
                self.planned_requirements(&free, layout.texture_type, vk::ImageTiling::OPTIMAL)?;
            let route = match self.two_plane_heap_layout(&layout, &free) {
                Some(_) => AliasRoute::Planes,
                None => AliasRoute::Unaliased,
            };
            HeapImage {
                plan: free,
                tiling: vk::ImageTiling::OPTIMAL,
                requirements,
                route,
            }
        } else {
            match self.heap_image(free.clone(), plan, layout.texture_type, class, mine) {
                // A multisample texture where neither its optimal image takes the heap's memory
                // nor the device's linear images can hold it (an NVIDIA GPU's `Shared` heap; a
                // `Shared` heap's depth texture on MoltenVK, whose optimal images take no
                // host-visible memory): its optimal image in memory of its own, as over host
                // pages, a shadow of its samples in Metal's tiles there where one serves (below).
                Err(error) if linear_refusal(&error) && twin.is_none() && layout.samples > 1 => {
                    let requirements = self.planned_requirements(
                        &free,
                        layout.texture_type,
                        vk::ImageTiling::OPTIMAL,
                    )?;
                    HeapImage {
                        plan: free,
                        tiling: vk::ImageTiling::OPTIMAL,
                        requirements,
                        route: AliasRoute::Lossless,
                    }
                }
                made => made?,
            }
        };
        // The twin takes none of the runtime's free usage, as `new_texture`'s, and is no
        // texture's image: its bytes alias nothing a program reads.
        let twin = match twin {
            Some(twin) => {
                Some(self.heap_image(twin.clone(), twin, layout.texture_type, class, Mine::Twin)?)
            }
            None => None,
        };
        // The images' own footprint: a heap's image takes neither the granularity padding nor a
        // page of its own (`ImagePlacement::in_heap`).
        let placement = self.shared().textures.placement().in_heap();
        let (mut size_and_align, twin_offset) = heap_extent(
            &image.requirements,
            twin.as_ref().map(|twin| &twin.requirements),
            placement,
        );
        if let Some(compressed) = compressed {
            let free = self.with_free_usage(compressed, layout.texture_type)?;
            let requirements =
                self.planned_requirements(&free, layout.texture_type, vk::ImageTiling::OPTIMAL)?;
            let (answer, _) = heap_extent(&requirements, None, placement);
            size_and_align = SizeAndAlign {
                size: size_and_align.size.max(answer.size),
                align: size_and_align.align.max(answer.align),
            };
        }
        // What the device's images take of the heap, kept apart from the answer: a texture held
        // to Metal's answer takes those bytes of memory of its own instead.
        let images = size_and_align;
        // Metal's answer where it is not the answer the runtime models for every descriptor
        // (`metal_heap_answer`): a 4:2:2 or two-plane extended-range texture's planes, an
        // `X32_Stencil8` or `BlockWritesOnly` texture as the descriptor Metal sizes it as.
        let special = self.metal_heap_answer(call, descriptor, &layout, image.plan.format)?;
        // Whether that is where Metal's answer comes from: a texture whose images do not fit such
        // a measured route's answer is refused, where one that does not fit the model's keeps its
        // device's answer, announced.
        let measured_route = special.is_some();
        // Metal's answer, where the runtime knows it, and the runtime's layout of the texture.
        let (metal_answer, mut metal_layout) = match special {
            Some((metal, metal_layout)) => (Some(metal), metal_layout),
            None => {
                let usage = effective_usage(descriptor.usage);
                let optimized = descriptor.allow_gpu_optimized_contents;
                let level = self.properties().metal_level;
                let format = image.plan.format;
                let measured = metal_heap_layout(&layout, usage, optimized, format, level);
                if measured.answer.is_none() {
                    self.degrade_unmeasured_size(call, &layout);
                }
                (measured.answer, measured.layout)
            }
        };
        // Metal's answer, which a program sizes a heap and places its resources by, is the
        // answer: the image lies in it where the images fit inside it, else the texture is a
        // shadow of its bytes there, its image in memory of its own (Metal's tiles below). A
        // 4:2:2 texture is always a shadow: its bytes then lie where Metal's linear rows do
        // (`heap_plane_layout`). A two-plane texture's image lies apart from the heap's bytes
        // (above). Metal's answer the texture cannot be held to, announced below.
        let mut past_metal = None;
        if let Some(metal) = metal_answer {
            let shape = HeapTextureShape {
                apart: matches!(
                    image.route,
                    AliasRoute::Unaliased | AliasRoute::Planes | AliasRoute::Lossless
                ),
                four_two_two: chroma_pairs::is_four_two_two(layout.pixel_format),
                twin: twin.is_some(),
                multisample: layout.samples > 1,
                shadowable: shadowable(&image.plan, metal.size),
            };
            match held_to_metal(images, metal, shape) {
                HeldToMetal::Images => size_and_align = metal,
                HeldToMetal::Shadow => {
                    // A multisample image, which no copy of rows shadows, lies in memory of its
                    // own, as one the heap's memory cannot hold (above); any other is a shadow of
                    // the heap's bytes, in Metal's tiles below where they serve.
                    image.route = match shape.multisample {
                        true => AliasRoute::Lossless,
                        false => match image.route {
                            kept @ (AliasRoute::Shadow | AliasRoute::MemoryShadow) => kept,
                            _ => AliasRoute::MemoryShadow,
                        },
                    };
                    size_and_align = metal;
                }
                HeldToMetal::Nothing if measured_route => {
                    return Err(Error::NotImplemented(RuntimeGap::HeapTextureMemory))
                }
                HeldToMetal::Nothing => {
                    past_metal = Some(metal);
                    size_and_align = SizeAndAlign {
                        size: size_and_align.size.max(metal.size),
                        align: size_and_align.align.max(metal.align),
                    };
                }
            }
        }
        // A texture whose image lies in memory of its own as a shadow of the heap's bytes (the
        // driver lays depth and stencil images apart from their memory, or the heap's memory
        // takes no image of it), or whose image the driver keeps GPU-optimized where Metal's
        // texture is not, is a shadow of its elements in Metal's tiles there where Metal's layout
        // is measured and the moves fit (`heap_tiled.rs`): its bytes are Metal's (a texture Metal
        // keeps losslessly compressed, its uncompressed tiles), and it takes Metal's answer.
        let shadowed = matches!(image.route, AliasRoute::Shadow | AliasRoute::MemoryShadow)
            || self.keeps_optimized_layout(descriptor, &layout);
        if shadowed && twin.is_none() {
            let level = self.properties().metal_level;
            let tiles = shadow_tiles(&layout, image.plan.format, level, &metal_layout);
            if let (Some(tiles), Some(answer)) = (tiles, metal_answer) {
                let serves = (tiles.tiles())
                    .is_some_and(|tiled| self.tiles_serve(&image.plan, tiled, answer.size));
                if serves {
                    image.route = AliasRoute::Tiled;
                    size_and_align = answer;
                    metal_layout = tiles;
                }
            }
        }
        // A multisample texture in memory of its own is a shadow of its samples in Metal's tiles
        // there (a compressed one's uncompressed tiles), where the device moves them
        // (`heap_tiled.rs`): its bytes are Metal's, and it takes Metal's answer.
        if image.route == AliasRoute::Lossless && layout.samples > 1 && twin.is_none() {
            let shadow = self.sample_tiles(&layout, &image.plan, &metal_layout, metal_answer)?;
            if let Some((plan, tiles, answer)) = shadow {
                image.plan = plan;
                image.route = AliasRoute::Tiled;
                size_and_align = answer;
                metal_layout = tiles;
            }
        }
        check_heap_base(size_and_align.align)?;
        // The size answered is the optimized texture's where the driver keeps the layout, but for
        // a shadow of Metal's tiles.
        if image.route != AliasRoute::Tiled {
            self.degrade_optimized_layout(descriptor, &layout);
        }
        if let Some(metal) = past_metal {
            let past = size_and_align.size > metal.size || size_and_align.align > metal.align;
            if past && !self.keeps_optimized_layout(descriptor, &layout) {
                self.degrade_past_metal_size(call, &layout, size_and_align, metal);
            }
        }
        Ok(HeapPlan {
            layout,
            image,
            twin: twin.zip(twin_offset),
            size_and_align,
            metal_layout,
        })
    }

    /// Announce that a heap texture of `layout` takes `answer` of a heap for `call`, more bytes
    /// or a coarser alignment than Metal's modelled answer `metal`
    /// (`Degradation::HeapTextureLargerThanMetal`): its device's image needs them.
    fn degrade_past_metal_size(
        &self,
        call: &'static str,
        layout: &TextureLayout,
        answer: SizeAndAlign,
        metal: SizeAndAlign,
    ) {
        self.degrade(Degradation::HeapTextureLargerThanMetal, || {
            let [width, height, depth] = layout.extent;
            let object = format!(
                "{call} of a {:?} {:?} texture of {width}x{height}x{depth}, {} levels, {} samples",
                layout.pixel_format, layout.texture_type, layout.levels, layout.samples
            );
            let why = format!(
                "the device's image takes {:#x} bytes at an alignment of {:#x} (its own, at \
                 least bufferImageGranularity, in whole granularity pages), where Metal answers \
                 {:#x} at {:#x}: the answer is the device's",
                answer.size, answer.align, metal.size, metal.align
            );
            (object, why)
        });
    }

    /// Announce that a heap texture of `layout` takes its device's size for `call` where Metal's
    /// layout of it is not modelled (`Degradation::UnmeasuredHeapTextureSize`).
    fn degrade_unmeasured_size(&self, call: &'static str, layout: &TextureLayout) {
        self.degrade(Degradation::UnmeasuredHeapTextureSize, || {
            let [width, height, depth] = layout.extent;
            let object = format!(
                "{call} of a {:?} {:?} texture of {width}x{height}x{depth}, {} levels, {} samples",
                layout.pixel_format, layout.texture_type, layout.levels, layout.samples
            );
            let why = "Metal's layout of the texture in a heap is not modelled (a format or \
                       sample count not measured): the answer is the device's image's"
                .to_owned();
            (object, why)
        });
    }

    /// Metal's answer to `heapTextureSizeAndAlignWithDescriptor:` for `descriptor` (checked as
    /// `layout`) where it is not the size of its own GPU's image of the texture, which the runtime answers
    /// for every other descriptor (the module docs); `None` for those. Measured on an M3 Max
    /// under macOS 26.7 (the consumer's report) and an M4 under 27.0.1:
    ///
    /// - A 4:2:2 texture (`GBGR422`, `BGRG422`) takes its rows of 2-byte texels at a multiple
    ///   of 16 bytes, the last unpadded, at an alignment of 16 (`four_two_two_heap_answer`).
    /// - An `X32_Stencil8` texture, a stencil texture (`texture_descriptor`), takes what a
    ///   `Stencil8` texture of the descriptor takes: the device's answer for that one.
    /// - A two-plane extended-range texture takes its planes as Metal lays them out
    ///   (`two_plane_heap_answer`).
    /// - A texture of Apple's private `BlockWritesOnly` usage (0x10000) takes what the descriptor
    ///   without it takes, one of that usage alone what a `ShaderRead` texture takes (the
    ///   consumer's report at 4ce09ab30): Metal sizes it so (a 16x16 `BGRA8Unorm` `Private` texture
    ///   of `ShaderRead | RenderTarget | BlockWritesOnly` answers 1152 on an M3 Max, as without the
    ///   bit), while its image takes the storage usage, which MoltenVK sizes as a `ShaderWrite`
    ///   texture's (1024). The image lies inside that answer where it fits, else in memory of its
    ///   own as a shadow of the heap's bytes, as for the others.
    ///
    /// With each answer, Metal's layout of the texture's bytes as the runtime holds it over host
    /// pages: a 4:2:2 or two-plane texture's, as its route lays it (`HeapLayout::Measured`); an
    /// `X32_Stencil8` texture's, a `Stencil8` texture's tiles of its image's (of `format`) stencil
    /// aspect (`metal_heap_layout`; an M3 Max under macOS 27.0.1: every texel of one-level 8 x 8
    /// to 300 x 200 textures, both ways, where a `Stencil8` texture's lie, and 1720 answers equal
    /// a `Stencil8` texture's); a `BlockWritesOnly` texture's, the descriptor's without it.
    fn metal_heap_answer(
        &self,
        call: &'static str,
        descriptor: &TextureDescriptor,
        layout: &TextureLayout,
        format: Format,
    ) -> Result<Option<(SizeAndAlign, HeapLayout)>, Error> {
        if let Some(answer) = planes_answer(layout) {
            return Ok(Some((answer, HeapLayout::Measured)));
        }
        let Some(other) = sized_as(descriptor) else {
            return Ok(None);
        };
        let other_layout = self.heap_layout(call, &other, Creator::SizeQuery)?;
        let p = self.heap_plan(call, &other, other_layout)?;
        let metal_layout = match descriptor.pixel_format {
            PixelFormat::X32Stencil8 => {
                let usage = effective_usage(descriptor.usage);
                let optimized = descriptor.allow_gpu_optimized_contents;
                let level = self.properties().metal_level;
                metal_heap_layout(layout, usage, optimized, format, level).layout
            }
            _ => p.metal_layout,
        };
        Ok(Some((p.size_and_align, metal_layout)))
    }

    /// One image of a heap texture of Metal's `texture_type` in a heap of `class`: `optimal`
    /// where its optimal image takes the heap's memory, else `linear`, the same image without the
    /// runtime's free usage, where the device's linear images can do all the optimal one can for
    /// the texture, else `optimal` in memory of its own as a shadow of the heap's bytes
    /// ([`AliasRoute::MemoryShadow`]) where `mine` lets one serve it and it does ([`shadowable`]),
    /// else refused as the device's lack (`Unsupported(Feature::HeapTextureTiling)`).
    ///
    /// The texture's own colour image (`own`, not a depth twin) on a driver whose compressed
    /// images hide writes through an alias
    /// ([`DriverFact::CompressedImagesHideAliasedWrites`]), where its optimal image takes the
    /// heap's memory, takes `optimal`, which the driver may compress, where Metal keeps the
    /// texture losslessly compressed and the device's policy keeps the driver's compression
    /// (`HeapCompressionPolicy::Compressed`, the default; [`AliasRoute::Compressed`]); else in
    /// order: `optimal` with host-transfer usage, where the device enabled `hostImageCopy` (the
    /// driver keeps such an image's bytes, [`DriverFact::HostTransferImagesKeepTheirBytes`]), the
    /// format takes the usage, and the device makes such an image of the plan in that memory; else
    /// `linear`, where it serves as above; else `optimal`, a near-equivalent ([`AliasRoute`]).
    fn heap_image(
        &self,
        optimal: ImagePlan,
        linear: ImagePlan,
        texture_type: TextureType,
        class: MemoryClass,
        mine: Mine,
    ) -> Result<HeapImage, Error> {
        let own = matches!(mine, Mine::Texture { .. });
        // The heap's memory (the same for every heap of the storage: `heap_requirements`), and
        // the image's optimal tiling where it takes that memory, else its linear tiling.
        let types = self.heap_types()?;
        let shared = self.shared();
        let heap = types.heap_memory(&lock(&shared.memory), types.buffer, class);
        let takes = |image: u32| takes_heap(&lock(&shared.memory), heap, image, class);
        let requirements =
            self.planned_requirements(&optimal, texture_type, vk::ImageTiling::OPTIMAL)?;
        if !takes(requirements.memory_type_bits) {
            return match self.linear_heap_image(linear, texture_type, &takes, AliasRoute::Direct) {
                Err(error)
                    if linear_refusal(&error)
                        && matches!(mine, Mine::Texture { shadows: true, .. })
                        && shadowable(&optimal, requirements.size) =>
                {
                    Ok(HeapImage {
                        plan: optimal,
                        tiling: vk::ImageTiling::OPTIMAL,
                        requirements,
                        route: AliasRoute::MemoryShadow,
                    })
                }
                made => made,
            };
        }
        let facts = &self.properties().driver_facts;
        if facts.contains(&DriverFact::DepthStencilImagesLieApart) && !colour(&optimal) {
            let (_, length) = heap_planes(
                optimal.format,
                optimal.pixel_format,
                optimal.extent,
                optimal.levels,
                optimal.layers,
            );
            let shadowed =
                own && optimal.samples == SampleCountFlags::COUNT_1 && length <= requirements.size;
            return Ok(HeapImage {
                plan: optimal,
                tiling: vk::ImageTiling::OPTIMAL,
                requirements,
                route: if shadowed {
                    AliasRoute::Shadow
                } else {
                    AliasRoute::Apart
                },
            });
        }
        let hides = own
            && colour(&optimal)
            && facts.contains(&DriverFact::CompressedImagesHideAliasedWrites);
        if !hides {
            return Ok(HeapImage {
                plan: optimal,
                tiling: vk::ImageTiling::OPTIMAL,
                requirements,
                route: AliasRoute::Direct,
            });
        }
        let compressed = matches!(
            mine,
            Mine::Texture {
                compressed: true,
                ..
            }
        );
        let policy = self.compile_device().policy().heap_compression;
        if compressed && policy == HeapCompressionPolicy::Compressed {
            return Ok(HeapImage {
                plan: optimal,
                tiling: vk::ImageTiling::OPTIMAL,
                requirements,
                route: AliasRoute::Compressed,
            });
        }
        if let Some(image) = self.host_transfer_image(&optimal, texture_type, &takes)? {
            return Ok(image);
        }
        match self.linear_heap_image(linear, texture_type, &takes, AliasRoute::Linear) {
            Err(error) if linear_refusal(&error) => {}
            made => return made,
        }
        Ok(HeapImage {
            plan: optimal,
            tiling: vk::ImageTiling::OPTIMAL,
            requirements,
            route: AliasRoute::Hidden,
        })
    }

    /// `optimal` with host-transfer usage, on the alias route of [`AliasRoute::HostTransfer`],
    /// where the device enabled `hostImageCopy` (`device/create.rs`, `device/adopt.rs`), the
    /// format takes the usage at optimal tiling, and the device makes such an image of the plan
    /// whose memory types `takes` the heap's; `None` elsewhere.
    fn host_transfer_image(
        &self,
        optimal: &ImagePlan,
        texture_type: TextureType,
        takes: &dyn Fn(u32) -> bool,
    ) -> Result<Option<HeapImage>, Error> {
        let shared = self.shared();
        let facts = shared.textures.facts(self.compile_device().profile());
        let transfer = FormatFeatureFlags2::HOST_IMAGE_TRANSFER;
        let keeps = (self.properties().driver_facts)
            .contains(&DriverFact::HostTransferImagesKeepTheirBytes);
        if !shared.host_image_copy || !keeps || !facts.optimal(optimal.format).contains(transfer) {
            return Ok(None);
        }
        let mut plan = optimal.clone();
        plan.usage |= ImageUsageFlags::HOST_TRANSFER;
        let requirements =
            match self.planned_requirements(&plan, texture_type, vk::ImageTiling::OPTIMAL) {
                Ok(requirements) => requirements,
                // The device makes no image of the plan's shape with the usage.
                Err(Error::Unsupported(_)) => return Ok(None),
                Err(error) => return Err(error),
            };
        if !takes(requirements.memory_type_bits) {
            return Ok(None);
        }
        Ok(Some(HeapImage {
            plan,
            tiling: vk::ImageTiling::OPTIMAL,
            requirements,
            route: AliasRoute::HostTransfer,
        }))
    }

    /// `linear`, a heap image of `route`, where the device's linear images can do all the
    /// optimal one can for the texture (extended usage included) and take the heap's memory
    /// (`takes`); else refused as the device's lack (`Unsupported(Feature::HeapTextureTiling)`).
    fn linear_heap_image(
        &self,
        linear: ImagePlan,
        texture_type: TextureType,
        takes: &dyn Fn(u32) -> bool,
        route: AliasRoute,
    ) -> Result<HeapImage, Error> {
        let facts = self
            .shared()
            .textures
            .facts(self.compile_device().profile());
        if let Some(LinearLack { format, features }) = linear_lack(&linear, &facts) {
            return Err(tiling_refusal(LinearNeed::Features { format, features }));
        }
        let requirements =
            self.planned_requirements(&linear, texture_type, vk::ImageTiling::LINEAR)?;
        if !takes(requirements.memory_type_bits) {
            return Err(tiling_refusal(LinearNeed::MemoryType));
        }
        Ok(HeapImage {
            plan: linear,
            tiling: vk::ImageTiling::LINEAR,
            requirements,
            route,
        })
    }

    /// The memory requirements of an image of `plan` (of Metal's `texture_type`) and `tiling`;
    /// `Unsupported` where the device makes no such image: for a linear one,
    /// `HeapTextureTiling`, naming what the device's linear images lack. The answer for an image
    /// of the same plan and tiling asked before where Vulkan fixes it (`texture/requirements.rs`),
    /// with no other query: it was kept only once the device's limits took the image. Else the
    /// device's limits for the image are asked (`vkGetPhysicalDeviceImageFormatProperties2`), then
    /// an image is made for the query (`WorkCounts::requirement_images_made`).
    fn planned_requirements(
        &self,
        plan: &ImagePlan,
        texture_type: TextureType,
        tiling: vk::ImageTiling,
    ) -> Result<vk::MemoryRequirements, Error> {
        let answers = &self.shared().textures.requirements;
        if let Some(requirements) = lock(answers).answer(plan, tiling) {
            return Ok(requirements);
        }
        let view_formats = plan.ash_view_formats();
        let mut list = vk::ImageFormatListCreateInfo::default();
        let info = plan
            .create_info(self.shared().sharing(), &view_formats, &mut list)
            .tiling(tiling)
            .initial_layout(initial_layout(tiling));
        let limits = self.image_limits(&info, &view_formats)?;
        self.shared().counts.requirement_limits_asked();
        if tiling == vk::ImageTiling::LINEAR {
            check_linear_limits(plan, &limits).map_err(tiling_refusal)?;
        } else {
            check_image_limits(plan, texture_type, &limits).map_err(Error::Unsupported)?;
        }
        let raw_device = self.shared().vk();
        // SAFETY: the device is live and `info` (with its format list) valid and within the
        // device's limits for it.
        let raw = unsafe { crate::creation::image(raw_device, &info) }?;
        self.shared().counts.requirement_image_made();
        // SAFETY: `raw` is a live image of this device, which nothing else uses: it is destroyed
        // right after the query, with no memory bound.
        let requirements = unsafe {
            let requirements = raw_device.get_image_memory_requirements(raw);
            release::object(raw_device, raw);
            requirements
        };
        lock(answers).keep(plan, tiling, requirements);
        Ok(requirements)
    }

    /// A texture of `descriptor` in `heap`: at `offset` of a placement heap, or where an
    /// automatic heap has room when `None`. Where a placement heap makes none (`Nil` or
    /// `OutOfMemory`) at an offset inside it, the device says so ([`HEAP_TEXTURE_REFUSED`]).
    pub(crate) fn new_heap_texture(
        &self,
        heap: &Heap,
        descriptor: &TextureDescriptor,
        offset: Option<u64>,
    ) -> Result<Texture, Error> {
        let mut planned = None;
        let made = self.planned_heap_texture(heap, descriptor, offset, &mut planned);
        if let (Err(e @ (Error::Nil(_) | Error::OutOfMemory(_))), Some(offset)) = (&made, offset) {
            if offset < heap.size() {
                let text = refused_text(self, heap, descriptor, offset, planned, e);
                self.inform(HEAP_TEXTURE_REFUSED, text);
            }
        }
        made
    }

    /// [`Device::new_heap_texture`], noting in `planned` what the making planned as it plans it.
    fn planned_heap_texture(
        &self,
        heap: &Heap,
        descriptor: &TextureDescriptor,
        offset: Option<u64>,
        planned: &mut Option<Planned>,
    ) -> Result<Texture, Error> {
        let (call, heap_type) = match offset {
            Some(_) => ("new_texture_at_offset", HeapType::Placement),
            None => ("new_texture", HeapType::Automatic),
        };
        let bytes = heap.check_type(call, heap_type)?.clone();
        // A detach waits for none being made in the heap (`detach.rs`), nor does a rebind of it
        // while it is detached, so whether it is holds while it is made.
        let _making = bytes.making_texture();
        // The descriptor first: what plain Metal answers nil for (a format of which it makes no
        // texture alone, `Unspecialized`) comes before the heap's storage mode and
        // `framebufferOnly`, which Metal stops for another format.
        let layout = match self.heap_layout(call, descriptor, Creator::Device) {
            // A placement heap makes a two-plane extended-range cube or cube array, which the
            // device and an automatic heap answer nil for: the runtime's gap.
            Err(Error::Nil(Nil {
                problem: Problem::Texture(TextureProblem::TypeForFormat),
                ..
            })) if heap_type == HeapType::Placement => {
                let pf = descriptor.pixel_format;
                return Err(Error::Unsupported(
                    Unsupported::not_implemented(Feature::Format).naming_format(pf),
                ));
            }
            layout => layout?,
        };
        let options = heap.placed_options(call, descriptor.resource_options)?;
        // A heap makes no framebuffer-only texture (plain Metal asserts on the rotation, the
        // consumer's report and our probe).
        if descriptor.framebuffer_only {
            return Err(invalid(
                call,
                Problem::Texture(TextureProblem::FramebufferOnly),
            ));
        }
        if layout.texture_type == TextureType::TextureBuffer {
            let size_and_align = texture_buffer_size_and_align(&layout)?;
            *planned = Some(Planned {
                size_and_align,
                image: None,
            });
            self.check_texture_placement(
                call,
                descriptor.placement,
                descriptor.usage,
                layout.texture_type,
            )?;
            let offset = heap_offset(heap, call, size_and_align, offset)?;
            // The buffer's drop gives an automatic heap's range back.
            let buffer = heap.placed_buffer(&bytes, offset, size_and_align.size, options);
            let id_use = IdUse::Program(descriptor.placement);
            let texture =
                texture_buffer_over(buffer, descriptor, &layout, id_use, TexelsHome::Heap)?;
            heap.hold_texture(&texture);
            return Ok(texture);
        }
        let p = self.heap_plan(call, descriptor, layout)?;
        *planned = Some(Planned::of(&p));
        self.check_texture_placement(
            call,
            descriptor.placement,
            descriptor.usage,
            p.layout.texture_type,
        )?;
        if bytes.caller_memory() == Some(CallerMemory::HostPages) {
            let p = self.host_heap_plan(call, p, descriptor, &bytes)?;
            *planned = Some(Planned::of(&p));
            let offset = heap_offset(heap, call, p.size_and_align, offset)?;
            let place = HeapPlace::new(heap, offset, p.size_and_align.size, false);
            return self.heap_texture_at((call, heap), &bytes, p, place, options, descriptor);
        }
        let (_, _, type_index) = bytes.memory();
        let images = std::iter::once(&p.image).chain(p.twin.as_ref().map(|(twin, _)| twin));
        for image in images {
            // A shadowed or unaliased image lies in memory of its own.
            let own_memory = matches!(
                image.route,
                AliasRoute::MemoryShadow
                    | AliasRoute::Unaliased
                    | AliasRoute::Planes
                    | AliasRoute::Lossless
                    | AliasRoute::Tiled
            );
            if !own_memory && image.requirements.memory_type_bits & (1 << type_index) == 0 {
                return Err(image.memory_refusal());
            }
        }
        let offset = heap_offset(heap, call, p.size_and_align, offset)?;
        // A placement heap's texture answers aliasable from the start, as Metal's does.
        let aliasable = heap.heap_type() == HeapType::Placement;
        let place = HeapPlace::new(heap, offset, p.size_and_align.size, aliasable);
        self.heap_texture_at((call, heap), &bytes, p, place, options, descriptor)
    }

    /// The plan of a texture of `descriptor` in a heap over the caller's host memory whose own
    /// buffer is `bytes` (`Device::new_heap_with_bytes_no_copy`), from `p`, its plan for an
    /// ordinary `Shared` heap: the size and alignment the device answers
    /// (`heap_texture_size_and_align`), which placed it, with the image that takes the heap's
    /// imported memory: its linear image, made for host allocations, where it does all the
    /// optimal one can, the device imports host memory for it at an offset (no dedicated
    /// allocation), the heap's memory type is one it takes and its requirements fit the answer;
    /// else a shadow of the heap's bytes (`AliasRoute::MemoryShadow`), where Vulkan copies the
    /// image to and from them (one sample) and its planes fit the answered size, as `p`'s shadow
    /// already is. An optimal image bound to imported host memory is no route: drivers lay it apart
    /// from the memory (MoltenVK's lies in a texture of its own, its imported memory being a
    /// buffer), or ask a dedicated allocation (RADV). Elsewhere, and for a texture with a depth
    /// twin, `NotImplemented(HeapTextureMemory)`.
    ///
    /// Over pages imported as dma-bufs (`dma_buf.rs`, where the device refused their pointer),
    /// where the linear image would be: an image of the driver's `DRM_FORMAT_MOD_LINEAR` layout at
    /// the texture's offset, bound to a dedicated import of a dma-buf of its own pages, where its
    /// bytes and alignment fit the answer ([`dma_buf_heap_plan`](Self::dma_buf_heap_plan)); else,
    /// and where it is not made after all as the texture is, the shadow, counted and announced.
    ///
    /// A texture of a plain format Metal lays out tiled (`HeapLayout::Tiled`) is a shadow of its
    /// elements tiled in the heap's bytes (`AliasRoute::Tiled`, `heap_tiled.rs`), where its optimal
    /// image takes copies and the moves fit the device; elsewhere it takes the routes above,
    /// announced untiled when it is made.
    ///
    /// A multisample texture is a shadow of its samples in Metal's tiles (`AliasRoute::Tiled`, a
    /// compressed one's uncompressed tiles, [`sample_tiles`](Self::sample_tiles)), where one
    /// serves; elsewhere its optimal image in memory of its own (`AliasRoute::Lossless`), announced
    /// when it is made (`Untiled::LOSSLESS_MULTISAMPLE`, or untiled).
    fn host_heap_plan(
        &self,
        call: &'static str,
        mut p: HeapPlan,
        descriptor: &TextureDescriptor,
        bytes: &Buffer,
    ) -> Result<HeapPlan, Error> {
        if p.twin.is_some() {
            return Err(Error::NotImplemented(RuntimeGap::HeapTextureMemory));
        }
        if p.layout.samples > 1 {
            // A shadow of its samples in Metal's tiles serves it over host pages as in any heap.
            if p.image.route == AliasRoute::Tiled {
                return Ok(p);
            }
            let facts = self
                .shared()
                .textures
                .facts(self.compile_device().profile());
            let (plan, _) = plan_images(&p.layout, descriptor.image_usage(), false, &facts)
                .map_err(|refusal| image_error(call, refusal))?;
            let plan = self.with_free_usage(plan, p.layout.texture_type)?;
            let answer = Some(p.size_and_align);
            let shadow = self.sample_tiles(&p.layout, &plan, &p.metal_layout, answer)?;
            if let Some((plan, tiles, _)) = shadow {
                let image = HeapImage {
                    plan,
                    tiling: vk::ImageTiling::OPTIMAL,
                    requirements: p.image.requirements,
                    route: AliasRoute::Tiled,
                };
                return Ok(HeapPlan {
                    image,
                    metal_layout: tiles,
                    ..p
                });
            }
            let image = HeapImage {
                plan,
                tiling: vk::ImageTiling::OPTIMAL,
                requirements: p.image.requirements,
                route: AliasRoute::Lossless,
            };
            return Ok(HeapPlan { image, ..p });
        }
        // A shadow of Metal's tiles serves it over host pages as in any heap.
        if p.image.route == AliasRoute::Tiled {
            return Ok(p);
        }
        if let HeapLayout::Tiled(tiled) = &p.metal_layout {
            match self.tiled_heap_image(call, &p, tiled, descriptor)? {
                Some(image) => return Ok(HeapPlan { image, ..p }),
                None => {
                    let reason = "the moves of its elements pass the device's limits";
                    p.metal_layout = HeapLayout::Untiled(Untiled::other(reason));
                }
            }
        }
        if matches!(
            p.image.route,
            AliasRoute::Shadow
                | AliasRoute::MemoryShadow
                | AliasRoute::Unaliased
                | AliasRoute::Planes
        ) {
            return Ok(p);
        }
        let type_index = bytes.pages_type_index();
        let facts = self
            .shared()
            .textures
            .facts(self.compile_device().profile());
        let (linear, _) = plan_images(&p.layout, descriptor.image_usage(), false, &facts)
            .map_err(|refusal| image_error(call, refusal))?;
        let SizeAndAlign { size, align } = p.size_and_align;
        // A driver that lays depth and stencil images apart from their memory makes no linear
        // one over it (MoltenVK: Metal makes no depth texture over a buffer).
        let driver = &self.properties().driver_facts;
        let apart = !colour(&linear) && driver.contains(&DriverFact::DepthStencilImagesLieApart);
        // Over pages imported as dma-buf memory (`dma_buf.rs`), an image of the driver's
        // `DRM_FORMAT_MOD_LINEAR` layout, bound to a dedicated import of a dma-buf of its own
        // pages as the texture is made (`heap_texture_at`), where it fits the answer; else the
        // shadow below, counted and announced.
        let dma_buf = bytes.over_dma_buf();
        if !apart && dma_buf {
            let image = self.dma_buf_heap_plan(linear, &p, &facts)?;
            return Ok(HeapPlan { image, ..p });
        }
        if !apart && linear_lack(&linear, &facts).is_none() {
            let tiling = vk::ImageTiling::LINEAR;
            if let Some(requirements) = self.host_image_requirements(&linear)? {
                if requirements.memory_type_bits & (1 << type_index) != 0
                    && requirements.alignment <= align
                    && requirements.size <= size
                {
                    let image = HeapImage {
                        plan: linear,
                        tiling,
                        requirements,
                        route: AliasRoute::Direct,
                    };
                    return Ok(HeapPlan { image, ..p });
                }
            }
        }
        let image = self.memory_shadow(linear, &p)?;
        Ok(HeapPlan { image, ..p })
    }

    /// The image of a texture of plan `p` in a heap over pages imported as dma-buf memory
    /// (`dma_buf.rs`), `linear` its plan: an image of the driver's `DRM_FORMAT_MOD_LINEAR` layout
    /// ([`Device::dma_buf_natural_layout`]) at the texture's offset, made and bound to a dedicated
    /// import of a dma-buf of its own pages as the texture is made (`heap_texture_at`), where the
    /// device makes one of the plan and its bytes and alignment fit the heap's answer; else a
    /// shadow of the heap's bytes ([`dma_buf_heap_shadow`](Self::dma_buf_heap_shadow)).
    #[cfg(unix)]
    fn dma_buf_heap_plan(
        &self,
        linear: ImagePlan,
        p: &HeapPlan,
        facts: &crate::image_plan::ImageFacts<'_>,
    ) -> Result<HeapImage, Error> {
        let SizeAndAlign { size, align } = p.size_and_align;
        let why = match self.dma_buf_natural_layout(&linear, facts)? {
            Ok(natural) if natural.size <= size && natural.alignment <= align => {
                return Ok(HeapImage {
                    plan: linear,
                    tiling: vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT,
                    // The optimal image's, which its shadow takes where none is made.
                    requirements: p.image.requirements,
                    route: AliasRoute::Direct,
                });
            }
            Ok(natural) => format!(
                "the driver's DRM_FORMAT_MOD_LINEAR image of it takes {} bytes at an alignment \
                 of {}, past the {size} at {align} the heap answers",
                natural.size, natural.alignment
            ),
            Err(why) => why,
        };
        self.dma_buf_heap_shadow(linear, p, why)
    }

    /// No pages are imported as dma-bufs on this system: the shadow, as before.
    #[cfg(not(unix))]
    fn dma_buf_heap_plan(
        &self,
        linear: ImagePlan,
        p: &HeapPlan,
        _facts: &crate::image_plan::ImageFacts<'_>,
    ) -> Result<HeapImage, Error> {
        self.memory_shadow(linear, p)
    }

    /// The image of a texture of plan `p` whose route is a `DRM_FORMAT_MOD_LINEAR` image over
    /// dma-buf pages ([`dma_buf_heap_plan`](Self::dma_buf_heap_plan)), at `offset` of the heap
    /// whose buffer is `bytes`: made in the driver's layout and bound to a dedicated import of
    /// the dma-buf the pages' source makes of its own pages ([`Device::dma_buf_image`]), with no
    /// lock of the runtime's held. Where none is made, `p` takes the shadow of the heap's bytes
    /// instead ([`dma_buf_heap_shadow`](Self::dma_buf_heap_shadow)), and nothing is answered.
    #[cfg(unix)]
    fn made_dma_buf_heap_image(
        &self,
        bytes: &Buffer,
        p: &mut HeapPlan,
        offset: u64,
    ) -> Result<Option<crate::dma_buf::DmaBufImage>, Error> {
        let pages = bytes.host_pages();
        let facts = self
            .shared()
            .textures
            .facts(self.compile_device().profile());
        let made = match pages.as_deref().and_then(HostPages::dma_buf) {
            None => Err("the heap's bytes lie in no dma-buf pages".to_owned()),
            Some((imports, start, type_index)) => {
                let pages = (&*imports, type_index, start + offset);
                let size = p.size_and_align.size;
                self.dma_buf_heap_image(&p.image.plan, &facts, pages, size)?
            }
        };
        let why = match made {
            Ok(made) => return Ok(Some(made)),
            Err(why) => why,
        };
        p.image = self.dma_buf_heap_shadow(p.image.plan.clone(), p, why)?;
        Ok(None)
    }

    /// The image of a heap texture of `plan` (viewed as `facts` says) in the driver's
    /// `DRM_FORMAT_MOD_LINEAR` layout ([`Device::dma_buf_natural_layout`]), where it takes no more
    /// than the texture's `size` bytes, over the one dma-buf the pages' source makes of its own
    /// pages ([`Device::dma_buf_image`]): `pages` are the heap's imports, their memory type and
    /// the host address of the texture's first byte. Why not, where none is made.
    #[cfg(unix)]
    fn dma_buf_heap_image(
        &self,
        plan: &ImagePlan,
        facts: &crate::image_plan::ImageFacts<'_>,
        (pages, type_index, at): (&crate::dma_buf::DmaBufPages, u32, u64),
        size: u64,
    ) -> Result<Result<crate::dma_buf::DmaBufImage, String>, Error> {
        let natural = match self.dma_buf_natural_layout(plan, facts)? {
            Ok(natural) if natural.size <= size => natural,
            Ok(natural) => {
                let taken = natural.size;
                let why = format!("the driver's image of it takes {taken} bytes, past its {size}");
                return Ok(Err(why));
            }
            Err(why) => return Ok(Err(why)),
        };
        let texels = crate::dma_buf::DmaBufTexels {
            pages,
            type_index,
            at,
            row_pitch: natural.row_pitch,
            slice_pitch: natural.slice_pitch,
        };
        self.dma_buf_image(plan, facts, &texels)
    }

    /// No pages are imported as dma-bufs on this system: never reached.
    #[cfg(not(unix))]
    fn made_dma_buf_heap_image(
        &self,
        _bytes: &Buffer,
        _p: &mut HeapPlan,
        _offset: u64,
    ) -> Result<Option<std::convert::Infallible>, Error> {
        Err(internal(
            "a heap texture over dma-buf pages on a system of none",
        ))
    }

    /// The image of a texture of plan `p` in a heap over host pages that is a shadow of the heap's
    /// bytes in memory of its own (`AliasRoute::MemoryShadow`): `linear`'s plan as an optimal
    /// image with the runtime's free usage, where Vulkan copies it to and from the bytes (one
    /// sample) and its planes fit the answered size; else `NotImplemented(HeapTextureMemory)`.
    fn memory_shadow(&self, linear: ImagePlan, p: &HeapPlan) -> Result<HeapImage, Error> {
        let optimal = self.with_free_usage(linear, p.layout.texture_type)?;
        if !shadowable(&optimal, p.size_and_align.size) {
            return Err(Error::NotImplemented(RuntimeGap::HeapTextureMemory));
        }
        Ok(HeapImage {
            plan: optimal,
            tiling: vk::ImageTiling::OPTIMAL,
            requirements: p.image.requirements,
            route: AliasRoute::MemoryShadow,
        })
    }

    /// [`memory_shadow`](Self::memory_shadow)'s image for a texture of plan `p` in a heap over
    /// pages imported as dma-buf memory, of which no image over a dma-buf of its own pages is made
    /// for `why` (`dma_buf.rs`): counted (`WorkCounts::dma_buf_image_fallbacks`) and announced
    /// once per device (`shadowed-texture`).
    fn dma_buf_heap_shadow(
        &self,
        linear: ImagePlan,
        p: &HeapPlan,
        why: String,
    ) -> Result<HeapImage, Error> {
        let image = self.memory_shadow(linear, p)?;
        self.shared().counts.dma_buf_image_fell_back();
        let layout = &p.layout;
        self.degrade(Degradation::ShadowedTexture, || {
            let [width, height, depth] = layout.extent;
            let object = format!(
                "a {:?} {:?} texture of {width}x{height}x{depth} in a heap",
                layout.pixel_format, layout.texture_type
            );
            let why = format!(
                "the heap's host pages are imported as dma-buf memory, and {why}: its image is \
                 a shadow of the heap's bytes"
            );
            (object, why)
        });
        Ok(image)
    }

    /// The image of a texture of plan `p` laid out as `tiled` in a heap over host pages
    /// (`AliasRoute::Tiled`): its optimal image with the runtime's free usage, in memory of its
    /// own; `None` where that image takes no copies or has more than one sample, the texture's
    /// answered bytes do not hold `tiled`, or the moves of its elements do not fit the device
    /// (`HeapTiledCopies::fit`).
    fn tiled_heap_image(
        &self,
        call: &'static str,
        p: &HeapPlan,
        tiled: &TiledTexture,
        descriptor: &TextureDescriptor,
    ) -> Result<Option<HeapImage>, Error> {
        let facts = self
            .shared()
            .textures
            .facts(self.compile_device().profile());
        let (plan, _) = plan_images(&p.layout, descriptor.image_usage(), false, &facts)
            .map_err(|refusal| image_error(call, refusal))?;
        let optimal = self.with_free_usage(plan, p.layout.texture_type)?;
        let serves = self.tiles_serve(&optimal, tiled, p.size_and_align.size);
        Ok(serves.then_some(HeapImage {
            plan: optimal,
            tiling: vk::ImageTiling::OPTIMAL,
            requirements: p.image.requirements,
            route: AliasRoute::Tiled,
        }))
    }

    /// How a texture of `layout` on `route`, laid out by Metal as `metal`, in a heap not over host
    /// pages lies otherwise than Metal lays it in the heap's bytes, where it does: Metal's own
    /// layout the runtime does not hold (Metal may keep it compressed, its samples, a layout not
    /// measured: `metal`'s reason; `Untiled::LOSSLESS` for one whose uncompressed tiles a shadow
    /// would hold; `Untiled::LOSSLESS_BOUND_SAMPLES` for a multisample one Metal keeps
    /// compressed, its image in the heap's bytes), a multisample texture's samples
    /// (`Untiled::MULTISAMPLE`), its image a shadow of the bytes in rows (`Untiled::ROWS`), or its
    /// image bound there in the driver's layout (`Untiled::DRIVER`). Not a colour image bound there
    /// on a driver whose such images lie as Metal's heap textures where Metal tiles the texture
    /// (`DriverFact::ColorImagesLieAsMetalHeapTextures` of one sample,
    /// `DriverFact::MultisampleColorImagesLieAsMetalHeapTextures` of more). `None` for a route
    /// whose own notice says where it lies (apart, a two-plane texture's planes) and a layout a
    /// measured route holds.
    fn device_heap_layout(
        &self,
        layout: &TextureLayout,
        route: AliasRoute,
        metal: &HeapLayout,
    ) -> Option<Untiled> {
        if matches!(
            route,
            AliasRoute::Apart | AliasRoute::Unaliased | AliasRoute::Planes
        ) {
            return None;
        }
        match metal {
            HeapLayout::Measured => return None,
            // Its image lies in the heap's bytes, not in memory of its own.
            HeapLayout::Untiled(untiled) if *untiled == Untiled::LOSSLESS_MULTISAMPLE => {
                return Some(Untiled::LOSSLESS_BOUND_SAMPLES);
            }
            HeapLayout::Untiled(untiled) => return Some(*untiled),
            HeapLayout::Uncompressed(_) => return Some(Untiled::LOSSLESS),
            HeapLayout::Tiled(_) => {}
        }
        let fact = match layout.samples {
            1 => DriverFact::ColorImagesLieAsMetalHeapTextures,
            _ => DriverFact::MultisampleColorImagesLieAsMetalHeapTextures,
        };
        let as_metal = route == AliasRoute::Direct
            && !is_depth_stencil(layout.pixel_format)
            && self.properties().driver_facts.contains(&fact);
        match route {
            _ if as_metal => None,
            _ if layout.samples > 1 => Some(Untiled::MULTISAMPLE),
            AliasRoute::Shadow | AliasRoute::MemoryShadow => Some(Untiled::ROWS),
            _ => Some(Untiled::DRIVER),
        }
    }

    /// The image of a multisample texture of `layout` held as a shadow of its samples in the
    /// tiles Metal lays them in (`AliasRoute::Tiled`; a compressed one's uncompressed tiles,
    /// [`shadow_tiles`]), where Metal lays it out as `metal` and answers `answer`: `plan` with the
    /// usages the moves take ([`with_sample_moves_usage`](Self::with_sample_moves_usage)), the
    /// tiles, and the answer, which the texture takes; `None` where the tiles are not known, do not
    /// fit the answer, or the device does not move the samples ([`tiles_serve`](Self::tiles_serve)).
    fn sample_tiles(
        &self,
        layout: &TextureLayout,
        plan: &ImagePlan,
        metal: &HeapLayout,
        answer: Option<SizeAndAlign>,
    ) -> Result<Option<(ImagePlan, HeapLayout, SizeAndAlign)>, Error> {
        let level = self.properties().metal_level;
        let (Some(tiles), Some(answer)) = (shadow_tiles(layout, plan.format, level, metal), answer)
        else {
            return Ok(None);
        };
        let plan = self.with_sample_moves_usage(plan, layout.texture_type)?;
        let serves =
            (tiles.tiles()).is_some_and(|tiled| self.tiles_serve(&plan, tiled, answer.size));
        Ok(serves.then_some((plan, tiles, answer)))
    }

    /// `plan` with the usages a multisample image's moves to and from a buffer take
    /// (`sample_interleave.rs`), where the device makes such an image: copies both ways, and a
    /// depth or stencil image's draws, which sample it and render into it; else `plan`.
    fn with_sample_moves_usage(
        &self,
        plan: &ImagePlan,
        texture_type: TextureType,
    ) -> Result<ImagePlan, Error> {
        let mut usage = ImageUsageFlags::TRANSFER_SRC | ImageUsageFlags::TRANSFER_DST;
        if !colour(plan) {
            usage |= ImageUsageFlags::SAMPLED | ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        }
        if plan.usage.contains(usage) {
            return Ok(plan.clone());
        }
        let candidate = ImagePlan {
            usage: plan.usage | usage,
            ..plan.clone()
        };
        Ok(match self.image_requirements(&candidate, texture_type)? {
            Some(_) => candidate,
            None => plan.clone(),
        })
    }

    /// Whether a shadow of Metal's tiles `tiled` serves an image of `plan` whose texture takes
    /// `size` bytes of its heap: an image that takes copies, whose tiles fit those bytes, and
    /// whose moves fit the device (`HeapTiledCopies::fit`); a multisample one, whose samples the
    /// device moves to and from a buffer ([`sample_moves_run`](Self::sample_moves_run)).
    fn tiles_serve(&self, plan: &ImagePlan, tiled: &TiledTexture, size: u64) -> bool {
        let copies = ImageUsageFlags::TRANSFER_SRC | ImageUsageFlags::TRANSFER_DST;
        let (planes, tight_len) = tight_planes(plan);
        plan.usage.contains(copies)
            && tiled.size <= size
            && HeapTiledCopies::fit(self, tiled, &planes, tight_len)
            && (plan.samples == SampleCountFlags::COUNT_1 || self.sample_moves_run(plan))
    }

    /// Whether the device moves the samples of a multisample image of `plan` to and from a buffer
    /// (`HeapTiledCopies::sample_moves`, `sample_interleave.rs`).
    fn sample_moves_run(&self, plan: &ImagePlan) -> bool {
        self.sample_moves_trial(plan).is_some()
    }

    /// Whether the device moves the samples of a multisample image of `plan` to and from a buffer,
    /// and what the moves into the image change (`drawn-depth-copy`), `None` where they are exact:
    /// the moves planned once each way for an image of one pixel of its format, usage and sample
    /// count, and remembered. `None` where it does not; a failure to make what the moves need
    /// answers so: the texture then lies in memory of its own.
    fn sample_moves_trial(&self, plan: &ImagePlan) -> Option<Option<&'static str>> {
        let key = (plan.format, plan.pixel_format, plan.usage, plan.samples);
        let answers = &self.shared().textures.sample_moves;
        if let Some(&(_, trial)) = lock(answers).iter().find(|(known, _)| *known == key) {
            return trial;
        }
        let trial = self.sample_moves_try(plan).unwrap_or(None);
        lock(answers).push((key, trial));
        trial
    }

    /// [`sample_moves_trial`](Self::sample_moves_trial)'s trial: the moves of an image of `plan`'s
    /// format, usage and sample count, one pixel of one slice, planned over no buffer.
    fn sample_moves_try(&self, plan: &ImagePlan) -> Result<Option<Option<&'static str>>, Error> {
        let samples = plan.samples.as_raw();
        let one = ImagePlan {
            extent: [1, 1, 1],
            levels: 1,
            layers: 1,
            image_type: metal2vulkan::vk::ImageType::TYPE_2D,
            ..plan.clone()
        };
        let descriptor = TextureDescriptor {
            texture_type: TextureType::Type2DMultisample,
            pixel_format: plan.pixel_format,
            width: 1,
            height: 1,
            sample_count: samples.into(),
            usage: TextureUsage::SHADER_READ,
            resource_options: ResourceOptions::private(),
            ..TextureDescriptor::default()
        };
        let texture = Device::new_planned_texture(self, &descriptor, one.clone())?;
        let (planes, _) = tight_planes(&one);
        HeapTiledCopies::samples_run(self, &texture, &planes)
    }

    /// The memory requirements of a linear image of `plan` made for host allocations, in the
    /// `UNDEFINED` layout as Vulkan has such an image made; `None` where the device imports no
    /// host memory for such an image (`VkExternalImageFormatProperties`), makes none within its
    /// limits, or asks a dedicated allocation of it (RADV's: an image bound at an offset of the
    /// heap's imported memory is none).
    fn host_image_requirements(
        &self,
        plan: &ImagePlan,
    ) -> Result<Option<vk::MemoryRequirements>, Error> {
        let view_formats = plan.ash_view_formats();
        let mut list = vk::ImageFormatListCreateInfo::default();
        let mut external = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
        let info = plan
            .create_info(self.shared().sharing(), &view_formats, &mut list)
            .tiling(vk::ImageTiling::LINEAR)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut external);
        let Ok(limits) = self.host_image_limits(&info, &view_formats)? else {
            return Ok(None);
        };
        if check_linear_limits(plan, &limits).is_err() {
            return Ok(None);
        }
        let raw_device = self.shared().vk();
        // SAFETY: the device is live and `info` (with its chain) valid and within the device's
        // limits for an image of host allocations.
        let raw = unsafe { crate::creation::image(raw_device, &info) }?;
        let mut dedicated = vk::MemoryDedicatedRequirements::default();
        let mut requirements = vk::MemoryRequirements2::default().push_next(&mut dedicated);
        let query = vk::ImageMemoryRequirementsInfo2::default().image(raw);
        // SAFETY: `raw` is a live image of this device, which nothing else uses: it is destroyed
        // right after the query, with no memory bound. The query is core in Vulkan 1.1.
        unsafe {
            raw_device.get_image_memory_requirements2(&query, &mut requirements);
            release::object(raw_device, raw);
        }
        let requirements = requirements.memory_requirements;
        Ok((dedicated.requires_dedicated_allocation == vk::FALSE).then_some(requirements))
    }

    /// An image of `image`'s plan and tiling bound at `at` of the heap's memory, whose
    /// requirements (the plan's) hold at most `size` bytes there, holding what its bytes held;
    /// `owner` its owner. The device's next submission moves it to the `GENERAL` layout (one over
    /// host pages, the next that names it).
    fn bind_heap_image(
        &self,
        binding: &HeapBinding,
        image: HeapImage,
        at: u64,
        size: u64,
        owner: Owner,
    ) -> Result<Image, Error> {
        let made = self.heap_raw_image(
            &image.plan,
            image.tiling,
            binding.host,
            binding.memory,
            at,
            size,
        )?;
        let (raw, initial) = made.ok_or_else(|| {
            internal("a heap texture's image does not lie where its plan placed it")
        })?;
        let made = Image {
            device: self.clone(),
            raw: Raw::new(raw),
            owner,
            plan: image.plan,
            options: binding.options,
            allow_gpu_optimized_contents: binding.allow_gpu_optimized_contents,
            identity: binding.identity,
            views: Mutex::new(KeyMap::default()),
            own_views: Mutex::new(KeyMap::default()),
            resolves: KeptResolves::new(),
            twin: None,
            decoded: None,
            second: std::sync::OnceLock::new(),
            form: Default::default(),
            kept_outside: AtomicBool::new(false),
            between: OnceLock::new(),
            signed_depth: AtomicBool::new(false),
            detached: Mutex::new(None),
            first_move: Mutex::new(None),
            host_copies: false,
            linear: linear_tiling(image.tiling),
            staging_copies: false,
            objects: ObjectsToken::default(),
        };
        let pages = binding.pages.as_ref().filter(|_| binding.host);
        let first = FirstMove::new(raw, made.plan().format, initial, pages.and_then(pages_hold));
        *lock(&made.first_move) = Some(first);
        Ok(made)
    }

    /// The image of a heap texture bound to the heap's bytes (`AliasRoute::Direct`), made while
    /// they are detached (`detach.rs`): presenting `stand_in` until a rebind gives them pages,
    /// which makes it again over them and swaps it in (`rebound_heap_images`). It has no
    /// `VkImage` of its own meanwhile.
    fn detached_heap_image(
        &self,
        binding: &HeapBinding,
        image: HeapImage,
        owner: Owner,
        stand_in: Arc<StandIn>,
    ) -> Image {
        Image {
            device: self.clone(),
            raw: Raw::new(stand_in.read()),
            owner,
            plan: image.plan,
            options: binding.options,
            allow_gpu_optimized_contents: binding.allow_gpu_optimized_contents,
            identity: binding.identity,
            views: Mutex::new(KeyMap::default()),
            own_views: Mutex::new(KeyMap::default()),
            resolves: KeptResolves::new(),
            twin: None,
            decoded: None,
            second: std::sync::OnceLock::new(),
            form: Default::default(),
            kept_outside: AtomicBool::new(false),
            between: OnceLock::new(),
            signed_depth: AtomicBool::new(false),
            detached: Mutex::new(Some(DetachedImage {
                stand_in,
                own: None,
            })),
            first_move: Mutex::new(None),
            host_copies: false,
            linear: linear_tiling(image.tiling),
            staging_copies: false,
            objects: ObjectsToken::default(),
        }
    }

    /// A `VkImage` of `plan` and `tiling` (made for host allocations where `host`) bound at `at`
    /// of `memory`, whose requirements hold at most `size` bytes there, with the layout it
    /// starts in, for its caller to move to `GENERAL`; `None`, nothing made, where its
    /// requirements do not (its alignment misses `at`, or it takes more bytes).
    fn heap_raw_image(
        &self,
        plan: &ImagePlan,
        tiling: vk::ImageTiling,
        host: bool,
        memory: vk::DeviceMemory,
        at: u64,
        size: u64,
    ) -> Result<Option<(vk::Image, vk::ImageLayout)>, Error> {
        let raw_device = self.shared().vk();
        let view_formats = plan.ash_view_formats();
        let mut list = vk::ImageFormatListCreateInfo::default();
        // An image made for host allocations starts `UNDEFINED`, as Vulkan has it
        // (`VUID-VkImageCreateInfo-pNext-01443`), as an optimal heap image does: its one move to
        // `GENERAL` is at its making, before any command reaches it.
        let initial = if host {
            vk::ImageLayout::UNDEFINED
        } else {
            initial_layout(tiling)
        };
        let mut external = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
        let mut info = plan
            .create_info(self.shared().sharing(), &view_formats, &mut list)
            .tiling(tiling)
            .initial_layout(initial);
        if host {
            info = info.push_next(&mut external);
        }
        // SAFETY: the device is live and `info` (with its chain) valid and within the device's
        // limits for it (`heap_plan`, or `host_heap_plan` for host memory).
        let raw = unsafe { crate::creation::image(raw_device, &info) }?;
        // SAFETY: `raw` is a live image of this device.
        let requirements = unsafe { raw_device.get_image_memory_requirements(raw) };
        if !at.is_multiple_of(requirements.alignment.max(1)) || requirements.size > size {
            // SAFETY: the image is live and unbound, and no command uses it.
            unsafe { release::object(raw_device, raw) };
            return Ok(None);
        }
        // SAFETY: the image is live and unbound; the memory is the heap's, which the texture
        // holds, of a type the image takes (`new_heap_texture`, `host_heap_plan`), at an offset
        // that meets the image's alignment and with its size inside the heap.
        if let Err(error) = unsafe { crate::creation::bind_image(raw_device, raw, memory, at) } {
            // SAFETY: the image is live and unbound, and no command uses it.
            unsafe { release::object(raw_device, raw) };
            return Err(vulkan(error));
        }
        Ok(Some((raw, initial)))
    }

    /// The images of `heap`'s live textures bound to its bytes, re-made over a rebind's new
    /// pages, `pages` (`rebind.rs`): each the same plan, bound at the texture's offset; swapped in
    /// once the rebind is made, the device's next submission that names it moving it to
    /// `GENERAL`. Over pages imported by their pointer, linear and made for host allocations as
    /// `host_heap_plan` made it; over pages imported as dma-bufs (`dma_buf.rs`), the
    /// `DRM_FORMAT_MOD_LINEAR` image `remade` holds for it, made before the rebind settled
    /// ([`Device::dma_buf_images_again`]), which its first move acquires from
    /// `VK_QUEUE_FAMILY_FOREIGN_EXT`. A texture held as a shadow of the heap's bytes, or a texture
    /// buffer, takes none: the one's copies and the other's views read the heap's buffer as they
    /// are recorded or made. Refused (`NotImplemented(RebindTextures)`) where no such image is made
    /// for a texture bound to the bytes: `remade` holds none for it, or the image made for host
    /// allocations does not lie in its place (one made over dma-buf pages before).
    pub(crate) fn rebound_heap_images(
        &self,
        heap: &Heap,
        pages: &Arc<HostPages>,
        remade: &mut Remade,
    ) -> Result<Vec<ImageSwap>, Error> {
        let (memory, base) = pages.memory();
        let dma_buf = pages.over_dma_buf();
        let (textures, _) = heap.textures_from(0);
        let mut swaps: Vec<ImageSwap> = Vec::new();
        for texture in &textures {
            let Storage::Image(image) = &texture.0.storage else {
                continue;
            };
            let Owner::Heap(place) = &image.owner else {
                continue;
            };
            if swaps.iter().any(|swap| swap.is_of(image)) {
                continue;
            }
            if dma_buf {
                let Some(objects) = remade.take(image) else {
                    return Err(Error::NotImplemented(RuntimeGap::RebindTextures));
                };
                let family = self.foreign_acquirer();
                // Dropped on a later failure, the images made so far go.
                swaps.push(ImageSwap::acquired(
                    Arc::clone(image),
                    objects,
                    pages,
                    family,
                ));
                continue;
            }
            // An image over dma-buf pages lies in the layout the driver gives a
            // `DRM_FORMAT_MOD_LINEAR` image, which its linear image need not keep.
            if image.raw.dedicated() {
                return Err(Error::NotImplemented(RuntimeGap::RebindTextures));
            }
            let made = self.heap_raw_image(
                image.plan(),
                vk::ImageTiling::LINEAR,
                true,
                memory,
                base + place.offset,
                place.size,
            )?;
            let Some((raw, initial)) = made else {
                return Err(Error::NotImplemented(RuntimeGap::RebindTextures));
            };
            let first = FirstMove::new(raw, image.plan().format, initial, pages_hold(pages));
            // Dropped on a later failure, the images made so far go.
            let objects = ImageObjects::new(self, raw);
            swaps.push(ImageSwap::new(Arc::clone(image), objects, first, pages));
        }
        Ok(swaps)
    }

    /// The images of `heap`'s live textures bound to its bytes, made for a rebind's new pages,
    /// which it imported as dma-bufs (`imports`, of memory type `type_index`), the heap's bytes at
    /// `pointer`, into `remade`: for each, a `DRM_FORMAT_MOD_LINEAR` image of its plan in the
    /// driver's layout ([`Device::dma_buf_natural_layout`]) at its offset, over the one dma-buf the
    /// pages' source makes of its own pages ([`Device::dma_buf_image`]). With no lock of the
    /// runtime's held, as the source is called once for each. Refused
    /// (`NotImplemented(RebindTextures)`) where the device makes no such image of one.
    #[cfg(unix)]
    pub(super) fn dma_buf_heap_images_again(
        &self,
        heap: &Heap,
        imports: &crate::dma_buf::DmaBufPages,
        type_index: u32,
        pointer: NonNull<u8>,
        remade: &mut Remade,
    ) -> Result<(), Error> {
        let facts = self
            .shared()
            .textures
            .facts(self.compile_device().profile());
        let (textures, _) = heap.textures_from(0);
        for texture in &textures {
            let Storage::Image(image) = &texture.0.storage else {
                continue;
            };
            let Owner::Heap(place) = &image.owner else {
                continue;
            };
            if remade.0.iter().any(|(made, _)| Arc::ptr_eq(made, image)) {
                continue;
            }
            let pages = (imports, type_index, pointer.as_ptr() as u64 + place.offset);
            let made = self.dma_buf_heap_image(image.plan(), &facts, pages, place.size)?;
            let made = made.map_err(|_| Error::NotImplemented(RuntimeGap::RebindTextures))?;
            let objects = ImageObjects::imported(self, made.raw, made.import);
            remade.0.push((Arc::clone(image), objects));
        }
        Ok(())
    }

    /// The texture of plan `p` bound at `place` of its heap, whose buffer is `bytes`, with
    /// `options` (the heap's), in the `GENERAL` layout and holding what its bytes held; its depth
    /// twin, where it has one, inside the same range.
    fn heap_texture_at(
        &self,
        (call, heap): (&'static str, &Heap),
        bytes: &Buffer,
        p: HeapPlan,
        place: HeapPlace,
        options: ResourceOptions,
        descriptor: &TextureDescriptor,
    ) -> Result<Texture, Error> {
        let (memory, base, _) = bytes.memory();
        let at = base + place.offset;
        let SizeAndAlign { size, .. } = p.size_and_align;
        // In a detached heap (`detach.rs`) the texture presents the stand-in of its image's plan
        // from its making: an image bound to the heap's bytes is made only once they lie in
        // pages again (`rebound_heap_images`), one in memory of its own presents it too.
        let stand_in = match bytes.detached() {
            true => Some(self.stand_in(&p.image.plan)?),
            false => None,
        };
        // Both images are the heap buffer's resource to hazard tracking.
        let binding = HeapBinding {
            memory,
            options,
            allow_gpu_optimized_contents: descriptor.allow_gpu_optimized_contents,
            identity: bytes.identity(),
            host: bytes.caller_memory() == Some(CallerMemory::HostPages),
            pages: bytes.host_pages(),
        };
        // Over dma-buf pages the image of the driver's `DRM_FORMAT_MOD_LINEAR` layout is made
        // here, its pages' source called once; where none is, the texture is a shadow of the
        // heap's bytes, as it was before such images.
        let mut p = p;
        let dma_buf_image = match p.image.tiling == vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT {
            true if stand_in.is_none() => {
                self.made_dma_buf_heap_image(bytes, &mut p, place.offset)?
            }
            _ => None,
        };
        let twin = p.twin;
        let route = p.image.route;
        let twin_route = twin.as_ref().map(|(twin, _)| twin.route);
        let host = binding.host;
        let metal_layout = p.metal_layout;
        let tiles = metal_layout.tiles().filter(|_| route == AliasRoute::Tiled);
        let image = if let Some(tiled) = tiles {
            let made = (place, options, descriptor);
            self.heap_tiled_image(bytes, p.image, &p.layout, tiled.clone(), made)?
        } else if matches!(route, AliasRoute::Shadow | AliasRoute::MemoryShadow) {
            // A shadowed image is no depth twin's texture: it has none.
            self.heap_shadow_image(bytes, p.image, &p.layout, place, options, descriptor)?
        } else if route == AliasRoute::Planes {
            self.heap_planes_image(bytes, p.image, &p.layout, place, options, descriptor)?
        } else if matches!(route, AliasRoute::Unaliased | AliasRoute::Lossless) {
            // Neither has a two-plane texture.
            // In a heap over host pages its uses are the heap's, as its other textures' are, so
            // a command that names it takes the heap's pages (a rebind's and a detach's rules,
            // `rebind.rs`, `detach.rs`); elsewhere a key of its own, as it shares no memory.
            let key = match binding.host {
                true => bytes.identity().key,
                false => ResourceKey::new(),
            };
            let identity = Identity {
                key,
                tracked: bytes.identity().tracked,
            };
            let mut own = *descriptor;
            own.resource_options = options;
            let owner =
                |placement, _| Owner::HeapUnaliased(Box::new(HeapUnaliased { placement, place }));
            // A heap's texture holds what its bytes held (the module docs): a compressed
            // multisample texture's image in memory of its own starts as that memory lies, as
            // Vulkan copies no zeros into a multisampled image.
            match route {
                AliasRoute::Lossless => {
                    self.new_image_as_it_lies(p.image.plan, &p.layout, &own, identity, owner)?
                }
                _ => self.new_image_owned(p.image.plan, &p.layout, &own, identity, owner)?,
            }
        } else {
            // The range is the image's from here: its last texture object, or the place's drop
            // on an early failure, gives it back. The image holds the heap's memory it is bound to,
            // never the heap.
            *lock(&place.memory) = bytes.memory_hold();
            let owner = Owner::Heap(Box::new(place));
            let image_size = twin.as_ref().map_or(size, |&(_, offset)| offset);
            let mut image = match (&stand_in, dma_buf_image) {
                (Some(stand_in), _) => {
                    self.detached_heap_image(&binding, p.image, owner, Arc::clone(stand_in))
                }
                #[cfg(unix)]
                (None, Some(made)) => {
                    let pages = binding.pages.as_ref().ok_or_else(|| {
                        internal("a heap texture over dma-buf pages the heap does not lie in")
                    })?;
                    self.dma_buf_texture_image(
                        made,
                        p.image.plan,
                        owner,
                        (binding.options, binding.allow_gpu_optimized_contents),
                        binding.identity,
                        pages,
                    )
                }
                _ => self.bind_heap_image(&binding, p.image, at, image_size, owner)?,
            };
            if let Some((twin, offset)) = twin {
                let twin_size = size - offset;
                let made =
                    self.bind_heap_image(&binding, twin, at + offset, twin_size, Owner::HeapTwin);
                image.twin = Some(Box::new(made?));
                let facts = self
                    .shared()
                    .textures
                    .facts(self.compile_device().profile());
                self.degrade_depth_twin(&p.layout, &facts);
            }
            image
        };
        // An image bound in the heap's own memory, which only the runtime's writes change, takes
        // a decoded twin where its format needs one (`texture/decoded_twin.rs`); one over the
        // caller's host pages, laid out apart from the heap's bytes or detached does not.
        let own = matches!(
            route,
            AliasRoute::Direct | AliasRoute::HostTransfer | AliasRoute::Linear | AliasRoute::Hidden
        );
        let twinned = own && !host && stand_in.is_none();
        // An image in memory of its own presents the stand-in too, keeping its own (its copies
        // took it), before its texture is made.
        if let Some(stand_in) = stand_in {
            if image.twin.is_some() {
                return Err(Error::NotImplemented(RuntimeGap::DetachTextures));
            }
            drop(image.present_stand_in(stand_in));
        }
        self.degrade_alias_route(route, &p.layout);
        if let Some(twin_route) = twin_route {
            self.degrade_alias_route(twin_route, &p.layout);
        }
        let image = match twinned {
            true => self.with_decoded_twin(image, &p.layout, descriptor, false)?,
            false => {
                self.degrade_block_decode(&p.layout);
                image
            }
        };
        let record = ViewRecord::of_descriptor(&p.layout, descriptor);
        let storage = Storage::Image(image.shared());
        let texture = Texture::made(
            self,
            storage,
            record,
            None,
            IdUse::Program(descriptor.placement),
        );
        // A multisample texture's moves between its image and Metal's tiles, which name its
        // image's views, are planned as its copies are recorded (`heap_tiled.rs`), of the texture
        // known now; `sample_tiles` found that the device runs them.
        if route == AliasRoute::Tiled && p.layout.samples > 1 {
            let copies = texture.shadow().and_then(|copies| copies.target());
            let Some(ShadowTarget::Tiled { tiles, .. }) = copies else {
                return Err(internal(
                    "a tiled multisample heap texture without its tiles",
                ));
            };
            let Some(inexact) = self.sample_moves_trial(texture.image_plan()?) else {
                return Err(internal(
                    "a tiled multisample heap texture whose samples the device does not move",
                ));
            };
            tiles.know_samples(self, call, &texture, inexact);
        }
        heap.hold_texture(&texture);
        // It holds what its bytes held: its twin decodes all of it before any command reads it.
        texture.decode_whole()?;
        // Over host pages, each texture laid out otherwise than Metal tiles it is announced, in
        // the program's call that made it, with its ID; and in any heap a multisample texture whose
        // image lies in memory of its own, and a shadow of the uncompressed tiles of a texture
        // Metal keeps compressed. In any other heap, a texture whose bytes there are not Metal's
        // layout (`device_heap_layout`).
        if !host && !matches!(route, AliasRoute::Lossless | AliasRoute::Tiled) {
            if let Some(untiled) = self.device_heap_layout(&p.layout, route, &metal_layout) {
                self.degrade_heap_layout(untiled, texture.object_id());
            }
        } else {
            match metal_layout {
                HeapLayout::Tiled(_) if route == AliasRoute::Tiled => {
                    self.shared().counts.heap_texture_tiled();
                }
                HeapLayout::Uncompressed(_) => {
                    let lossless = match p.layout.samples {
                        1 => Untiled::LOSSLESS,
                        _ => Untiled::LOSSLESS_SAMPLES,
                    };
                    self.degrade_heap_layout(lossless, texture.object_id());
                }
                HeapLayout::Untiled(untiled) => {
                    self.degrade_heap_layout(untiled, texture.object_id());
                }
                HeapLayout::Tiled(_) | HeapLayout::Measured => {}
            }
        }
        Ok(texture)
    }
}

impl Device {
    /// The image of a heap's shadow of `plan` (`heap_shadow_image`, `heap_tiled_image`), of
    /// `identity`, its owner made by `owner`. Where its copies in write every aspect (`every`,
    /// [`copies_every_aspect`]) it starts as its memory lies: each copy in writes every texel of
    /// every aspect before any command reads it (`shadow.rs`), so zeros there would never be read.
    /// Else it starts as a new texture of its storage mode does
    /// ([`new_image_owned`](Self::new_image_owned)), as no copy writes the aspect they leave out (an
    /// `X32_Stencil8` texture's depth).
    fn new_heap_shadow(
        &self,
        plan: ImagePlan,
        layout: &TextureLayout,
        descriptor: &TextureDescriptor,
        identity: Identity,
        every: bool,
        owner: impl FnOnce(Placement, vk::Image) -> Owner,
    ) -> Result<Image, Error> {
        match every {
            true => self.new_image_as_it_lies(plan, layout, descriptor, identity, owner),
            false => self.new_image_owned(plan, layout, descriptor, identity, owner),
        }
    }

    /// The image of a heap texture on [`AliasRoute::Shadow`]: `image`'s plan in memory of its own,
    /// whose texels the encoders copy from the heap's bytes (`bytes`) at `place` and back
    /// (`shadow.rs`), laid out as [`heap_planes`] says; of an identity of its own, of the heap's
    /// hazard tracking, and the heap's `options`, started by [`new_heap_shadow`](Self::new_heap_shadow).
    /// Its drop gives the range back.
    fn heap_shadow_image(
        &self,
        bytes: &Buffer,
        image: HeapImage,
        layout: &TextureLayout,
        place: HeapPlace,
        options: ResourceOptions,
        descriptor: &TextureDescriptor,
    ) -> Result<Image, Error> {
        let plan = image.plan;
        let (planes, length) = heap_planes(
            plan.format,
            plan.pixel_format,
            plan.extent,
            plan.levels,
            plan.layers,
        );
        let (raw, start) = bytes.at(place.offset);
        // A heap over host pages reads its `VkBuffer` as each copy is recorded: a rebind gives it
        // other pages (`rebind.rs`).
        let buffer = ShadowBuffer::of(bytes, raw);
        let (heap_bytes, tracked) = bytes.tracked(place.offset..place.offset + length);
        let regions: Vec<vk::BufferImageCopy> = planes
            .into_iter()
            .map(|region| vk::BufferImageCopy {
                buffer_offset: start + region.buffer_offset,
                ..region
            })
            .collect();
        let every = copies_every_aspect(&regions, plan.format);
        let shadow_range = UseRange::Image {
            aspects: image_aspects(plan.format).into(),
            levels: 0..plan.levels,
            layers: 0..plan.layers,
        };
        let identity = Identity {
            key: ResourceKey::new(),
            tracked: heap_bytes.tracked,
        };
        let mut own = *descriptor;
        own.resource_options = options;
        self.new_heap_shadow(plan, layout, &own, identity, every, |placement, image| {
            let copies = ShadowCopies {
                source: ShadowSource::Bytes {
                    buffer,
                    key: heap_bytes.key,
                    bytes: tracked,
                    target: ShadowTarget::Image {
                        image,
                        regions,
                        staging: None,
                        texel: None,
                        words: None,
                        unorm: None,
                    },
                    // A heap's bytes join no other memory (`host_aliases.rs`).
                    alias: None,
                },
                shadow_key: identity.key,
                shadow_range,
                tracked: identity.tracked,
                drops_writes: false,
                backing: Backing::of(bytes, false),
                merge_scratch: ScratchSpares::default(),
            };
            Owner::HeapShadow(Box::new(HeapShadow {
                placement,
                place,
                copies: Arc::new(copies),
            }))
        })
    }

    /// Metal's layout of the planes of a heap's two-plane texture of `layout` whose image is of
    /// `plan` (its first level's, `heap_two_planes.rs`), where the runtime holds the image as a
    /// shadow of them: a 1D, 2D, 1D array, 2D array or 3D texture of any level count (a 3D one's
    /// first depth slice, over which Metal lays every other), of one sample, whose image takes
    /// copies, whose moves fit the device. `None` for any other (one past the device's limits),
    /// which lies apart from the heap's bytes.
    fn two_plane_heap_layout(
        &self,
        layout: &TextureLayout,
        plan: &ImagePlan,
    ) -> Option<TwoPlaneHeapLayout> {
        use TextureType as T;
        let copies = ImageUsageFlags::TRANSFER_SRC | ImageUsageFlags::TRANSFER_DST;
        let [width, height, _] = layout.extent.map(u64::from);
        let shape = matches!(
            layout.texture_type,
            T::Type1D | T::Type2D | T::Type1DArray | T::Type2DArray | T::Type3D
        );
        let held =
            shape && plan.samples == SampleCountFlags::COUNT_1 && plan.usage.contains(copies);
        let planes = TwoPlaneHeapLayout::of_texture(
            layout.texture_type,
            [width, height],
            u64::from(layout.layers),
            layout.levels,
        );
        (held && HeapPlaneCopies::fit(self, &planes)).then_some(planes)
    }

    /// The image of a heap's two-plane texture on [`AliasRoute::Planes`]: `image`'s plan in
    /// memory of its own, whose texels the encoders move from its two planes in the heap's bytes
    /// (`bytes`) at `place` and back (`shadow.rs`, `heap_two_planes.rs`); of an identity of its
    /// own, of the heap's hazard tracking, and the heap's `options`. It starts as its memory lies,
    /// as each move in writes every texel (its color word and its alpha byte) before any command
    /// reads it. Its drop gives the range back.
    fn heap_planes_image(
        &self,
        bytes: &Buffer,
        image: HeapImage,
        layout: &TextureLayout,
        place: HeapPlace,
        options: ResourceOptions,
        descriptor: &TextureDescriptor,
    ) -> Result<Image, Error> {
        let plan = image.plan;
        let planes = self
            .two_plane_heap_layout(layout, &plan)
            .ok_or_else(|| internal("a two-plane heap texture's planes are not measured"))?;
        let (raw, start) = bytes.at(place.offset);
        let buffer = ShadowBuffer::of(bytes, raw);
        let length = planes.span();
        let (heap_bytes, tracked) = bytes.tracked(place.offset..place.offset + length);
        let hazard = options.hazard_tracking_mode()?;
        let staging = HeapPlaneCopies::staging(self, &planes, start, hazard)?;
        let moves = Box::new(HeapPlaneCopies::new(self, &staging, &planes, hazard)?);
        // The first level's slices; any later level lies in the image alone.
        let shadow_range = UseRange::Image {
            aspects: image_aspects(plan.format).into(),
            levels: 0..1,
            layers: 0..planes.slices as u32,
        };
        let identity = Identity {
            key: ResourceKey::new(),
            tracked: heap_bytes.tracked,
        };
        let pixel_format = layout.pixel_format;
        let mut own = *descriptor;
        own.resource_options = options;
        self.new_image_as_it_lies(plan, layout, &own, identity, |placement, image| {
            let copies = ShadowCopies {
                source: ShadowSource::Bytes {
                    buffer,
                    key: heap_bytes.key,
                    bytes: tracked,
                    target: ShadowTarget::Planes {
                        image,
                        staging,
                        planes: moves,
                        pixel_format,
                    },
                    // A heap's bytes join no other memory (`host_aliases.rs`).
                    alias: None,
                },
                shadow_key: identity.key,
                shadow_range,
                tracked: identity.tracked,
                drops_writes: false,
                backing: Backing::of(bytes, false),
                merge_scratch: ScratchSpares::default(),
            };
            Owner::HeapShadow(Box::new(HeapShadow {
                placement,
                place,
                copies: Arc::new(copies),
            }))
        })
    }

    /// The image of a heap texture on `AliasRoute::Tiled`: `image`'s plan in memory of its own,
    /// whose elements the encoders move from Metal's tiles, laid out as `tiled`, in the heap's
    /// bytes (`bytes`) at `place` and back (`shadow.rs`, `heap_tiled.rs`); of an identity of its
    /// own, of the heap's hazard tracking, and the heap's `options`, started by
    /// [`new_heap_shadow`](Self::new_heap_shadow). Its drop gives the range back.
    fn heap_tiled_image(
        &self,
        bytes: &Buffer,
        image: HeapImage,
        layout: &TextureLayout,
        tiled: TiledTexture,
        (place, options, descriptor): (HeapPlace, ResourceOptions, &TextureDescriptor),
    ) -> Result<Image, Error> {
        let plan = image.plan;
        let samples = plan.samples.as_raw();
        let (planes, tight_len) = tight_planes(&plan);
        let (regions, _) = heap_planes(
            plan.format,
            plan.pixel_format,
            plan.extent,
            plan.levels,
            plan.layers,
        );
        let every = copies_every_aspect(&regions, plan.format);
        if tiled.size > place.size {
            return Err(internal("a tiled heap texture's elements past its range"));
        }
        let (raw, start) = bytes.at(place.offset);
        let buffer = ShadowBuffer::of(bytes, raw);
        let (heap_bytes, tracked) = bytes.tracked(place.offset..place.offset + tiled.size);
        let hazard = options.hazard_tracking_mode()?;
        let staging = HeapTiledCopies::staging(self, &tiled, start, hazard)?;
        let shape = (&planes[..], tight_len, samples);
        let tiles = HeapTiledCopies::new(self, &staging, tiled, shape, &regions, hazard)?;
        let tiles = Box::new(tiles);
        let shadow_range = UseRange::Image {
            aspects: image_aspects(plan.format).into(),
            levels: 0..plan.levels,
            layers: 0..plan.layers,
        };
        let identity = Identity {
            key: ResourceKey::new(),
            tracked: heap_bytes.tracked,
        };
        let mut own = *descriptor;
        own.resource_options = options;
        self.new_heap_shadow(plan, layout, &own, identity, every, |placement, image| {
            let copies = ShadowCopies {
                source: ShadowSource::Bytes {
                    buffer,
                    key: heap_bytes.key,
                    bytes: tracked,
                    target: ShadowTarget::Tiled {
                        image,
                        staging,
                        tiles,
                    },
                    // A heap's bytes join no other memory (`host_aliases.rs`).
                    alias: None,
                },
                shadow_key: identity.key,
                shadow_range,
                tracked: identity.tracked,
                drops_writes: false,
                backing: Backing::of(bytes, false),
                merge_scratch: ScratchSpares::default(),
            };
            Owner::HeapShadow(Box::new(HeapShadow {
                placement,
                place,
                copies: Arc::new(copies),
            }))
        })
    }

    /// Announces, once per device, the alias route a heap texture of `layout` took where it is
    /// other than [`AliasRoute::Direct`].
    fn degrade_alias_route(&self, route: AliasRoute, layout: &TextureLayout) {
        let (kind, how) = match route {
            AliasRoute::Direct | AliasRoute::Lossless => return,
            AliasRoute::Shadow => {
                return self.degrade(Degradation::ShadowedTexture, || {
                    let [width, height, depth] = layout.extent;
                    let object = format!(
                        "a {:?} {:?} texture of {width}x{height}x{depth} in a heap",
                        layout.pixel_format, layout.texture_type
                    );
                    let why = "the device's driver lays depth and stencil images apart from the \
                               memory they are bound to \
                               (DriverFact::DepthStencilImagesLieApart): its image is a shadow \
                               of the heap's bytes"
                        .to_owned();
                    (object, why)
                });
            }
            AliasRoute::MemoryShadow => {
                return self.degrade(Degradation::ShadowedTexture, || {
                    let [width, height, depth] = layout.extent;
                    let object = format!(
                        "a {:?} {:?} texture of {width}x{height}x{depth} in a heap",
                        layout.pixel_format, layout.texture_type
                    );
                    let why = "the device binds no image of the texture to the heap's memory \
                               where the heap places it: its image is a shadow of the heap's bytes"
                        .to_owned();
                    (object, why)
                });
            }
            AliasRoute::Tiled => {
                return self.degrade(Degradation::ShadowedTexture, || {
                    let [width, height, depth] = layout.extent;
                    let object = format!(
                        "a {} {:?} texture of {width}x{height}x{depth} in a heap",
                        layout.pixel_format.name(),
                        layout.texture_type
                    );
                    let why = "Metal tiles its elements in the heap's bytes, which no image of \
                               the device bound there lays out: its image is a shadow of them"
                        .to_owned();
                    (object, why)
                });
            }
            AliasRoute::Planes => {
                let levels = layout.levels > 1;
                if levels || layout.texture_type == TextureType::Type1DArray {
                    self.degrade(Degradation::HeapTwoPlaneLayout, || {
                        let [width, height, depth] = layout.extent;
                        let object = format!(
                            "a {} {:?} texture of {width}x{height}x{depth}, {} levels, {} slices \
                             in a heap",
                            layout.pixel_format.name(),
                            layout.texture_type,
                            layout.levels,
                            layout.layers
                        );
                        let why = match levels {
                            true => {
                                "Metal's paths address its levels past the first at bytes \
                                     that differ by path: its first level's planes lie in the \
                                     heap's bytes, its other levels apart from them"
                            }
                            false => {
                                "Metal's GPU reads every texel of a 1D array at its first \
                                      word: its texels lie in the heap's bytes as Metal's CPU \
                                      copies and GPU writes lay them, and read as written"
                            }
                        };
                        (object, why.to_owned())
                    });
                }
                return self.degrade(Degradation::ShadowedTexture, || {
                    let [width, height, depth] = layout.extent;
                    let object = format!(
                        "a {} {:?} texture of {width}x{height}x{depth} in a heap",
                        layout.pixel_format.name(),
                        layout.texture_type
                    );
                    let why = "its image holds each texel's colour word and alpha byte side by \
                               side, where Metal keeps two planes in the heap's bytes: its image \
                               is a shadow of those planes, which the encoders move into it and \
                               back"
                        .to_owned();
                    (object, why)
                });
            }
            AliasRoute::Unaliased => {
                return self.degrade(Degradation::HeapTwoPlaneTexture, || {
                    let [width, height, depth] = layout.extent;
                    let object = format!(
                        "a {} {:?} texture of {width}x{height}x{depth} in a heap",
                        layout.pixel_format.name(),
                        layout.texture_type
                    );
                    let why = "its image holds each texel's colour word and alpha byte side by \
                               side, eight bytes a texel, more than Metal's two planes take in \
                               the heap: it lies in memory of its own, apart from the heap's \
                               bytes"
                        .to_owned();
                    (object, why)
                });
            }
            AliasRoute::Apart => {
                return self.degrade(Degradation::HeapDepthStencilApart, || {
                    let [width, height, depth] = layout.extent;
                    let object = format!(
                        "a {:?} {:?} texture of {width}x{height}x{depth} in a heap",
                        layout.pixel_format, layout.texture_type
                    );
                    let why = "the device's driver lays depth and stencil images apart from the \
                               memory they are bound to \
                               (DriverFact::DepthStencilImagesLieApart), and Vulkan copies no \
                               multisampled image, or the texels do not fit the texture's \
                               bytes, so no shadow of the heap's bytes holds them"
                        .to_owned();
                    (object, why)
                });
            }
            AliasRoute::HostTransfer => (
                Degradation::HeapTextureHostTransfer,
                "and keeps host-transfer images' texels in it \
                 (DriverFact::HostTransferImagesKeepTheirBytes)",
            ),
            AliasRoute::Linear => (
                Degradation::HeapTextureLinear,
                "and the texture's image takes no host-transfer usage on this device",
            ),
            AliasRoute::Hidden => (
                Degradation::HeapTextureHidesAliasedWrites,
                "and the texture's image takes neither host-transfer usage nor linear tiling on \
                 this device",
            ),
            AliasRoute::Compressed => (
                Degradation::HeapTextureCompressed,
                "and Metal keeps the texture losslessly compressed too, whose image the device's \
                 policy keeps compressed (HeapCompressionPolicy::Compressed)",
            ),
        };
        self.degrade(kind, || {
            let [width, height, depth] = layout.extent;
            let object = format!(
                "a {:?} {:?} texture of {width}x{height}x{depth} in a heap",
                layout.pixel_format, layout.texture_type
            );
            let why = format!(
                "the device's driver may keep a compressed image's texels apart from the memory \
                 it is bound to (DriverFact::CompressedImagesHideAliasedWrites), {how}"
            );
            (object, why)
        });
    }

    /// The memory types a heap's buffer and the images in it take, worked out once from a buffer
    /// and images made for the query and destroyed at once.
    fn heap_types(&self) -> Result<HeapTypes, Error> {
        let textures = &self.shared().textures;
        if let Some(&types) = textures.heap_types.get() {
            return Ok(types);
        }
        let color = vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST;
        let rgba8 = vk::Format::R8G8B8A8_UNORM;
        let optimal = self
            .query_image_types(rgba8, color, vk::ImageTiling::OPTIMAL)?
            .ok_or_else(|| internal("the device makes no optimal RGBA8Unorm image"))?;
        let depth = self
            .query_image_types(
                vk::Format::D16_UNORM,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                vk::ImageTiling::OPTIMAL,
            )?
            .ok_or_else(|| internal("the device makes no optimal D16Unorm image"))?;
        let linear = self
            .query_image_types(
                rgba8,
                vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
                vk::ImageTiling::LINEAR,
            )?
            .unwrap_or(0);
        let types = HeapTypes {
            buffer: self.query_buffer_types()?,
            color: optimal,
            depth,
            linear,
        };
        Ok(*textures.heap_types.get_or_init(|| types))
    }

    /// The memory types a 16x16 2D image of `format`, `usage` and `tiling` takes; `None` where
    /// the device makes no such image (every Vulkan device makes the optimal ones the runtime
    /// asks of it).
    fn query_image_types(
        &self,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        tiling: vk::ImageTiling,
    ) -> Result<Option<u32>, Error> {
        let shared = self.shared();
        let raw_device = shared.vk();
        let info = shared.sharing().image(
            vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D {
                    width: 16,
                    height: 16,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(tiling)
                .usage(usage)
                .initial_layout(initial_layout(tiling)),
        );
        let limits = self.image_limits(&info, &[])?;
        if limits.max_extent.width < 16 || limits.max_extent.height < 16 {
            return Ok(None);
        }
        // SAFETY: the device is live and `info` valid and within the device's limits for it.
        let raw = unsafe { crate::creation::image(raw_device, &info) }?;
        // SAFETY: `raw` is a live image of this device, which nothing else uses: it is destroyed
        // right after the query, with no memory bound.
        let requirements = unsafe {
            let requirements = raw_device.get_image_memory_requirements(raw);
            release::object(raw_device, raw);
            requirements
        };
        Ok(Some(requirements.memory_type_bits))
    }

    /// The memory types the runtime's buffers take: a 16-byte buffer's of their one usage.
    fn query_buffer_types(&self) -> Result<u32, Error> {
        let shared = self.shared();
        let raw_device = shared.vk();
        let info = shared.sharing().buffer(
            vk::BufferCreateInfo::default()
                .size(16)
                .usage(buffer_usage(shared.device_address)),
        );
        // SAFETY: the device is live and `info` valid.
        let raw = unsafe { crate::creation::buffer(raw_device, &info) }.map_err(vulkan)?;
        // SAFETY: `raw` is a live buffer of this device, which nothing else uses: it is
        // destroyed right after the query, with no memory bound.
        let requirements = unsafe {
            let requirements = raw_device.get_buffer_memory_requirements(raw);
            release::object(raw_device, raw);
            requirements
        };
        Ok(requirements.memory_type_bits)
    }

    /// The memory requirements a heap's buffer of `class`, whose own are `own`, is placed by: at a
    /// multiple of [`HEAP_BASE_ALIGNMENT`], in the memory types its textures' images take
    /// ([`HeapTypes::heap_memory`]).
    pub(crate) fn heap_requirements(
        &self,
        own: vk::MemoryRequirements,
        class: MemoryClass,
    ) -> Result<vk::MemoryRequirements, Error> {
        let types = self.heap_types()?;
        let memory_type_bits =
            types.heap_memory(&lock(&self.shared().memory), own.memory_type_bits, class);
        Ok(vk::MemoryRequirements {
            alignment: own.alignment.max(HEAP_BASE_ALIGNMENT),
            memory_type_bits,
            ..own
        })
    }
}
