//! Textures over buffers: `Buffer::new_texture`, its SPI form `Buffer::new_linear_texture`, and the
//! texture buffers `Device::new_texture` makes. A 2D texture or array over a buffer is a shadow
//! image (`shadow.rs`) over these texels.
//!
//! - **Metal's rules** decide which textures a buffer backs and where they lie in it
//!   (`linear_plan.rs`); what Metal stops is refused `Invalid`, and an array whose texels run
//!   past its buffer's length, which Metal makes, `NotImplemented`.
//! - **Texture buffers** are the buffer's bytes ([`Texels`]), viewed as a texel buffer where a
//!   function binds them: in the format the binding plan names, over the range it gives, made at
//!   the first bind that needs it and kept with the texture. Their uses are uses of the buffer's
//!   bytes, so the tracker orders them against every other use of the buffer. A
//!   `Device::new_texture` texture buffer is a texture buffer over a new buffer of the
//!   descriptor's options, as long as its row of texels.
//! - **CPU access** (`replace_region`, `get_bytes`) reads and writes the buffer's bytes where
//!   the texture lies in them, as the buffer's own `contents` would, for every texture over a
//!   buffer.
//! - **A texel buffer view in a format of one kind** (a format the device takes for one kind of
//!   texel buffer and not the other, as NVIDIA takes `R64_UINT` for storage texel buffers only):
//!   the runtime's buffers are made for both kinds, and Vulkan views a buffer only in a format
//!   that serves every usage the view has, so such a view names the one usage it needs
//!   (`VkBufferUsageFlags2CreateInfoKHR`), which `maintenance5` allows ([`view_usage`]).
//! - **Not yet** (`NotImplemented`): that view on a device that did not enable `maintenance5`;
//!   views, blits and mipmaps of textures over buffers.

use super::address::ObjectsToken;
use super::apple_only::AppleOnly;
use super::buffer::{Buffer, ReadApart};
use super::chroma_pairs;
use super::copy_plan::{surface_depth_unit, CopyProblem, CopyUnit, HostLayout, ImageRegion};
use super::degradation::Degradation;
use super::device::lock;
use super::error::{internal, invalid, nil, vulkan, Error, Problem, RuntimeGap};
use super::exit_guard;
use super::key_hash::KeyMap;
use super::linear_plan::{
    is_linear, plan_buffer_texture, plane_rows_share_bytes, BufferFacts, BufferLayout,
    BufferRefusal, BufferTexture, BufferTextureCall, BufferTextureProblem, Placed,
    LINEAR_ALIGNMENT,
};
use super::resource::{Identity, ResourceKey};
use super::rotated::{self, Frame, RotatedCopies};
use super::shadow::{
    Backing, ShadowBuffer, ShadowCopies, ShadowSource, ShadowStaging, ShadowTarget,
};
use super::shadow_copy_back::WordView;
use super::shadow_merge::ScratchSpares;
use super::shadow_merge::UnormView;
use super::texture::surface_two_planes;
use super::texture::{IdUse, Texture};
use super::texture_descriptor::{
    Creator, Refused, TextureDescriptor, TextureLayout, TextureProblem,
};
use super::tracker::{Access, Use, UseRange};
use super::two_planes;
use super::values::{CpuCacheMode, HazardTrackingMode, ResourceOptions, StorageMode};
use super::view_plan::ViewRecord;
use super::ycbcr::{surface_planes, IoSurfaceYCbCr};
use super::Device;
use crate::release;
use ash::vk;
use metal2vulkan::bind::{BoundTexture, BufferRange};
use metal2vulkan::device::{FormatUsage, VkFeature};
use metal2vulkan::error::{Feature, Unsupported};
use metal2vulkan::metal::format::{byte_format, ycbcr, ycbcr_format};
use metal2vulkan::metal::{PixelFormat, TextureSwizzleChannels, TextureType, TextureUsage};
use metal2vulkan::vk::{Format, ImageAspectFlags};
use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};

/// A texture's texels in a buffer's bytes, where Metal lays them out.
pub(super) struct Texels {
    pub(super) buffer: Buffer,
    /// Where the GPU's commands find the texels in the buffer.
    pub(super) layout: BufferLayout,
    /// Where the CPU's `replace_region` and `get_bytes` find them: `layout`, but for an IOSurface
    /// plane whose offset or pitch is not a multiple of 16 (`linear_plan.rs`).
    cpu: BufferLayout,
    /// Where a function reads and writes a texture buffer's texels: `layout`, but from its offset
    /// rounded down to 16 bytes, as plain Metal's functions read and write a texture buffer at an
    /// offset off 16 ([`Texels::texture_buffer`]).
    shader: BufferLayout,
    /// The bytes of one texel.
    texel_bytes: u64,
    pub(super) allow_gpu_optimized_contents: bool,
    /// The texture's resource options, its hazard tracking mode resolved, as `resourceOptions`
    /// reports them: the buffer's, which Metal's texture over a buffer takes, but an IOSurface
    /// plane's descriptor's (`Buffer::new_texture_with_iosurface_plane`). Its hazard tracking
    /// mode is its uses' (`Texels::tracked`).
    options: ResourceOptions,
    /// The texel buffer views bindings needed, by format, kind (storage or uniform) and range,
    /// made at their first bind. A finite set per texture: the formats its plans view it as.
    views: Arc<TexelViews>,
    /// A texture buffer's staging buffer and its copies, where Vulkan views no texels at the
    /// texels' offset in the buffer's `VkBuffer` (`Texels::texture_buffer`): its bindings view
    /// the staging buffer, a shadow the encoders copy the texels into and back (`shadow.rs`).
    staged: Option<Arc<ShadowCopies>>,
    /// What made `buffer` for the texels, where that is not the program.
    pub(super) home: TexelsHome,
    /// A texture buffer's attachment twins, by the pixel format a render pass renders it in
    /// (`Texture::attachment_twin`), made at the first pass that needs each.
    twins: Mutex<HashMap<PixelFormat, Texture>>,
    /// The alpha plane a two-plane extended-range texture over a surface's plane 0 keeps its
    /// alpha bytes in, where Metal's GPU finds them, `layout` holding its color words
    /// (`texture/surface_two_planes.rs`); `None` for every other texture.
    alpha: Option<BufferLayout>,
    /// The texture objects over the texels as a texture buffer's (`Storage::Texels`), and the
    /// caller's token for a texture buffer's range (`Device::new_texture_with_range_token`).
    pub(super) objects: ObjectsToken,
    /// The texels lie in an IOSurface's plane (`Buffer::new_texture_with_iosurface_plane`).
    surface: bool,
}

/// A texture's texel buffer views, by format, kind (storage or uniform) and range; destroyed
/// with it. Over a buffer's host pages, a rebind empties them, the views of the old pages going
/// with those pages (`rebind.rs`), and the next bind makes them again over the new ones.
pub(super) struct TexelViews {
    device: Device,
    views: Mutex<KeyMap<(Format, bool, u64, u64), vk::BufferView>>,
}

impl TexelViews {
    /// The views made so far, for the pages they view to keep: none from now on.
    pub(super) fn park(&self) -> ViewObjects {
        let views = lock(&self.views).drain().map(|(_, view)| view).collect();
        ViewObjects {
            device: self.device.clone(),
            views,
        }
    }
}

impl Drop for TexelViews {
    fn drop(&mut self) {
        // Once the process is exiting, the views are left to its end (`exit_guard`).
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        let raw = self.device.shared().vk();
        // SAFETY: no pending command buffer uses a view: a committed command buffer holds every
        // texture it binds until it completes.
        unsafe {
            for &view in lock(&self.views).values() {
                release::object(raw, view);
            }
        }
    }
}

/// Texel buffer views of host pages a rebind replaced, destroyed with those pages
/// ([`TexelViews::park`]).
pub(super) struct ViewObjects {
    device: Device,
    views: Vec<vk::BufferView>,
}

impl Drop for ViewObjects {
    fn drop(&mut self) {
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        let raw = self.device.shared().vk();
        // SAFETY: the pages that keep them go once every command buffer that used them
        // completed.
        unsafe {
            for &view in &self.views {
                release::object(raw, view);
            }
        }
    }
}

/// Where an IOSurface's plane lies in the buffer that stands for the surface
/// ([`Buffer::new_texture_with_iosurface_plane`], the stand-in for Metal's
/// `newTextureWithDescriptor:iosurface:plane:`, which reads these from the `IOSurfaceRef`).
///
/// Made with [`IoSurfacePlane::new`] and the `with_` methods (the struct is `#[non_exhaustive]`, so
/// a field added later breaks no caller).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IoSurfacePlane {
    /// The plane's first byte in the buffer, whose first byte is the surface's base:
    /// `IOSurfaceGetBaseAddressOfPlane` less `IOSurfaceGetBaseAddress`, or 0 for a surface
    /// without planes.
    pub offset: u64,
    /// The plane's `bytesPerRow`: `IOSurfaceGetBytesPerRowOfPlane`, or `IOSurfaceGetBytesPerRow`
    /// for a surface without planes.
    pub bytes_per_row: u64,
    /// Whether the surface has planes: `IOSurfaceGetPlaneCount` above 0, a surface made with
    /// plane info (`kIOSurfacePlaneInfo`, one plane dictionary or more).
    pub planar: bool,
    /// The value of `kIOSurfaceClientDirection` the surface was made with (an IOSurface
    /// property outside the public SDK's headers), `None` for a surface made without the key. A
    /// surface with it is refused, whatever the value
    /// ([`AppleOnly::SurfaceClientDirection`](super::AppleOnly::SurfaceClientDirection)).
    pub client_direction: Option<u64>,
    /// The surface's facts a two-plane YCbCr texture over it reads (its OSType, its chroma plane,
    /// its YCbCr matrix), `None` where the caller gives none: a texture of such a format is then
    /// refused. A two-plane extended-range texture (`RGB10A8_2P_XR10`, `RGB10A8_2P_XR10_sRGB`)
    /// reads plane 1's offset and `bytesPerRow` from them too (`chroma_offset`,
    /// `chroma_bytes_per_row`). Other formats ignore it.
    pub ycbcr: Option<IoSurfaceYCbCr>,
}

impl IoSurfacePlane {
    /// A plane at `offset` of the buffer standing for the surface, rows `bytes_per_row` apart,
    /// of a surface with planes or not (`planar`), made without `kIOSurfaceClientDirection`,
    /// with no YCbCr facts.
    pub const fn new(offset: u64, bytes_per_row: u64, planar: bool) -> IoSurfacePlane {
        IoSurfacePlane {
            offset,
            bytes_per_row,
            planar,
            client_direction: None,
            ycbcr: None,
        }
    }

    /// This plane of a surface made with `kIOSurfaceClientDirection` of `direction`.
    pub const fn with_client_direction(mut self, direction: u64) -> IoSurfacePlane {
        self.client_direction = Some(direction);
        self
    }

    /// This plane of a surface with `ycbcr`'s facts.
    pub const fn with_ycbcr(mut self, ycbcr: IoSurfaceYCbCr) -> IoSurfacePlane {
        self.ycbcr = Some(ycbcr);
        self
    }
}

/// What made the buffer a texture's texels are the bytes of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TexelsHome {
    /// The program's buffer, or one the device made for a texture buffer
    /// (`newTextureWithDescriptor:`).
    Buffer,
    /// A heap, which placed it for a texture buffer (`texture/heap.rs`): the texture answers the
    /// buffer's heap, offset and aliasing as its own.
    Heap,
    /// The device, for a shareable texture buffer (`newSharedTextureWithDescriptor:`,
    /// `texture/shared.rs`): a texture opened from its handle is the same texels.
    Shared,
}

/// The bytes of one texel of `pf`, the block of its direct Vulkan format (the bytes Metal's
/// format has, for every format the runtime holds exactly); `None` without one.
fn texel_bytes(pf: PixelFormat) -> Option<u64> {
    byte_format(pf)
        .and_then(|format| format.block())
        .map(|block| u64::from(block.size))
}

/// The error of `refusal` in `call`.
fn refusal_error(call: &'static str, refusal: BufferRefusal) -> Error {
    match refusal {
        BufferRefusal::Metal(problem @ BufferTextureProblem::PastAllocation { .. }) => {
            nil(call, Problem::BufferTexture(problem))
        }
        BufferRefusal::Metal(problem) => invalid(call, Problem::BufferTexture(problem)),
        BufferRefusal::TexelsPastLength => {
            Error::NotImplemented(RuntimeGap::TexelsPastBufferLength)
        }
        BufferRefusal::OverlappingRows => Error::NotImplemented(RuntimeGap::OverlappingSurfaceRows),
        BufferRefusal::MemorylessPlane => Error::NotImplemented(RuntimeGap::MemorylessSurfacePlane),
    }
}

impl Buffer {
    /// `newTextureWithDescriptor:offset:bytesPerRow:`: a texture of `descriptor` whose texels
    /// are the buffer's bytes from `offset`, rows `bytes_per_row` apart: a 2D texture or a
    /// texture buffer. The texture has the buffer's resource options, whatever the descriptor
    /// names, as a plain run of Metal's has (Metal's validation alone stops a descriptor's storage
    /// or CPU cache mode other than the buffer's).
    ///
    /// A 2D texture is a linear image aliasing the bytes where the device lays one out as Metal
    /// does, else an image the encoders copy the texels into before the commands that use it
    /// and back after the ones that write it; a texture buffer is the bytes themselves. A texture
    /// buffer may lie at any offset: at one off 16 bytes, which Metal's validation alone stops,
    /// functions read and write it from the offset rounded down to 16, and blits and the CPU at the
    /// offset, as on plain Metal.
    ///
    /// Refused: what Metal stops (`Invalid`, with a [`TextureProblem`](super::TextureProblem)
    /// or a [`BufferTextureProblem`](super::BufferTextureProblem)); a format of which plain
    /// Metal makes no texture alone (`Nil`, `TextureProblem::NoTextureOfFormat` or
    /// `Unspecialized`, once every rule Metal stops passed and before the buffer's own: offset,
    /// row pitch, bytes past the buffer); a format with no exact Vulkan format (a 4:2:2 one among
    /// them, which plain Metal makes over a buffer), and an image the device lacks (`Unsupported`);
    /// a 2D texture of a format the device cannot copy (`NotImplemented`); a swizzle that writes
    /// under `write_swizzle_enabled`, which Metal makes (`NotImplemented(WriteSwizzle)`). A texture
    /// buffer of more than one level is stopped (`BufferTextureProblem::MipmapLevelCount`), as
    /// Metal's validation stops it here.
    pub fn new_texture(
        &self,
        descriptor: &TextureDescriptor,
        offset: u64,
        bytes_per_row: u64,
    ) -> Result<Texture, Error> {
        self.texture_over(
            "new_texture",
            descriptor,
            offset,
            bytes_per_row,
            BufferTextureCall::Texture,
            None,
        )
    }

    /// `newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:` (SPI): as
    /// [`new_texture`](Self::new_texture), and also a 2D array, whose slices Metal lays out
    /// itself: each starts `bytes_per_row` × height, rounded up to 128 bytes, after the one
    /// before. The bytes-per-image argument is ignored, as plain Metal ignores it: its
    /// validation alone asserts that it is a multiple of 128.
    ///
    /// Refused as [`new_texture`](Self::new_texture) is; also an array whose texels run past
    /// the buffer's length where Metal makes it (`NotImplemented`); an array whose slices end past
    /// the buffer's `allocatedSize` (`Nil`, as Metal answers nil).
    pub fn new_linear_texture(
        &self,
        descriptor: &TextureDescriptor,
        offset: u64,
        bytes_per_row: u64,
        _bytes_per_image: u64,
    ) -> Result<Texture, Error> {
        self.texture_over(
            "new_linear_texture",
            descriptor,
            offset,
            bytes_per_row,
            BufferTextureCall::Linear,
            None,
        )
    }

    /// The stand-in for `newTextureWithDescriptor:iosurface:plane:` over the caller's memory: a
    /// 2D texture of `descriptor` whose texels are `plane`'s, the buffer standing for the
    /// IOSurface (a no-copy buffer over the surface's pages, `IOSurfaceGetBaseAddress` for
    /// `IOSurfaceGetAllocSize` bytes, or a buffer holding them). Metal's IOSurface rules apply,
    /// measured on the M3 Max and an M4, not [`new_texture`](Self::new_texture)'s buffer rules:
    ///
    /// - the texture takes and reports the descriptor's resource options, whatever the buffer's
    ///   (its default hazard tracking mode resolves to tracked), as an M4's plain run and its
    ///   validation do for `Shared` and `Managed`. Plain Metal also makes a `Private` one, whose
    ///   GPU and CPU read and write the surface's bytes as a `Shared` one's do, and so does the
    ///   runtime, over a buffer the CPU reaches; Metal's validation stops it;
    /// - a plane of a planar surface takes any offset and any `bytes_per_row` that covers its
    ///   row; a surface without planes, a `bytes_per_row` that is a multiple of 16, at an offset
    ///   that is one too (Metal asserts on another pitch). The row the pitch covers is the
    ///   **texture's** own, whatever the plane holds and whatever the rotation: a rotated
    ///   texture's plane holds the turned frame, and Metal still holds the pitch to the texture's
    ///   unturned row (measured on the M3 Max, plain Metal with no validation layer: a 4x8 plane
    ///   at its tight stride of 16 behind an 8x4 texture ends the process, "bytesPerRow (16) must
    ///   be greater or equal to (32) bytes");
    /// - `get_bytes` and `replace_region` read and write the plane where the surface lays it out:
    ///   from its offset, rows `bytes_per_row` apart;
    /// - the GPU's commands read and write it as Metal's GPU does: from the offset rounded down to
    ///   16, rows `bytes_per_row` rounded down to 16 apart, which are other bytes than the CPU's
    ///   where either is not a multiple of 16;
    /// - under 16 bytes a row, the GPU's rows are 16 bytes apart. There Metal's GPU reads every
    ///   row past the first as memory-dependent bytes; the runtime's reads them where its writes
    ///   put them, a near-equivalent announced once
    ///   ([`Degradation::NarrowSurfaceRows`](super::Degradation::NarrowSurfaceRows));
    /// - where that pitch is below the row, the GPU's rows share bytes: 1 to 15 of them, a whole
    ///   number of texels, between consecutive rows. A plain plane is made, its shadow's copies
    ///   reproducing the sharing with two regions a slice each way, so that a read sees it as
    ///   Metal's does and the shared bytes keep the earlier row's where two of its rows write
    ///   them. Which row Metal leaves there is the order its own command wrote the rows in, which
    ///   follows the command, the GPU pitch and the row count (`linear_plan.rs`'s **Rows that
    ///   overlap**); a kernel, the only writer such a plane takes here, moves its winner with the
    ///   threadgroup shape, so the earlier row's bytes are one of its answers. Announced once by
    ///   [`Degradation::CollidingSurfaceRowWrites`](super::Degradation::CollidingSurfaceRowWrites).
    ///
    /// Refused: a surface made with `kIOSurfaceClientDirection`, SPI
    /// ([`AppleOnly::SurfaceClientDirection`](super::AppleOnly::SurfaceClientDirection)); what
    /// Metal stops (`Invalid`, with a [`TextureProblem`](super::TextureProblem) or a
    /// [`BufferTextureProblem`](super::BufferTextureProblem): a type other than 2D, more than one
    /// level, a pitch below the row, a surface without planes at another pitch or offset, or one
    /// whose rows end past it, which plain Metal asserts on); `Memoryless` storage, which plain
    /// Metal makes and serves on the CPU while its GPU uses fail the command buffer
    /// (`NotImplemented(MemorylessSurfacePlane)`, which carries the measurement); rows that
    /// overlap on Metal's GPU where no shadow of the runtime's holds them
    /// (`NotImplemented(OverlappingSurfaceRows)`: a plane a render pass stores into, a two-plane
    /// YCbCr, two-plane extended-range, 4:2:2, depth or stencil one, and a blit that addresses
    /// such a plane's rows as bytes; `linear_plan.rs`); a plane's own rows or its GPU texels past
    /// the buffer, which Metal's surface holds in its pages
    /// (`NotImplemented(TexelsPastBufferLength)`); and as [`new_texture`](Self::new_texture)
    /// refuses a format or an image.
    ///
    /// A depth or stencil texture (`Depth16Unorm`, `Depth32Float`, `Depth32Float_Stencil8`,
    /// `Stencil8`, `X32_Stencil8`) over a plane of a surface with planes is made as Metal makes
    /// it, one aspect's unit a texel in the plane (its depth, else its stencil; the stencil of
    /// `Depth32Float_Stencil8` is not in the plane and reads 0): its reads, samples, blits and CPU
    /// access take the plane's bytes, and its render passes store it there in Metal's tiled layout,
    /// its elements' bytes alone (`texture/surface_depth.rs`), a pass loading the stencil of
    /// `Depth32Float_Stencil8` loads 0, and its views render as it does. A blit of that stencil is
    /// the runtime's gap (`NotImplemented(SurfaceDepthStencil)`), with what
    /// `texture/surface_depth.rs` lists. Over a surface without planes Metal stops it (`Invalid`,
    /// `BufferTextureProblem::NotLinear`).
    ///
    /// A plane of a 4:2:2 format (`GBGR422`, `BGRG422`: a CoreVideo `2vuy` or `yuvs` surface's)
    /// is made as Metal makes it, 2 bytes a texel, copied as a 4:2:2 texture of the device's is
    /// (`linear_plan.rs`).
    ///
    /// A two-plane YCbCr texture of 8 or 10 bits, converted (`YCBCR8_420_2P`,
    /// `YCBCR10_444_2P`, ...), through the sRGB curve (their `_sRGB` twins) or read as codes
    /// (`RGB8_420_2P`, `RGB10_420_2P`), and the packed 10-bit ones (`YCBCR10_420_2P_PACKED`, ...)
    /// and their `_sRGB` twins, over any plane of a surface of any OSType but the lossless
    /// compressed ones, or over plane 0 of a surface without planes, given the surface's facts
    /// ([`IoSurfacePlane::ycbcr`]), samples the texels Metal reads of the planes, and its copies
    /// and CPU calls move the luma codes, as Metal's do (`ycbcr.rs` states its rules and
    /// refusals: the luma plane the call's, the chroma plane the facts' plane 1 under plane 0 of a
    /// surface with planes, chroma codes 0 under any other, bytes past the buffer zeros). A packed
    /// texture's CPU calls move nothing, as Metal's; its blits with a buffer are refused
    /// (`NotImplemented(Format)`). A render target of a 4:2:0 or 4:2:2 format of an odd width or
    /// height is Metal's nil (`TextureProblem::ExtentForFormat`). Metal makes every two-plane
    /// YCbCr format over a surface's plane; another is the runtime's gap
    /// (`NotImplemented(Format)`).
    ///
    /// An `RGB10A8_2P_XR10` or `RGB10A8_2P_XR10_sRGB` texture over plane 0 of a two-plane
    /// `'b3a8'` surface, given plane 1's offset and `bytesPerRow` in the surface's facts
    /// ([`IoSurfacePlane::ycbcr`]: `chroma_offset`, `chroma_bytes_per_row`), keeps each texel's
    /// color word in plane 0 and its alpha byte in plane 1, where Metal's GPU reads and writes
    /// them; its CPU calls and its blits with a buffer move plane 0's color words, as Metal's do,
    /// and its fills and its blits with another such texture move the surface's bytes as Metal's
    /// do. Without the facts, over another surface or plane, and rotated, it is refused
    /// (`NotImplemented(Format)`, naming why).
    ///
    /// Another format of which plain Metal makes no texture alone
    /// (`RGBA16Float_XR`, `AnyDepth`, `AnyStencil`, `Unspecialized`) answers `Nil`, as Metal
    /// answers nil over a surface's plane, once every rule Metal stops passed
    /// and before the surface's own.
    pub fn new_texture_with_iosurface_plane(
        &self,
        descriptor: &TextureDescriptor,
        plane: IoSurfacePlane,
    ) -> Result<Texture, Error> {
        if plane.client_direction.is_some() {
            return Err(Error::AppleOnly(AppleOnly::SurfaceClientDirection));
        }
        let texture = self.texture_over(
            "new_texture_with_iosurface_plane",
            descriptor,
            plane.offset,
            plane.bytes_per_row,
            BufferTextureCall::IoSurfacePlane {
                planar: plane.planar,
            },
            plane.ycbcr,
        )?;
        // Metal's device takes a sampler ID as it makes a surface's texture, as for a texture of
        // its own (unlike a buffer's: `resource_ids.rs`).
        self.device().take_device_sampler();
        Ok(texture)
    }

    /// The texture of `descriptor` over the buffer that `call` (made through `form`) makes, over
    /// a surface of `surface`'s YCbCr facts where given. Over a heap's buffer it is among the
    /// heap's textures while it lives: `useHeap:` makes every resource the heap holds resident, so
    /// an encoder that used the heap makes it resident (`residency.rs`), and a shadow's texels are
    /// copied in for its commands.
    fn texture_over(
        &self,
        call: &'static str,
        descriptor: &TextureDescriptor,
        offset: u64,
        bytes_per_row: u64,
        form: BufferTextureCall,
        ycbcr_surface: Option<IoSurfaceYCbCr>,
    ) -> Result<Texture, Error> {
        let over = (offset, bytes_per_row);
        let texture = self.make_texture_over(call, descriptor, over, form, ycbcr_surface)?;
        if let Some(heap) = self.heap() {
            heap.hold_texture(&texture);
        }
        Ok(texture)
    }

    /// [`texture_over`](Self::texture_over)'s texture, at `offset` with `bytes_per_row` (`over`).
    fn make_texture_over(
        &self,
        call: &'static str,
        descriptor: &TextureDescriptor,
        (offset, bytes_per_row): (u64, u64),
        form: BufferTextureCall,
        ycbcr_surface: Option<IoSurfaceYCbCr>,
    ) -> Result<Texture, Error> {
        // A detach waits for none being made over the bytes (`detach.rs`), nor does a rebind of
        // them while they are detached, so whether they are holds while it is made.
        let _making = self.making_texture();
        let detached = self.detached();
        // A YCbCr format is made over a surface's planes with their facts (`ycbcr.rs`).
        let asked_format = descriptor.pixel_format;
        let surface = matches!(form, BufferTextureCall::IoSurfacePlane { .. });
        // A two-plane extended-range format reads the surface's plane 1 from the same facts.
        let surface_facts = ycbcr_surface;
        let ycbcr_surface = ycbcr_surface.filter(|_| ycbcr(asked_format) && surface);
        let device = self.device();
        let asked = device.texture_options(call, descriptor)?;
        device.check_texture_placement(call, asked.placement, asked.usage, asked.texture_type)?;
        let options = asked.resource_options;
        let storage = options.storage_mode()?;
        let cache = options.cpu_cache_mode()?;
        let hazard = options.hazard_tracking_mode()?;
        // A plain `newTextureWithDescriptor:offset:bytesPerRow:` takes the buffer's options
        // whatever the descriptor names (probe 9d); an IOSurface plane takes its descriptor's, its
        // default hazard tracking mode tracked, as the M4's plain run and validation do
        // (the plan refuses a memoryless one).
        let options = match form {
            BufferTextureCall::IoSurfacePlane { .. } => {
                let hazard = match hazard {
                    HazardTrackingMode::Default => HazardTrackingMode::Tracked,
                    hazard => hazard,
                };
                ResourceOptions::new(storage, cache, hazard)
            }
            BufferTextureCall::Texture | BufferTextureCall::Linear => self.resource_options(),
        };
        let descriptor = &TextureDescriptor {
            resource_options: options,
            ..asked
        };
        // A texture buffer of more than one level is stopped by `linear_plan`, as Metal's
        // buffer creator stops it, not refused as the device's gap.
        let rules = self.device().properties().texture_rules();
        let checked = match ycbcr_surface {
            Some(_) => descriptor.check_over_surface(rules),
            None => descriptor.check_made(Creator::Buffer, rules),
        };
        // A format of which plain Metal makes no texture alone answers nil over a buffer (the
        // linear SPI's too) and, but for a two-plane YCbCr one, over a surface's plane, once every
        // rule Metal stops passed and before the buffer's or the surface's own. Metal makes every
        // two-plane YCbCr format over a surface's plane (`YCBCR8_420_2P` and `YCBCR10_420_2P` over
        // a 'BGRA' surface's); the runtime makes those `ycbcr.rs` reads with the surface's facts,
        // and the others are its gap.
        let layout = match checked {
            Err(Refused::Nil(TextureProblem::NoTextureOfFormat))
                if surface && ycbcr_format(asked_format).is_some() =>
            {
                return Err(Error::Unsupported(
                    Unsupported::not_implemented(Feature::Format).naming_format(asked_format),
                ));
            }
            checked => checked.map_err(|refused| refused.error(call))?,
        };
        // Only an IOSurface plane takes a framebuffer-only texture: plain Metal asserts
        // "framebufferOnly == NO" over a buffer, or on its rotation.
        if descriptor.framebuffer_only && !surface {
            return Err(invalid(
                call,
                Problem::Texture(TextureProblem::FramebufferOnly),
            ));
        }
        let buffer = BufferFacts {
            length: self.length(),
        };
        let planar = matches!(form, BufferTextureCall::IoSurfacePlane { planar: true });
        let converted = match &ycbcr_surface {
            Some(facts) => Some(surface_planes(
                &layout,
                ([offset, bytes_per_row], planar),
                facts,
                self.length(),
                device,
            )?),
            None => None,
        };
        // Metal makes no render target of a 4:2:0 or 4:2:2 format whose chroma blocks do not
        // tile it (an odd width, or an odd height; plain Metal answers nil, the M3 Max under
        // macOS 27.0.1), and makes any other.
        if let Some((planes, _)) = &converted {
            let halved = planes.codes.subsampling.halved() != [false, false];
            let odd = planes.size.iter().any(|n| n % 2 != 0);
            if halved && odd && descriptor.usage.contains(TextureUsage::RENDER_TARGET) {
                return Err(Refused::Nil(TextureProblem::ExtentForFormat).error(call));
            }
        }
        // A two-plane extended-range texture over a surface's plane 0 keeps its color words in
        // the plane and its alpha bytes in the surface's plane 1 (`texture/surface_two_planes.rs`).
        let alpha_plane = match form {
            BufferTextureCall::IoSurfacePlane { planar }
                if two_planes::holds_planes(layout.pixel_format) =>
            {
                Some(surface_two_planes::alpha_plane(
                    layout.pixel_format,
                    planar,
                    surface_facts,
                )?)
            }
            _ => None,
        };
        // A format no buffer backs is Metal's refusal, whatever its Vulkan format, but for a
        // 4:2:2 one, which a plain run makes over a buffer (its validation asserts; probe 9
        // `fmt.240`, `fmt.241`) and the runtime holds over none: its textures' carrier
        // (`chroma_pairs.rs`) is an image of their own. An IOSurface plane's (a CoreVideo `2vuy` or
        // `yuvs` surface's) is made, as Metal makes it, 2 bytes a texel in the carrier
        // (`linear_plan.rs`).
        let pf = layout.pixel_format;
        let four_two_two = chroma_pairs::is_four_two_two(pf);
        // A depth or stencil texture over a surface's plane (`linear_plan.rs` stops every other
        // buffer's) keeps one aspect's unit a texel there.
        let depth_unit = surface_depth_unit(pf).filter(|_| surface);
        let texel_bytes = match (texel_bytes(pf), &converted) {
            _ if depth_unit.is_some() => depth_unit.map_or(0, |unit| u64::from(unit.bytes)),
            (Some(bytes), _) => bytes,
            // Its texels are the luma plane's codes, a code's bytes a texel; packed codes' units
            // of three (`ycbcr.rs`).
            (None, Some((planes, _))) => u64::from(planes.codes.bytes),
            // Its texels are plane 0's color words (`two_planes.rs`).
            (None, None) if alpha_plane.is_some() => u64::from(two_planes::UNIT.bytes),
            (None, None) if four_two_two && surface => u64::from(chroma_pairs::TEXEL_BYTES),
            (None, None) if !is_linear(pf) && !four_two_two => 0,
            (None, None) => {
                return Err(Error::Unsupported(
                    Unsupported::not_implemented(Feature::Format).naming_format(pf),
                ))
            }
        };
        // A packed format's luma rows are units of three codes, `⌈w / 3⌉` of them a row: the
        // plane is planned as those units.
        let planned = match &converted {
            Some((planes, _)) if planes.codes.packed => {
                let [w, h, d] = layout.extent;
                let across = planes.codes.units(u64::from(w)) as u32;
                TextureLayout {
                    extent: [across, h, d],
                    ..layout
                }
            }
            _ => layout,
        };
        let texture = BufferTexture {
            layout: planned,
            texel_bytes,
            storage,
            offset,
            bytes_per_row,
            call: form,
            usage: descriptor.usage,
        };
        // A rotated texture: its plane holds it turned, as Metal checks it, its CPU access and
        // blits address the plane as the texture's own extent at the plane's rows, and its
        // functions the stored frame (`texture/rotation.rs`).
        // A rotated two-plane YCbCr texture is not measured: the runtime's gap. Nor is a rotated
        // two-plane extended-range one over a surface.
        if (converted.is_some() || alpha_plane.is_some()) && descriptor.rotation != 0 {
            return Err(Error::Unsupported(
                Unsupported::not_implemented(Feature::Format).naming_format(pf),
            ));
        }
        // Over detached bytes the runtime makes no rotated texture, nor one of a surface's two
        // planes (`stand_in.rs`): a two-plane YCbCr texture's converted texels have no stand-in
        // of zero bytes.
        let apart = converted.is_some() || alpha_plane.is_some() || descriptor.rotation != 0;
        if detached && apart {
            return Err(Error::NotImplemented(RuntimeGap::DetachTextures));
        }
        let rotation = match u8::try_from(descriptor.rotation) {
            Ok(0) => None,
            Ok(rotation) => Some(turned_plane(call, &texture, rotation, &buffer)?),
            Err(_) => return Err(internal("a rotation past the descriptor's check")),
        };
        if let Some((rotation, stored)) = rotation {
            if self.device().is_lost() {
                return Err(Error::DeviceLost);
            }
            let rows = Texels::new(
                self.clone(),
                rotated_rows(&texture),
                texel_bytes,
                descriptor,
            );
            let stored = Texels::placed(self.clone(), stored, texel_bytes, descriptor);
            let id_use = IdUse::Program(descriptor.placement);
            return self.device().new_rotated_texture(
                call,
                [rows, stored].map(|texels| texels.with_options(options).over_surface(surface)),
                descriptor,
                (&layout, rotation),
                id_use,
            );
        }
        let placed = plan_buffer_texture(&texture, &buffer)
            .map_err(|refusal| refusal_error(call, refusal))?;
        if self.device().is_lost() {
            return Err(Error::DeviceLost);
        }
        if let Some(alpha) = alpha_plane {
            let texels = Texels::placed(self.clone(), placed, texel_bytes, descriptor)
                .with_options(options)
                .over_surface(surface);
            let id_use = IdUse::Program(descriptor.placement);
            return self
                .device()
                .new_surface_planes_texture(call, texels, descriptor, &layout, id_use, alpha);
        }
        if let Some((planes, span)) = converted {
            let texels = Texels::placed(self.clone(), placed, texel_bytes, descriptor)
                .with_options(options)
                .over_surface(surface);
            let id_use = IdUse::Program(descriptor.placement);
            return self.device().new_converted_texture(
                call,
                texels,
                descriptor,
                &layout,
                id_use,
                (&planes, span),
            );
        }
        if layout.texture_type == TextureType::TextureBuffer {
            let texels = Texels::texture_buffer(self.clone(), placed.gpu, texel_bytes, descriptor)?;
            Ok(texels.texture(descriptor, &layout, IdUse::Program(descriptor.placement)))
        } else {
            let texels = Texels::placed(self.clone(), placed, texel_bytes, descriptor)
                .with_options(options)
                .over_surface(surface);
            let id_use = IdUse::Program(descriptor.placement);
            let texture = self
                .device()
                .new_texture_over_buffer(call, texels, descriptor, &layout, id_use)?;
            if placed.narrow {
                self.degrade_narrow_rows(&layout, &placed);
            }
            if placed.gpu.shared_bytes() != 0 {
                self.degrade_colliding_rows(&layout, &placed);
            }
            Ok(texture)
        }
    }
}

/// The plane of a texture of `rotation` made as `texture` (of a rotated descriptor, over an
/// IOSurface plane): `rotation` and where the turned texels lie, checked as Metal checks them, as
/// a 2D texture of the extent it stores (`rotated::stored_extent`) at the plane's offset and
/// rows, with the pitch held to the texture's own row ([`covers_its_row`]), which the stored
/// extent's plan does not hold it to. Refused as that texture is, and
/// (`NotImplemented(RotatedTexture)`) for a plane whose offset or `bytesPerRow` is off 16 bytes,
/// or a format of a texel size other than 1, 2, 4, 8 or 16 bytes (a 4:2:2 one among them), whose
/// rotation is not measured.
///
/// A rotated plane's `bytesPerRow` must be a multiple of 16 here, so its GPU rows lie at the
/// pitch itself and never share bytes, which no shadow of a turned frame holds
/// (`linear_plan.rs`'s **Rows that overlap**).
///
/// It reads nothing of the buffer but its length, so a test reaches every refusal from the
/// input's own shape with no device — which the pitch rule needs, the texture it refuses being
/// one that ends Metal's process.
pub(super) fn turned_plane(
    call: &'static str,
    texture: &BufferTexture,
    rotation: u8,
    buffer: &BufferFacts,
) -> Result<(u8, Placed), Error> {
    let measured = texture.offset.is_multiple_of(LINEAR_ALIGNMENT)
        && texture.bytes_per_row.is_multiple_of(LINEAR_ALIGNMENT)
        && [1, 2, 4, 8, 16].contains(&texture.texel_bytes)
        && !chroma_pairs::is_four_two_two(texture.layout.pixel_format);
    if !measured {
        return Err(Error::NotImplemented(RuntimeGap::RotatedTexture));
    }
    covers_its_row(texture).map_err(|problem| refusal_error(call, problem.into()))?;
    let [width, height, depth] = texture.layout.extent;
    let [width, height] = rotated::stored_extent(rotation, [width, height]);
    let stored = BufferTexture {
        layout: TextureLayout {
            extent: [width, height, depth],
            ..texture.layout
        },
        ..*texture
    };
    let placed =
        plan_buffer_texture(&stored, buffer).map_err(|refusal| refusal_error(call, refusal))?;
    Ok((rotation, placed))
}

/// Whether `texture`'s `bytesPerRow` covers the **texture's** own packed row, which is Metal's
/// rule for a texture over an IOSurface plane whatever the plane holds and whatever the rotation.
///
/// Measured on the M3 Max under plain Metal with no validation layer: a 4x8 surface at its own
/// tight stride of 16 behind an 8x4 texture ends the process,
/// `_mtlValidateStrideTextureParameters:1938: failed assertion 'Texture Descriptor Validation
/// IOSurface texture: bytesPerRow (16) must be greater or equal to (32) bytes'` — 32 being the
/// texture's row, 8 texels of 4 bytes, not the plane's 16. A rotated texture's plane holds the
/// turned frame, so a quarter turn's plane row is the shorter of the two and
/// [`plan_buffer_texture`]'s own row check, which runs against the stored extent
/// (`Buffer::turned_plane`), passes a pitch Metal asserts on. This rule is the texture's extent
/// alone, so a test reaches it from the input's own shape with no device — which is the only way
/// it can be tested: building the texture Metal asserts on would end the test process.
fn covers_its_row(texture: &BufferTexture) -> Result<(), BufferTextureProblem> {
    let row_bytes = u64::from(texture.layout.extent[0]).saturating_mul(texture.texel_bytes);
    match texture.bytes_per_row >= row_bytes {
        true => Ok(()),
        false => Err(BufferTextureProblem::RowPitch {
            bytes_per_row: texture.bytes_per_row,
            row_bytes,
        }),
    }
}

/// The bytes a rotated texture made as `texture` has its CPU access and blits address: its own
/// extent at the plane's offset and rows, as plain Metal addresses them, which may run past the
/// stored frame's last row. One layout for the GPU and the CPU alike, and its rows never share a
/// byte: [`covers_its_row`] holds the pitch to the texture's own row
/// (`tests::buffer_texture::no_rotated_plane_s_gpu_rows_share_bytes`).
pub(super) fn rotated_rows(texture: &BufferTexture) -> BufferLayout {
    let [width, height, _] = texture.layout.extent.map(u64::from);
    BufferLayout {
        offset: texture.offset,
        row_pitch: texture.bytes_per_row,
        slice_pitch: texture.bytes_per_row.saturating_mul(height),
        row_bytes: width.saturating_mul(texture.texel_bytes),
        rows: height,
        slices: 1,
    }
}

impl Buffer {
    /// Announce `NarrowSurfaceRows` for an IOSurface plane of `layout` placed as `placed`, once
    /// it is made (`linear_plan.rs`).
    fn degrade_narrow_rows(&self, layout: &TextureLayout, placed: &Placed) {
        self.device().degrade(Degradation::NarrowSurfaceRows, || {
            let [width, height, _] = layout.extent;
            let object = format!(
                "{width}x{height} {:?} texture of an IOSurface plane at offset {}, {} bytes a row",
                layout.pixel_format, placed.cpu.offset, placed.cpu.row_pitch
            );
            let reason = format!(
                "Metal's GPU writes its rows {} bytes apart from byte {} and reads every row past \
                 the first as memory-dependent bytes",
                placed.gpu.row_pitch, placed.gpu.offset
            );
            (object, reason)
        });
    }

    /// Announce `CollidingSurfaceRowWrites` for an IOSurface plane of `layout` placed as `placed`,
    /// whose GPU rows share bytes, once it is made (`linear_plan.rs`'s **Rows that overlap**).
    fn degrade_colliding_rows(&self, layout: &TextureLayout, placed: &Placed) {
        self.device()
            .degrade(Degradation::CollidingSurfaceRowWrites, || {
                let [width, height, _] = layout.extent;
                let object = format!(
                    "{width}x{height} {:?} texture of an IOSurface plane at offset {}, {} bytes a \
                     row",
                    layout.pixel_format, placed.cpu.offset, placed.cpu.row_pitch
                );
                let reason = format!(
                    "Metal's GPU reads and writes its rows {} bytes apart from byte {}, {} bytes \
                     of texels each, so consecutive rows share {} bytes, and which row's writes \
                     stand there depends on the order Metal writes them in",
                    placed.gpu.row_pitch,
                    placed.gpu.offset,
                    placed.gpu.row_bytes,
                    placed.gpu.shared_bytes(),
                );
                (object, reason)
            });
    }
}

impl Texture {
    /// The texture a render pass renders into for this texture buffer (or view of one), as an
    /// attachment or a resolve target: a 2D texture of one row in the texture's pixel format
    /// over the same bytes of the same buffer, with an aliasing image or a shadow the encoder
    /// copies in and back (`render/shadows.rs`), made at the first pass that needs it and kept with
    /// the texels. Metal renders into a texture buffer plain, its texels the buffer's bytes and the
    /// bytes around them untouched (a probe, an M3 Max and an M4: `tb_device`, `tb_over_buffer`,
    /// `tb_over_buffer_load`, `tb_view_r32uint`, `tb_resolve`); its API validation stops a texture
    /// buffer made with render target usage ("MTLTextureUsageRenderTarget is not valid for a
    /// texture buffer").
    ///
    /// Refused `NotImplemented(Render)` past the device's largest framebuffer width, where Metal
    /// fails the command buffer (`tb_wide`, 20000 texels: an internal error), and for a texture
    /// buffer at an offset off 16 bytes, whose rendering is not measured (its functions read and
    /// write from the offset rounded down, its blits at the offset); otherwise as a 2D texture over
    /// a buffer is.
    pub(super) fn attachment_twin(&self) -> Result<Texture, Error> {
        const CALL: &str = "render_command_encoder";
        let texels = self
            .texels()
            .ok_or_else(|| internal("an attachment twin of a texture with an image"))?;
        let record = self.record();
        let pixel_format = record.pixel_format;
        texels.twin(pixel_format, || {
            let device = texels.buffer.device();
            let width = u64::from(record.extent()[0]);
            let unmeasured = !texels.layout.offset.is_multiple_of(LINEAR_ALIGNMENT);
            if width > u64::from(device.shared().max_framebuffer[0]) || unmeasured {
                return Err(Error::NotImplemented(RuntimeGap::Render));
            }
            let mut descriptor = TextureDescriptor::texture_2d(pixel_format, width, 1, false);
            descriptor.usage = TextureUsage::RENDER_TARGET;
            descriptor.resource_options = texels.options;
            descriptor.allow_gpu_optimized_contents = texels.allow_gpu_optimized_contents;
            let layout = descriptor
                .check(device.properties().texture_rules())
                .map_err(|refused| refused.error(CALL))?;
            let row_bytes = width.saturating_mul(texels.texel_bytes);
            let placed = BufferLayout {
                offset: texels.layout.offset,
                row_pitch: row_bytes,
                slice_pitch: row_bytes,
                row_bytes,
                rows: 1,
                slices: 1,
            };
            let over = Texels::new(
                texels.buffer.clone(),
                placed,
                texels.texel_bytes,
                &descriptor,
            )
            .with_options(texels.options)
            .over_surface(texels.surface);
            device.new_texture_over_buffer(CALL, over, &descriptor, &layout, IdUse::Runtime)
        })
    }
}

impl Device {
    /// `newTextureWithDescriptor:` of a texture buffer (`layout`, checked from `descriptor`): a
    /// texture buffer over a new buffer of the descriptor's options, as long as its texels;
    /// `newSharedTextureWithDescriptor:` of one where `home` is `Shared`.
    ///
    /// A `Memoryless` descriptor, of which plain Metal makes a texture (an Apple M3 Max under
    /// macOS 27.0.1: a 64-texel `RGBA8Unorm` texture buffer of `ShaderRead` usage answers
    /// `storageMode` 3, its other getters its descriptor's), lies over a `Private` buffer: no
    /// Vulkan buffer takes memory lazily. The texture answers the descriptor's mode, and its uses
    /// are a memoryless texture's (`texture.rs`): a command that uses it fails its command
    /// buffer, as plain Metal's (a dispatch that reads it and a blit out of it, measured), its
    /// views are nil, and its CPU access, at which plain Metal's process faults, is refused.
    /// Announced `memoryless-texture-buffer`: the bytes are memory Metal spends none of.
    pub(super) fn new_texture_buffer(
        &self,
        descriptor: &TextureDescriptor,
        layout: &TextureLayout,
        id_use: IdUse,
        home: TexelsHome,
    ) -> Result<Texture, Error> {
        let row_bytes = texture_buffer_bytes(layout)?;
        let options = descriptor.resource_options;
        let bytes = match options.storage_mode()? {
            StorageMode::Memoryless => {
                self.degrade(Degradation::MemorylessTextureBuffer, || {
                    let object = format!(
                        "a memoryless {:?} texture buffer of {} texels",
                        layout.pixel_format, layout.extent[0]
                    );
                    let why = format!("a Private buffer of its {row_bytes} bytes of texels");
                    (object, why)
                });
                options.with_storage_mode(StorageMode::Private)
            }
            _ => options,
        };
        let buffer = self.new_buffer(row_bytes, bytes)?;
        texture_buffer_over(buffer, descriptor, layout, id_use, home)
    }
}

/// The bytes of one texel of a texture buffer of `layout`, over a buffer of its own (a device's,
/// a shareable texture's or a heap's). Refused (`Unsupported`) for a format without a texel
/// size, which no texture buffer has.
fn texture_buffer_texel(layout: &TextureLayout) -> Result<u64, Error> {
    texel_bytes(layout.pixel_format).ok_or(Error::Unsupported(
        Unsupported::not_implemented(Feature::Format).naming_format(layout.pixel_format),
    ))
}

/// The bytes of a texture buffer of `layout`: its row of texels. Refused as
/// `texture_buffer_texel` refuses the format.
pub(super) fn texture_buffer_bytes(layout: &TextureLayout) -> Result<u64, Error> {
    Ok(u64::from(layout.extent[0]) * texture_buffer_texel(layout)?)
}

/// The texture buffer of `descriptor` (checked as `layout`) over the first bytes of `buffer`, as
/// many as `texture_buffer_bytes` answers, its ID as `id_use` says; `home` made `buffer` for it.
/// Refused as `new_buffer` refuses the texels' staging buffer.
pub(super) fn texture_buffer_over(
    buffer: Buffer,
    descriptor: &TextureDescriptor,
    layout: &TextureLayout,
    id_use: IdUse,
    home: TexelsHome,
) -> Result<Texture, Error> {
    let texel_bytes = texture_buffer_texel(layout)?;
    let row_bytes = u64::from(layout.extent[0]) * texel_bytes;
    let placed = BufferLayout {
        offset: 0,
        row_pitch: row_bytes,
        slice_pitch: row_bytes,
        row_bytes,
        rows: 1,
        slices: 1,
    };
    let mut texels = Texels::texture_buffer(buffer, placed, texel_bytes, descriptor)?;
    texels.home = home;
    // A memoryless texture buffer lies over a `Private` buffer (`Device::new_texture_buffer`) and
    // answers `Memoryless`, as Metal's does, with the CPU cache and hazard tracking modes the
    // buffer resolved: Metal resolves a texture buffer's default tracking to tracked (an M3 Max
    // answers `hazardTrackingMode` 2 for a memoryless one). Every other texture buffer takes its
    // buffer's options whole, which `Buffer::new_texture` has already made the descriptor's.
    if descriptor.resource_options.storage_mode()? == StorageMode::Memoryless {
        let options = texels.options.with_storage_mode(StorageMode::Memoryless);
        texels = texels.with_options(options);
    }
    Ok(texels.texture(descriptor, layout, id_use))
}

impl Texels {
    /// The texels of a texture of `descriptor` that `placed` lays out in `buffer`, `texel_bytes`
    /// each: for the GPU at its GPU layout, for the CPU at its CPU one.
    fn placed(
        buffer: Buffer,
        placed: Placed,
        texel_bytes: u64,
        descriptor: &TextureDescriptor,
    ) -> Texels {
        let mut texels = Texels::new(buffer, placed.gpu, texel_bytes, descriptor);
        texels.cpu = placed.cpu;
        texels
    }

    /// The texels a rotated texture's render passes turn theirs into, these being the texels of
    /// its other uses (`width` by `height` of a 2D texture at its plane's offset and rows): the
    /// plane's rows from the same offset, of the extent Metal stores the texture as at
    /// `rotation`. Both lie on one layout for the GPU and the CPU (the plane's offset and rows
    /// are multiples of 16 bytes, `Buffer::turned_plane`).
    pub(super) fn turned(&self, rotation: u8, [width, height]: [u32; 2]) -> Texels {
        let [width, height] = rotated::stored_extent(rotation, [width, height]);
        let row_bytes = u64::from(width) * self.texel_bytes;
        let layout = BufferLayout {
            row_bytes,
            rows: u64::from(height),
            slice_pitch: self.layout.row_pitch * u64::from(height),
            ..self.layout
        };
        let descriptor = TextureDescriptor {
            allow_gpu_optimized_contents: self.allow_gpu_optimized_contents,
            ..TextureDescriptor::default()
        };
        Texels::new(self.buffer.clone(), layout, self.texel_bytes, &descriptor)
            .with_options(self.options)
            .over_surface(self.surface)
    }

    /// These texels' buffer's bytes `span` as one row of `unit`-byte texels, of these texels'
    /// options: a depth or stencil texture's tiled twin's (`texture/surface_depth.rs`), whose
    /// bytes are Metal's tiles there.
    pub(super) fn tiled_span(&self, span: std::ops::Range<u64>, unit: u64) -> Texels {
        let bytes = span.end - span.start;
        let layout = BufferLayout {
            offset: span.start,
            row_pitch: bytes,
            slice_pitch: bytes,
            row_bytes: bytes,
            rows: 1,
            slices: 1,
        };
        let descriptor = TextureDescriptor {
            allow_gpu_optimized_contents: self.allow_gpu_optimized_contents,
            ..TextureDescriptor::default()
        };
        Texels::new(self.buffer.clone(), layout, unit, &descriptor)
            .with_options(self.options)
            .over_surface(self.surface)
    }

    /// The attachment twin of `pixel_format` kept with the texels (a texture buffer's, or a
    /// rotated texture's), made by `make` at the first pass that needs it, with the twins'
    /// lock not held: a twin over dma-buf pages calls the program's source (`dma_buf.rs`). Where
    /// another pass made one meanwhile, that one is kept and `make`'s goes, with no lock held.
    pub(super) fn twin(
        &self,
        pixel_format: PixelFormat,
        make: impl FnOnce() -> Result<Texture, Error>,
    ) -> Result<Texture, Error> {
        if let Some(twin) = lock(&self.twins).get(&pixel_format) {
            return Ok(twin.clone());
        }
        let made = make()?;
        let kept = (lock(&self.twins).entry(pixel_format))
            .or_insert_with(|| made.clone())
            .clone();
        Ok(kept)
    }

    /// The copies of a rotated texture's image of its pass's frame (`image` of `identity`, of
    /// `extent`) over these texels, the plane's rows that hold it turned at `rotation`
    /// (`rotated.rs`): through a staging buffer of their bytes and the rotation kernel. Refused
    /// as `RotatedCopies::new` refuses them.
    pub(super) fn rotated_copies(
        &self,
        image: vk::Image,
        identity: Identity,
        rotation: u8,
        extent: [u32; 2],
    ) -> Result<ShadowCopies, Error> {
        let layout = &self.layout;
        let (buffer, at) = self.buffer.at_for(layout.offset, ReadApart::Texture);
        let staging = self.staging(layout, at)?;
        let frame = Frame {
            rotation,
            extent,
            texel_bytes: self.texel_bytes,
            row_pitch: layout.row_pitch,
            span: layout.end() - layout.offset,
        };
        let hazard = self.options.hazard_tracking_mode()?;
        let turn = Box::new(RotatedCopies::new(
            self.buffer.device(),
            &staging,
            frame,
            hazard,
        )?);
        let (bytes, tracked) = self.buffer.tracked(layout.offset..layout.end());
        Ok(ShadowCopies {
            source: ShadowSource::Bytes {
                buffer: ShadowBuffer::of(&self.buffer, buffer),
                key: bytes.key,
                bytes: tracked,
                target: ShadowTarget::Rotated {
                    image,
                    staging,
                    turn,
                },
                alias: self.buffer.alias().map(|alias| alias.key),
            },
            shadow_key: identity.key,
            shadow_range: UseRange::Image {
                aspects: vk::ImageAspectFlags::COLOR,
                levels: 0..1,
                layers: 0..1,
            },
            tracked: self.tracked(),
            drops_writes: false,
            backing: self.backing(),
            merge_scratch: ScratchSpares::default(),
            touched: AtomicU64::new(0),
            stood: AtomicU64::new(0),
        })
    }

    /// The bytes of one texel.
    pub(super) fn texel_bytes(&self) -> u64 {
        self.texel_bytes
    }

    /// The texture's resource options (the `options` field), its hazard tracking mode resolved.
    pub(super) fn options(&self) -> ResourceOptions {
        self.options
    }

    /// Whether the texture's uses are tracked: its own hazard tracking mode, the buffer's but for
    /// an IOSurface plane's (whose descriptor's Metal takes). A use of the texels keys on the
    /// buffer's bytes whatever the mode, so a tracked texture's uses order against the buffer's
    /// tracked uses, and an untracked texture's are the program's to order, as Metal's.
    pub(super) fn tracked(&self) -> bool {
        self.options.hazard_tracking_mode().ok() == Some(HazardTrackingMode::Tracked)
    }

    /// These texels as a texture of `options` (its hazard tracking mode resolved) has them.
    fn with_options(self, options: ResourceOptions) -> Texels {
        Texels { options, ..self }
    }

    /// These texels, in an IOSurface's plane where `surface`.
    fn over_surface(self, surface: bool) -> Texels {
        Texels { surface, ..self }
    }

    /// Every byte of the buffer the texels lie in, for the GPU or the CPU: from the first to the
    /// last of either layout, and of the alpha plane where the texels have one.
    pub(super) fn span(&self) -> std::ops::Range<u64> {
        let (gpu, cpu, shader) = (&self.layout, &self.cpu, &self.shader);
        let start = gpu.offset.min(cpu.offset).min(shader.offset);
        let end = gpu.end().max(cpu.end()).max(shader.end());
        match &self.alpha {
            Some(alpha) => start.min(alpha.offset)..end.max(alpha.end()),
            None => start..end,
        }
    }

    /// Whether the texels lie in the buffer: every texture's but a rotated one's whose rows run
    /// past it (`rotated_rows`).
    pub(super) fn reached(&self) -> bool {
        self.span().end <= self.buffer.length()
    }

    /// Whether Metal's GPU reads and writes two of the texels' rows over bytes they share: an
    /// IOSurface plane's at its pitch rounded down to 16, the one shape that does
    /// ([`plane_rows_share_bytes`]). The shadow's copies hold the sharing
    /// ([`plane_copy_regions`]); a blit that would write the rows as bytes does not and is refused
    /// instead, as is a copy between them and an image texture either way, while a blit that reads
    /// them into a buffer is served: a write must settle which of two rows leaves the shared bytes
    /// behind and Metal settles it no one way, while a read settles nothing (`blit_texture.rs`).
    pub(super) fn plane_rows_overlap(&self) -> bool {
        plane_rows_share_bytes(&self.layout, &self.cpu)
    }

    /// The texels of a texture of `descriptor` at `placed` in `buffer`, `texel_bytes` each.
    fn new(
        buffer: Buffer,
        placed: BufferLayout,
        texel_bytes: u64,
        descriptor: &TextureDescriptor,
    ) -> Texels {
        let views = Arc::new(TexelViews {
            device: buffer.device().clone(),
            views: Mutex::new(KeyMap::default()),
        });
        // A rebind of the buffer's host pages empties the views (`rebind.rs`).
        buffer.watch_views(&views);
        Texels {
            layout: placed,
            cpu: placed,
            shader: placed,
            texel_bytes,
            allow_gpu_optimized_contents: descriptor.allow_gpu_optimized_contents,
            options: buffer.resource_options(),
            views,
            staged: None,
            home: TexelsHome::Buffer,
            twins: Mutex::new(HashMap::new()),
            alpha: None,
            objects: ObjectsToken::default(),
            surface: false,
            buffer,
        }
    }

    /// What holds the texels' bytes, as the CPU may write them (`shadow::Backing`).
    pub(super) fn backing(&self) -> Backing {
        Backing::of(&self.buffer, self.surface)
    }

    /// Where the CPU's `replace_region` and `get_bytes` find the texels.
    pub(super) fn cpu_layout(&self) -> BufferLayout {
        self.cpu
    }

    /// The texels' two layouts, `(gpu, cpu)`: where Metal's GPU finds them and where the CPU does
    /// (`linear_plan::Placed`), which together say whether its rows share bytes
    /// ([`plane_rows_share_bytes`]). A row copy of the texels as the buffer's bytes takes both
    /// (`blit_plan::plan_texels_rows_copy`).
    pub(super) fn row_layouts(&self) -> (&BufferLayout, &BufferLayout) {
        (&self.layout, &self.cpu)
    }

    /// The alpha plane of a two-plane extended-range texture over a surface's plane 0, where
    /// Metal's GPU finds it (`texture/surface_two_planes.rs`); `None` for every other texture.
    pub(super) fn alpha(&self) -> Option<&BufferLayout> {
        self.alpha.as_ref()
    }

    /// These texels with the alpha plane `alpha` (a two-plane extended-range texture's over a
    /// surface's plane 0, `texture/surface_two_planes.rs`).
    pub(super) fn with_alpha(self, alpha: BufferLayout) -> Texels {
        Texels {
            alpha: Some(alpha),
            ..self
        }
    }

    /// The texels of a texture buffer of `descriptor` at `placed` in `buffer`, `texel_bytes`
    /// each. A function reads and writes them from their offset rounded down to 16 bytes, as
    /// plain Metal's do where its validation alone stops an offset off 16 (an M4 under macOS
    /// 27.0.1), and blits and the CPU at the offset itself (`layout`). Viewed in place where the
    /// device's `minTexelBufferOffsetAlignment`, which the binding plan holds a view's offset to,
    /// divides the functions' offset in the buffer's `VkBuffer`; else through a staging buffer of
    /// their own, at its start (a no-copy buffer's bytes, or Metal's offset, need not lie at a
    /// multiple of it). Refused as `new_buffer` refuses the staging buffer.
    fn texture_buffer(
        buffer: Buffer,
        placed: BufferLayout,
        texel_bytes: u64,
        descriptor: &TextureDescriptor,
    ) -> Result<Texels, Error> {
        let mut texels = Texels::new(buffer, placed, texel_bytes, descriptor);
        texels.shader = BufferLayout {
            offset: placed.offset - placed.offset % LINEAR_ALIGNMENT,
            ..placed
        };
        let (raw, at) = (texels.buffer).at_for(texels.shader.offset, ReadApart::Texture);
        let device = texels.buffer.device();
        let limits = device.compile_device().profile().limits();
        if at.is_multiple_of(limits.min_texel_buffer_offset_alignment.max(1)) {
            return Ok(texels);
        }
        let staging = texels.staging(&texels.shader, at)?;
        let layout = &texels.shader;
        let shadow = staging.buffer.identity();
        let (bytes, tracked) = texels.buffer.tracked(layout.offset..layout.end());
        let shadow_range = UseRange::Bytes(0..staging.into.size);
        let copies = ShadowCopies {
            source: ShadowSource::Bytes {
                buffer: ShadowBuffer::of(&texels.buffer, raw),
                key: bytes.key,
                bytes: tracked,
                target: ShadowTarget::Buffer(staging),
                alias: texels.buffer.alias().map(|alias| alias.key),
            },
            shadow_key: shadow.key,
            shadow_range,
            tracked: texels.tracked(),
            drops_writes: false,
            backing: texels.backing(),
            merge_scratch: ScratchSpares::default(),
            touched: AtomicU64::new(0),
            stood: AtomicU64::new(0),
        };
        texels.staged = Some(Arc::new(copies));
        let (offset, alignment) = (
            texels.layout.offset,
            limits.min_texel_buffer_offset_alignment,
        );
        texels
            .buffer
            .device()
            .degrade(Degradation::StagedTextureBuffer, || {
                let object = format!("texture buffer over a buffer at offset {offset}");
                let reason = format!(
                    "its texels start at byte {at} of the buffer's VkBuffer, and the device views \
                     texels only at multiples of {alignment} bytes \
                     (minTexelBufferOffsetAlignment)"
                );
                (object, reason)
            });
        Ok(texels)
    }

    /// A texture buffer's staging copies, where its bindings view a staging buffer.
    pub(super) fn staged(&self) -> Option<&Arc<ShadowCopies>> {
        self.staged.as_ref()
    }

    /// The `VkBuffer` the texels' bindings view, and the texels' offset in it.
    fn viewed(&self) -> (vk::Buffer, u64) {
        match self.staged.as_deref().and_then(ShadowCopies::target) {
            Some(ShadowTarget::Buffer(staging)) => (staging.raw(), staging.into.dst_offset),
            _ => (self.buffer).at_for(self.shader.offset, ReadApart::Texture),
        }
    }

    /// The texture buffer of `descriptor` (checked as `layout`) these texels are, its ID as
    /// `id_use` says.
    fn texture(
        self,
        descriptor: &TextureDescriptor,
        layout: &TextureLayout,
        id_use: IdUse,
    ) -> Texture {
        let record = ViewRecord::of_descriptor(layout, descriptor);
        Texture::of_texels(self, record, id_use)
    }

    /// The copies of a shadowed texture over the texels (`shadow.rs`), whose image is `image`
    /// of `identity`, with `aspects`: one region per slice, the slice's rows of texels. Where
    /// the texels start in the buffer's `VkBuffer` at an offset that is not a multiple of their
    /// size, which Vulkan copies texels at (a no-copy buffer's bytes need not start at one), the
    /// copies go through a staging buffer of the texture's ([`ShadowStaging`]). Its copies back
    /// dispatch through `words`, its image's integer view, where given and its one slice is
    /// copied in place (`shadow_copy_back.rs`), and its merges read the texels through `unorm`,
    /// its image's view of its own 8-bit unsigned normalized format, where given so
    /// (`shadow_merge.rs`). Refused as `new_buffer` refuses the staging buffer.
    /// Whether these texels take a device copy on `device` (`texture.rs`, `dirty_pages.rs`): on
    /// a discrete GPU, tracked, over host pages whose memory carries a log, where the shadow's
    /// copy in is one region straight from the texels' place in the buffer's `VkBuffer` (one
    /// slice, rows that share no bytes, a first texel at a multiple of the texel's size there),
    /// the copy in a log decides.
    pub(super) fn device_copy(&self, device: &Device) -> bool {
        let layout = &self.layout;
        // The looks at the pages last: each takes their lock (`WorkCounts::no_copy_lies_read`),
        // which a texture on any other GPU never needs.
        device.properties().kind == crate::DeviceKind::Discrete
            && self.tracked()
            && layout.slices == 1
            && layout.shared_bytes() == 0
            && (self.buffer.at_for(layout.offset, ReadApart::Texture).1)
                .is_multiple_of(self.texel_bytes)
            && (self.buffer.host_pages()).is_some_and(|pages| pages.dirty().is_some())
    }

    pub(super) fn shadow_copies(
        &self,
        image: vk::Image,
        identity: Identity,
        aspects: vk::ImageAspectFlags,
        (words, unorm): (Option<WordView>, Option<UnormView>),
    ) -> Result<ShadowCopies, Error> {
        let layout = &self.layout;
        let (buffer, at) = self.buffer.at_for(layout.offset, ReadApart::Texture);
        // Vulkan copies a depth or stencil aspect at multiples of 4 bytes only.
        let depth = vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL;
        let align = match aspects.intersects(depth) {
            true => 4,
            false => self.texel_bytes,
        };
        let staging = if at.is_multiple_of(align) {
            None
        } else {
            Some(self.staging(layout, at)?)
        };
        let start = staging
            .as_ref()
            .map_or(at, |staging| staging.into.dst_offset);
        // The descriptor's checks bound a texture's extent and slices to `u32`s.
        let slices = layout.slices as u32;
        let (regions, back) = plane_copy_regions(layout, self.texel_bytes, start, aspects);
        // A copy back dispatches, and a merge fetches, over one region copied in place alone.
        let one_region = regions.len() == 1 && slices == 1 && staging.is_none();
        let (bytes, tracked) = self.buffer.tracked(layout.offset..layout.end());
        Ok(ShadowCopies {
            source: ShadowSource::Bytes {
                buffer: ShadowBuffer::of(&self.buffer, buffer),
                key: bytes.key,
                bytes: tracked,
                target: ShadowTarget::Image {
                    words: words.filter(|_| one_region),
                    unorm: unorm.filter(|_| one_region),
                    image,
                    regions,
                    back,
                    staging,
                    texel: Some(self.texel_bytes),
                },
                alias: self.buffer.alias().map(|alias| alias.key),
            },
            shadow_key: identity.key,
            shadow_range: UseRange::Image {
                aspects,
                levels: 0..1,
                layers: 0..slices,
            },
            tracked: self.tracked(),
            drops_writes: false,
            backing: self.backing(),
            merge_scratch: ScratchSpares::default(),
            touched: AtomicU64::new(0),
            stood: AtomicU64::new(0),
        })
    }

    /// The staging buffer of a shadowed texture or a texture buffer over the texels laid out as
    /// `layout`, which start at `at` of the buffer's `VkBuffer`: the texels' bytes from the first
    /// to the last, at its start, of the texture's hazard tracking, in a buffer of whole 4-byte
    /// words (the rotation kernel's, `rotated.rs`, binds them all).
    fn staging(&self, layout: &BufferLayout, at: u64) -> Result<ShadowStaging, Error> {
        let span = layout.end() - layout.offset;
        let hazard = self.options.hazard_tracking_mode()?;
        let options =
            ResourceOptions::new(StorageMode::Private, CpuCacheMode::DefaultCache, hazard);
        let buffer = self
            .buffer
            .device()
            .own_buffer(span.next_multiple_of(4), options)?;
        let (_, start) = buffer.at(0);
        let back = staged_rows_back(layout, start, at);
        Ok(ShadowStaging {
            into: vk::BufferCopy {
                src_offset: at,
                dst_offset: start,
                size: span,
            },
            back,
            buffer,
        })
    }

    /// The copy unit of CPU access to the texels: one texel.
    pub(super) fn unit(&self) -> CopyUnit {
        CopyUnit {
            bytes: self.texel_bytes as u32,
            extent: [1, 1],
            aspect: ImageAspectFlags::COLOR,
        }
    }

    /// What a binding plan reads of a texture buffer of `pixel_format` and `swizzle` over these
    /// texels: the texels' offset in the `VkBuffer` its bindings view (`Texels::viewed`), which
    /// the plan's view ranges address, and the swizzle a function's reads apply.
    pub(super) fn bound(
        &self,
        pixel_format: PixelFormat,
        swizzle: TextureSwizzleChannels,
    ) -> BoundTexture {
        BoundTexture::Buffer {
            pixel_format,
            offset: self.viewed().1,
            length: self.shader.row_bytes,
            bytes_per_row: self.shader.row_pitch,
            resource_id: None,
            swizzle,
        }
    }

    /// A use of the texels' bytes of the buffer, reading by `read` and writing by `write`; of
    /// the staging buffer, for a texture buffer viewed through one (a shadow's use).
    pub(super) fn bound_use(&self, read: Access, write: Access) -> Use<ResourceKey> {
        if let Some(copies) = &self.staged {
            return Use {
                key: copies.shadow_key,
                tracked: copies.tracked,
                range: copies.shadow_range.clone(),
                read,
                write,
            };
        }
        let shader = &self.shader;
        let (identity, range) = self.buffer.tracked(shader.offset..shader.end());
        Use {
            key: identity.key,
            tracked: self.tracked(),
            range: UseRange::Bytes(range),
            read,
            write,
        }
    }

    /// The texel buffer view of `range` of the buffer in `format`, a storage texel buffer's
    /// (`storage`) or a uniform one's, made at its first use, with the usage [`view_usage`]
    /// gives it.
    pub(super) fn view(
        &self,
        format: Format,
        storage: bool,
        range: BufferRange,
    ) -> Result<vk::BufferView, Error> {
        self.view_of(format, storage, (range.offset, range.range))
    }

    /// [`view`](Self::view) of `range` bytes from `offset` in the viewed `VkBuffer`.
    pub(super) fn view_of(
        &self,
        format: Format,
        storage: bool,
        (offset, range): (u64, u64),
    ) -> Result<vk::BufferView, Error> {
        let key = (format, storage, offset, range);
        let mut views = lock(&self.views.views);
        if let Some(&view) = views.get(&key) {
            return Ok(view);
        }
        let device = self.buffer.device();
        let profile = device.compile_device().profile();
        let usage = view_usage(
            |usage| profile.supports_format(format, usage),
            storage,
            device.enabled().features.contains(&VkFeature::Maintenance5),
        )?;
        // The range is the viewed `VkBuffer`'s already ([`Texels::bound`]).
        let (buffer, _) = self.viewed();
        let mut only = vk::BufferUsageFlags2CreateInfoKHR::default();
        let mut info = vk::BufferViewCreateInfo::default()
            .buffer(buffer)
            .format(format.into())
            .offset(offset)
            .range(range);
        if let Some(named) = usage.named() {
            only = only.usage(named);
            info = info.push_next(&mut only);
        }
        // SAFETY: the buffer is live and made for both texel buffer usages; `format` serves the
        // usages the view has (both, or the one its `VkBufferUsageFlags2CreateInfoKHR` names
        // under the `maintenance5` the device enabled, [`view_usage`]); the binding plan aligned
        // the range and bounded its texels (`bind::instantiate::texel_ranges`).
        let view =
            unsafe { device.shared().vk().create_buffer_view(&info, None) }.map_err(vulkan)?;
        views.insert(key, view);
        Ok(view)
    }

    /// The bytes of `region` (in units of one texel) at the texels' place in the buffer, as a
    /// layout of the buffer's bytes from the region's first, which starts that many bytes into
    /// the buffer.
    fn region_layout(&self, region: &ImageRegion, host: &HostLayout) -> (u64, HostLayout) {
        let layout = &self.cpu;
        let [x, y, _] = region.origin.map(u64::from);
        let start = layout.offset
            + layout.slice_pitch * u64::from(region.layer)
            + layout.row_pitch * y
            + self.texel_bytes * x;
        (
            start,
            HostLayout {
                bytes_per_row: layout.row_pitch,
                bytes_per_image: layout.slice_pitch,
                ..*host
            },
        )
    }

    /// The buffer's bytes `start..start + len` (`None` for more than `u64::MAX`), which lie in it:
    /// their first byte and their end. Refused for a buffer the CPU does not reach (`Private`).
    fn bytes(
        &self,
        call: &'static str,
        start: u64,
        len: Option<u64>,
    ) -> Result<(*mut u8, u64), Error> {
        let contents = self
            .buffer
            .contents()
            .ok_or_else(|| invalid(call, Problem::Copy(CopyProblem::CpuAccessPrivate)))?;
        // The texture lies in the buffer, and so do the rows of a region checked in its level.
        let end = len.and_then(|len| start.checked_add(len));
        let Some(end) = end.filter(|&end| end <= self.buffer.length()) else {
            return Err(internal("a texture's rows past its buffer"));
        };
        // SAFETY: `start` lies in the buffer's mapped bytes (checked above).
        Ok((unsafe { contents.as_ptr().add(start as usize) }, end))
    }

    /// `replaceRegion` of `region` from `bytes` laid out as `host` (checked by the texture): into
    /// the buffer's mapped bytes, and from there to its own where it has a host copy (a `Managed`
    /// buffer on a device without unified memory, `Buffer::upload`), as Metal's `replaceRegion`
    /// updates a managed texture for the GPU.
    pub(super) fn write(
        &self,
        call: &'static str,
        region: &ImageRegion,
        host: &HostLayout,
        bytes: &[u8],
    ) -> Result<(), Error> {
        let mut packed = vec![0u8; host.packed_len() as usize];
        host.pack(bytes, &mut packed);
        let (start, inside) = self.region_layout(region, host);
        let (target, end) = self.bytes(call, start, inside.host_len())?;
        let n = host.row_bytes as usize;
        for (at, from) in inside.rows() {
            // SAFETY: every row lies in the buffer's mapped bytes from `start` (`bytes`); a write
            // racing the GPU's use of them is the program's, as through `contents`.
            unsafe { std::ptr::copy_nonoverlapping(packed[from..].as_ptr(), target.add(at), n) };
        }
        // A write of host pages a consumer logs is the runtime's own, which its log does not see:
        // reported, so the copies of the pages take it (`dirty_pages.rs`).
        let at = target as u64;
        self.buffer.wrote_host(at..at + (end - start));
        self.buffer.upload(start..end)
    }

    /// `getBytes` of `region` into `bytes` laid out as `host` (checked by the texture): from the
    /// buffer's mapped bytes, a host copy's where it has one, which `synchronize_resource` updates,
    /// as Metal's managed texture needs `synchronizeTexture` on a device without unified memory.
    pub(super) fn read(
        &self,
        call: &'static str,
        region: &ImageRegion,
        host: &HostLayout,
        bytes: &mut [u8],
    ) -> Result<(), Error> {
        let mut packed = vec![0u8; host.packed_len() as usize];
        let (start, inside) = self.region_layout(region, host);
        let (source, _) = self.bytes(call, start, inside.host_len())?;
        let n = host.row_bytes as usize;
        for (at, to) in inside.rows() {
            // SAFETY: as in `write`.
            unsafe { std::ptr::copy_nonoverlapping(source.add(at), packed[to..].as_mut_ptr(), n) };
        }
        host.unpack(&packed, bytes);
        Ok(())
    }
}

/// The usage a texel buffer view of a format of the device's takes, beside the buffer's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ViewUsage {
    /// The buffer's: the format serves both kinds of texel buffer the buffer is made for.
    Buffer,
    /// Only the view's own kind, a storage texel buffer (`storage`) or a uniform one
    /// (`VkBufferUsageFlags2CreateInfoKHR`): the format serves it and not the other.
    Only { storage: bool },
}

impl ViewUsage {
    /// The usage a view of [`ViewUsage::Only`] names, or `None` for the buffer's.
    fn named(self) -> Option<vk::BufferUsageFlags2KHR> {
        match self {
            ViewUsage::Buffer => None,
            ViewUsage::Only { storage: true } => {
                Some(vk::BufferUsageFlags2KHR::STORAGE_TEXEL_BUFFER)
            }
            ViewUsage::Only { storage: false } => {
                Some(vk::BufferUsageFlags2KHR::UNIFORM_TEXEL_BUFFER)
            }
        }
    }
}

/// The usage of a texel buffer view of a storage texel buffer (`storage`) or a uniform one, in a
/// format `supports` says which kinds of texel buffer the device takes it for, on a device that
/// enabled `maintenance5` or not. The runtime makes every buffer for both kinds (`buffer::usage`),
/// and a view's format must serve every texel buffer usage the view has
/// (`VUID-VkBufferViewCreateInfo-format-08778`, `-08779`): the buffer's where the format serves
/// both, else only the view's own kind, which `maintenance5` lets a view name
/// (`VUID-VkBufferViewCreateInfo-pNext-08780`, `-08781`). Without `maintenance5` that view is
/// not made (`NotImplemented(Textures)`), nor one in a format that does not serve the view's
/// kind, which no binding plan names.
pub(super) fn view_usage(
    supports: impl Fn(FormatUsage) -> bool,
    storage: bool,
    maintenance5: bool,
) -> Result<ViewUsage, Error> {
    let (own, other) = if storage {
        (
            FormatUsage::StorageTexelBuffer,
            FormatUsage::UniformTexelBuffer,
        )
    } else {
        (
            FormatUsage::UniformTexelBuffer,
            FormatUsage::StorageTexelBuffer,
        )
    };
    match (supports(own), supports(other)) {
        (true, true) => Ok(ViewUsage::Buffer),
        (true, false) if maintenance5 => Ok(ViewUsage::Only { storage }),
        _ => Err(Error::NotImplemented(RuntimeGap::Textures)),
    }
}

/// The regions that copy the texels laid out as `layout` from a staging buffer, where they start at
/// `start`, back into the texture's `VkBuffer`, where they start at `at` (`ShadowStaging::back`):
/// one a row, but one for rows that lie one after another or share bytes.
///
/// A row that shares bytes with the one above (an IOSurface plane's on Metal's GPU,
/// `linear_plan::iosurface_gpu_layout`) extends that row's region rather than opening one of its
/// own: a region of each would write the shared bytes twice in one `vkCmdCopyBuffer`, which no
/// two destination regions of one may do (`VUID-vkCmdCopyBuffer-pRegions-12482`; older releases
/// read it off `-00117`'s union of the destination regions). The staging buffer holds the plane's
/// bytes as they lie, the shadow's copy back having written each of them once
/// ([`plane_copy_regions`]), so one region of the rows' whole span moves them as they are.
pub(super) fn staged_rows_back(layout: &BufferLayout, start: u64, at: u64) -> Vec<vk::BufferCopy> {
    let rows = (0..layout.slices).flat_map(|slice| {
        (0..layout.rows).map(move |row| slice * layout.slice_pitch + row * layout.row_pitch)
    });
    let mut back: Vec<vk::BufferCopy> = Vec::new();
    for row in rows {
        let end = start + row + layout.row_bytes;
        match back.last_mut() {
            Some(last) if last.src_offset + last.size >= start + row => {
                last.size = end - last.src_offset;
            }
            _ => back.push(vk::BufferCopy {
                src_offset: start + row,
                dst_offset: at + row,
                size: layout.row_bytes,
            }),
        }
    }
    back
}

/// The regions that copy the texels laid out as `layout`, `texel_bytes` each, starting at `start`
/// of the `VkBuffer` the copies address, into a 2D texture's shadow image of `aspects`, and the
/// regions that copy them back out of it (`shadow.rs`, [`Texels::shadow_copies`]): one region a
/// slice each way, the slice's rows of texels, where each row lies a row apart from the one above
/// or further. Metal's slices lie multiples of 128 bytes apart and its rows multiples of 16, so
/// of every texel size.
///
/// Where consecutive rows share bytes (an IOSurface plane's on Metal's GPU,
/// `linear_plan::iosurface_gpu_layout`: 1 to 15 bytes, a whole number of texels, between
/// consecutive rows alone), no one region states a slice: a region addresses its rows one
/// `bufferRowLength` apart, which is at least its width
/// (`VUID-vkCmdCopyBufferToImage-bufferRowLength-00195`), and the shared bytes hold a texel of
/// each of two rows. Each slice takes two regions instead, split at the texels the sharing
/// covers:
///
/// - going in, every row's shared head and the rest of every row. Their sources share the bytes,
///   which a copy into an image allows: a memory location may not be read by a source region and
///   written by a destination region of one copy
///   (`VUID-vkCmdCopyBufferToImage-pRegions-00173`), nor written by two destination regions
///   (`-12483`), and the two regions' image texels are disjoint and together every texel;
/// - going back, the rest of every row, and the head of row 0 alone. Those tile the slice's bytes
///   exactly once, so no byte is written by two destination regions
///   (`VUID-vkCmdCopyImageToBuffer-pRegions-12485`), and every shared byte takes the earlier
///   row's tail. Metal leaves there whichever row its own command wrote last, an order that
///   follows the command, the GPU pitch and the row count rather than one rule
///   (`linear_plan.rs`'s **Rows that overlap**); a kernel, the only writer such a plane takes,
///   moves its winner with the threadgroup shape, so the earlier row's tail is one of its answers
///   ([`Degradation::CollidingSurfaceRowWrites`](super::Degradation::CollidingSurfaceRowWrites)).
pub(super) fn plane_copy_regions(
    layout: &BufferLayout,
    texel_bytes: u64,
    start: u64,
    aspects: vk::ImageAspectFlags,
) -> (Vec<vk::BufferImageCopy>, Vec<vk::BufferImageCopy>) {
    let shared = layout.shared_bytes();
    let width = (layout.row_bytes / texel_bytes) as u32;
    let pitch = (layout.row_pitch / texel_bytes) as u32;
    // The descriptor's checks bound a texture's extent and slices to `u32`s.
    let (slices, rows) = (layout.slices as u32, layout.rows as u32);
    let region = |slice: u32, at: u64, x: u32, texels: u32, height: u32| vk::BufferImageCopy {
        buffer_offset: at + layout.slice_pitch * u64::from(slice),
        buffer_row_length: pitch,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: aspects,
            mip_level: 0,
            base_array_layer: slice,
            layer_count: 1,
        },
        image_offset: vk::Offset3D {
            x: x as i32,
            y: 0,
            z: 0,
        },
        image_extent: vk::Extent3D {
            width: texels,
            height,
            depth: 1,
        },
    };
    if shared == 0 {
        let regions: Vec<vk::BufferImageCopy> = (0..slices)
            .map(|slice| region(slice, start, 0, width, rows))
            .collect();
        return (regions.clone(), regions);
    }
    let head = (shared / texel_bytes) as u32;
    let (mut into, mut back) = (Vec::new(), Vec::new());
    for slice in 0..slices {
        into.push(region(slice, start, 0, head, rows));
        into.push(region(slice, start + shared, head, width - head, rows));
        back.push(region(slice, start, 0, head, 1));
        back.push(region(slice, start + shared, head, width - head, rows));
    }
    (into, back)
}
