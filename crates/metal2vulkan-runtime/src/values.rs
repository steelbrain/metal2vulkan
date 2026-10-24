//! The Metal values only the runtime takes: resource options and their modes, command-buffer
//! status, and `MTLSize` with the other geometric values, with Metal's raw values. A raw value
//! Metal does not define is a typed error, never a cast.
//!
//! These live here rather than in the generated `metal` module: the compile-and-plan layer never
//! reads them, and `MTLResourceOptions` is defined in Metal's headers through shifts of the other
//! types' values, which the enum generator does not take. The freeze decides whether the
//! generator learns them.

use super::error::{invalid, Error, Problem};

/// `MTLStorageMode`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum StorageMode {
    /// `MTLStorageModeShared`: memory the CPU and GPU share.
    #[default]
    Shared,
    /// `MTLStorageModeManaged`: a CPU copy synchronized explicitly.
    Managed,
    /// `MTLStorageModePrivate`: GPU-only memory.
    Private,
    /// `MTLStorageModeMemoryless`: tile memory, for render attachments only.
    Memoryless,
}

impl StorageMode {
    /// Metal's raw value.
    pub fn raw(self) -> u64 {
        match self {
            StorageMode::Shared => 0,
            StorageMode::Managed => 1,
            StorageMode::Private => 2,
            StorageMode::Memoryless => 3,
        }
    }

    /// The mode of Metal's raw value `raw`.
    pub fn from_raw(raw: u64) -> Result<StorageMode, Error> {
        Ok(match raw {
            0 => StorageMode::Shared,
            1 => StorageMode::Managed,
            2 => StorageMode::Private,
            3 => StorageMode::Memoryless,
            raw => return Err(undefined("MTLStorageMode", raw)),
        })
    }
}

/// `MTLCPUCacheMode`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CpuCacheMode {
    /// `MTLCPUCacheModeDefaultCache`.
    #[default]
    DefaultCache,
    /// `MTLCPUCacheModeWriteCombined`: a hint the runtime accepts and does not act on.
    WriteCombined,
}

impl CpuCacheMode {
    /// Metal's raw value.
    pub fn raw(self) -> u64 {
        match self {
            CpuCacheMode::DefaultCache => 0,
            CpuCacheMode::WriteCombined => 1,
        }
    }

    /// The mode of Metal's raw value `raw`.
    pub fn from_raw(raw: u64) -> Result<CpuCacheMode, Error> {
        Ok(match raw {
            0 => CpuCacheMode::DefaultCache,
            1 => CpuCacheMode::WriteCombined,
            raw => return Err(undefined("MTLCPUCacheMode", raw)),
        })
    }
}

/// `MTLHazardTrackingMode`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum HazardTrackingMode {
    /// `MTLHazardTrackingModeDefault`: tracked for a resource made outside a heap.
    #[default]
    Default,
    /// `MTLHazardTrackingModeUntracked`: the program orders the resource's accesses.
    Untracked,
    /// `MTLHazardTrackingModeTracked`.
    Tracked,
}

impl HazardTrackingMode {
    /// Metal's raw value.
    pub fn raw(self) -> u64 {
        match self {
            HazardTrackingMode::Default => 0,
            HazardTrackingMode::Untracked => 1,
            HazardTrackingMode::Tracked => 2,
        }
    }

    /// The mode of Metal's raw value `raw`.
    pub fn from_raw(raw: u64) -> Result<HazardTrackingMode, Error> {
        Ok(match raw {
            0 => HazardTrackingMode::Default,
            1 => HazardTrackingMode::Untracked,
            2 => HazardTrackingMode::Tracked,
            raw => return Err(undefined("MTLHazardTrackingMode", raw)),
        })
    }
}

/// `MTLResourceOptions`: the CPU cache mode in bits 0-3, the storage mode in bits 4-7, and the
/// hazard tracking mode in bits 8-9, as Metal packs them, and Metal's private bit 24
/// ([`ResourceOptions::PRIVATE_BIT`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ResourceOptions(u64);

impl ResourceOptions {
    const CACHE_SHIFT: u32 = 0;
    const STORAGE_SHIFT: u32 = 4;
    const HAZARD_SHIFT: u32 = 8;
    const DEFINED: u64 = 0xf | (0xf << 4) | (0x3 << 8);

    /// Bit 24 (0x100_0000), a private option (a consumer's read-only detached buffers and heaps
    /// pass it). [`from_raw`](Self::from_raw) takes it. What Metal does with it (the M3 Max under
    /// macOS 27.0.1), and the runtime:
    /// - a no-copy buffer and a heap are made with it, and report their options without it: Metal
    ///   keeps the GPU's writes from their memory, and which writes it keeps out and which it
    ///   faults on splits by the engine that writes (`read_only_view.rs`). What a shader or the
    ///   blit engine writes is dropped with no error -- a blit's fill, a kernel's stores, a vertex-
    ///   or fragment-stage store, a store through an address loaded out of an argument buffer, a
    ///   pass's visibility result -- and reaches neither the CPU nor a later command buffer. What a
    ///   render pass's own machinery writes faults the command buffer: a colour attachment over
    ///   that memory, a resolve into it, a depth or stencil attachment in a heap of the bit, and a
    ///   blit into a texture over or in it. The runtime makes them as without it, and announces
    ///   `metal2vulkan-degrade-private-resource-option` once; so does an indirect command buffer
    ///   made with it, which Metal makes and whose GPU-written records under it are not measured;
    /// - a buffer placed in a heap without it is made as without it, as Metal's is, announcing
    ///   nothing;
    /// - a buffer of a length or of bytes, and a texture descriptor, end Metal's process (the
    ///   buffer's suballocator and the texture descriptor's validation assert): refused `Invalid`
    ///   (`Problem::UndefinedValue`), as is a Vulkan buffer of the caller's
    ///   (`Device::new_buffer_with_vk_buffer`).
    pub const PRIVATE_BIT: u64 = 1 << 24;

    /// The options of these three modes.
    pub fn new(
        storage: StorageMode,
        cache: CpuCacheMode,
        hazard: HazardTrackingMode,
    ) -> ResourceOptions {
        ResourceOptions(
            (cache.raw() << Self::CACHE_SHIFT)
                | (storage.raw() << Self::STORAGE_SHIFT)
                | (hazard.raw() << Self::HAZARD_SHIFT),
        )
    }

    /// `MTLResourceStorageModeShared` with the other modes at their defaults.
    pub fn shared() -> ResourceOptions {
        ResourceOptions::default()
    }

    /// `MTLResourceStorageModePrivate` with the other modes at their defaults.
    pub fn private() -> ResourceOptions {
        ResourceOptions::new(
            StorageMode::Private,
            CpuCacheMode::DefaultCache,
            HazardTrackingMode::Default,
        )
    }

    /// The options of Metal's raw value `raw`: refused when it sets a bit outside the three
    /// fields and [`PRIVATE_BIT`](Self::PRIVATE_BIT), or a field to a value its type does not
    /// define.
    pub fn from_raw(raw: u64) -> Result<ResourceOptions, Error> {
        if raw & !(Self::DEFINED | Self::PRIVATE_BIT) != 0 {
            return Err(undefined("MTLResourceOptions", raw));
        }
        let options = ResourceOptions(raw);
        options.storage_mode()?;
        options.cpu_cache_mode()?;
        options.hazard_tracking_mode()?;
        Ok(options)
    }

    /// Metal's raw value.
    pub fn raw(self) -> u64 {
        self.0
    }

    /// The storage mode field.
    pub fn storage_mode(self) -> Result<StorageMode, Error> {
        StorageMode::from_raw((self.0 >> Self::STORAGE_SHIFT) & 0xf)
    }

    /// The CPU cache mode field.
    pub fn cpu_cache_mode(self) -> Result<CpuCacheMode, Error> {
        CpuCacheMode::from_raw((self.0 >> Self::CACHE_SHIFT) & 0xf)
    }

    /// The hazard tracking mode field.
    pub fn hazard_tracking_mode(self) -> Result<HazardTrackingMode, Error> {
        HazardTrackingMode::from_raw((self.0 >> Self::HAZARD_SHIFT) & 0x3)
    }

    /// These options with the hazard tracking mode field set to `mode`, every other bit kept
    /// ([`PRIVATE_BIT`](Self::PRIVATE_BIT) included).
    ///
    /// A resource reports the mode it resolved rather than the one its descriptor asked for, as
    /// Metal's own resources do: no resource of Metal's answers `Default`.
    pub(crate) fn with_hazard_tracking_mode(self, mode: HazardTrackingMode) -> ResourceOptions {
        ResourceOptions(
            (self.0 & !(0x3 << Self::HAZARD_SHIFT)) | (mode.raw() << Self::HAZARD_SHIFT),
        )
    }

    /// The options without [`PRIVATE_BIT`](Self::PRIVATE_BIT), and whether they set it.
    pub(crate) fn without_private_bit(self) -> (ResourceOptions, bool) {
        (
            ResourceOptions(self.0 & !Self::PRIVATE_BIT),
            self.0 & Self::PRIVATE_BIT != 0,
        )
    }
}

/// `MTLCommandBufferStatus`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CommandBufferStatus {
    /// `MTLCommandBufferStatusNotEnqueued`: not yet committed.
    NotEnqueued,
    /// `MTLCommandBufferStatusEnqueued`: never answered, since the runtime has no `enqueue`.
    Enqueued,
    /// `MTLCommandBufferStatusCommitted`: committed, not yet submitted whole.
    Committed,
    /// `MTLCommandBufferStatusScheduled`: its last batch is submitted.
    Scheduled,
    /// `MTLCommandBufferStatusCompleted`: its work completed.
    Completed,
    /// `MTLCommandBufferStatusError`: it failed; `CommandBuffer::error` says why.
    Error,
}

impl CommandBufferStatus {
    /// Metal's raw value.
    pub fn raw(self) -> u64 {
        match self {
            CommandBufferStatus::NotEnqueued => 0,
            CommandBufferStatus::Enqueued => 1,
            CommandBufferStatus::Committed => 2,
            CommandBufferStatus::Scheduled => 3,
            CommandBufferStatus::Completed => 4,
            CommandBufferStatus::Error => 5,
        }
    }
}

/// `MTLSize`: a grid or threadgroup size.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Size {
    /// `width`.
    pub width: u64,
    /// `height`.
    pub height: u64,
    /// `depth`.
    pub depth: u64,
}

impl Size {
    /// `MTLSizeMake(width, height, depth)`.
    pub fn new(width: u64, height: u64, depth: u64) -> Size {
        Size {
            width,
            height,
            depth,
        }
    }

    /// The three axes as a dispatch states them, or `None` when one is past `u32`.
    pub(super) fn axes(self) -> Option<[u32; 3]> {
        Some([
            u32::try_from(self.width).ok()?,
            u32::try_from(self.height).ok()?,
            u32::try_from(self.depth).ok()?,
        ])
    }

    /// The three axes as plain Metal's compute encoder takes a grid, a threadgroup count or a
    /// stage-in region: each axis modulo `2^32` (an M4 ran 3 threadgroups for `2^33 + 3` and none
    /// for `2^32`).
    pub(super) fn truncated(self) -> [u32; 3] {
        [self.width as u32, self.height as u32, self.depth as u32]
    }

    /// The axes' product as plain Metal's compute encoder checks a threadgroup's size against
    /// its limit: the 64-bit product, wrapping (a threadgroup of `2^62 + 1` by 4 ran 4 threads on
    /// an M4, one of `2^32 + 1` by 1 none).
    pub(super) fn wrapping_volume(self) -> u64 {
        self.width
            .wrapping_mul(self.height)
            .wrapping_mul(self.depth)
    }
}

/// `MTLOrigin`: a texel's coordinates in a texture level.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Origin {
    /// `x`.
    pub x: u64,
    /// `y`.
    pub y: u64,
    /// `z`.
    pub z: u64,
}

impl Origin {
    /// `MTLOriginMake(x, y, z)`.
    pub fn new(x: u64, y: u64, z: u64) -> Origin {
        Origin { x, y, z }
    }
}

/// `MTLRegion`: a box of texels in a texture level.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Region {
    /// `origin`: the box's first texel.
    pub origin: Origin,
    /// `size`: the box's extent, in texels.
    pub size: Size,
}

impl Region {
    /// `MTLRegionMake1D(x, width)`.
    pub fn new_1d(x: u64, width: u64) -> Region {
        Region::new_3d(x, 0, 0, width, 1, 1)
    }

    /// `MTLRegionMake2D(x, y, width, height)`.
    pub fn new_2d(x: u64, y: u64, width: u64, height: u64) -> Region {
        Region::new_3d(x, y, 0, width, height, 1)
    }

    /// `MTLRegionMake3D(x, y, z, width, height, depth)`.
    pub fn new_3d(x: u64, y: u64, z: u64, width: u64, height: u64, depth: u64) -> Region {
        Region {
            origin: Origin::new(x, y, z),
            size: Size::new(width, height, depth),
        }
    }
}

/// `MTLCoordinate2D`: a point in screen or physical pixels (a rasterization-rate map's).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Coordinate2D {
    /// `x`.
    pub x: f32,
    /// `y`.
    pub y: f32,
}

impl Coordinate2D {
    /// `MTLCoordinate2DMake(x, y)`.
    pub fn new(x: f32, y: f32) -> Coordinate2D {
        Coordinate2D { x, y }
    }
}

fn undefined(ty: &'static str, raw: u64) -> Error {
    invalid("from_raw", Problem::UndefinedValue { ty, raw })
}
