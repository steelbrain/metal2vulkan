//! Where a texture over a buffer's bytes lies, answered before it is made
//! ([`Buffer::texture_placement`](crate::Buffer::texture_placement)): by the route its making
//! takes (`texture/route.rs`), the same function, so the answer and the making never disagree.

use crate::in_place::{self, Refusal, ShadowCause};
use crate::texture::{HostWay, Route};

/// Where a texture over a buffer's bytes lies if made now with the same arguments, as
/// [`Buffer::texture_placement`](crate::Buffer::texture_placement) and
/// [`Buffer::texture_placement_with_iosurface_plane`](crate::Buffer::texture_placement_with_iosurface_plane)
/// answer it. A runtime extension: Metal has no such query, as a Mac's GPU works on a texture over
/// the caller's pages in place, where the runtime takes an image in place only where the device's
/// own check passed (docs/API.md, textures over host pages).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TexturePlacement {
    /// A linear image aliasing the buffer's host pages, in place: the device's check passed for
    /// the pages' kind, for every access the texture's usage takes and for texels that lie in the
    /// device's lines of memory as its own do, and the device made the image of the texture's
    /// plan at its pitch, where its requirements take the bytes' place and memory type (the query
    /// made it unbound and released it). Over pages imported as dma-bufs the image's making may
    /// still refuse at the bind, where the caller's source gives no one dma-buf of the pages
    /// ([`InPlaceRefusal::Source`]) or the driver refuses its import ([`InPlaceRefusal::DmaBuf`]):
    /// the texture is then a shadow, its notice (`shadowed-texture`) naming why. Over a copied host
    /// buffer ([`HostBacking::Copied`](crate::HostBacking::Copied)), the image lies over the copy
    /// the GPU uses in the pages' stead, in the copy's memory, where the check passed for the
    /// pages' kind and that memory's (`WorkCounts::copied_textures_in_place`).
    InPlace,
    /// A shadow now, as the device's check has not answered for the pages' kind and the
    /// texture's accesses: once it does, at the commit of the device's only uncommitted command
    /// buffer, the texture moves in place with no copy where the verdicts pass and the device
    /// makes its image in place (`WorkCounts::host_textures_moved_in_place`), and stays a shadow
    /// where they do not or the device refuses the image, as one made then would be. Over the
    /// device's own memory (a `DRM_FORMAT_MOD_LINEAR` image at Metal's pitch,
    /// [`NotHostPages`](Self::NotHostPages)) a texture made now moves to that image the same way
    /// (`WorkCounts::pooled_textures_moved_in_place`), where the verdicts of its pooled memory pass
    /// and the device makes the image.
    /// [`Device::settle_in_place_checks`](crate::Device::settle_in_place_checks) starts the check
    /// and says when its verdicts are known; the same query then answers what a texture made after
    /// takes.
    Pending,
    /// A shadow: an image of the runtime's own memory, which the encoders copy the texels into
    /// from the buffer before a command buffer's uses and back after its writes, announced
    /// (`shadowed-texture`) but for a device copy. Why.
    Shadow(TextureShadowReason),
    /// The buffer's bytes are no host pages (a buffer of the device's own memory, or the caller's
    /// `VkBuffer`): the texture is a linear image aliasing the buffer's memory, the device's own
    /// where it lays one out as Metal does, else, over the device's own memory, a
    /// `DRM_FORMAT_MOD_LINEAR` image at Metal's pitch where the device makes one and its check of
    /// such images over its pooled memory passed for the texture's accesses (the query made it
    /// unbound and released it); a texture over the device's own memory that takes neither answers
    /// its shadow's cause ([`TexturePlacement::Pending`] while that check has not answered). Its
    /// rows (and slices) lie one after another, or the buffer declares the bytes between them
    /// unobserved ([`BufferDescriptor::padding_unobserved`](crate::BufferDescriptor::padding_unobserved)),
    /// which the image then holds; else the answer is
    /// [`TextureShadowReason::BytesBetweenRows`].
    NotHostPages,
    /// A texture of a shape no image in place takes, over any buffer: a texture buffer (its
    /// functions view the buffer's bytes, or a staging copy of them), a rotated texture, a
    /// two-plane YCbCr or extended-range texture (each an image of its own whose copies turn or
    /// convert the texels), or one whose texels Metal lays out apart from an IOSurface plane
    /// (memory of its own). [`Buffer::new_texture_with_iosurface_plane`](crate::Buffer::new_texture_with_iosurface_plane)
    /// says how each is made.
    OtherShape,
}

/// Why a texture over a buffer's bytes is a shadow ([`TexturePlacement::Shadow`]). A runtime
/// extension: Metal has none.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureShadowReason {
    /// The device's policy asks a shadow of every texture over host pages
    /// (`HostPageTexturePolicy::Shadow`, one of `DevicePolicy::exact_answers`).
    Policy,
    /// A shader writes the texture (`ShaderWrite` or `ShaderAtomic`), and its texels are
    /// `texel_bytes` wide, a width of which the check's kernel stores none (it stores 1, 2, 4, 8
    /// and 16 bytes, every width a buffer backs): no check covers a driver's writes of such
    /// texels.
    StoreWidth {
        /// The bytes of one texel.
        texel_bytes: u64,
    },
    /// The device's check found `access` over the pages' kind changing a byte the CPU wrote beside
    /// the texels, for texels that lie in the device's lines as `row_ends` says.
    CheckFailed {
        /// The access the check failed for: the first of the texture's whose verdict did not pass.
        access: InPlaceAccess,
        /// How the texture's texels lie in the device's lines of memory.
        row_ends: InPlaceRowEnds,
    },
    /// The device's check made no image over the pages' kind for `access`, for texels that lie in
    /// the device's lines as `row_ends` says: the device makes no such import, image or operation,
    /// or the check could not run to its end.
    Unchecked {
        /// The access the check made no image for: the first of the texture's whose verdict did
        /// not pass.
        access: InPlaceAccess,
        /// How the texture's texels lie in the device's lines of memory.
        row_ends: InPlaceRowEnds,
    },
    /// The device makes no image in place over the texels: why.
    Refused(InPlaceRefusal),
    /// A tiled texture of `levels` levels, more than one
    /// ([`Buffer::new_tiled_texture`](crate::Buffer::new_tiled_texture)), over any buffer: one
    /// image of its own holds every level, its first a copy of the buffer's bytes kept in step
    /// with them, as no image over the buffer's memory holds levels past the first.
    Mipmapped {
        /// The texture's level count.
        levels: u32,
    },
    /// The texels run past the buffer's length, which no image over the buffer's memory reaches:
    /// the texture's tail holds those past it, zero when made.
    PastLength,
    /// Over a buffer of the device's own memory ([`TexturePlacement::NotHostPages`]'s): its rows
    /// (or slices) lie with bytes of the buffer between them that are not its texels, or share
    /// bytes, which an image over the memory would hold, and Vulkan lets an image's layout moves
    /// and a render pass's load and store write the bytes it holds; the buffer does not declare
    /// the bytes between rows unobserved
    /// ([`BufferDescriptor::padding_unobserved`](crate::BufferDescriptor::padding_unobserved),
    /// which no rows that share bytes take).
    BytesBetweenRows,
    /// On a discrete GPU, a `Managed` texture over a no-copy buffer's bytes (not an IOSurface
    /// plane's) that only samples or reads, over memory that carries a log of its writes, tracked,
    /// one slice laid out as one copy: a copy in the device's memory whose copies in take only the
    /// rows the log reports changed (`WorkCounts::device_copy_textures`), the GPU's copy of the two
    /// a `Managed` resource is on a Mac with a discrete GPU. Not announced.
    DeviceCopy,
    /// Over a `Private` buffer of the device's own memory whose rows the device's own linear
    /// image lays out otherwise than Metal, a tracked texture: its rows take `bytes` bytes to their
    /// pitch, past the bound up to which it takes a `DRM_FORMAT_MOD_LINEAR` image at Metal's pitch
    /// ([`TexturePlacement::NotHostPages`]), as such an image renders and samples slower than the
    /// shadow's own image with its copies past it, and over memory only the runtime's commands
    /// write both give the same answer. The same bound on every device. Over a `Shared` or
    /// `Managed` buffer, and for an untracked texture, the image is taken at any size.
    LinearCost {
        /// The bytes of the texture's rows, each to its pitch.
        bytes: u64,
    },
}

/// What a texture over host pages does with its image in place, each checked apart by the
/// device's check: the operations that may touch bytes bound to the image beside its texels
/// differ. A runtime extension: Metal has none.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InPlaceAccess {
    /// Its usage samples or reads it alone: its image takes its first layout move.
    Reads,
    /// Its usage renders into it (`RenderTarget`): its image also takes render passes' load and
    /// store operations and the feedback loop moves.
    Renders,
    /// Its usage writes it through a shader (`ShaderWrite` or `ShaderAtomic`): its image also
    /// takes kernels' storage writes of its texels, checked by their width: texels of 1 or 2
    /// bytes apart from those of 4, 8 or 16. Of a texture over a buffer's rows whose rows, each
    /// to its pitch, start and end on a line, its buffer holding the last row to its pitch, the
    /// check that lets those writes change the bytes between a row's texels and its pitch, which
    /// Metal's GPUs differ on; else the one that keeps every byte beside the texels.
    Writes,
}

/// How a texture's texels lie in the device's lines of memory (`nonCoherentAtomSize` bytes at a
/// multiple of it), each checked apart by the device's check: a device may write back a line that
/// holds a texel whole, putting back an older value over a CPU write beside the texels in that
/// line. A runtime extension: Metal has none.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InPlaceRowEnds {
    /// A line holds a texel and a byte beside the texels: between two rows or two slices, before
    /// the first texel or after the last.
    InLines,
    /// Every line holds texels alone or bytes beside them alone: the texels start and end on a
    /// line, and so does each row (or slice) where bytes lie between rows (or slices).
    OnLines,
}

/// Why the device makes no image in place over a texture's texels
/// ([`TextureShadowReason::Refused`]), as `WorkCounts::host_texture_shadows_refused_*` counts
/// them. [`Buffer::texture_placement`](crate::Buffer::texture_placement) answers each one the
/// image's making decides before it binds memory, by the same steps: all but `Source`, and
/// `DmaBuf` where the driver refuses the dma-buf's import, which only the bind finds over pages
/// imported as dma-bufs ([`TexturePlacement::InPlace`]). A runtime extension: Metal has none.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InPlaceRefusal {
    /// Its rows share bytes (an IOSurface plane's GPU rows): no image holds a texel twice.
    Rows,
    /// The buffer's bytes were detached from their pages (`Buffer::detach_backing`): no image is
    /// bound to them.
    Detached,
    /// The device makes no such image of the texture's plan: its format, features, usage or
    /// limits, or an import only a dedicated allocation takes.
    Image,
    /// No image the device makes lays the rows (or slices) at the buffer's pitch.
    Pitch,
    /// The image's alignment, size or memory type misses the bytes' place: among them a texture
    /// whose buffer ends inside its last row's pitch, whose image's last row takes the whole pitch,
    /// past the buffer's bytes (`docs/FAST-PATHS.md`, known gaps).
    Placement,
    /// Over pages imported as dma-bufs: the caller's source made no one dma-buf of the texture's
    /// own pages.
    Source,
    /// Over pages imported as dma-bufs: no `DRM_FORMAT_MOD_LINEAR` image over one dma-buf of the
    /// texture's own pages was made for another reason (the modifier's features or limits, the
    /// image's memory past the pages or of another type, or the driver's import).
    DmaBuf,
}

impl InPlaceAccess {
    /// The access `access` is.
    fn of(access: in_place::Access) -> InPlaceAccess {
        match access {
            in_place::Access::Reads => InPlaceAccess::Reads,
            in_place::Access::Renders | in_place::Access::UnobservedRenders => {
                InPlaceAccess::Renders
            }
            in_place::Access::Writes
            | in_place::Access::BufferWrites
            | in_place::Access::NarrowWrites
            | in_place::Access::NarrowBufferWrites => InPlaceAccess::Writes,
        }
    }
}

impl InPlaceRowEnds {
    /// The row ends `rows` are.
    fn of(rows: in_place::RowEnds) -> InPlaceRowEnds {
        match rows {
            in_place::RowEnds::InLines => InPlaceRowEnds::InLines,
            in_place::RowEnds::OnLines => InPlaceRowEnds::OnLines,
        }
    }
}

impl InPlaceRefusal {
    /// The refusal `refusal` is.
    fn of(refusal: Refusal) -> InPlaceRefusal {
        match refusal {
            Refusal::Rows => InPlaceRefusal::Rows,
            Refusal::Detached => InPlaceRefusal::Detached,
            Refusal::Image => InPlaceRefusal::Image,
            Refusal::Pitch => InPlaceRefusal::Pitch,
            Refusal::Placement => InPlaceRefusal::Placement,
            Refusal::Source => InPlaceRefusal::Source,
            Refusal::DmaBuf => InPlaceRefusal::DmaBuf,
        }
    }
}

impl Route {
    /// Where a texture that takes this route lies, as the query answers it.
    pub(crate) fn placement(&self) -> TexturePlacement {
        let cause = match self {
            Route::Levels(levels) => {
                let levels = *levels;
                return TexturePlacement::Shadow(TextureShadowReason::Mipmapped { levels });
            }
            Route::Past => return TexturePlacement::Shadow(TextureShadowReason::PastLength),
            Route::Device { holds: Ok(()) } => return TexturePlacement::NotHostPages,
            Route::Device { holds: Err(_) } => {
                return TexturePlacement::Shadow(TextureShadowReason::BytesBetweenRows)
            }
            Route::Host { way, .. } => match way {
                HostWay::InPlace { .. } => return TexturePlacement::InPlace,
                HostWay::DeviceCopy => {
                    return TexturePlacement::Shadow(TextureShadowReason::DeviceCopy)
                }
                HostWay::Shadow(cause) => cause,
            },
        };
        cause.placement()
    }
}

impl ShadowCause {
    /// Where a texture shadowed for this cause lies, as the query answers it.
    pub(crate) fn placement(&self) -> TexturePlacement {
        let reason = match *self {
            ShadowCause::CheckPending(..) => return TexturePlacement::Pending,
            ShadowCause::Exact => TextureShadowReason::Policy,
            ShadowCause::StoreWidth(texel_bytes) => TextureShadowReason::StoreWidth { texel_bytes },
            ShadowCause::CheckFailed(_, access, rows) => TextureShadowReason::CheckFailed {
                access: InPlaceAccess::of(access),
                row_ends: InPlaceRowEnds::of(rows),
            },
            ShadowCause::Unchecked(_, access, rows) => TextureShadowReason::Unchecked {
                access: InPlaceAccess::of(access),
                row_ends: InPlaceRowEnds::of(rows),
            },
            ShadowCause::Refused(refusal, _) => {
                TextureShadowReason::Refused(InPlaceRefusal::of(refusal))
            }
            ShadowCause::LinearCost(bytes) => TextureShadowReason::LinearCost { bytes },
        };
        TexturePlacement::Shadow(reason)
    }
}
