//! Rotated textures' copies: a framebuffer-only texture over an IOSurface plane made with a
//! rotation renders through an image of its pass's frame, whose texels the encoders turn into the
//! plane's bytes after a pass that stores it and back before one that uses it.
//!
//! What plain Metal does, measured on an M4 under macOS 27.0.1 and an M3 Max under 26.7 (the
//! consumer's report):
//!
//! - **Where a pass's texels land.** A pass draws a `W x H` frame (the texture's `width` and
//!   `height`); its texel `(x, y)` lands in the plane at `(H - 1 - y, x)` for rotation 1 and 4,
//!   which store an `H x W` image, at `(W - 1 - x, H - 1 - y)` for rotation 2, and at
//!   `(y, W - 1 - x)` for rotation 3, rows `bytesPerRow` apart from the plane's offset
//!   ([`stored_extent`], [`source_map`]). Its positions, viewport and scissor are the frame's, a
//!   resolve into it lands as a store does, and attachments of different rotations in one pass
//!   each land by their own.
//! - **Where a pass loads its texels from** (on an M4 under 27.0.1, on the M3 Max under 26.7, every
//!   frame, rotation and case twice). A pass that loads the texture sees the stored image turned
//!   back, the inverse of the store, at rotations 1 to 3. At rotation 4 it sees the plane
//!   transposed: its texel `(x, y)` is the plane's `(y, x)`, so a texel it loads and stores again
//!   lands mirrored left to right, at `(H - 1 - x', y')` of the plane where it lay at `(x', y')`.
//!   Metal loads and stores only the 32x32 tiles of the frame (from its origin) that a draw's
//!   primitives reach, after the viewport and scissor, fragments discarded or not, or every tile
//!   where another color attachment clears and stores; a pass without a draw (or with a second
//!   attachment cleared but no draw) leaves every texel where it lay, as does a tile no primitive
//!   reaches (a primitive off the frame reaches none). On the M3 an 8x16 block of a reached tile
//!   was left in place in two of 88 passes, a different block each time, Metal's own
//!   nondeterminism, which the runtime does not model; the M4 answered the same each time. The
//!   runtime loads by Metal's map ([`Turn::Load`]) in a pass whose draws reach tiles, every tile,
//!   and by the inverse of the store ([`Turn::Keep`]) in one whose draws reach none, so its store
//!   leaves every texel in place (`clear_reach::PassReach`). That is Metal's answer in a pass whose
//!   draws reach no tile or every tile (another attachment cleared and stored). In any other pass
//!   that draws, the tiles no draw reaches, which Metal keeps, turn: a near-equivalent announced
//!   once per device (`Degradation::RotatedLoadTiles`). Which tiles a draw reaches depends on where
//!   the GPU places the vertices, as a `DontCare` load's (`clear_reach.rs`).
//! - **The copies** ([`RotatedCopies`]): the shadow framework's (`shadow.rs`), the plane's bytes
//!   the truth and the frame's image the shadow. Into the image: the bytes from the plane's first
//!   texel to its last into a staging buffer, the rotation kernel (`bind::rotate_texels_module`)
//!   turning them into a tight buffer of the frame's texels, and that buffer into the image. Back:
//!   the image into the tight buffer, the kernel turning it into the staging buffer, and each of
//!   its rows into the plane, so the bytes between rows stay. Each step is ordered after the one
//!   before by a full barrier, the buffers are the texture's for its life, and the kernel's sets
//!   bind only them, so a plane over host pages a rebind moves keeps them: the copies read its
//!   `VkBuffer` as they are recorded.

use super::buffer::Buffer;
use super::error::{internal, Error, RuntimeGap};
use super::owned_kernel::{OwnedKernel, OwnedModule, OwnedSets};
use super::row_copies::RowCopies;
use super::shadow::{self, ShadowStaging};
use super::tracker::Barrier;
use super::values::{CpuCacheMode, HazardTrackingMode, ResourceOptions, StorageMode};
use super::Device;
use crate::device::functions::Vk;
use ash::vk;
use metal2vulkan::bind::RotateTexelsModule;

/// Where a shape lies in the shape buffer, per direction: a multiple of every
/// `minStorageBufferOffsetAlignment`.
const SHAPE_SLOT: u64 = 256;

/// Whether `rotation` is a quarter turn, whose store is the frame's height wide and its width
/// tall: 1, 3, and 4, which stores as 1 does.
pub(super) fn quarter(rotation: u8) -> bool {
    matches!(rotation, 1 | 3 | 4)
}

/// The extent of the image Metal stores a `[width, height]` frame of `rotation` as.
pub(super) fn stored_extent(rotation: u8, [width, height]: [u32; 2]) -> [u32; 2] {
    if quarter(rotation) {
        [height, width]
    } else {
        [width, height]
    }
}

/// Which way a copy turns a rotated frame's texels ([`source_map`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Turn {
    /// Into the frame from the plane, as a pass that draws loads it: Metal's load.
    Load,
    /// Into the frame from the plane by the inverse of the store: for a pass that draws nothing,
    /// whose store then leaves every texel in place, as Metal's pass that loads and stores no
    /// tile does.
    Keep,
    /// Back from the frame into the plane: Metal's store.
    Store,
}

/// The affine map from a texel of the destination to the texel of the source it takes, as
/// `bind::RotateTexelsModule::shape` takes it, for a `[width, height]` frame of `rotation`, the
/// `turn`'s (the module docs).
pub(super) fn source_map(rotation: u8, [width, height]: [u32; 2], turn: Turn) -> [i32; 6] {
    // Extents are at most 16384, so their last texels are `i32`s.
    let (w, h) = (width as i32 - 1, height as i32 - 1);
    match (rotation, turn) {
        // Loaded from (y, x), the plane's transpose.
        (4, Turn::Load) => [0, 1, 0, 1, 0, 0],
        // Stored at (h - y, x).
        (1 | 4, Turn::Load | Turn::Keep) => [0, -1, h, 1, 0, 0],
        (1 | 4, Turn::Store) => [0, 1, 0, -1, 0, h],
        // Stored at (w - x, h - y), its own inverse.
        (2, _) => [-1, 0, w, 0, -1, h],
        // Stored at (y, w - x).
        (3, Turn::Load | Turn::Keep) => [0, 1, 0, -1, 0, w],
        (3, Turn::Store) => [0, -1, w, 1, 0, 0],
        _ => [1, 0, 0, 0, 1, 0],
    }
}

/// Whether a pass that draws loads a `[width, height]` frame of `rotation` other than by the
/// inverse of its store, so that a texel it loads and stores again moves (rotation 4).
pub(super) fn loads_apart(rotation: u8, extent: [u32; 2]) -> bool {
    source_map(rotation, extent, Turn::Load) != source_map(rotation, extent, Turn::Keep)
}

/// The frame a rotated texture's copies turn: its pass's `extent`, its `rotation`, its texels'
/// bytes, and the rows of its plane, `row_pitch` apart, which span `span` bytes from the first
/// texel to the last.
#[derive(Clone, Copy, Debug)]
pub(super) struct Frame {
    pub(super) rotation: u8,
    pub(super) extent: [u32; 2],
    pub(super) texel_bytes: u64,
    pub(super) row_pitch: u64,
    pub(super) span: u64,
}

/// One direction of the kernel's moves: its shape, where it lies in the shape buffer, and the
/// dispatch's workgroups.
struct Move {
    shape: [u8; 52],
    at: u64,
    groups: [u32; 2],
}

/// A rotated texture's turns between its pass's frame and its plane's bytes (the module docs):
/// the tight buffer, the shapes, and the kernel's two sets (into the frame, then back).
pub(super) struct RotatedCopies {
    /// The frame's texels, tight; held for the texture's life. No command's uses name it: every
    /// copy writes the frame's image or the plane's bytes, so the tracker orders it after every
    /// other copy of the texture, and each records a full barrier first.
    _tight: Buffer,
    tight_raw: vk::Buffer,
    /// The tight buffer's copies to and from the frame's image.
    regions: [vk::BufferImageCopy; 1],
    /// The two shapes; held for the texture's life.
    _shapes: Buffer,
    shapes_raw: vk::Buffer,
    /// Into the frame by Metal's load ([`Turn::Load`]), then back ([`Turn::Store`]).
    moves: [Move; 2],
    /// The shape of the move into the frame by the inverse of the store ([`Turn::Keep`]): the
    /// first move's, with its slot, set and workgroups.
    keep: [u8; 52],
    /// Whether the load and the keep differ ([`loads_apart`]).
    apart: bool,
    kernel: OwnedKernel,
    sets: OwnedSets,
}

impl RotatedCopies {
    /// The turns of `frame` through `staging`, the staging buffer of its plane's bytes (at least
    /// `span` rounded up to 4 bytes long), of the texture's `hazard` tracking. Refused as
    /// `new_buffer` refuses the tight and shape buffers, and (`NotImplemented(RotatedTexture)`)
    /// where a buffer runs past the device's `maxStorageBufferRange` or a dispatch past its
    /// `maxComputeWorkGroupCount`.
    pub(super) fn new(
        device: &Device,
        staging: &ShadowStaging,
        frame: Frame,
        hazard: HazardTrackingMode,
    ) -> Result<RotatedCopies, Error> {
        let Frame {
            rotation,
            extent: [width, height],
            texel_bytes,
            row_pitch,
            span,
        } = frame;
        let [stored_width, stored_height] = stored_extent(rotation, [width, height]);
        let tight_pitch = u64::from(width) * texel_bytes;
        let tight_bytes = (tight_pitch * u64::from(height)).next_multiple_of(4);
        let staged_bytes = span.next_multiple_of(4);
        let limits = &device.properties().description.limits;
        let range = u64::from(limits.max_storage_buffer_range);
        if tight_bytes > range || staged_bytes > range {
            return Err(Error::NotImplemented(RuntimeGap::RotatedTexture));
        }
        let options =
            ResourceOptions::new(StorageMode::Private, CpuCacheMode::DefaultCache, hazard);
        let tight = device.own_buffer(tight_bytes, options)?;
        let shapes = device.own_buffer(2 * SHAPE_SLOT, options)?;
        let (tight_raw, tight_at) = tight.at(0);
        let (shapes_raw, shapes_at) = shapes.at(0);
        let (staging_raw, staging_at) = (staging.raw(), staging.into.dst_offset);
        let narrow = |v: u64| u32::try_from(v).map_err(|_| internal("a rotated frame past u32"));
        let grid = |words: u64| -> Result<[u32; 2], Error> {
            let local = u64::from(RotateTexelsModule::LOCAL_SIZE);
            let groups = words.div_ceil(local).max(1);
            let across = groups.min(u64::from(limits.max_compute_work_group_count[0]));
            let down = groups.div_ceil(across);
            if down > u64::from(limits.max_compute_work_group_count[1]) {
                return Err(Error::NotImplemented(RuntimeGap::RotatedTexture));
            }
            Ok([narrow(across)?, narrow(down)?])
        };
        let texel = narrow(texel_bytes)?;
        let tight_words = tight_bytes / 4;
        let stored_words = (row_pitch * u64::from(stored_height - 1)
            + u64::from(stored_width) * texel_bytes)
            .div_ceil(4);
        let into_groups = grid(tight_words)?;
        let back_groups = grid(stored_words)?;
        let local = RotateTexelsModule::LOCAL_SIZE;
        let into_shape = |turn: Turn| -> Result<[u8; 52], Error> {
            Ok(RotateTexelsModule::shape(
                texel,
                [narrow(tight_pitch)?, width, height],
                narrow(row_pitch)?,
                source_map(rotation, [width, height], turn),
                narrow(tight_words)?,
                into_groups[0] * local,
            ))
        };
        let keep = into_shape(Turn::Keep)?;
        let moves = [
            Move {
                shape: into_shape(Turn::Load)?,
                at: shapes_at,
                groups: into_groups,
            },
            Move {
                shape: RotateTexelsModule::shape(
                    texel,
                    [narrow(row_pitch)?, stored_width, stored_height],
                    narrow(tight_pitch)?,
                    source_map(rotation, [width, height], Turn::Store),
                    narrow(stored_words)?,
                    back_groups[0] * local,
                ),
                at: shapes_at + SHAPE_SLOT,
                groups: back_groups,
            },
        ];
        let shape_bytes = 4 * u64::from(RotateTexelsModule::SHAPE_WORDS);
        let kernel = device.owned_kernel(OwnedModule::RotateTexels)?;
        // Each set binds the source, the destination and the shape, in binding order.
        let sets = OwnedSets::write(
            device,
            &kernel,
            &[
                [
                    (staging_raw, staging_at, staged_bytes),
                    (tight_raw, tight_at, tight_bytes),
                    (shapes_raw, moves[0].at, shape_bytes),
                ],
                [
                    (tight_raw, tight_at, tight_bytes),
                    (staging_raw, staging_at, staged_bytes),
                    (shapes_raw, moves[1].at, shape_bytes),
                ],
            ],
        )?;
        let regions = [vk::BufferImageCopy {
            buffer_offset: tight_at,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
        }];
        Ok(RotatedCopies {
            _tight: tight,
            tight_raw,
            regions,
            _shapes: shapes,
            shapes_raw,
            moves,
            keep,
            apart: loads_apart(rotation, [width, height]),
            kernel,
            sets,
        })
    }

    /// Whether a pass that draws loads the frame other than by the inverse of its store
    /// ([`loads_apart`]).
    pub(super) fn loads_apart(&self) -> bool {
        self.apart
    }

    /// Record the `turn` of the frame's `image` from `bytes` (the plane's `VkBuffer`) through
    /// `staging`, into the frame or back, its rows back by `back` (`shadow::record_back`).
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording, outside a render pass; the image, the buffers and the
    /// sets are live until it completes, and the image rests in `GENERAL`; `back` is the copy's
    /// `ShadowCopy::staged_back`, over `bytes`.
    pub(super) unsafe fn record(
        &self,
        device: &Vk,
        command_buffer: vk::CommandBuffer,
        image: vk::Image,
        (bytes, staging): (vk::Buffer, &ShadowStaging),
        turn: Turn,
        back: Option<&RowCopies>,
    ) {
        let into = turn != Turn::Store;
        let (step, shape, set) = match turn {
            Turn::Load => (&self.moves[0], &self.moves[0].shape, self.sets.sets[0]),
            Turn::Keep => (&self.moves[0], &self.keep, self.sets.sets[0]),
            Turn::Store => (&self.moves[1], &self.moves[1].shape, self.sets.sets[1]),
        };
        let general = vk::ImageLayout::GENERAL;
        // SAFETY: the caller's contract; the copies lie in the buffers and the image (planned
        // when the texture was made), the shape's 52 bytes fit its slot, and the sets bind the
        // texture's own buffers.
        unsafe {
            Barrier::everything().record(device, command_buffer);
            if into {
                device.cmd_copy_buffer(command_buffer, bytes, staging.raw(), &[staging.into]);
            } else {
                device.cmd_copy_image_to_buffer(
                    command_buffer,
                    image,
                    general,
                    self.tight_raw,
                    &self.regions,
                );
            }
            device.cmd_update_buffer(command_buffer, self.shapes_raw, step.at, shape);
            Barrier::everything().record(device, command_buffer);
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.kernel.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.kernel.layout,
                0,
                &[set],
                &[],
            );
            device.cmd_dispatch(command_buffer, step.groups[0], step.groups[1], 1);
            Barrier::everything().record(device, command_buffer);
            if into {
                device.cmd_copy_buffer_to_image(
                    command_buffer,
                    self.tight_raw,
                    image,
                    general,
                    &self.regions,
                );
            } else {
                shadow::record_back(device, command_buffer, (staging, bytes), back);
            }
        }
    }
}
