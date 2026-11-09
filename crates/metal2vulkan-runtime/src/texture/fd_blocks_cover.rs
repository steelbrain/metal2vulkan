//! Whether a block of the runtime's buffers allocated exportable as an opaque file descriptor
//! holds every image a texture over a buffer takes in place over a plain block (`memory.rs`'s
//! "Buffers in exportable blocks"): the device's check makes its class of such blocks
//! (`in_place_check.rs`, `PagesClass::FdBlocks`) only where it does, so the device's buffers lie
//! there only where no texture over one loses its place (`in_place.rs`'s `fd_blocks_take`).
//!
//! Over a plain block's buffer a texture takes the device's linear image, else a
//! `DRM_FORMAT_MOD_LINEAR` one at Metal's pitch (`texture.rs`'s `alias_unbound`). Over such a
//! block it takes the linear image made for the handle type alone. So the device's answers must
//! give, for every pixel format a buffer backs and each usage a texture over a buffer reads,
//! writes and renders through, a linear image made for the handle type wherever it makes its own,
//! laid out as its own (row pitch, alignment, size) and taking each memory type its own takes of
//! those the blocks are; and no `DRM_FORMAT_MOD_LINEAR` image of its own memory at a pitch other
//! than its linear image's. Each image is made and released, never bound.

use crate::buffer_texture::texel_bytes;
use crate::device::lock;
use crate::dma_buf::ModifierMemory;
use crate::error::Error;
use crate::image_plan::plan_image;
use crate::linear_plan::{linear_lack, BufferLayout};
use crate::memory::FD_BLOCKS_HANDLE;
use crate::texture::linear_alignment::backed_formats;
use crate::texture_descriptor::{texture_usage, TextureLayout, TextureSwizzleChannels};
use crate::{release, Device};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};

/// The width and height of each texture the cover is asked of.
const EXTENT: [u32; 2] = [64, 4];

/// The usages each format is asked for: read, read and written, read and rendered to.
const USAGES: [TextureUsage; 3] = [
    TextureUsage::SHADER_READ,
    TextureUsage::SHADER_READ.union(TextureUsage::SHADER_WRITE),
    TextureUsage::SHADER_READ.union(TextureUsage::RENDER_TARGET),
];

/// The formats the `DRM_FORMAT_MOD_LINEAR` probe tries: one of each common texel width.
const PITCHED: [PixelFormat; 3] = [
    PixelFormat::R8Unorm,
    PixelFormat::RGBA8Unorm,
    PixelFormat::RGBA16Float,
];

impl Device {
    /// Whether a block of the runtime's buffers allocated exportable as an opaque file
    /// descriptor holds every image a texture over a buffer takes in place over a plain block
    /// (the module docs); else the first image it would lose, and why.
    pub(crate) fn fd_blocks_cover(&self) -> Result<(), String> {
        self.fd_blocks_cover_asked().unwrap_or_else(|error| {
            Err(format!(
                "the device's images were refused while its cover was asked: {error}"
            ))
        })
    }

    /// [`fd_blocks_cover`](Self::fd_blocks_cover)'s answer, its images made and released.
    fn fd_blocks_cover_asked(&self) -> Result<Result<(), String>, Error> {
        let profile = self.compile_device().profile();
        let facts = self.shared().textures.facts(profile);
        let types = lock(&self.shared().memory).fd_buffer_types.unwrap_or(0);
        let raw_device = self.shared().vk();
        let exported = ModifierMemory::Exported(FD_BLOCKS_HANDLE);
        let plan_of = |pf, usage| {
            let layout = TextureLayout {
                texture_type: TextureType::Type2D,
                pixel_format: pf,
                extent: [EXTENT[0], EXTENT[1], 1],
                levels: 1,
                layers: 1,
                samples: 1,
            };
            let usage = texture_usage(usage, TextureSwizzleChannels::DEFAULT);
            plan_image(&layout, usage, false, &facts).ok()
        };
        for pf in backed_formats() {
            for usage in USAGES {
                let Some(plan) = plan_of(pf, usage) else {
                    continue;
                };
                if linear_lack(&plan, &facts).is_some() {
                    continue;
                }
                let Ok(own) = self.linear_image(&plan)? else {
                    continue;
                };
                // SAFETY: the image is live and unbound, and no command uses it.
                unsafe { release::object(raw_device, own.raw) };
                let theirs = match self.external_linear_image(&plan, exported)? {
                    Ok(theirs) => theirs,
                    Err(refusal) => {
                        return Ok(Err(format!(
                            "a texture of {pf:?} and {usage:?}: the device makes its own linear \
                             image and none for an opaque file descriptor ({})",
                            refusal.describe(&plan)
                        )))
                    }
                };
                // SAFETY: the image is live and unbound, and no command uses it.
                unsafe { release::object(raw_device, theirs.raw) };
                let (mine, other) = (own.requirements, theirs.requirements);
                let typed = mine.memory_type_bits & types;
                let lies = theirs.report.row_pitch == own.report.row_pitch
                    && theirs.report.offset == own.report.offset
                    && theirs.report.size == own.report.size
                    && mine.alignment.is_multiple_of(other.alignment.max(1))
                    && other.size <= mine.size
                    && other.memory_type_bits & typed == typed;
                if !lies {
                    return Ok(Err(format!(
                        "a texture of {pf:?} and {usage:?}: the device's linear image for an \
                         opaque file descriptor lies otherwise than its own (rows {} and {} \
                         bytes apart, aligned to {} and {}, of types {:#x} and {:#x})",
                        theirs.report.row_pitch,
                        own.report.row_pitch,
                        other.alignment,
                        mine.alignment,
                        other.memory_type_bits,
                        mine.memory_type_bits
                    )));
                }
            }
        }
        if !self.makes_modifier_images() {
            return Ok(Ok(()));
        }
        for pf in PITCHED {
            let Some(plan) = plan_of(pf, TextureUsage::SHADER_READ) else {
                continue;
            };
            let Ok(own) = self.linear_image(&plan)? else {
                continue;
            };
            // SAFETY: the image is live and unbound, and no command uses it.
            unsafe { release::object(raw_device, own.raw) };
            let pitch = own.report.row_pitch;
            let row_bytes = u64::from(EXTENT[0]) * texel_bytes(pf).unwrap_or(1);
            for row_pitch in [pitch + 256, pitch * 2] {
                let rows = u64::from(EXTENT[1]);
                let layout = BufferLayout {
                    offset: 0,
                    row_pitch,
                    slice_pitch: row_pitch * rows,
                    row_bytes,
                    rows,
                    slices: 1,
                };
                let own_memory = ModifierMemory::Own;
                if let Ok(made) = self.pitched_image(&plan, &layout, &facts, own_memory)? {
                    // SAFETY: the image is live and unbound, and no command uses it.
                    unsafe { release::object(raw_device, made.raw) };
                    return Ok(Err(format!(
                        "a texture of {pf:?} with rows {row_pitch} bytes apart: the device makes \
                         a DRM_FORMAT_MOD_LINEAR image of its own memory at that pitch, where its \
                         linear image's rows lie {pitch} apart, and such a block takes none"
                    )));
                }
            }
        }
        Ok(Ok(()))
    }
}
