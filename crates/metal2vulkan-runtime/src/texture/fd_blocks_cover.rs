//! Whether a block of the runtime's buffers allocated exportable as an opaque file descriptor
//! holds every image a texture over a buffer that keeps its padding takes in place over a plain
//! block (`memory.rs`'s "Buffers in exportable blocks"): the device's check makes its class of
//! such blocks (`in_place_check.rs`, `PagesClass::FdBlocks`) only where it does, so the device's
//! buffers lie there only where no texture over one loses its place (`in_place.rs`'s
//! `fd_blocks_take`). A buffer that declares its padding unobserved lies in a plain block
//! (`buffer.rs`'s `NewBufferOf::Unobserved`), and is not asked of.
//!
//! Over a plain block's buffer a texture takes the device's linear image, else a
//! `DRM_FORMAT_MOD_LINEAR` one at Metal's pitch (`texture.rs`'s `alias_unbound`). Over such a
//! block it takes the linear image made for the handle type alone. Over a buffer that keeps its
//! padding, an image holds a texture's texels only where its rows lie one after another
//! (`linear_plan::holding`), at a pitch that is a multiple of 16 (`LINEAR_ALIGNMENT`). So the
//! device's answers must give, for every pixel format a buffer backs, each usage a texture over a
//! buffer reads, writes and renders through, and each width of `linear_alignment::WIDTHS`: a
//! linear image made for the handle type wherever it makes its own, laid out as its own (row
//! pitch, alignment, size) and taking each memory type its own takes of those the blocks are;
//! and, where its rows lie one after another at a multiple of 16 bytes that its own linear image
//! does not lie at, no `DRM_FORMAT_MOD_LINEAR` image of its own memory at that pitch. Each image
//! is made and released, never bound.

use crate::buffer_texture::texel_bytes;
use crate::device::lock;
use crate::dma_buf::ModifierMemory;
use crate::error::Error;
use crate::image_plan::{plan_image, ImagePlan};
use crate::linear_plan::{linear_lack, BufferLayout, LINEAR_ALIGNMENT};
use crate::memory::FD_BLOCKS_HANDLE;
use crate::texture::linear_alignment::{backed_formats, WIDTHS};
use crate::texture::LinearMade;
use crate::texture_descriptor::{texture_usage, TextureLayout, TextureSwizzleChannels};
use crate::{release, Device};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};

/// The rows of each texture the cover is asked of: a multiple of 4, as a driver may lay an
/// image's rows out in blocks of 4.
const ROWS: u32 = 4;

/// The usages each format is asked for: read, read and written, read and rendered to.
const USAGES: [TextureUsage; 3] = [
    TextureUsage::SHADER_READ,
    TextureUsage::SHADER_READ.union(TextureUsage::SHADER_WRITE),
    TextureUsage::SHADER_READ.union(TextureUsage::RENDER_TARGET),
];

impl Device {
    /// Whether a block of the runtime's buffers allocated exportable as an opaque file
    /// descriptor holds every image a texture over a buffer that keeps its padding takes in place
    /// over a plain block (the module docs); else the first image it would lose, and why.
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
        let release_unbound = |made: LinearMade| {
            // SAFETY: the image is live and unbound, and no command uses it.
            unsafe { release::object(raw_device, made.raw) };
        };
        let modifiers = self.makes_modifier_images();
        for pf in backed_formats() {
            // A format of no whole texel per pixel (a packed 4:2:2 one) backs a texture of no
            // width the widths' rows lie at.
            let Some(texel) = texel_bytes(pf) else {
                continue;
            };
            for usage in USAGES {
                for width in WIDTHS {
                    let Some(plan) = plan_of(pf, usage, width as u32, &facts) else {
                        continue;
                    };
                    let what = || format!("a texture of {pf:?}, {width}x{ROWS}, {usage:?}");
                    let own = match linear_lack(&plan, &facts) {
                        Some(_) => None,
                        None => self.linear_image(&plan)?.ok(),
                    };
                    let own_pitch = own.as_ref().map(|own| own.report.row_pitch);
                    if let Some(own) = own {
                        let theirs = (self.external_linear_image(&plan, exported))?;
                        let theirs = match theirs {
                            Ok(theirs) => theirs,
                            Err(refusal) => {
                                release_unbound(own);
                                return Ok(Err(format!(
                                    "{}: the device makes its own linear image and none for an \
                                     opaque file descriptor ({})",
                                    what(),
                                    refusal.describe(&plan)
                                )));
                            }
                        };
                        let (mine, other) = (&own.requirements, &theirs.requirements);
                        let typed = mine.memory_type_bits & types;
                        let lies = theirs.report.row_pitch == own.report.row_pitch
                            && theirs.report.offset == own.report.offset
                            && theirs.report.size == own.report.size
                            && mine.alignment.is_multiple_of(other.alignment.max(1))
                            && other.size <= mine.size
                            && other.memory_type_bits & typed == typed;
                        let why = (!lies).then(|| {
                            format!(
                                "{}: the device's linear image for an opaque file descriptor \
                                 lies otherwise than its own (rows {} and {} bytes apart, \
                                 aligned to {} and {}, of types {:#x} and {:#x})",
                                what(),
                                theirs.report.row_pitch,
                                own.report.row_pitch,
                                other.alignment,
                                mine.alignment,
                                other.memory_type_bits,
                                mine.memory_type_bits
                            )
                        });
                        release_unbound(own);
                        release_unbound(theirs);
                        if let Some(why) = why {
                            return Ok(Err(why));
                        }
                    }
                    // Rows one after another at a multiple of 16 bytes, where the device's own
                    // linear image does not lie: over a plain block, a DRM_FORMAT_MOD_LINEAR
                    // image at that pitch where the device makes one.
                    let packed = width * texel;
                    if !modifiers
                        || !packed.is_multiple_of(LINEAR_ALIGNMENT)
                        || own_pitch == Some(packed)
                    {
                        continue;
                    }
                    let rows = u64::from(ROWS);
                    let layout = BufferLayout {
                        offset: 0,
                        row_pitch: packed,
                        slice_pitch: packed * rows,
                        row_bytes: packed,
                        rows,
                        slices: 1,
                    };
                    let own_memory = ModifierMemory::Own;
                    if let Ok(made) = self.pitched_image(&plan, &layout, &facts, own_memory)? {
                        release_unbound(made);
                        return Ok(Err(format!(
                            "{}, its rows {packed} bytes apart, one after another: the device \
                             makes a DRM_FORMAT_MOD_LINEAR image of its own memory at that pitch, \
                             where its linear image's rows lie {}, and such a block takes none",
                            what(),
                            own_pitch.map_or("nowhere".to_owned(), |p| format!("{p} apart"))
                        )));
                    }
                }
            }
        }
        Ok(Ok(()))
    }
}

/// The plan of a 2D texture of `pf`, `width` texels wide and [`ROWS`] tall, of Metal `usage`;
/// `None` where the device makes no such texture.
fn plan_of(
    pf: PixelFormat,
    usage: TextureUsage,
    width: u32,
    facts: &crate::image_plan::ImageFacts<'_>,
) -> Option<ImagePlan> {
    let layout = TextureLayout {
        texture_type: TextureType::Type2D,
        pixel_format: pf,
        extent: [width, ROWS, 1],
        levels: 1,
        layers: 1,
        samples: 1,
    };
    let usage = texture_usage(usage, TextureSwizzleChannels::DEFAULT);
    plan_image(&layout, usage, false, facts).ok()
}
