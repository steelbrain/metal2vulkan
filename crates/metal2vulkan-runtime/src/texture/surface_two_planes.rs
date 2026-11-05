//! A two-plane extended-range texture (`RGB10A8_2P_XR10`, `RGB10A8_2P_XR10_sRGB`) over plane 0
//! of a two-plane IOSurface, whose texels lie in the surface's two planes as Metal reads and
//! writes them there (the consumer's macOS 26.6.2 guests, whose RenderBox makes one over a `'b3a8'`
//! surface at every boot).
//!
//! **Metal's answers** (our own probe `xrs1`: an Apple M4 under macOS 27.0.1,
//! plain; CoreVideo `'b3a8'` surfaces of 64 x 64, 37 x 5 and 33 x 17, a 16 x 8 texture over a
//! 64 x 64 one, and `IOSurfaceCreate` two-plane surfaces with plane 0 at 256, 272 and 260 bytes a
//! row and plane 1 at 64, 80 and 68, plane 1 right after plane 0 or 48 or 4 bytes past it; four
//! fills naming each byte's offset read back through a kernel's `read()`, a random fill, a
//! kernel's writes over bytes of `0xEE`, a draw at load and a clear, `getBytes` and
//! `replaceRegion`, blits to and from a buffer, and linear and nearest samples against a plain
//! copy):
//!
//! - Metal's GPU reads each texel's color word from plane 0 and its alpha byte from plane 1, each
//!   plane from its offset rounded down to 16 bytes, rows its `bytesPerRow` rounded down to 16
//!   apart (as for every surface plane, `linear_plan.rs`; the 260/68 surface with plane 1 4 bytes
//!   past plane 0 read at 0/256 and 1296/64), every texel exact (550's color codes and both
//!   formats' alpha). The sRGB twin lays its planes out alike.
//! - A kernel's writes, a draw and a clear land on those bytes alone, the bytes between the rows
//!   and the planes kept, bits 30 and 31 of each written color word 0. A pass that loads the
//!   texture writes every texel's color word back with bits 30 and 31 0, the texels it does not
//!   draw too.
//! - `getBytes` and `replaceRegion` move plane 0's color words alone, at the plane's own offset
//!   and `bytesPerRow` (the CPU's layout, `linear_plan.rs`), every bit of each; plane 1 is
//!   neither read nor written.
//! - Blits between a buffer and the texture move plane 0's color words alone, at its GPU rows,
//!   at every width (33, 37 and 61 texels too, where a texture of its own memory's blits move
//!   texels between rows, `two_planes.rs`).
//! - Samples read what a plain texture of the same texels reads, bit for bit.
//! - Metal also makes these formats over plane 1 of a two-plane surface, over planes of other
//!   OSTypes and over surfaces without planes; there it reads alpha 0, and over plane 1 its rows
//!   overlap. A blit between such a texture and another texture moves both planes' bytes at rows
//!   of plane 0's GPU rows (the defect, at this texture's rows), and a fill writes plane 0's words
//!   at its GPU rows.
//!
//! **Held** as a heap's such texture is (`heap_two_planes.rs`): one `R32G32_UINT` image (each
//! texel's color word, then its alpha byte), a shadow of the surface's bytes (`shadow.rs`,
//! `ShadowTarget::Planes`), whose copies move both planes through a staging buffer of the bytes
//! from plane 0's first to plane 1's last and the two-plane kernel, each row of each plane back on
//! its own. The texture's texels (`buffer_texture.rs`) are plane 0's color words, 4 bytes a texel,
//! which its CPU calls and blits with a buffer move, and its alpha plane, which
//! `synchronize_resource` copies too.
//!
//! **Refused** (`Unsupported`, `NotImplemented`, `Feature::Format`, naming what): without the
//! surface's facts ([`IoSurfacePlane::ycbcr`](crate::IoSurfacePlane::ycbcr), whose
//! `chroma_offset` and `chroma_bytes_per_row` name plane 1), over a surface without planes, of
//! another OSType than `'b3a8'`, over a plane other than 0, rotated, at rows of plane 0 under 16
//! bytes apart, with plane 1's rows overlapping, with plane 1 before plane 0's last texel or past
//! the buffer, and past the device's storage buffer range. A blit between such a texture and
//! another texture moves the surface's bytes as Metal's does (`two_planes/heap_bytes.rs`), and a
//! fill writes its words (`blit_fill.rs`).

use super::heap_two_planes::{HeapPlaneCopies, TwoPlaneHeapLayout};
use super::{image_error, IdUse, Storage, Texture};
use crate::buffer_texture::Texels;
use crate::degradation::Degradation;
use crate::error::{Error, RuntimeGap};
use crate::image_plan::plan_image;
use crate::linear_plan::{BufferLayout, LINEAR_ALIGNMENT};
use crate::shadow::{ShadowBuffer, ShadowCopies, ShadowSource, ShadowTarget};
use crate::shadow_merge::ScratchSpares;
use crate::texture_descriptor::{texture_usage, TextureDescriptor, TextureLayout};
use crate::tracker::UseRange;
use crate::view_plan::ViewRecord;
use crate::ycbcr::IoSurfaceYCbCr;
use crate::Device;
use ash::vk;
use metal2vulkan::error::{DiagName, Feature, Unsupported};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan::vk::{ImageUsageFlags, SampleCountFlags};
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

/// The OSType of the surfaces Metal's answers were measured over: `'b3a8'`.
const MEASURED_OSTYPE: u32 = u32::from_be_bytes(*b"b3a8");

/// A refusal of a texture of `pf` over a surface: `NotImplemented`, naming `what`.
fn refused(pf: PixelFormat, what: String) -> Error {
    let named = format!("{} {what}", pf.name());
    Error::Unsupported(Unsupported::not_implemented(Feature::Format).naming(DiagName::new(named)))
}

/// Plane 1 of the surface a two-plane extended-range texture of `pf` is asked of, given the
/// surface's `facts` and whether it has planes (`planar`): its offset and `bytesPerRow`. Refused
/// as the module docs say.
pub(crate) fn alpha_plane(
    pf: PixelFormat,
    planar: bool,
    facts: Option<IoSurfaceYCbCr>,
) -> Result<[u64; 2], Error> {
    let refused = |what: String| refused(pf, what);
    let Some(facts) = facts else {
        return Err(refused(
            "over a surface whose plane 1 the call does not give (IoSurfacePlane::ycbcr)".into(),
        ));
    };
    if !planar {
        return Err(refused("over a surface without planes".into()));
    }
    if facts.pixel_format != MEASURED_OSTYPE {
        let code = facts
            .pixel_format
            .to_be_bytes()
            .map(|b| b.escape_ascii().to_string())
            .concat();
        return Err(refused(format!("over a surface of OSType '{code}'")));
    }
    if facts.plane != 0 {
        return Err(refused(format!("of plane {}", facts.plane)));
    }
    Ok([facts.chroma_offset, facts.chroma_bytes_per_row])
}

/// Where the two planes of a texture whose color words lie at `color` (plane 0's GPU layout,
/// `linear_plan.rs`) and whose plane 1 lies at `alpha` (its offset and `bytesPerRow`) of a buffer
/// of `length` bytes lie for Metal's GPU: their layout from plane 0's first byte, and plane 1's
/// GPU layout (its offset and rows rounded down to 16 bytes). `Err` names why they are refused:
/// plane 1's rows overlap, it starts before plane 0's last texel, or it runs past the buffer.
pub(crate) fn surface_layout(
    color: &BufferLayout,
    [alpha_offset, alpha_bytes_per_row]: [u64; 2],
    length: u64,
) -> Result<(TwoPlaneHeapLayout, BufferLayout), String> {
    let width = color.row_bytes / 4;
    let height = color.rows;
    let pitch = alpha_bytes_per_row - alpha_bytes_per_row % LINEAR_ALIGNMENT;
    let alpha = BufferLayout {
        offset: alpha_offset - alpha_offset % LINEAR_ALIGNMENT,
        row_pitch: pitch,
        slice_pitch: pitch.saturating_mul(height),
        row_bytes: width,
        rows: height,
        slices: 1,
    };
    if alpha_bytes_per_row < width || (height > 1 && pitch < width) {
        return Err(format!(
            "with plane 1's rows {alpha_bytes_per_row} bytes apart ({pitch} on the GPU), under its \
             {width} bytes"
        ));
    }
    if alpha.offset < color.end() {
        return Err(format!(
            "with plane 1 at byte {alpha_offset}, before plane 0's last texel ends at byte {}",
            color.end()
        ));
    }
    if alpha.end() > length {
        return Err(format!(
            "with plane 1 ending at byte {}, past the buffer's {length}",
            alpha.end()
        ));
    }
    let layout = TwoPlaneHeapLayout::at_rows(
        [width, height],
        color.row_pitch,
        (alpha.offset - color.offset, alpha.row_pitch),
    );
    Ok((layout, alpha))
}

impl Device {
    /// The two-plane extended-range texture of `descriptor` (checked as `layout`) over `texels`,
    /// plane 0's color words of a surface, for Metal's `call`, whose plane 1 lies at `alpha` (its
    /// offset and `bytesPerRow`): its image, a shadow of both planes (the module docs). Refused
    /// as the module docs say, and as `new_texture` refuses the image.
    pub(crate) fn new_surface_planes_texture(
        &self,
        call: &'static str,
        texels: Texels,
        descriptor: &TextureDescriptor,
        layout: &TextureLayout,
        id_use: IdUse,
        alpha: [u64; 2],
    ) -> Result<Texture, Error> {
        let pf = layout.pixel_format;
        let color = texels.layout;
        let cpu = texels.cpu_layout();
        // Under 16 bytes a row, Metal's GPU reads every row past the first as memory-dependent
        // bytes (`linear_plan.rs`): not measured for these formats.
        if cpu.rows > 1 && cpu.row_pitch < LINEAR_ALIGNMENT {
            return Err(refused(
                pf,
                format!("with plane 0's rows {} bytes apart", cpu.row_pitch),
            ));
        }
        let (planes, alpha_layout) = surface_layout(&color, alpha, texels.buffer.length())
            .map_err(|what| refused(pf, what))?;
        if !HeapPlaneCopies::fit(self, &planes) {
            return Err(refused(
                pf,
                format!(
                    "of {} bytes, past the device's storage buffer range",
                    planes.span()
                ),
            ));
        }
        let facts = self
            .shared()
            .textures
            .facts(self.compile_device().profile());
        let usage = texture_usage(descriptor.usage, descriptor.swizzle);
        let plan = plan_image(layout, usage, false, &facts)
            .map_err(|refusal| image_error(call, refusal))?;
        let plan = self.with_undeclared_usage(plan, layout, usage, &facts)?;
        let copies = ImageUsageFlags::TRANSFER_SRC | ImageUsageFlags::TRANSFER_DST;
        if !plan.usage.contains(copies) || plan.samples != SampleCountFlags::COUNT_1 {
            return Err(Error::NotImplemented(RuntimeGap::Textures));
        }
        let hazard = texels.options().hazard_tracking_mode()?;
        let image = self.new_image(plan, layout, descriptor, hazard)?;
        let (raw, start) = texels.buffer.at(color.offset);
        let staging = HeapPlaneCopies::staging(self, &planes, start, hazard)?;
        let moves = Box::new(HeapPlaneCopies::new(self, &staging, &planes, hazard)?);
        let (bytes, tracked) = texels
            .buffer
            .tracked(color.offset..color.offset + planes.span());
        let copies = ShadowCopies {
            source: ShadowSource::Bytes {
                buffer: ShadowBuffer::of(&texels.buffer, raw),
                key: bytes.key,
                bytes: tracked,
                target: ShadowTarget::Planes {
                    image: image.raw.get(),
                    staging,
                    planes: moves,
                    pixel_format: pf,
                },
                alias: texels.buffer.alias().map(|alias| alias.key),
            },
            shadow_key: image.identity.key,
            shadow_range: UseRange::Image {
                aspects: vk::ImageAspectFlags::COLOR,
                levels: 0..1,
                layers: 0..1,
            },
            tracked: texels.tracked(),
            drops_writes: false,
            backing: texels.backing(),
            merge_scratch: ScratchSpares::default(),
            stood: AtomicU64::new(0),
            standing: Default::default(),
        };
        self.degrade(Degradation::ShadowedTexture, || {
            let [width, height, _] = layout.extent;
            let object = format!(
                "a {} texture of {width}x{height} over a surface's two planes",
                pf.name()
            );
            let why = "its image holds each texel's colour word and alpha byte side by side, \
                       where Metal keeps two planes in the surface's bytes: its image is a shadow \
                       of those planes, which the encoders move into it and back"
                .to_owned();
            (object, why)
        });
        let storage = Storage::Shadow {
            image: image.shared(),
            texels: Arc::new(texels.with_alpha(alpha_layout)),
            copies: Arc::new(copies),
        };
        let record = ViewRecord::of_descriptor(layout, descriptor);
        Ok(Texture::made(self, storage, record, None, id_use))
    }
}
