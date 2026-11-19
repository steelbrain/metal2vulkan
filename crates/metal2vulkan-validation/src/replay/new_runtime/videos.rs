//! Video surfaces: the textures over the planes of a buffer made as a video surface's pages
//! (`m2v.newDmaBufBufferOfLengthPaddingUnobserved`), played as the product makes them: one
//! exported surface of the runtime's (`rt::Device::new_exported_surface`) for the buffer, made at
//! its first texture from what the texture reads of the surface (its pixel format, width and
//! height), and every texture over a plane a view of that plane's texture at the texture's own
//! index. A picture landed (`m2v.externalWriteDone`) is the surface's acquire
//! (`rt::Surface::external_writes_done`). The stream holds no bytes of the pictures (the buffer is
//! Private, and another API wrote it), so the play writes none: what the textures read is the
//! surface's memory as it is, as the recorded buffer's was. The buffer stays, and the textures
//! read nothing of it.
//!
//! The surface's handle is the play's choice of the consumer's decoder route (`--video-handle`):
//! a dma-buf, the stream's own, with the layout the product passes for a VA-API import (pitch and
//! start multiples of 256, each plane holding the coded picture's rows: the luma's height rounded
//! up to 16, the chroma half that), or an opaque fd with the layout it passes for NVIDIA's (pitch
//! multiple 4, start multiple 1, no rows below).
//! The planes ask shader read and write, the usage the stream's textures over them ask; a device
//! whose image in place lacks storage (`rt::Gap::SurfaceUsage`) gets planes of shader read alone,
//! counted.
//!
//! The pixel formats played so are the two-plane 4:2:0 ones, 8-bit (`420v`, `420f`: R8 and RG8
//! planes) and 10-bit in 16 (`x420`, `xf20`: R16 and RG16), the chroma plane half the luma's
//! width and height, rounded up. A buffer whose first texture reads no facts or another pixel
//! format plays as any surface's pages do (`surfaces`), counted.

use super::{failed, whole_slices, NewRuntime, Object};
use crate::replay::args::{refuse, Fields, SURFACE_FACT_KEYS};
use crate::replay::player::{Made, PlayError};
use crate::replay::stream::Call;
use metal2vulkan_rt as rt;
use std::collections::{HashMap, HashSet};

/// The four characters of a pixel format of a surface, big-endian.
const fn fourcc(name: &[u8; 4]) -> u64 {
    u32::from_be_bytes(*name) as u64
}

/// The planes' pixel formats of a two-plane 4:2:0 surface of pixel format `raw`, luma first.
fn plane_formats(raw: u64) -> Option<[rt::PixelFormat; 2]> {
    const NV12: [u64; 2] = [fourcc(b"420v"), fourcc(b"420f")];
    const P010: [u64; 2] = [fourcc(b"x420"), fourcc(b"xf20")];
    if NV12.contains(&raw) {
        return Some([rt::PixelFormat::R8Unorm, rt::PixelFormat::RG8Unorm]);
    }
    if P010.contains(&raw) {
        return Some([rt::PixelFormat::R16Unorm, rt::PixelFormat::RG16Unorm]);
    }
    None
}

/// The consumer's decoder route: the surface's handle and the layout its other API asks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoHandle {
    /// A dma-buf, as a VA-API decoder imports it.
    DmaBuf,
    /// An opaque fd, as NVIDIA's route takes it.
    OpaqueFd,
}

impl VideoHandle {
    /// `dma-buf` or `opaque-fd`.
    pub fn parse(name: &str) -> Option<VideoHandle> {
        match name {
            "dma-buf" => Some(VideoHandle::DmaBuf),
            "opaque-fd" => Some(VideoHandle::OpaqueFd),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            VideoHandle::DmaBuf => "dma-buf",
            VideoHandle::OpaqueFd => "opaque-fd",
        }
    }

    fn export(self) -> rt::ExportHandle {
        match self {
            VideoHandle::DmaBuf => rt::ExportHandle::DmaBuf,
            VideoHandle::OpaqueFd => rt::ExportHandle::OpaqueFd,
        }
    }

    /// The pitch multiple, the start multiple, and each plane's rows below it for a luma of
    /// `height` rows and a chroma of `chroma` rows.
    fn layout(self, height: u32, chroma: u32) -> (u64, u64, [u32; 2]) {
        match self {
            VideoHandle::DmaBuf => {
                let rows = height.next_multiple_of(16);
                (256, 256, [rows - height, (rows / 2).saturating_sub(chroma)])
            }
            VideoHandle::OpaqueFd => (4, 1, [0, 0]),
        }
    }
}

/// A buffer's surface.
struct Video {
    surface: rt::Surface,
    /// The handle the other API would import, kept open while the surface lives.
    _export: rt::SurfaceExport,
}

/// The play's video surfaces.
pub(super) struct Videos {
    handle: VideoHandle,
    /// The live buffers made as a video surface's pages, not yet played otherwise.
    pages: HashSet<u64>,
    /// Each such buffer's surface, from its first texture.
    made: HashMap<u64, Video>,
    counts: Counts,
}

#[derive(Default)]
struct Counts {
    /// Surfaces made in place, and in the copy form by its reason.
    in_place: u64,
    copied: HashMap<String, u64>,
    /// Surfaces whose planes lost shader write (`rt::Gap::SurfaceUsage`).
    read_only: u64,
    /// Textures made as views of a plane.
    textures: u64,
    /// Acquires, one a picture landed in a surface, and pictures landed in a buffer before its
    /// first texture, with no surface to acquire.
    acquires: u64,
    before_surface: u64,
    /// Buffers played as any surface's pages: their first texture read no facts or another
    /// pixel format.
    otherwise: u64,
}

impl Videos {
    pub(super) fn new(handle: VideoHandle) -> Videos {
        Videos {
            handle,
            pages: HashSet::new(),
            made: HashMap::new(),
            counts: Counts::default(),
        }
    }

    /// Buffer `buffer` made as a video surface's pages.
    pub(super) fn pages(&mut self, buffer: u64) {
        self.pages.insert(buffer);
    }

    /// Whether buffer `buffer` plays as a video surface.
    pub(super) fn is_video(&self, buffer: u64) -> bool {
        self.pages.contains(&buffer)
    }

    /// `object` released: a buffer's surface goes with it (the runtime keeps its memory while a
    /// texture over a plane lives).
    pub(super) fn released(&mut self, object: u64) {
        self.pages.remove(&object);
        self.made.remove(&object);
    }

    /// The executed counters.
    pub(super) fn counted(&self) -> [(&'static str, u64); 6] {
        let c = &self.counts;
        [
            ("video_surfaces_in_place", c.in_place),
            ("video_surfaces_copied", c.copied.values().sum()),
            ("video_surface_textures", c.textures),
            ("video_surface_acquires", c.acquires),
            ("video_pictures_before_surface", c.before_surface),
            ("video_buffers_otherwise", c.otherwise),
        ]
    }

    /// The report's `video surfaces` line.
    pub(super) fn line(&self) -> String {
        let c = &self.counts;
        let mut copied: Vec<String> = (c.copied.iter())
            .map(|(why, n)| format!("{why} {n}"))
            .collect();
        copied.sort();
        format!(
            "video surfaces through the surface call ({}): {} in place, {} in the copy form ({}), \
             {} with planes of shader read alone; {} textures over their planes, {} acquires, {} \
             pictures landed before a buffer's first texture; {} buffers played as any surface's \
             pages",
            self.handle.name(),
            c.in_place,
            c.copied.values().sum::<u64>(),
            if copied.is_empty() {
                "-".to_string()
            } else {
                copied.join(", ")
            },
            c.read_only,
            c.textures,
            c.acquires,
            c.before_surface,
            c.otherwise,
        )
    }
}

impl NewRuntime {
    /// `newTextureWithDescriptor:iosurface:plane:` of `descriptor` over plane `plane` of video
    /// surface pages `call.receiver`, whose fields are `fields`: a view of the plane's texture of
    /// the buffer's surface, made at the buffer's first texture. None where the buffer plays as
    /// any surface's pages instead, from its first texture on.
    pub(super) fn video_texture(
        &mut self,
        call: &Call,
        d: &rt::TextureDescriptor,
        plane: u64,
        fields: &Fields,
    ) -> Result<Option<Option<(Object, Made)>>, PlayError> {
        let buffer = call.receiver;
        if !self.videos.made.contains_key(&buffer) {
            let Some(video) = self.video_surface(call, fields)? else {
                self.videos.pages.remove(&buffer);
                self.videos.counts.otherwise += 1;
                return Ok(None);
            };
            self.videos.made.insert(buffer, video);
        }
        let video = &self.videos.made[&buffer];
        let planes = video.surface.planes();
        let Some(texture) = usize::try_from(plane).ok().and_then(|at| planes.get(at)) else {
            let why = format!("plane {plane} of a video surface of {}", planes.len());
            return Err(refuse(call, why));
        };
        let texture = texture.clone();
        let (levels, slices) = (0..d.mipmap_level_count, whole_slices(d));
        let make = |index| {
            let (levels, slices) = (levels.clone(), slices.clone());
            let (format, kind) = (d.pixel_format, d.texture_type);
            texture.new_texture_view_with_swizzle(format, kind, levels, slices, d.swizzle, index)
        };
        let made = self.texture_or_substitute(call, d.usage, d, make)?;
        self.videos.counts.textures += 1;
        Ok(Some(made))
    }

    /// The surface of the buffer whose first texture `call` makes, of the pixel format, width and
    /// height its facts name: none where they name no two-plane 4:2:0 format.
    fn video_surface(&mut self, call: &Call, fields: &Fields) -> Result<Option<Video>, PlayError> {
        let Some(facts) = fields.fields("facts", &SURFACE_FACT_KEYS)? else {
            return Ok(None);
        };
        let raw = facts.required("pixelFormat", Fields::uint)?;
        let Some(formats) = plane_formats(raw) else {
            return Ok(None);
        };
        let width = facts.required("width", Fields::count)?;
        let height = facts.required("height", Fields::count)?;
        let chroma = [width.div_ceil(2), height.div_ceil(2)];
        let handle = self.videos.handle;
        let (pitch_multiple, start_multiple, rows_below) = handle.layout(height, chroma[1]);
        let extents = [[width, height], chroma];
        let make = |usage| {
            let specs: [rt::PlaneSpec; 2] = std::array::from_fn(|at| {
                let [w, h] = extents[at];
                let mut descriptor =
                    rt::TextureDescriptor::new(rt::TextureType::Type2D, formats[at], w, h);
                descriptor.usage = usage;
                rt::PlaneSpec {
                    descriptor,
                    index: None,
                    rows_below: rows_below[at],
                }
            });
            let spec = rt::SurfaceSpec {
                handle: handle.export(),
                pitch_multiple,
                start_multiple,
                planes: &specs,
            };
            self.device.new_exported_surface(spec)
        };
        let read = rt::TextureUsage::SHADER_READ;
        let made = match make(read | rt::TextureUsage::SHADER_WRITE) {
            Err(rt::Error::Unsupported(rt::Gap::SurfaceUsage { .. })) => {
                self.videos.counts.read_only += 1;
                make(read)
            }
            made => made,
        };
        let (surface, export) = made.map_err(failed(call))?;
        match export.form {
            rt::SurfaceForm::InPlace => self.videos.counts.in_place += 1,
            rt::SurfaceForm::Copy(why) => {
                *self
                    .videos
                    .counts
                    .copied
                    .entry(format!("{why:?}"))
                    .or_default() += 1;
            }
        }
        Ok(Some(Video {
            surface,
            _export: export,
        }))
    }

    /// `m2v.externalWriteDone` on video surface pages `call.receiver`: a picture landed, the
    /// surface's acquire; none where the buffer has no texture yet, counted.
    pub(super) fn video_picture(
        &mut self,
        call: &Call,
    ) -> Result<Option<(Object, Made)>, PlayError> {
        let Some(video) = self.videos.made.get(&call.receiver) else {
            self.videos.counts.before_surface += 1;
            return Ok(None);
        };
        video.surface.external_writes_done().map_err(failed(call))?;
        self.videos.counts.acquires += 1;
        Ok(None)
    }
}
