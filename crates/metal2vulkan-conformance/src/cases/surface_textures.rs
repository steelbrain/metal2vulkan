//! A texture of an ordinary pixel format over one plane of an IOSurface: what a write of its
//! texels and two reads of them answer.
//!
//! | case | the surface | the texture |
//! |---|---|---|
//! | `surface-textures-bgra-bgra8` | `'BGRA'`, 64x32, one plane of 4-byte texels, 256 bytes a row | `BGRA8Unorm` over it |
//! | `surface-textures-rgha-rgba16f` | `'RGhA'`, 64x32, one plane of 8-byte texels, 512 bytes a row | `RGBA16Float` over it |
//! | `surface-textures-420v-luma-r8` | `'420v'`, 64x32: a 64x32 plane of bytes at 0, then a 32x16 plane of byte pairs at 2,048, each 64 bytes a row | `R8Unorm` over plane 0 |
//! | `surface-textures-420v-chroma-rg8` | the same | `RG8Unorm` over plane 1 |
//! | `surface-textures-x420-luma-r16` | `'x420'`, 64x32: a 64x32 plane of 16-bit words at 0, then a 32x16 plane of word pairs at 4,096, each 128 bytes a row | `R16Unorm` over plane 0 |
//! | `surface-textures-x420-chroma-rg16` | the same | `RG16Unorm` over plane 1 |
//!
//! Each case makes the texture with `newTextureWithDescriptor:iosurface:plane:` (2D, `Shared`,
//! shader-read usage), writes every texel with `replaceRegion:mipmapLevel:withBytes:bytesPerRow:`,
//! reads them back with `getBytes:bytesPerRow:fromRegion:mipmapLevel:`, and reads them again on
//! the GPU, with a blit's `copyFromTexture:` into a `Shared` buffer. The bytes written name their
//! texel and byte, and each is below `0x40`, so every `RGBA16Float` channel is a finite half and
//! a copy keeps its bits.
//!
//! **What each side makes.** Metal's side lays the surface as the table says and reads the layout
//! back before a texture lies over it (`metal_surface`). The old runtime's side makes a buffer of
//! the surface's length that stands for it, as the runtime takes an IOSurface
//! (`Buffer::new_texture_with_iosurface_plane`). The new runtime's side makes the surface as its
//! consumer does: an exported surface of the planes (`Device::new_exported_surface`), laid by the
//! runtime's own rule, which the device's queries make in place or in the copy form. The replay
//! prints, for each case, how many surfaces of each form it made. A case reads the texture's texels
//! alone, never the surface's bytes, so where a side laid the planes is not compared.

use crate::allowance::Allowance;
use crate::case::{
    call, made, Abort, Case, Input, MetalSource, Side, SurfacePlane, SurfaceTexture,
};
use crate::render::{Plane, ReadAt, RenderWork};
use crate::transcript::{Aspect, ByteLayout, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::PixelFormat;

/// The surface's extent in texels: its luma plane's, and a one-plane surface's only plane's.
const SIZE: [u32; 2] = [64, 32];
/// A 4:2:0 surface's chroma plane: half the surface each way.
const CHROMA: [u32; 2] = [SIZE[0] / 2, SIZE[1] / 2];

/// One case: the surface, and the plane the texture lies over.
pub struct SurfaceTextures {
    /// The case's name, a literal, which is also its recording's file name.
    name: &'static str,
    surface: SurfaceTexture,
}

/// A surface of one plane, made without a plane dictionary: `size` texels of `texel` bytes,
/// `bytes_per_row` apart.
fn one_plane(
    ostype: [u8; 4],
    format: PixelFormat,
    texel: u64,
    bytes_per_row: u64,
) -> SurfaceTexture {
    SurfaceTexture {
        ostype,
        planar: false,
        planes: vec![SurfacePlane {
            format,
            size: SIZE,
            texel,
            offset: 0,
            bytes_per_row,
        }],
        length: bytes_per_row * u64::from(SIZE[1]),
        plane: 0,
    }
}

/// A 4:2:0 surface of two planes, made with a plane dictionary: the luma plane of `code`-byte
/// texels at 0, then the chroma plane of pairs of them, both `bytes_per_row` apart, and the
/// texture over `plane`.
fn two_plane(
    ostype: [u8; 4],
    [luma, chroma]: [PixelFormat; 2],
    code: u64,
    bytes_per_row: u64,
    plane: u32,
) -> SurfaceTexture {
    let chroma_at = bytes_per_row * u64::from(SIZE[1]);
    SurfaceTexture {
        ostype,
        planar: true,
        planes: vec![
            SurfacePlane {
                format: luma,
                size: SIZE,
                texel: code,
                offset: 0,
                bytes_per_row,
            },
            SurfacePlane {
                format: chroma,
                size: CHROMA,
                texel: 2 * code,
                offset: chroma_at,
                bytes_per_row,
            },
        ],
        length: chroma_at + bytes_per_row * u64::from(CHROMA[1]),
        plane,
    }
}

/// Every case of the family: its name and its surface.
fn table() -> Vec<(&'static str, SurfaceTexture)> {
    let v420 = [PixelFormat::R8Unorm, PixelFormat::RG8Unorm];
    let x420 = [PixelFormat::R16Unorm, PixelFormat::RG16Unorm];
    vec![
        (
            "surface-textures-bgra-bgra8",
            one_plane(*b"BGRA", PixelFormat::BGRA8Unorm, 4, 256),
        ),
        (
            "surface-textures-rgha-rgba16f",
            one_plane(*b"RGhA", PixelFormat::RGBA16Float, 8, 512),
        ),
        (
            "surface-textures-420v-luma-r8",
            two_plane(*b"420v", v420, 1, 64, 0),
        ),
        (
            "surface-textures-420v-chroma-rg8",
            two_plane(*b"420v", v420, 1, 64, 1),
        ),
        (
            "surface-textures-x420-luma-r16",
            two_plane(*b"x420", x420, 2, 128, 0),
        ),
        (
            "surface-textures-x420-chroma-rg16",
            two_plane(*b"x420", x420, 2, 128, 1),
        ),
    ]
}

/// Every case of the family.
pub fn all() -> Vec<Box<dyn Case>> {
    (table().into_iter())
        .map(|(name, surface)| Box::new(SurfaceTextures { name, surface }) as Box<dyn Case>)
        .collect()
}

impl SurfaceTextures {
    /// The plane the texture lies over.
    fn plane(&self) -> SurfacePlane {
        self.surface.planes[self.surface.plane as usize]
    }

    /// The texture's rows, packed: one texel after another.
    fn row_pitch(&self) -> u32 {
        let plane = self.plane();
        plane.size[0] * plane.texel as u32
    }

    /// The bytes the case writes, packed rows of the texture: byte `b` of texel `(x, y)` is
    /// `(x + 3y + 11b) mod 64`.
    fn texels(&self) -> Vec<u8> {
        let plane = self.plane();
        let [width, height] = plane.size;
        let texel = plane.texel as u32;
        let mut bytes = Vec::with_capacity((self.row_pitch() * height) as usize);
        for y in 0..height {
            for x in 0..width {
                bytes.extend((0..texel).map(|b| ((x + 3 * y + 11 * b) % 64) as u8));
            }
        }
        bytes
    }

    /// The layout both reads are recorded under: one row of the texture a row, a byte an element,
    /// so a diff names the texel's row and byte.
    fn layout(&self) -> ByteLayout {
        let [width, height] = self.plane().size;
        let pitch = self.row_pitch();
        ByteLayout {
            region: Region::Texture {
                level: 0,
                slice: 0,
                aspect: Aspect::Color,
                sample: 0,
                origin: [0, 0, 0],
                size: [width, height, 1],
            },
            element: Element::U8,
            row_offset: 0,
            row_elements: pitch,
            row_pitch: pitch,
            rows: height,
            image_pitch: pitch * height,
            images: 1,
            // The rows are packed, so the sites cover every byte.
            gaps: Gaps::Ignored,
        }
    }
}

impl Case for SurfaceTextures {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // The GPU read lands in a `Shared` buffer, which is what
        // [`crate::replay::shared_memory_notice`] is for; it loosens the notices site alone.
        //
        // The old runtime lays the plane's texture over a buffer, and where the device's own
        // linear image does not lie at the plane's pitch it reads and writes the texels through
        // a copy, which it announces once per device (`shadowed-texture`). Metal announces
        // nothing, so the notices site differs wherever the copy is taken, and the notice is
        // what justifies it; it too loosens the notices site alone. Measured on the Radeon 780M
        // (RADV PHOENIX, 53559): the four plane cases, whose rows are 64 and 128 bytes, take the
        // copy and the two with 256- and 512-byte rows do not; lavapipe takes it for none.
        // metal2vulkan-rt announces neither: it counts its surface forms instead.
        vec![
            crate::replay::shared_memory_notice(),
            crate::replay::notice_allowance(
                metal2vulkan_runtime::Degradation::ShadowedTexture.code(),
            ),
        ]
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let surface = &self.surface;
        let mut layout = Vec::new();
        layout.push(u8::from(surface.planar));
        layout.extend(surface.length.to_le_bytes());
        layout.extend(surface.plane.to_le_bytes());
        for plane in &surface.planes {
            layout.extend(plane.format.as_raw().to_le_bytes());
            for word in plane.size {
                layout.extend(word.to_le_bytes());
            }
            for word in [plane.texel, plane.offset, plane.bytes_per_row] {
                layout.extend(word.to_le_bytes());
            }
        }
        Ok(vec![
            Input::bytes("surface:ostype", &surface.ostype),
            Input::bytes("surface:layout", &layout),
            Input::bytes("texture:texels", &self.texels()),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let [width, height] = self.plane().size;
        let pitch = self.row_pitch();
        let length = (pitch * height) as usize;
        let texture = made(sites, "texture", side.surface_texture(&self.surface))?;
        call(
            sites,
            "replace-region",
            side.replace_region(texture, [width, height], pitch, &self.texels()),
            |_| Value::Absent,
        )?;
        let read = made(
            sites,
            "get-bytes",
            side.read_texture(texture, [width, height], pitch),
        )?;
        sites.push(Node::leaf("texels", Observed::bytes(read, self.layout())));
        let into = made(sites, "buffer", side.zeroed_buffer(length))?;
        let work = [RenderWork::ReadTexture {
            texture,
            into,
            offset: 0,
            row_pitch: pitch,
            size: [width, height],
            plane: Plane::Whole,
            at: ReadAt::BASE,
        }];
        let outcome = call(sites, "blit", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;
        if let Some(failure) = outcome.refusal(0) {
            sites.push(Node::leaf(
                "copy-from-texture",
                Observed::value(failure.recorded()),
            ));
            return Err(Abort {
                at: "copy-from-texture".to_owned(),
                failure: failure.clone(),
            });
        }
        let copied = made(sites, "read-buffer", side.read_buffer(into, 0, length))?;
        sites.push(Node::leaf("copied", Observed::bytes(copied, self.layout())));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every case's plane holds its texture: each plane's rows fit its pitch, and the planes lie
    /// in the surface's length one after the other.
    #[test]
    fn every_plane_lies_inside_its_surface() {
        for (name, surface) in table() {
            let mut end = 0;
            for plane in &surface.planes {
                assert!(
                    u64::from(plane.size[0]) * plane.texel <= plane.bytes_per_row,
                    "{name}"
                );
                assert!(plane.offset >= end, "{name}");
                end = plane.offset + plane.bytes_per_row * u64::from(plane.size[1]);
            }
            assert_eq!(end, surface.length, "{name}");
            assert!((surface.plane as usize) < surface.planes.len(), "{name}");
        }
    }

    /// Metal's recorded reads are the bytes the case wrote, both of them, in every run: what Metal
    /// answers is the texels themselves, so a side that answers them answers Metal.
    #[test]
    fn every_recorded_read_is_what_the_case_wrote() {
        let mut differ = Vec::new();
        for (name, surface) in table() {
            let case = SurfaceTextures { name, surface };
            let recording = crate::recording::read(name)
                .unwrap_or_else(|why| panic!("{name}: no recording ({why})"));
            for answer in &recording.answers {
                for outcome in &answer.outcomes {
                    for site in ["texels", "copied"] {
                        let held = outcome.transcript.root.child(site);
                        match held.map(|node| &node.observed.value) {
                            Some(Value::Bytes(bytes)) if *bytes == case.texels() => {}
                            _ => differ.push(format!("{name}/{site}")),
                        }
                    }
                }
            }
        }
        assert!(differ.is_empty(), "Metal read other bytes at {differ:?}");
    }

    /// The bytes written are below `0x40` each, so every half of an `RGBA16Float` texel is
    /// finite, and they name their texel: no two texels of a row hold the same bytes.
    #[test]
    fn the_written_texels_are_finite_and_tell_texels_apart() {
        for (name, surface) in table() {
            let case = SurfaceTextures { name, surface };
            let bytes = case.texels();
            assert!(bytes.iter().all(|&byte| byte < 0x40), "{name}");
            let texel = case.plane().texel as usize;
            let row = &bytes[..case.row_pitch() as usize];
            let mut seen: Vec<&[u8]> = row.chunks(texel).collect();
            let all = seen.len();
            seen.sort();
            seen.dedup();
            assert_eq!(seen.len(), all, "{name}");
        }
    }
}
