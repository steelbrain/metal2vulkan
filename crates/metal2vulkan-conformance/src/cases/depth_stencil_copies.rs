//! A blit copy between two `Depth32Float_Stencil8` textures.
//!
//! `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:
//! destinationLevel:destinationOrigin:` copies "a rectangle of pixels between textures" and takes
//! no blit option: the plane options (`MTLBlitOptionDepthFromDepthStencil`,
//! `MTLBlitOptionStencilFromDepthStencil`) belong to the copies between a texture and a buffer. So
//! a copy between two textures of a combined format moves each pixel whole, its depth and its
//! stencil. The case puts that to Metal and records what each plane of each texture holds after.
//!
//! **Measured here**: every call is public Metal, so record mode runs the program and the
//! recording holds the bytes Metal moved. The test `the_recorded_copy_moved_both_planes` holds the
//! recording to the host's model: the destination's region is the source's, every other texel of
//! both textures is what was written before the copy.
//!
//! The program, on two 8 x 4 textures of `ShaderRead | RenderTarget | PixelFormatView` usage,
//! private:
//!
//! - each texture's depth plane is written from a buffer through
//!   `MTLBlitOptionDepthFromDepthStencil` and its stencil plane through
//!   `MTLBlitOptionStencilFromDepthStencil`, every texel a value of its own, the source's and the
//!   destination's apart;
//! - one copy moves the source's 5 x 2 texels from (1, 1) to the destination's (2, 2), so the
//!   region is inside both textures and at another place in each;
//! - both planes of both textures are copied out through the same options and read.
//!
//! Every depth lies in `[0, 1]` and is a float with few bits set, so no device's clamping or
//! rounding decides.
//!
//! Nothing here can fault a GPU: blit copies between textures and buffers the program makes, no
//! dispatch, no draw and no loop.

use crate::allowance::Allowance;
use crate::case::{made, Abort, Case, Failure, Input, MetalSource, Side};
use crate::transcript::{ByteLayout, Element, Gaps, Observed, Region, Sites, Value};
use crate::work::{Plane, Resource, Step, TextureSpec, Work};
use metal2vulkan::metal::{PixelFormat, TextureUsage};

/// The textures' extent.
const W: u64 = 8;
const H: u64 = 4;
/// The texels of a texture.
const N: usize = (W * H) as usize;

/// The copy: from the source's origin, of this size, to the destination's origin.
const FROM: [u64; 2] = [1, 1];
const SIZE: [u64; 2] = [5, 2];
const TO: [u64; 2] = [2, 2];

/// The format both textures are of.
const FORMAT: PixelFormat = PixelFormat::Depth32FloatStencil8;

/// The byte the read-back buffers hold before the copies out, so a byte a copy did not write is a
/// recorded value rather than whatever was there.
const UNWRITTEN: u8 = 0xee;

/// The usage both textures are made with: shader-read, a render target, and viewable in another
/// format, as a texture a pass draws into and a shader then samples is made.
fn usage() -> u32 {
    TextureUsage::SHADER_READ
        .union(TextureUsage::RENDER_TARGET)
        .union(TextureUsage::PIXEL_FORMAT_VIEW)
        .as_raw()
}

/// The two textures, in the order the case makes them.
#[derive(Clone, Copy)]
enum Texture {
    Source,
    Destination,
}

/// Texel `i`'s depth in `texture` before the copy: `(i + 1) / 64` in the source and
/// `0.5 + i / 128` in the destination, each exact in a float and in `[0, 1]`.
fn depth(texture: Texture, i: usize) -> f32 {
    let i = i as f32;
    match texture {
        Texture::Source => (i + 1.0) / 64.0,
        Texture::Destination => 0.5 + i / 128.0,
    }
}

/// Texel `i`'s stencil in `texture` before the copy, every byte of a texture its own.
fn stencil(texture: Texture, i: usize) -> u8 {
    match texture {
        Texture::Source => (i * 37 + 11) as u8,
        Texture::Destination => 0xc0 ^ i as u8,
    }
}

/// A plane's bytes before the copy: the depths' little-endian words, or the stencils.
fn written(texture: Texture, plane: Plane) -> Vec<u8> {
    match plane {
        Plane::Stencil => (0..N).map(|i| stencil(texture, i)).collect(),
        _ => (0..N)
            .flat_map(|i| depth(texture, i).to_bits().to_le_bytes())
            .collect(),
    }
}

/// The bytes a texel of `plane` takes in a buffer.
fn per(plane: Plane) -> usize {
    match plane {
        Plane::Stencil => 1,
        _ => 4,
    }
}

/// The reads, in the order the program makes them: each site's name, texture and plane.
const READS: [(&str, Texture, Plane); 4] = [
    ("destination-depth", Texture::Destination, Plane::Depth),
    ("destination-stencil", Texture::Destination, Plane::Stencil),
    ("source-depth", Texture::Source, Plane::Depth),
    ("source-stencil", Texture::Source, Plane::Stencil),
];

/// A plane as a site's layout: the texture's rows, a texel's unit an element.
fn layout(plane: Plane) -> ByteLayout {
    let per = per(plane) as u32;
    let row = W as u32 * per;
    let length = u64::from(row) * H;
    ByteLayout {
        region: Region::Buffer { offset: 0, length },
        element: match plane {
            Plane::Stencil => Element::U8,
            _ => Element::U32,
        },
        row_offset: 0,
        row_elements: W as u32,
        row_pitch: row,
        rows: H as u32,
        image_pitch: length as u32,
        images: 1,
        gaps: Gaps::Ignored,
    }
}

/// The case.
pub struct DepthStencilCopies;

impl DepthStencilCopies {
    /// The program: both planes of both textures written, one copy between the textures, and
    /// both planes of both read back.
    fn program(&self) -> Work {
        let spec = TextureSpec::texture_2d(FORMAT.as_raw(), W, H).used(usage());
        let mut work = Work::new();
        let source = work.add(Resource::Texture { spec });
        let destination = work.add(Resource::Texture { spec });
        let index = |texture: Texture| match texture {
            Texture::Source => source,
            Texture::Destination => destination,
        };
        for texture in [Texture::Destination, Texture::Source] {
            for plane in [Plane::Depth, Plane::Stencil] {
                let buffer = work.buffer(written(texture, plane));
                work.step(Step::CopyBufferToTexture {
                    buffer,
                    offset: 0,
                    bytes_per_row: W * per(plane) as u64,
                    bytes_per_image: 0,
                    size: [W, H, 1],
                    texture: index(texture),
                    slice: 0,
                    level: 0,
                    origin: [0, 0, 0],
                    plane,
                });
            }
        }
        work.step(Step::CopyTextureToTexture {
            from: source,
            from_slice: 0,
            from_level: 0,
            from_origin: [FROM[0], FROM[1], 0],
            size: [SIZE[0], SIZE[1], 1],
            to: destination,
            to_slice: 0,
            to_level: 0,
            to_origin: [TO[0], TO[1], 0],
        });
        let mut outs = Vec::new();
        for (_, texture, plane) in READS {
            let buffer = work.buffer(vec![UNWRITTEN; N * per(plane)]);
            work.step(Step::CopyTextureToBuffer {
                texture: index(texture),
                slice: 0,
                level: 0,
                origin: [0, 0, 0],
                size: [W, H, 1],
                buffer,
                offset: 0,
                bytes_per_row: W * per(plane) as u64,
                bytes_per_image: 0,
                plane,
            });
            outs.push((buffer, N * per(plane)));
        }
        work.step(Step::Commit);
        for (buffer, length) in outs {
            work.step(Step::ReadBuffer {
                buffer,
                offset: 0,
                length,
            });
        }
        work
    }
}

impl Case for DepthStencilCopies {
    fn name(&self) -> &'static str {
        "depth-stencil-copies-d32s8"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Each depth plane takes its bytes by a copy from a buffer, which carries the note of an
        // upload into a 32-bit float depth (`restricted-depth-upload`). Every depth the case
        // writes lies in [0, 1], so the note moves no byte the case reads: one declaration, which
        // loosens the notices site alone.
        vec![crate::replay::notice_allowance(
            crate::degradation::Degradation::RestrictedDepthUpload.code(),
        )]
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let words = [FORMAT.as_raw(), usage()]
            .into_iter()
            .chain([FROM, SIZE, TO].into_iter().flatten().map(|at| at as u32))
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<u8>>();
        let mut inputs = vec![Input::bytes("copy", &words)];
        for (texture, name) in [
            (Texture::Source, "source"),
            (Texture::Destination, "destination"),
        ] {
            for (plane, which) in [(Plane::Depth, "depth"), (Plane::Stencil, "stencil")] {
                let label = format!("bytes:{name}-{which}");
                inputs.push(Input::bytes(label, &written(texture, plane)));
            }
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let done = made(sites, "work", side.work(&self.program()))?;
        for (at, (site, _, plane)) in READS.into_iter().enumerate() {
            let bytes = done.read(at).map_err(|why| Abort {
                at: site.to_owned(),
                failure: Failure::named("UnansweredRead", why),
            })?;
            sites.leaf(site, Observed::bytes(bytes.to_vec(), layout(plane)));
        }
        sites.value(
            "statuses",
            Value::Lines(done.statuses.iter().map(Clone::clone).collect()),
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The host's model of a plane after the copy: the destination's region is the source's, and
    /// every other texel of both textures is what was written.
    fn after(texture: Texture, plane: Plane) -> Vec<u8> {
        let mut bytes = written(texture, plane);
        if let Texture::Destination = texture {
            let source = written(Texture::Source, plane);
            let per = per(plane);
            for y in 0..SIZE[1] {
                for x in 0..SIZE[0] {
                    let from = ((FROM[1] + y) * W + FROM[0] + x) as usize * per;
                    let to = ((TO[1] + y) * W + TO[0] + x) as usize * per;
                    bytes[to..to + per].copy_from_slice(&source[from..from + per]);
                }
            }
        }
        bytes
    }

    /// The copy's region lies inside both textures, at another place in each.
    #[test]
    fn the_region_lies_inside_both_textures() {
        for origin in [FROM, TO] {
            assert!(origin[0] + SIZE[0] <= W && origin[1] + SIZE[1] <= H);
        }
        assert_ne!(FROM, TO);
    }

    /// The model moves the region's bytes and no other: each plane of the destination differs
    /// from what was written in exactly the region's texels, and the source not at all.
    #[test]
    fn the_model_moves_the_region_alone() {
        for (site, texture, plane) in READS {
            let moved = (written(texture, plane).chunks(per(plane)))
                .zip(after(texture, plane).chunks(per(plane)))
                .filter(|(before, after)| before != after)
                .count();
            let expected = match texture {
                Texture::Source => 0,
                Texture::Destination => (SIZE[0] * SIZE[1]) as usize,
            };
            assert_eq!(moved, expected, "{site}");
            assert_eq!(layout(plane).extent(), N * per(plane), "{site}");
        }
    }

    /// The committed recording holds the host's model at every site: Metal's copy between two
    /// `Depth32Float_Stencil8` textures moved both planes of the region and nothing else.
    ///
    /// The recording states Metal's own answer, so a release in which Metal moves one plane alone
    /// shows up as a moved recording and this test names the site.
    #[test]
    fn the_recorded_copy_moved_both_planes() {
        let name = DepthStencilCopies.name();
        let recording = crate::recording::read(name).unwrap_or_else(|why| {
            panic!("no recording of {name}: {why}; run `conformance record`")
        });
        let mut differ = Vec::new();
        for answer in &recording.answers {
            for outcome in &answer.outcomes {
                for (site, texture, plane) in READS {
                    let Some(node) = outcome.transcript.root.child(site) else {
                        differ.push(format!("no site {site}"));
                        continue;
                    };
                    let Value::Bytes(bytes) = &node.observed.value else {
                        differ.push(format!("{site}: no bytes"));
                        continue;
                    };
                    let model = after(texture, plane);
                    if *bytes != model {
                        differ.push(format!(
                            "{site}: Metal holds {bytes:02x?}, the model {model:02x?}"
                        ));
                    }
                }
            }
        }
        assert!(
            differ.is_empty(),
            "Metal's recorded copy is not the model:\n{}",
            differ.join("\n")
        );
    }
}
