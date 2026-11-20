//! Blit copies from a texture to a buffer whose rows lie a stride apart that is not a multiple of
//! M, the largest power of two up to 16 dividing the bytes of the copy's row: an even number of
//! texels of 1, 2, 4 and 8 bytes, each row followed by one texel's bytes the copy must not touch.
//!
//! **Why these.** A copy whose row stride is no multiple of M is the class the UHD 630's Windows
//! driver (101.2145) placed wrong as one region, its rows floor(stride / M) x M bytes apart where
//! the offset is a multiple of M, so the runtime records each such copy as two regions. The
//! compressed formats' copies of that class are in [`crate::cases::compressed_moves`]; these are
//! the uncompressed ones, at even widths, where the row is a multiple of two texels and M is
//! larger than one texel. A 16-byte texel's rows are a multiple of 16, so its copy is never of the
//! class, and the case keeps one as the row beside it.
//!
//! Every texture is three rows high, private, filled from tight bytes in which each byte is its
//! own position plus one, so a byte in the wrong place names where it came from. Each is copied
//! out twice, at offset 0 and at one texel, into buffers of [`BEFORE`] read back whole and
//! compared byte for byte, the padding between the rows and after the last included.
//!
//! Nothing here can fault a GPU: blit copies between textures and buffers the program makes, every
//! region inside its texture and its buffer, no dispatch, no draw and no loop.

use crate::case::{made, Abort, Case, Failure, Input, MetalSource, Side};
use crate::transcript::{ByteLayout, Element, Observed, Sites, Value};
use crate::work::{Plane, Resource, Step, TextureSpec, Work};
use metal2vulkan::metal::PixelFormat;

/// The case's name, as the report and the recording spell it.
pub const NAME: &str = "copy-out-strides";

/// The byte each destination holds before the copy, so a byte no copy reached is a recorded
/// value rather than whatever was there.
pub const BEFORE: u8 = 0xcd;

/// Every texture's height in rows.
const HEIGHT: u64 = 3;

/// One texture the case copies out of: its format, its texel's bytes and its width in texels.
struct Shape {
    site: &'static str,
    format: PixelFormat,
    texel: u64,
    width: u64,
}

/// The textures, each copied out at a row stride one texel past its row.
const SHAPES: [Shape; 7] = [
    Shape {
        site: "r8uint-2-wide",
        format: PixelFormat::R8Uint,
        texel: 1,
        width: 2,
    },
    Shape {
        site: "rg8uint-2-wide",
        format: PixelFormat::RG8Uint,
        texel: 2,
        width: 2,
    },
    Shape {
        site: "r32uint-2-wide",
        format: PixelFormat::R32Uint,
        texel: 4,
        width: 2,
    },
    Shape {
        site: "r32uint-4-wide",
        format: PixelFormat::R32Uint,
        texel: 4,
        width: 4,
    },
    Shape {
        site: "rgba16uint-2-wide",
        format: PixelFormat::RGBA16Uint,
        texel: 8,
        width: 2,
    },
    Shape {
        site: "rgba16uint-4-wide",
        format: PixelFormat::RGBA16Uint,
        texel: 8,
        width: 4,
    },
    Shape {
        site: "rgba32uint-2-wide",
        format: PixelFormat::RGBA32Uint,
        texel: 16,
        width: 2,
    },
];

impl Shape {
    /// A row's bytes.
    const fn row(&self) -> u64 {
        self.texel * self.width
    }

    /// The row stride the copy out takes: one texel past the row.
    const fn stride(&self) -> u64 {
        self.row() + self.texel
    }

    /// The texture's tight bytes: each byte its own position plus one.
    fn bytes(&self) -> Vec<u8> {
        (0..self.row() * HEIGHT).map(|i| (i + 1) as u8).collect()
    }

    /// The bytes a destination copied into at `offset` takes: the rows at their stride, and the
    /// last row followed by one stride more.
    const fn length(&self, offset: u64) -> u64 {
        offset + self.stride() * (HEIGHT + 1)
    }
}

/// The offsets each texture is copied out at: 0, and one texel.
fn offsets(shape: &Shape) -> [u64; 2] {
    [0, shape.texel]
}

/// The case.
pub struct CopyOutStrides;

impl CopyOutStrides {
    /// The program: each texture filled from its tight bytes, copied out at each offset into its
    /// own buffer of [`BEFORE`], and every buffer read back whole.
    fn program() -> Work {
        let mut work = Work::new();
        let mut outs = Vec::new();
        for shape in &SHAPES {
            let texture = work.add(Resource::Texture {
                spec: TextureSpec::texture_2d(shape.format.as_raw(), shape.width, HEIGHT),
            });
            let tight = work.buffer(shape.bytes());
            work.step(Step::CopyBufferToTexture {
                buffer: tight,
                offset: 0,
                bytes_per_row: shape.row(),
                bytes_per_image: 0,
                size: [shape.width, HEIGHT, 1],
                texture,
                slice: 0,
                level: 0,
                origin: [0, 0, 0],
                plane: Plane::Whole,
            });
            for offset in offsets(shape) {
                let length = shape.length(offset);
                let out = work.buffer(vec![BEFORE; length as usize]);
                work.step(Step::CopyTextureToBuffer {
                    texture,
                    slice: 0,
                    level: 0,
                    origin: [0, 0, 0],
                    size: [shape.width, HEIGHT, 1],
                    buffer: out,
                    offset,
                    bytes_per_row: shape.stride(),
                    bytes_per_image: 0,
                    plane: Plane::Whole,
                });
                outs.push((out, length));
            }
        }
        work.step(Step::Commit);
        for (buffer, length) in outs {
            work.step(Step::ReadBuffer {
                buffer,
                offset: 0,
                length: length as usize,
            });
        }
        work
    }
}

impl Case for CopyOutStrides {
    fn name(&self) -> &'static str {
        NAME
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let shapes: Vec<u8> = SHAPES
            .iter()
            .flat_map(|shape| {
                [
                    u64::from(shape.format.as_raw()),
                    shape.texel,
                    shape.width,
                    shape.stride(),
                ]
            })
            .chain([HEIGHT, u64::from(BEFORE)])
            .flat_map(u64::to_le_bytes)
            .collect();
        Ok(vec![Input::bytes("shapes", &shapes)])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let done = made(sites, "work", side.work(&Self::program()))?;
        let mut at = 0;
        for shape in &SHAPES {
            for offset in offsets(shape) {
                let site = format!("{}/offset-{offset}", shape.site);
                let bytes = done.read(at).map_err(|why| Abort {
                    at: site.clone(),
                    failure: Failure::named("UnansweredRead", why),
                })?;
                at += 1;
                let layout = ByteLayout::buffer(0, shape.length(offset), Element::U8);
                sites.leaf(&site, Observed::bytes(bytes.to_vec(), layout));
            }
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

    /// Every texture but the 16-byte one is copied out at a stride that is no multiple of M, the
    /// largest power of two up to 16 dividing its row's bytes; the 16-byte one at a stride that
    /// is.
    #[test]
    fn every_copy_but_the_16_byte_texels_is_of_the_class() {
        let power = |bytes: u64| {
            let mut m = 1;
            while m < 16 && bytes.is_multiple_of(m * 2) {
                m *= 2;
            }
            m
        };
        for shape in &SHAPES {
            let of_class = !shape.stride().is_multiple_of(power(shape.row()));
            assert_eq!(of_class, shape.texel != 16, "{}", shape.site);
            assert!(shape.width.is_multiple_of(2), "{}", shape.site);
        }
    }

    /// The texture's bytes name their positions and none is [`BEFORE`].
    #[test]
    fn no_texel_byte_is_the_before_byte() {
        for shape in &SHAPES {
            let bytes = shape.bytes();
            assert!(bytes.len() < usize::from(BEFORE), "{}", shape.site);
            assert!(!bytes.contains(&BEFORE), "{}", shape.site);
        }
    }

    /// Each copy's rows end inside its buffer.
    #[test]
    fn every_copy_lies_inside_its_buffer() {
        for shape in &SHAPES {
            for offset in offsets(shape) {
                let end = offset + shape.stride() * (HEIGHT - 1) + shape.row();
                assert!(end <= shape.length(offset), "{}", shape.site);
            }
        }
    }
}
