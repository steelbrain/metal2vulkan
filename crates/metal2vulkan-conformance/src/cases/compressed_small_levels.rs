//! Block-compressed copies of a level smaller than one block, at the whole block's size.
//!
//! A guest copied 4 x 4 texels at level 9 of a 2048 x 1024 BC3 texture of 12 levels, Managed into
//! Private. Level 9 is 4 x 2 texels, one block, and Apple's page for the copy asks for the size
//! clamped to the texture's edge ("If the block extends outside the bounds of the texture, clamp
//! sourceSize to the edge of the texture"); plain Metal copies the whole block. This family puts
//! that size to Metal at levels 9 (4 x 2), 10 (2 x 1) and 11 (1 x 1) of a 2048 x 1024 texture of
//! 12 levels, by every call that names a region:
//!
//! - `t2t`: `copyFromTexture:...toTexture:...` from a managed source into a private texture;
//! - `b2t`: `copyFromBuffer:...toTexture:...` of the level's one block into a private texture;
//! - `replace`: `replaceRegion:` of the block into a managed texture;
//! - `t2b`: `copyFromTexture:...toBuffer:...` of the source into 64 bytes of [`OUT_BEFORE`];
//! - `get`: `getBytes:` of the source into 64 bytes of [`OUT_BEFORE`].
//!
//! Levels 8 to 11 of the source hold blocks whose byte `k` (in the level's rows of blocks) is
//! `0x11 * level + 3 * k + 1`, the bytes of the probe `compressed_copies_of_small_levels.m` that
//! first measured this, and every destination texture's levels 8 to 11 hold [`FILL`], written at
//! each level's own extent. Each destination texture's levels 8 to 11 are copied out at their own
//! extent into one buffer of [`BEFORE`], packed, and read whole; each byte compared.
//!
//! **One case per format** (BC1_RGBA, BC3_RGBA), so a venue that holds no texture of one answers
//! its typed refusal there and the suite counts that format as
//! [`LacksFeature`](crate::verdict::Verdict::LacksFeature).
//!
//! Metal's validation layer takes every call but the copy from a buffer, which it stops:
//! `(destinationOrigin.y + sourceSize.height)(4) must be <= height(2)`. Plain Metal, the oracle,
//! takes all five and leaves each level's block where the clamped size leaves it.
//!
//! Nothing here can fault a GPU: host copies and blit copies between textures and buffers the
//! program makes, no dispatch, no draw and no loop, every region inside its level's last block.

use crate::allowance::Allowance;
use crate::case::{made, Abort, Case, Failure, Input, MetalSource, Side};
use crate::cases::compressed_moves::{block_allowances, fingerprint};
use crate::transcript::{ByteLayout, Element, Observed, Sites, Value};
use crate::work::{Plane, Rect, Resource, Step, TextureSpec, Work};
use metal2vulkan::metal::PixelFormat;

/// The byte every destination texture's levels 8 to 11 hold before the copies.
pub const FILL: u8 = 0xdd;

/// The byte a level read-back's buffer holds before it, so a byte no copy reached is a recorded
/// value.
pub const BEFORE: u8 = 0xcd;

/// The byte a buffer the source is copied into, or the bytes `getBytes:` writes, hold before.
pub const OUT_BEFORE: u8 = 0xee;

/// The texture's extent and level count.
const WIDTH: u64 = 2048;
const HEIGHT: u64 = 1024;
const LEVELS: u64 = 12;

/// The levels each texture's blocks are written in and read back from.
const WRITTEN: std::ops::RangeInclusive<u64> = 8..=11;

/// The levels smaller than a block, each copied at the whole block's size.
const SMALL: std::ops::RangeInclusive<u64> = 9..=11;

/// The bytes a buffer destination of the source, and `getBytes:`, take.
const OUT: u64 = 64;

/// One format: the case's name, the raw `MTLPixelFormat` and its block's bytes.
struct Format {
    name: &'static str,
    format: u32,
    block: u64,
}

const FORMATS: [Format; 2] = [
    Format {
        name: "compressed-small-levels-bc1-rgba",
        format: PixelFormat::BC1RGBA.as_raw(),
        block: 8,
    },
    Format {
        name: "compressed-small-levels-bc3-rgba",
        format: PixelFormat::BC3RGBA.as_raw(),
        block: 16,
    },
];

/// Level `level`'s extent in texels.
fn extent(level: u64) -> [u64; 3] {
    [(WIDTH >> level).max(1), (HEIGHT >> level).max(1), 1]
}

/// Level `level`'s blocks across and down.
fn blocks(level: u64) -> [u64; 2] {
    let [w, h, _] = extent(level);
    [w.div_ceil(4), h.div_ceil(4)]
}

/// Level `level`'s bytes in rows of blocks, packed: the source's, or `fill`.
fn level_bytes(level: u64, block: u64, fill: Option<u8>) -> Vec<u8> {
    let [across, down] = blocks(level);
    (0..across * down * block)
        .map(|k| fill.unwrap_or(((0x11 * level + 3 * k + 1) & 0xff) as u8))
        .collect()
}

/// Level `level`, whole.
fn whole(level: u64) -> Rect {
    let [w, h, _] = extent(level);
    Rect::new(0, 0, w, h)
}

/// The block's 4 x 4 texels at (0, 0).
const BLOCK: Rect = Rect::new(0, 0, 4, 4);

/// The bytes of levels 8 to 11 of a destination texture, packed in level order, as the copies
/// leave them: the source's at the small levels, [`FILL`] at level 8.
fn landed(block: u64) -> Vec<u8> {
    WRITTEN
        .flat_map(|level| level_bytes(level, block, (level == 8).then_some(FILL)))
        .collect()
}

/// The steps' read-backs, in the order the program asks for them: each site's name and length.
fn reads(block: u64) -> Vec<(String, u64)> {
    let levels = landed(block).len() as u64;
    let mut reads: Vec<(String, u64)> = ["t2t", "b2t", "replace"]
        .into_iter()
        .map(|kind| (format!("{kind}/levels-8-to-11"), levels))
        .collect();
    for kind in ["t2b", "get"] {
        for level in SMALL {
            reads.push((format!("{kind}/level-{level}"), OUT));
        }
    }
    reads
}

/// The case for one format.
pub struct CompressedSmallLevels {
    at: usize,
}

/// Every format's case, in `FORMATS` order.
pub fn all() -> Vec<CompressedSmallLevels> {
    (0..FORMATS.len())
        .map(|at| CompressedSmallLevels { at })
        .collect()
}

impl CompressedSmallLevels {
    fn format(&self) -> &'static Format {
        &FORMATS[self.at]
    }

    /// The program, in three command buffers so each one's copies read what the one before wrote:
    /// the textures written; each small level's block copied by each call at 4 x 4; the
    /// destinations' levels 8 to 11 copied out. Then every read-back, in the order [`reads`]
    /// names.
    pub fn program(&self) -> Work {
        let Format { format, block, .. } = *self.format();
        let spec = TextureSpec::texture_2d(format, WIDTH, HEIGHT).levels(LEVELS);
        let mut work = Work::new();
        let source = work.add(Resource::Texture {
            spec: spec.managed(),
        });
        let into = work.add(Resource::Texture { spec });
        let from_buffer = work.add(Resource::Texture { spec });
        let replaced = work.add(Resource::Texture {
            spec: spec.managed(),
        });
        // The textures' blocks, each level at its own extent: the source's by the host, a managed
        // destination's by the host, a private one's from a buffer.
        for level in WRITTEN {
            let row = blocks(level)[0] * block;
            for (texture, fill) in [(source, None), (replaced, Some(FILL))] {
                work.step(Step::ReplaceRegion {
                    texture,
                    level,
                    slice: 0,
                    region: whole(level),
                    bytes: level_bytes(level, block, fill),
                    rows: blocks(level)[1],
                    bytes_per_row: row,
                    bytes_per_image: 0,
                });
            }
            for texture in [into, from_buffer] {
                let filled = work.buffer(level_bytes(level, block, Some(FILL)));
                work.step(Step::CopyBufferToTexture {
                    buffer: filled,
                    offset: 0,
                    bytes_per_row: row,
                    bytes_per_image: 0,
                    size: extent(level),
                    texture,
                    slice: 0,
                    level,
                    origin: [0; 3],
                    plane: Plane::Whole,
                });
            }
        }
        work.step(Step::Commit);
        // Each small level's block at 4 x 4, by each call.
        let mut outs = Vec::new();
        for level in SMALL {
            work.step(Step::CopyTextureToTexture {
                from: source,
                from_slice: 0,
                from_level: level,
                from_origin: [0; 3],
                size: [4, 4, 1],
                to: into,
                to_slice: 0,
                to_level: level,
                to_origin: [0; 3],
            });
            let bytes = work.buffer(level_bytes(level, block, None));
            work.step(Step::CopyBufferToTexture {
                buffer: bytes,
                offset: 0,
                bytes_per_row: block,
                bytes_per_image: 0,
                size: [4, 4, 1],
                texture: from_buffer,
                slice: 0,
                level,
                origin: [0; 3],
                plane: Plane::Whole,
            });
            work.step(Step::ReplaceRegion {
                texture: replaced,
                level,
                slice: 0,
                region: BLOCK,
                bytes: level_bytes(level, block, None),
                rows: 1,
                bytes_per_row: block,
                bytes_per_image: 0,
            });
            let out = work.buffer(vec![OUT_BEFORE; OUT as usize]);
            work.step(Step::CopyTextureToBuffer {
                texture: source,
                slice: 0,
                level,
                origin: [0; 3],
                size: [4, 4, 1],
                buffer: out,
                offset: 0,
                bytes_per_row: block,
                bytes_per_image: block,
                plane: Plane::Whole,
            });
            outs.push(out);
        }
        work.step(Step::Commit);
        // Each destination's levels 8 to 11 at their own extent, packed into one buffer.
        let mut backs = Vec::new();
        for texture in [into, from_buffer, replaced] {
            let back = work.buffer(vec![BEFORE; landed(block).len()]);
            let mut offset = 0;
            for level in WRITTEN {
                let row = blocks(level)[0] * block;
                let image = row * blocks(level)[1];
                work.step(Step::CopyTextureToBuffer {
                    texture,
                    slice: 0,
                    level,
                    origin: [0; 3],
                    size: extent(level),
                    buffer: back,
                    offset,
                    bytes_per_row: row,
                    bytes_per_image: image,
                    plane: Plane::Whole,
                });
                offset += image;
            }
            backs.push((back, offset));
        }
        work.step(Step::Commit);
        for (buffer, length) in backs {
            work.step(Step::ReadBuffer {
                buffer,
                offset: 0,
                length: length as usize,
            });
        }
        for out in outs {
            work.step(Step::ReadBuffer {
                buffer: out,
                offset: 0,
                length: OUT as usize,
            });
        }
        for level in SMALL {
            work.step(Step::GetBytes {
                texture: source,
                level,
                slice: 0,
                region: BLOCK,
                bytes_per_row: block,
                bytes_per_image: 0,
                length: OUT as usize,
                before: OUT_BEFORE,
            });
        }
        work
    }
}

impl Case for CompressedSmallLevels {
    fn name(&self) -> &'static str {
        self.format().name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        block_allowances()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        Ok(vec![Input::bytes("program", &fingerprint(&self.program()))])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let done = made(sites, "work", side.work(&self.program()))?;
        for (at, (site, length)) in reads(self.format().block).into_iter().enumerate() {
            let bytes = done.read(at).map_err(|why| Abort {
                at: site.clone(),
                failure: Failure::named("UnansweredRead", why),
            })?;
            let layout = ByteLayout::buffer(0, length, Element::U8);
            sites.leaf(&site, Observed::bytes(bytes.to_vec(), layout));
        }
        sites.value("statuses", Value::Lines(done.statuses.clone()));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 64 bytes the source's block of `level` leaves in a buffer of [`OUT_BEFORE`]: the
    /// block, and the rest as it was.
    fn read_out(level: u64, block: u64) -> Vec<u8> {
        let mut bytes = vec![OUT_BEFORE; OUT as usize];
        bytes[..block as usize].copy_from_slice(&level_bytes(level, block, None));
        bytes
    }

    /// The model's bytes are the probe's: its FNV-1a 64 of levels 8 to 11 after a copy at level 9,
    /// 10 or 11, and of the 64 bytes a copy into a buffer leaves (job m3-max-73696), BC1 then BC3.
    #[test]
    fn the_model_is_the_probe_s_bytes() {
        let fnv = |bytes: &[u8]| {
            bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, &byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
            })
        };
        // One level's copy at a time: the others as the fill left them.
        let one = |level: u64, block: u64| -> Vec<u8> {
            WRITTEN
                .flat_map(|l| level_bytes(l, block, (l != level).then_some(FILL)))
                .collect()
        };
        let probe = [
            (
                8,
                [
                    0xdf21_4fe1_8bfe_662d,
                    0xfda6_5c9a_502d_a7a5,
                    0x3145_d998_df90_8f15,
                ],
            ),
            (
                16,
                [
                    0xcd21_5888_fb21_8a95,
                    0x716f_1b3f_1cb3_9e25,
                    0x0a3e_663a_b6db_29e5,
                ],
            ),
        ];
        let probe_out = [
            (
                8,
                [
                    0xeb66_b13e_7d84_37ad,
                    0xe466_c8da_c77e_3825,
                    0xe2d8_ac95_6a0c_98d5,
                ],
            ),
            (
                16,
                [
                    0xb5b6_a640_cc0c_0ed5,
                    0xe802_82a8_6672_47e5,
                    0x7e40_11a7_6428_e2e5,
                ],
            ),
        ];
        for ((block, levels), (_, outs)) in probe.into_iter().zip(probe_out) {
            for (level, (hash, out)) in SMALL.zip(levels.into_iter().zip(outs)) {
                assert_eq!(fnv(&one(level, block)), hash, "{block} level {level}");
                assert_eq!(fnv(&read_out(level, block)), out, "{block} level {level}");
            }
        }
    }

    /// The program asks for one read a site, in [`reads`]' order.
    #[test]
    fn every_read_has_its_site() {
        for case in all() {
            let work = case.program();
            let asked = work
                .steps
                .iter()
                .filter(|step| matches!(step, Step::ReadBuffer { .. } | Step::GetBytes { .. }))
                .count();
            assert_eq!(asked, reads(case.format().block).len(), "{}", case.name());
        }
    }
}
