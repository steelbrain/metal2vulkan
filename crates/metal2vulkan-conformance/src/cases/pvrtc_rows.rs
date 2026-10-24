//! How a PVRTC texture's blocks move under `MTLBlitOptionRowLinearPVRTC`, ported from
//! `runtime_lane::pvrtc_textures`' two row-linear lanes.
//!
//! PVRTC is the one compressed family whose blocks do **not** lie row by row in a texture by
//! default: Metal places them in a block order of its own, and
//! `MTLBlitOptionRowLinearPVRTC` with a `bytesPerRow` is what makes a copy move them row by row.
//! That option is the whole subject here, and the lane's rows are these, per format and in both
//! `Private` and `Shared` storage:
//!
//! - `whole-16` and `whole-32`: a whole level in and out at its own row of blocks;
//! - `region`: a 32-texel level filled with `0x11`, then 2 x 2 blocks written at texel (8, 4), then
//!   the whole level and the region read back — so the blocks outside the region are recorded as
//!   well as those inside it;
//! - `offset` and `padded` (the 4-bit formats, as the lane): a copy from byte 8 of the buffer, and
//!   rows 48 bytes apart, where the 16 bytes past each row's blocks must come back untouched;
//! - `level-0` to `level-4`: every level of a 16-texel chain, the last levels one block each;
//! - `t2t`: one texture's blocks copied whole into another's.
//!
//! **What the port changes.** The lane's expectation is our own probe's measurement **on an Apple
//! M4**, asserted as Rust arithmetic over bytes the lane itself generates. Record mode runs the
//! same copies on this M3 Max, so each recording holds Metal's own bytes and the arithmetic becomes
//! a cross-check (the test `every_recorded_row_linear_copy_moved_the_blocks`). A difference then
//! says which side moved.
//!
//! **Every row here is valid Metal usage** — the option with a nonzero `bytesPerRow` — so Metal's
//! own API validation layer lets the whole program run, and the case needs no recorded-only row.
//! The lane's `bytesPerRow`-of-0 rows, which validation stops and whose order is Metal's own Morton
//! order, are not ported and are named as their own packet below; its host copies are
//! [`crate::cases::pvrtc_cpu_order`].
//!
//! **One case per format**, so a venue that holds no texture of one answers its typed refusal there
//! and counts as lacking that format rather than taking the other seven down. A device without the
//! Vulkan format keeps the blocks in a carrier image — half as wide for a 2-bit format, whose block
//! is 8 x 4 texels — which is the mechanism these rows exist to check, and it announces
//! `carried-decoded-twin` once.
//!
//! Nothing here can fault a GPU: blit copies between textures and buffers the program makes, no
//! dispatch, no draw, no loop, every region block-aligned and inside its level.

use crate::allowance::Allowance;
use crate::case::{made, Abort, Case, Failure, Input, MetalSource, Side};
use crate::transcript::{ByteLayout, Element, Node, Observed, Sites, Value};
use crate::work::{Plane, Rect, Resource, Step, TextureSpec, Work};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan_runtime::Degradation;

/// The bytes one PVRTC block takes, 2-bit and 4-bit alike.
pub const BLOCK: u64 = 8;

/// The byte a read-back's buffer holds before the copy, as the lane's `get` fills it: a position no
/// block reached is a recorded value and not whatever was there.
pub const BEFORE: u8 = 0xee;

/// One PVRTC format: the case's name, its raw `MTLPixelFormat` (stated as
/// `PixelFormat::<name>.as_raw()`, so the value is named in the case's own text and not only
/// in a comment beside it), and how wide its block is in
/// texels (4 for a 4-bit format, 8 for a 2-bit one; every block is 4 texels high).
pub struct Format {
    pub name: &'static str,
    pub format: u32,
    pub block_width: u64,
}

/// Every PVRTC format, the 4-bit ones then the 2-bit ones, as the lane lists them.
pub const FORMATS: [Format; 8] = [
    Format {
        name: "pvrtc-rows-rgb-4bpp",
        format: PixelFormat::PVRTCRGB4BPP.as_raw(),
        block_width: 4,
    },
    Format {
        name: "pvrtc-rows-rgb-4bpp-srgb",
        format: PixelFormat::PVRTCRGB4BPPSRGB.as_raw(),
        block_width: 4,
    },
    Format {
        name: "pvrtc-rows-rgba-4bpp",
        format: PixelFormat::PVRTCRGBA4BPP.as_raw(),
        block_width: 4,
    },
    Format {
        name: "pvrtc-rows-rgba-4bpp-srgb",
        format: PixelFormat::PVRTCRGBA4BPPSRGB.as_raw(),
        block_width: 4,
    },
    Format {
        name: "pvrtc-rows-rgb-2bpp",
        format: PixelFormat::PVRTCRGB2BPP.as_raw(),
        block_width: 8,
    },
    Format {
        name: "pvrtc-rows-rgb-2bpp-srgb",
        format: PixelFormat::PVRTCRGB2BPPSRGB.as_raw(),
        block_width: 8,
    },
    Format {
        name: "pvrtc-rows-rgba-2bpp",
        format: PixelFormat::PVRTCRGBA2BPP.as_raw(),
        block_width: 8,
    },
    Format {
        name: "pvrtc-rows-rgba-2bpp-srgb",
        format: PixelFormat::PVRTCRGBA2BPPSRGB.as_raw(),
        block_width: 8,
    },
];

/// `n` bytes of the lane's blocks: each 8-byte block's first byte its index, the others `tag` plus
/// its place in the block, so a block that moved to the wrong place names where it came from.
pub fn blocks(n: usize, tag: u8) -> Vec<u8> {
    (0..n)
        .map(|i| {
            if i % 8 == 0 {
                (i / 8) as u8
            } else {
                tag + (i % 8) as u8
            }
        })
        .collect()
}

/// A level's blocks in row-linear order: block (x, y) at `(y * across + x) * BLOCK`.
///
/// It is the model of what a row-linear copy does — the arithmetic the lane asserts — kept here so
/// the recordings can be held against it.
pub struct Linear {
    pub across: u64,
    pub down: u64,
    pub bytes: Vec<u8>,
}

impl Linear {
    /// A level of `across` by `down` blocks, every byte zero.
    pub fn new(across: u64, down: u64) -> Self {
        Self {
            across,
            down,
            bytes: vec![0; (across * down * BLOCK) as usize],
        }
    }

    /// `count` blocks at block `at` of the level, taken from `src` at `offset`, its rows `row`
    /// apart: what a row-linear copy into the texture leaves.
    pub fn put(&mut self, at: [u64; 2], count: [u64; 2], src: &[u8], offset: u64, row: u64) {
        for y in 0..count[1] {
            for x in 0..count[0] {
                let from = (offset + y * row + x * BLOCK) as usize;
                let to = (((at[1] + y) * self.across + at[0] + x) * BLOCK) as usize;
                self.bytes[to..to + BLOCK as usize]
                    .copy_from_slice(&src[from..from + BLOCK as usize]);
            }
        }
    }

    /// The same blocks written over `dst` at `offset`, rows `row` apart: what a row-linear copy out
    /// of the texture leaves, over whatever the buffer already held.
    pub fn get(&self, at: [u64; 2], count: [u64; 2], dst: &mut [u8], offset: u64, row: u64) {
        for y in 0..count[1] {
            for x in 0..count[0] {
                let to = (offset + y * row + x * BLOCK) as usize;
                let from = (((at[1] + y) * self.across + at[0] + x) * BLOCK) as usize;
                dst[to..to + BLOCK as usize]
                    .copy_from_slice(&self.bytes[from..from + BLOCK as usize]);
            }
        }
    }
}

/// One read-back the program asks for: its site's name and what the model says it holds.
pub struct Expected {
    pub at: String,
    pub bytes: Vec<u8>,
}

/// A case's program and the model's bytes for each of its reads, in order.
pub struct Plan {
    pub work: Work,
    pub expected: Vec<Expected>,
}

/// A square 2D PVRTC texture of `side` texels and `levels` levels, read by a shader.
fn spec(format: u32, side: u64, levels: u64, private: bool) -> TextureSpec {
    let spec = TextureSpec::texture_2d(format, side, side).levels(levels);
    if private {
        spec
    } else {
        spec.shared()
    }
}

/// The row-linear copy of `count` blocks at block `at` of `level`, between `buffer` from `offset`
/// at `row` bytes a row and `texture`.
struct Copy {
    texture: usize,
    buffer: usize,
    level: u64,
    /// The texels the region covers: origin then size.
    region: Rect,
    offset: u64,
    row: u64,
    /// The region's blocks across and down, which the row pitch applies to.
    count: [u64; 2],
}

impl Copy {
    /// The step that writes the buffer's blocks into the texture.
    fn write(&self) -> Step {
        Step::CopyBufferToTexture {
            buffer: self.buffer,
            offset: self.offset,
            bytes_per_row: self.row,
            bytes_per_image: self.row * self.count[1],
            size: self.region.size,
            texture: self.texture,
            slice: 0,
            level: self.level,
            origin: self.region.origin,
            plane: Plane::RowLinearPvrtc,
        }
    }

    /// The step that reads the texture's blocks into the buffer.
    fn read(&self, length: u64) -> Step {
        Step::CopyTextureToBuffer {
            texture: self.texture,
            slice: 0,
            level: self.level,
            origin: self.region.origin,
            size: self.region.size,
            buffer: self.buffer,
            offset: self.offset,
            bytes_per_row: self.row,
            // The lane passes the whole buffer's bytes as the image pitch of a copy out.
            bytes_per_image: length,
            plane: Plane::RowLinearPvrtc,
        }
    }
}

/// The plan for `format`: both storage modes' rows, in the lane's order.
fn plan(format: &Format) -> Plan {
    let width = format.block_width;
    let mut work = Work::new();
    let mut expected = Vec::new();
    for private in [true, false] {
        let mode = if private { "private" } else { "shared" };
        // Whole levels, 16 and 32 texels a side.
        for side in [16u64, 32] {
            let (across, down) = (side / width, side / 4);
            let row = across * BLOCK;
            let n = (row * down) as usize;
            let bytes = blocks(n, 0xb0);
            let texture = work.add(Resource::Texture {
                spec: spec(format.format, side, 1, private),
            });
            let source = work.buffer(bytes.clone());
            let out = work.buffer(vec![BEFORE; n]);
            let whole = Rect::new(0, 0, side, side);
            let put = Copy {
                texture,
                buffer: source,
                level: 0,
                region: whole,
                offset: 0,
                row,
                count: [across, down],
            };
            work.step(put.write());
            let get = Copy { buffer: out, ..put };
            work.step(get.read(n as u64));
            work.step(Step::Commit);
            work.step(Step::ReadBuffer {
                buffer: out,
                offset: 0,
                length: n,
            });
            expected.push(Expected {
                at: format!("{mode}/whole-{side}"),
                bytes,
            });
        }
        // A region of 2 x 2 blocks at texel (8, 4) of a 32-texel level over a fill of 0x11, then
        // the whole level and the region read back.
        {
            let side = 32;
            let (across, down) = (side / width, side / 4);
            let row = across * BLOCK;
            let n = (row * down) as usize;
            let texture = work.add(Resource::Texture {
                spec: spec(format.format, side, 1, private),
            });
            let fill = work.buffer(vec![0x11; n]);
            let region_bytes = blocks(4 * BLOCK as usize, 0xd0);
            let region_in = work.buffer(region_bytes.clone());
            let whole_out = work.buffer(vec![BEFORE; n]);
            let region_out = work.buffer(vec![BEFORE; region_bytes.len()]);
            let whole = Rect::new(0, 0, side, side);
            let region = Rect::new(8, 4, 2 * width, 8);
            let at = [8 / width, 1];
            let whole_copy = |buffer| Copy {
                texture,
                buffer,
                level: 0,
                region: whole,
                offset: 0,
                row,
                count: [across, down],
            };
            let region_copy = |buffer| Copy {
                texture,
                buffer,
                level: 0,
                region,
                offset: 0,
                row: 2 * BLOCK,
                count: [2, 2],
            };
            work.step(whole_copy(fill).write());
            work.step(region_copy(region_in).write());
            work.step(whole_copy(whole_out).read(n as u64));
            work.step(region_copy(region_out).read(region_bytes.len() as u64));
            work.step(Step::Commit);
            for (buffer, length) in [(whole_out, n), (region_out, region_bytes.len())] {
                work.step(Step::ReadBuffer {
                    buffer,
                    offset: 0,
                    length,
                });
            }
            let mut level = Linear::new(across, down);
            let filled = vec![0x11; n];
            level.put([0, 0], [across, down], &filled, 0, row);
            level.put(at, [2, 2], &region_bytes, 0, 2 * BLOCK);
            let mut whole_want = vec![BEFORE; n];
            level.get([0, 0], [across, down], &mut whole_want, 0, row);
            let mut region_want = vec![BEFORE; region_bytes.len()];
            level.get(at, [2, 2], &mut region_want, 0, 2 * BLOCK);
            expected.push(Expected {
                at: format!("{mode}/region-whole"),
                bytes: whole_want,
            });
            expected.push(Expected {
                at: format!("{mode}/region"),
                bytes: region_want,
            });
        }
        // The 4-bit formats' two pitch rows: a copy from byte 8 of the buffer, and rows 48 bytes
        // apart whose 16 bytes of padding must come back untouched.
        if width == 4 {
            let side = 16;
            let (across, down) = (side / width, side / 4);
            let row = across * BLOCK;
            let n = (row * down) as usize;
            let texture = work.add(Resource::Texture {
                spec: spec(format.format, side, 1, private),
            });
            let shifted_bytes = blocks(2 * n, 0xb0);
            let shifted_in = work.buffer(shifted_bytes.clone());
            let shifted_out = work.buffer(vec![BEFORE; n]);
            let padded_row = 48;
            let padded_len = (padded_row * down) as usize;
            let padded_bytes = blocks(padded_len, 0xb0);
            let padded_in = work.buffer(padded_bytes.clone());
            let padded_out = work.buffer(vec![BEFORE; padded_len]);
            let whole = Rect::new(0, 0, side, side);
            let copy = |buffer, offset, row| Copy {
                texture,
                buffer,
                level: 0,
                region: whole,
                offset,
                row,
                count: [across, down],
            };
            work.step(copy(shifted_in, 8, row).write());
            work.step(copy(shifted_out, 0, row).read(n as u64));
            work.step(copy(padded_in, 0, padded_row).write());
            work.step(copy(padded_out, 0, padded_row).read(padded_len as u64));
            work.step(Step::Commit);
            for (buffer, length) in [(shifted_out, n), (padded_out, padded_len)] {
                work.step(Step::ReadBuffer {
                    buffer,
                    offset: 0,
                    length,
                });
            }
            let mut level = Linear::new(across, down);
            level.put([0, 0], [across, down], &shifted_bytes, 8, row);
            let mut shifted_want = vec![BEFORE; n];
            level.get([0, 0], [across, down], &mut shifted_want, 0, row);
            level.put([0, 0], [across, down], &padded_bytes, 0, padded_row);
            let mut padded_want = vec![BEFORE; padded_len];
            level.get([0, 0], [across, down], &mut padded_want, 0, padded_row);
            expected.push(Expected {
                at: format!("{mode}/offset"),
                bytes: shifted_want,
            });
            expected.push(Expected {
                at: format!("{mode}/padded"),
                bytes: padded_want,
            });
        }
        // Every level of a 16-texel chain, the last levels one block each.
        {
            let texture = work.add(Resource::Texture {
                spec: spec(format.format, 16, 5, private),
            });
            let mut reads = Vec::new();
            for level in 0..5u64 {
                let side: u64 = (16u64 >> level).max(1);
                let (across, down) = (side.div_ceil(width), side.div_ceil(4));
                let row = across * BLOCK;
                let n = (row * down) as usize;
                let bytes = blocks(n, 0x10 * (level as u8 + 1));
                let source = work.buffer(bytes.clone());
                let out = work.buffer(vec![BEFORE; n]);
                let copy = |buffer| Copy {
                    texture,
                    buffer,
                    level,
                    region: Rect::new(0, 0, side, side),
                    offset: 0,
                    row,
                    count: [across, down],
                };
                work.step(copy(source).write());
                reads.push((level, out, n, bytes));
            }
            for (level, out, n, _) in &reads {
                let side: u64 = (16u64 >> level).max(1);
                let (across, down) = (side.div_ceil(width), side.div_ceil(4));
                work.step(
                    Copy {
                        texture,
                        buffer: *out,
                        level: *level,
                        region: Rect::new(0, 0, side, side),
                        offset: 0,
                        row: across * BLOCK,
                        count: [across, down],
                    }
                    .read(*n as u64),
                );
            }
            work.step(Step::Commit);
            for (level, out, n, bytes) in reads {
                work.step(Step::ReadBuffer {
                    buffer: out,
                    offset: 0,
                    length: n,
                });
                expected.push(Expected {
                    at: format!("{mode}/level-{level}"),
                    bytes,
                });
            }
        }
        // One texture's blocks copied whole into another's.
        {
            let side = 16;
            let (across, down) = (side / width, side / 4);
            let row = across * BLOCK;
            let n = (row * down) as usize;
            let bytes = blocks(n, 0xb0);
            let from = work.add(Resource::Texture {
                spec: spec(format.format, side, 1, private),
            });
            let to = work.add(Resource::Texture {
                spec: spec(format.format, side, 1, true),
            });
            let source = work.buffer(bytes.clone());
            let out = work.buffer(vec![BEFORE; n]);
            let whole = Rect::new(0, 0, side, side);
            work.step(
                Copy {
                    texture: from,
                    buffer: source,
                    level: 0,
                    region: whole,
                    offset: 0,
                    row,
                    count: [across, down],
                }
                .write(),
            );
            work.step(Step::CopyTextureToTexture {
                from,
                from_slice: 0,
                from_level: 0,
                from_origin: [0, 0, 0],
                size: [side, side, 1],
                to,
                to_slice: 0,
                to_level: 0,
                to_origin: [0, 0, 0],
            });
            work.step(
                Copy {
                    texture: to,
                    buffer: out,
                    level: 0,
                    region: whole,
                    offset: 0,
                    row,
                    count: [across, down],
                }
                .read(n as u64),
            );
            work.step(Step::Commit);
            work.step(Step::ReadBuffer {
                buffer: out,
                offset: 0,
                length: n,
            });
            expected.push(Expected {
                at: format!("{mode}/t2t"),
                bytes,
            });
        }
    }
    Plan { work, expected }
}

/// The case for one format.
pub struct PvrtcRows {
    at: usize,
}

/// Every format's case, in the lane's order.
pub fn all() -> Vec<PvrtcRows> {
    (0..FORMATS.len()).map(|at| PvrtcRows { at }).collect()
}

impl PvrtcRows {
    /// The format the case copies.
    fn format(&self) -> &'static Format {
        &FORMATS[self.at]
    }

    /// The case's plan.
    pub fn plan(&self) -> Plan {
        plan(self.format())
    }
}

impl Case for PvrtcRows {
    fn name(&self) -> &'static str {
        FORMATS[self.at].name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // A device with no Vulkan format for this PVRTC format keeps the blocks in a carrier image
        // and its reads through a decoded twin, which the runtime announces once per device. Metal
        // announces nothing. Every block still compares exactly: the only site loosened is the set
        // of notices, which Metal holds empty, and the allowance fails the case if the notice did
        // not fire.
        // The buffer each pass reads its blocks back through is `Shared`, which a device with no
        // unified memory keeps in system memory because that is the only memory both sides can
        // reach: the venue's memory model, not this family's subject.
        vec![
            crate::replay::notice_allowance(Degradation::CarriedDecodedTwin.code()),
            crate::replay::shared_memory_notice(),
        ]
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let format = self.format();
        Ok(vec![
            Input::bytes(
                "format",
                &[format.format, format.block_width as u32]
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect::<Vec<u8>>(),
            ),
            // The whole program as the interpreter will run it: every descriptor, every region,
            // every pitch, every blit option and every byte written. A case whose program moves is
            // `Unrecorded` rather than compared against a recording of other bytes. The cost is
            // that a later addition to the copy vocabulary makes these recordings stale, which
            // fails loudly and is one re-record — the same price an OS update already carries.
            Input::bytes(
                "program",
                &crate::cases::compressed_moves::fingerprint(&self.plan().work),
            ),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let plan = self.plan();
        let done = made(sites, "work", side.work(&plan.work))?;
        // The reads are grouped under their storage mode, so a diff names the mode before the row.
        let mut modes: Vec<Node> = Vec::new();
        for (at, expected) in plan.expected.iter().enumerate() {
            let bytes = done.read(at).map_err(|why| Abort {
                at: expected.at.clone(),
                failure: Failure::named("UnansweredRead", why),
            })?;
            let (mode, leaf) = expected.at.split_once('/').ok_or_else(|| Abort {
                at: expected.at.clone(),
                failure: Failure::named("NoSuchSite", "a site names its storage mode".to_owned()),
            })?;
            let layout = ByteLayout::buffer(0, bytes.len() as u64, Element::U8);
            let node = Node::leaf(leaf.to_owned(), Observed::bytes(bytes.to_vec(), layout));
            match modes.last_mut() {
                Some(last) if last.name == mode => last.children.push(node),
                _ => modes.push(Node::branch(mode.to_owned(), vec![node])),
            }
        }
        for mode in modes {
            sites.push(mode);
        }
        sites.value("statuses", Value::Lines(done.statuses.clone()));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The cases name distinct recordings, and every plan's sites are distinct.
    #[test]
    fn the_formats_and_their_sites_are_distinct() {
        let named: BTreeSet<&str> = FORMATS.iter().map(|format| format.name).collect();
        assert_eq!(named.len(), FORMATS.len());
        for case in all() {
            let plan = case.plan();
            let sites: BTreeSet<&str> = plan.expected.iter().map(|e| e.at.as_str()).collect();
            assert_eq!(sites.len(), plan.expected.len(), "{}", case.name());
        }
    }

    /// Every plan asks for exactly as many reads as it holds expectations.
    #[test]
    fn every_plans_reads_match_its_expectations() {
        for case in all() {
            let plan = case.plan();
            assert_eq!(plan.work.reads(), plan.expected.len(), "{}", case.name());
        }
    }

    /// A padded row's expectation keeps the fill past its blocks, and no other expectation holds
    /// the fill at all: the padded row is the one row whose point is the bytes a copy leaves alone,
    /// so a model that filled it in would be checking nothing.
    #[test]
    fn only_the_padded_row_expects_the_fill() {
        for case in all() {
            let plan = case.plan();
            for expected in &plan.expected {
                let has_fill = expected.bytes.contains(&BEFORE);
                let padded = expected.at.ends_with("/padded");
                assert_eq!(
                    has_fill,
                    padded,
                    "{} {}: the fill byte and the padded row disagree",
                    case.name(),
                    expected.at
                );
            }
        }
    }

    /// Every recorded read-back is the lane's row-linear arithmetic's, byte for byte.
    ///
    /// This is the cross-check the port exists for: the recordings hold what Metal answered on this
    /// M3 Max, where the lane's numbers came from our own probe on an Apple M4. A difference is a
    /// finding either way and never something to edit away.
    #[test]
    fn every_recorded_row_linear_copy_moved_the_blocks() {
        let mut unrecorded = Vec::new();
        let mut differ = Vec::new();
        let mut checked = 0usize;
        for case in all() {
            let recording = match crate::recording::read(case.name()) {
                Ok(recording) => recording,
                Err(_) => {
                    unrecorded.push(case.name());
                    continue;
                }
            };
            let plan = case.plan();
            for answer in &recording.answers {
                for outcome in &answer.outcomes {
                    for expected in &plan.expected {
                        let (mode, leaf) = expected.at.split_once('/').expect("a mode and a leaf");
                        let node = outcome
                            .transcript
                            .root
                            .child(mode)
                            .and_then(|node| node.child(leaf));
                        let Some(node) = node else {
                            differ.push(format!("{}: no site {}", case.name(), expected.at));
                            continue;
                        };
                        let crate::transcript::Value::Bytes(bytes) = &node.observed.value else {
                            differ.push(format!("{}/{}: no bytes", case.name(), expected.at));
                            continue;
                        };
                        checked += bytes.len();
                        if bytes != &expected.bytes && differ.len() < 24 {
                            let at = bytes.iter().zip(&expected.bytes).position(|(a, b)| a != b);
                            differ.push(format!(
                                "{}/{}: {} bytes against the model's {}, first difference at \
                                 {at:?}",
                                case.name(),
                                expected.at,
                                bytes.len(),
                                expected.bytes.len(),
                            ));
                        }
                    }
                }
            }
        }
        assert!(
            differ.is_empty(),
            "{checked} bytes checked; Metal's recorded row-linear copies differ from the lane's \
             arithmetic:\n{}",
            differ.join("\n")
        );
        assert!(
            unrecorded.is_empty(),
            "no recording yet for {unrecorded:?}: run `conformance record` on a Mac"
        );
    }
}
