//! The order a PVRTC texture's blocks take under Metal's **host** copies, ported from
//! `runtime_lane::pvrtc_textures`' CPU-order lane.
//!
//! A PVRTC texture does not hold its blocks row by row. A blit under
//! `MTLBlitOptionRowLinearPVRTC` moves them row by row ([`crate::cases::pvrtc_rows`]); Metal's host
//! calls do not. `replaceRegion:` of a **whole level** takes its blocks in Z order with the **y
//! bit lowest** ([`cpu_order`]), and `getBytes:` gives them back in that same order — so a block at
//! (x, y) of a level's rows lands at a position neither row-major nor column-major.
//!
//! The lane measured that on an Apple **M4** with our own probe, as a rule fitted in Rust over
//! bytes it generates. Record mode runs the same calls on this M3 Max, so each recording holds
//! Metal's own bytes and the rule becomes a cross-check (the test
//! `every_recorded_host_copy_is_the_lanes_z_order`). If Apple's placement moved between the chips
//! or the releases, the recording moves with it and the test says which level.
//!
//! Per format, every level of a 16-texel chain (down to one block a level) and a 32-texel level,
//! each at **two pitches**: a `bytesPerRow` of 0, which Metal's own API validation allows, and the
//! level's own row of blocks, which its validation **stops** while plain Metal takes it. That
//! second pitch is why this family is worth recording at all: plain Metal is the behavioural
//! oracle, the validated run is a second recorded fact, and the suite keeps the two apart
//! ([`crate::validated`]). Nothing here is a *descriptor* Metal's validation rejects, which is the
//! shape the round's safety rule forbids; a copy its validation objects to still returns, and the
//! lane's probe measured these rows on two machines.
//!
//! Per level and pitch the case records two reads:
//!
//! - `put`: the host writes blocks carrying their position, and a **row-linear blit** reads the
//!   level back — so the recording says where each host-written block landed in the level's rows;
//! - `get`: a row-linear blit fills the level row by row, and `getBytes:` reads it — so the
//!   recording says which level block each host-read byte came from.
//!
//! The two are inverses of one rule, which is what makes the pair worth recording rather than one
//! of them: a placement that is wrong in both directions by the same permutation would round-trip
//! and tell us nothing.
//!
//! **One case per format**, as the rest of the compressed families. A device with no Vulkan format
//! for this one keeps the blocks in a carrier image and announces `carried-decoded-twin` once.
//!
//! **Not ported, and why.** The lane's `replaceRegion:`/`getBytes:` of a region **other than** a
//! whole level, and of a whole level at a *padded* pitch, which our runtime refuses typed
//! (`NotImplemented(PvrtcBlockOrder)`) because plain Metal places those blocks by no rule the
//! runtime has found. Metal runs them, so each is a refusal of ours wider than Metal's — a product
//! gap rather than a case, and it belongs in the known-refusals ledger with a todo behind it, not
//! in a tolerance here. Said plainly rather than quietly skipped.
//!
//! Nothing here can fault a GPU: host copies and blit copies over textures the program makes, no
//! dispatch, no draw, no loop, every region a whole level.

use crate::allowance::Allowance;
use crate::case::{made, Abort, Case, Failure, Input, MetalSource, Side};
use crate::cases::pvrtc_rows::{blocks, Linear, BEFORE, BLOCK};
use crate::transcript::{ByteLayout, Element, Observed, Sites, Value};
use crate::work::{Plane, Rect, Resource, Step, TextureSpec, Work};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan_runtime::Degradation;

/// One PVRTC format the lane walks in this test: the case's name, the raw `MTLPixelFormat`
/// (stated as `PixelFormat::<name>.as_raw()`, so the value is named in the case's own text and
/// not only in a comment beside it), and how
/// wide its block is in texels.
struct Format {
    name: &'static str,
    format: u32,
    block_width: u64,
}

/// The formats the lane's CPU-order test walks: two 4-bit ones and all four 2-bit ones.
const FORMATS: [Format; 6] = [
    Format {
        name: "pvrtc-cpu-order-rgba-4bpp",
        format: PixelFormat::PVRTCRGBA4BPP.as_raw(),
        block_width: 4,
    },
    Format {
        name: "pvrtc-cpu-order-rgb-4bpp-srgb",
        format: PixelFormat::PVRTCRGB4BPPSRGB.as_raw(),
        block_width: 4,
    },
    Format {
        name: "pvrtc-cpu-order-rgb-2bpp",
        format: PixelFormat::PVRTCRGB2BPP.as_raw(),
        block_width: 8,
    },
    Format {
        name: "pvrtc-cpu-order-rgb-2bpp-srgb",
        format: PixelFormat::PVRTCRGB2BPPSRGB.as_raw(),
        block_width: 8,
    },
    Format {
        name: "pvrtc-cpu-order-rgba-2bpp",
        format: PixelFormat::PVRTCRGBA2BPP.as_raw(),
        block_width: 8,
    },
    Format {
        name: "pvrtc-cpu-order-rgba-2bpp-srgb",
        format: PixelFormat::PVRTCRGBA2BPPSRGB.as_raw(),
        block_width: 8,
    },
];

/// The position at which plain Metal lays block (`x`, `y`) of a region `blocks` wide and high when
/// a copy states no row: the region's own Morton order, the **x** bit lowest, each axis as many
/// bits as its block count needs, the larger axis's extra bits on top.
fn morton(x: u64, y: u64, blocks: [u64; 2]) -> u64 {
    let bits = |n: u64| (0..64).find(|&b| 1u64 << b >= n).unwrap_or(0);
    let (nx, ny) = (bits(blocks[0]), bits(blocks[1]));
    let mut order = Vec::new();
    for i in 0..nx.max(ny) {
        if i < nx {
            order.push((x >> i) & 1);
        }
        if i < ny {
            order.push((y >> i) & 1);
        }
    }
    order.iter().enumerate().map(|(b, &v)| v << b).sum()
}

/// The position Metal's host copies give block (`x`, `y`) of a whole level `blocks` across and
/// down: Z order with the **y** bit lowest, which is the Morton order above with the axes
/// swapped.
pub fn cpu_order(x: u64, y: u64, blocks: [u64; 2]) -> u64 {
    morton(y, x, [blocks[1], blocks[0]])
}

/// One read-back the program asks for: its site's name and what the rule says it holds.
///
/// The bytes are what the cross-check holds the recordings against, which is this module's whole
/// purpose, so they are part of its surface rather than a private detail.
pub struct Expected {
    pub at: String,
    pub bytes: Vec<u8>,
}

/// A case's program and the rule's bytes for each of its reads, in order.
pub struct Plan {
    pub work: Work,
    pub expected: Vec<Expected>,
}

/// The plan for `format`: every level of a 16-texel chain and a 32-texel level, each at a pitch of
/// 0 and at the level's own row.
fn plan(format: &Format) -> Plan {
    let width = format.block_width;
    let mut work = Work::new();
    let mut expected = Vec::new();
    for (side, levels) in [(16u64, 5u64), (32, 1)] {
        // Shared storage, because the host reads and writes the texture itself.
        let texture = work.add(Resource::Texture {
            spec: TextureSpec::texture_2d(format.format, side, side)
                .levels(levels)
                .shared(),
        });
        for level in 0..levels {
            let texels = (side >> level).max(1);
            let across = texels.div_ceil(width);
            let down = texels.div_ceil(4);
            let row = across * BLOCK;
            let n = (row * down) as usize;
            let whole = Rect::new(0, 0, texels, texels);
            // Each block's place twice: its byte in the level's own rows, and its byte in Metal's
            // host order.
            let placed: Vec<(usize, usize)> = (0..down)
                .flat_map(|y| (0..across).map(move |x| (x, y)))
                .map(|(x, y)| {
                    let rows = ((y * across + x) * BLOCK) as usize;
                    (rows, (cpu_order(x, y, [across, down]) * BLOCK) as usize)
                })
                .collect();
            for pitch in [0, row] {
                let at = format!("{side}/level-{level}/pitch-{pitch}");
                // put: the host writes, a row-linear blit reads the level back.
                let host = blocks(n, 0xb0);
                let read_rows = work.buffer(vec![BEFORE; n]);
                work.step(Step::ReplaceRegion {
                    texture,
                    level,
                    slice: 0,
                    region: whole,
                    bytes: host.clone(),
                    rows: down,
                    bytes_per_row: pitch,
                    bytes_per_image: 0,
                });
                work.step(Step::CopyTextureToBuffer {
                    texture,
                    slice: 0,
                    level,
                    origin: [0, 0, 0],
                    size: [texels, texels, 1],
                    buffer: read_rows,
                    offset: 0,
                    bytes_per_row: row,
                    bytes_per_image: n as u64,
                    plane: Plane::RowLinearPvrtc,
                });
                work.step(Step::Commit);
                work.step(Step::ReadBuffer {
                    buffer: read_rows,
                    offset: 0,
                    length: n,
                });
                let mut want_put = vec![0u8; n];
                for &(rows, order) in &placed {
                    want_put[rows..rows + BLOCK as usize]
                        .copy_from_slice(&host[order..order + BLOCK as usize]);
                }
                expected.push(Expected {
                    at: format!("{at}/put"),
                    bytes: want_put,
                });
                // get: a row-linear blit fills the level row by row, the host reads it.
                let filled = blocks(n, 0xc0);
                let fill = work.buffer(filled.clone());
                work.step(Step::CopyBufferToTexture {
                    buffer: fill,
                    offset: 0,
                    bytes_per_row: row,
                    bytes_per_image: n as u64,
                    size: [texels, texels, 1],
                    texture,
                    slice: 0,
                    level,
                    origin: [0, 0, 0],
                    plane: Plane::RowLinearPvrtc,
                });
                work.step(Step::Commit);
                work.step(Step::GetBytes {
                    texture,
                    level,
                    slice: 0,
                    region: whole,
                    bytes_per_row: pitch,
                    bytes_per_image: 0,
                    length: n,
                    // A whole level's host read fills every byte it is given, so the fill is only
                    // a value Metal would never leave: a byte that stayed it is a byte the call did
                    // not write.
                    before: BEFORE,
                });
                let mut want_get = vec![0u8; n];
                let mut level_rows = Linear::new(across, down);
                level_rows.put([0, 0], [across, down], &filled, 0, row);
                for &(rows, order) in &placed {
                    want_get[order..order + BLOCK as usize]
                        .copy_from_slice(&level_rows.bytes[rows..rows + BLOCK as usize]);
                }
                expected.push(Expected {
                    at: format!("{at}/get"),
                    bytes: want_get,
                });
            }
        }
    }
    Plan { work, expected }
}

/// The case for one format.
pub struct PvrtcCpuOrder {
    at: usize,
}

/// Every format's case, in the lane's order.
pub fn all() -> Vec<PvrtcCpuOrder> {
    (0..FORMATS.len()).map(|at| PvrtcCpuOrder { at }).collect()
}

impl PvrtcCpuOrder {
    /// The format the case copies.
    fn format(&self) -> &'static Format {
        &FORMATS[self.at]
    }

    /// The case's plan.
    pub fn plan(&self) -> Plan {
        plan(self.format())
    }
}

impl Case for PvrtcCpuOrder {
    fn name(&self) -> &'static str {
        FORMATS[self.at].name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // A device with no Vulkan format for this one keeps the blocks in a carrier image and its
        // reads through a decoded twin, announced once per device; Metal announces nothing. Every
        // block still compares exactly: the only site loosened is the set of notices, and the
        // allowance fails the case if the notice did not fire.
        //
        // Two more. A device with no Vulkan format for a PVRTC one decodes its blocks on the CPU
        // to answer a host read of them (`cpu-block-decode`), which is a route to the same texels
        // and not a texel of its own; the RTX 5080 announced it for all six rows. And the buffer
        // each row reads its blocks back through is `Shared`
        // ([`crate::replay::shared_memory_notice`]).
        vec![
            crate::replay::notice_allowance(Degradation::CarriedDecodedTwin.code()),
            crate::replay::notice_allowance(Degradation::CpuBlockDecode.code()),
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
            // every pitch and every byte written.
            Input::bytes(
                "program",
                &crate::cases::compressed_moves::fingerprint(&self.plan().work),
            ),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let plan = self.plan();
        let done = made(sites, "work", side.work(&plan.work))?;
        for (at, expected) in plan.expected.iter().enumerate() {
            let bytes = done.read(at).map_err(|why| Abort {
                at: expected.at.clone(),
                failure: Failure::named("UnansweredRead", why),
            })?;
            let layout = ByteLayout::buffer(0, bytes.len() as u64, Element::U8);
            sites.leaf(expected.at.clone(), Observed::bytes(bytes.to_vec(), layout));
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

    /// Every plan asks for exactly as many reads as it holds expectations, and no expectation is
    /// empty: a zero-length site would compare nothing.
    #[test]
    fn every_plans_reads_match_its_expectations() {
        for case in all() {
            let plan = case.plan();
            assert_eq!(plan.work.reads(), plan.expected.len(), "{}", case.name());
            for expected in &plan.expected {
                assert!(
                    !expected.bytes.is_empty(),
                    "{} {}: an empty expectation",
                    case.name(),
                    expected.at
                );
            }
        }
    }

    /// Every (level, pitch) the plans use places its blocks in a **permutation** of the level's
    /// own bytes, and that permutation is not the level's row order.
    ///
    /// Both halves matter, and the first is load-bearing rather than tidy: the plan indexes a
    /// buffer of `across * down` blocks by the order's position, so a shape whose order reached
    /// past that buffer would be a panic rather than a comparison. It holds here because every
    /// level these cases read is a whole number of blocks across and down in powers of two.
    ///
    /// It is **not** true in general, and the lane says so: a 3-block axis needs two bits, so its
    /// order leaves positions no block takes. No case here reads such a level, and a case that did
    /// would have to record those positions rather than index by them.
    #[test]
    fn every_levels_host_order_is_a_permutation_of_its_own_bytes() {
        let mut shapes = BTreeSet::new();
        for format in &FORMATS {
            let width = format.block_width;
            for (side, levels) in [(16u64, 5u64), (32, 1)] {
                for level in 0..levels {
                    let texels = (side >> level).max(1);
                    shapes.insert((texels.div_ceil(width), texels.div_ceil(4)));
                }
            }
        }
        assert!(shapes.len() >= 6, "only {} shapes to check", shapes.len());
        let mut mixing = 0usize;
        for &(across, down) in &shapes {
            let places: Vec<u64> = (0..down)
                .flat_map(|y| (0..across).map(move |x| cpu_order(x, y, [across, down])))
                .collect();
            let distinct: BTreeSet<u64> = places.iter().copied().collect();
            assert_eq!(
                distinct,
                (0..across * down).collect::<BTreeSet<u64>>(),
                "{across}x{down}: the host order is no permutation of the level's bytes"
            );
            if places != (0..across * down).collect::<Vec<u64>>() {
                mixing += 1;
            }
        }
        assert!(
            mixing >= 2,
            "only {mixing} of the {} shapes place a block anywhere but its row-major byte, so the \
             cases would round-trip whatever the rule said",
            shapes.len()
        );
    }

    /// The order is injective for every block count up to eight a side, holes or not: two blocks
    /// never share a position, which is the part of the rule that holds everywhere.
    #[test]
    fn the_host_order_never_puts_two_blocks_in_one_place() {
        for across in 1..=8u64 {
            for down in 1..=8u64 {
                let places: Vec<u64> = (0..down)
                    .flat_map(|y| (0..across).map(move |x| cpu_order(x, y, [across, down])))
                    .collect();
                let distinct: BTreeSet<u64> = places.iter().copied().collect();
                assert_eq!(
                    distinct.len(),
                    places.len(),
                    "{across}x{down}: the order repeats a position"
                );
            }
        }
    }

    /// Every recorded host copy is the lane's Z order's, byte for byte.
    ///
    /// This is the cross-check the port exists for: the recordings hold what Metal answered on this
    /// M3 Max, where the lane's rule was fitted on an Apple M4. A difference is a finding either
    /// way — Apple's placement moved, or the rule was never Metal's — and never something to edit
    /// away.
    #[test]
    fn every_recorded_host_copy_is_the_lanes_z_order() {
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
                        let node = outcome.transcript.root.child(&expected.at);
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
                                "{}/{}: {} bytes against the rule's {}, first difference at {at:?}",
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
            "{checked} bytes checked; Metal's recorded host copies differ from the lane's Z \
             order:\n{}",
            differ.join("\n")
        );
        assert!(
            unrecorded.is_empty(),
            "no recording yet for {unrecorded:?}: run `conformance record` on a Mac"
        );
    }
}
