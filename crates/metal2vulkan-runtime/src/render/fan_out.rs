//! A rate-mapped pass's triangles that a host step records, each drawn once and fanned out to the
//! map's regions (`rate_mapped_pass.rs`, **Fan-out**): whether the device fans them out, what the
//! next command's draw records, the probe's and the routes' runs of an execution's commands, the
//! parts a flagged draw is drawn in, and the region table and viewport a fanned-out draw records
//! through.
//!
//! **Parts.** A draw the probe flagged, a list of triangles of one instance whose pipeline reads
//! no `[[primitive_id]]` and writes no memory (`render/executions.rs`, `in_parts`), is drawn in
//! parts in its place ([`parts`]): its runs of blocks (`bind::fan_out::BLOCK_TRIANGLES`) that
//! hold a triangle the probe flagged, region by region, the first cut to the first flagged
//! triangle and the last to the last, runs fewer than `rate_mapped_pass::RUN_GAP` triangles apart
//! joined and at most [`MOST_RUNS`] of them, the nearest joined first; and the triangles before,
//! between and after them, fanned out. Each part is a draw of a range of the draw's vertices or
//! indices, so each triangle keeps its vertices, its `vertex_id`s and its place among the draw's:
//! a pixel takes the draw's fragments in the draw's order. A run is drawn only into the regions
//! its blocks' bounds reach ([`run_bounds`], [`run_regions`]): the probe raises a block's bounds
//! to the buckets of segments each of its triangles reaches on each axis, widened, a straddling
//! triangle's those of its part where `w` is positive (`bind::fan_out::PROBE_BOUNDS_AT`), so a
//! region it skips is one no triangle of the run covers a pixel of. A flagged draw of any other
//! kind is drawn region by region whole, into every region.
//!
//! **Region-copy variant.** On a device that fans the pass's triangles out and enabled what the
//! variant takes ([`copies_regions`]), a draw of the pass that is not fanned out (the encoder's
//! own, a command the probe did not probe or whose host step does not probe, a flagged draw's
//! runs and a flagged draw drawn whole) is drawn region by region through its region-copy variant
//! (`RegionFanOut::RegionCopies`), whose vertex function puts each vertex in its region as a
//! fan-out stage puts a copy's. The regions go in the table's order, each recording pushing its
//! first region ([`region_push`]) and drawing as many regions from it as a direct draw's
//! multi-draw takes (`DrawIndex` the region past the first, which the vertex function stores as
//! `ViewportIndex`), else one. Each recording sets as many viewports as a multi-draw takes, every
//! one the target's over the render area ([`target`]), and each draw's scissor its region's
//! ([`table`]), so the rasterizer rejects a triangle outside the region by its scissor
//! (`bind::fan_out`, **One route per pass**). Every route of the pass hands the rasterizer the
//! same position for a vertex in a region, so a draw that tests `Equal` the depth another stored
//! meets it whichever route each took.

use super::executions::Command;
use crate::binding::PushWrite;
use crate::error::{internal, Error};
use crate::rate_mapped_pass::RUN_GAP;
use crate::render_values::{ScissorRect, Viewport};
use crate::Device;
use ash::vk;
use metal2vulkan::bind::fan_out::{
    bucket, BLOCK_SET, BLOCK_TRIANGLES, BOUND_BUCKETS, PROBE_BLOCKS, PROBE_BOUNDS_AT,
    PROBE_COUNT_WORDS, STRADDLE_SET,
};
use metal2vulkan::bind::ReplayFaults;
use metal2vulkan::device::{RateMapRegionPolicy, VkFeature};
use metal2vulkan::pipeline::RegionFanOut;
use std::sync::Arc;

/// Whether `device` fans a rate-mapped pass's triangles out where a host step records them: its
/// policy asks it (`RateMapRegionPolicy::FanOut`), and it enabled the features the fan-out and its
/// probe take.
pub(super) fn fans_out(device: &Device) -> bool {
    let compile = device.compile_device();
    let features = &compile.profile().enabled().features;
    compile.policy().rate_map_regions == RateMapRegionPolicy::FanOut
        && [
            VkFeature::GeometryShader,
            VkFeature::ShaderClipDistance,
            VkFeature::VertexPipelineStoresAndAtomics,
        ]
        .iter()
        .all(|feature| features.contains(feature))
}

/// What the draw of an execution's next command records where its host step fans the pass's
/// triangles out (the module docs): its probe, or its draw fanned out; a probe at the index of its
/// words in the probe's flags (`bind::fan_out::PROBE_DRAW_WORDS`, `BindState::region_fan_out`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FanOutDraw {
    pub(super) mode: RegionFanOut,
    pub(super) index: u32,
    /// A draw's: the copies of its triangles the probe counted, which its work counts
    /// (`rate_mapped_pass::fan_out_work`).
    pub(super) copies: u32,
    /// A draw the probe flagged: its flagged triangles, whose blocks its parts draw region by
    /// region (the module docs, **Parts**).
    pub(super) flagged: Option<Flagged>,
}

/// The triangles the probe flagged of a draw (`bind::fan_out::PROBE_SETS`): the first and the
/// last, and the bits of its blocks of `bind::fan_out::BLOCK_TRIANGLES` that hold one, block `b`
/// bit `b mod 32` of word `b / 32`, the last block also every triangle past the bits' blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Flagged {
    pub(super) first: u32,
    pub(super) last: u32,
    pub(super) blocks: [u32; PROBE_COUNT_WORDS as usize],
}

/// What the probe found of a draw it probed, from its words (`bind::fan_out::PROBE_DRAW_WORDS`):
/// the copies its triangles emit (its counts summed), whether a triangle it does not flag is too
/// wide for few copies (`bind::fan_out::FEW_COPIES`, a wide word marked), the triangles it
/// flagged, where it flagged one, how many it flagged (its flagged counts summed) and how many
/// of those straddle `w = 0` (its straddle counts summed), each of each instance once; and,
/// where it flagged one, its blocks' bounds (`bind::fan_out::PROBE_BOUNDS_AT`), which its runs
/// take ([`run_bounds`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Probed {
    pub(super) copies: u32,
    pub(super) wide: bool,
    pub(super) flagged: Option<Flagged>,
    pub(super) flagged_triangles: u32,
    pub(super) straddling_triangles: u32,
    pub(super) bounds: Option<Arc<[u32]>>,
}

impl Probed {
    /// What the probe found of a draw of its `words`.
    pub(super) fn of(words: &[u32]) -> Probed {
        let lanes = PROBE_COUNT_WORDS as usize;
        let set = |k: usize| words.iter().skip(1 + k * lanes).take(lanes).copied();
        let first = set(2).max().unwrap_or(0);
        let last = set(3).max().unwrap_or(0);
        let mut blocks = [0u32; PROBE_COUNT_WORDS as usize];
        for (bits, word) in blocks.iter_mut().zip(set(BLOCK_SET as usize)) {
            *bits = word;
        }
        let flagged = words.first().is_some_and(|&flag| flag != 0);
        let bounds = (words.iter().skip(PROBE_BOUNDS_AT as usize))
            .take(PROBE_BLOCKS as usize)
            .copied();
        Probed {
            copies: set(0).fold(0u32, u32::saturating_add),
            wide: set(1).any(|w| w != 0),
            flagged: flagged.then(|| Flagged {
                first: u32::MAX - first,
                last: last.saturating_sub(1),
                blocks,
            }),
            flagged_triangles: set(4).fold(0u32, u32::saturating_add),
            straddling_triangles: set(STRADDLE_SET as usize).fold(0u32, u32::saturating_add),
            bounds: flagged.then(|| bounds.collect()),
        }
    }
}

/// A part of a flagged draw (the module docs, **Parts**): `count` of its triangles from triangle
/// `first`, fanned out or region by region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Part {
    pub(super) first: u32,
    pub(super) count: u32,
    pub(super) fanned: bool,
}

/// The most runs of a flagged draw's blocks drawn region by region ([`parts`]): each records the
/// plan's region draws, so a flagged draw records at most this many times the region draws one
/// run from its first flagged triangle to its last records, the nearest runs joined first.
pub(super) const MOST_RUNS: usize = 4;

/// The parts of a draw of `triangles` triangles the probe `flagged` (the module docs, **Parts**):
/// its runs of blocks that hold a flagged triangle region by region, the first cut to the first
/// flagged triangle and the last to the last, those fewer than `RUN_GAP` triangles apart joined,
/// the nearest joined first while more than [`MOST_RUNS`] stand; and the triangles before, between
/// and after them fanned out, an empty part left out. Where `even` (a list of 16-bit indices, whose
/// ranges start 4-byte aligned as the draw's do) each part begins at an even triangle, a run
/// widened to one. A flag past the triangles, or with no block's bit, draws every triangle region
/// by region.
pub(super) fn parts(flagged: &Flagged, triangles: u32, even: bool) -> Vec<Part> {
    let Flagged {
        first,
        last,
        blocks,
    } = *flagged;
    let whole = vec![Part {
        first: 0,
        count: triangles,
        fanned: false,
    }];
    if first > last || last >= triangles {
        return whole;
    }
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for block in 0..PROBE_BLOCKS {
        if blocks[(block / 32) as usize] >> (block % 32) & 1 == 0 {
            continue;
        }
        let end = match block + 1 < PROBE_BLOCKS {
            true => (block + 1) * BLOCK_TRIANGLES,
            false => triangles,
        };
        let (mut start, mut end) = ((block * BLOCK_TRIANGLES).max(first), end.min(last + 1));
        if even {
            start &= !1;
            end = end.next_multiple_of(2).min(triangles);
        }
        if start >= end {
            continue;
        }
        match runs.last_mut() {
            Some(run) if start.saturating_sub(run.1) < RUN_GAP => run.1 = run.1.max(end),
            _ => runs.push((start, end)),
        }
    }
    if runs.is_empty() {
        return whole;
    }
    while runs.len() > MOST_RUNS {
        let at = (1..runs.len())
            .min_by_key(|&at| runs[at].0 - runs[at - 1].1)
            .unwrap_or(1);
        let (_, end) = runs.remove(at);
        runs[at - 1].1 = end;
    }
    let mut parts = Vec::with_capacity(2 * runs.len() + 1);
    let mut at = 0;
    for (start, end) in runs.into_iter().chain([(triangles, triangles)]) {
        if start > at {
            parts.push(Part {
                first: at,
                count: start - at,
                fanned: true,
            });
        }
        if end > start {
            parts.push(Part {
                first: start,
                count: end - start,
                fanned: false,
            });
        }
        at = end;
    }
    parts
}

/// The bounds of a run of a flagged draw's blocks drawn region by region (the module docs,
/// **Parts**), `part`, of the draw's blocks' `bounds` (`Probed::bounds`): the buckets of segments
/// of each axis its blocks' triangles reach, x's in bits 0 to 15 and y's in bits 16 to 31
/// (`bind::fan_out::PROBE_BOUNDS_AT`).
pub(super) fn run_bounds(bounds: &[u32], part: &Part) -> u32 {
    let block = |triangle: u32| (triangle / BLOCK_TRIANGLES).min(PROBE_BLOCKS - 1) as usize;
    let last = part.first + part.count.max(1) - 1;
    (bounds.get(block(part.first)..=block(last)))
        .map_or(u32::MAX, |blocks| blocks.iter().fold(0, |all, &b| all | b))
}

/// The regions of a table of `[nx, ny]` segments, in its order, that a run of bounds `bounds`
/// reaches ([`run_bounds`]): each region `k` of segments `(k mod nx, k / nx)` whose x segment's
/// bucket and y segment's bucket are among its bits (`bind::fan_out::bucket`), as the ranges of
/// consecutive regions they make.
pub(super) fn run_regions([nx, ny]: [u32; 2], bounds: u32) -> Vec<std::ops::Range<u32>> {
    let held = |segment: u32, segments: u32, shift: u32| {
        bounds >> (shift + bucket(segment, segments).min(BOUND_BUCKETS - 1)) & 1 != 0
    };
    let mut ranges: Vec<std::ops::Range<u32>> = Vec::new();
    for sy in (0..ny).filter(|&sy| held(sy, ny, BOUND_BUCKETS)) {
        for sx in (0..nx).filter(|&sx| held(sx, nx, 0)) {
            let k = sy * nx + sx;
            match ranges.last_mut() {
                Some(range) if range.end == k => range.end += 1,
                _ => ranges.push(k..k + 1),
            }
        }
    }
    ranges
}

/// An encoder's run of an execution's commands where its host step fans the pass's triangles
/// out: probing their draws, or drawing them by the probe's words.
pub(super) enum FanOutRun {
    Probe(Probe),
    Route(Arc<FanOutRoutes>),
}

/// The probe's run of an execution's commands.
pub(super) struct Probe {
    /// The probe's flags, `bind::fan_out::PROBE_DRAW_WORDS` words a probed draw, bound whole.
    flags: vk::DescriptorBufferInfo,
    /// The index the next probed draw takes.
    next: u32,
    /// The index of each command's draw, by its place among the commands, where it was probed.
    slots: Vec<Option<u32>>,
    /// A command's barrier was met, or a draw that writes memory: the probe, recorded before the
    /// pass, cannot see what a later command's vertex function may read, so no later command is
    /// probed.
    stopped: bool,
}

impl Probe {
    /// A probe whose first draw takes index `next` of `flags`.
    pub(super) fn new(flags: vk::DescriptorBufferInfo, next: u32) -> Probe {
        Probe {
            flags,
            next,
            slots: Vec::new(),
            stopped: false,
        }
    }

    /// The index its next draw would take, and each command's.
    pub(super) fn finish(self) -> (u32, Vec<Option<u32>>) {
        (self.next, self.slots)
    }
}

impl FanOutRun {
    /// The fan-out of the draw of the command at `place` among the commands: the probe's, where
    /// it probes on; the route's draw where the probe probed it, of few copies where it marked
    /// none of the triangles it fans out too wide for them, and the triangles it flagged where it
    /// flagged one, which the draw's parts take (the module docs, **Parts**); `None` for a draw
    /// the probe did not probe, recorded region by region.
    pub(super) fn begin(&self, place: usize) -> Option<FanOutDraw> {
        match self {
            FanOutRun::Probe(probe) => (!probe.stopped).then_some(FanOutDraw {
                mode: RegionFanOut::Probe,
                index: probe.next,
                copies: 0,
                flagged: None,
            }),
            FanOutRun::Route(routes) => {
                let index = routes.slots.get(place).copied().flatten()?;
                let probed = routes.probed.get(index as usize)?;
                Some(FanOutDraw {
                    mode: match probed.wide {
                        false => RegionFanOut::DrawFew,
                        true => RegionFanOut::Draw,
                    },
                    index,
                    copies: probed.copies,
                    flagged: probed.flagged,
                })
            }
        }
    }

    /// The blocks' bounds of the flagged draw of index `index` the probe found
    /// (`Probed::bounds`), where the route draws by the probe's words.
    pub(super) fn bounds(&self, index: u32) -> Option<Arc<[u32]>> {
        match self {
            FanOutRun::Probe(_) => None,
            FanOutRun::Route(routes) => {
                (routes.probed.get(index as usize)).and_then(|probed| probed.bounds.clone())
            }
        }
    }

    /// After the draw of the command at `place`: whether it ran its fan-out variant (`fanned`),
    /// and whether its pipeline writes memory (`writes`).
    pub(super) fn end(&mut self, place: usize, fanned: bool, writes: bool) {
        if let FanOutRun::Probe(probe) = self {
            if fanned && !probe.stopped {
                if probe.slots.len() <= place {
                    probe.slots.resize(place + 1, None);
                }
                probe.slots[place] = Some(probe.next);
                probe.next += 1;
            }
            probe.stopped |= writes;
        }
    }

    /// A command's barrier stands: under the probe, no command from it on is probed.
    pub(super) fn barrier(&mut self) {
        if let FanOutRun::Probe(probe) = self {
            probe.stopped = true;
        }
    }

    /// Whether it probes.
    pub(super) fn probes(&self) -> bool {
        matches!(self, FanOutRun::Probe(_))
    }

    /// The probe's flags, where it probes.
    pub(super) fn flags(&self) -> Option<vk::DescriptorBufferInfo> {
        match self {
            FanOutRun::Probe(probe) => Some(probe.flags),
            FanOutRun::Route(_) => None,
        }
    }
}

/// The routes of an execution's commands at its host step (the module docs): the commands as the
/// step read them, once, for the probe and the routes alike, from command `start`; the faults the
/// read found; the index of each command's draw among the probe's, by its place among the
/// commands that run, where it was probed; and what the probe found of each draw it probed, by
/// its index.
pub(super) struct FanOutRoutes {
    pub(super) commands: Vec<Option<Command>>,
    pub(super) start: u32,
    pub(super) faults: ReplayFaults,
    pub(super) slots: Vec<Option<u32>>,
    pub(super) probed: Arc<[Probed]>,
}

/// An encoder's region table ([`table`]): its bytes, until its first draw that binds it uploads
/// them, then the region of the command buffer's uploads that holds them
/// (`RenderLowering::fan_table_bound`).
pub(super) enum FanTable {
    Bytes(Vec<u8>),
    Bound(vk::DescriptorBufferInfo),
}

/// What a region table was made for: the encoder's first viewport and scissor, and the render
/// area (`RenderLowering::fan_table`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FanTableKey {
    viewport: [u64; 4],
    scissor: [u64; 4],
    extent: [u32; 2],
}

impl FanTableKey {
    /// The key of `viewport` and `scissor` over a render area of `extent`.
    pub(super) fn new(viewport: &Viewport, scissor: &ScissorRect, extent: [u32; 2]) -> FanTableKey {
        let v = viewport;
        FanTableKey {
            viewport: [v.origin_x, v.origin_y, v.width, v.height].map(f64::to_bits),
            scissor: [scissor.x, scissor.y, scissor.width, scissor.height],
            extent,
        }
    }
}

/// The target's viewport `[x, y, width, height]` a fan-out draw records through: the whole
/// render area `extent`, y flipped as each region's is (`rate_mapped_pass::on_device`).
fn target_rect([width, height]: [u32; 2]) -> [f32; 4] {
    let (width, height) = (width as f32, height as f32);
    [0.0, height, width, -height]
}

/// The region table of a fan-out draw over `regions`, each a region's viewport and scissor as the
/// device draws it (`RateMappedPass::draws`), through one viewport over the render area `extent`
/// (`bind::fan_out::table`), and its regions' scissors in its order (`bind::fan_out::scissors`),
/// which a draw through the region-copy variant draws through (the module docs, **Region-copy
/// variant**): `None` where they are not a grid of segments or hold no pixel.
pub(super) fn table(
    regions: &[(vk::Viewport, vk::Rect2D)],
    extent: [u32; 2],
) -> Option<FanTableMade> {
    if !(regions.iter()).any(|(_, s)| s.extent.width > 0 && s.extent.height > 0) {
        return None;
    }
    let regions: Vec<([f32; 4], [u32; 4])> = (regions.iter())
        .map(|(v, s)| {
            let scissor = [
                u32::try_from(s.offset.x).ok()?,
                u32::try_from(s.offset.y).ok()?,
                s.extent.width,
                s.extent.height,
            ];
            Some(([v.x, v.y, v.width, v.height], scissor))
        })
        .collect::<Option<_>>()?;
    let table = metal2vulkan::bind::fan_out::table(target_rect(extent), &regions)?;
    let scissors = (metal2vulkan::bind::fan_out::scissors(&regions)?.into_iter())
        .map(|[x, y, width, height]| {
            Some(vk::Rect2D {
                offset: vk::Offset2D {
                    x: i32::try_from(x).ok()?,
                    y: i32::try_from(y).ok()?,
                },
                extent: vk::Extent2D { width, height },
            })
        })
        .collect::<Option<_>>()?;
    // The table's segments of each axis, its first entry's first two words.
    let word = |at: usize| Some(u32::from_le_bytes(table.get(at..at + 4)?.try_into().ok()?));
    let grid = [word(0)?, word(4)?];
    Some(FanTableMade {
        table,
        scissors,
        grid,
    })
}

/// A fan-out draw's region table ([`table`]): its bytes, its regions' scissors in its order, and
/// its segments of each axis, which pick the regions a run's bounds reach ([`run_regions`]).
pub(super) struct FanTableMade {
    pub(super) table: Vec<u8>,
    pub(super) scissors: Vec<vk::Rect2D>,
    pub(super) grid: [u32; 2],
}

/// The viewports a recording of a draw through its region-copy variant sets, every one the
/// target's: as many as one multi-draw command takes, one without the multi-draw commands (the
/// module docs, **Region-copy variant**). Its `VkPipeline` takes as many
/// (`FixedState::viewports`).
pub(super) fn copy_viewports(device: &Device) -> u32 {
    (device.shared().multi_draws)
        .as_ref()
        .map_or(1, |multi| multi.batch().max(1))
}

/// Whether `device` draws a fanning pass's region-by-region draws through their region-copy
/// variant (the module docs, **Region-copy variant**): it fans out ([`fans_out`]), and it enabled
/// what the variant's vertex function takes to store its draw's index as `ViewportIndex` and its
/// several viewports (`LoweringInputs::region_copies`).
pub(super) fn copies_regions(device: &Device) -> bool {
    let compile = device.compile_device();
    let features = &compile.profile().enabled().features;
    fans_out(device)
        && [
            VkFeature::MultiViewport,
            VkFeature::ShaderOutputViewportIndex,
            VkFeature::ShaderDrawParameters,
        ]
        .iter()
        .all(|feature| features.contains(feature))
}

/// The push writes of a recording of a draw through its region-copy variant from region `first`
/// (the module docs, **Region-copy variant**): the draw's, `push`, with the vertex stage's word at
/// `offset` (`PushField::RegionFirst`) holding `first`.
pub(super) fn region_push(
    push: &[PushWrite],
    offset: u32,
    first: u32,
) -> Result<Vec<PushWrite>, Error> {
    let mut pushed = push.to_vec();
    let write = (pushed.iter_mut())
        .find(|(stages, at, bytes)| {
            stages.contains(vk::ShaderStageFlags::VERTEX)
                && *at <= offset
                && offset + 4 <= *at + bytes.len() as u32
        })
        .ok_or_else(|| {
            internal("a draw through its region-copy variant without its region's push")
        })?;
    let from = (offset - write.1) as usize;
    write.2[from..from + 4].copy_from_slice(&first.to_le_bytes());
    Ok(pushed)
}

/// The viewport and scissor a fan-out draw records through: one over the whole render area
/// `extent`, of the depth range `[min_depth, max_depth]`.
pub(super) fn target(
    extent: [u32; 2],
    [min_depth, max_depth]: [f32; 2],
) -> (vk::Viewport, vk::Rect2D) {
    let [x, y, width, height] = target_rect(extent);
    (
        vk::Viewport {
            x,
            y,
            width,
            height,
            min_depth,
            max_depth,
        },
        vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D {
                width: extent[0],
                height: extent[1],
            },
        },
    )
}

#[cfg(test)]
mod tests;
