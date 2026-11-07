//! Compute dispatches and their instantiation: [`BindingPlan::instantiate_dispatch`].
//!
//! **Regions**. A Metal dispatch becomes one or more regular Vulkan dispatches, each of an exact
//! local size, so Metal's `threads_per_threadgroup` and barrier semantics hold:
//!
//! - `Threadgroups` within `maxComputeWorkGroupCount`, and every `ThreadgroupsIndirect`, is one
//!   region with `MultiRegion = 0`: the grid comes from `NumWorkgroups`, so GPU-written counts
//!   need no CPU read (an indirect region's group counts are never an input of the plan).
//! - `Threads` with a grid that is not a multiple of the threadgroup size becomes up to eight
//!   regions per group-count chunk: the interior, and per axis an edge of one threadgroup of the
//!   remaining threads, all with `MultiRegion = 1`, so the grid and the region base come from the
//!   `DispatchRegion` push field. On the plan of a flat variant ([`VariantKey::flat_grid`]) it is
//!   one region instead, of the dispatch's own threadgroup size: its invocations count the grid's
//!   threads in order and each takes the one it counts to, those past the last returning before
//!   the body, so no edge and no group-count chunk is planned ([`regions`]).
//! - A count above `maxComputeWorkGroupCount` on an axis is split into chunks, each a region
//!   with `MultiRegion = 1` and its base in the push field. At most [`MAX_REGIONS`] regions are
//!   planned; more is `Unsupported{GroupCount}`.
//! - `ThreadgroupRange` (a tile dispatch over a region) is the range's chunks, each with
//!   `MultiRegion = 1`, its base the range's first threadgroup, and the grid the whole dispatch's.
//! - A tile function whose threadgroup is the tile (`LocalRules::tile_threadgroup`) runs its
//!   threads per tile in each region, but every region has `MultiRegion = 1` and carries the tile
//!   as Metal's threadgroup and the tiles times the tile as the grid, as plain Metal answers them
//!   under `threadgroupSizeMatchesTileSize`.
//!
//! Every region's local size is checked: in the flattened tiers `sn = x·y·z` against
//! `maxComputeWorkGroupSize[0]`, `maxComputeWorkGroupInvocations`, and
//! `maxComputeWorkgroupSubgroups · S`; otherwise each axis and the product; and against every
//! `LocalSizeConstraint{multiple}` (`ComputeDerivatives` outside the flattened tiers). Metal's
//! nominal size must be 1..=1024 per axis (the `DispatchRegion` packing).
//!
//! **Partial simdgroups**. A flattened tier runs whole subgroups of its size `S` only, so a
//! dispatch with a region whose `sn` is not a multiple of `S` (a partial simdgroup, or in
//! `Cluster32` simdgroups that do not fill a subgroup) runs on the kernel's emulated-simdgroups
//! variant (`VariantKey::emulated_simdgroups`), whose tier is exact for every threadgroup size and
//! whose local size is not flattened. The whole dispatch runs on the variant, so every region of
//! it, its whole-subgroup regions included, is checked against the variant's limits: each axis and
//! the product. The dispatch's variant ([`BindingPlan::dispatch_variant`]) is then the plan's with
//! emulated simdgroups, and instantiating it on the plan of another variant is
//! `BindError::NeedsVariant`, before anything is written. Only the variant's scratch is left to the
//! variant's own plan, since only its compile knows it. A dispatch of whole subgroups keeps the
//! flattened pipeline.
//!
//! **Push bytes.** Each region's push write covers the stage's whole push range, so every
//! statically used push byte is written (VUID-vkCmdDispatch-maintenance4-08602): the
//! `DispatchRegion` bytes (`lower::dispatch::region_bytes`) in every region, `MultiRegion` or
//! not, the stage-in region, the residual bind offsets, the slot guest bases, and zero at the
//! address holes the consumer fills ([`PushHole`]).
//!
//! **Stage-in region**. A kernel that reads the region (a `[[stage_in]]` step by thread position,
//! `[[grid_origin]]`, or `[[grid_size]]`) is dispatched only with one the encoder set: without it,
//! `BindError::MissingStageInRegion`, since what Metal reads then is not established. A step at a
//! rate above 1 adds the region's origin to the quotient, and an indexed step adds it to the index
//! it looks up at the thread position, as an M4 and an M3 Max read (`lower::stage_io::stage_in`):
//! the binder writes the region whatever its origin. Each region's `SpecKey::StageInReach` says
//! whether an element of a thread-position step may lie past the first `2^32` bytes from its bind
//! offset ([`stage_in_reach`]). It is false where none can, which drops the fetch's reach check,
//! so only a dispatch that can reach takes the `VkPipeline` that keeps it.
//!
//! **Indirect.** When any axis of `maxComputeWorkGroupCount` is below `2^32 - 1`, an indirect
//! dispatch carries an [`IndirectClamp`] prepass and its region reads the clamp's scratch
//! ([`IndirectArgs::ClampScratch`]).

use crate::bind::indirect_clamp::IndirectClamp;
use crate::bind::instantiate::{
    bind_buffers, put, slot_field, texel_ranges, BufferExtent, DescriptorBytes, DescriptorRange,
    FieldBytes, IndirectRef, PushConstantWrite, PushHole, RootEntry, RootTable,
};
use crate::bind::plan::compute::{internal, not_implemented, spec_key};
use crate::bind::plan::{
    AccessClass, BindingPlan, ByteRange, ComputeTemplate, LocalRules, ResourceUse, UseTarget,
};
use crate::bind::replay::ReplayRefusal;
use crate::bind::resident::{stage_root_table, table_entries};
use crate::bind::state::{BindState, BoundBuffer, StageInRegion};
use crate::cache::{SpecializationKey, VariantKey};
use crate::device::requirements::VkLimit;
use crate::error::{BindError, Capability, Feature, LowerFamily, Unsupported};
use crate::indirect_commands::BINDING_COUNT;
use crate::lower::dispatch::{pack_nominal, region_bytes};
use crate::pipeline::interface::{AccessSet, BufferForm, PushLoc};
use crate::pipeline::sym::{Axis, PushField, SlotClass, SpecKey, Stage};
use crate::vk;

/// The most regions one dispatch is split into. Every committed profile's
/// `maxComputeWorkGroupCount` is at least 65535 per axis, so a grid within `2^16 · 65535` groups
/// per axis stays well inside it.
pub const MAX_REGIONS: u64 = 4096;

/// A Metal compute dispatch.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Dispatch {
    /// `dispatchThreadgroups:threadsPerThreadgroup:`.
    Threadgroups {
        groups: [u32; 3],
        threads_per_group: [u32; 3],
    },
    /// `dispatchThreads:threadsPerThreadgroup:` (non-uniform edges become extra regions).
    Threads {
        threads: [u32; 3],
        threads_per_group: [u32; 3],
    },
    /// `dispatchThreadgroupsWithIndirectBuffer:` (GPU-written group counts).
    ThreadgroupsIndirect {
        indirect: IndirectRef,
        threads_per_group: [u32; 3],
    },
    /// The threadgroups `first .. first + count` of `dispatchThreadgroups:` of `groups`, each
    /// axis's range cut to the grid: each runs as in the whole dispatch (its
    /// `threadgroup_position_in_grid`, and the grid of `groups * threads_per_group` threads),
    /// and no other runs. A tile dispatch over a region (`dispatchThreadsPerTile:inRegion:`)
    /// runs the tiles the region touches so. Its regions take their first threadgroup and the
    /// grid from the `DispatchRegion` push field (`MultiRegion` 1).
    ThreadgroupRange {
        groups: [u32; 3],
        first: [u32; 3],
        count: [u32; 3],
        threads_per_group: [u32; 3],
    },
}

impl Dispatch {
    /// Metal's `threads_per_threadgroup` of the dispatch.
    pub fn threads_per_group(&self) -> [u32; 3] {
        match *self {
            Dispatch::Threadgroups {
                threads_per_group, ..
            }
            | Dispatch::Threads {
                threads_per_group, ..
            }
            | Dispatch::ThreadgroupsIndirect {
                threads_per_group, ..
            }
            | Dispatch::ThreadgroupRange {
                threads_per_group, ..
            } => threads_per_group,
        }
    }
}

/// What the consumer records for one dispatch: upload `descriptor_bytes` and `root_table`, bind
/// every plan descriptor (buffer descriptors over exactly `descriptor_ranges`), fill the
/// `push_holes` in every region's push bytes, record the `indirect_clamp` prepass when present,
/// and then each region in order.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DispatchCommands {
    /// One regular Vulkan dispatch per region: the uniform interior, `dispatchThreads` edges, and
    /// group-count splits. Empty when the grid is empty.
    pub regions: Vec<DispatchRegion>,
    /// The stage's `ArgumentTable` contents and its handle table.
    pub descriptor_bytes: Vec<DescriptorBytes>,
    /// The range of every buffer descriptor the plan writes as `Buffer` or `InlineBytes`.
    pub descriptor_ranges: Vec<DescriptorRange>,
    /// Per bound buffer slot, the window this dispatch can touch.
    pub extents: Vec<BufferExtent>,
    pub push_holes: Vec<PushHole>,
    /// The 32-byte region record a direct dispatch on a flat variant
    /// ([`VariantKey::flat_grid`]) reads: the consumer writes exactly these bytes where the
    /// device reads them by address, 16-byte aligned, and fills the single
    /// [`PushHole::ReplayRegionAddress`] with that address. `None` for every other dispatch: an
    /// indirect one's record its preprocess writes, and a plan that is not a flat variant's reads
    /// its region from its own push bytes.
    pub region_record: Option<[u8; 32]>,
    /// The resident root table, when the function reads it.
    pub root_table: Option<RootTable>,
    /// The indirect-clamp prepass to record before the regions, when group counts are GPU-written
    /// and the device's group-count limit is below `2^32 - 1`.
    pub indirect_clamp: Option<IndirectClamp>,
    /// The dispatch's own resources beyond the plan's `uses`: the 12 bytes of its indirect
    /// arguments, read by the clamp prepass as a storage buffer in `COMPUTE_SHADER`, or by
    /// `vkCmdDispatchIndirect` in `DRAW_INDIRECT` without one. The range is relative to the
    /// arguments' offset.
    pub uses: Vec<ResourceUse>,
}

/// Where an indirect region reads its group counts.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum IndirectArgs {
    /// Metal's arguments, directly (`vkCmdDispatchIndirect` from this buffer and offset).
    Buffer(IndirectRef),
    /// The scratch the `DispatchCommands.indirect_clamp` prepass wrote, at its start.
    ClampScratch,
    /// The device replay's arguments for the replayed command and this variant, which its
    /// preprocess wrote (`ReplayLayout::argument_offset` in the replay's scratch).
    Replay,
}

/// One Vulkan dispatch of a Metal dispatch.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DispatchRegion {
    /// The complete specialization of this region's pipeline: the plan's constants, the local
    /// size (and `sn` in the flattened tiers), and `MultiRegion`.
    pub specialization: SpecializationKey,
    /// Ignored for an indirect region.
    pub group_count: [u32; 3],
    pub push_constants: Vec<PushConstantWrite>,
    /// `vkCmdDispatchIndirect` from here instead of a direct dispatch.
    pub indirect: Option<IndirectArgs>,
    /// `REQUIRE_FULL_SUBGROUPS` when the tier needs it.
    pub stage_flags: vk::PipelineShaderStageCreateFlags,
    /// `requiredSubgroupSize` when the tier pins one.
    pub required_subgroup_size: Option<u32>,
}

const AXES: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];

fn limit(feature: Feature, limit: VkLimit, required: u64, available: u64) -> BindError {
    BindError::Unsupported(Unsupported::needs(
        feature,
        Capability::Limit {
            limit,
            required,
            available,
        },
    ))
}

/// Which pipeline runs a dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fit {
    /// The planned pipeline.
    Planned,
    /// The kernel's emulated-simdgroups variant: a region's threadgroup is not whole subgroups of
    /// the flattened tier's size.
    EmulatedSimdgroups,
}

/// A flattened tier's region of local size `r` that is not whole subgroups of its size: every
/// multiple of a flattened plan is its tier's subgroup size, since the flattened tiers refuse
/// derivatives (`ComputeDerivativesWithSimd`).
fn partial(rules: &LocalRules, r: [u32; 3]) -> bool {
    let n: u64 = r.iter().map(|&x| u64::from(x)).product();
    rules.flat
        && rules
            .multiples
            .iter()
            .any(|&m| m > 0 && !n.is_multiple_of(u64::from(m)))
}

/// The local-size rules of the kernel's emulated-simdgroups variant as the plan of its flattened
/// variant knows them: the same device limits, a local size that is not flattened, and no whole
/// subgroups (the flattened plan's multiples are all its tier's, [`partial`]). The variant's
/// scratch only its own plan knows.
fn emulated_rules(rules: &LocalRules) -> LocalRules {
    LocalRules {
        flat: false,
        multiples: Vec::new(),
        required_subgroup_size: None,
        stage_flags: vk::PipelineShaderStageCreateFlags::empty(),
        scratch: None,
        ..rules.clone()
    }
}

/// The pipeline that runs a dispatch whose regions have the local sizes `locals` (every axis at
/// least 1), with every region checked against that pipeline's rules. A flattened tier's dispatch
/// with a region that is not whole subgroups runs whole on the emulated-simdgroups variant, so each
/// of its regions is checked against the variant's rules ([`emulated_rules`]).
pub(crate) fn fit(rules: &LocalRules, locals: &[[u32; 3]]) -> Result<Fit, BindError> {
    if locals.iter().any(|&r| partial(rules, r)) {
        let emulated = emulated_rules(rules);
        for &r in locals {
            check_local(&emulated, r)?;
        }
        Ok(Fit::EmulatedSimdgroups)
    } else {
        for &r in locals {
            check_local(rules, r)?;
        }
        Ok(Fit::Planned)
    }
}

/// The checks of one region's local size `r` (every axis at least 1) under the rules of the
/// pipeline that runs it: a flattened tier's region is whole subgroups here ([`fit`]).
fn check_local(rules: &LocalRules, r: [u32; 3]) -> Result<(), BindError> {
    let n: u64 = r.iter().map(|&x| u64::from(x)).product();
    let total = Feature::LocalSize { axis: None };
    if rules.flat {
        let x = Feature::LocalSize {
            axis: Some(Axis::X),
        };
        if n > u64::from(rules.max_size[0]) {
            let max = u64::from(rules.max_size[0]);
            return Err(limit(x, VkLimit::MaxComputeWorkGroupSize(0), n, max));
        }
        if let Some(s) = rules.required_subgroup_size {
            let groups = n.div_ceil(u64::from(s.max(1)));
            let max = u64::from(rules.max_subgroups);
            if groups > max {
                return Err(limit(
                    total.clone(),
                    VkLimit::MaxComputeWorkgroupSubgroups,
                    groups,
                    max,
                ));
            }
        }
    } else {
        for (i, (&x, &max)) in r.iter().zip(&rules.max_size).enumerate() {
            if x > max {
                let axis = Feature::LocalSize {
                    axis: Some(AXES[i]),
                };
                let l = VkLimit::MaxComputeWorkGroupSize(i as u8);
                return Err(limit(axis, l, u64::from(x), u64::from(max)));
            }
        }
    }
    if n > u64::from(rules.max_invocations) {
        let max = u64::from(rules.max_invocations);
        return Err(limit(
            total,
            VkLimit::MaxComputeWorkGroupInvocations,
            n,
            max,
        ));
    }
    // Outside the flattened tiers a multiple is a derivative group's.
    if !rules.flat {
        for &m in &rules.multiples {
            if m > 0 && !n.is_multiple_of(u64::from(m)) {
                return Err(BindError::Unsupported(Unsupported::not_representable(
                    Feature::ComputeDerivatives,
                )));
            }
        }
    }
    // Emulated simdgroups: the region's scratch beside the fixed threadgroup memory.
    if let Some(s) = rules.scratch {
        let required = n
            .div_ceil(32)
            .saturating_mul(u64::from(s.per_simdgroup))
            .saturating_add(s.fixed);
        let available = u64::from(s.available);
        if required > available {
            return Err(limit(
                Feature::ThreadgroupMemorySize,
                VkLimit::MaxComputeSharedMemorySize,
                required,
                available,
            ));
        }
    }
    Ok(())
}

/// One Vulkan dispatch before its bytes: exact local size, group count, first threadgroup in the
/// Metal grid, and whether it takes the grid from the push field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Region {
    pub local: [u32; 3],
    pub groups: [u32; 3],
    pub base: [u32; 3],
    pub multi: bool,
}

/// One axis's pieces: `(base, count, local)`.
type Piece = (u32, u32, u32);

/// `count` groups of `local` threads from group `base`, in chunks of at most `max`.
fn chunks(out: &mut Vec<Piece>, base: u32, count: u32, local: u32, max: u32) {
    let max = max.max(1);
    let mut done = 0;
    while done < count {
        let n = (count - done).min(max);
        out.push((base + done, n, local));
        done += n;
    }
}

/// The regions of a direct dispatch whose axes split into `pieces`; `MultiRegion` whenever there
/// is more than one region or `force_multi`.
fn product(pieces: [Vec<Piece>; 3], force_multi: bool) -> Result<Vec<Region>, BindError> {
    let total: u64 = pieces.iter().map(|p| p.len() as u64).product();
    if total > MAX_REGIONS {
        let (i, _) = pieces
            .iter()
            .enumerate()
            .max_by_key(|(_, p)| p.len())
            .unwrap_or((0, &pieces[0]));
        return Err(BindError::Unsupported(Unsupported::not_representable(
            Feature::GroupCount {
                axis: Some(AXES[i]),
            },
        )));
    }
    let multi = force_multi || total > 1;
    let mut out = Vec::with_capacity(total as usize);
    for &(bx, cx, lx) in &pieces[0] {
        for &(by, cy, ly) in &pieces[1] {
            for &(bz, cz, lz) in &pieces[2] {
                out.push(Region {
                    local: [lx, ly, lz],
                    groups: [cx, cy, cz],
                    base: [bx, by, bz],
                    multi,
                });
            }
        }
    }
    Ok(out)
}

/// The regions of a dispatch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Planned {
    pub regions: Vec<Region>,
    /// The Metal grid (`threads_per_grid`) the `DispatchRegion` field reports (0 for an indirect
    /// dispatch, whose grid only the GPU knows).
    pub grid: [u32; 3],
    /// Metal's `threads_per_threadgroup`.
    pub nominal: [u32; 3],
    /// Some region runs only on the kernel's emulated-simdgroups variant.
    pub emulated: bool,
}

/// The widest tile threadgroup, and the tallest: Metal's largest tile, 32x32.
pub(crate) const TILE_THREADGROUP_SIDE: u32 = 32;

/// Whether the lowering runs a `width x height x 1` tile threadgroup in Metal's layout
/// (`lower::dispatch`'s `tile_pos` and `tile_index`, the order measured on an Apple M3 Max for every
/// side from 1 to 32): both sides even from 2 to 32, where every quad is whole so `sg · 32 + lane`
/// numbers the threads densely, or the single thread. A threadgroup with an odd side leaves the
/// lanes of the threads past it empty in its quads, which no Vulkan workgroup leaves.
fn tile_layout_measured(width: u32, height: u32) -> bool {
    if (width, height) == (1, 1) {
        return true;
    }
    let side = |n: u32| n.is_multiple_of(2) && (2..=TILE_THREADGROUP_SIDE).contains(&n);
    side(width) && side(height)
}

/// A tile function's dispatch: one threadgroup per tile, `Dispatch::Threadgroups`, of a threadgroup
/// the lowering lays out as Metal does ([`tile_layout_measured`]); any other is
/// `Unsupported{TileShading, NotImplemented}`: a threadgroup with an odd side, whose empty lanes
/// are not built, or one deeper than 1.
fn check_tile_dispatch(dispatch: &Dispatch) -> Result<(), BindError> {
    match *dispatch {
        Dispatch::Threadgroups {
            threads_per_group: [x, y, 1],
            ..
        }
        | Dispatch::ThreadgroupRange {
            threads_per_group: [x, y, 1],
            ..
        } if tile_layout_measured(x, y) => Ok(()),
        _ => Err(not_implemented(Feature::TileShading)),
    }
}

/// The grid of `groups` threadgroups of `l` threads, each axis at most `2^32 - 1` threads
/// (`Unsupported{GroupCount}` otherwise).
fn threadgroup_grid(groups: [u32; 3], l: [u32; 3]) -> Result<[u32; 3], BindError> {
    let mut grid = [0u32; 3];
    for i in 0..3 {
        grid[i] = groups[i].checked_mul(l[i]).ok_or_else(|| {
            BindError::Unsupported(Unsupported::not_representable(Feature::GroupCount {
                axis: Some(AXES[i]),
            }))
        })?;
    }
    Ok(grid)
}

/// The group counts of the one region a flat variant runs `need` threadgroups in: `need` spread
/// over x, then y, then z within `rules.max_count`, so their product is at least `need` (the
/// surplus invocations lie past the grid's last thread, which the lowering's guard retires). A
/// `need` the limits cannot hold is `Unsupported{GroupCount}`.
fn spread(need: u64, max_count: [u32; 3]) -> Result<[u32; 3], BindError> {
    let mut left = need;
    let mut out = [1u32; 3];
    for i in 0..3 {
        let max = u64::from(max_count[i].max(1));
        let take = left.min(max);
        out[i] = take as u32;
        left = left.div_ceil(take);
    }
    if left > 1 {
        return Err(BindError::Unsupported(Unsupported::not_representable(
            Feature::GroupCount {
                axis: Some(AXES[2]),
            },
        )));
    }
    Ok(out)
}

/// The regions of `dispatch` under `rules`, on the plan of a flat variant where `flat`
/// ([`VariantKey::flat_grid`]): there a `Threads` grid that is not a whole number of threadgroups
/// is one region of the dispatch's own threadgroup size, not an interior and its edges.
pub(crate) fn regions(
    rules: &LocalRules,
    dispatch: &Dispatch,
    flat: bool,
) -> Result<Planned, BindError> {
    if rules.tile {
        check_tile_dispatch(dispatch)?;
    }
    let l = dispatch.threads_per_group();
    // Metal's threadgroup: the tile where a tile function's threadgroup is the tile, which every
    // region then carries (`MultiRegion` 1), else the dispatch's.
    let nominal = rules.tile_threadgroup.unwrap_or(l);
    let tile_sized = rules.tile_threadgroup.is_some();
    for (i, (&x, &n)) in l.iter().zip(&nominal).enumerate() {
        if x == 0 || n == 0 || pack_nominal([x.max(n), 1, 1]).is_none() {
            return Err(BindError::Unsupported(Unsupported::not_representable(
                Feature::LocalSize {
                    axis: Some(AXES[i]),
                },
            )));
        }
    }
    let empty = |v: [u32; 3]| v.contains(&0);
    let planned = |regions: Vec<Region>, grid, emulated: bool| Planned {
        emulated: emulated && !regions.is_empty(),
        regions,
        grid,
        nominal,
    };
    match *dispatch {
        Dispatch::Threadgroups { groups, .. } => {
            let fit = fit(rules, &[l])?;
            let grid = threadgroup_grid(groups, nominal)?;
            if empty(groups) {
                return Ok(planned(Vec::new(), grid, false));
            }
            let pieces = std::array::from_fn(|i| {
                let mut p = Vec::new();
                chunks(&mut p, 0, groups[i], l[i], rules.max_count[i]);
                p
            });
            Ok(planned(
                product(pieces, tile_sized)?,
                grid,
                fit == Fit::EmulatedSimdgroups,
            ))
        }
        Dispatch::ThreadgroupRange {
            groups,
            first,
            count,
            ..
        } => {
            let fit = fit(rules, &[l])?;
            let grid = threadgroup_grid(groups, nominal)?;
            let start: [u32; 3] = std::array::from_fn(|i| first[i].min(groups[i]));
            let end: [u32; 3] =
                std::array::from_fn(|i| first[i].saturating_add(count[i]).min(groups[i]));
            if (0..3).any(|i| end[i] <= start[i]) {
                return Ok(planned(Vec::new(), grid, false));
            }
            let pieces = std::array::from_fn(|i| {
                let mut p = Vec::new();
                chunks(
                    &mut p,
                    start[i],
                    end[i] - start[i],
                    l[i],
                    rules.max_count[i],
                );
                p
            });
            // Every region takes its first threadgroup from the push field.
            Ok(planned(
                product(pieces, true)?,
                grid,
                fit == Fit::EmulatedSimdgroups,
            ))
        }
        Dispatch::Threads { threads, .. } => {
            if empty(threads) {
                return Ok(planned(Vec::new(), threads, false));
            }
            // A flat variant runs the whole grid in one region of the dispatch's own threadgroup
            // size: invocation `((wz · ny + wy) · nx + wx) · N + LocalInvocationIndex`, with
            // `N = l.x·l.y·l.z`, counts the grid's threads `x` fastest and takes the one it counts
            // to, and one past the grid's last thread returns before the body. The count runs over
            // the single dispatch's group counts, so splitting them into regions would restart it
            // in each and run the same threads again: the region is never chunked, and its counts
            // only have to cover the grid ([`spread`]).
            if flat {
                let fit = fit(rules, &[l])?;
                let n: u64 = l.iter().map(|&x| u64::from(x)).product();
                let total: u128 = threads.iter().map(|&x| u128::from(x)).product();
                let need = u64::try_from(total.div_ceil(u128::from(n))).map_err(|_| {
                    BindError::Unsupported(Unsupported::not_representable(Feature::GroupCount {
                        axis: None,
                    }))
                })?;
                let region = Region {
                    local: l,
                    groups: spread(need, rules.max_count)?,
                    base: [0; 3],
                    // As the one region of a flat variant's indirect dispatch, so that both run on
                    // the variant's one `VkPipeline`: a flat grid reads its threads from the
                    // region record whatever this says.
                    multi: false,
                };
                return Ok(planned(
                    vec![region],
                    threads,
                    fit == Fit::EmulatedSimdgroups,
                ));
            }
            let mut edges = false;
            let pieces: [Vec<Piece>; 3] = std::array::from_fn(|i| {
                let (full, rem) = (threads[i] / l[i], threads[i] % l[i]);
                let mut p = Vec::new();
                chunks(&mut p, 0, full, l[i], rules.max_count[i]);
                if rem > 0 {
                    edges = true;
                    p.push((full, 1, rem));
                }
                p
            });
            let out = product(pieces, edges)?;
            let locals: Vec<[u32; 3]> = out.iter().map(|r| r.local).collect();
            let fit = fit(rules, &locals)?;
            Ok(planned(out, threads, fit == Fit::EmulatedSimdgroups))
        }
        Dispatch::ThreadgroupsIndirect { .. } => {
            let fit = fit(rules, &[l])?;
            let region = Region {
                local: l,
                groups: [0; 3],
                base: [0; 3],
                multi: false,
            };
            Ok(planned(
                vec![region],
                [0; 3],
                fit == Fit::EmulatedSimdgroups,
            ))
        }
    }
}

/// The `StageInRegion` push field's bytes (two `uvec4`): the origin and the size of region `r`,
/// each padded with a zero `w`.
fn stage_in_region_bytes(r: StageInRegion) -> FieldBytes {
    let words = [r.origin, r.size]
        .into_iter()
        .flat_map(|v| v.into_iter().chain([0]));
    words.flat_map(|x| x.to_le_bytes()).collect()
}

impl BindingPlan {
    /// The commands of one compute dispatch of `state`. `state` must have the shape the plan was
    /// made for; the result depends on its offsets, lengths, and addresses and on the dispatch,
    /// never on the contents of an indirect buffer. A dispatch that runs on another variant of the
    /// pipeline ([`BindingPlan::dispatch_variant`]) is `BindError::NeedsVariant`.
    ///
    /// The plan of a flat variant ([`VariantKey::flat_grid`]) runs one region, which reads the
    /// grid from a 32-byte region record whose device address fills its single
    /// `PushHole::ReplayRegionAddress`:
    ///
    /// - A `ThreadgroupsIndirect` of the variant's threadgroup size runs at the group counts a
    ///   preprocess wrote with the grid's flat dispatch (`flat_dispatch`), which lie within the
    ///   device's limits, read by `vkCmdDispatchIndirect` with no clamp prepass; the record is the
    ///   one the preprocess wrote beside them, and the consumer orders both writes before the
    ///   dispatch.
    /// - A `Threads` dispatch runs at group counts of its own, enough threadgroups of its own
    ///   threadgroup size to cover the grid, whatever a whole number of them that grid is
    ///   (`regions`, a private reader that spreads the count a grid needs over the axes the
    ///   device admits); the record is [`DispatchCommands::region_record`], which the consumer
    ///   writes.
    ///
    /// Any other form is `Unsupported{GpuEncodedIndirectCommands, NotImplemented}`.
    ///
    /// The plan of a folded variant ([`VariantKey::folded_groups`]) runs one region too, of its
    /// threadgroup size, at the group counts a preprocess wrote (Metal's folded within the device's
    /// limits, `FlatArguments::FoldedThreadgroups`): `dispatch` is the `ThreadgroupsIndirect` of
    /// the call's threadgroup size, read by `vkCmdDispatchIndirect` with no clamp prepass, and the
    /// region reads its group counts and its grid builtins from the region record at its
    /// `PushHole::ReplayRegionAddress` (`MultiRegion` 1).
    pub fn instantiate_dispatch(
        &self,
        state: &BindState,
        dispatch: &Dispatch,
    ) -> Result<DispatchCommands, BindError> {
        let t = self.compute_template("instantiate_dispatch")?;
        let flat = self.variant.flat_grid;
        let records = match (flat || self.variant.folded_groups, dispatch) {
            (false, _) => Records::Binder,
            (true, Dispatch::ThreadgroupsIndirect { .. }) => Records::Flat,
            // A flat variant takes a direct dispatch too, whose record the consumer writes
            // ([`DispatchCommands::region_record`]).
            (true, Dispatch::Threads { .. }) if flat => Records::Flat,
            (true, _) => return Err(not_implemented(Feature::GpuEncodedIndirectCommands)),
        };
        let mut planned = regions(&t.local, dispatch, flat)?;
        if planned.emulated {
            return Err(BindError::NeedsVariant {
                variant: Box::new(self.emulated_variant()),
            });
        }
        // A folded variant's region reads its record.
        if self.variant.folded_groups {
            for r in &mut planned.regions {
                r.multi = true;
            }
        }
        instantiate(t, planned, state, dispatch, records)
    }

    /// The commands of one registered variant of a device replay
    /// ([`replay_preprocess`](crate::bind::replay_preprocess)): one region of `threads_per_group`
    /// threads per threadgroup whose group count is the replay's arguments
    /// ([`IndirectArgs::Replay`]), and no root table: every dispatch of the variant reads the slot
    /// table the preprocess wrote for its command, whose address fills its
    /// `PushHole::RootTableAddress` (`ReplayLayout::table_offset`). The region takes its grid,
    /// first threadgroup, and nominal threadgroup size from the region record the preprocess wrote
    /// for the dispatch's command and region (`MultiRegion` 1), whose address fills its
    /// `PushHole::ReplayRegionAddress` (`ReplayLayout::region_offset`), so one variant runs the
    /// interior, an edge, or a group-count chunk of any command. `state` binds no kernel buffer
    /// (the records do, `ReplayRefusal::KernelBufferBound` otherwise) and binds what the commands
    /// share: the threadgroup lengths of the variant and the resident set. A plan of a pipeline not
    /// compiled as a replay variant (`LoweringInputs::replay_bindings`), whose slots a record
    /// cannot supply, is `ReplayRefusal::NotReplayVariant`; a threadgroup size the plan's simd tier
    /// cannot run is `BindError::NeedsVariant`, as for any dispatch.
    pub fn instantiate_replay(
        &self,
        state: &BindState,
        threads_per_group: [u32; 3],
    ) -> Result<DispatchCommands, BindError> {
        let t = self.compute_template("instantiate_replay")?;
        check_replay(t, state)?;
        let dispatch = Dispatch::ThreadgroupsIndirect {
            indirect: IndirectRef::new(0, 0),
            threads_per_group,
        };
        let mut planned = regions(&t.local, &dispatch, self.variant.flat_grid)?;
        if planned.emulated {
            return Err(BindError::NeedsVariant {
                variant: Box::new(self.emulated_variant()),
            });
        }
        // Every region record states the dispatch's region.
        for r in &mut planned.regions {
            r.multi = true;
        }
        instantiate(t, planned, state, &dispatch, Records::Replay)
    }

    /// The variant of the pipeline a compute dispatch runs on: the plan's
    /// [`variant`](BindingPlan::variant), with `emulated_simdgroups` when a threadgroup of the
    /// dispatch has a partial simdgroup the kernel's flattened simd tier cannot run (its
    /// `threads_per_threadgroup`, or a `dispatchThreads` edge region, is not a multiple of the
    /// tier's subgroup size). A pure function of the plan and the dispatch, so a consumer can
    /// compile the variant before it binds. It reports the dispatch's local-size refusals as
    /// [`BindingPlan::instantiate_dispatch`] does, each region checked against the limits of the
    /// pipeline the result names: a result names a variant whose plan, for the same shape, states
    /// it and runs every region of the dispatch, except that the variant's plan checks the scratch
    /// its emulated simdgroups take beside the threadgroup memory (`ThreadgroupMemorySize`), which
    /// only the variant's compile knows.
    pub fn dispatch_variant(&self, dispatch: &Dispatch) -> Result<VariantKey, BindError> {
        let t = self.compute_template("dispatch_variant")?;
        Ok(
            if regions(&t.local, dispatch, self.variant.flat_grid)?.emulated {
                self.emulated_variant()
            } else {
                self.variant.clone()
            },
        )
    }

    /// The plan's variant with emulated simdgroups.
    fn emulated_variant(&self) -> VariantKey {
        let mut v = self.variant.clone();
        v.emulated_simdgroups = true;
        v
    }

    fn compute_template(&self, call: &str) -> Result<&ComputeTemplate, BindError> {
        self.compute
            .as_ref()
            .ok_or_else(|| internal(format!("{call} of a plan that is not a compute plan")))
    }

    /// The specialization [`instantiate_replay`](BindingPlan::instantiate_replay) gives the one
    /// region of a replay variant of `threads_per_group` threads: what the generated-commands path
    /// compares a consumer's variant against.
    pub(crate) fn replay_specialization(
        &self,
        threads_per_group: [u32; 3],
    ) -> Result<SpecializationKey, BindError> {
        let t = self.compute_template("replay_specialization")?;
        // A replay variant reads no stage-in region (`check_replay`), so no key bounds a fetch.
        Ok(region_specialization(t, threads_per_group, true, true))
    }
}

/// The specialization of a region of `local` threads of `t`: the template's fixed constants, the
/// local size, whether the region reads its record (`MultiRegion`), and whether a stage-in element
/// it fetches may lie past the first `2^32` bytes from its bind offset (`StageInReach`,
/// [`stage_in_reach`]).
fn region_specialization(
    t: &ComputeTemplate,
    local: [u32; 3],
    multi: bool,
    reach: bool,
) -> SpecializationKey {
    let n: u64 = local.iter().map(|&x| u64::from(x)).product();
    spec_key(&t.spec, |k| match k {
        SpecKey::LocalSize(a) => Some(u64::from(local[a as usize])),
        SpecKey::FlatLocalSize => Some(n),
        SpecKey::MultiRegion => Some(u64::from(multi)),
        SpecKey::StageInReach => Some(u64::from(reach)),
        other => t.fixed.iter().find(|(key, _)| *key == other).map(|f| f.1),
    })
}

/// The stride the compute encoder binds kernel buffer `index` with in `state`
/// (`setBuffer:offset:attributeStride:atIndex:` and its forms, `MTLAttributeStrideStatic`'s low
/// 32 bits), 0 where it binds none: what a stage-in layout of `MTLBufferLayoutStrideDynamic`
/// fetches at.
fn stage_in_stride(state: &BindState, index: u32) -> u32 {
    match state
        .stages
        .get(&Stage::Kernel)
        .and_then(|k| k.buffers.get(&index))
    {
        Some(
            &BoundBuffer::Buffer {
                attribute_stride: Some(s),
                ..
            }
            | &BoundBuffer::Bytes {
                attribute_stride: Some(s),
                ..
            },
        ) => s,
        _ => 0,
    }
}

/// The `SpecKey::StageInReach` of a dispatch of `t` over Metal's `grid` (0 on each axis for an
/// indirect dispatch, whose grid only the GPU knows) in stage-in region `region`, under `state`'s
/// strides. It says whether an element that a thread-position fetch reads may lie past the first
/// `2^32` bytes from its bind offset, where the fetch's reach check reads it as zero
/// (`lower::stage_io::fetch`). Each element is `origin + min(position / rate, size - 1)` on the
/// step's axis, so it is at most the origin plus the smaller of the region's last element and the
/// grid's last position over the rate. In a region of size 0 on the axis, whose size less 1 clamps
/// nothing, it is bounded by the grid alone. Where that bound is unknown (an indirect grid over
/// such a region) or the sum leaves 32 bits (the shader's add wraps), any element may be read.
fn stage_in_reach(
    t: &ComputeTemplate,
    region: Option<StageInRegion>,
    grid: [u32; 3],
    state: &BindState,
) -> bool {
    let Some(region) = region else {
        return !t.stage_in_grid.is_empty();
    };
    t.stage_in_grid.iter().any(|f| {
        let (Some(axis), Some(end)) = (
            crate::lower::stage_io::stage_in::grid_axis(f),
            crate::lower::stage_io::stage_in::element_end(f),
        ) else {
            return true;
        };
        let stride = f
            .stride
            .unwrap_or_else(|| stage_in_stride(state, f.buffer_index));
        // At stride 0 every element is the attribute's own bytes at the bind offset.
        if stride == 0 {
            return end > 1 << 32;
        }
        let rate = f.step_rate.max(1);
        let positions = grid[axis].checked_sub(1).map(|p| p / rate);
        let last = match (region.size[axis].checked_sub(1), positions) {
            (None, None) => return true,
            (None, Some(p)) => p,
            (Some(z), None) => z,
            (Some(z), Some(p)) => z.min(p),
        };
        let Some(element) = region.origin[axis].checked_add(last) else {
            return true;
        };
        u64::from(element) * u64::from(stride) + end > 1 << 32
    })
}

/// Whether `t` is a device-replay variant's template that `state` can instantiate: every buffer
/// slot is one of the 32 kernel buffers a record binds, read through its root entry; no field a
/// record cannot supply (a descriptor slot's residual, a slot's `SlotGuestBase`, the stage-in
/// region or a stage-in stride); and no kernel buffer bound by `state`.
fn check_replay(t: &ComputeTemplate, state: &BindState) -> Result<(), BindError> {
    let refused = |reason| BindError::InvalidReplay { reason };
    let slots = t.buffers.iter().all(|b| {
        b.form == BufferForm::Psb
            && b.slot.stage == Stage::Kernel
            && b.slot.class == SlotClass::Buffer
            && (b.slot.index as usize) < BINDING_COUNT
    });
    let fields = t.fields.iter().all(|f| {
        !matches!(
            f.field,
            PushField::SlotBindOffset(_)
                | PushField::SlotGuestBase(_)
                | PushField::StageInRegion
                | PushField::StageInStride(_)
        )
    });
    if !slots || !fields || t.stage_in_region {
        return Err(refused(ReplayRefusal::NotReplayVariant));
    }
    if state
        .stages
        .get(&Stage::Kernel)
        .is_some_and(|k| !k.buffers.is_empty())
    {
        return Err(refused(ReplayRefusal::KernelBufferBound));
    }
    Ok(())
}

/// What writes a dispatch's region records and group counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Records {
    /// None: the binder writes each region's `DispatchRegion`, and an indirect dispatch's counts
    /// are Metal's, clamped to the device's limits where it needs.
    Binder,
    /// A flat or folded variant's record at its `ReplayRegionAddress`: for an indirect dispatch
    /// the preprocess writes it, and the group counts at its indirect arguments within the
    /// device's limits; for a direct dispatch on a flat variant the consumer writes it
    /// ([`DispatchCommands::region_record`]) and the region takes its own group counts.
    Flat,
    /// The device replay's preprocess: the records per command and region, the arguments, and the
    /// slot table that stands for the root table.
    Replay,
}

/// The commands of `dispatch` planned as `planned`, whose region records and group counts
/// `records` writes: for a device-replay variant (whose `dispatch` is the `ThreadgroupsIndirect` of
/// its threadgroup size) the region reads the replay's arguments and no root table is built.
fn instantiate(
    t: &ComputeTemplate,
    planned: Planned,
    state: &BindState,
    dispatch: &Dispatch,
    records: Records,
) -> Result<DispatchCommands, BindError> {
    let replay = records == Records::Replay;
    // A kernel that reads the stage-in region needs the encoder's.
    let region = match t.stage_in_region {
        false => None,
        true => Some(
            state
                .stage_in_region
                .ok_or(BindError::MissingStageInRegion)?,
        ),
    };
    let (push_start, push_size) = t.push_range.unwrap_or((0, 0));
    let mut push = vec![0u8; push_size as usize];
    let mut table = vec![0u8; t.table.map_or(0, |(_, _, size)| size as usize)];
    let compute = vk::ShaderStageFlags::COMPUTE;

    // Buffer slots: descriptor ranges and residuals, root entries, extents; texel buffers.
    let (mut descriptor_ranges, mut extents) = (Vec::new(), Vec::new());
    let bound = bind_buffers(&t.buffers, state, &mut descriptor_ranges, &mut extents)?;
    descriptor_ranges.extend(texel_ranges(&t.resources, state)?);

    // The root table: the Psb entries and the resident set.
    let mut push_holes = Vec::new();
    let root_table = match t.root_first {
        // A replay variant's root table is its command's slot table.
        Some(_) if replay => None,
        Some(first) => {
            let statics: Vec<(u32, RootEntry)> = bound
                .statics
                .iter()
                .map(|(slot, e)| (slot.index, *e))
                .collect();
            Some(stage_root_table(&statics, first, &table_entries(state)))
        }
        None => None,
    };

    // Push fields that do not change per region.
    for f in &t.fields {
        let bytes: FieldBytes = match f.field {
            // Only a preprocess writes region records.
            PushField::ReplayRegionAddress if records == Records::Binder => {
                return Err(not_implemented(Feature::GpuEncodedIndirectCommands));
            }
            PushField::RootTableAddress
            | PushField::ExecutionStatusAddress
            | PushField::ReplayRegionAddress => {
                let PushLoc::Push { offset } = f.loc else {
                    return Err(not_implemented(Feature::Lowering(LowerFamily::Dispatch)));
                };
                push_holes.push(match f.field {
                    PushField::RootTableAddress => PushHole::RootTableAddress {
                        stages: compute,
                        offset,
                    },
                    PushField::ExecutionStatusAddress => PushHole::ExecutionStatusAddress {
                        stages: compute,
                        offset,
                    },
                    _ => PushHole::ReplayRegionAddress {
                        stages: compute,
                        offset,
                    },
                });
                continue;
            }
            PushField::DispatchRegion => continue,
            PushField::StageInRegion => stage_in_region_bytes(region.ok_or_else(|| {
                internal("a stage-in region push field without the region's fact")
            })?),
            // The stride the encoder bound the stage-in buffer or inline bytes with, 0 without
            // one, as Metal reads a dynamic layout: `MTLAttributeStrideStatic`'s is its low 32
            // bits, whose elements past the first the fetch reads past the bind offset's 2^32
            // bytes (`lower::stage_io::fetch`).
            PushField::StageInStride(index) => {
                FieldBytes::of(&stage_in_stride(state, index).to_le_bytes())
            }
            field => slot_field(field, &bound.residuals, &t.resources, state)
                .ok_or_else(|| internal(format!("a compute push field {field:?}")))?,
        };
        put(&mut push, push_start, &mut table, f.loc, &bytes)?;
    }
    let region_field = t
        .fields
        .iter()
        .find(|f| f.field == PushField::DispatchRegion)
        .map(|f| f.loc);

    // Regions.
    let Planned {
        regions: planned,
        grid,
        nominal,
        ..
    } = planned;
    let reach = stage_in_reach(t, region, grid, state);
    let mut regions = Vec::with_capacity(planned.len());
    for r in planned {
        let mut bytes = push.clone();
        if let Some(loc) = region_field {
            let b = region_bytes(r.base, grid, nominal)
                .ok_or_else(|| internal("an unpackable nominal threadgroup size"))?;
            put(&mut bytes, push_start, &mut [], loc, &b)?;
        }
        let specialization = region_specialization(t, r.local, r.multi, reach);
        let indirect = match *dispatch {
            Dispatch::ThreadgroupsIndirect { .. } if replay => Some(IndirectArgs::Replay),
            Dispatch::ThreadgroupsIndirect { indirect, .. } => {
                Some(if records == Records::Binder && needs_clamp(&t.local) {
                    IndirectArgs::ClampScratch
                } else {
                    IndirectArgs::Buffer(indirect)
                })
            }
            _ => None,
        };
        regions.push(DispatchRegion {
            specialization,
            group_count: r.groups,
            push_constants: if push_size == 0 {
                Vec::new()
            } else {
                vec![PushConstantWrite {
                    stages: compute,
                    offset: push_start,
                    bytes,
                }]
            },
            indirect,
            stage_flags: t.local.stage_flags,
            required_subgroup_size: t.local.required_subgroup_size,
        });
    }

    // A preprocess checks its group counts against the limit.
    let indirect_clamp = match *dispatch {
        Dispatch::ThreadgroupsIndirect { indirect, .. }
            if records == Records::Binder && needs_clamp(&t.local) =>
        {
            Some(
                IndirectClamp::new(indirect, t.local.max_count, t.storage_alignment).ok_or_else(
                    || {
                        BindError::Unsupported(Unsupported::not_representable(
                            Feature::GroupCount { axis: None },
                        ))
                    },
                )?,
            )
        }
        _ => None,
    };
    // A direct dispatch on a flat variant reads its region from the record the consumer writes
    // (base 0, the dispatch's threadgroup size, and the grid in threads), since the flat lowering
    // reads the record by address wherever its group counts come from.
    let region_record = match (records, *dispatch) {
        (Records::Flat, Dispatch::ThreadgroupsIndirect { .. }) | (Records::Binder, _) => None,
        (Records::Flat, _) => Some(
            region_bytes([0; 3], grid, nominal)
                .ok_or_else(|| internal("an unpackable nominal threadgroup size"))?,
        ),
        (Records::Replay, _) => None,
    };
    let mut descriptor_bytes = match t.table {
        Some((set, binding, _)) => vec![DescriptorBytes {
            set,
            binding,
            bytes: table,
        }],
        None => Vec::new(),
    };
    descriptor_bytes.extend(crate::bind::heap::table_bytes(&t.resources, state)?);
    let uses = match *dispatch {
        Dispatch::ThreadgroupsIndirect { .. } => {
            let (class, stages) = if indirect_clamp.is_some() {
                (
                    AccessClass::Storage,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                )
            } else {
                (
                    AccessClass::Indirect,
                    vk::PipelineStageFlags2::DRAW_INDIRECT,
                )
            };
            vec![ResourceUse {
                resource: UseTarget::Indirect,
                access: AccessSet::READ,
                class,
                stages,
                range: Some(ByteRange {
                    offset: 0,
                    len: u64::from(IndirectClamp::SCRATCH_BYTES),
                }),
            }]
        }
        Dispatch::Threadgroups { .. }
        | Dispatch::Threads { .. }
        | Dispatch::ThreadgroupRange { .. } => Vec::new(),
    };
    Ok(DispatchCommands {
        regions,
        descriptor_bytes,
        descriptor_ranges,
        extents,
        push_holes,
        region_record,
        root_table,
        indirect_clamp,
        uses,
    })
}

/// Whether an indirect dispatch needs the clamp prepass: some axis of the device's group-count
/// limit is below Metal's `2^32 - 1`.
fn needs_clamp(rules: &LocalRules) -> bool {
    rules.max_count.iter().any(|&c| c < u32::MAX)
}

#[cfg(test)]
mod tests;
