//! `lower::dispatch`: kernel local size, dispatch builtins, and regions.
//!
//! **Local size.** The local size is never a translation input. It is the specialization constants
//! `sx, sy, sz` (`SpecKey::LocalSize`, default 1), which are also Metal's `threads_per_threadgroup`
//! for the region being dispatched. In the tiers without flattening (`NoSubgroup`, `Linear`,
//! `PartialFragment`) the Vulkan workgroup size is their `WorkgroupSize` composite
//! ([`LocalSizeForm::Spec`]); in the flattened tiers (`Direct`, `Cluster32`) it is `(sn, 1, 1)`
//! with `sn = sx·sy·sz` a specialization constant the planner writes (`SpecKey::FlatLocalSize`,
//! [`LocalSizeForm::Flat`]), so no `OpSpecConstantOp` is placed in `WorkgroupSize` and no
//! `maintenance4` is needed. An object function's constants are keys of its own
//! (`SpecKey::ObjectLocalSize`, `ObjectFlatLocalSize` and `ObjectMultiRegion`,
//! [`DispatchPlan::keys`]): link gives a key one id over the pipeline, and a mesh render pipeline's
//! object and mesh threadgroups take separate sizes.
//!
//! **Deep threadgroups.** A kernel of the `Spec` form declares `SpecKey::DeepThreadgroup` ([`DispatchPlan::deep`]), under
//! which its `WorkgroupSize` is `(sn, 1, 1)`, `sn = sx·sy·sz` (each part an `OpSpecConstantOp`
//! select of the constants) and its position in the threadgroup `LocalInvocationIndex`
//! unflattened over `(sx, sy, sz)`: Metal allows 1024 threads along `z`, where a device's
//! `maxComputeWorkGroupSize[2]` may hold 64, and a threadgroup of at most 1024 threads in one row
//! fits every device the profile serves. The linear index keeps its value, `x` fastest either
//! way, and so does every builtin derived from it, the workgroup's and the grid's. The runtime
//! sets it for a threadgroup deeper than [`DEEP_THREADGROUP`] alone, so a pipeline of any other
//! size specializes it away.
//!
//! **Builtins** ([`KernelBuiltin`]). Every Metal kernel builtin derives from a handful of Vulkan
//! ones and the linear local index `lin`:
//!
//! - `lin` is `LocalInvocationIndex`, or, flattened, `SubgroupId · S + SubgroupLocalInvocationId`
//!   with `S` the subgroup size the pipeline pins, a constant (the `SubgroupSize` builtin only
//!   where the stage pins none), so every Metal identity (`tid = sg · 32 + lane`) holds whatever
//!   order the driver gives subgroups;
//! - the position in the threadgroup is `LocalInvocationId`, or, flattened, `lin` unflattened over
//!   `(sx, sy, sz)`;
//! - the simdgroup builtins are `lin & 31`, `lin >> 5`, 32, and `⌈N / 32⌉` with `N = sx·sy·sz`, and
//!   the compute quad builtins `lin & 3`, `lin >> 2`, and `⌈N / 4⌉` (Metal compute quads are four
//!   consecutive linear threads), in every tier;
//! - the grid builtins come from `WorkgroupId`, `NumWorkgroups`, and, for a multi-region
//!   dispatch, the region push field (below).
//!
//! **Tile functions**. Metal's tile dispatch lays its threads out as a fragment pass does, measured
//! on an Apple M3 Max for every threadgroup of 1 to 32 threads a side, in 16x16, 32x16 and 32x32
//! tiles alike: the threadgroup is `⌈width / 2⌉` by `⌈height / 2⌉` 2x2 quads, which simdgroups
//! take eight at a time in Metal's quad order (columns 8 quads wide, within a column its rows of 2
//! quads, within a row its blocks 4 quads wide, within a block its quads row-major), each quad's
//! four threads its lanes in row-major order, where a thread past an odd side leaves its lane
//! empty; and `thread_index_in_threadgroup` runs over the threadgroup's quads in row-major order
//! across its whole width, so for a 16-wide threadgroup it is not `sg · 32 + lane`. A tile function
//! therefore takes `lin` as Metal's `sg · 32 + lane` (its simd builtins and operations are
//! unchanged) and derives its position from it by that order (`KernelDecl::tile_pos`), and its
//! thread and quadgroup indices from the position (`KernelDecl::tile_index`). The binder runs a
//! tile function only over threadgroups whose lanes it fills, both sides even, or the single
//! thread (`bind::dispatch`).
//!
//! **Regions**. A kernel that reads a grid builtin (`thread_position_in_grid`,
//! `threadgroup_position_in_grid`, `threads_per_grid`, `threadgroups_per_grid`, and the
//! `dispatch_*` nominal sizes) declares the boolean `SpecKey::MultiRegion` and the push field
//! `PushField::DispatchRegion`, two `uvec4`:
//!
//! - `region[0].xyz`: the region's first threadgroup position in the Metal grid;
//! - `region[0].w`: Metal's nominal threads per threadgroup `L`, packed as
//!   `(Lx − 1) | (Ly − 1) << 10 | (Lz − 1) << 20` (each axis is at most 1024 in Metal);
//! - `region[1].xyz`: Metal's `threads_per_grid`; `region[1].w` is 0, or [`NONUNIFORM`] for a
//!   partial object threadgroup's draw or a kernel's ragged grid on its partial-threadgroups
//!   variant (below).
//!
//! With `MultiRegion = 0` (single-region `Threadgroups` and every `ThreadgroupsIndirect`) the push
//! field is not read: the region base is 0, `L = (sx, sy, sz)`, `threadgroups_per_grid =
//! NumWorkgroups`, and `threads_per_grid = NumWorkgroups · L`, so GPU-written indirect arguments
//! work. With `MultiRegion = 1` (the edge regions of `dispatchThreads` and group-count splits,
//! each dispatched with its exact local size) `threadgroup_position_in_grid = base + WorkgroupId`,
//! `thread_position_in_grid = (base + WorkgroupId) · L + position in threadgroup` (an edge axis
//! has one group), and `threadgroups_per_grid = ⌈threads_per_grid / L⌉`. Both forms are
//! evaluated and selected by the specialization constant; the packing never yields a zero divisor,
//! so the unselected form is always defined. An unfolded object function's single-region
//! `threadgroups_per_grid` is `⌈threads_per_grid / L⌉` too, of `NumWorkgroups · L` wrapped at 2^32
//! as Metal's object grid is (measured: an indirect draw runs every threadgroup and reports the
//! grid its wrapped threads fill); it is `NumWorkgroups` wherever no axis wraps.
//!
//! Under the identity domains ([`DispatchPlan::identity`]) every dispatch states its region,
//! and the region's grid selects the grid builtins' form: a dispatch the host knows
//! (`dispatchThreads`, `dispatchThreadgroups`, and each piece of a split of either) states Metal's
//! `threads_per_grid` in `region[1].xyz`, so the pieces of a split read Metal's whole grid; an
//! indirect dispatch states `region[1].xyz = 0` and takes the single-region form, its grid
//! `NumWorkgroups · L`, which needs its workgroup size to be Metal's threadgroup size. So one
//! pipeline runs every dispatch of a threadgroup size. The base is `region[0].xyz` on every
//! dispatch (0 for one that is not split and for an indirect one), and `MultiRegion` says only
//! whether `L` is the region's nominal size (1, a region whose threadgroup size is not its
//! pipeline's, as the binder's edge regions are) or the pipeline's `(sx, sy, sz)` (0, the
//! default): a dispatch at its pipeline's size computes its positions from that size and the base
//! alone, and the driver folds the region's decode away when it specializes.
//!
//! **Partial object threadgroups** ([`DispatchPlan::partial_groups`]). Metal's `drawMeshThreads`
//! with an object function launches the object threadgroups its threads fill, rounded up, the
//! last along an axis running only the threads left, as `dispatchThreads` runs a kernel's
//! (measured on an M3 Max, macOS 27.0.1: 6 threads in threadgroups of 4 run a threadgroup of 4
//! and one of 2, whose `threads_per_threadgroup` is 2). A draw is one dispatch of a local size,
//! so the binder draws the threadgroups at the whole size `L` with `MultiRegion` 1, Metal's
//! threads as `region[1].xyz`, and `region[1].w` [`NONUNIFORM`]. Where that flag is set, an
//! object function whose body holds no workgroup barrier answers its threadgroup's threads,
//! `min(L, threads_per_grid − position · L)` per axis, as `threads_per_threadgroup` and in its
//! simdgroup and quadgroup counts, and its position in the threadgroup as `lin` unflattened over
//! them (Metal's numbering of the smaller threadgroup); an invocation whose `lin` is at or past
//! their product returns before the body ([`emit_entry`]), so the body runs Metal's threads
//! alone. Without the flag every answer is the uniform one. A body that holds a workgroup barrier
//! is walked predicated instead (`simd::divergent`'s partial object threadgroups): such an
//! invocation is not live from entry ([`KernelDecl::live`] is the walk's first `live`), so it
//! reaches each barrier and runs nothing else of the body, where a returned one would leave the
//! others at a barrier it never reaches, which Vulkan leaves undefined.
//!
//! **Partial threadgroups of a kernel** (`LoweringInputs::partial_threadgroups`). A kernel's
//! `dispatchThreads` whose grid is not a whole number of threadgroups runs the same way on the
//! kernel's partial-threadgroups variant: the binder dispatches the threadgroups its threads fill,
//! rounded up, at the call's size `L`, per chunk of group counts, each region `MultiRegion` 1 with
//! `region[1].w` [`NONUNIFORM`], and the kernel's last threadgroup along an axis answers Metal's
//! partial threadgroup as an object function's does (measured on an M3 Max, macOS 27.0.1, as for
//! the split it replaces: `threads_per_threadgroup` is the threads left), its invocations past
//! them returning before the body or walking it predicated. Its simdgroups are `lin`'s, so the
//! last one of a partial threadgroup is Metal's partial simdgroup: its lanes past the threads are
//! inactive, and every lane read ballots its live lanes (`simd::partly_live`). One variant of
//! the call's threadgroup size runs every ragged grid, where the split runs an edge region of each
//! remainder, a `VkPipeline` of its own. A whole grid, and a threadgroup dispatch, run on the
//! variant's plan as the same regions unflagged: every invocation is live, and
//! `threads_per_threadgroup` is the call's.
//!
//! **Grid tiles** ([`DispatchPlan::grid_edge`], `RuntimeFact::GridTiles`). A kernel that does not
//! observe its threadgroup partition reads only `thread_position_in_grid` and `threads_per_grid`,
//! so neither Metal's threadgroups nor their partial edges reach it: only the set of positions
//! does. Under the identity domains its partial-threadgroups variant runs its grid in tiles. Under `MultiRegion` 1 its workgroup of
//! [`GRID_TILE_THREADS`] is one tile, whose power-of-two sides the region's nominal size gives
//! ([`grid_tile`] picks them from the grid's extents): the invocation's position is the tile's
//! origin `(base + WorkgroupId) << s` plus `LocalInvocationIndex` masked and shifted by the sides,
//! `s` the bits of `L − 1` per axis, all of them uniform values, so nothing divides. Where the
//! region is flagged [`NONUNIFORM`], which a runtime sets for a grid its workgroups do not divide,
//! an invocation whose position is past the region's grid on some axis returns before the body:
//! it compares its position with the grid's last thread (`region[1].xyz − 1`, wrapping), and
//! that test and the position it decodes sit inside one branch on the flag, a uniform value.
//! An unflagged region (a grid the workgroups divide, an indirect dispatch) branches past
//! them, and the body derives its position only where it reads it. Under
//! `MultiRegion` 0 it runs at its pipeline's size, its position `(base + WorkgroupId) · (sx, sy,
//! sz)` plus its position in the workgroup: an indirect dispatch, whose region's grid 0 wraps to
//! `u32::MAX`, so every invocation is live, or a direct one in workgroups of a size its runtime
//! picks, whose invocations past the grid return before the body as a tile's do. Every profile
//! (`Device::from_queries`) answers Metal's partial threadgroup as above.
//!
//! **Flat grids**. A flat variant (`LoweringInputs::flat_grid`, of a device replay or not) of a
//! kernel that does not observe its threadgroup partition runs any grid in threadgroups of its own
//! local size `N = sx·sy·sz`, dispatched `(nx, ny, nz)`: invocation `((wz · ny + wy) · nx + wx) ·
//! N + LocalInvocationIndex` counts the grid's threads `x` fastest over `threads_per_grid =
//! region[1].xyz`, in 64 bits, and takes the position it counts to; one past the grid's last thread
//! returns before the body ([`emit_entry`]). Such a kernel reads no builtin but
//! `thread_position_in_grid` and `threads_per_grid` (the compile refuses any other kernel under the
//! input). Of the call's threadgroups (`LoweringInputs::call_threadgroups`: a one-row grid in
//! threadgroups of one row), the invocation counts by `lin` in place of `LocalInvocationIndex`,
//! so in a flattened tier its thread is its simdgroup lane's, and workgroup `w` is Metal's
//! threadgroup `w` with its simdgroups and quadgroups: a kernel that observes its partition through
//! its simdgroups alone reads `thread_index_in_simdgroup`, `thread_index_in_quadgroup` and
//! `threads_per_simdgroup` too, and its simdgroup operations run on the lanes left live, as on
//! Metal's partial simdgroup. A module that would hold a workgroup barrier there is refused
//! (`select`).
//!
//! **Folded groups**. A folded variant (`LoweringInputs::folded_groups`) runs a threadgroup
//! dispatch whose group counts are GPU data as one dispatch `(nx, ny, nz)` within the device's
//! limits, in workgroups of the call's threadgroup size: workgroup `(wx, wy, wz)` is Metal's
//! threadgroup number `(wz · ny + wy) · nx + wx`, counted over Metal's group counts `region[1].xyz
//! / (sx, sy, sz)` `x` fastest, in 64 bits ([`KernelDecl::folded`]); a workgroup past the last
//! returns before the body as a whole ([`emit_entry`]). Its threadgroup and thread positions in the
//! grid come from that count; its other grid builtins read the region record at
//! `ReplayRegionAddress` (base 0, `MultiRegion` 1).
//!
//! **Facts.** Every kernel depends on `ThreadsPerThreadgroup` and `LocalSizeLimits`, and on
//! `IndirectGroupCount` (any kernel may be dispatched indirectly, and the planner clamps
//! GPU-written counts); a flattened kernel on `LocalSizeConstraint{multiple}` (the tier's subgroup
//! size); a kernel reading grid builtins on `DispatchGrid`; a flat grid on `DispatchGrid` and
//! `FlatGrid`; a folded variant on `DispatchGrid` and `FoldedGroups`; a kernel's
//! partial-threadgroups variant on `DispatchGrid` and `PartialThreadgroups`, and one that runs its
//! grid in tiles on `GridTiles`. A threadgroup-memory length is a `ThreadgroupMemoryLength{index}`
//! specialization ([`threadgroup_length`]), which the arena lowering declares through this module.
//!
//! Its hooks into `select` and `lower::emit`: the kernel's dispatch plan, made once the subgroup
//! tier is known ([`plan_kernel`]); dispatch builtin parameters ([`select_param`], and
//! [`emit_entry`], which derives each one the body reads once in the function's first block); and
//! the workgroup size, specialization constants, and builtin variables ([`declare`]).

use crate::air::abi::{Builtin, ParamBinding, ParamRole};
use crate::air::ir::{ArgIdx, Value};
use crate::device::subgroup::{self, SimdClass, Tier};
use crate::error::{Error, Feature, LowerFamily, Unsupported};
use crate::lower::arith;
use crate::lower::emit::EmitCx;
use crate::lower::select::{Plan, SelectCx};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::sym::{DispatchKeys, PushField, SpecKey};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{ConstId, DemoteTy, GlobalId, ScalarTy, TypeId, Value as SValue};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, ConvOp, ExecMode, GlslOp, IoDir, MemAccess, SpecOp, Type, UnOp,
};

/// The builder's error for a value the declarations do not hold.
fn dangling_value() -> crate::spirv::error::BuildError {
    crate::spirv::error::BuildError::Dangling(crate::spirv::error::Dangling::Value(SValue::new(
        u32::MAX,
    )))
}

fn not_lowered() -> Error {
    Unsupported::not_implemented(Feature::Lowering(LowerFamily::Dispatch)).into()
}

/// A Metal kernel builtin the dispatch family derives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KernelBuiltin {
    ThreadPositionInGrid,
    ThreadPositionInThreadgroup,
    ThreadgroupPositionInGrid,
    ThreadsPerThreadgroup,
    ThreadsPerGrid,
    ThreadgroupsPerGrid,
    ThreadIndexInThreadgroup,
    ThreadIndexInSimdgroup,
    SimdgroupIndexInThreadgroup,
    ThreadsPerSimdgroup,
    SimdgroupsPerThreadgroup,
    ThreadIndexInQuadgroup,
    QuadgroupIndexInThreadgroup,
    QuadgroupsPerThreadgroup,
    DispatchThreadsPerThreadgroup,
    DispatchSimdgroupsPerThreadgroup,
    /// A tile function's `[[render_target_array_index]]`: 0, the pass being one that is not layered
    /// (the binder refuses a layered pass of a tile function).
    RenderTargetArrayIndex,
}

impl KernelBuiltin {
    /// Every builtin of a compute kernel (a tile function also reads `RenderTargetArrayIndex`).
    pub const ALL: [KernelBuiltin; 16] = [
        KernelBuiltin::ThreadPositionInGrid,
        KernelBuiltin::ThreadPositionInThreadgroup,
        KernelBuiltin::ThreadgroupPositionInGrid,
        KernelBuiltin::ThreadsPerThreadgroup,
        KernelBuiltin::ThreadsPerGrid,
        KernelBuiltin::ThreadgroupsPerGrid,
        KernelBuiltin::ThreadIndexInThreadgroup,
        KernelBuiltin::ThreadIndexInSimdgroup,
        KernelBuiltin::SimdgroupIndexInThreadgroup,
        KernelBuiltin::ThreadsPerSimdgroup,
        KernelBuiltin::SimdgroupsPerThreadgroup,
        KernelBuiltin::ThreadIndexInQuadgroup,
        KernelBuiltin::QuadgroupIndexInThreadgroup,
        KernelBuiltin::QuadgroupsPerThreadgroup,
        KernelBuiltin::DispatchThreadsPerThreadgroup,
        KernelBuiltin::DispatchSimdgroupsPerThreadgroup,
    ];

    /// The kernel builtin of ABI builtin `b`. The stage-in region's `[[grid_origin]]` and
    /// `[[grid_size]]` are `lower::stage_io`'s, and the render target array index is a tile
    /// function's (refused in any other kernel by [`plan_kernel`]); any other render-stage builtin
    /// is not a kernel's.
    pub fn of(b: &Builtin) -> Result<KernelBuiltin, Error> {
        use KernelBuiltin as K;
        Ok(match b {
            Builtin::ThreadPositionInGrid => K::ThreadPositionInGrid,
            Builtin::ThreadPositionInThreadgroup => K::ThreadPositionInThreadgroup,
            Builtin::ThreadgroupPositionInGrid => K::ThreadgroupPositionInGrid,
            Builtin::ThreadsPerThreadgroup => K::ThreadsPerThreadgroup,
            Builtin::ThreadsPerGrid => K::ThreadsPerGrid,
            Builtin::ThreadgroupsPerGrid => K::ThreadgroupsPerGrid,
            Builtin::ThreadIndexInThreadgroup => K::ThreadIndexInThreadgroup,
            Builtin::ThreadIndexInSimdgroup => K::ThreadIndexInSimdgroup,
            Builtin::SimdgroupIndexInThreadgroup => K::SimdgroupIndexInThreadgroup,
            Builtin::ThreadsPerSimdgroup => K::ThreadsPerSimdgroup,
            Builtin::SimdgroupsPerThreadgroup => K::SimdgroupsPerThreadgroup,
            Builtin::ThreadIndexInQuadgroup => K::ThreadIndexInQuadgroup,
            Builtin::QuadgroupIndexInThreadgroup => K::QuadgroupIndexInThreadgroup,
            Builtin::QuadgroupsPerThreadgroup => K::QuadgroupsPerThreadgroup,
            Builtin::DispatchThreadsPerThreadgroup => K::DispatchThreadsPerThreadgroup,
            Builtin::DispatchSimdgroupsPerThreadgroup => K::DispatchSimdgroupsPerThreadgroup,
            // A kernel reads the layer it works on only as a tile function.
            Builtin::RenderTargetArrayIndex => K::RenderTargetArrayIndex,
            _ => return Err(not_lowered()),
        })
    }

    fn bit(self) -> u32 {
        1 << (self as u32)
    }

    /// Whether the builtin is a 3-component vector (`uint3`; a `uint` parameter reads `x`).
    pub fn is_vector(self) -> bool {
        use KernelBuiltin as K;
        matches!(
            self,
            K::ThreadPositionInGrid
                | K::ThreadPositionInThreadgroup
                | K::ThreadgroupPositionInGrid
                | K::ThreadsPerThreadgroup
                | K::ThreadsPerGrid
                | K::ThreadgroupsPerGrid
                | K::DispatchThreadsPerThreadgroup
        )
    }

    /// Whether the builtin depends on the region of a multi-region dispatch.
    pub fn reads_grid(self) -> bool {
        use KernelBuiltin as K;
        matches!(
            self,
            K::ThreadPositionInGrid
                | K::ThreadgroupPositionInGrid
                | K::ThreadsPerGrid
                | K::ThreadgroupsPerGrid
                | K::DispatchThreadsPerThreadgroup
                | K::DispatchSimdgroupsPerThreadgroup
        )
    }

    /// Whether the builtin answers Metal's threadgroup size, which a tile function compiled under
    /// `threadgroupSizeMatchesTileSize` takes from the tile: the region's nominal size there, where
    /// any other kernel answers its own threadgroup.
    pub fn follows_tile(self) -> bool {
        use KernelBuiltin as K;
        matches!(
            self,
            K::ThreadsPerThreadgroup | K::SimdgroupsPerThreadgroup | K::QuadgroupsPerThreadgroup
        )
    }

    /// Whether the builtin's answer depends on the tile pipeline's
    /// `threadgroupSizeMatchesTileSize` (`RuntimeFact::TileThreadgroup`): one of
    /// [`Self::follows_tile`], a `dispatch_` size, the position in the grid, the threads per grid,
    /// or the thread or quadgroup index in the threadgroup.
    fn tile_sized_answer(self) -> bool {
        use KernelBuiltin as K;
        self.follows_tile()
            || matches!(
                self,
                K::DispatchThreadsPerThreadgroup
                    | K::DispatchSimdgroupsPerThreadgroup
                    | K::ThreadPositionInGrid
                    | K::ThreadsPerGrid
                    | K::ThreadIndexInThreadgroup
                    | K::ThreadIndexInQuadgroup
                    | K::QuadgroupIndexInThreadgroup
            )
    }

    /// Whether using the builtin makes the function's simd class `LaneArithmetic`.
    pub fn lane_arithmetic(self) -> bool {
        use KernelBuiltin as K;
        matches!(
            self,
            K::ThreadIndexInSimdgroup
                | K::SimdgroupIndexInThreadgroup
                | K::SimdgroupsPerThreadgroup
                | K::ThreadsPerSimdgroup
        )
    }
}

/// A set of [`KernelBuiltin`]s.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct BuiltinSet(u32);

impl BuiltinSet {
    /// The set of `bs`.
    #[cfg(test)]
    pub fn of(bs: &[KernelBuiltin]) -> BuiltinSet {
        let mut s = BuiltinSet::default();
        for b in bs {
            s.insert(*b);
        }
        s
    }

    pub fn insert(&mut self, b: KernelBuiltin) {
        self.0 |= b.bit();
    }

    pub fn contains(self, b: KernelBuiltin) -> bool {
        self.0 & b.bit() != 0
    }

    /// Whether any builtin of the set satisfies `f`.
    pub fn any(self, f: impl Fn(KernelBuiltin) -> bool) -> bool {
        KernelBuiltin::ALL
            .iter()
            .any(|b| self.contains(*b) && f(*b))
    }
}

/// How a builtin's `u32`/`uvec3` value becomes its parameter's lowered representation
/// (`arith::lowered_type` of the parameter's AIR type).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Projection {
    /// A scalar of `bits` (16 or 32): the value, or `x` of a vector builtin.
    Scalar { bits: u8 },
    /// The first `len` (2 or 3) components of a vector builtin, as `bits`-wide integers.
    Vector { bits: u8, len: u8 },
}

impl Projection {
    /// The projection of builtin `b` onto lowered representation `lowered`, if one exists.
    pub fn of(b: KernelBuiltin, lowered: &DemoteTy) -> Option<Projection> {
        match lowered {
            DemoteTy::Vector { .. } if !b.is_vector() => None,
            _ => Projection::of_uvec3(lowered),
        }
    }

    /// The projection of a `uvec3` value (or its `x` for a scalar) onto lowered representation
    /// `lowered`, if one exists: 16- or 32-bit integers of 1 to 3 lanes.
    pub fn of_uvec3(lowered: &DemoteTy) -> Option<Projection> {
        let int = |s: &ScalarTy| match s {
            ScalarTy::Int {
                bits: bits @ (16 | 32),
            } => Some(*bits),
            _ => None,
        };
        match lowered {
            DemoteTy::Scalar(s) => int(s).map(|bits| Projection::Scalar { bits }),
            DemoteTy::Vector {
                elem,
                len: len @ (2 | 3),
            } => int(elem).map(|bits| Projection::Vector { bits, len: *len }),
            _ => None,
        }
    }

    /// The lowered representation this projection produces.
    pub fn lowered(self) -> DemoteTy {
        match self {
            Projection::Scalar { bits } => DemoteTy::Scalar(ScalarTy::Int { bits }),
            Projection::Vector { bits, len } => DemoteTy::Vector {
                elem: ScalarTy::Int { bits },
                len,
            },
        }
    }
}

/// One used builtin parameter of a kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlannedParam {
    /// The LLVM parameter index.
    pub param: u32,
    pub builtin: KernelBuiltin,
    pub projection: Projection,
}

/// How a kernel that reads a grid builtin tells a region of a dispatch the host knows
/// (`dispatchThreads`, `dispatchThreadgroups`, or a split of either, whose region the
/// `DispatchRegion` push field states) from a whole dispatch whose grid the GPU holds (an
/// indirect one, whose grid is `NumWorkgroups · L`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RegionSelect {
    /// The `MultiRegion` specialization constant.
    Spec(ConstId),
    /// Under the identity domains, the region's grid: `region[1].xyz` is Metal's
    /// `threads_per_grid` for every dispatch the host knows, and all 0 for an indirect one, so one
    /// pipeline runs both. The base is the region's on every dispatch, and the `MultiRegion`
    /// constant says only whether `L` is the region's nominal size.
    Sized(ConstId),
}

/// How a kernel's Vulkan workgroup size is expressed.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalSizeForm {
    /// `WorkgroupSize = (sx, sy, sz)`, each a `SpecKey::LocalSize` specialization constant.
    Spec,
    /// `WorkgroupSize = (sn, 1, 1)`, `sn = sx·sy·sz` the `SpecKey::FlatLocalSize` constant: the
    /// flattened Direct and Cluster32 tiers.
    Flat,
}

/// The dispatch plan of a kernel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DispatchPlan {
    pub local_size: LocalSizeForm,
    /// The builtins the kernel reads.
    pub builtins: BuiltinSet,
    /// Whether the kernel reads a grid builtin, so it declares `MultiRegion` and reads the
    /// `DispatchRegion` push field under it.
    pub multi_region: bool,
    /// Every used builtin parameter, by parameter index.
    pub params: Vec<PlannedParam>,
    /// The kernel is a tile function: its threads take Metal's tile-dispatch layout.
    pub tile: bool,
    /// The tile function's threadgroup is the tile
    /// (`LoweringInputs::threadgroup_size_matches_tile_size`): the builtins
    /// [`KernelBuiltin::follows_tile`] read the region's nominal size, and its thread index is
    /// `simdgroup · 32 + lane`.
    pub tile_sized: bool,
    /// The kernel runs its grid flat (`LoweringInputs::flat_grid`).
    pub flat_grid: bool,
    /// With `flat_grid`, its threadgroups are the call's (`LoweringInputs::call_threadgroups`):
    /// it counts its threads by `lin` and may read the lanes' builtins.
    pub call_threadgroups: bool,
    /// The specialization keys of its threadgroup size and grid form: an object function's own, the
    /// kernel's for every other stage.
    pub keys: DispatchKeys,
    /// The kernel folds Metal's group counts into its dispatch (`LoweringInputs::folded_groups`).
    pub folded_groups: bool,
    /// A folded object function: its draw's pieces each run `nx · ny · nz` of Metal's
    /// threadgroups, so its threadgroup's number adds `DrawIndex · nx · ny · nz`
    /// ([`KernelDecl::folded`]).
    pub draw_pieces: bool,
    /// An unfolded object function: its single-region `threadgroups_per_grid` is Metal's object
    /// grid's, the threadgroups its threads per grid fill once each axis wraps at 2^32
    /// (`bind::metal_object_grid`, measured): `⌈(NumWorkgroups · L mod 2^32) / L⌉`, which is
    /// `NumWorkgroups` unless an axis's threads wrap, as an indirect draw's may (Metal runs every
    /// threadgroup of it and reports the wrapped grid).
    pub object_grid: bool,
    /// An object function, or a kernel under `LoweringInputs::partial_threadgroups`
    /// (`lower::mesh::plan_partial_groups`): where its region says so ([`NONUNIFORM`]), its
    /// threadgroup runs only the threads Metal's partial threadgroup has (the module docs). It
    /// declares `MultiRegion` and reads the region whatever builtins it reads.
    pub partial_groups: bool,
    /// Of `partial_groups`, a kernel that does not observe its threadgroup partition, under the
    /// identity domains: under `MultiRegion` 1 its
    /// workgroup is one tile of the grid whose power-of-two sides the region's nominal size gives
    /// ([`grid_tile`]), and an invocation past the region's grid returns before the body, whatever
    /// the region's flags; under `MultiRegion` 0 it runs at its pipeline's size (the module docs'
    /// grid tiles).
    pub grid_edge: bool,
    /// Under the identity domains a kernel tells a region from an indirect dispatch by the
    /// region's grid, and reads its base and `L` as [`RegionSelect::Sized`] says.
    pub identity: bool,
    /// The subgroup size the pipeline pins in a flattened tier
    /// (`SubgroupRequirement::size`): `lin` multiplies the subgroup index by it as a constant and
    /// never reads `SubgroupSize`. `None` outside the flattened tiers and where the stage pins no
    /// size, where `lin` reads the builtin.
    pub pinned_subgroup_size: Option<u32>,
    /// A kernel of the [`LocalSizeForm::Spec`] form without compute derivatives (whose quads are two-dimensional): it declares
    /// `SpecKey::DeepThreadgroup`, under which its threadgroup runs as one row (the module docs).
    pub deep: bool,
    /// A kernel whose private arena lives in device memory (`memory::plan::ArenaLayout::device`):
    /// each invocation reaches its slot by its index in its dispatch ([`KernelDecl::arena_slot`]).
    pub private_arena: bool,
}

/// The dispatch family's module-scope objects: the kernel's, when the plan has one.
#[derive(Clone, Debug, Default)]
pub struct Declared {
    pub kernel: Option<KernelDecl>,
}

/// The builtin and projection of builtin parameter `b`, or its refusal: a builtin outside the
/// kernel set, or a parameter type whose lowered representation no projection produces.
fn plan_param(cx: &SelectCx<'_>, b: &ParamBinding) -> Result<PlannedParam, Error> {
    let ParamRole::Builtin(bi) = &b.role else {
        return Err(cx.invariant("a non-builtin parameter routed to dispatch"));
    };
    let builtin = KernelBuiltin::of(bi)?;
    if builtin == KernelBuiltin::RenderTargetArrayIndex && !crate::lower::access::tile::is_tile(cx)
    {
        return Err(crate::lower::access::tile::needs_tile());
    }
    let v = Value::Arg(ArgIdx::new(b.param));
    let ty = cx
        .value_ty(v)
        .ok_or_else(|| cx.invariant("a builtin parameter without a type"))?;
    let lowered = cx.lowered_repr(v, ty)?;
    let projection = Projection::of(builtin, &lowered).ok_or_else(not_lowered)?;
    Ok(PlannedParam {
        param: b.param,
        builtin,
        projection,
    })
}

/// Whether the function is a tile function whose threadgroup is the tile
/// (`LoweringInputs::threadgroup_size_matches_tile_size`).
fn tile_sized(cx: &SelectCx<'_>) -> bool {
    crate::lower::access::tile::is_tile(cx) && cx.inputs.threadgroup_size_matches_tile_size
}

/// Whether builtin `b` reads the `DispatchRegion` push field under `MultiRegion`: a grid builtin,
/// or, in a tile function whose threadgroup is the tile, one that answers the threadgroup's size.
fn reads_region(b: KernelBuiltin, tile_sized: bool) -> bool {
    b.reads_grid() || (tile_sized && b.follows_tile())
}

/// Whether a kernel's partial-threadgroups variant runs its grid in tiles
/// ([`DispatchPlan::grid_edge`]): a kernel that does not observe its threadgroup partition
/// (`pipeline::compile::partition`, decided from its AIR) under the identity domains, whose
/// regions state their base on every dispatch ([`RegionSelect::Sized`]).
fn grid_edge(cx: &SelectCx<'_>) -> bool {
    use crate::pipeline::compile::partition::{partition, Partition};
    cx.stage == crate::pipeline::sym::Stage::Kernel
        && crate::lower::access::identity(cx.profile)
        && partition(cx.entry) == Partition::Oblivious
}

/// Plan the kernel's dispatch under subgroup tier `tier`: the local-size form (flattened in the
/// Direct and Cluster32 tiers), the builtins it reads, and its facts. A kernel with compute
/// derivatives is refused in the flattened tiers (`ComputeDerivativesWithSimd`): flattening
/// destroys the 2D quad shape.
pub fn plan_kernel(cx: &mut SelectCx<'_>, tier: Tier) -> Result<DispatchPlan, Error> {
    let local_size = match tier {
        Tier::Direct | Tier::Cluster32 => LocalSizeForm::Flat,
        _ => LocalSizeForm::Spec,
    };
    if local_size == LocalSizeForm::Flat && cx.families.barrier.compute_derivatives {
        return Err(Unsupported::not_representable(Feature::ComputeDerivativesWithSimd).into());
    }
    let used = crate::lower::users::params_used(cx.body);
    let mut builtins = BuiltinSet::default();
    let mut params = Vec::new();
    for b in &cx.entry.ep.params {
        if !matches!(b.role, ParamRole::Builtin(_)) || !used.contains(&b.param) {
            continue;
        }
        // The stage-in region's builtins are `lower::stage_io`'s, and so is a mesh function's
        // amplification.
        if matches!(
            b.role,
            ParamRole::Builtin(
                Builtin::StageInGridOrigin
                    | Builtin::StageInGridSize
                    | Builtin::AmplificationId
                    | Builtin::AmplificationCount
            )
        ) {
            continue;
        }
        let p = plan_param(cx, b)?;
        builtins.insert(p.builtin);
        params.push(p);
    }
    // A `[[stage_in]]` attribute stepping by thread position reads it at entry.
    if cx.families.stage_io.reads_grid_position {
        builtins.insert(KernelBuiltin::ThreadPositionInGrid);
    }
    // A slice write every invocation shares reads its thread's index (`texture::cooperative`).
    if !cx.families.texture.cooperative.is_empty() {
        builtins.insert(KernelBuiltin::ThreadIndexInThreadgroup);
    }
    // A tile function's staging reads the tile and its threads.
    if crate::lower::access::tile::stages_any(&cx.families.access) {
        for b in crate::lower::access::tile::BUILTINS {
            builtins.insert(b);
        }
    }
    // An object or mesh function's wrapper reads its thread and its threadgroup's threads.
    if crate::lower::mesh::wraps(&cx.families.mesh) {
        for b in crate::lower::mesh::BUILTINS {
            builtins.insert(b);
        }
    }
    let tile_sized = tile_sized(cx);
    // A partial object threadgroup's threads come from the region.
    let partial_groups = cx.families.mesh.partial_groups;
    let multi_region = builtins.any(|b| reads_region(b, tile_sized)) || partial_groups;
    // The answers of the builtins its own parameters read follow the pipeline's flag.
    if crate::lower::access::tile::is_tile(cx)
        && params.iter().any(|p| p.builtin.tile_sized_answer())
    {
        cx.facts.insert(RuntimeFact::TileThreadgroup { tile_sized });
    }
    let flat_grid = cx.inputs.flat_grid;
    let call_threadgroups = flat_grid && cx.inputs.call_threadgroups;
    let folded_groups = cx.inputs.folded_groups;
    cx.facts.insert(RuntimeFact::ThreadsPerThreadgroup);
    cx.facts.insert(RuntimeFact::LocalSizeLimits);
    cx.facts.insert(RuntimeFact::IndirectGroupCount);
    let mut pinned_subgroup_size = None;
    if local_size == LocalSizeForm::Flat {
        let req = subgroup::requirement(tier, cx.stage, &cx.profile.fields().subgroup);
        pinned_subgroup_size = req.size;
        cx.facts.insert(RuntimeFact::LocalSizeConstraint {
            multiple: req.local_size_multiple,
        });
        if cx.families.barrier.rows_only {
            cx.facts.insert(RuntimeFact::RowLockstep);
        }
    }
    // A kernel compiled as its one-row variant states it, so its plans name the variant.
    if cx.inputs.one_row && cx.stage == crate::pipeline::sym::Stage::Kernel {
        cx.facts.insert(RuntimeFact::OneRow);
    }
    // So does one compiled as its unit-axes variant, which runs only threadgroups of one thread
    // along its axes.
    if cx.inputs.unit_axes != 0 && cx.stage == crate::pipeline::sym::Stage::Kernel {
        cx.facts.insert(RuntimeFact::UnitAxes {
            axes: cx.inputs.unit_axes,
            relief: 0,
        });
    }
    // A flat grid reads the grid's size whatever builtins it reads, and so does a folded
    // variant its group counts; the plans of both state their variant.
    if multi_region || flat_grid || folded_groups {
        cx.facts.insert(RuntimeFact::DispatchGrid);
    }
    if flat_grid {
        cx.facts.insert(RuntimeFact::FlatGrid);
    }
    if call_threadgroups {
        cx.facts.insert(RuntimeFact::CallThreadgroups);
    }
    if folded_groups {
        cx.facts.insert(RuntimeFact::FoldedGroups);
    }
    if partial_groups {
        cx.facts.insert(match cx.stage {
            crate::pipeline::sym::Stage::Kernel => RuntimeFact::PartialThreadgroups,
            _ => RuntimeFact::PartialObjectThreadgroups,
        });
    }
    let grid_edge = partial_groups && grid_edge(cx);
    if grid_edge {
        cx.facts.insert(RuntimeFact::GridTiles);
    }
    let deep = local_size == LocalSizeForm::Spec
        && cx.stage == crate::pipeline::sym::Stage::Kernel
        && !cx.families.barrier.compute_derivatives;
    Ok(DispatchPlan {
        local_size,
        builtins,
        multi_region,
        params,
        tile: crate::lower::access::tile::is_tile(cx),
        tile_sized,
        flat_grid,
        call_threadgroups,
        keys: SpecKey::dispatch_keys(cx.stage),
        folded_groups,
        draw_pieces: folded_groups && cx.stage == crate::pipeline::sym::Stage::Object,
        object_grid: !folded_groups && cx.stage == crate::pipeline::sym::Stage::Object,
        partial_groups,
        grid_edge,
        identity: crate::lower::access::identity(cx.profile),
        pinned_subgroup_size,
        deep,
        private_arena: cx.memory.priv_arena.device,
    })
}

/// Plan a used dispatch builtin parameter of a kernel: a builtin the family derives, of a type
/// whose lowered representation a projection produces. A grid builtin adds the `DispatchRegion`
/// push field, and a simd-lane builtin raises the simd class to `LaneArithmetic`.
pub fn select_param(cx: &mut SelectCx<'_>, b: &ParamBinding) -> Result<(), Error> {
    let p = plan_param(cx, b)?;
    if reads_region(p.builtin, tile_sized(cx)) {
        cx.push.insert(PushField::DispatchRegion);
    }
    if p.builtin.lane_arithmetic() && cx.simd == SimdClass::None {
        cx.simd = SimdClass::LaneArithmetic;
    }
    Ok(())
}

/// The work before the body, in the function's first block: derive each dispatch builtin parameter
/// the body reads ([`DispatchPlan::params`]) once and hold it for every block, which that block
/// dominates. Where the body first uses a parameter never moves its derivation, so a use that
/// canonicalization deletes cannot change how often the others derive it; a parameter the body does
/// not read is not derived.
pub fn emit_entry(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>) -> BResult<()> {
    let Some(plan) = cx.plan.dispatch.as_ref() else {
        return Ok(());
    };
    // The private arena's slot, in the first block, which every access of the arena reads.
    if plan.private_arena {
        cx.arena_slot(fb)?;
    }
    // A flat grid's invocation past the grid's last thread returns before the body. Its position
    // is derived here once, with the test, and is the `thread_position_in_grid` every parameter
    // reads: a second derivation would repeat the divisions in a construct of its own, which the
    // driver does not merge with this one. So is a grid edge's.
    let mut tested_position = None;
    if plan.flat_grid {
        let decl: &crate::lower::emit::Declared = cx.decl;
        let Some(decl) = decl.families.dispatch.kernel.as_ref() else {
            return Err(cx.fail("a flat grid without its kernel's declarations"));
        };
        let (pos, inside) =
            decl.flat(fb, &mut |fb| cx.push_field(fb, PushField::DispatchRegion))?;
        tested_position = Some(pos);
        let past = fb.unary(UnOp::LogicalNot, inside)?;
        let leave: ArmFn<'_, ()> = Box::new(|fb, _| {
            fb.ret(None)?;
            Ok(ArmEnd::Exits)
        });
        fb.if_(&mut (), past, leave, None)?;
    }
    // A folded variant's workgroup past Metal's last threadgroup returns before the body, as a
    // whole: the test reads only workgroup-uniform values.
    if plan.folded_groups {
        let decl: &crate::lower::emit::Declared = cx.decl;
        let Some(decl) = decl.families.dispatch.kernel.as_ref() else {
            return Err(cx.fail("folded groups without its kernel's declarations"));
        };
        let (_, inside) =
            decl.folded(fb, &mut |fb| cx.push_field(fb, PushField::DispatchRegion))?;
        let past = fb.unary(UnOp::LogicalNot, inside)?;
        let leave: ArmFn<'_, ()> = Box::new(|fb, _| {
            fb.ret(None)?;
            Ok(ArmEnd::Exits)
        });
        fb.if_(&mut (), past, leave, None)?;
    }
    // An invocation past a partial threadgroup's threads, or past the grid on a grid edge, returns
    // before the body, or, where the body is walked predicated for its barriers, walks it never
    // live.
    if plan.partial_groups {
        let decl: &crate::lower::emit::Declared = cx.decl;
        let Some(decl) = decl.families.dispatch.kernel.as_ref() else {
            return Err(cx.fail("partial threadgroups without the kernel's declarations"));
        };
        let mut region =
            |fb: &mut FunctionBuilder<'_>| cx.push_field(fb, PushField::DispatchRegion);
        if plan.grid_edge {
            // A grid tile's invocation past the grid returns (under either `MultiRegion`) where
            // its region is flagged ragged: only inside that uniform branch does it decode its
            // position and compare it with the region's last thread. An unflagged region (a
            // grid the workgroups divide, an indirect dispatch's, which states no grid) runs no
            // test, and the body derives its position where it reads it.
            let ragged = decl.region_ragged(fb, &mut region)?;
            let test: ArmFn<'_, ()> = Box::new(|fb, _| {
                let pos = decl.edge_position(fb, &mut region)?;
                let last = decl.edge_last(fb, &mut region)?;
                let below = fb.binary(BinOp::ULessThanEqual, pos, last)?;
                let inside = fb.unary(UnOp::All, below)?;
                let past = fb.unary(UnOp::LogicalNot, inside)?;
                let leave: ArmFn<'_, ()> = Box::new(|fb, _| {
                    fb.ret(None)?;
                    Ok(ArmEnd::Exits)
                });
                fb.if_(&mut (), past, leave, None)?;
                Ok(ArmEnd::FallThrough(Vec::new()))
            });
            fb.if_(&mut (), ragged, test, None)?;
        } else if let Some(live) = decl.live(fb, &mut region)? {
            let walked = cx
                .plan
                .families
                .simd
                .predication
                .as_ref()
                .is_some_and(|p| p.partial_groups);
            if walked {
                cx.entry_live = Some(live);
            } else {
                let past = fb.unary(UnOp::LogicalNot, live)?;
                let leave: ArmFn<'_, ()> = Box::new(|fb, _| {
                    fb.ret(None)?;
                    Ok(ArmEnd::Exits)
                });
                fb.if_(&mut (), past, leave, None)?;
            }
        }
    }
    let entry = cx.entry;
    for p in &plan.params {
        let binding = entry.ep.params.iter().find(|b| b.param == p.param);
        let (Some(b), Some(v)) = (binding, cx.plan.values.id(Value::Arg(ArgIdx::new(p.param))))
        else {
            return Err(cx.fail(format!("dispatch parameter {} is out of range", p.param)));
        };
        let x = param_value(cx, fb, b, tested_position)?;
        cx.hold(v, x);
    }
    Ok(())
}

/// The value of dispatch builtin parameter `b` in its lowered representation, derived in the open
/// block ([`emit_entry`] derives each in the function's first block).
fn param_value(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    b: &ParamBinding,
    tested_position: Option<SValue>,
) -> BResult<SValue> {
    let (plan, decl): (&Plan, &crate::lower::emit::Declared) = (cx.plan, cx.decl);
    let planned = plan
        .dispatch
        .as_ref()
        .and_then(|d| d.params.iter().find(|p| p.param == b.param))
        .copied();
    let decl = decl.families.dispatch.kernel.as_ref();
    let (Some(p), Some(decl)) = (planned, decl) else {
        return Err(cx.fail(format!("parameter {} has no dispatch value", b.param)));
    };
    // A flat grid derives the grid's builtins alone, and of the call's threadgroups the lanes'.
    if decl.flat_grid && !decl.flat_reads(p.builtin) {
        return Err(cx.fail(format!("parameter {} is not a flat grid's", b.param)));
    }
    let raw = match (p.builtin, tested_position) {
        (KernelBuiltin::ThreadPositionInGrid, Some(pos)) if decl.flat_grid || decl.grid_edge => pos,
        _ => decl.value(fb, p.builtin, &mut |fb| {
            cx.push_field(fb, PushField::DispatchRegion)
        })?,
    };
    match project(fb, raw, p.projection)? {
        Some(x) => Ok(x),
        None => Err(cx.fail(format!("parameter {} has no projection", b.param))),
    }
}

/// The kernel's thread position in the grid (a `uvec3`), derived in the open block: what a
/// `[[stage_in]]` attribute stepping by thread position reads, which [`plan_kernel`] declares the
/// builtins of.
pub fn grid_position(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>) -> BResult<SValue> {
    let decl: &crate::lower::emit::Declared = cx.decl;
    let Some(decl) = decl.families.dispatch.kernel.as_ref() else {
        return Err(cx.fail("a grid position outside a kernel"));
    };
    decl.value(fb, KernelBuiltin::ThreadPositionInGrid, &mut |fb| {
        cx.push_field(fb, PushField::DispatchRegion)
    })
}

/// Project a builtin's `u32`/`uvec3` value `v` onto `p`, or `None` when the arith family does not
/// produce `p`'s representation.
pub fn project(fb: &mut FunctionBuilder<'_>, v: SValue, p: Projection) -> BResult<Option<SValue>> {
    let Some(want) = arith::spirv_type(fb.module(), &p.lowered())? else {
        return Ok(None);
    };
    let ty = fb.value_ty(v)?;
    let vector = matches!(fb.module().types().get(ty)?, Type::Vector { .. });
    let (bits, narrowed) = match p {
        Projection::Scalar { bits } if vector => (bits, fb.composite_extract(v, &[0])?),
        Projection::Scalar { bits } => (bits, v),
        Projection::Vector { bits, len } if vector && len < 3 => {
            let comps: Vec<u32> = (0..u32::from(len)).collect();
            (bits, fb.vector_shuffle(v, v, &comps)?)
        }
        Projection::Vector { bits, .. } if vector => (bits, v),
        Projection::Vector { .. } => return Ok(None),
    };
    let x = if bits == 32 {
        narrowed
    } else {
        fb.convert(ConvOp::UConvert, narrowed, want)?
    };
    Ok((fb.value_ty(x)? == want).then_some(x))
}

/// Default values of the kernel's specialization constants. Production modules use
/// [`SpecDefaults::default`] (local size 1, one region); the planner writes the real values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpecDefaults {
    pub local: [u32; 3],
    pub multi_region: bool,
}

impl Default for SpecDefaults {
    fn default() -> SpecDefaults {
        SpecDefaults {
            local: [1, 1, 1],
            multi_region: false,
        }
    }
}

/// The kernel's specialization constants and the Vulkan builtin variables its builtins read.
#[derive(Clone, Debug)]
pub struct KernelDecl {
    pub form: LocalSizeForm,
    /// A tile function's thread layout ([`KernelDecl::tile_pos`]).
    tile: bool,
    /// The tile function's threadgroup is the tile ([`DispatchPlan::tile_sized`]).
    tile_sized: bool,
    /// The kernel runs its grid flat ([`KernelDecl::flat`]).
    flat_grid: bool,
    /// Of the call's threadgroups ([`DispatchPlan::call_threadgroups`]).
    call_threadgroups: bool,
    /// The kernel folds Metal's group counts into its dispatch ([`KernelDecl::folded`]).
    folded_groups: bool,
    /// A folded object function's `DrawIndex`, its draw's piece ([`DispatchPlan::draw_pieces`]).
    draw_index: Option<GlobalId>,
    uvec3: TypeId,
    /// `sx, sy, sz`.
    size: [ConstId; 3],
    /// The `WorkgroupSize` composite.
    workgroup: ConstId,
    /// `SpecKey::DeepThreadgroup`, where the plan declares it ([`DispatchPlan::deep`]).
    deep: Option<ConstId>,
    /// How the kernel tells a region, when it reads a grid builtin.
    multi_region: Option<RegionSelect>,
    local_id: Option<GlobalId>,
    local_index: Option<GlobalId>,
    workgroup_id: Option<GlobalId>,
    num_workgroups: Option<GlobalId>,
    subgroup_id: Option<GlobalId>,
    /// `SubgroupSize`, declared only where the stage pins no size ([`Self::pinned_subgroup_size`]).
    subgroup_size: Option<GlobalId>,
    subgroup_local: Option<GlobalId>,
    /// The subgroup size the pipeline pins ([`DispatchPlan::pinned_subgroup_size`]).
    pinned_subgroup_size: Option<u32>,
    /// An unfolded object function's grid ([`DispatchPlan::object_grid`]).
    object_grid: bool,
    /// A function whose threadgroup may be partial ([`DispatchPlan::partial_groups`]).
    partial_groups: bool,
    /// Its threadgroups run whole, its invocations past the grid return
    /// ([`DispatchPlan::grid_edge`]).
    grid_edge: bool,
}

/// Declare the kernel's workgroup size (module scope, before the entry is declared), its
/// specialization constants, and its builtin variables. A non-kernel plan declares nothing.
pub fn declare(
    mb: &mut ModuleBuilder,
    plan: &Plan,
    _modes: &mut Vec<ExecMode>,
) -> BResult<Declared> {
    let Some(dp) = &plan.dispatch else {
        return Ok(Declared::default());
    };
    Ok(Declared {
        kernel: Some(KernelDecl::declare(mb, dp, &SpecDefaults::default())?),
    })
}

/// Declare `ThreadgroupMemoryLength{index}`'s specialization constant (`SpecKey::TgLength`, a
/// `u32` byte length with default `default`) and record the fact: the length of a threadgroup
/// argument is set by `setThreadgroupMemoryLength` and reaches the module by specialization. The
/// arena lowering sizes the argument's array from it.
pub fn threadgroup_length(mb: &mut ModuleBuilder, index: u32, default: u32) -> BResult<ConstId> {
    let u32_ = mb.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    mb.spec(
        SpecKey::TgLength(index),
        u32_,
        ConstBits(u64::from(default)),
    )
}

/// Record that the function depends on the length of threadgroup argument `index`.
pub fn note_threadgroup_length(cx: &mut SelectCx<'_>, index: u32) {
    cx.facts
        .insert(RuntimeFact::ThreadgroupMemoryLength { index });
}

/// What the kernel's builtins read.
struct Needs {
    lin: bool,
    local_pos: bool,
    workgroup_id: bool,
    num_workgroups: bool,
}

impl Needs {
    fn of(dp: &DispatchPlan) -> Needs {
        use KernelBuiltin as K;
        // A flat grid counts its threads from its threadgroup's position and its own.
        if dp.flat_grid {
            return Needs {
                lin: true,
                local_pos: false,
                workgroup_id: true,
                num_workgroups: true,
            };
        }
        let has = |b| dp.builtins.contains(b);
        // A grid edge tests every invocation's position in the grid, whatever builtins it reads.
        let local_pos =
            has(K::ThreadPositionInThreadgroup) || has(K::ThreadPositionInGrid) || dp.grid_edge;
        // A partial object threadgroup counts its threads from its position and its own `lin`.
        let partial = dp.partial_groups;
        // A folded variant counts its threadgroup from its workgroup's position and the
        // dispatch's size, whatever builtins it reads.
        let folded = dp.folded_groups;
        // A tile function derives its position from `lin` in every tier.
        let lin = [
            K::ThreadIndexInThreadgroup,
            K::ThreadIndexInSimdgroup,
            K::SimdgroupIndexInThreadgroup,
            K::ThreadIndexInQuadgroup,
            K::QuadgroupIndexInThreadgroup,
        ]
        .into_iter()
        .any(has)
            || (local_pos && (dp.local_size == LocalSizeForm::Flat || dp.tile))
            || partial;
        // A private arena in device memory counts its slot from the workgroup's position.
        let arena = dp.private_arena;
        Needs {
            lin,
            local_pos,
            workgroup_id: folded
                || partial
                || arena
                || has(K::ThreadPositionInGrid)
                || has(K::ThreadgroupPositionInGrid),
            num_workgroups: folded
                || arena
                || has(K::ThreadsPerGrid)
                || has(K::ThreadgroupsPerGrid),
        }
    }
}

/// The nominal local size packing of `region[0].w`: 10 bits per axis, each `L − 1`.
pub const NOMINAL_AXIS_BITS: u32 = 10;

/// `region[1].w`'s flag of a draw or dispatch whose last threadgroup along an axis runs only the
/// threads left (`drawMeshThreads` with an object function, or a kernel's ragged
/// `dispatchThreads` on its partial-threadgroups variant, the module docs): a function of
/// `DispatchPlan::partial_groups` answers that threadgroup's threads.
pub const NONUNIFORM: u32 = 1;

/// Pack Metal's nominal threads per threadgroup `l` into `region[0].w` (each axis 1..=1024).
pub fn pack_nominal(l: [u32; 3]) -> Option<u32> {
    let mut out = 0u32;
    for (i, n) in l.into_iter().enumerate() {
        if n == 0 || n > 1 << NOMINAL_AXIS_BITS {
            return None;
        }
        out |= (n - 1) << (NOMINAL_AXIS_BITS * i as u32);
    }
    Some(out)
}

/// The threads of a grid tile ([`grid_tile`]): the workgroup of a kernel's grid tiles
/// (`RuntimeFact::GridTiles`).
pub const GRID_TILE_THREADS: u32 = 256;

/// The deepest threadgroup a kernel runs as its own workgroup on `Device::from_queries`' profile:
/// one deeper runs as one row (`SpecKey::DeepThreadgroup`). The least `maxComputeWorkGroupSize[2]`
/// of the devices that profile serves (Vulkan's own minimum).
pub const DEEP_THREADGROUP: u32 = 64;

/// The tile a kernel's grid tiles (`RuntimeFact::GridTiles`) cover a grid of `threads` with: sides
/// `2^a, 2^b, 2^c` with `a + b + c = 8`, so that one workgroup of [`GRID_TILE_THREADS`] is one
/// tile, dispatched as `⌈threads / tile⌉` workgroups per axis. Of every such tile it is the first
/// in one total order: the fewest threads launched, `∏ ⌈threads_i / side_i⌉ · side_i`; then the
/// fewest tiles partly past the grid; then the shallowest (`z` side 1 first); then, where the grid
/// has two rows or more, the height nearest 2 (`|b - 1|` smallest); then the widest. A pure
/// function of the grid's extents.
///
/// A side that divides its axis's extent launches nothing past it, so where the powers of two
/// that divide the extents reach 256 between them, the tile is the first of those that divide the
/// grid. A grid that one tile holds takes the first tile that holds it, each such tile being one
/// workgroup partly past the grid. Any other grid tries the 45 tiles in the order, passing over
/// the depths that cannot launch as few. A zero extent is divided by every power of two; such a
/// grid runs nothing.
pub fn grid_tile(threads: [u32; 3]) -> [u32; 3] {
    const BITS: u32 = GRID_TILE_THREADS.trailing_zeros();
    let rows = threads[1] >= 2;
    // The height nearest 2 of `lo..=hi` where the grid has rows, else the lowest (the widest).
    let height = |lo: u32, hi: u32| if rows { 1.clamp(lo, hi) } else { lo };
    let [x, y, z] = threads.map(|n| n.trailing_zeros().min(BITS));
    if x + y + z >= BITS {
        let c = BITS.saturating_sub(x + y);
        let b = height(BITS - c - x.min(BITS - c), y.min(BITS - c));
        return [1 << (BITS - c - b), 1 << b, 1 << c];
    }
    let [x, y, z] = threads.map(|n| u32::BITS - n.saturating_sub(1).leading_zeros());
    if x + y + z <= BITS {
        let b = height(y, BITS - z - x);
        return [1 << (BITS - z - b), 1 << b, 1 << z];
    }
    // Every tile in the order: a later tile wins only by launching fewer threads, or as many with
    // fewer tiles partly past the grid. The tiles `2^c` deep launch at least
    // `⌈z / 2^c⌉ · ⌈x·y / 2^(8 - c)⌉` tiles, so a depth whose bound exceeds the fewest yet is
    // passed over.
    let counts = |n: u32, k: u32| (u64::from(n.div_ceil(1 << k)), u64::from(n >> k));
    let xy = u64::from(threads[0]) * u64::from(threads[1]);
    let (mut best, mut fewest) = ([0; 3], (u128::MAX, u128::MAX));
    for c in 0..=BITS {
        let (tz, wz) = counts(threads[2], c);
        let rest = BITS - c;
        if u128::from(tz) * u128::from(xy.div_ceil(1 << rest)) > fewest.0 {
            continue;
        }
        // Heights 1, 0, 2, 3, ... where the grid has rows (nearest 2 first, the wider of two
        // as near), else 0, 1, 2, ... (the widest first).
        for step in 0..=rest {
            let b = match step {
                0 | 1 if rows && rest > 0 => 1 - step,
                _ => step,
            };
            let ((tx, wx), (ty, wy)) = (counts(threads[0], rest - b), counts(threads[1], b));
            let tiles = u128::from(tx * ty) * u128::from(tz);
            if tiles > fewest.0 {
                continue;
            }
            let partly = tiles - u128::from(wx * wy) * u128::from(wz);
            if (tiles, partly) < fewest {
                (best, fewest) = ([rest - b, b, c], (tiles, partly));
            }
        }
    }
    best.map(|k| 1 << k)
}

/// The `DispatchRegion` push field's bytes (two `uvec4`) for a region whose first threadgroup is
/// `base`, of a Metal grid of `threads` threads with nominal threadgroup size `l`.
pub fn region_bytes(base: [u32; 3], threads: [u32; 3], l: [u32; 3]) -> Option<[u8; 32]> {
    let nominal = pack_nominal(l)?;
    let words = [
        base[0], base[1], base[2], nominal, threads[0], threads[1], threads[2], 0,
    ];
    let mut out = [0u8; 32];
    for (chunk, w) in out.as_chunks_mut::<4>().0.iter_mut().zip(words) {
        chunk.copy_from_slice(&w.to_le_bytes());
    }
    Some(out)
}

/// The `DispatchRegion` bytes of a draw of the object threadgroups `threads` fill at nominal
/// size `l`, the last along an axis running only the threads left ([`NONUNIFORM`]).
pub fn partial_region_bytes(threads: [u32; 3], l: [u32; 3]) -> Option<[u8; 32]> {
    nonuniform_region_bytes([0; 3], threads, l)
}

/// [`region_bytes`] of a region flagged [`NONUNIFORM`]: its threadgroups the ones `threads` fill
/// at nominal size `l` from `base`, the last along an axis running only the threads left.
pub fn nonuniform_region_bytes(base: [u32; 3], threads: [u32; 3], l: [u32; 3]) -> Option<[u8; 32]> {
    let mut out = region_bytes(base, threads, l)?;
    out[28..].copy_from_slice(&NONUNIFORM.to_le_bytes());
    Some(out)
}

/// A source of the `DispatchRegion` push field's value (`array<uvec4, 2>`).
pub type RegionFn<'a> = dyn FnMut(&mut FunctionBuilder<'_>) -> BResult<SValue> + 'a;

impl KernelDecl {
    /// Declare the kernel's specialization constants (`sx, sy, sz`, and `sn` or `MultiRegion`
    /// when the plan needs them), its `WorkgroupSize`, and the Vulkan builtins its builtins read.
    pub fn declare(
        mb: &mut ModuleBuilder,
        dp: &DispatchPlan,
        defaults: &SpecDefaults,
    ) -> BResult<KernelDecl> {
        let u32_ = mb.ty(Type::Int {
            width: 32,
            signed: false,
        })?;
        let uvec3 = mb.ty(Type::Vector { elem: u32_, len: 3 })?;
        let [dx, dy, dz] = defaults.local;
        let keys = dp.keys;
        let [sx, sy, sz] = [
            (keys.local[0], dx),
            (keys.local[1], dy),
            (keys.local[2], dz),
        ]
        .map(|(k, d)| mb.spec(k, u32_, ConstBits(u64::from(d))));
        let size = [sx?, sy?, sz?];
        let mut deep = None;
        let workgroup = match dp.local_size {
            LocalSizeForm::Spec if dp.deep => {
                let b = mb.ty(Type::Bool)?;
                let one_row = mb.spec(SpecKey::DeepThreadgroup, b, ConstBits(0))?;
                let row = mb.spec_op(SpecOp::IMul, u32_, &[size[0], size[1]])?;
                let sn = mb.spec_op(SpecOp::IMul, u32_, &[row, size[2]])?;
                let one = mb.constant(Constant::Scalar {
                    ty: u32_,
                    bits: ConstBits(1),
                })?;
                let [x, y, z] = [(sn, size[0]), (one, size[1]), (one, size[2])]
                    .map(|(deep, own)| mb.spec_op(SpecOp::Select, u32_, &[one_row, deep, own]));
                deep = Some(one_row);
                mb.workgroup_size(x?, y?, z?)?
            }
            LocalSizeForm::Spec => mb.workgroup_size(size[0], size[1], size[2])?,
            LocalSizeForm::Flat => {
                let n = u64::from(dx) * u64::from(dy) * u64::from(dz);
                let sn = mb.spec(keys.flat, u32_, ConstBits(n))?;
                let one = mb.constant(Constant::Scalar {
                    ty: u32_,
                    bits: ConstBits(1),
                })?;
                mb.workgroup_size(sn, one, one)?
            }
        };
        let multi_region = match (dp.multi_region, dp.identity) {
            (false, _) => None,
            (true, identity) => {
                let b = mb.ty(Type::Bool)?;
                let mr = mb.spec(
                    keys.multi_region,
                    b,
                    ConstBits(u64::from(defaults.multi_region)),
                )?;
                Some(match identity {
                    true => RegionSelect::Sized(mr),
                    false => RegionSelect::Spec(mr),
                })
            }
        };
        let needs = Needs::of(dp);
        let flat = dp.local_size == LocalSizeForm::Flat;
        // A flat grid counts its threads by `LocalInvocationIndex` in either form: it reads no
        // simdgroup. One of the call's threadgroups counts them by `lin`, its simdgroups' lanes.
        let by_index = dp.flat_grid && !dp.call_threadgroups;
        // A grid tile counts by `LocalInvocationIndex` in either form too, and a deep
        // threadgroup's position is unflattened from it.
        let index = (needs.lin && (!flat || by_index || dp.grid_edge))
            || (needs.local_pos && deep.is_some())
            || dp.private_arena;
        let subgroups = needs.lin && flat && !by_index && !dp.grid_edge;
        use crate::spv::BuiltIn as B;
        let mut var = |on: bool, b: crate::spv::BuiltIn, ty: TypeId| -> BResult<Option<GlobalId>> {
            if on {
                mb.builtin(b, IoDir::Input, ty).map(Some)
            } else {
                Ok(None)
            }
        };
        let (workgroup_id, num_workgroups) = (
            var(needs.workgroup_id, B::WorkgroupId, uvec3)?,
            var(needs.num_workgroups, B::NumWorkgroups, uvec3)?,
        );
        Ok(KernelDecl {
            form: dp.local_size,
            tile: dp.tile,
            tile_sized: dp.tile_sized,
            flat_grid: dp.flat_grid,
            call_threadgroups: dp.call_threadgroups,
            folded_groups: dp.folded_groups,
            draw_index: var(dp.draw_pieces, B::DrawIndex, u32_)?,
            uvec3,
            size,
            workgroup,
            deep,
            multi_region,
            local_id: var(
                needs.local_pos && !flat && !dp.tile,
                B::LocalInvocationId,
                uvec3,
            )?,
            local_index: var(index, B::LocalInvocationIndex, u32_)?,
            workgroup_id,
            num_workgroups,
            subgroup_id: var(subgroups, B::SubgroupId, u32_)?,
            subgroup_size: var(
                subgroups && dp.pinned_subgroup_size.is_none(),
                B::SubgroupSize,
                u32_,
            )?,
            subgroup_local: var(subgroups, B::SubgroupLocalInvocationId, u32_)?,
            pinned_subgroup_size: dp.pinned_subgroup_size,
            object_grid: dp.object_grid,
            partial_groups: dp.partial_groups,
            grid_edge: dp.grid_edge,
        })
    }

    /// The device address (a `u64`) of the invocation's slot of `slot_bytes` in the private
    /// arena at `arena` (`PushField::PrivateArenaAddress`): `arena + i · slot_bytes` for its index
    /// in its dispatch `i = ((wz · ny + wy) · nx + wx) · n + LocalInvocationIndex`, `n` the
    /// workgroup's invocations. The runtime keeps a dispatch's invocations under 2^32, so `i` is
    /// counted in 32 bits.
    pub(crate) fn arena_slot(
        &self,
        fb: &mut FunctionBuilder<'_>,
        arena: SValue,
        slot_bytes: u32,
    ) -> BResult<SValue> {
        let wg = self.load(fb, self.workgroup_id)?;
        let groups = self.load(fb, self.num_workgroups)?;
        let index = self.load(fb, self.local_index)?;
        let size = fb.module().const_value(self.workgroup)?;
        let n = Self::volume(fb, size)?;
        let [wx, wy, wz] = [0, 1, 2].map(|i| fb.composite_extract(wg, &[i]));
        let [nx, ny] = [0, 1].map(|i| fb.composite_extract(groups, &[i]));
        let group = fb.binary(BinOp::IMul, wz?, ny?)?;
        let group = fb.binary(BinOp::IAdd, group, wy?)?;
        let group = fb.binary(BinOp::IMul, group, nx?)?;
        let group = fb.binary(BinOp::IAdd, group, wx?)?;
        let first = fb.binary(BinOp::IMul, group, n)?;
        let at = fb.binary(BinOp::IAdd, first, index)?;
        let u64_ = fb.module().ty(Type::Int {
            width: 64,
            signed: false,
        })?;
        let at = fb.convert(ConvOp::UConvert, at, u64_)?;
        let bytes = fb.module().constant(Constant::Scalar {
            ty: u64_,
            bits: ConstBits(u64::from(slot_bytes)),
        })?;
        let bytes = fb.module().const_value(bytes)?;
        let offset = fb.binary(BinOp::IMul, at, bytes)?;
        fb.binary(BinOp::IAdd, arena, offset)
    }

    fn load(&self, fb: &mut FunctionBuilder<'_>, g: Option<GlobalId>) -> BResult<SValue> {
        let Some(g) = g else {
            return Err(dangling_value());
        };
        let p = fb.module().global_value(g)?;
        fb.load(p, MemAccess::default())
    }

    /// Whether a simdgroup of the threadgroup may span two of its rows, so a lockstep barrier of
    /// pairs that meet only across rows must run (`air::Lockstep::across_rows`): the threadgroup
    /// has more than one row (`sy` or `sz` is not 1) and its rows are not whole simdgroups (`sx`
    /// is not a multiple of 32). A simdgroup is 32 consecutive linear threads from a multiple of
    /// 32, so otherwise each lies within one row. A test of specialization constants alone.
    /// `None` where the threads' positions are not `lin` over `(sx, sy, sz)`: a tile function's
    /// layout, partial threadgroups, a flat, folded or object grid, or the call's threadgroups.
    pub fn rows_may_split(&self, fb: &mut FunctionBuilder<'_>) -> BResult<Option<SValue>> {
        if self.tile
            || self.partial_groups
            || self.flat_grid
            || self.call_threadgroups
            || self.folded_groups
            || self.object_grid
        {
            return Ok(None);
        }
        let [sx, sy, sz] = self.size;
        let sx = fb.module().const_value(sx)?;
        let sy = fb.module().const_value(sy)?;
        let sz = fb.module().const_value(sz)?;
        let one = Self::k(fb, 1)?;
        let tall = fb.binary(BinOp::INotEqual, sy, one)?;
        let deep = fb.binary(BinOp::INotEqual, sz, one)?;
        let rows = fb.binary(BinOp::LogicalOr, tall, deep)?;
        let low = Self::k(fb, 31)?;
        let lanes = fb.binary(BinOp::BitwiseAnd, sx, low)?;
        let zero = Self::k(fb, 0)?;
        let ragged = fb.binary(BinOp::INotEqual, lanes, zero)?;
        fb.binary(BinOp::LogicalAnd, rows, ragged).map(Some)
    }

    fn k(fb: &mut FunctionBuilder<'_>, n: u32) -> BResult<SValue> {
        let c = fb.u32_const(n)?;
        fb.module().const_value(c)
    }

    fn splat(&self, fb: &mut FunctionBuilder<'_>, n: u32) -> BResult<SValue> {
        let x = Self::k(fb, n)?;
        fb.composite_construct(self.uvec3, &[x, x, x])
    }

    /// `(sx, sy, sz)`.
    fn size3(&self, fb: &mut FunctionBuilder<'_>) -> BResult<SValue> {
        let [x, y, z] = self.size;
        let parts = [
            fb.module().const_value(x)?,
            fb.module().const_value(y)?,
            fb.module().const_value(z)?,
        ];
        fb.composite_construct(self.uvec3, &parts)
    }

    /// `x · y · z` of a `uvec3`.
    fn volume(fb: &mut FunctionBuilder<'_>, v: SValue) -> BResult<SValue> {
        let x = fb.composite_extract(v, &[0])?;
        let y = fb.composite_extract(v, &[1])?;
        let z = fb.composite_extract(v, &[2])?;
        let xy = fb.binary(BinOp::IMul, x, y)?;
        fb.binary(BinOp::IMul, xy, z)
    }

    /// `⌈n / d⌉` for `d > 0`, without overflow.
    fn ceil_div(&self, fb: &mut FunctionBuilder<'_>, n: SValue, d: SValue) -> BResult<SValue> {
        let vector = fb.value_ty(n)? == self.uvec3;
        let q = fb.binary(BinOp::UDiv, n, d)?;
        let r = fb.binary(BinOp::UMod, n, d)?;
        let (zero, one) = if vector {
            (self.splat(fb, 0)?, self.splat(fb, 1)?)
        } else {
            (Self::k(fb, 0)?, Self::k(fb, 1)?)
        };
        let partial = fb.binary(BinOp::INotEqual, r, zero)?;
        let extra = fb.select(partial, one, zero)?;
        fb.binary(BinOp::IAdd, q, extra)
    }

    /// The linear local index `lin`; a grid tile's is `LocalInvocationIndex` in either form (its
    /// kernel reads no simdgroup).
    fn lin(&self, fb: &mut FunctionBuilder<'_>) -> BResult<SValue> {
        if self.grid_edge {
            return self.load(fb, self.local_index);
        }
        match self.form {
            LocalSizeForm::Spec => self.load(fb, self.local_index),
            LocalSizeForm::Flat => {
                let sg = self.load(fb, self.subgroup_id)?;
                // The size the pipeline pins, a constant: where a size is pinned the module never
                // reads `SubgroupSize`.
                let size = match self.pinned_subgroup_size {
                    Some(size) => Self::k(fb, size)?,
                    None => self.load(fb, self.subgroup_size)?,
                };
                let lane = self.load(fb, self.subgroup_local)?;
                let base = fb.binary(BinOp::IMul, sg, size)?;
                fb.binary(BinOp::IAdd, base, lane)
            }
        }
    }

    /// The position in the threadgroup: a tile function's layout of `lin` ([`Self::tile_pos`]),
    /// `LocalInvocationId`, or `lin` unflattened over `(sx, sy, sz)`; in a partial object
    /// threadgroup, `lin` unflattened over the threads it runs ([`Self::partial_threads`]).
    fn local_pos(
        &self,
        fb: &mut FunctionBuilder<'_>,
        region: &mut RegionFn<'_>,
    ) -> BResult<SValue> {
        if self.tile {
            let lin = self.lin(fb)?;
            return self.tile_pos(fb, lin);
        }
        let uniform = match self.form {
            LocalSizeForm::Spec => {
                let id = self.load(fb, self.local_id)?;
                match self.deep {
                    // A deep threadgroup's row, unflattened as the flattened tiers' is; any
                    // other's own position, which the driver keeps alone when it specializes.
                    Some(deep) => {
                        let index = self.load(fb, self.local_index)?;
                        let sx = fb.module().const_value(self.size[0])?;
                        let sy = fb.module().const_value(self.size[1])?;
                        let x = fb.binary(BinOp::UMod, index, sx)?;
                        let row = fb.binary(BinOp::UDiv, index, sx)?;
                        let y = fb.binary(BinOp::UMod, row, sy)?;
                        let z = fb.binary(BinOp::UDiv, row, sy)?;
                        let at = fb.composite_construct(self.uvec3, &[x, y, z])?;
                        let deep = fb.module().const_value(deep)?;
                        fb.select(deep, at, id)?
                    }
                    None => id,
                }
            }
            LocalSizeForm::Flat => {
                let lin = self.lin(fb)?;
                let sx = fb.module().const_value(self.size[0])?;
                let sy = fb.module().const_value(self.size[1])?;
                let x = fb.binary(BinOp::UMod, lin, sx)?;
                let row = fb.binary(BinOp::UDiv, lin, sx)?;
                let y = fb.binary(BinOp::UMod, row, sy)?;
                let z = fb.binary(BinOp::UDiv, row, sy)?;
                fb.composite_construct(self.uvec3, &[x, y, z])?
            }
        };
        let Some((partial, threads)) = self.partial_threads(fb, region)? else {
            return Ok(uniform);
        };
        let lin = self.lin(fb)?;
        let at = self.unflatten(fb, lin, threads)?;
        fb.select(partial, at, uniform)
    }

    /// `lin` unflattened over a threadgroup of `size` threads, `x` fastest; an axis of no thread
    /// divides as one of 1, so no value is undefined.
    fn unflatten(
        &self,
        fb: &mut FunctionBuilder<'_>,
        lin: SValue,
        size: SValue,
    ) -> BResult<SValue> {
        let one = Self::k(fb, 1)?;
        let axis = |fb: &mut FunctionBuilder<'_>, i: u32| -> BResult<SValue> {
            let n = fb.composite_extract(size, &[i])?;
            fb.glsl(GlslOp::UMax, &[n, one])
        };
        let (sx, sy) = (axis(fb, 0)?, axis(fb, 1)?);
        let x = fb.binary(BinOp::UMod, lin, sx)?;
        let row = fb.binary(BinOp::UDiv, lin, sx)?;
        let y = fb.binary(BinOp::UMod, row, sy)?;
        let z = fb.binary(BinOp::UDiv, row, sy)?;
        fb.composite_construct(self.uvec3, &[x, y, z])
    }

    /// Whether the region says the draw's threadgroups may be partial ([`NONUNIFORM`] under
    /// `MultiRegion`), and the threads this threadgroup runs: `min(L, threads_per_grid − position ·
    /// L)` per axis where it does (0 past the grid), `(sx, sy, sz)` where it does not. `None` but
    /// for a function of [`DispatchPlan::partial_groups`] whose threadgroups run whole
    /// ([`DispatchPlan::grid_edge`]).
    fn partial_threads(
        &self,
        fb: &mut FunctionBuilder<'_>,
        region: &mut RegionFn<'_>,
    ) -> BResult<Option<(SValue, SValue)>> {
        let (true, false, Some(sel)) = (self.partial_groups, self.grid_edge, self.multi_region)
        else {
            return Ok(None);
        };
        let mr = self.selected(fb, sel, region)?;
        let r = region(fb)?;
        let r1 = fb.composite_extract(r, &[1])?;
        let flags = fb.composite_extract(r1, &[3])?;
        let bit = Self::k(fb, NONUNIFORM)?;
        let flag = fb.binary(BinOp::BitwiseAnd, flags, bit)?;
        let zero = Self::k(fb, 0)?;
        let flagged = fb.binary(BinOp::INotEqual, flag, zero)?;
        let no = Self::flag(fb, false)?;
        let partial = fb.select(mr, flagged, no)?;
        // The threadgroup's position and `L`: a folded variant's from its count (its local size
        // is `L`), every other's from its region.
        let (group, l) = if self.folded_groups {
            (self.folded(fb, region)?.0, self.size3(fb)?)
        } else {
            let (base, l) = self.region(fb, region)?;
            (self.group_pos(fb, base)?, l)
        };
        let threads = fb.vector_shuffle(r1, r1, &[0, 1, 2])?;
        let first = fb.binary(BinOp::IMul, group, l)?;
        let inside = fb.binary(BinOp::ULessThan, first, threads)?;
        let left = fb.binary(BinOp::ISub, threads, first)?;
        let none = self.splat(fb, 0)?;
        let left = fb.select(inside, left, none)?;
        let runs = fb.glsl(GlslOp::UMin, &[left, l])?;
        let s = self.size3(fb)?;
        Ok(Some((partial, fb.select(partial, runs, s)?)))
    }

    /// Whether the invocation is one of its threadgroup's threads: `lin` below the threads a
    /// partial object threadgroup runs ([`Self::partial_threads`]), every invocation elsewhere.
    /// `None` but for an object function of [`DispatchPlan::partial_groups`].
    pub fn live(
        &self,
        fb: &mut FunctionBuilder<'_>,
        region: &mut RegionFn<'_>,
    ) -> BResult<Option<SValue>> {
        let Some((partial, threads)) = self.partial_threads(fb, region)? else {
            return Ok(None);
        };
        let n = Self::volume(fb, threads)?;
        let lin = self.lin(fb)?;
        let below = fb.binary(BinOp::ULessThan, lin, n)?;
        let yes = Self::flag(fb, true)?;
        Ok(Some(fb.select(partial, below, yes)?))
    }

    /// Of [`DispatchPlan::grid_edge`], the invocation's position in the grid (a `uvec3`). Under
    /// `MultiRegion` 1 its workgroup is one tile of the grid whose sides the region's nominal size
    /// `L` gives, each a power of two whose product is the workgroup's threads: with `m = L − 1`
    /// and `s` the bits set in `m` per axis, the position is `((base + WorkgroupId) << s) + (lin &
    /// m.x, (lin >> s.x) & m.y, lin >> (s.x + s.y))`, masks and shifts by uniform values, where
    /// `lin` is `LocalInvocationIndex`. Under `MultiRegion` 0 it is the uniform one at the
    /// pipeline's size, `(base + WorkgroupId) · (sx, sy, sz)` plus the position in the workgroup,
    /// which the driver keeps alone when it specializes either.
    pub fn edge_position(
        &self,
        fb: &mut FunctionBuilder<'_>,
        region: &mut RegionFn<'_>,
    ) -> BResult<SValue> {
        let Some(RegionSelect::Sized(sized)) = self.multi_region else {
            return Err(dangling_value());
        };
        let r = region(fb)?;
        let r0 = fb.composite_extract(r, &[0])?;
        let base = fb.vector_shuffle(r0, r0, &[0, 1, 2])?;
        let packed = fb.composite_extract(r0, &[3])?;
        let axis = Self::k(fb, (1 << NOMINAL_AXIS_BITS) - 1)?;
        let mask_x = fb.binary(BinOp::BitwiseAnd, packed, axis)?;
        let mut above = [mask_x; 2];
        for (i, mask) in (1..).zip(&mut above) {
            let sh = Self::k(fb, NOMINAL_AXIS_BITS * i)?;
            let bits = fb.binary(BinOp::ShiftRightLogical, packed, sh)?;
            *mask = fb.binary(BinOp::BitwiseAnd, bits, axis)?;
        }
        let [mask_y, mask_z] = above;
        let m = fb.composite_construct(self.uvec3, &[mask_x, mask_y, mask_z])?;
        let s = fb.unary(UnOp::BitCount, m)?;
        let sx = fb.composite_extract(s, &[0])?;
        let sy = fb.composite_extract(s, &[1])?;
        let lin = self.lin(fb)?;
        let x = fb.binary(BinOp::BitwiseAnd, lin, mask_x)?;
        let row = fb.binary(BinOp::ShiftRightLogical, lin, sx)?;
        let y = fb.binary(BinOp::BitwiseAnd, row, mask_y)?;
        let z = fb.binary(BinOp::ShiftRightLogical, row, sy)?;
        let local = fb.composite_construct(self.uvec3, &[x, y, z])?;
        let group = self.group_pos(fb, Some(base))?;
        let first = fb.binary(BinOp::ShiftLeftLogical, group, s)?;
        let tiled = fb.binary(BinOp::IAdd, first, local)?;
        let size = self.size3(fb)?;
        let first = fb.binary(BinOp::IMul, group, size)?;
        let at = self.local_pos(fb, region)?;
        let uniform = fb.binary(BinOp::IAdd, first, at)?;
        let sized = fb.module().const_value(sized)?;
        fb.select(sized, tiled, uniform)
    }

    /// Of [`DispatchPlan::grid_edge`], whether the region is flagged [`NONUNIFORM`]: a runtime
    /// flags the regions of a grid its workgroups do not divide (a workgroup-uniform `bool`).
    pub fn region_ragged(
        &self,
        fb: &mut FunctionBuilder<'_>,
        region: &mut RegionFn<'_>,
    ) -> BResult<SValue> {
        let Some(RegionSelect::Sized(_)) = self.multi_region else {
            return Err(dangling_value());
        };
        let r = region(fb)?;
        let r1 = fb.composite_extract(r, &[1])?;
        let flags = fb.composite_extract(r1, &[3])?;
        let bit = Self::k(fb, NONUNIFORM)?;
        let flag = fb.binary(BinOp::BitwiseAnd, flags, bit)?;
        let zero = Self::k(fb, 0)?;
        fb.binary(BinOp::INotEqual, flag, zero)
    }

    /// Of [`DispatchPlan::grid_edge`], the position of the grid's last thread: the region's grid
    /// `region[1].xyz` less one on each axis, wrapping, so an indirect dispatch's region, whose
    /// grid is 0, gives `u32::MAX` on each axis, which every position is within (a
    /// workgroup-uniform `uvec3`).
    pub fn edge_last(
        &self,
        fb: &mut FunctionBuilder<'_>,
        region: &mut RegionFn<'_>,
    ) -> BResult<SValue> {
        let Some(RegionSelect::Sized(_)) = self.multi_region else {
            return Err(dangling_value());
        };
        let r = region(fb)?;
        let r1 = fb.composite_extract(r, &[1])?;
        let grid = fb.vector_shuffle(r1, r1, &[0, 1, 2])?;
        let one = Self::k(fb, 1)?;
        let ones = fb.composite_construct(self.uvec3, &[one, one, one])?;
        fb.binary(BinOp::ISub, grid, ones)
    }

    /// A `bool` constant.
    fn flag(fb: &mut FunctionBuilder<'_>, v: bool) -> BResult<SValue> {
        let ty = fb.module().ty(Type::Bool)?;
        let c = fb.module().constant(Constant::Scalar {
            ty,
            bits: ConstBits(u64::from(v)),
        })?;
        fb.module().const_value(c)
    }

    /// A tile function's position of the thread whose `sg · 32 + lane` is `lin`, in a threadgroup
    /// `sx` by `sy` of `qw = ⌈sx / 2⌉` by `qh = ⌈sy / 2⌉` 2x2 quads: lane `lin mod 4` is thread
    /// `lin mod 4` (row-major) of the threadgroup's quad `k = lin / 4` in Metal's quad order. That
    /// order runs over columns 8 quads wide (the last narrower), within a column over its rows of
    /// 2 quads (the last 1 where `qh` is odd), within such a row over its blocks 4 quads wide (the
    /// last narrower), and within a block over its quads row-major: so a full column is `8 · qh`
    /// quads, its row `2 · cw` for a column `cw` wide, and its block `4 · rh` for a row `rh` high.
    /// Every divisor is at least 1 for any local size, so the shapes the binder refuses never divide
    /// by zero.
    fn tile_pos(&self, fb: &mut FunctionBuilder<'_>, lin: SValue) -> BResult<SValue> {
        let bin = |fb: &mut FunctionBuilder<'_>, op, a, n: u32| -> BResult<SValue> {
            let n = Self::k(fb, n)?;
            fb.binary(op, a, n)
        };
        let at_most = |fb: &mut FunctionBuilder<'_>, a, n: u32| -> BResult<SValue> {
            let n = Self::k(fb, n)?;
            let below = fb.binary(BinOp::ULessThan, a, n)?;
            fb.select(below, a, n)
        };
        let at_least_one = |fb: &mut FunctionBuilder<'_>, a| -> BResult<SValue> {
            let zero = Self::k(fb, 0)?;
            let one = Self::k(fb, 1)?;
            let none = fb.binary(BinOp::IEqual, a, zero)?;
            fb.select(none, one, a)
        };
        // `n - taken`, at most `most` and at least 1: the extent left of a column, row or block.
        let left = |fb: &mut FunctionBuilder<'_>, n, taken, most: u32| -> BResult<SValue> {
            let below = fb.binary(BinOp::ULessThan, taken, n)?;
            let rest = fb.binary(BinOp::ISub, n, taken)?;
            let zero = Self::k(fb, 0)?;
            let rest = fb.select(below, rest, zero)?;
            let rest = at_most(fb, rest, most)?;
            at_least_one(fb, rest)
        };
        let sx = fb.module().const_value(self.size[0])?;
        let sy = fb.module().const_value(self.size[1])?;
        let wide = bin(fb, BinOp::IAdd, sx, 1)?;
        let qw = bin(fb, BinOp::ShiftRightLogical, wide, 1)?;
        let tall = bin(fb, BinOp::IAdd, sy, 1)?;
        let qh = bin(fb, BinOp::ShiftRightLogical, tall, 1)?;
        let k = bin(fb, BinOp::ShiftRightLogical, lin, 2)?;
        let rx = bin(fb, BinOp::BitwiseAnd, lin, 1)?;
        let ry = bin(fb, BinOp::ShiftRightLogical, lin, 1)?;
        let ry = bin(fb, BinOp::BitwiseAnd, ry, 1)?;
        // The column: `8 · qh` quads each but the last.
        let column_quads = bin(fb, BinOp::ShiftLeftLogical, qh, 3)?;
        let column_quads = at_least_one(fb, column_quads)?;
        let c = fb.binary(BinOp::UDiv, k, column_quads)?;
        let kc = fb.binary(BinOp::UMod, k, column_quads)?;
        let c0 = bin(fb, BinOp::ShiftLeftLogical, c, 3)?;
        let cw = left(fb, qw, c0, 8)?;
        // The column's row of quads: `2 · cw` quads each but the last.
        let row_quads = bin(fb, BinOp::ShiftLeftLogical, cw, 1)?;
        let r = fb.binary(BinOp::UDiv, kc, row_quads)?;
        let kr = fb.binary(BinOp::UMod, kc, row_quads)?;
        let r0 = bin(fb, BinOp::ShiftLeftLogical, r, 1)?;
        let rh = left(fb, qh, r0, 2)?;
        // The row's block: `4 · rh` quads each but the last.
        let block_quads = bin(fb, BinOp::ShiftLeftLogical, rh, 2)?;
        let b = fb.binary(BinOp::UDiv, kr, block_quads)?;
        let kb = fb.binary(BinOp::UMod, kr, block_quads)?;
        let b0 = bin(fb, BinOp::ShiftLeftLogical, b, 2)?;
        let bw = left(fb, cw, b0, 4)?;
        let in_x = fb.binary(BinOp::UMod, kb, bw)?;
        let in_y = fb.binary(BinOp::UDiv, kb, bw)?;
        let qx = fb.binary(BinOp::IAdd, c0, b0)?;
        let qx = fb.binary(BinOp::IAdd, qx, in_x)?;
        let qy = fb.binary(BinOp::IAdd, r0, in_y)?;
        let x = bin(fb, BinOp::ShiftLeftLogical, qx, 1)?;
        let x = fb.binary(BinOp::IAdd, x, rx)?;
        let y = bin(fb, BinOp::ShiftLeftLogical, qy, 1)?;
        let y = fb.binary(BinOp::IAdd, y, ry)?;
        let z = Self::k(fb, 0)?;
        fb.composite_construct(self.uvec3, &[x, y, z])
    }

    /// A tile function's `thread_index_in_threadgroup` of the thread at position `pos`: its 2x2
    /// quad's index in row-major order over the threadgroup's `⌈sx / 2⌉` quads a row, times four,
    /// plus its place in the quad (row-major).
    fn tile_index(&self, fb: &mut FunctionBuilder<'_>, pos: SValue) -> BResult<SValue> {
        let bin = |fb: &mut FunctionBuilder<'_>, op, a, n: u32| -> BResult<SValue> {
            let n = Self::k(fb, n)?;
            fb.binary(op, a, n)
        };
        let x = fb.composite_extract(pos, &[0])?;
        let y = fb.composite_extract(pos, &[1])?;
        let sx = fb.module().const_value(self.size[0])?;
        let wide = bin(fb, BinOp::IAdd, sx, 1)?;
        let quads = bin(fb, BinOp::ShiftRightLogical, wide, 1)?;
        let qy = bin(fb, BinOp::ShiftRightLogical, y, 1)?;
        let qx = bin(fb, BinOp::ShiftRightLogical, x, 1)?;
        let row = fb.binary(BinOp::IMul, qy, quads)?;
        let quad = fb.binary(BinOp::IAdd, row, qx)?;
        let base = bin(fb, BinOp::ShiftLeftLogical, quad, 2)?;
        let ry = bin(fb, BinOp::BitwiseAnd, y, 1)?;
        let ry2 = bin(fb, BinOp::ShiftLeftLogical, ry, 1)?;
        let rx = bin(fb, BinOp::BitwiseAnd, x, 1)?;
        let within = fb.binary(BinOp::IAdd, ry2, rx)?;
        fb.binary(BinOp::IAdd, base, within)
    }

    /// Whether a flat grid derives builtin `b`: the grid's position and size, and, of the call's
    /// threadgroups ([`DispatchPlan::call_threadgroups`]), a lane's index in its simdgroup or
    /// quadgroup and the simdgroup's width, which its `lin` answers as Metal's threadgroup does.
    fn flat_reads(&self, b: KernelBuiltin) -> bool {
        use KernelBuiltin as K;
        match b {
            K::ThreadPositionInGrid | K::ThreadsPerGrid => true,
            K::ThreadIndexInSimdgroup | K::ThreadIndexInQuadgroup | K::ThreadsPerSimdgroup => {
                self.call_threadgroups
            }
            _ => false,
        }
    }

    /// `thread_index_in_threadgroup`: `lin`, or a tile function's index of its position
    /// ([`Self::tile_index`]), but `lin` where its threadgroup is the tile.
    fn thread_index(&self, fb: &mut FunctionBuilder<'_>) -> BResult<SValue> {
        let lin = self.lin(fb)?;
        if !self.tile || self.tile_sized {
            return Ok(lin);
        }
        let pos = self.tile_pos(fb, lin)?;
        self.tile_index(fb, pos)
    }

    /// The region base and Metal's nominal threadgroup size `L` under `MultiRegion`, or `(0, (sx,
    /// sy, sz))` when the kernel reads no grid builtin. Under [`RegionSelect::Sized`] the base is
    /// the region's whatever `MultiRegion` says, and `L` the region's nominal size only under it,
    /// so a pipeline specialized at 0 reads the base alone.
    fn region(
        &self,
        fb: &mut FunctionBuilder<'_>,
        region: &mut RegionFn<'_>,
    ) -> BResult<(Option<SValue>, SValue)> {
        let s = self.size3(fb)?;
        let Some(sel) = self.multi_region else {
            return Ok((None, s));
        };
        let r = region(fb)?;
        let r0 = fb.composite_extract(r, &[0])?;
        let base = fb.vector_shuffle(r0, r0, &[0, 1, 2])?;
        let packed = fb.composite_extract(r0, &[3])?;
        let mask = Self::k(fb, (1 << NOMINAL_AXIS_BITS) - 1)?;
        let one = Self::k(fb, 1)?;
        let mut axes = Vec::with_capacity(3);
        for i in 0..3u32 {
            let sh = Self::k(fb, NOMINAL_AXIS_BITS * i)?;
            let shifted = fb.binary(BinOp::ShiftRightLogical, packed, sh)?;
            let bits = fb.binary(BinOp::BitwiseAnd, shifted, mask)?;
            axes.push(fb.binary(BinOp::IAdd, bits, one)?);
        }
        let nominal = fb.composite_construct(self.uvec3, &axes)?;
        match sel {
            RegionSelect::Spec(_) => {
                let mr = self.selected(fb, sel, region)?;
                let zero = self.splat(fb, 0)?;
                let base = fb.select(mr, base, zero)?;
                let l = fb.select(mr, nominal, s)?;
                Ok((Some(base), l))
            }
            RegionSelect::Sized(sized) => {
                let sized = fb.module().const_value(sized)?;
                let l = fb.select(sized, nominal, s)?;
                Ok((Some(base), l))
            }
        }
    }

    /// `threads_per_grid` under `MultiRegion`: `region[1].xyz`.
    fn region_threads(
        &self,
        fb: &mut FunctionBuilder<'_>,
        region: &mut RegionFn<'_>,
    ) -> BResult<Option<(SValue, SValue)>> {
        let Some(sel) = self.multi_region else {
            return Ok(None);
        };
        let mr = self.selected(fb, sel, region)?;
        let r = region(fb)?;
        let r1 = fb.composite_extract(r, &[1])?;
        Ok(Some((mr, fb.vector_shuffle(r1, r1, &[0, 1, 2])?)))
    }

    /// Whether the dispatch is a region (a `bool`): `MultiRegion`, or under the identity domains
    /// whether the region's grid `region[1].xyz` is not all 0.
    fn selected(
        &self,
        fb: &mut FunctionBuilder<'_>,
        sel: RegionSelect,
        region: &mut RegionFn<'_>,
    ) -> BResult<SValue> {
        match sel {
            RegionSelect::Spec(mr) => fb.module().const_value(mr),
            RegionSelect::Sized(_) => {
                let r = region(fb)?;
                let r1 = fb.composite_extract(r, &[1])?;
                let [x, y, z] = [0, 1, 2].map(|i| fb.composite_extract(r1, &[i]));
                let xy = fb.binary(BinOp::BitwiseOr, x?, y?)?;
                let any = fb.binary(BinOp::BitwiseOr, xy, z?)?;
                let zero = Self::k(fb, 0)?;
                fb.binary(BinOp::INotEqual, any, zero)
            }
        }
    }

    /// A flat grid's thread position in the grid (a `uvec3`) and whether the invocation is inside
    /// the grid: its index `((wz · ny + wy) · nx + wx) · N + LocalInvocationIndex` over the
    /// dispatch (`lin` for the call's threadgroups, [`Self::lin`]), with `N = sx·sy·sz`, counted
    /// over `threads_per_grid` (`region[1].xyz`) `x` fastest. Inside the grid each axis's position is below its size, so it fits its `u32`;
    /// past it the position is unused.
    ///
    /// Three arms, each chosen by values uniform over the workgroup, so a workgroup takes one:
    ///
    /// - a workgroup with `wy = wz = 0` (group `wx`) whose last index `wx·N + N - 1` is below
    ///   2^32 counts in 32 bits from `wx·N`, its test and start read from the workgroup's id and
    ///   `N` alone (every workgroup of a dispatch whose groups fit along `x`);
    /// - any other, of a dispatch of at most 2^32 invocations (`nx · ny · nz · N <= 2^32`, so
    ///   every index is below 2^32), counts in 32 bits from `((wz · ny + wy) · nx + wx) · N`;
    /// - past that the index is 64-bit and divided by [`Self::divmod_64_by_32`].
    ///
    /// Devices without a 64-bit divider (Intel's) lower each 64-bit division to a long software
    /// sequence per invocation. ANV (Mesa 26.2) also compiles a whole kernel at SIMD8 while
    /// 64-bit `OpUDiv`/`OpUMod` sit in any arm of it, and merges `NumWorkgroups` from two sources
    /// per invocation: the first arm reads neither.
    fn flat(
        &self,
        fb: &mut FunctionBuilder<'_>,
        region: &mut RegionFn<'_>,
    ) -> BResult<(SValue, SValue)> {
        let u64_ = fb.module().ty(Type::Int {
            width: 64,
            signed: false,
        })?;
        let u32_ = fb.module().ty(Type::Int {
            width: 32,
            signed: false,
        })?;
        let wg = self.load(fb, self.workgroup_id)?;
        let groups = self.load(fb, self.num_workgroups)?;
        let lin = match self.call_threadgroups {
            true => self.lin(fb)?,
            false => self.load(fb, self.local_index)?,
        };
        let s = self.size3(fb)?;
        let n = Self::volume(fb, s)?;
        let r = region(fb)?;
        let grid = fb.composite_extract(r, &[1])?;
        let part = |fb: &mut FunctionBuilder<'_>, v: SValue| -> BResult<[SValue; 3]> {
            Ok([
                fb.composite_extract(v, &[0])?,
                fb.composite_extract(v, &[1])?,
                fb.composite_extract(v, &[2])?,
            ])
        };
        let [wx, wy, wz] = part(fb, wg)?;
        let [nx, ny, nz] = part(fb, groups)?;
        let [gx, gy, gz] = part(fb, grid)?;
        let w = |fb: &mut FunctionBuilder<'_>, v: SValue| fb.convert(ConvOp::UConvert, v, u64_);
        let limit = fb.module().constant(Constant::Scalar {
            ty: u64_,
            bits: ConstBits(1 << 32),
        })?;
        let limit = fb.module().const_value(limit)?;
        // A workgroup of the first row and layer (`wy = wz = 0`, every workgroup of a dispatch
        // whose groups fit along `x`) is group `wx`, whatever the counts: when its last index
        // `wx·N + N - 1` is below 2^32 its indices are counted in 32 bits from `wx·N`. The test
        // and `wx·N` read only the workgroup's id and the threadgroup size, values the driver
        // holds once per hardware thread, not per invocation; the counts (`NumWorkgroups`) a
        // driver may merge per invocation are read only past it.
        let (wx64, n64) = (w(fb, wx)?, w(fb, n)?);
        let row = fb.binary(BinOp::BitwiseOr, wy, wz)?;
        let zero = Self::k(fb, 0)?;
        let first_row = fb.binary(BinOp::IEqual, row, zero)?;
        let start = fb.binary(BinOp::IMul, wx64, n64)?;
        let end = fb.binary(BinOp::IAdd, start, n64)?;
        let below = fb.binary(BinOp::ULessThanEqual, end, limit)?;
        let row_fits = fb.binary(BinOp::LogicalAnd, first_row, below)?;
        let first: ArmFn<'_, ()> = Box::new(move |fb, _| {
            let first = fb.convert(ConvOp::UConvert, start, u32_)?;
            let index = fb.binary(BinOp::IAdd, first, lin)?;
            let x = fb.binary(BinOp::UMod, index, gx)?;
            let row = fb.binary(BinOp::UDiv, index, gx)?;
            let y = fb.binary(BinOp::UMod, row, gy)?;
            let z = fb.binary(BinOp::UDiv, row, gy)?;
            let inside = fb.binary(BinOp::ULessThan, z, gz)?;
            Ok(ArmEnd::FallThrough(vec![x, y, z, inside]))
        });
        let narrow: ArmFn<'_, ()> = Box::new(move |fb, _| {
            let group = fb.binary(BinOp::IMul, wz, ny)?;
            let group = fb.binary(BinOp::IAdd, group, wy)?;
            let group = fb.binary(BinOp::IMul, group, nx)?;
            let group = fb.binary(BinOp::IAdd, group, wx)?;
            let first = fb.binary(BinOp::IMul, group, n)?;
            let index = fb.binary(BinOp::IAdd, first, lin)?;
            let x = fb.binary(BinOp::UMod, index, gx)?;
            let row = fb.binary(BinOp::UDiv, index, gx)?;
            let y = fb.binary(BinOp::UMod, row, gy)?;
            let z = fb.binary(BinOp::UDiv, row, gy)?;
            let inside = fb.binary(BinOp::ULessThan, z, gz)?;
            Ok(ArmEnd::FallThrough(vec![x, y, z, inside]))
        });
        let wide: ArmFn<'_, ()> = Box::new(move |fb, _| {
            let w = |fb: &mut FunctionBuilder<'_>, v: SValue| fb.convert(ConvOp::UConvert, v, u64_);
            let (wx, wy, wz) = (w(fb, wx)?, w(fb, wy)?, w(fb, wz)?);
            let (nx, ny, n, lin) = (w(fb, nx)?, w(fb, ny)?, w(fb, n)?, w(fb, lin)?);
            let group = fb.binary(BinOp::IMul, wz, ny)?;
            let group = fb.binary(BinOp::IAdd, group, wy)?;
            let group = fb.binary(BinOp::IMul, group, nx)?;
            let group = fb.binary(BinOp::IAdd, group, wx)?;
            let first = fb.binary(BinOp::IMul, group, n)?;
            let index = fb.binary(BinOp::IAdd, first, lin)?;
            let lo = fb.convert(ConvOp::UConvert, index, u32_)?;
            let thirty_two = Self::k(fb, 32)?;
            let hi = fb.binary(BinOp::ShiftRightLogical, index, thirty_two)?;
            let hi = fb.convert(ConvOp::UConvert, hi, u32_)?;
            let (row_lo, row_hi, x) = Self::divmod_64_by_32(fb, [lo, hi], gx)?;
            let (z_lo, z_hi, y) = Self::divmod_64_by_32(fb, [row_lo, row_hi], gy)?;
            // `z < gz` in 64 bits; the position's `z` is its low word, as the conversion was.
            let zero = Self::k(fb, 0)?;
            let small = fb.binary(BinOp::IEqual, z_hi, zero)?;
            let below = fb.binary(BinOp::ULessThan, z_lo, gz)?;
            let inside = fb.binary(BinOp::LogicalAnd, small, below)?;
            Ok(ArmEnd::FallThrough(vec![x, y, z_lo, inside]))
        });
        // Any other workgroup: whether every invocation's index fits in 32 bits, `a = nx·ny`,
        // `b = a·nz`, `c = b·N` in 64 bits, each at most 2^32. While `a` and `b` are, the next
        // product cannot wrap (`nz`, `N` < 2^32); a product past a failed test is not read.
        let rest: ArmFn<'_, ()> = Box::new(move |fb, _| {
            let w = |fb: &mut FunctionBuilder<'_>, v: SValue| fb.convert(ConvOp::UConvert, v, u64_);
            let (nx64, ny64, nz64) = (w(fb, nx)?, w(fb, ny)?, w(fb, nz)?);
            let a = fb.binary(BinOp::IMul, nx64, ny64)?;
            let b = fb.binary(BinOp::IMul, a, nz64)?;
            let c = fb.binary(BinOp::IMul, b, n64)?;
            let fa = fb.binary(BinOp::ULessThanEqual, a, limit)?;
            let fbb = fb.binary(BinOp::ULessThanEqual, b, limit)?;
            let fc = fb.binary(BinOp::ULessThanEqual, c, limit)?;
            let fits = fb.binary(BinOp::LogicalAnd, fa, fbb)?;
            let fits = fb.binary(BinOp::LogicalAnd, fits, fc)?;
            let merged = fb.if_(&mut (), fits, narrow, Some(wide))?.0;
            Ok(ArmEnd::FallThrough(merged))
        });
        let merged = fb.if_(&mut (), row_fits, first, Some(rest))?.0;
        let [x, y, z, inside] = merged[..] else {
            return Err(crate::spirv::error::BuildError::Dangling(
                crate::spirv::error::Dangling::Value(SValue::new(u32::MAX)),
            ));
        };
        let pos = fb.composite_construct(self.uvec3, &[x, y, z])?;
        Ok((pos, inside))
    }

    /// The quotient (low word, high word) and remainder of the `u64` `n` (`[low, high]`) by the
    /// `u32` `d` (not 0), in 32-bit operations: `hi / d` and `r = hi % d` (so `r < d`), then 32
    /// restoring steps over the low word's bits, highest first, each `r = 2r + bit` and, when
    /// that is at least `d`, `r -= d` and the step's quotient bit set. `2r + bit < 2d < 2^33`: its
    /// bit 32 (`r`'s top bit before the shift) is the carry, and with it set the value is past `d`
    /// and `2r + bit - d < d` is exact modulo 2^32. The loop asks the driver not to unroll it.
    ///
    /// The flat variant's 64-bit arm divides by it rather than by a 64-bit `OpUDiv`/`OpUMod`:
    /// lowered on a device without a 64-bit divider (Intel's), four of those made the arm so large
    /// that the whole kernel compiled at a narrower SIMD width (SIMD8 for SIMD16 on ANV, Mesa
    /// 26.2), though the arm runs only past 2^32 invocations.
    fn divmod_64_by_32(
        fb: &mut FunctionBuilder<'_>,
        [lo, hi]: [SValue; 2],
        d: SValue,
    ) -> BResult<(SValue, SValue, SValue)> {
        use crate::spirv::builder::constructs::{LoopFn, LoopSpec};
        let u32_ = fb.module().ty(Type::Int {
            width: 32,
            signed: false,
        })?;
        let q_hi = fb.binary(BinOp::UDiv, hi, d)?;
        let r0 = fb.binary(BinOp::UMod, hi, d)?;
        let [r, q, i] = [
            fb.local_var(u32_, None)?,
            fb.local_var(u32_, None)?,
            fb.local_var(u32_, None)?,
        ];
        let [r, q, i] = [fb.var_ptr(r)?, fb.var_ptr(q)?, fb.var_ptr(i)?];
        let zero = Self::k(fb, 0)?;
        let steps = Self::k(fb, 32)?;
        fb.store(r, r0, MemAccess::default())?;
        fb.store(q, zero, MemAccess::default())?;
        fb.store(i, steps, MemAccess::default())?;
        let body: LoopFn<'_, ()> = Box::new(move |fb, cx, lc| {
            let k = fb.load(i, MemAccess::default())?;
            let zero = Self::k(fb, 0)?;
            let done = fb.binary(BinOp::IEqual, k, zero)?;
            let exit: ArmFn<'_, ()> = Box::new(move |fb, _| {
                lc.break_(fb)?;
                Ok(ArmEnd::Exits)
            });
            fb.if_(cx, done, exit, None)?;
            let one = Self::k(fb, 1)?;
            let k = fb.binary(BinOp::ISub, k, one)?;
            let rv = fb.load(r, MemAccess::default())?;
            let qv = fb.load(q, MemAccess::default())?;
            let bit = fb.binary(BinOp::ShiftRightLogical, lo, k)?;
            let bit = fb.binary(BinOp::BitwiseAnd, bit, one)?;
            let top = Self::k(fb, 31)?;
            let carry = fb.binary(BinOp::ShiftRightLogical, rv, top)?;
            let carry = fb.binary(BinOp::INotEqual, carry, zero)?;
            let twice = fb.binary(BinOp::ShiftLeftLogical, rv, one)?;
            let twice = fb.binary(BinOp::BitwiseOr, twice, bit)?;
            let reaches = fb.binary(BinOp::UGreaterThanEqual, twice, d)?;
            let take = fb.binary(BinOp::LogicalOr, carry, reaches)?;
            let less = fb.binary(BinOp::ISub, twice, d)?;
            let rv = fb.select(take, less, twice)?;
            let place = fb.binary(BinOp::ShiftLeftLogical, one, k)?;
            let place = fb.select(take, place, zero)?;
            let qv = fb.binary(BinOp::BitwiseOr, qv, place)?;
            fb.store(r, rv, MemAccess::default())?;
            fb.store(q, qv, MemAccess::default())?;
            fb.store(i, k, MemAccess::default())
        });
        let spec = LoopSpec {
            exit_sel: None,
            unrolling: crate::spirv::ir::Unrolling::DontUnroll,
        };
        fb.loop_(&mut (), spec, body)?;
        let q_lo = fb.load(q, MemAccess::default())?;
        let rem = fb.load(r, MemAccess::default())?;
        Ok((q_lo, q_hi, rem))
    }

    /// A folded variant's threadgroup position in the Metal grid (a `uvec3`) and whether its
    /// workgroup is one of Metal's threadgroups: the workgroup's index `(wz · ny + wy) · nx + wx`
    /// over the dispatch (plus `DrawIndex · nx · ny · nz` for a folded object function, whose
    /// draw's pieces each run that many), counted over Metal's group counts `x` fastest, all in 64
    /// bits. A kernel's counts are `threads_per_grid / (sx, sy, sz)` (`region[1].xyz`); a folded
    /// object function's are the record's base words (`region[0].xyz`, which no folded variant
    /// reads as a base), since its threads per grid are Metal's, wrapped to 32 bits
    /// (`bind::metal_object_grid`). Every value it reads is the workgroup's, so the whole workgroup
    /// is inside or past. Inside the grid each axis's position is below its
    /// count, so it fits its `u32`; past it the position is unused. A count of zero runs no
    /// workgroup (the preprocess writes a zero dispatch), so no running invocation divides by it.
    fn folded(
        &self,
        fb: &mut FunctionBuilder<'_>,
        region: &mut RegionFn<'_>,
    ) -> BResult<(SValue, SValue)> {
        let u64_ = fb.module().ty(Type::Int {
            width: 64,
            signed: false,
        })?;
        let wide = |fb: &mut FunctionBuilder<'_>, v: SValue, axis: u32| -> BResult<SValue> {
            let x = fb.composite_extract(v, &[axis])?;
            fb.convert(ConvOp::UConvert, x, u64_)
        };
        let wg = self.load(fb, self.workgroup_id)?;
        let dispatched = self.load(fb, self.num_workgroups)?;
        let [wx, wy, wz] = [wide(fb, wg, 0)?, wide(fb, wg, 1)?, wide(fb, wg, 2)?];
        let [nx, ny] = [wide(fb, dispatched, 0)?, wide(fb, dispatched, 1)?];
        let index = fb.binary(BinOp::IMul, wz, ny)?;
        let index = fb.binary(BinOp::IAdd, index, wy)?;
        let index = fb.binary(BinOp::IMul, index, nx)?;
        let mut index = fb.binary(BinOp::IAdd, index, wx)?;
        if let Some(draw) = self.draw_index {
            let piece = fb.module().global_value(draw)?;
            let piece = fb.load(piece, MemAccess::default())?;
            let piece = fb.convert(ConvOp::UConvert, piece, u64_)?;
            let nz = wide(fb, dispatched, 2)?;
            let size = fb.binary(BinOp::IMul, nx, ny)?;
            let size = fb.binary(BinOp::IMul, size, nz)?;
            let first = fb.binary(BinOp::IMul, piece, size)?;
            index = fb.binary(BinOp::IAdd, index, first)?;
        }
        let r = region(fb)?;
        let counts = match self.draw_index {
            Some(_) => {
                let runs = fb.composite_extract(r, &[0])?;
                fb.vector_shuffle(runs, runs, &[0, 1, 2])?
            }
            None => {
                let threads = fb.composite_extract(r, &[1])?;
                let threads = fb.vector_shuffle(threads, threads, &[0, 1, 2])?;
                let s = self.size3(fb)?;
                fb.binary(BinOp::UDiv, threads, s)?
            }
        };
        let [cx, cy, cz] = [
            wide(fb, counts, 0)?,
            wide(fb, counts, 1)?,
            wide(fb, counts, 2)?,
        ];
        let x = fb.binary(BinOp::UMod, index, cx)?;
        let row = fb.binary(BinOp::UDiv, index, cx)?;
        let y = fb.binary(BinOp::UMod, row, cy)?;
        let z = fb.binary(BinOp::UDiv, row, cy)?;
        let inside = fb.binary(BinOp::ULessThan, z, cz)?;
        let u32_ = fb.module().ty(Type::Int {
            width: 32,
            signed: false,
        })?;
        let mut axes = [x, y, z];
        for a in &mut axes {
            *a = fb.convert(ConvOp::UConvert, *a, u32_)?;
        }
        let pos = fb.composite_construct(self.uvec3, &axes)?;
        Ok((pos, inside))
    }

    /// The threadgroup position in the Metal grid: `base + WorkgroupId`.
    fn group_pos(&self, fb: &mut FunctionBuilder<'_>, base: Option<SValue>) -> BResult<SValue> {
        let wg = self.load(fb, self.workgroup_id)?;
        match base {
            Some(b) => fb.binary(BinOp::IAdd, b, wg),
            None => Ok(wg),
        }
    }

    /// Metal's threads per threadgroup: the region's nominal size where the tile function's
    /// threadgroup is the tile (which the binder's regions set to the tile), the threads a partial
    /// object threadgroup runs ([`Self::partial_threads`]), else the threadgroup that runs.
    fn threadgroup_size(
        &self,
        fb: &mut FunctionBuilder<'_>,
        region: &mut RegionFn<'_>,
    ) -> BResult<SValue> {
        if self.tile_sized {
            return Ok(self.region(fb, region)?.1);
        }
        match self.partial_threads(fb, region)? {
            Some((_, threads)) => Ok(threads),
            None => self.size3(fb),
        }
    }

    /// The threads the threadgroup runs, `(sx, sy, sz)`, whatever threadgroup Metal answers: a
    /// tile function's staging covers its tile with them (`lower::access::tile`).
    pub fn running_threads(&self, fb: &mut FunctionBuilder<'_>) -> BResult<SValue> {
        self.size3(fb)
    }

    /// The value of builtin `b` as a `u32` or `uvec3` ([`KernelBuiltin::is_vector`]). `region`
    /// produces the `DispatchRegion` push field's value; it is called only when the kernel reads
    /// a grid builtin.
    pub fn value(
        &self,
        fb: &mut FunctionBuilder<'_>,
        b: KernelBuiltin,
        region: &mut RegionFn<'_>,
    ) -> BResult<SValue> {
        use KernelBuiltin as K;
        let bin = |fb: &mut FunctionBuilder<'_>, op, a, n: u32| -> BResult<SValue> {
            let n = Self::k(fb, n)?;
            fb.binary(op, a, n)
        };
        // A folded variant's: its threadgroup's position unfolded from its workgroup's; every other
        // grid builtin reads the region record (`MultiRegion` 1).
        if self.folded_groups {
            match b {
                K::ThreadgroupPositionInGrid => return Ok(self.folded(fb, region)?.0),
                K::ThreadPositionInGrid => {
                    let g = self.folded(fb, region)?.0;
                    let s = self.size3(fb)?;
                    let first = fb.binary(BinOp::IMul, g, s)?;
                    let pos = self.local_pos(fb, region)?;
                    return fb.binary(BinOp::IAdd, first, pos);
                }
                _ => {}
            }
        }
        // A flat grid's: its position, and the region record's grid.
        if self.flat_grid {
            match b {
                K::ThreadPositionInGrid => return Ok(self.flat(fb, region)?.0),
                K::ThreadsPerGrid => {
                    let r = region(fb)?;
                    let r1 = fb.composite_extract(r, &[1])?;
                    return fb.vector_shuffle(r1, r1, &[0, 1, 2]);
                }
                _ => {}
            }
        }
        match b {
            K::ThreadIndexInThreadgroup => self.thread_index(fb),
            K::ThreadIndexInSimdgroup => {
                let lin = self.lin(fb)?;
                bin(fb, BinOp::BitwiseAnd, lin, 31)
            }
            K::SimdgroupIndexInThreadgroup => {
                let lin = self.lin(fb)?;
                bin(fb, BinOp::ShiftRightLogical, lin, 5)
            }
            // A quadgroup is four consecutive thread indices, in a tile function too.
            K::ThreadIndexInQuadgroup => {
                let index = self.thread_index(fb)?;
                bin(fb, BinOp::BitwiseAnd, index, 3)
            }
            K::QuadgroupIndexInThreadgroup => {
                let index = self.thread_index(fb)?;
                bin(fb, BinOp::ShiftRightLogical, index, 2)
            }
            // The simdgroup width is always 32.
            K::ThreadsPerSimdgroup => Self::k(fb, 32),
            // A tile function's pass is not layered.
            K::RenderTargetArrayIndex => Self::k(fb, 0),
            K::SimdgroupsPerThreadgroup | K::QuadgroupsPerThreadgroup => {
                let s = self.threadgroup_size(fb, region)?;
                let n = Self::volume(fb, s)?;
                let w = if b == K::SimdgroupsPerThreadgroup {
                    32
                } else {
                    4
                };
                let w = Self::k(fb, w)?;
                self.ceil_div(fb, n, w)
            }
            K::ThreadsPerThreadgroup => self.threadgroup_size(fb, region),
            K::ThreadPositionInThreadgroup => self.local_pos(fb, region),
            K::ThreadgroupPositionInGrid => {
                let (base, _) = self.region(fb, region)?;
                self.group_pos(fb, base)
            }
            K::ThreadPositionInGrid if self.grid_edge => self.edge_position(fb, region),
            K::ThreadPositionInGrid => {
                let (base, l) = self.region(fb, region)?;
                let g = self.group_pos(fb, base)?;
                let first = fb.binary(BinOp::IMul, g, l)?;
                let pos = self.local_pos(fb, region)?;
                fb.binary(BinOp::IAdd, first, pos)
            }
            K::DispatchThreadsPerThreadgroup => Ok(self.region(fb, region)?.1),
            K::DispatchSimdgroupsPerThreadgroup => {
                let (_, l) = self.region(fb, region)?;
                let n = Self::volume(fb, l)?;
                let w = Self::k(fb, 32)?;
                self.ceil_div(fb, n, w)
            }
            K::ThreadsPerGrid => {
                let groups = self.load(fb, self.num_workgroups)?;
                let s = self.size3(fb)?;
                let single = fb.binary(BinOp::IMul, groups, s)?;
                match self.region_threads(fb, region)? {
                    Some((mr, t)) => fb.select(mr, t, single),
                    None => Ok(single),
                }
            }
            K::ThreadgroupsPerGrid => {
                let mut groups = self.load(fb, self.num_workgroups)?;
                // An object function's grid: the threadgroups its wrapped threads fill.
                if self.object_grid {
                    let s = self.size3(fb)?;
                    let threads = fb.binary(BinOp::IMul, groups, s)?;
                    groups = self.ceil_div(fb, threads, s)?;
                }
                match self.region_threads(fb, region)? {
                    Some((mr, t)) => {
                        let (_, l) = self.region(fb, region)?;
                        let multi = self.ceil_div(fb, t, l)?;
                        fb.select(mr, multi, groups)
                    }
                    None => Ok(groups),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
