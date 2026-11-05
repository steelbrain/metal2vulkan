//! The arguments pass of an indirect mesh draw on a device with `VK_EXT_mesh_shader`.
//!
//! `drawMeshThreadgroupsWithIndirectBuffer:` reads its group counts (an
//! `MTLDispatchThreadgroupsIndirectArguments`) as it runs. `vkCmdDrawMeshTasksIndirectEXT` requires
//! every count within the device's task or mesh limits, so the draw cannot read Metal's arguments
//! in place: the consumer records [`MeshArguments`] first, one invocation of the crate-owned
//! module [`mesh_arguments_module`] that reads Metal's three words and writes what the draw reads
//! into a consumer-provided scratch of [`MeshArguments::scratch_bytes`].
//!
//! **What Metal draws**, measured by our probe on an M3 Max (macOS 27.0.1), direct and indirect
//! draws alike, with and without API validation, which stops none of them:
//!
//! - Without an object function the counts are the mesh grid. A grid with an axis past
//!   [`METAL_MESH_GRID_AXIS`] (65,535) draws nothing; so does one of exactly 2^20 threadgroups
//!   (1024 by 1024, 128 by 128 by 64, 4096 by 256) and every one measured of 2^21 or more. Grids
//!   of up to [`METAL_MESH_GRID_TOTAL`] (1,048,575, Apple's limit for the family) draw whole. Past
//!   that Metal is erratic (17 by 61,681 and 65,535 by 24 drew, 1025 by 1025 and 33 by 32,768 did
//!   not), and the module draws nothing, Metal's answer at 2^20 and above 2^21.
//! - With an object function the counts are the object grid, which Metal runs at every count
//!   that its GPU finishes within its watchdog's 5 s. An object function that does next to nothing
//!   ran 2^28 threadgroups in 2.4 s and 2^29 in 4.98 s, direct and indirect; 2^30 and 2^32 were
//!   discarded at 5 s, as was 2^29 of a costlier function: the command buffer fails with
//!   `MTLCommandBufferErrorInternal` (1, "Discarded (victim of GPU error/recovery)"), and nothing
//!   of it runs from the draw's encoder on (its clear, its draws before the grid's, the encoders
//!   after it; an encoder before it ran), nor does a command buffer of the queue committed while
//!   it ran. [`METAL_OBJECT_GRID_TOTAL`] is the most Metal ran; past it the runtime fails the
//!   command buffer so (`DrawCommands::discards_command_buffer`, `MESH_GRID_DISCARDED`). Metal's
//!   threads per grid along an axis are
//!   the axis's threadgroups times its threads per threadgroup wrapped to 32 bits, which its grid
//!   builtins report (`threads_per_grid`, and `threadgroups_per_grid` those threads over the
//!   threadgroup's, rounded up): an indirect draw runs every threadgroup of its arguments
//!   (4,294,970 by 1000 threads ran 4,294,970 threadgroups that each read a grid of 2704 threads,
//!   3 threadgroups), a direct draw only the threadgroups the wrapped threads fill, rounded up
//!   (that draw ran 3, 4,194,304 by 1024 none) ([`metal_object_grid`]). The mesh grid an object
//!   threadgroup sets follows the rule above (65,536 along `x` or `y`, 1024 by 1024 and 2048 by
//!   1024 launch nothing from it; 1023 by 1025 and 65,535 by 16 launch whole), which the object
//!   function's lowering holds it to (`lower::mesh`).
//!
//! **The mesh form's rule** ([`MeshGridRule`], its constants): where Metal draws nothing, it
//! writes zero counts, which draw nothing. Where Metal draws and the counts lie within the
//! device's `maxMeshWorkGroupCount` and `maxMeshWorkGroupTotalCount`, it copies them; every
//! committed profile with the extension holds Metal's mesh grids so. Where they do not, it writes
//! zero counts and sets `MESH_GRID_PAST_LIMITS` in the command buffer's `ExecutionStatus`, which the
//! consumer reports as a command-buffer error.
//!
//! **The folded object form** ([`MeshArgumentsForm::FoldedObject`]) runs any object grid in
//! pieces within the task stage's limits ([`fold_object_grid`], which a direct draw's binder
//! computes on the CPU): on a device that folds object grids (`lower::mesh::folds_object_grids`)
//! the object function is compiled folded (`LoweringInputs::folded_groups`), so a task workgroup
//! counts its Metal threadgroup from its workgroup's position, the dispatch's size and its
//! `DrawIndex`, reads the threadgroups Metal runs and its threads per grid from the 32-byte region
//! record at its `PushField::ReplayRegionAddress`, and launches nothing past Metal's last
//! threadgroup. The module writes, into the scratch ([`MeshArguments::OBJECT_RECORD`] and after):
//!
//! - the region record: the threadgroups Metal runs along each axis in the base's words (which a
//!   folded variant reads in their place, `lower::dispatch`), the packed threads per object
//!   threadgroup, and Metal's threads per grid ([`metal_object_grid`]);
//! - the draw count `P` ([`MeshArguments::OBJECT_COUNT`]) and [`MeshArguments::OBJECT_PIECES`]
//!   commands of one piece's dispatch ([`MeshArguments::OBJECT_COMMANDS`]), which
//!   `vkCmdDrawMeshTasksIndirectCountEXT` reads.
//!
//! A grid with a zero count draws nothing. One of more than [`METAL_OBJECT_GRID_TOTAL`]
//! threadgroups draws nothing and sets `MESH_GRID_DISCARDED`: Metal's GPU discards it. One that
//! more pieces than [`MeshArguments::OBJECT_PIECES`] would take (none on a device of the task
//! limits Vulkan requires) draws nothing and sets `MESH_GRID_PAST_LIMITS`: the device cannot
//! launch it.
//!
//! **The unfolded object form** ([`MeshArgumentsForm::Object`], a device that does not fold
//! object grids) is the mesh form's rule with Metal's discard as its limit: a grid of more than
//! [`METAL_OBJECT_GRID_TOTAL`] threadgroups writes zero counts and sets `MESH_GRID_DISCARDED`;
//! otherwise Metal's counts where they lie within the device's `maxTaskWorkGroupCount` and
//! `maxTaskWorkGroupTotalCount`, else zero counts and `MESH_GRID_PAST_LIMITS`.
//!
//! **The module** (fixed per device and form): one `GLCompute` entry `main` with local size
//! `(1, 1, 1)`, no loops, three storage buffers in set 0 and one push constant:
//!
//! | Binding | Buffer | Range |
//! |---|---|---|
//! | [`MeshArgumentsModule::SOURCE_BINDING`] | the Metal indirect buffer | [`MeshArguments::source_range`] |
//! | [`MeshArgumentsModule::SCRATCH_BINDING`] | the consumer's scratch | [`MeshArguments::scratch_bytes`] |
//! | [`MeshArgumentsModule::STATUS_BINDING`] | the `ExecutionStatus` buffer | `ExecutionStatus::BYTES` |
//!
//! The push constants ([`MeshArguments::push_constants`], `COMPUTE`, offset 0) are the byte distance
//! from the bound source range's start to the arguments (a multiple of 4), since a storage
//! descriptor's offset must be a multiple of `minStorageBufferOffsetAlignment` where Metal's
//! argument offset is only 4-aligned, and the folded object form's threads per object threadgroup,
//! packed as a region record packs them. Reads of the source are contained (an index past the
//! bound range reads 0).
//!
//! **Synchronization** (consumer): the source write → `SHADER_READ` in `COMPUTE_SHADER` before
//! the pass; the scratch `SHADER_WRITE` → `INDIRECT_COMMAND_READ` in `DRAW_INDIRECT`, and for the
//! folded object form `SHADER_READ` in `TASK_SHADER`, between the pass and the draw.

use std::sync::Arc;

use crate::bind::instantiate::{BufferRange, IndirectRef};
use crate::bind::mesh_scan::{div_ceil, for_range};
use crate::bind::replay::checked_mul;
use crate::bind::status::{StatusBits, StatusWord};
use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, InternalError, InternalKind};
use crate::pipeline::sym::{
    PushField, ResourceKind, ResourceSlot, SlotClass, SlotRef, Stage, Symbol, SyntheticBinding,
};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, AtomicOp, BinOp, ConstBits, Constant, ExecMode, ExecModel, GlslOp, LayoutRule,
    MemAccess, Scope, Semantics, StorageClass, Type, TypedModule, UnOp,
};

/// The most mesh threadgroups along an axis of a grid Metal draws without an object function
/// (the module docs).
pub const METAL_MESH_GRID_AXIS: u32 = 65_535;

/// The most mesh threadgroups of a grid Metal draws whole without an object function (the module
/// docs).
pub const METAL_MESH_GRID_TOTAL: u32 = 1_048_575;

/// The most object threadgroups of a draw the runtime runs: the most Metal's GPU ran within its
/// watchdog's 5 seconds, of an object function that does next to nothing (2^29 in 4.98 s on an
/// M3 Max; 2^30 was discarded, the module docs). A larger grid draws nothing and fails the
/// command buffer as Metal's GPU discards it.
pub const METAL_OBJECT_GRID_TOTAL: u32 = 1 << 29;

/// Whether Metal draws nothing for a mesh grid of `groups` without an object function: an axis
/// past [`METAL_MESH_GRID_AXIS`], or more than [`METAL_MESH_GRID_TOTAL`] threadgroups.
pub(crate) fn metal_mesh_grid_draws_nothing(groups: [u32; 3]) -> bool {
    let total: u128 = groups.iter().map(|&g| u128::from(g)).product();
    groups.iter().any(|&g| g > METAL_MESH_GRID_AXIS) || total > u128::from(METAL_MESH_GRID_TOTAL)
}

/// The first stage of an indirect mesh draw, whose threadgroups Metal's arguments count.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MeshArgumentsForm {
    /// The mesh function's, of a pipeline without an object function.
    Mesh,
    /// The object function's, on a device that does not fold object grids
    /// (`lower::mesh::folds_object_grids`): Metal's counts where the device's task stage launches
    /// them.
    Object,
    /// The object function's, compiled folded on a device that folds object grids: the fold of
    /// Metal's counts ([`fold_object_grid`]).
    FoldedObject,
}

/// A grid's limits: the largest count along each axis, and the largest product.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MeshGridLimits {
    /// The largest count along `x`, `y` and `z`.
    pub axes: [u32; 3],
    /// The largest product of the three counts.
    pub total: u32,
}

/// What [`mesh_arguments_module`] of one form holds the counts to (the module docs): Metal's
/// limits, past which it draws nothing, and the device's for the draw's first stage.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MeshGridRule {
    /// Metal's limits, or `None` where Metal draws every grid (an object grid).
    pub metal: Option<MeshGridLimits>,
    /// The device's limits for the first stage: mesh or task.
    pub device: MeshGridLimits,
}

impl MeshGridRule {
    /// The rule of `form` on a device of `limits`.
    pub fn of(form: MeshArgumentsForm, limits: &crate::device::MeshLimits) -> MeshGridRule {
        match form {
            MeshArgumentsForm::Mesh => MeshGridRule {
                metal: Some(MeshGridLimits {
                    axes: [METAL_MESH_GRID_AXIS; 3],
                    total: METAL_MESH_GRID_TOTAL,
                }),
                device: MeshGridLimits {
                    axes: limits.max_mesh_work_group_count,
                    total: limits.max_mesh_work_group_total_count,
                },
            },
            MeshArgumentsForm::Object | MeshArgumentsForm::FoldedObject => MeshGridRule {
                metal: None,
                device: MeshGridLimits {
                    axes: limits.max_task_work_group_count,
                    total: limits.max_task_work_group_total_count,
                },
            },
        }
    }
}

/// How a draw of an object grid runs on a device that folds object grids (the module docs): the
/// region record its object function reads, one piece's dispatch, and the pieces.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ObjectFold {
    /// The 32-byte region record: the threadgroups Metal runs along each axis (in the base's
    /// words), the packed threads per object threadgroup, Metal's threads per grid
    /// ([`metal_object_grid`]), and 0, or, for a `drawMeshThreads` that leaves a partial object
    /// threadgroup, the flag that its last threadgroup along an axis runs only the threads left
    /// (`lower::dispatch::NONUNIFORM`, 1).
    pub record: [u32; 8],
    /// The group counts of each piece's `VkDrawMeshTasksIndirectCommandEXT`, within the task
    /// stage's limits.
    pub dispatch: [u32; 3],
    /// The pieces, each a draw whose `DrawIndex` is its number: 0 for a grid that draws nothing.
    pub pieces: u32,
    /// The grid has more than [`METAL_OBJECT_GRID_TOTAL`] threadgroups, which Metal's GPU
    /// discards: it draws nothing, and the command buffer fails as Metal's does.
    pub discarded: bool,
    /// The grid takes more than [`MeshArguments::OBJECT_PIECES`] pieces (only on a device of task
    /// limits below Vulkan's required least), which the device cannot launch: it draws nothing,
    /// and the command buffer fails (not Metal's failure).
    pub past: bool,
}

impl ObjectFold {
    /// The fold's scratch as a direct draw uploads it (the pass's layout, through its pieces'
    /// commands): the region record at [`MeshArguments::OBJECT_RECORD`], the pieces at
    /// [`MeshArguments::OBJECT_COUNT`], and each piece's command from
    /// [`MeshArguments::OBJECT_COMMANDS`].
    pub fn bytes(&self) -> Vec<u8> {
        let mut words = self.record.to_vec();
        words.push(self.pieces);
        for _ in 0..self.pieces {
            words.extend(self.dispatch);
        }
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }
}

/// Metal's grid of an object draw of `groups` threadgroups of `threads` threads each (the module
/// docs): the threadgroups it runs along each axis, and the threads per grid its builtins report,
/// `groups · threads` wrapped to 32 bits. An indirect draw (`direct` false) runs every threadgroup;
/// a direct one those the wrapped threads fill, rounded up (measured on the M3 Max). `None` where
/// an axis has no thread.
pub fn metal_object_grid(
    groups: [u32; 3],
    threads: [u32; 3],
    direct: bool,
) -> Option<([u32; 3], [u32; 3])> {
    if threads.contains(&0) {
        return None;
    }
    let reported: [u32; 3] = std::array::from_fn(|i| groups[i].wrapping_mul(threads[i]));
    let runs = match direct {
        true => std::array::from_fn(|i| reported[i].div_ceil(threads[i])),
        false => groups,
    };
    Some((runs, reported))
}

/// The fold of an object draw of `groups` threadgroups of `threads` threads each (1 to 1024 an
/// axis), direct or indirect, on a task stage of `task`'s limits (the module docs); `None` when
/// `threads` does not pack. Each piece runs `nx` by `ny` workgroups of the threadgroups Metal runs
/// ([`metal_object_grid`]): `nx` the fewer of those and the axis and total limits, `ny` the
/// fewest rows that cover them within the `y` limit and the total.
pub fn fold_object_grid(
    groups: [u32; 3],
    threads: [u32; 3],
    task: MeshGridLimits,
    direct: bool,
) -> Option<ObjectFold> {
    let packed = crate::lower::dispatch::pack_nominal(threads)?;
    let (runs, reported) = metal_object_grid(groups, threads, direct)?;
    fold_runs(runs, reported, packed, 0, task)
}

/// The fold of a direct `drawMeshThreads` of `threads` threads in object threadgroups of `l`
/// threads each, on a task stage of `task`'s limits: the threadgroups the threads fill, rounded
/// up, whose last along an axis runs only the threads left, its record flagged
/// `lower::dispatch::NONUNIFORM` where one does (the partial object threadgroups the object
/// function runs, `RuntimeFact::PartialObjectThreadgroups`); `None` when `l` does not pack.
pub(crate) fn fold_object_threads(
    threads: [u32; 3],
    l: [u32; 3],
    task: MeshGridLimits,
) -> Option<ObjectFold> {
    let packed = crate::lower::dispatch::pack_nominal(l)?;
    let runs = std::array::from_fn(|i| threads[i].div_ceil(l[i]));
    let partial = (0..3).any(|i| !threads[i].is_multiple_of(l[i]));
    let flags = match partial {
        true => crate::lower::dispatch::NONUNIFORM,
        false => 0,
    };
    fold_runs(runs, threads, packed, flags, task)
}

/// The fold of `runs` threadgroups of Metal's (packed threads per threadgroup `packed`) whose
/// grid builtins report `reported` threads, its record's last word `flags`.
fn fold_runs(
    runs: [u32; 3],
    reported: [u32; 3],
    packed: u32,
    flags: u32,
    task: MeshGridLimits,
) -> Option<ObjectFold> {
    let none = |discarded, past| ObjectFold {
        record: [0, 0, 0, packed, 0, 0, 0, 0],
        dispatch: [0; 3],
        pieces: 0,
        discarded,
        past,
    };
    if runs.contains(&0) {
        return Some(none(false, false));
    }
    let n = runs.iter().map(|&c| u64::from(c)).product::<u64>();
    if n > u64::from(METAL_OBJECT_GRID_TOTAL) {
        return Some(none(true, false));
    }
    let n = n as u32;
    let total = object_total(task);
    let nx = n.min(task.axes[0]).min(total).max(1);
    let ny = n.div_ceil(nx).min(task.axes[1]).min(total / nx).max(1);
    let pieces = n.div_ceil(nx * ny);
    if pieces > MeshArguments::OBJECT_PIECES {
        return Some(none(false, true));
    }
    let [rx, ry, rz] = runs;
    let [tx, ty, tz] = reported;
    Some(ObjectFold {
        record: [rx, ry, rz, packed, tx, ty, tz, flags],
        dispatch: [nx, ny, 1],
        pieces,
        discarded: false,
        past: false,
    })
}

/// The task stage's total for a fold, at most 2^31 so that no division by it is the all-ones
/// integer's (`tests::owned_divisions`); a fold needs at most [`METAL_OBJECT_GRID_TOTAL`].
fn object_total(task: MeshGridLimits) -> u32 {
    task.total.min(1 << 31)
}

/// An arguments pass the consumer records before an indirect mesh draw on a device with
/// `VK_EXT_mesh_shader`: one invocation of [`mesh_arguments_module`] of the draw's form reads
/// Metal's arguments at `source` and writes, into a consumer-provided scratch of
/// [`MeshArguments::scratch_bytes`], the mesh form's `VkDrawMeshTasksIndirectCommandEXT`
/// (`DrawCommand::DrawMeshTasksIndirect` at offset 0 reads it; the unfolded object form's too) or
/// the folded object form's region record, draw count and commands
/// (`DrawCommand::DrawMeshTasksIndirectCount` reads them).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MeshArguments {
    pub source: IndirectRef,
    /// The module of the draw's first stage.
    pub form: MeshArgumentsForm,
    /// The range of the source buffer to bind at [`MeshArgumentsModule::SOURCE_BINDING`]: from
    /// `source.offset` aligned down to `minStorageBufferOffsetAlignment` (and to 4) through the
    /// 12 argument bytes.
    pub source_range: BufferRange,
    /// The push constant bytes (`COMPUTE`, offset 0): the arguments' byte offset inside
    /// `source_range`, then the packed threads per object threadgroup (0 but for the folded
    /// object form).
    pub push_constants: [u8; 8],
}

impl MeshArguments {
    /// Bytes of Metal's arguments (`MTLDispatchThreadgroupsIndirectArguments`).
    pub const ARGUMENT_BYTES: u32 = 12;
    /// The folded object form's region record's byte offset in the scratch (16-aligned).
    pub const OBJECT_RECORD: u32 = 0;
    /// The folded object form's draw count's byte offset in the scratch.
    pub const OBJECT_COUNT: u32 = 32;
    /// The folded object form's first command's byte offset in the scratch.
    pub const OBJECT_COMMANDS: u32 = 36;
    /// The folded object form's commands, the most pieces a draw takes: a piece of `nx` by `ny`
    /// workgroups, `ny` the rows of `nx` the task total holds, runs more than half of the 2^22
    /// threadgroups every task stage launches per draw (`maxTaskWorkGroupTotalCount`'s required
    /// least), so 256 pieces cover [`METAL_OBJECT_GRID_TOTAL`] on every device.
    pub const OBJECT_PIECES: u32 = 256;

    /// Bytes of the scratch the pass of `form` writes: the mesh and unfolded object forms'
    /// command, or the folded object form's region record, draw count and commands.
    pub const fn scratch_bytes(form: MeshArgumentsForm) -> u32 {
        match form {
            MeshArgumentsForm::Mesh | MeshArgumentsForm::Object => 12,
            MeshArgumentsForm::FoldedObject => Self::OBJECT_COMMANDS + 12 * Self::OBJECT_PIECES,
        }
    }

    /// The pass for Metal arguments at `source` of a draw whose first stage is `form`, with the
    /// source bound at an offset aligned to `alignment` (`minStorageBufferOffsetAlignment`) and,
    /// for the folded object form, `threads` threads per object threadgroup. `None` when
    /// `source.offset` is not a multiple of 4 (such a draw draws nothing), the range overflows,
    /// or the threads do not pack.
    pub fn new(
        source: IndirectRef,
        form: MeshArgumentsForm,
        alignment: u64,
        threads: [u32; 3],
    ) -> Option<Self> {
        if !source.offset.is_multiple_of(4) {
            return None;
        }
        let a = alignment.max(4);
        let start = source.offset - source.offset % a;
        let residual = u32::try_from(source.offset - start).ok()?;
        let end = source.offset.checked_add(u64::from(Self::ARGUMENT_BYTES))?;
        let packed = match form {
            MeshArgumentsForm::Mesh | MeshArgumentsForm::Object => 0,
            MeshArgumentsForm::FoldedObject => crate::lower::dispatch::pack_nominal(threads)?,
        };
        let mut push_constants = [0u8; 8];
        push_constants[..4].copy_from_slice(&residual.to_le_bytes());
        push_constants[4..].copy_from_slice(&packed.to_le_bytes());
        Some(MeshArguments {
            source,
            form,
            source_range: BufferRange {
                offset: start,
                range: end - start,
            },
            push_constants,
        })
    }
}

/// The crate-owned arguments module of one device and form, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeshArgumentsModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
    /// The rule its constants hold.
    pub rule: MeshGridRule,
}

impl MeshArgumentsModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The Metal indirect buffer, a storage buffer.
    pub const SOURCE_BINDING: u32 = 0;
    /// The consumer's scratch ([`MeshArguments::scratch_bytes`]), a storage buffer.
    pub const SCRATCH_BINDING: u32 = 1;
    /// The command buffer's `ExecutionStatus` buffer, a storage buffer.
    pub const STATUS_BINDING: u32 = 2;
    /// Bytes of the `COMPUTE` push-constant range at offset 0.
    pub const PUSH_BYTES: u32 = 8;
}

/// The binding symbol of the source buffer ([`MeshArgumentsModule::SOURCE_BINDING`]).
pub const SOURCE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
/// The binding symbol of the scratch buffer ([`MeshArgumentsModule::SCRATCH_BINDING`]).
pub const SCRATCH: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};
/// The binding symbol of the status buffer ([`MeshArgumentsModule::STATUS_BINDING`]).
pub const STATUS: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Synthetic(SyntheticBinding::ExecutionStatus),
    index: 0,
};
/// The push field of the source residual, as a descriptor slot's residual bind offset.
const RESIDUAL: PushField = PushField::SlotBindOffset(SlotRef {
    stage: Stage::Kernel,
    class: SlotClass::Buffer,
    index: 0,
});
/// The push field of the folded object form's packed threads per object threadgroup, at offset 4.
const PACKED_THREADS: PushField = PushField::SlotBindOffset(SlotRef {
    stage: Stage::Kernel,
    class: SlotClass::Buffer,
    index: 1,
});

/// A `Block { u32 words[]; }` storage buffer bound at `sym`.
fn words_buffer(b: &mut ModuleBuilder, sym: ResourceSlot, u32_: TypeId) -> BResult<GlobalId> {
    let rta = b.ty(Type::RuntimeArray {
        elem: u32_,
        stride: Some(4),
    })?;
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta])?;
    let block = b.ty(Type::Struct {
        members: [rta].into(),
        layout: Some(layout),
    })?;
    b.resource(sym, StorageClass::StorageBuffer, block)
}

fn u32_value(f: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = f.u32_const(v)?;
    f.module().const_value(c)
}

/// Whether `v`'s product is at most `total`, without overflow and without a division: a zero
/// count's product is 0, and otherwise the product is within where neither multiplication's high
/// word is set and its low word is at most `total` (`checked_mul`, as `bind::replay`'s).
fn total_within(f: &mut FunctionBuilder<'_>, v: [Value; 3], total: u32) -> BResult<Value> {
    let zero = u32_value(f, 0)?;
    let t = u32_value(f, total)?;
    let mut any_zero = f.binary(BinOp::IEqual, v[0], zero)?;
    for &c in &v[1..] {
        let z = f.binary(BinOp::IEqual, c, zero)?;
        any_zero = f.binary(BinOp::LogicalOr, any_zero, z)?;
    }
    let (xy, over_xy) = checked_mul(f, v[0], v[1], zero)?;
    let (xyz, over_xyz) = checked_mul(f, xy, v[2], zero)?;
    let over = f.binary(BinOp::LogicalOr, over_xy, over_xyz)?;
    let le = f.binary(BinOp::ULessThanEqual, xyz, t)?;
    let fits = f.unary(UnOp::LogicalNot, over)?;
    let fits = f.binary(BinOp::LogicalAnd, fits, le)?;
    f.binary(BinOp::LogicalOr, any_zero, fits)
}

/// Whether each of `v` is at most its axis of `limits`, and their product at most its total.
fn grid_within(
    f: &mut FunctionBuilder<'_>,
    v: [Value; 3],
    limits: MeshGridLimits,
) -> BResult<Value> {
    let mut within = total_within(f, v, limits.total)?;
    for (c, axis) in v.into_iter().zip(limits.axes) {
        let a = u32_value(f, axis)?;
        let le = f.binary(BinOp::ULessThanEqual, c, a)?;
        within = f.binary(BinOp::LogicalAnd, within, le)?;
    }
    Ok(within)
}

/// The `bool` constant `v`.
fn boolean(f: &mut FunctionBuilder<'_>, v: bool) -> BResult<Value> {
    let ty = f.module().ty(Type::Bool)?;
    let c = f.module().constant(Constant::Scalar {
        ty,
        bits: ConstBits(u64::from(v)),
    })?;
    f.module().const_value(c)
}

/// The mesh and unfolded object forms (the module docs): Metal's counts `v` into the scratch's
/// command where Metal draws them and the device launches them, else zero. Whether Metal's GPU
/// discards them (the unfolded object form's grid of more than [`METAL_OBJECT_GRID_TOTAL`]
/// threadgroups), and whether Metal draws them and the device cannot.
fn mesh_counts(
    f: &mut FunctionBuilder<'_>,
    rule: MeshGridRule,
    v: [Value; 3],
    scratch: Value,
) -> BResult<(Value, Value)> {
    let zero = u32_value(f, 0)?;
    let metal = match rule.metal {
        Some(limits) => grid_within(f, v, limits)?,
        None => total_within(f, v, METAL_OBJECT_GRID_TOTAL)?,
    };
    let device = grid_within(f, v, rule.device)?;
    let drawn = f.binary(BinOp::LogicalAnd, metal, device)?;
    for (axis, c) in v.into_iter().enumerate() {
        let i = u32_value(f, axis as u32)?;
        let count = f.select(drawn, c, zero)?;
        let out = f.access_chain(scratch, &[zero, i])?;
        f.store(out, count, MemAccess::default())?;
    }
    let discarded = match rule.metal {
        Some(_) => boolean(f, false)?,
        None => f.unary(UnOp::LogicalNot, metal)?,
    };
    let past = f.binary(BinOp::LogicalNotEqual, metal, drawn)?;
    Ok((discarded, past))
}

/// The folded object form ([`fold_object_grid`] of an indirect draw, step for step): Metal's counts
/// `v` of threadgroups of the `packed` threads fold into the scratch's region record, draw count
/// and commands. Whether Metal's GPU discards the grid ([`ObjectFold::discarded`]), and whether
/// the device cannot launch it ([`ObjectFold::past`]).
fn object_fold(
    f: &mut FunctionBuilder<'_>,
    u32_: TypeId,
    task: MeshGridLimits,
    (v, packed): ([Value; 3], Value),
    scratch: Value,
) -> BResult<(Value, Value)> {
    let zero = u32_value(f, 0)?;
    let one = u32_value(f, 1)?;
    let mut any_zero = f.binary(BinOp::IEqual, v[0], zero)?;
    for &c in &v[1..] {
        let z = f.binary(BinOp::IEqual, c, zero)?;
        any_zero = f.binary(BinOp::LogicalOr, any_zero, z)?;
    }
    // Metal's threads per grid, each wrapped to a `u32` (`metal_object_grid`: an indirect draw
    // runs every threadgroup).
    let mask = u32_value(f, (1 << crate::lower::dispatch::NOMINAL_AXIS_BITS) - 1)?;
    let mut per_grid = [zero; 3];
    for (axis, out) in per_grid.iter_mut().enumerate() {
        let shift = u32_value(f, crate::lower::dispatch::NOMINAL_AXIS_BITS * axis as u32)?;
        let t = f.binary(BinOp::ShiftRightLogical, packed, shift)?;
        let t = f.binary(BinOp::BitwiseAnd, t, mask)?;
        let t = f.binary(BinOp::IAdd, t, one)?;
        *out = f.binary(BinOp::IMul, v[axis], t)?;
    }
    // The threadgroups, at most `METAL_OBJECT_GRID_TOTAL`.
    let (xy, wraps_xy) = checked_mul(f, v[0], v[1], zero)?;
    let (n, wraps_n) = checked_mul(f, xy, v[2], zero)?;
    let mut discarded = f.binary(BinOp::LogicalOr, wraps_xy, wraps_n)?;
    let most = u32_value(f, METAL_OBJECT_GRID_TOTAL)?;
    let many = f.binary(BinOp::UGreaterThan, n, most)?;
    discarded = f.binary(BinOp::LogicalOr, discarded, many)?;
    // One piece's dispatch and the pieces, of at least one threadgroup (a zero count or a grid
    // past what runs draws nothing whatever they are).
    let n = f.glsl(GlslOp::UMax, &[n, one])?;
    let total = object_total(task);
    let axis_x = u32_value(f, task.axes[0].min(total).max(1))?;
    let nx = f.glsl(GlslOp::UMin, &[n, axis_x])?;
    let rows = div_ceil(f, n, nx)?;
    let axis_y = u32_value(f, task.axes[1].max(1))?;
    let total = u32_value(f, total)?;
    let room = f.binary(BinOp::UDiv, total, nx)?;
    let ny = f.glsl(GlslOp::UMin, &[rows, axis_y])?;
    let ny = f.glsl(GlslOp::UMin, &[ny, room])?;
    let ny = f.glsl(GlslOp::UMax, &[ny, one])?;
    let piece = f.binary(BinOp::IMul, nx, ny)?;
    let pieces = div_ceil(f, n, piece)?;
    let most = u32_value(f, MeshArguments::OBJECT_PIECES)?;
    let many = f.binary(BinOp::UGreaterThan, pieces, most)?;
    let over = f.binary(BinOp::LogicalOr, discarded, many)?;
    let none = f.binary(BinOp::LogicalOr, any_zero, over)?;
    let runs = f.unary(UnOp::LogicalNot, none)?;
    let some = f.unary(UnOp::LogicalNot, any_zero)?;
    let discarded = f.binary(BinOp::LogicalAnd, some, discarded)?;
    let kept = f.unary(UnOp::LogicalNot, discarded)?;
    let past = f.binary(BinOp::LogicalAnd, some, many)?;
    let past = f.binary(BinOp::LogicalAnd, past, kept)?;
    // The region record, the count and the commands.
    let word = |f: &mut FunctionBuilder<'_>, at: u32, value: Value| -> BResult<()> {
        let i = u32_value(f, at)?;
        let p = f.access_chain(scratch, &[zero, i])?;
        f.store(p, value, MemAccess::default())
    };
    let record = MeshArguments::OBJECT_RECORD / 4;
    for (i, c) in v.into_iter().enumerate() {
        let c = f.select(runs, c, zero)?;
        word(f, record + i as u32, c)?;
    }
    word(f, record + 3, packed)?;
    for (i, t) in per_grid.into_iter().enumerate() {
        let t = f.select(runs, t, zero)?;
        word(f, record + 4 + i as u32, t)?;
    }
    word(f, record + 7, zero)?;
    let count = f.select(runs, pieces, zero)?;
    word(f, MeshArguments::OBJECT_COUNT / 4, count)?;
    let nx = f.select(runs, nx, zero)?;
    let ny = f.select(runs, ny, zero)?;
    let nz = f.select(runs, one, zero)?;
    let first = u32_value(f, MeshArguments::OBJECT_COMMANDS / 4)?;
    let end = u32_value(f, MeshArguments::OBJECT_PIECES)?;
    for_range(f, u32_, (zero, end), move |f, i| {
        let zero = u32_value(f, 0)?;
        let three = u32_value(f, 3)?;
        let at = f.binary(BinOp::IMul, i, three)?;
        let at = f.binary(BinOp::IAdd, first, at)?;
        for (k, value) in [nx, ny, nz].into_iter().enumerate() {
            let k = u32_value(f, k as u32)?;
            let w = f.binary(BinOp::IAdd, at, k)?;
            let p = f.access_chain(scratch, &[zero, w])?;
            f.store(p, value, MemAccess::default())?;
        }
        Ok(())
    })?;
    Ok((discarded, past))
}

/// The arguments module of `form` for `rule` as a typed module, before serialization: the module
/// [`mesh_arguments_module`] serializes, which the CPU interpreter runs. Its bindings are the
/// symbols [`SOURCE`], [`SCRATCH`] and [`STATUS`]; its push fields are the source residual at
/// offset 0 and, for the folded object form, the packed threads per object threadgroup at
/// offset 4.
pub fn typed_module(form: MeshArgumentsForm, rule: MeshGridRule) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let source = words_buffer(&mut b, SOURCE, u32_)?;
    let scratch = words_buffer(&mut b, SCRATCH, u32_)?;
    let status = words_buffer(&mut b, STATUS, u32_)?;
    let residual = b.push_member(RESIDUAL, u32_, 0)?;
    let packed = match form {
        MeshArgumentsForm::Mesh | MeshArgumentsForm::Object => None,
        MeshArgumentsForm::FoldedObject => Some(b.push_member(PACKED_THREADS, u32_, 4)?),
    };
    let push = b.push_global()?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = u32_value(&mut f, 0)?;
    let two = u32_value(&mut f, 2)?;
    let push_value = f.module().global_value(push)?;
    let member = u32_value(&mut f, residual.index)?;
    let residual_ptr = f.access_chain(push_value, &[member])?;
    let residual_bytes = f.load(residual_ptr, MemAccess::default())?;
    let first = f.binary(BinOp::ShiftRightLogical, residual_bytes, two)?;
    let source_value = f.module().global_value(source)?;
    let len = f.array_length(source_value, 0)?;
    let mut v = [zero; 3];
    for (axis, out) in v.iter_mut().enumerate() {
        let i = u32_value(&mut f, axis as u32)?;
        let index = f.binary(BinOp::IAdd, first, i)?;
        // Contained: an index past the bound range reads word 0 and yields 0.
        let inb = f.binary(BinOp::ULessThan, index, len)?;
        let safe = f.select(inb, index, zero)?;
        let p = f.access_chain(source_value, &[zero, safe])?;
        let loaded = f.load(p, MemAccess::default())?;
        *out = f.select(inb, loaded, zero)?;
    }
    let scratch_value = f.module().global_value(scratch)?;
    let (discarded, past) = match packed {
        None => mesh_counts(&mut f, rule, v, scratch_value)?,
        Some(packed) => {
            let member = u32_value(&mut f, packed.index)?;
            let p = f.access_chain(push_value, &[member])?;
            let packed = f.load(p, MemAccess::default())?;
            object_fold(&mut f, u32_, rule.device, (v, packed), scratch_value)?
        }
    };
    let status_value = f.module().global_value(status)?;
    let word = u32_value(&mut f, StatusWord::Status.index())?;
    let discard_bit = u32_value(&mut f, StatusBits::MESH_GRID_DISCARDED.0)?;
    let past_bit = u32_value(&mut f, StatusBits::MESH_GRID_PAST_LIMITS.0)?;
    let discard_bit = f.select(discarded, discard_bit, zero)?;
    let past_bit = f.select(past, past_bit, zero)?;
    let bits = f.binary(BinOp::BitwiseOr, discard_bit, past_bit)?;
    let set = f.binary(BinOp::INotEqual, bits, zero)?;
    let then: ArmFn<'_, ()> = Box::new(move |fb, _| {
        let p = fb.access_chain(status_value, &[zero, word])?;
        fb.atomic(AtomicOp::Or, p, Scope::Device, Semantics::RELAXED, bits)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), set, then, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )?;
    b.finalize()
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(MeshArgumentsModule::SET),
        Symbol::Binding(s) if *s == SOURCE => Some(MeshArgumentsModule::SOURCE_BINDING),
        Symbol::Binding(s) if *s == SCRATCH => Some(MeshArgumentsModule::SCRATCH_BINDING),
        Symbol::Binding(s) if *s == STATUS => Some(MeshArgumentsModule::STATUS_BINDING),
        _ => None,
    }
}

/// The arguments module of `device` for draws whose first stage is `form`: built with the typed
/// builder, checked by the owned verifier against the device's profile (requirements included),
/// serialized, and numbered with the fixed bindings of [`MeshArgumentsModule`].
pub fn mesh_arguments_module(
    device: &Device,
    form: MeshArgumentsForm,
) -> Result<MeshArgumentsModule, Error> {
    let profile = device.profile();
    let rule = MeshGridRule::of(form, &profile.fields().limits.mesh);
    let m = typed_module(form, rule)?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the mesh arguments pass needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(MeshArgumentsModule {
        spirv: words.into(),
        requirements: r.requirements,
        rule,
    })
}

#[cfg(test)]
mod tests;
