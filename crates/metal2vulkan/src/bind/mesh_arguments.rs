//! The arguments pass of an indirect mesh draw on a device with `VK_EXT_mesh_shader`.
//!
//! `drawMeshThreadgroupsWithIndirectBuffer:` reads its group counts (an
//! `MTLDispatchThreadgroupsIndirectArguments`) as it runs. `vkCmdDrawMeshTasksIndirectEXT` requires
//! every count within the device's task or mesh limits, so the draw cannot read Metal's arguments
//! in place: the consumer records an arguments pass first, which reads Metal's three words and
//! writes what the draw reads into a scratch, by the rules below ([`MeshArguments`] names the
//! folded object form's words in it).
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
//! **The mesh form's rule** (Metal's limits, its constants, and the device's for the first
//! stage, [`MeshGridLimits`]): where Metal draws nothing, the pass writes zero counts, which draw
//! nothing. Where Metal draws and the counts lie within the
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
//! threadgroup. The pass writes, into the scratch:
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
//! **Synchronization** (consumer): the source write → `SHADER_READ` in `COMPUTE_SHADER` before
//! the pass; the scratch `SHADER_WRITE` → `INDIRECT_COMMAND_READ` in `DRAW_INDIRECT`, and for the
//! folded object form `SHADER_READ` in `TASK_SHADER`, between the pass and the draw.

use crate::bind::instantiate::{BufferRange, IndirectRef};

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
/// integer's; a fold needs at most [`METAL_OBJECT_GRID_TOTAL`].
fn object_total(task: MeshGridLimits) -> u32 {
    task.total.min(1 << 31)
}

/// An arguments pass the consumer records before an indirect mesh draw on a device with
/// `VK_EXT_mesh_shader` (the module docs): it reads Metal's arguments at `source` and writes, into
/// a scratch, the mesh form's `VkDrawMeshTasksIndirectCommandEXT`
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
    /// The range of the source buffer the pass reads: from `source.offset` aligned down to `minStorageBufferOffsetAlignment` (and to 4) through the
    /// 12 argument bytes.
    pub source_range: BufferRange,
    /// The push constant bytes (`COMPUTE`, offset 0): the arguments' byte offset inside
    /// `source_range`, then the packed threads per object threadgroup (0 but for the folded
    /// object form).
    pub push_constants: [u8; 8],
}

impl MeshArguments {
    /// The folded object form's draw count's byte offset in the scratch.
    pub const OBJECT_COUNT: u32 = 32;
    /// The folded object form's first command's byte offset in the scratch.
    pub const OBJECT_COMMANDS: u32 = 36;
    /// The folded object form's commands, the most pieces a draw takes: a piece of `nx` by `ny`
    /// workgroups, `ny` the rows of `nx` the task total holds, runs more than half of the 2^22
    /// threadgroups every task stage launches per draw (`maxTaskWorkGroupTotalCount`'s required
    /// least), so 256 pieces cover [`METAL_OBJECT_GRID_TOTAL`] on every device.
    pub const OBJECT_PIECES: u32 = 256;
}

#[cfg(test)]
mod tests;
