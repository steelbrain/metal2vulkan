//! A direct dispatch takes what its pipeline state planned for another grid of its class
//! (`pipeline::GridClass`, `Planned::grids`), as a guest's fix-up stream of a thread for each
//! element dispatches one width after another.
//!
//! [`every_grid_of_a_class_plans_once_and_answers_as_metal`]: two kernels, each dispatched in one
//! encoder over four classes in turn, six grids of each, every grid its own: one that observes no
//! threadgroup partition (its ragged grids on the flat variant, `compute::flat_direct`), of
//! ragged grids of two rows and of one row, whole grids of threads and of threadgroups; and one
//! that reads its `thread_index_in_threadgroup` (its ragged grids on the partial-threadgroups
//! variant, `compute::partial_direct`), of ragged one-row grids at two threadgroup sizes, whole
//! grids of threads and threadgroups. Every dispatch writes its own slice of one buffer, read back
//! word for word against Metal's answer, and every dispatch past the first of its class takes what
//! that one planned (`WorkCounts::dispatches_planned_across_grids`).
//!
//! [`shapes_and_classes_in_turn_each_plan_once`]: the flat kernel's output bound from buffers of
//! three length classes, three bind state shapes, by grids of two classes, dispatched in turn: six
//! plans, which the state keeps all of (`pipeline::PLANNED`), so each plans once.
//!
//! [`a_grid_past_the_device_limits_meets_the_refusal_a_fresh_state_meets`]: a grid of a class
//! planned before whose group counts no device holds is refused as a state that planned nothing
//! refuses it, since the plan's check of the grid against the limits runs at every dispatch.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The kernels are LLVM text written here
//! (`llvm-as`).

use super::blit_holds::{device, Scratch, WAIT};
use super::opencl_mode::{words, words_buffer};
use crate::{
    Buffer, CommandBufferStatus, ComputeCommandEncoder, ComputePipelineState, Device, Error,
    PipelineOptions, Size,
};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};

/// `flat2d(device uint *out [[buffer(0)]], constant uint &k [[buffer(1)]], uint2 gid
/// [[thread_position_in_grid]], uint2 size [[threads_per_grid]])`: `out[gid.y * size.x + gid.x] =
/// 3 * (gid.y * size.x + gid.x) + k`. It observes no threadgroup partition.
pub(super) const FLAT: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @flat2d(ptr addrspace(1) noundef %out, ptr addrspace(2) noundef %k, <2 x i32> noundef %gid, <2 x i32> noundef %size) {
entry:
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %w = extractelement <2 x i32> %size, i64 0
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %kv = load i32, ptr addrspace(2) %k, align 4
  %t = mul i32 %i, 3
  %v = add i32 %t, %kv
  %j = zext i32 %i to i64
  %p = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %j
  store i32 %v, ptr addrspace(1) %p, align 4
  ret void
}

!llvm.module.flags = !{!0, !1}
!air.kernel = !{!2}
!air.version = !{!9}
!air.language_version = !{!10}
!0 = !{i32 7, !"air.max_device_buffers", i32 31}
!1 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!2 = !{ptr @flat2d, !3, !4}
!3 = !{}
!4 = !{!5, !6, !7, !8}
!5 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!6 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"k"}
!7 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
!8 = !{i32 3, !"air.threads_per_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"size"}
!9 = !{i32 2, i32 8, i32 0}
!10 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// `index(device uint *out [[buffer(0)]], constant uint &k [[buffer(1)]], uint gid
/// [[thread_position_in_grid]], uint t [[thread_index_in_threadgroup]])`: `out[gid] = t << 16 |
/// k`. It observes its threadgroup partition.
const INDEX: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @index(ptr addrspace(1) noundef %out, ptr addrspace(2) noundef %k, i32 noundef %gid, i32 noundef %t) {
entry:
  %kv = load i32, ptr addrspace(2) %k, align 4
  %hi = shl i32 %t, 16
  %v = or i32 %hi, %kv
  %j = zext i32 %gid to i64
  %p = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %j
  store i32 %v, ptr addrspace(1) %p, align 4
  ret void
}

!llvm.module.flags = !{!0, !1}
!air.kernel = !{!2}
!air.version = !{!9}
!air.language_version = !{!10}
!0 = !{i32 7, !"air.max_device_buffers", i32 31}
!1 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!2 = !{ptr @index, !3, !4}
!3 = !{}
!4 = !{!5, !6, !7, !8}
!5 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!6 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"k"}
!7 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!8 = !{i32 3, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
!9 = !{i32 2, i32 8, i32 0}
!10 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// The words of the output each dispatch has to itself.
const SLICE: usize = 512;

/// What a word holds before a dispatch writes it.
const UNWRITTEN: u32 = u32::MAX;

/// The grids of each class a kernel is dispatched over, in turn.
const ROUNDS: usize = 6;

/// A dispatch of the test: `dispatchThreads:` of `threads`, or `dispatchThreadgroups:` of
/// `groups`, in threadgroups of `local`.
#[derive(Clone, Copy, Debug)]
enum Grid {
    Threads([u32; 2], [u32; 2]),
    Groups(u32, u32),
}

impl Grid {
    /// The grid's threads along `x` and `y`.
    fn threads(self) -> [u32; 2] {
        match self {
            Grid::Threads(threads, _) => threads,
            Grid::Groups(groups, local) => [groups * local, 1],
        }
    }

    /// The threadgroup's threads along `x` and `y`.
    fn local(self) -> [u32; 2] {
        match self {
            Grid::Threads(_, local) => local,
            Grid::Groups(_, local) => [local, 1],
        }
    }
}

/// The state of the kernel `name` of `text` on `device`.
pub(super) fn state(device: &Device, text: &str, name: &str) -> ComputePipelineState {
    let air = metal2vulkan_devtools::llvm_assemble_text(text).unwrap();
    let library = metal2vulkan::library::Library::from_air(&air).unwrap();
    let function = device
        .new_library_with_compile_library(library)
        .new_function_with_name(name)
        .unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// Dispatch `grid` of `state` in `encoder`, its output bound at `slice`, `k` its constant.
fn dispatch(
    encoder: &mut ComputeCommandEncoder,
    out: &Buffer,
    (slice, k): (usize, u32),
    grid: Grid,
) -> Result<(), Error> {
    encoder.set_buffer(Some(out), (slice * SLICE * 4) as u64, 0)?;
    encoder.set_bytes(&k.to_le_bytes(), 1)?;
    let [lx, ly] = grid.local();
    let local = Size::new(u64::from(lx), u64::from(ly), 1);
    match grid {
        Grid::Threads([x, y], _) => {
            encoder.dispatch_threads(Size::new(u64::from(x), u64::from(y), 1), local)
        }
        Grid::Groups(groups, _) => {
            encoder.dispatch_threadgroups(Size::new(u64::from(groups), 1, 1), local)
        }
    }
}

/// What Metal leaves in a slice for `grid` with constant `k`: `word(i, x)` at each of its threads,
/// `i` the thread's position counted along rows and `x` its column, and nothing past them.
fn metal(grid: Grid, word: impl Fn(u32, u32) -> u32) -> Vec<u32> {
    let [w, h] = grid.threads();
    (0..SLICE as u32)
        .map(|i| match i < w * h {
            true => word(i, i % w),
            false => UNWRITTEN,
        })
        .collect()
}

/// Dispatch `grids` of the kernel `name` of `text` in one encoder, each into its own slice with
/// its own constant, and check every slice against Metal's answer (`word` of the grid, the
/// thread's position along rows, its column and the constant) and the dispatches that took what
/// another grid of their class planned: every one past the first of its class.
fn run_classes(
    device: &Device,
    (text, name): (&str, &str),
    grids: &[Grid],
    classes: usize,
    word: impl Fn(Grid, u32, u32, u32) -> u32,
) {
    let state = state(device, text, name);
    let out = words_buffer(device, &vec![UNWRITTEN; SLICE * grids.len()]);
    let across = || device.work_counts().dispatches_planned_across_grids;
    let before = across();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    for (slice, &grid) in grids.iter().enumerate() {
        let k = slice as u32 + 1;
        dispatch(&mut encoder, &out, (slice, k), grid)
            .unwrap_or_else(|e| panic!("{name}: dispatch {slice} of {grid:?}: {e:?}"));
    }
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let written = words(&out);
    for (slice, &grid) in grids.iter().enumerate() {
        let k = slice as u32 + 1;
        let want = metal(grid, |i, x| word(grid, i, x, k));
        let got = &written[slice * SLICE..(slice + 1) * SLICE];
        assert_eq!(got, want, "{name}: dispatch {slice} of {grid:?}");
    }
    assert_eq!(
        across() - before,
        (grids.len() - classes) as u64,
        "{name}: the dispatches that took what another grid of their class planned"
    );
}

/// The two kernels over their four classes each, a grid of every class in turn.
#[test]
fn every_grid_of_a_class_plans_once_and_answers_as_metal() {
    let scratch = Scratch::new("grid-classes");
    let device = device(&scratch.0);
    let mut flat = Vec::new();
    let mut index = Vec::new();
    for r in 0..ROUNDS as u32 {
        // Ragged grids of three rows in threadgroups of two, and of one row; whole grids of
        // threads and of threadgroups.
        flat.push(Grid::Threads([5 + 7 * r, 3], [16, 2]));
        flat.push(Grid::Threads([1 + 23 * r, 1], [64, 1]));
        flat.push(Grid::Threads([64 * (r + 1), 1], [64, 1]));
        flat.push(Grid::Groups(r + 1, 64));
        // Ragged one-row grids in threadgroups of 64 and of 32; whole grids of threads and of
        // threadgroups.
        index.push(Grid::Threads([3 + 41 * r, 1], [64, 1]));
        index.push(Grid::Threads([64 * (r + 1), 1], [64, 1]));
        index.push(Grid::Threads([5 + 37 * r, 1], [32, 1]));
        index.push(Grid::Groups(r + 2, 32));
    }
    run_classes(&device, (FLAT, "flat2d"), &flat, 4, |_, i, _, k| 3 * i + k);
    run_classes(&device, (INDEX, "index"), &index, 4, |grid, _, x, k| {
        (x % grid.local()[0]) << 16 | k
    });
}

/// A grid of the flat kernel's class of ragged grids of more than one row, planned before, whose
/// threads no device's group counts hold (`2^32 - 1` along each axis): refused, and with the very
/// error a state that planned nothing gives it.
#[test]
fn a_grid_past_the_device_limits_meets_the_refusal_a_fresh_state_meets() {
    let scratch = Scratch::new("grid-classes-limits");
    let device = device(&scratch.0);
    let past = |state: &ComputePipelineState, first: Option<Grid>| {
        let out = words_buffer(&device, &vec![UNWRITTEN; SLICE * 2]);
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(state).unwrap();
        if let Some(first) = first {
            dispatch(&mut encoder, &out, (0, 1), first).unwrap();
        }
        encoder
            .set_buffer(Some(&out), (SLICE * 4) as u64, 0)
            .unwrap();
        encoder.set_bytes(&2u32.to_le_bytes(), 1).unwrap();
        let most = u64::from(u32::MAX);
        let refused = encoder
            .dispatch_threads(Size::new(most, most, most), Size::new(16, 2, 1))
            .expect_err("a grid past every device's group counts");
        encoder.end_encoding().unwrap();
        format!("{refused:?}")
    };
    let planned = state(&device, FLAT, "flat2d");
    let across = || device.work_counts().dispatches_planned_across_grids;
    let before = across();
    let after_plan = past(&planned, Some(Grid::Threads([5, 3], [16, 2])));
    assert_eq!(across(), before, "a grid past the limits took a kept plan");
    let fresh = past(&state(&device, FLAT, "flat2d"), None);
    assert_eq!(after_plan, fresh, "the refusals of a grid past the limits");
    assert!(
        fresh.contains("GroupCount"),
        "a grid past the limits refused otherwise: {fresh}"
    );
}

/// The bind state shapes [`shapes_and_classes_in_turn_each_plan_once`] dispatches in turn.
const SHAPES: usize = 3;

/// The flat kernel's output bound from buffers of [`SHAPES`] length classes (`BindState::shape`'s
/// `length_class`), each its own bind state shape, by ragged and whole one-row grids, two classes
/// ([`GridClass`](crate::pipeline::GridClass)): six plans, dispatched in turn [`ROUNDS`] times in
/// one encoder, every grid its own. Each plans at its first dispatch alone, and every later
/// dispatch takes what its shape and class planned (`WorkCounts::dispatches_planned_as_before`),
/// since the state keeps all six (`pipeline::PLANNED`): a state that kept four would push each
/// out before its turn came again, and plan every dispatch. Every dispatch writes its own slice,
/// word for word Metal's.
#[test]
fn shapes_and_classes_in_turn_each_plan_once() {
    let scratch = Scratch::new("grid-classes-shapes");
    let device = device(&scratch.0);
    let state = state(&device, FLAT, "flat2d");
    let mut grids = Vec::new();
    for r in 0..ROUNDS as u32 {
        for grid in [
            Grid::Threads([1 + 23 * r, 1], [64, 1]),
            Grid::Threads([64 * (r + 1), 1], [64, 1]),
        ] {
            grids.extend((0..SHAPES).map(|shape| (shape, grid)));
        }
    }
    let plans = 2 * SHAPES;
    // Lengths of one, two and four times the words every dispatch needs: three length classes.
    let outs: Vec<Buffer> = (0..SHAPES)
        .map(|shape| words_buffer(&device, &vec![UNWRITTEN; (SLICE * grids.len()) << shape]))
        .collect();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    let mut took = Vec::new();
    for (slice, &(shape, grid)) in grids.iter().enumerate() {
        let before = device.work_counts().dispatches_planned_as_before;
        dispatch(&mut encoder, &outs[shape], (slice, slice as u32 + 1), grid)
            .unwrap_or_else(|e| panic!("dispatch {slice} of {grid:?}: {e:?}"));
        took.push(device.work_counts().dispatches_planned_as_before - before);
    }
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let want: Vec<u64> = (0..grids.len()).map(|at| u64::from(at >= plans)).collect();
    assert_eq!(
        took, want,
        "the dispatches that took what their shape and class planned"
    );
    let written: Vec<Vec<u32>> = outs.iter().map(words).collect();
    for (slice, &(shape, grid)) in grids.iter().enumerate() {
        let k = slice as u32 + 1;
        let want = metal(grid, |i, _| 3 * i + k);
        let got = &written[shape][slice * SLICE..(slice + 1) * SLICE];
        assert_eq!(
            got, want,
            "dispatch {slice} of {grid:?} into the output of shape {shape}"
        );
    }
}
