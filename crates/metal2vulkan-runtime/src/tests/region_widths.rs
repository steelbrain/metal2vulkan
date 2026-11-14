//! A kernel that observes its threadgroup partition, dispatched as a program that sizes each
//! threadgroup to its grid does: `dispatchThreads` of `n` threads in threadgroups of `min(n, 64)`,
//! for `n` from 1 to 128, twice. Each dispatch keeps its exact split (the kernel reads
//! `thread_index_in_threadgroup`, so no flat variant runs it): a grid of up to 64 threads is one
//! region of its own width, and a wider one an interior of 64 and an edge of the remainder. That is
//! 128 region keys of one variant, each a `VkPipeline` of its own: the second pass must make none
//! of them again.
//!
//! The stream is the shape of a guest's run on an RX 7900 XTX (an index fix-up kernel at every
//! width from 1 to 64, with and without an edge), whose variant made each of its keys about 6,300
//! times in 146 s while its memo kept 32.
//!
//! Printed (`REGION-WIDTHS`): the `VkPipeline`s each pass made, each pass's wall time, and the
//! process's memory before and after the first pass (`spares_bench::process_memory`).
//!
//! [`two_kernels_past_the_device_bound_make_a_key_again_only_once_it_is_let_go`]: the device's
//! bound across every variant's region pipelines (`pipeline/region_ledger.rs`), a released
//! variant's kept instance included.
//!
//! [`a_full_ledger_s_settle_looks_at_the_keys_it_let_go_alone`]: what the kept variants' settle
//! after each key a full ledger lets go looks at. Printed (`KEPT-PRUNE`): the keys its settles
//! asked the ledger of, and the dispatch call's host time, the median and the least, in a release
//! build the A/B of the settle's cost.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The module is LLVM text written here
//! (`llvm-as`).

use super::blit_holds::{device, Scratch, WAIT};
use super::opencl_mode::{words, words_buffer};
use super::spares_bench::process_memory;
use super::uniform_bind_offsets::idle;
use crate::device::lock;
use crate::pipeline::kept::prune_looks;
use crate::{CommandBufferStatus, ComputePipelineState, Device, PipelineOptions, Size};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use std::ops::RangeInclusive;
use std::time::{Duration, Instant};

/// The widest threadgroup the stream asks for.
const WIDEST: u32 = 64;

/// The grids of a pass: `1..=GRIDS` threads.
const GRIDS: u32 = 128;

/// The region keys a pass takes: a single region at each width from 1 to [`WIDEST`], the interior
/// of [`WIDEST`] of a ragged grid, and its edge at each remainder from 1 to `WIDEST - 1`.
const KEYS: u64 = 2 * WIDEST as u64;

/// What a word holds before a dispatch writes it.
const UNWRITTEN: u32 = u32::MAX;

/// `widths(device uint *out [[buffer(0)]], uint gid [[thread_position_in_grid]], uint index
/// [[thread_index_in_threadgroup]])`: `out[gid] = index`.
pub(super) const KERNEL: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
target triple = \"air64_v28-apple-macosx26.0.0\"\n\n\
define void @widths(ptr addrspace(1) noundef %out, i32 noundef %gid, i32 noundef %index) {\n\
entry:\n\
  %at = zext i32 %gid to i64\n\
  %p = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %at\n\
  store i32 %index, ptr addrspace(1) %p, align 4\n\
  ret void\n\
}\n\n\
!llvm.module.flags = !{!0, !1}\n!air.kernel = !{!2}\n!air.version = !{!8}\n!air.language_version = !{!9}\n\
!0 = !{i32 7, !\"air.max_device_buffers\", i32 31}\n\
!1 = !{i32 7, !\"air.max_threadgroup_buffers\", i32 31}\n\
!2 = !{ptr @widths, !3, !4}\n!3 = !{}\n!4 = !{!5, !6, !7}\n\
!5 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}\n\
!6 = !{i32 1, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"gid\"}\n\
!7 = !{i32 2, !\"air.thread_index_in_threadgroup\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"index\"}\n\
!8 = !{i32 2, i32 8, i32 0}\n!9 = !{!\"Metal\", i32 4, i32 0, i32 0}\n";

/// [`KERNEL`]'s store, and the store of a kernel otherwise alike that writes `index + 1`.
/// The text's lines start unindented: a string continuation drops each line's leading spaces.
const STORE: &str = "store i32 %index, ptr addrspace(1) %p, align 4\n";
const STORE_PLUS_ONE: &str =
    "%plus = add i32 %index, 1\nstore i32 %plus, ptr addrspace(1) %p, align 4\n";

/// The state of `widths`, [`KERNEL`] or the kernel of `kernel`'s text.
pub(super) fn state(device: &Device, kernel: &str) -> ComputePipelineState {
    let air = metal2vulkan_devtools::llvm_assemble_text(kernel).unwrap();
    let library = metal2vulkan::library::Library::from_air(&air).unwrap();
    let function = device
        .new_library_with_compile_library(library)
        .new_function_with_name("widths")
        .unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// Dispatch `n` threads in threadgroups of `min(n, WIDEST)`, wait, and check what Metal writes.
fn run(device: &Device, state: &ComputePipelineState, n: u32) {
    dispatch(device, state, (n, n.min(WIDEST)), 0);
}

/// Dispatch `n` threads in threadgroups of `width`, wait, and check what Metal writes: `gid %
/// width + plus` at each of the grid's words, and nothing past them (at least [`GRIDS`] words).
/// The dispatch call's host time.
fn dispatch(
    device: &Device,
    state: &ComputePipelineState,
    (n, width): (u32, u32),
    plus: u32,
) -> Duration {
    let length = GRIDS.max(n);
    let out = words_buffer(device, &vec![UNWRITTEN; length as usize]);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    let threads = Size::new(u64::from(n), 1, 1);
    let group = Size::new(u64::from(width), 1, 1);
    let called = Instant::now();
    encoder.dispatch_threads(threads, group).unwrap();
    let call = called.elapsed();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let written = words(&out);
    let want: Vec<u32> = (0..length)
        .map(|gid| match gid < n {
            true => gid % width + plus,
            false => UNWRITTEN,
        })
        .collect();
    assert_eq!(written, want, "{n} threads in threadgroups of {width}");
    call
}

/// Two passes of the stream: the first makes each region key once, the second none.
#[test]
fn a_kernel_dispatched_at_every_width_makes_each_region_pipeline_once() {
    let scratch = Scratch::new("region-widths");
    let device = device(&scratch.0);
    let state = state(&device, KERNEL);
    let made = || device.work_counts().variant_pipelines_made;
    let evicted = || device.work_counts().region_pipelines_evicted;
    let evicted_before = evicted();
    let empty = process_memory();
    let mut passes = Vec::new();
    let mut times = Vec::new();
    let mut full = None;
    for _ in 0..2 {
        let before = made();
        let started = Instant::now();
        for n in 1..=GRIDS {
            run(&device, &state, n);
        }
        times.push(started.elapsed());
        passes.push(made() - before);
        full.get_or_insert_with(process_memory);
    }
    println!(
        "REGION-WIDTHS {}: {GRIDS} grids in threadgroups of up to {WIDEST}, twice: VkPipelines \
         made {passes:?}, evicted {}, passes took {times:?}; memory before {empty:?}, after the \
         first pass {full:?}",
        device.name(),
        evicted() - evicted_before
    );
    assert!(
        passes[0] <= KEYS,
        "the first pass made {} VkPipelines for {KEYS} keys",
        passes[0]
    );
    assert_eq!(passes[1], 0, "the second pass made VkPipelines again");
    assert_eq!(
        evicted(),
        evicted_before,
        "the variant's memo let a region pipeline go"
    );
}

/// The device's bound in the two kernels' test: one kernel's 64 keys and half the other's.
const BOUND: usize = 96;

/// Two kernels' region pipelines under a device bound of [`BOUND`] (`region_ledger.rs`), each
/// pass a kernel's grids of `n` threads in threadgroups of `n`, one key each:
///
/// - the first kernel's 64 (widths 1 to 64), then the second's 64, which let the first's 32 least
///   recently used go, and the second's again, which makes none;
/// - the first released, its kept instance holds the 32 the ledger lists; 16 more of the second
///   (widths 65 to 80) let 16 of those go, which the kept instance no longer holds once the next
///   lookup settles it;
/// - the first made again takes back its kept 16 and makes the 48 others, letting as many of the
///   second's go.
///
/// After every dispatch the ledger lists no more than the bound, and it counts each key it let go.
#[test]
fn two_kernels_past_the_device_bound_make_a_key_again_only_once_it_is_let_go() {
    let scratch = Scratch::new("region-ledger");
    let device = device(&scratch.0);
    let shared = device.shared();
    shared.region_ledger.bound_at(BOUND);
    let plus_one = KERNEL.replace(STORE, STORE_PLUS_ONE);
    assert_ne!(plus_one, KERNEL);
    let counts = || device.work_counts();
    // `VkPipeline`s made and keys let go by a pass of `state`'s `grids`, its kernel writing
    // `index + plus`.
    let pass = |state: &ComputePipelineState, plus: u32, grids: RangeInclusive<u32>| {
        let before = counts();
        for n in grids {
            dispatch(&device, state, (n, n), plus);
            let listed = shared.region_ledger.listed();
            assert!(listed <= BOUND, "the ledger lists {listed} past {BOUND}");
        }
        let after = counts();
        [
            after.variant_pipelines_made - before.variant_pipelines_made,
            after.region_pipelines_evicted - before.region_pipelines_evicted,
        ]
    };
    let first = state(&device, KERNEL);
    let second = state(&device, &plus_one);
    assert_eq!(pass(&first, 0, 1..=64), [64, 0], "the first kernel");
    assert_eq!(
        pass(&second, 1, 1..=64),
        [64, 32],
        "the second kernel past the bound"
    );
    assert_eq!(pass(&second, 1, 1..=64), [0, 0], "the second kernel again");

    // The first released whole: its objects and the 32 region pipelines the ledger lists come
    // to the kept variants once their last holders let go.
    drop(first);
    idle(&device);
    let until = Instant::now() + WAIT;
    while lock(&shared.kept).held() != (1, 32) {
        assert!(
            Instant::now() < until,
            "the kept variants hold {:?} after {WAIT:?}",
            lock(&shared.kept).held()
        );
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert_eq!(
        pass(&second, 1, 65..=80),
        [16, 16],
        "the second kernel's wider threadgroups"
    );
    assert_eq!(
        lock(&shared.kept).held(),
        (1, 16),
        "the kept instance after the ledger let 16 of its region pipelines go"
    );

    let taken_back = counts().variants_taken_back;
    let first = state(&device, KERNEL);
    assert_eq!(counts().variants_taken_back - taken_back, 1);
    assert_eq!(
        pass(&first, 0, 1..=64),
        [48, 48],
        "the first kernel made again"
    );
    assert_eq!(shared.region_ledger.listed(), BOUND);
}

/// The region pipelines of the released kernel in the settle test, as many as a variant keeps
/// (`pipeline/mod.rs`, `REGION_PIPELINES`), and the device's bound there.
const KEPT_KEYS: u32 = 256;

/// The keys of the other kernel in the settle test, each letting one of the kept ones go.
const LET_GO: u32 = 64;

/// A kernel's [`KEPT_KEYS`] region pipelines (widths 1 to 256, one key each) kept for its released
/// state under a device bound of as many, then another kernel's [`LET_GO`] keys, each letting the
/// least recently used kept one go: the settle at each of those dispatches asks the ledger of the
/// key it let go alone, not of every region pipeline the kept instance holds (which would be
/// 14,368 looks), and destroys that one.
#[test]
fn a_full_ledger_s_settle_looks_at_the_keys_it_let_go_alone() {
    let scratch = Scratch::new("region-prune");
    let device = device(&scratch.0);
    let shared = device.shared();
    shared.region_ledger.bound_at(KEPT_KEYS as usize);
    let first = state(&device, KERNEL);
    for n in 1..=KEPT_KEYS {
        dispatch(&device, &first, (n, n), 0);
    }
    assert_eq!(shared.region_ledger.listed(), KEPT_KEYS as usize);
    drop(first);
    idle(&device);
    let kept = (1, KEPT_KEYS as usize);
    let until = Instant::now() + WAIT;
    while lock(&shared.kept).held() != kept {
        assert!(
            Instant::now() < until,
            "the kept variants hold {:?} after {WAIT:?}",
            lock(&shared.kept).held()
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    let second = state(&device, &KERNEL.replace(STORE, STORE_PLUS_ONE));
    let evicted = device.work_counts().region_pipelines_evicted;
    let looks = prune_looks();
    let mut calls: Vec<Duration> = (1..=LET_GO)
        .map(|n| dispatch(&device, &second, (n, n), 1))
        .collect();
    let looked = prune_looks() - looks;
    calls.sort();
    println!(
        "KEPT-PRUNE {}: {KEPT_KEYS} kept region pipelines, {LET_GO} let go: the settles looked \
         at {looked} keys; dispatch call median {:?}, least {:?}",
        device.name(),
        calls[calls.len() / 2],
        calls[0]
    );
    assert_eq!(
        device.work_counts().region_pipelines_evicted - evicted,
        u64::from(LET_GO),
        "the keys the ledger let go"
    );
    assert_eq!(
        lock(&shared.kept).held(),
        (1, (KEPT_KEYS - LET_GO) as usize),
        "the kept instance after the ledger let {LET_GO} of its region pipelines go"
    );
    assert!(
        looked <= u64::from(LET_GO),
        "the settles looked at {looked} keys for the {LET_GO} the ledger let go"
    );
}
