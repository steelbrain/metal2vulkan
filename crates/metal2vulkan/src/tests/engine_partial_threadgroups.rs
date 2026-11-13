//! A kernel's partial-threadgroups variant (`LoweringInputs::partial_threadgroups`) end to end on
//! the CPU interpreter.
//!
//! A `dispatchThreads` grid that is not a whole number of threadgroups goes through the public
//! compile API and `BindingPlan::instantiate_dispatch` twice: on the plain pipeline's plan, which
//! splits it into its interior and edge regions (on the emulated-simdgroups variant where an edge
//! is not whole subgroups, as the runtime runs it), and on the partial-threadgroups variant's,
//! which runs the threadgroups its threads fill at the call's size, per chunk of group counts,
//! flagged `lower::dispatch::NONUNIFORM`. A grid of whole threadgroups, of either dispatch form,
//! runs on the variant's plan as the same regions unflagged, every thread live, on the one
//! specialization its threadgroup size's ragged grids take. Each region runs on the interpreter
//! under Vulkan's barrier rule (`BarrierRule::Vulkan`: an invocation that returned while others
//! wait at a barrier traps), in both scheduling orders, at the subgroup width the region pins.
//! Both give Metal's
//! words: the partial threadgroup's `threads_per_threadgroup`, its threads' positions and indices
//! over its own size, its simdgroups' indices, and, behind a threadgroup barrier, a neighbour's
//! threadgroup word and its simdgroup's sum over the live lanes alone. The variant's module passes
//! `spirv-val`, holds no workgroup barrier in a block its own uniformity does not prove uniform,
//! and is within its estimate; where its simdgroups are emulated it is refused
//! (`Unsupported{GpuEncodedIndirectCommands, NotImplemented}`), so the split stays. No GPU is used.
//!
//! `llvm-as` assembles the kernels; a missing tool skips visibly.

use super::engine_binder::{assemble_text, device, execute_with, words, Alloc, World};
use crate::bind::{Dispatch, DispatchCommands};
use crate::budget::Budget;
use crate::cache::VariantKey;
use crate::device::subgroup::Tier;
use crate::device::Device;
use crate::error::{BindError, Error, Feature};
use crate::library::Library;
use crate::pipeline::sym::Stage;
use crate::pipeline::testing::{compile_with, link_and_validate, typed_module_with};
use crate::pipeline::{
    ComputePipeline, ComputePipelineDescriptor, FunctionDescriptor, LoweringInputs, RuntimeFact,
    StageSource,
};
use crate::spirv::interp::{BarrierRule, Config, SubgroupSize};
use crate::spirv::ir::TypedModule;
use crate::spirv::testing::barriers::workgroup_barriers;

/// Profiles whose kernels run on 32-wide subgroups (Apple through MoltenVK, AMD, NVIDIA, Intel),
/// and one whose simdgroups are emulated (`lavapipe`).
const PROFILES: [&str; 5] = [
    "moltenvk-m3",
    "amd-rdna3-radv",
    "nvidia-rtx50-linux",
    "intel-irisxe-rplp-anv",
    "lavapipe",
];

/// The poison every output word starts as.
const POISON: u8 = 0xAB;

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// `shape`: for the thread at `pos` of a grid `grid` wide, `out[2i] = n.x | n.y << 8 | local.x <<
/// 16 | local.y << 24` and `out[2i + 1] = index | simdgroup << 16`, `i = pos.y · grid.x + pos.x`,
/// `n` its threadgroup's `threads_per_threadgroup`, `local` its position in it, `index` its
/// `thread_index_in_threadgroup` and `simdgroup` its `simdgroup_index_in_threadgroup`.
fn shape() -> String {
    format!(
        "{HEADER}define void @shape(ptr addrspace(1) noundef %out, <2 x i32> noundef %pos, <2 x i32> noundef %grid, <2 x i32> noundef %n, <2 x i32> noundef %local, i32 noundef %index, i32 noundef %sg) {{\n\
entry:\n\
  %px = extractelement <2 x i32> %pos, i64 0\n\
  %py = extractelement <2 x i32> %pos, i64 1\n\
  %gx = extractelement <2 x i32> %grid, i64 0\n\
  %row = mul i32 %py, %gx\n\
  %i = add i32 %row, %px\n\
  %i2 = shl i32 %i, 1\n\
  %nx = extractelement <2 x i32> %n, i64 0\n\
  %ny = extractelement <2 x i32> %n, i64 1\n\
  %lx = extractelement <2 x i32> %local, i64 0\n\
  %ly = extractelement <2 x i32> %local, i64 1\n\
  %w1 = shl i32 %ny, 8\n\
  %w2 = shl i32 %lx, 16\n\
  %w3 = shl i32 %ly, 24\n\
  %a = or i32 %nx, %w1\n\
  %b = or i32 %a, %w2\n\
  %w = or i32 %b, %w3\n\
  %s = shl i32 %sg, 16\n\
  %v = or i32 %index, %s\n\
  %iz = zext i32 %i2 to i64\n\
  %p0 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %iz\n\
  store i32 %w, ptr addrspace(1) %p0, align 4\n\
  %p1 = getelementptr inbounds i32, ptr addrspace(1) %p0, i64 1\n\
  store i32 %v, ptr addrspace(1) %p1, align 4\n\
  ret void\n\
}}\n\n\
!air.kernel = !{{!0}}\n!air.version = !{{!10}}\n!air.language_version = !{{!11}}\n\
!0 = !{{ptr @shape, !1, !2}}\n!1 = !{{}}\n!2 = !{{!3, !4, !5, !6, !7, !8, !9}}\n\
!3 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}}\n\
!4 = !{{i32 1, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint2\", !\"air.arg_name\", !\"pos\"}}\n\
!5 = !{{i32 2, !\"air.threads_per_grid\", !\"air.arg_type_name\", !\"uint2\", !\"air.arg_name\", !\"grid\"}}\n\
!6 = !{{i32 3, !\"air.threads_per_threadgroup\", !\"air.arg_type_name\", !\"uint2\", !\"air.arg_name\", !\"n\"}}\n\
!7 = !{{i32 4, !\"air.thread_position_in_threadgroup\", !\"air.arg_type_name\", !\"uint2\", !\"air.arg_name\", !\"local\"}}\n\
!8 = !{{i32 5, !\"air.thread_index_in_threadgroup\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"index\"}}\n\
!9 = !{{i32 6, !\"air.simdgroup_index_in_threadgroup\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"sg\"}}\n\
!10 = !{{i32 2, i32 8, i32 0}}\n!11 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n"
    )
}

/// `neighbour`: `a[t] = 3t + 1; threadgroup barrier; out[pos] = n << 24 | a[(t + 1) mod n] << 12 |
/// simd_sum(t)`, `t` the thread's `thread_index_in_threadgroup` and `n` its threadgroup's
/// `threads_per_threadgroup`; without `simd`, the sum's word is 0.
fn neighbour(simd: bool) -> String {
    let (sum, declare) = match simd {
        true => (
            "  %s = call i32 @air.simd_sum.u.i32(i32 %t)\n",
            "declare i32 @air.simd_sum.u.i32(i32) #0\n",
        ),
        false => ("  %s = add i32 0, 0\n", ""),
    };
    format!(
        "{HEADER}@a = internal unnamed_addr addrspace(3) global [64 x i32] undef, align 4\n\n\
define void @neighbour(ptr addrspace(1) noundef %out, i32 noundef %pos, i32 noundef %t, i32 noundef %n) {{\n\
entry:\n\
  %tz = zext i32 %t to i64\n\
  %pa = getelementptr inbounds [64 x i32], ptr addrspace(3) @a, i64 0, i64 %tz\n\
  %v0 = mul i32 %t, 3\n\
  %v1 = add i32 %v0, 1\n\
  store i32 %v1, ptr addrspace(3) %pa, align 4\n\
  tail call void @air.wg.barrier(i32 2, i32 1) #0\n\
  %t1 = add i32 %t, 1\n\
  %u = urem i32 %t1, %n\n\
  %uz = zext i32 %u to i64\n\
  %pu = getelementptr inbounds [64 x i32], ptr addrspace(3) @a, i64 0, i64 %uz\n\
  %got = load i32, ptr addrspace(3) %pu, align 4\n\
{sum}\
  %hi = shl i32 %n, 24\n\
  %mid = shl i32 %got, 12\n\
  %hm = or i32 %hi, %mid\n\
  %w = or i32 %hm, %s\n\
  %pz = zext i32 %pos to i64\n\
  %po = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %pz\n\
  store i32 %w, ptr addrspace(1) %po, align 4\n\
  ret void\n\
}}\n\n\
declare void @air.wg.barrier(i32, i32) #0\n\
{declare}\n\
attributes #0 = {{ convergent mustprogress nounwind willreturn }}\n\n\
!air.kernel = !{{!0}}\n!air.version = !{{!7}}\n!air.language_version = !{{!8}}\n\
!0 = !{{ptr @neighbour, !1, !2}}\n!1 = !{{}}\n!2 = !{{!3, !4, !5, !6}}\n\
!3 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}}\n\
!4 = !{{i32 1, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"pos\"}}\n\
!5 = !{{i32 2, !\"air.thread_index_in_threadgroup\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"t\"}}\n\
!6 = !{{i32 3, !\"air.threads_per_threadgroup\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"n\"}}\n\
!7 = !{{i32 2, i32 8, i32 0}}\n!8 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n"
    )
}

/// The threads Metal's threadgroup at `group` of a grid of `grid` threads in threadgroups of `l`
/// runs along each axis: `l`, but the threads left in the last.
fn metal_threads(grid: [u32; 3], l: [u32; 3], group: [u32; 3]) -> [u32; 3] {
    std::array::from_fn(|i| (grid[i] - group[i] * l[i]).min(l[i]))
}

/// The words Metal's `dispatchThreads` of `grid` in threadgroups of `l` leaves for [`shape`].
fn metal_shape(grid: [u32; 3], l: [u32; 3]) -> Vec<u32> {
    let mut out = vec![0; 2 * (grid[0] * grid[1]) as usize];
    for y in 0..grid[1] {
        for x in 0..grid[0] {
            let group = [x / l[0], y / l[1], 0];
            let n = metal_threads(grid, l, group);
            let (lx, ly) = (x % l[0], y % l[1]);
            let index = ly * n[0] + lx;
            let i = 2 * (y * grid[0] + x) as usize;
            out[i] = n[0] | n[1] << 8 | lx << 16 | ly << 24;
            out[i + 1] = index | (index / 32) << 16;
        }
    }
    out
}

/// The words Metal's `dispatchThreads` of `threads` threads in threadgroups of `l` leaves for
/// [`neighbour`]: its partial threadgroup's threads synchronize among themselves, and its last
/// simdgroup sums its live lanes alone.
fn metal_neighbour(threads: u32, l: u32, simd: bool) -> Vec<u32> {
    (0..threads)
        .map(|pos| {
            let n = metal_threads([threads, 1, 1], [l, 1, 1], [pos / l, 0, 0])[0];
            let t = pos % l;
            let got = ((t + 1) % n) * 3 + 1;
            let first = t - t % 32;
            let lanes = (n - first).min(32);
            let sum = match simd {
                true => (first..first + lanes).sum(),
                false => 0,
            };
            n << 24 | got << 12 | sum
        })
        .collect()
}

/// The compute pipeline of `entry` in `bc` on `dev` under `inputs`, and the typed module the
/// interpreter runs.
fn pipeline(
    dev: &Device,
    bc: &[u8],
    entry: &str,
    inputs: &LoweringInputs,
) -> Result<(ComputePipeline, TypedModule), Error> {
    let lib = Library::from_air(bc).expect("library");
    let f = FunctionDescriptor::new(lib.function(entry).expect("function"));
    let compiled = dev.compile_function(&f, inputs, &Budget::standard())?;
    let d = ComputePipelineDescriptor::new(StageSource::Compiled(compiled));
    let pipe = dev.compile_compute_pipeline(&d, &Budget::standard())?;
    let (_, m) = typed_module_with(bc, entry, dev.profile(), inputs)?;
    Ok((pipe, m))
}

/// The interpreter's subgroup width for the commands' regions: the one they pin, else 32.
fn width(c: &DispatchCommands) -> SubgroupSize {
    match c.regions.first().and_then(|r| r.required_subgroup_size) {
        Some(64) => SubgroupSize::W64,
        Some(16) => SubgroupSize::W16,
        Some(8) => SubgroupSize::W8,
        _ => SubgroupSize::W32,
    }
}

/// Run `dispatch` of `pipe` (module `m`) over an output of `words` words under Vulkan's barrier
/// rule: its commands and the words it left.
fn run_on(
    dev: &Device,
    (pipe, m): (&ComputePipeline, &TypedModule),
    words_out: usize,
    dispatch: &Dispatch,
) -> Result<(DispatchCommands, Vec<u32>), BindError> {
    let mut w = World::new([(0, Alloc::new(4 * words_out, 0))]);
    w.buffers.get_mut(&0).expect("out").bytes.fill(POISON);
    let state = w.state();
    let plan = pipe.plan(dev, &state.shape())?;
    let lanes = width(&plan.instantiate_dispatch(&state, dispatch)?);
    let cfg = Config {
        subgroup: lanes,
        barriers: BarrierRule::Vulkan,
        ..Config::default()
    };
    let c = execute_with(dev, (pipe, m), &mut w, dispatch, cfg)?;
    Ok((c, words(&w.buffers[&0].bytes)))
}

/// What the split answers for `dispatch` of `entry`: the plain pipeline's regions, on the variant
/// the plan asks for where it asks for one (emulated simdgroups for an edge that is not whole
/// subgroups), as the runtime runs it.
fn split(
    dev: &Device,
    bc: &[u8],
    entry: &str,
    words_out: usize,
    dispatch: &Dispatch,
) -> (DispatchCommands, Vec<u32>) {
    let (pipe, m) = pipeline(dev, bc, entry, &LoweringInputs::new()).expect("the plain pipeline");
    match run_on(dev, (&pipe, &m), words_out, dispatch) {
        Ok(out) => out,
        Err(BindError::NeedsVariant { variant }) => {
            let inputs = variant.lowering_inputs(Stage::Kernel);
            let (pipe, m) = pipeline(dev, bc, entry, &inputs).expect("the split's variant");
            run_on(dev, (&pipe, &m), words_out, dispatch).expect("the split's variant runs it")
        }
        Err(e) => panic!("the split: {e:?}"),
    }
}

/// The partial-threadgroups variant of `entry` for threadgroups of `l` on `dev`, as the runtime
/// asks for it (`ComputePipelineState::settle_partial`): the key the plain plan names for a
/// threadgroup dispatch of `l`, with the partial threadgroups. `None` where that key emulates
/// simdgroups, which no partial variant runs.
fn partial_inputs(dev: &Device, bc: &[u8], entry: &str, l: [u32; 3]) -> Option<VariantKey> {
    let (pipe, _) = pipeline(dev, bc, entry, &LoweringInputs::new()).expect("the plain pipeline");
    let state = World::new([(0, Alloc::new(4, 0))]).state();
    let plan = pipe.plan(dev, &state.shape()).expect("plan");
    let whole = Dispatch::Threadgroups {
        groups: [1, 1, 1],
        threads_per_group: l,
    };
    let mut key = plan
        .dispatch_variant(&whole)
        .expect("the threadgroup's variant");
    if key.emulated_simdgroups {
        return None;
    }
    key.partial_threadgroups = true;
    Some(key)
}

/// A `dispatchThreads` of `threads` in threadgroups of `l`.
fn threads(threads: [u32; 3], l: [u32; 3]) -> Dispatch {
    Dispatch::Threads {
        threads,
        threads_per_group: l,
    }
}

/// A `dispatchThreadgroups` of `groups` threadgroups of `l`.
fn groups(groups: [u32; 3], l: [u32; 3]) -> Dispatch {
    Dispatch::Threadgroups {
        groups,
        threads_per_group: l,
    }
}

/// The grid and the threadgroup of a direct dispatch.
fn grid_of(d: &Dispatch) -> ([u32; 3], [u32; 3]) {
    match *d {
        Dispatch::Threads {
            threads,
            threads_per_group,
        } => (threads, threads_per_group),
        Dispatch::Threadgroups {
            groups,
            threads_per_group,
        } => (
            std::array::from_fn(|i| groups[i] * threads_per_group[i]),
            threads_per_group,
        ),
        _ => panic!("not a direct dispatch"),
    }
}

/// Check the partial-threadgroups variant of `entry` on `stem` against the split and `metal` for
/// every dispatch of `grids` (the dispatch, the split's regions, the variant's): both give
/// Metal's words, on the region counts given, and every region of the variant's plan of a
/// threadgroup size takes one specialization, whatever its grid. The variant's module is valid,
/// walks no workgroup barrier in non-uniform control flow, is within its estimate and states its
/// fact. `false` where the variant is refused or its key would emulate simdgroups (the split
/// stays).
fn check(
    stem: &str,
    (bc, entry): (&[u8], &str),
    grids: &[(Dispatch, usize, usize)],
    metal: impl Fn([u32; 3], [u32; 3]) -> Vec<u32>,
    words_out: impl Fn([u32; 3]) -> usize,
) -> bool {
    let dev = device(stem, |d| d.limits.max_compute_work_group_count = [4, 4, 4]);
    let mut ran = false;
    let mut specializations = Vec::new();
    for (d, split_regions, partial_regions) in grids {
        let (threads, l) = grid_of(d);
        let Some(key) = partial_inputs(&dev, bc, entry, l) else {
            continue;
        };
        let inputs = key.lowering_inputs(Stage::Kernel);
        let (pipe, m) = match pipeline(&dev, bc, entry, &inputs) {
            Ok(p) => p,
            Err(Error::Unsupported(u)) if u.feature == Feature::GpuEncodedIndirectCommands => {
                return false;
            }
            Err(e) => panic!("{stem} {entry}: the partial variant: {e}"),
        };
        if !ran {
            let a = compile_with(bc, entry, dev.profile(), &inputs).expect("artifact");
            assert!(
                a.facts.contains(&RuntimeFact::PartialThreadgroups),
                "{stem} {entry}: the variant's fact"
            );
            link_and_validate(&a, dev.profile()).unwrap_or_else(|e| panic!("{stem} {entry}: {e}"));
            let (plan, tm) =
                typed_module_with(bc, entry, dev.profile(), &inputs).expect("typed module");
            assert!(
                plan.estimate.words >= a.module.words.len() as u64,
                "{stem} {entry}: estimate {} < emitted {}",
                plan.estimate.words,
                a.module.words.len()
            );
            let report = workgroup_barriers(&tm).expect("barriers");
            assert!(
                report.nonuniform.is_empty(),
                "{stem} {entry}: workgroup barriers in non-uniform blocks {:?}",
                report.nonuniform
            );
            ran = true;
        }
        let n = words_out(threads);
        let want = metal(threads, l);
        let (sc, split_words) = split(&dev, bc, entry, n, d);
        let (pc, partial_words) =
            run_on(&dev, (&pipe, &m), n, d).unwrap_or_else(|e| panic!("{stem} {entry}: {e:?}"));
        let form = match d {
            Dispatch::Threadgroups { .. } => "threadgroups",
            _ => "threads",
        };
        let tag = format!("{stem} {entry} {threads:?} in {l:?} ({form})");
        assert_eq!(&split_words[..want.len()], &want[..], "{tag}: the split");
        assert_eq!(&partial_words[..want.len()], &want[..], "{tag}: partial");
        assert_eq!(
            sc.regions.len(),
            *split_regions,
            "{tag}: the split's regions"
        );
        assert_eq!(pc.regions.len(), *partial_regions, "{tag}: partial regions");
        // Every partial region runs the call's threadgroup size: one specialization, a whole
        // grid's the ragged ones'.
        let first = &pc.regions[0].specialization;
        assert!(
            (pc.regions.iter()).all(|r| r.specialization == *first),
            "{tag}: one specialization"
        );
        match specializations.iter().find(|(size, _)| *size == l) {
            Some((_, kept)) => assert!(kept == first, "{tag}: its threadgroup size's"),
            None => specializations.push((l, first.clone())),
        }
    }
    ran
}

/// A ragged grid of a kernel that reads its threadgroup's size, its position and index in it,
/// and its simdgroup's index runs on the partial-threadgroups variant as the split runs its edges,
/// as Metal answers: in one region of the call's threadgroup size per chunk of group counts (the
/// device's limit lowered to 4 a dispatch), where the split runs the interior and an edge per
/// axis and corner.
#[test]
fn a_ragged_grid_runs_partial_threadgroups_as_the_split_runs_its_edges() {
    let Some(bc) = assemble_text("shape", &shape()) else {
        return;
    };
    // (dispatch, the split's regions, the variant's).
    let grids = [
        // 3 by 2 threadgroups of 8x8, the last column 5 wide and the last row 5 high.
        (threads([21, 13, 1], [8, 8, 1]), 4, 1),
        // A grid narrower and shorter than one threadgroup: the split's corner alone.
        (threads([5, 3, 1], [8, 4, 1]), 1, 1),
        // 10 threadgroups of 32 in chunks of at most 4, the last of 2 threads: the split's 9
        // whole ones in 3 chunks and its edge.
        (threads([290, 1, 1], [32, 1, 1]), 4, 3),
        // One row of 100 in threadgroups of 64.
        (threads([100, 1, 1], [64, 1, 1]), 2, 1),
    ];
    let mut ran = Vec::new();
    for stem in PROFILES {
        let emulated = tier_of(stem, &bc, "shape") == Tier::Emulated;
        let partial = check(stem, (&bc, "shape"), &grids, metal_shape, |t: [u32; 3]| {
            2 * (t[0] * t[1]) as usize
        });
        assert_eq!(partial, !emulated, "{stem}: runs partial threadgroups");
        if partial {
            ran.push(stem);
        }
    }
    println!("PARTIAL-THREADGROUPS shape: the variant ran on {ran:?}");
    assert!(
        ran.len() >= 4,
        "the subgroup profiles run the variant: {ran:?}"
    );
}

/// A ragged grid of a kernel that reads a neighbour's threadgroup word behind a threadgroup
/// barrier, and sums its simdgroup's thread indices, runs on the partial-threadgroups variant as
/// the split runs its edge: the partial threadgroup's threads synchronize among themselves and
/// its last simdgroup sums its live lanes alone. Where the kernel's simdgroups are emulated
/// (`lavapipe`), the variant is refused and the split stays; without the sum it runs there too.
#[test]
fn a_barrier_and_a_simdgroup_sum_run_partial_threadgroups_as_metal() {
    let (Some(sum), Some(plain)) = (
        assemble_text("neighbour_sum", &neighbour(true)),
        assemble_text("neighbour", &neighbour(false)),
    ) else {
        return;
    };
    // (dispatch, the split's regions, the variant's).
    let grids = [
        (threads([100, 1, 1], [64, 1, 1]), 2, 1),
        (threads([45, 1, 1], [32, 1, 1]), 2, 1),
        (threads([16, 1, 1], [64, 1, 1]), 1, 1),
    ];
    for (bc, simd) in [(&sum, true), (&plain, false)] {
        let metal = |t: [u32; 3], l: [u32; 3]| metal_neighbour(t[0], l[0], simd);
        let mut ran = Vec::new();
        for stem in PROFILES {
            let emulated = tier_of(stem, bc, "neighbour") == Tier::Emulated;
            let partial = check(stem, (bc, "neighbour"), &grids, metal, |t| t[0] as usize);
            assert_eq!(
                partial, !emulated,
                "{stem} simd {simd}: runs partial threadgroups"
            );
            if partial {
                ran.push(stem);
            }
        }
        println!("PARTIAL-THREADGROUPS neighbour simd {simd}: the variant ran on {ran:?}");
    }
}

/// A grid of whole threadgroups, of a `dispatchThreads` or a `dispatchThreadgroups`, runs on the
/// partial-threadgroups variant's plan as its ragged grids do, every thread live: as the same
/// regions unflagged, on the one specialization a ragged grid of its threadgroup size takes, with
/// the plain pipeline's words and Metal's. `metal2vulkan-runtime` runs a large kernel's whole
/// grids so.
#[test]
fn whole_grids_run_on_the_partial_plan_with_every_thread_live() {
    let (Some(shape_bc), Some(neighbour_bc)) = (
        assemble_text("shape_whole", &shape()),
        assemble_text("neighbour_whole", &neighbour(true)),
    ) else {
        return;
    };
    // (dispatch, the split's regions, the variant's): a ragged grid, then whole ones of its
    // threadgroup size; 10 threadgroups of 32 in chunks of at most 4.
    let shapes = [
        (threads([21, 13, 1], [8, 8, 1]), 4, 1),
        (threads([24, 16, 1], [8, 8, 1]), 1, 1),
        (groups([3, 2, 1], [8, 8, 1]), 1, 1),
        (threads([290, 1, 1], [32, 1, 1]), 4, 3),
        (threads([320, 1, 1], [32, 1, 1]), 3, 3),
        (groups([10, 1, 1], [32, 1, 1]), 3, 3),
    ];
    let neighbours = [
        (threads([100, 1, 1], [64, 1, 1]), 2, 1),
        (threads([128, 1, 1], [64, 1, 1]), 1, 1),
        (groups([2, 1, 1], [64, 1, 1]), 1, 1),
    ];
    let metal = |t: [u32; 3], l: [u32; 3]| metal_neighbour(t[0], l[0], true);
    let mut ran = Vec::new();
    for stem in PROFILES {
        let words = |t: [u32; 3]| 2 * (t[0] * t[1]) as usize;
        let shape = check(stem, (&shape_bc, "shape"), &shapes, metal_shape, words);
        let runs = tier_of(stem, &shape_bc, "shape") != Tier::Emulated;
        assert_eq!(shape, runs, "{stem}: shape runs partial threadgroups");
        let words = |t: [u32; 3]| t[0] as usize;
        let neighbour = check(
            stem,
            (&neighbour_bc, "neighbour"),
            &neighbours,
            metal,
            words,
        );
        let runs = tier_of(stem, &neighbour_bc, "neighbour") != Tier::Emulated;
        assert_eq!(
            neighbour, runs,
            "{stem}: neighbour runs partial threadgroups"
        );
        if shape && neighbour {
            ran.push(stem);
        }
    }
    println!("PARTIAL-THREADGROUPS whole grids: the variant ran on {ran:?}");
    assert!(
        ran.len() >= 4,
        "the subgroup profiles run the variant: {ran:?}"
    );
}

/// The simd tier `stem` compiles `bc`'s `entry` in.
fn tier_of(stem: &str, bc: &[u8], entry: &str) -> Tier {
    let dev = device(stem, |_| {});
    let (plan, _) =
        typed_module_with(bc, entry, dev.profile(), &LoweringInputs::new()).expect("plan");
    plan.simd.1
}

/// The variant is refused with another form of the kernel's: a flat grid, folded groups, a
/// replay's bindings, or emulated simdgroups.
#[test]
fn partial_threadgroups_are_refused_with_another_form() {
    let Some(bc) = assemble_text("neighbour_forms", &neighbour(false)) else {
        return;
    };
    let dev = device("moltenvk-m3", |_| {});
    let forms: [fn(&mut LoweringInputs); 4] = [
        |i| i.flat_grid = true,
        |i| i.folded_groups = true,
        |i| i.replay_bindings = true,
        |i| i.emulated_simdgroups = true,
    ];
    for (n, form) in forms.into_iter().enumerate() {
        let mut inputs = LoweringInputs::new();
        inputs.partial_threadgroups = true;
        form(&mut inputs);
        match pipeline(&dev, &bc, "neighbour", &inputs) {
            Err(Error::Unsupported(u)) => {
                assert_eq!(u.feature, Feature::GpuEncodedIndirectCommands, "form {n}");
            }
            other => panic!("form {n}: {:?}", other.map(|_| ())),
        }
    }
}
