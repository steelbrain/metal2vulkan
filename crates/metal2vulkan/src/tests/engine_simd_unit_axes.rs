//! A kernel's unit-axes variant (`VariantKey::unit_axes`, `LoweringInputs::unit_axes`).
//!
//! In a threadgroup of one thread along an axis (Metal's `threads_per_threadgroup` 1 there), every
//! thread's position along it is 0 in its threadgroup and its threadgroup's in the grid, so a
//! branch on it is uniform across the threadgroup. A kernel whose emulated simdgroups are walked
//! predicated only because such a position steers a synchronization states the axes whose variant
//! relieves the walk (`RuntimeFact::UnitAxes`), and its plan offers that variant to a dispatch of
//! such threadgroups (`BindingPlan::unit_axes_variant`). The authored kernel here sums its
//! simdgroup in a loop whose trip count reads `thread_position_in_grid.y` (and, for the control,
//! `.x`). Each module runs on the CPU interpreter at 8, 16 and 32 subgroup lanes in both
//! scheduling orders, under Vulkan's barrier rule. No GPU is used.
//!
//! `llvm-as` assembles the kernels; a missing tool skips visibly.

use std::collections::BTreeMap;

use crate::bind::{BindState, BoundBuffer, Dispatch, StageTable};
use crate::budget::Budget;
use crate::cache::VariantKey;
use crate::device::subgroup::Tier;
use crate::device::{Device, DevicePolicy};
use crate::error::BindError;
use crate::library::Library;
use crate::lower::select::Plan;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::sym::{Axis, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::{
    llvm_assemble, typed_module, typed_module_with, AssembleError, Scratch,
};
use crate::pipeline::{
    ComputePipeline, ComputePipelineDescriptor, FunctionDescriptor, LoweringInputs, StageSource,
};
use crate::spirv::interp::{
    run, BarrierRule, Buffer, Config, InterpError, Order, Resources, SubgroupSize, TrapKind,
};
use crate::spirv::ir::TypedModule;

/// `kernel void name(device uint *out [[buffer(0)]], uint3 gid [[thread_position_in_grid]],
/// uint t [[thread_index_in_threadgroup]])`: with `g = gid[axis]`, a loop of `(g & 3) + 1`
/// iterations, each adding `simd_sum(t + i)`, then `out[t + N gid.y] = acc`, `N` the
/// threadgroup's threads (a constant here: `n`).
fn kernel(name: &str, axis: u32, n: u32) -> String {
    format!(
        r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @{name}(ptr addrspace(1) %out, <3 x i32> %gid, i32 %t) #0 {{
entry:
  %g = extractelement <3 x i32> %gid, i64 {axis}
  %k = and i32 %g, 3
  %trip = add i32 %k, 1
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i1, %loop ]
  %acc = phi i32 [ 0, %entry ], [ %acc1, %loop ]
  %v = add i32 %t, %i
  %s = tail call i32 @air.simd_sum.u.i32(i32 %v)
  %acc1 = add i32 %acc, %s
  %i1 = add i32 %i, 1
  %more = icmp ult i32 %i1, %trip
  br i1 %more, label %loop, label %done
done:
  %y = extractelement <3 x i32> %gid, i64 1
  %row = mul i32 %y, {n}
  %at32 = add i32 %row, %t
  %at = zext i32 %at32 to i64
  %p = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %at
  store i32 %acc1, ptr addrspace(1) %p, align 4
  ret void
}}

declare i32 @air.simd_sum.u.i32(i32) #1

attributes #0 = {{ convergent mustprogress nounwind willreturn }}
attributes #1 = {{ convergent mustprogress nounwind willreturn memory(none) }}

!air.kernel = !{{!0}}
!air.version = !{{!6}}
!air.language_version = !{{!7}}

!0 = !{{ptr @{name}, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}}
!4 = !{{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"gid"}}
!5 = !{{i32 2, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}}
!6 = !{{i32 2, i32 8, i32 0}}
!7 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// The threadgroup every run takes: 8 x 1 x 8, two simdgroups, one thread along `y`.
const LOCAL: [u32; 3] = [8, 1, 8];
/// The threadgroups along `y` a run dispatches: trip counts 1 to 4 on the `y` kernel.
const ROWS: u32 = 4;
/// `RuntimeFact::UnitAxes`'s `relief` of a kernel whose loop reads `y`: the masks with `y`
/// (`y`, `x | y`, `y | z`, `x | y | z`), bit `m - 1` for mask `m`.
const RELIEF_Y: u8 = 1 << 1 | 1 << 2 | 1 << 5 | 1 << 6;
/// The same for a loop that reads `x`: the masks with `x`.
const RELIEF_X: u8 = 1 | 1 << 2 | 1 << 4 | 1 << 6;

fn assemble(name: &str, text: &str) -> Option<Vec<u8>> {
    let scratch = Scratch::new(&format!("m2v-unit-axes-{name}"));
    let path = scratch.path().join("kernel.ll");
    std::fs::write(&path, text).expect("write");
    match llvm_assemble(&path) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}"),
    }
}

fn lavapipe() -> Device {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/lavapipe.json");
    let text = std::fs::read_to_string(path).expect("profile");
    let desc =
        crate::device::DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::new(desc, DevicePolicy::default()).expect("device")
}

/// The unit axes `plan` states, with their relief (`RuntimeFact::UnitAxes`).
fn unit_axes(plan: &Plan) -> Option<(u8, u8)> {
    plan.facts.iter().find_map(|fact| match *fact {
        RuntimeFact::UnitAxes { axes, relief } => Some((axes, relief)),
        _ => None,
    })
}

/// The module of `name` under unit axes `axes`.
fn module(bc: &[u8], name: &str, dev: &Device, axes: u8) -> (Plan, TypedModule) {
    let mut inputs = LoweringInputs::new();
    inputs.unit_axes = axes;
    match axes {
        0 => typed_module(bc, name, dev.profile()),
        _ => typed_module_with(bc, name, dev.profile(), &inputs),
    }
    .unwrap_or_else(|e| panic!("{name} under {axes}: {e}"))
}

/// Buffer 0 after `ROWS` threadgroups of `local` along `y` ran `m` (planned as `plan`) with
/// `lanes`-wide subgroups, identical in both scheduling orders; or why a run did not complete.
fn run_rows(
    plan: &Plan,
    m: &TypedModule,
    local: [u32; 3],
    lanes: SubgroupSize,
) -> Result<Vec<u32>, InterpError> {
    let mut spec = BTreeMap::new();
    for (a, v) in [Axis::X, Axis::Y, Axis::Z].into_iter().zip(local) {
        spec.insert(SpecKey::LocalSize(a), u64::from(v));
    }
    let n: u32 = local.iter().product();
    spec.insert(SpecKey::FlatLocalSize, u64::from(n));
    let res = Resources {
        buffers: vec![Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: 0,
            }),
            address: None,
            bytes: vec![0; (n * ROWS * local[1]) as usize * 4],
        }],
        push: vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)],
        spec,
    };
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups: [1, ROWS, 1],
            subgroup: lanes,
            order,
            barriers: BarrierRule::Vulkan,
            ..Config::default()
        };
        outs.push(run(m, &cfg, res.clone())?.buffers.swap_remove(0));
    }
    assert_eq!(outs[0], outs[1], "order independent");
    let bytes = outs.swap_remove(0);
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect())
}

/// Metal's words for `LOCAL` threadgroups of the kernel whose loop reads `axis`: thread `t` of
/// threadgroup `w` runs `(g & 3) + 1` iterations, `g` its grid position on the axis (`t mod 8`
/// on `x`, `w` on `y`), each adding the sum of `t' + i` over the threads `t'` of its simdgroup
/// (32 consecutive threads) still in the loop.
fn expected(axis: u32) -> Vec<u32> {
    let n: u32 = LOCAL.iter().product();
    let trip = |t: u32, w: u32| match axis {
        0 => ((t % LOCAL[0]) & 3) + 1,
        _ => (w & 3) + 1,
    };
    let mut words = Vec::new();
    for w in 0..ROWS {
        for t in 0..n {
            let group = (t / 32) * 32..((t / 32) * 32 + 32).min(n);
            let acc = (0..trip(t, w))
                .map(|i| {
                    group
                        .clone()
                        .filter(|&u| trip(u, w) > i)
                        .map(|u| u + i)
                        .sum::<u32>()
                })
                .sum();
            words.push(acc);
        }
    }
    words
}

fn pipeline(dev: &Device, bc: &[u8], name: &str, key: &VariantKey) -> ComputePipeline {
    let f = Library::from_air(bc)
        .ok()
        .and_then(|lib| lib.function(name).ok())
        .expect("kernel");
    let budget = Budget::standard();
    let compiled = dev
        .compile_function(
            &FunctionDescriptor::new(f),
            &key.lowering_inputs(Stage::Kernel),
            &budget,
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    dev.compile_compute_pipeline(
        &ComputePipelineDescriptor::new(StageSource::Compiled(compiled)),
        &budget,
    )
    .unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn bind_state() -> BindState {
    let mut table = StageTable::default();
    table.buffers.insert(
        0,
        BoundBuffer::Buffer {
            length: 4096,
            offset: 0,
            gpu_address: 0x1_0000,
            device_address: None,
            attribute_stride: None,
        },
    );
    let mut state = BindState::default();
    state.stages.insert(Stage::Kernel, table);
    state
}

fn groups(threads_per_group: [u32; 3]) -> Dispatch {
    Dispatch::Threadgroups {
        groups: [1, ROWS, 1],
        threads_per_group,
    }
}

fn keyed(axes: u8) -> VariantKey {
    let mut key = VariantKey::new();
    key.unit_axes = axes;
    key
}

/// The emulated walk of a kernel whose simdgroup sum sits in a loop steered by
/// `thread_position_in_grid.y` is predicated, and its module states the masks of unit axes that
/// relieve it (every mask with `y`). Compiled for unit `y` it is walked unpredicated, states its
/// axes, and gives Metal's words in threadgroups of one thread along `y` at every width, the
/// same as the predicated module's; in a threadgroup of two threads along `y` it is not Metal's
/// kernel (two trip counts meet the exchange's barrier: Vulkan's rule traps), which its plan
/// never runs, sending such a dispatch back to the plain variant. Unit `x` relieves nothing of
/// it, and the same kernel steered by `.x` is relieved by no mask without `x`, so its plan
/// offers no variant for unit `y`.
#[test]
fn a_unit_axis_relieves_the_predicated_walk() {
    let dev = lavapipe();
    let (y, x) = ("unit_axes_y", "unit_axes_x");
    let n: u32 = LOCAL.iter().product();
    let Some(ybc) = assemble(y, &kernel(y, 1, n)) else {
        return;
    };
    let Some(xbc) = assemble(x, &kernel(x, 0, n)) else {
        return;
    };
    let lanes = [SubgroupSize::W8, SubgroupSize::W16, SubgroupSize::W32];

    // The kernel's own module: predicated, relieved by every mask with `y`.
    let (base, bm) = module(&ybc, y, &dev, 0);
    assert_eq!(base.simd.1, Tier::Emulated);
    assert!(
        base.families.simd.predication.is_some(),
        "walked predicated"
    );
    assert_eq!(unit_axes(&base), Some((0, RELIEF_Y)));
    // Its variant for unit `y`: unpredicated, stating its axes.
    let (unit, um) = module(&ybc, y, &dev, 0b010);
    assert_eq!(unit.simd.1, Tier::Emulated);
    assert!(
        unit.families.simd.predication.is_none(),
        "walked unpredicated"
    );
    assert_eq!(unit_axes(&unit), Some((0b010, 0)));
    let want = expected(1);
    for w in lanes {
        assert_eq!(run_rows(&base, &bm, LOCAL, w).as_ref(), Ok(&want), "{w:?}");
        assert_eq!(run_rows(&unit, &um, LOCAL, w).as_ref(), Ok(&want), "{w:?}");
    }
    // Two threads along `y`: the variant's barrier is met by two trip counts.
    match run_rows(&unit, &um, [8, 2, 4], SubgroupSize::W32) {
        Err(InterpError::Trap(trap)) => assert_eq!(trap.kind, TrapKind::NonUniformBarrier),
        other => panic!("the unit-y module at two threads along y: {other:?}"),
    }
    // Unit `x` relieves nothing.
    let (other, _) = module(&ybc, y, &dev, 0b001);
    assert!(other.families.simd.predication.is_some(), "unit x");
    assert_eq!(unit_axes(&other), Some((0b001, 0)));

    // The kernel steered by `x`: no relief for unit `y`.
    let (xbase, xm) = module(&xbc, x, &dev, 0);
    assert!(xbase.families.simd.predication.is_some());
    assert_eq!(unit_axes(&xbase), Some((0, RELIEF_X)));
    let (xunit, _) = module(&xbc, x, &dev, 0b010);
    assert!(xunit.families.simd.predication.is_some(), "x under unit y");
    let want = expected(0);
    for w in lanes {
        assert_eq!(run_rows(&xbase, &xm, LOCAL, w).as_ref(), Ok(&want), "{w:?}");
    }

    // The binder's offer.
    let state = bind_state();
    let shape = state.shape();
    let plan = pipeline(&dev, &ybc, y, &VariantKey::new())
        .plan(&dev, &shape)
        .expect("plan");
    assert_eq!(
        plan.compute.as_ref().map(|c| c.local.unit_relief),
        Some(RELIEF_Y)
    );
    let row = groups(LOCAL);
    assert_eq!(plan.dispatch_variant(&row), Ok(VariantKey::new()));
    assert_eq!(plan.unit_axes_variant(&row), Ok(Some(keyed(0b010))));
    // One thread along `x` and `y`: their variant, which `y` relieves too.
    assert_eq!(
        plan.unit_axes_variant(&groups([1, 1, 32])),
        Ok(Some(keyed(0b011)))
    );
    // No unit axis, or only `x`, which relieves nothing.
    assert_eq!(plan.unit_axes_variant(&groups([8, 2, 4])), Ok(None));
    assert_eq!(plan.unit_axes_variant(&groups([1, 4, 8])), Ok(None));
    // The plain variant runs the dispatch, as it ran before.
    assert!(plan.instantiate_dispatch(&state, &row).is_ok());

    // The variant's plan: runs threadgroups of one thread along `y` only, and offers nothing.
    let variant = pipeline(&dev, &ybc, y, &keyed(0b010));
    assert!(variant.dependencies.iter().any(|d| d.fact
        == RuntimeFact::UnitAxes {
            axes: 0b010,
            relief: 0
        }));
    let vplan = variant.plan(&dev, &shape).expect("variant plan");
    assert_eq!(
        vplan.variant,
        keyed(0b010),
        "the variant's plan states its key"
    );
    assert_eq!(vplan.dispatch_variant(&row), Ok(keyed(0b010)));
    assert!(vplan.instantiate_dispatch(&state, &row).is_ok());
    assert_eq!(vplan.unit_axes_variant(&row), Ok(None));
    let two = groups([8, 2, 4]);
    assert_eq!(vplan.dispatch_variant(&two), Ok(VariantKey::new()));
    assert_eq!(
        vplan.instantiate_dispatch(&state, &two),
        Err(BindError::NeedsVariant {
            variant: Box::new(VariantKey::new())
        })
    );

    // The kernel steered by `x` offers no variant for unit `y`.
    let xplan = pipeline(&dev, &xbc, x, &VariantKey::new())
        .plan(&dev, &shape)
        .expect("plan");
    assert_eq!(xplan.unit_axes_variant(&row), Ok(None));
}
