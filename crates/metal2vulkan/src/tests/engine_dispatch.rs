//! Dispatch builtins, barriers, and fences end to end.
//!
//! The owned fixture `engine/dispatch_builtins` goes through the whole pipeline: each kernel
//! compiles on the M1 profiles (or is refused exactly where its simd tier is), links, relocates,
//! and passes L3; its artifact carries the dispatch facts and specialization symbols; and its
//! typed module runs on the CPU interpreter with the planner's specialization and push values,
//! for `MultiRegion` 0 and 1 (a `dispatchThreads` edge region). Compute derivatives run on the
//! interpreter's `DerivativeGroupLinearKHR` model. Synthetic kernels pin that each builtin
//! parameter the body reads is derived once in the function's first block, and each push field
//! the plan places is loaded once there, wherever its first use sits. No GPU is used.
//!
//! `llvm-as` assembles the fixture; a missing tool skips visibly.

use std::collections::BTreeMap;

use crate::device::requirements::VkFeature;
use crate::device::subgroup::{self, OpBits, SimdClass, SubgroupRequirement, Tier};
use crate::device::DevicePolicy;
use crate::error::{Capability, Error, Feature, Needs};
use crate::lower::dispatch::{region_bytes, LocalSizeForm};
use crate::lower::select::Plan;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{Axis, PushField, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::{
    compile, device_f, engine_dir, link_and_validate, llvm_assemble, profile, status_reach,
    status_reach_on, typed_module, AssembleError, Scratch, StatusReach, PROFILE_F,
};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources, SubgroupSize, TrapKind};
use crate::spirv::ir::{Inst, StorageClass, Type, TypedModule};
use crate::spirv::testing::{committed_profiles, spirv_dis};

const PROFILES: [&str; 2] = ["moltenvk-m3", "runtime-floor"];

const MARKERS: [&str; 6] = [
    "thread_index_marker",
    "grid_position_marker",
    "group_position_marker",
    "threads_per_grid_marker",
    "groups_per_grid_marker",
    "simdgroup_index_marker",
];

fn fixture() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join("dispatch_builtins.ll")) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected dispatch_builtins.ll: {why}"),
    }
}

/// One Vulkan dispatch of a kernel: exact local size, group count, and (with `MultiRegion = 1`)
/// the region's `(base, threads_per_grid, nominal size)`.
struct Dispatch {
    local: [u32; 3],
    groups: [u32; 3],
    multi: Option<([u32; 3], [u32; 3], [u32; 3])>,
}

fn single(local: [u32; 3], groups: [u32; 3]) -> Dispatch {
    Dispatch {
        local,
        groups,
        multi: None,
    }
}

/// The device address of the command buffer's `ExecutionStatus` buffer in [`run_word0`].
const STATUS_AT: u64 = 0x0000_0042_0000_0000;

/// The `ExecutionStatus` buffer's bytes: through its last word.
const STATUS_BYTES: usize =
    crate::bind::status::StatusWord::SeqCstFences.byte_offset() as usize + 4;

/// Run `m` (planned as `plan`) once in each scheduling order with the planner's values and return
/// word 0 of buffer 0.
fn run_word0(plan: &Plan, m: &TypedModule, d: &Dispatch) -> u32 {
    let mut spec = BTreeMap::new();
    for (a, v) in [Axis::X, Axis::Y, Axis::Z].into_iter().zip(d.local) {
        spec.insert(SpecKey::LocalSize(a), u64::from(v));
    }
    let n: u32 = d.local.iter().product();
    spec.insert(SpecKey::FlatLocalSize, u64::from(n));
    let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        let PushLoc::Push { offset } = f.loc else {
            panic!("a field in the argument table");
        };
        let o = offset as usize;
        match f.field {
            PushField::DispatchRegion => {
                if let Some((base, threads, l)) = d.multi {
                    let b = region_bytes(base, threads, l).expect("pack");
                    push[o..o + 32].copy_from_slice(&b);
                }
            }
            // A zero residual: buffer 0 is bound at its start.
            PushField::SlotBindOffset(_) => {}
            // The `seq_cst` fence's status word.
            PushField::ExecutionStatusAddress => {
                push[o..o + 8].copy_from_slice(&STATUS_AT.to_le_bytes())
            }
            other => panic!("unexpected push field {other:?}"),
        }
    }
    if d.multi.is_some() {
        spec.insert(SpecKey::MultiRegion, 1);
    }
    let res = Resources {
        buffers: vec![
            Buffer {
                slot: Some(ResourceSlot {
                    stage: Stage::Kernel,
                    kind: ResourceKind::Buffer,
                    index: 0,
                }),
                address: None,
                bytes: vec![0; 16],
            },
            // At its address, and at the kernel's binding of it where the device does not
            // address buffers.
            Buffer {
                slot: Some(crate::lower::access::absent::symbol(Stage::Kernel)),
                address: Some(STATUS_AT),
                bytes: vec![0; STATUS_BYTES],
            },
        ],
        push,
        spec,
    };
    let mut words = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups: d.groups,
            subgroup: SubgroupSize::W32,
            order,
            ..Config::default()
        };
        let out = run(m, &cfg, res.clone()).expect("run");
        let b = &out.buffers[0];
        words.push(u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    }
    assert_eq!(words[0], words[1], "order independent");
    words[0]
}

/// Every kernel compiles on every M1 profile where its simd tier exists, links, relocates, and
/// passes L3; the artifact carries the dispatch facts and the specialization symbols of its
/// local-size form.
#[test]
fn dispatch_kernels_compile_link_and_validate() {
    let Some(bc) = fixture() else { return };
    for stem in PROFILES {
        let p = profile(stem, DevicePolicy::RESIDENT_TABLES).expect(stem);
        for entry in MARKERS
            .iter()
            .chain(&["threadgroup_barrier_store", "device_fence_store"])
        {
            let a = compile(&bc, entry, &p).unwrap_or_else(|e| panic!("{stem} {entry}: {e}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem} {entry}: {e}"));
            // The `seq_cst` fence reaches the status word by address, or through the kernel's
            // binding of it where the device does not address buffers.
            let reach = match *entry == "device_fence_store" {
                true => status_reach_on(&p),
                false => StatusReach::None,
            };
            assert_eq!(status_reach(&a), reach, "{stem} {entry}");
            for f in [
                RuntimeFact::ThreadsPerThreadgroup,
                RuntimeFact::LocalSizeLimits,
                RuntimeFact::IndirectGroupCount,
            ] {
                assert!(a.facts.contains(&f), "{stem} {entry}: {f:?}");
            }
            let grid = ["grid_position_marker", "group_position_marker"]
                .iter()
                .chain(&["threads_per_grid_marker", "groups_per_grid_marker"])
                .any(|g| g == entry);
            assert_eq!(
                a.facts.contains(&RuntimeFact::DispatchGrid),
                grid,
                "{stem} {entry}"
            );
            assert_eq!(
                a.interface.declares_spec(SpecKey::MultiRegion),
                grid,
                "{stem} {entry}"
            );
            assert!(!a.interface.declares_spec(SpecKey::FlatLocalSize));
            let region = a
                .interface
                .push
                .fields
                .iter()
                .any(|f| f.field == PushField::DispatchRegion);
            assert_eq!(region, grid, "{stem} {entry}");
        }
    }
}

/// A simdgroup barrier is cross-lane: on every committed profile, the kernel is flattened (`sn`,
/// full subgroups, `LocalSizeConstraint`) exactly where the profile pins 32 lanes for compute, and
/// elsewhere its simdgroups are emulated: no flattening, no subgroup requirement, and the barrier
/// at workgroup scope.
#[test]
fn a_simdgroup_barrier_flattens_or_is_emulated_with_the_tier() {
    let Some(bc) = fixture() else { return };
    let (mut flattened, mut emulated) = (0, 0);
    for (stem, p) in committed_profiles().expect("profiles") {
        let tier = subgroup::tier(
            &SimdClass::CrossLane(OpBits::BASIC),
            Stage::Kernel,
            &p.fields().subgroup,
        );
        match (compile(&bc, "simdgroup_barrier_marker", &p), tier) {
            (Ok(a), Ok(Tier::Emulated)) => {
                let (_, words) =
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
                assert!(!a.interface.declares_spec(SpecKey::FlatLocalSize), "{stem}");
                assert_eq!(a.subgroup, SubgroupRequirement::NONE, "{stem}");
                // Nothing is exchanged: no scratch.
                assert!(
                    !a.facts
                        .iter()
                        .any(|f| matches!(f, RuntimeFact::SimdgroupEmulation { .. })),
                    "{stem}"
                );
                let text = spirv_dis(&words);
                assert!(
                    text.contains("OpControlBarrier %uint_2 %uint_2"),
                    "{stem}\n{text}"
                );
                emulated += 1;
            }
            (Ok(a), Ok(_)) => {
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
                assert!(a.interface.declares_spec(SpecKey::FlatLocalSize), "{stem}");
                let multiple = a.subgroup.local_size_multiple;
                assert!(multiple >= 32, "{stem}");
                assert!(a
                    .facts
                    .contains(&RuntimeFact::LocalSizeConstraint { multiple }));
                assert!(a.subgroup.full_subgroups, "{stem}");
                flattened += 1;
            }
            (r, t) => panic!("{stem}: compile {r:?} against tier {t:?}"),
        }
    }
    assert!(
        flattened > 0 && emulated > 0,
        "{flattened} flattened, {emulated} emulated"
    );
}

/// The builtins' values on the interpreter under `MultiRegion = 0` (`Threadgroups`): each marker
/// is stored exactly when a thread's builtin matches.
#[test]
fn single_region_builtins_run_on_the_interpreter() {
    let Some(bc) = fixture() else { return };
    let p = profile("moltenvk-m3", DevicePolicy::RESIDENT_TABLES).expect("profile");
    let cases: [(&str, Dispatch, u32); 12] = [
        ("thread_index_marker", single([8, 1, 1], [1, 1, 1]), 105),
        ("thread_index_marker", single([4, 1, 1], [2, 1, 1]), 0),
        ("thread_index_marker", single([2, 3, 1], [1, 1, 1]), 105),
        ("grid_position_marker", single([16, 1, 1], [3, 1, 1]), 137),
        ("grid_position_marker", single([16, 1, 1], [2, 1, 1]), 0),
        ("group_position_marker", single([4, 1, 1], [3, 1, 1]), 202),
        ("group_position_marker", single([4, 1, 1], [2, 1, 1]), 0),
        ("threads_per_grid_marker", single([8, 1, 1], [5, 1, 1]), 140),
        ("threads_per_grid_marker", single([8, 1, 1], [4, 1, 1]), 0),
        ("groups_per_grid_marker", single([8, 1, 1], [3, 1, 1]), 303),
        ("simdgroup_index_marker", single([64, 1, 1], [1, 1, 1]), 301),
        ("simdgroup_index_marker", single([32, 1, 1], [2, 1, 1]), 0),
    ];
    for (entry, d, want) in cases {
        let (plan, m) = typed_module(&bc, entry, &p).unwrap_or_else(|e| panic!("{entry}: {e}"));
        assert_eq!(
            plan.dispatch.as_ref().map(|d| d.local_size),
            Some(LocalSizeForm::Spec)
        );
        assert_eq!(run_word0(&plan, &m, &d), want, "{entry} {:?}", d.local);
    }
}

/// `MultiRegion = 1`: the edge region of `dispatchThreads(40, 16)` (8 threads starting at
/// threadgroup 2) reports Metal's grid position, threadgroup position, and grid sizes; the interior
/// region does not contain thread 37 or threadgroup 2.
#[test]
fn edge_region_builtins_run_on_the_interpreter() {
    let Some(bc) = fixture() else { return };
    let p = profile("moltenvk-m3", DevicePolicy::RESIDENT_TABLES).expect("profile");
    let (t, l) = ([40, 1, 1], [16, 1, 1]);
    let interior = || Dispatch {
        local: [16, 1, 1],
        groups: [2, 1, 1],
        multi: Some(([0, 0, 0], t, l)),
    };
    let edge = || Dispatch {
        local: [8, 1, 1],
        groups: [1, 1, 1],
        multi: Some(([2, 0, 0], t, l)),
    };
    let cases: [(&str, Dispatch, u32); 7] = [
        ("grid_position_marker", edge(), 137),
        ("grid_position_marker", interior(), 0),
        ("group_position_marker", edge(), 202),
        ("group_position_marker", interior(), 0),
        ("threads_per_grid_marker", edge(), 140),
        ("threads_per_grid_marker", interior(), 140),
        ("groups_per_grid_marker", edge(), 303),
    ];
    for (entry, d, want) in cases {
        let (plan, m) = typed_module(&bc, entry, &p).unwrap_or_else(|e| panic!("{entry}: {e}"));
        assert_eq!(run_word0(&plan, &m, &d), want, "{entry} {:?}", d.local);
    }
}

/// Barriers and fences run on the interpreter (a workgroup barrier is a rendezvous of every
/// invocation, and a subgroup barrier of the whole subgroup); the simdgroup barrier's kernel is
/// flattened on `moltenvk-m3`, so thread 37 comes from the subgroup partition.
#[test]
fn barrier_kernels_run_on_the_interpreter() {
    let Some(bc) = fixture() else { return };
    let p = profile("moltenvk-m3", DevicePolicy::RESIDENT_TABLES).expect("profile");
    for (entry, local, want, form) in [
        (
            "threadgroup_barrier_store",
            [64, 1, 1],
            42,
            LocalSizeForm::Spec,
        ),
        ("device_fence_store", [16, 1, 1], 42, LocalSizeForm::Spec),
        (
            "simdgroup_barrier_marker",
            [64, 1, 1],
            137,
            LocalSizeForm::Flat,
        ),
        (
            "simdgroup_barrier_marker",
            [32, 1, 1],
            0,
            LocalSizeForm::Flat,
        ),
    ] {
        let (plan, m) = typed_module(&bc, entry, &p).unwrap_or_else(|e| panic!("{entry}: {e}"));
        assert_eq!(plan.dispatch.as_ref().map(|d| d.local_size), Some(form));
        assert_eq!(
            run_word0(&plan, &m, &single(local, [2, 1, 1])),
            want,
            "{entry}"
        );
        let tangled = usize::from(entry != "device_fence_store");
        assert_eq!(
            plan.tangled.len(),
            tangled,
            "{entry}: barriers are tangled, fences not"
        );
    }
    // The `seq_cst` fence orders through the kernel's binding of the status word where the device
    // does not address buffers (below the runtime's floor).
    let floor = device_f(DevicePolicy::RESIDENT_TABLES).expect(PROFILE_F);
    let (plan, m) =
        typed_module(&bc, "device_fence_store", floor.profile()).expect("device_fence_store");
    assert_eq!(
        plan.families.access.status_binding,
        Some(crate::lower::access::absent::StatusBinding::Required)
    );
    assert_eq!(run_word0(&plan, &m, &single([16, 1, 1], [2, 1, 1])), 42);
}

/// `store_const` with extra calls before its return (a synthetic kernel: Metal does not expose
/// derivatives in kernels), assembled from a scratch copy of the owned fixture.
fn store_const_with(tag: &str, calls: &[&str], decls: &[&str]) -> Option<Vec<u8>> {
    let text = std::fs::read_to_string(engine_dir().join("store_const.ll")).expect("read");
    let body = format!("{}\n  ret void\n}}", calls.join("\n"));
    let text = text.replacen("  ret void\n}", &body, 1);
    let decls = format!("\n{}\n\nattributes #0", decls.join("\n"));
    let text = text.replacen("\nattributes #0", &decls, 1);
    let scratch = Scratch::new(&format!("m2v-engine-dispatch-{tag}"));
    let dir = scratch.path();
    let path = dir.join("kernel.ll");
    std::fs::write(&path, text).expect("write");
    let out = llvm_assemble(&path);
    match out {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the kernel: {why}"),
    }
}

const DFDX: &str = "  %d = tail call float @air.dfdx.f32(float 1.000000e+00)";
const DFDX_DECL: &str = "declare float @air.dfdx.f32(float)";
const SIMD_BARRIER: &str = "  tail call void @air.simdgroup.barrier(i32 0, i32 4)";
const SIMD_BARRIER_DECL: &str = "declare void @air.simdgroup.barrier(i32, i32)";

/// A kernel's derivatives use `DerivativeGroupLinearKHR` where the profile has
/// `computeDerivativeGroupLinear` (and depend on a local size that is a multiple of 4), are refused
/// naming the feature elsewhere, and are refused as `ComputeDerivativesWithSimd` when a simdgroup
/// barrier flattens the kernel.
#[test]
fn compute_derivatives_need_the_linear_group_and_no_flattening() {
    let Some(deriv) = store_const_with("dfdx", &[DFDX], &[DFDX_DECL]) else {
        return;
    };
    let Some(with_simd) = store_const_with(
        "dfdx-simd",
        &[DFDX, SIMD_BARRIER],
        &[DFDX_DECL, SIMD_BARRIER_DECL],
    ) else {
        return;
    };
    let (mut ok, mut missing, mut flattened) = (0, 0, 0);
    for (stem, p) in committed_profiles().expect("profiles") {
        let linear = p.fields().images.compute_derivative_linear;
        match compile(&deriv, "store_const", &p) {
            Ok(a) => {
                assert!(linear, "{stem}");
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
                assert!(a
                    .facts
                    .contains(&RuntimeFact::LocalSizeConstraint { multiple: 4 }));
                ok += 1;
            }
            Err(Error::Unsupported(u)) => {
                assert!(!linear, "{stem}: {u:?}");
                assert_eq!(u.feature, Feature::ComputeDerivatives, "{stem}");
                assert_eq!(
                    u.needs,
                    Needs::AnyOf(vec![vec![Capability::Feature(
                        VkFeature::ComputeDerivativeGroupLinear
                    )]])
                );
                missing += 1;
            }
            Err(e) => panic!("{stem}: {e}"),
        }
        let tier = subgroup::tier(
            &SimdClass::CrossLane(OpBits::BASIC),
            Stage::Kernel,
            &p.fields().subgroup,
        );
        if linear && tier.is_ok() {
            match (compile(&with_simd, "store_const", &p), tier) {
                // Emulated simdgroups keep the local size.
                (Ok(a), Ok(Tier::Emulated)) => {
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
                }
                (Err(Error::Unsupported(u)), _) => {
                    assert_eq!(u.feature, Feature::ComputeDerivativesWithSimd, "{stem}");
                    flattened += 1;
                }
                (other, t) => panic!("{stem} {t:?}: {other:?}"),
            }
        }
    }
    assert!(
        ok > 0 && missing > 0 && flattened > 0,
        "{ok} ok, {missing} missing, {flattened} flattened"
    );
}

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// Kernel `deriv_k(device float *out [[buffer(0)]], uint t [[thread_index_in_threadgroup]])`
/// storing `dfdx(P)`, `dfdy(P)`, and `fwidth(P)` to `out[3t..3t + 3]` for `P = float(t)` (or
/// `float(t)²` when `square`), a synthetic kernel assembled from a scratch file.
fn derivative_kernel(square: bool) -> Option<Vec<u8>> {
    let p = if square {
        "  %g = uitofp i32 %1 to float\n  %f = fmul float %g, %g\n"
    } else {
        "  %f = uitofp i32 %1 to float\n"
    };
    let mut body = String::from(p);
    body.push_str("  %i = mul i32 %1, 3\n");
    for (k, op) in ["dfdx", "dfdy", "fwidth"].iter().enumerate() {
        body.push_str(&format!(
            "  %d{k} = tail call float @air.{op}.f32(float %f)\n  %i{k} = add i32 %i, {k}\n  %e{k} = zext i32 %i{k} to i64\n  %p{k} = getelementptr inbounds float, ptr addrspace(1) %0, i64 %e{k}\n  store float %d{k}, ptr addrspace(1) %p{k}, align 4\n"
        ));
    }
    let text = format!(
        "{HEADER}define void @deriv_k(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, i32 noundef %1) {{\nentry:\n{body}  ret void\n}}\n\n\
         declare float @air.dfdx.f32(float)\ndeclare float @air.dfdy.f32(float)\ndeclare float @air.fwidth.f32(float)\n\n\
         !air.kernel = !{{!9}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\
         !9 = !{{ptr @deriv_k, !10, !11}}\n!10 = !{{}}\n!11 = !{{!12, !13}}\n\
         !12 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"out\"}}\n\
         !13 = !{{i32 1, !\"air.thread_index_in_threadgroup\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"t\"}}\n\
         !17 = !{{i32 2, i32 8, i32 0}}\n!18 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n"
    );
    let scratch = Scratch::new(&format!("m2v-engine-derivative-{square}"));
    let dir = scratch.path();
    let path = dir.join("kernel.ll");
    std::fs::write(&path, text).expect("write");
    let out = llvm_assemble(&path);
    match out {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the kernel: {why}"),
    }
}

/// Run `deriv_k` (planned as `plan`) over one workgroup of `local` in both orders: buffer 0's
/// words, or the run's error.
fn run_derivatives(
    plan: &Plan,
    m: &TypedModule,
    local: [u32; 3],
) -> Result<Vec<u32>, crate::spirv::interp::InterpError> {
    let mut spec = BTreeMap::new();
    for (a, v) in [Axis::X, Axis::Y, Axis::Z].into_iter().zip(local) {
        spec.insert(SpecKey::LocalSize(a), u64::from(v));
    }
    let n: u32 = local.iter().product();
    spec.insert(SpecKey::FlatLocalSize, u64::from(n));
    for f in &plan.push.fields {
        assert!(
            matches!(
                (f.field, f.loc),
                (PushField::SlotBindOffset(_), PushLoc::Push { .. })
            ),
            "{f:?}"
        );
    }
    let res = Resources {
        buffers: vec![Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: 0,
            }),
            address: None,
            bytes: vec![0; n as usize * 12],
        }],
        push: vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)],
        spec,
    };
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            order,
            ..Config::default()
        };
        let out = run(m, &cfg, res.clone())?;
        outs.push(
            out.buffers[0]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<u32>>(),
        );
    }
    assert_eq!(outs[0], outs[1], "order independent");
    Ok(outs.swap_remove(0))
}

/// On the interpreter's derivative model: on every profile with `computeDerivativeGroupLinear`, a
/// kernel's `dfdx`, `dfdy`, and `fwidth` of its linear thread index are 1, 2, and 3 in every lane
/// (Metal's compute quads are four consecutive linear threads, positions `t & 3`), for 1D and 2D
/// local sizes; the value is affine in the group, so the fine or coarse choice the unqualified
/// operations leave to the device cannot matter. Of `t²`, whose rows differ, the choice matters and
/// the interpreter refuses to pick one.
#[test]
fn compute_derivatives_run_on_the_linear_group_model() {
    let (Some(linear), Some(square)) = (derivative_kernel(false), derivative_kernel(true)) else {
        return;
    };
    let mut ran = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        if !p.fields().images.compute_derivative_linear {
            continue;
        }
        let a = compile(&linear, "deriv_k", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let (plan, m) = typed_module(&linear, "deriv_k", &p).expect("module");
        for local in [[16, 1, 1], [8, 4, 1]] {
            let n = (local[0] * local[1]) as usize;
            let w = run_derivatives(&plan, &m, local).unwrap_or_else(|e| panic!("{stem}: {e}"));
            let want: Vec<u32> = (0..n)
                .flat_map(|_| [1f32, 2.0, 3.0].map(f32::to_bits))
                .collect();
            assert_eq!(w, want, "{stem} {local:?}");
        }
        let (plan, m) = typed_module(&square, "deriv_k", &p).expect("module");
        let e = run_derivatives(&plan, &m, [16, 1, 1]).expect_err("rows differ");
        assert!(
            matches!(e.trap_kind(), Some(TrapKind::ImplementationChoice(_))),
            "{stem}: {e}"
        );
        ran += 1;
    }
    assert!(ran > 0, "no profile has computeDerivativeGroupLinear");
}

/// A synthetic kernel `k(device uint* out [[buffer(0)]], uint gid [[thread_position_in_grid]])`
/// that reads word 0 and branches on it; `entry` is spliced into its first block, and `left` and
/// `right` into the two arms.
fn position_kernel(tag: &str, entry: &str, left: &str, right: &str) -> Option<Vec<u8>> {
    let body = format!(
        "{entry}  %v = load i32, ptr addrspace(1) %out, align 4
  %c = icmp eq i32 %v, 0
  br i1 %c, label %left, label %right
left:
{left}  br label %done
right:
{right}  br label %done
done:
  ret void
"
    );
    kernel_k(tag, &body)
}

/// The synthetic kernel `k(device uint* out [[buffer(0)]], uint gid [[thread_position_in_grid]])`
/// whose body, from its first block's instructions on, is `body`, assembled from a scratch copy.
fn kernel_k(tag: &str, body: &str) -> Option<Vec<u8>> {
    let text = format!(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"
target triple = \"air64_v28-apple-macosx26.0.0\"

define void @k(ptr addrspace(1) noundef \"air-buffer-no-alias\" %out, i32 noundef %gid) {{
entry:
{body}}}

!air.kernel = !{{!0}}
!air.version = !{{!90}}
!air.language_version = !{{!91}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!10, !11}}
!10 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}}
!11 = !{{i32 1, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"gid\"}}
!90 = !{{i32 2, i32 8, i32 0}}
!91 = !{{!\"Metal\", i32 4, i32 0, i32 0}}
"
    );
    let scratch = Scratch::new(&format!("m2v-engine-dispatch-{tag}"));
    let dir = scratch.path();
    let path = dir.join("kernel.ll");
    std::fs::write(&path, text).expect("write");
    let out = llvm_assemble(&path);
    match out {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {tag}: {why}"),
    }
}

/// The arms of [`position_kernel`] whose first uses of the position are in sibling blocks: the
/// left arm stores `3 · gid + 1` at word `gid + 1`, the right arm `gid` at word `gid + 2`.
const LEFT_USES: &str = "  %g1 = add i32 %gid, 1
  %i1 = zext i32 %g1 to i64
  %p1 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %i1
  %t = mul i32 %gid, 3
  %t1 = add i32 %t, 1
  store i32 %t1, ptr addrspace(1) %p1, align 4
";
const RIGHT_USES: &str = "  %g2 = add i32 %gid, 2
  %i2 = zext i32 %g2 to i64
  %p2 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %i2
  store i32 %gid, ptr addrspace(1) %p2, align 4
";
/// A use of the position in the first block that nothing observes.
const DEAD_USE: &str = "  %dead = mul i32 %gid, 7
";

/// How many times `m` loads each `Input` pointer it loads.
fn input_loads(m: &TypedModule) -> BTreeMap<u32, usize> {
    let mut loads = BTreeMap::new();
    for n in m.blocks().iter().flat_map(|b| &b.insts) {
        if let Inst::Load { ptr, .. } = &n.inst {
            let ty = m.value(*ptr).and_then(|d| m.types().get(d.ty));
            if let Ok(Type::Pointer {
                class: StorageClass::Input,
                ..
            }) = ty
            {
                *loads.entry(ptr.0).or_insert(0) += 1;
            }
        }
    }
    loads
}

/// Two kernels that differ only by a use of `thread_position_in_grid` in the first block that
/// nothing observes emit the same words on every committed profile, and a kernel whose only use of
/// the position is that one derives no builtin.
#[test]
fn a_dead_use_of_the_position_leaves_the_module_unchanged() {
    let (Some(with), Some(without), Some(unread)) = (
        position_kernel("dead", DEAD_USE, LEFT_USES, RIGHT_USES),
        position_kernel("live", "", LEFT_USES, RIGHT_USES),
        position_kernel(
            "unread",
            DEAD_USE,
            "  store i32 5, ptr addrspace(1) %out, align 4\n",
            "",
        ),
    ) else {
        return;
    };
    for (stem, p) in committed_profiles().expect("profiles") {
        let words = |bc: &[u8]| {
            let a = compile(bc, "k", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
            link_and_validate(&a, &p)
                .unwrap_or_else(|e| panic!("{stem}: {e}"))
                .1
        };
        assert_eq!(words(&with), words(&without), "{stem}");
        let (plan, m) = typed_module(&unread, "k", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(
            plan.dispatch.as_ref().is_some_and(|d| d.params.is_empty()),
            "{stem}"
        );
        assert!(input_loads(&m).is_empty(), "{stem}: {:?}", input_loads(&m));
    }
}

/// A position whose first uses sit in sibling arms is derived once, in the function's first block,
/// as when the first block uses it too: each builtin variable is loaded once on every committed
/// profile. On the interpreter each thread stores `3 · gid + 1` at word `gid + 1`.
#[test]
fn a_position_first_used_in_sibling_arms_is_derived_once() {
    let live_entry = "  %e = add i32 %gid, 7
  store i32 %e, ptr addrspace(1) %out, align 4
";
    let (Some(arms), Some(entry)) = (
        position_kernel("arms", "", LEFT_USES, RIGHT_USES),
        position_kernel("entry", live_entry, LEFT_USES, RIGHT_USES),
    ) else {
        return;
    };
    for (stem, p) in committed_profiles().expect("profiles") {
        for (what, bc) in [("arms", &arms), ("entry", &entry)] {
            let (_, m) = typed_module(bc, "k", &p).unwrap_or_else(|e| panic!("{stem} {what}: {e}"));
            let loads = input_loads(&m);
            assert!(!loads.is_empty(), "{stem} {what}");
            assert!(loads.values().all(|&n| n == 1), "{stem} {what}: {loads:?}");
        }
    }
    let p = profile("moltenvk-m3", DevicePolicy::RESIDENT_TABLES).expect("profile");
    let (plan, m) = typed_module(&arms, "k", &p).expect("compile");
    let mut spec = BTreeMap::new();
    for a in [Axis::X, Axis::Y, Axis::Z] {
        spec.insert(SpecKey::LocalSize(a), if a == Axis::X { 4 } else { 1 });
    }
    spec.insert(SpecKey::FlatLocalSize, 4);
    let push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    let res = Resources {
        buffers: vec![Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: 0,
            }),
            address: None,
            bytes: vec![0; 64],
        }],
        push,
        spec,
    };
    let cfg = Config {
        workgroups: [2, 1, 1],
        subgroup: SubgroupSize::W32,
        ..Config::default()
    };
    let out = run(&m, &cfg, res).expect("run");
    let words: Vec<u32> = out.buffers[0]
        .chunks(4)
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect();
    let want: Vec<u32> = std::iter::once(0)
        .chain((0..8).map(|gid| 3 * gid + 1))
        .chain(std::iter::repeat_n(0, 7))
        .collect();
    assert_eq!(words, want);
}

/// How many times `m` loads each push-block or `ArgumentTable` member: a load through an access
/// chain with constant indices into a `PushConstant` or `Uniform` variable, keyed by the chain.
pub(super) fn field_loads(m: &TypedModule) -> BTreeMap<(u32, Vec<u32>), usize> {
    let chains: BTreeMap<u32, (u32, Vec<u32>)> = m
        .blocks()
        .iter()
        .flat_map(|b| &b.insts)
        .filter_map(|n| match (&n.inst, n.result) {
            (Inst::AccessChain { base, indices, .. }, Some(r)) => {
                Some((r.0, (base.0, indices.iter().map(|i| i.0).collect())))
            }
            _ => None,
        })
        .collect();
    let mut loads = BTreeMap::new();
    for n in m.blocks().iter().flat_map(|b| &b.insts) {
        let Inst::Load { ptr, .. } = &n.inst else {
            continue;
        };
        let class = match m.value(*ptr).and_then(|d| m.types().get(d.ty)) {
            Ok(Type::Pointer { class, .. }) => *class,
            _ => continue,
        };
        if matches!(class, StorageClass::PushConstant | StorageClass::Uniform) {
            if let Some(key) = chains.get(&ptr.0) {
                *loads.entry(key.clone()).or_insert(0) += 1;
            }
        }
    }
    loads
}

/// A push field first read in two sibling arms (the buffer's bind offset, which each arm's store
/// reads where the profile binds the buffer through a descriptor) is loaded once, in the function's
/// first block: on every committed profile each push-block and `ArgumentTable` member is loaded at
/// most once. On the interpreter, threads 0 to 3 store 1 at word `gid` and threads 4 to 7 store 2
/// at word `gid + 8`.
#[test]
fn a_push_field_first_read_in_sibling_arms_is_loaded_once() {
    let body = "  %c = icmp ult i32 %gid, 4
  br i1 %c, label %left, label %right
left:
  %i1 = zext i32 %gid to i64
  %p1 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %i1
  store i32 1, ptr addrspace(1) %p1, align 4
  br label %done
right:
  %g2 = add i32 %gid, 8
  %i2 = zext i32 %g2 to i64
  %p2 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %i2
  store i32 2, ptr addrspace(1) %p2, align 4
  br label %done
done:
  ret void
";
    let Some(bc) = kernel_k("push-arms", body) else {
        return;
    };
    let mut read = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let (_, m) = typed_module(&bc, "k", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let loads = field_loads(&m);
        read += loads.len();
        assert!(loads.values().all(|&n| n == 1), "{stem}: {loads:?}");
    }
    assert!(read > 0, "no profile reads a push field");
    let p = profile("moltenvk-m3", DevicePolicy::RESIDENT_TABLES).expect("profile");
    let (plan, m) = typed_module(&bc, "k", &p).expect("compile");
    let mut spec = BTreeMap::new();
    for a in [Axis::X, Axis::Y, Axis::Z] {
        spec.insert(SpecKey::LocalSize(a), if a == Axis::X { 4 } else { 1 });
    }
    spec.insert(SpecKey::FlatLocalSize, 4);
    let push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    let res = Resources {
        buffers: vec![Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: 0,
            }),
            address: None,
            bytes: vec![0; 64],
        }],
        push,
        spec,
    };
    let cfg = Config {
        workgroups: [2, 1, 1],
        subgroup: SubgroupSize::W32,
        ..Config::default()
    };
    let out = run(&m, &cfg, res).expect("run");
    let words: Vec<u32> = out.buffers[0]
        .chunks(4)
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect();
    let mut want = vec![0u32; 16];
    for gid in 0..8 {
        if gid < 4 {
            want[gid] = 1;
        } else {
            want[gid + 8] = 2;
        }
    }
    assert_eq!(words, want);
}
