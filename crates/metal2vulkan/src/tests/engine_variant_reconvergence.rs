//! The reconvergence of a kernel module whose invocations past the grid return before the body:
//! the partial-threadgroups variant (`LoweringInputs::partial_threadgroups`) and the flat grid
//! (`LoweringInputs::flat_grid`). Its invocations past the grid return before the body, and SPIR-V states
//! reconvergence over all invocations, returned ones included (SPIR-V 1.6 section 2.2.5, "Dynamic
//! Instance": "Invocations may only resume executing the same dynamic instances when all
//! invocations reach the same static instruction"). Only `MaximallyReconvergesKHR` lets an
//! invocation that returned escape reconvergence (`SPV_KHR_maximal_reconvergence`,
//! "Reconvergence"). So such a module's tangled instructions take the profile's mode even where
//! the plain module's are reached together (`lower::simd::together`): on a profile with maximal
//! reconvergence both are `Guaranteed` with no note, and on one without it the early-returning
//! module records `ImplementationDefined` with `DeviceReconvergence` while the plain module is
//! `Guaranteed`. A block the workgroup uniformity proves uniform does not keep the live
//! invocations together either: the lowering adds selections of its own (a bounds-checked load or
//! store) whose condition differs between invocations, and after the return nothing reconverges
//! them at those merges.
//!
//! The kernels are the engine's `a27_a4_simdgroup_barrier` (a simdgroup barrier in the entry
//! block, loop-free), whose plain module needs no mode, and our own `k` (a shuffle after the merge
//! of a selection on the thread's parity), which reads only `thread_position_in_grid` and its
//! lane, so it has a flat variant of the call's threadgroups.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::cache::VariantKey;
use crate::device::requirements::VkFeature;
use crate::device::{Device, DevicePolicy, DeviceProfile};
use crate::lower::select::LowerInputs;
use crate::pipeline::artifact::{PrecisionNote, Reconvergence};
use crate::pipeline::sym::Stage;
use crate::pipeline::testing::{
    compile_with, engine_dir, link_and_validate, llvm_assemble, AssembleError,
};
use crate::spirv::testing::{committed_descriptions, committed_profiles, spirv_dis};

/// `k(b0, p, l)`: each thread stores 1 or 2 at `b0[p + 4096]` by its parity, then, past the merge
/// of that selection, the shuffle of the stored value from lane 0 plus its lane `l` at `b0[p]`;
/// `p` is its position in the grid, `l` its index in its simdgroup. The shuffle stands in a block
/// every thread runs together.
const AFTER_MERGE: &str = r#"define void @k(ptr addrspace(1) noundef "air-buffer-no-alias" %b0, i32 noundef %p, i32 noundef %l) {
entry:
  %odd = and i32 %p, 1
  %d = icmp eq i32 %odd, 0
  %i = zext i32 %p to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 %i
  %j = add i64 %i, 4096
  %q = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 %j
  br i1 %d, label %a, label %b

a:
  store i32 1, ptr addrspace(1) %q, align 4
  br label %m

b:
  store i32 2, ptr addrspace(1) %q, align 4
  br label %m

m:
  %v = phi i32 [ 1, %a ], [ 2, %b ]
  %s = tail call i32 @air.simd_shuffle.u.i32(i32 %v, i16 0)
  %t = add i32 %s, %l
  store i32 %t, ptr addrspace(1) %o, align 4
  ret void
}

declare i32 @air.simd_shuffle.u.i32(i32, i16)

!air.kernel = !{!0}
!air.version = !{!20}
!air.language_version = !{!21}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"b0"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"p"}
!5 = !{i32 2, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"l"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// `text` (an AIR module body) assembled, with the AIR target header.
fn assemble(name: &str, text: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-early-return-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, format!("{HEADER}{text}")).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}\n{text}"),
    }
}

/// The simdgroup barrier's plain module is `Guaranteed` with no note on every committed profile;
/// its partial-threadgroups variant is too where the profile has maximal reconvergence, and
/// records `ImplementationDefined` with `DeviceReconvergence` where it does not.
#[test]
fn a_simdgroup_barrier_s_variant_is_guaranteed_only_under_maximal_reconvergence() {
    let ll = engine_dir().join("a27_a4_simdgroup_barrier.air.ll");
    let bc = match llvm_assemble(&ll) {
        Ok(bc) => bc,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the fixture: {why}"),
    };
    let (mut with, mut without) = (0, 0);
    for (stem, p) in committed_profiles().expect("profiles") {
        let plain = compile_with(&bc, "a4_simdgroup_barrier", &p, &LowerInputs::new());
        let mut inputs = LowerInputs::new();
        inputs.partial_threadgroups = true;
        let variant = compile_with(&bc, "a4_simdgroup_barrier", &p, &inputs);
        let (Ok(plain), Ok(variant)) = (plain, variant) else {
            continue;
        };
        let state = |a: &crate::library::FunctionArtifact| {
            (
                a.reconvergence,
                a.precision.contains(&PrecisionNote::DeviceReconvergence),
            )
        };
        let guaranteed = (Reconvergence::Guaranteed, false);
        assert_eq!(state(&plain), guaranteed, "{stem}: the plain module");
        if p.fields().subgroup.maximal_reconvergence {
            assert_eq!(state(&variant), guaranteed, "{stem}: the variant");
            with += 1;
        } else {
            assert_eq!(
                state(&variant),
                (Reconvergence::ImplementationDefined, true),
                "{stem}: the variant"
            );
            without += 1;
        }
    }
    assert!(
        with > 0 && without > 0,
        "profiles compiled: {with} with maximal reconvergence, {without} without"
    );
}

/// [`AFTER_MERGE`]'s plain module states no mode and records `Guaranteed` with no note on every
/// committed profile, and on each with `shaderMaximalReconvergence` (and `shaderQuadControl`,
/// which requires it) cleared and with `shaderSubgroupUniformControlFlow` cleared too (no
/// committed profile without maximal
/// reconvergence takes a flat variant: it needs device addresses and full subgroups). Its flat
/// variant of the call's threadgroups (`VariantKey::flat_grid`, `VariantKey::call_threadgroups`),
/// whose invocations past the grid return before the body, takes the profile's mode wherever it
/// compiles: `MaximallyReconvergesKHR` and `Guaranteed` with maximal reconvergence, else
/// `SubgroupUniformControlFlowKHR` where offered, or no mode, with `ImplementationDefined` and
/// `DeviceReconvergence`. Both modules pass `spirv-val`.
#[test]
fn a_flat_grid_s_shuffle_after_a_merge_takes_the_profile_s_mode() {
    let Some(bc) = assemble("after_merge", AFTER_MERGE) else {
        return;
    };
    let mut key = VariantKey::new();
    key.flat_grid = true;
    key.call_threadgroups = true;
    let flat = key.lowering_inputs(Stage::Kernel);
    let reading = |inputs: &LowerInputs, stem: &str, p: &DeviceProfile| {
        let a = compile_with(&bc, "k", p, inputs).ok()?;
        let (_, words) = link_and_validate(&a, p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let text = spirv_dis(&words);
        let mode = match (
            text.contains("MaximallyReconvergesKHR"),
            text.contains("SubgroupUniformControlFlowKHR"),
        ) {
            (true, false) => "maximal",
            (false, true) => "ucf",
            (false, false) => "none",
            (true, true) => "both",
        };
        Some((
            mode,
            a.reconvergence,
            a.precision.contains(&PrecisionNote::DeviceReconvergence),
        ))
    };
    let mut profiles = committed_profiles().expect("profiles");
    // Quad control goes with maximal reconvergence: VK_KHR_shader_quad_control requires it
    // (`device::enables::check_invariants`).
    let cleared: [(&str, &[VkFeature]); 2] = [
        (
            "no-maximal",
            &[
                VkFeature::ShaderMaximalReconvergence,
                VkFeature::ShaderQuadControl,
            ],
        ),
        (
            "no-maximal-no-ucf",
            &[
                VkFeature::ShaderMaximalReconvergence,
                VkFeature::ShaderQuadControl,
                VkFeature::ShaderSubgroupUniformControlFlow,
            ],
        ),
    ];
    for (stem, desc) in committed_descriptions().expect("descriptions") {
        for (what, features) in cleared {
            let mut desc = desc.clone();
            for f in features {
                desc.features.remove(f);
            }
            let dev = Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES)
                .unwrap_or_else(|e| panic!("{stem} {what}: {e}"));
            profiles.push((format!("{stem} {what}"), dev.profile().clone()));
        }
    }
    let (mut with, mut without_ucf, mut without) = (0, 0, 0);
    for (stem, p) in profiles {
        let plain = reading(&LowerInputs::new(), &stem, &p)
            .unwrap_or_else(|| panic!("{stem}: the plain module refused"));
        assert_eq!(
            plain,
            ("none", Reconvergence::Guaranteed, false),
            "{stem}: the plain module"
        );
        let Some(flat) = reading(&flat, &stem, &p) else {
            continue;
        };
        let sg = p.fields().subgroup;
        let expected = if sg.maximal_reconvergence {
            with += 1;
            ("maximal", Reconvergence::Guaranteed, false)
        } else if sg.uniform_control_flow {
            without_ucf += 1;
            ("ucf", Reconvergence::ImplementationDefined, true)
        } else {
            without += 1;
            ("none", Reconvergence::ImplementationDefined, true)
        };
        assert_eq!(flat, expected, "{stem}: the flat module");
    }
    assert!(
        with > 0 && without_ucf > 0 && without > 0,
        "flat modules compiled: {with} with maximal reconvergence, {without_ucf} with only \
         subgroup-uniform control flow, {without} with neither"
    );
}
