//! A fragment function whose only tangled instructions take derivatives (`dfdx`, implicit-LOD
//! samples, LOD queries) and its reconvergence (`lower::simd::quad_loop`): it states no mode and
//! records `Guaranteed` with no `DeviceReconvergence` on every committed profile, but where one of
//! them lies in a loop a back edge re-enters from inside a selection, where it takes the mode the
//! profile gives a tangled entry and, without maximal reconvergence, the note. A fragment
//! function with a quad operation keeps its mode.
//!
//! `llvm-as` assembles the fixtures; without it the tests skip visibly.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::profile::DeviceProfile;
use crate::pipeline::artifact::{PrecisionNote, Reconvergence};
use crate::pipeline::testing::{
    compile, compile_kept, engine_dir, link_and_validate, llvm_assemble, AssembleError,
};
use crate::spirv::testing::{committed_profiles, spirv_dis};

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// A constexpr sampler: magnify nearest, minify linear, repeat.
const SAMPLER_WORDS: u64 = 34901797601020050;

fn assemble(name: &str, body: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-quad-loops-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("f.ll");
    std::fs::write(&ll, format!("{HEADER}{body}")).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}\n{body}"),
    }
}

fn fixture(stem: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{stem}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {stem}: {why}"),
    }
}

/// A fragment function `f` that, four trips, takes a value of its position and its trip number
/// that reads the quad (an implicit-LOD sample of texture slot 0, or `dfdx` where `derivative`)
/// and adds or subtracts it by its pixel column's parity. With `two_latches` each parity's block
/// branches back to the loop's header itself, so the loop is re-entered by back edges from
/// inside a selection; without, both join one latch first.
fn looped(two_latches: bool, derivative: bool) -> String {
    let take = if derivative {
        "%pv = fmul <4 x float> %pos, %s4\n  %v = tail call fast <4 x float> @air.dfdx.v4f32(<4 x float> %pv)"
    } else {
        "%r = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)\n  %v = extractvalue { <4 x float>, i8 } %r, 0"
    };
    let (incoming, arms, exit) = if two_latches {
        (
            "[ %i1, %a ], [ %i1, %b ]",
            "a:\n  %acc1 = fadd <4 x float> %acc, %v\n  br i1 %more, label %head, label %exit\n\n\
             b:\n  %acc2 = fsub <4 x float> %acc, %v\n  br i1 %more, label %head, label %exit\n",
            "exit:\n  %res = phi <4 x float> [ %acc1, %a ], [ %acc2, %b ]\n  ret <4 x float> %res\n",
        )
    } else {
        (
            "[ %i1, %latch ]",
            "a:\n  %acc1 = fadd <4 x float> %acc, %v\n  br label %latch\n\n\
             b:\n  %acc2 = fsub <4 x float> %acc, %v\n  br label %latch\n\n\
             latch:\n  %acc3 = phi <4 x float> [ %acc1, %a ], [ %acc2, %b ]\n  br i1 %more, label %head, label %exit\n",
            "exit:\n  ret <4 x float> %acc3\n",
        )
    };
    let accs = if two_latches {
        "[ %acc1, %a ], [ %acc2, %b ]"
    } else {
        "[ %acc3, %latch ]"
    };
    format!(
        r#"@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {SAMPLER_WORDS}, i64 0], align 8

define <4 x float> @f(<4 x float> %pos, ptr addrspace(1) %t) {{
entry:
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %x = extractelement <4 x float> %pos, i64 0
  %xi = fptoui float %x to i32
  %odd = and i32 %xi, 1
  %c = icmp eq i32 %odd, 0
  br label %head

head:
  %i = phi i32 [ 0, %entry ], {incoming}
  %acc = phi <4 x float> [ zeroinitializer, %entry ], {accs}
  %fi = uitofp i32 %i to float
  %s1 = insertelement <2 x float> poison, float %fi, i64 0
  %s2 = shufflevector <2 x float> %s1, <2 x float> poison, <2 x i32> zeroinitializer
  %s3 = insertelement <4 x float> poison, float %fi, i64 0
  %s4 = shufflevector <4 x float> %s3, <4 x float> poison, <4 x i32> zeroinitializer
  %uv0 = fmul <2 x float> %xy, <float 6.250000e-02, float 6.250000e-02>
  %uv = fadd <2 x float> %uv0, %s2
  {take}
  %i1 = add i32 %i, 1
  %more = icmp ult i32 %i1, 4
  br i1 %c, label %a, label %b

{arms}
{exit}}}

declare {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare <4 x float> @air.dfdx.v4f32(<4 x float>)

!air.fragment = !{{!0}}
!air.sampler_states = !{{!6}}
!0 = !{{ptr @f, !1, !3}}
!1 = !{{!2}}
!2 = !{{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}}
!3 = !{{!4, !5}}
!4 = !{{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}}
!5 = !{{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}}
!6 = !{{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}}
"#
    )
}

/// The mode `entry` of `bc` states on `p`, its reconvergence, whether it records
/// `DeviceReconvergence`, and whether its module holds a quad-scoped instruction in a re-entered
/// loop; its words linked and validated (`spirv-val`).
fn reading(
    bc: &[u8],
    entry: &str,
    stem: &str,
    p: &DeviceProfile,
) -> (&'static str, Reconvergence, bool, bool) {
    let (a, _, m) = compile_kept(bc, entry, p).unwrap_or_else(|e| panic!("{stem} {entry}: {e}"));
    let (_, words) = link_and_validate(&a, p).unwrap_or_else(|e| panic!("{stem} {entry}: L3: {e}"));
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
    let noted = a.precision.contains(&PrecisionNote::DeviceReconvergence);
    (
        mode,
        a.reconvergence,
        noted,
        crate::lower::simd::quad_loop::reentered(&m),
    )
}

/// A fragment function's derivatives in selections, after a selection's merge, after a loop and
/// inside a loop one latch closes state no mode and record `Guaranteed` with no
/// `DeviceReconvergence`, on every committed profile, as no loop re-enters them.
#[test]
fn derivatives_outside_a_re_entered_loop_state_no_mode_and_no_note() {
    let Some(tangle) = fixture("rch_helper_tangle") else {
        return;
    };
    let Some(one_latch) = assemble("one_latch", &looped(false, false)) else {
        return;
    };
    let Some(one_latch_dfdx) = assemble("one_latch_dfdx", &looped(false, true)) else {
        return;
    };
    let cases: [(&[u8], &str); 9] = [
        (&tangle, "rch_tangle_sample_after"),
        (&tangle, "rch_tangle_dfdx_after"),
        (&tangle, "rch_tangle_lod_after"),
        (&tangle, "rch_tangle_sample_before"),
        (&tangle, "rch_tangle_unread"),
        (&tangle, "rch_tangle_merge_after"),
        (&tangle, "rch_tangle_loop_after"),
        (&one_latch, "f"),
        (&one_latch_dfdx, "f"),
    ];
    for (stem, p) in committed_profiles().expect("profiles") {
        for (bc, entry) in cases {
            let (mode, reconvergence, noted, reentered) = reading(bc, entry, &stem, &p);
            assert_eq!(
                (mode, reconvergence, noted, reentered),
                ("none", Reconvergence::Guaranteed, false, false),
                "{stem} {entry}"
            );
        }
    }
}

/// A fragment function whose implicit-LOD sample or `dfdx` lies in a loop that two latches,
/// each inside an arm of a selection on the pixel's column, re-enter takes the profile's mode:
/// `MaximallyReconvergesKHR` and `Guaranteed` with maximal reconvergence; else
/// `SubgroupUniformControlFlowKHR` where offered, or no mode, with `ImplementationDefined` and
/// `DeviceReconvergence`.
#[test]
fn a_derivative_in_a_re_entered_loop_keeps_the_mode_and_the_note() {
    let Some(sample) = assemble("two_latches", &looped(true, false)) else {
        return;
    };
    let Some(dfdx) = assemble("two_latches_dfdx", &looped(true, true)) else {
        return;
    };
    for (stem, p) in committed_profiles().expect("profiles") {
        let sg = p.fields().subgroup;
        let expected = if sg.maximal_reconvergence {
            ("maximal", Reconvergence::Guaranteed, false, true)
        } else if sg.uniform_control_flow {
            ("ucf", Reconvergence::ImplementationDefined, true, true)
        } else {
            ("none", Reconvergence::ImplementationDefined, true, true)
        };
        for bc in [&sample, &dfdx] {
            assert_eq!(reading(bc, "f", &stem, &p), expected, "{stem}");
        }
    }
}

/// A fragment function with a quad operation keeps the mode wherever it compiles: its quad sum
/// reconverges, whatever its samples do.
#[test]
fn a_fragment_quad_operation_keeps_the_mode() {
    let Some(tangle) = fixture("rch_helper_tangle") else {
        return;
    };
    let mut compiled = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        if !p.fields().subgroup.maximal_reconvergence {
            continue;
        }
        for entry in ["rch_tangle_quad_after", "rch_tangle_quad_before"] {
            if compile(&tangle, entry, &p).is_err() {
                continue;
            }
            compiled += 1;
            let (mode, reconvergence, noted, _) = reading(&tangle, entry, &stem, &p);
            assert_eq!(
                (mode, reconvergence, noted),
                ("maximal", Reconvergence::Guaranteed, false),
                "{stem} {entry}"
            );
        }
    }
    assert!(compiled > 0, "no profile compiled a quad operation");
}
