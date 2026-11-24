//! A kernel's tangled instructions that its threads reach together wherever SPIR-V reconverges
//! them, and its reconvergence (`lower::simd::reentry`): a threadgroup barrier, which Metal
//! requires every thread of the threadgroup to reach, and a group operation in a block the
//! workgroup uniformity proves every thread runs together. An entry whose tangled instructions
//! are all such states no mode and records `Guaranteed` with no `DeviceReconvergence` on every
//! committed profile, but where one of them lies in a loop a back edge re-enters from inside a
//! selection, where it takes the mode the profile gives a tangled entry. A group operation after
//! a selection inside a per-thread branch, or in a loop whose exit differs between threads,
//! keeps the mode.
//!
//! The kernels are our own: `k(device uint *out [[buffer(0)]], uint t
//! [[thread_index_in_threadgroup]])`, each writing one value to `out[t]`.
//!
//! `llvm-as` assembles the fixtures; without it the tests skip visibly.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::profile::DeviceProfile;
use crate::pipeline::artifact::{PrecisionNote, Reconvergence};
use crate::pipeline::testing::{compile_kept, link_and_validate, llvm_assemble, AssembleError};
use crate::spirv::testing::{committed_profiles, spirv_dis};

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// Kernel `k` with `body`: the blocks from `entry` on, `%0` the buffer `out` and `%1` the
/// thread's index in its threadgroup; the body ends by storing to `out[t]` and returning.
fn kernel(name: &str, body: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let text = format!(
        "{HEADER}define void @k(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, i32 noundef %1) {{\nentry:\n{body}}}\n\n\
         declare void @air.wg.barrier(i32, i32)\ndeclare i32 @air.simd_shuffle.u.i32(i32, i16)\n\n\
         !air.kernel = !{{!9}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\
         !9 = !{{ptr @k, !10, !11}}\n!10 = !{{}}\n!11 = !{{!12, !13}}\n\
         !12 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}}\n\
         !13 = !{{i32 1, !\"air.thread_index_in_threadgroup\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"t\"}}\n\
         !17 = !{{i32 2, i32 8, i32 0}}\n!18 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n"
    );
    let dir = std::env::temp_dir().join(format!(
        "m2v-reconvergence-together-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, text).expect("write");
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

/// The store of `%r` to `out[t]` and the return, as the last block's tail.
const STORE: &str = "  %e = zext i32 %1 to i64\n  %p = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %e\n  store i32 %r, ptr addrspace(1) %p, align 4\n  ret void\n";

/// A threadgroup barrier and nothing else tangled.
fn barrier_alone() -> String {
    format!("  call void @air.wg.barrier(i32 2, i32 1)\n  %r = add i32 %1, 1\n{STORE}")
}

/// `take` (defining `%v` from `%acc`) at the head of a loop of four trips whose two latches, one
/// in each arm of a selection on the trip's parity, each branch back to the head or out: every
/// thread takes the same arm each trip, and the loop is re-entered by back edges from inside a
/// selection. (On the thread's parity instead, the loop's exit differs between threads, and the
/// kernel's barriers take the predicated walk, whose loop one latch closes.)
fn re_entered(take: &str) -> String {
    format!(
        "  br label %head\n\n\
         head:\n  %i = phi i32 [ 0, %entry ], [ %i1, %a ], [ %i1, %b ]\n\
         %acc = phi i32 [ %1, %entry ], [ %x1, %a ], [ %x2, %b ]\n  {take}\n\
         %i1 = add i32 %i, 1\n  %more = icmp ult i32 %i1, 4\n  %odd = and i32 %i, 1\n\
         %c = icmp eq i32 %odd, 0\n  br i1 %c, label %a, label %b\n\n\
         a:\n  %x1 = add i32 %v, 1\n  br i1 %more, label %head, label %exit\n\n\
         b:\n  %x2 = mul i32 %v, 3\n  br i1 %more, label %head, label %exit\n\n\
         exit:\n  %r = phi i32 [ %x1, %a ], [ %x2, %b ]\n{STORE}"
    )
}

/// A threadgroup barrier at the head of the re-entered loop.
const BARRIER_TAKE: &str = "call void @air.wg.barrier(i32 2, i32 1)\n  %v = add i32 %acc, 0";

/// A shuffle at the head of the re-entered loop, reading lane 0.
const SHUFFLE_TAKE: &str = "%v = tail call i32 @air.simd_shuffle.u.i32(i32 %acc, i16 0)";

/// A shuffle in a loop of four trips, each reading lane `i` of the running value.
fn shuffle_counted() -> String {
    format!(
        "  br label %head\n\n\
         head:\n  %i = phi i32 [ 0, %entry ], [ %i1, %head ]\n  %acc = phi i32 [ %1, %entry ], [ %s, %head ]\n\
         %l = trunc i32 %i to i16\n  %s = tail call i32 @air.simd_shuffle.u.i32(i32 %acc, i16 %l)\n\
         %i1 = add i32 %i, 1\n  %more = icmp ult i32 %i1, 4\n  br i1 %more, label %head, label %exit\n\n\
         exit:\n  %r = add i32 %s, 0\n{STORE}"
    )
}

/// A shuffle after the merge of a selection on the thread's parity, inside a branch only threads
/// below 8 take, which also stores its result: the store keeps the shuffle in the branch (with
/// the store after the branch, the shuffle is hoisted out of it, to the entry).
fn shuffle_after_merge_in_branch() -> String {
    "  %c = icmp ult i32 %1, 8\n  br i1 %c, label %in, label %done\n\n\
     in:\n  %odd = and i32 %1, 1\n  %d = icmp eq i32 %odd, 0\n  br i1 %d, label %a, label %b\n\n\
     a:\n  br label %m\n\nb:\n  br label %m\n\n\
     m:\n  %v = phi i32 [ 1, %a ], [ 2, %b ]\n  %s = tail call i32 @air.simd_shuffle.u.i32(i32 %v, i16 0)\n\
     %e = zext i32 %1 to i64\n  %p = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %e\n\
     store i32 %s, ptr addrspace(1) %p, align 4\n  br label %done\n\n\
     done:\n  ret void\n"
        .to_string()
}

/// A shuffle in a loop whose exit compares the trip with the thread's index.
fn shuffle_per_thread_exit() -> String {
    format!(
        "  br label %head\n\n\
         head:\n  %i = phi i32 [ 0, %entry ], [ %i1, %head ]\n  %acc = phi i32 [ %1, %entry ], [ %s, %head ]\n\
         %s = tail call i32 @air.simd_shuffle.u.i32(i32 %acc, i16 0)\n\
         %i1 = add i32 %i, 1\n  %more = icmp ult i32 %i1, %1\n  br i1 %more, label %head, label %exit\n\n\
         exit:\n  %r = add i32 %s, 0\n{STORE}"
    )
}

/// What the kernel states on `p`: its mode, its reconvergence, whether it records
/// `DeviceReconvergence`, whether its module holds a tangled instruction in a re-entered loop,
/// and whether it holds a subgroup shuffle (a subgroup tier, not the emulated one); its words
/// linked and validated (`spirv-val`). `None` where the profile refuses the kernel.
fn reading(
    bc: &[u8],
    stem: &str,
    p: &DeviceProfile,
) -> Option<(&'static str, Reconvergence, bool, bool, bool)> {
    let (a, _, m) = compile_kept(bc, "k", p).ok()?;
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
        crate::lower::simd::reentry::reentered(&m),
        text.contains("OpGroupNonUniformShuffle "),
    ))
}

/// The mode, reconvergence and note a tangled entry takes on `p`.
fn tangled_on(p: &DeviceProfile) -> (&'static str, Reconvergence, bool) {
    let sg = p.fields().subgroup;
    if sg.maximal_reconvergence {
        ("maximal", Reconvergence::Guaranteed, false)
    } else if sg.uniform_control_flow {
        ("ucf", Reconvergence::ImplementationDefined, true)
    } else {
        ("none", Reconvergence::ImplementationDefined, true)
    }
}

/// A kernel whose only tangled instruction is a threadgroup barrier states no mode and records
/// `Guaranteed` with no `DeviceReconvergence` on every committed profile.
#[test]
fn a_threadgroup_barrier_alone_states_no_mode() {
    let Some(bc) = kernel("barrier_alone", &barrier_alone()) else {
        return;
    };
    let mut maximal = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let r = reading(&bc, &stem, &p).unwrap_or_else(|| panic!("{stem}: refused"));
        assert_eq!(
            (r.0, r.1, r.2, r.3),
            ("none", Reconvergence::Guaranteed, false, false),
            "{stem}"
        );
        maximal += usize::from(p.fields().subgroup.maximal_reconvergence);
    }
    assert!(
        maximal > 0,
        "no committed profile has maximal reconvergence"
    );
}

/// A threadgroup barrier in a loop two latches inside a selection re-enter takes the profile's
/// mode: `MaximallyReconvergesKHR` and `Guaranteed` with maximal reconvergence; else
/// `SubgroupUniformControlFlowKHR` where offered, or no mode, with `ImplementationDefined` and
/// `DeviceReconvergence`. So does a shuffle there, wherever it compiles to a subgroup shuffle.
#[test]
fn a_tangled_instruction_in_a_re_entered_loop_keeps_the_mode() {
    let Some(barrier) = kernel("barrier_re_entered", &re_entered(BARRIER_TAKE)) else {
        return;
    };
    let Some(shuffle) = kernel("shuffle_re_entered", &re_entered(SHUFFLE_TAKE)) else {
        return;
    };
    let mut shuffled = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let (mode, reconvergence, noted) = tangled_on(&p);
        let r = reading(&barrier, &stem, &p).unwrap_or_else(|| panic!("{stem}: refused"));
        assert_eq!(
            (r.0, r.1, r.2, r.3),
            (mode, reconvergence, noted, true),
            "{stem} barrier"
        );
        let Some(r) = reading(&shuffle, &stem, &p) else {
            continue;
        };
        if !r.4 {
            continue;
        }
        shuffled += 1;
        assert_eq!(
            (r.0, r.1, r.2, r.3),
            (mode, reconvergence, noted, true),
            "{stem} shuffle"
        );
    }
    assert!(shuffled > 0, "no profile compiled a subgroup shuffle");
}

/// A shuffle in a loop of a constant trip count stands where every thread of the threadgroup
/// runs together: the kernel states no mode and records `Guaranteed` on every profile that
/// compiles it, a subgroup shuffle on some.
#[test]
fn a_shuffle_in_a_counted_loop_states_no_mode() {
    let Some(bc) = kernel("shuffle_counted", &shuffle_counted()) else {
        return;
    };
    let mut shuffled = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let Some(r) = reading(&bc, &stem, &p) else {
            continue;
        };
        assert_eq!(
            (r.0, r.1, r.2, r.3),
            ("none", Reconvergence::Guaranteed, false, false),
            "{stem}"
        );
        shuffled += usize::from(r.4 && p.fields().subgroup.maximal_reconvergence);
    }
    assert!(
        shuffled > 0,
        "no maximal profile compiled a subgroup shuffle"
    );
}

/// A shuffle after a merge inside a branch some threads skip, and a shuffle in a loop whose
/// exit differs between threads, keep the profile's mode wherever they compile to a subgroup
/// shuffle.
#[test]
fn a_shuffle_outside_uniform_control_flow_keeps_the_mode() {
    let Some(merged) = kernel("shuffle_after_merge", &shuffle_after_merge_in_branch()) else {
        return;
    };
    let Some(exits) = kernel("shuffle_per_thread_exit", &shuffle_per_thread_exit()) else {
        return;
    };
    let mut shuffled = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        for (case, bc) in [("after a merge", &merged), ("per-thread exit", &exits)] {
            let Some(r) = reading(bc, &stem, &p) else {
                continue;
            };
            if !r.4 {
                continue;
            }
            shuffled += 1;
            let (mode, reconvergence, noted) = tangled_on(&p);
            assert_eq!(
                (r.0, r.1, r.2),
                (mode, reconvergence, noted),
                "{stem} {case}"
            );
        }
    }
    assert!(shuffled > 0, "no profile compiled a subgroup shuffle");
}
