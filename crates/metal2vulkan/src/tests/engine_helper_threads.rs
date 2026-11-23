//! `quad_is_helper_thread` and `simd_is_helper_thread` end to end.
//!
//! The owned fixtures `engine/gsm2_helper_threads` (MSL 4.0) and
//! `engine/gsm2_helper_threads_msl22` (MSL 2.2) go through the whole pipeline:
//! - the kernel runs on the CPU interpreter in every kernel tier, and no thread is a helper,
//!   Metal's answer (measured on the M3 Max, over full and partial quads);
//! - the vertex function compiles to the constant `false`, with nothing to ask;
//! - each fragment function compiles, passes L3, and asks in the form its discards take:
//!   `OpIsHelperInvocationEXT` where a discard demotes, which a demoted invocation answers too;
//!   else the `HelperInvocation` input, or'd with a predicated discard's `discarded`, on
//!   `moltenvk-m3` with and without `shaderDemoteToHelperInvocation`. Where the toolchain is
//!   present, MoltenVK's MSL conversion of each module compiles.
//!
//! No GPU is used. `llvm-as` assembles the fixtures; a missing tool skips visibly.

use std::collections::BTreeMap;

use crate::device::requirements::VkFeature;
use crate::device::subgroup::Tier;
use crate::device::{DevicePolicy, DeviceProfile};
use crate::lower::fragment::DiscardForm;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, moltenvk_msl_compile, profile,
    typed_module, AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources, SubgroupSize};

fn fixture(name: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{name}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}"),
    }
}

/// The operands of every instruction of opcode `op` in `words` (after the header).
fn instructions(words: &[u32], op: crate::spv::Op) -> Vec<Vec<u32>> {
    let mut out = Vec::new();
    let mut at = 5;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        if count == 0 {
            break;
        }
        if words[at] & 0xffff == op as u32 {
            out.push(words[at + 1..at + count].to_vec());
        }
        at += count;
    }
    out
}

/// How a function's helper queries compiled: the form of its discard sites (`None` without one),
/// its `OpIsHelperInvocationEXT` instructions, and whether it reads the `HelperInvocation` input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Asked {
    form: Option<DiscardForm>,
    queries: usize,
    input: bool,
}

/// What `entry` of `bc` compiles to on `p`, after L3 and, on a MoltenVK profile where the
/// toolchain is present, MoltenVK's MSL compile of the module.
fn asked(bc: &[u8], entry: &str, stem: &str, p: &DeviceProfile) -> Asked {
    let (plan, _) = typed_module(bc, entry, p).unwrap_or_else(|e| panic!("{entry} on {stem}: {e}"));
    let fp = &plan.families.fragment;
    let mut forms = fp.discards.values().copied().collect::<Vec<_>>();
    forms.dedup();
    assert!(forms.len() <= 1, "{entry} on {stem}: {forms:?}");
    let a = compile(bc, entry, p).unwrap_or_else(|e| panic!("{entry} on {stem}: {e}"));
    let (_, words) =
        link_and_validate(&a, p).unwrap_or_else(|e| panic!("{entry} on {stem}: L3: {e}"));
    match stem
        .starts_with("moltenvk-")
        .then(|| moltenvk_msl_compile(&words))
    {
        None | Some(Ok(())) => {}
        Some(Err(AssembleError::Missing(why))) => eprintln!("SKIPPED: {why}"),
        Some(Err(AssembleError::Rejected(why))) => panic!("{entry} on {stem}: MSL: {why}"),
    }
    let input = instructions(&words, crate::spv::Op::Decorate)
        .iter()
        .any(|d| {
            d.get(1) == Some(&(crate::spv::Decoration::BuiltIn as u32))
                && d.get(2) == Some(&(crate::spv::BuiltIn::HelperInvocation as u32))
        });
    Asked {
        form: forms.first().copied(),
        queries: instructions(&words, crate::spv::Op::IsHelperInvocationEXT).len(),
        input,
    }
}

/// Each fragment function on `intel-arl-anv` (demote and `shaderQuadControl`) and on it without
/// demote, and the expected form, queries and input of each: the after and before cases ask twice
/// (`quad_` and `simd_`).
#[test]
fn fragment_helper_queries_take_the_form_of_their_discards() {
    let (Some(msl4), Some(msl22)) = (
        fixture("gsm2_helper_threads"),
        fixture("gsm2_helper_threads_msl22"),
    ) else {
        return;
    };
    let stem = "intel-arl-anv";
    let with = profile(stem, DevicePolicy::RESIDENT_TABLES).expect("profile");
    let without = {
        let mut desc = crate::pipeline::testing::description(stem).expect("description");
        let demote = VkFeature::ShaderDemoteToHelperInvocation;
        assert!(desc.features.remove(&demote), "{stem} has {demote:?}");
        crate::device::Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES)
            .expect("device")
            .profile()
            .clone()
    };
    let asks = |form, queries, input| Asked {
        form,
        queries,
        input,
    };
    let none = asks(None, 0, true);
    let demoted = asks(Some(DiscardForm::Demote), 2, false);
    let predicated = asks(Some(DiscardForm::Predicate), 0, true);
    let terminated = asks(Some(DiscardForm::Terminate), 0, true);
    let cases: [(&[u8], &str, Asked, Asked); 6] = [
        (&msl4, "fragment_helper_threads", none, none),
        (
            &msl4,
            "fragment_helper_threads_after_a_discard",
            demoted,
            predicated,
        ),
        (
            &msl4,
            "fragment_helper_threads_before_a_discard",
            demoted,
            predicated,
        ),
        // A simdgroup vote after the discard predicates every site, whatever the profile, and
        // exits a quad of discarded invocations and helpers before the vote.
        (
            &msl4,
            "fragment_helper_threads_beside_a_simdgroup_vote",
            predicated,
            predicated,
        ),
        // At MSL 2.2 a discard demotes where something tangled follows it, and terminates where
        // only the helper query, which is not tangled, does: a live invocation is no helper.
        (
            &msl22,
            "fragment_helper_threads_after_a_terminating_discard",
            demoted,
            predicated,
        ),
        (
            &msl22,
            "fragment_helper_thread_alone_after_a_terminating_discard",
            terminated,
            terminated,
        ),
    ];
    for (bc, entry, on_with, on_without) in cases {
        assert_eq!(asked(bc, entry, stem, &with), on_with, "{entry}");
        assert_eq!(
            asked(bc, entry, &format!("{stem} without demote"), &without),
            on_without,
            "{entry}"
        );
    }
}

/// A fragment function that discards, asks whether its thread is a helper and ballots the answers
/// over its quad records `PrecisionNote::FragmentSimdgroupGrouping` on every committed profile that
/// compiles it (the Radeon 780M's and Intel's ANV among them): an Apple GPU shades some quads of
/// such a function as several partial quads, which the device's quads are not. The same function
/// without the discard (`fragment_helper_threads`), and a function that discards and asks with no
/// quad operation (`fragment_helper_thread_alone_after_a_terminating_discard`), record none
/// anywhere.
#[test]
fn a_discarding_function_s_helper_ballots_record_the_grouping() {
    use crate::pipeline::artifact::PrecisionNote;
    use crate::spirv::testing::committed_profiles;
    let (Some(msl4), Some(msl22)) = (
        fixture("gsm2_helper_threads"),
        fixture("gsm2_helper_threads_msl22"),
    ) else {
        return;
    };
    let note = PrecisionNote::FragmentSimdgroupGrouping;
    let discarding: [(&[u8], &str); 3] = [
        (&msl4, "fragment_helper_threads_after_a_discard"),
        (&msl4, "fragment_helper_threads_before_a_discard"),
        (
            &msl22,
            "fragment_helper_threads_after_a_terminating_discard",
        ),
    ];
    let quiet: [(&[u8], &str); 2] = [
        (&msl4, "fragment_helper_threads"),
        (
            &msl22,
            "fragment_helper_thread_alone_after_a_terminating_discard",
        ),
    ];
    let mut noted = BTreeMap::new();
    for (name, p) in committed_profiles().expect("profiles") {
        for (bc, entry) in discarding {
            if let Ok(a) = compile(bc, entry, &p) {
                assert!(a.precision.contains(&note), "{entry} on {name}");
                *noted.entry(name.clone()).or_insert(0) += 1;
            }
        }
        for (bc, entry) in quiet {
            if let Ok(a) = compile(bc, entry, &p) {
                assert!(!a.precision.contains(&note), "{entry} on {name}");
            }
        }
    }
    // The 780M's and the Arrow Lake's profiles compile all three and record the note.
    for stem in ["amd-780m-radv", "intel-arl-anv"] {
        assert_eq!(noted.get(stem), Some(&3), "{stem}: {noted:?}");
    }
}

/// A vertex function asks nothing: no vertex is a helper.
#[test]
fn a_vertex_function_s_helper_query_is_false() {
    let Some(bc) = fixture("gsm2_helper_threads") else {
        return;
    };
    let entry = "helper_threads_in_a_vertex_function";
    for stem in ["moltenvk-m3", "lavapipe"] {
        let p = profile(stem, DevicePolicy::RESIDENT_TABLES).expect("profile");
        let a = compile(&bc, entry, &p).unwrap_or_else(|e| panic!("{entry} on {stem}: {e}"));
        let (_, words) =
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{entry} on {stem}: L3: {e}"));
        assert!(
            instructions(&words, crate::spv::Op::IsHelperInvocationEXT).is_empty(),
            "{stem}"
        );
    }
}

/// The kernel's words after one threadgroup of `local` threads at `lanes`, in both scheduling
/// orders: each thread's answers and its quad's and simdgroup's votes on them, with 0x100 set.
fn kernel_words(p: &DeviceProfile, tier: Tier, lanes: SubgroupSize, local: u32) -> Vec<u8> {
    let Some(bc) = fixture("gsm2_helper_threads") else {
        return Vec::new();
    };
    let entry = "helper_threads_in_a_kernel";
    let (plan, m) = typed_module(&bc, entry, p).unwrap_or_else(|e| panic!("{entry}: {e}"));
    assert_eq!(plan.simd.1, tier, "{entry}");
    let mut spec = BTreeMap::new();
    spec.extend(crate::pipeline::threadgroup_constants([local, 1, 1]));
    let res = Resources {
        buffers: vec![Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: 1,
            }),
            address: None,
            bytes: vec![0xAB; 4 * local as usize],
        }],
        push: vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)],
        spec,
    };
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups: [1, 1, 1],
            subgroup: lanes,
            order,
            ..Config::default()
        };
        outs.push(run(&m, &cfg, res.clone()).expect("run").buffers[0].clone());
    }
    assert_eq!(outs[0], outs[1], "{entry}: order independent");
    outs.swap_remove(0)
}

/// No kernel thread is a helper, in any tier, over full quads and a threadgroup of 42 whose last
/// quad holds two threads.
fn kernel_threads_are_no_helpers(p: &DeviceProfile, tier: Tier, widths: &[SubgroupSize]) {
    for &lanes in widths {
        for local in [64, 42] {
            let got = kernel_words(p, tier, lanes, local);
            if got.is_empty() {
                return;
            }
            let want: Vec<u8> = (0..local).flat_map(|_| 0x100u32.to_le_bytes()).collect();
            assert_eq!(got, want, "{local} threads at {lanes:?}");
        }
    }
}

#[test]
fn kernel_threads_are_no_helpers_direct() {
    let p = profile("moltenvk-m3", DevicePolicy::RESIDENT_TABLES).expect("profile");
    kernel_threads_are_no_helpers(&p, Tier::Direct, &[SubgroupSize::W32]);
}

#[test]
fn kernel_threads_are_no_helpers_cluster32() {
    kernel_threads_are_no_helpers(
        &super::engine_simd::wave64(),
        Tier::Cluster32,
        &[SubgroupSize::W64],
    );
}

#[test]
fn kernel_threads_are_no_helpers_emulated() {
    let p = profile("lavapipe", DevicePolicy::RESIDENT_TABLES).expect("profile");
    kernel_threads_are_no_helpers(
        &p,
        Tier::Emulated,
        &[
            SubgroupSize::W8,
            SubgroupSize::W16,
            SubgroupSize::W32,
            SubgroupSize::W64,
        ],
    );
}
