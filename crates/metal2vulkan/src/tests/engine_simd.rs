//! Simdgroup and quad operations end to end.
//!
//! The owned fixtures `engine/simd_ops` and `engine/quad_shuffle` go through the whole
//! pipeline and run on the CPU interpreter with a modelled subgroup (the CI tier): the `Direct`
//! tier at 32 lanes on `moltenvk-m3`, and the `Cluster32` tier at 64 lanes on a wave64-only
//! device (the committed `amd-780m-radv` description with `minSubgroupSize` 64, the shape of
//! AMD GCN). Every kernel's output is checked against expectations derived from
//! `thread_index_in_threadgroup` alone, in both scheduling orders, for several local sizes and
//! for divergent lanes. On every committed profile each kernel compiles, links, and passes L3,
//! or is refused exactly as its tier is (`SimdWidth` where 32 lanes cannot be pinned). No
//! GPU is used.
//!
//! `llvm-as` assembles the fixtures; a missing tool skips visibly.

use std::collections::BTreeMap;

use crate::air::error::IntrinsicError;
use crate::device::requirements::{StageCreateFlag, VkExtension, VkFeature, VkLimit, VkProperty};
use crate::device::subgroup::{self, OpBits, SimdClass, SubgroupRequirement, Tier};
use crate::device::{Device, DeviceDescription, DevicePolicy, DeviceProfile};
use crate::error::{Error, Feature, InputErrorKind};
use crate::lower::dispatch::LocalSizeForm;
use crate::lower::select::Plan;
use crate::pipeline::artifact::{PrecisionNote, Reconvergence};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::{
    compile, compile_kept, engine_dir, link_and_validate, llvm_assemble, profile, typed_module,
    AssembleError, Scratch,
};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources, SubgroupSize};
use crate::spirv::ir::TypedModule;
use crate::spirv::testing::{committed_profiles, spirv_dis};

fn assemble(stem: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{stem}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {stem}.ll: {why}"),
    }
}

/// A wave64-only compute device: the committed RDNA3 description with its subgroup size range
/// narrowed to 64 (the `Cluster32` tier).
pub(super) fn wave64() -> DeviceProfile {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/amd-780m-radv.json");
    let text = std::fs::read_to_string(&path).expect("read the RDNA3 description");
    let narrowed = text.replace("\"minSubgroupSize\": 32", "\"minSubgroupSize\": 64");
    assert_ne!(narrowed, text, "the description names minSubgroupSize");
    let desc = DeviceDescription::from_khronos_json(&narrowed, None).expect("description");
    Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES)
        .expect("device")
        .profile()
        .clone()
}

/// The committed `moltenvk-m3` description without `shaderInt64`: a 64-bit ballot is held as two
/// words.
fn moltenvk_without_int64() -> DeviceProfile {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/moltenvk-m3.json");
    let text = std::fs::read_to_string(&path).expect("read the MoltenVK description");
    let cleared = text.replace("\"shaderInt64\": true", "\"shaderInt64\": false");
    assert_ne!(cleared, text, "the description names shaderInt64");
    let desc = DeviceDescription::from_khronos_json(&cleared, None).expect("description");
    let p = Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES)
        .expect("device")
        .profile()
        .clone();
    assert!(!p.fields().leaf.int64);
    p
}

fn moltenvk() -> DeviceProfile {
    profile("moltenvk-m3", DevicePolicy::RESIDENT_TABLES).expect("profile")
}

/// Every word of buffer 0 after one workgroup of local size `local` ran `m` (planned as `plan`)
/// with `lanes`-wide subgroups, identical in both scheduling orders.
fn run_all(plan: &Plan, m: &TypedModule, local: [u32; 3], lanes: SubgroupSize) -> Vec<u32> {
    let mut spec = BTreeMap::new();
    spec.extend(crate::pipeline::threadgroup_constants(local));
    let n: u32 = local.iter().product();
    let push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        assert!(matches!(f.loc, PushLoc::Push { .. }), "{f:?}");
        // A zero residual: buffer 0 is bound at its start.
        assert!(
            matches!(f.field, PushField::SlotBindOffset(_)),
            "unexpected push field {f:?}"
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
            bytes: vec![0; n as usize * 4],
        }],
        push,
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
        let out = run(m, &cfg, res.clone()).expect("run");
        let words: Vec<u32> = out.buffers[0]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        outs.push(words);
    }
    assert_eq!(outs[0], outs[1], "order independent");
    outs.swap_remove(0)
}

fn lane(t: u32) -> u32 {
    t & 31
}

fn base(t: u32) -> u32 {
    t & !31
}

/// `f` folded over the threads of `t`'s simdgroup that satisfy `active`, in lane order.
fn over_simd(t: u32, active: &dyn Fn(u32) -> bool) -> Vec<u32> {
    (base(t)..base(t) + 32).filter(|&u| active(u)).collect()
}

/// The expected `out[t]` of each `simd_ops` kernel (see the fixture's MSL).
fn expected(kernel: &str, t: u32) -> u32 {
    let l = lane(t);
    let b = base(t);
    let q = t & 3;
    let qb = t & !3;
    let all = |_| true;
    match kernel {
        "simd_shuffle_k" => b + ((l * 7 + 3) & 31),
        "simd_broadcast_k" => b + 9,
        "simd_shuffle_up_k" => {
            if l >= 3 {
                t - 3
            } else {
                t
            }
        }
        "simd_shuffle_down_k" => {
            if l + 5 < 32 {
                t + 5
            } else {
                t
            }
        }
        "simd_shuffle_xor_k" => t ^ 13,
        "simd_rotate_k" => b + ((l + 7) & 31),
        "simd_fill_down_k" => {
            let s = l % 8;
            if s + 3 < 8 {
                t + 3
            } else {
                t - s + (s + 3) % 8 + 1000
            }
        }
        "simd_fill_up_k" => {
            if l >= 2 {
                t - 2
            } else {
                t - l + (l + 30) % 32 + 1000
            }
        }
        "simd_sum_k" => {
            let s: u32 = over_simd(t, &all).iter().sum();
            let f: u32 = over_simd(t, &all).iter().map(|u| u & 7).sum();
            s + f * 100_000
        }
        "simd_half_sum_k" => 496,
        "simd_minmax_k" => (1000 - 20) + (1000 + 11) * 10_000,
        "simd_bits_k" => {
            let or = over_simd(t, &all)
                .iter()
                .fold(0u32, |a, u| a | (1 << (u & 15)));
            let and = over_simd(t, &all)
                .iter()
                .fold(0xFFFFu32, |a, u| a & ((u | 0xF0) & 0xFFFF));
            or | (and << 16)
        }
        "simd_xor_k" => over_simd(t, &all)
            .iter()
            .fold(0u32, |a, u| a ^ ((u * u) & 0xFFFF)),
        "simd_prefix_k" => {
            let active = |u: u32| !lane(u).is_multiple_of(3);
            if !active(t) {
                return 7;
            }
            let incl: u32 = over_simd(t, &active).iter().filter(|&&u| u <= t).sum();
            let excl: u32 = over_simd(t, &active)
                .iter()
                .filter(|&&u| u < t)
                .map(|&u| lane(u))
                .sum();
            incl * 2 + excl * 100_000
        }
        "simd_float_prefix_k" => {
            if l < 4 {
                return 7;
            }
            (4..l).sum()
        }
        "simd_vote_k" => {
            let active = |u: u32| lane(u) != 6;
            if !active(t) {
                return 0;
            }
            let lanes = over_simd(t, &active);
            u32::from(lanes.iter().all(|&u| lane(u) != 6))
                + u32::from(lanes.contains(&40)) * 2
                + u32::from(lanes.iter().all(|&u| u < 40)) * 4
        }
        "simd_ballot_k" => {
            let active = |u: u32| lane(u) >= 2;
            if !active(t) {
                return 1;
            }
            over_simd(t, &active)
                .iter()
                .filter(|&&u| u % 3 == 0)
                .fold(0u32, |a, &u| a | (1 << lane(u)))
        }
        "simd_first_k" => {
            if l < 5 {
                return 3;
            }
            u32::from(l == 5) + (b + 5) * 10
        }
        "quad_ops_k" => (t ^ 1) + (qb + 2) * 1000 + (4 * qb + 6) * 1_000_000,
        "quad_relative_k" => {
            let up = if q >= 1 { t - 1 } else { t };
            let down = if q + 2 < 4 { t + 2 } else { t };
            let rot = qb + ((q + 3) & 3);
            up + down * 1000 + rot * 1_000_000
        }
        "quad_dynamic_k" => (t ^ ((t >> 2) & 3)) + (qb + 3) * 1000,
        // Metal's backend speculates the branch's arm (`air::canon::speculate`), so every lane of
        // the quad votes, one that skips the arm too: the M3 Max (macOS 27.0.1) answers 95 on
        // thread 0 (its mask 0xF) and 159 on thread 8 (lane 9 in the mask and the `any`).
        "quad_ballot_k" => {
            if t % 7 == 2 {
                return 99;
            }
            let quad: Vec<u32> = (qb..qb + 4).collect();
            let mask = quad.iter().fold(0u32, |a, &u| a | (1 << (u & 3)));
            let first = quad.first().copied() == Some(t);
            u32::from(first)
                + mask * 2
                + u32::from(quad.iter().all(|&u| u != 9)) * 64
                + u32::from(quad.contains(&9)) * 128
        }
        other => panic!("no expectation for {other}"),
    }
}

const KERNELS: [&str; 22] = [
    "simd_shuffle_k",
    "simd_broadcast_k",
    "simd_shuffle_up_k",
    "simd_shuffle_down_k",
    "simd_shuffle_xor_k",
    "simd_rotate_k",
    "simd_fill_down_k",
    "simd_fill_up_k",
    "simd_sum_k",
    "simd_half_sum_k",
    "simd_minmax_k",
    "simd_bits_k",
    "simd_xor_k",
    "simd_prefix_k",
    "simd_float_prefix_k",
    "simd_vote_k",
    "simd_ballot_k",
    "simd_first_k",
    "quad_ops_k",
    "quad_relative_k",
    "quad_dynamic_k",
    "quad_ballot_k",
];

/// Run every kernel of `simd_ops` under `p` (planned in `tier`) at `lanes` for each local size.
fn run_kernels(p: &DeviceProfile, tier: Tier, lanes: SubgroupSize, locals: &[[u32; 3]]) {
    let Some(bc) = assemble("simd_ops") else {
        return;
    };
    for kernel in KERNELS {
        let (plan, m) = typed_module(&bc, kernel, p).unwrap_or_else(|e| panic!("{kernel}: {e}"));
        assert_eq!(plan.simd.1, tier, "{kernel}");
        assert_eq!(
            plan.dispatch.as_ref().map(|d| d.local_size),
            Some(LocalSizeForm::Flat),
            "{kernel}: flattened"
        );
        for &local in locals {
            let out = run_all(&plan, &m, local, lanes);
            for (t, got) in out.iter().enumerate() {
                let want = expected(kernel, t as u32);
                assert_eq!(*got, want, "{kernel} {local:?} thread {t}");
            }
        }
    }
}

/// The `Direct` tier at 32 lanes, for one, two, and three simdgroups, a 2D threadgroup, and
/// divergent lanes.
#[test]
fn simd_kernels_run_direct_at_32_lanes() {
    run_kernels(
        &moltenvk(),
        Tier::Direct,
        SubgroupSize::W32,
        &[[32, 1, 1], [64, 1, 1], [8, 8, 1], [96, 1, 1]],
    );
}

/// Without `shaderInt64` every kernel (the 64-bit ballot as two words) computes the same.
#[test]
fn simd_kernels_run_without_int64() {
    run_kernels(
        &moltenvk_without_int64(),
        Tier::Direct,
        SubgroupSize::W32,
        &[[32, 1, 1], [64, 1, 1]],
    );
}

/// The committed `moltenvk-m3` description without Vulkan's quad class in any stage.
fn moltenvk_without_quads() -> DeviceProfile {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/moltenvk-m3.json");
    let text = std::fs::read_to_string(&path).expect("read the MoltenVK description");
    let cleared = text.replace("\"VK_SUBGROUP_FEATURE_QUAD_BIT\",", "");
    assert_ne!(cleared, text, "the description lists the quad class");
    let desc = DeviceDescription::from_khronos_json(&cleared, None).expect("description");
    let p = Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES)
        .expect("device")
        .profile()
        .clone();
    for stage in [Stage::Vertex, Stage::Fragment, Stage::Kernel] {
        assert!(!p.fields().subgroup.quad_instructions(stage), "{stage:?}");
    }
    p
}

/// Where no stage takes Vulkan's quad instructions, a quad broadcast and shuffle-xor by a constant
/// lane or mask (`quad_ops_k`) are shuffles, and every kernel computes the same in the `Direct`
/// tier. With the quad class, `quad_ops_k` takes the quad instructions.
#[test]
fn simd_kernels_run_without_quad_instructions() {
    run_kernels(
        &moltenvk_without_quads(),
        Tier::Direct,
        SubgroupSize::W32,
        &[[32, 1, 1], [64, 1, 1], [8, 8, 1]],
    );
    let Some(bc) = assemble("simd_ops") else {
        return;
    };
    for (p, quads) in [(moltenvk_without_quads(), false), (moltenvk(), true)] {
        let a = compile(&bc, "quad_ops_k", &p).expect("compile");
        let (_, words) = link_and_validate(&a, &p).expect("valid");
        let text = spirv_dis(&words);
        assert_eq!(text.contains("OpGroupNonUniformQuadBroadcast"), quads);
        assert_eq!(text.contains("OpGroupNonUniformQuadSwap"), quads);
        assert_eq!(text.contains("OpGroupNonUniformShuffleXor"), !quads);
    }
}

/// The `Cluster32` tier at 64 lanes: each subgroup holds two Metal simdgroups, so every operation
/// must stay in its own 32-lane cluster.
#[test]
fn simd_kernels_run_cluster32_at_64_lanes() {
    run_kernels(
        &wave64(),
        Tier::Cluster32,
        SubgroupSize::W64,
        &[[64, 1, 1], [128, 1, 1], [16, 8, 1]],
    );
}

/// The required case: `quad_shuffle` in a compute kernel matches the expectation derived from
/// `thread_index_in_threadgroup` (quads are four consecutive linear threads) in both tiers.
#[test]
fn quad_shuffle_matches_the_linear_thread_index() {
    let Some(bc) = assemble("quad_shuffle") else {
        return;
    };
    for (p, tier, lanes, local) in [
        (moltenvk(), Tier::Direct, SubgroupSize::W32, [32, 1, 1]),
        (moltenvk(), Tier::Direct, SubgroupSize::W32, [4, 8, 2]),
        (wave64(), Tier::Cluster32, SubgroupSize::W64, [8, 16, 1]),
    ] {
        let (plan, m) = typed_module(&bc, "quad_shuffle_rotate", &p).expect("compile");
        assert_eq!(plan.simd.1, tier);
        let out = run_all(&plan, &m, local, lanes);
        for (t, got) in out.iter().enumerate() {
            let t = t as u32;
            assert_eq!(*got, (t & !3) | ((t + 1) & 3), "{local:?} thread {t}");
        }
    }
}

/// On every committed profile each kernel compiles, links, relocates, and passes L3 in its
/// tier. A flattened kernel declares its subgroup requirement and local-size constraint. Where no
/// subgroup tier runs a 32-lane simdgroup the kernel's simdgroups are emulated: it needs no
/// subgroup and keeps its local size, including a kernel whose operation is outside
/// workgroup-uniform control flow (run predicated).
#[test]
fn simd_kernels_compile_or_refuse_with_the_tier() {
    let Some(bc) = assemble("simd_ops") else {
        return;
    };
    let (mut flat, mut emulated) = (0, 0);
    for (stem, p) in committed_profiles().expect("profiles") {
        let shuffle_tier = subgroup::tier(
            &SimdClass::CrossLane(OpBits::SHUFFLE),
            Stage::Kernel,
            &p.fields().subgroup,
        );
        for kernel in KERNELS {
            match compile(&bc, kernel, &p) {
                Ok(a) => {
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem} {kernel}: {e}"));
                    if a.interface.declares_spec(SpecKey::FlatLocalSize) {
                        let multiple = a.subgroup.local_size_multiple;
                        assert!(
                            multiple >= 32 && a.subgroup.full_subgroups,
                            "{stem} {kernel}"
                        );
                        assert!(a
                            .facts
                            .contains(&RuntimeFact::LocalSizeConstraint { multiple }));
                        flat += 1;
                    } else {
                        assert_eq!(shuffle_tier, Ok(Tier::Emulated), "{stem} {kernel}");
                        assert_eq!(a.subgroup, SubgroupRequirement::NONE, "{stem} {kernel}");
                        emulated += 1;
                    }
                }
                Err(Error::Unsupported(u)) => match u.feature {
                    // A value the profile cannot hold (the ballot's 64-bit result) is refused
                    // while selecting, before the tier.
                    Feature::Int64Arithmetic => {}
                    _ => panic!("{stem} {kernel}: {u:?}"),
                },
                Err(e) => panic!("{stem} {kernel}: {e}"),
            }
        }
    }
    assert!(
        flat > 0 && emulated > 0,
        "{flat} flattened, {emulated} emulated"
    );
}

/// A kernel whose tier pins its subgroup size states what the pin needs of the device
/// (`subgroupSizeControl`, the kernel stage in `requiredSubgroupSizeStages`, the size inside
/// `[minSubgroupSize, maxSubgroupSize]`), and one that needs full subgroups states their stage
/// flag; a kernel that pins nothing states none of the pin's facts. On every committed profile
/// the facts are within the target, and its required enables name the pin's features.
#[test]
fn a_pinned_subgroup_size_states_its_device_facts() {
    let Some(bc) = assemble("simd_ops") else {
        return;
    };
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles");
    let (mut pinned, mut unpinned) = (0, 0);
    for (stem, p) in committed_profiles().expect("profiles") {
        let desc = DeviceDescription::from_khronos_json(
            &std::fs::read_to_string(path.join(format!("{stem}.json"))).expect("read"),
            None,
        )
        .expect("description");
        for kernel in KERNELS {
            let Ok(a) = compile(&bc, kernel, &p) else {
                continue;
            };
            let r = &a.requirements;
            let control = r.features.contains(&VkFeature::SubgroupSizeControl);
            let stage = r
                .properties
                .contains(&VkProperty::RequiredSubgroupSizeStage(Stage::Kernel));
            let range = (
                r.limits.get(&VkLimit::MinSubgroupSize).copied(),
                r.limits.get(&VkLimit::MaxSubgroupSize).copied(),
            );
            if let Some(size) = a.subgroup.size {
                let size = Some(u64::from(size));
                assert!(control && stage, "{stem} {kernel}: {r:?}");
                assert_eq!(range, (size, size), "{stem} {kernel}");
                let enables = desc.required_enables(r).expect("enables");
                assert!(
                    enables.features.contains(&VkFeature::SubgroupSizeControl),
                    "{stem} {kernel}"
                );
                pinned += 1;
            } else {
                assert!(!stage && range == (None, None), "{stem} {kernel}: {r:?}");
                unpinned += 1;
            }
            assert_eq!(
                r.stage_flags
                    .contains(&StageCreateFlag::RequireFullSubgroups),
                a.subgroup.full_subgroups,
                "{stem} {kernel}"
            );
            assert_eq!(p.check(r), Ok(()), "{stem} {kernel}");
        }
    }
    assert!(
        pinned > 0 && unpinned > 0,
        "{pinned} pinned, {unpinned} unpinned"
    );
}

/// NVK reports `subgroupSizeControl` with an empty `requiredSubgroupSizeStages` (its subgroups are
/// always 32 wide). A kernel compiled for NVIDIA's driver, which can pin every stage, pins 32;
/// checked against the same description with the stages emptied, its requirements name the missing
/// stage before any pipeline is created. Compiled for that description itself, the kernel runs at
/// the API size, unpinned: the target decides the pin.
#[test]
fn a_device_that_cannot_pin_the_stage_refuses_a_pinned_kernel() {
    let Some(bc) = assemble("simd_ops") else {
        return;
    };
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/nvidia-rtx50-linux.json");
    let text = std::fs::read_to_string(&path).expect("read the RTX 50 description");
    let mut json: serde_json::Value = serde_json::from_str(&text).expect("json");
    fn empty_stages(v: &mut serde_json::Value, emptied: &mut u32) {
        match v {
            serde_json::Value::Object(map) => {
                for (k, v) in map.iter_mut() {
                    if k == "requiredSubgroupSizeStages" {
                        *v = serde_json::Value::Array(Vec::new());
                        *emptied += 1;
                    } else {
                        empty_stages(v, emptied);
                    }
                }
            }
            serde_json::Value::Array(items) => {
                for v in items {
                    empty_stages(v, emptied);
                }
            }
            _ => {}
        }
    }
    let mut emptied = 0;
    empty_stages(&mut json, &mut emptied);
    assert!(
        emptied > 0,
        "the description names requiredSubgroupSizeStages"
    );
    let nvk = |text: &str| {
        let desc = DeviceDescription::from_khronos_json(text, None).expect("description");
        Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES)
            .expect("device")
            .profile()
            .clone()
    };
    let nvidia = nvk(&text);
    let nvk = nvk(&json.to_string());
    let a = compile(&bc, "simd_shuffle_k", &nvidia).expect("compile for NVIDIA");
    assert_eq!(a.subgroup.size, Some(32));
    assert_eq!(nvidia.check(&a.requirements), Ok(()));
    assert_eq!(
        nvk.check(&a.requirements),
        Err(crate::error::Capability::Property(
            VkProperty::RequiredSubgroupSizeStage(Stage::Kernel)
        ))
    );
    let a = compile(&bc, "simd_shuffle_k", &nvk).expect("compile for NVK");
    assert_eq!(a.subgroup.size, None);
    assert!(a.subgroup.full_subgroups);
    assert_eq!(nvk.check(&a.requirements), Ok(()));
}

/// A kernel with a tangled instruction declares `MaximallyReconvergesKHR` where the profile has
/// maximal reconvergence (reconvergence guaranteed), and records `ImplementationDefined` where it
/// has neither mode, beside `PrecisionNote::DeviceReconvergence`.
#[test]
fn reconvergence_follows_the_profile() {
    let Some(bc) = assemble("simd_ops") else {
        return;
    };
    let maximal = moltenvk();
    assert!(maximal.fields().subgroup.maximal_reconvergence);
    let a = compile(&bc, "simd_vote_k", &maximal).expect("compile");
    let (_, words) = link_and_validate(&a, &maximal).expect("valid");
    assert!(spirv_dis(&words).contains("MaximallyReconvergesKHR"));
    assert_eq!(a.reconvergence, Reconvergence::Guaranteed);
    assert!(!a.precision.contains(&PrecisionNote::DeviceReconvergence));

    let pascal = profile("nvidia-pascal-linux", DevicePolicy::RESIDENT_TABLES).expect("profile");
    let sg = pascal.fields().subgroup;
    assert!(!sg.maximal_reconvergence && !sg.uniform_control_flow);
    let a = compile(&bc, "simd_vote_k", &pascal).expect("compile");
    let (_, words) = link_and_validate(&a, &pascal).expect("valid");
    let text = spirv_dis(&words);
    assert!(
        !text.contains("MaximallyReconvergesKHR")
            && !text.contains("SubgroupUniformControlFlowKHR")
    );
    assert_eq!(a.reconvergence, Reconvergence::ImplementationDefined);
    assert!(a.precision.contains(&PrecisionNote::DeviceReconvergence));
}

/// Float sums record the device's association order.
#[test]
fn float_sums_record_their_precision() {
    let Some(bc) = assemble("simd_ops") else {
        return;
    };
    let p = moltenvk();
    let a = compile(&bc, "simd_half_sum_k", &p).expect("compile");
    assert!(a.precision.contains(&PrecisionNote::ReductionOrder));
    let a = compile(&bc, "simd_xor_k", &p).expect("compile");
    assert!(!a.precision.contains(&PrecisionNote::ReductionOrder));
}

/// `gsm2_vertex_quad_permutes`' quad broadcasts and shuffle-xors by a constant lane or mask in
/// vertex functions are shuffles through the ballot of the live lanes on every committed profile,
/// since a vertex quad may lack a lane past the draw's vertex count: each function compiles,
/// declares no quad capability, and validates. A profile whose vertex stage has no subgroups
/// (MoltenVK's) refuses them.
#[test]
fn vertex_quad_permutes_shuffle_through_the_live_lanes() {
    let Some(bc) = assemble("gsm2_vertex_quad_permutes") else {
        return;
    };
    let mut shuffled = 0;
    for entry in [
        "quad_permutes_in_a_vertex_function",
        "quad_pair_permutes_in_a_vertex_function",
        "quad_permutes_in_branches_of_a_vertex_function",
    ] {
        for (name, p) in committed_profiles().expect("profiles") {
            let model = &p.fields().subgroup;
            let a = match compile(&bc, entry, &p) {
                Ok(a) => a,
                Err(e) => {
                    assert!(!model.vertex.supported, "{entry} on {name}: {e}");
                    continue;
                }
            };
            let (_, words) = link_and_validate(&a, &p).expect("valid");
            let text = spirv_dis(&words);
            // A vertex quad may lack a lane past the draw's vertex count, so every permute reads
            // its lane by a shuffle through the live lanes, never a quad instruction.
            assert!(
                !text.contains("OpGroupNonUniformQuadBroadcast")
                    && !text.contains("OpCapability GroupNonUniformQuad"),
                "{entry} on {name}"
            );
            assert!(
                text.contains("OpGroupNonUniformShuffle ")
                    && text.contains("OpGroupNonUniformBallotBitExtract"),
                "{entry} on {name}"
            );
            shuffled += 1;
        }
    }
    assert!(shuffled >= 3, "{shuffled} functions compiled");
}

/// A vertex function's quads hold no quad instruction, so they ask nothing of
/// `subgroupQuadOperationsInAllStages`, which Vulkan asks of quad instructions outside the
/// fragment and compute stages: on the Iris Xe's description with the property cleared, each of
/// `gsm2_vertex_quad_permutes`' vertex functions compiles to a valid module that holds no quad
/// instruction or capability.
#[test]
fn vertex_quads_need_no_quad_operations_in_all_stages() {
    let Some(bc) = assemble("gsm2_vertex_quad_permutes") else {
        return;
    };
    let mut desc =
        crate::pipeline::testing::description("intel-irisxe-rplp-anv").expect("description");
    assert!(desc.subgroup.quad_operations_in_all_stages);
    desc.subgroup.quad_operations_in_all_stages = false;
    let p = Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES)
        .expect("device")
        .profile()
        .clone();
    for entry in [
        "quad_permutes_in_a_vertex_function",
        "quad_pair_permutes_in_a_vertex_function",
        "quad_permutes_in_branches_of_a_vertex_function",
    ] {
        let a = compile(&bc, entry, &p).unwrap_or_else(|e| panic!("{entry}: {e}"));
        let (_, words) = link_and_validate(&a, &p).expect("valid");
        let text = spirv_dis(&words);
        assert!(!text.contains("GroupNonUniformQuad"), "{entry}");
    }
}

/// A fragment function of simdgroup operations (`fragment_discard_counts`: sums before and after a
/// discard) records `PrecisionNote::FragmentSimdgroupGrouping` on every committed profile that
/// compiles it. A fragment that reads only its quad lane (`simd_masks`' `quad_lane_f`)
/// records none anywhere, and neither does a vertex function of simdgroup operations
/// (`rvc_simd_vertex`).
#[test]
fn fragment_simdgroup_functions_record_the_grouping() {
    let note = PrecisionNote::FragmentSimdgroupGrouping;
    let (Some(counts), Some(masks), Some(vertex)) = (
        assemble("fragment_discard_counts"),
        assemble("simd_masks"),
        assemble("rvc_simd_vertex"),
    ) else {
        return;
    };
    let mut noted = 0;
    for (name, p) in committed_profiles().expect("profiles") {
        if let Ok(a) = compile(&counts, "discard_counts", &p) {
            assert!(a.precision.contains(&note), "discard_counts on {name}");
            noted += 1;
        }
        if let Ok(a) = compile(&masks, "quad_lane_f", &p) {
            assert!(!a.precision.contains(&note), "quad_lane_f on {name}");
        }
        if let Ok(a) = compile(&vertex, "simd_functions_in_a_vertex_function", &p) {
            assert!(!a.precision.contains(&note), "rvc_simd_vertex on {name}");
        }
    }
    assert!(noted > 0, "no profile recorded the note");
}

/// A vertex function of simdgroup operations (`rvc_simd_vertex`: a sum, shuffles and a ballot) or
/// of quad operations (`gsm2_vertex_quad_permutes`' broadcasts and shuffle-xors) records
/// `PrecisionNote::VertexSimdgroupGrouping` on every committed profile that compiles it; a
/// kernel's (`simd_ops`' sum) and a fragment function's (`fragment_discard_counts`' sums) record
/// none.
#[test]
fn vertex_simdgroup_functions_record_the_grouping() {
    let note = PrecisionNote::VertexSimdgroupGrouping;
    let (Some(simd), Some(quads), Some(kernels), Some(counts)) = (
        assemble("rvc_simd_vertex"),
        assemble("gsm2_vertex_quad_permutes"),
        assemble("simd_ops"),
        assemble("fragment_discard_counts"),
    ) else {
        return;
    };
    let (mut noted, mut unnoted) = (0, 0);
    for (name, p) in committed_profiles().expect("profiles") {
        for (bc, entry) in [
            (&simd, "simd_functions_in_a_vertex_function"),
            (&quads, "quad_permutes_in_a_vertex_function"),
        ] {
            if let Ok(a) = compile(bc, entry, &p) {
                assert!(a.precision.contains(&note), "{entry} on {name}");
                noted += 1;
            }
        }
        for (bc, entry) in [(&kernels, "simd_sum_k"), (&counts, "discard_counts")] {
            if let Ok(a) = compile(bc, entry, &p) {
                assert!(!a.precision.contains(&note), "{entry} on {name}");
                unnoted += 1;
            }
        }
    }
    assert!(noted > 0, "no profile compiled a vertex function");
    assert!(
        unnoted > 0,
        "no profile compiled a kernel or fragment function"
    );
}

/// An operation over the group's active lanes records `PrecisionNote::QuadsSplitByClipping` in a
/// fragment function alone (`fragment_discard_counts`' sums), where clipping may split a quad, on
/// every committed profile that compiles it; a kernel's (`simd_ops`' sum, vote, ballot and first
/// lane) and a vertex function's (`rvc_simd_vertex`) record none, since no clipping splits their
/// groups.
#[test]
fn only_a_fragment_function_records_quads_split_by_clipping() {
    let note = PrecisionNote::QuadsSplitByClipping;
    let (Some(counts), Some(kernels), Some(vertex)) = (
        assemble("fragment_discard_counts"),
        assemble("simd_ops"),
        assemble("rvc_simd_vertex"),
    ) else {
        return;
    };
    let (mut noted, mut unnoted) = (0, 0);
    for (name, p) in committed_profiles().expect("profiles") {
        if let Ok(a) = compile(&counts, "discard_counts", &p) {
            assert!(a.precision.contains(&note), "discard_counts on {name}");
            noted += 1;
        }
        for kernel in ["simd_sum_k", "simd_vote_k", "simd_ballot_k", "simd_first_k"] {
            if let Ok(a) = compile(&kernels, kernel, &p) {
                assert!(!a.precision.contains(&note), "{kernel} on {name}");
                unnoted += 1;
            }
        }
        if let Ok(a) = compile(&vertex, "simd_functions_in_a_vertex_function", &p) {
            assert!(!a.precision.contains(&note), "rvc_simd_vertex on {name}");
        }
    }
    assert!(noted > 0, "no profile recorded the note");
    assert!(unnoted > 0, "no profile compiled a kernel");
}

/// A kernel with compute derivatives and a simdgroup operation is refused as
/// `ComputeDerivativesWithSimd` wherever the operation flattens it (flattening destroys the 2D quad
/// shape the derivatives need). Emulated simdgroups keep the local size, so there the kernel
/// compiles and passes L3.
#[test]
fn compute_derivatives_with_a_simd_operation_are_refused() {
    let text = std::fs::read_to_string(engine_dir().join("store_const.ll")).expect("read");
    let calls = "  %d = tail call float @air.dfdx.f32(float 1.000000e+00)\n  \
                 %s = tail call i32 @air.simd_shuffle.u.i32(i32 1, i16 0)\n  ret void\n}";
    let text = text.replacen("  ret void\n}", calls, 1);
    let decls = "\ndeclare float @air.dfdx.f32(float)\n\
                 declare i32 @air.simd_shuffle.u.i32(i32, i16)\n\nattributes #0";
    let text = text.replacen("\nattributes #0", decls, 1);
    let scratch = Scratch::new("m2v-engine-simd");
    let dir = scratch.path();
    let path = dir.join("kernel.ll");
    std::fs::write(&path, text).expect("write");
    let bc = llvm_assemble(&path);
    let bc = match bc {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the kernel: {why}"),
    };
    let mut refused = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let tier = subgroup::tier(
            &SimdClass::CrossLane(OpBits::SHUFFLE),
            Stage::Kernel,
            &p.fields().subgroup,
        );
        if !p.fields().images.compute_derivative_linear || tier.is_err() {
            continue;
        }
        match (compile(&bc, "store_const", &p), tier) {
            (Ok(a), Ok(Tier::Emulated)) => {
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
            }
            (Err(Error::Unsupported(u)), _) => {
                assert_eq!(u.feature, Feature::ComputeDerivativesWithSimd, "{stem}");
                refused += 1;
            }
            (other, t) => panic!("{stem} {t:?}: {other:?}"),
        }
    }
    assert!(
        refused > 0,
        "no profile has both compute derivatives and a tier"
    );
}

/// Metal has no simdgroup or quad function of 64-bit integers but the ballot (its `metal_simdgroup`
/// and `metal_quadgroup` headers enable them for 8- to 32-bit integers, `half`, `float`, and
/// `double` only), so a call of one is invalid input on every committed profile, never
/// `Int64Arithmetic` or a subgroup capability.
#[test]
fn sixty_four_bit_group_operations_are_not_metal() {
    let text = std::fs::read_to_string(engine_dir().join("store_const.ll")).expect("read");
    let calls = [
        (
            "%s = tail call i64 @air.simd_shuffle.s.i64(i64 1, i16 0)",
            "declare i64 @air.simd_shuffle.s.i64(i64, i16)",
        ),
        (
            "%s = tail call i64 @air.simd_sum.u.i64(i64 1)",
            "declare i64 @air.simd_sum.u.i64(i64)",
        ),
        (
            "%s = tail call <2 x i64> @air.simd_broadcast.u.v2i64(<2 x i64> zeroinitializer, i16 1)",
            "declare <2 x i64> @air.simd_broadcast.u.v2i64(<2 x i64>, i16)",
        ),
        (
            "%s = tail call i64 @air.quad_shuffle_xor.s.i64(i64 1, i16 1)",
            "declare i64 @air.quad_shuffle_xor.s.i64(i64, i16)",
        ),
    ];
    for (call, decl) in calls {
        let body = format!("  {call}\n  ret void\n}}");
        let text = text.replacen("  ret void\n}", &body, 1);
        let text = text.replacen("\nattributes #0", &format!("\n{decl}\n\nattributes #0"), 1);
        let scratch = Scratch::new("m2v-engine-simd64");
        let dir = scratch.path();
        let path = dir.join("kernel.ll");
        std::fs::write(&path, text).expect("write");
        let bc = llvm_assemble(&path);
        let bc = match bc {
            Ok(b) => b,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the kernel: {why}"),
        };
        for (stem, p) in committed_profiles().expect("profiles") {
            match compile(&bc, "store_const", &p) {
                Err(Error::InvalidInput(e)) => assert_eq!(
                    e.kind,
                    InputErrorKind::Intrinsic(IntrinsicError::SignatureMismatch),
                    "{stem} {call}"
                ),
                other => panic!("{stem} {call}: {other:?}"),
            }
        }
    }
}

/// Where the profile has `shaderSubgroupUniformControlFlow` but not maximal reconvergence (the
/// committed `moltenvk-m3` description with `shaderMaximalReconvergence` cleared), a kernel with a
/// tangled instruction declares `SubgroupUniformControlFlowKHR`, requires that feature and its
/// extensions, passes L3, and records `ImplementationDefined` (the mode does not define
/// reconvergence after divergence), beside `PrecisionNote::DeviceReconvergence`.
#[test]
fn uniform_control_flow_mode_without_maximal_reconvergence() {
    let Some(bc) = assemble("simd_ops") else {
        return;
    };
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/moltenvk-m3.json");
    let text = std::fs::read_to_string(&path).expect("read the MoltenVK description");
    let cleared = text.replace(
        "\"shaderMaximalReconvergence\": true",
        "\"shaderMaximalReconvergence\": false",
    );
    assert_ne!(
        cleared, text,
        "the description names shaderMaximalReconvergence"
    );
    let desc = DeviceDescription::from_khronos_json(&cleared, None).expect("description");
    let p = Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES)
        .expect("device")
        .profile()
        .clone();
    let sg = p.fields().subgroup;
    assert!(sg.uniform_control_flow && !sg.maximal_reconvergence);
    let a = compile(&bc, "simd_vote_k", &p).expect("compile");
    let (_, words) = link_and_validate(&a, &p).expect("valid");
    let text = spirv_dis(&words);
    assert!(text.contains("SubgroupUniformControlFlowKHR"));
    assert!(!text.contains("MaximallyReconvergesKHR"));
    assert!(a
        .requirements
        .features
        .contains(&VkFeature::ShaderSubgroupUniformControlFlow));
    assert!(!a
        .requirements
        .features
        .contains(&VkFeature::ShaderMaximalReconvergence));
    assert!(a
        .requirements
        .extensions
        .contains(&VkExtension::KhrShaderSubgroupUniformControlFlow));
    assert_eq!(a.reconvergence, Reconvergence::ImplementationDefined);
    assert!(a.precision.contains(&PrecisionNote::DeviceReconvergence));
}

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// Kernel `uniform_k(device uint *out [[buffer(0)]], uint t [[thread_index_in_threadgroup]])`
/// with `body` (which ends by storing `7` to `out[t]`), assembled from a scratch file.
fn uniform_kernel(tag: &str, body: &str) -> Option<Vec<u8>> {
    let text = format!(
        "{HEADER}define void @uniform_k(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, i32 noundef %1) {{\nentry:\n{body}  %e = zext i32 %1 to i64\n  %p = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %e\n  store i32 7, ptr addrspace(1) %p, align 4\n  ret void\n}}\n\n\
         declare i1 @air.is_uniform.i32(i32)\ndeclare i1 @air.is_uniform.f16(half)\n\
         declare i1 @air.is_uniform.v4f32(<4 x float>)\ndeclare void @llvm.assume(i1 noundef)\n\n\
         !air.kernel = !{{!9}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\
         !9 = !{{ptr @uniform_k, !10, !11}}\n!10 = !{{}}\n!11 = !{{!12, !13}}\n\
         !12 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}}\n\
         !13 = !{{i32 1, !\"air.thread_index_in_threadgroup\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"t\"}}\n\
         !17 = !{{i32 2, i32 8, i32 0}}\n!18 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n"
    );
    let scratch = Scratch::new(&format!("m2v-engine-uniform-{tag}"));
    let dir = scratch.path();
    let path = dir.join("kernel.ll");
    std::fs::write(&path, text).expect("write");
    let bc = llvm_assemble(&path);
    match bc {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {tag}: {why}"),
    }
}

/// `air.is_uniform` is Metal's `uniform<T>` assertion. The frontend feeds its result to
/// `llvm.assume` (directly, or through a phi the assume consumes); decode drops the assume, so
/// the result has no use left but that phi, and the asserted `true` stands for it. Such a kernel
/// selects `simd.assumed_uniform` once per call, emits no subgroup instruction, leaves the simd
/// class `None` (so it compiles and passes L3 on every committed profile, those that cannot pin
/// 32 lanes included), and runs exactly. A float assertion (`uniform<half>`, `uniform<float4>`)
/// is the same assertion.
#[test]
fn uniform_assertions_feeding_an_assume_are_true_and_need_no_subgroup() {
    let direct =
        "  %u = tail call i1 @air.is_uniform.i32(i32 %1)\n  tail call void @llvm.assume(i1 %u)\n";
    let through_phi = "  %c = icmp ult i32 %1, 8\n  br i1 %c, label %a, label %b\na:\n  %ua = tail call i1 @air.is_uniform.i32(i32 %1)\n  br label %j\nb:\n  %ub = tail call i1 @air.is_uniform.i32(i32 0)\n  br label %j\nj:\n  %u = phi i1 [ %ua, %a ], [ %ub, %b ]\n  tail call void @llvm.assume(i1 %u)\n";
    let float = "  %h = uitofp i32 %1 to half\n  %uh = tail call i1 @air.is_uniform.f16(half %h)\n  tail call void @llvm.assume(i1 %uh)\n  %f = uitofp i32 %1 to float\n  %v = insertelement <4 x float> zeroinitializer, float %f, i32 0\n  %uv = tail call i1 @air.is_uniform.v4f32(<4 x float> %v)\n  tail call void @llvm.assume(i1 %uv)\n";
    for (tag, body, calls) in [
        ("direct", direct, 1u32),
        ("phi", through_phi, 2),
        ("float", float, 2),
    ] {
        let Some(bc) = uniform_kernel(tag, body) else {
            return;
        };
        for (stem, p) in committed_profiles().expect("profiles") {
            let a = compile(&bc, "uniform_k", &p).unwrap_or_else(|e| panic!("{tag} {stem}: {e}"));
            assert_eq!(
                a.strategies
                    .entries
                    .get("simd.assumed_uniform")
                    .map(|e| e.uses),
                Some(calls),
                "{tag} {stem}: {:?}",
                a.strategies.entries
            );
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{tag} {stem}: L3: {e}"));
            assert!(
                !spirv_dis(&words).contains("OpGroupNonUniform"),
                "{tag} {stem}"
            );
        }
        let p = moltenvk();
        let (plan, m) = typed_module(&bc, "uniform_k", &p).expect("plan");
        assert_eq!(plan.simd.0, SimdClass::None, "{tag}");
        for local in [[32, 1, 1], [16, 1, 1]] {
            let n = local[0] as usize;
            assert_eq!(
                run_all(&plan, &m, local, SubgroupSize::W32)[..n],
                vec![7; n][..],
                "{tag} {local:?}"
            );
        }
    }
}

/// The kernels of `engine/simd_masks`.
const MASK_KERNELS: [&str; 4] = [
    "mask_quad_ballot_k",
    "mask_quad_ballot_divergent_k",
    "mask_simd_active_k",
    "mask_simd_active_divergent_k",
];

/// The expected `out[t]` of each `simd_masks` kernel in a threadgroup of `n` threads (see the
/// fixture's MSL): a quad ballot's bit `j` is quad lane `j`, and a simdgroup's active mask (whose
/// upper half is folded in, zero) has bit `i` for each active lane `i` of the simdgroup.
fn mask_expected(kernel: &str, n: u32, t: u32) -> u32 {
    let quad = (t & !3)..((t & !3) + 4).min(n);
    let simd = (t & !31)..((t & !31) + 32).min(n);
    let ballot = |active: &dyn Fn(u32) -> bool| {
        quad.clone()
            .filter(|&u| active(u) && u % 3 == 0)
            .fold(0u32, |a, u| a | (1 << (u & 3)))
    };
    let mask = |active: &dyn Fn(u32) -> bool| {
        simd.clone()
            .filter(|&u| active(u))
            .fold(0u32, |a, u| a | (1 << (u & 31)))
    };
    match kernel {
        "mask_quad_ballot_k" => ballot(&|_| true),
        "mask_quad_ballot_divergent_k" if t % 7 == 2 => 99,
        // Metal's backend speculates the branch's arm (`air::canon::speculate`): a lane that skips
        // it votes too (the M3 Max, macOS 27.0.1, answers 2 on thread 8, the skipping lane 9's).
        "mask_quad_ballot_divergent_k" => ballot(&|_| true),
        "mask_simd_active_k" => mask(&|_| true),
        "mask_simd_active_divergent_k" if t & 31 < 3 => 1,
        "mask_simd_active_divergent_k" => mask(&|u| u & 31 >= 3),
        other => panic!("no expectation for {other}"),
    }
}

/// `quad_ballot` and `simd_active_threads_mask` (Apple's `air.quad_ballot` and
/// `air.simd_active_threads_mask.i64`) give Metal's masks in the `Direct` tier at 32 lanes
/// (with and without `shaderInt64`) and the `Cluster32` tier at 64, uniform and divergent, in
/// both scheduling orders.
#[test]
fn quad_ballots_and_simd_active_masks_run_in_every_subgroup_tier() {
    let Some(bc) = assemble("simd_masks") else {
        return;
    };
    let runs: [(DeviceProfile, Tier, SubgroupSize, &[[u32; 3]]); 3] = [
        (
            moltenvk(),
            Tier::Direct,
            SubgroupSize::W32,
            &[[32, 1, 1], [64, 1, 1], [8, 8, 1]],
        ),
        (
            moltenvk_without_int64(),
            Tier::Direct,
            SubgroupSize::W32,
            &[[64, 1, 1]],
        ),
        (
            wave64(),
            Tier::Cluster32,
            SubgroupSize::W64,
            &[[64, 1, 1], [128, 1, 1]],
        ),
    ];
    for (p, tier, lanes, locals) in runs {
        for kernel in MASK_KERNELS {
            let (plan, m) =
                typed_module(&bc, kernel, &p).unwrap_or_else(|e| panic!("{kernel}: {e}"));
            assert_eq!(plan.simd.1, tier, "{kernel}");
            let a = compile(&bc, kernel, &p).unwrap_or_else(|e| panic!("{kernel}: {e}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{kernel}: L3: {e}"));
            for &local in locals {
                let n: u32 = local.iter().product();
                let out = run_all(&plan, &m, local, lanes);
                for (t, got) in out.iter().enumerate() {
                    let want = mask_expected(kernel, n, t as u32);
                    assert_eq!(*got, want, "{kernel} {local:?} {lanes:?} thread {t}");
                }
            }
        }
    }
}

/// A fragment's `[[thread_index_in_quadgroup]]` is its subgroup lane's low two bits where the
/// profile has fragment subgroups that hold whole quads (Vulkan's quad scope: four adjacent lanes
/// from a multiple of 4, in a fragment shader the 2x2 pixels in Metal's order), and the pixel's
/// place in its aligned 2x2 block, read from `FragCoord`, where it has none. On every committed
/// profile the function compiles and passes L3: through `SubgroupLocalInvocationId` exactly where
/// the fragment stage's lane has a tier, and elsewhere through `FragCoord` with no subgroup
/// capability.
#[test]
fn a_fragment_quad_lane_is_its_subgroup_lane_or_its_place_in_the_pixel_quad() {
    let Some(bc) = assemble("simd_masks") else {
        return;
    };
    let (mut by_lane, mut by_position) = (0, 0);
    for (stem, p) in committed_profiles().expect("profiles") {
        let quad = subgroup::tier(
            &SimdClass::LaneArithmetic,
            Stage::Fragment,
            &p.fields().subgroup,
        );
        let a = compile(&bc, "quad_lane_f", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let (_, words) = link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let text = spirv_dis(&words);
        assert!(text.contains("OpBitwiseAnd"), "{stem}: the lane's low bits");
        if quad.is_ok() {
            assert!(
                text.contains("SubgroupLocalInvocationId"),
                "{stem}: the lane"
            );
            by_lane += 1;
        } else {
            assert!(text.contains("BuiltIn FragCoord"), "{stem}: the pixel");
            assert!(
                text.contains("OpConvertFToU"),
                "{stem}: the pixel's integer part"
            );
            assert!(text.contains("OpShiftLeftLogical"), "{stem}: the row's bit");
            assert!(
                !text.contains("SubgroupLocalInvocationId") && !text.contains("GroupNonUniform"),
                "{stem}: no subgroup"
            );
            by_position += 1;
        }
    }
    assert!(by_lane > 0 && by_position > 0, "{by_lane} {by_position}");
}

/// A fragment's `[[thread_index_in_quadgroup]]` asks for fragment subgroups and not the quad
/// class: Vulkan's quad scope is four adjacent subgroup lanes from a multiple of 4 wherever the
/// stage has subgroups (9.27.9), whatever operation classes it lists. On MoltenVK's description
/// without the quad class, `quad_lane_f` reads `SubgroupLocalInvocationId` and declares no quad
/// capability; with the fragment stage cleared from `subgroupSupportedStages`, it reads the pixel's
/// place in its 2x2 block from `FragCoord` and declares no subgroup capability.
#[test]
fn a_fragment_quad_lane_needs_fragment_subgroups_and_no_quad_class() {
    let Some(bc) = assemble("simd_masks") else {
        return;
    };
    let text = |p: &DeviceProfile| {
        let a = compile(&bc, "quad_lane_f", p).expect("compile");
        let (_, words) = link_and_validate(&a, p).expect("valid");
        spirv_dis(&words)
    };
    let lane = text(&moltenvk_without_quads());
    assert!(lane.contains("SubgroupLocalInvocationId"), "the lane");
    assert!(!lane.contains("GroupNonUniformQuad"), "no quad class");
    let mut desc = crate::pipeline::testing::description("moltenvk-m3").expect("description");
    desc.subgroup.supported_stages = crate::vk::ShaderStageFlags::COMPUTE;
    let p = Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES)
        .expect("device")
        .profile()
        .clone();
    let parity = text(&p);
    assert!(parity.contains("OpShiftLeftLogical"), "the row's bit");
    assert!(
        !parity.contains("SubgroupLocalInvocationId") && !parity.contains("GroupNonUniform"),
        "no subgroup"
    );
}

/// The owned fixture `gsm2_fragment_lanes`. A fragment's `[[thread_index_in_simdgroup]]`, read as a
/// `uint` or `ushort` parameter or as a Metal 3.1 program-scope `uint`, and beside the shuffles and
/// ballot of `fragment_lane_shuffles`, is the lane of its tier's 32-lane simdgroup, over every
/// committed profile and the wave64-only device ([`wave64`], the shape of AMD GCN):
/// - on every profile whose fragment stage runs 32-lane subgroups (`Tier::Direct`), or 64- or
///   128-lane ones split into 32-lane clusters (`Tier::Cluster32`, the wave64 device: the lane is
///   the subgroup lane's low five bits) it records `PrecisionNote::FragmentSimdgroupGrouping`;
/// - where the stage's subgroups are narrower than 32 (`Tier::PartialFragment`) it is refused
///   `SimdWidth`, needing what the stage lacks for 32 lanes; where the stage has no
///   subgroup tier, that refusal (`SubgroupOperations`);
/// - `fragment_lane_shuffles` on a 32-lane stage without `shaderQuadControl` is refused needing
///   it: its shuffles and ballot count Metal's helpers, which only full quads give.
///
/// Each compiled function passes L3 and reads `SubgroupLocalInvocationId`. A
/// `[[threads_per_simdgroup]]` parameter is 32 on every profile, with no subgroup capability and no
/// note.
#[test]
fn a_fragment_simdgroup_lane_is_the_lane_of_a_32_lane_simdgroup_where_the_tier_gives_one() {
    use crate::device::subgroup::width_needs;
    use crate::error::{Capability, Needs};
    let Some(bc) = assemble("gsm2_fragment_lanes") else {
        return;
    };
    let note = PrecisionNote::FragmentSimdgroupGrouping;
    let lane = OpBits::BASIC;
    let shuffles = lane.union(OpBits::BALLOT).union(OpBits::SHUFFLE);
    let (mut noted, mut narrow, mut clustered) = (0, 0, 0);
    let profiles = committed_profiles().expect("profiles");
    for (stem, p) in profiles
        .into_iter()
        .chain([("wave64".to_owned(), wave64())])
    {
        let model = &p.fields().subgroup;
        for (entry, ops) in [
            ("fragment_lane", lane),
            ("fragment_lane_ushort", lane),
            ("fragment_global_lane", lane),
            ("fragment_lane_shuffles", shuffles),
        ] {
            let tier = subgroup::tier(&SimdClass::CrossLane(ops), Stage::Fragment, model);
            match (compile(&bc, entry, &p), &tier) {
                (Ok(a), Ok(t @ (Tier::Direct | Tier::Cluster32))) => {
                    assert!(a.precision.contains(&note), "{stem} {entry}: the note");
                    noted += 1;
                    let (_, words) = link_and_validate(&a, &p)
                        .unwrap_or_else(|e| panic!("{stem} {entry}: L3: {e}"));
                    let text = spirv_dis(&words);
                    assert!(
                        text.contains("BuiltIn SubgroupLocalInvocationId"),
                        "{stem} {entry}: the lane"
                    );
                    if *t == Tier::Cluster32 {
                        assert!(
                            text.contains("OpBitwiseAnd"),
                            "{stem} {entry}: the low bits"
                        );
                        clustered += 1;
                    }
                }
                // Shuffles and a ballot count the helpers Metal runs, which a profile without
                // `shaderQuadControl` cannot give.
                (Err(Error::Unsupported(u)), Ok(Tier::Direct | Tier::Cluster32))
                    if ops == shuffles && !model.quad_control =>
                {
                    assert_eq!(
                        (u.feature, u.needs),
                        (
                            Feature::SubgroupOperations,
                            Needs::AnyOf(vec![vec![Capability::Feature(
                                VkFeature::ShaderQuadControl
                            )]])
                        ),
                        "{stem} {entry}"
                    );
                }
                (Err(Error::Unsupported(u)), Ok(Tier::PartialFragment)) => {
                    let needs = Needs::AnyOf(vec![width_needs(
                        Stage::Fragment,
                        model.stage(Stage::Fragment),
                    )]);
                    assert_eq!(
                        (u.feature, u.needs),
                        (Feature::SimdWidth, needs),
                        "{stem} {entry}"
                    );
                    narrow += 1;
                }
                (Err(Error::Unsupported(u)), Err(t)) => {
                    assert_eq!(u.feature, t.feature, "{stem} {entry}: the tier's refusal");
                }
                (r, t) => panic!("{stem} {entry}: {:?} on tier {t:?}", r.map(|_| ())),
            }
        }
        let a = compile(&bc, "fragment_simd_width", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(!a.precision.contains(&note), "{stem}: the width's note");
        let (_, words) = link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let text = spirv_dis(&words);
        assert!(
            !text.contains("GroupNonUniform") && !text.contains("SubgroupLocalInvocationId"),
            "{stem}: the width reads no subgroup"
        );
    }
    assert!(
        noted > 0 && narrow > 0 && clustered > 0,
        "noted {noted}, narrow {narrow}, clustered {clustered}"
    );
}

/// Kernel `group_k(device float *out [[buffer(0)]], uint t [[thread_index_in_threadgroup]])`
/// storing `out[t] = <call>(float(t)<extra>)`, the call carrying `flags`, assembled from a
/// scratch file.
fn float_group_kernel(tag: &str, call: &str, flags: &str, extra: &str) -> Option<Vec<u8>> {
    let extra_ty = if extra.is_empty() { "" } else { ", i16" };
    let text = format!(
        "{HEADER}define void @group_k(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, i32 noundef %1) {{\nentry:\n  \
         %x = uitofp i32 %1 to float\n  %s = tail call {flags}float @air.{call}.f32(float %x{extra})\n  \
         %e = zext i32 %1 to i64\n  %p = getelementptr inbounds float, ptr addrspace(1) %0, i64 %e\n  \
         store float %s, ptr addrspace(1) %p, align 4\n  ret void\n}}\n\n\
         declare float @air.{call}.f32(float{extra_ty})\n\n\
         !air.kernel = !{{!9}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\
         !9 = !{{ptr @group_k, !10, !11}}\n!10 = !{{}}\n!11 = !{{!12, !13}}\n\
         !12 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"out\"}}\n\
         !13 = !{{i32 1, !\"air.thread_index_in_threadgroup\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"t\"}}\n\
         !17 = !{{i32 2, i32 8, i32 0}}\n!18 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n"
    );
    let scratch = Scratch::new(&format!("m2v-engine-float-group-{tag}"));
    let dir = scratch.path();
    let path = dir.join("kernel.ll");
    std::fs::write(&path, text).expect("write");
    let bc = llvm_assemble(&path);
    match bc {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {tag}: {why}"),
    }
}

/// Whether `words` declares `OpExecutionMode <entry> SignedZeroInfNanPreserve <width>`.
fn declares_signed_zero_inf_nan(words: &[u32], width: u32) -> bool {
    let mut i = 5;
    while let Some(w) = words.get(i) {
        let len = (w >> 16) as usize;
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        if w & 0xFFFF == crate::spv::Op::ExecutionMode as u32
            && len == 4
            && words[i + 2] == crate::spv::ExecutionMode::SignedZeroInfNanPreserve as u32
            && words[i + 3] == width
        {
            return true;
        }
        i += len;
    }
    false
}

/// A simdgroup or quad sum, prefix sum, minimum, or maximum of floats is float arithmetic on every
/// tier (the subgroup instructions, and the emulated tier's own adds, minima, and maxima, whose
/// `-0` identity keeps a sum's sign only where signed zeros are preserved). Without all of `nnan`,
/// `ninf`, and `nsz` on the call its entry declares `SignedZeroInfNanPreserve 32` where the profile
/// supports it and notes `DeviceSignedZeroInfNan` where it does not; a `fast` call needs neither,
/// and neither does a shuffle, which moves bits and computes nothing.
#[test]
fn float_group_arithmetic_without_fast_flags_keeps_signed_zeros() {
    let profiles = [
        ("moltenvk-m3", moltenvk(), Tier::Direct),
        ("wave64", wave64(), Tier::Cluster32),
        (
            "lavapipe",
            profile("lavapipe", DevicePolicy::RESIDENT_TABLES).expect("profile"),
            Tier::Emulated,
        ),
        (
            "runtime-floor",
            profile("runtime-floor", DevicePolicy::RESIDENT_TABLES).expect("profile"),
            Tier::Emulated,
        ),
    ];
    let arithmetic = [
        "simd_sum",
        "simd_prefix_inclusive_sum",
        "simd_prefix_exclusive_sum",
        "simd_min",
        "simd_max",
        "quad_sum",
        "quad_min",
        "quad_max",
    ];
    let mut kernels = Vec::new();
    for call in arithmetic {
        // `nnan ninf` and `ninf nsz` each lack one flag: they still keep signed zeros.
        for (flags, needs) in [
            ("", true),
            ("fast ", false),
            ("nnan ninf nsz ", false),
            ("nnan ninf ", true),
            ("ninf nsz ", true),
        ] {
            let tag = format!("{call}-{}", flags.trim().replace(' ', "-"));
            let Some(bc) = float_group_kernel(&tag, call, flags, "") else {
                return;
            };
            kernels.push((tag, bc, needs));
        }
    }
    for call in ["simd_shuffle", "quad_shuffle"] {
        let Some(bc) = float_group_kernel(call, call, "", ", i16 1") else {
            return;
        };
        kernels.push((call.to_string(), bc, false));
    }
    let (mut declared, mut noted) = (0, 0);
    for (stem, p, tier) in &profiles {
        let supported = p.fields().float_controls.f32.signed_zero_inf_nan_preserve;
        for (tag, bc, needs) in &kernels {
            let (a, plan, _) =
                compile_kept(bc, "group_k", p).unwrap_or_else(|e| panic!("{stem} {tag}: {e}"));
            assert_eq!(plan.simd.1, *tier, "{stem} {tag}");
            let (_, words) =
                link_and_validate(&a, p).unwrap_or_else(|e| panic!("{stem} {tag}: {e}"));
            let mode = declares_signed_zero_inf_nan(&words, 32);
            let note = a.precision.contains(&PrecisionNote::DeviceSignedZeroInfNan);
            assert_eq!(mode, *needs && supported, "{stem} {tag}: the mode");
            assert_eq!(note, *needs && !supported, "{stem} {tag}: the note");
            declared += usize::from(mode);
            noted += usize::from(note);
        }
    }
    assert!(
        declared > 0 && noted > 0,
        "{declared} declared, {noted} noted"
    );
}

/// A vertex function's relative shuffles and fill read a lane past the draw's vertex count through
/// the ballot of the live lanes, so no shuffle names an inactive invocation. Each vertex is a
/// subgroup of its own on the interpreter, which traps a shuffle from an inactive lane:
/// `gsm2_vertex_votes_shifts`' shifts keep their own value for the shuffle up by 3 at lane 0, and
/// the shuffle down by 5 and the fill up by 2 (from lanes 5 and 30, past the draw) run and write a
/// word, which is not promised (a thread reads only from an active thread, MSL 4.1 section
/// 6.10.2), on RADV's `Cluster32` vertex stage, NVIDIA's `Direct` one, and MoltenVK's unlisted one
/// (the lane at entry).
#[test]
fn vertex_lane_reads_past_the_draw_name_no_inactive_lane() {
    let Some(bc) = assemble("gsm2_vertex_votes_shifts") else {
        return;
    };
    let entry = "simd_shifts_in_a_vertex_function";
    let n = 5u32;
    let input: Vec<f32> = (0..n).map(|v| 1.0 + 0.5 * v as f32).collect();
    for (stem, tier, lanes) in [
        ("amd-780m-radv", Tier::Cluster32, SubgroupSize::W64),
        ("nvidia-rtx5080-laptop", Tier::Direct, SubgroupSize::W32),
    ] {
        let p = profile(stem, DevicePolicy::RESIDENT_TABLES).expect("profile");
        let (a, plan, m) = compile_kept(&bc, entry, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(plan.simd.1, tier, "{stem}");
        assert_eq!(plan.families.simd.partly_live.len(), 3, "{stem}");
        let buffers = [
            input
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<u8>>(),
            vec![0xAB; 12 * n as usize],
        ]
        .into_iter()
        .enumerate()
        .map(|(i, bytes)| Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Vertex,
                kind: ResourceKind::Buffer,
                index: i as u32,
            }),
            address: None,
            bytes,
        })
        .collect();
        let res = Resources {
            buffers,
            push: vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)],
            spec: BTreeMap::new(),
        };
        for order in [Order::Forward, Order::Reverse] {
            let cfg = Config {
                workgroups: [n, 1, 1],
                subgroup: lanes,
                order,
                ..Config::default()
            };
            let out = run(&m, &cfg, res.clone())
                .unwrap_or_else(|e| panic!("{stem} {order:?}: {e}"))
                .buffers;
            let got: Vec<u32> = out[1]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| u32::from_le_bytes(*w))
                .collect();
            for (v, x) in input.iter().enumerate() {
                assert_eq!(got[3 * v], x.to_bits(), "{stem} {order:?} vertex {v}");
                assert!(
                    got[3 * v + 1..3 * v + 3].iter().all(|&w| w != 0xABAB_ABAB),
                    "{stem} {order:?} vertex {v}: written"
                );
            }
        }
    }
}

/// A fragment function whose group operations read, combine, or count other lanes declares
/// `RequireFullQuadsKHR` on every committed profile with `shaderQuadControl`, needs that feature,
/// and reads every lane by a shuffle through the ballot of the live lanes (a quad may lack a
/// lane), except a reduction (`fragment_quad_scans`' product), whose one quad broadcast per lane
/// follows that lane's quad vote; a profile without it (`moltenvk-m3`) whose stage has the
/// classes refuses the function, needing the feature. A fragment that only reads its quad lane
/// (`simd_masks`' `quad_lane_f`) declares no such mode and needs no such feature anywhere.
#[test]
fn fragment_group_operations_run_full_quads() {
    use crate::error::{Capability, Unsupported};
    let (Some(quads), Some(masks)) = (assemble("gsm2_quad_ops"), assemble("simd_masks")) else {
        return;
    };
    let (mut full, mut refused) = (0, 0);
    for (name, p) in committed_profiles().expect("profiles") {
        let control = p.fields().subgroup.quad_control;
        for entry in [
            "fragment_quad_fills",
            "fragment_quad_fills_and_rotations",
            "fragment_simd_fills",
            "fragment_quad_scans",
        ] {
            match compile(&quads, entry, &p) {
                Ok(a) => {
                    assert!(control, "{entry} on {name}");
                    assert!(
                        a.requirements
                            .features
                            .contains(&VkFeature::ShaderQuadControl),
                        "{entry} on {name}"
                    );
                    let (_, words) = link_and_validate(&a, &p).expect("valid");
                    let text = spirv_dis(&words);
                    let count = |op: &str| text.lines().filter(|l| l.contains(op)).count();
                    assert!(
                        text.contains("RequireFullQuadsKHR")
                            && text.contains("OpCapability QuadControlKHR")
                            && count("OpGroupNonUniformQuadBroadcast")
                                == count("OpGroupNonUniformQuadAnyKHR")
                            && !text.contains("OpGroupNonUniformQuadSwap"),
                        "{entry} on {name}"
                    );
                    full += 1;
                }
                // Without the feature the function is refused: needing it, or, where the stage
                // lacks the classes its tier needs, as before.
                Err(Error::Unsupported(u)) if !control => {
                    let want = Unsupported::needs(
                        Feature::SubgroupOperations,
                        Capability::Feature(VkFeature::ShaderQuadControl),
                    );
                    refused += usize::from((u.feature, u.needs) == (want.feature, want.needs));
                }
                Err(e) => panic!("{entry} on {name}: {e}"),
            }
        }
        if let Ok(a) = compile(&masks, "quad_lane_f", &p) {
            assert!(
                !a.requirements
                    .features
                    .contains(&VkFeature::ShaderQuadControl),
                "quad_lane_f on {name}"
            );
            let (_, words) = link_and_validate(&a, &p).expect("valid");
            assert!(
                !spirv_dis(&words).contains("RequireFullQuadsKHR"),
                "quad_lane_f on {name}"
            );
        }
    }
    assert!(full >= 4 * 10, "{full} fragment functions with full quads");
    assert!(
        refused > 0,
        "no profile refused a fragment function for quad control"
    );
}

/// A fragment's quad reduction reads its quad through quad instructions alone, which Vulkan runs
/// over the quad's helpers in every entry (section 9.31): on every committed profile with
/// `shaderQuadControl`, `fragment_quad_bitwise`'s AND, OR and XOR each ask
/// `OpGroupNonUniformQuadAnyKHR` whether each of the four quad lanes takes part and read each by
/// `OpGroupNonUniformQuadBroadcast` (4 of each per reduction, 12 in all), and its quad prefix sum
/// alone reads by a ballot and shuffles. A kernel's quad reductions (`quad_bitwise`), whose quads
/// hold no helpers, keep the active lanes' word and take no quad vote.
#[test]
fn fragment_quad_reductions_read_their_quads_by_quad_instructions() {
    let Some(quads) = assemble("gsm2_quad_ops") else {
        return;
    };
    let count = |text: &str, op: &str| text.lines().filter(|l| l.contains(op)).count();
    let mut fragments = 0;
    for (name, p) in committed_profiles().expect("profiles") {
        if !p.fields().subgroup.quad_control {
            continue;
        }
        let a = compile(&quads, "fragment_quad_bitwise", &p).expect("compile");
        let (_, words) = link_and_validate(&a, &p).expect("valid");
        let text = spirv_dis(&words);
        assert_eq!(
            count(&text, "OpGroupNonUniformQuadAnyKHR"),
            12,
            "fragment_quad_bitwise on {name}"
        );
        assert_eq!(
            count(&text, "OpGroupNonUniformQuadBroadcast"),
            12,
            "fragment_quad_bitwise on {name}"
        );
        assert_eq!(
            count(&text, "OpGroupNonUniformBallot "),
            1,
            "fragment_quad_bitwise on {name}: the prefix sum's ballot alone"
        );
        fragments += 1;
        if let Ok(a) = compile(&quads, "quad_bitwise", &p) {
            let (_, words) = link_and_validate(&a, &p).expect("valid");
            let text = spirv_dis(&words);
            assert_eq!(
                count(&text, "OpGroupNonUniformQuadAnyKHR"),
                0,
                "quad_bitwise on {name}"
            );
        }
    }
    assert!(fragments > 0, "no committed profile has quad control");
}

/// A fragment that writes its quad's `quad_any` and `quad_all` of a predicate of its position.
const FRAGMENT_QUAD_VOTES: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define <4 x i32> @quad_votes(<4 x float> %pos) {
entry:
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %p = call <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float> %xy)
  %x = extractelement <2 x i32> %p, i64 0
  %y = extractelement <2 x i32> %p, i64 1
  %right = icmp ugt i32 %x, %y
  %left = icmp ule i32 %x, %y
  %any = call i1 @air.quad_any(i1 %right)
  %all = call i1 @air.quad_all(i1 %left)
  %a = zext i1 %any to i32
  %b = zext i1 %all to i32
  %o0 = insertelement <4 x i32> <i32 0, i32 0, i32 0, i32 1>, i32 %a, i64 0
  %o = insertelement <4 x i32> %o0, i32 %b, i64 1
  ret <4 x i32> %o
}

declare <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float>)
declare i1 @air.quad_any(i1)
declare i1 @air.quad_all(i1)

!air.fragment = !{!0}
!0 = !{ptr @quad_votes, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"uint4"}
!3 = !{!4}
!4 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
"#;

/// A fragment's quad votes are Vulkan's quad votes, which it runs over the quad's helpers in every
/// entry (section 9.31): on every committed profile with `shaderQuadControl`, [`FRAGMENT_QUAD_VOTES`]
/// holds one `OpGroupNonUniformQuadAllKHR`, one `OpGroupNonUniformQuadAnyKHR` and no ballot. A
/// kernel's quad votes (`a3_quad_votes_divergent`), whose quads hold no helpers, take no quad vote
/// instruction.
#[test]
fn fragment_quad_votes_are_quad_vote_instructions() {
    let fragment = match metal2vulkan_devtools::llvm_assemble_text(FRAGMENT_QUAD_VOTES) {
        Ok(bitcode) => bitcode,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected quad_votes: {why}"),
    };
    let Some(kernel) = assemble("a3_quad_votes_divergent") else {
        return;
    };
    let count = |text: &str, op: &str| text.lines().filter(|l| l.contains(op)).count();
    let mut fragments = 0;
    for (name, p) in committed_profiles().expect("profiles") {
        if !p.fields().subgroup.quad_control {
            continue;
        }
        let a = compile(&fragment, "quad_votes", &p).expect("compile");
        let (_, words) = link_and_validate(&a, &p).expect("valid");
        let text = spirv_dis(&words);
        assert_eq!(
            [
                count(&text, "OpGroupNonUniformQuadAllKHR"),
                count(&text, "OpGroupNonUniformQuadAnyKHR"),
                count(&text, "OpGroupNonUniformBallot "),
            ],
            [1, 1, 0],
            "quad_votes on {name}"
        );
        fragments += 1;
        if let Ok(a) = compile(&kernel, "a3_quad_votes_divergent", &p) {
            let (_, words) = link_and_validate(&a, &p).expect("valid");
            let text = spirv_dis(&words);
            assert_eq!(
                [
                    count(&text, "OpGroupNonUniformQuadAllKHR"),
                    count(&text, "OpGroupNonUniformQuadAnyKHR"),
                ],
                [0, 0],
                "a3_quad_votes_divergent on {name}"
            );
        }
    }
    assert!(fragments > 0, "no committed profile has quad control");
}

/// MoltenVK 1.4.2 drops `VK_KHR_shader_quad_control` (`shaderQuadControl`) on Apple's GPUs. The
/// committed `moltenvk-m3` description, a capture of 1.4.2 that states neither the feature nor
/// its extension, refuses each fragment function whose group operations observe other lanes,
/// needing the feature.
#[test]
fn a_device_without_quad_control_refuses_full_quads() {
    use crate::error::{Capability, Unsupported};
    let Some(quads) = assemble("gsm2_quad_ops") else {
        return;
    };
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/moltenvk-m3.json");
    let text = std::fs::read_to_string(&path).expect("read the MoltenVK description");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    assert!(!desc.features.contains(&VkFeature::ShaderQuadControl));
    assert!(!desc.extensions.contains(&VkExtension::KhrShaderQuadControl));
    let without = Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES)
        .expect("device")
        .profile()
        .clone();
    assert!(!without.fields().subgroup.quad_control);
    let want = Unsupported::needs(
        Feature::SubgroupOperations,
        Capability::Feature(VkFeature::ShaderQuadControl),
    );
    for entry in [
        "fragment_quad_fills",
        "fragment_quad_fills_and_rotations",
        "fragment_simd_fills",
        "fragment_quad_scans",
    ] {
        match compile(&quads, entry, &without) {
            Err(Error::Unsupported(u)) => assert_eq!(
                (&u.feature, &u.needs),
                (&want.feature, &want.needs),
                "{entry}"
            ),
            other => panic!("{entry} without the feature: {:?}", other.map(|_| ())),
        }
    }
}
