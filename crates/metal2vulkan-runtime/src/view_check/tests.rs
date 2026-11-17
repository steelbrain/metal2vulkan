//! The view check (`view_check.rs`), the parts that need no device: its facts on fixed medians, on
//! both sides of each bound, and the record that keeps its answer.
//!
//! - **The facts.** The per-lane fact only where the storage buffer's per-lane reads took less
//!   than `PER_LANE_COSTLY` of the uniform buffer's at both windows; the texel fact only where
//!   the texel buffer's uniform reads over imported pages took less than `TEXEL_FASTER` of the
//!   uniform buffer's at both; neither where its arms were not run. The medians of the GPUs the
//!   bounds come from give their facts: NVIDIA's per-lane ratios, RADV's none, the Iris Xe's
//!   texel ratio.
//! - **The record.** Its medians read back for its identity alone, whole and unchanged, and never
//!   a record whose device arms were not run.

use super::{
    arms, decode, encode, facts_of, record_name, Arm, Medians, ViewVerdict, ARMS, KERNELS,
    PER_LANE_COSTLY, TEXEL_FASTER,
};
use crate::in_place::DriverIdentity;
use metal2vulkan::bind::{CheckReads, CheckView};
use metal2vulkan::device::DriverWorkaround;
use std::collections::BTreeSet;

const LANES: DriverWorkaround = DriverWorkaround::UniformBufferPerLaneReadsSerialize;
const TEXELS: DriverWorkaround = DriverWorkaround::ExternalUniformReadsSkipACache;

/// Medians of 1000 ns everywhere but the storage per-lane arms (`per_lane` times the uniform
/// buffer's) and the imported texel uniform arms (`texel` times the uniform buffer's there),
/// the imported arms all 0 where `host` is false.
fn medians(per_lane: [f64; 2], texel: [f64; 2], host: bool) -> Medians {
    let mut out = [1000u64; ARMS];
    for (at, arm) in arms().iter().enumerate() {
        let window = usize::from(arm.vecs != 64);
        let pair = KERNELS[arm.kernel];
        out[at] = match (arm.host, pair) {
            (true, _) if !host => 0,
            (false, (CheckView::Storage, CheckReads::PerLane)) => {
                (1000.0 * per_lane[window]) as u64
            }
            (true, (CheckView::Texel, CheckReads::Uniform)) => (1000.0 * texel[window]) as u64,
            _ => 1000,
        };
    }
    Medians(out)
}

#[test]
fn the_facts_hold_only_below_their_bounds_at_both_windows() {
    let below = PER_LANE_COSTLY - 0.01;
    let above = PER_LANE_COSTLY + 0.01;
    let texel_below = TEXEL_FASTER - 0.01;
    let texel_above = TEXEL_FASTER + 0.01;
    let facts = |m: Medians| m.costs().facts;
    assert_eq!(
        facts(medians([below; 2], [1.0; 2], true)),
        BTreeSet::from([LANES])
    );
    assert_eq!(
        facts(medians([below, above], [1.0; 2], true)),
        BTreeSet::new(),
        "one window past the bound"
    );
    assert_eq!(
        facts(medians([above; 2], [texel_below; 2], true)),
        BTreeSet::from([TEXELS])
    );
    assert_eq!(
        facts(medians([1.0; 2], [texel_below, texel_above], true)),
        BTreeSet::new()
    );
    assert_eq!(
        facts(medians([below; 2], [texel_below; 2], true)),
        BTreeSet::from([LANES, TEXELS]),
        "both hold together"
    );
    assert_eq!(
        facts(medians([1.0; 2], [texel_below; 2], false)),
        BTreeSet::new(),
        "no imported pages, no texel fact"
    );
    let costs = medians([0.5, 0.25], [1.0; 2], false).costs();
    assert_eq!(costs.verdict, ViewVerdict::Measured);
    assert_eq!(costs.per_lane, Some(0.5), "the larger window's ratio");
    assert_eq!(costs.texel, None);
}

/// The ratios the bounds come from give their facts: NVIDIA's storage per-lane reads at 0.011 to
/// 0.107 of the uniform buffer's (the per-lane fact), RADV's at 0.96 to 1.00 (none), and the Iris
/// Xe's texel uniform reads over imported memory at 0.40 to 0.49 of the uniform buffer's (the
/// texel fact), a driver reading both alike over imported memory none.
#[test]
fn the_measured_gpus_get_their_facts() {
    assert_eq!(facts_of(Some(0.107), Some(1.0)), BTreeSet::from([LANES]));
    assert_eq!(facts_of(Some(0.011), None), BTreeSet::from([LANES]));
    assert_eq!(facts_of(Some(0.96), Some(0.98)), BTreeSet::new());
    assert_eq!(facts_of(Some(1.0), Some(0.49)), BTreeSet::from([TEXELS]));
    assert_eq!(facts_of(None, None), BTreeSet::new());
}

/// The arms: each window's eight device arms, then each window's two imported ones.
#[test]
fn the_arms_are_each_kernel_at_each_window_then_the_imported_ones() {
    let arms = arms();
    assert_eq!(arms.len(), ARMS);
    assert_eq!(arms.iter().filter(|arm| arm.host).count(), 4);
    assert!(arms[..16].iter().all(|arm| !arm.host));
    let distinct: BTreeSet<(usize, bool, u32)> = (arms.iter())
        .map(|&Arm { kernel, host, vecs }| (kernel, host, vecs))
        .collect();
    assert_eq!(distinct.len(), ARMS);
}

fn identity() -> DriverIdentity {
    DriverIdentity {
        vendor_id: 0x10de,
        device_id: 0x1e87,
        driver_version: 0x9b11_8000,
        driver_id: 4,
        driver_name: "NVIDIA".to_owned(),
        driver_info: "620.70.06".to_owned(),
        device_uuid: [7; 16],
    }
}

#[test]
fn a_record_reads_back_for_its_identity_alone() {
    let identity = identity();
    let kept = medians([0.05, 0.1], [0.9, 1.1], true);
    let bytes = encode(&identity, &kept);
    assert_eq!(decode(&identity, &bytes), Some(kept));
    let unimported = medians([0.05, 0.1], [0.9, 1.1], false);
    let bytes_unimported = encode(&identity, &unimported);
    assert_eq!(decode(&identity, &bytes_unimported), Some(unimported));
    let mut other = identity.clone();
    other.driver_version += 1;
    assert_eq!(decode(&other, &bytes), None, "another release");
    assert_ne!(record_name(&identity), record_name(&other));
    let mut flipped = bytes.clone();
    flipped[60] ^= 1;
    assert_eq!(decode(&identity, &flipped), None, "the checksum");
    assert_eq!(
        decode(&identity, &bytes[..bytes.len() - 1]),
        None,
        "cut short"
    );
    let mut unrun = kept;
    unrun.0[0] = 0;
    assert_eq!(decode(&identity, &encode(&identity, &unrun)), None);
}

/// An answer with facts makes the compile device of the description with them measured, whose
/// fingerprint differs from the description's own; an answer without makes none, and the device
/// keeps compiling with its own.
#[test]
fn an_answer_with_facts_makes_the_measured_compile_device() {
    use super::{ViewCheck, ViewCosts};
    use metal2vulkan::device::{Device as CompileDevice, DeviceDescription, DevicePolicy};
    let path = metal2vulkan_devtools::validation_dir().join("profiles/lavapipe.json");
    let text = std::fs::read_to_string(path).unwrap();
    let description = DeviceDescription::from_khronos_json(&text, None).unwrap();
    let policy = DevicePolicy::default();
    let base = CompileDevice::new(description.clone(), policy).unwrap();
    let check = ViewCheck::new(description.clone(), policy);
    check.settle(
        medians([1.0; 2], [1.0; 2], true).costs(),
        None,
        "none".to_owned(),
    );
    assert!(
        check.measured().is_none(),
        "no fact, no other compile device"
    );
    let check = ViewCheck::new(description, policy);
    let costs = ViewCosts {
        facts: BTreeSet::from([LANES, TEXELS]),
        ..medians([0.05; 2], [0.4; 2], true).costs()
    };
    check.settle(costs, None, "both".to_owned());
    let measured = check.measured().expect("the measured compile device");
    let workarounds = measured.profile().workarounds();
    assert!(workarounds.contains(&LANES) && workarounds.contains(&TEXELS));
    assert!(!base.profile().workarounds().contains(&LANES));
    assert_ne!(measured.profile_fingerprint(), base.profile_fingerprint());
}
