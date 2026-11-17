//! The view check (`view_check.rs`), the parts that need no device: its facts on fixed medians, on
//! both sides of each bound, and the record that keeps its answer.
//!
//! - **The facts.** The per-lane fact only where the storage buffer's per-lane reads took less
//!   than `PER_LANE_COSTLY` of the uniform buffer's at both windows. The texel fact only where, of
//!   the reads it moves over imported pages (the uniform ones, and the per-lane ones where the
//!   per-lane fact does not hold) at both windows, the texel buffer took less than
//!   `TEXEL_NO_SLOWER` of the uniform buffer's time in every one and less than `TEXEL_FASTER` in
//!   at least one; neither where its arms were not run. The ratios of the GPUs the bounds come
//!   from give their facts: NVIDIA's per-lane ratios, RADV's none, the Iris Xe's texel ratios.
//! - **The record.** Its medians read back for its identity alone, whole and unchanged, and never
//!   a record whose device arms were not run.

use super::{
    arms, decode, encode, facts_of, record_name, Arm, Medians, ViewVerdict, ARMS, KERNELS,
    PER_LANE_COSTLY, TEXEL_FASTER, TEXEL_NO_SLOWER,
};
use crate::in_place::DriverIdentity;
use metal2vulkan::bind::{CheckReads, CheckView};
use metal2vulkan::device::DriverWorkaround;
use std::collections::BTreeSet;

const LANES: DriverWorkaround = DriverWorkaround::UniformBufferPerLaneReadsSerialize;
const TEXELS: DriverWorkaround = DriverWorkaround::ExternalUniformReadsSkipACache;

/// Medians of 1000 ns everywhere but the storage per-lane arms (`per_lane` times the uniform
/// buffer's) and the imported texel arms (`uniform` and `lanes` times the uniform buffer's there,
/// for the uniform and the per-lane reads), the imported arms all 0 where `host` is false; each
/// pair of ratios is the 1 KiB window's, then the 4 KiB one's.
fn medians(per_lane: [f64; 2], uniform: [f64; 2], lanes: [f64; 2], host: bool) -> Medians {
    let mut out = [1000u64; ARMS];
    for (at, arm) in arms().iter().enumerate() {
        let window = usize::from(arm.vecs != 64);
        let nanos = |ratio: [f64; 2]| (1000.0 * ratio[window]) as u64;
        out[at] = match (arm.host, KERNELS[arm.kernel]) {
            (true, _) if !host => 0,
            (false, (CheckView::Storage, CheckReads::PerLane)) => nanos(per_lane),
            (true, (CheckView::Texel, CheckReads::Uniform)) => nanos(uniform),
            (true, (CheckView::Texel, CheckReads::PerLane)) => nanos(lanes),
            _ => 1000,
        };
    }
    Medians(out)
}

#[test]
fn the_per_lane_fact_holds_only_below_its_bound_at_both_windows() {
    let below = PER_LANE_COSTLY - 0.01;
    let above = PER_LANE_COSTLY + 0.01;
    let facts = |m: Medians| m.costs().facts;
    assert_eq!(
        facts(medians([below; 2], [1.0; 2], [1.0; 2], true)),
        BTreeSet::from([LANES])
    );
    assert_eq!(
        facts(medians([below, above], [1.0; 2], [1.0; 2], true)),
        BTreeSet::new(),
        "one window past the bound"
    );
    let costs = medians([0.5, 0.25], [1.0; 2], [1.0; 2], false).costs();
    assert_eq!(costs.verdict, ViewVerdict::Measured);
    assert_eq!(costs.per_lane, Some(0.5), "the larger window's ratio");
    assert_eq!((costs.texel_fastest, costs.texel_slowest), (None, None));
}

#[test]
fn the_texel_fact_needs_one_read_clearly_faster_and_none_slower() {
    let fast = TEXEL_FASTER - 0.01;
    let not_fast = TEXEL_FASTER + 0.01;
    let tied = TEXEL_NO_SLOWER - 0.01;
    let slower = TEXEL_NO_SLOWER + 0.01;
    let facts = |uniform, lanes| medians([1.0; 2], uniform, lanes, true).costs().facts;
    assert_eq!(facts([fast; 2], [tied; 2]), BTreeSet::from([TEXELS]));
    assert_eq!(
        facts([tied, fast], [tied; 2]),
        BTreeSet::from([TEXELS]),
        "one read at one window clearly faster"
    );
    assert_eq!(
        facts([tied; 2], [tied, fast]),
        BTreeSet::from([TEXELS]),
        "a per-lane read, the per-lane fact not holding"
    );
    assert_eq!(
        facts([not_fast; 2], [tied; 2]),
        BTreeSet::new(),
        "none fast"
    );
    assert_eq!(
        facts([fast; 2], [tied, slower]),
        BTreeSet::new(),
        "a per-lane read slower at one window"
    );
    assert_eq!(
        facts([fast, slower], [fast; 2]),
        BTreeSet::new(),
        "a uniform read slower at one window"
    );
    assert_eq!(
        medians([1.0; 2], [fast; 2], [fast; 2], false).costs().facts,
        BTreeSet::new(),
        "no imported pages, no texel fact"
    );
    let costs = medians([1.0; 2], [0.5, 0.75], [0.9, 0.25], true).costs();
    assert_eq!(costs.texel_fastest, Some(0.25));
    assert_eq!(costs.texel_slowest, Some(0.9));
}

/// Where the per-lane fact holds, the per-lane reads keep their storage view, so the texel fact
/// is judged on the uniform reads alone, both facts holding together.
#[test]
fn where_the_per_lane_fact_holds_the_texel_fact_is_judged_on_the_uniform_reads() {
    let lanes_costly = [PER_LANE_COSTLY - 0.01; 2];
    let fast = [0.5; 2];
    let slower = [1.5; 2];
    assert!(fast[0] < TEXEL_FASTER && slower[0] > TEXEL_NO_SLOWER);
    let costs = medians(lanes_costly, fast, slower, true).costs();
    assert_eq!(costs.facts, BTreeSet::from([LANES, TEXELS]));
    assert_eq!(
        costs.texel_slowest,
        Some(0.5),
        "the per-lane reads not judged"
    );
    let costs = medians(lanes_costly, [1.0; 2], fast, true).costs();
    assert_eq!(
        costs.facts,
        BTreeSet::from([LANES]),
        "per-lane gains moved none"
    );
}

/// The ratios the bounds come from give their facts. Per-lane: NVIDIA's storage per-lane reads at
/// 0.011 to 0.107 of the uniform buffer's (the fact), RADV's at 0.96 to 1.00 (none). Texel, over
/// imported memory, worked out from the Iris Xe's records (job m3-max-37780): from each invocation
/// walking the window at 0.026 to the workgroup walking 4 KiB at 0.86 (the fact); this check's
/// uniform reads on the Iris Xe at 0.48 and 0.755 (job m3-max-38288; its per-lane reads over
/// imported pages were not run then); on the Radeon 780M at 0.93 and 0.97 (none: no read clearly
/// faster), on the RX 7900 XTX at 1.23 and 1.41 and on the RTX 3090 at 2.21 and 2.30 (none:
/// slower; the 3090's per-lane fact holds, so its per-lane reads are not judged).
#[test]
fn the_measured_gpus_get_their_facts() {
    assert_eq!(facts_of(Some(0.107), None, None), BTreeSet::from([LANES]));
    assert_eq!(facts_of(Some(0.011), None, None), BTreeSet::from([LANES]));
    assert_eq!(facts_of(Some(0.96), None, None), BTreeSet::new());
    let iris_records = facts_of(Some(1.0), Some(0.026), Some(0.86));
    assert_eq!(iris_records, BTreeSet::from([TEXELS]));
    let iris_uniform = facts_of(Some(4.798), Some(0.48), Some(0.755));
    assert_eq!(iris_uniform, BTreeSet::from([TEXELS]));
    assert_eq!(
        facts_of(Some(1.001), Some(0.93), Some(0.97)),
        BTreeSet::new()
    );
    assert_eq!(facts_of(Some(1.0), Some(1.23), Some(1.41)), BTreeSet::new());
    let rtx_3090 = facts_of(Some(0.075), Some(2.21), Some(2.30));
    assert_eq!(rtx_3090, BTreeSet::from([LANES]));
    assert_eq!(facts_of(None, None, None), BTreeSet::new());
}

/// The arms: each window's eight device arms, then each window's four imported ones.
#[test]
fn the_arms_are_each_kernel_at_each_window_then_the_imported_ones() {
    let arms = arms();
    assert_eq!(arms.len(), ARMS);
    assert_eq!(arms.iter().filter(|arm| arm.host).count(), 8);
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
    let kept = medians([0.05, 0.1], [0.9, 1.1], [0.5, 0.6], true);
    let bytes = encode(&identity, &kept);
    assert_eq!(decode(&identity, &bytes), Some(kept));
    let unimported = medians([0.05, 0.1], [0.9, 1.1], [0.5, 0.6], false);
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
        medians([1.0; 2], [1.0; 2], [1.0; 2], true).costs(),
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
        ..medians([0.05; 2], [0.4; 2], [0.4; 2], true).costs()
    };
    check.settle(costs, None, "both".to_owned());
    let measured = check.measured().expect("the measured compile device");
    let workarounds = measured.profile().workarounds();
    assert!(workarounds.contains(&LANES) && workarounds.contains(&TEXELS));
    assert!(!base.profile().workarounds().contains(&LANES));
    assert_ne!(measured.profile_fingerprint(), base.profile_fingerprint());
}
