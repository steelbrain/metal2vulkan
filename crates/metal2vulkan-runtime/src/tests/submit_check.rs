//! The device's submission checks' verdicts and records (`submit_check.rs`), from what they
//! measure: no device.

use crate::in_place::DriverIdentity;
use crate::submit_check::{
    decode, encode, merge_try, merge_verdict, record_name, tax_try, tax_verdict, tried, Measured,
    SubmitChecks, Timed, Verdict, MERGE_RATIO, ROUNDS, TAX_FLOOR, TAX_PHASES, TAX_SUBMITS, TRIES,
};

/// The probe's medians (2026-11-07, Mesa 26.2.6 under Linux 7.3.1, MoltenVK 1.4.2 under macOS
/// 27.0.1), in nanoseconds: one call of four infos and four calls apart; a submission with one
/// import of 16 MiB held and without (the tax check's form before it held `TAX_IMPORTS`).
const RADEON_780M: ([i64; 2], [i64; 2]) = ([13_200, 37_300], [20_700, 10_400]);
const RX_7900_XTX: ([i64; 2], [i64; 2]) = ([13_400, 35_000], [15_200, 8_400]);
const RX_550: ([i64; 2], [i64; 2]) = ([11_600, 35_100], [54_700, 8_900]);
const IRIS_XE: ([i64; 2], [i64; 2]) = ([5_100, 13_100], [4_700, 4_500]);
const M3_MAX: ([i64; 2], [i64; 2]) = ([4_500, 4_600], [2_700, 3_000]);

/// Mesa's drivers merge the calls (RADV on three generations, ANV) and MoltenVK does not; RADV
/// taxes every submission for one held import and ANV and MoltenVK do not: the verdicts each
/// device's numbers give, whatever its identity.
#[test]
fn the_probe_s_numbers_give_each_device_its_verdicts() {
    let cases = [
        ("Radeon 780M", RADEON_780M, Verdict::Holds, Verdict::Holds),
        ("RX 7900 XTX", RX_7900_XTX, Verdict::Holds, Verdict::Holds),
        ("RX 550", RX_550, Verdict::Holds, Verdict::Holds),
        ("Iris Xe", IRIS_XE, Verdict::Holds, Verdict::Fails),
        ("M3 Max", M3_MAX, Verdict::Fails, Verdict::Fails),
    ];
    for (name, ([merged, apart], [held, bare]), merges, taxed) in cases {
        assert_eq!(merge_verdict(merged, apart), merges, "{name}");
        assert_eq!(tax_verdict(held, bare), taxed, "{name}");
    }
}

/// The bounds themselves: a call at the ratio merges and one past it does not; a held import
/// that adds the floor, or an eighth of a slow submission, is no tax, and one past it is; no
/// measurement (a check that timed nothing) merges.
#[test]
fn the_verdicts_meet_their_bounds() {
    let apart = 10_000;
    let at = (MERGE_RATIO * apart as f64) as i64;
    assert_eq!(merge_verdict(at, apart), Verdict::Holds);
    assert_eq!(merge_verdict(at + 1, apart), Verdict::Fails);
    assert_eq!(merge_verdict(0, 0), Verdict::Fails);
    assert_eq!(tax_verdict(4_000 + TAX_FLOOR, 4_000), Verdict::Fails);
    assert_eq!(tax_verdict(4_001 + TAX_FLOOR, 4_000), Verdict::Holds);
    let slow = 80_000;
    assert_eq!(tax_verdict(slow + slow / 8, slow), Verdict::Fails);
    assert_eq!(tax_verdict(slow + slow / 8 + 1, slow), Verdict::Holds);
}

/// The import tax's phases, without the import first: each `(floor, rest)`, a phase's least
/// submission and its others.
fn phases(each: [(i64, i64); TAX_PHASES]) -> Vec<Vec<i64>> {
    (each.iter())
        .map(|&(floor, rest)| {
            let mut times = vec![rest; TAX_SUBMITS];
            times[TAX_SUBMITS / 2] = floor;
            times
        })
        .collect()
}

/// The middle of `times`.
fn median(times: &[i64]) -> i64 {
    let mut times = times.to_vec();
    times.sort_unstable();
    times[times.len() / 2]
}

/// A device whose held import costs 5.7 us a submission (the Radeon 780M's floors), its phases
/// without the import lengthened by load beside the check in all but one submission each: the
/// medians of all submissions answer that the import adds nothing, as they did on the 780M under
/// a test's load, and the floors answer that it adds its cost.
#[test]
fn a_taxed_device_under_load_answers_from_its_floors() {
    let loaded = phases([
        (8_600, 25_000),
        (14_300, 15_000),
        (8_700, 25_000),
        (14_200, 15_000),
        (8_500, 25_000),
        (14_400, 15_000),
        (8_800, 25_000),
        (14_500, 15_000),
    ]);
    let (held, bare): (Vec<_>, Vec<_>) =
        (loaded.iter().enumerate()).partition(|(at, _)| at % 2 == 1);
    let all = |phases: Vec<(usize, &Vec<i64>)>| -> Vec<i64> {
        phases
            .into_iter()
            .flat_map(|(_, times)| times.clone())
            .collect()
    };
    let medians = tax_verdict(median(&all(held)), median(&all(bare)));
    assert_eq!(medians, Verdict::Fails, "the medians' answer");
    let floors = tax_try(&loaded);
    assert_eq!(
        floors,
        Timed {
            verdict: Some(Verdict::Holds),
            floors: (14_200, 8_500),
        }
    );
    let quiet = phases([
        (8_600, 9_000),
        (8_700, 9_100),
        (8_500, 9_000),
        (8_800, 9_000),
        (8_600, 9_200),
        (8_700, 9_000),
        (8_500, 9_100),
        (8_600, 9_000),
    ]);
    assert_eq!(tax_try(&quiet).verdict, Some(Verdict::Fails), "no tax");
}

/// One phase or one round that load lengthened throughout moves no floor (another of its form
/// in the same half is quiet), and leaves the verdict as it was; load that lengthened every phase
/// or round of a form in one half splits the halves, and the try gives no verdict.
#[test]
fn a_try_answers_only_where_its_halves_agree() {
    let one_slow = phases([
        (8_600, 9_000),
        (14_300, 15_000),
        (30_000, 31_000),
        (14_200, 15_000),
        (8_500, 9_000),
        (14_400, 15_000),
        (8_700, 9_000),
        (14_100, 15_000),
    ]);
    assert_eq!(tax_try(&one_slow).verdict, Some(Verdict::Holds));
    let half_slow = phases([
        (8_600, 9_000),
        (14_300, 15_000),
        (8_700, 9_000),
        (14_200, 15_000),
        (30_000, 31_000),
        (14_400, 15_000),
        (29_000, 31_000),
        (14_100, 15_000),
    ]);
    assert_eq!(tax_try(&half_slow).verdict, None);
    let apart = vec![35_000; ROUNDS];
    let mut merged = vec![13_000; ROUNDS];
    assert_eq!(merge_try(&merged, &apart).verdict, Some(Verdict::Holds));
    merged[1..ROUNDS / 2].fill(34_000);
    assert_eq!(merge_try(&merged, &apart).verdict, Some(Verdict::Holds));
    merged[..ROUNDS / 2].fill(34_000);
    assert_eq!(merge_try(&merged, &apart).verdict, None);
}

/// The tries: the first that agrees answers, recorded; a refused call answers `Unchecked`,
/// recorded; [`TRIES`] that disagree answer `Unchecked` for the device's life, not recorded, the
/// last try's floors kept.
#[test]
fn tries_that_all_disagree_are_unchecked_and_unrecorded() {
    let timed = |verdict| Timed {
        verdict,
        floors: (1, 2),
    };
    let checks = SubmitChecks::new();
    let mut tries = 0;
    let second = tried(&checks, || {
        tries += 1;
        Some(Some(timed((tries == 2).then_some(Verdict::Holds))))
    });
    assert_eq!((second, tries), (Some((Verdict::Holds, (1, 2), true)), 2));
    let refused = tried(&checks, || Some(None));
    assert_eq!(refused, Some((Verdict::Unchecked, (0, 0), true)));
    let mut tries = 0;
    let split = tried(&checks, || {
        tries += 1;
        Some(Some(timed(None)))
    });
    assert_eq!(
        (split, tries),
        (Some((Verdict::Unchecked, (1, 2), false)), TRIES)
    );
    assert_eq!(tried(&checks, || None), None, "a stopped check");
}

fn identity(device_id: u32) -> DriverIdentity {
    DriverIdentity {
        vendor_id: 0x1002,
        device_id,
        driver_version: 0x6802006,
        driver_id: 3,
        driver_name: "radv".into(),
        driver_info: "Mesa 26.2.6".into(),
        device_uuid: [7; 16],
    }
}

/// A record decodes for its own identity alone, every verdict and measurement as written, and
/// names itself apart from another identity's; a record with a pending verdict, a truncated one
/// and one whose bytes changed decode to nothing.
#[test]
fn a_record_holds_its_identity_s_verdicts() {
    let (one, other) = (identity(0x15bf), identity(0x1114));
    let verdicts = [Verdict::Holds, Verdict::Holds, Verdict::Fails];
    let measured = Measured {
        merged: 13_200,
        apart: 37_300,
        held: 20_700,
        bare: 10_400,
    };
    let bytes = encode(&one, verdicts, measured);
    assert_eq!(decode(&one, &bytes), Some((verdicts, measured)));
    assert_eq!(decode(&other, &bytes), None);
    assert_ne!(record_name(&one), record_name(&other));
    assert_eq!(decode(&one, &bytes[..bytes.len() - 1]), None);
    let mut changed = bytes.clone();
    changed[60] ^= 1;
    assert_eq!(decode(&one, &changed), None);
    let pending = encode(
        &one,
        [Verdict::Holds, Verdict::Pending, Verdict::Fails],
        measured,
    );
    assert_eq!(decode(&one, &pending), None);
}
