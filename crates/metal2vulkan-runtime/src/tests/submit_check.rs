//! The device's submission checks' verdicts and records (`submit_check.rs`), from what they
//! measure: no device.

use crate::in_place::DriverIdentity;
use crate::submit_check::{
    decode, encode, merge_clear, merge_tried, merge_try, merge_verdict, record_name, tax_clear,
    tax_holds, tax_tried, tax_try, tax_verdict, tried, Measured, SubmitChecks, Timed, Verdict,
    MERGE_FAILS_CLEAR, MERGE_HOLDS_CLEAR, MERGE_RATIO, MERGE_REPEATS, ROUNDS, TAX_FLOOR,
    TAX_MARGIN, TAX_PHASES, TAX_REPEATS, TAX_SUBMITS, TRIES,
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

/// The import tax's phases, those [`tax_holds`] names with the imports held: each
/// `(floor, rest)`, a phase's least submission and its others.
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
        (8_700, 25_000),
        (14_300, 15_000),
        (14_200, 15_000),
        (14_400, 15_000),
        (14_500, 15_000),
        (8_500, 25_000),
        (8_800, 25_000),
    ]);
    let (held, bare): (Vec<_>, Vec<_>) =
        (loaded.iter().enumerate()).partition(|&(at, _)| tax_holds(at));
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
        (30_000, 31_000),
        (14_300, 15_000),
        (14_200, 15_000),
        (14_400, 15_000),
        (14_100, 15_000),
        (8_500, 9_000),
        (8_700, 9_000),
    ]);
    assert_eq!(tax_try(&one_slow).verdict, Some(Verdict::Holds));
    let half_slow = phases([
        (8_600, 9_000),
        (8_700, 9_000),
        (14_300, 15_000),
        (14_200, 15_000),
        (14_400, 15_000),
        (14_100, 15_000),
        (30_000, 31_000),
        (29_000, 31_000),
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

/// The import tax's phases that hold the imports run one after another, so a try makes and
/// releases one set of [`TAX_IMPORTS`](crate::submit_check::TAX_IMPORTS) imports (each import and
/// release holds back every other thread's submission on NVIDIA's driver); each half of the
/// phases holds as many with the imports as without; the first and the last hold none.
#[test]
fn a_tax_try_holds_one_set_of_imports_through_half_its_phases() {
    let held: Vec<bool> = (0..TAX_PHASES).map(tax_holds).collect();
    let sets = (0..TAX_PHASES)
        .filter(|&at| held[at] && (at == 0 || !held[at - 1]))
        .count();
    assert_eq!(sets, 1, "the sets of imports a try makes: {held:?}");
    for half in held.chunks(TAX_PHASES / 2) {
        let holding = half.iter().filter(|&&holds| holds).count();
        assert_eq!(holding, half.len() / 2, "{held:?}");
    }
    assert!(!held[0] && !held[TAX_PHASES - 1], "{held:?}");
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
    let checks = SubmitChecks::new(false);
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

/// What the merge's tries answer ([`merge_tried`]) where each answers the next of `answers` (the
/// last again past them), and how many it made.
fn merge_answers(answers: &[Option<Timed>]) -> (Option<(Verdict, (i64, i64), bool)>, usize) {
    let checks = SubmitChecks::new(false);
    let mut tries = 0;
    let answered = merge_tried(&checks, || {
        let answer = answers[tries.min(answers.len() - 1)];
        tries += 1;
        Some(answer)
    });
    (answered, tries)
}

/// A merge try that answered `verdict` with `floors` over the whole try.
fn merge_answer(verdict: Option<Verdict>, floors: (i64, i64)) -> Option<Timed> {
    Some(Timed { verdict, floors })
}

/// A merge verdict is recorded only where its try's floors are clear of the load beside it and
/// each of [`MERGE_REPEATS`] tries more answers it so: the quiet Radeon 780M's `Holds` and the
/// quiet M3 Max's `Fails` are recorded after their repeats. The loaded tries of a full run of the
/// device tests answer for the device's life alone, unrecorded, where one try alone recorded
/// them before ([`tried`]): the Radeon 780M's `Fails` at 0.69 of its calls apart, and the M3
/// Max's `Holds` at 0.29, clear but not repeated.
#[test]
fn a_merge_verdict_is_recorded_only_clear_of_the_load_and_repeated() {
    let holds = |floors| merge_answer(Some(Verdict::Holds), floors);
    let fails = |floors| merge_answer(Some(Verdict::Fails), floors);
    let radeon = (9_200, 26_500);
    let recorded = (Some((Verdict::Holds, radeon, true)), 1 + MERGE_REPEATS);
    assert_eq!(
        merge_answers(&[holds(radeon)]),
        recorded,
        "the quiet Radeon 780M"
    );
    let m3 = (6_100, 7_000);
    let recorded = (Some((Verdict::Fails, m3, true)), 1 + MERGE_REPEATS);
    assert_eq!(merge_answers(&[fails(m3)]), recorded, "the quiet M3 Max");
    let loaded_radeon = (63_779, 92_121);
    let loaded_m3 = (4_542, 15_833);
    for (answer, verdict, floors, name) in [
        (
            fails(loaded_radeon),
            Verdict::Fails,
            loaded_radeon,
            "the loaded Radeon 780M",
        ),
        (
            holds(loaded_m3),
            Verdict::Holds,
            loaded_m3,
            "the loaded M3 Max",
        ),
    ] {
        let checks = SubmitChecks::new(false);
        let before = tried(&checks, || Some(answer));
        assert_eq!(before, Some((verdict, floors, true)), "{name}, one try");
        let again = match verdict {
            Verdict::Fails => holds(radeon),
            _ => fails(m3),
        };
        let unrecorded = Some((verdict, floors, false));
        assert_eq!(merge_answers(&[answer, again]).0, unrecorded, "{name}");
    }
    let unrecorded = (Some((Verdict::Holds, radeon, false)), 2);
    for (again, why) in [
        (fails(m3), "a repeat that fails"),
        (holds((15_000, 26_500)), "a repeat short of the clear bound"),
        (merge_answer(None, radeon), "a repeat that splits"),
        (None, "a repeat refused"),
    ] {
        assert_eq!(merge_answers(&[holds(radeon), again]), unrecorded, "{why}");
    }
    let apart = 10_000;
    let at = |bound: f64| (bound * apart as f64) as i64;
    assert!(merge_clear(Verdict::Holds, (at(MERGE_HOLDS_CLEAR), apart)));
    assert!(!merge_clear(
        Verdict::Holds,
        (at(MERGE_HOLDS_CLEAR) + 1, apart)
    ));
    assert!(merge_clear(Verdict::Fails, (at(MERGE_FAILS_CLEAR), apart)));
    assert!(!merge_clear(
        Verdict::Fails,
        (at(MERGE_FAILS_CLEAR) - 1, apart)
    ));
    assert!(!merge_clear(Verdict::Unchecked, radeon));
    assert!(!merge_clear(Verdict::Holds, (0, 0)));
}

/// What the import tax's tries answer ([`tax_tried`]) where each answers the next of `answers`
/// (the last again past them), and how many it made.
fn tax_answers(answers: &[Option<Timed>]) -> (Option<(Verdict, (i64, i64), bool)>, usize) {
    let checks = SubmitChecks::new(false);
    let mut tries = 0;
    let answered = tax_tried(&checks, || {
        let answer = answers[tries.min(answers.len() - 1)];
        tries += 1;
        Some(answer)
    });
    (answered, tries)
}

/// An import tax `Holds` is recorded only where it passes its bound [`TAX_MARGIN`] times over in
/// its try and in each of [`TAX_REPEATS`] tries more: the Radeon 780M's is recorded after its
/// repeats; one whose repeat answers `Fails`, falls short of the margin, splits or is refused
/// answers for the device's life with its first try's floors, unrecorded; a `Fails` is recorded
/// after one try.
#[test]
fn an_import_tax_holds_is_recorded_only_past_its_margin_and_repeated() {
    let holds = |floors| {
        Some(Timed {
            verdict: Some(Verdict::Holds),
            floors,
        })
    };
    let fails = |floors| {
        Some(Timed {
            verdict: Some(Verdict::Fails),
            floors,
        })
    };
    let radeon = (20_700, 10_400);
    let recorded = (Some((Verdict::Holds, radeon, true)), 1 + TAX_REPEATS);
    assert_eq!(tax_answers(&[holds(radeon)]), recorded, "the Radeon 780M");
    let unrecorded = (Some((Verdict::Holds, radeon, false)), 2);
    let short = holds((12_000, 10_400));
    let split = Some(Timed {
        verdict: None,
        floors: radeon,
    });
    for (again, why) in [
        (fails((10_500, 10_400)), "a repeat that fails"),
        (short, "a repeat short of the margin"),
        (split, "a repeat that splits"),
        (None, "a repeat refused"),
    ] {
        assert_eq!(tax_answers(&[holds(radeon), again]), unrecorded, "{why}");
    }
    let m3 = (2_700, 3_000);
    let one = (Some((Verdict::Fails, m3, true)), 1);
    assert_eq!(tax_answers(&[fails(m3)]), one, "a Fails");
    let bare = 4_000;
    assert!(!tax_clear((bare + TAX_MARGIN * TAX_FLOOR, bare)));
    assert!(tax_clear((bare + TAX_MARGIN * TAX_FLOOR + 1, bare)));
    let slow = 80_000;
    assert!(!tax_clear((slow + TAX_MARGIN * (slow / 8), slow)));
    assert!(tax_clear((slow + TAX_MARGIN * (slow / 8) + 1, slow)));
}

/// An import tax `Holds` short of its margin answers `Fails` for the device's life, unrecorded,
/// after one try: the RTX 5080 Laptop GPU's under a loop of the device tests (3.4 against 1.9 us,
/// past the bound by half, where its quiet devices answered `Fails` from 2.1 to 2.2 us both) and
/// lavapipe's under the device tests (1.4 against 0.2 us, past it by a fifth), each of which
/// answered `Holds` and took a taxed device's routes the quiet devices never take.
#[test]
fn an_import_tax_holds_short_of_its_margin_answers_fails_unrecorded() {
    let holds = |floors| {
        Some(Timed {
            verdict: Some(Verdict::Holds),
            floors,
        })
    };
    for (floors, name) in [
        ((3_400, 1_900), "the RTX 5080 under load"),
        ((1_400, 200), "lavapipe under load"),
    ] {
        assert_eq!(tax_verdict(floors.0, floors.1), Verdict::Holds, "{name}");
        assert!(!tax_clear(floors), "{name}");
        let fails = (Some((Verdict::Fails, floors, false)), 1);
        assert_eq!(tax_answers(&[holds(floors)]), fails, "{name}");
    }
}

/// Import tax tries that all split answer `Fails` for the device's life, unrecorded, where the
/// last one's floors over all its phases fall short of the bound [`TAX_MARGIN`] times over: the M3
/// Max's device under the load of the device tests, 2.5 against 1.2 us, which answered
/// `Unchecked` and took no fresh pages where the five others of its runs answered `Fails`. Past
/// the margin (the Radeon 780M's floors) they stay `Unchecked` so; a merge's split tries are
/// unchanged.
#[test]
fn import_tax_tries_that_all_split_short_of_the_margin_answer_fails_unrecorded() {
    let split = |floors| {
        Some(Timed {
            verdict: None,
            floors,
        })
    };
    let m3 = (2_500, 1_200);
    assert!(!tax_clear(m3));
    let fails = (Some((Verdict::Fails, m3, false)), TRIES);
    assert_eq!(tax_answers(&[split(m3)]), fails, "the M3 Max under load");
    let radeon = (20_700, 10_400);
    let unchecked = (Some((Verdict::Unchecked, radeon, false)), TRIES);
    assert_eq!(
        tax_answers(&[split(radeon)]),
        unchecked,
        "a tax past the margin"
    );
    let checks = SubmitChecks::new(false);
    let merge = tried(&checks, || Some(split(m3)));
    assert_eq!(merge, Some((Verdict::Unchecked, m3, false)), "a merge's");
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
