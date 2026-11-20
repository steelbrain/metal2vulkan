//! Tests of the run-to-run comparison, over reports built by hand: no device, no case and no
//! recording, so each reading that may move is moved on purpose and one at a time.

use super::{Moved, Stability};
use crate::case::Venue;
use crate::replay::{Allowances, Replayed};
use crate::report::Report;
use crate::transcript::{Node, Observed, Sites, Value};
use crate::verdict::Verdict;

/// The venue every report in these tests is on: the comparison is of two runs on one device, so
/// the venue never moves.
fn venue() -> Venue {
    Venue {
        device: "stub".to_owned(),
        driver: "stub".to_owned(),
        api: "1.2".to_owned(),
        lossless_compression: true,
        host: None,
    }
}

/// One report of one case: its verdict and the one site it observed.
fn report(verdict: Verdict, observed: Value) -> Report {
    let mut sites = Sites::new();
    sites.push(Node::leaf("output", Observed::value(observed)));
    let mut built = Report::new();
    built.add(Replayed {
        case: "one-case".to_owned(),
        venue: venue(),
        verdict,
        allowances: Allowances::default(),
        measured_on: None,
        ours: sites.finish(),
    });
    built
}

/// The reading named `reading`, out of what moved.
fn moved_reading(stability: &Stability, reading: &str) -> Option<Moved> {
    stability
        .moved()
        .into_iter()
        .find(|row| row.reading == reading)
}

/// A value that reads one number on one run and another on the next is flagged, with the case,
/// the site and both readings in order.
///
/// A one-shot gate reads the second number as a regression, because that is the only reading it
/// has.
#[test]
fn a_value_that_moves_between_runs_is_flagged_with_both_readings() {
    let first = report(Verdict::Exact, Value::Uint(2));
    let again = report(Verdict::Exact, Value::Uint(3));
    let mut stability = Stability::new(2);
    stability.observe(2, &first, &again);
    assert!(!stability.is_stable());
    let row = moved_reading(&stability, "the site case/output").expect("the site");
    assert_eq!(row.case, "one-case");
    assert!(row.runs[0].contains("\"uint\":2"), "{:?}", row.runs);
    assert!(row.runs[1].contains("\"uint\":3"), "{:?}", row.runs);
    let printed = stability.render();
    assert!(
        printed.contains("one-case: the site case/output moved"),
        "{printed}"
    );
    assert!(printed.contains("run 1: "), "{printed}");
    assert!(printed.contains("run 2: "), "{printed}");
}

/// A **verdict** that moves is flagged, and so is the site whose value moved under it: the two are
/// separate readings, because a verdict can move with no value moving (an allowance that fired on
/// one run and not the next) and a value can move inside a verdict that stays the same word.
#[test]
fn a_verdict_and_a_value_that_move_are_each_their_own_reading() {
    let first = report(Verdict::Exact, Value::Uint(7));
    let again = report(
        Verdict::Divergent {
            found: vec![crate::diff::Divergence::new(
                "case/output",
                crate::diff::DivergenceKind::Value {
                    metal: Value::Uint(7),
                    ours: Value::Uint(9),
                },
            )],
            refused: Vec::new(),
        },
        Value::Uint(9),
    );
    let mut stability = Stability::new(2);
    stability.observe(2, &first, &again);
    let verdict = moved_reading(&stability, "verdict").expect("the verdict");
    assert_eq!(
        verdict.runs,
        vec!["exact".to_owned(), "divergent".to_owned()]
    );
    let site = moved_reading(&stability, "the site case/output").expect("the site");
    // The reading is the site's own serialization, which is faithful: a summary that printed
    // "4096 bytes" of any 4096 bytes would read two differing byte arrays as equal.
    assert!(site.runs[0].contains("\"uint\":7"), "{:?}", site.runs);
    assert!(site.runs[1].contains("\"uint\":9"), "{:?}", site.runs);
}

/// Two runs that read the same thing move nothing, and one run alone claims nothing.
#[test]
fn an_unchanged_run_moves_nothing() {
    let first = report(Verdict::Exact, Value::Uint(7));
    let again = report(Verdict::Exact, Value::Uint(7));
    let mut stability = Stability::new(2);
    stability.observe(2, &first, &again);
    assert!(stability.is_stable(), "{:?}", stability.moved());
    let printed = stability.render();
    assert!(
        printed.contains("read the same in all of them"),
        "{printed}"
    );
    // And one run alone claims nothing: nothing was compared, so there is nothing to print.
    let alone = Stability::new(1);
    assert!(alone.is_stable());
    assert_eq!(alone.render(), "");
}

/// A reading that moves at run 3 and not at run 2 prints one entry a run, in order, with the runs
/// that read run 1's reading saying so.
///
/// The ordering is the whole point of the output: "which case, which site, the readings in order"
/// is what somebody acts on, and a row that said only "it moved" would send them back to the
/// device to find out when.
#[test]
fn the_readings_are_in_run_order_however_late_one_moves() {
    let first = report(Verdict::Exact, Value::Uint(2));
    let same = report(Verdict::Exact, Value::Uint(2));
    let third = report(Verdict::Exact, Value::Uint(5));
    let fourth = report(Verdict::Exact, Value::Uint(2));
    let mut stability = Stability::new(4);
    stability.observe(2, &first, &same);
    stability.observe(3, &first, &third);
    stability.observe(4, &first, &fourth);
    let row = moved_reading(&stability, "the site case/output").expect("the site");
    let read: Vec<bool> = row
        .runs
        .iter()
        .map(|one| one.contains("\"uint\":5"))
        .collect();
    assert_eq!(read, [false, false, true, false], "{:?}", row.runs);
}

/// A site one run emitted and another did not is a move, said in words rather than left as a
/// missing row.
///
/// A case that stops early on one run and not on another emits fewer sites, which is the shape a
/// flaky refusal takes; the comparison has to name it, because a reading that is sometimes absent
/// is as unstable as one that is sometimes wrong.
#[test]
fn a_site_only_one_run_emitted_is_a_move() {
    let first = report(Verdict::Exact, Value::Uint(7));
    let mut sites = Sites::new();
    let mut again = Report::new();
    again.add(Replayed {
        case: "one-case".to_owned(),
        venue: venue(),
        verdict: Verdict::Exact,
        allowances: Allowances::default(),
        measured_on: None,
        ours: {
            sites.push(Node::leaf("elsewhere", Observed::value(Value::Uint(7))));
            sites.finish()
        },
    });
    let mut stability = Stability::new(2);
    stability.observe(2, &first, &again);
    let gone = moved_reading(&stability, "the site case/output").expect("the site run 2 lost");
    assert!(gone.runs[0].contains("\"uint\":7"), "{:?}", gone.runs);
    assert_eq!(gone.runs[1], super::NO_SUCH_SITE);
    let fresh = moved_reading(&stability, "the site case/elsewhere").expect("the site run 2 grew");
    assert_eq!(fresh.runs[0], super::NO_SUCH_SITE);
    assert!(fresh.runs[1].contains("\"uint\":7"), "{:?}", fresh.runs);
}

/// A reading longer than [`super::READING_CHARS`] is held by its length and digest, so flattening
/// a run costs a line a site whatever the sites observed — and two different long readings still
/// read as a move.
#[test]
fn a_long_reading_is_held_by_its_digest_and_still_moves() {
    let long = |byte: u8| Value::Bytes(vec![byte; 4096]);
    let first = report(Verdict::Exact, long(0xAA));
    let again = report(Verdict::Exact, long(0xBB));
    let mut stability = Stability::new(2);
    stability.observe(2, &first, &again);
    let row = moved_reading(&stability, "the site case/output").expect("the site");
    for read in &row.runs {
        assert!(read.contains("characters, sha256 "), "{read}");
        assert!(
            read.chars().count() < super::READING_CHARS,
            "a held reading stays short: {read}"
        );
    }
    assert_ne!(row.runs[0], row.runs[1]);
    // And the same long value twice moves nothing: the digest is a projection of the value, not
    // of the run.
    let mut steady = Stability::new(2);
    steady.observe(2, &first, &report(Verdict::Exact, long(0xAA)));
    assert!(steady.is_stable(), "{:?}", steady.moved());
}
