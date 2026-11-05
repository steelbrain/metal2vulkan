//! Degrade notices (`degradation.rs`): once per device and kind, a warning named by the kind's code
//! and carrying its payload, in one closed vocabulary that docs/API.md's catalogue lists exactly.

use crate::degradation::{Announced, DegradeNotice};
use crate::{Degradation, DegradationCategory, MessageSeverity};
use metal2vulkan::pipeline::PrecisionNote;
use std::collections::{BTreeMap, BTreeSet};

#[test]
fn each_kind_is_announced_once() {
    let announced = Announced::default();
    for &kind in Degradation::ALL {
        assert!(announced.now(kind), "{kind:?}, first");
        assert!(!announced.now(kind), "{kind:?}, second");
    }
    for &kind in Degradation::ALL {
        assert!(!announced.now(kind), "{kind:?}, after every kind");
    }
    // Another device's kinds are its own.
    let other = Announced::default();
    assert!(other.now(Degradation::CopiedDrawBytes));
}

/// A kind the device's options ask for at each occurrence is announced at every one; every other
/// kind once, as without them.
#[test]
fn a_kind_asked_for_at_each_occurrence_is_announced_at_each() {
    let each = [
        Degradation::ReleasedCommandState,
        Degradation::UnlistedArgumentSampler,
    ];
    let announced = Announced::new(&each.into_iter().collect());
    for &kind in Degradation::ALL {
        let asked = each.contains(&kind);
        assert!(announced.now(kind), "{kind:?}, first");
        assert_eq!(announced.now(kind), asked, "{kind:?}, second");
        assert_eq!(announced.now(kind), asked, "{kind:?}, third");
    }
}

#[test]
fn codes_are_distinct_and_decode() {
    let mut seen = BTreeMap::new();
    for &kind in Degradation::ALL {
        let code = kind.code();
        let rest = code
            .strip_prefix("metal2vulkan-degrade-")
            .unwrap_or_default();
        assert!(
            !rest.is_empty()
                && rest
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-'),
            "{kind:?}: {code}"
        );
        assert_eq!(seen.insert(code, kind), None, "{code} names two kinds");
        assert_eq!(Degradation::from_code(code), Some(kind));
        assert!(!kind.cost().is_empty(), "{kind:?}");
        // Its name, code and cost read with single spaces: a string literal continued onto the
        // next line without a `\` keeps the line's break and indent, which every relayed notice
        // line would carry.
        for text in [format!("{kind:?}"), code.to_owned(), kind.cost().to_owned()] {
            assert!(
                !text.contains("  ") && !text.contains(['\n', '\t']),
                "{kind:?}: {text:?} holds two spaces, a newline or a tab"
            );
        }
    }
    assert_eq!(Degradation::from_code("metal2vulkan-degrade-"), None);
}

#[test]
fn every_precision_note_has_its_kind() {
    let notes: Vec<(Degradation, PrecisionNote)> = Degradation::ALL
        .iter()
        .filter_map(|&kind| kind.precision_note().map(|note| (kind, note)))
        .collect();
    for &(kind, note) in &notes {
        assert_eq!(Degradation::from_precision_note(note), kind);
        assert_eq!(kind.category(), DegradationCategory::Precision, "{kind:?}");
        assert_eq!(format!("{kind:?}"), format!("{note:?}"));
    }
    // `from_precision_note` matches every note (the compiler checks it), and each kind maps back.
    assert!(!notes.is_empty());
}

#[test]
fn notices_are_warnings_named_by_code() {
    for &kind in Degradation::ALL {
        let notice = DegradeNotice::new(kind, "an object".to_owned(), "a reason".to_owned());
        let message = notice.clone().message();
        assert_eq!(message.severity, MessageSeverity::Warning);
        assert_eq!(message.id_name, kind.code());
        assert_eq!(message.id, 0);
        let performance = kind.category() == DegradationCategory::Performance;
        assert_eq!(message.kind.performance, performance, "{kind:?}");
        assert_eq!(message.kind.general, !performance, "{kind:?}");
        assert!(!message.kind.validation);
        assert_eq!(
            message.text,
            format!(
                "metal2vulkan: an object: a reason; {} [{}]",
                kind.cost(),
                kind.code()
            )
        );
        assert_eq!(message.degradation, Some(notice));
    }
}

/// docs/API.md's catalogue lists every kind's code once, with its category, and no other code.
#[test]
fn the_catalogue_lists_every_kind_and_only_them() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/API.md");
    let guide = std::fs::read_to_string(path).unwrap();
    let mut listed = BTreeMap::new();
    for line in guide.lines() {
        let Some(entry) = line.strip_prefix("- `metal2vulkan-degrade-") else {
            continue;
        };
        let (rest, after) = entry.split_once('`').unwrap_or_default();
        let code = format!("metal2vulkan-degrade-{rest}");
        let category = if after.starts_with(" (performance)") {
            DegradationCategory::Performance
        } else if after.starts_with(" (precision)") {
            DegradationCategory::Precision
        } else if after.starts_with(" (validity)") {
            DegradationCategory::Validity
        } else {
            panic!("{code}: no category: {line}");
        };
        assert_eq!(listed.insert(code.clone(), category), None, "{code} twice");
    }
    for &kind in Degradation::ALL {
        assert_eq!(
            listed.remove(kind.code()),
            Some(kind.category()),
            "{kind:?} ({}) is not listed with its category",
            kind.code()
        );
    }
    assert!(listed.is_empty(), "codes of no kind: {listed:?}");
}

/// A kind asked for at every occurrence may be announced on the program's own threads, in a
/// runtime thread running the program's code, and in a host step on any thread; not on a runtime
/// thread by itself, nor in a pipeline pool's piece. Any other kind anywhere.
#[test]
fn a_kind_at_every_occurrence_is_announced_in_the_program_s_call_or_a_host_step() {
    use crate::degradation::each_in_order;
    use crate::exit_guard::{host_step, program, spawn};
    use crate::pipeline::pool::{Joined, Pool};
    use std::cell::Cell;
    // The program's own thread.
    assert!(each_in_order(true), "the program's thread");
    assert!(host_step(|| each_in_order(true)), "a host step on it");
    // A runtime thread: by itself, running the program's code, in a host step.
    let seen = spawn(std::thread::Builder::new(), || {
        let in_program = Cell::new(false);
        program(|| in_program.set(each_in_order(true)));
        let step = host_step(|| each_in_order(true));
        let after = each_in_order(true);
        (
            each_in_order(true),
            in_program.get(),
            step,
            after,
            each_in_order(false),
        )
    })
    .unwrap()
    .join()
    .unwrap();
    assert_eq!(
        seen,
        (false, true, true, false, true),
        "a runtime thread: by itself, the program's code, a host step, after it, any other kind"
    );
    // A pool's piece, on the pool's thread; the joined set's first piece, the caller's own, is a
    // piece too: it runs beside the pool's.
    let pool = Pool::default();
    let pieces: Vec<Joined<(bool, bool)>> = (0..2)
        .map(|_| -> Joined<(bool, bool)> {
            Box::new(|| (each_in_order(true), each_in_order(false)))
        })
        .collect();
    let seen: Vec<_> = pool
        .join_all(pieces, true, |_, _| {})
        .into_iter()
        .map(Option::unwrap)
        .collect();
    assert_eq!(
        seen,
        vec![(false, true), (false, true)],
        "the caller's piece, then a pool piece"
    );
}

/// An announcement inside a pipeline pool's piece ends the process with its named line in a build
/// with debug assertions, and is answered as misplaced in another (the notice thread then delivers
/// it, `misplaced_notices.rs`), as is one of a kind asked for at every occurrence on a runtime
/// thread by itself. A forked child announcing inside a piece ends on `SIGABRT` where debug
/// assertions are on, and exits otherwise; the parent then checks the rule's answers in a joined
/// set's pieces (the caller's own and a pool thread's), on a runtime thread, and on the program's
/// thread.
#[test]
fn an_announcement_inside_a_pipeline_pool_piece_ends_the_process_where_debug_assertions_are_on() {
    use crate::degradation::{announcement_breach, check_announcement, Degradation, NoticePlace};
    use crate::exit_guard::spawn;
    use crate::pipeline::pool::{Joined, Pool};
    // First, before this process starts a thread: a child whose one piece, run on its own
    // thread as the joined set's first, announces.
    // SAFETY: the child runs a piece on its only thread (the joined set's first piece runs on
    // the caller's), writes a line and aborts, or exits at once.
    let child = unsafe { libc::fork() };
    if child == 0 {
        let pieces: Vec<Joined<()>> = vec![Box::new(|| {
            let place = check_announcement(Degradation::EncodedUseRebound, false);
            if place != Some(NoticePlace::PipelinePoolPiece) {
                // SAFETY: ends the child at once, telling the parent the answer was wrong.
                unsafe { libc::_exit(3) };
            }
        })];
        drop(Pool::default().join_all(pieces, true, |_, _| {}));
        // SAFETY: ends the child at once; the check did not end it.
        unsafe { libc::_exit(0) };
    }
    assert!(child > 0, "fork");
    let mut status = 0;
    // SAFETY: `child` is this process's child.
    let reaped = unsafe { libc::waitpid(child, &mut status, 0) };
    assert_eq!(reaped, child);
    if cfg!(debug_assertions) {
        assert!(
            libc::WIFSIGNALED(status) && libc::WTERMSIG(status) == libc::SIGABRT,
            "the child announcing inside a piece was not aborted: {status}"
        );
    } else {
        assert!(
            libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0,
            "the child announcing inside a piece did not go on, answered as misplaced: {status}"
        );
    }
    // The program's thread keeps the rule.
    assert_eq!(
        (announcement_breach(false), announcement_breach(true)),
        (None, None)
    );
    // A joined set's pieces break it, whatever the kind.
    let pool = Pool::default();
    type Answers = (Option<NoticePlace>, Option<NoticePlace>);
    let pieces: Vec<Joined<Answers>> = (0..2)
        .map(|_| -> Joined<Answers> {
            Box::new(|| (announcement_breach(false), announcement_breach(true)))
        })
        .collect();
    for seen in pool.join_all(pieces, true, |_, _| {}) {
        let (once, each) = seen.unwrap();
        let place = once.expect("an announcement inside a piece");
        assert_eq!(place, NoticePlace::PipelinePoolPiece);
        assert_eq!(
            place.line(),
            "metal2vulkan-runtime: a degradation was announced inside a pipeline pool piece"
        );
        assert_eq!(each, Some(place));
    }
    // A runtime thread by itself: a kind asked for at every occurrence alone.
    let (once, each) = spawn(std::thread::Builder::new(), || {
        (announcement_breach(false), announcement_breach(true))
    })
    .unwrap()
    .join()
    .unwrap();
    assert_eq!(once, None);
    assert_eq!(each, Some(NoticePlace::OutsideProgramCall));
    assert!(
        NoticePlace::OutsideProgramCall
            .line()
            .contains("outside the program's call"),
        "{:?}",
        NoticePlace::OutsideProgramCall.line()
    );
}

/// A misplaced notice in a build without debug assertions: met inside a pipeline pool's piece
/// while the meeting thread holds a lock, it reaches the sink unchanged, from the notice thread,
/// whose handler takes that lock (on the meeting thread it would wait for itself); a second
/// misplaced notice reaches it too; `misplaced-notice` follows the first alone, naming its kind
/// and place, an exact performance kind. The check a build without debug assertions makes is
/// [`placement`] without ending the process, which this test makes in every build (in such a
/// build `check_announcement` itself is checked too); a gate's build with debug assertions ends
/// the process at the breach instead (the test above).
#[test]
fn a_misplaced_notice_reaches_the_sink_unchanged_from_the_notice_thread_and_is_recorded_once() {
    use crate::degradation::{
        check_announcement, placement, MisplacedNotice, NoticePlace, MISPLACED_ENDS,
    };
    use crate::misplaced_notices::{deliver, THREAD};
    use crate::pipeline::pool::{Joined, Pool};
    use crate::{DebugMessage, Exactness, MessageSink};
    use std::sync::{mpsc, Arc, Mutex};
    use std::time::Duration;
    assert_eq!(MISPLACED_ENDS, cfg!(debug_assertions));
    let held = Arc::new(Mutex::new(()));
    let (heard, hearing) = mpsc::channel::<(Option<String>, DebugMessage)>();
    let sink = {
        let held = Arc::clone(&held);
        let heard = Mutex::new(heard);
        MessageSink::new(move |message| {
            // A handler that takes the lock the meeting thread held.
            drop(held.lock().unwrap());
            let name = std::thread::current().name().map(str::to_owned);
            heard.lock().unwrap().send((name, message.clone())).unwrap();
        })
    };
    let announced = Arc::new(Announced::new(&BTreeSet::new()));
    let notice = |kind: Degradation| {
        DegradeNotice::new(kind, format!("{kind:?}'s object"), "its reason".to_owned()).message()
    };
    let sent = [
        notice(Degradation::SplitDispatch),
        notice(Degradation::SplitBufferFill),
    ];
    let pieces: Vec<Joined<Vec<Option<NoticePlace>>>> = vec![Box::new({
        let (held, sink, announced, sent) = (
            Arc::clone(&held),
            sink.clone(),
            Arc::clone(&announced),
            sent.clone(),
        );
        move || {
            let _held = held.lock().unwrap();
            sent.into_iter()
                .map(|message| {
                    let kind = message.degradation.as_ref().unwrap().degradation;
                    let place = placement(kind, false, false);
                    if !cfg!(debug_assertions) {
                        assert_eq!(check_announcement(kind, false), place);
                    }
                    if let Some(place) = place {
                        let misplaced = MisplacedNotice { kind, place };
                        deliver(Some(&sink), &announced, misplaced, message);
                    }
                    place
                })
                .collect()
        }
    })];
    let places = Pool::default()
        .join_all(pieces, true, |_, _| {})
        .pop()
        .flatten()
        .unwrap();
    assert_eq!(places, vec![Some(NoticePlace::PipelinePoolPiece); 2]);
    let wait = Duration::from_secs(10);
    let delivered: Vec<(Option<String>, DebugMessage)> = (0..3)
        .map(|_| hearing.recv_timeout(wait).expect("a delivered notice"))
        .collect();
    assert!(
        hearing.recv_timeout(Duration::from_millis(200)).is_err(),
        "three, no more"
    );
    for (name, _) in &delivered {
        assert_eq!(
            name.as_deref(),
            Some(THREAD),
            "delivered from the notice thread"
        );
    }
    assert_eq!(delivered[0].1, sent[0], "the first notice, unchanged");
    let record = delivered[1].1.degradation.as_ref().expect("a notice");
    assert_eq!(record.degradation, Degradation::MisplacedNotice);
    assert_eq!(
        record.misplaced,
        Some(MisplacedNotice {
            kind: Degradation::SplitDispatch,
            place: NoticePlace::PipelinePoolPiece,
        })
    );
    assert!(
        record.object.contains("split-dispatch"),
        "{}",
        record.object
    );
    assert_eq!(
        delivered[2].1, sent[1],
        "the second notice, unchanged, with no record"
    );
    assert_eq!(
        Degradation::MisplacedNotice.category(),
        DegradationCategory::Performance
    );
    assert_eq!(Degradation::MisplacedNotice.exactness(), Exactness::Exact);
}

/// The kinds met once per object, which the device announces once and counts each object of
/// (`Device::degraded_objects`), are exactly the five whose notices name an object's ID: each
/// counter starts at zero and counts its own kind's objects alone; every other kind is not
/// counted.
#[test]
fn the_kinds_met_per_object_are_counted_and_no_other() {
    use crate::work_counts::Counters;
    let per_object = [
        Degradation::StaleDetachedReach,
        Degradation::DetachedStorageKeepsWrites,
        Degradation::HeapTextureLosslessLayout,
        Degradation::HeapTextureBlockCompressedLayout,
        Degradation::HeapTextureUntiledLayout,
    ];
    let counters = Counters::default();
    for &kind in Degradation::ALL {
        let counted = counters.degraded_objects(kind);
        assert_eq!(counted.is_some(), per_object.contains(&kind), "{kind:?}");
        assert!(counted.is_none_or(|n| n == 0), "{kind:?}");
    }
    counters.stale_detached_reach();
    counters.detached_storage_reach();
    counters.detached_storage_reach();
    for (n, kind) in per_object[2..].iter().enumerate() {
        for _ in 0..=n {
            counters.heap_layout_degraded(*kind);
        }
    }
    let counted: Vec<_> = per_object
        .iter()
        .map(|kind| counters.degraded_objects(*kind))
        .collect();
    assert_eq!(counted, [Some(1), Some(2), Some(1), Some(2), Some(3)]);
}

/// A stage that records `PrecisionNote::FunctionStorageAssumed` names, in numbers, the private
/// memory a thread of it keeps and what the device holds, the floor of a device not measured.
#[test]
fn function_storage_assumed_names_the_bytes_and_the_device_s_evidence() {
    use crate::degradation::function_storage_reason;
    use metal2vulkan::device::FunctionStorage;
    let recorded = "its translation records `PrecisionNote::FunctionStorageAssumed`";
    assert_eq!(
        function_storage_reason(recorded, 102_400, FunctionStorage::Assumed(16_384)),
        format!(
            "{recorded}: it keeps 102400 bytes of private memory a thread, past the 16384 bytes a \
             thread the floor of a device whose Function storage is not measured"
        )
    );
    assert_eq!(
        function_storage_reason(recorded, 40_000, FunctionStorage::Measured(32_768)),
        format!(
            "{recorded}: it keeps 40000 bytes of private memory a thread, past the 32768 bytes a \
             thread measured on this device"
        )
    );
}
