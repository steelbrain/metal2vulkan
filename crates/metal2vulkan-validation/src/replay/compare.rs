//! Two plays' reports compared for a timing verdict (`docs/REPLAY-STREAM.md`, section 10, two
//! plays compared): `command-replay compare-plays`. A verdict at a frame needs both plays played
//! by a player that serves every read class a stream has (its report has the `managed reads`
//! line), the frame's player-side counts equal in both (the stream's calls, draws and dispatches,
//! and the backend's skipped and refused counts), and equal digests there (the `check` lines of
//! the frame). The stale reads the line counts bar nothing: where the digests differ, they are the
//! first suspect.

use std::collections::BTreeMap;

/// The most lines of one kind the compare names one by one.
const NAMED: usize = 8;

/// One frame's player-side counts: each key and its value.
type FrameCounts = BTreeMap<String, u64>;

/// What a compare reads of one report.
#[derive(Default)]
struct Play {
    /// The `managed reads` line's stale reads' first frame, where the line names one, and whether
    /// the report has the line at all.
    served: bool,
    stale_from: Option<u64>,
    /// Whether the backend kept the stream's Managed buffers in two copies.
    two_copies: bool,
    frames: BTreeMap<u64, FrameCounts>,
    /// Each frame's wall and GPU-busy nanoseconds.
    ends: BTreeMap<u64, (u64, u64)>,
    /// Each frame's checks: object and digest.
    checks: BTreeMap<u64, Vec<(u64, String)>>,
}

/// Whether a frame line's `executed` key is a player-side count: a skip or a refusal.
fn player_side(key: &str) -> bool {
    key.starts_with("skipped_") || key.ends_with("_refusals")
}

/// The `key=value` words of `words` with a value that is a count.
fn pairs<'w>(words: impl Iterator<Item = &'w str>) -> impl Iterator<Item = (&'w str, u64)> {
    words.filter_map(|word| {
        let (key, value) = word.split_once('=')?;
        Some((key, value.parse().ok()?))
    })
}

fn read(report: &str) -> Play {
    let mut play = Play::default();
    for line in report.lines() {
        let mut words = line.split_whitespace();
        match (words.next(), words.next()) {
            (Some("frame"), Some(frame)) => {
                let Ok(frame) = frame.parse() else { continue };
                let mut counts = FrameCounts::new();
                let mut part = "";
                for word in words {
                    if !word.contains('=') {
                        part = word;
                        continue;
                    }
                    let Some((key, value)) = pairs(std::iter::once(word)).next() else {
                        continue;
                    };
                    let counted = match part {
                        "stream" => ["calls", "draws", "dispatches"].contains(&key),
                        _ => player_side(key),
                    };
                    if counted {
                        counts.insert(format!("{part}.{key}"), value);
                    }
                }
                play.frames.insert(frame, counts);
            }
            (Some("end"), Some(frame)) => {
                let Ok(frame) = frame.parse() else { continue };
                let found: BTreeMap<&str, u64> = pairs(words).collect();
                let wall = found.get("wall_nanos").copied().unwrap_or(0);
                let busy = found.get("gpu_busy").copied().unwrap_or(0);
                play.ends.insert(frame, (wall, busy));
            }
            (Some("check"), Some(object)) => {
                let object = object.trim_start_matches('#').parse();
                let (Some("frame"), Some(frame), Some(sha)) =
                    (words.next(), words.next(), words.next())
                else {
                    continue;
                };
                if let (Ok(object), Ok(frame)) = (object, frame.parse()) {
                    let checks = play.checks.entry(frame).or_default();
                    checks.push((object, sha.to_owned()));
                }
            }
            (Some("managed"), Some("reads")) => {
                play.served = true;
                let stale = line.split_once("; stale ").map(|(_, rest)| rest);
                let from = stale.and_then(|rest| rest.split_once("first at frame "));
                let from =
                    from.and_then(|(_, rest)| rest.split(|c: char| !c.is_ascii_digit()).next());
                play.stale_from = from.and_then(|frame| frame.parse().ok());
            }
            (Some("managed"), Some("two")) => play.two_copies = true,
            _ => {}
        }
    }
    play
}

/// The first key whose counts differ between two frames, with both values (0 where one lacks it).
fn differs(a: &FrameCounts, b: &FrameCounts) -> Option<(String, u64, u64)> {
    let keys = a.keys().chain(b.keys());
    keys.map(|key| {
        let get = |counts: &FrameCounts| counts.get(key).copied().unwrap_or(0);
        (key.clone(), get(a), get(b))
    })
    .find(|(_, x, y)| x != y)
}

/// The compare's lines, and whether frame `frame`, where one is asked, has a verdict.
pub fn compare_plays(a: &str, b: &str, frame: Option<u64>) -> (Vec<String>, bool) {
    let (a, b) = (read(a), read(b));
    let mut lines = Vec::new();
    for (name, play) in [("a", &a), ("b", &b)] {
        lines.push(match play.served {
            true => format!("served {name}: yes"),
            false => format!(
                "served {name}: no: no managed reads line, so a player older than the made-with \
                 and first-use rules"
            ),
        });
    }
    let both: Vec<u64> = (a.frames.keys())
        .filter(|f| b.frames.contains_key(f))
        .copied()
        .collect();
    let unequal: Vec<(u64, (String, u64, u64))> = (both.iter())
        .filter_map(|f| Some((*f, differs(&a.frames[f], &b.frames[f])?)))
        .collect();
    lines.push(format!(
        "frames a {} b {}; both {}; counts equal in {}, unequal in {}",
        a.frames.len(),
        b.frames.len(),
        both.len(),
        both.len() - unequal.len(),
        unequal.len()
    ));
    for (f, (key, x, y)) in unequal.iter().take(NAMED) {
        lines.push(format!("unequal frame {f}: {key} a={x} b={y}"));
    }
    let digested: Vec<u64> = (a.checks.keys())
        .filter(|f| b.checks.contains_key(f))
        .copied()
        .collect();
    let differing: Vec<u64> = (digested.iter())
        .filter(|f| a.checks[f] != b.checks[f])
        .copied()
        .collect();
    lines.push(format!(
        "digests: {} frames with checks in both; equal in {}, differ in {}",
        digested.len(),
        digested.len() - differing.len(),
        differing.len()
    ));
    let suspect = |f: u64| {
        let from = [&a, &b]
            .into_iter()
            .filter(|play| play.two_copies)
            .filter_map(|play| play.stale_from)
            .min();
        match from {
            Some(from) if from <= f => {
                format!("; stale reads from frame {from}, the first suspect")
            }
            _ => String::new(),
        }
    };
    for f in differing.iter().take(NAMED) {
        lines.push(format!("digests differ at frame {f}{}", suspect(*f)));
    }
    let Some(f) = frame else {
        return (lines, true);
    };
    let no = |why: String| format!("no verdict at frame {f}: {why}");
    let verdict = if !(a.served && b.served) {
        Err(no(
            "a play's player does not serve every read class".to_owned()
        ))
    } else if !both.contains(&f) {
        Err(no("a play lacks the frame".to_owned()))
    } else if let Some((_, (key, x, y))) = unequal.iter().find(|(g, _)| *g == f) {
        Err(no(format!("counts differ ({key} a={x} b={y})")))
    } else if !digested.contains(&f) {
        Err(no("no digest there in both plays".to_owned()))
    } else if differing.contains(&f) {
        Err(no(format!("the digests differ{}", suspect(f))))
    } else {
        match (a.ends.get(&f), b.ends.get(&f)) {
            (Some(&(wall_a, busy_a)), Some(&(wall_b, busy_b))) => Ok(format!(
                "verdict at frame {f}: b is {} than a by GPU-busy time (a {busy_a} ns, b \
                 {busy_b} ns); wall a {wall_a} ns, b {wall_b} ns",
                faster(busy_a, busy_b)
            )),
            _ => Err(no("a play lacks the frame's end".to_owned())),
        }
    };
    let ok = verdict.is_ok();
    lines.push(match verdict {
        Ok(line) | Err(line) => line,
    });
    (lines, ok)
}

/// How `b` compares with `a`: "N.N% faster", "N.N% slower" or "level".
fn faster(a: u64, b: u64) -> String {
    if a == 0 || a == b {
        return "level".to_owned();
    }
    let percent = (a.abs_diff(b) as f64) * 100.0 / (a as f64);
    match b < a {
        true => format!("{percent:.1}% faster"),
        false => format!("{percent:.1}% slower"),
    }
}
