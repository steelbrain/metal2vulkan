//! The census's selection, cache, classification, and comparison over owned synthetic cases in a
//! scratch corpus, compiled by a scripted compiler. Real compiles through a compile worker are
//! `tests/authored_compile_census.rs`'s.

use super::*;
use crate::case::AuthoredCase;
use crate::engine::{FailureKind, Outcome};
use crate::hash::sha256_bytes;
use crate::store::CorpusStore;
use crate::worker::{ProcessFailure, WorkerFailure};
use crate::ScratchDir;
use base64::Engine as _;
use std::sync::Mutex;

/// A kernel case over a synthetic module, named `name` (its AIR's hash is the name's): its
/// sources are never read, since the scripted compiler admits nothing.
fn case(name: &str) -> AuthoredCase {
    let bytes = vec![0u8; 16];
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    let mut case: AuthoredCase = serde_json::from_value(serde_json::json!({
        "air_sha256": sha256_bytes(name.as_bytes()),
        "case_id": "pending",
        "name": name,
        "entry": "census_kernel",
        "stage": "kernel",
        "buffers": [{"binding": 0, "role": "output", "initial_bytes_b64": b64}],
        "dispatch": {"grid": [4, 1, 1], "threads_per_threadgroup": [4, 1, 1]},
        "output": {"kind": "buffer", "binding": 0, "offset": 0, "length": bytes.len()},
        "compare": {"kind": "exact"},
        "execution_safety": "loop_free",
    }))
    .unwrap();
    case.case_id = case.computed_case_id().unwrap();
    case
}

/// A scratch corpus holding `names`' cases, and an index and expectation file beside it.
struct Corpus {
    scratch: ScratchDir,
}

impl Corpus {
    fn new(names: &[&str]) -> Corpus {
        let scratch = ScratchDir::new("authored-compile-census").unwrap();
        let corpus = Corpus { scratch };
        for name in names {
            corpus.store().put_case(case(name)).unwrap();
        }
        corpus
    }

    fn root(&self) -> &Path {
        self.scratch.path()
    }

    fn store(&self) -> CorpusStore {
        CorpusStore::new(self.root())
    }

    fn index(&self) -> PathBuf {
        self.root().join(".index.sqlite")
    }

    fn expectations(&self) -> PathBuf {
        expectations_path(self.root())
    }

    /// Run the census with `compiler`, on `profiles` when given (a bless), over the corpus's
    /// index, or `index` when given.
    fn run(
        &self,
        compiler: &Scripted,
        profiles: Option<&[&str]>,
        bless: bool,
        index: Option<&Path>,
    ) -> Result<Report, String> {
        self.probe(compiler, profiles, None, bless, index)
    }

    /// [`Corpus::run`], a probe of `cases` when given.
    fn probe(
        &self,
        compiler: &Scripted,
        profiles: Option<&[&str]>,
        cases: Option<&[&str]>,
        bless: bool,
        index: Option<&Path>,
    ) -> Result<Report, String> {
        let mode = if bless { Mode::Bless } else { Mode::Check };
        self.execute(compiler, profiles, cases, mode, &[], index)
    }

    /// An additions bless ([`Mode::BlessAdditions`]) that adds `required`'s missing profile-legs.
    fn additions(
        &self,
        compiler: &Scripted,
        required: &[&str],
        index: Option<&Path>,
    ) -> Result<Report, String> {
        self.execute(compiler, None, None, Mode::BlessAdditions, required, index)
    }

    fn execute(
        &self,
        compiler: &Scripted,
        profiles: Option<&[&str]>,
        cases: Option<&[&str]>,
        mode: Mode,
        required: &[&str],
        index: Option<&Path>,
    ) -> Result<Report, String> {
        let parse = |labels: &[&str]| {
            labels
                .iter()
                .map(|label| ProfileLeg::parse(label).unwrap())
                .collect::<Vec<_>>()
        };
        let default_index = self.index();
        let expectations = self.expectations();
        run(
            &Options {
                root: self.root(),
                index: index.unwrap_or(&default_index),
                expectations: &expectations,
                profiles: profiles.map(parse),
                cases: cases.map(|ids| ids.iter().map(|id| id.to_string()).collect()),
                jobs: 3,
                mode,
                required: parse(required),
                // The first profile-leg keeps its cells' artifact digests, as a venue's does.
                artifacts: parse(&LEGS[..1]),
            },
            compiler,
        )
    }
}

/// Two committed profile-legs the census compiles for.
const LEGS: [&str; 2] = ["lavapipe", "lavapipe+robustness-off"];

/// A compiler that compiles nothing: each cell is its scripted outcome (compiled when it has
/// none), and every cell it is asked for is recorded.
#[derive(Default)]
struct Scripted {
    outcomes: Mutex<BTreeMap<(String, String), Result<CellOutcome, CellFailure>>>,
    host_replayed: Mutex<BTreeMap<String, usize>>,
    artifacts: Mutex<BTreeMap<(String, String), String>>,
    asked: Mutex<Vec<(String, String)>>,
}

impl Scripted {
    /// Script the artifact digest a compile of `name` on `label` reports.
    fn digest(&self, name: &str, label: &str, digest: &str) {
        self.artifacts
            .lock()
            .unwrap()
            .insert((name.into(), label.into()), digest.into());
    }

    fn set(&self, name: &str, label: &str, outcome: Result<CellOutcome, CellFailure>) {
        self.outcomes
            .lock()
            .unwrap()
            .insert((name.into(), label.into()), outcome);
    }

    /// The cells asked for since the last call, sorted.
    fn take_asked(&self) -> Vec<(String, String)> {
        let mut asked = self.asked.lock().unwrap().drain(..).collect::<Vec<_>>();
        asked.sort();
        asked
    }
}

impl CaseCompiler for Scripted {
    fn compile(&self, case: &AuthoredCase, profiles: &[&CensusProfile]) -> CaseResult {
        let cells = profiles
            .iter()
            .map(|profile| {
                let label = profile.leg.label();
                self.asked
                    .lock()
                    .unwrap()
                    .push((case.name.clone(), label.clone()));
                let artifact = self
                    .artifacts
                    .lock()
                    .unwrap()
                    .get(&(case.name.clone(), label.clone()))
                    .cloned();
                CellResult {
                    outcome: self
                        .outcomes
                        .lock()
                        .unwrap()
                        .get(&(case.name.clone(), label))
                        .cloned()
                        .unwrap_or(Ok(CellOutcome::Compiled)),
                    host_replayed: self
                        .host_replayed
                        .lock()
                        .unwrap()
                        .get(&case.name)
                        .copied()
                        .unwrap_or(0),
                    compiles: 1,
                    cost: Cost {
                        elapsed: Duration::from_millis(7),
                        slowest: Duration::from_millis(7),
                        peak_resident_bytes: Some(1 << 20),
                        allocator_peak_bytes: Some(1 << 19),
                    },
                    artifact,
                }
            })
            .collect();
        CaseResult {
            cells,
            worker: Some(Measured {
                elapsed: Duration::from_millis(9),
                peak_resident_bytes: Some(3 << 20),
                allocator_peak_bytes: Some(1 << 21),
            }),
            sources: Some(SourceReads {
                by_range: 1,
                by_shard: 0,
            }),
        }
    }
}

fn every(names: &[&str]) -> Vec<(String, String)> {
    let mut out = names
        .iter()
        .flat_map(|name| LEGS.map(|label| (name.to_string(), label.to_string())))
        .collect::<Vec<_>>();
    out.sort();
    out
}

fn refused(label: &str) -> Result<CellOutcome, CellFailure> {
    Ok(CellOutcome::Refused(label.into()))
}

fn internal() -> Result<CellOutcome, CellFailure> {
    Err(CellFailure {
        class: FailureClass::Internal,
        key: "internal".into(),
        detail: "Invariant: ValueId(30) has no value in this block".into(),
    })
}

/// A bless compiles every cell once and writes the expectations; a rerun with nothing changed
/// admits and compiles nothing, and passes.
#[test]
fn a_warm_rerun_compiles_nothing() {
    let corpus = Corpus::new(&["alpha", "beta", "gamma"]);
    let compiler = Scripted::default();
    compiler.set(
        "beta",
        LEGS[1],
        refused("not_supported{executor_not_implemented}"),
    );
    let blessed = corpus.run(&compiler, Some(&LEGS), true, None).unwrap();
    assert!(
        blessed.blessed && blessed.passed(),
        "{:#?}",
        blessed.lines()
    );
    assert_eq!(compiler.take_asked(), every(&["alpha", "beta", "gamma"]));
    assert_eq!((blessed.compiled, blessed.refused), (5, 1));
    let written = Expectations::load(&corpus.expectations()).unwrap().unwrap();
    assert_eq!(written.cases.len(), 3);
    assert_eq!(
        written
            .profiles
            .iter()
            .map(ProfileLeg::label)
            .collect::<Vec<_>>(),
        LEGS
    );

    let warm = corpus.run(&compiler, None, false, None).unwrap();
    assert!(warm.passed(), "{:#?}", warm.lines());
    assert!(compiler.take_asked().is_empty());
    assert_eq!(
        (warm.selected_cases, warm.selected_cells, warm.compiles),
        (0, 0, 0)
    );
    assert_eq!(warm.cached_cells, 6);
    assert_eq!((warm.compiled, warm.refused), (5, 1));
    assert!(warm.differences.is_empty());
}

/// An internal error fails the run, is never cached, and is never blessed: a bless with it
/// leaves the expectations as they were, and the next run compiles that cell again.
#[test]
fn an_internal_error_fails_and_is_neither_cached_nor_blessed() {
    let corpus = Corpus::new(&["alpha", "beta"]);
    let compiler = Scripted::default();
    corpus.run(&compiler, Some(&LEGS), true, None).unwrap();
    let before = std::fs::read_to_string(corpus.expectations()).unwrap();
    compiler.take_asked();

    // A new product: a fresh cache, and one cell that fails.
    let fresh = corpus.root().join("fresh.sqlite");
    compiler.set("alpha", LEGS[0], internal());
    let failed = corpus.run(&compiler, None, true, Some(&fresh)).unwrap();
    assert!(!failed.passed() && !failed.blessed);
    assert_eq!(failed.failures.len(), 1);
    assert_eq!(failed.failures[0].failure.class, FailureClass::Internal);
    assert_eq!(failed.failures[0].profile, LEGS[0]);
    assert!(failed.differences.is_empty(), "{:#?}", failed.differences);
    assert!(failed
        .lines()
        .iter()
        .any(|line| line.starts_with("authored-compile-failure\t")
            && line.contains("class=internal")));
    assert_eq!(
        std::fs::read_to_string(corpus.expectations()).unwrap(),
        before
    );
    compiler.take_asked();

    // Only the failed cell is compiled again.
    let again = corpus.run(&compiler, None, false, Some(&fresh)).unwrap();
    assert_eq!(
        compiler.take_asked(),
        vec![("alpha".to_string(), LEGS[0].to_string())]
    );
    assert_eq!(again.failures.len(), 1);
}

/// A worker stop fails the run like an internal error; its class is the worker boundary's.
#[test]
fn a_worker_stop_fails() {
    let corpus = Corpus::new(&["alpha"]);
    let compiler = Scripted::default();
    corpus.run(&compiler, Some(&LEGS), true, None).unwrap();
    let stop = WorkerFailure::Process(
        ProcessFailure::DeadlineExceeded,
        None,
        crate::worker::Measured {
            elapsed: Duration::from_secs(20),
            peak_resident_bytes: None,
            allocator_peak_bytes: None,
        },
    );
    let outcome = compile_outcome(CompileFailure::Stopped(stop));
    let failure = outcome.clone().unwrap_err();
    assert_eq!(failure.class, FailureClass::WorkerStop);
    assert_eq!(
        failure.key,
        Outcome::Failed {
            failure: FailureKind::Timeout
        }
        .key()
    );
    compiler.set("alpha", LEGS[1], outcome);
    let fresh = corpus.root().join("fresh.sqlite");
    let report = corpus.run(&compiler, None, false, Some(&fresh)).unwrap();
    assert!(!report.passed());
    assert_eq!(report.failures.len(), 1);
    assert_eq!(report.failures[0].failure.class, FailureClass::WorkerStop);
}

/// The product's typed outcomes classify as the executor's: a defect fails, any other refuses
/// with the executor's label.
#[test]
fn product_outcomes_classify_as_the_executor_classifies_them() {
    let internal = compile_outcome(CompileFailure::NotCompiled(NotCompiled::Product(
        Outcome::Failed {
            failure: FailureKind::Internal,
        },
        "Invariant".into(),
    )));
    assert_eq!(internal.unwrap_err().class, FailureClass::Internal);
    let typed = Outcome::Failed {
        failure: FailureKind::Cancelled,
    };
    assert!(!typed.is_defect());
    let label = NotSupported::Product {
        outcome: typed.clone(),
        detail: String::new(),
        at_bind: false,
    }
    .outcome();
    assert_eq!(
        compile_outcome(CompileFailure::NotCompiled(NotCompiled::Product(
            typed,
            "detail".into()
        ))),
        Ok(CellOutcome::Refused(label))
    );
    assert_eq!(
        executor_outcome(
            ExecutorError::Failed("does not check".into()),
            FailureClass::Case
        )
        .unwrap_err()
        .class,
        FailureClass::Case
    );
}

/// A case that compiled before and is refused now is a regression: the run fails until a bless.
#[test]
fn a_compiled_case_that_is_refused_now_fails_as_a_regression() {
    let corpus = Corpus::new(&["alpha", "beta"]);
    let compiler = Scripted::default();
    corpus.run(&compiler, Some(&LEGS), true, None).unwrap();

    let fresh = corpus.root().join("fresh.sqlite");
    let label = "not_supported{product:unsupported/feature}";
    compiler.set("beta", LEGS[1], refused(label));
    let report = corpus.run(&compiler, None, false, Some(&fresh)).unwrap();
    assert!(!report.passed());
    assert!(report.failures.is_empty());
    let beta = case("beta").case_id;
    assert_eq!(
        report.differences,
        vec![Difference::Changed {
            case_id: beta.clone(),
            profile: LEGS[1].into(),
            was: Cell::covered(CellOutcome::Compiled),
            now: Cell::covered(CellOutcome::Refused(label.into())),
        }]
    );
    assert!(report.differences[0].is_regression());
    assert!(report
        .lines()
        .iter()
        .any(|line| line.starts_with(&format!("authored-compile-difference\tregressed\t{beta}"))));

    // A bless records it and prints the same difference; the next run passes.
    let blessed = corpus.run(&compiler, None, true, Some(&fresh)).unwrap();
    assert!(blessed.blessed && blessed.passed());
    assert_eq!(blessed.differences, report.differences);
    assert!(corpus
        .run(&compiler, None, false, Some(&fresh))
        .unwrap()
        .passed());
}

/// A case added since the bless and a case removed since are differences; only the new case
/// compiles.
#[test]
fn new_and_removed_cases_are_differences() {
    let corpus = Corpus::new(&["alpha", "beta"]);
    let compiler = Scripted::default();
    corpus.run(&compiler, Some(&LEGS), true, None).unwrap();
    compiler.take_asked();

    let beta = case("beta");
    corpus
        .store()
        .delete_named_case(&beta.air_sha256, None, &beta.name)
        .unwrap();
    corpus.store().put_case(case("delta")).unwrap();
    let report = corpus.run(&compiler, None, false, None).unwrap();
    assert!(!report.passed());
    assert_eq!(compiler.take_asked(), every(&["delta"]));
    let mut expected = vec![
        Difference::NewCase {
            case_id: case("delta").case_id,
        },
        Difference::RemovedCase {
            case_id: beta.case_id,
        },
    ];
    expected.sort();
    let mut found = report.differences.clone();
    found.sort();
    assert_eq!(found, expected);
}

/// Host-replayed command buffers are reported as uncovered, from a compile and from the cache,
/// and a bless records each cell's count in the expectation file.
#[test]
fn host_replayed_command_buffers_are_reported_uncovered() {
    let corpus = Corpus::new(&["alpha", "beta"]);
    let compiler = Scripted::default();
    compiler
        .host_replayed
        .lock()
        .unwrap()
        .insert("beta".into(), 2);
    let cold = corpus.run(&compiler, Some(&LEGS), true, None).unwrap();
    let warm = corpus.run(&compiler, None, false, None).unwrap();
    assert!(warm.passed(), "{:#?}", warm.lines());
    let beta = case("beta").case_id;
    for report in [&cold, &warm] {
        assert_eq!(
            report.uncovered,
            LEGS.map(|label| (beta.clone(), label.to_string(), 2))
                .to_vec()
        );
    }
    let text = std::fs::read_to_string(corpus.expectations()).unwrap();
    assert!(
        text.contains(&format!(
            "\n{beta}\tcompiled uncovered=2\tcompiled uncovered=2\n"
        )),
        "{text}"
    );
    assert_eq!(
        Expectations::parse(&text).unwrap().cases,
        BTreeMap::from([
            (
                case("alpha").case_id,
                vec![Cell::covered(CellOutcome::Compiled); 2]
            ),
            (
                beta,
                vec![
                    Cell {
                        outcome: CellOutcome::Compiled,
                        uncovered: 2,
                        artifact: None,
                    };
                    2
                ]
            ),
        ])
    );
}

/// An uncovered count that changes is a difference like a changed outcome, in either direction:
/// a check fails on it, an additions bless refuses it and writes nothing, and only a bless records
/// it. An additions bless records the counts of the cells it adds.
#[test]
fn a_changed_uncovered_count_fails_until_a_bless_records_it() {
    let corpus = Corpus::new(&["alpha", "beta"]);
    let compiler = Scripted::default();
    compiler.set("beta", LEGS[1], refused("not_supported{bind}"));
    compiler
        .host_replayed
        .lock()
        .unwrap()
        .insert("beta".into(), 1);
    corpus.run(&compiler, Some(&LEGS), true, None).unwrap();
    let before = std::fs::read_to_string(corpus.expectations()).unwrap();

    // The product now host-replays a command buffer of alpha's, and one more of beta's.
    compiler
        .host_replayed
        .lock()
        .unwrap()
        .extend([("alpha".to_string(), 1), ("beta".to_string(), 2)]);
    let fresh = corpus.root().join("fresh.sqlite");
    let check = corpus.run(&compiler, None, false, Some(&fresh)).unwrap();
    assert!(!check.passed() && check.failures.is_empty());
    let (alpha, beta) = (case("alpha").case_id, case("beta").case_id);
    let cell = |outcome: Result<CellOutcome, CellFailure>, uncovered| Cell {
        outcome: outcome.unwrap(),
        uncovered,
        artifact: None,
    };
    let compiled = || Ok(CellOutcome::Compiled);
    let mut wanted = Vec::new();
    for (case_id, was, now) in [(&alpha, 0, 1), (&beta, 1, 2)] {
        for label in LEGS {
            let outcome = if case_id == &beta && label == LEGS[1] {
                refused("not_supported{bind}")
            } else {
                compiled()
            };
            wanted.push(Difference::Changed {
                case_id: case_id.clone(),
                profile: label.into(),
                was: cell(outcome.clone(), was),
                now: cell(outcome, now),
            });
        }
    }
    wanted.sort();
    let mut found = check.differences.clone();
    found.sort();
    assert_eq!(found, wanted);
    assert!(found.iter().all(|difference| !difference.is_regression()));
    let lines = check.lines();
    assert!(
        lines.contains(&format!(
            "authored-compile-difference\tuncovered-changed\t{alpha}\tprofile={}\twas=compiled\tnow=compiled uncovered=1",
            LEGS[0]
        )),
        "{lines:#?}"
    );
    assert!(
        lines.contains(&format!(
            "authored-compile-difference\tuncovered-changed\t{beta}\tprofile={}\twas=not_supported{{bind}} uncovered=1\tnow=not_supported{{bind}} uncovered=2",
            LEGS[1]
        )),
        "{lines:#?}"
    );

    // An additions bless refuses the changed counts, even beside a new case, and writes nothing.
    corpus.store().put_case(case("delta")).unwrap();
    compiler
        .host_replayed
        .lock()
        .unwrap()
        .insert("delta".into(), 3);
    let additions = corpus.additions(&compiler, &[], Some(&fresh)).unwrap();
    assert!(!additions.passed() && !additions.blessed && !additions.written);
    assert!(additions
        .lines()
        .iter()
        .any(|line| line.ends_with("\tchanged_cells=4\tblessed=false\twritten=false")));
    assert_eq!(
        std::fs::read_to_string(corpus.expectations()).unwrap(),
        before
    );

    // With the counts back as blessed, the additions bless writes the new case with its count.
    compiler
        .host_replayed
        .lock()
        .unwrap()
        .extend([("alpha".to_string(), 0), ("beta".to_string(), 1)]);
    let other = corpus.root().join("other.sqlite");
    let added = corpus.additions(&compiler, &[], Some(&other)).unwrap();
    assert!(added.passed() && added.written, "{:#?}", added.lines());
    let delta = case("delta").case_id;
    assert_eq!(
        Expectations::load(&corpus.expectations())
            .unwrap()
            .unwrap()
            .cases[&delta],
        vec![cell(compiled(), 3); 2]
    );

    // A bless records the changed counts, and the next check passes.
    compiler
        .host_replayed
        .lock()
        .unwrap()
        .extend([("alpha".to_string(), 1), ("beta".to_string(), 2)]);
    let blessed = corpus.run(&compiler, None, true, Some(&fresh)).unwrap();
    assert!(blessed.blessed && blessed.written, "{:#?}", blessed.lines());
    let check = corpus.run(&compiler, None, false, Some(&fresh)).unwrap();
    assert!(check.passed(), "{:#?}", check.lines());
}

/// A check needs the expectation file and compiles its profile-legs; only a bless names others.
#[test]
fn a_check_compiles_the_expectation_files_profiles() {
    let corpus = Corpus::new(&["alpha"]);
    let compiler = Scripted::default();
    let missing = corpus.run(&compiler, None, false, None).unwrap_err();
    assert!(missing.contains("bless it first"), "{missing}");
    let named = corpus.run(&compiler, Some(&LEGS), false, None).unwrap_err();
    assert!(named.contains("--profile"), "{named}");
    corpus.run(&compiler, Some(&LEGS[..1]), true, None).unwrap();
    // A bless that adds a profile-leg prints it and compiles only its cells.
    compiler.take_asked();
    let widened = corpus.run(&compiler, Some(&LEGS), true, None).unwrap();
    assert_eq!(
        widened.differences,
        vec![Difference::AddedProfile(LEGS[1].into())]
    );
    assert_eq!(
        compiler.take_asked(),
        vec![("alpha".into(), LEGS[1].into())]
    );
}

/// The expectation file round-trips, and a malformed one is refused with its line.
#[test]
fn the_expectation_file_round_trips_and_refuses_malformed_lines() {
    let expectations = Expectations {
        profiles: LEGS.map(|label| ProfileLeg::parse(label).unwrap()).to_vec(),
        cases: BTreeMap::from([
            (
                "a".repeat(64),
                vec![
                    Cell::covered(CellOutcome::Compiled),
                    Cell::covered(CellOutcome::Refused("not_supported{bind}".into())),
                ],
            ),
            (
                "b".repeat(64),
                vec![
                    Cell {
                        outcome: CellOutcome::Compiled,
                        uncovered: 1,
                        artifact: None,
                    },
                    Cell {
                        outcome: CellOutcome::Refused("not_supported{a label} with spaces".into()),
                        uncovered: 12,
                        artifact: None,
                    },
                ],
            ),
            (
                "c".repeat(64),
                vec![Cell::covered(CellOutcome::Compiled); 2],
            ),
        ]),
        build: None,
    };
    let text = expectations.render().unwrap();
    assert!(
        text.contains(&format!(
            "\n{}\tcompiled uncovered=1\tnot_supported{{a label}} with spaces uncovered=12\n",
            "b".repeat(64)
        )),
        "{text}"
    );
    assert_eq!(Expectations::parse(&text).unwrap(), expectations);
    let header = format!("case_id\t{}\t{}\n", LEGS[0], LEGS[1]);
    let uncovered = |cell: &str| format!("{}\tcompiled\t{cell}\n", "a".repeat(64));
    for (body, why) in [
        (format!("{}\tcompiled\n", "a".repeat(64)), "1 cells"),
        (uncovered("compiled uncovered=0"), "no uncovered count"),
        (uncovered("compiled uncovered=01"), "no uncovered count"),
        (uncovered("compiled uncovered=x"), "no uncovered count"),
        (uncovered("compiled uncovered="), "no uncovered count"),
        (uncovered("uncovered=2"), "not an outcome"),
        (uncovered(" uncovered=2"), "not an outcome"),
        (
            format!(
                "{0}\tcompiled\tcompiled\n{0}\tcompiled\tcompiled\n",
                "a".repeat(64)
            ),
            "twice",
        ),
        ("nothex\tcompiled\tcompiled\n".to_string(), "case_id"),
    ] {
        let error = Expectations::parse(&format!("{header}{body}")).unwrap_err();
        assert!(error.contains(why), "{error}");
    }
    assert!(Expectations::parse("case_id\tlavapipe\tlavapipe\n")
        .unwrap_err()
        .contains("twice"));
    assert_eq!(
        ProfileLeg::parse("lavapipe+robustness-off").unwrap(),
        ProfileLeg {
            stem: "lavapipe".into(),
            leg: Leg::RobustnessOff
        }
    );
    assert!(ProfileLeg::parse("lava pipe").is_err());
}

/// The key of a profile-leg moves with the product's census key, the executor ABI, and the
/// census's version, and differs between a profile's legs.
#[test]
fn each_profile_leg_has_its_own_key() {
    let stated = CensusProfile::load(&ProfileLeg::parse(LEGS[0]).unwrap()).unwrap();
    let off = CensusProfile::load(&ProfileLeg::parse(LEGS[1]).unwrap()).unwrap();
    assert_ne!(stated.key, off.key);
    let census = CensusKey::engine(LEGS[0], stated.device.device().profile());
    assert_eq!(stated.key, profile_key(&census));
    let other_product = CensusKey {
        product: "another product".into(),
        ..census
    };
    assert_ne!(profile_key(&other_product), stated.key);
}

/// A probe compiles only its cases, reports each cell, compares only their rows (nothing when no
/// expectation file exists), reads the cache like any run, and never blesses; an unknown case is
/// refused.
#[test]
fn a_probe_compiles_only_its_cases_and_compares_their_rows() {
    let corpus = Corpus::new(&["alpha", "beta"]);
    let compiler = Scripted::default();
    let alpha = case("alpha").case_id;
    let beta = case("beta").case_id;

    let first = corpus
        .probe(&compiler, Some(&LEGS), Some(&[alpha.as_str()]), false, None)
        .unwrap();
    assert!(
        first.passed() && first.differences.is_empty(),
        "{:#?}",
        first.lines()
    );
    assert_eq!(compiler.take_asked(), every(&["alpha"]));
    assert_eq!((first.cases, first.case_workers), (1, 1));
    assert_eq!(
        first
            .probed
            .iter()
            .map(|cell| (cell.profile.as_str(), cell.compiled.is_some()))
            .collect::<Vec<_>>(),
        LEGS.map(|label| (label, true)).to_vec()
    );
    let line = format!(
        "authored-compile-cell\t{alpha}\tprofile={}\toutcome=compiled\tcached=false\tcompiles=1\t",
        LEGS[0]
    );
    assert!(
        first.lines().iter().any(|l| l.starts_with(&line)),
        "{:#?}",
        first.lines()
    );
    assert_eq!(
        first.peak_case_resident.as_ref().map(|m| m.value),
        Some(3 << 20)
    );

    // Blessed, then beta regresses on a new product: a probe of alpha passes and of beta fails.
    corpus.run(&compiler, Some(&LEGS), true, None).unwrap();
    assert_eq!(compiler.take_asked(), every(&["beta"]));
    compiler.set("beta", LEGS[0], refused("not_supported{bind}"));
    let fresh = corpus.root().join("fresh.sqlite");
    let probe = |id: &str| {
        corpus
            .probe(&compiler, None, Some(&[id]), false, Some(&fresh))
            .unwrap()
    };
    let passing = probe(&alpha);
    assert!(passing.passed(), "{:#?}", passing.lines());
    assert_eq!(compiler.take_asked(), every(&["alpha"]));
    let failing = probe(&beta);
    assert!(!failing.passed());
    assert_eq!(failing.differences.len(), 1, "{:#?}", failing.differences);
    assert!(failing.differences[0].is_regression());
    compiler.take_asked();
    let cached = probe(&alpha);
    assert!(compiler.take_asked().is_empty());
    assert!(cached.probed.iter().all(|cell| cell.compiled.is_none()));
    assert!(cached.lines().iter().any(|l| l.ends_with("\tcached=true")));

    let bless = corpus
        .probe(&compiler, None, Some(&[alpha.as_str()]), true, None)
        .unwrap_err();
    assert!(bless.contains("--case"), "{bless}");
    let unknown = corpus
        .probe(
            &compiler,
            None,
            Some(&["f".repeat(64).as_str()]),
            false,
            None,
        )
        .unwrap_err();
    assert!(unknown.contains("no committed case"), "{unknown}");
}

/// A case worker that stops (here one that exits at once with no report) fails every cell of
/// its case with the worker boundary's class, and nothing is cached or blessed.
#[cfg(unix)]
#[test]
fn a_stopped_case_worker_fails_every_cell_of_its_case() {
    let corpus = Corpus::new(&["alpha"]);
    let compiler = CaseWorkers {
        root: corpus.root().to_path_buf(),
        index: corpus.index(),
        program: PathBuf::from("true"),
    };
    let index = corpus.index();
    let expectations = corpus.expectations();
    let report = run(
        &Options {
            root: corpus.root(),
            index: &index,
            expectations: &expectations,
            profiles: Some(LEGS.map(|label| ProfileLeg::parse(label).unwrap()).to_vec()),
            cases: None,
            jobs: 1,
            mode: Mode::Bless,
            required: Vec::new(),
            artifacts: Vec::new(),
        },
        &compiler,
    )
    .unwrap();
    assert!(!report.passed() && !report.blessed);
    assert_eq!(report.failures.len(), LEGS.len(), "{:#?}", report.lines());
    assert!(report.failures.iter().all(|failed| {
        failed.failure.class == FailureClass::WorkerStop
            && failed.failure.detail.starts_with("the case worker: ")
    }));
    assert_eq!(report.case_workers, 1);
    assert!(!expectations.exists());
}

/// Every kind of outcome the census can bless, built by its own classification from the
/// executor's and the product's typed refusals (labels with spaces, braces, parentheses, and
/// nested detail), is written by a bless and read back as itself: from the file, by a warm run
/// from the cache, and by a cold run compared with the file. A label no cell can hold is a tool
/// failure, and the writer refuses one.
#[test]
fn every_outcome_kind_round_trips_through_a_bless() {
    use crate::case::out_of_bounds::ContainmentForm;
    use crate::check::admission::ExecutionRefused;
    use crate::coverage::RobustBufferForm;
    use crate::engine::{LimitKind, UnsupportedReason};
    use crate::requirement::ToolingRequirement;
    let product = |outcome: Outcome| {
        compile_outcome(CompileFailure::NotCompiled(NotCompiled::Product(
            outcome,
            "a detail\twith a tab\nand a line".into(),
        )))
    };
    let missing = |feature: &str, capability: &str| {
        product(Outcome::DeviceCapabilityMissing {
            feature: feature.into(),
            capability: capability.into(),
        })
    };
    let unsupported = |reason, feature: &str| {
        product(Outcome::Unsupported {
            reason,
            feature: feature.into(),
        })
    };
    let executor = |not_supported| {
        executor_outcome(
            ExecutorError::NotSupported(not_supported),
            FailureClass::Case,
        )
    };
    let outcomes = [
        Ok(CellOutcome::Compiled),
        missing("LayeredRendering", "Feature(ShaderOutputViewportIndex)"),
        missing(
            "Format",
            "Format { format: VK_FORMAT_R16_UINT, usage: StorageImage }",
        ),
        missing(
            "InterfaceComponents",
            "Limit { limit: MaxFragmentOutputAttachments, required: 5, available: 4 }",
        ),
        missing(
            "PrimitiveId",
            "Feature(GeometryShader) | Feature(TessellationShader)",
        ),
        missing(
            "RayTracing",
            "Feature(RayQuery)+Feature(AccelerationStructure)",
        ),
        unsupported(UnsupportedReason::NotImplemented, "Linking"),
        unsupported(UnsupportedReason::NotRepresentable, "Lowering(Texture)"),
        product(Outcome::NeedsInput {
            kind: "function_constant".into(),
        }),
        product(Outcome::InvalidInput {
            kind: "bitcode".into(),
        }),
        product(Outcome::InvalidDescriptor {
            kind: "FunctionConstantIndex".into(),
        }),
        product(Outcome::LimitExceeded {
            kind: LimitKind::Structural,
            limit: "words 4194304".into(),
        }),
        executor(NotSupported::Bind("a range above the limit".into())),
        executor(NotSupported::NotEnabled(vec!["shaderInt64".into()])),
        executor(NotSupported::NotImplemented("textures".into())),
        executor(NotSupported::Requirements(BTreeSet::from([
            ToolingRequirement::FunctionConstantLiteral,
            ToolingRequirement::VertexAttributeLiteral,
        ]))),
        executor(NotSupported::Uncontained {
            forms: BTreeSet::from([RobustBufferForm::LoadStore, RobustBufferForm::Atomic]),
            unchecked: BTreeSet::new(),
            variant: None,
        }),
        executor(NotSupported::Uncontained {
            forms: BTreeSet::new(),
            unchecked: BTreeSet::from([ContainmentForm::LoadedPointer]),
            variant: None,
        }),
        executor_outcome(
            ExecutorError::Refused(ExecutionRefused::NotFuelBounded(
                crate::check::loop_free::CycleEvidence::CfgCycle {
                    function: "main".into(),
                },
            )),
            FailureClass::Case,
        ),
    ];
    for outcome in &outcomes {
        assert!(outcome.is_ok(), "{outcome:?}");
    }
    assert!(outcomes
        .iter()
        .any(|outcome| matches!(outcome, Ok(CellOutcome::Refused(label)) if label.contains(' '))));

    // Each case has one outcome on the first profile-leg and the next on the second.
    let names = (0..outcomes.len())
        .map(|index| format!("outcome-{index}"))
        .collect::<Vec<_>>();
    let corpus = Corpus::new(&names.iter().map(String::as_str).collect::<Vec<_>>());
    let compiler = Scripted::default();
    let mut wanted = BTreeMap::new();
    for (index, name) in names.iter().enumerate() {
        let cells = [&outcomes[index], &outcomes[(index + 1) % outcomes.len()]];
        for (label, outcome) in LEGS.iter().zip(cells) {
            compiler.set(name, label, outcome.clone());
        }
        wanted.insert(
            case(name).case_id,
            cells
                .map(|cell| Cell::covered(cell.clone().unwrap()))
                .to_vec(),
        );
    }
    let blessed = corpus.run(&compiler, Some(&LEGS), true, None).unwrap();
    assert!(
        blessed.blessed && blessed.passed(),
        "{:#?}",
        blessed.lines()
    );
    let text = std::fs::read_to_string(corpus.expectations()).unwrap();
    let read = Expectations::parse(&text).unwrap();
    assert_eq!(read.cases, wanted);
    assert_eq!(read.render().unwrap(), text);

    let warm = corpus.run(&compiler, None, false, None).unwrap();
    assert!(warm.passed(), "{:#?}", warm.lines());
    assert_eq!((warm.cached_cells, warm.compiles), (2 * outcomes.len(), 0));
    let fresh = corpus.root().join("fresh.sqlite");
    let cold = corpus.run(&compiler, None, false, Some(&fresh)).unwrap();
    assert!(
        cold.passed() && cold.differences.is_empty(),
        "{:#?}",
        cold.lines()
    );
    assert_eq!(cold.compiles, 2 * outcomes.len());

    // A label no cell can hold never reaches the file.
    for label in [
        "a\tb",
        "a\nb",
        " padded",
        "",
        "compiled",
        "uncovered=1",
        "a uncovered=1",
        "a uncovered=x",
    ] {
        let failure = refusal(label.into()).unwrap_err();
        assert_eq!(
            (failure.class, failure.key.as_str()),
            (FailureClass::Tool, "refusal_label"),
            "{label:?}"
        );
        let unholdable = Expectations {
            profiles: vec![ProfileLeg::parse(LEGS[0]).unwrap()],
            cases: BTreeMap::from([(
                "a".repeat(64),
                vec![Cell::covered(CellOutcome::Refused(label.into()))],
            )]),
            build: None,
        };
        assert!(unholdable.render().is_err(), "{label:?}");
    }
}

/// The committed expectation file parses (a header of profile-legs and a row per case with a cell
/// on each) and is what a bless renders. Which cases and profile-legs it lists is the gate's
/// additions bless's to keep current, never a test of the ordinary suite.
#[test]
fn the_committed_expectations_parse() {
    let path = expectations_path(&crate::source::corpus_root());
    let expected = Expectations::load(&path)
        .unwrap()
        .unwrap_or_else(|| panic!("{} is not committed", path.display()));
    assert!(!expected.profiles.is_empty() && !expected.cases.is_empty());
    // It is byte for byte what a bless writes, so an additions bless with nothing to add leaves it.
    assert_eq!(
        expected.render().unwrap(),
        std::fs::read_to_string(&path).unwrap()
    );
}

/// An additions bless writes a new case's row and drops a removed case's, compiling only the new
/// case; run again, it writes nothing.
#[test]
fn an_additions_bless_adds_new_cases_and_drops_removed_ones() {
    let corpus = Corpus::new(&["alpha", "beta"]);
    let compiler = Scripted::default();
    corpus.run(&compiler, Some(&LEGS), true, None).unwrap();
    compiler.take_asked();

    let beta = case("beta");
    corpus
        .store()
        .delete_named_case(&beta.air_sha256, None, &beta.name)
        .unwrap();
    corpus.store().put_case(case("gamma")).unwrap();
    compiler.set("gamma", LEGS[1], refused("not_supported{bind}"));
    let report = corpus.additions(&compiler, &[], None).unwrap();
    assert!(
        report.passed() && report.blessed && report.written,
        "{:#?}",
        report.lines()
    );
    assert_eq!(compiler.take_asked(), every(&["gamma"]));
    let mut found = report.differences.clone();
    found.sort();
    let mut wanted = vec![
        Difference::NewCase {
            case_id: case("gamma").case_id,
        },
        Difference::RemovedCase {
            case_id: beta.case_id,
        },
    ];
    wanted.sort();
    assert_eq!(found, wanted);
    assert!(report.lines().contains(
        &"authored-compile-additions\tadded_cases=1\tremoved_cases=1\tadded_profiles=none\tchanged_cells=0\tblessed=true\twritten=true"
            .to_string()
    ));
    let written = Expectations::load(&corpus.expectations()).unwrap().unwrap();
    assert_eq!(
        written.cases,
        BTreeMap::from([
            (
                case("alpha").case_id,
                vec![Cell::covered(CellOutcome::Compiled); 2]
            ),
            (
                case("gamma").case_id,
                vec![
                    Cell::covered(CellOutcome::Compiled),
                    Cell::covered(CellOutcome::Refused("not_supported{bind}".into()))
                ]
            ),
        ])
    );

    let again = corpus.additions(&compiler, &[], None).unwrap();
    assert!(again.passed() && again.blessed && !again.written);
    assert!(again.differences.is_empty() && compiler.take_asked().is_empty());
}

/// An additions bless fails on a changed cell the file holds, in either direction, and writes
/// nothing, not even the additions beside it; so does a failed cell.
#[test]
fn an_additions_bless_refuses_a_changed_cell_and_writes_nothing() {
    let corpus = Corpus::new(&["alpha", "beta"]);
    let compiler = Scripted::default();
    compiler.set("beta", LEGS[0], refused("not_supported{bind}"));
    corpus.run(&compiler, Some(&LEGS), true, None).unwrap();
    let before = std::fs::read_to_string(corpus.expectations()).unwrap();

    // A new product: alpha regresses, beta now compiles, and a new case and venue arrive.
    compiler.set("alpha", LEGS[1], refused("not_supported{not_enabled}"));
    compiler.set("beta", LEGS[0], Ok(CellOutcome::Compiled));
    corpus.store().put_case(case("delta")).unwrap();
    let fresh = corpus.root().join("fresh.sqlite");
    let report = corpus
        .additions(&compiler, &["runtime-floor"], Some(&fresh))
        .unwrap();
    assert!(!report.passed() && !report.blessed && !report.written);
    let changed = report
        .differences
        .iter()
        .filter(|difference| !difference.is_addition())
        .collect::<Vec<_>>();
    assert_eq!(changed.len(), 2, "{:#?}", report.differences);
    assert!(changed.iter().any(|difference| difference.is_regression()));
    assert!(report
        .lines()
        .iter()
        .any(|line| line.starts_with("authored-compile-difference\tregressed\t")));
    assert!(report
        .lines()
        .iter()
        .any(|line| line.ends_with("\tchanged_cells=2\tblessed=false\twritten=false")));
    assert_eq!(
        std::fs::read_to_string(corpus.expectations()).unwrap(),
        before
    );

    // Only additions, but a failed cell: nothing is written either.
    let compiler = Scripted::default();
    compiler.set("beta", LEGS[0], refused("not_supported{bind}"));
    compiler.set("delta", LEGS[0], internal());
    let other = corpus.root().join("other.sqlite");
    let failed = corpus.additions(&compiler, &[], Some(&other)).unwrap();
    assert!(!failed.passed() && !failed.blessed && !failed.written);
    assert_eq!(failed.failures.len(), 1);
    assert_eq!(
        std::fs::read_to_string(corpus.expectations()).unwrap(),
        before
    );
}

/// An additions bless that adds a new venue's profile-leg records each of its cells' uncovered
/// count, compiling only that leg's cells; the file's own cells keep theirs, and a check over the
/// written file, warm or cold, finds no difference.
#[test]
fn an_added_profile_leg_records_its_uncovered_counts() {
    let corpus = Corpus::new(&["alpha", "beta"]);
    let compiler = Scripted::default();
    compiler.set("beta", LEGS[1], refused("not_supported{bind}"));
    compiler
        .host_replayed
        .lock()
        .unwrap()
        .insert("beta".into(), 2);
    corpus.run(&compiler, Some(&LEGS[..1]), true, None).unwrap();
    compiler.take_asked();

    let added = corpus.additions(&compiler, &LEGS, None).unwrap();
    assert!(added.passed() && added.written, "{:#?}", added.lines());
    assert_eq!(
        added.differences,
        vec![Difference::AddedProfile(LEGS[1].into())]
    );
    assert_eq!(
        compiler.take_asked(),
        vec![
            ("alpha".to_string(), LEGS[1].to_string()),
            ("beta".to_string(), LEGS[1].to_string()),
        ]
    );
    let text = std::fs::read_to_string(corpus.expectations()).unwrap();
    let beta = case("beta").case_id;
    assert!(
        text.contains(&format!(
            "\n{beta}\tcompiled uncovered=2\tnot_supported{{bind}} uncovered=2\n"
        )),
        "{text}"
    );
    assert_eq!(
        Expectations::parse(&text).unwrap().cases,
        BTreeMap::from([
            (
                case("alpha").case_id,
                vec![Cell::covered(CellOutcome::Compiled); 2]
            ),
            (
                beta,
                vec![
                    Cell {
                        outcome: CellOutcome::Compiled,
                        uncovered: 2,
                        artifact: None,
                    },
                    Cell {
                        outcome: CellOutcome::Refused("not_supported{bind}".into()),
                        uncovered: 2,
                        artifact: None,
                    },
                ]
            ),
        ])
    );
    for index in [None, Some(corpus.root().join("fresh.sqlite"))] {
        let check = corpus
            .run(&compiler, None, false, index.as_deref())
            .unwrap();
        assert!(
            check.passed() && check.differences.is_empty(),
            "{:#?}",
            check.lines()
        );
    }
}

/// An additions bless adds a new venue's profile-legs the file lacks, compiling only their cells,
/// and keeps the file's own; it names no profile-leg itself, and needs the file.
#[test]
fn an_additions_bless_adds_a_new_venues_profile_legs() {
    let corpus = Corpus::new(&["alpha", "beta"]);
    let compiler = Scripted::default();
    let missing = corpus.additions(&compiler, &LEGS, None).unwrap_err();
    assert!(missing.contains("bless it first"), "{missing}");
    corpus.run(&compiler, Some(&LEGS[..1]), true, None).unwrap();
    compiler.take_asked();

    let report = corpus.additions(&compiler, &LEGS, None).unwrap();
    assert!(report.passed() && report.written, "{:#?}", report.lines());
    assert_eq!(
        report.differences,
        vec![Difference::AddedProfile(LEGS[1].into())]
    );
    assert_eq!(
        compiler.take_asked(),
        vec![
            ("alpha".to_string(), LEGS[1].to_string()),
            ("beta".to_string(), LEGS[1].to_string()),
        ]
    );
    assert!(report
        .lines()
        .iter()
        .any(|line| line.contains(&format!("\tadded_profiles={}\t", LEGS[1]))));
    let written = Expectations::load(&corpus.expectations()).unwrap().unwrap();
    assert_eq!(
        written
            .profiles
            .iter()
            .map(ProfileLeg::label)
            .collect::<Vec<_>>(),
        LEGS
    );
    // The file's profile-legs now include it: a check compiles both, and passes.
    let check = corpus.run(&compiler, None, false, None).unwrap();
    assert!(check.passed() && check.profiles == LEGS);

    let named = corpus
        .execute(
            &compiler,
            Some(&LEGS),
            None,
            Mode::BlessAdditions,
            &[],
            None,
        )
        .unwrap_err();
    assert!(named.contains("--bless-additions"), "{named}");
}

/// A profile-leg an additions bless adds goes after the file's own, wherever the venue list puts
/// it, and stays there: the written file is what a bless renders (the committed-file test's
/// check), and another additions bless over it writes nothing.
#[test]
fn an_added_profile_leg_keeps_its_place_in_later_runs() {
    let corpus = Corpus::new(&["alpha", "beta"]);
    let compiler = Scripted::default();
    // The file holds the second leg; the venue list names the first before it.
    corpus.run(&compiler, Some(&LEGS[1..]), true, None).unwrap();
    let added = corpus.additions(&compiler, &LEGS, None).unwrap();
    assert!(added.passed() && added.written, "{:#?}", added.lines());
    let text = std::fs::read_to_string(corpus.expectations()).unwrap();
    let written = Expectations::parse(&text).unwrap();
    assert_eq!(
        written
            .profiles
            .iter()
            .map(ProfileLeg::label)
            .collect::<Vec<_>>(),
        vec![LEGS[1], LEGS[0]]
    );
    assert_eq!(written.render().unwrap(), text);

    compiler.take_asked();
    for index in [None, Some(corpus.root().join("fresh.sqlite"))] {
        let again = corpus
            .additions(&compiler, &LEGS, index.as_deref())
            .unwrap();
        assert!(
            again.passed() && again.blessed && !again.written,
            "{:#?}",
            again.lines()
        );
        assert!(again.differences.is_empty());
        assert_eq!(again.profiles, vec![LEGS[1], LEGS[0]]);
    }
    assert_eq!(
        std::fs::read_to_string(corpus.expectations()).unwrap(),
        text
    );
}

/// A digest the scripted compiler reports: `byte` repeated.
fn digest(byte: char) -> String {
    byte.to_string().repeat(64)
}

/// A cell holds an artifact digest only after `compiled` with nothing uncovered; a refusal label
/// cannot end in a word a digest starts with; and a file with a digest names the build it is at.
#[test]
fn a_cell_holds_its_artifact_digest_at_a_named_build() {
    let digested = Cell {
        outcome: CellOutcome::Compiled,
        uncovered: 0,
        artifact: Some(digest('a')),
    };
    assert_eq!(
        digested.text(),
        format!("compiled artifact={}", digest('a'))
    );
    assert_eq!(Cell::parse(&digested.text()).unwrap(), digested);
    for (text, why) in [
        (
            format!("not_supported{{bind}} artifact={}", digest('a')),
            "no compiled case",
        ),
        ("compiled artifact=ab".to_string(), "64 hexadecimal"),
        (format!("compiled artifact={}", digest('A')), "lowercase"),
    ] {
        let error = Cell::parse(&text).unwrap_err();
        assert!(error.contains(why), "{text}: {error}");
    }
    assert!(CellOutcome::refused(format!("a label artifact={}", digest('a'))).is_err());
    // A digest beside an uncovered count, or on a refusal, is not a cell the file can hold.
    for cell in [
        Cell {
            uncovered: 1,
            ..digested.clone()
        },
        Cell {
            outcome: CellOutcome::Refused("not_supported{bind}".into()),
            ..digested.clone()
        },
    ] {
        let expectations = Expectations {
            build: Some(ArtifactBuild::this()),
            profiles: vec![ProfileLeg::parse(LEGS[0]).unwrap()],
            cases: BTreeMap::from([("a".repeat(64), vec![cell])]),
        };
        assert!(expectations.render().is_err());
    }

    let expectations = Expectations {
        build: Some(ArtifactBuild::this()),
        profiles: LEGS.map(|label| ProfileLeg::parse(label).unwrap()).to_vec(),
        cases: BTreeMap::from([(
            "a".repeat(64),
            vec![digested.clone(), Cell::covered(CellOutcome::Compiled)],
        )]),
    };
    let text = expectations.render().unwrap();
    assert!(
        text.contains(&format!(
            "\nartifacts_at\tproduct={}\texecutor={}\ncase_id\t",
            crate::observation::TRANSLATOR_FINGERPRINT,
            ENGINE_EXECUTOR_ABI
        )),
        "{text}"
    );
    assert_eq!(Expectations::parse(&text).unwrap(), expectations);
    // A digest needs its build: neither written nor read without it.
    let unbuilt = Expectations {
        build: None,
        ..expectations.clone()
    };
    assert!(unbuilt.render().is_err());
    let without_line = text
        .lines()
        .filter(|line| !line.starts_with("artifacts_at"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(Expectations::parse(&without_line)
        .unwrap_err()
        .contains("no `artifacts_at` line"));
    for line in [
        "artifacts_at\tproduct=ab\texecutor=v1",
        "artifacts_at\texecutor=v1",
        "artifacts_at\tproduct=",
    ] {
        let malformed = text.replacen(
            text.lines()
                .find(|l| l.starts_with("artifacts_at"))
                .unwrap(),
            line,
            1,
        );
        assert!(Expectations::parse(&malformed).is_err(), "{line}");
    }
}

/// The census keeps a digest only on the profile-legs that record them (a venue's), and writes
/// the build it computed them at.
#[test]
fn a_bless_records_digests_on_the_venues_profile_legs_only() {
    let corpus = Corpus::new(&["alpha", "beta"]);
    let compiler = Scripted::default();
    for name in ["alpha", "beta"] {
        for label in LEGS {
            compiler.digest(name, label, &digest('1'));
        }
    }
    compiler.set("beta", LEGS[0], refused("not_supported{bind}"));
    let blessed = corpus.run(&compiler, Some(&LEGS), true, None).unwrap();
    assert!(blessed.passed(), "{:#?}", blessed.lines());
    let written = Expectations::load(&corpus.expectations()).unwrap().unwrap();
    assert_eq!(written.build, Some(ArtifactBuild::this()));
    let alpha = &written.cases[&case("alpha").case_id];
    assert_eq!(
        alpha[0].artifact,
        Some(digest('1')),
        "the venue's leg keeps it"
    );
    assert_eq!(alpha[1].artifact, None, "another leg does not");
    let beta = &written.cases[&case("beta").case_id];
    assert_eq!(
        beta[0],
        Cell::covered(CellOutcome::Refused("not_supported{bind}".into()))
    );

    // Warm, the cache gives the digests back: no compile, no difference.
    let warm = corpus.run(&compiler, None, false, None).unwrap();
    assert!(warm.passed() && warm.compiles == 0, "{:#?}", warm.lines());
    assert!(warm.artifacts_at_this_build);
    assert_eq!(warm.moved_artifacts, 0);
}

/// At the build the file's digests were computed at, a moved digest is a difference (the census is
/// not deterministic, or the file was edited); at another, it is no difference, and an additions
/// bless records it with this build, while a changed outcome still refuses the bless.
#[test]
fn a_moved_digest_differs_only_at_the_files_own_build() {
    let corpus = Corpus::new(&["alpha", "beta"]);
    let compiler = Scripted::default();
    compiler.digest("alpha", LEGS[0], &digest('1'));
    compiler.digest("beta", LEGS[0], &digest('2'));
    corpus.run(&compiler, Some(&LEGS), true, None).unwrap();

    // Same build, another digest: a difference that fails the check and refuses an additions bless.
    compiler.digest("alpha", LEGS[0], &digest('3'));
    let cold = corpus.root().join("cold.sqlite");
    let check = corpus.run(&compiler, None, false, Some(&cold)).unwrap();
    assert!(!check.passed(), "{:#?}", check.lines());
    assert!(check.artifacts_at_this_build);
    assert_eq!(check.moved_artifacts, 1);
    assert!(check.lines().iter().any(|line| line.starts_with(&format!(
        "authored-compile-difference\tartifact-changed\t{}\tprofile={}",
        case("alpha").case_id,
        LEGS[0]
    ))));
    let before = std::fs::read_to_string(corpus.expectations()).unwrap();
    let not_blessed = corpus.additions(&compiler, &[], Some(&cold)).unwrap();
    assert!(!not_blessed.blessed && !not_blessed.written);
    assert_eq!(
        std::fs::read_to_string(corpus.expectations()).unwrap(),
        before
    );

    // The file at another build: the moved digest is no difference.
    let other = ArtifactBuild {
        product: "f".repeat(64),
        executor: "vulkan-engine-v1".into(),
    };
    let at_other = before.replacen(&ArtifactBuild::this().line(), &other.line(), 1);
    std::fs::write(corpus.expectations(), &at_other).unwrap();
    let check = corpus.run(&compiler, None, false, Some(&cold)).unwrap();
    assert!(check.passed(), "{:#?}", check.lines());
    assert!(!check.artifacts_at_this_build);
    assert_eq!(check.moved_artifacts, 1);
    assert!(check
        .lines()
        .iter()
        .any(|line| line.contains("\tartifacts_at_this_build=false\tmoved_artifacts=1\t")));

    // An additions bless records it at this build; a later check at this build passes.
    let blessed = corpus.additions(&compiler, &[], Some(&cold)).unwrap();
    assert!(blessed.blessed && blessed.written, "{:#?}", blessed.lines());
    let written = Expectations::load(&corpus.expectations()).unwrap().unwrap();
    assert_eq!(written.build, Some(ArtifactBuild::this()));
    assert_eq!(
        written.cases[&case("alpha").case_id][0].artifact,
        Some(digest('3'))
    );
    assert_eq!(
        written.cases[&case("beta").case_id][0].artifact,
        Some(digest('2'))
    );
    assert!(corpus
        .run(&compiler, None, false, Some(&cold))
        .unwrap()
        .passed());

    // At another build again, an outcome that changed still refuses the additions bless.
    std::fs::write(
        corpus.expectations(),
        std::fs::read_to_string(corpus.expectations())
            .unwrap()
            .replacen(&ArtifactBuild::this().line(), &other.line(), 1),
    )
    .unwrap();
    let written = std::fs::read_to_string(corpus.expectations()).unwrap();
    // (A new product: the cache holds nothing under its key.)
    compiler.set("beta", LEGS[0], refused("not_supported{bind}"));
    let fresh = corpus.root().join("fresh.sqlite");
    let regressed = corpus.additions(&compiler, &[], Some(&fresh)).unwrap();
    assert!(!regressed.blessed && !regressed.written);
    assert!(regressed.differences.iter().any(Difference::is_regression));
    assert_eq!(
        std::fs::read_to_string(corpus.expectations()).unwrap(),
        written
    );
}

/// A table of census version 2 (no digest column) is extended in place, and its cells, under
/// older keys, are compiled again.
#[test]
fn a_version_2_cache_table_gains_the_digest_column() {
    let corpus = Corpus::new(&["alpha"]);
    let connection = Connection::open(corpus.index()).unwrap();
    connection
        .execute_batch(&format!(
            "CREATE TABLE {TABLE} (
               case_id TEXT NOT NULL, profile TEXT NOT NULL, key TEXT NOT NULL,
               outcome TEXT NOT NULL, host_replayed INTEGER NOT NULL, compiles INTEGER NOT NULL,
               elapsed_ms INTEGER NOT NULL, peak_resident_bytes INTEGER,
               allocator_peak_bytes INTEGER, PRIMARY KEY(case_id, profile));
             INSERT INTO {TABLE} VALUES ('{}', 'lavapipe', 'old-key', 'compiled', 0, 1, 1, NULL, NULL);",
            case("alpha").case_id
        ))
        .unwrap();
    drop(connection);
    let compiler = Scripted::default();
    compiler.digest("alpha", LEGS[0], &digest('4'));
    let report = corpus.run(&compiler, Some(&LEGS), true, None).unwrap();
    assert!(report.passed(), "{:#?}", report.lines());
    assert_eq!(report.cached_cells, 0);
    let warm = corpus.run(&compiler, None, false, None).unwrap();
    assert_eq!((warm.cached_cells, warm.compiles), (2, 0));
    assert!(warm.passed());
}
