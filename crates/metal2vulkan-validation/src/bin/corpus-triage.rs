use metal2vulkan_validation::bitcode_audit::peak_rss_bytes;
use metal2vulkan_validation::budgets::{BudgetAccumulator, BudgetReport};
use metal2vulkan_validation::census::{run_authoring_capability_census, AuthoringCensusRun};
use metal2vulkan_validation::cutover::scope::OutOfScope;
use metal2vulkan_validation::engine::{self, FailureKind, Outcome};
use metal2vulkan_validation::index::{
    default_index_path, reconcile_checkout_rows, select_queue, QueueRow, QueueState,
};
use metal2vulkan_validation::pool::parallel;
use metal2vulkan_validation::requirement::{ToolingRequirement, UnobservedEffect};
use metal2vulkan_validation::source::{
    corpus_root, for_each_indexed_source_analysis, public_source_set, PublicSources,
};
use metal2vulkan_validation::translation_audit::{
    engine_census, engine_cost_rows, engine_fingerprint, engine_recorded_profiles,
    select_engine_batch, write_engine_results, CostRow, EngineAuditResult, RecordedCost,
    SelectionMode, WorkerMeasure,
};
use metal2vulkan_validation::triage::{
    classify, read_cached, select_cached_audit_target_after, write_cached, AuditTarget,
    StructuralTriage,
};
use metal2vulkan_validation::worker::{self, alloc::CountingAllocator, WorkerFailure};
use metal2vulkan_validation::ScratchDir;
use std::collections::BTreeMap;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

mod corpus_triage_specialization;

const LARGE_TRANSLATION_SOURCE_BYTES: usize = 256 * 1024;
const MAX_LARGE_TRANSLATION_JOBS: usize = 2;
const SERIALIZED_TRANSLATION_MAX_BYTES: usize = 384 * 1024;
const SERIALIZED_TRANSLATION_CFG_CALL_WORK: usize = 140_000;

// The worker counts its allocations: it enforces the byte budget inside the child and measures the
// peak attributable bytes each engine attempt's ledger must account for.
#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

fn main() {
    worker::audit::serve_if_requested();
    // The authored compile census compiles each case in a case worker of this binary, and each of
    // the case's pipelines in a compile worker of it.
    metal2vulkan_validation::authored_compile::serve_if_requested();
    worker::compile::serve_if_requested();
    let result = match std::env::args().nth(1).as_deref() {
        Some("--translation-worker") => run_translation_worker(),
        Some(worker::l3::STAGE_ARG) => run_l3_stage(),
        _ => run(),
    };
    if let Err(error) = result {
        eprintln!("corpus-triage: {error}");
        std::process::exit(1);
    }
}

#[derive(Clone, Copy)]
enum AuditKind {
    AuthoringCapabilities,
    RasterizationRateMaps,
    SampleRate,
    UlongTextures,
    FragmentVaryings,
    LinkedFunctions,
    TileFunctions,
    MeshDraws,
    Translation,
    BitcodeReader,
    Budgets,
    Specialization,
    AuthoredCompile,
}

#[derive(Clone, Copy)]
enum TranslationSelection<'a> {
    Indexed {
        after: Option<&'a str>,
        limit: usize,
        mode: SelectionMode,
    },
    HashFile(&'a std::path::Path),
}

impl AuditKind {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "authoring-capabilities" => Ok(Self::AuthoringCapabilities),
            "rasterization-rate-maps" => Ok(Self::RasterizationRateMaps),
            "sample-rate" => Ok(Self::SampleRate),
            "ulong-textures" => Ok(Self::UlongTextures),
            "fragment-varyings" => Ok(Self::FragmentVaryings),
            "linked-functions" => Ok(Self::LinkedFunctions),
            "tile-functions" => Ok(Self::TileFunctions),
            "mesh-draws" => Ok(Self::MeshDraws),
            "translation" => Ok(Self::Translation),
            "bitcode-reader" => Ok(Self::BitcodeReader),
            "budgets" => Ok(Self::Budgets),
            "specialization" => Ok(Self::Specialization),
            "authored-compile" => Ok(Self::AuthoredCompile),
            _ => Err(format!(
                "unknown audit {value:?}; expected authoring-capabilities, rasterization-rate-maps, sample-rate, ulong-textures, fragment-varyings, linked-functions, tile-functions, mesh-draws, translation, bitcode-reader, budgets, specialization, or authored-compile"
            )),
        }
    }
}

fn run() -> Result<(), String> {
    let mut root = corpus_root();
    let mut index = None;
    let mut limit = 200usize;
    let mut summary_only = false;
    let mut audit = None;
    let mut after = None;
    let mut current_fingerprint = false;
    let mut retry_failures = false;
    let mut reclassify_all = false;
    let mut library_modules = false;
    let mut hash_file = None;
    let mut profiles = Vec::new();
    let mut jobs = default_jobs();
    let mut bless = false;
    let mut bless_additions = false;
    let mut expectations = None;
    let mut cases = std::collections::BTreeSet::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--corpus" => root = PathBuf::from(required(&mut args, "--corpus")?),
            "--private-corpus" => metal2vulkan_validation::source::use_private_corpus_root(
                PathBuf::from(required(&mut args, "--private-corpus")?),
            )?,
            "--index" => index = Some(PathBuf::from(required(&mut args, "--index")?)),
            "--limit" => {
                limit = required(&mut args, "--limit")?
                    .parse()
                    .map_err(|error| format!("invalid --limit: {error}"))?
            }
            "--jobs" => {
                jobs = required(&mut args, "--jobs")?
                    .parse()
                    .map_err(|error| format!("invalid --jobs: {error}"))?
            }
            "--summary-only" => summary_only = true,
            "--audit" => audit = Some(AuditKind::parse(&required(&mut args, "--audit")?)?),
            "--after" => after = Some(required(&mut args, "--after")?),
            "--current-fingerprint" => current_fingerprint = true,
            "--retry-failures" => retry_failures = true,
            "--reclassify-all" => reclassify_all = true,
            "--library-modules" => library_modules = true,
            "--hash-file" => hash_file = Some(PathBuf::from(required(&mut args, "--hash-file")?)),
            "--profile" => profiles.push(required(&mut args, "--profile")?),
            "--bless" => bless = true,
            "--bless-additions" => bless_additions = true,
            "--case" => {
                cases.insert(required(&mut args, "--case")?);
            }
            "--expectations" => {
                expectations = Some(PathBuf::from(required(&mut args, "--expectations")?))
            }
            "-h" | "--help" => {
                println!(
                    "usage: corpus-triage [--corpus DIR] [--private-corpus DIR] [--index PATH] [--limit N] [--jobs N] [--summary-only] [--audit authoring-capabilities [--reclassify-all] | --audit rasterization-rate-maps|sample-rate|ulong-textures|fragment-varyings|linked-functions|tile-functions|mesh-draws|bitcode-reader [--after SHA256] | --audit bitcode-reader --library-modules [--after SHA256] | --audit translation --profile P... [--after SHA256] [--current-fingerprint | --retry-failures | --hash-file PATH] | --audit budgets --profile P... | --audit specialization --profile P... | --audit authored-compile [--expectations PATH] [--bless [--profile P[+robustness-off]]... | --bless-additions | --case ID... [--profile P[+robustness-off]]...]]\n\n--audit authored-compile compiles every committed authored case of --corpus without a device, with the requests the engine executor builds for it (its constants, linked modules, bind shape, dispatch, companion stage, and device-replay variants), on each profile-leg of the expectation file (--expectations, by default authored-compile.tsv under the corpus root), each pipeline in a compile worker under the 20-s and 500-MiB ceilings, --jobs cases at once. Each case's outcome on each profile-leg (compiled, or the executor's refusal label) is cached in the index under the product's census key, the executor ABI, and the census's version, so a warm rerun compiles nothing; failures are never cached. It fails on any failed cell (an internal error, a worker stop, a case that does not check or map) and on any difference from the expectation file (a regression, a changed refusal, a newly compiled case, a new or removed case). --bless rewrites the file from the run and prints the differences (never with a failed cell); with --profile it compiles those profile-legs instead of the file's. --bless-additions (each train's gate step) compiles the file's profile-legs plus every C3 venue's profile-leg the file lacks (crates/metal2vulkan-validation/cutover/c3-venues.json), fails as a check does on any failed cell and on any changed cell of a case and profile-leg the file holds, writing nothing, and otherwise writes the file with the new cases and profile-legs added and the removed cases dropped, only when that changes it; its authored-compile-additions line counts what it added and removed. Each case is admitted, prepared, and compiled in its own case worker (a child of this binary under the 500-MiB ceiling, its wall ceiling 20 s plus 40 s per profile-leg), so this process holds only the committed cases and their cells; a case worker reads each private source by the byte range --index records (read only, never repaired), and one the index does not locate by parsing its whole shard (counted as sources_by_range and sources_by_shard). --case ID (repeatable) is a probe: it compiles only those committed cases (on --profile's profile-legs when named, else the file's), prints an authored-compile-cell line per cell with its outcome or failure and what its compiles measured, and compares those cases' rows with the expectation file when it exists; it never blesses. Host-replayed command buffers, whose pipelines the producer's records decide on the device, are reported as uncovered.\n\n--audit translation translates each row's retained bitcode (a public .ll row, which retains none, is assembled with llvm-as) with the engine for every --profile P (a committed profile under crates/metal2vulkan-validation/profiles, derived under the default untrusted policy), one profile after another, and records typed outcome classes under a per-profile census key (the product sources, SEMANTICS_VERSION, the lane's version, the profile, and the llvm-as and spirv-val versions; no other validation source); its summary is the non-gating shadow census. Rows stream from the index through the workers, so the parent's memory does not grow with --limit. --jobs defaults to the host's available logical CPU count (the equivalent of nproc). An explicit --jobs N overrides that default. --hash-file reads a whitespace-delimited SHA-256 from the first field of each non-empty line and audits exactly those indexed sources. --audit bitcode-reader decodes each selected row's retained bitcode with the engine's reader and compares its structure with llvm-dis (METAL2VULKAN_LLVM_DIS); it also reports the intrinsic census (calls decoded to air.*/llvm.* and every name kept as an unknown intrinsic, with rows, calls, and whether the decoder lists its family as not decoded) and every function called but declared without a body that is no intrinsic (what the module needs linked), with rows and calls; without --summary-only, each row's such functions too. Without --library-modules it only reads the index. With --library-modules it first syncs the library-module index with the module shards on disk (incrementally: only a new or changed shard is opened, and a warm index opens none), then selects the indexed library modules (the linked dependencies harvest retains apart from the sources) instead of the sources, in module-hash order after --after, at most --limit, reading only the selected modules' byte ranges. --audit bitcode-reader runs each attempt (a row's decode) in its own killable worker under the 20-s and 500-MiB ceilings: a stopped attempt is reported as worker-stopped <outcome> and counted in worker_stopped, beside the workers' peak RSS, allocator peak, and slowest time. --audit budgets reports the budget gates of each --profile P from the rows the current engine fingerprint recorded (slowest row, p50/p99 attempt time, peak worker RSS and the largest counting-allocator peak a worker reported, and the ledger >= 0.7 x allocator-peak check); a row stopped at the memory ceiling or by a worker abort, or refused at an in-process budget, fails it whatever its measure. It only reads the index and fails when a gate fails. Timeouts are transient and never recorded, so --audit budgets cannot see them; instead --audit translation exits non-zero when its batch has a budget stop (a timeout, OOM, worker abort, or in-process budget refusal) or a row above the 20-s or 500-MiB ceiling (its sampled RSS or its allocator peak), after reporting every profile (translation-audit-budgets-failure lines name them). Its outcome classes, shadow census, and batch ledger lines stay non-gating. --audit specialization first plans, in killable workers, every indexed row without a plan under the planner key (the product's sources, the lane's version, the census tools) or whose authored cases changed: the row's declared function constants and their value sets (the authored cases' sets; zero and one, which leave optional constants undefined; zero_defined and one_defined; a strength-2 binary covering array over every constant), kept whole when over the 16-set bound and reported. It plans at most --limit rows, in hash order (a row not yet planned has no set selected; the summaries count them). Then, for each --profile P in turn, it translates each planned set of each planned row, and each row whose current plain engine census outcome (which must cover every indexed row) is needs_input other than a function constant under the inputs set (no constants), at most --limit attempts (whole rows), each attempt in its own killable worker under the 20-s and 500-MiB ceilings, deriving the inputs the row's AIR states (a kernel's stage-input descriptor, a post-tessellation vertex descriptor, tile attachment formats, a storage texture's format, a visible function's retained module). Attempts are recorded in the lane's own table under the profile's engine census key extended by the lane's version; transient outcomes are not recorded. It prints the planning batch and the plans' summary (with the fifth function-constant operand's agreement with \"required\"), then per profile the batch (with its lanes and budgets) and the report over every recorded attempt: rows by plain outcome, outcomes by kind, inputs derived and left with their reasons, refusal classes ranked by rows with their value-dependent rows (ray-tracing classes listed apart), and the recorded attempts' budgets. It exits non-zero, after reporting everything, when a planning or attempt worker was stopped at or ran above a ceiling, or an attempt's ledger peak is below 7/10 of its allocator peak (each compile of an attempt is metered and measured on its own, and the attempt's cost is its last compile's)."
                );
                return Ok(());
            }
            _ => return Err(format!("unknown argument {arg:?}")),
        }
    }
    if limit == 0 {
        return Err("--limit must be greater than zero".into());
    }
    if jobs == 0 {
        return Err("--jobs must be greater than zero".into());
    }
    if (bless || bless_additions || expectations.is_some() || !cases.is_empty())
        && !matches!(audit, Some(AuditKind::AuthoredCompile))
    {
        return Err(
            "--bless, --bless-additions, --expectations, and --case require --audit \
             authored-compile"
                .into(),
        );
    }
    if bless && bless_additions {
        return Err(
            "--bless rewrites the file; --bless-additions only adds to it: name one".into(),
        );
    }
    if matches!(audit, Some(AuditKind::AuthoredCompile)) {
        if after.is_some()
            || current_fingerprint
            || retry_failures
            || reclassify_all
            || library_modules
            || hash_file.is_some()
        {
            return Err(
                "--audit authored-compile takes only --corpus, --private-corpus, --index, --jobs, \
                 --expectations, --bless, --bless-additions, --case, and a bless's or a probe's \
                 --profile"
                    .into(),
            );
        }
        return audit_authored_compile(
            &root,
            index,
            expectations,
            &profiles,
            (!cases.is_empty()).then_some(cases),
            jobs,
            if bless {
                metal2vulkan_validation::authored_compile::Mode::Bless
            } else if bless_additions {
                metal2vulkan_validation::authored_compile::Mode::BlessAdditions
            } else {
                metal2vulkan_validation::authored_compile::Mode::Check
            },
        );
    }
    if after.is_some() && audit.is_none() {
        return Err("--after requires --audit".into());
    }
    if current_fingerprint && !matches!(audit, Some(AuditKind::Translation)) {
        return Err("--current-fingerprint requires --audit translation".into());
    }
    if retry_failures && !matches!(audit, Some(AuditKind::Translation)) {
        return Err("--retry-failures requires --audit translation".into());
    }
    if reclassify_all && !matches!(audit, Some(AuditKind::AuthoringCapabilities)) {
        return Err("--reclassify-all requires --audit authoring-capabilities".into());
    }
    if library_modules && !matches!(audit, Some(AuditKind::BitcodeReader)) {
        return Err("--library-modules requires --audit bitcode-reader".into());
    }
    if hash_file.is_some() && !matches!(audit, Some(AuditKind::Translation)) {
        return Err("--hash-file requires --audit translation".into());
    }
    if hash_file.is_some() && (after.is_some() || current_fingerprint || retry_failures) {
        return Err(
            "--hash-file is mutually exclusive with --after, --current-fingerprint, and --retry-failures"
                .into(),
        );
    }
    if current_fingerprint && retry_failures {
        return Err("--current-fingerprint and --retry-failures are mutually exclusive".into());
    }
    if after.as_deref().is_some_and(|hash| {
        hash.len() != 64
            || !hash
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    }) {
        return Err("--after must be a lowercase SHA-256".into());
    }

    check_profiles(audit, &profiles)?;

    let index = index.unwrap_or_else(|| default_index_path(&root));
    // The index may have been synchronized by a checkout at another commit, whose public fixtures
    // and authored cases are not this one's. Every mode that selects rows from it makes its public
    // rows and case rows this checkout's first; the authoring census does so as part of its run.
    // The budgets audit reads only recorded translation rows.
    let public = public_source_set()?;
    if !matches!(
        audit,
        Some(AuditKind::AuthoringCapabilities | AuditKind::Budgets)
    ) {
        for line in reconcile_checkout_rows(&root, &index, &public)?.report_lines() {
            println!("{line}");
        }
    }
    if let Some(audit) = audit {
        return match audit {
            AuditKind::AuthoringCapabilities => {
                if after.is_some() {
                    return Err(
                        "--after is not accepted by the authoring-capabilities audit".into(),
                    );
                }
                audit_authoring_capabilities(&root, &index, &public, limit, jobs, reclassify_all)
            }
            AuditKind::RasterizationRateMaps => {
                audit_rasterization_rate_maps(&root, &index, after.as_deref(), limit, summary_only)
            }
            AuditKind::SampleRate => {
                audit_sample_rate(&root, &index, after.as_deref(), limit, summary_only)
            }
            AuditKind::UlongTextures => {
                audit_ulong_textures(&root, &index, after.as_deref(), limit, summary_only)
            }
            AuditKind::FragmentVaryings => {
                audit_fragment_varyings(&root, &index, after.as_deref(), limit, summary_only)
            }
            AuditKind::LinkedFunctions => {
                audit_linked_functions(&root, &index, after.as_deref(), limit, summary_only)
            }
            AuditKind::TileFunctions => {
                audit_tile_functions(&root, &index, after.as_deref(), limit, summary_only)
            }
            AuditKind::MeshDraws => {
                audit_mesh_draws(&root, &index, after.as_deref(), limit, summary_only)
            }
            AuditKind::BitcodeReader => audit_bitcode_reader(
                &root,
                &index,
                if library_modules {
                    ReaderSelection::LibraryModules
                } else {
                    ReaderSelection::Sources
                },
                after.as_deref(),
                limit,
                jobs,
                summary_only,
            ),
            AuditKind::Budgets => {
                if after.is_some() {
                    return Err("--after is not accepted by the budgets audit".into());
                }
                let mut failed = Vec::new();
                for profile in &profiles {
                    if !audit_budgets(&index, profile, summary_only)? {
                        failed.push(profile.as_str());
                    }
                }
                if failed.is_empty() {
                    Ok(())
                } else {
                    Err(format!("budget gates failed for {}", failed.join(", ")))
                }
            }
            AuditKind::Specialization => {
                if after.is_some() {
                    return Err("--after is not accepted by the specialization audit".into());
                }
                // Per-profile sweeps run sequentially: `jobs` bounds one profile's workers.
                corpus_triage_specialization::audit_specialization(
                    &root,
                    &index,
                    jobs,
                    summary_only,
                    limit,
                    &profiles,
                )
            }
            // Dispatched before the index is touched, above.
            AuditKind::AuthoredCompile => {
                Err("--audit authored-compile is dispatched before the other audits".into())
            }
            AuditKind::Translation => {
                let selection = if let Some(path) = hash_file.as_deref() {
                    TranslationSelection::HashFile(path)
                } else {
                    TranslationSelection::Indexed {
                        after: after.as_deref(),
                        limit,
                        mode: if retry_failures {
                            SelectionMode::RetryCurrentFailures
                        } else if current_fingerprint {
                            SelectionMode::CurrentFingerprint
                        } else {
                            SelectionMode::Discovery
                        },
                    }
                };
                // Per-profile sweeps run sequentially: `jobs` bounds the workers of one profile,
                // never a product of profiles and jobs. Every profile runs; the sweep then fails if
                // any batch breached a ceiling.
                let mut breached = Vec::new();
                for profile in &profiles {
                    let breaches = audit_translation_new(
                        &root,
                        &index,
                        jobs,
                        summary_only,
                        selection,
                        profile,
                    )?;
                    if !breaches.is_empty() {
                        breached.push(format!("{profile} ({})", breaches.len()));
                    }
                }
                if breached.is_empty() {
                    Ok(())
                } else {
                    Err(format!(
                        "translation attempts breached the 20-s/500-MiB ceilings for {} \
                         (translation-audit-budgets-failure lines name them)",
                        breached.join(", ")
                    ))
                }
            }
        };
    }
    let queue = select_queue(&index, QueueState::Unplanned, limit)?;
    let selected_hashes = queue
        .iter()
        .map(|row| row.air_sha256.clone())
        .collect::<Vec<_>>();
    let mut results = read_cached(&index, &selected_hashes)?;
    let missing = queue
        .iter()
        .filter(|row| !results.contains_key(&row.air_sha256))
        .map(|row| row.air_sha256.clone())
        .collect::<Vec<_>>();
    let mut fresh = BTreeMap::new();
    for_each_indexed_source_analysis(&root, &index, &missing, |source| {
        fresh.insert(source.key().into_owned(), classify(&source)?);
        Ok(())
    })?;
    write_cached(
        &index,
        fresh.iter().map(|(hash, result)| (hash.as_str(), result)),
    )?;
    results.append(&mut fresh);
    let mut summary = TriageSummary::default();
    for row in queue {
        let result = results
            .get(&row.air_sha256)
            .ok_or_else(|| format!("selected AIR {} has no triage result", row.air_sha256))?;
        summary.process(row, result, summary_only);
    }
    println!(
        "summary\tselected={}\tactionable={}\tsignatures={}",
        summary.actionable,
        summary.actionable,
        if summary_only {
            "skipped".to_string()
        } else {
            summary.signatures.len().to_string()
        }
    );
    for (stage, count) in summary.stages {
        println!("stage\t{stage}\t{count}");
    }
    for (requirement, count) in summary.requirements {
        println!("requirement\t{requirement}\t{count}");
    }
    for (effect, count) in summary.unobserved_effects {
        println!("unobserved-effect\t{effect}\t{count}\t{}", effect.reason());
    }
    if !summary_only {
        for (signature, count) in summary.signatures {
            println!("signature\t{signature}\t{count}");
        }
    }
    Ok(())
}

/// Check `--profile` against the audit: `--audit translation`, `--audit budgets`, and
/// `--audit specialization` take one or more committed profiles, each once (`--audit authored-compile`'s
/// bless and probe profiles are checked where they are read); no other audit takes one.
fn check_profiles(audit: Option<AuditKind>, profiles: &[String]) -> Result<(), String> {
    match audit {
        Some(AuditKind::Translation | AuditKind::Budgets | AuditKind::Specialization) => {
            if profiles.is_empty() {
                return Err(
                    "--audit translation, budgets, and specialization take one or more --profile P"
                        .into(),
                );
            }
            let mut unique = profiles.to_vec();
            unique.sort();
            unique.dedup();
            if unique.len() != profiles.len() {
                return Err("each --profile may be given once".into());
            }
            Ok(())
        }
        _ if !profiles.is_empty() => Err(
            "--profile requires --audit translation, budgets, specialization, or \
             authored-compile"
                .into(),
        ),
        _ => Ok(()),
    }
}

/// `--audit authored-compile`: the census over the corpus at `root` (or a probe of `cases`), its
/// cells cached in `index` (the corpus's by default), compared with `expectations` (the corpus's
/// `authored-compile.tsv` by default), or blessed into it as `mode` says (an additions bless adds
/// every committed C3 venue's missing profile-legs); each case in a case worker of this binary.
fn audit_authored_compile(
    root: &std::path::Path,
    index: Option<PathBuf>,
    expectations: Option<PathBuf>,
    profiles: &[String],
    cases: Option<std::collections::BTreeSet<String>>,
    jobs: usize,
    mode: metal2vulkan_validation::authored_compile::Mode,
) -> Result<(), String> {
    use metal2vulkan_validation::authored_compile::{
        expectations_path, run, venue_legs, CaseWorkers, Mode, Options, ProfileLeg,
    };
    use metal2vulkan_validation::c3::Venues;
    let index = index.unwrap_or_else(|| default_index_path(root));
    let expectations = expectations.unwrap_or_else(|| expectations_path(root));
    let profiles = if profiles.is_empty() {
        None
    } else {
        Some(
            profiles
                .iter()
                .map(|label| ProfileLeg::parse(label))
                .collect::<Result<Vec<_>, _>>()?,
        )
    };
    // The case worker resolves the corpus from its own working directory.
    let absolute = |path: &std::path::Path| {
        std::path::absolute(path).map_err(|error| format!("resolve {}: {error}", path.display()))
    };
    // The C3 venues' profile-legs: a bless-additions run adds the file's missing ones, and every
    // run keeps their cells' artifact digests.
    let venues = venue_legs(&Venues::load(&Venues::committed_path())?);
    let compiler = CaseWorkers {
        root: absolute(root)?,
        index: absolute(&index)?,
        program: std::env::current_exe()
            .map_err(|error| format!("resolve the case worker executable: {error}"))?,
    };
    let report = run(
        &Options {
            root,
            index: &index,
            expectations: &expectations,
            profiles,
            cases,
            jobs,
            mode,
            required: match mode {
                Mode::BlessAdditions => venues.clone(),
                Mode::Check | Mode::Bless => Vec::new(),
            },
            artifacts: venues,
        },
        &compiler,
    )?;
    for line in report.lines() {
        println!("{line}");
    }
    println!(
        "authored-compile-parent\tpeak_rss_bytes={}\tindex={}\texpectations={}",
        peak_rss_bytes(),
        index.display(),
        expectations.display()
    );
    if report.passed() {
        return Ok(());
    }
    Err(format!(
        "the authored compile census failed: {} failed cell(s), {} difference(s) from {}{}",
        report.failures.len(),
        report.differences.len(),
        expectations.display(),
        match mode {
            Mode::Check => "",
            Mode::Bless => " (not blessed: a failed cell is never written)",
            Mode::BlessAdditions =>
                " (not blessed: a failed cell, or a changed cell the file holds, is never written)",
        }
    ))
}

fn default_jobs() -> usize {
    std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(1)
}

fn audit_authoring_capabilities(
    root: &std::path::Path,
    index: &std::path::Path,
    public: &PublicSources,
    limit: usize,
    jobs: usize,
    reclassify_all: bool,
) -> Result<(), String> {
    let AuthoringCensusRun {
        public: reconciliation,
        cases,
        selected,
        selection_elapsed,
        classify_elapsed: source_elapsed,
        read: read_stats,
        summary,
    } = run_authoring_capability_census(root, index, public, limit, jobs, reclassify_all)?;
    // The index's public rows are this checkout's before selection.
    for line in reconciliation.report_lines() {
        println!("{line}");
    }
    println!("{}", cases.report_line());
    // The census covers indexed rows, one per entry of every module. Rows Metal itself cannot build
    // are closed without a reference and counted apart.
    println!(
        "authoring-capability-summary\tselected={}\ttotal={}\tclassified={}\tremaining={}\tunresolved={}\tindex_select_ms={}\tclassify_cache_ms={}\tsource_shards_opened={}\tsource_bytes_read={}\trepair_shards_scanned={}\trepair_bytes_scanned={}\tmetal_refused={}",
        selected,
        summary.total_sources,
        summary.classified_sources,
        summary.remaining_sources(),
        summary.unresolved_sources,
        selection_elapsed.as_millis(),
        source_elapsed.as_millis(),
        read_stats.source_shards_opened,
        read_stats.source_bytes_read,
        read_stats.repair_shards_scanned,
        read_stats.repair_bytes_scanned,
        summary.metal_refused_sources,
    );
    // Each requirement with the executors it stops and whose limit it is; a tooling requirement
    // makes its rows unresolved.
    for (requirement, count) in &summary.requirements {
        println!(
            "authoring-requirement\t{requirement}\t{count}\tstops={}\tlimit={}",
            requirement
                .stops()
                .iter()
                .map(|executor| executor.as_str())
                .collect::<Vec<_>>()
                .join(","),
            requirement.limit()
        );
    }
    // An executor's tooling gaps keep the census open; Metal's own refusals are closed.
    for (executor, stops) in &summary.executor_stops {
        println!(
            "authoring-executor-stops\t{executor}\t{}\ttooling_gap={}\tmetal_refused={}",
            stops.sources, stops.tooling_gap, stops.metal_refused,
        );
    }
    // Authorable, not unresolved: these rows' cases do not evidence the named effect.
    for (effect, count) in &summary.unobserved_effects {
        println!(
            "authoring-unobserved-effect\t{effect}\t{count}\t{}",
            effect.reason()
        );
    }
    for (intrinsic, count) in &summary.unrecognized_air_intrinsics {
        println!("unrecognized-air-intrinsic\t{intrinsic}\t{count}");
    }
    for line in public.report_lines() {
        println!("{line}");
    }
    // The executor contract covers the Metal oracle and both candidates; the engine's
    // product-support gaps are reported beside it from the recorded translation rows, reading no
    // source.
    for profile in engine_recorded_profiles(index)? {
        print_shadow_census(index, &profile)?;
    }
    match summary.closure_gap() {
        Some(gap) => Err(gap),
        None => Ok(()),
    }
}

/// `--audit translation --profile P`: translate each selected row with the new
/// engine for `profile` in killable workers (its retained bitcode, or a public row's AIR text
/// assembled with `llvm-as`), record typed outcomes under the profile's census key, and report
/// the batch and the profile's shadow census. Rows stream from the index through the workers.
///
/// Returns the batch's [`BudgetReport::ceiling_breaches`] (budget stops, including timeouts,
/// which are transient and never recorded, and rows above the 20-s or 500-MiB ceilings); the
/// caller exits non-zero when any profile has one. The outcome classes and the shadow census stay
/// non-gating, and so does the batch's ledger line (`--audit budgets` gates it).
fn audit_translation_new(
    root: &std::path::Path,
    index: &std::path::Path,
    jobs: usize,
    summary_only: bool,
    selection: TranslationSelection<'_>,
    profile_stem: &str,
) -> Result<Vec<String>, String> {
    let profile = engine::load_profile(profile_stem)?;
    let fingerprint = engine_fingerprint(profile_stem, &profile);
    drop(profile);
    let started = Instant::now();
    let hashes = match selection {
        TranslationSelection::Indexed { after, limit, mode } => {
            select_engine_batch(index, profile_stem, &fingerprint, mode, after, limit)?
        }
        TranslationSelection::HashFile(path) => read_hash_file(path)?,
    };
    let selection_elapsed = started.elapsed();
    // The parent keeps these tallies and the budget report's own facts, never a row.
    let mut transient_not_recorded = 0usize;
    let mut outcomes = BTreeMap::<(String, String), usize>::new();
    let (mut translated, mut defects) = (0usize, 0usize);
    let mut budgets = BudgetAccumulator::new();
    let sweep = sweep_translation_rows(
        &hashes,
        jobs,
        "translation-audit",
        &|hashes, consume| {
            metal2vulkan_validation::source::for_each_indexed_source_with_blob(
                root,
                index,
                hashes,
                |row| consume(EngineRow::from(row)),
            )
        },
        &|row| audit_engine_source(root, index, row, profile_stem),
        &mut |result: &EngineAuditResult, _| {
            let outcome = &result.attempt.outcome;
            let (label, key) = (outcome.label().to_string(), outcome.key());
            translated += usize::from(*outcome == Outcome::Translated);
            defects += usize::from(outcome.is_defect());
            if !summary_only {
                println!("translation-audit\t{}\t{label}\t{key}", result.air_sha256);
            }
            if outcome.is_transient() {
                report_transient(
                    "translation-audit",
                    profile_stem,
                    &result.air_sha256,
                    (&label, &key),
                    result.attempt.detail.as_deref(),
                    result.worker,
                );
            }
            budgets.add(&CostRow {
                air_sha256: result.air_sha256.clone(),
                outcome: (label.clone(), key.clone()),
                cost: RecordedCost::of(result).unwrap_or_default(),
            });
            *outcomes.entry((label, key)).or_default() += 1;
        },
        &mut |checkpoint| {
            transient_not_recorded +=
                write_engine_results(index, profile_stem, &fingerprint, checkpoint)?;
            Ok(())
        },
    )?;
    let (slowest_row, slowest_ms) = sweep.slowest();
    println!(
        "translation-audit-summary\tengine=new\tprofile={profile_stem}\tfingerprint={fingerprint}\tselected={}\ttranslated={translated}\tnot_translated={}\tdefects={defects}\ttransient_not_recorded={transient_not_recorded}\tindex_select_ms={}\ttranslate_validate_ms={}\tslowest_air_sha256={slowest_row}\tslowest_ms={slowest_ms}\tparent_peak_rss_bytes={}\t{}\tindexed_rows={}\treread_rows={}\tsource_shards_opened={}\tsource_bytes_read={}\trepair_shards_scanned={}\trepair_bytes_scanned={}",
        sweep.rows,
        sweep.rows - translated,
        selection_elapsed.as_millis(),
        sweep.wall.as_millis(),
        peak_rss_bytes(),
        sweep.parent,
        sweep.read.rows,
        sweep.reread,
        sweep.read.source_shards_opened,
        sweep.read.source_bytes_read,
        sweep.read.repair_shards_scanned,
        sweep.read.repair_bytes_scanned,
    );
    // Ray tracing is out of scope for the launch: its outcomes print apart.
    for ((label, key), count) in &outcomes {
        match OutOfScope::of_recorded(label, key) {
            None => println!("translation-audit-outcome\t{count}\t{label}\t{key}"),
            Some(scope) => println!(
                "translation-audit-out-of-scope\t{}\t{count}\t{label}\t{key}",
                scope.as_str()
            ),
        }
    }
    let report = budgets.finish();
    print_budget_report(
        "translation-audit-budgets",
        profile_stem,
        &report,
        summary_only,
    );
    print_shadow_census(index, profile_stem)?;
    Ok(report.ceiling_breaches())
}

/// `--audit budgets --profile P`: the budget gates over every row the current engine
/// fingerprint of `profile_stem` recorded (read-only). Returns whether every gate passed.
///
/// Timeouts and allocation failures are transient and never recorded, so this report is blind to
/// them; the engine translation sweep that met one exits non-zero (its per-batch report sees
/// every attempted row, [`BudgetReport::ceiling_breaches`]). Recorded `oom` and `worker_exit`
/// rows, with the measure up to their stop, and `limit_exceeded budget` rows fail here too.
fn audit_budgets(
    index: &std::path::Path,
    profile_stem: &str,
    summary_only: bool,
) -> Result<bool, String> {
    let profile = engine::load_profile(profile_stem)?;
    let fingerprint = engine_fingerprint(profile_stem, &profile);
    drop(profile);
    let rows = engine_cost_rows(index, profile_stem, &fingerprint)?;
    let report = BudgetReport::from_rows(&rows);
    println!("budgets-fingerprint\t{profile_stem}\t{fingerprint}");
    print_budget_report("budgets", profile_stem, &report, summary_only);
    Ok(report.failures().is_empty())
}

/// One summary line of a [`BudgetReport`] and one line per failed gate.
fn print_budget_report(label: &str, profile_stem: &str, report: &BudgetReport, summary_only: bool) {
    fn pair(value: &Option<(String, u64)>) -> (&str, u64) {
        value
            .as_ref()
            .map_or(("none", 0), |(row, n)| (row.as_str(), *n))
    }
    let (slowest_row, slowest_ms) = pair(&report.slowest);
    let (rss_row, rss) = pair(&report.peak_resident);
    let (allocator_row, allocator) = pair(&report.peak_allocator);
    let opt = |value: Option<u64>| value.map_or_else(|| "none".to_string(), |v| v.to_string());
    let failures = report.failures();
    println!(
        "{label}-summary\tprofile={profile_stem}\trows={}\ttimed={}\tslowest_air_sha256={slowest_row}\tslowest_ms={slowest_ms}\tp50_ms={}\tp99_ms={}\tpeak_rss_air_sha256={rss_row}\tpeak_rss_bytes={rss}\tpeak_allocator_air_sha256={allocator_row}\tpeak_allocator_bytes={allocator}\tallocator_measured={}\tbudget_stops={}\tledger_checked={}\tledger_below_floor={}\tledger_min_ratio={}\tledger_min_ratio_air_sha256={}\tgates={}",
        report.rows,
        report.timed,
        opt(report.p50_ms),
        opt(report.p99_ms),
        report.allocator_measured,
        report.budget_stops.len(),
        report.ledger_checked,
        report.ledger_shortfalls.len(),
        report
            .ledger_min_ratio
            .as_ref()
            .map_or_else(|| "none".to_string(), |(_, r)| format!("{r:.3}")),
        report
            .ledger_min_ratio
            .as_ref()
            .map_or("none", |(row, _)| row.as_str()),
        if failures.is_empty() { "pass" } else { "fail" },
    );
    // A summary-only run names the first failures and counts the rest; the full list is one
    // `--audit budgets` away.
    let shown = if summary_only {
        SUMMARY_FAILURE_LINES
    } else {
        usize::MAX
    };
    for failure in failures.iter().take(shown) {
        println!("{label}-failure\t{profile_stem}\t{failure}");
    }
    if failures.len() > shown {
        println!(
            "{label}-failure\t{profile_stem}\t... {} more (run without --summary-only)",
            failures.len() - shown
        );
    }
}

/// Failure lines a `--summary-only` budget report prints before counting the rest.
const SUMMARY_FAILURE_LINES: usize = 10;

/// Print the engine's shadow census for `profile_stem` at its current fingerprint. It never
/// gates anything.
fn print_shadow_census(index: &std::path::Path, profile_stem: &str) -> Result<(), String> {
    let profile = match engine::load_profile(profile_stem) {
        Ok(profile) => profile,
        Err(error) => {
            println!(
                "shadow-census\tprofile={profile_stem}\tgating=no\tstale={}",
                single_line(&error)
            );
            return Ok(());
        }
    };
    let fingerprint = engine_fingerprint(profile_stem, &profile);
    let census = engine_census(index, profile_stem, &fingerprint)?;
    let defects = census.defects();
    // Ray tracing is out of scope for the launch: the class counts hold every row, and its rows are
    // counted again apart, with their outcomes on lines of their own.
    let mut out_of_scope = BTreeMap::<OutOfScope, usize>::new();
    for ((label, key), count) in &census.outcomes {
        if let Some(scope) = OutOfScope::of_recorded(label, key) {
            *out_of_scope.entry(scope).or_default() += count;
        }
    }
    let out_of_scope_fields = OutOfScope::ALL
        .iter()
        .map(|scope| {
            format!(
                "\tout_of_scope_{}={}",
                scope.as_str(),
                out_of_scope.get(scope).copied().unwrap_or(0)
            )
        })
        .collect::<String>();
    println!(
        "shadow-census\tprofile={profile_stem}\tfingerprint={fingerprint}\tgating=no\ttotal={}\tattempted={}\tremaining={}\ttranslated={}\tneeds_input={}\tunsupported={}\tdevice_capability_missing={}\tinvalid_input={}\tinvalid_descriptor={}\tlimit_exceeded={}\tfailed={}\tdefects={defects}{out_of_scope_fields}",
        census.total_sources,
        census.attempted,
        census.remaining,
        census.count("translated"),
        census.count("needs_input"),
        census.count("unsupported"),
        census.count("device_capability_missing"),
        census.count("invalid_input"),
        census.count("invalid_descriptor"),
        census.count("limit_exceeded"),
        census.count("failed"),
    );
    for ((label, key), count) in &census.outcomes {
        match OutOfScope::of_recorded(label, key) {
            None => println!("shadow-census-outcome\t{profile_stem}\t{count}\t{label}\t{key}"),
            Some(scope) => println!(
                "shadow-census-out-of-scope\t{profile_stem}\t{}\t{count}\t{label}\t{key}",
                scope.as_str()
            ),
        }
    }
    Ok(())
}

/// Where a row runs in the schedule both engines share.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lane {
    /// The first phase, on every worker.
    Small,
    /// The second phase, one row at a time on one worker: dense call CFGs whose combined cost
    /// serializes them.
    Serialized,
    /// The second phase, on at most [`MAX_LARGE_TRANSLATION_JOBS`] workers, largest first.
    Large,
}

/// What a sweep's schedule reads of a row.
trait SweepRow: Send {
    /// The row's key ([`metal2vulkan_validation::source::SourceRow::key`]).
    fn key(&self) -> std::borrow::Cow<'_, str>;
    fn lane(&self) -> Lane;
    /// Second-phase rows run largest first by this size.
    fn size(&self) -> usize;
}

impl SweepRow for metal2vulkan_validation::source::SourceRow {
    fn key(&self) -> std::borrow::Cow<'_, str> {
        metal2vulkan_validation::source::SourceRow::key(self)
    }

    fn lane(&self) -> Lane {
        if is_serialized_cost_translation_source(self) {
            Lane::Serialized
        } else if is_costly_translation_source(self) {
            Lane::Large
        } else {
            Lane::Small
        }
    }

    fn size(&self) -> usize {
        self.air_ll.len()
    }
}

/// Streams the indexed rows of the given hashes, in that order, into the consumer, one row at a
/// time (`source::for_each_indexed_source_*`).
type ReadRows<'a, T> = dyn Fn(
        &[String],
        &mut dyn FnMut(T) -> Result<(), String>,
    ) -> Result<metal2vulkan_validation::source::IndexedSourceReadStats, String>
    + Sync
    + 'a;

/// Rows persisted per checkpoint: a killed sweep loses at most this many.
const CHECKPOINT_ROWS: usize = 10;

/// What a sweep measured; the rows themselves went to its `record` and `checkpoint_rows`.
#[derive(Debug, Default)]
struct SweepSummary {
    rows: usize,
    wall: Duration,
    slowest: Option<(String, u128)>,
    /// Second-phase rows, read again after the first phase classified them.
    reread: usize,
    read: metal2vulkan_validation::source::IndexedSourceReadStats,
    /// The parent's own memory when the sweep ended.
    parent: ParentMemory,
}

/// The supervising parent's own memory when a sweep ends: its resident bytes once the allocator's
/// free pages went back to the system, how many that released, its live heap, and its live heap's
/// high-water mark since it started. The gap between resident and live bytes shows what the
/// allocator still keeps. Written into a lane's summary line.
#[derive(Debug, Default)]
struct ParentMemory {
    resident: Option<u64>,
    released: Option<u64>,
    live_heap: Option<u64>,
    live_heap_peak: Option<u64>,
}

impl ParentMemory {
    /// Release the allocator's free pages ([`worker::alloc::release_free_memory`]) and measure.
    fn released_now() -> ParentMemory {
        let before = worker::process::own_resident_bytes();
        worker::alloc::release_free_memory();
        let resident = worker::process::own_resident_bytes();
        ParentMemory {
            resident,
            released: before.zip(resident).map(|(b, a)| b.saturating_sub(a)),
            live_heap: worker::alloc::installed().then(|| worker::alloc::live_bytes() as u64),
            live_heap_peak: worker::alloc::peak_bytes(),
        }
    }
}

impl std::fmt::Display for ParentMemory {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let known = |value: Option<u64>| value.map_or("unknown".to_string(), |v| v.to_string());
        write!(
            out,
            "parent_rss_bytes={}\tparent_released_bytes={}\tparent_live_heap_bytes={}\tparent_live_heap_peak_bytes={}",
            known(self.resident),
            known(self.released),
            known(self.live_heap),
            known(self.live_heap_peak),
        )
    }
}

/// One stderr line for an attempt its lane does not record because the outcome is transient (a
/// deadline, cancellation, tooling, an allocation failure), so every unrecorded attempt's cause is
/// in the run's log: the lane, the profile, the attempt, its outcome, what the parent measured, and
/// the detail on one bounded line.
fn report_transient(
    lane: &str,
    profile: &str,
    attempt: &str,
    (label, key): (&str, &str),
    detail: Option<&str>,
    worker: Option<WorkerMeasure>,
) {
    let known = |value: Option<u64>| value.map_or("unknown".to_string(), |v| v.to_string());
    eprintln!(
        "{lane}-transient\tprofile={profile}\tattempt={attempt}\toutcome={label}\tkey={key}\telapsed_ms={}\tpeak_resident_bytes={}\tdetail={}",
        known(worker.map(|measure| measure.elapsed_ms)),
        known(worker.and_then(|measure| measure.peak_resident_bytes)),
        engine::bounded_detail(detail.unwrap_or("none")),
    );
}

impl SweepSummary {
    fn slowest(&self) -> (&str, u128) {
        self.slowest
            .as_ref()
            .map_or(("none", 0), |(row, ms)| (row.as_str(), *ms))
    }
}

/// Run `audit` over the indexed rows of `hashes` on at most `jobs` workers, streaming each row
/// from the index to a worker and each result to `record` and, every [`CHECKPOINT_ROWS`] rows,
/// to `checkpoint_rows`. The parent never holds the selection's rows: a phase holds at most its
/// channels' rows, one row per feed, and one row per worker until `audit` releases it (a census
/// worker's input is written before its child runs), whatever the selection's size. What the parent
/// has freed goes back to the system when the sweep starts and when it ends, so the parent's memory
/// does not grow from profile to profile.
///
/// Both engines share this schedule. The first phase reads every row once and runs the small
/// ones on every worker; a serialized or large row is dropped after its lane is known (only its
/// hash and size are kept) and read again in the second phase, where one worker drains the
/// serialized lane and then joins the large lane, whose concurrency stays within
/// [`MAX_LARGE_TRANSLATION_JOBS`].
fn sweep_translation_rows<T: SweepRow, R: Send>(
    hashes: &[String],
    jobs: usize,
    label: &str,
    read: &ReadRows<'_, T>,
    audit: &(dyn Fn(T) -> R + Sync),
    record: &mut dyn FnMut(&R, u128),
    checkpoint_rows: &mut dyn FnMut(&[R]) -> Result<(), String>,
) -> Result<SweepSummary, String> {
    // What the work before this sweep freed (a previous profile's report) goes back first.
    worker::alloc::release_free_memory();
    let started = Instant::now();
    let worker_count = jobs.min(hashes.len().max(1));
    let cancelled = AtomicBool::new(false);
    let mut collector = Collector {
        label,
        jobs,
        total: hashes.len(),
        completed: 0,
        slowest: None,
        pending: Vec::with_capacity(CHECKPOINT_ROWS),
        error: None,
        record,
        checkpoint_rows,
    };
    // Small rows are bounded by the large-lane threshold, so a queue per worker keeps every worker
    // fed; the second phase's rows hand off directly.
    let first = Phase {
        feeds: vec![hashes],
        capacity: worker_count,
        plans: vec![vec![0]; worker_count],
        defer: true,
    };
    let (mut read_stats, mut deferred) =
        run_phase(read, &first, audit, &mut collector, &cancelled)?;
    // Each second-phase lane runs largest first.
    deferred.sort_by(|(left_hash, _, left_size), (right_hash, _, right_size)| {
        right_size
            .cmp(left_size)
            .then_with(|| left_hash.cmp(right_hash))
    });
    let lane_hashes = |lane: Lane| {
        deferred
            .iter()
            .filter(|(_, row_lane, _)| *row_lane == lane)
            .map(|(hash, _, _)| hash.clone())
            .collect::<Vec<_>>()
    };
    let (serialized, large) = (lane_hashes(Lane::Serialized), lane_hashes(Lane::Large));
    drop(deferred);
    let (_, serialized_workers, large_workers) =
        translation_phase_worker_counts(jobs, hashes.len(), serialized.len(), large.len());
    let mut second = Phase {
        feeds: Vec::new(),
        capacity: 0,
        plans: Vec::new(),
        defer: false,
    };
    let mut serialized_feed = None;
    let mut large_feed = None;
    if !serialized.is_empty() {
        serialized_feed = Some(second.feeds.len());
        second.feeds.push(serialized.as_slice());
    }
    if !large.is_empty() {
        large_feed = Some(second.feeds.len());
        second.feeds.push(large.as_slice());
    }
    for _ in 0..serialized_workers {
        second
            .plans
            .push(serialized_feed.into_iter().chain(large_feed).collect());
    }
    for _ in 0..large_workers {
        second.plans.push(large_feed.into_iter().collect());
    }
    let reread = serialized.len() + large.len();
    if reread != 0 {
        let (stats, _) = run_phase(read, &second, audit, &mut collector, &cancelled)?;
        read_stats = add_read_stats(read_stats, stats);
    }
    if collector.completed != hashes.len() {
        return Err(format!(
            "{label} workers stopped after {}/{} rows",
            collector.completed,
            hashes.len()
        ));
    }
    Ok(SweepSummary {
        rows: collector.completed,
        wall: started.elapsed(),
        slowest: collector.slowest,
        reread,
        read: read_stats,
        parent: ParentMemory::released_now(),
    })
}

/// A sweep's parent side: it records each result as it arrives and persists every
/// [`CHECKPOINT_ROWS`].
struct Collector<'a, R> {
    label: &'a str,
    jobs: usize,
    total: usize,
    completed: usize,
    slowest: Option<(String, u128)>,
    pending: Vec<R>,
    /// The first persistence failure: it cancels the sweep, and later results are drained unread.
    error: Option<String>,
    record: &'a mut dyn FnMut(&R, u128),
    checkpoint_rows: &'a mut dyn FnMut(&[R]) -> Result<(), String>,
}

impl<R> Collector<'_, R> {
    fn receive(&mut self, air_sha256: String, result: R, elapsed_ms: u128, cancel: &AtomicBool) {
        if self.error.is_some() {
            return;
        }
        self.completed += 1;
        (self.record)(&result, elapsed_ms);
        if self
            .slowest
            .as_ref()
            .is_none_or(|(_, slowest)| elapsed_ms > *slowest)
        {
            self.slowest = Some((air_sha256, elapsed_ms));
        }
        self.pending.push(result);
        if self.pending.len() == CHECKPOINT_ROWS {
            self.flush(cancel);
        }
    }

    /// Persist the pending rows before reporting progress, so a killed or interrupted census
    /// loses at most one short checkpoint and its next resumable selection skips every completed
    /// row. A failure cancels the sweep: workers stop claiming rows and the bounded in-flight
    /// results are drained.
    fn flush(&mut self, cancel: &AtomicBool) {
        if self.pending.is_empty() || self.error.is_some() {
            return;
        }
        match (self.checkpoint_rows)(&self.pending) {
            Ok(()) => eprintln!(
                "# {} progress={}/{} jobs={}",
                self.label, self.completed, self.total, self.jobs
            ),
            Err(error) => {
                cancel.store(true, Ordering::Release);
                self.error = Some(error);
            }
        }
        self.pending.clear();
    }
}

/// A row's hash, lane, and size, kept in place of a second-phase row the first phase read.
type Deferred = (String, Lane, usize);

/// One phase of a sweep: its feeds and which feeds each worker drains.
struct Phase<'h> {
    /// Each feed streams the rows of its hashes, in order, into its own channel.
    feeds: Vec<&'h [String]>,
    /// The rows a feed's channel holds for its workers (0: a direct hand-off).
    capacity: usize,
    /// Per worker, the feeds (indices into `feeds`) it drains, in order.
    plans: Vec<Vec<usize>>,
    /// A feed passes only rows of [`Lane::Small`] and returns the others as [`Deferred`].
    defer: bool,
}

/// Run `phase`, collecting on the calling thread. Only the workers hold a channel's receiver, so
/// a feed stops when its workers do. Returns the summed read statistics and the deferred rows.
fn run_phase<T: SweepRow, R: Send>(
    read: &ReadRows<'_, T>,
    phase: &Phase<'_>,
    audit: &(dyn Fn(T) -> R + Sync),
    collector: &mut Collector<'_, R>,
    cancelled: &AtomicBool,
) -> Result<
    (
        metal2vulkan_validation::source::IndexedSourceReadStats,
        Vec<Deferred>,
    ),
    String,
> {
    let Phase {
        feeds,
        capacity,
        plans,
        defer,
    } = phase;
    let (capacity, defer) = (*capacity, *defer);
    std::thread::scope(|scope| {
        // A stalled checkpoint applies backpressure instead of retaining a completed-row backlog.
        let (result_sender, result_receiver) = bounded_worker_channel(plans.len());
        let mut receivers = Vec::with_capacity(feeds.len());
        let mut producers = Vec::with_capacity(feeds.len());
        for &hashes in feeds.iter() {
            let (row_sender, row_receiver) = mpsc::sync_channel::<T>(capacity);
            receivers.push(Arc::new(Mutex::new(row_receiver)));
            producers.push(scope.spawn(move || {
                let mut deferred = Vec::new();
                let read = read(hashes, &mut |row| {
                    if cancelled.load(Ordering::Acquire) {
                        return Err("the sweep was cancelled".to_string());
                    }
                    if defer && row.lane() != Lane::Small {
                        deferred.push((row.key().into_owned(), row.lane(), row.size()));
                        return Ok(());
                    }
                    row_sender
                        .send(row)
                        .map_err(|_| "translation workers stopped".to_string())
                });
                read.map(|stats| (stats, deferred))
            }));
        }
        for plan in plans {
            let lanes: Vec<_> = plan
                .iter()
                .map(|&feed| Arc::clone(&receivers[feed]))
                .collect();
            let result_sender = result_sender.clone();
            scope.spawn(move || {
                for lane in &lanes {
                    loop {
                        if cancelled.load(Ordering::Acquire) {
                            return;
                        }
                        let next = match lane.lock() {
                            Ok(receiver) => receiver.recv(),
                            Err(_) => return,
                        };
                        // A closed, drained channel ends this lane.
                        let Ok(row) = next else {
                            break;
                        };
                        let air_sha256 = row.key().into_owned();
                        let started = Instant::now();
                        let result = audit(row);
                        let elapsed_ms = started.elapsed().as_millis();
                        if result_sender
                            .send((air_sha256, result, elapsed_ms))
                            .is_err()
                        {
                            return;
                        }
                    }
                }
            });
        }
        drop(receivers);
        drop(result_sender);
        for (air_sha256, result, elapsed_ms) in result_receiver {
            collector.receive(air_sha256, result, elapsed_ms, cancelled);
        }
        collector.flush(cancelled);
        let mut stats = metal2vulkan_validation::source::IndexedSourceReadStats::default();
        let mut deferred = Vec::new();
        let mut read_error = None;
        for producer in producers {
            match producer.join() {
                Ok(Ok((read, rows))) => {
                    stats = add_read_stats(stats, read);
                    deferred.extend(rows);
                }
                Ok(Err(error)) => {
                    read_error.get_or_insert(error);
                }
                Err(_) => {
                    read_error.get_or_insert_with(|| "the row reader panicked".to_string());
                }
            }
        }
        // A persistence failure caused any cancellation a feed reports, so it is the error.
        match (collector.error.clone(), read_error) {
            (Some(error), _) | (None, Some(error)) => Err(error),
            (None, None) => Ok((stats, deferred)),
        }
    })
}

fn add_read_stats(
    left: metal2vulkan_validation::source::IndexedSourceReadStats,
    right: metal2vulkan_validation::source::IndexedSourceReadStats,
) -> metal2vulkan_validation::source::IndexedSourceReadStats {
    metal2vulkan_validation::source::IndexedSourceReadStats {
        rows: left.rows + right.rows,
        source_shards_opened: left.source_shards_opened + right.source_shards_opened,
        source_bytes_read: left.source_bytes_read + right.source_bytes_read,
        repair_shards_scanned: left.repair_shards_scanned + right.repair_shards_scanned,
        repair_bytes_scanned: left.repair_bytes_scanned + right.repair_bytes_scanned,
    }
}

fn read_hash_file(path: &std::path::Path) -> Result<Vec<String>, String> {
    let contents = fs::read_to_string(path)
        .map_err(|error| format!("read hash file {}: {error}", path.display()))?;
    let mut hashes = std::collections::BTreeSet::new();
    for (line_index, line) in contents.lines().enumerate() {
        let Some(hash) = line.split_whitespace().next() else {
            continue;
        };
        if hash.len() != 64
            || !hash
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(format!(
                "{}:{} first field must be a lowercase SHA-256",
                path.display(),
                line_index + 1
            ));
        }
        hashes.insert(hash.to_string());
    }
    if hashes.is_empty() {
        return Err(format!("hash file {} is empty", path.display()));
    }
    Ok(hashes.into_iter().collect())
}

fn translation_phase_worker_counts(
    jobs: usize,
    source_count: usize,
    serialized_count: usize,
    large_count: usize,
) -> (usize, usize, usize) {
    let worker_count = jobs.min(source_count.max(1));
    let serialized_worker_count = usize::from(serialized_count != 0 && worker_count != 0);
    let large_worker_count = MAX_LARGE_TRANSLATION_JOBS
        .saturating_sub(serialized_worker_count)
        .min(large_count)
        .min(worker_count.saturating_sub(serialized_worker_count));
    (worker_count, serialized_worker_count, large_worker_count)
}

fn bounded_worker_channel<T>(worker_count: usize) -> (mpsc::SyncSender<T>, mpsc::Receiver<T>) {
    mpsc::sync_channel(worker_count)
}

fn is_costly_translation_source(source: &metal2vulkan_validation::source::SourceRow) -> bool {
    source.air_ll.len() >= LARGE_TRANSLATION_SOURCE_BYTES
        || source.air_ll.contains("air.visible_function_table")
            && (source.air_ll.contains("inttoptr") || source.air_ll.contains("ptrtoint"))
}

fn is_serialized_cost_translation_source(
    source: &metal2vulkan_validation::source::SourceRow,
) -> bool {
    let (block_count, call_count) = translation_cfg_counts(&source.air_ll);
    source.air_ll.len() <= SERIALIZED_TRANSLATION_MAX_BYTES
        && block_count.saturating_mul(call_count) >= SERIALIZED_TRANSLATION_CFG_CALL_WORK
}

fn translation_cfg_counts(air_ll: &str) -> (usize, usize) {
    air_ll.lines().fold((0, 0), |(blocks, calls), line| {
        let is_block = line
            .split_whitespace()
            .next()
            .and_then(|head| head.strip_suffix(':'))
            .is_some_and(|name| {
                !name.is_empty()
                    && name.chars().all(|character| {
                        character.is_ascii_alphanumeric() || "_.".contains(character)
                    })
            });
        let trimmed = line.trim_start();
        let is_call = trimmed.starts_with("call ") || trimmed.contains(" call ");
        (blocks + usize::from(is_block), calls + usize::from(is_call))
    })
}

fn write_translation_worker_input(
    destination: impl Write,
    input: &impl serde::Serialize,
) -> Result<(), String> {
    let mut writer = BufWriter::new(destination);
    serde_json::to_writer(&mut writer, input)
        .map_err(|error| format!("encode translation worker input: {error}"))?;
    writer
        .flush()
        .map_err(|error| format!("flush translation worker input: {error}"))
}

/// What a `--translation-worker` child runs on its row.
#[derive(Clone, Copy)]
enum WorkerLane<'a> {
    /// The engine for a committed profile.
    Engine(&'a str),
    /// The specialization audit's planning of one row.
    SpecializationPlan,
    /// One specialization attempt for a committed profile.
    SpecializationSet(&'a str),
}

/// Run one row in a killable `--translation-worker` child under the 20-second and 500-MiB
/// ceilings and return its decoded result. `lane` selects what the child runs. `write_input`
/// writes the child's input and owns the row, which is released once the input is written,
/// before the child runs.
fn run_worker_process<R: serde::de::DeserializeOwned>(
    root: &std::path::Path,
    index: &std::path::Path,
    lane: WorkerLane<'_>,
    write_input: impl FnOnce(fs::File) -> Result<(), String>,
) -> Result<worker::WorkerRun<R>, WorkerFailure> {
    let scratch = ScratchDir::new("translation-audit-worker").map_err(WorkerFailure::Setup)?;
    let translation_tmp = scratch.path().join("translation");
    fs::create_dir(&translation_tmp).map_err(|error| {
        WorkerFailure::Setup(format!("create {}: {error}", translation_tmp.display()))
    })?;
    let executable = std::env::current_exe().map_err(|error| {
        WorkerFailure::Setup(format!("resolve corpus-triage executable: {error}"))
    })?;
    let mut command = Command::new(executable);
    command
        .arg("--translation-worker")
        .arg(&translation_tmp)
        .arg(root)
        .arg(index)
        // Every temporary file the worker and its tools create (`llvm-as`, `spirv-val`) lands in
        // this parent-owned scratch, which is removed on every exit path, a killed worker's too.
        .env("TMPDIR", &translation_tmp);
    match lane {
        WorkerLane::Engine(profile) => {
            command.arg(profile);
        }
        WorkerLane::SpecializationPlan => {
            command.arg(corpus_triage_specialization::PLAN_WORKER);
        }
        WorkerLane::SpecializationSet(profile) => {
            command
                .arg(corpus_triage_specialization::SET_WORKER)
                .arg(profile);
        }
    }
    worker::run_json(command, "translation-audit-worker-io", write_input)
}

/// An engine census row as the parent holds it: its lane, decided from its AIR text as it is
/// read, and only what the worker consumes ([`engine::RowInput`]). A row with retained bitcode
/// drops its text there, so the parent holds its blob alone.
struct EngineRow {
    input: engine::RowInput<'static>,
    lane: Lane,
    size: usize,
}

impl From<metal2vulkan_validation::source::SourceRow> for EngineRow {
    fn from(row: metal2vulkan_validation::source::SourceRow) -> EngineRow {
        let (lane, size) = (row.lane(), row.size());
        EngineRow {
            input: engine::RowInput::from(row),
            lane,
            size,
        }
    }
}

impl SweepRow for EngineRow {
    fn key(&self) -> std::borrow::Cow<'_, str> {
        std::borrow::Cow::Borrowed(&self.input.air_sha256)
    }

    fn lane(&self) -> Lane {
        self.lane
    }

    fn size(&self) -> usize {
        self.size
    }
}

/// One engine row through the killable worker; a worker-boundary stop is a typed outcome.
fn audit_engine_source(
    root: &std::path::Path,
    index: &std::path::Path,
    row: EngineRow,
    profile: &str,
) -> EngineAuditResult {
    let air_sha256 = row.input.air_sha256.to_string();
    // A stop at the worker boundary keeps what the parent measured up to it, so the time and
    // memory gates see the rows that breach the ceilings.
    let failed =
        |outcome: Outcome, detail: String, measured: Option<worker::Measured>| EngineAuditResult {
            air_sha256: air_sha256.clone(),
            attempt: engine::Attempt {
                outcome,
                detail: Some(engine::bounded_detail(&detail)),
                words: None,
                cost: None,
            },
            worker: measured.map(WorkerMeasure::from),
        };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_worker_process::<EngineAuditResult>(
            root,
            index,
            WorkerLane::Engine(profile),
            move |stdin| write_translation_worker_input(stdin, &row.input),
        )
    })) {
        Ok(Ok(run)) if run.result.air_sha256 == air_sha256 => EngineAuditResult {
            worker: Some(WorkerMeasure::from(run.measured())),
            ..run.result
        },
        Ok(Ok(run)) => failed(
            Outcome::Failed {
                failure: FailureKind::WorkerExit,
            },
            format!(
                "worker reported row {} for {air_sha256}",
                run.result.air_sha256
            ),
            Some(run.measured()),
        ),
        Ok(Err(failure)) => failed(failure.outcome(), failure.message(), failure.measured()),
        Err(_) => failed(
            Outcome::Failed {
                failure: FailureKind::Tooling,
            },
            "translation audit parent panicked while supervising the worker".into(),
            None,
        ),
    }
}

fn run_translation_worker() -> Result<(), String> {
    metal2vulkan_validation::source::adopt_parent_private_corpus()?;
    worker::alloc::enable_budget(worker::MEMORY_LIMIT_BYTES);
    let started = Instant::now();
    // Argument 2 is the parent's scratch, which it made the worker's TMPDIR: the tools the worker
    // runs (`llvm-as`, `spirv-val`) find it there.
    let corpus_root = std::env::args_os()
        .nth(3)
        .map(PathBuf::from)
        .ok_or("translation worker corpus root is unavailable")?;
    let index = std::env::args_os()
        .nth(4)
        .map(PathBuf::from)
        .ok_or("translation worker index path is unavailable")?;
    match std::env::args().nth(5).as_deref() {
        Some(corpus_triage_specialization::PLAN_WORKER) => {
            return corpus_triage_specialization::plan_worker(started)
        }
        Some(corpus_triage_specialization::SET_WORKER) => {
            let profile = std::env::args()
                .nth(6)
                .ok_or("specialization worker profile is unavailable")?;
            return corpus_triage_specialization::set_worker(
                &corpus_root,
                &index,
                &profile,
                started,
            );
        }
        _ => {}
    }
    let profile = std::env::args()
        .nth(5)
        .ok_or("translation worker profile is unavailable")?;
    let input: engine::RowInput<'static> = serde_json::from_reader(std::io::stdin().lock())
        .map_err(|error| format!("decode translation worker input: {error}"))?;
    let profile = engine::load_profile(&profile)?;
    let (result, pending) =
        audit_engine_row(input, &profile, started.checked_add(worker::DEADLINE));
    match pending {
        // `spirv-val` runs in a fresh image of this executable, which reports.
        Some(pending) => worker::l3::report_after_l3(ENGINE_STAGE, &result, pending),
        None => worker::report(&result),
    }
}

/// The L3 stage of an engine translation worker ([`worker::l3`]).
const ENGINE_STAGE: &str = "engine";

/// The L3 stage a translation worker's image was replaced with: `--l3-stage STAGE PENDING`.
fn run_l3_stage() -> Result<(), String> {
    let stage = std::env::args().nth(2).ok_or("the L3 stage is unnamed")?;
    let pending = std::env::args_os()
        .nth(3)
        .map(PathBuf::from)
        .ok_or("the L3 stage's pending file is unavailable")?;
    match stage.as_str() {
        ENGINE_STAGE => worker::l3::serve(&pending, |result: &mut EngineAuditResult| {
            &mut result.attempt
        }),
        corpus_triage_specialization::SET_STAGE => {
            corpus_triage_specialization::set_stage(&pending)
        }
        other => Err(format!("no L3 stage {other}")),
    }
}

/// The engine on one row inside the worker up to L3's `spirv-val`
/// ([`engine::translate_row_unvalidated`]): its retained bitcode, or a public row's AIR text
/// assembled with `llvm-as`. Text `llvm-as` rejects is a recorded `no_bitcode`; a retained blob
/// that is not base64 (index corruption) and a missing `llvm-as` are transient `tooling`, reported
/// and never recorded as the row's outcome.
fn audit_engine_row(
    input: engine::RowInput<'_>,
    profile: &metal2vulkan::device::DeviceProfile,
    deadline: Option<Instant>,
) -> (EngineAuditResult, Option<engine::PendingL3>) {
    let (attempt, pending) =
        engine::translate_row_unvalidated(&input.source, &input.entry, profile, deadline);
    let result = EngineAuditResult {
        attempt,
        air_sha256: input.air_sha256.into_owned(),
        worker: None,
    };
    (result, pending)
}

/// Select up to `limit` indexed sources after `after`, in hash order, without writing the index.
fn select_sources_after(
    index: &std::path::Path,
    after: Option<&str>,
    limit: usize,
) -> Result<Vec<String>, String> {
    let connection = rusqlite::Connection::open_with_flags(
        index,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|error| format!("open index {}: {error}", index.display()))?;
    let mut statement = connection
        .prepare(
            "SELECT air_sha256 FROM sources WHERE (?1 IS NULL OR air_sha256 > ?1) \
             ORDER BY air_sha256 LIMIT ?2",
        )
        .map_err(|error| format!("prepare bitcode-reader selection: {error}"))?;
    let hashes = statement
        .query_map(rusqlite::params![after, limit as i64], |row| {
            row.get::<_, String>(0)
        })
        .map_err(|error| format!("query bitcode-reader selection: {error}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("read bitcode-reader selection: {error}"))?;
    Ok(hashes)
}

/// The rows `--audit bitcode-reader` selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReaderSelection {
    /// The indexed sources (entry modules), by AIR hash.
    Sources,
    /// The indexed library modules (`--library-modules`): the linked dependencies harvest keeps
    /// apart from the sources, by module hash.
    LibraryModules,
}

impl ReaderSelection {
    fn name(self) -> &'static str {
        match self {
            ReaderSelection::Sources => "sources",
            ReaderSelection::LibraryModules => "library-modules",
        }
    }

    /// The summary's keys for the shards the selected rows were read from and their bytes.
    fn read_keys(self) -> (&'static str, &'static str) {
        match self {
            ReaderSelection::Sources => ("source_shards_opened", "source_bytes_read"),
            ReaderSelection::LibraryModules => {
                ("library_module_shards_opened", "library_module_bytes_read")
            }
        }
    }
}

/// `--audit bitcode-reader`: decode each selected row's retained bitcode with the engine's
/// reader under the per-attempt meter (20 s deadline, ledger cap), `jobs` rows at a time, each in
/// its own killable worker of this executable (`bitcode_audit::audit_row_in_worker`), and compare
/// its structural fingerprint with `llvm-dis` text of the same bytes. Rows stream through a bounded
/// queue; no result is written to the index. Resumable with `--after LAST`, over the sources or the
/// library modules (`selection`). The library modules' selection syncs the library-module index
/// first (`select_current_library_modules`), since no other lane this audit relies on keeps it
/// current.
fn audit_bitcode_reader(
    root: &std::path::Path,
    index: &std::path::Path,
    selection: ReaderSelection,
    after: Option<&str>,
    limit: usize,
    jobs: usize,
    summary_only: bool,
) -> Result<(), String> {
    use metal2vulkan_validation::bitcode_audit::{audit_row_in_worker, Outcome, RowResult};
    use metal2vulkan_validation::bitcode_audit::{peak_rss_bytes, resolve_llvm_dis};
    use metal2vulkan_validation::library_module::{
        for_each_indexed_library_module, select_current_library_modules,
    };

    let llvm_dis = resolve_llvm_dis()?;
    let program = std::env::current_exe().map_err(|error| {
        format!("resolve the bitcode-reader audit's worker executable: {error}")
    })?;
    let started = Instant::now();
    let hashes = match selection {
        ReaderSelection::Sources => select_sources_after(index, after, limit)?,
        ReaderSelection::LibraryModules => {
            let (synced, hashes) = select_current_library_modules(root, index, after, limit)?;
            println!(
                "bitcode-reader-library-modules\tshards_scanned={}\tbytes_scanned={}",
                synced.shards_scanned, synced.bytes_scanned
            );
            hashes
        }
    };
    let mut results: Vec<(String, RowResult)> = Vec::with_capacity(hashes.len());
    let read = parallel(
        jobs,
        jobs.saturating_mul(2),
        |send| match selection {
            ReaderSelection::Sources => {
                metal2vulkan_validation::source::for_each_indexed_source_with_blob(
                    root,
                    index,
                    &hashes,
                    |source| send((source.key().into_owned(), source.blob_b64)),
                )
                .map(|read| (read.source_shards_opened, read.source_bytes_read))
            }
            // A library module always retains its blob (`LibraryModuleRow::validate`).
            ReaderSelection::LibraryModules => {
                for_each_indexed_library_module(root, index, &hashes, |module| {
                    send((module.module_sha256, Some(module.blob_b64)))
                })
                .map(|read| (read.shards_opened, read.bytes_read))
            }
        },
        |(hash, blob): (String, Option<String>)| {
            (
                hash,
                audit_row_in_worker(&program, blob.as_deref(), &llvm_dis),
            )
        },
        |result| {
            results.push(result);
            Ok(())
        },
    )?;
    results.sort_by(|a, b| a.0.cmp(&b.0));
    let mut shapes = BTreeMap::<String, usize>::new();
    let (mut matched, mut mismatched, mut refused, mut dis_failed) = (0, 0, 0, 0);
    let (mut no_blob, mut invalid_blob) = (0, 0);
    let mut workers = worker::audit::Tally::default();
    let mut slowest: Option<(&str, Duration)> = None;
    let (mut peak_ledger, mut peak_work) = (0u64, 0u64);
    let (mut air_calls, mut llvm_calls, mut unknown_calls) = (0usize, 0usize, 0usize);
    // Unknown intrinsic name -> (rows, calls); the same for calls to declared, undefined
    // functions that are no intrinsics (what each module needs linked).
    let mut unknown = BTreeMap::<String, (usize, usize)>::new();
    let mut external = BTreeMap::<String, (usize, usize)>::new();
    let mut external_calls = 0usize;
    for (hash, r) in &results {
        if let Some(census) = &r.census {
            air_calls += census.air;
            llvm_calls += census.llvm;
            for (name, &calls) in &census.unknown {
                unknown_calls += calls;
                let entry = unknown.entry(name.clone()).or_default();
                entry.0 += 1;
                entry.1 += calls;
            }
            for (name, &calls) in &census.external {
                external_calls += calls;
                let entry = external.entry(name.clone()).or_default();
                entry.0 += 1;
                entry.1 += calls;
            }
        }
        match &r.outcome {
            Outcome::Match => matched += 1,
            Outcome::Mismatch(_) => mismatched += 1,
            Outcome::Refused { .. } => refused += 1,
            Outcome::DisassemblyFailed(_) => dis_failed += 1,
            Outcome::NoBlob => no_blob += 1,
            Outcome::InvalidBlob(_) => invalid_blob += 1,
            Outcome::Stopped { .. } => {}
        }
        workers.record(r.worker, matches!(r.outcome, Outcome::Stopped { .. }));
        let shape = r.outcome.shape();
        if !summary_only {
            println!(
                "bitcode-reader\t{hash}\t{}\t{}",
                r.decode.as_millis(),
                shape
            );
            // Which rows need which functions linked, to join with their translation outcomes.
            for (name, calls) in r.census.iter().flat_map(|census| &census.external) {
                println!("bitcode-reader-external-row\t{hash}\t{calls}\t{name}");
            }
        }
        *shapes.entry(shape).or_default() += 1;
        if slowest.is_none_or(|(_, d)| r.decode > d) {
            slowest = Some((hash, r.decode));
        }
        peak_ledger = peak_ledger.max(r.ledger_peak);
        peak_work = peak_work.max(r.work);
    }
    let (shards_key, bytes_key) = selection.read_keys();
    println!(
        "bitcode-reader-summary\tselection={}\tselected={}\tmatched={matched}\tmismatched={mismatched}\trefused={refused}\tllvm_dis_failed={dis_failed}\tno_blob={no_blob}\tinvalid_blob={invalid_blob}\t{}\tslowest_air_sha256={}\tslowest_decode_ms={}\tpeak_ledger_bytes={peak_ledger}\tpeak_work_units={peak_work}\tprocess_peak_rss_bytes={}\t{shards_key}={}\t{bytes_key}={}\tjobs={jobs}\twall_ms={}\tfirst={}\tlast={}",
        selection.name(),
        hashes.len(),
        workers.fields(),
        slowest.map_or("none", |(h, _)| h),
        slowest.map_or(0, |(_, d)| d.as_millis()),
        peak_rss_bytes(),
        read.0,
        read.1,
        started.elapsed().as_millis(),
        hashes.first().map_or("none", String::as_str),
        hashes.last().map_or("none", String::as_str),
    );
    for (shape, count) in shapes {
        println!("bitcode-reader-shape\t{count}\t{shape}");
    }
    let listed = |name: &str| metal2vulkan_air::intrinsics::undecoded_family(name).is_some();
    println!(
        "bitcode-reader-intrinsics\tair_calls={air_calls}\tllvm_calls={llvm_calls}\tunknown_calls={unknown_calls}\tunknown_names={}\tunlisted_names={}",
        unknown.len(),
        unknown.keys().filter(|name| !listed(name)).count()
    );
    for (name, (rows, calls)) in unknown {
        let status = if listed(&name) { "listed" } else { "unlisted" };
        println!("bitcode-reader-unknown-intrinsic\t{rows}\t{calls}\t{status}\t{name}");
    }
    println!(
        "bitcode-reader-external\texternal_calls={external_calls}\texternal_names={}",
        external.len()
    );
    for (name, (rows, calls)) in external {
        println!("bitcode-reader-external-call\t{rows}\t{calls}\t{name}");
    }
    Ok(())
}

/// The focused audit of rasterization-rate maps: the rows whose cached facts describe
/// the family ([`AuditTarget::RasterizationRateMaps`], independent of any requirement), each
/// reclassified from its source with where its coordinate-mapping calls read their maps, and how
/// many carry a tooling requirement (`rasterization_rate_map_placement` among them).
fn audit_rasterization_rate_maps(
    root: &std::path::Path,
    index: &std::path::Path,
    after: Option<&str>,
    limit: usize,
    summary_only: bool,
) -> Result<(), String> {
    let hashes =
        select_cached_audit_target_after(index, AuditTarget::RasterizationRateMaps, after, limit)?;
    let mut shapes = BTreeMap::<String, usize>::new();
    let mut requirements = BTreeMap::<String, usize>::new();
    let mut required_sources = 0usize;
    for_each_indexed_source_analysis(root, index, &hashes, |source| {
        let shape = metal2vulkan_validation::triage::audit_rasterization_rate_map(&source)?
            .ok_or_else(|| {
                format!(
                    "{} is described as mapping coordinates through a rate map but calls none",
                    source.key()
                )
            })?
            .shape();
        let classified = classify(&source)?;
        required_sources += usize::from(!classified.tooling_requirements.is_empty());
        for requirement in &classified.tooling_requirements {
            *requirements.entry(requirement.to_string()).or_default() += 1;
        }
        if !summary_only {
            let names = classified
                .tooling_requirements
                .iter()
                .map(|requirement| requirement.as_str())
                .collect::<Vec<_>>();
            println!(
                "rate-map\t{}\t{shape}\trequirements={}",
                source.key(),
                if names.is_empty() {
                    "none".into()
                } else {
                    names.join(",")
                }
            );
        }
        *shapes.entry(shape).or_default() += 1;
        Ok(())
    })?;
    println!(
        "rate-map-summary\tselected={}\tfirst={}\tlast={}\trequired_sources={}\tshapes={}",
        hashes.len(),
        hashes.first().map_or("none", String::as_str),
        hashes.last().map_or("none", String::as_str),
        required_sources,
        shapes.len()
    );
    for (requirement, count) in requirements {
        println!("rate-map-requirement\t{requirement}\t{count}");
    }
    for (shape, count) in shapes {
        println!("rate-map-shape\t{count}\t{shape}");
    }
    Ok(())
}

/// The focused audit of sample-rate features: the rows whose cached facts describe the family
/// ([`AuditTarget::SampleRate`], independent of any requirement), each reclassified from its source
/// with the features it uses, and how many carry a tooling requirement.
fn audit_sample_rate(
    root: &std::path::Path,
    index: &std::path::Path,
    after: Option<&str>,
    limit: usize,
    summary_only: bool,
) -> Result<(), String> {
    let hashes = select_cached_audit_target_after(index, AuditTarget::SampleRate, after, limit)?;
    let mut shapes = BTreeMap::<String, usize>::new();
    let mut requirements = BTreeMap::<String, usize>::new();
    let mut required_sources = 0usize;
    for_each_indexed_source_analysis(root, index, &hashes, |source| {
        let classified = classify(&source)?;
        if classified.sample_rate.is_empty() {
            return Err(format!(
                "{} is described as using a sample-rate feature but uses none",
                source.key()
            ));
        }
        let shape = classified
            .sample_rate
            .iter()
            .map(|feature| {
                serde_json::to_value(feature)
                    .ok()
                    .and_then(|value| value.as_str().map(str::to_owned))
                    .unwrap_or_else(|| format!("{feature:?}"))
            })
            .collect::<Vec<_>>()
            .join("+");
        required_sources += usize::from(!classified.tooling_requirements.is_empty());
        for requirement in &classified.tooling_requirements {
            *requirements.entry(requirement.to_string()).or_default() += 1;
        }
        if !summary_only {
            let names = classified
                .tooling_requirements
                .iter()
                .map(|requirement| requirement.as_str())
                .collect::<Vec<_>>();
            println!(
                "sample-rate\t{}\t{}\t{shape}\trequirements={}",
                source.key(),
                source.stage,
                if names.is_empty() {
                    "none".into()
                } else {
                    names.join(",")
                }
            );
        }
        *shapes.entry(shape).or_default() += 1;
        Ok(())
    })?;
    println!(
        "sample-rate-summary\tselected={}\tfirst={}\tlast={}\trequired_sources={}\tshapes={}",
        hashes.len(),
        hashes.first().map_or("none", String::as_str),
        hashes.last().map_or("none", String::as_str),
        required_sources,
        shapes.len()
    );
    for (requirement, count) in requirements {
        println!("sample-rate-requirement\t{requirement}\t{count}");
    }
    for (shape, count) in shapes {
        println!("sample-rate-shape\t{count}\t{shape}");
    }
    Ok(())
}

/// The focused audit of Metal's `ulong` textures: the rows whose cached facts describe the family
/// ([`AuditTarget::UlongTextures`], independent of any requirement), each reclassified from its
/// source with its `ulong` texture operations, whether it is a public fixture's row or a private
/// one, and its requirements.
fn audit_ulong_textures(
    root: &std::path::Path,
    index: &std::path::Path,
    after: Option<&str>,
    limit: usize,
    summary_only: bool,
) -> Result<(), String> {
    let hashes = select_cached_audit_target_after(index, AuditTarget::UlongTextures, after, limit)?;
    let mut operations = BTreeMap::<String, usize>::new();
    let mut requirements = BTreeMap::<String, usize>::new();
    let (mut public_rows, mut private_rows) = (0usize, 0usize);
    for_each_indexed_source_analysis(root, index, &hashes, |source| {
        let classified = classify(&source)?;
        let calls =
            metal2vulkan_validation::executor_contract::ulong_texture_calls(&classified.air_calls)
                .collect::<Vec<_>>();
        if !classified.ulong_texture || calls.is_empty() {
            return Err(format!(
                "{} is described as operating on a ulong texture but calls no ulong texture \
                 operation",
                source.key()
            ));
        }
        let public = metal2vulkan_validation::source::public_source_path(&source.label).is_some();
        if public {
            public_rows += 1;
        } else {
            private_rows += 1;
        }
        for call in &calls {
            *operations.entry((*call).to_owned()).or_default() += 1;
        }
        for requirement in &classified.tooling_requirements {
            *requirements.entry(requirement.to_string()).or_default() += 1;
        }
        if !summary_only {
            let names = classified
                .tooling_requirements
                .iter()
                .map(|requirement| requirement.as_str())
                .collect::<Vec<_>>();
            println!(
                "ulong-texture\t{}\t{}\t{}\t{}\toperations={}\trequirements={}",
                source.key(),
                source.stage,
                source.entry,
                if public {
                    source.label.as_str()
                } else {
                    "private"
                },
                calls.len(),
                if names.is_empty() {
                    "none".into()
                } else {
                    names.join(",")
                }
            );
        }
        Ok(())
    })?;
    println!(
        "ulong-texture-summary\tselected={}\tpublic={public_rows}\tprivate={private_rows}\tfirst={}\tlast={}",
        hashes.len(),
        hashes.first().map_or("none", String::as_str),
        hashes.last().map_or("none", String::as_str),
    );
    for (requirement, count) in requirements {
        println!("ulong-texture-requirement\t{requirement}\t{count}");
    }
    for (operation, count) in operations {
        println!("ulong-texture-operation\t{count}\t{operation}");
    }
    Ok(())
}

/// The focused audit of fragment functions' varyings: the rows whose cached facts describe the
/// family ([`AuditTarget::FragmentVaryings`], independent of any requirement), each reclassified
/// from its source with the declared types of the varyings it reads, which a case gives authored
/// values (`varying_values`) or the passthrough's defaults, whether it is a public fixture's row or
/// a private one, and its requirements.
fn audit_fragment_varyings(
    root: &std::path::Path,
    index: &std::path::Path,
    after: Option<&str>,
    limit: usize,
    summary_only: bool,
) -> Result<(), String> {
    let hashes =
        select_cached_audit_target_after(index, AuditTarget::FragmentVaryings, after, limit)?;
    let mut types = BTreeMap::<String, usize>::new();
    let mut requirements = BTreeMap::<String, usize>::new();
    let (mut public_rows, mut private_rows) = (0usize, 0usize);
    for_each_indexed_source_analysis(root, index, &hashes, |source| {
        let classified = classify(&source)?;
        if classified.fragment_varyings.is_empty() {
            return Err(format!(
                "{} is described as a fragment function that reads varyings but reads none",
                source.key()
            ));
        }
        let public = metal2vulkan_validation::source::public_source_path(&source.label).is_some();
        if public {
            public_rows += 1;
        } else {
            private_rows += 1;
        }
        for ty in &classified.fragment_varyings {
            *types.entry(ty.clone()).or_default() += 1;
        }
        for requirement in &classified.tooling_requirements {
            *requirements.entry(requirement.to_string()).or_default() += 1;
        }
        if !summary_only {
            let names = classified
                .tooling_requirements
                .iter()
                .map(|requirement| requirement.as_str())
                .collect::<Vec<_>>();
            println!(
                "fragment-varyings\t{}\t{}\t{}\ttypes={}\trequirements={}",
                source.key(),
                source.entry,
                if public {
                    source.label.as_str()
                } else {
                    "private"
                },
                classified.fragment_varyings.join(","),
                if names.is_empty() {
                    "none".into()
                } else {
                    names.join(",")
                }
            );
        }
        Ok(())
    })?;
    println!(
        "fragment-varyings-summary\tselected={}\tpublic={public_rows}\tprivate={private_rows}\tfirst={}\tlast={}",
        hashes.len(),
        hashes.first().map_or("none", String::as_str),
        hashes.last().map_or("none", String::as_str),
    );
    for (requirement, count) in requirements {
        println!("fragment-varyings-requirement\t{requirement}\t{count}");
    }
    for (ty, count) in types {
        println!("fragment-varyings-type\t{count}\t{ty}");
    }
    Ok(())
}

/// The focused audit of tile functions: the rows whose cached facts describe the family
/// ([`AuditTarget::TileFunctions`], independent of any requirement), each reclassified from its
/// source with the implicit-imageblock operations it calls (each texel type names the plane format
/// a case authors its render target in), whether it is a public fixture's row or a private one,
/// and its requirements.
fn audit_tile_functions(
    root: &std::path::Path,
    index: &std::path::Path,
    after: Option<&str>,
    limit: usize,
    summary_only: bool,
) -> Result<(), String> {
    let hashes = select_cached_audit_target_after(index, AuditTarget::TileFunctions, after, limit)?;
    let mut operations = BTreeMap::<String, usize>::new();
    let mut requirements = BTreeMap::<String, usize>::new();
    let (mut public_rows, mut private_rows) = (0usize, 0usize);
    for_each_indexed_source_analysis(root, index, &hashes, |source| {
        let classified = classify(&source)?;
        if !classified.tile_function {
            return Err(format!(
                "{} is described as a tile function but has no implicit imageblock attachment",
                source.key()
            ));
        }
        let calls = classified
            .air_calls
            .keys()
            .filter(|name| name.contains(".implicit_imageblock."))
            .collect::<Vec<_>>();
        let public = metal2vulkan_validation::source::public_source_path(&source.label).is_some();
        if public {
            public_rows += 1;
        } else {
            private_rows += 1;
        }
        for call in &calls {
            *operations.entry((*call).clone()).or_default() += 1;
        }
        for requirement in &classified.tooling_requirements {
            *requirements.entry(requirement.to_string()).or_default() += 1;
        }
        if !summary_only {
            let names = classified
                .tooling_requirements
                .iter()
                .map(|requirement| requirement.as_str())
                .collect::<Vec<_>>();
            println!(
                "tile-function\t{}\t{}\t{}\toperations={}\trequirements={}",
                source.key(),
                source.entry,
                if public {
                    source.label.as_str()
                } else {
                    "private"
                },
                calls.len(),
                if names.is_empty() {
                    "none".into()
                } else {
                    names.join(",")
                }
            );
        }
        Ok(())
    })?;
    println!(
        "tile-function-summary\tselected={}\tpublic={public_rows}\tprivate={private_rows}\tfirst={}\tlast={}",
        hashes.len(),
        hashes.first().map_or("none", String::as_str),
        hashes.last().map_or("none", String::as_str),
    );
    for (requirement, count) in requirements {
        println!("tile-function-requirement\t{requirement}\t{count}");
    }
    for (operation, count) in operations {
        println!("tile-function-operation\t{count}\t{operation}");
    }
    Ok(())
}

/// The focused audit of object and mesh functions: the rows whose cached facts describe the family
/// ([`AuditTarget::MeshDraws`], independent of any requirement), each reclassified from its source
/// with its stage, its payload, the stages a row of it needs that nothing supplies (which generated
/// stage would let a case draw it), whether it is a public fixture's row or a private one, and its
/// requirements. The summary counts the rows by stage, by the stages they need, and by the set of
/// stages each needs.
fn audit_mesh_draws(
    root: &std::path::Path,
    index: &std::path::Path,
    after: Option<&str>,
    limit: usize,
    summary_only: bool,
) -> Result<(), String> {
    let hashes = select_cached_audit_target_after(index, AuditTarget::MeshDraws, after, limit)?;
    let mut stages = BTreeMap::<String, usize>::new();
    let mut missing = BTreeMap::<String, usize>::new();
    let mut needs = BTreeMap::<String, usize>::new();
    let mut requirements = BTreeMap::<String, usize>::new();
    let (mut public_rows, mut private_rows) = (0usize, 0usize);
    let names = |stages: &[metal2vulkan_validation::case::Stage]| {
        if stages.is_empty() {
            "none".to_string()
        } else {
            stages
                .iter()
                .map(|stage| stage.entry_list())
                .collect::<Vec<_>>()
                .join(",")
        }
    };
    for_each_indexed_source_analysis(root, index, &hashes, |source| {
        let classified = classify(&source)?;
        let Some(draw) = &classified.mesh_draw else {
            return Err(format!(
                "{} is described as an object or mesh function but reflects no mesh draw",
                source.key()
            ));
        };
        let public = metal2vulkan_validation::source::public_source_path(&source.label).is_some();
        if public {
            public_rows += 1;
        } else {
            private_rows += 1;
        }
        let stage = draw.stage.entry_list().to_string();
        *stages.entry(stage.clone()).or_default() += 1;
        for needed in &draw.missing {
            *missing.entry(needed.entry_list().to_string()).or_default() += 1;
        }
        *needs
            .entry(format!("{stage}:{}", names(&draw.missing)))
            .or_default() += 1;
        for requirement in &classified.tooling_requirements {
            *requirements.entry(requirement.to_string()).or_default() += 1;
        }
        if !summary_only {
            let names_of = classified
                .tooling_requirements
                .iter()
                .map(|requirement| requirement.as_str())
                .collect::<Vec<_>>();
            println!(
                "mesh-draw\t{}\t{}\t{}\tstage={stage}\tpayload={}\tmissing={}\trequirements={}",
                source.key(),
                source.entry,
                if public {
                    source.label.as_str()
                } else {
                    "private"
                },
                match (draw.payload, draw.payload_size) {
                    (false, _) => "none".to_string(),
                    (true, Some(size)) => size.to_string(),
                    (true, None) => "unsized".to_string(),
                },
                names(&draw.missing),
                if names_of.is_empty() {
                    "none".into()
                } else {
                    names_of.join(",")
                }
            );
        }
        Ok(())
    })?;
    println!(
        "mesh-draw-summary\tselected={}\tpublic={public_rows}\tprivate={private_rows}\tfirst={}\tlast={}",
        hashes.len(),
        hashes.first().map_or("none", String::as_str),
        hashes.last().map_or("none", String::as_str),
    );
    for (stage, count) in stages {
        println!("mesh-draw-stage\t{stage}\t{count}");
    }
    for (stage, count) in missing {
        println!("mesh-draw-missing\t{stage}\t{count}");
    }
    for (need, count) in needs {
        println!("mesh-draw-needs\t{need}\t{count}");
    }
    for (requirement, count) in requirements {
        println!("mesh-draw-requirement\t{requirement}\t{count}");
    }
    Ok(())
}

/// The focused audit of entries that link visible functions: the rows whose cached facts describe
/// the family ([`AuditTarget::LinkedFunctions`], independent of any requirement), each reclassified
/// from its source with the symbols it references, the public linked modules (owned MSL under
/// `validation/fixtures/public/linked/`) that define each, whether it is a public fixture's row or
/// a private one, and its requirements.
fn audit_linked_functions(
    root: &std::path::Path,
    index: &std::path::Path,
    after: Option<&str>,
    limit: usize,
    summary_only: bool,
) -> Result<(), String> {
    let hashes =
        select_cached_audit_target_after(index, AuditTarget::LinkedFunctions, after, limit)?;
    let mut definers = BTreeMap::<String, Vec<String>>::new();
    for module in metal2vulkan_validation::library_module::public_library_modules()? {
        for function in
            metal2vulkan_validation::library_module::defined_function_names(&module.air_ll)
        {
            definers
                .entry(function)
                .or_default()
                .push(module.label.clone());
        }
    }
    let mut symbols = BTreeMap::<String, usize>::new();
    let mut requirements = BTreeMap::<String, usize>::new();
    let (mut public_rows, mut private_rows) = (0usize, 0usize);
    for_each_indexed_source_analysis(root, index, &hashes, |source| {
        let classified = classify(&source)?;
        if classified.linked_functions.is_empty() {
            return Err(format!(
                "{} is described as referencing visible functions but references none",
                source.key()
            ));
        }
        let public = metal2vulkan_validation::source::public_source_path(&source.label).is_some();
        if public {
            public_rows += 1;
        } else {
            private_rows += 1;
        }
        for symbol in &classified.linked_functions {
            *symbols.entry(symbol.clone()).or_default() += 1;
        }
        for requirement in &classified.tooling_requirements {
            *requirements.entry(requirement.to_string()).or_default() += 1;
        }
        if !summary_only {
            let names = classified
                .tooling_requirements
                .iter()
                .map(|requirement| requirement.as_str())
                .collect::<Vec<_>>();
            let covered = classified
                .linked_functions
                .iter()
                .filter(|symbol| definers.contains_key(*symbol))
                .count();
            println!(
                "linked-functions\t{}\t{}\t{}\tsymbols={}\tpublic_defined={covered}\trequirements={}",
                source.key(),
                source.entry,
                if public {
                    source.label.as_str()
                } else {
                    "private"
                },
                classified.linked_functions.join(","),
                if names.is_empty() {
                    "none".into()
                } else {
                    names.join(",")
                }
            );
        }
        Ok(())
    })?;
    println!(
        "linked-functions-summary\tselected={}\tpublic={public_rows}\tprivate={private_rows}\tfirst={}\tlast={}",
        hashes.len(),
        hashes.first().map_or("none", String::as_str),
        hashes.last().map_or("none", String::as_str),
    );
    for (requirement, count) in requirements {
        println!("linked-functions-requirement\t{requirement}\t{count}");
    }
    for (symbol, count) in symbols {
        let modules = definers
            .get(&symbol)
            .map_or_else(|| "none".to_string(), |labels| labels.join(","));
        println!("linked-functions-symbol\t{count}\t{symbol}\tpublic_modules={modules}");
    }
    Ok(())
}

#[derive(Default)]
struct TriageSummary {
    signatures: BTreeMap<String, usize>,
    actionable: usize,
    requirements: BTreeMap<ToolingRequirement, usize>,
    unobserved_effects: BTreeMap<UnobservedEffect, usize>,
    stages: BTreeMap<String, usize>,
}

impl TriageSummary {
    fn process(&mut self, source: QueueRow, result: &StructuralTriage, summary_only: bool) {
        *self.stages.entry(source.stage.clone()).or_default() += 1;
        for requirement in &result.tooling_requirements {
            *self.requirements.entry(*requirement).or_default() += 1;
        }
        for effect in result.unobserved_effects() {
            *self.unobserved_effects.entry(effect).or_default() += 1;
        }
        if !summary_only {
            *self.signatures.entry(result.signature.clone()).or_default() += 1;
        }
        self.actionable += 1;
        if !summary_only {
            println!("{}", actionable_line(&source, result));
        }
    }
}

/// One queue row: what blocks its cases (`authoring_review` when nothing does) and, last, what its
/// cases cannot evidence even when authored (`unobserved=none` when nothing).
fn actionable_line(source: &QueueRow, result: &StructuralTriage) -> String {
    let requirements = result
        .tooling_requirements
        .iter()
        .map(|requirement| requirement.as_str())
        .collect::<Vec<_>>()
        .join(",");
    let unobserved = result
        .unobserved_effects()
        .into_iter()
        .map(|effect| effect.as_str())
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "actionable\t{}\t{}\t{}\t{}\t{}\tunobserved={}",
        source.air_sha256,
        source.stage,
        result.signature,
        source.entry,
        if requirements.is_empty() {
            "authoring_review"
        } else {
            requirements.as_str()
        },
        if unobserved.is_empty() {
            "none"
        } else {
            unobserved.as_str()
        }
    )
}

fn single_line(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn required(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn actionable_rows_name_their_unobserved_effects_beside_their_requirements() {
        let row = QueueRow {
            air_sha256: "11".repeat(32),
            stage: "Kernel".into(),
            entry: "main".into(),
            label: String::new(),
            state: QueueState::Unplanned,
        };
        let mut result = StructuralTriage {
            signature: "sig".into(),
            has_cfg_cycle: false,
            air_calls: BTreeMap::new(),
            unrecognized_air_intrinsics: BTreeMap::new(),
            resource_markers: BTreeMap::new(),
            uses_device_addresses: false,
            uses_indirect_commands: false,
            fragment_imageblock: false,
            pipeline_state_members: false,
            tile_function: false,
            mesh_draw: None,
            rasterization_rate_map: false,
            sample_rate: Vec::new(),
            ulong_texture: false,
            fragment_varyings: Vec::new(),
            linked_functions: Vec::new(),
            unobserved_calls: BTreeMap::new(),
            tooling_requirements: Vec::new(),
        };
        let prefix = format!("actionable\t{}\tKernel\tsig\tmain", "11".repeat(32));
        assert_eq!(
            actionable_line(&row, &result),
            format!("{prefix}\tauthoring_review\tunobserved=none")
        );
        result.uses_indirect_commands = true;
        for symbol in [
            "air.set_barrier_compute_command",
            "air.set_kernel_buffer_compute_command.p1i8",
        ] {
            result
                .unobserved_calls
                .insert(symbol.into(), UnobservedEffect::IndirectCommandContents);
        }
        assert_eq!(
            actionable_line(&row, &result),
            format!("{prefix}\tauthoring_review\tunobserved=indirect_command_contents")
        );
        result.tooling_requirements = vec![ToolingRequirement::IndirectCommandBuffer];
        assert_eq!(
            actionable_line(&row, &result),
            format!("{prefix}\tindirect_command_buffer\tunobserved=indirect_command_contents")
        );
        let mut summary = TriageSummary::default();
        summary.process(row, &result, true);
        assert_eq!(
            summary.unobserved_effects,
            BTreeMap::from([(UnobservedEffect::IndirectCommandContents, 1)])
        );
    }

    #[test]
    fn translation_worker_input_batches_escaped_text_and_reports_flush_errors() {
        #[derive(Default)]
        struct CountingWriter {
            bytes: Vec<u8>,
            writes: usize,
            fail: bool,
        }
        impl Write for CountingWriter {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                if self.fail {
                    return Err(std::io::Error::other("destination failed"));
                }
                self.writes += 1;
                self.bytes.extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut source = metal2vulkan_validation::source::SourceRow {
            air_sha256: "11".repeat(32),
            stage: "Kernel".into(),
            entry: "main".into(),
            air_ll: "line \"quoted\"\n".repeat(10_000),
            blob_b64: None,
            lib_sha256s: vec![],
            label: "synthetic".into(),
        };
        let mut destination = CountingWriter::default();
        write_translation_worker_input(&mut destination, &source).unwrap();
        assert_eq!(
            serde_json::from_slice::<metal2vulkan_validation::source::SourceRow>(
                &destination.bytes
            )
            .unwrap(),
            source
        );
        assert!(
            destination.writes <= destination.bytes.len() / 4096 + 1,
            "escaped text must not issue a write for each escape"
        );
        source.air_ll.clear();
        let error = write_translation_worker_input(
            CountingWriter {
                fail: true,
                ..CountingWriter::default()
            },
            &source,
        )
        .unwrap_err();
        assert!(error.contains("flush translation worker input"));
    }

    fn synthetic_source(hash: &str, air_ll: String) -> metal2vulkan_validation::source::SourceRow {
        metal2vulkan_validation::source::SourceRow {
            air_sha256: hash.repeat(64),
            stage: "Kernel".into(),
            entry: "main".into(),
            air_ll,
            blob_b64: None,
            lib_sha256s: vec!["22".repeat(32)],
            label: format!("local/{hash}.ll"),
        }
    }

    fn lanes(sources: &[metal2vulkan_validation::source::SourceRow]) -> Vec<Lane> {
        sources.iter().map(SweepRow::lane).collect()
    }

    #[test]
    fn translation_work_separates_large_sources() {
        let sources = vec![
            synthetic_source("a", "x".repeat(10)),
            synthetic_source("b", "x".repeat(LARGE_TRANSLATION_SOURCE_BYTES + 1)),
            synthetic_source("c", "x".repeat(LARGE_TRANSLATION_SOURCE_BYTES)),
            synthetic_source("d", "x".repeat(100)),
        ];
        assert_eq!(
            lanes(&sources),
            [Lane::Small, Lane::Large, Lane::Large, Lane::Small]
        );
    }

    #[test]
    fn translation_work_bounds_sub_megabyte_cfgs_by_serialized_cost() {
        let sources = vec![
            synthetic_source("a", "x".repeat(LARGE_TRANSLATION_SOURCE_BYTES - 1)),
            synthetic_source("b", "x".repeat(LARGE_TRANSLATION_SOURCE_BYTES)),
        ];
        assert_eq!(lanes(&sources), [Lane::Small, Lane::Large]);
    }

    #[test]
    fn translation_work_bounds_device_address_function_table_rows_by_cost() {
        let sources = vec![
            synthetic_source(
                "a",
                "call void @air.visible_function_table()\n%p = inttoptr i64 %x to ptr".into(),
            ),
            synthetic_source("b", "call void @air.visible_function_table()".into()),
            synthetic_source("c", "%p = inttoptr i64 %x to ptr".into()),
        ];
        assert_eq!(lanes(&sources), [Lane::Large, Lane::Small, Lane::Small]);
        assert!(is_costly_translation_source(&sources[0]));
        assert!(!is_serialized_cost_translation_source(&sources[0]));
        assert!(!is_serialized_cost_translation_source(&sources[1]));
        assert!(!is_serialized_cost_translation_source(&sources[2]));
    }

    #[test]
    fn translation_work_serializes_dense_call_cfgs_by_combined_cost() {
        const BLOCKS: usize = 350;
        const CALLS: usize = 400;
        let mut air_ll = String::new();
        for index in 0..BLOCKS {
            air_ll.push_str(&format!(
                "block{index}:\n  call void @helper()\n  br label %block{index}\n"
            ));
        }
        for _ in BLOCKS..CALLS {
            air_ll.push_str("  call void @helper()\n");
        }
        let source = synthetic_source("1", air_ll);
        assert_eq!(translation_cfg_counts(&source.air_ll), (BLOCKS, CALLS));
        assert_eq!(BLOCKS * CALLS, SERIALIZED_TRANSLATION_CFG_CALL_WORK);
        assert!(is_serialized_cost_translation_source(&source));
        assert_eq!(source.lane(), Lane::Serialized);
    }

    /// A reader over rows held by the test, streamed in the requested order.
    fn read_from(
        rows: &[metal2vulkan_validation::source::SourceRow],
        hashes: &[String],
        consume: &mut dyn FnMut(metal2vulkan_validation::source::SourceRow) -> Result<(), String>,
    ) -> Result<metal2vulkan_validation::source::IndexedSourceReadStats, String> {
        for hash in hashes {
            let row = rows
                .iter()
                .find(|row| &row.air_sha256 == hash)
                .ok_or_else(|| format!("no row {hash}"))?;
            consume(row.clone())?;
        }
        Ok(metal2vulkan_validation::source::IndexedSourceReadStats {
            rows: hashes.len(),
            ..Default::default()
        })
    }

    /// At `jobs = 1` the only worker is the serialized worker and no dedicated large worker
    /// exists; every lane must still be audited exactly once: small rows first in selection
    /// order, then the serialized lane, then the large lane, each second-phase lane largest first.
    #[test]
    fn a_single_job_sweep_audits_every_lane() {
        let mut dense = String::new();
        for index in 0..350 {
            dense.push_str(&format!(
                "block{index}:\n  call void @helper()\n  br label %block{index}\n"
            ));
        }
        dense.push_str(&"  call void @helper()\n".repeat(50));
        let sources = vec![
            synthetic_source("a", "x".repeat(LARGE_TRANSLATION_SOURCE_BYTES)),
            synthetic_source("b", dense),
            synthetic_source("c", "ret void".into()),
            synthetic_source("d", "x".repeat(LARGE_TRANSLATION_SOURCE_BYTES + 1)),
            synthetic_source("e", "ret void".into()),
        ];
        assert_eq!(
            lanes(&sources),
            [
                Lane::Large,
                Lane::Serialized,
                Lane::Small,
                Lane::Large,
                Lane::Small
            ]
        );
        assert_eq!(translation_phase_worker_counts(1, 5, 1, 2), (1, 1, 0));
        let hashes: Vec<_> = sources.iter().map(|s| s.air_sha256.clone()).collect();
        for jobs in 1..=3 {
            let audited = Mutex::new(Vec::new());
            let mut recorded = Vec::new();
            let sweep = sweep_translation_rows(
                &hashes,
                jobs,
                "test",
                &|hashes, consume| read_from(&sources, hashes, consume),
                &|source| {
                    audited.lock().unwrap().push(source.air_sha256.clone());
                    source.air_sha256
                },
                &mut |result: &String, _| recorded.push(result.clone()),
                &mut |_| Ok(()),
            )
            .unwrap();
            assert_eq!((sweep.rows, sweep.reread), (5, 3), "jobs={jobs}");
            assert_eq!(sweep.read.rows, 8, "jobs={jobs}: 5 rows, 3 read again");
            let mut audited = audited.into_inner().unwrap();
            if jobs == 1 {
                let order = ["c", "e", "b", "d", "a"].map(|hash| hash.repeat(64));
                assert_eq!(audited, order);
            }
            audited.sort();
            recorded.sort();
            assert_eq!(audited, hashes, "jobs={jobs}");
            assert_eq!(recorded, hashes, "jobs={jobs}");
        }
    }

    /// A synthetic row that counts how many of its kind are alive at once.
    struct CountedRow {
        hash: String,
        lane: Lane,
        live: Arc<(AtomicUsize, AtomicUsize)>,
    }

    impl CountedRow {
        fn new(hash: &str, lane: Lane, live: &Arc<(AtomicUsize, AtomicUsize)>) -> CountedRow {
            let now = live.0.fetch_add(1, Ordering::SeqCst) + 1;
            live.1.fetch_max(now, Ordering::SeqCst);
            CountedRow {
                hash: hash.to_string(),
                lane,
                live: Arc::clone(live),
            }
        }
    }

    impl Drop for CountedRow {
        fn drop(&mut self) {
            self.live.0.fetch_sub(1, Ordering::SeqCst);
        }
    }

    impl SweepRow for CountedRow {
        fn key(&self) -> std::borrow::Cow<'_, str> {
            std::borrow::Cow::Borrowed(&self.hash)
        }
        fn lane(&self) -> Lane {
            self.lane
        }
        fn size(&self) -> usize {
            self.hash.len()
        }
    }

    fn counted_lane(index: usize) -> Lane {
        match index % 50 {
            0 => Lane::Serialized,
            n if n % 7 == 0 => Lane::Large,
            _ => Lane::Small,
        }
    }

    /// The parent streams rows. Whatever the selection's size, a sweep holds at most its channels',
    /// feeds', and workers' rows at once (`2 * workers + 1` in the first phase, 4 in the second),
    /// audits and records each row exactly once, and persists them in checkpoints of at most ten. A
    /// sweep that collected its rows first would hold all of them.
    #[test]
    fn a_sweep_holds_a_bounded_number_of_rows_whatever_the_selection_size() {
        for jobs in [1, 3, 8] {
            for rows in [12, 1_500] {
                let live = Arc::new((AtomicUsize::new(0), AtomicUsize::new(0)));
                let hashes: Vec<String> = (0..rows).map(|index| format!("{index:064x}")).collect();
                let lane_of =
                    |hash: &str| counted_lane(hashes.iter().position(|h| h == hash).unwrap());
                let reads = AtomicUsize::new(0);
                let read = |hashes: &[String],
                            consume: &mut dyn FnMut(CountedRow) -> Result<(), String>| {
                    for hash in hashes {
                        reads.fetch_add(1, Ordering::SeqCst);
                        consume(CountedRow::new(hash, lane_of(hash), &live))?;
                    }
                    Ok(metal2vulkan_validation::source::IndexedSourceReadStats {
                        rows: hashes.len(),
                        ..Default::default()
                    })
                };
                let mut recorded = Vec::new();
                let (mut persisted, mut largest_checkpoint) = (0usize, 0usize);
                let sweep = sweep_translation_rows(
                    &hashes,
                    jobs,
                    "test",
                    &read,
                    &|row: CountedRow| {
                        std::thread::yield_now();
                        row.hash.clone()
                    },
                    &mut |result: &String, _| recorded.push(result.clone()),
                    &mut |checkpoint| {
                        persisted += checkpoint.len();
                        largest_checkpoint = largest_checkpoint.max(checkpoint.len());
                        Ok(())
                    },
                )
                .unwrap();
                let deferred = (0..rows)
                    .filter(|index| counted_lane(*index) != Lane::Small)
                    .count();
                assert_eq!((sweep.rows, sweep.reread), (rows, deferred));
                assert_eq!(reads.load(Ordering::SeqCst), rows + deferred);
                recorded.sort();
                assert_eq!(recorded, hashes, "jobs={jobs} rows={rows}");
                assert_eq!(persisted, rows);
                assert!(largest_checkpoint <= CHECKPOINT_ROWS);
                let workers = jobs.min(rows);
                let peak = live.1.load(Ordering::SeqCst);
                assert!(
                    peak <= (2 * workers + 1).max(4),
                    "jobs={jobs} rows={rows}: {peak} rows alive at once"
                );
                assert_eq!(live.0.load(Ordering::SeqCst), 0, "every row was released");
            }
        }
    }

    /// A failed checkpoint cancels the sweep: its error is reported, no feed or worker is left
    /// blocked, and only the rows already in flight are audited after it.
    #[test]
    fn a_failed_checkpoint_stops_the_sweep_with_bounded_in_flight_work() {
        for jobs in [1, 4] {
            let live = Arc::new((AtomicUsize::new(0), AtomicUsize::new(0)));
            let hashes: Vec<String> = (0..2_000).map(|index| format!("{index:064x}")).collect();
            let audited = AtomicUsize::new(0);
            let error = sweep_translation_rows(
                &hashes,
                jobs,
                "test",
                &|hashes, consume| {
                    for hash in hashes {
                        consume(CountedRow::new(hash, Lane::Small, &live))?;
                    }
                    Ok(Default::default())
                },
                &|row: CountedRow| {
                    audited.fetch_add(1, Ordering::SeqCst);
                    row.hash.clone()
                },
                &mut |_: &String, _| {},
                &mut |_| Err("database is locked".to_string()),
            )
            .unwrap_err();
            assert_eq!(error, "database is locked");
            let audited = audited.load(Ordering::SeqCst);
            assert!(
                audited <= CHECKPOINT_ROWS + 2 * jobs,
                "jobs={jobs}: {audited} rows audited after a failed first checkpoint"
            );
            assert_eq!(live.0.load(Ordering::SeqCst), 0);
        }
    }

    #[test]
    fn translation_phases_keep_large_work_on_the_bounded_worker_subset() {
        assert_eq!(translation_phase_worker_counts(16, 203, 0, 3), (16, 0, 2));
        assert_eq!(translation_phase_worker_counts(16, 203, 3, 20), (16, 1, 1));
    }

    #[test]
    fn jobs_default_to_available_parallelism() {
        assert_eq!(
            default_jobs(),
            std::thread::available_parallelism().unwrap().get()
        );
    }

    #[test]
    fn translation_result_backlog_is_bounded_by_worker_count() {
        let (sender, _receiver) = bounded_worker_channel(2);
        sender.try_send(1usize).unwrap();
        sender.try_send(2usize).unwrap();
        assert!(matches!(
            sender.try_send(3usize),
            Err(mpsc::TrySendError::Full(3))
        ));
    }

    #[test]
    fn hash_file_accepts_audit_manifests_and_deduplicates_sources() {
        let scratch = ScratchDir::new("corpus-triage-hash-file").unwrap();
        let path = scratch.path().join("sources.txt");
        fs::write(
            &path,
            format!(
                "{} Fragment first\n\n{} Kernel second\n{} duplicate\n",
                "11".repeat(32),
                "22".repeat(32),
                "11".repeat(32)
            ),
        )
        .unwrap();
        assert_eq!(
            read_hash_file(&path).unwrap(),
            vec!["11".repeat(32), "22".repeat(32)]
        );
    }

    #[test]
    fn hash_file_rejects_noncanonical_hashes() {
        let scratch = ScratchDir::new("corpus-triage-bad-hash-file").unwrap();
        let path = scratch.path().join("sources.txt");
        fs::write(&path, format!("{}\n", "AA".repeat(32))).unwrap();
        let error = read_hash_file(&path).unwrap_err();
        assert!(error.contains("first field must be a lowercase SHA-256"));
    }

    /// `--profile` belongs to the audits that translate or read a profile's rows, each needing at
    /// least one, each once; no other audit takes one.
    #[test]
    fn profile_audits_take_profiles_and_others_take_none() {
        let profiles = |names: &[&str]| names.iter().map(|n| n.to_string()).collect::<Vec<_>>();
        for audit in [
            AuditKind::Translation,
            AuditKind::Budgets,
            AuditKind::Specialization,
        ] {
            assert!(check_profiles(Some(audit), &[])
                .unwrap_err()
                .contains("take one or more --profile P"));
            assert!(check_profiles(Some(audit), &profiles(&["b", "a"])).is_ok());
            assert!(check_profiles(Some(audit), &profiles(&["a", "a"]))
                .unwrap_err()
                .contains("once"));
        }
        assert!(check_profiles(Some(AuditKind::BitcodeReader), &[]).is_ok());
        assert!(check_profiles(None, &[]).is_ok());
        assert!(check_profiles(Some(AuditKind::BitcodeReader), &profiles(&["a"])).is_err());
        assert!(check_profiles(None, &profiles(&["a"])).is_err());
    }

    #[test]
    fn worker_boundary_stops_are_typed() {
        use metal2vulkan_validation::worker::{Measured, ProcessFailure};
        let cases = [
            (
                WorkerFailure::Process(
                    ProcessFailure::DeadlineExceeded,
                    Some("late".into()),
                    Measured::default(),
                ),
                "timeout",
                "worker deadline exceeded: late",
            ),
            (
                WorkerFailure::Process(ProcessFailure::DeadlineExceeded, None, Measured::default()),
                "timeout",
                "worker deadline exceeded",
            ),
            (
                WorkerFailure::Process(
                    ProcessFailure::MemoryLimitExceeded {
                        limit: 1,
                        measured: 2,
                    },
                    Some(String::new()),
                    Measured::default(),
                ),
                "oom",
                "worker exceeded 1 byte process-group resident-memory budget (measured 2 bytes): ",
            ),
            (
                WorkerFailure::Exit("abort".into(), Measured::default()),
                "worker_exit",
                "translation worker failed under 500 MiB memory limit: abort",
            ),
            (
                WorkerFailure::Setup("spawn translation worker: nope".into()),
                "tooling",
                "spawn translation worker: nope",
            ),
        ];
        for (failure, key, message) in cases {
            assert_eq!(failure.outcome().key(), key);
            assert_eq!(failure.message(), message);
        }
        assert!(WorkerFailure::Setup(String::new()).outcome().is_transient());
        assert!(!WorkerFailure::Exit(String::new(), Measured::default())
            .outcome()
            .is_transient());
    }

    /// The worker translates a row's retained bitcode, or assembles a public row's AIR text; a row
    /// with a blob sends no text; text `llvm-as` rejects is a recorded `no_bitcode`; a corrupt blob
    /// is transient `tooling`.
    #[test]
    fn engine_worker_translates_retained_bitcode_and_assembles_public_text() {
        use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};
        let profile = engine::load_profile("runtime-floor").unwrap();
        let text = fs::read_to_string(engine_dir().join("store_const.ll")).unwrap();
        let mut source = metal2vulkan_validation::source::SourceRow {
            air_sha256: "11".repeat(32),
            stage: "Kernel".into(),
            entry: "store_const".into(),
            air_ll: "this is not LLVM assembly".into(),
            blob_b64: None,
            lib_sha256s: vec![],
            label: "public/synthetic.ll".into(),
        };
        let run = |source: &metal2vulkan_validation::source::SourceRow| {
            // The row crosses the worker boundary as JSON.
            let row = EngineRow::from(source.clone());
            let json = serde_json::to_vec(&row.input).unwrap();
            let input: engine::RowInput<'static> = serde_json::from_slice(&json).unwrap();
            let (mut result, pending) = audit_engine_row(input, &profile, None);
            // What the worker's L3 stage settles, here in this process.
            if let Some(pending) = pending {
                result.attempt.settle_l3(pending.check());
            }
            result
        };
        let bitcode = match llvm_assemble(&engine_dir().join("store_const.ll")) {
            Ok(bytes) => bytes,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected store_const.ll: {why}"),
        };
        let result = run(&source);
        assert_eq!(result.air_sha256, source.air_sha256);
        assert_eq!(result.attempt.outcome.key(), "no_bitcode", "{result:?}");
        assert!(!result.attempt.outcome.is_defect() && !result.attempt.outcome.is_transient());
        source.air_ll = text;
        let result = run(&source);
        assert_eq!(result.attempt.outcome, Outcome::Translated, "{result:?}");
        let assembled_words = result.attempt.words;
        source.blob_b64 = Some("not base64!".into());
        let result = run(&source);
        assert_eq!(result.attempt.outcome.key(), "tooling");
        assert!(result.attempt.outcome.is_transient());
        source.blob_b64 = Some(base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            &bitcode,
        ));
        let row = EngineRow::from(source.clone());
        assert_eq!((row.lane, row.size), (Lane::Small, source.air_ll.len()));
        assert!(matches!(row.input.source, engine::RowSource::Retained(_)));
        let json = serde_json::to_string(&row.input).unwrap();
        assert!(
            json.contains("\"retained\"") && !json.contains("\"text\""),
            "a retained row sends no text: {json}"
        );
        let result = run(&source);
        assert_eq!(result.attempt.outcome, Outcome::Translated, "{result:?}");
        assert_eq!(result.attempt.words, assembled_words);
    }
}
