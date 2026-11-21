use metal2vulkan_validation::authored_compile::CurrentArtifacts;
use metal2vulkan_validation::c1::reviewed::ReviewedList as C1List;
use metal2vulkan_validation::c3::bound::UNSTATED_WORDS;
use metal2vulkan_validation::c3::reference::{CheckedFacts, References};
use metal2vulkan_validation::c3::{self, C3Input, C3Report, ReviewedList, Venues, Verdict};
use metal2vulkan_validation::cutover::scope::OutOfScope;
use metal2vulkan_validation::cutover::List as _;
use metal2vulkan_validation::index::{
    default_index_path, evidence_status, select_queue, status_counts, EvidenceState, QueueState,
};
use metal2vulkan_validation::source::corpus_root;
use metal2vulkan_validation::store::CorpusStore;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

fn main() {
    if let Err(error) = run() {
        eprintln!("corpus-status: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut root = corpus_root();
    let mut index = None;
    let mut limit = 20usize;
    let mut c3 = false;
    let mut reviewed = ReviewedList::committed_path();
    let mut c1_reviewed = None;
    let mut venues = None;
    let mut references = None;
    let mut case_ids = Vec::new();
    let mut venue_keys = false;
    let mut rekey = None;
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
            "--c3" => c3 = true,
            "--reviewed" => reviewed = PathBuf::from(required(&mut args, "--reviewed")?),
            "--venues" => venues = Some(PathBuf::from(required(&mut args, "--venues")?)),
            "--references" => {
                references = Some(PathBuf::from(required(&mut args, "--references")?))
            }
            "--venue-keys" => venue_keys = true,
            "--rekey-venues" => rekey = Some(PathBuf::from(required(&mut args, "--rekey-venues")?)),
            "--case-id" => case_ids.push(required(&mut args, "--case-id")?),
            "--c1-reviewed" => {
                c1_reviewed = Some(PathBuf::from(required(&mut args, "--c1-reviewed")?))
            }
            "-h" | "--help" => {
                println!(
                    "usage: corpus-status [--corpus DIR] [--private-corpus DIR] [--index PATH] [--limit N]\n\
                     \x20                    [--case-id HASH]...\n\
                     \x20      corpus-status --c3 [--venues PATH] [--references PATH] [--corpus DIR]\n\
                     \x20                    [--private-corpus DIR] [--index PATH]\n\
                     \x20                    [--reviewed PATH] [--c1-reviewed PATH] [--limit N]\n\
                     \x20                    [--case-id HASH]...\n\
                     \x20      corpus-status --venue-keys [--venues PATH]\n\
                     \x20      corpus-status --rekey-venues LIST [--corpus DIR]\n\n\
                     Without a mode it prints the index's queue and an evidence line, with its\n\
                     slots, for each of the first --limit cases whose Metal evidence is not\n\
                     current or whose engine slots hold a failed or stale row (the old\n\
                     engine's rows are no evidence and are not listed); with --case-id, for each\n\
                     named case whatever its evidence's state (a case the store does not hold\n\
                     fails).\n\n\
                     --c3 prints the venues' report, each case's verdict on every venue, from the case and\n\
                     observation shards alone: per engine slot of each venue of the venue list\n\
                     (default validation/cutover/c3-venues.json), each case's verdict\n\
                     (match, reviewed, device_gap, out_of_scope_ray_tracing, reference_gap,\n\
                     mismatch, refused, stale, missing). A case whose current Metal row is a\n\
                     reference gap (the reference device lacks the Apple GPU family its pipeline\n\
                     needs) reads reference_gap, is printed (c3-reference-gap), and needs the\n\
                     reviewed list's reference_gaps entry signed off. A venue's\n\
                     slots must cover its\n\
                     population: the public cases (lavapipe), or every Metal-qualified case (the real\n\
                     GPUs). A row that recorded another device or driver than its venue declares is\n\
                     stale (stale_device). Each case is judged against one Metal reference row:\n\
                     the reference environment's of the reference list (default\n\
                     validation/cutover/c3-references.json), whose rows are printed\n\
                     (c3-reference), with the current rows of environments it does not name\n\
                     (c3-reference-unlisted). A row whose case has no reference row is\n\
                     stale until Metal requalifies the case (stale_without_metal). A case the device\n\
                     refused holds a refusal record: a\n\
                     current one is refused, or a device_gap when C1's reviewed list (default\n\
                     validation/cutover/c1-reviewed.json) holds its product outcome as a device gap\n\
                     of the venue's profile; each such C1 entry is printed (c3-device-gap) and its\n\
                     sign-off counts in C3's, and a device-capability refusal C1 does not list is a\n\
                     proposed C1 entry (c3-gap-proposal). A current mismatch of a case the\n\
                     reviewed list holds under an answer gap (fragment_subgroup_under_two_quads,\n\
                     clipped_primitive_split), in a slot whose venue's device has that gap,\n\
                     decided from its committed profile, where the case's structure meets it\n\
                     (clipped_primitive_split: a multisampled fragment case drawn by the\n\
                     full-screen companion that reads [[sample_mask]]; the structure comes from\n\
                     checking the case once, kept in the index (--index) under the build's\n\
                     key), stays a mismatch whose\n\
                     reason names the gap (deferred{{answer_gap:...}}): open until an exact\n\
                     emulation exists, never signed off; each such entry is printed\n\
                     (c3-answer-gap ... open). Any other current mismatch\n\
                     is reviewed only when the reviewed list (default\n\
                     validation/cutover/c3-reviewed.json) holds its slot, candidate, output, and\n\
                     Metal output (an apple_reading), or holds its case as a fast_math_regrouping\n\
                     reading, keyed by its cause, and the row's comparison is unbounded by\n\
                     FastMathRewrites, in any slot. It then prints each reviewed\n\
                     observation's state (a regrouping observation holds while its slot's row\n\
                     shows the cause, and reads changed once it does not), and each list entry\n\
                     with its sign-off (awaiting_user or\n\
                     signed_off, as C1's list prints it). For an apple_reading it judges every\n\
                     word the device differs from Metal in against Apple's stated accuracy\n\
                     (c3-bound, c3-bound-word, c3-bound-unstated), a word where\n\
                     Apple states none staying open where the row records the product's\n\
                     lowering as a near-equivalent (near_equivalent), and refuses a sign-off\n\
                     whose words are not all within it (c3-refused). --limit caps the\n\
                     stale and missing rows printed per slot. Exits 0 only when C3 holds, which\n\
                     needs the user's sign-off of every entry. --case-id prints, per slot, each named\n\
                     case's verdict (c3-case SLOT CASE VERDICT): match included, which the open rows\n\
                     never list, or not_judged.\n\n\
                     --venue-keys prints, one JSON object per venue of the venue list, the key this\n\
                     build's venue records hold: the venue's committed profile, its\n\
                     file's SHA-256, and its translation key ({{\"stem\", \"sha256\",\n\
                     \"fingerprint\"}}). --rekey-venues migrates venue records to this build's keys\n\
                     without running anything: LIST is a JSON array of {{\"stem\", \"from\":\n\
                     {{\"sha256\", \"fingerprint\"}}, \"to\": {{...}}}}, each \"to\" this build's key of\n\
                     \"stem\" (as --venue-keys prints it), and every row and refusal, of either\n\
                     backend, that recorded \"stem\" at exactly \"from\" records \"to\" instead, in\n\
                     one transaction; no other record changes, and a second run changes nothing.\n\
                     Use it only for a change whose authored-compile census leaves every artifact\n\
                     digest byte-identical: the records then stay their venue's\n\
                     evidence. It prints each rekey's rewritten rows and refusals."
                );
                return Ok(());
            }
            _ => return Err(format!("unknown argument {arg:?}")),
        }
    }
    // The corpus's index: the queue's, and where a case's checked facts are kept
    // (`CheckedFacts`).
    let index = index.unwrap_or_else(|| default_index_path(&root));
    if venue_keys {
        let venues = Venues::load(&venues.unwrap_or_else(Venues::committed_path))?;
        for (stem, key) in venues.committed_keys()? {
            let line = serde_json::json!({
                "stem": stem,
                "sha256": key.sha256,
                "fingerprint": key.fingerprint,
            });
            println!("{line}");
        }
        return Ok(());
    }
    if let Some(list) = rekey {
        let rekeys = c3::VenueRekey::load_checked(&list)?;
        let counts = CorpusStore::new(&root).rekey_venue_records(&rekeys)?;
        for (rekey, (rows, refusals)) in rekeys.iter().zip(counts) {
            println!(
                "rekey\t{}\t{}\t{}\trows={rows}\trefusals={refusals}",
                rekey.stem, rekey.from.fingerprint, rekey.to.fingerprint
            );
        }
        return Ok(());
    }
    if c3 {
        let venues = Venues::load(&venues.unwrap_or_else(Venues::committed_path))?;
        let list = ReviewedList::load(&reviewed)?;
        let c1 = C1List::load(&c1_reviewed.unwrap_or_else(C1List::committed_path))?;
        // A venue's records of another build count while they executed this build's artifacts, the
        // corpus's census file's when it is at this build.
        let keys = venues.keys(CurrentArtifacts::load(&root)?)?;
        let references = References::load(&references.unwrap_or_else(References::committed_path))?;
        let facts = CheckedFacts::new(&root, Some(&index));
        let report = c3::report(
            &C3Input::read(&root)?,
            &list,
            &c1,
            &venues,
            &keys,
            &references,
            &facts,
        );
        print_c3(&report, limit);
        for slot in &report.slots {
            for case_id in &case_ids {
                println!(
                    "c3-case\t{}\t{case_id}\t{}",
                    slot.slot,
                    slot.verdict_of(case_id)
                        .map_or("not_judged", |verdict| verdict.as_str())
                );
            }
        }
        if !report.holds() {
            return Err("C3 does not hold".into());
        }
        return Ok(());
    }
    if venues.is_some() || references.is_some() || c1_reviewed.is_some() {
        return Err("--venues, --references, and --c1-reviewed need --c3".into());
    }
    for (state, count) in status_counts(&index)? {
        println!("{state}\t{count}");
    }
    for row in select_queue(&index, QueueState::Authored, limit)? {
        println!(
            "action\t{}\t{}\t{}\t{}",
            row.state.as_str(),
            row.air_sha256,
            row.entry,
            row.label,
        );
    }
    // The engine slots (`ENV@engine`), the only candidate evidence (cutover X2b): a case
    // the executor has not run yet is not a problem, a failed or stale row is.
    let candidate_problem =
        |state: EvidenceState| matches!(state, EvidenceState::Failed | EvidenceState::Stale);
    // A venue's records are held to its committed profile, and those of another build to this
    // build's artifacts.
    let keys = Venues::load(&Venues::committed_path())?.keys(CurrentArtifacts::load(&root)?)?;
    // Each case's Metal dependency is its one reference row.
    let references = References::load(&References::committed_path())?;
    // Named cases are each listed, current or not; otherwise those of the first `limit` cases whose
    // evidence is not all current.
    let named = case_ids.iter().map(String::as_str).collect::<BTreeSet<_>>();
    let mut listed = BTreeSet::new();
    for row in evidence_status(&root, &keys, &references)?
        .into_iter()
        .filter(|row| named.is_empty() || named.contains(row.case_id.as_str()))
        .take(if named.is_empty() { limit } else { usize::MAX })
    {
        if !named.is_empty() {
            listed.insert(row.case_id.clone());
        }
        if !named.is_empty() || row.metal != EvidenceState::Current || candidate_problem(row.vulkan)
        {
            println!(
                "evidence\t{}\tmetal={}\tvulkan={}\t{}",
                row.case_id,
                row.metal.as_str(),
                row.vulkan.as_str(),
                row.translation_error.as_deref().unwrap_or(&row.name)
            );
            for slot in row.slots {
                println!(
                    "slot\t{}\t{}\t{}\t{}",
                    row.case_id,
                    slot.backend,
                    slot.environment_id,
                    slot.state.as_str()
                );
            }
        }
    }
    if let Some(unknown) = named.iter().find(|case_id| !listed.contains(**case_id)) {
        return Err(format!("case_id {unknown} not found"));
    }
    Ok(())
}

fn print_c3(report: &C3Report, limit: usize) {
    let reference = &report.reference;
    println!(
        "c3-reference\t{}\tcurrent_rows={}\tother_identity={}\tjudging={}",
        reference.environment_id,
        reference.current_rows,
        reference.other_identity,
        reference.judging
    );
    for (environment, rows) in &reference.unlisted {
        println!("c3-reference-unlisted\t{environment}\tcurrent_rows={rows}");
    }
    for slot in &report.slots {
        let counts = [Verdict::Match, Verdict::Reviewed, Verdict::DeviceGap]
            .into_iter()
            .chain(OutOfScope::ALL.map(Verdict::OutOfScope))
            .chain([
                Verdict::ReferenceGap,
                Verdict::Mismatch,
                Verdict::Refused,
                Verdict::Stale,
                Verdict::Missing,
            ])
            .map(|verdict| format!("{}={}", verdict.as_str(), slot.count(verdict)))
            .collect::<Vec<_>>()
            .join("\t");
        println!(
            "c3-slot\t{}\t{}\tpopulation={}\t{counts}\tmissing_without_metal={}\tstale_without_metal={}\tstale_device={}\tholds={}",
            slot.kind.as_str(),
            slot.slot,
            slot.population,
            slot.missing_without_metal,
            slot.stale_without_metal,
            slot.stale_device,
            slot.holds()
        );
        let mut printed = BTreeMap::<Verdict, usize>::new();
        for row in &slot.rows {
            let shown = printed.entry(row.verdict).or_default();
            // Every mismatch, reviewed, refused, and device-gap row; stale and missing rows up to
            // the limit.
            if matches!(row.verdict, Verdict::Stale | Verdict::Missing) && *shown >= limit {
                continue;
            }
            *shown += 1;
            println!(
                "c3-row\t{}\t{}\t{}\t{}\t{}{}",
                slot.slot,
                row.case_id,
                row.verdict.as_str(),
                row.reason
                    .as_ref()
                    .map_or_else(|| "-".to_string(), |reason| reason.label()),
                row.name,
                row.refusal
                    .as_ref()
                    .map(|refusal| format!(
                        "\trefusal={}\t{}",
                        refusal.class.as_str(),
                        single_line(&refusal.detail)
                    ))
                    .unwrap_or_default()
            );
        }
    }
    for row in &report.readings {
        println!(
            "c3-reading\t{}\t{}\t{}\tcause={}\tcurrent={}\t{}",
            row.case_id,
            row.environment_id,
            row.state.label(),
            row.cause.as_str(),
            row.current,
            row.name
        );
        let Some(bound) = &row.bound else {
            continue;
        };
        println!(
            "c3-bound\t{}\t{}\tdiffering={}\twithin={}\tunstated={}\tnear_equivalent={}\toutside={}\tuncovered={}\tnot_checkable={}\topen={}",
            row.case_id,
            row.environment_id,
            bound.words.len(),
            bound.count("within"),
            bound.count("unstated"),
            bound.count("near_equivalent"),
            bound.count("outside"),
            bound.count("uncovered"),
            bound.count("not_checkable"),
            bound.open(UNSTATED_WORDS).count(),
        );
        let mut unstated = BTreeMap::<&str, usize>::new();
        for word in &bound.words {
            match word.verdict.label() {
                "within" => {}
                "unstated" => *unstated.entry(word.verdict.detail()).or_default() += 1,
                label => println!(
                    "c3-bound-word\t{}\t{}\tword={}\t{}\t{label}\t{}",
                    row.case_id,
                    row.environment_id,
                    word.word,
                    word.part.as_str(),
                    word.verdict.detail()
                ),
            }
        }
        for (why, count) in unstated {
            println!(
                "c3-bound-unstated\t{}\t{}\tcount={count}\t{why}",
                row.case_id, row.environment_id
            );
        }
    }
    for refused in &report.refused {
        println!(
            "c3-refused\t{}\topen_words={}\t{}",
            refused.case_id,
            refused.words.len(),
            refused.name
        );
    }
    for entry in &report.entries {
        println!(
            "c3-entry\t{}\tcause={}\t{}\t{}",
            entry.case_id,
            entry.cause.as_str(),
            entry.sign_off.label(),
            entry.name
        );
    }
    for gap in &report.device_gaps {
        println!(
            "c3-device-gap\t{}\t{}\t{}\tclass={}\tcases={}\t{}",
            gap.profile,
            gap.outcome,
            gap.key,
            gap.class.as_str(),
            gap.cases,
            gap.sign_off.label()
        );
    }
    for proposal in &report.gap_proposals {
        println!(
            "c3-gap-proposal\t{}\t{}\t{}\tcases={}",
            proposal.profile, proposal.outcome, proposal.key, proposal.cases
        );
    }
    for gap in &report.reference_gaps {
        println!(
            "c3-reference-gap\t{}\trecorded={}\tlisted={}\tslots={}\t{}\t{}",
            gap.case_id,
            gap.recorded
                .map_or_else(|| "-".to_string(), |family| format!("GPUFamily{family}")),
            gap.listed.as_ref().map_or_else(
                || "-".to_string(),
                |(family, _)| format!("GPUFamily{family}")
            ),
            gap.slots,
            gap.label(),
            gap.name
        );
    }
    for gap in &report.answer_gaps {
        println!(
            "c3-answer-gap\t{}\tgap={}\tslots={}\topen\t{}",
            gap.case_id,
            gap.gap.as_str(),
            gap.slots,
            gap.name
        );
    }
    println!(
        "c3\tholds={}\tslots_holding={}/{}\treadings={}\tstale_readings={}\tdevice_gaps={}\tgap_proposals={}\treference_gaps={}\tstale_reference_gaps={}\tawaiting_sign_off={}\tsigned_off={}\trefused_sign_offs={}",
        report.holds(),
        report.slots.iter().filter(|slot| slot.holds()).count(),
        report.slots.len(),
        report.readings.len(),
        report.stale_readings(),
        report.device_gaps.len(),
        report.gap_proposals.len(),
        report.reference_gaps.len(),
        report.stale_reference_gaps(),
        report.awaiting_sign_off(),
        report.signed_off(),
        report.refused.len()
    );
}

fn single_line(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn required(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}
