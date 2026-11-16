//! The committed prose cites no design record, work package or plan item.
//!
//! `AGENTS.md`: code, comments and docs cite no design-record entries, work-package codes or
//! TODO section numbers, and the tree keeps no plan, status or journal. The authored cases'
//! `rationale` and `no_effect`, and a refusal record's `detail`, are prose a reader meets, so
//! the rule reaches them. None of the three is part of any identity, cache key or validity
//! rule (`computed_case_id` hashes `AuthoredCase::semantic`, `computed_input_sha256` its own
//! `Inputs`, and `index::refusal_staleness` reads a refusal's ABI, fingerprint, venue and
//! identity), so this check is over wording alone.
//!
//! The walk reads `corpus/cases` and `corpus/observations` by name and never the corpus root.
//! Do not widen it to the root as a tidy-up: `corpus/local/` is a private corpus that is not
//! ours, is gitignored, and is never read by a test, committed or regenerated on our
//! initiative. A walk of the root would read it wherever it exists and pass in a tree where it
//! happens to be absent, which is a test passing for the wrong reason.
//!
//! The vocabulary below was taken from a census of every code-shaped token in the committed
//! prose, not a sample: a sample is what once hid `STATUS wave 11 owes this reading` behind a
//! `TUS wave` that matched nothing. `CONTENT` holds the tokens that look like codes and are
//! not, so a true citation cannot be waved through by widening the pattern.

use std::fs;
use std::path::{Path, PathBuf};

/// The corpus root. Only the two committed directories under it are ever read; see the module
/// docs for why the walk is not widened to the root.
fn corpus() -> PathBuf {
    metal2vulkan_devtools::validation_dir().join("corpus")
}

/// A design-record, work-package or plan citation, by the spelling each family uses. `D` followed
/// by digits is checked apart, since `D3D11` is content.
const CITATIONS: [&str; 10] = [
    "DESIGN",
    "BUILD_PLAN",
    "STATUS",
    "handback",
    "TODO",
    "docs/TODO.md",
    "WP-",
    "UNV-metal-probe",
    "WRG-math-probe",
    "TUS wave",
];

/// The work-package and ladder codes the census found, which no pattern would guess.
const CODES: [&str; 18] = [
    "K7b", "K14d", "L1d", "L7d", "L7e", "L7f", "L9b2", "V2b", "V5", "B3b", "B3", "A2", "A18",
    "A20", "Q1", "Q1F", "Q3", "Q1F-math",
];

/// Tokens shaped like a code that are content and must never trip the check: Apple devices and
/// chips, an alpha-8 plane, a kernel's own bound, the C standard, a channel count, the
/// validation ladder's rungs (which live in `src/c1.rs` and `src/c3.rs`), a SPIR-V image format
/// and Direct3D.
const CONTENT: [&str; 13] = [
    "M3", "M4", "A8", "A14", "K4", "C99", "C4", "C1", "C3", "R32f", "D3D11", "D3D12", "D3D",
];

/// Every citation `text` holds. `CONTENT` is removed first, so a content token that contains a
/// code's spelling cannot report one.
fn citations(text: &str) -> Vec<String> {
    let mut rest = text.to_string();
    for content in CONTENT {
        rest = rest.replace(content, " ");
    }
    let mut found = Vec::new();
    for needle in CITATIONS.iter().chain(CODES.iter()) {
        if word_at(&rest, needle) {
            found.push((*needle).to_string());
        }
    }
    found.extend(design_records(&rest));
    found
}

/// `needle` occurs in `rest` as a word: nothing alphanumeric before it, and nothing alphanumeric
/// after it unless the needle already ends in punctuation (`WP-`, which a code follows).
///
/// The boundary is the whole point. `contains` would read `EVs SWP-stagein-m3`, an evidence run's
/// name, as a `WP-` work package; and it would read the `TUS` inside `STATUS` as a `TUS wave`,
/// which is how a phantom code once stood in for six real plan-status notes.
fn word_at(rest: &str, needle: &str) -> bool {
    let open_ended = !needle.ends_with(|c: char| c.is_alphanumeric());
    rest.match_indices(needle).any(|(at, _)| {
        let plain = |c: Option<char>| c.is_none_or(|c| !c.is_alphanumeric() && c != '_');
        let before = rest[..at].chars().next_back();
        plain(before) && (open_ended || plain(rest[at + needle.len()..].chars().next()))
    })
}

/// A `D<n>` or `D<n>.<n>` design-record reference, bounded as a word.
fn design_records(rest: &str) -> Vec<String> {
    let bytes = rest.as_bytes();
    let mut found = Vec::new();
    for (at, _) in rest.match_indices('D') {
        let before = rest[..at].chars().next_back();
        if before.is_some_and(|c| c.is_alphanumeric() || c == '_') {
            continue;
        }
        let mut end = at + 1;
        while end < bytes.len() && bytes[end].is_ascii_digit() {
            end += 1;
        }
        if end == at + 1 {
            continue;
        }
        if bytes.get(end) == Some(&b'.') && bytes.get(end + 1).is_some_and(u8::is_ascii_digit) {
            end += 1;
            while end < bytes.len() && bytes[end].is_ascii_digit() {
                end += 1;
            }
        }
        if bytes
            .get(end)
            .is_none_or(|c| !c.is_ascii_alphanumeric() && *c != b'_')
        {
            found.push(rest[at..end].to_string());
        }
    }
    found
}

/// Every committed prose value, as (shard, which field, the text).
fn committed_prose() -> Vec<(String, &'static str, String)> {
    let mut out = Vec::new();
    let mut shards = Vec::new();
    // Only the committed directories. `corpus/local/` is a private corpus that is not ours and
    // is gitignored; a test that walked the whole corpus root would read it where it exists.
    for committed in ["cases", "observations"] {
        collect_shards(&corpus().join(committed), &mut shards);
    }
    shards.sort();
    for shard in shards {
        let text = fs::read_to_string(&shard).expect("read a committed shard");
        let where_ = shard.display().to_string();
        for line in text.lines().filter(|line| !line.trim().is_empty()) {
            let value: serde_json::Value =
                serde_json::from_str(line).expect("a committed shard row parses");
            for field in ["rationale", "no_effect"] {
                if let Some(prose) = value.get(field).and_then(serde_json::Value::as_str) {
                    out.push((where_.clone(), field, prose.to_string()));
                }
            }
            if let Some(prose) = value
                .get("refusal")
                .and_then(|refusal| refusal.get("detail"))
                .and_then(serde_json::Value::as_str)
            {
                out.push((where_.clone(), "refusal.detail", prose.to_string()));
            }
        }
    }
    out
}

/// Every `*.jsonl` under `root`, in no particular order: the caller sorts, so the check does not
/// depend on the filesystem's.
fn collect_shards(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if path.is_dir() {
            collect_shards(&path, out);
        } else if path
            .extension()
            .is_some_and(|extension| extension == "jsonl")
        {
            out.push(path);
        }
    }
}

#[test]
fn committed_prose_cites_no_design_record_or_work_package() {
    let prose = committed_prose();
    assert!(
        prose.len() > 2000,
        "the committed corpus holds {} prose values; the check read the wrong tree",
        prose.len()
    );
    let mut offences = Vec::new();
    for (shard, field, text) in &prose {
        let found = citations(text);
        if !found.is_empty() {
            offences.push(format!("{shard} [{field}]: {}", found.join(", ")));
        }
    }
    assert!(
        offences.is_empty(),
        "{} committed prose value(s) cite a design record, work package or plan item; say what \
         the case tests instead:\n{}",
        offences.len(),
        offences.join("\n")
    );
}

/// The check reports a citation of every family, and reports none of the content tokens that
/// are shaped like one. Both halves are synthetic, so neither depends on the corpus.
#[test]
fn the_prose_check_reads_citations_apart_from_content() {
    for cited in [
        "as DESIGN D272 states",
        "per D62.7 and D90.2",
        "BUILD_PLAN WP-L1d item 6 lowers it",
        "STATUS wave 11 owes this reading",
        "TODO 6.2 covers the zero case",
        "docs/TODO.md 4.2 asks for it",
        "the B3b handback's observations",
        "DESIGN D195 (K7b): two patches",
        "the comparison classes for V5",
        "Q3 model (control Q3)",
        "Zero A18 matrix extent",
        "one-invocation A20 convolution case",
        "WP-UNV's probe UNV-metal-probe-1 read 0x7e00",
    ] {
        assert!(
            !citations(cited).is_empty(),
            "the check let a citation through: {cited}"
        );
    }
    let content = "As an M3 Max and an M4 read it, EVs SWP-stagein-m3 and -m4. \
                   Neither the M3 Max nor the M4 has family 10; the guarded tail to K4 = 296 \
                   under C99 Annex F, with W, H and C4*n_batch all one. The C1 row and the C3 \
                   subgroup gap type as R32f, where D3D11's reference tessellator agrees. The \
                   A8 plane is biased, and the A14 NHWC kernel multiplies bfloat matrices.";
    assert!(
        citations(content).is_empty(),
        "the check fired on content: {:?}",
        citations(content)
    );
}
