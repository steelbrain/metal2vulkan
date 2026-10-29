//! The public surface of the three published crates, signature by signature.
//!
//! `api_surface` pins the **names** `metal2vulkan` exports and the construction class of each
//! public type; `metal2vulkan-air`'s own `api_surface` pins its names. Neither pins a method's
//! parameters or its return type, neither covers `metal2vulkan-runtime` at all, and neither sees a
//! method or an enum variant disappear — so a parameter added to a public method, a return type
//! widened to an `Option`, a unit variant that became a struct variant, and a `pub fn` that
//! stopped being public are all breaking changes those tables pass. This test is what makes them
//! fail.
//!
//! `scripts/api-signatures/signatures.py` derives the surface from the sources and
//! `tests/api_signatures.tsv` is what it must derive. The derivation reads source text only: no
//! build, no toolchain, and no rustdoc, whose JSON surface is nightly-only while this workspace
//! is pinned to stable. So it gives the same answer under any feature set, and the run is seconds.
//!
//! A record is `kind<TAB>key<TAB>signature`. The kinds are the declaration itself (`struct`,
//! `enum`, `fn`, `trait`, `type`, `const`, `static`), a public field or an enum variant (`field`,
//! `variant`), the `pub fn`, `pub const` and `pub type` of an inherent impl wherever in the crate
//! it lives (`method`, `assoc-const`, `assoc-type`), the header of each trait impl
//! (`trait-impl`), a public trait's own items (`trait-item`), a declaring macro's template
//! (`macro-template`) and the items one of its calls states (`macro-decl`, `macro-member`). A
//! constant's declaration and its value are two records (`const` and `const-value`, `assoc-const`
//! and `assoc-const-value`), so a value that moves is a different line from a type that moves, in
//! its own section of the report.
//!
//! # What blessing it asserts
//!
//! `python3 scripts/api-signatures/signatures.py --bless` rewrites the table and prints every
//! record that moved, with both sides of a change. Running it asserts that each of those moves is
//! **intended**, and that a removed or changed line — which is a breaking change of a published
//! crate — is one the commit means to make and says so in its message. It is not a way to make a
//! red test green: a line here is what a consumer's source is written against.
//!
//! # What it cannot see
//!
//! - A trait impl past its header. `impl Display for X` is a record; its `fmt` is not.
//! - A derived or a blanket impl, which no source states. The `#[derive(..)]` on the type is part
//!   of its record, so a derive that is dropped does show.
//! - A method a macro expands. The declaring macro's template stands for every type it declares,
//!   so a change to the generated methods shows once, against the template, not once per type.
//! - A value the build computes. `cache::SOURCE_DIGEST` is `env!("M2V_SOURCE_DIGEST")`: its type
//!   is a record and its value is that call, never the digest.
//! - The values of `cache::SEMANTICS_VERSION`, `cache::ARTIFACT_FORMAT_VERSION` and
//!   `cache::RUNTIME_VERSION`. Their declarations are pinned like any other constant; their
//!   values are not recorded, because this repository bumps them whenever behaviour, an encoding
//!   or a run changes — several times a day — and a table that holds them is blessed by reflex
//!   within a day. Every other public constant's value is a record. `--check` holds each of the
//!   three to a constant the surface still has, so the exception cannot outlive its constant.
//! - A generic bound past the declaration's own text, and whitespace: two spellings of one
//!   signature read as a change. Bracket spacing and the comma the formatter leaves before a
//!   closing bracket are put in one form first, so a signature that merely crossed the line width
//!   does not read as a change in each of its arguments.
//! - `#[cfg]`: a gated item is read as present, with its gate kept in the signature, so a gate
//!   that moves reads as a changed signature rather than as an item that came and went.
//! - A name two types share. Where a surface name's spelling also names a private type in the
//!   same crate, or a type of another published crate whose items this one re-exports, their
//!   records merge: twenty-one keys carry two. That over-reports — a change to the namesake shows
//!   against the surface name — and never under-reports.
//! - `#[doc(hidden)]`, which `api_surface` calls public but not API: the runtime's
//!   `generated_commands`, `Execution` and `CommandsRecorded` are not pinned here.
//! - `Send`, `Sync` and anything else the type system implies. `api_surface` and the runtime's
//!   `consumer_auto_traits` own those.
//!
//! The derivation's own parse is held to the sources three ways, because a check over text fails
//! quietly: `--check` reproduces `api_surface`'s module-root table exactly on the real tree, the
//! table itself is the pin so a parse that went blind reads as thousands of removals, and
//! `--selftest` runs the derivation over little trees whose answer is known by construction, each
//! case asserting both a record that must be there and one that must not.
//!
//! `python3 scripts/api-signatures/signatures.py --check` prints the per-crate record counts and
//! the diff; `--print` prints the derived records.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The repository root.
fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The table's own path.
fn table_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/api_signatures.tsv")
}

/// The derivation run with `verb`, as its own words and its exit status.
fn derivation(verb: &str) -> (bool, String) {
    let script = root().join("scripts/api-signatures/signatures.py");
    // `-B` so the run leaves no `__pycache__` beside the script: a test writes nothing into the
    // tree it is checking.
    let run = Command::new("python3")
        .arg("-I")
        .arg("-B")
        .arg(&script)
        .arg(verb)
        .current_dir(root())
        .output()
        .unwrap_or_else(|error| panic!("python3 {} {verb}: {error}", script.display()));
    let out = format!(
        "{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    (run.status.success(), out)
}

/// The table is the surface the sources state, record for record.
///
/// The failure names every record that moved and both sides of every change. A line here is a
/// published crate's API: see the module documentation for what blessing it asserts.
#[test]
fn the_public_signatures_are_the_committed_table() {
    let (ok, out) = derivation("--check");
    assert!(ok, "{out}");
}

/// The derivation records what it claims and refuses what it must.
///
/// The table catches a derivation that went **stale**. It cannot catch one that went **vacuous**:
/// a shape the parse stops recognising takes records out and a smaller surface then reads like the
/// truth, which is how `#[doc(hidden)]` and `#[cfg(test)]` items leaked into an earlier draft of
/// this derivation as if they were API. `--selftest` runs it over trees it writes itself, where
/// the answer follows from the sources, and every case asserts both directions.
#[test]
fn the_derivation_still_sees_what_it_claims() {
    let (ok, out) = derivation("--selftest");
    assert!(ok, "{out}");
}

/// The table is a table of the whole surface, not of a corner of it.
///
/// `--check` holds the table to the derivation, and holds the derivation to a floor. Neither
/// notices a table that was truncated and blessed while the derivation was truncated the same
/// way, so the committed file is read here on its own terms: every published crate is in it, in
/// quantity, and each kind of record it claims to carry is present. The floor is far under the
/// real counts and says only that a table of a few items is wrong rather than quiet.
#[test]
fn the_table_covers_all_three_published_crates() {
    let path = table_path();
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    let mut per_crate: BTreeMap<&str, usize> = BTreeMap::new();
    let mut per_kind: BTreeMap<&str, usize> = BTreeMap::new();
    for line in text.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.splitn(3, '\t');
        let kind = parts.next().unwrap_or_default();
        let key = parts.next().unwrap_or_default();
        assert!(
            parts.next().is_some() && !kind.is_empty() && !key.is_empty(),
            "{path:?}: not a `kind<TAB>key<TAB>signature` record: {line:?}"
        );
        *per_kind.entry(kind).or_default() += 1;
        *per_crate
            .entry(key.split("::").next().unwrap_or_default())
            .or_default() += 1;
    }
    for name in ["metal2vulkan", "metal2vulkan-runtime", "metal2vulkan-air"] {
        let got = per_crate.get(name).copied().unwrap_or_default();
        assert!(
            got >= 1000,
            "{path:?} holds {got} record(s) for {name}, which is not that crate's public \
             surface: the table was truncated, not derived. Counts: {per_crate:?}"
        );
    }
    // The kinds the module documentation says a record may be. One that no longer appears is a
    // parse that stopped seeing a whole class of item, which the per-crate counts would not move
    // enough to catch.
    for kind in [
        "struct",
        "enum",
        "trait",
        "type",
        "fn",
        "const",
        "const-value",
        "field",
        "variant",
        "method",
        "assoc-const",
        "assoc-const-value",
        "trait-impl",
        "trait-item",
        "macro-template",
        "macro-decl",
        "macro-member",
    ] {
        assert!(
            per_kind.contains_key(kind),
            "{path:?} holds no `{kind}` record, so the derivation no longer sees that kind of \
             public item. Kinds: {per_kind:?}"
        );
    }
}
