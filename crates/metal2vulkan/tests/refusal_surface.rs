//! The refusal surface, and the basis of every refusal that claims Metal stops an input.
//!
//! `scripts/refusal-surface/refusal_surface.py` reads every typed refusal of the three published
//! crates and the new runtime's floor from the sources and writes `docs/REFUSALS.md`. Its **Basis** column says where the
//! belief that an input must be refused comes from: derived from the construction site wherever the
//! site states it (a capability a device lacks, a piece not built, a cap of this crate's), and
//! declared in the refusal's own doc comment, as a last line `Basis: <value> (the evidence)`,
//! wherever the claim is about Metal itself, which no construction site can state.
//!
//! The runtime's `Problem` is nothing but such claims — every variant says Metal refuses, asserts,
//! answers nil or ends its process over an input — so every variant of it carries a declared
//! basis, and this test is what makes an absent or malformed one fail. A refusal whose basis is not
//! settled says `Basis: unknown (why)`: the project's rule is that unknown stays visible, not that
//! every row gets a green word.
//!
//! Most of those claims are not in `Problem` itself but in the problem enums it forwards to, and
//! those are read too. The scan finds them by structure: a variant whose whole payload is one enum
//! the runtime declares forwards to it, and the enum it names is read as `Problem` is, so a
//! sub-problem added later is picked up with it. Where the forwarding variant says
//! `Basis: delegated(ThatEnum)`, every variant of `ThatEnum` declares a basis of its own and this
//! test fails over an absent or malformed one; where it states a basis itself (a runtime extension
//! Metal has no counterpart for), the family derives that value from it.
//!
//! The vocabulary is one list, in the script: `metal-measured` and `metal-documented` (Metal itself
//! refuses, on a run of ours or on Apple's own words), `metal-takes` (measured: plain Metal takes
//! the input, so the refusal reaches past Metal — an open defect, recorded), `device`, `ours`,
//! `unknown`, and `delegated(ThatProblem)` for a variant that only forwards.
//!
//! # What it cannot see
//!
//! - Whether a citation is *true*. It holds a claim about Metal to carrying evidence that names a
//!   device or a release, never to that evidence being right.
//! - Whether the grade matches the evidence beside it. A variant whose prose reports a measurement
//!   that names no device or release reads `unknown` here, which is the honest answer and not the
//!   last one: the measurement is what has to be re-run, not the word.
//! - Anything a `Problem` reaches other than through a variant whose payload is one problem enum:
//!   a boxed error of another type (`ArgumentEncoderProblem::Layout`) carries its own refusals,
//!   and those are not rows here.
//! - The other eight sections' per-variant Metal claims. Their basis is derived from the
//!   constructor, which states the device axis and not Metal's answer; a declared `Basis:` line
//!   there is read and held to the vocabulary, and is not yet required.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The repository root.
fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The generator run with `verb`, as its own words and its exit status.
fn surface(verb: &str) -> (bool, String) {
    let script = root().join("scripts/refusal-surface/refusal_surface.py");
    // `-I` so the run reads no module beside the interpreter's caller, `-B` so it leaves no
    // `__pycache__` beside the script: a test writes nothing into the tree it is checking.
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

/// Every refusal that claims Metal stops an input carries the basis of that claim.
///
/// The failure names each variant and what is wrong with its `Basis:` line. Add the line to the
/// variant's doc comment, as its last paragraph, and regenerate
/// `docs/REFUSALS.md`; `--missing-basis` lists them all.
#[test]
fn every_metal_refusal_states_its_basis() {
    let (ok, out) = surface("--missing-basis");
    assert!(ok, "{out}");
    assert!(
        out.trim().is_empty(),
        "refusals whose basis is absent or malformed:\n{out}\n\
         Each variant of the runtime's `Problem` ends its doc comment with `Basis: <value> (the \
         evidence)`; a basis that is not settled is `Basis: unknown (why)`, never nothing. The \
         values are in scripts/refusal-surface/refusal_surface.py."
    );
}

/// `docs/REFUSALS.md` is the surface the sources state, row for row, basis included.
///
/// Regenerate it with `python3 scripts/refusal-surface/refusal_surface.py`; never edit the tables.
#[test]
fn the_refusal_surface_is_the_committed_document() {
    let (ok, out) = surface("--check");
    assert!(ok, "{out}");
}

/// The basis marker's parse reads what it claims and refuses what it must.
///
/// The column is worth no more than that parse. A marker shape it stops recognising takes a claim
/// out of the table and a quieter table then reads like the truth, which is the very failure this
/// whole column exists to catch — so `--selftest` runs the parse over markers written in the
/// script, and every case asserts both directions: a good marker read as what it says, a bad one
/// named and refused.
#[test]
fn the_basis_parse_still_refuses_what_it_must() {
    let (ok, out) = surface("--selftest");
    assert!(ok, "{out}");
}

/// Every problem enum a `Problem` forwards to is read, and read from a file with variants in it.
///
/// The basis of a forwarded claim is only as good as the scan that reaches it: were the discovery
/// to stop finding a family, that family's variants would simply leave the table and
/// `--missing-basis` would go quiet, which is the failure the whole column exists to catch. So the
/// count is cross-checked against the sources rather than against the script: this test counts the
/// `delegated(` markers in the runtime's `Problem` itself, and the scan must have resolved exactly
/// that many families, each to a file and to at least one variant.
#[test]
fn the_surface_reads_the_problems_a_problem_forwards_to() {
    let (ok, out) = surface("--delegations");
    assert!(ok, "{out}");
    let refusals = root().join("docs/REFUSALS.md");
    let document = std::fs::read_to_string(&refusals)
        .unwrap_or_else(|error| panic!("{}: {error}", refusals.display()));
    let mut required = 0;
    for line in out.lines() {
        let field: Vec<&str> = line.split('\t').collect();
        assert_eq!(
            field.len(),
            6,
            "a delegations line of another shape: {line}"
        );
        let (enum_name, kind, file, variants) = (field[0], field[1], field[2], field[3]);
        assert_ne!(
            file, "?",
            "{enum_name} forwards from {} to no file",
            field[4]
        );
        assert_ne!(
            variants, "0",
            "{enum_name} was read from {file} with no variant in it"
        );
        // And its variants reach the table: a family the scan finds but does not emit is the
        // quiet table this column exists to prevent.
        assert!(
            document.contains(&format!("| `{enum_name}::")),
            "{enum_name} was read from {file} but no row of it is in {}",
            refusals.display()
        );
        if kind == "required" {
            required += 1;
        }
    }
    let error_rs = root().join("crates/metal2vulkan-runtime/src/error.rs");
    let markers = std::fs::read_to_string(&error_rs)
        .unwrap_or_else(|error| panic!("{}: {error}", error_rs.display()))
        .matches("Basis: delegated(")
        .count();
    assert!(
        markers > 0,
        "no `Basis: delegated(` marker in {}",
        error_rs.display()
    );
    assert_eq!(
        required, markers,
        "{markers} `Basis: delegated(` markers in the runtime's `Problem`, but the scan resolved \
         {required} families to read. Each delegated variant's payload must be an enum the runtime \
         declares; `--delegations` lists what the scan found.\n{out}"
    );
}
