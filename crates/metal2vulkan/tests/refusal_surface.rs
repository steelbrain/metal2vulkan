//! The refusal surface, and the basis of every refusal that claims Metal stops an input.
//!
//! `scripts/refusal-surface/refusal_surface.py` reads every typed refusal of the three published
//! crates from the sources and writes `docs/REFUSALS.md`. Its **Basis**
//! column says where the belief that an input must be refused comes from: derived from the
//! construction site wherever the site states it (a capability a device lacks, a piece not built, a
//! cap of this crate's), and declared in the refusal's own doc comment, as a last line
//! `Basis: <value> (the evidence)`, wherever the claim is about Metal itself, which no construction
//! site can state.
//!
//! The runtime's refusal enums (each variant of its `Error` whose whole payload is one enum it
//! declares) are such claims, so every variant of them carries a declared basis, and this test is
//! what makes an absent or malformed one fail. A refusal whose basis is not settled says
//! `Basis: unknown (why)`: the project's rule is that unknown stays visible, not that every row gets
//! a green word.
//!
//! The vocabulary is one list, in the script: `metal-measured` and `metal-documented` (Metal itself
//! refuses, on a run of ours or on Apple's own words), `metal-takes` (measured: plain Metal takes
//! the input, so the refusal reaches past Metal — an open defect, recorded), `device`, `ours` and
//! `unknown`.
//!
//! # What it cannot see
//!
//! - Whether a citation is *true*. It holds a claim about Metal to carrying evidence that names a
//!   device or a release, never to that evidence being right.
//! - Whether the grade matches the evidence beside it. A variant whose prose reports a measurement
//!   that names no device or release reads `unknown` here, which is the honest answer and not the
//!   last one: the measurement is what has to be re-run, not the word.
//! - The other sections' per-variant Metal claims. Their basis is derived from the constructor,
//!   which states the device axis and not Metal's answer; a declared `Basis:` line there is read and
//!   held to the vocabulary, and is not yet required.

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
         Each variant of the runtime's refusal enums ends its doc comment with \
         `Basis: <value> (the evidence)`; a basis that is not settled is `Basis: unknown (why)`, \
         never nothing. The values are in scripts/refusal-surface/refusal_surface.py."
    );
}

/// `docs/REFUSALS.md` is the surface the sources state, row for row, basis included, and the
/// runtime's generated `error/basis.rs` answers each refusal's basis from the same lines.
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
