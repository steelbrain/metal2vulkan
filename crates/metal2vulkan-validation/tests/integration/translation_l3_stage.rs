//! The translation worker runs `spirv-val` in its L3 stage, a fresh image of `corpus-triage` that
//! replaced the translating one, so the validator never runs beside the translation's memory: an
//! attempt's peak is the larger of the two, not their sum.
//!
//! A stand-in `spirv-val` (`METAL2VULKAN_SPIRV_VAL`) records whether its parent is the L3 stage and
//! accepts or rejects every module. The census of one public row must find the module validated
//! by a child of the L3 stage, and a rejection there must still be the row's recorded L3 outcome.

#![cfg(unix)]

use metal2vulkan_validation::{source, ScratchDir};
use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};
use std::process::Command;

/// A stand-in `spirv-val` in `dir`: `--version` prints `version`; otherwise it appends a line to
/// `calls` saying whether its parent is the L3 stage (a process whose arguments hold
/// `--l3-stage engine`) and exits with `status`, printing `diagnostic`.
///
/// It finds its parent with `pgrep`, never `ps`: macOS's `ps` is setuid root, and the worker's
/// process-group monitor cannot read a root process's memory (`proc_pid_rusage` answers `EPERM`),
/// which fails the attempt as a tooling error whenever a sample lands while `ps` runs.
fn stand_in(
    dir: &Path,
    name: &str,
    version: &str,
    calls: &Path,
    status: u8,
    diagnostic: &str,
) -> PathBuf {
    let path = dir.join(name);
    let script = format!(
        "#!/bin/sh\n\
         if [ \"$1\" = --version ]; then echo '{version}'; exit 0; fi\n\
         if pgrep -f -- '--l3-stage engine' | grep -qx \"$PPID\"; then\n\
         echo \"$PPID: --l3-stage engine\" >> '{calls}'\n\
         else\n\
         echo \"$PPID: not the L3 stage\" >> '{calls}'\n\
         fi\n\
         echo '{diagnostic}'\n\
         exit {status}\n",
        calls = calls.display(),
    );
    std::fs::write(&path, script).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path
}

/// `corpus-triage`'s engine census of the rows in `hashes` on `vk12-floor`, with `spirv_val`
/// as the validator: its stdout.
fn census(corpus: &Path, hashes: &Path, spirv_val: &Path) -> String {
    let output = Command::new(bin_exe!("corpus-triage"))
        .arg("--corpus")
        .arg(corpus)
        .args([
            "--audit",
            "translation",
            "--profile",
            "runtime-floor",
            "--hash-file",
        ])
        .arg(hashes)
        .args(["--summary-only", "--jobs", "1"])
        .env("METAL2VULKAN_SPIRV_VAL", spirv_val)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    assert!(
        output.status.success(),
        "{stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
    stdout
}

fn summary(stdout: &str) -> &str {
    stdout
        .lines()
        .find(|line| line.starts_with("translation-audit-summary\t"))
        .unwrap_or_else(|| panic!("{stdout}"))
}

#[test]
fn spirv_val_runs_in_the_l3_stage_that_replaced_the_translation() {
    if let Err(why) = metal2vulkan_devtools::tool_version("llvm-as") {
        eprintln!("SKIPPED: {why:?}");
        return;
    }
    let sources = source::public_source_set().unwrap();
    let row = sources
        .rows
        .iter()
        .find(|row| row.entry == "store_const" && row.label.ends_with("/store_const.ll"))
        .expect("the public store_const row");
    let scratch = ScratchDir::new("translation-l3-stage").unwrap();
    let corpus = scratch.path().join("corpus");
    std::fs::create_dir(&corpus).unwrap();
    let indexed = Command::new(bin_exe!("corpus-index"))
        .arg("--corpus")
        .arg(&corpus)
        .output()
        .unwrap();
    assert!(
        indexed.status.success(),
        "{}",
        String::from_utf8_lossy(&indexed.stderr)
    );
    let hashes = scratch.path().join("hashes.txt");
    std::fs::write(&hashes, format!("{}\n", row.air_sha256)).unwrap();
    let calls = scratch.path().join("calls.txt");

    let accepting = stand_in(
        scratch.path(),
        "accepting",
        "SPIRV-Tools vL3-stage accepting",
        &calls,
        0,
        "",
    );
    let stdout = census(&corpus, &hashes, &accepting);
    assert!(summary(&stdout).contains("\ttranslated=1\t"), "{stdout}");
    let parents = std::fs::read_to_string(&calls).unwrap();
    assert!(!parents.is_empty(), "spirv-val never ran: {stdout}");
    for parent in parents.lines() {
        assert!(
            parent.contains("--l3-stage engine"),
            "spirv-val ran under {parent:?}, not the worker's L3 stage"
        );
    }

    // A rejection in the L3 stage is the row's recorded outcome, with the validator's diagnostic.
    let rejecting = stand_in(
        scratch.path(),
        "rejecting",
        "SPIRV-Tools vL3-stage rejecting",
        &calls,
        1,
        "error: the stand-in rejects every module",
    );
    let stdout = census(&corpus, &hashes, &rejecting);
    let line = summary(&stdout);
    assert!(
        line.contains("\ttranslated=0\t") && line.contains("\tdefects=1\t"),
        "{stdout}"
    );
    let index = rusqlite::Connection::open(corpus.join(".index.sqlite")).unwrap();
    let details: Vec<String> = index
        .prepare("SELECT detail FROM new_engine_translation_audit WHERE air_sha256 = ?1 AND detail IS NOT NULL")
        .unwrap()
        .query_map([&row.air_sha256], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert!(
        details
            .iter()
            .any(|d| d.starts_with("L3: ") && d.contains("the stand-in rejects every module")),
        "{details:?}"
    );
}
