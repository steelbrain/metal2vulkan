//! Every committed `.ll` fixture assembles with the oldest LLVM a gate assembles it with, and the
//! product decodes the module and its AIR metadata ABI.
//!
//! Linux CI and the Linux gate image assemble the fixtures with Ubuntu 26.04's LLVM 21, while a
//! newer `llvm-dis` may write text LLVM 21's `llvm-as` rejects (LLVM 22's one-argument
//! `llvm.lifetime.*`, LLVM 23's `splat (float f0x...)`). The pinned major is the `llvm:` line of
//! `public/engine/TOOLCHAIN`, the major `scripts/regen-air-fixtures` writes the owned
//! companions with. Every `.ll` under `validation/fixtures/` goes through an `llvm-as` of that
//! major, except the typed-pointer `<name>.air.ll` halves, which only `xcrun metal-as` reads.
//!
//! The tool is the first of `METAL2VULKAN_LLVM_AS`, `llvm-as-<major>` and `llvm-as` on `PATH`,
//! and Homebrew's `llvm@<major>` whose `--version` reports that major. Without one the test skips
//! visibly, unless `METAL2VULKAN_TEST_REQUIRE_PINNED_LLVM=1` (CI's Linux job, where Ubuntu's
//! `llvm` package is the pinned major), which makes a missing tool a failure.
//!
//! No fixture is excused from either check. A hand-written fixture is held to the same text and the
//! same metadata grammar as one the Metal front end compiles: the two that were not (a misnumbered
//! SSA value, and a nested `air.struct_type_info` spelled as a bare trailing node) had been listed
//! as known exceptions here and in `tests/air_abi.rs`, and their authored cases went stale behind
//! the lists.

use std::path::{Path, PathBuf};
use std::process::Command;

use metal2vulkan::Budget;
use metal2vulkan_air::budget::Meter;
use metal2vulkan_validation::ScratchDir;

const REQUIRE: &str = "METAL2VULKAN_TEST_REQUIRE_PINNED_LLVM";

fn fixtures_root() -> PathBuf {
    metal2vulkan_devtools::validation_dir().join("fixtures")
}

/// The LLVM major `public/engine/TOOLCHAIN` pins (`llvm: <major> (...)`).
fn pinned_major() -> u32 {
    let path = fixtures_root().join("public/engine/TOOLCHAIN");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    text.lines()
        .find_map(|line| line.strip_prefix("llvm: "))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|major| major.parse().ok())
        .unwrap_or_else(|| panic!("{} has no `llvm: <major>` line", path.display()))
}

/// The version line and major of an LLVM tool (`Ubuntu LLVM version 21.1.8`), if it runs.
fn llvm_version(tool: &Path) -> Option<(String, u32)> {
    let output = Command::new(tool).arg("--version").output().ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    let line = text.lines().find(|line| line.contains("version "))?.trim();
    let after = &line[line.find("version ")? + "version ".len()..];
    let major = after.split('.').next()?.parse().ok()?;
    Some((line.to_owned(), major))
}

/// The first `llvm-as` of `major` (module docs), or every candidate found with its version.
fn pinned_llvm_as(major: u32) -> Result<(PathBuf, String), Vec<String>> {
    let mut candidates: Vec<PathBuf> = std::env::var_os("METAL2VULKAN_LLVM_AS")
        .map(PathBuf::from)
        .into_iter()
        .collect();
    let path = std::env::var_os("PATH").unwrap_or_default();
    for name in [format!("llvm-as-{major}"), "llvm-as".to_owned()] {
        candidates.extend(std::env::split_paths(&path).map(|dir| dir.join(&name)));
    }
    for prefix in ["/opt/homebrew/opt", "/usr/local/opt"] {
        candidates.push(Path::new(prefix).join(format!("llvm@{major}/bin/llvm-as")));
    }
    let mut seen = Vec::new();
    for tool in candidates.into_iter().filter(|tool| tool.is_file()) {
        match llvm_version(&tool) {
            Some((line, found)) if found == major => return Ok((tool, line)),
            Some((line, _)) => seen.push(format!("{}: {line}", tool.display())),
            None => seen.push(format!("{}: no LLVM version", tool.display())),
        }
    }
    Err(seen)
}

/// Every `.ll` below `dir` that is not an `.air.ll`, sorted.
fn ll_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries =
        std::fs::read_dir(dir).unwrap_or_else(|error| panic!("read {}: {error}", dir.display()));
    for entry in entries {
        let path = entry.unwrap().path();
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        if path.is_dir() {
            ll_files(&path, out);
        } else if name.ends_with(".ll") && !name.ends_with(".air.ll") {
            out.push(path);
        }
    }
}

#[test]
fn every_committed_ll_fixture_assembles_with_the_pinned_llvm_and_decodes() {
    let major = pinned_major();
    let (tool, version) = match pinned_llvm_as(major) {
        Ok(found) => found,
        Err(seen) => {
            let why = format!(
                "no llvm-as of the pinned LLVM {major} (set METAL2VULKAN_LLVM_AS or put \
                 llvm-as-{major} on PATH); found: {seen:?}"
            );
            assert!(
                std::env::var_os(REQUIRE).is_none_or(|v| v != "1"),
                "{REQUIRE}=1: {why}"
            );
            eprintln!(
                "SKIPPED every_committed_ll_fixture_assembles_with_the_pinned_llvm_and_decodes: \
                 {why}"
            );
            return;
        }
    };
    let root = fixtures_root();
    let mut files = Vec::new();
    ll_files(&root, &mut files);
    files.sort();
    assert!(
        !files.is_empty(),
        "no .ll fixtures under {}",
        root.display()
    );
    let scratch = ScratchDir::new("pinned-llvm-fixtures").expect("scratch directory");
    let bitcode = scratch.path().join("fixture.bc");
    let (mut rejected, mut undecoded) = (Vec::new(), Vec::new());
    for path in &files {
        let relative = path.strip_prefix(&root).unwrap().to_string_lossy();
        let relative = relative.replace(std::path::MAIN_SEPARATOR, "/");
        let output = Command::new(&tool)
            .arg(path)
            .arg("-o")
            .arg(&bitcode)
            .output()
            .unwrap_or_else(|error| panic!("run {}: {error}", tool.display()));
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            rejected.push(format!("{relative}:\n{}", stderr.trim_end()));
            continue;
        }
        let bytes = std::fs::read(&bitcode)
            .unwrap_or_else(|error| panic!("read {}: {error}", bitcode.display()));
        // The module and its metadata ABI together: `air::decode` refuses a module whose ABI
        // does not decode.
        if let Err(error) = metal2vulkan_air::decode(&bytes, &mut Meter::new(&Budget::default())) {
            undecoded.push(format!("{relative}: {error}"));
        }
    }
    assert!(
        rejected.is_empty(),
        "{} rejects {} committed .ll fixture(s); write them as LLVM {major}'s llvm-dis would:\n{}",
        version,
        rejected.len(),
        rejected.join("\n")
    );
    assert!(
        undecoded.is_empty(),
        "the product does not decode {} committed .ll fixture(s) assembled by {version}; write \
         their metadata in Apple's AIR ABI grammar:\n{}",
        undecoded.len(),
        undecoded.join("\n")
    );
    eprintln!(
        "{} .ll fixtures assemble with {} ({version}) and decode",
        files.len(),
        tool.display()
    );
}
