//! Checks of the test trees that read the checkout through this crate's directory helpers.

use std::path::{Path, PathBuf};

/// The trees, relative to the workspace, whose tests run from nextest archives carried to other
/// machines (`cargo nextest run --archive-file ... --workspace-remap <checkout>`): all of the
/// validation crate (its lib test binary and the library code it calls; its integration tests and
/// binaries ride a full archive).
const CARRIED: [&str; 2] = [
    "crates/metal2vulkan-validation/src",
    "crates/metal2vulkan-validation/tests",
];

/// The `.rs` files at or below `path`.
fn sources(path: &Path, out: &mut Vec<PathBuf>) {
    if path.is_dir() {
        for entry in std::fs::read_dir(path).unwrap() {
            sources(&entry.unwrap().path(), out);
        }
    } else if path.extension().is_some_and(|extension| extension == "rs") {
        out.push(path.to_path_buf());
    }
}

/// `env!("CARGO_MANIFEST_DIR")` is the directory the test binary was built in. A carried archive
/// runs where that directory does not exist, so a test that reads a fixture, a committed case, a
/// profile or a source file by it fails there ("No such file or directory"), or finds nothing to
/// read. The carried trees read the checkout through [`crate::workspace_dir`] and the helpers on it,
/// which take the directory the test runner names as the test runs; this lists every baked site
/// left, as `file:line`.
#[test]
fn the_carried_tests_read_the_checkout_where_the_runner_names_it() {
    let root = crate::workspace_dir();
    let mut files = Vec::new();
    for tree in CARRIED {
        let path = root.join(tree);
        assert!(path.exists(), "{} is missing", path.display());
        sources(&path, &mut files);
    }
    files.sort();
    assert!(files.len() > 300, "{} sources read", files.len());
    let mut baked = Vec::new();
    for file in &files {
        let text = std::fs::read_to_string(file).unwrap();
        let relative = file.strip_prefix(&root).unwrap().display().to_string();
        for (at, line) in text.lines().enumerate() {
            if line.contains("env!(\"CARGO_MANIFEST_DIR\")") {
                baked.push(format!("{relative}:{}", at + 1));
            }
        }
    }
    assert!(
        baked.is_empty(),
        "{} reads of the build's directory in the carried tests (read through \
         metal2vulkan_devtools::{{workspace_dir, validation_dir, fixtures_dir, engine_dir}}):\n{}",
        baked.len(),
        baked.join("\n")
    );
}
