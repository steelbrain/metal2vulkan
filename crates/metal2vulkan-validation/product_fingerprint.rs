//! Validation's product fingerprint: the dependency of candidate observations, of both
//! translation-census keys, and of the structural-triage analyzer key, which add only versioned
//! validation semantics to it. `build.rs` sets `METAL2VULKAN_PRODUCT_FINGERPRINT` from
//! [`product_fingerprint`], and the crate's tests take the same function over synthetic
//! repositories: one definition, so the build and the tests cannot walk different trees.

use crate::build_support::{hash_files, product_hash};
use sha2::Digest;
use std::fs;
use std::path::{Path, PathBuf};

/// The product fingerprint of the workspace's crates directory `crates`, as hex: the translator's
/// own sources (`metal2vulkan/`, `build_support::product_hash`: exactly the files it can compile or
/// read, the AIR front end's crate and the workspace's manifest with them), then the dev-tools
/// crate's (`metal2vulkan-devtools/`: its manifest and every
/// file under its `src/`), whose assemblers give the public rows and cases their bitcode, and
/// nothing of `metal2vulkan-validation/`.
pub fn product_fingerprint(crates: &Path) -> String {
    let mut hash = product_hash(&crates.join("metal2vulkan"));
    hash_files(&mut hash, crates, &devtools_files(crates));
    hex(&hash.finalize())
}

/// The dev-tools crate's manifest and every file under its `src/`, sorted.
fn devtools_files(crates: &Path) -> Vec<PathBuf> {
    let devtools = crates.join("metal2vulkan-devtools");
    let mut files = vec![devtools.join("Cargo.toml")];
    collect_files(&devtools.join("src"), &mut files);
    files.sort();
    files
}

/// Every file below `directory`, pushed in sorted order; Cargo reruns the build script when
/// anything under it changes.
pub fn collect_files(directory: &Path, output: &mut Vec<PathBuf>) {
    println!("cargo:rerun-if-changed={}", directory.display());
    let mut entries = fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("read {}: {error}", directory.display()))
        .map(|entry| entry.expect("directory entry").path())
        .collect::<Vec<_>>();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            collect_files(&path, output);
        } else if path.is_file() {
            output.push(path);
        }
    }
}

/// `bytes` as lowercase hex, two digits a byte: the form of every digest string here.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
