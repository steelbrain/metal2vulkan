//! No merge-conflict marker reaches the tree. A landing that resolves a conflict by hand can leave
//! `git`'s markers behind, and nothing else notices where the file is one no compiler reads: on
//! 2026-10-23 a crate's README carried a pair of them for an hour. The check walks the tree rather
//! than asking `git`, so it holds in a packaged crate and in a tree without git.

use std::fs;
use std::path::{Path, PathBuf};

/// The directories under the repository's root that hold what we write by hand. The root's own
/// files are read too, without descending into anything this list does not name.
const ROOTS: [&str; 4] = ["crates", "docs", "scripts", ".github"];

/// The directories the walk does not descend into, by their path from the root. They are
/// generated, assembled or committed whole, never hand-resolved, and walking the corpus alone is
/// hundreds of thousands of entries.
const SKIP: [&str; 5] = [
    "crates/metal2vulkan-validation/corpus",
    "crates/metal2vulkan-validation/fixtures",
    "crates/metal2vulkan-conformance/recordings",
    "target",
    ".git",
];

/// The extensions of the files the check reads: what a conflict could reach and a human wrote.
const EXTENSIONS: [&str; 11] = [
    "rs", "md", "toml", "tsv", "yml", "yaml", "sh", "py", "txt", "sha256", "ll",
];

/// `git`'s labelled markers, each at the start of a line and followed by the side it names: the
/// two a merge writes and the third `diff3` adds. The separator between the sides is seven `=`
/// alone, which is also Markdown's underline for a heading, so the check leaves it to these three.
/// Each is built rather than written out, so this file is not itself a hit.
fn markers() -> [String; 3] {
    [
        format!("{} ", "<".repeat(7)),
        format!("{} ", ">".repeat(7)),
        format!("{} ", "|".repeat(7)),
    ]
}

/// Whether the check reads a file of this name.
fn reads(path: &Path) -> bool {
    path.extension()
        .is_some_and(|ext| EXTENSIONS.contains(&ext.to_string_lossy().as_ref()))
}

/// Every file under `dir` the check reads, depth first, skipping [`SKIP`]'s directories. With
/// `descend` false only the files directly in `dir` are taken, which is how the root is read.
fn walk(root: &Path, dir: &Path, descend: bool, found: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if entry.file_type().is_ok_and(|kind| kind.is_dir()) {
            let relative = path.strip_prefix(root).unwrap_or(&path).to_string_lossy();
            if descend && !SKIP.contains(&relative.as_ref()) {
                walk(root, &path, true, found);
            }
        } else if reads(&path) {
            found.push(path);
        }
    }
}

#[test]
fn no_conflict_markers_in_the_tree() {
    let root =
        fs::canonicalize(metal2vulkan_devtools::workspace_dir()).expect("the repository's root");
    let markers = markers();
    let mut files = Vec::new();
    walk(&root, &root, false, &mut files);
    for dir in ROOTS {
        walk(&root, &root.join(dir), true, &mut files);
    }
    assert!(
        files.len() > 2000,
        "the walk read only {} files, so it is looking in the wrong place",
        files.len()
    );
    let mut hits = Vec::new();
    for file in &files {
        let Ok(text) = fs::read_to_string(file) else {
            continue;
        };
        for (number, line) in text.lines().enumerate() {
            if markers.iter().any(|marker| line.starts_with(marker)) {
                hits.push(format!(
                    "{}:{}: {line}",
                    file.strip_prefix(&root).unwrap_or(file).display(),
                    number + 1
                ));
            }
        }
    }
    assert!(
        hits.is_empty(),
        "{} conflict marker lines reached the tree:\n{}",
        hits.len(),
        hits.join("\n")
    );
}
