//! What each crate the workspace publishes carries on its own. Every crate lies under `crates/` and
//! is packaged from its own directory, so a file at the repository's root reaches no package: the
//! license files, which every published package ships, are copied into each published crate, and
//! these tests hold every copy to the root's file of the same name. The license is dual,
//! `GPL-2.0-or-later OR LGPL-3.0-or-later` at the recipient's choice: `LICENSE` is the notice that
//! says so, and the other files are the texts it names (the LGPL v3 incorporates the GPL v3).

use std::fs;
use std::path::Path;

/// The license files at the repository's root, each copied into every published crate.
const LICENSE_FILES: [&str; 4] = [
    "LICENSE",
    "LICENSE-GPL-2.0",
    "LICENSE-GPL-3.0",
    "LICENSE-LGPL-3.0",
];

/// The workspace members under `crates/` whose manifest does not say `publish = false`.
fn published_crates(crates: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(crates)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|dir| {
            fs::read_to_string(dir.join("Cargo.toml")).is_ok_and(|manifest| {
                !manifest.lines().any(|line| {
                    line.split('#')
                        .next()
                        .unwrap_or_default()
                        .split_whitespace()
                        .collect::<String>()
                        == "publish=false"
                })
            })
        })
        .map(|dir| dir.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

/// The names of the files directly in `dir` that start with `LICENSE`, sorted.
fn license_files(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .unwrap_or_else(|error| panic!("{}: {error}", dir.display()))
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name.starts_with("LICENSE"))
        .collect();
    names.sort();
    names
}

#[test]
fn every_published_crate_carries_the_repository_license() {
    let crates = metal2vulkan_devtools::workspace_dir().join("crates");
    let root = crates.join("..");
    assert_eq!(
        license_files(&root),
        LICENSE_FILES,
        "the repository's license files changed: list them here and copy each into every \
         published crate"
    );
    let published = published_crates(&crates);
    assert_eq!(
        published,
        ["metal2vulkan", "metal2vulkan-air", "metal2vulkan-runtime"],
        "the crates without `publish = false` changed: give a new one copies of the repository's \
         license files, then list it here"
    );
    for name in &published {
        let dir = crates.join(name);
        assert_eq!(
            license_files(&dir),
            LICENSE_FILES,
            "crates/{name} carries other license files than the repository's root"
        );
        for file in LICENSE_FILES {
            let original = fs::read(root.join(file)).expect("the repository's license file");
            let copy = fs::read(dir.join(file)).unwrap_or_else(|error| {
                panic!("crates/{name}/{file}: {error} (copy the repository's {file} there)")
            });
            assert!(
                copy == original,
                "crates/{name}/{file} differs from the repository's {file}"
            );
        }
    }
}

#[test]
fn the_license_notice_names_the_declared_license_and_its_texts() {
    let root = metal2vulkan_devtools::workspace_dir();
    let manifest = fs::read_to_string(root.join("Cargo.toml")).expect("the workspace's manifest");
    let declared = manifest
        .lines()
        .find_map(|line| line.strip_prefix("license = "))
        .expect("the workspace's `license`")
        .trim_matches('"');
    let notice = fs::read_to_string(root.join("LICENSE")).expect("the repository's LICENSE");
    let tag = format!("SPDX-License-Identifier: {declared}");
    assert!(
        notice.lines().any(|line| line == tag),
        "LICENSE does not say `{tag}`, the license the manifests declare"
    );
    for text in &LICENSE_FILES[1..] {
        assert!(notice.contains(text), "LICENSE does not name {text}");
    }
}
