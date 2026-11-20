//! The Metal API coverage map (`scripts/metal-coverage/coverage.tsv`, which writes
//! docs/METAL-COVERAGE.md) names only counterparts that exist.
//!
//! Each row is a public item of the macOS SDK's Metal headers, by name. An implemented row names
//! its counterpart in the runtime (`metal2vulkan-rt/src`) or the translator (`src`, `air/src`):
//! a type (`Type`), a method or associated item (`Type::name`), or a field (`Type.name`), or
//! several of these, comma-separated. This test reads those sources and holds every counterpart
//! to a public item of that name: the type declared `pub`, a method `pub fn` inside an `impl` of
//! the type, a field
//! `pub` inside the type's struct. A renamed or removed counterpart fails here until the table
//! says so. It also holds the table's vocabulary (status, size) and the generated document's
//! totals to the table. `coverage.py --sdk` checks the table's names against the SDK on a Mac.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// The repository root.
fn root() -> PathBuf {
    metal2vulkan_devtools::workspace_dir()
}

/// The table's rows, by column name.
fn rows() -> Vec<BTreeMap<String, String>> {
    let text = fs::read_to_string(root().join("scripts/metal-coverage/coverage.tsv")).unwrap();
    let mut lines = text.lines();
    let head: Vec<&str> = lines.next().unwrap().split('\t').collect();
    lines
        .map(|line| {
            let fields: Vec<&str> = line.split('\t').collect();
            assert_eq!(fields.len(), head.len(), "a row of another width: {line}");
            head.iter()
                .zip(fields)
                .map(|(h, f)| (h.to_string(), f.to_string()))
                .collect()
        })
        .collect()
}

/// Every `.rs` file under `dir`, with comments blanked (so a brace or name in a comment counts
/// for nothing).
fn sources(dir: &Path, out: &mut Vec<String>) {
    let mut entries: Vec<_> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(uncommented(&fs::read_to_string(&path).unwrap()));
        }
    }
}

/// `text` with `//` comments removed, line by line (a `//` inside a string literal is rare
/// enough in declarations to ignore).
fn uncommented(text: &str) -> String {
    text.lines()
        .map(|line| match line.find("//") {
            Some(at) => &line[..at],
            None => line,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The text of the brace block that opens at or after `from` in `text`.
fn block(text: &str, from: usize) -> &str {
    let Some(open) = text[from..].find('{').map(|i| from + i) else {
        return "";
    };
    let mut depth = 0usize;
    for (i, c) in text[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return &text[open..open + i + 1];
                }
            }
            _ => {}
        }
    }
    &text[open..]
}

/// Whether `text` holds `word` as a whole identifier at `at`.
fn ident_at(text: &str, at: usize, word: &str) -> bool {
    let ident = |c: char| c.is_ascii_alphanumeric() || c == '_';
    text[at..].starts_with(word)
        && !text[..at].ends_with(ident)
        && !text[at + word.len()..].starts_with(ident)
}

/// Every position of `word` as a whole identifier in `text`.
fn positions<'a>(text: &'a str, word: &'a str) -> impl Iterator<Item = usize> + 'a {
    text.match_indices(word)
        .map(|(at, _)| at)
        .filter(move |&at| ident_at(text, at, word))
}

/// Whether a file declares type `ty` public.
fn declares(text: &str, ty: &str) -> bool {
    ["pub struct ", "pub enum ", "pub trait ", "pub type "]
        .iter()
        .any(|kw| {
            text.match_indices(kw)
                .any(|(at, _)| ident_at(text, at + kw.len(), ty))
        })
}

/// The bodies of `impl` blocks of `ty` in `text`: `impl Ty`, `impl<..> Ty<..>`, `impl Trait for
/// Ty`.
fn impls<'a>(text: &'a str, ty: &str) -> Vec<&'a str> {
    let mut out = Vec::new();
    for (at, _) in text.match_indices("impl") {
        if !ident_at(text, at, "impl") {
            continue;
        }
        let Some(open) = text[at..].find('{').map(|i| at + i) else {
            continue;
        };
        let header = &text[at..open];
        if header.contains(';') {
            continue;
        }
        // The implemented type is the header's last path segment before its generics, after
        // `for` where there is one.
        let target = header.rsplit(" for ").next().unwrap_or(header);
        let target = target.trim_start_matches("impl");
        let target = target.split(" where ").next().unwrap_or(target);
        let ends_with_ty = positions(target, ty).any(|p| {
            let rest = target[p + ty.len()..].trim_start();
            rest.is_empty() || rest.starts_with('<')
        });
        if ends_with_ty {
            out.push(block(text, at));
        }
    }
    out
}

/// Whether the sources hold `name` as an associated item of `ty` (`pub fn`, `pub const`).
fn has_item(files: &[String], ty: &str, name: &str) -> bool {
    files.iter().any(|text| {
        impls(text, ty).iter().any(|body| {
            ["pub fn ", "pub const fn ", "pub unsafe fn ", "pub const "]
                .iter()
                .any(|kw| {
                    body.match_indices(kw)
                        .any(|(at, _)| ident_at(body, at + kw.len(), name))
                })
        })
    })
}

/// Whether the sources hold `name` as a public field of struct `ty`.
fn has_field(files: &[String], ty: &str, name: &str) -> bool {
    files.iter().any(|text| {
        text.match_indices("pub struct ")
            .filter(|(at, _)| ident_at(text, at + "pub struct ".len(), ty))
            .any(|(at, _)| {
                let body = block(text, at);
                body.match_indices("pub ").any(|(f, _)| {
                    ident_at(body, f + 4, name)
                        && body[f + 4 + name.len()..].trim_start().starts_with(':')
                })
            })
    })
}

/// Every implemented row's counterpart exists in its crate's sources, and every row is well
/// formed.
#[test]
fn every_counterpart_the_coverage_map_names_exists() {
    let root = root();
    let mut runtime = Vec::new();
    sources(&root.join("crates/metal2vulkan-rt/src"), &mut runtime);
    let mut translator = Vec::new();
    sources(&root.join("crates/metal2vulkan/src"), &mut translator);
    sources(&root.join("crates/metal2vulkan-air/src"), &mut translator);
    let mut missing = Vec::new();
    for row in rows() {
        let what = format!("{} {}", row["container"], row["name"]);
        match row["status"].as_str() {
            "I" => {
                let files = match row["crate"].as_str() {
                    "runtime" => &runtime,
                    "translator" => &translator,
                    other => panic!("{what}: crate {other:?}"),
                };
                let counterpart = row["counterpart"].as_str();
                let found = counterpart.split(", ").all(|piece| {
                    if let Some((ty, name)) = piece.split_once("::") {
                        has_item(files, ty, name)
                    } else if let Some((ty, name)) = piece.split_once('.') {
                        has_field(files, ty, name)
                    } else {
                        files.iter().any(|text| declares(text, piece))
                    }
                });
                if !found {
                    missing.push(format!("{what}: {} {counterpart}", row["crate"]));
                }
            }
            "M" => assert!(
                ["S", "M", "L", "XL"].contains(&row["size"].as_str()) && !row["group"].is_empty(),
                "{what}: a missing row needs a size and a group"
            ),
            "O" => assert!(!row["note"].is_empty(), "{what}: out of scope says why"),
            other => panic!("{what}: status {other:?}"),
        }
    }
    assert!(
        missing.is_empty(),
        "counterparts the coverage map names that the sources lack:\n{}",
        missing.join("\n")
    );
}

/// docs/METAL-COVERAGE.md's totals are the table's (`coverage.py` writes it).
#[test]
fn the_coverage_documents_totals_are_the_tables() {
    let rows = rows();
    let doc = fs::read_to_string(root().join("docs/METAL-COVERAGE.md")).unwrap();
    for (status, name) in [
        ("I", "implemented"),
        ("M", "missing"),
        ("O", "out of scope"),
    ] {
        let count = rows.iter().filter(|r| r["status"] == status).count();
        let line = format!("| {name} | {count} |");
        assert!(
            doc.contains(&line),
            "docs/METAL-COVERAGE.md lacks `{line}`: run scripts/metal-coverage/coverage.py"
        );
    }
    let all = format!("| all | {} |", rows.len());
    assert!(doc.contains(&all), "docs/METAL-COVERAGE.md lacks `{all}`");
}
