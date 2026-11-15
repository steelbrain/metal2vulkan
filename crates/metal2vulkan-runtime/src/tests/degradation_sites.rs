//! Every [`Degradation`] kind has a lane site, held without a device.
//!
//! The validation crate's device test `runtime_lane::degrade_notices::
//! every_kind_has_a_site_a_lane_test_takes` holds the closed vocabulary to the lane's sites, but
//! its module compiles only under `m2v_gpu` or `m2v_cpu_icd`, so a kind that lands with no site
//! leaves that test red where nobody reads it until someone runs the device lane by hand. The two
//! halves of that test are not alike, though: a site's kinds are a `const [Degradation; N]` in its
//! lane module, and the list of sites is a `const` array in the device test's own body, so **the
//! site set is static source data** and only the announcement it stands for needs hardware.
//!
//! So the check splits. This test reads the sites off the lane's sources and the declared kinds off
//! [`Degradation::ALL`], and fails in the ordinary gate, on any machine, when a declared kind is
//! named by no site or a site names a kind the runtime no longer declares. The device test keeps
//! its own half: that each site's work really announces its kinds on a device. Neither replaces the
//! other, and no kind is ever excused from either.
//!
//! A precision note's kind, and `ImplementationDefinedReconvergence`, are taken by a stage's
//! recorded note rather than by a site const, exactly as the device test takes them, so they are
//! held to being declared and to being absent from every site's const.
//!
//! Like `degradation_catalogue` (and as `tests/mod.rs` says), this reads the repository beside this
//! crate, so it runs from the workspace checkout and not from a packaged crate.

use crate::Degradation;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// The lane's module tree: `runtime_lane.rs` and every `.rs` file below `runtime_lane/`, under
/// the validation crate.
const LANE: &str = "src/runtime_lane";

/// The device test whose site list this test reads, and the `let` binding that holds it.
const DEVICE_TEST: &str = "fn every_kind_has_a_site_a_lane_test_takes()";

/// A site const's declaration: where it is and the kinds it names, in order.
#[derive(Debug)]
struct Site {
    /// The lane file declaring it, relative to the lane's directory.
    file: String,
    /// The length its type states.
    length: usize,
    /// The `Degradation::` variant names its initializer lists.
    kinds: Vec<String>,
}

/// Each lane source as (path relative to the lane's directory, text): `runtime_lane.rs` beside the
/// directory, then every `.rs` file below it, in sorted order.
fn lane_sources() -> Vec<(String, String)> {
    let lane = metal2vulkan_devtools::validation_dir().join(LANE);
    let module = lane.with_extension("rs");
    let mut files = vec![module];
    collect(&lane, &mut files);
    files
        .iter()
        .map(|path| {
            let name = match path.strip_prefix(&lane) {
                Ok(relative) => relative.to_string_lossy().into_owned(),
                Err(_) => "runtime_lane.rs".to_owned(),
            };
            let text = std::fs::read_to_string(path)
                .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
            (name, text)
        })
        .collect()
}

/// Every `.rs` file below `directory`, pushed in sorted order, directories walked in place.
fn collect(directory: &Path, output: &mut Vec<PathBuf>) {
    let mut entries = std::fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("read {}: {error}", directory.display()))
        .map(|entry| entry.expect("a lane directory entry").path())
        .collect::<Vec<_>>();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            collect(&path, output);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            output.push(path);
        }
    }
}

/// `text` with every `//` comment to the end of its line dropped, so a doc link naming a kind is no
/// site and a commented-out entry is no entry.
fn uncommented(text: &str) -> String {
    text.lines()
        .map(|line| match line.find("//") {
            Some(start) => &line[..start],
            None => line,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The identifier starting at `text`'s front (letters, digits and underscores), and what follows.
fn identifier(text: &str) -> (&str, &str) {
    let end = text
        .find(|c: char| !c.is_alphanumeric() && c != '_')
        .unwrap_or(text.len());
    text.split_at(end)
}

/// Every name `text` reaches as `Degradation::<name>`, in order.
fn named_kinds(text: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("Degradation::") {
        let (name, after) = identifier(&rest[start + "Degradation::".len()..]);
        if !name.is_empty() {
            names.push(name.to_owned());
        }
        rest = after;
    }
    names
}

/// The slice of `text` from `open`'s bracket to its match, brackets balanced, and what follows.
fn bracketed(text: &str) -> Option<(&str, &str)> {
    let mut depth = 0usize;
    for (at, c) in text.char_indices() {
        match c {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    return Some((&text[1..at], &text[at + 1..]));
                }
            }
            _ => {}
        }
    }
    None
}

/// Every `const <NAME>: [Degradation; <N>] = [...];` the lane declares, by name. A name declared
/// twice panics: the device test's list would name one of them and this test the other.
fn sites(sources: &[(String, String)]) -> BTreeMap<String, Site> {
    let mut sites: BTreeMap<String, Site> = BTreeMap::new();
    for (file, text) in sources {
        let text = uncommented(text);
        let mut rest = text.as_str();
        while let Some(start) = rest.find("const ") {
            let (name, after) = identifier(&rest[start + "const ".len()..]);
            rest = after;
            let Some(typed) = after.strip_prefix(": [Degradation; ") else {
                continue;
            };
            let (length, after) = identifier(typed);
            let Some(initializer) = after.strip_prefix("] =") else {
                continue;
            };
            let initializer = initializer.trim_start();
            let Some((listed, after)) = bracketed(initializer) else {
                continue;
            };
            if !after.starts_with(';') {
                continue;
            }
            rest = after;
            let site = Site {
                file: file.clone(),
                length: length.parse().expect("a site's length is a number"),
                kinds: named_kinds(listed),
            };
            if let Some(earlier) = sites.insert(name.to_owned(), site) {
                panic!(
                    "{name} is declared in {} and in {file}, so a site's name names two sites",
                    earlier.file
                );
            }
        }
    }
    sites
}

/// The site names the device test's `lists` array holds, in order: each entry is `&<NAME>` or
/// `&<NAME>[..]`.
fn listed_names(degrade_notices: &str) -> Vec<String> {
    let test = degrade_notices
        .find(DEVICE_TEST)
        .unwrap_or_else(|| panic!("`{DEVICE_TEST}` is the device test this test reads"));
    let body = &degrade_notices[test..];
    let open = body
        .find("let lists = [")
        .expect("the device test binds its sites as `let lists = [`");
    let (listed, _) = bracketed(&body[open + "let lists = ".len()..])
        .expect("the device test's site list closes its bracket");
    let names: Vec<String> = uncommented(listed)
        .lines()
        .filter_map(|line| {
            let entry = line.trim().strip_prefix('&')?;
            let (name, _) = identifier(entry);
            (!name.is_empty()).then(|| name.to_owned())
        })
        .collect();
    assert!(
        !names.is_empty(),
        "the device test's site list parsed empty, so this test would read no site at all"
    );
    names
}

/// Every kind the lane's sites name is declared, every declared kind a stage does not record is
/// named by a site, and no site names a kind another site names.
///
/// The failure to expect: a `Degradation` variant lands with no lane site, and this names it.
#[test]
fn every_declared_kind_is_named_by_a_lane_site() {
    let sources = lane_sources();
    let sites = sites(&sources);
    let (_, degrade_notices) = sources
        .iter()
        .find(|(file, _)| file == "degrade_notices.rs")
        .expect("the lane declares its degrade notices");

    // Each site the device test lists is declared, and plainly lists the kinds its type counts: a
    // site that reaches another site's element (`STEPPED`, which the list does not hold) would read
    // short here, so it fails rather than losing a kind.
    let mut listed: BTreeMap<String, String> = BTreeMap::new();
    let mut unresolved = Vec::new();
    let mut indirect = Vec::new();
    let mut twice = Vec::new();
    for name in listed_names(degrade_notices) {
        let Some(site) = sites.get(&name) else {
            unresolved.push(name);
            continue;
        };
        if site.kinds.len() != site.length {
            indirect.push(format!(
                "{name} ({}) counts {} kinds and names {:?}",
                site.file, site.length, site.kinds
            ));
            continue;
        }
        for kind in &site.kinds {
            if let Some(earlier) = listed.insert(kind.clone(), name.clone()) {
                twice.push(format!("{kind} is named by {earlier} and by {name}"));
            }
        }
    }
    assert!(
        unresolved.is_empty(),
        "the device test lists sites no lane file declares: {unresolved:?}"
    );
    assert!(
        indirect.is_empty(),
        "a listed site does not plainly name its kinds, so a kind of it would go unread: {indirect:?}"
    );
    assert!(twice.is_empty(), "sites name a kind twice: {twice:?}");

    // The declared vocabulary, by the name a site writes. A kind a stage records its note for has
    // no site: the device test takes those from `Degradation::ALL` itself.
    let declared: BTreeMap<String, Degradation> = Degradation::ALL
        .iter()
        .map(|&kind| (format!("{kind:?}"), kind))
        .collect();
    assert_eq!(
        declared.len(),
        Degradation::ALL.len(),
        "two kinds debug alike, so a site's name cannot pick one"
    );
    let staged = |kind: Degradation| {
        kind.precision_note().is_some() || kind == Degradation::ImplementationDefinedReconvergence
    };

    let unknown: Vec<_> = listed
        .keys()
        .filter(|k| !declared.contains_key(*k))
        .collect();
    assert!(
        unknown.is_empty(),
        "lane sites name kinds the runtime does not declare: {unknown:?} \
         (a renamed or removed variant leaves its site behind)"
    );
    let staged_with_a_site: Vec<_> = listed
        .keys()
        .filter(|k| declared.get(*k).copied().is_some_and(staged))
        .collect();
    assert!(
        staged_with_a_site.is_empty(),
        "a stage's recorded note is the site of these kinds, so no site const may name them: \
         {staged_with_a_site:?}"
    );
    let siteless: BTreeSet<&str> = declared
        .iter()
        .filter(|(name, &kind)| !staged(kind) && !listed.contains_key(*name))
        .map(|(name, _)| name.as_str())
        .collect();
    assert!(
        siteless.is_empty(),
        "these declared kinds are named by no lane site, so the device test \
         `every_kind_has_a_site_a_lane_test_takes` is red: {siteless:?}. \
         Give each a site: a `const [Degradation; N]` beside the lane test that takes it, listed \
         in that test's `lists`."
    );
}
