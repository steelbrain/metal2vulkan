//! The feature vocabulary against the member tables that must cover it.
//!
//! [`VkFeature`] is the vocabulary; a `VkFeature` by itself enables nothing. Two tables in two
//! crates carry the structure member each feature sets or is read from, and the compiler holds
//! neither to the vocabulary:
//!
//! - `crates/metal2vulkan/src/device/khronos.rs`'s `FEATURE_MEMBERS`: the member name a captured
//!   Khronos profile states the feature by. A feature with no row there is read from a device but
//!   never from its committed profile, which is what stopped the RADV venue over `taskShader` and
//!   `meshShader`.
//! - `crates/metal2vulkan-validation/src/candidate/platform/enables.rs`'s `MEMBERS`: the member a
//!   validation context enables and reads back. A feature mapped to `None` there (which reads as
//!   the deliberate "no `VkFeature` of ours corresponds") rather than to its own name is enabled
//!   by no context and reported as enabled by none.
//!
//! This test reads the two tables from the sources, as `refusal_surface` reads the runtime's
//! refusals, because each is private to its crate: a feature half-added to the vocabulary fails
//! here, in the first crate's own suite, instead of in a later crate's run or on a device.
//!
//! # What it cannot see
//!
//! - Whether a row names the **right** member. It holds each table to having one row per feature,
//!   never to that row's structure or field being the registry's.
//! - The validation table's rows for members the vocabulary does not name (its `None` rows): they
//!   are not features, and that crate's own `every_feature_is_enabled_or_excluded_by_name` and
//!   `a_member_the_vocabulary_does_not_name_carries_its_registry_name` hold them.

use metal2vulkan::device::VkFeature;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// The repository root.
fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The text of the source file `relative` to the repository root.
fn source(relative: &str) -> String {
    let path = root().join(relative);
    std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

/// The region of `text` between the first line that is exactly `opens` and the next line that is
/// exactly `close`: one table's rows, and nothing around them.
fn region<'a>(path: &str, text: &'a str, opens: &str, close: &str) -> &'a str {
    let open = format!("\n{opens}\n");
    let start = text.find(&open).unwrap_or_else(|| {
        panic!("{path}: no line {opens:?}, so no table was read: it was renamed or moved")
    }) + open.len();
    let shut = format!("\n{close}\n");
    let end = text[start..]
        .find(&shut)
        .unwrap_or_else(|| panic!("{path}: the table at {opens:?} has no closing line {close:?}"))
        + start;
    &text[start..end]
}

/// The identifier `rest` begins with.
fn identifier(rest: &str) -> String {
    rest.chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect()
}

/// Every feature `region` names as `F::<feature>`: the Khronos reader's rows
/// (`("shaderInt64", F::ShaderInt64)`) and the validation context's (`=> Some(F::ShaderInt64)`).
/// A row that names none (that table's `None`, a member the vocabulary does not name) adds
/// nothing.
fn qualified_features(region: &str) -> Vec<String> {
    region
        .match_indices("F::")
        .map(|(at, prefix)| identifier(&region[at + prefix.len()..]))
        .filter(|name| !name.is_empty())
        .collect()
}

/// `table` in `path` covers exactly `expected`, each feature once.
fn covers(path: &str, table: &str, found: &[String], expected: &BTreeSet<String>) {
    let mut once = BTreeSet::new();
    for name in found {
        assert!(
            once.insert(name.clone()),
            "{path}: {table} has two rows for {name}"
        );
    }
    let missing: Vec<&String> = expected.difference(&once).collect();
    assert!(
        missing.is_empty(),
        "{path}: {table} has no member for {missing:?}. Every feature of the vocabulary needs a \
         row in each of the four member tables (this file's module comment lists them and what \
         each missing row breaks); a feature `VkFeature::ALL` names and a member table does not \
         is a half-added feature."
    );
    let stale: Vec<&String> = once.difference(expected).collect();
    assert!(
        stale.is_empty(),
        "{path}: {table} has a member for {stale:?}, which the vocabulary does not name (or no \
         longer names). Either the feature was removed from `VkFeature` and this row with it, or \
         the row's name is misspelled, in which case the feature it meant has no member."
    );
}

/// Every feature of the vocabulary, by the name its variant carries.
fn vocabulary() -> BTreeSet<String> {
    VkFeature::ALL
        .iter()
        .map(|feature| format!("{feature:?}"))
        .collect()
}

/// The validation context's `MEMBERS` names one structure member per vocabulary feature, and the
/// Khronos reader names the member a profile states it by: a feature no table covers fails here
/// rather than in the crate that holds the table.
#[test]
fn every_member_table_covers_the_vocabulary() {
    let expected = vocabulary();
    assert!(expected.len() > 150, "{} features read", expected.len());

    let khronos = "crates/metal2vulkan/src/device/khronos.rs";
    let text = source(khronos);
    let rows = region(
        khronos,
        &text,
        "pub(super) const FEATURE_MEMBERS: &[(&str, VkFeature)] = {",
        "};",
    );
    covers(
        khronos,
        "FEATURE_MEMBERS",
        &qualified_features(rows),
        &expected,
    );

    let validation = "crates/metal2vulkan-validation/src/candidate/platform/enables.rs";
    let text = source(validation);
    let rows = region(validation, &text, "members! {", "}");
    covers(validation, "MEMBERS", &qualified_features(rows), &expected);
}
