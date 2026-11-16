//! Every committed case over a public fixture passes the current case checker.
//!
//! A case is installed under the checker of its day and then left alone. `corpus-case-check
//! --recheck-all` re-reads the whole store, but it needs the private sources, so no gate runs it:
//! two public cases (`c570d15a…` and `b77f3324…`) stayed installed after the checker began
//! decoding their AIR through `llvm-as` and the product's metadata ABI, which their hand-written
//! fixtures did not satisfy (a nested `air.struct_type_info` spelled as a bare trailing node, and
//! SSA values numbered out of order). A public case needs nothing private -- its AIR is a
//! committed fixture -- so this test re-checks every one of them, and a fixture or checker change
//! that leaves a committed public case behind fails here, in CI, instead.
//!
//! A case counts as public when its `air_sha256` is a public fixture module's (the module of one
//! source row, or of a row per entry when it declares several), or when it names a public fixture
//! in `public_source`.
//!
//! A fixture edited without its case being re-keyed leaves the case's `air_sha256` naming no
//! public module, which by hash alone looks like a case over private AIR. So every public case
//! names its fixture (`public_source`, outside the identity), and
//! [`every_public_case_names_its_fixture_and_the_fixture_is_its_air`] holds the two together in
//! both directions.

use std::collections::BTreeSet;

use metal2vulkan_validation::case::AuthoredCase;
use metal2vulkan_validation::check::check_case;
use metal2vulkan_validation::source::{check_public_source, public_source_set};
use metal2vulkan_validation::store::CorpusStore;

fn committed_cases() -> Vec<AuthoredCase> {
    let root = metal2vulkan_devtools::validation_dir().join("corpus");
    CorpusStore::new(&root)
        .read_all_cases()
        .expect("committed cases")
}

/// Both directions over the committed cases: a case naming a public fixture names one that exists
/// and whose text hashes to the case's `air_sha256` (so an edited fixture whose case was not
/// re-keyed fails here, named by its path), and a case whose `air_sha256` is a public module's
/// names exactly that module.
#[test]
fn every_public_case_names_its_fixture_and_the_fixture_is_its_air() {
    let public = public_source_set().expect("public fixtures");
    let cases = committed_cases();
    let mut errors = Vec::new();
    let mut named = 0usize;
    for case in &cases {
        if let Some(path) = &case.public_source {
            named += 1;
            if let Err(error) = check_public_source(path, &case.air_sha256) {
                errors.push(format!("{} {}: {error}", case.case_id, case.name));
            }
        }
        if let Some(path) = public.path_of(&case.air_sha256) {
            if case.public_source.as_deref() != Some(path.as_str()) {
                errors.push(format!(
                    "{} {}: its AIR is the public fixture {path}, but public_source is {:?}; \
                     reinstall it with corpus-case-check --install",
                    case.case_id, case.name, case.public_source
                ));
            }
        }
    }
    assert!(
        named > 100,
        "only {named} committed cases name a public fixture"
    );
    errors.sort();
    assert!(
        errors.is_empty(),
        "{} committed case(s) and their public fixtures disagree:\n{}",
        errors.len(),
        errors.join("\n")
    );
}

#[test]
fn every_committed_public_case_passes_the_checker() {
    let root = metal2vulkan_devtools::validation_dir().join("corpus");
    let public = public_source_set().expect("public fixtures");
    let modules = public
        .rows
        .iter()
        .map(|row| row.air_sha256.as_str())
        .collect::<BTreeSet<_>>();
    let cases = committed_cases()
        .into_iter()
        .filter(|case| modules.contains(case.air_sha256.as_str()) || case.public_source.is_some())
        .collect::<Vec<_>>();
    // The committed store holds well over a hundred public cases; a filter that selects none
    // would pass vacuously.
    assert!(
        cases.len() > 100,
        "only {} committed public cases",
        cases.len()
    );
    let workers = std::thread::available_parallelism().map_or(1, usize::from);
    let chunk = cases.len().div_ceil(workers);
    let mut stale = std::thread::scope(|scope| {
        let handles = cases
            .chunks(chunk)
            .map(|cases| {
                let root = &root;
                scope.spawn(move || {
                    cases
                        .iter()
                        .filter_map(|case| {
                            let errors = check_case(root, case.clone()).err()?;
                            Some(format!(
                                "{} {}: {}",
                                case.case_id,
                                case.name,
                                errors.join("; ")
                            ))
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("checker thread"))
            .collect::<Vec<_>>()
    });
    stale.sort();
    assert!(
        stale.is_empty(),
        "{} of {} committed public cases no longer check against their AIR; repair the fixture \
         and reinstall the case under its new AIR, or fix the checker:\n{}",
        stale.len(),
        cases.len(),
        stale.join("\n")
    );
}
