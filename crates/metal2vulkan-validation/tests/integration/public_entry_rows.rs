//! Every entry point of every public fixture module is a source row: the index refresh reports the
//! rows and the modules they come from, and the authoring-capability census classifies each entry
//! row like any other, with no module outside it.

use metal2vulkan_validation::{source, ScratchDir};
use std::collections::BTreeSet;
use std::process::{Command, Output};

fn run(binary: &std::path::Path, args: &[&str], corpus: &std::path::Path) -> (Output, String) {
    let output = Command::new(binary)
        .arg("--corpus")
        .arg(corpus)
        .args(args)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    (output, stdout)
}

fn public_source_lines(stdout: &str) -> Vec<&str> {
    stdout
        .lines()
        .filter(|line| line.starts_with("public-source"))
        .collect()
}

#[test]
fn corpus_index_and_the_census_cover_every_entry_of_every_public_module() {
    let sources = source::public_source_set().unwrap();
    let modules = sources
        .rows
        .iter()
        .map(|row| row.air_sha256.as_str())
        .collect::<BTreeSet<_>>();
    let keys = sources
        .rows
        .iter()
        .map(|row| row.key().into_owned())
        .collect::<BTreeSet<_>>();
    // The committed engine/ fixtures include multi-entry modules: more rows than modules, each
    // under its own key.
    let rows = sources.rows.len();
    assert!(
        rows > modules.len(),
        "{rows} rows of {} modules",
        modules.len()
    );
    assert_eq!(keys.len(), rows);
    let report = sources.report_lines();
    assert_eq!(
        report,
        [format!(
            "public-sources\trows={rows}\tmodules={}",
            modules.len()
        )]
    );

    // A corpus with no private sources: every indexed row is a public one.
    let scratch = ScratchDir::new("public-entry-rows").unwrap();
    let (output, stdout) = run(&bin_exe!("corpus-index"), &[], scratch.path());
    assert!(
        output.status.success(),
        "{stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(public_source_lines(&stdout), report, "{stdout}");

    // One batch selects every public row, whatever the default limit.
    let limit = rows.to_string();
    let (output, stdout) = run(
        &bin_exe!("corpus-triage"),
        &[
            "--audit",
            "authoring-capabilities",
            "--summary-only",
            "--limit",
            &limit,
            "--jobs",
            "2",
        ],
        scratch.path(),
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    let summary = stdout
        .lines()
        .find(|line| line.starts_with("authoring-capability-summary\t"))
        .unwrap_or_else(|| panic!("{stdout}{stderr}"));
    assert!(
        summary.contains(&format!(
            "\ttotal={rows}\tclassified={rows}\tremaining=0\tunresolved="
        )),
        "{summary}"
    );
    assert!(!summary.contains("excluded"), "{summary}");
    assert!(!stderr.contains("outside the census"), "{stderr}");
    assert_eq!(public_source_lines(&stdout), report, "{stdout}");
    // The census is closed exactly when no row is unresolved, and otherwise fails naming only
    // the unresolved rows.
    let unresolved = summary
        .split('\t')
        .find_map(|field| field.strip_prefix("unresolved="))
        .unwrap()
        .parse::<usize>()
        .unwrap();
    assert_eq!(output.status.success(), unresolved == 0, "{stdout}{stderr}");
    if unresolved != 0 {
        assert!(
            stderr.contains(&format!(
                "{unresolved} source(s) still have validation support gaps"
            )),
            "{stderr}"
        );
    }
}
