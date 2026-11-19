//! No script that sets `pipefail` pipes into a quiet `grep`. `grep -q` exits at its first match,
//! which can end the command writing into the pipe with SIGPIPE before it has written everything;
//! under `pipefail` the pipeline's status is then the writer's, and it reads false on a match.
//! Whether that happens depends on scheduling, so a check written that way passes alone and fails
//! on a loaded host: `scripts/linux-gate-policy` once accepted a scratch directory inside the
//! checkout that way. Such a test is matched in the shell instead (`[[ ... == *...* ]]`).

use std::fs;
use std::path::{Path, PathBuf};

/// Every file under `dir`, depth first, skipping build and cache directories.
fn walk(dir: &Path, found: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if entry.file_type().is_ok_and(|kind| kind.is_dir()) {
            if !matches!(entry.file_name().to_str(), Some("target" | "__pycache__")) {
                walk(&path, found);
            }
        } else {
            found.push(path);
        }
    }
}

/// Whether the command `segment` (the text after a pipe) is a `grep` with a quiet option.
fn quiet_grep(segment: &str) -> bool {
    let mut words = segment.split_whitespace();
    if !matches!(words.next(), Some("grep" | "egrep" | "fgrep")) {
        return false;
    }
    words.take_while(|word| word.starts_with('-')).any(|word| {
        matches!(word, "--quiet" | "--silent")
            || (!word.starts_with("--") && word[1..].contains('q'))
    })
}

/// Whether `line` pipes into a quiet `grep`: a single `|` (not `||`) followed by one.
fn pipes_into_quiet_grep(line: &str) -> bool {
    if line.trim_start().starts_with('#') {
        return false;
    }
    let bytes = line.as_bytes();
    (0..bytes.len()).any(|at| {
        bytes[at] == b'|'
            && (at == 0 || bytes[at - 1] != b'|')
            && bytes.get(at + 1) != Some(&b'|')
            && quiet_grep(
                line[at + 1..]
                    .split([';', '&', ')', '|'])
                    .next()
                    .unwrap_or(""),
            )
    })
}

#[test]
fn the_detector_finds_a_piped_quiet_grep_and_nothing_else() {
    for hit in [
        "printf '%s\\n' \"$up\" | grep -qxF \"$id\"",
        "if tar --version 2>/dev/null | grep -q bsdtar; then",
        "x | grep -E -q y",
        "x |grep --quiet y",
    ] {
        assert!(pipes_into_quiet_grep(hit), "{hit}");
    }
    for miss in [
        "x || grep -q y",
        "grep -q y file",
        "x | grep -c y",
        "# x | grep -q y",
        "[[ $up == *\"$id\"* ]]",
    ] {
        assert!(!pipes_into_quiet_grep(miss), "{miss}");
    }
}

#[test]
fn no_script_that_sets_pipefail_pipes_into_a_quiet_grep() {
    let root =
        fs::canonicalize(metal2vulkan_devtools::workspace_dir()).expect("the repository's root");
    let mut files = Vec::new();
    walk(&root.join("scripts"), &mut files);
    let scripts: Vec<(PathBuf, String)> = (files.into_iter())
        .filter_map(|path| fs::read_to_string(&path).ok().map(|text| (path, text)))
        .filter(|(_, text)| text.contains("pipefail"))
        .collect();
    assert!(
        scripts.len() >= 10,
        "only {} scripts set pipefail, so the walk is looking in the wrong place",
        scripts.len()
    );
    let hits: Vec<String> = (scripts.iter())
        .flat_map(|(path, text)| {
            (text.lines().enumerate())
                .filter(|(_, line)| pipes_into_quiet_grep(line))
                .map(move |(at, line)| format!("{}:{}: {}", path.display(), at + 1, line.trim()))
        })
        .collect();
    assert!(
        hits.is_empty(),
        "a quiet grep after a pipe under pipefail can read false on a match:\n{}",
        hits.join("\n")
    );
}
