//! A binary of this crate that sets `test = false` in `Cargo.toml` holds no unit test: Cargo builds
//! no test harness of such a binary, so a `#[test]` in its sources would silently never run.

use std::path::{Path, PathBuf};

/// The value of `key` in one manifest table, unquoted.
fn value(table: &str, key: &str) -> Option<String> {
    table.lines().find_map(|line| {
        let rest = line
            .trim()
            .strip_prefix(key)?
            .trim_start()
            .strip_prefix('=')?;
        Some(rest.split('#').next()?.trim().trim_matches('"').to_owned())
    })
}

/// The files of the modules `file` declares with `mod name;`, resolved as rustc does for a module
/// file: beside a crate root or `mod.rs`, in a directory named after any other file, or at a
/// `#[path]` relative to the declaring file's directory.
fn declared_modules(file: &Path, text: &str, is_root: bool) -> Vec<PathBuf> {
    let dir = file.parent().expect("a source file's directory");
    let own_dir = if is_root || file.file_name().is_some_and(|name| name == "mod.rs") {
        dir.to_path_buf()
    } else {
        dir.join(file.file_stem().expect("a source file's stem"))
    };
    let mut path_attr = None;
    let mut modules = Vec::new();
    for line in text.lines().map(str::trim) {
        if let Some(rest) = line.strip_prefix("#[path") {
            let quoted = rest.split('"').nth(1).expect("a quoted #[path]");
            path_attr = Some(dir.join(quoted));
            continue;
        }
        let declaration = line
            .strip_prefix("pub ")
            .or_else(|| line.strip_prefix("pub(crate) "))
            .unwrap_or(line);
        let Some(name) = declaration
            .strip_prefix("mod ")
            .and_then(|rest| rest.strip_suffix(';'))
        else {
            if !line.starts_with("#[") {
                path_attr = None;
            }
            continue;
        };
        let name = name.trim();
        modules.push(path_attr.take().unwrap_or_else(|| {
            let flat = own_dir.join(format!("{name}.rs"));
            if flat.is_file() {
                flat
            } else {
                own_dir.join(name).join("mod.rs")
            }
        }));
    }
    modules
}

#[test]
fn a_binary_without_a_test_harness_holds_no_test() {
    let root = metal2vulkan_devtools::validation_dir();
    let manifest = std::fs::read_to_string(root.join("Cargo.toml")).unwrap();
    let (mut binaries, mut offenders) = (0, Vec::new());
    for table in manifest.split("\n[[bin]]\n").skip(1) {
        let table = table.split("\n[").next().unwrap_or_default();
        if value(table, "test").as_deref() != Some("false") {
            continue;
        }
        binaries += 1;
        let crate_root = root.join(value(table, "path").expect("a [[bin]] path"));
        let mut files = vec![(crate_root, true)];
        while let Some((file, is_root)) = files.pop() {
            let text = std::fs::read_to_string(&file)
                .unwrap_or_else(|error| panic!("read {}: {error}", file.display()));
            if text.contains("#[test]") {
                offenders.push(file.clone());
            }
            files.extend(
                declared_modules(&file, &text, is_root)
                    .into_iter()
                    .map(|module| (module, false)),
            );
        }
    }
    assert!(binaries > 0, "no [[bin]] sets test = false");
    assert!(
        offenders.is_empty(),
        "a binary with `test = false` holds a #[test] that never runs; drop its `test = false`: {offenders:?}"
    );
}
