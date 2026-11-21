//! The product's source-fingerprint inputs, for validation's caches.
//!
//! [`product_hash`] is the product's source fingerprint: the validation crate's product fingerprint
//! (its `build.rs`, the dependency of its observations, census keys, and analyzer key) is this
//! hash. It covers exactly the files that can be compiled into the product or read by it, found by
//! following the crate roots' module tree the way rustc resolves it, so an edit to a test-only
//! module, a test fixture, or a file no module declares leaves it unchanged. It covers the AIR
//! front end's crate (`metal2vulkan-air`) the same way while the product names it as a path
//! dependency (in a published build the hashed manifest's exact pin names its sources).
//! [`product_inputs`] is what the build script that takes it watches.
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Component, Path, PathBuf};

/// Names what [`product_hash`] hashes, so a fingerprint taken under the earlier definition (the
/// manifest, the build inputs, and every file under `src/`) never equals one taken under this one,
/// even for a tree where the two file sets agree.
const PRODUCT_DEFINITION: &[u8] =
    b"metal2vulkan product sources: the crate roots' module trees, the AIR front end's included, \
      each crate's build script where one lies beside its manifest\0";

/// The product crate's source fingerprint: its manifest, its `build.rs` (if any), and
/// [`product_files`], the same of the AIR front end's crate where the manifest names it by path,
/// and the workspace's manifest where the product's inherits from it, each by its path relative to
/// `root`, length, and bytes, after a tag naming this definition. It asks Cargo to rerun the build
/// script when one of [`product_inputs`] changes.
pub fn product_hash(root: &Path) -> Sha256 {
    let inputs = product_inputs(root);
    for directory in &inputs.watched {
        println!("cargo:rerun-if-changed={}", directory.display());
    }
    let mut hash = Sha256::new();
    hash.update(PRODUCT_DEFINITION);
    hash_files(&mut hash, root, &inputs.files);
    hash
}

/// What [`product_hash`] reads, and what its build script watches.
pub struct ProductInputs {
    /// The files it hashes, sorted: the manifest, `build.rs` (if any), and [`product_files`];
    /// the AIR front end's manifest, `build.rs` (if any), and [`product_files`] where the
    /// product's manifest names that crate by path; and the workspace's manifest where the
    /// product's inherits from it ([`workspace_manifest`]).
    /// [`hash_files`] asks Cargo to rerun the script when one of them changes or goes missing.
    /// That covers every way the set itself can change but a discovered crate root: a moved or
    /// deleted module leaves a hashed file missing, a new module, `#[path]`, include, or declared
    /// crate root edits a hashed file, and rustc refuses a module with two candidate files, so a
    /// new file cannot silently change the one a declaration resolves to.
    pub files: Vec<PathBuf>,
    /// The directories it watches beside them, recursively: `src/` (the product's, and the front
    /// end's) when a crate root can appear
    /// there with no hashed file changing, because the manifest lets Cargo discover binaries
    /// (`autobins`, on by default) or names no library path while `src/lib.rs` does not exist.
    /// Empty when every crate root is declared, so an edit to a test-only module (`src/tests/`)
    /// reruns no build script. A watched path must exist: Cargo reruns a build script on every
    /// build while one is missing, so the script cannot watch where `src/bin/` would be.
    pub watched: Vec<PathBuf>,
}

/// [`ProductInputs`] of the product below `root`.
pub fn product_inputs(root: &Path) -> ProductInputs {
    let mut files = vec![root.join("Cargo.toml")];
    // The product's manifest turns Cargo's build-script discovery off (`build = false`), as the
    // front end's does, so a build script it declares is a manifest edit; one beside it is
    // hashed whether it runs or not.
    let build = root.join("build.rs");
    if build.is_file() {
        files.push(build);
    }
    files.extend(workspace_manifest(root));
    files.extend(product_files(root));
    let mut watched = if crate_roots(root).discoverable {
        vec![root.join("src")]
    } else {
        Vec::new()
    };
    if let Some(front_end) = front_end_dir(root) {
        files.push(front_end.join("Cargo.toml"));
        // Its manifest turns Cargo's build-script discovery off (`build = false`), so a build
        // script it declares is a manifest edit; one beside it is hashed whether it runs or not.
        let build = front_end.join("build.rs");
        if build.is_file() {
            files.push(build);
        }
        files.extend(product_files(&front_end));
        if crate_roots(&front_end).discoverable {
            watched.push(front_end.join("src"));
        }
    }
    files.sort();
    files.dedup();
    ProductInputs { files, watched }
}

/// The workspace's manifest, when the product's manifest inherits from it (a `workspace = true`
/// key: in this repository the crates take their edition, license and shared dependency versions
/// from the root `Cargo.toml`, which also holds the build profiles), named through `root`
/// (`root/../../Cargo.toml`) so that every build script hashes it by the same relative path; `None`
/// when the manifest inherits nothing, as a published build's normalized manifest does. A manifest
/// that inherits from no workspace above `root` fails the build rather than leave it out.
fn workspace_manifest(root: &Path) -> Option<PathBuf> {
    let manifest = fs::read_to_string(root.join("Cargo.toml"))
        .unwrap_or_else(|error| panic!("read {}: {error}", root.join("Cargo.toml").display()));
    let inherits = manifest.lines().any(|line| {
        let key = line.split('#').next().unwrap_or_default();
        let key = key.split_whitespace().collect::<String>();
        [".workspace=true", "{workspace=true", ",workspace=true"]
            .iter()
            .any(|form| key.contains(form))
    });
    if !inherits {
        return None;
    }
    let absolute = fs::canonicalize(root)
        .unwrap_or_else(|error| panic!("resolve {}: {error}", root.display()));
    let mut relative = root.to_path_buf();
    for ancestor in absolute.ancestors().skip(1) {
        relative.push("..");
        let is_workspace = fs::read_to_string(ancestor.join("Cargo.toml")).is_ok_and(|text| {
            text.lines().any(|line| {
                line.trim()
                    .strip_prefix("[workspace]")
                    .is_some_and(|rest| rest.trim().is_empty() || rest.trim().starts_with('#'))
            })
        });
        if is_workspace {
            return Some(relative.join("Cargo.toml"));
        }
    }
    panic!(
        "{} inherits from a workspace, but no workspace manifest lies above it",
        root.join("Cargo.toml").display()
    )
}

/// The name of the AIR front end's crate in the product's manifest.
const FRONT_END: &str = "metal2vulkan-air";

/// The directory of the AIR front end's crate, when the product's manifest names it by `path`
/// (this repository); `None` when it names it by version alone (a published build, whose
/// normalized manifest drops the path and whose exact pin `=0.x.y` names an immutable release).
/// It reads the dependency inline (`metal2vulkan-air = { ... }` under `[dependencies]`) or as its
/// own `[dependencies.metal2vulkan-air]` table; a path it cannot read fails the build rather than
/// leave the front end's sources out.
fn front_end_dir(root: &Path) -> Option<PathBuf> {
    let manifest = fs::read_to_string(root.join("Cargo.toml"))
        .unwrap_or_else(|error| panic!("read {}: {error}", root.join("Cargo.toml").display()));
    let own_table = format!("[dependencies.{FRONT_END}]");
    let mut table = String::new();
    let mut path = None;
    for line in manifest.lines().map(str::trim) {
        if line.starts_with('[') {
            table = table_header(line);
            continue;
        }
        let value = if table == "[dependencies]" {
            match line
                .strip_prefix(FRONT_END)
                .and_then(|rest| rest.trim_start().strip_prefix('='))
            {
                Some(value) => value,
                None => continue,
            }
        } else if table == own_table {
            line
        } else {
            continue;
        };
        let Some((key, after)) = path_key(value) else {
            continue;
        };
        let quoted = after
            .trim_start()
            .strip_prefix('=')
            .map(str::trim_start)
            .and_then(|rest| rest.strip_prefix('"'))
            .and_then(|rest| rest.split_once('"'))
            .map(|(path, _)| path)
            .unwrap_or_else(|| panic!("Cargo.toml: a {FRONT_END} {key} the walk cannot read"));
        assert!(path.is_none(), "Cargo.toml: {FRONT_END} names two paths");
        path = Some(root.join(quoted));
    }
    let dir = path?;
    assert!(
        dir.join("Cargo.toml").is_file(),
        "{FRONT_END} at {} has no manifest",
        dir.display()
    );
    Some(dir)
}

/// The `path` key in a dependency's value or table line and the text after it, if the line has
/// one: `path` at the line's start (its own table) or after `{` or `,` (an inline table).
fn path_key(value: &str) -> Option<(&'static str, &str)> {
    let mut rest = value;
    loop {
        let trimmed = rest.trim_start();
        if let Some(after) = trimmed.strip_prefix("path") {
            if after.trim_start().starts_with('=') {
                return Some(("path", after));
            }
        }
        let next = rest.find(['{', ','])?;
        rest = &rest[next + 1..];
    }
}

/// Every file below `root` the product can compile or read, sorted: each crate root of the library
/// and the binaries (the manifest's `[lib]` and `[[bin]]` paths and the roots Cargo discovers), every
/// module file their `mod` declarations reach as rustc resolves them (`#[path]` included), and every
/// file an `include!`, `include_str!`, or `include_bytes!` of a reached item names.
///
/// A module or item is left out only when its `cfg` cannot hold without `test`: `#[cfg(test)]`,
/// `#[cfg(all(test, ...))]`, or a file's own `#![cfg(test)]`, with everything below it. Every other
/// predicate (a feature, a platform, `not(test)`) may hold for some build, so it is followed. A form
/// the walk cannot resolve fails the build rather than leaving an input out: a missing module file,
/// an include whose path is not a string literal, a conditional `#[path]`, an include in a
/// `macro_rules!` body (rustc resolves it against each invoking file), an include macro imported
/// under another name, a manifest table header or `autobins` value it cannot read, or a `[[bin]]`
/// without a path under `autobins = false`.
pub fn product_files(root: &Path) -> Vec<PathBuf> {
    let mut walk = Walk {
        root,
        files: BTreeSet::new(),
        modules: BTreeSet::new(),
    };
    for crate_root in crate_roots(root).roots {
        let dir = parent(&crate_root);
        walk.module_file(
            crate_root,
            ModuleDir {
                dir,
                relative: None,
            },
        );
    }
    walk.files.iter().map(|file| root.join(file)).collect()
}

/// Hash `files` (each below `root`) in order by relative path, length, and bytes, and ask Cargo to
/// rerun the build script when one changes.
pub fn hash_files(hash: &mut Sha256, root: &Path, files: &[PathBuf]) {
    for path in files {
        println!("cargo:rerun-if-changed={}", path.display());
        let relative = path.strip_prefix(root).expect("fingerprint input");
        let bytes = fs::read(path)
            .unwrap_or_else(|error| panic!("read fingerprint input {}: {error}", path.display()));
        hash.update(relative.to_string_lossy().as_bytes());
        hash.update([0]);
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
}

/// The crate roots of the library and the binaries, and whether Cargo can find one by itself.
struct CrateRoots {
    /// Each root, relative to `root`: the `path` of the manifest's `[lib]` table (`src/lib.rs` by
    /// default) and of each `[[bin]]` table, and, unless the manifest turns `autobins` off, Cargo's
    /// automatically discovered binaries (`src/main.rs`, `src/bin/*.rs`, `src/bin/*/main.rs`).
    roots: Vec<PathBuf>,
    /// A crate root can appear below `src/` with no manifest or module edit: Cargo discovers
    /// binaries, or the manifest names no library path while `src/lib.rs` does not exist.
    discoverable: bool,
}

/// The [`CrateRoots`] of the product below `root`. Under `autobins = false` each `[[bin]]` table
/// needs its `path`: the walk does not infer one from the binary's name.
fn crate_roots(root: &Path) -> CrateRoots {
    let manifest = fs::read_to_string(root.join("Cargo.toml"))
        .unwrap_or_else(|error| panic!("read {}: {error}", root.join("Cargo.toml").display()));
    let mut table = String::new();
    let mut lib = None;
    let mut autobins = true;
    let mut roots = BTreeSet::new();
    // Whether some `[[bin]]` table, and whether the current one, lacks a `path`.
    let mut pathless_bin = false;
    let mut bin_without_path = false;
    // Lines inside a multi-line array or string are values, never headers or keys.
    let mut open_brackets = 0usize;
    let mut in_string = None;
    for line in manifest.lines().map(str::trim) {
        let continued = open_brackets > 0 || in_string.is_some();
        open_brackets = value_brackets(line, open_brackets, &mut in_string);
        if continued {
            continue;
        }
        if line.starts_with('[') {
            pathless_bin |= bin_without_path;
            table = table_header(line);
            bin_without_path = table == "[[bin]]";
            open_brackets = 0;
            continue;
        }
        if table.is_empty()
            && (line.starts_with("lib") || line.starts_with("bin") || line.starts_with("package"))
        {
            let key = line.split(['=', ' ', '\t']).next().unwrap_or_default();
            if key.starts_with("lib.") || key.starts_with("bin.") || key.starts_with("package.") {
                panic!("Cargo.toml: a dotted target key the walk cannot read: {line}");
            }
        }
        if table == "[package]" {
            if let Some(value) = line
                .strip_prefix("autobins")
                .and_then(|rest| rest.trim_start().strip_prefix('='))
            {
                autobins = match value.split('#').next().unwrap_or_default().trim() {
                    "true" => true,
                    "false" => false,
                    _ => panic!("Cargo.toml: an autobins value the walk cannot read: {line}"),
                };
            }
            continue;
        }
        let Some(value) = line
            .strip_prefix("path")
            .and_then(|rest| rest.trim_start().strip_prefix('='))
        else {
            continue;
        };
        let path = value
            .trim()
            .strip_prefix('"')
            .and_then(|rest| rest.split('"').next())
            .unwrap_or_else(|| panic!("Cargo.toml: a target path is not a string: {line}"));
        match table.as_str() {
            "[lib]" => lib = Some(normal(Path::new(path), Path::new("Cargo.toml"))),
            "[[bin]]" => {
                roots.insert(normal(Path::new(path), Path::new("Cargo.toml")));
                bin_without_path = false;
            }
            _ => {}
        }
    }
    pathless_bin |= bin_without_path;
    assert!(
        autobins || !pathless_bin,
        "Cargo.toml: a [[bin]] without a path under autobins = false, which the walk cannot resolve"
    );
    let library_can_appear = match lib {
        Some(lib) => {
            roots.insert(lib);
            false
        }
        None if root.join("src/lib.rs").is_file() => {
            roots.insert(PathBuf::from("src/lib.rs"));
            false
        }
        None => true,
    };
    if autobins {
        if root.join("src/main.rs").is_file() {
            roots.insert(PathBuf::from("src/main.rs"));
        }
        if let Ok(entries) = fs::read_dir(root.join("src/bin")) {
            for entry in entries {
                let path = entry.expect("directory entry").path();
                let name = path.file_name().expect("entry name").to_string_lossy();
                let relative = Path::new("src/bin").join(&*name);
                if path.is_file() && name.ends_with(".rs") {
                    roots.insert(relative);
                } else if path.join("main.rs").is_file() {
                    roots.insert(relative.join("main.rs"));
                }
            }
        }
    }
    for crate_root in &roots {
        assert!(
            root.join(crate_root).is_file(),
            "crate root {} does not exist",
            crate_root.display()
        );
    }
    assert!(!roots.is_empty(), "the product has no crate root");
    CrateRoots {
        roots: roots.into_iter().collect(),
        discoverable: autobins || library_can_appear,
    }
}

/// A manifest table header, spelled `[name]` or `[[name]]` without its whitespace, quotes around a
/// whole name, and trailing comment. A header it cannot read fails rather than leave a target
/// root unread.
fn table_header(line: &str) -> String {
    let (array, rest) = match line.strip_prefix("[[") {
        Some(rest) => (true, rest),
        None => (false, &line[1..]),
    };
    let close = if array { "]]" } else { "]" };
    let unreadable = || -> ! { panic!("Cargo.toml: a table header the walk cannot read: {line}") };
    let end = rest.find(close).unwrap_or_else(|| unreadable());
    let after = rest[end + close.len()..].trim_start();
    if !(after.is_empty() || after.starts_with('#')) {
        unreadable();
    }
    let name = rest[..end].trim();
    let name = [('"', '"'), ('\'', '\'')]
        .iter()
        .find_map(|&(open, close)| name.strip_prefix(open)?.strip_suffix(close))
        .unwrap_or(name);
    if name.is_empty() || name.contains(['[', ']']) {
        unreadable();
    }
    if array {
        format!("[[{name}]]")
    } else {
        format!("[{name}]")
    }
}

/// The open-bracket depth of a manifest's values after `line`, from `depth` before it, and whether
/// a multi-line string (three double or three single quotes) is open after it; brackets inside
/// strings and comments do not count. A table header's own brackets are not values: the caller
/// resets the depth after one.
fn value_brackets(line: &str, mut depth: usize, in_string: &mut Option<&'static str>) -> usize {
    let mut rest = line;
    while !rest.is_empty() {
        if let Some(delimiter) = *in_string {
            match rest.find(delimiter) {
                Some(end) => {
                    rest = &rest[end + delimiter.len()..];
                    *in_string = None;
                }
                None => return depth,
            }
            continue;
        }
        if let Some(delimiter) = ["\"\"\"", "'''"].into_iter().find(|d| rest.starts_with(d)) {
            *in_string = Some(delimiter);
            rest = &rest[delimiter.len()..];
            continue;
        }
        let mut chars = rest.chars();
        match chars.next() {
            Some('#') => return depth,
            Some('[') => depth += 1,
            Some(']') => depth = depth.saturating_sub(1),
            Some(quote @ ('"' | '\'')) => {
                // A one-line string: to its closing quote (a basic string's escapes skipped).
                let mut escaped = false;
                for character in chars.by_ref() {
                    if escaped {
                        escaped = false;
                    } else if quote == '"' && character == '\\' {
                        escaped = true;
                    } else if character == quote {
                        break;
                    }
                }
            }
            _ => {}
        }
        rest = chars.as_str();
    }
    depth
}

/// The directory a module's `mod` declarations resolve against, as rustc keeps it: the module
/// file's directory, plus its own name when the file is not a crate root, `mod.rs`, or a `#[path]`
/// file (`a/b.rs` declares `a/b/c.rs`).
#[derive(Clone)]
struct ModuleDir {
    dir: PathBuf,
    relative: Option<String>,
}

impl ModuleDir {
    fn with_relative(&self) -> PathBuf {
        match &self.relative {
            Some(name) => self.dir.join(name),
            None => self.dir.clone(),
        }
    }
}

/// The walk over the module tree. Every path is relative to `root` and lexically normal.
struct Walk<'a> {
    root: &'a Path,
    /// Each file the product compiles or reads.
    files: BTreeSet<PathBuf>,
    /// Each module file already walked.
    modules: BTreeSet<PathBuf>,
}

impl Walk<'_> {
    fn read(&self, file: &Path) -> String {
        let path = self.root.join(file);
        fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read product source {}: {error}", path.display()))
    }

    fn module_file(&mut self, file: PathBuf, module: ModuleDir) {
        if !self.modules.insert(file.clone()) {
            return;
        }
        let text = self.read(&file);
        let tokens = lex(&text, &file);
        let (inner, items) = inner_attributes(&tokens);
        if cfg_of(&inner, &file) == Compiled::Never {
            return;
        }
        let file_dir = parent(&file);
        self.files.insert(file);
        for attribute in inner {
            self.includes(attribute, &file_dir);
        }
        self.items(items, &module, &file_dir);
    }

    fn inline_module(&mut self, body: &[Token<'_>], module: &ModuleDir, file_dir: &Path) {
        let (inner, items) = inner_attributes(body);
        if cfg_of(&inner, file_dir) == Compiled::Never {
            return;
        }
        for attribute in inner {
            self.includes(attribute, file_dir);
        }
        self.items(items, module, file_dir);
    }

    /// Walk a module's items: follow each `mod` the product may compile and scan every other item
    /// it may compile for includes.
    fn items(&mut self, mut rest: &[Token<'_>], module: &ModuleDir, file_dir: &Path) {
        while !rest.is_empty() {
            let mut attributes = Vec::new();
            while let [Token::Punct(b'#'), Token::Group(b'[', attribute), tail @ ..] = rest {
                attributes.push(attribute.as_slice());
                rest = tail;
            }
            let compiled = cfg_of(&attributes, file_dir);
            let mut start = 0;
            if matches!(rest.first(), Some(Token::Ident(word)) if *word == "pub") {
                start = 1;
                if matches!(rest.get(1), Some(Token::Group(b'(', _))) {
                    start = 2;
                }
            }
            if let (Some(Token::Ident(word)), Some(Token::Ident(name))) =
                (rest.get(start), rest.get(start + 1))
            {
                if *word == "mod" {
                    let body = rest.get(start + 2);
                    rest = rest.get(start + 3..).unwrap_or_default();
                    if compiled == Compiled::Never {
                        continue;
                    }
                    for attribute in &attributes {
                        self.includes(attribute, file_dir);
                    }
                    let path = path_attribute(&attributes, file_dir);
                    match body {
                        Some(Token::Punct(b';')) => self.declared(name, path, module),
                        Some(Token::Group(b'{', body)) => {
                            let inner = match path {
                                Some(path) => ModuleDir {
                                    dir: normal(&module.dir.join(path), file_dir),
                                    relative: None,
                                },
                                None => ModuleDir {
                                    dir: module.with_relative().join(name),
                                    relative: None,
                                },
                            };
                            self.inline_module(body, &inner, file_dir);
                        }
                        _ => panic!(
                            "{}: `mod {name}` is neither a declaration nor a block",
                            file_dir.display()
                        ),
                    }
                    continue;
                }
            }
            // Any other item ends at its first `;` or brace group: never later than the item does.
            let end = rest
                .iter()
                .position(|token| matches!(token, Token::Punct(b';') | Token::Group(b'{', _)))
                .map_or(rest.len(), |position| position + 1);
            if compiled != Compiled::Never {
                for attribute in &attributes {
                    self.includes(attribute, file_dir);
                }
                self.includes(&rest[..end], file_dir);
            }
            rest = &rest[end..];
        }
    }

    /// Follow `mod name;` from `module`: the `#[path]` file, else `name.rs` or `name/mod.rs`.
    fn declared(&mut self, name: &str, path: Option<String>, module: &ModuleDir) {
        let (file, child) = match path {
            // rustc treats every `#[path]` file as a `mod.rs` file.
            Some(path) => {
                let file = normal(&module.dir.join(path), &module.dir);
                let dir = parent(&file);
                (
                    file,
                    ModuleDir {
                        dir,
                        relative: None,
                    },
                )
            }
            None => {
                let base = module.with_relative();
                let default = base.join(format!("{name}.rs"));
                let secondary = base.join(name).join("mod.rs");
                match (
                    self.root.join(&default).is_file(),
                    self.root.join(&secondary).is_file(),
                ) {
                    (true, false) => (
                        default,
                        ModuleDir {
                            dir: base,
                            relative: Some(name.to_string()),
                        },
                    ),
                    (false, true) => (
                        secondary,
                        ModuleDir {
                            dir: base.join(name),
                            relative: None,
                        },
                    ),
                    (true, true) => panic!(
                        "module `{name}` is both {} and {}",
                        default.display(),
                        secondary.display()
                    ),
                    (false, false) => panic!(
                        "module `{name}` has no file: neither {} nor {}",
                        default.display(),
                        secondary.display()
                    ),
                }
            }
        };
        self.module_file(file, child);
    }

    /// Record every file an `include!`, `include_str!`, or `include_bytes!` in `tokens` names,
    /// relative to the file the invocation is in, and scan an included Rust file the same way.
    fn includes(&mut self, tokens: &[Token<'_>], file_dir: &Path) {
        for (index, token) in tokens.iter().enumerate() {
            let Token::Ident(word) = token else {
                if let Token::Group(_, inner) = token {
                    self.includes(inner, file_dir);
                }
                continue;
            };
            if *word == "mod"
                && matches!(tokens.get(index + 1), Some(Token::Ident(_)))
                && matches!(tokens.get(index + 2), Some(Token::Punct(b';')))
            {
                panic!(
                    "{}: a module file declared inside an item or an included file is not followed",
                    file_dir.display()
                );
            }
            if *word == "mod" && matches!(tokens.get(index + 1), Some(Token::Punct(b'$'))) {
                panic!(
                    "{}: a macro that declares modules is not followed",
                    file_dir.display()
                );
            }
            // rustc resolves an include in a macro body against each file that invokes the macro,
            // which the walk does not track.
            if *word == "macro_rules"
                && matches!(tokens.get(index + 1), Some(Token::Punct(b'!')))
                && matches!(tokens.get(index + 3), Some(Token::Group(_, body)) if names_an_include(body))
            {
                panic!(
                    "{}: an include inside a `macro_rules!` body is not followed",
                    file_dir.display()
                );
            }
            // A renamed include macro would be invoked by a name the walk does not know.
            if *word == "use" {
                let end = tokens[index..]
                    .iter()
                    .position(|token| *token == Token::Punct(b';'))
                    .map_or(tokens.len(), |end| index + end);
                if renames_an_include(&tokens[index + 1..end]) {
                    panic!(
                        "{}: a renamed include macro is not followed",
                        file_dir.display()
                    );
                }
            }
            if !INCLUDES.contains(word)
                || !matches!(tokens.get(index + 1), Some(Token::Punct(b'!')))
            {
                continue;
            }
            let Some(Token::Group(_, arguments)) = tokens.get(index + 2) else {
                continue;
            };
            let path = match arguments.as_slice() {
                [Token::Literal(Some(path))] | [Token::Literal(Some(path)), Token::Punct(b',')] => {
                    path
                }
                _ => panic!(
                    "{}: `{word}!` names its file by something other than a string literal",
                    file_dir.display()
                ),
            };
            let file = normal(&file_dir.join(path), file_dir);
            if !self.root.join(&file).is_file() {
                panic!("`{word}!` names {}, which is not a file", file.display());
            }
            if self.files.insert(file.clone()) && *word == "include" {
                let text = self.read(&file);
                let included = lex(&text, &file);
                self.includes(&included, &parent(&file));
            }
        }
    }
}

/// The macros that read a file by path.
const INCLUDES: [&str; 3] = ["include", "include_str", "include_bytes"];

/// Whether `tokens` (at any depth) invoke an include macro.
fn names_an_include(tokens: &[Token<'_>]) -> bool {
    tokens.iter().enumerate().any(|(index, token)| match token {
        Token::Ident(word) => {
            INCLUDES.contains(word) && matches!(tokens.get(index + 1), Some(Token::Punct(b'!')))
        }
        Token::Group(_, inner) => names_an_include(inner),
        _ => false,
    })
}

/// Whether a `use` tree's `tokens` (at any depth) import an include macro under another name.
fn renames_an_include(tokens: &[Token<'_>]) -> bool {
    tokens.iter().enumerate().any(|(index, token)| match token {
        Token::Ident(word) => {
            INCLUDES.contains(word) && matches!(tokens.get(index + 1), Some(Token::Ident("as")))
        }
        Token::Group(_, inner) => renames_an_include(inner),
        _ => false,
    })
}

/// The directory of a relative file path (`""` for a file at the root).
fn parent(file: &Path) -> PathBuf {
    file.parent().map(Path::to_path_buf).unwrap_or_default()
}

/// `path` relative to the product root with `.` and `..` resolved lexically; a path that leaves
/// the root (or is absolute) fails, named with `context`.
fn normal(path: &Path, context: &Path) -> PathBuf {
    let mut parts: Vec<Component<'_>> = Vec::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if parts.pop().is_none() {
                    panic!(
                        "{}: {} leaves the product root",
                        context.display(),
                        path.display()
                    );
                }
            }
            Component::Normal(_) => parts.push(component),
            Component::RootDir | Component::Prefix(_) => panic!(
                "{}: {} is not a path below the product root",
                context.display(),
                path.display()
            ),
        }
    }
    parts.iter().collect()
}

/// Whether the product can compile an item, judged by its `cfg` alone with `test` off.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Compiled {
    /// Never: the predicate needs `test`.
    Never,
    /// Under some features, platforms, or flags.
    Maybe,
    /// Always.
    Always,
}

impl Compiled {
    fn negate(self) -> Self {
        match self {
            Self::Never => Self::Always,
            Self::Maybe => Self::Maybe,
            Self::Always => Self::Never,
        }
    }

    fn all(values: impl IntoIterator<Item = Self>) -> Self {
        values
            .into_iter()
            .fold(Self::Always, |all, value| match (all, value) {
                (Self::Never, _) | (_, Self::Never) => Self::Never,
                (Self::Always, Self::Always) => Self::Always,
                _ => Self::Maybe,
            })
    }

    fn any(values: impl IntoIterator<Item = Self>) -> Self {
        values
            .into_iter()
            .fold(Self::Never, |any, value| match (any, value) {
                (Self::Always, _) | (_, Self::Always) => Self::Always,
                (Self::Never, Self::Never) => Self::Never,
                _ => Self::Maybe,
            })
    }
}

/// What the `cfg` and `cfg_attr(..., cfg(...))` attributes among `attributes` (each the contents of
/// one `#[...]` or `#![...]`) allow.
fn cfg_of(attributes: &[&[Token<'_>]], context: &Path) -> Compiled {
    Compiled::all(
        attributes
            .iter()
            .map(|attribute| attribute_cfg(attribute, Compiled::Always, context)),
    )
}

/// One attribute's `cfg` under `condition` (the predicates of the `cfg_attr`s around it).
fn attribute_cfg(attribute: &[Token<'_>], condition: Compiled, context: &Path) -> Compiled {
    match attribute {
        [Token::Ident(name), Token::Group(b'(', inner)] if *name == "cfg" => {
            // `cfg_attr(P, cfg(Q))` compiles the item unless P holds and Q does not.
            Compiled::any([condition.negate(), predicate(inner, context)])
        }
        [Token::Ident(name), Token::Group(b'(', inner)] if *name == "cfg_attr" => {
            let mut parts = split_commas(inner).into_iter();
            let predicate = predicate(parts.next().unwrap_or_default(), context);
            let condition = Compiled::all([condition, predicate]);
            Compiled::all(parts.map(|attribute| {
                // A `#[path]` that only a test build applies moves no product module.
                if condition != Compiled::Never
                    && matches!(attribute.first(), Some(Token::Ident(name)) if *name == "path")
                {
                    panic!(
                        "{}: a conditional `#[path]` is not followed",
                        context.display()
                    );
                }
                attribute_cfg(attribute, condition, context)
            }))
        }
        _ => Compiled::Always,
    }
}

/// A `cfg` predicate: `test` is off, and every other atom may hold.
fn predicate(tokens: &[Token<'_>], context: &Path) -> Compiled {
    match tokens {
        [Token::Ident(name)] if *name == "test" => Compiled::Never,
        [Token::Ident(_)] | [Token::Ident(_), Token::Punct(b'='), Token::Literal(_)] => {
            Compiled::Maybe
        }
        [Token::Ident(name), Token::Group(b'(', inner)] => {
            let parts = split_commas(inner);
            let values = parts.iter().map(|part| predicate(part, context));
            match *name {
                "all" => Compiled::all(values),
                "any" => Compiled::any(values),
                "not" if parts.len() == 1 => predicate(parts[0], context).negate(),
                _ => panic!("{}: unknown cfg predicate `{name}`", context.display()),
            }
        }
        _ => panic!(
            "{}: a cfg predicate the walk cannot read",
            context.display()
        ),
    }
}

/// The value of a `#[path = "..."]` among `attributes`.
fn path_attribute(attributes: &[&[Token<'_>]], context: &Path) -> Option<String> {
    attributes.iter().find_map(|attribute| match attribute {
        [Token::Ident(name), Token::Punct(b'='), Token::Literal(Some(path))] if *name == "path" => {
            Some(path.clone())
        }
        [Token::Ident(name), ..] if *name == "path" => {
            panic!("{}: a `#[path]` the walk cannot read", context.display())
        }
        _ => None,
    })
}

/// The comma-separated parts of a group's contents (a trailing comma adds no part).
fn split_commas<'t, 'a>(tokens: &'t [Token<'a>]) -> Vec<&'t [Token<'a>]> {
    let mut parts: Vec<&[Token<'a>]> = tokens.split(|token| *token == Token::Punct(b',')).collect();
    if parts.last().is_some_and(|part| part.is_empty()) {
        parts.pop();
    }
    parts
}

/// The leading inner attributes (`#![...]`, their contents) of a module's tokens, and its items.
fn inner_attributes<'t, 'a>(
    mut tokens: &'t [Token<'a>],
) -> (Vec<&'t [Token<'a>]>, &'t [Token<'a>]) {
    let mut attributes = Vec::new();
    while let [Token::Punct(b'#'), Token::Punct(b'!'), Token::Group(b'[', attribute), rest @ ..] =
        tokens
    {
        attributes.push(attribute.as_slice());
        tokens = rest;
    }
    (attributes, tokens)
}

/// A token of Rust source, as far as the walk reads it: comments and whitespace are gone and
/// delimited groups are nested.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Token<'a> {
    Ident(&'a str),
    /// Any other single punctuation byte.
    Punct(u8),
    /// A literal: a string literal's value (plain, raw, byte, or C), `None` for any other.
    Literal(Option<String>),
    /// A delimited group by its opening delimiter: `(`, `[`, or `{`.
    Group(u8, Vec<Token<'a>>),
}

/// Lex `text` (the file `file`) into token trees: whitespace, comments, and lifetimes dropped,
/// strings and characters read whole, so nothing in a comment or literal reads as code.
fn lex<'a>(text: &'a str, file: &Path) -> Vec<Token<'a>> {
    let bytes = text.as_bytes();
    let fail = |at: usize, what: &str| -> ! {
        let line = bytes[..at.min(bytes.len())]
            .iter()
            .filter(|&&byte| byte == b'\n')
            .count()
            + 1;
        panic!("{}:{line}: {what}", file.display())
    };
    let mut stack: Vec<(u8, Vec<Token<'a>>)> = vec![(0, Vec::new())];
    let mut at = 0;
    while at < bytes.len() {
        let byte = bytes[at];
        let token = match byte {
            _ if byte.is_ascii_whitespace() => {
                at += 1;
                continue;
            }
            b'/' if bytes.get(at + 1) == Some(&b'/') => {
                at = bytes[at..]
                    .iter()
                    .position(|&byte| byte == b'\n')
                    .map_or(bytes.len(), |end| at + end);
                continue;
            }
            b'/' if bytes.get(at + 1) == Some(&b'*') => {
                let mut depth = 0usize;
                loop {
                    match bytes.get(at..at + 2) {
                        Some(b"/*") => {
                            depth += 1;
                            at += 2;
                        }
                        Some(b"*/") => {
                            depth -= 1;
                            at += 2;
                            if depth == 0 {
                                break;
                            }
                        }
                        Some(_) => at += 1,
                        None => fail(at, "an unterminated block comment"),
                    }
                }
                continue;
            }
            b'(' | b'[' | b'{' => {
                stack.push((byte, Vec::new()));
                at += 1;
                continue;
            }
            b')' | b']' | b'}' => {
                let open = match byte {
                    b')' => b'(',
                    b']' => b'[',
                    _ => b'{',
                };
                let Some((delimiter, inner)) = stack.pop().filter(|_| !stack.is_empty()) else {
                    fail(at, "an unopened delimiter")
                };
                if delimiter != open {
                    fail(at, "mismatched delimiters");
                }
                at += 1;
                Token::Group(open, inner)
            }
            b'"' => {
                let (value, end) = string(bytes, at + 1).unwrap_or_else(|| fail(at, "a string"));
                at = end;
                Token::Literal(Some(value))
            }
            b'\'' => match character(bytes, at + 1) {
                Some(end) => {
                    at = end;
                    Token::Literal(None)
                }
                // A lifetime or a label: nothing the walk reads.
                None => {
                    at = identifier_end(bytes, at + 1);
                    continue;
                }
            },
            _ if byte.is_ascii_digit() => {
                at = identifier_end(bytes, at);
                Token::Literal(None)
            }
            _ if byte == b'_' || byte.is_ascii_alphabetic() || byte >= 0x80 => {
                let end = identifier_end(bytes, at);
                let word = &text[at..end];
                match (word, bytes.get(end)) {
                    ("b" | "c", Some(b'"')) => {
                        let (value, after) =
                            string(bytes, end + 1).unwrap_or_else(|| fail(at, "a string"));
                        at = after;
                        Token::Literal(Some(value))
                    }
                    ("b", Some(b'\'')) => {
                        at = character(bytes, end + 1).unwrap_or_else(|| fail(at, "a byte"));
                        Token::Literal(None)
                    }
                    ("r" | "br" | "cr", Some(b'"' | b'#')) => {
                        let hashes = bytes[end..].iter().take_while(|&&b| b == b'#').count();
                        if word == "r" && bytes.get(end + hashes) != Some(&b'"') {
                            // A raw identifier, `r#name`.
                            let start = end + 1;
                            at = identifier_end(bytes, start);
                            stack
                                .last_mut()
                                .expect("open group")
                                .1
                                .push(Token::Ident(&text[start..at]));
                            continue;
                        }
                        let (value, after) = raw_string(text, end + hashes + 1, hashes)
                            .unwrap_or_else(|| fail(at, "a raw string"));
                        at = after;
                        Token::Literal(Some(value))
                    }
                    _ => {
                        at = end;
                        Token::Ident(word)
                    }
                }
            }
            _ => {
                at += 1;
                Token::Punct(byte)
            }
        };
        stack.last_mut().expect("open group").1.push(token);
    }
    match stack.pop() {
        Some((0, tokens)) if stack.is_empty() => tokens,
        _ => fail(bytes.len(), "an unclosed delimiter"),
    }
}

fn identifier_end(bytes: &[u8], start: usize) -> usize {
    bytes[start..]
        .iter()
        .position(|&byte| !(byte == b'_' || byte.is_ascii_alphanumeric() || byte >= 0x80))
        .map_or(bytes.len(), |end| start + end)
}

/// A character literal's end, after the closing quote, given the byte after its opening one; `None`
/// when the quote opens a lifetime or label instead.
fn character(bytes: &[u8], start: usize) -> Option<usize> {
    if bytes.get(start) == Some(&b'\\') {
        let close = bytes
            .get(start + 2..)?
            .iter()
            .position(|&byte| byte == b'\'')?;
        return Some(start + 2 + close + 1);
    }
    let first = *bytes.get(start)?;
    let width = match first {
        0x00..=0x7f => 1,
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        _ => 4,
    };
    (bytes.get(start + width) == Some(&b'\'')).then_some(start + width + 1)
}

/// A quoted string's value and its end, after the closing quote, given the byte after the opening
/// quote.
fn string(bytes: &[u8], start: usize) -> Option<(String, usize)> {
    let mut at = start;
    while at < bytes.len() {
        match bytes[at] {
            b'\\' => at += 2,
            b'"' => {
                let raw = std::str::from_utf8(&bytes[start..at]).ok()?;
                return Some((unescape(raw), at + 1));
            }
            _ => at += 1,
        }
    }
    None
}

/// A raw string's value and its end, given the byte after its opening quote and its `#` count.
fn raw_string(text: &str, start: usize, hashes: usize) -> Option<(String, usize)> {
    let close = format!("\"{}", "#".repeat(hashes));
    let end = text.get(start..)?.find(&close)?;
    Some((
        text[start..start + end].to_string(),
        start + end + close.len(),
    ))
}

/// The value of a string literal's body.
fn unescape(raw: &str) -> String {
    let mut value = String::with_capacity(raw.len());
    let mut chars = raw.chars().peekable();
    while let Some(character) = chars.next() {
        if character != '\\' {
            value.push(character);
            continue;
        }
        match chars.next() {
            Some('n') => value.push('\n'),
            Some('r') => value.push('\r'),
            Some('t') => value.push('\t'),
            Some('0') => value.push('\0'),
            Some('x') => {
                let digits: String = chars.by_ref().take(2).collect();
                if let Ok(byte) = u8::from_str_radix(&digits, 16) {
                    value.push(char::from(byte));
                }
            }
            Some('u') => {
                let digits: String = chars
                    .by_ref()
                    .skip_while(|&c| c == '{')
                    .take_while(|&c| c != '}')
                    .filter(|&c| c != '_')
                    .collect();
                if let Some(c) = u32::from_str_radix(&digits, 16)
                    .ok()
                    .and_then(char::from_u32)
                {
                    value.push(c);
                }
            }
            // A line continuation skips the newline and the next line's leading whitespace.
            Some('\n' | '\r') => {
                while chars.peek().is_some_and(|c| c.is_whitespace()) {
                    chars.next();
                }
            }
            Some(other) => value.push(other),
            None => {}
        }
    }
    value
}
