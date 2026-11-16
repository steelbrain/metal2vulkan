//! The product fingerprint covers exactly what the product can compile or read:
//! `build_support::product_files` follows the crate roots' module tree as rustc resolves it, and
//! leaves out every module or item whose `cfg` needs `test`, with everything below it. The build
//! scripts that take it watch exactly those inputs (`build_support::product_inputs`).

use crate::build_support::{product_files, product_hash, product_inputs};
use crate::product_fingerprint::product_fingerprint;
use crate::ScratchDir;
use sha2::Digest;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn write(root: &Path, path: &str, text: &str) {
    let path = root.join(path);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

fn fingerprint(root: &Path) -> String {
    crate::hash::hex_encode(product_hash(root).finalize())
}

fn files(root: &Path) -> BTreeSet<PathBuf> {
    product_files(root)
        .into_iter()
        .map(|file| file.strip_prefix(root).unwrap().to_path_buf())
        .collect()
}

/// The library root: modules under every kind of `cfg`, a `#[path]` module, includes in an inner
/// attribute and in an item, an include in a test-only item, and comments and literals that name
/// modules and includes a naive scan would follow.
const LIB: &str = r##"//! A product. mod named_in_a_doc_comment;
#![cfg_attr(docs, doc = include_str!("guide.md"))]
mod product;
mod folder;
#[cfg(feature = "extra")]
pub mod gated;
#[cfg(any(test, feature = "tools"))]
pub(crate) mod tools;
#[cfg(not(test))]
mod production;
#[cfg_attr(feature = "extra", cfg(test))]
mod unless_extra;
#[cfg(test)]
mod tests;
#[cfg(all(test, feature = "extra"))]
mod gated_tests;
#[path = "generated/table.rs"]
mod table;
mod own_test_cfg;
pub const DATA: &[u8] = include_bytes!("data.bin");
/* mod in_a_block_comment; /* nested */ mod still_in_it; include_str!("missing.txt"); */
pub const TEXT: &str = "mod in_a_string; include_str!(\"missing.txt\")";
pub const RAW: &str = r#"mod in_a_raw_string; ""#;
pub const QUOTE: char = '"';
pub const BYTE: u8 = b'{';
pub fn first<'a>(text: &'a str) -> &'a str {
    'outer: loop {
        break 'outer text;
    }
}
#[cfg(test)]
pub fn fixture() -> &'static str {
    include_str!("test_fixture.txt")
}
"##;

/// Each file of the synthetic product, and whether the product can compile or read it.
const TREE: &[(&str, &str, bool)] = &[
    ("src/lib.rs", LIB, true),
    (
        "src/product.rs",
        "mod nested;\n#[cfg(test)]\nmod tests;\n#[cfg(test)]\nmod inline_tests {\n    const \
         FIXTURE: &str = include_str!(\"inline_fixture.txt\");\n    mod helper;\n}\nmod inline {\n    \
         mod deep;\n}\n",
        true,
    ),
    ("src/product/nested.rs", "pub fn nested() {}\n", true),
    ("src/product/inline/deep.rs", "pub fn deep() {}\n", true),
    ("src/product/tests.rs", "mod helpers;\n", false),
    ("src/product/tests/helpers.rs", "fn helper() {}\n", false),
    ("src/product/inline_tests/helper.rs", "fn helper() {}\n", false),
    // A `mod.rs` file declares its modules beside itself.
    ("src/folder/mod.rs", "mod child;\n#[cfg(test)]\nmod tests;\n", true),
    ("src/folder/child.rs", "pub fn child() {}\n", true),
    ("src/folder/tests.rs", "fn tests() {}\n", false),
    ("src/gated.rs", "pub fn gated() {}\n", true),
    ("src/tools.rs", "pub fn tools() {}\n", true),
    ("src/production.rs", "pub fn production() {}\n", true),
    ("src/unless_extra.rs", "pub fn unless_extra() {}\n", true),
    ("src/tests.rs", "mod more;\n", false),
    ("src/tests/more.rs", "fn more() {}\n", false),
    ("src/gated_tests.rs", "fn gated_tests() {}\n", false),
    // A `#[path]` file declares its modules beside itself.
    ("src/generated/table.rs", "mod row;\n", true),
    ("src/generated/row.rs", "pub fn row() {}\n", true),
    ("src/own_test_cfg.rs", "#![cfg(test)]\nmod hidden;\n", false),
    ("src/own_test_cfg/hidden.rs", "fn hidden() {}\n", false),
    ("src/guide.md", "A guide.\n", true),
    ("src/data.bin", "\u{1}\u{2}", true),
    ("src/test_fixture.txt", "fixture\n", false),
    ("src/inline_fixture.txt", "fixture\n", false),
    ("src/main.rs", "mod cli;\nfn main() {}\n", true),
    ("src/cli.rs", "pub fn cli() {}\n", true),
    ("src/orphan.rs", "fn orphan() {}\n", false),
    ("src/README.md", "Notes.\n", false),
    ("examples/demo.rs", "fn main() {}\n", false),
    ("tests/integration.rs", "#[test]\nfn integration() {}\n", false),
    ("validation/src/lib.rs", "pub fn census() {}\n", false),
];

/// The synthetic product's manifest, its `[package]` table ending in `package_tail`.
fn manifest(package_tail: &str) -> String {
    format!(
        "[package]\nname = \"product\"\n{package_tail}\n[lib]\npath = \"src/lib.rs\"\n\n[[bin]]\n\
         name = \"tool\"\npath = \"src/main.rs\"\n\n[[example]]\nname = \"demo\"\npath = \
         \"examples/demo.rs\"\n"
    )
}

fn product(root: &Path) {
    write(root, "Cargo.toml", &manifest(""));
    write(root, "build.rs", "fn main() {}\n");
    write(root, "build_support.rs", "// fingerprint inputs\n");
    for (path, text, _) in TREE {
        write(root, path, text);
    }
}

/// The file set: the module tree under every feature, `#[path]` and `include*!` targets
/// included, test-only modules (with their subtrees), a test-only item's include, and files no
/// module reaches excluded.
#[test]
fn the_product_fingerprint_covers_exactly_the_compiled_module_tree() {
    let scratch = ScratchDir::new("product-files").unwrap();
    let root = scratch.path();
    product(root);
    let expected = TREE
        .iter()
        .filter(|(_, _, product)| *product)
        .map(|(path, _, _)| PathBuf::from(path))
        .collect::<BTreeSet<_>>();
    assert_eq!(files(root), expected);
}

/// Editing or adding a file the product cannot compile or read leaves the fingerprint unchanged;
/// editing any file it can (a module, a feature-gated module, a binary's module, a `#[path]`
/// module, an included data file, the manifest, the build script) changes it.
#[test]
fn the_product_fingerprint_follows_exactly_the_product_files() {
    let scratch = ScratchDir::new("product-fingerprint").unwrap();
    let root = scratch.path();
    product(root);
    let before = fingerprint(root);
    for (path, text, product) in TREE {
        write(root, path, &format!("{text}\n// edited\n"));
        if *product {
            assert_ne!(fingerprint(root), before, "an edit to {path} is an input");
        } else {
            assert_eq!(
                fingerprint(root),
                before,
                "an edit to {path} is not an input"
            );
        }
        write(root, path, text);
    }
    for path in [
        "src/new_orphan.rs",
        "src/scratch/notes.rs",
        "src/new/tests.rs",
    ] {
        write(root, path, "fn unreachable() {}\n");
        assert_eq!(fingerprint(root), before, "{path} is reached by no module");
    }
    for path in ["Cargo.toml", "build.rs", "build_support.rs"] {
        let text = std::fs::read_to_string(root.join(path)).unwrap();
        write(root, path, &format!("{text}\n# edited\n"));
        assert_ne!(fingerprint(root), before, "an edit to {path} is an input");
        write(root, path, &text);
    }
    assert_eq!(fingerprint(root), before);
}

/// Whether the build script asks Cargo to rerun when `path` (below `root`) changes: it hashes the
/// file, or watches a directory above it.
fn watched(root: &Path, path: &str) -> bool {
    let inputs = product_inputs(root);
    let path = root.join(path);
    inputs.files.contains(&path) || inputs.watched.iter().any(|dir| path.starts_with(dir))
}

/// The build script watches exactly what can change the product fingerprint. Under a manifest that
/// declares every crate root (`autobins = false`), that is the files it hashes and no directory: an
/// edit to a test-only module (`src/tests/`) reruns nothing, while every product module (so its
/// edit or deletion), the manifest, and a binary the manifest gains still rerun it, and a file
/// under `src/bin/` Cargo would not build stays out. Under a manifest that lets Cargo discover
/// binaries, or a library, it also watches `src/`, where one can appear with no hashed file
/// changing.
#[test]
fn the_build_script_watches_exactly_what_can_change_the_product() {
    let scratch = ScratchDir::new("product-inputs").unwrap();
    let root = scratch.path();
    product(root);
    write(root, "Cargo.toml", &manifest("autobins = false\n"));
    assert!(product_inputs(root).watched.is_empty());
    for (path, _, product) in TREE {
        assert_eq!(watched(root, path), *product, "{path}");
    }
    for path in ["Cargo.toml", "build.rs", "build_support.rs"] {
        assert!(watched(root, path), "{path}");
    }
    write(root, "src/bin/undeclared.rs", "fn main() {}\n");
    assert!(!watched(root, "src/bin/undeclared.rs"));
    assert!(!files(root).contains(Path::new("src/bin/undeclared.rs")));
    write(
        root,
        "Cargo.toml",
        &format!(
            "{}\n[[bin]]\nname = \"extra\"\npath = \"src/extra/main.rs\"\n",
            manifest("autobins = false\n")
        ),
    );
    write(root, "src/extra/main.rs", "mod part;\nfn main() {}\n");
    write(root, "src/extra/part.rs", "pub fn part() {}\n");
    for path in ["src/extra/main.rs", "src/extra/part.rs"] {
        assert!(watched(root, path), "a declared binary's {path}");
    }
    assert!(product_inputs(root).watched.is_empty());

    // A discovering manifest: binaries, then the library.
    write(root, "Cargo.toml", &manifest(""));
    assert_eq!(product_inputs(root).watched, vec![root.join("src")]);
    assert!(files(root).contains(Path::new("src/bin/undeclared.rs")));
    write(
        root,
        "Cargo.toml",
        "[package]\nname = \"product\"\nautobins = false\n[[bin]]\nname = \"tool\"\npath = \
         \"src/main.rs\"\n",
    );
    std::fs::remove_file(root.join("src/lib.rs")).unwrap();
    assert_eq!(product_inputs(root).watched, vec![root.join("src")]);
}

/// A form the walk cannot resolve fails rather than leaving an input out.
#[test]
fn the_product_walk_refuses_what_it_cannot_resolve() {
    let refused = |lib: &str| {
        let scratch = ScratchDir::new("product-walk-refusal").unwrap();
        let root = scratch.path();
        write(root, "Cargo.toml", "[package]\nname = \"product\"\n");
        write(root, "src/lib.rs", lib);
        write(root, "src/data.txt", "data\n");
        let root = root.to_path_buf();
        std::panic::catch_unwind(move || product_files(&root)).is_err()
    };
    assert!(
        refused("mod missing;\n"),
        "a declared module without a file"
    );
    assert!(
        refused("const A: &str = include_str!(concat!(\"a\", \".txt\"));\n"),
        "an include whose path is not a string literal"
    );
    assert!(
        refused("#[cfg_attr(feature = \"x\", path = \"x.rs\")]\nmod x;\n"),
        "a conditional `#[path]`"
    );
    assert!(
        refused("macro_rules! declare {\n    ($name:ident) => {\n        mod $name;\n    };\n}\n"),
        "a macro that declares modules"
    );
    // rustc resolves a macro body's include against each file that invokes the macro, here
    // `src/data.txt` from `src/lib.rs` but `src/user/data.txt` from `src/user.rs`, so even a body
    // whose file exists beside its definition is refused.
    assert!(
        refused(
            "macro_rules! data {\n    () => {\n        include_str!(\"data.txt\")\n    };\n}\n\
             pub const DATA: &str = data!();\n"
        ),
        "an include in a macro body"
    );
    assert!(
        refused("use std::include_str as text;\npub const DATA: &str = text!(\"data.txt\");\n"),
        "a renamed include"
    );
    assert!(
        refused("use std::{fmt, include_bytes as bytes};\n"),
        "a renamed include in a use group"
    );
    assert!(!refused("pub fn product() {}\n"));
    assert!(!refused(
        "use std::include_str;\npub const DATA: &str = include_str!(\"data.txt\");\n"
    ));
}

/// The manifest's `[lib]` and `[[bin]]` paths are read past what a line-by-line reading would
/// trip on (a header's trailing comment and spacing, multi-line arrays and strings), and a manifest
/// it cannot read fails rather than leave a moved crate root unhashed.
#[test]
fn the_product_walk_reads_the_manifests_target_roots() {
    let roots = |manifest: &str| {
        let scratch = ScratchDir::new("product-walk-manifest").unwrap();
        let root = scratch.path().to_path_buf();
        write(&root, "Cargo.toml", manifest);
        write(&root, "src/lib.rs", "pub fn default_root() {}\n");
        write(&root, "src/core.rs", "pub fn moved_root() {}\n");
        write(&root, "src/tool.rs", "fn main() {}\n");
        std::panic::catch_unwind(move || files(&root)).ok()
    };
    let moved = BTreeSet::from([PathBuf::from("src/core.rs"), PathBuf::from("src/tool.rs")]);
    let manifest = "[package]\nname = \"product\"\nexclude = [\n    [\"nested\", \"array\"],\n    \
                    \"[lib]\",\n]\ndescription = \"\"\"\n[lib]\npath = \"src/lib.rs\"\n\"\"\"\n\n\
                    [lib] # the library\npath = \"src/core.rs\" # moved\n\n[[ bin ]]\nname = \
                    \"tool\"\npath = \"src/tool.rs\"\n\n[target.'cfg(unix)'.dependencies]\n";
    assert_eq!(roots(manifest), Some(moved));
    assert_eq!(
        roots("[package]\nname = \"product\"\n[\"lib\"]\npath = \"src/core.rs\"\n"),
        Some(BTreeSet::from([PathBuf::from("src/core.rs")]))
    );
    assert_eq!(
        roots(
            "[package]\nname = \"product\"\nautobins = false # declared\n[[bin]]\nname = \
             \"tool\"\npath = \"src/tool.rs\"\n"
        ),
        Some(BTreeSet::from([
            PathBuf::from("src/lib.rs"),
            PathBuf::from("src/tool.rs")
        ]))
    );
    for unreadable in [
        "[package]\nname = \"product\"\n[lib\npath = \"src/core.rs\"\n",
        "[package]\nname = \"product\"\n[lib] path = \"src/core.rs\"\n",
        "lib.path = \"src/core.rs\"\n[package]\nname = \"product\"\n",
        "[package]\nname = \"product\"\n[lib]\npath = 'src/core.rs'\n",
        "package.autobins = false\n[package]\nname = \"product\"\n",
        "[package]\nname = \"product\"\nautobins = \"no\"\n",
        "[package]\nname = \"product\"\nautobins = false\n[[bin]]\nname = \"tool\"\n",
        "[package]\nname = \"product\"\nautobins = false\n[[bin]]\nname = \"tool\"\n[features]\n",
    ] {
        assert_eq!(roots(unreadable), None, "{unreadable}");
    }
}

/// The AIR front end's crate (`metal2vulkan-air`) is part of the product fingerprint where the
/// manifest names it by path, inline or as its own table: its manifest, its build script, and its
/// module tree, test-only modules left out. A published manifest names it by version alone, and
/// then nothing of it is read; a path the walk cannot read fails.
#[test]
fn the_product_fingerprint_covers_the_front_end_crate_by_path() {
    let scratch = ScratchDir::new("product-front-end").unwrap();
    let root = scratch.path();
    product(root);
    write(
        root,
        "front/Cargo.toml",
        "[package]\nname = \"metal2vulkan-air\"\nautobins = false\n\n[lib]\npath = \"src/lib.rs\"\n",
    );
    write(root, "front/build.rs", "fn main() {}\n");
    write(
        root,
        "front/src/lib.rs",
        "pub mod read;\n#[cfg(test)]\nmod tests;\n",
    );
    write(root, "front/src/read.rs", "pub fn read() {}\n");
    write(root, "front/src/tests.rs", "fn tests() {}\n");
    let front_end = |dependency: &str| {
        write(
            root,
            "Cargo.toml",
            &format!("{}\n{dependency}", manifest("")),
        );
        product_inputs(root)
            .files
            .into_iter()
            .filter_map(|file| {
                let file = file.strip_prefix(root).unwrap();
                file.starts_with("front").then(|| file.to_path_buf())
            })
            .collect::<BTreeSet<_>>()
    };
    let covered = [
        "front/Cargo.toml",
        "front/build.rs",
        "front/src/lib.rs",
        "front/src/read.rs",
    ]
    .map(PathBuf::from)
    .into_iter()
    .collect::<BTreeSet<_>>();
    let inline = "[dependencies]\nmetal2vulkan-air = { version = \"=0.1.0\", path = \"front\" }\n";
    let table = "[dependencies.metal2vulkan-air]\nversion = \"=0.1.0\"\npath = \"front\"\n";
    assert_eq!(front_end(inline), covered);
    assert_eq!(front_end(table), covered);
    assert!(front_end("[dependencies.metal2vulkan-air]\nversion = \"=0.1.0\"\n").is_empty());
    assert!(front_end("[dependencies]\nmetal2vulkan-air = \"=0.1.0\"\n").is_empty());

    front_end(inline);
    let before = fingerprint(root);
    write(root, "front/src/tests.rs", "fn tests() {}\n// edited\n");
    assert_eq!(
        fingerprint(root),
        before,
        "a test-only module is not an input"
    );
    write(root, "front/src/read.rs", "pub fn read() {}\n// edited\n");
    assert_ne!(fingerprint(root), before, "a front-end module is an input");

    let unreadable = "[dependencies]\nmetal2vulkan-air = { version = \"=0.1.0\", path = front }\n";
    write(
        root,
        "Cargo.toml",
        &format!("{}\n{unreadable}", manifest("")),
    );
    let root = root.to_path_buf();
    assert!(std::panic::catch_unwind(move || product_inputs(&root)).is_err());
}

/// This build's product fingerprint is the walk over this workspace's translator, then the
/// runtime's and the dev-tools crates (`product_fingerprint.rs`, which `build.rs` shares), each in
/// its directory beside this crate, and the walk leaves out each crate's test-only modules: no file
/// it lists is declared only under `#[cfg(test)]`.
#[test]
fn this_builds_product_fingerprint_is_the_walk_over_the_product() {
    let crates = metal2vulkan_devtools::workspace_dir().join("crates");
    assert_eq!(
        product_fingerprint(&crates),
        env!("METAL2VULKAN_PRODUCT_FINGERPRINT")
    );
    let root = crates.join("metal2vulkan");
    let product = files(&root);
    for root_file in ["src/lib.rs", "src/main.rs"] {
        assert!(product.contains(Path::new(root_file)), "{root_file}");
    }
    // A module every product build compiles, and two only tests compile, by their declarations.
    assert!(product.contains(Path::new("src/indirect_commands/host.rs")));
    assert!(!product.contains(Path::new("src/pipeline/testing.rs")));
    assert!(!product.contains(Path::new("src/pipeline/api_tests.rs")));
    // The runtime's crate by its module tree, its tests left out.
    let runtime = files(&crates.join("metal2vulkan-runtime"));
    assert!(runtime.contains(Path::new("src/lib.rs")));
    assert!(!runtime.iter().any(|file| file.starts_with("src/tests")));
    // The front end's crate by its module tree, its test-only modules left out, named through the
    // translator's manifest path (`../metal2vulkan-air`).
    let inputs = product_inputs(&root);
    let front_end = |file: &str| {
        inputs
            .files
            .contains(&root.join("../metal2vulkan-air").join(file))
    };
    assert!(front_end("Cargo.toml") && front_end("src/lib.rs"));
    assert!(front_end("src/canon/rate_map_data.rs"));
    assert!(!front_end("src/intrinsics/families_tests.rs"));
    // The translator inherits from the workspace, so the workspace's manifest is an input.
    assert!(inputs.files.contains(&root.join("../../Cargo.toml")));
    // The manifests declare every crate root, so the build scripts watch no directory, and an edit
    // to the crate's own tests (`src/tests/`) reruns neither.
    assert!(inputs.watched.is_empty(), "{:?}", inputs.watched);
    assert!(
        !product.iter().any(|file| file.starts_with("src/tests")),
        "the crate's tests are not product inputs"
    );
}

/// A product manifest that inherits from its workspace (`edition.workspace = true`, a dependency's
/// `workspace = true`) makes the workspace's manifest an input, named through the product's
/// directory, so an edit to the edition, a shared dependency or a profile there moves the
/// fingerprint; a manifest that inherits nothing reads no other manifest.
#[test]
fn the_product_fingerprint_covers_the_workspace_manifest_it_inherits_from() {
    let scratch = ScratchDir::new("product-workspace").unwrap();
    let workspace = scratch.path();
    write(
        workspace,
        "Cargo.toml",
        "[workspace] # the members\nmembers = [\"crates/product\"]\n\n[workspace.package]\n\
         edition = \"2021\"\n",
    );
    let root = workspace.join("crates/product");
    product(&root);
    let inherits = |tail: &str| {
        write(&root, "Cargo.toml", &manifest(tail));
        product_inputs(&root)
            .files
            .contains(&root.join("../../Cargo.toml"))
    };
    assert!(!inherits(""));
    assert!(inherits("edition.workspace = true\n"));
    assert!(inherits("[dependencies]\nsha2 = { workspace = true }\n"));
    assert!(inherits(
        "[dependencies]\nash = { version = \"0\", workspace = true }\n"
    ));
    assert!(!inherits("# edition.workspace = true\n"));

    inherits("edition.workspace = true\n");
    let before = fingerprint(&root);
    write(
        workspace,
        "Cargo.toml",
        "[workspace]\nmembers = [\"crates/product\"]\n\n[workspace.package]\nedition = \"2024\"\n",
    );
    assert_ne!(
        fingerprint(&root),
        before,
        "an edit to the workspace's manifest is an input"
    );
}

/// The validation crate builds its own copy of the translator's `build_support.rs` (no crate
/// includes a file outside its own directory), and the two are one file, so the product
/// fingerprint this crate takes is the walk the translator's build script takes.
#[test]
fn the_build_support_copy_is_the_translators() {
    let crate_dir = metal2vulkan_devtools::validation_dir();
    let ours = std::fs::read(crate_dir.join("build_support.rs")).unwrap();
    let translators = std::fs::read(crate_dir.join("../metal2vulkan/build_support.rs")).unwrap();
    assert!(
        ours == translators,
        "crates/metal2vulkan-validation/build_support.rs differs from the translator's \
         crates/metal2vulkan/build_support.rs: copy the translator's file over it"
    );
}
