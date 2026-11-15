//! `Error::Invalid` is misuse plain Metal itself refuses, and `Error::Nil` what plain Metal
//! answers nil for; Metal's API validation layer is not Metal, so a rule only the layer stops is
//! no refusal at all. These tests walk the runtime's own sources: every refusal its docs label
//! `Invalid`, and every problem a call refuses `Invalid`, checking that none is one Metal is
//! documented to answer nil for.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

/// A documented item of a source file.
struct Doc {
    /// The file.
    path: PathBuf,
    /// The doc's first line, from 1.
    line: usize,
    /// The doc's text, its lines joined.
    text: String,
    /// The enum the item sits in, where it is one's variant.
    within: Option<String>,
    /// The line documented: the first after the doc that is no attribute or comment.
    item: String,
}

/// The runtime's sources but its tests: each file's path and its text before an inline test
/// module.
fn sources() -> Vec<(PathBuf, String)> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                if path.file_name().is_some_and(|name| name != "tests") {
                    walk(&path, out);
                }
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                out.push(path);
            }
        }
    }
    let mut paths = Vec::new();
    walk(
        &metal2vulkan_devtools::workspace_dir().join("crates/metal2vulkan-runtime/src"),
        &mut paths,
    );
    paths.sort();
    paths
        .into_iter()
        .map(|path| {
            let text = fs::read_to_string(&path).unwrap();
            let lines: Vec<&str> = text.lines().collect();
            let end = (0..lines.len())
                .find(|&i| {
                    lines[i].trim() == "#[cfg(test)]"
                        && lines.get(i + 1).is_some_and(|next| {
                            let next = next.trim();
                            next.starts_with("mod ") && next.ends_with('{')
                        })
                })
                .unwrap_or(lines.len());
            (path, lines[..end].join("\n"))
        })
        .collect()
}

/// Whether `text` has the word `nil`.
fn says_nil(text: &str) -> bool {
    text.split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .any(|word| word == "nil")
}

/// Whether `text` labels a refusal `Invalid`.
fn labels_invalid(text: &str) -> bool {
    text.contains("`Invalid") || text.contains("Error::Invalid")
}

/// Whether `text` labels a refusal `Nil`.
fn labels_nil(text: &str) -> bool {
    text.contains("`Nil") || text.contains("Error::Nil")
}

/// The clauses of a doc: its sentences, and their parts between semicolons.
fn clauses(text: &str) -> impl Iterator<Item = &str> {
    text.split(". ").flat_map(|sentence| sentence.split("; "))
}

/// The identifier `text` starts with.
fn leading_ident(text: &str) -> &str {
    let end = text
        .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .unwrap_or(text.len());
    &text[..end]
}

/// The identifier `text` ends with.
fn trailing_ident(text: &str) -> &str {
    let start = text
        .rfind(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .map_or(0, |i| i + 1);
    &text[start..]
}

/// The docs of `text`, a file at `path`: `///` and `//!` blocks, each with the enum it sits in
/// (the last `enum` opened before it, while its variants' indentation lasts).
fn docs(path: &Path, text: &str) -> Vec<Doc> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    let mut within: Option<String> = None;
    let mut i = 0;
    while i < lines.len() {
        let trimmed = lines[i].trim();
        if let Some(at) = trimmed.find("enum ") {
            let head = &trimmed[..at];
            if head.is_empty() || head.trim_end().starts_with("pub") {
                within = Some(leading_ident(&trimmed[at + 5..]).to_owned());
            }
        } else if trimmed == "}" && !lines[i].starts_with(' ') {
            within = None;
        }
        if !(trimmed.starts_with("///") || trimmed.starts_with("//!")) {
            i += 1;
            continue;
        }
        let start = i;
        let mut text = Vec::new();
        while i < lines.len() {
            let doc = lines[i].trim();
            if !(doc.starts_with("///") || doc.starts_with("//!")) {
                break;
            }
            text.push(doc[3..].trim());
            i += 1;
        }
        let item = lines[i..]
            .iter()
            .map(|line| line.trim())
            .find(|line| !line.starts_with("#[") && !line.starts_with("//"))
            .unwrap_or("")
            .to_owned();
        out.push(Doc {
            path: path.to_owned(),
            line: start + 1,
            text: text.join(" "),
            within: within.clone(),
            item,
        });
    }
    out
}

/// The enum variants documented as Metal's nil alone, `(enum, variant)`: a variant whose doc
/// says nil and labels it `Nil`, and labels no `Invalid`.
fn nil_only(docs: &[Doc]) -> BTreeSet<(String, String)> {
    docs.iter()
        .filter(|doc| says_nil(&doc.text) && labels_nil(&doc.text) && !labels_invalid(&doc.text))
        .filter_map(|doc| {
            let variant = leading_ident(&doc.item);
            let is_variant = variant.starts_with(|c: char| c.is_ascii_uppercase())
                && matches!(
                    doc.item[variant.len()..].chars().next(),
                    None | Some(' ' | '(' | ',')
                );
            let within = doc.within.as_ref()?;
            is_variant.then(|| (within.clone(), variant.to_owned()))
        })
        .collect()
}

/// The text of the parenthesized span `text` opens at its `(`, `open`.
fn span(text: &str, open: usize) -> &str {
    let mut depth = 0;
    for (i, c) in text[open..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
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

/// The `Enum::Variant` paths of `text`.
fn paths(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(at) = text[from..].find("::") {
        let at = from + at;
        let (before, after) = (trailing_ident(&text[..at]), leading_ident(&text[at + 2..]));
        if !before.is_empty() && !after.is_empty() {
            out.push((before.to_owned(), after.to_owned()));
        }
        from = at + 2;
    }
    out
}

/// The text of the braced body `text` opens at its `{`, `open`.
fn body(text: &str, open: usize) -> &str {
    let mut depth = 0;
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

/// The file's functions that return a `Problem` (`fn name(..) -> Problem {`), each with the
/// `Enum::Variant` paths of its body: a refusal that calls one refuses those.
fn problem_helpers(text: &str) -> Vec<(String, Vec<(String, String)>)> {
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(at) = text[from..].find("-> Problem {") {
        let at = from + at;
        let open = at + "-> Problem ".len();
        if let Some(function) = text[..at].rfind("fn ") {
            if !text[function..at].contains(['{', ';']) {
                let name = leading_ident(&text[function + 3..]).to_owned();
                out.push((name, paths(body(text, open))));
            }
        }
        from = open;
    }
    out
}

/// The `Enum::Variant` paths of each refusal `text` makes `Invalid`: inside a call of
/// `invalid`, of a helper of the file that wraps it (a function returning `Error` whose body
/// starts with `invalid(`, or a one-line closure `let name = |..| invalid(`), or of a `Stops`
/// refusal, and a problem converted with `.into()` (a texture descriptor's, `Refused::Stops`);
/// a call inside such a refusal of a function of the file that returns a `Problem` refuses the
/// paths of its body.
fn invalid_paths(text: &str) -> Vec<(String, String)> {
    let helpers = problem_helpers(text);
    let mut callees = BTreeSet::from(["invalid".to_owned(), "Stops".to_owned()]);
    let mut from = 0;
    while let Some(at) = text[from..].find("invalid(") {
        let at = from + at;
        let before = text[..at].trim_end();
        if before.ends_with('|') {
            if let Some(binding) = before.rfind("let ") {
                if !before[binding..].contains(['\n', ';']) {
                    callees.insert(leading_ident(&before[binding + 4..]).to_owned());
                }
            }
        } else if let Some(head) = before.strip_suffix('{') {
            if let Some(function) = head.rfind("fn ") {
                let signature = &head[function..];
                if signature.contains("-> Error") && !signature.contains(['{', ';']) {
                    callees.insert(leading_ident(&head[function + 3..]).to_owned());
                }
            }
        }
        from = at + 1;
    }
    let mut out = Vec::new();
    for callee in &callees {
        let call = format!("{callee}(");
        let mut from = 0;
        while let Some(at) = text[from..].find(&call) {
            let at = from + at;
            let before = &text[..at];
            let ident_before = before.ends_with(|c: char| c.is_ascii_alphanumeric() || c == '_');
            if !ident_before && !before.ends_with("fn ") {
                let refusal = span(text, at + callee.len());
                out.extend(paths(refusal));
                for (helper, helper_paths) in &helpers {
                    let call = format!("{helper}(");
                    let calls = refusal.match_indices(&call).any(|(i, _)| {
                        !refusal[..i].ends_with(|c: char| c.is_ascii_alphanumeric() || c == '_')
                    });
                    if calls {
                        out.extend(helper_paths.iter().cloned());
                    }
                }
            }
            from = at + call.len();
        }
    }
    let problems: BTreeSet<(String, String)> = paths(text)
        .into_iter()
        .filter(|(problem, _)| problem.ends_with("Problem"))
        .collect();
    for (problem, variant) in problems {
        let path = format!("{problem}::{variant}");
        let mut from = 0;
        while let Some(at) = text[from..].find(&path) {
            let mut rest = text[at + from + path.len()..].trim_start();
            if rest.starts_with('{') {
                rest = rest[rest.find('}').map_or(rest.len(), |end| end + 1)..].trim_start();
            }
            if rest.starts_with(".into()") {
                out.push((problem.clone(), variant.clone()));
            }
            from += at + path.len();
        }
    }
    out
}

/// The walk's own parts on a synthetic source: its docs, the variants documented as nil alone,
/// and the paths of the refusals made `Invalid` through each form.
#[test]
fn the_walk_finds_nil_variants_and_invalid_refusals() {
    let text = "\
pub enum FooProblem {
    /// Metal answers nil (`Nil`).
    Gone,
    /// Metal stops it (`Invalid`), or answers nil (`Nil`).
    Both { at: u64 },
    /// Metal stops it.
    Stopped,
    /// Metal answers nil (`Nil`).
    Far { at: u64 },
}

fn problem(call: &'static str, problem: FooProblem) -> Error {
    invalid(call, Problem::Foo(problem))
}

fn far(&self, at: u64) -> Problem {
    Problem::Foo(FooProblem::Far { at })
}

fn f() {
    problem(CALL, FooProblem::Stopped);
    nil(CALL, Problem::Foo(FooProblem::Gone));
    let stop = |p| invalid(CALL, Problem::Foo(p));
    stop(FooProblem::Both { at: 1 });
    Err(Refused::Stops(TextureProblem::Zero));
    Err(BarProblem::Wide { at: 2 }.into());
    Err(invalid(CALL, self.far(3)));
}
";
    let docs = docs(Path::new("synthetic.rs"), text);
    assert_eq!(docs.len(), 4);
    assert_eq!(
        nil_only(&docs),
        BTreeSet::from([
            ("FooProblem".to_owned(), "Gone".to_owned()),
            ("FooProblem".to_owned(), "Far".to_owned()),
        ])
    );
    let found: BTreeSet<(String, String)> = invalid_paths(text).into_iter().collect();
    let want = [
        ("Problem", "Foo"),
        ("FooProblem", "Stopped"),
        ("FooProblem", "Both"),
        ("TextureProblem", "Zero"),
        ("BarProblem", "Wide"),
        ("FooProblem", "Far"),
    ];
    for (problem, variant) in want {
        assert!(
            found.contains(&(problem.to_owned(), variant.to_owned())),
            "{problem}::{variant} in {found:?}"
        );
    }
    assert!(!found.contains(&("FooProblem".to_owned(), "Gone".to_owned())));
}

/// No doc of the runtime labels a refusal `Invalid` in a clause that says Metal answers nil,
/// unless the clause labels the nil `Nil` too.
#[test]
fn no_refusal_documented_invalid_is_metal_s_nil() {
    let mut wrong = Vec::new();
    for (path, text) in sources() {
        for doc in docs(&path, &text) {
            for clause in clauses(&doc.text) {
                if says_nil(clause) && labels_invalid(clause) && !labels_nil(clause) {
                    wrong.push(format!("{}:{}: {clause}", doc.path.display(), doc.line));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Every problem a call of the runtime refuses `Invalid` is one Metal stops: none is a variant
/// documented as Metal's nil alone.
#[test]
fn no_invalid_refusal_is_a_problem_metal_answers_nil_for() {
    let sources = sources();
    let all: Vec<Doc> = sources
        .iter()
        .flat_map(|(path, text)| docs(path, text))
        .collect();
    let nil_only = nil_only(&all);
    // The walk sees the nil problems the runtime documents.
    for (problem, variant) in [
        ("Problem", "NotShareable"),
        ("TextureProblem", "NoTextureOfFormat"),
        ("TextureProblem", "Unspecialized"),
        ("ViewProblem", "NotAFormat"),
        ("RenderPassProblem", "NoTargets"),
        ("RenderPassProblem", "TileSize"),
        ("ArgumentEncoderProblem", "NoNestedArgumentBuffer"),
        ("BufferTextureProblem", "PastAllocation"),
        ("SparseProblem", "TextureType"),
        ("Problem", "BufferLength"),
        ("HeapProblem", "Size"),
        ("HeapProblem", "Offset"),
        ("HeapProblem", "PastEnd"),
    ] {
        assert!(
            nil_only.contains(&(problem.to_owned(), variant.to_owned())),
            "{problem}::{variant} is not documented as Metal's nil: {nil_only:?}"
        );
    }
    let mut wrong = Vec::new();
    for (path, text) in &sources {
        for refused in invalid_paths(text) {
            if nil_only.contains(&refused) {
                wrong.push(format!("{}: {}::{}", path.display(), refused.0, refused.1));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "refused Invalid, documented as nil: {wrong:#?}"
    );
}
