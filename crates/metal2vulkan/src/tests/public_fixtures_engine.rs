//! Engine support inventory over the public fixtures, one column per committed profile.
//!
//! Every `.ll` fixture at the top level of `validation/fixtures/public/` and in its `engine/`
//! directory is assembled to bitcode (`llvm-as`; a typed-pointer `.air.ll` through `xcrun
//! metal-as`), every entry point its ABI declares is compiled by the engine for every committed
//! profile under the default policy with no function constants and default lowering inputs, and
//! the outcome is recorded:
//!
//! - `ok`: compiled, linked, relocated, and passed L3 (requirements within the profile and
//!   `spirv-val` under the profile's environment; `G-val`);
//! - `Unsupported{feature; needs}`, `NeedsInput{input}`, `InvalidInput{kind}`,
//!   `LimitExceeded{limit}`: the typed refusal, without its input location.
//!
//! A fixture whose ABI does not decode has one row with entry `-`; a fixture the installed
//! assembler rejects has one row with entry `-` and `unassembled` in every column. Everything else
//! (an internal error, a transient budget or cancellation, or an `ok` artifact that fails L3) is a
//! translator bug and fails the test.
//!
//! The table must equal `public_fixture_expectations_engine.tsv` exactly. A change from `ok`
//! to a refusal needs a reviewed update whose commit states the reason. On a mismatch the test
//! names every difference: a cell that moved under a row the table already has, by its fixture,
//! entry and profile and from what to what, apart from a row that is new or gone. The two classes
//! answer different questions — a moved cell is a fixture the engine now compiles or refuses
//! differently, a new or gone row is a fixture or an entry point that came or went — so a review
//! attributes each to the commit that explains it. Rows of a fixture whose assembler is not
//! installed (`metal-as` off macOS) are skipped visibly and taken from the expectations.
//!
//! The review is the first step and the rewrite is the second: once every named difference is
//! accounted for, an ignored bless rewrites the table from what the build observes, as
//! `golden_hashes` does for the golden tables. It names the differences again as it takes
//! them, and needs every assembler, so it runs on macOS:
//!
//! ```text
//! cargo nextest run -p metal2vulkan --all-features --run-ignored ignored-only -E 'test(bless_engine_inventory)'
//! ```
//!
//! Writing the table is the bless's, not a paste-ready block's on a failure: nextest captures a
//! test's output through a pipe that holds no tab character, so a printed block arrives with its
//! columns collapsed and cannot be pasted back. Every message a part prints is readable under that
//! capture, which is why the differences are named and the rows are not printed.
//!
//! The check is split so nextest runs it in parallel, and so the fixtures that need macOS are
//! apart from those that do not (one test took 136-353 s on the Mac): the table equals the
//! expectations exactly when its header and its fixtures' order are the expectations'
//! ([`inventory_lists_every_fixture_in_order`]) and each fixture's rows are its expected rows (the
//! `llvm_as` parts: the `llvm-as` fixtures; the `metal_as` parts: the typed-pointer `.air.ll`
//! ones, which skip visibly off macOS). Each part takes the fixtures whose position among its
//! assembler's fixtures is its number modulo the part count.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::device::DeviceProfile;
use crate::error::{Error, Needs};
use crate::pipeline::testing::{
    compile, engine_dir, fixtures_dir, link_and_validate, llvm_assemble, metal_assemble, meter,
    AssembleError,
};
use crate::spirv::testing::committed_profiles;
use crate::tests::golden_hashes::parallel;

/// The committed table, relative to the crate root; [`EXPECTED`] is its text.
const TABLE: &str = "tests/public_fixture_expectations_engine.tsv";

const EXPECTED: &str = include_str!("../../tests/public_fixture_expectations_engine.tsv");

/// The command that rewrites the expectations from what the build observes, once the named
/// differences have been reviewed ([`bless_engine_inventory`]).
const BLESS: &str = "cargo nextest run -p metal2vulkan --all-features --run-ignored ignored-only \
                     -E 'test(bless_engine_inventory)'";

/// `.ll` files directly in `dir`, sorted by name.
fn ll_files(dir: &Path) -> Vec<PathBuf> {
    let mut paths = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .map(|entry| entry.expect("read fixture directory entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "ll"))
        .collect::<Vec<_>>();
    paths.sort();
    paths
}

/// Every fixture as (inventory name, path).
fn fixtures() -> Vec<(String, PathBuf)> {
    let mut out = Vec::new();
    for (prefix, dir) in [("", fixtures_dir()), ("engine/", engine_dir())] {
        for p in ll_files(&dir) {
            let name = p.file_name().unwrap().to_str().unwrap();
            out.push((format!("{prefix}{name}"), p));
        }
    }
    out
}

/// The bitcode of a fixture: typed-pointer `.air.ll` through `metal-as`, the rest through
/// `llvm-as`.
fn assemble(path: &Path) -> Result<Vec<u8>, AssembleError> {
    if is_metal_as(path) {
        metal_assemble(path)
    } else {
        llvm_assemble(path)
    }
}

/// The inventory cell of a refusal; `None` for an outcome that is a translator bug.
pub(super) fn refusal(e: &Error) -> Option<String> {
    Some(match e {
        Error::Unsupported(u) => {
            let needs = match &u.needs {
                Needs::AnyOf(alts) => format!("AnyOf{alts:?}"),
                Needs::NotImplemented => "NotImplemented".to_owned(),
                Needs::NotRepresentable => "NotRepresentable".to_owned(),
                other => format!("{other:?}"),
            };
            format!("Unsupported{{{:?}; {needs}}}", u.feature)
        }
        Error::NeedsInput(i) => format!("NeedsInput{{{i:?}}}"),
        Error::InvalidInput(i) => format!("InvalidInput{{{:?}}}", i.kind),
        Error::LimitExceeded(l) => format!("LimitExceeded{{{l:?}}}"),
        _ => return None,
    })
}

/// The cell of `entry` of `bitcode` on profile `p`.
fn cell(bitcode: &[u8], fixture: &str, entry: &str, stem: &str, p: &DeviceProfile) -> String {
    match compile(bitcode, entry, p) {
        Ok(a) => {
            link_and_validate(&a, p)
                .unwrap_or_else(|why| panic!("{fixture} {entry} on {stem}: L3 failed: {why}"));
            "ok".to_owned()
        }
        Err(e) => refusal(&e)
            .unwrap_or_else(|| panic!("{fixture} {entry} on {stem}: translator bug: {e}")),
    }
}

/// The expected rows of each fixture, keyed by fixture name.
fn expected_rows() -> BTreeMap<&'static str, Vec<&'static str>> {
    let mut rows: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for line in EXPECTED.lines().skip(1) {
        let fixture = line.split('\t').next().unwrap_or_default();
        rows.entry(fixture).or_default().push(line);
    }
    rows
}

/// The inventory's header line: `fixture`, `entry`, and each committed profile's stem.
fn header(profiles: &[(String, DeviceProfile)]) -> String {
    let mut line = String::from("fixture\tentry");
    for (stem, _) in profiles {
        line.push('\t');
        line.push_str(stem);
    }
    line
}

/// One inventory line: its fixture, its entry, and one cell per committed profile, with its
/// newline. Every row the inventory writes, observed or blessed, goes through here, so the
/// separator [`index_rows`] splits on is written in one place.
fn row(fixture: &str, entry: &str, cells: impl IntoIterator<Item = String>) -> String {
    let mut line = format!("{fixture}\t{entry}");
    for cell in cells {
        line.push('\t');
        line.push_str(&cell);
    }
    line.push('\n');
    line
}

/// The table as the expectations hold it: [`header`], then one block of rows per fixture in
/// [`fixtures`]' order. The bless writes this and the committed file is this, so the parts compare
/// a file one function shapes.
fn render_table(profiles: &[(String, DeviceProfile)], blocks: &[String]) -> String {
    let mut s = header(profiles);
    s.push('\n');
    for rows in blocks {
        s.push_str(rows);
    }
    s
}

/// The inventory rows of fixture `name` (each line with its newline), and whether they were taken
/// from the expectations because its assembler is not installed (a visible skip).
fn fixture_rows(
    name: &str,
    path: &Path,
    profiles: &[(String, DeviceProfile)],
    expected_rows: &BTreeMap<&str, Vec<&str>>,
) -> (String, bool) {
    let mut observed = String::new();
    let bitcode = match assemble(path) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED {name}: {why}; rows taken from the expectations");
            for line in expected_rows.get(name).into_iter().flatten() {
                observed.push_str(line);
                observed.push('\n');
            }
            return (observed, true);
        }
        Err(AssembleError::Rejected(why)) => {
            eprintln!("{name}: assembler rejected the fixture: {why}");
            let cells = profiles.iter().map(|_| "unassembled".to_owned());
            observed.push_str(&row(name, "-", cells));
            return (observed, false);
        }
    };
    let entries = match crate::air::decode(&bitcode, &mut meter()).map_err(Error::from) {
        Ok(d) => d
            .abi
            .entries
            .iter()
            .map(|e| e.name.to_string())
            .collect::<Vec<_>>(),
        Err(e) => {
            eprintln!("{name}: decode: {e}");
            let c = refusal(&e).unwrap_or_else(|| panic!("{name}: decode: translator bug: {e}"));
            let cells = profiles.iter().map(|_| c.clone());
            observed.push_str(&row(name, "-", cells));
            return (observed, false);
        }
    };
    assert!(!entries.is_empty(), "{name}: no entry points");
    for entry in entries {
        let cells = profiles
            .iter()
            .map(|(stem, p)| cell(&bitcode, name, &entry, stem, p));
        observed.push_str(&row(name, &entry, cells));
    }
    (observed, false)
}

/// The expectations' header and the order of their fixtures are the inventory's: its first line
/// is [`header`], and its rows are one block per fixture, in [`fixtures`]' order. With each
/// fixture's rows checked by a part, the observed table equals the expectations exactly.
#[test]
fn inventory_lists_every_fixture_in_order() {
    let profiles = committed_profiles().expect("committed profiles");
    let mut lines = EXPECTED.lines();
    assert_eq!(
        lines.next(),
        Some(header(&profiles).as_str()),
        "the header of tests/public_fixture_expectations_engine.tsv is not the committed \
         profiles' stems"
    );
    let mut blocks: Vec<&str> = Vec::new();
    for line in lines {
        let fixture = line.split('\t').next().unwrap_or_default();
        if blocks.last() != Some(&fixture) {
            assert!(
                !blocks.contains(&fixture),
                "{fixture}: its rows in tests/public_fixture_expectations_engine.tsv are not \
                 one block"
            );
            blocks.push(fixture);
        }
    }
    let names = fixtures().into_iter().map(|(n, _)| n).collect::<Vec<_>>();
    if blocks != names {
        let first = blocks
            .iter()
            .zip(&names)
            .position(|(b, n)| b != n)
            .unwrap_or(blocks.len().min(names.len()));
        panic!(
            "the fixtures of tests/public_fixture_expectations_engine.tsv are not the public \
             fixtures in order ({} listed vs {} fixtures); first difference at fixture {}: \
             expected {:?}, found {:?}",
            blocks.len(),
            names.len(),
            first + 1,
            blocks.get(first),
            names.get(first)
        );
    }
}

/// Part `part` of `of` of the fixtures `metal_as` selects (the `.air.ll` ones, or the others):
/// each fixture's observed rows must be its expected rows.
fn check_part(metal_as: bool, part: usize, of: usize) {
    let profiles = committed_profiles().expect("committed profiles");
    let expected_rows = expected_rows();
    let mut skipped = 0usize;
    let mut observed = String::new();
    let mut expected = String::new();
    let selected = fixtures()
        .into_iter()
        .filter(|(_, path)| is_metal_as(path) == metal_as)
        .enumerate()
        .filter(|(i, _)| i % of == part)
        .map(|(_, f)| f);
    for (name, path) in selected {
        let (rows, skip) = fixture_rows(&name, &path, &profiles, &expected_rows);
        skipped += usize::from(skip);
        observed.push_str(&rows);
        for line in expected_rows.get(name.as_str()).into_iter().flatten() {
            expected.push_str(line);
            expected.push('\n');
        }
    }
    if skipped > 0 {
        eprintln!("SKIPPED {skipped} fixture(s) whose assembler is not installed");
    }
    if observed != expected {
        let d = named_differences(&expected, &observed, &profiles);
        panic!(
            "the engine inventory changed in this part: {} cell(s) moved under a row the table \
             already has, and {} row(s) are new or gone ({} observed vs {} expected rows). \
             Review each one, then take them into {TABLE} with\n  {BLESS}\n{}{}",
            d.cells.len(),
            d.rows.len(),
            observed.lines().count(),
            expected.lines().count(),
            shown(&d.cells),
            shown(&d.rows),
        );
    }
}

/// Every difference between `expected` and `observed`, printed one per line before it is returned.
/// A line carries no tab (`fixture | entry | profile: was -> now`), so a reader gets all of it from
/// nextest's captured output, which holds no tab character.
fn named_differences(expected: &str, observed: &str, profiles: &[(String, DeviceProfile)]) -> Diff {
    let stems = profiles
        .iter()
        .map(|(stem, _)| stem.clone())
        .collect::<Vec<_>>();
    let d = diff_rows(expected, observed, &stems);
    eprintln!("---- every difference ----");
    for line in d.cells.iter().chain(&d.rows) {
        eprintln!("  {line}");
    }
    eprintln!("---- end ----");
    d
}

/// The differences between a part's expected and observed rows, named one by one.
#[derive(Default)]
struct Diff {
    /// `fixture | entry | profile: was -> now` for each cell of a row both tables hold, and
    /// `new cell`/`gone cell` where the row's width moved (a committed profile came or went).
    cells: Vec<String>,
    /// `new row`/`gone row` for each (fixture, entry) only one table holds.
    rows: Vec<String>,
}

/// At most [`SHOWN`] of `lines`, indented, with a count of the rest.
fn shown(lines: &[String]) -> String {
    let mut s = String::new();
    for l in lines.iter().take(SHOWN) {
        s.push_str("  ");
        s.push_str(l);
        s.push('\n');
    }
    if lines.len() > SHOWN {
        s.push_str(&format!("  ... and {} more\n", lines.len() - SHOWN));
    }
    s
}

/// How many differences of each class a panic names; the test prints every one before it.
const SHOWN: usize = 40;

/// A row's key — its fixture and entry — and its cells, one per committed profile.
fn split_row(line: &str) -> ((&str, &str), Vec<&str>) {
    let mut f = line.split('\t');
    let fixture = f.next().unwrap_or_default();
    let entry = f.next().unwrap_or_default();
    ((fixture, entry), f.collect())
}

/// The rows of a table block, keyed by fixture and entry. A fixture's entries are its ABI's, so
/// the key is unique; a repeat would hide a row, so it keeps its own key by its position.
fn index_rows(text: &str) -> BTreeMap<(&str, &str), Vec<&str>> {
    let mut rows: BTreeMap<(&str, &str), Vec<&str>> = BTreeMap::new();
    for (i, line) in text.lines().enumerate() {
        let (key, cells) = split_row(line);
        if rows.insert(key, cells).is_some() {
            panic!(
                "{}: two rows of entry {}, the second at line {}",
                key.0,
                key.1,
                i + 1
            );
        }
    }
    rows
}

/// Each difference between `expected` and `observed`, as a cell that moved under a row both hold
/// or a row only one holds. A moved cell names its profile (`stems`, in column order), what the
/// table has and what the engine now produces, so a reviewer can attribute it without diffing the
/// printed block by hand.
fn diff_rows(expected: &str, observed: &str, stems: &[String]) -> Diff {
    let (exp, obs) = (index_rows(expected), index_rows(observed));
    let stem = |i: usize| stems.get(i).map_or("?", String::as_str);
    let mut d = Diff::default();
    for (&(fixture, entry), cells) in &obs {
        let at = |i: usize| format!("{fixture} | {entry} | {}", stem(i));
        let Some(was) = exp.get(&(fixture, entry)) else {
            d.rows
                .push(format!("new row: {fixture} | {entry}: {}", cells.join(" ")));
            continue;
        };
        for (i, cell) in cells.iter().enumerate() {
            match was.get(i) {
                Some(b) if b == cell => {}
                Some(b) => d.cells.push(format!("{}: {b} -> {cell}", at(i))),
                None => d.cells.push(format!("new cell: {}: {cell}", at(i))),
            }
        }
        for (i, gone) in was.iter().enumerate().skip(cells.len()) {
            d.cells.push(format!("gone cell: {}: {gone}", at(i)));
        }
    }
    for (&(fixture, entry), cells) in &exp {
        if !obs.contains_key(&(fixture, entry)) {
            d.rows.push(format!(
                "gone row: {fixture} | {entry}: {}",
                cells.join(" ")
            ));
        }
    }
    d
}

/// Whether `path` is a typed-pointer `.air.ll` fixture, which `metal-as` assembles ([`assemble`]).
fn is_metal_as(path: &Path) -> bool {
    path.to_str().is_some_and(|s| s.ends_with(".air.ll"))
}

/// One test per part, so nextest runs the parts at once.
macro_rules! parts {
    ($metal_as:expr, $of:expr; $($test:ident = $part:expr),* $(,)?) => {
        $(
            #[test]
            fn $test() {
                super::check_part($metal_as, $part, $of);
            }
        )*
    };
}

/// The fixtures `llvm-as` assembles, on every host.
mod llvm_as {
    parts!(false, 8; part_1_of_8 = 0, part_2_of_8 = 1, part_3_of_8 = 2, part_4_of_8 = 3,
        part_5_of_8 = 4, part_6_of_8 = 5, part_7_of_8 = 6, part_8_of_8 = 7);
}

/// The typed-pointer `.air.ll` fixtures `metal-as` assembles: checked on macOS, skipped visibly
/// elsewhere.
mod metal_as {
    parts!(true, 4; part_1_of_4 = 0, part_2_of_4 = 1, part_3_of_4 = 2, part_4_of_4 = 3);
}

/// Rewrite `tests/public_fixture_expectations_engine.tsv` from what this build observes: `BLESS`
/// above.
///
/// It is ignored and named, so taking a change is a second, deliberate step after the parts have
/// named every difference — the parts never write the table, and nothing blesses it on a plain
/// run. It names the differences again as it takes them, and refuses without every assembler,
/// whose fixtures' rows would otherwise be written back from the expectations unobserved.
///
/// It observes every fixture at once, as the golden tables' bless does, rather than taking the
/// parts in turn: one pass over all of them stays well inside nextest's slow-test limit, which a
/// serial pass over twelve parts' worth of fixtures would approach.
#[test]
#[ignore]
fn bless_engine_inventory() {
    let profiles = committed_profiles().expect("committed profiles");
    let expected_rows = expected_rows();
    let fixtures = fixtures();
    let observations = parallel(fixtures.len(), |i| {
        let (name, path) = &fixtures[i];
        fixture_rows(name, path, &profiles, &expected_rows)
    });
    let skipped = fixtures
        .iter()
        .zip(&observations)
        .filter(|(_, (_, skip))| *skip)
        .map(|((name, _), _)| name.as_str())
        .collect::<Vec<_>>();
    assert!(
        skipped.is_empty(),
        "refusing to bless without every assembler (skipped: {skipped:?}); bless on macOS"
    );
    let blocks = observations
        .into_iter()
        .map(|(rows, _)| rows)
        .collect::<Vec<_>>();
    let observed = blocks.concat();
    let expected = EXPECTED.split_once('\n').map_or("", |(_, body)| body);
    let d = named_differences(expected, &observed, &profiles);
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(TABLE);
    std::fs::write(&path, render_table(&profiles, &blocks)).expect("write the expectations");
    eprintln!(
        "{TABLE}: blessed {} rows x {} profiles, taking {} moved cell(s) and {} new or gone row(s)",
        observed.lines().count(),
        profiles.len(),
        d.cells.len(),
        d.rows.len(),
    );
}

/// What the bless writes is the file the parts read: [`render_table`] puts the committed profiles'
/// [`header`] first, keeps each block's rows together in the order it was given them, and separates
/// a row's fixture, entry and cells by the tab [`index_rows`] splits on. Re-rendering the committed
/// table's own rows reproduces it byte for byte, so a bless rewrites it in place without reshaping
/// it.
///
/// A printout cannot stand in for this: nextest's captured output holds no tab character, so a row
/// written with a separator chosen to survive capture would not read back here.
#[test]
fn the_blessed_table_is_the_file_the_parts_read() {
    let profiles = committed_profiles().expect("committed profiles");
    let cells = |c: &str| profiles.iter().map(|_| c.to_owned()).collect::<Vec<_>>();
    let blocks = [
        format!(
            "{}{}",
            row("b.ll", "main", cells("ok")),
            row("b.ll", "helper", cells("ok"))
        ),
        row("a.ll", "-", cells("unassembled")),
    ];
    let text = render_table(&profiles, &blocks);

    let (first, body) = text.split_once('\n').expect("a header line");
    assert_eq!(first, header(&profiles));
    assert!(text.ends_with('\n'), "{text:?}");
    let rows = body.lines().collect::<Vec<_>>();
    for r in &rows {
        assert_eq!(
            r.split('\t').count(),
            2 + profiles.len(),
            "a row with one cell per profile: {r:?}"
        );
    }
    assert_eq!(
        rows.iter().map(|r| split_row(r).0).collect::<Vec<_>>(),
        [("b.ll", "main"), ("b.ll", "helper"), ("a.ll", "-")],
        "the blocks' rows, in the order they were given"
    );
    assert_eq!(index_rows(body).len(), 3);

    let mut committed: Vec<String> = Vec::new();
    let mut last = "";
    for line in EXPECTED.lines().skip(1) {
        let fixture = line.split('\t').next().unwrap_or_default();
        if fixture != last {
            committed.push(String::new());
            last = fixture;
        }
        let block = committed.last_mut().expect("a block");
        block.push_str(line);
        block.push('\n');
    }
    assert_eq!(
        render_table(&profiles, &committed),
        EXPECTED,
        "{TABLE} is not what a bless of its own rows writes"
    );
}

/// The mismatch diagnostic on synthetic blocks: it names the row, the entry and the profile of a
/// cell that moved and says from what to what, keeps that apart from a row that is new or gone,
/// and says nothing about the rows that held. Without the naming a reviewer only learned that
/// "the inventory changed".
#[test]
fn the_mismatch_diagnostic_names_each_changed_row() {
    let stems = ["a-gpu".to_owned(), "b-gpu".to_owned()];
    let expected = "f.ll\tmain\tok\tok\nf.ll\thelper\tok\tok\ng.ll\tmain\tok\tok\n";

    // A cell that moved under a row the table already has.
    let observed = "f.ll\tmain\tok\tok\nf.ll\thelper\tok\tUnsupported{Foo; NotImplemented}\n\
                    g.ll\tmain\tok\tok\n";
    let d = diff_rows(expected, observed, &stems);
    assert_eq!(
        d.cells,
        ["f.ll | helper | b-gpu: ok -> Unsupported{Foo; NotImplemented}"]
    );
    assert!(d.rows.is_empty(), "{:?}", d.rows);

    // A row only one block holds is the other class, and the cells of the rows that held are
    // never named.
    let observed = "f.ll\tmain\tok\tok\ng.ll\tmain\tok\tok\ng.ll\tsecond\tok\tok\n";
    let d = diff_rows(expected, observed, &stems);
    assert!(d.cells.is_empty(), "{:?}", d.cells);
    assert_eq!(
        d.rows,
        [
            "new row: g.ll | second: ok ok",
            "gone row: f.ll | helper: ok ok"
        ]
    );

    // Equal blocks differ in nothing.
    let d = diff_rows(expected, expected, &stems);
    assert!(d.cells.is_empty() && d.rows.is_empty());

    // A committed profile that came or went widens or narrows every row; the cell says which
    // column it was, so the reader does not read it as a behaviour change.
    let observed = "f.ll\tmain\tok\tok\tok\nf.ll\thelper\tok\tok\tok\ng.ll\tmain\tok\tok\tok\n";
    let d = diff_rows(
        expected,
        observed,
        &["a-gpu".to_owned(), "b-gpu".to_owned(), "c".to_owned()],
    );
    assert_eq!(
        d.cells,
        [
            "new cell: f.ll | helper | c: ok",
            "new cell: f.ll | main | c: ok",
            "new cell: g.ll | main | c: ok"
        ]
    );
    let d = diff_rows(observed, expected, &stems);
    assert_eq!(
        d.cells,
        [
            "gone cell: f.ll | helper | ?: ok",
            "gone cell: f.ll | main | ?: ok",
            "gone cell: g.ll | main | ?: ok"
        ]
    );
}
