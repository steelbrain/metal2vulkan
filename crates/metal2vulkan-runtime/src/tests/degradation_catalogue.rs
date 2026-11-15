//! docs/API.md's degrade-notice catalogue against the declarations in `degradation.rs`. The
//! catalogue is prose written for a consumer, so it recasts a kind's rustdoc rather than copying
//! it, and most of a declaration's words are no contract: its trigger names the file that lowers
//! the degradation, where the entry names none, and a sentence of either may say in its own terms
//! what the other says in its. Three things the declarations state exactly, which these tests
//! keep:
//!
//! - **The item.** An entry opens with the Rust item it documents, so an entry cannot drift onto
//!   another kind: a runtime kind's [`Degradation`] variant, a note's kind the `PrecisionNote`
//!   variant it announces.
//! - **The cost.** A runtime kind's entry gives, after `Cost: `, the words
//!   [`Degradation::cost`] announces, before whatever more it says. The announced text is a
//!   contract (the stderr line and `DebugMessage::text` carry it), so the catalogue states it
//!   rather than paraphrasing it; an entry may go on past it, which 11 do today. Every note's
//!   kind announces one generic cost, which its entry states in the note's own terms, so a note's
//!   entry is held to its item alone.
//! - **The identifiers of the trigger.** An entry names every
//!   [`metal2vulkan::device::DriverFact`] its declaration's trigger names, and every Metal pixel
//!   format the trigger names in a code span. A consumer predicts a notice from its device's
//!   facts and the format it binds, so dropping one of those names leaves a reader unable to tell
//!   whether the notice reaches them, where dropping a clause of prose does not. Only the
//!   identifiers are held, never the sentence around them; 42 of the kinds are declared with one
//!   today.
//!
//! No rule holds a declaration's prose. Containment of a trigger's first sentence, even with the
//! source-file parentheticals stripped, fails most of the kinds: the catalogue recasts a trigger
//! rather than quoting it, so such a check would read a difference in voice as a defect.

use crate::Degradation;
use std::collections::BTreeMap;

/// docs/API.md's catalogue, each code's entry as one line of text: the words after the code and
/// its category (`- \`<code>\` (<category>). `), with every continuation line.
fn catalogue() -> BTreeMap<String, String> {
    let path = metal2vulkan_devtools::workspace_dir().join("docs/API.md");
    let guide = std::fs::read_to_string(path).unwrap();
    let mut entries = BTreeMap::new();
    let mut open: Option<(String, String)> = None;
    for line in guide.lines() {
        if let Some(rest) = line.strip_prefix("- `metal2vulkan-degrade-") {
            if let Some((code, text)) = open.take() {
                entries.insert(code, text);
            }
            let (kind, after) = rest
                .split_once('`')
                .expect("an entry's code closes its span");
            let (_, text) = after
                .split_once(')')
                .expect("an entry's category closes its bracket");
            let text = text.trim_start_matches('.').trim_start();
            open = Some((format!("metal2vulkan-degrade-{kind}"), text.to_owned()));
        } else if open.is_some() {
            match line
                .strip_prefix("  ")
                .filter(|more| !more.trim().is_empty())
            {
                Some(more) => {
                    let (_, text) = open.as_mut().expect("an entry is open");
                    text.push(' ');
                    text.push_str(more);
                }
                None => {
                    let (code, text) = open.take().expect("an entry is open");
                    entries.insert(code, text);
                }
            }
        }
    }
    if let Some((code, text)) = open {
        entries.insert(code, text);
    }
    entries
}

/// `text`'s words, so that the catalogue's markdown and an announced string compare: every run of
/// whitespace one space (the catalogue wraps its lines, a cost string its literal) and backticks
/// dropped, so a code span and the plain identifier a notice prints read alike.
fn words(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .replace('`', "")
}

/// The words `entry` states as its cost, after `Cost: `, or `None` where it states none.
fn stated_cost(entry: &str) -> Option<String> {
    let entry = words(entry);
    let (_, cost) = entry.split_once("Cost: ")?;
    Some(cost.to_owned())
}

/// Whether `entry` gives `cost` as the cost it states: those words, and then whatever more the
/// entry says.
fn states_cost(entry: &str, cost: &str) -> bool {
    stated_cost(entry).is_some_and(|stated| stated.starts_with(&words(cost)))
}

/// Every entry opens with the Rust item whose kind it documents.
#[test]
fn an_entry_opens_with_its_kind_s_item() {
    let catalogue = catalogue();
    for &kind in Degradation::ALL {
        let entry = catalogue
            .get(kind.code())
            .unwrap_or_else(|| panic!("{kind:?} ({}) has no entry", kind.code()));
        let item = match kind.precision_note() {
            Some(note) => format!("`PrecisionNote::{note:?}`:"),
            None => format!("`{kind:?}`:"),
        };
        assert!(
            entry.trim_start().starts_with(&item),
            "{}'s entry opens with {:?}, not {item}",
            kind.code(),
            entry
                .trim_start()
                .chars()
                .take(item.len())
                .collect::<String>(),
        );
    }
}

/// A runtime kind's entry states the cost the kind announces, word for word, before whatever more
/// it says.
#[test]
fn a_runtime_kind_s_entry_states_the_cost_it_announces() {
    let catalogue = catalogue();
    for &kind in Degradation::ALL {
        if kind.precision_note().is_some() {
            continue;
        }
        let entry = &catalogue[kind.code()];
        let Some(stated) = stated_cost(entry) else {
            panic!("{}'s entry states no cost", kind.code());
        };
        let cost = words(kind.cost());
        assert!(
            stated.starts_with(&cost),
            "{}'s entry states a cost the kind does not announce.\n\
             announced: {cost}\n\
             stated:    {stated}\n\
             they part at: {}",
            kind.code(),
            parting(&cost, &stated),
        );
    }
}

/// Where `announced` and `stated` first differ, with what follows in each.
fn parting(announced: &str, stated: &str) -> String {
    let at = announced
        .char_indices()
        .zip(stated.char_indices())
        .find(|((_, a), (_, b))| a != b)
        .map_or(announced.len().min(stated.len()), |((i, _), _)| i);
    format!(
        "{:?} then announced {:?}, stated {:?}",
        &announced[..at],
        &announced[at..],
        stated.get(at..).unwrap_or_default(),
    )
}

/// The cost check reads a difference rather than only a resemblance: an entry that paraphrases a
/// cost, shortens it, drops a clause from its middle or states none at all fails it. These stand
/// in for an entry whose words drift from the kind's, which is how the check is meant to fail.
#[test]
fn an_entry_that_only_resembles_the_cost_does_not_state_it() {
    let cost = "a scratch image per pass, kept while the pass runs";
    // The words, alone or before more of the entry's own.
    assert!(states_cost(
        "`K`: A trigger. Cost: a scratch image per pass, kept while the pass runs.",
        cost
    ));
    assert!(states_cost(
        "`K`: A trigger. Cost: a scratch image per pass, kept while the pass runs; the copy in is \
         cut to the render area.",
        cost
    ));
    // Wrapped over the catalogue's lines, and with a code span where the cost prints plainly.
    assert!(states_cost(
        "Cost: a scratch   image per\npass, kept while the pass runs.",
        cost
    ));
    assert!(states_cost(
        "Cost: a scratch image per `pass`, kept while the pass runs.",
        "a scratch image per pass, kept while the pass runs"
    ));
    // A paraphrase, a shortening, a clause dropped from the middle, another kind's cost, and no
    // cost at all.
    assert!(!states_cost(
        "Cost: a scratch image for each pass, kept while the pass runs.",
        cost
    ));
    assert!(!states_cost("Cost: a scratch image per pass.", cost));
    assert!(!states_cost(
        "Cost: a scratch image per pass, kept while it runs.",
        cost
    ));
    assert!(!states_cost(
        "Cost: its texels are copied into an image.",
        cost
    ));
    assert!(!states_cost(
        "`K`: A trigger: a scratch image per pass, kept while the pass runs.",
        cost
    ));
}

/// Each kind's declaration in `degradation.rs`, keyed by its code: the rustdoc of its line in
/// `degradations!`, every line of it as one line of text.
fn declarations() -> BTreeMap<String, String> {
    let path = metal2vulkan_devtools::workspace_dir()
        .join("crates/metal2vulkan-runtime/src/degradation.rs");
    let source = std::fs::read_to_string(path).unwrap();
    let (_, invocation) = source
        .split_once("\ndegradations! {")
        .expect("`degradations!` is invoked once");
    let mut declarations = BTreeMap::new();
    let mut doc: Vec<&str> = Vec::new();
    for line in invocation.lines() {
        let line = line.trim();
        match line.strip_prefix("///") {
            Some(more) => doc.push(more.trim()),
            None => {
                if let Some(code) = declared_code(line) {
                    declarations.insert(code, doc.join(" "));
                    doc.clear();
                }
            }
        }
    }
    declarations
}

/// The code `line` declares, for a line of `degradations!` that declares a kind
/// (`Kind = "code", Category, "cost";` or a note's `Note = "code";`), else `None`.
fn declared_code(line: &str) -> Option<String> {
    let (item, rest) = line.split_once(" = \"")?;
    let named = item.starts_with(|first: char| first.is_ascii_uppercase())
        && item.chars().all(|c| c.is_ascii_alphanumeric());
    let (code, _) = rest.split_once('"')?;
    named.then(|| format!("metal2vulkan-degrade-{code}"))
}

/// Every Metal pixel format's name, as a declaration and an entry spell it (`PixelFormat::name`
/// without its `MTLPixelFormat` prefix).
fn metal_formats() -> Vec<&'static str> {
    metal2vulkan::metal::PixelFormat::ALL
        .iter()
        .map(|&pf| {
            pf.name()
                .strip_prefix("MTLPixelFormat")
                .expect("a pixel format's Metal name names the type")
        })
        .collect()
}

/// The identifiers `declaration` names that a consumer predicts the kind by: each
/// [`metal2vulkan::device::DriverFact`] it names, and each Metal pixel format it names in a code
/// span (a format named in prose alone is a word of the prose, which the entry recasts).
fn named_identifiers(declaration: &str) -> Vec<String> {
    let facts = declaration
        .match_indices("DriverFact::")
        .map(|(at, marker)| {
            let rest = &declaration[at + marker.len()..];
            let end = rest
                .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                .unwrap_or(rest.len());
            format!("DriverFact::{}", &rest[..end])
        });
    let formats = metal_formats()
        .into_iter()
        .filter(|name| declaration.contains(&format!("`{name}`")))
        .map(str::to_owned);
    let mut named: Vec<String> = facts.chain(formats).collect();
    named.sort();
    named.dedup();
    named.retain(|name| !name.ends_with("::"));
    named
}

/// Whether `text` names `identifier`: the identifier as a whole word, not inside a longer one, so
/// that `A8Unorm` is not read in `BGRA8Unorm` nor `R8Unorm` in `R8Unorm_sRGB`. A `::` either side
/// is no part of it, so an entry that writes a `DriverFact` as a path from the crate root names it.
fn names(text: &str, identifier: &str) -> bool {
    let part = |c: char| c.is_ascii_alphanumeric() || c == '_';
    text.match_indices(identifier).any(|(at, _)| {
        let before = text[..at].chars().next_back();
        let after = text[at + identifier.len()..].chars().next();
        !before.is_some_and(part) && !after.is_some_and(part)
    })
}

/// An entry names every `DriverFact` and every Metal pixel format its kind's declaration names.
/// These are identifiers, not prose: a consumer predicts a notice from the device's facts and the
/// format it binds, so an entry that drops one leaves a reader unable to tell whether the notice
/// reaches them. Only the identifiers are held, never the sentence around them, which the
/// catalogue recasts in its own voice.
#[test]
fn an_entry_names_the_identifiers_its_declaration_names() {
    let catalogue = catalogue();
    let declarations = declarations();
    assert_eq!(
        declarations.len(),
        Degradation::ALL.len(),
        "`degradations!` declares one kind per `Degradation::ALL` kind; the declarations read: {:?}",
        declarations.keys().collect::<Vec<_>>(),
    );
    let mut covered = 0;
    let mut facts = 0;
    let mut formats = 0;
    let mut dropped: Vec<String> = Vec::new();
    for &kind in Degradation::ALL {
        let declaration = declarations
            .get(kind.code())
            .unwrap_or_else(|| panic!("{kind:?} ({}) has no declaration", kind.code()));
        let named = named_identifiers(declaration);
        if named.is_empty() {
            continue;
        }
        covered += 1;
        facts += named
            .iter()
            .filter(|n| n.starts_with("DriverFact::"))
            .count();
        formats += named
            .iter()
            .filter(|n| !n.starts_with("DriverFact::"))
            .count();
        let entry = words(&catalogue[kind.code()]);
        let gone: Vec<&String> = named.iter().filter(|name| !names(&entry, name)).collect();
        if !gone.is_empty() {
            dropped.push(format!("{}: {gone:?} of {named:?}", kind.code()));
        }
    }
    assert!(
        covered > 0 && facts > 0 && formats > 0,
        "the check reads no identifier out of the declarations: \
         {covered} kinds, {facts} driver facts, {formats} formats"
    );
    assert!(
        dropped.is_empty(),
        "{} of the {covered} entries whose declaration names an identifier do not name one of \
         them:\n{}",
        dropped.len(),
        dropped.join("\n"),
    );
}

/// The identifier check reads the declarations and the entries rather than passing on their shape:
/// a declaration that names a fact or a format in a code span is read, one that names a format in
/// prose alone is not, and an entry that drops a named identifier, or holds it inside a longer
/// one, fails. These stand in for an entry that leaves out a fact or a format, which is how the
/// check is meant to fail.
#[test]
fn an_entry_that_drops_a_named_identifier_does_not_name_it() {
    // A declaration's identifiers: a `DriverFact` path however it is written, and a format in a
    // code span; a format in prose alone, and a bare `DriverFact`, are no identifiers of it.
    assert_eq!(
        named_identifiers(
            "Not where [`metal2vulkan::device::DriverFact::TileStoreIsMetals`] holds, for a \
             `BGRA8Unorm` or `R8Unorm_sRGB` texture. A DriverFact decides. An RGBA32Float one is \
             exact."
        ),
        [
            "BGRA8Unorm".to_owned(),
            "DriverFact::TileStoreIsMetals".to_owned(),
            "R8Unorm_sRGB".to_owned(),
        ],
    );
    assert!(named_identifiers("A texture over a buffer, with no fact or format.").is_empty());
    // An entry names an identifier in a code span, which `words` leaves bare, or plainly.
    assert!(names(
        "a DriverFact::TileStoreIsMetals device",
        "DriverFact::TileStoreIsMetals"
    ));
    assert!(names("of BGRA8Unorm, and more", "BGRA8Unorm"));
    assert!(names("of a BGRA8Unorm.", "BGRA8Unorm"));
    // A path from the crate root names the fact its last segments are.
    assert!(names(
        "a metal2vulkan::device::DriverFact::TileStoreIsMetals device",
        "DriverFact::TileStoreIsMetals"
    ));
    // One dropped, one inside a longer identifier, one that is a longer format's head, and one
    // whose path is another's.
    assert!(!names(
        "the driver stores it as Metal's own DontCare",
        "DriverFact::DontCareStoreIsMetals"
    ));
    assert!(!names("of A8Unorm and BGRA8Unorm", "RA8Unorm"));
    assert!(!names("of R8Unorm_sRGB alone", "R8Unorm"));
    assert!(!names(
        "DriverFact::DontCareStoreIsOneOfMetals holds",
        "DriverFact::DontCareStoreIsMetals"
    ));
    // The declarations parse: every kind's line is read, and nothing else is.
    let declarations = declarations();
    assert_eq!(declarations.len(), Degradation::ALL.len());
    assert!(declarations[Degradation::ShadowedTexture.code()].contains("DriverFact"));
}
