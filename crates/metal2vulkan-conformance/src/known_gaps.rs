//! The known-gaps ledger: our open gaps, each one an open bug with a todo behind it.
//!
//! A gap is never compliant — a refusal is [`Verdict::Refused`] and a difference nobody announced
//! is [`Verdict::Divergent`] — and this module does not change that. What it changes is the **exit
//! status** of a replay run, and only that: a case that holds a known gap visible would otherwise
//! make every run red, and a report that is always red is a report nobody reads, which would cost
//! us the next real gap.
//!
//! # It is an expectation, not an allowlist
//!
//! `AGENTS.md` is explicit that an allowlist resolves no gap, and this must not become one. Two
//! things keep it from becoming one, and both are mechanical rather than a matter of discipline:
//!
//! - **A gap the ledger does not list fails the run** ([`Expectation::Unlisted`]): a new gap
//!   appeared and somebody has to look at it.
//! - **A listed row the run no longer holds also fails the run** ([`Expectation::NoLongerHeld`]),
//!   saying the gap looks fixed and the row must go.
//!
//! That second direction is the point. Closing a gap forces the ledger edit, so a row cannot
//! outlive the bug it stands for. A one-directional check would quietly keep passing after the fix
//! and the file would drift into permission for whatever happened to be in it. It has already
//! earned its keep: a product fix landed in this round and the replay named four stale rows before
//! anybody reasoned about the fix.
//!
//! # Two kinds of gap, one file
//!
//! A row names the verdict it expects ([`Expects`]). **A refusal** is a call plain Metal answered
//! and we did not. **A divergence** is a field that differs with nothing announced to justify it.
//! Both are bugs of ours by the owner's standing rule, both have to stay visible, and both need the
//! same two-direction discipline, so they share one file and one mechanism: `AGENTS.md` forbids
//! parallel string lists, and two ledgers would be exactly that.
//!
//! # Per venue
//!
//! A row may name the venue it is expected on ([`KnownGap::venue`]), because a gap can be one
//! venue's and not another's: a pipeline the runtime refuses on MoltenVK for a structural reason
//! can translate on lavapipe, so a global row would fail the lavapipe run for existing. A row
//! naming no venue applies everywhere, which is the **stronger** claim and not the looser one:
//! the second direction then holds it on every venue the run selected its case on, so widening a
//! row's scope makes it harder to satisfy rather than easier. That is what stops a venue-scoped row
//! from becoming a way to silence a venue.
//!
//! The report keeps printing every one of these rows as `Refused` or `Divergent`, with our answer
//! and Metal's, with Metal's own words where its API validation refuses the call too, and counted
//! in the suite's own numbers. Nothing is laundered.
//!
//! A row is held only against the cases a run selected, so replaying one case does not read every
//! other row as fixed.
//!
//! # Typed reasons
//!
//! Three reasons are the ledger's own vocabulary ([`Reason`]), stated at the head of a row's last
//! field: `refused: not built yet (<lane>)`, a feature the streams use that a lane is building;
//! `refused: no stream uses it yet (<census run>)`, a call replay's census found no stream using;
//! and `refused: the consumer's own (<object>)`, one of a closed list of Metal objects and getters
//! the runtime keeps no counterpart of by design, which its consumer implements
//! ([`ConsumerObject`]). A row takes that reason only where the runtime's public API has no item
//! that could answer it, and leaves it the day the runtime grows one. Every replay prints their
//! counts ([`open_work`]): the rows not built yet are the cutover's remaining work, and the
//! consumer's own are counted on a line of their own, outside it.

use crate::case::Venue;
use crate::replay::Replayed;
use crate::verdict::Verdict;
use std::collections::{BTreeMap, BTreeSet};

/// The committed ledger, compiled in so the check needs no path and no filesystem: a replay run
/// inside a container reads the same rows as one on this Mac.
const LEDGER: &str = include_str!("../known-gaps.tsv");

/// The new runtime's side's ledger ([`crate::rt_side`]), compiled in as [`LEDGER`] is.
const RT_LEDGER: &str = include_str!("../known-gaps-rt.tsv");

/// The venue field of a row that applies to every venue.
pub const EVERY_VENUE: &str = "*";

/// The verdict a row expects at its site.
///
/// Only these two. A case with no Metal answer, a stale recording and a case the venue could not
/// run are never written down: the first two must fail loudly rather than be expected, and the
/// third is not a gap of ours at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Expects {
    /// We returned a typed error where plain Metal answered the call.
    Refused,
    /// A field differs and nothing we announced justifies it.
    Divergent,
}

impl Expects {
    /// The word the ledger's third field holds, which is also the verdict's own word.
    pub fn word(self) -> &'static str {
        match self {
            Expects::Refused => "refused",
            Expects::Divergent => "divergent",
        }
    }

    /// The verb the report uses of a row that still holds.
    fn verb(self) -> &'static str {
        match self {
            Expects::Refused => "refuses",
            Expects::Divergent => "diverges at",
        }
    }

    /// What `word` names.
    fn of(word: &str) -> Option<Self> {
        match word {
            "refused" => Some(Expects::Refused),
            "divergent" => Some(Expects::Divergent),
            _ => None,
        }
    }
}

impl std::fmt::Display for Expects {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.write_str(self.word())
    }
}

/// One gap we have written down as open.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct KnownGap {
    /// The case that holds it.
    pub case: String,
    /// The venue it is expected on: [`EVERY_VENUE`] for all of them, or a pattern held against the
    /// venue's own description ([`KnownGap::covers_venue`]).
    pub venue: String,
    /// The verdict the row expects at its site.
    pub expects: Expects,
    /// The site, as node names from the root.
    pub site: String,
    /// What the run answers there, as the report prints it: our typed error for a refusal, and for
    /// a divergence the one leaf fact that moved, which names Metal's value and ours. Part of the
    /// row's identity, so a gap that changed into another gap at the same site is a new gap rather
    /// than this one.
    pub answer: String,
    /// The gap the row is waiting on, in words: what Metal does, what we do instead, and where the
    /// fix belongs. Not compared — it is what a reader needs — but a row must carry one.
    pub waiting_on: String,
}

impl KnownGap {
    /// The reason the row states, where the ledger's vocabulary names it ([`Reason`]): its
    /// waiting-on field's head. [`parse`] refuses a malformed head, so a row read from a ledger
    /// that names no vocabulary reason is its own words.
    pub fn reason(&self) -> Reason {
        Reason::of(&self.waiting_on).unwrap_or(Reason::Words)
    }

    /// What identifies the row within a venue: the case, the verdict it expects, the site and the
    /// answer there.
    fn key(&self) -> (&str, Expects, &str, &str) {
        (&self.case, self.expects, &self.site, &self.answer)
    }

    /// Whether the row is expected on `venue`.
    ///
    /// [`EVERY_VENUE`] covers all of them. Otherwise the field is held, without case, against the
    /// venue's whole description — its device name, a CPU implementation's host vector instruction
    /// set ([`Venue::host`]), its driver and API version — so a row can name one device
    /// (`Apple M3 Max`), one implementation (`llvmpipe`), one host's (`on x86-64 AVX2`) or a family
    /// of them, and a venue that joins later needs no new vocabulary. A pattern that matches nothing holds
    /// the row nowhere, so the gap it stands for reads as unlisted and fails the run: a typo is
    /// loud rather than silent.
    pub fn covers_venue(&self, venue: &Venue) -> bool {
        if self.venue == EVERY_VENUE {
            return true;
        }
        venue
            .to_string()
            .to_lowercase()
            .contains(&self.venue.to_lowercase())
    }
}

/// The head of a row waiting on a feature the streams use that a lane is building: the lane follows
/// in parentheses.
const NOT_BUILT_YET: &str = "refused: not built yet";

/// The head of a row waiting on a call no stream uses: the census run that found none follows in
/// parentheses.
const NO_STREAM_USES_IT: &str = "refused: no stream uses it yet";

/// The head of a row of a Metal object or getter the runtime keeps no counterpart of, which its
/// consumer implements: the object's name ([`ConsumerObject::name`]) follows in parentheses.
const THE_CONSUMERS: &str = "refused: the consumer's own";

/// A Metal object or getter family the runtime keeps no counterpart of by design, which its
/// consumer implements ([`Reason::TheConsumers`]), each named in the runtime's README as the
/// consumer's. The list is closed: a new one is the planner's ruling, not an edit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConsumerObject {
    /// `MTLRenderPassDescriptor`'s read-back of its own fields: the runtime's
    /// `RenderPassDescriptor` is a value its consumer fills.
    RenderPassDescriptor,
    /// `MTLHeap` and its getters: the runtime's blocks are memory its consumer places resources
    /// in, and no heap. A heap's size and alignment for a buffer or a texture is the device's
    /// query, the runtime's own, and not this.
    Heap,
    /// `MTLComputeCommandEncoder`'s `dispatchType`: the runtime takes it as the encoder's
    /// argument and keeps no getter.
    DispatchType,
    /// `MTLTexture`'s `cpuCacheMode`, `hazardTrackingMode` and `allowGPUOptimizedContents`: the
    /// runtime's `TextureDescriptor` states none of them.
    TextureDescriptorFields,
    /// A pipeline state's reflection and the `MTLPipelineOption` bits a creation takes, each named:
    /// the runtime builds no reflection and acts on no option, and its consumer builds reflection
    /// from the library's AIR. An option outside the named ones is not this object.
    PipelineReflection,
    /// `MTLTexture`'s `parentTexture`, `parentRelativeLevel`, `parentRelativeSlice` and
    /// `shareable`: the runtime's view keeps the texture whose image it reads and no getter, and
    /// its consumer keeps the texture it made a view of and what it shared. A view's own shape and
    /// swizzle are the runtime's, and not this.
    TextureViewParents,
    /// The `label` of an object or a descriptor: the runtime keeps no label on anything, and its
    /// consumer keeps the label it set.
    Label,
    /// The `device` of a library or a function: the runtime's library and functions are the
    /// translator's, which keep no device, and its consumer knows which device it made each on.
    Device,
}

impl ConsumerObject {
    /// Every object, in the order the count line names them.
    pub const ALL: [ConsumerObject; 8] = [
        ConsumerObject::RenderPassDescriptor,
        ConsumerObject::Heap,
        ConsumerObject::DispatchType,
        ConsumerObject::TextureDescriptorFields,
        ConsumerObject::PipelineReflection,
        ConsumerObject::TextureViewParents,
        ConsumerObject::Label,
        ConsumerObject::Device,
    ];

    /// The name a row states in its parentheses.
    pub const fn name(self) -> &'static str {
        match self {
            ConsumerObject::RenderPassDescriptor => "MTLRenderPassDescriptor",
            ConsumerObject::Heap => "MTLHeap",
            ConsumerObject::DispatchType => "dispatchType",
            ConsumerObject::TextureDescriptorFields => {
                "cpuCacheMode, hazardTrackingMode, allowGPUOptimizedContents"
            }
            ConsumerObject::PipelineReflection => {
                "MTLComputePipelineReflection, MTLRenderPipelineReflection, \
                 MTLPipelineOptionFailOnBinaryArchiveMiss, MTLPipelineOptionArgumentInfo, \
                 MTLPipelineOptionBufferTypeInfo, MTLPipelineOptionBindingInfo"
            }
            ConsumerObject::TextureViewParents => {
                "parentTexture, parentRelativeLevel, parentRelativeSlice, shareable"
            }
            ConsumerObject::Label => "label",
            ConsumerObject::Device => "device",
        }
    }

    /// The object a row's parenthesis names, or why it names none of the list.
    fn named(name: &str) -> Result<ConsumerObject, String> {
        (Self::ALL.into_iter().find(|object| object.name() == name)).ok_or_else(|| {
            let names: Vec<&str> = Self::ALL.iter().map(|object| object.name()).collect();
            format!(
                "a row of the consumer's own names one of the closed list ({}), and \"{name}\" is \
                 none: a new one is the planner's ruling",
                names.join("; ")
            )
        })
    }
}

/// Why a row is open, where the ledger's vocabulary names it: the head of its waiting-on field,
/// which the replay counts ([`open_work`]). Every other row's reason is its own words.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Reason {
    /// `refused: not built yet (<lane>)`: a feature the streams use that `lane` is building. The
    /// landing that builds it deletes the row. Their count is the cutover's remaining work, and it
    /// is zero at the cutover.
    NotBuiltYet { lane: String },
    /// `refused: no stream uses it yet (<census>)`: a call the census run `census` of replay's
    /// streams found no stream using. The day a census finds a use, the row's reason becomes
    /// [`Reason::NotBuiltYet`] with its lane.
    NoStreamUsesIt { census: String },
    /// `refused: the consumer's own (<object>)`: one of the closed list of Metal objects and getters
    /// the runtime keeps no counterpart of by design ([`ConsumerObject`]). The consumer implements
    /// what Metal's object answers, so this side has nothing of the runtime's to compare. It is
    /// outside the cutover's count.
    TheConsumers { object: ConsumerObject },
    /// The row's own words: a device's own answer, what Metal does not document, and the like.
    Words,
}

impl Reason {
    /// The reason `waiting_on` states at its head, or why a head that names the vocabulary is
    /// malformed: a reason without its parenthesis, or with an empty one.
    fn of(waiting_on: &str) -> Result<Reason, String> {
        if let Some(lane) = named_after(waiting_on, NOT_BUILT_YET)? {
            return Ok(Reason::NotBuiltYet { lane });
        }
        if let Some(census) = named_after(waiting_on, NO_STREAM_USES_IT)? {
            return Ok(Reason::NoStreamUsesIt { census });
        }
        if let Some(name) = named_after(waiting_on, THE_CONSUMERS)? {
            let object = ConsumerObject::named(&name)?;
            return Ok(Reason::TheConsumers { object });
        }
        Ok(Reason::Words)
    }
}

/// The name in parentheses after `head` where `waiting_on` starts with it, `None` where it does
/// not, and why where the parenthesis is missing or empty.
fn named_after(waiting_on: &str, head: &str) -> Result<Option<String>, String> {
    let Some(rest) = waiting_on.strip_prefix(head) else {
        return Ok(None);
    };
    (rest.strip_prefix(" ("))
        .and_then(|rest| rest.split_once(')'))
        .map(|(named, _)| named.trim())
        .filter(|named| !named.is_empty())
        .map(|named| Some(named.to_owned()))
        .ok_or_else(|| format!("a row waiting on \"{head}\" names who: \"{head} (<name>)\""))
}

/// What `ledger`'s typed reasons count, as two lines of a replay's report: its rows not built yet
/// by lane, which is the cutover's remaining work, and its rows no stream uses by census run; then,
/// on a line of their own and outside the cutover's count, its rows of the consumer's own by
/// object.
pub fn open_work(ledger: &[KnownGap], file: &str) -> String {
    let mut lanes = BTreeMap::<String, usize>::new();
    let mut censuses = BTreeMap::<String, usize>::new();
    let mut objects = BTreeMap::<&'static str, usize>::new();
    for row in ledger {
        match row.reason() {
            Reason::NotBuiltYet { lane } => *lanes.entry(lane).or_default() += 1,
            Reason::NoStreamUsesIt { census } => *censuses.entry(census).or_default() += 1,
            Reason::TheConsumers { object } => *objects.entry(object.name()).or_default() += 1,
            Reason::Words => {}
        }
    }
    fn each<K: std::fmt::Display>(counted: &BTreeMap<K, usize>, apart: &str) -> String {
        match counted.is_empty() {
            true => String::new(),
            false => {
                let named: Vec<String> = counted
                    .iter()
                    .map(|(name, n)| format!("{name} {n}"))
                    .collect();
                format!(" ({})", named.join(apart))
            }
        }
    }
    format!(
        "{file}: {} row(s) not built yet{}, the cutover's remaining work; {} row(s) no stream \
         uses yet{}\n{file}: {} row(s) the consumer's own{}, which its consumer implements and the \
         cutover does not count",
        lanes.values().sum::<usize>(),
        each(&lanes, ", "),
        censuses.values().sum::<usize>(),
        each(&censuses, ", "),
        objects.values().sum::<usize>(),
        each(&objects, "; "),
    )
}

/// Why a run is not what the ledger expects. Each of these fails the run.
#[derive(Clone, Debug, PartialEq)]
pub enum Expectation {
    /// A gap the ledger does not list: one nobody has written down.
    Unlisted {
        case: String,
        venue: String,
        expects: Expects,
        site: String,
        answer: String,
    },
    /// A listed row the run no longer holds on a venue it names: the gap looks fixed, and the row
    /// must go.
    NoLongerHeld {
        gap: KnownGap,
        /// The venue it was held against.
        venue: String,
    },
    /// A failure the ledger has no business covering. A refusal and a divergence are the only
    /// expected gaps; an unrecorded case and a stale recording are never written down.
    NotLedgerable { case: String, why: String },
}

impl std::fmt::Display for Expectation {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expectation::Unlisted {
                case,
                venue,
                expects,
                site,
                answer,
            } => write!(
                out,
                "{case} on {venue} is {expects} at {site} and the known-gaps ledger does not list \
                 it: {answer}\n  a new gap. Fix it, or add a row to known-gaps.tsv naming the gap \
                 it waits on"
            ),
            Expectation::NoLongerHeld { gap, venue } => write!(
                out,
                "{} on {venue} no longer {} {}, and the known-gaps ledger still lists it\n  the \
                 gap looks fixed: drop the row from known-gaps.tsv, or scope its venue field if it \
                 is another venue's. The ledger must not outlive the bug it stands for.\n  the row \
                 was waiting on: {}",
                gap.case,
                gap.expects.verb(),
                gap.site,
                gap.waiting_on
            ),
            Expectation::NotLedgerable { case, why } => write!(
                out,
                "{case}: {why}\n  the ledger covers a refusal and a divergence; this is neither"
            ),
        }
    }
}

/// The row a reader should add for an unlisted gap, in the ledger's own format, with the one field
/// only a human can fill left as a marker.
///
/// It exists because the report is read by people and by tools, and reconstructing a row's `site`
/// and `answer` from prose is where transcription goes wrong: one stream in this round nearly
/// committed a row for a site that does not exist, because its parser attributed later cases'
/// lines to the last case it had seen. Printing the row removes the reconstruction step entirely.
/// The marker is not a valid `waiting on` field in spirit and the author must replace it; an empty
/// one [`parse`] already refuses.
pub const UNWRITTEN: &str =
    "WRITE THE GAP HERE: what Metal does, what we do instead, and where the fix belongs";

impl Expectation {
    /// The ledger row to add, for an expectation that is an unlisted gap.
    pub fn row(&self) -> Option<String> {
        match self {
            Expectation::Unlisted {
                case,
                venue,
                expects,
                site,
                answer,
            } => Some(format!(
                "{case}\t{venue}\t{expects}\t{site}\t{answer}\t{UNWRITTEN}"
            )),
            _ => None,
        }
    }
}

/// The committed ledger's rows.
pub fn ledger() -> Result<Vec<KnownGap>, String> {
    parse(LEDGER)
}

/// The rows of the new runtime's side's ledger, `known-gaps-rt.tsv`: what a replay of that side
/// is held to, as [`ledger`]'s rows hold the old runtime's.
pub fn rt_ledger() -> Result<Vec<KnownGap>, String> {
    parse(RT_LEDGER)
}

/// `text`'s rows: tab-separated, with blank lines and lines from `#` ignored.
pub fn parse(text: &str) -> Result<Vec<KnownGap>, String> {
    let mut rows = Vec::new();
    for (number, line) in text.lines().enumerate() {
        // Only the carriage return is stripped: a trailing tab is a field a row left empty, and
        // trimming the line would hide it as a missing field instead.
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        let at = number + 1;
        let fields: Vec<&str> = line.split('\t').collect();
        let [case, venue, expects, site, answer, waiting_on] = fields.as_slice() else {
            return Err(format!(
                "known-gaps.tsv line {at}: {} tab-separated fields, and a row has six: the case, \
                 the venue ({EVERY_VENUE} for every venue), the verdict it expects (refused or \
                 divergent), the site, our answer there and the gap it waits on",
                fields.len()
            ));
        };
        let Some(expects) = Expects::of(expects.trim()) else {
            return Err(format!(
                "known-gaps.tsv line {at}: {case} expects {expects:?}, and a row expects either \
                 refused — a call plain Metal answered and we did not — or divergent, a field that \
                 differs with nothing announced to justify it"
            ));
        };
        if venue.trim().is_empty() {
            return Err(format!(
                "known-gaps.tsv line {at}: {case} names no venue. A row that holds everywhere says \
                 so with {EVERY_VENUE}, because an empty field reads as a row nobody scoped"
            ));
        }
        if waiting_on.trim().is_empty() {
            return Err(format!(
                "known-gaps.tsv line {at}: {case} names no gap. A row is an open bug, so it says \
                 what Metal does, what we do instead, and where the fix belongs"
            ));
        }
        Reason::of(waiting_on.trim())
            .map_err(|why| format!("known-gaps.tsv line {at}: {case}: {why}"))?;
        rows.push(KnownGap {
            case: case.trim().to_owned(),
            venue: venue.trim().to_owned(),
            expects,
            site: site.trim().to_owned(),
            answer: answer.trim().to_owned(),
            waiting_on: waiting_on.trim().to_owned(),
        });
    }
    Ok(rows)
}

/// One gap a run held, as a ledger row is written: the case, the verdict, the site and the answer.
///
/// The venue is not part of it, because the key is only ever read inside one row's own venue.
type Key<'a> = (&'a str, Expects, String, String);

/// How a replay run's gaps stand against `ledger`: nothing, where what the run held is exactly the
/// ledger's rows over the cases and venues the run selected, and otherwise every way they differ.
///
/// `rows` is what replay observed. A row the ledger covers still holds its own verdict; this only
/// says whether the run held any surprise.
pub fn against(rows: &[Replayed], ledger: &[KnownGap]) -> Vec<Expectation> {
    let mut found = Vec::new();
    // What each row held, as the keys a ledger row is written in. Kept per row, because the
    // discipline is per venue: a gap one venue holds says nothing about another's.
    let held: Vec<(&Replayed, BTreeSet<Key>)> = rows
        .iter()
        .map(|row| (row, gaps_of(row).collect()))
        .collect();

    for (row, gaps) in &held {
        if gaps.is_empty() && !row.verdict.is_compliant() {
            // A verdict the ledger cannot carry at all: an unrecorded case or a stale answer.
            found.push(Expectation::NotLedgerable {
                case: row.case.clone(),
                why: row.verdict.to_string(),
            });
            continue;
        }
        for (case, expects, site, answer) in gaps {
            let listed = ledger.iter().any(|gap| {
                gap.covers_venue(&row.venue)
                    && gap.key() == (case, *expects, site.as_str(), answer.as_str())
            });
            if !listed {
                found.push(Expectation::Unlisted {
                    case: row.case.clone(),
                    venue: row.venue.label(),
                    expects: *expects,
                    site: site.clone(),
                    answer: answer.clone(),
                });
            }
        }
    }

    // The other direction, which is what keeps the file from rotting into a silencer: a row the
    // run no longer holds is a gap that looks fixed, and the ledger has to lose it. Held per
    // venue, and only against the cases a run selected, so replaying one case does not read the
    // rest as fixed and a run on one venue does not read another venue's rows as fixed.
    for gap in ledger {
        for (row, gaps) in &held {
            if row.case != gap.case || !gap.covers_venue(&row.venue) || !compared(row) {
                continue;
            }
            let still = gaps.iter().any(|(case, expects, site, answer)| {
                (*case, *expects, site.as_str(), answer.as_str()) == gap.key()
            });
            if !still {
                found.push(Expectation::NoLongerHeld {
                    gap: gap.clone(),
                    venue: row.venue.label(),
                });
            }
        }
    }
    found
}

/// Whether `row` compared against Metal at all, which is what the second direction needs before
/// it may read a ledger row as fixed.
///
/// [`Verdict::Unrecorded`] is the one verdict that says nothing was compared: no Metal answer
/// stands for the run, so the run holds no opinion about the case's rows in either direction. It
/// already fails the run outright ([`Expectation::NotLedgerable`]) — what this stops is the run
/// *also* saying every row of that case looks fixed, which is a claim about a comparison that did
/// not happen and is the step at which a live row nearly got retired. Two cases of this suite read
/// `Unrecorded` on every host but the one that recorded them, and the two ledger rows they carry
/// were reported as looking fixed on an x86-64 llvmpipe run; the fingerprint those cases state is
/// host-independent now, and this holds the reading whatever makes a case unrecorded next.
fn compared(row: &Replayed) -> bool {
    !matches!(row.verdict, Verdict::Unrecorded { .. })
}

/// Every gap one replayed row held, as the keys a ledger row is written in.
///
/// A row that both diverges and refuses yields both. The verdict of such a row is `Divergent`,
/// since that is the more severe of the two, and it carries the refused sites with it precisely so
/// that a refusal row cannot go unchecked behind a divergence — which would have been a way for a
/// divergence to launder a refusal.
fn gaps_of(row: &Replayed) -> impl Iterator<Item = Key<'_>> + '_ {
    let (refused, diverged): (&[String], &[crate::diff::Divergence]) = match &row.verdict {
        Verdict::Refused { at, .. } => (at, &[]),
        Verdict::Divergent { found, refused } => (refused, found),
        _ => (&[], &[]),
    };
    let case = row.case.as_str();
    refused
        .iter()
        .map(move |printed| {
            let (site, answer) = split_site(printed);
            (case, Expects::Refused, site.to_owned(), answer.to_owned())
        })
        .chain(diverged.iter().map(move |divergence| {
            (
                case,
                Expects::Divergent,
                divergence.path.clone(),
                divergence.kind.to_string(),
            )
        }))
}

/// A refused site as the verdict prints it, split back into the site and our error there.
///
/// [`Verdict::Refused`] holds `"<site>: <error>"`, and the error may hold the separator itself, so
/// the split takes the first one alone.
fn split_site(printed: &str) -> (&str, &str) {
    let (site, error) = printed.split_once(": ").unwrap_or((printed, ""));
    // The verdict appends what documents the refusal, which is prose for a reader and not part of
    // the row's identity: a gap does not change because Apple reworded its validation message.
    let error = error
        .split_once(" (documented, not excused, by ")
        .map_or(error, |(error, _)| error);
    (site.trim(), error.trim())
}
