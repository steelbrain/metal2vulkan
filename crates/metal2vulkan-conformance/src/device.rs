//! Opening a runtime device for the suite, and collecting the notices it announces.
//!
//! The device is made with explicit options and no environment: the accurate form of each of the
//! runtime's speed-for-accuracy switches ([`Policy::Exact`]), a shader cache directory this run owns
//! and removes, and a message sink that keeps the degrade-notice codes the
//! device delivered. A warm cache never stands in for a translation the suite compares, and a
//! notice is never lost: it is the thing that justifies a loosened field.
//!
//! # A notice belongs to the case whose work announced it
//!
//! One device serves a whole selection, and its sink hears every notice every case provokes. The
//! sink therefore has to **attribute** each one, and that attribution is what [`Notices`] is: a
//! notice is put in the scope of the case that is running ([`Notices::open`]), or, with no case's
//! scope open, in the venue's own list or the run's late list. There is no fourth place for one to
//! go, so no case can be handed a notice its own work did not announce.
//!
//! What this replaces was a shared list the sink appended to and each case drained. Attribution was
//! then "whatever was in the list when the case asked", which is not the same thing: a notice the
//! runtime delivers off the calling thread (a scheduler thread, a command buffer's host step) can
//! land after the case that provoked it has drained, and the next case to drain takes it. One
//! notice misattributed that way is a divergence reported against a case that did nothing, and a
//! real degradation missing from the case that did.

use metal2vulkan_runtime::{
    DebugMessage, Degradation, Device, DeviceKind, DeviceOptions, MessageSink, ShaderCacheOptions,
};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// A degrade notice no case's scope was open for: the harness could not attribute it, so it is
/// reported rather than given to a case that did not provoke it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unattributed {
    /// The notice's code.
    pub code: String,
    /// The case whose scope closed last before it arrived, which is the likeliest provoker and is
    /// never charged for it.
    pub after: Option<String>,
}

impl std::fmt::Display for Unattributed {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.after {
            Some(case) => write!(out, "{} (after {case} finished)", self.code),
            None => write!(out, "{} (before any case ran)", self.code),
        }
    }
}

/// Where the notices a device delivers are put.
#[derive(Default)]
struct Heard {
    /// The case whose scope is open, with the codes heard since it opened.
    open: Option<(String, Vec<String>)>,
    /// The case whose scope closed last, so a notice arriving late can say what it came after.
    closed: Option<String>,
    /// The codes heard before any case's scope opened: the device's own, as it was made.
    venue: Vec<String>,
    /// The codes heard with no case's scope open, once one had been.
    late: Vec<Unattributed>,
}

/// The degrade-notice codes a device announces, each attributed to the case whose scope was open
/// when it arrived.
///
/// Not `Clone`: the sink is the one shared handle, so nothing can hold a second way to read or
/// drain a case's notices. [`Notices::open`] takes `&mut self`, which is what makes "one case's
/// scope at a time" a fact the borrow checker holds rather than a convention — two open scopes
/// would be the misattribution this type exists to prevent, and the code that would do it does
/// not compile.
#[derive(Default)]
pub struct Notices(Arc<Mutex<Heard>>);

impl Notices {
    /// A sink for a device's options: it attributes each notice's code and drops the rest.
    pub fn sink(&self) -> MessageSink {
        let heard = Arc::clone(&self.0);
        MessageSink::new(move |message: &DebugMessage| {
            if let Some(notice) = message.degradation.as_ref() {
                put(&heard, notice.degradation.code());
            }
        })
    }

    /// Announce codes the way [`Notices::sink`] does, for a test of the attribution itself.
    ///
    /// The runtime's own `DegradeNotice` cannot be built outside its crate, so a test of this
    /// module reaches the sink's own step — the code, put where the scopes say — rather than
    /// fabricating a device message. It closes over the same handle the sink does, so it is
    /// callable while a case's scope is open, as the device is.
    #[cfg(test)]
    pub(crate) fn announcer_for_tests(&self) -> impl Fn(&str) + use<> {
        let heard = Arc::clone(&self.0);
        move |code: &str| put(&heard, code)
    }

    /// Open `case`'s scope: every notice the device announces until the scope closes is this
    /// case's.
    ///
    /// `&mut self` is the guarantee. A second scope cannot be open while this one is, so there is
    /// no moment at which the sink has two cases to choose between.
    pub fn open(&mut self, case: &str) -> CaseNotices<'_> {
        if let Ok(mut heard) = self.0.lock() {
            // A scope left open by a `CaseNotices` that was leaked rather than dropped would
            // otherwise keep taking the next case's notices. Closing it here keeps the invariant
            // whatever a caller did with the guard.
            if let Some((held, _)) = heard.open.take() {
                heard.closed = Some(held);
            }
            heard.open = Some((case.to_owned(), Vec::new()));
        }
        CaseNotices(self)
    }

    /// The codes the device announced as it was made, before any case's scope opened: the venue's
    /// own, which say what this host's driver and release cost and are no case's.
    pub fn venue(&self) -> Vec<String> {
        self.0
            .lock()
            .map(|mut heard| std::mem::take(&mut heard.venue))
            .unwrap_or_default()
    }

    /// Every notice that arrived with no case's scope open after the first one had opened.
    ///
    /// These are the harness's own failures of attribution, not any case's difference: the run
    /// reports them so a notice is never lost, and no case is charged for one.
    pub fn unattributed(&self) -> Vec<Unattributed> {
        self.0
            .lock()
            .map(|heard| heard.late.clone())
            .unwrap_or_default()
    }
}

/// Put `code` where the current scope says it belongs.
fn put(heard: &Arc<Mutex<Heard>>, code: &str) {
    let Ok(mut heard) = heard.lock() else {
        return;
    };
    if let Some((_, codes)) = heard.open.as_mut() {
        codes.push(code.to_owned());
        return;
    }
    match heard.closed.clone() {
        None => heard.venue.push(code.to_owned()),
        after => heard.late.push(Unattributed {
            code: code.to_owned(),
            after,
        }),
    }
}

/// One case's open notice scope: the only way to read the notices a case's own work announced.
///
/// It has no constructor of its own — [`Notices::open`] is the only source — and it is not
/// `Clone`, so a case's notices cannot be read from anywhere but the case's own run. Dropping it
/// closes the scope, after which a notice the runtime delivers late is unattributed rather than
/// the next case's.
pub struct CaseNotices<'a>(&'a Notices);

impl CaseNotices<'_> {
    /// The codes this case's own work has announced, taken away.
    pub fn take(&self) -> Vec<String> {
        self.0
             .0
            .lock()
            .map(|mut heard| match heard.open.as_mut() {
                Some((_, codes)) => std::mem::take(codes),
                None => Vec::new(),
            })
            .unwrap_or_default()
    }
}

impl Drop for CaseNotices<'_> {
    fn drop(&mut self) {
        if let Ok(mut heard) = self.0 .0.lock() {
            if let Some((case, codes)) = heard.open.take() {
                // A notice the case announced and never took is still the case's, and losing it
                // silently is the one thing this module must not do: it reads as unattributed,
                // which the run reports.
                for code in codes {
                    heard.late.push(Unattributed {
                        code,
                        after: Some(case.clone()),
                    });
                }
                heard.closed = Some(case);
            }
        }
    }
}

/// A shader cache directory one run owns: a new, empty directory in the suite's scratch root,
/// removed when dropped, after the device that used it -- and, if this process is killed before
/// that, by the next run of the suite ([`metal2vulkan_devtools::scratch`]).
pub struct ScratchCache(metal2vulkan_devtools::scratch::Owned);

impl ScratchCache {
    /// A directory this run owns, not yet made (the device makes it).
    pub fn new() -> std::io::Result<Self> {
        metal2vulkan_devtools::scratch::Owned::reserved("shader-cache").map(Self)
    }

    /// The directory.
    pub fn path(&self) -> &std::path::Path {
        self.0.path()
    }
}

/// Which physical devices to open.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Venues {
    /// A GPU: discrete, integrated or virtual.
    Gpu,
    /// A CPU implementation (lavapipe), which the no-Metal replay leg runs on.
    Cpu,
}

impl Venues {
    /// The device kinds it names.
    fn kinds(self) -> Vec<DeviceKind> {
        match self {
            Venues::Gpu => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
            Venues::Cpu => vec![DeviceKind::Cpu],
        }
    }
}

/// The Vulkan loader to open, where this host does not have one on its default path.
fn loader() -> Option<PathBuf> {
    // SAFETY: opening the loader library runs only its initializers; the entry is dropped.
    if unsafe { ash::Entry::load() }.is_ok() {
        return None;
    }
    [
        "/opt/homebrew/lib/libvulkan.1.dylib",
        "/opt/homebrew/lib/libvulkan.dylib",
        "/usr/local/lib/libvulkan.1.dylib",
        "/usr/local/lib/libvulkan.dylib",
    ]
    .into_iter()
    .map(PathBuf::from)
    .find(|path| path.is_file())
}

/// Which of the two forms of every speed-for-accuracy choice the device takes.
///
/// Both are the suite's business. `Exact` is what a conformance run should hold the runtime to;
/// `Default` is what a consumer gets, and comparing a case under both is how the suite says
/// whether a difference is the policy's or the implementation's. The owner's rule keeps both forms
/// and asks for tests of both settings, so the command can select either.
///
/// **Neither changes a native answer.** The device's native arithmetic and value answers are the
/// contract under both (AGENTS.md, rule 1): no switch emulates Metal's bits where the device has a
/// native form, so the notes that state the device's own answer fire alike under either policy
/// ([`crate::replay::NATIVE_NOTES`]). The one arithmetic switch, `half_chains`, acts only on a
/// device without `shaderFloat16`, which has no native `half` to answer with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    /// `DevicePolicy::exact_answers()`: the accurate form of each of the runtime's
    /// speed-for-accuracy switches, which are all it selects — `heap_compression`
    /// `Uncompressed`, `completion` `Exact`, `batch_order` `Exact`, `render_lowering` `AtCall`,
    /// `host_page_textures` `Shadow`, `half_chains` `EachOperation` and `pipeline_waits` `Exact`.
    Exact,
    /// The runtime's own defaults: the fast form of every choice, which is what a consumer runs.
    Default,
}

/// The suite's device options under `policy`: every notice at every occurrence, the cache this run
/// owns, and `notices`' sink.
///
/// `announce_each` over every kind is what makes a case's notices its own. A device announces a
/// kind once by default, so on a shared device the first case to provoke a notice holds it and
/// every later case sees nothing: the same case then reads as announced or as exact depending on
/// which cases ran before it in the same process. That would make the notices site — the suite's
/// one mechanism for "a difference is a bug unless we announced it" — depend on the selection and
/// its order. At every occurrence, each case's own work announces its own notices, so a case's
/// verdict is the case's, under either policy.
pub fn options_under(policy: Policy, notices: &Notices, cache: &ScratchCache) -> DeviceOptions {
    let mut options = DeviceOptions::default();
    if policy == Policy::Exact {
        options.policy = options.policy.exact_answers();
    }
    options.announce_each = Degradation::ALL.iter().copied().collect();
    options.shader_cache = ShaderCacheOptions::in_directory(cache.path());
    options.loader = loader();
    options.messages = Some(notices.sink());
    options
}

/// The suite's device options: [`Policy::Exact`], the cache this run owns, and `notices`' sink.
pub fn options(notices: &Notices, cache: &ScratchCache) -> DeviceOptions {
    options_under(Policy::Exact, notices, cache)
}

/// The runtime's default device among the physical devices `venues` names, under `policy`.
pub fn open_under(
    venues: Venues,
    policy: Policy,
    notices: &Notices,
    cache: &ScratchCache,
) -> Result<Device, String> {
    let mut options = options_under(policy, notices, cache);
    options.kinds = venues.kinds();
    Device::system_default_with(&options)
        .map_err(|error| format!("the runtime made no {venues:?} device: {error}"))
}

/// The runtime's default device among the physical devices `venues` names, under [`Policy::Exact`].
pub fn open(venues: Venues, notices: &Notices, cache: &ScratchCache) -> Result<Device, String> {
    open_under(venues, Policy::Exact, notices, cache)
}

/// The guard a family of **pure-query cases** rests on: run every case of a selection on the
/// runtime, and name every site whose answer is the runtime's `Error::Invalid`.
///
/// `Error::Invalid` is this tree's one reading of "Metal's own validation stops this call, which
/// ends Metal's process" — [`crate::cases::heap_metal_sizes`] and [`crate::cases::sparse_heaps`]
/// state the distinction and each hold their own tables to it by hand. This is the same guard,
/// generalized to the whole transcript: a family whose cases carry on past a refusal records it at
/// a site instead of aborting, so the thing to scan is the tree rather than the return value.
///
/// It is also the **order of work** those two modules describe: a table is replayed on the runtime
/// first, where an `Invalid` is a recorded value and nothing else, and only a table that answers no
/// `Invalid` anywhere is ever handed to Metal.
///
/// `Err` is the reason no verdict could be taken at all — no device here — which a caller reports
/// rather than passing over. Every other refusal (ours, or Metal's own nil) is a legitimate
/// recorded value and is not named.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub fn sites_the_runtime_calls_invalid(
    cases: &[Box<dyn crate::case::Case>],
) -> Result<Vec<String>, String> {
    use crate::transcript::{Node, Value};

    /// Whether a recorded value is the runtime's `Error::Invalid`. `Failure` keeps the error as the
    /// variant path a transcript holds, not as the typed error, so the kind is read from there.
    fn is_invalid(value: &Value) -> bool {
        let Value::Variant { name, payload } = value else {
            return false;
        };
        if name != "Err" {
            return false;
        }
        matches!(&**payload, Value::Variant { name, .. } if name == "Error::Invalid")
    }

    /// Every site under `node` whose answer is `Invalid`, by its path.
    fn walk(path: &str, node: &Node, found: &mut Vec<String>) {
        let here = match path.is_empty() {
            true => node.name.to_string(),
            false => format!("{path}/{}", node.name),
        };
        if is_invalid(&node.observed.value) {
            found.push(here.clone());
        }
        for child in &node.children {
            walk(&here, child, found);
        }
    }

    let mut notices = Notices::default();
    let cache = ScratchCache::new().map_err(|why| format!("no scratch cache: {why}"))?;
    let venue = if cfg!(m2v_gpu) {
        Venues::Gpu
    } else {
        Venues::Cpu
    };
    let runtime = open(venue, &notices, &cache)?;
    let mut found = Vec::new();
    for case in cases {
        let mut side = crate::runtime_side::RuntimeSide::new(&runtime, notices.open(case.name()));
        let mut sites = crate::transcript::Sites::new();
        // An abort is the case's own refusal of a call it cannot carry on without, which is a
        // recorded verdict and not a validation stop; what it leaves emitted is still scanned.
        let aborted = case.run(&mut side, &mut sites).err();
        // `finish` consumes the sites, which is why the abort is kept first.
        let mut here = Vec::new();
        walk("", &sites.finish().root, &mut here);
        for site in here {
            found.push(format!("{}: {site}", case.name()));
        }
        if let Some(abort) = aborted {
            if is_invalid(&abort.failure.recorded()) {
                found.push(format!("{}: the call that aborted the case", case.name()));
            }
        }
    }
    Ok(found)
}

#[cfg(test)]
mod tests {
    use super::{Notices, Unattributed};

    /// The property the suite rests on, at the one place a case's notices are decided: **a notice
    /// belongs to the case whose work announced it.**
    ///
    /// A shared list the sink appended to and each case drained could not say this. Attribution
    /// was "whatever was in the list when the case asked", so a notice the runtime delivers off
    /// the calling thread — a scheduler thread, a command buffer's host step, a resource being let
    /// go — landed on whichever case drained next.
    #[test]
    fn a_notice_belongs_to_the_case_whose_scope_was_open() {
        let mut notices = Notices::default();
        let announce = notices.announcer_for_tests();

        let first = notices.open("first");
        announce("a-notice-the-first-case-provoked");
        assert_eq!(first.take(), vec!["a-notice-the-first-case-provoked"]);
        drop(first);

        let second = notices.open("second");
        announce("a-notice-the-second-case-provoked");
        assert_eq!(second.take(), vec!["a-notice-the-second-case-provoked"]);
        drop(second);

        assert_eq!(notices.unattributed(), Vec::new());
    }

    /// The direction that cost a verdict: a notice the first case provoked, delivered **after** it
    /// was compared, is not the next case's. The next case sees nothing, which is exactly what it
    /// would see running alone.
    #[test]
    fn a_notice_that_arrives_between_cases_reaches_neither() {
        let mut notices = Notices::default();
        let announce = notices.announcer_for_tests();

        let first = notices.open("first");
        assert_eq!(first.take(), Vec::<String>::new());
        drop(first);

        // The notice the first case's work provoked, arriving late.
        announce("late");

        let second = notices.open("second");
        assert_eq!(
            second.take(),
            Vec::<String>::new(),
            "the second case was handed a notice its own work did not announce"
        );
        drop(second);

        // Never lost, either: the run reports it against no case at all.
        assert_eq!(
            notices.unattributed(),
            vec![Unattributed {
                code: "late".to_owned(),
                after: Some("first".to_owned()),
            }]
        );
    }

    /// What the device announced as it was made is the venue's, and reaches no case however many
    /// cases run after it.
    #[test]
    fn the_venues_own_notices_reach_no_case() {
        let mut notices = Notices::default();
        let announce = notices.announcer_for_tests();
        announce("what-the-device-cost-to-make");

        let case = notices.open("a-case");
        assert_eq!(case.take(), Vec::<String>::new());
        drop(case);

        assert_eq!(notices.venue(), vec!["what-the-device-cost-to-make"]);
        assert_eq!(notices.unattributed(), Vec::new());
    }

    /// A notice a case announced and never read is reported, not dropped: the owner's rule is that
    /// an accuracy note is logged and announced. It is still charged to no case, because the
    /// case's comparison is already over.
    #[test]
    fn a_notice_a_case_never_took_is_reported_rather_than_lost() {
        let mut notices = Notices::default();
        let announce = notices.announcer_for_tests();
        let case = notices.open("a-case");
        announce("announced-as-the-case-let-its-last-resource-go");
        drop(case);
        assert_eq!(
            notices.unattributed(),
            vec![Unattributed {
                code: "announced-as-the-case-let-its-last-resource-go".to_owned(),
                after: Some("a-case".to_owned()),
            }]
        );
    }
}
