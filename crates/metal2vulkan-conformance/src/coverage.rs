//! The timing-coverage census: which calls have a cost case, and which are untimed.
//!
//! A per-call cost table over an unknown fraction of the API is a table nobody can act on, so
//! "untimed" is a number tracked and driven down the way "unrecorded" is. The plan's words: *a
//! census over the public API — which calls have a cost case and which do not. Today the honest
//! answer is "most of it".* This module is what makes that answer a figure instead of a feeling.
//!
//! # Two censuses, because they answer different questions, and one is exact
//!
//! **The suite's own vocabulary** ([`Census::side`]) is exact and needs no guessing. Every call a
//! case can make is a method of [`crate::case::Side`], and a `times` run says which of them the
//! selection actually made. A `Side` call no case makes is a hole in the suite's coverage of its
//! own vocabulary, and the list of them is work to do.
//!
//! **The runtime's public surface** ([`Census::runtime`]) is the question a consumer cares about,
//! and it can only be answered approximately from here. A `Side` call reaches the runtime through
//! our own code, which calls further helpers, so knowing which public method a timed call reaches
//! needs a call graph this crate has no business building. What the census does instead is honest
//! and still useful: it counts the public methods of the runtime's **frame surface** — the objects
//! a frame makes its calls on — and says which are **named anywhere** in the suite's own
//! runtime-side sources.
//!
//! That count is an **upper bound** on timed coverage, and the report says so in those words. A
//! method named in our sources is at most reachable by a timed call; a method named nowhere in
//! them cannot possibly be timed. The gap between the two numbers is the actionable figure, and
//! narrowing the upper bound to a real one is a call graph's job, named as the next step rather
//! than quietly assumed away.
//!
//! The frame-surface list is the one `scripts/api-cost-coverage` already uses for the other
//! harness's table, kept the same here deliberately: two censuses over two different denominators
//! would be two numbers nobody can put beside each other.

use std::collections::BTreeSet;

/// The objects a frame makes its calls on: the device and its queue, the command buffer, its
/// encoders, the resources they bind and the objects they wait on.
///
/// The rest of the runtime's surface is descriptors, queries, formats, errors and small value
/// types, whose cost a frame does not pay call by call. The same list as
/// `scripts/api-cost-coverage`'s, so the two censuses share a denominator.
const FRAME: &[&str] = &[
    "Device",
    "CommandQueue",
    "CommandBuffer",
    "RenderCommandEncoder",
    "ComputeCommandEncoder",
    "BlitCommandEncoder",
    "ResourceStateCommandEncoder",
    "Buffer",
    "Texture",
    "Heap",
    "SharedEvent",
    "Event",
    "Fence",
    "IndirectCommandBuffer",
    "IndirectRenderCommand",
    "IndirectComputeCommand",
    "ArgumentEncoder",
];

/// The suite's own sources that drive the runtime: where a public runtime method is named if the
/// suite reaches it at all.
const DRIVING: &[&str] = &[
    "runtime_side.rs",
    "runtime_work.rs",
    "render.rs",
    "work.rs",
    "texture.rs",
    "texture_query.rs",
    "queue_slots.rs",
    "event_orders.rs",
    "handler_waits.rs",
    "scheduled_handlers.rs",
    "shared_events.rs",
    "device.rs",
    "replay.rs",
];

/// How much of the call surface the selection times.
#[derive(Clone, Debug)]
pub struct Census {
    /// The rounds the measurement behind it took.
    pub rounds: u32,
    /// The venue it was measured on.
    pub venue: String,
    /// The suite's own vocabulary: every [`crate::case::Side`] call the timing wrapper opens a
    /// span for, and how many times the selection made it.
    pub side: Vec<(String, u64)>,
    /// The `Side` calls the timing wrapper forwards **without** a span, so no selection can time
    /// them. They are listed rather than counted as holes: a census that reports a gap which is a
    /// decision is a census nobody can act on. Read from the wrapper's own source, so a call a
    /// stream forwards without a span appears here instead of vanishing.
    pub unspanned: Vec<String>,
    /// The `Side` calls the timing wrapper does not forward at all. **A defect, never a gap**: a
    /// `Side` method has a default answering `Failure::unavailable`, so an unforwarded one answers
    /// "no such call" where the side answers fine — and in a cost table it reads as a free call.
    /// `tests::the_announcing_side_forwards_every_call` fails on a non-empty list: one guard
    /// over the one forwarding impl, rather than a second copy of it here.
    pub unforwarded: Vec<String>,
    /// The runtime's frame surface: every public method, and whether the suite's own sources name
    /// it at all.
    pub runtime: Vec<(String, bool)>,
}

impl Census {
    /// Take the census over `measured`.
    pub fn of(measured: &crate::times::Measurement) -> Result<Self, String> {
        let timed = timed_calls(measured);
        let forwarded = forwarded_calls()?;
        let mut side = Vec::new();
        let mut unspanned = Vec::new();
        let mut unforwarded = Vec::new();
        for call in side_calls()? {
            match forwarded.iter().find(|(name, _)| *name == call) {
                Some((_, true)) => {
                    let calls = timed.get(&call).copied().unwrap_or(0);
                    side.push((call, calls));
                }
                Some((_, false)) => unspanned.push(call),
                None => unforwarded.push(call),
            }
        }
        let named = named_in_the_suite()?;
        let runtime = frame_surface()?
            .into_iter()
            .map(|method| {
                let reached = named.contains(method.split("::").nth(1).unwrap_or(&method));
                (method, reached)
            })
            .collect();
        Ok(Self {
            rounds: measured.rounds,
            venue: measured.venue.clone(),
            side,
            unspanned,
            unforwarded,
            runtime,
        })
    }

    /// The `Side` calls the selection never made.
    pub fn untimed_side(&self) -> Vec<&str> {
        self.side
            .iter()
            .filter(|(_, calls)| *calls == 0)
            .map(|(call, _)| call.as_str())
            .collect()
    }

    /// The runtime's frame-surface methods the suite's own sources never name, so no case can be
    /// timing them.
    pub fn untouched_runtime(&self) -> Vec<&str> {
        self.runtime
            .iter()
            .filter(|(_, reached)| !reached)
            .map(|(method, _)| method.as_str())
            .collect()
    }

    /// The census, as the command prints it.
    pub fn render(&self) -> String {
        let mut out = String::new();
        let timed = self.side.len() - self.untimed_side().len();
        let reached = self.runtime.len() - self.untouched_runtime().len();
        out.push_str(&format!(
            "timing coverage, over {} round(s) of the selection on {}\n\n",
            self.rounds, self.venue
        ));
        out.push_str(&format!(
            "the suite's own vocabulary: {timed} of the {} Side call(s) the timing wrapper spans \
             are timed by this selection\n",
            self.side.len()
        ));
        if !self.unspanned.is_empty() {
            out.push_str(&format!(
                "  {} more are forwarded without a span on purpose, so they are not holes: {}\n",
                self.unspanned.len(),
                self.unspanned.join(", ")
            ));
        }
        if !self.unforwarded.is_empty() {
            out.push_str(&format!(
                "  {} Side call(s) the wrapper does not forward at all, which is a DEFECT: {}\n",
                self.unforwarded.len(),
                self.unforwarded.join(", ")
            ));
        }
        out.push_str(&format!(
            "the runtime's frame surface: {reached} of {} public method(s) are named anywhere in \
             the suite's runtime-side sources\n",
            self.runtime.len()
        ));
        out.push_str(
            "  that second figure is an UPPER BOUND on timed coverage, not a measurement of it: a \
             method our sources\n  name is at most reachable by a timed call, while one they never \
             name cannot be timed at all. Narrowing it\n  to a real figure needs a call graph; the \
             gap is the work.\n",
        );
        // The honest baseline stays in the generated output, not only in a report: "timing is
        // recorded for every suite" must never come to mean "recorded for everything we can see".
        if !self.runtime.is_empty() {
            out.push_str(&format!(
                "  SO AT LEAST {:.0}% OF THE RUNTIME'S FRAME SURFACE IS UNTIMEABLE BY ANY CASE \
                 ({} of {} methods).\n  Every case being timed does not make the API timed.\n",
                100.0 * (self.runtime.len() - reached) as f64 / self.runtime.len() as f64,
                self.runtime.len() - reached,
                self.runtime.len()
            ));
        }
        let untimed = self.untimed_side();
        if untimed.is_empty() {
            out.push_str("\nevery Side call is timed by this selection\n");
        } else {
            out.push_str(&format!(
                "\n{} Side call(s) no case in this selection makes, so nothing times them:\n",
                untimed.len()
            ));
            for call in untimed {
                out.push_str(&format!("  {call}\n"));
            }
        }
        let untouched = self.untouched_runtime();
        if untouched.is_empty() {
            out.push_str("\nthe suite's sources name every frame-surface method\n");
        } else {
            out.push_str(&format!(
                "\n{} frame-surface method(s) the suite's sources never name, so no case can time \
                 them:\n",
                untouched.len()
            ));
            for method in untouched {
                out.push_str(&format!("  {method}\n"));
            }
        }
        out
    }
}

/// How many times the measurement saw each call, by name.
fn timed_calls(measured: &crate::times::Measurement) -> std::collections::BTreeMap<String, u64> {
    let mut seen = std::collections::BTreeMap::new();
    for case in &measured.cases {
        for call in &case.per_call {
            let held = seen.entry(call.call.clone()).or_insert(0);
            *held += call.most_calls;
        }
    }
    seen
}

/// Every `Side` method the timing wrapper forwards, and whether it opens a span, read from
/// `watch.rs`'s own source.
///
/// Read rather than listed for the same reason [`side_calls`] is: a list would rot, and it would
/// rot the flattering way, by making a forgotten method look like a decision.
fn forwarded_calls() -> Result<Vec<(String, bool)>, String> {
    let source = read_source("watch.rs")?;
    let start = source
        .find("impl<S: Side, W: Watcher> Side for Watching<S, W> {")
        .ok_or("watch.rs holds no `impl<S: Side, W: Watcher> Side for Watching<S, W> {`")?;
    let body = &source[start..];
    let end = body
        .find("\n}\n")
        .ok_or("watch.rs's forwarding impl has no close at the left margin")?;
    let body = &body[..end];
    let mut found = Vec::new();
    let mut name = String::new();
    for line in body.lines() {
        if let Some(rest) = line.strip_prefix("    fn ") {
            if !name.is_empty() {
                found.push((std::mem::take(&mut name), false));
            }
            name = rest
                .chars()
                .take_while(|letter| letter.is_ascii_alphanumeric() || *letter == '_')
                .collect();
            continue;
        }
        // A span whose name is not the method's would attribute one call's cost to another, so
        // the name is matched and not merely the call to `watch`.
        if !name.is_empty() && line.contains(&format!("self.watch(\"{name}\")")) {
            found.push((std::mem::take(&mut name), true));
        }
    }
    if !name.is_empty() {
        found.push((name, false));
    }
    Ok(found)
}

/// Every method of [`crate::case::Side`], read from its own source.
///
/// Read rather than listed, because a list here would be a second copy of the trait and would rot
/// the first time a stream added a call — and it would rot the **safe** way, by making coverage
/// look complete.
fn side_calls() -> Result<Vec<String>, String> {
    let source = read_source("case.rs")?;
    let start = source
        .find("pub trait Side {")
        .ok_or("case.rs holds no `pub trait Side {`")?;
    let body = &source[start..];
    let end = body
        .find("\n}\n")
        .ok_or("case.rs's `pub trait Side` has no close at the left margin")?;
    Ok(methods_at_indent(&body[..end], "    fn "))
}

/// Every public method of the runtime's frame surface, as `Type::method`.
fn frame_surface() -> Result<Vec<String>, String> {
    let root = runtime_src()?;
    let mut found = Vec::new();
    for file in rust_files(&root)? {
        let source = std::fs::read_to_string(&file)
            .map_err(|error| format!("{}: {error}", file.display()))?;
        for (kind, body) in inherent_impls(&source) {
            if !FRAME.contains(&kind.as_str()) {
                continue;
            }
            for method in methods_at_indent(&body, "    pub fn ") {
                found.push(format!("{kind}::{method}"));
            }
            for method in methods_at_indent(&body, "    pub unsafe fn ") {
                found.push(format!("{kind}::{method}"));
            }
        }
    }
    found.sort();
    found.dedup();
    Ok(found)
}

/// Every identifier the suite's own runtime-side sources call as a method (`.name(`).
///
/// A textual reach, deliberately: it answers "could the suite possibly be timing this" and never
/// "is it", which is what [`Census::runtime`] is documented to mean.
fn named_in_the_suite() -> Result<BTreeSet<String>, String> {
    let mut named = BTreeSet::new();
    for file in DRIVING {
        let source = read_source(file)?;
        let mut rest = source.as_str();
        while let Some(at) = rest.find('.') {
            rest = &rest[at + 1..];
            let name: String = rest
                .chars()
                .take_while(|letter| letter.is_ascii_alphanumeric() || *letter == '_')
                .collect();
            if !name.is_empty() && rest[name.len()..].starts_with('(') {
                named.insert(name);
            }
        }
    }
    Ok(named)
}

/// The names of the methods `body` declares with the prefix `head` at one indent.
fn methods_at_indent(body: &str, head: &str) -> Vec<String> {
    body.lines()
        .filter_map(|line| {
            let rest = line.strip_prefix(head)?;
            let name: String = rest
                .chars()
                .take_while(|letter| letter.is_ascii_alphanumeric() || *letter == '_')
                .collect();
            (!name.is_empty()).then_some(name)
        })
        .collect()
}

/// Each inherent `impl Type {` block of `source`, as its type's name and its body.
///
/// A trait impl (`impl Trait for Type`) is not part of the named surface, and neither is anything
/// indented, which in this tree is inside a function or a module.
fn inherent_impls(source: &str) -> Vec<(String, String)> {
    let mut found = Vec::new();
    for (index, line) in source.lines().enumerate() {
        let Some(rest) = line.strip_prefix("impl") else {
            continue;
        };
        if !line.ends_with('{') {
            continue;
        }
        // `impl<'a> Type<'a> {`: step over the generics, then take the type's name.
        let rest = rest.trim_start();
        let rest = match rest.strip_prefix('<') {
            Some(after) => match after.find('>') {
                Some(close) => after[close + 1..].trim_start(),
                None => continue,
            },
            None => rest,
        };
        let kind: String = rest
            .chars()
            .take_while(|letter| letter.is_ascii_alphanumeric() || *letter == '_')
            .collect();
        if kind.is_empty() || rest[kind.len()..].trim_start().starts_with("for ") {
            continue;
        }
        // The block runs to the next `}` at the left margin.
        let body: String = source
            .lines()
            .skip(index + 1)
            .take_while(|line| *line != "}")
            .collect::<Vec<_>>()
            .join("\n");
        found.push((kind, body));
    }
    found
}

/// Every `.rs` file under `root`, recursively.
fn rust_files(root: &std::path::Path) -> Result<Vec<std::path::PathBuf>, String> {
    let mut found = Vec::new();
    let mut to_read = vec![root.to_path_buf()];
    while let Some(directory) = to_read.pop() {
        let entries = std::fs::read_dir(&directory)
            .map_err(|error| format!("{}: {error}", directory.display()))?;
        for entry in entries {
            let entry = entry.map_err(|error| format!("{}: {error}", directory.display()))?;
            let path = entry.path();
            if path.is_dir() {
                to_read.push(path);
            } else if path.extension().is_some_and(|kind| kind == "rs") {
                found.push(path);
            }
        }
    }
    found.sort();
    Ok(found)
}

/// One of the suite's own sources, read from the checkout the binary runs from
/// ([`metal2vulkan_devtools::workspace_dir`]).
///
/// The census reads sources, so it needs the tree. This crate is never published
/// (`publish = false`), so it always runs from a checkout.
fn read_source(file: &str) -> Result<String, String> {
    let path = (metal2vulkan_devtools::workspace_dir())
        .join("crates")
        .join("metal2vulkan-conformance")
        .join("src")
        .join(file);
    std::fs::read_to_string(&path).map_err(|error| format!("{}: {error}", path.display()))
}

/// The runtime crate's sources, beside this crate's in the checkout the binary runs from.
fn runtime_src() -> Result<std::path::PathBuf, String> {
    let path = (metal2vulkan_devtools::workspace_dir())
        .join("crates")
        .join("metal2vulkan-runtime")
        .join("src");
    path.canonicalize()
        .map_err(|error| format!("{}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A span's name is the method's name, in every forwarded method that opens one.
    ///
    /// A span named for another call would attribute one call's cost to a different row, which is
    /// a wrong table rather than a missing one. [`forwarded_calls`] matches the name, so a
    /// mismatch reaches this test as a call with no span.
    #[test]
    fn the_only_calls_without_a_span_are_the_two_that_are_not_the_implementations() {
        let unspanned: Vec<String> = forwarded_calls()
            .unwrap()
            .into_iter()
            .filter(|(_, spanned)| !*spanned)
            .map(|(name, _)| name)
            .collect();
        assert_eq!(
            unspanned,
            ["venue", "announced"],
            "the set of Side calls the timing wrapper does not measure has changed; either a span \
             was dropped or its name no longer matches its method"
        );
    }

    /// The runtime's frame surface is read from the runtime's own sources, so the census's
    /// denominator cannot be a stale list.
    #[test]
    fn the_frame_surface_is_read_from_the_runtime() {
        let surface = frame_surface().unwrap();
        assert!(
            surface.len() > 100,
            "the frame surface came out at {} methods, so the parse found almost nothing",
            surface.len()
        );
        for expected in ["Texture::width", "Buffer::length", "Device::name"] {
            assert!(
                surface.iter().any(|method| method == expected),
                "{expected} is not in the frame surface the census read"
            );
        }
        // Every entry is `Type::method` of a frame type, never a free function or a trait impl's.
        for method in &surface {
            let (kind, _) = method
                .split_once("::")
                .unwrap_or_else(|| panic!("{method}"));
            assert!(FRAME.contains(&kind), "{method} is not a frame type's");
        }
    }

    /// The suite's own driving sources are all read: a file named here that no longer exists would
    /// silently shrink the census's numerator and make coverage look worse than it is.
    #[test]
    fn every_driving_source_the_census_names_exists() {
        for file in DRIVING {
            read_source(file).unwrap_or_else(|why| panic!("{file}: {why}"));
        }
        assert!(!named_in_the_suite().unwrap().is_empty());
    }
}
