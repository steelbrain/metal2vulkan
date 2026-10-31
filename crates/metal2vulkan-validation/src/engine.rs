//! Engine translation and its typed outcomes.
//!
//! Validation tools exercise the engine only: the old engine's lanes and the flag that
//! selected it are deleted. The engine translates a row's retained bitcode for one committed
//! device profile, in-process inside the caller's killable worker, and passes L3 (link, relocate,
//! requirements ⊆ profile, `spirv-val` under the profile's environments) before a row counts as
//! translated.
//!
//! [`Outcome`] is the one validation outcome vocabulary: a `From<&Error>` over the product's
//! error taxonomy, never a second list of reasons.

use crate::l3::L3Failure;
use metal2vulkan::device::VkLimit;
use metal2vulkan::device::{DevicePolicy, DeviceProfile, Requirements};
use metal2vulkan::error::{
    CanonError, Capability, Error, ErrorClass, Feature, InputError, InputErrorKind, InternalError,
    InternalKind, Limit, Location, Needs, RuntimeInput,
};
use metal2vulkan::library::{CompiledFunction, FunctionConstantValues};
use metal2vulkan::pipeline::{
    ColorGroup, CompileReport, EntryDescriptor, EntryRef, LoweringInputs, PipelineLayout,
};
use metal2vulkan::Budget;
use metal2vulkan_air::abi::{AbiInt, OutputRole, Stage as AbiStage};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Why an attempt produced no row-level verdict from the translator itself, or a defect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureKind {
    /// The translator panicked (caught at its entry point or in the worker).
    Panic,
    /// The wall-clock deadline passed (engine meter or worker watchdog). Transient.
    Timeout,
    /// The attempt was cancelled. Transient.
    Cancelled,
    /// The worker breached its memory budget.
    Oom,
    /// The produced module failed L3 (requirements ⊆ profile or `spirv-val`).
    Invalid,
    /// An internal invariant of the translator failed (builder, verifier, requirements).
    Internal,
    /// The host allocator failed a reservation the translator's byte ledger admitted
    /// (`Internal{AllocationFailed}`). Transient like a timeout, and a defect.
    AllocationFailed,
    /// The worker exited abnormally without reporting a result (for example an allocator abort).
    WorkerExit,
    /// No bitcode reached the engine: the row retains none and `llvm-as` rejected its AIR
    /// text ([`RowSource::bitcode`]). Not the translator's verdict and not a defect.
    NoBitcode,
    /// The validation tooling itself failed (spawn, scratch, I/O). Transient.
    Tooling,
    /// A product error variant this crate does not classify yet.
    Unclassified,
}

impl FailureKind {
    pub const ALL: [FailureKind; 11] = [
        FailureKind::Panic,
        FailureKind::Timeout,
        FailureKind::Cancelled,
        FailureKind::Oom,
        FailureKind::Invalid,
        FailureKind::Internal,
        FailureKind::AllocationFailed,
        FailureKind::WorkerExit,
        FailureKind::NoBitcode,
        FailureKind::Tooling,
        FailureKind::Unclassified,
    ];

    /// The kind whose [`FailureKind::as_str`] is `key`.
    pub fn parse(key: &str) -> Option<FailureKind> {
        FailureKind::ALL
            .into_iter()
            .find(|failure| failure.as_str() == key)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            FailureKind::Panic => "panic",
            FailureKind::Timeout => "timeout",
            FailureKind::Cancelled => "cancelled",
            FailureKind::Oom => "oom",
            FailureKind::Invalid => "invalid",
            FailureKind::Internal => "internal",
            FailureKind::AllocationFailed => "allocation_failed",
            FailureKind::WorkerExit => "worker_exit",
            FailureKind::NoBitcode => "no_bitcode",
            FailureKind::Tooling => "tooling",
            FailureKind::Unclassified => "unclassified",
        }
    }
}

/// The typed outcome class of one translation attempt.
///
/// String payloads are the product's own typed values formatted for reporting (a `Feature`, a
/// `Capability` list, a `Limit`); they are grouping keys, never inputs to a decision.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(tag = "class", rename_all = "snake_case")]
pub enum Outcome {
    /// A module was produced and passed L3.
    Translated,
    /// A pipeline input the request did not supply, e.g. `function_constant`.
    NeedsInput { kind: String },
    /// Valid Metal this translator does not implement, that Vulkan cannot represent, or that the
    /// device's driver is measured to run wrong.
    Unsupported {
        reason: UnsupportedReason,
        feature: String,
    },
    /// Valid Metal that needs a device capability the profile lacks (`Needs::AnyOf`).
    DeviceCapabilityMissing { feature: String, capability: String },
    /// Malformed or ABI-inconsistent input.
    InvalidInput { kind: String },
    /// The request (entry, constants) is inconsistent with the library.
    InvalidDescriptor { kind: String },
    /// A deterministic cap was exceeded. `kind` is decided from the typed `Limit`.
    LimitExceeded { kind: LimitKind, limit: String },
    /// A defect, a transient outcome, or a tooling failure.
    Failed { failure: FailureKind },
}

/// Why an `Unsupported` outcome is refused: on every device, or on a device whose driver has a
/// measured defect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnsupportedReason {
    /// Not implemented by this translator version.
    NotImplemented,
    /// No Vulkan 1.2+ mechanism exists.
    NotRepresentable,
    /// The device's driver is measured to run the mechanism wrong (`Needs::DriverDefect`, the
    /// profile's `DriverWorkaround`), or to run one the runtime itself records wrong
    /// (`Needs::RuntimeDriverDefect`).
    DriverDefect,
}

impl UnsupportedReason {
    pub fn as_str(self) -> &'static str {
        match self {
            UnsupportedReason::NotImplemented => "not_implemented",
            UnsupportedReason::NotRepresentable => "not_representable",
            UnsupportedReason::DriverDefect => "driver_defect",
        }
    }
}

/// Which kind of deterministic cap a `LimitExceeded` outcome hit.
///
/// A row hitting `WorkUnits` or `LedgerBytes` is still a translator bug under the AGENTS.md budgets
/// (`WorkUnits` trips before the 20 s ceiling, so it stands in for a timeout). The structural caps
/// bound input size and output shape and are permanent refusals.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LimitKind {
    /// The per-attempt work-unit or ledger-byte budget: a defect.
    Budget,
    /// A structural cap on the input or output (instructions, words, nesting, ...).
    Structural,
}

impl LimitKind {
    pub const ALL: [LimitKind; 2] = [LimitKind::Budget, LimitKind::Structural];

    /// The kind whose [`LimitKind::as_str`] is `key`.
    pub fn parse(key: &str) -> Option<LimitKind> {
        LimitKind::ALL.into_iter().find(|kind| kind.as_str() == key)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            LimitKind::Budget => "budget",
            LimitKind::Structural => "structural",
        }
    }

    /// The kind of `limit`, or `None` for a `Limit` variant this crate does not classify yet.
    pub fn of(limit: &Limit) -> Option<LimitKind> {
        match limit {
            Limit::WorkUnits | Limit::LedgerBytes => Some(LimitKind::Budget),
            Limit::InlinedInstructions
            | Limit::OutputBlocks
            | Limit::IrreducibleDepth
            | Limit::SpirvWords
            | Limit::TypeCount
            | Limit::MetadataNodes
            | Limit::BitstreamNesting
            | Limit::InstructionCount
            | Limit::LayoutNesting
            | Limit::ConstructNesting
            | Limit::PrivateArenaBytes
            | Limit::CallStackBytes => Some(LimitKind::Structural),
            _ => None,
        }
    }
}

/// Keys are bounded so one pathological payload cannot bloat the index or a report line.
const KEY_LIMIT: usize = 256;

fn bounded(value: String) -> String {
    if value.len() <= KEY_LIMIT {
        return value;
    }
    let mut end = KEY_LIMIT;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    value[..end].to_string()
}

/// How an outcome records a refused Metal feature: its `Debug` rendering, bounded. The one
/// rendering of [`Outcome::Unsupported`]'s and [`Outcome::DeviceCapabilityMissing`]'s `feature`,
/// so a reader that recognizes a recorded feature compares it with this rendering of the typed
/// value, never with a spelling of its own.
pub fn feature_key(feature: &Feature) -> String {
    bounded(format!("{feature:?}"))
}

/// How an outcome records one device capability: its `Debug` rendering, one `+`-joined token of
/// [`capabilities_key`].
pub fn capability_key(capability: &Capability) -> String {
    format!("{capability:?}")
}

/// How [`Outcome::DeviceCapabilityMissing`] records `Needs::AnyOf`'s alternatives: each
/// alternative's capabilities ([`capability_key`]) joined by `+`, the alternatives by ` | `,
/// bounded to a key's 256 bytes, so a long key may have lost alternatives at its end
/// ([`capabilities_complete`]).
pub fn capabilities_key(alternatives: &[Vec<Capability>]) -> String {
    bounded(
        alternatives
            .iter()
            .map(|all| all.iter().map(capability_key).collect::<Vec<_>>().join("+"))
            .collect::<Vec<_>>()
            .join(" | "),
    )
}

/// Whether a recorded [`capabilities_key`] holds every alternative: it is shorter than the bound,
/// so nothing was cut from its end.
pub fn capabilities_complete(capability: &str) -> bool {
    capability.len() < KEY_LIMIT
}

/// The leading `depth` variant names of a `Debug` rendering: `Bitcode(UnknownRecord { code: 5 })`
/// at depth 2 is `Bitcode::UnknownRecord`. Payload values (offsets, codes) are dropped so the key
/// groups rows by the error's typed shape.
fn variant_path(debug: &str, depth: usize) -> String {
    let mut names = Vec::new();
    let mut rest = debug;
    while names.len() < depth {
        rest = rest.trim_start_matches(['(', ' ']);
        let end = rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(rest.len());
        let name = &rest[..end];
        if name.is_empty() || !name.starts_with(|c: char| c.is_ascii_uppercase()) {
            break;
        }
        names.push(name);
        rest = &rest[end..];
        if !rest.starts_with('(') {
            break;
        }
    }
    names.join("::")
}

impl Outcome {
    /// The coarse class label: the index `outcome` column.
    pub fn label(&self) -> &'static str {
        match self {
            Outcome::Translated => "translated",
            Outcome::NeedsInput { .. } => "needs_input",
            Outcome::Unsupported { .. } => "unsupported",
            Outcome::DeviceCapabilityMissing { .. } => "device_capability_missing",
            Outcome::InvalidInput { .. } => "invalid_input",
            Outcome::InvalidDescriptor { .. } => "invalid_descriptor",
            Outcome::LimitExceeded { .. } => "limit_exceeded",
            Outcome::Failed { .. } => "failed",
        }
    }

    /// The typed sub-key within the class (empty for classes without one).
    pub fn key(&self) -> String {
        match self {
            Outcome::Translated => String::new(),
            Outcome::NeedsInput { kind } => kind.clone(),
            Outcome::Unsupported { reason, feature } => format!("{} {feature}", reason.as_str()),
            Outcome::DeviceCapabilityMissing {
                feature,
                capability,
            } => format!("{feature} needs {capability}"),
            Outcome::InvalidInput { kind } | Outcome::InvalidDescriptor { kind } => kind.clone(),
            Outcome::LimitExceeded { kind, limit } => format!("{} {limit}", kind.as_str()),
            Outcome::Failed { failure } => failure.as_str().to_string(),
        }
    }

    /// Outcomes that depend on host load or memory, the caller, or the tooling rather than the
    /// input. They never enter a persisted census row.
    pub fn is_transient(&self) -> bool {
        matches!(
            self,
            Outcome::Failed {
                failure: FailureKind::Timeout
                    | FailureKind::Cancelled
                    | FailureKind::AllocationFailed
                    | FailureKind::Tooling
            }
        )
    }

    /// Outcomes that are translator or harness defects: there must be none. Budget breaches
    /// (`LimitExceeded` of kind [`LimitKind::Budget`]) count.
    pub fn is_defect(&self) -> bool {
        matches!(
            self,
            Outcome::LimitExceeded {
                kind: LimitKind::Budget,
                ..
            } | Outcome::Failed {
                failure: FailureKind::Panic
                    | FailureKind::Timeout
                    | FailureKind::Oom
                    | FailureKind::Invalid
                    | FailureKind::Internal
                    | FailureKind::AllocationFailed
                    | FailureKind::WorkerExit
                    | FailureKind::Unclassified
            }
        )
    }

    /// [`Outcome::is_defect`] of a recorded `(label, key)` pair: the typed failure kind or limit
    /// kind is the key's leading token, as [`Outcome::key`] writes it.
    pub fn recorded_is_defect(label: &str, key: &str) -> bool {
        let leading = key.split(' ').next().unwrap_or_default();
        match label {
            "failed" => FailureKind::parse(key)
                .is_some_and(|failure| Outcome::Failed { failure }.is_defect()),
            "limit_exceeded" => LimitKind::parse(leading).is_some_and(|kind| {
                Outcome::LimitExceeded {
                    kind,
                    limit: String::new(),
                }
                .is_defect()
            }),
            _ => false,
        }
    }

    /// The product's coarse class ([`InputError::class`]) of the input error a recorded
    /// `invalid_input` pair names, as [`Outcome::key`] writes it from `Outcome::from(&Error)`:
    /// the class of the typed kind whose rendering is the key. Only the canonicalizer's kinds
    /// ([`CanonError`]) are read, since each renders whole (it carries no payload); every other
    /// key, and every other label, is `None` (C1's list reads it to admit guessed old output on
    /// malformed input, and refuses what it cannot class).
    pub fn recorded_input_class(label: &str, key: &str) -> Option<ErrorClass> {
        if label != "invalid_input" {
            return None;
        }
        [
            CanonError::StaticInitNotEvaluable,
            CanonError::UndefCondition,
            CanonError::CallTypeMismatch,
        ]
        .into_iter()
        .map(|kind| InputError::new(InputErrorKind::Canon(kind), Location::module()))
        .find(|error| Outcome::from(&Error::InvalidInput(error.clone())).key() == key)
        .map(|error| error.class())
    }

    /// The refusal a recorded `(label, key)` pair holds, as [`Outcome::key`] writes it: an
    /// [`Outcome::Unsupported`] (`unsupported`, `<reason> <feature>`) or an
    /// [`Outcome::DeviceCapabilityMissing`] (`device_capability_missing`, `<feature> needs
    /// <capability>`). `None` for every other class, and for a key not in that form.
    pub fn recorded_refusal(label: &str, key: &str) -> Option<Outcome> {
        match label {
            "unsupported" => {
                let (reason, feature) = key.split_once(' ')?;
                let reason = [
                    UnsupportedReason::NotImplemented,
                    UnsupportedReason::NotRepresentable,
                    UnsupportedReason::DriverDefect,
                ]
                .into_iter()
                .find(|known| known.as_str() == reason)?;
                Some(Outcome::Unsupported {
                    reason,
                    feature: feature.to_string(),
                })
            }
            "device_capability_missing" => {
                let (feature, capability) = key.split_once(" needs ")?;
                Some(Outcome::DeviceCapabilityMissing {
                    feature: feature.to_string(),
                    capability: capability.to_string(),
                })
            }
            _ => None,
        }
    }
}

impl From<&Error> for Outcome {
    /// The outcome classes of an engine error.
    fn from(error: &Error) -> Outcome {
        match error {
            Error::Unsupported(u) => {
                let feature = feature_key(&u.feature);
                match &u.needs {
                    Needs::AnyOf(alternatives) => Outcome::DeviceCapabilityMissing {
                        feature,
                        capability: capabilities_key(alternatives),
                    },
                    Needs::NotImplemented => Outcome::Unsupported {
                        reason: UnsupportedReason::NotImplemented,
                        feature,
                    },
                    Needs::NotRepresentable => Outcome::Unsupported {
                        reason: UnsupportedReason::NotRepresentable,
                        feature,
                    },
                    Needs::DriverDefect(_) | Needs::RuntimeDriverDefect(_) => {
                        Outcome::Unsupported {
                            reason: UnsupportedReason::DriverDefect,
                            feature,
                        }
                    }
                    _ => Outcome::Failed {
                        failure: FailureKind::Unclassified,
                    },
                }
            }
            Error::NeedsInput(input) => Outcome::NeedsInput {
                kind: bounded(variant_path(&format!("{input:?}"), 1)),
            },
            Error::InvalidInput(e) => Outcome::InvalidInput {
                kind: variant_path(&format!("{:?}", e.kind), 2),
            },
            Error::InvalidDescriptor(d) => Outcome::InvalidDescriptor {
                kind: variant_path(&format!("{d:?}"), 1),
            },
            Error::LimitExceeded(limit) => match LimitKind::of(limit) {
                Some(kind) => Outcome::LimitExceeded {
                    kind,
                    limit: format!("{limit:?}"),
                },
                None => Outcome::Failed {
                    failure: FailureKind::Unclassified,
                },
            },
            Error::Budget(_) => Outcome::Failed {
                failure: FailureKind::Timeout,
            },
            Error::Cancelled => Outcome::Failed {
                failure: FailureKind::Cancelled,
            },
            Error::Internal(internal) => Outcome::Failed {
                failure: match internal.kind {
                    InternalKind::Panic => FailureKind::Panic,
                    InternalKind::AllocationFailed => FailureKind::AllocationFailed,
                    _ => FailureKind::Internal,
                },
            },
            _ => Outcome::Failed {
                failure: FailureKind::Unclassified,
            },
        }
    }
}

/// One engine attempt: its outcome and a bounded human-readable detail.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attempt {
    pub outcome: Outcome,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// Final SPIR-V words of a translated module.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub words: Option<u64>,
    /// What the compile cost, when it ran.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<AttemptCost>,
}

impl Attempt {
    /// An attempt with no words and no measured cost (it stopped before or outside the compile).
    pub fn uncosted(outcome: Outcome, detail: Option<String>) -> Attempt {
        Attempt {
            outcome,
            detail,
            words: None,
            cost: None,
        }
    }

    /// Settle this attempt, as it stands if `spirv-val` accepts its module
    /// ([`translate_unvalidated`]), with L3's `checked` result (or panic): unchanged, and `true`,
    /// when the module was accepted; else L3's outcome, keeping the compile's cost without the
    /// module's hash, and `false`.
    pub fn settle_l3(&mut self, checked: std::thread::Result<Result<(), L3Failure>>) -> bool {
        let failed = match checked {
            Ok(Ok(())) => return true,
            Ok(Err(l3)) => l3_attempt(&l3),
            Err(payload) => panic_attempt(payload.as_ref()),
        };
        self.outcome = failed.outcome;
        self.detail = failed.detail;
        self.words = None;
        if let Some(cost) = &mut self.cost {
            cost.spirv_sha256 = None;
        }
        false
    }
}

/// What one engine compile cost: the meter's deterministic counts, the heap peak the worker's
/// counting allocator attributed to it, and each lowering strategy's emitted words.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptCost {
    /// The byte ledger's peak.
    pub ledger_peak: u64,
    /// Deterministic work units.
    pub work_units: u64,
    /// Peak live heap bytes attributable to the compile, measured by
    /// [`crate::worker::alloc::CountingAllocator`]; `None` when the process does not count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allocator_peak: Option<u64>,
    /// Words each lowering strategy emitted, from the artifact's `StrategyCensus`. Empty when the
    /// compile failed.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub strategy_words: BTreeMap<String, u64>,
    /// SHA-256 of the linked module's SPIR-V words (little-endian bytes) when it translated, so
    /// two sweeps can tell byte for byte which rows a change moved. An optional measurement older
    /// rows decode as absent and no outcome depends on (`ENGINE_CENSUS_VERSION`'s exemption).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spirv_sha256: Option<String>,
}

/// Detail text is bounded like the old audit's (`spirv-val` failures carry a disassembly).
pub const DETAIL_LIMIT: usize = 4096;

pub fn bounded_detail(detail: &str) -> String {
    detail
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(DETAIL_LIMIT)
        .collect()
}

/// Translate entry `entry` of `bitcode` with the engine for `profile`, with no function
/// constants and default lowering inputs, then link it alone, relocate it, and run L3. A kernel
/// that reads `[[stage_in]]` attributes asks for its stage-input descriptor
/// (`NeedsInput(StageInputDescriptor)`): it is translated again, on a fresh meter, under the one
/// its AIR implies (`executor_contract::stage_inputs::implied_descriptor`), and keeps the
/// `NeedsInput` when the tool cannot derive one. A fragment function refused for writing more
/// colors than the profile draws at once is translated as the render pipeline of its own colors
/// draws it there, in color groups (`color_groups`): one module a group, each linked and checked
/// by L3.
///
/// `deadline` is the per-attempt wall deadline of the engine's meter. Every stage runs under
/// `catch_unwind`, so a panic is an outcome, never an unwind into the caller.
pub fn translate_new(
    bitcode: &[u8],
    entry: &str,
    profile: &DeviceProfile,
    deadline: Option<Instant>,
) -> Attempt {
    translate_new_linked(bitcode, entry, profile, deadline).0
}

/// [`translate_new`], and a translated module's final words (the words L3 accepted) with the
/// pipeline layout link gave it alone (what `msl_portability::moltenvk` configures MoltenVK's
/// conversion from).
pub fn translate_new_linked(
    bitcode: &[u8],
    entry: &str,
    profile: &DeviceProfile,
    deadline: Option<Instant>,
) -> (Attempt, Option<(Vec<u32>, PipelineLayout)>) {
    validated(translate_new_unvalidated(bitcode, entry, profile, deadline))
}

/// [`translate_new`] up to L3's `spirv-val` ([`translate_unvalidated`]).
pub fn translate_new_unvalidated(
    bitcode: &[u8],
    entry: &str,
    profile: &DeviceProfile,
    deadline: Option<Instant>,
) -> (Attempt, Option<PendingL3>) {
    let descriptor = EntryDescriptor::new(bitcode, EntryRef::Name(entry));
    translate_modules_unvalidated(profile, deadline, |budget| {
        let (compiled, report) =
            match profile.compile_entry(&descriptor, &LoweringInputs::new(), budget) {
                (Err(e @ Error::NeedsInput(RuntimeInput::StageInputDescriptor { .. })), report) => {
                    match crate::executor_contract::stage_inputs::implied_descriptor(
                        bitcode,
                        entry,
                        &FunctionConstantValues::defaults(),
                    ) {
                        Ok(Some(stage_input)) => {
                            let mut inputs = LoweringInputs::new();
                            inputs.stage_input = Some(stage_input);
                            profile.compile_entry(&descriptor, &inputs, budget)
                        }
                        Ok(None) | Err(_) => (Err(e), report),
                    }
                }
                other => other,
            };
        match compiled {
            Err(refusal) if color_limit(&refusal) => {
                color_groups(profile, &descriptor, budget, (refusal, report))
            }
            compiled => (compiled.map(|function| vec![function]), report),
        }
    })
}

/// Whether `error` is the refusal of a function's color outputs past what the profile draws at
/// once (`InterfaceComponents` at `MaxFragmentOutputAttachments` or `MaxColorAttachments`).
fn color_limit(error: &Error) -> bool {
    let Error::Unsupported(u) = error else {
        return false;
    };
    let limit = |c: &Capability| {
        matches!(
            c,
            Capability::Limit {
                limit: VkLimit::MaxFragmentOutputAttachments | VkLimit::MaxColorAttachments,
                ..
            }
        )
    };
    u.feature == Feature::InterfaceComponents
        && matches!(&u.needs, Needs::AnyOf(sets) if sets.iter().flatten().any(limit))
}

/// The fragment function of `descriptor`, refused alone for writing more colors than `profile`
/// draws at once (`refusal`, after `report`), compiled as each color group of the render pipeline
/// whose attachments are the colors it writes ([`ColorGroup::of`], `LoweringInputs::color_group`),
/// as the runtime draws that pipeline on the profile (`pipeline::compile::color_group`): every
/// group's module, in group order, or the first group's refusal. A function that is not a fragment
/// function, or whose colors take no group, keeps `refusal`, and so does one of more groups that
/// writes memory, which the pipeline's link refuses so ([`CompiledFunction::writes_memory`]).
/// Each compile runs on a fresh meter of `budget`; the report is their total work and their
/// highest ledger peak.
fn color_groups(
    profile: &DeviceProfile,
    descriptor: &EntryDescriptor<'_>,
    budget: &Budget,
    (refusal, report): (Error, CompileReport),
) -> (Result<Vec<CompiledFunction>, Error>, CompileReport) {
    let present: BTreeSet<u32> =
        match profile.specialized_entry(descriptor, &LoweringInputs::new(), budget) {
            Ok(entry) if matches!(entry.ep.stage, AbiStage::Fragment { .. }) => entry
                .ep
                .outputs
                .iter()
                .filter_map(|output| match output.role {
                    OutputRole::RenderTarget {
                        attachment: AbiInt::Const(color),
                        ..
                    } => Some(color),
                    _ => None,
                })
                .collect(),
            Ok(_) | Err(_) => return (Err(refusal), report),
        };
    let width = ColorGroup::width_on(profile);
    let Some(grouping) = ColorGroup::of(present.into_iter().collect(), width, 0) else {
        return (Err(refusal), report);
    };
    let mut total = report;
    let mut modules = Vec::with_capacity(grouping.groups() as usize);
    for group in 0..grouping.groups() {
        let Some(of_group) = ColorGroup::of(grouping.present.clone(), width, group) else {
            return (Err(refusal), total);
        };
        // The colors of the other groups, at their locations past the group's, have no
        // attachment in the group's pipeline: stored nowhere, as `compile_render_variant` compiles
        // the group's variant (`LoweringInputs::unattached_colors`).
        let mut inputs = LoweringInputs::new();
        inputs.unattached_colors = (grouping.present.iter())
            .filter(|color| !of_group.colors().contains(color))
            .map(|&color| of_group.location(color))
            .collect();
        inputs.color_group = Some(of_group);
        let (compiled, report) = profile.compile_entry(descriptor, &inputs, budget);
        total.work_units += report.work_units;
        total.ledger_peak = total.ledger_peak.max(report.ledger_peak);
        match compiled {
            Ok(function) => modules.push(function),
            Err(error) => return (Err(error), total),
        }
    }
    if grouping.groups() > 1 && modules.iter().any(CompiledFunction::writes_memory) {
        return (Err(refusal), total);
    }
    (Ok(modules), total)
}

/// One engine attempt whose compile `compile` makes under the per-attempt wall deadline
/// `deadline`, linked alone and checked by L3 for `profile` in this process:
/// [`translate_unvalidated`], then [`PendingL3::validate`]. [`translate_new_linked`] and the
/// specialization audit's attempts (`specialization::attempt`) are its two compiles.
pub fn translate_with(
    profile: &DeviceProfile,
    deadline: Option<Instant>,
    compile: impl FnOnce(&Budget) -> (Result<CompiledFunction, Error>, CompileReport),
) -> (Attempt, Option<(Vec<u32>, PipelineLayout)>) {
    validated(translate_unvalidated(profile, deadline, compile))
}

/// An attempt [`translate_unvalidated`] left `pending`, settled by L3 in this process
/// ([`PendingL3::validate`]); any other attempt as it is.
pub fn validated(
    (attempt, pending): (Attempt, Option<PendingL3>),
) -> (Attempt, Option<(Vec<u32>, PipelineLayout)>) {
    match pending {
        Some(pending) => pending.validate(attempt),
        None => (attempt, None),
    }
}

/// One engine attempt up to L3's `spirv-val`: each compile `compile` makes
/// (`DeviceProfile::compile_entry`) runs on a fresh meter of the per-attempt wall deadline
/// `deadline`, so a compile retried with more inputs starts its retry afresh, and the attempt's
/// cost is the report of the compile it returns, its last. The compile runs under the counting
/// allocator's measure and `catch_unwind`; the function is then linked alone and relocated
/// (`DeviceProfile::link_standalone`), and its requirements checked against `profile` (every stage
/// under `catch_unwind`, so a panic is an outcome, never an unwind into the caller).
///
/// A module that got that far comes back with what L3 still has to run on it ([`PendingL3`]),
/// and the attempt as it stands if `spirv-val` accepts the module (translated, with its words and
/// their hash); every other attempt comes back final. A worker runs the pending `spirv-val` in a
/// fresh image of itself, so nothing of the translation is resident beside the validator
/// ([`crate::worker::l3`]).
pub fn translate_unvalidated(
    profile: &DeviceProfile,
    deadline: Option<Instant>,
    compile: impl FnOnce(&Budget) -> (Result<CompiledFunction, Error>, CompileReport),
) -> (Attempt, Option<PendingL3>) {
    translate_modules_unvalidated(profile, deadline, |budget| {
        let (compiled, report) = compile(budget);
        (compiled.map(|function| vec![function]), report)
    })
}

/// [`translate_unvalidated`] of a compile that may make several modules of the one function, its
/// color groups' ([`color_groups`]): each is linked alone, relocated and checked against
/// `profile` as one module is, and the first refusal is the attempt's. The attempt's words are
/// theirs together, its hash that of their bytes in order, and its strategies' words their sums;
/// L3 checks every module ([`PendingL3`]).
fn translate_modules_unvalidated(
    profile: &DeviceProfile,
    deadline: Option<Instant>,
    compile: impl FnOnce(&Budget) -> (Result<Vec<CompiledFunction>, Error>, CompileReport),
) -> (Attempt, Option<PendingL3>) {
    let budget = Budget::with_deadline(deadline);
    // The allocator window is exactly the compiles' scope: everything their ledgers are charged
    // for (decode through serialization) and nothing after it.
    let (compiled, allocator_peak) = crate::worker::alloc::measure(|| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| compile(&budget)))
    });
    let cost = |report: CompileReport, strategy_words| {
        Some(AttemptCost {
            ledger_peak: report.ledger_peak,
            work_units: report.work_units,
            allocator_peak,
            strategy_words,
            spirv_sha256: None,
        })
    };
    let no_module = || {
        Error::Internal(InternalError::new(
            InternalKind::Invariant,
            "a compile that made no module",
        ))
    };
    let (functions, report) = match compiled {
        Ok((Ok(functions), report)) if !functions.is_empty() => (functions, report),
        Ok((Ok(_), report)) => {
            return (
                Attempt {
                    cost: cost(report, BTreeMap::new()),
                    ..error_attempt(&no_module())
                },
                None,
            )
        }
        Ok((Err(error), report)) => {
            return (
                Attempt {
                    cost: cost(report, BTreeMap::new()),
                    ..error_attempt(&error)
                },
                None,
            )
        }
        Err(payload) => {
            return (
                Attempt {
                    cost: cost(CompileReport::default(), BTreeMap::new()),
                    ..panic_attempt(payload.as_ref())
                },
                None,
            )
        }
    };
    let mut strategy_words = BTreeMap::new();
    for function in &functions {
        for (name, used) in &function.strategies().entries {
            *strategy_words.entry(name.clone()).or_insert(0) += used.words;
        }
    }
    let cost = cost(report, strategy_words);
    let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut modules = Vec::with_capacity(functions.len());
        for function in &functions {
            let module = profile.link_standalone(function).map_err(Err)?;
            crate::l3::check_requirements(function.requirements(), profile).map_err(Ok)?;
            modules.push(PendingModule {
                plan: L3Plan::of(profile, function.requirements()),
                words: module.words,
                layout: module.layout,
            });
        }
        let mut modules = modules.into_iter();
        let first = modules.next().ok_or_else(|| Err(no_module()))?;
        Ok::<_, Result<L3Failure, Error>>(PendingL3 {
            plan: first.plan,
            words: first.words,
            layout: first.layout,
            groups: modules.collect(),
        })
    }));
    match run {
        Ok(Ok(pending)) => {
            let bytes: Vec<u8> = pending
                .modules()
                .flat_map(|(words, _)| words.iter().flat_map(|w| w.to_le_bytes()))
                .collect();
            let attempt = Attempt {
                outcome: Outcome::Translated,
                detail: None,
                words: Some(pending.modules().map(|(words, _)| words.len() as u64).sum()),
                cost: cost.map(|cost| AttemptCost {
                    spirv_sha256: Some(crate::hash::sha256_bytes(&bytes)),
                    ..cost
                }),
            };
            (attempt, Some(pending))
        }
        Ok(Err(Err(error))) => (
            Attempt {
                cost,
                ..error_attempt(&error)
            },
            None,
        ),
        Ok(Err(Ok(l3))) => (
            Attempt {
                cost,
                ..l3_attempt(&l3)
            },
            None,
        ),
        Err(payload) => (
            Attempt {
                cost,
                ..panic_attempt(payload.as_ref())
            },
            None,
        ),
    }
}

/// What L3 runs `spirv-val` under: the target environments and layout relaxations the profile's
/// plan names (`DeviceProfile::spirv_validation`), and the module's requirements as a rejection
/// reports them.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct L3Plan {
    pub target_envs: Vec<String>,
    pub relaxations: Vec<String>,
    pub requirements: String,
}

impl L3Plan {
    /// The plan of `profile` for a module of `requirements`.
    pub fn of(profile: &DeviceProfile, requirements: &Requirements) -> L3Plan {
        let plan = profile.spirv_validation();
        L3Plan {
            target_envs: plan.target_envs.iter().map(|e| e.to_string()).collect(),
            relaxations: plan.relaxations.iter().map(|r| r.to_string()).collect(),
            requirements: format!("{requirements:?}"),
        }
    }

    /// `spirv-val` on `words` under every environment of the plan.
    pub fn check(&self, words: &[u32]) -> Result<(), L3Failure> {
        let envs: Vec<&str> = self.target_envs.iter().map(String::as_str).collect();
        let relaxations: Vec<&str> = self.relaxations.iter().map(String::as_str).collect();
        crate::l3::spirv_val(words, &envs, &relaxations, &self.requirements)
    }

    /// [`L3Plan::check`] on the module in the file `path`, which this process never reads.
    pub fn check_file(&self, path: &Path) -> Result<(), L3Failure> {
        let envs: Vec<&str> = self.target_envs.iter().map(String::as_str).collect();
        let relaxations: Vec<&str> = self.relaxations.iter().map(String::as_str).collect();
        metal2vulkan_devtools::spirv::check_l3_file(
            path,
            &envs,
            &relaxations,
            &relaxations.join(" "),
            &self.requirements,
        )
    }
}

/// A module [`translate_unvalidated`] linked, relocated, and found within the profile, which
/// `spirv-val` has yet to check: its words, the pipeline layout link gave it alone, and L3's plan;
/// for a function translated in color groups (`color_groups`), the first group's, and the later
/// groups' modules, which L3 checks after it.
#[derive(Debug)]
pub struct PendingL3 {
    pub words: Vec<u32>,
    pub layout: PipelineLayout,
    pub plan: L3Plan,
    pub groups: Vec<PendingModule>,
}

/// A later color group's module of a [`PendingL3`]: its words, its layout and L3's plan for it.
#[derive(Debug)]
pub struct PendingModule {
    pub words: Vec<u32>,
    pub layout: PipelineLayout,
    pub plan: L3Plan,
}

impl PendingL3 {
    /// L3's `spirv-val` in this process: `attempt` (as it stands if the module is accepted) as L3
    /// leaves it ([`Attempt::settle_l3`]), with the words and layout when it accepts them (the
    /// first group's, for a function translated in color groups).
    pub fn validate(self, mut attempt: Attempt) -> (Attempt, Option<(Vec<u32>, PipelineLayout)>) {
        let linked = attempt
            .settle_l3(self.check())
            .then_some((self.words, self.layout));
        (attempt, linked)
    }

    /// L3's `spirv-val` on each module in this process, in order, under `catch_unwind`: the
    /// first failure, else success.
    pub fn check(&self) -> std::thread::Result<Result<(), L3Failure>> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.modules()
                .try_for_each(|(words, plan)| plan.check(words))
        }))
    }

    /// Each module's words and L3's plan for it, the first group's first.
    pub fn modules(&self) -> impl Iterator<Item = (&[u32], &L3Plan)> {
        std::iter::once((self.words.as_slice(), &self.plan)).chain(
            self.groups
                .iter()
                .map(|module| (module.words.as_slice(), &module.plan)),
        )
    }
}

/// Where an engine attempt on an indexed row gets its bitcode.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RowSource<'a> {
    /// The row's retained bitcode, base64 as the index stores it.
    Retained(Cow<'a, str>),
    /// The AIR text of a row that retains no bitcode (a public `.ll` fixture), which `llvm-as`
    /// assembles, as `candidate::engine` does for an authored case.
    Text(Cow<'a, str>),
}

/// Why a row gave the engine no bitcode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BitcodeError {
    /// The retained blob is not base64: index corruption, not a property of the row.
    CorruptBlob(String),
    /// `llvm-as` or its scratch could not run: the host's tooling, not the row.
    Tooling(String),
    /// `llvm-as` ran and rejected the row's AIR text.
    Rejected(String),
}

impl RowSource<'static> {
    /// The source of a row with retained blob `blob_b64` and AIR text `air_ll`: the blob when it
    /// has one, else the text. A row with a blob never gives its sanitized text to the engine.
    pub fn of(blob_b64: Option<String>, air_ll: String) -> RowSource<'static> {
        match blob_b64 {
            Some(blob) => RowSource::Retained(Cow::Owned(blob)),
            None => RowSource::Text(Cow::Owned(air_ll)),
        }
    }
}

impl<'a> RowSource<'a> {
    /// The source of indexed row `row`, borrowed ([`RowSource::of`] without the copies).
    pub fn of_row(row: &'a crate::source::SourceRow) -> RowSource<'a> {
        match &row.blob_b64 {
            Some(blob) => RowSource::Retained(Cow::Borrowed(blob)),
            None => RowSource::Text(Cow::Borrowed(&row.air_ll)),
        }
    }
}

impl RowSource<'_> {
    /// The bitcode: the decoded blob, or the text assembled with `llvm-as` (resolved like every
    /// other tool: `METAL2VULKAN_LLVM_AS`, the Homebrew LLVM prefixes, then `PATH`) in a scratch
    /// directory under the process's temporary directory, removed before this returns. A worker
    /// that translates rows (the census's) runs with its parent's scratch as its
    /// temporary directory, so a killed worker strands nothing. The assembler reads the text on
    /// standard input, so neither the bitcode nor a rejection's detail names that directory.
    pub fn bitcode(&self) -> Result<Vec<u8>, BitcodeError> {
        use metal2vulkan_devtools::{llvm_assemble, AssembleError};
        match self {
            RowSource::Retained(blob) => {
                base64::Engine::decode(&base64::engine::general_purpose::STANDARD, blob.as_ref())
                    .map_err(|error| BitcodeError::CorruptBlob(error.to_string()))
            }
            RowSource::Text(text) => {
                let scratch =
                    crate::ScratchDir::new("engine-row").map_err(BitcodeError::Tooling)?;
                let ll = scratch.path().join("row.ll");
                std::fs::write(&ll, text.as_bytes()).map_err(|error| {
                    BitcodeError::Tooling(format!("write {}: {error}", ll.display()))
                })?;
                llvm_assemble(&ll).map_err(|error| match error {
                    AssembleError::Missing(why) => BitcodeError::Tooling(why),
                    AssembleError::Rejected(why) => BitcodeError::Rejected(why),
                })
            }
        }
    }
}

impl std::fmt::Display for BitcodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BitcodeError::CorruptBlob(why) => {
                write!(
                    f,
                    "index corruption: the retained blob is not base64: {why}"
                )
            }
            BitcodeError::Tooling(why) => write!(f, "cannot assemble the row's AIR text: {why}"),
            BitcodeError::Rejected(why) => write!(f, "llvm-as rejected the row's AIR text: {why}"),
        }
    }
}

impl BitcodeError {
    /// The attempt of a row that gave no bitcode. `llvm-as` rejecting the text is `failed
    /// no_bitcode`, recorded and not a defect (the census key names the assembler); corruption and
    /// missing tooling are transient `tooling`, never recorded.
    pub fn attempt(&self) -> Attempt {
        let failure = match self {
            BitcodeError::CorruptBlob(_) | BitcodeError::Tooling(_) => FailureKind::Tooling,
            BitcodeError::Rejected(_) => FailureKind::NoBitcode,
        };
        Attempt::uncosted(
            Outcome::Failed { failure },
            Some(bounded_detail(&self.to_string())),
        )
    }
}

/// What an engine translation worker reads of an indexed row: its identity (its key,
/// [`crate::source::SourceRow::key`]), its entry, and where its bitcode comes from. Every lane that translates indexed rows with the engine in a
/// worker sends this (the translation census). A row with retained bitcode carries
/// no AIR text, so neither the parent holding it nor the worker input holds the text.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RowInput<'a> {
    pub air_sha256: Cow<'a, str>,
    pub entry: Cow<'a, str>,
    pub source: RowSource<'a>,
}

impl From<crate::source::SourceRow> for RowInput<'static> {
    fn from(row: crate::source::SourceRow) -> RowInput<'static> {
        RowInput {
            air_sha256: Cow::Owned(row.key().into_owned()),
            entry: Cow::Owned(row.entry),
            source: RowSource::of(row.blob_b64, row.air_ll),
        }
    }
}

/// [`translate_new`] of entry `entry` of `source`'s bitcode ([`RowSource::bitcode`]), or the
/// attempt of a row that gave none ([`BitcodeError::attempt`]).
pub fn translate_row(
    source: &RowSource<'_>,
    entry: &str,
    profile: &DeviceProfile,
    deadline: Option<Instant>,
) -> Attempt {
    translate_row_linked(source, entry, profile, deadline).0
}

/// [`translate_row`], and a translated module's final words and pipeline layout
/// ([`translate_new_linked`]).
pub fn translate_row_linked(
    source: &RowSource<'_>,
    entry: &str,
    profile: &DeviceProfile,
    deadline: Option<Instant>,
) -> (Attempt, Option<(Vec<u32>, PipelineLayout)>) {
    validated(translate_row_unvalidated(source, entry, profile, deadline))
}

/// [`translate_row`] up to L3's `spirv-val` ([`translate_unvalidated`]): what the translation
/// worker runs before its L3 stage ([`crate::worker::l3`]).
pub fn translate_row_unvalidated(
    source: &RowSource<'_>,
    entry: &str,
    profile: &DeviceProfile,
    deadline: Option<Instant>,
) -> (Attempt, Option<PendingL3>) {
    match source.bitcode() {
        Ok(bitcode) => translate_new_unvalidated(&bitcode, entry, profile, deadline),
        Err(error) => (error.attempt(), None),
    }
}

fn error_attempt(error: &Error) -> Attempt {
    Attempt::uncosted(
        Outcome::from(error),
        Some(bounded_detail(&error.to_string())),
    )
}

fn panic_attempt(payload: &(dyn std::any::Any + Send)) -> Attempt {
    Attempt::uncosted(
        Outcome::Failed {
            failure: FailureKind::Panic,
        },
        Some(bounded_detail(&format!(
            "engine panicked: {}",
            payload
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("non-string panic payload")
        ))),
    )
}

/// The attempt of a module L3 did not accept ([`l3_outcome`]), with no words and no cost.
fn l3_attempt(failure: &L3Failure) -> Attempt {
    Attempt::uncosted(
        l3_outcome(failure),
        Some(bounded_detail(&format!("L3: {failure}"))),
    )
}

/// An L3 rejection is an invalid module (a defect); a validator that could not run is transient
/// `tooling`, never recorded as the module's outcome.
fn l3_outcome(failure: &L3Failure) -> Outcome {
    Outcome::Failed {
        failure: match failure {
            L3Failure::Requirements(_) | L3Failure::Rejected(_) => FailureKind::Invalid,
            L3Failure::Tooling(_) => FailureKind::Tooling,
            // A future L3 failure kind is not assumed to be the module's fault or the tool's.
            _ => FailureKind::Unclassified,
        },
    }
}

/// The committed device-profile directory (`validation/profiles`).
pub fn profiles_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("profiles")
}

/// The stems of every committed device profile, sorted.
pub fn committed_profiles() -> Result<Vec<String>, String> {
    let dir = profiles_dir();
    let mut stems = std::fs::read_dir(&dir)
        .map_err(|error| format!("read {}: {error}", dir.display()))?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            (path.extension()? == "json")
                .then(|| path.file_stem()?.to_str().map(str::to_string))
                .flatten()
        })
        .collect::<Vec<_>>();
    stems.sort();
    Ok(stems)
}

/// The committed profile `stem` derived under the default (untrusted) policy, the policy every
/// engine census row is recorded under.
pub fn load_profile(stem: &str) -> Result<DeviceProfile, String> {
    Ok(load_device(stem)?.profile().clone())
}

/// The engine's `Device` for committed profile `stem` under the default untrusted policy:
/// what the public compile API compiles against (its profile is [`load_profile`]'s).
pub fn load_device(stem: &str) -> Result<metal2vulkan::device::Device, String> {
    metal2vulkan::device::Device::new(load_description(stem)?, DevicePolicy::default())
        .map_err(|error| format!("{stem}: {error}"))
}

/// The `DeviceDescription` of committed profile `stem`, as the Khronos JSON reader builds it:
/// what a compile worker rebuilds [`load_device`]'s `Device` from.
pub fn load_description(stem: &str) -> Result<metal2vulkan::device::DeviceDescription, String> {
    let committed = committed_profiles()?;
    if !committed.iter().any(|known| known == stem) {
        return Err(format!(
            "unknown profile {stem:?}; committed profiles: {}",
            committed.join(", ")
        ));
    }
    let path = profiles_dir().join(format!("{stem}.json"));
    let text =
        std::fs::read_to_string(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    metal2vulkan::device::DeviceDescription::from_khronos_json(&text, None)
        .map_err(|error| format!("{stem}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal2vulkan::device::VkFeature;
    use metal2vulkan::error::{
        Capability, DescriptorError, DiagName, Feature, InternalError, LowerFamily, Unsupported,
    };

    #[test]
    fn product_errors_map_to_their_typed_classes() {
        let cases: Vec<(Error, &str, &str)> = vec![
            (
                Error::Unsupported(Unsupported::not_implemented(Feature::Lowering(
                    LowerFamily::Texture,
                ))),
                "unsupported",
                "not_implemented Lowering(Texture)",
            ),
            (
                Error::Unsupported(Unsupported::not_representable(Feature::HandleAsPointer)),
                "unsupported",
                "not_representable HandleAsPointer",
            ),
            // A driver's measured defect keys by reason and feature; the defect stays in the
            // detail.
            (
                Error::Unsupported(Unsupported::driver_defect(
                    Feature::FramebufferFetch,
                    metal2vulkan::device::DriverWorkaround::EarlyTestsDropMultisampledImageWrites,
                )),
                "unsupported",
                "driver_defect FramebufferFetch",
            ),
            // A defect the runtime records for its own mechanism keys the same way.
            (
                Error::Unsupported(Unsupported::runtime_driver_defect(
                    Feature::SeparateMultisampledDepthStencil,
                    metal2vulkan::error::RuntimeDriverDefect::MultisampledDepthStencilColourCopies,
                )),
                "unsupported",
                "driver_defect SeparateMultisampledDepthStencil",
            ),
            // A refusal that names its symbol keys by reason and feature alone; the symbol and the
            // location stay in the detail.
            (
                Error::Unsupported(
                    Unsupported::not_implemented(Feature::RayTracing)
                        .naming(DiagName::new("air.intersect.triangle_data")),
                ),
                "unsupported",
                "not_implemented RayTracing",
            ),
            (
                Error::Unsupported(Unsupported::needs(
                    Feature::Int64Atomics,
                    Capability::Feature(VkFeature::ShaderInt64),
                )),
                "device_capability_missing",
                "Int64Atomics needs Feature(ShaderInt64)",
            ),
            (
                Error::NeedsInput(RuntimeInput::VertexDescriptor { attribute: 3 }),
                "needs_input",
                "VertexDescriptor",
            ),
            (
                Error::InvalidDescriptor(DescriptorError::EntryNotFound {
                    requested: DiagName::new("main"),
                    available: 2,
                }),
                "invalid_descriptor",
                "EntryNotFound",
            ),
            (
                Error::LimitExceeded(Limit::WorkUnits),
                "limit_exceeded",
                "budget WorkUnits",
            ),
            (
                Error::LimitExceeded(Limit::SpirvWords),
                "limit_exceeded",
                "structural SpirvWords",
            ),
            (
                Error::Budget(metal2vulkan::error::BudgetExceeded::Time),
                "failed",
                "timeout",
            ),
            (Error::Cancelled, "failed", "cancelled"),
            (
                Error::Internal(InternalError::new(InternalKind::Panic, "boom")),
                "failed",
                "panic",
            ),
            (
                Error::Internal(InternalError::new(InternalKind::Verify, "bad")),
                "failed",
                "internal",
            ),
            (
                Error::Internal(InternalError::new(InternalKind::AllocationFailed, "host")),
                "failed",
                "allocation_failed",
            ),
        ];
        for (error, label, key) in cases {
            let outcome = Outcome::from(&error);
            assert_eq!(outcome.label(), label, "{error:?}");
            assert_eq!(outcome.key(), key, "{error:?}");
            if label == "unsupported" {
                assert_eq!(
                    Outcome::recorded_refusal(label, key),
                    Some(outcome),
                    "{error:?}"
                );
            }
        }
    }

    /// An outcome is transient exactly when the product error is `Retry::Later`, so validation
    /// never records what the product would never memoize.
    #[test]
    fn transient_outcomes_follow_the_product_retry() {
        use metal2vulkan::error::Retry;
        let internal = |kind| Error::Internal(InternalError::new(kind, "detail"));
        for error in [
            Error::Cancelled,
            Error::Budget(metal2vulkan::error::BudgetExceeded::Time),
            internal(InternalKind::AllocationFailed),
            internal(InternalKind::Panic),
            internal(InternalKind::Build),
            internal(InternalKind::Verify),
            internal(InternalKind::RequirementsExceedProfile),
            internal(InternalKind::Invariant),
            Error::LimitExceeded(Limit::LedgerBytes),
        ] {
            let outcome = Outcome::from(&error);
            assert_eq!(
                outcome.is_transient(),
                error.retry() == Retry::Later,
                "{error:?}"
            );
        }
        // Like a timeout: reported as a defect, never recorded.
        let allocation = Outcome::from(&internal(InternalKind::AllocationFailed));
        assert!(allocation.is_transient() && allocation.is_defect());
        assert!(Outcome::recorded_is_defect("failed", "allocation_failed"));
    }

    #[test]
    fn transient_outcomes_are_exactly_the_retry_later_and_tooling_ones() {
        assert!(Outcome::from(&Error::Cancelled).is_transient());
        assert!(
            Outcome::from(&Error::Budget(metal2vulkan::error::BudgetExceeded::Time)).is_transient()
        );
        assert!(Outcome::Failed {
            failure: FailureKind::Tooling
        }
        .is_transient());
        assert!(!Outcome::Failed {
            failure: FailureKind::Oom
        }
        .is_transient());
        assert!(!Outcome::Translated.is_transient());
        // Budget{Time} is transient for the product and still a defect under the AGENTS ceiling.
        assert!(
            Outcome::from(&Error::Budget(metal2vulkan::error::BudgetExceeded::Time)).is_defect()
        );
    }

    /// `WorkUnits`/`LedgerBytes` breaches are translator bugs under the AGENTS budgets; the
    /// structural caps are permanent refusals. Decided from the typed `Limit`, and the recorded
    /// `(label, key)` form agrees with the typed outcome.
    #[test]
    fn budget_limit_breaches_are_defects_and_structural_caps_are_not() {
        let budget = [Limit::WorkUnits, Limit::LedgerBytes];
        let structural = [
            Limit::InlinedInstructions,
            Limit::OutputBlocks,
            Limit::IrreducibleDepth,
            Limit::SpirvWords,
            Limit::TypeCount,
            Limit::MetadataNodes,
            Limit::BitstreamNesting,
            Limit::InstructionCount,
            Limit::LayoutNesting,
            Limit::ConstructNesting,
            Limit::PrivateArenaBytes,
            Limit::CallStackBytes,
        ];
        for (limits, defect) in [(&budget[..], true), (&structural[..], false)] {
            for limit in limits {
                let outcome = Outcome::from(&Error::LimitExceeded(*limit));
                assert_eq!(outcome.label(), "limit_exceeded", "{limit:?}");
                assert_eq!(outcome.is_defect(), defect, "{limit:?}");
                assert!(!outcome.is_transient(), "{limit:?}");
                assert_eq!(
                    Outcome::recorded_is_defect(outcome.label(), &outcome.key()),
                    defect,
                    "{limit:?}"
                );
            }
        }
        assert!(Outcome::recorded_is_defect("failed", "oom"));
        assert!(!Outcome::recorded_is_defect("failed", "no_bitcode"));
        assert!(!Outcome::recorded_is_defect("limit_exceeded", "WorkUnits"));
        assert!(!Outcome::recorded_is_defect("unsupported", "budget x"));
    }

    #[test]
    fn l3_rejections_are_invalid_and_validator_failures_are_transient_tooling() {
        for (failure, key, defect, transient) in [
            (L3Failure::Requirements("r".into()), "invalid", true, false),
            (L3Failure::Rejected("bad id".into()), "invalid", true, false),
            (
                L3Failure::Tooling("no spirv-val".into()),
                "tooling",
                false,
                true,
            ),
        ] {
            let outcome = l3_outcome(&failure);
            assert_eq!(outcome.key(), key, "{failure:?}");
            assert_eq!(outcome.is_defect(), defect, "{failure:?}");
            assert_eq!(outcome.is_transient(), transient, "{failure:?}");
        }
    }

    /// A recorded `invalid_input` key reads back the product's own class of the kind it renders,
    /// and only for the canonicalizer's kinds.
    #[test]
    fn a_recorded_input_error_reads_back_the_products_class() {
        for (kind, class) in [
            (CanonError::CallTypeMismatch, ErrorClass::Malformed),
            (CanonError::UndefCondition, ErrorClass::Malformed),
            (CanonError::StaticInitNotEvaluable, ErrorClass::Unsupported),
        ] {
            let error = InputError::new(InputErrorKind::Canon(kind), Location::module());
            assert_eq!(error.class(), class);
            let outcome = Outcome::from(&Error::InvalidInput(error));
            assert_eq!(
                Outcome::recorded_input_class(outcome.label(), &outcome.key()),
                Some(class),
                "{outcome:?}"
            );
        }
        assert_eq!(
            Outcome::recorded_input_class("invalid_input", "Canon::CallTypeMismatch"),
            Some(ErrorClass::Malformed)
        );
        for (label, key) in [
            ("invalid_descriptor", "Canon::CallTypeMismatch"),
            ("unsupported", "Canon::CallTypeMismatch"),
            ("invalid_input", "Abi::BadValue"),
            ("invalid_input", "Canon"),
            ("invalid_input", ""),
        ] {
            assert_eq!(
                Outcome::recorded_input_class(label, key),
                None,
                "{label} {key}"
            );
        }
    }

    #[test]
    fn variant_paths_drop_payload_values() {
        assert_eq!(
            variant_path("Bitcode(UnknownRecord { code: 5 })", 2),
            "Bitcode::UnknownRecord"
        );
        assert_eq!(variant_path("Abi(MissingNode)", 2), "Abi::MissingNode");
        assert_eq!(
            variant_path("EntryNotFound { requested: x }", 1),
            "EntryNotFound"
        );
        assert_eq!(variant_path("Limit(WorkUnits)", 1), "Limit");
    }

    #[test]
    fn outcomes_round_trip_through_json() {
        let outcome = Outcome::Unsupported {
            reason: UnsupportedReason::NotImplemented,
            feature: "Lowering(Simd)".into(),
        };
        let text = serde_json::to_string(&outcome).unwrap();
        assert_eq!(serde_json::from_str::<Outcome>(&text).unwrap(), outcome);
    }

    #[test]
    fn keys_are_bounded_on_char_boundaries() {
        let long = "é".repeat(400);
        let key = bounded(long);
        assert!(key.len() <= KEY_LIMIT);
        assert!(key.chars().all(|c| c == 'é'));
    }

    #[test]
    fn committed_profiles_load_and_unknown_stems_are_refused() {
        let stems = committed_profiles().unwrap();
        assert!(stems.iter().any(|s| s == "vk12-floor"), "{stems:?}");
        assert!(load_profile("vk12-floor").is_ok());
        let error = load_profile("../moltenvk-m3").unwrap_err();
        assert!(error.contains("unknown profile"), "{error}");
    }

    /// The walking-skeleton kernel translates and passes L3 on the floor profile; a missing
    /// `llvm-as` skips visibly.
    #[test]
    fn owned_store_const_fixture_translates_and_passes_l3() {
        use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};
        let bitcode = match llvm_assemble(&engine_dir().join("store_const.ll")) {
            Ok(bytes) => bytes,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected store_const.ll: {why}"),
        };
        let profile = load_profile("vk12-floor").unwrap();
        let attempt = translate_new(&bitcode, "store_const", &profile, None);
        assert_eq!(attempt.outcome, Outcome::Translated, "{attempt:?}");
        assert!(attempt.words.is_some_and(|words| words > 5));
        // The compile's cost is measured: the meter's counts and, because the lib test binary
        // installs the counting allocator, an attributable heap peak.
        let cost = attempt.cost.as_ref().unwrap();
        assert!(cost.ledger_peak > 0 && cost.work_units > 0, "{cost:?}");
        assert!(cost.allocator_peak.is_some_and(|peak| peak > 0), "{cost:?}");
        let missing = translate_new(&bitcode, "no_such_entry", &profile, None);
        assert_eq!(missing.outcome.label(), "invalid_descriptor", "{missing:?}");
    }

    /// A fragment function of more colors than vk12-floor draws at once (four), refused alone, is
    /// translated as each color group of the pipeline of its colors, every module checked by L3:
    /// five and seven colors in two groups, colors 0 and 5 in one renumbered. One that writes
    /// memory keeps its limit refusal, which a pipeline of more groups meets at link. On lavapipe
    /// (eight) each is one module, as it was.
    #[test]
    fn a_fragment_function_of_more_colors_than_the_floor_translates_in_color_groups() {
        use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};
        let bitcode = match llvm_assemble(&engine_dir().join("color_groups.ll")) {
            Ok(bytes) => bytes,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected color_groups.ll: {why}"),
        };
        for (stem, groups) in [("vk12-floor", 2), ("lavapipe", 1)] {
            let profile = load_profile(stem).unwrap();
            for (entry, modules) in [("cg_five", groups), ("cg_seven", groups), ("cg_sparse", 1)] {
                let (attempt, pending) = translate_new_unvalidated(&bitcode, entry, &profile, None);
                assert_eq!(attempt.outcome, Outcome::Translated, "{entry} on {stem}");
                let pending = pending.unwrap();
                assert_eq!(pending.modules().count(), modules, "{entry} on {stem}");
                let words: usize = pending.modules().map(|(words, _)| words.len()).sum();
                assert_eq!(attempt.words, Some(words as u64), "{entry} on {stem}");
                let (attempt, linked) = pending.validate(attempt);
                assert_eq!(attempt.outcome, Outcome::Translated, "{entry} on {stem}");
                assert!(linked.is_some(), "{entry} on {stem}");
            }
            let counts = translate_new(&bitcode, "cg_counts", &profile, None);
            let expected = match stem {
                "vk12-floor" => "device_capability_missing",
                _ => "translated",
            };
            assert_eq!(
                counts.outcome.label(),
                expected,
                "cg_counts on {stem}: {counts:?}"
            );
        }
    }

    /// A kernel reading stage inputs, which asks for its stage-input descriptor, is translated
    /// under the one its AIR implies; a kernel of the same library that reads only the stage-in
    /// region needs none.
    #[test]
    fn a_stage_input_kernel_translates_under_its_implied_descriptor() {
        use metal2vulkan_devtools::engine_dir;
        let profile = load_profile("moltenvk-m3").unwrap();
        let text = std::fs::read_to_string(engine_dir().join("kernel_stage_in.ll")).unwrap();
        let source = RowSource::of(None, text);
        if let Err(BitcodeError::Tooling(why)) = source.bitcode() {
            eprintln!("SKIPPED: {why}");
            return;
        }
        for entry in ["stage_in_fetch", "stage_in_region", "stage_in_region_short"] {
            let attempt = translate_row(&source, entry, &profile, None);
            assert_eq!(attempt.outcome, Outcome::Translated, "{entry}: {attempt:?}");
        }
    }

    /// A row that retains no bitcode is assembled from its AIR text and translates exactly as the
    /// same bytes retained would; text `llvm-as` rejects is a recorded, non-defect `no_bitcode`; a
    /// corrupt blob is transient `tooling`. A missing `llvm-as` skips visibly.
    #[test]
    fn rows_without_retained_bitcode_are_assembled_from_their_text() {
        use metal2vulkan_devtools::engine_dir;
        let profile = load_profile("vk12-floor").unwrap();
        let text = std::fs::read_to_string(engine_dir().join("store_const.ll")).unwrap();
        let source = RowSource::of(None, text.clone());
        assert_eq!(source, RowSource::Text(Cow::Borrowed(text.as_str())));
        let bitcode = match source.bitcode() {
            Ok(bytes) => bytes,
            Err(BitcodeError::Tooling(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(other) => panic!("store_const.ll did not assemble: {other:?}"),
        };
        let assembled = translate_row(&source, "store_const", &profile, None);
        assert_eq!(assembled.outcome, Outcome::Translated, "{assembled:?}");
        // A row with a blob gives the blob, never its text.
        let blob = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &bitcode);
        let retained = RowSource::of(Some(blob), text);
        assert!(matches!(retained, RowSource::Retained(_)), "{retained:?}");
        let from_blob = translate_row(&retained, "store_const", &profile, None);
        assert_eq!(
            (&from_blob.outcome, from_blob.words),
            (&assembled.outcome, assembled.words)
        );

        let rejected = translate_row(
            &RowSource::Text(Cow::Borrowed("this is not LLVM assembly")),
            "store_const",
            &profile,
            None,
        );
        assert_eq!(rejected.outcome.key(), "no_bitcode", "{rejected:?}");
        assert!(!rejected.outcome.is_defect() && !rejected.outcome.is_transient());
        assert!(rejected
            .detail
            .as_deref()
            .is_some_and(|detail| detail.starts_with("llvm-as rejected")));

        let corrupt = translate_row(
            &RowSource::Retained(Cow::Borrowed("not base64!")),
            "store_const",
            &profile,
            None,
        );
        assert_eq!(corrupt.outcome.key(), "tooling", "{corrupt:?}");
        assert!(corrupt.outcome.is_transient());
    }

    /// A public row's recorded attempt is a function of its text. Each assembly runs in its own
    /// scratch directory, named for the process and a counter, and `llvm-as` named a module without
    /// a `source_filename` line after the file it read, so the row's bitcode, its work units and
    /// ledger peak, and a rejection's detail changed from one run to the next. Two attempts on one
    /// row record the same bitcode, outcome, cost, and detail. A missing `llvm-as` skips visibly.
    #[test]
    fn a_public_rows_attempt_does_not_follow_its_scratch_directory() {
        use metal2vulkan_devtools::engine_dir;
        let profile = load_profile("vk12-floor").unwrap();
        let text: String = std::fs::read_to_string(engine_dir().join("store_const.ll"))
            .unwrap()
            .lines()
            .filter(|line| !line.starts_with("source_filename"))
            .map(|line| format!("{line}\n"))
            .collect();
        let source = RowSource::of(None, text);
        let bitcode = match source.bitcode() {
            Ok(bytes) => bytes,
            Err(BitcodeError::Tooling(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(other) => panic!("store_const.ll did not assemble: {other:?}"),
        };
        assert!(
            source.bitcode().unwrap() == bitcode,
            "one row's text assembled to two bitcodes"
        );
        let recorded = |source: &RowSource<'_>| {
            let attempt = translate_row(source, "store_const", &profile, None);
            let cost = attempt
                .cost
                .map(|cost| (cost.work_units, cost.ledger_peak, cost.strategy_words));
            (attempt.outcome, attempt.detail, attempt.words, cost)
        };
        let first = recorded(&source);
        assert_eq!(first.0, Outcome::Translated, "{first:?}");
        assert_eq!(recorded(&source), first);
        let rejected = RowSource::Text(Cow::Borrowed("this is not LLVM assembly"));
        assert_eq!(recorded(&rejected), recorded(&rejected));
    }

    #[test]
    fn garbage_bitcode_is_typed_invalid_input_not_a_panic() {
        let profile = load_profile("vk12-floor").unwrap();
        let attempt = translate_new(b"not bitcode", "main", &profile, None);
        assert_eq!(attempt.outcome.label(), "invalid_input", "{attempt:?}");
        assert!(attempt.detail.is_some());
        // A refused compile still reports what it cost, and no strategy words.
        assert!(attempt
            .cost
            .as_ref()
            .is_some_and(|cost| cost.strategy_words.is_empty()));
    }
}
