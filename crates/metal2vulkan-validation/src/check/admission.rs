//! Executor admission: what may run where, decided before any device opens.
//!
//! The Metal oracle and the Vulkan candidate take an [`AdmittedCase`], which only [`admit`]
//! constructs. Admission checks the case and then applies the GPU rules:
//!
//! - the Metal oracle runs a loop-free case as compiled and any other case only through
//!   fuel-instrumented AIR, so it refuses a cyclic case whose cycles the fuel transform cannot all
//!   bound ([`FuelScope::Uncovered`]).
//! - the engine candidate is not held to loop-freedom: its modules run as compiled, cyclic ones
//!   included, on every device (the user's decision of 2026-10-02). Nothing here stops a submitted
//!   shader that does not end, since submitted commands are not cancelled: the finite device wait
//!   ends only the host's wait (the run fails), the per-run timeout ends only the host process, and
//!   the hang tripwire only refuses later runs. The OS or driver GPU timeout is what stops the
//!   shader.
//!
//! A refusal is its own outcome, [`ExecutionRefused`]: never a tooling requirement, never a
//! mismatch, and never evidence. A run the engine executor cannot perform (the product
//! refused the case, the device did not enable what it needs, or the executor does not
//! implement a resource yet) is [`NotSupported`], also never evidence. A candidate run with no
//! current Metal observation to compare against is [`NeedsMetal`]: the case needs Metal
//! qualification first.

use super::loop_free::{CycleEvidence, FuelScope, LoopClass};
use super::{check_case, CheckedCase};
use crate::case::AuthoredCase;
use crate::gpu::DeviceType;
use crate::requirement::ToolingRequirement;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;
use std::path::Path;

/// Which executor will run the case, and how.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionRequest {
    /// Fresh Metal qualification. Metal has only GPU devices; a case that is not mechanically
    /// loop-free runs fuel-instrumented AIR.
    MetalOracle,
    /// The engine compute candidate on `device`: the binder plans the dispatch, and the
    /// compiled modules run as compiled, loops included.
    EngineCandidate { device: DeviceType },
}

impl ExecutionRequest {
    pub fn device_type(self) -> DeviceType {
        match self {
            Self::MetalOracle => DeviceType::Gpu,
            Self::EngineCandidate { device } => device,
        }
    }
}

/// A run an executor refuses by policy, before any device is requested.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionRefused {
    /// A Metal run of a cyclic case with a cycle oracle fuel cannot bound.
    NotFuelBounded(CycleEvidence),
}

impl ExecutionRefused {
    /// The stable outcome label reported by tools.
    pub fn outcome(&self) -> &'static str {
        match self {
            Self::NotFuelBounded(_) => "execution_refused{not_fuel_bounded}",
        }
    }
}

/// Why the engine executor produced no observation although nothing failed: never evidence,
/// never a mismatch.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NotSupported {
    /// The product refused the case (compile or bind): its typed outcome class and detail, and
    /// whether the binder refused it (`at_bind`: what the case binds, its resources or its
    /// dispatch, rather than its translation or the device's description).
    Product {
        outcome: crate::engine::Outcome,
        detail: String,
        #[serde(default)]
        at_bind: bool,
    },
    /// The binder refused the case's bind state or dispatch on this device (a range above the
    /// descriptor limit, a missing device address, a local size the device cannot run, ...).
    Bind(String),
    /// The opened device did not enable, or does not provide, what the pipeline requires.
    NotEnabled(Vec<String>),
    /// The case needs something this executor does not implement yet (typed
    /// `Unsupported{NotImplemented}` of the tool): a gap that depends on the compiled pipeline's
    /// plan, or a case whose literals do not map.
    NotImplemented(String),
    /// The case carries tooling requirements that stop this executor: the families the shared
    /// vocabulary names (`ToolingRequirement::stops`), which the capability census counts per
    /// executor.
    Requirements(BTreeSet<ToolingRequirement>),
    /// An out-of-bounds case whose run would reach memory no binding gave it, so it is not run
    /// there: the case's module checks the accesses of its targets of the `unchecked` forms
    /// nowhere (`out_of_bounds::Check::Unchecked`), or, for a run through the runtime, the
    /// kernel's `variant` contains one of its targets otherwise than the first variant the
    /// translator lowered of it (`platform::runtime_bounds`), so which module a dispatch runs
    /// would decide it.
    Uncontained {
        #[serde(default)]
        unchecked: BTreeSet<crate::case::out_of_bounds::ContainmentForm>,
        #[serde(default)]
        variant: Option<metal2vulkan::pipeline::KernelVariant>,
    },
    /// The run was restricted to a committed profile (`--profiles-layer`) that states capabilities
    /// the device, as it is, lacks: the restricted context's first check refused before it opened.
    /// `gaps` names each member.
    Restriction { profile: String, gaps: Vec<String> },
}

impl NotSupported {
    /// The stable outcome label reported by tools.
    pub fn outcome(&self) -> String {
        match self {
            Self::Product { outcome, .. } => {
                format!("not_supported{{{}:{}}}", outcome.label(), outcome.key())
            }
            Self::Bind(_) => "not_supported{bind}".into(),
            Self::NotEnabled(_) => "not_supported{not_enabled}".into(),
            Self::NotImplemented(_) => "not_supported{executor_not_implemented}".into(),
            Self::Requirements(requirements) => format!(
                "not_supported{{tooling_requirement:{}}}",
                requirement_names(requirements)
            ),
            Self::Uncontained { unchecked, variant } => format!(
                "not_supported{{uncontained:{}}}",
                unchecked
                    .iter()
                    .map(|form| format!("unchecked {}", form.as_str()))
                    .chain(variant.map(|variant| format!("variant {variant:?}")))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            Self::Restriction { .. } => "not_supported{profile_exceeds_device}".into(),
        }
    }
}

fn requirement_names(requirements: &BTreeSet<ToolingRequirement>) -> String {
    requirements
        .iter()
        .map(|requirement| requirement.as_str())
        .collect::<Vec<_>>()
        .join(",")
}

impl fmt::Display for NotSupported {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Product { detail, .. } | Self::Bind(detail) => {
                write!(formatter, "{}: {detail}", self.outcome())
            }
            Self::NotEnabled(missing) => {
                write!(formatter, "{}: {}", self.outcome(), missing.join(", "))
            }
            Self::NotImplemented(what) => write!(formatter, "{}: {what}", self.outcome()),
            Self::Requirements(_) => write!(
                formatter,
                "{}: the executor does not run these families",
                self.outcome()
            ),
            Self::Uncontained { variant: None, .. } => write!(
                formatter,
                "{}: the compiled module checks the accesses of these forms nowhere, neither in \
                 the shader nor by the device's robustness, so the case's out-of-bounds accesses \
                 are not run",
                self.outcome()
            ),
            Self::Uncontained {
                variant: Some(variant),
                ..
            } => write!(
                formatter,
                "{}: the kernel's {variant:?} module contains an out-of-bounds target otherwise \
                 than its first module, so which one a dispatch runs would decide the case's \
                 out-of-bounds accesses, which are not run",
                self.outcome()
            ),
            Self::Restriction { profile, gaps } => write!(
                formatter,
                "{}: profile {profile} states {} capabilities the device lacks: {}",
                self.outcome(),
                gaps.len(),
                gaps.join("; ")
            ),
        }
    }
}

impl fmt::Display for ExecutionRefused {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFuelBounded(evidence) => write!(
                formatter,
                "{}: the Metal oracle runs a cyclic case only when fuel bounds every cycle ({evidence})",
                self.outcome()
            ),
        }
    }
}

/// The process that held the device died by a signal while it ran a case: the engine
/// candidate's device worker (`worker::device`) contains the crash, so the tool that started it
/// reports this instead of dying with it. A driver fault and an executor fault look alike here. The
/// worker's stderr carries the validation-layer messages as the layer reported them: a message
/// points at the executor, and silence does not clear it (the layer does not check everything, such
/// as an acceleration structure's geometry contents). Never evidence and never a mismatch: the run
/// failed, typed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceCrash {
    /// The signal that ended the worker.
    pub signal: i32,
    /// The kind of device the worker held.
    pub device: DeviceType,
    /// What else the parent knows: the host GPU tripwire it wrote for a GPU, and the worker's
    /// bounded stderr.
    pub detail: String,
}

impl DeviceCrash {
    /// The stable outcome label reported by tools: `device_crash{SIGSEGV}`.
    pub fn outcome(&self) -> String {
        format!("device_crash{{{}}}", signal_name(self.signal))
    }
}

impl fmt::Display for DeviceCrash {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}: the {} device worker died by signal {} while it ran the case; nothing it read \
             is evidence; {}",
            self.outcome(),
            self.device.as_str(),
            self.signal,
            self.detail
        )
    }
}

/// The conventional name of `signal`, or its number.
fn signal_name(signal: i32) -> String {
    let name = match signal {
        4 => "SIGILL",
        5 => "SIGTRAP",
        6 => "SIGABRT",
        8 => "SIGFPE",
        9 => "SIGKILL",
        11 => "SIGSEGV",
        13 => "SIGPIPE",
        15 => "SIGTERM",
        // SIGBUS is 7 on Linux and 10 on the BSDs, macOS among them.
        7 if cfg!(target_os = "linux") => "SIGBUS",
        10 if !cfg!(target_os = "linux") => "SIGBUS",
        _ => return format!("signal{signal}"),
    };
    name.into()
}

/// A candidate has no Metal observation to compare against: the case's row in the Metal environment
/// the run names is missing, or is not current evidence for the checked case
/// (`MetalObservation::current`). The case needs Metal qualification (`corpus-metal`) before a
/// candidate can run it. Reported before any device opens; never evidence and never a mismatch.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeedsMetal {
    pub case_id: String,
    /// The Metal environment the candidate run named.
    pub environment_id: String,
    pub gap: MetalGap,
}

/// Why a case's Metal row is not current evidence.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetalGap {
    /// No row for the case in the environment.
    Missing,
    /// The row was recorded by another Metal oracle (`ORACLE_ABI`), whose output a candidate may
    /// not be held to: requalifying records it under the current one.
    OracleAbi { recorded: String, current: String },
    /// The row is under the current oracle but does not hold for the checked case: another AIR
    /// or other inputs, or a row whose own content is invalid.
    OtherInputs,
    /// The row records that the reference device cannot run the case: Metal's pipeline creation
    /// named an Apple GPU family the device lacks. No candidate can be compared with it, and
    /// requalifying on that device records the same gap.
    ReferenceGap { apple_family: u32 },
}

impl NeedsMetal {
    /// The stable outcome label reported by tools: `needs_metal_qualification{oracle_abi}`.
    pub fn outcome(&self) -> String {
        let gap = match self.gap {
            MetalGap::Missing => "missing",
            MetalGap::OracleAbi { .. } => "oracle_abi",
            MetalGap::OtherInputs => "other_inputs",
            MetalGap::ReferenceGap { .. } => "reference_gap",
        };
        format!("needs_metal_qualification{{{gap}}}")
    }
}

impl fmt::Display for NeedsMetal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (case, environment) = (&self.case_id, &self.environment_id);
        write!(formatter, "{}: ", self.outcome())?;
        match &self.gap {
            MetalGap::Missing => write!(
                formatter,
                "case {case} has no Metal observation in environment {environment}"
            )?,
            MetalGap::OracleAbi { recorded, current } => write!(
                formatter,
                "case {case}'s Metal observation in environment {environment} was recorded by \
                 oracle {recorded}, not the current {current}"
            )?,
            MetalGap::OtherInputs => write!(
                formatter,
                "case {case}'s Metal observation in environment {environment} does not hold for \
                 the checked case's AIR and inputs"
            )?,
            MetalGap::ReferenceGap { apple_family } => {
                return write!(
                    formatter,
                    "case {case}'s Metal observation in environment {environment} is a reference \
                     gap: the device lacks Apple GPU family {apple_family}, which Metal's pipeline \
                     creation named; a device with that family qualifies it"
                )
            }
        }
        formatter.write_str("; qualify the case on Metal (corpus-metal) before a candidate runs it")
    }
}

/// A run recorded under a C3 venue found the device it opened unlike the venue's committed profile:
/// another identity, or a description that differs, either way, in a member the profile reads.
/// Nothing is recorded; the batch stops, since every later run would meet the same device. The
/// venue's profile is recaptured from the device (with the venue's declared identity, if the driver
/// moved) before the venue records again.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VenueMismatch {
    /// The operator environment the run records under.
    pub environment_id: String,
    /// The venue's committed profile.
    pub profile: String,
    /// Each difference, `member: profile's (device's)` for the identity and `path: expected …,
    /// queried …` for the description.
    pub differences: Vec<String>,
}

impl fmt::Display for VenueMismatch {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "venue_mismatch{{{}}}: the device differs from venue {}'s profile {} in {} member(s): \
             {}; recapture the profile from the device before the venue records",
            self.profile,
            self.environment_id,
            self.profile,
            self.differences.len(),
            self.differences.join("; ")
        )
    }
}

/// Why an executor produced no observation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutorError {
    /// Refused by policy; see [`ExecutionRefused`].
    Refused(ExecutionRefused),
    /// The case, the device, or the run failed.
    Failed(String),
    /// The engine executor cannot run the case; see [`NotSupported`].
    NotSupported(NotSupported),
    /// The engine candidate's device worker died by a signal; see [`DeviceCrash`].
    DeviceCrash(DeviceCrash),
    /// A candidate has no current Metal observation to compare against; see [`NeedsMetal`].
    NeedsMetal(NeedsMetal),
    /// A venue's run opened a device unlike its profile; see [`VenueMismatch`].
    VenueMismatch(VenueMismatch),
}

impl fmt::Display for ExecutorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refused(refused) => refused.fmt(formatter),
            Self::Failed(error) => formatter.write_str(error),
            Self::NotSupported(not_supported) => not_supported.fmt(formatter),
            Self::DeviceCrash(crash) => crash.fmt(formatter),
            Self::NeedsMetal(needs) => needs.fmt(formatter),
            Self::VenueMismatch(mismatch) => mismatch.fmt(formatter),
        }
    }
}

impl From<NeedsMetal> for ExecutorError {
    fn from(needs: NeedsMetal) -> Self {
        Self::NeedsMetal(needs)
    }
}

impl From<NotSupported> for ExecutorError {
    fn from(not_supported: NotSupported) -> Self {
        Self::NotSupported(not_supported)
    }
}

impl From<String> for ExecutorError {
    fn from(error: String) -> Self {
        Self::Failed(error)
    }
}

impl From<&str> for ExecutorError {
    fn from(error: &str) -> Self {
        Self::Failed(error.into())
    }
}

impl From<ExecutionRefused> for ExecutorError {
    fn from(refused: ExecutionRefused) -> Self {
        Self::Refused(refused)
    }
}

/// A checked case an executor has agreed to run for one [`ExecutionRequest`].
pub struct AdmittedCase {
    checked: CheckedCase,
    request: ExecutionRequest,
}

impl AdmittedCase {
    pub fn checked(&self) -> &CheckedCase {
        &self.checked
    }

    pub fn request(&self) -> ExecutionRequest {
        self.request
    }

    /// Hand the checked case to the executor admitted for `request`. Any other executor gets an
    /// error, so an admission for one device or executor cannot authorize another.
    pub(crate) fn into_checked_for(self, request: ExecutionRequest) -> Result<CheckedCase, String> {
        if self.request == request {
            Ok(self.checked)
        } else {
            Err(format!(
                "case admitted for {:?} cannot run as {request:?}",
                self.request
            ))
        }
    }
}

/// The GPU rules alone, for a case with mechanical class `class` and oracle fuel scope `fuel`.
pub fn refusal(
    class: &LoopClass,
    fuel: &FuelScope,
    request: ExecutionRequest,
) -> Option<ExecutionRefused> {
    if request.device_type() != DeviceType::Gpu {
        return None;
    }
    match request {
        // An engine run is not held to loop-freedom: its modules run as compiled.
        ExecutionRequest::EngineCandidate { .. } => None,
        ExecutionRequest::MetalOracle => class
            .evidence()
            .and(fuel.evidence())
            .cloned()
            .map(ExecutionRefused::NotFuelBounded),
    }
}

/// Check `case` and admit it for `request`, or refuse it. Opens no device.
pub fn admit(
    root: &Path,
    case: AuthoredCase,
    request: ExecutionRequest,
) -> Result<AdmittedCase, ExecutorError> {
    // A request refused whatever the case holds is refused before the case is read.
    if let Some(refused) = refusal(&LoopClass::LoopFree, &FuelScope::Covered, request) {
        return Err(refused.into());
    }
    let checked = check_case(root, case).map_err(|errors| errors.join("; "))?;
    if let Some(refused) = refusal(checked.loop_class(), &checked.fuel_scope(), request) {
        return Err(refused.into());
    }
    Ok(AdmittedCase { checked, request })
}

#[cfg(test)]
mod tests;
