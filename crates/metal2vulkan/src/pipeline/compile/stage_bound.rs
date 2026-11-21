//! The stage bound: every module kept under the compiled bound, on every device.
//!
//! A driver's compiler takes memory that grows faster than the stage, in the process that creates
//! the pipeline, so one creation past the largest stage measured to compile may take the host's
//! memory. No Vulkan query states how large a stage a device's compiler takes, so every stage, on
//! every device, is held under the compiled bound, the heaviest stage measured to compile on the
//! slowest compiler measured, one form for every device.
//!
//! A module whose entry weighs more than the bound (`spirv::roll::entry_weight`: the entry's
//! instructions with each call counted as its callee's body, as the driver inlines every call)
//! after the repeated regions are rolled is past it:
//!
//! - a kernel that took the `Emulated` simdgroup tier only in place of the `Direct` tier's pin to
//!   the device's widest mode (`lower::simd::widest_pin`) is planned again keeping the pin
//!   (`lower::simd::WidestPin::Kept`), its emulated module dropped and what its plan and emission
//!   hold of the ledger released. Only the selection is made again: the entry is decoded,
//!   specialized, ordered and analyzed once, so the kernel is charged what any compile of it is
//!   charged, whichever path reached it (`super::Analyzed`);
//! - every other module past it, a kernel planned again that still is among them, is refused
//!   (`Unsupported`, `Feature::StageSize`, `Needs::StageBound`, naming as its symbol the module's
//!   entry weight and the bound it passed): a device whose compiler would take a heavier stage is
//!   refused it all the same, a known gap of `metal2vulkan-runtime`.
//!
//! **The compiled bound.** [`COMPILED_STAGE_WEIGHT`] is the heaviest entry measured to compile
//! within host-sized memory: on ANV, Mesa 26.2.6, on the Iris Xe of RPL-P, compute modules created
//! cold through `vkCreateComputePipelines` (job m3-max-26147) of entry weights 73,814, 75,886 and
//! 99,109 took 45 s, 47 s and 132 s and peak resident sets of 1.76, 2.24 and 3.65 GiB, and one of
//! entry weight 532,213 passed 16 GiB at 358 s without finishing, where it was stopped. No margin
//! is taken below it, as no weight above it is measured to compile; it rises when a larger compile
//! is measured within host-sized memory.

use super::{analyze_one, Planned, Request};
use crate::budget::Meter;
use crate::device::DeviceProfile;
use crate::error::{DiagName, Error, Feature, Needs, Unsupported};
use crate::lower::emit::Emitted;
use crate::lower::simd::WidestPin;

/// The heaviest entry ([`crate::spirv::roll::entry_weight`]) every stage is handed: the largest
/// measured to compile (the module docs' compiled bound).
pub(crate) const COMPILED_STAGE_WEIGHT: u64 = 99_109;

/// The entry weight past which no module is handed to the device: the measured one in production
/// ([`StageBounds::MEASURED`]), another in a test, so it reaches the bound with a small module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StageBounds {
    /// Every stage's bound.
    pub(crate) compiled: u64,
}

impl StageBounds {
    /// The bound measured (the module docs).
    pub(crate) const MEASURED: StageBounds = StageBounds {
        compiled: COMPILED_STAGE_WEIGHT,
    };
}

/// `r`'s function planned and emitted with its repeated regions rolled, under `bounds` (the
/// module docs): planned again keeping the widest mode's pin where only that pin's emulation
/// passed the bound, else refused, naming the module's entry weight and the bound
/// ([`past_bound`]).
pub(super) fn emit(
    r: &Request<'_>,
    p: &DeviceProfile,
    bounds: StageBounds,
    m: &mut Meter,
) -> Result<(Planned, Emitted), Error> {
    let analyzed = analyze_one(r, p, m)?;
    let mut widest = WidestPin::Weighed;
    loop {
        let before = m.ledger();
        let (plan, kernel) = analyzed.select(p, widest, m)?;
        let mut emitted = m.weighted(crate::pipeline::compile::weights::EMIT, |m| {
            crate::lower::emit::emit_module(&analyzed.spec, &plan, m)
        })?;
        // Every driver compiles each repeated region of the entry again; above the measured size,
        // repeated regions are written once, as a loop.
        m.weighted(crate::pipeline::compile::weights::ROLL, |m| {
            crate::spirv::roll::roll_entry(&mut emitted.module, m)
        })?;
        // A load of a uniform view that every trip of its loop reads alike leaves the loop.
        m.weighted(crate::pipeline::compile::weights::HOIST, |m| {
            crate::spirv::hoist::hoist_uniform_loads(&mut emitted.module, p.robustness(), m)
        })?;
        let weight = crate::spirv::roll::entry_weight(&emitted.module, m)?;
        if weight <= bounds.compiled {
            return Ok((analyzed.planned(plan, kernel), emitted));
        }
        if widest == WidestPin::Kept || !plan.families.simd.widest_pin {
            return Err(refused(weight, bounds.compiled));
        }
        // The emulated plan and module go, and with them what they hold of the ledger.
        drop((plan, emitted));
        let held = m.ledger().saturating_sub(before);
        m.release(usize::try_from(held).unwrap_or(usize::MAX));
        widest = WidestPin::Kept;
    }
}

/// The refusal of a module of entry weight `weight` past `bound`.
fn refused(weight: u64, bound: u64) -> Error {
    Error::Unsupported(Unsupported {
        feature: Feature::StageSize,
        needs: Needs::StageBound,
        location: None,
        symbol: Some(past_bound(weight, bound)),
    })
}

/// What a stage's refusal names as its symbol: the module's entry weight and the bound it passed.
fn past_bound(weight: u64, bound: u64) -> DiagName {
    DiagName::new(format!("entry weight {weight}, past the bound {bound}"))
}
