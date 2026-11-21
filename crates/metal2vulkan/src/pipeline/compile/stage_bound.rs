//! The stage bound: every module weighed, and a kernel kept under the compiled bound where a lighter
//! plan of it is.
//!
//! A driver's compiler takes memory that grows faster than the stage, in the process that creates
//! the pipeline, so one creation past the largest stage measured to compile may take the host's
//! memory. No Vulkan query states how large a stage a device's compiler takes, so the bound is the
//! heaviest stage measured to compile on the slowest compiler measured, one bound for every device.
//!
//! Every module's entry is weighed (`spirv::roll::entry_weight`: the entry's instructions with each
//! call counted as its callee's body, as the driver inlines every call) after the repeated regions
//! are rolled and written as kept routines (`spirv::outline`), and the weight is the module's
//! ([`ModuleStats::entry_weight`]). A kernel past the
//! bound that took the `Emulated` simdgroup tier only in place of the `Direct` tier's pin to the
//! device's widest mode (`lower::simd::widest_pin`) is planned again keeping the pin
//! (`lower::simd::WidestPin::Kept`), its emulated module dropped and what its plan and emission
//! hold of the ledger released. Only the selection is made again: the entry is decoded,
//! specialized, ordered and analyzed once, so the kernel is charged what any compile of it is
//! charged, whichever path reached it (`super::Analyzed`). Every other module past the bound, a
//! kernel planned again that still is among them, is made as it weighs: whoever creates its
//! pipeline reads the weight against [`COMPILED_STAGE_WEIGHT`] (`metal2vulkan-runtime` compiles
//! such a kernel only in a helper process the consumer caps).
//!
//! **The compiled bound.** [`COMPILED_STAGE_WEIGHT`] is the heaviest entry measured to compile
//! within host-sized memory: on ANV, Mesa 26.2.6, on the Iris Xe of RPL-P, compute modules created
//! cold through `vkCreateComputePipelines` (job m3-max-26147) of entry weights 73,814, 75,886 and
//! 99,109 took 45 s, 47 s and 132 s and peak resident sets of 1.76, 2.24 and 3.65 GiB, and one of
//! entry weight 532,213 passed 16 GiB at 358 s without finishing, where it was stopped. No margin
//! is taken below it, as no weight above it is measured to compile; it rises when a larger compile
//! is measured within host-sized memory.
//!
//! [`ModuleStats::entry_weight`]: crate::pipeline::ModuleStats::entry_weight

use super::{analyze_one, Planned, Request};
use crate::budget::Meter;
use crate::device::DeviceProfile;
use crate::error::Error;
use crate::lower::emit::Emitted;
use crate::lower::simd::WidestPin;

/// The heaviest entry weight ([`ModuleStats::entry_weight`]) measured to compile within host-sized
/// memory (the module docs' compiled bound). A stage that weighs more is one whose creation may
/// take the host's memory.
///
/// [`ModuleStats::entry_weight`]: crate::pipeline::ModuleStats::entry_weight
pub const COMPILED_STAGE_WEIGHT: u64 = 99_109;

/// The entry weight past which a kernel is planned again keeping its pin: the measured one in
/// production ([`StageBounds::MEASURED`]), another in a test, so it reaches the bound with a small
/// module.
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

/// `r`'s function planned and emitted with its repeated regions rolled, with its entry weight:
/// planned again keeping the widest mode's pin where only that pin's emulation passed `bounds`
/// (the module docs).
pub(super) fn emit(
    r: &Request<'_>,
    p: &DeviceProfile,
    bounds: StageBounds,
    m: &mut Meter,
) -> Result<(Planned, Emitted, u64), Error> {
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
        // A driver that keeps a routine compiles it once: repeated regions are written once, as
        // kept routines.
        m.weighted(crate::pipeline::compile::weights::ROLL, |m| {
            crate::spirv::outline::outline_entry(&mut emitted.module, m)
        })?;
        let weight = crate::spirv::roll::entry_weight(&emitted.module, m)?;
        if weight <= bounds.compiled || widest == WidestPin::Kept || !plan.families.simd.widest_pin
        {
            return Ok((analyzed.planned(plan, kernel), emitted, weight));
        }
        // The emulated plan and module go, and with them what they hold of the ledger.
        drop((plan, emitted));
        let held = m.ledger().saturating_sub(before);
        m.release(usize::try_from(held).unwrap_or(usize::MAX));
        widest = WidestPin::Kept;
    }
}
