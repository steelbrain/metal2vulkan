//! The stage bound: every module kept under the heaviest stage measured to compile within
//! host-sized memory, on every device.
//!
//! A driver's shader compiler can take memory that grows faster than the stage it compiles, inside
//! the process that creates the pipeline. Measured on ANV, Mesa 26.2.6, on the Iris Xe of RPL-P:
//! compute modules created cold through `vkCreateComputePipelines` (job m3-max-26147) of entry
//! weights 73,814, 75,886 and 99,109 took 45 s, 47 s and 132 s and peak resident sets of 1.76,
//! 2.24 and 3.65 GiB, and one of entry weight 532,213 passed 16 GiB at 358 s without finishing,
//! where it was stopped. Far above that, the same driver faults: Mesa's common pipeline runtime
//! (`vk_pipeline.c`) writes each stage into a serialized form that names an object by a 20-bit
//! index, which wraps past 2^20 objects (a stage of entry weight 1,603,989 faulted inside
//! `vkCreateComputePipelines`).
//!
//! The bound is one form for every device: the heaviest stage measured to compile on the slowest
//! compiler measured ([`COMPILED_STAGE_WEIGHT`]). No margin is taken below it, as no weight above
//! it is measured to compile there; it rises when a larger compile is measured within host-sized
//! memory. A device whose compiler would take a heavier stage is refused it all the same, a
//! known gap (`docs/FAST-PATHS.md`).
//!
//! A module whose entry weighs more than the bound (`spirv::roll::entry_weight`: the entry's
//! instructions with each call counted as its callee's body, as the driver inlines every call)
//! after the repeated regions are rolled is past it:
//!
//! - a kernel that took the `Emulated` simdgroup tier only in place of the `Direct` tier's pin to
//!   the device's widest mode (`lower::simd::widest_pin`, which the subgroup size range decides)
//!   is planned again keeping the pin (`lower::simd::WidestPin::Kept`), its emulated module
//!   dropped and what its plan and emission hold of the ledger released. Only the selection is
//!   made again: the entry is decoded, specialized, ordered and analyzed once, so the kernel is
//!   charged what any compile of it is charged, whichever path reached it (`super::Analyzed`);
//! - every other module past it, a kernel planned again that still is among them, is refused
//!   (`Unsupported`, `Feature::StageSize`, `Needs::StageBound`, and as its symbol the module's
//!   entry weight and the bound it passed).

use super::{analyze_one, Planned, Request};
use crate::budget::Meter;
use crate::device::DeviceProfile;
use crate::error::{DiagName, Error, Feature, Needs, Unsupported};
use crate::lower::emit::Emitted;
use crate::lower::simd::WidestPin;

/// The heaviest entry ([`crate::spirv::roll::entry_weight`]) any device is handed: the largest
/// measured to compile within host-sized memory (the module docs).
pub(crate) const COMPILED_STAGE_WEIGHT: u64 = 99_109;

/// `r`'s function planned and emitted with its repeated regions rolled, under the plan's word
/// estimate `word_cap` and the stage bound `bound` ([`COMPILED_STAGE_WEIGHT`] in production, a
/// smaller one in a test, so it reaches the bound with a small module): planned again keeping the
/// widest mode's pin where only that pin's emulation passed it, else refused, naming the module's
/// entry weight and the bound ([`past_bound`]).
pub(super) fn emit(
    r: &Request<'_>,
    p: &DeviceProfile,
    (word_cap, bound): (u64, u64),
    m: &mut Meter,
) -> Result<(Planned, Emitted), Error> {
    let analyzed = analyze_one(r, p, m)?;
    let mut widest = WidestPin::Weighed;
    loop {
        let before = m.ledger();
        let (plan, kernel) = analyzed.select(p, widest, m)?;
        plan.estimate.check(word_cap)?;
        let mut emitted = crate::lower::emit::emit_module(&analyzed.spec, &plan, m)?;
        // Every driver compiles each repeated region of the entry again; above the measured size,
        // repeated regions are written once, as a loop.
        crate::spirv::roll::roll_entry(&mut emitted.module, m)?;
        // A load of a uniform view that every trip of its loop reads alike leaves the loop.
        crate::spirv::hoist::hoist_uniform_loads(&mut emitted.module, p.containment(), m)?;
        let weight = crate::spirv::roll::entry_weight(&emitted.module, m)?;
        if weight <= bound {
            return Ok((analyzed.planned(plan, kernel), emitted));
        }
        if widest == WidestPin::Kept || !plan.families.simd.widest_pin {
            return Err(past_bound(weight, bound));
        }
        // The emulated plan and module go, and with them what they hold of the ledger.
        drop((plan, emitted));
        let held = m.ledger().saturating_sub(before);
        m.release(usize::try_from(held).unwrap_or(usize::MAX));
        widest = WidestPin::Kept;
    }
}

/// The refusal of a stage of entry weight `weight`, past the bound `bound`: its symbol names both.
fn past_bound(weight: u64, bound: u64) -> Error {
    let named = format!("entry weight {weight}, past the bound {bound}");
    Error::Unsupported(Unsupported {
        feature: Feature::StageSize,
        needs: Needs::StageBound,
        location: None,
        symbol: Some(DiagName::new(named)),
    })
}
