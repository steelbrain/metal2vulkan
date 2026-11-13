//! The stage bound: a module kept under the size its driver's compiler takes intact
//! (`DriverWorkaround::SerializedStageWrapsLargeModules`).
//!
//! Mesa's common pipeline runtime (`vk_pipeline.c`), which ANV's pipelines go through, writes each
//! stage after the driver's preprocessing into a precompiled shader (`nir_serialize`) and reads it
//! back (`nir_deserialize`) before compiling it, on every pipeline it makes, cached or not. The
//! serialized form names a value by a 20-bit index, and the bound (`MAX_OBJECT_IDS`, 2^20 objects:
//! values, blocks, variables) is held only by an assertion a release build leaves out: past it the
//! indices wrap, the reader takes other objects for them, and the driver faults inside
//! `vkCreateComputePipelines` (Mesa 26.2.6, ANV on the Iris Xe).
//!
//! On a profile with the defect, a module whose entry weighs more than [`SERIALIZED_STAGE_WEIGHT`]
//! (`spirv::roll::entry_weight`: the entry's instructions with each call counted as its callee's
//! body, as the driver inlines every call) after the repeated regions are rolled is past the bound:
//!
//! - a kernel that took the `Emulated` simdgroup tier only in place of the `Direct` tier's pin to
//!   the device's widest mode (`lower::simd::widest_pin`) is planned again keeping the pin
//!   (`lower::simd::WidestPin::Kept`), its emulated module dropped and what its plan and emission
//!   hold of the ledger released. Only the selection is made again: the entry is decoded,
//!   specialized, ordered and analyzed once, so the kernel is charged what any compile of it is
//!   charged, whichever path reached it (`super::Analyzed`);
//! - every other module past it, a kernel planned again that still is among them, is refused
//!   (`Unsupported`, `Feature::StageSize`, `Needs::DriverDefect`): Metal compiles it, so the
//!   refusal is the device's gap, recorded in `docs/FAST-PATHS.md`'s known gaps.
//!
//! **The bound.** The driver's object count is not the module's: it counts the stage after its
//! own inlining and optimization. Measured on ANV, Mesa 26.2.6, on the Iris Xe of RPL-P (the count
//! each serialized stage holds, read where the driver reads it back), over 22 compute stages of
//! entry weights 1,667 to 1,565,083: 0.14 to 0.99 objects per instruction of the entry's weight,
//! 0.70 for the heaviest, whose 1,096,128 objects faulted the driver. [`SERIALIZED_STAGE_WEIGHT`]
//! is three quarters of 2^20: a stage at the bound stays under the driver's 2^20 objects at up to
//! 4/3 objects per instruction, a third above the largest ratio measured.

use super::{analyze_one, Planned, Request};
use crate::budget::Meter;
use crate::device::profile::DriverWorkaround;
use crate::device::DeviceProfile;
use crate::error::{Error, Feature, Unsupported};
use crate::lower::emit::Emitted;
use crate::lower::simd::WidestPin;

/// The heaviest entry ([`crate::spirv::roll::entry_weight`]) a profile with
/// `DriverWorkaround::SerializedStageWrapsLargeModules` is handed (the module docs' bound).
pub(crate) const SERIALIZED_STAGE_WEIGHT: u64 = 3 << 18;

/// `r`'s function planned and emitted with its repeated regions rolled, under the plan's word
/// estimate `word_cap` and, on a profile with the defect, under the stage bound `weight_cap` (the
/// module docs): planned again keeping the widest mode's pin where only that pin's emulation passed
/// it, else refused.
pub(super) fn emit(
    r: &Request<'_>,
    p: &DeviceProfile,
    (word_cap, weight_cap): (u64, u64),
    m: &mut Meter,
) -> Result<(Planned, Emitted), Error> {
    let defect = DriverWorkaround::SerializedStageWrapsLargeModules;
    let bounded = p.workarounds().contains(&defect);
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
        if !bounded || crate::spirv::roll::entry_weight(&emitted.module, m)? <= weight_cap {
            return Ok((analyzed.planned(plan, kernel), emitted));
        }
        if widest == WidestPin::Kept || !plan.families.simd.widest_pin {
            return Err(Unsupported::driver_defect(Feature::StageSize, defect).into());
        }
        // The emulated plan and module go, and with them what they hold of the ledger.
        drop((plan, emitted));
        let held = m.ledger().saturating_sub(before);
        m.release(usize::try_from(held).unwrap_or(usize::MAX));
        widest = WidestPin::Kept;
    }
}
