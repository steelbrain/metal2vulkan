//! The stage bounds: a module kept under the sizes its driver's compiler takes intact
//! (`DriverWorkaround::SerializedStageWrapsLargeModules`) and in host-sized memory
//! (`DriverWorkaround::CompilerMemoryGrowsWithStageSize`), where `Device::new`'s profile names
//! those defects; and on a profile `Device::from_queries` built, every module kept under the
//! compiled bound on every device (`device::profile::StageBound`).
//!
//! Mesa's common pipeline runtime (`vk_pipeline.c`), which ANV's pipelines go through, writes each
//! stage after the driver's preprocessing into a precompiled shader (`nir_serialize`) and reads it
//! back (`nir_deserialize`) before compiling it, on every pipeline it makes, cached or not. The
//! serialized form names a value by a 20-bit index, and the bound (`MAX_OBJECT_IDS`, 2^20 objects:
//! values, blocks, variables) is held only by an assertion a release build leaves out: past it the
//! indices wrap, the reader takes other objects for them, and the driver faults inside
//! `vkCreateComputePipelines` (Mesa 26.2.6, ANV on the Iris Xe). Below that bound, the driver's
//! compiler takes memory that grows faster than the stage, in the process that creates the
//! pipeline, so one creation past the largest stage measured to compile may take the host's
//! memory.
//!
//! On a profile with either defect, a module whose entry weighs more than its bound
//! (`spirv::roll::entry_weight`: the entry's instructions with each call counted as its callee's
//! body, as the driver inlines every call) after the repeated regions are rolled is past it:
//!
//! - a kernel that took the `Emulated` simdgroup tier only in place of the `Direct` tier's pin to
//!   the device's widest mode (`lower::simd::widest_pin`) is planned again keeping the pin
//!   (`lower::simd::WidestPin::Kept`), its emulated module dropped and what its plan and emission
//!   hold of the ledger released. Only the selection is made again: the entry is decoded,
//!   specialized, ordered and analyzed once, so the kernel is charged what any compile of it is
//!   charged, whichever path reached it (`super::Analyzed`);
//! - every other module past it, a kernel planned again that still is among them, is refused
//!   (`Unsupported`, `Feature::StageSize`, `Needs::DriverDefect`, naming the serialized stage's
//!   defect past its bound and the compiler's memory below it, and as its symbol the module's
//!   entry weight and the bound it passed): Metal compiles it, so the refusal is the device's
//!   gap, recorded in `docs/FAST-PATHS.md`'s known gaps.
//!
//! **Every device.** A profile `Device::from_queries` built reads no driver identity, so it names
//! no defect: it holds every stage, on every device, under the compiled bound, the heaviest stage
//! measured to compile on the slowest compiler measured, one form for every device. A kernel past
//! it is planned again keeping the widest mode's pin as above, and every other module past it is
//! refused as `Needs::StageBound`, with the same symbol: a device whose compiler would take a
//! heavier stage is refused it all the same, a known gap of `metal2vulkan-rt`.
//!
//! **The serialized bound.** The driver's object count is not the module's: it counts the stage
//! after its own inlining and optimization. Measured on ANV, Mesa 26.2.6, on the Iris Xe of RPL-P
//! (the count each serialized stage holds, read where the driver reads it back), over 22 compute
//! stages: the heaviest, of entry weight 1,603,989, held 1,096,128 objects, 0.683 per instruction
//! of its weight, and faulted the driver; the other 21 held 0.14 to 0.99 objects per instruction
//! of a count of the entry's instructions that ran 1.9 to 2.7% below the entry weight on the five
//! modules both counted. [`SERIALIZED_STAGE_WEIGHT`] is three quarters of 2^20: a stage at the
//! bound stays under the driver's 2^20 objects at up to 4/3 objects per instruction, a third above
//! the largest ratio measured.
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
use crate::device::profile::{DriverWorkaround, StageBound};
use crate::device::DeviceProfile;
use crate::error::{DiagName, Error, Feature, Needs, Unsupported};
use crate::lower::emit::Emitted;
use crate::lower::simd::WidestPin;

/// The heaviest entry ([`crate::spirv::roll::entry_weight`]) a profile with
/// `DriverWorkaround::SerializedStageWrapsLargeModules` is handed (the module docs' serialized
/// bound).
pub(crate) const SERIALIZED_STAGE_WEIGHT: u64 = 3 << 18;

/// The heaviest entry ([`crate::spirv::roll::entry_weight`]) a profile with
/// `DriverWorkaround::CompilerMemoryGrowsWithStageSize`, or one whose every stage is bounded
/// (`StageBound::Every`), is handed: the largest measured to compile (the module docs' compiled
/// bound).
pub(crate) const COMPILED_STAGE_WEIGHT: u64 = 99_109;

/// The entry weights past which a profile with each defect, or one whose every stage is bounded, is
/// not handed a module: the measured ones in production ([`StageBounds::MEASURED`]), others in a
/// test, so it reaches each bound with a small module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StageBounds {
    /// `DriverWorkaround::SerializedStageWrapsLargeModules`' bound.
    pub(crate) serialized: u64,
    /// `DriverWorkaround::CompilerMemoryGrowsWithStageSize`'s bound, and every stage's on a
    /// profile whose every stage is bounded.
    pub(crate) compiled: u64,
}

impl StageBounds {
    /// The bounds measured on the defects' drivers (the module docs).
    pub(crate) const MEASURED: StageBounds = StageBounds {
        serialized: SERIALIZED_STAGE_WEIGHT,
        compiled: COMPILED_STAGE_WEIGHT,
    };

    /// `p`'s bounds, each with what a module past it is refused as: on a profile whose every
    /// stage is bounded, the compiled bound alone; else its defects', the serialized stage's
    /// first, so a module past both is refused for the one that faults the driver.
    fn of(self, p: &DeviceProfile) -> Vec<(Past, u64)> {
        if p.stage_bound() == StageBound::Every {
            return vec![(Past::Every, self.compiled)];
        }
        [
            (
                DriverWorkaround::SerializedStageWrapsLargeModules,
                self.serialized,
            ),
            (
                DriverWorkaround::CompilerMemoryGrowsWithStageSize,
                self.compiled,
            ),
        ]
        .into_iter()
        .filter(|(defect, _)| p.workarounds().contains(defect))
        .map(|(defect, bound)| (Past::Defect(defect), bound))
        .collect()
    }
}

/// What a module past a bound is refused as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Past {
    /// The driver's stage-size defect (`Needs::DriverDefect`).
    Defect(DriverWorkaround),
    /// The bound every stage is held under (`Needs::StageBound`).
    Every,
}

/// `r`'s function planned and emitted with its repeated regions rolled, under the plan's word
/// estimate `word_cap` and, on a profile with a stage bound's defect or whose every stage is
/// bounded, under `bounds` (the module docs): planned again keeping the widest mode's pin where
/// only that pin's emulation passed one, else refused as the first bound the module passes, naming
/// the module's entry weight and that bound ([`past_bound`]).
pub(super) fn emit(
    r: &Request<'_>,
    p: &DeviceProfile,
    (word_cap, bounds): (u64, StageBounds),
    m: &mut Meter,
) -> Result<(Planned, Emitted), Error> {
    let bounded = bounds.of(p);
    let analyzed = analyze_one(r, p, m)?;
    let mut widest = WidestPin::Weighed;
    loop {
        let before = m.ledger();
        let (plan, kernel) = analyzed.select(p, widest, m)?;
        plan.estimate.check(word_cap)?;
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
            crate::spirv::hoist::hoist_uniform_loads(&mut emitted.module, p.containment(), m)
        })?;
        let past = match bounded.is_empty() {
            true => None,
            false => {
                let weight = crate::spirv::roll::entry_weight(&emitted.module, m)?;
                bounded
                    .iter()
                    .find(|&&(_, cap)| weight > cap)
                    .map(|&(past, cap)| (past, weight, cap))
            }
        };
        let Some((past, weight, bound)) = past else {
            return Ok((analyzed.planned(plan, kernel), emitted));
        };
        if widest == WidestPin::Kept || !plan.families.simd.widest_pin {
            return Err(refused(past, weight, bound));
        }
        // The emulated plan and module go, and with them what they hold of the ledger.
        drop((plan, emitted));
        let held = m.ledger().saturating_sub(before);
        m.release(usize::try_from(held).unwrap_or(usize::MAX));
        widest = WidestPin::Kept;
    }
}

/// The refusal of a module of entry weight `weight` past `bound`, as `past` says.
fn refused(past: Past, weight: u64, bound: u64) -> Error {
    let named = past_bound(weight, bound);
    match past {
        Past::Defect(defect) => Unsupported::driver_defect(Feature::StageSize, defect)
            .naming(named)
            .into(),
        Past::Every => Error::Unsupported(Unsupported {
            feature: Feature::StageSize,
            needs: Needs::StageBound,
            location: None,
            symbol: Some(named),
        }),
    }
}

/// What a stage's refusal names as its symbol: the module's entry weight and the bound it passed.
fn past_bound(weight: u64, bound: u64) -> DiagName {
    DiagName::new(format!("entry weight {weight}, past the bound {bound}"))
}
