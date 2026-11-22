//! The stage bound: every module weighed, and a kernel kept under the compiled bound where a lighter
//! plan of it is.
//!
//! A driver's compiler takes memory that grows faster than the stage, in the process that creates
//! the pipeline, so one heavy creation may take the host's memory. No Vulkan query states how large
//! a stage a device's compiler takes, so the bound is one for every device.
//!
//! Every module's entry is weighed after the repeated regions are rolled
//! (`spirv::roll::bound_weight`: the entry's instructions with each call counted as its callee's
//! body, as the driver inlines every call, and each subgroup instruction counted as
//! `spirv::roll::SUBGROUP_WEIGHT` instructions), and the weight is the module's
//! ([`ModuleStats::entry_weight`]). A kernel past the bound that took the `Emulated` simdgroup tier
//! only in place of the `Direct` tier's pin to the device's widest mode (`lower::simd::widest_pin`)
//! is planned again keeping the pin (`lower::simd::WidestPin::Kept`), and both modules are held:
//! the re-plan takes the pinned module where it weighs less than the emulated one by the stage
//! weight; otherwise the emulated one goes on. The other plan and module are dropped and what they
//! hold of the ledger released. Only the selection is made again: the entry is decoded,
//! specialized, ordered and analyzed once, so the kernel is charged what any compile of it is
//! charged, whichever path reached it (`super::Analyzed`). Every other module past the bound, and
//! the one a re-plan takes where it still is among them, is made as it weighs: whoever creates its
//! pipeline reads the weight against [`COMPILED_STAGE_WEIGHT`] (`metal2vulkan-runtime` compiles
//! such a kernel only in a helper process the consumer caps).
//!
//! **The compiled bound.** [`COMPILED_STAGE_WEIGHT`] is 39,014: the heaviest kernel measured to
//! compile in the process that creates its pipeline within 512 MiB of that process's resident
//! growth on every driver measured. Made cold (no pipeline cache, the driver's shader cache off)
//! and alone in a process of its own, a real program's kernel of that weight grew the process by
//! 350 MiB in 6.8 s on ANV (Mesa 26.2.6, the Iris Xe of RPL-P; job m3-max-66048), 148 MiB in
//! 1.8 s on NVIDIA's driver (620.70.06, an RTX 3090; job m3-max-66050) and 80 MiB in 0.55 s on
//! RADV (Mesa 26.2.6, a Radeon 780M; job m3-max-66049). Above it, on the same Iris Xe, an older
//! build's compute modules of entry weights 73,814, 75,886 and 99,109, created cold through
//! `vkCreateComputePipelines`, took 45 s, 47 s and 132 s and peak resident sets of 1.76, 2.24 and
//! 3.65 GiB (job m3-max-26147), and the consumer's helper compiled kernels of entry weights
//! 121,523 and 128,372 within peak resident sets of 2,620 and 2,763 MiB and those of 186,065 to
//! 424,132 passed its 3 GiB cap (job m3-max-65807). No margin is taken below it: it rises when a
//! heavier kernel is measured to compile within 512 MiB. It is the kernels' bound:
//! `metal2vulkan-runtime` refuses a render stage only past a bound of its own, as no helper
//! compiles a render stage (its README, "Kernels past the stage bound"). The helper's kernels of
//! 121,523 to 424,132 hold no subgroup instruction as the RTX 3090's and the Iris Xe's profiles
//! lower them, so each weighs the same with or without `spirv::roll::SUBGROUP_WEIGHT` (jobs
//! m3-max-68033 and m3-max-68034).
//!
//! [`ModuleStats::entry_weight`]: crate::pipeline::ModuleStats::entry_weight

use super::{analyze_one, Analyzed, Planned, Request};
use crate::budget::Meter;
use crate::device::DeviceProfile;
use crate::error::Error;
use crate::library::KernelFacts;
use crate::lower::emit::Emitted;
use crate::lower::select::Plan;
use crate::lower::simd::WidestPin;

/// The heaviest entry weight ([`ModuleStats::entry_weight`]) of a kernel measured to compile within
/// 512 MiB of its creating process's resident growth on every driver measured (the module docs'
/// compiled bound). A kernel that weighs more is one whose creation may take the host's memory.
///
/// [`ModuleStats::entry_weight`]: crate::pipeline::ModuleStats::entry_weight
pub const COMPILED_STAGE_WEIGHT: u64 = 39_014;

/// The weight past which a kernel is planned again keeping its pin: the compiled one in
/// production ([`StageBounds::MEASURED`]), another in a test, so it reaches the bound with a small
/// module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StageBounds {
    /// Every stage's bound.
    pub(crate) compiled: u64,
}

impl StageBounds {
    /// The compiled bound (the module docs).
    pub(crate) const MEASURED: StageBounds = StageBounds {
        compiled: COMPILED_STAGE_WEIGHT,
    };
}

/// `r`'s function planned and emitted with its repeated regions rolled, with its bound weight:
/// where only the widest mode's pin's emulation passed `bounds`, planned again keeping the pin and
/// the lighter of the two modules taken (the module docs).
pub(super) fn emit(
    r: &Request<'_>,
    p: &DeviceProfile,
    bounds: StageBounds,
    m: &mut Meter,
) -> Result<(Planned, Emitted, u64), Error> {
    let analyzed = analyze_one(r, p, m)?;
    let weighed = made(&analyzed, p, WidestPin::Weighed, m)?;
    if weighed.weight <= bounds.compiled || !weighed.plan.families.simd.widest_pin {
        return Ok(weighed.planned(analyzed));
    }
    // Past the bound only as the pin's emulation: the plan keeping the pin, both held.
    let emulated = weighed;
    let pinned = made(&analyzed, p, WidestPin::Kept, m)?;
    let (on, off) = if pinned.weight < emulated.weight {
        (pinned, emulated)
    } else {
        (emulated, pinned)
    };
    // The other plan and module go, and with them what they hold of the ledger.
    let held = off.held;
    drop(off);
    m.release(usize::try_from(held).unwrap_or(usize::MAX));
    Ok(on.planned(analyzed))
}

/// `r`'s function planned keeping the widest mode's pin or weighing it as `widest` says, emitted as
/// [`emit`] makes each plan it compares, with its module's bound weight.
#[cfg(test)]
pub(super) fn emit_as(
    r: &Request<'_>,
    p: &DeviceProfile,
    widest: WidestPin,
    m: &mut Meter,
) -> Result<(Plan, crate::spirv::ir::TypedModule, u64), Error> {
    let analyzed = analyze_one(r, p, m)?;
    let Made {
        plan,
        emitted,
        weight,
        ..
    } = made(&analyzed, p, widest, m)?;
    Ok((plan, emitted.module, weight))
}

/// One plan of an analyzed kernel, its module, the module's bound weight, and what the plan and
/// module hold of the ledger.
struct Made {
    plan: Plan,
    kernel: KernelFacts,
    emitted: Emitted,
    weight: u64,
    held: u64,
}

impl Made {
    fn planned(self, analyzed: Analyzed) -> (Planned, Emitted, u64) {
        (
            analyzed.planned(self.plan, self.kernel),
            self.emitted,
            self.weight,
        )
    }
}

/// `analyzed` planned keeping the widest mode's pin or weighing it as `widest` says, emitted with
/// its repeated regions rolled, and weighed.
fn made(
    analyzed: &Analyzed,
    p: &DeviceProfile,
    widest: WidestPin,
    m: &mut Meter,
) -> Result<Made, Error> {
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
        crate::spirv::hoist::hoist_uniform_loads(&mut emitted.module, m)
    })?;
    // A structurizer variable whose value no later trip of a loop reads is ended at each
    // trip's top, so the driver carries none of it around the loop.
    crate::spirv::carry::end_dead_carries(&mut emitted.module, m)?;
    let weight = crate::spirv::roll::bound_weight(&emitted.module, m)?;
    Ok(Made {
        plan,
        kernel,
        emitted,
        weight,
        held: m.ledger().saturating_sub(before),
    })
}
