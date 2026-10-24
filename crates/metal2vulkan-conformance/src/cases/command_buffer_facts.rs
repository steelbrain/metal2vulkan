//! What a command buffer says about itself: `device`, `commandQueue`, the four timing getters and
//! `resourceStateCommandEncoder`.
//!
//! Seven of `MTLCommandBuffer`'s eight unexercised items, in one case, because they are one
//! experiment: a command buffer's owners and its timings are facts about **that** command buffer,
//! and a case that read them from several would be reading several command buffers.
//! [`crate::command_buffer_facts`] says what is recorded and why a time never is.
//!
//! # The timings are the subject, and they are read as orderings
//!
//! Nothing in this tree held an implementation to the shape of `GPUStartTime`, `GPUEndTime`,
//! `kernelStartTime` and `kernelEndTime`. The runtime builds all four from its own clocks and
//! announces three different degradations over them, and the only thing watching was a lane
//! assertion that one of them eventually stops being zero. This case states the contract instead:
//! zero before the work, positive after, and each interval ordered, with the GPU's interval not
//! beginning or ending before the kernel's.
//!
//! **Two of those are a measurement, not a known answer.** Whether Metal's `GPUStartTime` is at or
//! after its `kernelStartTime`, and whether its `GPUEndTime` is at or after its `kernelEndTime`,
//! is what the recording will say. They are recorded because they are the two an implementation
//! can get wrong without any other row moving, and the case is run several times so a pair that
//! flaps says so in its own recording rather than passing as whichever run came last.
//!
//! # Why the GPU pair may be refused, and why that is recorded rather than smoothed
//!
//! `GPUStartTime` and `GPUEndTime` need timestamps from the device. The runtime refuses them typed
//! where the queue family writes none, and the case records that refusal against Metal's answer —
//! a real gap, stated as one, rather than four booleans computed from two zeros that would have
//! read as agreement.
//!
//! # Why it is safe
//!
//! Three command buffers: two that only make a command buffer and read its queue's label, one that
//! fills a few bytes of the case's own buffer, and one that opens a resource state encoder and ends
//! it with nothing encoded. No shader, no draw, no dispatch, no loop, no mapping, and every wait is
//! bounded.

use crate::allowance::Allowance;
use crate::case::{made, Abort, Case, Input, MetalSource, Side};
use crate::replay::notice_allowance;
use crate::transcript::Sites;
use metal2vulkan_runtime::Degradation;

/// The case's name.
pub const COMMAND_BUFFER_FACTS: &str = "command-buffer-facts";

/// The notices the runtime can announce over the GPU times, and the whole of what this case takes.
///
/// `UncalibratedGpuTimes` is **not** here, though the runtime can announce it: on this oracle it
/// never fires, and a declaration that outlives its own reason keeps asserting that somebody
/// examined a notice nobody has looked at since. A venue that does announce it reads as
/// `UndeclaredNotice`, naming the code and the case, which is the loud form and the moment to look
/// at it.
const GPU_TIME_NOTICES: [Degradation; 1] = [Degradation::GpuTimesFromHost];

/// A case over one command buffer's own answers.
pub struct CommandBufferFactsCase {
    name: &'static str,
}

/// The case.
pub fn command_buffer_facts() -> CommandBufferFactsCase {
    CommandBufferFactsCase {
        name: COMMAND_BUFFER_FACTS,
    }
}

/// Every case of the family.
pub fn all() -> Vec<Box<dyn Case>> {
    vec![Box::new(command_buffer_facts())]
}

impl Case for CommandBufferFactsCase {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        // What the case feeds is the shape of the experiment, which is the side's own: the bytes
        // the timed command buffer fills, and the two queue labels. They are stated here so a
        // change to either is a change to the case's inputs.
        Ok(vec![Input::bytes(
            "timed-bytes-and-queue-labels",
            crate::command_buffer_facts::TIMERS.join(",").as_bytes(),
        )])
    }

    fn record_runs(&self) -> u32 {
        // Two of the recorded booleans are an ordering of two clocks, and a clock read is a race
        // however wide the window. More runs do not remove that; they record it, because a
        // recording holds the set of outcomes with their frequencies, so a row that flaps says so
        // instead of passing as whichever run came last.
        5
    }

    fn allowances(&self) -> Vec<Allowance> {
        // The runtime builds `GPUStartTime` and `GPUEndTime` from its own clocks and announces how
        // on the venues that need it. The case takes the notice and compares every value exactly;
        // what it may not do is take a different *number* because of one, and it does not.
        // `GpuTimesFromHost`: the runtime answers the GPU times from the host clock where the
        // driver's own timestamps wait for every queue's (MoltenVK on Apple's GPUs).
        // `UncalibratedGpuTimes`: it places them without calibrated timestamps where the device
        // offers none. The orderings this case compares do not depend on either, which is why the
        // case can take the notice and still compare every value exactly.
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the
        // notices site alone, so every value of this family still compares against Metal's
        // exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09): it fired for this case and each
        // was divergent on that field and nothing else (replay-nvidia.log). On MoltenVK, where
        // the chip's GPU shares the CPU's memory, it cannot fire at all, which is why no
        // recording of this suite declares it.
        std::iter::once(crate::replay::shared_memory_notice())
            .chain(
                GPU_TIME_NOTICES
                    .iter()
                    .map(|code| notice_allowance(code.code())),
            )
            .collect()
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        // The call's own site and the facts take **different** names: two siblings under one name
        // are paired by whichever namesake the comparison reaches first, which is how a round
        // reported six cases divergent having compared no byte of either side. The suite holds it
        // for every recording (`tests::no_recorded_site_shares_its_name_with_a_sibling`), and it
        // caught this case's first recording.
        let facts = made(sites, "command-buffer", side.command_buffer_facts())?;
        sites.push(facts.node("facts"));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The case states its inputs, names itself once, and declares the two notices the runtime can
    /// announce over the GPU times, the venue's own memory notice, and nothing else.
    ///
    /// A fourth notice slipping in would be a value the case took for a reason nobody wrote down,
    /// which is what this count is for.
    #[test]
    fn the_case_states_its_inputs_and_only_the_gpu_time_notices() {
        let cases = all();
        assert_eq!(cases.len(), 1);
        let case = &cases[0];
        assert_eq!(case.name(), COMMAND_BUFFER_FACTS);
        assert!(!case.inputs().expect("the inputs").is_empty());
        let allowed = case.allowances();
        // The two GPU-time notices, and the one a card without unified memory announces for the
        // `Shared` buffer the case reads its answer back through.
        assert_eq!(allowed.len(), GPU_TIME_NOTICES.len() + 1);
        // And the case records more than one run, because two of its booleans read a clock.
        assert!(case.record_runs() > 1);
    }
}
