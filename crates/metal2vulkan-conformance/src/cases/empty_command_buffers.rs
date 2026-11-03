//! What a command buffer with nothing encoded answers, alone on an idle queue and committed behind
//! a long fill of its queue: its statuses in its handlers and after its wait, the shape of its four
//! timing getters, and its place in its queue's order. [`crate::empty_command_buffers`] says what
//! is recorded and why a time never is.
//!
//! # Why this is a case of its own
//!
//! The runtime completes such a command buffer with no submission at all, once the work its queue
//! submitted before it completed (`command_buffer/submit.rs`). Nothing else in the suite commits a
//! command buffer with nothing encoded behind work, so this is where that path meets Metal's
//! answer: the status a handler reads, the order the completed handlers run in, and where the GPU
//! times sit against the fill's.
//!
//! # Why it is safe
//!
//! Three command buffers: an empty one, a fill of [`crate::empty_command_buffers::FILL_BYTES`] of a
//! `Private` buffer of the case's own, and an empty one behind it. No shader, no draw, no dispatch,
//! no loop, no mapping; every status a handler reads is sent on a channel read with a bound, and
//! every wait is for work that ends on its own.

use crate::allowance::Allowance;
use crate::case::{Abort, Case, Input, MetalSource, Side};
use crate::empty_command_buffers::{EMPTY, FILL, FILL_BYTES};
use crate::replay::notice_allowance;
use crate::transcript::{Node, Observed, Sites};
use metal2vulkan_runtime::Degradation;

/// The case's name.
pub const EMPTY_COMMAND_BUFFERS: &str = "empty-command-buffers";

/// The site the row sits at.
const AT: &str = "empty-command-buffers";

/// A case over what command buffers with nothing encoded answer.
pub struct EmptyCommandBuffersCase {
    name: &'static str,
}

/// Every case of the family.
pub fn all() -> Vec<Box<dyn Case>> {
    vec![Box::new(EmptyCommandBuffersCase {
        name: EMPTY_COMMAND_BUFFERS,
    })]
}

impl Case for EmptyCommandBuffersCase {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        Ok(vec![
            Input::bytes("fill-bytes", &FILL_BYTES.to_le_bytes()),
            Input::bytes("handler-names", format!("{FILL},{EMPTY}").as_bytes()),
        ])
    }

    fn record_runs(&self) -> u32 {
        // The timing flags order two clocks' readings, which is a race however wide the margin:
        // more runs record it, as the recording holds the set of outcomes with their frequencies.
        5
    }

    fn allowances(&self) -> Vec<Allowance> {
        // `GpuTimesFromHost`: the runtime answers the fill's GPU times from the host clock where
        // the driver's own timestamps wait for every queue's (MoltenVK on Apple's GPUs), and says
        // so once. The orderings this case compares hold for either source, so every value still
        // compares exactly; the notice site alone takes it.
        vec![notice_allowance(Degradation::GpuTimesFromHost.code())]
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        match side.empty_command_buffers() {
            Ok(facts) => sites.push(facts.node(AT)),
            // Every fact is in the one call, so a failure ends the row.
            Err(failure) => sites.push(Node::leaf(AT, Observed::value(failure.recorded()))),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The case states its inputs, names itself once, takes the GPU-time notice alone, and
    /// records more than one run.
    #[test]
    fn the_case_states_its_inputs_and_only_the_gpu_time_notice() {
        let cases = all();
        assert_eq!(cases.len(), 1);
        let case = &cases[0];
        assert_eq!(case.name(), EMPTY_COMMAND_BUFFERS);
        assert!(!case.inputs().expect("the inputs").is_empty());
        assert_eq!(case.allowances().len(), 1);
        assert!(case.record_runs() > 1);
    }
}
