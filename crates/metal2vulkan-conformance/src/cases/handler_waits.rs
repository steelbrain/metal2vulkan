//! The three waits Metal does not come back from: a command buffer's own handler waiting for the
//! transition it is itself inside.
//!
//! A command buffer's scheduled and completed handlers run on threads of Metal's own, and a wait
//! made from one of them **for that same command buffer** is a wait for a transition the handler is
//! in the middle of. Metal does not detect it. The call does not return, the handler's thread stays
//! inside it, and the command buffer whose handler it is can never reach the state the wait is for.
//!
//! So there is no value here and no behaviour: three cases, each one site, each holding the single
//! fact there is — Metal never came back — against our typed refusal at the same site. That is
//! [`Verdict::MetalNeverReturned`](crate::verdict::Verdict::MetalNeverReturned), which is settled
//! before any allowance is consulted, so a case can neither loosen its way into it nor out of the
//! divergence it becomes if we ever answer where Metal did not.
//!
//! # Nothing of this runs on Metal, ever
//!
//! [`crate::metal_side::MetalSide`] does not implement
//! [`Side::handler_wait`] and must not: the call it would make is
//! the call that takes Metal's process with it. Record mode refuses these three rows before
//! anything is encoded ([`crate::case::MetalSource::forbidden`]), and their recordings are written
//! by hand from the probe that died, through [`crate::record::never_returned`] — the only way such
//! an answer is built. A test here holds each committed file to exactly what that function builds
//! from the case's own stated transcript, so a recording cannot drift from the case that states it.
//!
//! Our own leg is safe and bounded both ways: the runtime refuses the wait at once, so the handler
//! returns, the command buffer completes, and the case's own thread waits for that completion with
//! a bound.
//!
//! # What is deliberately not here
//!
//! A **completed** handler's `waitUntilScheduled` is the fourth combination, and the probe says
//! Metal returns from it, scheduling having already happened. It is left out: running it on Metal
//! would stake a record run on that prose, and the suite's standing rule against a shape that may
//! not come back does not bend for a sentence. The runtime's `Ok(())` there stays a `runtime_lane`
//! assertion.
//!
//! The runtime's **bounded** wait (`wait_until_completed_for`) is left out too, for a different
//! reason: Metal has no bounded wait, so its `TimedOut` is not a Metal fact and there is nothing
//! for a conformance case to compare it against.

use crate::case::{Abort, Case, Input, MetalSource, Side};
use crate::handler_waits::{HandlerKind, OwnWait, WaitKind};
use crate::replay::NOTICES_SITE;
use crate::transcript::{Node, Observed, Sites, Transcript};

/// Where the probe that died is written down.
const PROBED_IN: &str = "metal2vulkan-validation runtime_lane::handler_waits, \
                         waits_from_a_command_buffer_s_own_handlers_answer_at_once";

/// The chip the probe died on, as the suite names it.
///
/// It is not this round's oracle, and it cannot become one: making the call is what ends the
/// process, so these three rows can no more be re-taken here than there.
/// [`crate::recording::Provenance::NeverReturned`] holds no chip — a row with no value has no
/// measurement to attribute — so it goes in [`crate::recording::Answer::device`], which
/// [`crate::recording::Answer::measured_elsewhere`] reads, and the report counts these rows among
/// the answers that did not come from this round's oracle.
const PROBED_ON: &str = crate::recording::APPLE_M4;

/// The runs behind the answer: the probe made the call once and did not come back from it.
const PROBE_RUNS: u32 = 1;

/// The command buffer every one of these cases makes, as the input fingerprint names it. One blit
/// filling the first 64 bytes of a shared buffer, committed once: the smallest piece of work that
/// has both transitions to wait for.
const COMMAND_BUFFER: &[u8] = b"one blit filling 64 bytes with 0x11, committed once";

/// A case over one wait a handler makes for its own command buffer.
pub struct OwnHandlerWait {
    name: &'static str,
    wait: OwnWait,
    says: &'static str,
}

/// The three cases: every handler-and-wait pair the probe found Metal does not return from.
pub fn all() -> Vec<OwnHandlerWait> {
    vec![
        OwnHandlerWait {
            name: SCHEDULED_UNTIL_COMPLETED,
            wait: OwnWait::new(HandlerKind::Scheduled, WaitKind::UntilCompleted),
            says: "a scheduled handler's wait for its own command buffer's completion never \
                   returns: Metal does not detect it, the handler's thread stays inside the call, \
                   and the command buffer whose handler it is cannot complete while it is there",
        },
        OwnHandlerWait {
            name: SCHEDULED_UNTIL_SCHEDULED,
            wait: OwnWait::new(HandlerKind::Scheduled, WaitKind::UntilScheduled),
            says: "a scheduled handler's wait for its own command buffer's scheduling never \
                   returns: the transition it waits for is the one it is itself inside",
        },
        OwnHandlerWait {
            name: COMPLETED_UNTIL_COMPLETED,
            wait: OwnWait::new(HandlerKind::Completed, WaitKind::UntilCompleted),
            says: "a completed handler's wait for its own command buffer's completion never \
                   returns: the transition it waits for is the one it is itself inside",
        },
    ]
}

/// A scheduled handler waiting for its own command buffer to complete.
pub const SCHEDULED_UNTIL_COMPLETED: &str = "handler-waits-scheduled-until-completed";

/// A scheduled handler waiting for its own command buffer to be scheduled.
pub const SCHEDULED_UNTIL_SCHEDULED: &str = "handler-waits-scheduled-until-scheduled";

/// A completed handler waiting for its own command buffer to complete.
pub const COMPLETED_UNTIL_COMPLETED: &str = "handler-waits-completed-until-completed";

impl OwnHandlerWait {
    /// The transcript the probe that died states: the one site, holding the negative fact.
    ///
    /// A site that is **present and negative** is the whole point. Leaving it out would let the row
    /// read as a pass, which is exactly the hazard the record kind exists for: "Metal never
    /// returned" and "nobody looked" must not print the same.
    pub fn stated_transcript(&self) -> Result<Transcript, String> {
        let mut sites = Sites::new();
        sites.push(crate::case::inputs_node(&self.inputs()?));
        sites.push(crate::handler_waits::never_returned(self.wait.site()));
        sites.notices(NOTICES_SITE, &[] as &[&str]);
        Ok(sites.finish())
    }

    /// The whole answer the case's recording holds, built the only way such an answer may be built
    /// ([`crate::record::never_returned`]).
    ///
    /// The case states its own recording, rather than a file stating it: that is what makes a
    /// hand-written recording checkable. A test holds the committed file to this, and the file was
    /// first written from it.
    pub fn stated_answer(&self) -> Result<crate::recording::Answer, String> {
        let MetalSource::DidNotReturn { from, call, .. } = self.metal_source() else {
            return Err(format!(
                "{}: its Metal source is not a call Metal left",
                self.name
            ));
        };
        Ok(crate::record::never_returned(
            metal2vulkan_runtime::MetalLevel::NEWEST,
            from,
            call,
            PROBED_ON,
            PROBE_RUNS,
            self.stated_transcript()?,
        ))
    }
}

impl Case for OwnHandlerWait {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::did_not_return(PROBED_IN, self.wait.site(), self.says)
    }

    fn allowances(&self) -> Vec<crate::allowance::Allowance> {
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the notices
        // site alone: every value of this family still compares against Metal's exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09), where it fired for all 3 rows of the family
        // and each row was divergent on that field and nothing else (replay-nvidia.log).
        vec![crate::replay::shared_memory_notice()]
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        Ok(vec![
            Input::bytes("wait", &self.wait.words()),
            Input::bytes("command-buffer", COMMAND_BUFFER),
        ])
    }

    fn carried_transcript(&self) -> Result<Option<(u32, Transcript)>, String> {
        Ok(Some((PROBE_RUNS, self.stated_transcript()?)))
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let at = self.wait.site();
        // Both arms emit the same site, because the site is what the comparison settles on: a
        // typed error there is the row's pass, and any answer at all is a divergence.
        match side.handler_wait(self.wait) {
            Ok(value) => sites.push(Node::leaf(at, Observed::value(value))),
            Err(failure) => sites.push(Node::leaf(at, Observed::value(failure.recorded()))),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recording::{self, Provenance, Recording, Validated};

    /// Each committed recording is exactly what [`crate::record::never_returned`] builds from the
    /// case's own stated answer.
    ///
    /// The files are written by hand because the shape cannot be run, which is precisely why they
    /// need tying down: without this, a hand-written file could say anything, and a change to the
    /// case's sites or inputs would leave it stale in silence.
    #[test]
    fn every_recording_is_the_answer_its_case_states() {
        for case in all() {
            let stated =
                Recording::new(case.name(), Vec::new(), vec![case.stated_answer().unwrap()]);
            let committed =
                recording::read(case.name()).unwrap_or_else(|why| panic!("{}: {why}", case.name()));
            // Metal's answer alone: the cost and times halves of the file are our own
            // measurements on a venue, taken by other legs, and a case states nothing about
            // either.
            assert_eq!(
                committed.without_costs().without_times(),
                stated,
                "{}: its committed recording is not the answer the case states",
                case.name()
            );
        }
    }

    /// Every one of them says Metal never returned, and says that Metal's validation layer was
    /// never asked either.
    #[test]
    fn every_answer_names_the_call_and_claims_no_validated_fact() {
        for case in all() {
            let answer = case.stated_answer().unwrap();
            assert_eq!(
                answer.provenance,
                Provenance::NeverReturned {
                    from: PROBED_IN.to_owned(),
                    call: case.wait.site().to_owned(),
                },
                "{}",
                case.name()
            );
            assert_eq!(answer.validated, Validated::NotRun, "{}", case.name());
        }
    }

    /// Each answer is counted among those that did not come from this round's oracle.
    ///
    /// `Provenance::measured_elsewhere` reads the provenance's own chip, which a never-returned
    /// row has none of, so reading it alone left all three out of the report's per-chip count --
    /// the one thing that count exists to prevent. [`crate::recording::Answer::measured_elsewhere`]
    /// reads the field these rows do carry.
    #[test]
    fn every_answer_is_counted_as_coming_from_another_chip() {
        for case in all() {
            let answer = case.stated_answer().unwrap();
            assert_eq!(
                answer.measured_elsewhere(),
                Some(recording::APPLE_M4),
                "{}: the report would not count it",
                case.name()
            );
        }
    }

    /// Record mode refuses all three, in the case's own words.
    #[test]
    fn record_mode_refuses_every_one_of_them() {
        for case in all() {
            assert_eq!(
                case.metal_source().forbidden(),
                Some(case.says),
                "{}: record mode would run it",
                case.name()
            );
        }
    }
}
