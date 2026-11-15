//! A lane device the driver refused for want of what other processes' devices hold, asked for
//! again once they may have let it go (lane tooling).
//!
//! The lane runs one test process per core, and the GPU's driver shares what a device needs among
//! every process's devices. NVIDIA's gives each `VkQueue` a channel, each with an instance block
//! of its own, and refuses a device where it cannot allocate one (`vkCreateDevice` answering
//! `VK_ERROR_INITIALIZATION_FAILED` or `VK_ERROR_OUT_OF_HOST_MEMORY`, the kernel logging that it
//! could not allocate a channel's instance block). An RTX 2080 (NVIDIA 620.70.06) holds about 210
//! channels across processes with nothing else running: 11 runtime devices of the queue plan's 18
//! `VkQueue`s and 3 of the least plan's 3, the 15th refused (`examples/queue_plan_cost.rs`,
//! `--runtime-fill`). A 16-thread host's 16 lane processes ask it for up to 288 at one device of
//! the plan each, and 188 of the lane's modules make devices of the plan, whose cases run side by
//! side: there the driver refused devices of the plan, which the runtime then made with its least
//! plan, announcing `least-queue-plan`, and lane devices of 3 `VkQueue`s outright
//! (`Error::DeviceRefused`), in 33 of 2,054 tests of the two lane legs of one round.
//!
//! Neither answer is the test's: its device is the one it asked for, and the GPU lacked room for
//! it only while the lane's other processes held theirs. So the lane makes such a device again,
//! the one it made dropped first, after a pause that doubles from [`FIRST_PAUSE`] up to
//! [`LONGEST_PAUSE`], until [`ROOM_WAIT`] in all has passed, and then hands back the answer it
//! has. Any other answer is handed back at once. Each pause is written to standard error as a
//! `LANE-DEVICE-ROOM` line naming the answer, the pause and the time waited so far: the evidence
//! that the wait ran. The runtime's own refusal and its least plan are its own tests' to check
//! (`tests::least_queue_plan`, the driver's refusal stood in for).

use crate::candidate::runtime_case::CaseError;
use metal2vulkan_runtime::{DebugMessage, Degradation, Error};
use std::time::Duration;

/// The first pause before a device is asked for again.
const FIRST_PAUSE: Duration = Duration::from_millis(250);

/// The longest pause between two asks.
const LONGEST_PAUSE: Duration = Duration::from_secs(2);

/// The most a lane device waits for room, in all: a lane test takes 2 to 20 s, so the other
/// processes' devices of a refused one's time have gone well within it.
const ROOM_WAIT: Duration = Duration::from_secs(120);

/// What `attempt` makes, asked for again while it answers that the GPU lacked room (`Some`,
/// saying how), with `pause` between: the made value dropped first, then a pause of
/// [`FIRST_PAUSE`] doubling up to [`LONGEST_PAUSE`], until the pauses reach [`ROOM_WAIT`]. The
/// last answer is handed back, whether it had room or not.
pub(super) fn with_room<T>(
    mut pause: impl FnMut(Duration),
    mut attempt: impl FnMut() -> (T, Option<String>),
) -> T {
    let mut waited = Duration::ZERO;
    let mut next = FIRST_PAUSE;
    loop {
        let (made, short) = attempt();
        let Some(answer) = short else {
            return made;
        };
        if waited >= ROOM_WAIT {
            eprintln!("LANE-DEVICE-ROOM: {answer}; no room after {waited:?}, handed back");
            return made;
        }
        drop(made);
        eprintln!("LANE-DEVICE-ROOM: {answer}; asked again in {next:?} ({waited:?} waited)");
        pause(next);
        waited += next;
        next = (next * 2).min(LONGEST_PAUSE);
    }
}

/// The driver's refusal of a device for want of what other devices hold, where `error` is that
/// (`Error::DeviceRefused`, naming each attempt); `None` for any other error.
pub(super) fn refused_for_room(error: &CaseError) -> Option<String> {
    match error {
        CaseError::Runtime { error, .. } => match &**error {
            Error::DeviceRefused(refused) => Some(refused.to_string()),
            _ => None,
        },
        _ => None,
    }
}

/// Why the device whose making delivered `made` took the runtime's least queue plan, the driver
/// having refused its plan for want of what other devices hold, where it announced
/// `least-queue-plan`.
pub(super) fn least_plan_taken(made: &[DebugMessage]) -> Option<String> {
    (made.iter())
        .filter_map(|message| message.degradation.as_ref())
        .find(|notice| notice.degradation == Degradation::LeastQueuePlan)
        .map(|notice| format!("least-queue-plan: {}", notice.reason))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// An answer that lacked room is asked for again after a pause that doubles, and the first
    /// that had room is handed back.
    #[test]
    fn a_short_answer_is_asked_for_again_until_one_has_room() {
        let pauses = RefCell::new(Vec::new());
        let mut asked = 0;
        let made = with_room(
            |pause| pauses.borrow_mut().push(pause),
            || {
                asked += 1;
                let short = (asked < 4).then(|| format!("refused {asked}"));
                (asked, short)
            },
        );
        assert_eq!(made, 4);
        let pauses = pauses.into_inner();
        let expected = [FIRST_PAUSE, FIRST_PAUSE * 2, FIRST_PAUSE * 4];
        assert_eq!(pauses, expected);
    }

    /// An answer with room is handed back with no pause.
    #[test]
    fn an_answer_with_room_is_handed_back_at_once() {
        let mut pauses = 0;
        assert_eq!(with_room(|_| pauses += 1, || ("made", None)), "made");
        assert_eq!(pauses, 0);
    }

    /// An answer that never has room is asked for until the pauses reach the bound, none past
    /// the longest, and the last answer is handed back.
    #[test]
    fn an_answer_that_never_has_room_is_handed_back_at_the_bound() {
        let pauses = RefCell::new(Vec::new());
        let mut asked = 0;
        let made = with_room(
            |pause| pauses.borrow_mut().push(pause),
            || {
                asked += 1;
                (asked, Some("refused".to_owned()))
            },
        );
        let pauses = pauses.into_inner();
        assert_eq!(made, pauses.len() + 1);
        assert!(pauses.iter().all(|&pause| pause <= LONGEST_PAUSE));
        let waited: Duration = pauses.iter().sum();
        assert!(waited >= ROOM_WAIT, "{waited:?}");
        assert!(waited < ROOM_WAIT + LONGEST_PAUSE, "{waited:?}");
    }
}
