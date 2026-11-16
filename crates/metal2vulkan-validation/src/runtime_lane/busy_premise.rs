//! A step that needs the GPU busy past its calls, run again where the busy work ended first (lane
//! tooling).
//!
//! A step of `device_copies`' `did_modify_range` oracle, and each storage mode of `first_use`'s
//! host copies of a new texture, commits busy work (`first_use`'s `Busy::commit`, about 2 s of GPU
//! time at its measured rate) and then makes its calls. Those return in milliseconds: 5 to 47 ms
//! from the busy work's first commit to the last call's return on an RX 7900 XTX (RADV), alone and
//! in a lane leg; the texture copies 20 to 670 us each on a Radeon 780M. The step's premise is that the GPU has not
//! run its command buffer when its calls return: the end mark the busy work writes last still
//! reads 0. A host that keeps the test off the CPU past the busy window breaks the premise with
//! nothing wrong in the runtime. One step on the XTX missed it so, in a lane leg whose node's 16
//! CPUs also ran an RTX 2080 suite, lavapipe and a replay; and one texture copy on the Radeon 780M,
//! in a leg at its memory limit beside a virtual machine's boot, whose calls took 140 and 667 us
//! and whose end mark turned within a millisecond of reading 0. The window cannot be stretched to fit
//! any host: its length rests on a rate measured once, and Vulkan does not promise that a running
//! shader sees the host's release.
//!
//! So such a try gives no answer about the premise. The step still checks the bytes it read, which
//! must be one of the answers allowed whatever the timing. A `LANE-BUSY-MISS` line goes to
//! standard error, saying what each call took and how long each of the busy work's command
//! buffers ran on the GPU. Then the step runs again, at most [`TRIES`] times in all. A call that
//! waits for the GPU's earlier work misses on every try, and the step then fails with each try's
//! times.

/// The most tries of a step: a host that starves the test past the busy window three times in a
/// row is past what the lane's calls can be judged on.
pub(super) const TRIES: u32 = 3;

/// Run `attempt` (given its try's number, from 1) until a try answers that its premise held
/// (`None`), at most [`TRIES`] times. A try that answers how its premise missed (`Some`) is
/// written to standard error as a `LANE-BUSY-MISS` line for `step` and run again. Panics once
/// every try missed, with each try's answer.
pub(super) fn until_premise_held(step: &str, mut attempt: impl FnMut(u32) -> Option<String>) {
    let mut missed = Vec::new();
    for count in 1..=TRIES {
        let Some(miss) = attempt(count) else {
            return;
        };
        eprintln!("LANE-BUSY-MISS {step}, try {count} of {TRIES}: {miss}");
        missed.push(miss);
    }
    panic!(
        "{step}: the busy work ended first in each of {TRIES} tries: {}",
        missed.join("; ")
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A try whose premise held is the step's only one.
    #[test]
    fn a_held_premise_is_tried_once() {
        let mut tries = Vec::new();
        until_premise_held("step", |count| {
            tries.push(count);
            None
        });
        assert_eq!(tries, [1]);
    }

    /// A try whose premise missed is run again, and the first that held ends the step.
    #[test]
    fn a_missed_premise_is_tried_again() {
        let mut tries = Vec::new();
        until_premise_held("step", |count| {
            tries.push(count);
            (count < TRIES).then(|| format!("missed {count}"))
        });
        assert_eq!(tries, [1, 2, 3]);
    }

    /// A step whose premise missed on every try fails, naming each try's miss.
    #[test]
    #[should_panic(
        expected = "step: the busy work ended first in each of 3 tries: missed 1; \
                               missed 2; missed 3"
    )]
    fn a_premise_missed_on_every_try_fails() {
        until_premise_held("step", |count| Some(format!("missed {count}")));
    }
}
