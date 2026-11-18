//! What one command buffer answers about itself: its two owners, its four timing getters, and the
//! encoder kind nothing else in the suite opens.
//!
//! `MTLCommandBuffer` has twenty-one implemented items and the suite reached thirteen of them.
//! Seven of the eight it did not are here: `device`, `commandQueue`, `kernelStartTime`,
//! `kernelEndTime`, `GPUStartTime`, `GPUEndTime` and `resourceStateCommandEncoder`. The eighth is
//! the protocol declaration itself, which no call site can reach.
//!
//! # Why a time is never recorded, and what is recorded instead
//!
//! The four timing getters answer a `CFTimeInterval` — this machine's clock, in seconds. A
//! recording that held one would be a recording of when somebody ran the suite, and it would
//! differ on its own next run. So nothing here records a time.
//!
//! What is durable is the **shape** of the four answers, and it is a contract a consumer leans on:
//!
//! - each getter answers **zero** on a command buffer that has not completed, so a consumer that
//!   reads one early reads a zero rather than a stale number;
//! - each answers a **positive** number once the command buffer completed;
//! - each interval's end is at or after its start, and the GPU's interval does not begin or end
//!   before the kernel's did.
//!
//! Those are booleans, and a boolean compares exactly on every venue and in every year. A venue
//! whose `GPUEndTime` precedes its `GPUStartTime` records a different boolean and the case moves,
//! which is the point: nothing in this tree watched for that before.
//!
//! # The GPU pair is a `Result`, because one side may have no answer
//!
//! `GPUStartTime` and `GPUEndTime` need the device to write timestamps. The runtime refuses them
//! typed where its queue family writes none, and answers them from the host — announcing
//! `Degradation::GpuTimesFromHost` — where the driver's own timestamps wait for every queue's.
//! So [`CommandBufferFacts::gpu`] holds either the flags or the refusal, and a venue with no
//! timestamps records the refusal rather than four booleans derived from zeros. The kernel pair
//! needs no device at all and is always flags.
//!
//! # Why the two owners are booleans too
//!
//! `device` and `commandQueue` answer objects, and an object's address is not a fact a recording
//! may hold either. What a case can state is the **identity** a consumer depends on: the command
//! buffer's `device` is the device its queue was made on — compared by `registryID`, the number
//! Metal gives a device for exactly this — and its `commandQueue` is the queue object it was made
//! by. Each is one boolean, and an implementation that handed back another device or another queue
//! fails it.
//!
//! # Why it is safe
//!
//! One command buffer of one small blit — a fill of a few bytes of the side's own buffer — is
//! committed and waited for. No shader runs, no draw, no dispatch, no loop, and the wait is
//! bounded. The resource state encoder is opened on a command buffer of its own and ended with no
//! update encoded at all, which maps nothing and takes no heap memory.

use crate::case::Failure;
use crate::transcript::{Node, Observed, Value};

/// The flags the GPU pair comes to, where the side answered both getters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuFlags {
    /// Whether `GPUStartTime` and `GPUEndTime` answered zero before the commit.
    pub zero_before: [bool; 2],
    /// Whether each answered a number greater than zero once the command buffer completed.
    pub positive_after: [bool; 2],
    /// Whether `GPUEndTime` is at or after `GPUStartTime`.
    pub interval_is_ordered: bool,
    /// Whether `GPUStartTime` is at or after `kernelStartTime`.
    pub starts_no_earlier_than_the_kernel: bool,
    /// Whether `GPUEndTime` is at or after `kernelEndTime`.
    pub ends_no_earlier_than_the_kernel: bool,
}

/// What one command buffer answered about itself.
#[derive(Clone, Debug, PartialEq)]
pub struct CommandBufferFacts {
    /// `device`: whether its `registryID` is that of the device the queue was made on, or the
    /// side's answer that the getter is the consumer's own bookkeeping
    /// ([`CommandBufferFacts::the_consumers`]).
    pub device_is_the_queues: Result<bool, Failure>,
    /// `commandQueue`: for each of two distinctly labelled queues, whether the command buffer
    /// made on it answers **its own** queue's label, or the side's answer that the label and the
    /// getter are the consumer's own bookkeeping.
    pub queue_label_is_its_own: Result<[bool; 2], Failure>,
    /// Whether `kernelStartTime` and `kernelEndTime` answered zero before the commit.
    pub kernel_zero_before: [bool; 2],
    /// Whether each answered a number greater than zero once the command buffer completed.
    pub kernel_positive_after: [bool; 2],
    /// Whether `kernelEndTime` is at or after `kernelStartTime`.
    pub kernel_interval_is_ordered: bool,
    /// The GPU pair's flags, or the side's typed refusal of the two getters.
    pub gpu: Result<GpuFlags, Failure>,
    /// `resourceStateCommandEncoder`: whether one was opened on a fresh command buffer, or the
    /// side's typed refusal of the encoder. The encoder is ended at once with nothing encoded
    /// into it.
    pub resource_state_encoder_opened: Result<bool, Failure>,
    /// The status of the command buffer the timings were read from, once it completed.
    pub status: Value,
    /// The status of the command buffer the resource state encoder was opened on.
    pub state_encoder_status: Value,
}

/// The four timing getters, in the order both sides read them and the order a site names them.
pub const TIMERS: [&str; 4] = [
    "kernelStartTime",
    "kernelEndTime",
    "GPUStartTime",
    "GPUEndTime",
];

impl CommandBufferFacts {
    /// The facts a command buffer's answers come to.
    ///
    /// `kernel_before` and `kernel_after` are `kernelStartTime` and `kernelEndTime` before the
    /// commit and after the completion; `gpu_before` and `gpu_after` the same pair of
    /// `GPUStartTime` and `GPUEndTime`, or the side's refusal of them.
    ///
    /// Both sides build their answer through this one function, so neither can read the order of
    /// the four differently from the other.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device_is_the_queues: bool,
        queue_label_is_its_own: [bool; 2],
        kernel_before: [f64; 2],
        kernel_after: [f64; 2],
        gpu: Result<([f64; 2], [f64; 2]), Failure>,
        resource_state_encoder_opened: bool,
        status: Value,
        state_encoder_status: Value,
    ) -> Self {
        Self {
            device_is_the_queues: Ok(device_is_the_queues),
            queue_label_is_its_own: Ok(queue_label_is_its_own),
            kernel_zero_before: [kernel_before[0] == 0.0, kernel_before[1] == 0.0],
            kernel_positive_after: [kernel_after[0] > 0.0, kernel_after[1] > 0.0],
            kernel_interval_is_ordered: kernel_after[1] >= kernel_after[0],
            gpu: gpu.map(|(before, after)| GpuFlags {
                zero_before: [before[0] == 0.0, before[1] == 0.0],
                positive_after: [after[0] > 0.0, after[1] > 0.0],
                interval_is_ordered: after[1] >= after[0],
                starts_no_earlier_than_the_kernel: after[0] >= kernel_after[0],
                ends_no_earlier_than_the_kernel: after[1] >= kernel_after[1],
            }),
            resource_state_encoder_opened: Ok(resource_state_encoder_opened),
            status,
            state_encoder_status,
        }
    }

    /// The facts of a side that keeps none of the consumer's own bookkeeping, `why` saying so:
    /// a queue's label and a command buffer's `device` and `commandQueue` never reach the GPU,
    /// so a runtime that leaves them to its consumer answers `TheConsumers` at their sites.
    pub fn the_consumers(mut self, why: &str) -> Self {
        let failure = Failure::named("TheConsumers", why);
        self.device_is_the_queues = Err(failure.clone());
        self.queue_label_is_its_own = Err(failure);
        self
    }

    /// The facts of a side that refuses the resource state encoder, with its refusal.
    pub fn without_resource_state_encoder(mut self, refusal: Failure) -> Self {
        self.resource_state_encoder_opened = Err(refusal);
        self
    }

    /// The facts as the sites a case emits under one node, so both sides emit one shape.
    pub fn node(&self, name: impl Into<crate::transcript::NodeName>) -> Node {
        let flag = |at: String, held: bool| Node::leaf(at, Observed::value(Value::Bool(held)));
        let answer = |at: String, held: Result<bool, &Failure>| {
            Node::leaf(
                at,
                Observed::value(held.map_or_else(Failure::recorded, Value::Bool)),
            )
        };
        let labels = self.queue_label_is_its_own.as_ref();
        let mut node = Node::branch(name, Vec::new())
            .with(answer(
                "device-is-the-queues".to_owned(),
                self.device_is_the_queues.as_ref().copied(),
            ))
            .with(answer(
                "queue-label-is-its-own/0".to_owned(),
                labels.map(|labels| labels[0]),
            ))
            .with(answer(
                "queue-label-is-its-own/1".to_owned(),
                labels.map(|labels| labels[1]),
            ));
        for (at, getter) in TIMERS[..2].iter().enumerate() {
            node = node.with(flag(
                format!("zero-before-commit/{getter}"),
                self.kernel_zero_before[at],
            ));
            node = node.with(flag(
                format!("positive-after-completion/{getter}"),
                self.kernel_positive_after[at],
            ));
        }
        node = node.with(flag(
            "kernel-interval-is-ordered".to_owned(),
            self.kernel_interval_is_ordered,
        ));
        match &self.gpu {
            Err(failure) => {
                node = node.with(Node::leaf("gpu-times", Observed::value(failure.recorded())));
            }
            Ok(flags) => {
                node = node.with(Node::leaf(
                    "gpu-times",
                    Observed::value(Value::ok(Value::Absent)),
                ));
                for (at, getter) in TIMERS[2..].iter().enumerate() {
                    node = node.with(flag(
                        format!("zero-before-commit/{getter}"),
                        flags.zero_before[at],
                    ));
                    node = node.with(flag(
                        format!("positive-after-completion/{getter}"),
                        flags.positive_after[at],
                    ));
                }
                node = node
                    .with(flag(
                        "gpu-interval-is-ordered".to_owned(),
                        flags.interval_is_ordered,
                    ))
                    .with(flag(
                        "gpu-starts-no-earlier-than-the-kernel".to_owned(),
                        flags.starts_no_earlier_than_the_kernel,
                    ))
                    .with(flag(
                        "gpu-ends-no-earlier-than-the-kernel".to_owned(),
                        flags.ends_no_earlier_than_the_kernel,
                    ));
            }
        }
        node.with(answer(
            "resource-state-encoder-opened".to_owned(),
            self.resource_state_encoder_opened.as_ref().copied(),
        ))
        .with(Node::leaf("status", Observed::value(self.status.clone())))
        .with(Node::leaf(
            "state-encoder-status",
            Observed::value(self.state_encoder_status.clone()),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// A set of answers whose order is the one Metal is expected to give, for the guards below.
    fn ordered() -> CommandBufferFacts {
        facts([100.0, 100.5], Ok(([0.0, 0.0], [100.25, 101.0])))
    }

    /// The facts of a kernel pair and a GPU answer, with everything else held still.
    fn facts(
        kernel_after: [f64; 2],
        gpu: Result<([f64; 2], [f64; 2]), Failure>,
    ) -> CommandBufferFacts {
        CommandBufferFacts::new(
            true,
            [true, true],
            [0.0, 0.0],
            kernel_after,
            gpu,
            true,
            Value::unit_variant("Completed"),
            Value::unit_variant("Completed"),
        )
    }

    /// Each ordering reads the times, and **fails on the times that break it**.
    ///
    /// Each half matters. Without the first the orderings would be constants; without the second a
    /// getter whose answers came out backwards would record the same booleans as one that did not,
    /// and the case would compare nothing four times over.
    #[test]
    fn each_ordering_fails_on_the_times_that_break_it() {
        let held = ordered();
        assert!(held.kernel_interval_is_ordered);
        let flags = held.gpu.clone().expect("the GPU pair answered");
        assert!(flags.interval_is_ordered);
        assert!(flags.starts_no_earlier_than_the_kernel);
        assert!(flags.ends_no_earlier_than_the_kernel);

        // A kernel interval that ends before it starts.
        assert!(
            !facts([100.0, 99.0], Ok(([0.0, 0.0], [100.25, 101.0]))).kernel_interval_is_ordered
        );
        // A GPU interval that ends before it starts.
        let backwards = facts([100.0, 100.5], Ok(([0.0, 0.0], [101.0, 100.25])))
            .gpu
            .expect("answered");
        assert!(!backwards.interval_is_ordered);
        // A GPU that starts before the kernel did.
        let early = facts([100.0, 100.5], Ok(([0.0, 0.0], [99.0, 101.0])))
            .gpu
            .expect("answered");
        assert!(!early.starts_no_earlier_than_the_kernel);
        // A GPU that ends before the kernel's submission did.
        let short = facts([100.0, 100.5], Ok(([0.0, 0.0], [100.25, 100.4])))
            .gpu
            .expect("answered");
        assert!(!short.ends_no_earlier_than_the_kernel);
    }

    /// A zero before the commit and a positive number after read as such, and each flag fails on
    /// the answer that breaks it.
    #[test]
    fn the_zero_and_positive_flags_fail_on_the_answers_that_break_them() {
        let held = ordered();
        assert_eq!(held.kernel_zero_before, [true; 2]);
        assert_eq!(held.kernel_positive_after, [true; 2]);
        let early = CommandBufferFacts::new(
            true,
            [true, true],
            [0.0, 7.0],
            [100.0, 100.5],
            Ok(([3.0, 0.0], [0.0, 101.0])),
            true,
            Value::unit_variant("Completed"),
            Value::unit_variant("Completed"),
        );
        assert_eq!(early.kernel_zero_before, [true, false]);
        let flags = early.gpu.expect("answered");
        assert_eq!(flags.zero_before, [false, true]);
        assert_eq!(flags.positive_after, [false, true]);
    }

    /// A refused GPU pair records the refusal and **no** GPU flags: a venue whose device writes no
    /// timestamps compares a refusal against Metal's answer rather than four booleans derived from
    /// zeros, which would have read as agreement.
    #[test]
    fn a_refused_gpu_pair_records_the_refusal_and_no_flags() {
        let refused = facts(
            [100.0, 100.5],
            Err(Failure::named("Unsupported", "GpuTimes".to_owned())),
        );
        assert!(refused.gpu.is_err());
        let names: BTreeSet<String> = refused
            .node("facts")
            .children
            .iter()
            .map(|child| child.name.to_string())
            .collect();
        assert!(names.contains("gpu-times"));
        assert!(!names.contains("gpu-interval-is-ordered"));
        assert!(!names.contains("zero-before-commit/GPUStartTime"));
        // And the kernel half is still there, so a refusal of one pair does not hide the other.
        assert!(names.contains("kernel-interval-is-ordered"));
        assert!(names.contains("zero-before-commit/kernelStartTime"));
    }

    /// Every flag the facts hold reaches a site of its own, so none of them can be set and then
    /// never compared.
    ///
    /// The count is written out: a flag added to the struct and not to the node would otherwise be
    /// a fact nobody compares, which is exactly the shape this round found six times.
    #[test]
    fn every_flag_reaches_a_site_of_its_own() {
        let node = ordered().node("facts");
        let names: BTreeSet<String> = node
            .children
            .iter()
            .map(|child| child.name.to_string())
            .collect();
        assert_eq!(names.len(), node.children.len(), "{names:?}");
        // The device, a flag a labelled queue, two flags a timer, one kernel ordering, the GPU
        // answer, three GPU orderings, the encoder, two statuses.
        assert_eq!(
            node.children.len(),
            1 + 2 + 2 * TIMERS.len() + 1 + 1 + 3 + 1 + 2
        );
        for getter in TIMERS {
            assert!(names.contains(&format!("zero-before-commit/{getter}")));
            assert!(names.contains(&format!("positive-after-completion/{getter}")));
        }
    }
}
