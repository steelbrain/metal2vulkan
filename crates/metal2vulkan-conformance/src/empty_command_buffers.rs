//! What a command buffer with nothing encoded answers: on an idle queue, and committed behind a
//! long fill of its queue.
//!
//! A command buffer that encodes nothing needs no GPU work, and an implementation may complete it
//! without its GPU at all: the runtime takes such a command buffer with no `vkQueueSubmit`
//! (`WorkCounts::command_buffers_unsubmitted`). What a program can still observe is Metal's
//! contract for it, and that is what is recorded here:
//!
//! - its status, from inside its scheduled handler, its completed handler and after its wait;
//! - the shape of its four timing getters: zero before the commit, positive after, the kernel's
//!   interval ordered, and its GPU interval **one point** (shorter than [`ONE_POINT`]) at or after
//!   its kernel end;
//! - behind a fill of its queue: that it completes after the fill (the completed handlers' order,
//!   and the fill's status as the empty one's completed handler reads it), that its kernel interval
//!   begins at or after the fill's ends, and that its GPU point is at or after the fill's GPU end.
//!
//! As [`crate::command_buffer_facts`] says, a time is never recorded, only a sign or an order of
//! two, so each is a boolean that compares exactly. The GPU pair is a `Result`, because a side may
//! refuse it typed where its device writes no timestamps.
//!
//! Measured on the M3 Max (macOS 27.0.1), Metal's empty command buffer on an idle queue read a GPU
//! interval of 0 to 0.3 us, 1 to 9 us after its `kernelEndTime`; behind a fill, its GPU point came
//! 65 to 135 us after the fill's `GPUEndTime`.

use crate::case::Failure;
use crate::event_orders::Status;
use crate::transcript::{Node, NodeName, Observed, Value};
use std::time::Duration;

/// The bytes of the fill the second empty command buffer is committed behind: enough that it is
/// still running as the empty one is committed, on every venue measured.
pub const FILL_BYTES: u64 = 64 << 20;

/// The longest a GPU interval may be and still read as one point: an empty command buffer's on
/// Metal is 0 to 0.3 us.
pub const ONE_POINT: f64 = 1e-6;

/// The longest the probe waits for a handler it reads a status from.
pub const ARRIVAL: Duration = Duration::from_secs(30);

/// The name the fill's completed handler records itself by.
pub const FILL: &str = "fill";

/// The name the empty command buffer's completed handler records itself by.
pub const EMPTY: &str = "empty";

/// The GPU pair's flags of the empty command buffer on an idle queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AloneGpu {
    /// Whether `GPUStartTime` and `GPUEndTime` answered zero before the commit.
    pub zero_before: [bool; 2],
    /// Whether each answered a number greater than zero once it completed.
    pub positive_after: [bool; 2],
    /// Whether `GPUEndTime` is at or after `GPUStartTime` and less than [`ONE_POINT`] after it.
    pub one_point: bool,
    /// Whether `GPUStartTime` is at or after `kernelEndTime`.
    pub starts_no_earlier_than_the_kernel_end: bool,
}

/// What the empty command buffer on an idle queue answered.
#[derive(Clone, Debug, PartialEq)]
pub struct Alone {
    /// Its status, from inside its scheduled handler.
    pub in_the_scheduled_handler: Status,
    /// Its status, from inside its completed handler.
    pub in_the_completed_handler: Status,
    /// Its status once its wait returned.
    pub after_the_wait: Status,
    /// Whether `kernelStartTime` and `kernelEndTime` answered zero before the commit.
    pub kernel_zero_before: [bool; 2],
    /// Whether each answered a number greater than zero once it completed.
    pub kernel_positive_after: [bool; 2],
    /// Whether `kernelEndTime` is at or after `kernelStartTime`.
    pub kernel_interval_is_ordered: bool,
    /// The GPU pair's flags, or the side's typed refusal of the two getters.
    pub gpu: Result<AloneGpu, Failure>,
}

impl Alone {
    /// The facts the answers come to: the kernel pair and the GPU pair before the commit and
    /// after the completion, and the three statuses. Both sides build their answer through this
    /// one function.
    pub fn new(
        statuses: [Status; 3],
        kernel: ([f64; 2], [f64; 2]),
        gpu: Result<([f64; 2], [f64; 2]), Failure>,
    ) -> Self {
        let [in_the_scheduled_handler, in_the_completed_handler, after_the_wait] = statuses;
        let (kernel_before, kernel_after) = kernel;
        Self {
            in_the_scheduled_handler,
            in_the_completed_handler,
            after_the_wait,
            kernel_zero_before: kernel_before.map(|time| time == 0.0),
            kernel_positive_after: kernel_after.map(|time| time > 0.0),
            kernel_interval_is_ordered: kernel_after[1] >= kernel_after[0],
            gpu: gpu.map(|(before, after)| AloneGpu {
                zero_before: before.map(|time| time == 0.0),
                positive_after: after.map(|time| time > 0.0),
                one_point: one_point(after),
                starts_no_earlier_than_the_kernel_end: after[0] >= kernel_after[1],
            }),
        }
    }
}

/// The GPU pair's flags of the empty command buffer behind the fill.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BehindGpu {
    /// Whether its GPU interval is one point ([`ONE_POINT`]).
    pub one_point: bool,
    /// Whether its `GPUStartTime` is at or after the fill's `GPUEndTime`.
    pub starts_no_earlier_than_the_fills_end: bool,
}

/// What the empty command buffer committed behind the fill answered.
#[derive(Clone, Debug, PartialEq)]
pub struct Behind {
    /// The completed handlers of the two, in the order they were called.
    pub handler_order: Vec<String>,
    /// The fill's status, from inside the empty one's completed handler.
    pub fills_status_in_the_empty_ones_handler: Status,
    /// The empty one's status once its wait returned.
    pub after_the_wait: Status,
    /// Whether its `kernelStartTime` is at or after the fill's `kernelEndTime`.
    pub kernel_starts_no_earlier_than_the_fills_end: bool,
    /// The GPU pair's flags, or the side's typed refusal of the getters (the fill's or its own).
    pub gpu: Result<BehindGpu, Failure>,
}

impl Behind {
    /// The facts the answers come to: the handler order, the two statuses, the kernel start of the
    /// empty one and the kernel end of the fill, and the fill's and the empty one's GPU pairs.
    pub fn new(
        handler_order: Vec<String>,
        statuses: [Status; 2],
        (kernel_start, fills_kernel_end): (f64, f64),
        gpu: Result<([f64; 2], [f64; 2]), Failure>,
    ) -> Self {
        let [fills_status_in_the_empty_ones_handler, after_the_wait] = statuses;
        Self {
            handler_order,
            fills_status_in_the_empty_ones_handler,
            after_the_wait,
            kernel_starts_no_earlier_than_the_fills_end: kernel_start >= fills_kernel_end,
            gpu: gpu.map(|(fill, empty)| BehindGpu {
                one_point: one_point(empty),
                starts_no_earlier_than_the_fills_end: empty[0] >= fill[1],
            }),
        }
    }
}

/// Whether a GPU interval `[start, end]` is one point: ordered, and shorter than [`ONE_POINT`].
fn one_point([start, end]: [f64; 2]) -> bool {
    end >= start && end - start < ONE_POINT
}

/// What the two empty command buffers answered.
#[derive(Clone, Debug, PartialEq)]
pub struct EmptyCommandBuffers {
    /// The one on an idle queue.
    pub alone: Alone,
    /// The one committed behind a fill of its queue.
    pub behind: Behind,
}

impl EmptyCommandBuffers {
    /// The facts as the sites a case emits under one node, so both sides emit one shape.
    pub fn node(&self, name: impl Into<NodeName>) -> Node {
        let flag = |at: &str, held: bool| Node::leaf(at, Observed::value(Value::Bool(held)));
        let status = |at: &str, status: &Status| Node::leaf(at, Observed::value(status.value()));
        let alone = &self.alone;
        let mut node = Node::branch(name, Vec::new())
            .with(status(
                "alone/status/in-the-scheduled-handler",
                &alone.in_the_scheduled_handler,
            ))
            .with(status(
                "alone/status/in-the-completed-handler",
                &alone.in_the_completed_handler,
            ))
            .with(status("alone/status/after-the-wait", &alone.after_the_wait))
            .with(flag(
                "alone/zero-before-commit/kernelStartTime",
                alone.kernel_zero_before[0],
            ))
            .with(flag(
                "alone/zero-before-commit/kernelEndTime",
                alone.kernel_zero_before[1],
            ))
            .with(flag(
                "alone/positive-after-completion/kernelStartTime",
                alone.kernel_positive_after[0],
            ))
            .with(flag(
                "alone/positive-after-completion/kernelEndTime",
                alone.kernel_positive_after[1],
            ))
            .with(flag(
                "alone/kernel-interval-is-ordered",
                alone.kernel_interval_is_ordered,
            ));
        node = match &alone.gpu {
            Err(failure) => node.with(Node::leaf(
                "alone/gpu-times",
                Observed::value(failure.recorded()),
            )),
            Ok(gpu) => node
                .with(Node::leaf(
                    "alone/gpu-times",
                    Observed::value(Value::ok(Value::Absent)),
                ))
                .with(flag(
                    "alone/zero-before-commit/GPUStartTime",
                    gpu.zero_before[0],
                ))
                .with(flag(
                    "alone/zero-before-commit/GPUEndTime",
                    gpu.zero_before[1],
                ))
                .with(flag(
                    "alone/positive-after-completion/GPUStartTime",
                    gpu.positive_after[0],
                ))
                .with(flag(
                    "alone/positive-after-completion/GPUEndTime",
                    gpu.positive_after[1],
                ))
                .with(flag("alone/gpu-interval-is-one-point", gpu.one_point))
                .with(flag(
                    "alone/gpu-starts-no-earlier-than-the-kernel-end",
                    gpu.starts_no_earlier_than_the_kernel_end,
                )),
        };
        let behind = &self.behind;
        node = node
            .with(Node::leaf(
                "behind-a-fill/completed-handler-order",
                Observed::value(Value::Lines(behind.handler_order.clone())),
            ))
            .with(status(
                "behind-a-fill/fills-status-in-the-empty-ones-completed-handler",
                &behind.fills_status_in_the_empty_ones_handler,
            ))
            .with(status(
                "behind-a-fill/status/after-the-wait",
                &behind.after_the_wait,
            ))
            .with(flag(
                "behind-a-fill/kernel-starts-no-earlier-than-the-fills-kernel-end",
                behind.kernel_starts_no_earlier_than_the_fills_end,
            ));
        match &behind.gpu {
            Err(failure) => node.with(Node::leaf(
                "behind-a-fill/gpu-times",
                Observed::value(failure.recorded()),
            )),
            Ok(gpu) => node
                .with(Node::leaf(
                    "behind-a-fill/gpu-times",
                    Observed::value(Value::ok(Value::Absent)),
                ))
                .with(flag(
                    "behind-a-fill/gpu-interval-is-one-point",
                    gpu.one_point,
                ))
                .with(flag(
                    "behind-a-fill/gpu-starts-no-earlier-than-the-fills-gpu-end",
                    gpu.starts_no_earlier_than_the_fills_end,
                )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn completed() -> Status {
        Status::read("Completed")
    }

    /// Metal's shape: zeros before, a point after the kernel end, behind the fill's end.
    fn held() -> EmptyCommandBuffers {
        EmptyCommandBuffers {
            alone: Alone::new(
                [Status::read("Scheduled"), completed(), completed()],
                ([0.0, 0.0], [100.0, 100.00001]),
                Ok(([0.0, 0.0], [100.00002, 100.0000202])),
            ),
            behind: Behind::new(
                vec![FILL.to_owned(), EMPTY.to_owned()],
                [completed(), completed()],
                (101.0, 100.9),
                Ok(([101.0, 101.5], [101.6, 101.6])),
            ),
        }
    }

    /// Each flag reads the times, and fails on the times that break it.
    #[test]
    fn each_flag_fails_on_the_times_that_break_it() {
        let facts = held();
        let alone = facts.alone.gpu.clone().expect("answered");
        assert!(alone.one_point && alone.starts_no_earlier_than_the_kernel_end);
        let behind = facts.behind.gpu.clone().expect("answered");
        assert!(behind.one_point && behind.starts_no_earlier_than_the_fills_end);
        assert!(facts.behind.kernel_starts_no_earlier_than_the_fills_end);
        // A GPU interval of 2 us is no point, nor is one that ends before it starts.
        assert!(!one_point([1.0, 1.000002]));
        assert!(!one_point([1.0, 0.9999999]));
        // A GPU point before the kernel end.
        let early = Alone::new(
            [completed(), completed(), completed()],
            ([0.0, 0.0], [100.0, 100.5]),
            Ok(([0.0, 0.0], [100.4, 100.4])),
        );
        assert!(
            !early
                .gpu
                .expect("answered")
                .starts_no_earlier_than_the_kernel_end
        );
        // A GPU point before the fill's end, and a kernel start before the fill's kernel end.
        let ahead = Behind::new(
            vec![EMPTY.to_owned(), FILL.to_owned()],
            [Status::read("Scheduled"), completed()],
            (100.0, 100.9),
            Ok(([101.0, 101.5], [101.4, 101.4])),
        );
        assert!(!ahead.kernel_starts_no_earlier_than_the_fills_end);
        assert!(
            !ahead
                .gpu
                .expect("answered")
                .starts_no_earlier_than_the_fills_end
        );
    }

    /// Every fact reaches a site of its own, and a refused GPU pair records the refusal and no GPU
    /// flags.
    #[test]
    fn every_fact_reaches_a_site_of_its_own() {
        let node = held().node("facts");
        let names: BTreeSet<String> = (node.children.iter())
            .map(|child| child.name.to_string())
            .collect();
        assert_eq!(names.len(), node.children.len(), "{names:?}");
        // Alone: three statuses, four kernel flags, the ordering, the GPU answer and six GPU
        // flags. Behind: the order, two statuses, the kernel ordering, the GPU answer and two
        // GPU flags.
        assert_eq!(
            node.children.len(),
            (3 + 4 + 1 + 1 + 6) + (1 + 2 + 1 + 1 + 2)
        );
        let mut refused = held();
        refused.alone.gpu = Err(Failure::named("Unsupported", "GpuTimes".to_owned()));
        refused.behind.gpu = Err(Failure::named("Unsupported", "GpuTimes".to_owned()));
        let node = refused.node("facts");
        let names: BTreeSet<String> = (node.children.iter())
            .map(|child| child.name.to_string())
            .collect();
        assert!(names.contains("alone/gpu-times") && names.contains("behind-a-fill/gpu-times"));
        assert!(!names.contains("alone/gpu-interval-is-one-point"));
        assert!(!names.contains("behind-a-fill/gpu-interval-is-one-point"));
        assert!(names.contains("alone/kernel-interval-is-ordered"));
    }
}
