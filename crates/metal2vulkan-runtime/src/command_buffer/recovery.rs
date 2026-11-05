//! The recoveries of a device's GPU from a fault or a hang, as Metal's GPU makes them, and what
//! they do to a predicated loop that runs again meanwhile (`rerun.rs`).
//!
//! Metal's GPU recovers from a page fault or a hang device-wide: a predicated loop running on the
//! device when it does stops there. Measured on the M3 Max (macOS 27.0.1): a `While` loop of
//! 160,000 passes, its unit's encoder followed by a mark and a blit, committed on a queue; 50 ms on,
//! two command buffers drawing primitive type 9 committed on that queue or another one.
//! - A fault (an indexed draw, `PageFault`; 6 runs, committed 50 ms on or at once) failed the
//!   loop's command buffer as a victim (`MTLCommandBufferErrorInternal`,
//!   `CommandBufferError::GpuRecoveryVictim`) 0.0 to 0.02 ms before the faulting one completed,
//!   3,463 to 39,098 passes in. The loop ran on through the fault's 8 ms hold, and nothing after
//!   its encoder ran.
//! - A hang (a direct draw, `GpuHang`; 2 runs) ended the loop with no error 0.05 to 0.08 ms
//!   before the hanging one failed, about 53,450 passes in. Its command buffer completed cleanly
//!   and nothing after the loop's encoder ran.
//! - A loop committed behind faults that recovered before it started ran all its passes.
//!
//! The runtime answers so: each hold (`hang.rs`) raises the device's recovery from what it holds
//! for as its wait goes, the GPU's recovery ([`Recoveries::raise`]). A fault's hold that later
//! failures follow raises the fault's once it read `Scheduled`, and the later ones' as its wait
//! goes. A command buffer a program commits as failed with Metal's code of a page fault or a hang
//! (`CommandBuffer::commit_failed`, `CommandBufferError::Injected` of 3 or 2) stands for that
//! error: the recovery from it is raised as it completes, beside its queue's count
//! (`revocation.rs`, [`injected`]). Before each run again a loop asks for the first recovery
//! since its command buffer's first submission ([`Recoveries::since`]):
//! - after a fault's, it fails as a victim;
//! - after a hang's, it stops with no error ([`cut`]): from the check's part on, nothing of the
//!   command buffer runs, and it completes cleanly. Whether its later event signals fire is not
//!   measured; they fire.
//!
//! A fault committed on the loop's own queue behind the loop waits for the loop to end in the
//! runtime, as every batch of a queue does, where Metal's GPU runs it during the loop
//! (`docs/FAST-PATHS.md`, known gaps).

use super::super::gpu_time;
use super::rerun::LoopCheck;
use super::{lock, CommandBufferShared, State};
use crate::deferred::Deferred;
use crate::encoder::Recorded;
use crate::error::CommandBufferError;
use std::sync::atomic::{AtomicU64, Ordering};

/// A device's GPU recoveries (the module docs): when the last from a fault and the last from a
/// hang came, in host nanoseconds (`gpu_time::host_nanos`); 0 for none.
#[derive(Default)]
pub(crate) struct Recoveries {
    fault: AtomicU64,
    hang: AtomicU64,
}

/// What the GPU recovered from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Recovery {
    /// A page fault, or a draw it fails as a victim of its recovery
    /// (`CommandBufferError::PageFault`, `GpuRecoveryVictim`).
    Fault,
    /// A hang (`CommandBufferError::GpuHang`).
    Hang,
}

impl Recoveries {
    /// The GPU recovered from `recovery`, now.
    pub(super) fn raise(&self, recovery: Recovery) {
        let at = match recovery {
            Recovery::Fault => &self.fault,
            Recovery::Hang => &self.hang,
        };
        at.fetch_max(gpu_time::host_nanos(), Ordering::AcqRel);
    }

    /// The GPU's first recovery after host time `since` (nanoseconds), of each kind's last:
    /// `None` where none came since.
    pub(super) fn since(&self, since: u64) -> Option<Recovery> {
        let after = |at: &AtomicU64| Some(at.load(Ordering::Acquire)).filter(|&at| at > since);
        match (after(&self.fault), after(&self.hang)) {
            (Some(fault), Some(hang)) if hang < fault => Some(Recovery::Hang),
            (Some(_), _) => Some(Recovery::Fault),
            (None, Some(_)) => Some(Recovery::Hang),
            (None, None) => None,
        }
    }
}

/// What the GPU recovers from as a command buffer that failed with `error` completes: a page
/// fault or a hang a program committed as failed in its place by its Metal code (the module
/// docs); `None` for every other error. The runtime's own raise theirs at their hold's release
/// (`hang.rs`).
pub(super) fn injected(error: CommandBufferError) -> Option<Recovery> {
    let CommandBufferError::Injected(code) = error else {
        return None;
    };
    match code {
        _ if code == CommandBufferError::PageFault.metal_code() => Some(Recovery::Fault),
        _ if code == CommandBufferError::GpuHang.metal_code() => Some(Recovery::Hang),
        _ => None,
    }
}

/// The first recovery of `buffer`'s device since its first submission (the module docs).
pub(super) fn recovered(buffer: &CommandBufferShared) -> Option<Recovery> {
    let since = buffer.kernel.start();
    buffer.device.shared().recoveries.since(since)
}

/// The loop of `check` stopped at a hang's recovery (the module docs): every part of `buffer`
/// from the one holding the check on runs nothing, each keeping its place among the batches with
/// its fences' waits and updates alone (`State::fail_part`), and the command buffer completes
/// with no error.
pub(super) fn cut(buffer: &CommandBufferShared, check: &LoopCheck) {
    let mut state = lock(&buffer.state);
    let State {
        recorded, retained, ..
    } = &mut *state;
    let holds_check = |part: &Recorded| {
        part.segments.iter().any(|segment| match &segment.work {
            Deferred::Loop(held) => std::ptr::eq(&**held, check),
            _ => false,
        })
    };
    let Some(from) = recorded.iter().position(holds_check) else {
        return;
    };
    for part in &mut recorded[from..] {
        State::fail_part(part, retained);
    }
}
