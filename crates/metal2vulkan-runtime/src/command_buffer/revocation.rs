//! A queue Metal stops serving after GPU errors its command buffers caused.
//!
//! Measured on the M3 Max with macOS 27.0.1 (about sixty command buffers over five runs; each a
//! blit, a pass of one draw of primitive type 9, a blit, an event signal and a blit, then one
//! committed at once, one committed some milliseconds on, and one committed after the first
//! completed, on queues used in turn):
//!
//! - **Caused errors count, per queue.** A command buffer that fails with
//!   `MTLCommandBufferErrorPageFault` (3) or `MTLCommandBufferErrorTimeout` (2, a GPU hang)
//!   counts against its queue. One that fails as a victim of the GPU's recovery
//!   (`MTLCommandBufferErrorInternal`, "Discarded (victim of GPU error/recovery)") does not:
//!   three on one queue, and a victim then a fault, left the queue serving. A command buffer
//!   Metal's watchdog fails counts too: on the same Mac two command buffers of one queue, each
//!   waiting for a shared event no one signalled, failed in turn with
//!   `MTLCommandBufferErrorTimeout` ("Caused GPU Timeout Error
//!   (00000002:kIOGPUCommandBufferCallbackErrorTimeout)") 5.00 s after their commits, and the
//!   fill committed after them failed with `MTLCommandBufferErrorAccessRevoked` 0.24 ms after its
//!   commit, nothing of it run, where a new queue's ran.
//! - **The second one revokes the queue.** Once the queue's second command buffer with a caused
//!   error completes (two faults, two hangs, a hang then a fault), every command buffer committed
//!   on the queue after that fails at its commit: `Scheduled` and completed 0.02 to 0.06 ms after
//!   its commit, with status `Error` and `MTLCommandBufferErrorAccessRevoked` (4, "Ignored (for
//!   causing prior/excessive GPU errors)
//!   (00000004:kIOGPUCommandBufferCallbackErrorSubmissionsIgnored)"); none of its work runs, none
//!   of its event signals fires, and its GPU and kernel times span nothing. Two seconds on it was
//!   so still. One committed before that completion, 4.5 ms into the second fault, ran; one
//!   committed 0.5 ms after it, and later ones, did not.
//! - **Other queues serve on.** A new queue of the same device ran normally after others were
//!   revoked, its own first fault or hang leaving it serving.
//! - The second faulting command buffer itself answers as the first did (its other encoders ran
//!   in 9 of 11 runs, nothing of it in 2), as does one committed at once after it.
//!
//! The runtime counts the command buffers of a queue that complete with a failure of its own
//! making that Metal's GPU causes: [`CommandBufferError::PageFault`] as the runtime fails an
//! encoder Metal's GPU faults (a draw of primitive type 9 over indices, a render encoder's
//! execution of an indirect command buffer under a state it cannot inherit), and
//! [`CommandBufferError::GpuHang`]; and one a program commits as failed in their place
//! (`CommandBuffer::commit_failed`, [`CommandBufferError::Injected`] of Metal's code of either,
//! 3 or 2), which stands for what Metal's GPU caused. The watchdog's `GpuHang` (`stall.rs`) counts
//! as Metal's timeout does. The count is the queue's over every place of it: a recovery's new
//! place of the queue keeps it (`queue.rs`), as Metal's queue, which no recovery replaces, keeps
//! its own. Of the other codes it counts none: a
//! victim's (1) does not count on Metal, and the rest are not caused GPU errors or not measured.
//! A failure the driver relays is the driver's own queue's, which Metal counts itself under
//! MoltenVK. From the completion of the second one, before its
//! status reads final, the queue is revoked: a command buffer committed on it then fails with
//! [`CommandBufferError::AccessRevoked`] as its commit's work, from its first encoder, its event
//! signals dropped (`commit_now`). One committed before runs as committed, whatever its event
//! waits (Metal's answer for one held by a wait across the revocation is not measured).

use crate::error::CommandBufferError;
use std::sync::atomic::{AtomicU32, Ordering};

/// The command buffers whose GPU error Metal counts against their queue that revoke it.
const REVOKING: u32 = 2;

/// A queue's count of its command buffers that completed with a GPU error they caused (the
/// module docs).
#[derive(Default)]
pub(crate) struct QueueErrors {
    caused: AtomicU32,
}

impl QueueErrors {
    /// A command buffer of the queue completes with `error` (`None` for none), before its status
    /// reads final: counted where the GPU caused it ([`caused`]).
    pub(super) fn completed(&self, error: Option<CommandBufferError>) {
        if error.is_some_and(caused) {
            self.caused.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// Whether Metal ignores the queue's command buffers committed now: its second command buffer
    /// with a caused error completed.
    pub(super) fn revoked(&self) -> bool {
        self.caused.load(Ordering::SeqCst) >= REVOKING
    }
}

/// Whether a command buffer that completed with `error` counts against its queue (the module
/// docs): a page fault or a hang, the runtime's or one a program injected in its place by its
/// Metal code.
pub(crate) fn caused(error: CommandBufferError) -> bool {
    let of = |caused: CommandBufferError| caused.metal_code();
    match error {
        CommandBufferError::PageFault | CommandBufferError::GpuHang => true,
        CommandBufferError::Injected(code) => {
            code == of(CommandBufferError::PageFault) || code == of(CommandBufferError::GpuHang)
        }
        _ => false,
    }
}
