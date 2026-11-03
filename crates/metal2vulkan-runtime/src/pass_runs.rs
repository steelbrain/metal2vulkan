//! Runs of passes over one linear attachment rendered apart, merged once (`linear_merge.rs`).
//!
//! A pass that renders a bound linear attachment through the image its texture keeps copies the
//! box its draws write in, keeps it, renders, and merges the words it changed back into the
//! surface's bytes (`linear_attachment.rs`). A WebGPU frame split at every indexed draw renders
//! hundreds of such passes in one command buffer, each over the same attachment: hundreds of
//! copies in, keeps and merges, each between barriers from every command to every command.
//!
//! - **A run** is a command buffer's passes over one attachment rendered apart through one kept
//!   image, in one batch, each loading and storing it, with nothing between them that may touch
//!   the surface's bytes or the kept image. Its first pass copies its box in and keeps it, as a
//!   pass does alone; each later pass, whose box lies in the run's, renders over what the one
//!   before left in the kept image, with no copy in and no keep; and the merge after its last
//!   pass compares the first pass's keep with the image the last pass left, so it writes every
//!   word one of the passes changed, and no other. A pass whose box reaches past its run's ends
//!   that run and opens the next over its whole render area, which every later box lies in.
//! - **The merge goes last.** Each pass of a run leaves its merge to its batch's submission
//!   ([`RunMerge`], `Deferred::RunMerge`), a segment between its own command buffers and the
//!   next encoder's; a later pass of the run, as its encoder ends, takes it over, and that
//!   segment records nothing. The merge recorded is the run's last pass's, in that pass's place,
//!   between the same barriers. Its uses are noted as each pass ends, as a merge recorded there
//!   notes them, so the tracker orders the work after each pass after a write of the bytes.
//! - **Why the bytes are Metal's.** Within the run, nothing reads the surface's bytes or writes
//!   them another way (below): between two of its passes the bytes are what the first copy in
//!   read, and the kept image holds them, as Metal's tile stores would have left them, merged.
//!   Each pass then renders over what Metal's would load, and the merge after the last writes
//!   each word as the last pass that changed it left it, as the merges after each pass would
//!   have, one after the other; a word changed and changed back is written either way with the
//!   value it holds. Another writer's write of a word no pass changed is kept, as it is by
//!   merges after each pass; one of a word a pass changed races that pass in Metal too. After the
//!   run the bytes are those merges' and, at the merge's end, a barrier from every command to
//!   every command orders all later work after it, as the merge after each pass does.
//! - **What ends a run** (its last merge recorded, the scratch lent on, `merge_scratch.rs`): any
//!   encoder between its passes but a render encoder that touches none of it (a blit, a compute
//!   or a resource state encoder ends it, whatever it names); a render encoder that may touch the
//!   surface's bytes or the kept image: one whose commands named the surface's buffer or its host
//!   range, reached memory they do not name (a root or handle table, a heap, an execution of
//!   render commands, a memory barrier inside the pass), waited for or updated a fence,
//!   resolved, wrote visibility results, or recorded later (a pending draw, a tile dispatch, a
//!   later color group), or that has an attachment over those bytes or rendered through that
//!   image; an event or semaphore wait or a signal, a host step (any new batch, where another
//!   queue's work may come between); a pass whose box reaches past the run's, or one that
//!   cannot join (below); and the command buffer's end. The work after the run then sees the
//!   bytes merged: its last merge lies before it in submission order.
//! - **Which passes join** (each a run's first or a later pass): a pass encoded once, not again:
//!   neither its commands nor its attachments name a buffer over host pages, which a rebind would
//!   move and encode its encoder again (`command_buffer/reencode.rs`), it renders over no
//!   rasterization-rate map, whose reset does the same, and executes no render commands, which a
//!   host step records; outside a predicated loop's laying, which its check may submit again;
//!   of one part and one color group; whose first instance does not record inline; that names
//!   neither the surface's buffer nor its host range and touches nothing past its attachments
//!   (as above); and whose other attachments lie over none of the surface's bytes. Every other
//!   pass over a linear attachment rendered apart merges after itself, as before; an encoder
//!   encoded again changes no run.
//!
//! Counted: `WorkCounts::linear_merges_recorded` (one per run, one per pass that merges alone)
//! and `WorkCounts::linear_merges_taken_over`.

use super::linear_merge::LinearMerge;
use super::resource::ResourceKey;
use crate::device::functions::Vk;
use ash::vk;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// A run's pass's merge, left to its batch's submission (`Deferred::RunMerge`): recorded
/// there unless a later pass of the run took it over.
pub(super) struct RunMerge {
    merge: Arc<LinearMerge>,
    taken: AtomicBool,
}

impl RunMerge {
    /// The merge `merge` of a run's pass, not taken over.
    pub(super) fn new(merge: Arc<LinearMerge>) -> Arc<RunMerge> {
        Arc::new(RunMerge {
            merge,
            taken: AtomicBool::new(false),
        })
    }

    /// Whether a later pass of its run took it over: its segment records nothing.
    pub(super) fn taken(&self) -> bool {
        self.taken.load(Ordering::Acquire)
    }

    /// Take it over: a later pass of its run left its own merge to its batch's submission,
    /// which writes what both changed.
    pub(super) fn take_over(&self) {
        self.taken.store(true, Ordering::Release);
    }

    /// Record the merge, not taken over, into `command_buffer` on its own
    /// (`LinearMerge::record_alone`).
    ///
    /// # Safety
    ///
    /// As for `LinearMerge::record_alone`.
    pub(super) unsafe fn record(&self, device: &Vk, command_buffer: vk::CommandBuffer) {
        // SAFETY: the caller's contract.
        unsafe { self.merge.record_alone(device, command_buffer) }
    }
}

/// What a run's passes have in one: the image they render, and the surface's own image and its
/// subresource (`(level, layer, layers)`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct RunKey {
    pub(super) backing: vk::Image,
    pub(super) surface: vk::Image,
    pub(super) subresource: (u32, u32, u32),
}

/// Where an encoder ends among its command buffer's: the encoders ended before it, and the
/// batches so far, with whether the current one signals (`State::run_point`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct RunPoint {
    pub(super) ended: usize,
    pub(super) batches: usize,
    pub(super) signals: bool,
}

/// A pass's color that may join a run: the run's key, the key of its surface's bytes, and a box
/// (`[x0, y0, x1, y1]`, half-open): the one its draws write in, or a run's.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Joining {
    pub(super) key: RunKey,
    pub(super) bytes: ResourceKey,
    pub(super) area: [u32; 4],
}

/// What a color that may join a run takes as its pass ends ([`PassRuns::end_of`]).
pub(super) enum Joined {
    /// It goes on with the run before it: the run's box, its merge, and its last pass's merge
    /// left to its submission, which the pass takes over once it left its own.
    On {
        area: [u32; 4],
        merge: Arc<LinearMerge>,
        last: Arc<RunMerge>,
    },
    /// The run before it ended there, the box its draws write in reaching past the run's: it
    /// opens one over its whole render area.
    Outgrew,
    /// It opens a run over the box its draws write in.
    Opens,
}

/// An open run: its color's box, its merge, the latest pass's merge left to its submission,
/// where that pass ended, and how many render encoders ended since that touch none of it.
struct Open {
    joining: Joining,
    merge: Arc<LinearMerge>,
    last: Arc<RunMerge>,
    at: RunPoint,
    passed: usize,
}

impl Open {
    /// Whether a pass ending at `now` follows the run's last pass with nothing between but the
    /// render encoders that passed it, in its batch, which signals nothing yet.
    fn follows(&self, now: RunPoint) -> bool {
        now.ended == self.at.ended + 1 + self.passed
            && now.batches == self.at.batches
            && !self.at.signals
            && !now.signals
    }
}

/// Whether the box `inner` lies inside `outer`.
fn within(inner: [u32; 4], outer: [u32; 4]) -> bool {
    inner[0] >= outer[0] && inner[1] >= outer[1] && inner[2] <= outer[2] && inner[3] <= outer[3]
}

/// A command buffer's open runs (the module docs).
#[derive(Default)]
pub(super) struct PassRuns {
    open: Vec<Open>,
}

impl PassRuns {
    /// For a render encoder ending at `now`, whose colors `joining` may join a run: what each
    /// takes, and the merges of the runs it ends, to lend their scratch on. A run goes on with
    /// the color of its key and bytes where the pass follows its last (`Open::follows`) and the
    /// color's box lies in the run's; it ends where the color's box reaches past it
    /// ([`Joined::Outgrew`]), where the pass does not follow it, or where the pass `touches` it
    /// (by the run's key and the key of its surface's bytes); every other run counts the pass
    /// as passed.
    pub(super) fn end_of(
        &mut self,
        now: RunPoint,
        joining: &[Joining],
        touches: impl Fn(&RunKey, ResourceKey) -> bool,
    ) -> (Vec<Joined>, Vec<Arc<LinearMerge>>) {
        let mut joined: Vec<Joined> = joining.iter().map(|_| Joined::Opens).collect();
        let mut ended = Vec::new();
        self.open.retain_mut(|run| {
            let follows = run.follows(now);
            let at = (joining.iter())
                .position(|j| j.key == run.joining.key && j.bytes == run.joining.bytes);
            match at {
                Some(at) if follows && within(joining[at].area, run.joining.area) => {
                    joined[at] = Joined::On {
                        area: run.joining.area,
                        merge: Arc::clone(&run.merge),
                        last: Arc::clone(&run.last),
                    };
                    false
                }
                Some(at) if follows => {
                    joined[at] = Joined::Outgrew;
                    ended.push(Arc::clone(&run.merge));
                    false
                }
                _ if !follows || touches(&run.joining.key, run.joining.bytes) => {
                    ended.push(Arc::clone(&run.merge));
                    false
                }
                _ => {
                    run.passed += 1;
                    true
                }
            }
        });
        (joined, ended)
    }

    /// Open (or go on with) the run of `joining`, whose pass ending at `now` left its merge
    /// `last`, of `merge`, to its batch's submission.
    pub(super) fn open(
        &mut self,
        joining: Joining,
        merge: Arc<LinearMerge>,
        last: Arc<RunMerge>,
        now: RunPoint,
    ) {
        self.open.push(Open {
            joining,
            merge,
            last,
            at: now,
            passed: 0,
        });
    }
}
