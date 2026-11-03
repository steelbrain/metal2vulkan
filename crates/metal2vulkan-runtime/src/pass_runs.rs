//! Runs of passes over one attachment merged once: a linear attachment rendered apart
//! (`linear_merge.rs`), or a shadowed texture merged back (`shadow.rs`, `shadow_merge.rs`).
//!
//! A pass that renders a bound linear attachment through the image its texture keeps copies the
//! box its draws write in, keeps it, renders, and merges the words it changed back into the
//! surface's bytes (`linear_attachment.rs`). A pass that renders a shadowed texture whose truth
//! another writer may write (a no-copy buffer's host pages, a `Shared` buffer's bytes) keeps the
//! shadow's texels before it and merges the words it changed back as its encoder ends
//! (`shadow.rs`'s **Merged back**); its encoder takes the image the one before it left holding
//! the truth, with no copy in (`shadow::ShadowCarry`). A WebGPU frame split at every indexed draw,
//! or a window server drawing each layer in a pass of its own, renders hundreds of such passes in
//! one command buffer, each over the same attachment: hundreds of keeps and merges (and, for a
//! linear attachment, copies in), each between barriers from every command to every command.
//!
//! - **A run** is a command buffer's passes over one attachment (a linear attachment rendered apart
//!   through one kept image, or one shadowed texture), in one batch, each loading and storing it,
//!   with nothing between them that may touch its bytes or the image the passes render; a shadowed
//!   texture's opens at the second such pass (**The merge goes last**). Its first pass copies in
//!   (or takes the image carried) and keeps as a pass does alone; each later pass renders over what
//!   the one before left in the image, with no copy in and no keep; and the merge after its last
//!   pass compares the first pass's keep with the image the last pass left, so it writes every word
//!   one of the passes changed, and no other. A linear attachment's pass whose box reaches past its
//!   run's ends that run and opens the next over its whole render area, which every later box lies
//!   in. A shadowed texture's pass that writes outside the rectangle its run's keep holds merges
//!   what the run kept before it and keeps every texel again; one that reads outside the texels its
//!   image holds (the first pass copied in its scissors' alone) merges it and copies every texel in
//!   first (`ShadowSync::before`, as a command of one encoder would); and the run goes on over that
//!   keep, every later pass inside both.
//! - **The merge goes last.** Each pass of a run leaves its merge to its batch's submission
//!   ([`RunMerge`], `Deferred::RunMerge`), a segment between its own command buffers and the next
//!   encoder's; a later pass of the run, as its encoder ends, takes it over, and that segment
//!   records nothing. The merge recorded is the run's last pass's, in that pass's place, between
//!   barriers from every command to every command. Its uses are noted as each pass ends, as a merge
//!   recorded there notes them, so the tracker orders the work after each pass after a write of the
//!   bytes. A segment splits its encoder's commands into another `VkCommandBuffer`, and records its
//!   own at the submission, which a pass no later pass goes on from pays for nothing. So a shadowed
//!   texture's pass that would open a run merges in place instead, as its encoder ends, as a pass
//!   alone does, and only marks where a run may start ([`PassRuns::placed`]); the pass that follows
//!   it opens the run, its image carried with no copy in, keeping it again, as nothing may write
//!   the bytes between a run's keep and its merge but that merge (a byte its passes changed and
//!   changed back keeps the bytes' value, below).
//! - **Why the bytes are Metal's.** Within the run, nothing reads the attachment's bytes or
//!   writes them another way (below): between two of its passes the bytes are what the first
//!   copy in read (or what the work before the run left), and the image holds them, as Metal's
//!   tile stores would have left them, merged. Each pass then renders over what Metal's would
//!   load, and the merge after the last writes each word as the last pass that changed it left
//!   it, as the merges after each pass would have, one after the other; a word changed and
//!   changed back is written either way with the value it holds. Another writer's write of a word
//!   no pass changed is kept, as it is by merges after each pass; one of a word a pass changed
//!   races that pass in Metal too, which the program cannot order inside the run (it has no
//!   signal, wait or host step). A shadowed texture's merge that stores whole blocks of its rows
//!   (`shadow.rs`'s **Stored attachments**) stores each word of a block holding a change that no
//!   pass changed as the bytes held it when the merge read it, as a pass's merge alone does: only
//!   a write made between that read and the store is undone. After the run the bytes are those
//!   merges' and, at the merge's end, a barrier from every command to every command orders all
//!   later work after it, as the merge after each pass does.
//! - **What ends a run** (its last merge recorded where its last pass left it, its scratch lent
//!   on, `merge_scratch.rs`): any encoder between its passes but a render encoder that touches
//!   none of it (a blit, a compute or a resource state encoder ends it, whatever it names); a
//!   render encoder that may touch the attachment's bytes or the image: one whose commands named
//!   the bytes' buffer or its host range, reached memory they do not name (a root or handle
//!   table, a heap, an execution of render commands, a memory barrier inside the pass), waited
//!   for or updated a fence, resolved, wrote visibility results, or recorded later (a pending
//!   draw, a tile dispatch, a later color group), or that has an attachment over those bytes or
//!   rendered through that image, or whose draws or attachments take a shadowed texture over
//!   them (the run's own included, bound to a draw); an event or semaphore wait or a signal, a
//!   host step (any new batch, where another queue's work may come between); a pass whose box
//!   reaches past a linear run's, or one that cannot join (below); and the command buffer's end.
//!   A shadowed texture's run ends too where its encoder no longer carries the image
//!   (`shadow::ShadowCarry`: dropped at a fence's wait, or a write of the truth another way).
//!   The work after the run then sees the bytes merged: its last merge lies before it in
//!   submission order.
//! - **Which passes join** (each a run's first or a later pass): a pass of one part and one
//!   color group, of one render pass instance for a shadowed texture, outside a predicated loop's
//!   laying, whose first instance does not record inline, that names neither the bytes' buffer
//!   nor its host range and touches nothing past its attachments (as above), whose other
//!   attachments lie over none of the bytes, and that renders over no rasterization-rate map.
//!   A linear attachment's pass joins only where it is encoded once: neither its commands nor its
//!   attachments name a buffer over host pages, which a rebind would move and encode its encoder
//!   again (`command_buffer/reencode.rs`), and it executes no render commands. A shadowed
//!   texture's: one loaded and stored as a color (no stand-in, no planes, no resolve), whose
//!   truth is a buffer's bytes it merges back into, laid out as one image's texels. Every other
//!   pass merges after itself, as before; an encoder encoded again changes no run.
//! - **Encoded again.** A shadowed texture's run may hold passes over host pages, which a rebind
//!   before the command buffer's first submission encodes again (`command_buffer/reencode.rs`):
//!   each pass whose commands named a buffer that moved, and no other. Such a pass leaves its
//!   run ([`RunMerge::leave`]): encoded again, it copies in, keeps and merges after itself as a
//!   pass alone does, and its runs' passes before it and after it form runs of their own. The
//!   merge it had taken over records after all, at the pass before it, so the bytes hold what
//!   the passes before it wrote when it copies them in; and where a pass after it had taken its
//!   merge over, it keeps the run's scratch again after its own merge
//!   ([`RunMerge::keep_after`]), so that pass's run merges what the passes after it change
//!   against what the bytes hold then. The merges of a pass encoded again take no scratch the
//!   command buffer or its queue lends (`ShadowMerge::new`): its place may lie inside a run's
//!   span. A run's merge records from what its pass resolved, the bytes its encoder named
//!   (`Deferred::resolved_at_its_calls`): its command buffer's holds on host pages go at the
//!   commit, and a rebind of one encodes the encoders that named it again rather than being
//!   refused.
//!
//! Counted: `WorkCounts::linear_merges_recorded` (one per linear run, one per pass that merges
//! alone) and `WorkCounts::linear_merges_taken_over`; a shadowed texture's run's keep and merge
//! once each, and those of the pass merged in place before it, in `WorkCounts::shadow_copies`
//! (its merge as it records) and `WorkCounts::shadow_texels_kept`, each scratch planned once
//! (`WorkCounts::shadow_merge_dispatches_planned`).

use super::linear_merge::LinearMerge;
use super::resource::ResourceKey;
use super::shadow::{ShadowCopy, Step};
use super::shadow_merge::{MergeLease, ShadowMerge};
use super::tracker::Barrier;
use crate::device::functions::Vk;
use ash::vk;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// The merge a run's pass leaves to its batch's submission: a linear attachment's
/// (`LinearMerge::record_alone`), or a shadowed texture's, a merge of its shadow's copies.
pub(super) enum RunWork {
    Linear(Arc<LinearMerge>),
    Shadow(ShadowCopy),
}

/// A run's pass's merge, left to its batch's submission (`Deferred::RunMerge`): recorded there
/// unless a later pass of the run took it over and was not encoded again since.
pub(super) struct RunMerge {
    work: RunWork,
    taken: AtomicBool,
    /// The pass that took it over was encoded again, and left the run ([`leave`](Self::leave)).
    given_back: AtomicBool,
    /// The merge of the run's pass before it, which this one took over.
    took: Option<Arc<RunMerge>>,
}

impl RunMerge {
    /// The merge `work` of a run's pass, not taken over, that took `took` over.
    pub(super) fn new(work: RunWork, took: Option<Arc<RunMerge>>) -> Arc<RunMerge> {
        Arc::new(RunMerge {
            work,
            taken: AtomicBool::new(false),
            given_back: AtomicBool::new(false),
            took,
        })
    }

    /// Whether a later pass of its run took it over, and still holds it: its segment records
    /// nothing.
    pub(super) fn taken(&self) -> bool {
        self.taken.load(Ordering::Acquire) && !self.given_back.load(Ordering::Acquire)
    }

    /// Take it over: a later pass of its run left its own merge to its batch's submission,
    /// which writes what both changed.
    pub(super) fn take_over(&self) {
        self.taken.store(true, Ordering::Release);
    }

    /// Its pass was encoded again, in place of its first recording and this merge (the module's
    /// **Encoded again**): the merge it took over records after all.
    pub(super) fn leave(&self) {
        if let Some(before) = &self.took {
            before.given_back.store(true, Ordering::Release);
        }
    }

    /// The keep its pass, encoded again, records after itself where a later pass of its run took
    /// its merge over: of the image that pass went on over into the scratch its run's merge
    /// compares (the module's **Encoded again**). `None` for a linear attachment's run, whose
    /// passes are never encoded again, and a merge no later pass took over.
    pub(super) fn keep_after(&self) -> Option<ShadowCopy> {
        let RunWork::Shadow(copy) = &self.work else {
            return None;
        };
        let Step::Merge(merge) = &copy.step else {
            return None;
        };
        self.taken.load(Ordering::Acquire).then(|| ShadowCopy {
            copies: Arc::clone(&copy.copies),
            step: Step::Keep(Arc::clone(merge)),
            kept: false,
            area: None,
            cause: None,
            moves: None,
        })
    }

    /// The shadowed texture's merge it records, for the counters; `None` for a linear
    /// attachment's.
    pub(super) fn shadow(&self) -> Option<&ShadowCopy> {
        match &self.work {
            RunWork::Shadow(copy) => Some(copy),
            RunWork::Linear(_) => None,
        }
    }

    /// Record the merge, not taken over, into `command_buffer` on its own, between barriers from
    /// everything before and to everything after (`LinearMerge::record_alone`).
    ///
    /// # Safety
    ///
    /// As for `LinearMerge::record_alone`, or `ShadowCopy::record` for a shadowed texture's:
    /// `command_buffer` is recording, outside a render pass; the merge, its images, its scratch
    /// and the texture's bytes are held until it completes, and the images rest in `GENERAL`.
    pub(super) unsafe fn record(&self, device: &Vk, command_buffer: vk::CommandBuffer) {
        // SAFETY: the caller's contract.
        unsafe {
            match &self.work {
                RunWork::Linear(merge) => merge.record_alone(device, command_buffer),
                RunWork::Shadow(copy) => {
                    Barrier::everything().record(device, command_buffer);
                    copy.record(device, command_buffer, None);
                    Barrier::everything().record(device, command_buffer);
                }
            }
        }
    }
}

/// What a run's passes have in one: for a linear attachment, the image they render, and the
/// surface's own image and its subresource (`(level, layer, layers)`); for a shadowed texture,
/// its shadow's identity (`ShadowCopies::shadow_key`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum RunKey {
    Linear {
        backing: vk::Image,
        surface: vk::Image,
        subresource: (u32, u32, u32),
    },
    Shadow(ResourceKey),
}

/// Where an encoder ends among its command buffer's: the encoders ended before it, and the
/// batches so far, with whether the current one signals (`State::run_point`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct RunPoint {
    pub(super) ended: usize,
    pub(super) batches: usize,
    pub(super) signals: bool,
}

/// A pass's color that may join a run: the run's key, the key of its bytes (a linear
/// attachment's surface's, a shadowed texture's truth), and, for a linear attachment, a box
/// (`[x0, y0, x1, y1]`, half-open): the one its draws write in, or a run's.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Joining {
    pub(super) key: RunKey,
    pub(super) bytes: ResourceKey,
    pub(super) area: [u32; 4],
}

impl Joining {
    /// Whether it may go on with `run`, of its key: a linear attachment's box lies in the run's;
    /// a shadowed texture's always may (a write outside its run's keep merges and keeps again,
    /// the module's **A run**).
    fn fits(&self, run: &Joining) -> bool {
        match self.key {
            RunKey::Linear { .. } => within(self.area, run.area),
            RunKey::Shadow(_) => true,
        }
    }
}

/// What an open run holds for its next pass: a linear attachment's merge, its scratch lent on as
/// the run ends; or the merge of a shadowed texture's run, whose keep it compares, with the hold
/// on its scratch or kept image, which no other merge takes until the run ends.
pub(super) enum RunHeld {
    Linear(Arc<LinearMerge>),
    Shadow {
        merge: Arc<ShadowMerge>,
        lease: Option<MergeLease>,
    },
}

impl RunHeld {
    /// The run ended: its merge lies before every command recorded from here on, so its
    /// scratch is lent on (`merge_scratch.rs`), and a shadowed texture's kept image goes back to
    /// its queue (`texture/kept_image.rs`).
    pub(super) fn end(self) {
        match self {
            RunHeld::Linear(merge) => merge.lend(),
            RunHeld::Shadow { lease, .. } => drop(lease),
        }
    }
}

/// What a color that may join a run takes as its pass ends ([`PassRuns::end_of`]).
pub(super) enum Joined {
    /// It goes on with the run before it: the run's box (a linear attachment's), what the run
    /// holds, and its last pass's merge left to its submission, which the pass takes over once
    /// it left its own.
    On {
        area: [u32; 4],
        held: RunHeld,
        last: Arc<RunMerge>,
    },
    /// The run before it ended there, the box its draws write in reaching past the run's: it
    /// opens one over its whole render area.
    Outgrew,
    /// It opens a run over the box its draws write in.
    Opens,
    /// It follows a shadowed texture's pass that merged in place ([`PassRuns::placed`]): it
    /// opens a run, keeping the texture again, its merge left to its submission.
    Placed,
}

/// An open run: its color, what it holds and the latest pass's merge left to its submission
/// (neither for a pass merged in place, [`PassRuns::placed`]), where that pass ended, and how many
/// render encoders ended since that touch none of it.
struct Open {
    joining: Joining,
    left: Option<(RunHeld, Arc<RunMerge>)>,
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
    /// takes, and what the runs it ends hold, to let go of ([`RunHeld::end`]). A run goes on
    /// with the color of its key and bytes where the pass follows its last (`Open::follows`) and
    /// the color fits it ([`Joining::fits`]); it ends where a linear color's box reaches past it
    /// ([`Joined::Outgrew`]), where the pass does not follow it, or where the pass `touches` it
    /// (by the run's key and the key of its bytes); every other run counts the pass as passed.
    pub(super) fn end_of(
        &mut self,
        now: RunPoint,
        joining: &[Joining],
        touches: impl Fn(&RunKey, ResourceKey) -> bool,
    ) -> (Vec<Joined>, Vec<RunHeld>) {
        let mut joined: Vec<Joined> = joining.iter().map(|_| Joined::Opens).collect();
        let mut ended = Vec::new();
        let mut open = Vec::with_capacity(self.open.len());
        for mut run in self.open.drain(..) {
            let follows = run.follows(now);
            let at = (joining.iter())
                .position(|j| j.key == run.joining.key && j.bytes == run.joining.bytes);
            match at {
                Some(at) if follows && joining[at].fits(&run.joining) => {
                    joined[at] = match run.left {
                        Some((held, last)) => Joined::On {
                            area: run.joining.area,
                            held,
                            last,
                        },
                        None => Joined::Placed,
                    };
                }
                Some(at) if follows => {
                    joined[at] = Joined::Outgrew;
                    ended.extend(run.left.map(|(held, _)| held));
                }
                _ if !follows || touches(&run.joining.key, run.joining.bytes) => {
                    ended.extend(run.left.map(|(held, _)| held));
                }
                _ => {
                    run.passed += 1;
                    open.push(run);
                }
            }
        }
        self.open = open;
        (joined, ended)
    }

    /// Open (or go on with) the run of `joining`, holding `held`, whose pass ending at `now` left
    /// its merge `last` to its batch's submission.
    pub(super) fn open(
        &mut self,
        joining: Joining,
        held: RunHeld,
        last: Arc<RunMerge>,
        now: RunPoint,
    ) {
        self.open.push(Open {
            joining,
            left: Some((held, last)),
            at: now,
            passed: 0,
        });
    }

    /// Mark where a run of `joining`'s shadowed texture may start: its pass ending at `now`
    /// merged in place (the module's **The merge goes last**), and the pass that follows it opens
    /// the run ([`Joined::Placed`]).
    pub(super) fn placed(&mut self, joining: Joining, now: RunPoint) {
        self.open.push(Open {
            joining,
            left: None,
            at: now,
            passed: 0,
        });
    }

    /// End every open run as the command buffer commits, after which none goes on: what each
    /// holds, to let go of ([`RunHeld::end`]).
    pub(super) fn close(&mut self) -> Vec<RunHeld> {
        self.open
            .drain(..)
            .filter_map(|run| run.left.map(|(held, _)| held))
            .collect()
    }
}
