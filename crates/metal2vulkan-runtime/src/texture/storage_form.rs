//! A fetched render target's storage form: a color render target that a framebuffer fetch through
//! the pixel interlock reads, whose image lacks `STORAGE` (the device lays it out otherwise with
//! it, `Device::with_free_usage`), renders each fetching pass into a scratch image copied in and
//! out (`fetch-through-scratch`, `attachment_stand_in.rs`). The first such pass notes the texture;
//! a later commit gives it an image with `STORAGE`, its texels copied in once, and every later pass
//! fetches it in place.
//!
//! The same switch moves a shadow over host pages in place: a texture made before its pages'
//! in-place verdicts were known is a shadow (`ShadowCause::CheckPending`, `in_place.rs`), an
//! unknown and not a lack, so it waits for them, and a later commit gives it the image in place
//! over its texels that it would have had, had they been known as it was made. So does a shadow
//! over a buffer of the device's pooled memory made before that class's verdicts
//! (`in_place::PagesClass::Pooled`): it moves to the `DRM_FORMAT_MOD_LINEAR` image at Metal's
//! pitch bound in the buffer's memory that it would have had.
//!
//! - **Moves in place.** Such a shadow waits in [`StorageForms::moves`] until its pages' class
//!   knows every verdict (`Device::in_place_verdicts_given`, as the check settles, from its
//!   thread, before a call waiting for its verdicts wakes, and at the texture's making, which take
//!   no handle of a program's texture).
//!   Where its usage and those verdicts take an image in place (`in_place::in_place`), as they
//!   would for a texture made then, it waits for a switch; else it stays a shadow, as such a
//!   texture is. At the switch's commit the image in place is made as the texture's making makes
//!   it (over host pages `Device::in_place_image`: the declared plan, with the feedback loop usage
//!   and the storage usage where the device makes them so, noted for a rebind and for its first
//!   move; over the pooled memory `Device::pooled_move_image`: the first of the plans with the
//!   feedback loop usage and the usage the texture lacks the device makes it of, noted for its
//!   first move); where the device makes none (a refused pitch, its placement), or the buffer's
//!   bytes were detached or copied meanwhile, or a binding or a pass took a usage or render
//!   shadow of it, it stays a shadow for good. Where every encoder of the switch's command buffer
//!   ran, the texture presents the image in place from then on (`TextureShared::moved`), counted
//!   as a texture made with it is (over host pages as one in place over its class, and in
//!   `WorkCounts::host_textures_moved_in_place`; over the pooled memory as an image at Metal's
//!   pitch, and in `WorkCounts::pooled_textures_moved_in_place`); its slots'
//!   elements are written again as a storage form's are. No copy moves texels: between encoders
//!   the shadow holds nothing the buffer lacks (`shadow.rs`, "The buffer is the truth"), and the
//!   image in place is the buffer's bytes, whose uses hazard tracking orders after the shadow's
//!   copies back and merges of every command buffer committed before (both name the buffer's
//!   bytes). A view made of the texture before the move keeps the shadow, which goes on copying
//!   in and back as a shadow over the same bytes beside an image in place does.
//! - **The shadow goes after the move.** The switch's last part writes the shadow's image, a
//!   tracked write of every level and layer, so hazard tracking orders the committing command
//!   buffer after every use of the shadow committed before, on any queue, and that command buffer
//!   holds a [`ShadowAfterMove`]: once it completed, where the texture moved in and no view of it
//!   made before the move presents the shadow, the shadow's `VkImage`, views and memory go
//!   (`Image::let_go_unused`), and its merges' kept scratch buffers, counted
//!   (`WorkCounts::host_texture_shadows_freed_after_move`, over the pooled memory
//!   `WorkCounts::pooled_texture_shadows_freed_after_move`). Nothing names the shadow by then:
//!   every later command presents the image in place, and the commit bound no heap set. A view
//!   of the texture made before the move, or a shadow made over the shadow's image (a slot
//!   binding's usage shadow, a view's, which holds it as its truth, `shadow::ImageTruth`), keeps
//!   it until that and the texture went, as does an untracked texture, whose uses on other queues
//!   the program orders itself.
//!
//! - **Which textures.** A texture's own image (not a view's), the runtime's in memory of its own
//!   (not shareable, not a heap's, a buffer's or the caller's), `Private`, its hazards tracked,
//!   whose plan [`storage_candidate`] gives `STORAGE`, and whose `VkImage`
//!   was never handed out ([`Texture::vk_image`](crate::Texture::vk_image) keeps it for good).
//!   Noted once, at the first fetching pass that renders it into a scratch image for want of
//!   `STORAGE`, or at the first draw that reads it after a barrier in a pass that renders it and
//!   splits the pass for want of `STORAGE` (`render/snapshot.rs`: a later fetch would render it
//!   through a stand-in its snapshot does not see), whose later passes' readers then take the
//!   snapshot. An untracked texture keeps its scratch: the program orders its uses on other
//!   queues with its own fences and events, which the copy below could not follow.
//! - **When: the commit of the only uncommitted command buffer.** A command buffer that commits
//!   while it is the device's only one not committed (`queue.rs`'s
//!   [`Uncommitted`](crate::queue::Uncommitted)), whatever is in flight, takes the switch where
//!   (`command_buffer/storage_form.rs`): every encoder of it may run (no injected failure, no
//!   encoder that fails or faults it), it waits for no event or binary semaphore, leaves no
//!   segment to record, and it bound no heap set; and where the device's scheduler holds no
//!   batch (every command buffer committed before was submitted) and no command buffer that may
//!   read the heap sets is incomplete (no heap ticket is out). A command buffer with a
//!   submission check takes it too: a program that checks every command buffer commits no other
//!   kind.
//! - **How.** Before the commit, the image of the plan with `STORAGE` is made and placed, and a
//!   last part of the command buffer moves it to `GENERAL` and copies every level and layer of
//!   the old image into it: a tracked write of the texture, so hazard tracking orders it after
//!   every use of the texture committed before, on any queue, and every later use after it. The
//!   part is the runtime's: no submission check sees it (it is no encoder of the program's, and
//!   lists no use). The switch rides on the command buffer ([`Riding`]) until its last batch is
//!   submitted, its checks run: the scheduler then settles it, the swap's verdict being whether
//!   every encoder of it ran, and its thread ends it once that pass let go of the lock, before
//!   the command buffer's transition to `Scheduled`, which waits for it as for submission
//!   handlers (`scheduler/mod.rs`): so the program sees the swap, and its announcement, by the
//!   time it can see the command buffer scheduled or completed. A thread that makes a command
//!   buffer meanwhile waits until the switch ended, ending it itself once settled
//!   ([`Device::storage_form_wait`]): it then sees the swap and its count
//!   (`WorkCounts::storage_forms_taken`), which the switch makes before its flag drops; where the
//!   scheduler's thread ended it, the announcement may follow on that thread, still before the
//!   committing command buffer is scheduled. Where every encoder ran, under the image's views lock (which [`Texture::vk_image`](crate::Texture::vk_image)
//!   takes to hand the image out) the new `VkImage` swaps in and the plan with `STORAGE` is the
//!   image's from then on; the old image's views are emptied. The old image, its views and its
//!   memory go as the committing command buffer completes, after every earlier use of them: its
//!   copy waits for those. The texture and its views keep their resource IDs: each texture
//!   slot's generation changes, so the device's texture table and heap sets write their
//!   elements again ([`heap_sets::renew`]) while no command buffer that may read them is
//!   pending. `allocatedSize` keeps the bytes the texture was made with. Where a check failed a
//!   submission of it (or its submission failed), nothing swaps: the old image stays the
//!   texture's, the image made goes with the command buffer, and the texture waits again.
//! - **Nothing names the old image after the swap.** No other command buffer is uncommitted, and
//!   none committed before is held. A new one counts itself as it takes its slot and then reads
//!   the switch's flag ([`StorageForms::switching`]), which is up from before the switch reads
//!   the count until the switch ended: so either the switch sees it and leaves the texture for a
//!   later commit, or it waits until the switch ended (finishing it itself once settled) and its
//!   encoders name the image the texture has then, their uses after the copy in hazard
//!   tracking. It waits at most [`SWITCH_WAIT`], and not at all on the scheduler's thread (a
//!   submission check that makes a command buffer): past that, it gives the switch up, so
//!   nothing swaps, and goes on with the old image. A shadow of the image made before the swap
//!   (a slot binding's usage shadow, as a draw reading its own pass's color attachment binds,
//!   or a view's) names the image's `VkImage` as each of its copies is recorded
//!   (`shadow::ImageTruth`), so its later copies take the image with `STORAGE`.
//! - **Once per texture,** by construction: the storage form is a `OnceLock` of the image, which
//!   is noted at most once. A switch the device refuses (the image's limits, memory) leaves the
//!   texture its scratch, for good; a commit that fails or is given up leaves it waiting.
//! - **Announced** once per device as `fetched-target-storage` (performance): the image now has
//!   `STORAGE`, which on MoltenVK is `ShaderWrite` usage, laid out without lossless compression.
//! - **The gates' cost.** A commit reads two atomics, [`StorageForms::switching`] and
//!   [`StorageForms::wanted`], 0 wherever no texture waits; a new command buffer reads one,
//!   `switching`, up only during a switch. Counting the uncommitted command buffers adds one
//!   atomic add to a slot's taking and one subtraction to its commit (or its drop uncommitted),
//!   under its queue's lock. The stall: once per texture, the next command buffer made after a
//!   commit that took a switch waits until the scheduler's thread submitted it (a checked
//!   commit's after its check ran) and ended the switch, as a commit submits nothing itself
//!   (`scheduler/inbox.rs`).

use super::{
    texture_aspects, AliasCount, Image, ImageObjects, Owner, Storage, Texture, TextureShared,
};
use crate::buffer_texture::Texels;
use crate::degradation::Degradation;
use crate::device::lock;
use crate::encoder::EncoderCore;
use crate::error::Error;
use crate::exit_guard;
use crate::framebuffer_fetch::storage_candidate;
use crate::heap_sets;
use crate::image_barrier::{image_barrier, ImageDst};
use crate::image_plan::ImagePlan;
use crate::in_place::{self, Access as InPlaceAccess, Padding, PagesClass, RowEnds, Verdict};
use crate::memory::Placement;
use crate::tracker::Access;
use crate::values::StorageMode;
use crate::work_counts::Counters;
use crate::Device;
use ash::vk;
use metal2vulkan::metal::{TextureType, TextureUsage};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, Weak};
use std::time::{Duration, Instant};

/// The longest a new command buffer waits for a switch under way before it gives it up (the
/// module docs): far above the scheduler's latency in running a check and submitting.
const SWITCH_WAIT: Duration = Duration::from_millis(100);

/// The device's textures waiting for their storage form (the module docs), in its
/// `TextureState`.
#[derive(Default)]
pub(crate) struct StorageForms {
    /// How many textures wait: [`waiting`](Self::waiting)'s length, set under its lock and read
    /// without it, the gate each commit reads.
    wanted: AtomicUsize,
    /// A switch is under way, from before it reads the uncommitted count until it ended: set and
    /// cleared under [`waiting`](Self::waiting)'s lock, the gate each new command buffer reads
    /// once it counted itself, and each commit.
    switching: AtomicBool,
    /// Signaled as a switch ends, and as one rides no more on a command buffer that its
    /// scheduler settled.
    changed: Condvar,
    /// The textures waiting, and the switch under way: the switch's lock.
    waiting: Mutex<Waiting>,
    /// The shadows over host pages (or the device's pooled memory) made before their class's
    /// in-place verdicts, until those are known (`Device::in_place_verdicts_given`): then those
    /// whose verdicts pass wait in [`waiting`](Self::waiting) to move in place, and the others
    /// stay shadows.
    moves: Mutex<Vec<MoveWait>>,
    /// Run by [`close`](Self::close) right after the flag dropped and the lock went, before
    /// the wake: where a thread that ends a switch may be held up (`tests/storage_forms.rs`).
    #[cfg(test)]
    pause: Option<Box<dyn Fn() + Send + Sync>>,
}

/// A shadow over host pages (or the device's pooled memory) made before its class's in-place
/// verdicts, waiting for them ([`StorageForms::moves`]): what decides whether it moves once they
/// are known, read with no handle of the texture taken, so the thread that gives a verdict never
/// holds a program's texture (whose last handle would then go on that thread, its ID given back
/// after the program's release).
struct MoveWait {
    texture: Weak<TextureShared>,
    class: PagesClass,
    /// How its texels lie in the device's lines of memory (`in_place::RowEnds`).
    rows: RowEnds,
    /// Whether a shader's writes may change its rows' padding (`in_place::Padding`).
    padding: Padding,
    usage: TextureUsage,
    /// The bytes of one of its texels.
    texel: u64,
}

impl MoveWait {
    /// Whether it moves, once its class knows every verdict: `None` while one is pending; else
    /// whether its usage and those verdicts take an image in place (`in_place::in_place`), as
    /// they would for a texture made now.
    fn moves(&self, device: &Device) -> Option<bool> {
        let checks = &device.shared().in_place;
        let verdict = |class, access, rows| checks.verdict(class, access, rows);
        let pending = |access| verdict(self.class, access, self.rows) == Verdict::Pending;
        if InPlaceAccess::ALL.into_iter().any(pending) {
            return None;
        }
        let policy = device.compile_device().policy().host_page_textures;
        let pages = (self.class, self.rows, self.texel, self.padding);
        Some(in_place::in_place(policy, self.usage, pages, verdict).is_ok())
    }
}

/// What [`StorageForms`]'s lock guards.
#[derive(Default)]
struct Waiting {
    /// The textures waiting for their storage form.
    textures: Vec<Weak<TextureShared>>,
    /// The switch under way rides on this command buffer's commit (the command buffer holds it).
    riding: Weak<Riding>,
    /// A new command buffer gave the switch under way up: nothing swaps.
    given_up: bool,
}

/// What an image keeps of its storage form (the module docs).
#[derive(Default)]
pub(super) struct Form {
    /// The storage form, once taken.
    taken: OnceLock<Taken>,
    /// The image's texture was noted to take it (at most once).
    noted: AtomicBool,
    /// The image's `VkImage` was handed out: it keeps it. Set and read under the views lock.
    exported: AtomicBool,
}

/// An image's storage form: its plan with `STORAGE`, and the memory its `VkImage` is placed in.
struct Taken {
    plan: ImagePlan,
    placement: Placement,
}

impl Form {
    /// The plan of the storage form, once taken.
    pub(super) fn plan(&self) -> Option<&ImagePlan> {
        self.taken.get().map(|taken| &taken.plan)
    }

    /// The memory the storage form's image is placed in, once taken.
    pub(super) fn placement(&self) -> Option<&Placement> {
        self.taken.get().map(|taken| &taken.placement)
    }
}

/// A `VkImage` with views of it, and the memory it is placed in, that go when this does: the
/// storage form's image made for a commit, until it swaps in; then the image it replaced, with
/// that image's views and first memory. The committing command buffer retains it until it
/// completes, and the switch until it ends.
struct Swapped {
    /// Dropped first: the image and its views.
    objects: ImageObjects,
    /// Then the memory they were in.
    memory: Freed,
}

/// Memory of the device's that goes when this does.
struct Freed {
    device: Device,
    placement: Placement,
}

impl Drop for Freed {
    fn drop(&mut self) {
        // Once the process is exiting, the memory is left to its end (`exit_guard`).
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        let shared = self.device.shared();
        lock(&shared.memory).free(shared.vk(), &self.placement);
    }
}

/// A shadow over host pages (or the device's pooled memory) that may move in place at a commit
/// (the module docs), held by the committing command buffer, which wrote its image last of every
/// command buffer committed before it: as that command buffer completed, or went, where the
/// texture moved in place and it and this alone hold the shadow's image (no view of the texture
/// made before the move presents it, and no shadow made over it holds it as its truth), the
/// image's `VkImage`, views and memory go, and its merges' kept scratch buffers, counted
/// (`WorkCounts::host_texture_shadows_freed_after_move`, or
/// `WorkCounts::pooled_texture_shadows_freed_after_move`). Else nothing: what else holds it keeps
/// the shadow until that and the texture went.
struct ShadowAfterMove {
    /// The texture, held weakly: whether it lives is read, never taken.
    texture: Weak<TextureShared>,
    image: Arc<Image>,
    copies: Arc<crate::shadow::ShadowCopies>,
    /// Set where the shadow moved in place ([`Prepared::move_in`]).
    moved: Arc<AtomicBool>,
    /// The shadow is over the device's pooled memory, not over host pages: counted apart.
    pooled: bool,
}

impl Drop for ShadowAfterMove {
    fn drop(&mut self) {
        if !self.moved.load(Ordering::Acquire) {
            return;
        }
        // The texture's storage and this hold the image, and nothing else: no view made before
        // the move presents it, and no shadow made over it holds it as its truth, which alone
        // could hold it again. Gone, the texture's storage holds it no more, and the image goes
        // as this drops.
        if self.texture.strong_count() == 0 || Arc::strong_count(&self.image) != 2 {
            return;
        }
        self.copies.merge_scratch.let_go();
        self.image.let_go_unused();
        let counts = &self.image.device.shared().counts;
        match self.pooled {
            true => counts.pooled_texture_shadow_freed_after_move(),
            false => counts.host_texture_shadow_freed_after_move(),
        }
    }
}

/// A texture's storage form, made and its copy recorded into the committing command buffer's
/// last part, or a shadow's image in place, made, to swap in once the commit submitted it. The
/// texture is held weakly while the switch rides: its end takes it again inside a passing hold
/// (`resource_ids.rs`), on whichever thread ends it.
pub(crate) struct Prepared {
    texture: Weak<TextureShared>,
    kind: Kind,
}

/// What a [`Prepared`] swaps in.
enum Kind {
    /// A fetched render target's storage form: its plan, and its image.
    Form {
        plan: ImagePlan,
        swapped: Arc<Mutex<Swapped>>,
    },
    /// A shadow's move in place (the module docs): the aliasing image's storage, until it moves
    /// in, what it counts as it moves in, and whether it moved in, which lets the shadow go
    /// ([`ShadowAfterMove`]).
    Move {
        storage: Mutex<Option<Storage>>,
        counted: MoveCount,
        moved: Arc<AtomicBool>,
    },
}

/// What a shadow's move in place counts as it moves in, as a texture made with its image is
/// counted.
#[derive(Clone, Copy)]
enum MoveCount {
    /// Over host pages: their class, whether the image is pitched, whether a shader writes the
    /// texture (`WorkCounts::host_textures_in_place` and those beside it).
    Host(PagesClass, bool, bool),
    /// Over the device's pooled memory: the aliasing image's counts
    /// (`WorkCounts::pooled_pitched_images` and those beside it).
    Pooled(AliasCount),
}

/// What a waiting texture's storage form takes, read inside a passing hold
/// (`Texture::storage_form_wanted`), its image made with none under way ([`Wanted::make`]).
enum Wanted {
    /// A fetched render target's storage form: its image, the form's plan, and the texture's
    /// type.
    Form {
        image: Arc<Image>,
        plan: ImagePlan,
        texture_type: TextureType,
    },
    /// A shadow's move in place: its texels, what it moves with, and whether the image in place
    /// takes storage usage.
    Move {
        texels: Arc<Texels>,
        pending: in_place::PendingMove,
        storable: bool,
    },
}

/// A waiting texture's storage form's image, made with no passing hold under way
/// ([`Wanted::make`]).
enum Made {
    /// A fetched render target's storage form: its image, the form's plan, and the form's
    /// `VkImage` in what goes with it.
    Form {
        image: Arc<Image>,
        plan: ImagePlan,
        raw: vk::Image,
        swapped: Arc<Mutex<Swapped>>,
    },
    /// A shadow's image in place, and what it counts as it moves in (as [`Kind::Move`]).
    Move {
        storage: Storage,
        counted: MoveCount,
    },
    /// No image in place: the device refused it, and the shadow stops waiting.
    Refused,
}

impl Wanted {
    /// The storage form's image, made on `device`: `None` where the device refuses a fetched
    /// render target's, which then stops waiting.
    fn make(self, device: &Device) -> Option<Made> {
        match self {
            Wanted::Form {
                image,
                plan,
                texture_type,
            } => {
                let class =
                    plan.memory_class(StorageMode::Private, &device.properties().driver_defects);
                let (raw, placement) = device
                    .placed_image(&plan, texture_type, class, device.shared().sharing())
                    .ok()?;
                let swapped = Arc::new(Mutex::new(Swapped {
                    objects: ImageObjects::new(device, raw),
                    memory: Freed {
                        device: device.clone(),
                        placement,
                    },
                }));
                Some(Made::Form {
                    image,
                    plan,
                    raw,
                    swapped,
                })
            }
            Wanted::Move {
                texels,
                pending,
                storable,
            } => {
                let facts = (device.shared().textures).facts(device.compile_device().profile());
                // Over the device's own memory (pooled, or exported), the image there, as the
                // texture's making makes it (`Device::pooled_move_image`).
                if pending.class.over_own_memory() {
                    let made = device.pooled_move_image(&pending, &texels, &facts);
                    let Ok(Some((image, count))) = made else {
                        return Some(Made::Refused);
                    };
                    return Some(Made::Move {
                        storage: Storage::Alias { image, texels },
                        counted: MoveCount::Pooled(count),
                    });
                }
                let made = (&*texels, pending.class, storable);
                let image = device.in_place_image(&pending.plan, &pending.descriptor, made, &facts);
                let Ok(Ok((image, pitched))) = image else {
                    return Some(Made::Refused);
                };
                let writes = InPlaceAccess::Writes.takes(pending.usage);
                Some(Made::Move {
                    storage: Storage::Alias { image, texels },
                    counted: MoveCount::Host(pending.class, pitched, writes),
                })
            }
        }
    }
}

/// A switch under way (the module docs): its flag is up, and the textures it took from the
/// waiting list, held weakly (`Switch::prepare` takes them inside a passing hold).
pub(crate) struct Switch {
    textures: Vec<Weak<TextureShared>>,
}

/// A switch that rides on a command buffer's commit (the module docs), which holds it: its
/// storage forms made and their copy appended to the command buffer, until a thread ends it
/// once the scheduler settled it. Dropped unfinished (the command buffer went before
/// any thread finished it), it gives the switch up: nothing swaps, and the textures wait again.
pub(crate) struct Riding {
    device: Device,
    inner: Mutex<RidingState>,
}

struct RidingState {
    /// The switch and its storage forms, until a thread finishes it.
    switch: Option<(Switch, Vec<Prepared>)>,
    /// Settled: whether every encoder of the command buffer ran, its last batch submitted.
    settled: Option<bool>,
}

impl Riding {
    /// The switch, its storage forms made for `device`, to ride on a command buffer's commit.
    pub(crate) fn new(device: &Device, switch: Switch, prepared: Vec<Prepared>) -> Arc<Riding> {
        let riding = Arc::new(Riding {
            device: device.clone(),
            inner: Mutex::new(RidingState {
                switch: Some((switch, prepared)),
                settled: None,
            }),
        });
        let forms = &device.shared().textures.storage_forms;
        lock(&forms.waiting).riding = Arc::downgrade(&riding);
        riding
    }

    /// The scheduler submitted the command buffer's last batch (or failed it), `ran` whether
    /// every encoder of it ran: a thread may end the switch now.
    pub(crate) fn settle(&self, ran: bool) {
        let mut inner = lock(&self.inner);
        inner.settled.get_or_insert(ran);
    }

    /// The switch, its storage forms and its verdict, once settled; taken once.
    fn settled(&self) -> Option<(Switch, Vec<Prepared>, bool)> {
        let mut inner = lock(&self.inner);
        let ran = inner.settled?;
        let (switch, prepared) = inner.switch.take()?;
        Some((switch, prepared, ran))
    }
}

impl Drop for Riding {
    fn drop(&mut self) {
        let unfinished = lock(&self.inner).switch.take();
        if let Some((switch, prepared)) = unfinished {
            let waiting = lock(&self.device.shared().textures.storage_forms.waiting);
            self.device.end_switch(waiting, (switch, prepared, false));
        }
    }
}

impl Image {
    /// The `VkImage`, handed out: the image keeps it for good (it takes no storage form after).
    /// Whether its `VkImage` was handed out (`Texture::vk_image`): the caller may use it
    /// anywhere from then on, unseen.
    pub(super) fn handed_out(&self) -> bool {
        self.form.exported.load(Ordering::Relaxed)
    }

    fn export(&self) -> vk::Image {
        let _views = lock(&self.views);
        self.form.exported.store(true, Ordering::Relaxed);
        self.raw.get()
    }
}

impl Texture {
    /// Note the texture, a color render target that a pass fetches through the pixel interlock
    /// and renders into a scratch image for want of `STORAGE`, or whose reader after a barrier
    /// splits its pass for want of it (`render/snapshot.rs`), to take its storage form where it
    /// may (the module docs); once.
    pub(crate) fn want_storage_form(&self) {
        if self.0.parent.is_some() {
            return;
        }
        let Storage::Image(image) = self.storage() else {
            return;
        };
        let form = &image.form;
        let own = matches!(
            image.owner,
            Owner::Runtime {
                shareable: false,
                ..
            }
        );
        // Noted already: a reader that splits its pass notes it at each such draw
        // (`render/snapshot.rs`), so the plan is asked no more.
        if form.noted.load(Ordering::Relaxed)
            || !own
            || !image.identity.tracked
            || self.storage_mode() != StorageMode::Private
            || form.taken.get().is_some()
            || form.exported.load(Ordering::Relaxed)
        {
            return;
        }
        let profile = image.device.compile_device().profile();
        if storage_candidate(image.plan(), profile).is_none()
            || form.noted.swap(true, Ordering::AcqRel)
        {
            return;
        }
        let forms = &image.device.shared().textures.storage_forms;
        let mut waiting = lock(&forms.waiting);
        waiting.textures.push(self.downgrade());
        forms.wanted.store(waiting.textures.len(), Ordering::SeqCst);
    }

    /// The texture's `VkImage`, handed out ([`Texture::vk_image`](crate::Texture::vk_image)):
    /// the texture keeps it (the module docs), in `GENERAL`, its first move made first where it
    /// waits (`texture/first_move.rs`). Refused as `image` is.
    pub(crate) fn exported_image(&self) -> Result<vk::Image, Error> {
        let image = self.image()?;
        let lies_in = match self.storage() {
            Storage::Alias { texels, .. } => Some(texels.buffer.at(0).0),
            _ => None,
        };
        image.settle_handed_out(lies_in)?;
        // The caller may write through the handle whenever it likes, and no call of the runtime's
        // sees it: the image's texels are never held on the host again
        // (`texture/host_shadow.rs`).
        image.shadow_off();
        Ok(image.export())
    }

    /// What the texture's storage form takes, read inside a passing hold (`Switch::prepare`),
    /// its image made after: `None` where the texture may no longer take one, a shadow that may
    /// move no more then ceasing to wait (it moved, its image was handed out, or its verdicts or
    /// its bytes keep it a shadow).
    fn storage_form_wanted(&self) -> Option<Wanted> {
        let image = match self.storage() {
            Storage::Image(image) => image,
            Storage::Shadow { .. } => return self.move_wanted(),
            _ => return None,
        };
        if image.form.taken.get().is_some() || image.form.exported.load(Ordering::Relaxed) {
            return None;
        }
        let plan = storage_candidate(image.plan(), image.device.compile_device().profile())?;
        Some(Wanted::Form {
            image: Arc::clone(image),
            plan,
            texture_type: self.texture_type(),
        })
    }

    /// What a shadow's move in place takes (the module docs): the image in place over its
    /// texels is made as the texture's would have been had its pages' verdicts been known as it
    /// was made.
    fn move_wanted(&self) -> Option<Wanted> {
        let Storage::Shadow { image, texels, .. } = &self.0.storage else {
            return None;
        };
        let mut pending = lock(&self.0.pending_move);
        let wanted = pending.as_ref().and_then(|pending| {
            if self.0.moved.get().is_some() || self.shadowed_again() {
                return None;
            }
            let checks = &image.device.shared().in_place;
            let verdict = |class, access, rows| checks.verdict(class, access, rows);
            let policy = image.device.compile_device().policy().host_page_textures;
            let pages = (
                pending.class,
                pending.rows,
                texels.texel_bytes(),
                pending.padding,
            );
            if texels.buffer.detached() || texels.buffer.copied() {
                return None;
            }
            in_place::in_place(policy, pending.usage, pages, verdict).ok()?;
            // As the texture would have taken it as it was made (`Device::storage_over_buffer`).
            let made = (pending.class, pending.rows, pages.2);
            let storable = in_place::storable(pending.usage, made, verdict);
            Some(Wanted::Move {
                texels: Arc::clone(texels),
                pending: pending.clone(),
                storable,
            })
        });
        if wanted.is_none() {
            *pending = None;
        }
        wanted
    }

    /// The texture's storage form, `made` for it with no passing hold under way, and its copy
    /// recorded into `core` (the committing command buffer's last part), inside the hold that
    /// took the texture again (`Switch::prepare`); `None` where the device refused its image, or
    /// the texture may no longer take it, a shadow then ceasing to wait.
    fn prepared(&self, made: Made, core: &mut EncoderCore) -> Option<Prepared> {
        let kind = match made {
            Made::Form {
                image,
                plan,
                raw,
                swapped,
            } => {
                let copied = self.form_copied((&image, &plan, raw), &swapped, core);
                copied.then_some(Kind::Form { plan, swapped })
            }
            Made::Move { storage, counted } => {
                let moves = self.0.moved.get().is_none() && !self.shadowed_again();
                if !moves {
                    *lock(&self.0.pending_move) = None;
                }
                let pooled = matches!(counted, MoveCount::Pooled(_));
                moves.then(|| Kind::Move {
                    storage: Mutex::new(Some(storage)),
                    counted,
                    moved: self.shadow_goes_after_move(core, pooled),
                })
            }
            Made::Refused => {
                *lock(&self.0.pending_move) = None;
                None
            }
        };
        Some(Prepared {
            texture: self.downgrade(),
            kind: kind?,
        })
    }

    /// Record into `core` the copy of the texture's image into its storage form's, `raw` of
    /// `plan` in `swapped`: whether it was recorded, not where the image was handed out or took
    /// its storage form meanwhile.
    fn form_copied(
        &self,
        (image, plan, raw): (&Image, &ImagePlan, vk::Image),
        swapped: &Arc<Mutex<Swapped>>,
        core: &mut EncoderCore,
    ) -> bool {
        if image.form.taken.get().is_some() || image.form.exported.load(Ordering::Relaxed) {
            return false;
        }
        let access = Access::new(
            crate::blit::TRANSFER,
            vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE,
        );
        let Ok(written) = self.image_use(0..plan.levels, 0..plan.layers, access, true) else {
            return false;
        };
        let from = image.raw.get();
        let aspects = texture_aspects(plan);
        let regions = whole_copy(plan, aspects);
        let family = core.queue_flags();
        core.command(
            crate::blit::TRANSFER,
            &[written],
            |device, command_buffer| {
                let range = vk::ImageSubresourceRange {
                    aspect_mask: aspects,
                    base_mip_level: 0,
                    level_count: vk::REMAINING_MIP_LEVELS,
                    base_array_layer: 0,
                    layer_count: vk::REMAINING_ARRAY_LAYERS,
                };
                let layouts = (vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL);
                let made = image_barrier(
                    raw,
                    range,
                    layouts,
                    vk::AccessFlags::empty(),
                    ImageDst::TransferWrite,
                    family,
                );
                // SAFETY: the part's command buffer is recording, after a barrier from everything
                // recorded before it; the new image is live and unused, the old one live and in
                // `GENERAL` (every image rests there outside render passes); both are of one format,
                // extent, levels, layers and samples, and the regions lie in both. The command buffer
                // retains the new image, and the old one until it completes.
                unsafe {
                    device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[made],
                    );
                    device.cmd_copy_image(
                        command_buffer,
                        from,
                        vk::ImageLayout::GENERAL,
                        raw,
                        vk::ImageLayout::GENERAL,
                        &regions,
                    );
                }
            },
        );
        core.retain(Arc::clone(swapped));
        true
    }

    /// Where the texture's shadow moves in place at the commit whose last part `core` is (the
    /// module docs), let its image go once that command buffer completed: a tracked write of
    /// every level and layer of it, which orders the command buffer after every use of the
    /// shadow committed before, on any queue, and the command buffer's hold on what lets it go
    /// then ([`ShadowAfterMove`], `pooled` where the shadow is over the device's pooled memory).
    /// The flag the switch sets where the shadow moved in. Nothing is written or held for an
    /// untracked texture, whose shadow stays until it goes: the program orders its uses on other
    /// queues with its own fences and events.
    fn shadow_goes_after_move(&self, core: &mut EncoderCore, pooled: bool) -> Arc<AtomicBool> {
        let moved = Arc::new(AtomicBool::new(false));
        let Storage::Shadow { image, copies, .. } = &self.0.storage else {
            return moved;
        };
        if !image.identity.tracked {
            return moved;
        }
        let plan = image.plan();
        let access = Access::new(crate::blit::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE);
        let Ok(written) = self.image_use(0..plan.levels, 0..plan.layers, access, true) else {
            return moved;
        };
        core.command(crate::blit::TRANSFER, &[written], |_, _| {});
        core.retain(ShadowAfterMove {
            texture: self.downgrade(),
            image: Arc::clone(image),
            copies: Arc::clone(copies),
            moved: Arc::clone(&moved),
            pooled,
        });
        moved
    }

    /// Whether a binding or a pass took an image of the texture's other than its shadow, made
    /// from it, which a move would leave behind: a usage shadow (`texture/usage_shadow.rs`) or a
    /// render shadow (`texture/ycbcr_image.rs`).
    fn shadowed_again(&self) -> bool {
        self.0.usage_shadow.get().is_some() || lock(&self.0.render_shadow).is_some()
    }

    /// Wait, a shadow over host pages (or the device's pooled memory) made before its class's
    /// in-place verdicts, to move in place once they pass (the module docs): `pending` what it
    /// moves with.
    pub(crate) fn wait_to_move_in_place(&self, pending: in_place::PendingMove) {
        let Storage::Shadow { texels, .. } = &self.0.storage else {
            return;
        };
        let wait = MoveWait {
            texture: self.downgrade(),
            class: pending.class,
            rows: pending.rows,
            padding: pending.padding,
            usage: pending.usage,
            texel: texels.texel_bytes(),
        };
        *lock(&self.0.pending_move) = Some(pending);
        let device = self.device();
        let forms = &device.shared().textures.storage_forms;
        lock(&forms.moves).push(wait);
        // The verdicts may have come since it asked them.
        device.in_place_verdicts_given();
    }
}

/// One copy of every level, all layers at once, of an image of `plan`'s shape, of `aspects`.
fn whole_copy(plan: &ImagePlan, aspects: vk::ImageAspectFlags) -> Vec<vk::ImageCopy> {
    let volume = plan.image_type == metal2vulkan::vk::ImageType::TYPE_3D;
    (0..plan.levels)
        .map(|level| {
            let subresource = vk::ImageSubresourceLayers {
                aspect_mask: aspects,
                mip_level: level,
                base_array_layer: 0,
                layer_count: plan.layers,
            };
            let [width, height, depth] = plan.extent.map(|axis| (axis >> level.min(31)).max(1));
            vk::ImageCopy {
                src_subresource: subresource,
                src_offset: vk::Offset3D::default(),
                dst_subresource: subresource,
                dst_offset: vk::Offset3D::default(),
                extent: vk::Extent3D {
                    width,
                    height,
                    depth: if volume { depth } else { 1 },
                },
            }
        })
        .collect()
}

impl Prepared {
    /// Swap the storage form in (the module docs): the bytes of the image's first memory and of
    /// its storage form's, where it swapped in; `None` where the texture's image was handed out
    /// meanwhile, the made image then going with the command buffer.
    fn swap_in(&self, texture: &Texture) -> Option<(u64, u64)> {
        let Kind::Form { plan, swapped } = &self.kind else {
            return None;
        };
        let Storage::Image(image) = texture.storage() else {
            return None;
        };
        let Owner::Runtime { placement: old, .. } = &image.owner else {
            return None;
        };
        let mut views = lock(&image.views);
        if image.form.exported.load(Ordering::Relaxed) || image.form.taken.get().is_some() {
            return None;
        }
        let mut swapped = lock(swapped);
        let bytes = swapped.memory.placement.size();
        // The image frees its storage form's memory from now on (`Image::drop`); its first
        // memory goes with its first `VkImage`, as the command buffer completes.
        let made = std::mem::replace(&mut swapped.memory.placement, old.handed_over());
        let taken = Taken {
            plan: plan.clone(),
            placement: made,
        };
        if let Err(taken) = image.form.taken.set(taken) {
            // Not reached: every setter holds the views lock, under which it was not taken.
            swapped.memory.placement = taken.placement;
            return None;
        }
        image.views_go();
        let objects = &mut swapped.objects;
        let (first, import) = image.raw.swap(objects.raw, objects.import.take());
        objects.raw = first;
        objects.import = import;
        swapped
            .objects
            .views
            .extend(views.drain().map(|(_, view)| view));
        Some((old.size(), bytes))
    }

    /// Move a shadow in place (the module docs): from now on the texture presents the image in
    /// place over its texels (`TextureShared::moved`). Whether it moved: not where a binding or
    /// a pass took another image of it meanwhile, the image made then going with the command
    /// buffer.
    fn move_in(&self, texture: &Texture) -> bool {
        let Kind::Move { storage, moved, .. } = &self.kind else {
            return false;
        };
        if texture.shadowed_again() {
            return false;
        }
        let Some(storage) = lock(storage).take() else {
            return false;
        };
        *lock(&texture.0.pending_move) = None;
        let swapped = texture.0.moved.set(storage).is_ok();
        moved.store(swapped, Ordering::Release);
        swapped
    }
}

impl Device {
    /// A new command buffer, counted among the uncommitted, waits out a switch under way (the
    /// module docs), finishing it itself once its command buffer's scheduler settled it: one
    /// atomic read where none is. It gives the switch up past [`SWITCH_WAIT`], and at once on the
    /// scheduler's thread, which the switch waits for.
    pub(crate) fn storage_form_wait(&self) {
        let shared = self.shared();
        let forms = &shared.textures.storage_forms;
        if !forms.switching.load(Ordering::SeqCst) {
            return;
        }
        let until = Instant::now() + SWITCH_WAIT;
        loop {
            let mut waiting = lock(&forms.waiting);
            if !forms.switching.load(Ordering::SeqCst) || waiting.given_up {
                return;
            }
            // Read under the lock, which the scheduler's notice of a settled switch takes: a
            // settling after this read wakes the wait below. The command buffer's handle goes
            // with the lock free on every path (`Riding`'s drop takes it).
            let riding = waiting.riding.upgrade();
            if let Some(settled) = riding.as_ref().and_then(|riding| riding.settled()) {
                self.end_switch(waiting, settled);
                drop(riding);
                return;
            }
            let left = until.saturating_duration_since(Instant::now());
            if left.is_zero() || shared.scheduler.is_current() {
                waiting.given_up = true;
                drop(waiting);
                drop(riding);
                return;
            }
            let waited = forms
                .changed
                .wait_timeout(waiting, left)
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .0;
            drop(waited);
            drop(riding);
        }
    }

    /// End the switch under way where its command buffer's scheduler settled it (the module
    /// docs): on the scheduler's thread once the pass that settled it let go of the lock, before
    /// the command buffer's transition to `Scheduled`. One atomic read where none is under way.
    pub(crate) fn finish_storage_forms(&self) {
        let forms = &self.shared().textures.storage_forms;
        if !forms.switching.load(Ordering::SeqCst) {
            return;
        }
        let waiting = lock(&forms.waiting);
        let riding = waiting.riding.upgrade();
        match riding.as_ref().and_then(|riding| riding.settled()) {
            Some(settled) => self.end_switch(waiting, settled),
            None => drop(waiting),
        }
        // The command buffer's handle goes with the lock free (`Riding`'s drop takes it).
        drop(riding);
    }

    /// The device's in-place check settled (`in_place_check.rs`), before a call waiting for its
    /// verdicts wakes, or a shadow began to wait: each shadow made before its class's verdicts
    /// whose class knows them all now stops waiting for them, to wait for a switch that moves it
    /// in place where they pass, or to stay a shadow where they do not (the module docs); those
    /// gone stop waiting. No texture's handle is taken: the caller may be the check's thread. One
    /// lock where no shadow waits.
    pub(crate) fn in_place_verdicts_given(&self) {
        let forms = &self.shared().textures.storage_forms;
        let mut ready = Vec::new();
        {
            let mut moves = lock(&forms.moves);
            if moves.is_empty() {
                return;
            }
            moves.retain(|wait| {
                if wait.texture.strong_count() == 0 {
                    return false;
                }
                match wait.moves(self) {
                    None => true,
                    Some(moves) => {
                        if moves {
                            ready.push(Weak::clone(&wait.texture));
                        }
                        false
                    }
                }
            });
        }
        if !ready.is_empty() {
            let mut waiting = lock(&forms.waiting);
            waiting.textures.extend(ready);
            forms.wanted.store(waiting.textures.len(), Ordering::SeqCst);
        }
    }

    /// Whether a texture waits for its storage form: one atomic read, each commit's gate.
    pub(crate) fn storage_forms_wanted(&self) -> bool {
        let forms = &self.shared().textures.storage_forms;
        forms.wanted.load(Ordering::SeqCst) > 0
    }

    /// A switch, under way, where the device allows one now (the module docs): no other is, the
    /// committing command buffer, which the caller checked, is its only uncommitted one, the
    /// scheduler holds no batch, and no command buffer that may read the heap sets is
    /// incomplete. It takes the waiting textures.
    pub(crate) fn storage_form_switch(&self) -> Option<Switch> {
        let shared = self.shared();
        // Another command buffer open now: no switch, and no lock taken (the count is read again
        // under the lock, once the flag is up).
        if shared.uncommitted.count() != 1 {
            return None;
        }
        let forms = &shared.textures.storage_forms;
        let mut waiting = lock(&forms.waiting);
        if forms.switching.load(Ordering::SeqCst) || waiting.textures.is_empty() {
            return None;
        }
        forms.switching.store(true, Ordering::SeqCst);
        let quiet = shared.uncommitted.count() == 1
            && !self.is_lost()
            && shared.scheduler.holds_none()
            && !heap_sets::tickets_out(self);
        if !quiet {
            forms.switching.store(false, Ordering::SeqCst);
            drop(waiting);
            // A new command buffer that read the flag up waits for the lock or the signal.
            forms.changed.notify_all();
            return None;
        }
        waiting.given_up = false;
        // Those gone stop waiting as the switch prepares.
        let textures = std::mem::take(&mut waiting.textures);
        forms.wanted.store(0, Ordering::SeqCst);
        drop(waiting);
        Some(Switch { textures })
    }

    /// End the switch under way (`waiting` its lock, held), `ran` whether every encoder of its
    /// command buffer ran, which settled it, or `false` where it rides on none (the module
    /// docs): the storage forms swap in where it ran and no new command buffer gave it up, the
    /// heap's elements are written again, the swaps are counted before the flag drops, and, once
    /// the lock is free, the switch is announced;
    /// else the textures it made forms for wait again, and the images made for them go with the
    /// command buffer.
    fn end_switch(
        &self,
        mut waiting: MutexGuard<'_, Waiting>,
        (switch, prepared, ran): (Switch, Vec<Prepared>, bool),
    ) {
        let shared = self.shared();
        let forms = &shared.textures.storage_forms;
        // The textures, taken again on whichever thread ends the switch (the scheduler's, or a
        // new command buffer's): a passing hold (`resource_ids.rs`) until they go.
        let passing = self.passing_hold();
        let textures: Vec<Option<Texture>> = (prepared.iter())
            .map(|form| Texture::upgrade(&form.texture))
            .collect();
        let made = prepared.iter().zip(&textures);
        let made: Vec<(&Prepared, &Texture)> = made
            .filter_map(|(form, texture)| texture.as_ref().map(|texture| (form, texture)))
            .collect();
        let (swapped, moved): (Vec<(&Texture, (u64, u64))>, Vec<(&Prepared, &Texture)>) = if ran
            && !waiting.given_up
        {
            let swapped = (made.iter())
                .filter_map(|&(form, texture)| form.swap_in(texture).map(|bytes| (texture, bytes)))
                .collect();
            let moved = (made.iter().copied())
                .filter(|&(form, texture)| form.move_in(texture))
                .collect();
            (swapped, moved)
        } else {
            let again = made.iter().map(|&(_, texture)| texture.downgrade());
            waiting.textures.extend(again);
            (Vec::new(), Vec::new())
        };
        if !swapped.is_empty() || !moved.is_empty() {
            // The texture slots' elements view the new images; no command buffer that may read
            // them is pending (the module docs).
            lock(&shared.resource_ids).renew_textures();
            heap_sets::renew(self);
        }
        // Each move is counted as a texture made with its image is, before the flag drops.
        for (form, texture) in &moved {
            let Kind::Move { counted, .. } = &form.kind else {
                continue;
            };
            match *counted {
                MoveCount::Host(class, pitched, written) => {
                    // A copied host buffer's shadow does not move (`move_wanted`).
                    let copied = false;
                    (shared.counts).host_texture_in_place(class, (pitched, written, copied));
                    shared.counts.host_texture_moved_in_place();
                    let feedback = metal2vulkan::vk::ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT;
                    let plan = texture.image_plan();
                    if plan.is_ok_and(|plan| plan.usage.contains(feedback)) {
                        shared.counts.feedback_loop_alias_image();
                    }
                }
                MoveCount::Pooled(count) => {
                    count.count(&shared.counts);
                    shared.counts.pooled_texture_moved_in_place();
                }
            }
        }
        // What the announcements name, for once the textures went.
        let notices: Vec<_> = (swapped.iter())
            .map(|&(texture, bytes)| {
                let size = (texture.width(), texture.height());
                (texture.pixel_format(), size, bytes)
            })
            .collect();
        let swaps = swapped.len();
        drop((swapped, moved, made));
        // The lock goes, then the textures' handles and the passing hold, then the
        // announcements, which run the program's callback.
        forms.close(waiting, &shared.counts, swaps);
        drop(textures);
        drop(passing);
        for (format, (width, height), (old_bytes, bytes)) in notices {
            // Once per device: the scheduler's thread, outside any program call, may end it.
            self.degrade_once(Degradation::FetchedTargetStorage, || {
                let object = format!("a {format:?} {width}x{height} render target");
                let reason = format!(
                    "a pass fetched it through the pixel interlock while its image had no \
                     `STORAGE` usage, so that pass rendered into a scratch image; at the commit \
                     of the device's only uncommitted command buffer, its image was made again \
                     with `STORAGE` ({bytes} bytes, where it held {old_bytes}) and its texels \
                     copied in, so its later passes fetch it in place"
                );
                (object, reason)
            });
        }
        drop(prepared);
        drop(switch);
    }

    /// Give up a switch whose storage forms ride on no command buffer (they were not appended):
    /// nothing swaps, and the textures it made forms for wait again.
    pub(crate) fn give_up_switch(&self, switch: Switch, prepared: Vec<Prepared>) {
        let waiting = lock(&self.shared().textures.storage_forms.waiting);
        self.end_switch(waiting, (switch, prepared, false));
    }

    /// Give up a switch before it made any storage form: every texture it took waits again.
    pub(crate) fn put_back_switch(&self, switch: Switch) {
        let mut waiting = lock(&self.shared().textures.storage_forms.waiting);
        let again = switch.textures.iter().map(Weak::clone);
        waiting.textures.extend(again);
        self.end_switch(waiting, (switch, Vec::new(), false));
    }
}

impl StorageForms {
    /// Close the switch under way, `waiting` its lock (held), that swapped `swaps` storage forms
    /// in: count them (`counts`), let go of its command buffer's handle and of a new command
    /// buffer's giving it up, gate the commits on the textures still waiting, drop the flag and
    /// the lock, and wake every new command buffer waiting it out. Counted before the flag drops:
    /// a new command buffer that reads it down reads the count, whichever thread ended the
    /// switch.
    fn close(&self, mut waiting: MutexGuard<'_, Waiting>, counts: &Counters, swaps: usize) {
        for _ in 0..swaps {
            counts.storage_form_taken();
        }
        waiting.riding = Weak::new();
        waiting.given_up = false;
        self.wanted.store(waiting.textures.len(), Ordering::SeqCst);
        self.switching.store(false, Ordering::SeqCst);
        drop(waiting);
        #[cfg(test)]
        if let Some(pause) = &self.pause {
            pause();
        }
        self.changed.notify_all();
    }
}

#[cfg(test)]
impl StorageForms {
    /// A switch under way, its flag up as [`Device::storage_form_switch`] raises it, whose
    /// [`close`](Self::close) runs `pause` right after the flag dropped.
    pub(crate) fn under_way(pause: impl Fn() + Send + Sync + 'static) -> StorageForms {
        let forms = StorageForms {
            pause: Some(Box::new(pause)),
            ..StorageForms::default()
        };
        forms.switching.store(true, Ordering::SeqCst);
        forms
    }

    /// Whether the switch's flag is up: the gate a new command buffer reads first
    /// ([`Device::storage_form_wait`]).
    pub(crate) fn switching(&self) -> bool {
        self.switching.load(Ordering::SeqCst)
    }

    /// End the switch under way's bookkeeping, `swaps` storage forms swapped in.
    pub(crate) fn close_after(&self, counts: &Counters, swaps: usize) {
        self.close(lock(&self.waiting), counts, swaps);
    }
}

impl Switch {
    /// The waiting textures' storage forms, their copies recorded into `core` (the committing
    /// command buffer's last part). A texture the device refuses one stops waiting, as does one
    /// that is gone. The textures are taken on the committing thread, which may be the runtime's
    /// own (the in-place check commits its passes), inside passing holds (`resource_ids.rs`),
    /// none of which makes an image: what each form takes is read in one, the images are made
    /// with none under way, and the copies recorded in another.
    pub(crate) fn prepare(&self, device: &Device, core: &mut EncoderCore) -> Vec<Prepared> {
        let passing = device.passing_hold();
        let textures: Vec<Texture> = (self.textures.iter())
            .filter_map(Texture::upgrade)
            .collect();
        let wanted: Vec<(Weak<TextureShared>, Wanted)> = (textures.iter())
            .filter_map(|texture| Some((texture.downgrade(), texture.storage_form_wanted()?)))
            .collect();
        drop(textures);
        drop(passing);
        let made: Vec<(Weak<TextureShared>, Made)> = (wanted.into_iter())
            .filter_map(|(texture, wanted)| Some((texture, wanted.make(device)?)))
            .collect();
        let passing = device.passing_hold();
        let mut textures = Vec::new();
        let mut unused = Vec::new();
        let mut prepared = Vec::new();
        for (texture, made) in made {
            match Texture::upgrade(&texture) {
                Some(texture) => {
                    prepared.extend(texture.prepared(made, core));
                    textures.push(texture);
                }
                None => unused.push(made),
            }
        }
        drop(textures);
        drop(passing);
        // The images made for textures that went meanwhile go with no passing hold under way.
        drop(unused);
        prepared
    }
}
