//! Managed storage (`MTLStorageModeManaged`): `Buffer::did_modify_range`.
//! `BlitCommandEncoder::synchronize_resource` is in `blit.rs`.
//!
//! A managed buffer follows Metal's contract for the device's memory model
//! ([`DeviceProperties::has_unified_memory`](super::DeviceProperties::has_unified_memory)):
//!
//! - **With unified memory** (an integrated GPU, a CPU implementation), it lives in host-visible
//!   coherent memory, cached where the device has such
//!   ([`MemoryClass::Managed`](super::memory::MemoryClass)), which the GPU uses itself, as Apple
//!   silicon keeps managed resources in memory the CPU and GPU share. Coherent memory needs no
//!   flush and no invalidate: the CPU's writes reach the GPU at the next submission, and every
//!   command buffer ends with a barrier to the host. So these calls check a managed buffer's
//!   range and record nothing more. On a resource of another storage mode they do nothing, as
//!   plain Metal does (its validation stops them).
//! - **Without unified memory** (a discrete GPU), it has two copies, as Metal's managed resources
//!   have on such a device: its own bytes in device-local memory, which the GPU uses, and a host
//!   copy in such cached memory, which `contents` maps (`buffer.rs`'s `Mirror`). `did_modify_range`
//!   notes the range ([`Modified`]); the next commit of a command buffer whose commands may use
//!   the buffer (its encoders' uses name it, `use_resource` made it resident, or a command may
//!   reach memory no use names), of any of the device's queues, takes the range, and the next
//!   submission whose commands may use the buffer of a command buffer committed at or after that
//!   commit copies it from the host copy to the device's own, before any of them, after the
//!   command buffers before it on its `VkQueue` (`texture/first_move.rs`'s `register_upload`):
//!   Metal's shape, the upload made before the GPU uses the buffer, on the user's own queue. A
//!   command buffer committed before, whose submission the scheduler makes after the later
//!   commit, never takes it, so none reads a range only a later commit took. A
//!   commit that uses none of it leaves the range noted, and its submission leaves the copy
//!   waiting, so the copy never lands under a command buffer of another queue still to read the
//!   bytes it replaces (`synchronize_resource` copies the buffer back where the command buffer
//!   runs it). A program
//!   that skips them sees stale bytes there, as on a Mac with a discrete GPU. A GPU writing
//!   host-cached memory across the bus ran a kernel's writes at 0.03 to 0.1 GB/s on an RTX 3090
//!   and an RTX 5080, where a device-local write and a copy back ran at several GB/s. A `Managed`
//!   no-copy buffer is two copies the same way, its host copy the caller's pages
//!   (`managed_copy.rs`).
//! - **The bytes a command buffer reads** follow the device's policy
//!   (`DevicePolicy::managed_copy_in`). Under the default, `ManagedCopyInPolicy::AtSubmission`,
//!   the GPU copies the range from the host copy as that submission runs, with no copy on the CPU
//!   (but of pages no import of their own maps and no guest RAM's chunk imports hold, which the
//!   CPU copies at the commit into staging memory, `managed_copy.rs`): a command buffer reads the bytes the host copy held at its commit, and a
//!   CPU write after the commit, before the GPU ran the command buffer, may reach it. Macs differ
//!   there, so the cheaper answer is the default: on an AMD Radeon Pro 5300M (macOS 26.7.1, two
//!   copies), a kernel copying a 4 KiB `Managed` buffer behind a spin kernel read the CPU's write
//!   made before the commit (after the call, and with the command buffer encoded before the
//!   write), and not one made 30 ms after the commit while the GPU still spun; on an Intel UHD
//!   Graphics 630 (unified memory, one copy) it read the later write, as the GPU reads the memory
//!   the CPU wrote last. Under `ManagedCopyInPolicy::AtCommit`, which
//!   `DevicePolicy::exact_answers()` selects, the commit takes the range's bytes as the host copy
//!   holds them then, before the call returns: a CPU copy of each range noted since the last
//!   commit into staging buffers of host-cached memory, freed once the submission that copies
//!   them in completed (`managed_copy.rs`, at most `STAGED_BYTES` staged ahead), so a write after
//!   the commit reaches no command buffer committed before it, the 5300M's answer. A range the
//!   staging buffers find no memory for is copied in from the host copy as that submission runs,
//!   as under the default.

use super::buffer::{Buffer, WeakBuffer};
use super::device::lock;
use super::error::{invalid, Error, Problem};
use super::resource::ResourceKey;
use super::values::StorageMode;
use super::ObjectId;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::Mutex;

impl Buffer {
    /// `didModifyRange:`: the CPU wrote `range` of a managed buffer; command buffers committed
    /// after this call see it. Where the buffer has a host copy (a device without unified memory),
    /// the range is taken at the next commit of a command buffer that may use the buffer and
    /// copied from the host copy to its own bytes by that command buffer's submission, or one
    /// before it that may use the buffer, before any of its commands, read as the device's policy
    /// says (`DevicePolicy::managed_copy_in`, the module docs); the call does not wait for the
    /// GPU. On a buffer of another storage mode it does nothing, whatever the range,
    /// as plain Metal does (RT2 `managed.didmodify.shared`: `Shared` and `Private` buffers, in
    /// range and past it; Metal's validation stops it). Refused for a managed buffer's range past
    /// the buffer.
    pub fn did_modify_range(&self, range: Range<u64>) -> Result<(), Error> {
        // Its parts held while the call reaches them, across a recovery (`Buffer::held_parts`).
        let _held = self.held_parts();
        const CALL: &str = "did_modify_range";
        if self.storage_mode() != StorageMode::Managed {
            return Ok(());
        }
        let length = self.length();
        if range.start > range.end || range.end > length {
            return Err(invalid(
                CALL,
                Problem::RangeOutOfBounds {
                    offset: range.start,
                    size: range.end.saturating_sub(range.start),
                    length,
                },
            ));
        }
        // A two-copy buffer over the caller's pages (`managed_copy.rs`): the range the program
        // asked, which the next commit that may use the buffer copies in.
        if self.caller_copy().is_some() && !range.is_empty() {
            let counts = &self.generation().shared().counts;
            counts.managed_asked(range.end - range.start, true);
        }
        self.modified(range);
        Ok(())
    }

    /// The CPU wrote `range` of the buffer's host copy, where it has one: the next commit of a
    /// command buffer that may use the buffer takes the bytes and copies them in ([`Modified`]).
    pub(super) fn modified(&self, range: Range<u64>) {
        if self.mirror().is_none() || range.is_empty() {
            return;
        }
        self.generation().shared().modified.note(self, range);
    }
}

/// The ranges of buffers with a host copy the CPU wrote and named (`did_modify_range`, a CPU
/// write through a texture over the buffer) that no commit of a command buffer that may use
/// their buffer took yet: the next such commit copies each in, as the host copy holds it then
/// (the module docs), and the commits' stamps. A buffer gone meanwhile takes its ranges with it.
/// Each buffer's ranges are kept by its key, and indexed by what owns its memory
/// (`used_resources::owner_of`), so a commit reads what its command buffer's uses name once and
/// takes the ranges of the owners among them: its cost follows what it uses, and the ranges of
/// buffers no commit uses cost it nothing ([`Noted`]).
#[derive(Default)]
pub(super) struct Modified(Mutex<Ranges>);

#[derive(Default)]
struct Ranges {
    noted: Noted<ResourceKey, WeakBuffer>,
    /// The commits so far: the stamp of the last ([`Modified::take_at_commit`]).
    commits: u64,
}

/// Which of the owners of the buffers whose ranges are noted a committing command buffer may use
/// ([`Modified::take_at_commit`]).
pub(crate) enum MayUse {
    /// All of them: a command may reach memory no use names, or its uses are not read yet.
    All,
    /// Those its encoders' uses name or made resident, by identity.
    Of(BTreeSet<ObjectId>),
}

impl Modified {
    /// Note `range` of `buffer`: once, where a range noted of it already holds it.
    fn note(&self, buffer: &Buffer, range: Range<u64>) {
        let key = buffer.identity().key;
        let owner = crate::used_resources::owner_of(buffer.generation().shared(), key);
        let of = || buffer.downgrade();
        lock(&self.0)
            .noted
            .note((key, owner), of, range, |of| !of.lives());
    }

    /// A commit of a command buffer whose commands may use the owners of the noted buffers as
    /// `may_use` answers (asked with whether an owner's buffers have ranges noted, and how many
    /// owners do; not asked where no owner's are): take the ranges noted of those it may use,
    /// and of buffers no owner holds (the runtime's own), and copy their bytes in from their host
    /// copies, as they hold them as the submission runs or now, before the command buffer is
    /// submitted, as the device's policy says (`Buffer::upload_at_commit`). The commit's stamp,
    /// which the command buffer keeps: one past every commit's taken before it, under the same
    /// lock. Each upload carries it, and only a batch of a command buffer whose stamp is at least
    /// as large takes it (`texture/first_move.rs`'s `register_upload`): its own submission that
    /// may use the buffer copies them in, or a later commit's, never one committed before, which
    /// reads no range only a later commit took. The ranges of the rest stay noted, in their order.
    pub(crate) fn take_at_commit(
        &self,
        may_use: impl FnOnce(&dyn Fn(ObjectId) -> bool, usize) -> MayUse,
    ) -> u64 {
        let (commit, taken) = {
            let mut ranges = lock(&self.0);
            ranges.commits += 1;
            // Its uses read only where an owner's buffers have ranges noted.
            let used = match ranges.noted.owners() {
                0 => MayUse::Of(BTreeSet::new()),
                owners => {
                    let noted = &ranges.noted;
                    may_use(&|owner| noted.owns(owner), owners)
                }
            };
            let taken = ranges.noted.take(&used);
            (ranges.commits, taken)
        };
        // Each buffer upgraded with the lock free, and let go of with it free.
        for (buffer, range) in taken {
            if let Some(buffer) = buffer.upgrade() {
                buffer.upload_at_commit(range, commit);
            }
        }
        commit
    }
}

/// Ranges of buffers named by `B`, each buffer's kept by its key `K` with what owns its memory
/// ([`Modified`]): taken in the order each buffer's first was noted, a buffer's own in theirs.
/// Indexed by owner, so a take looks at the buffers of the owners it is given and of none, never
/// at the rest; a buffer gone is dropped as a note finds the ranges past twice their count at the
/// last such sweep, so the ranges of buffers that went unused do not pile up.
pub(super) struct Noted<K, B> {
    /// Each buffer's ranges, by its key.
    by_key: BTreeMap<K, NotedOf<B>>,
    /// The keys of the buffers each owner holds, by the owner's identity.
    by_owner: BTreeMap<ObjectId, BTreeSet<K>>,
    /// The keys of the buffers no owner holds, which every take takes.
    unowned: BTreeSet<K>,
    /// The next buffer's place in the order.
    next: u64,
    /// The buffers past which the next note drops those gone.
    sweep_at: usize,
}

/// One buffer's noted ranges ([`Noted`]).
struct NotedOf<B> {
    of: B,
    owner: Option<ObjectId>,
    /// Its place in the order the buffers were first noted.
    at: u64,
    ranges: Vec<Range<u64>>,
}

/// The fewest buffers at which a note drops those gone ([`Noted`]).
const SWEEP_FLOOR: usize = 64;

impl<K, B> Default for Noted<K, B> {
    fn default() -> Self {
        Noted {
            by_key: BTreeMap::new(),
            by_owner: BTreeMap::new(),
            unowned: BTreeSet::new(),
            next: 0,
            sweep_at: SWEEP_FLOOR,
        }
    }
}

impl<K: Ord + Copy, B> Noted<K, B> {
    /// Note `range` of the buffer of `key`, which `owner` holds (`None`: no owner), made by `of`
    /// where none of its ranges is noted: once, where a range noted of it already holds it. A
    /// buffer `gone` answers for is dropped first where its key is noted again, and every one
    /// once the buffers noted pass their sweep's bound.
    pub(super) fn note(
        &mut self,
        (key, owner): (K, Option<ObjectId>),
        of: impl FnOnce() -> B,
        range: Range<u64>,
        gone: impl Fn(&B) -> bool,
    ) {
        if self.by_key.get(&key).is_some_and(|noted| gone(&noted.of)) {
            self.remove(key);
        }
        if self.by_key.len() >= self.sweep_at {
            let went: Vec<K> = (self.by_key.iter())
                .filter(|(_, noted)| gone(&noted.of))
                .map(|(key, _)| *key)
                .collect();
            for key in went {
                self.remove(key);
            }
            self.sweep_at = (2 * self.by_key.len()).max(SWEEP_FLOOR);
        }
        let next = &mut self.next;
        let noted = self.by_key.entry(key).or_insert_with(|| {
            *next += 1;
            NotedOf {
                of: of(),
                owner,
                at: *next,
                ranges: Vec::new(),
            }
        });
        if noted.ranges.is_empty() {
            match noted.owner {
                Some(owner) => {
                    self.by_owner.entry(owner).or_default().insert(key);
                }
                None => {
                    self.unowned.insert(key);
                }
            }
        }
        let held = (noted.ranges.iter()).any(|at| at.start <= range.start && range.end <= at.end);
        if !held {
            noted.ranges.push(range);
        }
    }

    /// Whether the buffers `owner` holds have ranges noted.
    pub(super) fn owns(&self, owner: ObjectId) -> bool {
        self.by_owner.contains_key(&owner)
    }

    /// How many owners hold buffers whose ranges are noted.
    pub(super) fn owners(&self) -> usize {
        self.by_owner.len()
    }

    /// The ranges of the buffers `used` names and of those no owner holds, by buffer in the order
    /// each was first noted, a buffer's in theirs; the rest stay noted.
    pub(super) fn take(&mut self, used: &MayUse) -> Vec<(B, Range<u64>)>
    where
        B: Clone,
    {
        let keys: Vec<K> = match used {
            MayUse::All => self.by_key.keys().copied().collect(),
            MayUse::Of(owners) => (owners.iter())
                .filter_map(|owner| self.by_owner.get(owner))
                .flatten()
                .chain(&self.unowned)
                .copied()
                .collect(),
        };
        let mut taken: Vec<NotedOf<B>> = keys
            .into_iter()
            .filter_map(|key| self.remove(key))
            .collect();
        taken.sort_by_key(|noted| noted.at);
        (taken.into_iter())
            .flat_map(|noted| {
                let of = noted.of;
                (noted.ranges.into_iter()).map(move |range| (of.clone(), range))
            })
            .collect()
    }

    /// Drop the buffer of `key`'s ranges, and give them back.
    fn remove(&mut self, key: K) -> Option<NotedOf<B>> {
        let noted = self.by_key.remove(&key)?;
        match noted.owner {
            Some(owner) => {
                if let Some(keys) = self.by_owner.get_mut(&owner) {
                    keys.remove(&key);
                    if keys.is_empty() {
                        self.by_owner.remove(&owner);
                    }
                }
            }
            None => {
                self.unowned.remove(&key);
            }
        }
        Some(noted)
    }
}
