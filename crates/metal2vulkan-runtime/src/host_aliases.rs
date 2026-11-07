//! No-copy buffers over the same host memory: hazard tracking keys each buffer's uses on the buffer
//! (`tracker.rs`), so two no-copy buffers over one range of the caller's memory were two resources
//! to it, and nothing ordered a write through one before a read through the other: the consumer
//! stands each texture of an IOSurface in with a texture of its plane over a new no-copy buffer of
//! the surface's pages (`Buffer::new_texture_with_iosurface_plane`), so two textures of one surface
//! lay over two buffers. Metal orders two textures over one IOSurface as it orders one resource's
//! accesses: in one encoder, in one command buffer and across one queue's command buffers (an M4,
//! macOS 27.0.1, plain and under API validation; across queues only as
//! the program orders them).
//!
//! - **One memory, one key.** A no-copy buffer made over exactly the bytes a live one covers
//!   (the same first byte and length; not a heap's) joins it: the device notes the first such
//!   buffer's key as the memory's ([`Alias`]), and each command's uses of a buffer that joined
//!   carry a use of the memory's key too, of the same bytes and access ([`HostAliases::augment`],
//!   `EncoderCore`). The tracker orders those as one resource's, inside an encoder, between
//!   encoders and command buffers, and across queues as it orders any. The first buffer's own
//!   uses name the memory's key already, and its work from before the second buffer was made is
//!   ordered against the second's.
//! - **One memory, one import.** A buffer that joins a memory, of the memory class of a live
//!   import of it, lies in that import's `VkBuffer` (`HostImport`, `Device::buffer_in_import`):
//!   one resource to the driver, where the tracker's barriers are not all the ordering there is.
//!   A copied host buffer (`copied.rs`) joins the memory as an import does, and a later buffer of
//!   either class lies in its copy while no import of its class lives ([`Kind::Copy`]): the copy
//!   is the runtime's own memory of one class whatever the storage mode, and two copies of one
//!   memory agree only through the pages, at a completion's merge back and a commit's copy in, so
//!   a read committed before the write's completion read the bytes from before it (the
//!   consumer's two textures over one surface past its import gate, every texel stale in one
//!   command buffer and across queues an event ordered).
//!   MoltenVK records no barrier between blits and leaves their order to Metal's hazard
//!   tracking, which is by `MTLBuffer`: two imports were two, and a blit's write through one
//!   raced a read through the other in one encoder and in one command buffer (an M4: mixed texels);
//!   one import is one `MTLBuffer`, which Metal orders. Each buffer stays its own Metal object,
//!   with its own Metal addresses, object ID, key, pages hold and release: a command buffer lists
//!   and holds, and marks reached unnamed (`live_buffers.rs`), the buffers its commands use, not
//!   their memory, as Metal marks them; its deallocator runs once nothing holds its pages, as an
//!   import of its own would let it, while the import lives on in the buffers that still lie in it,
//!   and its keep-alive goes then too, since those buffers keep the same bytes mapped
//!   (`buffer.rs`'s `HostPages`). Copies between two such buffers are copies within one `VkBuffer`,
//!   moved as within one buffer (`blit.rs`, `blit_texture.rs`). A buffer rebound onto the bytes a
//!   live buffer was made over, of the class of its import, lies in that import too, joining no key
//!   (`Device::rebind_host_pages`; a heap's own buffer, and bytes with a texture's aliasing image
//!   over them, import their own), and in a live copy of them where its textures were planned at
//!   the offset the copy lays the bytes at (its first byte) and no image in place rides on it.
//! - **Ownership stays the buffer's.** The added uses are hazard tracking's alone: what a command
//!   holds and lists (`EncoderCore::touch`, rebind holds, `used_resources`) is named by the
//!   command's own uses, each buffer by its own key.
//! - **Shadows.** A shadowed texture over a buffer that joined compares its truth both ways
//!   (`ShadowCopies::meets`), so a write through the other buffer, or a texture over it, makes
//!   the shadow copy in again, as a write through its own buffer does.
//! - **Not joined.** Buffers over memory that overlaps without being the same range, a heap's
//!   bytes (a heap's own copy too), and the memory a rebound buffer takes, keep their own keys,
//!   and a buffer whose GPU writes are dropped imports a read-only view of its own, or is a copy
//!   of its own (a copy shared with it would take its writes): Metal orders no two
//!   no-copy buffers over one allocation (the same probe's `host` cases read stale or mixed
//!   texels), so the runtime's joining of the same range is the IOSurface's ordering, which a
//!   race over plain no-copy buffers may also give. A rebound buffer leaves the memory it was
//!   made over, keeping the key it joined (`Device::rebind_host_pages`); its pages stay in the
//!   import while commands recorded before the rebind hold them. Two buffers made at once over
//!   the same memory may each import it (each joins after its import): they share the key, not
//!   the `VkBuffer`.
//! - **Ordered by the bytes they lie over now** (the consumer's boot whose keys 113 and 115, over
//!   the same 1080p surface's pages, were never ordered: one of them was rebound there, or made
//!   after a remap of them, so neither joined the other). Joining keeps to the memory a buffer was
//!   made over, for its import; hazard tracking instead orders the uses of every buffer over host
//!   pages, made or rebound there, by the bytes it lies over now ([`Placements`]): the keys over
//!   one range of bytes share a key (the first placed there while any is, as a memory's), and each
//!   use of a key's bytes carries a use of its range's key and of the key of each other range those
//!   bytes overlap, clipped to them ([`HostAliases::augment`]). So two keys over overlapping bytes
//!   are ordered where they overlap, whatever made them, at a cost of one lookup a use while no two
//!   ranges meet, and of one use added per range met otherwise. An encoder keeps the keys it
//!   found to take no other use ([`Quiet`]): while no buffer joins a memory, is placed, moves or
//!   goes, its commands whose uses of bytes name only those look nothing up and take no lock.
//!   Each range keeps the ranges that meet it, so a use reads them there rather than looking them
//!   up by address, but where they outnumber the classes of length that lookup visits
//!   ([`Placements::sharing`]). A remap reported keeps a placement: it orders more than the pages
//!   need, never less. A heap's own buffer is placed as a buffer is.
//! - **Same address, same pages.** Sharing an import assumes the pages at the bytes' addresses
//!   are the ones the import holds: Metal's `newBufferWithBytesNoCopy:` wires the pages at the
//!   address when it makes each buffer, so a buffer made after the caller replaced the pages
//!   under an address (an unmap and a map there) is over the new pages. The caller tells the
//!   device so (`Device::host_range_remapped`): every live buffer made over memory with a byte
//!   in the range then leaves it, as a rebound buffer does ([`HostAliases::remapped`]), so a
//!   later buffer imports the pages there anew and joins no earlier key (the consumer's
//!   `[blocks]` setup keeps one host address for a guest address across the guest's unmap and
//!   remap). Without the call, a buffer over the same bytes as a live one lies in its import, and
//!   so over the pages that were there when that one was made.
//! - **Cheap, from anywhere.** The call sits on the caller's unmap path, at each release of a
//!   buffer's pages and over whole reservations of several GiB: with no memory noted and no
//!   bytes placed it returns without the lock, and otherwise it finds the memories with a byte in
//!   the range by address ([`Memories`]), never reading them all, and ends them without reading
//!   their buffers' notes ([`Over`]); the placed ranges it touches it counts by the same lookup
//!   (`WorkCounts::host_imports_remapped`). The lock is a leaf: nothing else is locked, and no buffer, pages hold or
//!   deallocator is dropped, while it is held (the imports noted are weak), so the call may be
//!   made from any thread, a deallocator's among them.

use super::buffer::HostImport;
use super::device::lock;
use super::key_hash::KeyMap;
use super::memory::MemoryClass;
use super::resource::ResourceKey;
use super::tracker::{Use, UseRange};
use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};

/// The key a buffer that joined another's memory adds to its uses: the memory's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Alias {
    pub(super) key: ResourceKey,
}

/// What a memory's import is to the buffers made over the memory later ([`HostAliases::import`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    /// An import of the caller's pages, in memory of this class: a later buffer of the class lies
    /// in it.
    Import(MemoryClass),
    /// A copy of the caller's pages in memory of the runtime's own (`copied.rs`): a later buffer
    /// of either class lies in it while no import of its class lives, as the copy's memory is of
    /// one class whatever the buffer's storage mode, the memory a copy of its own would take.
    Copy,
}

/// The device's no-copy buffers by the host memory they were made over (the module docs), and
/// the imports of it they lie in (`I`: a `HostImport`, but in unit tests).
pub(super) struct HostAliases<I = HostImport> {
    /// Whether any buffer joined another's memory: else no use takes another.
    any: AtomicBool,
    /// Whether a range of placed bytes has a second key or meets another range
    /// ([`Placements::crowded`]): else no use takes another for its bytes.
    crowded: AtomicBool,
    /// How many memories are noted: with none, a reported remap has none to end.
    noted: AtomicUsize,
    /// How many ranges of bytes keys lie over ([`Placements`]): with none and no memory noted, a
    /// reported remap meets nothing.
    placed: AtomicUsize,
    /// How many times the buffers that joined a memory or the placements changed: while it
    /// stands, [`augment`](Self::augment) gives the same uses for the same uses
    /// (`noted_apart.rs`, [`Quiet`]). Each change counts once it is made, under the lock.
    version: AtomicU64,
    /// How many times uses were looked up under the lock ([`augment_into`](Self::augment_into),
    /// `WorkCounts::host_alias_lookups`).
    lookups: AtomicU64,
    inner: Mutex<Inner<I>>,
}

/// The keys found to take no other use while the host aliases stood at `version`
/// ([`HostAliases::augment_into`]): no buffer joined another's memory under them, and the range
/// of bytes each lies over, if any, holds no other key and meets no other range
/// ([`Placed::crowded`]). An encoder keeps them, so a command whose uses of bytes name only such
/// keys looks nothing up and takes no lock while no buffer joins a memory, is placed, moves or
/// goes; a change of the version forgets them.
#[derive(Default)]
pub(super) struct Quiet {
    version: u64,
    keys: KeyMap<ResourceKey, ()>,
}

impl Quiet {
    /// Forget the keys found while the host aliases stood at a version other than `version`.
    fn stand(&mut self, version: u64) {
        if self.version != version {
            self.keys.clear();
            self.version = version;
        }
    }

    /// Whether `used` takes no other use as it is: it is of no bytes, or of a key found quiet.
    fn holds(&self, used: &Use<ResourceKey>) -> bool {
        !matches!(used.range, UseRange::Bytes(_)) || self.keys.contains_key(&used.key)
    }
}

impl<I> Default for HostAliases<I> {
    fn default() -> Self {
        HostAliases {
            any: AtomicBool::new(false),
            crowded: AtomicBool::new(false),
            noted: AtomicUsize::new(0),
            placed: AtomicUsize::new(0),
            version: AtomicU64::new(0),
            lookups: AtomicU64::new(0),
            inner: Mutex::new(Inner {
                made_over: KeyMap::default(),
                memories: Memories::default(),
                aliases: Arc::default(),
                serial: 0,
                placements: Placements::default(),
            }),
        }
    }
}

struct Inner<I> {
    /// The memory each live buffer was made over, by its key, until it leaves it.
    made_over: KeyMap<ResourceKey, Over>,
    /// Each memory with a live buffer made over it, by its first byte's host address and its
    /// length.
    memories: Memories<Memory<I>>,
    /// The buffers that joined a memory, by their key: a snapshot each command reads.
    aliases: Arc<KeyMap<ResourceKey, Alias>>,
    /// The serial of the memory noted last.
    serial: u64,
    /// The bytes each buffer over host pages lies over now.
    placements: Placements,
}

/// The memory a live buffer was made over: its first byte's host address and its length, and
/// the serial it was noted with. A remap ends a memory without reading its buffers' notes; a
/// memory noted again over the same bytes takes a new serial, which those notes do not name.
#[derive(Clone, Copy)]
struct Over {
    memory: (u64, u64),
    serial: u64,
}

/// A memory live buffers were made over.
struct Memory<I> {
    /// The first such buffer's key: the memory's.
    key: ResourceKey,
    /// How many live buffers were made over it.
    buffers: usize,
    /// Its imports a buffer made, by kind: a later buffer lies in one while it lives
    /// ([`HostAliases::import`]).
    imports: Vec<(Kind, Weak<I>)>,
    /// The serial it was noted with ([`Over`]).
    serial: u64,
}

impl<I> HostAliases<I> {
    /// The live import of `class` of `memory`, the bytes a live buffer was made over (its first
    /// byte's host address and its length), else its live copy ([`Kind::Copy`]): a buffer of
    /// `class` made over them lies in it.
    pub(super) fn import(&self, memory: (u64, u64), class: MemoryClass) -> Option<Arc<I>> {
        let inner = lock(&self.inner);
        let memory = inner.memories.get(memory)?;
        let live = |kind: Kind| {
            (memory.imports.iter())
                .filter(|(made, _)| *made == kind)
                .find_map(|(_, import)| import.upgrade())
        };
        live(Kind::Import(class)).or_else(|| live(Kind::Copy))
    }

    /// Whether a live import of `class` was made over memory that holds all of `memory` and more
    /// (`WorkCounts::object_churn.imports_inside_live`): a buffer over part of a live no-copy
    /// buffer's bytes, which imports its own.
    pub(super) fn inside_live(&self, memory: (u64, u64), class: MemoryClass) -> bool {
        let inner = lock(&self.inner);
        let end = memory.0.saturating_add(memory.1);
        let found = inner
            .memories
            .overlapping(memory.0, end)
            .filter(|&(start, length)| {
                (start, length) != memory
                    && start <= memory.0
                    && end <= start.saturating_add(length)
            })
            .any(|at| {
                inner.memories.get(at).is_some_and(|noted| {
                    (noted.imports.iter()).any(|(made, import)| {
                        *made == Kind::Import(class) && import.strong_count() > 0
                    })
                })
            });
        found
    }

    /// Note the buffer of `key`, made over `memory` (its first byte's host address and its
    /// length) and lying in `import`, of `kind`: it joins the memory a live buffer was made over,
    /// or its key becomes the memory's; later buffers over the memory lie in `import`, where
    /// [`import`](Self::import) finds it, unless a live import of its kind is noted already.
    pub(super) fn join(
        &self,
        key: ResourceKey,
        memory: (u64, u64),
        kind: Kind,
        import: &Arc<I>,
    ) -> Option<Alias> {
        let mut inner = lock(&self.inner);
        let Inner {
            made_over,
            memories,
            aliases,
            serial,
            placements: _,
        } = &mut *inner;
        let noted = memories.get_or_insert_with(memory, || {
            *serial += 1;
            Memory {
                key,
                buffers: 0,
                imports: Vec::new(),
                serial: *serial,
            }
        });
        noted.buffers += 1;
        noted
            .imports
            .retain(|(_, import)| import.strong_count() > 0);
        if !noted.imports.iter().any(|(made, _)| *made == kind) {
            noted.imports.push((kind, Arc::downgrade(import)));
        }
        let owner = noted.key;
        let over = Over {
            memory,
            serial: noted.serial,
        };
        made_over.insert(key, over);
        self.noted.store(memories.len(), Ordering::Relaxed);
        if owner == key {
            return None;
        }
        let alias = Alias { key: owner };
        Arc::make_mut(aliases).insert(key, alias);
        self.any.store(true, Ordering::Relaxed);
        self.changed();
        Some(alias)
    }

    /// The memory the buffer of `key` joined, if it joined one.
    pub(super) fn of(&self, key: ResourceKey) -> Option<Alias> {
        if !self.any.load(Ordering::Relaxed) {
            return None;
        }
        lock(&self.inner).aliases.get(&key).copied()
    }

    /// The buffer of `key` no longer lies in the memory it was made over (a rebind): later
    /// buffers over that memory do not join it. It keeps the key it joined.
    pub(super) fn leave(&self, key: ResourceKey) {
        let mut inner = lock(&self.inner);
        if let Some(over) = inner.made_over.remove(&key) {
            release(&mut inner.memories, over);
            self.noted.store(inner.memories.len(), Ordering::Relaxed);
        }
    }

    /// The caller replaced the pages under the host addresses `start..end`
    /// (`Device::host_range_remapped`): every live buffer made over memory with a byte among
    /// them leaves it, as a rebound buffer does ([`HostAliases::leave`]), keeping its pages, its
    /// import and the key it joined, so that a later buffer over any of those addresses neither
    /// lies in an import made before nor joins their key. With no memory noted and no bytes
    /// placed it returns at once; else it ends the memories the lookup by address finds
    /// ([`Memories::overlapping`]).
    ///
    /// Returns how many ranges of bytes buffers over host pages lie over now ([`Placements`]:
    /// each an import, two only where buffers of two memory classes lie over the same bytes)
    /// have a page among those `start..end` covers, rounded out to `unit` bytes, the import's
    /// unit: the imports whose pages the replacement touches, which a driver's import by pointer
    /// takes anew at its next submission (`WorkCounts::host_imports_remapped`).
    pub(super) fn remapped(&self, start: u64, end: u64, unit: u64) -> usize {
        if start >= end
            || (self.noted.load(Ordering::Relaxed) == 0 && self.placed.load(Ordering::Relaxed) == 0)
        {
            return 0;
        }
        let mut inner = lock(&self.inner);
        let ended: Vec<(u64, u64)> = inner.memories.overlapping(start, end).collect();
        for memory in ended {
            inner.memories.remove(memory);
        }
        self.noted.store(inner.memories.len(), Ordering::Relaxed);
        // An import's pages are its bytes rounded out to the unit: one has a page the remap
        // covers exactly when its bytes meet the remap's range rounded out to the unit.
        let unit = unit.max(1);
        let from = start - start % unit;
        let to = end.div_ceil(unit).saturating_mul(unit);
        inner.placements.ranges.overlapping(from, to).count()
    }

    /// Whether the bytes `start..end`, rounded out to `unit` bytes, the import's unit, lie within
    /// the bytes some buffer over host pages lies over now ([`Placements`]: each an import's),
    /// rounded out likewise: an import alive now holds each of their pages
    /// (`WorkCounts::host_rebind_imports_covered`).
    pub(super) fn covered(&self, start: u64, end: u64, unit: u64) -> bool {
        if start >= end || self.placed.load(Ordering::Relaxed) == 0 {
            return false;
        }
        let unit = unit.max(1);
        let from = start - start % unit;
        let to = end.div_ceil(unit).saturating_mul(unit);
        let inner = lock(&self.inner);
        let mut ranges = inner.placements.ranges.overlapping(from, to);
        ranges.any(|(at, length)| {
            let end = at
                .saturating_add(length)
                .div_ceil(unit)
                .saturating_mul(unit);
            at - at % unit <= from && to <= end
        })
    }

    /// The buffer of `key` is gone.
    pub(super) fn forget(&self, key: ResourceKey) {
        let mut inner = lock(&self.inner);
        if let Some(over) = inner.made_over.remove(&key) {
            release(&mut inner.memories, over);
            self.noted.store(inner.memories.len(), Ordering::Relaxed);
        }
        if inner.aliases.contains_key(&key) {
            Arc::make_mut(&mut inner.aliases).remove(&key);
            self.any.store(!inner.aliases.is_empty(), Ordering::Relaxed);
        }
        inner.placements.unplace(key);
        self.crowded
            .store(inner.placements.crowded(), Ordering::Relaxed);
        (self.placed).store(inner.placements.ranges.len(), Ordering::Relaxed);
        self.changed();
    }

    /// The buffer of `key` (a no-copy buffer, or a no-copy heap's own buffer, whose uses are of
    /// bytes counted from its first) lies over the host bytes `at` (its first byte's address and
    /// its length) from now on, made or rebound there, and no longer where it lay before
    /// ([`Placements`]).
    pub(super) fn place(&self, key: ResourceKey, at: (u64, u64)) {
        let mut inner = lock(&self.inner);
        inner.placements.unplace(key);
        inner.placements.place(key, at);
        self.crowded
            .store(inner.placements.crowded(), Ordering::Relaxed);
        (self.placed).store(inner.placements.ranges.len(), Ordering::Relaxed);
        self.changed();
    }

    /// The buffer of `key` lies over no host bytes from now on (its bytes detached).
    pub(super) fn unplace(&self, key: ResourceKey) {
        let mut inner = lock(&self.inner);
        inner.placements.unplace(key);
        self.crowded
            .store(inner.placements.crowded(), Ordering::Relaxed);
        (self.placed).store(inner.placements.ranges.len(), Ordering::Relaxed);
        self.changed();
    }

    /// Count a change of the buffers that joined a memory or of the placements, made under the
    /// lock the caller holds ([`version`](Self::version)).
    fn changed(&self) {
        self.version.fetch_add(1, Ordering::Relaxed);
    }

    /// How many times the buffers that joined a memory or the placements changed: two calls of
    /// [`augment`](Self::augment) with the same uses, between which it stood, give the same
    /// uses. A change made before a call (one the caller ordered before it) counts by then.
    pub(super) fn version(&self) -> u64 {
        self.version.load(Ordering::Relaxed)
    }

    /// How many uses looked the ranges their bytes meet up by address
    /// (`WorkCounts::host_range_lookups`, [`Placements::sharing`]).
    pub(super) fn range_lookups(&self) -> u64 {
        lock(&self.inner).placements.looked_up.get()
    }

    /// `uses` and, for each use of a buffer that joined a memory, a use of the memory's key of
    /// the same bytes and access, and for each use of bytes of a placed key, the uses of the
    /// keys its bytes share ([`Placements::sharing`]); `None` where no use takes another (`uses`
    /// as they are).
    pub(super) fn augment(&self, uses: &[Use<ResourceKey>]) -> Option<Vec<Use<ResourceKey>>> {
        let mut all = Vec::new();
        self.augment_into(uses, &mut all, None).then_some(all)
    }

    /// [`augment`](Self::augment) into `all`, emptied first: whether a use took another (`all`
    /// then holds `uses` and the uses they took; else it stays empty), so a caller that keeps
    /// `all` makes no vector at each call. The keys a given `quiet` holds while the version
    /// stands take no other use, and the keys found to take none join them, so a caller that
    /// keeps `quiet` too looks up no command whose uses of bytes name only such keys: one load of
    /// the version.
    pub(super) fn augment_into(
        &self,
        uses: &[Use<ResourceKey>],
        all: &mut Vec<Use<ResourceKey>>,
        mut quiet: Option<&mut Quiet>,
    ) -> bool {
        all.clear();
        let joined = self.any.load(Ordering::Relaxed);
        let crowded = self.crowded.load(Ordering::Relaxed);
        if !joined && !crowded {
            return false;
        }
        if let Some(quiet) = quiet.as_deref_mut() {
            // Read before the lock: a change made after it forgets the keys at the next command.
            quiet.stand(self.version());
            if uses.iter().all(|used| quiet.holds(used)) {
                return false;
            }
        }
        self.lookups.fetch_add(1, Ordering::Relaxed);
        let inner = lock(&self.inner);
        let aliases = joined.then_some(&*inner.aliases);
        let placements = crowded.then_some(&inner.placements);
        augmented((aliases, placements), uses, all, quiet)
    }

    /// How many times [`augment_into`](Self::augment_into) looked uses up under the lock
    /// (`WorkCounts::host_alias_lookups`).
    pub(super) fn lookups(&self) -> u64 {
        self.lookups.load(Ordering::Relaxed)
    }
}

/// One live buffer less over the memory `over` names: the memory is forgotten with its last,
/// unless a remap ended it already.
fn release<I>(memories: &mut Memories<Memory<I>>, over: Over) {
    let Some(noted) = memories.get_mut(over.memory) else {
        return;
    };
    if noted.serial != over.serial {
        return;
    }
    noted.buffers -= 1;
    if noted.buffers == 0 {
        memories.remove(over.memory);
    }
}

/// [`HostAliases::augment_into`] over, where given, `aliases` (the buffers that joined a memory)
/// and `placements`, into `all`, empty: the uses added follow `uses`, which go in at the first.
/// Whether any was added. The uses of keys `quiet` holds are passed over, and the keys of the
/// others that take no use, whatever their bytes, join them.
pub(super) fn augmented(
    (aliases, placements): (Option<&KeyMap<ResourceKey, Alias>>, Option<&Placements>),
    uses: &[Use<ResourceKey>],
    all: &mut Vec<Use<ResourceKey>>,
    mut quiet: Option<&mut Quiet>,
) -> bool {
    let mut add = |added: Use<ResourceKey>| {
        if all.is_empty() {
            all.reserve(2 * uses.len());
            all.extend_from_slice(uses);
        }
        all.push(added);
    };
    for used in uses {
        let UseRange::Bytes(bytes) = &used.range else {
            continue;
        };
        if quiet.as_deref().is_some_and(|quiet| quiet.holds(used)) {
            continue;
        }
        let joined = aliases
            .and_then(|aliases| aliases.get(&used.key))
            .map(|alias| alias.key);
        if let Some(key) = joined {
            add(Use {
                key,
                ..used.clone()
            });
        }
        let shares = placements.is_some_and(|placements| {
            placements.sharing(used.key, bytes, |key, range| {
                // The memory's key carries the same bytes already.
                if Some(key) != joined || range != *bytes {
                    add(Use {
                        key,
                        range: UseRange::Bytes(range),
                        ..used.clone()
                    });
                }
            })
        });
        if joined.is_none() && !shares {
            if let Some(quiet) = quiet.as_deref_mut() {
                quiet.keys.insert(used.key, ());
            }
        }
    }
    !all.is_empty()
}

/// The bytes each buffer over host pages lies over now, by key, and the ranges of bytes some key
/// lies over, each with the key its keys' uses share: the first placed there, while any key lies
/// there (the module docs).
#[derive(Default)]
pub(super) struct Placements {
    /// Each placed key's bytes: their first byte's host address and their length.
    at: KeyMap<ResourceKey, (u64, u64)>,
    /// Each range of bytes some key lies over, by its first byte's address and its length.
    ranges: Memories<Placed>,
    /// How many ranges are crowded ([`Placed::crowded`]).
    crowded: usize,
    /// How many uses looked the ranges their bytes meet up by address ([`Placements::sharing`]).
    looked_up: Cell<u64>,
}

/// A range of host bytes keys lie over.
struct Placed {
    /// The key its keys' uses share: the first placed there while any key lies there.
    key: ResourceKey,
    /// How many keys lie over it.
    keys: usize,
    /// Whether its key lies over it still.
    first: bool,
    /// The other ranges that meet it (share a byte with it), each with its key (a range's key
    /// is its own while it lives), in the order the lookup by address finds them ([`seat`]).
    meets: Vec<((u64, u64), ResourceKey)>,
}

impl Placed {
    /// Whether a use of a key over it takes another: a second key lies there, its key went, or
    /// another range meets it.
    fn crowded(&self) -> bool {
        self.keys > 1 || !self.first || !self.meets.is_empty()
    }

    /// The range over `at`, whose key is `key`, meets it from now on.
    fn meet(&mut self, at: (u64, u64), key: ResourceKey) {
        let index = self
            .meets
            .partition_point(|&(other, _)| seat(other) < seat(at));
        self.meets.insert(index, (at, key));
    }

    /// The range over `at` meets it no longer.
    fn part(&mut self, at: (u64, u64)) {
        let index = self
            .meets
            .partition_point(|&(other, _)| seat(other) < seat(at));
        if self.meets.get(index).is_some_and(|&(other, _)| other == at) {
            self.meets.remove(index);
        }
    }
}

/// Where the lookup by address finds the range over `at` among others ([`Memories::candidates`]):
/// by class of length, then by first byte and length.
fn seat(at: (u64, u64)) -> (usize, u64, u64) {
    (class(at.1), at.0, at.1)
}

impl Placements {
    /// Whether any range is crowded: else no use takes another.
    pub(super) fn crowded(&self) -> bool {
        self.crowded > 0
    }

    /// Place `key`, lying over no bytes, over `at`.
    pub(super) fn place(&mut self, key: ResourceKey, at: (u64, u64)) {
        if at.1 == 0 {
            return;
        }
        self.at.insert(key, at);
        if let Some(range) = self.ranges.get_mut(at) {
            let was = range.crowded();
            range.keys += 1;
            self.crowded += usize::from(!was && range.crowded());
            return;
        }
        let end = at.0.saturating_add(at.1);
        let met: Vec<(u64, u64)> = self.ranges.overlapping(at.0, end).collect();
        let mut meets = Vec::with_capacity(met.len());
        for other in met {
            if let Some(range) = self.ranges.get_mut(other) {
                let was = range.crowded();
                range.meet(at, key);
                self.crowded += usize::from(!was && range.crowded());
                meets.push((other, range.key));
            }
        }
        let range = Placed {
            key,
            keys: 1,
            first: true,
            meets,
        };
        self.crowded += usize::from(range.crowded());
        self.ranges.get_or_insert_with(at, || range);
    }

    /// `key` lies over no bytes from now on.
    pub(super) fn unplace(&mut self, key: ResourceKey) {
        let Some(at) = self.at.remove(&key) else {
            return;
        };
        let Some(range) = self.ranges.get_mut(at) else {
            return;
        };
        let was = range.crowded();
        range.keys -= 1;
        if range.key == key {
            range.first = false;
        }
        if range.keys > 0 {
            let now = range.crowded();
            self.crowded = self.crowded + usize::from(now) - usize::from(was);
            return;
        }
        self.crowded -= usize::from(was);
        let Some(gone) = self.ranges.remove(at) else {
            return;
        };
        for (other, _) in gone.meets {
            if let Some(range) = self.ranges.get_mut(other) {
                let was = range.crowded();
                range.part(at);
                self.crowded -= usize::from(was && !range.crowded());
            }
        }
    }

    /// For a use of `bytes` of `key` (counted from its first byte), each other key those bytes
    /// share, and the bytes of it they are (counted from its range's first byte), to `each`: its
    /// own range's key where another, and each other range's the bytes meet, clipped to it, in
    /// the order the lookup by address finds them. The ranges the bytes may meet are those that
    /// meet the key's own range, which it keeps: they are read from there while no more than the
    /// classes of length the lookup by address visits (a range query each, where a kept range is
    /// a comparison); a use of a range met by more, as one over many others' bytes is, looks up
    /// those its bytes meet (`WorkCounts::host_range_lookups`). Whether the key's uses may share
    /// another's at all: it lies over a crowded range ([`Placed::crowded`]), whatever `bytes`.
    pub(super) fn sharing(
        &self,
        key: ResourceKey,
        bytes: &std::ops::Range<u64>,
        mut each: impl FnMut(ResourceKey, std::ops::Range<u64>),
    ) -> bool {
        let Some(&at) = self.at.get(&key) else {
            return false;
        };
        let Some(own) = self.ranges.get(at) else {
            return false;
        };
        if !own.crowded() {
            return false;
        }
        if own.key != key {
            each(own.key, bytes.clone());
        }
        if own.meets.is_empty() {
            return true;
        }
        // The host bytes the use covers, within the key's own.
        let start = at.0.saturating_add(bytes.start.min(at.1));
        let end = at.0.saturating_add(bytes.end.min(at.1));
        let mut clipped = |other: (u64, u64), key: ResourceKey| {
            let from = start.max(other.0);
            let to = end.min(other.0.saturating_add(other.1));
            if from < to {
                each(key, from - other.0..to - other.0);
            }
        };
        if own.meets.len() <= self.ranges.held_classes() {
            for &(other, key) in &own.meets {
                clipped(other, key);
            }
            return true;
        }
        self.looked_up.set(self.looked_up.get() + 1);
        for other in self.ranges.overlapping(start, end) {
            if other == at {
                continue;
            }
            if let Some(range) = self.ranges.get(other) {
                clipped(other, range.key);
            }
        }
        true
    }
}

/// Values by the host bytes they are over (their first byte's address and their length), found
/// by address: the memories a remap ends. Each class of length (`class`) is ordered by first
/// byte, so the lookup of those with a byte in a range reads, in each class that holds a value,
/// the values whose first byte lies from the class's longest length before the range to its end
/// ([`Memories::candidates`]): the ones it finds, and of the class only those over the one
/// address half its longest length before the range, which end before it. The classes that hold
/// none it does not visit.
pub(super) struct Memories<T> {
    /// The values by class of length, each by first byte's address and length.
    classes: Vec<BTreeMap<(u64, u64), T>>,
    /// The classes that hold a value, bit `class` each.
    held: u128,
    /// How many values there are.
    len: usize,
}

/// The classes of length: one for each power of two from 2 a `u64` length may reach, and one
/// for the lengths of 0 and 1.
const CLASSES: usize = u64::BITS as usize + 1;
const _: () = assert!(
    CLASSES <= u128::BITS as usize,
    "a class a bit of `Memories::held`"
);

/// The class of `length`: the least `b` with `length <= 2^b` (so its lengths are past the class
/// below's longest).
fn class(length: u64) -> usize {
    (u64::BITS - length.saturating_sub(1).leading_zeros()) as usize
}

/// The longest length of the class `class`.
fn longest(class: usize) -> u64 {
    u32::try_from(class)
        .ok()
        .and_then(|class| 1u64.checked_shl(class))
        .unwrap_or(u64::MAX)
}

impl<T> Default for Memories<T> {
    fn default() -> Self {
        Memories {
            classes: (0..CLASSES).map(|_| BTreeMap::new()).collect(),
            held: 0,
            len: 0,
        }
    }
}

impl<T> Memories<T> {
    /// How many values there are.
    pub(super) fn len(&self) -> usize {
        self.len
    }

    /// The value over `memory` (its first byte's address and its length).
    pub(super) fn get(&self, memory: (u64, u64)) -> Option<&T> {
        self.classes[class(memory.1)].get(&memory)
    }

    /// The value over `memory`, to change.
    pub(super) fn get_mut(&mut self, memory: (u64, u64)) -> Option<&mut T> {
        self.classes[class(memory.1)].get_mut(&memory)
    }

    /// The value over `memory`, made by `make` where there is none.
    pub(super) fn get_or_insert_with(
        &mut self,
        memory: (u64, u64),
        make: impl FnOnce() -> T,
    ) -> &mut T {
        let class = class(memory.1);
        let by = &mut self.classes[class];
        if !by.contains_key(&memory) {
            self.len += 1;
            self.held |= 1 << class;
        }
        by.entry(memory).or_insert_with(make)
    }

    /// Remove the value over `memory`.
    pub(super) fn remove(&mut self, memory: (u64, u64)) -> Option<T> {
        let class = class(memory.1);
        let by = &mut self.classes[class];
        let removed = by.remove(&memory);
        self.len -= usize::from(removed.is_some());
        if by.is_empty() {
            self.held &= !(1 << class);
        }
        removed
    }

    /// The memories the lookup of those with a byte in `start..end` reads (the type's docs): of
    /// each class that holds a value, those whose first byte lies from its longest length before
    /// `start` to `end`.
    pub(super) fn candidates(&self, start: u64, end: u64) -> impl Iterator<Item = (u64, u64)> + '_ {
        let mut held = self.held;
        let classes = std::iter::from_fn(move || {
            let class = (held != 0).then(|| held.trailing_zeros() as usize)?;
            held &= held - 1;
            Some(class)
        });
        classes.flat_map(move |class| {
            let from = start.saturating_sub(longest(class) - 1);
            self.classes[class]
                .range((from, 0)..(end.max(from), 0))
                .map(|(&memory, _)| memory)
        })
    }

    /// How many classes hold a value: those the lookup visits.
    pub(super) fn held_classes(&self) -> usize {
        self.held.count_ones() as usize
    }

    /// The memories with a byte in `start..end`.
    pub(super) fn overlapping(
        &self,
        start: u64,
        end: u64,
    ) -> impl Iterator<Item = (u64, u64)> + '_ {
        self.candidates(start, end)
            .filter(move |&(at, length)| at < end && start < at.saturating_add(length))
    }
}
