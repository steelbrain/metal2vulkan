//! The indices the new runtime's backend gives textures and sampler states, which the runtime
//! leaves to its consumer: the index the stream names for the object where the runtime takes it
//! (a descriptor's or a view's resource index, else the ID the program saw of the object), else
//! the lowest index of the object's class that no object holds and the program did not reserve,
//! as Metal's pool hands out a sampler state's ID. An index is held by the object placed there
//! last, as the runtime's is: a later object at an index an earlier one holds takes it, and the
//! earlier one's release leaves it.
//!
//! An index is owed to the commits of the command buffers that named an object while it held the
//! index, the object live or released since: a placement there takes the index's element from
//! under that work, which Metal leaves to the
//! program never to do while the work runs (docs/RESOURCE-ID.md), and the program's own timing
//! kept that duty. The player keeps it by waiting for those commits to retire before such a
//! placement, and counts the waits and their time. Where one of them is held on records not yet
//! played (its command buffer waits for an event only a later record signals), the recording
//! device gave the index to the new object before that commit could complete, as its index policy
//! does: the player takes the index without waiting, as the recording did, and counts it on a line
//! of its own. The index stays owed to the held commit.

use metal2vulkan_rt as rt;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// The slot of `class`: textures, samplers (and any class the runtime adds, which the backend
/// asks for no index of).
fn slot(class: rt::IndexClass) -> usize {
    match class {
        rt::IndexClass::Texture => 0,
        _ => 1,
    }
}

#[derive(Default)]
pub(in crate::replay) struct Indices {
    /// The indices of each class an object of the play holds, each with the object placed there
    /// last (`None`: a constexpr sampler, for good).
    held: [BTreeMap<u32, Option<u64>>; 2],
    /// The indices of each class the program reserved.
    reserved: [BTreeSet<u32>; 2],
    /// The class slot and index each live object was placed at; it holds the index while it is
    /// the one placed there last (`held`).
    objects: HashMap<u64, (usize, u32)>,
    /// The objects placed at the index the stream named.
    named: u64,
    /// The objects the stream named an index for that the runtime refused (it is outside the
    /// object's class): placed at the lowest free index instead.
    refused: u64,
    /// The class slots and indices that the objects each command buffer not yet committed named
    /// held when it named them. An object named after a later one took its index holds it no
    /// more: the index is the later one's.
    naming: HashMap<u64, HashSet<(usize, u32)>>,
    /// Each index's last commit that named an object holding it, until a placement there finds
    /// it retired.
    owed: HashMap<(usize, u32), u64>,
    /// The placements that waited for an index's owed commits, and the nanoseconds they waited.
    waits: u64,
    wait_nanos: u64,
    /// The placements of a sampler state equal to the one placed last at its index, which took
    /// the index without waiting for the commits in flight it was owed to.
    equal: u64,
    /// The placements that took the index without waiting for a commit it was owed to that is held
    /// on records not yet played, as the recording device's index policy gave it.
    unwaited: u64,
}

impl Indices {
    /// `index` of `class` reserved by the program (`reserveResourceIndicesForResourceType:...`):
    /// no object takes it but where the stream names it.
    pub(in crate::replay) fn reserve(&mut self, class: rt::IndexClass, index: u32) {
        if class.range().contains(&index) {
            self.reserved[slot(class)].insert(index);
        }
    }

    /// `index` of `class` held by `object` until [`give`](Self::give) or until another object is
    /// placed there, or for good where `object` is `None` (a constexpr sampler).
    pub(in crate::replay) fn hold(
        &mut self,
        object: Option<u64>,
        class: rt::IndexClass,
        index: u32,
    ) {
        self.held[slot(class)].insert(index, object);
        if let Some(object) = object {
            self.objects.insert(object, (slot(class), index));
        }
    }

    /// The index `object` held, free again where no later object was placed there. The command
    /// buffers not yet committed that named it while it held the index still owe it once they
    /// are (`named`).
    pub(in crate::replay) fn give(&mut self, object: u64) {
        if let Some((slot, index)) = self.objects.remove(&object) {
            if self.held[slot].get(&index) == Some(&Some(object)) {
                self.held[slot].remove(&index);
            }
        }
    }

    /// An encoder of `command_buffer`, not yet committed, named `named`: the index each of them
    /// holds now is owed to its commit, once it is.
    pub(in crate::replay) fn named(&mut self, command_buffer: u64, named: &[u64]) {
        for object in named {
            let Some(&(slot, index)) = self.objects.get(object) else {
                continue;
            };
            if self.held[slot].get(&index) == Some(&Some(*object)) {
                self.naming
                    .entry(command_buffer)
                    .or_default()
                    .insert((slot, index));
            }
        }
    }

    fn owe(&mut self, key: (usize, u32), commit: u64) {
        let owed = self.owed.entry(key).or_default();
        *owed = commit.max(*owed);
    }

    /// `command_buffer` committed as commit `commit`: each index an object it named held when it
    /// was named is owed to it.
    pub(in crate::replay) fn committed(&mut self, command_buffer: u64, commit: u64) {
        for key in self.naming.remove(&command_buffer).unwrap_or_default() {
            self.owe(key, commit);
        }
    }

    /// `command_buffer` released uncommitted: it runs nothing, and owes nothing.
    pub(in crate::replay) fn ended(&mut self, command_buffer: u64) {
        self.naming.remove(&command_buffer);
    }

    /// The commit a placement at `index` of `class` waits for: the last that named an object
    /// holding it, where that has not retired (`retired`, the commits that have).
    pub(in crate::replay) fn owed(
        &mut self,
        class: rt::IndexClass,
        index: u32,
        retired: u64,
    ) -> Option<u64> {
        let key = (slot(class), index);
        let commit = *self.owed.get(&key)?;
        if commit <= retired {
            self.owed.remove(&key);
            return None;
        }
        Some(commit)
    }

    /// A placement at `index` of `class` of a sampler state equal to the one placed there last:
    /// it displaces nothing, so it takes the index without waiting for the commits the index is
    /// owed to, and is counted where one of them has not retired (`retired`, the commits that
    /// have). They stay owed: a placement of another sampler state there waits for them.
    pub(in crate::replay) fn equal(&mut self, class: rt::IndexClass, index: u32, retired: u64) {
        let key = (slot(class), index);
        match self.owed.get(&key) {
            Some(&commit) if commit > retired => self.equal += 1,
            Some(_) => drop(self.owed.remove(&key)),
            None => {}
        }
    }

    /// A placement took its index without waiting for an owed commit held on records not yet
    /// played, as the recording device's index policy did; the index stays owed to the commit.
    pub(in crate::replay) fn unwaited(&mut self) {
        self.unwaited += 1;
    }

    /// The index `object` holds, where it holds one.
    pub(in crate::replay) fn index_of(&self, object: u64) -> Option<u32> {
        self.objects.get(&object).map(|&(_, index)| index)
    }

    /// A placement waited `nanos` for an index's owed commits, which have retired.
    pub(in crate::replay) fn waited(&mut self, class: rt::IndexClass, index: u32, nanos: u64) {
        self.owed.remove(&(slot(class), index));
        self.waits += 1;
        self.wait_nanos += nanos;
    }

    /// The index [`place`](Self::place) takes for `named`: it where the runtime takes it (an
    /// index of the class), else the lowest free one.
    pub(in crate::replay) fn target(
        &self,
        class: rt::IndexClass,
        named: Option<u32>,
    ) -> Option<u32> {
        match named {
            Some(index) if class.range().contains(&index) => Some(index),
            _ => self.lowest(class, 0),
        }
    }

    /// The lowest index of `class` from `from` that no object holds and the program did not
    /// reserve.
    pub(in crate::replay) fn lowest(&self, class: rt::IndexClass, from: u32) -> Option<u32> {
        let (held, reserved) = (&self.held[slot(class)], &self.reserved[slot(class)]);
        let range = class.range();
        (from.max(range.start)..range.end)
            .find(|index| !held.contains_key(index) && !reserved.contains(index))
    }

    /// The highest index of `class` that no object holds, the program did not reserve and `skip`
    /// does not name: away from the lowest indices, which the program's own objects take.
    pub(in crate::replay) fn highest(
        &self,
        class: rt::IndexClass,
        skip: impl Fn(u32) -> bool,
    ) -> Option<u32> {
        let (held, reserved) = (&self.held[slot(class)], &self.reserved[slot(class)]);
        (class.range())
            .rev()
            .find(|index| !held.contains_key(index) && !reserved.contains(index) && !skip(*index))
    }

    /// `make` at `named` where the stream names an index and the runtime takes it (an index of
    /// the object's class), else at the lowest free index of `class`; what `object` then holds.
    /// None where every index of the class is held.
    pub(in crate::replay) fn place<T>(
        &mut self,
        object: u64,
        class: rt::IndexClass,
        named: Option<u32>,
        make: impl Fn(u32) -> Result<T, rt::Error>,
    ) -> Option<Result<T, rt::Error>> {
        if let Some(index) = named {
            match make(index) {
                Err(rt::Error::Refused(rt::Refusal::IndicesNotGranted { .. })) => self.refused += 1,
                made => {
                    if made.is_ok() {
                        self.named += 1;
                        self.hold(Some(object), class, index);
                    }
                    return Some(made);
                }
            }
        }
        let index = self.lowest(class, 0)?;
        let made = make(index);
        if made.is_ok() {
            self.hold(Some(object), class, index);
        }
        Some(made)
    }

    /// `make` at the [`highest`](Self::highest) free index of `class`; what `object` then holds.
    /// None where every index of the class is held.
    pub(in crate::replay) fn top<T>(
        &mut self,
        object: u64,
        class: rt::IndexClass,
        make: impl Fn(u32) -> Result<T, rt::Error>,
    ) -> Option<Result<T, rt::Error>> {
        let index = self.highest(class, |_| false)?;
        let made = make(index);
        if made.is_ok() {
            self.hold(Some(object), class, index);
        }
        Some(made)
    }

    /// The report's line: the objects placed at the index the stream named, those whose named
    /// index the runtime refused, the indices the program reserved, and the placements that
    /// waited for an index's owed commits, with the time waited.
    pub(in crate::replay) fn line(&self) -> String {
        let reserved: usize = self.reserved.iter().map(BTreeSet::len).sum();
        format!(
            "indices named {}: objects placed at the index the stream named; {} whose named index \
             the runtime refused, placed at the lowest free one; {reserved} reserved; {} placed \
             at an index whose holders' commits were in flight, after waiting for them ({} ns); {} \
             sampler states placed without waiting at an index whose last sampler state had the \
             same descriptor while its commits were in flight",
            self.named, self.refused, self.waits, self.wait_nanos, self.equal
        )
    }

    /// The report's line of the placements that took an index without waiting for a commit owed
    /// held on records not yet played.
    pub(in crate::replay) fn unwaited_line(&self) -> String {
        format!(
            "indices given while a commit they owe is held {}: placed without waiting at an index \
             owed to a command buffer no record played so far lets complete, as the recording \
             device's index policy gave the index to a new object before that command buffer \
             completed",
            self.unwaited
        )
    }
}
