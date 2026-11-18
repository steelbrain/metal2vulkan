//! The indices the new runtime's backend gives textures and sampler states, which the runtime
//! leaves to its consumer: the index the stream names for the object where the runtime takes it
//! (a descriptor's or a view's resource index, else the ID the program saw of the object), else
//! the lowest index of the object's class that no object holds and the program did not reserve,
//! as Metal's pool hands out a sampler state's ID. An index is held by the object placed there
//! last, as the runtime's is: a later object at an index an earlier one holds takes it, and the
//! earlier one's release leaves it.

use metal2vulkan_rt as rt;
use std::collections::{BTreeMap, BTreeSet, HashMap};

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
    /// The class slot and index each live object holds.
    objects: HashMap<u64, (usize, u32)>,
    /// The objects placed at the index the stream named.
    named: u64,
    /// The objects the stream named an index for that the runtime refused (it is outside the
    /// object's class): placed at the lowest free index instead.
    refused: u64,
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

    /// The index `object` held, free again where no later object was placed there.
    pub(in crate::replay) fn give(&mut self, object: u64) {
        if let Some((slot, index)) = self.objects.remove(&object) {
            if self.held[slot].get(&index) == Some(&Some(object)) {
                self.held[slot].remove(&index);
            }
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
    /// index the runtime refused, and the indices the program reserved.
    pub(in crate::replay) fn line(&self) -> String {
        let reserved: usize = self.reserved.iter().map(BTreeSet::len).sum();
        format!(
            "indices named {}: objects placed at the index the stream named; {} whose named index \
             the runtime refused, placed at the lowest free one; {reserved} reserved",
            self.named, self.refused
        )
    }
}
