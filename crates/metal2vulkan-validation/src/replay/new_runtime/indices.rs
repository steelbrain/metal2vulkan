//! The indices the new runtime's backend gives textures and sampler states, which the runtime
//! leaves to its consumer: the index the stream names for the object where the runtime takes it
//! (a descriptor's or a view's resource index, else the ID the program saw of the object), else
//! the lowest index of the object's class that no object holds and the program did not reserve,
//! as Metal's pool hands out a sampler state's ID.

use metal2vulkan_rt as rt;
use std::collections::{BTreeSet, HashMap};

/// The slot of `class`: textures, storage-capable textures, samplers (and any class the runtime
/// adds, which the backend asks for no index of).
fn slot(class: rt::IndexClass) -> usize {
    match class {
        rt::IndexClass::Texture => 0,
        rt::IndexClass::StorageTexture => 1,
        _ => 2,
    }
}

#[derive(Default)]
pub(in crate::replay) struct Indices {
    /// The indices of each class an object of the play holds, a constexpr sampler's for good.
    held: [BTreeSet<u32>; 3],
    /// The indices of each class the program reserved.
    reserved: [BTreeSet<u32>; 3],
    /// The class slot and index each live object holds.
    objects: HashMap<u64, (usize, u32)>,
    /// The objects placed at the index the stream named.
    named: u64,
    /// The objects the stream named an index for that the runtime refused (an object lives
    /// there, or it is outside the object's class): placed at the lowest free index instead.
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

    /// `index` of `class` held by `object` until [`give`](Self::give), or for good where
    /// `object` is `None` (a constexpr sampler).
    pub(in crate::replay) fn hold(
        &mut self,
        object: Option<u64>,
        class: rt::IndexClass,
        index: u32,
    ) {
        self.held[slot(class)].insert(index);
        if let Some(object) = object {
            self.objects.insert(object, (slot(class), index));
        }
    }

    /// The index `object` held, free again.
    pub(in crate::replay) fn give(&mut self, object: u64) {
        if let Some((slot, index)) = self.objects.remove(&object) {
            self.held[slot].remove(&index);
        }
    }

    /// The lowest index of `class` from `from` that no object holds, the program did not reserve
    /// and `skip` does not name.
    pub(in crate::replay) fn lowest(
        &self,
        class: rt::IndexClass,
        from: u32,
        skip: impl Fn(u32) -> bool,
    ) -> Option<u32> {
        let (held, reserved) = (&self.held[slot(class)], &self.reserved[slot(class)]);
        let range = class.range();
        (from.max(range.start)..range.end)
            .find(|index| !held.contains(index) && !reserved.contains(index) && !skip(*index))
    }

    /// `make` at `named` where the stream names an index and the runtime takes it, else at the
    /// lowest free index of `class`, an index the runtime finds in use passed over (an object the
    /// stream released that work in flight still holds); what `object` then holds. None where
    /// every index of the class is in use.
    pub(in crate::replay) fn place<T>(
        &mut self,
        object: u64,
        class: rt::IndexClass,
        named: Option<u32>,
        make: impl Fn(u32) -> Result<T, rt::Error>,
    ) -> Option<Result<T, rt::Error>> {
        if let Some(index) = named {
            match make(index) {
                Err(rt::Error::Refused(
                    rt::Refusal::IndexInUse { .. } | rt::Refusal::IndicesNotGranted { .. },
                )) => self.refused += 1,
                made => {
                    if made.is_ok() {
                        self.named += 1;
                        self.hold(Some(object), class, index);
                    }
                    return Some(made);
                }
            }
        }
        let mut from = 0;
        while let Some(index) = self.lowest(class, from, |_| false) {
            match make(index) {
                Err(rt::Error::Refused(rt::Refusal::IndexInUse { .. })) => from = index + 1,
                made => {
                    if made.is_ok() {
                        self.hold(Some(object), class, index);
                    }
                    return Some(made);
                }
            }
        }
        None
    }

    /// `make` at the highest index of `class` that no object holds and the program did not
    /// reserve, an index the runtime finds in use passed over: one away from the lowest indices,
    /// which the program's own objects take; what `object` then holds. None where every index of
    /// the class is in use.
    pub(in crate::replay) fn top<T>(
        &mut self,
        object: u64,
        class: rt::IndexClass,
        make: impl Fn(u32) -> Result<T, rt::Error>,
    ) -> Option<Result<T, rt::Error>> {
        let range = class.range();
        let mut below = range.end;
        loop {
            let (held, reserved) = (&self.held[slot(class)], &self.reserved[slot(class)]);
            let index = (range.start..below)
                .rev()
                .find(|index| !held.contains(index) && !reserved.contains(index))?;
            match make(index) {
                Err(rt::Error::Refused(rt::Refusal::IndexInUse { .. })) => below = index,
                made => {
                    if made.is_ok() {
                        self.hold(Some(object), class, index);
                    }
                    return Some(made);
                }
            }
        }
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
