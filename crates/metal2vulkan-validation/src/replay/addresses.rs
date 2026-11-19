//! The address map (`docs/REPLAY-STREAM.md`, section 5): the recorded ranges of the live objects,
//! each with the address the player's runtime gave its object, nested where one lies in another (a
//! buffer in a block); where a typed address goes; and the census of the 8-byte words of written
//! bytes that fall in a range, which moves none of them.

use std::collections::{BTreeMap, HashMap};

/// One live range: its end, its object and the object's new address, and the ranges inside it.
struct Range {
    end: u64,
    object: u64,
    base: u64,
    inner: BTreeMap<u64, Range>,
}

/// The live recorded ranges, by start, each level's ranges apart.
#[derive(Default)]
pub struct AddressMap {
    top: BTreeMap<u64, Range>,
    /// Each object's recorded range.
    of: HashMap<u64, (u64, u64)>,
}

impl AddressMap {
    /// Add `object`'s recorded range `[start, start + length)`, which the player's runtime put at
    /// `base`. It lies inside a live range (one of the same bounds included: it is the inner one) or
    /// apart from each, or holds live ranges whole. A live range it meets in part ends, its inner
    /// ranges kept: the program put this object in that memory while the older one lived on, so a
    /// word there is this object's. An object with a live range is refused. The count is the live
    /// ranges it ended.
    pub fn insert(
        &mut self,
        object: u64,
        start: u64,
        length: u64,
        base: u64,
    ) -> Result<u64, String> {
        let end = start
            .checked_add(length)
            .filter(|_| length > 0)
            .ok_or_else(|| format!("#{object}: no range at {start:#x} of {length:#x} bytes"))?;
        if self.of.contains_key(&object) {
            return Err(format!("#{object} has a live range already"));
        }
        let mut ended = 0;
        loop {
            let mut level = &mut self.top;
            while let Some(at) = container(level, start, end) {
                level = &mut level.get_mut(&at).expect("a live range").inner;
            }
            let below = level.range(..start).next_back();
            let below = below.filter(|(_, range)| range.end > start);
            let held: Vec<u64> = level.range(start..end).map(|(&at, _)| at).collect();
            let parted = held.iter().filter(|at| level[at].end > end);
            let parted: Vec<u64> = (below.map(|(_, range)| range.object).into_iter())
                .chain(parted.map(|at| level[at].object))
                .collect();
            if parted.is_empty() {
                let inner = held
                    .into_iter()
                    .filter_map(|at| level.remove(&at).map(|range| (at, range)))
                    .collect();
                level.insert(
                    start,
                    Range {
                        end,
                        object,
                        base,
                        inner,
                    },
                );
                self.of.insert(object, (start, end));
                return Ok(ended);
            }
            for older in parted {
                self.remove(older);
                ended += 1;
            }
        }
    }

    /// Remove `object`'s range, if it has one; the ranges inside it stay, at its level.
    pub fn remove(&mut self, object: u64) {
        let Some((start, end)) = self.of.remove(&object) else {
            return;
        };
        let mut level = &mut self.top;
        loop {
            let found = level
                .get(&start)
                .is_some_and(|range| range.object == object);
            if found {
                let range = level.remove(&start).expect("found");
                level.extend(range.inner);
                return;
            }
            match container(level, start, end) {
                Some(at) => level = &mut level.get_mut(&at).expect("a live range").inner,
                None => return,
            }
        }
    }

    /// Whether `object` has a live range.
    pub fn holds(&self, object: u64) -> bool {
        self.of.contains_key(&object)
    }

    /// Where `value` goes: its innermost live range's new address plus its offset into the range.
    pub fn relocated(&self, value: u64) -> Option<u64> {
        let mut level = &self.top;
        let mut found = None;
        while let Some((&start, range)) = level.range(..=value).next_back() {
            if value >= range.end {
                break;
            }
            found = Some(range.base.wrapping_add(value - start));
            level = &range.inner;
        }
        found
    }

    /// Count the 8-byte words of `bytes` that start at a multiple of 8 from the target's start
    /// (`bytes` starts `offset` bytes into the target) and lie in a live range, and the 8-byte
    /// values at 4 past a multiple of 8 that do. Nothing is rewritten: a word's value is no reason
    /// to move it.
    pub fn census(&self, bytes: &[u8], offset: u64) -> (u64, u64) {
        // The live ranges' span: a level's ranges are disjoint, so the top level's first starts it
        // and its last ends it. A word outside it lies in no range and takes no lookup.
        let (Some((&low, _)), Some((_, last))) =
            (self.top.first_key_value(), self.top.last_key_value())
        else {
            return (0, 0);
        };
        let high = last.end;
        let relocated = |value: u64| match (low..high).contains(&value) {
            true => self.relocated(value),
            false => None,
        };
        let word = |at: usize, bytes: &[u8]| {
            let at = bytes.get(at..at + 8)?;
            Some(u64::from_le_bytes(at.try_into().expect("8 bytes")))
        };
        let counted = |first: u64| {
            (first as usize..bytes.len())
                .step_by(8)
                .filter(|&at| word(at, bytes).and_then(relocated).is_some())
                .count() as u64
        };
        (
            counted((8 - offset % 8) % 8),
            counted((12 - offset % 8) % 8),
        )
    }

    /// The first 8-byte word of `bytes` at a multiple of 8 from the target's start (`bytes` starts
    /// `offset` bytes into the target) whose value lies in the live range of an object `noted`
    /// holds, at any level: the word's offset in the target, its value and that object.
    pub fn noted_word(
        &self,
        bytes: &[u8],
        offset: u64,
        noted: impl Fn(u64) -> bool,
    ) -> Option<(u64, u64, u64)> {
        let first = ((8 - offset % 8) % 8) as usize;
        (first..bytes.len().saturating_sub(7))
            .step_by(8)
            .find_map(|at| {
                let value = u64::from_le_bytes(bytes[at..at + 8].try_into().expect("8 bytes"));
                let object = self.holders(value).find(|&object| noted(object))?;
                Some((offset + at as u64, value, object))
            })
    }

    /// The objects whose live ranges hold `value`, outermost first.
    fn holders(&self, value: u64) -> impl Iterator<Item = u64> + '_ {
        let mut level = Some(&self.top);
        std::iter::from_fn(move || {
            let (_, range) = level?.range(..=value).next_back()?;
            if value >= range.end {
                level = None;
                return None;
            }
            level = Some(&range.inner);
            Some(range.object)
        })
    }
}

/// The start of the range of `level` that holds `[start, end)` whole, if one does.
fn container(level: &BTreeMap<u64, Range>, start: u64, end: u64) -> Option<u64> {
    let (&at, range) = level.range(..=start).next_back()?;
    (range.end >= end).then_some(at)
}
