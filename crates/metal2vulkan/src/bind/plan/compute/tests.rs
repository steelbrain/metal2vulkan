//! Specialization encoding of compute plans. Plans of compiled pipelines are
//! tested end to end in `src/tests/engine_binder.rs`.

use super::*;
use crate::pipeline::sym::Axis;

#[test]
fn specialization_data_packs_each_value_at_its_size_in_id_order() {
    let c = |id, key, size| SpecConstant {
        id,
        key,
        size,
        default: 0,
        unread_bits: 0,
    };
    let layout = [
        c(0, SpecKey::LocalSize(Axis::X), 4),
        c(1, SpecKey::MultiRegion, 4),
        c(2, SpecKey::TgLength(3), 4),
    ];
    let key = spec_key(&layout, |k| match k {
        SpecKey::LocalSize(_) => Some(0x0102_0304),
        SpecKey::TgLength(3) => Some(48),
        _ => None,
    });
    assert_eq!(key.data, vec![4, 3, 2, 1, 48, 0, 0, 0]);
    let ids: Vec<(u32, u32, u32)> = key
        .entries
        .iter()
        .map(|e| (e.constant_id, e.offset, e.size))
        .collect();
    assert_eq!(ids, vec![(0, 0, 4), (2, 4, 4)]);
}

/// A key's lists are made at once at the size every constant of the layout takes, not grown a
/// push at a time: where each constant has a value, each list's capacity is its length.
#[test]
fn specialization_lists_are_made_whole_at_once() {
    let c = |id, key, size| SpecConstant {
        id,
        key,
        size,
        default: 0,
        unread_bits: 0,
    };
    let layout = [
        c(0, SpecKey::LocalSize(Axis::X), 4),
        c(1, SpecKey::LocalSize(Axis::Y), 4),
        c(2, SpecKey::LocalSize(Axis::Z), 4),
        c(3, SpecKey::FlatLocalSize, 4),
        c(4, SpecKey::MultiRegion, 1),
    ];
    let key = spec_key(&layout, |_| Some(1));
    assert_eq!((key.entries.len(), key.data.len()), (5, 17));
    assert_eq!(key.entries.capacity(), key.entries.len());
    assert_eq!(key.data.capacity(), key.data.len());
}
