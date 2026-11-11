//! A command buffer's descriptor pools (`descriptor_arena.rs`): how many pools its commands' set-0
//! sets open, and which pool each set comes from.

use crate::descriptor_arena::{Book, Room, SetNeeds, FIRST, MOST, MOST_DESCRIPTORS};
use ash::vk;

/// Needs of `counts` by descriptor type, without flags.
fn needs(counts: &[(vk::DescriptorType, u32)]) -> SetNeeds {
    let mut counts: Vec<(i32, u32)> = counts.iter().map(|&(ty, n)| (ty.as_raw(), n)).collect();
    counts.sort_unstable();
    SetNeeds {
        counts,
        flags: vk::DescriptorPoolCreateFlags::empty(),
    }
}

/// Allocate one set of `needs` in `book`: the pool it comes from, and whether it opened one.
fn allocate(book: &mut Book, needs: &SetNeeds) -> (usize, bool) {
    let (at, opened) = match book.place(needs) {
        Ok(at) => (at, false),
        Err(sets) => {
            book.opened(Room::of(needs, sets));
            (book.rooms.len() - 1, true)
        }
    };
    book.take(at, needs);
    (at, opened)
}

/// The regression check: 2000 dispatches of one variant in one command buffer open seven pools (16,
/// 32, ... 1024 sets), not 2000; each pool fills before the next opens.
#[test]
fn many_commands_of_one_variant_open_pools_that_double() {
    let two_buffers = needs(&[(vk::DescriptorType::STORAGE_BUFFER, 2)]);
    let mut book = Book::default();
    let mut opened = Vec::new();
    for command in 0..2000 {
        let (at, new) = allocate(&mut book, &two_buffers);
        if new {
            opened.push(command);
        }
        assert_eq!(at, book.rooms.len() - 1, "command {command}");
    }
    assert_eq!(opened, [0, 16, 48, 112, 240, 496, 1008]);
    // Every pool but the newest is full; the newest, of `MOST` sets, holds the last 992.
    for room in &book.rooms[..book.rooms.len() - 1] {
        assert_eq!(room.sets, 0);
        assert_eq!(
            room.counts,
            [(vk::DescriptorType::STORAGE_BUFFER.as_raw(), 0)]
        );
    }
    let newest = book.rooms.last().unwrap();
    assert_eq!(newest.sets, MOST - 992);
    assert_eq!(
        newest.counts,
        [(
            vk::DescriptorType::STORAGE_BUFFER.as_raw(),
            2 * (MOST - 992)
        )]
    );
}

/// A command buffer of one command opens one pool of `FIRST` sets.
#[test]
fn one_command_opens_one_pool() {
    let one = needs(&[
        (vk::DescriptorType::UNIFORM_BUFFER, 1),
        (vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 3),
    ]);
    let mut book = Book::default();
    assert_eq!(book.place(&one), Err(FIRST));
    allocate(&mut book, &one);
    assert_eq!(book.rooms.len(), 1);
    assert_eq!(book.rooms[0].sets, FIRST - 1);
    assert_eq!(
        book.rooms[0].counts,
        [
            (
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER.as_raw(),
                3 * (FIRST - 1)
            ),
            (vk::DescriptorType::UNIFORM_BUFFER.as_raw(), FIRST - 1),
        ]
    );
}

/// A set of many descriptors opens pools of fewer sets, at most `MOST_DESCRIPTORS` descriptors
/// each, and one set a pool when one set needs more.
#[test]
fn a_pool_holds_at_most_its_descriptor_bound() {
    let wide = needs(&[(vk::DescriptorType::SAMPLED_IMAGE, 5000)]);
    let book = Book::default();
    let sets = book.place(&wide).unwrap_err();
    assert_eq!(u64::from(sets), MOST_DESCRIPTORS / 5000);
    let huge = needs(&[(vk::DescriptorType::SAMPLED_IMAGE, 100_000)]);
    assert_eq!(book.place(&huge), Err(1));
}

/// Two variants of other descriptor types each take sets from the pools with room for them: a
/// pool opened for one is not used by the other, and neither opens a pool while one of its own
/// has room.
#[test]
fn each_needs_takes_the_newest_pool_with_room() {
    let buffers = needs(&[(vk::DescriptorType::STORAGE_BUFFER, 2)]);
    let images = needs(&[(vk::DescriptorType::STORAGE_IMAGE, 1)]);
    let mut book = Book::default();
    let mut from_buffers = std::collections::BTreeSet::new();
    let mut from_images = std::collections::BTreeSet::new();
    for _ in 0..40 {
        from_buffers.insert(allocate(&mut book, &buffers).0);
        from_images.insert(allocate(&mut book, &images).0);
    }
    assert!(from_buffers.is_disjoint(&from_images));
    // 40 sets each: 16 + 64 for one, 32 + 128 for the other (they open pools in turn).
    assert_eq!(book.rooms.len(), 4);
}

/// A pool of other flags (a heap's update-after-bind) is never shared.
#[test]
fn pools_of_other_flags_are_not_shared() {
    let plain = needs(&[(vk::DescriptorType::STORAGE_BUFFER, 1)]);
    let mut after_bind = plain.clone();
    after_bind.flags = vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND;
    let mut book = Book::default();
    let (first, _) = allocate(&mut book, &plain);
    let (second, opened) = allocate(&mut book, &after_bind);
    assert!(opened);
    assert_ne!(first, second);
}

/// A fitted arena counts the descriptors of the pool it would open from the set's needs alone
/// (`SetNeeds::pool_descriptors`): what that pool's room holds, every type's count saturating as
/// the room's does.
#[test]
fn a_pools_descriptors_are_counted_from_its_sets_needs_as_its_room_holds_them() {
    let mixed = needs(&[
        (vk::DescriptorType::STORAGE_BUFFER, 3),
        (vk::DescriptorType::SAMPLED_IMAGE, 2),
        (vk::DescriptorType::STORAGE_IMAGE, 1),
    ]);
    let large = needs(&[(vk::DescriptorType::STORAGE_BUFFER, u32::MAX / 2)]);
    for set in [&mixed, &large] {
        for sets in [1, 2, FIRST, MOST, 7] {
            assert_eq!(
                set.pool_descriptors(sets),
                Room::of(set, sets).descriptors(),
                "{sets} sets"
            );
        }
    }
}
