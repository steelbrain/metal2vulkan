//! The held depths' moves between encoders (`texture/held_depth.rs`): each batch moves a held
//! image into the layout each encoder that names it needs, from the layout the batches submitted
//! before left it in, and a move is a write of the whole image the tracker orders; an encoder of
//! a group recorded into one `VkCommandBuffer` (`encoder_fold.rs`) has its moves go before the
//! group.

use crate::encoder_fold::GroupPlaces;
use crate::resource::ResourceKey;
use crate::texture::held_depth::{need, BatchMoves, HeldImage, HeldImages, HELD};
use crate::tracker::{Access, ImagePlace, Use, UseRange};
use ash::vk::{self, Handle};
use metal2vulkan::vk::{ImageAspectFlags, ImageLayout};
use std::collections::BTreeSet;

const GENERAL: ImageLayout = ImageLayout::GENERAL;

/// A registry of one held 4-layer depth image, and its key.
fn one() -> (HeldImages, ResourceKey) {
    let images = HeldImages::default();
    let key = ResourceKey::new();
    let raw = vk::Image::from_raw(0x10);
    images.insert(HeldImage::new(key, raw, ImageAspectFlags::DEPTH, 1, 4));
    (images, key)
}

/// An encoder's record of what it names and holds: `held` holds `key`, which `named` names.
fn record(key: ResourceKey, named: bool, held: bool) -> (BTreeSet<ResourceKey>, Vec<ResourceKey>) {
    let names = match named {
        true => BTreeSet::from([key]),
        false => BTreeSet::new(),
    };
    let holds = match held {
        true => vec![key],
        false => Vec::new(),
    };
    (names, holds)
}

/// The moves of a batch of `encoders`, each its own place.
fn moves(
    images: &HeldImages,
    encoders: &[(BTreeSet<ResourceKey>, Vec<ResourceKey>)],
) -> BatchMoves {
    images.moves(
        (encoders.iter().enumerate()).map(|(at, (named, held))| (at, named, held.as_slice())),
    )
}

/// The moves before each of a batch's `count` encoders, as (old, new) layouts.
fn layouts(moves: &BatchMoves, count: usize) -> Vec<Vec<(ImageLayout, ImageLayout)>> {
    (0..count)
        .map(|at| {
            moves
                .transitions(at)
                .iter()
                .map(|t| (t.old, t.new))
                .collect()
        })
        .collect()
}

/// A holding encoder needs the attachment layout, any other `GENERAL`.
#[test]
fn a_holding_encoder_needs_the_attachment_layout() {
    assert_eq!(need(true), HELD);
    assert_eq!(need(false), GENERAL);
    assert_eq!(HELD, ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
}

/// Hold, sample, hold in one batch: in, out (one release), in; the next batch's holding encoder
/// moves nothing once the first was submitted, and a sampling one moves it out.
#[test]
fn a_batch_moves_the_image_into_each_encoder_s_layout() {
    let (images, key) = one();
    let batch = [
        record(key, true, true),
        record(key, true, false),
        record(key, true, true),
    ];
    let planned = moves(&images, &batch);
    assert_eq!(
        layouts(&planned, 3),
        vec![
            vec![(GENERAL, HELD)],
            vec![(HELD, GENERAL)],
            vec![(GENERAL, HELD)]
        ]
    );
    assert_eq!(planned.released, 1);
    planned.submitted();
    let again = moves(&images, &[record(key, true, true)]);
    assert_eq!(layouts(&again, 1), vec![Vec::new()]);
    assert!(again.is_empty());
    assert_eq!(again.released, 0);
    again.submitted();
    let sampled = moves(&images, &[record(key, true, false)]);
    assert_eq!(layouts(&sampled, 1), vec![vec![(HELD, GENERAL)]]);
    assert_eq!(sampled.released, 1);
}

/// A batch not submitted leaves the layout as the batches before left it: the next plans from
/// there.
#[test]
fn a_batch_not_submitted_changes_no_layout() {
    let (images, key) = one();
    let dropped = moves(&images, &[record(key, true, true)]);
    assert_eq!(layouts(&dropped, 1), vec![vec![(GENERAL, HELD)]]);
    drop(dropped);
    let next = moves(&images, &[record(key, true, false)]);
    assert_eq!(layouts(&next, 1), vec![Vec::new()]);
}

/// An encoder that names no held image moves none, and the image stays as it was for the next.
#[test]
fn an_encoder_that_does_not_name_the_image_moves_nothing() {
    let (images, key) = one();
    moves(&images, &[record(key, true, true)]).submitted();
    let other = ResourceKey::new();
    let unnamed = moves(&images, &[(BTreeSet::from([other]), Vec::new())]);
    assert_eq!(layouts(&unnamed, 1), vec![Vec::new()]);
    unnamed.submitted();
    let held = moves(&images, &[record(key, true, true)]);
    assert_eq!(layouts(&held, 1), vec![Vec::new()]);
}

/// A move's use is a tracked write of the whole image in every stage, so the tracker orders it
/// after every earlier use of the image, on any queue, and every later use after it.
#[test]
fn a_move_is_a_write_of_the_whole_image() {
    let (images, key) = one();
    let planned = moves(&images, &[record(key, true, true)]);
    let expected = Use::writes(
        key,
        true,
        UseRange::Image {
            aspects: vk::ImageAspectFlags::DEPTH,
            levels: 0..1,
            layers: 0..4,
            place: ImagePlace::WHOLE,
        },
        Access::new(
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_WRITE,
        ),
    );
    assert_eq!(planned.uses(0), &[expected][..]);
}

/// An image gone is moved no more; a registry of none plans no move at all.
#[test]
fn an_image_gone_is_not_moved() {
    let (images, key) = one();
    images.remove(key);
    assert!(!images.holds(key));
    let planned = moves(
        &images,
        &[record(key, true, true), record(key, true, false)],
    );
    assert_eq!(layouts(&planned, 2), vec![Vec::new(), Vec::new()]);
    assert!(planned.is_empty());
}

/// A group of three encoders recorded into the first's `VkCommandBuffer` (`encoder_fold.rs`),
/// then one of its own: the second, which samples the image the batch before left held, has its
/// move go before the group's first, where the shared `VkCommandBuffer` begins, not before its
/// own part, which holds no `VkCommandBuffer` and comes after the commands it moves for.
#[test]
fn a_grouped_encoder_s_move_goes_before_its_group() {
    let (images, key) = one();
    moves(&images, &[record(key, true, true)]).submitted();
    let group = 0..3;
    let mut places = GroupPlaces::default();
    let at: Vec<usize> = [Some(&group), Some(&group), Some(&group), None]
        .into_iter()
        .enumerate()
        .map(|(at, group)| places.place(at, at, group))
        .collect();
    assert_eq!(at, [0, 0, 0, 3]);
    let batch = [
        record(key, false, false),
        record(key, true, false),
        record(key, false, false),
        record(key, true, false),
    ];
    let planned = images.moves(
        (batch.iter().zip(&at)).map(|((named, held), &place)| (place, named, held.as_slice())),
    );
    assert_eq!(
        layouts(&planned, 4),
        vec![vec![(HELD, GENERAL)], Vec::new(), Vec::new(), Vec::new()]
    );
    assert_eq!(planned.released, 1);
}

/// Two groups of one batch each take their own first part's place, and an encoder of no group
/// its own.
#[test]
fn each_group_s_encoders_take_their_own_group_s_first_place() {
    let (first, second) = (0..2, 3..5);
    let mut places = GroupPlaces::default();
    let groups = [
        Some(&first),
        Some(&first),
        None,
        Some(&second),
        Some(&second),
    ];
    let at: Vec<usize> = (groups.into_iter().enumerate())
        .map(|(at, group)| places.place(at, at, group))
        .collect();
    assert_eq!(at, [0, 0, 2, 3, 3]);
}
