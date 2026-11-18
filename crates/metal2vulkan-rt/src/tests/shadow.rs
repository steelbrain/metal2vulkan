//! The copy form's planning (`crate::shadow`), on the host: what a draw that reads an attachment
//! copies into its image's shadow first.
//!
//! - A slot read at the fragment's own pixel reads the scissor a pixel wider each way, clamped
//!   into its level; an empty scissor reads nothing; any other slot reads every level and layer
//!   of its texture.
//! - The pass's first read copies everything it reads; a read the shadow holds copies nothing; a
//!   wider read copies only the band that is new where that is one area, else all of it; after a
//!   memory barrier after draws, the same read copies again.
//! - A held area that covers another drops it, and past eight the oldest goes.
//! - A copy's barriers move each subresource once, the rendered level and layer apart from the
//!   rest.

use crate::shadow::{own_pixel, parted, ranges, whole, Area, Held};

/// An area of level 0, layer 0.
fn rect(x: [u32; 2], y: [u32; 2]) -> Area {
    Area {
        level: 0,
        layers: [0, 1],
        x,
        y,
    }
}

#[test]
fn an_own_pixel_read_is_the_scissor_a_pixel_wider_clamped_into_its_level() {
    // Inside the level: a pixel wider each way.
    assert_eq!(
        own_pixel(0, 0, [64, 64], [[8, 16], [4, 12]]),
        rect([7, 17], [3, 13])
    );
    // At the level's edges: clamped.
    assert_eq!(
        own_pixel(0, 0, [64, 64], [[0, 64], [0, 64]]),
        rect([0, 64], [0, 64])
    );
    // A level smaller than the scissor: the clamped coordinate reads its last texel.
    assert_eq!(
        own_pixel(2, 3, [16, 16], [[20, 40], [0, 8]]),
        Area {
            level: 2,
            layers: [3, 4],
            x: [15, 16],
            y: [0, 9],
        }
    );
    // An empty scissor draws no fragment, and reads nothing.
    assert!(own_pixel(0, 0, [64, 64], [[8, 8], [0, 64]]).is_empty());
    assert!(own_pixel(0, 0, [64, 64], [[0, 64], [9, 3]]).is_empty());
}

#[test]
fn any_other_read_is_every_level_and_layer_of_its_texture() {
    assert_eq!(
        whole(1, 2, 6, 6, [100, 30]),
        vec![
            Area {
                level: 1,
                layers: [6, 12],
                x: [0, 50],
                y: [0, 15],
            },
            Area {
                level: 2,
                layers: [6, 12],
                x: [0, 25],
                y: [0, 7],
            },
        ]
    );
    // A level of one texel on an axis stays one texel.
    assert_eq!(
        whole(3, 1, 0, 1, [8, 2]),
        vec![Area {
            level: 3,
            layers: [0, 1],
            x: [0, 1],
            y: [0, 1],
        }]
    );
}

#[test]
fn the_first_read_copies_it_and_a_held_read_copies_nothing() {
    let mut held = Held::default();
    let read = rect([0, 64], [0, 64]);
    assert_eq!(held.plan(&[read]), vec![read]);
    assert_eq!(held.plan(&[read]), Vec::<Area>::new());
    // A read inside what the shadow holds copies nothing either.
    assert_eq!(held.plan(&[rect([8, 16], [8, 16])]), Vec::<Area>::new());
    // An empty read copies nothing.
    assert_eq!(held.plan(&[rect([4, 4], [0, 64])]), Vec::<Area>::new());
}

#[test]
fn a_wider_read_copies_only_its_new_band_where_that_is_one_area() {
    let mut held = Held::default();
    assert_eq!(
        held.plan(&[rect([0, 64], [0, 32])]),
        vec![rect([0, 64], [0, 32])]
    );
    // Taller: the band below what is held.
    assert_eq!(
        held.plan(&[rect([0, 64], [0, 48])]),
        vec![rect([0, 64], [32, 48])]
    );
    // Wider on one side, within the held rows: the band to the right.
    assert_eq!(
        held.plan(&[rect([0, 80], [0, 48])]),
        vec![rect([64, 80], [0, 48])]
    );
    // What is new is two bands (an L): all of the read is copied.
    let corner = rect([0, 96], [0, 64]);
    assert_eq!(held.plan(&[corner]), vec![corner]);
    assert_eq!(held.plan(&[corner]), Vec::<Area>::new());
}

#[test]
fn a_read_of_another_level_or_layer_copies_all_of_it() {
    let mut held = Held::default();
    held.plan(&[rect([0, 64], [0, 64])]);
    let other_level = Area {
        level: 1,
        ..rect([0, 32], [0, 32])
    };
    assert_eq!(held.plan(&[other_level]), vec![other_level]);
    let other_layer = Area {
        layers: [1, 2],
        ..rect([0, 64], [0, 64])
    };
    assert_eq!(held.plan(&[other_layer]), vec![other_layer]);
    // Layers held whole hold a read of one of them.
    let mut held = Held::default();
    let layers = Area {
        layers: [0, 6],
        ..rect([0, 16], [0, 16])
    };
    held.plan(&[layers]);
    let one = Area {
        layers: [3, 4],
        ..rect([0, 16], [0, 16])
    };
    assert_eq!(held.plan(&[one]), Vec::<Area>::new());
}

#[test]
fn after_a_memory_barrier_the_same_read_copies_again() {
    let mut held = Held::default();
    let read = rect([7, 17], [3, 13]);
    assert_eq!(held.plan(&[read]), vec![read]);
    held.clear();
    assert_eq!(held.plan(&[read]), vec![read]);
}

#[test]
fn reads_of_one_draw_copy_what_an_earlier_one_did_not() {
    let mut held = Held::default();
    let left = rect([0, 32], [0, 64]);
    let both = rect([0, 64], [0, 64]);
    assert_eq!(
        held.plan(&[left, both]),
        vec![left, rect([32, 64], [0, 64])]
    );
    assert_eq!(held.plan(&[left, both]), Vec::<Area>::new());
}

#[test]
fn a_covering_area_drops_what_it_covers_and_past_eight_the_oldest_goes() {
    let mut held = Held::default();
    let strips: Vec<Area> = (0..9).map(|i| rect([i * 8, i * 8 + 8], [0, 8])).collect();
    for strip in &strips {
        assert_eq!(held.plan(&[*strip]), vec![*strip]);
    }
    // The first strip went past eight: read again, it is copied again.
    assert_eq!(held.plan(&[strips[0]]), vec![strips[0]]);
    // The last ones are still held.
    assert_eq!(held.plan(&[strips[8]]), Vec::<Area>::new());
    // One area over every strip drops them all and holds them.
    let all = rect([0, 72], [0, 8]);
    assert_eq!(held.plan(&[all]), vec![all]);
    for strip in &strips {
        assert_eq!(held.plan(&[*strip]), Vec::<Area>::new());
    }
}

#[test]
fn a_copys_barriers_move_each_subresource_once_the_rendered_one_apart() {
    let at = |level, layers| Area {
        level,
        layers,
        x: [0, 4],
        y: [0, 4],
    };
    let joined = ranges(&[at(0, [0, 2]), at(1, [0, 1]), at(0, [1, 4]), at(0, [6, 7])]);
    assert_eq!(joined, vec![(0, [0, 4]), (0, [6, 7]), (1, [0, 1])]);
    // Adjacent ranges of one level join; an empty area moves nothing.
    assert_eq!(
        ranges(&[at(2, [0, 1]), at(2, [1, 2]), at(3, [5, 5])]),
        vec![(2, [0, 2])]
    );
    // The rendered level and layer is parted from the rest, which is at home.
    assert_eq!(
        parted(&joined, (0, 2)),
        (
            true,
            vec![(0, [0, 2]), (0, [3, 4]), (0, [6, 7]), (1, [0, 1])]
        )
    );
    assert_eq!(parted(&[(0, [2, 3])], (0, 2)), (true, Vec::new()));
    assert_eq!(parted(&joined, (1, 1)), (false, joined.clone()));
}
