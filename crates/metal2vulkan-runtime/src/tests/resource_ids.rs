//! The resource-ID spaces (`resource_ids.rs`): plain IDs from 1, the lowest free one reused, a
//! space per kind of resource; placed IDs, reservations, and each ID's slot, whose generation
//! changes with its holder.

use super::capabilities::{committed_descriptions, created};
use crate::resource_ids::{
    check_capacity, texture_arrays, ArrayBound, Holder, IdHold, IdPlacement, IdSpace, ResourceId,
    ResourceIds,
};
use crate::Error;
use metal2vulkan::bind::HANDLE_TABLE_CAPACITY;
use metal2vulkan::device::{Device, DevicePolicy, VkLimit};
use metal2vulkan::error::{Capability, Feature, Unsupported};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, Weak};

fn ids() -> Arc<Mutex<ResourceIds>> {
    Arc::default()
}

fn id(ids: &Arc<Mutex<ResourceIds>>, space: IdSpace) -> IdHold {
    IdHold::take(ids, space, IdPlacement::Any)
}

fn at(ids: &Arc<Mutex<ResourceIds>>, space: IdSpace, id: u64) -> IdHold {
    IdHold::take(ids, space, IdPlacement::At(ResourceId(id)))
}

/// A texture ID taken at `placement` that holds its slot, as a program's texture's does (its
/// holder never lives: the slot's bookkeeping is what these tests read).
fn held(ids: &Arc<Mutex<ResourceIds>>, placement: IdPlacement) -> IdHold {
    let hold = IdHold::take(ids, IdSpace::Texture, placement);
    hold.hold(Holder::Texture(Weak::new()));
    hold
}

/// The generation of texture slot `id`, and whether it has a holder.
fn slot(ids: &Arc<Mutex<ResourceIds>>, id: u64) -> (u64, bool) {
    let (generation, holder) = ids.lock().unwrap().texture(id);
    (generation, holder.is_some())
}

#[test]
fn ids_are_dense_from_one() {
    let ids = ids();
    let held: Vec<IdHold> = (0..4).map(|_| id(&ids, IdSpace::Texture)).collect();
    let got: Vec<ResourceId> = held.iter().map(IdHold::id).collect();
    assert_eq!(got, [1, 2, 3, 4].map(ResourceId));
}

#[test]
fn the_lowest_free_id_is_reused() {
    let ids = ids();
    let mut held: Vec<Option<IdHold>> = (0..4).map(|_| Some(id(&ids, IdSpace::Texture))).collect();
    held[2] = None;
    held[1] = None;
    assert_eq!(id(&ids, IdSpace::Texture).id(), ResourceId(2));
    // The ID just taken went back on drop, so 2 is still the lowest free one.
    let two = id(&ids, IdSpace::Texture);
    let three = id(&ids, IdSpace::Texture);
    assert_eq!((two.id(), three.id()), (ResourceId(2), ResourceId(3)));
    assert_eq!(id(&ids, IdSpace::Texture).id(), ResourceId(5));
}

#[test]
fn ids_freed_at_the_top_come_back_in_order() {
    let ids = ids();
    let one = id(&ids, IdSpace::Sampler);
    let two = id(&ids, IdSpace::Sampler);
    let three = id(&ids, IdSpace::Sampler);
    drop(two);
    drop(three);
    // 2 and 3 are free and at the top: the next two IDs are 2 and 3 again, then 4.
    let again: Vec<IdHold> = (0..3).map(|_| id(&ids, IdSpace::Sampler)).collect();
    let got: Vec<ResourceId> = again.iter().map(IdHold::id).collect();
    assert_eq!(got, [2, 3, 4].map(ResourceId));
    drop(again);
    drop(one);
    assert_eq!(id(&ids, IdSpace::Sampler).id(), ResourceId(1));
}

#[test]
fn textures_and_samplers_have_spaces_of_their_own() {
    let ids = ids();
    let texture = id(&ids, IdSpace::Texture);
    let sampler = id(&ids, IdSpace::Sampler);
    let second = id(&ids, IdSpace::Texture);
    assert_eq!(texture.id(), ResourceId(1));
    assert_eq!(sampler.id(), ResourceId(1));
    assert_eq!(second.id(), ResourceId(2));
}

/// Probe 2's order on an M3 Max and an M4: three tables made among textures and a sampler state
/// take 1, 2 and 3, and a table made once the first is released takes 1.
#[test]
fn function_tables_have_a_space_of_their_own() {
    let ids = ids();
    let _texture = id(&ids, IdSpace::Texture);
    let first = id(&ids, IdSpace::FunctionTable);
    let _sampler = id(&ids, IdSpace::Sampler);
    let second = id(&ids, IdSpace::FunctionTable);
    let _texture2 = id(&ids, IdSpace::Texture);
    let third = id(&ids, IdSpace::FunctionTable);
    let got = [&first, &second, &third].map(|held| held.id());
    assert_eq!(got, [1, 2, 3].map(ResourceId));
    drop(first);
    assert_eq!(id(&ids, IdSpace::FunctionTable).id(), ResourceId(1));
}

#[test]
fn zero_is_never_an_id() {
    let ids = ids();
    for _ in 0..3 {
        let held = id(&ids, IdSpace::Texture);
        assert_ne!(held.id(), ResourceId(0));
    }
    // A placement at 0 is plain allocation, as Metal's forced index 0 is.
    assert_eq!(at(&ids, IdSpace::Texture, 0).id(), ResourceId(1));
}

/// A placed ID is the one asked for, held or not, and plain allocation does not count it: it may
/// hand the ID out again, and a placed ID going frees no plain one.
#[test]
fn placed_ids_are_outside_plain_allocation() {
    let ids = ids();
    let placed = at(&ids, IdSpace::Texture, 2);
    let again = at(&ids, IdSpace::Texture, 2);
    assert_eq!((placed.id(), again.id()), (ResourceId(2), ResourceId(2)));
    let plain: Vec<IdHold> = (0..2).map(|_| id(&ids, IdSpace::Texture)).collect();
    let got: Vec<ResourceId> = plain.iter().map(IdHold::id).collect();
    assert_eq!(got, [1, 2].map(ResourceId));
    let over_one = at(&ids, IdSpace::Texture, 1);
    drop(over_one);
    assert_eq!(
        id(&ids, IdSpace::Texture).id(),
        ResourceId(3),
        "1 is still the plain one's"
    );
}

/// Plain allocation skips reserved IDs, made before it handed one out, for good; a reservation
/// once it handed one out reserves nothing.
#[test]
fn reserved_ids_stay_out_of_plain_allocation() {
    let ids = ids();
    assert!(ids
        .lock()
        .unwrap()
        .reserve(IdSpace::Texture, &[ResourceId(2), ResourceId(4)]));
    let plain: Vec<IdHold> = (0..3).map(|_| id(&ids, IdSpace::Texture)).collect();
    let got: Vec<ResourceId> = plain.iter().map(IdHold::id).collect();
    assert_eq!(got, [1, 3, 5].map(ResourceId));
    // A resource placed at a reserved ID and gone leaves it reserved.
    drop(at(&ids, IdSpace::Texture, 2));
    drop(plain);
    let again: Vec<IdHold> = (0..3).map(|_| id(&ids, IdSpace::Texture)).collect();
    let got: Vec<ResourceId> = again.iter().map(IdHold::id).collect();
    assert_eq!(got, [1, 3, 5].map(ResourceId));
    // Too late: ignored.
    assert!(!ids
        .lock()
        .unwrap()
        .reserve(IdSpace::Texture, &[ResourceId(6)]));
    assert_eq!(id(&ids, IdSpace::Texture).id(), ResourceId(6));
    // The other space has handed out nothing yet.
    assert!(ids
        .lock()
        .unwrap()
        .reserve(IdSpace::Sampler, &[ResourceId(1)]));
    assert_eq!(id(&ids, IdSpace::Sampler).id(), ResourceId(2));
}

/// Each slot's generation changes with its holder: a resource made with the ID, placed or plain,
/// and its holder going. A resource that took the ID but no longer holds the slot changes
/// nothing as it goes, and a slot whose holder went is gone.
#[test]
fn a_slots_generation_changes_with_its_holder() {
    let ids = ids();
    assert_eq!(slot(&ids, 1), (0, false), "never held");
    let first = held(&ids, IdPlacement::Any);
    let (one, holder) = slot(&ids, 1);
    assert!(one > 0 && holder);
    let second = held(&ids, IdPlacement::At(ResourceId(1)));
    let (two, _) = slot(&ids, 1);
    assert!(two > one);
    drop(first);
    assert_eq!(
        slot(&ids, 1),
        (two, true),
        "the first no longer held the slot"
    );
    drop(second);
    assert_eq!(slot(&ids, 1), (0, false), "the holder went");
}

/// The runtime's own resources take no plain ID and leave reservations open.
#[test]
fn internal_resources_take_no_id() {
    let ids = ids();
    let internal = IdHold::internal(&ids, IdSpace::Texture);
    assert_eq!(internal.id(), ResourceId(0));
    assert!(ids
        .lock()
        .unwrap()
        .reserve(IdSpace::Texture, &[ResourceId(1)]));
    assert_eq!(id(&ids, IdSpace::Texture).id(), ResourceId(2));
    drop(internal);
}

/// A pipeline state's placement keeps its index's low 32 bits, as Metal did on an M4: `2^32 + 5` is
/// 5, `2^32` plain allocation, as 0 and `Any` are; and is refused at 16384 or more after the cut,
/// which ended Metal's process.
#[test]
fn a_pipeline_placement_keeps_its_low_32_bits_below_the_pipeline_limit() {
    use crate::error::{Error, Invalid, Problem};
    use crate::resource_ids::pipeline_placement;
    let at = |id: u64| IdPlacement::At(ResourceId(id));
    let placed = |placement| pipeline_placement("new_compute_pipeline_state", placement);
    assert_eq!(placed(IdPlacement::Any), Ok(IdPlacement::Any));
    assert_eq!(placed(at(0)), Ok(at(0)));
    assert_eq!(placed(at(16383)), Ok(at(16383)));
    assert_eq!(placed(at((1 << 32) + 5)), Ok(at(5)));
    assert_eq!(placed(at((1 << 33) + 7)), Ok(at(7)));
    assert_eq!(placed(at(1 << 32)), Ok(at(0)));
    for (index, kept) in [
        (16384, 16384),
        (65536, 65536),
        (1 << 31, 1 << 31),
        ((1 << 32) - 1, (1 << 32) - 1),
        ((1 << 32) + 16384, 16384),
    ] {
        assert_eq!(
            placed(at(index)),
            Err(Error::Invalid(Invalid::new(
                "new_compute_pipeline_state",
                Problem::ResourceId {
                    id: kept,
                    limit: 16384
                }
            )))
        );
    }
    // A plain allocation's placement: the lowest free ID, as `Any`'s.
    let ids = ids();
    let plain = IdHold::take(&ids, IdSpace::Pipeline, placed(at(1 << 32)).unwrap());
    assert_eq!(plain.id(), ResourceId(1));
}

/// One reservation call's IDs and the plain IDs Metal handed out after it, in runs (first,
/// last), each measured on an M4: textures and sampler states start with a pool of 16, pipeline
/// states of 32. `b*` were predicted before they were measured.
const MEASURED: &[(&str, IdSpace, &[u64], &[(u64, u64)])] = &[
    (
        "o1_0_10",
        IdSpace::Texture,
        &[10],
        &[(11, 15), (1, 9), (16, 41)],
    ),
    (
        "p_4_6",
        IdSpace::Texture,
        &[4, 6],
        &[(5, 5), (1, 3), (7, 26)],
    ),
    (
        "p_3_5",
        IdSpace::Texture,
        &[3, 5],
        &[(4, 4), (1, 2), (6, 26)],
    ),
    (
        "p_2_10",
        IdSpace::Texture,
        &[2, 10],
        &[(1, 1), (11, 15), (3, 9), (16, 42)],
    ),
    (
        "p_7_14",
        IdSpace::Texture,
        &[7, 14],
        &[(15, 15), (1, 6), (8, 13), (16, 42)],
    ),
    (
        "p_20_30",
        IdSpace::Texture,
        &[20, 30],
        &[(31, 31), (21, 29), (1, 19), (32, 74)],
    ),
    (
        "t_10_20_30",
        IdSpace::Texture,
        &[10, 20, 30],
        &[(31, 31), (1, 9), (11, 19), (21, 29), (32, 75)],
    ),
    (
        "b0_d",
        IdSpace::Texture,
        &[40, 6, 44, 12, 100, 101, 102],
        &[
            (41, 43),
            (1, 5),
            (7, 11),
            (103, 127),
            (13, 39),
            (45, 99),
            (128, 307),
        ],
    ),
    (
        "multi_0",
        IdSpace::Texture,
        &[2, 16382],
        &[(1, 1), (16383, 16383), (3, 8)],
    ),
    (
        "o1_1_10",
        IdSpace::Sampler,
        &[10],
        &[(11, 15), (1, 9), (16, 41)],
    ),
    (
        "b1_a",
        IdSpace::Sampler,
        &[3, 11, 25],
        &[(1, 2), (26, 31), (4, 10), (12, 24), (32, 75)],
    ),
    (
        "o1_4_17",
        IdSpace::Pipeline,
        &[17],
        &[(18, 31), (1, 16), (32, 73)],
    ),
    (
        "o1_4_33",
        IdSpace::Pipeline,
        &[33],
        &[(34, 63), (1, 32), (64, 137)],
    ),
    (
        "multi_4_1",
        IdSpace::Pipeline,
        &[16382, 2],
        &[(1, 1), (16383, 16383), (3, 8)],
    ),
];

/// The IDs of `runs`, in order.
fn expand(runs: &[(u64, u64)]) -> Vec<u64> {
    runs.iter()
        .flat_map(|&(first, last)| first..=last)
        .collect()
}

/// Plain IDs after a reservation are Metal's, each case's whole measured sequence.
#[test]
fn plain_ids_after_a_reservation_are_metal_s() {
    for &(case, space, reserved, runs) in MEASURED {
        let ids = ids();
        let reserved: Vec<ResourceId> = reserved.iter().copied().map(ResourceId).collect();
        assert!(ids.lock().unwrap().reserve(space, &reserved), "{case}");
        let want = expand(runs);
        let held: Vec<IdHold> = want.iter().map(|_| id(&ids, space)).collect();
        let got: Vec<u64> = held.iter().map(|held| held.id().0).collect();
        assert_eq!(got, want, "{case}");
    }
}

/// An ID given back joins its free neighbours, so a run of them beats a lone one (2, 3 and 5 of 1
/// to 8 given back are taken as 5, 2, 3, then 9; `c_res10_3_12`: after a reservation of 10 and IDs
/// 11 to 15 and 1 to 3, 3 and 12 given back are taken as 12, then 3 to 9, then 16).
#[test]
fn an_id_given_back_joins_its_free_neighbours() {
    let ids = ids();
    let mut held: Vec<Option<IdHold>> = (0..8).map(|_| Some(id(&ids, IdSpace::Texture))).collect();
    for at in [1, 2, 4] {
        held[at] = None;
    }
    let next: Vec<IdHold> = (0..4).map(|_| id(&ids, IdSpace::Texture)).collect();
    let got: Vec<u64> = next.iter().map(|held| held.id().0).collect();
    assert_eq!(got, [5, 2, 3, 9]);

    let ids = self::ids();
    assert!(ids
        .lock()
        .unwrap()
        .reserve(IdSpace::Texture, &[ResourceId(10)]));
    let mut held: Vec<Option<IdHold>> = (0..8).map(|_| Some(id(&ids, IdSpace::Texture))).collect();
    let got: Vec<u64> = held.iter().flatten().map(|held| held.id().0).collect();
    assert_eq!(got, [11, 12, 13, 14, 15, 1, 2, 3]);
    held[7] = None;
    held[1] = None;
    let next: Vec<IdHold> = (0..9).map(|_| id(&ids, IdSpace::Texture)).collect();
    let got: Vec<u64> = next.iter().map(|held| held.id().0).collect();
    assert_eq!(got, [12, 3, 4, 5, 6, 7, 8, 9, 16]);
}

/// Only a space's first reservation with an ID counts, whether or not the space handed one out
/// since; a call of no ID, or of 0 alone, reserves nothing (16, then 5, keeps 16 alone;
/// `empty_first`; an M3 Max handed out these 17 IDs after it, where the M4's pool handed out 1 to
/// 4 again later).
#[test]
fn only_the_first_reservation_with_an_id_counts() {
    let ids = ids();
    let mut all = ids.lock().unwrap();
    assert!(all.reserve(IdSpace::Texture, &[]));
    assert!(all.reserve(IdSpace::Texture, &[ResourceId(0)]));
    assert!(all.reserve(IdSpace::Texture, &[ResourceId(16)]));
    assert!(!all.reserve(IdSpace::Texture, &[ResourceId(5)]));
    drop(all);
    let held: Vec<IdHold> = (0..17).map(|_| id(&ids, IdSpace::Texture)).collect();
    let got: Vec<u64> = held.iter().map(|held| held.id().0).collect();
    assert_eq!(got, expand(&[(1, 15), (17, 18)]));
}

/// A call of 0 alone leaves the pool made, so a later reservation keeps the pool's first range
/// beside its own and plain allocation hands that range's IDs out twice, as Metal's does (an M3
/// Max, macOS 27.0.1: textures, 0 then 10 gives 11 to 15, 1 to 9, then 1 to 31; 0 then 16, the
/// first range equal to one left, 1 to 15, then 17 on); a lone 0 once the pool started is ignored,
/// as Metal ignores and logs it (5, then 0: 1 to 4, then 6 on).
#[test]
fn a_lone_zero_leaves_the_pool_made() {
    for (reserved, runs) in [
        (10, &[(11, 15), (1, 9), (1, 31)][..]),
        (16, &[(1, 15), (17, 61)][..]),
    ] {
        let ids = ids();
        let mut all = ids.lock().unwrap();
        assert!(all.reserve(IdSpace::Texture, &[ResourceId(0)]));
        assert!(all.reserve(IdSpace::Texture, &[ResourceId(reserved)]));
        drop(all);
        let want = expand(runs);
        let held: Vec<IdHold> = want.iter().map(|_| id(&ids, IdSpace::Texture)).collect();
        let got: Vec<u64> = held.iter().map(|held| held.id().0).collect();
        assert_eq!(got, want, "0, then {reserved}");
    }
    let ids = ids();
    let mut all = ids.lock().unwrap();
    assert!(all.reserve(IdSpace::Texture, &[ResourceId(5)]));
    assert!(!all.reserve(IdSpace::Texture, &[ResourceId(0)]), "late");
    drop(all);
    let held: Vec<IdHold> = (0..8).map(|_| id(&ids, IdSpace::Texture)).collect();
    let got: Vec<u64> = held.iter().map(|held| held.id().0).collect();
    assert_eq!(got, expand(&[(1, 4), (6, 9)]));
}

/// Visible-function tables' first pool is 32, as Metal's (2 to 4 of 1 to 5
/// given back come back before 6, 2 to 6 of 1 to 13 before 14, and 2 to 10 of 1 to 29 after 30
/// and 31).
#[test]
fn function_tables_start_with_a_pool_of_32() {
    for (made, gone, want) in [
        (5, 2..=4, &[2, 3, 4, 6, 7, 8][..]),
        (13, 2..=6, &[2, 3, 4, 5, 6, 14, 15, 16]),
        (29, 2..=10, &[30, 31, 2, 3, 4, 5, 6, 7, 8, 9, 10, 32]),
    ] {
        let ids = ids();
        let mut held: Vec<Option<IdHold>> = (0..made)
            .map(|_| Some(id(&ids, IdSpace::FunctionTable)))
            .collect();
        for given_back in gone {
            held[given_back - 1] = None;
        }
        let next: Vec<IdHold> = want
            .iter()
            .map(|_| id(&ids, IdSpace::FunctionTable))
            .collect();
        let got: Vec<u64> = next.iter().map(|held| held.id().0).collect();
        assert_eq!(got, want);
    }
}

/// Orders Metal's texture pool hands out after the device's first command queue or indirect
/// command buffer and then a reservation, measured on an M4 (plain and validated
/// alike, a queue or a buffer first alike): the reservation's ranges beside the pool's first
/// range, `[1, 16)`, from which Metal hands out IDs a live texture holds a second time. Each case
/// is the reservation and Metal's whole measured order, as runs of IDs.
const MADE_FIRST: &[(&str, &[u64], &[(u64, u64)])] = &[
    ("t.queue.64+65+66.4", &[64, 65, 66], &[(1, 4)]),
    (
        "t.queue.20.70",
        &[20],
        &[(21, 31), (1, 15), (1, 15), (16, 19), (32, 56)],
    ),
    ("t.queue.40.70", &[40], &[(1, 15), (41, 63), (1, 32)]),
    ("t.queue.1.40", &[1], &[(2, 15), (1, 15), (16, 26)]),
    (
        "t.queue.10.40",
        &[10],
        &[(11, 15), (1, 9), (1, 15), (16, 26)],
    ),
    ("t.queue.15.40", &[15], &[(1, 14), (1, 15), (16, 26)]),
    ("t.queue.16.50", &[16], &[(1, 15), (17, 51)]),
    (
        "t.queue.17.50",
        &[17],
        &[(18, 31), (1, 15), (1, 16), (32, 36)],
    ),
    ("t.queue.32.70", &[32], &[(1, 15), (1, 31), (33, 56)]),
    ("t.queue.33.40", &[33], &[(1, 15), (34, 58)]),
    ("t.queue.1000.40", &[1000], &[(1, 15), (1001, 1023), (1, 2)]),
    ("t.queue.20000.20", &[20000], &[(1, 15), (20001, 20005)]),
    ("t.queue.5+40.40", &[5, 40], &[(1, 4), (1, 15), (41, 61)]),
];

/// A texture pool the device made before its reservation (`ResourceIds::make_texture_pool`, a
/// command queue's or an indirect command buffer's) hands IDs out in Metal's whole order, handing
/// out a second time each ID Metal hands out again while a texture holds it; without the
/// reservation, or made after it, the order is the plain one.
#[test]
fn a_texture_pool_made_before_its_reservation_hands_out_metal_s_ids() {
    for &(case, reserved, runs) in MADE_FIRST {
        let ids = ids();
        ids.lock().unwrap().make_texture_pool();
        let reserved: Vec<ResourceId> = reserved.iter().copied().map(ResourceId).collect();
        assert!(
            ids.lock().unwrap().reserve(IdSpace::Texture, &reserved),
            "{case}"
        );
        let metal = expand(runs);
        let held: Vec<IdHold> = metal.iter().map(|_| id(&ids, IdSpace::Texture)).collect();
        let got: Vec<u64> = held.iter().map(|held| held.id().0).collect();
        assert_eq!(got, metal, "{case}");
    }
    // Made, but no reservation: the plain order. Made after a reservation: as before.
    let ids = ids();
    ids.lock().unwrap().make_texture_pool();
    let held: Vec<IdHold> = (0..4).map(|_| id(&ids, IdSpace::Texture)).collect();
    let got: Vec<u64> = held.iter().map(|held| held.id().0).collect();
    assert_eq!(got, [1, 2, 3, 4]);
    let ids = self::ids();
    assert!(ids
        .lock()
        .unwrap()
        .reserve(IdSpace::Texture, &[64, 65, 66].map(ResourceId)));
    ids.lock().unwrap().make_texture_pool();
    let held: Vec<IdHold> = (0..4).map(|_| id(&ids, IdSpace::Texture)).collect();
    let got: Vec<u64> = held.iter().map(|held| held.id().0).collect();
    assert_eq!(got, [67, 68, 69, 70]);
}

/// IDs given back in a made pool join their neighbours as before, and plain allocation hands out
/// the IDs live textures hold where Metal does (an M4: after a queue, a
/// reservation of 64 to 66 and textures 1 to 3, 2 given back is taken as 2, 4, 5; after a
/// reservation of 40 and textures 1 to 15 and 41 to 45, 3 as 3, 46 to 63, then 1, which a texture
/// holds; after a reservation of 20 and textures 21 to 31 and 1 to 15, 5 as 5, then 1 to 7, which
/// textures hold).
#[test]
fn ids_given_back_in_a_made_pool_follow_metal_s_order() {
    for (reserved, made, gone, metal) in [
        (&[64, 65, 66][..], 3, 2, &[2, 4, 5][..]),
        (
            &[40],
            20,
            3,
            &[
                3, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 1,
            ],
        ),
        (&[20], 26, 5, &[5, 1, 2, 3, 4, 5, 6, 7]),
    ] {
        let ids = ids();
        ids.lock().unwrap().make_texture_pool();
        let reserved: Vec<ResourceId> = reserved.iter().copied().map(ResourceId).collect();
        assert!(ids.lock().unwrap().reserve(IdSpace::Texture, &reserved));
        let mut held: Vec<Option<IdHold>> = (0..made)
            .map(|_| Some(id(&ids, IdSpace::Texture)))
            .collect();
        let at = held
            .iter()
            .position(|held| held.as_ref().is_some_and(|h| h.id().0 == gone))
            .expect("a texture holds it");
        held[at] = None;
        let next: Vec<IdHold> = metal.iter().map(|_| id(&ids, IdSpace::Texture)).collect();
        let got: Vec<u64> = next.iter().map(|held| held.id().0).collect();
        assert_eq!(got, metal);
    }
}

/// Orders Metal's texture pool hands out once IDs it handed out twice are given back, measured on
/// an M4 (plain, each case twice alike; the model was fitted to these). Each case:
/// whether a command queue comes first, the reservation, the plain textures made, the positions
/// (in the order made) of those then released, in order, as runs, and Metal's next IDs, as runs.
/// After a queue and 20 reserved, positions 11 to 25 hold 1 to 15 and 26 to 40 hold them again.
const GIVEN_BACK_TWICE: &[(bool, &[u64], usize, Runs<usize>, Runs<u64>)] = &[
    (true, &[20], 26, &[(15, 15)], &[(5, 5), (1, 7)]),
    (true, &[20], 41, &[(15, 15)], &[(5, 5), (16, 19), (32, 34)]),
    (true, &[20], 41, &[(30, 30)], &[(5, 5), (16, 19), (32, 34)]),
    // Both holders of 5, in either order: 5 once.
    (
        true,
        &[20],
        41,
        &[(15, 15), (30, 30)],
        &[(5, 5), (16, 19), (32, 34)],
    ),
    (
        true,
        &[20],
        41,
        &[(30, 30), (15, 15)],
        &[(5, 5), (16, 19), (32, 34)],
    ),
    // 5, 5, 6, 6: 6, then 5 and 6; 5, 6, 5, 6: 5 and 6 twice.
    (
        true,
        &[20],
        41,
        &[(15, 15), (30, 30), (16, 16), (31, 31)],
        &[(6, 6), (5, 6), (16, 19), (32, 34)],
    ),
    (true, &[20], 41, &[(15, 16)], &[(5, 6), (16, 19), (32, 33)]),
    (true, &[20], 41, &[(30, 31)], &[(5, 6), (16, 19), (32, 33)]),
    (
        true,
        &[20],
        41,
        &[(15, 16), (30, 31)],
        &[(5, 6), (5, 6), (16, 19), (32, 33)],
    ),
    // 1 to 15 given back once, then twice.
    (true, &[20], 41, &[(11, 25)], &[(1, 19), (32, 32)]),
    (true, &[20], 41, &[(11, 40)], &[(1, 15), (1, 19), (32, 37)]),
    (
        true,
        &[20],
        45,
        &[(15, 15), (30, 30), (44, 44)],
        &[(5, 5), (19, 19), (32, 37)],
    ),
    (
        true,
        &[10],
        30,
        &[(11, 11), (25, 25)],
        &[(7, 7), (12, 12), (17, 22)],
    ),
    (
        true,
        &[10],
        30,
        &[(11, 11), (25, 25), (12, 12), (26, 26)],
        &[(7, 8), (12, 13), (17, 20)],
    ),
    (false, &[], 6, &[(2, 2)], &[(3, 3), (7, 8)]),
];

/// Runs of values, each `(first, last)`.
type Runs<T> = &'static [(T, T)];

/// IDs a made pool handed out twice come back as Metal's pool gives them back
/// ([`GIVEN_BACK_TWICE`]): an ID given back with nothing free beside it is one range however
/// often, and a range a join grows to equal another is a second.
#[test]
fn ids_held_twice_come_back_as_metal_s_pool_gives_them() {
    for &(queue, reserved, made, released, then) in GIVEN_BACK_TWICE {
        let ids = ids();
        if queue {
            ids.lock().unwrap().make_texture_pool();
        }
        let reserved: Vec<ResourceId> = reserved.iter().copied().map(ResourceId).collect();
        if !reserved.is_empty() {
            assert!(ids.lock().unwrap().reserve(IdSpace::Texture, &reserved));
        }
        let mut held: Vec<Option<IdHold>> = (0..made)
            .map(|_| Some(id(&ids, IdSpace::Texture)))
            .collect();
        for &(first, last) in released {
            for slot in &mut held[first..=last] {
                assert!(slot.is_some(), "{released:?}: released twice");
                *slot = None;
            }
        }
        let metal = expand(then);
        let next: Vec<IdHold> = metal.iter().map(|_| id(&ids, IdSpace::Texture)).collect();
        let got: Vec<u64> = next.iter().map(|held| held.id().0).collect();
        assert_eq!(
            got, metal,
            "{reserved:?}, {made} made, {released:?} given back"
        );
    }
}

/// A set-1 image array reaches no further than the handle table (`HANDLE_TABLE_CAPACITY`, 65536
/// entries, link's cap on every heap array) on every committed profile, so a texture ID past it is
/// refused as not implemented rather than taken where no array holds it: MoltenVK's and lavapipe's
/// update-after-bind image limits of 1,000,000 and more, and RADV's, ANV's and NVIDIA's, read
/// alone, reached past it. Where a device limit is the lesser (the desktop baseline's 128 sampled
/// and 8 storage images a stage), the refusal names it.
#[test]
fn texture_arrays_reach_no_further_than_the_handle_table() {
    let reach = u64::from(HANDLE_TABLE_CAPACITY);
    let table = (reach, ArrayBound::HandleTable);
    let refused = |id: u64, array: (u64, ArrayBound)| match check_capacity(id, array) {
        Ok(()) => None,
        Err(Error::Unsupported(u)) => Some(u),
        Err(other) => panic!("{other:?}"),
    };
    let mut arrays = BTreeMap::new();
    for (stem, desc) in committed_descriptions() {
        let (enabled, _) = created(&stem, &desc);
        let compile = Device::new(enabled, DevicePolicy::UNTRUSTED).unwrap();
        for array in texture_arrays(compile.profile()) {
            assert!(array.0 <= reach, "{stem}: {array:?}");
            if array.0 > 0 {
                assert_eq!(refused(array.0 - 1, array), None, "{stem}: {array:?}");
            }
            let past = refused(array.0, array).unwrap();
            let want = match array.1 {
                ArrayBound::HandleTable => {
                    Unsupported::not_implemented(Feature::ArgumentBufferTextures)
                }
                ArrayBound::Limit(limit) => Unsupported::needs(
                    Feature::ArgumentBufferTextures,
                    Capability::Limit {
                        limit,
                        required: array.0 + 1,
                        available: array.0,
                    },
                ),
            };
            assert_eq!(past, want, "{stem}");
        }
        arrays.insert(stem, texture_arrays(compile.profile()));
    }
    let limit = |n: u64, limit: VkLimit| (n, ArrayBound::Limit(limit));
    for (stem, want) in [
        ("moltenvk-m3", [table, table]),
        ("lavapipe", [table, table]),
        ("amd-780m-radv", [table, table]),
        ("intel-irisxe-rplp-anv", [table, table]),
        ("nvidia-rtx5080-laptop", [table, table]),
        (
            "lunarg-desktop-baseline-2023",
            [
                limit(
                    128,
                    VkLimit::MaxPerStageDescriptorUpdateAfterBindSampledImages,
                ),
                limit(
                    8,
                    VkLimit::MaxPerStageDescriptorUpdateAfterBindStorageImages,
                ),
            ],
        ),
    ] {
        assert_eq!(arrays.get(stem), Some(&want), "{stem}");
    }
}
