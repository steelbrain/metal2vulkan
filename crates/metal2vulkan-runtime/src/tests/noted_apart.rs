//! The last command an unordered encoder noted apart (`noted_apart.rs`).

use super::super::binding::{ReachingPrefix, ResidentPrefix};
use super::super::noted_apart::{Moved, NotedApart, NotedInto};
use super::super::resource::ResourceKey;
use super::super::tracker::{Access, Use, UseRange};
use ash::vk;
use std::collections::BTreeSet;

/// A fragment shader's read of `bytes` of the buffer `key`.
fn read(key: ResourceKey, bytes: std::ops::Range<u64>) -> Use<ResourceKey> {
    let access = Access::new(
        vk::PipelineStageFlags2::FRAGMENT_SHADER,
        vk::AccessFlags2::SHADER_STORAGE_READ,
    );
    Use::reads(key, true, UseRange::Bytes(bytes), access)
}

/// A command reaching a buffer through the resident set alone is the last one noted while its
/// uses, the keys it so reaches, its stages and the state it is noted into are the last's: not
/// with another key reached so, another use, other stages, another core, a part or held image
/// more, a device with buffers over host pages where it had none, or after the host aliases
/// changed (their version).
#[test]
fn a_command_reaching_the_same_resident_keys_is_noted_already() {
    let (argument, reached, other) = (ResourceKey::new(), ResourceKey::new(), ResourceKey::new());
    let uses = [read(argument, 0..64), read(reached, 0..4096)];
    let resident = BTreeSet::from([reached]);
    let stages = vk::PipelineStageFlags2::FRAGMENT_SHADER;
    let into = NotedInto {
        core: 7,
        parts: 1,
        held: 0,
        rebinds: false,
        aliases: 3,
    };
    let mut noted = NotedApart::default();
    assert!(!noted.holds(stages, (&uses, &resident), into));
    noted.note_fresh(stages, (&uses, &resident), into, |_| ());
    assert!(noted.holds(stages, (&uses, &resident), into));
    // The same uses, reaching no key or another through the resident set alone.
    assert!(!noted.holds(stages, (&uses, &BTreeSet::new()), into));
    assert!(!noted.holds(stages, (&uses, &BTreeSet::from([argument])), into));
    assert!(!noted.holds(stages, (&uses, &BTreeSet::from([reached, other])), into));
    // Another use, other stages.
    let more = [read(argument, 0..64), read(reached, 0..8192)];
    assert!(!noted.holds(stages, (&more, &resident), into));
    let vertex = vk::PipelineStageFlags2::VERTEX_SHADER;
    assert!(!noted.holds(vertex, (&uses, &resident), into));
    // Another state noted into.
    let changed = [
        NotedInto { core: 8, ..into },
        NotedInto { parts: 2, ..into },
        NotedInto { held: 1, ..into },
        NotedInto {
            rebinds: true,
            ..into
        },
        NotedInto { aliases: 4, ..into },
    ];
    for other in changed {
        assert!(!noted.holds(stages, (&uses, &resident), other));
    }
    // Noted again with another resident key: that one holds, the first no longer.
    let others = BTreeSet::from([other]);
    noted.note_fresh(stages, (&uses, &others), into, |_| ());
    assert!(noted.holds(stages, (&uses, &others), into));
    assert!(!noted.holds(stages, (&uses, &resident), into));
}

/// A draw whose uses begin with those of what its encoder made resident by identity
/// (`binding::ReachingPrefix`) is noted already where the last one began with the same identities
/// and its uses past them are the last's, and records fresh only those of its uses past them that
/// differ from the last's: not with another identity of them, or of the keys they reach through
/// the resident set alone, or after a command without them (every use fresh then). Builds with
/// debug assertions check each answer against the one the whole uses give.
#[test]
fn a_draw_beginning_with_the_same_resident_uses_compares_only_its_others() {
    let keys: [ResourceKey; 3] = std::array::from_fn(|_| ResourceKey::new());
    let own = |end| read(keys[2], 0..end);
    let uses = |end| [read(keys[0], 0..4096), read(keys[1], 0..4096), own(end)];
    let resident = BTreeSet::from([keys[0], keys[1]]);
    let stages = vk::PipelineStageFlags2::FRAGMENT_SHADER;
    let prefix = ReachingPrefix {
        prefix: ResidentPrefix { id: 5, len: 2 },
        resident: 6,
        stages,
    };
    let into = NotedInto {
        core: 7,
        parts: 1,
        held: 0,
        rebinds: false,
        aliases: 3,
    };
    let mut noted = NotedApart::default();
    let (first, second) = (uses(64), uses(128));
    let draw = |noted: &mut NotedApart, (uses, prefix): (&[Use<ResourceKey>], ReachingPrefix)| {
        noted.note_draw_fresh(stages, (uses, &resident, prefix), into, <[_]>::to_vec)
    };
    // The first draw, noted into no state: all of its uses.
    assert_eq!(draw(&mut noted, (&first, prefix)), first);
    assert!(noted.holds_draw(stages, (&first, &resident, prefix), into));
    assert!(!noted.holds_draw(stages, (&second, &resident, prefix), into));
    // Another identity of the uses, or of the keys they reach through the resident set alone.
    let others = [
        ReachingPrefix {
            prefix: ResidentPrefix { id: 9, len: 2 },
            ..prefix
        },
        ReachingPrefix {
            resident: 9,
            ..prefix
        },
    ];
    for other in others {
        assert!(!noted.holds_draw(stages, (&first, &resident, other), into));
    }
    assert!(!noted.holds(stages, (&first, &resident), into));
    assert_eq!(draw(&mut noted, (&second, prefix)), [own(128)]);
    assert!(noted.holds_draw(stages, (&second, &resident, prefix), into));
    assert_eq!(draw(&mut noted, (&first, others[0])), first);
    // After a command without them, every use is fresh.
    noted.note_fresh(stages, (&first, &resident), into, |_| ());
    assert!(!noted.holds_draw(stages, (&first, &resident, prefix), into));
    assert_eq!(draw(&mut noted, (&second, prefix)), second);
}

/// A command noted into the state the last one was noted into records the uses that differ from
/// the last one's at their position, and is kept whole though only those change in place: the
/// next command holds by its uses as by a copy of them, where some changed, more came or fewer.
#[test]
fn a_command_noted_after_another_keeps_its_uses_changed_in_place() {
    let keys: [ResourceKey; 4] = std::array::from_fn(|_| ResourceKey::new());
    let stages = vk::PipelineStageFlags2::FRAGMENT_SHADER;
    let into = NotedInto {
        core: 7,
        parts: 1,
        held: 0,
        rebinds: false,
        aliases: 3,
    };
    let resident = BTreeSet::new();
    let commands = [
        vec![read(keys[0], 0..64), read(keys[1], 0..64)],
        vec![read(keys[0], 0..64), read(keys[1], 64..128)],
        vec![
            read(keys[0], 0..64),
            read(keys[1], 64..128),
            read(keys[2], 0..16),
            read(keys[3], 0..16),
        ],
        vec![
            read(keys[3], 0..16),
            read(keys[1], 64..128),
            read(keys[2], 0..16),
        ],
        vec![read(keys[3], 0..16)],
        vec![],
        vec![read(keys[2], 0..8), read(keys[0], 0..64)],
    ];
    let mut noted = NotedApart::default();
    let mut last: Vec<Use<ResourceKey>> = Vec::new();
    for uses in &commands {
        let fresh = noted.note_fresh(stages, (uses, &resident), into, <[_]>::to_vec);
        let differ = (uses.iter().enumerate()).filter(|(at, used)| last.get(*at) != Some(*used));
        // The first command, noted into no state, records all of its uses.
        assert_eq!(
            fresh,
            differ.map(|(_, used)| used.clone()).collect::<Vec<_>>()
        );
        assert!(noted.holds(stages, (uses, &resident), into));
        for other in commands.iter().filter(|other| *other != uses) {
            assert!(!noted.holds(stages, (other, &resident), into));
        }
        last.clone_from(uses);
    }
}

/// A draw's uses of what its encoder made resident are touched already where the last draw that
/// began with some, in the same encoder while the device's buffers over host pages stand as they
/// did, began with the same: not with another identity, in another encoder, or once the device
/// has such buffers where it had none.
#[test]
fn a_draw_s_resident_uses_are_touched_once_by_their_identity() {
    let prefix = ResidentPrefix { id: 5, len: 2 };
    let mut noted = NotedApart::default();
    assert_eq!(noted.touched(prefix, (7, false)), 0);
    assert_eq!(noted.touched(prefix, (7, false)), 2);
    let other = ResidentPrefix { id: 6, len: 2 };
    for (prefix, state) in [
        (other, (7, false)),
        (prefix, (8, false)),
        (prefix, (8, true)),
    ] {
        assert_eq!(noted.touched(prefix, state), 0);
    }
}

/// A command whose use differs from the last one's at its place only by bytes inside that one's
/// range, of the same resource and access types, records nothing for it (`Use::covers`), as a
/// draw whose buffer offset moves along a buffer reads from a later offset to its end, and its
/// own use is kept: the command noted after it compares against that, and a range past it is
/// recorded. A moved command (`NotedApart::note_moved`) takes it so too, and is noted as before
/// only where every use is the last one's.
#[test]
fn a_use_inside_the_last_one_s_range_is_noted_already() {
    let (buffer, other) = (ResourceKey::new(), ResourceKey::new());
    let stages = vk::PipelineStageFlags2::FRAGMENT_SHADER;
    let into = NotedInto {
        core: 7,
        parts: 1,
        held: 0,
        rebinds: false,
        aliases: 3,
    };
    let resident = BTreeSet::new();
    let at = |offset| [read(other, 0..16), read(buffer, offset..4096)];
    let mut noted = NotedApart::default();
    let fresh = |noted: &mut NotedApart, uses: &[Use<ResourceKey>]| {
        noted.note_fresh(stages, (uses, &resident), into, <[_]>::to_vec)
    };
    assert_eq!(fresh(&mut noted, &at(0)), at(0));
    assert_eq!(fresh(&mut noted, &at(256)), []);
    assert_eq!(noted.covered(), 1);
    // Kept: the command's own uses, which the next command compares with.
    assert!(noted.holds(stages, (&at(256), &resident), into));
    assert!(!noted.holds(stages, (&at(0), &resident), into));
    assert_eq!(fresh(&mut noted, &at(512)), []);
    assert_eq!(noted.covered(), 1);
    // Bytes past it, or another access of them, are recorded.
    assert_eq!(fresh(&mut noted, &at(256)), [at(256)[1].clone()]);
    assert_eq!(noted.covered(), 0);
    let wider = [read(other, 0..16), read(buffer, 0..8192)];
    assert_eq!(fresh(&mut noted, &wider), [wider[1].clone()]);
    let written = Use::writes(buffer, true, UseRange::Bytes(512..4096), wider[1].read);
    let mixed = [read(other, 0..16), written];
    assert_eq!(fresh(&mut noted, &mixed), [mixed[1].clone()]);
    // Moved: covered, as before, and recorded past it.
    let mut moved = NotedApart::default();
    fresh(&mut moved, &at(0));
    assert!(moved.moves((&at(256), &resident), into));
    let note = |moved: &mut NotedApart, uses: &[Use<ResourceKey>]| {
        let record = |_, fresh: &[Use<ResourceKey>]| fresh.to_vec();
        match moved.note_moved((uses, &resident), &[1], record) {
            Moved::Recorded(fresh, covered) => (Some(fresh), covered, false),
            Moved::Covered(covered) => (None, covered, false),
            Moved::AsBefore => (None, 0, true),
        }
    };
    assert_eq!(note(&mut moved, &at(256)), (None, 1, false));
    assert!(moved.holds(stages, (&at(256), &resident), into));
    assert_eq!(note(&mut moved, &at(256)), (None, 0, true));
    assert_eq!(
        note(&mut moved, &at(0)),
        (Some(vec![at(0)[1].clone()]), 0, false)
    );
    let past = [read(other, 0..16), read(buffer, 0..8192)];
    assert_eq!(
        note(&mut moved, &past),
        (Some(vec![past[1].clone()]), 0, false)
    );
    assert!(moved.holds(stages, (&past, &resident), into));
}
