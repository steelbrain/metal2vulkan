//! The last command an unordered encoder noted apart (`noted_apart.rs`).

use super::super::binding::{ReachingPrefix, ResidentPrefix};
use super::super::noted_apart::{NotedApart, NotedInto};
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
    noted.note(stages, (&uses, &resident), into);
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
    noted.note(stages, (&uses, &others), into);
    assert!(noted.holds(stages, (&uses, &others), into));
    assert!(!noted.holds(stages, (&uses, &resident), into));
}

/// A draw whose uses begin with those of what its encoder made resident by identity
/// (`binding::ReachingPrefix`) is noted already where the last one began with the same identities
/// and its uses past them are the last's, and finds fresh only those of its uses past them that
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
    noted.note_draw(stages, (&first, &resident, prefix), into);
    assert!(noted.holds_draw(stages, (&first, &resident, prefix), into));
    assert!(!noted.holds_draw(stages, (&second, &resident, prefix), into));
    assert_eq!(noted.fresh_draw((&second, prefix.prefix), into), [own(128)]);
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
    assert_eq!(noted.fresh_draw((&second, others[0].prefix), into), second);
    // After a command without them, every use is fresh.
    noted.note(stages, (&first, &resident), into);
    assert!(!noted.holds_draw(stages, (&first, &resident, prefix), into));
    assert_eq!(noted.fresh_draw((&second, prefix.prefix), into), second);
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
