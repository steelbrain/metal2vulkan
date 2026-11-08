//! The last command an unordered encoder noted apart (`EncoderCore::note_reaching`: a render
//! encoder's draw), kept so a command noted with the same uses notes nothing again.
//!
//! Noting a command records its uses in the encoder's current part (`EncoderUses::record`: each
//! tracked use unioned into the one of its resource and range, the part's stages widened) and
//! touches their keys (`EncoderCore::touch_reaching`: each kept among the encoder's keys, each
//! buffer over host pages held, each key the resident set does not reach named). Each step only
//! adds, and adding an equal use, key or name again changes nothing. So a command is noted
//! already when it matches the last noted command in its uses, the keys of them it reaches
//! through the resident set alone, its stages, and the host aliases' version
//! (`HostAliases::version`: while it stands, the uses recorded, those with the host aliases' uses
//! added by `HostAliases::augment`, are the same for the same uses, so a command noted already
//! does not work them out). It must also be noted while the encoder records into the same part,
//! holds the same images in their attachment layout, and the device's buffers over host pages
//! are there as they were (`Rebinds::any`). Its noting is skipped
//! (`WorkCounts::commands_noted_as_before`); every other command is noted as before and kept.
//! The encoder's parts and held images only grow, so their counts tell a change apart.
//!
//! A command noted into the same state as the last one, but with some uses changed, records
//! only the uses that differ from the last one's at their position (`NotedApart::note_fresh`):
//! each use equal to the last command's at its place, and so the host aliases' uses beside it, is
//! in the part already, and recording it again changes nothing. Its stages widen the part as
//! before, and the uses kept for the next command change at those positions alone.
//!
//! A use of bytes the kept use at its place covers (`Use::covers`: the same resource, tracking
//! and access types, over a range holding its own) is in the part already too: the part keeps
//! uses by the bytes they reach, and every answer it gives by overlap holds them, as do the host
//! aliases' uses beside it (each a range of another resource over the same bytes) and the
//! command buffer's notes of them. It is not recorded (`WorkCounts::uses_noted_covered`), as a
//! draw whose buffer offset moves along a buffer reads from a later offset to its end, and it is
//! kept in place of the covering one: the kept uses stay the last command's, each of them in the
//! part, inside a use recorded before where it was not recorded itself.
//!
//! A draw's uses may begin with those of what its encoder made resident, which draws keep from
//! one to the next and name by identity (`binding::ReachingPrefix`, with the identity of the keys
//! they reach through the resident set alone): such a draw is noted through
//! `EncoderCore::note_draw`, which keeps their identity in place of them and of those keys, and a
//! draw noted after it with the same identities compares, records and touches only its uses past
//! them (`WorkCounts::resident_uses_noted` counts the ones a draw walked). Those uses are equal
//! at each place, and the keys equal, wherever the identities are; the keys a draw touched stay
//! kept, and named, while the encoder is open. A command without them is noted as above. Builds
//! with debug assertions keep the whole uses too, and check each answer against theirs.
//!
//! A render encoder keeps it with its bind state (`live_bind.rs`), which the next render encoder
//! starts from, so its storage is kept from one encoder to the next and a draw allocates none:
//! it names the encoder core that noted it (`EncoderCore::id`, no other core's), and another
//! encoder's command is never taken for one it noted.

use ash::vk;
use std::collections::BTreeSet;

use super::binding::{ReachingPrefix, ResidentPrefix};
use super::resource::ResourceKey;
use super::tracker::Use;

/// What the last command noted apart was noted with (the module docs).
#[derive(Clone, Default, PartialEq)]
pub(super) struct NotedApart {
    /// The core of the encoder that noted it and kept it, if any.
    core: Option<u64>,
    stages: vk::PipelineStageFlags2,
    /// The identities of the uses of what its encoder made resident at their start and of the
    /// keys it reached through the resident set alone, where it had them
    /// ([`note_draw_fresh`](NotedApart::note_draw_fresh)): then `uses` holds its uses past them, and
    /// `resident` nothing.
    prefix: Option<ReachingPrefix>,
    /// Its uses as given, and the keys of them it reached through the resident set alone, in
    /// order.
    uses: Vec<Use<ResourceKey>>,
    resident: Vec<ResourceKey>,
    /// The encoder's parts and held images then.
    parts: usize,
    held: usize,
    /// Whether the device had a buffer over host pages then.
    rebinds: bool,
    /// The host aliases' version then.
    aliases: u64,
    /// The uses of the command being noted that differ from the last one's, and their
    /// positions ([`note_fresh`](NotedApart::note_fresh)), kept so their storage is too.
    fresh: Vec<Use<ResourceKey>>,
    fresh_at: Vec<usize>,
    /// How many of the uses of the command being noted the kept use at their position covered
    /// ([`covered`](NotedApart::covered)).
    covered: usize,
    /// The uses of what the encoder made resident the last draw that began with some touched
    /// (`EncoderCore::note_draw`), by identity, with its encoder's core and whether the device
    /// had a buffer over host pages then: while those stand, a draw that begins with the same
    /// touched them already.
    touched: Option<(u64, ResidentPrefix, bool)>,
    /// Its whole uses and resident keys, which builds with debug assertions answer by too.
    #[cfg(debug_assertions)]
    whole: (Vec<Use<ResourceKey>>, Vec<ResourceKey>),
}

/// The encoder's state a command is noted into: its core, its parts and held images (counts),
/// whether the device has a buffer over host pages, and the host aliases' version
/// (`HostAliases::version`).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct NotedInto {
    pub(super) core: u64,
    pub(super) parts: usize,
    pub(super) held: usize,
    pub(super) rebinds: bool,
    pub(super) aliases: u64,
}

/// What [`NotedApart::note_moved`] found of a command's moved uses: some the kept uses neither
/// equal nor cover, recorded (with what recording them answered, and how many were covered);
/// none but some covered, noted already; or every use the last command's, noted as before.
pub(super) enum Moved<T> {
    Recorded(T, usize),
    Covered(usize),
    AsBefore,
}

/// Every field but the count of the last command's covered uses, its stages by their raw value
/// (ash's flags print nothing).
impl std::fmt::Debug for NotedApart {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let NotedApart {
            core,
            stages,
            prefix,
            uses,
            resident,
            parts,
            held,
            rebinds,
            aliases,
            fresh,
            fresh_at,
            covered: _,
            touched,
            #[cfg(debug_assertions)]
                whole: _,
        } = self;
        (f.debug_struct("NotedApart"))
            .field("core", core)
            .field("stages", &stages.as_raw())
            .field(
                "prefix",
                &prefix.map(|p| (p.prefix, p.resident, p.stages.as_raw())),
            )
            .field("uses", uses)
            .field("resident", resident)
            .field("parts", parts)
            .field("held", held)
            .field("rebinds", rebinds)
            .field("aliases", aliases)
            .field("fresh", fresh)
            .field("fresh_at", fresh_at)
            .field("touched", touched)
            .finish()
    }
}

impl NotedApart {
    /// Whether a command of `stages` with `uses`, of which it reaches the keys `resident`
    /// through the resident set alone, noted into `into`, is the last one noted (the module
    /// docs).
    pub(super) fn holds(
        &self,
        stages: vk::PipelineStageFlags2,
        (uses, resident): (&[Use<ResourceKey>], &BTreeSet<ResourceKey>),
        into: NotedInto,
    ) -> bool {
        self.core == Some(into.core)
            && self.stages == stages
            && self.into() == into
            && self.prefix.is_none()
            && *self.uses == *uses
            && self.resident.iter().eq(resident)
    }

    /// [`holds`](Self::holds) of a draw whose `uses` begin with those of what its encoder made
    /// resident of the identities `prefix`: by those, and its uses past them.
    pub(super) fn holds_draw(
        &self,
        stages: vk::PipelineStageFlags2,
        (uses, resident, prefix): (&[Use<ResourceKey>], &BTreeSet<ResourceKey>, ReachingPrefix),
        into: NotedInto,
    ) -> bool {
        let holds = self.core == Some(into.core)
            && self.stages == stages
            && self.into() == into
            && self.prefix == Some(prefix)
            && *self.uses == uses[prefix.prefix.len..];
        // What the whole uses answer, but where the identities changed with the uses the same.
        #[cfg(debug_assertions)]
        {
            let (whole, keys) = &self.whole;
            let as_whole = self.core == Some(into.core)
                && self.stages == stages
                && self.into() == into
                && **whole == *uses
                && keys.iter().eq(resident);
            assert!(
                holds == as_whole || (as_whole && self.prefix != Some(prefix)),
                "a draw noted apart held by its uses' identity other than by its uses"
            );
        }
        #[cfg(not(debug_assertions))]
        let _ = resident;
        holds
    }

    /// Whether a command with `uses`, of which it reaches the keys `resident` through the
    /// resident set alone, noted into `into`, may be noted as the last one moved
    /// ([`note_moved`](Self::note_moved)): the last one was noted into `into`, with no uses of
    /// what its encoder made resident, as many uses and the same keys reached so. Its caller
    /// knows the positions where its uses may differ from the last one's, each of the same key
    /// and access types (a draw whose calls moved buffer offsets alone:
    /// `render/draw_again/rebound.rs`).
    pub(super) fn moves(
        &self,
        (uses, resident): (&[Use<ResourceKey>], &BTreeSet<ResourceKey>),
        into: NotedInto,
    ) -> bool {
        self.core == Some(into.core)
            && self.into() == into
            && self.prefix.is_none()
            && self.uses.len() == uses.len()
            && self.resident.iter().eq(resident)
    }

    /// Note the command [`moves`](Self::moves) answers for, whose `uses` (reaching the keys
    /// `resident` through the resident set alone) differ from the last one's at the positions
    /// `at` alone: `record` records those of them that the kept use at their position neither
    /// equals nor covers, which the encoder's current part may lack (the module docs), and every
    /// one of them that differs is kept in place. What it found ([`Moved`]). Its stages are the
    /// last one's, its uses' access types being theirs.
    pub(super) fn note_moved<T>(
        &mut self,
        (uses, resident): (&[Use<ResourceKey>], &BTreeSet<ResourceKey>),
        at: &[usize],
        record: impl FnOnce(vk::PipelineStageFlags2, &[Use<ResourceKey>]) -> T,
    ) -> Moved<T> {
        self.fresh.clear();
        self.fresh_at.clear();
        let mut covered = 0;
        for &at in at {
            let (kept, used) = (&mut self.uses[at], &uses[at]);
            if kept == used {
                continue;
            }
            // Noted already, and kept as the command's own (the module docs).
            if kept.covers(used) {
                kept.clone_from(used);
                covered += 1;
                continue;
            }
            self.fresh.push(used.clone());
            self.fresh_at.push(at);
        }
        #[cfg(debug_assertions)]
        {
            let moved = |place: &usize| at.contains(place);
            let kept = self.uses.iter().enumerate().zip(uses);
            assert!(
                kept.filter(|((place, _), _)| !moved(place))
                    .all(|((_, kept), used)| kept == used),
                "a command noted moved differs from the last one where it did not move"
            );
            let stages = (uses.iter()).fold(vk::PipelineStageFlags2::empty(), |all, used| {
                all | used.read.stages | used.write.stages
            });
            assert!(
                stages == self.stages,
                "a command noted moved runs in other stages than the last one"
            );
        }
        #[cfg(debug_assertions)]
        self.note_whole((uses, resident));
        #[cfg(not(debug_assertions))]
        let _ = resident;
        if self.fresh.is_empty() {
            // Noted already, as the last one (`holds`) where every use is its.
            return match covered {
                0 => Moved::AsBefore,
                covered => Moved::Covered(covered),
            };
        }
        let answer = record(self.stages, &self.fresh);
        for (&at, used) in self.fresh_at.iter().zip(&self.fresh) {
            self.uses[at].clone_from(used);
        }
        Moved::Recorded(answer, covered)
    }

    /// How many uses the last [`note_fresh`](Self::note_fresh) or
    /// [`note_draw_fresh`](Self::note_draw_fresh) found covered by the kept use at their
    /// position, and so noted already (the module docs).
    pub(super) fn covered(&self) -> usize {
        self.covered
    }

    /// Whether the last command was noted into `into`: then a command noted with its stages and
    /// uses, reaching the same keys through the resident set alone, is noted already
    /// ([`holds`](Self::holds)).
    pub(super) fn noted_into(&self, into: NotedInto) -> bool {
        self.core == Some(into.core) && self.into() == into
    }

    /// The last command's uses, as given, where it began with no use of what its encoder made
    /// resident: a draw kept to draw again, which reads through no root or handle table
    /// (`render.rs`'s plain draw), so whose plan lists none.
    pub(super) fn uses(&self) -> &[Use<ResourceKey>] {
        debug_assert!(
            self.prefix.is_none(),
            "a draw kept to draw again began with uses of what its encoder made resident"
        );
        &self.uses
    }

    /// Note the command of `stages` with `uses`, of which it reaches the keys `resident` through
    /// the resident set alone, into `into`: `record` records its uses that the encoder's current
    /// part may lack (all of them, unless the last command noted was noted into `into`, then
    /// those that differ from its use at the same position: the module docs), and it is kept as
    /// the last command, its kept uses changed at those positions alone. What `record` answers.
    pub(super) fn note_fresh<T>(
        &mut self,
        stages: vk::PipelineStageFlags2,
        (uses, resident): (&[Use<ResourceKey>], &BTreeSet<ResourceKey>),
        into: NotedInto,
        record: impl FnOnce(&[Use<ResourceKey>]) -> T,
    ) -> T {
        let found = self.found_fresh(uses, into);
        let answer = record(if found { &self.fresh } else { uses });
        self.core = Some(into.core);
        self.stages = stages;
        self.prefix = None;
        self.keep(uses, found);
        self.resident.clear();
        self.resident.extend(resident);
        self.keep_into(into);
        #[cfg(debug_assertions)]
        self.note_whole((uses, resident));
        answer
    }

    /// [`note_fresh`](Self::note_fresh) a draw whose `uses` begin with those of what its encoder
    /// made resident of the identities `prefix`: those it keeps by their identities, and its uses
    /// past them alone, `record` recording all of them unless the last command began with the
    /// same, then those of its uses past them that differ from the last one's at the same
    /// position (the module docs).
    pub(super) fn note_draw_fresh<T>(
        &mut self,
        stages: vk::PipelineStageFlags2,
        (uses, resident, prefix): (&[Use<ResourceKey>], &BTreeSet<ResourceKey>, ReachingPrefix),
        into: NotedInto,
        record: impl FnOnce(&[Use<ResourceKey>]) -> T,
    ) -> T {
        let found = self.found_fresh_draw((uses, prefix.prefix), into);
        let answer = record(if found { &self.fresh } else { uses });
        self.core = Some(into.core);
        self.stages = stages;
        self.prefix = Some(prefix);
        self.keep(&uses[prefix.prefix.len..], found);
        self.resident.clear();
        self.keep_into(into);
        #[cfg(debug_assertions)]
        self.note_whole((uses, resident));
        #[cfg(not(debug_assertions))]
        let _ = resident;
        answer
    }

    /// Keep `uses` as the last command's: where `found`, those of them that differ from the kept
    /// ones at their position are `fresh` at `fresh_at` ([`found_fresh`](Self::found_fresh)), so
    /// only those positions change (and those past the kept ones are added, every one of them
    /// fresh, in order); else all of them, copied.
    fn keep(&mut self, uses: &[Use<ResourceKey>], found: bool) {
        if !found {
            self.uses.clear();
            self.uses.extend_from_slice(uses);
            return;
        }
        self.uses.truncate(uses.len());
        for (&at, used) in self.fresh_at.iter().zip(&self.fresh) {
            match self.uses.get_mut(at) {
                Some(kept) => kept.clone_from(used),
                None => self.uses.push(used.clone()),
            }
        }
        debug_assert!(
            *self.uses == *uses,
            "a command noted apart kept other uses than its own"
        );
    }

    /// Keep the encoder's state `into` as the last command's.
    fn keep_into(&mut self, into: NotedInto) {
        self.parts = into.parts;
        self.held = into.held;
        self.rebinds = into.rebinds;
        self.aliases = into.aliases;
    }

    /// Keep a command's whole `uses` and `resident` keys (builds with debug assertions).
    #[cfg(debug_assertions)]
    fn note_whole(&mut self, (uses, resident): (&[Use<ResourceKey>], &BTreeSet<ResourceKey>)) {
        let (whole, keys) = &mut self.whole;
        whole.clear();
        whole.extend_from_slice(uses);
        keys.clear();
        keys.extend(resident);
    }

    /// How many of a draw's uses, which begin with those of what its encoder made resident of
    /// the identity `prefix`, its encoder (`core`) touched already while the device's buffers
    /// over host pages stand as they did (`rebinds`): those, where the last draw that began with
    /// some, in that encoder, began with the same (the module docs); then none. They are kept
    /// as touched.
    pub(super) fn touched(
        &mut self,
        prefix: ResidentPrefix,
        (core, rebinds): (u64, bool),
    ) -> usize {
        let now = Some((core, prefix, rebinds));
        let touched = match self.touched == now {
            true => prefix.len,
            false => 0,
        };
        self.touched = now;
        touched
    }

    /// Whether the last command noted was noted into `into`: then `fresh` holds the uses of
    /// `uses` that differ from its use at the same position, which the encoder's current part may
    /// lack (the module docs), and `fresh_at` their positions, and each use its use there covers
    /// is kept in place of it, noted already (counted in `covered`); else the part may lack any
    /// of them.
    fn found_fresh(&mut self, uses: &[Use<ResourceKey>], into: NotedInto) -> bool {
        self.covered = 0;
        if self.core != Some(into.core) || NotedApart::into(self) != into {
            return false;
        }
        self.fresh.clear();
        self.fresh_at.clear();
        for (at, used) in uses.iter().enumerate() {
            match self.uses.get_mut(at) {
                Some(kept) if kept == used => {}
                Some(kept) if kept.covers(used) => {
                    kept.clone_from(used);
                    self.covered += 1;
                }
                _ => {
                    self.fresh.push(used.clone());
                    self.fresh_at.push(at);
                }
            }
        }
        true
    }

    /// [`found_fresh`](Self::found_fresh) of a draw whose `uses` begin with those of what its
    /// encoder made resident of the identity `prefix`: only where the last command began with
    /// the same, then of its uses past them.
    fn found_fresh_draw(
        &mut self,
        (uses, prefix): (&[Use<ResourceKey>], ResidentPrefix),
        into: NotedInto,
    ) -> bool {
        self.covered = 0;
        let fresh = self.prefix.is_some_and(|then| then.prefix == prefix)
            && self.found_fresh(&uses[prefix.len..], into);
        // Each use found fresh differs from the last command's at its place, and each other that
        // does is covered by it (a use past those of what the encoder made resident:
        // `found_fresh`).
        #[cfg(debug_assertions)]
        if fresh {
            let whole = &self.whole.0;
            let differ = |(at, used): &(usize, &Use<ResourceKey>)| whole.get(*at) != Some(*used);
            let covered = |(at, used): &(usize, &Use<ResourceKey>)| {
                whole.get(*at).is_some_and(|last| last.covers(used))
            };
            let past = prefix.len;
            let at = |(at, _): (usize, &Use<ResourceKey>)| at;
            let fresh_at = self.fresh_at.iter().map(|at| at + past);
            assert!(
                (uses.iter().enumerate())
                    .filter(|used| differ(used) && !covered(used))
                    .map(at)
                    .eq(fresh_at),
                "a draw noted apart found fresh by its uses' identity other uses than by its uses"
            );
        }
        fresh
    }

    fn into(&self) -> NotedInto {
        NotedInto {
            core: self.core.unwrap_or_default(),
            parts: self.parts,
            held: self.held,
            rebinds: self.rebinds,
            aliases: self.aliases,
        }
    }
}
