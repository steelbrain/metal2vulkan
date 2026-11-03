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
//! only the uses that differ from the last one's at their position (`NotedApart::fresh`): each
//! use equal to the last command's at its place, and so the host aliases' uses beside it, is in
//! the part already, and recording it again changes nothing. Its stages widen the part as before.
//!
//! A render encoder keeps it with its bind state (`live_bind.rs`), which the next render encoder
//! starts from, so its storage is kept from one encoder to the next and a draw allocates none:
//! it names the encoder core that noted it (`EncoderCore::id`, no other core's), and another
//! encoder's command is never taken for one it noted.

use ash::vk;
use std::collections::BTreeSet;

use super::resource::ResourceKey;
use super::tracker::Use;

/// What the last command noted apart was noted with (the module docs).
#[derive(Clone, Default, PartialEq)]
pub(super) struct NotedApart {
    /// The core of the encoder that noted it and kept it, if any.
    core: Option<u64>,
    stages: vk::PipelineStageFlags2,
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
    /// The uses of the command being noted that differ from the last one's
    /// ([`fresh`](NotedApart::fresh)), kept so their storage is too.
    fresh: Vec<Use<ResourceKey>>,
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
            && *self.uses == *uses
            && self.resident.iter().eq(resident)
    }

    /// Whether the last command was noted into `into`: then a command noted with its stages and
    /// uses, reaching the same keys through the resident set alone, is noted already
    /// ([`holds`](Self::holds)).
    pub(super) fn noted_into(&self, into: NotedInto) -> bool {
        self.core == Some(into.core) && self.into() == into
    }

    /// The last command's uses, as given.
    pub(super) fn uses(&self) -> &[Use<ResourceKey>] {
        &self.uses
    }

    /// Keep the command just noted, of `stages` with `uses`, of which it reached the keys
    /// `resident` through the resident set alone, noted into `into`.
    pub(super) fn note(
        &mut self,
        stages: vk::PipelineStageFlags2,
        (uses, resident): (&[Use<ResourceKey>], &BTreeSet<ResourceKey>),
        into: NotedInto,
    ) {
        self.core = Some(into.core);
        self.stages = stages;
        // Its storage is kept from one command to the next.
        self.uses.clear();
        self.uses.extend_from_slice(uses);
        self.resident.clear();
        self.resident.extend(resident);
        self.parts = into.parts;
        self.held = into.held;
        self.rebinds = into.rebinds;
        self.aliases = into.aliases;
    }

    /// The uses of `uses`, a command noted into `into`, that the encoder's current part may
    /// lack: all of them, unless the last command noted was noted into `into`, then those that
    /// differ from its use at the same position (the module docs).
    pub(super) fn fresh<'s>(
        &'s mut self,
        uses: &'s [Use<ResourceKey>],
        into: NotedInto,
    ) -> &'s [Use<ResourceKey>] {
        if self.core != Some(into.core) || NotedApart::into(self) != into {
            return uses;
        }
        self.fresh.clear();
        let last = &self.uses;
        let differ = |(at, used): &(usize, &Use<ResourceKey>)| last.get(*at) != Some(*used);
        self.fresh.extend(
            uses.iter()
                .enumerate()
                .filter(differ)
                .map(|(_, used)| used.clone()),
        );
        &self.fresh
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
