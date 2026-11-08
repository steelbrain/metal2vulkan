//! New textures that read zero where Metal reads its fixed pattern, announced where a command
//! that runs first observes that content.
//!
//! Where the device cannot write a new compressed `Private` texture's pattern exactly
//! (`Device::fresh_patterned`), the texture reads zero in those levels instead. The difference is
//! only seen if something reads those levels before a write covers them. Nearly every
//! multisampled depth target is cleared before anything reads it. So the texture announces
//! `fresh-pattern-zeroed` only at the first command, in the order the commands run, that reads one
//! of those subresources while it is still unwritten, not when it is made.
//!
//! - **Unwritten.** Each pattern level of each image layer starts unwritten, in each aspect the
//!   image holds. The tracker keeps, per level, layer and aspect, the boxes of texels no write has
//!   covered yet, so writes of parts that together cover the level write it as one whole write
//!   does. What counts as a write of a box:
//!   - a render pass's clear of an attachment, over the pass's render area in the attachment's
//!     layers (a 3D attachment's depth planes), in each aspect it clears, logged as its encoder
//!     ends ([`FreshUnwritten::cleared`]; as its encoder ends, not as it is made). Where the pass
//!     renders the attachment through another image (a stand-in copied to and from it, separate
//!     depth and stencil textures' scratch), the clear reaches the texture only by the copy back,
//!     so it counts only where the pass stores that aspect: a `DontCare` store leaves the texture's
//!     texels as they were, as Metal keeps them;
//!   - a pass's resolve, over the render area of the resolve texture's level and layers, in the
//!     aspects it resolves (and the whole level where Metal writes the clear color past the
//!     attachment, `Resolve::clear`), as its encoder ends ([`FreshUnwritten::resolved`]);
//!   - a blit's destination box in the aspect it copies ([`FreshUnwritten::wrote`]): a copy from a
//!     buffer, per sample into a multisampled texture too, a copy from a texture, a mipmap
//!     generation's levels.
//!
//!   Nothing else counts: a draw, a `DontCare` load or a shader write may leave any texel
//!   unwritten, so its texels stay as they were. The boxes are bounded ([`MAX_BOXES`] a level's
//!   layer's aspect): a write that would split them past the bound takes only the boxes it covers
//!   or trims to one, so what stays unwritten still holds every texel no write covered (a read may
//!   then be announced that an unbounded record would not, never the reverse).
//! - **Observed.** A pass that loads an unwritten subresource (`Load`, of any aspect) observes it
//!   (logged as its encoder is made, [`FreshUnwritten::begin_pass`]) where its render area reaches
//!   an unwritten texel of that aspect; a command whose uses read one otherwise observes it (logged
//!   as it is encoded, `EncoderCore`'s `command_reaching` and `note_reaching`) where an unwritten
//!   texel lies in the levels, layers and aspects it reads: a blit from it, a shader that samples,
//!   reads or fetches it, an input attachment. An attachment's own reads (blending, the depth and
//!   stencil tests) are its load's, which its pass decided. The device announces the kind at the
//!   commit of the command buffer whose command observed it, inside that program call, naming the
//!   texture and the reason, and stops tracking the texture.
//! - **Order: the commit order.** Metal runs a queue's command buffers in commit order, whatever
//!   order they were encoded in, on however many threads. So a command buffer logs its commands'
//!   writes and reads of noted textures in the order they were encoded ([`FreshLog`]), and its
//!   commit takes the log ([`FreshUnwritten::commit`], before the scheduler sees it): the record
//!   follows the commits, one command buffer's reads and writes at a time. A pass that loads a
//!   texture after a pass of a command buffer committed before it cleared it observes nothing,
//!   though it was encoded first, and one that loads it before a clear committed after it
//!   announces it, whichever was encoded first. A command buffer dropped uncommitted takes
//!   nothing, nor does a refused commit's, which runs nothing; the log of one committed as failed
//!   is taken whole, as its encoders were encoded.
//! - **Cheap.** With no such texture live, a command checks one flag and logs nothing, and a commit
//!   checks it once. With one live, a command logs only what reaches a noted texture, and not a
//!   repeat of that texture's last entry ([`FreshLog::add`]): a draw that reads none logs nothing
//!   and allocates nothing. A texture's entry goes when every subresource is written, when it is
//!   announced, or when the texture is released.
//!
//! A new multisampled float texture whose pattern holds a NaN is noted alike where the device
//! writes the pattern by a copy that converts the NaNs (`nan_clear::converting_copy`): its samples
//! hold the device's NaNs, announced (`device-nan-encoding`) where a command first reads them
//! ([`FreshKind`]).
//!
//! A new render target's 32-bit float depth whose pattern copies write exactly, a NaN with the sign
//! set, is noted too ([`FreshKind::SignedDepth`]): every read of its bits is Metal's, but Metal's
//! depth test compares the stored value's magnitude, which lies above every depth, and a pass with
//! depth writes stores it without its sign (our probe `fd32`, the M3 Max under macOS 27.0.1). A
//! command's read observes nothing of it. A pass that loads its depth where its attachment's level
//! reaches an unwritten texel ([`FreshUnwritten::loads_signed_depth`], weighed at its commit as any
//! read) announces it as a write that may store such a value does (`signed-depth-test`, where the
//! device's test is not Metal's), and the texture is no longer noted.

use super::command_buffer::CommandBuffer;
use super::device::lock;
use super::error::Error;
use super::key_hash::KeyMap;
use super::render_pass::{AspectTarget, PassLayout, Target};
use super::render_values::{LoadAction, StoreAction};
use super::resolve::Resolve;
use super::resource::ResourceKey;
use super::sparse::emulated::Written;
use super::texture::Texture;
use super::texture_descriptor::TextureLayout;
use super::tracker::{Access, Use, UseRange};
use super::Device;
use ash::vk::{self, ImageAspectFlags};
use metal2vulkan::vk::{Format, ImageType};
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

/// The device's new textures that read zero where Metal reads its pattern and that no command has
/// observed or written whole yet (the module docs).
#[derive(Default)]
pub(super) struct FreshUnwritten {
    /// Whether any texture is noted: without one, a command logs nothing and a commit takes
    /// nothing.
    any: AtomicBool,
    textures: Mutex<KeyMap<ResourceKey, Unwritten>>,
    /// Held while a commit takes its command buffer's log ([`FreshUnwritten::commit`]), so each
    /// command buffer's reads and writes apply together, in the order of the commits.
    commits: Mutex<()>,
}

/// A command buffer's reads and writes of noted textures, in the order its commands were
/// encoded, which its commit takes ([`FreshUnwritten::commit`], the module docs' **Order**):
/// logged only while a texture is noted. A leaf lock, apart from the command buffer's state.
#[derive(Default)]
pub(crate) struct FreshLog {
    events: Mutex<Vec<FreshEvent>>,
    /// Whether an event was logged: a commit of a command buffer that logged none takes not even
    /// the lock, whose first take allocates where the platform's mutex is made lazily (macOS).
    logged: AtomicBool,
}

/// One entry of a [`FreshLog`].
enum FreshEvent {
    /// `aspects` of `texels` of the texture `key` written ([`FreshUnwritten::written`]).
    Written {
        key: ResourceKey,
        aspects: ImageAspectFlags,
        texels: Written,
    },
    /// A read of `aspects` of `texels` ([`FreshUnwritten::observed_within`]).
    Read {
        key: ResourceKey,
        aspects: ImageAspectFlags,
        texels: Written,
    },
    /// A read of `aspects` of `levels` and `layers`, any texel ([`FreshUnwritten::observed`]).
    ReadRange {
        key: ResourceKey,
        aspects: ImageAspectFlags,
        levels: Range<u32>,
        layers: Range<u32>,
    },
    /// A pass's load of the depth of `texels` of `texture`, whose image's key is `key`
    /// ([`FreshUnwritten::loaded_signed_depth`]).
    LoadsDepth {
        key: ResourceKey,
        texels: Written,
        texture: Texture,
    },
}

/// How far back [`FreshLog::add`] looks for its texture's last entry.
const REPEATS: usize = 8;

impl FreshLog {
    /// Log `event` where its texture is noted now and the record weighs it
    /// ([`FreshEvent::weighed_by`]), unless it repeats its texture's last entry in the log,
    /// within the last [`REPEATS`]. Either way the commit's answer is the same: a texture not
    /// noted now never is again, its entry being made with its image and never after, an entry's
    /// kind never changes, and a repeated read, write or load finds the record as the first one
    /// left it. So a command that reads a noted texture again and again grows the log once, and
    /// one that reads none logs nothing and takes no allocation.
    fn add(&self, fresh: &FreshUnwritten, event: FreshEvent) {
        let weighed = lock(&fresh.textures)
            .get(&event.key())
            .is_some_and(|entry| event.weighed_by(entry));
        if !weighed {
            return;
        }
        let mut events = lock(&self.events);
        let last = (events.iter().rev().take(REPEATS)).find(|logged| logged.key() == event.key());
        if last != Some(&event) {
            events.push(event);
            self.logged.store(true, Ordering::Release);
        }
    }
}

impl FreshEvent {
    /// The texture's image the entry is of.
    fn key(&self) -> ResourceKey {
        match self {
            FreshEvent::Written { key, .. }
            | FreshEvent::Read { key, .. }
            | FreshEvent::ReadRange { key, .. }
            | FreshEvent::LoadsDepth { key, .. } => *key,
        }
    }

    /// Whether the commit's replay of it can change `entry` or announce it: a write any noted
    /// texture's, a read any but a [`FreshKind::SignedDepth`] one's (which no read observes), a
    /// pass's load of the depth only such a one's.
    fn weighed_by(&self, entry: &Unwritten) -> bool {
        let signed = entry.notice.kind == FreshKind::SignedDepth;
        match self {
            FreshEvent::Written { .. } => true,
            FreshEvent::Read { .. } | FreshEvent::ReadRange { .. } => !signed,
            FreshEvent::LoadsDepth { .. } => signed,
        }
    }
}

/// Entries of the same kind of the same texels, a load's texture aside (its key names its image).
impl PartialEq for FreshEvent {
    fn eq(&self, other: &FreshEvent) -> bool {
        use FreshEvent::{LoadsDepth, Read, ReadRange, Written};
        match (self, other) {
            (
                Written {
                    key,
                    aspects,
                    texels,
                },
                Written {
                    key: k,
                    aspects: a,
                    texels: t,
                },
            )
            | (
                Read {
                    key,
                    aspects,
                    texels,
                },
                Read {
                    key: k,
                    aspects: a,
                    texels: t,
                },
            ) => (key, aspects, texels) == (k, a, t),
            (
                ReadRange {
                    key,
                    aspects,
                    levels,
                    layers,
                },
                ReadRange {
                    key: k,
                    aspects: a,
                    levels: l,
                    layers: y,
                },
            ) => (key, aspects, levels, layers) == (k, a, l, y),
            (
                LoadsDepth { key, texels, .. },
                LoadsDepth {
                    key: k, texels: t, ..
                },
            ) => (key, texels) == (k, t),
            _ => false,
        }
    }
}

/// A noted texture: which texels of each aspect of its pattern levels' layers no write has
/// covered yet, and what the notice names.
struct Unwritten {
    /// The levels Metal's pattern covers, from level 0.
    levels: u32,
    /// The image's layers.
    layers: u32,
    /// The image's aspects, each tracked apart: a copy writes one aspect of a depth/stencil
    /// texture.
    aspects: Vec<ImageAspectFlags>,
    /// The image's extent at level 0 (a 3D image's depth planes in the last).
    extent: [u32; 3],
    /// The boxes of each level's layer's aspect no write has covered (level-major, then layer,
    /// then aspect), disjoint; empty once written whole.
    unwritten: Vec<Vec<Bounds>>,
    notice: Notice,
}

/// A command's read of a box of a texture's image (`EncoderCore::command_reading`): the image's
/// key, the aspects, and the texels (a level's layers and a box in it).
pub(super) struct FreshRead {
    pub(super) key: ResourceKey,
    pub(super) aspects: ImageAspectFlags,
    pub(super) texels: Written,
}

/// The most boxes a level's layer's aspect keeps unwritten (the module docs).
const MAX_BOXES: usize = 32;

/// A box of a level's texels: from `lo` up to `hi` (exclusive) on each axis, `[x, y, z]` (a 3D
/// image's depth planes in `z`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Bounds {
    lo: [u32; 3],
    hi: [u32; 3],
}

impl Bounds {
    /// The texels of `texels`'s box inside `level`, the extent of its level.
    fn of(texels: &Written, level: [u32; 3]) -> Bounds {
        Bounds {
            lo: std::array::from_fn(|axis| texels.origin[axis].min(level[axis])),
            hi: std::array::from_fn(|axis| {
                (texels.origin[axis].saturating_add(texels.size[axis])).min(level[axis])
            }),
        }
    }

    /// Whether the box holds no texel.
    fn is_empty(&self) -> bool {
        (0..3).any(|axis| self.lo[axis] >= self.hi[axis])
    }

    /// Whether the two boxes share a texel.
    fn meets(&self, other: &Bounds) -> bool {
        (0..3).all(|axis| self.lo[axis] < other.hi[axis] && other.lo[axis] < self.hi[axis])
    }

    /// The texels of the box outside `other`, as at most six disjoint boxes: the slabs before
    /// and after it in depth, then in rows within its depth, then in columns within its rows.
    fn minus(&self, other: &Bounds) -> Vec<Bounds> {
        if !self.meets(other) {
            return vec![*self];
        }
        let mut rest = Vec::new();
        let mut inside = *self;
        for axis in [2, 1, 0] {
            if inside.lo[axis] < other.lo[axis] {
                let mut before = inside;
                before.hi[axis] = other.lo[axis];
                rest.push(before);
                inside.lo[axis] = other.lo[axis];
            }
            if other.hi[axis] < inside.hi[axis] {
                let mut after = inside;
                after.lo[axis] = other.hi[axis];
                rest.push(after);
                inside.hi[axis] = other.hi[axis];
            }
        }
        rest
    }

    /// The one box the two make, where they span the same texels on two axes and abut on the
    /// third.
    fn joined(&self, other: &Bounds) -> Option<Bounds> {
        let same = |axis: usize| self.lo[axis] == other.lo[axis] && self.hi[axis] == other.hi[axis];
        let differs: Vec<usize> = (0..3).filter(|&axis| !same(axis)).collect();
        let &[axis] = differs.as_slice() else {
            return None;
        };
        let abut = self.hi[axis] == other.lo[axis] || other.hi[axis] == self.lo[axis];
        abut.then(|| {
            let mut joined = *self;
            joined.lo[axis] = self.lo[axis].min(other.lo[axis]);
            joined.hi[axis] = self.hi[axis].max(other.hi[axis]);
            joined
        })
    }
}

/// `unwritten` (disjoint boxes) less the texels of `written`, boxes that abut joined; past
/// [`MAX_BOXES`] only the boxes `written` covers or trims to one change (the module docs).
fn less(unwritten: &[Bounds], written: &Bounds) -> Vec<Bounds> {
    let mut rest: Vec<Bounds> = unwritten.iter().flat_map(|b| b.minus(written)).collect();
    let mut joining = true;
    while joining {
        joining = false;
        'pairs: for i in 0..rest.len() {
            for j in i + 1..rest.len() {
                if let Some(joined) = rest[i].joined(&rest[j]) {
                    rest[i] = joined;
                    rest.swap_remove(j);
                    joining = true;
                    break 'pairs;
                }
            }
        }
    }
    if rest.len() <= MAX_BOXES {
        return rest;
    }
    unwritten
        .iter()
        .flat_map(|b| match b.minus(written) {
            trimmed if trimmed.len() <= 1 => trimmed,
            _ => vec![*b],
        })
        .collect()
}

/// The extent of `level` of an image of `extent`.
fn level_extent(extent: [u32; 3], level: u32) -> [u32; 3] {
    extent.map(|axis| (axis >> level.min(31)).max(1))
}

/// What the announcement names: the texture's layout and the image's format, and what the texture
/// reads instead of Metal's pattern.
#[derive(Clone, Copy, Debug)]
pub(super) struct Notice {
    pub(super) layout: TextureLayout,
    pub(super) format: Format,
    pub(super) kind: FreshKind,
}

/// What a noted texture reads where Metal reads its pattern, and so what its notice announces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FreshKind {
    /// Zero (`fresh-pattern-zeroed`): the device writes no exact pattern.
    Zeroed,
    /// The pattern copied into a multisampled float image by a device whose copy converts its
    /// NaNs (`RuntimeDriverDefect::MultisampledCopiesConvertFloatNans`): each NaN as the device
    /// converts it (`device-nan-encoding`).
    NansConverted,
    /// Metal's pattern itself, copied into a render target's 32-bit float depth: a NaN with the
    /// sign set, which every read gives as Metal's, and which Metal's depth test compares by its
    /// magnitude. Only a pass that loads it observes it, and announces it
    /// ([`FreshUnwritten::loads_signed_depth`]); nothing is announced for it here.
    SignedDepth,
}

impl Unwritten {
    /// Whether an unwritten texel of `aspects` lies in `texels` (a level's layers and a box in it).
    fn reaches(&self, aspects: ImageAspectFlags, texels: &Written) -> bool {
        let bounds = Bounds::of(texels, level_extent(self.extent, texels.level));
        let levels = texels.level..texels.level + 1;
        (self.indices(aspects, levels, texels.layers.clone()))
            .any(|i| self.unwritten[i].iter().any(|b| b.meets(&bounds)))
    }

    /// The records of `aspects` of `levels` and `layers` inside the texture's pattern levels,
    /// layers and aspects.
    fn indices(
        &self,
        aspects: ImageAspectFlags,
        levels: Range<u32>,
        layers: Range<u32>,
    ) -> impl Iterator<Item = usize> + '_ {
        let levels = levels.start..levels.end.min(self.levels);
        let layers = layers.start..layers.end.min(self.layers);
        let planes = self.aspects.len();
        let named: Vec<usize> = (self.aspects.iter().enumerate())
            .filter(|(_, aspect)| aspects.intersects(**aspect))
            .map(|(plane, _)| plane)
            .collect();
        levels.flat_map(move |level| {
            let named = named.clone();
            layers.clone().flat_map(move |layer| {
                let first = (level * self.layers + layer) as usize * planes;
                named.clone().into_iter().map(move |plane| first + plane)
            })
        })
    }
}

impl FreshUnwritten {
    /// Note the texture `key`, unwritten in every texel of each of `aspects` (the image's) of
    /// every layer of `layers` of its first `levels` levels, of an image of `extent`.
    pub(super) fn note(
        &self,
        key: ResourceKey,
        levels: u32,
        layers: u32,
        aspects: ImageAspectFlags,
        extent: [u32; 3],
        notice: Notice,
    ) {
        let aspects: Vec<ImageAspectFlags> = (0..32)
            .map(|bit| ImageAspectFlags::from_raw(1 << bit))
            .filter(|aspect| aspects.contains(*aspect))
            .collect();
        let unwritten = (0..levels)
            .flat_map(|level| {
                let whole = Bounds {
                    lo: [0; 3],
                    hi: level_extent(extent, level),
                };
                (0..layers * aspects.len() as u32).map(move |_| vec![whole])
            })
            .collect();
        let entry = Unwritten {
            levels,
            layers,
            aspects,
            extent,
            unwritten,
            notice,
        };
        let mut textures = lock(&self.textures);
        textures.insert(key, entry);
        self.any.store(true, Ordering::Relaxed);
    }

    /// Whether any texture is noted.
    pub(super) fn any(&self) -> bool {
        self.any.load(Ordering::Relaxed)
    }

    /// `aspects` of `texels` (a level's layers and a box in it) of the texture `key` are written:
    /// no longer unwritten. The texture's entry goes once nothing is.
    pub(super) fn written(&self, key: ResourceKey, aspects: ImageAspectFlags, texels: &Written) {
        let mut textures = lock(&self.textures);
        let Some(entry) = textures.get_mut(&key) else {
            return;
        };
        let bounds = Bounds::of(texels, level_extent(entry.extent, texels.level));
        if bounds.is_empty() {
            return;
        }
        let levels = texels.level..texels.level + 1;
        let indices: Vec<usize> = entry
            .indices(aspects, levels, texels.layers.clone())
            .collect();
        for index in indices {
            entry.unwritten[index] = less(&entry.unwritten[index], &bounds);
        }
        if entry.unwritten.iter().all(Vec::is_empty) {
            textures.remove(&key);
            self.any.store(!textures.is_empty(), Ordering::Relaxed);
        }
    }

    /// A command reads `aspects` of `levels` and `layers` of the texture `key`, any texel: the
    /// notice to announce where one is still unwritten, after which the texture is no longer
    /// noted.
    pub(super) fn observed(
        &self,
        key: ResourceKey,
        aspects: ImageAspectFlags,
        levels: Range<u32>,
        layers: Range<u32>,
    ) -> Option<Notice> {
        self.observe(key, |entry| {
            entry.notice.kind != FreshKind::SignedDepth
                && (entry.indices(aspects, levels, layers)).any(|i| !entry.unwritten[i].is_empty())
        })
    }

    /// A command reads `aspects` of `texels` (a level's layers and a box in it) of the texture
    /// `key`: the notice to announce where a texel it reads is still unwritten, after which the
    /// texture is no longer noted.
    pub(super) fn observed_within(
        &self,
        key: ResourceKey,
        aspects: ImageAspectFlags,
        texels: &Written,
    ) -> Option<Notice> {
        self.observe(key, |entry| {
            entry.notice.kind != FreshKind::SignedDepth && entry.reaches(aspects, texels)
        })
    }

    /// A pass loads the depth of `texels` (a level's layers and a box in it) of the texture `key`:
    /// whether its entry is a [`FreshKind::SignedDepth`] one with an unwritten texel there, after
    /// which the texture is no longer noted.
    pub(super) fn loaded_signed_depth(&self, key: ResourceKey, texels: &Written) -> bool {
        self.observe(key, |entry| {
            let depth = ImageAspectFlags::DEPTH;
            entry.notice.kind == FreshKind::SignedDepth && entry.reaches(depth, texels)
        })
        .is_some()
    }

    /// The notice of the texture `key` where `reaches` its entry (a read reaches an unwritten
    /// texel), after which the texture is no longer noted.
    fn observe(
        &self,
        key: ResourceKey,
        reaches: impl FnOnce(&Unwritten) -> bool,
    ) -> Option<Notice> {
        let mut textures = lock(&self.textures);
        let entry = textures.get(&key)?;
        if !reaches(entry) {
            return None;
        }
        let notice = entry.notice;
        textures.remove(&key);
        self.any.store(!textures.is_empty(), Ordering::Relaxed);
        Some(notice)
    }

    /// The command buffer whose log is `log` commits (`command_buffer/mod.rs`'s `commit_now`,
    /// before its queue's scheduler sees it, in the order its queue runs it): where `runs` (all
    /// but a refused commit, whose commands run nothing), its reads and writes of noted textures
    /// apply to the record in the order they were encoded, as one, and each read that reached an
    /// unwritten texel is announced once the record's locks are free, in the committing call (the
    /// module docs' **Order**). One atomic read where no texture is noted, two where the command
    /// buffer logged nothing; the log goes with the command buffer.
    pub(crate) fn commit(device: &Device, log: &FreshLog, runs: bool) {
        let fresh = &device.shared().fresh;
        if !fresh.any() || !log.logged.load(Ordering::Acquire) {
            return;
        }
        // Committed, the command buffer encodes nothing more: its log goes to the record whole.
        let events = std::mem::take(&mut *lock(&log.events));
        if !runs || events.is_empty() {
            return;
        }
        let mut zeroed = Vec::new();
        let mut signed = Vec::new();
        {
            let _turn = lock(&fresh.commits);
            for event in &events {
                match event {
                    FreshEvent::Written {
                        key,
                        aspects,
                        texels,
                    } => fresh.written(*key, *aspects, texels),
                    FreshEvent::Read {
                        key,
                        aspects,
                        texels,
                    } => zeroed.extend(fresh.observed_within(*key, *aspects, texels)),
                    FreshEvent::ReadRange {
                        key,
                        aspects,
                        levels,
                        layers,
                    } => {
                        let (levels, layers) = (levels.clone(), layers.clone());
                        zeroed.extend(fresh.observed(*key, *aspects, levels, layers));
                    }
                    FreshEvent::LoadsDepth {
                        key,
                        texels,
                        texture,
                    } => {
                        if fresh.loaded_signed_depth(*key, texels) {
                            signed.push(texture);
                        }
                    }
                }
            }
        }
        for notice in zeroed {
            device.degrade_fresh_pattern(notice);
        }
        for texture in signed {
            texture.degrade_signed_depth("new_texture");
        }
    }

    /// A command of `maker`'s reads each texture `uses` read in a level and layer (a blit's, a
    /// shader's or an input attachment's read), logged for the commit: in the boxes `reads` names
    /// of it where it names any, else any texel of those levels and layers. An attachment's own
    /// reads are its pass's load, which the pass decides as it begins
    /// ([`begin_pass`](Self::begin_pass)).
    pub(super) fn observe_reads(
        maker: &CommandBuffer,
        uses: &[Use<ResourceKey>],
        reads: &[FreshRead],
    ) {
        if !maker.generation().fresh.any() {
            return;
        }
        let attachment = vk::AccessFlags2::COLOR_ATTACHMENT_READ
            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ;
        let reading = |used: &&Use<ResourceKey>| !(used.read.access & !attachment).is_empty();
        let (fresh, log) = (&maker.generation().fresh, maker.fresh_log());
        for used in uses.iter().filter(reading) {
            let UseRange::Image {
                aspects,
                levels,
                layers,
            } = &used.range
            else {
                continue;
            };
            let mut boxes = reads.iter().filter(|read| read.key == used.key).peekable();
            match boxes.peek().is_some() {
                true => {
                    for read in boxes {
                        let event = FreshEvent::Read {
                            key: read.key,
                            aspects: read.aspects,
                            texels: read.texels.clone(),
                        };
                        log.add(fresh, event);
                    }
                }
                false => {
                    let event = FreshEvent::ReadRange {
                        key: used.key,
                        aspects: *aspects,
                        levels: levels.clone(),
                        layers: layers.clone(),
                    };
                    log.add(fresh, event);
                }
            }
        }
    }

    /// A render pass of `layout` begins in `maker` (the module docs): each attachment's texture,
    /// a noted one, is read over the render area in its layers in each aspect it loads (`Load`),
    /// logged for the commit. The texture is the attachment's own where the pass renders it
    /// through another image: a stand-in's own texture, and separate depth and stencil textures,
    /// each in its aspect. Its clears are logged as it ends ([`cleared`](Self::cleared)).
    pub(super) fn begin_pass(maker: &CommandBuffer, layout: &PassLayout) -> Result<(), Error> {
        let fresh = &maker.device().shared().fresh;
        if !fresh.any() {
            return Ok(());
        }
        let loads = |aspect: &Aspect| aspect.load == LoadAction::Load;
        pass_targets(layout, loads, |target, aspects| {
            let texels = pass_texels(target, layout.extent)?;
            if let Some(key) = image_key(&target.texture, &texels)? {
                let event = FreshEvent::Read {
                    key,
                    aspects,
                    texels,
                };
                maker.fresh_log().add(fresh, event);
            }
            Ok(())
        })
    }

    /// A render pass of `maker`'s being planned loads the depth of `target`, its depth
    /// attachment (the module docs), logged for the commit: where the attachment's texture is a
    /// [`FreshKind::SignedDepth`] one with an unwritten texel in the attachment's level and
    /// layers then, the commit announces `signed-depth-test` for it, and it is no longer noted.
    /// The whole level is weighed, as the render area is not decided yet.
    pub(super) fn loads_signed_depth(maker: &CommandBuffer, target: &Target) -> Result<(), Error> {
        let fresh = &maker.device().shared().fresh;
        if !fresh.any() {
            return Ok(());
        }
        let texels = pass_texels(target, [u32::MAX; 2])?;
        if let Some(key) = image_key(&target.texture, &texels)? {
            let event = FreshEvent::LoadsDepth {
                key,
                texels,
                texture: target.texture.clone(),
            };
            maker.fresh_log().add(fresh, event);
        }
        Ok(())
    }

    /// A render pass of `maker`'s, of `layout`, ends (its layout as the pass decided it, before
    /// its `DontCare` loads take clears, `clear_reach.rs`): each attachment's texture is written
    /// over the render area in its layers in each aspect the pass clears, where the clear
    /// reaches it, logged for the commit: an attachment the pass renders itself, or one it
    /// renders through another image (a stand-in copied to and from it, separate depth and
    /// stencil textures' scratch) where the pass stores that aspect, whose copy back writes it.
    /// Where such a pass does not store it (`DontCare`), or renders through a backing nothing
    /// copies back, the texture keeps its texels, as Metal keeps them, and stays unwritten.
    pub(super) fn cleared(maker: &CommandBuffer, layout: &PassLayout) -> Result<(), Error> {
        let fresh = &maker.device().shared().fresh;
        if !fresh.any() {
            return Ok(());
        }
        let clears = |aspect: &Aspect| aspect.load == LoadAction::Clear && aspect.reaches;
        pass_targets(layout, clears, |target, aspects| {
            let texels = pass_texels(target, layout.extent)?;
            if let Some(key) = image_key(&target.texture, &texels)? {
                let event = FreshEvent::Written {
                    key,
                    aspects,
                    texels,
                };
                maker.fresh_log().add(fresh, event);
            }
            Ok(())
        })
    }

    /// A render pass of `maker`'s ended whose resolves are `resolved` (each resolve attachment,
    /// with the aspects of the resolve texture it writes), over its render area `extent` from the
    /// origin, logged for the commit: each writes those aspects of the resolve texture's level and
    /// layers over the render area (the attachment's and the texture it is copied into), and the
    /// whole level where Metal writes the clear color past the attachment (`Resolve::clear`), as
    /// Metal's resolve stores them.
    pub(super) fn resolved(
        maker: &CommandBuffer,
        resolved: &[(&Resolve, metal2vulkan::vk::ImageAspectFlags)],
        extent: [u32; 2],
    ) -> Result<(), Error> {
        let fresh = &maker.device().shared().fresh;
        if !fresh.any() {
            return Ok(());
        }
        for (resolve, aspects) in resolved {
            let copied = resolve
                .copy_to
                .iter()
                .map(|target| (target, resolve.clear.is_some()));
            for (target, whole) in std::iter::once((&resolve.attachment, false)).chain(copied) {
                let mut texels = pass_texels(target, extent)?;
                if whole {
                    texels.size[0] = u32::MAX;
                    texels.size[1] = u32::MAX;
                }
                if let Some(key) = image_key(&target.texture, &texels)? {
                    let event = FreshEvent::Written {
                        key,
                        aspects: (*aspects).into(),
                        texels,
                    };
                    maker.fresh_log().add(fresh, event);
                }
            }
        }
        Ok(())
    }

    /// A blit of `maker`'s wrote `written` of `texture`: each the aspects it wrote and a box (in
    /// the image's levels and layers), every sample of a multisampled texture alike (the module
    /// docs), logged for the commit.
    pub(super) fn wrote(
        maker: &CommandBuffer,
        texture: &Texture,
        written: &[(ImageAspectFlags, Written)],
    ) -> Result<(), Error> {
        let fresh = &maker.device().shared().fresh;
        if !fresh.any() {
            return Ok(());
        }
        for (aspects, texels) in written {
            if let Some(key) = image_key(texture, texels)? {
                let event = FreshEvent::Written {
                    key,
                    aspects: *aspects,
                    texels: texels.clone(),
                };
                maker.fresh_log().add(fresh, event);
            }
        }
        Ok(())
    }

    /// A blit's reads of `read` of `texture` (each the aspects and a box, in the image's levels
    /// and layers), for `EncoderCore::command_named`: none where no texture is noted, nor for a
    /// texture over a buffer's bytes.
    pub(super) fn reads(
        device: &Device,
        texture: &Texture,
        read: &[(ImageAspectFlags, Written)],
    ) -> Result<Vec<FreshRead>, Error> {
        if !device.shared().fresh.any() {
            return Ok(Vec::new());
        }
        let mut reads = Vec::with_capacity(read.len());
        for (aspects, texels) in read {
            if let Some(key) = image_key(texture, texels)? {
                reads.push(FreshRead {
                    key,
                    aspects: *aspects,
                    texels: texels.clone(),
                });
            }
        }
        Ok(reads)
    }

    /// The texture `key` is released: no longer noted.
    pub(super) fn forget(&self, key: ResourceKey) {
        if !self.any() {
            return;
        }
        let mut textures = lock(&self.textures);
        if textures.remove(&key).is_some() {
            self.any.store(!textures.is_empty(), Ordering::Relaxed);
        }
    }
}

/// The key of `texture`'s image, whose levels and layers `texels` names; `None` for a texture
/// over a buffer's bytes.
fn image_key(texture: &Texture, texels: &Written) -> Result<Option<ResourceKey>, Error> {
    let levels = texels.level..texels.level + 1;
    let used = texture.image_use(levels, texels.layers.clone(), Access::NONE, false)?;
    Ok(matches!(used.range, UseRange::Image { .. }).then_some(used.key))
}

/// One aspect of a pass's attachment, as [`pass_targets`] gives it to a test: its load action,
/// and whether what the pass renders reaches the attachment's own texture: where the pass
/// renders that texture itself, or through another image copied back to it, where it stores the
/// aspect (`Store`, `StoreAndMultisampleResolve`, or a depth's `CustomSampleDepthStore`); not
/// through a backing nothing copies back (a memoryless texture's, the kept one).
struct Aspect {
    load: LoadAction,
    reaches: bool,
}

/// `each` of each attachment's own texture in `layout` (a stand-in's own, separate depth and
/// stencil textures each in its aspect), with the aspects for which `test` holds, where any does;
/// a hidden depth and stencil, and an aspect no attachment names (`keep_depth`, `keep_stencil`),
/// never. It lists them in no storage of its own: a pass begins and ends with a noted texture
/// live elsewhere and allocates nothing for that (the module docs, **Cheap**).
fn pass_targets<'a>(
    layout: &'a PassLayout,
    test: impl Fn(&Aspect) -> bool,
    mut each: impl FnMut(&'a Target, ImageAspectFlags) -> Result<(), Error>,
) -> Result<(), Error> {
    let stores = |store: StoreAction| {
        matches!(
            store,
            StoreAction::Store
                | StoreAction::StoreAndMultisampleResolve
                | StoreAction::CustomSampleDepthStore
        )
    };
    for color in layout.colors.iter().flatten() {
        let stand_in = color.stand_in.as_ref();
        let aspect = Aspect {
            load: color.load,
            reaches: stand_in.is_none_or(|s| s.copies() && stores(color.store)),
        };
        if test(&aspect) {
            each(
                stand_in.map_or(&color.target, |s| &s.own),
                ImageAspectFlags::COLOR,
            )?;
        }
    }
    let (depth, stencil) = (ImageAspectFlags::DEPTH, ImageAspectFlags::STENCIL);
    for ds in layout.depth_stencil.iter().filter(|ds| !ds.hidden) {
        // Separate textures render through a scratch copied to and from them, as a stand-in that
        // copies does (`depth_stencil_scratch.rs`).
        let through = match (&ds.separate, &ds.stand_in) {
            (Some(_), _) => Some(true),
            (None, Some(stand_in)) => Some(stand_in.copies()),
            (None, None) => None,
        };
        let by = |aspect: &Option<AspectTarget>, kept: bool, flag| match aspect {
            // A NaN clear depth written into the texture before the pass, which loads it
            // (`nan_clear.rs`), is the texture's clear.
            Some(a) if !kept && a.written_nan => {
                let aspect = Aspect {
                    load: LoadAction::Clear,
                    reaches: true,
                };
                match test(&aspect) {
                    true => flag,
                    false => ImageAspectFlags::empty(),
                }
            }
            Some(a) if !kept => {
                let aspect = Aspect {
                    load: a.load,
                    reaches: through.is_none_or(|copies| copies && stores(a.store)),
                };
                match test(&aspect) {
                    true => flag,
                    false => ImageAspectFlags::empty(),
                }
            }
            _ => ImageAspectFlags::empty(),
        };
        let aspects =
            by(&ds.depth, ds.keep_depth, depth) | by(&ds.stencil, ds.keep_stencil, stencil);
        let own: [Option<(&Target, ImageAspectFlags)>; 2] = match (&ds.separate, &ds.stand_in) {
            (Some(separate), _) => [
                separate.depth.as_ref().map(|t| (t, depth)),
                Some((&separate.stencil, stencil)),
            ],
            (None, Some(stand_in)) => [Some((&stand_in.own, depth | stencil)), None],
            (None, None) => [Some((&ds.target, depth | stencil)), None],
        };
        for (target, its) in own.into_iter().flatten() {
            if !(aspects & its).is_empty() {
                each(target, aspects & its)?;
            }
        }
    }
    Ok(())
}

/// The texels a pass's attachment `target` covers over the render area `extent` from the
/// origin: its level's layers, or a 3D image's depth planes.
fn pass_texels(target: &Target, extent: [u32; 2]) -> Result<Written, Error> {
    let volume = target.texture.image_plan()?.image_type == ImageType::TYPE_3D;
    let planes = target.layer..target.layer + target.layers;
    Ok(match volume {
        true => Written {
            level: target.level,
            layers: 0..1,
            origin: [0, 0, planes.start],
            size: [extent[0], extent[1], target.layers],
        },
        false => Written {
            level: target.level,
            layers: planes,
            origin: [0; 3],
            size: [extent[0], extent[1], 1],
        },
    })
}
