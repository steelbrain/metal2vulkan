//! Shadowed textures in a render pass: `shadow.rs`'s rules, with a render pass instance as the
//! command. A draw records into its instance's secondary command buffer, where no copy may go, so
//! each instance gathers the shadowed textures its draws use and their uses of buffers' bytes, and
//! `end_encoding` plans the copies one instance at a time: the first instance's before every other
//! step of the pass (the separate depth and stencil copies, the stand-ins' copies, the clears),
//! each later instance's between it and the one before, after the barrier that orders them, with a
//! barrier after each copy. What the pass wrote through a texture is copied back as the encoder
//! ends, after the pass's own steps.
//!
//! - **Attachments.** A shadowed attachment (its texture, or the one a stand-in renders for; a
//!   depth or stencil attachment's are heap shadows) or resolve target is a use of every instance:
//!   copied in before the first, so a load, a clear, and the texels the pass leaves alone are the
//!   buffer's; copied back at the end where the pass stores it (`Store`,
//!   `StoreAndMultisampleResolve`) or resolves into it. A `DontCare` store copies nothing back: the
//!   buffer keeps its bytes, Metal's answer (a Metal probe on the M3 Max and an M4, and the
//!   DontCare-store probe, `kept_bytes.rs`). A pass that reaches no tile
//!   (`clear_reach::PassReach::Nothing`) leaves a color attachment it loads, resolves nothing from
//!   and stores (or stores `DontCare`) as it was, as Metal loads and stores only the tiles a draw
//!   reaches: where its texture's shadow is an image of a buffer's bytes, nothing is copied in or
//!   back. A pass whose draws do reach tiles, some or (where it clears and stores an attachment)
//!   every one, changes a color attachment it loads (as the encoder set its load, so no clear
//!   before the pass reaches it: a clear past the render area clears attachments loaded `Clear`
//!   alone, `clear_reach.rs`) only inside its draws' scissors (`clear_reach::DrawnArea`): only
//!   those texels are copied back, and, where the pass
//!   takes the texture in no other way, only those are copied in, as nothing the pass reads
//!   outside them reaches anything but the image (its load is stored back unchanged, and a
//!   fragment reads the attachment at its own pixel alone). A texture over a buffer with an
//!   aliasing image renders into that image: the buffer's bytes, with no copy; but a pass that
//!   does not store it renders through an image its texture keeps, from which nothing is copied
//!   back either (`kept_bytes.rs`).
//! - **Two ways.** An instance that writes through a shadowed texture (in a draw, or as an
//!   attachment it stores) and also writes its bytes another way keeps the texture before it
//!   and merges it after it, as `shadow.rs` does for a command: the merge goes between it and the
//!   next instance, or as the encoder ends.
//! - **Tile dispatches** (`render/tile_dispatch.rs`). A dispatch records between the instance
//!   before it and the one after, outside both, so it gathers its own shadowed textures and bytes,
//!   and [`plan`] takes it in the pass's order among the instances, with the attachments a use of
//!   it too (it stages them): its copies go between the instance before it and it, and what it
//!   wrote through a texture is copied back before the next command that touches the bytes another
//!   way, or as the encoder ends.
//! - **Executions** (`render/executions.rs`). An execution of render commands records its pass
//!   apart, as a segment of the encoder; the copies before the pass and as the encoder ends stay
//!   the encoder's, and the copies between its instances are the segment's, each between its
//!   two instances as above. So the call gathers, into its instance, everything shadowed its
//!   commands may use: the shadowed textures of the tables they take and those the encoder made
//!   resident (writing where a use may write), and the bytes of the buffers they may reach, the
//!   resident buffers and heaps and, where the commands inherit the encoder's buffers, those it
//!   bound (writing where the usage may, an inherited one always). Refused
//!   (`NotImplemented(IndirectRenderCommands)`): a command's draw that uses a shadowed texture its
//!   call did not gather (none can: its tables and residency are the call's).

use crate::buffer::Buffer;
use crate::clear_reach::DrawnArea;
use crate::encoder::EncoderCore;
use crate::error::Error;
use crate::render_pass::{ColorTarget, PassLayout, Target};
use crate::render_target;
use crate::render_values::{LoadAction, StoreAction};
use crate::residency::Used;
use crate::resource::ResourceKey;
use crate::shadow::{self, ShadowCopies, ShadowCopy, ShadowUse, Within};
use crate::tracker::{Access, Use, UseRange};
use crate::work_counts::{ColorTaken, Counters};
use crate::Device;
use ash::vk;
use metal2vulkan::metal::TextureType;
use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;

/// The most ranges an instance keeps of one buffer's reads, or of its writes, before the two
/// nearest merge, as the tracker bounds its entries (`tracker::ENTRIES_PER_KEY`): the gap between
/// them then counts as used, which can only add a copy, or a merge where a texture's bytes lay
/// in the gap.
const RANGES_PER_BUFFER: usize = 16;

/// What one render pass instance's draws gathered (the module docs).
#[derive(Default)]
pub(super) struct InstanceShadows {
    /// The shadowed textures the draws use, each once.
    shadows: Vec<ShadowUse>,
    /// The bytes of buffers the draws use, by buffer and whether they write: ranges in order,
    /// apart, merged where they meet (at most [`RANGES_PER_BUFFER`]).
    bytes: BTreeMap<(ResourceKey, bool), Vec<Range<u64>>>,
    /// The images the draws use, by image and whether they write: the aspects, levels and
    /// layers of every such use, one range spanning them all (which can only add a copy, or
    /// refuse an instance that writes one image apart two ways), for the shadows of images'
    /// views of another layout (`texture/view_shadow.rs`).
    images: BTreeMap<(ResourceKey, bool), UseRange>,
    /// The uses the last [`add`](InstanceShadows::add) gathered, in order: a use equal to the
    /// one at its position there is gathered already (gathering one again changes nothing, as
    /// each range only grows), so a draw that uses what the last one did gathers nothing new.
    last: Vec<Use<ResourceKey>>,
}

/// `range` of the buffer `key` as a use, read or written (`write`), for `shadow.rs`, which reads
/// a use's buffer, its range, and whether it writes, and nothing else of it.
fn bytes_use(key: ResourceKey, write: bool, range: Range<u64>) -> Use<ResourceKey> {
    let access = Access::new(
        vk::PipelineStageFlags2::ALL_COMMANDS,
        if write {
            vk::AccessFlags2::MEMORY_WRITE
        } else {
            vk::AccessFlags2::MEMORY_READ
        },
    );
    let range = UseRange::Bytes(range);
    if write {
        Use::writes(key, false, range, access)
    } else {
        Use::reads(key, false, range, access)
    }
}

/// One range spanning the image ranges `a` and `b`: their aspects, and the levels and layers from
/// the first of either to the last.
fn span(a: &UseRange, b: &UseRange) -> UseRange {
    match (a, b) {
        (
            UseRange::Image {
                aspects,
                levels,
                layers,
            },
            UseRange::Image {
                aspects: b_aspects,
                levels: b_levels,
                layers: b_layers,
            },
        ) => UseRange::Image {
            aspects: *aspects | *b_aspects,
            levels: levels.start.min(b_levels.start)..levels.end.max(b_levels.end),
            layers: layers.start.min(b_layers.start)..layers.end.max(b_layers.end),
        },
        // An image's uses are all of one kind; another kind on one key takes the later.
        (_, b) => b.clone(),
    }
}

/// `range` of the resource `key` as a use, read or written (`write`), as [`bytes_use`] makes one
/// of bytes.
fn range_use(key: ResourceKey, write: bool, range: UseRange) -> Use<ResourceKey> {
    let access = Access::new(
        vk::PipelineStageFlags2::ALL_COMMANDS,
        if write {
            vk::AccessFlags2::MEMORY_WRITE
        } else {
            vk::AccessFlags2::MEMORY_READ
        },
    );
    if write {
        Use::writes(key, false, range, access)
    } else {
        Use::reads(key, false, range, access)
    }
}

/// Add `range` to `ranges` (in order, apart): merged with those it meets or touches, and past
/// [`RANGES_PER_BUFFER`] the two nearest merged.
fn insert(ranges: &mut Vec<Range<u64>>, mut range: Range<u64>) {
    // Room for as many as it holds before two merge, made once: a buffer whose draws move along
    // it would grow the list several times.
    if ranges.capacity() == 0 {
        ranges.reserve_exact(RANGES_PER_BUFFER + 1);
    }
    ranges.retain(|other| {
        let meets = other.start <= range.end && range.start <= other.end;
        if meets {
            range = range.start.min(other.start)..range.end.max(other.end);
        }
        !meets
    });
    let at = ranges.partition_point(|other| other.start < range.start);
    ranges.insert(at, range);
    if ranges.len() > RANGES_PER_BUFFER {
        let nearest = (1..ranges.len()).min_by_key(|&i| ranges[i].start - ranges[i - 1].end);
        if let Some(i) = nearest {
            let next = ranges.remove(i);
            ranges[i - 1].end = next.end;
        }
    }
}

impl InstanceShadows {
    /// Whether the draws use no shadowed texture.
    pub(super) fn is_empty(&self) -> bool {
        self.shadows.is_empty()
    }

    /// Whether each of `shadows` is among the gathered textures, writing only where the gathered
    /// use may: an execution's draw uses only what its call gathered (`render/executions.rs`).
    pub(super) fn covers(&self, shadows: &[ShadowUse]) -> bool {
        shadows.iter().all(|shadow| {
            self.shadows.iter().any(|had| {
                had.copies.shadow_key == shadow.copies.shadow_key && (had.writes || !shadow.writes)
            })
        })
    }

    /// Gather the bytes of `buffer`, the whole of it, read or written (`write`): what an
    /// execution's commands may reach of a buffer (the module docs).
    pub(super) fn add_bytes(&mut self, buffer: &Buffer, write: bool) {
        let (identity, range) = buffer.tracked(0..buffer.length());
        let used = bytes_use(identity.key, write, range);
        self.add(&[], &[used]);
    }

    /// Gather a recorded draw's `shadows` and `uses`.
    pub(super) fn add(&mut self, shadows: &[ShadowUse], uses: &[Use<ResourceKey>]) {
        for shadow in shadows {
            shadow::join(&mut self.shadows, shadow);
        }
        let last = &self.last;
        let gathered = |(at, used): &(usize, &Use<ResourceKey>)| last.get(*at) == Some(*used);
        let fresh = uses.iter().enumerate().filter(|at| !gathered(at));
        for (_, used) in fresh {
            for (write, access) in [(false, used.read), (true, used.write)] {
                if access.is_empty() {
                    continue;
                }
                match &used.range {
                    UseRange::Bytes(range) => {
                        let ranges = self.bytes.entry((used.key, write)).or_default();
                        insert(ranges, range.clone());
                    }
                    image => {
                        self.images
                            .entry((used.key, write))
                            .and_modify(|held| *held = span(held, image))
                            .or_insert_with(|| image.clone());
                    }
                }
            }
        }
        // Its storage is kept from one draw to the next.
        self.last.clear();
        self.last.extend_from_slice(uses);
    }

    /// The gathered bytes and images as uses.
    fn uses(&self) -> impl Iterator<Item = Use<ResourceKey>> + '_ {
        let bytes = self.bytes.iter().flat_map(|(&(key, write), ranges)| {
            ranges
                .iter()
                .map(move |range| bytes_use(key, write, range.clone()))
        });
        let images = self
            .images
            .iter()
            .map(|(&(key, write), range)| range_use(key, write, range.clone()));
        bytes.chain(images)
    }
}

/// The shadows of the shadowed textures a pass of `layout` renders or resolves into, and of those
/// its instances' draws and its tile dispatches use (`gathered`), each once: the truths whose
/// copies go around its instances.
pub(super) fn pass_shadows<'a>(
    layout: &PassLayout,
    gathered: impl IntoIterator<Item = &'a InstanceShadows>,
) -> Vec<Arc<ShadowCopies>> {
    let attachments = attachment_shadows(layout, true, None, None);
    let used = (attachments.iter()).chain(gathered.into_iter().flat_map(|g| g.shadows.iter()));
    let mut shadows: Vec<Arc<ShadowCopies>> = Vec::new();
    for used in used {
        if !shadows.iter().any(|had| Arc::ptr_eq(had, &used.copies)) {
            shadows.push(Arc::clone(&used.copies));
        }
    }
    shadows
}

/// Whether `used`, a resource an execution's commands may reach that its call did not gather (the
/// module docs), may touch the truth of one of `shadows` (a pass's, [`pass_shadows`]): it is a
/// shadowed texture, or a texture, buffer or heap whose image or bytes meet one's truth, the
/// device's host aliases joining bytes that share memory. An indirect command buffer's records lie
/// in no texture.
pub(super) fn meets_truths(device: &Device, shadows: &[Arc<ShadowCopies>], used: &Used) -> bool {
    let reached = match used {
        Used::IndirectCommandBuffer(_) => return false,
        Used::Texture(texture) if texture.shadow().is_some() => return true,
        Used::Texture(texture) => texture.bound_use(Access::NONE, Access::NONE),
        Used::Buffer(buffer) | Used::Heap { bytes: buffer, .. } => {
            let (identity, range) = buffer.tracked(0..buffer.length());
            bytes_use(identity.key, true, range)
        }
    };
    let reached = [reached];
    let augmented = device.shared().aliases.augment(&reached);
    let reached = augmented.as_deref().unwrap_or(&reached);
    (reached.iter()).any(|used| shadows.iter().any(|shadow| shadow.meets(used)))
}

/// The copies before each of `instances` of a pass of `layout` (its render pass instances and
/// tile dispatches, in order), planned through `core`'s shadowed textures
/// (`EncoderCore::plan_shadows`), the attachments a use of each, a pass whose draws reach tiles
/// where `reached` (`clear_reach::PassReach`), inside `drawn` where its draws' scissors bound
/// them (`clear_reach::DrawnArea::bounds`) and nothing but they write the color attachments it
/// names as loaded by index (their loads as the encoder set them). Each of `instances` comes
/// with the rectangle bounding its own draws' scissors, where `drawn` holds and it bounds them
/// (an instance of draws alone, after the pass's first, which the steps before the pass's
/// instances do not precede): its writes of those colors change their bytes inside it alone
/// ([`bounded_color`], `shadow::Within`), and an attachment's shadow is copied back there.
/// `area` is the draws' scissor rectangles `drawn` bounds, which the first instance's colors
/// are counted against (`WorkCounts::shadow_attachment_tiles_covered`).
pub(super) fn plan<'a>(
    core: &mut EncoderCore,
    instances: impl Iterator<Item = (&'a InstanceShadows, Option<[u32; 4]>)>,
    layout: &PassLayout,
    (reached, drawn): (bool, Option<([u32; 4], &[bool])>),
    area: Option<&DrawnArea>,
) -> Result<Vec<Vec<ShadowCopy>>, Error> {
    let attached = render_target::attachment_uses(layout)?;
    // The colors whose writes an instance bounds, and the uses of the other attachments.
    let bounded: Vec<usize> = (layout.colors.iter().enumerate())
        .filter(|&(index, color)| {
            let loaded = drawn.is_some_and(|(_, loaded)| loaded.get(index) == Some(&true));
            color
                .as_ref()
                .is_some_and(|color| loaded && bounded_color(color, layout))
        })
        .map(|(index, _)| index)
        .collect();
    let others = render_target::attachment_uses_but(layout, |index| bounded.contains(&index))?;
    let mut writes = Vec::with_capacity(bounded.len());
    for color in bounded
        .iter()
        .filter_map(|&index| layout.colors[index].as_ref())
    {
        let texture = &color.target.texture;
        let uses = render_target::target_uses(&color.target, false)?;
        writes.push((uses, texture.usage_shadow_copies()));
    }
    let mut copies = Vec::new();
    for (index, (instance, bound)) in instances.enumerate() {
        let bound = bound.filter(|_| drawn.is_some());
        let attachments = attachment_shadows(layout, reached, drawn, bound);
        if index == 0 {
            for shadow in &attachments {
                core.shadow_attached(shadow.copies.shadow_key);
            }
            count_colors(&core.device().shared().counts, layout, reached, drawn, area);
        }
        let mut shadows = Vec::with_capacity(attachments.len() + instance.shadows.len());
        for shadow in attachments.iter().chain(&instance.shadows) {
            shadow::join(&mut shadows, shadow);
        }
        let uses: Vec<Use<ResourceKey>> = instance.uses().chain(attached.iter().cloned()).collect();
        copies.push(core.plan_shadows(&shadows, &uses)?);
        match bound.filter(|_| !writes.is_empty()) {
            Some(area) => {
                let uses: Vec<Use<ResourceKey>> =
                    instance.uses().chain(others.iter().cloned()).collect();
                let within: Vec<Within> = (writes.iter())
                    .map(|(uses, texels)| Within {
                        uses: uses.clone(),
                        texels: texels.clone(),
                        area,
                    })
                    .collect();
                core.shadows_used(&shadows, &uses, &within);
            }
            None => core.shadows_used(&shadows, &uses, &[]),
        }
    }
    Ok(copies)
}

/// Whether a pass of `layout` has a shadowed attachment or resolve target.
pub(super) fn attachments_shadowed(layout: &PassLayout) -> bool {
    !attachment_shadows(layout, true, None, None).is_empty()
}

/// Whether the pass changes a texel of `color` (as the encoder set its load, which a loaded
/// color's is) only inside its draws' scissors, at the texel of the pixel: a single-sampled
/// color it loads, resolves nowhere, renders itself (no stand-in, no sample planes), the whole
/// of a one-level, one-slice 2D texture. Its load and store write back what they read
/// (`VK_ATTACHMENT_LOAD_OP_LOAD`: "the previous contents of the image within the render area
/// will be preserved as the initial values"; `VK_ATTACHMENT_STORE_OP_STORE`: "the contents
/// generated during the render pass and within the render area are written to memory"), and a
/// fragment's samples outside the scissor rectangle "will have their coverage set to 0" (the
/// Fragment Operations chapter's Scissor Test).
fn bounded_color(color: &ColorTarget, layout: &PassLayout) -> bool {
    layout.samples == 1
        && color.load == LoadAction::Load
        && color.resolve.is_none()
        && color.copied_resolve.is_none()
        && color.stand_in.is_none()
        && color.planes.is_none()
        && one_image(&color.target)
}

/// A color attachment of `layout`'s pass that the pass loads and stores and loads other than by
/// the inverse of its store (a rotated texture's frame of rotation 4, `rotated.rs`), named as a
/// degrade notice names it: where a pass whose draws reach the tiles their primitives touch
/// turns the tiles no draw reaches, which Metal keeps.
pub(super) fn loaded_apart(layout: &PassLayout) -> Option<String> {
    layout.colors.iter().enumerate().find_map(|(index, color)| {
        let color = color.as_ref()?;
        let stores = matches!(
            color.store,
            StoreAction::Store | StoreAction::StoreAndMultisampleResolve | StoreAction::Unknown
        );
        let texture = &color.target.texture;
        let apart = texture.shadow().is_some_and(|copies| copies.loads_apart());
        (color.load == LoadAction::Load && stores && apart).then(|| {
            format!(
                "color attachment {index} of a render pass, a {}x{} {:?} frame of a rotated \
                     texture, loaded",
                texture.width(),
                texture.height(),
                texture.pixel_format()
            )
        })
    })
}

/// A color attachment of `layout`'s pass that the pass loads and stores and that is a two-plane
/// YCbCr texture's render shadow, which stores every texel of the tiles its copy back covers
/// (`ycbcr.rs`), named as a degrade notice names it.
pub(super) fn ycbcr_loaded(layout: &PassLayout) -> Option<String> {
    layout.colors.iter().enumerate().find_map(|(index, color)| {
        let color = color.as_ref()?;
        let stores = matches!(
            color.store,
            StoreAction::Store | StoreAction::StoreAndMultisampleResolve | StoreAction::Unknown
        );
        let texture = &color.target.texture;
        let rendered = texture
            .shadow()
            .is_some_and(|copies| copies.left_untouched())
            && metal2vulkan::metal::format::ycbcr_renders(texture.pixel_format());
        (color.load == LoadAction::Load && stores && rendered).then(|| {
            format!(
                "color attachment {index} of a render pass, a {}x{} {:?} texture, loaded",
                texture.width(),
                texture.height(),
                texture.pixel_format()
            )
        })
    })
}

/// The shadowed textures among a pass's attachments and resolve targets: a store writes through
/// one (an unknown store action counts as one), and so does a resolve. A depth or stencil
/// attachment's are heap shadows (`texture/heap.rs`): its texture, each aspect's where the two are
/// different textures, the texture a stand-in renders for, and its resolve targets. A color
/// attachment of a pass whose draws reach no tile (`reached` false) is left in place
/// ([`ShadowUse::kept`]). A color attachment `drawn` names as loaded changes only inside its
/// rectangle, which bounds the draws' scissors ([`ShadowUse::area`]), and in an instance whose
/// draws' scissors `instance` bounds, only inside that.
fn attachment_shadows(
    layout: &PassLayout,
    reached: bool,
    drawn: Option<([u32; 4], &[bool])>,
    instance: Option<[u32; 4]>,
) -> Vec<ShadowUse> {
    let mut shadows = Vec::new();
    let stores = |store: StoreAction| {
        matches!(
            store,
            StoreAction::Store | StoreAction::StoreAndMultisampleResolve | StoreAction::Unknown
        )
    };
    if let Some(ds) = &layout.depth_stencil {
        let writes = [&ds.depth, &ds.stencil]
            .into_iter()
            .flatten()
            .any(|aspect| stores(aspect.store));
        // Every aspect of the texture clears over the whole of it (`ShadowUse::overwrites`).
        let overwrites = ds.separate.is_none()
            && ds.stand_in.is_none()
            && !ds.keep_depth
            && !ds.keep_stencil
            && [&ds.depth, &ds.stencil]
                .into_iter()
                .flatten()
                .all(|aspect| aspect.load == LoadAction::Clear)
            && whole(&ds.target, layout);
        let own = ds.stand_in.as_ref().map(|s| &s.own);
        let targets: Vec<&crate::render_pass::Target> = match &ds.separate {
            Some(separate) => separate.targets().collect(),
            None => vec![&ds.target],
        };
        for target in targets.into_iter().chain(own) {
            if let Some(copies) = target.texture.shadow() {
                let used = ShadowUse {
                    copies: Arc::clone(copies),
                    writes,
                    overwrites,
                    kept: false,
                    area: None,
                    loads: None,
                    tile_stores: false,
                };
                shadow::join(&mut shadows, &used);
            }
        }
        let resolves = ds
            .resolve
            .iter()
            .flat_map(|r| r.written())
            .flat_map(|(resolve, _)| std::iter::once(&resolve.attachment).chain(&resolve.copy_to));
        for target in resolves {
            if let Some(copies) = target.texture.shadow() {
                let used = ShadowUse {
                    copies: Arc::clone(copies),
                    writes: true,
                    overwrites: false,
                    kept: false,
                    area: None,
                    loads: None,
                    tile_stores: false,
                };
                shadow::join(&mut shadows, &used);
            }
        }
    }
    let colors = layout.colors.iter().enumerate();
    for (index, color) in colors.filter_map(|(index, color)| Some((index, color.as_ref()?))) {
        let stores = matches!(
            color.store,
            StoreAction::Store | StoreAction::StoreAndMultisampleResolve | StoreAction::Unknown
        );
        let ColorTake {
            overwrites,
            untouched,
            area,
        } = color_take(layout, index, color, reached, drawn);
        let own = color.stand_in.as_ref().map(|s| &s.own);
        for target in std::iter::once(&color.target).chain(own) {
            if let Some(copies) = target.texture.shadow() {
                if untouched && copies.left_untouched() {
                    continue;
                }
                let used = ShadowUse {
                    copies: Arc::clone(copies),
                    writes: stores,
                    overwrites,
                    kept: !reached,
                    area: area.map(|area| instance.unwrap_or(area)),
                    loads: area,
                    // A color it loads and stores inside its draws' scissors: Metal's store
                    // writes back each tile its draws reach whole (`shadow.rs`).
                    tile_stores: stores && area.is_some(),
                };
                shadow::join(&mut shadows, &used);
            }
        }
        let resolves = color
            .resolve
            .iter()
            .flat_map(|r| std::iter::once(&r.attachment).chain(&r.copy_to))
            .chain(&color.copied_resolve);
        for target in resolves {
            if let Some(copies) = target.texture.shadow() {
                let used = ShadowUse {
                    copies: Arc::clone(copies),
                    writes: true,
                    overwrites: false,
                    kept: false,
                    area: None,
                    loads: None,
                    tile_stores: false,
                };
                shadow::join(&mut shadows, &used);
            }
        }
    }
    shadows
}

/// How a pass takes a color attachment ([`color_take`]).
struct ColorTake {
    /// It clears every texel of the attachment's texture ([`ShadowUse::overwrites`]).
    overwrites: bool,
    /// It changes no texel and reads none: a pass that reaches no tile and loads it.
    untouched: bool,
    /// The rectangle of texels it changes the attachment inside, and reads it inside but at its
    /// load, which stores back what it read ([`ShadowUse::area`], [`ShadowUse::loads`]); `None`
    /// for anywhere.
    area: Option<[u32; 4]>,
}

/// How a pass of `layout` that reaches tiles where `reached`, inside `drawn` where its draws'
/// scissors bound them (with the colors it loads by index, as [`plan`] takes them), takes its
/// color attachment `color`, at `index`.
fn color_take(
    layout: &PassLayout,
    index: usize,
    color: &ColorTarget,
    reached: bool,
    drawn: Option<([u32; 4], &[bool])>,
) -> ColorTake {
    let overwrites = color.load == LoadAction::Clear
        && color.stand_in.is_none()
        && color.planes.is_none()
        && whole(&color.target, layout);
    // A pass that reaches no tile and loads the attachment leaves its texels as they were:
    // Metal loads and stores only the tiles a draw reaches (`clear_reach::PassReach`), and
    // Vulkan's load and store write back what they read. Nothing resolves from it, so the
    // pass needs none of the texels: a texture over a buffer's bytes in its own image takes
    // no copy in, and nothing is copied back.
    let untouched = !reached
        && color.load == LoadAction::Load
        && matches!(color.store, StoreAction::Store | StoreAction::DontCare)
        && color.resolve.is_none()
        && color.copied_resolve.is_none()
        && color.stand_in.is_none()
        && color.planes.is_none();
    // A pass that loads the attachment changes no texel outside its draws' scissors: a
    // fragment's samples outside the scissor rectangle "will have their coverage set to 0"
    // (the Fragment Operations chapter's Scissor Test), and its load and store write back
    // what they read (as above). Each Vulkan scissor is Metal's rectangle clipped to the
    // pixels its viewport overlaps (`render_state.rs`), which Metal's draw writes no pixel
    // past, and `drawn` bounds them all. So only the texels inside it are copied back, of
    // the attachment's texture where it is the whole of a one-level, one-slice texture.
    // Nor does the pass read one outside it into anything but the image: the load reads
    // every texel, but the store writes it back unchanged, a fragment reads the attachment
    // at its own pixel alone (a blend, framebuffer fetch, programmable blending, an
    // imageblock), and the area is unbounded wherever a fragment's write is not bounded so
    // (a rasterization rate map, an execution of render commands) or something reads past
    // its pixel (a tile dispatch reads the whole tile; a resolve reads every sample, and is
    // excluded below). So only those texels are copied in (`ShadowUse::loads`); a draw that
    // takes the texture another way (bound, or made resident) joins a use that reads
    // anywhere, and a later command that needs the texels outside the rectangle copies them in,
    // in this encoder or a later one of the command buffer (`shadow::ShadowCarry`).
    let loaded = drawn.filter(|(_, loaded)| loaded.get(index) == Some(&true));
    let area = loaded
        .map(|(area, _)| area)
        .filter(|_| bounded_color(color, layout));
    ColorTake {
        overwrites,
        untouched,
        area,
    }
}

/// Count how a pass of `layout` takes each of its color attachments whose texture is shadowed
/// (`WorkCounts::shadow_attachment_loads_within` and kin), as [`attachment_shadows`] plans them.
fn count_colors(
    counts: &Counters,
    layout: &PassLayout,
    reached: bool,
    drawn: Option<([u32; 4], &[bool])>,
    area: Option<&DrawnArea>,
) {
    let tiles = area.and_then(DrawnArea::covered_tiles);
    let colors = layout.colors.iter().enumerate();
    for (index, color) in colors.filter_map(|(index, color)| Some((index, color.as_ref()?))) {
        let Some(copies) = color.target.texture.shadow() else {
            continue;
        };
        let take = color_take(layout, index, color, reached, drawn);
        let taken = match color.load {
            _ if take.untouched && copies.left_untouched() => ColorTaken::Untouched,
            _ if take.overwrites => ColorTaken::Cleared,
            LoadAction::Clear => ColorTaken::ClearedInPart,
            _ if take.area.is_some() => ColorTaken::LoadedWithin,
            _ => ColorTaken::Loaded,
        };
        counts.shadow_color_taken(taken);
        if let (ColorTaken::LoadedWithin, Some((covered, bounded))) = (taken, tiles) {
            counts.shadow_attachment_tiles(covered, bounded);
        }
    }
}

/// Whether `target` is the whole of its texture, a 2D texture of one level, and the pass's
/// render area covers it: a clear of it replaces every texel.
fn whole(target: &Target, layout: &PassLayout) -> bool {
    one_image(target)
        && [target.texture.width(), target.texture.height()] == layout.extent.map(u64::from)
}

/// Whether `target` is the whole of its texture, a 2D texture of one level and one slice, whose
/// pixels are its texels.
fn one_image(target: &Target) -> bool {
    let texture = &target.texture;
    texture.texture_type() == TextureType::Type2D
        && texture.mipmap_level_count() == 1
        && texture.array_length() == 1
        && target.level == 0
        && target.layer == 0
        && target.layers == 1
}

#[cfg(test)]
mod tests {
    use super::{InstanceShadows, RANGES_PER_BUFFER};
    use crate::resource::ResourceKey;
    use crate::tracker::{Access, Use, UseRange};
    use ash::vk;
    use std::ops::Range;

    fn bytes(key: ResourceKey, range: std::ops::Range<u64>, write: bool) -> Use<ResourceKey> {
        let access = Access::new(
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::SHADER_STORAGE_READ,
        );
        let (read, written) = if write {
            (Access::NONE, access)
        } else {
            (access, Access::NONE)
        };
        Use {
            key,
            tracked: true,
            range: UseRange::Bytes(range),
            read,
            write: written,
        }
    }

    /// An instance keeps each buffer's read bytes and written bytes apart, merged where ranges
    /// meet or touch, so draws over one buffer leave few ranges; uses of images are not bytes and
    /// are left out; past `RANGES_PER_BUFFER` ranges of one buffer, the two nearest merge.
    #[test]
    fn an_instance_merges_the_bytes_its_draws_use() {
        let (a, b, c) = (ResourceKey::new(), ResourceKey::new(), ResourceKey::new());
        let mut gathered = InstanceShadows::default();
        for start in (0..64).step_by(16) {
            gathered.add(&[], &[bytes(a, start..start + 16, false)]);
        }
        gathered.add(&[], &[bytes(a, 100..120, false), bytes(a, 8..24, true)]);
        gathered.add(&[], &[bytes(b, 0..4, false), bytes(a, 110..130, false)]);
        let image = Use {
            range: UseRange::Image {
                aspects: vk::ImageAspectFlags::COLOR,
                levels: 0..1,
                layers: 0..1,
            },
            ..bytes(c, 0..0, true)
        };
        gathered.add(&[], &[image]);
        let keys: Vec<_> = gathered.bytes.keys().copied().collect();
        assert_eq!(keys, [(a, false), (a, true), (b, false)]);
        assert_eq!(gathered.bytes[&(a, false)], [0..64, 100..130]);
        let one = |start, end| [Range { start, end }];
        assert_eq!(gathered.bytes[&(a, true)], one(8, 24));
        assert_eq!(gathered.bytes[&(b, false)], one(0, 4));
        assert!(gathered.is_empty());
        // Seventeen ranges of one buffer 90 bytes apart: the first two merge; then one 20 bytes
        // past the third: it merges with that one.
        let mut bounded = InstanceShadows::default();
        for i in 0..17 {
            bounded.add(&[], &[bytes(a, 100 * i..100 * i + 10, true)]);
        }
        let ranges = &bounded.bytes[&(a, true)];
        assert_eq!(ranges.len(), RANGES_PER_BUFFER);
        assert_eq!(ranges[0], 0..110);
        bounded.add(&[], &[bytes(a, 330..340, true)]);
        let ranges = &bounded.bytes[&(a, true)];
        assert_eq!(ranges.len(), RANGES_PER_BUFFER);
        assert!(ranges.contains(&(300..340)), "{ranges:?}");
        // Its list took room for every range it keeps and the merge's one at its first, and
        // never grew.
        assert_eq!(ranges.capacity(), RANGES_PER_BUFFER + 1);
    }
}
