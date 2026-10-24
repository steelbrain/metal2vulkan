//! An attachment larger than its pass's render area: how far Metal's clear reaches past the
//! render area, and what a `DontCare` load leaves.
//!
//! **Metal**, measured on the M3 Max (macOS 26.7) and an M4 (macOS 27.0.1), every answer the
//! same on both, each case with and without API validation, which takes every one (44 cases). A
//! pass renders over its render area, from the origin (the smallest attachment's extent, or the
//! descriptor's target size), but its attachments' stores reach past it:
//!
//! - **A `Clear` load** of an attachment the pass stores clears it from the origin over the render
//!   area rounded up to whole tiles of [`TILE`] pixels a side, clipped to the attachment's level,
//!   in a pass with a draw call (one whose primitives cover nothing, or of no vertex, included),
//!   and in each layer of a layered pass, with a draw call or not; a pass of one layer without a
//!   draw call clears the whole level and slice. So for every format measured (`R8Unorm` to `RGBA32Float`, 1 to 16 bytes a pixel, and
//!   eight `RGBA8Unorm` or four `RGBA32Float` attachments at 4 samples), at 1, 2 and 4 samples,
//!   with a descriptor's tile of 16x16, for `Depth32Float`, `Depth16Unorm`, `Stencil8` and
//!   `Depth32Float_Stencil8`, at a level or slice past the first, over render areas of 1x1,
//!   33x33, 40x24, 40x40, 96x8 and 100x100, and whichever tiles the draws touch. Separate depth
//!   and stencil textures are measured apart (`depth_stencil_scratch.rs`).
//! - **A `DontCare` load** of an attachment the pass stores keeps its texels in a pass without a
//!   draw (or whose draws draw nothing, or no vertex). Where a draw's primitives touch a tile
//!   (32x32 pixels at 1 and 2 samples, 32x16 at 4, color and depth alike on both chips), the
//!   tile holds zero before the draws (a color, every color attachment of the pass) or the
//!   attachment's clear value (depth, stencil); the tiles no primitive touches keep their texels.
//!   Which tiles those are depends on where the GPU places the vertices.
//! - A `Load` keeps the texels past the render area, and a `DontCare` store stores nothing.
//!
//! **The runtime** gives the `Clear` load's answer and the `DontCare` load's near-equivalent, as
//! the encoder ends ([`dont_care_loads`], [`ReachClears`]): a pass has a draw call where the
//! encoder encoded one, whether it draws or not, and draws where a draw call drew vertices,
//! indices, patches or meshes through a pipeline that rasterizes; an execution of an indirect
//! command buffer's render commands, or a tile dispatch, is taken as a draw call that draws (both
//! unmeasured).
//!
//! - A `DontCare`-loaded attachment the pass stores (but separate depth and stencil textures, and
//!   memoryless ones) loads in a pass whose draws reach no tile, and clears to zero (a color) or
//!   its clear value (depth, stencil) in one whose draws reach every tile, both Metal's answer
//!   ([`PassReach`]). In a pass whose draws reach the tiles their primitives touch, it clears
//!   too: Metal's answer in each tile a draw touches, and in a tile none touches, where Metal
//!   keeps the texels, a near-equivalent announced once per device
//!   (`Degradation::DontCareLoadTiles`); the exact path is not taken. Such a pass clears only the
//!   tiles its draws' viewports and scissors reach ([`DrawnArea`]), at the encoder's tile size (the
//!   pass's `tileWidth` and `tileHeight`, which set Metal's grid, as measured), a clear before the
//!   pass, and loads: every other tile keeps its texels, Metal's answer (a draw writes no pixel
//!   outside them), and a pass whose draws reach no pixel loads, as one whose draws draw nothing.
//! - A `DontCare`-loaded memoryless color attachment clears to zero in a pass whose draws reach a
//!   tile: Metal's answer in every tile a fragment reads it (an Apple M4), and it is
//!   never stored, so the answer is exact.
//! - A pass without a draw call whose attachments all load with `Load` or `DontCare` and store
//!   with `Store` or `DontCare`, resolving nothing, and that sets no depth/stencil state with depth
//!   writes, changes no texel: it records nothing (`render/untouched.rs`), before any of the steps
//!   here.
//! - Each `Clear`-loaded attachment (a converted one too) whose clear reaches past the render
//!   area ([`reach`]) is cleared there before the pass by a render pass of its own: the attachment
//!   alone, loaded and stored, over the pass's layers, `vkCmdClearAttachments` clearing the reach
//!   of its cleared aspects, and the pass then loads it ([`ReachClears::load_in_pass`]). Neither
//!   is a load operation's clear: on a device that renders through Metal (MoltenVK) that clears
//!   a pass without a draw over the whole attachment, as Metal does, so the reach pass's would,
//!   and the pass's own would in a pass whose draw calls draw nothing (lane `clear_reach` on
//!   MoltenVK). A stand-in's copy back to its texture covers the reach (`attachment_stand_in.rs`).
//!   So is a cleared attachment the pass's first instance leaves out where no whole-subresource
//!   clear may stand for its load operation (`render.rs`'s `cleared_ahead`), over the render area
//!   where it stores nothing.
//! - A pass whose first instance records inline clears none before it: that instance, begun at a
//!   draw that records (of a vertex and an instance at least), renders over the reach its colors
//!   share, and its load operations' clear reaches it (`render/inline_first.rs`). Unmeasured on
//!   MoltenVK past the render area.

use super::clear_value;
use super::encoder::EncoderCore;
use super::error::Error;
use super::render_pass::{AspectClear, PassLayout, Target};
use super::render_target::{self, FramebufferObject, RenderPassObject};
use super::render_values::{LoadAction, StoreAction};
use super::resource::ResourceKey;
use super::tracker::{Barrier, Use};
use super::values::StorageMode;
use super::Device;
use ash::vk;
use metal2vulkan::metal::format::ClearRoute;
use metal2vulkan::metal::PixelFormat;
use std::collections::BTreeSet;

/// The side, in pixels, of the tiles a `Clear` load's clear reaches past the render area in, on
/// both axes: Metal's on the M3 Max and an M4, the same at every format, sample count and tile
/// size measured (the module docs). The one place the runtime states it: every device clears as
/// Metal's do.
///
/// **Executed on a driver that is not Metal.** On MoltenVK this constant is unobservable — the
/// driver is Metal and Metal's own tiles produce the reach whether [`reach`] rounds or not, so
/// replacing the rounding with the render area exactly moves no texel of the conformance suite's
/// `pass-bound-clear-reach` on an M3 Max. On llvmpipe (LLVM 23.1.1, x86-64) the runtime must
/// synthesize the reach, the case replays exact, and that same one-line change moves 511, 960 and
/// 960 texels of the case's `area-33x33`, `area-8x40` and `area-40x8` cells. The `8x40` and `40x8`
/// pair is what pins the side to 32 rather than 16 or 64, and pins the rounding to each axis on its
/// own; `area-33x33` is what pins the clip to the attachment. RADV and ANV are unrun.
pub(super) const TILE: u32 = 32;

/// The extent from the origin that a `Clear` load of an attachment of `extent` clears in a pass
/// over the render area `area`: the render area rounded up to whole [`TILE`]s, clipped to the
/// attachment; the whole attachment where `whole` (a pass of one layer without a draw).
pub(super) fn reach(area: [u32; 2], extent: [u32; 2], whole: bool) -> [u32; 2] {
    if whole {
        return extent;
    }
    [0, 1].map(|axis| {
        area[axis]
            .div_ceil(TILE)
            .saturating_mul(TILE)
            .min(extent[axis])
    })
}

/// The extent from the origin that a `Clear` load of an attachment of `extent` clears in
/// `layout`'s pass, a pass with a draw call where `called`: [`reach`] over its render area.
pub(super) fn cleared(layout: &PassLayout, extent: [u32; 2], called: bool) -> [u32; 2] {
    reach(layout.extent, extent, !called && layout.layers == 1)
}

/// Whether Metal's `store` keeps an attachment's texels.
pub(super) fn stores(store: StoreAction) -> bool {
    matches!(
        store,
        StoreAction::Store | StoreAction::StoreAndMultisampleResolve
    )
}

/// The tiles a render pass's draws reach, where Metal loads and stores its attachments, as far as
/// the host decides them: what a `DontCare` load clears (above), which tiles a rotated texture's
/// frame turns (`render/shadows.rs`), and which tiles the load pass of an extended-range or padded
/// attachment rewrites (`render/unused_bits_load.rs`). An execution of an indirect command buffer's
/// render commands, or a tile dispatch, counts as a draw call that draws (unmeasured).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PassReach {
    /// No tile: the pass has no draw call, or its draw calls draw nothing (of no vertex or no
    /// instance, or through a pipeline that rasterizes nothing) and no attachment is cleared
    /// and stored (`nodraw`, `instances0_t11`, `no_raster_t11`, `clear_c1_store_nodraw`).
    Nothing,
    /// Every tile, within the render area rounded up to whole tiles: a pass of one layer with a
    /// draw call (one that draws nothing too) and an attachment loaded `Clear` and stored (a color,
    /// depth and stencil attachment, `clear_c1_store_instances0` and `clear_depth_store_instances0`
    /// of no instance). A `Clear` load with another store, a separate depth or stencil texture's,
    /// and a pass of more layers are unmeasured: none of them decides it.
    Every,
    /// The tiles the draws' primitives reach, which the host does not know: each use of it
    /// takes every tile and announces it (until the reach is captured).
    Draws,
}

impl PassReach {
    /// The reach of `layout`'s pass, whose store actions are settled and whose `DontCare` loads
    /// are not yet given their answers, with a draw call where `called` and one that draws where
    /// `drew`.
    pub(super) fn of(layout: &PassLayout, called: bool, drew: bool) -> PassReach {
        let cleared = |load: LoadAction, store: StoreAction, memoryless: bool| {
            load == LoadAction::Clear && store == StoreAction::Store && !memoryless
        };
        let colors = layout
            .colors
            .iter()
            .flatten()
            .any(|c| cleared(c.load, c.store, c.memoryless));
        let depth_stencil = layout
            .depth_stencil
            .as_ref()
            .filter(|ds| ds.separate.is_none() && !ds.hidden)
            .is_some_and(|ds| {
                [ds.depth, ds.stencil]
                    .into_iter()
                    .flatten()
                    .any(|a| cleared(a.load, a.store, a.memoryless))
            });
        if !called {
            PassReach::Nothing
        } else if layout.layers == 1 && (colors || depth_stencil) {
            PassReach::Every
        } else if drew {
            PassReach::Draws
        } else {
            PassReach::Nothing
        }
    }
}

/// Whether `target`'s texture is memoryless.
fn memoryless(target: &Target) -> bool {
    target.texture.storage_mode() == StorageMode::Memoryless
}

/// The clear value a `DontCare` load that clears gives a color attachment of `pf`: the format's
/// zero as Metal's tile holds it, the texel a `Clear` load of `(0, 0, 0, 0)` writes
/// (`clear_value::color`). An extended-range format's is the code of 0 (384), which reads 0,
/// where zero bits read `-384/510` in color (plain Metal on an M4: a `DontCare`-loaded
/// `RGB10A8_2P_XR10` attachment's cleared texels read 0, the runtime's zero bits read -0.7529);
/// every format held directly takes zero bits.
pub(super) fn dont_care_clear(pf: PixelFormat) -> vk::ClearColorValue {
    // A bfloat zero is the code 0 by either release's rule.
    clear_value::color(pf, [0.0; 4], ClearRoute::ThroughHalf).unwrap_or_default()
}

/// Give each `DontCare` load of `layout` that the module docs name its answer in a pass of
/// `reach`: `Load` where it reaches no tile, else `Clear`, to the format's zero for a color
/// attachment ([`dont_care_clear`]) and to its clear value for depth and stencil. Which attachments took it, and whether the pass took
/// the near-equivalent (its draws reach the tiles their primitives touch): the attachment it
/// names, for the notice.
pub(super) fn dont_care_loads(layout: &mut PassLayout, reach: PassReach) -> DontCareLoads {
    let load = match reach {
        PassReach::Nothing => LoadAction::Load,
        PassReach::Every | PassReach::Draws => LoadAction::Clear,
    };
    let drew = reach == PassReach::Draws;
    let mut loads = DontCareLoads {
        named: None,
        colors: Vec::new(),
        depth: false,
        stencil: false,
    };
    for (index, color) in layout.colors.iter_mut().enumerate() {
        let Some(color) = color.as_mut() else {
            continue;
        };
        // A memoryless color reads zero in every tile a draw touches, the only tiles where a
        // fragment reads it (an Apple M4: a framebuffer fetch of one read 0 in
        // 52 runs of 52, also after a pass that cleared other bytes into the tiles), and it is
        // never stored: a clear to zero is Metal's answer exactly, in a pass that draws.
        if color.load == LoadAction::DontCare && color.memoryless {
            if reach != PassReach::Nothing {
                color.load = LoadAction::Clear;
                color.clear = dont_care_clear(color.target.texture.pixel_format());
            }
            continue;
        }
        if color.load != LoadAction::DontCare || color.memoryless || !stores(color.store) {
            continue;
        }
        color.load = load;
        color.clear = dont_care_clear(color.target.texture.pixel_format());
        loads.colors.push(index);
        if drew {
            loads
                .named
                .get_or_insert_with(|| format!("color attachment {index}"));
        }
    }
    if let Some(ds) = layout
        .depth_stencil
        .as_mut()
        .filter(|ds| ds.separate.is_none())
    {
        for (name, aspect, taken) in [
            ("depth", ds.depth.as_mut(), &mut loads.depth),
            ("stencil", ds.stencil.as_mut(), &mut loads.stencil),
        ] {
            let Some(aspect) = aspect else {
                continue;
            };
            if aspect.load != LoadAction::DontCare || aspect.memoryless || !stores(aspect.store) {
                continue;
            }
            aspect.load = load;
            *taken = true;
            if drew {
                loads
                    .named
                    .get_or_insert_with(|| format!("the {name} attachment"));
            }
        }
    }
    loads
}

/// What [`dont_care_loads`] did: the attachment it names for the notice where the pass took the
/// near-equivalent, and the color attachments (by index) and depth and stencil aspects whose
/// `DontCare` load it gave an answer.
pub(super) struct DontCareLoads {
    pub(super) named: Option<String>,
    colors: Vec<usize>,
    depth: bool,
    stencil: bool,
}

impl DontCareLoads {
    /// Whether [`dont_care_loads`] gave color attachment `index`'s `DontCare` load an answer.
    pub(super) fn color(&self, index: usize) -> bool {
        self.colors.contains(&index)
    }

    /// Whether it gave the depth and the stencil aspects' `DontCare` loads an answer.
    pub(super) fn aspects(&self) -> [bool; 2] {
        [self.depth, self.stencil]
    }
}

/// The most distinct pixel rectangles [`DrawnArea`] keeps; past them it is the whole render area.
const MOST_RECTS: usize = 1024;

/// The words of a row of tiles in [`DrawnArea::covered_tiles`]: 512 tiles, the 16384 pixels of
/// the widest render area.
const ROW_WORDS: usize = 8;

/// The pixels a pass's draws may write, as far as the host bounds them: each draw that draws
/// writes only inside its viewports' pixels within their scissor rectangles
/// (`EncoderState::vk_scissors`), so a tile none of those reaches is one Metal's draws do not,
/// and keeps its texels (on an Apple M4 under macOS 27.0.1: a sub-rectangle of the
/// viewport or of the scissor zeroes exactly the `DontCare`-loaded tiles it overlaps, and every
/// other tile keeps its texels). The whole render area where a draw's pixels are not bounded so:
/// an execution of render commands, a tile dispatch, a pass through a rasterization-rate map, or
/// more than [`MOST_RECTS`] distinct rectangles.
#[derive(Clone, Debug, Default)]
pub(super) struct DrawnArea {
    /// Each distinct `[x0, y0, x1, y1]`, half-open, inside the render area.
    rects: Rects,
    whole: bool,
}

/// Distinct pixel rectangles: the first held in place, as a pass's draws mostly reach one (its
/// render area), so such a pass allocates none for them; the others in a set.
#[derive(Clone, Debug, Default)]
struct Rects {
    first: Option<[u32; 4]>,
    /// Empty while `first` is `None`; never holds `first`.
    rest: BTreeSet<[u32; 4]>,
}

impl Rects {
    /// Add `rect`, where it is not here yet.
    fn insert(&mut self, rect: [u32; 4]) {
        match self.first {
            None => self.first = Some(rect),
            Some(first) if first == rect => {}
            Some(_) => {
                self.rest.insert(rect);
            }
        }
    }

    fn len(&self) -> usize {
        usize::from(self.first.is_some()) + self.rest.len()
    }

    fn iter(&self) -> impl Iterator<Item = &[u32; 4]> {
        self.first.iter().chain(&self.rest)
    }

    fn clear(&mut self) {
        self.first = None;
        self.rest.clear();
    }
}

impl DrawnArea {
    /// An area that bounds nothing, an execution's commands' (which the host does not bound).
    pub(super) fn unbounded() -> DrawnArea {
        DrawnArea {
            rects: Rects::default(),
            whole: true,
        }
    }

    /// Add a draw's rectangles, its scissors clipped to its viewports' pixels.
    pub(super) fn add(&mut self, rects: &[vk::Rect2D]) {
        if self.whole {
            return;
        }
        for r in rects {
            let (x, y) = (r.offset.x.max(0) as u32, r.offset.y.max(0) as u32);
            let (w, h) = (r.extent.width, r.extent.height);
            if w == 0 || h == 0 {
                continue;
            }
            self.rects
                .insert([x, y, x.saturating_add(w), y.saturating_add(h)]);
        }
        if self.rects.len() > MOST_RECTS {
            self.everywhere();
        }
    }

    /// The rectangle `[x0, y0, x1, y1)` bounding every rectangle, half-open in the render area's
    /// pixels: `None` where the area bounds nothing, or holds no rectangle.
    pub(super) fn bounds(&self) -> Option<[u32; 4]> {
        if self.whole {
            return None;
        }
        self.rects
            .iter()
            .copied()
            .reduce(|[a0, a1, a2, a3], [b0, b1, b2, b3]| {
                [a0.min(b0), a1.min(b1), a2.max(b2), a3.max(b3)]
            })
    }

    /// Of the tiles of [`TILE`] pixels a side, from the render area's origin, that
    /// [`bounds`](Self::bounds) reaches, how many its rectangles reach, and how many it reaches:
    /// `None` where it bounds nothing. A count (`WorkCounts::shadow_attachment_tiles_covered`):
    /// what a copy cut to the rectangles' tiles would leave out of one cut to their bounds.
    pub(super) fn covered_tiles(&self) -> Option<(u64, u64)> {
        let [x0, y0, x1, y1] = self.bounds()?;
        let (tx0, ty0) = (x0 / TILE, y0 / TILE);
        let (tx1, ty1) = (x1.div_ceil(TILE), y1.div_ceil(TILE));
        let bounded = u64::from(tx1 - tx0) * u64::from(ty1 - ty0);
        // A row of tiles in the bits of `ROW_WORDS` words; a wider one, which no render area
        // has, counts as covered.
        if (tx1 - tx0) as usize > ROW_WORDS * 64 {
            return Some((bounded, bounded));
        }
        let mut covered = 0;
        for ty in ty0..ty1 {
            let (top, bottom) = (ty * TILE, (ty + 1) * TILE);
            let mut row = [0u64; ROW_WORDS];
            for &[rx0, ry0, rx1, ry1] in self.rects.iter() {
                if rx0 >= rx1 || ry0 >= ry1 || ry0 >= bottom || ry1 <= top {
                    continue;
                }
                for tile in rx0 / TILE - tx0..rx1.div_ceil(TILE) - tx0 {
                    row[(tile / 64) as usize] |= 1 << (tile % 64);
                }
            }
            covered += row
                .iter()
                .map(|word| u64::from(word.count_ones()))
                .sum::<u64>();
        }
        Some((covered, bounded))
    }

    /// Add `other`'s rectangles: anywhere where it bounds nothing.
    pub(super) fn join(&mut self, other: &DrawnArea) {
        if other.whole {
            self.everywhere();
            return;
        }
        if self.whole {
            return;
        }
        for &rect in other.rects.iter() {
            self.rects.insert(rect);
        }
        if self.rects.len() > MOST_RECTS {
            self.everywhere();
        }
    }

    /// Bound nothing: the draws may write anywhere in the render area.
    pub(super) fn everywhere(&mut self) {
        self.whole = true;
        self.rects.clear();
    }

    /// The tiles of `layout`'s pass the rectangles reach, at the encoder's tile size (the pass's
    /// `tileWidth` and `tileHeight`, or 32 by 32 pixels, 32 by 16 at 4 samples; the module docs):
    /// `None` where that is every tile of the render area rounded up to whole tiles, the area is
    /// not bounded, or the pass's sample count has no measured tile size.
    pub(super) fn tiles(&self, layout: &PassLayout) -> Option<DrawnTiles> {
        let tile = layout.tile.tile.filter(|_| !self.whole)?;
        self.tiles_of(tile, layout.extent)
    }

    /// The tiles of `tile` pixels a side of a render area of `extent` the rectangles reach:
    /// `None` where that is every one ([`tiles`](Self::tiles)).
    fn tiles_of(&self, tile: [u32; 2], extent: [u32; 2]) -> Option<DrawnTiles> {
        let [per_row, rows] = [0, 1].map(|axis| extent[axis].div_ceil(tile[axis]));
        // A rectangle that reaches every tile on its own (a draw over the whole render area, as
        // most are): every tile, with no grid made.
        let every = |&[x0, y0, x1, y1]: &[u32; 4]| {
            x0 < tile[0]
                && y0 < tile[1]
                && x1.div_ceil(tile[0]) >= per_row
                && y1.div_ceil(tile[1]) >= rows
        };
        if per_row == 0 || rows == 0 || self.rects.iter().any(every) {
            return None;
        }
        let mut tiles = DrawnTiles {
            tile,
            per_row,
            rows,
            words: vec![0; (per_row as usize * rows as usize).div_ceil(32)],
        };
        for &[x0, y0, x1, y1] in self.rects.iter() {
            for ty in y0 / tile[1]..y1.div_ceil(tile[1]).min(rows) {
                for tx in x0 / tile[0]..x1.div_ceil(tile[0]).min(per_row) {
                    let t = (ty * per_row + tx) as usize;
                    tiles.words[t / 32] |= 1 << (t % 32);
                }
            }
        }
        let all = (0..per_row as usize * rows as usize)
            .all(|t| tiles.words[t / 32] & (1 << (t % 32)) != 0);
        (!all).then_some(tiles)
    }
}

/// The tiles a pass's draws may reach, of the render area rounded up to whole tiles
/// ([`DrawnArea::tiles`]): `tile` pixels a side, `per_row` by `rows` of them, one bit each, row
/// after row.
#[derive(Clone, Debug)]
pub(super) struct DrawnTiles {
    tile: [u32; 2],
    per_row: u32,
    rows: u32,
    words: Vec<u32>,
}

impl DrawnTiles {
    /// A tile's width and height in pixels.
    pub(super) fn tile(&self) -> [u32; 2] {
        self.tile
    }

    /// Whether no tile is reached.
    pub(super) fn is_empty(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    /// The reached tiles' count.
    pub(super) fn count(&self) -> u64 {
        self.words.iter().map(|w| u64::from(w.count_ones())).sum()
    }

    /// Whether tile `(tx, ty)` is reached: none past the grid.
    pub(super) fn contains(&self, tx: u32, ty: u32) -> bool {
        let t = (ty as usize) * (self.per_row as usize) + tx as usize;
        tx < self.per_row && ty < self.rows && self.words[t / 32] & (1 << (t % 32)) != 0
    }

    /// The reached tiles as rectangles of pixels, each run of them along a row of tiles, clipped
    /// to `extent` from the origin (an attachment's reach).
    fn rects(&self, extent: [u32; 2]) -> Vec<vk::Rect2D> {
        let [tw, th] = self.tile;
        let mut out = Vec::new();
        for ty in 0..self.rows {
            let mut tx = 0;
            while tx < self.per_row {
                if !self.contains(tx, ty) {
                    tx += 1;
                    continue;
                }
                let first = tx;
                while tx < self.per_row && self.contains(tx, ty) {
                    tx += 1;
                }
                let [x0, y0] = [first * tw, ty * th];
                let [x1, y1] = [(tx * tw).min(extent[0]), ((ty + 1) * th).min(extent[1])];
                if x0 < x1 && y0 < y1 {
                    out.push(vk::Rect2D {
                        offset: vk::Offset2D {
                            x: x0 as i32,
                            y: y0 as i32,
                        },
                        extent: vk::Extent2D {
                            width: x1 - x0,
                            height: y1 - y0,
                        },
                    });
                }
            }
        }
        out
    }
}

/// One attachment's clear before its pass: the attachment, a render pass of it alone, its
/// framebuffer over the clear's extent and the pass's layers, the rectangles cleared (the extent,
/// or the tiles a `DontCare` load's draws may reach), the aspects cleared, and the clear value.
struct ReachClear {
    attachment: Reached,
    pass: RenderPassObject,
    framebuffer: FramebufferObject,
    area: vk::Rect2D,
    rects: Vec<vk::Rect2D>,
    layers: u32,
    aspects: vk::ImageAspectFlags,
    value: vk::ClearValue,
}

/// The attachment a [`ReachClear`] clears: a color attachment's index, or the depth/stencil
/// attachment's depth and stencil aspects, each whether it is cleared.
#[derive(Clone, Copy)]
enum Reached {
    Color(usize),
    DepthStencil { depth: bool, stencil: bool },
}

/// The clears before a pass of its attachments that a `Clear` load clears past the render area
/// (the module docs), and of the cleared attachments its first instance leaves out that no
/// whole-subresource clear may stand for, with the uses they make. The clears are held until the
/// command buffer completes.
pub(super) struct ReachClears {
    clears: Vec<ReachClear>,
    uses: Vec<Use<ResourceKey>>,
}

impl ReachClears {
    /// The clears of `layout`'s attachments, a pass with a draw call where `called` (the module
    /// docs); `left_out` names the color indices whose clear the pass must make before it
    /// whatever their store action (its first instance leaves them out, and no whole-subresource
    /// clear stands for them). Where `bounded` gives the `DontCare` loads that took a `Clear` and
    /// the tiles the pass's draws may reach, each of those clears only those tiles, before the
    /// pass. `None` when there is none.
    pub(super) fn plan(
        device: &Device,
        layout: &PassLayout,
        called: bool,
        left_out: &[u32],
        bounded: Option<(&DontCareLoads, &DrawnTiles)>,
    ) -> Result<Option<ReachClears>, Error> {
        let area = layout.extent;
        let mut clears = Vec::new();
        let mut uses = Vec::new();
        for (index, color) in layout.colors.iter().enumerate() {
            let Some(color) = color else {
                continue;
            };
            if color.load != LoadAction::Clear || color.memoryless {
                continue;
            }
            let extent = match stores(color.store) {
                true => cleared(layout, color.target.extent, called),
                false => area,
            };
            // A color held in sample planes keeps its `Clear` load (`load_in_pass`): its tiles
            // are not bounded.
            let tiles = bounded
                .filter(|(loads, _)| loads.colors.contains(&index) && color.planes.is_none())
                .map(|(_, tiles)| tiles);
            if extent == area && !left_out.contains(&(index as u32)) && tiles.is_none() {
                continue;
            }
            let rects = tiles.map_or_else(|| vec![rect(extent)], |t| t.rects(extent));
            let (pass, framebuffer, made) =
                render_target::reach_pass(device, &color.target, false, extent)?;
            uses.extend(made);
            clears.push(ReachClear {
                attachment: Reached::Color(index),
                pass,
                framebuffer,
                area: rect(extent),
                rects,
                layers: color.target.layers,
                aspects: vk::ImageAspectFlags::COLOR,
                value: vk::ClearValue { color: color.clear },
            });
        }
        if let Some(ds) = layout
            .depth_stencil
            .as_ref()
            .filter(|ds| ds.separate.is_none() && !memoryless(&ds.target))
        {
            let cleared_aspect = |aspect: Option<&super::render_pass::AspectTarget>| {
                aspect.is_some_and(|a| {
                    a.load == LoadAction::Clear && !a.memoryless && stores(a.store)
                })
            };
            let (depth, stencil) = (
                cleared_aspect(ds.depth.as_ref()),
                cleared_aspect(ds.stencil.as_ref()),
            );
            let extent = cleared(layout, ds.target.extent, called);
            let mut value = vk::ClearDepthStencilValue::default();
            for aspect in [ds.depth.as_ref(), ds.stencil.as_ref()]
                .into_iter()
                .flatten()
            {
                match aspect.clear {
                    AspectClear::Depth(depth) => value.depth = depth,
                    AspectClear::Stencil(stencil) => value.stencil = stencil,
                }
            }
            // The aspects a bounded `DontCare` load cleared clear its tiles alone; the others
            // their whole reach, where it passes the render area.
            let (tiles, bound) = match bounded {
                Some((loads, tiles)) => (Some(tiles), [loads.depth, loads.stencil]),
                None => (None, [false, false]),
            };
            let parts = [
                ([depth && !bound[0], stencil && !bound[1]], None),
                ([depth && bound[0], stencil && bound[1]], tiles),
            ];
            for ([depth, stencil], tiles) in parts {
                if !(depth || stencil) || (extent == area && tiles.is_none()) {
                    continue;
                }
                let rects = tiles.map_or_else(|| vec![rect(extent)], |t| t.rects(extent));
                let (pass, framebuffer, made) =
                    render_target::reach_pass(device, &ds.target, true, extent)?;
                uses.extend(made);
                let mut aspects = vk::ImageAspectFlags::empty();
                if depth {
                    aspects |= vk::ImageAspectFlags::DEPTH;
                }
                if stencil {
                    aspects |= vk::ImageAspectFlags::STENCIL;
                }
                clears.push(ReachClear {
                    attachment: Reached::DepthStencil { depth, stencil },
                    pass,
                    framebuffer,
                    area: rect(extent),
                    rects,
                    layers: ds.target.layers,
                    aspects,
                    value: vk::ClearValue {
                        depth_stencil: value,
                    },
                });
            }
        }
        Ok((!clears.is_empty()).then_some(ReachClears { clears, uses }))
    }

    /// Load each attachment these clears clear, a `Clear` load, in `layout`'s pass: the clear
    /// before the pass covers the render area as well, and a pass's own clear on a device that
    /// renders through Metal (MoltenVK) reaches the whole attachment in a pass without a draw,
    /// where Metal's own reaches whole tiles in a pass with a draw call (the module docs). A
    /// color attachment held in sample planes keeps its load (`sample_planes.rs` clears them).
    pub(super) fn load_in_pass(&self, layout: &mut PassLayout) {
        for clear in &self.clears {
            match clear.attachment {
                Reached::Color(index) => {
                    if let Some(color) = layout
                        .colors
                        .get_mut(index)
                        .and_then(Option::as_mut)
                        .filter(|color| color.load == LoadAction::Clear && color.planes.is_none())
                    {
                        color.load = LoadAction::Load;
                    }
                }
                Reached::DepthStencil { depth, stencil } => {
                    let Some(ds) = layout.depth_stencil.as_mut() else {
                        continue;
                    };
                    for (cleared, aspect) in
                        [(depth, ds.depth.as_mut()), (stencil, ds.stencil.as_mut())]
                    {
                        if let Some(aspect) =
                            aspect.filter(|a| cleared && a.load == LoadAction::Clear)
                        {
                            aspect.load = LoadAction::Load;
                        }
                    }
                }
            }
        }
    }

    /// Record the clears in `core`'s command buffer, between barriers from everything before
    /// and to everything after, and hold them until it completes.
    pub(super) fn record(self, core: &mut EncoderCore) {
        let ReachClears { clears, uses } = self;
        core.device().shared().counts.pool_compiled(
            crate::work_counts::PoolCompiled::AttachmentClear,
            clears.len() as u64,
        );
        core.barrier(Barrier::everything());
        core.command(
            render_target::ATTACHMENT_STAGES,
            &uses,
            |raw, command_buffer| {
                // A bounded clear of no tile clears nothing; its attachment still loads.
                for clear in clears.iter().filter(|clear| !clear.rects.is_empty()) {
                    let begin = vk::RenderPassBeginInfo::default()
                        .render_pass(clear.pass.raw)
                        .framebuffer(clear.framebuffer.raw)
                        .render_area(clear.area);
                    let attachments = [vk::ClearAttachment {
                        aspect_mask: clear.aspects,
                        color_attachment: 0,
                        clear_value: clear.value,
                    }];
                    let rects: Vec<vk::ClearRect> = clear
                        .rects
                        .iter()
                        .map(|&rect| vk::ClearRect {
                            rect,
                            base_array_layer: 0,
                            layer_count: clear.layers,
                        })
                        .collect();
                    // SAFETY: the encoder's command buffer is recording outside a render pass; the
                    // pass, its framebuffer and the attachment are live until it completes (held
                    // below, and the texture by the encoder); its image rests in `GENERAL`, the
                    // pass's layout; the area is the framebuffer's, and the layers its own; the
                    // aspects are the attachment format's, and the subpass's one attachment, color
                    // 0 or its depth/stencil attachment, is the one cleared; every rectangle lies
                    // in the area.
                    unsafe {
                        raw.cmd_begin_render_pass(
                            command_buffer,
                            &begin,
                            vk::SubpassContents::INLINE,
                        );
                        raw.cmd_clear_attachments(command_buffer, &attachments, &rects);
                        raw.cmd_end_render_pass(command_buffer);
                    }
                }
            },
        );
        core.barrier(Barrier::everything());
        core.retain(clears);
    }
}

/// The rectangle of `extent` from the origin.
fn rect([width, height]: [u32; 2]) -> vk::Rect2D {
    vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: vk::Extent2D { width, height },
    }
}

#[cfg(test)]
mod tests {
    use super::{dont_care_clear, DrawnArea};
    use ash::vk;
    use metal2vulkan::metal::format::{ClearRoute, ExtendedRange};
    use metal2vulkan::metal::PixelFormat;

    /// A `DontCare` load that clears writes each format's zero as Metal's tile holds it: an
    /// extended-range attachment's raw texel of `(0, 0, 0, 0)`, its codes 384, not zero bits; a
    /// format held directly, zero bits.
    #[test]
    fn dont_care_clears_to_the_formats_zero() {
        for pf in [
            PixelFormat::RGB10A82PXR10,
            PixelFormat::RGB10A82PXR10SRGB,
            PixelFormat::BGRA10XR,
            PixelFormat::BGRA10XRSRGB,
            PixelFormat::BGR10XR,
            PixelFormat::BGR10XRSRGB,
        ] {
            let form = ExtendedRange::of(pf).expect("an extended-range format");
            // SAFETY: every member of the union is four 32-bit values; any bits are a `u32`.
            let got = unsafe { dont_care_clear(pf).uint32 };
            assert_eq!(
                got,
                form.clear_texel([0.0; 4], ClearRoute::ThroughHalf),
                "{pf:?}"
            );
            assert_ne!(got, [0; 4], "{pf:?}");
        }
        for pf in [
            PixelFormat::RGBA16Float,
            PixelFormat::BGRA8Unorm,
            PixelFormat::R32Uint,
            PixelFormat::RGBA16Bfloat,
        ] {
            // SAFETY: as above.
            let got = unsafe { dont_care_clear(pf).uint32 };
            assert_eq!(got, [0; 4], "{pf:?}");
        }
        // A bfloat zero is the code 0 by either release's rule.
        let form = ExtendedRange::Rgba16Bfloat;
        for rule in [ClearRoute::ThroughHalf, ClearRoute::Straight] {
            assert_eq!(form.clear_texel([0.0; 4], rule), [0; 4], "{rule:?}");
        }
    }

    /// The tiles the rectangles cover, against those their bounding rectangle reaches, are
    /// counted in 32-pixel tiles from the origin: two rectangles far apart cover their own
    /// tiles alone, where their bounds reach every tile between them.
    #[test]
    fn covered_tiles_count_the_rectangles_tiles_against_their_bounds() {
        let rect = |x: i32, y: i32, w: u32, h: u32| vk::Rect2D {
            offset: vk::Offset2D { x, y },
            extent: vk::Extent2D {
                width: w,
                height: h,
            },
        };
        let cases: [(Vec<vk::Rect2D>, Option<(u64, u64)>); 6] = [
            (vec![], None),
            // A 1920x1080 render area's every tile: 60 by 34.
            (vec![rect(0, 0, 1920, 1080)], Some((2040, 2040))),
            // Off the grid: tiles 0 to 2 by 0 to 1.
            (vec![rect(31, 1, 34, 32)], Some((6, 6))),
            // Two corners at either end of a 1920x1080 area, four tiles each.
            (
                vec![rect(0, 0, 64, 64), rect(1856, 1024, 64, 56)],
                Some((8, 2040)),
            ),
            // Overlapping rectangles count each tile once.
            (vec![rect(0, 0, 96, 32), rect(32, 0, 96, 32)], Some((4, 4))),
            // An L: the bounds reach the corner the rectangles leave out.
            (vec![rect(0, 0, 64, 32), rect(0, 32, 32, 32)], Some((3, 4))),
        ];
        for (case, (rects, expected)) in cases.iter().enumerate() {
            let mut area = DrawnArea::default();
            area.add(rects);
            assert_eq!(area.covered_tiles(), *expected, "case {case}");
        }
        assert_eq!(DrawnArea::unbounded().covered_tiles(), None);
    }

    /// The tiles a pass's rectangles reach are the tiles each overlaps, whether one of them
    /// reaches every tile (no grid made) or none does: checked tile by tile.
    #[test]
    fn drawn_tiles_are_the_tiles_the_rectangles_overlap() {
        let rect = |x: i32, y: i32, w: u32, h: u32| vk::Rect2D {
            offset: vk::Offset2D { x, y },
            extent: vk::Extent2D {
                width: w,
                height: h,
            },
        };
        let tile = [32u32, 16];
        let extent = [100u32, 70];
        let cases = [
            vec![rect(0, 0, 100, 70)],
            vec![rect(31, 15, 69, 55)],
            vec![rect(32, 0, 68, 70)],
            vec![rect(0, 16, 100, 54)],
            vec![rect(0, 0, 96, 64)],
            vec![rect(0, 0, 97, 65)],
            vec![rect(5, 5, 10, 10), rect(0, 0, 100, 70)],
            vec![rect(0, 0, 50, 70), rect(50, 0, 50, 70)],
            vec![rect(40, 20, 10, 10)],
            vec![],
        ];
        for (case, rects) in cases.iter().enumerate() {
            let mut area = DrawnArea::default();
            area.add(rects);
            let [per_row, rows] = [0, 1].map(|axis| extent[axis].div_ceil(tile[axis]));
            let reached = |tx: u32, ty: u32| {
                area.rects.iter().any(|&[x0, y0, x1, y1]| {
                    x0 < (tx + 1) * tile[0]
                        && tx * tile[0] < x1
                        && y0 < (ty + 1) * tile[1]
                        && ty * tile[1] < y1
                })
            };
            let every = (0..rows).all(|ty| (0..per_row).all(|tx| reached(tx, ty)));
            match area.tiles_of(tile, extent) {
                None => assert!(every, "case {case}"),
                Some(tiles) => {
                    assert!(!every, "case {case}");
                    for ty in 0..rows {
                        for tx in 0..per_row {
                            let reach = reached(tx, ty);
                            assert_eq!(tiles.contains(tx, ty), reach, "case {case}: {tx}, {ty}");
                        }
                    }
                }
            }
        }
    }
}
