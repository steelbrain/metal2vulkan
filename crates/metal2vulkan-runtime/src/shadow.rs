//! The copy form of attachment reads: what a draw that reads, through a texture, an attachment its
//! own pass renders copies first into the image's shadow, on a device that reads attachments
//! without the attachment feedback loop (`floor::Capabilities::feedback_loop`).
//!
//! An image a draw may read so has one shadow, an image of its format, extent, levels and layers
//! whose home is shader-read-only, made at the first draw that reads it and freed with it. A
//! texture that reaches the rendered level and layer reads the shadow through its second element
//! (`crate::arrays`); one that reaches neither reads the image itself, at home. A draw copies into
//! the shadow what it reads that the shadow does not hold, with no look-ahead: everything it reads
//! at the pass's first draw that reads the image, and later only what is new, as one area of each
//! read where what is new is one. What the shadow holds counts from the pass's first copy into it:
//! a memory barrier after draws, which orders what they rendered before later reads, leaves it
//! holding nothing, so the next draw that reads the image copies what it reads again.
//!
//! Where the pass cleared the attachment and the draw is the pass's first, the rendered level and
//! layer holds only the clear colour inside Metal's render area: where every read of that level
//! and layer lies in it alone and inside that area, the shadow's is cleared to the same colour
//! (the pass's clear value, the shadow of the attachment's format) in place of the copy, and the
//! rest of what the draw reads is copied. A draw of the pass recorded before, of any pipeline,
//! leaves the copy.
//!
//! A slot the fragment function reads only at its own pixel (`StageModule::own_pixel_textures`)
//! reads the rectangle of the draw's scissor, a pixel wider each way for its helper invocations,
//! clamped into its level; any other slot reads every level and layer its texture reaches.

/// The most areas a shadow's [`Held`] keeps: past them it forgets the oldest, which a later read
/// copies again.
const HELD: usize = 8;

/// A box of an image's texels: one level, a range of its layers, and a rectangle of the level,
/// each `[first, end)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Area {
    pub(crate) level: u32,
    pub(crate) layers: [u32; 2],
    pub(crate) x: [u32; 2],
    pub(crate) y: [u32; 2],
}

impl Area {
    /// Whether it holds no texel.
    pub(crate) fn is_empty(&self) -> bool {
        [self.layers, self.x, self.y]
            .iter()
            .any(|&[first, end]| first >= end)
    }

    /// Its texels.
    pub(crate) fn texels(&self) -> u64 {
        [self.layers, self.x, self.y]
            .iter()
            .map(|&[first, end]| u64::from(end.saturating_sub(first)))
            .product()
    }

    /// Whether it holds a texel of level `level`'s layer `layer`.
    pub(crate) fn reaches(&self, (level, layer): (u32, u32)) -> bool {
        !self.is_empty() && self.level == level && self.layers[0] <= layer && layer < self.layers[1]
    }

    /// Whether it holds every texel of `other`.
    fn contains(&self, other: &Area) -> bool {
        self.level == other.level
            && covers(self.layers, other.layers)
            && covers(self.x, other.x)
            && covers(self.y, other.y)
    }

    /// What of it to copy where `held` is copied already: what is left of it past `held` where
    /// that is one area (empty where `held` holds all of it), else all of it.
    fn past(&self, held: &Area) -> Area {
        if held.level != self.level || !covers(held.layers, self.layers) {
            return *self;
        }
        match (covers(held.x, self.x), covers(held.y, self.y)) {
            (true, _) => less(self.y, held.y).map_or(*self, |y| Area { y, ..*self }),
            (false, true) => less(self.x, held.x).map_or(*self, |x| Area { x, ..*self }),
            (false, false) => *self,
        }
    }
}

/// Whether `outer` holds every value of `inner`.
fn covers(outer: [u32; 2], inner: [u32; 2]) -> bool {
    outer[0] <= inner[0] && inner[1] <= outer[1]
}

/// `a` but `b`, where what is left is one range, empty where `b` covers `a`; `None` where `b`
/// lies inside `a` and leaves a range on each side.
fn less(a: [u32; 2], b: [u32; 2]) -> Option<[u32; 2]> {
    let [first, end] = a;
    if b[1] <= first || end <= b[0] {
        return Some(a);
    }
    match (b[0] <= first, end <= b[1]) {
        (true, true) => Some([first, first]),
        (true, false) => Some([b[1], end]),
        (false, true) => Some([first, b[0]]),
        (false, false) => None,
    }
}

/// What a draw reads through a slot its fragment function reads only at its own pixel, of a
/// texture whose first level and layer are `level` and `layer` of its image, that level `size`
/// texels: the rectangle bounding the draw's `scissor` (per axis `[first, end)`), a pixel wider
/// each way for its helper invocations, clamped into the level as the read's coordinate is.
/// Empty for an empty scissor, which draws no fragment.
pub(crate) fn own_pixel(level: u32, layer: u32, size: [u32; 2], scissor: [[u32; 2]; 2]) -> Area {
    let span = |[first, end]: [u32; 2], size: u32| match first < end && size > 0 {
        true => {
            let last = size - 1;
            [first.saturating_sub(1).min(last), end.min(last) + 1]
        }
        false => [0, 0],
    };
    let (x, y) = (span(scissor[0], size[0]), span(scissor[1], size[1]));
    let layers = match x[0] < x[1] && y[0] < y[1] {
        true => [layer, layer + 1],
        false => [layer, layer],
    };
    Area {
        level,
        layers,
        x,
        y,
    }
}

/// What a draw reads through any other slot: every level and layer of a texture whose first
/// level and layer are `level` and `layer` of its image, of `levels` levels and `layers` layers,
/// the image's first level `size` texels. One area a level.
pub(crate) fn whole(level: u32, levels: u32, layer: u32, layers: u32, size: [u32; 2]) -> Vec<Area> {
    (level..level + levels)
        .map(|level| Area {
            level,
            layers: [layer, layer + layers],
            x: [0, (size[0] >> level).max(1)],
            y: [0, (size[1] >> level).max(1)],
        })
        .collect()
}

/// Whether a clear of `rendered`, the level and layer a pass cleared that no draw of the pass has
/// rendered since, gives the shadow what the image holds of each of `areas` that reaches it: one
/// does, and each such lies in that level and layer alone and inside Metal's render area `area`
/// (`[width, height]` from the level's origin), which holds only the clear colour.
pub(crate) fn clears(areas: &[Area], rendered: (u32, u32), area: [u32; 2]) -> bool {
    let layer = rendered.1;
    let mut reaching = areas.iter().filter(|a| a.reaches(rendered)).peekable();
    reaching.peek().is_some()
        && reaching
            .all(|a| a.layers == [layer, layer + 1] && a.x[1] <= area[0] && a.y[1] <= area[1])
}

/// The subresources `areas` reach, each a level and a range of its layers, the ranges of one level
/// joined where they meet or overlap: what a copy's barriers move, each subresource once.
pub(crate) fn ranges(areas: &[Area]) -> Vec<(u32, [u32; 2])> {
    let mut ranges: Vec<(u32, [u32; 2])> = (areas.iter())
        .filter(|area| !area.is_empty())
        .map(|area| (area.level, area.layers))
        .collect();
    ranges.sort_unstable();
    let mut joined: Vec<(u32, [u32; 2])> = Vec::new();
    for (level, layers) in ranges {
        match joined.last_mut() {
            Some((last, span)) if *last == level && layers[0] <= span[1] => {
                span[1] = span[1].max(layers[1]);
            }
            _ => joined.push((level, layers)),
        }
    }
    joined
}

/// `ranges` parted at the rendered `level` and `layer`: whether one of them holds it, and the
/// rest of them without it, which lie at home.
pub(crate) fn parted(
    ranges: &[(u32, [u32; 2])],
    (level, layer): (u32, u32),
) -> (bool, Vec<(u32, [u32; 2])>) {
    let mut rendered = false;
    let mut rest = Vec::new();
    for &(at, [first, end]) in ranges {
        if at == level && (first..end).contains(&layer) {
            rendered = true;
            rest.extend(
                [[first, layer], [layer + 1, end]]
                    .into_iter()
                    .filter(|[first, end]| first < end)
                    .map(|span| (at, span)),
            );
        } else {
            rest.push((at, [first, end]));
        }
    }
    (rendered, rest)
}

/// What a shadow holds of its image since the pass's first copy into it or its last memory
/// barrier after draws: areas, each copied whole, at most [`HELD`] of them.
#[derive(Debug, Default)]
pub(crate) struct Held {
    areas: Vec<Area>,
}

impl Held {
    /// The copies a draw that reads `reads` of the image makes first: for each read, what of it
    /// the shadow does not hold, the least area one held area leaves of it, or all of it; the
    /// shadow holds every read from then.
    pub(crate) fn plan(&mut self, reads: &[Area]) -> Vec<Area> {
        let mut copies = Vec::new();
        for read in reads.iter().filter(|read| !read.is_empty()) {
            let copy = (self.areas.iter())
                .map(|held| read.past(held))
                .min_by_key(Area::texels)
                .unwrap_or(*read);
            if !copy.is_empty() {
                copies.push(copy);
                self.hold(*read);
            }
        }
        copies
    }

    /// Whether the shadow holds every read of `reads`: whether [`plan`](Self::plan) copies none.
    pub(crate) fn holds(&self, reads: &[Area]) -> bool {
        (reads.iter().filter(|read| !read.is_empty()))
            .all(|read| (self.areas.iter()).any(|held| read.past(held).is_empty()))
    }

    /// The shadow holds `area`: a held area inside it is dropped, and past [`HELD`] the oldest.
    fn hold(&mut self, area: Area) {
        self.areas.retain(|held| !area.contains(held));
        if self.areas.len() == HELD {
            self.areas.remove(0);
        }
        self.areas.push(area);
    }

    /// A memory barrier after draws: the shadow holds nothing of what they rendered.
    pub(crate) fn clear(&mut self) {
        self.areas.clear();
    }
}
