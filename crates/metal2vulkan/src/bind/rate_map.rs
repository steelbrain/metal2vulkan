//! Rasterization-rate maps: the consumer's `MTLRasterizationRateMap`.
//!
//! A function that maps coordinates through a `rasterization_rate_map_decoder` reads its
//! `constant rasterization_rate_map_data &` argument as a buffer of Metal's parameter data, in
//! Apple's layout (`air::canon::rate_map_data` has it). The consumer implements the Metal object
//! with a [`RasterizationRateMap`]: `newRasterizationRateMapWithDescriptor:` is
//! [`RasterizationRateMap::new`] of a [`RateMapDescriptor`]; `physicalSizeForLayer:`,
//! `physicalGranularity`, `screenSize`, and `layerCount` are its accessors;
//! `parameterBufferSizeAndAlign` is [`RasterizationRateMap::parameter_data_size`] (24,624 bytes,
//! Metal's for every map) and [`RasterizationRateMap::PARAMETER_DATA_ALIGNMENT`];
//! `copyParameterDataToBuffer:offset:` writes [`RasterizationRateMap::to_bytes`] at the offset,
//! Metal's bytes; and `mapScreenToPhysicalCoordinates:forLayer:` and
//! `mapPhysicalToScreenCoordinates:forLayer:` are [`RasterizationRateMap::screen_to_physical`] and
//! [`RasterizationRateMap::physical_to_screen`], Apple's CPU arithmetic over those bytes. A
//! shader's answers are [`RasterizationRateMap::shader_screen_to_physical`] and
//! [`RasterizationRateMap::shader_physical_to_screen`], Apple's GPU arithmetic over the same bytes,
//! which differs from the CPU's (the granule of the coordinate plus half a pixel, two roundings
//! where the CPU fuses, denormals flushed, the layer unclamped). Measured on an Apple M4: the bytes
//! of 45 maps, and every CPU and GPU answer probed over them, bit for bit, the sign of a zero
//! included. The translated decoder's `float` arithmetic is the device's: it computes the shader's
//! answers bit for bit on a device that reads `float` denormals as Metal does, and with the
//! device's own denormals elsewhere (`PrecisionNote::DeviceDenormals`), as the SPIR-V interpreter
//! keeps them. The buffer is then bound like any other ([`BoundBuffer::Buffer`]).
//!
//! **Compiling a descriptor.** Metal leaves the compiled map to the device: a layer's quality
//! samples "may be rounded up ... depending on hardware support". [`RasterizationRateMap::new`]
//! compiles by Apple M3's rule, measured (24,854 descriptors on macOS 26.7, every tile of the
//! 24,615 it compiles reproduced, the other 239 refused as [`UnpinnedCompile::DenseSamples`]), per
//! axis of screen extent `W` and quality samples `q_0..q_{n-1}`, in `f32` arithmetic:
//!
//! - each sample becomes `Q_i = max(min(q_i + 1/256, 1), 0.01)` (`NaN` reads 1); one sample is
//!   the quality everywhere, and `n ≥ 2` samples are a piecewise linear function with sample `i`
//!   at `i·W/(n-1)`, evaluated at an integer screen coordinate `x < W` as
//!   `Q_k + (Q_{k+1} - Q_k)·(u - k)` for `u = x·(n-1)/W` and `k = min(⌊u⌋, n - 2)`, and as
//!   `Q_{n-1}` from `W` on;
//! - tiles of [`PHYSICAL_GRANULARITY`] (`G` = 32) physical pixels are laid from screen 0: tile `k`
//!   starts at `S_k` and spans the largest number of screen pixels `m` with `m ≤ ⌊G / Q⌋`, for
//!   `Q` the function's maximum at the integer coordinates `S_k..=S_k + m` (a peak between two
//!   integers is not seen);
//! - the tile that reaches `W` is the last: its `e = W - S_k` screen pixels (`e < m`) map to the
//!   largest `p < min(e, G)` with `e/p` a multiple of 1/16 and `e·G ≤ m·p`, or to `min(e, G)`
//!   physical pixels where no `p` is such (`G` where `e = m`);
//! - a map of two layers has one vertical axis, compiled from the larger of the layers' vertical
//!   functions at each coordinate; each layer's horizontal axis is its own;
//! - a map of no layers has no horizontal axis (its horizontal tables are zero words) and one
//!   vertical axis, compiled from no quality function: every coordinate at the least quality;
//! - an axis of no screen pixels has one tile of no screen and no physical pixels, whatever its
//!   quality, so the layer's physical extent on it is 0 and its tables hold Apple's least line
//!   (`air::canon::rate_map_data`). Metal's API validation stops such a descriptor; plain Metal
//!   compiles it, measured on an Apple M3 Max under macOS 27.0.1.
//!
//! Outside the bounds it is measured in, [`RasterizationRateMap::new`] refuses
//! ([`RateMapError::Unpinned`], [`UnpinnedCompile`]) rather than guess: a screen extent above
//! 16384 (Apple's compile crashes or returns a corrupt map above it), more than two layers (the
//! most Apple M3 supports), more quality samples on an axis than its extent plus one, a sample
//! outside `[0, 1]` on an axis of two or more samples, and an axis of samples that are not all
//! equal past the most measured: 9 on a map of one layer, 4 on a map of two.
//!
//! [`RasterizationRateMap::from_layers`] takes explicit tiles ([`RateMapAxis`]) instead, for a
//! consumer that compiles by another rule, within what Metal's parameter data holds: tiles of 32
//! physical pixels, at most 16384 screen pixels a side, at most two layers, one vertical axis.
//!
//! **Rendering a rate-mapped pass** (`MTLRenderPassDescriptor.rasterizationRateMap`) is the
//! consumer's, on every device region by region ([`RasterizationRateMap::regions`]): into targets
//! of the map's physical size, each region drawn with the guest's viewport composed with its tiles'
//! line. The image is Metal's in structure; over a map that is not the identity its lines are this
//! crate's tiles', a near-equivalent the plan records ([`RateMapRegions::precision`]).
//!
//! [`BoundBuffer::Buffer`]: crate::bind::BoundBuffer::Buffer

use std::fmt;

use crate::air::canon::rate_map_data as format;
use crate::metal::Mutability;

/// The physical granularity of a map: the width and height in physical pixels of every tile but
/// each axis's last (Apple M3 and M4 report the same, and Metal's parameter data is per 32
/// pixels).
pub const PHYSICAL_GRANULARITY: u32 = format::GRANULARITY;

/// The largest screen width or height a compiled rasterization rate map takes
/// ([`RasterizationRateMap::new`] refuses a larger one, `RateMapError::ScreenExtent`; an extent of
/// 0 it compiles, as plain Metal does). Metal's own API validation stops a descriptor past 32,768
/// and one of 0, of which the runtime holds to the upper bound alone.
pub const MAX_SCREEN_EXTENT: u32 = 1 << 16;

/// Apple's compile adds 1/256 to every quality sample before it clamps it.
const QUALITY_BIAS: f32 = 1.0 / 256.0;

/// The least quality Apple's compile clamps a sample to: 100 screen pixels per physical one.
const MIN_QUALITY: f32 = 0.01;

/// Apple's last tile spans a multiple of 1/16 screen pixels per physical pixel.
const LAST_TILE_STEPS: u32 = 16;

/// The largest screen extent whose compile is measured, and the most Metal's parameter data holds.
const MAX_COMPILED_EXTENT: u32 = format::MAX_EXTENT;

/// The most layers whose compile is measured: Apple M3's
/// `supportsRasterizationRateMapWithLayerCount:`, and the most Metal's parameter data holds.
const MAX_COMPILED_LAYERS: usize = format::LAYERS as usize;

/// The most quality samples, not all equal, on one axis whose compile is measured: on a map of
/// one layer (index 0) and of two (index 1).
const MAX_MEASURED_SAMPLES: [usize; MAX_COMPILED_LAYERS] = [9, 4];

/// A rasterization-rate map's description (`MTLRasterizationRateMapDescriptor`).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct RateMapDescriptor {
    /// `screenSize`: the width and height in screen pixels of the region from the origin.
    pub screen_size: [u32; 2],
    /// `layers`, from layer 0.
    pub layers: Vec<RateMapLayerDescriptor>,
    /// `label`: the name the map keeps. The compile ignores it.
    pub label: Option<String>,
    /// `mutability` (Apple's private `setMutability:`): `Immutable` until set, as Metal's
    /// descriptor answers. A map made of a `Mutable` descriptor is compiled only at its reset
    /// (the runtime's `RasterizationRateMap::reset_using_descriptor`). The compile ignores it.
    pub mutability: Mutability,
}

impl Default for RateMapDescriptor {
    /// [`RateMapDescriptor::new`] of an empty screen.
    fn default() -> RateMapDescriptor {
        RateMapDescriptor::new([0, 0])
    }
}

impl RateMapDescriptor {
    /// A descriptor of `screen_size` with no layers and no label, `Immutable`.
    pub fn new(screen_size: [u32; 2]) -> RateMapDescriptor {
        RateMapDescriptor {
            screen_size,
            layers: Vec::new(),
            label: None,
            mutability: Mutability::Immutable,
        }
    }
}

/// One layer's quality samples (`MTLRasterizationRateLayerDescriptor`): on each axis the first
/// sample is the quality at the left (top) edge, the last at the right (bottom) edge, and the
/// others are spaced evenly between, each from 0 (lowest) to 1 (full rate).
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RateMapLayerDescriptor {
    /// `horizontal` (`horizontalSampleStorage`).
    pub horizontal: Vec<f32>,
    /// `vertical` (`verticalSampleStorage`).
    pub vertical: Vec<f32>,
}

impl RateMapLayerDescriptor {
    /// A layer of these samples.
    pub fn new(horizontal: Vec<f32>, vertical: Vec<f32>) -> RateMapLayerDescriptor {
        RateMapLayerDescriptor {
            horizontal,
            vertical,
        }
    }
}

/// Why a rate map cannot be built.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RateMapError {
    /// [`RasterizationRateMap::from_layers`] has no layer, so no vertical axis
    /// ([`RasterizationRateMap::new`] compiles a map of none).
    NoLayers,
    /// A screen extent is 0 or above [`MAX_SCREEN_EXTENT`], or explicit tiles cover more than
    /// 16384 screen pixels, the most Metal's parameter data holds.
    ScreenExtent { extent: u32 },
    /// A layer's axis has no quality sample.
    NoSamples { layer: usize, axis: usize },
    /// A granularity other than 32: Metal's parameter data holds a line per 32 pixels.
    Granularity { granularity: u32 },
    /// An axis has no tile, or tile `tile` is malformed: an empty extent, a physical extent above
    /// its screen extent or the granularity, or one below the granularity before the last tile.
    Tile { tile: usize },
    /// A layer's axis covers another screen extent than the map's.
    ExtentMismatch { layer: usize, axis: usize },
    /// [`RasterizationRateMap::new`] does not compile the descriptor: it is outside the bounds
    /// where the compile is measured to be Apple's.
    Unpinned { reason: UnpinnedCompile },
    /// [`RasterizationRateMap::from_layers`] has more than two layers: Metal's parameter data
    /// holds two horizontal axes.
    LayerCount { count: usize },
    /// Layer `layer`'s vertical axis is not layer 0's: Metal's parameter data holds one vertical
    /// axis, every layer's.
    VerticalAxis { layer: usize },
    /// Tile `tile`'s line has an offset exactly halfway between two of Metal's 1/256-pixel
    /// steps, where how Apple rounds it is not measured (Apple's compile makes no such tile).
    OffsetTie { tile: usize },
}

/// Why [`RasterizationRateMap::new`] does not compile a descriptor: Apple M3's compile was not
/// measured there, or cannot be. The bounds are this crate's, not Metal's.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnpinnedCompile {
    /// A screen extent above 16384: Apple's compile crashed on every width from 24576 and a
    /// height of 32768, and returned a map of a corrupt physical size at a height of 16385.
    ScreenExtent { extent: u32 },
    /// More than two layers: Apple M3 supports two, and its compile crashes on more.
    LayerCount { count: usize },
    /// An axis with more quality samples than its screen extent plus one: Apple's compile then
    /// sees peaks between screen pixels by a rule not measured.
    DenseSamples {
        layer: usize,
        axis: usize,
        samples: usize,
        extent: u32,
    },
    /// A sample outside `[0, 1]` (or `NaN`) on an axis of two or more samples: such samples
    /// were measured alone only.
    SampleOutOfRange {
        layer: usize,
        axis: usize,
        index: usize,
    },
    /// An axis of more samples, not all equal, than the most measured on a map of its layer
    /// count, `max`: 9 on a map of one layer, 4 on a map of two. Apple's compile sees peaks
    /// between pixels on denser axes ([`UnpinnedCompile::DenseSamples`]), so a count past the
    /// measured one is not taken to follow the rule. Equal samples are the one quality they
    /// share, at any count.
    SampleCount {
        layer: usize,
        axis: usize,
        samples: usize,
        max: usize,
    },
}

impl fmt::Display for UnpinnedCompile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnpinnedCompile::ScreenExtent { extent } => write!(
                f,
                "screen extent {extent} is above {MAX_COMPILED_EXTENT}, where Apple's compile \
                 crashes or corrupts the map"
            ),
            UnpinnedCompile::LayerCount { count } => write!(
                f,
                "{count} layers, above the {MAX_COMPILED_LAYERS} Apple's compile was measured with"
            ),
            UnpinnedCompile::DenseSamples {
                layer,
                axis,
                samples,
                extent,
            } => write!(
                f,
                "layer {layer} axis {axis} has {samples} samples over {extent} pixels, a density \
                 whose compile is not measured"
            ),
            UnpinnedCompile::SampleOutOfRange { layer, axis, index } => write!(
                f,
                "layer {layer} axis {axis} sample {index} is outside [0, 1] beside other samples, \
                 whose compile is not measured"
            ),
            UnpinnedCompile::SampleCount {
                layer,
                axis,
                samples,
                max,
            } => write!(
                f,
                "layer {layer} axis {axis} has {samples} samples that are not all equal, above the \
                 {max} whose compile is measured"
            ),
        }
    }
}

impl fmt::Display for RateMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RateMapError::NoLayers => f.write_str("explicit tiles need a layer"),
            RateMapError::ScreenExtent { extent } => write!(
                f,
                "screen extent {extent} is past {MAX_SCREEN_EXTENT}, or explicit tiles' is \
                 past {MAX_COMPILED_EXTENT}"
            ),
            RateMapError::NoSamples { layer, axis } => {
                write!(f, "layer {layer} axis {axis} has no quality sample")
            }
            RateMapError::Granularity { granularity } => write!(
                f,
                "granularity {granularity} is not {PHYSICAL_GRANULARITY}, Metal's parameter \
                 data's"
            ),
            RateMapError::Tile { tile } => write!(f, "tile {tile} is malformed"),
            RateMapError::ExtentMismatch { layer, axis } => write!(
                f,
                "layer {layer} axis {axis} does not cover the map's screen extent"
            ),
            RateMapError::Unpinned { reason } => {
                write!(f, "not compiled as Apple compiles it: {reason}")
            }
            RateMapError::LayerCount { count } => write!(
                f,
                "{count} layers, more than the {MAX_COMPILED_LAYERS} Metal's parameter data holds"
            ),
            RateMapError::OffsetTie { tile } => write!(
                f,
                "tile {tile}'s offset lies halfway between two 1/256-pixel steps, whose rounding \
                 is not measured"
            ),
            RateMapError::VerticalAxis { layer } => write!(
                f,
                "layer {layer}'s vertical axis is not layer 0's, where Metal's parameter data \
                 holds one"
            ),
        }
    }
}

impl std::error::Error for RateMapError {}

/// One axis of a compiled layer: its tiles in screen order.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RateMapAxis {
    granularity: u32,
    /// `(screen extent, physical extent)` per tile.
    tiles: Vec<(u32, u32)>,
}

impl RateMapAxis {
    /// The axis of these tiles, each `(screen extent, physical extent)` in pixels from screen 0.
    /// The granularity is 32, Metal's; every tile spans at least as many
    /// screen pixels as physical ones (quality at most 1:1); every tile but the last is
    /// `granularity` physical pixels wide, and the last at most that; the tiles cover at most
    /// 16384 screen pixels; and no tile's offset ties ([`RateMapError::OffsetTie`]). Those are
    /// the axes Metal's parameter data holds.
    pub fn from_tiles(granularity: u32, tiles: &[(u32, u32)]) -> Result<RateMapAxis, RateMapError> {
        if granularity != PHYSICAL_GRANULARITY {
            return Err(RateMapError::Granularity { granularity });
        }
        if tiles.is_empty() {
            return Err(RateMapError::Tile { tile: 0 });
        }
        let last = tiles.len() - 1;
        let (mut screen, mut physical) = (0u32, 0u32);
        for (i, &(s, p)) in tiles.iter().enumerate() {
            let wide = if i == last {
                p <= granularity
            } else {
                p == granularity
            };
            if s == 0 || p == 0 || p > s || !wide {
                return Err(RateMapError::Tile { tile: i });
            }
            if format::offset_tie(screen, physical, s, p) {
                return Err(RateMapError::OffsetTie { tile: i });
            }
            physical += p;
            screen = screen
                .checked_add(s)
                .filter(|&w| w <= MAX_COMPILED_EXTENT)
                .ok_or(RateMapError::ScreenExtent {
                    extent: screen.saturating_add(s),
                })?;
        }
        Ok(RateMapAxis {
            granularity,
            tiles: tiles.to_vec(),
        })
    }

    /// The physical width of every tile but the last.
    pub fn granularity(&self) -> u32 {
        self.granularity
    }

    /// The tiles, each `(screen extent, physical extent)`.
    pub fn tiles(&self) -> &[(u32, u32)] {
        &self.tiles
    }

    /// The screen extent the tiles cover.
    pub fn screen_extent(&self) -> u32 {
        self.tiles.iter().map(|&(s, _)| s).sum()
    }

    /// The physical extent the tiles cover.
    pub fn physical_extent(&self) -> u32 {
        self.tiles.iter().map(|&(_, p)| p).sum()
    }

    /// Compile one axis of `extent` screen pixels by the module's rule (Apple's) from the quality
    /// samples of `layers`: one layer's, or for a vertical axis every layer's, whose functions the
    /// axis follows the larger of.
    fn compile(extent: u32, layers: &[&[f32]]) -> RateMapAxis {
        let g = PHYSICAL_GRANULARITY;
        let functions: Vec<Quality> = layers
            .iter()
            .map(|samples| Quality::new(extent, samples))
            .collect();
        let at = |x: u32| {
            functions
                .iter()
                .map(|f| f.at(x))
                .fold(MIN_QUALITY, f32::max)
        };
        let mut tiles = Vec::new();
        let mut start = 0u32;
        loop {
            // The longest tile whose quality covers the function at every integer coordinate it
            // spans, its end included: the maximum only grows with `m`, so the first `m` that
            // fails ends the search, at most `⌊G / 0.01⌋` pixels on.
            let mut max = at(start);
            let mut m = 0u32;
            loop {
                let next = m + 1;
                max = max.max(at(start + next));
                if next as f32 > (g as f32 / max).floor() {
                    break;
                }
                m = next;
            }
            if start + m >= extent {
                let rest = extent - start;
                let p = if rest < m { last_tile(m, rest) } else { g };
                tiles.push((rest, p));
                break;
            }
            tiles.push((m, g));
            start += m;
        }
        RateMapAxis {
            granularity: g,
            tiles,
        }
    }
}

/// Apple's last tile: `rest` screen pixels short of a tile of `m` map to the most physical pixels
/// below `rest` and the granularity whose ratio to `rest` is a multiple of 1/16 and no coarser than
/// `m`'s, or to `rest` (at most the granularity) where none is.
fn last_tile(m: u32, rest: u32) -> u32 {
    let g = PHYSICAL_GRANULARITY;
    (1..rest.min(g))
        .rev()
        .find(|&p| (LAST_TILE_STEPS * rest).is_multiple_of(p) && g * rest <= m * p)
        .unwrap_or(rest.min(g))
}

/// An axis's quality function as Apple's compile evaluates it, in `f32`.
struct Quality {
    extent: u32,
    /// `max(min(q_i + 1/256, 1), 0.01)`, `NaN` reading 1.
    samples: Vec<f32>,
}

impl Quality {
    fn new(extent: u32, samples: &[f32]) -> Quality {
        let samples = samples
            .iter()
            .map(|&q| {
                if q.is_nan() {
                    1.0
                } else {
                    (q + QUALITY_BIAS).clamp(MIN_QUALITY, 1.0)
                }
            })
            .collect();
        Quality { extent, samples }
    }

    /// The function at integer screen coordinate `x`.
    fn at(&self, x: u32) -> f32 {
        let n = self.samples.len();
        let last = self.samples.last().copied().unwrap_or(1.0);
        if n < 2 || x >= self.extent {
            return last;
        }
        let u = x as f32 * (n - 1) as f32 / self.extent as f32;
        let k = (u as usize).min(n - 2);
        let t = u - k as f32;
        self.samples[k] + (self.samples[k + 1] - self.samples[k]) * t
    }
}

/// A compiled rasterization-rate map (`MTLRasterizationRateMap`) and its parameter data.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RasterizationRateMap {
    screen_size: [u32; 2],
    layers: Vec<[RateMapAxis; 2]>,
    /// The vertical axis, every layer's (a map of no layers has one too).
    vertical: RateMapAxis,
    words: Vec<u32>,
}

impl RasterizationRateMap {
    /// The alignment, in bytes, of the parameter data in its buffer
    /// (`parameterBufferSizeAndAlign.align`).
    pub const PARAMETER_DATA_ALIGNMENT: u64 = 4;

    /// The parameter data's size in bytes (`parameterBufferSizeAndAlign.size`): 24,624, Metal's
    /// for every map ([`Self::parameter_data_size`]).
    pub const PARAMETER_DATA_SIZE: u64 = format::BYTES as u64;

    /// Compile `desc` (`newRasterizationRateMapWithDescriptor:`) by this module's rule, Apple
    /// M3's, or refuse it ([`RateMapError::Unpinned`]) outside the bounds the rule is measured in.
    /// A descriptor of no layers compiles to a map of none.
    pub fn new(desc: &RateMapDescriptor) -> Result<RasterizationRateMap, RateMapError> {
        let [width, height] = desc.screen_size;
        for extent in [width, height] {
            if extent > MAX_SCREEN_EXTENT {
                return Err(RateMapError::ScreenExtent { extent });
            }
        }
        for (l, layer) in desc.layers.iter().enumerate() {
            for (axis, samples) in [&layer.horizontal, &layer.vertical].into_iter().enumerate() {
                if samples.is_empty() {
                    return Err(RateMapError::NoSamples { layer: l, axis });
                }
            }
        }
        let unpinned = |reason| Err(RateMapError::Unpinned { reason });
        for extent in [width, height] {
            if extent > MAX_COMPILED_EXTENT {
                return unpinned(UnpinnedCompile::ScreenExtent { extent });
            }
        }
        if desc.layers.len() > MAX_COMPILED_LAYERS {
            return unpinned(UnpinnedCompile::LayerCount {
                count: desc.layers.len(),
            });
        }
        for (l, layer) in desc.layers.iter().enumerate() {
            let axes = [(&layer.horizontal, width), (&layer.vertical, height)];
            for (axis, (samples, extent)) in axes.into_iter().enumerate() {
                if samples.len() < 2 {
                    continue;
                }
                // An axis of no screen pixels has no coordinate the quality function is read at,
                // so its one tile is the degenerate one whatever the samples: Apple's bytes for a
                // 0 by 64 map of two horizontal samples are those of its one-sample map, measured.
                if extent == 0 {
                    continue;
                }
                if samples.len() - 1 > extent as usize {
                    return unpinned(UnpinnedCompile::DenseSamples {
                        layer: l,
                        axis,
                        samples: samples.len(),
                        extent,
                    });
                }
                if let Some(index) = samples.iter().position(|q| !(0.0..=1.0).contains(q)) {
                    return unpinned(UnpinnedCompile::SampleOutOfRange {
                        layer: l,
                        axis,
                        index,
                    });
                }
                let max = MAX_MEASURED_SAMPLES[desc.layers.len() - 1];
                if samples.len() > max && samples.iter().any(|&q| q != samples[0]) {
                    return unpinned(UnpinnedCompile::SampleCount {
                        layer: l,
                        axis,
                        samples: samples.len(),
                        max,
                    });
                }
            }
        }
        let verticals: Vec<&[f32]> = desc.layers.iter().map(|l| l.vertical.as_slice()).collect();
        // A map of no layers has a vertical axis all the same, of no function: the least
        // quality everywhere, as Apple's compile gives it.
        let vertical = RateMapAxis::compile(height, &verticals);
        let layers = desc
            .layers
            .iter()
            .map(|layer| {
                [
                    RateMapAxis::compile(width, &[&layer.horizontal]),
                    vertical.clone(),
                ]
            })
            .collect();
        Ok(RasterizationRateMap::build(
            desc.screen_size,
            layers,
            &vertical,
        ))
    }

    /// [`Self::new`] of `desc`, its tables written over `earlier`, the parameter data's words a
    /// map held, as Metal's private `resetUsingDescriptor:` writes them: every entry this map's
    /// tables write ([`Self::words`] holds which: each table's lines, and zero words past them),
    /// and every other word as `earlier` holds it. So a map reset to fewer layers, or to a smaller
    /// screen, keeps the entries past its own (measured on an Apple M3 Max under macOS 27.0.1:
    /// every word of maps of no layers reset from maps of one and two layers and from a taller
    /// screen). Words past `earlier`'s are zero. Only the parameter data differs from
    /// [`Self::new`]'s map: the CPU coordinate maps read no entry past a map's own.
    pub fn new_over(
        desc: &RateMapDescriptor,
        earlier: &[u32],
    ) -> Result<RasterizationRateMap, RateMapError> {
        let mut map = RasterizationRateMap::new(desc)?;
        let mut words = vec![0u32; format::WORDS as usize];
        let kept = earlier.len().min(words.len());
        words[..kept].copy_from_slice(&earlier[..kept]);
        map.encode_into(&mut words);
        map.words = words;
        Ok(map)
    }

    /// Whether every layer's every tile spans as many screen pixels as physical ones: the map
    /// is the identity inside the screen.
    pub fn is_identity(&self) -> bool {
        self.layers.iter().all(|axes| {
            axes.iter()
                .all(|axis| axis.tiles.iter().all(|&(s, p)| s == p))
        })
    }

    /// The map of these compiled layers, each `[horizontal, vertical]`, over `screen_size`: at
    /// least one and at most two layers, each axis covering the map's screen extent, the layers
    /// sharing one vertical axis (what Metal's parameter data holds).
    pub fn from_layers(
        screen_size: [u32; 2],
        layers: Vec<[RateMapAxis; 2]>,
    ) -> Result<RasterizationRateMap, RateMapError> {
        let [_, vertical] = layers.first().ok_or(RateMapError::NoLayers)?;
        if layers.len() > MAX_COMPILED_LAYERS {
            return Err(RateMapError::LayerCount {
                count: layers.len(),
            });
        }
        for (l, layer) in layers.iter().enumerate() {
            for (a, (axis, extent)) in layer.iter().zip(screen_size).enumerate() {
                if axis.screen_extent() != extent {
                    return Err(RateMapError::ExtentMismatch { layer: l, axis: a });
                }
            }
            if layer[1] != *vertical {
                return Err(RateMapError::VerticalAxis { layer: l });
            }
        }
        let vertical = vertical.clone();
        Ok(RasterizationRateMap::build(screen_size, layers, &vertical))
    }

    /// The map of `layers` (at most two, each axis's tiles of 32 physical pixels over at most
    /// 16384 screen pixels) and the vertical axis `vertical` (every layer's), with its parameter
    /// data.
    fn build(
        screen_size: [u32; 2],
        layers: Vec<[RateMapAxis; 2]>,
        vertical: &RateMapAxis,
    ) -> RasterizationRateMap {
        let mut map = RasterizationRateMap {
            screen_size,
            layers,
            vertical: vertical.clone(),
            words: Vec::new(),
        };
        let mut words = vec![0u32; format::WORDS as usize];
        map.encode_into(&mut words);
        map.words = words;
        map
    }

    /// Write its tables' entries into `words` (the parameter data's), leaving every other word.
    fn encode_into(&self, words: &mut [u32]) {
        for (l, [horizontal, _]) in (0u32..).zip(&self.layers) {
            encode(
                words,
                horizontal,
                format::table(l, 0, false),
                format::table(l, 0, true),
            );
        }
        encode(
            words,
            &self.vertical,
            format::table(0, 1, false),
            format::table(0, 1, true),
        );
    }

    /// `screenSize`.
    pub fn screen_size(&self) -> [u32; 2] {
        self.screen_size
    }

    /// `layerCount`.
    pub fn layer_count(&self) -> u32 {
        self.layers.len() as u32
    }

    /// `physicalGranularity`: the width and height of every tile but each axis's last, 32 by 32
    /// (a map of no layers answers the same on Metal).
    pub fn physical_granularity(&self) -> [u32; 2] {
        [PHYSICAL_GRANULARITY; 2]
    }

    /// `physicalSizeForLayer:`, or `None` past the layers.
    pub fn physical_size(&self, layer: u32) -> Option<[u32; 2]> {
        let [x, y] = self.layers.get(layer as usize)?;
        Some([x.physical_extent(), y.physical_extent()])
    }

    /// Layer `layer`'s axes, `[horizontal, vertical]`.
    pub fn layer(&self, layer: u32) -> Option<&[RateMapAxis; 2]> {
        self.layers.get(layer as usize)
    }

    /// The parameter data's size in bytes (`parameterBufferSizeAndAlign.size`): 24,624, Metal's
    /// for every map.
    pub fn parameter_data_size(&self) -> u64 {
        u64::from(format::BYTES)
    }

    /// The parameter data's words, Apple's layout (`air::canon::rate_map_data`).
    pub fn words(&self) -> &[u32] {
        &self.words
    }

    /// The parameter data (little-endian words): what `copyParameterDataToBuffer:offset:` writes,
    /// Metal's bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// The layer whose tables Apple's CPU map reads for `layer`: past the map's layers, the last.
    fn cpu_layer(&self, layer: u32) -> u32 {
        layer.min(self.layer_count().saturating_sub(1))
    }

    /// The physical extent of layer `layer`'s tables: a map of no layers has a vertical axis and
    /// no horizontal one.
    fn physical_extents(&self, layer: u32) -> [u32; 2] {
        let width = self
            .layers
            .get(layer as usize)
            .map_or(0, |[x, _]| x.physical_extent());
        [width, self.vertical.physical_extent()]
    }

    /// The vertical axis, every layer's; a map of no layers has one too (the module docs).
    pub fn vertical(&self) -> &RateMapAxis {
        &self.vertical
    }

    /// `mapScreenToPhysicalCoordinates:forLayer:`: where screen coordinate `c` lands in layer
    /// `layer`'s physical space, as Apple's CPU map answers it (a layer past the map's reads the
    /// last; past the screen the first or last tile's line extends). Over a map of no layers,
    /// where Metal's query ends the process, the decode of its bytes: 0 across.
    pub fn screen_to_physical(&self, c: [f32; 2], layer: u32) -> [f32; 2] {
        format::screen_to_physical(&self.words, self.cpu_layer(layer), c, self.screen_size)
    }

    /// A shader's `map_screen_to_physical_coordinates(c, layer)` over this map's parameter data,
    /// as Apple's GPU decoder computes it, and the translated decoder on a device that reads
    /// `float` denormals as Metal does: the granule of `c` plus half a pixel, two roundings,
    /// denormals flushed, and the layer's tables unclamped (a layer the map lacks reads the words
    /// where its tables would be).
    pub fn shader_screen_to_physical(&self, c: [f32; 2], layer: u32) -> [f32; 2] {
        format::shader_screen_to_physical(&self.words, layer, c)
    }

    /// `mapPhysicalToScreenCoordinates:forLayer:`: the screen coordinate of physical coordinate
    /// `c` in layer `layer`, as Apple's CPU map answers it (a layer past the map's reads the
    /// last; past the physical extent the first or last tile's line extends). Over a map of no
    /// layers, as [`Self::screen_to_physical`].
    pub fn physical_to_screen(&self, c: [f32; 2], layer: u32) -> [f32; 2] {
        let layer = self.cpu_layer(layer);
        format::physical_to_screen(&self.words, layer, c, self.physical_extents(layer))
    }

    /// A shader's `map_physical_to_screen_coordinates(c, layer)` over this map's parameter data,
    /// as Apple's GPU decoder computes it, and the translated decoder on a device that reads
    /// `float` denormals as Metal does: denormals flushed, and the layer's tables unclamped, as
    /// [`Self::shader_screen_to_physical`].
    pub fn shader_physical_to_screen(&self, c: [f32; 2], layer: u32) -> [f32; 2] {
        format::shader_physical_to_screen(&self.words, layer, c)
    }
}

/// Write `axis`'s tables into `words` at `to_screen` and `to_physical` (the first below the
/// second, each a table's words inside `words`).
fn encode(words: &mut [u32], axis: &RateMapAxis, to_screen: u32, to_physical: u32) {
    let table = format::TABLE_WORDS as usize;
    let (low, high) = words.split_at_mut((to_physical as usize).min(words.len()));
    let at = to_screen as usize;
    if let (Some(screen), Some(physical)) = (low.get_mut(at..at + table), high.get_mut(..table)) {
        format::encode_axis(&axis.tiles, screen, physical);
    }
}

mod regions;
pub use regions::{RateMapRegion, RateMapRegions, RegionError, RegionLimits};

#[cfg(test)]
mod tests;

#[cfg(test)]
mod apple_tests;
