//! `RasterizationRateMap` (`MTLRasterizationRateMap`), `Device::new_rasterization_rate_map` and
//! `RasterizationRateMap::reset_using_descriptor` (Metal's bytes and arithmetic).
//!
//! - **The compile layer's map.** A map is the compile layer's compile of its descriptor
//!   ([`metal2vulkan::bind::RasterizationRateMap::new`], Apple's rule), with the descriptor's
//!   label. Its parameter data is Metal's: 24,624 bytes in Apple's layout, which a function's
//!   `rasterization_rate_map_decoder` reads from a buffer bound like any other, with Apple's GPU
//!   arithmetic; and its coordinate maps are Apple's CPU arithmetic over the same bytes. Measured
//!   on an Apple M4 and an M3 Max: the bytes of every map probed and every answer, bit for bit. A
//!   render pass over a map that is not the identity draws by regions whose lines are the tiles'
//!   exact lines, a near-equivalent of Apple's fixed-point ones, so the map's creation (or the
//!   reset that compiles it) announces the `rasterization-rate-map` degradation.
//! - **What Metal's API validation stops**, measured by our probes on an M3
//!   Max and an M4 alike, is refused at creation, `Invalid` with [`RateMapProblem`]: a screen width
//!   or height past 32768, and a layer count past 2, but for a `Mutable` descriptor,
//!   whose map plain Metal makes whatever it holds (below).
//!   Its two other stops are **not** the runtime's, because plain Metal makes the map. One is a
//!   screen side of 0, whose map the runtime compiles as Metal does: its physical extent on that
//!   axis is 0, its parameter data is Metal's, and `mapScreenToPhysicalCoordinates:forLayer:`
//!   over it ends Metal's process, so the runtime refuses that one call, `Invalid` with
//!   [`RateMapProblem::EmptyScreen`] (`mapPhysicalToScreenCoordinates:` answers on Metal, and the
//!   runtime answers with it). The other is a layer count of 0 (below).
//!   The count is Metal's `layerCount`:
//!   the layers from index 0 up to the first without samples on an axis, which Metal's descriptor
//!   holds as nil; the map is made of those, and the layers after are ignored, as Metal ignores
//!   them. Metal's validation also stops a count of 0, but Metal makes such a map without it, and
//!   so does the runtime: its parameter data is Metal's (no horizontal tables, a vertical axis of
//!   the least quality), and a render pass over it runs its load, clear, and store actions and
//!   each draw's vertex function, and rasterizes nothing.
//! - **Where the compile is this crate's bound.** Metal takes a screen of up to 32768 pixels a
//!   side and any samples; the compile layer compiles only where Apple's compile is measured,
//!   and refuses the rest, `Error::RateMap` with its
//!   [`UnpinnedCompile`](metal2vulkan::bind::UnpinnedCompile): a screen past 16384 a side (where Apple's
//!   compile crashes or corrupts the map) and sample densities whose compile is not measured. An
//!   axis of no screen pixels is not among them: it has one tile of no screen and no physical
//!   pixels whatever its samples, so no coordinate reads its quality function and the density
//!   bound is never applied to it.
//! - **Metal's answers.** `screenSize` is the descriptor's, depth 1; `physicalGranularity` 32 by
//!   32, depth 1; `parameterBufferSizeAndAlign` 24,624 bytes, align 4; the coordinate maps past
//!   the layers read the last layer (both Macs, probes 7 and 9). `physicalSizeForLayer:` past the
//!   layers is the layers' shared vertical extent on Metal too, and the width the map last held
//!   for that layer below 2, 0 for one it never held (a reset leaves it, below); past layer 1
//!   Metal reads past its own layer table (probe 9: a screen side, the vertical extent or a small
//!   number, by index; a map of no layers answers 16 by 16 at layer 2): the runtime answers width
//!   0 there, and its vertical extent. A map of no layers' coordinate maps end the process on
//!   Metal (all 12 probed queries): the runtime refuses them, `Invalid` with
//!   [`RateMapProblem::NoLayers`].
//! - **Mutability** (Apple's private `mutability`, the M3 Max under macOS 27.0.1, each case in a
//!   process of its own). A map answers its descriptor's (`Immutable` unless set). A map of a
//!   `Mutable` descriptor is not compiled until a reset: it answers the descriptor's screen size
//!   and layer count, a physical size of 0 by 0 (depth 1) at every layer, the screen-to-physical
//!   map (0, 0), granularity and parameter size as every map, and its copy writes 24,624 zeros;
//!   its physical-to-screen map ends Metal's process, refused here, `Invalid` with
//!   [`RateMapProblem::NotCompiled`]. Plain Metal makes such a map of a descriptor its API
//!   validation stops (three layers, a screen side of 0 or of 40000), and so does the runtime: it
//!   answers the same, and its reset compiles as any. A pass over it runs its load, clear and store actions and
//!   each draw's vertex function, and rasterizes nothing. A map of `Default` (0) or any other
//!   mutability is compiled at creation, as an immutable one.
//! - **A reset** (`resetUsingDescriptor:`, Apple's private call, on a map of any mutability)
//!   compiles the descriptor into the map, which keeps its label and mutability. Metal's API
//!   validation does not look at the reset's descriptor; a layer count past 2 ends Metal's
//!   process (refused, `Invalid` with [`RateMapProblem::LayerCount`]), and the compile layer's
//!   bounds hold (`Error::RateMap`); a reset to a screen side of 0 compiles, as a fresh map of one
//!   does. Metal writes the new map's tables over the old: every entry the
//!   new map writes (`metal2vulkan::bind::RasterizationRateMap::new_over`), and every other word
//!   as it was (a map reset to no layers keeps its horizontal tables, and its physical width at
//!   each layer below 2), so the parameter data is Metal's. A render pass reads the map as the GPU
//!   runs it: a reset after the encoder is made, after its draws, after `end_encoding`, and after
//!   `commit` while the command buffer waits for an event shows in the pass, and one in two
//!   passes of one command buffer alike; a reset after the command buffer completed does not.
//!   The runtime plans each draw by the map as the encoder takes it; a pass that took a map since
//!   reset is encoded again, from its log, before the submission of its command buffer
//!   (`rate_mapped_pass.rs`).
//! - **A map of no layers made, or reset, after layers**. The device notes when it releases a map
//!   of one or more layers (its last handle dropped); a map of no layers made after that, or reset
//!   from a map of layers, is one whose passes Metal draws (an Apple M4 and an Apple M3 Max drew a
//!   64-pixel map's rows 0 to 15 as with no map), where the runtime draws nothing, so a draw in
//!   such a pass announces `zero-layer-map-after-release` (`rate_mapped_pass.rs`).
//! - **The copy.** `copy_parameter_data_to_buffer` writes the parameter data through the buffer's
//!   mapping, at any offset, as Metal takes a misaligned one. It refuses a buffer of another
//!   device, a `Private` buffer (Metal's copy ends the process there), and a write past the
//!   buffer's length (Metal's writes past it). The bytes are Metal's.

use super::buffer::Buffer;
use super::capabilities::RATE_MAP_LAYERS;
use super::degradation::Degradation;
use super::device::{lock, Device};
use super::error::{internal, invalid, Error, Problem};
use super::heap::SizeAndAlign;
use super::values::{Coordinate2D, Size};
use metal2vulkan::bind::{RateMapDescriptor, RateMapError, UnpinnedCompile};
use metal2vulkan::metal::Mutability;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

/// The largest screen width or height Metal's API validation takes ("screenSize.width (0)
/// must be >0 and <=32768"); of its two bounds the runtime holds to this one alone, plain Metal
/// compiling a side of 0 (the module docs).
const MAX_SCREEN_EXTENT: u32 = 32768;

/// What a rasterization-rate map call did wrong (`Problem::RateMap`): what Metal's API validation
/// stops at `newRasterizationRateMapWithDescriptor:` (an M3 Max and an M4), and what ends Metal's
/// process at a reset or a coordinate map (the M3 Max under macOS 27.0.1).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RateMapProblem {
    /// A screen width or height past 32768. Metal's API validation stops a side of 0 too, and
    /// plain Metal compiles such a map, which the runtime makes (the module docs).
    ///
    /// Basis: unknown (plain Metal's answer for a screen side past 32768 is unmeasured; the side of
    /// 0 its validation stops is compiled by plain Metal and is not refused).
    ScreenSize {
        /// The screen width given.
        width: u32,
        /// The screen height given.
        height: u32,
    },
    /// A layer count past 2 ("layerCount (3) must be >0 and <=2"): the layers from index 0 up to
    /// the first without samples on an axis, the count Metal's descriptor answers. A count of 0,
    /// which Metal's validation also stops, makes a map of no layers, as plain Metal does (the
    /// module docs). At a reset, which Metal's validation does not check, such a count ends
    /// Metal's process.
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal's process ends, SIGBUS, at
    /// `newRasterizationRateMapWithDescriptor:` of 3 and of 4 layers, where 1 and 2 make the map).
    LayerCount {
        /// The descriptor's layer count.
        count: usize,
    },
    /// A coordinate map of a map of no layers, which ends Metal's process.
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal's process ends, SIGSEGV,
    /// at a coordinate map of a map of no layers, which it makes).
    NoLayers,
    /// `mapScreenToPhysicalCoordinates:forLayer:` of a map with a screen width or height of 0,
    /// which ends Metal's process (`mapPhysicalToScreenCoordinates:` of such a map answers, and
    /// the runtime answers with it).
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal ends the process, SIGSEGV,
    /// at `mapScreenToPhysicalCoordinates:forLayer:` of a map of a 64 by 0 screen, where
    /// `mapPhysicalToScreenCoordinates:forLayer:` of that same map answers).
    EmptyScreen {
        /// The map's screen width.
        width: u32,
        /// The map's screen height.
        height: u32,
    },
    /// `mapPhysicalToScreenCoordinates:forLayer:` of a mutable map not yet reset, which ends
    /// Metal's process.
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal ends the process, SIGSEGV,
    /// at `mapPhysicalToScreenCoordinates:forLayer:` of a map of a `Mutable` descriptor not yet
    /// reset, where `mapScreenToPhysicalCoordinates:forLayer:` of that same map answers (0, 0)).
    NotCompiled,
}

/// A rasterization-rate map (`MTLRasterizationRateMap`). Cheap to clone (a retain); `Send +
/// Sync`. Two handles are equal when they are the same map.
#[derive(Clone)]
pub struct RasterizationRateMap(Arc<MapShared>);

struct MapShared {
    device: Device,
    label: Option<String>,
    mutability: Mutability,
    /// What the map is now: a reset puts a new one in its place, so a pass keeps the one it took
    /// (`rate_mapped_pass.rs`).
    state: Mutex<Arc<MapState>>,
}

/// What a map is at one time (the module docs).
pub(super) struct MapState {
    shape: Shape,
    /// Each layer's physical width below 2 as the map holds it: a layer the map lacks keeps the
    /// width an earlier compile gave it, 0 where none did.
    widths: [u32; 2],
    /// A map of no layers made after its device released a map of layers, or reset from layers:
    /// one whose passes Metal draws from the memory they left (the module docs).
    after_release: bool,
}

/// A map's compile, or a mutable map's descriptor's screen size and layer count before its first
/// reset.
enum Shape {
    Compiled(metal2vulkan::bind::RasterizationRateMap),
    Unset { screen_size: [u32; 2], layers: u32 },
}

impl MapState {
    /// The compile layer's map, `None` for a mutable map not yet reset.
    pub(super) fn compiled(&self) -> Option<&metal2vulkan::bind::RasterizationRateMap> {
        match &self.shape {
            Shape::Compiled(map) => Some(map),
            Shape::Unset { .. } => None,
        }
    }

    /// `screenSize`'s width and height.
    pub(super) fn screen_size(&self) -> [u32; 2] {
        match &self.shape {
            Shape::Compiled(map) => map.screen_size(),
            Shape::Unset { screen_size, .. } => *screen_size,
        }
    }

    /// `layerCount`.
    pub(super) fn layer_count(&self) -> u32 {
        match &self.shape {
            Shape::Compiled(map) => map.layer_count(),
            Shape::Unset { layers, .. } => *layers,
        }
    }

    /// Whether every layer is the first's, so each slice is rasterized alike: a map not yet
    /// reset rasterizes nothing in any.
    pub(super) fn layers_alike(&self) -> bool {
        match &self.shape {
            Shape::Compiled(map) => {
                (1..map.layer_count()).all(|layer| map.layer(layer) == map.layer(0))
            }
            Shape::Unset { .. } => true,
        }
    }

    /// Whether its passes draw from the memory an earlier map left (the module docs).
    pub(super) fn made_after_release(&self) -> bool {
        self.after_release
    }

    /// `physicalSizeForLayer:` (the module docs): a layer's physical width and height, depth 1;
    /// past the layers, the width the map holds for it below layer 2 (else 0) and the vertical
    /// extent the layers share; 0 by 0 at every layer of a map not yet reset.
    pub(super) fn physical_size(&self, layer: u64) -> Size {
        let Shape::Compiled(map) = &self.shape else {
            return Size::new(0, 0, 1);
        };
        if let Some([width, height]) = u32::try_from(layer).ok().and_then(|l| map.physical_size(l))
        {
            return Size::new(width.into(), height.into(), 1);
        }
        let width = usize::try_from(layer)
            .ok()
            .and_then(|l| self.widths.get(l))
            .copied()
            .unwrap_or(0);
        Size::new(width.into(), map.vertical().physical_extent().into(), 1)
    }

    /// The state of a fresh map of `descriptor`, the device having released a map of layers
    /// where `released`: compiled, where Metal's API validation and the compile layer let it, but
    /// for a `Mutable` one, which plain Metal makes of any descriptor (the module docs).
    pub(super) fn made(descriptor: &RateMapDescriptor, released: bool) -> Result<MapState, Error> {
        if descriptor.mutability == Mutability::Mutable {
            return Ok(MapState {
                shape: Shape::Unset {
                    screen_size: descriptor.screen_size,
                    layers: counted(descriptor) as u32,
                },
                widths: [0; 2],
                after_release: false,
            });
        }
        let map = compile(descriptor)?;
        Ok(MapState::of(map, [0; 2], released))
    }

    /// The state of `map`, its widths below layer 2 over `earlier`'s, and of no layers drawing
    /// from what an earlier map left where `after` (the module docs).
    fn of(
        map: metal2vulkan::bind::RasterizationRateMap,
        mut widths: [u32; 2],
        after: bool,
    ) -> MapState {
        for (layer, width) in widths.iter_mut().enumerate() {
            if let Some([held, _]) = map.physical_size(layer as u32) {
                *width = held;
            }
        }
        MapState {
            after_release: map.layer_count() == 0 && after,
            shape: Shape::Compiled(map),
            widths,
        }
    }

    /// The parameter data's words as the map holds them: 24,624 zero bytes before a mutable
    /// map's first reset.
    pub(super) fn words(&self) -> std::borrow::Cow<'_, [u32]> {
        match &self.shape {
            Shape::Compiled(map) => std::borrow::Cow::Borrowed(map.words()),
            Shape::Unset { .. } => std::borrow::Cow::Owned(vec![0; WORDS]),
        }
    }

    /// The state a reset to `descriptor` leaves a map of this state in, the device having
    /// released a map of layers where `released` (the module docs): `descriptor` compiled over
    /// its tables, its widths kept past the new layers. Refused where Metal ends the process, a
    /// layer count past 2, and where the compile layer does not compile it.
    pub(super) fn reset(
        &self,
        descriptor: &RateMapDescriptor,
        released: bool,
    ) -> Result<MapState, Error> {
        let count = counted(descriptor);
        if count as u64 > RATE_MAP_LAYERS {
            return Err(invalid(
                "reset_using_descriptor",
                Problem::RateMap(RateMapProblem::LayerCount { count }),
            ));
        }
        let map = compile_over(descriptor, Some(&self.words()))?;
        let after = self.compiled().is_some_and(|map| map.layer_count() > 0)
            || self.after_release
            || released;
        Ok(MapState::of(map, self.widths, after))
    }
}

/// `physicalGranularity`'s width and height, every map's: Metal's parameter data holds a line per
/// 32 pixels, compiled or not (the M3 Max under macOS 27.0.1).
const GRANULARITY: u64 = 32;

/// The parameter data's 32-bit words.
const WORDS: usize = (metal2vulkan::bind::RasterizationRateMap::PARAMETER_DATA_SIZE / 4) as usize;

impl Drop for MapShared {
    fn drop(&mut self) {
        let layered = self
            .state
            .get_mut()
            .is_ok_and(|state| state.compiled().is_some_and(|map| map.layer_count() > 0));
        if layered {
            self.device
                .shared()
                .layered_rate_map_released
                .store(true, Ordering::Relaxed);
        }
    }
}

impl std::fmt::Debug for RasterizationRateMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state();
        f.debug_struct("RasterizationRateMap")
            .field("label", &self.0.label)
            .field("mutability", &self.0.mutability)
            .field("screen_size", &state.screen_size())
            .field("layer_count", &state.layer_count())
            .field("compiled", &state.compiled().is_some())
            .finish()
    }
}

impl PartialEq for RasterizationRateMap {
    fn eq(&self, other: &RasterizationRateMap) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for RasterizationRateMap {}

impl Device {
    /// `newRasterizationRateMapWithDescriptor:` (the module docs): refused `Invalid` where Metal's
    /// API validation stops the descriptor, and `Error::RateMap` where the compile layer does not
    /// compile it. A `Mutable` descriptor's map is made of any descriptor, as plain Metal makes
    /// it, and compiled at its first reset.
    pub fn new_rasterization_rate_map(
        &self,
        descriptor: &RateMapDescriptor,
    ) -> Result<RasterizationRateMap, Error> {
        let released = self
            .shared()
            .layered_rate_map_released
            .load(Ordering::Relaxed);
        let state = MapState::made(descriptor, released)?;
        degrade_compiled(self, &state);
        Ok(RasterizationRateMap(Arc::new(MapShared {
            device: self.clone(),
            label: descriptor.label.clone(),
            mutability: descriptor.mutability,
            state: Mutex::new(Arc::new(state)),
        })))
    }
}

/// Announce that a pass over the map `state` holds draws by regions of this crate's lines, where
/// it is not the identity (the module docs).
fn degrade_compiled(device: &Device, state: &MapState) {
    let Some(map) = state.compiled() else {
        return;
    };
    if map.is_identity() {
        return;
    }
    device.degrade(Degradation::RasterizationRateMap, || {
        let [width, height] = map.screen_size();
        (
            format!("a rasterization rate map of {width} by {height} screen pixels"),
            "a render pass over it draws by regions whose lines are its tiles' exact lines, where \
             Apple's are its fixed-point ones, and a mutable map's pass, which Metal rasterizes \
             past the screen's edge along its last tiles' lines, stops at the map's physical \
             size; its parameter data and coordinate maps are Metal's"
                .to_owned(),
        )
    });
}

/// Metal's `layerCount` of `descriptor`: its layers up to the first without samples on an axis,
/// which Metal's descriptor holds as nil.
fn counted(descriptor: &RateMapDescriptor) -> usize {
    descriptor
        .layers
        .iter()
        .take_while(|layer| !(layer.horizontal.is_empty() || layer.vertical.is_empty()))
        .count()
}

/// What Metal's API validation stops at `newRasterizationRateMapWithDescriptor:` (the module
/// docs), which the runtime refuses of a map compiled at creation.
fn validate(descriptor: &RateMapDescriptor) -> Result<(), Error> {
    const CALL: &str = "new_rasterization_rate_map";
    let refused = |problem| invalid(CALL, Problem::RateMap(problem));
    let [width, height] = descriptor.screen_size;
    if width > MAX_SCREEN_EXTENT || height > MAX_SCREEN_EXTENT {
        return Err(refused(RateMapProblem::ScreenSize { width, height }));
    }
    let count = counted(descriptor);
    if count as u64 > RATE_MAP_LAYERS {
        return Err(refused(RateMapProblem::LayerCount { count }));
    }
    Ok(())
}

/// The compile layer's map of `descriptor`, or why Metal's API validation (`Invalid`) or the
/// compile layer (`Error::RateMap`) stops it (the module docs).
pub(super) fn compile(
    descriptor: &RateMapDescriptor,
) -> Result<metal2vulkan::bind::RasterizationRateMap, Error> {
    validate(descriptor)?;
    compile_over(descriptor, None)
}

/// The compile layer's map of `descriptor`, Metal's layers of it, written over `earlier`'s words
/// where given (`metal2vulkan::bind::RasterizationRateMap::new_over`): `Error::RateMap` where the
/// compile layer does not compile it, a screen side past its bound among them.
fn compile_over(
    descriptor: &RateMapDescriptor,
    earlier: Option<&[u32]>,
) -> Result<metal2vulkan::bind::RasterizationRateMap, Error> {
    let count = counted(descriptor);
    let counted;
    let descriptor = if count == descriptor.layers.len() {
        descriptor
    } else {
        let mut kept = descriptor.clone();
        kept.layers.truncate(count);
        counted = kept;
        &counted
    };
    let made = match earlier {
        None => metal2vulkan::bind::RasterizationRateMap::new(descriptor),
        Some(words) => metal2vulkan::bind::RasterizationRateMap::new_over(descriptor, words),
    };
    made.map_err(|error| match error {
        RateMapError::Unpinned { reason } => Error::RateMap(reason),
        RateMapError::ScreenExtent { extent } => {
            Error::RateMap(UnpinnedCompile::ScreenExtent { extent })
        }
        other => internal(&format!(
            "the compile layer refused a rate map Metal compiles: {other}"
        )),
    })
}

/// `physicalSizeForLayer:` of a fresh map compiled as `map` (the module docs).
#[cfg(test)]
pub(super) fn physical_size(map: &metal2vulkan::bind::RasterizationRateMap, layer: u64) -> Size {
    MapState::of(map.clone(), [0; 2], false).physical_size(layer)
}

/// A layer index as the compile layer's coordinate maps take it: one past `u32` reads the last
/// layer, as every index past the layers does.
fn layer_index(layer: u64) -> u32 {
    u32::try_from(layer).unwrap_or(u32::MAX)
}

impl RasterizationRateMap {
    /// `device`.
    pub fn device(&self) -> &Device {
        &self.0.device
    }

    /// What the map is now, which a pass takes as its encoder opens (`rate_mapped_pass.rs`).
    pub(super) fn state(&self) -> Arc<MapState> {
        Arc::clone(&lock(&self.0.state))
    }

    /// The map's identity among the device's, while a handle of it lives.
    pub(super) fn id(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }

    /// Whether `state` is what the map is now: a pass that took another was planned by a map
    /// since reset (`rate_mapped_pass.rs`).
    pub(super) fn holds(&self, state: &Arc<MapState>) -> bool {
        Arc::ptr_eq(&lock(&self.0.state), state)
    }

    /// `label`: the descriptor's it was made with; a reset keeps it.
    pub fn label(&self) -> Option<&str> {
        self.0.label.as_deref()
    }

    /// `mutability` (Apple's private property): the descriptor's it was made with; a reset keeps
    /// it. A `Mutable` map is compiled only at its first reset (the module docs).
    pub fn mutability(&self) -> Mutability {
        self.0.mutability
    }

    /// `screenSize`: the descriptor's width and height, depth 1 (as Metal answers on an M3 Max
    /// and an M4, where its header says 0).
    pub fn screen_size(&self) -> Size {
        let [width, height] = self.state().screen_size();
        Size::new(width.into(), height.into(), 1)
    }

    /// `physicalGranularity`: the physical width and height of every tile but each axis's last,
    /// 32 by 32, depth 1, for every map.
    pub fn physical_granularity(&self) -> Size {
        Size::new(GRANULARITY, GRANULARITY, 1)
    }

    /// `layerCount`: the compiled map's, or a mutable map's descriptor's before its first reset.
    pub fn layer_count(&self) -> u64 {
        self.state().layer_count().into()
    }

    /// `parameterBufferSizeAndAlign`: 24,624 bytes, aligned to 4, Metal's for every map.
    pub fn parameter_buffer_size_and_align(&self) -> SizeAndAlign {
        SizeAndAlign {
            size: metal2vulkan::bind::RasterizationRateMap::PARAMETER_DATA_SIZE,
            align: metal2vulkan::bind::RasterizationRateMap::PARAMETER_DATA_ALIGNMENT,
        }
    }

    /// `physicalSizeForLayer:` (the module docs).
    pub fn physical_size_for_layer(&self, layer: u64) -> Size {
        self.state().physical_size(layer)
    }

    /// `mapScreenToPhysicalCoordinates:forLayer:`: where `screen` lands in the layer's physical
    /// space, as Apple's CPU map answers it, bit for bit; a layer past the map's reads the last.
    /// (0, 0) before a mutable map's first reset. Refused where Metal's ends the process,
    /// `Invalid` with [`RateMapProblem`]: over a map of no layers (`NoLayers`), and over one of a
    /// screen width or height of 0 (`EmptyScreen`).
    pub fn map_screen_to_physical_coordinates(
        &self,
        screen: Coordinate2D,
        layer: u64,
    ) -> Result<Coordinate2D, Error> {
        const CALL: &str = "map_screen_to_physical_coordinates";
        let state = self.state();
        let Some(map) = state.compiled() else {
            return Ok(Coordinate2D::new(0.0, 0.0));
        };
        if map.layer_count() == 0 {
            return Err(invalid(CALL, Problem::RateMap(RateMapProblem::NoLayers)));
        }
        let [width, height] = map.screen_size();
        if width == 0 || height == 0 {
            return Err(invalid(
                CALL,
                Problem::RateMap(RateMapProblem::EmptyScreen { width, height }),
            ));
        }
        let [x, y] = map.screen_to_physical([screen.x, screen.y], layer_index(layer));
        Ok(Coordinate2D::new(x, y))
    }

    /// `mapPhysicalToScreenCoordinates:forLayer:`: the screen coordinate of `physical` in the
    /// layer, as Apple's CPU map answers it, bit for bit; a layer past the map's reads the last.
    /// Refused where Metal's ends the process, `Invalid` with [`RateMapProblem`]: over a map of no
    /// layers (`NoLayers`), and before a mutable map's first reset (`NotCompiled`).
    pub fn map_physical_to_screen_coordinates(
        &self,
        physical: Coordinate2D,
        layer: u64,
    ) -> Result<Coordinate2D, Error> {
        const CALL: &str = "map_physical_to_screen_coordinates";
        let state = self.state();
        let Some(map) = state.compiled() else {
            return Err(invalid(CALL, Problem::RateMap(RateMapProblem::NotCompiled)));
        };
        if map.layer_count() == 0 {
            return Err(invalid(CALL, Problem::RateMap(RateMapProblem::NoLayers)));
        }
        let [x, y] = map.physical_to_screen([physical.x, physical.y], layer_index(layer));
        Ok(Coordinate2D::new(x, y))
    }

    /// `copyParameterDataToBuffer:offset:`: Metal's parameter data, written at `offset` of
    /// `buffer` through its mapping (the module docs); 24,624 zeros before a mutable map's first
    /// reset. Refused for a buffer of another device (`ForeignObject`), a `Private` one
    /// (`StorageMode`), and a write past its length (`RangeOutOfBounds`).
    pub fn copy_parameter_data_to_buffer(&self, buffer: &Buffer, offset: u64) -> Result<(), Error> {
        const CALL: &str = "copy_parameter_data_to_buffer";
        if !buffer.device().same(&self.0.device) {
            return Err(invalid(CALL, Problem::ForeignObject));
        }
        let Some(contents) = buffer.contents() else {
            return Err(invalid(CALL, Problem::StorageMode));
        };
        let state = self.state();
        let words = state.words();
        let size = metal2vulkan::bind::RasterizationRateMap::PARAMETER_DATA_SIZE;
        let length = buffer.length();
        if offset.checked_add(size).is_none_or(|end| end > length) {
            return Err(invalid(
                CALL,
                Problem::RangeOutOfBounds {
                    offset,
                    size,
                    length,
                },
            ));
        }
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        // SAFETY: the buffer's mapping holds its `length` bytes from `contents`, and
        // `offset + size` is at most that. A command that reads or writes those bytes meanwhile
        // races, as on Metal.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                contents.as_ptr().add(offset as usize),
                bytes.len(),
            )
        };
        // A write of host pages a consumer logs is the runtime's own, which its log does not see:
        // reported, so the copies of the pages take it (`dirty_pages.rs`).
        let first = contents.as_ptr() as u64 + offset;
        buffer.wrote_host(first..first + bytes.len() as u64);
        Ok(())
    }

    /// `resetUsingDescriptor:` (Apple's private call, the module docs): `descriptor` compiled
    /// into the map over the tables it holds, its label and mutability kept. Refused where Metal
    /// ends the process, a layer count past 2 (`Invalid` with [`RateMapProblem::LayerCount`]),
    /// and, `Error::RateMap`, where the compile layer does not compile it (Metal's validation
    /// does not look at the reset's descriptor); the map is then as it was.
    /// A pass that took the map before the reset and whose command buffer is not yet submitted
    /// draws by the map as reset, as Metal's GPU reads it (`rate_mapped_pass.rs`).
    pub fn reset_using_descriptor(&self, descriptor: &RateMapDescriptor) -> Result<(), Error> {
        let device = &self.0.device;
        let released = device
            .shared()
            .layered_rate_map_released
            .load(Ordering::Relaxed);
        let state = Arc::new(self.state().reset(descriptor, released)?);
        degrade_compiled(device, &state);
        // In place under the lock a submission reads its mark under: each command buffer
        // committed and not yet submitted whose pass took the map is encoded again now, on this
        // thread, as a rebind's are (`command_buffer/reencode.rs`); one not yet committed is at
        // its commit.
        let again = device
            .shared()
            .rebinds
            .reset_map(self.id(), || *lock(&self.0.state) = state);
        for buffer in again {
            buffer.reencode();
        }
        Ok(())
    }
}
