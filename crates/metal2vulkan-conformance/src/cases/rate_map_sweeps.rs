//! Sweeps of the rasterization-rate space: every axis of it compared at the values where its
//! derived answer **steps**, and at the ends of what a program can state.
//!
//! [`crate::cases::rate_maps`] is this family's other half, and the two ask different questions.
//! That one is about the two descriptors where plain Metal and Metal's own API validation layer
//! disagree, and compares each of them whole, all 24,624 bytes of parameter data included. This one
//! is about **depth**: a rate map is almost nothing but numbers, and an item counted as exercised
//! because one case called it once says nothing about whether the hundred other values of it
//! agree. [`crate::rate_map_space`] is the space these cases walk and the reasoning for every point
//! of it; this module is the four walks.
//!
//! # Every site is something the map derives
//!
//! A case that sets a quality sample and reads it back passes whether or not Metal did anything
//! with the sample, and sweeping that comparison compares nothing thirty times instead of once. So
//! nothing here reads back an input. Each swept point compiles a map and compares what the compile
//! **derived** from it:
//!
//! - `physicalSizeForLayer:` of every layer the map has;
//! - `physicalGranularity`, `screenSize` and `layerCount`, which the compile answers;
//! - `parameterBufferSizeAndAlign`;
//! - `mapScreenToPhysicalCoordinates:forLayer:` and `mapPhysicalToScreenCoordinates:forLayer:`
//!   over [`crate::rate_map_space::GRID`], by their `f32` bits — the two calls that read the tile
//!   tables back out as arithmetic, and the finest-grained derived answer a map has.
//!
//! That the derived answer really does move with the input is not taken on trust:
//! `the_derived_answer_moves_when_a_quality_moves` compiles two maps that differ in one quality
//! sample and fails unless their derived answers differ. A sweep whose points all derived one
//! answer would pass every site and measure nothing, and that test is what stands between this
//! module and that.
//!
//! # Where Metal quantizes, two inputs derive one answer
//!
//! The compile biases every quality by 1/256 and clamps it into `[0.01, 1]`, then lays tiles of
//! `⌊32 / Q⌋` screen pixels — a step function. So below `0.01 - 1/256` every quality derives one
//! answer, at or above `1 - 1/256` every quality derives another, and inside a step every quality
//! derives the step's. [`crate::rate_map_space::same_bucket`] states those pairs **as a
//! prediction**, and `the_quantization_holds_where_the_space_predicts_it` holds the recorded
//! answers to it: a pair declared one bucket whose answers differ is a finding about the clamp, and
//! the sweep straddles each step it names so the pair either side of one compares two answers
//! rather than one twice. That is a real finding about the quantization, not a case passing by
//! coincidence.
//!
//! # Why every case here is safe
//!
//! Nothing reaches the GPU. A rate map is compiled on the CPU, its getters are CPU reads, and both
//! coordinate maps are CPU arithmetic over the map's own parameter data; no command buffer is made
//! and no pass is opened, so no case can fault the device, hang it or trip its watchdog.
//!
//! Every device-limited number is bounded by the device's own answer or by a measured limit, and
//! [`crate::rate_map_space`] holds each shape to it with a test of its own: the layer count by
//! `supportsRasterizationRateMapWithLayerCount:` (1 and 2 granted, 3 and 4 refused on this oracle,
//! which the suite's `device-rate-map-layer-counts` recording states), the screen extent at 1024
//! against Metal's own validation bound of 32,768, and the samples per axis at the most Apple's
//! compile was measured with. The three shapes **measured** to end plain Metal's process are never
//! built: a descriptor of three layers, and either coordinate map over a map of no layers or of a
//! zero screen extent.

use crate::allowance::{Allowance, Justification, Loosening};
use crate::case::{
    made, Abort, Case, CoordinateProbe, Input, MappedCoordinates, MetalSource, RateMapForm,
    RateMapLayer, RateMapRequest, Side,
};
use crate::rate_map_space as space;
use crate::replay::NOTICES_SITE;
use crate::transcript::{Node, Observed, Sites, Value};
use metal2vulkan_runtime::Degradation;

/// The quality sweep: one layer at [`space::SWEEP_EXTENT`], over every quality of
/// [`space::qualities`].
pub const QUALITIES: &str = "rate-map-quality-sweep";

/// The screen sweep: one quality over every extent of [`space::SCREENS`].
pub const SCREENS: &str = "rate-map-screen-sweep";

/// The sample-count sweep: a peak profile at every count from 1 to the most Apple's compile was
/// measured with.
pub const SAMPLES: &str = "rate-map-sample-sweep";

/// The layer sweep: every layer count the device grants, and the two-layer shapes that say which
/// vertical axis the compile keeps.
pub const LAYERS: &str = "rate-map-layer-sweep";

/// The descriptor forms: every way Metal's descriptor object can be built, judged on the map each
/// compiles.
pub const FORMS: &str = "rate-map-descriptor-forms";

/// Which sweep a case is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Walk {
    /// The quality axis, at one extent and one layer.
    Qualities,
    /// The screen-extent axis, at one quality and one layer.
    Screens,
    /// The samples-per-axis axis, at one extent and one layer.
    Samples,
    /// The layer axis, at one extent.
    Layers,
    /// The descriptor's construction forms, at one shape.
    Forms,
}

/// One sweep of the space.
pub struct RateMapSweep {
    name: &'static str,
    walk: Walk,
}

/// Every sweep of the family.
pub fn all() -> Vec<RateMapSweep> {
    vec![
        RateMapSweep {
            name: QUALITIES,
            walk: Walk::Qualities,
        },
        RateMapSweep {
            name: SCREENS,
            walk: Walk::Screens,
        },
        RateMapSweep {
            name: SAMPLES,
            walk: Walk::Samples,
        },
        RateMapSweep {
            name: LAYERS,
            walk: Walk::Layers,
        },
        RateMapSweep {
            name: FORMS,
            walk: Walk::Forms,
        },
    ]
}

/// One point of a sweep: the descriptor it states, and the site its answers sit under.
struct Point {
    /// The site's name, which says which point of the space this is.
    at: String,
    screen: [u32; 2],
    layers: Vec<RateMapLayer>,
    form: RateMapForm,
}

impl Walk {
    /// Every point this walk compares.
    fn points(self) -> Vec<Point> {
        match self {
            // The quality axis. One layer, both of its axes at the swept quality, so a step shows
            // in the width and the height alike and a side that crossed the two reads one of them
            // wrong at every asymmetric point of the other walks.
            // Each quality is swept at the extent its own step is visible at
            // (`space::visible_extent`): a tile span wider than the screen is one tile either side
            // of its step, so a point swept too narrow compares one answer twice however carefully
            // its quality was chosen. The site name carries the extent for that reason, so a reader
            // of a recording can see which screen a point was measured on.
            Walk::Qualities => space::qualities()
                .into_iter()
                .map(|quality| Point {
                    at: format!("{}-at-{}", quality.at, quality.extent),
                    screen: [quality.extent, quality.extent],
                    layers: vec![space::uniform_layer(quality.quality)],
                    form: RateMapForm::ScreenSizeThenSetLayer,
                })
                .collect(),
            // The screen axis. The extent moves on the **width** only, with the height held at the
            // sweep's own extent: an extent that moved on both axes together could not tell a side
            // that reads the width where the height belongs from one that does not.
            Walk::Screens => space::SCREENS
                .iter()
                .map(|extent| Point {
                    at: format!("extent-{extent}"),
                    screen: [*extent, space::SWEEP_EXTENT],
                    layers: vec![space::uniform_layer(space::SCREEN_SWEEP_QUALITY)],
                    form: RateMapForm::ScreenSizeThenSetLayer,
                })
                .collect(),
            // The sample axis. The peak profile on the horizontal axis and one sample on the
            // vertical, so the count moves one axis and the other is the control.
            Walk::Samples => space::sample_counts()
                .into_iter()
                .map(|count| Point {
                    at: format!("samples-{count}"),
                    screen: [space::SWEEP_EXTENT, space::SWEEP_EXTENT],
                    layers: vec![RateMapLayer {
                        horizontal: space::peak_profile(count),
                        vertical: vec![space::PEAK_HIGH],
                    }],
                    form: RateMapForm::ScreenSizeThenSetLayer,
                })
                .collect(),
            Walk::Layers => space::layer_shapes()
                .into_iter()
                .map(|(at, layers)| Point {
                    at: at.to_owned(),
                    screen: [space::SWEEP_EXTENT, space::SWEEP_EXTENT],
                    layers,
                    form: RateMapForm::ScreenSizeThenSetLayer,
                })
                .collect(),
            // Every form, each over the layer count it takes. The one-layer convenience takes
            // exactly one, so the forms are compared at one layer where they all can be and the
            // two that take more are compared at two as well.
            Walk::Forms => {
                let one = vec![space::uniform_layer(space::SCREEN_SWEEP_QUALITY)];
                let two = vec![
                    RateMapLayer::uniform(1.0, space::SCREEN_SWEEP_QUALITY),
                    RateMapLayer::uniform(0.25, space::SCREEN_SWEEP_QUALITY),
                ];
                let mut out = Vec::new();
                for form in [
                    RateMapForm::ScreenSizeThenSetLayer,
                    RateMapForm::ScreenSizeAndLayer,
                    RateMapForm::ScreenSizeLayerCountLayers,
                    RateMapForm::LayersSubscript,
                    RateMapForm::Holed,
                ] {
                    out.push(Point {
                        at: format!("{}-one-layer", form.at()),
                        screen: [space::SWEEP_EXTENT, space::SWEEP_EXTENT],
                        layers: one.clone(),
                        form,
                    });
                    if form.takes(two.len()) {
                        out.push(Point {
                            at: format!("{}-two-layers", form.at()),
                            screen: [space::SWEEP_EXTENT, space::SWEEP_EXTENT],
                            layers: two.clone(),
                            form,
                        });
                    }
                }
                out
            }
        }
    }
}

impl Point {
    /// Whether this point's map has a layer for the coordinate maps to be asked in.
    ///
    /// The holed form's map has **none** — that is what the form measures — and both coordinate
    /// maps end Metal's process over a map of no layers, so this point's coordinate sites are not
    /// asked for. The same holds for a map of a zero screen extent, which the screen sweep never
    /// states.
    fn maps_coordinates(&self) -> bool {
        let layers = match self.form {
            RateMapForm::Holed => 0,
            _ => self.layers.len(),
        };
        layers > 0 && self.screen[0] > 0 && self.screen[1] > 0
    }

    /// The layer indices this point's derived answers are read at: every layer the compiled map
    /// has, and no index past it.
    ///
    /// Past its own layer table Metal reads whatever lies there, so an index past the count is
    /// never asked — the reason [`crate::cases::rate_maps`] gives for reading layer 0 alone.
    fn layer_indices(&self) -> Vec<u64> {
        let count = match self.form {
            RateMapForm::Holed => 0,
            _ => self.layers.len() as u64,
        };
        (0..count).collect()
    }

    /// Whether the compiled map is the identity inside its screen, so that making it announces
    /// nothing.
    ///
    /// A map whose every quality sample is 1 maps every screen pixel to itself; anything else is
    /// drawn region by region, which the runtime announces as the `rasterization-rate-map`
    /// degradation when the map is made. A point that is not the identity is why this family keeps
    /// a notice allowance.
    fn identity(&self) -> bool {
        self.screen[0] == 0
            || self.screen[1] == 0
            || self.layers.iter().all(|layer| {
                [&layer.horizontal, &layer.vertical]
                    .into_iter()
                    .all(|axis| axis.iter().all(|sample| *sample >= 1.0))
            })
    }

    /// The coordinate grid over this point's map, with each physical probe scaled to the layer's
    /// own derived extent.
    ///
    /// The physical side of [`space::GRID`] is a **fraction**, because the extent is what the
    /// sweep is measuring: an absolute physical coordinate inside one map is outside another, so a
    /// fraction reads the same place in every map of the walk. The fraction is scaled by the
    /// side's **own** answer and never by the other side's, for the reason
    /// [`crate::cases::rate_maps`] gives of the parameter buffer's length: a side that derived a
    /// different extent has already diverged at `physical_sizes`, and feeding it the other side's
    /// number would hide that behind a second difference.
    fn probes(&self, physical: &[[u64; 3]]) -> Vec<CoordinateProbe> {
        let mut out = Vec::new();
        for (layer, extent) in physical.iter().enumerate() {
            for (x, y) in space::GRID.screen {
                out.push(CoordinateProbe {
                    screen: [*x, *y],
                    physical: [0.0, 0.0],
                    layer: layer as u64,
                });
            }
            for (fx, fy) in space::GRID.physical {
                out.push(CoordinateProbe {
                    screen: [0.0, 0.0],
                    physical: [fx * extent[0] as f32, fy * extent[1] as f32],
                    layer: layer as u64,
                });
            }
        }
        out
    }
}

/// A float as a site's value, by its bits, so a signed zero and a one-ulp difference both compare.
fn float(bits: u32) -> Value {
    Value::Float {
        bits: bits.into(),
        width: crate::transcript::FloatWidth::Single,
    }
}

/// One mapped answer as the pair of coordinates it is.
fn mapped(answer: &MappedCoordinates) -> Value {
    Value::fields([
        (
            "screen_to_physical",
            Value::List(vec![
                float(answer.screen_to_physical[0]),
                float(answer.screen_to_physical[1]),
            ]),
        ),
        (
            "physical_to_screen",
            Value::List(vec![
                float(answer.physical_to_screen[0]),
                float(answer.physical_to_screen[1]),
            ]),
        ),
    ])
}

impl Case for RateMapSweep {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        // Every descriptor the sweep states, in order: the screen extent, each layer's samples per
        // axis, and the form. A table that moves invalidates the recording rather than being read
        // against an older one, which matters more here than anywhere else in this family: the
        // whole claim of a sweep is which points of the space it reached.
        let mut words = Vec::new();
        for point in self.walk.points() {
            words.extend(point.at.as_bytes());
            words.push(0);
            words.extend(point.screen[0].to_le_bytes());
            words.extend(point.screen[1].to_le_bytes());
            words.push(point.form as u8);
            words.extend(
                u32::try_from(point.layers.len())
                    .unwrap_or(u32::MAX)
                    .to_le_bytes(),
            );
            for layer in &point.layers {
                for axis in [&layer.horizontal, &layer.vertical] {
                    words.extend(u32::try_from(axis.len()).unwrap_or(u32::MAX).to_le_bytes());
                    for sample in axis {
                        words.extend(sample.to_le_bytes());
                    }
                }
            }
        }
        // The coordinate grid is an input too: a record taken over six screen probes compared
        // against a record taken over eight is a different observation.
        let mut grid = Vec::new();
        for (x, y) in space::GRID.screen {
            grid.extend(x.to_le_bytes());
            grid.extend(y.to_le_bytes());
        }
        for (x, y) in space::GRID.physical {
            grid.extend(x.to_le_bytes());
            grid.extend(y.to_le_bytes());
        }
        Ok(vec![
            Input::bytes("descriptors", &words),
            Input::bytes("coordinate-grid", &grid),
        ])
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Every point of every sweep but one states a quality below 1 somewhere, so making its map
        // announces the rate-map degradation: a pass over such a map is drawn region by region, a
        // near-equivalent the runtime declares (`docs/API.md`'s degrade catalogue). Metal announces
        // nothing, so the notices set is the one field that may differ.
        //
        // The allowance is held to its evidence by the suite itself — the notice must have been
        // announced on the run compared — and it is the only loosening this family declares. Every
        // derived answer compares exactly, which is the point of the sweep.
        // Either way the case declares the notice the venue's own memory model provokes: the
        // buffer it reads its answer back through is `Shared`, which a device with no unified
        // memory keeps in system memory because that is the only memory both sides can reach.
        let shared = crate::replay::notice_allowance(Degradation::SharedInSystemMemory.code());
        if self.walk.points().iter().all(|point| point.identity()) {
            return vec![shared];
        }
        vec![
            shared,
            Allowance::new(
                format!("case/{NOTICES_SITE}/*"),
                Loosening::Unchecked,
                Justification::notice(Degradation::RasterizationRateMap),
            ),
        ]
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        for point in self.walk.points() {
            let mut node = Node::branch(point.at.clone(), Vec::new());
            let request = RateMapRequest {
                screen: point.screen,
                layers: &point.layers,
                label: None,
                form: point.form,
            };
            // A point whose map one side will not compile is that side's answer at this point, and
            // the walk carries on to the next: a sweep that stopped at its first refusal would
            // measure the points before it and say nothing about the rest, which for a sweep is
            // the whole value lost.
            let map = match side.rate_map(&request) {
                Ok(map) => map,
                Err(failure) => {
                    sites.push(node.with(Node::leaf("map", Observed::value(failure.recorded()))));
                    continue;
                }
            };
            node = node.with(Node::leaf(
                "device-is-ours",
                match side.rate_map_device_is_the_sides(map) {
                    Ok(same) => Observed::value(Value::ok(Value::Bool(same))),
                    Err(failure) => Observed::value(failure.recorded()),
                },
            ));
            let indices = point.layer_indices();
            let facts = match side.rate_map_facts(map, &indices) {
                Ok(facts) => facts,
                Err(failure) => {
                    sites.push(node.with(Node::leaf("facts", Observed::value(failure.recorded()))));
                    continue;
                }
            };
            node = node.with(facts.node("facts"));
            if point.maps_coordinates() {
                let probes = point.probes(&facts.physical_sizes);
                let answers = match side.rate_map_coordinates(map, &probes) {
                    Ok(answers) => answers,
                    Err(failure) => {
                        sites.push(node.with(Node::leaf(
                            "coordinates",
                            Observed::value(failure.recorded()),
                        )));
                        continue;
                    }
                };
                let mut mapped_node = Node::branch("coordinates", Vec::new());
                for (index, answer) in answers.iter().enumerate() {
                    let probe = &probes[index];
                    mapped_node = mapped_node.with(Node::leaf(
                        format!("layer-{}-probe-{index}", probe.layer),
                        Observed::value(mapped(answer)),
                    ));
                }
                node = node.with(mapped_node);
            }
            sites.push(node);
        }
        Ok(())
    }
}

/// The parameter data of a handful of swept descriptors, word by word.
///
/// The sweeps above compare what the tile tables **answer**; this compares the tables themselves,
/// all 24,624 bytes of them, for the shapes where the answers alone would leave the most unsaid. It
/// is a separate case because the bytes are the same size for every map (measured: 24,624 at every
/// extent and every layer count probed), so a sweep that carried them at every point would hold
/// megabytes of recording to say what a handful of shapes already say.
pub struct RateMapParameterData {
    name: &'static str,
    screen: [u32; 2],
    layers: Vec<RateMapLayer>,
}

/// The shapes whose parameter data is compared whole.
///
/// `rate-map-no-layers` and `rate-map-zero-screen-width` already compare theirs
/// ([`crate::cases::rate_maps`]), so these are the three shapes neither of those reaches: the
/// identity, a map whose last tile is partial, and a two-layer map, which is the only shape whose
/// tables hold two horizontal axes and so the only one where a side writing layer 1's axis over
/// layer 0's can be caught in the bytes.
pub fn parameter_data() -> Vec<RateMapParameterData> {
    vec![
        RateMapParameterData {
            name: "rate-map-parameters-identity",
            screen: [space::SWEEP_EXTENT, space::SWEEP_EXTENT],
            layers: vec![space::uniform_layer(1.0)],
        },
        // 257 is one screen pixel past eight whole tiles at quality 1, so its last tile is partial
        // and the tables hold the rounding the last-tile rule does.
        RateMapParameterData {
            name: "rate-map-parameters-partial-tile",
            screen: [257, 33],
            layers: vec![space::uniform_layer(space::SCREEN_SWEEP_QUALITY)],
        },
        RateMapParameterData {
            name: "rate-map-parameters-two-layers",
            screen: [space::SWEEP_EXTENT, space::SWEEP_EXTENT],
            layers: vec![
                RateMapLayer::uniform(1.0, space::SCREEN_SWEEP_QUALITY),
                RateMapLayer::uniform(0.25, space::SCREEN_SWEEP_QUALITY),
            ],
        },
    ]
}

impl Case for RateMapParameterData {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut words = Vec::new();
        words.extend(self.screen[0].to_le_bytes());
        words.extend(self.screen[1].to_le_bytes());
        for layer in &self.layers {
            for axis in [&layer.horizontal, &layer.vertical] {
                words.extend(u32::try_from(axis.len()).unwrap_or(u32::MAX).to_le_bytes());
                for sample in axis {
                    words.extend(sample.to_le_bytes());
                }
            }
        }
        Ok(vec![
            Input::bytes("descriptor", &words),
            Input::bytes("parameter-copy-offset", &0u64.to_le_bytes()),
        ])
    }

    fn allowances(&self) -> Vec<Allowance> {
        // The identity announces nothing, on either venue; anything else announces the rate-map
        // degradation when the map is made, for the reason the sweeps' own allowance gives.
        let identity = self.layers.iter().all(|layer| {
            [&layer.horizontal, &layer.vertical]
                .into_iter()
                .all(|axis| axis.iter().all(|sample| *sample >= 1.0))
        });
        // Either way the case declares the notice the venue's own memory model provokes: the
        // buffer it reads its answer back through is `Shared`, which a device with no unified
        // memory keeps in system memory because that is the only memory both sides can reach.
        let shared = crate::replay::notice_allowance(Degradation::SharedInSystemMemory.code());
        if identity {
            return vec![shared];
        }
        vec![
            shared,
            Allowance::new(
                format!("case/{NOTICES_SITE}/*"),
                Loosening::Unchecked,
                Justification::notice(Degradation::RasterizationRateMap),
            ),
        ]
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let request = RateMapRequest::new(self.screen, &self.layers);
        let map = made(sites, "map", side.rate_map(&request))?;
        let indices: Vec<u64> = (0..self.layers.len() as u64).collect();
        let facts = match side.rate_map_facts(map, &indices) {
            Ok(facts) => {
                sites.push(facts.node("facts"));
                facts
            }
            Err(failure) => {
                sites.push(Node::leaf("facts", Observed::value(failure.recorded())));
                return Err(Abort {
                    at: "facts".to_owned(),
                    failure,
                });
            }
        };
        // The buffer is sized from this side's own answer, never from the other's, for the reason
        // `rate_maps` gives: asking a side that answered a different parameter size to write past
        // its own buffer would turn one readable difference into a crash.
        let length = usize::try_from(facts.parameter_size).unwrap_or(usize::MAX);
        let buffer = made(sites, "parameter-buffer", side.zeroed_buffer(length))?;
        made(
            sites,
            "copy-parameter-data",
            side.copy_rate_map_parameters(map, buffer, 0),
        )?;
        let bytes = match side.read_buffer(buffer, 0, length) {
            Ok(bytes) => bytes,
            Err(failure) => {
                sites.push(Node::leaf(
                    "parameter-data",
                    Observed::value(failure.recorded()),
                ));
                return Err(Abort {
                    at: "parameter-data".to_owned(),
                    failure,
                });
            }
        };
        // One word per element, so a diff names the word of the parameter data that moved rather
        // than a byte offset. Apple's layout is a table of 32-bit words.
        sites.leaf(
            "parameter-data",
            Observed::bytes(
                bytes,
                crate::transcript::ByteLayout {
                    region: crate::transcript::Region::Buffer {
                        offset: 0,
                        length: facts.parameter_size,
                    },
                    element: crate::transcript::Element::U32,
                    row_offset: 0,
                    row_elements: u32::try_from(length / 4).unwrap_or(u32::MAX),
                    row_pitch: u32::try_from(length).unwrap_or(u32::MAX),
                    rows: 1,
                    image_pitch: u32::try_from(length).unwrap_or(u32::MAX),
                    images: 1,
                    gaps: crate::transcript::Gaps::Compared,
                },
            ),
        );
        Ok(())
    }
}

/// The falsifications. Each one has to **fail** on the input that distinguishes the check from its
/// absence, which for a sweep means: the derived answer moves when an input moves, the points of a
/// walk are distinct, and no shape reaches a value the device does not grant.
#[cfg(test)]
mod tests {
    use super::*;
    use metal2vulkan::bind::{RasterizationRateMap, RateMapDescriptor, RateMapLayerDescriptor};
    use std::collections::BTreeSet;

    /// The compiled map of a descriptor the sweep states, through the public compile.
    ///
    /// This is used to check the **sweep's own design** — that its points derive different answers
    /// — and never to decide what an answer should be. Metal is the answer, and these tests say
    /// nothing about Metal.
    fn compiled(screen: [u32; 2], layers: &[RateMapLayer]) -> Option<RasterizationRateMap> {
        let mut descriptor = RateMapDescriptor::new(screen);
        descriptor.layers = layers
            .iter()
            .map(|layer| {
                RateMapLayerDescriptor::new(layer.horizontal.clone(), layer.vertical.clone())
            })
            .collect();
        RasterizationRateMap::new(&descriptor).ok()
    }

    /// The physical extent of layer 0 of a compiled descriptor.
    fn extent(screen: [u32; 2], layers: &[RateMapLayer]) -> Option<[u32; 2]> {
        compiled(screen, layers).and_then(|map| map.physical_size(0))
    }

    /// **Everything a case of this family compares**, for layer 0: the physical extent and the
    /// screen-to-physical mapping over the sweep's own grid, by bits.
    ///
    /// The falsifications are written against this rather than against the extent alone, because
    /// the extent alone is a coarser quantization than the tile rule: measured here, the two sides
    /// of the step at 48 screen pixels a tile derive one extent of 176 while laying tiles of 48 and
    /// 47 screen pixels, because the last-tile rule absorbs the difference. The coordinate maps
    /// read the tile lines out directly and do move. A test on the extent alone would have called
    /// that step unstraddled and sent somebody to widen a table that was already right.
    fn derived(screen: [u32; 2], layers: &[RateMapLayer]) -> Option<Vec<u32>> {
        let map = compiled(screen, layers)?;
        let size = map.physical_size(0)?;
        let mut out = vec![size[0], size[1]];
        for (x, y) in space::GRID.screen {
            let [px, py] = map.screen_to_physical([*x, *y], 0);
            out.push(px.to_bits());
            out.push(py.to_bits());
        }
        Some(out)
    }

    /// Every point of the quality sweep's derived answers, each compiled at its **own** extent,
    /// keyed by the quality's name. The extent is part of the point, so reading two points at one
    /// extent would compare maps the sweep never states.
    fn quality_answers() -> std::collections::BTreeMap<String, Vec<u32>> {
        space::qualities()
            .into_iter()
            .filter_map(|quality| {
                derived(
                    [quality.extent, quality.extent],
                    &[space::uniform_layer(quality.quality)],
                )
                .map(|answers| (quality.at.to_owned(), answers))
            })
            .collect()
    }

    /// **The falsification this family exists to survive.** Two maps that differ in exactly one
    /// quality sample derive different answers, so a site reading a derived answer is reading
    /// something the quality decided.
    ///
    /// Without this, every site of the quality sweep could be a constant and every one would pass.
    /// The two qualities are the two sides of the tile rule's step at 64 screen pixels a tile:
    /// the top of the bucket, and the nearest quality over it. **Not** the two sides of a clamp —
    /// the clamps are measured not to move the derived extent at this screen size, which
    /// [`space::same_derived_answer`] states and which is why the first version of this test
    /// passed for the wrong reason.
    #[test]
    fn the_derived_answer_moves_when_a_quality_moves() {
        let edge = space::step_edge(64);
        let over = space::step_over(64).expect("a quality crosses the step at 64");
        let extent = space::visible_extent(space::tile_span(edge));
        let screen = [extent, extent];
        let inside =
            derived(screen, &[space::uniform_layer(edge)]).expect("the top of the bucket compiles");
        let outside =
            derived(screen, &[space::uniform_layer(over)]).expect("the quality over it compiles");
        assert_ne!(
            inside, outside,
            "the two sides of the step at 64 screen pixels a tile derive the same extent \
             {inside:?}, so every site of the quality sweep could be a constant and pass"
        );
    }

    /// And the other direction, which is the honest half: where the space predicts two inputs
    /// derive **one** answer, they do. A pair that disagreed here would be a finding about the
    /// quantization and not a broken test, which is why the message says so.
    #[test]
    fn the_predicted_quantization_collapses_the_pairs_it_names() {
        let by_name = quality_answers();
        for (left, right, why) in space::same_derived_answer() {
            let (a, b) = (by_name.get(left), by_name.get(right));
            if let (Some(a), Some(b)) = (a, b) {
                assert_eq!(
                    a, b,
                    "{left} derives {a:?} and {right} derives {b:?}, where the space predicts one \
                     answer: {why}"
                );
            }
        }
    }

    /// The quality sweep reaches more than one derived answer, and the pairs the space predicts
    /// are one bucket really are one: a step the table straddles derives two answers and a clamp
    /// it names derives one.
    ///
    /// This is where "two inputs, one derived answer" stops being an excuse and becomes a
    /// measurement: the clamp pairs **must** agree and the step pairs **must not**.
    #[test]
    fn the_quantization_holds_where_the_space_predicts_it() {
        let by_name = quality_answers();
        let distinct: BTreeSet<Vec<u32>> = by_name.values().cloned().collect();
        assert!(
            distinct.len() >= 8,
            "the quality sweep derives only {} distinct extents over {} points, so most of it \
             compares one answer repeatedly: {distinct:?}",
            distinct.len(),
            by_name.len()
        );
        for (left, right, why) in space::same_bucket() {
            let (a, b) = (by_name.get(left), by_name.get(right));
            if let (Some(a), Some(b)) = (a, b) {
                assert_eq!(a, b, "{left} and {right} are declared one bucket ({why})");
            }
        }
        for m in space::STEP_TILES {
            // The clamp-pinned ends of the space have no step above them, which the space's own
            // `the_sweep_straddles_every_step_it_names` states and which is why the table has no
            // second half for them.
            if space::step_over(*m).is_none() {
                continue;
            }
            let at = by_name.get(&format!("step-m{m}-at"));
            let over = by_name.get(&format!("step-m{m}-over"));
            if let (Some(at), Some(over)) = (at, over) {
                assert_ne!(
                    at, over,
                    "m{m}'s two sides derive one extent {at:?}, so the pair compares one answer \
                     twice"
                );
            }
        }
    }

    /// **Every collapse the quality sweep exhibits is one the space predicted.** Where two points
    /// derive one answer, [`space::same_bucket`] or [`space::same_derived_answer`] has to say so,
    /// with a reason.
    ///
    /// This is the completeness half of the quantization claim, and it is the test that keeps the
    /// sweep honest as it grows. Measured here: 34 points derive 24 distinct answers, so 10 points
    /// are a second reading of a bucket already read. Each of those is a real statement about how
    /// Metal quantizes — and an unlisted one would be a site that looks like a measurement, adds a
    /// row to the recording and compares nothing new. A pair that collapses for a reason nobody
    /// wrote down fails here, naming the pair, which is the push to either explain it or drop the
    /// point.
    #[test]
    fn every_collapse_of_the_quality_sweep_is_predicted() {
        let answers = quality_answers();
        // The predicted classes, as a union-find over both prediction lists. A name a list uses
        // that the table does not hold is a typo and fails below rather than silently excusing
        // nothing.
        let mut parent: std::collections::BTreeMap<String, String> =
            answers.keys().map(|at| (at.clone(), at.clone())).collect();
        fn root(parent: &std::collections::BTreeMap<String, String>, at: &str) -> String {
            let mut here = at.to_owned();
            while parent[&here] != here {
                here = parent[&here].clone();
            }
            here
        }
        // A point's site name carries the extent it was swept at, so a prediction naming the bare
        // quality is matched against the one point whose name begins with it.
        // Both prediction lists and `quality_answers` name a point by the quality's own name. The
        // extent a point is swept at is part of its SITE name in the transcript, which the
        // recording carries and this test has no need of.
        let predictions: Vec<(&str, &str, &str)> = space::same_bucket()
            .into_iter()
            .chain(space::same_derived_answer())
            .collect();
        for (left, right, why) in &predictions {
            assert!(
                answers.contains_key(*left) && answers.contains_key(*right),
                "a prediction names {left} and {right}, and the quality table holds no such \
                 point ({why})"
            );
            let (a, b) = ((*left).to_owned(), (*right).to_owned());
            let (ra, rb) = (root(&parent, &a), root(&parent, &b));
            if ra != rb {
                parent.insert(ra, rb);
            }
        }
        // The classes the answers actually fall into.
        let mut by_answer: std::collections::BTreeMap<&Vec<u32>, Vec<&String>> =
            std::collections::BTreeMap::new();
        for (at, answer) in &answers {
            by_answer.entry(answer).or_default().push(at);
        }
        let mut unexplained = Vec::new();
        for group in by_answer.values() {
            if group.len() < 2 {
                continue;
            }
            let first = root(&parent, group[0]);
            for other in &group[1..] {
                if root(&parent, other) != first {
                    unexplained.push(format!("{} and {}", group[0], other));
                }
            }
        }
        assert!(
            unexplained.is_empty(),
            "these points of the quality sweep derive one answer and no prediction of              `rate_map_space` says why, so each is a site that compares nothing new: {}",
            unexplained.join("; ")
        );
        // And the sweep is not mostly collapse: most of its points read a bucket of their own.
        let distinct = by_answer.len();
        assert!(
            distinct * 2 > answers.len(),
            "the quality sweep derives {distinct} distinct answers over {} points, so over half              of it is a second reading of a bucket already read",
            answers.len()
        );
    }

    /// The descriptor forms collapse the way the family claims they do: **every form over one
    /// layer compiles one map**, every form that takes two compiles one map of two, and the holed
    /// form compiles a map of no layers whatever was set above the hole.
    ///
    /// Here the collapse IS the conformance claim rather than a quantization, which is why it is
    /// asserted and not merely explained: a form that compiled a different map would be a
    /// difference between two Metal construction paths, and the point of the walk is that there is
    /// none. Our side reaches every form the one way it has, so this test checks the walk's shape;
    /// what Metal does with each path is what the recording measures.
    #[test]
    fn the_descriptor_forms_compile_one_map_each_layer_count() {
        let mut by_count: std::collections::BTreeMap<usize, BTreeSet<Vec<u32>>> =
            std::collections::BTreeMap::new();
        let mut holed = BTreeSet::new();
        for point in Walk::Forms.points() {
            if point.form == RateMapForm::Holed {
                // The holed descriptor compiles a map of no layers, whose coordinate maps end
                // Metal's process, so its derived answer here is its layer count alone.
                holed.insert(point.layer_indices().len());
                continue;
            }
            let answer = derived(point.screen, &point.layers).expect("every form's shape compiles");
            by_count
                .entry(point.layers.len())
                .or_default()
                .insert(answer);
        }
        for (count, answers) in &by_count {
            assert_eq!(
                answers.len(),
                1,
                "the forms over {count} layer(s) compile {} different maps, so the walk is                  comparing shapes rather than construction paths",
                answers.len()
            );
        }
        assert_eq!(
            by_count.keys().copied().collect::<Vec<_>>(),
            vec![1, 2],
            "the forms walk must state one layer and two, the whole of what the device grants"
        );
        assert_eq!(
            holed,
            BTreeSet::from([0]),
            "the holed form must compile a map of no layers whatever was set above the hole"
        );
    }

    /// The screen sweep reaches a derived extent that is **not** monotonic in the screen extent,
    /// which is the property an implementation rounding the last tile the other way gets wrong.
    ///
    /// 32 and 33 are the pair: a screen extent one past a whole tile derives a **smaller** physical
    /// extent than the whole tile does.
    #[test]
    fn the_screen_sweep_reaches_the_non_monotonic_step() {
        let quality = [space::uniform_layer(space::SCREEN_SWEEP_QUALITY)];
        let whole = extent([32, space::SWEEP_EXTENT], &quality).expect("32 compiles");
        let past = extent([33, space::SWEEP_EXTENT], &quality).expect("33 compiles");
        assert!(
            past[0] < whole[0],
            "33 screen pixels derive {past:?} and 32 derive {whole:?}, so the sweep no longer \
             reaches the non-monotonic step it is here for"
        );
        assert!(
            space::SCREENS.contains(&32) && space::SCREENS.contains(&33),
            "the sweep must state both sides of the step it measures"
        );
    }

    /// The sample sweep's counts derive more than one answer, so the count is something the
    /// compile reads rather than a number the case states and nothing uses.
    ///
    /// A linear ramp fails this — every count from 2 to 9 derives one extent — which is why
    /// [`space::peak_profile`] is a peak.
    #[test]
    fn the_sample_sweep_reaches_more_than_one_derived_answer() {
        let extents: Vec<[u32; 2]> = space::sample_counts()
            .into_iter()
            .filter_map(|count| {
                extent(
                    [space::SWEEP_EXTENT, space::SWEEP_EXTENT],
                    &[RateMapLayer {
                        horizontal: space::peak_profile(count),
                        vertical: vec![space::PEAK_HIGH],
                    }],
                )
            })
            .collect();
        let distinct: BTreeSet<[u32; 2]> = extents.iter().copied().collect();
        assert!(
            distinct.len() >= 3,
            "the sample sweep derives {} distinct extents over {} counts, so the profile does not \
             distinguish them: {distinct:?}",
            distinct.len(),
            extents.len()
        );
    }

    /// A two-layer map whose layers disagree vertically derives **one** vertical extent for both,
    /// so the shape says which vertical axis the compile keeps rather than echoing the layers.
    #[test]
    fn the_layer_sweep_says_which_vertical_axis_the_compile_keeps() {
        let shapes = space::layer_shapes();
        let (_, layers) = shapes
            .iter()
            .find(|(at, _)| *at == "two-layers-vertical-differ")
            .expect("the shape that disagrees vertically is stated");
        let map = compiled([space::SWEEP_EXTENT, space::SWEEP_EXTENT], layers)
            .expect("the disagreeing shape compiles");
        let zero = map.physical_size(0).expect("layer 0 of a two-layer map");
        let one = map.physical_size(1).expect("layer 1 of a two-layer map");
        assert_eq!(
            zero[1], one[1],
            "the two layers derive vertical extents {} and {}, where Metal's parameter data holds \
             one vertical axis for every layer",
            zero[1], one[1]
        );
        // And it is the larger of the two the compile keeps, not layer 0's: layer 0 asks for 0.25
        // and layer 1 for 1.0, so a compile taking layer 0's would derive the narrower extent.
        let lower = compiled(
            [space::SWEEP_EXTENT, space::SWEEP_EXTENT],
            &[RateMapLayer::uniform(space::SCREEN_SWEEP_QUALITY, 0.25)],
        )
        .expect("the lower vertical quality compiles");
        assert!(
            zero[1] > lower.physical_size(0).expect("layer 0 of a one-layer map")[1],
            "the disagreeing shape derives layer 0's own vertical extent, so the shape says \
             nothing about which axis the compile keeps"
        );
    }

    /// Every point of every walk has its own site, so a diff names one point of the space.
    #[test]
    fn every_point_of_every_walk_has_its_own_site() {
        for sweep in all() {
            let points = sweep.walk.points();
            let names: BTreeSet<String> = points.iter().map(|point| point.at.clone()).collect();
            assert_eq!(
                names.len(),
                points.len(),
                "{}: a site name is used twice",
                sweep.name
            );
            assert!(!points.is_empty(), "{}: an empty walk", sweep.name);
        }
    }

    /// No walk states a shape outside the bounds the space holds itself to: more layers than the
    /// device grants, an extent past the swept bound, or more samples than Apple's compile was
    /// measured with.
    #[test]
    fn no_point_of_any_walk_leaves_the_measured_bounds() {
        for sweep in all() {
            for point in sweep.walk.points() {
                assert!(
                    point.layers.len() <= space::MAX_LAYERS,
                    "{} point {} asks for {} layers and the device grants {}",
                    sweep.name,
                    point.at,
                    point.layers.len(),
                    space::MAX_LAYERS
                );
                for axis in point.screen {
                    assert!(
                        axis <= space::MAX_SWEPT_EXTENT,
                        "{} point {} asks for an extent of {axis}, past the swept bound {}",
                        sweep.name,
                        point.at,
                        space::MAX_SWEPT_EXTENT
                    );
                }
                let max = space::MAX_SWEPT_SAMPLES[point.layers.len().max(1) - 1];
                for layer in &point.layers {
                    for samples in [&layer.horizontal, &layer.vertical] {
                        assert!(
                            !samples.is_empty(),
                            "{} point {} has an axis of no samples, which Metal answers nil for",
                            sweep.name,
                            point.at
                        );
                        assert!(
                            samples.len() <= max,
                            "{} point {} has {} samples on an axis, past the {max} measured",
                            sweep.name,
                            point.at,
                            samples.len()
                        );
                    }
                }
            }
        }
        for shape in parameter_data() {
            assert!(shape.layers.len() <= space::MAX_LAYERS);
            for axis in shape.screen {
                assert!(axis > 0 && axis <= space::MAX_SWEPT_EXTENT);
            }
        }
    }

    /// No point whose map has no layer is asked for a coordinate, which is one of the three shapes
    /// measured to end Metal's process.
    #[test]
    fn no_point_asks_a_coordinate_of_a_map_that_would_fault() {
        for sweep in all() {
            for point in sweep.walk.points() {
                if point.form == RateMapForm::Holed {
                    assert!(
                        !point.maps_coordinates(),
                        "{} point {} would ask a coordinate of a map of no layers",
                        sweep.name,
                        point.at
                    );
                    assert!(
                        point.layer_indices().is_empty(),
                        "{} point {} would read a physical size past the map's layer table",
                        sweep.name,
                        point.at
                    );
                }
                if point.maps_coordinates() {
                    assert!(
                        point.screen[0] > 0 && point.screen[1] > 0,
                        "{} point {} would ask screen-to-physical of a zero screen extent",
                        sweep.name,
                        point.at
                    );
                }
            }
        }
    }

    /// Every form that says it takes a layer count really does, and the forms walk states each one
    /// at a count it takes.
    #[test]
    fn every_form_is_stated_at_a_layer_count_it_takes() {
        for point in Walk::Forms.points() {
            assert!(
                point.form.takes(point.layers.len()),
                "the forms walk states {:?} at {} layers, which it does not take",
                point.form,
                point.layers.len()
            );
        }
        let forms: BTreeSet<&str> = Walk::Forms
            .points()
            .iter()
            .map(|point| point.form.at())
            .collect();
        assert_eq!(forms.len(), 5, "a descriptor form is not stated: {forms:?}");
    }

    /// The walks fingerprint differently, so no recording of one is read against another's.
    #[test]
    fn the_walks_have_their_own_fingerprints() {
        let words: BTreeSet<Vec<u8>> = all()
            .iter()
            .map(|sweep| {
                sweep
                    .inputs()
                    .expect("a sweep's inputs are its own tables")
                    .into_iter()
                    .flat_map(|input| format!("{:?}", input.digest).into_bytes())
                    .collect()
            })
            .collect();
        assert_eq!(words.len(), all().len(), "two walks fingerprint the same");
    }
}
