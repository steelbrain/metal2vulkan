//! A degenerate screen extent reached by every **other** construction route, each compared whole.
//!
//! Metal's `MTLRasterizationRateMapDescriptor` has four ways to reach one state
//! ([`crate::case::RateMapForm`]), and a program that reaches it one way must get the map it would
//! have got another way. Two halves of this family already say something about that, and neither
//! says this:
//!
//! - [`crate::cases::rate_maps`]'s `rate-map-zero-screen-width` compares a **degenerate** screen
//!   whole — the label, the screen size, the granularity, the layer count,
//!   `physicalSizeForLayer:` of layer 0, `parameterBufferSizeAndAlign` and all 24,624 parameter
//!   bytes — but by **one** route, `rasterizationRateMapDescriptorWithScreenSize:` then
//!   `setLayer:atIndex:`, because it predates the others.
//! - [`crate::cases::rate_map_sweeps`]'s `rate-map-descriptor-forms` compares every route, but at
//!   `space::SWEEP_EXTENT`, where nothing is degenerate.
//!
//! So the crossing was covered by nothing: a route other than `setLayer:atIndex:` reaching a
//! descriptor whose screen has a zero axis. These three cases are that crossing at the shape the
//! covered route was measured at, each compared **whole**, parameter data included:
//!
//! | Case | Route |
//! |---|---|
//! | `rate-map-zero-width-with-screen-size-and-layer` | `rasterizationRateMapDescriptorWithScreenSize:layer:` |
//! | `rate-map-zero-width-with-layer-count-and-layers` | `rasterizationRateMapDescriptorWithScreenSize:layerCount:layers:` |
//! | `rate-map-zero-width-layers-subscript` | `rasterizationRateMapDescriptorWithScreenSize:` then `layers[0] = layer` |
//!
//! **The descriptor is `rate-map-zero-screen-width`'s, to the byte.** The same screen (`0` by
//! `64`), the same single layer (one quality sample of a half on each axis), the same label
//! (`conformance-zero-width`), so the construction route is the only thing that moved between the
//! four. That is what makes the four recordings a side-by-side reading rather than four separate
//! measurements, and it is why the label is shared rather than made unique per case: a differing
//! label would be a second thing that moved, and `facts/label` is one of the compared fields.
//!
//! The route is in each case's **input fingerprint**, which is why these are a module of their own
//! rather than three more rows of [`crate::cases::rate_maps`]: that module's fingerprint states the
//! screen, the layers and the label and not the route, so adding the route to it would have
//! invalidated two committed recordings to say nothing new about either.
//!
//! Our side has one construction path — the runtime's
//! [`RateMapDescriptor`](metal2vulkan::bind::RateMapDescriptor) is a plain struct a consumer fills
//! — so all four cases feed it the same struct and it answers the same map. That is the comparison:
//! every route Metal offers must compile the map our one struct compiles, at a degenerate extent as
//! much as anywhere else.
//!
//! [`crate::cases::rate_map_sweeps`]'s `rate-map-degenerate-form-sweep` is the breadth half of the
//! same crossing: the same routes over all three degenerate screens (`0` by `64`, `64` by `0` and
//! `0` by `0`), comparing every derived answer but not the parameter bytes, which are one fixed
//! size for every map and would hold megabytes of recording to say what these three say.
//!
//! # Why these cases are safe
//!
//! Nothing here reaches the GPU, for the reason [`crate::cases::rate_maps`] gives: a rate map is
//! compiled on the CPU, its getters are CPU reads and `copyParameterDataToBuffer:offset:` is a CPU
//! copy into shared storage. No command buffer is made and no pass is opened.
//!
//! The one call that is **not** safe over such a map is
//! `mapScreenToPhysicalCoordinates:forLayer:`, which is measured to end plain Metal's process with
//! a SIGSEGV over a zero screen extent — on `0` by `64`, `64` by `0` and `0` by `0` alike — and
//! which the runtime refuses for that reason. No case here asks for it, which is the same bound
//! `rate-map-zero-screen-width` keeps.

use crate::allowance::{Allowance, Justification, Loosening};
use crate::case::{
    made, Abort, Case, Input, MetalSource, RateMapForm, RateMapLayer, RateMapRequest, Side,
};
use crate::degradation::Degradation;
use crate::replay::NOTICES_SITE;
use crate::transcript::{ByteLayout, Element, Gaps, Node, Observed, Region, Sites};

/// The screen every case here states: `rate-map-zero-screen-width`'s, whose compiled answer on this
/// family's oracle is `0` by `33` physical pixels.
const SCREEN: [u32; 2] = [0, 64];

/// The quality sample each axis of the one layer carries, also `rate-map-zero-screen-width`'s. A
/// half is inside a step of the tile rule rather than on an edge of one, so nothing here rests on
/// which side of a step a clamp put the sample.
const QUALITY: f32 = 0.5;

/// The label every case states, shared on purpose (the module docs): with one label the four
/// descriptors differ in their construction route alone.
const LABEL: &str = "conformance-zero-width";

/// The layer index the cases read `physicalSizeForLayer:` at.
///
/// Layer 0 alone, the bound [`crate::cases::rate_maps`] states and for its reason: past its own
/// layer table Metal reads whatever lies there, so an index past the map's `layerCount` is never
/// asked.
const LAYERS_READ: &[u64] = &[0];

/// The site each call and answer sits at, the names `rate-map-zero-screen-width` uses, so a reader
/// comparing the four recordings reads one set of paths.
mod site {
    pub const MAP: &str = "map";
    pub const FACTS: &str = "facts";
    pub const BUFFER: &str = "parameter-buffer";
    pub const COPY: &str = "copy-parameter-data";
    pub const PARAMETERS: &str = "parameter-data";
}

/// One construction route over the degenerate screen, compared whole.
pub struct DegenerateFormCase {
    name: &'static str,
    /// How the Metal side builds the descriptor. The runtime has one way, so this moves only
    /// Metal's leg — which is the point.
    form: RateMapForm,
}

/// Every case of the crossing: each one-layer route but the one `rate-map-zero-screen-width`
/// already compares at this shape.
///
/// [`RateMapForm::Holed`] is not here, and that is a bound rather than an omission: it builds a
/// descriptor Metal counts as having **no** layers, and a map of no layers over a zero screen
/// extent is a crossing nobody has run on this chip. Its two halves are each measured safe alone —
/// `rate-map-no-layers` at 256 by 256 and `rate-map-zero-screen-width` at 0 by 64 — and a
/// descriptor of three layers and both coordinate maps over a map of no layers are measured to end
/// plain Metal's process, so an unmeasured crossing of two shapes next to those is not one to reach
/// for in a recording run.
pub fn all() -> Vec<DegenerateFormCase> {
    vec![
        DegenerateFormCase {
            name: "rate-map-zero-width-with-screen-size-and-layer",
            form: RateMapForm::ScreenSizeAndLayer,
        },
        DegenerateFormCase {
            name: "rate-map-zero-width-with-layer-count-and-layers",
            form: RateMapForm::ScreenSizeLayerCountLayers,
        },
        DegenerateFormCase {
            name: "rate-map-zero-width-layers-subscript",
            form: RateMapForm::LayersSubscript,
        },
    ]
}

impl DegenerateFormCase {
    /// The one layer every case states.
    fn layers(&self) -> Vec<RateMapLayer> {
        vec![RateMapLayer::uniform(QUALITY, QUALITY)]
    }
}

impl Case for DegenerateFormCase {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        // The descriptor and the **route** that built it. The route is the whole of what
        // distinguishes these cases from each other and from `rate-map-zero-screen-width`, so a
        // recording taken by one route can never be read against a case that now states another.
        let layers = self.layers();
        let mut words = Vec::new();
        words.extend(SCREEN[0].to_le_bytes());
        words.extend(SCREEN[1].to_le_bytes());
        words.push(self.form as u8);
        words.extend(
            u32::try_from(layers.len())
                .unwrap_or(u32::MAX)
                .to_le_bytes(),
        );
        for layer in &layers {
            for axis in [&layer.horizontal, &layer.vertical] {
                words.extend(u32::try_from(axis.len()).unwrap_or(u32::MAX).to_le_bytes());
                for sample in axis {
                    words.extend(sample.to_le_bytes());
                }
            }
        }
        Ok(vec![
            Input::bytes("descriptor", &words),
            Input::bytes("label", LABEL.as_bytes()),
            Input::bytes("parameter-copy-offset", &0u64.to_le_bytes()),
        ])
    }

    fn allowances(&self) -> Vec<Allowance> {
        // The same declaration `rate-map-zero-screen-width` makes, because the map is its map: the
        // rate-map degradation the runtime announces when it makes a map that is not the identity. `0` by `64` at a half is not: its
        // vertical axis lays 64 screen pixels into 33 physical ones, so a pass over it would be
        // drawn region by region, a near-equivalent the runtime declares (`docs/API.md`'s degrade
        // catalogue). Metal announces nothing, so the notices set is the one field that may differ,
        // and the allowance is held to its evidence — the notice must actually have been announced
        // on the run compared.
        vec![Allowance::new(
            format!("case/{NOTICES_SITE}/*"),
            Loosening::Unchecked,
            Justification::notice(Degradation::RasterizationRateMap),
        )]
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let layers = self.layers();
        let request = RateMapRequest {
            screen: SCREEN,
            layers: &layers,
            label: Some(LABEL),
            form: self.form,
        };
        let map = made(sites, site::MAP, side.rate_map(&request))?;
        let facts = match side.rate_map_facts(map, LAYERS_READ) {
            Ok(facts) => {
                sites.push(facts.node(site::FACTS));
                facts
            }
            Err(failure) => {
                sites.push(Node::leaf(site::FACTS, Observed::value(failure.recorded())));
                return Err(Abort {
                    at: site::FACTS.to_owned(),
                    failure,
                });
            }
        };
        // The buffer is sized from this side's own answer, never from the other's, for the reason
        // `rate_maps` gives: a side that answered a different parameter size has already diverged
        // at `parameter_size`, and asking it to write past its own buffer would turn one readable
        // difference into a crash.
        let length = usize::try_from(facts.parameter_size).unwrap_or(usize::MAX);
        let buffer = made(sites, site::BUFFER, side.zeroed_buffer(length))?;
        made(
            sites,
            site::COPY,
            side.copy_rate_map_parameters(map, buffer, 0),
        )?;
        let bytes = match side.read_buffer(buffer, 0, length) {
            Ok(bytes) => bytes,
            Err(failure) => {
                sites.push(Node::leaf(
                    site::PARAMETERS,
                    Observed::value(failure.recorded()),
                ));
                return Err(Abort {
                    at: site::PARAMETERS.to_owned(),
                    failure,
                });
            }
        };
        // One word per element, so a diff names the word of the parameter data that moved rather
        // than a byte offset. Apple's layout is a table of 32-bit words.
        sites.leaf(
            site::PARAMETERS,
            Observed::bytes(
                bytes,
                ByteLayout {
                    region: Region::Buffer {
                        offset: 0,
                        length: facts.parameter_size,
                    },
                    element: Element::U32,
                    row_offset: 0,
                    row_elements: u32::try_from(length / 4).unwrap_or(u32::MAX),
                    row_pitch: u32::try_from(length).unwrap_or(u32::MAX),
                    rows: 1,
                    image_pitch: u32::try_from(length).unwrap_or(u32::MAX),
                    images: 1,
                    gaps: Gaps::Compared,
                },
            ),
        );
        Ok(())
    }
}

/// The falsifications: the crossing is a crossing, and nothing here reaches the call that ends
/// Metal's process.
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// Every one-layer route Metal offers is either compared here or compared at this shape
    /// already, so the crossing has no silent hole.
    ///
    /// The list it is held against is [`RateMapForm`]'s own one-layer routes, so a route added to
    /// the enum fails here until somebody says where it is compared at a degenerate extent.
    #[test]
    fn every_one_layer_route_is_compared_at_the_degenerate_screen() {
        let every = [
            RateMapForm::ScreenSizeThenSetLayer,
            RateMapForm::ScreenSizeAndLayer,
            RateMapForm::ScreenSizeLayerCountLayers,
            RateMapForm::LayersSubscript,
            RateMapForm::Holed,
        ];
        // A `Vec` and not a set: `RateMapForm` is `Eq` and not `Ord`, and ordering it is a change
        // to another stream's type rather than a need of this test.
        let ours: Vec<RateMapForm> = all().iter().map(|case| case.form).collect();
        for form in every {
            let compared = ours.contains(&form)
                // `setLayer:atIndex:` at this shape is `rate-map-zero-screen-width`.
                || form == RateMapForm::ScreenSizeThenSetLayer
                // The holed route builds a descriptor of no layers, which this module states is
                // an unmeasured crossing and keeps away from.
                || form == RateMapForm::Holed;
            assert!(
                compared,
                "{form:?} reaches a descriptor of one layer and nothing compares it at a \
                 degenerate screen extent"
            );
        }
        assert_eq!(
            ours.len(),
            3,
            "the crossing must compare exactly the three routes nothing else reaches at this \
             shape: {ours:?}"
        );
        let mut distinct = ours.clone();
        distinct.dedup();
        assert_eq!(
            distinct.len(),
            ours.len(),
            "a route is compared twice: {ours:?}"
        );
        assert!(
            !ours.contains(&RateMapForm::ScreenSizeThenSetLayer),
            "the control route is compared at this shape by `rate-map-zero-screen-width`, so a \
             case here would record the same descriptor twice"
        );
    }

    /// Every case states a screen with a zero axis and a route that takes one layer, and no case
    /// states the shapes measured to end plain Metal's process.
    #[test]
    fn every_case_states_a_degenerate_screen_and_one_layer() {
        for case in all() {
            assert!(
                SCREEN[0] == 0 || SCREEN[1] == 0,
                "{}: the screen is not degenerate, so the case is a second reading of \
                 `rate-map-descriptor-forms`",
                case.name
            );
            assert_eq!(case.layers().len(), 1, "{}", case.name);
            assert!(
                case.form.takes(1),
                "{}: {:?} does not take a descriptor of one layer",
                case.name,
                case.form
            );
            for layer in case.layers() {
                for axis in [&layer.horizontal, &layer.vertical] {
                    assert!(
                        !axis.is_empty(),
                        "{}: an axis of no samples, which Metal answers nil for",
                        case.name
                    );
                }
            }
        }
    }

    /// Every case runs on Metal, states its descriptor as an input, and loosens nothing but the
    /// notices site — the shape `tests::the_rate_map_cases_name_what_justifies_them` holds the
    /// family's other half to.
    ///
    /// Every answer the map itself gives therefore compares **exactly**, which is the whole claim
    /// of these cases: a route that compiled a different map at a degenerate extent has nowhere to
    /// hide.
    #[test]
    fn every_case_runs_on_metal_and_loosens_only_its_notices() {
        for case in all() {
            assert_eq!(case.metal_source(), MetalSource::Run, "{}", case.name);
            assert!(case.metal_source().forbidden().is_none(), "{}", case.name);
            let inputs = case.inputs().expect("a case states its inputs");
            assert!(
                inputs.iter().any(|input| input.name == "descriptor"),
                "{}: the descriptor is not an input, so the fingerprint says nothing about what \
                 the case fed: {inputs:?}",
                case.name
            );
            let allowances = case.allowances();
            assert_eq!(
                allowances.len(),
                1,
                "{}: the rate-map degradation is the one notice this shape declares",
                case.name
            );
            for allowance in allowances {
                match &allowance.justified_by {
                    Justification::Degradation(code) => {
                        assert!(
                            *code == Degradation::RasterizationRateMap.code(),
                            "{}: {code}",
                            case.name
                        );
                        assert_eq!(allowance.path, format!("case/{NOTICES_SITE}/*"));
                    }
                    other => panic!("{}: an unexpected tolerance: {other:?}", case.name),
                }
            }
        }
    }

    /// The cases fingerprint differently from each other, so no recording of one route is read
    /// against another's.
    ///
    /// They differ in the route alone, which is exactly the thing this crossing is about: without
    /// the route in the fingerprint all three would state one input and a recording taken by one
    /// route would be read as the others'.
    #[test]
    fn the_routes_have_their_own_fingerprints() {
        let digests: BTreeSet<String> = all()
            .iter()
            .map(|case| {
                case.inputs()
                    .expect("a case's inputs are its descriptor and route")
                    .into_iter()
                    .map(|input| format!("{:?}", input.digest))
                    .collect::<Vec<_>>()
                    .join("/")
            })
            .collect();
        assert_eq!(
            digests.len(),
            all().len(),
            "two routes fingerprint the same, so a recording of one would be read as the other's"
        );
    }

    /// Our side's compiled map is the same whatever route the case names, which is the claim the
    /// recording holds Metal to.
    ///
    /// This says nothing about Metal — the recording is that — and it is here because the runtime
    /// has one construction path: a case that somehow fed it a different struct per route would be
    /// comparing shapes rather than construction paths, and this is what would catch it.
    #[test]
    fn our_side_compiles_one_map_for_every_route() {
        use metal2vulkan::bind::{RateMapDescriptor, RateMapLayerDescriptor};
        let mut answers = BTreeSet::new();
        for case in all() {
            let mut descriptor = RateMapDescriptor::new(SCREEN);
            descriptor.layers = case
                .layers()
                .iter()
                .map(|layer| {
                    RateMapLayerDescriptor::new(layer.horizontal.clone(), layer.vertical.clone())
                })
                .collect();
            let map = metal2vulkan::bind::RasterizationRateMap::new(&descriptor)
                .expect("a zero screen width compiles");
            answers.insert((
                map.physical_size(0),
                map.screen_size(),
                map.is_identity(),
                map.words().to_vec(),
            ));
        }
        assert_eq!(
            answers.len(),
            1,
            "our side compiled {} different maps for the three routes",
            answers.len()
        );
        let (physical, _, identity, _) = answers.into_iter().next().expect("one answer");
        // The shape the allowance rests on: the map is not the identity, so the runtime announces
        // the rate-map degradation and every case here has to declare it.
        assert!(
            !identity,
            "0 by 64 at a half compiles the identity, so the notice these cases declare would \
             never fire and the declaration would be checking nothing"
        );
        assert_eq!(
            physical,
            Some([0, 33]),
            "our side derives another physical extent than the one this family's oracle is \
             measured at"
        );
    }
}
