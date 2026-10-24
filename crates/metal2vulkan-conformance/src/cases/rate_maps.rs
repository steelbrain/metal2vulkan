//! Rasterization-rate maps: the family where Metal's API validation and plain Metal disagree.
//!
//! Both cases here are descriptors Metal's own API validation layer refuses and plain Metal
//! compiles anyway, which is what makes them the family that exercises the recording's second
//! oracle ([`crate::validated`]). Measured on this Mac, an Apple M3 Max on macOS 27.0.1:
//!
//! | Descriptor | Plain Metal | Metal's API validation |
//! |---|---|---|
//! | 256 by 256 with no layer | a map of no layers | `layerCount (0) must be >0 and <=2.` |
//! | 0 by 64, one half-quality layer | a map of one layer | `screenSize.width (0) must be >0 and <=32768.` |
//!
//! Both are behavioural cases: plain Metal makes each map, so the runtime must make it too, and
//! every answer compares — the label, the screen size, the granularity, the layer count,
//! `physicalSizeForLayer:` of layer 0, the parameter buffer's size and alignment, and all 24,624
//! bytes of the parameter data word by word. The validated fact sits beside each and changes
//! nothing, which is the ruling: plain Metal is the oracle because that is how the runtime is
//! used.
//!
//! [`ZERO_SCREEN_WIDTH`] was the suite's one refusal when it was written: the runtime answered
//! `Invalid` with `RateMapProblem::ScreenSize` where plain Metal compiled the map. That was a gap
//! of ours, by the owner's standing rule that what plain Metal allows we must allow, and it is
//! closed: the runtime now compiles a zero screen extent into an axis of one tile of no screen and
//! no physical pixels, Apple's, so the case compares like any other and its ledger row is gone.
//! Over such a map only `mapScreenToPhysicalCoordinates:forLayer:` stays refused, because that one
//! call ends Metal's own process; neither case makes it.
//!
//! # Why this family is safe
//!
//! Nothing here reaches the GPU. A rate map is compiled on the CPU, its getters are CPU reads, and
//! `copyParameterDataToBuffer:offset:` is a CPU copy into shared storage; no command buffer is
//! made and no pass is opened. So neither case can fault the device, hang it or trip its watchdog,
//! and the validation complaint both provoke is a CPU-side check.
//!
//! Three shapes of this family are deliberately **not** cases, because they are not safe: a
//! descriptor of three layers ends plain Metal's process (verified: SIGSEGV with no validation
//! layer at all), the coordinate maps of a map of no layers end it too, and so does
//! `mapScreenToPhysicalCoordinates:forLayer:` over a map of a zero screen extent (verified: a
//! SIGSEGV at the first such call on 0 by 64, 64 by 0 and 0 by 0 alike, where the
//! physical-to-screen map of the same map answers) — which is why the runtime refuses all three.
//! A pass over a mutable map is out of this round by the user's rule.

use crate::allowance::{Allowance, Justification, Loosening};
use crate::case::{made, Abort, Case, Input, MetalSource, RateMapLayer, RateMapRequest, Side};
use crate::replay::NOTICES_SITE;
use crate::transcript::{ByteLayout, Element, Gaps, Node, Observed, Region, Sites};
use metal2vulkan_runtime::Degradation;

/// The case of a descriptor Metal counts as having no layers.
pub const NO_LAYERS: &str = "rate-map-no-layers";

/// The case of a descriptor whose screen width is 0.
pub const ZERO_SCREEN_WIDTH: &str = "rate-map-zero-screen-width";

/// The words Metal's API validation used of a descriptor of no layers, which [`NO_LAYERS`]
/// records as its second fact.
pub const NO_LAYERS_SAYS: &str = "layerCount (0) must be >0 and <=2.";

/// The words Metal's API validation used of a screen width of 0, which justify the runtime's
/// refusal in [`ZERO_SCREEN_WIDTH`].
pub const ZERO_WIDTH_SAYS: &str = "screenSize.width (0) must be >0 and <=32768.";

/// The layer index the cases read `physicalSizeForLayer:` at.
///
/// Layer 0 alone, on purpose. Past its own layer table Metal reads whatever lies there — a map of
/// no layers answers 16 by 16 at layer 2 — and the runtime answers a width of 0 instead, a
/// difference the rate-map module already records. A case must not quietly turn that into a
/// divergence of this family's, so it asks only where Metal's answer is a measured fact.
const LAYERS_READ: &[u64] = &[0];

/// The site each call and answer sits at.
mod site {
    pub const MAP: &str = "map";
    pub const FACTS: &str = "facts";
    pub const BUFFER: &str = "parameter-buffer";
    pub const COPY: &str = "copy-parameter-data";
    pub const PARAMETERS: &str = "parameter-data";
}

/// A case over one rasterization-rate map descriptor.
pub struct RateMapCase {
    name: &'static str,
    /// The descriptor's screen width and height in screen pixels.
    screen: [u32; 2],
    /// The descriptor's layers.
    layers: Vec<RateMapLayer>,
    /// The descriptor's label, which the map keeps.
    label: &'static str,
    /// Whether the compiled map is the identity inside its screen. A map that is not announces
    /// the `rasterization-rate-map` degradation when it is made, because a pass over it would be
    /// drawn region by region, so such a case keeps one notice allowance; Metal announces nothing.
    identity: bool,
}

/// Every case of the family.
pub fn all() -> Vec<RateMapCase> {
    vec![
        // A descriptor holding no layer: Metal's `layerCount` is 0, its validation refuses the
        // descriptor, and plain Metal compiles a map of no layers. The screen size is the one the
        // runtime lane's own table of this chip was taken at.
        RateMapCase {
            name: NO_LAYERS,
            screen: [256, 256],
            layers: Vec::new(),
            label: "conformance-no-layers",
            identity: true,
        },
        // A descriptor of no screen width: its validation refuses it, and plain Metal compiles a
        // map of one layer, which the runtime compiles too. Its vertical axis is not the
        // identity, so making it announces the rate-map degradation.
        RateMapCase {
            name: ZERO_SCREEN_WIDTH,
            screen: [0, 64],
            layers: vec![RateMapLayer::uniform(0.5, 0.5)],
            label: "conformance-zero-width",
            identity: false,
        },
    ]
}

impl RateMapCase {
    /// The descriptor both sides take.
    fn request(&self) -> RateMapRequest<'_> {
        RateMapRequest {
            screen: self.screen,
            layers: &self.layers,
            label: Some(self.label),
            // The form every case of this family took before the others existed
            // (`rasterizationRateMapDescriptorWithScreenSize:` then `setLayer:atIndex:`). The
            // other four are swept by `rate_map_sweeps`, which compares the map each compiles.
            form: crate::case::RateMapForm::ScreenSizeThenSetLayer,
        }
    }
}

impl Case for RateMapCase {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        // The descriptor is the whole of what the case feeds: its screen size, every layer's
        // quality samples along each axis in order, and the label the map keeps. A layer's empty
        // axis is part of the fingerprint, since it is what the case is about.
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
            Input::bytes("label", self.label.as_bytes()),
            Input::bytes("parameter-copy-offset", &0u64.to_le_bytes()),
        ])
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Exact is the default and every answer of these cases keeps it. A map of the identity
        // announces nothing, on either venue (verified on MoltenVK and on lavapipe), so its case
        // carries no allowance at all: a tolerance that never fires is one nobody is checking, and
        // a device that does announce something there should fail the case until somebody names
        // why.
        // Either way the case declares the notice its `Shared` read-back buffer provokes
        // ([`crate::replay::shared_memory_notice`]), which the RTX 5080 announced for both rows.
        let shared = crate::replay::shared_memory_notice();
        if self.identity {
            return vec![shared];
        }
        // A map that is not the identity announces the rate-map degradation when it is made, a
        // near-equivalent the runtime declares (`docs/API.md`'s degrade catalogue): a pass over
        // such a map is drawn region by region, on this crate's tile lines. Metal announces
        // nothing, so the notices set is the one field that may differ, and the allowance is held
        // to its evidence — the notice must actually have been announced on the run compared.
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
        let map = made(sites, site::MAP, side.rate_map(&self.request()))?;
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
        // The buffer is sized from this side's own answer, never from the other's: a side that
        // answers a different parameter size has already diverged at `parameter_size`, and asking
        // it to write past its own buffer would turn one readable difference into a crash.
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
