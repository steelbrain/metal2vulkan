//! `MTLDevice`'s pure queries, asked directly: the ones whose answer the runtime computes from a
//! **model of Metal's rule** rather than reading the device.
//!
//! [`crate::device_queries`] states the test a query has to pass to be here, names each of the six
//! that do, and names every `MTLDevice` query that does not and why. Read it before adding a row:
//! this family is not "the device's answers", it is the answers that are a contract with Metal and
//! with nothing else, which is what a recording can hold.
//!
//! # The cases, and what each one would catch
//!
//! - **`device-heap-buffer-sizes`**: `heapBufferSizeAndAlignWithLength:options:` over lengths
//!   either side of 256 and under five `MTLResourceOptions`. The runtime answers `length` bytes at
//!   256 for every length and every options word — a constant its own source calls *measured* with
//!   no run behind it — so a sweep is what says the size is not rounded, the alignment is not the
//!   length's, and the options change nothing;
//! - **`device-vertex-amplification-counts`**: `supportsVertexAmplificationCount:` from 0 to 16.
//!   The runtime's constant 8 comes from a probe its documentation names and nobody kept; this is
//!   the run that keeps it, and it is the case that would catch the boundary moving either way;
//! - **`device-rate-map-layer-counts`**: `supportsRasterizationRateMapWithLayerCount:` from 1 to 4,
//!   where the runtime answers yes for 1 and 2 and no above;
//! - **`device-linear-texture-alignments`**: `minimumLinearTextureAlignmentForPixelFormat:` over
//!   sixteen colour formats from one byte a texel to sixteen. The runtime answers each from the
//!   device's linear images, 16 for every one of them on the oracle's MoltenVK, so a format whose
//!   alignment Metal makes the texel's, or a probe that answered more there, would read wrong here;
//! - **`device-sparse-region-conversions`**: both of Metal's region conversions over the same
//!   tables, each direction at three tile sizes, and the pixels-to-tiles direction in **both**
//!   alignment modes. The rows are chosen so inward and outward differ: a region that starts and
//!   ends inside a tile covers one tile outward and none inward.
//!
//! The last of those found something worth writing down. **Metal's inward conversion of a region
//! that lies strictly inside one tile underflows**: on this oracle it answers an extent of
//! 288230376151711743 (`0x3ff_ffff_ffff_ffff`) on both the width and the height rather than an
//! empty region, which is a `-1` tile count carried through Metal's own shift. The runtime's model
//! reproduces it exactly, and this case is now what holds it there: a change that "fixed" our
//! answer to 0 would be a divergence from Metal, and before this case nothing in the tree would
//! have said so.
//!
//! # Exactness
//!
//! Every answer is a number or a bool. The family declares no tolerance and announces nothing: no
//! case touches a texture, a sampler, a float or the GPU.
//!
//! # Why every case is safe
//!
//! No query compiles, allocates or reaches a queue. [`crate::device_queries`] names the bound each
//! sweep is kept inside and what already measured it; the pixel-format sweep in particular follows
//! [`crate::cases::heap_metal_sizes`]' order of work, because Metal's alignment query **asserts**
//! on a compressed, depth or stencil format and on a format the host's Metal describes none of.

use crate::case::{Abort, Case, Input, MetalSource, Side};
use crate::device_queries::{
    box_value, boxes_value, size_and_align, Box6, HeapBufferQuery, RegionQuery,
};
use crate::transcript::{Node, Observed, Sites, Value};
use metal2vulkan::metal::PixelFormat;

/// The case names, as the report and the recordings spell them.
pub const HEAP_BUFFERS: &str = "device-heap-buffer-sizes";
pub const AMPLIFICATION: &str = "device-vertex-amplification-counts";
pub const RATE_MAP_LAYERS: &str = "device-rate-map-layer-counts";
pub const LINEAR_ALIGNMENTS: &str = "device-linear-texture-alignments";
pub const REGIONS: &str = "device-sparse-region-conversions";

/// The lengths the heap-buffer sweep asks about: 1, then either side of the alignment it claims,
/// then two whole multiples and one far above.
///
/// A length of **0** is not here, for the reason [`crate::device_queries`] gives: nothing in the
/// tree says Metal's suballocator takes it.
const LENGTHS: [u64; 8] = [1, 15, 255, 256, 257, 511, 4096, 1 << 20];

/// The `MTLResourceOptions` words the heap-buffer sweep asks under, as Metal's own bits: the
/// default (shared, default cache, default tracking), private storage, managed storage,
/// write-combined caching, and untracked hazards.
///
/// Five words whose storage, cache and hazard fields are each moved by one of them, because the
/// claim under test is that **none** of the three changes the answer.
const OPTIONS: [(u64, &str); 5] = [
    (0x0, "default"),
    (0x2 << 4, "private"),
    (0x1 << 4, "managed"),
    (0x1, "write-combined"),
    (0x1 << 8, "untracked"),
];

/// The amplification counts the sweep asks about: 0 through 16, which are the counts the runtime's
/// own documentation says were probed on this chip and on an M4.
const AMPLIFICATION_COUNTS: std::ops::RangeInclusive<u64> = 0..=16;

/// The rasterization-rate map layer counts the sweep asks about.
///
/// From **1**, not 0: the runtime answers no for 0 but nothing in the tree says Metal's query takes
/// it, and 4 is two above the count the runtime models as the last supported one.
const LAYER_COUNTS: std::ops::RangeInclusive<u64> = 1..=4;

/// The pixel formats the alignment sweep asks about: colour formats a buffer can back, from one
/// byte a texel to sixteen, with a packed format, two sRGB forms and a shared-exponent one among
/// them.
///
/// Every one of them is a format this chip's Metal describes and none is compressed, depth or
/// stencil — which is the same set the runtime answers a number rather than `Invalid` for, and
/// the runtime's leg is run first to say so ([`crate::device_queries`]).
const FORMATS: [PixelFormat; 16] = [
    PixelFormat::A8Unorm,
    PixelFormat::R8Unorm,
    PixelFormat::R8Snorm,
    PixelFormat::R8Uint,
    PixelFormat::R16Float,
    PixelFormat::RG8Unorm,
    PixelFormat::R32Uint,
    PixelFormat::R32Float,
    PixelFormat::RGBA8Unorm,
    PixelFormat::RGBA8UnormSRGB,
    PixelFormat::BGRA8Unorm,
    PixelFormat::BGRA8UnormSRGB,
    PixelFormat::RGB10A2Unorm,
    PixelFormat::RGB9E5Float,
    PixelFormat::RG32Float,
    PixelFormat::RGBA32Float,
];

/// The regions every conversion row converts.
///
/// Chosen so the two alignment modes answer differently: the first starts and ends **inside** one
/// 64-texel tile, so outward covers one tile and inward covers none; the second starts inside a
/// tile and ends inside the next; the third is tile-aligned at both ends, where the modes agree;
/// the fourth is empty, and the fifth is a 3D box.
const BOXES: [Box6; 5] = [
    [8, 8, 0, 16, 16, 1],
    [32, 0, 0, 64, 64, 1],
    [64, 128, 0, 128, 64, 1],
    [16, 16, 0, 0, 0, 1],
    [1, 2, 3, 70, 130, 4],
];

/// The conversions the region case asks for.
///
/// Three tile sizes: Metal's own 64 by 64 for a four-byte format, a 32 by 32 one, and a tile with a
/// **zero** axis, which the runtime's documentation says was measured on an M3 Max and an M4 under
/// both alignment modes as leaving the caller's regions as they were.
const CONVERSIONS: [RegionQuery; 4] = [
    RegionQuery::new(&BOXES, [64, 64, 1]),
    RegionQuery::new(&BOXES, [64, 64, 1]).inward(),
    RegionQuery::new(&BOXES, [32, 32, 1]),
    RegionQuery::new(&BOXES, [0, 64, 1]),
];

/// Which query one case sweeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Query {
    HeapBuffers,
    Amplification,
    RateMapLayers,
    LinearAlignments,
    Regions,
}

/// A case over one of `MTLDevice`'s pure queries.
pub struct DeviceQueryCase {
    name: &'static str,
    query: Query,
}

/// Every case of the family.
pub fn all() -> Vec<DeviceQueryCase> {
    vec![
        DeviceQueryCase {
            name: HEAP_BUFFERS,
            query: Query::HeapBuffers,
        },
        DeviceQueryCase {
            name: AMPLIFICATION,
            query: Query::Amplification,
        },
        DeviceQueryCase {
            name: RATE_MAP_LAYERS,
            query: Query::RateMapLayers,
        },
        DeviceQueryCase {
            name: LINEAR_ALIGNMENTS,
            query: Query::LinearAlignments,
        },
        DeviceQueryCase {
            name: REGIONS,
            query: Query::Regions,
        },
    ]
}

/// Every heap-buffer query the sweep asks, in order.
fn heap_buffer_queries() -> Vec<(HeapBufferQuery, &'static str)> {
    let mut out = Vec::new();
    for length in LENGTHS {
        for (options, word) in OPTIONS {
            out.push((HeapBufferQuery { length, options }, word));
        }
    }
    out
}

impl Case for DeviceQueryCase {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn shaders(&self) -> Result<Vec<crate::recording::ShaderRef>, String> {
        // No case here compiles anything: the queries are arithmetic over the case's own numbers.
        Ok(Vec::new())
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut words = Vec::new();
        match self.query {
            Query::HeapBuffers => {
                for (query, _) in heap_buffer_queries() {
                    words.extend(query.words());
                }
            }
            Query::Amplification => {
                for count in AMPLIFICATION_COUNTS {
                    words.extend(count.to_le_bytes());
                }
            }
            Query::RateMapLayers => {
                for layers in LAYER_COUNTS {
                    words.extend(layers.to_le_bytes());
                }
            }
            Query::LinearAlignments => {
                for format in FORMATS {
                    words.extend(format.as_raw().to_le_bytes());
                }
            }
            Query::Regions => {
                for query in CONVERSIONS {
                    words.extend(query.words());
                }
            }
        }
        Ok(vec![Input::bytes("queries", &words)])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        match self.query {
            Query::HeapBuffers => {
                for (query, word) in heap_buffer_queries() {
                    let at = format!("{}-{word}", query.site());
                    crate::case::probe(
                        sites,
                        &at,
                        side.heap_buffer_size_and_align(&query),
                        size_and_align,
                    );
                }
            }
            Query::Amplification => {
                for count in AMPLIFICATION_COUNTS {
                    let at = format!("count-{count}");
                    crate::case::probe(
                        sites,
                        &at,
                        side.supports_vertex_amplification_count(count),
                        |held| Value::Bool(*held),
                    );
                }
            }
            Query::RateMapLayers => {
                for layers in LAYER_COUNTS {
                    let at = format!("layers-{layers}");
                    crate::case::probe(
                        sites,
                        &at,
                        side.supports_rasterization_rate_map_layers(layers),
                        |held| Value::Bool(*held),
                    );
                }
            }
            Query::LinearAlignments => {
                for format in FORMATS {
                    let at = crate::device_queries::format_site(format);
                    crate::case::probe(
                        sites,
                        &at,
                        side.minimum_linear_texture_alignment(format),
                        |held| Value::Uint(*held),
                    );
                }
            }
            Query::Regions => {
                for query in CONVERSIONS {
                    let at = query.site("pixels-to-tiles");
                    crate::case::probe(
                        sites,
                        &at,
                        side.convert_sparse_pixel_regions(&query),
                        |held: &Vec<Box6>| boxes_value(held),
                    );
                    // The other direction takes no alignment mode, so it is asked once per tile
                    // size rather than once per mode: a second site under the inward row would be
                    // the same call under a different name.
                    if !query.inward {
                        let at = query.site("tiles-to-pixels");
                        crate::case::probe(
                            sites,
                            &at,
                            side.convert_sparse_tile_regions(&query),
                            |held: &Vec<Box6>| boxes_value(held),
                        );
                    }
                }
                // What went in, recorded beside what came out: a reader of the recording should not
                // have to find this module to know which boxes the answers are of.
                sites.push(Node::leaf(
                    "regions-in",
                    Observed::value(Value::List(BOXES.iter().map(box_value).collect())),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// No two rows of a sweep share a site name, which is what keeps one row's answer from being
    /// compared against another's.
    #[test]
    fn every_row_of_every_sweep_names_its_own_site() {
        let heap: BTreeSet<String> = heap_buffer_queries()
            .iter()
            .map(|(query, word)| format!("{}-{word}", query.site()))
            .collect();
        assert_eq!(heap.len(), heap_buffer_queries().len());
        let formats: BTreeSet<String> = FORMATS
            .iter()
            .map(|format| crate::device_queries::format_site(*format))
            .collect();
        assert_eq!(formats.len(), FORMATS.len(), "two formats share a name");
        let mut regions: Vec<String> = Vec::new();
        for query in CONVERSIONS {
            regions.push(query.site("pixels-to-tiles"));
            if !query.inward {
                regions.push(query.site("tiles-to-pixels"));
            }
        }
        let unique: BTreeSet<&String> = regions.iter().collect();
        assert_eq!(unique.len(), regions.len(), "{regions:?}");
    }

    /// The five options words move the storage, cache and hazard fields apart from one another, so
    /// the claim "none of the three changes the answer" is really asked of each.
    #[test]
    fn the_options_words_move_each_field_of_the_resource_options() {
        let bits: Vec<u64> = OPTIONS.iter().map(|(bits, _)| *bits).collect();
        assert!(bits.iter().any(|held| held & 0xf != 0), "no cache mode");
        assert!(
            bits.iter().any(|held| held & (0xf << 4) != 0),
            "no storage mode"
        );
        assert!(
            bits.iter().any(|held| held & (0x3 << 8) != 0),
            "no hazard mode"
        );
        let unique: BTreeSet<u64> = bits.iter().copied().collect();
        assert_eq!(unique.len(), bits.len(), "two options words are equal");
    }

    /// The region table holds a box strictly inside one tile of the 64-texel conversions, which is
    /// what makes the two alignment modes answer differently. Without such a row the inward case
    /// would record the outward case's answer and nobody would notice.
    #[test]
    fn a_region_lies_strictly_inside_one_tile() {
        let inside = BOXES.iter().any(|held| {
            held[3] > 0 && held[4] > 0 && held[0] % 64 != 0 && (held[0] + held[3]) % 64 != 0
        });
        assert!(
            inside,
            "no region starts and ends off a 64-texel tile boundary, so inward and outward \
             alignment cannot differ: {BOXES:?}"
        );
    }

    /// Each case fingerprints its own sweep, so no recording is read against another case's.
    #[test]
    fn the_five_cases_have_five_fingerprints() {
        let words: BTreeSet<Vec<u8>> = all()
            .iter()
            .map(|case| {
                case.inputs()
                    .expect("the inputs are the case's own numbers")
                    .into_iter()
                    .flat_map(|input| input.digest.hex.into_bytes())
                    .collect()
            })
            .collect();
        assert_eq!(words.len(), all().len());
    }
}
