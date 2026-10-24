//! Samples of a texture of Apple's two-plane extended-range format.
//!
//! A 16 by 16 texture of `RGB10A8_2P_XR10` (`p2`) or its sRGB twin (`p2s`) is filled the way the
//! probe filled it — the alpha bytes first, by the owned fixture `xr_two_planes`' `float` write of
//! `b / 255`, and then the colour words by `replaceRegion:` — and sampled at 4,096 coordinates
//! through the owned fixture `xr_samples`' kernels: the 256 texel centres, 512 random points inside
//! the texture, 512 within `32 / 8192` of a texel boundary on each axis, and 2,816 points in
//! `[-0.3, 1.3)` on each axis, so past the edges on both sides.
//!
//! Sixteen cases: each format, each of four sampler states (nearest or linear, clamp to edge or
//! repeat), each with a `float` and a `half` result. The two clamp-to-edge linear cases hold a
//! second site as well, the same 4,096 samples through the fixture's **constexpr** sampler of that
//! state, which is a claim of its own: Metal filters through a sampler state the host made as it
//! filters through one the shader declares.
//!
//! **What the port changes.** `runtime_lane::two_plane_samples` holds these sixteen answers as
//! sixteen SHA-256 constants taken on an Apple M4, and compares a digest. On a difference it prints
//! the first sixteen results as hex and nothing more, so the lane cannot say which of the 4,096
//! coordinates moved. The values are recoverable and small — a `float` case is 4,096 `float4`s, 64
//! KiB — so record mode runs every case on native Metal and the recordings hold all 4,096 results
//! of each. The comparison then names the sample, the lane and both values, and the digest stays
//! beside the value as a projection of it.
//!
//! Every result compares exactly and the cases carry **no allowance**: on both venues this round
//! covers, the runtime announces nothing here. A venue that does will read as divergent until
//! whoever adds it names the notice with a run behind it.
//!
//! Nothing here can fault a GPU: two compute passes a case, one texture, three buffers, no loop.
//! The coordinates do reach outside the texture, which is what the address modes are here for and
//! what Metal defines.

use crate::allowance::Allowance;
use crate::case::{
    made, Abort, Bindings, Case, Failure, Handle, Input, MetalSource, SamplerSpec, Side,
    TextureRole, TextureSpec,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::{PixelFormat, SamplerAddressMode, SamplerMinMagFilter};
use metal2vulkan_runtime::Degradation;

/// The fixture that fills the texture.
const FILL_FIXTURE: &str = "xr_two_planes";

/// The kernel that writes `float4`s to the texture, which is how the alpha plane is filled.
const WRITE_FLOAT: &str = "xr2p_write_float";

/// The fixture that samples.
const SAMPLE_FIXTURE: &str = "xr_samples";

/// The texture's side.
const SIDE: u32 = 16;

/// The texture's texels.
const TEXELS: usize = (SIDE * SIDE) as usize;

/// The coordinates each case samples at.
const SAMPLES: usize = 4096;

/// The threads in a threadgroup of the sampling dispatch, as the probe dispatched it.
const GROUP: u32 = 64;

/// The threads in each dimension of the fill dispatch's threadgroup.
const FILL_GROUP: u32 = 32;

/// The bytes of a colour word, which is what `replaceRegion:` moves of a two-plane texel.
const WORD_BYTES: u32 = 4;

/// The byte every result holds before the pass.
const BLANK: u8 = 0xab;

/// The seed the probe drew the texels from.
const TEXEL_SEED: u64 = 4;

/// The seed the probe drew the coordinates from.
const COORDINATE_SEED: u64 = 5;

/// The probe's generator: the xorshift stream the two-plane cases are drawn from.
struct Rng(u64);

impl Rng {
    /// The stream of `seed`, warmed as the probe warms it.
    fn new(seed: u64) -> Self {
        let mut state = seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(0x632B_E59B_D9B4_E019);
        if state == 0 {
            state = 1;
        }
        let mut rng = Rng(state);
        for _ in 0..8 {
            rng.next();
        }
        rng
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(2_685_821_657_736_338_717)
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
}

/// Seed [`TEXEL_SEED`]'s texels: plane 0's colour words, four bytes each, and the alpha bytes.
fn planes() -> (Vec<u8>, Vec<u8>) {
    let mut rng = Rng::new(TEXEL_SEED);
    let (mut words, mut alpha) = (Vec::new(), Vec::new());
    for _ in 0..TEXELS {
        let high = rng.below(4);
        let red = rng.below(1024);
        let green = rng.below(1024);
        let blue = rng.below(1024);
        words.extend(((high << 30 | red << 20 | green << 10 | blue) as u32).to_le_bytes());
        alpha.push(rng.below(256) as u8);
    }
    (words, alpha)
}

/// The `float4`s the fill kernel writes: `(0, 0, 0, b / 255)` a texel, so the write sets the alpha
/// plane and the colour words are the host's.
fn fill_values() -> Vec<u8> {
    let (_, alpha) = planes();
    alpha
        .iter()
        .flat_map(|byte| [0.0f32, 0.0, 0.0, f32::from(*byte) / 255.0])
        .flat_map(f32::to_le_bytes)
        .collect()
}

/// Seed [`COORDINATE_SEED`]'s 4,096 normalized coordinates, as the probe drew them.
fn coordinates() -> Vec<u8> {
    let mut rng = Rng::new(COORDINATE_SEED);
    let side = f64::from(SIDE);
    let unit = |rng: &mut Rng| (rng.next() >> 11) as f64 / 9_007_199_254_740_992.0;
    let mut bytes = Vec::with_capacity(SAMPLES * 8);
    for sample in 0..SAMPLES as u64 {
        let pair = if sample < TEXELS as u64 {
            let center = |axis: u64| (axis as f32 + 0.5) / SIDE as f32;
            [
                center(sample % u64::from(SIDE)),
                center(sample / u64::from(SIDE)),
            ]
        } else if sample < 768 {
            let x = unit(&mut rng) as f32;
            [x, unit(&mut rng) as f32]
        } else if sample < 1280 {
            let near = |rng: &mut Rng| {
                let edge = rng.below(u64::from(SIDE) + 1) as f64 / side;
                (edge + (rng.below(65) as f64 - 32.0) / 8192.0) as f32
            };
            let x = near(&mut rng);
            [x, near(&mut rng)]
        } else {
            let x = (-0.3 + 1.6 * unit(&mut rng)) as f32;
            [x, (-0.3 + 1.6 * unit(&mut rng)) as f32]
        };
        bytes.extend(pair.iter().flat_map(|axis| axis.to_le_bytes()));
    }
    bytes
}

/// What a result's lanes are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lanes {
    Half,
    Float,
}

impl Lanes {
    /// The runtime-sampler kernel that answers this way.
    fn runtime_entry(self) -> &'static str {
        match self {
            Lanes::Half => "xr_sample_runtime_half",
            Lanes::Float => "xr_sample_runtime",
        }
    }

    /// The constexpr-sampler kernel that answers this way.
    fn constexpr_entry(self) -> &'static str {
        match self {
            Lanes::Half => "xr_sample_constexpr_half",
            Lanes::Float => "xr_sample_constexpr",
        }
    }

    fn element(self) -> Element {
        match self {
            Lanes::Half => Element::F16,
            Lanes::Float => Element::F32,
        }
    }

    /// Bytes one result's four lanes take.
    fn stride(self) -> u32 {
        4 * self.element().size()
    }
}

/// The layout the results are read under: one row a sample of four lanes, so a diff names the
/// coordinate and the lane rather than a byte offset.
fn layout(lanes: Lanes) -> ByteLayout {
    let stride = lanes.stride();
    let length = u64::from(stride) * SAMPLES as u64;
    ByteLayout {
        region: Region::Buffer { offset: 0, length },
        element: lanes.element(),
        row_offset: 0,
        row_elements: 4,
        row_pitch: stride,
        rows: SAMPLES as u32,
        image_pitch: stride * SAMPLES as u32,
        images: 1,
        // Each sample's four lanes fill its row, so there is no padding between them.
        gaps: Gaps::Ignored,
    }
}

/// The sixteen case names, in the order [`all`] builds them: each format, each sampler state,
/// each result type. Written out rather than formatted, so every case's name is a `&'static str`
/// the suite can hold and `conformance replay <name>` can select.
const NAMES: [&str; 16] = [
    "xr-two-plane-samples-p2-nc-f",
    "xr-two-plane-samples-p2-nc-h",
    "xr-two-plane-samples-p2-lc-f",
    "xr-two-plane-samples-p2-lc-h",
    "xr-two-plane-samples-p2-nr-f",
    "xr-two-plane-samples-p2-nr-h",
    "xr-two-plane-samples-p2-lr-f",
    "xr-two-plane-samples-p2-lr-h",
    "xr-two-plane-samples-p2s-nc-f",
    "xr-two-plane-samples-p2s-nc-h",
    "xr-two-plane-samples-p2s-lc-f",
    "xr-two-plane-samples-p2s-lc-h",
    "xr-two-plane-samples-p2s-nr-f",
    "xr-two-plane-samples-p2s-nr-h",
    "xr-two-plane-samples-p2s-lr-f",
    "xr-two-plane-samples-p2s-lr-h",
];

/// A case: one format, one sampler state, one result type.
pub struct TwoPlaneSamples {
    name: &'static str,
    format: PixelFormat,
    /// The probe's name for the sampler state: `nc`, `lc`, `nr`, `lr`.
    state: &'static str,
    lanes: Lanes,
    /// Whether the fixture's constexpr sampler is of this state too, so the case can compare both.
    /// The fixture declares a linear clamp-to-edge sampler, which is `lc`.
    constexpr_too: bool,
}

/// Every pairing of format, sampler state and result type, as the lane sweeps them.
pub fn all() -> Vec<TwoPlaneSamples> {
    let mut cases = Vec::new();
    let mut names = NAMES.iter();
    for format in [PixelFormat::RGB10A82PXR10, PixelFormat::RGB10A82PXR10SRGB] {
        for state in ["nc", "lc", "nr", "lr"] {
            for lanes in [Lanes::Float, Lanes::Half] {
                cases.push(TwoPlaneSamples {
                    name: names.next().expect("a name per pairing"),
                    format,
                    state,
                    lanes,
                    constexpr_too: state == "lc",
                });
            }
        }
    }
    cases
}

/// The sampler state the probe's name means: nearest or linear (`n`, `l`), clamp to edge or repeat
/// (`c`, `r`), normalized coordinates either way.
fn sampler_spec(state: &str) -> SamplerSpec {
    let mut spec = SamplerSpec::new();
    if state.starts_with('l') {
        spec.min_filter = SamplerMinMagFilter::Linear;
        spec.mag_filter = SamplerMinMagFilter::Linear;
    }
    if state.ends_with('r') {
        spec.s_address_mode = SamplerAddressMode::Repeat;
        spec.t_address_mode = SamplerAddressMode::Repeat;
    }
    spec
}

/// What one sampling pass runs over: the kernel, the texture it reads and the coordinates it
/// reads. One value rather than three arguments, so the two passes of a case read the same way.
#[derive(Clone, Copy)]
struct Sampling {
    pipeline: Handle,
    texture: Handle,
    coords: Handle,
}

impl TwoPlaneSamples {
    /// Take the 4,096 samples `over`'s kernel gives through `samplers`, and answer the bytes.
    fn samples(
        &self,
        side: &mut dyn Side,
        sites: &mut Sites,
        at: &str,
        over: Sampling,
        samplers: &[Handle],
    ) -> Result<Vec<u8>, Abort> {
        let Sampling {
            pipeline,
            texture,
            coords,
        } = over;
        let length = SAMPLES * self.lanes.stride() as usize;
        let output = made(
            sites,
            &format!("{at}/buffer"),
            side.buffer(&vec![BLANK; length]),
        )?;
        let textures = [texture];
        let buffers = [coords, output];
        let command_buffer = made(
            sites,
            &format!("{at}/dispatch"),
            side.dispatch_threads(
                pipeline,
                &Bindings::new(&textures, &buffers).sampled(samplers),
                &[],
                [SAMPLES as u32, 1, 1],
                [GROUP, 1, 1],
            ),
        )?;
        let status = made(sites, &format!("{at}/status"), side.wait(command_buffer))?;
        if status != Value::unit_variant("Completed") {
            sites.push(Node::leaf(
                format!("{at}/status"),
                Observed::value(status.clone()),
            ));
            return Err(Abort {
                at: format!("{at}/status"),
                failure: Failure::named("CommandBufferStatus", crate::diff::print(&status)),
            });
        }
        let bytes = made(
            sites,
            &format!("{at}/read-back"),
            side.read_buffer(output, 0, length),
        )?;
        if bytes.len() != length {
            return Err(Abort {
                at: format!("{at}/read-back"),
                failure: Failure::named(
                    "ShortReadBack",
                    format!("{} bytes of {length}", bytes.len()),
                ),
            });
        }
        Ok(bytes)
    }

    /// The node one set of results emits: every sample raw, with the digest beside it as a
    /// projection of the value, which is the form the lane's constant is.
    fn results_node(&self, name: &str, bytes: Vec<u8>) -> Node {
        let digest = Digest::sha256(&bytes);
        Node::leaf(
            name,
            Observed::bytes(bytes, layout(self.lanes)).digested(digest),
        )
    }
}

/// The degrade notices one of these cases may announce. A list rather than a device query,
/// because a case never branches on the device: it runs and reads what came back, and a venue that
/// announces a kind not here fails the case until the kind is added.
///
/// Declaring a kind states what our side announced at the notices site and loosens no value:
/// every sampled channel still compares exactly.
const PASS_NOTICES: &[Degradation] = &[
    // The buffer each pass writes its samples into is `Shared`.
    // A device with no unified memory keeps a `Shared` allocation in system memory because that is
    // the only memory both sides can reach, so this notice belongs to the venue's memory model
    // rather than to anything these cases do. A device whose driver is Metal announces nothing
    // for it.
    Degradation::SharedInSystemMemory,
];

impl Case for TwoPlaneSamples {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        PASS_NOTICES
            .iter()
            .map(|kind| crate::replay::notice_allowance(kind.code()))
            .collect()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        let mut entries = vec![
            (FILL_FIXTURE, WRITE_FLOAT),
            (SAMPLE_FIXTURE, self.lanes.runtime_entry()),
        ];
        if self.constexpr_too {
            entries.push((SAMPLE_FIXTURE, self.lanes.constexpr_entry()));
        }
        for (fixture, entry) in entries {
            for extension in [RUNTIME_TEXT, METAL_TEXT] {
                shaders.push(ShaderRef {
                    fixture: format!("{fixture}{extension}"),
                    entry: entry.to_owned(),
                    source_sha256: crate::sha256_hex(
                        Fixture {
                            name: fixture,
                            entry,
                        }
                        .text(extension)?
                        .as_bytes(),
                    ),
                });
            }
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let (words, _) = planes();
        let fill = Fixture {
            name: FILL_FIXTURE,
            entry: WRITE_FLOAT,
        };
        let sample = Fixture {
            name: SAMPLE_FIXTURE,
            entry: self.lanes.runtime_entry(),
        };
        Ok(vec![
            Input {
                name: "air:fill".to_owned(),
                digest: Digest::sha256(fill.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed:fill".to_owned(),
                digest: Digest::sha256(fill.text(METAL_TEXT)?.as_bytes()),
            },
            Input {
                name: "air:sample".to_owned(),
                digest: Digest::sha256(sample.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed:sample".to_owned(),
                digest: Digest::sha256(sample.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("format", &self.format.as_raw().to_le_bytes()),
            Input::bytes("sampler", &sampler_spec(self.state).words()),
            Input::bytes("texture:words", &words),
            Input::bytes("texture:alpha-values", &fill_values()),
            Input::bytes("coordinates", &coordinates()),
            Input::bytes("buffer:blank", &[BLANK]),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let stop = |at: &str, why: String| Abort {
            at: at.to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        };
        let fill_source = Fixture {
            name: FILL_FIXTURE,
            entry: WRITE_FLOAT,
        }
        .loaded()
        .map_err(|why| stop("library/fill", why))?;
        let sample_source = Fixture {
            name: SAMPLE_FIXTURE,
            entry: self.lanes.runtime_entry(),
        }
        .loaded()
        .map_err(|why| stop("library/sample", why))?;

        let fill_library = made(sites, "library/fill", side.library(&fill_source.source()))?;
        let fill_pipeline = made(
            sites,
            "pipeline/fill",
            side.compute_pipeline(fill_library, WRITE_FLOAT),
        )?;
        let sample_library = made(
            sites,
            "library/sample",
            side.library(&sample_source.source()),
        )?;
        let runtime_pipeline = made(
            sites,
            "pipeline/runtime-sampler",
            side.compute_pipeline(sample_library, self.lanes.runtime_entry()),
        )?;

        // The texture, filled as the probe filled it: the alpha plane by the kernel's `float`
        // write, and then the colour words by the host's `replaceRegion:`.
        let texture = made(
            sites,
            "texture",
            side.texture_2d_in_role(
                &TextureSpec::new(
                    self.format,
                    [SIDE, SIDE],
                    SIDE * WORD_BYTES,
                    TextureRole::ReadWrite,
                ),
                &[],
            ),
        )?;
        let values = made(sites, "fill/values", side.buffer(&fill_values()))?;
        let width = made(sites, "fill/width", side.buffer(&SIDE.to_le_bytes()))?;
        let fill_textures = [texture];
        let fill_buffers = [values, width];
        let fill = made(
            sites,
            "fill/dispatch",
            side.dispatch_threads(
                fill_pipeline,
                &Bindings::new(&fill_textures, &fill_buffers),
                &[],
                [SIDE, SIDE, 1],
                [FILL_GROUP, FILL_GROUP, 1],
            ),
        )?;
        made(sites, "fill/status", side.wait(fill))?;
        let (words, _) = planes();
        made(
            sites,
            "fill/replace-region",
            side.replace_region(texture, [SIDE, SIDE], SIDE * WORD_BYTES, &words),
        )?;

        let coords = made(sites, "coordinates", side.buffer(&coordinates()))?;
        let sampler = made(sites, "sampler", side.sampler(&sampler_spec(self.state)))?;

        let mut output = Node::branch("output", Vec::new());
        let samplers = [sampler];
        let bytes = self.samples(
            side,
            sites,
            "runtime-sampler",
            Sampling {
                pipeline: runtime_pipeline,
                texture,
                coords,
            },
            &samplers,
        )?;
        output = output.with(self.results_node("runtime-sampler", bytes));

        if self.constexpr_too {
            let entry = self.lanes.constexpr_entry();
            let pipeline = made(
                sites,
                "pipeline/constexpr-sampler",
                side.compute_pipeline(sample_library, entry),
            )?;
            let bytes = self.samples(
                side,
                sites,
                "constexpr-sampler",
                Sampling {
                    pipeline,
                    texture,
                    coords,
                },
                &[],
            )?;
            output = output.with(self.results_node("constexpr-sampler", bytes));
        }
        sites.push(output);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The generator is the probe's: the FNV-1a words the probe printed of seed 4's planes and
    /// seed 5's coordinates, and their SHA-256, which the lane pins beside them.
    #[test]
    fn the_generators_are_the_probes() {
        let fnv = |bytes: &[u8]| {
            bytes.iter().fold(2_166_136_261u32, |hash, byte| {
                (hash ^ u32::from(*byte)).wrapping_mul(16_777_619)
            })
        };
        let (words, alpha) = planes();
        assert_eq!(fnv(&words), 0xa9c5_8e27);
        assert_eq!(fnv(&alpha), 0xe564_a75d);
        let coords = coordinates();
        assert_eq!(fnv(&coords), 0x1efe_3caa);
        for (bytes, want) in [
            (
                &words,
                "a763539f6746b2fc54e795303a2a02ac9727cc31382c7e3b9b03cd63c1d7b7d3",
            ),
            (
                &alpha,
                "96997d686459de2ca5fa082997cd27fcf23e0839edef9a3397f7f07c7457ff31",
            ),
            (
                &coords,
                "508d5b75cbb92c96744c9eb9651df674316de37fdf460e1e3a9568bfa5b78cb5",
            ),
        ] {
            assert_eq!(crate::sha256_hex(bytes), want);
        }
    }

    /// Sixteen cases, each a different question, and the two the fixture's constexpr sampler
    /// matches are the clamp-to-edge linear ones.
    #[test]
    fn the_cases_sweep_every_pairing_once() {
        let cases = all();
        assert_eq!(cases.len(), 16);
        let names: BTreeSet<&str> = cases.iter().map(|case| case.name).collect();
        assert_eq!(names.len(), 16);
        // Each name says which format, state and result type the case is of, so the names and the
        // pairings cannot drift apart.
        for case in &cases {
            let named = match case.format {
                PixelFormat::RGB10A82PXR10 => "p2",
                _ => "p2s",
            };
            assert_eq!(
                case.name,
                format!(
                    "xr-two-plane-samples-{named}-{}-{}",
                    case.state,
                    match case.lanes {
                        Lanes::Half => 'h',
                        Lanes::Float => 'f',
                    }
                )
            );
        }
        let with_constexpr: Vec<&TwoPlaneSamples> =
            cases.iter().filter(|case| case.constexpr_too).collect();
        assert_eq!(with_constexpr.len(), 4);
        for case in with_constexpr {
            assert_eq!(case.state, "lc");
            let spec = sampler_spec(case.state);
            assert_eq!(spec.min_filter, SamplerMinMagFilter::Linear);
            assert_eq!(spec.s_address_mode, SamplerAddressMode::ClampToEdge);
        }
    }

    /// The layout covers every result and names the sample as its row, so a diff over 4,096
    /// samples says which coordinate moved.
    #[test]
    fn the_layout_names_the_sample() {
        for lanes in [Lanes::Half, Lanes::Float] {
            let layout = layout(lanes);
            assert_eq!(layout.extent(), SAMPLES * lanes.stride() as usize);
            assert_eq!(
                layout.at(0, 4095, 3),
                4095 * lanes.stride() as usize + 3 * lanes.element().size() as usize
            );
        }
    }

    /// The first 256 coordinates are the texel centres, which is what makes a nearest case's
    /// first 256 samples read one texel each.
    #[test]
    fn the_first_coordinates_are_the_texel_centres() {
        let bytes = coordinates();
        assert_eq!(bytes.len(), SAMPLES * 8);
        for texel in 0..TEXELS {
            let at = texel * 8;
            let axis = |offset: usize| {
                f32::from_le_bytes([
                    bytes[at + offset],
                    bytes[at + offset + 1],
                    bytes[at + offset + 2],
                    bytes[at + offset + 3],
                ])
            };
            let center = |index: usize| (index as f32 + 0.5) / SIDE as f32;
            assert_eq!(axis(0), center(texel % SIDE as usize), "texel {texel} x");
            assert_eq!(axis(4), center(texel / SIDE as usize), "texel {texel} y");
        }
    }
}
