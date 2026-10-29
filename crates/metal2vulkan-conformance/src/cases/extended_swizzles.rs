//! A swizzled **extended-range, bfloat or packed-normalized** texture read, sampled and gathered at
//! a sampled slot, ported from `runtime_lane::extended_swizzles`.
//!
//! These are the formats whose image holds raw codes the function decodes, so a swizzle in the
//! view's component mapping hands a channel's codes to another channel's lane. Metal applies the
//! texture's swizzle to the texel it reads — each channel the source its swizzle names, `Zero` 0 and
//! `One` 1.0, border texels included — and a gather of a component gathers the swizzle's source of
//! it. The lane asked whether our swizzled reads, samples and gathers of such a texture are its
//! unswizzled ones swizzled, as Metal's are, and compared them against the **unswizzled texture on
//! the same device**, because for the extended-range formats it held no Metal rule at all.
//!
//! **The port records Metal's own words for every swizzle**, which is strictly more than the lane
//! had: comparison is then exact against Metal rather than against our own unswizzled leg. The
//! lane's two claims survive as committed cross-checks over the recordings —
//! `every_recorded_swizzle_is_the_unswizzled_texture_swizzled`, which is the lane's whole subject,
//! and `every_recorded_packed_read_is_metals_rule`, the code-to-value rule the lane did hold for the
//! four packed formats (each channel's code `c` of `b` bits reads `c / (2^b - 1)`, correctly
//! rounded, and an alpha the format lacks reads 1).
//!
//! The kernels are the owned fixture `fpv_swizzled_reads`'s, all binding the texture at
//! `[[texture(0)]]`, writing `o[i]` (buffer 0) from position `p[i]` (buffer 1): `fpv_swz_read`
//! reads a texel coordinate, `fpv_swz_sample` samples a normalized coordinate through the runtime
//! sampler at `[[sampler(0)]]`, and `fpv_swz_gather_x` through `_w` gather one component of the
//! footprint.
//!
//! **The positions are a stated reduction of the lane's**, and this is the one place the port is
//! narrower than the lane. The lane sweeps three offsets on each axis independently, nine points in
//! each of the 60 cells of the texture and the ring of cells around it: 540 positions, which at
//! six sampled operations and five swizzles would put each format's recording well past the
//! recording ceiling. The case keeps [`OFFSETS`] on the cell's **diagonal**, three points a cell
//! and 180 in all: a point below the half-texel line on both axes, the texel centre where a linear
//! sample reduces to the texel itself, and a point above it on both axes. Between them every tap of
//! a 2 by 2 linear footprint is reached, in the cells inside the texture and in the ring outside it
//! where the footprint holds only border texels. The six points the lane also takes are the mixed
//! ones (below on one axis, above on the other), and they are not here.
//!
//! One case per format, so a venue that holds no texture of one answers its typed refusal at that
//! row rather than leaving the other fourteen sweeps uncompared — which is what the lane does when
//! it prints `EXT-SWIZZLE-REFUSED` and moves on.
//!
//! The output buffer holds a fill byte before each dispatch (`BLANK`), so a lane the kernel did
//! not write is a recorded value rather than whatever was there.
//!
//! Nothing here can fault a GPU: one compute pass an operation, read-only textures, at most 180
//! threads, no loop. A read past the texture's extent is the texel-containment form MSL defines.

use crate::case::{
    call, made, Abort, Bindings, Case, Failure, Handle, Input, MetalSource, SamplerSpec, Side,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::texture::{TextureRequest, Upload};
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::{
    PixelFormat, SamplerAddressMode, SamplerBorderColor, SamplerMinMagFilter, TextureSwizzle,
    TextureSwizzleChannels,
};

/// The fixture every kernel lives in.
const FIXTURE: &str = "fpv_swizzled_reads";

/// The texture's width and height, as the probe made it.
pub const W: u32 = 8;
pub const H: u32 = 4;

/// The byte the output buffer holds before a dispatch.
const BLANK: u8 = 0x5a;

/// The offsets inside a cell the case samples and gathers at, on the cell's diagonal: below the
/// half-texel line on both axes, the texel centre, and above it on both axes.
pub const OFFSETS: [f64; 3] = [0.25, 0.5, 0.875];

/// A format the case sweeps.
pub struct Format {
    pub name: &'static str,
    pub format: PixelFormat,
    /// Bytes one texel takes, which the case states because no `MTLTexture` answers it.
    pub texel_bytes: u32,
    /// Each channel's `(shift, bits)` in the texel's 16-bit word, red first, `None` where the
    /// format lacks the channel: the packed-normalized formats, whose code-to-value rule the lane
    /// held. `None` for a format whose codes the lane had no rule for.
    pub packed: Option<[Option<(u32, u32)>; 4]>,
}

/// The one-plane extended-range formats, the padded 10-bit ones, their `_sRGB` twins, the bfloat
/// formats, and the four packed normalized ones whose rule the lane held.
///
/// The packed fields are the Vulkan formats' that hold them (`R5G6B5_UNORM_PACK16`,
/// `R5G5B5A1_UNORM_PACK16`, `A1R5G5B5_UNORM_PACK16`, `R4G4B4A4_UNORM_PACK16`).
pub const FORMATS: [Format; 15] = [
    Format {
        name: "extended-swizzles-bgra10xr",
        format: PixelFormat::BGRA10XR,
        texel_bytes: 8,
        packed: None,
    },
    Format {
        name: "extended-swizzles-bgra10xr-srgb",
        format: PixelFormat::BGRA10XRSRGB,
        texel_bytes: 8,
        packed: None,
    },
    Format {
        name: "extended-swizzles-bgr10xr",
        format: PixelFormat::BGR10XR,
        texel_bytes: 4,
        packed: None,
    },
    Format {
        name: "extended-swizzles-bgr10xr-srgb",
        format: PixelFormat::BGR10XRSRGB,
        texel_bytes: 4,
        packed: None,
    },
    Format {
        name: "extended-swizzles-r10unormx6",
        format: PixelFormat::R10UnormX6,
        texel_bytes: 2,
        packed: None,
    },
    Format {
        name: "extended-swizzles-r10unormx6-srgb",
        format: PixelFormat::R10UnormX6SRGB,
        texel_bytes: 2,
        packed: None,
    },
    Format {
        name: "extended-swizzles-rg10unormx12",
        format: PixelFormat::RG10UnormX12,
        texel_bytes: 4,
        packed: None,
    },
    Format {
        name: "extended-swizzles-rg10unormx12-srgb",
        format: PixelFormat::RG10UnormX12SRGB,
        texel_bytes: 4,
        packed: None,
    },
    Format {
        name: "extended-swizzles-r16bfloat",
        format: PixelFormat::R16Bfloat,
        texel_bytes: 2,
        packed: None,
    },
    Format {
        name: "extended-swizzles-rg16bfloat",
        format: PixelFormat::RG16Bfloat,
        texel_bytes: 4,
        packed: None,
    },
    Format {
        name: "extended-swizzles-rgba16bfloat",
        format: PixelFormat::RGBA16Bfloat,
        texel_bytes: 8,
        packed: None,
    },
    Format {
        name: "extended-swizzles-b5g6r5unorm",
        format: PixelFormat::B5G6R5Unorm,
        texel_bytes: 2,
        packed: Some([Some((11, 5)), Some((5, 6)), Some((0, 5)), None]),
    },
    Format {
        name: "extended-swizzles-a1bgr5unorm",
        format: PixelFormat::A1BGR5Unorm,
        texel_bytes: 2,
        packed: Some([Some((11, 5)), Some((6, 5)), Some((1, 5)), Some((0, 1))]),
    },
    Format {
        name: "extended-swizzles-bgr5a1unorm",
        format: PixelFormat::BGR5A1Unorm,
        texel_bytes: 2,
        packed: Some([Some((10, 5)), Some((5, 5)), Some((0, 5)), Some((15, 1))]),
    },
    Format {
        name: "extended-swizzles-abgr4unorm",
        format: PixelFormat::ABGR4Unorm,
        texel_bytes: 2,
        packed: Some([Some((12, 4)), Some((8, 4)), Some((4, 4)), Some((0, 4))]),
    },
];

/// The swizzles, the default first, as the lane made them.
pub fn swizzles() -> [(&'static str, TextureSwizzleChannels); 5] {
    use TextureSwizzle::{Alpha, Blue, Green, One, Red, Zero};
    let s = |red, green, blue, alpha| TextureSwizzleChannels {
        red,
        green,
        blue,
        alpha,
    };
    [
        ("rgba", TextureSwizzleChannels::DEFAULT),
        ("ar10", s(Alpha, Red, One, Zero)),
        ("0ag1", s(Zero, Alpha, Green, One)),
        ("bgra", s(Blue, Green, Red, Alpha)),
        ("gggg", s(Green, Green, Green, Green)),
    ]
}

/// The operations a case runs, each with the kernel it dispatches and whether it samples.
///
/// `rf` reads a texel coordinate; `sn` and `sl` sample through the nearest and the linear sampler;
/// `g0` to `g3` gather x, y, z or w through the nearest one.
pub const OPS: [(&str, &str); 7] = [
    ("rf", "fpv_swz_read"),
    ("sn", "fpv_swz_sample"),
    ("sl", "fpv_swz_sample"),
    ("g0", "fpv_swz_gather_x"),
    ("g1", "fpv_swz_gather_y"),
    ("g2", "fpv_swz_gather_z"),
    ("g3", "fpv_swz_gather_w"),
];

/// Whether `op` samples, and so binds a sampler: every operation but the read.
fn samples(op: &str) -> bool {
    op != "rf"
}

/// Whether `op` samples through the linear sampler rather than the nearest one.
fn linear(op: &str) -> bool {
    op == "sl"
}

/// The reads' texel coordinates: every texel of the texture, then four past it.
pub fn reads() -> Vec<[u32; 2]> {
    let mut at: Vec<[u32; 2]> = (0..W * H).map(|i| [i % W, i / W]).collect();
    at.extend([[W, 0], [0, H], [W + 3, 1], [2, H + 5]]);
    at
}

/// The samples' and gathers' normalized positions: [`OFFSETS`] on the diagonal of each cell of the
/// texture and of the ring of cells around it, rows outer.
pub fn positions() -> Vec<[f32; 2]> {
    let mut points = Vec::new();
    for ky in -1..=i64::from(H) {
        for kx in -1..=i64::from(W) {
            for offset in OFFSETS {
                points.push([
                    ((kx as f64 + offset) / f64::from(W)) as f32,
                    ((ky as f64 + offset) / f64::from(H)) as f32,
                ]);
            }
        }
    }
    points
}

/// The bytes a texture of the format of raw value `raw` holds, `bytes` a texel: the probe's
/// 64-bit linear congruential state seeded by the raw value, each byte its top 8 bits.
pub fn texels(raw: u32, bytes: u32) -> Vec<u8> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ u64::from(raw);
    (0..W * H * bytes)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 56) as u8
        })
        .collect()
}

/// Buffer 1's bytes for the read: each texel coordinate's two words.
fn read_coords() -> Vec<u8> {
    reads()
        .iter()
        .flatten()
        .flat_map(|axis| axis.to_le_bytes())
        .collect()
}

/// Buffer 1's bytes for a sample or a gather: each position's two floats.
fn sample_coords() -> Vec<u8> {
    positions()
        .iter()
        .flatten()
        .flat_map(|axis| axis.to_le_bytes())
        .collect()
}

/// The sampler of `filter`, at clamp-to-border opaque white on both axes, as the lane made it.
fn sampler_spec(filter: SamplerMinMagFilter) -> SamplerSpec {
    let mut spec = SamplerSpec::filtered(filter);
    spec = spec.addressed(SamplerAddressMode::ClampToBorderColor);
    spec.border_color = SamplerBorderColor::OpaqueWhite;
    spec
}

/// The lanes `op` dispatches, which is the positions it reads.
fn lanes(op: &str) -> usize {
    match samples(op) {
        true => positions().len(),
        false => reads().len(),
    }
}

/// The bytes the output buffer of `op` holds: one `float4` a lane.
fn out_bytes(op: &str) -> usize {
    lanes(op) * 16
}

/// The layout `op`'s answer is read under: one row a lane, four unsigned words each, so a diff
/// names the position and the channel and prints the bits Metal produced.
fn layout(op: &str) -> ByteLayout {
    let length = out_bytes(op);
    ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: length as u64,
        },
        element: Element::U32,
        row_offset: 0,
        row_elements: 4,
        row_pitch: 16,
        rows: lanes(op) as u32,
        image_pitch: length as u32,
        images: 1,
        // The rows cover every byte of the buffer, so there is no gap between them.
        gaps: Gaps::Ignored,
    }
}

/// One format's sweep over the five swizzles and the seven operations.
pub struct ExtendedSwizzles {
    at: usize,
}

/// Every format's case.
pub fn all() -> Vec<ExtendedSwizzles> {
    (0..FORMATS.len())
        .map(|at| ExtendedSwizzles { at })
        .collect()
}

impl ExtendedSwizzles {
    /// The format the case sweeps.
    pub fn format(&self) -> &'static Format {
        &FORMATS[self.at]
    }

    /// The texture of `swizzle`, loaded with the format's bytes.
    fn request<'a>(
        &self,
        uploads: &'a [Upload<'a>],
        swizzle: TextureSwizzleChannels,
    ) -> TextureRequest<'a> {
        TextureRequest::read_2d(self.format().format.as_raw(), W, H)
            .swizzled(swizzle)
            .loaded(uploads)
    }
}

impl Case for ExtendedSwizzles {
    fn name(&self) -> &'static str {
        FORMATS[self.at].name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    /// The venue's memory model, and for a format the module filters itself the device's
    /// arithmetic of that filter (`bicubic-filter-arithmetic`).
    ///
    /// Each op reads every lane of its sweep in one dispatch, in groups of at most 64 threads, and
    /// no lane count here is a whole number of such groups. That shape used to make our side run
    /// the interior and the partial edge as dispatches of their own, announcing
    /// `metal2vulkan-degrade-split-dispatch`, and the family declared the code for it. It does not
    /// any more: a ragged `dispatchThreads:` on a kernel that cannot observe its threadgroup
    /// partition runs as Metal's single dispatch, and `Degradation::SplitDispatch` fires on more
    /// than one region. Measured on this M3 Max, 8 runs of the whole 1,103-case set: every row of
    /// this family announced no notice at all, where the recordings taken before that landing hold
    /// `metal2vulkan-degrade-split-dispatch` in their cost record's `degraded` set. So the
    /// declaration is gone, and a split here reads as an undeclared notice at once.
    ///
    /// The loosening that is left reaches `case/notices/*` alone. Every texel this family records
    /// is still compared bit for bit, which is the point of the family: a swizzle moves channels
    /// and an extended-range or packed format decides what a channel's bits are, and neither is a
    /// tolerance. `every_recorded_swizzle_is_the_unswizzled_texture_swizzled` and
    /// `every_recorded_packed_read_is_metals_rule` hold those words to Metal's own answers.
    fn allowances(&self) -> Vec<crate::allowance::Allowance> {
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the
        // notices site alone, so every value of this family still compares against Metal's
        // exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09): it fired for
        // all 15 rows of the family, and each was divergent on that field and
        // nothing else (replay-nvidia.log). On MoltenVK, where
        // the chip's GPU shares the CPU's memory, it cannot fire at all, which is why no
        // recording of this suite declares it.
        //
        // An extended-range or padded 10-bit format's linear sample (`sl`) blends its decoded taps
        // in the device's arithmetic, announced as `bicubic-filter-arithmetic` (the M3 Max, each
        // of the four extended-range formats); its values move by it (`known-gaps.tsv`).
        let mut named = vec![crate::replay::shared_memory_notice()];
        if matches!(
            self.format().format,
            PixelFormat::BGRA10XR
                | PixelFormat::BGRA10XRSRGB
                | PixelFormat::BGR10XR
                | PixelFormat::BGR10XRSRGB
                | PixelFormat::R10UnormX6
                | PixelFormat::R10UnormX6SRGB
                | PixelFormat::RG10UnormX12
                | PixelFormat::RG10UnormX12SRGB
        ) {
            named.push(crate::replay::notice_allowance(
                metal2vulkan_runtime::Degradation::BicubicFilterArithmetic.code(),
            ));
        }
        named
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        let mut entries: Vec<&str> = OPS.iter().map(|(_, kernel)| *kernel).collect();
        entries.dedup();
        for entry in entries {
            let fixture = Fixture {
                name: FIXTURE,
                entry,
            };
            for extension in [RUNTIME_TEXT, METAL_TEXT] {
                shaders.push(ShaderRef {
                    fixture: format!("{FIXTURE}{extension}"),
                    entry: entry.to_owned(),
                    source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
                });
            }
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = Fixture::new(FIXTURE);
        let format = self.format();
        let mut inputs = vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("buffer:1/reads", &read_coords()),
            Input::bytes("buffer:1/positions", &sample_coords()),
            Input::bytes("buffer:0/blank", &vec![BLANK; out_bytes("sn")]),
        ];
        // Every texture the sweep makes whole, with its texels, so a texture whose format, extent,
        // swizzle or bytes move is a change to the case's inputs.
        let bytes = texels(format.format.as_raw(), format.texel_bytes);
        let uploads = [Upload::level_0([W, H], W * format.texel_bytes, H, &bytes)];
        for (name, swizzle) in swizzles() {
            inputs.push(Input::bytes(
                format!("texture:{name}"),
                &self.request(&uploads, swizzle).words(),
            ));
        }
        // Both sampler states, so a sampler that moves is a change to the inputs.
        for (name, filter) in [
            ("nearest", SamplerMinMagFilter::Nearest),
            ("linear", SamplerMinMagFilter::Linear),
        ] {
            inputs.push(Input::bytes(
                format!("sampler:{name}"),
                &sampler_spec(filter).words(),
            ));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = Fixture::new(FIXTURE).loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let mut pipelines: Vec<(&str, Handle)> = Vec::new();
        for (_, kernel) in OPS {
            if pipelines.iter().any(|(held, _)| *held == kernel) {
                continue;
            }
            let pipeline = made(
                sites,
                &format!("pipeline/{kernel}"),
                side.compute_pipeline(library, kernel),
            )?;
            pipelines.push((kernel, pipeline));
        }
        let nearest = made(
            sites,
            "sampler/nearest",
            side.sampler(&sampler_spec(SamplerMinMagFilter::Nearest)),
        )?;
        let linear_state = made(
            sites,
            "sampler/linear",
            side.sampler(&sampler_spec(SamplerMinMagFilter::Linear)),
        )?;
        let reads_at = made(sites, "buffer/reads", side.buffer(&read_coords()))?;
        let positions_at = made(sites, "buffer/positions", side.buffer(&sample_coords()))?;
        let format = self.format();
        let bytes = texels(format.format.as_raw(), format.texel_bytes);
        let uploads = [Upload::level_0([W, H], W * format.texel_bytes, H, &bytes)];
        for (name, swizzle) in swizzles() {
            let texture = made(
                sites,
                &format!("{name}/texture"),
                side.texture(&self.request(&uploads, swizzle)),
            )?;
            for (op, kernel) in OPS {
                let at = |site: &str| format!("{name}/{op}/{site}");
                let pipeline = pipelines
                    .iter()
                    .find(|(held, _)| *held == kernel)
                    .map(|(_, pipeline)| *pipeline)
                    .expect("a pipeline the case made");
                let out = made(
                    sites,
                    &at("buffer/out"),
                    side.buffer(&vec![BLANK; out_bytes(op)]),
                )?;
                let coords = match samples(op) {
                    true => positions_at,
                    false => reads_at,
                };
                let states = [match linear(op) {
                    true => linear_state,
                    false => nearest,
                }];
                let bound = [texture];
                let buffers = [out, coords];
                let bindings = Bindings::new(&bound, &buffers);
                let bindings = match samples(op) {
                    true => bindings.sampled(&states),
                    false => bindings,
                };
                let count = lanes(op) as u32;
                let command_buffer = made(
                    sites,
                    &at("dispatch"),
                    side.dispatch_threads(
                        pipeline,
                        &bindings,
                        &[],
                        [count, 1, 1],
                        [count.min(64), 1, 1],
                    ),
                )?;
                call(
                    sites,
                    &at("status"),
                    side.wait(command_buffer),
                    Clone::clone,
                )?;
                let got = call(
                    sites,
                    &at("read-back"),
                    side.read_buffer(out, 0, out_bytes(op)),
                    |_| Value::Absent,
                )?;
                sites.push(Node::leaf(
                    format!("{name}/{op}"),
                    Observed::bytes(got, layout(op)),
                ));
            }
        }
        Ok(())
    }
}

/// The value a channel whose swizzle source is `source` reads from the unswizzled `texel`.
pub fn pick(source: TextureSwizzle, texel: [u32; 4]) -> u32 {
    match source {
        TextureSwizzle::Zero => 0,
        TextureSwizzle::One => 1.0f32.to_bits(),
        TextureSwizzle::Red => texel[0],
        TextureSwizzle::Green => texel[1],
        TextureSwizzle::Blue => texel[2],
        _ => texel[3],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::collections::BTreeSet;

    /// Every outcome's bytes at `site`, over every answer of `case`'s recording.
    fn recorded(recording: &crate::recording::Recording, site: &str) -> Vec<Vec<u8>> {
        let mut found = Vec::new();
        for answer in &recording.answers {
            for outcome in &answer.outcomes {
                if let Some(node) = outcome.transcript.root.child(site) {
                    if let Value::Bytes(bytes) = &node.observed.value {
                        found.push(bytes.clone());
                    }
                }
            }
        }
        found
    }

    /// `bytes` as one `float4`'s four words a lane.
    fn quads(bytes: &[u8]) -> Vec<[u32; 4]> {
        bytes
            .as_chunks::<16>()
            .0
            .iter()
            .map(|quad| {
                std::array::from_fn(|at| {
                    u32::from_le_bytes(quad[4 * at..4 * at + 4].try_into().expect("four bytes"))
                })
            })
            .collect()
    }

    /// Metal's answer for `op` of a texture made with `swizzle`, from the **unswizzled** texture's
    /// own recorded answers `plain` (by operation): a read's and a sample's channels each its
    /// source's; a gather of component `c` the gather of its source's component, or 0 or 1 at
    /// every tap.
    fn swizzled(
        op: &str,
        swizzle: TextureSwizzleChannels,
        plain: &BTreeMap<&str, Vec<[u32; 4]>>,
        at: usize,
    ) -> Option<[u32; 4]> {
        let sources = [swizzle.red, swizzle.green, swizzle.blue, swizzle.alpha];
        match op.strip_prefix('g') {
            None => {
                let unswizzled = plain.get(op)?[at];
                Some(sources.map(|source| pick(source, unswizzled)))
            }
            Some(component) => {
                let source = sources[component.parse::<usize>().ok()?];
                let gathered = match source {
                    TextureSwizzle::Zero => return Some([0; 4]),
                    TextureSwizzle::One => return Some([1.0f32.to_bits(); 4]),
                    TextureSwizzle::Red => "g0",
                    TextureSwizzle::Green => "g1",
                    TextureSwizzle::Blue => "g2",
                    _ => "g3",
                };
                plain.get(gathered).map(|held| held[at])
            }
        }
    }

    /// Every swizzled read, sample and gather Metal recorded is the **unswizzled** texture's
    /// recorded answer swizzled.
    ///
    /// This is the lane's whole subject, now checked against Metal's own words on both sides of the
    /// equation rather than against our own unswizzled leg: for the extended-range and bfloat
    /// formats the lane held no Metal rule at all, so it could only ask whether *we* were
    /// self-consistent. Here both terms are Metal's.
    #[test]
    fn every_recorded_swizzle_is_the_unswizzled_texture_swizzled() {
        let mut unrecorded = Vec::new();
        let mut differ = Vec::new();
        let mut checked = 0usize;
        for format in &FORMATS {
            let Ok(recording) = crate::recording::read(format.name) else {
                unrecorded.push(format.name);
                continue;
            };
            // The unswizzled texture's answers, one per operation, from this recording.
            let mut plain: BTreeMap<&str, Vec<[u32; 4]>> = BTreeMap::new();
            for (op, _) in OPS {
                if let Some(bytes) = recorded(&recording, &format!("rgba/{op}")).first() {
                    plain.insert(op, quads(bytes));
                }
            }
            if plain.len() != OPS.len() {
                differ.push(format!(
                    "{}: the unswizzled sweep holds {} of {} operations",
                    format.name,
                    plain.len(),
                    OPS.len()
                ));
                continue;
            }
            for (name, swizzle) in swizzles() {
                if name == "rgba" {
                    continue;
                }
                for (op, _) in OPS {
                    let site = format!("{name}/{op}");
                    for bytes in recorded(&recording, &site) {
                        for (at, got) in quads(&bytes).into_iter().enumerate() {
                            let Some(want) = swizzled(op, swizzle, &plain, at) else {
                                continue;
                            };
                            checked += 1;
                            if got != want && differ.len() < 24 {
                                differ.push(format!(
                                    "{}/{site} at {at}: Metal {got:08x?}, the unswizzled \
                                     texture's swizzled {want:08x?}",
                                    format.name
                                ));
                            }
                        }
                    }
                }
            }
        }
        assert!(
            differ.is_empty(),
            "{checked} results checked; Metal's swizzled answers are not its unswizzled ones \
             swizzled:\n{}",
            differ.join("\n")
        );
        assert!(
            unrecorded.is_empty(),
            "no recording yet for {unrecorded:?}: run `conformance record` on a Mac"
        );
        assert!(checked > 0, "the recordings held no result to check");
    }

    /// Metal's texel `(x, y)` of a texture of the packed normalized format `fields` describes,
    /// holding `bytes`: each channel's code `c` of `b` bits reads `c / (2^b - 1)`, correctly
    /// rounded, and an alpha the format lacks reads 1. `None` for a texel outside the texture.
    fn packed_texel(
        fields: &[Option<(u32, u32)>; 4],
        bytes: &[u8],
        (x, y): (i64, i64),
    ) -> Option<[u32; 4]> {
        if !(0..i64::from(W)).contains(&x) || !(0..i64::from(H)).contains(&y) {
            return None;
        }
        let at = ((y as usize * W as usize) + x as usize) * 2;
        let word = u32::from(u16::from_le_bytes([bytes[at], bytes[at + 1]]));
        Some(fields.map(|field| match field {
            Some((shift, bits)) => {
                let max = (1u32 << bits) - 1;
                let code = (word >> shift) & max;
                // An odd denominator's quotient is never an `f32` tie, so rounding it through
                // `f64` is the correctly rounded quotient.
                ((f64::from(code) / f64::from(max)) as f32).to_bits()
            }
            None => 1.0f32.to_bits(),
        }))
    }

    /// Metal's answer for `op` at `at` of an **unswizzled** texture of the packed normalized
    /// format `fields` describes, where the operation reads texels inside the texture alone: a read
    /// of a texel, a nearest sample's `floor(u n)` texel, and a gather's footprint — the texels
    /// `i, i + 1` from `floor(u n - 1/2)` on each axis, in gather order `(i, j + 1)`,
    /// `(i + 1, j + 1)`, `(i + 1, j)`, `(i, j)`.
    ///
    /// `None` for a linear sample, an access past the edges, and anything the rule does not cover.
    fn metal_packed(
        fields: &[Option<(u32, u32)>; 4],
        bytes: &[u8],
        op: &str,
        at: usize,
    ) -> Option<[u32; 4]> {
        let texel = |where_at| packed_texel(fields, bytes, where_at);
        let position = |at: usize| {
            positions().get(at).map(|point| {
                (
                    f64::from(point[0]) * f64::from(W),
                    f64::from(point[1]) * f64::from(H),
                )
            })
        };
        match op {
            "rf" => {
                let [x, y] = *reads().get(at)?;
                texel((i64::from(x), i64::from(y)))
            }
            "sn" => {
                let (x, y) = position(at)?;
                texel((x.floor() as i64, y.floor() as i64))
            }
            "sl" => None,
            gather => {
                let component: usize = gather.strip_prefix('g')?.parse().ok()?;
                let (x, y) = position(at)?;
                let (i, j) = ((x - 0.5).floor() as i64, (y - 0.5).floor() as i64);
                let taps = [(i, j + 1), (i + 1, j + 1), (i + 1, j), (i, j)];
                let mut out = [0; 4];
                for (slot, tap) in out.iter_mut().zip(taps) {
                    *slot = texel(tap)?[component];
                }
                Some(out)
            }
        }
    }

    /// Every unswizzled read, nearest sample and gather of a **packed normalized** texture that
    /// Metal recorded is the lane's code-to-value rule's.
    ///
    /// The four packed formats are the only ones of the fifteen the lane held a Metal rule for, so
    /// this is where the recordings can be held to a computed answer rather than to each other. A
    /// linear sample and an access past the edges are not here: the rule covers neither, which is
    /// exactly why the lane compared those against the unswizzled texture instead.
    #[test]
    fn every_recorded_packed_read_is_metals_rule() {
        let mut unrecorded = Vec::new();
        let mut differ = Vec::new();
        let mut checked = 0usize;
        let mut families = 0usize;
        for format in &FORMATS {
            let Some(fields) = format.packed else {
                continue;
            };
            families += 1;
            let Ok(recording) = crate::recording::read(format.name) else {
                unrecorded.push(format.name);
                continue;
            };
            let bytes = texels(format.format.as_raw(), format.texel_bytes);
            for (op, _) in OPS {
                let site = format!("rgba/{op}");
                for held in recorded(&recording, &site) {
                    for (at, got) in quads(&held).into_iter().enumerate() {
                        let Some(want) = metal_packed(&fields, &bytes, op, at) else {
                            continue;
                        };
                        checked += 1;
                        if got != want && differ.len() < 24 {
                            differ.push(format!(
                                "{}/{site} at {at}: Metal {got:08x?}, the rule's {want:08x?}",
                                format.name
                            ));
                        }
                    }
                }
            }
        }
        assert_eq!(families, 4, "the four packed normalized formats");
        assert!(
            differ.is_empty(),
            "{checked} results checked; Metal's recorded texels differ from the lane's \
             code-to-value rule:\n{}",
            differ.join("\n")
        );
        assert!(
            unrecorded.is_empty(),
            "no recording yet for {unrecorded:?}: run `conformance record` on a Mac"
        );
        assert!(checked > 0, "the recordings held no packed texel to check");
    }

    /// The cases name distinct recordings and distinct formats, and the operations distinct sites.
    #[test]
    fn the_cases_and_the_sites_are_distinct() {
        let named: BTreeSet<&str> = FORMATS.iter().map(|format| format.name).collect();
        assert_eq!(named.len(), FORMATS.len());
        let formats: BTreeSet<u32> = FORMATS
            .iter()
            .map(|format| format.format.as_raw())
            .collect();
        assert_eq!(formats.len(), FORMATS.len());
        let ops: BTreeSet<&str> = OPS.iter().map(|(op, _)| *op).collect();
        assert_eq!(ops.len(), OPS.len());
        let swizzled: BTreeSet<&str> = swizzles().iter().map(|(name, _)| *name).collect();
        assert_eq!(swizzled.len(), swizzles().len());
    }

    /// Each operation's layout covers exactly the bytes its read-back holds.
    #[test]
    fn the_layouts_cover_their_read_backs() {
        for (op, _) in OPS {
            assert_eq!(layout(op).extent(), out_bytes(op), "{op}");
        }
    }

    /// The reduced position set still reaches what it claims: the ring of cells outside the texture
    /// on every side, the texel centres, and both sides of the half-texel line.
    ///
    /// Without those a sweep could pass while sampling nothing a border texel reaches, which is
    /// half of what the family asks.
    #[test]
    fn the_positions_reach_the_border_and_both_sides_of_the_half_texel_line() {
        let points = positions();
        assert_eq!(
            points.len(),
            (W as usize + 2) * (H as usize + 2) * OFFSETS.len()
        );
        assert!(
            points.iter().any(|p| p[0] < 0.0 && p[1] < 0.0),
            "past the low corner"
        );
        assert!(
            points.iter().any(|p| p[0] > 1.0 && p[1] > 1.0),
            "past the high corner"
        );
        assert!(points
            .iter()
            .any(|p| p[0] < 0.0 && (0.0..1.0).contains(&p[1])));
        assert!(points
            .iter()
            .any(|p| (0.0..1.0).contains(&p[0]) && p[1] < 0.0));
        // Inside the texture, a point below the half-texel line on both axes, one at the centre,
        // and one above it: the three a linear footprint tells apart.
        let inside: Vec<[f64; 2]> = points
            .iter()
            .filter(|p| (0.0..1.0).contains(&p[0]) && (0.0..1.0).contains(&p[1]))
            .map(|p| {
                [
                    f64::from(p[0]) * f64::from(W) % 1.0,
                    f64::from(p[1]) * f64::from(H) % 1.0,
                ]
            })
            .collect();
        assert!(inside.iter().any(|p| p[0] < 0.49 && p[1] < 0.49), "below");
        assert!(
            inside.iter().any(|p| (0.49..0.51).contains(&p[0])),
            "the texel centre"
        );
        assert!(inside.iter().any(|p| p[0] > 0.51 && p[1] > 0.51), "above");
        // And the reads cover every texel of the texture and four past it.
        assert_eq!(reads().len(), (W * H) as usize + 4);
        assert!(reads().iter().any(|[x, _]| *x >= W));
        assert!(reads().iter().any(|[_, y]| *y >= H));
    }
}
