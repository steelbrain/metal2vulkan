//! Blending into render targets of Apple's extended-range formats: the codes a blend stores.
//!
//! The destination of these blends is the thing that makes them worth recording: the probe filled
//! each target with **random raw codes**, including the low bits a format's conversion is supposed
//! to drop and the two top bits a 10:10:10 word does not read, so a blend that reads a bit it should
//! not shows up as a code rather than as a rounding. That is also why the target is of shared
//! storage: an arbitrary code reaches a texture only through `replaceRegion:`, since neither a clear
//! nor a kernel write can set one.
//!
//! One case per format and blend — 18 of them — each a 128 x 64 target holding the probe's codes,
//! one full-screen draw of the owned fixture `rt6_samples`'s `half4` fill over the probe's 8,192
//! outputs under a `Load` load action, and a blend of source factor one, added, for color and alpha
//! alike, with destination factor:
//!
//! - zero (`one-zero`), which writes the source through the attachment's own conversion;
//! - one (`add`), which adds the destination's decoded value, so what the format decoded matters;
//! - one minus the source's alpha (`premul`), the consumer's own blend.
//!
//! A single-plane target's codes are blitted out. A two-plane target's plane 0 is blitted as its
//! words, and its alpha plane is recorded as the `float4` Metal's own `xr_two_planes`'s
//! `xr2p_read_float` reads back — the whole read rather than the lane's rounded alpha byte, since
//! the read is Metal's answer and the rounding is the lane's way of meeting a probe digest.
//!
//! The target generator is the probe's own, rebuilt here, and `the_targets_are_the_probes` checks it
//! against all eight digests the probe printed. Note that it is a **different** xorshift from the
//! one in [`super::extended_range_targets`]: this one stores the shifted state and multiplies
//! without storing the product. Both are the probe's; they are two generators, not one.
//!
//! Nothing here can fault a GPU: single-sampled targets of the case's own extent, one full-screen
//! triangle a pass, no loop, and every fetch inside its buffer.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Bindings, Case, Input, MetalSource, Side};
use crate::recording::ShaderRef;
use crate::render::{
    Blend, ColorState, Dispatch, PassColor, Plane, ReadAt, RenderPass, RenderPipelineSpec,
    RenderStep, RenderWork, TargetSpec,
};
use crate::transcript::{Node, Observed, Sites, Value};
use metal2vulkan::metal::BlendFactor;
use metal2vulkan_runtime::Degradation;

use super::extended_range_targets::{
    fixture, fixture_inputs, half_bytes, not_read, program, read_layout, shader_refs, shape,
    texel_layout, Format, FILLS, FILL_HALF, FORMATS, HEIGHT, PLANES, READ_FLOAT, VERTEX, WIDTH,
};

/// The kernel that writes a `float4` into a two-plane extended-range texture, which is how the
/// probe put the alpha plane's bytes there before the draw.
const WRITE_FLOAT: &str = "xr2p_write_float";

/// The outputs one draw covers.
const COUNT: usize = (WIDTH * HEIGHT) as usize;

/// The bytes a `float4` read-back takes.
const READ_BYTES: u32 = 16;

/// The probe's xorshift64* for the destination codes.
///
/// It is not the generator in [`super::extended_range_targets`]: that one stores the multiplied
/// word, this one stores the shifted state and multiplies without storing. Keeping both exactly as
/// the probe wrote them is what lets the digests below hold.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        let mut state = seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(0x632B_E59B_D9B4_E019);
        if state == 0 {
            state = 1;
        }
        let mut generator = Rng(state);
        for _ in 0..8 {
            generator.next();
        }
        generator
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(2_685_821_657_736_338_717)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// The probe's random raw destination texels of `format`, from the seed `1000 +` the format's Metal
/// number: plane 0's bytes, and a two-plane format's alpha bytes.
///
/// A `BGRA10_XR` texel is four 16-bit fields, each a code shifted up six with six random low bits —
/// bits the format does not read, so a blend that reads them shows. Alpha's code sits mostly in
/// `[384, 894]`, the range a unit alpha covers. Every other format's texel is one word of three
/// codes with two random top bits.
fn destination(format: &Format) -> (Vec<u8>, Vec<u8>) {
    let mut generator = Rng::new(1000 + u64::from(format.format.as_raw()));
    let code = |generator: &mut Rng, alpha: bool| -> u64 {
        if alpha && generator.below(10) < 8 {
            return 384 + generator.below(511);
        }
        generator.below(1024)
    };
    let (mut plane, mut alpha) = (Vec::new(), Vec::new());
    for _ in 0..COUNT {
        if format.texel == 8 {
            for channel in 0..4 {
                let field = (code(&mut generator, channel == 3) << 6 | generator.below(64)) as u16;
                plane.extend(field.to_le_bytes());
            }
            continue;
        }
        let top = generator.below(4) << 30;
        let (red, green) = (code(&mut generator, false), code(&mut generator, false));
        let word = (top | red << 20 | green << 10 | code(&mut generator, false)) as u32;
        plane.extend(word.to_le_bytes());
        if !format.copies {
            alpha.push(generator.below(256) as u8);
        }
    }
    (plane, alpha)
}

/// The `float4`s the alpha-plane write feeds: `(0, 0, 0, byte / 255)` a texel, as the probe wrote
/// them.
fn alpha_values(alpha: &[u8]) -> Vec<u8> {
    alpha
        .iter()
        .flat_map(|&byte| [0.0, 0.0, 0.0, f32::from(byte) / 255.0])
        .flat_map(f32::to_le_bytes)
        .collect()
}

/// One blend the lane sweeps: its name and the destination factor, under source factor one, added,
/// for color and alpha alike.
const BLENDS: [(&str, BlendFactor); 3] = [
    ("one-zero", BlendFactor::Zero),
    ("add", BlendFactor::One),
    ("premul", BlendFactor::OneMinusSourceAlpha),
];

/// A case: one format blended under one of the three blends.
pub struct ExtendedRangeBlend {
    name: &'static str,
    format: Format,
    blend: Blend,
    /// The blend's name, for the input fingerprint's label.
    blend_name: &'static str,
}

/// Every pairing of a format with a blend, as the lane sweeps them.
pub fn all() -> Vec<ExtendedRangeBlend> {
    // The names are written out rather than built, so a case's recording file is named by something
    // a reader can grep for.
    const NAMES: [[&str; 3]; 6] = [
        [
            "xr-blend-bgra10xr-one-zero",
            "xr-blend-bgra10xr-add",
            "xr-blend-bgra10xr-premul",
        ],
        [
            "xr-blend-bgra10xr-srgb-one-zero",
            "xr-blend-bgra10xr-srgb-add",
            "xr-blend-bgra10xr-srgb-premul",
        ],
        [
            "xr-blend-bgr10xr-one-zero",
            "xr-blend-bgr10xr-add",
            "xr-blend-bgr10xr-premul",
        ],
        [
            "xr-blend-bgr10xr-srgb-one-zero",
            "xr-blend-bgr10xr-srgb-add",
            "xr-blend-bgr10xr-srgb-premul",
        ],
        [
            "xr-blend-planes-one-zero",
            "xr-blend-planes-add",
            "xr-blend-planes-premul",
        ],
        [
            "xr-blend-planes-srgb-one-zero",
            "xr-blend-planes-srgb-add",
            "xr-blend-planes-srgb-premul",
        ],
    ];
    let mut cases = Vec::new();
    for (format, names) in FORMATS.iter().zip(NAMES) {
        for ((blend_name, factor), name) in BLENDS.iter().zip(names) {
            cases.push(ExtendedRangeBlend {
                name,
                format: *format,
                blend: Blend::added(BlendFactor::One, *factor),
                blend_name,
            });
        }
    }
    cases
}

impl ExtendedRangeBlend {
    /// The target the case blends into: shared, because its destination codes go in through
    /// `replaceRegion:`, and shader-written because a two-plane target's alpha plane is written by
    /// a kernel first.
    fn target(&self) -> TargetSpec {
        TargetSpec::read(self.format.format, [WIDTH, HEIGHT])
            .written()
            .shared()
    }

    /// The pipeline: the `half4` fill into one blended attachment of the target's format.
    fn pipeline<'a>(&self, colors: &'a [Option<ColorState>]) -> RenderPipelineSpec<'a> {
        RenderPipelineSpec::new(VERTEX, FILL_HALF, colors)
    }

    /// The attachment state the pipeline blends with.
    fn colors(&self) -> [Option<ColorState>; 1] {
        [Some(
            ColorState::new(self.format.format).blended(self.blend),
        )]
    }
}

impl Case for ExtendedRangeBlend {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // `rt6_samples` is a fast-math module, so a pipeline that stores its `half4` fill into a
        // color attachment announces that the half output's rounding into the attachment may not be
        // Metal's. Only the notice set is loosened, so every stored code still compares exactly: a
        // device that blends as Metal does shows as announced with nothing differing -- which is
        // what this M3 Max and lavapipe both show -- and one that does not shows the code. The
        // allowance fails the case if the notice stops firing, so it cannot outlive its reason.
        //
        // The pass loads and stores an extended-range colour attachment, so our load pass
        // (`render/unused_bits_load.rs`) rewrites every 32 x 32 tile of the render area where
        // Metal's rewrites only the tiles a draw's primitives reach, announced as
        // `unreached-tiles-rewritten`. The difference it names cannot reach this case: the draw is
        // one full-screen triangle over the whole 128 x 64 target, so every tile is a reached
        // tile, and every stored code compares exactly on MoltenVK and on lavapipe alike. So this
        // is a declaration of what our side announced and not a tolerance -- no code is loosened.
        //
        // The two extended-range blend kinds (`extended-range-blend-source` and
        // `extended-range-blends-stored-codes`) are deliberately **not** named here: neither fired
        // on either venue of this round, and an allowance no run justified is a claim without
        // evidence. A venue that needs one adds it with the run that shows it.
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the
        // notices site alone, so every value of this family still compares against Metal's
        // exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09): it fired for all 18 rows of the family and each
        // was divergent on that field and nothing else (replay-nvidia.log). On MoltenVK, where
        // the chip's GPU shares the CPU's memory, it cannot fire at all, which is why no
        // recording of this suite declares it.
        std::iter::once(crate::replay::shared_memory_notice())
            .chain(
                [
                    Degradation::HalfOutputFastMathBlend,
                    Degradation::UnreachedTilesRewritten,
                ]
                .into_iter()
                .map(|kind| crate::replay::notice_allowance(kind.code())),
            )
            .collect()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = shader_refs(FILLS, &[VERTEX, FILL_HALF])?;
        if !self.format.copies {
            shaders.extend(shader_refs(PLANES, &[WRITE_FLOAT, READ_FLOAT])?);
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs("fills", FILLS)?;
        if !self.format.copies {
            inputs.extend(fixture_inputs("planes", PLANES)?);
        }
        let (plane, alpha) = destination(&self.format);
        let colors = self.colors();
        inputs.extend([
            Input::bytes("format", &self.format.format.as_raw().to_le_bytes()),
            // The blend is the question the case asks, so its six fields and its name are inputs.
            Input::bytes("blend", &self.blend.words()),
            Input::bytes("blend-name", self.blend_name.as_bytes()),
            Input::bytes("pipeline", &self.pipeline(&colors).words()),
            Input::bytes("target", &self.target().words()),
            // The destination the blend reads, which is the point of the family.
            Input::bytes("texture:plane0", &plane),
            Input::bytes("buffer:values", &half_bytes()),
            Input::bytes("buffer:shape", &shape(WIDTH, 1)),
        ]);
        if !alpha.is_empty() {
            inputs.push(Input::bytes("texture:alpha", &alpha));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let (plane, alpha) = destination(&self.format);
        let fills = fixture(FILLS, VERTEX)
            .loaded()
            .map_err(|why| not_read("library/fills", why))?;
        let library = made(sites, "library/fills", side.library(&fills.source()))?;
        let colors = self.colors();
        let pipeline = made(
            sites,
            "pipeline",
            side.render_pipeline(library, &self.pipeline(&colors)),
        )?;
        let values = made(sites, "buffer/values", side.buffer(&half_bytes()))?;
        let extent = made(sites, "buffer/shape", side.buffer(&shape(WIDTH, 1)))?;
        let target = made(sites, "target", side.target(&self.target()))?;

        // The two-plane formats' alpha plane goes in first, by a kernel write of `b / 255`, and
        // plane 0's codes after it: that is the order the probe wrote them, and plane 0's words
        // would otherwise be rewritten by the kernel's own conversion.
        let mut planes_library = None;
        if !alpha.is_empty() {
            let planes = fixture(PLANES, WRITE_FLOAT)
                .loaded()
                .map_err(|why| not_read("library/planes", why))?;
            let held = made(sites, "library/planes", side.library(&planes.source()))?;
            planes_library = Some(held);
            let write = made(
                sites,
                "pipeline/write",
                side.compute_pipeline(held, WRITE_FLOAT),
            )?;
            let feed = made(sites, "buffer/alpha", side.buffer(&alpha_values(&alpha)))?;
            let command_buffer = made(
                sites,
                "dispatch/alpha",
                side.dispatch_threads(
                    write,
                    &Bindings::new(&[target], &[feed, extent]),
                    &[],
                    [WIDTH, HEIGHT, 1],
                    [32, 32, 1],
                ),
            )?;
            call(
                sites,
                "status/alpha",
                side.wait(command_buffer),
                Clone::clone,
            )?;
        }
        call(
            sites,
            "replace-region",
            side.replace_region(target, [WIDTH, HEIGHT], WIDTH * self.format.texel, &plane),
            |_| Value::Absent,
        )?;

        let pass = RenderWork::Pass(
            RenderPass::clearing(vec![PassColor::loaded(target)]).drawing(vec![
                RenderStep::Pipeline(pipeline),
                RenderStep::FragmentBuffer {
                    buffer: values,
                    offset: 0,
                    index: 0,
                },
                RenderStep::FragmentBuffer {
                    buffer: extent,
                    offset: 0,
                    index: 1,
                },
                RenderStep::triangle(),
            ]),
        );
        let texel_bytes = (WIDTH * HEIGHT * self.format.texel) as usize;
        let texels_buffer = made(sites, "buffer/texels", side.zeroed_buffer(texel_bytes))?;
        let mut work = vec![
            pass,
            RenderWork::ReadTexture {
                texture: target,
                into: texels_buffer,
                offset: 0,
                row_pitch: WIDTH * self.format.texel,
                size: [WIDTH, HEIGHT],
                plane: Plane::Whole,
                at: ReadAt::BASE,
            },
        ];
        // A two-plane target's alpha plane is not in the blit, so the case also records what
        // Metal's own kernel reads back from the whole texel.
        let reads_buffer = match planes_library {
            Some(held) => {
                let read = made(
                    sites,
                    "pipeline/read",
                    side.compute_pipeline(held, READ_FLOAT),
                )?;
                let bytes = (WIDTH * HEIGHT * READ_BYTES) as usize;
                let buffer = made(sites, "buffer/reads", side.zeroed_buffer(bytes))?;
                work.push(RenderWork::Dispatch(Dispatch {
                    pipeline: read,
                    buffers: vec![(buffer, 0), (extent, 0)],
                    textures: vec![target],
                    grid: [WIDTH, HEIGHT, 1],
                    threads: [32, 32, 1],
                }));
                Some((buffer, bytes))
            }
            None => None,
        };
        let outcome = call(sites, "render", side.render(&work), |_| Value::Absent)?;
        program(sites, "program", &outcome, &work);

        let stored = call(
            sites,
            "read-back/texels",
            side.read_buffer(texels_buffer, 0, texel_bytes),
            |_| Value::Absent,
        )?;
        sites.push(Node::leaf(
            "texels",
            Observed::bytes(stored, texel_layout(&self.format, [WIDTH, HEIGHT])),
        ));
        if let Some((buffer, bytes)) = reads_buffer {
            let got = call(
                sites,
                "read-back/reads",
                side.read_buffer(buffer, 0, bytes),
                |_| Value::Absent,
            )?;
            sites.push(Node::leaf(
                "reads",
                Observed::bytes(got, read_layout(WIDTH, HEIGHT)),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The probe's digests of each format's destination texels before the draw (its `.dst0` and
    /// `.dsta` files), in the order [`FORMATS`] lists the formats.
    const TARGETS: [(&str, Option<&str>); 6] = [
        (
            "5500646f25136cb1ee712c537bf52bd652692caf5ac75bfec166ad054851391c",
            None,
        ),
        (
            "ec04259f2cf0abd058a45a43d2ae42eee1d2b69f9a75380a6e220f2b80ee3885",
            None,
        ),
        (
            "86f240ed1326042e075dfb6b630025040e7d0900516125a06e202d05dd59b4ea",
            None,
        ),
        (
            "10d671ec19741d21a6cd2d5684458fff5cb4bfcfdf9a6f7656497ad057874073",
            None,
        ),
        (
            "3a4d452aaf8e6c1ac7f54f37d6ae19ea7d9fff3ac8092124734437f6a21ec62f",
            Some("773162c1efa6ad307364ff468da6540c4e991d2f70a7aed787dc303d77870d7b"),
        ),
        (
            "5f100cc206d94f512a8a0e72c14685adb2de0bf1d22333bc7f6fc767dd55fbf1",
            Some("9629c65d83e460c4a303776ab33ca4fe0a35ae89f9f69356792fa49bd82c183e"),
        ),
    ];

    /// The rebuilt generator gives the probe's destination texels: all eight digests the probe
    /// printed. Without this the cases would compare Metal's answer against other destination
    /// codes the moment the generator drifted, which is the one thing the blend's answer depends
    /// on most.
    #[test]
    fn the_targets_are_the_probes() {
        for (format, (plane_digest, alpha_digest)) in FORMATS.iter().zip(TARGETS) {
            let (plane, alpha) = destination(format);
            assert_eq!(
                plane.len(),
                COUNT * format.texel as usize,
                "{}",
                format.slug
            );
            assert_eq!(
                crate::sha256_hex(&plane),
                plane_digest,
                "{}: plane 0",
                format.slug
            );
            match alpha_digest {
                Some(digest) => {
                    assert_eq!(alpha.len(), COUNT, "{}", format.slug);
                    assert_eq!(
                        crate::sha256_hex(&alpha),
                        digest,
                        "{}: the alpha plane",
                        format.slug
                    );
                }
                None => assert!(alpha.is_empty(), "{}", format.slug),
            }
        }
    }

    /// Every case names a distinct row and a distinct question, so no two share a recording.
    #[test]
    fn the_cases_name_distinct_rows() {
        let cases = all();
        let named: BTreeSet<&str> = cases.iter().map(|case| case.name).collect();
        assert_eq!(named.len(), cases.len());
        let asked: BTreeSet<(u32, Vec<u8>)> = cases
            .iter()
            .map(|case| (case.format.format.as_raw(), case.blend.words()))
            .collect();
        assert_eq!(asked.len(), cases.len());
        assert_eq!(cases.len(), FORMATS.len() * BLENDS.len());
    }

    /// The three blends differ in the one field they are meant to: the destination factor, on the
    /// color channels and the alpha channel alike, with the source factor one and the operation
    /// add throughout.
    #[test]
    fn the_blends_differ_only_in_the_destination_factor() {
        for (_, factor) in BLENDS {
            let blend = Blend::added(BlendFactor::One, factor);
            assert_eq!(blend.source_rgb, BlendFactor::One);
            assert_eq!(blend.source_alpha, BlendFactor::One);
            assert_eq!(blend.destination_rgb, factor);
            assert_eq!(blend.destination_alpha, factor);
        }
        let factors: BTreeSet<u32> = BLENDS.iter().map(|(_, factor)| factor.as_raw()).collect();
        assert_eq!(factors.len(), BLENDS.len());
    }

    /// The alpha feed is one `float4` a texel whose alpha is the probe's byte over 255, and the
    /// other three channels are zero: a feed that wrote a colour would rewrite plane 0 before the
    /// host's codes went in.
    #[test]
    fn the_alpha_feed_writes_the_alpha_alone() {
        let bytes = alpha_values(&[0, 128, 255]);
        assert_eq!(bytes.len(), 3 * 16);
        let word = |at: usize| f32::from_le_bytes(bytes[at * 4..at * 4 + 4].try_into().unwrap());
        assert_eq!(word(0), 0.0);
        assert_eq!(word(1), 0.0);
        assert_eq!(word(2), 0.0);
        assert_eq!(word(3), 0.0);
        assert_eq!(word(7), 128.0 / 255.0);
        assert_eq!(word(11), 1.0);
    }

    /// The read-back layout covers the whole `float4` read of the draws' extent, and the texel
    /// layout the whole blitted level: a layout that reached further would read past its buffer.
    #[test]
    fn the_layouts_cover_their_buffers() {
        assert_eq!(
            read_layout(WIDTH, HEIGHT).extent(),
            (WIDTH * HEIGHT * READ_BYTES) as usize
        );
        for format in &FORMATS {
            assert_eq!(
                texel_layout(format, [WIDTH, HEIGHT]).extent(),
                (WIDTH * HEIGHT * format.texel) as usize,
                "{}",
                format.slug
            );
        }
    }
}
