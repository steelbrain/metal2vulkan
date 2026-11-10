//! Blending into render targets of Apple's padded 10-bit formats: the codes a blend stores.
//!
//! `R10Unorm_X6`, `RG10Unorm_X12`, their `_sRGB` twins and `RGB10A2Unorm_sRGB` hold 10-bit
//! normalized codes (`RGB10A2Unorm_sRGB`'s alpha a 2-bit one), each of the first four in the top
//! ten bits of a 16-bit word. No Vulkan format holds them, so the runtime renders such a target
//! as its raw unsigned words and the fragment function computes the blend over an ordered read of
//! the attachment, as for the extended-range formats ([`super::extended_range_blends`]). Metal
//! makes such a pipeline and draws it (the M3 Max under macOS 27.0.1,
//! `probes/padded_and_bfloat_blends.m`): it blends the output, the second output and the blend
//! color **unclamped**, every channel alpha included, over the tile's `half` of the destination,
//! and stores the result's code.
//!
//! One case per format and blend, each a 128 x 64 target of random raw words (the codes and the
//! six low bits no padded word reads, or `RGB10A2Unorm_sRGB`'s whole word), one full-screen draw
//! under a `Load` load action, and the target's words blitted out:
//!
//! - `premul-half`: the owned fixture `rt6_samples`' `half4` fill under premultiplied source-over
//!   (one, one minus the source's alpha, for color and alpha);
//! - `add-float`: its `float4` fill under one plus one;
//! - `dual-half`: the owned fixture `xrb_dual`'s `half4` output and second output under
//!   `Source1Color` and `Source1Alpha` over their complements;
//! - `premul-red-half` (`RG10Unorm_X12`) and `premul-rgb-half` (`RGB10A2Unorm_sRGB`): the
//!   premultiplied blend under a write mask of some of the format's channels.
//!
//! The outputs are [`super::extended_range_targets`]' 8,192: values across and past `[0, 1]`
//! on both sides, code boundaries, NaN, the infinities and both zeros. The second outputs are
//! the same values in reverse order.
//!
//! The target is of shared storage and shader-write usage: its random words go in through
//! `replaceRegion:`, and a texture Metal keeps in its optimized layout (a colour format of
//! neither `ShaderWrite` nor `PixelFormatView` usage) moved only some of a padded target's
//! texels through blits in the probe.
//!
//! Nothing here can fault a GPU: single-sampled targets of the case's own extent, one full-screen
//! triangle a pass, no loop, and every fetch inside its buffer.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Case, Input, MetalSource, Side};
use crate::recording::ShaderRef;
use crate::render::{
    Blend, ColorState, PassColor, Plane, ReadAt, RenderPass, RenderPipelineSpec, RenderStep,
    RenderWork, TargetSpec,
};
use crate::transcript::{Element, Node, Observed, Sites, Value};
use metal2vulkan::metal::{BlendFactor as F, BlendOperation as O, ColorWriteMask, PixelFormat};
use metal2vulkan_runtime::Degradation;

use super::extended_range_targets::{
    fixture, fixture_inputs, float_bytes, half_bytes, not_read, program, shader_refs, shape,
    texel_layout, Format, FILLS, FILL_FLOAT, FILL_HALF, HEIGHT, VERTEX, WIDTH,
};

/// The owned fixture whose fragment functions return an output and a second one
/// (`[[color(0), index(1)]]`), built without fast math.
const DUAL: &str = "xrb_dual";
const DUAL_VERTEX: &str = "xrb_dual_vertex";
const DUAL_HALF: &str = "xrb_dual_half";

/// The outputs one draw covers.
const COUNT: usize = (WIDTH * HEIGHT) as usize;

/// The padded formats, as targets whose words Metal copies.
const PADDED: [Format; 5] = [
    Format {
        slug: "r10x6",
        format: PixelFormat::R10UnormX6,
        texel: 2,
        word: Element::U16,
        words: 1,
        copies: true,
    },
    Format {
        slug: "r10x6-srgb",
        format: PixelFormat::R10UnormX6SRGB,
        texel: 2,
        word: Element::U16,
        words: 1,
        copies: true,
    },
    Format {
        slug: "rg10x12",
        format: PixelFormat::RG10UnormX12,
        texel: 4,
        word: Element::U16,
        words: 2,
        copies: true,
    },
    Format {
        slug: "rg10x12-srgb",
        format: PixelFormat::RG10UnormX12SRGB,
        texel: 4,
        word: Element::U16,
        words: 2,
        copies: true,
    },
    Format {
        slug: "rgb10a2-srgb",
        format: PixelFormat::RGB10A2UnormSRGB,
        texel: 4,
        word: Element::U32,
        words: 1,
        copies: true,
    },
];

/// The function a case draws, and the blend it draws under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Draw {
    /// `rt6_samples`' `half4` fill, premultiplied source-over.
    PremulHalf,
    /// `rt6_samples`' `float4` fill, one plus one.
    AddFloat,
    /// `xrb_dual`'s `half4` output and second output, `Source1` over its complement.
    DualHalf,
}

impl Draw {
    fn blend(self) -> Blend {
        match self {
            Draw::PremulHalf => Blend::added(F::One, F::OneMinusSourceAlpha),
            Draw::AddFloat => Blend::added(F::One, F::One),
            Draw::DualHalf => Blend::separate(
                (F::Source1Color, F::OneMinusSource1Color, O::Add),
                (F::Source1Alpha, F::OneMinusSource1Alpha, O::Add),
            ),
        }
    }

    /// The fixture, its vertex function and its fragment function.
    fn program(self) -> (&'static str, &'static str, &'static str) {
        match self {
            Draw::PremulHalf => (FILLS, VERTEX, FILL_HALF),
            Draw::AddFloat => (FILLS, VERTEX, FILL_FLOAT),
            Draw::DualHalf => (DUAL, DUAL_VERTEX, DUAL_HALF),
        }
    }

    /// The outputs' bytes.
    fn values(self) -> Vec<u8> {
        match self {
            Draw::AddFloat => float_bytes(),
            Draw::PremulHalf | Draw::DualHalf => half_bytes(),
        }
    }

    /// Whether the function returns a second output.
    fn dual(self) -> bool {
        self == Draw::DualHalf
    }
}

/// The second outputs' bytes: the outputs' `half4`s in reverse order, so each pixel's second
/// output is another pixel's output.
fn second_bytes() -> Vec<u8> {
    let halves = half_bytes();
    halves
        .as_chunks::<8>()
        .0
        .iter()
        .rev()
        .flatten()
        .copied()
        .collect()
}

/// A case: one padded format blended by one draw under one write mask.
pub struct PaddedBlend {
    name: &'static str,
    format: Format,
    draw: Draw,
    mask: ColorWriteMask,
}

/// Every case of the family.
pub fn all() -> Vec<PaddedBlend> {
    // The names are written out rather than built, so a case's recording file is named by something
    // a reader can grep for.
    const NAMES: [[&str; 3]; 5] = [
        [
            "padded-blend-r10x6-premul-half",
            "padded-blend-r10x6-add-float",
            "padded-blend-r10x6-dual-half",
        ],
        [
            "padded-blend-r10x6-srgb-premul-half",
            "padded-blend-r10x6-srgb-add-float",
            "padded-blend-r10x6-srgb-dual-half",
        ],
        [
            "padded-blend-rg10x12-premul-half",
            "padded-blend-rg10x12-add-float",
            "padded-blend-rg10x12-dual-half",
        ],
        [
            "padded-blend-rg10x12-srgb-premul-half",
            "padded-blend-rg10x12-srgb-add-float",
            "padded-blend-rg10x12-srgb-dual-half",
        ],
        [
            "padded-blend-rgb10a2-srgb-premul-half",
            "padded-blend-rgb10a2-srgb-add-float",
            "padded-blend-rgb10a2-srgb-dual-half",
        ],
    ];
    let draws = [Draw::PremulHalf, Draw::AddFloat, Draw::DualHalf];
    let mut cases = Vec::new();
    for (format, names) in PADDED.iter().zip(NAMES) {
        for (draw, name) in draws.into_iter().zip(names) {
            cases.push(PaddedBlend {
                name,
                format: *format,
                draw,
                mask: ColorWriteMask::ALL,
            });
        }
    }
    let rgb = ColorWriteMask::RED
        .union(ColorWriteMask::GREEN)
        .union(ColorWriteMask::BLUE);
    cases.push(PaddedBlend {
        name: "padded-blend-rg10x12-premul-red-half",
        format: PADDED[2],
        draw: Draw::PremulHalf,
        mask: ColorWriteMask::RED,
    });
    cases.push(PaddedBlend {
        name: "padded-blend-rgb10a2-srgb-premul-rgb-half",
        format: PADDED[4],
        draw: Draw::PremulHalf,
        mask: rgb,
    });
    cases
}

/// The xorshift64* of the destination words.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(2_685_821_657_736_338_717)
    }
}

/// The random raw words of a `format` target, from the seed `2000 +` the format's Metal number:
/// every bit random, a padded word's six low bits and `RGB10A2Unorm_sRGB`'s alpha among them.
fn destination(format: &Format) -> Vec<u8> {
    let mut generator = Rng(2000 + u64::from(format.format.as_raw()));
    // A small seed's first words are mostly zero bits: past them first.
    for _ in 0..8 {
        generator.next();
    }
    (0..COUNT * format.texel as usize)
        .map(|_| (generator.next() >> 56) as u8)
        .collect()
}

impl PaddedBlend {
    /// The target: shared, its words going in through `replaceRegion:`, and of shader-write
    /// usage, which keeps Metal from its optimized layout (the module docs).
    fn target(&self) -> TargetSpec {
        TargetSpec::read(self.format.format, [WIDTH, HEIGHT])
            .written()
            .shared()
    }

    /// The attachment state the pipeline blends with.
    fn colors(&self) -> [Option<ColorState>; 1] {
        [Some(
            ColorState::new(self.format.format)
                .masked(self.mask)
                .blended(self.draw.blend()),
        )]
    }
}

impl Case for PaddedBlend {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason); it loosens the notices
        // site alone. The pass loads and stores a padded colour attachment, so its draw announces
        // `extended-range-bits-kept`: the device's store keeps the bits the codes leave unused in
        // the texels the draw does not write, where Metal's pass makes them canonical. The draw is
        // one full-screen triangle over the whole target, so a texel it does not write is one its
        // mask leaves channels of, and every stored word still compares exactly. `rt6_samples` is
        // a fast-math module, so a pipeline that blends its `half4` fill announces
        // `half-output-fast-math-blend`. The blend in the device's arithmetic announces
        // `extended-range-blend-source`, a native note the suite declares for every case
        // (`crate::replay::NATIVE_NOTES`); a word the device's arithmetic moves is a known gap.
        let mut named = vec![
            crate::replay::shared_memory_notice(),
            crate::replay::notice_allowance(Degradation::ExtendedRangeBitsKept.code()),
        ];
        if self.draw == Draw::PremulHalf {
            named.push(crate::replay::notice_allowance(
                Degradation::HalfOutputFastMathBlend.code(),
            ));
        }
        named
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let (name, vertex, fragment) = self.draw.program();
        shader_refs(name, &[vertex, fragment])
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let (name, _, _) = self.draw.program();
        let mut inputs = fixture_inputs("program", name)?;
        let colors = self.colors();
        let (_, vertex, fragment) = self.draw.program();
        inputs.extend([
            Input::bytes("format", &self.format.format.as_raw().to_le_bytes()),
            Input::bytes("blend", &self.draw.blend().words()),
            Input::bytes("mask", &self.mask.as_raw().to_le_bytes()),
            Input::bytes(
                "pipeline",
                &RenderPipelineSpec::new(vertex, fragment, &colors).words(),
            ),
            Input::bytes("target", &self.target().words()),
            Input::bytes("texture:destination", &destination(&self.format)),
            Input::bytes("buffer:values", &self.draw.values()),
            Input::bytes("buffer:shape", &shape(WIDTH, 1)),
        ]);
        if self.draw.dual() {
            inputs.push(Input::bytes("buffer:second", &second_bytes()));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let (name, vertex, fragment) = self.draw.program();
        let loaded = fixture(name, vertex)
            .loaded()
            .map_err(|why| not_read("library", why))?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let colors = self.colors();
        let pipeline = made(
            sites,
            "pipeline",
            side.render_pipeline(library, &RenderPipelineSpec::new(vertex, fragment, &colors)),
        )?;
        let values = made(sites, "buffer/values", side.buffer(&self.draw.values()))?;
        let extent = made(sites, "buffer/shape", side.buffer(&shape(WIDTH, 1)))?;
        let seconds = match self.draw.dual() {
            true => Some(made(sites, "buffer/second", side.buffer(&second_bytes()))?),
            false => None,
        };
        let target = made(sites, "target", side.target(&self.target()))?;
        call(
            sites,
            "replace-region",
            side.replace_region(
                target,
                [WIDTH, HEIGHT],
                WIDTH * self.format.texel,
                &destination(&self.format),
            ),
            |_| Value::Absent,
        )?;
        let mut steps = vec![
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
        ];
        if let Some(seconds) = seconds {
            steps.push(RenderStep::FragmentBuffer {
                buffer: seconds,
                offset: 0,
                index: 2,
            });
        }
        steps.push(RenderStep::triangle());
        let texel_bytes = (WIDTH * HEIGHT * self.format.texel) as usize;
        let texels = made(sites, "buffer/texels", side.zeroed_buffer(texel_bytes))?;
        let work = vec![
            RenderWork::Pass(RenderPass::clearing(vec![PassColor::loaded(target)]).drawing(steps)),
            RenderWork::ReadTexture {
                texture: target,
                into: texels,
                offset: 0,
                row_pitch: WIDTH * self.format.texel,
                size: [WIDTH, HEIGHT],
                plane: Plane::Whole,
                at: ReadAt::BASE,
            },
        ];
        let outcome = call(sites, "render", side.render(&work), |_| Value::Absent)?;
        program(sites, "program", &outcome, &work);
        let stored = call(
            sites,
            "read-back/texels",
            side.read_buffer(texels, 0, texel_bytes),
            |_| Value::Absent,
        )?;
        sites.push(Node::leaf(
            "texels",
            Observed::bytes(stored, texel_layout(&self.format, [WIDTH, HEIGHT])),
        ));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// Every case names a distinct row and a distinct question, so no two share a recording.
    #[test]
    fn the_cases_name_distinct_rows() {
        let cases = all();
        let named: BTreeSet<&str> = cases.iter().map(|case| case.name).collect();
        assert_eq!(named.len(), cases.len());
        let asked: BTreeSet<(u32, Vec<u8>, u32, &str)> = cases
            .iter()
            .map(|case| {
                (
                    case.format.format.as_raw(),
                    case.draw.blend().words(),
                    case.mask.as_raw(),
                    case.draw.program().2,
                )
            })
            .collect();
        assert_eq!(asked.len(), cases.len());
        assert_eq!(cases.len(), PADDED.len() * 3 + 2);
    }

    /// The destination covers the target and differs between formats; the second outputs are
    /// the outputs reversed, a `half4` each.
    #[test]
    fn the_inputs_cover_their_targets() {
        let mut seen = BTreeSet::new();
        for format in &PADDED {
            let bytes = destination(format);
            assert_eq!(
                bytes.len(),
                COUNT * format.texel as usize,
                "{}",
                format.slug
            );
            assert_eq!(
                texel_layout(format, [WIDTH, HEIGHT]).extent(),
                bytes.len(),
                "{}",
                format.slug
            );
            assert!(seen.insert(bytes), "{}", format.slug);
        }
        let (halves, seconds) = (half_bytes(), second_bytes());
        assert_eq!(halves.len(), COUNT * 8);
        assert_eq!(seconds.len(), halves.len());
        assert_eq!(seconds[..8], halves[halves.len() - 8..]);
    }
}
