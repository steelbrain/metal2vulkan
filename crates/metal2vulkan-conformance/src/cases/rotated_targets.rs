//! `ARGB8Unorm` render targets against `RGBA8Unorm` ones: the bytes Metal stores, masked and
//! blended.
//!
//! # What the lane this ports was actually checking
//!
//! The runtime lane `rotated_targets` says Metal stores, masks and blends `ARGB8Unorm` as
//! `RGBA8Unorm` with the bytes rotated, alpha first. It then checks that claim by drawing into an
//! `ARGB8Unorm` target **of ours** and comparing it with an `RGBA8Unorm` target **of ours**, rotated
//! — so its expectation is our own arithmetic on both sides of the equals sign. A blend of ours that
//! stored the wrong code in *both* formats passes it. The rotation rule came from a Metal probe, but
//! the stored codes never did.
//!
//! So this port does not carry the rule. It asks Metal the question directly: the same destination
//! texels and the same drawn values into a target of each format, and **both formats' bytes
//! recorded from Metal**. The recording then holds Metal's own `ARGB8Unorm` codes, our side is
//! compared against those rather than against itself, and the rotation claim is a relation between
//! two recorded Metal facts instead of an assertion in a doc comment.
//!
//! # What this measured on the oracle
//!
//! Metal renders format 84 on this M3 Max: the pipeline is made, the pass completes, and the texels
//! read back. Over 64 texels of pseudo-random destinations and tie-heavy drawn values, unblended
//! and under straight alpha, every `ARGB8Unorm` texel held the `RGBA8Unorm` texel's bytes rotated,
//! byte for byte (`probes/rotated_tiles_and_argb.m`). The cases below widen that to the lane's
//! thirteen write-mask and blend states over 8,192 texels and record both legs.
//!
//! # The shape
//!
//! One case per write-mask-and-blend state. Each makes a 128 x 64 target of each format in shared
//! storage — shared because the destination is **arbitrary raw codes**, which reach a texture only
//! through `replaceRegion:` — puts the destination in (rotated for `ARGB8Unorm`, so both formats
//! start from the same texel values), draws the owned fixture `rt6_samples`'s full-screen
//! `float4` fill over the case's outputs under `Load`, and blits the texels out. Both read-backs are
//! sites, so a divergence names the format whose bytes moved.
//!
//! The one case Metal answers and we refuse is kept apart ([`destination_alpha`]): a blend reading
//! Metal's destination alpha in a *colour* factor, which no Vulkan factor gives on the rotated
//! image. It records Metal running the draw against our typed `NotImplemented(Format)`, so the
//! refusal is a row of the ledger rather than a tolerance.
//!
//! Nothing here can fault a GPU: single-sampled targets of the case's own extent, one full-screen
//! triangle a pass, no loop, and every fetch inside its buffer.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Case, Input, MetalSource, Side};
use crate::recording::ShaderRef;
use crate::render::{
    Blend, BlendColorChannels, ColorState, PassColor, Plane, ReadAt, RenderPass,
    RenderPipelineSpec, RenderStep, RenderWork, TargetSpec,
};
use crate::transcript::{Node, Observed, Sites, Value};
use metal2vulkan::metal::{BlendFactor, BlendOperation, ColorWriteMask, PixelFormat};
use metal2vulkan_runtime::Degradation;

use super::extended_range_targets::{
    fixture, fixture_inputs, not_read, program, shader_refs, shape, VERTEX,
};

/// The target's extent, the one `extended_range_blends` uses, so a reader comparing the two
/// families reads one shape. 8,192 texels of each format is 64 KiB of recording a case.
const WIDTH: u32 = 128;
const HEIGHT: u32 = 64;
/// The texels one draw covers.
const COUNT: usize = (WIDTH * HEIGHT) as usize;
/// Four bytes a texel, in both formats.
const TEXEL: u32 = 4;

/// The owned fixture whose full-screen fill the draws use.
const FILLS: &str = "rt6_samples";
/// The fragment that writes one `float4` per texel out of its buffer, so a drawn value is exactly
/// the code the host put there.
const FILL_FLOAT: &str = "rt6_fill_rgba32float";

/// The blend colour every pass sets, the lane's own, which a `BlendAlpha` factor reads.
const BLEND_COLOR: [f32; 4] = [0.2, 0.4, 0.6, 0.8];

/// The two formats each case draws into: Metal's `RGBA8Unorm` and the `ARGB8Unorm` that is said to
/// hold its bytes rotated.
const FORMATS: [(&str, PixelFormat); 2] = [
    ("rgba8", PixelFormat::RGBA8Unorm),
    ("argb8", PixelFormat::ARGB8Unorm),
];

/// The lane's xorshift64*: it stores the shifted state and multiplies without storing the product.
/// Written out so the values are the lane's own rather than a library's.
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

/// The drawn `float4`s, from the lane's seed 85: each channel a code's value (6 in 10), a tie at
/// half a code (2 in 10), a value in `[0, 1]` off the codes (1 in 10), or one of -0.25, 1.25, -0
/// and 1 (1 in 10).
///
/// The ties are the point. Half a code is exactly where a store's rounding rule shows, and a value
/// outside `[0, 1]` is where its clamp shows, so a format that converted differently from
/// `RGBA8Unorm` would differ here rather than agreeing by luck.
fn outputs() -> Vec<[f32; 4]> {
    let mut generator = Rng::new(85);
    let channel = |generator: &mut Rng| -> f32 {
        match generator.below(10) {
            0..=5 => generator.below(256) as f32 / 255.0,
            6 | 7 => (generator.below(255) as f32 + 0.5) / 255.0,
            8 => generator.below(1 << 24) as f32 / (1 << 24) as f32,
            _ => [-0.25, 1.25, -0.0, 1.0][generator.below(4) as usize],
        }
    };
    (0..COUNT)
        .map(|_| [(); 4].map(|_| channel(&mut generator)))
        .collect()
}

/// The drawn values' bytes, as the fragment's buffer holds them.
fn float_bytes() -> Vec<u8> {
    outputs()
        .iter()
        .flat_map(|texel| texel.iter().flat_map(|channel| channel.to_le_bytes()))
        .collect()
}

/// The `RGBA8Unorm` texels each target holds before its draw, from the lane's seed 8585.
fn destinations() -> Vec<[u8; 4]> {
    let mut generator = Rng::new(8585);
    (0..COUNT)
        .map(|_| [(); 4].map(|_| generator.below(256) as u8))
        .collect()
}

/// `ARGB8Unorm`'s bytes of an `RGBA8Unorm` texel: alpha first.
///
/// This is the relation under test. It is applied to the **destination** so both formats' targets
/// start from the same texel values; it is never applied to anything Metal answered.
fn rotated([red, green, blue, alpha]: [u8; 4]) -> [u8; 4] {
    [alpha, red, green, blue]
}

/// The destination bytes a target of `format` is loaded with.
fn destination_bytes(format: PixelFormat) -> Vec<u8> {
    let turn = format == PixelFormat::ARGB8Unorm;
    destinations()
        .iter()
        .flat_map(|texel| match turn {
            true => rotated(*texel),
            false => *texel,
        })
        .collect()
}

/// One case: its case name, the state's own short name, the write mask, and the blend (`None`:
/// off) as the colour and alpha factors and operations.
///
/// Both names are literals rather than built from each other, so a recording's file name is
/// something a reader can grep for in this file.
struct State {
    case: &'static str,
    label: &'static str,
    mask: ColorWriteMask,
    blend: Option<[(BlendFactor, BlendFactor, BlendOperation); 2]>,
}

/// The thirteen states the lane sweeps: every write mask and blend it measured, in its order.
fn states() -> Vec<State> {
    use BlendFactor as F;
    use BlendOperation as O;
    const ALL: ColorWriteMask = ColorWriteMask::ALL;
    let state = |case, label, mask, blend| State {
        case,
        label,
        mask,
        blend,
    };
    vec![
        state("argb8-targets-unblended", "unblended", ALL, None),
        state(
            "argb8-targets-unblended-red-alpha",
            "unblended-red-alpha",
            ColorWriteMask::RED.union(ColorWriteMask::ALPHA),
            None,
        ),
        state(
            "argb8-targets-multiply",
            "multiply",
            ALL,
            Some([
                (F::DestinationColor, F::Zero, O::Add),
                (F::DestinationAlpha, F::Zero, O::Add),
            ]),
        ),
        state(
            "argb8-targets-source-color",
            "source-color",
            ALL,
            Some([
                (F::SourceColor, F::OneMinusSourceColor, O::Add),
                (F::SourceAlpha, F::OneMinusSourceAlpha, O::Add),
            ]),
        ),
        state(
            "argb8-targets-blend-alpha",
            "blend-alpha",
            ALL,
            Some([
                (F::BlendAlpha, F::OneMinusBlendAlpha, O::Add),
                (F::BlendAlpha, F::OneMinusBlendAlpha, O::Add),
            ]),
        ),
        state(
            "argb8-targets-max",
            "max",
            ALL,
            Some([(F::One, F::One, O::Max), (F::One, F::One, O::Max)]),
        ),
        state(
            "argb8-targets-straight",
            "straight",
            ALL,
            Some([
                (F::SourceAlpha, F::OneMinusSourceAlpha, O::Add),
                (F::SourceAlpha, F::OneMinusSourceAlpha, O::Add),
            ]),
        ),
        state(
            "argb8-targets-straight-alpha-one",
            "straight-alpha-one",
            ALL,
            Some([
                (F::SourceAlpha, F::OneMinusSourceAlpha, O::Add),
                (F::One, F::OneMinusSourceAlpha, O::Add),
            ]),
        ),
        state(
            "argb8-targets-straight-red-green-alpha",
            "straight-red-green-alpha",
            ColorWriteMask::RED
                .union(ColorWriteMask::GREEN)
                .union(ColorWriteMask::ALPHA),
            Some([
                (F::SourceAlpha, F::OneMinusSourceAlpha, O::Add),
                (F::One, F::OneMinusSourceAlpha, O::Add),
            ]),
        ),
        state(
            "argb8-targets-premultiplied",
            "premultiplied",
            ALL,
            Some([
                (F::One, F::OneMinusSourceAlpha, O::Add),
                (F::One, F::OneMinusSourceAlpha, O::Add),
            ]),
        ),
        state(
            "argb8-targets-additive-alpha",
            "additive-alpha",
            ALL,
            Some([(F::SourceAlpha, F::One, O::Add), (F::Zero, F::One, O::Add)]),
        ),
        state(
            "argb8-targets-reverse-subtract-alpha",
            "reverse-subtract-alpha",
            ALL,
            Some([
                (F::SourceAlpha, F::One, O::ReverseSubtract),
                (F::SourceAlpha, F::One, O::ReverseSubtract),
            ]),
        ),
        state(
            "argb8-targets-subtract-one-minus-alpha",
            "subtract-one-minus-alpha",
            ALL,
            Some([
                (F::One, F::OneMinusSourceAlpha, O::Subtract),
                (F::One, F::OneMinusSourceAlpha, O::Subtract),
            ]),
        ),
    ]
}

/// A case: one write-mask-and-blend state, drawn into a target of each format.
pub struct RotatedTarget {
    state: State,
}

/// Every state the lane sweeps, and the one blend Metal answers that we refuse.
pub fn all() -> Vec<Box<dyn Case>> {
    let mut cases: Vec<Box<dyn Case>> = Vec::new();
    for state in states() {
        cases.push(Box::new(RotatedTarget { state }));
    }
    cases.push(Box::new(destination_alpha()));
    cases
}

impl RotatedTarget {
    /// The blend the state names, as the render vocabulary states one.
    fn blend(&self) -> Option<Blend> {
        self.state
            .blend
            .map(|[(sr, dr, rgb), (sa, da, alpha)]| Blend {
                source_rgb: sr,
                destination_rgb: dr,
                rgb,
                source_alpha: sa,
                destination_alpha: da,
                alpha,
            })
    }

    /// The attachment state a target of `format` is drawn into.
    fn colors(&self, format: PixelFormat) -> [Option<ColorState>; 1] {
        let mut state = ColorState::new(format).masked(self.state.mask);
        if let Some(blend) = self.blend() {
            state = state.blended(blend);
        }
        [Some(state)]
    }

    /// The target: shared, because its destination codes go in through `replaceRegion:`.
    fn target(format: PixelFormat) -> TargetSpec {
        TargetSpec::read(format, [WIDTH, HEIGHT]).shared()
    }
}

/// The buffers and library every format's leg shares, made once before either leg runs.
struct Shared {
    library: crate::case::Handle,
    /// The drawn `float4`s, at fragment buffer 0.
    values: crate::case::Handle,
    /// The fixture's extent, at fragment buffer 1.
    extent: crate::case::Handle,
}

/// The sites one format's leg emits, and the bytes it read back.
fn draw_into(
    side: &mut dyn Side,
    sites: &mut Sites,
    label: &str,
    shared: &Shared,
    colors: &[Option<ColorState>],
    format: PixelFormat,
) -> Result<(), Abort> {
    let Shared {
        library,
        values,
        extent,
    } = *shared;
    let pipeline = made(
        sites,
        &format!("pipeline/{label}"),
        side.render_pipeline(
            library,
            &RenderPipelineSpec::new(VERTEX, FILL_FLOAT, colors),
        ),
    )?;
    let target = made(
        sites,
        &format!("target/{label}"),
        side.target(&RotatedTarget::target(format)),
    )?;
    call(
        sites,
        &format!("replace-region/{label}"),
        side.replace_region(
            target,
            [WIDTH, HEIGHT],
            WIDTH * TEXEL,
            &destination_bytes(format),
        ),
        |_| Value::Absent,
    )?;
    let pass = RenderWork::Pass(
        RenderPass::clearing(vec![PassColor::loaded(target)]).drawing(vec![
            RenderStep::Pipeline(pipeline),
            RenderStep::BlendColor(BlendColorChannels::new(BLEND_COLOR)),
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
    let texel_bytes = (WIDTH * HEIGHT * TEXEL) as usize;
    let into = made(
        sites,
        &format!("buffer/texels/{label}"),
        side.zeroed_buffer(texel_bytes),
    )?;
    let work = vec![
        pass,
        RenderWork::ReadTexture {
            texture: target,
            into,
            offset: 0,
            row_pitch: WIDTH * TEXEL,
            size: [WIDTH, HEIGHT],
            plane: Plane::Whole,
            at: ReadAt::BASE,
        },
    ];
    let outcome = call(
        sites,
        &format!("render/{label}"),
        side.render(&work),
        |_| Value::Absent,
    )?;
    program(sites, &format!("program/{label}"), &outcome, &work);
    let stored = call(
        sites,
        &format!("read-back/{label}"),
        side.read_buffer(into, 0, texel_bytes),
        |_| Value::Absent,
    )?;
    sites.push(Node::leaf(
        format!("texels/{label}"),
        Observed::bytes(stored, layout()),
    ));
    Ok(())
}

/// The layout both formats' texels are read under: one row of the layout per row of texels, so a
/// diff names the texel by its position and prints the bytes it stored.
fn layout() -> crate::transcript::ByteLayout {
    use crate::transcript::{Aspect, ByteLayout, Element, Gaps, Region};
    let pitch = WIDTH * TEXEL;
    ByteLayout {
        region: Region::Texture {
            level: 0,
            slice: 0,
            aspect: Aspect::Color,
            sample: 0,
            origin: [0, 0, 0],
            size: [WIDTH, HEIGHT, 1],
        },
        element: Element::U8,
        row_offset: 0,
        row_elements: pitch,
        row_pitch: pitch,
        rows: HEIGHT,
        image_pitch: pitch * HEIGHT,
        images: 1,
        // The rows are packed, so the sites cover every byte and there is no gap.
        gaps: Gaps::Ignored,
    }
}

impl Case for RotatedTarget {
    fn name(&self) -> &'static str {
        self.state.case
    }

    fn metal_source(&self) -> MetalSource {
        // Nothing here is Apple SPI. Format 84 has no public header name, but a pixel format is a
        // raw value in `MTLTextureDescriptor` and in a pipeline's colour attachment, and Metal
        // makes, draws into, blends and reads back a target of it through entirely public calls --
        // measured on this chip before the case was written.
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // One allowance, and it loosens **no texel**: it names the notice this venue announces.
        //
        // Metal has no notices at all, so a recording's notice set is empty and any notice of ours
        // has to be declared by its code or the case cannot be read on the venue that announces it.
        // Which code depends on the state, not on the device: a state that blends keeps the
        // device's own conversion of the channels it blends
        // (`PrecisionNote::DeviceColorConversion`), and an unblended one has the fragment round
        // each output by Metal's rule itself (`ShaderColorConversion`). So the case names the one
        // its own blend implies, and a venue that announces neither (MoltenVK on this M3 Max, where
        // every byte of both legs matched) leaves it unfired, which the report says and the suite
        // documents as a venue-specific allowance doing its job.
        //
        // **No tolerance on any texel, deliberately.** On lavapipe the blended states store a code
        // one apart from Metal's, and that is exactly the shape `DeviceColorConversion` describes
        // -- but the note's own definition ends "validation derives no tolerance and no explanation
        // from it: a difference stays a mismatch for review". Loosening a texel here would be this
        // suite overruling that, so the twelve blended states stay divergent on lavapipe and are a
        // per-venue divergence to carry, not a tolerance to write. The two unblended states, whose
        // texels match bit for bit there, read announced.
        // Beside it, the notice the venue's own memory model provokes: the buffer each case
        // reads its target back through is `Shared`, which a device with no unified memory keeps
        // in system memory because that is the only memory both sides can reach.
        vec![
            crate::replay::notice_allowance(match self.state.blend {
                None => Degradation::ShaderColorConversion.code(),
                Some(_) => Degradation::DeviceColorConversion.code(),
            }),
            crate::replay::shared_memory_notice(),
        ]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs(FILLS, &[VERTEX, FILL_FLOAT])
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs("fills", FILLS)?;
        let colors = self.colors(PixelFormat::RGBA8Unorm);
        inputs.extend([
            // The state is the question the case asks, so the mask and the blend are inputs.
            Input::bytes("write-mask", &self.state.mask.as_raw().to_le_bytes()),
            Input::bytes("state-name", self.state.label.as_bytes()),
            Input::bytes(
                "pipeline",
                &RenderPipelineSpec::new(VERTEX, FILL_FLOAT, &colors).words(),
            ),
            Input::bytes("blend-color", &BlendColorChannels::new(BLEND_COLOR).words()),
            Input::bytes("buffer:values", &float_bytes()),
            Input::bytes("buffer:shape", &shape(WIDTH, 1)),
        ]);
        for (label, format) in FORMATS {
            inputs.extend([
                Input::bytes(
                    format!("target:{label}"),
                    &RotatedTarget::target(format).words(),
                ),
                // The destination the blend and the mask read, which is the point of the family.
                Input::bytes(format!("destination:{label}"), &destination_bytes(format)),
            ]);
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let fills = fixture(FILLS, VERTEX)
            .loaded()
            .map_err(|why| not_read("library/fills", why))?;
        let library = made(sites, "library/fills", side.library(&fills.source()))?;
        let values = made(sites, "buffer/values", side.buffer(&float_bytes()))?;
        let extent = made(sites, "buffer/shape", side.buffer(&shape(WIDTH, 1)))?;
        let shared = Shared {
            library,
            values,
            extent,
        };
        for (label, format) in FORMATS {
            let colors = self.colors(format);
            draw_into(side, sites, label, &shared, &colors, format)?;
        }
        Ok(())
    }
}

// ---- The blend Metal answers and we refuse ------------------------------------------------------

/// The case for a blend that reads Metal's **destination alpha in a colour factor** of an
/// `ARGB8Unorm` target.
///
/// Our runtime holds the format in an `R8G8B8A8_UNORM` image of the rotated bytes, so the image's
/// alpha channel is Metal's blue. A colour factor that reads Metal's destination alpha would have
/// to read the image's *red*, and no Vulkan blend factor does, so the pipeline is refused
/// `NotImplemented(Format)`. Metal runs the draw.
///
/// It is its own case so the refusal is one row of the known-refusals ledger at one site, rather
/// than a tolerance folded into the thirteen states. Metal's leg records normally: the pipeline and
/// the draw are public calls, and the case reads back what Metal stored, so if the gap is ever
/// closed the recording is already there to compare against.
pub fn destination_alpha() -> RotatedTarget {
    RotatedTarget {
        state: State {
            case: "argb8-targets-destination-alpha",
            label: "destination-alpha",
            mask: ColorWriteMask::ALL,
            blend: Some([
                (
                    BlendFactor::DestinationAlpha,
                    BlendFactor::OneMinusDestinationAlpha,
                    BlendOperation::Add,
                ),
                (
                    BlendFactor::SourceAlpha,
                    BlendFactor::OneMinusSourceAlpha,
                    BlendOperation::Add,
                ),
            ]),
        },
    }
}
