//! **A pipeline that names a colour attachment its pass does not attach**: the pipeline states
//! colour 0 as `BGRA8Unorm` and colour 1 as `RG16Float`, its fragment function writes both, and the
//! pass attaches colour 0 alone, leaving colour 1's descriptor unset so its texture is nil.
//!
//! The case asks what colour 0 holds after one draw, and what the command buffer answers. It reads
//! colour 0's whole level 0 and the command buffer's status and error; colour 1 has no texture, so
//! nothing of it can be read.
//!
//! The draw is confined to the target's **top half** (`DRAWN`), so the plane holds two regions a
//! reader can tell apart: what the fragment wrote where it ran, and the pass's clear where it did
//! not. A side that drops the draw reads as the clear in both.
//!
//! Every value colour 0's buffer holds is a code of 255 (`k / 255` as a `float`), which
//! `BGRA8Unorm` stores as the code `k` under any conversion within half a code, and the four
//! channels sit in four disjoint bands, so a swapped channel and a texel read from another position
//! are told apart by value. The clear's codes are in none of the drawn bands.
//!
//! The fixture is the owned `rt6_pair` (`rt6_pair_two`: colour 0 from fragment buffer 0, colour 1
//! from fragment buffer 2, each at the fragment's own texel).
//!
//! # Why this case is safe
//!
//! One command buffer: one pass over an 8 by 8 target, one draw of three vertices, no loop in
//! either stage, the scissor rectangle inside the attachment, every vertex inside the clip volume.

use crate::case::{call, made, Abort, Case, Failure, Handle, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, PassColor, RenderPass, RenderPipelineSpec, RenderStep, RenderWork, ScissorBox,
    TargetSpec,
};
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::{PixelFormat, PrimitiveType};

/// The owned fixture: the fullscreen triangle and the fragment that writes both colours.
const FIXTURE: &str = "rt6_pair";

/// `rt6_pair_vertex`: `float4(4 * a - 1, 0.5, 1)` at each of three vertex ids.
const VERTEX: &str = "rt6_pair_vertex";

/// `rt6_pair_two`: colour 0 from fragment buffer 0 and colour 1 from fragment buffer 2.
const FRAGMENT: &str = "rt6_pair_two";

/// The case's name.
pub const NAME: &str = "pass-pipeline-color-unattached";

/// The side of the target, in texels.
const SIDE: u32 = 8;

/// The rectangle the draw is confined to: the target's top half.
const DRAWN: ScissorBox = ScissorBox::new(0, 0, SIDE as u64, (SIDE / 2) as u64);

/// Colour 0's format, the pass's one attachment.
const ATTACHED: PixelFormat = PixelFormat::BGRA8Unorm;

/// Colour 1's format, which the pipeline states and the pass attaches nothing for.
const UNATTACHED: PixelFormat = PixelFormat::RG16Float;

/// The clear colour 0 takes, as codes of 255 in `(r, g, b, a)` order: in none of the drawn bands.
const CLEAR: [u32; 4] = [3, 5, 7, 250];

/// The codes the fragment writes into colour 0 at texel `at`, in `(r, g, b, a)` order: red in
/// `[64, 127]`, green in `[128, 191]`, blue in `[192, 255]` and alpha in `[0, 63]`.
fn drawn_codes(at: u32) -> [u32; 4] {
    [64 + at, 128 + at, 192 + at, at]
}

/// The clear as Metal's `double` clear colour.
fn clear_color() -> [f64; 4] {
    CLEAR.map(|code| f64::from(code) / 255.0)
}

/// The bytes of fragment buffer 0: one `float4` a texel, row-major (`y * width + x`), each channel
/// `k / 255` for the code [`drawn_codes`] states.
fn color0_bytes() -> Vec<u8> {
    (0..SIDE * SIDE)
        .flat_map(|at| {
            drawn_codes(at)
                .into_iter()
                .flat_map(|code| (code as f32 / 255.0).to_le_bytes())
        })
        .collect()
}

/// The bytes of fragment buffer 2: what the fragment returns for colour 1, which has no texture.
fn color1_bytes() -> Vec<u8> {
    (0..SIDE * SIDE)
        .flat_map(|_| {
            [0.5f32, 0.25, 0.0, 1.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
        })
        .collect()
}

/// The bytes of the `uint` the fixture reads its row width from, at fragment buffer index 1.
fn width_bytes() -> Vec<u8> {
    SIDE.to_le_bytes().to_vec()
}

/// The pipeline's two colour states: colour 0 and colour 1, both stated.
fn color_states() -> [Option<ColorState>; 2] {
    [
        Some(ColorState::new(ATTACHED)),
        Some(ColorState::new(UNATTACHED)),
    ]
}

/// The pipeline over `states`.
fn pipeline_spec(states: &[Option<ColorState>]) -> RenderPipelineSpec<'_> {
    RenderPipelineSpec::new(VERTEX, FRAGMENT, states)
}

/// Colour 0's target: shared, so its level 0 is read back by the host.
fn target() -> TargetSpec {
    TargetSpec::plain(ATTACHED, [SIDE; 2]).shared()
}

/// The pass: colour 0 cleared and stored, colour 1 not stated at all, and one triangle confined to
/// [`DRAWN`].
fn pass(texture: Handle, pipeline: Handle, buffers: [Handle; 3]) -> RenderPass {
    RenderPass::clearing(vec![PassColor::cleared(texture, clear_color())]).drawing(vec![
        RenderStep::Pipeline(pipeline),
        RenderStep::fragment_buffer(buffers[0], 0),
        RenderStep::fragment_buffer(buffers[1], 1),
        RenderStep::fragment_buffer(buffers[2], 2),
        RenderStep::Scissor(DRAWN),
        RenderStep::Draw {
            primitive: PrimitiveType::Triangle,
            first: 0,
            count: 3,
            instances: 1,
        },
    ])
}

/// The layout colour 0's level 0 is read under: one byte an element.
fn layout() -> ByteLayout {
    let row_pitch = 4 * SIDE;
    ByteLayout {
        region: Region::Texture {
            level: 0,
            slice: 0,
            aspect: Aspect::Color,
            sample: 0,
            origin: [0, 0, 0],
            size: [SIDE, SIDE, 1],
        },
        element: Element::U8,
        row_offset: 0,
        row_elements: row_pitch,
        row_pitch,
        rows: SIDE,
        image_pitch: row_pitch * SIDE,
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// The family's one case.
pub struct PassUnattachedColor;

impl Case for PassUnattachedColor {
    fn name(&self) -> &'static str {
        NAME
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let held = Fixture::new(FIXTURE);
        [RUNTIME_TEXT, METAL_TEXT]
            .into_iter()
            .map(|extension| {
                Ok(ShaderRef {
                    fixture: format!("{FIXTURE}{extension}"),
                    entry: format!("{VERTEX}+{FRAGMENT}"),
                    source_sha256: crate::sha256_hex(held.text(extension)?.as_bytes()),
                })
            })
            .collect()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let held = Fixture::new(FIXTURE);
        let states = color_states();
        let zero = Handle(0);
        Ok(vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(held.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(held.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("buffer:color0", &color0_bytes()),
            Input::bytes("buffer:color1", &color1_bytes()),
            Input::bytes("buffer:width", &width_bytes()),
            Input::bytes("render-pipeline", &pipeline_spec(&states).words()),
            Input::bytes("target", &target().words()),
            Input::bytes("pass", &pass(zero, zero, [zero; 3]).words()),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = Fixture::new(FIXTURE).loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let states = color_states();
        let pipeline = made(
            sites,
            "render-pipeline",
            side.render_pipeline(library, &pipeline_spec(&states)),
        )?;
        let texture = made(sites, "target", side.target(&target()))?;
        let buffers = [
            made(sites, "buffer:color0", side.buffer(&color0_bytes()))?,
            made(sites, "buffer:width", side.buffer(&width_bytes()))?,
            made(sites, "buffer:color1", side.buffer(&color1_bytes()))?,
        ];
        let outcome = call(
            sites,
            "render",
            side.render(&[RenderWork::Pass(pass(texture, pipeline, buffers))]),
            |outcome| outcome.status.clone(),
        )?;
        let mut drawn = Node::branch("drawn", Vec::new()).with(Node::leaf(
            "pass",
            Observed::value(match outcome.refusal(0) {
                Some(failure) => failure.recorded(),
                None => Value::ok(Value::Absent),
            }),
        ));
        if let Some(failure) = outcome.refusal(0) {
            sites.push(drawn);
            return Err(Abort {
                at: "drawn/pass".to_owned(),
                failure: failure.clone(),
            });
        }
        drawn = drawn.with(Node::leaf(
            "command-buffer-error",
            Observed::value(match &outcome.error {
                Some(error) => Value::Variant {
                    name: "Some".to_owned(),
                    payload: Box::new(error.clone()),
                },
                None => Value::unit_variant("None"),
            }),
        ));
        let bytes = call(
            sites,
            "read-back",
            side.read_texture(texture, [SIDE; 2], 4 * SIDE),
            |_| Value::Absent,
        )?;
        drawn = drawn.with(Node::leaf("color0", Observed::bytes(bytes, layout())));
        sites.push(drawn);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The bytes colour 0 holds at texel `(x, y)` if the draw wrote it, in `BGRA8Unorm`'s byte
    /// order.
    fn drawn_texel(x: u32, y: u32) -> [u8; 4] {
        let [r, g, b, a] = drawn_codes(y * SIDE + x);
        [b as u8, g as u8, r as u8, a as u8]
    }

    /// The bytes colour 0 holds where the pass's clear is left, in `BGRA8Unorm`'s byte order.
    fn clear_texel() -> [u8; 4] {
        let [r, g, b, a] = CLEAR;
        [b as u8, g as u8, r as u8, a as u8]
    }

    /// **Every code colour 0's buffer holds is a code of `BGRA8Unorm` and survives its conversion,
    /// and no drawn code of a channel is the clear's code of that channel.**
    #[test]
    fn every_code_is_exact_and_the_clear_is_apart() {
        for at in 0..SIDE * SIDE {
            for (channel, code) in drawn_codes(at).into_iter().enumerate() {
                assert!(code <= 255, "texel {at} channel {channel}: {code}");
                let value = code as f32 / 255.0;
                // Within a thousandth of a code of `k`, so any conversion within half a code
                // stores `k`.
                assert!((f64::from(value) * 255.0 - f64::from(code)).abs() < 1e-3);
                assert_ne!(code, CLEAR[channel], "texel {at} channel {channel}");
            }
        }
        // Every texel of the drawn rectangle is its own four bytes, so a texel read from another
        // position is caught.
        let texels: BTreeSet<[u8; 4]> = (0..SIDE * SIDE)
            .map(|at| drawn_texel(at % SIDE, at / SIDE))
            .collect();
        assert_eq!(texels.len(), (SIDE * SIDE) as usize);
        assert!(!texels.contains(&clear_texel()));
        assert_eq!(color0_bytes().len(), (16 * SIDE * SIDE) as usize);
        assert_eq!(color1_bytes().len(), (16 * SIDE * SIDE) as usize);
    }

    /// **The pipeline names two colours and the pass attaches one**, which is the whole shape.
    #[test]
    fn the_pipeline_names_a_colour_the_pass_does_not_attach() {
        let states = color_states();
        assert_eq!(states.len(), 2);
        assert!(states.iter().all(Option::is_some));
        let built = pass(Handle(0), Handle(0), [Handle(0); 3]);
        assert_eq!(built.colors.len(), 1);
        assert!(built.colors[0].texture.is_some());
        assert!(built.depth.is_none() && built.stencil.is_none());
        assert!(!DRAWN.past([SIDE; 2]));
        assert!(DRAWN.height < u64::from(SIDE));
    }
}
