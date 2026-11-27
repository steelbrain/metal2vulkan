//! **A pipeline that names a colour attachment between two its pass attaches**: the pipeline
//! states colour 0 as `BGRA8Unorm`, colour 1 as `RG16Float` and colour 2 as `RGBA8Unorm`, and one
//! pass attaches colours 0 and 2 alone, leaving colour 1's descriptor unset so its texture is nil.
//! [`crate::cases::pass_unattached_colors`] is the shape whose missing colour is the last.
//!
//! The command buffer draws the same triangle twice with the same state: first into a pass that
//! attaches all three colours (the reference), then into the pass that lacks colour 1. Each pass
//! clears its attachments to the same colours and confines its draw to the target's top half
//! (`DRAWN`), so each plane holds what the fragment wrote beside the clear. The case reads both
//! passes' colours 0 and 2, the reference's colour 1, and the command buffer's status and error.
//! Where the lacking pass's two planes equal the reference's byte for byte, the colour the pass
//! lacks moved nothing in the other two.
//!
//! The fixture is the owned `color_groups` (`cg_cover`'s fullscreen triangle, whose varying `tint`
//! `cg_five` writes to colours 0 to 4 in five swizzles). Colours 3 and 4 have no format in the
//! pipeline, which every pass of this case shares, so they are no part of what the two passes'
//! comparison asks.
//!
//! # Why this case is safe
//!
//! One command buffer: two passes over 8 by 8 targets, one draw of three vertices each, no loop in
//! either stage, the scissor rectangle inside the attachments, every vertex inside the clip volume.

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

/// The owned fixture: the fullscreen triangle and the fragment that writes colours 0 to 4.
const FIXTURE: &str = "color_groups";

/// `cg_cover`: the fullscreen triangle, its varying `tint` of the vertex's corner.
const VERTEX: &str = "cg_cover";

/// `cg_five`: `tint` and four swizzles of it into colours 0 to 4.
const FRAGMENT: &str = "cg_five";

/// The case's name.
pub const NAME: &str = "pass-pipeline-color-unattached-between";

/// The side of every target, in texels.
const SIDE: u32 = 8;

/// The rectangle each draw is confined to: the targets' top half.
const DRAWN: ScissorBox = ScissorBox::new(0, 0, SIDE as u64, (SIDE / 2) as u64);

/// The pipeline's three colour formats, by index. Colour 1 is the one the lacking pass leaves
/// unattached.
const FORMATS: [PixelFormat; 3] = [
    PixelFormat::BGRA8Unorm,
    PixelFormat::RG16Float,
    PixelFormat::RGBA8Unorm,
];

/// The clear each colour takes in either pass, as `(r, g, b, a)`: none of them a value `tint`'s
/// swizzles take at a drawn texel's centre, whose alpha-place channel is 0.25 or a corner's
/// interpolant.
const CLEARS: [[f64; 4]; 3] = [
    [3.0 / 255.0, 5.0 / 255.0, 7.0 / 255.0, 250.0 / 255.0],
    [0.125, 0.375, 0.0, 1.0],
    [11.0 / 255.0, 13.0 / 255.0, 17.0 / 255.0, 240.0 / 255.0],
];

/// Every colour's texel is 4 bytes: `BGRA8Unorm`, `RG16Float` and `RGBA8Unorm` alike.
const TEXEL: u32 = 4;

/// The colours the lacking pass attaches.
const ATTACHED: [usize; 2] = [0, 2];

/// The pipeline's three colour states.
fn color_states() -> [Option<ColorState>; 3] {
    FORMATS.map(|format| Some(ColorState::new(format)))
}

/// The pipeline over `states`.
fn pipeline_spec(states: &[Option<ColorState>]) -> RenderPipelineSpec<'_> {
    RenderPipelineSpec::new(VERTEX, FRAGMENT, states)
}

/// Colour `which`'s target: shared, so its level 0 is read back by the host.
fn target(which: usize) -> TargetSpec {
    TargetSpec::plain(FORMATS[which], [SIDE; 2]).shared()
}

/// A pass over `textures`, by colour index (`None` leaves the index unset), each cleared to its
/// own colour and stored, and one triangle confined to [`DRAWN`].
fn pass(textures: &[Option<Handle>], pipeline: Handle) -> RenderPass {
    let colors = (textures.iter().enumerate())
        .map(|(which, texture)| match texture {
            Some(texture) => PassColor::cleared(*texture, CLEARS[which]),
            None => PassColor::none(),
        })
        .collect();
    RenderPass::clearing(colors).drawing(vec![
        RenderStep::Pipeline(pipeline),
        RenderStep::Scissor(DRAWN),
        RenderStep::Draw {
            primitive: PrimitiveType::Triangle,
            first: 0,
            count: 3,
            instances: 1,
        },
    ])
}

/// The reference pass's attachments: all three colours.
fn every(textures: [Handle; 3]) -> Vec<Option<Handle>> {
    textures.map(Some).to_vec()
}

/// The lacking pass's attachments: colours 0 and 2, colour 1 unset.
fn lacking(textures: [Handle; 3]) -> Vec<Option<Handle>> {
    vec![Some(textures[0]), None, Some(textures[2])]
}

/// The layout a colour's level 0 is read under: one byte an element.
fn layout() -> ByteLayout {
    let row_pitch = TEXEL * SIDE;
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
pub struct PassUnattachedBetween;

impl Case for PassUnattachedBetween {
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
        let mut inputs = vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(held.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(held.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("render-pipeline", &pipeline_spec(&states).words()),
            Input::bytes("pass:every", &pass(&every([zero; 3]), zero).words()),
            Input::bytes("pass:lacking", &pass(&lacking([zero; 3]), zero).words()),
        ];
        for which in 0..FORMATS.len() {
            inputs.push(Input::bytes(
                format!("target:{which}"),
                &target(which).words(),
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
        let states = color_states();
        let pipeline = made(
            sites,
            "render-pipeline",
            side.render_pipeline(library, &pipeline_spec(&states)),
        )?;
        // Each pass's own three targets, so neither pass's planes hold the other's draw.
        let mut targets = [[Handle(0); 3]; 2];
        for (at, role) in ["every", "lacking"].into_iter().enumerate() {
            for which in 0..FORMATS.len() {
                targets[at][which] = made(
                    sites,
                    &format!("target:{role}:{which}"),
                    side.target(&target(which)),
                )?;
            }
        }
        let outcome = call(
            sites,
            "render",
            side.render(&[
                RenderWork::Pass(pass(&every(targets[0]), pipeline)),
                RenderWork::Pass(pass(&lacking(targets[1]), pipeline)),
            ]),
            |outcome| outcome.status.clone(),
        )?;
        let mut drawn = Node::branch("drawn", Vec::new());
        for (at, role) in ["every", "lacking"].into_iter().enumerate() {
            drawn = drawn.with(Node::leaf(
                format!("pass:{role}"),
                Observed::value(match outcome.refusal(at) {
                    Some(failure) => failure.recorded(),
                    None => Value::ok(Value::Absent),
                }),
            ));
        }
        if let Some((at, failure)) = (0..2).find_map(|at| outcome.refusal(at).map(|f| (at, f))) {
            sites.push(drawn);
            return Err(Abort {
                at: format!("drawn/pass:{}", ["every", "lacking"][at]),
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
        let reads: [(&str, usize, &[usize]); 2] =
            [("every", 0, &[0, 1, 2]), ("lacking", 1, &ATTACHED)];
        for (role, at, colors) in reads {
            for &which in colors {
                let bytes = call(
                    sites,
                    &format!("read-back/{role}/{which}"),
                    side.read_texture(targets[at][which], [SIDE; 2], TEXEL * SIDE),
                    |_| Value::Absent,
                )?;
                drawn = drawn.with(Node::leaf(
                    format!("{role}:color{which}"),
                    Observed::bytes(bytes, layout()),
                ));
            }
        }
        sites.push(drawn);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The pipeline names three colours, the reference pass attaches all three and the lacking
    /// pass colours 0 and 2**, which is the whole shape.
    #[test]
    fn the_lacking_pass_leaves_the_middle_colour_unset() {
        let states = color_states();
        assert!(states.iter().all(Option::is_some));
        let handles = [Handle(1), Handle(2), Handle(3)];
        let reference = pass(&every(handles), Handle(0));
        assert!(reference.colors.iter().all(|c| c.texture.is_some()));
        let built = pass(&lacking(handles), Handle(0));
        let attached: Vec<bool> = built.colors.iter().map(|c| c.texture.is_some()).collect();
        assert_eq!(attached, vec![true, false, true]);
        assert!(built.depth.is_none() && built.stencil.is_none());
        assert!(!DRAWN.past([SIDE; 2]));
        assert!(DRAWN.height < u64::from(SIDE));
        assert_eq!(ATTACHED, [0, 2]);
    }
}
