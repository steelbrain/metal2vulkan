//! What each draw of a **multi-draw pass** sees of the state set before it.
//!
//! Almost every render case in the suite is one draw in one pass, so the encoder's state is set
//! once and never changed while a pass is open. A program's pass is a list of draws with state
//! changes between them: another pipeline, another binding, another viewport, and then a draw that
//! must see exactly the last thing set and nothing of what came before.
//!
//! This case is one pass of five draws whose state changes between each pair, each draw writing a
//! region of its own, so the target is a record of what every draw saw:
//!
//! | columns | what wrote them | what it says |
//! |---|---|---|
//! | 0..4 | draw 1, then draw 4 under a write mask | the mask of the pipeline the **later** draw set, applied over what the earlier one stored |
//! | 4..8 | draw 2 | a draw that changed its **fragment buffer offset** and nothing else |
//! | 8..10 | draw 3, under a viewport two pixels wide | a draw that changed the **viewport** mid-pass |
//! | 10..12 | nobody | the viewport confined draw 3 rather than merely moving it |
//! | 12..16 | draw 5 | a draw that changed the **pipeline and its texture binding** mid-pass, and whose pipeline's mask is its own rather than the previous pipeline's |
//!
//! Each region is a different value, and none of them is any other's, so a draw that saw the wrong
//! state writes a value this case can name rather than one that happens to pass. The one region
//! nobody writes holds the clear, which is what makes the viewport a real question: a side that
//! took the viewport as a transform without confining the draw would fill that region too.
//!
//! The target is `RGBA32Float`, so what a draw stored is the `float` the fragment function
//! returned and the whole expectation is arithmetic this module states rather than a measurement
//! adopted from the run: `tob_solid` returns the `float4` its buffer holds unchanged, and
//! `tob_sample` samples texture 0 at the pixel's own position over the texture's extent, which for
//! a texture of the target's extent is a texel centre.
//!
//! The write mask is read over the **destination** that another draw of the same pass stored, which
//! is the half of a mask no single-draw case can ask: `color_write_masks` loads a target the host
//! prefilled, so it asks whether the mask reached the pipeline. This asks whether the mask of the
//! pipeline a draw set applies to that draw and to the texels an earlier draw of the same pass
//! left.
//!
//! Nothing here can fault a GPU: one pass, five draws, one 16 by 16 target, no loop.

use crate::allowance::Allowance;
use crate::case::{
    call, made, Abort, Case, Failure, Input, MetalSource, SamplerSpec, Side, TextureRole,
    TextureSpec,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, PassColor, Plane, RenderPass, RenderPipelineSpec, RenderStep, RenderWork,
    TargetSpec, ViewportRect,
};
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::{ColorWriteMask, PixelFormat, PrimitiveType};
use metal2vulkan_runtime::Degradation;

/// The format the target carries: a `float` stored as it is, so every expectation here is
/// arithmetic rather than a rounding rule.
const FORMAT: PixelFormat = PixelFormat::RGBA32Float;

/// Bytes one texel of [`FORMAT`] takes.
const TEXEL: u32 = 16;

/// The target's side, in texels.
const SIDE: u32 = 16;

/// Bytes the target's whole level 0 takes.
const PLANE: u32 = TEXEL * SIDE * SIDE;

/// The bands the draws aim at: four columns each.
const BANDS: u32 = 4;

/// The clear the pass loads the target with. The one region no draw writes holds it, so a draw
/// that was not confined to its viewport is seen rather than inferred.
const CLEAR: f32 = -7.5;

/// The four colours the solid draws read out of one buffer at four offsets, each channel an exact
/// binary fraction.
const COLORS: [[f32; 4]; 4] = [
    [0.25, 0.5, 0.75, 1.0],
    [0.125, 0.375, 0.625, 0.875],
    [2.5, -1.25, 0.0, 1.0],
    [-0.5, 3.75, 0.5, 0.25],
];

/// The mask the fourth draw's pipeline carries: the red and blue channels, so the texels it draws
/// over keep the green and alpha the first draw stored.
const MASK: ColorWriteMask = ColorWriteMask::RED.union(ColorWriteMask::BLUE);

/// The owned fixture of the rectangle, the solid colour and the sample.
const FIXTURE: &str = "tob_renders";

/// `tob_rect`: a rectangle from a buffer's two clip-space corners.
const RECT: &str = "tob_rect";

/// `tob_solid`: the `float4` a buffer holds, unchanged.
const SOLID: &str = "tob_solid";

/// `tob_sample`: texture 0 sampled at the pixel's position over the texture's extent.
const SAMPLE: &str = "tob_sample";

/// The draw a rectangle takes: two triangles of six vertices, one instance.
const RECT_DRAW: RenderStep = RenderStep::Draw {
    primitive: PrimitiveType::Triangle,
    first: 0,
    count: 6,
    instances: 1,
};

/// The three pipelines the pass sets, by index in the order the case makes them.
const UNMASKED: usize = 0;
const MASKED: usize = 1;
const SAMPLING: usize = 2;

/// The colour attachment of the pipeline at `which`.
fn color_state(which: usize) -> [Option<ColorState>; 1] {
    let state = ColorState::new(FORMAT);
    [Some(match which {
        MASKED => state.masked(MASK),
        _ => state,
    })]
}

/// The pipeline at `which`, over `colors`.
fn pipeline(which: usize, colors: &[Option<ColorState>]) -> RenderPipelineSpec<'_> {
    RenderPipelineSpec::new(
        RECT,
        match which {
            SAMPLING => SAMPLE,
            _ => SOLID,
        },
        colors,
    )
}

/// Band `at` of [`BANDS`], as `tob_rect` takes a rectangle.
fn band(at: u32) -> [f32; 4] {
    let edge = |step: u32| 2.0 * step as f32 / BANDS as f32 - 1.0;
    [edge(at), -1.0, edge(at + 1), 1.0]
}

/// The whole target, as a rectangle.
const WHOLE: [f32; 4] = [-1.0, -1.0, 1.0, 1.0];

/// The viewport the third draw sets: two pixel columns of the third band, so the band's other two
/// columns keep the clear.
const NARROW: ViewportRect = ViewportRect {
    origin: [8.0, 0.0],
    size: [2.0, SIDE as f64],
    depth: [0.0, 1.0],
};

/// The viewport the fourth draw sets back: the whole target, which is Metal's own default.
const WIDE: ViewportRect = ViewportRect {
    origin: [0.0, 0.0],
    size: [SIDE as f64, SIDE as f64],
    depth: [0.0, 1.0],
};

/// A rectangle's bytes, as the vertex function's `constant float4 &` binding takes them.
fn rect_bytes(rect: [f32; 4]) -> Vec<u8> {
    rect.iter().flat_map(|edge| edge.to_le_bytes()).collect()
}

/// The buffer the solid draws read their colours out of.
fn colors_buffer() -> Vec<u8> {
    COLORS
        .iter()
        .flat_map(|color| {
            color
                .iter()
                .flat_map(|channel| channel.to_le_bytes())
                .collect::<Vec<u8>>()
        })
        .collect()
}

/// What the sampled texture holds at texel `(x, y)`: values no draw of the pass can write, so a
/// sample that reached the wrong texel, or a region the sampling draw never covered, is named
/// rather than guessed. Each is an integer a `float` holds exactly.
fn sampled_texel(x: u32, y: u32) -> [f32; 4] {
    [8.0 + x as f32, 16.0 + y as f32, 0.5, 2.0]
}

/// The sampled texture's bytes, which the host writes before the pass.
fn sampled_texels() -> Vec<u8> {
    (0..SIDE)
        .flat_map(|y| (0..SIDE).map(move |x| sampled_texel(x, y)))
        .flat_map(|texel| {
            texel
                .iter()
                .flat_map(|channel| channel.to_le_bytes())
                .collect::<Vec<u8>>()
        })
        .collect()
}

/// The sampled texture's spec: shader-read, at the target's extent so a sample lands on a texel
/// centre.
fn sampled_spec() -> TextureSpec {
    TextureSpec::new(FORMAT, [SIDE; 2], TEXEL * SIDE, TextureRole::Read)
}

/// The layout the target's whole level 0 is read under: one row of the layout a row of texels, one
/// element a channel.
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
        element: Element::F32,
        row_offset: 0,
        row_elements: 4 * SIDE,
        row_pitch,
        rows: SIDE,
        image_pitch: row_pitch * SIDE,
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// The case.
pub struct DrawStateChanges;

/// The case the suite holds.
pub fn boxed() -> Vec<Box<dyn Case>> {
    vec![Box::new(DrawStateChanges)]
}

impl Case for DrawStateChanges {
    fn name(&self) -> &'static str {
        "multi-draw-state-changes"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // No texel is loosened: a `RGBA32Float` texel is the `float` a fragment function returned,
        // so what every draw saw is an exact answer or it is a bug. The normalized-conversion
        // notice cannot apply, because nothing here stores a normalized channel.
        // The one entry is a **notice declaration** on the notices site, which loosens no value.
        // Metal announces nothing, so a notice of ours has to be declared or the comparison cannot
        // be made at all, and `device-arithmetic` is the one this case's own functions imply:
        // `tob_sample` divides the pixel's position by the sampled texture's extent, and our side
        // records the note where a float division takes Vulkan's precision bounds rather than
        // correct rounding (`metal2vulkan::pipeline::artifact`'s
        // `PrecisionNote::DeviceArithmetic`). Which device does that is the device's answer; that
        // this case can announce it is the case's own shape.
        //
        // Measured: it fires on lavapipe and not on MoltenVK on this M3 Max, and on lavapipe every
        // texel of this case compared **exactly** -- the whole difference was the announcement.
        vec![crate::replay::notice_allowance(
            Degradation::DeviceArithmetic.code(),
        )]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let fixture = Fixture::new(FIXTURE);
        [RUNTIME_TEXT, METAL_TEXT]
            .into_iter()
            .map(|extension| {
                Ok(ShaderRef {
                    fixture: format!("{FIXTURE}{extension}"),
                    entry: format!("{RECT}+{SOLID}+{SAMPLE}"),
                    source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
                })
            })
            .collect()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = Fixture::new(FIXTURE);
        let mut inputs = vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
        ];
        for which in [UNMASKED, MASKED, SAMPLING] {
            let colors = color_state(which);
            inputs.push(Input::bytes(
                format!("render-pipeline:{which}"),
                &pipeline(which, &colors).words(),
            ));
        }
        inputs.push(Input::bytes(
            "target",
            &TargetSpec::plain(FORMAT, [SIDE; 2]).words(),
        ));
        inputs.push(Input::bytes("texture:sampled", &sampled_texels()));
        inputs.push(Input::bytes("sampler", &SamplerSpec::new().words()));
        inputs.push(Input::bytes("buffer:colors", &colors_buffer()));
        let mut rects: Vec<u8> = (0..BANDS).flat_map(|at| rect_bytes(band(at))).collect();
        rects.extend(rect_bytes(WHOLE));
        inputs.push(Input::bytes("rects", &rects));
        let mut viewports = NARROW.words();
        viewports.extend(WIDE.words());
        inputs.push(Input::bytes("viewports", &viewports));
        inputs.push(Input::bytes("clear", &CLEAR.to_le_bytes()));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = Fixture::new(FIXTURE).loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let mut states = Vec::new();
        for which in [UNMASKED, MASKED, SAMPLING] {
            let colors = color_state(which);
            states.push(made(
                sites,
                &format!("render-pipeline:{which}"),
                side.render_pipeline(library, &pipeline(which, &colors)),
            )?);
        }
        let target = made(
            sites,
            "target",
            side.target(&TargetSpec::plain(FORMAT, [SIDE; 2])),
        )?;
        let sampled = made(
            sites,
            "texture:sampled",
            side.texture_2d_in_role(&sampled_spec(), &sampled_texels()),
        )?;
        let sampler = made(sites, "sampler", side.sampler(&SamplerSpec::new()))?;
        let colors = made(sites, "buffer:colors", side.buffer(&colors_buffer()))?;
        let read_back = made(
            sites,
            "buffer:read-back",
            side.zeroed_buffer(PLANE as usize),
        )?;

        let color_at = |at: u32| RenderStep::FragmentBuffer {
            buffer: colors,
            offset: u64::from(at) * 16,
            index: 0,
        };
        let rect = |rect: [f32; 4]| RenderStep::VertexBytes {
            bytes: rect_bytes(rect),
            index: 0,
        };
        let steps = vec![
            // Draw 1: the unmasked pipeline over the first band.
            RenderStep::Pipeline(states[UNMASKED]),
            rect(band(0)),
            color_at(0),
            RECT_DRAW,
            // Draw 2: the binding and the rectangle move; the pipeline does not.
            rect(band(1)),
            color_at(1),
            RECT_DRAW,
            // Draw 3: the viewport moves, and the rectangle is the whole target, so where the draw
            // lands is the viewport's answer alone.
            RenderStep::Viewport(NARROW),
            rect(WHOLE),
            color_at(2),
            RECT_DRAW,
            // Draw 4: the viewport goes back and the pipeline changes to the masked one, over the
            // texels draw 1 stored.
            RenderStep::Viewport(WIDE),
            RenderStep::Pipeline(states[MASKED]),
            rect(band(0)),
            color_at(3),
            RECT_DRAW,
            // Draw 5: the pipeline changes again, to one that samples, and binds its texture and
            // its sampler. Its own mask is Metal's default, not the masked pipeline's.
            RenderStep::Pipeline(states[SAMPLING]),
            RenderStep::FragmentTexture {
                texture: sampled,
                index: 0,
            },
            RenderStep::FragmentSampler { sampler, index: 0 },
            rect(band(3)),
            RECT_DRAW,
        ];
        let work = vec![
            RenderWork::Pass(
                RenderPass::clearing(vec![PassColor::cleared(target, [f64::from(CLEAR); 4])])
                    .drawing(steps),
            ),
            RenderWork::ReadTexture {
                texture: target,
                into: read_back,
                offset: 0,
                row_pitch: TEXEL * SIDE,
                size: [SIDE; 2],
                plane: Plane::Whole,
            },
        ];
        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;
        let names = ["pass", "read"];
        let mut pieces = Node::branch("pieces", Vec::new());
        for (at, what) in names.iter().enumerate() {
            pieces = pieces.with(Node::leaf(
                *what,
                Observed::value(match outcome.refused.get(at).and_then(Option::as_ref) {
                    Some(failure) => failure.recorded(),
                    None => Value::ok(Value::Absent),
                }),
            ));
        }
        sites.push(pieces);
        if let Some((at, failure)) = names
            .iter()
            .enumerate()
            .find_map(|(at, what)| outcome.refused.get(at)?.as_ref().map(|f| (*what, f)))
        {
            return Err(Abort {
                at: format!("pieces/{at}"),
                failure: failure.clone(),
            });
        }
        let bytes = call(
            sites,
            "read-back",
            side.read_buffer(read_back, 0, PLANE as usize),
            |_| Value::Absent,
        )?;
        sites.push(
            Node::branch("drawn", Vec::new())
                .with(Node::leaf("texels", Observed::bytes(bytes, layout()))),
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::Transcript;
    use std::collections::BTreeSet;

    /// What the case says texel `(x, y)` of the target holds, from the state each draw was given and
    /// nothing Metal was asked.
    ///
    /// It is a function rather than a table so that the cross-check below and the module's own
    /// consistency test read the same rule.
    fn expected(x: u32, y: u32) -> [f32; 4] {
        match x {
            // Draw 1 stored colour 0; draw 4 drew colour 3 over it under a mask of red and blue, so
            // green and alpha are still draw 1's.
            0..=3 => [COLORS[3][0], COLORS[0][1], COLORS[3][2], COLORS[0][3]],
            // Draw 2, which changed its colour offset and nothing else.
            4..=7 => COLORS[1],
            // Draw 3, confined to the two columns its viewport named.
            8..=9 => COLORS[2],
            // The two columns of the third band the viewport kept the draw out of.
            10..=11 => [CLEAR; 4],
            // Draw 5, which sampled texel (x, y) of its own texture.
            _ => sampled_texel(x, y),
        }
    }

    /// The one outcome of the case's recorded Metal answer, or `None` where nothing has recorded
    /// it. One pass, five draws, no race: the runs fold into one outcome.
    fn only_outcome() -> Option<Transcript> {
        let recording = crate::recording::read(DrawStateChanges.name()).ok()?;
        let answer = recording.answers.first()?;
        assert_eq!(answer.outcomes.len(), 1, "more than one outcome");
        Some(answer.outcomes[0].transcript.clone())
    }

    /// The `float4` at texel `(x, y)` of the target's bytes.
    fn value(bytes: &[u8], x: u32, y: u32) -> [f32; 4] {
        let at = ((y * SIDE + x) * TEXEL) as usize;
        let mut channels = [0f32; 4];
        for (channel, held) in channels.iter_mut().enumerate() {
            let word = at + 4 * channel;
            *held = f32::from_le_bytes([
                bytes[word],
                bytes[word + 1],
                bytes[word + 2],
                bytes[word + 3],
            ]);
        }
        channels
    }

    /// The five regions the five draws and the clear leave are five **different** answers, so a
    /// draw that saw another draw's state writes a value this case can name.
    ///
    /// Written over the rule itself rather than over a recording, so it holds with no device and
    /// fails the moment a colour, a mask or the clear is changed into another region's value.
    #[test]
    fn the_regions_are_five_distinct_answers() {
        let regions: Vec<[u32; 4]> = [0u32, 4, 8, 10, 12]
            .iter()
            .map(|x| expected(*x, 0).map(f32::to_bits))
            .collect();
        let distinct: BTreeSet<[u32; 4]> = regions.iter().copied().collect();
        assert_eq!(distinct.len(), regions.len(), "two regions answer alike");
        // The masked region is neither what draw 1 stored nor what draw 4 drew, so it can only be
        // read as the mask having applied to draw 4 over draw 1's texels.
        let masked = expected(0, 0);
        assert_ne!(masked, COLORS[0], "the mask let draw 4 change nothing");
        assert_ne!(masked, COLORS[3], "the mask was not applied");
        // Every channel the mask names comes from draw 4 and every channel it does not from draw 1,
        // which is the rule stated channel by channel.
        for (channel, bit) in [
            ColorWriteMask::RED,
            ColorWriteMask::GREEN,
            ColorWriteMask::BLUE,
            ColorWriteMask::ALPHA,
        ]
        .into_iter()
        .enumerate()
        {
            let from = if MASK.contains(bit) { 3 } else { 0 };
            assert_eq!(masked[channel], COLORS[from][channel], "channel {channel}");
        }
        // The sampled texture holds a value no draw can write, in every texel, and every texel its
        // own, so a sample at the wrong texel is caught.
        let sampled: BTreeSet<[u32; 4]> = (0..SIDE)
            .flat_map(|y| (0..SIDE).map(move |x| sampled_texel(x, y).map(f32::to_bits)))
            .collect();
        assert_eq!(sampled.len(), (SIDE * SIDE) as usize);
        for color in COLORS.iter().chain(std::iter::once(&[CLEAR; 4])) {
            assert!(!sampled.contains(&color.map(f32::to_bits)));
        }
        // The narrow viewport is inside the third band and leaves two of its columns alone, which
        // is what makes the clear region a fact about the viewport.
        assert_eq!(NARROW.origin[0], 8.0);
        assert_eq!(NARROW.origin[0] + NARROW.size[0], 10.0);
        assert_eq!(expected(10, 0), [CLEAR; 4]);
        assert_eq!(layout().extent(), PLANE as usize);
        // The three pipelines are three different pipelines.
        let words: Vec<Vec<u8>> = [UNMASKED, MASKED, SAMPLING]
            .into_iter()
            .map(|which| {
                let colors = color_state(which);
                pipeline(which, &colors).words()
            })
            .collect();
        for (at, ours) in words.iter().enumerate() {
            for theirs in &words[at + 1..] {
                assert_ne!(ours, theirs);
            }
        }
    }

    /// Metal's recorded target holds, in every texel, what the state each draw was given says it
    /// should.
    ///
    /// This is the cross-check the case owes: the whole expectation is arithmetic -- a `float`
    /// stored as it is, a mask over a destination, a viewport's two columns, a sample at a texel
    /// centre -- so a disagreement is a finding about which state a draw saw and not a number to
    /// adopt.
    #[test]
    fn metal_gave_each_draw_the_state_set_before_it() {
        let Some(recorded) = only_outcome() else {
            println!("multi-draw-state-changes: no recording in the tree yet");
            return;
        };
        let drawn = {
            let node = recorded
                .root
                .child("drawn")
                .and_then(|node| node.child("texels"))
                .expect("no drawn/texels site");
            let Value::Bytes(bytes) = &node.observed.value else {
                panic!("drawn/texels holds no bytes");
            };
            bytes.clone()
        };
        assert_eq!(drawn.len(), PLANE as usize);
        let mut differ = Vec::new();
        for y in 0..SIDE {
            for x in 0..SIDE {
                let (got, want) = (value(&drawn, x, y), expected(x, y));
                if got != want {
                    differ.push(format!(
                        "({x}, {y}): Metal {got:?}, the state says {want:?}"
                    ));
                }
            }
        }
        for line in differ.iter().take(20) {
            println!("MULTI-DRAW-DIFFERS: {line}");
        }
        assert!(
            differ.is_empty(),
            "{} of {} texels differ from the state each draw was given",
            differ.len(),
            SIDE * SIDE
        );
    }
}
