//! A fetching function that discards, beside a `Depth32Float_Stencil8` attachment, under a depth
//! test that writes, on the device the cfg selects, under the validation layer.
//!
//! `tsw_fetch_discard` (without early tests) reads color 1 (`RG16Float`) as `(x, y)`, discards
//! where `x` is 0, and writes `tint · w` to color 0 (`RGBA16Float`) and `(w, y / 4)` to color 1,
//! `w = x / 2 + y`. Metal runs it for every fragment and tests after it: a fragment that fails
//! the depth test stores nothing. In the pixel interlock the function stores color 1 before
//! Vulkan's tests, so under a depth test that writes it runs the test itself against a stand-in
//! of the depth, which the runtime fills before each run of such draws and copies back after it;
//! through input attachments (lavapipe) the device's tests come after it as Metal's do. Either way
//! each case draws Metal's answer, which the lane computes by Metal's rule: color 1 cleared to
//! `(0.75, 0.25)`, the depth to 1, color 0 to 0, then each full-screen draw at its depth under
//! `Less` with writes, each surviving one changing all three.
//!
//! The cases, each of 4² texels at one sample:
//! - **one pipeline**: draws at 0.5, 0.7 (fails) and 0.3;
//! - **split by a plain draw**: 0.5, then a draw of another pipeline that does not fetch (white
//!   into color 0, the device's own depth test) at 0.4, then 0.45 (fails against the plain draw's
//!   depth, copied into the stand-in) and 0.35;
//! - **two pipelines** (the coordinator's RADV case: draws of different pipelines meet in the
//!   interlock): the function in two pipelines (color 0 unblended, and blended by `One` and
//!   `Zero`, which stores the same), alternating at 0.5, 0.6 (fails against the first's depth),
//!   0.4 and 0.45 (fails), each draw's test reading the stand-in the one before wrote;
//! - **loaded**: a pass that stores a plain draw's depth of 0.5, then a pass that loads it and
//!   draws the function at 0.6 (fails) and 0.2.
//!
//! Each case's colors and depth are read back and compared bit for bit; the form and the notices
//! are printed (`shader-side-depth-test` once in the interlock).
//!
//! The same cases run again at four samples over Maps' form: each color and the depth 4x and of
//! `RenderTarget` usage alone (no `ShaderRead`), resolved into single-sampled textures as each
//! pass ends (the depth by sample 0). In the interlock the runtime fills a run's stand-in (a plane
//! per sample where the fragment function computes each sample's depth) from the depth by a
//! kernel that samples it, which the depth may do only where the runtime gave it `SAMPLED` at no
//! cost, and the split case needs that copy. Every draw covers each pixel whole at one depth, so
//! every sample of a pixel holds the same values and the resolves are Metal's per-sample answer;
//! a copy that moved a sample into another's plane would not show here. A device whose fragment
//! function has no sample's own depth (neither `VkProperty::SampleFragCoordDepth`, `FragCoord.z`
//! at each sample, nor `VkProperty::CenterFragCoordDepth`, from which it computes it; one of
//! which the compiler's form needs at more than one sample) refuses each case's first tested
//! draw, typed, `NotImplemented(FramebufferFetch)` at `Attachment::Depth`, and the case stops
//! there.

use super::{form, halves, library, through_inputs, TEXELS};
use crate::runtime_lane::buffer_with;
use crate::runtime_lane::degrade_notices::counts;
use crate::runtime_lane::memoryless::memoryless_options;
use crate::runtime_lane::resolves::{lane, quiet, Lane, FULL, SIZE};
use metal2vulkan::device::VkProperty;
use metal2vulkan::error::{Attachment, BindError, Feature, Needs};
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{
    BlendFactor, ColorWriteMask, CompareFunction, PixelFormat, PrimitiveType, TextureType,
    TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, Degradation, Error, LoadAction, PipelineOptions,
    RenderPassDescriptor, RenderPipelineState, StoreAction, Texture, TextureDescriptor,
};

/// The kind the lane's draws take where the function tests the depth itself (once a device).
pub(in crate::runtime_lane) const SHADER_DEPTH: [Degradation; 1] =
    [Degradation::ShaderSideDepthTest];

/// The colors' formats and the depth's.
const COLOR: PixelFormat = PixelFormat::RGBA16Float;
const COVERAGE: PixelFormat = PixelFormat::RG16Float;
const DEPTH: PixelFormat = PixelFormat::Depth32FloatStencil8;

/// The tint the function multiplies by `w`.
const TINT: [f32; 4] = [1.0, 0.5, 0.25, 0.0];

/// Which pipeline a draw takes.
#[derive(Clone, Copy, Debug)]
enum Drawn {
    /// The fetching function, color 0 unblended.
    Tested,
    /// The fetching function, color 0 blended by `One` and `Zero`: another pipeline.
    Other,
    /// White into color 0, color 1 unwritten: no fetch.
    Plain,
}

/// What the attachments hold, by Metal's rule (the module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Held {
    color: [f32; 4],
    coverage: [f32; 2],
    depth: f32,
}

impl Held {
    /// The clears.
    const CLEARED: Held = Held {
        color: [0.0; 4],
        coverage: [0.75, 0.25],
        depth: 1.0,
    };

    /// After a draw of `drawn` at `z` under `Less` with writes.
    fn after(self, drawn: Drawn, z: f32) -> Held {
        if z >= self.depth {
            return self;
        }
        match drawn {
            Drawn::Plain => Held {
                color: [1.0; 4],
                depth: z,
                ..self
            },
            Drawn::Tested | Drawn::Other => {
                let [x, y] = self.coverage;
                if x == 0.0 {
                    return self;
                }
                let w = x * 0.5 + y;
                Held {
                    color: TINT.map(|t| t * w),
                    coverage: [w, y * 0.25],
                    depth: z,
                }
            }
        }
    }
}

/// The lane's pipelines and state.
struct Kit {
    tested: RenderPipelineState,
    other: RenderPipelineState,
    plain: RenderPipelineState,
}

impl Kit {
    /// The pipelines at `samples`.
    fn new(lane: &Lane, samples: u32) -> Kit {
        let vertices = library(lane, "rt6_resolves");
        let fetching = library(lane, "tsw_fetch_discard");
        let function = |library: &metal2vulkan_runtime::Library, name| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let pipeline = |fragment: StageSource, color: ColorAttachmentDescriptor| {
            let mut descriptor = RenderPipelineDescriptor::new(function(&vertices, FULL));
            descriptor.fragment = Some(fragment);
            descriptor.color_attachments[0] = Some(color);
            descriptor.color_attachments[1] = Some(ColorAttachmentDescriptor::new(COVERAGE));
            descriptor.depth_format = Some(DEPTH);
            descriptor.stencil_format = Some(DEPTH);
            descriptor.raster_sample_count = samples;
            lane.device
                .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                .unwrap()
        };
        let mut blended = ColorAttachmentDescriptor::new(COLOR);
        blended.blending_enabled = true;
        blended.source_rgb_blend_factor = BlendFactor::One;
        blended.source_alpha_blend_factor = BlendFactor::One;
        blended.destination_rgb_blend_factor = BlendFactor::Zero;
        blended.destination_alpha_blend_factor = BlendFactor::Zero;
        let mut plain = RenderPipelineDescriptor::new(function(&vertices, FULL));
        plain.fragment = Some(function(&vertices, "rt6_resolve_white"));
        plain.color_attachments[0] = Some(ColorAttachmentDescriptor::new(COLOR));
        let mut unwritten = ColorAttachmentDescriptor::new(COVERAGE);
        unwritten.write_mask = ColorWriteMask::NONE;
        plain.color_attachments[1] = Some(unwritten);
        plain.depth_format = Some(DEPTH);
        plain.stencil_format = Some(DEPTH);
        plain.raster_sample_count = samples;
        Kit {
            tested: pipeline(
                function(&fetching, "tsw_fetch_discard"),
                ColorAttachmentDescriptor::new(COLOR),
            ),
            other: pipeline(function(&fetching, "tsw_fetch_discard"), blended),
            plain: lane
                .device
                .new_render_pipeline_state(&plain, &PipelineOptions::default())
                .unwrap(),
        }
    }

    fn state(&self, drawn: Drawn) -> &RenderPipelineState {
        match drawn {
            Drawn::Tested => &self.tested,
            Drawn::Other => &self.other,
            Drawn::Plain => &self.plain,
        }
    }
}

/// The textures a case draws into: the colors and the depth, and at more than one sample their
/// single-sampled resolves.
struct Targets {
    colors: [Texture; 2],
    depth: Texture,
    resolves: Option<([Texture; 2], Texture)>,
    /// The depth is memoryless: neither loaded nor stored, resolved.
    memoryless: bool,
}

impl Targets {
    /// Fresh textures at `samples`: Maps' form at four (`Lane::msaa`, `RenderTarget` usage
    /// alone), its depth `memoryless` where asked (Maps' own: 4x `Depth32Float_Stencil8`,
    /// `RenderTarget` usage, memoryless).
    fn new(lane: &Lane, samples: u32, memoryless: bool) -> Targets {
        match samples {
            1 => Targets {
                colors: [lane.single(COLOR), lane.single(COVERAGE)],
                depth: lane.single(DEPTH),
                resolves: None,
                memoryless: false,
            },
            _ => Targets {
                colors: [lane.msaa(COLOR), lane.msaa(COVERAGE)],
                depth: match memoryless {
                    false => lane.msaa(DEPTH),
                    true => {
                        let mut descriptor =
                            TextureDescriptor::texture_2d(DEPTH, SIZE, SIZE, false);
                        descriptor.texture_type = TextureType::Type2DMultisample;
                        descriptor.sample_count = samples.into();
                        descriptor.usage = TextureUsage::RENDER_TARGET;
                        descriptor.resource_options = memoryless_options();
                        lane.device.new_texture(&descriptor).unwrap()
                    }
                },
                resolves: Some((
                    [lane.single(COLOR), lane.single(COVERAGE)],
                    lane.single(DEPTH),
                )),
                memoryless,
            },
        }
    }

    /// What the lane reads: the resolves, else the textures drawn.
    fn read(&self) -> (&[Texture; 2], &Texture) {
        match &self.resolves {
            Some((colors, depth)) => (colors, depth),
            None => (&self.colors, &self.depth),
        }
    }
}

/// A pass into `targets`, the colors cleared (color 1 to `(0.75, 0.25)`), the depth loaded by
/// `load` (cleared to 1), everything stored, and at more than one sample resolved too.
fn pass(targets: &Targets, load: LoadAction) -> RenderPassDescriptor {
    let mut pass = RenderPassDescriptor::new();
    let clears = [
        ClearColor::new(0.0, 0.0, 0.0, 0.0),
        ClearColor::new(0.75, 0.25, 0.0, 0.0),
    ];
    let store = match targets.resolves {
        Some(_) => StoreAction::StoreAndMultisampleResolve,
        None => StoreAction::Store,
    };
    for (index, (texture, clear)) in targets.colors.iter().zip(clears).enumerate() {
        let target = &mut pass.color_attachments[index];
        target.texture = Some(texture.clone());
        target.load_action = LoadAction::Clear;
        target.clear_color = clear;
        target.store_action = store;
        target.resolve_texture = targets.resolves.as_ref().map(|(r, _)| r[index].clone());
    }
    let depth = &targets.depth;
    pass.depth_attachment.texture = Some(depth.clone());
    pass.depth_attachment.load_action = load;
    pass.depth_attachment.clear_depth = 1.0;
    pass.depth_attachment.store_action = match targets.memoryless {
        true => StoreAction::MultisampleResolve,
        false => store,
    };
    pass.depth_attachment.resolve_texture = targets.resolves.as_ref().map(|(_, d)| d.clone());
    pass.stencil_attachment.texture = Some(depth.clone());
    pass.stencil_attachment.load_action = LoadAction::Clear;
    pass.stencil_attachment.store_action = match targets.memoryless {
        true => StoreAction::DontCare,
        false => StoreAction::Store,
    };
    pass
}

/// Run a pass of `descriptor` drawing `draws` (each a pipeline and a depth) under `Less` with
/// writes; the first draw refused, if any, its draw and refusal, and no draw after it.
fn draw(
    lane: &Lane,
    kit: &Kit,
    descriptor: &RenderPassDescriptor,
    draws: &[(Drawn, f32)],
) -> Option<((Drawn, f32), Error)> {
    let mut less = DepthStencilDescriptor::default();
    less.depth_compare_function = CompareFunction::Less;
    less.depth_write_enabled = true;
    let less = lane.device.new_depth_stencil_state(&less).unwrap();
    let tint = buffer_with(&lane.device, &halves(&TINT)).unwrap();
    let mut refused = None;
    lane.run(descriptor, |lane, encoder, held: &mut Vec<Buffer>| {
        encoder.set_depth_stencil_state(&less).unwrap();
        encoder.set_fragment_buffer(Some(&tint), 0, 0).unwrap();
        for &(drawn, z) in draws {
            let depth = buffer_with(&lane.device, &z.to_le_bytes()).unwrap();
            encoder.set_render_pipeline_state(kit.state(drawn)).unwrap();
            encoder.set_vertex_buffer(Some(&depth), 0, 0).unwrap();
            let result = encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0);
            held.push(depth);
            if let Err(error) = result {
                refused = Some(((drawn, z), error));
                break;
            }
        }
    })
    .unwrap();
    refused
}

/// A pass's draws, each a pipeline and a depth.
type Draws = Vec<(Drawn, f32)>;

/// Metal's answer for each case (the module docs), at one sample.
#[test]
fn a_fetch_under_a_written_depth_test_stores_what_metal_s_later_test_keeps() {
    cases_at(1, false);
}

/// Maps' own depth: 4x `Depth32Float_Stencil8` of `RenderTarget` usage, memoryless, cleared and
/// never stored (the consumer's r21 log). The cases but the loaded one (a memoryless depth loads
/// nothing) at four samples, the depth resolved: the split cases' later instances copy the depth
/// in from the backing the split pass renders it into, which the sampling kernel reads, so the
/// backing takes `SAMPLED` where the device's depth/colour copies are refused.
#[test]
fn maps_pass_shapes_copy_the_depth_in() {
    cases_at(4, true);
}

/// Metal's answer for each case at four samples over Maps' form, or, where the device's fragment
/// function cannot compute each sample's depth, each case's first tested draw refused, typed (the
/// module docs).
#[test]
fn at_four_samples_a_fetch_under_a_written_depth_test_stores_what_metal_s_later_test_keeps() {
    cases_at(4, false);
}

/// The cases (the module docs) at `samples`, the depth `memoryless` where asked (no loaded case).
fn cases_at(samples: u32, memoryless: bool) {
    use Drawn::{Other, Plain, Tested};
    let (messages, cache, lane) = lane();
    if !fetches(&lane.device) {
        return;
    }
    let device = lane.device.name().to_owned();
    let created = std::time::Instant::now();
    let kit = Kit::new(&lane, samples);
    let created = created.elapsed();
    let mut first_pass = None;
    // Where the device is measured to give no sample its own depth: neither `FragCoord.z` at
    // each sample nor a centre's from which the function computes it.
    let properties = &lane.device.compile_device().profile().enabled().properties;
    let per_sample = [
        VkProperty::SampleFragCoordDepth,
        VkProperty::CenterFragCoordDepth,
    ]
    .iter()
    .any(|p| properties.contains(p));
    let refusing = samples > 1 && !through_inputs(&lane) && !per_sample;
    messages.take_notices();
    let cases: [(&str, Draws, Draws); 4] = [
        (
            "one pipeline",
            vec![(Tested, 0.5), (Tested, 0.7), (Tested, 0.3)],
            vec![],
        ),
        (
            "split by a plain draw",
            vec![(Tested, 0.5), (Plain, 0.4), (Tested, 0.45), (Tested, 0.35)],
            vec![],
        ),
        (
            "two pipelines",
            vec![(Tested, 0.5), (Other, 0.6), (Other, 0.4), (Tested, 0.45)],
            vec![],
        ),
        (
            "loaded",
            vec![(Plain, 0.5)],
            vec![(Tested, 0.6), (Tested, 0.2)],
        ),
    ];
    for (case, first, second) in cases {
        if memoryless && !second.is_empty() {
            continue;
        }
        let what = match memoryless {
            true => format!("{case} at {samples}x over a memoryless depth on {device}"),
            false => format!("{case} at {samples}x on {device}"),
        };
        let targets = Targets::new(&lane, samples, memoryless);
        let drawn = std::time::Instant::now();
        let mut refused = draw(&lane, &kit, &pass(&targets, LoadAction::Clear), &first);
        let drawn = drawn.elapsed();
        eprintln!("{what}: the first pass took {drawn:?}");
        first_pass.get_or_insert(drawn);
        let mut want = first
            .iter()
            .fold(Held::CLEARED, |held, &(drawn, z)| held.after(drawn, z));
        if refused.is_none() && !second.is_empty() {
            refused = draw(&lane, &kit, &pass(&targets, LoadAction::Load), &second);
            let loaded = Held {
                depth: want.depth,
                ..Held::CLEARED
            };
            want = second
                .iter()
                .fold(loaded, |held, &(drawn, z)| held.after(drawn, z));
        }
        match (refused, refusing) {
            (None, false) => {}
            (
                Some(((drawn, z), Error::Unsupported(u) | Error::Bind(BindError::Unsupported(u)))),
                true,
            ) if !matches!(drawn, Drawn::Plain) => {
                assert_eq!(
                    (
                        u.feature,
                        u.needs,
                        u.location.as_ref().and_then(|at| at.attachment())
                    ),
                    (
                        Feature::FramebufferFetch,
                        Needs::NotImplemented,
                        Some(Attachment::Depth)
                    ),
                    "{what}"
                );
                eprintln!(
                    "a fetch under a written depth test, {what}: refused at the draw {drawn:?} at \
                     {z}, at the depth attachment, typed (no computed sample depth)"
                );
                continue;
            }
            (refused, refusing) => panic!(
                "{what} (refusal expected {refusing}): refused {refused:?}",
                refused = refused.map(|((drawn, z), error)| format!("{drawn:?} at {z}: {error}"))
            ),
        }
        let (colors, depth) = targets.read();
        let read = |texture: &Texture, bytes, option| lane.read(texture, (0, 0), bytes, option);
        assert_eq!(
            read(&colors[0], 8, BlitOption::None),
            halves(&want.color).repeat(TEXELS),
            "{what}: color 0, want {want:?}"
        );
        assert_eq!(
            read(&colors[1], 4, BlitOption::None),
            halves(&want.coverage).repeat(TEXELS),
            "{what}: color 1, want {want:?}"
        );
        assert_eq!(
            read(depth, 4, BlitOption::DepthFromDepthStencil),
            want.depth.to_le_bytes().repeat(TEXELS),
            "{what}: the depth, want {want:?}"
        );
        eprintln!(
            "a fetch under a written depth test, {what} through {}: Metal's answer {want:?}",
            form(&lane)
        );
    }
    let notices = counts(&messages.take_notices());
    eprintln!(
        "a fetch under a written depth test at {samples}x on {device}: announced {notices:?}"
    );
    let in_shader = !through_inputs(&lane) && !refusing;
    for kind in SHADER_DEPTH {
        assert_eq!(
            notices.get(&kind),
            in_shader.then_some(&1),
            "{kind:?} at {samples}x on {device}"
        );
    }
    // The pipelines the stand-in's copies run, where they go by the kernel and the draw: made by
    // the states' creation, none by a draw.
    let counts = lane.device.work_counts();
    eprintln!(
        "a fetch under a written depth test at {samples}x on {device}: the states made in \
         {created:?}, the first case's first pass in {first_pass:?}; the stand-in's copies' \
         pipelines made {}, {} of them by a draw",
        counts.stand_in_copies_made, counts.stand_in_copies_made_at_draws
    );
    assert_eq!(
        counts.stand_in_copies_made_at_draws, 0,
        "{samples}x on {device}: a draw made the stand-in's copies' pipelines"
    );
    drop(cache);
    quiet(lane, &messages);
}
