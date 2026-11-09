//! A fetching function that discards, beside a `Depth32Float_Stencil8` attachment, under a
//! stencil test that acts and a depth test that writes, on the device the cfg selects, under the
//! validation layer.
//!
//! `tsw_fetch_discard` (without early tests) reads color 1 (`RG16Float`) as `(x, y)`, discards
//! where `x` is 0, and writes `tint · w` to color 0 (`RGBA16Float`) and `(w, y / 4)` to color 1,
//! `w = x / 2 + y`. Metal runs it for every fragment and tests after it, the stencil test first: a
//! fragment that fails the stencil test runs the face's stencil-failure operation, one that passes
//! it and fails the depth test the depth-failure operation, and one that passes both the pass
//! operation, each under the face's write mask, and only that one stores its colors and depth. In
//! the pixel interlock the function stores color 1 before Vulkan's tests, so it runs both tests
//! itself, against stand-ins of the depth and the stencil, which the runtime fills before each run
//! of such draws and copies back after it; through input attachments (lavapipe) the device's tests
//! come after it as Metal's do. Either way each case draws Metal's answer, which the lane computes
//! by Metal's rule: color 1 cleared to `(0.75, 0.25)`, color 0 to 0, the depth to 1 and the stencil
//! to 0x40, then each full-screen draw at its depth under `Less` with writes and its stencil state.
//! The fullscreen triangle is back-facing under Metal's default winding (`shader_stencil`'s Metal
//! probe on an M4) and front-facing under `CounterClockwise`; the face a draw does not take
//! compares `Never`, so a function that took the wrong face would drop every fragment.
//!
//! The cases, each of 4² texels:
//! - **faces and operations**: the reference passing and failing under `Equal`, `Less` through a
//!   read mask and `GreaterEqual`, the failure operation `Invert` and the depth-failure operation
//!   `DecrementWrap` under a write mask of 0x0f and of 0xff, the pass operations
//!   `IncrementClamp`, `Zero` under a write mask of 0xf0 and `Replace`, the last draw under
//!   `CounterClockwise` (the front face);
//! - **split by a plain draw**: a draw of another pipeline that does not fetch (white into color
//!   0, the device's own tests) replaces the stencil between two tested draws, the second of
//!   which tests the plain draw's stencil, copied into the stand-in;
//! - **loaded**: a pass that stores a plain draw's depth and stencil, then a pass that loads them
//!   and draws the function, failing the stencil test and then passing both.
//!
//! Each case's colors, depth and stencil are read back and compared bit for bit; the form and the
//! notices are printed (`shader-side-depth-test` once in the interlock). At four samples (each
//! attachment 4x, resolved as each pass ends, the stencil by sample 0) a device with
//! multisampled storage images and `maintenance8` draws the same answers through a multisampled
//! stand-in, copied by its image copies (RADV). Otherwise each case's first tested draw is
//! refused, typed, and the case stops there: `NotImplemented` at `Attachment::Depth` on a device
//! that gives no sample its own depth (`shader_depth`; NVIDIA, ANV), else at `Attachment::Stencil`
//! `NotImplemented` on one whose storage images are single-sample (MoltenVK; the stencil's sample
//! planes are not built), and needing `maintenance8` on one without it. A pass whose stencil
//! attachment is a `Stencil8` texture, with no depth attachment, its draws' depth `Always`
//! without writes, is refused at the stencil attachment in the interlock (the stand-in's form
//! needs the depth test beside it), typed `NotImplemented(FramebufferFetch)`, and draws Metal's
//! answer through input attachments.

use super::{fetches, form, halves, library, reads_planes, through_inputs, TEXELS};
use crate::runtime_lane::buffer_with;
use crate::runtime_lane::degrade_notices::counts;
use crate::runtime_lane::resolves::{lane, quiet, Lane, FULL};
use metal2vulkan::device::{VkFeature, VkProperty};
use metal2vulkan::error::{Attachment, BindError, Capability, Feature, Needs};
use metal2vulkan::metal::state::{DepthStencilDescriptor, StencilDescriptor};
use metal2vulkan::metal::{CompareFunction, PixelFormat, PrimitiveType, StencilOperation, Winding};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, Degradation, Error, LoadAction, PipelineOptions,
    RenderPassDescriptor, RenderPipelineState, StoreAction, Texture,
};

/// The colors' formats and the depth's.
const COLOR: PixelFormat = PixelFormat::RGBA16Float;
const COVERAGE: PixelFormat = PixelFormat::RG16Float;
const DEPTH: PixelFormat = PixelFormat::Depth32FloatStencil8;

/// The tint the function multiplies by `w`.
const TINT: [f32; 4] = [1.0, 0.5, 0.25, 0.0];

/// The stencil's clear.
const CLEARED_STENCIL: u8 = 0x40;

/// Which pipeline a draw takes.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Drawn {
    /// The fetching function.
    Tested,
    /// White into color 0, color 1 unwritten: no fetch.
    Plain,
}

/// One face's stencil state and reference.
#[derive(Clone, Copy, Debug)]
struct Face {
    compare: CompareFunction,
    fail: StencilOperation,
    depth_fail: StencilOperation,
    pass: StencilOperation,
    read_mask: u32,
    write_mask: u32,
    reference: u32,
}

impl Face {
    /// A face that drops every fragment and writes nothing: the face a draw must not take.
    const NEVER: Face = Face {
        compare: CompareFunction::Never,
        fail: StencilOperation::Keep,
        depth_fail: StencilOperation::Keep,
        pass: StencilOperation::Keep,
        read_mask: 0xff,
        write_mask: 0xff,
        reference: 0,
    };

    /// `compare` of `reference` through all masks, its operations `Keep` but `pass`.
    const fn passing(compare: CompareFunction, reference: u32, pass: StencilOperation) -> Face {
        Face {
            compare,
            fail: StencilOperation::Keep,
            depth_fail: StencilOperation::Keep,
            pass,
            read_mask: 0xff,
            write_mask: 0xff,
            reference,
        }
    }

    fn descriptor(&self) -> StencilDescriptor {
        let mut d = StencilDescriptor::default();
        d.stencil_compare_function = self.compare;
        d.stencil_failure_operation = self.fail;
        d.depth_failure_operation = self.depth_fail;
        d.depth_stencil_pass_operation = self.pass;
        d.read_mask = self.read_mask;
        d.write_mask = self.write_mask;
        d
    }
}

/// One draw: its pipeline, depth, face states, and whether the triangle is front-facing
/// (`CounterClockwise`; the default winding makes it back-facing).
#[derive(Clone, Copy, Debug)]
struct Step {
    drawn: Drawn,
    z: f32,
    front: Face,
    back: Face,
    front_facing: bool,
}

/// A back-facing draw of `drawn` at `z` under `back`, the front face `Never`.
const fn back(drawn: Drawn, z: f32, back: Face) -> Step {
    Step {
        drawn,
        z,
        front: Face::NEVER,
        back,
        front_facing: false,
    }
}

/// Whether `a cmp b` under Metal's `compare`, `a` the masked reference and `b` the masked
/// stored value.
fn compares(compare: CompareFunction, a: u32, b: u32) -> bool {
    match compare {
        CompareFunction::Never => false,
        CompareFunction::Less => a < b,
        CompareFunction::Equal => a == b,
        CompareFunction::LessEqual => a <= b,
        CompareFunction::Greater => a > b,
        CompareFunction::NotEqual => a != b,
        CompareFunction::GreaterEqual => a >= b,
        CompareFunction::Always => true,
        other => panic!("a compare function the lane does not draw: {other:?}"),
    }
}

/// Stencil operation `op` on stored value `s` with reference `r` (8 bits).
fn operated(op: StencilOperation, s: u32, r: u32) -> u32 {
    match op {
        StencilOperation::Keep => s,
        StencilOperation::Zero => 0,
        StencilOperation::Replace => r,
        StencilOperation::IncrementClamp => (s + 1).min(0xff),
        StencilOperation::DecrementClamp => s.saturating_sub(1),
        StencilOperation::Invert => !s & 0xff,
        StencilOperation::IncrementWrap => (s + 1) & 0xff,
        StencilOperation::DecrementWrap => s.wrapping_sub(1) & 0xff,
        other => panic!("a stencil operation the lane does not draw: {other:?}"),
    }
}

/// What the attachments hold, by Metal's rule (the module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Held {
    color: [f32; 4],
    coverage: [f32; 2],
    depth: f32,
    stencil: u8,
}

impl Held {
    /// The clears.
    const CLEARED: Held = Held {
        color: [0.0; 4],
        coverage: [0.75, 0.25],
        depth: 1.0,
        stencil: CLEARED_STENCIL,
    };

    /// After `step`, `depth` false where the pass has no depth attachment.
    fn after(self, step: &Step, depth: bool) -> Held {
        let face = match step.front_facing {
            true => step.front,
            false => step.back,
        };
        let s = u32::from(self.stencil);
        let r = face.reference & 0xff;
        let write = |op| {
            let v = operated(op, s, r);
            ((s & !face.write_mask) | (v & face.write_mask)) as u8
        };
        if !compares(face.compare, r & face.read_mask, s & face.read_mask) {
            return Held {
                stencil: write(face.fail),
                ..self
            };
        }
        if depth && step.z >= self.depth {
            return Held {
                stencil: write(face.depth_fail),
                ..self
            };
        }
        let stencil = write(face.pass);
        let depth = match depth {
            true => step.z,
            false => self.depth,
        };
        match step.drawn {
            Drawn::Plain => Held {
                color: [1.0; 4],
                depth,
                stencil,
                ..self
            },
            Drawn::Tested => {
                let [x, y] = self.coverage;
                if x == 0.0 {
                    return self;
                }
                let w = x * 0.5 + y;
                Held {
                    color: TINT.map(|t| t * w),
                    coverage: [w, y * 0.25],
                    depth,
                    stencil,
                }
            }
        }
    }
}

/// The lane's pipelines at one attachment layout.
struct Kit {
    tested: RenderPipelineState,
    plain: RenderPipelineState,
}

impl Kit {
    /// The pipelines at `samples`, beside `DEPTH` as depth and stencil, or a `Stencil8` stencil
    /// alone where `stencil_only`.
    fn new(lane: &Lane, samples: u32, stencil_only: bool) -> Kit {
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
        let pipeline = |fragment: StageSource, coverage: ColorAttachmentDescriptor| {
            let mut descriptor = RenderPipelineDescriptor::new(function(&vertices, FULL));
            descriptor.fragment = Some(fragment);
            descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(COLOR));
            descriptor.color_attachments[1] = Some(coverage);
            match stencil_only {
                true => descriptor.stencil_format = Some(PixelFormat::Stencil8),
                false => {
                    descriptor.depth_format = Some(DEPTH);
                    descriptor.stencil_format = Some(DEPTH);
                }
            }
            descriptor.raster_sample_count = samples;
            lane.device
                .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                .unwrap()
        };
        let mut unwritten = ColorAttachmentDescriptor::new(COVERAGE);
        unwritten.write_mask = metal2vulkan::metal::ColorWriteMask::NONE;
        Kit {
            tested: pipeline(
                function(&fetching, "tsw_fetch_discard"),
                ColorAttachmentDescriptor::new(COVERAGE),
            ),
            plain: pipeline(function(&vertices, "rt6_resolve_white"), unwritten),
        }
    }

    fn state(&self, drawn: Drawn) -> &RenderPipelineState {
        match drawn {
            Drawn::Tested => &self.tested,
            Drawn::Plain => &self.plain,
        }
    }
}

/// The textures a case draws into, and at more than one sample their single-sampled resolves.
struct Targets {
    colors: [Texture; 2],
    depth: Texture,
    resolves: Option<([Texture; 2], Texture)>,
    /// The depth texture is a `Stencil8` stencil alone.
    stencil_only: bool,
}

impl Targets {
    fn new(lane: &Lane, samples: u32, stencil_only: bool) -> Targets {
        let depth_format = match stencil_only {
            true => PixelFormat::Stencil8,
            false => DEPTH,
        };
        match samples {
            1 => Targets {
                colors: [lane.single(COLOR), lane.single(COVERAGE)],
                depth: lane.single(depth_format),
                resolves: None,
                stencil_only,
            },
            _ => Targets {
                colors: [lane.msaa(COLOR), lane.msaa(COVERAGE)],
                depth: lane.msaa(depth_format),
                resolves: Some((
                    [lane.single(COLOR), lane.single(COVERAGE)],
                    lane.single(depth_format),
                )),
                stencil_only,
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

/// A pass into `targets`, the colors cleared (color 1 to `(0.75, 0.25)`), the depth and stencil
/// loaded by `load` (cleared to 1 and 0x40), everything stored, and at more than one sample
/// resolved too (the stencil by sample 0).
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
    let resolve = targets.resolves.as_ref().map(|(_, d)| d.clone());
    if !targets.stencil_only {
        pass.depth_attachment.texture = Some(depth.clone());
        pass.depth_attachment.load_action = load;
        pass.depth_attachment.clear_depth = 1.0;
        pass.depth_attachment.store_action = store;
        pass.depth_attachment.resolve_texture = resolve.clone();
    }
    pass.stencil_attachment.texture = Some(depth.clone());
    pass.stencil_attachment.load_action = load;
    pass.stencil_attachment.clear_stencil = u32::from(CLEARED_STENCIL);
    pass.stencil_attachment.store_action = store;
    pass.stencil_attachment.resolve_texture = resolve;
    pass
}

/// Run a pass of `descriptor` drawing `steps` under `Less` with writes where the pass has a
/// `depth` (else `Always` without writes, which uses none) and each step's stencil state; the
/// first draw refused, if any, its step and refusal, and no draw after it.
fn draw(
    lane: &Lane,
    kit: &Kit,
    descriptor: &RenderPassDescriptor,
    steps: &[Step],
    depth: bool,
) -> Option<(Step, Error)> {
    let tint = buffer_with(&lane.device, &halves(&TINT)).unwrap();
    let states: Vec<_> = steps
        .iter()
        .map(|step| {
            let mut state = DepthStencilDescriptor::default();
            if depth {
                state.depth_compare_function = CompareFunction::Less;
                state.depth_write_enabled = true;
            }
            state.front_face_stencil = step.front.descriptor();
            state.back_face_stencil = step.back.descriptor();
            lane.device.new_depth_stencil_state(&state).unwrap()
        })
        .collect();
    let mut refused = None;
    lane.run(descriptor, |lane, encoder, held: &mut Vec<Buffer>| {
        encoder.set_fragment_buffer(Some(&tint), 0, 0).unwrap();
        for (step, state) in steps.iter().zip(&states) {
            let depth = buffer_with(&lane.device, &step.z.to_le_bytes()).unwrap();
            encoder
                .set_render_pipeline_state(kit.state(step.drawn))
                .unwrap();
            encoder.set_depth_stencil_state(state).unwrap();
            encoder.set_stencil_reference_values(step.front.reference, step.back.reference);
            encoder.set_front_facing_winding(match step.front_facing {
                true => Winding::CounterClockwise,
                false => Winding::Clockwise,
            });
            encoder.set_vertex_buffer(Some(&depth), 0, 0).unwrap();
            let result = encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0);
            held.push(depth);
            if let Err(error) = result {
                refused = Some((*step, error));
                break;
            }
        }
    })
    .unwrap();
    refused
}

/// The cases (the module docs): each a name and its passes' steps, the second pass loading the
/// first's depth and stencil.
fn cases() -> [(&'static str, Vec<Step>, Vec<Step>); 3] {
    use CompareFunction::{Always, Equal, GreaterEqual, Less, NotEqual};
    use Drawn::{Plain, Tested};
    use StencilOperation::{DecrementWrap, IncrementClamp, IncrementWrap, Invert, Keep, Replace};
    [
        (
            "faces and operations",
            vec![
                // 0x40 + 1: both tests pass.
                back(Tested, 0.5, Face::passing(Always, 0, IncrementClamp)),
                // 0x40 against 0x41 fails: 0x41 with its low four bits inverted, 0x4e.
                back(
                    Tested,
                    0.3,
                    Face {
                        fail: Invert,
                        write_mask: 0x0f,
                        ..Face::passing(Equal, 0x40, Keep)
                    },
                ),
                // The stencil passes, the depth fails (0.6 against 0.5): 0x4d.
                back(
                    Tested,
                    0.6,
                    Face {
                        depth_fail: DecrementWrap,
                        ..Face::passing(Equal, 0x4e, Keep)
                    },
                ),
                // 0x0c < 0x0d through the read mask 0x0f; both pass: the high four bits zeroed,
                // 0x0d.
                back(
                    Tested,
                    0.4,
                    Face {
                        read_mask: 0x0f,
                        write_mask: 0xf0,
                        ..Face::passing(Less, 0x0c, StencilOperation::Zero)
                    },
                ),
                // Front-facing: 0x10 >= 0x0d; both pass: 0x10.
                Step {
                    drawn: Tested,
                    z: 0.35,
                    front: Face::passing(GreaterEqual, 0x10, Replace),
                    back: Face::NEVER,
                    front_facing: true,
                },
            ],
            vec![],
        ),
        (
            "split by a plain draw",
            vec![
                back(Tested, 0.5, Face::passing(Always, 0, IncrementClamp)),
                back(Plain, 0.45, Face::passing(Always, 0x22, Replace)),
                back(Tested, 0.4, Face::passing(Equal, 0x22, IncrementWrap)),
            ],
            vec![],
        ),
        (
            "loaded",
            vec![back(Plain, 0.5, Face::passing(Always, 0x33, Replace))],
            vec![
                back(
                    Tested,
                    0.2,
                    Face {
                        fail: StencilOperation::Zero,
                        ..Face::passing(NotEqual, 0x33, Keep)
                    },
                ),
                back(Tested, 0.3, Face::passing(Equal, 0, IncrementClamp)),
            ],
        ),
    ]
}

/// Metal's answer for each case at one sample.
#[test]
fn a_fetch_under_an_acting_stencil_test_stores_what_metal_s_later_tests_keep() {
    cases_at(1);
}

/// Metal's answer for each case at four samples where the device has the multisampled stand-in,
/// else each case's first tested draw refused, typed (the module docs).
#[test]
fn at_four_samples_a_fetch_under_an_acting_stencil_test_stores_what_metal_s_later_tests_keep() {
    cases_at(4);
}

/// Whether `refused` is the refusal the lane expects of a tested draw: `FramebufferFetch`
/// needing `needs` at `attachment`.
fn refused_at(
    refused: &Option<(Step, Error)>,
    (attachment, needs): (Attachment, Needs),
    what: &str,
) -> bool {
    match refused {
        Some((step, Error::Unsupported(u) | Error::Bind(BindError::Unsupported(u))))
            if step.drawn == Drawn::Tested =>
        {
            assert_eq!(
                (
                    u.feature.clone(),
                    u.needs.clone(),
                    u.location.as_ref().and_then(|at| at.attachment())
                ),
                (Feature::FramebufferFetch, needs, Some(attachment)),
                "{what}"
            );
            eprintln!(
                "a fetch under an acting stencil test, {what}: refused at the draw at {}, at the \
                 {attachment:?} attachment, typed: {:?}",
                step.z, u.needs
            );
            true
        }
        _ => false,
    }
}

/// The cases (the module docs) at `samples`.
fn cases_at(samples: u32) {
    let (messages, cache, lane) = lane();
    if !fetches(&lane.device) {
        return;
    }
    let device = lane.device.name().to_owned();
    let kit = Kit::new(&lane, samples, false);
    let inputs = through_inputs(&lane);
    // In the interlock at more than one sample: the depth form's refusal first, where the device
    // gives no sample its own depth (`shader_depth`), then the stencil's (the module docs).
    let enabled = lane.device.compile_device().profile().enabled();
    let per_sample = [
        VkProperty::SampleFragCoordDepth,
        VkProperty::CenterFragCoordDepth,
    ]
    .iter()
    .any(|p| enabled.properties.contains(p));
    let maintenance8 = enabled.features.contains(&VkFeature::Maintenance8);
    let refusal = match samples > 1 && !inputs {
        false => None,
        true if !per_sample => Some((Attachment::Depth, Needs::NotImplemented)),
        true if reads_planes(&lane) => Some((Attachment::Stencil, Needs::NotImplemented)),
        true if !maintenance8 => Some((
            Attachment::Stencil,
            Needs::AnyOf(vec![vec![Capability::Feature(VkFeature::Maintenance8)]]),
        )),
        true => None,
    };
    messages.take_notices();
    for (case, first, second) in cases() {
        let what = format!("{case} at {samples}x on {device}");
        let targets = Targets::new(&lane, samples, false);
        let mut refused = draw(
            &lane,
            &kit,
            &pass(&targets, LoadAction::Clear),
            &first,
            true,
        );
        let mut want = first
            .iter()
            .fold(Held::CLEARED, |held, step| held.after(step, true));
        if refused.is_none() && !second.is_empty() {
            refused = draw(
                &lane,
                &kit,
                &pass(&targets, LoadAction::Load),
                &second,
                true,
            );
            let loaded = Held {
                depth: want.depth,
                stencil: want.stencil,
                ..Held::CLEARED
            };
            want = second
                .iter()
                .fold(loaded, |held, step| held.after(step, true));
        }
        if let Some(expected) = &refusal {
            assert!(
                refused_at(&refused, expected.clone(), &what),
                "{what}: refused {:?}",
                refused.map(|(step, error)| format!("{step:?}: {error}"))
            );
            continue;
        }
        assert!(
            refused.is_none(),
            "{what}: refused {:?}",
            refused.map(|(step, error)| format!("{step:?}: {error}"))
        );
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
        assert_eq!(
            read(depth, 1, BlitOption::StencilFromDepthStencil),
            [want.stencil].repeat(TEXELS),
            "{what}: the stencil, want {want:?}"
        );
        eprintln!(
            "a fetch under an acting stencil test, {what} through {}: Metal's answer {want:?}",
            form(&lane)
        );
    }
    let notices = counts(&messages.take_notices());
    eprintln!(
        "a fetch under an acting stencil test at {samples}x on {device}: announced {notices:?}"
    );
    let in_shader = !inputs && refusal.is_none();
    assert_eq!(
        notices.get(&Degradation::ShaderSideDepthTest),
        in_shader.then_some(&1),
        "ShaderSideDepthTest at {samples}x on {device}"
    );
    drop(cache);
    quiet(lane, &messages);
}

/// A `Stencil8` stencil alone (no depth attachment) under an acting stencil test: Metal's answer
/// through input attachments, else the first tested draw refused at the stencil attachment,
/// typed `NotImplemented(FramebufferFetch)` (the module docs).
#[test]
fn a_fetch_under_a_stencil_test_without_a_depth_attachment_is_metals_through_inputs_else_refused() {
    let (messages, cache, lane) = lane();
    if !fetches(&lane.device) {
        return;
    }
    let device = lane.device.name().to_owned();
    let kit = Kit::new(&lane, 1, true);
    let what = format!("a Stencil8 stencil alone on {device}");
    let targets = Targets::new(&lane, 1, true);
    let steps = [back(
        Drawn::Tested,
        0.5,
        Face {
            fail: StencilOperation::Zero,
            ..Face::passing(CompareFunction::Equal, 0x41, StencilOperation::Keep)
        },
    )];
    let refused = draw(
        &lane,
        &kit,
        &pass(&targets, LoadAction::Clear),
        &steps,
        false,
    );
    if through_inputs(&lane) {
        assert!(refused.is_none(), "{what}");
        let want = steps
            .iter()
            .fold(Held::CLEARED, |held, step| held.after(step, false));
        let (colors, stencil) = targets.read();
        let read = |texture: &Texture, bytes, option| lane.read(texture, (0, 0), bytes, option);
        assert_eq!(
            read(&colors[0], 8, BlitOption::None),
            halves(&want.color).repeat(TEXELS),
            "{what}: color 0, want {want:?}"
        );
        assert_eq!(
            read(stencil, 1, BlitOption::None),
            [want.stencil].repeat(TEXELS),
            "{what}: the stencil, want {want:?}"
        );
        eprintln!("{what} through {}: Metal's answer {want:?}", form(&lane));
    } else {
        assert!(
            refused_at(
                &refused,
                (Attachment::Stencil, Needs::NotImplemented),
                &what
            ),
            "{what}: refused {:?}",
            refused.map(|(step, error)| format!("{step:?}: {error}"))
        );
    }
    drop(cache);
    quiet(lane, &messages);
}
