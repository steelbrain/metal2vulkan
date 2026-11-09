//! Fetching functions that return a stencil reference, a sample mask, or both
//! (`dsx_fetch_returned_stencil`), beside a `Depth32Float_Stencil8` attachment, under tests that
//! act, on the device the cfg selects, under the validation layer.
//!
//! Each function returns half of the color it fetched plus a quarter, and the stencil (`x`) and
//! the sample mask (`y`) its buffer holds. Each case is a pass of one draw of the fullscreen
//! triangle (back-facing) at depth 0.5 over a fresh 4² `R32Float` target cleared to 0.75, beside
//! the attachment cleared to depth 1 and stencil 0x10, under depth `Less` with writes and the
//! case's stencil face, the encoder's reference 0x20. Metal's answer, measured on an M4:
//!
//! - the returned stencil's low 8 bits are the reference of both faces, in place of the
//!   encoder's;
//! - a sample outside the returned mask (bit 0 at one sample; the invocation's own sample's bit
//!   per sample) keeps its color and depth; where the function returns no stencil it runs no
//!   stencil operation either, and where it does, the stencil test runs for it with the returned
//!   reference and the depth test fails: the face's stencil-failure operation where the stencil
//!   test fails, else its depth-failure operation;
//! - every other sample is tested as usual: a fragment that passes stores 0.625 and depth 0.5.
//!
//! In the pixel interlock the function runs both tests itself, taking what it returns; through
//! input attachments (lavapipe) the device's own tests take it, and a sample outside the mask of a
//! function that also returns a stencil runs no operation there, which is not Metal's: that case is
//! printed and not compared there. At four samples (each attachment 4x, resolved as the pass ends:
//! the color's average, the depth and the stencil by sample 0) a device with the multisampled
//! stand-in draws the same answers; another refuses the draw, typed, as `stencil_stand_in` names
//! (its depth first where no sample has its own depth). A device without
//! `VK_EXT_shader_stencil_export` refuses the function returning both at its pipeline, typed (the
//! shadow takes no sample mask beside it). Each case's colors, depth and stencil are read back and
//! compared bit for bit; the form is printed.

use super::{fetches, form, library, reads_planes, through_inputs, TEXELS, Z};
use crate::runtime_lane::buffer_with;
use crate::runtime_lane::resolves::{lane, quiet, Lane, FULL};
use metal2vulkan::device::{VkFeature, VkProperty};
use metal2vulkan::error::{Attachment, BindError, Capability, Feature, Needs};
use metal2vulkan::metal::state::{DepthStencilDescriptor, StencilDescriptor};
use metal2vulkan::metal::{CompareFunction, PixelFormat, PrimitiveType, StencilOperation};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, Error, LoadAction, PipelineOptions, RenderPassDescriptor,
    StoreAction, Texture,
};

/// The depth and stencil attachment's format.
const DEPTH: PixelFormat = PixelFormat::Depth32FloatStencil8;

/// The color target's clear, and what a fragment that passes stores.
const CLEAR: f32 = 0.75;
const STORED: f32 = 0.625;

/// The stencil's clear, and the encoder's reference.
const CLEARED_STENCIL: u8 = 0x10;
const REFERENCE: u32 = 0x20;

/// What a fixture function returns beside its color.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Returns {
    Stencil,
    Mask,
    Both,
}

impl Returns {
    /// The fixture function.
    fn function(self) -> &'static str {
        match self {
            Returns::Stencil => "fetch_returned_stencil",
            Returns::Mask => "fetch_returned_mask",
            Returns::Both => "fetch_returned_stencil_mask",
        }
    }
}

/// One case: its name, the function, the stencil and mask its buffer holds, and the stencil
/// face (both faces alike).
#[derive(Clone, Copy, Debug)]
struct Case {
    name: &'static str,
    function: Returns,
    stencil: u32,
    mask: u32,
    compare: CompareFunction,
    fail: StencilOperation,
    depth_fail: StencilOperation,
    pass: StencilOperation,
}

impl Case {
    /// Whether the function returns a stencil, and a sample mask.
    fn returns(&self) -> (bool, bool) {
        (
            self.function != Returns::Mask,
            self.function != Returns::Stencil,
        )
    }

    /// The stencil face.
    fn face(&self) -> StencilDescriptor {
        let mut d = StencilDescriptor::default();
        d.stencil_compare_function = self.compare;
        d.stencil_failure_operation = self.fail;
        d.depth_failure_operation = self.depth_fail;
        d.depth_stencil_pass_operation = self.pass;
        d.read_mask = 0xff;
        d.write_mask = 0xff;
        d
    }
}

/// What one sample holds after the draw: whether its color was stored, its depth, its stencil.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Sample {
    stored: bool,
    depth: f32,
    stencil: u8,
}

/// Whether `a cmp b` under Metal's `compare`.
fn compares(compare: CompareFunction, a: u32, b: u32) -> bool {
    match compare {
        CompareFunction::Never => false,
        CompareFunction::Less => a < b,
        CompareFunction::Equal => a == b,
        CompareFunction::NotEqual => a != b,
        CompareFunction::GreaterEqual => a >= b,
        CompareFunction::Always => true,
        other => panic!("a compare function the lane does not draw: {other:?}"),
    }
}

/// Stencil operation `op` on stored value `s` with reference `r` (8 bits).
fn operated(op: StencilOperation, s: u32, r: u32) -> u8 {
    let v = match op {
        StencilOperation::Keep => s,
        StencilOperation::Zero => 0,
        StencilOperation::Replace => r,
        StencilOperation::IncrementClamp => (s + 1).min(0xff),
        StencilOperation::Invert => !s & 0xff,
        other => panic!("a stencil operation the lane does not draw: {other:?}"),
    };
    v as u8
}

/// Metal's answer for sample `sample` of `case` (the module docs).
fn metal(case: &Case, sample: u32) -> Sample {
    let (stencil_out, mask_out) = case.returns();
    let s = u32::from(CLEARED_STENCIL);
    let r = match stencil_out {
        true => case.stencil & 0xff,
        false => REFERENCE,
    };
    let kept = Sample {
        stored: false,
        depth: 1.0,
        stencil: CLEARED_STENCIL,
    };
    let covered = !mask_out || case.mask >> sample & 1 == 1;
    let stencil_passes = compares(case.compare, r, s);
    if !covered {
        return match stencil_out {
            false => kept,
            true => Sample {
                stencil: operated(
                    match stencil_passes {
                        true => case.depth_fail,
                        false => case.fail,
                    },
                    s,
                    r,
                ),
                ..kept
            },
        };
    }
    if !stencil_passes {
        return Sample {
            stencil: operated(case.fail, s, r),
            ..kept
        };
    }
    // Depth 0.5 under `Less` against the cleared 1 passes.
    Sample {
        stored: true,
        depth: Z,
        stencil: operated(case.pass, s, r),
    }
}

/// The cases (the module docs).
fn cases() -> Vec<Case> {
    use CompareFunction::{Always, Equal, Never, NotEqual};
    use StencilOperation::{IncrementClamp, Invert, Keep, Replace, Zero};
    let case = |name, function, stencil, mask, compare, pass| Case {
        name,
        function,
        stencil,
        mask,
        compare,
        fail: Zero,
        depth_fail: IncrementClamp,
        pass,
    };
    let (stencil, mask, both) = (Returns::Stencil, Returns::Mask, Returns::Both);
    vec![
        case("its low 8 bits replace", stencil, 0x1a5, 0, Always, Replace),
        case("compared, passing", stencil, 0x110, 0, Equal, Invert),
        case("compared, failing", stencil, 0x30, 0, Equal, Invert),
        case("a mask of every sample", mask, 0, 0xf, Always, Invert),
        case("a mask of samples 0 and 2", mask, 0, 0x5, Always, Invert),
        case("a mask of samples 1 and 3", mask, 0, 0xa, Always, Invert),
        case("a mask of none, failing", mask, 0, 0, Never, Invert),
        case("both, samples 0 and 2", both, 0x30, 0x5, Always, Replace),
        case("both, samples 1 and 3", both, 0x30, 0xa, Always, Replace),
        // Outside the mask the stencil test runs with the returned reference, the depth test
        // failing: 0x10 against the stored 0x10 passes under `Equal` and fails under
        // `NotEqual`, where the encoder's 0x20 would fail and pass.
        case(
            "both, none, the stencil failing",
            both,
            0x30,
            0,
            Equal,
            Replace,
        ),
        case(
            "both, none, the stencil passing",
            both,
            0x10,
            0,
            Equal,
            Keep,
        ),
        case(
            "both, none, NotEqual failing",
            both,
            0x10,
            0,
            NotEqual,
            Keep,
        ),
    ]
}

/// The textures a case draws into, and at more than one sample their single-sampled resolves.
struct Targets {
    color: Texture,
    depth: Texture,
    resolves: Option<(Texture, Texture)>,
}

impl Targets {
    fn new(lane: &Lane, samples: u32) -> Targets {
        match samples {
            1 => Targets {
                color: lane.single(PixelFormat::R32Float),
                depth: lane.single(DEPTH),
                resolves: None,
            },
            _ => Targets {
                color: lane.msaa(PixelFormat::R32Float),
                depth: lane.msaa(DEPTH),
                resolves: Some((lane.single(PixelFormat::R32Float), lane.single(DEPTH))),
            },
        }
    }

    /// The pass: the color cleared to 0.75, the depth to 1 and the stencil to 0x10, everything
    /// stored, and at more than one sample resolved too (the depth and stencil by sample 0).
    fn pass(&self) -> RenderPassDescriptor {
        let mut pass = RenderPassDescriptor::new();
        let store = match self.resolves {
            Some(_) => StoreAction::StoreAndMultisampleResolve,
            None => StoreAction::Store,
        };
        let color = &mut pass.color_attachments[0];
        color.texture = Some(self.color.clone());
        color.load_action = LoadAction::Clear;
        color.clear_color = ClearColor::new(f64::from(CLEAR), 0.0, 0.0, 0.0);
        color.store_action = store;
        color.resolve_texture = self.resolves.as_ref().map(|(c, _)| c.clone());
        let resolve = self.resolves.as_ref().map(|(_, d)| d.clone());
        pass.depth_attachment.texture = Some(self.depth.clone());
        pass.depth_attachment.load_action = LoadAction::Clear;
        pass.depth_attachment.clear_depth = 1.0;
        pass.depth_attachment.store_action = store;
        pass.depth_attachment.resolve_texture = resolve.clone();
        pass.stencil_attachment.texture = Some(self.depth.clone());
        pass.stencil_attachment.load_action = LoadAction::Clear;
        pass.stencil_attachment.clear_stencil = u32::from(CLEARED_STENCIL);
        pass.stencil_attachment.store_action = store;
        pass.stencil_attachment.resolve_texture = resolve;
        pass
    }

    /// What the lane reads: the resolves, else the textures drawn.
    fn read(&self) -> (&Texture, &Texture) {
        match &self.resolves {
            Some((color, depth)) => (color, depth),
            None => (&self.color, &self.depth),
        }
    }
}

/// The refusal a draw of an in-shader stencil test at more than one sample meets on `lane`'s
/// device in the interlock (`stencil_stand_in`'s order): the depth's where no sample has its own
/// depth, then the stencil's; `None` where the device draws it.
fn refusal(lane: &Lane) -> Option<(Attachment, Needs)> {
    let enabled = lane.device.compile_device().profile().enabled();
    let per_sample = [
        VkProperty::SampleFragCoordDepth,
        VkProperty::CenterFragCoordDepth,
    ]
    .iter()
    .any(|p| enabled.properties.contains(p));
    match () {
        () if !per_sample => Some((Attachment::Depth, Needs::NotImplemented)),
        () if reads_planes(lane) => Some((Attachment::Stencil, Needs::NotImplemented)),
        () if !enabled.features.contains(&VkFeature::Maintenance8) => Some((
            Attachment::Stencil,
            Needs::AnyOf(vec![vec![Capability::Feature(VkFeature::Maintenance8)]]),
        )),
        () => None,
    }
}

/// Metal's answer for every case at `samples` (the module docs).
fn cases_at(samples: u32) {
    let (messages, _cache, lane) = lane();
    if !fetches(&lane.device) {
        return;
    }
    let device = lane.device.name().to_owned();
    let vertices = library(&lane, "rt6_resolves");
    let returning = library(&lane, "dsx_fetch_returned_stencil");
    let function = |library: &metal2vulkan_runtime::Library, name| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let inputs = through_inputs(&lane);
    let shadow = !lane
        .device
        .compile_device()
        .profile()
        .fragment()
        .stencil_export;
    let refused_form = match samples > 1 && !inputs {
        true => refusal(&lane),
        false => None,
    };
    for case in cases() {
        let what = format!(
            "{} ({}) at {samples}x on {device}",
            case.name,
            case.function.function()
        );
        let (stencil_out, mask_out) = case.returns();
        // Through input attachments the device's own tests drop a sample outside the mask of a
        // function that also returns a stencil, which Metal tests (the module docs).
        if inputs && stencil_out && mask_out && (0..samples).any(|s| case.mask >> s & 1 == 0) {
            eprintln!(
                "a fetch returning a stencil and a sample mask, {what} through {}: not compared \
                 (a sample outside the mask runs no stencil operation, where Metal's runs one)",
                form(&lane)
            );
            continue;
        }
        let mut descriptor = RenderPipelineDescriptor::new(function(&vertices, FULL));
        descriptor.fragment = Some(function(&returning, case.function.function()));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::R32Float));
        descriptor.depth_format = Some(DEPTH);
        descriptor.stencil_format = Some(DEPTH);
        descriptor.raster_sample_count = samples;
        let state = match lane
            .device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        {
            // Without stencil export the compiled pipeline holds a returned stencil in the
            // shader-side stencil's shadow, which refuses a returned sample mask beside it.
            Err(Error::Unsupported(u) | Error::Bind(BindError::Unsupported(u)))
                if shadow && stencil_out && mask_out =>
            {
                assert_eq!(
                    (u.feature.clone(), u.needs.clone()),
                    (Feature::StencilExport, Needs::NotImplemented),
                    "{what}"
                );
                eprintln!(
                    "a fetch returning a stencil or a sample mask, {what}: refused at the \
                     pipeline, typed: {:?} {:?}",
                    u.feature, u.needs
                );
                continue;
            }
            Ok(_) if shadow && stencil_out && mask_out => panic!("{what}: not refused"),
            state => state.unwrap_or_else(|error| panic!("{what}: the pipeline: {error}")),
        };
        let mut d = DepthStencilDescriptor::default();
        d.depth_compare_function = CompareFunction::Less;
        d.depth_write_enabled = true;
        d.front_face_stencil = case.face();
        d.back_face_stencil = case.face();
        let ds_state = lane.device.new_depth_stencil_state(&d).unwrap();
        let targets = Targets::new(&lane, samples);
        let mut drawn = None;
        lane.run(&targets.pass(), |lane, encoder, held: &mut Vec<Buffer>| {
            let z = buffer_with(&lane.device, &Z.to_le_bytes()).unwrap();
            let returned: Vec<u8> = [case.stencil, case.mask]
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect();
            let returned = buffer_with(&lane.device, &returned).unwrap();
            encoder.set_render_pipeline_state(&state).unwrap();
            encoder.set_vertex_buffer(Some(&z), 0, 0).unwrap();
            encoder.set_fragment_buffer(Some(&returned), 0, 0).unwrap();
            encoder.set_depth_stencil_state(&ds_state).unwrap();
            encoder.set_stencil_reference_value(REFERENCE);
            drawn = Some(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
            held.push(z);
            held.push(returned);
        })
        .unwrap_or_else(|error| panic!("{what}: the pass: {error}"));
        let drawn = drawn.expect("the draw");
        if let Some((attachment, needs)) = &refused_form {
            match drawn {
                Err(Error::Unsupported(u) | Error::Bind(BindError::Unsupported(u))) => {
                    assert_eq!(
                        (
                            u.feature.clone(),
                            u.needs.clone(),
                            u.location.as_ref().and_then(|at| at.attachment())
                        ),
                        (Feature::FramebufferFetch, needs.clone(), Some(*attachment)),
                        "{what}"
                    );
                    eprintln!(
                        "a fetch returning a stencil or a sample mask, {what}: refused at the \
                         draw, at the {attachment:?} attachment, typed: {:?}",
                        u.needs
                    );
                    continue;
                }
                other => panic!("{what}: not refused: {other:?}"),
            }
        }
        drawn.unwrap_or_else(|error| panic!("{what}: the draw: {error}"));
        let want: Vec<Sample> = (0..samples).map(|s| metal(&case, s)).collect();
        let stored = want.iter().filter(|s| s.stored).count() as f32;
        let color = (stored * STORED + (samples as f32 - stored) * CLEAR) / samples as f32;
        let (color_read, depth_read) = targets.read();
        assert_eq!(
            lane.read(color_read, (0, 0), 4, BlitOption::None),
            color.to_le_bytes().repeat(TEXELS),
            "{what}: the color, want {want:?}"
        );
        assert_eq!(
            lane.read(depth_read, (0, 0), 4, BlitOption::DepthFromDepthStencil),
            want[0].depth.to_le_bytes().repeat(TEXELS),
            "{what}: the depth, want {want:?}"
        );
        assert_eq!(
            lane.read(depth_read, (0, 0), 1, BlitOption::StencilFromDepthStencil),
            [want[0].stencil].repeat(TEXELS),
            "{what}: the stencil, want {want:?}"
        );
        eprintln!(
            "a fetch returning a stencil or a sample mask, {what} through {}: Metal's answer \
             {want:?}",
            form(&lane)
        );
    }
    quiet(lane, &messages);
}

/// Metal's answer for each case at one sample.
#[test]
fn a_fetch_tests_with_the_stencil_and_sample_mask_it_returns() {
    cases_at(1);
}

/// Metal's answer for each case at four samples where the device has the multisampled
/// stand-in, else each draw refused, typed (the module docs).
#[test]
fn at_four_samples_a_fetch_tests_with_the_stencil_and_sample_mask_it_returns() {
    cases_at(4);
}
