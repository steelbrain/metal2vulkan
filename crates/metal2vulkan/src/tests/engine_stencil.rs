//! A fragment function's `[[stencil]]` output end to end: the direct `FragStencilRefEXT` where the
//! device has stencil export, the shader-side stencil over a shadow inside the ordered pixel
//! interlock where it has only the interlock, and a refusal naming both ways elsewhere; the render
//! pipeline's contract and the binder's writes and push words. The public fixture
//! `fragment_depth_stencil` returns a depth and a stencil. No GPU is used.
//!
//! `llvm-as` assembles the fixture; a missing tool skips visibly.

use crate::bind::{BindState, DrawCall, WriteSource};
use crate::budget::Budget;
use crate::device::requirements::{FormatUsage, VkExtension, VkFeature, VkProperty};
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{BindError, Capability, Error, Feature, Needs};
use crate::library::Library;
use crate::lower::select::LowerInputs;
use crate::metal::state::DepthStencilDescriptor;
use crate::metal::{CompareFunction, PixelFormat, StencilOperation};
use crate::pipeline::descriptor::{FunctionDescriptor, RenderPipelineDescriptor, StageSource};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::layout::BindingSource;
use crate::pipeline::stage_io::StageBuiltin;
use crate::pipeline::sym::{PushField, SyntheticBinding};
use crate::pipeline::testing::{
    compile, compile_with, engine_dir, fixtures_dir, interlocked_coherence, link_and_validate,
    llvm_assemble, AssembleError,
};
use crate::pipeline::RenderPipeline;
use crate::spirv::testing::committed_profiles;
use crate::vk;

fn assemble(path: &std::path::Path) -> Option<Vec<u8>> {
    match llvm_assemble(path) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {}: {why}", path.display()),
    }
}

const ENTRY: &str = "fragment_depth_stencil";

fn stencil_fixture() -> Option<Vec<u8>> {
    assemble(&fixtures_dir().join("fragment_depth_stencil.ll"))
}

fn device(stem: &str) -> Device {
    device_without(stem, &[])
}

/// Committed profile `stem` with `features` removed from its description.
fn device_without(stem: &str, features: &[VkFeature]) -> Device {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    for f in features {
        desc.features.remove(f);
    }
    Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).expect("device")
}

/// On every committed profile the output is the direct `FragStencilRefEXT` where the device has
/// stencil export, the shader-side stencil where it has the ordered pixel interlock (every NVIDIA
/// profile), and otherwise refused naming the extension or what the shader-side stencil lacks. The
/// shader-side module passes L3, runs its body inside the interlock, holds every access there
/// coherent, and states its fact and push field.
#[test]
fn the_stencil_output_takes_the_extension_the_shadow_or_a_refusal_naming_both() {
    let Some(bc) = stencil_fixture() else {
        return;
    };
    let mut forms = [0usize; 3];
    for (stem, p) in committed_profiles().expect("profiles") {
        let f = p.fields();
        let result = compile(&bc, ENTRY, &p);
        if f.fragment.stencil_export {
            forms[0] += 1;
            let a = result.unwrap_or_else(|e| panic!("{stem}: {e}"));
            assert!(
                !a.facts
                    .iter()
                    .any(|x| matches!(x, RuntimeFact::ShadowStencil { .. })),
                "{stem}"
            );
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        } else if f.fragment.pixel_interlock {
            forms[1] += 1;
            let a = result.unwrap_or_else(|e| panic!("{stem}: {e}"));
            assert!(
                a.facts.contains(&RuntimeFact::ShadowStencil {
                    formatless: false,
                    multisampled: false
                }),
                "{stem}: {:?}",
                a.facts
            );
            assert!(
                a.interface
                    .push
                    .fields
                    .iter()
                    .any(|pf| pf.field == PushField::ShadowStencilState),
                "{stem}"
            );
            assert!(a
                .requirements
                .features
                .contains(&VkFeature::FragmentShaderPixelInterlock));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
            let n = interlocked_coherence(&words).unwrap_or_else(|e| panic!("{stem}: {e}"));
            assert_eq!(n.storage_image, 2, "{stem}: the shadow's read and write");
            // Compiled for a multisampled pipeline: a multisampled shadow, tested per sample of
            // the invocation's coverage, the samples that pass written to `SampleMask`.
            let mut inputs = LowerInputs::new();
            inputs.multisampled = true;
            let ms = compile_with(&bc, ENTRY, &p, &inputs)
                .unwrap_or_else(|e| panic!("{stem}: multisampled: {e}"));
            assert!(
                ms.facts.contains(&RuntimeFact::ShadowStencil {
                    formatless: false,
                    multisampled: true
                }),
                "{stem}: {:?}",
                ms.facts
            );
            assert!(ms
                .requirements
                .features
                .contains(&VkFeature::ShaderStorageImageMultisample));
            assert!(ms
                .interface
                .io
                .builtins
                .contains(&StageBuiltin::SampleMaskOut));
            assert!(!ms
                .interface
                .io
                .builtins
                .contains(&StageBuiltin::SampleMaskIn));
            let (_, words) = link_and_validate(&ms, &p)
                .unwrap_or_else(|e| panic!("{stem}: multisampled L3: {e}"));
            let n = interlocked_coherence(&words).unwrap_or_else(|e| panic!("{stem}: {e}"));
            assert_eq!(n.storage_image, 2, "{stem}: the shadow's read and write");
        } else {
            forms[2] += 1;
            match result {
                Err(Error::Unsupported(u)) => {
                    assert_eq!(u.feature, Feature::StencilExport, "{stem}");
                    let Needs::AnyOf(a) = u.needs else {
                        panic!("{stem}: {:?}", u.needs)
                    };
                    assert_eq!(
                        a[0],
                        [Capability::Extension(VkExtension::ExtShaderStencilExport)],
                        "{stem}"
                    );
                    assert!(
                        a[1].contains(&Capability::Feature(
                            VkFeature::FragmentShaderPixelInterlock
                        )),
                        "{stem}: {a:?}"
                    );
                }
                other => panic!("{stem}: {:?}", other.map(|_| ())),
            }
        }
    }
    // The committed profiles cover each kind of device.
    assert!(forms.iter().all(|&n| n > 0), "{forms:?}");
}

/// A render pipeline of `stage_io_vertex_position` and the fixture's fragment with a combined
/// depth/stencil attachment (none when `!attached`).
fn render(io: &[u8], stencil: &[u8], attached: bool) -> RenderPipelineDescriptor {
    let function = |bc: &[u8], name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            Library::from_air(bc)
                .expect("library")
                .function(name)
                .expect("function"),
        ))
    };
    let mut d = RenderPipelineDescriptor::new(function(io, "stage_io_vertex_position"));
    d.fragment = Some(function(stencil, ENTRY));
    if attached {
        d.depth_format = Some(PixelFormat::Depth32FloatStencil8);
        d.stencil_format = Some(PixelFormat::Depth32FloatStencil8);
    }
    d
}

fn refusal(r: Result<RenderPipeline, Error>) -> (Feature, Needs) {
    match r {
        Err(Error::Unsupported(u)) => (u.feature, u.needs),
        other => panic!("{:?}", other.map(|_| ())),
    }
}

/// A render pipeline with a stencil attachment states the shadow (`R8_UINT`, declared with its
/// format) and requires its storage access; one without states none. A multisampled pipeline states
/// a shadow of its samples and requires multisampled storage images of that count and
/// `maintenance8` for the copies. Link refuses alpha to coverage as needing the extension.
#[test]
fn the_render_pipeline_states_the_shadow_and_refuses_what_it_cannot_order() {
    let (Some(io), Some(stencil)) = (
        assemble(&engine_dir().join("stage_io.ll")),
        stencil_fixture(),
    ) else {
        return;
    };
    let dev = device("nvidia-rtx50-linux");
    let b = Budget::standard();
    let p = dev
        .compile_render_pipeline(&render(&io, &stencil, true), &b)
        .expect("pipeline");
    let s = p.graphics.shadow_stencil.expect("the shadow");
    assert_eq!((s.format, s.formatless), (vk::Format::R8_UINT, false));
    assert!(p
        .requirements
        .formats
        .contains(&(vk::Format::R8_UINT, FormatUsage::StorageImage)));
    let b_shadow = p.layout.sets.iter().flat_map(|s| &s.bindings).find(|b| {
        b.source
            == BindingSource::Synthetic {
                stage: crate::pipeline::sym::Stage::Fragment,
                binding: SyntheticBinding::ShadowStencil,
            }
    });
    assert!(b_shadow.is_some(), "{:?}", p.layout.sets);
    assert_eq!(
        b_shadow.map(|b| b.ty),
        Some(vk::DescriptorType::STORAGE_IMAGE)
    );
    let none = dev
        .compile_render_pipeline(&render(&io, &stencil, false), &b)
        .expect("pipeline without a stencil");
    assert!(none.graphics.shadow_stencil.is_none());

    // Four samples: a multisampled shadow, copied by image copies (`maintenance8`), of a count
    // the device's storage images support; without a stencil attachment the neutral is of that
    // count too, and nothing is copied.
    let mut ms = render(&io, &stencil, true);
    ms.raster_sample_count = 4;
    let four = dev.compile_render_pipeline(&ms, &b).expect("four samples");
    let s = four
        .graphics
        .shadow_stencil
        .expect("the multisampled shadow");
    assert_eq!(
        (s.format, s.formatless, s.samples),
        (vk::Format::R8_UINT, false, 4)
    );
    // The shadow's writes are the product's: a function that writes no memory of its own records
    // nothing of its runs, and tests each covered sample.
    assert!(!four.stages.iter().any(|m| m
        .precision_notes
        .contains(&crate::pipeline::PrecisionNote::DeviceFragmentRuns)));
    assert!(four
        .requirements
        .features
        .contains(&VkFeature::Maintenance8));
    assert!(four
        .requirements
        .features
        .contains(&VkFeature::ShaderStorageImageMultisample));
    assert!(four
        .requirements
        .properties
        .contains(&VkProperty::StorageImageSampleCount(4)));
    let mut ms_none = render(&io, &stencil, false);
    ms_none.raster_sample_count = 4;
    let four_none = dev
        .compile_render_pipeline(&ms_none, &b)
        .expect("four samples without a stencil");
    assert!(four_none.graphics.shadow_stencil.is_none());
    assert!(!four_none
        .requirements
        .features
        .contains(&VkFeature::Maintenance8));
    assert!(four_none
        .requirements
        .properties
        .contains(&VkProperty::StorageImageSampleCount(4)));
    // A device without `maintenance8` cannot copy a multisampled shadow.
    let no_copy = device_without("nvidia-rtx50-linux", &[VkFeature::Maintenance8]);
    assert_eq!(
        refusal(no_copy.compile_render_pipeline(&ms, &b)),
        (
            Feature::StencilExport,
            Needs::AnyOf(vec![
                vec![Capability::Extension(VkExtension::ExtShaderStencilExport)],
                vec![Capability::Feature(VkFeature::Maintenance8)],
            ])
        )
    );
    let mut a2c = render(&io, &stencil, true);
    a2c.alpha_to_coverage = true;
    assert_eq!(
        refusal(dev.compile_render_pipeline(&a2c, &b)),
        (
            Feature::StencilExport,
            Needs::AnyOf(vec![vec![Capability::Extension(
                VkExtension::ExtShaderStencilExport
            )]])
        )
    );
}

fn draw() -> DrawCall {
    DrawCall::Primitives {
        vertex_start: 0,
        vertex_count: 3,
        instance_count: 1,
        base_instance: 0,
    }
}

/// The fragment push words of `state`'s draw under `p`.
fn words(p: &RenderPipeline, dev: &Device, state: &BindState) -> Result<[u32; 2], BindError> {
    let plan = p.plan(dev, &state.shape())?;
    let cmds = plan.instantiate_draw(state, &draw())?;
    let placed = p
        .interface
        .push
        .iter()
        .flat_map(|pl| &pl.fields)
        .find(|f| f.field == PushField::ShadowStencilState)
        .expect("the stencil state field");
    let crate::pipeline::PushLoc::Push { offset } = placed.loc else {
        panic!("the state spilled: {placed:?}")
    };
    let w = cmds
        .push_constants
        .iter()
        .find(|w| w.offset <= offset && offset + 8 <= w.offset + w.bytes.len() as u32)
        .expect("the push write");
    let at = (offset - w.offset) as usize;
    let word =
        |i: usize| u32::from_le_bytes(w.bytes[at + 4 * i..at + 4 * i + 4].try_into().unwrap());
    Ok([word(0), word(1)])
}

/// The binder writes the shadow (a neutral where the pass has no stencil attachment, whose words
/// say nothing is attached), writes each face's state in the documented bits, refuses a state whose
/// stencil write could depend on the depth test as needing the extension, and refuses a pass of
/// more than one layer as not implemented.
#[test]
fn the_binder_writes_the_shadow_and_the_encoders_stencil_state() {
    let (Some(io), Some(stencil)) = (
        assemble(&engine_dir().join("stage_io.ll")),
        stencil_fixture(),
    ) else {
        return;
    };
    let dev = device("nvidia-rtx50-linux");
    let b = Budget::standard();
    let p = dev
        .compile_render_pipeline(&render(&io, &stencil, true), &b)
        .expect("pipeline");
    let plan = p.plan(&dev, &BindState::default().shape()).expect("plan");
    assert_eq!(
        plan.descriptors
            .iter()
            .filter(|w| w.source == WriteSource::ShadowStencil)
            .count(),
        1,
        "{:?}",
        plan.descriptors
    );

    let mut state = BindState::default();
    let mut ds = DepthStencilDescriptor::default();
    ds.front_face_stencil.stencil_compare_function = CompareFunction::Less;
    ds.front_face_stencil.stencil_failure_operation = StencilOperation::IncrementClamp;
    ds.front_face_stencil.depth_stencil_pass_operation = StencilOperation::Replace;
    ds.front_face_stencil.depth_failure_operation = StencilOperation::Replace;
    ds.front_face_stencil.read_mask = 0x0F;
    ds.front_face_stencil.write_mask = 0xF0;
    ds.back_face_stencil.stencil_compare_function = CompareFunction::Equal;
    ds.back_face_stencil.depth_stencil_pass_operation = StencilOperation::Invert;
    ds.back_face_stencil.depth_failure_operation = StencilOperation::Invert;
    state.depth_stencil = ds;
    let [front, back] = words(&p, &dev, &state).expect("words");
    let fields = |w: u32| {
        (
            w & 7,
            (w >> 3) & 7,
            (w >> 6) & 7,
            (w >> 9) & 0xFF,
            (w >> 17) & 0xFF,
            w >> 31,
        )
    };
    assert_eq!(fields(front), (1, 3, 2, 0x0F, 0xF0, 1));
    assert_eq!(fields(back), (2, 0, 5, 0xFF, 0xFF, 1));

    // A depth comparison that can fail, with a face whose depth-failure and pass operations
    // differ, makes the write depend on the depth test, which runs after the function.
    ds.depth_compare_function = CompareFunction::Less;
    state.depth_stencil = ds;
    words(&p, &dev, &state).expect("equal depth-failure and pass operations");
    ds.back_face_stencil.depth_failure_operation = StencilOperation::Keep;
    state.depth_stencil = ds;
    match words(&p, &dev, &state) {
        Err(BindError::Unsupported(u)) => assert_eq!(
            (u.feature, u.needs),
            (
                Feature::StencilExport,
                Needs::AnyOf(vec![vec![Capability::Extension(
                    VkExtension::ExtShaderStencilExport
                )]])
            )
        ),
        other => panic!("{other:?}"),
    }
    // Under `Always` the depth test cannot fail.
    ds.depth_compare_function = CompareFunction::Always;
    state.depth_stencil = ds;
    words(&p, &dev, &state).expect("an `Always` depth test");

    let layered = BindState {
        render_target_array_length: 2,
        ..Default::default()
    };
    match p.plan(&dev, &layered.shape()) {
        Err(BindError::Unsupported(u)) => assert_eq!(
            (u.feature, u.needs),
            (Feature::StencilExport, Needs::NotImplemented)
        ),
        other => panic!("{:?}", other.map(|_| ())),
    }

    // Without a stencil attachment a neutral stands in and nothing is attached.
    let none = dev
        .compile_render_pipeline(&render(&io, &stencil, false), &b)
        .expect("pipeline without a stencil");
    let plan = none
        .plan(&dev, &BindState::default().shape())
        .expect("plan");
    assert!(!plan
        .descriptors
        .iter()
        .any(|w| w.source == WriteSource::ShadowStencil));
    let [front, back] = words(&none, &dev, &BindState::default()).expect("words");
    assert_eq!((front >> 31, back >> 31), (0, 0));
}
