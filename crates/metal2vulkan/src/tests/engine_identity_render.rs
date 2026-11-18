//! Render pipelines under the identity domains (`PointerDomain::Identity` with
//! `HandleDomain::Identity`), over the owned fixture `examples/runtime_examples` (`colored_vertex`,
//! which reads positions and colours from buffers 0 and 1 by vertex ID, and `colored_fragment`,
//! which returns its flat colour):
//!
//! - On `Device::from_queries`' profile the pipeline compiles: the vertex function reaches each of
//!   its buffer slots through its own device address, pushed (`SlotGuestBase`) in its window, with
//!   no descriptor, no root table and no parameter block; the fragment function pushes nothing;
//!   both modules pass L3.
//! - On `Device::new`'s profile, the same pipeline is refused as `Unsupported{IdentityDomains}`
//!   (not implemented), as before: its push window is not the floor's.
//! - On `Device::from_queries`' profile the two stages push one range, dense by link: with
//!   `sampled_fragment`, which pushes its constexpr sampler's index, the fragment function's
//!   fields start at the vertex function's end rounded up to 8 (16), where the fixed windows put
//!   them at 64. A fragment compiled on its own at base 0 and linked after that vertex function is
//!   `InvalidDescriptor(PushBase)`; compiled under the vertex function's
//!   `CompiledFunction::fragment_push_base` it links.
//! - On `Device::from_queries`' profile with no buffer robustness enabled, every `[[stage_in]]`
//!   attribute is fetched in fixed function (the owned fixture `engine/icb_merged_attributes`'s
//!   `ima_vertex`, a `float2` position and a mark read from a `short`): two bindings, the formats
//!   `R32G32_SFLOAT` and `R16_SINT`, nothing pulled, both modules L3-valid. A layout stepped per
//!   instance at rate 2 without the divisor feature is refused as `Unsupported{VertexStepRate}`,
//!   needing `vertexAttributeInstanceRateDivisor`; with it, the layout is an instance-rate binding
//!   of divisor 2 whether or not the device reports `supportsNonZeroFirstInstance`, and at a rate
//!   past `maxVertexAttribDivisor` it is refused naming the limit.
//!
//! `llvm-as` assembles the fixture; a missing tool skips visibly. No GPU is used.

use crate::budget::Budget;
use crate::device::requirements::VkFeature;
use crate::device::{Device, DevicePolicy, HandleDomain, PointerDomain};
use crate::error::{Capability, DescriptorError, Error, Feature, Needs};
use crate::library::Library;
use crate::metal::{PixelFormat, VertexFormat, VertexStepFunction};
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
    VertexAttributeDescriptor, VertexBufferLayoutDescriptor, VertexDescriptor,
};
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{PushField, SlotClass, SlotRef, Stage};
use crate::pipeline::testing::{description, engine_dir, llvm_assemble, AssembleError};
use crate::pipeline::vertex::VertexStride;
use crate::spirv::testing::check_words_valid;
use crate::vk;

/// Profiles with `bufferDeviceAddress` and `shaderInt64`.
const PROFILES: [&str; 2] = ["lavapipe", "moltenvk-m3"];

fn identity() -> DevicePolicy {
    DevicePolicy {
        pointer_domain: PointerDomain::Identity,
        handle_domain: HandleDomain::Identity,
        ..DevicePolicy::default()
    }
}

/// The fixture's library, or `None` (printed) where `llvm-as` is missing.
fn library() -> Option<Library> {
    let ll = engine_dir().join("../examples/runtime_examples.ll");
    match llvm_assemble(&ll) {
        Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the library")),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(e) => panic!("{}: {e:?}", ll.display()),
    }
}

/// `colored_vertex` and `colored_fragment` into one `BGRA8Unorm` attachment.
fn descriptor(library: &Library) -> RenderPipelineDescriptor {
    let function = |name| {
        StageSource::Function(FunctionDescriptor::new(
            library.function(name).expect("the function"),
        ))
    };
    let mut d = RenderPipelineDescriptor::new(function("colored_vertex"));
    d.fragment = Some(function("colored_fragment"));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
    d
}

#[test]
fn from_queries_lowers_a_vertex_function_over_its_pushed_buffer_addresses() {
    let Some(library) = library() else {
        return;
    };
    let d = descriptor(&library);
    for stem in PROFILES {
        let desc = description(stem).unwrap_or_else(|e| panic!("{e}"));
        let device = Device::from_queries(desc, identity()).unwrap_or_else(|e| panic!("{e}"));
        let p = (device.compile_render_pipeline(&d, &Budget::standard()))
            .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        let stages: Vec<Stage> = p.interface.push.iter().map(|s| s.stage).collect();
        assert_eq!(stages, [Stage::Vertex, Stage::Fragment], "{stem}");
        let vertex = &p.interface.push[0];
        let pushed: Vec<(PushField, bool)> = (vertex.fields.iter())
            .map(|f| (f.field, matches!(f.loc, PushLoc::Push { .. })))
            .collect();
        let slot = |index| SlotRef {
            stage: Stage::Vertex,
            class: SlotClass::Buffer,
            index,
        };
        assert_eq!(
            pushed,
            [
                (PushField::SlotGuestBase(slot(0)), true),
                (PushField::SlotGuestBase(slot(1)), true)
            ],
            "{stem}: the vertex function's buffer addresses, pushed"
        );
        assert_eq!(vertex.argument_table_size, 0, "{stem}: no parameter block");
        assert!(
            p.interface.push[1].fields.is_empty(),
            "{stem}: the fragment function pushes nothing"
        );
        assert!(
            !p.interface.needs_pointer_translation && !p.interface.needs_handle_translation,
            "{stem}: no root or handle table"
        );
        assert!(
            p.layout.sets.iter().all(|s| s.bindings.is_empty()),
            "{stem}: no descriptor"
        );
        for stage in &p.stages {
            check_words_valid(&stage.spirv, &p.requirements, device.profile())
                .unwrap_or_else(|e| panic!("{stem}: {:?}: {e}", stage.stage));
        }
    }
}

#[test]
fn from_queries_pushes_a_render_pipeline_dense_by_link() {
    let Some(library) = library() else {
        return;
    };
    let mut d = descriptor(&library);
    d.fragment = Some(StageSource::Function(FunctionDescriptor::new(
        library.function("sampled_fragment").expect("the function"),
    )));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float));
    for stem in PROFILES {
        let desc = description(stem).unwrap_or_else(|e| panic!("{e}"));
        let device = Device::from_queries(desc, identity()).unwrap_or_else(|e| panic!("{e}"));
        let budget = Budget::standard();
        let p = (device.compile_render_pipeline(&d, &budget))
            .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        let ranges: Vec<Option<(u32, u32)>> =
            p.interface.push.iter().map(|s| s.push_range).collect();
        assert_eq!(
            ranges[0],
            Some((0, 16)),
            "{stem}: two buffer addresses from 0"
        );
        let fragment = ranges[1].unwrap_or_else(|| panic!("{stem}: the fragment pushes"));
        assert_eq!(
            fragment.0, 16,
            "{stem}: the fragment's fields follow the vertex's"
        );
        // The stages compiled apart, then linked.
        let keys = (device.render_pipeline_key(&d, &budget)).unwrap_or_else(|e| panic!("{e:?}"));
        let function = |s: &Option<StageSource>| match s {
            Some(StageSource::Function(f)) => f.clone(),
            _ => panic!("a function stage"),
        };
        let vertex = (device.compile_function(
            &function(&Some(d.vertex.clone())),
            keys.vertex_inputs.as_ref().expect("vertex inputs"),
            &budget,
        ))
        .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        assert_eq!(vertex.fragment_push_base(), 16, "{stem}");
        let mut inputs = keys.fragment_inputs.clone().expect("fragment inputs");
        let compiled = |inputs: &crate::pipeline::lower_inputs::LoweringInputs| {
            (device.compile_function(&function(&d.fragment), inputs, &budget))
                .unwrap_or_else(|e| panic!("{stem}: {e:?}"))
        };
        let mut apart = d.clone();
        apart.vertex = StageSource::Compiled(vertex.clone());
        apart.fragment = Some(StageSource::Compiled(compiled(&inputs)));
        match device.compile_render_pipeline(&apart, &budget) {
            Err(Error::InvalidDescriptor(DescriptorError::PushBase {
                fragment,
                vertex_end,
            })) => assert_eq!((fragment, vertex_end), (0, 16), "{stem}"),
            other => panic!("{stem}: {:?}", other.map(|_| "linked")),
        }
        inputs.push_base = vertex.fragment_push_base();
        apart.fragment = Some(StageSource::Compiled(compiled(&inputs)));
        let linked = (device.compile_render_pipeline(&apart, &budget))
            .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        assert_eq!(
            linked.interface.push[1].push_range.map(|r| r.0),
            Some(16),
            "{stem}"
        );
    }
}

#[test]
fn device_new_still_refuses_render_functions_under_the_identity_domains() {
    let Some(library) = library() else {
        return;
    };
    let d = descriptor(&library);
    for stem in PROFILES {
        let desc = description(stem).unwrap_or_else(|e| panic!("{e}"));
        let device = Device::new(desc, identity()).unwrap_or_else(|e| panic!("{e}"));
        match device.compile_render_pipeline(&d, &Budget::standard()) {
            Err(Error::Unsupported(u)) => {
                assert_eq!(
                    (u.feature, u.needs),
                    (Feature::IdentityDomains, Needs::NotImplemented),
                    "{stem}"
                );
            }
            other => panic!("{stem}: {:?}", other.map(|_| "compiled")),
        }
    }
}

/// `icb_merged_attributes`' library, or `None` (printed) where `llvm-as` is missing.
fn ima_library() -> Option<Library> {
    let ll = engine_dir().join("icb_merged_attributes.ll");
    match llvm_assemble(&ll) {
        Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the library")),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(e) => panic!("{}: {e:?}", ll.display()),
    }
}

/// `ima_vertex` and `ima_fragment` into one `RGBA32Uint` attachment: attribute 0 a `float2` in
/// layout 0 of stride 8, attribute 1 a `short` in layout 1 of stride 4, both stepped per vertex.
fn ima_descriptor(library: &Library) -> RenderPipelineDescriptor {
    let function = |name| {
        StageSource::Function(FunctionDescriptor::new(
            library.function(name).expect("the function"),
        ))
    };
    let mut d = RenderPipelineDescriptor::new(function("ima_vertex"));
    d.fragment = Some(function("ima_fragment"));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
    let mut vertex = VertexDescriptor::new();
    let attributes = [(VertexFormat::Float2, 0), (VertexFormat::Short, 1)];
    for (a, (format, layout)) in attributes.into_iter().enumerate() {
        let attribute = VertexAttributeDescriptor::new(format, 0, layout);
        vertex.attributes.insert(a as u32, attribute);
    }
    vertex
        .layouts
        .insert(0, VertexBufferLayoutDescriptor::new(Some(8)));
    vertex
        .layouts
        .insert(1, VertexBufferLayoutDescriptor::new(Some(4)));
    d.vertex_descriptor = Some(vertex);
    d
}

/// `stem`'s profile under the identity domains with no buffer robustness and without
/// `vertexAttributeInstanceRateDivisor`, as the new runtime enables neither where a device does
/// not offer them.
fn fixed_function_device(stem: &str) -> Device {
    let mut desc = description(stem).unwrap_or_else(|e| panic!("{e}"));
    for robust in [
        VkFeature::RobustBufferAccess,
        VkFeature::RobustBufferAccess2,
    ] {
        desc.features.remove(&robust);
    }
    desc.features
        .remove(&VkFeature::VertexAttributeInstanceRateDivisor);
    Device::from_queries(desc, identity()).unwrap_or_else(|e| panic!("{e}"))
}

/// `stem`'s profile under the identity domains with no buffer robustness, with
/// `VK_KHR_vertex_attribute_divisor`'s `vertexAttributeInstanceRateDivisor` enabled, a
/// `maxVertexAttribDivisor` of 4 and `supportsNonZeroFirstInstance` as `nonzero`.
fn divisor_device(stem: &str, nonzero: bool) -> Device {
    use crate::device::requirements::VkExtension;
    let mut desc = description(stem).unwrap_or_else(|e| panic!("{e}"));
    for robust in [
        VkFeature::RobustBufferAccess,
        VkFeature::RobustBufferAccess2,
    ] {
        desc.features.remove(&robust);
    }
    desc.features
        .insert(VkFeature::VertexAttributeInstanceRateDivisor);
    desc.extensions
        .insert(VkExtension::KhrVertexAttributeDivisor);
    desc.limits.max_vertex_attrib_divisor = 4;
    desc.limits.supports_non_zero_first_instance = nonzero;
    Device::from_queries(desc, identity()).unwrap_or_else(|e| panic!("{e}"))
}

/// Under the identity domains with no buffer robustness, `ima_vertex`'s attributes are fetched in
/// fixed function, its `short` mark into a variable declared `int`, and a layout stepped per
/// instance at rate 2 is refused by the divisor feature it needs.
#[test]
fn from_queries_fetches_every_attribute_in_fixed_function() {
    let Some(library) = ima_library() else {
        return;
    };
    let d = ima_descriptor(&library);
    for stem in PROFILES {
        let device = fixed_function_device(stem);
        let p = (device.compile_render_pipeline(&d, &Budget::standard()))
            .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        let formats: Vec<vk::Format> = p.vertex_input.attributes.iter().map(|a| a.format).collect();
        assert_eq!(
            formats,
            [vk::Format::R32G32_SFLOAT, vk::Format::R16_SINT],
            "{stem}: both attributes in fixed function"
        );
        assert_eq!(p.vertex_input.bindings.len(), 2, "{stem}");
        assert!(p.vertex_input.pulled.is_empty(), "{stem}: nothing pulled");
        for stage in &p.stages {
            check_words_valid(&stage.spirv, &p.requirements, device.profile())
                .unwrap_or_else(|e| panic!("{stem}: {:?}: {e}", stage.stage));
        }
        let mut stepped = d.clone();
        let layouts = &mut stepped
            .vertex_descriptor
            .as_mut()
            .expect("the layouts")
            .layouts;
        let layout = layouts.get_mut(&1).expect("layout 1");
        layout.step_function = VertexStepFunction::PerInstance;
        layout.step_rate = 2;
        match device.compile_render_pipeline(&stepped, &Budget::standard()) {
            Err(Error::Unsupported(u)) => assert_eq!(
                (u.feature, u.needs),
                (
                    Feature::VertexStepRate,
                    Needs::AnyOf(vec![vec![Capability::Feature(
                        VkFeature::VertexAttributeInstanceRateDivisor
                    )]])
                ),
                "{stem}"
            ),
            other => panic!("{stem}: {:?}", other.map(|_| "compiled")),
        }
    }
}

/// Under the identity domains with `vertexAttributeInstanceRateDivisor`, a layout stepped per
/// instance at rate 2 is an instance-rate binding of divisor 2, both modules L3-valid, whether or
/// not the device reports `supportsNonZeroFirstInstance` (the consumer refuses a draw at a base
/// instance other than 0 where it does not); at rate 5, past the `maxVertexAttribDivisor` of 4, it
/// is refused naming the limit.
#[test]
fn from_queries_divides_an_instance_step_where_the_divisor_feature_is_enabled() {
    let Some(library) = ima_library() else {
        return;
    };
    let stepped = |rate| {
        let mut d = ima_descriptor(&library);
        let layouts = &mut d.vertex_descriptor.as_mut().expect("the layouts").layouts;
        let layout = layouts.get_mut(&1).expect("layout 1");
        layout.step_function = VertexStepFunction::PerInstance;
        layout.step_rate = rate;
        d
    };
    for stem in PROFILES {
        for nonzero in [false, true] {
            let device = divisor_device(stem, nonzero);
            let p = (device.compile_render_pipeline(&stepped(2), &Budget::standard()))
                .unwrap_or_else(|e| panic!("{stem} {nonzero}: {e:?}"));
            let marks = (p.vertex_input.bindings.iter())
                .find(|binding| binding.buffer_index == 1)
                .expect("the marks' binding");
            assert_eq!(
                (marks.input_rate, marks.divisor),
                (vk::VertexInputRate::INSTANCE, 2),
                "{stem} {nonzero}"
            );
            assert!(p.vertex_input.pulled.is_empty(), "{stem}: nothing pulled");
            for stage in &p.stages {
                check_words_valid(&stage.spirv, &p.requirements, device.profile())
                    .unwrap_or_else(|e| panic!("{stem} {nonzero}: {:?}: {e}", stage.stage));
            }
            match device.compile_render_pipeline(&stepped(5), &Budget::standard()) {
                Err(Error::Unsupported(u)) => assert_eq!(
                    (u.feature, u.needs),
                    (
                        Feature::VertexStepRate,
                        Needs::AnyOf(vec![vec![Capability::Limit {
                            limit: crate::device::requirements::VkLimit::MaxVertexAttribDivisor,
                            required: 5,
                            available: 4,
                        }]])
                    ),
                    "{stem} {nonzero}"
                ),
                other => panic!("{stem} {nonzero}: {:?}", other.map(|_| "compiled")),
            }
        }
    }
}

/// Under the identity domains Metal's constant step is a vertex-rate binding of stride 0, with no
/// divisor and whatever the layout's own stride: on lavapipe, without
/// `vertexAttributeInstanceRateDivisor` and without `supportsNonZeroFirstInstance`, `ima_vertex`'s
/// mark layout at the constant step links to that binding, both modules L3-valid. On the
/// portability subset of `moltenvk-m3`, whose `minVertexInputBindingStrideAlignment` of 1 makes a
/// stride of 0 illegal (`VUID-VkVertexInputBindingDescription-stride-04456`), it is refused as
/// `Unsupported{VertexInputLayout}`. Before the vertex-rate form both were
/// `Unsupported{VertexStepRate}`, needing the divisor.
#[test]
fn from_queries_fetches_a_constant_step_at_vertex_rate_with_stride_zero() {
    let Some(library) = ima_library() else {
        return;
    };
    let mut d = ima_descriptor(&library);
    let layouts = &mut d.vertex_descriptor.as_mut().expect("the layouts").layouts;
    let layout = layouts.get_mut(&1).expect("layout 1");
    layout.step_function = VertexStepFunction::Constant;
    layout.step_rate = 0;
    let lavapipe = fixed_function_device("lavapipe");
    assert!(
        !lavapipe
            .profile()
            .fields()
            .limits
            .supports_non_zero_first_instance,
        "lavapipe's description reports supportsNonZeroFirstInstance, so this proves less"
    );
    let p = (lavapipe.compile_render_pipeline(&d, &Budget::standard()))
        .unwrap_or_else(|e| panic!("lavapipe: {e:?}"));
    let bindings: Vec<(u32, vk::VertexInputRate, u32, VertexStride)> = (p.vertex_input.bindings)
        .iter()
        .map(|b| (b.buffer_index, b.input_rate, b.divisor, b.stride))
        .collect();
    assert_eq!(
        bindings,
        [
            (0, vk::VertexInputRate::VERTEX, 1, VertexStride::Static(8)),
            (1, vk::VertexInputRate::VERTEX, 1, VertexStride::Static(0)),
        ],
        "lavapipe: the constant step's binding"
    );
    assert!(p.vertex_input.pulled.is_empty(), "lavapipe: nothing pulled");
    for stage in &p.stages {
        check_words_valid(&stage.spirv, &p.requirements, lavapipe.profile())
            .unwrap_or_else(|e| panic!("lavapipe: {:?}: {e}", stage.stage));
    }
    let portable = fixed_function_device("moltenvk-m3");
    match portable.compile_render_pipeline(&d, &Budget::standard()) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::VertexInputLayout, "moltenvk-m3: {u:?}")
        }
        other => panic!("moltenvk-m3: {:?}", other.map(|_| "compiled")),
    }
}
