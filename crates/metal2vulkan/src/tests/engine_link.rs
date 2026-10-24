//! Render and compute pipelines linked end to end from compiled engine functions.
//!
//! The owned fixture `engine/stage_io` compiles under the vertex-fetch projection of a
//! vertex descriptor, and its vertex and fragment entries link as one render pipeline: attributes
//! at dense locations in `VertexInputState`, every fragment input at its vertex output's
//! location, colors at their attachments, and both modules pass L3 under the pipeline's table.
//! The pipeline-level refusals (a foreign projection, a color attachment of another class, a
//! dual-source blend without its second output) are typed. A kernel reached by device address
//! states that the binder must build the resident root table. No GPU is used.
//!
//! `llvm-as` assembles the fixtures; a missing tool skips visibly.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::budget::SPIRV_WORDS;
use crate::device::requirements::{FormatUsage, VkFeature};
use crate::device::{Device, DeviceDescription, DevicePolicy, DeviceProfile};
use crate::error::{DescriptorError, Error, Feature, LowerFamily, Needs, RuntimeInput};
use crate::library::{FunctionArtifact, FunctionConstantValues};
use crate::metal::{BlendFactor, PixelFormat, VertexFormat};
use crate::pipeline::compile::{compile_bitcode, EntryRef};
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, VertexAttributeDescriptor, VertexBufferLayoutDescriptor,
    VertexDescriptor,
};
use crate::pipeline::link::vertex::vertex_fetch_projection;
use crate::pipeline::link::{link, LinkInputs, Linked, RenderLinkInputs};
use crate::pipeline::lower_inputs::{LoweringInputs, VertexFetch};
use crate::pipeline::relocate::relocate;
use crate::pipeline::stage_io::{IoNumeric, IoRole};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, SlotClass, SlotRef, Stage, VaryingKey};
use crate::pipeline::testing::{
    compile, engine_dir, fragment_inputs, link_and_validate, llvm_assemble, meter, profile,
    AssembleError,
};
use crate::spirv::testing::{check_words_valid, committed_profiles, spirv_dis};
use crate::vk;

/// Profiles whose format tables list the fixture's vertex and attachment formats.
const PROFILES: [&str; 3] = ["moltenvk-m3", "lavapipe", "amd-rdna3-radv"];

fn assemble(name: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(name)) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}"),
    }
}

fn prof(stem: &str) -> DeviceProfile {
    profile(stem, DevicePolicy::default()).expect("committed profile")
}

/// Committed profile `stem` without buffer robustness: its containment is checked in the shader.
fn without_robustness(stem: &str) -> DeviceProfile {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    desc.features.remove(&VkFeature::RobustBufferAccess);
    desc.features.remove(&VkFeature::RobustBufferAccess2);
    Device::new(desc, DevicePolicy::default())
        .expect("device")
        .profile()
        .clone()
}

/// The vertex descriptor of `stage_io_vertex`'s `Attributes`: one interleaved per-vertex buffer.
fn vertex_descriptor() -> VertexDescriptor {
    let mut d = VertexDescriptor::new();
    d.layouts
        .insert(0, VertexBufferLayoutDescriptor::new(Some(24)));
    for (index, format, offset) in [
        (0, VertexFormat::Float3, 0),
        (1, VertexFormat::Half2, 12),
        (4, VertexFormat::UShort2, 16),
        (7, VertexFormat::UChar4, 20),
    ] {
        d.attributes
            .insert(index, VertexAttributeDescriptor::new(format, offset, 0));
    }
    d
}

/// `stage_io_fragment`'s targets: a float color at 0 and an unsigned-integer one at 2.
fn render_inputs(d: VertexDescriptor) -> RenderLinkInputs {
    let mut r = RenderLinkInputs::new();
    r.vertex_descriptor = Some(d);
    r.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    r.color_attachments[2] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
    r.depth_format = Some(PixelFormat::Depth32Float);
    r
}

fn render(r: &RenderLinkInputs) -> LinkInputs {
    LinkInputs::Render(Box::new(r.clone()))
}

/// Compile `entry` with lowering inputs `inputs`.
fn compile_with(
    bc: &[u8],
    entry: &str,
    p: &DeviceProfile,
    inputs: &LoweringInputs,
) -> Result<Arc<FunctionArtifact>, Error> {
    compile_bitcode(
        bc,
        EntryRef::Name(entry),
        &FunctionConstantValues::defaults(),
        p,
        inputs,
        &mut meter(),
        SPIRV_WORDS,
    )
}

/// The vertex function compiled under the projection of `d` on `p` (the two-step flow of a
/// render pipeline: the function's attribute classes, then the projection as a lowering input).
fn vertex_under(
    bc: &[u8],
    d: &VertexDescriptor,
    p: &DeviceProfile,
) -> Result<Arc<FunctionArtifact>, Error> {
    let first = compile(bc, "stage_io_vertex", p)?;
    let classes: BTreeMap<u32, IoNumeric> = first
        .interface
        .io
        .attributes()
        .into_iter()
        .map(|(a, (n, _))| (a, n))
        .collect();
    let mut inputs = LoweringInputs::new();
    inputs.vertex_fetch = Some(vertex_fetch_projection(Some(d), &classes, p)?);
    compile_with(bc, "stage_io_vertex", p, &inputs)
}

/// The relocated `Location` literal of `key` in `a`'s words.
fn location(a: &FunctionArtifact, words: &[u32], key: &VaryingKey) -> u32 {
    let at = a
        .module
        .relocs
        .iter()
        .find(|(s, _)| *s == crate::pipeline::sym::Symbol::Location(key.clone()))
        .map(|(_, at)| *at)
        .unwrap_or_else(|| panic!("no Location reloc for {key:?}"));
    words[at as usize]
}

/// Relocate every stage under `l` and run L3 on each.
fn validate_all(l: &Linked, stages: &[&FunctionArtifact], p: &DeviceProfile) -> Vec<Vec<u32>> {
    stages
        .iter()
        .map(|a| {
            let words = relocate(a, &l.table).expect("relocate");
            check_words_valid(&words, &a.requirements, p).unwrap_or_else(|e| panic!("L3: {e}"));
            words
        })
        .collect()
}

/// The fixture's vertex and fragment entries link as one render pipeline on every profile with
/// the formats: dense attribute locations in `VertexInputState`, fragment inputs at their vertex
/// outputs' locations, colors at their attachments, and L3 on both relocated modules.
#[test]
fn a_vertex_fragment_pipeline_links_end_to_end() {
    let Some(bc) = assemble("stage_io.ll") else {
        return;
    };
    for stem in PROFILES {
        let p = prof(stem);
        assert!(
            p.supports_format(vk::Format::R32G32B32_SFLOAT, FormatUsage::VertexBuffer),
            "{stem} lists vertex formats"
        );
        let d = vertex_descriptor();
        let v = vertex_under(&bc, &d, &p).unwrap_or_else(|e| panic!("{stem}: vertex: {e}"));
        let r = render_inputs(d);
        let f =
            compile_with(&bc, "stage_io_fragment", &p, &fragment_inputs(&r, &p)).expect("fragment");
        let l = link(&[&v, &f], &render(&r), &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let words = validate_all(&l, &[&v, &f], &p);
        let (vw, fw) = (&words[0], &words[1]);
        // Attributes 0, 1, 4, 7 at locations 0..4, fixed function, in one binding.
        let attrs: Vec<(u32, u32, vk::Format, u32)> = l
            .vertex_input
            .attributes
            .iter()
            .map(|a| (a.location, a.binding, a.format, a.offset))
            .collect();
        assert_eq!(
            attrs,
            vec![
                (0, 0, vk::Format::R32G32B32_SFLOAT, 0),
                (1, 0, vk::Format::R16G16_SFLOAT, 12),
                (2, 0, vk::Format::R16G16_UINT, 16),
                (3, 0, vk::Format::R8G8B8A8_UINT, 20),
            ],
            "{stem}"
        );
        assert!(l.vertex_input.pulled.is_empty(), "{stem}");
        for (n, index) in [0u32, 1, 4, 7].into_iter().enumerate() {
            assert_eq!(
                location(&v, vw, &VaryingKey::attribute(index)),
                n as u32,
                "{stem}: attribute {index}"
            );
        }
        for input in &f.interface.io.inputs {
            assert_eq!(
                location(&f, fw, &input.key),
                location(&v, vw, &input.key),
                "{stem}: {:?}",
                input.key
            );
        }
        for c in &f.interface.io.outputs {
            let IoRole::Color { attachment, .. } = c.role else {
                panic!("fragment output role {:?}", c.role)
            };
            assert_eq!(location(&f, fw, &c.key), attachment, "{stem}");
        }
        let g = l.graphics.as_ref().expect("graphics");
        assert_eq!(g.color_attachments.len(), 3, "{stem}");
        assert_eq!(g.depth_format, Some(vk::Format::D32_SFLOAT), "{stem}");
    }
}

/// Pipeline-level refusals: a stage compiled under another projection, a signed-integer
/// attribute format (pulled, which is not lowered yet), and a color attachment whose numeric
/// class differs from the output's.
#[test]
fn pipeline_mismatches_are_typed_refusals() {
    let Some(bc) = assemble("stage_io.ll") else {
        return;
    };
    let p = prof("moltenvk-m3");
    let v = vertex_under(&bc, &vertex_descriptor(), &p).expect("vertex");
    let f = compile(&bc, "stage_io_fragment", &p).expect("fragment");
    // Attribute 4 as a signed format: its projection is pulled, so the fixed-function stage is
    // foreign to the descriptor; compiled under that projection it is fetched in the shader
    // and links.
    let mut signed = vertex_descriptor();
    signed.attributes.insert(
        4,
        VertexAttributeDescriptor::new(VertexFormat::Short2, 16, 0),
    );
    assert_eq!(
        link(&[&v, &f], &render(&render_inputs(signed.clone())), &p).unwrap_err(),
        Error::InvalidDescriptor(DescriptorError::VertexAttribute { attribute: 4 })
    );
    let pulled = vertex_under(&bc, &signed, &p).expect("pulled attribute");
    assert!(pulled.interface.io.pulled.iter().any(|x| x.index == 4));
    link(&[&pulled, &f], &render(&render_inputs(signed.clone())), &p).expect("pulled links");
    let mut r = render_inputs(vertex_descriptor());
    // A float attachment for the `uint4` output: Metal's pipeline error.
    r.color_attachments[2] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    assert_eq!(
        link(&[&v, &f], &render(&r), &p).unwrap_err(),
        Error::InvalidDescriptor(DescriptorError::ColorAttachment { index: 2 })
    );
    // A signed-integer attachment: the fragment compiled for an unsigned one declares the output
    // `uint`, so it is a stage of another pipeline.
    r.color_attachments[2] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Sint));
    assert_eq!(
        link(&[&v, &f], &render(&r), &p).unwrap_err(),
        Error::InvalidDescriptor(DescriptorError::ColorAttachment { index: 2 })
    );
    // An absent attachment discards the output.
    r.color_attachments[2] = None;
    link(&[&v, &f], &render(&r), &p).expect("absent attachment");
}

/// A fragment function compiled for a render pipeline that has no attachment at a color it writes
/// (`LoweringInputs::unattached_colors`) keeps no output variable there, so the linked pipeline
/// writes no location without an attachment, as Metal discards the write: its interface lists the
/// color as unattached, the pipeline's color outputs leave it out, and both relocated modules pass
/// L3. Link refuses it in a pipeline with an attachment there, and under alpha to coverage for
/// color 0, whose alpha the coverage reads (`NeedsInput(ColorWrites)`). Compiled without the input,
/// the function still writes the location.
#[test]
fn a_color_without_an_attachment_is_written_nowhere() {
    let Some(bc) = assemble("stage_io.ll") else {
        return;
    };
    let uint_output = "OpTypePointer Output %v4uint";
    for stem in PROFILES {
        let p = prof(stem);
        let v = vertex_under(&bc, &vertex_descriptor(), &p).expect("vertex");
        let attached = render_inputs(vertex_descriptor());
        let mut r = attached.clone();
        r.color_attachments[2] = None;
        let plain =
            compile_with(&bc, "stage_io_fragment", &p, &fragment_inputs(&r, &p)).expect("fragment");
        let mut inputs = fragment_inputs(&r, &p);
        inputs.unattached_colors.insert(2);
        let f = compile_with(&bc, "stage_io_fragment", &p, &inputs).expect("unattached");
        let colors = |a: &FunctionArtifact| -> Vec<u32> {
            a.interface
                .io
                .outputs
                .iter()
                .filter_map(|o| match o.role {
                    IoRole::Color { attachment, .. } => Some(attachment),
                    _ => None,
                })
                .collect()
        };
        assert_eq!(colors(&plain), [0, 2], "{stem}");
        assert_eq!(colors(&f), [0], "{stem}");
        assert_eq!(f.interface.io.unattached_colors, [2], "{stem}");
        assert!(plain.interface.io.unattached_colors.is_empty(), "{stem}");
        for (a, outputs, written) in [(&plain, 0b101, true), (&f, 0b1, false)] {
            let l = link(&[&v, a], &render(&r), &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
            let g = l.graphics.as_ref().expect("graphics");
            assert_eq!(g.color_outputs, outputs, "{stem}");
            let words = validate_all(&l, &[&v, a], &p);
            let text = spirv_dis(&words[1]);
            assert_eq!(text.contains(uint_output), written, "{stem}\n{text}");
        }
        // With the attachment the write is Metal's to store: the stage is another pipeline's.
        assert_eq!(
            link(&[&v, &f], &render(&attached), &p).unwrap_err(),
            Error::NeedsInput(RuntimeInput::ColorWrites),
            "{stem}"
        );
        link(&[&v, &plain], &render(&attached), &p).expect("plain fragment");
        // Color 0's alpha decides the coverage under alpha to coverage, attachment or not.
        let mut covered = r.clone();
        covered.color_attachments[0] = None;
        covered.alpha_to_coverage = true;
        let mut inputs = fragment_inputs(&covered, &p);
        inputs.unattached_colors.extend([0, 2]);
        let f = compile_with(&bc, "stage_io_fragment", &p, &inputs).expect("no color 0");
        assert_eq!(colors(&f), [0; 0], "{stem}");
        assert_eq!(
            link(&[&v, &f], &render(&covered), &p).unwrap_err(),
            Error::NeedsInput(RuntimeInput::ColorWrites),
            "{stem}"
        );
    }
}

/// Without buffer robustness nothing bounds fixed-function vertex input
/// (`VUID-vkCmdDraw-None-02721`), so the projection pulls every attribute. On each profile with
/// its robustness removed, the fixture's descriptor (the one `a_vertex_fragment_pipeline_links_end_to_end`
/// fetches fixed function) links with no fixed-function binding or attribute, every attribute is
/// pulled from buffer 0 through a read-only storage descriptor, and both modules pass L3. The
/// vertex stage compiled with fixed-function input for the same profile is another pipeline's:
/// link refuses it against the descriptor.
#[test]
fn without_buffer_robustness_every_attribute_is_pulled() {
    let Some(bc) = assemble("stage_io.ll") else {
        return;
    };
    for stem in PROFILES {
        let p = without_robustness(stem);
        let d = vertex_descriptor();
        let v = vertex_under(&bc, &d, &p).unwrap_or_else(|e| panic!("{stem}: vertex: {e}"));
        let attrs = v.interface.io.attributes();
        assert_eq!(attrs.len(), 4, "{stem}");
        assert!(
            attrs.values().all(|(_, f)| matches!(
                f,
                VertexFetch::Pulled {
                    buffer_index: 0,
                    ..
                }
            )),
            "{stem}: {attrs:?}"
        );
        assert!(v.interface.io.inputs.is_empty(), "{stem}");
        let r = render_inputs(d);
        let f =
            compile_with(&bc, "stage_io_fragment", &p, &fragment_inputs(&r, &p)).expect("fragment");
        let l = link(&[&v, &f], &render(&r), &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        validate_all(&l, &[&v, &f], &p);
        assert!(l.vertex_input.bindings.is_empty(), "{stem}");
        assert!(l.vertex_input.attributes.is_empty(), "{stem}");
        let pulled: Vec<(u32, u32)> = l
            .vertex_input
            .pulled
            .iter()
            .map(|a| (a.attribute, a.buffer_index))
            .collect();
        assert_eq!(pulled, [(0, 0), (1, 0), (4, 0), (7, 0)], "{stem}");
        let source = ResourceSlot {
            stage: Stage::Vertex,
            kind: ResourceKind::Buffer,
            index: 0,
        };
        assert!(
            v.interface
                .descriptors
                .iter()
                .any(|b| b.slot == source && b.ty == vk::DescriptorType::STORAGE_BUFFER),
            "{stem}: buffer 0's storage descriptor"
        );
        // The same function with fixed-function input, as `compile_function` builds it from the
        // default lowering inputs: not this descriptor's stage on this profile.
        let fixed = compile(&bc, "stage_io_vertex", &p).expect("fixed-function vertex");
        assert!(fixed.interface.io.pulled.is_empty(), "{stem}");
        assert_eq!(
            link(&[&fixed, &f], &render(&r), &p).unwrap_err(),
            Error::InvalidDescriptor(DescriptorError::VertexAttribute { attribute: 0 }),
            "{stem}"
        );
    }
}

/// `signed_color`'s targets: a float color at 0, a signed-integer one at 1, an unsigned-integer
/// one at 2.
fn signed_targets() -> RenderLinkInputs {
    let mut r = RenderLinkInputs::new();
    r.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    r.color_attachments[1] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Sint));
    r.color_attachments[2] = Some(ColorAttachmentDescriptor::new(PixelFormat::RG32Uint));
    r
}

/// The declared class of `a`'s color output to `attachment`.
fn color_class(a: &FunctionArtifact, attachment: u32) -> Option<IoNumeric> {
    a.interface
        .io
        .outputs
        .iter()
        .find(|o| {
            o.role
                == IoRole::Color {
                    attachment,
                    index: 0,
                }
        })
        .map(|o| o.ty.numeric)
}

/// A 32-bit integer color output to a signed-integer attachment is declared `int` when the fragment
/// compiles with that attachment in `LoweringInputs::signed_color_attachments`, its bits stored by
/// `OpBitcast`, and links against the attachment (L3 on both relocated modules). Compiled without
/// it, or linked against an unsigned attachment, the stage is another pipeline's
/// (`ColorAttachment`); a narrower integer output to one is `NotImplemented`.
#[test]
fn a_signed_integer_attachment_takes_an_int_output() {
    let Some(bc) = assemble("signed_color.ll") else {
        return;
    };
    for stem in PROFILES {
        let p = prof(stem);
        for format in [
            vk::Format::R32G32B32A32_SINT,
            vk::Format::R32G32_UINT,
            vk::Format::R8G8B8A8_UNORM,
        ] {
            assert!(
                p.supports_format(format, FormatUsage::ColorAttachment),
                "{stem}: {format:?}"
            );
        }
        let v = compile(&bc, "signed_color_v", &p).expect("vertex");
        let mut inputs = fragment_inputs(&signed_targets(), &p);
        inputs.signed_color_attachments.insert(1);
        let f = compile_with(&bc, "signed_targets_f", &p, &inputs).expect("fragment");
        assert_eq!(color_class(&f, 0), Some(IoNumeric::Float), "{stem}");
        assert_eq!(color_class(&f, 1), Some(IoNumeric::Sint), "{stem}");
        assert_eq!(color_class(&f, 2), Some(IoNumeric::Uint), "{stem}");
        let l = link(&[&v, &f], &render(&signed_targets()), &p).expect("signed pipeline");
        let words = validate_all(&l, &[&v, &f], &p);
        let text = spirv_dis(&words[1]);
        assert!(
            text.contains("%_ptr_Output_v4int = OpTypePointer Output %v4int"),
            "{stem}: the int4 output\n{text}"
        );
        assert!(
            text.contains("OpBitcast %v4int"),
            "{stem}: its bits\n{text}"
        );
        assert!(
            text.contains("%_ptr_Output_v2uint = OpTypePointer Output %v2uint"),
            "{stem}: the uint2 output\n{text}"
        );
        // The whole return as the signed color.
        let mut r = RenderLinkInputs::new();
        r.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Sint));
        inputs.signed_color_attachments = [0].into();
        inputs.color_writes = fragment_inputs(&r, &p).color_writes;
        let whole = compile_with(&bc, "signed_color_f", &p, &inputs).expect("int4 return");
        assert_eq!(color_class(&whole, 0), Some(IoNumeric::Sint), "{stem}");
        let l = link(&[&v, &whole], &render(&r), &p).expect("int4 pipeline");
        validate_all(&l, &[&v, &whole], &p);
        // A narrow signed output under the input is not lowered; without it, it compiles.
        match compile_with(&bc, "narrow_signed_color_f", &p, &inputs) {
            Err(Error::Unsupported(u)) => assert_eq!(
                (u.feature, u.needs),
                (
                    Feature::Lowering(LowerFamily::StageIo),
                    Needs::NotImplemented
                ),
                "{stem}"
            ),
            got => panic!("{stem}: a narrow signed output: {:?}", got.map(|_| ())),
        }
        compile(&bc, "narrow_signed_color_f", &p).expect("narrow output for an unsigned target");
        // Another pipeline's stage: compiled without the attachment, or linked to a `uint` one.
        let plain = compile(&bc, "signed_targets_f", &p).expect("fragment without the input");
        assert_eq!(color_class(&plain, 1), Some(IoNumeric::Uint), "{stem}");
        assert_eq!(
            link(&[&v, &plain], &render(&signed_targets()), &p).unwrap_err(),
            Error::InvalidDescriptor(DescriptorError::ColorAttachment { index: 1 }),
            "{stem}"
        );
        let mut unsigned = signed_targets();
        unsigned.color_attachments[1] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
        assert_eq!(
            link(&[&v, &f], &render(&unsigned), &p).unwrap_err(),
            Error::InvalidDescriptor(DescriptorError::ColorAttachment { index: 1 }),
            "{stem}"
        );
    }
}

/// A blend that reads `Source1` needs the fragment function's `index(1)` output at that
/// attachment; the dual-source entry links, the single-output one is refused.
#[test]
fn dual_source_blending_needs_the_second_output() {
    let Some(bc) = assemble("stage_io.ll") else {
        return;
    };
    let p = prof("moltenvk-m3");
    let v = compile(&bc, "stage_io_vertex_position", &p).expect("vertex");
    let dual = compile(&bc, "stage_io_fragment_dual_source", &p).expect("dual source");
    let single = compile(&bc, "stage_io_fragment_point", &p).expect("single output");
    let mut r = RenderLinkInputs::new();
    let mut blend = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
    blend.blending_enabled = true;
    blend.destination_rgb_blend_factor = BlendFactor::OneMinusSource1Color;
    r.color_attachments[0] = Some(blend);
    let l = link(&[&v, &dual], &render(&r), &p).expect("dual-source pipeline");
    validate_all(&l, &[&v, &dual], &p);
    assert_eq!(
        l.table.locations[&VaryingKey::color(0, 1)],
        (0, 0),
        "the second output shares location 0 (its Index is 1)"
    );
    assert_eq!(
        link(&[&v, &single], &render(&r), &p).unwrap_err(),
        Error::InvalidDescriptor(DescriptorError::ColorAttachment { index: 0 })
    );
}

/// A function that reaches a buffer by device address reads the resident root table, and says so in
/// its interface and in the pipeline's; one that does not, does not.
#[test]
fn a_device_address_slot_needs_pointer_translation() {
    let Some(bc) = assemble("store_const.ll") else {
        return;
    };
    let mut inputs = LoweringInputs::new();
    inputs.psb_slots.insert(SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Buffer,
        index: 0,
    });
    let mut addressed = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let plain = compile(&bc, "store_const", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(!plain.interface.needs_pointer_translation, "{stem}");
        let Ok(a) = compile_with(&bc, "store_const", &p, &inputs) else {
            // No buffer device address on this profile: the variant is refused (typed).
            continue;
        };
        addressed += 1;
        assert!(a.interface.needs_pointer_translation, "{stem}");
        let l = link(&[&a], &LinkInputs::compute(), &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(l.interface.needs_pointer_translation, "{stem}");
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
    }
    assert!(addressed > 0, "some committed profile has device addresses");
}

/// The fixture's vertex function compiles fixed function for the descriptor's attributes.
#[test]
fn the_projection_of_the_fixture_is_fixed_function() {
    let Some(bc) = assemble("stage_io.ll") else {
        return;
    };
    let p = prof("moltenvk-m3");
    let v = vertex_under(&bc, &vertex_descriptor(), &p).expect("vertex");
    let attrs = v.interface.io.attributes();
    assert_eq!(attrs.len(), 4);
    // Every attribute is fixed function; attribute 1's `Half2` is the one narrow-float format,
    // whose widening a `float` parameter would canonicalize (this one reads it as `half2`).
    for (a, (_, f)) in &attrs {
        assert_eq!(
            *f,
            VertexFetch::FixedFunction {
                narrow_float: *a == 1
            },
            "attribute {a}"
        );
    }
}

/// A fragment function's write to a half-float channel records the device's NaN encoding
/// (`PrecisionNote::DeviceNanEncoding`) only where the profile's driver is not measured to store
/// Metal's NaN: a `float` output's everywhere but MoltenVK, lavapipe and NVIDIA's measured GPUs, a
/// `half` output's everywhere but NVIDIA's measured GPUs and MoltenVK, which writes it through a
/// 16-bit output. A write to an `RGBA8Unorm` attachment records none. Where the function selects
/// the attachment's rules at the draw, the pipeline keeps the note its rule word selects
/// (`color_writes::pipeline_writes`).
#[test]
fn half_float_writes_note_the_nan_encoding_where_the_device_rewrites_it() {
    use crate::pipeline::artifact::PrecisionNote;
    let Some(bc) = assemble("rt6_samples.ll") else {
        return;
    };
    let note = PrecisionNote::DeviceNanEncoding;
    for (stem, float_note, half_note) in [
        ("lavapipe", false, true),
        ("gate-lavapipe-x86_64", false, true),
        ("amd-rdna3-radv", true, true),
        ("intel-irisxe-rplp-anv", true, true),
        ("nvidia-rtx50-linux", true, true),
        ("nvidia-rtx5080-laptop", false, false),
        ("moltenvk-m3", false, false),
        ("moltenvk-m4", false, false),
    ] {
        let p = prof(stem);
        let v = compile(&bc, "rt6_fill_vertex", &p).expect("vertex");
        for (entry, noted) in [
            ("rt6_fill_rgba32float", float_note),
            ("rt6_fill_rgba16float", half_note),
        ] {
            for (pf, want) in [
                (PixelFormat::RGBA16Float, noted),
                (PixelFormat::RGBA8Unorm, false),
            ] {
                let what = format!("{stem} {entry} {pf:?}");
                let mut r = RenderLinkInputs::new();
                r.color_attachments[0] = Some(ColorAttachmentDescriptor::new(pf));
                let f = compile_with(&bc, entry, &p, &fragment_inputs(&r, &p))
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                let l = link(&[&v, &f], &render(&r), &p).unwrap_or_else(|e| panic!("{what}: {e}"));
                let g = l.graphics.as_ref().expect("graphics");
                let (notes, _) = crate::pipeline::color_writes::pipeline_writes(
                    &f,
                    &g.color_write_rules,
                    &g.color_attachments,
                    g.alpha_to_coverage,
                )
                .unwrap_or_else(|| panic!("{what}: no rule word"));
                assert_eq!(notes.contains(&note), want, "{what}: {notes:?}");
            }
        }
    }
}

/// Where the profile's driver is measured to store a 16-bit output's bits
/// (`DriverDeviation::HalfColorOutputBits`) and to convert to 16 bits as Metal does
/// (`MetalHalfConversion`), declares 16-bit outputs (with `Int16` and `Float16`), and does not
/// store the output's NaN as Metal does, an output to a half-float attachment is written through a
/// 16-bit output variable as the bits of the half Metal stores, valid and linked: on MoltenVK (the
/// M3 Max and the M4) a `half` output (it stores a `float`'s NaN as Metal does). Such an output
/// rounds nothing, so the attachment is listed as written as a half's bits, not as rounded, takes
/// no Vulkan blend of one and zero, and the module requires both properties. RADV and ANV, which
/// declare 16-bit outputs but store their NaNs as `0xfe00` too, keep the 32-bit output, as do
/// lavapipe and NVIDIA, which declare none. An `RGBA8Unorm` attachment keeps the 32-bit output.
#[test]
fn half_float_attachments_take_the_halfs_bits_through_a_16_bit_output() {
    let Some(bc) = assemble("rt6_samples.ll") else {
        return;
    };
    for (stem, float_route, half_route) in [
        ("amd-rdna3-radv", false, false),
        ("intel-irisxe-rplp-anv", false, false),
        ("moltenvk-m3", false, true),
        ("moltenvk-m4", false, true),
        ("lavapipe", false, false),
        ("nvidia-rtx5080-laptop", false, false),
    ] {
        let p = prof(stem);
        let v = compile(&bc, "rt6_fill_vertex", &p).expect("vertex");
        for (entry, route) in [
            ("rt6_fill_rgba32float", float_route),
            ("rt6_fill_rgba16float", half_route),
        ] {
            for (pf, want) in [
                (PixelFormat::RGBA16Float, route),
                (PixelFormat::R16Float, route),
                (PixelFormat::RGBA8Unorm, false),
            ] {
                let what = format!("{stem} {entry} {pf:?}");
                let mut r = RenderLinkInputs::new();
                r.color_attachments[0] = Some(ColorAttachmentDescriptor::new(pf));
                let f = compile_with(&bc, entry, &p, &fragment_inputs(&r, &p))
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                let l = link(&[&v, &f], &render(&r), &p).unwrap_or_else(|e| panic!("{what}: {e}"));
                let words = validate_all(&l, &[&v, &f], &p);
                let text = spirv_dis(&words[1]);
                assert_eq!(
                    text.contains("OpCapability StorageInputOutput16"),
                    want,
                    "{what}\n{text}"
                );
                if want {
                    assert!(
                        text.contains("%_ptr_Output_v4half = OpTypePointer Output %v4half"),
                        "{what}\n{text}"
                    );
                    assert!(f.interface.io.half_bit_colors.contains(&0), "{what}");
                    assert!(!f.interface.io.rounded_colors.contains(&0), "{what}");
                    for property in [
                        crate::device::requirements::VkProperty::HalfColorOutputBits,
                        crate::device::requirements::VkProperty::MetalHalfConversion,
                    ] {
                        assert!(
                            f.requirements.properties.contains(&property),
                            "{what}: {property:?}"
                        );
                    }
                    let mut blended = r.clone();
                    if let Some(a) = blended.color_attachments[0].as_mut() {
                        a.blending_enabled = true;
                    }
                    let b = compile_with(&bc, entry, &p, &fragment_inputs(&blended, &p))
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                    let l = link(&[&v, &b], &render(&blended), &p)
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                    let attachment = l.graphics.as_ref().expect("graphics").color_attachments[0]
                        .as_ref()
                        .expect("attachment");
                    assert!(attachment.blend.is_none(), "{what} blended");
                }
            }
        }
    }
}

/// Under `TextureWriteRoundingMode::NearestEven` a `float` output to a half-float attachment rounds
/// to nearest even in the function on every profile, MoltenVK's included (listed as rounded, no
/// device conversion noted, the device's blend of one and zero dropped), and the pipeline links
/// valid; a `half` output, already a half, rounds nothing (on MoltenVK it is written as a half's
/// bits). Under alpha-to-coverage the alpha rounds too, and a pipeline whose function rounds it
/// records `PrecisionNote::CoverageReadsRoundedAlpha`. After a blend of other factors the device
/// rounds the blend's result, which Metal rounds to nearest even: the channels are written as
/// computed and the module announces `PrecisionNote::BlendedWriteRounding` (no device conversion
/// noted), MoltenVK's too.
#[test]
fn half_attachments_round_to_nearest_even_under_mode_two() {
    use crate::metal::{BlendFactor, TextureWriteRoundingMode};
    use crate::pipeline::artifact::PrecisionNote;
    use crate::pipeline::color_writes::HALF_NEAREST;
    let Some(bc) = assemble("rt6_samples.ll") else {
        return;
    };
    for stem in [
        "moltenvk-m3",
        "moltenvk-m4",
        "lavapipe",
        "amd-780m-radv",
        "nvidia-rtx5080-laptop",
    ] {
        let p = prof(stem);
        let v = compile(&bc, "rt6_fill_vertex", &p).expect("vertex");
        let mut r = RenderLinkInputs::new();
        r.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float));
        r.texture_write_rounding_mode = TextureWriteRoundingMode::NearestEven;
        let mut covered = r.clone();
        covered.alpha_to_coverage = true;
        for (entry, r) in [
            ("rt6_fill_rgba32float", &r),
            ("rt6_fill_rgba16float", &r),
            ("rt6_fill_rgba32float", &covered),
            ("rt6_fill_rgba16float", &covered),
        ] {
            let what = format!("{stem} {entry} coverage {}", r.alpha_to_coverage);
            let f = compile_with(&bc, entry, &p, &fragment_inputs(r, &p))
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            assert_eq!(
                f.interface.io.color_writes.get(&0),
                Some(&[Some(HALF_NEAREST); 4]),
                "{what}"
            );
            let float = entry == "rt6_fill_rgba32float";
            assert_eq!(f.interface.io.rounded_colors.contains(&0), float, "{what}");
            assert!(
                !f.precision.contains(&PrecisionNote::DeviceColorConversion),
                "{what}: {:?}",
                f.precision
            );
            let l = link(&[&v, &f], &render(r), &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            validate_all(&l, &[&v, &f], &p);
            let g = l.graphics.as_ref().expect("graphics");
            let (notes, _) = crate::pipeline::color_writes::pipeline_writes(
                &f,
                &g.color_write_rules,
                &g.color_attachments,
                g.alpha_to_coverage,
            )
            .unwrap_or_else(|| panic!("{what}: no rule word"));
            assert_eq!(
                notes.contains(&PrecisionNote::CoverageReadsRoundedAlpha),
                float && r.alpha_to_coverage,
                "{what}: {notes:?}"
            );
        }
        let mut blended = r.clone();
        if let Some(a) = blended.color_attachments[0].as_mut() {
            a.blending_enabled = true;
            a.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
        }
        let f = compile_with(
            &bc,
            "rt6_fill_rgba32float",
            &p,
            &fragment_inputs(&blended, &p),
        )
        .unwrap_or_else(|e| panic!("{stem} blended: {e}"));
        assert!(
            f.precision.contains(&PrecisionNote::BlendedWriteRounding)
                && !f.precision.contains(&PrecisionNote::DeviceColorConversion),
            "{stem} blended: {:?}",
            f.precision
        );
        assert!(
            !f.interface.io.rounded_colors.contains(&0),
            "{stem} blended"
        );
        let l = link(&[&v, &f], &render(&blended), &p)
            .unwrap_or_else(|e| panic!("{stem} blended: {e}"));
        validate_all(&l, &[&v, &f], &p);
    }
}

/// A fragment function's `float` texture write that Apple's `textureWriteRoundingMode` 2 stores
/// otherwise (owned fixture `tob_renders`: `tob_write` into a 2D texture, `tob_write_buffer` into a
/// texture buffer) records the mode it compiled under in `StageIo::texture_write_rounding` where
/// its slot's storage choice declares no format (none at all, or `UNDEFINED`), a half-float format
/// (by the profile's class, and rounding toward zero) or a packed small-float one, and nothing
/// where it declares another (`R8G8B8A8_UNORM`). At the default mode such a stage links, valid.
/// Under `NearestEven` one whose choice declares such a format is refused naming
/// `PrivateRenderField::TextureWriteRounding`, and one whose bound texture decides states
/// `RuntimeFact::StorageTextureWriteRounding` (`halves`: a `float` write) and links, valid, for the
/// binder to refuse such a texture at the draw. A stage compiled under the other mode does not link
/// (`NeedsInput(ColorWrites)`). A stage that records nothing, and a fragment function that writes
/// no texture (`tob_solid`), compile and link alike at either mode.
#[test]
fn mode_two_refuses_fragment_texture_writes_it_would_round() {
    use crate::error::PrivateRenderField;
    use crate::metal::TextureWriteRoundingMode as Mode;
    use crate::pipeline::facts::RuntimeFact;
    use crate::pipeline::lower_inputs::{StorageImageChoice, StorageImageClass};
    let Some(bc) = assemble("tob_renders.ll") else {
        return;
    };
    let slot = SlotRef {
        stage: Stage::Fragment,
        class: SlotClass::Texture,
        index: 0,
    };
    let refused = |e: &Error| {
        matches!(e, Error::Unsupported(u) if u.feature
            == Feature::PrivateRenderState(PrivateRenderField::TextureWriteRounding)
            && u.needs == Needs::NotImplemented)
    };
    let stated = RuntimeFact::StorageTextureWriteRounding { slot, halves: true };
    let choice = |f: vk::Format, c: StorageImageClass| Some(StorageImageChoice::new(f, c));
    let identity = StorageImageClass::Identity;
    // Where the write goes: the bound texture decides, the choice declares such a format, or
    // another.
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Goes {
        Bound,
        Declared,
        Other,
    }
    for stem in ["moltenvk-m4", "lavapipe", "amd-780m-radv"] {
        let p = prof(stem);
        let v = compile(&bc, "tob_rect", &p).expect("vertex");
        let mut r = RenderLinkInputs::new();
        r.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        let mut two = r.clone();
        two.texture_write_rounding_mode = Mode::NearestEven;
        let cases = [
            ("tob_write", None, Goes::Bound),
            (
                "tob_write",
                choice(vk::Format::UNDEFINED, identity),
                Goes::Bound,
            ),
            // The class the profile writes a half-float texture by (`Identity` where its device
            // is measured to store Metal's halves).
            (
                "tob_write",
                choice(
                    vk::Format::R16G16B16A16_SFLOAT,
                    crate::metal::format::half_write_class(&p),
                ),
                Goes::Declared,
            ),
            (
                "tob_write",
                choice(
                    vk::Format::R16G16B16A16_SFLOAT,
                    StorageImageClass::HalfTowardZero,
                ),
                Goes::Declared,
            ),
            (
                "tob_write",
                choice(vk::Format::B10G11R11_UFLOAT_PACK32, identity),
                Goes::Declared,
            ),
            (
                "tob_write",
                choice(vk::Format::R8G8B8A8_UNORM, identity),
                Goes::Other,
            ),
            ("tob_write_buffer", None, Goes::Bound),
            ("tob_solid", None, Goes::Other),
        ];
        for (entry, choice, goes) in cases {
            let what = format!("{stem} {entry} {choice:?}");
            let inputs = |r: &RenderLinkInputs| {
                let mut inputs = fragment_inputs(r, &p);
                if let Some(c) = choice {
                    inputs.storage_images.insert(slot, c);
                }
                inputs
            };
            // The default mode: recorded where it may be such, linked, valid.
            let f =
                compile_with(&bc, entry, &p, &inputs(&r)).unwrap_or_else(|e| panic!("{what}: {e}"));
            assert_eq!(
                f.interface.io.texture_write_rounding,
                (goes != Goes::Other).then_some(Mode::TowardZero),
                "{what}"
            );
            assert!(!f.facts.contains(&stated), "{what}");
            let l = link(&[&v, &f], &render(&r), &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            validate_all(&l, &[&v, &f], &p);
            // Compiled under the default's mode, into a pipeline of mode 2.
            match link(&[&v, &f], &render(&two), &p) {
                Err(e) if goes != Goes::Other => {
                    assert_eq!(e, Error::NeedsInput(RuntimeInput::ColorWrites), "{what}")
                }
                Ok(_) if goes == Goes::Other => {}
                other => panic!("{what}: {other:?}"),
            }
            // Mode 2: refused where declared, stated where the bound texture decides.
            match (compile_with(&bc, entry, &p, &inputs(&two)), goes) {
                (Err(e), Goes::Declared) => assert!(refused(&e), "{what}: {e}"),
                (Ok(f), Goes::Bound | Goes::Other) => {
                    let bound = goes == Goes::Bound;
                    assert_eq!(
                        f.interface.io.texture_write_rounding,
                        bound.then_some(Mode::NearestEven),
                        "{what}"
                    );
                    assert_eq!(f.facts.contains(&stated), bound, "{what}");
                    let l = link(&[&v, &f], &render(&two), &p)
                        .unwrap_or_else(|e| panic!("{what}: mode 2: {e}"));
                    validate_all(&l, &[&v, &f], &p);
                }
                (other, _) => panic!("{what}: mode 2: {:?}", other.map(|_| ())),
            }
        }
    }
}

/// Where the profile's driver is measured to convert a `float` output to a half-float or unsigned
/// small float channel as Metal does (`MetalSmallFloatColorWrites`: RADV on the Radeon 780M, ANV on
/// the Iris Xe of RPL-P, NVIDIA's on the RTX 3090), or an output to a 16-bit unorm or snorm channel
/// (`MetalNorm16ColorWrites`: the 780M), the fragment function writes it as computed: the
/// attachment is not listed as rounded (its blend of one and zero kept), the module requires the
/// property, and it links valid. Only a NaN is chosen, as `0x7fc00000` where the device stores that
/// as Metal's NaN and its own NaN would not be (an unsigned small float, and a half-float channel
/// on the 3090), and as -1.0 for a blended snorm channel, so `OpIsNan` appears exactly there.
/// Elsewhere (the UHD 770 of RPL-S, the RTX 5080 Laptop GPU, lavapipe; a `half` output to
/// `RG11B10Float`) the output rounds, listed, requiring neither. Each holds of the channels' own
/// rules (`color_writes::exact_writes`), and where the function selects the attachment's rules at
/// the draw the pipeline lists the attachment, keeps its blend and requires the property alike.
#[test]
fn measured_color_conversions_are_left_to_the_device() {
    use crate::device::requirements::VkProperty;
    let Some(bc) = assemble("rt6_samples.ll") else {
        return;
    };
    let small = VkProperty::MetalSmallFloatColorWrites;
    let norm16 = VkProperty::MetalNorm16ColorWrites;
    // The RTX 3090 (not a committed capture): the RTX 5080 Laptop GPU's capture with its id.
    let rtx_3090 = {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../metal2vulkan-validation/profiles/nvidia-rtx5080-laptop.json");
        let text = std::fs::read_to_string(path).expect("profile");
        let mut d = DeviceDescription::from_khronos_json(&text, None).expect("description");
        d.device_id = crate::device::driver_facts::NVIDIA_RTX_3090;
        Device::new(d, DevicePolicy::default())
            .expect("device")
            .profile()
            .clone()
    };
    for (stem, p, small_taken, norm16_taken, nan_selected) in [
        ("amd-780m-radv", prof("amd-780m-radv"), true, true, false),
        (
            "intel-irisxe-rplp-anv",
            prof("intel-irisxe-rplp-anv"),
            true,
            false,
            false,
        ),
        ("rtx-3090", rtx_3090, true, false, true),
        (
            "intel-uhd770-rpls-anv",
            prof("intel-uhd770-rpls-anv"),
            true,
            false,
            false,
        ),
        (
            "nvidia-rtx5080-laptop",
            prof("nvidia-rtx5080-laptop"),
            false,
            false,
            false,
        ),
        ("lavapipe", prof("lavapipe"), false, false, false),
    ] {
        let v = compile(&bc, "rt6_fill_vertex", &p).expect("vertex");
        for (entry, float) in [
            ("rt6_fill_rgba32float", true),
            ("rt6_fill_rgba16float", false),
        ] {
            for (pf, property, taken, nan) in [
                (
                    PixelFormat::RGBA16Float,
                    small,
                    small_taken && float,
                    nan_selected,
                ),
                (
                    PixelFormat::R16Float,
                    small,
                    small_taken && float,
                    nan_selected,
                ),
                (PixelFormat::RG11B10Float, small, small_taken && float, true),
                (PixelFormat::RGBA16Unorm, norm16, norm16_taken, false),
                (PixelFormat::RG16Snorm, norm16, norm16_taken, false),
            ] {
                for blended in [false, true] {
                    let what = format!("{stem} {entry} {pf:?} blended {blended}");
                    let mut r = RenderLinkInputs::new();
                    let mut a = ColorAttachmentDescriptor::new(pf);
                    a.blending_enabled = blended;
                    r.color_attachments[0] = Some(a);
                    let mut exact = fragment_inputs(&r, &p);
                    exact.color_writes = crate::pipeline::color_writes::exact_writes(
                        [0],
                        &r.color_attachments,
                        (r.alpha_to_coverage, r.texture_write_rounding_mode),
                        &p,
                    );
                    let f = compile_with(&bc, entry, &p, &exact)
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                    let l =
                        link(&[&v, &f], &render(&r), &p).unwrap_or_else(|e| panic!("{what}: {e}"));
                    let words = validate_all(&l, &[&v, &f], &p);
                    let text = spirv_dis(&words[1]);
                    // A `half` output to a half-float channel is a half already: nothing rounds.
                    let held =
                        !float && matches!(pf, PixelFormat::RGBA16Float | PixelFormat::R16Float);
                    let listed = f.interface.io.rounded_colors.contains(&0);
                    assert_eq!(listed, !taken && !held, "{what}");
                    assert_eq!(
                        f.requirements.properties.contains(&property),
                        taken,
                        "{what}"
                    );
                    let blend = l.graphics.as_ref().expect("graphics").color_attachments[0]
                        .as_ref()
                        .expect("attachment")
                        .blend
                        .is_some();
                    assert_eq!(blend, blended && !listed, "{what}");
                    if taken {
                        let snorm = blended && pf == PixelFormat::RG16Snorm;
                        assert_eq!(text.contains("OpIsNan"), nan || snorm, "{what}\n{text}");
                    }
                    // The pipeline's map, whose attachment may select its rules at the draw.
                    let s = compile_with(&bc, entry, &p, &fragment_inputs(&r, &p))
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                    let l =
                        link(&[&v, &s], &render(&r), &p).unwrap_or_else(|e| panic!("{what}: {e}"));
                    validate_all(&l, &[&v, &s], &p);
                    let g = l.graphics.as_ref().expect("graphics");
                    let (_, rounded) = crate::pipeline::color_writes::pipeline_writes(
                        &s,
                        &g.color_write_rules,
                        &g.color_attachments,
                        g.alpha_to_coverage,
                    )
                    .unwrap_or_else(|| panic!("{what}: no rule word"));
                    assert_eq!(rounded.contains(&0), listed, "{what} selected");
                    assert_eq!(
                        s.requirements.properties.contains(&property),
                        taken,
                        "{what} selected"
                    );
                    let attachment = g.color_attachments[0].as_ref().expect("attachment");
                    assert_eq!(attachment.blend.is_some(), blend, "{what} selected");
                }
            }
        }
    }
}

/// A render pipeline's graphics state states what its vertex function writes, read from the
/// compiled AIR: `ob3_index_buffer`, which stores to a device buffer, writes memory and no point
/// size; `rt6_point`, which writes `[[point_size]]`, writes a point size and no memory; `rt6_full`
/// writes neither, its module's default point size of 1 included. The two that return a position
/// link with and without a fragment function (`rt6_white`); `ob3_index_buffer`, which returns
/// nothing, links alone, as a rasterization-disabled draw takes it. On a desktop and a MoltenVK
/// profile.
#[test]
fn the_graphics_state_states_what_the_vertex_function_writes() {
    let (Some(ob3), Some(rt6)) = (
        assemble("ob3_index_buffer.ll"),
        assemble("rt6_render_state.ll"),
    ) else {
        return;
    };
    for stem in ["lavapipe", "moltenvk-m3"] {
        let p = prof(stem);
        let mut r = RenderLinkInputs::new();
        r.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        let white = compile_with(&rt6, "rt6_white", &p, &fragment_inputs(&r, &p))
            .unwrap_or_else(|e| panic!("{stem} rt6_white: {e}"));
        for (bc, entry, positioned, memory, point_size) in [
            (&ob3, "ob3_index_buffer", false, true, false),
            (&rt6, "rt6_point", true, false, true),
            (&rt6, "rt6_full", true, false, false),
        ] {
            let what = format!("{stem} {entry}");
            let v = compile(bc, entry, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            let mut shapes = vec![vec![&*v]];
            if positioned {
                shapes.push(vec![&*v, &*white]);
            }
            for stages in shapes {
                let graphics = link(&stages, &render(&r), &p)
                    .unwrap_or_else(|e| panic!("{what}: link: {e}"))
                    .graphics
                    .expect("graphics");
                let shape = format!("{what} ({} stages)", stages.len());
                assert_eq!(graphics.vertex_writes_memory, memory, "{shape}");
                assert_eq!(graphics.writes_point_size, point_size, "{shape}");
            }
        }
    }
}
