//! Post-tessellation vertex functions as Vulkan tessellation evaluation shaders.
//!
//! The owned fixture `engine/tessellation` has a quad patch that reads its control points from
//! a buffer by `[[patch_id]]`, a triangle patch that reads `patch_control_point` fields, a
//! per-patch attribute, and `[[instance_id]]`, a control point at a computed index, and a
//! `[[point_size]]` output. Each compiles to a `TessellationEvaluation` entry with its patch domain
//! and passes L3 (`spirv-val` and requirements within the profile) on every committed profile
//! with `tessellationShader`; the artifact states the evaluation interface the pipeline's control
//! stage must meet, and every attribute it reads is pulled from its source buffer. No GPU is used.
//!
//! `llvm-as` assembles the fixture; a missing tool skips visibly.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::bind::{
    BindState, BoundBuffer, DrawCall, IndexBufferRef, IndirectRef, StageTable,
    TessellationFactorBuffer, TessellationFactors, UseTarget, WriteSource,
};
use crate::budget::{Budget, SPIRV_WORDS};
use crate::device::requirements::{VkFeature, VkLimit};
use crate::device::{Device, DeviceDescription, DevicePolicy, DeviceProfile, DriverWorkaround};
use crate::error::{
    BindError, Capability, DescriptorError, Error, Feature, LowerFamily, Needs, RuntimeInput,
};
use crate::library::{FunctionArtifact, FunctionConstantValues, Library};
use crate::metal::{
    TessellationControlPointIndexType, TessellationFactorStepFunction, TessellationPartitionMode,
    VertexFormat, VertexStepFunction, Winding,
};
use crate::pipeline::artifact::{PrecisionNote, RenderPipeline, StageModule};
use crate::pipeline::compile::{compile_bitcode, EntryRef};
use crate::pipeline::descriptor::{
    FunctionDescriptor, RenderPipelineDescriptor, StageSource, TessellationDescriptor,
    VertexAttributeDescriptor, VertexBufferLayoutDescriptor, VertexDescriptor,
};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::{PatchDomain, TessellationStage};
use crate::pipeline::layout::BindingSource;
use crate::pipeline::link::tessellation::patch_fetch_projection;
use crate::pipeline::link::{link, LinkInputs};
use crate::pipeline::lower_inputs::{LoweringInputs, VertexFetch, VertexFetchProjection};
use crate::pipeline::stage_io::IoNumeric;
use crate::pipeline::sym::{PushField, Stage, SyntheticBinding};
use crate::pipeline::testing::{
    engine_dir, link_and_validate, llvm_assemble, meter, profile, AssembleError,
};
use crate::spirv::testing::{check_words_valid, committed_profiles};
use crate::vk;

fn fixture() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join("tessellation.ll")) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected tessellation.ll: {why}"),
    }
}

/// Bitcode of AIR text `text`, assembled in a scratch directory the call removes.
fn assemble_text(name: &str, text: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-tessellation-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("t.ll");
    std::fs::write(&ll, text).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    match r {
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

fn compile(
    bc: &[u8],
    entry: &str,
    p: &DeviceProfile,
    proj: Option<VertexFetchProjection>,
) -> Result<Arc<FunctionArtifact>, Error> {
    let mut inputs = LoweringInputs::new();
    inputs.vertex_fetch = proj;
    compile_bitcode(
        bc,
        EntryRef::Name(entry),
        &FunctionConstantValues::defaults(),
        p,
        &inputs,
        &mut meter(),
        SPIRV_WORDS,
    )
}

/// The fixture's vertex descriptor: the control points' position, uv, and color from buffer 0
/// (24-byte elements, per control point) and the tint from buffer 1 (per patch).
fn descriptor() -> VertexDescriptor {
    let mut d = VertexDescriptor::new();
    for (a, format, offset, buffer) in [
        (0, VertexFormat::Float4, 0, 0),
        (1, VertexFormat::Half2, 16, 0),
        (2, VertexFormat::UChar4, 20, 0),
        (5, VertexFormat::Float4, 0, 1),
    ] {
        d.attributes
            .insert(a, VertexAttributeDescriptor::new(format, offset, buffer));
    }
    let mut points = VertexBufferLayoutDescriptor::new(Some(24));
    points.step_function = VertexStepFunction::PerPatchControlPoint;
    let mut patches = VertexBufferLayoutDescriptor::new(Some(16));
    patches.step_function = VertexStepFunction::PerPatch;
    d.layouts.insert(0, points);
    d.layouts.insert(1, patches);
    d
}

/// The projection of every attribute of [`descriptor`].
fn projection() -> VertexFetchProjection {
    let classes = BTreeMap::from([
        (0, IoNumeric::Float),
        (1, IoNumeric::Float),
        (2, IoNumeric::Uint),
        (5, IoNumeric::Float),
    ]);
    patch_fetch_projection(Some(&descriptor()), &classes).expect("projection")
}

/// Committed profiles by stem.
type Profiles = Vec<(String, DeviceProfile)>;

/// The committed profiles with and without `tessellationShader`.
fn profiles() -> (Profiles, Profiles) {
    committed_profiles()
        .expect("committed profiles")
        .into_iter()
        .partition(|(_, p)| p.fields().fragment.tessellation_shader)
}

/// The operands of every instruction of opcode `op` in `words` (after the header).
fn instructions(words: &[u32], op: crate::spv::Op) -> Vec<Vec<u32>> {
    let mut out = Vec::new();
    let mut at = 5;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        if count == 0 {
            break;
        }
        if words[at] & 0xffff == op as u32 {
            out.push(words[at + 1..at + count].to_vec());
        }
        at += count;
    }
    out
}

/// The module is a `TessellationEvaluation` entry of `mode`'s domain, and each of `patch`'s
/// locations is a `Patch` input.
fn assert_evaluation(words: &[u32], mode: crate::spv::ExecutionMode, patch: &[u32]) {
    let entries = instructions(words, crate::spv::Op::EntryPoint);
    assert_eq!(entries.len(), 1);
    assert_eq!(
        entries[0][0],
        crate::spv::ExecutionModel::TessellationEvaluation as u32
    );
    let modes: Vec<u32> = instructions(words, crate::spv::Op::ExecutionMode)
        .iter()
        .map(|m| m[1])
        .collect();
    assert!(modes.contains(&(mode as u32)), "{modes:?}");
    let decorations = instructions(words, crate::spv::Op::Decorate);
    for location in patch {
        let at_location = |var: u32| {
            decorations.iter().any(|d| {
                d[0] == var && d[1] == crate::spv::Decoration::Location as u32 && d[2] == *location
            })
        };
        assert!(
            decorations
                .iter()
                .any(|d| d[1] == crate::spv::Decoration::Patch as u32 && at_location(d[0])),
            "no patch input at location {location}"
        );
    }
}

/// `[[position_in_patch]]` is `TessCoord`, `[[patch_id]]` the patch input at location 0, and the
/// module needs `tessellationShader`; on a profile without it the function is an emulated vertex
/// entry.
#[test]
fn a_quad_patch_is_an_evaluation_shader_on_every_profile_with_tessellation() {
    let Some(bc) = fixture() else { return };
    let (with, without) = profiles();
    assert!(!with.is_empty() && !without.is_empty());
    for (stem, p) in &with {
        let a = compile(&bc, "tess_quad_buffer", p, None).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(
            a.interface.tessellation,
            Some(TessellationStage::new(PatchDomain::Quad, 4)),
            "{stem}"
        );
        assert_eq!(a.interface.stage, Stage::Vertex);
        assert_eq!(
            a.interface.vk_stage(),
            vk::ShaderStage::TESSELLATION_EVALUATION
        );
        assert!(a
            .requirements
            .features
            .contains(&VkFeature::TessellationShader));
        let (_, words) = link_and_validate(&a, p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_evaluation(
            &words,
            crate::spv::ExecutionMode::Quads,
            &[TessellationStage::PATCH_INDEX_LOCATION],
        );
    }
    // Without it, the function is a plain vertex entry that draws the tessellator's triangles
    // itself, baking in Metal's default tessellation state.
    for (stem, p) in &without {
        let a = compile(&bc, "tess_quad_buffer", p, None).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(
            a.interface.tessellation,
            Some(
                TessellationStage::new(PatchDomain::Quad, 4)
                    .emulated(TessellationDescriptor::default())
            ),
            "{stem}"
        );
        assert_eq!(a.interface.vk_stage(), vk::ShaderStage::VERTEX, "{stem}");
        assert!(!a
            .requirements
            .features
            .contains(&VkFeature::TessellationShader));
        let (_, words) = link_and_validate(&a, p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let entries = instructions(&words, crate::spv::Op::EntryPoint);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0][0], crate::spv::ExecutionModel::Vertex as u32);
    }
}

/// An evaluation shader has no fixed-function input, so a function that reads a patch or
/// control-point attribute needs the vertex descriptor.
#[test]
fn patch_attributes_need_the_vertex_descriptor() {
    let Some(bc) = fixture() else { return };
    let p = prof("moltenvk-m3");
    for entry in ["tess_triangle_points", "tess_dynamic_point"] {
        match compile(&bc, entry, &p, None) {
            Err(Error::NeedsInput(RuntimeInput::VertexDescriptor { attribute: 5 }))
                if entry == "tess_triangle_points" => {}
            Err(Error::NeedsInput(RuntimeInput::VertexDescriptor { attribute: 0 }))
                if entry == "tess_dynamic_point" => {}
            other => panic!("{entry}: {other:?}"),
        }
    }
    // An attribute the projection names fixed-function is not the evaluation shader's.
    let mut proj = projection();
    proj.attributes.insert(
        5,
        VertexFetch::FixedFunction {
            narrow_float: false,
        },
    );
    assert_eq!(
        compile(&bc, "tess_triangle_points", &p, Some(proj)).map(|_| ()),
        Err(Error::InvalidDescriptor(DescriptorError::VertexAttribute {
            attribute: 5
        }))
    );
}

/// Control-point fields at `patch · N + k` of their per-control-point layout, a per-patch attribute
/// at the patch index, and `[[instance_id]]` from its patch input; a field no instruction reads
/// (`color`) is not fetched.
#[test]
fn control_points_and_patch_inputs_are_pulled() {
    let Some(bc) = fixture() else { return };
    let (with, _) = profiles();
    for (stem, p) in &with {
        let a = compile(&bc, "tess_triangle_points", p, Some(projection()))
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(
            a.interface.tessellation,
            Some(TessellationStage::new(PatchDomain::Triangle, 3))
        );
        let pulled: Vec<u32> = a.interface.io.pulled.iter().map(|x| x.index).collect();
        assert_eq!(pulled, [0, 1, 5], "{stem}");
        for buffer in [0, 1] {
            assert!(
                a.interface
                    .push
                    .fields
                    .iter()
                    .any(|f| f.field == PushField::VertexBufferStride(buffer)),
                "{stem}: the stride of buffer {buffer}"
            );
            assert!(a
                .facts
                .contains(&RuntimeFact::VertexBufferStride { layout: buffer }));
        }
        let (_, words) = link_and_validate(&a, p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_evaluation(
            &words,
            crate::spv::ExecutionMode::Triangles,
            &[
                TessellationStage::PATCH_INDEX_LOCATION,
                TessellationStage::INSTANCE_LOCATION,
            ],
        );
        // A control point at a computed index.
        let a = compile(&bc, "tess_dynamic_point", p, Some(projection()))
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let pulled: Vec<u32> = a.interface.io.pulled.iter().map(|x| x.index).collect();
        assert_eq!(pulled, [0, 2], "{stem}");
        link_and_validate(&a, p).unwrap_or_else(|e| panic!("{stem}: {e}"));
    }
}

/// A control-point read is named by what its source buffers' slots are: descriptor views
/// (`tessellation.control_point`, under the profile's buffer encoding) or device addresses
/// (`tessellation.control_point.psb`, here a `BufferRangeLimit` variant's), which are checked in
/// the shader in every leg.
#[test]
fn a_control_point_read_is_named_by_the_memory_it_reaches() {
    use crate::device::profile::BufferAccess;
    use crate::pipeline::sym::{SlotClass, SlotRef};

    let Some(bc) = fixture() else { return };
    let (with, _) = profiles();
    let mut addressed_profiles = 0;
    for (stem, p) in with
        .iter()
        .filter(|(_, p)| p.fields().buffer_access == BufferAccess::Addressed)
    {
        let reads = |psb: bool| {
            let mut inputs = LoweringInputs::new();
            inputs.vertex_fetch = Some(projection());
            if psb {
                inputs.psb_slots = [0, 1]
                    .map(|index| SlotRef {
                        stage: Stage::Vertex,
                        class: SlotClass::Buffer,
                        index,
                    })
                    .into();
            }
            let a = compile_bitcode(
                &bc,
                EntryRef::Name("tess_triangle_points"),
                &FunctionConstantValues::defaults(),
                p,
                &inputs,
                &mut meter(),
                SPIRV_WORDS,
            )
            .unwrap_or_else(|e| panic!("{stem} psb={psb}: {e}"));
            a.strategies
                .entries
                .iter()
                .filter(|(name, _)| name.starts_with("tessellation.control_point"))
                .map(|(name, used)| (name.clone(), used.uses))
                .collect::<Vec<_>>()
        };
        let views = reads(false);
        let uses = match views.as_slice() {
            [(name, uses)] if name == "tessellation.control_point" => *uses,
            other => panic!("{stem}: {other:?}"),
        };
        assert_eq!(
            reads(true),
            [("tessellation.control_point.psb".to_owned(), uses)],
            "{stem}"
        );
        addressed_profiles += 1;
    }
    assert!(addressed_profiles > 0);
}

/// Control points used whole (merged in a phi) fetch every field Metal does not mark unused.
/// `tess_whole_point` reads no color, but Metal marks the field used, as the struct is; its
/// attribute is fetched.
#[test]
fn a_control_point_used_whole_fetches_its_used_fields() {
    let Some(bc) = fixture() else { return };
    let (with, _) = profiles();
    for (stem, p) in &with {
        let a = compile(&bc, "tess_whole_point", p, Some(projection()))
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let pulled: Vec<u32> = a.interface.io.pulled.iter().map(|x| x.index).collect();
        assert_eq!(pulled, [0, 1, 2], "{stem}");
        link_and_validate(&a, p).unwrap_or_else(|e| panic!("{stem}: {e}"));
    }
}

/// A field Metal marks unused (`air.arg_unused`) is zero and unfetched also when its struct is used
/// whole, so a descriptor may omit its attribute. `tess_triangle_points` with its third control
/// point's uv read through a `select` of two control points (a shape Metal's compiler does not emit
/// with the mark, derived here from the fixture's text) fetches no color.
#[test]
fn an_unused_field_of_a_control_point_used_whole_is_unfetched() {
    let text = std::fs::read_to_string(engine_dir().join("tessellation.ll")).expect("fixture");
    let ty = "{ <4 x float>, <2 x half>, <4 x i8> }";
    let read = format!("  %19 = extractvalue {ty} %14, 1\n");
    assert_eq!(text.matches(&read).count(), 1, "the fixture changed");
    let whole = format!(
        "  %cnd = icmp eq i32 %3, 7\n  %sel = select i1 %cnd, {ty} %9, {ty} %14\n  \
         %19 = extractvalue {ty} %sel, 1\n"
    );
    let Some(bc) = assemble_text("select", &text.replace(&read, &whole)) else {
        return;
    };
    let mut d = descriptor();
    d.attributes.remove(&2);
    let classes = BTreeMap::from([
        (0, IoNumeric::Float),
        (1, IoNumeric::Float),
        (5, IoNumeric::Float),
    ]);
    let proj = patch_fetch_projection(Some(&d), &classes).expect("projection");
    let (with, _) = profiles();
    for (stem, p) in &with {
        let a = compile(&bc, "tess_triangle_points", p, Some(proj.clone()))
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let pulled: Vec<u32> = a.interface.io.pulled.iter().map(|x| x.index).collect();
        assert_eq!(pulled, [0, 1, 5], "{stem}");
        link_and_validate(&a, p).unwrap_or_else(|e| panic!("{stem}: {e}"));
    }
}

/// A control-point field its function-constant predicate removes is not declared (MSL 5.8.1.3), so
/// Metal fetches nothing for it and defines no value for its read: the read is zero and unfetched,
/// and a descriptor may omit its attribute. `tess_triangle_points` reads its third control point's
/// `uv`; the fixture's text gates that field by a predicate global holding 0 or 1. Gated off, no
/// `uv` attribute is pulled; gated on, it is pulled as without the gate.
#[test]
fn a_gated_off_control_point_field_reads_zero_unfetched() {
    let text = std::fs::read_to_string(engine_dir().join("tessellation.ll")).expect("fixture");
    let node = "!20 = !{i32 0, !\"air.patch_control_point_input\", !21, !22, !23, !24}\n";
    assert_eq!(text.matches(node).count(), 1, "the fixture changed");
    let gated = "!20 = !{i32 0, !\"air.patch_control_point_input\", !21, !22, !90, !24}\n\
                 !90 = !{!\"air.function_constant\", !91, !\"air.location_index\", i32 1, i32 1, \
                 !\"air.arg_type_name\", !\"half2\", !\"air.arg_name\", !\"uv\"}\n\
                 !91 = !{ptr addrspace(2) @has_uv, !\"bool\", !\"has_uv\"}\n";
    let first = "; Function Attrs:";
    assert!(text.contains(first), "the fixture changed");
    let (with, _) = profiles();
    for (gate, pulled) in [(0, vec![0, 5]), (1, vec![0, 1, 5])] {
        let global =
            format!("@has_uv = internal addrspace(2) global i8 {gate}, align 1\n\n{first}");
        let edited = text.replacen(first, &global, 1).replace(node, gated);
        let Some(bc) = assemble_text("gated", &edited) else {
            return;
        };
        let mut d = descriptor();
        let mut classes = BTreeMap::from([
            (0, IoNumeric::Float),
            (1, IoNumeric::Float),
            (2, IoNumeric::Uint),
            (5, IoNumeric::Float),
        ]);
        if gate == 0 {
            d.attributes.remove(&1);
            classes.remove(&1);
        }
        let proj = patch_fetch_projection(Some(&d), &classes).expect("projection");
        for (stem, p) in &with {
            let a = compile(&bc, "tess_triangle_points", p, Some(proj.clone()))
                .unwrap_or_else(|e| panic!("{stem}, gate {gate}: {e}"));
            let got: Vec<u32> = a.interface.io.pulled.iter().map(|x| x.index).collect();
            assert_eq!(got, pulled, "{stem}, gate {gate}");
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{stem}, gate {gate}: {e}"));
        }
    }
}

/// A `ushort` `[[patch_id]]` or `[[instance_id]]` holds the low 16 bits of its 32-bit patch input
/// zero-extended: each load of the input is masked.
#[test]
fn ushort_patch_and_instance_indices_are_masked() {
    let Some(bc) = fixture() else { return };
    let (with, _) = profiles();
    for (stem, p) in &with {
        let a = compile(&bc, "tess_short_ids", p, None).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let (_, words) = link_and_validate(&a, p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let decorations = instructions(&words, crate::spv::Op::Decorate);
        let masks: Vec<u32> = instructions(&words, crate::spv::Op::Constant)
            .iter()
            .filter(|c| c.get(2) == Some(&0xffff))
            .map(|c| c[1])
            .collect();
        let ands = instructions(&words, crate::spv::Op::BitwiseAnd);
        let loads = instructions(&words, crate::spv::Op::Load);
        for location in [
            TessellationStage::PATCH_INDEX_LOCATION,
            TessellationStage::INSTANCE_LOCATION,
        ] {
            let decorated = |var: u32, d: crate::spv::Decoration, value: Option<u32>| {
                decorations.iter().any(|x| {
                    x[0] == var && x[1] == d as u32 && value.is_none_or(|v| x.get(2) == Some(&v))
                })
            };
            let var = decorations
                .iter()
                .map(|d| d[0])
                .find(|&v| {
                    decorated(v, crate::spv::Decoration::Patch, None)
                        && decorated(v, crate::spv::Decoration::Location, Some(location))
                })
                .unwrap_or_else(|| panic!("{stem}: no patch input at location {location}"));
            let loaded: Vec<u32> = loads.iter().filter(|l| l[2] == var).map(|l| l[1]).collect();
            assert!(
                !loaded.is_empty(),
                "{stem}: location {location} is never loaded"
            );
            for x in loaded {
                assert!(
                    ands.iter().any(|a| {
                        (a[2] == x && masks.contains(&a[3])) || (a[3] == x && masks.contains(&a[2]))
                    }),
                    "{stem}: a load of location {location} is not masked to 16 bits"
                );
            }
        }
    }
}

/// Link gives an evaluation stage's descriptor bindings and push constant range the evaluation
/// stage, the Vulkan stage its module is, not the vertex stage of its Metal argument tables.
#[test]
fn an_evaluation_stage_layout_names_the_evaluation_stage() {
    let Some(bc) = fixture() else { return };
    let (with, _) = profiles();
    let evaluation = vk::ShaderStage::TESSELLATION_EVALUATION.flags();
    for (stem, p) in &with {
        for (entry, proj) in [
            ("tess_quad_buffer", None),
            ("tess_triangle_points", Some(projection())),
        ] {
            let a = compile(&bc, entry, p, proj).unwrap_or_else(|e| panic!("{stem}: {e}"));
            let layout = link(&[&a], &LinkInputs::standalone(), p)
                .unwrap_or_else(|e| panic!("{stem} {entry}: {e}"))
                .layout;
            let bindings: Vec<_> = layout.sets.iter().flat_map(|s| &s.bindings).collect();
            assert!(
                !bindings.is_empty() || !layout.push_constants.is_empty(),
                "{stem} {entry}: an empty layout"
            );
            for b in bindings {
                assert_eq!(
                    b.stages, evaluation,
                    "{stem} {entry}: binding {}",
                    b.binding
                );
            }
            for r in &layout.push_constants {
                assert_eq!(r.stages, evaluation, "{stem} {entry}: push range");
            }
        }
    }
}

/// The tessellator emits only triangles, which no point size sizes, so a `[[point_size]]` output is
/// dropped exactly: no `PointSize` builtin (which an evaluation shader writes only with
/// `shaderTessellationAndGeometryPointSize`), and a valid evaluation shader on every profile with
/// tessellation, its position still written.
#[test]
fn a_point_size_output_is_dropped() {
    let Some(bc) = fixture() else { return };
    let (with, _) = profiles();
    assert!(!with.is_empty());
    for (stem, p) in &with {
        let a = compile(&bc, "tess_point_size", p, None).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(
            a.interface.tessellation,
            Some(TessellationStage::new(PatchDomain::Quad, 4)),
            "{stem}"
        );
        let (_, words) = link_and_validate(&a, p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_evaluation(&words, crate::spv::ExecutionMode::Quads, &[]);
        let builtins: Vec<u32> = instructions(&words, crate::spv::Op::Decorate)
            .iter()
            .filter(|d| d[1] == crate::spv::Decoration::BuiltIn as u32)
            .map(|d| d[2])
            .collect();
        assert!(
            !builtins.contains(&(crate::spv::BuiltIn::PointSize as u32)),
            "{stem}: {builtins:?}"
        );
        assert!(
            builtins.contains(&(crate::spv::BuiltIn::Position as u32)),
            "{stem}: {builtins:?}"
        );
    }
}

/// The projection pulls every attribute as its descriptor entry states it, and refuses a class
/// mismatch and a missing entry as a vertex function's projection does.
#[test]
fn the_patch_projection_pulls_every_attribute() {
    let d = descriptor();
    let proj = projection();
    for (a, attr) in &d.attributes {
        let layout = d.layouts.get(&attr.buffer_index).expect("layout");
        assert_eq!(
            proj.attributes.get(a),
            Some(&VertexFetch::pulled(attr, layout))
        );
    }
    let float_as_uint = BTreeMap::from([(0, IoNumeric::Uint)]);
    assert_eq!(
        patch_fetch_projection(Some(&d), &float_as_uint),
        Err(Error::Unsupported(
            crate::error::Unsupported::not_implemented(Feature::Lowering(
                LowerFamily::Tessellation
            ))
        ))
    );
    let missing = BTreeMap::from([(9, IoNumeric::Float)]);
    assert_eq!(
        patch_fetch_projection(Some(&d), &missing),
        Err(Error::InvalidDescriptor(DescriptorError::VertexAttribute {
            attribute: 9
        }))
    );
}

/// Our own fragment functions: one reading the evaluation stage's `color` varying, and one
/// reading `[[primitive_id]]` beside it.
const FRAGMENTS: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define <4 x float> @tess_fragment(<4 x float> %0) local_unnamed_addr #0 {
  ret <4 x float> %0
}

define <4 x float> @tess_fragment_primitive(<4 x float> %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = tail call fast float @air.convert.f.f32.u.i32(i32 %1)
  %4 = insertelement <4 x float> %0, float %3, i64 3
  ret <4 x float> %4
}

declare float @air.convert.f.f32.u.i32(i32) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.fragment = !{!9, !14}
!air.compile_options = !{!17, !18}
!air.version = !{!19}
!air.language_version = !{!20}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @tess_fragment, !10, !12}
!10 = !{!11}
!11 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!12 = !{!13}
!13 = !{i32 0, !"air.fragment_input", !"user(locn0)", !"air.center", !"air.perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"color"}
!14 = !{ptr @tess_fragment_primitive, !10, !15}
!15 = !{!13, !16}
!16 = !{i32 1, !"air.primitive_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"prim"}
!17 = !{!"air.compile.denorms_disable"}
!18 = !{!"air.compile.fast_math_enable"}
!19 = !{i32 2, i32 8, i32 0}
!20 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

fn device(stem: &str) -> Device {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::new(desc, DevicePolicy::default()).expect("device")
}

/// A render pipeline of post-tessellation function `vertex` of the fixture, with fragment
/// function `fragment` of [`FRAGMENTS`], the fixture's vertex descriptor, and `tessellation`.
fn tessellation_pipeline(
    bc: &[u8],
    fragments: &[u8],
    vertex: &str,
    fragment: Option<&str>,
    tessellation: Option<TessellationDescriptor>,
) -> RenderPipelineDescriptor {
    let lib = Library::from_air(bc).expect("library");
    let frag = Library::from_air(fragments).expect("fragment library");
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(
        lib.function(vertex).expect("vertex"),
    )));
    d.fragment = fragment.map(|f| {
        StageSource::Function(FunctionDescriptor::new(frag.function(f).expect("fragment")))
    });
    d.vertex_descriptor = Some(descriptor());
    d.tessellation = tessellation;
    d
}

/// The emulated pipeline `d` on `dev` (of committed profile `stem`, without
/// `tessellationShader`), or `None` where the device's vertex stage cannot bind the emulation's
/// buffers beside the function's: a typed refusal naming `maxPerStageDescriptorStorageBuffers`
/// at the device's own value. Any other refusal fails the test.
fn emulated(dev: &Device, d: &RenderPipelineDescriptor, stem: &str) -> Option<RenderPipeline> {
    match dev.compile_render_pipeline(d, &Budget::standard()) {
        Ok(p) => Some(p),
        Err(Error::Unsupported(u)) if u.feature == Feature::DescriptorCount => {
            let available = u64::from(
                dev.profile()
                    .fields()
                    .limits
                    .max_per_stage_descriptor_storage_buffers,
            );
            let Needs::AnyOf(any) = &u.needs else {
                panic!("{stem}: {u:?}");
            };
            assert!(
                matches!(
                    any.as_slice(),
                    [one] if matches!(
                        one.as_slice(),
                        [Capability::Limit {
                            limit: VkLimit::MaxPerStageDescriptorStorageBuffers,
                            available: a,
                            ..
                        }] if *a == available
                    )
                ),
                "{stem}: {u:?}"
            );
            None
        }
        Err(e) => panic!("{stem}: {e}"),
    }
}

/// The execution modes of the module `words`' entry.
fn modes(words: &[u32]) -> Vec<Vec<u32>> {
    instructions(words, crate::spv::Op::ExecutionMode)
        .into_iter()
        .map(|m| m[1..].to_vec())
        .collect()
}

/// A render pipeline of a post-tessellation vertex function links four stages (the synthesized
/// vertex and control stages, the evaluation stage, the fragment stage) into one `PATCH_LIST`
/// pipeline of one control point per patch: the control stage owns the factor and patch index
/// buffers' bindings, the control-point index buffer's for a control-point index type, and its push
/// range, declares the descriptor's modes, and every module passes L3 on every committed profile
/// with `tessellationShader`. Without it the pipeline is two stages, the post-tessellation function
/// emulated as a `TRIANGLE_LIST` vertex stage that owns the draw's work, records and index
/// buffers, stating the descriptor's state as its emulated tessellation.
#[test]
fn a_tessellation_pipeline_links_four_stages_on_every_profile_with_tessellation() {
    let Some(bc) = fixture() else { return };
    let Some(fragments) = assemble_text("fragments", FRAGMENTS) else {
        return;
    };
    let (with, without) = profiles();
    let quad = TessellationDescriptor::default();
    let triangle = TessellationDescriptor {
        partition_mode: TessellationPartitionMode::FractionalEven,
        max_tessellation_factor: 32,
        factor_step_function: TessellationFactorStepFunction::PerPatchAndPerInstance,
        output_winding_order: Winding::CounterClockwise,
        factor_scale_enabled: true,
        control_point_index_type: TessellationControlPointIndexType::UInt16,
        ..Default::default()
    };
    let cases = [
        (
            "tess_quad_buffer",
            quad,
            PatchDomain::Quad,
            4,
            [
                crate::spv::ExecutionMode::Quads,
                crate::spv::ExecutionMode::SpacingEqual,
                crate::spv::ExecutionMode::VertexOrderCcw,
            ],
        ),
        (
            "tess_triangle_points",
            triangle,
            PatchDomain::Triangle,
            3,
            [
                crate::spv::ExecutionMode::Triangles,
                crate::spv::ExecutionMode::SpacingFractionalEven,
                crate::spv::ExecutionMode::VertexOrderCw,
            ],
        ),
    ];
    // The device's tessellator orders the triangles on every profile with it but MoltenVK's on an
    // Apple GPU, which hands the patches to Metal's tessellator.
    let mut ordered = [0usize; 2];
    for (stem, _) in &with {
        let dev = device(stem);
        let own_order = dev
            .profile()
            .workarounds()
            .contains(&DriverWorkaround::TessellatorNotMetals);
        ordered[usize::from(own_order)] += 1;
        for (entry, t, domain, points, expected) in &cases {
            let d = tessellation_pipeline(&bc, &fragments, entry, Some("tess_fragment"), Some(*t));
            let p = dev
                .compile_render_pipeline(&d, &Budget::standard())
                .unwrap_or_else(|e| panic!("{stem} {entry}: {e}"));
            let stages: Vec<vk::ShaderStage> = p.stages.iter().map(|m| m.stage).collect();
            assert_eq!(
                stages,
                [
                    vk::ShaderStage::VERTEX,
                    vk::ShaderStage::TESSELLATION_CONTROL,
                    vk::ShaderStage::TESSELLATION_EVALUATION,
                    vk::ShaderStage::FRAGMENT
                ],
                "{stem} {entry}"
            );
            assert_eq!(p.graphics.topology, vk::PrimitiveTopology::PATCH_LIST);
            assert!(!p
                .graphics
                .dynamic_states
                .contains(&vk::DynamicState::PRIMITIVE_TOPOLOGY));
            let ts = p.graphics.tessellation.expect("tessellation state");
            assert_eq!(ts.patch_control_points, 1);
            assert_eq!((ts.domain, ts.control_points), (*domain, *points));
            assert_eq!(ts.factor_step_function, t.factor_step_function);
            assert_eq!(ts.control_point_index_type, t.control_point_index_type);
            let control = vk::ShaderStage::TESSELLATION_CONTROL.flags();
            let factors: Vec<_> = p
                .layout
                .sets
                .iter()
                .flat_map(|s| &s.bindings)
                .filter(|b| {
                    b.source
                        == BindingSource::Synthetic {
                            stage: Stage::Vertex,
                            binding: SyntheticBinding::TessellationFactors,
                        }
                })
                .collect();
            assert_eq!(factors.len(), 1, "{stem} {entry}");
            assert_eq!(factors[0].stages, control);
            assert_eq!(factors[0].ty, vk::DescriptorType::STORAGE_BUFFER);
            // The patch index buffer is the control stage's too.
            let indices: Vec<_> = p
                .layout
                .sets
                .iter()
                .flat_map(|s| &s.bindings)
                .filter(|b| {
                    b.source
                        == BindingSource::Synthetic {
                            stage: Stage::Vertex,
                            binding: SyntheticBinding::PatchIndices,
                        }
                })
                .collect();
            assert_eq!(indices.len(), 1, "{stem} {entry}");
            assert_eq!(indices[0].stages, control);
            assert_eq!(indices[0].ty, vk::DescriptorType::STORAGE_BUFFER);
            let points: Vec<_> = p
                .layout
                .sets
                .iter()
                .flat_map(|s| &s.bindings)
                .filter(|b| {
                    b.source
                        == BindingSource::Synthetic {
                            stage: Stage::Vertex,
                            binding: SyntheticBinding::ControlPointIndices,
                        }
                })
                .collect();
            let indexed = t.control_point_index_type != TessellationControlPointIndexType::None;
            assert_eq!(points.len(), usize::from(indexed), "{stem} {entry}");
            for b in points {
                assert_eq!(b.stages, control);
                assert_eq!(b.ty, vk::DescriptorType::STORAGE_BUFFER);
            }
            let ranges: Vec<_> = p
                .layout
                .push_constants
                .iter()
                .filter(|r| r.stages == control)
                .map(|r| (r.offset, r.size))
                .collect();
            assert_eq!(ranges, [(ts.push_offset, 24)], "{stem} {entry}");
            for r in &p.layout.push_constants {
                assert!(
                    r.stages == control
                        || r.offset + r.size <= ts.push_offset
                        || r.offset >= ts.push_offset + 24,
                    "{stem} {entry}: {r:?} overlaps the control range"
                );
            }
            let control_modes = modes(&p.stages[1].spirv);
            for mode in expected {
                assert!(
                    control_modes.contains(&vec![*mode as u32]),
                    "{stem} {entry}: {mode:?} in {control_modes:?}"
                );
            }
            assert!(
                control_modes.contains(&vec![crate::spv::ExecutionMode::OutputVertices as u32, 1])
            );
            // The readings no Metal observation settles are noted on the evaluation stage alone:
            // the winding and the factors' edge cases always, the scale and the per-instance step
            // where the descriptor asks for them, and the device's triangle order where its
            // tessellator is not Apple's.
            let tessellation_notes = |m: &StageModule| -> Vec<PrecisionNote> {
                let ours = [
                    PrecisionNote::TessellationWinding,
                    PrecisionNote::TessellationFactorEdgeCases,
                    PrecisionNote::TessellationFactorScale,
                    PrecisionNote::TessellationFactorInstance,
                    PrecisionNote::TessellationTriangleOrder,
                ];
                m.precision_notes
                    .iter()
                    .filter(|n| ours.contains(n))
                    .copied()
                    .collect()
            };
            let mut notes = vec![
                PrecisionNote::TessellationWinding,
                PrecisionNote::TessellationFactorEdgeCases,
            ];
            if t.factor_scale_enabled {
                notes.push(PrecisionNote::TessellationFactorScale);
            }
            if matches!(
                t.factor_step_function,
                TessellationFactorStepFunction::PerInstance
                    | TessellationFactorStepFunction::PerPatchAndPerInstance
            ) {
                notes.push(PrecisionNote::TessellationFactorInstance);
            }
            if own_order {
                notes.push(PrecisionNote::TessellationTriangleOrder);
            }
            notes.sort_unstable();
            assert_eq!(tessellation_notes(&p.stages[2]), notes, "{stem} {entry}");
            for i in [0, 1, 3] {
                assert_eq!(tessellation_notes(&p.stages[i]), [], "{stem} {entry} {i}");
            }
            for m in &p.stages {
                check_words_valid(&m.spirv, &p.requirements, dev.profile())
                    .unwrap_or_else(|e| panic!("{stem} {entry} {:?}: L3: {e}", m.stage));
            }
        }
    }
    assert!(
        ordered.iter().all(|&n| n > 0),
        "profiles with Metal's tessellator and with their own: {ordered:?}"
    );
    // Without it, two stages: the vertex function draws the tessellator's triangles as a
    // triangle list from the draw's work buffer and records, and owns the index buffers.
    let mut compiled = vec![0; cases.len()];
    for (stem, _) in &without {
        let dev = device(stem);
        for (i, (entry, t, domain, points, _)) in cases.iter().enumerate() {
            let d = tessellation_pipeline(&bc, &fragments, entry, Some("tess_fragment"), Some(*t));
            let Some(p) = emulated(&dev, &d, stem) else {
                continue;
            };
            compiled[i] += 1;
            let stages: Vec<vk::ShaderStage> = p.stages.iter().map(|m| m.stage).collect();
            assert_eq!(
                stages,
                [vk::ShaderStage::VERTEX, vk::ShaderStage::FRAGMENT],
                "{stem} {entry}"
            );
            assert_eq!(p.graphics.topology, vk::PrimitiveTopology::TRIANGLE_LIST);
            assert!(!p
                .graphics
                .dynamic_states
                .contains(&vk::DynamicState::PRIMITIVE_TOPOLOGY));
            assert_eq!(p.graphics.tessellation, None);
            let e = p.graphics.emulated_tessellation.expect("emulated state");
            assert_eq!(
                (e.domain, e.control_points, e.descriptor),
                (*domain, *points, *t)
            );
            assert!(!p
                .requirements
                .features
                .contains(&VkFeature::TessellationShader));
            let vertex = vk::ShaderStage::VERTEX.flags();
            let synthetic = |binding: SyntheticBinding| -> Vec<_> {
                p.layout
                    .sets
                    .iter()
                    .flat_map(|s| &s.bindings)
                    .filter(|b| {
                        b.source
                            == BindingSource::Synthetic {
                                stage: Stage::Vertex,
                                binding,
                            }
                    })
                    .collect()
            };
            for binding in [
                SyntheticBinding::TessellationWork,
                SyntheticBinding::TessellationRecords,
            ] {
                let found = synthetic(binding);
                assert_eq!(found.len(), 1, "{stem} {entry} {binding:?}");
                assert_eq!(found[0].stages, vertex);
                assert_eq!(found[0].ty, vk::DescriptorType::STORAGE_BUFFER);
            }
            // The scan reads the factors; the pipeline does not bind them.
            assert!(synthetic(SyntheticBinding::TessellationFactors).is_empty());
            for binding in [
                SyntheticBinding::PatchIndices,
                SyntheticBinding::ControlPointIndices,
            ] {
                for b in synthetic(binding) {
                    assert_eq!(b.stages, vertex, "{stem} {entry} {binding:?}");
                }
            }
            // The emulation draws Metal's triangles in Metal's order: no tessellation note.
            for m in &p.stages {
                assert_eq!(
                    m.precision_notes
                        .iter()
                        .filter(|n| matches!(
                            n,
                            PrecisionNote::TessellationWinding
                                | PrecisionNote::TessellationFactorEdgeCases
                                | PrecisionNote::TessellationTriangleOrder
                        ))
                        .count(),
                    0,
                    "{stem} {entry}"
                );
                check_words_valid(&m.spirv, &p.requirements, dev.profile())
                    .unwrap_or_else(|e| panic!("{stem} {entry} {:?}: L3: {e}", m.stage));
            }
        }
    }
    // Each function compiles on some profile; a profile of too few storage buffers refuses it.
    assert!(compiled.iter().all(|&n| n > 0), "{compiled:?}");
}

/// What the control stage does not implement or the device lacks is a typed refusal: a factor
/// maximum Metal rejects, a fragment function reading `[[primitive_id]]`, and multiview
/// tessellation without `multiviewTessellationShader` (MoltenVK's profile; RADV's has it and the
/// pipeline requires it).
#[test]
fn what_the_control_stage_cannot_do_is_refused_typed() {
    let Some(bc) = fixture() else { return };
    let Some(fragments) = assemble_text("fragments", FRAGMENTS) else {
        return;
    };
    let dev = device("moltenvk-m3");
    let budget = Budget::standard();
    let compile = |d: &RenderPipelineDescriptor| dev.compile_render_pipeline(d, &budget);
    let not_implemented = |r: Result<_, Error>, what: &str| match r {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::Tessellation, "{what}");
            assert_eq!(u.needs, Needs::NotImplemented, "{what}");
        }
        other => panic!("{what}: {:?}", other.map(|_| ())),
    };
    not_implemented(
        compile(&tessellation_pipeline(
            &bc,
            &fragments,
            "tess_quad_buffer",
            Some("tess_fragment_primitive"),
            None,
        )),
        "primitive id",
    );
    let t = TessellationDescriptor {
        max_tessellation_factor: 24,
        ..Default::default()
    };
    match compile(&tessellation_pipeline(
        &bc,
        &fragments,
        "tess_quad_buffer",
        None,
        Some(t),
    )) {
        Err(Error::InvalidDescriptor(DescriptorError::MaxTessellationFactor {
            factor: 24,
            partition_mode: TessellationPartitionMode::Pow2,
        })) => {}
        other => panic!("{:?}", other.map(|_| ())),
    }
    let mut d = tessellation_pipeline(&bc, &fragments, "tess_quad_buffer", None, None);
    d.max_vertex_amplification_count = 2;
    match compile(&d) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::VertexAmplification);
            assert_eq!(
                u.needs,
                Needs::AnyOf(vec![vec![Capability::Feature(
                    VkFeature::MultiviewTessellationShader
                )]])
            );
        }
        other => panic!("{:?}", other.map(|_| ())),
    }
    let radv = device("amd-rdna3-radv");
    let p = radv
        .compile_render_pipeline(&d, &budget)
        .expect("radv has multiview tessellation");
    assert!(p
        .requirements
        .features
        .contains(&VkFeature::MultiviewTessellationShader));
    for m in &p.stages {
        check_words_valid(&m.spirv, &p.requirements, radv.profile())
            .unwrap_or_else(|e| panic!("{:?}: L3: {e}", m.stage));
    }
}

/// A tessellation pipeline's plan writes the encoder's factor buffer at the control stage's binding
/// (a zero buffer without one), pushes each stage's range at its Vulkan stage, and draws patches:
/// one vertex per patch from the patch start, the control stage's push range holding the factor
/// buffer's residual bind offset, instance stride, and `half` scale, and the factor buffer bound
/// over the factors the draw reads. A draw of primitives is refused.
#[test]
fn a_tessellation_plan_draws_patches() {
    let Some(bc) = fixture() else { return };
    let Some(fragments) = assemble_text("fragments", FRAGMENTS) else {
        return;
    };
    let t = TessellationDescriptor {
        factor_step_function: TessellationFactorStepFunction::PerPatchAndPerInstance,
        factor_scale_enabled: true,
        ..Default::default()
    };
    for stem in ["moltenvk-m3", "lavapipe", "lunarg-desktop-baseline-2023"] {
        let dev = device(stem);
        let d = tessellation_pipeline(
            &bc,
            &fragments,
            "tess_triangle_points",
            Some("tess_fragment"),
            Some(t),
        );
        let p = dev
            .compile_render_pipeline(&d, &Budget::standard())
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let ts = p.graphics.tessellation.expect("tessellation state");
        let factors = p
            .layout
            .sets
            .iter()
            .flat_map(|s| s.bindings.iter().map(move |b| (s.set, b)))
            .find(|(_, b)| {
                b.source
                    == BindingSource::Synthetic {
                        stage: Stage::Vertex,
                        binding: SyntheticBinding::TessellationFactors,
                    }
            })
            .map(|(set, b)| (set, b.binding))
            .expect("factor binding");

        let mut state = BindState::default();
        let mut table = StageTable::default();
        for index in [0, 1] {
            table.buffers.insert(
                index,
                BoundBuffer::Buffer {
                    length: 4096,
                    offset: 0,
                    gpu_address: 0x1_0000,
                    device_address: None,
                    attribute_stride: None,
                },
            );
        }
        state.stages.insert(Stage::Vertex, table);
        // Without a factor buffer the zero buffer stands in and every patch is culled.
        let plan = p.plan(&dev, &state.shape()).expect("plan");
        let write = |plan: &crate::bind::BindingPlan| {
            plan.descriptors
                .iter()
                .find(|w| (w.set, w.binding) == factors)
                .map(|w| w.source.clone())
                .expect("a factor write")
        };
        assert!(matches!(write(&plan), WriteSource::Neutral(_)), "{stem}");

        let (offset, stride) = (64, 256);
        state.tessellation_factors = TessellationFactors::new(
            Some(TessellationFactorBuffer::new(4096, offset, stride)),
            1.3,
        );
        let plan = p.plan(&dev, &state.shape()).expect("plan");
        assert_eq!(write(&plan), WriteSource::TessellationFactors, "{stem}");
        assert!(plan
            .uses
            .iter()
            .any(|u| u.resource == UseTarget::TessellationFactors
                && u.stages == vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER));
        let draw = DrawCall::Patches {
            control_points: 3,
            patch_start: 2,
            patch_count: 5,
            patch_index: None,
            instance_count: 2,
            base_instance: 1,
            control_point_index: None,
        };
        let cmds = plan.instantiate_draw(&state, &draw).expect("draw");
        assert_eq!(
            cmds.draw,
            vk::DrawCommand::Draw {
                vertex_count: 5,
                instance_count: 2,
                first_vertex: 2,
                first_instance: 1
            }
        );
        // Every push write is one of the layout's ranges, at its stage.
        for w in &cmds.push_constants {
            assert!(
                p.layout.push_constants.iter().any(|r| r.stages == w.stages
                    && r.offset == w.offset
                    && r.size as usize == w.bytes.len()),
                "{stem}: {w:?} is not a layout range"
            );
            assert!(
                !w.stages.contains(vk::ShaderStage::VERTEX.flags()),
                "{stem}"
            );
        }
        let control = cmds
            .push_constants
            .iter()
            .find(|w| w.stages == vk::ShaderStage::TESSELLATION_CONTROL.flags())
            .expect("control push");
        assert_eq!(control.offset, ts.push_offset);
        // Patches 2..7 of instances 1..3, 8 bytes each, 256 bytes an instance.
        let start = offset + 2 * 8 + 256;
        let end = offset + 6 * 8 + 2 * 256 + 8;
        let a = dev
            .profile()
            .fields()
            .limits
            .min_storage_buffer_offset_alignment
            .max(4);
        let lo = start - start % a;
        let range = cmds
            .descriptor_ranges
            .iter()
            .find(|r| (r.set, r.binding) == factors)
            .expect("factor range");
        assert_eq!(range.range.offset, lo, "{stem}");
        assert_eq!(range.range.range, end.next_multiple_of(16) - lo, "{stem}");
        let word =
            |k: usize| u32::from_le_bytes(control.bytes[4 * k..4 * k + 4].try_into().unwrap());
        assert_eq!(word(0), (offset as u32).wrapping_sub(lo as u32), "{stem}");
        assert_eq!(word(1), stride as u32);
        // 1.3 as a `half`: 1 + 307/1024.
        assert_eq!(f32::from_bits(word(2)), 1.0 + 307.0 / 1024.0);
        // No patch index buffer: none bound, and the control stage told so.
        assert_eq!((word(3), word(4)), (0, 0), "{stem}");
        let patch_indices = p
            .layout
            .sets
            .iter()
            .flat_map(|s| s.bindings.iter().map(move |b| (s.set, b)))
            .find(|(_, b)| {
                b.source
                    == BindingSource::Synthetic {
                        stage: Stage::Vertex,
                        binding: SyntheticBinding::PatchIndices,
                    }
            })
            .map(|(set, b)| (set, b.binding))
            .expect("patch index binding");
        assert!(matches!(
            plan.descriptors
                .iter()
                .find(|w| (w.set, w.binding) == patch_indices)
                .map(|w| &w.source),
            Some(WriteSource::PatchIndices { .. })
        ));
        assert!(!cmds
            .descriptor_ranges
            .iter()
            .any(|r| (r.set, r.binding) == patch_indices));
        // A patch index buffer: bound over the indices of patches 2..7 from offset 64, its
        // residual and 1 pushed, read in the control stage.
        let indexed = DrawCall::Patches {
            control_points: 3,
            patch_start: 2,
            patch_count: 5,
            patch_index: Some((IndexBufferRef::new(7, 4096), 64)),
            instance_count: 2,
            base_instance: 1,
            control_point_index: None,
        };
        let cmds = plan
            .instantiate_draw(&state, &indexed)
            .expect("indexed draw");
        let range = cmds
            .descriptor_ranges
            .iter()
            .find(|r| (r.set, r.binding) == patch_indices)
            .expect("patch index range");
        let start = 64 + 2 * 4;
        assert_eq!(range.range.offset, start - start % a, "{stem}");
        let control = cmds
            .push_constants
            .iter()
            .find(|w| w.stages == vk::ShaderStage::TESSELLATION_CONTROL.flags())
            .expect("control push");
        let word =
            |k: usize| u32::from_le_bytes(control.bytes[4 * k..4 * k + 4].try_into().unwrap());
        assert_eq!(
            (word(3), word(4)),
            (64u32.wrapping_sub(range.range.offset as u32), 1),
            "{stem}"
        );
        assert!(cmds
            .uses
            .iter()
            .any(|u| u.resource == UseTarget::PatchIndices
                && u.stages == vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER));

        assert_eq!(
            plan.instantiate_draw(
                &state,
                &DrawCall::Primitives {
                    vertex_start: 0,
                    vertex_count: 3,
                    instance_count: 1,
                    base_instance: 0
                }
            ),
            Err(BindError::InvalidDrawKind { patches: false })
        );
        let indirect = DrawCall::PatchesIndirect {
            control_points: 3,
            patch_index: None,
            indirect: IndirectRef::new(9, 32),
            control_point_index: None,
        };
        let cmds = plan.instantiate_draw(&state, &indirect).expect("indirect");
        assert_eq!(
            cmds.draw,
            vk::DrawCommand::DrawIndirect {
                offset: 32,
                draw_count: 1,
                stride: 16
            }
        );
        assert!(!cmds.indirect_needs_arguments, "{stem}");
    }
}

/// An emulated tessellation pipeline's plan binds the draw's work buffer and records and no factor
/// buffer, and a patch draw scans: the work buffer's header holds a direct draw's arguments and the
/// factor buffer's residual bind offset, instance stride and `half` scale, its records are its own,
/// the scan binds the factors the draw reads (a `COMPUTE_SHADER` use), and the draw is chunk 0's
/// `vkCmdDrawIndirect` from the work buffer, with the chunk pushed to the vertex stage. An indirect
/// draw's records are in the pool and the consumer copies Metal's arguments into the header for
/// the scan, so the CPU does not read them.
#[test]
fn an_emulated_tessellation_plan_scans_and_draws_chunks() {
    use crate::bind::{BufferRange, TessellationRecords};
    use crate::lower::tessellation::work;
    let Some(bc) = fixture() else { return };
    let Some(fragments) = assemble_text("fragments", FRAGMENTS) else {
        return;
    };
    let t = TessellationDescriptor {
        factor_step_function: TessellationFactorStepFunction::PerPatchAndPerInstance,
        factor_scale_enabled: true,
        ..Default::default()
    };
    let (_, without) = profiles();
    let mut planned = 0;
    for (stem, _) in &without {
        let dev = device(stem);
        let d = tessellation_pipeline(
            &bc,
            &fragments,
            "tess_triangle_points",
            Some("tess_fragment"),
            Some(t),
        );
        let Some(p) = emulated(&dev, &d, stem) else {
            continue;
        };
        planned += 1;
        let e = p.graphics.emulated_tessellation.expect("emulated");
        let mut state = BindState::default();
        let mut table = StageTable::default();
        for index in [0, 1] {
            table.buffers.insert(
                index,
                BoundBuffer::Buffer {
                    length: 4096,
                    offset: 0,
                    gpu_address: 0x1_0000,
                    device_address: None,
                    attribute_stride: None,
                },
            );
        }
        state.stages.insert(Stage::Vertex, table);
        let (offset, stride) = (64, 256);
        state.tessellation_factors = TessellationFactors::new(
            Some(TessellationFactorBuffer::new(4096, offset, stride)),
            1.3,
        );
        let plan = p.plan(&dev, &state.shape()).expect("plan");
        for source in [
            WriteSource::TessellationWork,
            WriteSource::TessellationRecords,
        ] {
            assert_eq!(
                plan.descriptors
                    .iter()
                    .filter(|w| w.source == source)
                    .count(),
                1,
                "{stem} {source:?}"
            );
        }
        assert!(!plan
            .descriptors
            .iter()
            .any(|w| w.source == WriteSource::TessellationFactors));
        assert!(plan
            .uses
            .iter()
            .any(|u| u.resource == UseTarget::TessellationFactors
                && u.stages == vk::PipelineStageFlags2::COMPUTE_SHADER));
        let draw = DrawCall::Patches {
            control_points: 3,
            patch_start: 2,
            patch_count: 5,
            patch_index: None,
            instance_count: 2,
            base_instance: 1,
            control_point_index: None,
        };
        let cmds = plan.instantiate_draw(&state, &draw).expect("draw");
        let scan = cmds.tessellation_dispatch.clone().expect("a scan");
        assert_eq!(scan.kernel, e.scan_kernel());
        assert_eq!(scan.records, TessellationRecords::Exact { records: 10 });
        assert_eq!(scan.records.bytes(), 10 * 32);
        assert_eq!(scan.arguments, None);
        // Patches 2..7 of instances 1..3, 8 bytes each, 256 bytes an instance.
        let start = offset + 2 * 8 + 256;
        let end = offset + 6 * 8 + 2 * 256 + 8;
        let a = dev
            .profile()
            .fields()
            .limits
            .min_storage_buffer_offset_alignment
            .max(4);
        let lo = start - start % a;
        assert_eq!(
            scan.factors,
            Some(BufferRange {
                offset: lo,
                range: end.next_multiple_of(16) - lo
            }),
            "{stem}"
        );
        let h = |w: u32| scan.header[w as usize];
        assert_eq!(
            [
                h(work::PATCH_COUNT),
                h(work::INSTANCE_COUNT),
                h(work::PATCH_START),
                h(work::BASE_INSTANCE),
            ],
            [5, 2, 2, 1]
        );
        // The residual from the bound range's start, modulo `2^32` (the range starts at the first
        // factor read, past the bind offset).
        assert_eq!(
            h(work::FACTOR_OFFSET),
            (offset as u32).wrapping_sub(lo as u32),
            "{stem}"
        );
        assert_eq!(h(work::FACTOR_STRIDE), stride as u32);
        // 1.3 as a `half`: 1 + 307/1024.
        assert_eq!(f32::from_bits(h(work::SCALE)), 1.0 + 307.0 / 1024.0);
        // Ten records, far below a chunk of triangles at the most per patch: one chunk.
        assert_eq!(scan.chunks(0), 1);
        let words = scan.work(1);
        assert_eq!(words.len() as u64, work::words(1));
        assert_eq!(words[work::CHUNKS as usize], 1);
        let (writes, command) = scan.draw(0);
        assert_eq!(cmds.draw, command);
        assert_eq!(
            command,
            vk::DrawCommand::DrawIndirect {
                offset: work::draw_offset(0),
                draw_count: 1,
                stride: 16
            }
        );
        assert!(!writes.is_empty(), "{stem}");
        for w in &writes {
            assert_eq!(w.stages, vk::ShaderStage::VERTEX.flags(), "{stem}");
            assert_eq!(w.bytes, 0u32.to_le_bytes());
        }
        let (writes, _) = scan.draw(3);
        assert!(writes.iter().all(|w| w.bytes == 3u32.to_le_bytes()));
        assert!(!cmds.indirect_needs_arguments, "{stem}");

        let indirect = DrawCall::PatchesIndirect {
            control_points: 3,
            patch_index: None,
            indirect: IndirectRef::new(9, 36),
            control_point_index: None,
        };
        let cmds = plan.instantiate_draw(&state, &indirect).expect("indirect");
        let scan = cmds.tessellation_dispatch.clone().expect("a scan");
        assert_eq!(scan.records, TessellationRecords::Pool);
        // Copied from their offset into the header, which the binder leaves zero.
        assert_eq!(scan.arguments, Some(36));
        assert_eq!(&scan.header[..4], &[0; 4]);
        assert!(!cmds.indirect_needs_arguments, "{stem}");
        assert!(cmds.uses.iter().any(|u| u.resource == UseTarget::Indirect
            && u.stages == vk::PipelineStageFlags2::COMPUTE_SHADER));
        // A pool of 1 MiB: every record it holds at the most triangles each.
        let records = (1u64 << 20) / 32;
        let most = u64::from(e.scan_kernel().max_patch_triangles());
        assert_eq!(
            u64::from(scan.chunks(1 << 20)),
            (records * most).div_ceil(work::CHUNK_TRIANGLES).max(1)
        );
    }
    assert!(planned > 0);
}

/// A post-tessellation vertex function bakes the descriptor's tessellation state into its
/// lowering inputs, and so into its compile key, only on a device that emulates it: a native
/// device's vertex key and inputs are the same for every tessellation state, and an emulating
/// device's differ by state, Metal's defaults keyed as no state.
#[test]
fn only_an_emulated_function_keys_the_tessellation_state() {
    let Some(bc) = fixture() else { return };
    let Some(fragments) = assemble_text("fragments", FRAGMENTS) else {
        return;
    };
    let other = TessellationDescriptor {
        partition_mode: TessellationPartitionMode::FractionalOdd,
        max_tessellation_factor: 32,
        output_winding_order: Winding::CounterClockwise,
        ..Default::default()
    };
    let budget = Budget::standard();
    let (with, without) = profiles();
    let keys = |dev: &Device, t: Option<TessellationDescriptor>| {
        let d = tessellation_pipeline(&bc, &fragments, "tess_quad_buffer", None, t);
        let k = dev.render_pipeline_key(&d, &budget).expect("keys");
        (k.vertex, k.vertex_inputs.expect("inputs"))
    };
    for (stem, _) in &with {
        let dev = device(stem);
        let (none, inputs) = keys(&dev, None);
        assert_eq!(inputs.tessellation, None, "{stem}");
        for t in [TessellationDescriptor::default(), other] {
            assert_eq!(keys(&dev, Some(t)), (none, inputs.clone()), "{stem}");
        }
    }
    for (stem, _) in &without {
        let dev = device(stem);
        let (none, inputs) = keys(&dev, None);
        assert_eq!(inputs.tessellation, None, "{stem}");
        assert_eq!(
            keys(&dev, Some(TessellationDescriptor::default())),
            (none, inputs),
            "{stem}"
        );
        let (key, inputs) = keys(&dev, Some(other));
        assert_eq!(inputs.tessellation, Some(other), "{stem}");
        assert_ne!(key, none, "{stem}");
    }
}

/// The storage buffers an emulated post-tessellation vertex function binds beside its own (the
/// draw's work buffer and records, and the patch index buffer it reads) count toward
/// `maxPerStageDescriptorStorageBuffers`: on a device of one fewer than the vertex stage's storage
/// buffers the pipeline is refused, typed, naming that limit, and on one of exactly as many it
/// compiles.
#[test]
fn an_emulated_function_counts_its_draw_s_buffers_against_the_stage_s_limit() {
    let Some(bc) = fixture() else { return };
    let Some(fragments) = assemble_text("fragments", FRAGMENTS) else {
        return;
    };
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/vk12-floor.json");
    let text = std::fs::read_to_string(path).expect("profile");
    let with_storage = |buffers: u32| {
        let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        desc.limits.max_per_stage_descriptor_storage_buffers = buffers;
        Device::new(desc, DevicePolicy::default()).expect("device")
    };
    let budget = Budget::standard();
    // Its control points and patch inputs are pulled from two buffers, which no device address
    // stands in for.
    let d = tessellation_pipeline(&bc, &fragments, "tess_triangle_points", None, None);
    let roomy = with_storage(64)
        .compile_render_pipeline(&d, &budget)
        .expect("the pipeline");
    assert!(roomy.graphics.emulated_tessellation.is_some());
    let bindings: Vec<_> = roomy
        .layout
        .sets
        .iter()
        .flat_map(|s| &s.bindings)
        .filter(|b| {
            b.ty == vk::DescriptorType::STORAGE_BUFFER
                && b.stages.contains(vk::ShaderStageFlags::VERTEX)
        })
        .collect();
    for binding in [
        SyntheticBinding::TessellationWork,
        SyntheticBinding::TessellationRecords,
        SyntheticBinding::PatchIndices,
    ] {
        assert!(
            bindings.iter().any(|b| b.source
                == BindingSource::Synthetic {
                    stage: Stage::Vertex,
                    binding,
                }),
            "{binding:?}"
        );
    }
    let n = bindings.len() as u32;
    with_storage(n)
        .compile_render_pipeline(&d, &budget)
        .expect("the pipeline at the stage's buffers");
    match with_storage(n - 1).compile_render_pipeline(&d, &budget) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::DescriptorCount);
            assert_eq!(
                u.needs,
                Needs::AnyOf(vec![vec![Capability::Limit {
                    limit: VkLimit::MaxPerStageDescriptorStorageBuffers,
                    required: u64::from(n),
                    available: u64::from(n - 1),
                }]])
            );
        }
        other => panic!("expected a descriptor-count refusal, got {other:?}"),
    }
}

/// Behind an emulated post-tessellation vertex function a fragment function reads its
/// `[[primitive_id]]` from the primitive ID's flat varying (`IoRole::PrimitiveId`), which the
/// vertex function writes as its patch's position, on every profile without `tessellationShader`:
/// with `geometryShader`, whose fragment stage declares `PrimitiveId`, under the fragment's
/// `LoweringInputs::primitive_id_varying`; without it, as every fragment function there reads it.
/// The vertex function never takes the plain vertex function's form (`(VertexIndex - first) / n`):
/// its pipeline reads no `PushField::VerticesPerPrimitive`. A pipeline whose fragment function
/// reads no `[[primitive_id]]` carries no varying. Behind an evaluation stage it stays refused.
#[test]
fn an_emulated_pipeline_reads_the_primitive_id_from_the_patch_position() {
    let Some(bc) = fixture() else { return };
    let Some(fragments) = assemble_text("fragments", FRAGMENTS) else {
        return;
    };
    let budget = Budget::standard();
    let (with, without) = profiles();
    let mut declaring = 0;
    for (stem, profile) in &without {
        let dev = device(stem);
        let declares = !crate::pipeline::compile::primitive_id_varying_on(profile);
        declaring += usize::from(declares);
        for fragment in ["tess_fragment_primitive", "tess_fragment"] {
            let d =
                tessellation_pipeline(&bc, &fragments, "tess_quad_buffer", Some(fragment), None);
            let k = dev.render_pipeline_key(&d, &budget).expect("keys");
            let reads = fragment == "tess_fragment_primitive";
            let what = format!("{stem} {fragment}");
            let vertex_inputs = k.vertex_inputs.expect("vertex inputs");
            let fragment_inputs = k.fragment_inputs.expect("fragment inputs");
            assert_eq!(vertex_inputs.primitive_id_varying, reads, "{what}");
            assert_eq!(
                fragment_inputs.primitive_id_varying,
                reads && declares,
                "{what}"
            );
            // The stages as the pipeline compiles them: the vertex function writes the varying
            // exactly where the fragment function reads it.
            let role = |list: &[crate::pipeline::stage_io::IoVariable]| {
                list.iter()
                    .any(|v| v.role == crate::pipeline::stage_io::IoRole::PrimitiveId)
            };
            let stage = |source: Option<&StageSource>, inputs| match source {
                Some(StageSource::Function(f)) => dev
                    .compile_function(f, inputs, &budget)
                    .unwrap_or_else(|e| panic!("{what}: {e}"))
                    .artifact()
                    .interface
                    .io
                    .clone(),
                _ => panic!("{what}: a function stage"),
            };
            let vertex = stage(Some(&d.vertex), &vertex_inputs);
            let fragment_io = stage(d.fragment.as_ref(), &fragment_inputs);
            assert_eq!(role(&vertex.outputs), reads, "{what}: the vertex writes");
            assert_eq!(
                role(&fragment_io.inputs),
                reads,
                "{what}: the fragment reads"
            );
            let p = dev
                .compile_render_pipeline(&d, &budget)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            assert!(
                !p.interface
                    .push
                    .iter()
                    .flat_map(|pl| &pl.fields)
                    .any(|f| f.field == PushField::VerticesPerPrimitive),
                "{what}: the plain vertex function's divisor"
            );
            assert!(p.graphics.flat_fragment_inputs || !reads, "{what}");
            for m in &p.stages {
                check_words_valid(&m.spirv, &p.requirements, dev.profile())
                    .unwrap_or_else(|e| panic!("{what} {:?}: L3: {e}", m.stage));
            }
        }
    }
    assert!(
        declaring < without.len(),
        "a committed profile without geometryShader or tessellationShader"
    );
    for (stem, _) in &with {
        let d = tessellation_pipeline(
            &bc,
            &fragments,
            "tess_quad_buffer",
            Some("tess_fragment_primitive"),
            None,
        );
        match device(stem).compile_render_pipeline(&d, &budget) {
            Err(Error::Unsupported(u)) => {
                assert_eq!(u.feature, Feature::Tessellation, "{stem}");
                assert_eq!(u.needs, Needs::NotImplemented, "{stem}");
            }
            other => panic!("{stem}: {:?}", other.map(|_| ())),
        }
    }
}
