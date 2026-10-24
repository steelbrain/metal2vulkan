//! Varyings a vertex function writes the same at every vertex of a primitive, read flat by the
//! fragment function of its pipeline, over the inline module [`MODULE`]: a vertex function
//! `flt_vertex` whose outputs are
//!
//! - `tint`, a `constant` buffer's element at `[[instance_id]]`: uniform;
//! - `uv`, a `device` buffer's element at `[[vertex_id]]`: per vertex;
//! - `k`, arithmetic on `[[instance_id]]`: uniform;
//! - `h`, a `half2` from a `constant` buffer at `[[instance_id]]`: uniform;
//! - `id`, `[[instance_id]]` itself: uniform, an integer the fragment reads flat anyway;
//! - `d`, a `device` buffer's element at `[[instance_id]]`: not proven (device memory may be
//!   written while the draw runs);
//! - `v`, `[[vertex_id]]` converted: per vertex;
//! - `a`, attribute 0: uniform exactly where the vertex descriptor steps it per instance or at a
//!   constant step (fixed-function input or fetched in the shader);
//!
//! and a fragment function `flt_fragment` that reads each with a stated interpolation (`tint` and
//! `uv` center perspective, `k` center without perspective, `h` centroid, `id` flat, `d` and `v`
//! center, `a` sample).
//!
//! The module is our own: `xcrun metal -S -emit-llvm` of the MSL in [`MODULE`]'s comment,
//! written with opaque pointers and without its alias metadata. `llvm-as` assembles it; a
//! missing tool skips visibly. No GPU is used.

use std::collections::BTreeSet;
use std::path::Path;

use crate::budget::Budget;
use crate::cache::VariantKey;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{DescriptorError, Error, RuntimeInput};
use crate::library::Library;
use crate::metal::{PixelFormat, VertexFormat, VertexStepFunction};
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
    VertexAttributeDescriptor, VertexBufferLayoutDescriptor, VertexDescriptor,
};
use crate::pipeline::lower_inputs::{LoweringInputs, VertexFetch};
use crate::pipeline::stage_io::{Interpolation, IoRole};
use crate::pipeline::sym::VaryingKey;
use crate::pipeline::testing::AssembleError;
use crate::pipeline::{RenderPipeline, StageEntries};
use crate::spirv::testing::check_words_valid;

/// The module, from
///
/// ```metal
/// struct VIn { float4 c [[attribute(0)]]; };
/// struct VOut {
///     float4 pos [[position]]; float4 tint [[user(tint)]]; float2 uv [[user(uv)]];
///     float k [[user(k)]]; half2 h [[user(h)]]; uint id [[user(id)]]; float d [[user(d)]];
///     float v [[user(v)]]; float4 a [[user(a)]];
/// };
/// vertex VOut flt_vertex(VIn in [[stage_in]], uint vid [[vertex_id]], uint iid [[instance_id]],
///                        const device float4 *pos [[buffer(0)]],
///                        constant float4 *tints [[buffer(1)]],
///                        const device float *dev [[buffer(2)]],
///                        constant half2 *halves [[buffer(3)]]) {
///     VOut o; o.pos = pos[vid]; o.tint = tints[iid]; o.uv = pos[vid].xy;
///     o.k = float(iid) * 0.5f + 3.0f; o.h = halves[iid]; o.id = iid; o.d = dev[iid];
///     o.v = float(vid); o.a = in.c; return o;
/// }
/// struct FIn {
///     float4 tint [[user(tint), center_perspective]]; float2 uv [[user(uv), center_perspective]];
///     float k [[user(k), center_no_perspective]]; half2 h [[user(h), centroid_perspective]];
///     uint id [[user(id), flat]]; float d [[user(d), center_perspective]];
///     float v [[user(v), center_perspective]]; float4 a [[user(a), sample_perspective]];
/// };
/// fragment float4 flt_fragment(FIn in [[stage_in]]) {
///     return float4(in.tint.x + in.uv.x, in.k + float(in.h.x), float(in.id) + in.d + in.v,
///                   in.a.w);
/// }
/// ```
const MODULE: &str = r#"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define <{ <4 x float>, <4 x float>, <2 x float>, float, <2 x half>, i32, float, float, <4 x float> }> @flt_vertex(<4 x float> %0, i32 noundef %1, i32 noundef %2, ptr addrspace(1) noundef readonly "air-buffer-no-alias" %3, ptr addrspace(2) noundef readonly "air-buffer-no-alias" %4, ptr addrspace(1) noundef readonly "air-buffer-no-alias" %5, ptr addrspace(2) noundef readonly "air-buffer-no-alias" %6) local_unnamed_addr #0 {
  %8 = zext i32 %1 to i64
  %9 = getelementptr inbounds <4 x float>, ptr addrspace(1) %3, i64 %8
  %10 = load <4 x float>, ptr addrspace(1) %9, align 16
  %11 = zext i32 %2 to i64
  %12 = getelementptr inbounds <4 x float>, ptr addrspace(2) %4, i64 %11
  %13 = load <4 x float>, ptr addrspace(2) %12, align 16
  %14 = shufflevector <4 x float> %10, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %15 = tail call fast float @air.convert.f.f32.u.i32(i32 %2) #2
  %16 = fmul fast float %15, 5.000000e-01
  %17 = fadd fast float %16, 3.000000e+00
  %18 = getelementptr inbounds <2 x half>, ptr addrspace(2) %6, i64 %11
  %19 = load <2 x half>, ptr addrspace(2) %18, align 4
  %20 = getelementptr inbounds float, ptr addrspace(1) %5, i64 %11
  %21 = load float, ptr addrspace(1) %20, align 4
  %22 = tail call fast float @air.convert.f.f32.u.i32(i32 %1) #2
  %23 = insertvalue <{ <4 x float>, <4 x float>, <2 x float>, float, <2 x half>, i32, float, float, <4 x float> }> undef, <4 x float> %10, 0
  %24 = insertvalue <{ <4 x float>, <4 x float>, <2 x float>, float, <2 x half>, i32, float, float, <4 x float> }> %23, <4 x float> %13, 1
  %25 = insertvalue <{ <4 x float>, <4 x float>, <2 x float>, float, <2 x half>, i32, float, float, <4 x float> }> %24, <2 x float> %14, 2
  %26 = insertvalue <{ <4 x float>, <4 x float>, <2 x float>, float, <2 x half>, i32, float, float, <4 x float> }> %25, float %17, 3
  %27 = insertvalue <{ <4 x float>, <4 x float>, <2 x float>, float, <2 x half>, i32, float, float, <4 x float> }> %26, <2 x half> %19, 4
  %28 = insertvalue <{ <4 x float>, <4 x float>, <2 x float>, float, <2 x half>, i32, float, float, <4 x float> }> %27, i32 %2, 5
  %29 = insertvalue <{ <4 x float>, <4 x float>, <2 x float>, float, <2 x half>, i32, float, float, <4 x float> }> %28, float %21, 6
  %30 = insertvalue <{ <4 x float>, <4 x float>, <2 x float>, float, <2 x half>, i32, float, float, <4 x float> }> %29, float %22, 7
  %31 = insertvalue <{ <4 x float>, <4 x float>, <2 x float>, float, <2 x half>, i32, float, float, <4 x float> }> %30, <4 x float> %0, 8
  ret <{ <4 x float>, <4 x float>, <2 x float>, float, <2 x half>, i32, float, float, <4 x float> }> %31
}

declare float @air.convert.f.f32.u.i32(i32) local_unnamed_addr #1

define <4 x float> @flt_fragment(<4 x float> %0, <2 x float> %1, float %2, <2 x half> %3, i32 %4, float %5, float %6, <4 x float> %7) local_unnamed_addr #1 {
  %9 = extractelement <4 x float> %0, i64 0
  %10 = extractelement <2 x float> %1, i64 0
  %11 = fadd fast float %10, %9
  %12 = insertelement <4 x float> undef, float %11, i64 0
  %13 = extractelement <2 x half> %3, i64 0
  %14 = fpext half %13 to float
  %15 = fadd fast float %14, %2
  %16 = insertelement <4 x float> %12, float %15, i64 1
  %17 = tail call fast float @air.convert.f.f32.u.i32(i32 %4) #2
  %18 = fadd fast float %6, %5
  %19 = fadd fast float %18, %17
  %20 = insertelement <4 x float> %16, float %19, i64 2
  %21 = shufflevector <4 x float> %20, <4 x float> %7, <4 x i32> <i32 0, i32 1, i32 2, i32 7>
  ret <4 x float> %21
}

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(argmem: read) }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nounwind willreturn memory(none) }

!air.vertex = !{!9}
!air.fragment = !{!28}
!air.compile_options = !{!40, !41, !42}
!air.version = !{!44}
!air.language_version = !{!45}

!9 = !{ptr @flt_vertex, !10, !20}
!10 = !{!11, !12, !13, !14, !15, !16, !17, !18, !19}
!11 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!12 = !{!"air.vertex_output", !"user(tint)", !"air.arg_type_name", !"float4", !"air.arg_name", !"tint"}
!13 = !{!"air.vertex_output", !"user(uv)", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
!14 = !{!"air.vertex_output", !"user(k)", !"air.arg_type_name", !"float", !"air.arg_name", !"k"}
!15 = !{!"air.vertex_output", !"user(h)", !"air.arg_type_name", !"half2", !"air.arg_name", !"h"}
!16 = !{!"air.vertex_output", !"user(id)", !"air.arg_type_name", !"uint", !"air.arg_name", !"id"}
!17 = !{!"air.vertex_output", !"user(d)", !"air.arg_type_name", !"float", !"air.arg_name", !"d"}
!18 = !{!"air.vertex_output", !"user(v)", !"air.arg_type_name", !"float", !"air.arg_name", !"v"}
!19 = !{!"air.vertex_output", !"user(a)", !"air.arg_type_name", !"float4", !"air.arg_name", !"a"}
!20 = !{!21, !22, !23, !24, !25, !26, !27}
!21 = !{i32 0, !"air.vertex_input", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"float4", !"air.arg_name", !"c"}
!22 = !{i32 1, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vid"}
!23 = !{i32 2, !"air.instance_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"iid"}
!24 = !{i32 3, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!25 = !{i32 4, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"tints"}
!26 = !{i32 5, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"dev"}
!27 = !{i32 6, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"half2", !"air.arg_name", !"halves"}
!28 = !{ptr @flt_fragment, !29, !31}
!29 = !{!30}
!30 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!31 = !{!32, !33, !34, !35, !36, !37, !38, !39}
!32 = !{i32 0, !"air.fragment_input", !"user(tint)", !"air.center", !"air.perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"tint"}
!33 = !{i32 1, !"air.fragment_input", !"user(uv)", !"air.center", !"air.perspective", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
!34 = !{i32 2, !"air.fragment_input", !"user(k)", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float", !"air.arg_name", !"k"}
!35 = !{i32 3, !"air.fragment_input", !"user(h)", !"air.centroid", !"air.perspective", !"air.arg_type_name", !"half2", !"air.arg_name", !"h"}
!36 = !{i32 4, !"air.fragment_input", !"user(id)", !"air.flat", !"air.arg_type_name", !"uint", !"air.arg_name", !"id"}
!37 = !{i32 5, !"air.fragment_input", !"user(d)", !"air.center", !"air.perspective", !"air.arg_type_name", !"float", !"air.arg_name", !"d"}
!38 = !{i32 6, !"air.fragment_input", !"user(v)", !"air.center", !"air.perspective", !"air.arg_type_name", !"float", !"air.arg_name", !"v"}
!39 = !{i32 7, !"air.fragment_input", !"user(a)", !"air.sample", !"air.perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"a"}
!40 = !{!"air.compile.denorms_disable"}
!41 = !{!"air.compile.fast_math_enable"}
!42 = !{!"air.compile.framebuffer_fetch_enable"}
!44 = !{i32 2, i32 8, i32 0}
!45 = !{!"Metal", i32 3, i32 1, i32 0}
"#;

fn module() -> Option<Vec<u8>> {
    match metal2vulkan_devtools::llvm_assemble_text(MODULE) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the module: {why}"),
    }
}

fn device(stem: &str) -> Device {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let description = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::new(description, DevicePolicy::default()).expect("device")
}

fn function(bc: &[u8], name: &str) -> FunctionDescriptor {
    FunctionDescriptor::new(
        Library::from_air(bc)
            .expect("library")
            .function(name)
            .unwrap_or_else(|e| panic!("{name}: {e}")),
    )
}

/// The pipeline of `flt_vertex` and `flt_fragment`, attribute 0 a `float4` of buffer 7 stepped
/// by `step`, into an `RGBA32Float` attachment at `samples` samples.
fn pipeline(bc: &[u8], step: VertexStepFunction, samples: u32) -> RenderPipelineDescriptor {
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(function(bc, "flt_vertex")));
    d.fragment = Some(StageSource::Function(function(bc, "flt_fragment")));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Float));
    d.raster_sample_count = samples;
    let mut vertex = VertexDescriptor::new();
    let mut layout = VertexBufferLayoutDescriptor::new(Some(16));
    layout.step_function = step;
    layout.step_rate = u32::from(step != VertexStepFunction::Constant);
    vertex.layouts.insert(7, layout);
    vertex.attributes.insert(
        0,
        VertexAttributeDescriptor::new(VertexFormat::Float4, 0, 7),
    );
    d.vertex_descriptor = Some(vertex);
    d
}

fn keys(names: &[&str]) -> BTreeSet<VaryingKey> {
    names
        .iter()
        .map(|n| VaryingKey::user(format!("user({n})").as_bytes()))
        .collect()
}

/// Every stage of `p` passes L3 on `dev`.
fn valid(p: &RenderPipeline, dev: &Device, what: &str) {
    for m in &p.stages {
        check_words_valid(&m.spirv, &p.requirements, dev.profile())
            .unwrap_or_else(|e| panic!("{what} {:?}: L3: {e}", m.stage));
    }
}

/// Whether the key flow's vertex inputs fetch attribute 0 in fixed function.
fn fixed_function(inputs: &LoweringInputs) -> bool {
    inputs
        .vertex_fetch
        .as_ref()
        .and_then(|p| p.attributes.get(&0))
        .is_none_or(|f| matches!(f, VertexFetch::FixedFunction { .. }))
}

/// The vertex function lists the varyings it writes alike at every vertex of a primitive (a
/// constant buffer's or arithmetic's at the instance, the instance itself, and attribute 0 where
/// it steps per instance or at a constant step), never one of the vertex or of device memory;
/// the pipeline's key gives the fragment function those it interpolates, and the compile reads
/// them flat, where its key and the compiled pipeline's agree. A fixed-function attribute
/// stepped per instance is in the vertex function's key exactly where it makes a varying alike.
#[test]
fn a_fragment_reads_flat_what_its_vertex_function_writes_alike_at_every_vertex() {
    let Some(bc) = module() else {
        return;
    };
    let budget = Budget::standard();
    for stem in [
        "nvidia-rtx5080-laptop",
        "amd-780m-radv",
        "lavapipe",
        "moltenvk-m3",
    ] {
        let dev = device(stem);
        for (step, samples) in [
            (VertexStepFunction::PerVertex, 1),
            (VertexStepFunction::PerInstance, 1),
            (VertexStepFunction::Constant, 1),
            (VertexStepFunction::Constant, 4),
        ] {
            let what = format!("{stem} {step:?} at {samples}");
            let stepped = step != VertexStepFunction::PerVertex;
            let d = pipeline(&bc, step, samples);
            let k = dev
                .render_variant_key(&d, &VariantKey::new(), &budget, &StageEntries::new())
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            let vertex_inputs = k.vertex_inputs.clone().expect("vertex inputs");
            let instanced =
                step == VertexStepFunction::PerInstance && fixed_function(&vertex_inputs);
            assert_eq!(
                vertex_inputs.instanced_attributes,
                if instanced {
                    [0].into()
                } else {
                    BTreeSet::new()
                },
                "{what}"
            );
            let vertex = dev
                .compile_function(&function(&bc, "flt_vertex"), &vertex_inputs, &budget)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            assert_eq!(*vertex.key(), k.vertex, "{what}");
            let mut written = vec!["tint", "k", "h", "id"];
            let mut read = vec!["tint", "k", "h"];
            if stepped {
                written.push("a");
                read.push("a");
            }
            let io = &vertex.artifact().interface.io;
            assert_eq!(
                io.uniform_outputs.iter().cloned().collect::<BTreeSet<_>>(),
                keys(&written),
                "{what}"
            );
            assert_eq!(
                io.instanced_attributes,
                vertex_inputs
                    .instanced_attributes
                    .iter()
                    .copied()
                    .collect::<Vec<_>>(),
                "{what}"
            );
            let inputs = k.fragment_inputs.clone().expect("fragment inputs");
            assert_eq!(inputs.uniform_varyings, keys(&read), "{what}");
            let fragment = dev
                .compile_function(&function(&bc, "flt_fragment"), &inputs, &budget)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            assert_eq!(Some(*fragment.key()), k.fragment, "{what}");
            let io = &fragment.artifact().interface.io;
            assert_eq!(
                io.uniform_inputs.iter().cloned().collect::<BTreeSet<_>>(),
                keys(&read),
                "{what}"
            );
            for v in &io.inputs {
                let IoRole::Varying { interpolation, .. } = &v.role else {
                    continue;
                };
                let flat = keys(&read).contains(&v.key) || v.key == VaryingKey::user(b"user(id)");
                assert_eq!(
                    *interpolation == Some(Interpolation::Flat),
                    flat,
                    "{what} {:?}",
                    v.key
                );
            }
            let pipe = dev
                .compile_render_variant(&d, &VariantKey::new(), &budget)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            assert_eq!(pipe.key, k.pipeline, "{what}");
            valid(&pipe, &dev, &what);

            // The stages compiled on their own link as the pipeline.
            let mut separate = d.clone();
            separate.vertex = StageSource::Compiled(vertex.clone());
            separate.fragment = Some(StageSource::Compiled(fragment));
            let linked = dev
                .compile_render_pipeline(&separate, &budget)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            assert_eq!(linked.key, k.pipeline, "{what}");

            // A vertex function compiled to take attribute 0 as stepped per instance is refused
            // behind a descriptor that steps it per vertex.
            if instanced {
                let mut other = pipeline(&bc, VertexStepFunction::PerVertex, samples);
                other.vertex = StageSource::Compiled(vertex);
                assert_eq!(
                    dev.compile_render_pipeline(&other, &budget).err(),
                    Some(Error::InvalidDescriptor(DescriptorError::VertexAttribute {
                        attribute: 0
                    })),
                    "{what}"
                );
            }
        }
    }
}

/// A fragment function compiled to read flat a varying its pipeline's vertex function does not
/// write alike at every vertex is compiled for another pipeline: `NeedsInput(UniformVaryings)`,
/// never a wrong read. One compiled to read fewer of them flat links.
#[test]
fn a_fragment_compiled_for_another_vertex_function_is_refused() {
    let Some(bc) = module() else {
        return;
    };
    let budget = Budget::standard();
    let dev = device("nvidia-rtx5080-laptop");
    let reader = function(&bc, "flt_fragment");
    let linked = |step, uniform: BTreeSet<VaryingKey>| {
        let mut d = pipeline(&bc, step, 1);
        let k = dev
            .render_variant_key(&d, &VariantKey::new(), &budget, &StageEntries::new())
            .expect("keys");
        let mut inputs = k.fragment_inputs.expect("fragment inputs");
        inputs.uniform_varyings = uniform;
        let fragment = dev
            .compile_function(&reader, &inputs, &budget)
            .expect("fragment");
        d.fragment = Some(StageSource::Compiled(fragment));
        dev.compile_render_pipeline(&d, &budget)
    };
    // `a` is per vertex behind a per-vertex step, and `uv` always.
    for (forged, step) in [
        (keys(&["tint", "a"]), VertexStepFunction::PerVertex),
        (keys(&["uv"]), VertexStepFunction::Constant),
    ] {
        assert_eq!(
            linked(step, forged).err(),
            Some(Error::NeedsInput(RuntimeInput::UniformVaryings)),
            "{step:?}"
        );
    }
    assert!(linked(VertexStepFunction::Constant, keys(&["a", "tint"])).is_ok());
    assert!(linked(VertexStepFunction::Constant, BTreeSet::new()).is_ok());
}
