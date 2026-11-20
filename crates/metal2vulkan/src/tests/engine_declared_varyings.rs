//! A varying AIR declares reaches the module even when the shader never writes it.
//!
//! The old engine skipped the store of an output whose value was statically undefined. Its
//! unreferenced-global cleanup then removed the output variable, its `Location` and its interface
//! entry, and the fragment compiled from the same Metal struct consumed an input nothing produced
//! (27 vertex and 3 tessellation-evaluation sources of a 2880-source sample). AIR declares the
//! member because Metal declares it; that the shader leaves it undefined makes its value
//! undefined, not its existence (`tests/vertex_output_declared_locations.rs`, deleted with the old
//! engine in 0.2.0).
//!
//! The engine declares a stage's interface from its AIR signature (`pipeline::link::contract`
//! requires the module's interface variables to be exactly the interface list), and its serializer
//! prunes no global. These tests pin the pair end to end: a vertex function that leaves a declared
//! varying undefined, and a fragment function that reads it, link into a render pipeline, and the
//! vertex module's output locations are the fragment module's input locations.
//!
//! `llvm-as` assembles the templates; without it the tests skip visibly.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::library::Library;
use crate::metal::PixelFormat;
use crate::pipeline::descriptor::ColorAttachmentDescriptor;
use crate::pipeline::testing::{llvm_assemble, AssembleError};
use crate::pipeline::{FunctionDescriptor, RenderPipelineDescriptor, StageSource};
use crate::Budget;

const HEADER: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"
"#;

const TRAILER: &str = r#"!air.version = !{!93}
!air.language_version = !{!94}
!93 = !{i32 2, i32 8, i32 0}
!94 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// `struct V { float4 position [[position]]; float2 uv; float2 unwritten; }`: the vertex function
/// writes `position` and `uv` and leaves `unwritten` at its `undef` initial value (legal Metal, and
/// the shape a `[[function_constant]]`-gated member collapses to once the gated arm folds away).
/// The fragment function reads both varyings.
const SHADERS: &str = r#"define <{ <4 x float>, <2 x float>, <2 x float> }> @vert(i32 noundef %vid) {
entry:
  %f = uitofp i32 %vid to float
  %p = insertelement <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 1.000000e+00>, float %f, i32 0
  %uv = insertelement <2 x float> zeroinitializer, float %f, i32 1
  %r0 = insertvalue <{ <4 x float>, <2 x float>, <2 x float> }> undef, <4 x float> %p, 0
  %r1 = insertvalue <{ <4 x float>, <2 x float>, <2 x float> }> %r0, <2 x float> %uv, 1
  ret <{ <4 x float>, <2 x float>, <2 x float> }> %r1
}

define <4 x float> @frag(<4 x float> %pos, <2 x float> %uv, <2 x float> %unwritten) {
entry:
  %x = extractelement <2 x float> %uv, i32 0
  %y = extractelement <2 x float> %unwritten, i32 1
  %v0 = insertelement <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 1.000000e+00>, float %x, i32 0
  %v1 = insertelement <4 x float> %v0, float %y, i32 1
  ret <4 x float> %v1
}

!air.vertex = !{!0}
!air.fragment = !{!10}
!0 = !{ptr @vert, !1, !5}
!1 = !{!2, !3, !4}
!2 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!3 = !{!"air.vertex_output", !"generated(2uvDv2_f)", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
!4 = !{!"air.vertex_output", !"generated(9unwrittenDv2_f)", !"air.arg_type_name", !"float2", !"air.arg_name", !"unwritten"}
!5 = !{!6}
!6 = !{i32 0, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vid"}
!10 = !{ptr @frag, !11, !13}
!11 = !{!12}
!12 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4", !"air.arg_name", !"color"}
!13 = !{!14, !15, !16}
!14 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!15 = !{i32 1, !"air.fragment_input", !"generated(2uvDv2_f)", !"air.center", !"air.perspective", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
!16 = !{i32 2, !"air.fragment_input", !"generated(9unwrittenDv2_f)", !"air.center", !"air.perspective", !"air.arg_type_name", !"float2", !"air.arg_name", !"unwritten"}
"#;

/// `llvm-as` of `text`, through a scratch file that is removed at once. `None` when `llvm-as` is
/// not installed (the reason is printed).
fn assemble(text: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "m2v-declared-varyings-{}-{}.ll",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let assembled = llvm_assemble(&path);
    std::fs::remove_file(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    match assembled {
        Ok(bitcode) => Some(bitcode),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejects an input:\n{why}\n{text}"),
    }
}

fn device(stem: &str) -> Device {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let description = DeviceDescription::from_khronos_json(&text, None).expect(stem);
    Device::from_queries(description, DevicePolicy::default()).expect(stem)
}

/// The `Location` of every user variable of storage class `class` in `words` (built-ins carry no
/// `Location`), sorted.
fn locations(words: &[u32], class: u32) -> Vec<u32> {
    const HEADER_WORDS: usize = 5;
    const OP_VARIABLE: u32 = 59;
    const OP_DECORATE: u32 = 71;
    const LOCATION: u32 = 30;
    let mut variables = Vec::new();
    let mut decorated = Vec::new();
    let mut at = HEADER_WORDS;
    while at < words.len() {
        let length = (words[at] >> 16) as usize;
        assert!(length > 0, "an instruction of zero words at {at}");
        match words[at] & 0xffff {
            OP_VARIABLE if words[at + 3] == class => variables.push(words[at + 2]),
            OP_DECORATE if words[at + 2] == LOCATION => {
                decorated.push((words[at + 1], words[at + 3]))
            }
            _ => {}
        }
        at += length;
    }
    let mut out: Vec<u32> = decorated
        .into_iter()
        .filter(|(id, _)| variables.contains(id))
        .map(|(_, location)| location)
        .collect();
    out.sort_unstable();
    out
}

/// Profiles with and without device addresses, and the floor.
const PROFILES: [&str; 3] = ["nvidia-rtx5080-laptop", "moltenvk-m3", "runtime-floor"];

#[test]
fn an_unwritten_varying_has_a_producer_for_the_fragment_that_reads_it() {
    const INPUT: u32 = 1;
    const OUTPUT: u32 = 3;
    let Some(bitcode) = assemble(&format!("{HEADER}\n{SHADERS}\n{TRAILER}")) else {
        return;
    };
    for stem in PROFILES {
        let device = device(stem);
        let library = Library::from_air(&bitcode).expect("the shaders decode");
        let vertex = library.function("vert").expect("the vertex function");
        let fragment = library.function("frag").expect("the fragment function");
        let mut descriptor =
            RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(vertex)));
        descriptor.fragment = Some(StageSource::Function(FunctionDescriptor::new(fragment)));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        let pipeline = device
            .compile_render_pipeline(&descriptor, &Budget::default())
            .unwrap_or_else(|e| panic!("{stem}: the pipeline links: {e:?}"));
        let [vertex, fragment] = &pipeline.stages[..] else {
            panic!("{stem}: a vertex and a fragment stage");
        };
        let written = locations(&vertex.spirv, OUTPUT);
        let read = locations(&fragment.spirv, INPUT);
        assert_eq!(
            written.len(),
            2,
            "{stem}: both declared varyings are output variables, the unwritten one included: \
             {written:?}"
        );
        assert_eq!(
            written, read,
            "{stem}: the vertex module produces {written:?} and the fragment module reads {read:?}"
        );
    }
}
