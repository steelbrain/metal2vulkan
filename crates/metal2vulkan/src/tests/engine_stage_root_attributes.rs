//! Every attribute on a stage root reaches its effect, or the compile refuses it, on every stage.
//!
//! All three stage roots (`!air.kernel`, `!air.vertex`, `!air.fragment`) state their per-entry
//! attributes in one place: extra operands on the root node, past the `(function, outputs,
//! inputs)` triple. They do not state them in one form. `air.max_work_group_size` and `air.patch`
//! are references to a keyed node, and `early_fragment_tests` is a bare string operand on the root
//! itself. The old engine's decode once read the tail for one stage only, and two attributes went
//! unread: 51 of a 2880-source sample declared early depth testing and were emitted with Vulkan's
//! late test, which runs every store of a depth-rejected fragment (`tests/stage_root_attributes.rs`,
//! deleted with the old engine in 0.2.0).
//!
//! The engine reads the tail in one place (`air::abi`'s `stage_attribute`): each attribute is
//! read only by the stage that models it, and anything else is
//! `InvalidInput(Abi(UnknownStageAttribute(key)))` at the entry's metadata, naming the key the root
//! states, as the old engine's refusal named it. These tests hold that from the public API, for both forms of operand and
//! for every stage: an attribute nobody models, and one that another stage models (a kernel
//! carrying `early_fragment_tests` is not a kernel with early depth testing). Each template
//! compiles with a bare root and with the attribute its own stage models, so each refusal is the
//! attribute's.
//!
//! The attributes that are read reach their effect end to end: `early_fragment_tests` is the
//! module's `EarlyFragmentTests` execution mode, and only when declared (early testing is
//! observable: it suppresses a depth-rejected fragment's stores, so adding it unasked is as wrong
//! as dropping it); `air.max_work_group_size` bounds the threadgroup size the compute pipeline
//! reports. The old guard also refused a dispatch past the ceiling at translation; the engine
//! has no dispatch at compile time, and reports the bound instead (`ComputePipelineMetalInfo`).
//!
//! `llvm-as` assembles the templates; without it the tests skip visibly.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{AbiError, Error, InputErrorKind};
use crate::library::Library;
use crate::pipeline::testing::{link_standalone, llvm_assemble, AssembleError};
use crate::pipeline::{ComputePipelineDescriptor, FunctionDescriptor, LoweringInputs, StageSource};
use crate::Budget;

const HEADER: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"
"#;

/// The module-level metadata every template ends with (numbered from 90 up, clear of the
/// templates' own, and of the attribute node `!80`).
const TRAILER: &str = r#"!air.version = !{!93}
!air.language_version = !{!94}
!93 = !{i32 2, i32 8, i32 0}
!94 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// A fragment that writes a device buffer and returns a color. `ATTR` is spliced into the root's
/// attribute tail.
const FRAGMENT: &str = r#"define <4 x float> @frag(<4 x float> %pos, ptr addrspace(1) noundef %out) {
entry:
  store float 1.000000e+00, ptr addrspace(1) %out, align 4
  ret <4 x float> %pos
}

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !3ATTR}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4", !"air.arg_name", !"color"}
!3 = !{!4, !5}
!4 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#;

/// A kernel that writes a device buffer.
const KERNEL: &str = r#"define void @k(ptr addrspace(1) noundef %out) {
entry:
  store float 1.000000e+00, ptr addrspace(1) %out, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2ATTR}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#;

/// An ordinary vertex function: its root carries `ATTR` where a post-tessellation one carries
/// `air.patch`.
const VERTEX: &str = r#"define <4 x float> @vert(i32 noundef %vid) {
entry:
  %f = uitofp i32 %vid to float
  %v0 = insertelement <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 1.000000e+00>, float %f, i32 0
  ret <4 x float> %v0
}

!air.vertex = !{!0}
!0 = !{ptr @vert, !1, !3ATTR}
!1 = !{!2}
!2 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!3 = !{!4}
!4 = !{i32 0, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vid"}
"#;

/// The attribute operand spliced into a root, the node definition it needs (if any), and the key
/// the root states, which a refusal names.
struct Attribute {
    label: &'static str,
    operand: &'static str,
    node: &'static str,
    key: &'static str,
}

const UNKNOWN_STRING: Attribute = Attribute {
    label: "an unknown string attribute",
    operand: r#", !"air.invented_stage_attribute""#,
    node: "",
    key: "air.invented_stage_attribute",
};

const UNKNOWN_NODE: Attribute = Attribute {
    label: "an unknown keyed attribute",
    operand: ", !80",
    node: "!80 = !{!\"air.invented_stage_attribute\", i32 1}\n",
    key: "air.invented_stage_attribute",
};

const EARLY_FRAGMENT_TESTS: Attribute = Attribute {
    label: "early_fragment_tests",
    operand: r#", !"early_fragment_tests""#,
    node: "",
    key: "early_fragment_tests",
};

const MAX_WORK_GROUP_SIZE: Attribute = Attribute {
    label: "air.max_work_group_size",
    operand: ", !80",
    node: "!80 = !{!\"air.max_work_group_size\", i32 64}\n",
    key: "air.max_work_group_size",
};

const PATCH: Attribute = Attribute {
    label: "air.patch",
    operand: ", !80",
    node: "!80 = !{!\"air.patch\", !\"triangle\", !\"air.patch_control_point\", i32 3}\n",
    key: "air.patch",
};

const BARE: Attribute = Attribute {
    label: "a bare root",
    operand: "",
    node: "",
    key: "",
};

/// One stage's template, its entry, the attribute its own stage models, and the attributes of
/// the other stages.
struct StageCase {
    stage: &'static str,
    template: &'static str,
    entry: &'static str,
    own: &'static Attribute,
    others: [&'static Attribute; 2],
}

const STAGES: [StageCase; 3] = [
    StageCase {
        stage: "fragment",
        template: FRAGMENT,
        entry: "frag",
        own: &EARLY_FRAGMENT_TESTS,
        others: [&MAX_WORK_GROUP_SIZE, &PATCH],
    },
    StageCase {
        stage: "kernel",
        template: KERNEL,
        entry: "k",
        own: &MAX_WORK_GROUP_SIZE,
        others: [&EARLY_FRAGMENT_TESTS, &PATCH],
    },
    StageCase {
        stage: "vertex",
        template: VERTEX,
        entry: "vert",
        own: &PATCH,
        others: [&EARLY_FRAGMENT_TESTS, &MAX_WORK_GROUP_SIZE],
    },
];

/// `template` with `attribute` spliced into its root, as a whole module.
fn module(template: &str, attribute: &Attribute) -> String {
    let spliced = template.replace("ATTR", attribute.operand);
    assert!(!template.contains("!80 ="), "the template defines !80");
    assert_ne!(spliced, template, "the template has no ATTR splice point");
    format!("{HEADER}\n{spliced}{}{TRAILER}", attribute.node)
}

/// `llvm-as` of `text`, through a scratch file that is removed at once. `None` when `llvm-as` is
/// not installed (the reason is printed).
fn assemble(text: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "m2v-stage-root-attributes-{}-{}.ll",
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
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejects a template:\n{why}\n{text}"),
    }
}

fn device() -> Device {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/nvidia-rtx5080-laptop.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let description =
        DeviceDescription::from_khronos_json(&text, None).expect("nvidia-rtx5080-laptop");
    Device::from_queries(description, DevicePolicy::RESIDENT_TABLES).expect("nvidia-rtx5080-laptop")
}

/// Load `bitcode` as a library and compile its one function: the first refusal on the way, from
/// the library or the compile. `None` when `llvm-as` is missing.
fn compile(text: &str, device: &Device) -> Option<Result<(), Error>> {
    let bitcode = assemble(text)?;
    Some((|| {
        let library = Library::from_air(&bitcode)?;
        let function = FunctionDescriptor::new(library.function_at(0)?);
        device
            .compile_function(&function, &LoweringInputs::new(), &Budget::default())
            .map(|_| ())
    })())
}

/// The key `error` names if it is the ABI decode's refusal of an unread attribute on entry
/// `entry`'s root: `UnknownStageAttribute` at the entry's metadata, not at one of its arguments.
fn unread_attribute(error: &Error, entry: &str) -> Option<String> {
    let Error::InvalidInput(input) = error else {
        return None;
    };
    let InputErrorKind::Abi(AbiError::UnknownStageAttribute(key)) = &input.kind else {
        return None;
    };
    input
        .at
        .in_metadata()
        .is_some_and(|(at, arg, _)| {
            at.is_some_and(|name| name.to_string() == entry) && arg.is_none()
        })
        .then(|| key.to_string())
}

#[test]
fn every_template_compiles_with_a_bare_root() {
    let device = device();
    for case in &STAGES {
        let Some(outcome) = compile(&module(case.template, &BARE), &device) else {
            return;
        };
        if let Err(error) = outcome {
            panic!(
                "the {} template must compile with {}: {error:?}",
                case.stage, BARE.label
            );
        }
    }
}

/// The attribute a stage models is read, not refused: the control for the refusals below. A
/// vertex function with `air.patch` is a post-tessellation one, which this template's inputs do
/// not satisfy, so for it only the decode's acceptance is checked.
#[test]
fn each_stage_reads_the_attribute_it_models() {
    let device = device();
    for case in &STAGES {
        let Some(outcome) = compile(&module(case.template, case.own), &device) else {
            return;
        };
        match outcome {
            Ok(()) => {}
            Err(error) if case.stage == "vertex" => assert!(
                unread_attribute(&error, case.entry).is_none(),
                "the vertex root's {} was refused as unknown: {error:?}",
                case.own.label
            ),
            Err(error) => panic!(
                "the {} template must compile with {}: {error:?}",
                case.stage, case.own.label
            ),
        }
    }
}

/// An attribute no stage models is refused by every stage, in both operand forms. Silence here is
/// what let `early_fragment_tests` and `air.max_work_group_size` go unread in the old engine.
#[test]
fn an_attribute_no_stage_models_is_refused_by_every_stage() {
    let device = device();
    for case in &STAGES {
        for attribute in [&UNKNOWN_STRING, &UNKNOWN_NODE] {
            let Some(outcome) = compile(&module(case.template, attribute), &device) else {
                return;
            };
            match outcome {
                Ok(()) => panic!("the {} root accepted {}", case.stage, attribute.label),
                Err(error) => assert_eq!(
                    unread_attribute(&error, case.entry).as_deref(),
                    Some(attribute.key),
                    "the {} root's {} must be refused naming its key at entry {}: {error:?}",
                    case.stage,
                    attribute.label,
                    case.entry
                ),
            }
        }
    }
}

/// An attribute another stage models is still unmodeled here: a kernel carrying
/// `early_fragment_tests` is not a kernel with early depth testing, and a fragment carrying
/// `air.max_work_group_size` has no threadgroup to bound.
#[test]
fn an_attribute_belonging_to_another_stage_is_refused() {
    let device = device();
    for case in &STAGES {
        for attribute in case.others {
            let Some(outcome) = compile(&module(case.template, attribute), &device) else {
                return;
            };
            match outcome {
                Ok(()) => panic!("the {} root accepted {}", case.stage, attribute.label),
                Err(error) => assert_eq!(
                    unread_attribute(&error, case.entry).as_deref(),
                    Some(attribute.key),
                    "the {} root's {} must be refused naming its key at entry {}: {error:?}",
                    case.stage,
                    attribute.label,
                    case.entry
                ),
            }
        }
    }
}

/// The execution modes the final words declare.
fn execution_modes(words: &[u32]) -> Vec<u32> {
    const HEADER_WORDS: usize = 5;
    const OP_EXECUTION_MODE: u32 = 16;
    let mut modes = Vec::new();
    let mut at = HEADER_WORDS;
    while at < words.len() {
        let length = (words[at] >> 16) as usize;
        assert!(length > 0, "an instruction of zero words at {at}");
        if words[at] & 0xffff == OP_EXECUTION_MODE {
            modes.push(words[at + 2]);
        }
        at += length;
    }
    modes
}

/// `early_fragment_tests` is the `EarlyFragmentTests` execution mode of the linked module, and a
/// fragment that did not declare it does not get it. Both keep `OriginUpperLeft`.
#[test]
fn early_fragment_tests_reaches_the_execution_mode_only_when_declared() {
    const MODE_ORIGIN_UPPER_LEFT: u32 = 7;
    const MODE_EARLY_FRAGMENT_TESTS: u32 = 9;
    let device = device();
    for (attribute, early) in [(&BARE, false), (&EARLY_FRAGMENT_TESTS, true)] {
        let Some(bitcode) = assemble(&module(FRAGMENT, attribute)) else {
            return;
        };
        let library = Library::from_air(&bitcode).expect("the fragment template decodes");
        let function = FunctionDescriptor::new(library.function_at(0).expect("the fragment"));
        let compiled = device
            .compile_function(&function, &LoweringInputs::new(), &Budget::default())
            .unwrap_or_else(|e| panic!("the fragment with {} compiles: {e:?}", attribute.label));
        let (_, words) = link_standalone(compiled.artifact(), device.profile())
            .unwrap_or_else(|e| panic!("the fragment with {} links: {e:?}", attribute.label));
        let modes = execution_modes(&words);
        assert!(
            modes.contains(&MODE_ORIGIN_UPPER_LEFT),
            "{}: {modes:?}",
            attribute.label
        );
        assert_eq!(
            modes.contains(&MODE_EARLY_FRAGMENT_TESTS),
            early,
            "{}: {modes:?}",
            attribute.label
        );
    }
}

/// The threadgroup size a compute pipeline reports is at most the kernel's declared
/// `air.max_work_group_size`; with no declaration, or Metal's 0 ("the device maximum"), it is the
/// device's own bound.
#[test]
fn the_declared_threadgroup_ceiling_bounds_the_reported_maximum() {
    const ZERO: Attribute = Attribute {
        label: "air.max_work_group_size 0",
        operand: ", !80",
        node: "!80 = !{!\"air.max_work_group_size\", i32 0}\n",
        key: "air.max_work_group_size",
    };
    let device = device();
    let reported = |attribute: &Attribute| -> Option<u32> {
        let bitcode = assemble(&module(KERNEL, attribute))?;
        let library = Library::from_air(&bitcode).expect("the kernel template decodes");
        let function = FunctionDescriptor::new(library.function_at(0).expect("the kernel"));
        let budget = Budget::default();
        let compiled = device
            .compile_function(&function, &LoweringInputs::new(), &budget)
            .unwrap_or_else(|e| panic!("the kernel with {} compiles: {e:?}", attribute.label));
        let descriptor = ComputePipelineDescriptor::new(StageSource::Compiled(compiled));
        let pipeline = device
            .compile_compute_pipeline(&descriptor, &budget)
            .unwrap_or_else(|e| panic!("the kernel with {} links: {e:?}", attribute.label));
        Some(pipeline.metal_info.max_total_threads_per_threadgroup)
    };
    let Some(bare) = reported(&BARE) else {
        return;
    };
    assert!(bare > 64, "the device's own maximum is {bare}");
    assert_eq!(reported(&MAX_WORK_GROUP_SIZE), Some(64));
    assert_eq!(reported(&ZERO), Some(bare));
}
