//! Every public field of the translator's linked render and compute state has the runtime's
//! verdict, so a field the translator adds cannot be dropped silently.
//!
//! The runtime receives `RenderPipeline` (its `GraphicsState` and `StageModule`s) and
//! `ComputePipeline` (its `ComputePipelineMetalInfo` and `StageModule`). Each public field of the
//! five structs takes one verdict here:
//!
//! - [`Applied`](Verdict::Applied): the runtime builds what it states, so its product sources
//!   read the field;
//! - [`Refused`](Verdict::Refused): the runtime refuses what it states, typed (`unbuilt`,
//!   `refuse` and `draws` in `render.rs`, `interface` in `pipeline.rs`, or a draw's refusal), for
//!   the reason given;
//! - [`Fact`](Verdict::Fact): a fact with nothing for the runtime to apply, for the reason given
//!   (the module carries it, the runtime's inputs or its floor never state it, the call it serves
//!   is refused).
//!
//! The test reads the structs' fields from the translator's sources, so a field without a verdict,
//! or a verdict without a field, fails it.

use crate::tests::{crate_dir, product_sources};

/// What the runtime does with one field of the translator's state.
enum Verdict {
    Applied,
    Refused(&'static str),
    Fact(&'static str),
}

use Verdict::{Applied, Fact, Refused};

/// The draw's set-0 bindings, which `draws` refuses (`of_the_device`).
const SET_0: &str = "a set-0 binding, which draws() refuses";
/// What the runtime never asks the translator for.
const VARIANT: &str = "stated only under a VariantKey the runtime never asks for";

const GRAPHICS_STATE: &[(&str, Verdict)] = &[
    (
        "topology",
        Fact(
            "the runtime makes the same list topology from the descriptor's class and the \
             tessellation state; each draw's topology is dynamic state",
        ),
    ),
    ("color_attachments", Applied),
    ("depth_format", Applied),
    ("stencil_format", Applied),
    ("rasterization_samples", Applied),
    ("alpha_to_coverage", Applied),
    ("alpha_to_one", Applied),
    ("rasterizer_discard", Applied),
    ("dynamic_states", Applied),
    ("input_attachments", Refused(SET_0)),
    (
        "rasterization_order_attachment_access",
        Refused("set only beside input attachments, a set-0 binding draws() refuses"),
    ),
    (
        "max_vertex_amplification_count",
        Refused("refuse() refuses the descriptor's count other than 1"),
    ),
    (
        "declares_layer",
        Fact("it matters only to multiview, which refuse() refuses"),
    ),
    ("tessellation", Applied),
    ("fragment_imageblock", Applied),
    (
        "vertex_writes_memory",
        Fact("the runtime records each draw once, in no rate-mapped pass"),
    ),
    (
        "writes_viewport_index",
        Fact(
            "the runtime holds one viewport and refuses several; without multiViewport the \
             module stores no ViewportIndex",
        ),
    ),
    ("storage_attachments", Refused(SET_0)),
    ("shadow_stencil", Applied),
    ("line_band", Applied),
    ("region_fan_out", Applied),
    ("writes_point_size", Applied),
    (
        "writes_depth",
        Fact("it matters only in depth clamp mode, which the encoder refuses"),
    ),
    (
        "reads_barycentric_coord",
        Fact("the runtime draws every primitive's vertices in the guest's order"),
    ),
    (
        "reads_primitive_id",
        Fact("the runtime draws each Metal primitive as one Vulkan primitive"),
    ),
    (
        "fragment_writes_memory",
        Fact("the runtime stands for no depth or stencil test by a state every fragment passes"),
    ),
    (
        "color_outputs",
        Fact("the runtime takes every attachment of a draw's pass as written"),
    ),
    ("sample_mask", Applied),
    ("snapshot_reads", Refused(VARIANT)),
    ("logic_op", Applied),
];

const RENDER_PIPELINE: &[(&str, Verdict)] = &[
    (
        "key",
        Fact("the runtime keys a content by its descriptor and a lowering by its own key"),
    ),
    (
        "profile",
        Fact("the runtime compiles each pipeline on its own device"),
    ),
    ("stages", Applied),
    ("layout", Applied),
    ("vertex_input", Applied),
    ("graphics", Applied),
    ("dependencies", Applied),
    ("interface", Applied),
    ("requirements", Applied),
    (
        "variant_bits",
        Fact("the runtime specializes each stage by its own specialization layout"),
    ),
];

const STAGE_MODULE: &[(&str, Verdict)] = &[
    ("stage", Applied),
    ("entry_point", Applied),
    ("spirv", Applied),
    ("specialization", Applied),
    ("required_subgroup_size", Applied),
    ("stage_flags", Applied),
    (
        "reconvergence",
        Fact("the stage's precision notes carry it (DeviceReconvergence)"),
    ),
    ("precision_notes", Applied),
    ("strategies", Fact("a census of the lowering, for coverage")),
    (
        "constant_blob",
        Fact("under the identity domains the module holds its constant data"),
    ),
    (
        "rounded_colors",
        Fact("what the rounding costs, which the module pays"),
    ),
    (
        "specialized_notes",
        Fact("the runtime announces every note of precision_notes, these among them"),
    ),
    ("own_pixel_textures", Applied),
    ("snapshot_textures", Refused(VARIANT)),
    ("snapshot_refusals", Refused(VARIANT)),
    (
        "level_reads",
        Fact("the runtime takes a read of a texture as one of its whole bound view"),
    ),
    (
        "private_bytes",
        Fact("the bytes FunctionStorageAssumed's announcement may name"),
    ),
    ("stats", Fact("counts of the module, for diagnostics")),
];

const COMPUTE_PIPELINE: &[(&str, Verdict)] = &[
    (
        "key",
        Fact("the runtime keys a content by its descriptor and a lowering by its own key"),
    ),
    (
        "profile",
        Fact("the runtime compiles each pipeline on its own device"),
    ),
    ("stage", Applied),
    ("layout", Applied),
    ("dependencies", Applied),
    ("interface", Applied),
    ("metal_info", Applied),
    (
        "requirements",
        Fact("the translator's profile is the set device creation enables"),
    ),
    (
        "variant_bits",
        Fact("the runtime specializes the kernel by its own specialization layout"),
    ),
];

const COMPUTE_METAL_INFO: &[(&str, Verdict)] = &[
    ("max_total_threads_per_threadgroup", Applied),
    ("metal_max_total_threads_per_threadgroup", Applied),
    ("thread_execution_width", Applied),
    ("static_threadgroup_memory_length", Applied),
    (
        "support_indirect_command_buffers",
        Refused("a descriptor that asks for indirect command buffers is refused"),
    ),
];

/// The public fields of `pub struct name` in `text`, in order, line comments removed.
fn fields(text: &str, name: &str) -> Vec<String> {
    let code: String = text
        .lines()
        .map(|line| line.split("//").next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n");
    let head = format!("pub struct {name} {{");
    let start = code
        .find(&head)
        .unwrap_or_else(|| panic!("no `{head}` in the translator's sources"))
        + head.len();
    let mut depth = 1;
    let mut end = start;
    for (at, c) in code[start..].char_indices() {
        depth += match c {
            '{' => 1,
            '}' => -1,
            _ => 0,
        };
        if depth == 0 {
            end = start + at;
            break;
        }
    }
    code[start..end]
        .lines()
        .filter_map(|line| line.trim_start().strip_prefix("pub "))
        .filter_map(|rest| rest.split_once(':'))
        .map(|(field, _)| field.trim().to_string())
        .filter(|field| field.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'))
        .collect()
}

/// Whether `code` reads `field` as a field (`.field` not followed by an identifier character or
/// a call's parenthesis).
fn reads(code: &str, field: &str) -> bool {
    let needle = format!(".{field}");
    code.match_indices(&needle).any(|(at, _)| {
        let next = code[at + needle.len()..].chars().next();
        !next.is_some_and(|c| c.is_ascii_alphanumeric() || c == '_' || c == '(')
    })
}

#[test]
fn every_field_of_the_translators_linked_state_has_the_runtimes_verdict() {
    let pipeline = crate_dir().join("../metal2vulkan/src/pipeline");
    let source = |file: &str| {
        let path = pipeline.join(file);
        std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()))
    };
    let (graphics, artifact) = (source("graphics.rs"), source("artifact.rs"));
    let code: String = product_sources()
        .iter()
        .flat_map(|(_, text)| text.lines())
        .map(|line| line.split("//").next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n");
    let structs = [
        ("GraphicsState", &graphics, GRAPHICS_STATE),
        ("RenderPipeline", &artifact, RENDER_PIPELINE),
        ("StageModule", &artifact, STAGE_MODULE),
        ("ComputePipeline", &artifact, COMPUTE_PIPELINE),
        ("ComputePipelineMetalInfo", &artifact, COMPUTE_METAL_INFO),
    ];
    let mut problems = Vec::new();
    for (name, text, verdicts) in structs {
        let declared = fields(text, name);
        assert!(!declared.is_empty(), "{name} declares no public field");
        for field in &declared {
            if !verdicts.iter().any(|(f, _)| f == field) {
                problems.push(format!(
                    "{name}::{field} has no verdict: the runtime applies it, refuses it typed, or \
                     it is a fact with nothing to apply (and why)"
                ));
            }
        }
        for (field, verdict) in verdicts {
            if !declared.iter().any(|f| f == field) {
                problems.push(format!("{name}::{field} has a verdict and no field"));
                continue;
            }
            match verdict {
                Applied if !reads(&code, field) => problems.push(format!(
                    "{name}::{field} is applied, but no product source reads it"
                )),
                Refused(why) | Fact(why) if why.is_empty() => {
                    problems.push(format!("{name}::{field} names no reason"))
                }
                _ => {}
            }
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

/// The parser takes a struct's public fields, its private ones and those of other structs not.
#[test]
fn the_field_reader_takes_one_structs_public_fields() {
    let text = "pub struct A {\n    /// `{ b }` in a doc.\n    pub b: u32,\n    c: u32,\n    \
                pub d: Option<Vec<u8>>, // e: f\n}\npub struct B {\n    pub e: u32,\n}\n";
    assert_eq!(fields(text, "A"), ["b", "d"]);
    assert_eq!(fields(text, "B"), ["e"]);
    assert!(reads("x.b + y.d", "b"));
    assert!(!reads("x.bc + y.b()", "b"));
}
