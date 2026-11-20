//! A kernel that samples with a `min_lod_clamp`.
//!
//! Apple's compiler service crashes building the compute pipeline of a kernel that samples a
//! texture at an implicit level with a positive or NaN `min_lod_clamp`: `makeComputePipelineState`
//! fails with `XPC_ERROR_CONNECTION_INTERRUPTED` (macOS 26.7, M3 Max, Metal 32023.921). Measured:
//!
//! - `a2_kernel_lod_filter` itself, by the oracle's route for a public fixture (`metal-as`,
//!   `metallib`, the compute pipeline): its clamp-only call and its bias-and-clamp call each crash
//!   it, at its run-time clamp and at a constant 1; with both clamps zeroed it builds. Its case
//!   (`a9d1ba8d`) was interrupted the same way at oracle v35. With both clamps set to one
//!   constant, it crashes at 2^-24, 0.25, 1, 2, 15, 1000, +infinity, and a quiet NaN
//!   (`0x7FF8000000000000`), and builds at 0, -0.25, -1, -1000, and -infinity.
//! - Kernels in MSL: a constant and a run-time clamp, with and without a bias, on a 2D texture,
//!   and a run-time clamp on a 2D array, a 3D, and a cube texture and in a 2D depth texture's
//!   `sample_compare`, all crash; so did the Apple report's bicubic, linear, nearest, and argument
//!   samplers at a constant clamp. `min_lod_clamp(0.0f)`, `bias` alone, a plain sample, and a
//!   compare without a clamp build, and so does a sample with gradients and a run-time clamp.
//! - Fragment functions build with every one of those 2D forms, and a vertex function with
//!   gradients and a clamp builds.
//!
//! So Metal has no answer for such a kernel: [`ToolingRequirement::KernelMinLodClamp`] is Metal's
//! own limit ([`Limit::Metal`]), which stops every executor and closes its row without a reference.
//!
//! The scan claims only the measured forms, since the requirement closes a row. Every measured
//! kernel was a compute kernel: a kernel with no imageblock, built by `makeComputePipelineState`. A
//! tile function (a kernel with implicit imageblock attachments, built as a render pipeline's tile
//! stage) and a kernel with `[[imageblock]]` tiles were not measured, so the scan is given the
//! declared reflection of the entry and claims nothing for either ([`compute_kernel`]), nor for a
//! source that does not reflect. It reads the entry's text, as classification does (the
//! module, or its projection onto the entry for a module that declares several), in the entry's
//! function and the functions it reaches by direct calls, which its pipeline compiles. A call
//! counts when the product decodes its `air.*` name as a sample at an implicit level (`i1 false` at
//! its `TexRole::LodMode` position: no option, or a bias) of a 2D, 2D array, 3D, or cube colour
//! texture, or as a compare sample of a 2D depth texture, and its operand at `TexRole::MinLodClamp`
//! is a value computed at run time or a constant that is not at most zero: positive (finite or
//! +infinity) or a NaN, the split the measured constants follow. Positive constants outside the
//! measured 2^-24 to 1000, and NaNs of another sign or payload, are claimed by that split,
//! unmeasured; -0.0, which compares equal to zero, is no claim, as zero and every measured negative
//! constant build. A sample with gradients is no claim (it builds), and neither is an explicit
//! level, another texture shape, a clamp spelled otherwise (`undef`, `poison`, a constant
//! expression), or a call whose operands the layout does not read: none was measured, so such a row
//! stays authorable, and a case of it meets Metal's answer, or its failure, at the oracle. Every
//! measured call was in the entry's own body. A call in a function the entry reaches by direct
//! calls is claimed as one there, unmeasured, since the pipeline compiles both; one in a function
//! the entry reaches only through a pointer (a function table, an indirect call) or not at all is
//! no claim.
//!
//! [`Limit::Metal`]: crate::requirement::Limit::Metal

use super::{instruction, split_top_level};
use crate::air::reflect::{ShaderInterface, ShaderStage};
use crate::requirement::ToolingRequirement;
use metal2vulkan_air::intrinsics::{AirIntrinsic, ImageDim, ImageShape, TexOp, TexRole, Texture};
use std::collections::{BTreeMap, BTreeSet};

/// [`ToolingRequirement::KernelMinLodClamp`] when `reflection`, the declared reflection of the
/// source's entry, is a compute kernel's ([`compute_kernel`]) and a call its entry's pipeline
/// compiles samples in a measured form with a clamp ([`samples_with_min_lod_clamp`]).
pub fn kernel_min_lod_clamp_requirement(
    source_ll: &str,
    reflection: &ShaderInterface,
) -> Option<ToolingRequirement> {
    (compute_kernel(reflection)
        && reflection
            .entry_point
            .as_deref()
            .is_some_and(|entry| samples_with_min_lod_clamp(source_ll, entry)))
    .then_some(ToolingRequirement::KernelMinLodClamp)
}

/// Whether `reflection` is a compute kernel's, the only pipeline whose refusal was measured: a
/// kernel with no imageblock. A kernel with implicit imageblock attachments is a tile function,
/// and neither its tile pipeline nor a kernel with `[[imageblock]]` tiles was measured.
pub fn compute_kernel(reflection: &ShaderInterface) -> bool {
    reflection.stage == ShaderStage::Kernel
        && reflection.implicit_imageblock_attachments.is_empty()
        && reflection.imageblock_layouts.is_empty()
}

/// Whether a call in the function `entry` of `source_ll`, or in a function it reaches by direct
/// calls, samples at an implicit level, in a texture shape whose Metal refusal was measured, with
/// a `min_lod_clamp` Metal refuses. A module that does not define `entry` has none.
pub fn samples_with_min_lod_clamp(source_ll: &str, entry: &str) -> bool {
    if !source_ll.contains("@air.sample_") {
        return false;
    }
    let bodies = function_bodies(source_ll);
    let Some((&entry, _)) = bodies.get_key_value(entry) else {
        return false;
    };
    let mut reached = BTreeSet::from([entry]);
    let mut pending = vec![entry];
    while let Some(function) = pending.pop() {
        for callee in bodies[function]
            .iter()
            .filter_map(|text| direct_callee(text))
        {
            if let Some((&callee, _)) = bodies.get_key_value(callee) {
                if reached.insert(callee) {
                    pending.push(callee);
                }
            }
        }
    }
    reached
        .iter()
        .flat_map(|function| &bodies[function])
        .filter_map(|text| clamp_operand(text))
        .any(clamps)
}

/// The instruction texts of each function `source_ll` defines, by name.
pub(super) fn function_bodies(source_ll: &str) -> BTreeMap<&str, Vec<&str>> {
    let mut bodies = BTreeMap::new();
    let mut current = None::<(&str, Vec<&str>)>;
    for line in source_ll.lines() {
        if let Some(define) = line.strip_prefix("define ") {
            current = define
                .find('@')
                .and_then(|at| global_name(&define[at + 1..]))
                .map(|(name, _)| (name, Vec::new()));
        } else if line.starts_with('}') {
            if let Some((name, body)) = current.take() {
                bodies.insert(name, body);
            }
        } else if let Some((_, body)) = current.as_mut() {
            if let Some((_, text)) = instruction(line) {
                body.push(text);
            }
        }
    }
    bodies
}

/// The function a call instruction's text names directly (`call ... @f(...)`); `None` for any
/// other instruction and for a call through a pointer (`call ... %f(...)`). The callee is the
/// first named value directly followed by the operand list: a type's name (`%struct.t`) is not.
pub(super) fn direct_callee(text: &str) -> Option<&str> {
    let mut rest = &text[text.find("call ")? + "call ".len()..];
    loop {
        let at = rest.find(['@', '%'])?;
        let (name, after) = global_name(&rest[at + 1..])?;
        if after.starts_with('(') {
            return (rest.as_bytes()[at] == b'@').then_some(name);
        }
        rest = after;
    }
}

/// The name at the start of `text` (after its sigil), quoted or not, and the text after it.
fn global_name(text: &str) -> Option<(&str, &str)> {
    if let Some(quoted) = text.strip_prefix('"') {
        let end = quoted.find('"')?;
        return Some((&quoted[..end], &quoted[end + 1..]));
    }
    let end = text
        .find(|ch: char| !(ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '$' | '-')))
        .unwrap_or(text.len());
    Some((&text[..end], &text[end..]))
}

/// Whether Metal's refusal was measured for `texture`'s operation and shape: a sample (not one with
/// gradients) of a 2D, 2D array, 3D, or cube colour texture, or a compare sample of a 2D depth
/// texture.
fn measured_form(texture: &Texture) -> bool {
    let TexOp::Sample { compare, .. } = texture.op else {
        return false;
    };
    let ImageShape {
        dim,
        array,
        ms,
        depth,
    } = texture.image;
    !ms && match (dim, depth) {
        (ImageDim::D2, false) => !compare,
        (ImageDim::D3 | ImageDim::Cube, false) => !compare && !array,
        (ImageDim::D2, true) => compare && !array,
        _ => false,
    }
}

/// The `min_lod_clamp` operand of the call `text` when it samples in a measured form
/// ([`measured_form`]) at an implicit level: `None` for any other call, and for one whose operands
/// do not split into the layout's positions.
fn clamp_operand(text: &str) -> Option<&str> {
    let callee = &text[text.find("call ")?..];
    let name_start = callee.find("@air.")? + 1;
    let name_length = callee[name_start..].find('(')?;
    let name = &callee[name_start..name_start + name_length];
    let Some(AirIntrinsic::Texture(texture)) = AirIntrinsic::parse(name) else {
        return None;
    };
    if !measured_form(&texture) {
        return None;
    }
    let layout = texture.layout();
    let clamp = layout.index(TexRole::MinLodClamp)?;
    let lod_mode = layout.index(TexRole::LodMode)?;
    let arguments = &callee[name_start + name_length + 1..];
    let mut depth = 0usize;
    let end = arguments.char_indices().find_map(|(index, ch)| match ch {
        '(' => {
            depth += 1;
            None
        }
        ')' if depth == 0 => Some(index),
        ')' => {
            depth -= 1;
            None
        }
        _ => None,
    });
    let operands = split_top_level(&arguments[..end?]);
    let implicit_level = operands
        .get(lod_mode)
        .and_then(|operand| operand.split_whitespace().last())
        == Some("false");
    (operands.len() == layout.positions().count() && implicit_level)
        .then(|| operands.get(clamp).copied())
        .flatten()
}

/// Whether the typed operand `operand` (`float %15`, `float 1.000000e+00`, possibly with parameter
/// attributes before the value) is a clamp Metal refuses in a kernel: a value computed at run time,
/// or a constant that is not at most zero (positive, +infinity, or a NaN). Zero of either sign and
/// negative constants build, and any other spelling is no claim.
fn clamps(operand: &str) -> bool {
    let Some(value) = operand.split_whitespace().last() else {
        return false;
    };
    if value.starts_with('%') {
        return true;
    }
    // LLVM spells a float it cannot print exactly in decimal as the hex of its double.
    let constant = match value.strip_prefix("0x") {
        Some(bits) => u64::from_str_radix(bits, 16).ok().map(f64::from_bits),
        None => value.parse::<f64>().ok(),
    };
    constant.is_some_and(|constant| constant > 0.0 || constant.is_nan())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::requirement::{Executor, Limit};
    use crate::source::SourceRow;

    /// The declared interface of `row`'s entry, as classification takes it; `None` when it does
    /// not reflect.
    fn reflection(row: &SourceRow) -> Option<ShaderInterface> {
        let stage = match row.stage.as_str() {
            "Kernel" => crate::case::Stage::Kernel,
            "Vertex" => crate::case::Stage::Vertex,
            "Fragment" => crate::case::Stage::Fragment,
            _ => return None,
        };
        crate::air::reflect::inspect(row.air_ll.as_str(), stage, &row.entry, &[]).ok()
    }

    fn public_row(label: &str) -> SourceRow {
        crate::source::public_source_set()
            .unwrap()
            .rows
            .into_iter()
            .find(|row| row.label == label)
            .unwrap()
    }

    /// Whether the kernel `@k` of `module` samples with a clamp Metal refuses.
    fn claims(module: &str) -> bool {
        samples_with_min_lod_clamp(module, "k")
    }

    /// A kernel whose body is the one instruction `call`.
    fn kernel(call: &str) -> String {
        format!(
            "define void @k(ptr addrspace(1) %t, ptr addrspace(1) %out, float %c) {{\n  {call}\n  \
             ret void\n}}\n"
        )
    }

    fn sample(clamp: &str) -> String {
        format!(
            "%s = tail call {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1) \
             readonly captures(none) %t, ptr addrspace(2) readonly captures(none) \
             @__air_sampler_state, <2 x float> zeroinitializer, i1 true, <2 x i32> \
             zeroinitializer, i1 false, float 0.000000e+00, {clamp}, i32 0) #2"
        )
    }

    fn gradient(clamp: &str) -> String {
        format!(
            "%s = call {{ <4 x float>, i8 }} @air.sample_texture_2d_grad.v4f32(ptr addrspace(1) \
             %t, ptr addrspace(2) @__air_sampler_state, <2 x float> zeroinitializer, <2 x \
             float> zeroinitializer, <2 x float> zeroinitializer, {clamp}, i1 false, <2 x i32> \
             zeroinitializer, i32 0)"
        )
    }

    /// A sample's clamp operand is found by the product's layout. A run-time value and each
    /// constant class Metal was measured to refuse clamp (positive, +infinity, a NaN, spelled as
    /// LLVM spells them); zero of either sign and negative constants, measured to build, do not,
    /// and neither does a spelling no measurement covers or an operand list the layout does not
    /// read.
    #[test]
    fn a_clamp_is_a_run_time_value_or_a_constant_metal_refuses() {
        for clamp in [
            "float %c",
            "float noundef %c",
            "float 0x3E70000000000000",
            "float 2.500000e-01",
            "float 1.000000e+00",
            "float 0x3FF8000000000000",
            "float 1.000000e+03",
            "float 0x7FF0000000000000",
            "float 0x7FF8000000000000",
        ] {
            assert!(claims(&kernel(&sample(clamp))), "{clamp}");
        }
        for no_claim in [
            "float 0.000000e+00",
            "float -0.000000e+00",
            "float noundef 0.000000e+00",
            "float 0x0000000000000000",
            "float 0x8000000000000000",
            "float -2.500000e-01",
            "float -1.000000e+00",
            "float -1.000000e+03",
            "float 0xFFF0000000000000",
            "float undef",
            "float poison",
        ] {
            assert!(!claims(&kernel(&sample(no_claim))), "{no_claim}");
        }
        // One operand short of the layout: the clamp cannot be read, so it is no claim.
        let short = sample("float 1.000000e+00").replace(", i32 0)", ")");
        assert!(!claims(&kernel(&short)));
        // A gradient sample with a clamp builds on Metal, so it is no claim at any clamp.
        for clamp in ["float 1.000000e+00", "float %c"] {
            assert!(!claims(&kernel(&gradient(clamp))), "{clamp}");
        }
        // A sample with no clamp position (a gather), and no sample at all.
        let gather = "%g = call { <4 x float>, i8 } @air.gather_texture_2d.v4f32(ptr \
                      addrspace(1) %t, ptr addrspace(2) @s, <2 x float> zeroinitializer, i1 \
                      false, <2 x i32> zeroinitializer, i32 0, i32 0)";
        assert!(!claims(&kernel(gather)));
        assert!(!claims(&kernel("%x = fadd float %c, %c")));
    }

    /// A call of `name` whose operands are spelled from the product's layout: `lod_mode` at the
    /// LOD-mode position, `clamp` at the clamp position, and a placeholder elsewhere.
    fn laid_out(name: &str, lod_mode: &str, clamp: &str) -> String {
        let Some(AirIntrinsic::Texture(texture)) = AirIntrinsic::parse(name) else {
            panic!("{name} is no texture operation the product decodes");
        };
        let operands = texture
            .layout()
            .positions()
            .map(|(role, _)| match role {
                TexRole::MinLodClamp => clamp,
                TexRole::LodMode => lod_mode,
                _ => "i32 0",
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!("%s = call {{ <4 x float>, i8 }} @{name}({operands})")
    }

    /// The measured forms carry a run-time clamp at an implicit level: a sample of a 2D, 2D
    /// array, 3D, or cube texture and a compare sample of a 2D depth texture. An explicit level,
    /// a gradient sample (which builds), and the shapes not measured carry none.
    #[test]
    fn only_the_measured_sample_forms_carry_a_clamp() {
        for name in [
            "air.sample_texture_2d.v4f32",
            "air.sample_texture_2d_array.v4f32",
            "air.sample_texture_3d.v4f32",
            "air.sample_texture_cube.v4f32",
            "air.sample_compare_depth_2d.f32",
        ] {
            assert!(
                claims(&kernel(&laid_out(name, "i1 false", "float %c"))),
                "{name}"
            );
            assert!(
                !claims(&kernel(&laid_out(name, "i1 true", "float %c"))),
                "{name} at an explicit level"
            );
        }
        for name in [
            "air.sample_texture_2d_grad.v4f32",
            "air.sample_texture_cube_array.v4f32",
            "air.sample_compare_depth_2d_array.f32",
        ] {
            assert!(
                !claims(&kernel(&laid_out(name, "i1 false", "float %c"))),
                "{name}"
            );
        }
    }

    /// Over the committed public source rows (a row per entry), exactly two kernels carry the
    /// requirement, each read from its entry's text as classification reads it:
    /// `a2_kernel_lod_filter` (it samples with clamps read from its input buffer, measured to
    /// crash Metal's compiler) and `sampling_completion`'s `sample_bias_kernel` (its own body
    /// samples a 2D texture with `bias(0.5)` and `min_lod_clamp(1.0)`, a measured form). No other
    /// entry of `sampling_completion` carries it, so the requirement is the entry's own, not its
    /// module's. Classification gives it to the row, and the checker's shared check refuses any
    /// case of it, because Metal cannot build its pipeline. `bcu_forms`, which samples a bicubic
    /// kernel with `bias` instead, carries none.
    #[test]
    fn the_committed_kernels_that_sample_with_a_clamp_carry_it() {
        let rows = crate::source::public_source_set().unwrap().rows;
        let carrying = rows
            .iter()
            .filter(|row| {
                let entry = crate::triage::entry_row(row).unwrap();
                reflection(&entry).is_some_and(|reflection| {
                    kernel_min_lod_clamp_requirement(&entry.air_ll, &reflection).is_some()
                })
            })
            .map(|row| (row.label.as_str(), row.entry.as_str()))
            .collect::<Vec<_>>();
        assert_eq!(
            carrying,
            [
                (
                    "public/engine/a2_kernel_lod_filter.ll",
                    "a2_kernel_lod_filter"
                ),
                ("public/engine/sampling_completion.ll", "sample_bias_kernel"),
            ]
        );
        for row in rows
            .iter()
            .filter(|row| row.label == "public/engine/sampling_completion.ll")
        {
            assert_eq!(
                crate::triage::classify_summary(row)
                    .unwrap()
                    .tooling_requirements
                    .contains(&ToolingRequirement::KernelMinLodClamp),
                row.entry == "sample_bias_kernel",
                "{}",
                row.entry
            );
        }

        let row = rows
            .iter()
            .find(|row| row.label == "public/engine/a2_kernel_lod_filter.ll")
            .unwrap();
        assert!(crate::triage::classify_summary(row)
            .unwrap()
            .tooling_requirements
            .contains(&ToolingRequirement::KernelMinLodClamp));
        let interface = reflection(row).unwrap();
        let requirements = super::super::unsupported_source_requirements(&row.air_ll, &interface);
        assert!(requirements.contains(&ToolingRequirement::KernelMinLodClamp));
        let gap = super::super::evidence_gap(&requirements).unwrap();
        assert!(gap.contains("kernel_min_lod_clamp"), "{gap}");
        assert!(
            gap.contains("Metal cannot build the case's pipeline"),
            "{gap}"
        );

        let bicubic = rows
            .iter()
            .find(|row| row.label == "public/engine/bcu_forms.ll")
            .unwrap();
        assert_eq!(
            kernel_min_lod_clamp_requirement(&bicubic.air_ll, &reflection(bicubic).unwrap()),
            None
        );
    }

    /// Only a compute kernel carries the requirement. It stops every executor, since Metal cannot
    /// build the pipeline and no candidate has a reference to be compared with, and it is Metal's
    /// own limit, so it closes its row instead of keeping the census open.
    #[test]
    fn only_a_compute_kernel_carries_the_requirement_and_it_is_metals_limit() {
        let row = public_row("public/engine/a2_kernel_lod_filter.ll");
        let mut reflection = reflection(&row).unwrap();
        assert_eq!(
            kernel_min_lod_clamp_requirement(&row.air_ll, &reflection),
            Some(ToolingRequirement::KernelMinLodClamp)
        );
        reflection.stage = ShaderStage::Fragment;
        assert_eq!(
            kernel_min_lod_clamp_requirement(&row.air_ll, &reflection),
            None
        );
        assert_eq!(ToolingRequirement::KernelMinLodClamp.stops(), Executor::ALL);
        assert_eq!(ToolingRequirement::KernelMinLodClamp.limit(), Limit::Metal);
        assert!(!ToolingRequirement::KernelMinLodClamp.blocks_closure());
    }

    /// The scan reads the entry's function and the functions it reaches by direct calls, through
    /// any depth: a clamp in a function the entry never calls, or calls only through a pointer,
    /// is no claim, and neither is a module that does not define the entry.
    #[test]
    fn only_the_functions_the_entry_reaches_by_direct_calls_are_scanned() {
        let clamped = sample("float %c");
        let module = |entry_body: &str| {
            format!(
                "define void @k(ptr addrspace(1) %t, float %c) {{\n  {entry_body}\n  ret \
                 void\n}}\n\ndefine internal fastcc void @middle(ptr addrspace(1) %t, float \
                 %c) {{\n  tail call fastcc void @\"leaf.helper\"(ptr addrspace(1) %t, float \
                 %c)\n  ret void\n}}\n\ndefine internal fastcc void @\"leaf.helper\"(ptr \
                 addrspace(1) %t, float %c) {{\n  {clamped}\n  ret void\n}}\n"
            )
        };
        // Two direct calls deep, through a quoted name.
        let direct = module("tail call fastcc void @middle(ptr addrspace(1) %t, float %c)");
        assert!(claims(&direct));
        // The same functions, never called; called through a pointer; or another entry.
        for unreached in [
            module("%x = fadd float %c, %c"),
            module("tail call fastcc void %fn(ptr addrspace(1) %t, float %c)"),
            module("store ptr @middle, ptr addrspace(1) %t, align 8"),
        ] {
            assert!(!claims(&unreached), "{unreached}");
            assert!(samples_with_min_lod_clamp(&unreached, "leaf.helper"));
        }
        assert!(!samples_with_min_lod_clamp(&direct, "absent"));
        // A call whose return type names a struct type still names its callee.
        assert_eq!(direct_callee("%r = call %struct.t @f(ptr @g)"), Some("f"));
        assert_eq!(direct_callee("%r = call %struct.t %f(ptr @g)"), None);
        assert_eq!(
            direct_callee("%r = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr %t)"),
            Some("air.sample_texture_2d.v4f32")
        );
    }

    /// A tile function is no claim: a kernel with implicit imageblock attachments that samples
    /// at an implicit level with a run-time clamp carries no requirement, through classification
    /// or the shared check, because its tile pipeline was never measured. The same module counts
    /// once its reflection has no imageblock, so the imageblock is what keeps it out.
    #[test]
    fn a_tile_function_that_samples_with_a_clamp_carries_none() {
        let mut row = public_row("public/kernel_implicit_imageblock_three_widths.ll");
        let call = "  %clamp = uitofp i32 %6 to float\n  %sampled = tail call { <4 x float>, i8 } \
                    @air.sample_texture_2d.v4f32(ptr addrspace(1) %1, ptr addrspace(2) null, <2 x \
                    float> zeroinitializer, i1 true, <2 x i32> zeroinitializer, i1 false, float \
                    0.000000e+00, float %clamp, i32 0)\n  ret void\n";
        row.air_ll = row.air_ll.replacen("  ret void\n", call, 1);
        row.air_ll.push_str(
            "declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr \
             addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)\n",
        );
        assert!(samples_with_min_lod_clamp(&row.air_ll, &row.entry));
        let triage = crate::triage::classify_summary(&row).unwrap();
        assert!(
            triage.tile_function,
            "the product reflects the tile function"
        );
        assert!(!triage
            .tooling_requirements
            .contains(&ToolingRequirement::KernelMinLodClamp));
        let mut reflection = reflection(&row).unwrap();
        assert!(
            !super::super::unsupported_source_requirements(&row.air_ll, &reflection)
                .contains(&ToolingRequirement::KernelMinLodClamp)
        );
        // Its imageblock parameter is reflected both ways: as the tile's implicit attachments
        // and as an imageblock layout.
        assert!(!reflection.implicit_imageblock_attachments.is_empty());
        assert!(!reflection.imageblock_layouts.is_empty());
        reflection.implicit_imageblock_attachments.clear();
        assert!(!compute_kernel(&reflection));
        reflection.imageblock_layouts.clear();
        assert_eq!(
            kernel_min_lod_clamp_requirement(&row.air_ll, &reflection),
            Some(ToolingRequirement::KernelMinLodClamp)
        );
    }
}
