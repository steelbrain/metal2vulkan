//! Vertex fetch, varyings, stage builtins, and fragment inputs and outputs end to end.
//!
//! The owned fixture `engine/stage_io` goes through the whole per-function pipeline: each
//! vertex and fragment entry compiles (or is refused with the capability it needs), and its
//! module passes L3 (`spirv-val` and requirements within the profile) under the stage-only
//! numbering link gives it alone. The vertex and fragment entries that share a varying struct link
//! as a render pipeline in `src/tests/engine_link.rs`. No GPU is used.
//!
//! `llvm-as` assembles the fixture; a missing tool skips visibly.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::budget::SPIRV_WORDS;
use crate::device::DevicePolicy;
use crate::error::{Capability, Error, Feature, Needs};
use crate::library::{FunctionArtifact, FunctionConstantValues};
use crate::metal::{VertexFormat, VertexStepFunction};
use crate::pipeline::compile::{compile_bitcode, plan_bitcode, EntryRef};
use crate::pipeline::descriptor::{
    VertexAttributeDescriptor, VertexBufferLayoutDescriptor, VertexDescriptor,
};
use crate::pipeline::link::vertex::vertex_fetch_projection;
use crate::pipeline::link::{link, LinkInputs, LinkTable};
use crate::pipeline::lower_inputs::{LoweringInputs, VertexFetch, VertexFetchProjection};
use crate::pipeline::relocate::relocate;
use crate::pipeline::stage_io::{
    DepthMode, Interpolation, IoNumeric, IoRole, Sampling, StageBuiltin,
};
use crate::pipeline::sym::{Symbol, VaryingKey};
use crate::pipeline::testing::{
    compile, engine_dir, llvm_assemble, meter, profile, zero_relocations, AssembleError, Scratch,
};
use crate::spirv::testing::check_words_valid;

const PROFILES: [&str; 3] = ["moltenvk-m3", "lunarg-desktop-baseline-2023", "vk12-floor"];

fn fixture() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join("stage_io.ll")) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected stage_io.ll: {why}"),
    }
}

fn prof(stem: &str) -> crate::device::DeviceProfile {
    profile(stem, DevicePolicy::default()).expect("committed profile")
}

/// Whether `words` hold an `OpDecorate <id> Invariant` (opcode 71, decoration 18).
fn decorates_invariant(words: &[u32]) -> bool {
    let mut i = 5;
    while let Some(&w) = words.get(i) {
        let (len, op) = ((w >> 16) as usize, w & 0xFFFF);
        if op == 71 && len == 3 && words.get(i + 2) == Some(&18) {
            return true;
        }
        if len == 0 {
            return false;
        }
        i += len;
    }
    false
}

/// The table link gives `a` alone (the stage-only numbering).
fn alone(a: &FunctionArtifact, p: &crate::device::DeviceProfile) -> LinkTable {
    link(&[a], &LinkInputs::standalone(), p)
        .unwrap_or_else(|e| panic!("link: {e}"))
        .table
}

/// Relocate `a` under `t` and run L3.
fn validate(a: &FunctionArtifact, t: &LinkTable, p: &crate::device::DeviceProfile) -> Vec<u32> {
    let words = relocate(a, t).expect("relocate");
    check_words_valid(&words, &a.requirements, p).unwrap_or_else(|e| panic!("L3: {e}"));
    words
}

/// The entries every profile compiles, alone, under their stage-only numbering.
#[test]
fn stage_entries_compile_and_validate_alone() {
    let Some(bc) = fixture() else { return };
    for stem in PROFILES {
        let p = prof(stem);
        for entry in [
            "stage_io_vertex",
            "stage_io_vertex_position",
            "stage_io_fragment",
        ] {
            let a = compile(&bc, entry, &p).unwrap_or_else(|e| panic!("{entry} on {stem}: {e}"));
            let t = alone(&a, &p);
            validate(&a, &t, &p);
        }
    }
}

/// The artifact states every `Location` key's role, the declared (32-bit) type, and the builtins;
/// narrow values keep their AIR width in `source_bits`.
#[test]
fn the_interface_contract_describes_every_key() {
    let Some(bc) = fixture() else { return };
    let p = prof("moltenvk-m3");
    let v = compile(&bc, "stage_io_vertex", &p).expect("vertex");
    let io = &v.interface.io;
    let attrs: Vec<(u32, IoNumeric, u8, u8)> = io
        .inputs
        .iter()
        .map(|x| match x.role {
            IoRole::Attribute { index, .. } => {
                assert_eq!(x.key, VaryingKey::attribute(index));
                (index, x.ty.numeric, x.ty.components, x.ty.source_bits)
            }
            ref r => panic!("vertex input role {r:?}"),
        })
        .collect();
    assert_eq!(
        attrs,
        vec![
            (0, IoNumeric::Float, 3, 32),
            (1, IoNumeric::Float, 2, 16),
            (4, IoNumeric::Uint, 2, 16),
            (7, IoNumeric::Uint, 4, 8),
        ]
    );
    assert_eq!(
        io.outputs.len(),
        3,
        "uv, tint, id (position and point size are builtins)"
    );
    for o in &io.outputs {
        let IoRole::Varying {
            user,
            interpolation,
        } = &o.role
        else {
            panic!("vertex output role {:?}", o.role);
        };
        assert_eq!(o.key, VaryingKey::user(&user.0));
        assert_eq!(*interpolation, None);
    }
    assert_eq!(
        io.builtins,
        vec![
            StageBuiltin::VertexIndex,
            StageBuiltin::InstanceIndex,
            StageBuiltin::Position { invariant: true },
            StageBuiltin::PointSize { default: false },
        ]
    );
    // Every Location reloc of the module is a key the interface states, and back.
    let mut keys: Vec<VaryingKey> = v
        .module
        .relocs
        .iter()
        .filter_map(|(s, _)| match s {
            Symbol::Location(k) => Some(k.clone()),
            _ => None,
        })
        .collect();
    keys.sort();
    let mut stated: Vec<VaryingKey> = io.variables().map(|x| x.key.clone()).collect();
    stated.sort();
    assert_eq!(keys, stated);

    let bare = compile(&bc, "stage_io_vertex_position", &p).expect("bare position");
    assert!(bare.interface.io.inputs.is_empty() && bare.interface.io.outputs.is_empty());
    assert_eq!(
        bare.interface.io.builtins,
        vec![
            StageBuiltin::VertexIndex,
            StageBuiltin::Position { invariant: true },
            StageBuiltin::PointSize { default: true },
        ],
        "a vertex function without a point-size output writes the default; its position is \
         invariant, `[[invariant]]` or not"
    );
    assert!(
        decorates_invariant(&validate(&bare, &alone(&bare, &p), &p)),
        "a position not `[[invariant]]` in Metal carries `Invariant` too"
    );

    let f = compile(&bc, "stage_io_fragment", &p).expect("fragment");
    let io = &f.interface.io;
    let interp: Vec<Option<Interpolation>> = io
        .inputs
        .iter()
        .map(|x| match &x.role {
            IoRole::Varying { interpolation, .. } => *interpolation,
            r => panic!("fragment input role {r:?}"),
        })
        .collect();
    assert_eq!(
        interp,
        vec![
            Some(Interpolation::Smooth {
                sampling: Sampling::Center,
                perspective: true
            }),
            Some(Interpolation::Smooth {
                sampling: Sampling::Centroid,
                perspective: false
            }),
            Some(Interpolation::Flat),
        ]
    );
    let colors: Vec<(u32, u32, IoNumeric)> = io
        .outputs
        .iter()
        .map(|x| match x.role {
            IoRole::Color { attachment, index } => {
                assert_eq!(x.key, VaryingKey::color(attachment, index));
                (attachment, index, x.ty.numeric)
            }
            ref r => panic!("fragment output role {r:?}"),
        })
        .collect();
    assert_eq!(
        colors,
        vec![(0, 0, IoNumeric::Float), (2, 0, IoNumeric::Uint)]
    );
    assert_eq!(
        io.builtins,
        vec![
            StageBuiltin::FragCoord,
            StageBuiltin::FrontFacing,
            StageBuiltin::SampleMaskIn,
            StageBuiltin::FragDepth(DepthMode::Greater),
            StageBuiltin::SampleMaskOut,
        ]
    );
}

/// Link changes only relocation literals. Two different numberings of the same stage give the same
/// words once the relocation slots are zeroed.
#[test]
fn locations_are_relocation_only() {
    let Some(bc) = fixture() else { return };
    let p = prof("moltenvk-m3");
    let v = compile(&bc, "stage_io_vertex", &p).expect("vertex");
    let t = alone(&v, &p);
    let mut shifted = t.clone();
    for l in shifted.locations.values_mut() {
        l.0 += 5;
    }
    let (a, b) = (validate(&v, &t, &p), validate(&v, &shifted, &p));
    assert_ne!(a, b);
    assert_eq!(zero_relocations(&v, &a), zero_relocations(&v, &b));
}

/// The refusals of the fixture's feature-dependent entries name the capability (the primitive id's
/// too), and the invariant position compiles with its `Invariant` decoration.
#[test]
fn feature_dependent_entries_name_their_capability() {
    let Some(bc) = fixture() else { return };
    let floor = prof("vk12-floor");
    let needs = |entry: &str, feature: Feature, cap: Capability| match compile(&bc, entry, &floor) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, feature, "{entry}");
            assert_eq!(u.needs, Needs::AnyOf(vec![vec![cap]]), "{entry}");
        }
        other => panic!("{entry}: {other:?}"),
    };
    use crate::device::requirements::VkFeature as F;
    needs(
        "stage_io_vertex_clip",
        Feature::ClipDistance,
        Capability::Feature(F::ShaderClipDistance),
    );
    // A vertex `Layer` takes the output feature or a geometry stage that writes it.
    match compile(&bc, "stage_io_vertex_layer", &floor) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::LayeredRendering);
            assert_eq!(
                u.needs,
                Needs::AnyOf(vec![
                    vec![Capability::Feature(F::ShaderOutputLayer)],
                    vec![Capability::Feature(F::GeometryShader)],
                ])
            );
        }
        other => panic!("stage_io_vertex_layer: {other:?}"),
    }
    needs(
        "stage_io_fragment_point",
        Feature::SampleRateShading,
        Capability::Feature(F::SampleRateShading),
    );
    // Without either enabling capability of a fragment `PrimitiveId`, the ID is the flat varying
    // the pipeline's vertex function writes (`engine_primitive_ids`).
    let a = compile(&bc, "stage_io_fragment_primitive", &floor).expect("primitive");
    assert!(a
        .interface
        .io
        .inputs
        .iter()
        .any(|v| v.role == crate::pipeline::stage_io::IoRole::PrimitiveId));
    needs(
        "stage_io_fragment_dual_source",
        Feature::DualSourceBlending,
        Capability::Feature(F::DualSrcBlend),
    );
    let m = prof("moltenvk-m3");
    for entry in [
        "stage_io_vertex_clip",
        "stage_io_vertex_layer",
        "stage_io_fragment_point",
        "stage_io_fragment_primitive",
        "stage_io_fragment_dual_source",
    ] {
        let a = compile(&bc, entry, &m).unwrap_or_else(|e| panic!("{entry}: {e}"));
        validate(&a, &alone(&a, &m), &m);
    }
    let a = compile(&bc, "stage_io_vertex_invariant", &m).expect("invariant position");
    assert_eq!(
        a.interface.io.builtins,
        vec![
            StageBuiltin::VertexIndex,
            StageBuiltin::Position { invariant: true },
            StageBuiltin::PointSize { default: true },
        ]
    );
    let words = validate(&a, &alone(&a, &m), &m);
    assert!(
        decorates_invariant(&words),
        "the position output carries `Invariant`"
    );
}

/// The vertex fetch projection: fixed function where the format and step have an exact Vulkan form,
/// pulled otherwise (a constant step always), with the source buffer, offset, and step in the key.
/// A pulled attribute compiles to an in-shader fetch; one whose source buffer is also a buffer
/// argument of the function reads through that argument's slot, bound once.
#[test]
fn the_vertex_fetch_projection_selects_fixed_function_or_pulled() {
    let Some(bc) = fixture() else { return };
    let p = prof("moltenvk-m3");
    let mut desc = VertexDescriptor::new();
    desc.layouts
        .insert(0, VertexBufferLayoutDescriptor::new(Some(16)));
    let mut constant = VertexBufferLayoutDescriptor::new(Some(0));
    constant.step_function = VertexStepFunction::Constant;
    desc.layouts.insert(2, constant);
    desc.attributes.insert(
        0,
        VertexAttributeDescriptor::new(VertexFormat::Float3, 0, 0),
    );
    desc.attributes
        .insert(1, VertexAttributeDescriptor::new(VertexFormat::Half2, 4, 2));
    // The function reads both as `float` (the one projection takes the declared classes).
    let floats = BTreeMap::from([(0, IoNumeric::Float), (1, IoNumeric::Float)]);
    let proj = vertex_fetch_projection(Some(&desc), &floats, &p).expect("projection");
    assert_eq!(
        proj.attributes[&0],
        VertexFetch::FixedFunction {
            narrow_float: false
        }
    );
    assert_eq!(
        proj.attributes[&1],
        VertexFetch::Pulled {
            format: VertexFormat::Half2,
            buffer_index: 2,
            offset: 4,
            step_function: VertexStepFunction::Constant,
            step_rate: 0,
        }
    );

    let mut missing = VertexDescriptor::new();
    missing
        .attributes
        .insert(0, VertexAttributeDescriptor::new(VertexFormat::Float, 0, 3));
    assert!(matches!(
        vertex_fetch_projection(Some(&missing), &floats, &p),
        Err(Error::InvalidDescriptor(_))
    ));

    let run = |proj: VertexFetchProjection| -> Result<Arc<FunctionArtifact>, Error> {
        let mut inputs = LoweringInputs::new();
        inputs.vertex_fetch = Some(proj);
        compile_bitcode(
            &bc,
            EntryRef::Name("stage_io_vertex"),
            &FunctionConstantValues::defaults(),
            &p,
            &inputs,
            &mut meter(),
            SPIRV_WORDS,
        )
    };
    let mut fixed = VertexFetchProjection::new();
    fixed.attributes.insert(
        0,
        VertexFetch::FixedFunction {
            narrow_float: false,
        },
    );
    let a = run(fixed).expect("fixed-function projection");
    validate(&a, &alone(&a, &p), &p);
    let a = run(proj).expect("pulled attribute");
    let pulled: Vec<u32> = a.interface.io.pulled.iter().map(|x| x.index).collect();
    assert_eq!(pulled, [1]);
    validate(&a, &alone(&a, &p), &p);

    // Buffer 1 is also the function's `scale` argument: one widened slot.
    let mut shared = VertexFetchProjection::new();
    let mut layout = VertexBufferLayoutDescriptor::new(Some(0));
    layout.step_function = VertexStepFunction::Constant;
    shared.attributes.insert(
        1,
        VertexFetch::pulled(
            &VertexAttributeDescriptor::new(VertexFormat::Half2, 0, 1),
            &layout,
        ),
    );
    let a = run(shared).expect("pulled attribute on an argument's buffer");
    validate(&a, &alone(&a, &p), &p);
    assert_eq!(a.interface.descriptors.len(), 1, "one binding for buffer 1");
    assert_eq!(a.interface.slots.len(), 1, "one slot for buffer 1");
}

/// The plan's size estimate bounds the emitted words of every stage entry.
#[test]
fn the_estimate_bounds_every_stage_entry() {
    let Some(bc) = fixture() else { return };
    let p = prof("moltenvk-m3");
    for entry in [
        "stage_io_vertex",
        "stage_io_vertex_position",
        "stage_io_vertex_clip",
        "stage_io_vertex_layer",
        "stage_io_fragment",
        "stage_io_fragment_point",
        "stage_io_fragment_dual_source",
    ] {
        let plan = plan_bitcode(
            &bc,
            EntryRef::Name(entry),
            &FunctionConstantValues::defaults(),
            &p,
            &LoweringInputs::new(),
            &mut meter(),
        )
        .expect("plan");
        let a = compile(&bc, entry, &p).expect("compile");
        let words = a.module.words.len() as u64;
        assert!(
            plan.estimate.words >= words,
            "{entry}: estimate {} < emitted {words}",
            plan.estimate.words
        );
    }
}

/// The variables `words` decorate `BuiltIn PointSize` (`OpDecorate <id> BuiltIn(11) PointSize(1)`)
/// that an `OpStore` (opcode 62) writes through.
fn stored_point_sizes(words: &[u32]) -> Vec<u32> {
    let mut decorated = Vec::new();
    let mut stored = Vec::new();
    let mut i = 5;
    while let Some(&w) = words.get(i) {
        let (len, op) = ((w >> 16) as usize, w & 0xFFFF);
        if len == 0 {
            break;
        }
        let operands = &words[i + 1..(i + len).min(words.len())];
        match (op, operands) {
            (71, [id, 11, 1]) => decorated.push(*id),
            (62, [pointer, ..]) => stored.push(*pointer),
            _ => {}
        }
        i += len;
    }
    decorated.retain(|id| stored.contains(id));
    decorated
}

/// The values `words` store to `PointSize` (`OpDecorate <id> BuiltIn(11) PointSize(1)`), through
/// the variable itself or an access chain into it (a mesh function's arrayed output), each as
/// `Some(true)` where it is Metal's largest point applied to a size (an `OpSelect`, opcode 169,
/// whose first choice is the `float` constant 511.0), `Some(false)` where it is another select
/// or a constant (opcode 43), and `None` where another instruction computes it.
pub(super) fn point_size_stores(words: &[u32]) -> Vec<Option<bool>> {
    let limit = 511.0f32.to_bits();
    let mut insts = Vec::new();
    let mut i = 5;
    while let Some(&w) = words.get(i) {
        let (len, op) = ((w >> 16) as usize, w & 0xFFFF);
        if len == 0 {
            break;
        }
        insts.push((op, &words[i + 1..(i + len).min(words.len())]));
        i += len;
    }
    let mut sizes: Vec<u32> = insts
        .iter()
        .filter_map(|(op, o)| match (op, o) {
            (71, [id, 11, 1]) => Some(*id),
            _ => None,
        })
        .collect();
    // Access chains (opcodes 65 and 66) into a size variable point to sizes too.
    for (op, o) in &insts {
        if let (65 | 66, [_, result, base, ..]) = (op, o) {
            if sizes.contains(base) {
                sizes.push(*result);
            }
        }
    }
    // A constant or a select: each defines its second operand.
    let defined = |id: u32| {
        insts
            .iter()
            .find(|(op, o)| matches!(op, 43 | 169) && o.get(1) == Some(&id))
    };
    let constant = |id: u32| match defined(id) {
        Some((43, [_, _, bits])) => Some(*bits),
        _ => None,
    };
    insts
        .iter()
        .filter_map(|(op, o)| match (op, o) {
            (62, [pointer, value, ..]) if sizes.contains(pointer) => Some(*value),
            _ => None,
        })
        .map(|value| {
            defined(value).map(|(op, o)| match (op, o) {
                (169, [_, _, _, chosen, _]) => constant(*chosen) == Some(limit),
                _ => false,
            })
        })
        .collect()
}

/// A vertex function's `[[point_size]]` is stored as Metal draws it, a size past Metal's largest
/// point (511 pixels, `stage_io::METAL_POINT_SIZE_LIMIT`) as 511, on every profile: Vulkan clamps a
/// point only to the device's `pointSizeRange`, up to 2047.94 on the RTX 5080 and 8191.875 on RADV.
/// The default 1.0 of a function without one is stored as it is.
#[test]
fn a_point_size_past_metals_largest_point_is_stored_as_it() {
    let Some(bc) = fixture() else { return };
    for stem in PROFILES {
        let p = prof(stem);
        let a = compile(&bc, "stage_io_vertex", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let words = validate(&a, &alone(&a, &p), &p);
        assert_eq!(
            point_size_stores(&words),
            vec![Some(true)],
            "{stem}: the one `[[point_size]]` store takes Metal's largest point"
        );
        let bare = compile(&bc, "stage_io_vertex_position", &p).expect("bare position");
        let words = validate(&bare, &alone(&bare, &p), &p);
        assert_eq!(
            point_size_stores(&words),
            vec![Some(false)],
            "{stem}: the default point size is stored as it is"
        );
    }
}

/// A vertex function writing `[[point_size]]` records `PrecisionNote::DevicePointSize` on a device
/// whose largest point is below Metal's 511 pixels (lavapipe's 256, ANV's 255.875, the 2023
/// baseline's 64), which draws a size between the two smaller than Metal; not on one that reaches
/// 511 (MoltenVK on the M3, the RTX 5080's 2047.94, RADV's 8191.88, SwiftShader's 1023), where the
/// stored clamp gives Metal's size. A function storing only the default 1.0 records nothing on
/// either.
#[test]
fn a_device_below_metals_largest_point_announces_its_point_size() {
    use crate::pipeline::artifact::PrecisionNote;
    let Some(bc) = fixture() else { return };
    let note = PrecisionNote::DevicePointSize;
    for (stem, below) in [
        ("lavapipe", true),
        ("intel-irisxe-rplp-anv", true),
        ("lunarg-desktop-baseline-2023", true),
        ("moltenvk-m3", false),
        ("nvidia-rtx5080-laptop", false),
        ("amd-780m-radv", false),
        ("swiftshader", false),
    ] {
        let p = prof(stem);
        let a = compile(&bc, "stage_io_vertex", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(a.precision.contains(&note), below, "{stem}");
        let bare = compile(&bc, "stage_io_vertex_position", &p)
            .unwrap_or_else(|e| panic!("{stem} bare: {e}"));
        assert!(
            !bare.precision.contains(&note),
            "{stem}: the default point size"
        );
    }
}

/// A vertex function without outputs (a rasterization-disabled `vertex void`) writes the default
/// point size too. A point-list pipeline's vertex stage must write `PointSize` on a device without
/// `maintenance5`, under rasterizer discard as well
/// (`VUID-VkGraphicsPipelineCreateInfo-topology-08773`); `swiftshader` and
/// `lunarg-desktop-baseline-2023` state no `VK_KHR_maintenance5`, and the module does not know the
/// topology. The owned `ob3_index_buffer` function returns `void` and stores to a buffer.
#[test]
fn a_vertex_function_without_outputs_writes_the_default_point_size() {
    let bc = match llvm_assemble(&engine_dir().join("ob3_index_buffer.ll")) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected ob3_index_buffer.ll: {why}"),
    };
    for stem in ["swiftshader", "lunarg-desktop-baseline-2023", "moltenvk-m3"] {
        let p = prof(stem);
        let a = compile(&bc, "ob3_index_buffer", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(a.interface.io.outputs.is_empty(), "{stem}: a void function");
        assert_eq!(
            a.interface.io.builtins,
            vec![
                StageBuiltin::VertexIndex,
                StageBuiltin::PointSize { default: true },
            ],
            "{stem}"
        );
        let words = validate(&a, &alone(&a, &p), &p);
        assert_eq!(
            stored_point_sizes(&words).len(),
            1,
            "{stem}: one `PointSize` variable, written"
        );
    }
}

/// An attribute a function-constant predicate removes is not declared (MSL 5.8.1.3), so Metal gives
/// its read no value: the read is zero, and the function declares only the attributes it has; with
/// the constant true the attribute is an ordinary input.
#[test]
fn a_gated_off_attribute_in_use_reads_zero() {
    use crate::library::FunctionConstantValue;
    use crate::metal::DataType;
    let bc = match llvm_assemble(&engine_dir().join("stage_io_gated.ll")) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected stage_io_gated.ll: {why}"),
    };
    let p = prof("moltenvk-m3");
    let run = |on: bool| {
        compile_bitcode(
            &bc,
            EntryRef::Name("stage_io_vertex_gated"),
            &FunctionConstantValues::defaults().with_index(
                0,
                FunctionConstantValue::new(DataType::Bool, [u8::from(on)]),
            ),
            &p,
            &LoweringInputs::new(),
            &mut meter(),
            SPIRV_WORDS,
        )
    };
    let attributes = |a: &FunctionArtifact| -> Vec<u32> {
        a.interface
            .io
            .inputs
            .iter()
            .map(|x| match x.role {
                IoRole::Attribute { index, .. } => index,
                ref r => panic!("vertex input role {r:?}"),
            })
            .collect()
    };
    let a = run(false).expect("gated-off attribute");
    assert_eq!(attributes(&a), [0]);
    validate(&a, &alone(&a, &p), &p);
    let a = run(true).expect("gated-on attribute");
    assert_eq!(attributes(&a), [0, 1]);
    validate(&a, &alone(&a, &p), &p);
}

/// Metal rejects `long`/`ulong` as a vertex attribute, a varying, a stage builtin, a color output,
/// and a color input ("type 'long' is not valid for attribute"), so a 64-bit integer there is
/// invalid input on every committed profile, never `Int64Arithmetic` (which `shaderInt64` would not
/// fix). Each case edits one fixture entry's value to `i64`.
#[test]
fn sixty_four_bit_interface_values_are_not_metal() {
    use crate::air::error::AbiError;
    use crate::error::InputErrorKind;
    use crate::spirv::testing::committed_profiles;
    let edit = |stem: &str, edits: &[(&str, &str)]| -> Option<Vec<u8>> {
        let mut text = std::fs::read_to_string(engine_dir().join(format!("{stem}.ll")))
            .expect("read the fixture");
        for (from, to) in edits {
            assert!(text.contains(from), "{stem}: {from}");
            text = text.replacen(from, to, 1);
        }
        let scratch = Scratch::new(&format!("m2v-io64-{stem}"));
        let dir = scratch.path();
        let path = dir.join("edited.ll");
        std::fs::write(&path, text).expect("write");
        let bc = llvm_assemble(&path);
        match bc {
            Ok(b) => Some(b),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                None
            }
            Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {stem}: {why}"),
        }
    };
    let ret = "<{ <4 x float>, <4 x i32>, float, i32 }>";
    let vret = "<{ <4 x float>, <2 x half>, <4 x float>, i32, float }>";
    let vret64 = "<{ <4 x float>, <2 x half>, <4 x float>, i64, float }>";
    let mut varying = vec![
        ("i32 %23, 3", "i64 %w23, 3"),
        (
            "  %24 = insertvalue",
            "  %w23 = zext i32 %23 to i64\n  %24 = insertvalue",
        ),
    ];
    // The return type in the definition, the five `insertvalue`s, and the `ret`.
    varying.extend(std::iter::repeat_n((vret, vret64), 7));
    let cases = [
        // The `ushort2` vertex attribute at location 4 as a `<2 x i64>`.
        (
            "stage_io",
            "stage_io_vertex",
            vec![
                ("<2 x i16> %2", "<2 x i64> %2"),
                (
                    "%16 = extractelement <2 x i16> %2, i64 0\n  %17 = zext i16 %16 to i32",
                    "%16 = extractelement <2 x i64> %2, i64 0\n  %17 = trunc i64 %16 to i32",
                ),
                (
                    "%18 = extractelement <2 x i16> %2, i64 1\n  %19 = zext i16 %18 to i32",
                    "%18 = extractelement <2 x i64> %2, i64 1\n  %19 = trunc i64 %18 to i32",
                ),
            ],
        ),
        // The `vertex_id` input builtin as an `i64`.
        (
            "stage_io",
            "stage_io_vertex",
            vec![
                ("i32 noundef %5, i32 noundef %6", "i64 noundef %5, i32 noundef %6"),
                (
                    "%15 = mul i32 %5, 7",
                    "%t5 = trunc i64 %5 to i32\n  %15 = mul i32 %t5, 7",
                ),
            ],
        ),
        // The `uint id` varying output as an `i64`.
        ("stage_io", "stage_io_vertex", varying),
        // The flat `uint id` fragment input as an `i64`.
        (
            "stage_io",
            "stage_io_fragment",
            vec![
                ("i32 %3, i1 noundef %4", "i64 %3, i1 noundef %4"),
                (
                    "%15 = insertelement <4 x i32> <i32 poison, i32 poison, i32 poison, i32 9>, i32 %3, i64 0",
                    "%t3 = trunc i64 %3 to i32\n  %15 = insertelement <4 x i32> <i32 poison, i32 poison, i32 poison, i32 9>, i32 %t3, i64 0",
                ),
            ],
        ),
        // The `uint4` color output 2 as a `<4 x i64>`.
        (
            "stage_io",
            "stage_io_fragment",
            vec![
                (
                    ret,
                    "<{ <4 x float>, <4 x i64>, float, i32 }>",
                ),
                (
                    "%22 = insertvalue <{ <4 x float>, <4 x i32>, float, i32 }> undef",
                    "%22 = insertvalue <{ <4 x float>, <4 x i64>, float, i32 }> undef",
                ),
                (
                    "%23 = insertvalue <{ <4 x float>, <4 x i32>, float, i32 }> %22, <4 x i32> %17, 1",
                    "%w17 = zext <4 x i32> %17 to <4 x i64>\n  %23 = insertvalue <{ <4 x float>, <4 x i64>, float, i32 }> %22, <4 x i64> %w17, 1",
                ),
                (
                    "%24 = insertvalue <{ <4 x float>, <4 x i32>, float, i32 }> %23",
                    "%24 = insertvalue <{ <4 x float>, <4 x i64>, float, i32 }> %23",
                ),
                (
                    "%25 = insertvalue <{ <4 x float>, <4 x i32>, float, i32 }> %24",
                    "%25 = insertvalue <{ <4 x float>, <4 x i64>, float, i32 }> %24",
                ),
                (
                    "ret <{ <4 x float>, <4 x i32>, float, i32 }> %25",
                    "ret <{ <4 x float>, <4 x i64>, float, i32 }> %25",
                ),
            ],
        ),
        // The `float` color input of `fetch_scalar` as an `i64`.
        (
            "fragment_fetch",
            "fetch_scalar",
            vec![(
                "define float @fetch_scalar(float noundef %0) local_unnamed_addr #2 {\n  %2 = fmul fast float %0, 5.000000e-01",
                "define float @fetch_scalar(i64 noundef %0) local_unnamed_addr #2 {\n  %2 = uitofp i64 %0 to float",
            )],
        ),
    ];
    for (stem, entry, edits) in cases {
        let Some(bc) = edit(stem, &edits) else {
            return;
        };
        for (profile, p) in committed_profiles().expect("profiles") {
            match compile(&bc, entry, &p) {
                Err(Error::InvalidInput(e)) => assert_eq!(
                    e.kind,
                    InputErrorKind::Abi(AbiError::RoleTypeMismatch),
                    "{profile} {stem} {entry}"
                ),
                other => panic!("{profile} {stem} {entry}: {other:?}"),
            }
        }
    }
}

/// Apple's Metal compiler rejects a `bfloat` fragment return type ("invalid return type
/// 'bfloat4' (vector of 4 'bfloat' values) for fragment function"), so a fragment
/// function whose color output is `bfloat` is invalid input on every committed profile, decided by
/// the output's AIR type: `stage_io_fragment_point`'s `half4` color edited to `bfloat4`, its type
/// name too. The unedited function compiles.
#[test]
fn a_bfloat_color_output_is_not_metal() {
    use crate::air::error::AbiError;
    use crate::error::InputErrorKind;
    use crate::spirv::testing::committed_profiles;
    let stem = "stage_io";
    let entry = "stage_io_fragment_point";
    let edits: [(&str, &str); 3] = [
        (
            "define <4 x half> @stage_io_fragment_point(",
            "define <4 x bfloat> @stage_io_fragment_point(",
        ),
        (
            "  ret <4 x half> %7\n}",
            "  %w7 = fpext <4 x half> %7 to <4 x float>\n  %b7 = fptrunc <4 x float> %w7 to <4 x bfloat>\n  ret <4 x bfloat> %b7\n}",
        ),
        (
            "!\"air.arg_type_name\", !\"half4\"}",
            "!\"air.arg_type_name\", !\"bfloat4\"}",
        ),
    ];
    let assemble = |edits: &[(&str, &str)]| -> Option<Vec<u8>> {
        let mut text = std::fs::read_to_string(engine_dir().join(format!("{stem}.ll")))
            .expect("read the fixture");
        for (from, to) in edits {
            assert!(text.contains(from), "{stem}: {from}");
            text = text.replacen(from, to, 1);
        }
        let scratch = Scratch::new(&format!("m2v-bfout-{stem}"));
        let dir = scratch.path();
        let path = dir.join("edited.ll");
        std::fs::write(&path, text).expect("write");
        let bc = llvm_assemble(&path);
        match bc {
            Ok(b) => Some(b),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                None
            }
            Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {stem}: {why}"),
        }
    };
    let (Some(plain), Some(bfloat)) = (assemble(&[]), assemble(&edits)) else {
        return;
    };
    for (profile, p) in committed_profiles().expect("profiles") {
        match compile(&bfloat, entry, &p) {
            Err(Error::InvalidInput(e)) => assert_eq!(
                e.kind,
                InputErrorKind::Abi(AbiError::RoleTypeMismatch),
                "{profile}"
            ),
            other => panic!("{profile}: {other:?}"),
        }
        assert!(
            !matches!(compile(&plain, entry, &p), Err(Error::InvalidInput(_))),
            "{profile}: the half color output"
        );
    }
}

/// A fragment function that returns a `half` color output to an attachment the pipeline has
/// records a note or lists the attachment for the pipeline's, decided by the output's AIR type
/// (`stage_io_fragment_point`'s `half4` to color 0); neither where the pipeline has no attachment
/// there, nor for `float` and `uint` outputs (`stage_io_fragment`'s). Which is the module's math
/// mode's: the fixture states fast math (`air.compile.fast_math_enable`), records no note and lists
/// the attachment (`StageIo::half_fast_colors`), whose write mask decides the pipeline's note
/// (`a_half_output_with_fast_math_is_noted_where_its_attachment_stores`); the same module stating
/// no math mode records `PrecisionNote::HalfOutputPrecision` alone. Stating `fast_math_disable` it
/// records no note and lists the attachment (`StageIo::half_blend_colors`), whose blend decides the
/// pipeline's note (`a_half_output_without_fast_math_is_noted_where_its_attachment_blends`).
#[test]
fn a_half_color_output_records_its_precision() {
    use crate::lower::select::LowerInputs;
    use crate::pipeline::artifact::PrecisionNote;
    use crate::pipeline::testing::compile_with;
    let Some(bc) = fixture() else { return };
    let fast = PrecisionNote::HalfOutputFastMathBlend;
    let safe = PrecisionNote::HalfOutputPrecision;
    let Some(disabled) = math_mode_edited(Some("air.compile.fast_math_disable")) else {
        return;
    };
    let Some(unstated) = math_mode_edited(None) else {
        return;
    };
    // Profiles with sample-rate shading, which the function's sample input needs.
    for stem in ["moltenvk-m3", "moltenvk-m4", "amd-rdna3-radv", "lavapipe"] {
        let p = prof(stem);
        let half = compile(&bc, "stage_io_fragment_point", &p).expect("half");
        assert!(!half.precision.contains(&fast), "{stem}");
        assert!(!half.precision.contains(&safe), "{stem}");
        assert!(half.interface.io.half_blend_colors.is_empty(), "{stem}");
        assert_eq!(half.interface.io.half_fast_colors, [0], "{stem}");
        let half = compile(&unstated, "stage_io_fragment_point", &p).expect("no math mode");
        assert!(half.precision.contains(&safe), "{stem}: no math mode");
        assert!(!half.precision.contains(&fast), "{stem}: no math mode");
        assert!(half.interface.io.half_blend_colors.is_empty(), "{stem}");
        assert!(half.interface.io.half_fast_colors.is_empty(), "{stem}");
        let half = compile(&disabled, "stage_io_fragment_point", &p).expect("without fast math");
        assert!(!half.precision.contains(&safe), "{stem}: without fast math");
        assert!(!half.precision.contains(&fast), "{stem}: without fast math");
        assert_eq!(half.interface.io.half_blend_colors, [0], "{stem}");
        assert!(half.interface.io.half_fast_colors.is_empty(), "{stem}");
        let mut inputs = LowerInputs::new();
        inputs.unattached_colors.insert(0);
        let unattached =
            compile_with(&bc, "stage_io_fragment_point", &p, &inputs).expect("unattached");
        assert!(!unattached.precision.contains(&fast), "{stem}");
        assert!(!unattached.precision.contains(&safe), "{stem}");
        assert!(
            unattached.interface.io.half_fast_colors.is_empty(),
            "{stem}"
        );
        let float = compile(&bc, "stage_io_fragment", &p).expect("float");
        assert!(!float.precision.contains(&fast), "{stem}");
        assert!(!float.precision.contains(&safe), "{stem}");
    }
}

/// A `half` color output of a module without fast math (the `stage_io` fixture stating
/// `fast_math_disable`) lists its attachment (`StageIo::half_blend_colors`), and the pipeline's
/// stage records `PrecisionNote::HalfOutputPrecision` only where that attachment blends
/// (`color_writes::pipeline_writes`, `blends_output`): not unblended, not under factors one and
/// zero added (which store the output as no blend does), not where no channel is written; under
/// "over", one and one, and `Min` it does (Metal rounds each step of a blend to `half`, where a
/// device blends in its own arithmetic).
#[test]
fn a_half_output_without_fast_math_is_noted_where_its_attachment_blends() {
    use crate::metal::{PixelFormat, TextureWriteRoundingMode};
    use crate::pipeline::artifact::PrecisionNote;
    use crate::pipeline::color_writes::{pipeline_writes, rule_words};
    use crate::pipeline::descriptor::ColorAttachmentDescriptor;
    use crate::pipeline::graphics::{BlendState, ColorAttachmentState};
    use crate::vk::{BlendFactor as F, BlendOp as O, ColorComponentFlags, Format};
    let Some(disabled) = math_mode_edited(Some("air.compile.fast_math_disable")) else {
        return;
    };
    let note = PrecisionNote::HalfOutputPrecision;
    let blend = |s: F, d: F, op: O| {
        Some(BlendState {
            src_color: s,
            dst_color: d,
            color_op: op,
            src_alpha: s,
            dst_alpha: d,
            alpha_op: op,
        })
    };
    let all = ColorComponentFlags::known_bits();
    let cases = [
        ("unblended", all, None, false),
        ("one and zero", all, blend(F::ONE, F::ZERO, O::ADD), true),
        (
            "no channel written",
            ColorComponentFlags::empty(),
            blend(F::ONE, F::ONE_MINUS_SRC_ALPHA, O::ADD),
            false,
        ),
        (
            "over",
            all,
            blend(F::ONE, F::ONE_MINUS_SRC_ALPHA, O::ADD),
            true,
        ),
        ("one and one", all, blend(F::ONE, F::ONE, O::ADD), true),
        ("min", all, blend(F::ONE, F::ONE, O::MIN), true),
    ];
    for stem in ["moltenvk-m3", "amd-rdna3-radv", "lavapipe"] {
        let p = prof(stem);
        let f = compile(&disabled, "stage_io_fragment_point", &p).expect("half");
        assert_eq!(f.interface.io.half_blend_colors, [0], "{stem}");
        assert!(!f.precision.contains(&note), "{stem}");
        let descriptors = [Some(ColorAttachmentDescriptor::new(
            PixelFormat::RGBA16Float,
        ))];
        let rounding = (false, TextureWriteRoundingMode::TowardZero);
        let rules = rule_words(&f.interface.io, &descriptors, rounding, &p).expect("rules");
        for (what, mask, b, want) in cases {
            let colors = [Some(ColorAttachmentState::new(
                Format::R16G16B16A16_SFLOAT,
                mask,
                b,
            ))];
            let (notes, _) = pipeline_writes(&f, &rules, &colors, false).expect(what);
            assert_eq!(notes.contains(&note), want, "{stem}: {what}: {notes:?}");
        }
        // A pipeline without the attachment's state (none listed) records none.
        let (notes, _) = pipeline_writes(&f, &rules, &[], false).expect("no attachments");
        assert!(!notes.contains(&note), "{stem}: no attachments");
    }
}

/// A `half` color output of a module with fast math (the `stage_io` fixture) lists its attachment
/// (`StageIo::half_fast_colors`), and the pipeline's stage records
/// `PrecisionNote::HalfOutputFastMathBlend` only where that attachment stores some channel
/// (`color_writes::pipeline_writes`, `stores_output`): unblended or blended alike, but not under a
/// write mask of none, blended or not, nor one naming none of its format's channels, nor where
/// the pipeline lacks the attachment's state. An
/// attachment that stores nothing holds nothing the output could differ by (the macOS 26
/// window-server pass's colors of mask none beside its fetched one).
#[test]
fn a_half_output_with_fast_math_is_noted_where_its_attachment_stores() {
    use crate::metal::{PixelFormat, TextureWriteRoundingMode};
    use crate::pipeline::artifact::PrecisionNote;
    use crate::pipeline::color_writes::{pipeline_writes, rule_words};
    use crate::pipeline::descriptor::ColorAttachmentDescriptor;
    use crate::pipeline::graphics::{BlendState, ColorAttachmentState};
    use crate::vk::{BlendFactor as F, BlendOp as O, ColorComponentFlags, Format};
    let Some(bc) = fixture() else { return };
    let note = PrecisionNote::HalfOutputFastMathBlend;
    let over = Some(BlendState {
        src_color: F::ONE,
        dst_color: F::ONE_MINUS_SRC_ALPHA,
        color_op: O::ADD,
        src_alpha: F::ONE,
        dst_alpha: F::ONE_MINUS_SRC_ALPHA,
        alpha_op: O::ADD,
    });
    let all = ColorComponentFlags::known_bits();
    let none = ColorComponentFlags::empty();
    let cases = [
        ("unblended", all, None, true),
        ("over", all, over, true),
        ("red alone", ColorComponentFlags::R, over, true),
        ("no channel written", none, None, false),
        ("no channel written, blended", none, over, false),
    ];
    for stem in ["moltenvk-m3", "amd-rdna3-radv", "lavapipe"] {
        let p = prof(stem);
        let f = compile(&bc, "stage_io_fragment_point", &p).expect("half");
        assert_eq!(f.interface.io.half_fast_colors, [0], "{stem}");
        assert!(!f.precision.contains(&note), "{stem}");
        let descriptors = [Some(ColorAttachmentDescriptor::new(
            PixelFormat::RGBA16Float,
        ))];
        let rounding = (false, TextureWriteRoundingMode::TowardZero);
        let rules = rule_words(&f.interface.io, &descriptors, rounding, &p).expect("rules");
        for (what, mask, b, want) in cases {
            let colors = [Some(ColorAttachmentState::new(
                Format::R16G16B16A16_SFLOAT,
                mask,
                b,
            ))];
            let (notes, _) = pipeline_writes(&f, &rules, &colors, false).expect(what);
            assert_eq!(notes.contains(&note), want, "{stem}: {what}: {notes:?}");
        }
        // A one-channel format under a mask of blue alone stores nothing; under red it stores.
        for (mask, want) in [
            (ColorComponentFlags::B, false),
            (ColorComponentFlags::R, true),
        ] {
            let colors = [Some(ColorAttachmentState::new(
                Format::R16_SFLOAT,
                mask,
                over,
            ))];
            let (notes, _) = pipeline_writes(&f, &rules, &colors, false).expect("R16");
            assert_eq!(
                notes.contains(&note),
                want,
                "{stem}: R16 {mask:?}: {notes:?}"
            );
        }
        // A pipeline without the attachment's state (none listed) records none.
        let (notes, _) = pipeline_writes(&f, &rules, &[], false).expect("no attachments");
        assert!(!notes.contains(&note), "{stem}: no attachments");
    }
}

/// Under `HalfBlendPolicy::Exact`, on every committed profile that orders framebuffer fetch through
/// the pixel interlock alone, a `half4` output of a module without fast math into a blended
/// `RGBA16Float` attachment the pipeline names for that blend alone is written in the storage form
/// with Metal's 16-bit float blend computed by the function
/// (`RuntimeFact::InterlockedFramebufferFetch` with the blend, rounded), passes L3, links into
/// the pipeline's storage attachments, lists no attachment for the note and records none; a blend
/// of the blend color reads `PushField::HalfBlendColor`. The same function with fast math, or a
/// pipeline under alpha to coverage, leaves the attachment to the device's blend, as the default
/// policy does, whose fetch inputs name none.
#[test]
fn an_exact_half_blend_is_computed_in_the_storage_form() {
    use crate::device::HalfBlendPolicy;
    use crate::metal::{BlendFactor as F, PixelFormat};
    use crate::pipeline::artifact::PrecisionNote;
    use crate::pipeline::descriptor::ColorAttachmentDescriptor;
    use crate::pipeline::facts::RuntimeFact;
    use crate::pipeline::link::fetch::storage_attachments;
    use crate::pipeline::link::RenderLinkInputs;
    use crate::pipeline::lower_inputs::{BlendEquation, FramebufferFetchInputs};
    use crate::pipeline::sym::PushField;
    use crate::pipeline::testing::{compile_with, fragment_inputs, link_and_validate};
    let Some(disabled) = math_mode_edited(Some("air.compile.fast_math_disable")) else {
        return;
    };
    let Some(enabled) = fixture() else {
        return;
    };
    let note = PrecisionNote::HalfOutputPrecision;
    let exact = DevicePolicy {
        half_blend: HalfBlendPolicy::Exact,
        ..DevicePolicy::default()
    };
    let attachment = |s: F, d: F| {
        let mut c = ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float);
        c.blending_enabled = true;
        c.source_rgb_blend_factor = s;
        c.destination_rgb_blend_factor = d;
        c.source_alpha_blend_factor = s;
        c.destination_alpha_blend_factor = d;
        c
    };
    // The function's lowering inputs in a pipeline of `c` at color 0, the attachment named for
    // the blend alone, as `pipeline::compile::fetch_inputs` names it.
    let inputs = |r: &RenderLinkInputs, p: &crate::device::DeviceProfile| {
        let mut inputs = fragment_inputs(r, p);
        inputs.framebuffer_fetch = Some(FramebufferFetchInputs {
            half_blends: [0].into(),
            ..FramebufferFetchInputs::of([0], &r.color_attachments, false, r.alpha_to_coverage, 1)
        });
        inputs
    };
    let mut checked = 0;
    let profiles = crate::spirv::testing::committed_profiles_with(exact).expect("profiles");
    for (stem, p) in profiles {
        if !crate::lower::fragment::blends_half_in_shader(&p) {
            continue;
        }
        for (s, d, reads_color) in [
            (F::One, F::OneMinusSourceAlpha, false),
            (F::BlendColor, F::OneMinusBlendColor, true),
        ] {
            let what = format!("{stem} {s:?} {d:?}");
            let c = attachment(s, d);
            let eq = BlendEquation::of(&c);
            assert!(
                crate::lower::fragment::half_blend_attachment(
                    c.pixel_format,
                    c.blending_enabled,
                    c.write_mask,
                    eq
                ),
                "{what}"
            );
            let mut r = RenderLinkInputs::new();
            r.color_attachments[0] = Some(c);
            let f = compile_with(&disabled, "stage_io_fragment_point", &p, &inputs(&r, &p))
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            let fetched = f.facts.iter().find_map(|fact| match fact {
                RuntimeFact::InterlockedFramebufferFetch {
                    color: 0,
                    writes,
                    blend,
                    rounded,
                    ..
                } => Some((*writes, *blend, *rounded)),
                _ => None,
            });
            assert_eq!(fetched, Some((true, eq, true)), "{what}: {:?}", f.facts);
            assert!(f.interface.io.half_blend_colors.is_empty(), "{what}");
            assert!(!f.precision.contains(&note), "{what}");
            let pushes =
                |field: PushField| f.interface.push.fields.iter().any(|x| x.field == field);
            assert_eq!(pushes(PushField::HalfBlendColor), reads_color, "{what}");
            assert!(!pushes(PushField::BlendColor), "{what}");
            link_and_validate(&f, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let storage = storage_attachments(&r, Some(&f), &p)
                .unwrap_or_else(|e| panic!("{what}: link: {e}"));
            assert_eq!(
                storage.iter().map(|u| u.color_index).collect::<Vec<_>>(),
                [0],
                "{what}"
            );
            // With fast math the device blends, and the function lists the attachment for the
            // pipeline's note.
            let fast = compile_with(&enabled, "stage_io_fragment_point", &p, &inputs(&r, &p))
                .unwrap_or_else(|e| panic!("{what}: fast: {e}"));
            assert!(fast
                .facts
                .iter()
                .all(|fact| !matches!(fact, RuntimeFact::InterlockedFramebufferFetch { .. })));
            assert!(
                !fast
                    .precision
                    .contains(&PrecisionNote::HalfOutputFastMathBlend),
                "{what}: fast"
            );
            assert_eq!(fast.interface.io.half_fast_colors, [0], "{what}: fast");
            // Under alpha to coverage the storage form is not exact: the device blends.
            r.alpha_to_coverage = true;
            let covered = compile_with(&disabled, "stage_io_fragment_point", &p, &inputs(&r, &p))
                .unwrap_or_else(|e| panic!("{what}: coverage: {e}"));
            assert!(covered
                .facts
                .iter()
                .all(|fact| !matches!(fact, RuntimeFact::InterlockedFramebufferFetch { .. })));
            assert_eq!(covered.interface.io.half_blend_colors, [0], "{what}");
            checked += 1;
        }
    }
    assert!(checked >= 2, "no committed profile took the form");
}

/// The `stage_io` fixture with its `air.compile.fast_math_enable` option replaced by `mode`, or
/// dropped from the module's compile options where `None`, assembled; `None` (skipped) without
/// `llvm-as`.
fn math_mode_edited(mode: Option<&str>) -> Option<Vec<u8>> {
    let option = "!70 = !{!\"air.compile.fast_math_enable\"}";
    let list = "!air.compile_options = !{!69, !70, !71}";
    let text = std::fs::read_to_string(engine_dir().join("stage_io.ll")).expect("read the fixture");
    assert!(
        text.contains(option) && text.contains(list),
        "the fixture's math mode"
    );
    let text = match mode {
        Some(mode) => text.replacen(option, &format!("!70 = !{{!\"{mode}\"}}"), 1),
        None => text.replacen(list, "!air.compile_options = !{!69, !71}", 1),
    };
    let scratch = Scratch::new("m2v-hmath");
    let dir = scratch.path();
    let path = dir.join("edited.ll");
    std::fs::write(&path, text).expect("write");
    let bc = llvm_assemble(&path);
    match bc {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the edit: {why}"),
    }
}

/// The channels a pipeline's 32-bit float attachments give a `half` output at `float` precision
/// (`color_writes::half_output_lanes`): every stored channel unblended, or blended one and one
/// under `Add`; the channels the format lacks or the mask drops none; an attachment with a
/// channel blended otherwise, or of another format, not listed.
#[test]
fn float32_attachments_list_their_half_output_lanes() {
    use crate::metal::{BlendFactor, BlendOperation, ColorWriteMask, PixelFormat};
    use crate::pipeline::color_writes::half_output_lanes;
    use crate::pipeline::descriptor::ColorAttachmentDescriptor as C;
    use crate::pipeline::lower_inputs::HalfOutputLane as L;
    let added = |pf| {
        let mut c = C::new(pf);
        c.blending_enabled = true;
        c.destination_rgb_blend_factor = BlendFactor::One;
        c.destination_alpha_blend_factor = BlendFactor::One;
        c
    };
    let mut masked = C::new(PixelFormat::RGBA32Float);
    masked.write_mask = ColorWriteMask::RED;
    let mut over = added(PixelFormat::RGBA32Float);
    over.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
    let mut subtracted = added(PixelFormat::R32Float);
    subtracted.rgb_blend_operation = BlendOperation::Subtract;
    let colors = [
        Some(C::new(PixelFormat::R32Float)),
        Some(added(PixelFormat::RG32Float)),
        Some(masked),
        Some(over),
        Some(subtracted),
        Some(C::new(PixelFormat::RGBA16Float)),
        None,
        Some(added(PixelFormat::RGBA32Float)),
    ];
    let lanes = half_output_lanes(&colors);
    let f = Some(L::Float32);
    let a = Some(L::Float32Added);
    assert_eq!(
        lanes,
        BTreeMap::from([
            (0, [f, None, None, None]),
            (1, [a, a, None, None]),
            (2, [f, None, None, None]),
            (7, [a, a, a, a]),
        ])
    );
}

/// The `stage_io` fixture's fragment entry that returns a `half4` to color 0.
const HALF_ENTRY: &str = "stage_io_fragment_point";

/// A fast `half` product as that entry's last operation (its return, [`half_returned`]).
const HALF_PRODUCT: &str = "  %p = fmul fast <4 x half> %7, %7\n  ret <4 x half> %p\n}";

/// The `stage_io` fixture with [`HALF_ENTRY`]'s return replaced by `to`, assembled; `None`
/// (skipped) without `llvm-as`.
fn half_returned(to: &str) -> Option<Vec<u8>> {
    let last = "  ret <4 x half> %7\n}";
    let text = std::fs::read_to_string(engine_dir().join("stage_io.ll")).expect("read the fixture");
    assert!(text.contains(last), "the fixture's return");
    let text = text.replacen(last, to, 1);
    let scratch = Scratch::new("m2v-hof32");
    let dir = scratch.path();
    let path = dir.join("edited.ll");
    std::fs::write(&path, text).expect("write");
    let bc = llvm_assemble(&path);
    match bc {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the edit: {why}"),
    }
}

/// A `half` output's last operation Metal computes at `float` precision into a 32-bit float
/// attachment is selected in `float` (`ArithPlan::unrounded`), and the output lists no attachment
/// for `PrecisionNote::HalfOutputFastMathBlend` (`StageIo::half_fast_colors`; the fixture's module
/// states fast math) where every stored channel takes such a lane: a product into channels
/// unblended, blended one and one, or of `RG32Float`; a sum unblended. A sum into a blended
/// channel, an operation without every fast-math flag, and one with another use keep the `half`
/// and list the attachment; so does a pipeline without the map. No function records the note
/// itself. Each module validates.
#[test]
fn a_half_output_into_a_float32_attachment_keeps_metals_float() {
    use crate::pipeline::artifact::PrecisionNote;
    use crate::pipeline::lower_inputs::HalfOutputLane as L;
    use crate::pipeline::testing::compile_with;
    let entry = HALF_ENTRY;
    let edited = half_returned;
    let product = HALF_PRODUCT;
    let sum = "  %p = fadd fast <4 x half> %7, %7\n  ret <4 x half> %p\n}";
    let plain = "  %p = fmul <4 x half> %7, %7\n  ret <4 x half> %p\n}";
    let twice =
        "  %p = fmul fast <4 x half> %7, %7\n  %s = extractelement <4 x half> %p, i64 0\n  \
                 %t = insertelement <4 x half> %p, half %s, i64 3\n  ret <4 x half> %t\n}";
    let f = Some(L::Float32);
    let a = Some(L::Float32Added);
    let cases: [(&str, &str, [Option<L>; 4], bool); 7] = [
        ("a product", product, [f; 4], true),
        ("a product added", product, [a; 4], true),
        (
            "a product into RG32Float",
            product,
            [f, f, None, None],
            true,
        ),
        ("a sum", sum, [f; 4], true),
        ("a sum added", sum, [a; 4], false),
        ("a product without fast math", plain, [f; 4], false),
        ("a product used twice", twice, [f; 4], false),
    ];
    // The fixture's module states fast math.
    let note = PrecisionNote::HalfOutputFastMathBlend;
    let consts = FunctionConstantValues::defaults();
    for (what, to, lanes, marked) in cases {
        let Some(bc) = edited(to) else { return };
        for stem in ["moltenvk-m4", "amd-rdna3-radv", "lavapipe"] {
            let p = prof(stem);
            let mut inputs = crate::lower::select::LowerInputs::new();
            inputs.half_output_lanes.insert(0, lanes);
            let plan = plan_bitcode(
                &bc,
                EntryRef::Name(entry),
                &consts,
                &p,
                &inputs,
                &mut meter(),
            )
            .unwrap_or_else(|e| panic!("{what} on {stem}: {e}"));
            assert_eq!(
                plan.families.arith.unrounded.len(),
                usize::from(marked),
                "{what} on {stem}"
            );
            let art = compile_with(&bc, entry, &p, &inputs)
                .unwrap_or_else(|e| panic!("{what} on {stem}: {e}"));
            assert!(!art.precision.contains(&note), "{what} on {stem}");
            let listed = art.interface.io.half_fast_colors == [0];
            assert_eq!(listed, !marked, "{what} on {stem}");
            assert_eq!(
                art.interface.io.half_output_lanes,
                BTreeMap::from([(0, lanes)]),
                "{what} on {stem}"
            );
            let t = alone(&art, &p);
            assert!(!validate(&art, &t, &p).is_empty(), "{what} on {stem}");
            let none = compile(&bc, entry, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            assert!(!none.precision.contains(&note), "{what} on {stem}: no map");
            let listed = none.interface.io.half_fast_colors == [0];
            assert!(listed, "{what} on {stem}: no map");
        }
    }
}

/// A missing half-output map loses Metal's `float`, never the pipeline: a `half` product compiled
/// without the map records none, lists its attachment for the note (`StageIo::half_fast_colors`),
/// and links into a pipeline whose `RGBA32Float`
/// attachment lists it, unblended or added (the added one compiled under its color writes, whose
/// channels are the device's where the profile does not store Metal's); one compiled under the
/// unblended attachment's entry links into that pipeline unnoted and is refused against the added
/// one; and a `float` output compiled under the map records none.
#[test]
fn a_missing_half_output_map_links() {
    use crate::device::requirements::VkProperty;
    use crate::error::RuntimeInput;
    use crate::metal::{BlendFactor, PixelFormat, TextureWriteRoundingMode};
    use crate::pipeline::artifact::PrecisionNote;
    use crate::pipeline::color_writes::{check, color_writes, half_output_lanes};
    use crate::pipeline::descriptor::ColorAttachmentDescriptor as C;
    use crate::pipeline::testing::compile_with;
    let Some(bc) = half_returned(HALF_PRODUCT) else {
        return;
    };
    let Some(io) = fixture() else { return };
    let plain = [Some(C::new(PixelFormat::RGBA32Float))];
    let mut added = C::new(PixelFormat::RGBA32Float);
    added.blending_enabled = true;
    added.destination_rgb_blend_factor = BlendFactor::One;
    added.destination_alpha_blend_factor = BlendFactor::One;
    let added = [Some(added)];
    // The fixture's module states fast math.
    let note = PrecisionNote::HalfOutputFastMathBlend;
    let rounding = (false, TextureWriteRoundingMode::default());
    let mut inputs = LoweringInputs::new();
    inputs.half_output_lanes = half_output_lanes(&plain);
    for stem in ["moltenvk-m4", "amd-rdna3-radv", "lavapipe"] {
        let p = prof(stem);
        let none = compile(&bc, HALF_ENTRY, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(none.interface.io.half_output_lanes.is_empty(), "{stem}");
        assert!(!none.precision.contains(&note), "{stem}");
        assert_eq!(none.interface.io.half_fast_colors, [0], "{stem}");
        check(Some(&none), &plain, rounding, &p).unwrap_or_else(|e| panic!("{stem}: no map: {e}"));
        let mut added_writes = LoweringInputs::new();
        added_writes.color_writes = color_writes([0], &added, rounding, &p);
        let metal = p
            .fields()
            .enabled
            .properties
            .contains(&VkProperty::MetalColorWrites);
        assert_eq!(added_writes.color_writes.is_empty(), metal, "{stem}");
        if !metal {
            assert!(
                matches!(
                    check(Some(&none), &added, rounding, &p),
                    Err(Error::NeedsInput(RuntimeInput::ColorWrites))
                ),
                "{stem}: the added blend's color writes"
            );
        }
        let none = compile_with(&bc, HALF_ENTRY, &p, &added_writes)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(none.interface.io.half_output_lanes.is_empty(), "{stem}");
        assert!(!none.precision.contains(&note), "{stem}");
        assert_eq!(none.interface.io.half_fast_colors, [0], "{stem}");
        check(Some(&none), &added, rounding, &p).unwrap_or_else(|e| panic!("{stem}: no map: {e}"));
        let with =
            compile_with(&bc, HALF_ENTRY, &p, &inputs).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(!with.precision.contains(&note), "{stem}");
        assert!(with.interface.io.half_fast_colors.is_empty(), "{stem}");
        check(Some(&with), &plain, rounding, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(
            matches!(
                check(Some(&with), &added, rounding, &p),
                Err(Error::NeedsInput(RuntimeInput::ColorWrites))
            ),
            "{stem}: another blend"
        );
        let mut float_inputs = inputs.clone();
        float_inputs.color_writes = added_writes.color_writes.clone();
        let float = compile_with(&io, "stage_io_fragment_primitive", &p, &float_inputs)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(float.interface.io.half_output_lanes.is_empty(), "{stem}");
        check(Some(&float), &added, rounding, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
    }
}
