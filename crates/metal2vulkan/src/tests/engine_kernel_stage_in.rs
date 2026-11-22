//! Kernel `[[stage_in]]` through the pulled fetch. No device is opened.
//!
//! Each kernel of the owned fixture `kernel_stage_in` goes through the public compile API under a
//! compute pipeline's stage-input descriptor:
//!
//! - The descriptor refusals are typed.
//! - A `Compiled` kernel's fetches must be the pipeline descriptor's; keys take the descriptor
//!   canonically.
//! - Every committed profile validates the modules (L3).
//! - A kernel the stage-in region reaches is never a device-replay variant.
//! - A dynamic layout records its stride, a fixed one none; a step by thread position records
//!   the region.
//!
//! `llvm-as` assembles the fixture; without it the tests skip visibly.

use std::path::Path;

use crate::budget::{Budget, SPIRV_WORDS};
use crate::device::profile::BufferAccess;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{DescriptorError, Error, Feature, LowerFamily, Needs, RuntimeInput};
use crate::library::{FunctionConstantValues, Library};
use crate::metal::{AttributeFormat, IndexType, StepFunction};
use crate::pipeline::compile::{compile_bitcode, plan_bitcode, EntryRef};
use crate::pipeline::descriptor::{
    StageInputAttributeDescriptor, StageInputDescriptor, StageInputLayoutDescriptor,
};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::lower_inputs::LoweringInputs;
use crate::pipeline::sym::{SlotClass, SlotRef, Stage};
use crate::pipeline::testing::{
    engine_dir, link_and_validate, llvm_assemble, meter, AssembleError,
};
use crate::pipeline::{
    ComputePipeline, ComputePipelineDescriptor, FunctionDescriptor, StageSource,
};
use crate::spirv::testing::committed_profiles;

fn fixture() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join("kernel_stage_in.ll")) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected kernel_stage_in.ll: {why}"),
    }
}

/// Committed profile `stem` as a device.
fn device(stem: &str) -> Device {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).expect("device")
}

/// A stage-input descriptor: `attributes` as `(index, format, offset, buffer)`, `layouts` as
/// `(buffer, stride, step, rate)`, and the index buffer.
fn descriptor(
    attributes: &[(u32, AttributeFormat, u32, u32)],
    layouts: &[(u32, u32, StepFunction, u32)],
    index: (IndexType, u32),
) -> StageInputDescriptor {
    let mut d = StageInputDescriptor::new();
    for &(i, format, offset, buffer) in attributes {
        d.attributes.insert(
            i,
            StageInputAttributeDescriptor::new(format, offset, buffer),
        );
    }
    for &(buffer, stride, step, rate) in layouts {
        let mut l = StageInputLayoutDescriptor::new(Some(stride));
        l.step_function = step;
        l.step_rate = rate;
        d.layouts.insert(buffer, l);
    }
    (d.index_type, d.index_buffer_index) = index;
    d
}

/// The fetch kernel's attributes: `f` (float3 into float4) at offset 4 and `h` (half2) at offset
/// 16 of layout 1, `u` (ushort4) of layout 2, and `i` (char) at offset 5 of layout 3.
fn fetch_attributes() -> [(u32, AttributeFormat, u32, u32); 4] {
    [
        (0, AttributeFormat::Float3, 4, 1),
        (1, AttributeFormat::UShort4, 0, 2),
        (2, AttributeFormat::Char, 5, 3),
        (3, AttributeFormat::Half2, 16, 1),
    ]
}

/// Layout 1 of stride 24, layout 2 of stride 8 with steps `x` and `y`, and layout 3 constant
/// at a step rate of 0, as Metal requires of a constant step (its stride is not read).
fn fetch_descriptor(
    x: (StepFunction, u32),
    y: (StepFunction, u32),
    index: IndexType,
) -> StageInputDescriptor {
    descriptor(
        &fetch_attributes(),
        &[
            (1, 24, x.0, x.1),
            (2, 8, y.0, y.1),
            (3, 999, StepFunction::Constant, 0),
        ],
        (index, 4),
    )
}

fn kslot(index: u32) -> SlotRef {
    SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Buffer,
        index,
    }
}

fn inputs(d: &StageInputDescriptor) -> LoweringInputs {
    let mut i = LoweringInputs::new();
    i.stage_input = Some(d.clone());
    i
}

fn function(bc: &[u8], entry: &str) -> FunctionDescriptor {
    let lib = Library::from_air(bc).expect("library");
    FunctionDescriptor::new(lib.function(entry).expect("function"))
}

fn compute(f: FunctionDescriptor, d: Option<&StageInputDescriptor>) -> ComputePipelineDescriptor {
    let mut c = ComputePipelineDescriptor::new(StageSource::Function(f));
    c.stage_input = d.cloned();
    c
}

fn pipeline(
    dev: &Device,
    bc: &[u8],
    entry: &str,
    d: Option<&StageInputDescriptor>,
) -> Result<ComputePipeline, Error> {
    dev.compile_compute_pipeline(&compute(function(bc, entry), d), &Budget::standard())
}

fn not_implemented(r: Result<ComputePipeline, Error>, feature: Feature) {
    match r {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, feature);
            assert_eq!(u.needs, Needs::NotImplemented);
        }
        other => panic!("{feature:?}: {other:?}"),
    }
}

/// A kernel reading a stage input needs the descriptor; one that lacks the attribute or its layout,
/// or steps every 0th thread, is invalid, and so are a render-stage step and a constant step at a
/// rate other than 0; an indexed step at a rate other than 1, and a format of the other numeric
/// class, are not implemented.
#[test]
fn descriptor_refusals_are_typed() {
    let Some(bc) = fixture() else { return };
    let dev = device("moltenvk-m3");
    let fetch = |d: Option<&StageInputDescriptor>| pipeline(&dev, &bc, "stage_in_fetch", d);
    let grid = |x: (StepFunction, u32)| {
        fetch_descriptor(
            x,
            (StepFunction::ThreadPositionInGridY, 1),
            IndexType::UInt16,
        )
    };
    let valid = grid((StepFunction::ThreadPositionInGridX, 1));
    assert!(fetch(Some(&valid)).is_ok());

    assert!(matches!(
        fetch(None),
        Err(Error::NeedsInput(RuntimeInput::StageInputDescriptor {
            attribute: 0
        }))
    ));
    let mut d = valid.clone();
    d.attributes.remove(&3);
    assert!(matches!(
        fetch(Some(&d)),
        Err(Error::InvalidDescriptor(
            DescriptorError::StageInputAttribute { attribute: 3 }
        ))
    ));
    let mut d = valid.clone();
    d.layouts.remove(&2);
    assert!(matches!(
        fetch(Some(&d)),
        Err(Error::InvalidDescriptor(
            DescriptorError::StageInputLayout { layout: 2 }
        ))
    ));
    for step in [
        StepFunction::ThreadPositionInGridX,
        StepFunction::ThreadPositionInGridXIndexed,
    ] {
        assert!(matches!(
            fetch(Some(&grid((step, 0)))),
            Err(Error::InvalidDescriptor(
                DescriptorError::StageInputStepRate {
                    layout: 1,
                    step_rate: 0
                }
            ))
        ));
    }
    // A constant step at a step rate other than 0 stops Metal's pipeline creation at an
    // assertion, with and without API validation.
    let mut d = valid.clone();
    d.layouts.get_mut(&3).expect("layout 3").step_rate = 1;
    assert!(matches!(
        fetch(Some(&d)),
        Err(Error::InvalidDescriptor(
            DescriptorError::StageInputStepRate {
                layout: 3,
                step_rate: 1
            }
        ))
    ));

    not_implemented(
        fetch(Some(&grid((StepFunction::ThreadPositionInGridXIndexed, 2)))),
        Feature::KernelStageIn,
    );
    // A vertex stage's step function stops Metal's compute pipeline creation at an assertion, the
    // default per-vertex step of a fresh layout among them.
    for step in [
        StepFunction::PerVertex,
        StepFunction::PerInstance,
        StepFunction::PerPatch,
        StepFunction::PerPatchControlPoint,
    ] {
        assert!(
            matches!(
                fetch(Some(&grid((step, 1)))),
                Err(Error::InvalidDescriptor(
                    DescriptorError::StageInputStepFunction {
                        layout: 1,
                        step_function
                    }
                )) if step_function == step
            ),
            "{step:?}"
        );
    }
    let mut d = valid.clone();
    d.layouts
        .insert(1, StageInputLayoutDescriptor::new(Some(24)));
    assert!(matches!(
        fetch(Some(&d)),
        Err(Error::InvalidDescriptor(
            DescriptorError::StageInputStepFunction {
                layout: 1,
                step_function: StepFunction::PerVertex
            }
        ))
    ));
    // `u` is a `uint4`: a float format is a conversion whose Metal meaning is not established.
    let mut d = valid.clone();
    d.attributes.insert(
        1,
        StageInputAttributeDescriptor::new(AttributeFormat::Float4, 0, 2),
    );
    not_implemented(fetch(Some(&d)), Feature::Lowering(LowerFamily::StageIo));
    let mut d = valid.clone();
    d.attributes.insert(
        0,
        StageInputAttributeDescriptor::new(AttributeFormat::Invalid, 4, 1),
    );
    assert!(matches!(
        fetch(Some(&d)),
        Err(Error::InvalidDescriptor(
            DescriptorError::PlaceholderEnumValue { .. }
        ))
    ));
}

/// The kernel's key takes the descriptor canonically (a constant layout's stride and rate, and an
/// unread layout or index buffer, do not change it), the pipeline key is the one
/// `compute_pipeline_key` computes, and a `Compiled` kernel must fetch as the pipeline's
/// descriptor does.
#[test]
fn compiled_kernels_and_keys_follow_the_descriptor() {
    let Some(bc) = fixture() else { return };
    let dev = device("moltenvk-m3");
    let f = function(&bc, "stage_in_fetch");
    let a = fetch_descriptor(
        (StepFunction::ThreadPositionInGridX, 1),
        (StepFunction::ThreadPositionInGridY, 1),
        IndexType::UInt16,
    );
    let key = |d: &StageInputDescriptor| dev.function_compile_key(&f, &inputs(d));

    // Canonical: the constant layout's stride and rate, an extra layout, and the index buffer of
    // a descriptor without an indexed step (the key, which takes no check of the layouts).
    let mut same = a.clone();
    let l3 = same.layouts.get_mut(&3).expect("layout 3");
    (l3.stride, l3.step_rate) = (None, 1);
    same.layouts
        .insert(9, StageInputLayoutDescriptor::new(Some(64)));
    (same.index_type, same.index_buffer_index) = (IndexType::UInt32, 6);
    assert_eq!(key(&same), key(&a));
    let mut other = a.clone();
    other.layouts.get_mut(&2).expect("layout 2").stride = Some(12);
    assert_ne!(key(&other), key(&a));
    let mut dynamic = a.clone();
    dynamic.layouts.get_mut(&2).expect("layout 2").stride = None;
    assert_ne!(key(&dynamic), key(&a));
    assert_ne!(
        key(&a),
        dev.function_compile_key(&f, &LoweringInputs::new())
    );

    let pipe = dev
        .compile_compute_pipeline(&compute(f.clone(), Some(&a)), &Budget::standard())
        .expect("pipeline");
    assert_eq!(
        pipe.key,
        dev.compute_pipeline_key(&compute(f.clone(), Some(&a)))
    );

    let c = dev
        .compile_function(&f, &inputs(&a), &Budget::standard())
        .expect("compile");
    assert_eq!(*c.key(), key(&a));
    let compiled = |d: Option<&StageInputDescriptor>| {
        let mut p = ComputePipelineDescriptor::new(StageSource::Compiled(c.clone()));
        p.stage_input = d.cloned();
        dev.compile_compute_pipeline(&p, &Budget::standard())
    };
    let from_compiled = compiled(Some(&a)).expect("the descriptor it was compiled under");
    assert_eq!(from_compiled.stage.spirv, pipe.stage.spirv);
    // Metal refuses the extra layout of a nonzero stride that no attribute names, and takes it at
    // stride 0.
    assert!(matches!(
        compiled(Some(&same)),
        Err(Error::InvalidDescriptor(
            DescriptorError::StageInputLayout { layout: 9 }
        ))
    ));
    // At stride 0 the extra layout is Metal's unset one; the constant layout's rate goes back to
    // 0, as Metal requires of a pipeline.
    same.layouts
        .insert(9, StageInputLayoutDescriptor::new(Some(0)));
    same.layouts.get_mut(&3).expect("layout 3").step_rate = 0;
    assert!(compiled(Some(&same)).is_ok());
    assert!(matches!(
        compiled(Some(&other)),
        Err(Error::InvalidDescriptor(
            DescriptorError::StageInputAttribute { attribute: 1 }
        ))
    ));
    assert!(matches!(
        compiled(None),
        Err(Error::NeedsInput(RuntimeInput::StageInputDescriptor {
            attribute: 0
        }))
    ));
    // A kernel compiled on its own without the descriptor needs it.
    assert!(matches!(
        dev.compile_function(&f, &LoweringInputs::new(), &Budget::standard()),
        Err(Error::NeedsInput(RuntimeInput::StageInputDescriptor {
            attribute: 0
        }))
    ));
}

/// The descriptors the validation checks compile under: every step.
fn every_step() -> Vec<StageInputDescriptor> {
    use StepFunction as S;
    vec![
        fetch_descriptor(
            (S::ThreadPositionInGridX, 1),
            (S::ThreadPositionInGridY, 1),
            IndexType::UInt16,
        ),
        fetch_descriptor(
            (S::ThreadPositionInGridX, 3),
            (S::ThreadPositionInGridY, 2),
            IndexType::UInt16,
        ),
        fetch_descriptor(
            (S::ThreadPositionInGridXIndexed, 1),
            (S::ThreadPositionInGridYIndexed, 1),
            IndexType::UInt32,
        ),
        fetch_descriptor((S::Constant, 1), (S::Constant, 1), IndexType::UInt16),
    ]
}

/// L3 on every committed profile: each step's module links and validates wherever it compiles.
#[test]
fn every_committed_profile_validates_the_fetch() {
    let Some(bc) = fixture() else { return };
    let region = descriptor(
        &[(0, AttributeFormat::Float, 0, 1)],
        &[(1, 4, StepFunction::ThreadPositionInGridX, 1)],
        (IndexType::UInt16, 0),
    );
    let mut cases: Vec<(&str, StageInputDescriptor)> = every_step()
        .into_iter()
        .map(|d| ("stage_in_fetch", d))
        .collect();
    cases.push(("stage_in_region", region.clone()));
    cases.push(("stage_in_region_short", region));
    let mut compiled = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        // The `BufferRangeLimit` variant reaches the source and index buffers by device address
        // where the profile has them.
        let addressed = p.fields().buffer_access == BufferAccess::Addressed;
        let variants: &[bool] = if addressed { &[false, true] } else { &[false] };
        for ((entry, d), &psb) in cases
            .iter()
            .flat_map(|c| variants.iter().map(move |v| (c, v)))
        {
            let mut i = inputs(d);
            if psb {
                i.psb_slots = [1, 2, 4].into_iter().map(kslot).collect();
            }
            let consts = FunctionConstantValues::defaults();
            let a = match compile_bitcode(
                &bc,
                EntryRef::Name(entry),
                &consts,
                &p,
                &i,
                &mut meter(),
                SPIRV_WORDS,
            ) {
                Ok(a) => a,
                Err(Error::Unsupported(u)) if u.needs != Needs::NotImplemented => continue,
                Err(e) => panic!("{stem} {entry} psb={psb} {d:?}: {e}"),
            };
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem} {entry}: {e}"));
            // The artifact lists every fetched attribute with its descriptor entries.
            let listed = a
                .interface
                .io
                .stage_in
                .iter()
                .map(|s| s.index)
                .collect::<Vec<_>>();
            let want: Vec<u32> = if *entry == "stage_in_fetch" {
                vec![0, 1, 2, 3]
            } else if *entry == "stage_in_region" {
                vec![0]
            } else {
                Vec::new()
            };
            assert_eq!(listed, want, "{stem} {entry}");
            compiled += 1;
        }
    }
    assert!(compiled >= cases.len(), "compiled {compiled}");
}

/// A kernel the stage-in region reaches is never a device-replay variant, so the replay, which
/// ignores a command's region, never runs one with its region dropped. Compiled as a replay variant
/// (`LoweringInputs::replay_bindings`) under a stage-input descriptor, each fixture kernel (the
/// fetch, and the `[[grid_origin]]` and `[[grid_size]]` readers) is refused as
/// `GpuEncodedIndirectCommands` on every committed profile with device addresses; a reader of the
/// region without a descriptor that compiles plans `PushField::StageInRegion`.
#[test]
fn a_kernel_the_stage_in_region_reaches_is_no_replay_variant() {
    use crate::pipeline::sym::PushField;
    let Some(bc) = fixture() else { return };
    let region = descriptor(
        &[(0, AttributeFormat::Float, 0, 1)],
        &[(1, 4, StepFunction::ThreadPositionInGridX, 1)],
        (IndexType::UInt16, 0),
    );
    let fetch = fetch_descriptor(
        (StepFunction::ThreadPositionInGridX, 1),
        (StepFunction::ThreadPositionInGridY, 1),
        IndexType::UInt16,
    );
    let consts = FunctionConstantValues::defaults();
    let mut checked = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        if p.fields().buffer_access != BufferAccess::Addressed {
            continue;
        }
        for (entry, d) in [
            ("stage_in_fetch", &fetch),
            ("stage_in_region", &region),
            ("stage_in_region_short", &region),
        ] {
            let mut i = inputs(d);
            i.replay_bindings = true;
            let r = compile_bitcode(
                &bc,
                EntryRef::Name(entry),
                &consts,
                &p,
                &i,
                &mut meter(),
                SPIRV_WORDS,
            );
            match r {
                Err(Error::Unsupported(u)) => {
                    assert_eq!(
                        u.feature,
                        Feature::GpuEncodedIndirectCommands,
                        "{stem} {entry}"
                    )
                }
                other => panic!("{stem} {entry}: {:?}", other.map(|_| ())),
            }
            checked += 1;
        }
        let mut i = LoweringInputs::new();
        i.replay_bindings = true;
        if let Ok(plan) = plan_bitcode(
            &bc,
            EntryRef::Name("stage_in_region_short"),
            &consts,
            &p,
            &i,
            &mut meter(),
        ) {
            assert!(
                plan.push
                    .fields
                    .iter()
                    .any(|f| f.field == PushField::StageInRegion),
                "{stem}"
            );
        }
    }
    assert!(checked > 0, "no committed profile with device addresses");
}

/// A layout of a fixed stride fetches at it whatever stride the encoder binds its buffer with, so
/// a kernel whose layouts all have one records no `RuntimeFact::StageInStride`. A layout of
/// `MTLBufferLayoutStrideDynamic` reads the stride its buffer is bound with: the kernel records
/// `RuntimeFact::StageInStride` for each dynamic layout it reads, and a constant step's layout reads
/// no stride.
#[test]
fn a_dynamic_layout_records_its_stride_and_a_fixed_one_none() {
    let Some(bc) = fixture() else { return };
    let mut d = fetch_descriptor(
        (StepFunction::ThreadPositionInGridX, 1),
        (StepFunction::ThreadPositionInGridY, 1),
        IndexType::UInt16,
    );
    let dev = device("moltenvk-m3");
    let strides = |d: &StageInputDescriptor| -> Vec<u32> {
        let pipe = pipeline(&dev, &bc, "stage_in_fetch", Some(d)).expect("pipeline");
        pipe.dependencies
            .iter()
            .filter_map(|f| match f.fact {
                RuntimeFact::StageInStride { layout } => Some(layout),
                _ => None,
            })
            .collect()
    };
    assert!(strides(&d).is_empty(), "fixed strides");
    for layout in [1, 2, 3] {
        d.layouts.get_mut(&layout).expect("layout").stride = None;
    }
    assert_eq!(strides(&d), [1, 2], "dynamic strides");
}

/// A step by thread position at any rate or through the index buffer reads the stage-in region,
/// so the kernel records `RuntimeFact::StageInRegion`.
#[test]
fn a_step_by_thread_position_records_the_region() {
    let Some(bc) = fixture() else { return };
    let dev = device("moltenvk-m3");
    for x in [
        (StepFunction::ThreadPositionInGridX, 2),
        (StepFunction::ThreadPositionInGridXIndexed, 1),
        (StepFunction::ThreadPositionInGridX, 1),
    ] {
        let d = fetch_descriptor(x, (StepFunction::Constant, 0), IndexType::UInt16);
        let pipe = pipeline(&dev, &bc, "stage_in_fetch", Some(&d)).expect("pipeline");
        assert!(
            pipe.dependencies
                .iter()
                .any(|f| f.fact == RuntimeFact::StageInRegion),
            "{x:?}: {:?}",
            pipe.dependencies
        );
    }
}

/// A `half` attribute read into the kernel's `half2` and reinterpreted unchanged keeps the field's
/// own bits, as Metal's fetch does: `stage_in_fetch`'s `as_type<ushort>(in.h.x)` and `.y` reuse
/// the bits the fetch widened, through the vector the fetch constructs of its lanes
/// (`narrow::to_bits`), so its module encodes no `half` (no `OpFConvert` to a 16-bit float, no
/// `PackHalf2x16`) on any committed profile, with the attribute's format `Half2` (both lanes read)
/// or `Half` (the second lane missing, `0`). An encode there narrows the device's widening of a
/// NaN field to the device's NaN: the RTX 5080 Laptop read `0x7E01` and `0xFE01` as `0x7FFF`
/// (job m3-max-68835).
#[test]
fn a_half_attribute_reinterpreted_unchanged_encodes_no_half() {
    use crate::lower::arith::narrow::quantize_form;
    let Some(bc) = fixture() else { return };
    let (mut compiled, mut encoding) = (Vec::new(), Vec::new());
    for format in [AttributeFormat::Half2, AttributeFormat::Half] {
        let mut d = fetch_descriptor(
            (StepFunction::ThreadPositionInGridX, 1),
            (StepFunction::ThreadPositionInGridY, 1),
            IndexType::UInt16,
        );
        d.attributes
            .insert(3, StageInputAttributeDescriptor::new(format, 16, 1));
        for (stem, p) in committed_profiles().expect("profiles") {
            let consts = FunctionConstantValues::defaults();
            let a = match compile_bitcode(
                &bc,
                EntryRef::Name("stage_in_fetch"),
                &consts,
                &p,
                &inputs(&d),
                &mut meter(),
                SPIRV_WORDS,
            ) {
                Ok(a) => a,
                Err(Error::Unsupported(u)) if u.needs != Needs::NotImplemented => continue,
                Err(e) => panic!("{stem} {format:?}: {e}"),
            };
            let encodes = half_encodes(&a.module.words);
            if encodes > 0 {
                encoding.push((stem.clone(), format, encodes));
            }
            compiled.push((stem, format, quantize_form(&p)));
        }
    }
    // Each `half` form of the profiles is among the modules: with `shaderFloat16` under the float
    // controls and without them, and without it.
    let forms: std::collections::BTreeSet<_> =
        compiled.iter().map(|(_, _, q)| format!("{q:?}")).collect();
    assert_eq!(forms.len(), 3, "a profile of each form: {compiled:?}");
    assert!(
        encoding.is_empty(),
        "{} of {} modules encode a half: {encoding:?}",
        encoding.len(),
        compiled.len()
    );
}

/// The module's encodes of a `half` to its bits by the device: its `OpFConvert`s to a 16-bit
/// float (or a vector of them) and its `GLSL.std.450 PackHalf2x16`s.
fn half_encodes(words: &[u32]) -> usize {
    use crate::spirv::ir::GlslOp;
    use crate::spirv_binary::{parse_bytes, Operand};
    use crate::spv::Op;
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let parsed = parse_bytes(&bytes).expect("parse");
    let id = |o: Option<&Operand>| match o {
        Some(Operand::Word(w) | Operand::IdRef(w)) => Some(*w),
        _ => None,
    };
    let pack = GlslOp::PackHalf2x16.to_spirv() as u32;
    let mut halves = std::collections::HashSet::new();
    let mut encodes = 0;
    for i in &parsed.instructions {
        match i.opcode {
            Op::TypeFloat if id(i.operands.first()) == Some(16) => halves.extend(i.result_id),
            Op::TypeVector if id(i.operands.first()).is_some_and(|e| halves.contains(&e)) => {
                halves.extend(i.result_id)
            }
            Op::FConvert if i.result_type.is_some_and(|t| halves.contains(&t)) => encodes += 1,
            Op::ExtInst
                if matches!(
                    i.operands.get(1),
                    Some(Operand::LiteralExtInstInteger(n)) if *n == pack
                ) =>
            {
                encodes += 1
            }
            _ => {}
        }
    }
    encodes
}
