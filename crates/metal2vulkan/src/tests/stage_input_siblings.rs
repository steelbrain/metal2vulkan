//! What a sweep of compute pipeline states over **one** `[[stage_in]]` kernel costs per state, as
//! an attribute-format sweep is: one kernel, one stride, one attribute offset, and a different
//! attribute **format** a state. No device is opened; the compile layer is reached directly.
//!
//! # Why counts and not times
//!
//! The question these answer is whether the work one such state costs **grows with how many the
//! sweep has already made**, which a timing cannot settle on a shared host and a count settles
//! exactly. [`Meter::work`] counts deterministic work units -- the same input on the same profile
//! spends the same, whatever the load -- and [`StageEntries::counts`] counts AIR decodes and the
//! decodes an earlier ask spared. Both are read here; no wall clock is.
//!
//! # What they hold
//!
//! - **Flat, not growing.** A sibling's compile costs the same work units whether it is the first
//!   of one or the thirty-second of thirty-two: the compile layer keeps no cache, so a sibling's
//!   cost is a function of its own inputs alone. There is no work here that scales with the
//!   siblings ([`a_siblings_compile_costs_the_same_however_many_came_before`]).
//! - **Flat is not free.** What every sibling of such a sweep redoes is its kernel's decode and
//!   specialization, which its stage-input descriptor does not decide: a [`StageEntries`] shared
//!   between the siblings decodes and specializes the kernel **once** and spares every later one,
//!   and compiles the same module from it, where one object a sibling decodes once a sibling
//!   ([`a_sweeps_siblings_share_one_decode`]). What it spares is time and not budget: each
//!   sibling is charged the work units of its own unshared compile, the entry it found included.
//!
//! A render creation gives one such object to its checks, its keys and its compiles; a compute
//! creation gives none ([`Device::compile_function`] rather than
//! [`Device::compile_function_from_entries`]), so what the second test counts as spared is what a
//! sweep of compute states over one kernel pays today.

use std::path::Path;

use crate::budget::{Budget, Meter};
use crate::device::requirements::VkFeature;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::library::Library;
use crate::metal::{AttributeFormat, StepFunction};
use crate::pipeline::compile::{compile_stage, compile_stage_from, InputsFrom, StageEntries};
use crate::pipeline::descriptor::{
    StageInputAttributeDescriptor, StageInputDescriptor, StageInputLayoutDescriptor,
};
use crate::pipeline::lower_inputs::LoweringInputs;
use crate::pipeline::testing::{engine_dir, llvm_assemble, AssembleError};
use crate::pipeline::FunctionDescriptor;

/// The entry the sweep is over: a kernel whose first `[[stage_in]]` member is a `float4`, so every
/// format below reads into it.
const ENTRY: &str = "stage_in_fetch";

/// The attribute formats of a sibling, one a state, every one of them a source encoding a `float4`
/// attribute reads: what varies between the siblings of an attribute-format sweep, over one
/// kernel, one stride and one offset. Thirty-two of them, so a run of 32 gives each its own.
const FORMATS: [AttributeFormat; 32] = [
    AttributeFormat::Float4,
    AttributeFormat::Float3,
    AttributeFormat::Float2,
    AttributeFormat::Float,
    AttributeFormat::Half4,
    AttributeFormat::Half3,
    AttributeFormat::Half2,
    AttributeFormat::Half,
    AttributeFormat::UChar4Normalized,
    AttributeFormat::UChar3Normalized,
    AttributeFormat::UChar2Normalized,
    AttributeFormat::UCharNormalized,
    AttributeFormat::Char4Normalized,
    AttributeFormat::Char3Normalized,
    AttributeFormat::Char2Normalized,
    AttributeFormat::CharNormalized,
    AttributeFormat::UShort4Normalized,
    AttributeFormat::UShort3Normalized,
    AttributeFormat::UShort2Normalized,
    AttributeFormat::UShortNormalized,
    AttributeFormat::Short4Normalized,
    AttributeFormat::Short3Normalized,
    AttributeFormat::Short2Normalized,
    AttributeFormat::ShortNormalized,
    AttributeFormat::UChar4NormalizedBGRA,
    AttributeFormat::Int1010102Normalized,
    AttributeFormat::UInt1010102Normalized,
    AttributeFormat::FloatRG11B10,
    AttributeFormat::FloatRGB9E5,
    AttributeFormat::UChar4,
    AttributeFormat::UShort4,
    AttributeFormat::Short4,
];

/// The sibling counts a sweep is measured at: the case the measurement came from makes 29, and the
/// cases it is compared against make one.
const RUNS: [usize; 6] = [1, 2, 4, 8, 16, 32];

/// The committed profile the sweep compiles for.
fn device() -> Device {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/moltenvk-m3.json");
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    desc.features.insert(VkFeature::RobustBufferAccess);
    Device::new(desc, DevicePolicy::default()).expect("device")
}

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

/// The sweep's one kernel.
fn kernel(bitcode: &[u8]) -> FunctionDescriptor {
    let library = Library::from_air(bitcode).expect("library");
    FunctionDescriptor::new(library.function(ENTRY).expect("function"))
}

/// Sibling `format`'s lowering inputs: the kernel's four attributes over two layouts of one
/// stride each, with attribute 0 reading `format`. Nothing but that format differs between
/// siblings, as an attribute-format sweep's rows differ.
fn sibling(format: AttributeFormat) -> LoweringInputs {
    let mut stage_input = StageInputDescriptor::new();
    for (index, attribute) in [
        (0, StageInputAttributeDescriptor::new(format, 0, 1)),
        (
            1,
            StageInputAttributeDescriptor::new(AttributeFormat::UShort4, 16, 2),
        ),
        (
            2,
            StageInputAttributeDescriptor::new(AttributeFormat::Char, 5, 3),
        ),
        (
            3,
            StageInputAttributeDescriptor::new(AttributeFormat::Half2, 24, 1),
        ),
    ] {
        stage_input.attributes.insert(index, attribute);
    }
    for (buffer, stride, step) in [
        (1, 32, StepFunction::ThreadPositionInGridX),
        (2, 32, StepFunction::ThreadPositionInGridX),
        (3, 0, StepFunction::Constant),
    ] {
        let mut layout = StageInputLayoutDescriptor::new(Some(stride));
        layout.step_function = step;
        stage_input.layouts.insert(buffer, layout);
    }
    let mut inputs = LoweringInputs::new();
    inputs.stage_input = Some(stage_input);
    inputs
}

/// The work units compiling sibling `format` of the sweep costs on its own, as the creation path
/// compiles it: no entry shared, so the compile decodes and specializes the kernel itself.
fn alone(device: &Device, kernel: &FunctionDescriptor, format: AttributeFormat) -> u64 {
    let inputs = sibling(format);
    let mut meter = Meter::new(&Budget::standard());
    compile_stage(
        &kernel.function,
        &kernel.constants,
        (&kernel.linked, &kernel.libraries),
        device.profile(),
        InputsFrom::Given(&inputs),
        &mut meter,
    )
    .expect("the sibling compiles");
    meter.work()
}

/// A sibling's compile costs the same deterministic work however many siblings of the sweep were
/// compiled before it: 1, 2, 4, 8, 16 and 32 states over one kernel, and the work units of the
/// state that reads a given format are the same in every run that reaches it.
///
/// This is the shape claim, and it is an equality rather than a bound: the compile layer keeps no
/// cache, so a state's cost is a function of its own inputs and there is nothing here for a
/// sibling count to grow. A per-state cost that rose with the siblings would be superlinear work
/// in the sweep, and these runs would read it.
#[test]
fn a_siblings_compile_costs_the_same_however_many_came_before() {
    let Some(bitcode) = fixture() else { return };
    let device = device();
    let kernel = kernel(&bitcode);
    // Each format's work units, as the run of that length read them.
    let mut by_run: Vec<(usize, Vec<u64>)> = Vec::new();
    for &n in &RUNS {
        let work = FORMATS[..n]
            .iter()
            .map(|&format| alone(&device, &kernel, format))
            .collect::<Vec<_>>();
        by_run.push((n, work));
    }
    let (_, longest) = by_run.last().expect("a run");
    for (n, work) in &by_run {
        for (at, (&read, &whole)) in work.iter().zip(longest.iter()).enumerate() {
            assert_eq!(
                read,
                whole,
                "sibling {at} of a run of {n} must cost the work it costs as sibling {at} of a \
                 run of {}: {read} work units against {whole}, so the run length decides nothing",
                FORMATS.len()
            );
        }
    }
    // And the run as a whole costs the sum of its siblings, each flat: so the sweep is linear in
    // the siblings, never quadratic. Reported, so the figures stand beside the equality.
    let totals = by_run
        .iter()
        .map(|(n, work)| (*n, work.iter().sum::<u64>()))
        .collect::<Vec<_>>();
    let (_, one) = totals.first().expect("a run");
    for (n, total) in &totals {
        let per = total / u64::try_from(*n).expect("a run length");
        assert!(
            per <= one * 2,
            "a run of {n} states must not cost more than twice the one state's work units a \
             state: {per} a state against {one}, over {totals:?}"
        );
    }
    eprintln!("work units per run (siblings, total): {totals:?}");
}

/// Every sibling of such a sweep decodes and specializes the same kernel again, and need not: the
/// stage-input descriptor decides the **lowering** and not the specialization, so one
/// [`StageEntries`] shared between the siblings decodes the kernel once and spares every later
/// one, while an object a sibling decodes once a sibling.
///
/// Counted in both directions, so neither reading is vacuous: shared reads one decode and `n - 1`
/// spared, and apart reads `n` decodes and nothing spared. The module is held equal to the plain
/// compile's word for word in both, so the sharing changes no output.
#[test]
fn a_sweeps_siblings_share_one_decode() {
    let Some(bitcode) = fixture() else { return };
    let device = device();
    let kernel = kernel(&bitcode);
    // The module the creation path's own compile produces for each sibling, which the shared and
    // apart forms are both held to.
    let plain = FORMATS
        .iter()
        .map(|&format| {
            let inputs = sibling(format);
            let mut meter = Meter::new(&Budget::standard());
            let compiled = compile_stage(
                &kernel.function,
                &kernel.constants,
                (&kernel.linked, &kernel.libraries),
                device.profile(),
                InputsFrom::Given(&inputs),
                &mut meter,
            )
            .expect("the sibling compiles");
            (
                format,
                compiled.artifact().module.words.clone(),
                meter.work(),
            )
        })
        .collect::<Vec<_>>();

    // One object every sibling of the sweep reads, as a render creation gives one object to its
    // stages.
    let shared = StageEntries::new();
    let mut shared_work = Vec::new();
    for (format, words, _) in &plain {
        let inputs = sibling(*format);
        let mut meter = Meter::new(&Budget::standard());
        let compiled = compile_stage_from(
            &kernel,
            device.profile(),
            InputsFrom::Given(&inputs),
            &shared,
            &mut meter,
        )
        .expect("the sibling compiles");
        assert_eq!(
            &compiled.artifact().module.words,
            words,
            "a sibling compiled from a shared entry must be the module its own compile makes, \
             word for word ({format:?})"
        );
        shared_work.push(meter.work());
    }
    let n = u64::try_from(FORMATS.len()).expect("the formats");
    assert_eq!(
        shared.counts(),
        (1, n - 1),
        "{n} siblings over one kernel must decode it once and spare the other {} decodes",
        n - 1
    );

    // The same asks with an object each, which is what the creation path does today: every
    // sibling decodes and specializes the kernel itself.
    let mut apart_work = Vec::new();
    let mut apart_decodes = 0;
    for (format, words, _) in &plain {
        let inputs = sibling(*format);
        let entries = StageEntries::new();
        let mut meter = Meter::new(&Budget::standard());
        let compiled = compile_stage_from(
            &kernel,
            device.profile(),
            InputsFrom::Given(&inputs),
            &entries,
            &mut meter,
        )
        .expect("the sibling compiles");
        assert_eq!(
            &compiled.artifact().module.words,
            words,
            "a sibling compiled from its own entry must be the module its own compile makes, \
             word for word ({format:?})"
        );
        let (made, found) = entries.counts();
        assert_eq!(
            (made, found),
            (1, 0),
            "an object of its own decodes for its one sibling and spares nothing ({format:?})"
        );
        apart_decodes += made;
        apart_work.push(meter.work());
    }
    assert_eq!(
        apart_decodes, n,
        "{n} siblings with an object each must decode the kernel {n} times"
    );

    // A found entry is charged what making it charged: every sibling costs the work units of its
    // own unshared compile, from the shared entry or from one of its own.
    for (at, (format, _, plain)) in plain.iter().enumerate() {
        assert_eq!(
            (shared_work[at], apart_work[at]),
            (*plain, *plain),
            "a sibling must cost the work units of its own compile, {plain}, from a shared entry \
             and from one of its own ({format:?})"
        );
    }
    // The entries are told apart by the kernel and its depth alone, never by the stage input, so
    // the sweep's siblings are one entry and not thirty-two: the object holds one.
    let held = format!("{shared:?}");
    assert!(
        held.contains("made: 1"),
        "the shared object must hold one entry for the sweep, not one a sibling: {held}"
    );
}
