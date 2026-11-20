//! What a compute pipeline's **unread** stage-input attributes cost its kernel's translation.
//!
//! A `MTLStageInputOutputDescriptor` may state attributes the kernel's `[[stage_in]]` does not
//! read: Metal ties the two together only through the layout agreement
//! (`StageInputDescriptor::check_layouts`), never by requiring a set attribute to reach a
//! declaration. The translator fetches **one attribute at a time**
//! (`lower::stage_io::stage_in::param`, reached from `lower::select::select_params` for a
//! parameter an instruction uses, through `StageInFetch::of(d, attribute)`), so an attribute no
//! instruction reads reaches no instruction of the module.
//!
//! These measure that, in the two readings that settle it: the module a kernel compiles to does
//! **not** move when an attribute it does not read moves, and the key it is compiled under
//! **does**.
//!
//! # The open bug these pin
//!
//! The key is the wider of the two. `StageInputDescriptor::canonical`, which is what a compile key
//! takes of the descriptor (`pipeline::compile::key::canonical_inputs`), filters the **layouts** to
//! the ones an attribute names and keeps **every attribute**; `compute_fields`
//! (`pipeline::compile::api`) writes the whole descriptor into the compute pipeline key beside it.
//! So a kernel reading five attributes is translated again, and its pipeline compiled again, for
//! every value of a sixth it never reads — [`an_unread_attribute_keys_a_translation_of_its_own`]
//! counts how many, and [`an_unread_attribute_changes_no_word_of_the_module`] says every one of
//! those translations produced the same module.
//!
//! Narrowing the key needs the set of attributes the entry reads, which is a property of the
//! decoded and specialized entry (`lower::select::select_params` plans a stage-in parameter only
//! where an instruction uses it). `canonical` has no entry, and neither
//! `Device::function_compile_key` nor `Device::compute_pipeline_key` decodes one, so the narrowing
//! is not a change to `canonical` alone. The vertex path has the entry where it builds its
//! projection (`pipeline::compile::declared_attributes`, read in `lower_inputs`), which is why its
//! key carries the attributes its entry reads and no others.
//!
//! The two counts below are asserted at what they read today, so the narrowing cannot land
//! quietly: it fails these and names its new number.
//!
//! # Why keys and words, and not a time
//!
//! A key is a digest of what the compile depends on, so two descriptors of one key are one
//! translation and two keys are two. The count of distinct keys over a sweep is therefore the
//! number of translations that sweep costs, exactly, on any host under any load. The words are the
//! other half: they say what those translations had in common.
//!
//! The sweep is [`crate::tests::stage_in_unread_key::rows`], which mirrors the conformance case
//! `stage-in-unread-attributes` axis for axis: an unread attribute's format, its byte offset, the
//! buffer it names, its layout's step function and step rate, its layout's stride, and the
//! descriptor's index type and index buffer index, with the rows that cross two of those at once.

use std::collections::BTreeSet;
use std::path::Path;

use crate::budget::{Budget, Meter};
use crate::device::requirements::VkFeature;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::library::Library;
use crate::metal::{AttributeFormat, IndexType, StepFunction};
use crate::pipeline::compile::{compile_stage, InputsFrom, StageEntries};
use crate::pipeline::descriptor::{
    ComputePipelineDescriptor, StageInputAttributeDescriptor, StageInputDescriptor,
    StageInputLayoutDescriptor, StageSource,
};
use crate::pipeline::lower_inputs::LoweringInputs;
use crate::pipeline::testing::{engine_dir, llvm_assemble, AssembleError};
use crate::pipeline::FunctionDescriptor;

/// The entry the sweep is over: a kernel whose `[[stage_in]]` declares and reads attributes 0 to 4.
const ENTRY: &str = "stage_in_fetch_wide";

/// The Metal buffer index every read attribute's layout reads.
const SOURCE: u32 = 1;

/// The index buffer indices the sweep names.
const INDEX_A: u32 = 2;
const INDEX_B: u32 = 3;

/// Buffer indices no read attribute names, for an unread attribute's own layout.
const SPARE_A: u32 = 4;
const SPARE_B: u32 = 5;

/// The read layout's stride, which is one element of the source buffer.
const ELEMENT: u32 = 64;

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

/// The five attributes the kernel declares and reads, which every row states identically: each at
/// its own offset in one 64-byte element of [`SOURCE`].
fn read_attributes(d: &mut StageInputDescriptor) {
    for (index, format, offset) in [
        (0, AttributeFormat::Float4, 0),
        (1, AttributeFormat::UInt4, 16),
        (2, AttributeFormat::Int, 32),
        (3, AttributeFormat::Half2, 36),
        (4, AttributeFormat::Int4, 48),
    ] {
        d.attributes.insert(
            index,
            StageInputAttributeDescriptor::new(format, offset, SOURCE),
        );
    }
    let mut layout = StageInputLayoutDescriptor::new(Some(ELEMENT));
    layout.step_function = StepFunction::ThreadPositionInGridX;
    layout.step_rate = 1;
    d.layouts.insert(SOURCE, layout);
}

/// One attribute the descriptor sets past the kernel's five.
#[derive(Clone, Copy, Debug)]
struct Unread {
    index: u32,
    format: AttributeFormat,
    offset: u32,
    buffer: u32,
}

/// One buffer layout past the read attributes' own.
#[derive(Clone, Copy, Debug)]
struct Spare {
    buffer: u32,
    stride: Option<u32>,
    step: StepFunction,
    rate: u32,
}

/// An unread attribute at index 5.
fn at5(format: AttributeFormat, offset: u32, buffer: u32) -> Unread {
    Unread {
        index: 5,
        format,
        offset,
        buffer,
    }
}

/// A spare layout of the read layout's own stride, stepped by the thread's x position at rate 1.
fn plain_spare(buffer: u32) -> Spare {
    Spare {
        buffer,
        stride: Some(ELEMENT),
        step: StepFunction::ThreadPositionInGridX,
        rate: 1,
    }
}

/// The descriptor of one row: the read set, then the row's own attributes and layouts and its
/// index fields.
fn descriptor(
    unread: &[Unread],
    spares: &[Spare],
    index: (IndexType, u32),
) -> StageInputDescriptor {
    let mut d = StageInputDescriptor::new();
    read_attributes(&mut d);
    for held in unread {
        d.attributes.insert(
            held.index,
            StageInputAttributeDescriptor::new(held.format, held.offset, held.buffer),
        );
    }
    for held in spares {
        let mut layout = StageInputLayoutDescriptor::new(held.stride);
        layout.step_function = held.step;
        layout.step_rate = held.rate;
        d.layouts.insert(held.buffer, layout);
    }
    d.index_type = index.0;
    d.index_buffer_index = index.1;
    d
}

/// The sweep: one descriptor a row, the first of them the baseline that states the read set alone.
///
/// Every axis of the conformance case `stage-in-unread-attributes`, and its crossings.
fn rows() -> Vec<(String, StageInputDescriptor)> {
    let plain = (IndexType::UInt32, INDEX_A);
    let mut rows = vec![(
        "no-unread-attributes".to_owned(),
        descriptor(&[], &[], plain),
    )];
    // The format axis. `Invalid` is an unset attribute, so it names the read buffer and states no
    // spare layout: a spare layout of a nonzero stride no set attribute names is a descriptor
    // Metal stops at.
    for format in [
        AttributeFormat::UInt,
        AttributeFormat::Float4,
        AttributeFormat::UChar4Normalized,
        AttributeFormat::Half2,
        AttributeFormat::Int4,
        AttributeFormat::Short2,
        AttributeFormat::UInt1010102Normalized,
        AttributeFormat::Invalid,
    ] {
        let row = match format {
            AttributeFormat::Invalid => descriptor(&[at5(format, 0, SOURCE)], &[], plain),
            _ => descriptor(&[at5(format, 0, SPARE_A)], &[plain_spare(SPARE_A)], plain),
        };
        rows.push((format!("format-{}", format.name()), row));
    }
    // The offset axis.
    for offset in [0, 4, 32, 48] {
        rows.push((
            format!("offset-{offset}"),
            descriptor(
                &[at5(AttributeFormat::UShort4, offset, SPARE_A)],
                &[plain_spare(SPARE_A)],
                plain,
            ),
        ));
    }
    // The buffer axis.
    for (name, buffer) in [
        ("source", SOURCE),
        ("spare-a", SPARE_A),
        ("spare-b", SPARE_B),
    ] {
        let spares: Vec<Spare> = match buffer {
            SOURCE => Vec::new(),
            held => vec![plain_spare(held)],
        };
        rows.push((
            format!("buffer-{name}"),
            descriptor(&[at5(AttributeFormat::Short2, 4, buffer)], &spares, plain),
        ));
    }
    // The step axis.
    for (name, step, rate) in [
        ("constant", StepFunction::Constant, 0),
        ("grid-x", StepFunction::ThreadPositionInGridX, 1),
        ("grid-x-rate-3", StepFunction::ThreadPositionInGridX, 3),
        ("grid-y", StepFunction::ThreadPositionInGridY, 1),
        (
            "grid-x-indexed",
            StepFunction::ThreadPositionInGridXIndexed,
            1,
        ),
        (
            "grid-y-indexed",
            StepFunction::ThreadPositionInGridYIndexed,
            1,
        ),
    ] {
        rows.push((
            format!("step-{name}"),
            descriptor(
                &[at5(AttributeFormat::UShort2, 0, SPARE_A)],
                &[Spare {
                    buffer: SPARE_A,
                    stride: Some(ELEMENT),
                    step,
                    rate,
                }],
                plain,
            ),
        ));
    }
    // The stride axis.
    for (name, stride) in [
        ("fixed-64", Some(ELEMENT)),
        ("fixed-16", Some(16)),
        ("dynamic", None),
    ] {
        rows.push((
            format!("stride-{name}"),
            descriptor(
                &[at5(AttributeFormat::Char4, 0, SPARE_A)],
                &[Spare {
                    buffer: SPARE_A,
                    stride,
                    step: StepFunction::ThreadPositionInGridX,
                    rate: 1,
                }],
                plain,
            ),
        ));
    }
    // The index fields, which only an indexed step reads at all.
    for (name, index) in [
        ("index-type-uint16", (IndexType::UInt16, INDEX_A)),
        ("index-type-uint32", (IndexType::UInt32, INDEX_A)),
        ("index-buffer-3", (IndexType::UInt32, INDEX_B)),
    ] {
        rows.push((
            name.to_owned(),
            descriptor(
                &[at5(AttributeFormat::UInt, 0, SPARE_A)],
                &[Spare {
                    buffer: SPARE_A,
                    stride: Some(ELEMENT),
                    step: StepFunction::ThreadPositionInGridXIndexed,
                    rate: 1,
                }],
                index,
            ),
        ));
    }
    // The crossings.
    rows.push((
        "widest-format-at-the-furthest-offset".to_owned(),
        descriptor(
            &[at5(AttributeFormat::Int4, 48, SPARE_A)],
            &[plain_spare(SPARE_A)],
            plain,
        ),
    ));
    rows.push((
        "inside-the-read-buffer".to_owned(),
        descriptor(&[at5(AttributeFormat::Half2, 4, SOURCE)], &[], plain),
    ));
    rows.push((
        "a-constant-step-beside-a-y-step".to_owned(),
        descriptor(
            &[
                at5(AttributeFormat::UInt, 0, SPARE_A),
                Unread {
                    index: 6,
                    format: AttributeFormat::UInt4,
                    offset: 16,
                    buffer: SPARE_B,
                },
            ],
            &[
                Spare {
                    buffer: SPARE_A,
                    stride: Some(ELEMENT),
                    step: StepFunction::Constant,
                    rate: 0,
                },
                Spare {
                    buffer: SPARE_B,
                    stride: Some(ELEMENT),
                    step: StepFunction::ThreadPositionInGridY,
                    rate: 1,
                },
            ],
            plain,
        ),
    ));
    rows.push((
        "an-indexed-step-beside-a-dynamic-stride".to_owned(),
        descriptor(
            &[
                at5(AttributeFormat::UInt, 0, SPARE_A),
                Unread {
                    index: 6,
                    format: AttributeFormat::UInt,
                    offset: 0,
                    buffer: SPARE_B,
                },
            ],
            &[
                Spare {
                    buffer: SPARE_A,
                    stride: Some(ELEMENT),
                    step: StepFunction::ThreadPositionInGridXIndexed,
                    rate: 1,
                },
                Spare {
                    buffer: SPARE_B,
                    stride: None,
                    step: StepFunction::ThreadPositionInGridX,
                    rate: 1,
                },
            ],
            plain,
        ),
    ));
    rows.push((
        "three-unread-over-three-buffers".to_owned(),
        descriptor(
            &[
                at5(AttributeFormat::Short2, 4, SOURCE),
                Unread {
                    index: 6,
                    format: AttributeFormat::Int4,
                    offset: 0,
                    buffer: SPARE_A,
                },
                Unread {
                    index: 7,
                    format: AttributeFormat::UChar4Normalized,
                    offset: 12,
                    buffer: SPARE_B,
                },
            ],
            &[
                Spare {
                    buffer: SPARE_A,
                    stride: Some(ELEMENT),
                    step: StepFunction::ThreadPositionInGridX,
                    rate: 3,
                },
                Spare {
                    buffer: SPARE_B,
                    stride: Some(16),
                    step: StepFunction::Constant,
                    rate: 0,
                },
            ],
            plain,
        ),
    ));
    rows.push((
        "an-unset-attribute-beside-a-set-one".to_owned(),
        descriptor(
            &[
                at5(AttributeFormat::Invalid, 0, SOURCE),
                Unread {
                    index: 6,
                    format: AttributeFormat::UInt,
                    offset: 0,
                    buffer: SPARE_A,
                },
            ],
            &[plain_spare(SPARE_A)],
            plain,
        ),
    ));
    rows.push((
        "an-unset-attribute-on-a-zero-stride-layout".to_owned(),
        descriptor(
            &[at5(AttributeFormat::Invalid, 0, SPARE_A)],
            &[Spare {
                buffer: SPARE_A,
                stride: Some(0),
                step: StepFunction::ThreadPositionInGridX,
                rate: 1,
            }],
            plain,
        ),
    ));
    rows.push((
        "unread-at-attribute-15".to_owned(),
        descriptor(
            &[Unread {
                index: 15,
                format: AttributeFormat::UInt,
                offset: 0,
                buffer: SPARE_A,
            }],
            &[plain_spare(SPARE_A)],
            plain,
        ),
    ));
    rows
}

/// The lowering inputs of a row: the whole descriptor, as a creation states it before it learns
/// what the kernel reads.
fn inputs(d: &StageInputDescriptor) -> LoweringInputs {
    let mut inputs = LoweringInputs::new();
    inputs.stage_input = Some(d.clone());
    inputs
}

/// The lowering inputs a creation compiles the kernel of a row under: [`inputs`] narrowed to the
/// attributes the kernel reads (`Device::kernel_inputs_read`), through `entries`, so the sweep
/// decodes and specializes the one kernel once however many rows it reads.
fn read_inputs(
    device: &Device,
    kernel: &FunctionDescriptor,
    entries: &StageEntries,
    d: &StageInputDescriptor,
) -> LoweringInputs {
    device
        .kernel_inputs_read(kernel, &inputs(d), &Budget::standard(), entries)
        .expect("the kernel's read attributes")
}

/// The sweep's rows are all different descriptors, so nothing below is a comparison of a
/// descriptor with itself.
///
/// Without this a row dropped or mistyped into another's shape would make every count below
/// smaller for no reason but the duplicate.
#[test]
fn every_row_states_a_descriptor_of_its_own() {
    let rows = rows();
    let held: BTreeSet<String> = rows.iter().map(|(_, d)| format!("{d:?}")).collect();
    assert_eq!(
        held.len(),
        rows.len(),
        "{} rows state {} descriptors",
        rows.len(),
        held.len()
    );
    let names: BTreeSet<&str> = rows.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(names.len(), rows.len());
}

/// Every row compiles to the **same module, word for word**: an attribute the kernel does not read
/// reaches no instruction, over the whole sweep.
///
/// This is the evidence for what the key below leaves out, and it holds whether or not the key
/// leaves it out: the words are a property of the lowering, which reads one attribute at a time.
/// A row whose words differed would be an input that has to stay in the key.
#[test]
fn an_unread_attribute_changes_no_word_of_the_module() {
    let Some(bitcode) = fixture() else { return };
    let device = device();
    let kernel = kernel(&bitcode);
    let rows = rows();
    let mut baseline: Option<(String, Vec<u32>)> = None;
    for (name, d) in &rows {
        let inputs = inputs(d);
        let mut meter = Meter::new(&Budget::standard());
        let compiled = compile_stage(
            &kernel.function,
            &kernel.constants,
            (&kernel.linked, &kernel.libraries),
            device.profile(),
            InputsFrom::Given(&inputs),
            &mut meter,
        )
        .unwrap_or_else(|why| panic!("{name} compiles: {why:?}"));
        let words = compiled.artifact().module.words.clone();
        match &baseline {
            None => baseline = Some((name.clone(), words)),
            Some((first, held)) => assert_eq!(
                &words, held,
                "{name} compiled to a module the baseline {first} did not: an unread attribute \
                 reached an instruction, so it belongs in the compile key"
            ),
        }
    }
    let (_, held) = baseline.expect("a row");
    eprintln!(
        "{} rows, one module of {} SPIR-V words",
        rows.len(),
        held.len()
    );
}

/// The whole sweep is **one function compile key**, and so one translation: a kernel that reads
/// five attributes is compiled once however the sixth it never reads is stated.
///
/// This is the measurement of the fix. `Device::kernel_inputs_read` narrows a creation's lowering
/// inputs to the attributes the kernel's entry reads, and a creation keys its kernel and compiles
/// it under those, so the 36 rows ask for one
/// compiled function and the memo of compiled functions answers 35 of them. Before the narrowing
/// each row was a key of its own and each bought a translation of bytes the first already
/// produced; [`an_unread_attribute_changes_no_word_of_the_module`] is what says those were the
/// same bytes.
///
/// The **compute pipeline key stays one a row**, which is deliberate and is not the waste: it
/// states the descriptor as a creation reads it past the fetch (its layout agreement, which
/// refuses an attribute whose buffer has no layout read or not, and the buffers its reflection
/// lists for every set attribute), and a creation of a key the memo lacks links and relocates a
/// module it already has rather than translating one.
///
/// The counts are asserted at what they read, so neither a narrowing that went further nor one
/// that came back can land quietly.
#[test]
fn an_unread_attribute_keys_a_translation_of_its_own() {
    let Some(bitcode) = fixture() else { return };
    let device = device();
    let kernel = kernel(&bitcode);
    let entries = StageEntries::new();
    let rows = rows();
    let mut function_keys: BTreeSet<[u8; 32]> = BTreeSet::new();
    let mut whole_keys: BTreeSet<[u8; 32]> = BTreeSet::new();
    let mut pipeline_keys: BTreeSet<[u8; 32]> = BTreeSet::new();
    for (_, d) in &rows {
        let read = read_inputs(&device, &kernel, &entries, d);
        function_keys.insert(*device.function_compile_key(&kernel, &read).digest());
        whole_keys.insert(*device.function_compile_key(&kernel, &inputs(d)).digest());
        let mut descriptor = ComputePipelineDescriptor::new(StageSource::Function(kernel.clone()));
        descriptor.stage_input = Some(d.clone());
        pipeline_keys.insert(*device.compute_pipeline_key(&descriptor).digest());
    }
    let (specialized, shared) = entries.counts();
    eprintln!(
        "{} rows: {} function compile key(s) read, {} stated whole, {} compute pipeline key(s); \
         {specialized} entry specialized, {shared} shared",
        rows.len(),
        function_keys.len(),
        whole_keys.len(),
        pipeline_keys.len(),
    );
    assert_eq!(
        function_keys.len(),
        1,
        "the sweep's {} rows are {} function compile keys, so {} of them buy a translation of \
         bytes another produced",
        rows.len(),
        function_keys.len(),
        function_keys.len() - 1
    );
    assert_eq!(
        whole_keys.len(),
        rows.len(),
        "the descriptor stated whole is still {} keys: the narrowing is `kernel_inputs_read`'s \
         and not a change to what a key makes of the inputs it is given",
        rows.len()
    );
    assert_eq!(
        pipeline_keys.len(),
        rows.len(),
        "the compute pipeline key states the whole descriptor, as `compute_fields` writes it"
    );
    assert_eq!(
        (specialized, shared),
        (1, rows.len() as u64 - 1),
        "the sweep decodes and specializes the one kernel once and shares it {} times",
        rows.len() - 1
    );
}

/// Every row narrows to the read set alone, and the **canonical form** of every narrowed row is
/// one descriptor: the one key above is that, and not two descriptors a digest happened to agree
/// on.
///
/// The narrowed descriptors themselves are not all equal, and must not be: the narrowing keeps
/// what it keeps verbatim, the index type and index buffer index among it, so that every
/// `StageInFetch::of` of a read attribute answers what it answered under the whole descriptor.
/// Writing those at Metal's defaults where no kept layout steps through the index buffer is
/// `StageInputDescriptor::canonical`'s to do, and the key hashes the canonical form
/// (`pipeline::compile::key::canonical_inputs`), so that is what this compares.
#[test]
fn every_row_narrows_to_the_read_set_alone() {
    let Some(bitcode) = fixture() else { return };
    let device = device();
    let kernel = kernel(&bitcode);
    let entries = StageEntries::new();
    let rows = rows();
    let mut held: Option<(String, StageInputDescriptor)> = None;
    for (name, d) in &rows {
        let read = read_inputs(&device, &kernel, &entries, d);
        let narrowed = read.stage_input.clone().expect("a stage-input descriptor");
        assert_eq!(
            narrowed.attributes.keys().copied().collect::<Vec<u32>>(),
            vec![0, 1, 2, 3, 4],
            "{name} narrows to attributes the kernel does not read, or drops one it does"
        );
        assert_eq!(
            narrowed.layouts.keys().copied().collect::<Vec<u32>>(),
            vec![SOURCE],
            "{name} keeps a layout no read attribute names, or drops the one they read"
        );
        let canonical = narrowed.canonical();
        match &held {
            None => held = Some((name.clone(), canonical)),
            Some((first, first_canonical)) => assert_eq!(
                &canonical, first_canonical,
                "{name} narrows to a descriptor the baseline {first} does not fetch alike"
            ),
        }
    }
}

/// The module a row compiles to under the **narrowed** inputs is the module it compiles to under
/// the whole descriptor, word for word, over the whole sweep.
///
/// This is the exactness of the narrowing itself, and the other half of
/// [`an_unread_attribute_changes_no_word_of_the_module`]: that one says the 36 descriptors compile
/// alike, and this one says compiling under the attributes the kernel reads compiles to the same
/// thing as compiling under all of them. A row whose words moved would be an input the narrowing
/// must not drop.
#[test]
fn narrowing_the_inputs_changes_no_word_of_the_module() {
    let Some(bitcode) = fixture() else { return };
    let device = device();
    let kernel = kernel(&bitcode);
    let entries = StageEntries::new();
    let compile = |inputs: &LoweringInputs, name: &str| -> Vec<u32> {
        let mut meter = Meter::new(&Budget::standard());
        compile_stage(
            &kernel.function,
            &kernel.constants,
            (&kernel.linked, &kernel.libraries),
            device.profile(),
            InputsFrom::Given(inputs),
            &mut meter,
        )
        .unwrap_or_else(|why| panic!("{name} compiles: {why:?}"))
        .artifact()
        .module
        .words
        .clone()
    };
    for (name, d) in &rows() {
        let whole = compile(&inputs(d), name);
        let read = compile(&read_inputs(&device, &kernel, &entries, d), name);
        assert_eq!(
            read, whole,
            "{name} compiled to another module under the attributes the kernel reads: an \
             attribute the narrowing drops reaches an instruction"
        );
    }
}

/// Every row's pipeline is **created from the kernel compiled under the narrowed inputs and the
/// descriptor as given**, which is what a creation does, and none of them refuses.
///
/// This is the check a count cannot make. The narrowing takes inputs out of the kernel's compile,
/// and what a creation reads of the descriptor past the fetch still reads it whole: the pipeline
/// compares every fetch its artifact recorded against the descriptor it was given
/// (`check_stage_in`), and holds the descriptor's layouts to Metal's agreement
/// (`check_layouts`). A narrowing that moved a read attribute's fetch, or that let the
/// descriptor's own refusals go, would not change a key count — it would surface here, as a
/// refusal of a descriptor Metal accepts or an acceptance of one it stops.
///
/// Each row's pipeline also keeps the key `Device::compute_pipeline_key` states for its
/// descriptor, so a consumer that looks one up before creating it finds what the creation made.
#[test]
fn a_narrowed_kernel_still_makes_every_row_s_pipeline() {
    let Some(bitcode) = fixture() else { return };
    let device = device();
    let kernel = kernel(&bitcode);
    let entries = StageEntries::new();
    let budget = Budget::standard();
    for (name, d) in &rows() {
        let read = read_inputs(&device, &kernel, &entries, d);
        let compiled = device
            .compile_function(&kernel, &read, &budget)
            .unwrap_or_else(|why| panic!("{name}'s kernel compiles: {why:?}"));
        let mut descriptor =
            ComputePipelineDescriptor::new(StageSource::Compiled(compiled.clone()));
        descriptor.stage_input = Some(d.clone());
        let pipeline = device
            .compile_compute_pipeline(&descriptor, &budget)
            .unwrap_or_else(|why| {
                panic!(
                    "{name}'s pipeline links the kernel compiled under the attributes it reads: \
                     {why:?}"
                )
            });
        assert_eq!(
            &pipeline.key,
            &device.compute_pipeline_key(&descriptor),
            "{name}'s pipeline keeps a key its descriptor does not state"
        );
        assert_eq!(
            compiled.key(),
            &device.function_compile_key(&kernel, &read),
            "{name}'s kernel keeps a key the inputs it compiled under do not state"
        );
    }
}

/// An attribute the kernel **does** read keys a translation of its own: moving one of the five, or
/// the layout they read, gives a key apart from the baseline's and from the others'.
///
/// This holds with the key as it is and with the key narrowed, so it is not a control for the
/// narrowing — it is the bound the narrowing must not cross. A narrowing that went one attribute
/// too far would still pass the count above and fail here. Two of the five are moved — the
/// `float4`'s format and the `int4`'s offset — so neither the first attribute nor the last is the
/// only one checked.
#[test]
fn a_read_attribute_is_part_of_the_key() {
    let Some(bitcode) = fixture() else { return };
    let device = device();
    let kernel = kernel(&bitcode);
    let entries = StageEntries::new();
    let plain = (IndexType::UInt32, INDEX_A);
    let base = descriptor(&[], &[], plain);
    let key = |d: &StageInputDescriptor| {
        *device
            .function_compile_key(&kernel, &read_inputs(&device, &kernel, &entries, d))
            .digest()
    };
    let mut moved_format = base.clone();
    moved_format.attributes.insert(
        0,
        StageInputAttributeDescriptor::new(AttributeFormat::Half4, 0, SOURCE),
    );
    let mut moved_offset = base.clone();
    moved_offset.attributes.insert(
        4,
        StageInputAttributeDescriptor::new(AttributeFormat::Int4, 16, SOURCE),
    );
    let mut moved_layout = base.clone();
    let mut layout = StageInputLayoutDescriptor::new(Some(ELEMENT));
    layout.step_function = StepFunction::ThreadPositionInGridY;
    layout.step_rate = 1;
    moved_layout.layouts.insert(SOURCE, layout);
    let keys: BTreeSet<[u8; 32]> = [&base, &moved_format, &moved_offset, &moved_layout]
        .into_iter()
        .map(key)
        .collect();
    assert_eq!(
        keys.len(),
        4,
        "the read attributes' format, offset and layout must each key a translation of its own"
    );
}
