//! How many distinct vertex modules one vertex function turns into over a format sweep, and which
//! input accounts for each.
//!
//! A render pipeline creation translates its vertex function under the vertex-fetch projection the
//! descriptor and the device decide (`pipeline::link::vertex`), and the runtime makes one
//! `VkShaderModule` per distinct SPIR-V. So a sweep that creates one pipeline per
//! `MTLVertexFormat` pays for as many translations as it has distinct projections and as many
//! driver shader conversions as it has distinct modules. This module measures both over the
//! conformance suite's own sweeps, through the owned fixture they use (`dsx_int_attributes`, whose
//! `int_attribute_to_float4` and `int_attribute_to_half4` read attribute 0 as a `float4` and as a
//! `half4`), and attributes every module to the input that makes it:
//!
//! - **fixed-function input, one module, one translation.** The whole format is `VertexAttribute`
//!   state and no instruction of the module names it, so every format the device fetches in fixed
//!   function shares one projection and one module. What the device's conversion of a format may
//!   answer otherwise than Metal's fetch -- a normalized format's quotient, a narrow float's NaN
//!   read as `float` -- is noted where the render pipeline links, on its vertex stage
//!   (`PrecisionNote::DeviceVertexConversion`, `PrecisionNote::DeviceNanEncoding`), or, where the
//!   device is measured to convert a normalized format as Metal does, required of the device in
//!   its place (`VkProperty::MetalNormalizedVertexFetch`), so neither keys a translation
//!   (`a_pipeline_notes_what_the_device_converts_from_one_translation`).
//! - **one module per pulled format.** A format the fixed-function path cannot fetch is decoded in
//!   the shader, and each decode is its own code. Necessary: no format of the 45-format sweep is
//!   pulled on the suite's venue.
//! - **the step function: nothing, where the device fetches it.** Metal's constant step is a
//!   Vulkan binding of divisor 0 and stride 0 (`metal::vertex::attribute_fetch`,
//!   `link::vertex::binding_stride`), which is `VertexAttribute` and `VertexBinding` state and
//!   names no instruction, so a constant-step sweep costs exactly what the per-vertex sweep costs.
//!   Where that binding is not available -- the portability subset forbids a stride of 0, which is
//!   what makes the fetch independent of the base instance -- every format is pulled instead and
//!   the sweep is one module a creation: 45 of 45 on `moltenvk-m3`, 1 on a native driver.
//!
//! Counts, not times: a count is the same on a loaded host as on a quiet one. No GPU is used, and
//! `llvm-as` assembles the fixture, so a missing tool skips visibly.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::budget::SPIRV_WORDS;
use crate::device::{DevicePolicy, DeviceProfile};
use crate::library::{FunctionArtifact, FunctionConstantValues};
use crate::metal::{VertexFormat, VertexStepFunction};
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::compile::{compile_bitcode, EntryRef};
use crate::pipeline::descriptor::{
    VertexAttributeDescriptor, VertexBufferLayoutDescriptor, VertexDescriptor,
};
use crate::pipeline::link::vertex::vertex_fetch_projection;
use crate::pipeline::lower_inputs::{LoweringInputs, VertexFetch, VertexFetchProjection};
use crate::pipeline::stage_io::IoNumeric;
use crate::pipeline::testing::{engine_dir, llvm_assemble, meter, profile, AssembleError};

/// The venue the suite's recordings of these cases were taken on.
const PROFILE: &str = "moltenvk-m3";

/// Bytes the sweeps give each format in a record, which is also each attribute's offset step
/// (`metal2vulkan-conformance::attribute_coding::SLOT`).
const SLOT: u32 = 16;

/// The entry whose `[[attribute(0)]]` is a `float4`.
const TO_FLOAT4: &str = "int_attribute_to_float4";

/// The entry whose `[[attribute(0)]]` is a `half4`.
const TO_HALF4: &str = "int_attribute_to_half4";

fn fixture() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join("dsx_int_attributes.ll")) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => {
            panic!("llvm-as rejected dsx_int_attributes.ll: {why}")
        }
    }
}

fn prof() -> DeviceProfile {
    profile(PROFILE, DevicePolicy::default()).expect("committed profile")
}

/// Every format of the SDK table a float-class attribute can read, as `cases::vertex_formats`
/// selects them: all but `Invalid` and the eight 32-bit integer formats, which reach the shader as
/// integers and need an integer-declared attribute.
fn float_readable() -> Vec<VertexFormat> {
    VertexFormat::ALL
        .iter()
        .copied()
        .filter(|f| {
            !matches!(
                f,
                VertexFormat::Invalid
                    | VertexFormat::Int
                    | VertexFormat::Int2
                    | VertexFormat::Int3
                    | VertexFormat::Int4
                    | VertexFormat::UInt
                    | VertexFormat::UInt2
                    | VertexFormat::UInt3
                    | VertexFormat::UInt4
            )
        })
        .collect()
}

/// The sweeps' descriptor for `format` at slot `at`: attribute 0 of that format at byte
/// `SLOT · at` of buffer 1, whose one layout holds a record of `stride` bytes under `step`.
fn desc(
    format: VertexFormat,
    at: usize,
    stride: u32,
    step: VertexStepFunction,
) -> VertexDescriptor {
    let mut d = VertexDescriptor::new();
    d.attributes.insert(
        0,
        VertexAttributeDescriptor::new(format, SLOT * at as u32, 1),
    );
    let mut l = VertexBufferLayoutDescriptor::new(Some(stride));
    l.step_function = step;
    l.step_rate = 1;
    d.layouts.insert(1, l);
    d
}

/// The projection of attribute 0 for `format` at slot `at` under `step`.
fn projection(
    p: &DeviceProfile,
    format: VertexFormat,
    at: usize,
    stride: u32,
    step: VertexStepFunction,
) -> VertexFetchProjection {
    let attributes = BTreeMap::from([(0, IoNumeric::Float)]);
    vertex_fetch_projection(Some(&desc(format, at, stride, step)), &attributes, p)
        .expect("the sweeps' descriptors are valid")
}

/// The artifact of `entry` compiled under `proj`.
fn artifact(
    bc: &[u8],
    p: &DeviceProfile,
    entry: &str,
    proj: &VertexFetchProjection,
) -> Arc<FunctionArtifact> {
    let mut inputs = LoweringInputs::new();
    inputs.vertex_fetch = Some(proj.clone());
    compile_bitcode(
        bc,
        EntryRef::Name(entry),
        &FunctionConstantValues::defaults(),
        p,
        &inputs,
        &mut meter(),
        SPIRV_WORDS,
    )
    .expect("the fixture compiles under every projection of the sweeps")
}

/// The final SPIR-V words of `entry` compiled under `proj`.
fn words(bc: &[u8], p: &DeviceProfile, entry: &str, proj: &VertexFetchProjection) -> Vec<u32> {
    artifact(bc, p, entry, proj).module.words.to_vec()
}

/// One sweep's counts: its creations, the distinct projections they compile under (so the
/// translations), the distinct modules (so the driver's shader conversions), and the formats the
/// shader pulls.
struct Counted {
    creations: usize,
    projections: usize,
    modules: usize,
    pulled: Vec<&'static str>,
}

fn count(bc: &[u8], p: &DeviceProfile, entry: &str, step: VertexStepFunction) -> Counted {
    let formats = float_readable();
    let stride = SLOT * formats.len() as u32;
    let mut projections = BTreeSet::new();
    let mut modules = BTreeSet::new();
    let mut pulled = Vec::new();
    for (at, &format) in formats.iter().enumerate() {
        let proj = projection(p, format, at, stride, step);
        if matches!(proj.attributes[&0], VertexFetch::Pulled { .. }) {
            pulled.push(format.name());
        }
        modules.insert(words(bc, p, entry, &proj));
        projections.insert(proj);
    }
    Counted {
        creations: formats.len(),
        projections: projections.len(),
        modules: modules.len(),
        pulled,
    }
}

/// `vertex-format-float4`'s own sweep under the default: the device fetches every one of the 45
/// formats in fixed function, and the whole sweep is **one** module of one translation.
#[test]
fn a_float_read_compiles_one_module_for_the_whole_sweep() {
    let Some(bc) = fixture() else { return };
    let got = count(&bc, &prof(), TO_FLOAT4, VertexStepFunction::PerVertex);
    assert_eq!(got.creations, 45, "the swept formats");
    assert_eq!(
        got.pulled,
        Vec::<&str>::new(),
        "no format of the sweep is pulled on {PROFILE} under the default"
    );
    assert_eq!(
        (got.projections, got.modules),
        (1, 1),
        "projections and modules of {} creations",
        got.creations
    );
}

/// The same sweep read as a `half4`: one module of one translation too.
#[test]
fn a_half_read_compiles_one_module_for_the_whole_sweep() {
    let Some(bc) = fixture() else { return };
    let got = count(&bc, &prof(), TO_HALF4, VertexStepFunction::PerVertex);
    assert_eq!(got.creations, 45, "the swept formats");
    assert_eq!(
        (got.projections, got.modules),
        (1, 1),
        "projections and modules of {} creations",
        got.creations
    );
}

/// A render pipeline notes what the device's fixed-function vertex input converts, from its own
/// vertex descriptor, on its vertex stage, while one translation of the function serves every
/// format the device fetches. Over `vertex-format-float4`'s and `-half4`'s sweeps, a normalized
/// format's pipeline records `PrecisionNote::DeviceVertexConversion` whatever width the function
/// reads it at on lavapipe and RADV, which do not take `DriverDeviation::MetalNormalizedVertexFetch`;
/// on `moltenvk-m3`, which does, it records none and requires
/// `VkProperty::MetalNormalizedVertexFetch`, but for a 16-bit format read as `half`, whose `float`
/// rounds to `half` twice and keeps the note. A narrow float's records
/// `PrecisionNote::DeviceNanEncoding` where the function reads it as a 32-bit `float` (a `half`
/// read keeps the code's own bits), and every other fixed-function format's records neither and
/// requires no such property. Every such creation of one entry keys its vertex function alike
/// (`RenderPipelineKeys::vertex`), so a runtime translates it once, and its vertex stage is one
/// module. The function compiled alone records neither note and requires no such property: its
/// module is every format's.
#[test]
fn a_pipeline_notes_what_the_device_converts_from_one_translation() {
    use crate::budget::Budget;
    use crate::device::requirements::VkProperty;
    use crate::device::Device;
    use crate::error::DriverDeviation;
    use crate::library::Library;
    use crate::lower::stage_io::fetch::{encoding, Encoding};
    use crate::pipeline::testing::description;
    use crate::pipeline::{FunctionDescriptor, RenderPipelineDescriptor, StageSource};
    let Some(bc) = fixture() else { return };
    let library = Library::from_air(&bc).expect("library");
    let formats = float_readable();
    let stride = SLOT * formats.len() as u32;
    let fetch = VkProperty::MetalNormalizedVertexFetch;
    for stem in [PROFILE, "lavapipe", "amd-780m-radv"] {
        let dev = Device::new(
            description(stem).expect("committed description"),
            DevicePolicy::default(),
        )
        .expect("device");
        // MoltenVK on the M3 Max's identity is measured to convert every normalized code as
        // Metal does; lavapipe and RADV are not.
        let metal = dev
            .profile()
            .takes_driver_deviation(DriverDeviation::MetalNormalizedVertexFetch);
        assert_eq!(metal, stem == PROFILE, "{stem}");
        for entry in [TO_FLOAT4, TO_HALF4] {
            let function = FunctionDescriptor::new(library.function(entry).expect("entry"));
            let mut keys = BTreeSet::new();
            let mut modules = BTreeSet::new();
            let (mut converted, mut relied, mut nans, mut fixed) = (0, 0, 0, 0);
            for (at, &format) in formats.iter().enumerate() {
                let what = format!("{stem} {entry} {}", format.name());
                let mut d = RenderPipelineDescriptor::new(StageSource::Function(function.clone()));
                d.rasterization_enabled = false;
                d.vertex_descriptor = Some(desc(format, at, stride, VertexStepFunction::PerVertex));
                let pipeline = dev
                    .compile_render_pipeline(&d, &Budget::standard())
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                let notes = &pipeline.stages[0].precision_notes;
                let requires = pipeline.requirements.properties.contains(&fetch);
                if !pipeline.vertex_input.pulled.is_empty() {
                    // A format the device cannot fetch: the module's decode is its own.
                    assert!(
                        !notes.contains(&PrecisionNote::DeviceVertexConversion) && !requires,
                        "{what}: a pulled attribute's decode is not the device's: {notes:?}"
                    );
                    continue;
                }
                fixed += 1;
                let e = encoding(format).expect("encoding");
                // A 16-bit code's `float` quotient rounded to `half` again is not Metal's `half`
                // at every code, however the device converts it.
                let twice = entry == TO_HALF4
                    && matches!(
                        e,
                        Encoding::Int {
                            bits: 16,
                            normalized: true,
                            ..
                        }
                    );
                let measured = e.is_normalized() && metal && !twice;
                let conversion = e.is_normalized() && !measured;
                let nan = e.widens_narrow_float() && entry == TO_FLOAT4;
                assert_eq!(
                    (
                        notes.contains(&PrecisionNote::DeviceVertexConversion),
                        notes.contains(&PrecisionNote::DeviceNanEncoding)
                    ),
                    (conversion, nan),
                    "{what}: {notes:?}"
                );
                assert_eq!(
                    requires, measured,
                    "{what}: the pipeline requires the measured conversion exactly where it \
                     records no note for it: {:?}",
                    pipeline.requirements.properties
                );
                converted += usize::from(conversion);
                relied += usize::from(measured);
                nans += usize::from(nan);
                keys.insert(
                    dev.render_pipeline_key(&d, &Budget::standard())
                        .unwrap_or_else(|e| panic!("{what}: {e}"))
                        .vertex,
                );
                modules.insert(pipeline.stages[0].spirv.to_vec());
            }
            assert_eq!(
                (keys.len(), modules.len()),
                (1, 1),
                "{stem} {entry}: translations and modules of {fixed} fixed-function creations"
            );
            // Every normalized format, and every narrow float read as `float`, is fetched in
            // fixed function on the three devices: on `moltenvk-m3` the 19 normalized formats
            // rely on the measured conversion but the eight 16-bit ones read as `half`, which
            // note it, and elsewhere all 19 note it.
            let noted = match (metal, entry) {
                (false, _) => 19,
                (true, TO_FLOAT4) => 0,
                (true, _) => 8,
            };
            assert_eq!(
                (converted, relied, nans),
                (noted, 19 - noted, if entry == TO_FLOAT4 { 5 } else { 0 }),
                "{stem} {entry}: the creations noting each, and relying on the measured conversion"
            );
            // The function compiled under the shared projection notes neither.
            let p = dev.profile();
            let proj = projection(
                p,
                VertexFormat::UChar4Normalized,
                0,
                stride,
                VertexStepFunction::PerVertex,
            );
            let a = artifact(&bc, p, entry, &proj);
            assert!(
                !a.precision.contains(&PrecisionNote::DeviceVertexConversion)
                    && !a.precision.contains(&PrecisionNote::DeviceNanEncoding)
                    && !a.requirements.properties.contains(&fetch),
                "{stem} {entry}: {:?}",
                a.precision
            );
        }
    }
}

/// `integer-vertex-attributes`' own sweep: sixteen unnormalized 8- and 16-bit integer formats,
/// two entry points and two step functions. Metal's constant step is a fetch no fixed-function
/// vertex input expresses, so each of its 32 creations decodes its own format in the shader, while
/// all sixteen formats of a per-vertex creation share one module an entry. Two source functions,
/// 34 modules.
#[test]
fn the_integer_attribute_sweep_compiles_thirty_four_modules_of_two_functions() {
    let Some(bc) = fixture() else { return };
    let p = prof();
    const FORMATS: [VertexFormat; 16] = [
        VertexFormat::UChar,
        VertexFormat::UChar2,
        VertexFormat::UChar3,
        VertexFormat::UChar4,
        VertexFormat::Char,
        VertexFormat::Char2,
        VertexFormat::Char3,
        VertexFormat::Char4,
        VertexFormat::UShort,
        VertexFormat::UShort2,
        VertexFormat::UShort3,
        VertexFormat::UShort4,
        VertexFormat::Short,
        VertexFormat::Short2,
        VertexFormat::Short3,
        VertexFormat::Short4,
    ];
    let stride = SLOT * FORMATS.len() as u32;
    let mut projections = BTreeSet::new();
    let mut modules = BTreeSet::new();
    let mut creations = 0;
    let mut per_vertex_pulled = 0;
    for entry in [TO_FLOAT4, TO_HALF4] {
        for step in [VertexStepFunction::PerVertex, VertexStepFunction::Constant] {
            for (at, &format) in FORMATS.iter().enumerate() {
                creations += 1;
                let proj = projection(&p, format, at, stride, step);
                if step == VertexStepFunction::PerVertex
                    && matches!(proj.attributes[&0], VertexFetch::Pulled { .. })
                {
                    per_vertex_pulled += 1;
                }
                modules.insert(words(&bc, &p, entry, &proj));
                projections.insert((entry, proj));
            }
        }
    }
    assert_eq!(creations, 64, "the sweep's pipelines");
    assert_eq!(
        per_vertex_pulled, 0,
        "every unnormalized integer format is fetched in fixed function per vertex"
    );
    // One module an entry for all sixteen per-vertex formats, and one per constant-step format.
    assert_eq!(
        (projections.len(), modules.len()),
        (2 + 2 * FORMATS.len(), 2 + 2 * FORMATS.len()),
        "projections and modules of {creations} creations"
    );
}

/// Native drivers where a constant-step binding is available: the two divisor features,
/// `supportsNonZeroFirstInstance`, and no portability subset to forbid a stride of 0.
const NATIVE: [&str; 4] = [
    "intel-arl-anv",
    "nvidia-rtx5080-laptop",
    "amd-780m-radv",
    "intel-irisxe-anv",
];

/// `vertex-format-constant-step`'s own sweep on the venue its recordings were taken on: the
/// portability subset's `minVertexInputBindingStrideAlignment` forbids the stride of 0 a
/// constant-step binding needs, so every one of the 45 formats is decoded in the shader and the
/// sweep is **45** modules -- one a creation, the whole explosion. The pulled fetch carries the
/// format, the offset and the layout, so no two of the 45 share a projection either.
#[test]
fn the_constant_step_sweep_pulls_every_format_on_a_portability_device() {
    let Some(bc) = fixture() else { return };
    let got = count(&bc, &prof(), TO_FLOAT4, VertexStepFunction::Constant);
    assert_eq!(got.creations, 45, "the swept formats");
    assert_eq!(
        got.pulled.len(),
        45,
        "every format of the sweep is pulled on {PROFILE}"
    );
    assert_eq!(
        (got.projections, got.modules),
        (45, 45),
        "projections and modules of {} creations",
        got.creations
    );
}

/// The same sweep's pulled decodes note the device's product: a normalized format's decode is the
/// code times its maximum's reciprocal, and the function records
/// `PrecisionNote::DeviceVertexConversion` wherever that misses Metal's fetch at some code, on
/// `moltenvk-m3` too, whose fixed-function conversion is measured to be Metal's: read as a
/// `float4`, 18 of the 19 normalized formats (all but `Int1010102Normalized`, whose 10-bit fields
/// Metal computes as the product), and read as a `half4`, the eight 16-bit ones. No other format
/// records it.
#[test]
fn a_pulled_normalized_decode_notes_the_device_s_product() {
    let Some(bc) = fixture() else { return };
    let p = prof();
    let formats = float_readable();
    let stride = SLOT * formats.len() as u32;
    for (entry, want) in [(TO_FLOAT4, 18), (TO_HALF4, 8)] {
        let mut noted = Vec::new();
        for (at, &format) in formats.iter().enumerate() {
            let proj = projection(&p, format, at, stride, VertexStepFunction::Constant);
            assert!(
                matches!(proj.attributes[&0], VertexFetch::Pulled { .. }),
                "{format:?} is pulled on {PROFILE}"
            );
            let a = artifact(&bc, &p, entry, &proj);
            if a.precision.contains(&PrecisionNote::DeviceVertexConversion) {
                noted.push(format);
            }
        }
        assert_eq!(noted.len(), want, "{entry}: {noted:?}");
        let normalized = |f: &VertexFormat| {
            crate::lower::stage_io::fetch::encoding(*f).is_ok_and(|e| e.is_normalized())
        };
        assert!(noted.iter().all(normalized), "{entry}: {noted:?}");
        assert!(
            !noted.contains(&VertexFormat::Int1010102Normalized),
            "{entry}"
        );
    }
}

/// The same sweep where a constant-step binding is available: fixed-function vertex input fetches
/// the formats the per-vertex sweep fetches, so the constant step costs exactly what the
/// per-vertex step costs -- **one** module of one translation over the 45 creations, for a `float`
/// read and a `half` one alike. 45 modules to 1. A format the device does not accept as a vertex buffer
/// is pulled under either step and is named as such, not as a step the step function costs
/// (`FloatRGB9E5` on the Intel profiles).
#[test]
fn a_constant_step_binding_compiles_one_module_for_the_whole_sweep() {
    let Some(bc) = fixture() else { return };
    for stem in NATIVE {
        let p = profile(stem, DevicePolicy::default()).expect("committed profile");
        let got = count(&bc, &p, TO_FLOAT4, VertexStepFunction::Constant);
        let per_vertex = count(&bc, &p, TO_FLOAT4, VertexStepFunction::PerVertex);
        assert_eq!(got.creations, 45, "{stem}: the swept formats");
        assert_eq!(
            got.pulled, per_vertex.pulled,
            "{stem}: the constant step pulls what the per-vertex step pulls and no more"
        );
        assert_eq!(
            (got.projections, got.modules),
            (per_vertex.projections, per_vertex.modules),
            "{stem}: the constant-step sweep costs what the per-vertex sweep costs"
        );
        let half = count(&bc, &p, TO_HALF4, VertexStepFunction::Constant);
        let half_per_vertex = count(&bc, &p, TO_HALF4, VertexStepFunction::PerVertex);
        assert_eq!(
            (half.projections, half.modules),
            (half_per_vertex.projections, half_per_vertex.modules),
            "{stem}: the same for a `half` read"
        );
        // The absolute figure where the device accepts every swept format as a vertex buffer.
        if got.pulled.is_empty() {
            assert_eq!(
                (got.projections, got.modules, half.projections, half.modules),
                (1, 1, 1, 1),
                "{stem}: projections and modules of {} creations",
                got.creations
            );
        } else {
            assert_eq!(
                got.pulled,
                vec!["MTLVertexFormatFloatRGB9E5"],
                "{stem}: the formats the device does not fetch as a vertex buffer"
            );
            assert_eq!(
                (got.projections, got.modules, half.projections, half.modules),
                (2, 2, 2, 2),
                "{stem}: projections and modules of {} creations",
                got.creations
            );
        }
    }
}

/// `integer-vertex-attributes`' sweep where a constant-step binding is available: the step
/// function is not key data of a fixed-function fetch, so all 64 creations -- 16 formats, 2 entry points, 2 step functions -- share
/// **two** modules, one an entry point. 34 modules to 2.
#[test]
fn the_integer_attribute_sweep_compiles_two_modules_where_the_constant_step_is_fixed_function() {
    let Some(bc) = fixture() else { return };
    const FORMATS: [VertexFormat; 16] = [
        VertexFormat::UChar,
        VertexFormat::UChar2,
        VertexFormat::UChar3,
        VertexFormat::UChar4,
        VertexFormat::Char,
        VertexFormat::Char2,
        VertexFormat::Char3,
        VertexFormat::Char4,
        VertexFormat::UShort,
        VertexFormat::UShort2,
        VertexFormat::UShort3,
        VertexFormat::UShort4,
        VertexFormat::Short,
        VertexFormat::Short2,
        VertexFormat::Short3,
        VertexFormat::Short4,
    ];
    let stride = SLOT * FORMATS.len() as u32;
    for stem in NATIVE {
        let p = profile(stem, DevicePolicy::default()).expect("committed profile");
        let mut projections = BTreeSet::new();
        let mut modules = BTreeSet::new();
        let mut creations = 0;
        let mut pulled = 0;
        for entry in [TO_FLOAT4, TO_HALF4] {
            for step in [VertexStepFunction::PerVertex, VertexStepFunction::Constant] {
                for (at, &format) in FORMATS.iter().enumerate() {
                    creations += 1;
                    let proj = projection(&p, format, at, stride, step);
                    if matches!(proj.attributes[&0], VertexFetch::Pulled { .. }) {
                        pulled += 1;
                    }
                    modules.insert(words(&bc, &p, entry, &proj));
                    projections.insert((entry, proj));
                }
            }
        }
        assert_eq!(creations, 64, "{stem}: the sweep's pipelines");
        assert_eq!(
            pulled, 0,
            "{stem}: every integer format is fetched in fixed function under either step"
        );
        assert_eq!(
            (projections.len(), modules.len()),
            (2, 2),
            "{stem}: projections and modules of {creations} creations"
        );
    }
}

/// What makes the constant-step binding exact, stated as the fetch it is: instance rate with a
/// divisor of 0, which reads element `firstInstance` at the binding's stride, on a binding link
/// emits with a stride of 0 -- so element 0 for every base instance, as Metal's constant step
/// fetches the layout's first element for every base instance. Each of the four conditions
/// withheld on its own puts the fetch back in the shader.
#[test]
fn a_constant_step_is_a_zero_divisor_only_where_a_zero_stride_is_legal() {
    use crate::device::requirements::VkFeature;
    use crate::metal::vertex::{attribute_fetch, AttributeFetch};
    use crate::pipeline::testing::description;
    let format = VertexFormat::Float4;
    let fetch = |p: &DeviceProfile| attribute_fetch(format, VertexStepFunction::Constant, 0, p);
    let zero_divisor = AttributeFetch::FixedFunction {
        format: crate::vk::Format::R32G32B32A32_SFLOAT,
        input_rate: crate::vk::VertexInputRate::INSTANCE,
        divisor: 0,
    };
    let derive = |desc: crate::device::DeviceDescription| {
        crate::device::Device::new(desc, DevicePolicy::default())
            .expect("the description derives")
            .profile()
            .clone()
    };
    for stem in NATIVE {
        let desc = description(stem).expect("committed description");
        assert_eq!(fetch(&derive(desc.clone())), Ok(zero_divisor), "{stem}");
        for withheld in [
            VkFeature::VertexAttributeInstanceRateZeroDivisor,
            VkFeature::VertexAttributeInstanceRateDivisor,
        ] {
            let mut d = desc.clone();
            d.features.remove(&withheld);
            assert_eq!(
                fetch(&derive(d)),
                Ok(AttributeFetch::Pulled),
                "{stem} without {withheld:?}"
            );
        }
        let mut d = desc.clone();
        d.limits.supports_non_zero_first_instance = false;
        assert_eq!(
            fetch(&derive(d)),
            Ok(AttributeFetch::Pulled),
            "{stem} without supportsNonZeroFirstInstance"
        );
        let mut d = desc.clone();
        d.limits.min_vertex_input_binding_stride_alignment = Some(1);
        assert_eq!(
            fetch(&derive(d)),
            Ok(AttributeFetch::Pulled),
            "{stem} under the portability subset's stride rule"
        );
    }
    // The venue the recordings come from carries that rule, and reports the features.
    let m3 = prof();
    assert!(
        m3.fields().fragment.vertex_attribute_zero_divisor
            && m3.fields().limits.supports_non_zero_first_instance,
        "{PROFILE} reports the zero divisor and a nonzero first instance"
    );
    assert_eq!(
        m3.fields().limits.min_vertex_input_binding_stride_alignment,
        Some(1),
        "{PROFILE} is a portability device whose vertex stride must be at least 1"
    );
    assert_eq!(fetch(&m3), Ok(AttributeFetch::Pulled), "{PROFILE}");
}
