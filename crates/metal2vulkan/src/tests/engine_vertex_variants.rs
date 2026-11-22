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
//!   (`PrecisionNote::DeviceVertexConversion`, `PrecisionNote::DeviceNanEncoding`), so no note
//!   keys a translation (`a_pipeline_notes_what_the_device_converts_from_one_translation`).
//! - **one module per pulled format.** Outside the identity domains every attribute is decoded in
//!   the shader, and each decode is its own code: the 45-format sweep is 45 modules. Under them a
//!   format the fixed-function path cannot fetch is refused: of the 45, `FloatRGB9E5`, on every
//!   device (`metal::vertex::attribute_fetch`).
//! - **the step function: nothing, where the device fetches it.** Under the identity domains
//!   Metal's constant step is a Vulkan vertex-rate binding of stride 0
//!   (`metal::vertex::attribute_fetch`, `link::vertex::binding_stride`), which is
//!   `VertexAttribute` and `VertexBinding` state and names no instruction, so a constant-step sweep
//!   of the integer formats costs exactly what the per-vertex sweep costs. Outside them every
//!   format is pulled instead and the sweep is one module a creation: 45 of 45 on `moltenvk-m3`.
//!
//! Counts, not times: a count is the same on a loaded host as on a quiet one. No GPU is used, and
//! `llvm-as` assembles the fixture, so a missing tool skips visibly.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::budget::SPIRV_WORDS;
use crate::device::{DevicePolicy, DeviceProfile};
use crate::error::{Error, Feature};
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
    profile(PROFILE, DevicePolicy::RESIDENT_TABLES).expect("committed profile")
}

/// The venue's profile under the identity domains, the runtime's.
fn identity() -> DeviceProfile {
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

/// The projection of attribute 0 for `format` at slot `at` under `step`, or its refusal.
fn try_projection(
    p: &DeviceProfile,
    format: VertexFormat,
    at: usize,
    stride: u32,
    step: VertexStepFunction,
) -> Result<VertexFetchProjection, Error> {
    let attributes = BTreeMap::from([(0, IoNumeric::Float)]);
    vertex_fetch_projection(Some(&desc(format, at, stride, step)), &attributes, p)
}

/// The projection of attribute 0 for `format` at slot `at` under `step`.
fn projection(
    p: &DeviceProfile,
    format: VertexFormat,
    at: usize,
    stride: u32,
    step: VertexStepFunction,
) -> VertexFetchProjection {
    try_projection(p, format, at, stride, step).expect("the sweeps' descriptors are valid")
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
/// translations), the distinct modules (so the driver's shader conversions), the formats the
/// shader pulls, and the formats whose projection is refused as a format the device does not
/// fetch.
struct Counted {
    creations: usize,
    projections: usize,
    modules: usize,
    pulled: Vec<&'static str>,
    refused: Vec<&'static str>,
}

fn count(bc: &[u8], p: &DeviceProfile, entry: &str, step: VertexStepFunction) -> Counted {
    let formats = float_readable();
    let stride = SLOT * formats.len() as u32;
    let mut projections = BTreeSet::new();
    let mut modules = BTreeSet::new();
    let mut pulled = Vec::new();
    let mut refused = Vec::new();
    for (at, &format) in formats.iter().enumerate() {
        let proj = match try_projection(p, format, at, stride, step) {
            Ok(proj) => proj,
            Err(Error::Unsupported(u)) if u.feature == Feature::Format => {
                refused.push(format.name());
                continue;
            }
            Err(e) => panic!("{format:?}: {e}"),
        };
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
        refused,
    }
}

/// `vertex-format-float4`'s own sweep under the identity domains: the device fetches 44 of the 45
/// formats in fixed function, which share **one** module of one translation, and `FloatRGB9E5`,
/// which no fixed-function input fetches, is refused, as on every device. Outside them every
/// format is pulled: one module a creation.
#[test]
fn a_float_read_compiles_one_module_for_the_fetched_formats() {
    let Some(bc) = fixture() else { return };
    let got = count(&bc, &identity(), TO_FLOAT4, VertexStepFunction::PerVertex);
    assert_eq!(got.creations, 45, "the swept formats");
    assert_eq!(
        (got.pulled.len(), got.refused),
        (0, vec!["MTLVertexFormatFloatRGB9E5"]),
        "the formats of the sweep pulled and refused on {PROFILE} under the identity domains"
    );
    assert_eq!(
        (got.projections, got.modules),
        (1, 1),
        "projections and modules of {} creations",
        got.creations
    );
    let got = count(&bc, &prof(), TO_FLOAT4, VertexStepFunction::PerVertex);
    assert_eq!(
        (got.pulled.len(), got.projections, got.modules),
        (45, 45, 45),
        "pulled formats, projections and modules outside the identity domains"
    );
}

/// The same sweep read as a `half4`: one module of one translation too, and one a creation
/// outside the identity domains.
#[test]
fn a_half_read_compiles_one_module_for_the_fetched_formats() {
    let Some(bc) = fixture() else { return };
    let got = count(&bc, &identity(), TO_HALF4, VertexStepFunction::PerVertex);
    assert_eq!(got.creations, 45, "the swept formats");
    assert_eq!(
        (got.refused.len(), got.projections, got.modules),
        (1, 1, 1),
        "refusals, projections and modules of {} creations",
        got.creations
    );
    let got = count(&bc, &prof(), TO_HALF4, VertexStepFunction::PerVertex);
    assert_eq!(
        (got.projections, got.modules),
        (45, 45),
        "projections and modules outside the identity domains"
    );
}

/// A render pipeline notes what the device's fixed-function vertex input converts, from its own
/// vertex descriptor, on its vertex stage, while one translation of the function serves every
/// format the device fetches (under the identity domains, where the sweep's attributes are fixed
/// function and a format the device does not fetch is refused). Over `vertex-format-float4`'s and
/// `-half4`'s sweeps, a normalized
/// format's pipeline records `PrecisionNote::DeviceVertexConversion` whatever width the function
/// reads it at, on `moltenvk-m3`, lavapipe and RADV alike. A narrow float's records
/// `PrecisionNote::DeviceNanEncoding` where the function reads it as a 32-bit `float` (a `half`
/// read keeps the code's own bits), and every other fixed-function format's records neither.
/// Every such creation of one entry
/// keys its vertex function alike (`RenderPipelineKeys::vertex`), so a runtime translates it once,
/// and its vertex stage is one module. The function compiled alone records neither note: its
/// module is every format's.
#[test]
fn a_pipeline_notes_what_the_device_converts_from_one_translation() {
    use crate::budget::Budget;
    use crate::device::Device;
    use crate::library::Library;
    use crate::lower::stage_io::fetch::encoding;
    use crate::pipeline::testing::description;
    use crate::pipeline::{FunctionDescriptor, RenderPipelineDescriptor, StageSource};
    let Some(bc) = fixture() else { return };
    let library = Library::from_air(&bc).expect("library");
    let formats = float_readable();
    let stride = SLOT * formats.len() as u32;
    for stem in [PROFILE, "lavapipe", "amd-780m-radv"] {
        let dev = Device::from_queries(
            description(stem).expect("committed description"),
            DevicePolicy::default(),
        )
        .expect("device");
        for entry in [TO_FLOAT4, TO_HALF4] {
            let function = FunctionDescriptor::new(library.function(entry).expect("entry"));
            let mut keys = BTreeSet::new();
            let mut modules = BTreeSet::new();
            let (mut converted, mut nans, mut fixed) = (0, 0, 0);
            for (at, &format) in formats.iter().enumerate() {
                let what = format!("{stem} {entry} {}", format.name());
                let mut d = RenderPipelineDescriptor::new(StageSource::Function(function.clone()));
                d.rasterization_enabled = false;
                d.vertex_descriptor = Some(desc(format, at, stride, VertexStepFunction::PerVertex));
                let pipeline = match dev.compile_render_pipeline(&d, &Budget::standard()) {
                    Ok(pipeline) => pipeline,
                    Err(Error::Unsupported(u)) if u.feature == Feature::Format => continue,
                    Err(e) => panic!("{what}: {e}"),
                };
                let notes = &pipeline.stages[0].precision_notes;
                assert!(pipeline.vertex_input.pulled.is_empty(), "{what}");
                fixed += 1;
                let e = encoding(format).expect("encoding");
                let conversion = e.is_normalized();
                let nan = e.widens_narrow_float() && entry == TO_FLOAT4;
                assert_eq!(
                    (
                        notes.contains(&PrecisionNote::DeviceVertexConversion),
                        notes.contains(&PrecisionNote::DeviceNanEncoding)
                    ),
                    (conversion, nan),
                    "{what}: {notes:?}"
                );
                converted += usize::from(conversion);
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
            // fixed function on the three devices, and all 19 normalized formats note it.
            assert_eq!(
                (converted, nans),
                (19, if entry == TO_FLOAT4 { 5 } else { 0 }),
                "{stem} {entry}: the creations noting each"
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
                    && !a.precision.contains(&PrecisionNote::DeviceNanEncoding),
                "{stem} {entry}: {:?}",
                a.precision
            );
        }
    }
}

/// `integer-vertex-attributes`' own sweep on its venue: sixteen unnormalized 8- and 16-bit integer
/// formats, two entry points and two step functions. Outside the identity domains each of the 64
/// creations decodes its own format in the shader: 64 modules. Under them all sixteen formats of a
/// per-vertex creation share one module an entry, and the constant step's binding, of stride 0,
/// is refused by the portability subset's stride rule. On a native driver the whole sweep is two
/// modules
/// (`the_integer_attribute_sweep_compiles_two_modules_where_the_constant_step_is_fixed_function`).
#[test]
fn the_integer_attribute_sweep_compiles_one_module_a_creation_outside_the_identity_domains() {
    let Some(bc) = fixture() else { return };
    let (p, identity) = (prof(), identity());
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
    let (mut creations, mut pulled) = (0, 0);
    let (mut fixed, mut refused) = (BTreeSet::new(), 0);
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
                match try_projection(&identity, format, at, stride, step) {
                    Ok(proj) if step == VertexStepFunction::PerVertex => {
                        fixed.insert(words(&bc, &identity, entry, &proj));
                    }
                    Err(Error::Unsupported(u))
                        if step == VertexStepFunction::Constant
                            && u.feature == Feature::VertexInputLayout =>
                    {
                        refused += 1
                    }
                    Ok(_) => panic!("{format:?} {step:?}: fetched under the identity domains"),
                    Err(e) => panic!("{format:?} {step:?}: {e}"),
                }
            }
        }
    }
    assert_eq!(creations, 64, "the sweep's pipelines");
    assert_eq!(
        pulled, 64,
        "every format is pulled outside the identity domains"
    );
    assert_eq!(
        (projections.len(), modules.len()),
        (64, 64),
        "projections and modules of {creations} creations"
    );
    assert_eq!(
        (fixed.len(), refused),
        (2, 32),
        "per-vertex modules and constant-step refusals under the identity domains"
    );
}

/// Native drivers, with no portability subset to forbid the stride of 0 the identity domains'
/// constant-step binding takes.
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

/// `integer-vertex-attributes`' sweep under the identity domains, whose constant step is a
/// binding: the step
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

/// What makes the constant-step binding exact, stated as the fetch it is: under the identity
/// domains a vertex-rate binding (divisor 1) that link emits with a stride of 0, so element 0 for
/// every vertex, instance and base instance, as Metal's constant step fetches the layout's first
/// element whatever they are. Outside them the constant step is pulled on every profile, and on
/// the recordings' portability venue, whose stride rule forbids a stride of 0.
#[test]
fn a_constant_step_is_a_zero_stride_vertex_binding_under_the_identity_domains() {
    use crate::metal::vertex::{attribute_fetch, AttributeFetch};
    let format = VertexFormat::Float4;
    let fetch = |p: &DeviceProfile| attribute_fetch(format, VertexStepFunction::Constant, 0, p);
    let binding = AttributeFetch::FixedFunction {
        format: crate::vk::Format::R32G32B32A32_SFLOAT,
        input_rate: crate::vk::VertexInputRate::VERTEX,
        divisor: 1,
    };
    for stem in NATIVE {
        let identity = profile(stem, DevicePolicy::default()).expect("committed profile");
        assert_eq!(fetch(&identity), Ok(binding), "{stem}");
        let tables = profile(stem, DevicePolicy::RESIDENT_TABLES).expect("committed profile");
        assert_eq!(fetch(&tables), Ok(AttributeFetch::Pulled), "{stem}");
    }
    assert_eq!(fetch(&prof()), Ok(AttributeFetch::Pulled), "{PROFILE}");
}
