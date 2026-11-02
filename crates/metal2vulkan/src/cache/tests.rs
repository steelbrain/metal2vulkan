//! Tests of the artifact encoding: round trips of compiled functions and pipelines over the owned
//! fixtures, and refusal of stale, corrupted, foreign, other-profile, and other-key bytes,
//! including bodies that pass the checksum but fail decoding or the sanity scan.
//!
//! `llvm-as` assembles the owned fixtures; a missing tool skips a test visibly.

use crate::budget::Budget;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::library::{CompiledFunction, FunctionStage, Library};
use crate::metal;
use crate::pipeline::artifact::{ComputePipeline, RenderPipeline};
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, FunctionDescriptor,
    RenderPipelineDescriptor, StageSource,
};
use crate::pipeline::lower_inputs::LoweringInputs;
use crate::pipeline::sym::{SlotClass, SlotRef, Stage};
use crate::pipeline::testing::{engine_dir, llvm_assemble, AssembleError};
use std::path::Path;

use super::codec::{decode, encode, encode_chunked, Codec, Malformed, CHUNK};
use super::envelope::{seal, seal_to, Kind};
use super::scan::{scan_relocatable, scan_words};
use super::{
    ArtifactError, PipelineArtifact, PipelineKey, ARTIFACT_FORMAT_VERSION, SEMANTICS_VERSION,
};

/// Owned fixture `stem` as bitcode, or `None` (skipped visibly) without `llvm-as`.
fn fixture(stem: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{stem}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {stem}.ll: {why}"),
    }
}

/// Committed profile `stem` as a device with the default policy.
fn device(stem: &str) -> Device {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(&path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::new(desc, DevicePolicy::default()).expect("device")
}

/// `store_const` compiled on `dev`, with its compute pipeline.
fn store_const(dev: &Device) -> Option<(CompiledFunction, ComputePipeline)> {
    let bc = fixture("store_const")?;
    let f = Library::from_air(&bc)
        .expect("library")
        .function("store_const")
        .expect("function");
    let compiled = dev
        .compile_function(
            &FunctionDescriptor::new(f),
            &LoweringInputs::new(),
            &Budget::standard(),
        )
        .expect("compile");
    let pipe = dev
        .compile_compute_pipeline(
            &ComputePipelineDescriptor::new(StageSource::Compiled(compiled.clone())),
            &Budget::standard(),
        )
        .expect("pipeline");
    Some((compiled, pipe))
}

/// `c` with its artifact changed by `edit`, sealed with a valid frame and checksum.
fn resealed(
    c: &CompiledFunction,
    edit: impl FnOnce(&mut crate::library::FunctionArtifact),
) -> Vec<u8> {
    let mut artifact = c.artifact().clone();
    edit(&mut artifact);
    seal(
        Kind::Function,
        artifact.key.digest(),
        &artifact.profile,
        &encode(&artifact),
    )
}

/// Every function of every owned fixture that compiles on `moltenvk-m3` round-trips, as a
/// compiled function and as a pipeline, and a pipeline linked from a decoded function equals the
/// one linked from the original.
#[test]
fn every_compiled_fixture_round_trips() {
    let dev = device("moltenvk-m3");
    let mut stems: Vec<String> = std::fs::read_dir(engine_dir())
        .expect("fixtures")
        .filter_map(|e| {
            let name = e.ok()?.file_name().into_string().ok()?;
            let stem = name.strip_suffix(".ll")?;
            (!stem.ends_with(".air")).then(|| stem.to_owned())
        })
        .collect();
    stems.sort();
    let mut functions = 0usize;
    let mut pipelines = 0usize;
    let mut two_stage = 0usize;
    let fragment_vertex = fixture("stage_io").map(|bc| {
        Library::from_air(&bc)
            .expect("library")
            .function("stage_io_vertex_position")
            .expect("vertex")
    });
    for stem in &stems {
        let Some(bc) = fixture(stem) else {
            return;
        };
        let lib = Library::from_air(&bc).expect("library");
        for info in lib.functions().collect::<Vec<_>>() {
            let f = lib.function(info.name).expect("function");
            let fd = FunctionDescriptor::new(f.clone());
            if let Ok(c) = dev.compile_function(&fd, &LoweringInputs::new(), &Budget::standard()) {
                let back = CompiledFunction::from_bytes(&c.to_bytes(), &dev).expect(info.name);
                assert_eq!(back, c, "{stem}/{}", info.name);
                assert_eq!(back.to_bytes(), c.to_bytes(), "the encoding is canonical");
                functions += 1;
            }
            match info.stage {
                FunctionStage::Kernel => {
                    let d = ComputePipelineDescriptor::new(StageSource::Function(fd));
                    if let Ok(p) = dev.compile_compute_pipeline(&d, &Budget::standard()) {
                        let back =
                            ComputePipeline::from_bytes(&p.to_bytes(), &dev).expect(info.name);
                        assert_eq!(back, p, "{stem}/{}", info.name);
                        let mut written = Vec::new();
                        p.write_to(&mut written).expect(info.name);
                        assert_eq!(written, p.to_bytes(), "{stem}/{}: written", info.name);
                        pipelines += 1;
                    }
                }
                FunctionStage::Vertex | FunctionStage::Fragment => {
                    let d = if info.stage == FunctionStage::Vertex {
                        let mut d = RenderPipelineDescriptor::new(StageSource::Function(fd));
                        d.rasterization_enabled = false;
                        d
                    } else {
                        let Some(v) = &fragment_vertex else { continue };
                        let mut d = RenderPipelineDescriptor::new(StageSource::Function(
                            FunctionDescriptor::new(v.clone()),
                        ));
                        d.fragment = Some(StageSource::Function(fd));
                        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(
                            metal::PixelFormat::RGBA8Unorm,
                        ));
                        d
                    };
                    if let Ok(p) = dev.compile_render_pipeline(&d, &Budget::standard()) {
                        let back =
                            RenderPipeline::from_bytes(&p.to_bytes(), &dev).expect(info.name);
                        assert_eq!(back, p, "{stem}/{}", info.name);
                        let mut written = Vec::new();
                        p.write_to(&mut written).expect(info.name);
                        assert_eq!(written, p.to_bytes(), "{stem}/{}: written", info.name);
                        pipelines += 1;
                        two_stage += usize::from(p.stages.len() == 2);
                    }
                }
                _ => {}
            }
        }
    }
    eprintln!(
        "round-tripped {functions} functions and {pipelines} pipelines ({two_stage} two-stage)"
    );
    assert!(
        functions >= 10 && pipelines >= 10 && two_stage >= 1,
        "{functions} / {pipelines} / {two_stage}"
    );
}

/// A pipeline linked from a decoded `CompiledFunction` equals the one linked from the original,
/// and the decoded function keeps its profile and kernel facts.
#[test]
fn a_decoded_function_links_like_the_original() {
    let dev = device("moltenvk-m3");
    let Some((compiled, pipe)) = store_const(&dev) else {
        return;
    };
    let back = CompiledFunction::from_bytes(&compiled.to_bytes(), &dev).expect("decode");
    assert_eq!(back.artifact().profile, dev.profile().fingerprint());
    assert_eq!(back.artifact().kernel, compiled.artifact().kernel);
    let d = ComputePipelineDescriptor::new(StageSource::Compiled(back));
    assert_eq!(
        dev.compile_compute_pipeline(&d, &Budget::standard()),
        Ok(pipe)
    );
    assert_eq!(
        CompiledFunction::from_bytes_for_key(&compiled.to_bytes(), compiled.key(), &dev),
        Ok(compiled.clone())
    );
}

/// A function decoded from cache bytes answers what each of its slots may be stored through, the
/// same as the function it was compiled from: the question a consumer asks of a state it took from
/// a cache, in a process that never saw the AIR. The encoding carries `slot_writes`, and a compile
/// keeps it whatever the reflection of the same entry could describe, so the decoded answer is a
/// determined one and not an empty list (which would read as a state that binds no slot).
#[test]
fn a_decoded_function_answers_its_slot_writes() {
    let dev = device("moltenvk-m3");
    let Some(bc) = fixture("store_const") else {
        return;
    };
    let f = Library::from_air(&bc)
        .expect("library")
        .function("store_const")
        .expect("function");
    let fd = FunctionDescriptor::new(f);
    let compiled = dev
        .compile_function(&fd, &LoweringInputs::new(), &Budget::standard())
        .expect("compile");
    let back = CompiledFunction::from_bytes(&compiled.to_bytes(), &dev).expect("decode");

    let from_function = ComputePipelineDescriptor::new(StageSource::Function(fd))
        .slot_writes(&Budget::standard())
        .expect("the function path answers");
    let from_decoded = ComputePipelineDescriptor::new(StageSource::Compiled(back))
        .slot_writes(&Budget::standard())
        .expect("a decoded function answers");
    assert_eq!(from_decoded, from_function);

    // `device uint *out [[buffer(0)]]`: one slot, read-write, as Metal answers a device buffer.
    let slots = from_decoded.kernel_slots();
    assert_eq!(slots.len(), 1, "{slots:?}");
    assert_eq!(slots[0].index, Some(0));
    assert_eq!(
        crate::reflection::SlotWrite::may_write(slots, metal::BindingType::Buffer, 0),
        crate::reflection::SlotWritability::Maybe
    );
    // Nothing binds buffer 1, so no store of the state can land there: the answer a consumer
    // needs before it binds a read-only buffer.
    assert_eq!(
        crate::reflection::SlotWrite::may_write(slots, metal::BindingType::Buffer, 1),
        crate::reflection::SlotWritability::Never
    );
}

/// Bytes of another profile are `WrongDevice`, bytes of another key too when the caller names
/// the key, and bytes of another artifact kind are `Corrupt`.
#[test]
fn foreign_bytes_are_refused() {
    let dev = device("moltenvk-m3");
    let Some((compiled, pipe)) = store_const(&dev) else {
        return;
    };
    let other = device("vk12-floor");
    assert_ne!(other.profile().fingerprint(), dev.profile().fingerprint());
    assert_eq!(
        CompiledFunction::from_bytes(&compiled.to_bytes(), &other),
        Err(ArtifactError::WrongDevice)
    );
    assert_eq!(
        ComputePipeline::from_bytes(&pipe.to_bytes(), &other),
        Err(ArtifactError::WrongDevice)
    );
    let wrong_key = crate::library::FunctionCompileKey::new([7; 32]);
    assert_eq!(
        CompiledFunction::from_bytes_for_key(&compiled.to_bytes(), &wrong_key, &dev),
        Err(ArtifactError::WrongDevice)
    );
    assert_eq!(
        ComputePipeline::from_bytes_for_key(&pipe.to_bytes(), &PipelineKey::new([7; 32]), &dev),
        Err(ArtifactError::WrongDevice)
    );
    assert_eq!(
        ComputePipeline::from_bytes_for_key(&pipe.to_bytes(), &pipe.key, &dev),
        Ok(pipe.clone())
    );
    // A function is not a pipeline, nor a compute pipeline a render pipeline.
    assert_eq!(
        ComputePipeline::from_bytes(&compiled.to_bytes(), &dev),
        Err(ArtifactError::Corrupt)
    );
    assert_eq!(
        RenderPipeline::from_bytes(&pipe.to_bytes(), &dev),
        Err(ArtifactError::Corrupt)
    );
    // Bytes that are no artifact of this crate at all.
    for junk in [&b""[..], b"\x03\x02\x23\x07", &[0u8; 256][..]] {
        assert_eq!(
            CompiledFunction::from_bytes(junk, &dev),
            Err(ArtifactError::Stale)
        );
    }
}

/// Another format or semantics version is `Stale`; any other changed byte, a truncation, or an
/// appended byte is `Corrupt` (the checksum).
#[test]
fn corrupted_bytes_are_refused() {
    let dev = device("moltenvk-m3");
    let Some((compiled, _)) = store_const(&dev) else {
        return;
    };
    let good = compiled.to_bytes();
    for (at, expect) in [
        (0, ArtifactError::Stale),
        (8, ArtifactError::Stale),
        (12, ArtifactError::Stale),
        (16, ArtifactError::Corrupt),
        (20, ArtifactError::Corrupt),
        (52, ArtifactError::Corrupt),
        (84, ArtifactError::Corrupt),
        (92, ArtifactError::Corrupt),
        (good.len() / 2, ArtifactError::Corrupt),
        (good.len() - 1, ArtifactError::Corrupt),
    ] {
        let mut bad = good.clone();
        bad[at] ^= 0x01;
        assert_eq!(
            CompiledFunction::from_bytes(&bad, &dev),
            Err(expect),
            "byte {at}"
        );
    }
    assert_eq!(
        CompiledFunction::from_bytes(&good[..good.len() - 1], &dev),
        Err(ArtifactError::Corrupt)
    );
    let mut longer = good.clone();
    longer.push(0);
    assert_eq!(
        CompiledFunction::from_bytes(&longer, &dev),
        Err(ArtifactError::Corrupt)
    );
    assert!(CompiledFunction::from_bytes(&good, &dev).is_ok());
}

/// An artifact framed by any earlier format version is `Stale`, even with a valid checksum: it is
/// refused, never decoded under this version's layout (version 4 added
/// `RuntimeFact::HeapNumericType`).
#[test]
fn earlier_format_versions_are_stale() {
    let dev = device("moltenvk-m3");
    let Some((compiled, pipe)) = store_const(&dev) else {
        return;
    };
    const { assert!(ARTIFACT_FORMAT_VERSION >= 4) };
    for good in [compiled.to_bytes(), pipe.to_bytes()] {
        for version in 1..ARTIFACT_FORMAT_VERSION {
            let end = good.len() - 32;
            let mut old = good[..end].to_vec();
            old[8..12].copy_from_slice(&version.to_le_bytes());
            use sha2::Digest;
            let checksum = sha2::Sha256::digest(&old);
            old.extend_from_slice(&checksum);
            assert_eq!(
                CompiledFunction::from_bytes(&old, &dev).err(),
                Some(ArtifactError::Stale),
                "version {version}"
            );
            assert_eq!(
                ComputePipeline::from_bytes(&old, &dev).err(),
                Some(ArtifactError::Stale),
                "version {version}"
            );
        }
    }
}

/// The fact format version 4 added encodes and decodes.
#[test]
fn the_heap_numeric_type_fact_round_trips() {
    use crate::pipeline::facts::RuntimeFact;
    use crate::pipeline::sym::{
        ImageAccess, ImageDim, ImageHandleClass, SampledType, StorageFormatClass,
    };
    let fact = RuntimeFact::HeapNumericType {
        class: ImageHandleClass {
            dim: ImageDim::D2,
            arrayed: true,
            multisampled: false,
            sampled_type: SampledType::Float,
            depth_compare: false,
            access: ImageAccess::Storage,
            format: StorageFormatClass::Unknown,
        },
    };
    assert_eq!(decode::<RuntimeFact>(&encode(&fact)), Ok(fact));
}

/// The `AnisotropicSampler` fact encodes and decodes.
#[test]
fn the_anisotropic_sampler_fact_round_trips() {
    use crate::pipeline::facts::RuntimeFact;
    let fact = RuntimeFact::AnisotropicSampler {
        slot: SlotRef {
            stage: Stage::Fragment,
            class: SlotClass::Sampler,
            index: 3,
        },
    };
    assert_eq!(decode::<RuntimeFact>(&encode(&fact)), Ok(fact));
}

/// The fact format version 157 added encodes and decodes, and the artifact of a function whose
/// `ulong` slots a `ulong` atomic updates carries it through its bytes: a compiled function of the
/// owned fixture `unk_ulong_atomics` on a device with 64-bit image atomics, and its pipeline,
/// decode to themselves with the fact for the atomic slot and none for the slot only read and
/// written.
#[test]
fn the_ulong_atomic_view_fact_round_trips() {
    use crate::pipeline::facts::RuntimeFact;
    let slot = |index| SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Texture,
        index,
    };
    let fact = RuntimeFact::UlongAtomicView { slot: slot(5) };
    assert_eq!(decode::<RuntimeFact>(&encode(&fact)), Ok(fact));
    let Some(bc) = fixture("unk_ulong_atomics") else {
        return;
    };
    let dev = device("amd-rdna3-radv");
    let f = Library::from_air(&bc)
        .expect("library")
        .function("unk_ulong_atomic_rw")
        .expect("function");
    let compiled = dev
        .compile_function(
            &FunctionDescriptor::new(f),
            &LoweringInputs::new(),
            &Budget::standard(),
        )
        .expect("compile");
    let facts = &compiled.artifact().facts;
    assert!(facts.contains(&RuntimeFact::UlongAtomicView { slot: slot(0) }));
    assert!(!facts.contains(&RuntimeFact::UlongAtomicView { slot: slot(1) }));
    let bytes = compiled.to_bytes();
    let back = CompiledFunction::from_bytes(&bytes, &dev).expect("decode");
    assert_eq!(back.artifact(), compiled.artifact());
    assert_eq!(back.to_bytes(), bytes);
    let pipe = dev
        .compile_compute_pipeline(
            &ComputePipelineDescriptor::new(StageSource::Compiled(compiled)),
            &Budget::standard(),
        )
        .expect("pipeline");
    let bytes = pipe.to_bytes();
    let back = ComputePipeline::from_bytes(&bytes, &dev).expect("decode");
    assert_eq!(back.to_bytes(), bytes);
    assert!(back
        .dependencies
        .iter()
        .any(|d| d.fact == RuntimeFact::UlongAtomicView { slot: slot(0) }));
}

/// The members format version 178 appends encode and decode, and each decodes as itself:
/// `VkFeature::Maintenance5` and the three framebuffer `VkLimit`s.
#[test]
fn the_version_178_members_round_trip() {
    use crate::device::requirements::{VkFeature, VkLimit};
    let feature = VkFeature::Maintenance5;
    assert_eq!(decode::<VkFeature>(&encode(&feature)), Ok(feature));
    for limit in [
        VkLimit::MaxFramebufferWidth,
        VkLimit::MaxFramebufferHeight,
        VkLimit::MaxFramebufferLayers,
    ] {
        assert_eq!(decode::<VkLimit>(&encode(&limit)), Ok(limit));
    }
}

/// The members format version 179 appends encode and decode, and each decodes as itself:
/// `VkFeature::{OcclusionQueryPrecise, DepthBiasControl, FloatRepresentation, DepthBiasExact}`,
/// `VkExtension::ExtDepthBiasControl`, and `PushField::IndexCapacity`.
#[test]
fn the_version_179_members_round_trip() {
    use crate::device::requirements::{VkExtension, VkFeature};
    use crate::pipeline::sym::PushField;
    let field = PushField::IndexCapacity;
    assert_eq!(decode::<PushField>(&encode(&field)), Ok(field));
    for feature in [
        VkFeature::OcclusionQueryPrecise,
        VkFeature::DepthBiasControl,
        VkFeature::FloatRepresentation,
        VkFeature::DepthBiasExact,
    ] {
        assert_eq!(decode::<VkFeature>(&encode(&feature)), Ok(feature));
    }
    let extension = VkExtension::ExtDepthBiasControl;
    assert_eq!(decode::<VkExtension>(&encode(&extension)), Ok(extension));
}

/// The member format version 197 appends encodes and decodes as itself:
/// `PrecisionNote::FunctionStorageAssumed`.
#[test]
fn the_version_197_members_round_trip() {
    use crate::pipeline::artifact::PrecisionNote;
    let note = PrecisionNote::FunctionStorageAssumed;
    assert_eq!(decode::<PrecisionNote>(&encode(&note)), Ok(note));
}

/// The member format version 209 appends encodes and decodes as itself:
/// `PrecisionNote::QuadsCountedBeforeDiscard`.
#[test]
fn the_version_209_members_round_trip() {
    use crate::pipeline::artifact::PrecisionNote;
    let note = PrecisionNote::QuadsCountedBeforeDiscard;
    assert_eq!(decode::<PrecisionNote>(&encode(&note)), Ok(note));
}

/// The member format version 219 appends encodes and decodes as itself:
/// `PrecisionNote::ViewBaseTexelOffset`.
#[test]
fn the_version_219_members_round_trip() {
    use crate::pipeline::artifact::PrecisionNote;
    let note = PrecisionNote::ViewBaseTexelOffset;
    assert_eq!(decode::<PrecisionNote>(&encode(&note)), Ok(note));
}

/// The member format version 216 appends encodes and decodes as itself:
/// `RuntimeFact::TileImageblock`.
#[test]
fn the_version_216_members_round_trip() {
    use crate::pipeline::facts::RuntimeFact;
    let fact = RuntimeFact::TileImageblock {
        data_size: 8,
        alignment: 4,
    };
    assert_eq!(decode::<RuntimeFact>(&encode(&fact)), Ok(fact));
}

/// The member format version 190 appends encodes and decodes as itself:
/// `RuntimeFact::AliasedImageblock`.
#[test]
fn the_version_190_members_round_trip() {
    use crate::pipeline::facts::RuntimeFact;
    let fact = RuntimeFact::AliasedImageblock { data_size: 24 };
    assert_eq!(decode::<RuntimeFact>(&encode(&fact)), Ok(fact));
}

/// The fields format version 193 adds encode and decode with the fact and the pipeline state that
/// carry them: `RuntimeFact::FragmentImageblock`'s and `FragmentImageblock`'s `data_size` and
/// `alignment`.
#[test]
fn the_version_193_fields_round_trip() {
    use crate::pipeline::facts::RuntimeFact;
    use crate::pipeline::{FragmentImageblock, ImageblockMechanism};
    let fact = RuntimeFact::FragmentImageblock {
        words: 3,
        writes: true,
        master: [5; 32],
        data_size: 11,
        alignment: 2,
    };
    assert_eq!(decode::<RuntimeFact>(&encode(&fact)), Ok(fact));
    let state = FragmentImageblock {
        mechanism: ImageblockMechanism::StorageImageInterlock,
        words: 8,
        master: [6; 32],
        data_size: 32,
        alignment: 16,
    };
    assert_eq!(decode::<FragmentImageblock>(&encode(&state)), Ok(state));
}

/// The members format version 187 appends encode and decode, and each decodes as itself:
/// `PushField::ViewportCount`, `RuntimeFact::ViewportCount`, and `IoRole::ViewportIndex`.
#[test]
fn the_version_187_members_round_trip() {
    use crate::pipeline::facts::RuntimeFact;
    use crate::pipeline::stage_io::IoRole;
    use crate::pipeline::sym::PushField;
    let field = PushField::ViewportCount;
    assert_eq!(decode::<PushField>(&encode(&field)), Ok(field));
    let fact = RuntimeFact::ViewportCount;
    assert_eq!(decode::<RuntimeFact>(&encode(&fact)), Ok(fact));
    let role = IoRole::ViewportIndex;
    assert_eq!(decode::<IoRole>(&encode(&role)), Ok(role));
}

/// The members format version 201 appends encode and decode, and each decodes as itself:
/// `PushField::RenderTargetArrayLength` and `IoRole::RenderTargetArrayIndex`.
#[test]
fn the_version_201_members_round_trip() {
    use crate::pipeline::stage_io::IoRole;
    use crate::pipeline::sym::PushField;
    let field = PushField::RenderTargetArrayLength;
    assert_eq!(decode::<PushField>(&encode(&field)), Ok(field));
    let role = IoRole::RenderTargetArrayIndex;
    assert_eq!(decode::<IoRole>(&encode(&role)), Ok(role));
}

/// The members format versions 195 and 196 append, `VkLimit::FunctionStorageBytesPerThread` and
/// `RuntimeFact::CallStackDepth`, encode and decode as themselves.
#[test]
fn the_version_195_and_196_members_round_trip() {
    use crate::device::requirements::VkLimit;
    use crate::pipeline::facts::RuntimeFact;
    let limit = VkLimit::FunctionStorageBytesPerThread;
    assert_eq!(decode::<VkLimit>(&encode(&limit)), Ok(limit));
    let fact = RuntimeFact::CallStackDepth { depth: 1024 };
    assert_eq!(decode::<RuntimeFact>(&encode(&fact)), Ok(fact));
}

/// The member format version 191 appends, `VkProperty::ColorAttachmentSampleCount`, encodes and
/// decodes as itself with both its fields: the format and the count.
#[test]
fn the_version_191_member_round_trips() {
    use crate::device::requirements::VkProperty;
    use crate::vk;
    for property in [
        VkProperty::ColorAttachmentSampleCount(vk::Format::R32G32B32A32_UINT, 4),
        VkProperty::ColorAttachmentSampleCount(vk::Format::A2B10G10R10_UNORM_PACK32, 8),
    ] {
        assert_eq!(decode::<VkProperty>(&encode(&property)), Ok(property));
    }
}

/// The members format version 180 appends encode and decode, and each decodes as itself: the eight
/// `VkFeature`s, the four `VkExtension`s, the six `VkProperty`s with every `ResolveMode`, and the
/// two `VkLimit`s; and `VkFeature::ImageViewFormatSwizzle`, which version 204 appends after them.
#[test]
fn the_version_180_members_round_trip() {
    use crate::device::requirements::{ResolveMode, VkExtension, VkFeature, VkLimit, VkProperty};
    for feature in [
        VkFeature::MultisampleArrayImage,
        VkFeature::ImageView2DOn3DImage,
        VkFeature::SeparateStencilMaskRef,
        VkFeature::FullDrawIndexUint32,
        VkFeature::DepthClipEnable,
        VkFeature::DepthClampControl,
        VkFeature::PrimitiveTopologyListRestart,
        VkFeature::PrimitiveTopologyPatchListRestart,
        VkFeature::ImageViewFormatSwizzle,
    ] {
        assert_eq!(decode::<VkFeature>(&encode(&feature)), Ok(feature));
    }
    for extension in [
        VkExtension::ExtDepthClipEnable,
        VkExtension::ExtDepthClampControl,
        VkExtension::ExtSampleLocations,
        VkExtension::ExtPrimitiveTopologyListRestart,
    ] {
        assert_eq!(decode::<VkExtension>(&encode(&extension)), Ok(extension));
    }
    let mut properties = vec![
        VkProperty::IndependentResolveNone,
        VkProperty::IndependentResolve,
        VkProperty::VariableSampleLocations,
        VkProperty::SampleLocationSampleCount(4),
    ];
    for mode in ResolveMode::ALL {
        properties.push(VkProperty::DepthResolveMode(mode));
        properties.push(VkProperty::StencilResolveMode(mode));
    }
    for property in properties {
        assert_eq!(decode::<VkProperty>(&encode(&property)), Ok(property));
    }
    for limit in [
        VkLimit::MaxDrawIndexedIndexValue,
        VkLimit::MaxTessellationPatchSize,
        VkLimit::MaxMeshOutputVertices,
        VkLimit::MaxMeshOutputPrimitives,
        VkLimit::MaxMeshOutputComponents,
        VkLimit::MaxTaskSharedMemorySize,
        VkLimit::MaxMeshSharedMemorySize,
        VkLimit::MaxMeshWorkGroupCount(2),
        VkLimit::MaxMeshWorkGroupTotalCount,
        VkLimit::MaxMeshWorkGroupSize(1),
        VkLimit::MaxMeshWorkGroupInvocations,
        VkLimit::MaxMeshOutputLayers,
    ] {
        assert_eq!(decode::<VkLimit>(&encode(&limit)), Ok(limit));
    }
}

/// The members format version 327 appends encode and decode, and each decodes as itself:
/// `VkFeature::ExtendedDynamicState2` and `VkExtension::ExtExtendedDynamicState2`.
#[test]
fn the_version_327_members_round_trip() {
    use crate::device::requirements::{VkExtension, VkFeature};
    let feature = VkFeature::ExtendedDynamicState2;
    assert_eq!(decode::<VkFeature>(&encode(&feature)), Ok(feature));
    let extension = VkExtension::ExtExtendedDynamicState2;
    assert_eq!(decode::<VkExtension>(&encode(&extension)), Ok(extension));
}

/// The members format version 234 appends encode and decode, each as itself: the object function's
/// specialization keys and the task stage's and the payload's limits.
#[test]
fn the_version_234_members_round_trip() {
    use crate::device::requirements::VkLimit;
    use crate::pipeline::sym::{Axis, SpecKey};
    for key in [
        SpecKey::ObjectLocalSize(Axis::Y),
        SpecKey::ObjectFlatLocalSize,
        SpecKey::ObjectMultiRegion,
    ] {
        assert_eq!(decode::<SpecKey>(&encode(&key)), Ok(key));
    }
    for limit in [
        VkLimit::MaxTaskWorkGroupCount(1),
        VkLimit::MaxTaskWorkGroupTotalCount,
        VkLimit::MaxTaskWorkGroupSize(2),
        VkLimit::MaxTaskWorkGroupInvocations,
        VkLimit::MaxTaskPayloadSize,
        VkLimit::MaxTaskPayloadAndSharedMemorySize,
        VkLimit::MaxMeshPayloadAndSharedMemorySize,
    ] {
        assert_eq!(decode::<VkLimit>(&encode(&limit)), Ok(limit));
    }
}

/// The member format version 253 appends encodes and decodes as itself: the minimum level a clamped
/// LOD query outside a fragment function reads of a sampler slot.
#[test]
fn the_version_253_members_round_trip() {
    use crate::pipeline::sym::SpecKey;
    let key = SpecKey::SamplerMinLevel(SlotRef {
        stage: Stage::Vertex,
        class: SlotClass::Sampler,
        index: 5,
    });
    assert_eq!(decode::<SpecKey>(&encode(&key)), Ok(key));
}

/// The members format version 267 appends encode and decode as themselves: a sparse texture slot's
/// constant lanes, as a specialization and as a fact.
#[test]
fn the_version_267_members_round_trip() {
    use crate::pipeline::facts::RuntimeFact;
    use crate::pipeline::sym::SpecKey;
    let slot = SlotRef {
        stage: Stage::Fragment,
        class: SlotClass::Texture,
        index: 3,
    };
    let key = SpecKey::ConstantLanes(slot);
    assert_eq!(decode::<SpecKey>(&encode(&key)), Ok(key));
    let fact = RuntimeFact::ConstantLanes { slot };
    assert_eq!(decode::<RuntimeFact>(&encode(&fact)), Ok(fact));
}

/// The member format version 326 appends encodes and decodes as itself: whether a texture slot's
/// bound format has no family Metal's `half` conversion is measured for on a device sampling
/// through Metal's own texture unit.
#[test]
fn the_version_326_member_round_trips() {
    use crate::pipeline::sym::SpecKey;
    let key = SpecKey::UnmeasuredTexels(SlotRef {
        stage: Stage::Fragment,
        class: SlotClass::Texture,
        index: 7,
    });
    assert_eq!(decode::<SpecKey>(&encode(&key)), Ok(key));
    assert_ne!(
        encode(&key),
        encode(&SpecKey::ConstantLanes(SlotRef {
            stage: Stage::Fragment,
            class: SlotClass::Texture,
            index: 7,
        }))
    );
}

/// The member format version 374 appends encodes and decodes as itself: the axes on which a texture
/// slot's bound view's parent's base-level size, which Metal's texel-offset rule reads, is not a
/// power of two where the view's own is.
#[test]
fn the_version_374_member_round_trips() {
    use crate::pipeline::sym::SpecKey;
    let slot = SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Texture,
        index: 4,
    };
    let key = SpecKey::OffsetBaseAxes(slot);
    assert_eq!(decode::<SpecKey>(&encode(&key)), Ok(key));
    assert_ne!(encode(&key), encode(&SpecKey::BoundTexelFormat(slot)));
}

/// The member format version 377 appends encodes and decodes as itself, with the key it carries: a
/// sampler slot's word, carried as data, and a texture slot's format class.
#[test]
fn the_version_377_member_round_trips() {
    use crate::pipeline::sym::{PushField, SpecKey};
    let sampler = SlotRef {
        stage: Stage::Fragment,
        class: SlotClass::Sampler,
        index: 3,
    };
    let texture = SlotRef {
        stage: Stage::Fragment,
        class: SlotClass::Texture,
        index: 3,
    };
    for key in [
        SpecKey::SamplerState {
            slot: sampler,
            fields: 0x0003_00ff,
        },
        SpecKey::TexelFormat(texture),
    ] {
        let field = PushField::Specialized(key);
        assert_eq!(decode::<PushField>(&encode(&field)), Ok(field));
        assert_ne!(encode(&field), encode(&key));
    }
}

/// The members format versions 153, 154, 155, 157 and 161 joined from every side encode and
/// decode together in one render pipeline: `PrecisionNote::CubeGradientFootprint` and
/// `CombinedSampler`, `RuntimeFact::InterlockedFramebufferFetch` and
/// `GraphicsState::storage_attachments`, `SpecKey::TexelFormat`, `RuntimeFact::TexelFormat` and
/// `SrgbTexture`, `VkProperty::MetalSampler`, and `PrecisionNote::HalfTexelConversion`,
/// `HalfSampleFilter`, `SrgbTexelDecode`, and `FloatSampleFilter`,
/// `HandleClass::Sampler { compare, integer }` and
/// `SyntheticBinding::SamplerTable`, and `RuntimeFact::UlongAtomicView`, each beside the others
/// rather than only on its own side's fixtures.
#[test]
fn the_merged_members_round_trip_in_one_render_pipeline() {
    use crate::device::requirements::VkProperty;
    use crate::pipeline::artifact::{PrecisionNote, RuntimeDependency, SpecConstant};
    use crate::pipeline::facts::{Mechanism, RuntimeFact};
    use crate::pipeline::graphics::StorageAttachmentUse;
    use crate::pipeline::layout::{BindingSource, DescriptorBinding, DescriptorSetLayout};
    use crate::pipeline::sym::{
        CombinedSampler, HandleClass, ResourceKind, SamplerRef, SpecKey, SyntheticBinding,
    };
    use crate::vk;
    let texture = SlotRef {
        stage: Stage::Fragment,
        class: SlotClass::Texture,
        index: 3,
    };
    let sampler = CombinedSampler::new(SamplerRef::Slot(1), true, false);
    let kind = ResourceKind::CombinedTextureSampler(sampler);
    assert_eq!(decode::<ResourceKind>(&encode(&kind)), Ok(kind));
    let notes = [
        PrecisionNote::CubeGradientFootprint,
        PrecisionNote::HalfTexelConversion,
        PrecisionNote::HalfSampleFilter,
        PrecisionNote::SrgbTexelDecode,
        PrecisionNote::FloatSampleFilter,
    ];
    let dev = device("moltenvk-m3");
    let Some(bc) = fixture("stage_io") else {
        return;
    };
    let f = Library::from_air(&bc)
        .expect("library")
        .function("stage_io_vertex_position")
        .expect("vertex");
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f)));
    d.rasterization_enabled = false;
    let mut p = dev
        .compile_render_pipeline(&d, &Budget::standard())
        .expect("pipeline");
    p.stages[0].precision_notes.extend(notes);
    let id = p.stages[0].specialization.constants.len() as u32;
    p.stages[0].specialization.constants.push(SpecConstant {
        id,
        key: SpecKey::TexelFormat(texture),
        size: 4,
        default: 0,
        unread_bits: 0,
    });
    p.requirements.properties.insert(VkProperty::MetalSampler);
    let set = p.layout.sets.len() as u32;
    p.layout.sets.push(DescriptorSetLayout {
        set,
        bindings: vec![
            DescriptorBinding {
                binding: 0,
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                count: 1,
                stages: vk::ShaderStageFlags::FRAGMENT,
                source: BindingSource::CombinedTextureSampler { texture, sampler },
                immutable_sampler: None,
                flags: vk::DescriptorBindingFlags::empty(),
                variables: 1,
            },
            DescriptorBinding {
                binding: 1,
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                count: 1,
                stages: vk::ShaderStageFlags::FRAGMENT,
                source: BindingSource::Texture(texture),
                immutable_sampler: None,
                flags: vk::DescriptorBindingFlags::empty(),
                variables: 1,
            },
        ],
        flags: vk::DescriptorSetLayoutCreateFlags::empty(),
    });
    // An argument buffer's sampler array of a comparison use of an integer texel, and the stage's
    // sampler table.
    let argument_samplers = vec![
        DescriptorBinding {
            binding: 2,
            ty: vk::DescriptorType::SAMPLER,
            count: 16,
            stages: vk::ShaderStageFlags::FRAGMENT,
            source: BindingSource::Heap(HandleClass::Sampler {
                compare: true,
                integer: true,
            }),
            immutable_sampler: None,
            flags: vk::DescriptorBindingFlags::PARTIALLY_BOUND,
            variables: 1,
        },
        DescriptorBinding {
            binding: 3,
            ty: vk::DescriptorType::STORAGE_BUFFER,
            count: 1,
            stages: vk::ShaderStageFlags::FRAGMENT,
            source: BindingSource::Synthetic {
                stage: Stage::Fragment,
                binding: SyntheticBinding::SamplerTable,
            },
            immutable_sampler: None,
            flags: vk::DescriptorBindingFlags::empty(),
            variables: 1,
        },
    ];
    if let Some(last) = p.layout.sets.last_mut() {
        last.bindings.extend(argument_samplers);
    }
    p.graphics.storage_attachments.push(StorageAttachmentUse {
        color_index: 0,
        format: vk::Format::R8G8B8A8_UNORM,
        writes: true,
        formatless: false,
        view: crate::pipeline::StorageAttachmentView::SamplePlanes { layers: 4 },
        view_alias: Some(vk::Format::R32_UINT),
        kept: vk::ColorComponentFlags::B,
    });
    p.dependencies.push(RuntimeDependency {
        fact: RuntimeFact::InterlockedFramebufferFetch {
            color: 0,
            format: vk::Format::R8G8B8A8_UNORM,
            writes: true,
            early_tests: true,
            kept: vk::ColorComponentFlags::A,
            blend: Some(crate::pipeline::lower_inputs::BlendEquation {
                source_rgb: crate::metal::BlendFactor::One,
                destination_rgb: crate::metal::BlendFactor::OneMinusSourceAlpha,
                rgb_operation: crate::metal::BlendOperation::Add,
                source_alpha: crate::metal::BlendFactor::One,
                destination_alpha: crate::metal::BlendFactor::Zero,
                alpha_operation: crate::metal::BlendOperation::Add,
            }),
            rounded: true,
        },
        mechanism: Mechanism::RenderPassContract,
    });
    p.dependencies.push(RuntimeDependency {
        fact: RuntimeFact::TexelFormat { slot: texture },
        mechanism: Mechanism::Specialization,
    });
    p.dependencies.push(RuntimeDependency {
        fact: RuntimeFact::SrgbTexture { slot: texture },
        mechanism: Mechanism::Variant,
    });
    let ulong_view = RuntimeFact::UlongAtomicView { slot: texture };
    p.dependencies.push(RuntimeDependency {
        fact: ulong_view,
        mechanism: Mechanism::BinderView,
    });
    let bytes = encode(&p);
    let back = decode::<RenderPipeline>(&bytes).expect("decoded");
    assert_eq!(back, p);
    assert_eq!(encode(&back), bytes, "the encoding is canonical");
    let sealed = RenderPipeline::from_bytes(&p.to_bytes(), &dev).expect("sealed");
    assert_eq!(sealed, p);
    assert!(notes
        .iter()
        .all(|n| sealed.stages[0].precision_notes.contains(n)));
    assert!(sealed.stages[0]
        .specialization
        .constants
        .iter()
        .any(|c| c.key == SpecKey::TexelFormat(texture)));
    assert!(sealed
        .requirements
        .properties
        .contains(&VkProperty::MetalSampler));
    let bindings = sealed.layout.sets.last().map(|s| &s.bindings[..]);
    assert!(matches!(
        bindings.map(|b| &b[0].source),
        Some(BindingSource::CombinedTextureSampler { sampler, .. })
            if sampler.compare
    ));
    assert_eq!(
        bindings.map(|b| &b[1].source),
        Some(&BindingSource::Texture(texture))
    );
    assert!(matches!(
        bindings.map(|b| &b[2].source),
        Some(BindingSource::Heap(HandleClass::Sampler {
            compare: true,
            integer: true
        }))
    ));
    assert!(matches!(
        bindings.map(|b| &b[3].source),
        Some(BindingSource::Synthetic {
            binding: SyntheticBinding::SamplerTable,
            ..
        })
    ));
    for fact in [
        RuntimeFact::TexelFormat { slot: texture },
        RuntimeFact::SrgbTexture { slot: texture },
    ] {
        assert!(
            sealed.dependencies.iter().any(|d| d.fact == fact),
            "{fact:?}"
        );
    }
    assert_eq!(sealed.graphics.storage_attachments.len(), 1);
    assert!(sealed.dependencies.iter().any(|d| matches!(
        d.fact,
        RuntimeFact::InterlockedFramebufferFetch {
            early_tests: true,
            ..
        }
    )));
    assert!(sealed.dependencies.iter().any(|d| d.fact == ulong_view));
}

/// The emulated-simdgroups variant of a kernel moltenvk-m3 runs on whole subgroups round trips as a
/// function and as a pipeline, and the decoded pipeline's plan states the variant, so a cached
/// variant settles as the compiled one does.
#[test]
fn the_emulated_simdgroups_variant_round_trips() {
    use crate::bind::BindShape;
    use crate::cache::VariantKey;
    use crate::pipeline::facts::RuntimeFact;
    let dev = device("moltenvk-m3");
    let Some(bc) = fixture("simd_ops") else {
        return;
    };
    let f = Library::from_air(&bc)
        .and_then(|lib| lib.function("simd_sum_k"))
        .expect("kernel");
    let mut key = VariantKey::new();
    key.emulated_simdgroups = true;
    let compiled = dev
        .compile_function(
            &FunctionDescriptor::new(f),
            &key.lowering_inputs(Stage::Kernel),
            &Budget::standard(),
        )
        .expect("variant");
    assert!(compiled
        .artifact()
        .facts
        .contains(&RuntimeFact::EmulatedSimdgroups));
    assert_eq!(
        CompiledFunction::from_bytes(&compiled.to_bytes(), &dev),
        Ok(compiled.clone())
    );
    let pipeline = dev
        .compile_compute_pipeline(
            &ComputePipelineDescriptor::new(StageSource::Compiled(compiled)),
            &Budget::standard(),
        )
        .expect("pipeline");
    let decoded = ComputePipeline::from_bytes(&pipeline.to_bytes(), &dev).expect("decoded");
    assert_eq!(decoded, pipeline);
    let plan = decoded.plan(&dev, &BindShape::default()).expect("plan");
    assert_eq!(plan.variant, key);
}

/// The `SrgbTexture` fact round trips, and a kernel compiled under the `SrgbTexture` variant (a
/// slot bound to an 8-bit sRGB format the device holds as its UNORM twin) round trips as a
/// function and as a pipeline, whose decoded plan for a shape binding such a texture at the slot
/// states the variant, so a cached variant settles as the compiled one does; a shape binding an
/// sRGB format the device samples states none.
#[test]
fn the_srgb_texture_variant_round_trips() {
    use crate::bind::{BindState, BoundTexture, StageTable, TextureFacts};
    use crate::cache::VariantKey;
    use crate::pipeline::artifact::PrecisionNote;
    use crate::pipeline::facts::RuntimeFact;
    let slot = SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Texture,
        index: 2,
    };
    let fact = RuntimeFact::SrgbTexture { slot };
    assert_eq!(decode::<RuntimeFact>(&encode(&fact)), Ok(fact));
    let note = PrecisionNote::SrgbTexelDecode;
    assert_eq!(decode::<PrecisionNote>(&encode(&note)), Ok(note));

    let dev = device("lavapipe");
    let Some(bc) = fixture("texture_ops") else {
        return;
    };
    let f = Library::from_air(&bc)
        .and_then(|lib| lib.function("short_texels"))
        .expect("kernel");
    let mut key = VariantKey::new();
    key.srgb_textures.insert(slot);
    let compiled = dev
        .compile_function(
            &FunctionDescriptor::new(f),
            &key.lowering_inputs(Stage::Kernel),
            &Budget::standard(),
        )
        .expect("variant");
    assert!(compiled
        .artifact()
        .facts
        .contains(&RuntimeFact::SrgbTexture { slot }));
    assert_eq!(
        CompiledFunction::from_bytes(&compiled.to_bytes(), &dev),
        Ok(compiled.clone())
    );
    let pipeline = dev
        .compile_compute_pipeline(
            &ComputePipelineDescriptor::new(StageSource::Compiled(compiled)),
            &Budget::standard(),
        )
        .expect("pipeline");
    let decoded = ComputePipeline::from_bytes(&pipeline.to_bytes(), &dev).expect("decoded");
    assert_eq!(decoded, pipeline);
    let mut table = StageTable::default();
    // lavapipe samples no `RG8Unorm_sRGB`, which the runtime holds as `RG8Unorm`, and samples
    // `RGBA8Unorm_sRGB` itself.
    for (pf, twin) in [
        (metal::PixelFormat::RG8UnormSRGB, true),
        (metal::PixelFormat::RGBA8UnormSRGB, false),
    ] {
        let mut facts = TextureFacts::new(metal::TextureType::Type3D, pf);
        facts.usage = metal::TextureUsage::SHADER_READ;
        table.textures.insert(
            2,
            BoundTexture::Texture {
                facts,
                resource_id: None,
            },
        );
        let mut state = BindState::default();
        state.stages.insert(Stage::Kernel, table.clone());
        let plan = decoded.plan(&dev, &state.shape()).expect("plan");
        match twin {
            true => assert_eq!(plan.variant.srgb_textures, key.srgb_textures, "{pf:?}"),
            false => assert!(plan.variant.srgb_textures.is_empty(), "{pf:?}"),
        }
    }
}

/// The `CubicSampler` fact round trips, and a kernel compiled under the cubic-sampler variant round
/// trips as a function and as a pipeline, whose decoded plan for a shape binding a sampler with a
/// `Cubic` filter at the slot states the variant (the bound word's `SamplerWord::CUBIC_STATE`
/// bits), so a cached variant settles as the compiled one does. A nearest sampler at the slot
/// states none; every sample through the slot states the fact, with the variant or without it.
#[test]
fn the_cubic_sampler_variant_round_trips() {
    use crate::bind::{
        BindState, BoundSampler, BoundTexture, SamplerDescriptor, StageTable, TextureFacts,
    };
    use crate::cache::VariantKey;
    use crate::metal::sampler::SamplerWord;
    use crate::pipeline::facts::RuntimeFact;
    let slot = SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Sampler,
        index: 0,
    };
    let fact = RuntimeFact::CubicSampler { slot };
    assert_eq!(decode::<RuntimeFact>(&encode(&fact)), Ok(fact));

    let dev = device("lavapipe");
    let Some(bc) = fixture("gsm2_cubic_samplers") else {
        return;
    };
    let function = || {
        Library::from_air(&bc)
            .and_then(|lib| lib.function("gsm2_cubic_samples"))
            .expect("kernel")
    };
    let plain = dev
        .compile_function(
            &FunctionDescriptor::new(function()),
            &LoweringInputs::new(),
            &Budget::standard(),
        )
        .expect("plain");
    assert!(plain.artifact().facts.contains(&fact));
    let mut cubic = SamplerDescriptor::new();
    cubic.min_filter = metal::SamplerMinMagFilter::Cubic;
    cubic.mag_filter = metal::SamplerMinMagFilter::Linear;
    cubic.s_address_mode = metal::SamplerAddressMode::Repeat;
    cubic.t_address_mode = metal::SamplerAddressMode::ClampToBorderColor;
    cubic.border_color = metal::SamplerBorderColor::OpaqueWhite;
    let mut key = VariantKey::new();
    key.cubic_samplers
        .insert(slot, SamplerWord::of(&cubic).0 & SamplerWord::CUBIC_STATE);
    let compiled = dev
        .compile_function(
            &FunctionDescriptor::new(function()),
            &key.lowering_inputs(Stage::Kernel),
            &Budget::standard(),
        )
        .expect("variant");
    assert!(compiled.artifact().facts.contains(&fact));
    assert_eq!(
        CompiledFunction::from_bytes(&compiled.to_bytes(), &dev),
        Ok(compiled.clone())
    );
    let pipeline = dev
        .compile_compute_pipeline(
            &ComputePipelineDescriptor::new(StageSource::Compiled(compiled)),
            &Budget::standard(),
        )
        .expect("pipeline");
    let decoded = ComputePipeline::from_bytes(&pipeline.to_bytes(), &dev).expect("decoded");
    assert_eq!(decoded, pipeline);
    let shape = |desc: SamplerDescriptor| {
        let mut table = StageTable::default();
        let mut facts =
            TextureFacts::new(metal::TextureType::Type2D, metal::PixelFormat::RGBA8Unorm);
        facts.usage = metal::TextureUsage::SHADER_READ;
        table.textures.insert(
            0,
            BoundTexture::Texture {
                facts,
                resource_id: None,
            },
        );
        table.samplers.insert(
            0,
            BoundSampler {
                desc,
                resource_id: None,
            },
        );
        let mut state = BindState::default();
        state.stages.insert(Stage::Kernel, table);
        state.shape()
    };
    let plan = decoded.plan(&dev, &shape(cubic)).expect("plan");
    assert_eq!(plan.variant.cubic_samplers, key.cubic_samplers);
    let plan = decoded
        .plan(&dev, &shape(SamplerDescriptor::new()))
        .expect("plan");
    assert!(plan.variant.cubic_samplers.is_empty());
}

/// A render pipeline whose vertex function is compiled without its `Layer` output round trips,
/// keeps its `RenderTargetArrayLength` fact, and the decoded pipeline's plan for a pass of one
/// layer states the variant, so a cached variant settles as the compiled one does. The variant that
/// writes `Layer` keeps the length's push field, which the other lacks.
#[test]
fn the_dropped_vertex_layer_variant_round_trips() {
    use crate::bind::BindShape;
    use crate::cache::VariantKey;
    use crate::pipeline::facts::RuntimeFact;
    use crate::pipeline::sym::PushField;
    let dev = device("moltenvk-m3");
    let Some(bc) = fixture("vertex_amplification") else {
        return;
    };
    let f = Library::from_air(&bc)
        .and_then(|lib| lib.function("amp_vertex_layer"))
        .expect("vertex");
    let d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f)));
    for dropped in [false, true] {
        let mut key = VariantKey::new();
        key.dropped_vertex_layer = dropped;
        let pipeline = dev
            .compile_render_variant(&d, &key, &Budget::standard())
            .expect("pipeline");
        let decoded = RenderPipeline::from_bytes(&pipeline.to_bytes(), &dev).expect("decoded");
        assert_eq!(decoded, pipeline);
        assert!(decoded.dependencies.iter().any(|d| d.fact
            == RuntimeFact::RenderTargetArrayLength {
                layer_dropped: dropped
            }));
        let reads_length = decoded.interface.push.iter().any(|p| {
            p.fields
                .iter()
                .any(|f| f.field == PushField::RenderTargetArrayLength)
        });
        assert_eq!(reads_length, !dropped);
        let shape = BindShape {
            render_target_array_length: if dropped { 1 } else { 2 },
            ..BindShape::default()
        };
        let plan = decoded.plan(&dev, &shape).expect("plan");
        assert_eq!(plan.variant, key);
    }
}

/// A render pipeline compiled under the amplification draws variant round trips with its fact and
/// its amplification push fields, and the decoded pipeline's plan for a state whose amplification
/// multiview cannot express states the variant, so a cached variant settles as the compiled one
/// does.
#[test]
fn the_amplification_draws_variant_round_trips() {
    use crate::bind::{BindState, VertexAmplification, ViewMapping};
    use crate::cache::VariantKey;
    use crate::pipeline::facts::RuntimeFact;
    use crate::pipeline::sym::PushField;
    use crate::pipeline::AmplificationDraws;
    let dev = device("moltenvk-m3");
    let Some(bc) = fixture("vertex_amplification") else {
        return;
    };
    let lib = Library::from_air(&bc).expect("library");
    let stage = |name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            lib.function(name).expect("function"),
        ))
    };
    let mut d = RenderPipelineDescriptor::new(stage("amp_vertex"));
    d.fragment = Some(stage("amp_fragment"));
    d.max_vertex_amplification_count = 2;
    let mut key = VariantKey::new();
    key.amplification_draws = Some(AmplificationDraws::new(true, true));
    let pipeline = dev
        .compile_render_variant(&d, &key, &Budget::standard())
        .expect("pipeline");
    let decoded = RenderPipeline::from_bytes(&pipeline.to_bytes(), &dev).expect("decoded");
    assert_eq!(decoded, pipeline);
    assert!(decoded.dependencies.iter().any(|d| d.fact
        == RuntimeFact::AmplificationDraws {
            layered: true,
            viewports: true
        }));
    for field in [
        PushField::AmplificationId,
        PushField::AmplificationLayerOffset,
        PushField::AmplificationViewportOffset,
    ] {
        assert!(
            decoded
                .interface
                .push
                .iter()
                .any(|p| p.fields.iter().any(|f| f.field == field)),
            "{field:?}"
        );
    }
    let state = BindState {
        render_target_array_length: 2,
        vertex_amplification: VertexAmplification::new(
            2,
            vec![ViewMapping::new(1, 1), ViewMapping::new(0, 0)],
        ),
        ..Default::default()
    };
    let plan = decoded.plan(&dev, &state.shape()).expect("plan");
    assert_eq!(plan.variant, key);
}

/// A kernel compiled under the storage-view variant round trips with its uniform-view facts (kept,
/// viewed, spare, and the alias check), and the decoded pipeline's plan for a shape that binds a
/// large buffer at the kept slot states the variant, so a cached variant settles as the compiled
/// one does. Format version 158's members: a pipeline whose stage takes its combined image samplers
/// apart (`smp_runtime_array` on `moltenvk-m3`) round-trips its separate texture and separate
/// sampler bindings, and the symbols its relocations name.
#[test]
fn the_separate_bindings_round_trip() {
    use crate::pipeline::layout::BindingSource;
    let dev = device("moltenvk-m3");
    let Some(bc) = fixture("smp_runtime_array") else {
        return;
    };
    let f = Library::from_air(&bc)
        .and_then(|lib| lib.function("smp_runtime_array"))
        .expect("kernel");
    let d = ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f)));
    let pipeline = dev
        .compile_compute_pipeline(&d, &Budget::standard())
        .expect("pipeline");
    let sources: Vec<_> = pipeline.layout.sets[0]
        .bindings
        .iter()
        .map(|b| b.source)
        .collect();
    assert_eq!(
        sources
            .iter()
            .filter(|s| matches!(s, BindingSource::SeparateTexture { .. }))
            .count(),
        32
    );
    assert!(sources
        .iter()
        .any(|s| matches!(s, BindingSource::SeparateSampler { .. })));
    let decoded = ComputePipeline::from_bytes(&pipeline.to_bytes(), &dev).expect("decoded");
    assert_eq!(decoded, pipeline);
    assert_eq!(
        decoded.to_bytes(),
        pipeline.to_bytes(),
        "the encoding is canonical"
    );
    // Within the limit the same kernel keeps its combined bindings, which round-trip alike.
    let within = device("lavapipe");
    let pipeline = within
        .compile_compute_pipeline(&d, &Budget::standard())
        .expect("pipeline");
    assert_eq!(
        pipeline.layout.sets[0]
            .bindings
            .iter()
            .filter(|b| matches!(b.source, BindingSource::CombinedTextureSampler { .. }))
            .count(),
        32
    );
    let decoded = ComputePipeline::from_bytes(&pipeline.to_bytes(), &within).expect("decoded");
    assert_eq!(decoded, pipeline);
}

/// A combined binding's symbol and a separate texture's, whose index packs its use above its
/// texture slot, round-trip through the codec for every use, at the slots' bounds.
#[test]
fn combined_and_separate_texture_symbols_round_trip() {
    use crate::pipeline::sym::{CombinedSampler, ResourceKind, ResourceSlot, SamplerRef};
    for sampler in [
        SamplerRef::Slot(3),
        SamplerRef::Constexpr([u64::MAX >> 1, 7]),
    ] {
        for rank in 0..4u32 {
            let c = CombinedSampler::new(sampler, rank & 2 != 0, rank & 1 != 0);
            for texture in [0, 127, (1 << 28) - 1] {
                let combined = ResourceSlot {
                    stage: Stage::Fragment,
                    kind: ResourceKind::CombinedTextureSampler(c),
                    index: texture,
                };
                let separate =
                    ResourceSlot::separate_texture(Stage::Fragment, texture, c).expect("symbol");
                for s in [combined, separate] {
                    assert_eq!(decode::<ResourceSlot>(&encode(&s)), Ok(s));
                }
                assert_eq!(
                    decode::<ResourceSlot>(&encode(&separate)).map(|s| s.separate_texture_of()),
                    Ok(Some((texture, c)))
                );
            }
        }
    }
}

#[test]
fn the_storage_view_variant_round_trips() {
    use crate::bind::{BindState, BoundBuffer, StageTable};
    use crate::cache::VariantKey;
    use crate::pipeline::facts::{RuntimeFact, UniformViewChoice};
    let dev = device("vk12-floor");
    let Some(bc) = fixture("descriptor_limit_uniform_views") else {
        return;
    };
    let f = Library::from_air(&bc)
        .and_then(|lib| lib.function("descriptor_limit_uniform_views"))
        .expect("kernel");
    let slot = |index| SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Buffer,
        index,
    };
    let mut key = VariantKey::new();
    key.storage_view_slots.insert(slot(2));
    let compiled = dev
        .compile_function(
            &FunctionDescriptor::new(f),
            &key.lowering_inputs(Stage::Kernel),
            &Budget::standard(),
        )
        .expect("function");
    let pipeline = dev
        .compile_compute_pipeline(
            &ComputePipelineDescriptor::new(StageSource::Compiled(compiled)),
            &Budget::standard(),
        )
        .expect("pipeline");
    let decoded = ComputePipeline::from_bytes(&pipeline.to_bytes(), &dev).expect("decoded");
    assert_eq!(decoded, pipeline);
    let has = |fact| decoded.dependencies.iter().any(|d| d.fact == fact);
    let range = |index, choice| RuntimeFact::UniformViewRange {
        slot: slot(index),
        choice,
    };
    assert!(has(range(2, UniformViewChoice::Kept)));
    assert!(has(range(4, UniformViewChoice::Viewed)));
    assert!(has(range(0, UniformViewChoice::Spare)));
    assert!(has(RuntimeFact::UniformViewAliasing { slot: slot(3) }));
    let mut t = StageTable::default();
    for (index, length) in [
        (0, 256),
        (1, 256),
        (2, 1 << 16),
        (3, 16),
        (4, 256),
        (5, 256),
    ] {
        t.buffers.insert(
            index,
            BoundBuffer::Buffer {
                length,
                offset: 0,
                gpu_address: u64::from(index + 1) << 32,
                device_address: None,
                attribute_stride: None,
            },
        );
    }
    let mut state = BindState::default();
    state.stages.insert(Stage::Kernel, t);
    let plan = decoded.plan(&dev, &state.shape()).expect("plan");
    assert_eq!(plan.variant, key);
}

/// A body that passes the checksum is still decoded strictly (every byte used) and scanned: a
/// relocation off its decoration literal, a nonzero relocation slot, a broken header, a zero-word
/// instruction, or a mismatched key are `Corrupt` (the linear sanity scan).
#[test]
fn checksummed_bodies_are_still_scanned() {
    let dev = device("moltenvk-m3");
    let Some((compiled, _)) = store_const(&dev) else {
        return;
    };
    assert!(!compiled.artifact().module.relocs.is_empty());
    assert_eq!(
        CompiledFunction::from_bytes(&resealed(&compiled, |_| {}), &dev),
        Ok(compiled.clone())
    );
    let edits: Vec<(&str, Box<dyn Fn(&mut crate::library::FunctionArtifact)>)> = vec![
        (
            "reloc off its literal",
            Box::new(|a| a.module.relocs[0].1 -= 1),
        ),
        (
            "reloc past the end",
            Box::new(|a| a.module.relocs[0].1 = u32::MAX),
        ),
        (
            "duplicate reloc",
            Box::new(|a| {
                let r = a.module.relocs[0].clone();
                a.module.relocs.insert(0, r);
            }),
        ),
        (
            "nonzero reloc slot",
            Box::new(|a| {
                let at = a.module.relocs[0].1 as usize;
                a.module.words[at] = 3;
            }),
        ),
        ("magic", Box::new(|a| a.module.words[0] ^= 1)),
        ("version", Box::new(|a| a.module.words[1] = 0x0001_0600)),
        ("zero bound", Box::new(|a| a.module.words[3] = 0)),
        (
            "bound past instructions",
            Box::new(|a| a.module.words[3] = u32::MAX),
        ),
        (
            "zero-word instruction",
            Box::new(|a| a.module.words[5] &= 0xffff),
        ),
        (
            "instruction past the end",
            Box::new(|a| {
                let last = a.module.words.len() - 1;
                a.module.words[last] += 1 << 16;
            }),
        ),
    ];
    for (what, edit) in edits {
        let bytes = resealed(&compiled, |a| edit(a));
        assert_eq!(
            CompiledFunction::from_bytes(&bytes, &dev),
            Err(ArtifactError::Corrupt),
            "{what}"
        );
    }
    // A frame naming another key than its body.
    let bytes = seal(
        Kind::Function,
        &[1; 32],
        &dev.profile().fingerprint(),
        &encode(compiled.artifact()),
    );
    assert_eq!(
        CompiledFunction::from_bytes(&bytes, &dev),
        Err(ArtifactError::Corrupt)
    );
    // A trailing body byte, under a valid frame.
    let mut body = encode(compiled.artifact());
    body.push(0);
    let bytes = seal(
        Kind::Function,
        compiled.key().digest(),
        &dev.profile().fingerprint(),
        &body,
    );
    assert_eq!(
        CompiledFunction::from_bytes(&bytes, &dev),
        Err(ArtifactError::Corrupt)
    );
}

/// The decoder never trusts a count: a sequence longer than the bytes left, a non-canonical map
/// or set, a bool other than 0 or 1, and an unknown variant name are `Malformed`, without
/// allocating what the count names.
#[test]
fn the_codec_is_strict() {
    let mut huge = Vec::new();
    huge.extend_from_slice(&u32::MAX.to_le_bytes());
    assert_eq!(decode::<Vec<u64>>(&huge), Err(Malformed));
    assert_eq!(decode::<String>(&huge), Err(Malformed));
    assert_eq!(decode::<bool>(&[2]), Err(Malformed));

    let slot = |index| SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Buffer,
        index,
    };
    let set: std::collections::BTreeSet<SlotRef> = [slot(1), slot(2)].into();
    let bytes = encode(&set);
    assert_eq!(decode(&bytes), Ok(set));
    // Swap the two items: out of order.
    let item = (bytes.len() - 4) / 2;
    let mut swapped = bytes[..4].to_vec();
    swapped.extend_from_slice(&bytes[4 + item..]);
    swapped.extend_from_slice(&bytes[4..4 + item]);
    assert_eq!(
        decode::<std::collections::BTreeSet<SlotRef>>(&swapped),
        Err(Malformed)
    );
    // A duplicate item.
    let mut dup = bytes[..4].to_vec();
    dup.extend_from_slice(&bytes[4..4 + item]);
    dup.extend_from_slice(&bytes[4..4 + item]);
    assert_eq!(
        decode::<std::collections::BTreeSet<SlotRef>>(&dup),
        Err(Malformed)
    );

    let stage = encode(&Stage::Fragment);
    assert_eq!(stage, b"\x08Fragment");
    assert_eq!(decode::<Stage>(b"\x04Tile"), Err(Malformed));
    assert_eq!(decode::<Stage>(&stage[..3]), Err(Malformed));
    assert_eq!(
        Stage::get(&mut super::codec::Reader::new(&stage)),
        Ok(Stage::Fragment)
    );
}

/// A chunked encoding hands over the bytes `encode` gives, in pieces of at most `CHUNK` bytes,
/// except a byte sequence of at least `CHUNK`, which goes as the value's own slice, uncopied;
/// `seal_to` frames them as `seal` does; and a byte sequence decodes from the bytes.
#[test]
fn a_chunked_encoding_is_the_encoding() {
    use std::sync::Arc;
    let big: Arc<[u8]> = (0..3 * CHUNK + 17).map(|i| i as u8).collect();
    let words: Vec<u32> = (0..CHUNK as u32).collect();
    let value: (Vec<Arc<[u8]>>, (Vec<u32>, Box<[u8]>)) = (
        vec![Arc::from(&b"xy"[..]), Arc::clone(&big), Arc::from(&[][..])],
        (words, Box::from(&big[..CHUNK - 1])),
    );
    let whole = encode(&value);
    let mut chunks: Vec<Vec<u8>> = Vec::new();
    encode_chunked(&value, &mut |chunk| chunks.push(chunk.to_vec()));
    assert_eq!(chunks.concat(), whole);
    assert!(chunks.len() > 4, "{} chunks", chunks.len());
    for chunk in &chunks {
        assert!(
            chunk.len() <= CHUNK || chunk[..] == big[..],
            "{}",
            chunk.len()
        );
    }
    assert!(chunks.iter().any(|chunk| chunk[..] == big[..]));
    assert_eq!(decode(&whole), Ok(value.clone()));

    let mut framed = Vec::new();
    seal_to(
        Kind::ComputePipeline,
        &[0xaa; 32],
        &[0xbb; 32],
        &|sink| encode_chunked(&value, sink),
        &mut |chunk| framed.extend_from_slice(chunk),
    );
    assert_eq!(
        framed,
        seal(Kind::ComputePipeline, &[0xaa; 32], &[0xbb; 32], &whole)
    );

    // A byte sequence's count is checked against the bytes before any is copied.
    let mut short = encode(&big);
    short.pop();
    assert_eq!(decode::<Arc<[u8]>>(&short), Err(Malformed));
    assert_eq!(decode::<Vec<u8>>(&short), Err(Malformed));
}

/// A pipeline whose constant blob is non-zero and longer than a chunk (196,615 bytes) writes
/// exactly `to_bytes`' bytes, also through a writer that takes 7 bytes per call, and they decode
/// to the pipeline.
#[test]
fn a_pipeline_with_a_long_blob_writes_its_bytes() {
    /// A writer that takes at most 7 bytes per call.
    struct Short(Vec<u8>);
    impl std::io::Write for Short {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            let n = bytes.len().min(7);
            self.0.extend_from_slice(&bytes[..n]);
            Ok(n)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let dev = device("moltenvk-m3");
    let Some((_, mut p)) = store_const(&dev) else {
        return;
    };
    let blob: std::sync::Arc<[u8]> = (0..3 * CHUNK + 7).map(|i| (i % 251) as u8 + 1).collect();
    p.stage.constant_blob = Some(blob);
    let bytes = p.to_bytes();
    let mut whole = Vec::new();
    p.write_to(&mut whole).expect("write");
    assert_eq!(whole, bytes);
    let mut short = Short(Vec::new());
    p.write_to(&mut short).expect("write");
    assert_eq!(short.0, bytes);
    assert_eq!(ComputePipeline::from_bytes(&bytes, &dev), Ok(p));
}

/// The frame layout is stable: a change fails here and needs an `ARTIFACT_FORMAT_VERSION` bump.
#[test]
fn the_frame_layout_is_pinned() {
    let bytes = seal(Kind::RenderPipeline, &[0xaa; 32], &[0xbb; 32], b"body");
    let mut expected = Vec::new();
    expected.extend_from_slice(b"M2VARTF\0");
    expected.extend_from_slice(&ARTIFACT_FORMAT_VERSION.to_le_bytes());
    expected.extend_from_slice(&SEMANTICS_VERSION.to_le_bytes());
    expected.extend_from_slice(&3u32.to_le_bytes());
    expected.extend_from_slice(&[0xaa; 32]);
    expected.extend_from_slice(&[0xbb; 32]);
    expected.extend_from_slice(&4u64.to_le_bytes());
    expected.extend_from_slice(b"body");
    assert_eq!(bytes[..bytes.len() - 32], expected[..]);
    use sha2::Digest;
    assert_eq!(
        bytes[bytes.len() - 32..],
        sha2::Sha256::digest(&expected)[..]
    );
}

/// The scan accepts what the serializer writes and refuses a module shorter than its header.
#[test]
fn the_scan_bounds_the_header() {
    assert_eq!(scan_words(&[]), Err(Malformed));
    assert_eq!(
        scan_words(&[crate::spv::MAGIC_NUMBER, 0x0001_0500, 0, 1]),
        Err(Malformed)
    );
    // Header only, bound 1: no ids.
    assert_eq!(
        scan_words(&[crate::spv::MAGIC_NUMBER, 0x0001_0500, 0, 1, 0]),
        Ok(())
    );
    let dev = device("moltenvk-m3");
    let Some((compiled, pipe)) = store_const(&dev) else {
        return;
    };
    assert_eq!(scan_relocatable(&compiled.artifact().module), Ok(()));
    assert_eq!(scan_words(&pipe.stage.spirv), Ok(()));
}

/// A pipeline's content digest is what the pipeline says: it holds through the encoding and under
/// another key or profile fingerprint (the cache identity, which a semantics bump moves), and moves
/// with a word of SPIR-V, the layout, or a note. Compute and render roots differ.
#[test]
fn a_content_digest_is_the_content_not_the_key() {
    use crate::pipeline::artifact::PrecisionNote;
    let dev = device("moltenvk-m3");
    let Some((_, pipe)) = store_const(&dev) else {
        return;
    };
    let digest = pipe.content_digest();
    let decoded = ComputePipeline::from_bytes(&pipe.to_bytes(), &dev).expect("decode");
    assert_eq!(decoded.content_digest(), digest);

    let mut renamed = pipe.clone();
    renamed.key = PipelineKey::new([7; 32]);
    renamed.profile = [9; 32];
    assert_eq!(renamed.content_digest(), digest);

    let mut edited = pipe.clone();
    let mut words = edited.stage.spirv.to_vec();
    let last = words.len() - 1;
    words[last] ^= 1;
    edited.stage.spirv = words.into();
    assert_ne!(edited.content_digest(), digest);
    let mut edited = pipe.clone();
    edited.layout.push_descriptor_eligible = !edited.layout.push_descriptor_eligible;
    assert_ne!(edited.content_digest(), digest);
    let mut edited = pipe.clone();
    edited
        .stage
        .precision_notes
        .push(PrecisionNote::ReductionOrder);
    assert_ne!(edited.content_digest(), digest);

    let Some(bc) = fixture("stage_io") else {
        return;
    };
    let f = Library::from_air(&bc)
        .expect("library")
        .function("stage_io_vertex_position")
        .expect("vertex");
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f)));
    d.rasterization_enabled = false;
    let render = dev
        .compile_render_pipeline(&d, &Budget::standard())
        .expect("pipeline");
    let digest = render.content_digest();
    let decoded = RenderPipeline::from_bytes(&render.to_bytes(), &dev).expect("decode");
    assert_eq!(decoded.content_digest(), digest);
    let mut renamed = render.clone();
    renamed.key = PipelineKey::new([7; 32]);
    renamed.profile = [9; 32];
    assert_eq!(renamed.content_digest(), digest);
    let mut edited = render.clone();
    edited.graphics.alpha_to_one = !edited.graphics.alpha_to_one;
    assert_ne!(edited.content_digest(), digest);
}

/// A pipeline state's creation compile on a device that reads storage textures only with a declared
/// format takes stand-ins (`runtime::pipeline::stand_in`: the first of `R32Float`, `R32Uint` and
/// `R32Sint` each slot's texels take). On ANV on Iris Xe (RPL-P), the compile of
/// `a2_storage_read_formats` (a float, a uint and an int `read_write` texture, read) under those
/// stand-ins is the compile of the bind of those formats: its plan for that bind names its own key,
/// and the bind's compile has the same function and pipeline cache keys (what the disk cache stores
/// and loads them under) and byte-identical artifacts, so the compile carries nothing of the
/// stand-in beyond its key. Every bind of another format at any of the slots plans another key,
/// except where the plan does not bind that texture at all: a texture the slot cannot take is
/// replaced by the slot's neutral (`IncompatibleBinding`, `CrossClassNeutral` for a storage read
/// of another numeric class, or `AstcStorageNeutral` for an ASTC texture), an `R32` texture of the
/// slot's texel type, whose compile is the stand-in's.
#[test]
fn a_stand_in_compile_is_the_compile_of_its_formats_bind_and_no_other() {
    use crate::bind::{BindState, BoundTexture, PlanDiagnostic, StageTable, TextureFacts};
    use crate::cache::VariantKey;
    use crate::error::{Error, RuntimeInput};
    use crate::metal::format::storage_image_choice;
    use crate::metal::{PixelFormat, TextureType, TextureUsage};

    let Some(bc) = fixture("a2_storage_read_formats") else {
        return;
    };
    let dev = device("intel-irisxe-rplp-anv");
    let profile = dev.profile();
    assert!(!profile.fields().images.storage_read_without_format);
    let descriptor = FunctionDescriptor::new(
        Library::from_air(&bc)
            .and_then(|lib| lib.function("a2_storage_read_formats"))
            .expect("kernel"),
    );
    let budget = Budget::standard();
    // The runtime's compile of a variant (`Variant::compiled`, without the cache).
    let compile = |key: &VariantKey| -> Result<(CompiledFunction, ComputePipeline), Error> {
        let inputs = key.lowering_inputs(Stage::Kernel);
        let function = dev.compile_function(&descriptor, &inputs, &budget)?;
        assert_eq!(
            *function.key(),
            dev.function_compile_key(&descriptor, &inputs)
        );
        let pipeline = dev.compile_compute_pipeline(
            &ComputePipelineDescriptor::new(StageSource::Compiled(function.clone())),
            &budget,
        )?;
        Ok((function, pipeline))
    };
    let slot = |index: u32| SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Texture,
        index,
    };
    assert!(matches!(
        compile(&VariantKey::new()),
        Err(Error::NeedsInput(RuntimeInput::StorageTextureFormat { .. }))
    ));
    let formats = [
        (0, PixelFormat::R32Float),
        (1, PixelFormat::R32Uint),
        (2, PixelFormat::R32Sint),
    ];
    let mut stand_in_key = VariantKey::new();
    for (index, pf) in formats {
        let choice = storage_image_choice(pf, profile).expect("a stand-in's choice");
        stand_in_key.storage_images.insert(slot(index), choice);
    }
    let (stand_in_function, stand_in) = compile(&stand_in_key).expect("the stand-in compile");
    // The plan of a bind of `bound`, as the runtime's `settle` reads it off the creation compile.
    let planned = |bound: &[(u32, PixelFormat)]| {
        let mut table = StageTable::default();
        for &(index, pf) in bound {
            let mut facts = TextureFacts::new(TextureType::Type2D, pf);
            facts.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
            let texture = BoundTexture::Texture {
                facts,
                resource_id: None,
            };
            table.textures.insert(index, texture);
        }
        let mut state = BindState::default();
        state.stages.insert(Stage::Kernel, table);
        stand_in.plan(&dev, &state.shape())
    };
    let own = planned(&formats).expect("the stand-ins' bind plans");
    assert!(own.diagnostics.is_empty(), "{:?}", own.diagnostics);
    assert_eq!(own.variant, stand_in_key);
    let (own_function, own_pipeline) = compile(&own.variant).expect("the bind's compile");
    assert_eq!(own_function.key(), stand_in_function.key());
    assert_eq!(own_pipeline.key(), stand_in.key());
    assert_eq!(own_function.to_bytes(), stand_in_function.to_bytes());
    assert_eq!(own_pipeline.to_bytes(), stand_in.to_bytes());
    // Any other format a texture can have, at any slot: another key, or the slot's neutral.
    let (mut other_keys, mut replaced) = (0, 0);
    let mut same = Vec::new();
    for at in 0..formats.len() {
        for &pf in metal::PixelFormat::ALL {
            if pf == formats[at].1 || pf == PixelFormat::Invalid {
                continue;
            }
            let mut bound = formats;
            bound[at].1 = pf;
            let Ok(plan) = planned(&bound) else {
                continue;
            };
            let neutral = plan.diagnostics.iter().any(|d| match d {
                PlanDiagnostic::IncompatibleBinding { slot: s, .. }
                | PlanDiagnostic::CrossClassNeutral { slot: s, .. }
                | PlanDiagnostic::AstcStorageNeutral { slot: s, .. } => *s == slot(bound[at].0),
                _ => false,
            });
            if neutral {
                replaced += 1;
            } else if plan.variant == stand_in_key {
                same.push((at, pf));
            } else {
                other_keys += 1;
            }
        }
    }
    assert!(
        same.is_empty(),
        "binds that plan the stand-ins' key: {same:?}"
    );
    // An ASTC texture, of every profile and footprint, at a storage slot: the slot's neutral, and
    // the plan says why (`PlanDiagnostic::AstcStorageNeutral`), which the runtime announces.
    let astc = metal::PixelFormat::ALL
        .iter()
        .filter(|&&pf| crate::bind::AstcDecode::of(pf).is_some());
    for &pf in astc {
        let mut bound = formats;
        bound[0].1 = pf;
        let plan = planned(&bound).unwrap_or_else(|e| panic!("{pf:?}: {e}"));
        let said = plan.diagnostics.iter().any(
            |d| matches!(d, PlanDiagnostic::AstcStorageNeutral { slot: s, .. } if *s == slot(0)),
        );
        assert!(said, "{pf:?}: {:?}", plan.diagnostics);
    }
    assert!(other_keys > 0, "no other format plans");
    eprintln!(
        "{other_keys} binds of other formats plan other keys; {replaced} are replaced by the \
         slot's neutral"
    );
}

/// Kernel buffer slot `index`.
fn kernel_buffer(index: u32) -> SlotRef {
    SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Buffer,
        index,
    }
}

/// A variant key with device-address kernel buffer slots `slots`.
fn addressed(slots: &[u32]) -> super::VariantKey {
    let mut key = super::VariantKey::new();
    key.psb_slots = slots.iter().copied().map(kernel_buffer).collect();
    key
}

/// The large-buffer form's variant runs every dispatch or draw whose plan asks for fewer
/// device-address slots, and no other variant's.
#[test]
fn the_large_buffer_variant_covers_plans_of_fewer_device_address_slots() {
    let large = addressed(&[0, 3]);
    assert!(large.covers(&large));
    assert!(large.covers(&addressed(&[3])));
    assert!(large.covers(&super::VariantKey::new()));
    assert!(!large.covers(&addressed(&[1])));
    assert!(!addressed(&[3]).covers(&large));
    let mut emulated = addressed(&[3]);
    emulated.emulated_simdgroups = true;
    assert!(!large.covers(&emulated));
}

/// The large-buffer form moves every slot whose range a large buffer exceeds to its device address,
/// unless the shape binds it without one (bytes), keeping what the key held; on a device without
/// device addresses (the fact a plan-time check) it moves none, so a variant still follows the
/// bound buffers there.
#[test]
fn the_large_buffer_form_moves_the_range_limited_slots_to_device_addresses() {
    use crate::bind::{BindShape, BufferShape, StageShape};
    use crate::pipeline::{Mechanism, RuntimeDependency, RuntimeFact};
    let limited = |index, mechanism| {
        let slot = kernel_buffer(index);
        RuntimeDependency::new(RuntimeFact::BufferRangeLimit { slot }, mechanism)
    };
    let addressed_device = [
        limited(0, Mechanism::Variant),
        limited(2, Mechanism::Variant),
        RuntimeDependency::new(
            RuntimeFact::ThreadsPerThreadgroup,
            Mechanism::Specialization,
        ),
    ];
    let unbound = BindShape::default();
    let empty = super::VariantKey::new();
    assert_eq!(
        empty.large_buffer_form(&addressed_device, &unbound),
        addressed(&[0, 2])
    );
    let mut stage = StageShape::default();
    stage.buffers.insert(
        2,
        BufferShape::Bytes {
            attribute_stride: None,
        },
    );
    let mut bytes = BindShape::default();
    bytes.stages.insert(Stage::Kernel, stage);
    assert_eq!(
        empty.large_buffer_form(&addressed_device, &bytes),
        addressed(&[0])
    );
    let mut emulated = addressed(&[5]);
    emulated.emulated_simdgroups = true;
    let mut widened = addressed(&[0, 5]);
    widened.emulated_simdgroups = true;
    assert_eq!(
        emulated.large_buffer_form(&addressed_device, &bytes),
        widened
    );
    let floor_device = [limited(0, Mechanism::PlanTimeCheck)];
    assert_eq!(empty.large_buffer_form(&floor_device, &unbound), empty);
}

/// A render descriptor's linked-use answers and reflection decode as they were encoded, under
/// their descriptor key on the device they were encoded for; another key or profile is
/// `WrongDevice`, another kind of frame `Corrupt`, a changed byte `Corrupt`.
#[test]
fn a_descriptors_answers_and_reflection_round_trip_under_its_key() {
    use crate::reflection::{
        DirectArgumentUse, LinkedUseAnswers, LinkedUseGap, UnwrittenFragmentInputs,
    };
    let dev = device("moltenvk-m3");
    let other = device("vk12-floor");
    let key = PipelineKey::new([3; 32]);
    let wrong_key = PipelineKey::new([4; 32]);
    let mut answers = LinkedUseAnswers::default();
    let back = |bytes: &[u8]| LinkedUseAnswers::from_bytes_for_key(bytes, &key, &dev);
    assert_eq!(back(&answers.to_bytes(&key, &dev)), Ok(answers.clone()));
    answers.vertex_returns_value = Some(false);
    answers.fragment_writes_depth_unattached = Some(true);
    answers.stages_using_direct_arguments = Some(vec![
        DirectArgumentUse {
            stage: Stage::Vertex,
            gap: None,
        },
        DirectArgumentUse {
            stage: Stage::Fragment,
            gap: Some(LinkedUseGap::UnmeasuredColorWrite),
        },
    ]);
    answers.unwritten_fragment_inputs = Some(Some(UnwrittenFragmentInputs {
        names: vec!["color".into(), "uv".into()],
        gap: Some(LinkedUseGap::StackMemory),
    }));
    let bytes = answers.to_bytes(&key, &dev);
    assert_eq!(back(&bytes), Ok(answers.clone()));
    answers.unwritten_fragment_inputs = Some(None);
    assert_eq!(back(&answers.to_bytes(&key, &dev)), Ok(answers.clone()));
    assert_eq!(
        LinkedUseAnswers::from_bytes_for_key(&bytes, &wrong_key, &dev),
        Err(ArtifactError::WrongDevice)
    );
    assert_eq!(
        LinkedUseAnswers::from_bytes_for_key(&bytes, &key, &other),
        Err(ArtifactError::WrongDevice)
    );
    let mut changed = bytes.clone();
    let at = changed.len() / 2;
    changed[at] ^= 1;
    assert_eq!(back(&changed), Err(ArtifactError::Corrupt));
    assert_eq!(
        crate::reflection::RenderPipelineReflection::from_bytes_for_key(&bytes, &key, &dev),
        Err(ArtifactError::Corrupt),
        "answers are not a reflection"
    );

    let Some(bc) = fixture("stage_io") else {
        return;
    };
    let lib = Library::from_air(&bc).expect("library");
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(
        lib.function("stage_io_vertex_position").expect("vertex"),
    )));
    d.fragment = Some(StageSource::Function(FunctionDescriptor::new(
        lib.function("stage_io_fragment_primitive")
            .expect("fragment"),
    )));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(
        metal::PixelFormat::RGBA8Unorm,
    ));
    let reflection = d.reflection(&Budget::standard()).expect("reflection");
    let key = dev.render_descriptor_key(&d);
    let bytes = reflection.to_bytes(&key, &dev);
    assert_eq!(
        crate::reflection::RenderPipelineReflection::from_bytes_for_key(&bytes, &key, &dev),
        Ok(reflection.clone())
    );
    assert_eq!(
        crate::reflection::RenderPipelineReflection::from_bytes_for_key(&bytes, &wrong_key, &dev),
        Err(ArtifactError::WrongDevice)
    );
    assert_eq!(
        LinkedUseAnswers::from_bytes_for_key(&bytes, &key, &dev),
        Err(ArtifactError::Corrupt),
        "a reflection is not answers"
    );
}
