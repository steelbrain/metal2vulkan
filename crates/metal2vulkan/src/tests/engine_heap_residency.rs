//! The residency of a texture an argument buffer holds.
//!
//! The owned fixture `engine/tail_argument_sparse_samples` samples the material texture of its
//! constant argument buffer 0 with `sparse_sample` through a constexpr sampler that filters
//! linearly between levels with anisotropy: `argument_sparse_fragment` returns the value where
//! `resident()` holds and else the sample clamped by the material's `min_lod_clamp`, the two
//! samples' `{half4, i8}` results merged in a `phi` and the word tested `(status & 1) == 0`;
//! `argument_sparse_residency_fragment` returns the residency it read beside the value.
//!
//! Each fragment compiles behind `argument_sparse_vertex` wherever the device has what it uses,
//! its heap access answering the word of a texture that is not sparse (resident) and the stage
//! stating `RuntimeFact::HeapResidency`; before, the observed word of a heap element was refused
//! (`Lowering(Texture)`, `NotImplemented`). Its plan holds a texture that is not sparse in the
//! heap, and refuses a sparse one the fragment's handle table reaches, whose residency the
//! module does not answer. No GPU is used.

use std::path::Path;

use crate::bind::{BindState, ResourceClass, TextureFacts, UsedResource};
use crate::budget::Budget;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{BindError, Error, Feature, Needs, Unsupported};
use crate::library::Library;
use crate::metal::{PixelFormat, RenderStages, TextureType, TextureUsage};
use crate::pipeline::artifact::RenderPipeline;
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::sym::Stage;
use crate::pipeline::testing::{engine_dir, llvm_assemble, AssembleError};
use crate::spirv::testing::{check_words_valid, committed_profiles};

/// The fixture's fragment functions.
const FRAGMENTS: [&str; 2] = [
    "argument_sparse_fragment",
    "argument_sparse_residency_fragment",
];

/// The committed profiles each fragment must compile on.
const COMPILES_ON: [&str; 3] = ["moltenvk-m3", "lavapipe", "amd-780m-radv"];

/// The fixture's library, assembled with `llvm-as`; `None` where the tool is missing (the test
/// then skips visibly).
fn fixture_library() -> Option<Library> {
    match llvm_assemble(&engine_dir().join("tail_argument_sparse_samples.ll")) {
        Ok(bc) => Some(Library::from_air(&bc).expect("library")),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => {
            panic!("llvm-as rejected tail_argument_sparse_samples: {why}")
        }
    }
}

/// The device of committed profile `stem`.
fn device(stem: &str) -> Device {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../metal2vulkan-validation/profiles/{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::new(desc, DevicePolicy::default()).expect("device")
}

/// The pipeline of fragment `entry` behind `argument_sparse_vertex`, color 0 `RGBA16Float`.
fn pipeline(dev: &Device, library: &Library, entry: &str) -> Result<RenderPipeline, Error> {
    let function = |name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            library.function(name).expect("function"),
        ))
    };
    let mut d = RenderPipelineDescriptor::new(function("argument_sparse_vertex"));
    d.fragment = Some(function(entry));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float));
    dev.compile_render_pipeline(&d, &Budget::standard())
}

/// Whether `pipe` states the fragment's heap residency.
fn states_residency(pipe: &RenderPipeline) -> bool {
    pipe.dependencies.iter().any(|d| {
        d.fact
            == RuntimeFact::HeapResidency {
                stage: Stage::Fragment,
            }
    })
}

/// A 2D `RGBA16Float` texture the fragment stage reads, sparse where `sparse`.
fn material(sparse: bool) -> UsedResource {
    let mut facts = TextureFacts::new(TextureType::Type2D, PixelFormat::RGBA16Float);
    facts.usage = TextureUsage::SHADER_READ;
    facts.sparse = sparse;
    let mut u = UsedResource::new(ResourceClass::Texture(facts), 1);
    u.resource_id = Some(0x5150_0001);
    u.stages = RenderStages::FRAGMENT;
    u
}

#[test]
fn a_heap_textures_observed_residency_compiles_and_a_sparse_member_is_refused() {
    let Some(library) = fixture_library() else {
        return;
    };
    let mut compiled = Vec::new();
    for (stem, _) in committed_profiles().expect("profiles") {
        let dev = device(&stem);
        for entry in FRAGMENTS {
            let what = format!("{stem} {entry}");
            match pipeline(&dev, &library, entry) {
                Ok(pipe) => {
                    assert!(states_residency(&pipe), "{what}: {:?}", pipe.dependencies);
                    for s in &pipe.stages {
                        check_words_valid(&s.spirv, &pipe.requirements, dev.profile())
                            .unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                    }
                    compiled.push(what);
                }
                // A device that lacks what the fragment uses (`shaderResourceMinLod` for the
                // clamp, or the heap's own needs) refuses it by name.
                Err(Error::Unsupported(Unsupported {
                    needs: Needs::AnyOf(_),
                    ..
                })) => assert!(!COMPILES_ON.contains(&stem.as_str()), "{what}"),
                Err(e) => panic!("{what}: {e}"),
            }
        }
    }
    for stem in COMPILES_ON {
        for entry in FRAGMENTS {
            let what = format!("{stem} {entry}");
            assert!(compiled.contains(&what), "{what} did not compile");
        }
    }
    // The plan: a texture that is not sparse is a member, a sparse one the fragment reaches is
    // refused.
    let dev = device("moltenvk-m3");
    for entry in FRAGMENTS {
        let pipe = pipeline(&dev, &library, entry).expect("pipeline");
        let state = |sparse: bool| BindState {
            used_resources: vec![material(sparse)],
            ..Default::default()
        };
        pipe.plan(&dev, &state(false).shape())
            .unwrap_or_else(|e| panic!("{entry}: {e}"));
        match pipe.plan(&dev, &state(true).shape()) {
            Err(BindError::Unsupported(u)) => {
                assert_eq!(u.feature, Feature::ArgumentBufferTextures, "{entry}");
                assert_eq!(u.needs, Needs::NotImplemented, "{entry}");
            }
            Ok(_) => panic!("{entry}: a sparse member is planned"),
            Err(e) => panic!("{entry}: {e}"),
        }
    }
}
