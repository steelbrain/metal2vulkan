//! Every state of one mesh render descriptor shares one reflection
//! (`pipeline/render.rs`: `Device::new_mesh_render_pipeline_state_with_reflection` and
//! `GraphicsState::reflection`).
//!
//! # What this guards
//!
//! `MeshPipelineDescriptor::reflection` decodes the object, mesh and fragment functions' AIR. A
//! drawing pipeline's reflection goes through the shader cache under
//! `Device::render_descriptor_key`, so every state of one descriptor shares the one answer; a mesh
//! pipeline's reached neither the cache nor any memo, so each state of one descriptor decoded all
//! three stages again, and so did every read of the state's own `compiled_reflection`. Metal
//! compiles the library once and answers each state from it.
//!
//! The check is a **count** of the reflections the device worked out, not a time: the count holds
//! on any host under any load. It is of this device, whose shader cache is given a directory of
//! its own below, so it depends on nothing that ran before it in the process or on disk -- which a
//! count read off a shared cache directory would.
//!
//! It is held in both directions. Over the count is a key too coarse or a memo that stopped
//! answering -- and a key too coarse is the worse fault, because two descriptors would then share
//! one reflection, which is wrong and not merely slow, so the second test below reads two
//! descriptors and requires two. Under it is a derivation that no longer goes through the cache at
//! all, so nothing counts it, which is what removing the routing reads as; it fails too rather
//! than looking like an improvement.

use crate::{Device, DeviceOptions, PipelineOption, PipelineOptions, ShaderCacheOptions};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, FunctionDescriptor, MeshPipelineDescriptor, StageSource,
};
use std::path::{Path, PathBuf};

/// A scratch directory for the device's shader cache, removed when dropped: the count below is of
/// one device's work, so it must not read what an earlier run of this test wrote.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Scratch {
        let path = std::env::temp_dir().join(format!(
            "m2v-mesh-reflection-derived-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Scratch(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The `mesh_draws` fixture's object, mesh and fragment functions as a mesh render descriptor
/// with two `RGBA32Uint` targets: three module-sourced stages, so its reflection comes from their
/// AIR, which is what a per-state derivation pays for.
fn descriptor() -> MeshPipelineDescriptor {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/engine/mesh_draws.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    let stage =
        |name| StageSource::Function(FunctionDescriptor::new(library.function(name).unwrap()));
    let mut descriptor = MeshPipelineDescriptor::new(stage("mesh_draws_mesh"));
    descriptor.object = Some(stage("mesh_draws_object"));
    descriptor.fragment = Some(stage("mesh_draws_fragment"));
    for color in &mut descriptor.color_attachments[..2] {
        *color = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
    }
    descriptor
}

/// A device whose shader cache has `directory` to itself, so the reflection count below is of this
/// device's work alone.
fn device(directory: &Path) -> Device {
    let options = DeviceOptions {
        shader_cache: ShaderCacheOptions {
            directory: Some(directory.to_owned()),
            ..ShaderCacheOptions::default()
        },
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// Two states of one mesh descriptor, each asked for its reflection, and each state's own
/// `compiled_reflection` read twice: the device works out one reflection.
#[test]
fn every_state_of_one_mesh_descriptor_shares_one_reflection() {
    let directory = Scratch::new();
    let device = device(&directory.0);
    let descriptor = descriptor();
    let options = PipelineOptions::new(PipelineOption::BINDING_INFO);
    let first = match device.new_mesh_render_pipeline_state_with_reflection(&descriptor, &options) {
        Ok((state, reflection)) => {
            assert!(reflection.is_some(), "a reflecting creation answers one");
            state
        }
        // A device without mesh shading refuses the creation before any reflection, so there is
        // nothing to count; the gap this guards cannot arise on it.
        Err(error) => {
            eprintln!("mesh states are not made on this device ({error:?}): nothing to count");
            return;
        }
    };
    assert_eq!(
        device.shared().shader_cache.reflections_derived(),
        1,
        "the first reflecting creation works out one reflection"
    );
    let (second, reflection) = device
        .new_mesh_render_pipeline_state_with_reflection(&descriptor, &options)
        .unwrap();
    assert!(reflection.is_some());
    for state in [&first, &second] {
        for _ in 0..2 {
            state.compiled_reflection().unwrap();
        }
    }
    assert_eq!(
        device.shared().shader_cache.reflections_derived(),
        1,
        "a second state of the same descriptor, and four reads of the two states' own \
         compiled_reflection, work out none: all of them answer from the one"
    );
}

/// Two mesh descriptors that differ derive two reflections: the key tells them apart, so no
/// descriptor is ever answered another's reflection. This is the half a count catches on the high
/// side, and the fault it guards against -- a key too coarse -- is a wrong answer, not a slow one.
#[test]
fn two_mesh_descriptors_derive_their_own_reflections() {
    let directory = Scratch::new();
    let device = device(&directory.0);
    let options = PipelineOptions::new(PipelineOption::BINDING_INFO);
    // The same object and mesh stages, with and without the fragment stage, whose bindings the
    // reflection lists: two descriptors whose reflections genuinely differ.
    let with_fragment = descriptor();
    let mut without_fragment = descriptor();
    without_fragment.fragment = None;
    let mut reflections = Vec::new();
    for descriptor in [&with_fragment, &without_fragment] {
        match device.new_mesh_render_pipeline_state_with_reflection(descriptor, &options) {
            Ok((_, reflection)) => reflections.push(reflection.unwrap()),
            Err(error) => {
                eprintln!("mesh states are not made on this device ({error:?}): nothing to count");
                return;
            }
        }
    }
    assert_eq!(
        device.shared().shader_cache.reflections_derived(),
        2,
        "two descriptors that differ each work out their own reflection"
    );
    assert_ne!(
        reflections[0].fragment_bindings(),
        reflections[1].fragment_bindings(),
        "the descriptor without a fragment stage reflects no fragment bindings, so a key that \
         shared one reflection between the two would have answered one of them wrongly"
    );
}
