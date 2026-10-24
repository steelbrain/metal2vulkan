//! Every draw that binds a read-only buffer neutral shares the device's one zero buffer
//! (`neutral.rs`, and `binding::descriptor`'s `WriteSource::Neutral` arm).
//!
//! # What this guards
//!
//! A plan binds `NeutralResource::ZeroBuffer` where a function reads a buffer binding that nothing
//! is bound to. Its length is `maxUniformBufferRange` capped at 64 KiB, which the device's own
//! limits fix, and its contents are zeros: neither can have changed since the device was made.
//! It was nevertheless taken as a fresh zero-filled region from the command buffer's upload arena
//! at **every draw and dispatch that bound one**, and a region that long is past half the arena's
//! chunk, so each took a dedicated 64 KiB `VkBuffer` of its own, allocated, memory-bound and
//! written to zero. A triangle drawn with neither of its two `setBytes` slots set cost two of
//! them -- 128 KiB of allocate-and-zero a draw, which against a frame of draws is the whole of the
//! budget rather than a part of it.
//!
//! The device now makes one zero buffer per length its plans ask for and keeps it for its
//! lifetime, as it keeps its neutral images, texel buffers and samplers.
//!
//! # Why only a read-only binding
//!
//! One buffer shared by every command of the device is sound only where no shader can store
//! through the binding. A Vulkan uniform-buffer descriptor is read-only by construction, so that
//! is the binding served from it (`binding::read_only_buffer`, decided from the descriptor type
//! alone). A writable binding keeps a region of its own, exactly as every binding did, so nothing
//! a shader stores is ever read by another command.
//!
//! # Counts, not times
//!
//! Every assertion is a count: the zero buffers the device made, and the deltas of work counters
//! the runtime already keeps. A count holds on any host under any load. Each is read as a
//! difference across this test's own device, whose shader cache has a directory of its own, so it
//! depends on nothing that ran before it.
//!
//! Held in both directions. `neutral_zeros_made` must be exactly **1**: more than one is a memo
//! that stopped answering, and **zero** means no neutral was bound at all, so the test would be
//! proving nothing about the path it names. What it does not cover is the writable half -- no
//! fixture here leaves a storage buffer binding unbound -- and the dispatch path, which binds
//! neutrals by the same code and is unexercised by these draws.

use crate::{
    Device, DeviceKind, DeviceOptions, LoadAction, PipelineOptions, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions, ShaderCacheOptions, StoreAction, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::path::{Path, PathBuf};

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Scratch {
        let path = std::env::temp_dir().join(format!(
            "m2v-zz-encode-probe-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
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

fn device(cache: &Path) -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

fn state(device: &Device, fragment: &str) -> RenderPipelineState {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/engine/tob_renders.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    let library = device.new_library_with_compile_library(library);
    let vertex = library.new_function_with_name("tob_rect").unwrap();
    let fragment = library.new_function_with_name(fragment).unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
    descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// Draw `draws` identical triangles in one pass, neither of the pipeline's two `setBytes` slots
/// set, so every draw binds the read-only zero neutral at both. Answer the device's zero-buffer
/// count and what the work counters moved.
fn drawn(draws: usize) -> (u64, crate::WorkCounts, crate::WorkCounts) {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let solid = state(&device, "tob_solid");
    let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 32, 32, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::shared();
    let target = device.new_texture(&d).unwrap();
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Load;
    color.store_action = StoreAction::Store;
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(&solid).unwrap();
    let before = device.work_counts();
    for _ in 0..draws {
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
    }
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer
        .wait_until_completed_for(std::time::Duration::from_secs(60))
        .unwrap();
    assert_eq!(command_buffer.error(), None, "the pass ran");
    let after = device.work_counts();
    (device.neutral_zeros_made(), before, after)
}

/// One draw and sixty-four draws each make the device's zero buffer once, and neither allocates,
/// binds or zeroes a `VkBuffer` to bind a neutral.
#[test]
fn draws_share_the_devices_one_zero_buffer() {
    for draws in [1, 64] {
        let (made, before, after) = drawn(draws);
        assert_eq!(
            made, 1,
            "{draws} draws made {made} zero buffers, not 1: zero would mean no neutral was bound \
             and the check proved nothing, more than one that they stopped sharing it"
        );
        // The per-draw allocate-and-zero this guards: a fresh dedicated 64 KiB `VkBuffer` a draw,
        // bound to memory and written to zero, and destroyed or parked after.
        let cleared = after.new_buffer_bytes_cleared - before.new_buffer_bytes_cleared;
        assert_eq!(
            cleared, 0,
            "{draws} draws wrote {cleared} bytes of a new buffer to zero to bind a neutral"
        );
        let churn = |a: &crate::ObjectChurn, b: &crate::ObjectChurn| {
            (
                a.vk_buffers_made - b.vk_buffers_made,
                a.vk_buffer_bindings - b.vk_buffer_bindings,
                a.vk_buffers_destroyed - b.vk_buffers_destroyed,
            )
        };
        assert_eq!(
            churn(&after.object_churn, &before.object_churn),
            (0, 0, 0),
            "{draws} draws made, bound or destroyed a `VkBuffer` to bind a neutral"
        );
    }
}

/// Sixty-four draws of one pipeline with one set of bindings bind their descriptor set **once**:
/// every draw's set is now the same one, which the encoder's set cache answers, where a fresh
/// zero region a draw made each draw's set differ and cost a rebind.
#[test]
fn draws_that_share_the_zero_buffer_bind_their_set_once() {
    let (_, before, after) = drawn(64);
    let binds = after.commands_replayed.cmd_bind_descriptor_sets
        - before.commands_replayed.cmd_bind_descriptor_sets;
    let draws = after.commands_replayed.cmd_draw - before.commands_replayed.cmd_draw;
    assert_eq!(
        (binds, draws),
        (1, 64),
        "64 draws recorded {binds} descriptor-set binds and {draws} draws"
    );
}
