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
//! proving nothing about the path it names.
//!
//! The dispatch path binds neutrals through the same code, and is covered below by a kernel with
//! a `constant` slot left unbound. So is the writable half, by the same kernel with its written
//! `device` slot left unbound instead: that one must still take a region of its own per dispatch,
//! and the test requires it to, so a later change cannot quietly start sharing one buffer between
//! commands that may store through the binding.
//!
//! # A dispatch count's own cost
//!
//! The two dispatch checks read `new_buffer_bytes_cleared` as the difference between **two
//! dispatch counts**, not as an absolute, so that what the command buffer clears whatever its
//! dispatch count is not counted against the neutral: a region of the upload arena clears the
//! bytes its own data does not cover as it is handed out (`upload.rs`). The read-only check reads
//! its step against the same kernel's all-bound shape, which binds no neutral at all, so what it
//! asks is whether the read-only neutral adds anything to what a dispatch already clears.
//!
//! The fixed cost a command buffer pays whatever its dispatch count used to be a chunk: the first
//! region any command took opened the arena's first chunk, 4 KiB, which was a new buffer cleared
//! whole. It is gone -- a chunk is made without a clear -- and `tests::upload_chunk_zeros` is
//! where that is held.
//!
//! What none of it covers: a draw or dispatch whose read-only slot is unbound **and** whose
//! length differs from `zero_buffer_bytes` on this device, since one device offers one length; and
//! the fully-bound case, which binds no neutral at all and whose per-draw work was already one
//! `vkCmdDraw` and nothing else.

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
///
/// Where the variant pushes its set 0 (`pipeline::objects::pushes_set_zero`: lavapipe, and any
/// device of Vulkan 1.4 or `VK_KHR_push_descriptor` whose set 0 a push descriptor set layout
/// holds), each of the draws records with it pushed and none binds a set: the first draw's push
/// carries its writes, and every later draw's carries none, since set 0 holds them already
/// (`render/bound.rs`), so the 64 draws record **one** push. Either way set 0 is recorded once.
#[test]
fn draws_that_share_the_zero_buffer_bind_their_set_once() {
    let (_, before, after) = drawn(64);
    let recorded = |of: fn(&crate::CommandsRecorded) -> u64| {
        of(&after.commands_recorded) - of(&before.commands_recorded)
    };
    let binds = recorded(|c| c.cmd_bind_descriptor_sets);
    let pushes = recorded(|c| c.cmd_push_descriptor_set);
    let draws = recorded(|c| c.cmd_draw);
    let pushing = after.draws_pushing_set_zero - before.draws_pushing_set_zero;
    println!("NEUTRAL-ZEROS: 64 draws, {pushing} pushing set 0: {binds} binds, {pushes} pushes");
    assert!(
        pushing == 0 || pushing == 64,
        "{pushing} of 64 draws of one variant pushed set 0: its variant pushes for all or none"
    );
    let want = match pushing {
        0 => (1, 0, 64),
        _ => (0, 1, 64),
    };
    assert_eq!(
        (binds, pushes, draws),
        want,
        "64 draws, {pushing} of them pushing set 0, recorded {binds} descriptor-set binds, \
         {pushes} pushes and {draws} draws"
    );
}

/// `scale_and_offset` of the examples fixture, whose `input` and `output` are written `device`
/// pointers (storage buffers) and whose `scale` is a `constant` (a read-only uniform buffer).
/// `unbound` is the buffer index left unset, which the plan binds a neutral at; an `unbound` past
/// the last slot (3) leaves every slot bound, so the plan binds no neutral.
fn dispatched(dispatches: usize, unbound: u64) -> (u64, crate::WorkCounts, crate::WorkCounts) {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/examples/runtime_examples.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    let function = device
        .new_library_with_compile_library(library)
        .new_function_with_name("scale_and_offset")
        .unwrap();
    let state = device
        .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
        .unwrap();
    let buffer = device.new_buffer(1024, ResourceOptions::shared()).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    for index in [0, 1, 2] {
        if index != unbound {
            encoder.set_buffer(Some(&buffer), 0, index as u32).unwrap();
        }
    }
    let before = device.work_counts();
    for _ in 0..dispatches {
        encoder
            .dispatch_threads(crate::Size::new(1, 1, 1), crate::Size::new(1, 1, 1))
            .unwrap();
    }
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer
        .wait_until_completed_for(std::time::Duration::from_secs(60))
        .unwrap();
    assert_eq!(command_buffer.error(), None, "the dispatches ran");
    let after = device.work_counts();
    (device.neutral_zeros_made(), before, after)
}

/// A kernel's unbound `constant` slot is a read-only binding: the dispatches share the device's
/// one zero buffer, so twice as many of them cost nothing more -- no second zero buffer, no
/// further `VkBuffer`, and not a byte more written to zero than the same dispatches cost with
/// every slot bound.
///
/// The bytes are read as a per-dispatch step against the **all-bound** shape, which binds no
/// neutral at all, so the question this check asks is the one it is about: does the read-only
/// neutral add anything to what a dispatch already clears?
#[test]
fn dispatches_share_the_devices_one_zero_buffer() {
    let run = |dispatches, unbound| {
        let (made, before, after) = dispatched(dispatches, unbound);
        (
            made,
            after.new_buffer_bytes_cleared - before.new_buffer_bytes_cleared,
            after.object_churn.vk_buffers_made - before.object_churn.vk_buffers_made,
        )
    };
    let (made_8, cleared_8, buffers_8) = run(8, 2);
    let (made_16, cleared_16, buffers_16) = run(16, 2);
    assert_eq!(
        (made_8, made_16),
        (1, 1),
        "8 and 16 dispatches made {made_8} and {made_16} zero buffers for their unbound \
         `constant` slot, not 1 each: zero would mean no neutral was bound and the check proved \
         nothing, more than one that they stopped sharing it"
    );
    // The per-dispatch allocate-and-zero this guards: a fresh dedicated 64 KiB `VkBuffer` a
    // dispatch, bound to memory and written to zero, and destroyed or parked after.
    assert_eq!(
        buffers_16, buffers_8,
        "8 dispatches made {buffers_8} new `VkBuffer`(s) and 16 made {buffers_16}: eight more \
         dispatches must make none for a read-only neutral they share"
    );
    let (_, bound_8, _) = run(8, 3);
    let (_, bound_16, _) = run(16, 3);
    assert_eq!(
        cleared_16 - cleared_8,
        bound_16 - bound_8,
        "eight further dispatches wrote {} bytes to zero with the `constant` slot unbound and {} \
         with every slot bound: a read-only neutral they share must add nothing to what a \
         dispatch clears",
        cleared_16 - cleared_8,
        bound_16 - bound_8
    );
}

/// A kernel's unbound **written** `device` slot is not read-only, so its neutral keeps a region of
/// its own per dispatch, exactly as every neutral binding did before the device kept one buffer.
///
/// This is the half that must NOT be shared: the kernel stores through `output`, and one buffer
/// shared by every command would hand a later command what an earlier one wrote. The claim is
/// that the regions still scale with the dispatches -- each further dispatch clears a region's
/// worth more -- so a change that started sharing them fails here rather than silently handing
/// one command another's stores. Measured as the step between two dispatch counts rather than as
/// an absolute, because what a region costs is a chunk size and an alignment, neither of which is
/// this test's business, and because the command buffer's own fixed cost is not a region at all
/// (the module docs). A step is what catches sharing: shared, every count clears the same fixed
/// bytes and the first step is zero.
///
/// The device's zero-buffer count is deliberately not asserted here: `make_neutrals` pre-makes it
/// at the pipeline's creation, whose plan has every slot unbound and so a read-only neutral among
/// them, which is exactly what pre-making is for.
#[test]
fn a_written_slots_neutral_is_not_shared() {
    let cleared = |dispatches| {
        let (_, before, after) = dispatched(dispatches, 1);
        after.new_buffer_bytes_cleared - before.new_buffer_bytes_cleared
    };
    // One more dispatch's own region, past the command buffer's fixed cost, which both counts pay
    // once.
    let one = cleared(2) - cleared(1);
    assert!(
        one > 0,
        "a second dispatch with a written slot unbound cleared no further bytes, so the two \
         shared one region"
    );
    let eight = cleared(16) - cleared(8);
    assert_eq!(
        eight,
        one * 8,
        "one further dispatch cleared {one} bytes more and eight further dispatches cleared \
         {eight} more, which is not one region each: a written slot's neutral must not be shared \
         between commands"
    );
}
