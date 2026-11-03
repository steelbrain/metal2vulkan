//! The runtime's own textures take no texture ID (`resource_ids.rs`): a pass of separate depth
//! and stencil textures renders through a combined scratch image (`depth_stencil_scratch.rs`),
//! and a memoryless attachment through a backing the runtime keeps (`attachment_stand_in.rs`),
//! each a texture of the runtime's, made as the program encodes. The program's textures made
//! before, between and after them take Metal's IDs: 1 on, in the order the program made them.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch, WAIT};
use crate::{
    ClearColor, CpuCacheMode, Device, HazardTrackingMode, LoadAction, RenderPassDescriptor,
    ResourceOptions, StorageMode, StoreAction, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};

/// A 4x4 texture of the program's of `pf`, `usage` and `storage`, at a plain ID.
fn texture(device: &Device, pf: PixelFormat, usage: TextureUsage, storage: StorageMode) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(pf, 4, 4, false);
    descriptor.usage = usage;
    descriptor.resource_options = match storage {
        StorageMode::Private => ResourceOptions::private(),
        storage => ResourceOptions::new(
            storage,
            CpuCacheMode::DefaultCache,
            HazardTrackingMode::Default,
        ),
    };
    device.new_texture(&descriptor).unwrap()
}

/// A private 4x4 `RGBA8Unorm` texture of the program's.
fn plain(device: &Device) -> Texture {
    let usage = TextureUsage::SHADER_READ;
    texture(device, PixelFormat::RGBA8Unorm, usage, StorageMode::Private)
}

/// The program's textures take IDs 1 to 7 in the order it makes them, with a pass of separate
/// depth and stencil textures and a pass of a memoryless attachment encoded between them: the
/// scratch image and the backing those passes render through take none.
#[test]
fn the_runtime_s_own_textures_take_no_texture_id() {
    let scratch = Scratch::new("internal-texture-ids");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let target = TextureUsage::RENDER_TARGET;
    let mut ids = Vec::new();
    let mut held = Vec::new();
    let mut note = |t: Texture, ids: &mut Vec<u64>| {
        ids.push(t.gpu_resource_id().0);
        held.push(t.clone());
        t
    };
    note(plain(&device), &mut ids);
    let private = StorageMode::Private;
    let depth = note(
        texture(&device, PixelFormat::Depth32Float, target, private),
        &mut ids,
    );
    let stencil = note(
        texture(&device, PixelFormat::Stencil8, target, private),
        &mut ids,
    );
    let transient = StorageMode::Memoryless;
    let memoryless = note(
        texture(&device, PixelFormat::RGBA8Unorm, target, transient),
        &mut ids,
    );
    let command_buffer = queue.command_buffer().unwrap();
    // Separate depth and stencil textures: the pass renders into a combined scratch image.
    let mut descriptor = RenderPassDescriptor::new();
    descriptor.depth_attachment.texture = Some(depth);
    descriptor.depth_attachment.load_action = LoadAction::Clear;
    descriptor.depth_attachment.store_action = StoreAction::Store;
    descriptor.stencil_attachment.texture = Some(stencil);
    descriptor.stencil_attachment.load_action = LoadAction::Clear;
    descriptor.stencil_attachment.store_action = StoreAction::Store;
    let encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
    note(plain(&device), &mut ids);
    encoder.end_encoding().unwrap();
    // A memoryless attachment: the pass renders into a backing the runtime keeps.
    let mut descriptor = RenderPassDescriptor::new();
    let color = &mut descriptor.color_attachments[0];
    color.texture = Some(memoryless);
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::DontCare;
    color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
    let encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
    note(plain(&device), &mut ids);
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    note(plain(&device), &mut ids);
    assert_eq!(ids, [1, 2, 3, 4, 5, 6, 7], "on {}", device.name());
}
