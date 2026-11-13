//! What a device's recovery carries of the program's textures (`texture/carry.rs`): a texture of
//! `Device::new_texture`, a view of it and a texture over a buffer the recovery carried go on as
//! the same texture objects, made again on the new `VkDevice`, and serve there; those over pages
//! the recovery carried read the bytes the pages hold.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs, on the kind of device
//! the cfg selects (docs/VALIDATION.md). No GPU fault: the loss is marked (`Device::recover_for_test`).

use super::device_recovery::{heard_device, outcome, recovery_notices, Heard, Scratch, WAIT};
use super::device_recovery_carry::{fill, live_vk_devices, read, BYTES};
use crate::fresh_pages::FRESH_PAGES_PAST;
use crate::{
    BlitOption, ClearColor, CommandBufferStatus, CommandQueue, CpuCacheMode, Device, Error,
    HazardTrackingMode, LoadAction, Origin, RenderPassDescriptor, ResourceOptions, Size,
    StorageMode, StoreAction, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};

/// The side of the square `RGBA8Unorm` textures the test makes: [`BYTES`] bytes of texels.
const SIDE: u64 = 32;

/// A `Private` `RGBA8Unorm` texture of [`SIDE`] by [`SIDE`] texels, of `usage`.
fn descriptor(usage: TextureUsage) -> TextureDescriptor {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = usage;
    descriptor.resource_options = ResourceOptions::private();
    descriptor
}

/// `descriptor` with `options`.
fn with_options(mut descriptor: TextureDescriptor, options: ResourceOptions) -> TextureDescriptor {
    descriptor.resource_options = options;
    descriptor
}

/// `texture`'s texels as `queue`'s GPU work reads them: copied into a new buffer, read back.
/// Refused as the copy's encoding refuses `texture`.
fn texels(device: &Device, queue: &CommandQueue, texture: &Texture) -> Result<Vec<u8>, Error> {
    let target = device.new_buffer(BYTES, ResourceOptions::private())?;
    let command_buffer = queue.command_buffer()?;
    let mut blit = command_buffer.blit_command_encoder()?;
    let (size, origin) = (Size::new(SIDE, SIDE, 1), Origin::new(0, 0, 0));
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        origin,
        size,
        &target,
        0,
        4 * SIDE,
        0,
        BlitOption::None,
    )?;
    blit.end_encoding()?;
    command_buffer.commit()?;
    assert_eq!(
        outcome(&command_buffer),
        (CommandBufferStatus::Completed, None)
    );
    read(device, queue, &target)
}

/// `texture` cleared to the colour whose four channels are `unorm` by a render pass of `queue`,
/// completed. Refused as the pass's encoding refuses `texture`.
fn clear(queue: &CommandQueue, texture: &Texture, unorm: f64) -> Result<(), Error> {
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(texture.clone());
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    color.clear_color = ClearColor::new(unorm, unorm, unorm, unorm);
    let command_buffer = queue.command_buffer()?;
    let encoder = command_buffer.render_command_encoder(&pass)?;
    encoder.end_encoding()?;
    command_buffer.commit()?;
    assert_eq!(
        outcome(&command_buffer),
        (CommandBufferStatus::Completed, None)
    );
    Ok(())
}

/// A recovery carries a `Private` render target of `new_texture`, a view of it and a texture over
/// a `Private` buffer it carried, as its notice counts: each the same texture object on the next
/// generation (its `ObjectId`, resource ID and label). The render target's contents were lost
/// with the `VkDevice` and it renders again, its view and a view made after the recovery reading
/// what it rendered; the texture over the buffer reads the buffer's bytes, zero as the buffer's
/// contents were lost, then what the GPU writes into the buffer. Their parts on the lost
/// generation keep its `VkDevice` until the textures go.
#[test]
fn a_recovery_carries_textures_their_views_and_those_over_carried_buffers() {
    let scratch = Scratch::new();
    let heard = Heard::default();
    let device = heard_device(&scratch.0, &heard);
    let queue = device.new_command_queue().unwrap();
    let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let target = device.new_texture(&descriptor(usage)).unwrap();
    target.set_label("target");
    let view = target.new_texture_view(PixelFormat::RGBA8Unorm).unwrap();
    clear(&queue, &target, 1.0).unwrap();
    assert_eq!(
        texels(&device, &queue, &view).unwrap(),
        vec![255; BYTES as usize]
    );
    let buffer = device
        .new_buffer(BYTES, ResourceOptions::private())
        .unwrap();
    fill(&queue, &buffer, 7);
    let over = (buffer.new_texture(&descriptor(TextureUsage::SHADER_READ), 0, 4 * SIDE)).unwrap();
    assert_eq!(
        texels(&device, &queue, &over).unwrap(),
        vec![7; BYTES as usize]
    );
    let identity = |texture: &Texture| (texture.object_id(), texture.gpu_resource_id());
    let before = [identity(&target), identity(&view), identity(&over)];
    let from = heard.lock().unwrap().len();
    device.recover_for_test().unwrap();
    let said = recovery_notices(&heard, from, "the device");
    assert_eq!(said.len(), 1, "{said:?}");
    assert!(
        said[0].contains(
            ", and 1 textures whose contents were lost too (they read as new ones: 0 `Shared`, 0 \
             `Managed`, 1 `Private` or memoryless), 1 textures over the buffers carried and 1 \
             texture views; left refused"
        ),
        "{said:?}"
    );
    // The same texture objects, on the next generation.
    for texture in [&target, &view, &over] {
        assert!(texture.device().generation_serves());
    }
    assert_eq!(
        [identity(&target), identity(&view), identity(&over)],
        before
    );
    assert_eq!(target.label().as_deref(), Some("target"));
    // The texture over the buffer reads the buffer's bytes there.
    assert_eq!(
        texels(&device, &queue, &over).unwrap(),
        vec![0; BYTES as usize]
    );
    fill(&queue, &buffer, 9);
    assert_eq!(
        texels(&device, &queue, &over).unwrap(),
        vec![9; BYTES as usize]
    );
    // The render target renders there, and its views read what it rendered.
    let unorm = 128.0 / 255.0;
    clear(&queue, &target, unorm).unwrap();
    let later = target.new_texture_view(PixelFormat::RGBA8Unorm).unwrap();
    for texture in [&target, &view, &later] {
        assert_eq!(
            texels(&device, &queue, texture).unwrap(),
            vec![128; BYTES as usize]
        );
    }
    // The lost `VkDevice` goes with the textures' parts there.
    assert_eq!(live_vk_devices(&device, 2), 2);
    drop((view, later, target, over));
    assert_eq!(live_vk_devices(&device, 1), 1);
}

/// What a recovery keeps of textures' contents, by where their texels lie: a `Shared` texture
/// over a buffer in fresh pages reads the bytes the host wrote before it, through the GPU; a
/// `Shared` texture buffer of `new_texture` is made over its texels' buffer where the recovery
/// carried that with its bytes (a block of the runtime's own pages), and reads them; and the
/// textures of `new_texture` whose only copy was their image read as new ones, counted by storage
/// mode in the recovery's notice.
#[test]
fn a_recovery_keeps_the_bytes_of_textures_over_carried_pages() {
    let scratch = Scratch::new();
    let heard = Heard::default();
    let device = heard_device(&scratch.0, &heard);
    // The device's checks answered first, so that where a held import costs its submissions
    // nothing, its `Shared` buffers lie in blocks of its own pages (`Holds::PageBuffers`).
    let _settled = device.settle_in_place_checks(WAIT);
    let _settled = device.settle_submit_checks(WAIT);
    let queue = device.new_command_queue().unwrap();
    let shared = ResourceOptions::shared();
    let read_only = descriptor(TextureUsage::SHADER_READ);
    // A `Shared` texture over a buffer of the program's in fresh pages, its bytes the host's.
    let fresh = {
        let made_on = device.new_fence().device();
        let fresh = made_on
            .fresh_pages_check_buffer(2 * FRESH_PAGES_PAST)
            .unwrap();
        made_on.visible(fresh)
    };
    assert!(
        fresh.in_fresh_pages(),
        "{} took no fresh pages: their carry is not exercised",
        device.name()
    );
    let pointer = fresh.contents().unwrap();
    // SAFETY: the buffer's first `BYTES` bytes, which no work in flight uses.
    unsafe { std::ptr::write_bytes(pointer.as_ptr(), 0x5a, BYTES as usize) };
    let over = (fresh.new_texture(&with_options(read_only, shared), 0, 4 * SIDE)).unwrap();
    assert_eq!(
        texels(&device, &queue, &over).unwrap(),
        vec![0x5a; BYTES as usize]
    );
    // A `Shared` texture buffer of `new_texture`, its texels' bytes the host's.
    let buffer_descriptor = TextureDescriptor::texture_buffer(
        PixelFormat::RGBA8Unorm,
        SIDE * SIDE,
        shared,
        TextureUsage::SHADER_READ,
    );
    let texture_buffer = device.new_texture(&buffer_descriptor).unwrap();
    let texels_buffer = texture_buffer.texels().unwrap().buffer.clone();
    let kept = texels_buffer.in_page_block();
    eprintln!(
        "CARRY-TEXTURE-BUFFER: {}: the `Shared` texture buffer's texels lie in a block of the \
         runtime's own pages: {kept}",
        device.name()
    );
    let pointer = texels_buffer.contents().unwrap();
    // SAFETY: the buffer's `BYTES` bytes, which no work in flight uses.
    unsafe { std::ptr::write_bytes(pointer.as_ptr(), 0x3c, BYTES as usize) };
    // A texture of `new_texture` of each storage mode, whose only copy is its image.
    let managed = ResourceOptions::new(
        StorageMode::Managed,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    );
    let lost = [shared, managed, ResourceOptions::private()].map(|options| {
        device
            .new_texture(&with_options(read_only, options))
            .unwrap()
    });
    let from = heard.lock().unwrap().len();
    device.recover_for_test().unwrap();
    let said = recovery_notices(&heard, from, "the device");
    assert_eq!(said.len(), 1, "{said:?}");
    let (shared_lost, carried_over) = match kept {
        true => (1, 2),
        false => (2, 1),
    };
    assert!(
        said[0].contains(&format!(
            ", and {} textures whose contents were lost too (they read as new ones: {shared_lost} \
             `Shared`, 1 `Managed`, 1 `Private` or memoryless), {carried_over} textures over the \
             buffers carried and 0 texture views",
            shared_lost + 2
        )),
        "{said:?}"
    );
    // The texture over the fresh pages reads the host's bytes there, through the GPU.
    assert!(over.device().generation_serves());
    assert_eq!(
        texels(&device, &queue, &over).unwrap(),
        vec![0x5a; BYTES as usize]
    );
    // The texture buffer lies over its texels' buffer, carried with its bytes, where the recovery
    // carried that; else over a new one.
    let now = texture_buffer.texels().unwrap().buffer.clone();
    assert_eq!(now.object_id() == texels_buffer.object_id(), kept);
    if kept {
        let pointer = now.contents().unwrap();
        // SAFETY: the buffer's `BYTES` bytes, which no work in flight uses.
        let bytes = unsafe { std::slice::from_raw_parts(pointer.as_ptr(), BYTES as usize) };
        assert_eq!(bytes, &[0x3c; BYTES as usize][..]);
    }
    // The textures whose only copy was their image serve as new ones there.
    for texture in lost.iter().chain([&texture_buffer]) {
        assert!(texture.device().generation_serves());
    }
}
