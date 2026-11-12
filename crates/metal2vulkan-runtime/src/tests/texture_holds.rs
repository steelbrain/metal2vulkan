//! A texture a committed command buffer uses, released by the program while that command buffer
//! still waits behind other work, keeps its `VkImage` until the command buffer completed, as a
//! committed Metal command buffer retains the resources its commands use: its image is released
//! (`WorkCounts::object_churn.images_own_memory`) only after, and a validation layer running sees
//! no `VUID-vkDestroyImage-image-01000`. Each leg commits work the command buffer waits behind
//! ([`busy`]), then the command buffer, then releases the texture:
//!
//! - a `Private` `RGBA16Float` render target with mipmaps that a blit copies a buffer into and
//!   generates the mipmaps of, and a render pass clears;
//! - a `Shared` `RGBA16Float` texture that a host call (`replace_region`) wrote, which a blit reads
//!   into a buffer, whose bytes are the call's;
//! - a new `Private` `RGBA16Float` render target released at once, whose zeros the device may
//!   still be writing.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{busy, device, Scratch, FILLED, WAIT};
use crate::{
    BlitOption, CommandBuffer, CommandBufferStatus, CommandQueue, Device, LoadAction, Origin,
    Region, RenderPassDescriptor, ResourceOptions, Size, StoreAction, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};

/// The textures' side, in texels.
const SIDE: u64 = 64;

/// An `RGBA16Float` texel's bytes.
const TEXEL: u64 = 8;

/// The half-float word every texel's channels hold (1.5).
const WORD: u16 = 0x3e00;

/// An `RGBA16Float` texture of `options`, `usage`, mipmapped or not.
fn texture(device: &Device, options: ResourceOptions, usage: TextureUsage, mips: bool) -> Texture {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, SIDE, SIDE, mips);
    d.usage = usage;
    d.resource_options = options;
    device.new_texture(&d).unwrap()
}

/// A level 0's bytes, every channel [`WORD`].
fn bytes() -> Vec<u8> {
    WORD.to_ne_bytes()
        .repeat((SIDE * SIDE * TEXEL / 2) as usize)
}

/// Commit `command_buffer` behind work of `queue`'s, release `texture`, and check its image went
/// after the command buffer completed, not before; the work it waited behind, to wait for.
fn release_in_flight(
    leg: &str,
    queue: &CommandQueue,
    command_buffer: &CommandBuffer,
    texture: Texture,
) -> CommandBuffer {
    let device = queue.device();
    let filled = device
        .new_buffer(FILLED, ResourceOptions::private())
        .unwrap();
    let busy = busy(queue, &filled);
    command_buffer.commit().unwrap();
    let released = || device.work_counts().object_churn.images_own_memory;
    let before = released();
    let pending = command_buffer.status() != CommandBufferStatus::Completed;
    drop(texture);
    let at_release = released() - before;
    let still = command_buffer.status() != CommandBufferStatus::Completed;
    eprintln!(
        "TEXTURE-HOLDS {leg} on {}: pending at the release {pending}, still after it {still}, \
         images released at the release {at_release}",
        device.name()
    );
    if pending && still {
        assert_eq!(
            at_release, 0,
            "{leg}: the texture's image went while a command buffer that uses it ran"
        );
    } else {
        eprintln!("TEXTURE-HOLDS {leg}: the command buffer completed before the release");
    }
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    busy
}

#[test]
fn a_private_target_a_committed_blit_and_pass_write_keeps_its_image_until_completion() {
    let scratch = Scratch::new("texture-private-written");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    let target = texture(&device, ResourceOptions::private(), usage, true);
    let source = device
        .new_buffer_with_bytes(&bytes(), ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_texture(
        &source,
        0,
        SIDE * TEXEL,
        SIDE * SIDE * TEXEL,
        Size::new(SIDE, SIDE, 1),
        &target,
        0,
        0,
        Origin::new(0, 0, 0),
        BlitOption::None,
    )
    .unwrap();
    blit.generate_mipmaps(&target).unwrap();
    blit.end_encoding().unwrap();
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    let encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.end_encoding().unwrap();
    // The pass's attachment, the last of the program's holds on the texture but the command
    // buffer's.
    drop(pass);
    let busy = release_in_flight("private written", &queue, &command_buffer, target);
    busy.wait_until_completed_for(WAIT).unwrap();
}

#[test]
fn a_shared_texture_a_host_call_wrote_and_a_committed_blit_reads_keeps_its_image_until_completion()
{
    let scratch = Scratch::new("texture-shared-replaced");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let shared = texture(
        &device,
        ResourceOptions::shared(),
        TextureUsage::SHADER_READ,
        false,
    );
    let bytes = bytes();
    shared
        .replace_region(
            Region::new_2d(0, 0, SIDE, SIDE),
            0,
            0,
            &bytes,
            SIDE * TEXEL,
            0,
        )
        .unwrap();
    let read = device
        .new_buffer(SIDE * SIDE * TEXEL, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        &shared,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(SIDE, SIDE, 1),
        &read,
        0,
        SIDE * TEXEL,
        SIDE * SIDE * TEXEL,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    let busy = release_in_flight("shared replaced", &queue, &command_buffer, shared);
    busy.wait_until_completed_for(WAIT).unwrap();
    let contents = read.contents().unwrap();
    // SAFETY: the buffer's `length` bytes, which no command buffer writes now.
    let back = unsafe { std::slice::from_raw_parts(contents.as_ptr(), read.length() as usize) };
    assert!(
        back == bytes.as_slice(),
        "the blit read the host call's bytes"
    );
}

#[test]
fn a_new_private_target_released_at_once_goes_cleanly() {
    let scratch = Scratch::new("texture-private-new");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    for _ in 0..8 {
        drop(texture(&device, ResourceOptions::private(), usage, true));
    }
    // A command buffer after them, which carries any of their work still queued.
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}
