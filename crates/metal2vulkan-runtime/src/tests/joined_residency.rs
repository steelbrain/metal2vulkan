//! A command buffer's residency joins the resources its ended encoders made resident where its
//! own work reads it (`command_buffer::State::joined_residency`), never as an encoder ends: the
//! encoders opened after, their shadowed textures inherited (none here), join none, a read joins
//! each ended encoder's resources once, in the order they ended, and a read after joins none
//! more (`WorkCounts::resident_resources_joined`). The shadowed textures (`shadow.rs`) an ended
//! encoder made resident reach the encoders opened after, whether its resources are joined yet or
//! not, each once and in the order they were made resident; making a texture resident reads
//! nothing of it, the open that first asks looks at each once (`Residency::find_shadowed`).
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use crate::{
    Buffer, CommandBufferStatus, Device, DeviceKind, DeviceOptions, ResourceOptions, Texture,
    TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, ResourceUsage, TextureUsage};

/// The device the cfg selects: a CPU device under `m2v_cpu_icd`, else a GPU.
fn device() -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// Two compute encoders make 16 buffers resident each, 8 of them the same: ending them and
/// opening the second joins nothing; the command buffer's residency, read, joins all 32 and holds
/// the 24 buffers once each, the first encoder's first; read again, it joins none.
#[test]
fn ended_encoders_join_their_command_buffer_s_residency_where_it_is_read() {
    let device = device();
    let queue = device.new_command_queue().unwrap();
    let buffers: Vec<Buffer> = (0..24)
        .map(|_| device.new_buffer(256, ResourceOptions::default()).unwrap())
        .collect();
    let joined = || device.work_counts().resident_resources_joined;
    let before = joined();
    let command_buffer = queue.command_buffer().unwrap();
    for some in [&buffers[..16], &buffers[8..]] {
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        let resources: Vec<_> = some.iter().map(|buffer| buffer.as_resource()).collect();
        encoder
            .use_resources(&resources, ResourceUsage::READ)
            .unwrap();
        encoder.end_encoding().unwrap();
    }
    assert_eq!(joined() - before, 0, "an encoder's end or open joined");
    let residency = command_buffer.residency().unwrap();
    assert_eq!(joined() - before, 32);
    let held: Vec<&Buffer> = (residency.resources())
        .filter_map(|used| match used {
            crate::residency::Used::Buffer(buffer) => Some(buffer),
            _ => None,
        })
        .collect();
    assert!(
        held.iter().copied().eq(buffers.iter()),
        "each buffer once, in the order made resident"
    );
    drop(command_buffer.residency().unwrap());
    assert_eq!(joined() - before, 32, "a second read joined again");
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed().unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// A texture over a `Shared` buffer's bytes in rows of 352 bytes, not a multiple of 256, so a
/// shadow wherever the device lays no image's rows so (`shadow_carried_read.rs`): its own buffer,
/// held with it.
fn shadowed(device: &Device) -> (Buffer, Texture) {
    let buffer = device
        .new_buffer(352 * 5, ResourceOptions::shared())
        .unwrap();
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 68, 5, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = buffer.resource_options();
    let texture = buffer.new_texture(&descriptor, 0, 352).unwrap();
    assert!(texture.shadow().is_some(), "the device took no shadow");
    (buffer, texture)
}

/// Three compute encoders of one command buffer: the first makes a plain texture and a shadowed
/// one resident, the second another shadowed one and the first again, and the third, after a read
/// joined the first two into the command buffer's residency, the plain one. Each next encoder
/// starts with the shadowed textures alone, the first's first, whether the encoders before are
/// joined or not.
#[test]
fn shadowed_textures_made_resident_reach_the_next_encoders_in_order() {
    let device = device();
    let queue = device.new_command_queue().unwrap();
    let ((_first_bytes, first), (_second_bytes, second)) = (shadowed(&device), shadowed(&device));
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 4, 4, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    let plain = device.new_texture(&descriptor).unwrap();
    assert!(plain.shadow().is_none());
    let command_buffer = queue.command_buffer().unwrap();
    let inherited = || {
        let residency = command_buffer.inherited_residency(false);
        residency.textures().cloned().collect::<Vec<Texture>>()
    };
    let resident = |textures: &[&Texture]| {
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        let resources: Vec<_> = textures
            .iter()
            .map(|texture| texture.as_resource())
            .collect();
        encoder
            .use_resources(&resources, ResourceUsage::READ)
            .unwrap();
        encoder.end_encoding().unwrap();
    };
    resident(&[&plain, &first]);
    assert_eq!(inherited(), [first.clone()], "after the first encoder");
    resident(&[&second, &first]);
    assert_eq!(
        inherited(),
        [first.clone(), second.clone()],
        "after the second encoder, neither joined"
    );
    drop(command_buffer.residency().unwrap());
    resident(&[&plain]);
    assert_eq!(
        inherited(),
        [first.clone(), second.clone()],
        "after the third encoder, the first two joined"
    );
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed().unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}
