//! Device tests of an optimal `Private` texture as the device cache of a surface's plane, on the
//! device the cfg selects, under the validation layer: one `BGRA8Unorm` texture of 300x200 texels
//! (`Device::new_texture`), views of it in its own format and its sRGB twin
//! (`Texture::new_texture_view_with_placement`), and its texels moved in from and out to no-copy
//! buffers over private anonymous pages at a plane's offset and pitch, both multiples of 16 and the
//! pitch past the row's bytes (`copy_from_buffer_to_texture`, `copy_from_texture_to_buffer`).
//!
//! - **The round trip.** A copy in from one buffer, then a copy out to another, in a command buffer
//!   each, gives the texels back exactly, and leaves every byte between the rows, before the first
//!   and after the last as it was, in both buffers.
//! - **Two views, two encoders.** A render pass clears and stores through the sRGB view, and the
//!   next encoder, a blit, copies the texels out through the other: with hazard tracking, and with
//!   untracked resources and a fence the pass updates after its fragments and the blit waits for,
//!   the copy reads the pass's texels, and the layer's synchronization validation reports no
//!   hazard, as Metal orders both. A view's uses are its texture's image's (`Texture::bound_use`):
//!   the tracker keys them on the image, not on the view object. The control: the same untracked
//!   with no fence, which Metal leaves to the program, is reported (a read after write), so the
//!   layer sees the pair of accesses the clean runs order.

use super::host_memory::{no_copy, not_importable, on_device, run, Mapping};
use super::{validated_device_with, LaneChecks, Messages, ScratchCache};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{PixelFormat, RenderStages, TextureUsage};
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, CommandQueue, CpuCacheMode, DebugMessage, Device,
    HazardTrackingMode, IdPlacement, LoadAction, Origin, RenderPassDescriptor, ResourceOptions,
    Size, StorageMode, StoreAction, Texture, TextureDescriptor, ViewPlacement,
};

/// The plane's size in texels.
const WIDTH: u64 = 300;
const HEIGHT: u64 = 200;
/// A row's texel bytes, and the plane's pitch: the next multiple of 16 past them, and one more.
const ROW: u64 = WIDTH * 4;
const PITCH: u64 = ROW + 32;
/// Where the plane starts in its buffer: a nonzero multiple of 16.
const OFFSET: u64 = 48;
/// The buffers' length: a multiple of every host page size, past the plane's last row.
const LEN: usize = 256 << 10;
/// What every byte outside the plane's texels holds.
const SENTINEL: u8 = 0xa5;
/// The pass's clear, through the sRGB view: 0 and 1 encode exactly.
const CLEAR: [f64; 4] = [1.0, 0.0, 1.0, 1.0];
/// Its texel in memory, in the texture's `BGRA` order.
const CLEARED: [u8; 4] = [0xff, 0x00, 0xff, 0xff];

/// The plane's texture, a render target the program samples, of `options`.
fn descriptor(options: ResourceOptions) -> TextureDescriptor {
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, WIDTH, HEIGHT, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = options;
    descriptor
}

/// Private storage, tracked as `hazards` says.
fn private(hazards: HazardTrackingMode) -> ResourceOptions {
    ResourceOptions::new(StorageMode::Private, CpuCacheMode::DefaultCache, hazards)
}

/// Texels that name their place, no byte of them [`SENTINEL`].
fn texels() -> Vec<u8> {
    (0..WIDTH * HEIGHT * 4)
        .map(|i| match (i * 7 + i / ROW) % 251 {
            byte if byte as u8 == SENTINEL => 0,
            byte => byte as u8,
        })
        .collect()
}

/// The bytes of a buffer holding `rows` (`ROW` bytes each) at the plane's offset and pitch, and
/// [`SENTINEL`] everywhere else.
fn plane_bytes(rows: &[u8]) -> Vec<u8> {
    let mut bytes = vec![SENTINEL; LEN];
    for (y, row) in rows.chunks(ROW as usize).enumerate() {
        let at = (OFFSET + y as u64 * PITCH) as usize;
        bytes[at..at + row.len()].copy_from_slice(row);
    }
    bytes
}

/// A no-copy buffer over `pages`; `None`, printed, where the device imports no host memory.
fn over(device: &Device, pages: &Mapping) -> Option<Buffer> {
    // SAFETY: the mapping outlives the buffer, and only the buffer's commands use it while the
    // test's command buffers run.
    match unsafe {
        no_copy(
            device,
            pages.pointer(),
            LEN,
            ResourceOptions::shared(),
            None,
        )
    } {
        Ok(buffer) => Some(buffer),
        Err(error) if not_importable(&error) => {
            eprintln!("PLANE-CACHE: the device imports no host memory: {error}");
            None
        }
        Err(error) => panic!("a no-copy buffer: {error}"),
    }
}

/// The whole plane's size.
fn size() -> Size {
    Size::new(WIDTH, HEIGHT, 1)
}

/// The first byte where `got` and `want` differ, named for `what`.
fn same_bytes(got: &[u8], want: &[u8], what: &str) {
    if let Some(at) = got.iter().zip(want).position(|(g, w)| g != w) {
        let (row, column) = (
            (at as u64).saturating_sub(OFFSET) / PITCH,
            (at as u64).saturating_sub(OFFSET) % PITCH,
        );
        panic!(
            "{what}: byte {at} (row {row}, byte {column} of its pitch) is {:#04x}, not {:#04x}",
            got[at], want[at]
        );
    }
    assert_eq!(got.len(), want.len(), "{what}");
}

#[test]
fn an_optimal_private_texture_round_trips_between_no_copy_buffers_at_a_plane_s_offset_and_pitch() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let (source, target) = (Mapping::anonymous(LEN), Mapping::anonymous(LEN));
        let texels = texels();
        let given = plane_bytes(&texels);
        source.write(0, &given);
        target.fill(SENTINEL);
        let Some(upload) = over(device, &source) else {
            return;
        };
        let download = over(device, &target).unwrap();
        let texture = device
            .new_texture(&descriptor(private(HazardTrackingMode::Default)))
            .unwrap();
        // The fill: the guest's texels into the device cache.
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_texture(
            &upload,
            OFFSET,
            PITCH,
            PITCH * HEIGHT,
            size(),
            &texture,
            0,
            0,
            Origin::new(0, 0, 0),
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        run(&command_buffer);
        // The download: the device cache's texels out to the guest's pages.
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            &texture,
            0,
            0,
            Origin::new(0, 0, 0),
            size(),
            &download,
            OFFSET,
            PITCH,
            PITCH * HEIGHT,
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        run(&command_buffer);
        same_bytes(&target.read(0..LEN), &given, "the download");
        same_bytes(&source.read(0..LEN), &given, "the upload's pages");
    });
}

/// How the pass and the copy are ordered: by hazard tracking, by a fence over untracked
/// resources, or not at all (the control).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Order {
    Tracked,
    Fenced,
    Unordered,
}

/// Clear and store the texture through its sRGB view in a render pass, then copy its texels out
/// through its view in its own format in the next encoder, a blit, into a no-copy buffer at the
/// plane's offset and pitch, ordered as `order` says; the buffer's bytes, `None` where the device
/// imports no host memory.
fn views_in_two_encoders(device: &Device, queue: &CommandQueue, order: Order) -> Option<Vec<u8>> {
    let target = Mapping::anonymous(LEN);
    target.fill(SENTINEL);
    let download = over(device, &target)?;
    let hazards = match order {
        Order::Tracked => HazardTrackingMode::Tracked,
        Order::Fenced | Order::Unordered => HazardTrackingMode::Untracked,
    };
    let texture = device.new_texture(&descriptor(private(hazards))).unwrap();
    let srgb = PixelFormat::BGRA8UnormSRGB;
    let own = PixelFormat::BGRA8Unorm;
    // Both views are Vulkan views of the texture's image, never a shadow of it.
    for format in [srgb, own] {
        assert_eq!(texture.view_placement(format), Ok(ViewPlacement::Image));
    }
    let written = texture
        .new_texture_view_with_placement(srgb, IdPlacement::Any)
        .unwrap();
    let read: Texture = texture
        .new_texture_view_with_placement(own, IdPlacement::Any)
        .unwrap();
    let fence = device.new_fence();
    let command_buffer = queue.command_buffer().unwrap();
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(written);
    attachment.load_action = LoadAction::Clear;
    attachment.store_action = StoreAction::Store;
    let [r, g, b, a] = CLEAR;
    attachment.clear_color = ClearColor::new(r, g, b, a);
    let mut render = command_buffer.render_command_encoder(&pass).unwrap();
    if order == Order::Fenced {
        render.update_fence(&fence, RenderStages::FRAGMENT).unwrap();
    }
    render.end_encoding().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    if order == Order::Fenced {
        blit.wait_for_fence(&fence).unwrap();
    }
    blit.copy_from_texture_to_buffer(
        &read,
        0,
        0,
        Origin::new(0, 0, 0),
        size(),
        &download,
        OFFSET,
        PITCH,
        PITCH * HEIGHT,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    run(&command_buffer);
    Some(target.read(0..LEN))
}

/// The messages the layer reported for `test`, run on a device whose layer checks
/// synchronization, once the device is gone.
fn reported(test: impl FnOnce(&Device)) -> Vec<DebugMessage> {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            LaneChecks {
                synchronization: true,
                ..LaneChecks::default()
            }
            .apply(options)
        })
        .unwrap();
        test(&device);
    }
    messages.take()
}

/// The hazards among `messages`, every message printed with `order` for the lane's log.
fn hazards(order: Order, messages: &[DebugMessage]) -> Vec<String> {
    for message in messages {
        eprintln!(
            "PLANE-CACHE {order:?}: {} {}",
            message.id_name,
            message.text.replace('\n', " ")
        );
    }
    (messages.iter())
        .filter(|message| message.id_name.starts_with("SYNC-HAZARD"))
        .map(|message| message.id_name.clone())
        .collect()
}

#[test]
fn a_view_s_pass_orders_another_view_s_copy_in_the_next_encoder_tracked_or_fenced() {
    let cleared = plane_bytes(&CLEARED.repeat((WIDTH * HEIGHT) as usize));
    for order in [Order::Tracked, Order::Fenced] {
        let mut bytes = None;
        let messages = reported(|device| {
            let queue = device.new_command_queue().unwrap();
            bytes = views_in_two_encoders(device, &queue, order);
        });
        let Some(bytes) = bytes else {
            return;
        };
        let found = hazards(order, &messages);
        assert!(
            messages.is_empty(),
            "{order:?}: the layer reported {found:?}: {messages:#?}"
        );
        same_bytes(
            &bytes,
            &cleared,
            &format!("{order:?}: the copy through the other view"),
        );
    }
    // The control: unordered, the layer reports the copy's read after the pass's write.
    let mut ran = false;
    let messages = reported(|device| {
        let queue = device.new_command_queue().unwrap();
        ran = views_in_two_encoders(device, &queue, Order::Unordered).is_some();
    });
    if !ran {
        return;
    }
    let found = hazards(Order::Unordered, &messages);
    eprintln!("PLANE-CACHE Unordered: hazards {found:?}");
    assert!(
        found.iter().any(|id| id == "SYNC-HAZARD-READ-AFTER-WRITE"),
        "the unordered control reported no read after write: {messages:#?}"
    );
}
