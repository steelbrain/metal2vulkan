//! A render pass that turns a texture's two faces while it logs its calls for a rebind, on the
//! device the cfg selects, under the validation layer (the runtime's `texture/faces.rs`,
//! **Encoded again**). While the device has a no-copy buffer over host pages, every render pass
//! logs its calls, so that a rebind of such a buffer before its command buffer's first submission
//! can encode it again (`replay.rs`). Such a pass turns as any other: a `Private` texture whose
//! first attached read gave it two faces turns to the other at a pass that clears it whole, its
//! read of the texels from before the pass taking the face presented before with no copy in.
//!
//! The texture is [`WIDTH`] x [`HEIGHT`] `BGRA8Unorm`. Each pass clears it and draws, through
//! [`RECT`], `tob_read` (the texel under each pixel, written as read), or loads it and draws
//! `tob_solid` there (`fixtures/public/engine/tob_renders.ll`, with `tob_rect`); so the texels
//! inside [`RECT`] after a read are those it read, Metal's answer the texels from before the pass.
//!
//! The lane: a clear and a solid `0x40`; the first read, which makes the faces; a solid `0x50`; a logged read that clears to
//! `0x30` (it turns); a solid `0x62`; then a command buffer behind a shared-event wait holding a
//! blit of the texture into the buffer over host pages, and a read that clears to `0x90`, its
//! corners from that buffer (it turns back). A rebind of the buffer before the signal encodes both
//! encoders again: the blit reads the texels from before the turn, `0x30` around `0x62`, and the
//! pass takes its turn again, reading the `0x62` with no copy in. The same lane beside an open
//! command buffer, where no pass turns and every read copies in, reads the same bytes. The layer
//! reports nothing; under `METAL2VULKAN_LANE_VALIDATION=sync` its synchronization checks run too.

use super::attachment_reads::function;
use super::host_memory::{no_copy, not_importable, on_device, Mapping};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor};
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, CommandBuffer, CommandQueue, Device, EventRef, HostMemory,
    LoadAction, Origin, PipelineOptions, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, ScissorRect, Size, StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(60);

/// The texture: `WIDTH` texels a row of 4 bytes, `HEIGHT` rows.
const WIDTH: u64 = 64;
const HEIGHT: u64 = 16;
const ROW: u64 = WIDTH * 4;
const LENGTH: u64 = ROW * HEIGHT;

/// Where the reads and the solid draws draw: `[8, 24) x [4, 12)`.
const RECT: ScissorRect = ScissorRect {
    x: 8,
    y: 4,
    width: 16,
    height: 8,
};

/// The host pages' length, and where the corners lie in them, past the bytes the blit writes.
const PAGES: usize = 128 << 10;
const CORNERS_AT: u64 = 64 << 10;

/// The texture's bytes where every texel outside [`RECT`] holds `outside` and every one inside
/// it `inside`, in each channel.
fn image(outside: u8, inside: u8) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(LENGTH as usize);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let rows = RECT.y..RECT.y + RECT.height;
            let columns = RECT.x..RECT.x + RECT.width;
            let value = match rows.contains(&y) && columns.contains(&x) {
                true => inside,
                false => outside,
            };
            bytes.extend([value; 4]);
        }
    }
    bytes
}

/// The corners of the draws: the whole target, which [`RECT`] scissors.
fn corners() -> Vec<u8> {
    [-1.0f32, -1.0, 1.0, 1.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect()
}

/// What a pass draws through [`RECT`]: a solid value in every channel, or the read.
#[derive(Clone, Copy)]
enum Draw {
    Solid(u8),
    Read,
}

/// The pipelines and queue of the lane.
struct Tools {
    queue: CommandQueue,
    solid: RenderPipelineState,
    read: RenderPipelineState,
}

impl Tools {
    fn new(device: &Device) -> Tools {
        let pipeline = |name: &str| {
            let mut d =
                RenderPipelineDescriptor::new(function(device, "tob_rect", "tob_renders.ll"));
            d.fragment = Some(function(device, name, "tob_renders.ll"));
            d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
            device
                .new_render_pipeline_state(&d, &PipelineOptions::default())
                .unwrap()
        };
        Tools {
            queue: device.new_command_queue().unwrap(),
            solid: pipeline("tob_solid"),
            read: pipeline("tob_read"),
        }
    }

    /// Encode into `command_buffer` a pass over `texture`, cleared to `clear` where it is given,
    /// else loaded, that draws `draw`, its corners from `buffer` at `offset`.
    fn encode(
        &self,
        command_buffer: &CommandBuffer,
        texture: &Texture,
        (clear, draw): (Option<u8>, Draw),
        (buffer, offset): (&Buffer, u64),
    ) {
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(texture.clone());
        color.store_action = StoreAction::Store;
        match clear {
            Some(value) => {
                let unorm = f64::from(value) / 255.0;
                color.load_action = LoadAction::Clear;
                color.clear_color = ClearColor::new(unorm, unorm, unorm, unorm);
            }
            None => color.load_action = LoadAction::Load,
        }
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        match draw {
            Draw::Read => {
                encoder.set_render_pipeline_state(&self.read).unwrap();
                encoder.set_fragment_texture(Some(texture), 0).unwrap();
            }
            Draw::Solid(value) => {
                encoder.set_render_pipeline_state(&self.solid).unwrap();
                let unorm = f32::from(value) / 255.0;
                let rgba: Vec<u8> = [unorm; 4].iter().flat_map(|f| f.to_le_bytes()).collect();
                encoder.set_fragment_bytes(&rgba, 0).unwrap();
            }
        }
        (encoder.set_vertex_buffer(Some(buffer), offset, 0)).unwrap();
        encoder.set_scissor_rect(RECT).unwrap();
        (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
        encoder.end_encoding().unwrap();
    }

    /// Encode into `command_buffer` a copy of `texture` into `into`, at its start.
    fn encode_read(command_buffer: &CommandBuffer, texture: &Texture, into: &Buffer) {
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        (blit.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(WIDTH, HEIGHT, 1),
            into,
            0,
            ROW,
            LENGTH,
            BlitOption::None,
        ))
        .unwrap();
        blit.end_encoding().unwrap();
    }

    /// `texture`'s bytes, through a copy in a command buffer of its own.
    fn bytes(&self, texture: &Texture) -> Vec<u8> {
        let device = self.queue.device();
        let back = (device.new_buffer(LENGTH, ResourceOptions::shared())).unwrap();
        let command_buffer = self.queue.command_buffer().unwrap();
        Tools::encode_read(&command_buffer, texture, &back);
        run(&command_buffer);
        let bytes = back.contents().unwrap().as_ptr();
        // SAFETY: the buffer's bytes, which no command writes now.
        unsafe { std::slice::from_raw_parts(bytes, LENGTH as usize) }.to_vec()
    }
}

/// Commit `command_buffer`, wait for it, and check it completed without error.
fn run(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// The counts the lane reads, between two snapshots: the texels copied in to an image's shadow
/// (`shadow_texels_copied_in_images`), the turns (`attached_read_swaps`), the copies in for a
/// pass that turned nothing beside another command buffer (`attached_read_unswapped_busy`), and
/// the encoders a rebind encoded again (`encoders_reencoded`) or refused to
/// (`reencodes_refused`).
#[derive(Debug, Default, PartialEq, Eq)]
struct Moved {
    copied_in: u64,
    swaps: u64,
    busy: u64,
    reencoded: u64,
    refused: u64,
}

impl Moved {
    fn of(before: &WorkCounts, after: &WorkCounts) -> Moved {
        let delta = |count: fn(&WorkCounts) -> u64| count(after) - count(before);
        Moved {
            copied_in: delta(|c| c.shadow_texels_copied_in_images),
            swaps: delta(|c| c.attached_read_swaps),
            busy: delta(|c| c.attached_read_unswapped_busy),
            reencoded: delta(|c| c.encoders_reencoded),
            refused: delta(|c| c.reencodes_refused),
        }
    }
}

/// What a lane moved and read: its logged pass's move, its command buffer's (from its encoding to
/// its completion), the bytes its blit read through its encoding again, and the texture's after.
struct Lane {
    logged: Moved,
    again: Moved,
    blit: Vec<u8>,
    after: Vec<u8>,
}

/// The lane of the module docs over a buffer over `first`, rebound over `second`, beside an open
/// command buffer where `busy`; `None`, printed, where the device imports no host memory.
fn lane(
    device: &Device,
    tools: &Tools,
    [first, second]: [&Mapping; 2],
    busy: bool,
) -> Option<Lane> {
    for pages in [first, second] {
        pages.write(CORNERS_AT as usize, &corners());
    }
    // SAFETY: the mapping outlives the device (made before it, dropped after it), and only the
    // buffer's commands use it while the buffer lives.
    let made = unsafe {
        no_copy(
            device,
            first.pointer(),
            PAGES,
            ResourceOptions::default(),
            None,
        )
    };
    let pages = match made {
        Ok(pages) => pages,
        Err(error) if not_importable(&error) => {
            eprintln!("LOGGED-FACE-TURNS: the device imports no host memory: {error:?}");
            return None;
        }
        Err(error) => panic!("{error:?}"),
    };
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, WIDTH, HEIGHT, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    let texture = device.new_texture(&descriptor).unwrap();
    let open = busy.then(|| tools.queue.command_buffer().unwrap());
    // Each pass takes its corners from the pages, as the command buffer behind the wait does:
    // its draws' pipelines are made before that command buffer's commit, which a rebind refuses
    // while one is still being made (`RebindProblem::EncodedUses`).
    let corners_in = (&pages, CORNERS_AT);
    let pass = |clear: Option<u8>, draw: Draw| {
        let before = device.work_counts();
        let command_buffer = tools.queue.command_buffer().unwrap();
        tools.encode(&command_buffer, &texture, (clear, draw), corners_in);
        run(&command_buffer);
        Moved::of(&before, &device.work_counts())
    };
    pass(Some(0x10), Draw::Solid(0x40));
    pass(Some(0x20), Draw::Read);
    pass(None, Draw::Solid(0x50));
    let logged = pass(Some(0x30), Draw::Read);
    pass(None, Draw::Solid(0x62));
    let before = device.work_counts();
    let gate = device.new_shared_event().unwrap();
    let command_buffer = tools.queue.command_buffer().unwrap();
    (command_buffer.encode_wait_for_event(EventRef::Shared(&gate), 1)).unwrap();
    Tools::encode_read(&command_buffer, &texture, &pages);
    tools.encode(
        &command_buffer,
        &texture,
        (Some(0x90), Draw::Read),
        corners_in,
    );
    command_buffer.commit().unwrap();
    // SAFETY: as above.
    let memory = unsafe { HostMemory::new(second.pointer()) };
    // SAFETY: as above.
    unsafe { pages.rebind_bytes_no_copy(memory, None, None) }.unwrap();
    gate.set_signaled_value(1);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let again = Moved::of(&before, &device.work_counts());
    let blit = second.read(0..LENGTH as usize);
    drop(open);
    let after = tools.bytes(&texture);
    Some(Lane {
        logged,
        again,
        blit,
        after,
    })
}

/// The module docs' lane, turned and beside an open command buffer: the same bytes, Metal's.
#[test]
fn a_logged_pass_turns_and_its_encoding_again_reads_the_texels_from_before_it() {
    // Made before the device, dropped after it.
    let mappings: [Mapping; 4] = std::array::from_fn(|_| Mapping::anonymous(PAGES));
    on_device(|device| {
        let tools = Tools::new(device);
        let Some(turned) = lane(device, &tools, [&mappings[0], &mappings[1]], false) else {
            return;
        };
        let copied = lane(device, &tools, [&mappings[2], &mappings[3]], true).unwrap();
        for (lane, name) in [(&turned, "turned"), (&copied, "copied")] {
            eprintln!(
                "LOGGED-FACE-TURNS {name}: the logged pass {:?}; the command buffer encoded \
                 again {:?}",
                lane.logged, lane.again
            );
            assert_eq!(
                (lane.again.reencoded, lane.again.refused),
                (2, 0),
                "{name}: the blit and the pass encoded again"
            );
            let differ =
                |got: &[u8], want: &[u8]| (0..got.len()).filter(|&at| got[at] != want[at]).count();
            let blit = differ(&lane.blit, &image(0x30, 0x62));
            assert_eq!(
                blit, 0,
                "{name}: the blit encoded again: {blit} bytes differ"
            );
            let after = differ(&lane.after, &image(0x90, 0x62));
            assert_eq!(
                after, 0,
                "{name}: the pass encoded again: {after} bytes differ"
            );
        }
        assert_eq!(
            (turned.logged.swaps, turned.logged.copied_in),
            (1, 0),
            "the logged pass turns, with no copy in"
        );
        assert_eq!(
            (turned.again.swaps, turned.again.copied_in),
            (1, 0),
            "the pass turns, and its encoding again takes the turn again with no copy in"
        );
        assert_eq!(
            (copied.logged.swaps, copied.logged.busy, copied.again.swaps),
            (0, 1, 0),
            "beside an open command buffer, no pass turns"
        );
        assert!(
            copied.again.copied_in > 0,
            "beside an open command buffer, the pass copies in"
        );
    });
}
