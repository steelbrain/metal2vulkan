//! Device tests of a memoryless texture's uses past a render pass's attachment, on the device the
//! cfg selects, under the validation layer.
//!
//! Plain Metal binds a memoryless texture at any texture slot (an M3 Max under macOS 27.0.1, its
//! API validation off; the layer stops the binding). A command that uses it fails the command
//! buffer with `MTLCommandBufferErrorInvalidResource` (9), from the submission that holds the
//! command on, as Metal splits the command buffer: a render pass alone, a run of blit or of
//! compute encoders that no event command splits. The submissions before it run and their event
//! signals fire; nothing from it on runs, a render pass's clear included; the queue's next command
//! buffer runs. A draw or dispatch uses it where its function statically uses the slot, whether or
//! not its run reads it, and of nothing too; `use_resource` and a blit into or out of it use it. A
//! binding no function uses, one replaced before the draw, a memory barrier or a hint fail nothing.
//! Each shape here is one the probes ran (`conformance/probes/memoryless_*.m`), held to its answer,
//! or the same question in a form the lane's fixtures state: a slot rebound to another texture
//! rather than to nil, and a texture read only on pixels the draw never covers rather than behind
//! a flag that is never set.

use super::attachment_reads::{corners, floats, function};
use super::degrade_notices::on_device;
use super::degrade_notices::on_device_with;
use super::{at_plan, buffer_with, read_buffer, WAIT};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{
    PixelFormat, PrimitiveType, RenderStages, ResourceUsage, TextureType, TextureUsage,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor,
};
use metal2vulkan_runtime::{
    BlitCommandEncoder, BlitOption, Buffer, ClearColor, CommandBuffer, CommandBufferError,
    CommandBufferStatus, CommandQueue, ComputeCommandEncoder, ComputePipelineState, CpuCacheMode,
    Device, Error, EventRef, HazardTrackingMode, LoadAction, Nil, Origin, PipelineOptions, Problem,
    Region, RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState, ResourceOptions,
    SharedEvent, Size, StorageMode, StoreAction, Texture, TextureDescriptor,
};

/// The textures' side, in pixels: `a4_fragment_select_sample` reads a 16² target's texels.
const SIDE: u64 = 16;

/// The target's texels before the pass: opaque red 64.
const BEFORE: [u8; 4] = [64, 0, 0, 255];

/// The pass's clear: opaque green.
const CLEARED: [u8; 4] = [0, 255, 0, 255];

/// What one command buffer left: its status and error, its marks (words), and each event's
/// signaled value.
#[derive(Debug, PartialEq)]
struct Ran {
    status: CommandBufferStatus,
    error: Option<CommandBufferError>,
    marks: Vec<u32>,
    events: Vec<u64>,
}

/// The failure a memoryless texture's use leaves.
const FAILED: (CommandBufferStatus, Option<CommandBufferError>) = (
    CommandBufferStatus::Error,
    Some(CommandBufferError::InvalidResource),
);

/// A command buffer that completes.
const COMPLETED: (CommandBufferStatus, Option<CommandBufferError>) =
    (CommandBufferStatus::Completed, None);

/// A memoryless 16² `RGBA8Unorm` texture of `levels` levels a draw reads, a kernel writes and a
/// pass renders.
fn memoryless(device: &Device, levels: u64) -> Texture {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    d.mipmap_level_count = levels;
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    d.resource_options = ResourceOptions::new(
        StorageMode::Memoryless,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    );
    let texture = device.new_texture(&d).unwrap();
    assert_eq!(texture.storage_mode(), StorageMode::Memoryless);
    texture
}

/// A `Shared` 16² `RGBA8Unorm` texture, every texel `texel`.
fn shared(device: &Device, texel: [u8; 4]) -> Texture {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&d).unwrap();
    let texels = texel.repeat((SIDE * SIDE) as usize);
    let region = Region::new_2d(0, 0, SIDE, SIDE);
    texture
        .replace_region(region, 0, 0, &texels, SIDE * 4, 0)
        .unwrap();
    texture
}

/// Every texel of `texture`, tight rows.
fn texels(texture: &Texture) -> Vec<u8> {
    let mut bytes = vec![0; (SIDE * SIDE * 4) as usize];
    let region = Region::new_2d(0, 0, SIDE, SIDE);
    texture
        .get_bytes(&mut bytes, SIDE * 4, 0, region, 0, 0)
        .unwrap();
    bytes
}

/// The `tob_rect` state with the fragment function `name` of `fixture`, into `RGBA8Unorm`.
fn state(device: &Device, name: &str, fixture: &str) -> RenderPipelineState {
    let mut d = RenderPipelineDescriptor::new(function(device, "tob_rect", "tob_renders.ll"));
    d.fragment = Some(function(device, name, fixture));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    device
        .new_render_pipeline_state(&d, &PipelineOptions::default())
        .unwrap()
}

/// The kernel `name` of `fixture`.
fn kernel(device: &Device, name: &str, fixture: &str) -> ComputePipelineState {
    let d = ComputePipelineDescriptor::new(function(device, name, fixture));
    device
        .new_compute_pipeline_state(&d, &PipelineOptions::default())
        .unwrap()
}

/// Commit `command_buffer` of `queue`, wait for it, and commit the queue's next command buffer,
/// which fills the last of `words` marks with `0x77` and completes: what the first left.
fn ran(
    queue: &CommandQueue,
    command_buffer: &CommandBuffer,
    marks: &Buffer,
    words: u64,
    events: &[&SharedEvent],
) -> Ran {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let next = queue.command_buffer().unwrap();
    let mut blit = next.blit_command_encoder().unwrap();
    let last = (words - 1) * 4;
    blit.fill_buffer(marks, last..last + 4, 0x77).unwrap();
    blit.end_encoding().unwrap();
    next.commit().unwrap();
    next.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        next.status(),
        CommandBufferStatus::Completed,
        "the next one"
    );
    let marks = read_buffer(marks, 0, words * 4)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect();
    Ran {
        status: command_buffer.status(),
        error: command_buffer.error(),
        marks,
        events: events.iter().map(|event| event.signaled_value()).collect(),
    }
}

/// A blit encoder of `command_buffer` that fills mark `at` with `value`.
fn fill(command_buffer: &CommandBuffer, marks: &Buffer, at: u64, value: u8) {
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(marks, at * 4..at * 4 + 4, value).unwrap();
    blit.end_encoding().unwrap();
}

/// A word of four `value` bytes.
fn word(value: u8) -> u32 {
    u32::from_le_bytes([value; 4])
}

/// The probe's draw shapes (`memoryless_bind.m`): one command buffer of a blit filling mark 0, a
/// signal of event `before`, a pass over a target holding [`BEFORE`] that clears it to
/// [`CLEARED`] and draws the shape's draws over it, a blit filling mark 1, and a signal of event
/// `after`. Where a draw's function statically uses a slot that holds the memoryless texture, an
/// empty draw's and one that reads it only on pixels the draw never covers too, the pass runs
/// nothing, the target keeping [`BEFORE`], and neither does the blit after it nor the signal after
/// it; the blit and the signal before it run. Otherwise it completes: a draw that leaves it
/// unused, one after rebinding the slot to nil, and a pass of no draw.
#[test]
fn a_draw_that_uses_a_memoryless_texture_fails_its_pass_on() {
    on_device(|device, _messages| {
        let queue = device.new_command_queue().unwrap();
        let read = state(device, "tob_read", "tob_renders.ll");
        let solid = state(device, "tob_solid", "tob_renders.ll");
        let select = state(
            device,
            "a4_fragment_select_sample",
            "a4_fragment_select_sample.ll",
        );
        let sampler = device.new_sampler_state(&SamplerDescriptor::new()).unwrap();
        let other = shared(device, [0, 0, 255, 255]);
        let texture = memoryless(device, 1);
        let blue = floats([0.0, 0.0, 1.0, 1.0]);
        // A draw of `pipeline` over the target's pixels `rect`, of `vertices`.
        let draw = |encoder: &mut RenderCommandEncoder,
                    pipeline: &RenderPipelineState,
                    rect: [u64; 4],
                    vertices: u32| {
            encoder.set_render_pipeline_state(pipeline).unwrap();
            encoder.set_vertex_bytes(&corners16(rect), 0).unwrap();
            encoder.set_fragment_bytes(&blue, 0).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, vertices, 1, 0)
                .unwrap();
        };
        let whole = [0, 0, SIDE, SIDE];
        type Shape<'a> = (&'a str, Box<dyn Fn(&mut RenderCommandEncoder) + 'a>, bool);
        let shapes: Vec<Shape> = vec![
            (
                "read",
                Box::new(|e| {
                    e.set_fragment_texture(Some(&texture), 0).unwrap();
                    draw(e, &read, whole, 6);
                }),
                true,
            ),
            (
                "empty",
                Box::new(|e| {
                    e.set_fragment_texture(Some(&texture), 0).unwrap();
                    draw(e, &read, whole, 0);
                }),
                true,
            ),
            // `a` (texture 0) on pixels whose centre's x is past 8 alone: the draw covers none.
            (
                "branch",
                Box::new(|e| {
                    e.set_fragment_texture(Some(&texture), 0).unwrap();
                    e.set_fragment_texture(Some(&other), 1).unwrap();
                    e.set_fragment_sampler_state(Some(&sampler), 0).unwrap();
                    draw(e, &select, [0, 0, 8, SIDE], 6);
                }),
                true,
            ),
            (
                "earlier",
                Box::new(|e| {
                    draw(e, &solid, whole, 6);
                    e.set_fragment_texture(Some(&texture), 0).unwrap();
                    draw(e, &read, whole, 6);
                }),
                true,
            ),
            (
                "unused",
                Box::new(|e| {
                    e.set_fragment_texture(Some(&texture), 0).unwrap();
                    draw(e, &solid, whole, 6);
                }),
                false,
            ),
            (
                "rebind",
                Box::new(|e| {
                    e.set_fragment_texture(Some(&texture), 0).unwrap();
                    e.set_fragment_texture(Some(&other), 0).unwrap();
                    draw(e, &read, whole, 6);
                }),
                false,
            ),
            (
                "rebind_nil",
                Box::new(|e| {
                    e.set_fragment_texture(Some(&texture), 0).unwrap();
                    e.set_fragment_texture(None, 0).unwrap();
                    draw(e, &solid, whole, 6);
                }),
                false,
            ),
            (
                "no_draw",
                Box::new(|e| {
                    e.set_fragment_texture(Some(&texture), 0).unwrap();
                }),
                false,
            ),
            // `use_resource` of it fails the pass with no draw after it.
            (
                "use_resource",
                Box::new(|e| {
                    let (usage, stages) = (ResourceUsage::READ, RenderStages::FRAGMENT);
                    e.use_resource(texture.as_resource(), usage, stages)
                        .unwrap();
                }),
                true,
            ),
        ];
        for (name, shape, fails) in shapes {
            let target = shared(device, BEFORE);
            let marks = buffer_with(device, &[0u8; 12]).unwrap();
            let before = device.new_shared_event().unwrap();
            let after = device.new_shared_event().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            fill(&command_buffer, &marks, 0, 0x11);
            command_buffer
                .encode_signal_event(EventRef::from(&before), 1)
                .unwrap();
            let mut pass = RenderPassDescriptor::new();
            let color = &mut pass.color_attachments[0];
            color.texture = Some(target.clone());
            color.load_action = LoadAction::Clear;
            color.clear_color = ClearColor::new(0.0, 1.0, 0.0, 1.0);
            color.store_action = StoreAction::Store;
            let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
            shape(&mut encoder);
            encoder.end_encoding().unwrap();
            fill(&command_buffer, &marks, 1, 0x22);
            command_buffer
                .encode_signal_event(EventRef::from(&after), 1)
                .unwrap();
            let ran = ran(&queue, &command_buffer, &marks, 3, &[&before, &after]);
            let left = texels(&target);
            if fails {
                assert_eq!((ran.status, ran.error), FAILED, "{name}");
                assert_eq!(ran.marks, [word(0x11), 0, word(0x77)], "{name}");
                assert_eq!(ran.events, [1, 0], "{name}");
                assert_eq!(left, BEFORE.repeat((SIDE * SIDE) as usize), "{name}");
            } else {
                assert_eq!((ran.status, ran.error), COMPLETED, "{name}");
                assert_eq!(ran.marks, [word(0x11), word(0x22), word(0x77)], "{name}");
                assert_eq!(ran.events, [1, 1], "{name}");
                let drawn = match name {
                    "no_draw" => CLEARED,
                    _ => [0, 0, 255, 255],
                };
                assert_eq!(left, drawn.repeat((SIDE * SIDE) as usize), "{name}");
            }
        }
    });
}

/// [`corners`] of a 16² target: the lane's helper is of a 32² one, so each pixel bound doubles.
fn corners16([x0, y0, x1, y1]: [u64; 4]) -> Vec<u8> {
    corners([2 * x0, 2 * y0, 2 * x1, 2 * y1])
}

/// One encoder of the probe's sequences (`memoryless_split.m`).
#[derive(Clone, Copy, Debug)]
enum Step {
    /// A compute encoder storing 42 into its mark (`store_const`).
    Compute,
    /// A compute encoder whose kernel writes the memoryless texture (`tob_two_ways`).
    ComputeUses,
    /// A blit encoder filling its mark with `0x11`.
    Blit,
    /// A blit encoder copying the memoryless texture into a buffer.
    BlitUses,
    /// A signal of the sequence's event.
    Signal,
}

/// The sequence `steps` in one command buffer, each encoder with a mark of its own, then the
/// queue's next: each encoder's mark, its own value where it ran and 0 where not, and the event's
/// signaled value.
fn sequence(device: &Device, steps: &[Step]) -> Ran {
    let store = kernel(device, "store_const", "store_const.ll");
    let writes = kernel(device, "tob_two_ways", "tob_renders.ll");
    let texture = memoryless(device, 1);
    let words = steps.len() as u64 + 1;
    let marks = buffer_with(device, &vec![0u8; (words * 4) as usize]).unwrap();
    let scratch = buffer_with(device, &vec![0u8; (SIDE * SIDE * 4) as usize]).unwrap();
    let event = device.new_shared_event().unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let one = Size::new(1, 1, 1);
    for (at, step) in steps.iter().enumerate() {
        let at = at as u64;
        match step {
            Step::Compute | Step::ComputeUses => {
                let mut compute = command_buffer.compute_command_encoder().unwrap();
                let uses = matches!(step, Step::ComputeUses);
                compute
                    .set_compute_pipeline_state(if uses { &writes } else { &store })
                    .unwrap();
                compute.set_buffer(Some(&marks), at * 4, 0).unwrap();
                if uses {
                    compute.set_texture(Some(&texture), 0).unwrap();
                    compute.set_bytes(&0u32.to_le_bytes(), 1).unwrap();
                }
                compute.dispatch_threadgroups(one, one).unwrap();
                compute.end_encoding().unwrap();
            }
            Step::Blit => fill(&command_buffer, &marks, at, 0x11),
            Step::BlitUses => {
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                copy_out(&mut blit, &texture, &scratch, SIDE).unwrap();
                blit.end_encoding().unwrap();
            }
            Step::Signal => command_buffer
                .encode_signal_event(EventRef::from(&event), 1)
                .unwrap(),
        }
    }
    ran(&queue, &command_buffer, &marks, words, &[&event])
}

/// A copy of `texture`'s `side`² texels at its origin into `buffer`, tight rows.
fn copy_out(
    blit: &mut BlitCommandEncoder,
    texture: &Texture,
    buffer: &Buffer,
    side: u64,
) -> Result<(), Error> {
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(side, side, 1),
        buffer,
        0,
        SIDE * 4,
        SIDE * SIDE * 4,
        BlitOption::None,
    )
}

/// The probe's sequences: a use of the memoryless texture fails the run of encoders of its kind it
/// lies in, which an encoder of the other kind or an event signal ends, and everything after it;
/// what came before that run runs, and a signal encoded before it fires.
#[test]
fn a_memoryless_use_fails_its_run_of_encoders_on() {
    use Step::{Blit, BlitUses, Compute, ComputeUses, Signal};
    on_device_with(at_plan, |device, _messages| {
        let (stored, filled, last) = (42, word(0x11), word(0x77));
        let cases: [(&str, &[Step], &[u32], u64); 7] = [
            ("CcC", &[Compute, ComputeUses, Compute], &[0, 0, 0], 0),
            ("BbB", &[Blit, BlitUses, Blit], &[0, 0, 0], 0),
            ("CbC", &[Compute, BlitUses, Compute], &[stored, 0, 0], 0),
            ("BcB", &[Blit, ComputeUses, Blit], &[filled, 0, 0], 0),
            (
                "BSbB",
                &[Blit, Signal, BlitUses, Blit],
                &[filled, 0, 0, 0],
                1,
            ),
            ("bSB", &[BlitUses, Signal, Blit], &[0, 0, 0], 0),
            ("CSc", &[Compute, Signal, ComputeUses], &[stored, 0, 0], 1),
        ];
        for (name, steps, marks, event) in cases {
            let ran = sequence(device, steps);
            assert_eq!((ran.status, ran.error), FAILED, "{name}");
            let mut want = marks.to_vec();
            want.push(last);
            assert_eq!(ran.marks, want, "{name}");
            assert_eq!(ran.events, [event], "{name}");
        }
        // Without a use: every encoder runs and the signal fires.
        let ran = sequence(device, &[Blit, Signal, Compute]);
        assert_eq!((ran.status, ran.error), COMPLETED);
        assert_eq!(ran.marks, [filled, 0, stored, last]);
        assert_eq!(ran.events, [1]);
    });
}

/// The probe's compute and blit uses past a sequence's (`memoryless_uses.m`,
/// `memoryless_edges.m`, `memoryless_more.m`): in one command buffer, a blit encoder filling mark
/// 0, then an encoder of the case's kind that fills or stores mark 1 before the case's command and
/// mark 2 after it, then a blit encoder filling mark 3. A use fails the whole command buffer (the
/// blits are of one run with a blit case, and a compute case's run is the case alone); the rest
/// complete.
#[test]
fn memoryless_compute_and_blit_uses_fail_as_metal_s() {
    on_device(|device, _messages| {
        let store = kernel(device, "store_const", "store_const.ll");
        let writes = kernel(device, "tob_two_ways", "tob_renders.ll");
        let texture = memoryless(device, 1);
        let mipmapped = memoryless(device, 3);
        let other = shared(device, BEFORE);
        let scratch = buffer_with(device, &vec![0xc3u8; (SIDE * SIDE * 4) as usize]).unwrap();
        let origin = Origin::new(0, 0, 0);
        let size = Size::new(SIDE, SIDE, 1);
        type Blit<'a> = Box<dyn Fn(&mut BlitCommandEncoder) -> Result<(), Error> + 'a>;
        let blits: Vec<(&str, Blit, bool)> = vec![
            (
                "to_buffer",
                Box::new(|b| copy_out(b, &texture, &scratch, SIDE)),
                true,
            ),
            (
                "of_nothing",
                Box::new(|b| copy_out(b, &texture, &scratch, 0)),
                true,
            ),
            (
                "from_buffer",
                Box::new(|b| {
                    b.copy_from_buffer_to_texture(
                        &scratch,
                        0,
                        SIDE * 4,
                        SIDE * SIDE * 4,
                        size,
                        &texture,
                        0,
                        0,
                        origin,
                        BlitOption::None,
                    )
                }),
                true,
            ),
            (
                "from_texture",
                Box::new(|b| {
                    b.copy_from_texture_to_texture(
                        &texture, 0, 0, origin, size, &other, 0, 0, origin,
                    )
                }),
                true,
            ),
            (
                "to_texture",
                Box::new(|b| {
                    b.copy_from_texture_to_texture(
                        &other, 0, 0, origin, size, &texture, 0, 0, origin,
                    )
                }),
                true,
            ),
            (
                "whole",
                Box::new(|b| b.copy_texture(&texture, &other)),
                true,
            ),
            (
                "mipmaps",
                Box::new(|b| b.generate_mipmaps(&mipmapped)),
                true,
            ),
            (
                "no_slices",
                Box::new(|b| b.copy_texture_slices(&texture, 0, 0, &other, 0, 0, 0, 1)),
                false,
            ),
            (
                "one_level_mipmaps",
                Box::new(|b| b.generate_mipmaps(&texture)),
                false,
            ),
            (
                "optimize_gpu",
                Box::new(|b| b.optimize_contents_for_gpu_access(&texture)),
                false,
            ),
            (
                "optimize_cpu",
                Box::new(|b| b.optimize_contents_for_cpu_access(&texture)),
                false,
            ),
            (
                "synchronize",
                Box::new(|b| b.synchronize_resource(texture.as_resource())),
                false,
            ),
        ];
        let queue = device.new_command_queue().unwrap();
        for (name, case, fails) in blits {
            let marks = buffer_with(device, &[0u8; 20]).unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            fill(&command_buffer, &marks, 0, 0x11);
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(&marks, 4..8, 0x22).unwrap();
            assert_eq!(case(&mut blit), Ok(()), "{name}");
            blit.fill_buffer(&marks, 8..12, 0x33).unwrap();
            blit.end_encoding().unwrap();
            fill(&command_buffer, &marks, 3, 0x44);
            let ran = ran(&queue, &command_buffer, &marks, 5, &[]);
            let last = word(0x77);
            if fails {
                assert_eq!((ran.status, ran.error), FAILED, "{name}");
                assert_eq!(ran.marks, [0, 0, 0, 0, last], "{name}");
            } else {
                assert_eq!((ran.status, ran.error), COMPLETED, "{name}");
                let all = [word(0x11), word(0x22), word(0x33), word(0x44), last];
                assert_eq!(ran.marks, all, "{name}");
            }
        }
        type Compute<'a> = Box<dyn Fn(&mut ComputeCommandEncoder) + 'a>;
        let one = Size::new(1, 1, 1);
        let computes: Vec<(&str, Compute, bool)> = vec![
            (
                "read",
                Box::new(|c| {
                    c.set_compute_pipeline_state(&writes).unwrap();
                    c.set_texture(Some(&texture), 0).unwrap();
                    c.set_bytes(&0u32.to_le_bytes(), 1).unwrap();
                    c.dispatch_threadgroups(one, one).unwrap();
                }),
                true,
            ),
            (
                "no_threadgroups",
                Box::new(|c| {
                    c.set_compute_pipeline_state(&writes).unwrap();
                    c.set_texture(Some(&texture), 0).unwrap();
                    c.set_bytes(&0u32.to_le_bytes(), 1).unwrap();
                    c.dispatch_threadgroups(Size::new(0, 1, 1), one).unwrap();
                }),
                true,
            ),
            (
                "empty_threadgroup",
                Box::new(|c| {
                    c.set_compute_pipeline_state(&writes).unwrap();
                    c.set_texture(Some(&texture), 0).unwrap();
                    c.set_bytes(&0u32.to_le_bytes(), 1).unwrap();
                    c.dispatch_threadgroups(one, Size::new(0, 0, 0)).unwrap();
                }),
                true,
            ),
            (
                "use_resource",
                Box::new(|c| {
                    let usage = ResourceUsage::READ;
                    c.use_resource(texture.as_resource(), usage).unwrap();
                }),
                true,
            ),
            (
                "unused",
                Box::new(|c| {
                    c.set_texture(Some(&texture), 0).unwrap();
                }),
                false,
            ),
            (
                "barrier",
                Box::new(|c| {
                    let resources = [texture.as_resource()];
                    c.memory_barrier_with_resources(&resources).unwrap();
                }),
                false,
            ),
        ];
        for (name, case, fails) in computes {
            let marks = buffer_with(device, &[0u8; 20]).unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            fill(&command_buffer, &marks, 0, 0x11);
            let mut compute = command_buffer.compute_command_encoder().unwrap();
            compute.set_compute_pipeline_state(&store).unwrap();
            compute.set_buffer(Some(&marks), 4, 0).unwrap();
            compute.dispatch_threadgroups(one, one).unwrap();
            case(&mut compute);
            compute.set_compute_pipeline_state(&store).unwrap();
            compute.set_buffer(Some(&marks), 8, 0).unwrap();
            compute.dispatch_threadgroups(one, one).unwrap();
            compute.end_encoding().unwrap();
            fill(&command_buffer, &marks, 3, 0x44);
            let ran = ran(&queue, &command_buffer, &marks, 5, &[]);
            let last = word(0x77);
            if fails {
                assert_eq!((ran.status, ran.error), FAILED, "{name}");
                assert_eq!(ran.marks, [word(0x11), 0, 0, 0, last], "{name}");
            } else {
                assert_eq!((ran.status, ran.error), COMPLETED, "{name}");
                let all = [word(0x11), 42, 42, word(0x44), last];
                assert_eq!(ran.marks, all, "{name}");
            }
        }
    });
}

/// A memoryless texture's CPU calls move nothing and its views are nil, as plain Metal's
/// (`memoryless_uses.m`, an M3 Max under macOS 27.0.1).
#[test]
fn memoryless_cpu_access_moves_nothing_and_views_are_nil() {
    on_device(|device, _messages| {
        let texture = memoryless(device, 1);
        let region = Region::new_2d(0, 0, SIDE, SIDE);
        let written = vec![0x5a; (SIDE * SIDE * 4) as usize];
        let mut read = vec![0xc3; (SIDE * SIDE * 4) as usize];
        assert_eq!(
            texture.replace_region(region, 0, 0, &written, SIDE * 4, 0),
            Ok(())
        );
        assert_eq!(
            texture.get_bytes(&mut read, SIDE * 4, 0, region, 0, 0),
            Ok(())
        );
        assert!(read.iter().all(|&byte| byte == 0xc3), "get_bytes wrote");
        let nil = |call| Err(Error::Nil(Nil::new(call, Problem::StorageMode)));
        assert_eq!(
            texture
                .new_texture_view(PixelFormat::RGBA8UnormSRGB)
                .map(|_| ()),
            nil("new_texture_view")
        );
        assert_eq!(
            texture
                .new_texture_view(PixelFormat::RGBA8Unorm)
                .map(|_| ()),
            nil("new_texture_view")
        );
        assert_eq!(
            texture
                .new_texture_view_with_levels_slices(
                    PixelFormat::RGBA8Unorm,
                    TextureType::Type2D,
                    0..1,
                    0..1
                )
                .map(|_| ()),
            nil("new_texture_view_with_levels_slices")
        );
    });
}
