//! A texture a draw reads where its own render pass renders it (`texture/attached_reads.rs`), a
//! `Private` texture of the device's own memory: its first attached read gives it two faces
//! (`texture/faces.rs`), and each later pass that clears it whole turns it to the other face, its
//! read taking the face from before the pass with no copy in (`render/face_swap.rs`).
//!
//! Each pass clears the texture and draws, through [`RECT`], a solid [`BENEATH`] and then a read
//! of the texel under each pixel that writes what it read over it; the pass before it drew a
//! solid value there. So the texels inside [`RECT`] after the pass are those its read took: the
//! texels from before the pass, the M3 Max's answer and the copy's for a draw after another of
//! its pass with no barrier between (`texture/attached_reads.rs`), and every texel outside it the
//! clear's. The read is the pass's second draw: a pass's first draw that samples a color the pass
//! clears reads the clear, as the x86 Macs answer, through a copy cleared to it. A read of the
//! wrong face takes a value two passes old. Each step checks those bytes and what
//! the pass moved: the texels copied in (`WorkCounts::shadow_texels_copied_in_images`, none for a
//! turned pass) and the turns, reads and copies between the faces it counted.
//!
//! The steps: the first attached read, which makes the faces and copies in; a turn; a blit read
//! after it, of the new face; a turn back; a pass while another command buffer is open, which
//! turns nothing and copies in; a turn whose command buffer commits after one made meanwhile that
//! reads the texture, whose submission copies the face from before the turn into the new one
//! first; and a turn whose command buffer goes uncommitted, after which the next read of the
//! texture takes that copy, and the texture turns again.
//!
//! A second test runs while the device has a buffer over host pages, so every pass logs its calls
//! for a rebind (`replay.rs`): a logged pass turns as any other, and a rebind of the buffer before
//! a turning command buffer's submission encodes again its blit that read the texture before the
//! turn and its turning pass, which read the texels each read at its first encoding
//! (`texture/faces.rs`, **Encoded again**). The same lane beside an open command buffer, where no
//! pass turns and every read copies in, with no rebind, reads the same bytes.
//!
//! A third test executes an indirect command buffer before a pass that clears the texture, in the
//! same command buffer: its command reads the texture through an argument buffer naming its
//! resource ID. The commit records the command, binding the heap sets after the pass opened, so
//! a turn there would have the command read the face the pass turned to, which holds texels two
//! passes old. The pass turns nothing (`Unturned::Busy`) and copies in, and the command reads the
//! texels from before the pass, Metal's answer.
//!
//! A fourth and a fifth test encode a pass beside a command buffer that commits, held as it
//! commits once its last command buffer ended and before the scheduler holds its batches: the
//! committing command buffer counts among the device's uncommitted ones until then
//! (`queue.rs`'s `Uncommitted`), so the pass turns nothing and copies in. A turn there would
//! leave a batch submitted after it that a face fix copies around, though its end may already lie
//! in its last command buffer (the fourth: the submission found that end where the batch does not
//! end), and whose rendering the fix's copy back would overwrite with the face from before (the
//! fifth, whose end is a command buffer of its own while another texture's turn is pending).
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are `tob_rect`,
//! `tob_solid` and `tob_read` of the owned fixture `validation/fixtures/public/engine/tob_renders.ll`,
//! and `rt_heap_zero` of `validation/fixtures/public/engine/kernel_read_through.ll`: `out[i]` is
//! the argument buffer's texture read at `p[i].xy` and level `p[i].w`.

use super::blit_holds::{device, over_host, Host, Scratch, WAIT};
use super::encoder_holds::{bytes_of, library};
use crate::command_buffer::completion::trace::{self, Point};
use crate::{
    BlitOption, Buffer, ClearColor, CommandBuffer, CommandQueue, Device, EventRef,
    IndirectCommandBufferDescriptor, LoadAction, Origin, PipelineOptions, RenderCommandEncoder,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, ScissorRect, Size, StoreAction,
    Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::metal::{
    IndirectCommandType, PixelFormat, PrimitiveType, ResourceUsage, TextureUsage,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

/// The texture: `WIDTH` texels a row of 4 bytes (`BGRA8Unorm`), `HEIGHT` rows.
const WIDTH: u64 = 64;
const HEIGHT: u64 = 16;
const ROW: u64 = WIDTH * 4;
const LENGTH: u64 = ROW * HEIGHT;

/// Where the solid draws and the reads draw: `[8, 24) x [4, 12)`.
const RECT: ScissorRect = ScissorRect {
    x: 8,
    y: 4,
    width: 16,
    height: 8,
};

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

/// The counts a test of the faces reads, between two snapshots.
#[derive(Debug, Default, PartialEq, Eq)]
struct Moved {
    /// `WorkCounts::shadow_texels_copied_in_images`: the copies in of an image's shadow.
    copied_in: u64,
    /// `texture_faces_made`, `attached_read_swaps`, `attached_read_swap_reads`,
    /// `attached_read_swap_fixes`.
    made: u64,
    swaps: u64,
    reads: u64,
    fixes: u64,
    /// `attached_read_unswapped_first` and `attached_read_unswapped_busy`, and every
    /// `attached_read_unswapped_*` in all.
    first: u64,
    busy: u64,
    unswapped: u64,
}

impl Moved {
    fn of(before: &WorkCounts, after: &WorkCounts) -> Moved {
        let delta = |count: fn(&WorkCounts) -> u64| count(after) - count(before);
        let unswapped = |c: &WorkCounts| {
            c.attached_read_unswapped_load
                + c.attached_read_unswapped_levels
                + c.attached_read_unswapped_area
                + c.attached_read_unswapped_storage
                + c.attached_read_unswapped_frozen
                + c.attached_read_unswapped_first
                + c.attached_read_unswapped_busy
                + c.attached_read_unswapped_pass
                + c.attached_read_unswapped_split
        };
        Moved {
            copied_in: delta(|c| c.shadow_texels_copied_in_images),
            made: delta(|c| c.texture_faces_made),
            swaps: delta(|c| c.attached_read_swaps),
            reads: delta(|c| c.attached_read_swap_reads),
            fixes: delta(|c| c.attached_read_swap_fixes),
            first: delta(|c| c.attached_read_unswapped_first),
            busy: delta(|c| c.attached_read_unswapped_busy),
            unswapped: unswapped(after) - unswapped(before),
        }
    }
}

/// What a pass draws through [`RECT`].
#[derive(Clone, Copy)]
enum Draw {
    /// Nothing.
    Nothing,
    /// A solid value in every channel.
    Solid(u8),
    /// A solid [`BENEATH`], then the read of the texel under each pixel, written over it as it
    /// was read.
    Read,
}

/// The solid a pass that reads draws before its read, which the read writes over: no texel the
/// texture holds before a pass.
const BENEATH: u8 = 0xff;

/// The pipelines the passes draw with, and the buffers the reads back copy into.
struct Tools {
    queue: CommandQueue,
    solid: RenderPipelineState,
    read: RenderPipelineState,
    back: Buffer,
    beside: Buffer,
}

impl Tools {
    fn new(device: &Device) -> Tools {
        let renders = library(device, "engine/tob_renders.ll");
        let vertex = renders.new_function_with_name("tob_rect").unwrap();
        let pipeline = |name: &str| {
            let fragment = renders.new_function_with_name(name).unwrap();
            let mut descriptor =
                RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
            descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
            descriptor.color_attachments[0] =
                Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
            let options = PipelineOptions::default();
            device
                .new_render_pipeline_state(&descriptor, &options)
                .unwrap()
        };
        let buffer = || {
            device
                .new_buffer(LENGTH, ResourceOptions::shared())
                .unwrap()
        };
        Tools {
            queue: device.new_command_queue().unwrap(),
            solid: pipeline("tob_solid"),
            read: pipeline("tob_read"),
            back: buffer(),
            beside: buffer(),
        }
    }

    /// Encode into `command_buffer` a pass over `texture`, cleared to `clear` where it is given
    /// (every channel), else loaded, that draws `draw`.
    fn encode(
        &self,
        command_buffer: &CommandBuffer,
        texture: &Texture,
        clear: Option<u8>,
        draw: Draw,
    ) {
        self.encode_over(command_buffer, texture, (clear, draw), None);
    }

    /// As [`encode`](Self::encode), its draw's corners read from `corners_in`, a buffer and the
    /// offset of the corners in it, where it is given, else from bytes set at the call.
    fn encode_over(
        &self,
        command_buffer: &CommandBuffer,
        texture: &Texture,
        (clear, draw): (Option<u8>, Draw),
        corners_in: Option<(&Buffer, u64)>,
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
        let corners = bytes_of(&[-1.0f32, -1.0, 1.0, 1.0], f32::to_le_bytes);
        if !matches!(draw, Draw::Nothing) {
            match corners_in {
                Some((buffer, offset)) => encoder.set_vertex_buffer(Some(buffer), offset, 0),
                None => encoder.set_vertex_bytes(&corners, 0),
            }
            .unwrap();
            encoder.set_scissor_rect(RECT).unwrap();
        }
        let solid = |encoder: &mut RenderCommandEncoder, value: u8| {
            encoder.set_render_pipeline_state(&self.solid).unwrap();
            let unorm = f32::from(value) / 255.0;
            let rgba = bytes_of(&[unorm; 4], f32::to_le_bytes);
            encoder.set_fragment_bytes(&rgba, 0).unwrap();
        };
        match draw {
            Draw::Nothing => {}
            Draw::Read => {
                // The solid the read draws over, so the read is no first draw (the module docs).
                solid(&mut encoder, BENEATH);
                (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
                encoder.set_render_pipeline_state(&self.read).unwrap();
                encoder.set_fragment_texture(Some(texture), 0).unwrap();
            }
            Draw::Solid(value) => solid(&mut encoder, value),
        }
        if !matches!(draw, Draw::Nothing) {
            (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
        }
        encoder.end_encoding().unwrap();
    }

    /// Commit `command_buffer` and wait for it.
    fn run(command_buffer: &CommandBuffer) {
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
    }

    /// [`encode`](Self::encode) in a command buffer of its own, run: what it moved.
    fn pass(&self, texture: &Texture, clear: Option<u8>, draw: Draw) -> Moved {
        let device = self.queue.device();
        let before = device.work_counts();
        let command_buffer = self.queue.command_buffer().unwrap();
        self.encode(&command_buffer, texture, clear, draw);
        Tools::run(&command_buffer);
        Moved::of(&before, &device.work_counts())
    }

    /// Encode into `command_buffer` a copy of `texture` into `into`.
    fn encode_read(command_buffer: &CommandBuffer, texture: &Texture, into: &Buffer) {
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
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
        )
        .unwrap();
        blit.end_encoding().unwrap();
    }

    /// `buffer`'s bytes, which no command writes now.
    fn contents(buffer: &Buffer) -> Vec<u8> {
        let bytes = buffer.contents().unwrap().as_ptr();
        // SAFETY: the buffer's bytes, which no command writes now.
        unsafe { std::slice::from_raw_parts(bytes, LENGTH as usize) }.to_vec()
    }

    /// `texture`'s bytes, through a copy in a command buffer of its own, and what it moved.
    fn bytes(&self, texture: &Texture) -> (Vec<u8>, Moved) {
        let device = self.queue.device();
        let before = device.work_counts();
        let command_buffer = self.queue.command_buffer().unwrap();
        Tools::encode_read(&command_buffer, texture, &self.back);
        Tools::run(&command_buffer);
        (
            Tools::contents(&self.back),
            Moved::of(&before, &device.work_counts()),
        )
    }
}

/// The texture the passes render and read: `Private`, its reads sampled.
fn texture(device: &Device) -> Texture {
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, WIDTH, HEIGHT, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    device.new_texture(&descriptor).unwrap()
}

/// The steps of the module docs: every read takes the texels from before its pass, the clears
/// and solid draws land where the next command reads them, and a turned pass copies nothing in.
#[test]
fn a_texture_s_faces_take_turns_for_its_attached_reads() {
    let scratch = Scratch::new("attached-read-faces");
    let device = device(&scratch.0);
    let tools = Tools::new(&device);
    let texture = texture(&device);
    let check = |want: Vec<u8>, what: &str| {
        let (got, moved) = tools.bytes(&texture);
        assert_eq!(
            moved.fixes, 0,
            "{what}: a read with no turn pending copies no face"
        );
        let differ = (0..got.len()).filter(|&at| got[at] != want[at]).count();
        assert_eq!(differ, 0, "{what}: {differ} bytes differ");
    };
    // 0x10 everywhere, 0x40 inside the rectangle; then the first attached read, which makes the
    // faces and copies the read's texels in.
    let moved = tools.pass(&texture, Some(0x10), Draw::Nothing);
    assert_eq!(moved.made, 0, "a pass that reads nothing makes no faces");
    tools.pass(&texture, None, Draw::Solid(0x40));
    let moved = tools.pass(&texture, Some(0x20), Draw::Read);
    eprintln!("ATTACHED-READ-FACES: the first read: {moved:?}");
    assert!(moved.copied_in > 0, "the first read copies in");
    assert_eq!(
        (moved.made, moved.swaps),
        (1, 0),
        "the first read makes the faces"
    );
    assert_eq!(
        (moved.first, moved.unswapped),
        (1, 1),
        "counted as the first"
    );
    check(image(0x20, 0x40), "the first read");
    // A turn: the read takes 0x50, from before the pass, with no copy in; the blit after it reads
    // the new face.
    tools.pass(&texture, None, Draw::Solid(0x50));
    let moved = tools.pass(&texture, Some(0x30), Draw::Read);
    eprintln!("ATTACHED-READ-FACES: a turn: {moved:?}");
    assert_eq!(moved.copied_in, 0, "the turned pass copies nothing in");
    let turned = Moved {
        swaps: 1,
        reads: 1,
        ..Moved::default()
    };
    assert_eq!(moved, turned, "the turn");
    check(image(0x30, 0x50), "the turned pass");
    // A turn back.
    tools.pass(&texture, None, Draw::Solid(0x60));
    let moved = tools.pass(&texture, Some(0x70), Draw::Read);
    assert_eq!(moved, turned, "the turn back");
    check(image(0x70, 0x60), "the turn back");
    // Another command buffer open: no turn, the copy in.
    tools.pass(&texture, None, Draw::Solid(0x61));
    let open = tools.queue.command_buffer().unwrap();
    let moved = tools.pass(&texture, Some(0x80), Draw::Read);
    drop(open);
    eprintln!("ATTACHED-READ-FACES: beside an open command buffer: {moved:?}");
    assert!(moved.copied_in > 0, "the pass beside it copies in");
    assert_eq!(
        (moved.swaps, moved.busy, moved.unswapped),
        (0, 1, 1),
        "busy"
    );
    check(image(0x80, 0x61), "the pass beside an open command buffer");
    // A turn, and a command buffer made after it that reads the texture and commits first: its
    // submission copies the face from before the turn into the new one, which it reads; the turned
    // pass still reads the texels from before it.
    tools.pass(&texture, None, Draw::Solid(0x62));
    let before = device.work_counts();
    let turning = tools.queue.command_buffer().unwrap();
    tools.encode(&turning, &texture, Some(0x90), Draw::Read);
    let meanwhile = tools.queue.command_buffer().unwrap();
    Tools::encode_read(&meanwhile, &texture, &tools.beside);
    Tools::run(&meanwhile);
    Tools::run(&turning);
    let moved = Moved::of(&before, &device.work_counts());
    eprintln!("ATTACHED-READ-FACES: a turn with a command buffer made meanwhile: {moved:?}");
    assert_eq!(moved.copied_in, 0, "the turned pass copies nothing in");
    assert_eq!(
        (moved.swaps, moved.reads, moved.fixes),
        (1, 1, 1),
        "one fix"
    );
    let read = Tools::contents(&tools.beside);
    let want = image(0x80, 0x62);
    let differ = (0..read.len()).filter(|&at| read[at] != want[at]).count();
    assert_eq!(
        differ, 0,
        "the command buffer made meanwhile: {differ} bytes differ"
    );
    check(
        image(0x90, 0x62),
        "the turned pass after one made meanwhile",
    );
    // A turn whose command buffer goes uncommitted: the next read takes the face from before it,
    // through a fix, and the texture turns again after. The counts' read lowers the encoder's end,
    // which held the command buffer (`render/deferral.rs`).
    tools.pass(&texture, None, Draw::Solid(0x63));
    let before = device.work_counts();
    let dropped = tools.queue.command_buffer().unwrap();
    tools.encode(&dropped, &texture, Some(0xa0), Draw::Read);
    drop(dropped);
    let (got, moved) = tools.bytes(&texture);
    let all = Moved::of(&before, &device.work_counts());
    eprintln!("ATTACHED-READ-FACES: an uncommitted turn: {all:?}");
    assert_eq!(
        (all.swaps, moved.fixes),
        (1, 1),
        "the read after the uncommitted turn"
    );
    let want = image(0x90, 0x63);
    let differ = (0..got.len()).filter(|&at| got[at] != want[at]).count();
    assert_eq!(
        differ, 0,
        "the read after an uncommitted turn: {differ} bytes differ"
    );
    tools.pass(&texture, None, Draw::Solid(0x64));
    let moved = tools.pass(&texture, Some(0xb0), Draw::Read);
    assert_eq!(moved, turned, "the turn after the uncommitted one");
    check(image(0xb0, 0x64), "the turn after the uncommitted one");
}

/// Where the corners lie in the pages of the second test: past the bytes its blit writes.
const CORNERS_AT: u64 = 64 << 10;

/// What a lane of the second test moved and read.
struct Logged {
    /// What its logged pass moved, the one before the command buffer the rebind encodes again.
    logged: Moved,
    /// What that command buffer moved, from its encoding to its completion, and how many of its
    /// encoders the rebind encoded again.
    again: Moved,
    reencoded: u64,
    /// The bytes its blit read, through its encoding again: the texture before the turning pass.
    blit: Vec<u8>,
    /// The texture's bytes after the command buffer.
    after: Vec<u8>,
}

/// A lane of the second test over a buffer over `first`, rebound over `second` (each holding the
/// corners at [`CORNERS_AT`]); where `busy`, beside an open command buffer, so that no pass turns,
/// and not rebound: its pass's first copy in goes to its batch's submission
/// (`shadow_standing.rs`, `shadow::GatedCopyIn`), a host step that keeps what the command buffer's
/// commands took until it completes, so a rebind is refused (`RebindProblem::EncodedUses`).
/// `None`, printed, where the device imports no host memory.
fn logged_lane(
    device: &Device,
    tools: &Tools,
    [first, second]: [&Host; 2],
    busy: bool,
) -> Option<Logged> {
    let pages = over_host(device, first, &Arc::new(AtomicUsize::new(0)))?;
    let corners = [-1.0f32, -1.0, 1.0, 1.0].map(f32::to_bits);
    let at = (CORNERS_AT / 4) as usize;
    first.set(at, &corners);
    second.set(at, &corners);
    let texture = texture(device);
    let open = busy.then(|| tools.queue.command_buffer().unwrap());
    // The faces, made by the first read; then a logged pass that reads 0x50 inside the rectangle.
    // Each pass takes its corners from the pages, as the command buffer behind the wait does.
    let corners_in = Some((&pages, CORNERS_AT));
    let pass = |clear: Option<u8>, draw: Draw| {
        let before = device.work_counts();
        let command_buffer = tools.queue.command_buffer().unwrap();
        tools.encode_over(&command_buffer, &texture, (clear, draw), corners_in);
        Tools::run(&command_buffer);
        Moved::of(&before, &device.work_counts())
    };
    pass(Some(0x10), Draw::Nothing);
    pass(None, Draw::Solid(0x40));
    pass(Some(0x20), Draw::Read);
    pass(None, Draw::Solid(0x50));
    let logged = pass(Some(0x30), Draw::Read);
    pass(None, Draw::Solid(0x62));
    // Behind a wait: a blit of the texture into the pages, then a pass that clears it and reads
    // it, its corners from the pages. The rebind before the signal encodes both again.
    let rebinds = !busy;
    let before = device.work_counts();
    let gate = device.new_shared_event().unwrap();
    let command_buffer = tools.queue.command_buffer().unwrap();
    (command_buffer.encode_wait_for_event(EventRef::Shared(&gate), 1)).unwrap();
    Tools::encode_read(&command_buffer, &texture, &pages);
    tools.encode_over(
        &command_buffer,
        &texture,
        (Some(0x90), Draw::Read),
        corners_in,
    );
    command_buffer.commit().unwrap();
    if rebinds {
        // SAFETY: the pages outlive the buffer (declared before the device, dropped after it),
        // and only the buffer's commands use them while it lives.
        unsafe { pages.rebind_bytes_no_copy(second.memory(), None, None) }.unwrap();
    }
    gate.set_signaled_value(1);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let counts = device.work_counts();
    let refused = counts.reencodes_refused - before.reencodes_refused;
    assert_eq!(refused, 0, "a rebind refused to encode an encoder again");
    let again = Moved::of(&before, &counts);
    let reencoded = counts.encoders_reencoded - before.encoders_reencoded;
    let read = match rebinds {
        true => second,
        false => first,
    };
    let words = &read.words()[..(LENGTH / 4) as usize];
    let blit = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    drop(open);
    let (after, _) = tools.bytes(&texture);
    Some(Logged {
        logged,
        again,
        reencoded,
        blit,
        after,
    })
}

/// The second test of the module docs: with a buffer over host pages alive, a logged pass turns
/// and copies nothing in; the rebind encodes the turning command buffer's blit and pass again, the
/// blit reading the texels from before the turn and the pass taking its turn again, reading them
/// too with no copy in. Its bytes are those the same lane reads where no pass turns, every read
/// copying in, with no rebind ([`logged_lane`]).
#[test]
fn a_logged_pass_turns_and_its_encoding_again_meets_the_faces_as_first_met() {
    // The pages, declared before the device that uses them, so dropped after it.
    let hosts: [Host; 4] = std::array::from_fn(|_| Host::new());
    let scratch = Scratch::new("attached-read-faces-logged");
    let device = device(&scratch.0);
    let tools = Tools::new(&device);
    let Some(turned) = logged_lane(&device, &tools, [&hosts[0], &hosts[1]], false) else {
        return;
    };
    let copied = logged_lane(&device, &tools, [&hosts[2], &hosts[3]], true).unwrap();
    for (lane, name) in [(&turned, "turned"), (&copied, "copied")] {
        eprintln!(
            "ATTACHED-READ-FACES: logged, {name}: the logged pass {:?}; encoded again ({} \
             encoders) {:?}",
            lane.logged, lane.reencoded, lane.again
        );
        let differ =
            |got: &[u8], want: &[u8]| (0..got.len()).filter(|&at| got[at] != want[at]).count();
        let blit = differ(&lane.blit, &image(0x30, 0x62));
        assert_eq!(blit, 0, "{name}: the blit: {blit} bytes differ");
        let after = differ(&lane.after, &image(0x90, 0x62));
        assert_eq!(after, 0, "{name}: the pass: {after} bytes differ");
    }
    let turn = Moved {
        swaps: 1,
        reads: 1,
        ..Moved::default()
    };
    assert_eq!(turned.logged, turn, "the logged pass turns");
    assert_eq!(
        (turned.reencoded, copied.reencoded),
        (2, 0),
        "the rebind encodes the blit and the pass again"
    );
    assert_eq!(
        (
            turned.again.swaps,
            turned.again.copied_in,
            turned.again.unswapped
        ),
        (1, 0, 0),
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
    assert_eq!(
        turned.blit, copied.blit,
        "the blit's bytes, turned and copied"
    );
    assert_eq!(
        turned.after, copied.after,
        "the texture's bytes, turned and copied"
    );
}

/// The third test of the module docs: an execution of an indirect command buffer, then a pass
/// that clears the texture and reads it, in one command buffer. The execution's command reads
/// the texel at (12, 6), inside [`RECT`], and at (0, 0), outside it, through an argument buffer
/// naming the texture's ID: 0x50 and 0x20, the texels from before the pass. The face the pass
/// would turn to holds 0x40 inside [`RECT`], what the first read copied in, and no texel the
/// texture held outside it (lavapipe read 0 there).
#[test]
fn an_execution_before_a_clearing_pass_reads_the_texels_from_before_it() {
    let scratch = Scratch::new("attached-read-faces-execution");
    let device = device(&scratch.0);
    let tools = Tools::new(&device);
    let texture = texture(&device);
    // The faces, made by the first read, which copies its 0x40 into the other face; then the
    // texture's own face holds 0x20 and 0x50.
    tools.pass(&texture, Some(0x10), Draw::Nothing);
    tools.pass(&texture, None, Draw::Solid(0x40));
    let moved = tools.pass(&texture, Some(0x20), Draw::Read);
    assert_eq!(moved.made, 1, "the first read makes the faces");
    tools.pass(&texture, None, Draw::Solid(0x50));
    // The execution: one command of `rt_heap_zero`, two reads.
    let function = library(&device, "engine/kernel_read_through.ll")
        .new_function_with_name("rt_heap_zero")
        .unwrap();
    let mut descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.support_indirect_command_buffers = true;
    let state =
        (device.new_compute_pipeline_state(&descriptor, &PipelineOptions::default())).unwrap();
    let holding =
        |bytes: &[u8]| (device.new_buffer_with_bytes(bytes, ResourceOptions::shared())).unwrap();
    let argument = holding(&texture.gpu_resource_id().0.to_le_bytes());
    let positions: Vec<u8> = ([12i32, 6, 0, 0, 0, 0, 0, 0].iter())
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let positions = holding(&positions);
    let out = holding(&[0; 32]);
    let mut icb_descriptor = IndirectCommandBufferDescriptor::new();
    icb_descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH_THREADS;
    icb_descriptor.max_kernel_buffer_bind_count = 3;
    let icb = device
        .new_indirect_command_buffer(&icb_descriptor, 1, ResourceOptions::shared())
        .unwrap();
    let command = icb.indirect_compute_command_at(0).unwrap();
    command.set_compute_pipeline_state(&state).unwrap();
    for (index, buffer) in (0..).zip([&positions, &out, &argument]) {
        command.set_kernel_buffer(Some(buffer), 0, index).unwrap();
    }
    let threads = Size::new(2, 1, 1);
    (command.concurrent_dispatch_threads(threads, threads)).unwrap();
    let before = device.work_counts();
    let command_buffer = tools.queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    let buffers = [
        positions.as_resource(),
        out.as_resource(),
        argument.as_resource(),
    ];
    let usage = ResourceUsage::READ | ResourceUsage::WRITE;
    encoder.use_resources(&buffers, usage).unwrap();
    (encoder.use_resource(texture.as_resource(), ResourceUsage::READ)).unwrap();
    encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
    encoder.end_encoding().unwrap();
    tools.encode(&command_buffer, &texture, Some(0x30), Draw::Read);
    Tools::run(&command_buffer);
    let moved = Moved::of(&before, &device.work_counts());
    let read: Vec<f32> = {
        let bytes = out.contents().unwrap().as_ptr();
        // SAFETY: the buffer's 32 bytes, which no command writes now.
        let bytes = unsafe { std::slice::from_raw_parts(bytes, 32) };
        (bytes.as_chunks::<4>().0.iter())
            .map(|word| f32::from_le_bytes(*word))
            .collect()
    };
    let codes: Vec<u8> = read.iter().map(|v| (v * 255.0).round() as u8).collect();
    eprintln!(
        "ATTACHED-READ-FACES: an execution before a clearing pass: {moved:?}, read {codes:x?}"
    );
    assert_eq!(
        codes,
        [[0x50; 4], [0x20; 4]].concat(),
        "the execution's reads: the texels from before the pass"
    );
    assert_eq!(
        (moved.swaps, moved.busy),
        (0, 1),
        "the pass turns nothing after an execution"
    );
    let (got, _) = tools.bytes(&texture);
    let want = image(0x30, 0x50);
    let differ = (0..got.len()).filter(|&at| got[at] != want[at]).count();
    assert_eq!(differ, 0, "the pass: {differ} bytes differ");
}

/// The points a lane of the commit tests holds, let go as this drops, a failing test's unwinding
/// too.
struct HeldCommit;

impl HeldCommit {
    fn new() -> HeldCommit {
        trace::hold_at(Point::Committing);
        HeldCommit
    }
}

impl Drop for HeldCommit {
    fn drop(&mut self) {
        trace::release(Point::Committing);
    }
}

/// A lane of the commit tests (the module docs): a command buffer `committing` that draws into
/// the texture, reads it beside an open one, `turning`, and draws over the read, held as it commits
/// (`Point::Committing`: its last command buffer ended, its batches not yet the scheduler's);
/// meanwhile `turning`'s pass, which clears the texture and reads it, opens. Where `folds`, the
/// committing command buffer's one batch ends in its last command buffer; else a turn of another
/// texture whose command buffer went uncommitted is pending as it commits, so its end is a command
/// buffer of its own. Every step's bytes are checked: the committing command buffer's pass
/// leaves 0x30 outside [`RECT`] and 0x70 inside it, which the turning pass reads.
fn beside_a_commit(lane: &str, folds: bool) {
    let scratch = Scratch::new(lane);
    let device = device(&scratch.0);
    let tools = Tools::new(&device);
    // The texture whose turn is pending, of the lane whose end does not fold.
    let other = (!folds).then(|| texture(&device));
    let texture = texture(&device);
    let check = |want: Vec<u8>, what: &str| {
        let (got, _) = tools.bytes(&texture);
        let differ = (0..got.len()).filter(|&at| got[at] != want[at]).count();
        assert_eq!(differ, 0, "{lane}: {what}: {differ} bytes differ");
    };
    // The faces, made by the first read; then 0x20 outside the rectangle and 0x50 inside it.
    tools.pass(&texture, Some(0x10), Draw::Nothing);
    tools.pass(&texture, None, Draw::Solid(0x40));
    let moved = tools.pass(&texture, Some(0x20), Draw::Read);
    assert_eq!(moved.made, 1, "{lane}: the first read makes the faces");
    check(image(0x20, 0x40), "the first read");
    tools.pass(&texture, None, Draw::Solid(0x50));
    check(image(0x20, 0x50), "the solid draw");
    // A turn of the other texture, whose command buffer goes uncommitted: pending until a batch
    // names that texture, which none does.
    if let Some(other) = &other {
        tools.pass(other, Some(0x10), Draw::Nothing);
        let moved = tools.pass(other, Some(0x20), Draw::Read);
        assert_eq!(moved.made, 1, "{lane}: the other texture's faces");
        let before = device.work_counts();
        let dropped = tools.queue.command_buffer().unwrap();
        tools.encode(&dropped, other, Some(0x30), Draw::Read);
        drop(dropped);
        let moved = Moved::of(&before, &device.work_counts());
        assert_eq!(
            moved.swaps, 1,
            "{lane}: the other texture's uncommitted turn"
        );
    }
    let before = device.work_counts();
    let turning = tools.queue.command_buffer().unwrap();
    let committing = tools.queue.command_buffer().unwrap();
    // A draw before the read, so the read's copy in is recorded in its encoder, not left to the
    // batch's submission (`shadow_standing.rs`, **The copy in**), which would hold the batch's end
    // apart and the heap sets' ticket the turn waits for.
    tools.encode(&committing, &texture, None, Draw::Solid(0x55));
    tools.encode(&committing, &texture, Some(0x30), Draw::Read);
    tools.encode(&committing, &texture, None, Draw::Solid(0x70));
    let encoded = device.work_counts();
    let held = HeldCommit::new();
    let commit = {
        let committing = committing.clone();
        std::thread::spawn(move || committing.commit())
    };
    assert!(
        trace::reached(Point::Committing, WAIT),
        "{lane}: the commit reached its hold"
    );
    tools.encode(&turning, &texture, Some(0x60), Draw::Read);
    drop(held);
    commit.join().unwrap().unwrap();
    committing.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        committing.error(),
        None,
        "{lane}: the committing command buffer"
    );
    let ended = device.work_counts();
    let (folded, apart) = (
        ended.finales_folded - encoded.finales_folded,
        ended.finales_apart - encoded.finales_apart,
    );
    eprintln!("ATTACHED-READ-FACES: {lane}: its end folded {folded}, apart {apart}");
    let want = match folds {
        true => (1, 0),
        false => (0, 1),
    };
    assert_eq!((folded, apart), want, "{lane}: the committing batch's end");
    check(image(0x30, 0x70), "the committing command buffer");
    Tools::run(&turning);
    check(image(0x60, 0x70), "the pass beside the commit");
    let moved = Moved::of(&before, &device.work_counts());
    eprintln!("ATTACHED-READ-FACES: {lane}: {moved:?}");
    assert_eq!(
        (moved.swaps, moved.busy, moved.fixes),
        (0, 2, 0),
        "{lane}: neither pass turns, beside an open command buffer or a commit"
    );
    drop(other);
}

/// The fourth test of the module docs: the committing command buffer's batch ends in its last
/// command buffer.
#[test]
fn a_pass_beside_a_commit_turns_nothing_and_the_commit_s_end_stays_folded() {
    beside_a_commit("attached-read-faces-commit-folded", true);
}

/// The fifth test of the module docs: the committing command buffer's batch ends in a command
/// buffer of its own, so its submission would take a turn's copies.
#[test]
fn a_pass_beside_a_commit_turns_nothing_and_the_commit_s_pixels_stay() {
    beside_a_commit("attached-read-faces-commit-apart", false);
}
