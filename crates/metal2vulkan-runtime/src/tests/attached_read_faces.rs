//! A texture a draw reads where its own render pass renders it (`texture/attached_reads.rs`), a
//! `Private` texture of the device's own memory: its first attached read gives it two faces
//! (`texture/faces.rs`), and each later pass that clears it whole turns it to the other face, its
//! read taking the face from before the pass with no copy in (`render/face_swap.rs`).
//!
//! Each pass clears the texture and draws, through [`RECT`], a read of the texel under each pixel
//! that writes what it read; the pass before it drew a solid value there. So the texels inside
//! [`RECT`] after the pass are those its read took: the texels from before the pass, Metal's
//! answer and the copy's (`texture/attached_reads.rs`), and every texel outside it the clear's.
//! A read of the wrong face takes a value two passes old. Each step checks those bytes and what
//! the pass moved: the texels copied in (`WorkCounts::shadow_texels_copied_in`, none for a
//! turned pass) and the turns, reads and copies between the faces it counted.
//!
//! The steps: the first attached read, which makes the faces and copies in; a turn; a blit read
//! after it, of the new face; a turn back; a pass while another command buffer is open, which
//! turns nothing and copies in; a turn whose command buffer commits after one made meanwhile that
//! reads the texture, whose submission copies the face from before the turn into the new one
//! first; and a turn whose command buffer goes uncommitted, after which the next read of the
//! texture takes that copy, and the texture turns again.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are `tob_rect`,
//! `tob_solid` and `tob_read` of the owned fixture `validation/fixtures/public/engine/tob_renders.ll`.

use super::blit_holds::{device, Scratch, WAIT};
use super::encoder_holds::{bytes_of, library};
use crate::{
    BlitOption, Buffer, ClearColor, CommandBuffer, CommandQueue, Device, LoadAction, Origin,
    PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions, ScissorRect, Size,
    StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};

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
    /// The read of the texel under each pixel, written as it was read.
    Read,
}

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
        match draw {
            Draw::Nothing => {}
            Draw::Read => {
                encoder.set_render_pipeline_state(&self.read).unwrap();
                encoder.set_fragment_texture(Some(texture), 0).unwrap();
            }
            Draw::Solid(value) => {
                encoder.set_render_pipeline_state(&self.solid).unwrap();
                let unorm = f32::from(value) / 255.0;
                let rgba = bytes_of(&[unorm; 4], f32::to_le_bytes);
                encoder.set_fragment_bytes(&rgba, 0).unwrap();
            }
        }
        if !matches!(draw, Draw::Nothing) {
            encoder.set_vertex_bytes(&corners, 0).unwrap();
            encoder.set_scissor_rect(RECT).unwrap();
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

/// The steps of the module docs: every read takes the texels from before its pass, the clears
/// and solid draws land where the next command reads them, and a turned pass copies nothing in.
#[test]
fn a_texture_s_faces_take_turns_for_its_attached_reads() {
    let scratch = Scratch::new("attached-read-faces");
    let device = device(&scratch.0);
    let tools = Tools::new(&device);
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, WIDTH, HEIGHT, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    let texture = device.new_texture(&descriptor).unwrap();
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
