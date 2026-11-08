//! A shadowed texture made with the IOSurface-plane constructor over a `Private` buffer the device
//! made (rows padded so that no image holds them on any device), as a consumer's device copy of a
//! surface is: only the runtime's commands write its truth, so its image's standing holds across
//! command buffers (`shadow_standing.rs`), and each copy back is a plain copy, no merge
//! (`ShadowCopies::merges_back`). A later command buffer's first copy in moves nothing where
//! nothing written since met the truth or the image; every write that did reaches what the
//! texture reads, and every write through the texture reaches the buffer.
//!
//! Each read goes through the shadow's image: a kernel (`invert_texels`, the owned fixture
//! `validation/fixtures/public/examples/runtime_examples.ll`) reads the texture and writes each
//! texel's color inverted, its alpha kept, into a plain texture the test reads back. (A blit's
//! read of a texture over a buffer copies the buffer's rows, never the image.) The writes are a
//! blit fill of the buffer from another queue's command buffer, as a host's transfer queue fills
//! a surface's device copy, and a pass drawing into the texture (`tob_rect`, `tob_solid` of
//! `engine/tob_renders.ll`), which a blit read of the buffer then reads.
//!
//! Each leg checks both halves: the bytes read are the buffer's as every write left them, and
//! the counts say which copies in moved nothing (`WorkCounts::shadow_copies_in_skipped_by_standing`).
//! A standing that never held leaves the bytes right and the counts wrong; one that held over a
//! write leaves the counts right and the bytes wrong. A shadowed render target over host pages a
//! no-copy buffer imports, whose CPU writes no count sees, never stands (a shadow under
//! `HostPageTexturePolicy::Shadow`, `blit_holds::shadow_device`: under the default policy the
//! device's in-place check may put it in place, with no image to stand).
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, over_host, shadow_device, Host, Scratch, WAIT};
use super::encoder_holds::{bytes_of, library};
use crate::{
    Buffer, CommandQueue, ComputePipelineState, Device, IoSurfacePlane, LoadAction,
    PipelineOptions, Region, RenderPassDescriptor, RenderPipelineState, ResourceOptions,
    ScissorRect, Size, StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

/// The plane: `WIDTH` texels a row, `HEIGHT` rows, rows of `PITCH` bytes, 64 past each row's
/// texels.
pub(super) const WIDTH: u64 = 64;
pub(super) const HEIGHT: u64 = 16;
const ROW: u64 = WIDTH * 4;
pub(super) const PITCH: u64 = ROW + 64;
pub(super) const LENGTH: u64 = PITCH * HEIGHT;

/// What the texture is read and written with: the kernel that reads it, the plain texture the
/// kernel writes, the pipeline that draws into it, and a `Shared` buffer the buffer's bytes are
/// read into.
pub(super) struct Tools {
    queue: CommandQueue,
    invert: ComputePipelineState,
    inverted: Texture,
    solid: RenderPipelineState,
    read: Buffer,
}

impl Tools {
    pub(super) fn new(device: &Device) -> Tools {
        let examples = library(device, "examples/runtime_examples.ll");
        let kernel = examples.new_function_with_name("invert_texels").unwrap();
        let options = PipelineOptions::default();
        let invert = (device.new_compute_pipeline_state_with_function(&kernel, &options)).unwrap();
        let mut plain =
            TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, WIDTH, HEIGHT, false);
        plain.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        let inverted = device.new_texture(&plain).unwrap();
        let renders = library(device, "engine/tob_renders.ll");
        let vertex = renders.new_function_with_name("tob_rect").unwrap();
        let fragment = renders.new_function_with_name("tob_solid").unwrap();
        let mut descriptor =
            RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
        descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
        let solid = (device.new_render_pipeline_state(&descriptor, &options)).unwrap();
        let read = (device.new_buffer(LENGTH, ResourceOptions::shared())).unwrap();
        Tools {
            queue: device.new_command_queue().unwrap(),
            invert,
            inverted,
            solid,
            read,
        }
    }

    /// `texture`'s texels as the kernel reads them in a command buffer of its own, inverted, row
    /// after row, and the counts it moved.
    pub(super) fn texels(&self, texture: &Texture) -> (Vec<u8>, Stood) {
        let device = self.queue.device();
        let before = device.work_counts();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&self.invert).unwrap();
        encoder.set_texture(Some(texture), 0).unwrap();
        encoder.set_texture(Some(&self.inverted), 1).unwrap();
        (encoder.dispatch_threads(Size::new(WIDTH, HEIGHT, 1), Size::new(8, 8, 1))).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        let counts = stood(&before, &device.work_counts());
        (self.inverted_texels(), counts)
    }

    /// The texels the kernel last wrote, row after row.
    fn inverted_texels(&self) -> Vec<u8> {
        let mut bytes = vec![0; (ROW * HEIGHT) as usize];
        let region = Region::new_2d(0, 0, WIDTH, HEIGHT);
        (self.inverted.get_bytes(&mut bytes, ROW, 0, region, 0, 0)).unwrap();
        bytes
    }

    /// A pass loading `texture` that draws `value` (every channel) over all of it, or inside
    /// `scissor`, in a command buffer of its own: the merge dispatches it planned, and the counts
    /// it moved.
    fn draw(&self, texture: &Texture, value: u8, scissor: Option<ScissorRect>) -> (u64, Stood) {
        let device = self.queue.device();
        let before = device.work_counts();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut pass = RenderPassDescriptor::new();
        let attachment = &mut pass.color_attachments[0];
        attachment.texture = Some(texture.clone());
        attachment.load_action = LoadAction::Load;
        attachment.store_action = StoreAction::Store;
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&self.solid).unwrap();
        if let Some(scissor) = scissor {
            encoder.set_scissor_rect(scissor).unwrap();
        }
        let corners = bytes_of(&[-1.0f32, -1.0, 1.0, 1.0], f32::to_le_bytes);
        encoder.set_vertex_bytes(&corners, 0).unwrap();
        let unorm = f32::from(value) / 255.0;
        let color = bytes_of(&[unorm; 4], f32::to_le_bytes);
        encoder.set_fragment_bytes(&color, 0).unwrap();
        (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        let after = device.work_counts();
        let merges = after.shadow_merge_dispatches_planned - before.shadow_merge_dispatches_planned;
        (merges, stood(&before, &after))
    }

    /// `buffer`'s first `len` bytes, as a blit copy into the read buffer moves them.
    fn bytes(&self, buffer: &Buffer, len: u64) -> Vec<u8> {
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        (blit.copy_from_buffer_to_buffer(buffer, 0, &self.read, 0, len)).unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        let bytes = self.read.contents().unwrap().as_ptr();
        // SAFETY: the read buffer's first `len` bytes, which no command writes now.
        unsafe { std::slice::from_raw_parts(bytes, len as usize) }.to_vec()
    }
}

/// The plane's texture over a `Private` buffer, its bytes filled with 0x11.
struct Plane {
    tools: Tools,
    buffer: Buffer,
    texture: Texture,
}

impl Plane {
    fn new(device: &Device) -> Plane {
        let tools = Tools::new(device);
        let buffer = (device.new_buffer(LENGTH, ResourceOptions::private())).unwrap();
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, WIDTH, HEIGHT, false);
        descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        let plane = IoSurfacePlane::new(0, PITCH, true);
        let texture = buffer.new_texture_with_iosurface_plane(&descriptor, plane);
        let texture = texture.unwrap();
        assert!(texture.shadow().is_some(), "padded rows make a shadow");
        let plane = Plane {
            tools,
            buffer,
            texture,
        };
        plane.fill(0..LENGTH, 0x11, &plane.tools.queue);
        plane
    }

    /// A fill of the buffer's `bytes` with `value` in a command buffer of its own on `queue`.
    fn fill(&self, bytes: std::ops::Range<u64>, value: u8, queue: &CommandQueue) {
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&self.buffer, bytes, value).unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
    }

    fn texels(&self) -> (Vec<u8>, Stood) {
        self.tools.texels(&self.texture)
    }
}

/// The standing's counts between two snapshots of a device's: the copies in decided against it,
/// those it skipped and their texels, and the texels of those that moved them by where each was
/// decided and by what the standing held (`WorkCounts::shadow_texels_unskipped_*`); of those whose
/// standing never stood, by what last kept a command buffer's end from leaving it standing
/// (`WorkCounts::shadow_texels_never_stood_*`).
pub(super) struct Stood {
    against: u64,
    pub(super) skipped: u64,
    texels: u64,
    at: [u64; 6],
    standing: [u64; 5],
    never: [u64; 5],
}

/// Where an unskipped copy in was decided, as `Stood::at` orders them.
const UNSTANDING: usize = 0;
const COMMAND_BUFFER: usize = 4;
const LET_GO: usize = 5;
/// What its standing held, as `Stood::standing` orders them.
const STOOD: usize = 0;
const NEVER_STOOD: usize = 1;
const TRUTH_WRITTEN: usize = 3;

/// Of `stood`, the texels its unskipped copies in moved at `at` alone, and those whose standing
/// held `standing` alone: every other count zero.
fn moved(stood: &Stood, at: usize, standing: Option<usize>) -> u64 {
    let texels = stood.at[at];
    let mut at_only = [0; 6];
    at_only[at] = texels;
    assert_eq!(stood.at, at_only, "where the copies in were decided");
    let mut standing_only = [0; 5];
    if let Some(standing) = standing {
        standing_only[standing] = texels;
    }
    assert_eq!(stood.standing, standing_only, "what their standing held");
    texels
}

fn stood(before: &WorkCounts, after: &WorkCounts) -> Stood {
    let delta = |count: fn(&WorkCounts) -> u64| count(after) - count(before);
    let at = [
        delta(|c| c.shadow_texels_unskipped_unstanding),
        delta(|c| c.shadow_texels_unskipped_between_instances),
        delta(|c| c.shadow_texels_unskipped_after_encoder_writes),
        delta(|c| c.shadow_texels_unskipped_recorded_whole),
        delta(|c| c.shadow_texels_unskipped_command_buffer),
        delta(|c| c.shadow_texels_unskipped_let_go),
    ];
    let standing = [
        delta(|c| c.shadow_texels_unskipped_stood),
        delta(|c| c.shadow_texels_unskipped_never_stood),
        delta(|c| c.shadow_texels_unskipped_unnamed_since),
        delta(|c| c.shadow_texels_unskipped_truth_written),
        delta(|c| c.shadow_texels_unskipped_image_written),
    ];
    let never = [
        delta(|c| c.shadow_texels_never_stood_unmet),
        delta(|c| c.shadow_texels_never_stood_not_current),
        delta(|c| c.shadow_texels_never_stood_dropped),
        delta(|c| c.shadow_texels_never_stood_partial),
        delta(|c| c.shadow_texels_never_stood_end),
    ];
    let skipped_texels = delta(|c| c.shadow_texels_skipped_by_standing);
    // Every first copy in of a private truth is skipped or counted where it was decided; and every
    // one of a shadow that may stand that moved its texels, by what its standing held.
    let first = delta(|c| c.shadow_texels_first_copied_in_private_buffers)
        + delta(|c| c.shadow_texels_first_copied_in_private_surfaces);
    assert_eq!(
        first,
        skipped_texels + at.iter().sum::<u64>(),
        "skipped and unskipped: {at:?}"
    );
    assert_eq!(
        at[1..].iter().sum::<u64>(),
        standing.iter().sum::<u64>(),
        "where {at:?}, standing {standing:?}"
    );
    assert_eq!(
        standing[NEVER_STOOD],
        never.iter().sum::<u64>(),
        "never stood {never:?}"
    );
    Stood {
        against: delta(|c| c.shadow_copies_in_against_standing),
        skipped: delta(|c| c.shadow_copies_in_skipped_by_standing),
        texels: skipped_texels,
        at,
        standing,
        never,
    }
}

/// Whether every texel the kernel wrote is a texel of the byte `value` inverted: its three color
/// channels `!value`, its alpha `value`.
pub(super) fn inverted(texels: &[u8], value: u8) -> bool {
    (texels.chunks(4)).all(|texel| texel == [!value, !value, !value, value])
}

/// Every byte `value`.
fn all(bytes: &[u8], value: u8) -> bool {
    bytes.iter().all(|&byte| byte == value)
}

/// A second command buffer's read of the texture moves nothing in; a fill of the buffer by
/// another queue's command buffer since lets go of the standing, and the read after it takes the
/// fill; a pass drawing into the texture reaches the buffer's bytes by a plain copy back (no
/// merge), leaving the bytes between rows, and the read after it holds the pass's texels and
/// moves nothing in.
#[test]
fn a_private_plane_s_image_stands_until_a_command_writes_the_buffer_or_the_texture() {
    let scratch = Scratch::new("shadow-standing");
    let plane = Plane::new(&device(&scratch.0));
    let (first, counts) = plane.texels();
    assert!(inverted(&first, 0x11), "the first read takes the bytes");
    assert_eq!(
        counts.against, 1,
        "the first copy in is decided at submission"
    );
    assert_eq!(counts.skipped, 0, "nothing stood yet");
    let texels = WIDTH * HEIGHT;
    assert_eq!(
        moved(&counts, LET_GO, Some(NEVER_STOOD)),
        texels,
        "never stood"
    );
    assert_eq!(
        counts.never,
        [texels, 0, 0, 0, 0],
        "no command buffer's end had met it"
    );
    let (second, counts) = plane.texels();
    assert!(inverted(&second, 0x11));
    assert_eq!(counts.skipped, 1, "the image stood");
    assert_eq!(counts.texels, WIDTH * HEIGHT, "every texel");
    // A host's transfer queue fills rows 1 and 2: the next read copies in again.
    let transfer = plane.tools.queue.device().new_command_queue().unwrap();
    plane.fill(PITCH..3 * PITCH, 0x5A, &transfer);
    let (filled, counts) = plane.texels();
    let row = ROW as usize;
    assert!(
        inverted(&filled[row..3 * row], 0x5A),
        "the fill reached the texture"
    );
    assert!(inverted(&filled[..row], 0x11) && inverted(&filled[3 * row..], 0x11));
    assert_eq!(counts.skipped, 0, "the fill let go");
    assert_eq!(
        moved(&counts, LET_GO, Some(TRUTH_WRITTEN)),
        texels,
        "the fill wrote the truth"
    );
    let (_, counts) = plane.texels();
    assert_eq!(counts.skipped, 1, "and it stood again");
    // A pass draws 0x99 into every texel: a blit read of the buffer finds it, and the bytes
    // between rows as the fills left them.
    let (merges, counts) = plane.tools.draw(&plane.texture, 0x99, None);
    assert_eq!(merges, 0, "a copy back, no merge");
    assert_eq!(counts.skipped, 1, "the pass's load took the standing image");
    let bytes = plane.tools.bytes(&plane.buffer, LENGTH);
    for y in 0..HEIGHT as usize {
        let at = y * PITCH as usize;
        assert!(all(&bytes[at..at + row], 0x99), "row {y}'s texels");
        let gap = if (1..3).contains(&y) { 0x5A } else { 0x11 };
        assert!(
            all(&bytes[at + row..at + PITCH as usize], gap),
            "row {y}'s gap"
        );
    }
    let (written, counts) = plane.texels();
    assert!(inverted(&written, 0x99), "the read holds the pass's texels");
    assert_eq!(counts.skipped, 1, "the pass's end left the image standing");
    // A pass that draws 0x33 inside rows 0 to 4 alone: its first copy in still takes (here,
    // skips) every texel, so its end leaves the image standing whole for the next read.
    let rows = ScissorRect {
        x: 0,
        y: 0,
        width: WIDTH,
        height: 4,
    };
    let (_, counts) = plane.tools.draw(&plane.texture, 0x33, Some(rows));
    assert_eq!(
        counts.skipped, 1,
        "the scissored pass's load took the standing image"
    );
    let (scissored, counts) = plane.texels();
    assert!(inverted(&scissored[..4 * row], 0x33), "the scissored rows");
    assert!(inverted(&scissored[4 * row..], 0x99), "and the rest");
    assert_eq!(
        counts.skipped, 1,
        "the scissored pass left the image standing"
    );
}

/// A command buffer that fills the buffer and then reads the texture in a later encoder copies
/// in, the fill included; so does every command buffer once the buffer's `VkBuffer` was handed
/// out, which the caller's own work may write.
#[test]
fn a_private_plane_copies_in_after_its_own_command_buffer_s_write_and_once_handed_out() {
    let scratch = Scratch::new("shadow-standing-own");
    let plane = Plane::new(&device(&scratch.0));
    let _ = plane.texels();
    let (_, counts) = plane.texels();
    assert_eq!(counts.skipped, 1, "the image stood");
    let tools = &plane.tools;
    let device = tools.queue.device();
    let before = device.work_counts();
    let command_buffer = tools.queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(&plane.buffer, 0..PITCH, 0x77).unwrap();
    blit.end_encoding().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&tools.invert).unwrap();
    encoder.set_texture(Some(&plane.texture), 0).unwrap();
    encoder.set_texture(Some(&tools.inverted), 1).unwrap();
    (encoder.dispatch_threads(Size::new(WIDTH, HEIGHT, 1), Size::new(8, 8, 1))).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let counts = stood(&before, &device.work_counts());
    let read = tools.inverted_texels();
    let row = ROW as usize;
    assert!(
        inverted(&read[..row], 0x77),
        "the read took its own command buffer's fill"
    );
    assert!(inverted(&read[row..], 0x11));
    assert_eq!(counts.skipped, 0);
    let texels = WIDTH * HEIGHT;
    let written = counts.standing[STOOD] + counts.standing[TRUTH_WRITTEN];
    assert_eq!(
        (counts.at[COMMAND_BUFFER], written),
        (texels, texels),
        "decided by its own command buffer's write"
    );
    // SAFETY: the handle is only read; no work of the caller's uses it.
    let _ = unsafe { plane.buffer.vk_buffer() };
    let _ = plane.texels();
    let (_, counts) = plane.texels();
    assert_eq!(counts.against, 0, "handed out, it never stands");
    assert_eq!(moved(&counts, UNSTANDING, None), texels, "counted so");
}

/// A shadowed render target over host pages a no-copy buffer imports (under the shadow policy)
/// never stands: no count sees the CPU's writes. A CPU write between two command buffers reaches
/// the second one's read, and no copy in of either is decided against a standing.
#[test]
fn a_render_target_over_host_pages_never_stands() {
    let host = Host::new();
    let scratch = Scratch::new("shadow-standing-host");
    let device = shadow_device(&scratch.0);
    let runs = Arc::new(AtomicUsize::new(0));
    let Some(buffer) = over_host(&device, &host, &runs) else {
        return;
    };
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, WIDTH, HEIGHT, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    let texture = buffer.new_texture(&descriptor, 0, ROW).unwrap();
    assert!(
        texture.shadow().is_some(),
        "a render target over host pages"
    );
    let tools = Tools::new(&device);
    host.fill(0x1111_1111);
    let (first, counts) = tools.texels(&texture);
    assert!(inverted(&first, 0x11), "the first read takes the pages");
    assert_eq!((counts.against, counts.skipped), (0, 0));
    host.fill(0x5A5A_5A5A);
    let (second, counts) = tools.texels(&texture);
    assert!(
        inverted(&second, 0x5A),
        "the second read takes the CPU's write"
    );
    assert_eq!((counts.against, counts.skipped), (0, 0), "nothing stands");
}
