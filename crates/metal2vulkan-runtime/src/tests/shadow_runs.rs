//! Runs of passes over a shadowed texture merged back, merged once (`pass_runs.rs`): a 2D
//! `BGRA8Unorm` texture of 1024 x 1024 texels over a no-copy buffer's host pages whose rows lie
//! 15360 bytes apart (a 3840-texel surface's), so each row holds 11264 bytes past its texels that
//! no pass writes: a shadow on every device, its writes merged back into the bytes
//! (`shadow.rs`'s **Merged back**).
//!
//! One command buffer mixes passes over the texture (each loading and storing it, blending a
//! rectangle over it) with the encoders and commands that end a run or pass it: a 1x1
//! attachmentless render encoder, a pass over another texture, a blit reading the texture's
//! texels, a blit reading the buffer's bytes, an event signal, a pass naming the texture's
//! buffer, and a pass over a second texture over the same bytes. The sequence runs three times:
//! over the texture, where its passes form the runs the steps make, each of more than one pass
//! kept and merged after its first pass, in place, and once more for the rest;
//! over the texture again with a blit between every two encoders, which ends every run, so every
//! pass keeps and merges for itself; and over a private texture of the same first texels. The
//! texels each blit read mid-sequence, and those the sequence left, are the same in all three,
//! and the bytes between the rows keep theirs.
//!
//! A second sequence draws over the whole texture under a scissor a pass at a time: the first
//! pass keeps its scissor's texels and merges in place, the second opens the run, keeping its
//! own scissor's, the third, outside them, merges that keep and keeps every texel, and the run
//! goes on over that keep, its merges storing whole blocks of the rows as a pass's alone does
//! (`shadow.rs`'s **Stored attachments**).
//!
//! A third sequence encodes passes over the texture, some binding a second no-copy buffer
//! (which their function does not read), behind a wait for a shared event only the host
//! signals; the second buffer is rebound before the signal, which encodes those passes again
//! before the command buffer's first submission (`command_buffer/reencode.rs`): each leaves its
//! run, and the texels are still the private texture's.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are the owned fixture
//! `validation/fixtures/public/engine/tob_renders.ll` (`llvm-as`).

use super::blit_holds::{device, Scratch, WAIT};
use super::encoder_holds::{bytes_of, library};
use super::linear_runs::attachmentless;
use super::merge_scratch::{blit, drawn, read, staged, TEXEL};
use crate::{
    BlitOption, Buffer, CommandBuffer, CommandQueue, Device, Error, EventRef, HostMemory,
    LoadAction, Origin, PipelineOptions, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, ScissorRect, Size, StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::metal::{BlendFactor, PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::alloc::Layout;
use std::ptr::NonNull;

/// The texture's width and height, and its rows' pitch in the buffer.
const W: u64 = 1024;
const H: u64 = 1024;
const ROW: u64 = 15360;

/// The byte the bytes between the rows hold, which no pass writes.
const BETWEEN: u8 = 0x5a;

/// One step of the sequence.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Step {
    /// A pass over the texture blending rectangle `i` (`drawn`).
    Draw(usize),
    /// A 1x1 render encoder of no attachment, one draw.
    Lone,
    /// A pass over another texture, binding the texture's buffer as a fragment buffer its
    /// function does not read where `true`, else a buffer of its own.
    Other(bool),
    /// A blit copying the texture's texels into a buffer, read after the command buffer.
    Read,
    /// A blit copying the buffer's bytes into a buffer, read after the command buffer (the
    /// private texture's texels, for it).
    Bytes,
    /// An event signal.
    Signal,
    /// A pass over a second texture over the same bytes blending rectangle `i` (the private
    /// texture itself, for it).
    Alias(usize),
}

use Step::{Alias, Bytes, Draw, Lone, Other, Read, Signal};

/// The passes of the sequence's last run, which goes to the command buffer's end.
const LAST_RUN: usize = 40;

/// The sequence, and the runs it makes, each of the passes noted:
/// 1. `Draw(0)` to `Draw(3)`: the 1x1 encoder and the pass over another texture pass it; the
///    blit reading the texels ends it (4 passes);
/// 2. `Draw(4)`, `Draw(5)`: the blit reading the bytes ends it (2);
/// 3. `Draw(6)`, `Draw(7)`: the signal ends it (2);
/// 4. `Draw(8)`, the first pass after the signal, which ends as the signal is pending: no pass
///    follows it (`pass_runs::Open::follows`) (1);
/// 5. `Draw(9)`: the pass naming the texture's buffer ends it (1);
/// 6. `Draw(10)`: the pass over the second texture over the same bytes ends it (1);
/// 7. `Alias(11)`, a run of its own over the second texture, which the next pass over the
///    texture ends (1);
/// 8. `Draw(12)`, `Draw(13)`: the blit reading the bytes ends it (2);
/// 9. `Draw(14)` on, [`LAST_RUN`] passes, past a 1x1 encoder, to the command buffer's end.
fn steps() -> Vec<Step> {
    let mut steps = vec![
        Draw(0),
        Draw(1),
        Lone,
        Draw(2),
        Other(false),
        Draw(3),
        Read,
        Draw(4),
        Draw(5),
        Bytes,
        Draw(6),
        Draw(7),
        Signal,
        Draw(8),
        Draw(9),
        Other(true),
        Draw(10),
        Alias(11),
        Draw(12),
        Draw(13),
        Bytes,
    ];
    for i in 0..LAST_RUN {
        if i == LAST_RUN / 2 {
            steps.push(Lone);
        }
        steps.push(Draw(14 + i));
    }
    steps
}

/// The passes over the texture or the second one among [`steps`], and the spans of passes that
/// follow each other they make, each begun by one of [`FIRSTS`]: its first pass merges in place
/// and its second opens the run, each keeping ([`keeping`]).
const PASSES: u64 = 14 + LAST_RUN as u64;
const RUNS: u64 = 9;
const FIRSTS: [Step; RUNS as usize] = [
    Draw(0),
    Draw(4),
    Draw(6),
    Draw(8),
    Draw(9),
    Draw(10),
    Alias(11),
    Draw(12),
    Draw(14),
];

/// The passes of [`steps`] over the texture or the second one, in the spans [`FIRSTS`] begin.
fn spans() -> Vec<Vec<Step>> {
    let mut spans: Vec<Vec<Step>> = Vec::new();
    for step in steps() {
        if !matches!(step, Draw(_) | Alias(_)) {
            continue;
        }
        if FIRSTS.contains(&step) {
            spans.push(Vec::new());
        }
        spans
            .last_mut()
            .expect("the first pass begins a span")
            .push(step);
    }
    spans
}

/// The passes that keep: each span's first, which merges in place, and its second, which opens
/// the run (`pass_runs.rs`'s **The merge goes last**).
fn keeping() -> Vec<Step> {
    (spans().iter())
        .flat_map(|span| span.iter().take(2).copied())
        .collect()
}

/// How many of the spans hold more than one pass, and so a run besides the pass merged in place.
fn longer_runs() -> u64 {
    spans().iter().filter(|span| span.len() > 1).count() as u64
}

/// The state that blends a half-transparent rectangle (`tob_rect`, corners from vertex buffer 0)
/// of the color in fragment buffer 0 (`tob_solid`) over a `BGRA8Unorm` color, of no depth.
fn blending(device: &Device) -> RenderPipelineState {
    let library = library(device, "engine/tob_renders.ll");
    let function = |name: &str| {
        let function = library.new_function_with_name(name).unwrap();
        StageSource::Function(function.function_descriptor())
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("tob_rect"));
    descriptor.fragment = Some(function("tob_solid"));
    let mut color = ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm);
    color.blending_enabled = true;
    color.source_rgb_blend_factor = BlendFactor::SourceAlpha;
    color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
    color.source_alpha_blend_factor = BlendFactor::One;
    color.destination_alpha_blend_factor = BlendFactor::Zero;
    descriptor.color_attachments[0] = Some(color);
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// What the texture's bytes hold first, rows `row` bytes apart: each texel's bytes a counting
/// pattern of its place among the texels, which no pass draws, and [`BETWEEN`] between the rows.
fn first_bytes(row: u64) -> Vec<u8> {
    (0..row * H)
        .map(|i| match i % row < W * TEXEL {
            true => ((i / row * W * TEXEL + i % row) * 7 % 251) as u8,
            false => BETWEEN,
        })
        .collect()
}

/// The texels of `bytes`, rows [`ROW`] bytes apart, packed.
fn texels(bytes: &[u8]) -> Vec<u8> {
    (bytes.chunks(ROW as usize))
        .flat_map(|r| &r[..(W * TEXEL) as usize])
        .copied()
        .collect()
}

/// The bytes between the rows of `bytes` that no longer hold [`BETWEEN`].
fn between_changed(bytes: &[u8]) -> usize {
    (0..bytes.len())
        .filter(|&i| (i as u64 % ROW) >= W * TEXEL && bytes[i] != BETWEEN)
        .count()
}

/// Page-aligned host memory of `length` bytes, freed when dropped: declared before the device's
/// buffer over it, so dropped after it.
struct Host(NonNull<u8>, Layout);

impl Host {
    fn new(length: u64) -> Host {
        let layout =
            Layout::from_size_align(length.next_multiple_of(64 << 10) as usize, 64 << 10).unwrap();
        // SAFETY: the layout's size is not zero.
        let pointer = unsafe { std::alloc::alloc_zeroed(layout) };
        Host(NonNull::new(pointer).unwrap(), layout)
    }

    /// Its pages, for a no-copy buffer or a rebind.
    ///
    /// # Safety
    ///
    /// They outlive every buffer over them, which alone uses them while it lives.
    unsafe fn memory(&self) -> HostMemory {
        // SAFETY: the caller's contract.
        unsafe { HostMemory::new(self.0) }
    }

    fn length(&self) -> u64 {
        self.1.size() as u64
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        // SAFETY: allocated with this layout in `new`.
        unsafe { std::alloc::dealloc(self.0.as_ptr(), self.1) };
    }
}

/// A no-copy buffer over `host`; `None`, printed, where the device imports no host memory.
fn over_pages(device: &Device, host: &Host) -> Option<Buffer> {
    // SAFETY: the memory outlives the buffer, and only its commands use it meanwhile.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(
            host.memory(),
            host.length(),
            ResourceOptions::default(),
            None,
        )
    };
    match made {
        Ok(buffer) => Some(buffer),
        Err(Error::Unsupported(gap)) => {
            eprintln!("SHADOW-RUNS: the device imports no host memory: {gap:?}");
            None
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    }
}

/// A `BGRA8Unorm` texture of [`W`] x [`H`] over `buffer`'s first bytes, rows [`ROW`] apart: a
/// shadow merged back.
fn texture_over(buffer: &Buffer) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, W, H, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = buffer.resource_options();
    let texture = buffer.new_texture(&descriptor, 0, ROW).unwrap();
    assert!(
        texture.shadow().is_some_and(|copies| copies.merges_back()),
        "the texture is a shadow merged back"
    );
    texture
}

/// A `Private` `BGRA8Unorm` texture of [`W`] x [`H`] holding [`first_bytes`]'s texels.
fn private_texture(queue: &CommandQueue) -> Texture {
    let device = queue.device();
    let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, W, H, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::private();
    let texture = device.new_texture(&d).unwrap();
    let packed = W * TEXEL;
    let source = staged(device, &first_bytes(packed));
    blit(queue, |b| {
        (b.copy_from_buffer_to_texture(
            &source,
            0,
            packed,
            packed * H,
            Size::new(W, H, 1),
            &texture,
            0,
            0,
            Origin::new(0, 0, 0),
            BlitOption::None,
        ))
        .unwrap()
    });
    texture
}

/// Where a pass's rectangle comes from: [`drawn`]'s for its index, or a buffer holding its
/// corners, bound as vertex buffer 0.
#[derive(Clone, Copy)]
enum Corners<'a> {
    Drawn,
    From(&'a Buffer),
}

/// Encode a pass over `texture`, loading and storing it, that blends rectangle `i` of
/// `corners` in `i`'s color with `state`; binding `named` as fragment buffer 1, which the
/// function does not read.
fn pass(
    command_buffer: &CommandBuffer,
    texture: &Texture,
    state: &RenderPipelineState,
    (i, corners): (usize, Corners<'_>),
    named: Option<&Buffer>,
) {
    let mut descriptor = RenderPassDescriptor::new();
    let color = &mut descriptor.color_attachments[0];
    color.texture = Some(texture.clone());
    color.load_action = LoadAction::Load;
    color.store_action = StoreAction::Store;
    let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
    encoder.set_render_pipeline_state(state).unwrap();
    let (rect, rgba) = drawn(i);
    match corners {
        Corners::Drawn => {
            (encoder.set_vertex_bytes(&bytes_of(&rect, f32::to_le_bytes), 0)).unwrap();
        }
        Corners::From(buffer) => encoder.set_vertex_buffer(Some(buffer), 0, 0).unwrap(),
    }
    (encoder.set_fragment_bytes(&bytes_of(&rgba, f32::to_le_bytes), 0)).unwrap();
    if let Some(named) = named {
        encoder.set_fragment_buffer(Some(named), 0, 1).unwrap();
    }
    (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
    encoder.end_encoding().unwrap();
}

/// Encode a blit copying `texture`'s texels into `out`, packed.
fn read_texels(command_buffer: &CommandBuffer, texture: &Texture, out: &Buffer) {
    let packed = W * TEXEL;
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    (encoder.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(W, H, 1),
        out,
        0,
        packed,
        packed * H,
        BlitOption::None,
    ))
    .unwrap();
    encoder.end_encoding().unwrap();
}

/// What the sequence runs over: the texture, its buffer (`None` for the private texture), the
/// second texture over its bytes, the other texture, and the states.
struct Scene<'a> {
    texture: &'a Texture,
    buffer: Option<&'a Buffer>,
    alias: &'a Texture,
    other: &'a Texture,
    blending: &'a RenderPipelineState,
    attachmentless: &'a RenderPipelineState,
}

/// What a read mid-sequence copied: the texels, packed, and the bytes between the rows that
/// changed (none for the private texture).
type Got = (Vec<u8>, usize);

/// [`steps`] over `over` in one command buffer, committed and completed; a blit between every
/// two encoders where `apart`, copying 4 bytes between buffers of its own. What each [`Read`]
/// and [`Bytes`] copied, and the steps that kept a shadowed texture's texels as they were
/// encoded (`WorkCounts::shadow_texels_kept`).
fn sequence(queue: &CommandQueue, over: &Scene<'_>, apart: bool) -> (Vec<Got>, Vec<Step>) {
    let device = queue.device();
    let packed = W * TEXEL;
    let unread = device.new_buffer(256, ResourceOptions::private()).unwrap();
    let (from, into) = (staged(device, &[1; 4]), staged(device, &[0; 4]));
    let event = device.new_event().unwrap();
    let steps = steps();
    let length = |step: &Step| match (step, over.buffer) {
        (Read, _) | (Bytes, None) => Some(packed * H),
        (Bytes, Some(_)) => Some(ROW * H),
        _ => None,
    };
    let reads = (steps.iter())
        .filter_map(|step| Some((*step, length(step)?)))
        .map(|(step, length)| {
            let out = device.new_buffer(length, ResourceOptions::shared());
            (step, out.unwrap(), length)
        })
        .collect::<Vec<_>>();
    let mut read_into = reads.iter();
    let mut keeps = Vec::new();
    let mut kept = device.work_counts().shadow_texels_kept;
    let command_buffer = queue.command_buffer().unwrap();
    for (at, step) in steps.iter().enumerate() {
        if apart && at > 0 {
            let mut encoder = command_buffer.blit_command_encoder().unwrap();
            (encoder.copy_from_buffer_to_buffer(&from, 0, &into, 0, 4)).unwrap();
            encoder.end_encoding().unwrap();
        }
        match *step {
            Draw(i) => {
                let drawn = (i, Corners::Drawn);
                pass(&command_buffer, over.texture, over.blending, drawn, None);
            }
            Alias(i) => {
                let drawn = (i, Corners::Drawn);
                pass(&command_buffer, over.alias, over.blending, drawn, None);
            }
            Lone => {
                let mut lone = RenderPassDescriptor::new();
                lone.render_target_width = 1;
                lone.render_target_height = 1;
                lone.default_raster_sample_count = 1;
                let mut encoder = command_buffer.render_command_encoder(&lone).unwrap();
                (encoder.set_render_pipeline_state(over.attachmentless)).unwrap();
                let corners = [-1.0f32, -1.0, 1.0, 1.0];
                (encoder.set_vertex_bytes(&bytes_of(&corners, f32::to_le_bytes), 0)).unwrap();
                (encoder.set_fragment_bytes(&bytes_of(&[0.0f32; 4], f32::to_le_bytes), 0)).unwrap();
                (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
                encoder.end_encoding().unwrap();
            }
            Other(names) => {
                let named = match (names, over.buffer) {
                    (true, Some(buffer)) => buffer,
                    _ => &unread,
                };
                let drawn = (at, Corners::Drawn);
                pass(
                    &command_buffer,
                    over.other,
                    over.blending,
                    drawn,
                    Some(named),
                );
            }
            Read => read_texels(&command_buffer, over.texture, &read_into.next().unwrap().1),
            Bytes => {
                let (_, out, length) = read_into.next().unwrap();
                match over.buffer {
                    Some(buffer) => {
                        let mut encoder = command_buffer.blit_command_encoder().unwrap();
                        (encoder.copy_from_buffer_to_buffer(buffer, 0, out, 0, *length)).unwrap();
                        encoder.end_encoding().unwrap();
                    }
                    None => read_texels(&command_buffer, over.texture, out),
                }
            }
            Signal => {
                (command_buffer.encode_signal_event(EventRef::from(&event), at as u64)).unwrap()
            }
        }
        // A pass's keep is planned as its encoder ends, which a reading of the counts lowers.
        let now = device.work_counts().shadow_texels_kept;
        if now > kept {
            keeps.push(*step);
        }
        kept = now;
    }
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let reads = (reads.iter())
        .map(|(step, out, length)| {
            let got = read(out, *length);
            match (step, over.buffer) {
                (Bytes, Some(_)) => (texels(&got), between_changed(&got)),
                _ => (got, 0),
            }
        })
        .collect();
    (reads, keeps)
}

/// The texels of the private texture, packed.
fn texels_of(queue: &CommandQueue, texture: &Texture) -> Vec<u8> {
    let out = (queue.device())
        .new_buffer(W * TEXEL * H, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    read_texels(&command_buffer, texture, &out);
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    read(&out, W * TEXEL * H)
}

/// The bytes of `host`, which no command uses now.
fn host_bytes(host: &Host, length: u64) -> Vec<u8> {
    // SAFETY: inside the allocation, which no command writes now (its command buffers completed).
    unsafe { std::slice::from_raw_parts(host.0.as_ptr(), length as usize) }.to_vec()
}

/// Write [`first_bytes`] into `host`, which no command uses now.
fn fill(host: &Host) {
    let first = first_bytes(ROW);
    // SAFETY: inside the allocation, which no command uses now.
    unsafe { std::ptr::copy_nonoverlapping(first.as_ptr(), host.0.as_ptr(), first.len()) };
}

/// The counts the runs are told by, between two readings.
struct Counted {
    copies: u64,
    kept: u64,
    planned: u64,
    copied_in: u64,
}

impl Counted {
    fn between(before: &WorkCounts, after: &WorkCounts) -> Counted {
        Counted {
            copies: after.shadow_copies - before.shadow_copies,
            kept: after.shadow_texels_kept - before.shadow_texels_kept,
            planned: after.shadow_merge_dispatches_planned - before.shadow_merge_dispatches_planned,
            copied_in: after.shadow_texels_copied_in - before.shadow_texels_copied_in,
        }
    }
}

/// Assert that `got` reads `expected`'s texels, and its bytes between the rows kept theirs.
fn same(what: &str, got: &Got, expected: &[u8]) {
    let differ = (0..got.0.len())
        .filter(|&i| got.0[i] != expected[i])
        .count();
    assert_eq!(
        differ, 0,
        "{what}: the texels differ from the private texture's"
    );
    assert_eq!(got.1, 0, "{what}: bytes between the rows changed");
}

/// [`steps`] over the texture as runs and with every pass apart, against the private texture's:
/// the same texels read mid-sequence and left at its end, the bytes between the rows kept, and
/// one keep and one merge a span of one pass, two for a longer one (every pass's where apart),
/// each span's first two passes the ones that keep ([`keeping`]).
#[test]
fn runs_over_host_pages_keep_and_merge_once() {
    let host = Host::new(ROW * H);
    let cache = Scratch::new("shadow-runs");
    let device = device(&cache.0);
    let queue = device.new_command_queue().unwrap();
    let blending = blending(&device);
    let attachmentless = attachmentless(&device);
    let other = private_texture(&queue);
    let reference = private_texture(&queue);
    let on_reference = Scene {
        texture: &reference,
        buffer: None,
        alias: &reference,
        other: &other,
        blending: &blending,
        attachmentless: &attachmentless,
    };
    let (expected, _) = sequence(&queue, &on_reference, false);
    let left = texels_of(&queue, &reference);
    let Some(buffer) = over_pages(&device, &host) else {
        return;
    };
    let mut counted = Vec::new();
    for apart in [false, true] {
        fill(&host);
        let texture = texture_over(&buffer);
        let alias = texture_over(&buffer);
        let on_texture = Scene {
            texture: &texture,
            buffer: Some(&buffer),
            alias: &alias,
            other: &other,
            blending: &blending,
            attachmentless: &attachmentless,
        };
        let before = device.work_counts();
        let (got, keeps) = sequence(&queue, &on_texture, apart);
        let after = device.work_counts();
        let count = Counted::between(&before, &after);
        println!(
            "SHADOW-RUNS {W}x{H}, rows of {ROW}, apart {apart}: shadow copies {}, texels kept {}, \
             merge dispatches planned {}, texels copied in {}; merges reused {}; kept at \
             {keeps:?}",
            count.copies,
            count.kept,
            count.planned,
            count.copied_in,
            after.shadow_merge_scratches_reused - before.shadow_merge_scratches_reused,
        );
        let passes = (steps().into_iter()).filter(|step| matches!(step, Draw(_) | Alias(_)));
        let firsts: Vec<Step> = match apart {
            true => passes.collect(),
            false => keeping(),
        };
        assert_eq!(keeps, firsts, "apart {apart}: the passes that keep");
        for (at, (got, expected)) in got.iter().zip(&expected).enumerate() {
            same(&format!("apart {apart}, read {at}"), got, &expected.0);
        }
        drop((texture, alias));
        let bytes = host_bytes(&host, ROW * H);
        same(
            &format!("apart {apart}, the bytes left"),
            &(texels(&bytes), between_changed(&bytes)),
            &left,
        );
        counted.push(count);
    }
    let (runs, apart) = (&counted[0], &counted[1]);
    // Each keep and merge is over every texel, alike for every pass apart.
    assert_eq!(apart.kept % PASSES, 0, "every pass apart keeps alike");
    assert_eq!(apart.planned % PASSES, 0, "every pass apart merges alike");
    let (kept, planned) = (apart.kept / PASSES, apart.planned / PASSES);
    assert_eq!(kept, W * H, "a pass apart keeps every texel");
    let keeps = RUNS + longer_runs();
    assert_eq!(
        runs.kept,
        keeps * kept,
        "a keep a span, two for a longer one"
    );
    assert_eq!(
        runs.planned,
        keeps * planned,
        "a merge planned for each keep"
    );
    assert_eq!(
        runs.copied_in, apart.copied_in,
        "the runs copy in what the passes apart do"
    );
    assert_eq!(
        apart.copies - runs.copies,
        2 * (PASSES - RUNS) - 2 * longer_runs(),
        "each pass of a run past its first neither keeps nor merges"
    );
}

/// The passes of [`a_pass_alone_merges_in_place`]'s run.
const ALONE_RUN: usize = 3;

/// A pass over the texture alone in its command buffer, then [`ALONE_RUN`] passes over it in
/// another. The pass alone merges in place as its encoder ends, so its encoder's commands go on
/// in no other `VkCommandBuffer` (`WorkCounts::command_buffers_begun_continuations`), and it
/// copies in, keeps and merges once each. Of the run, the first pass does the same; the second
/// keeps again and opens the run, and it and each later pass leave their merges to their
/// batch's submission, their encoders' commands going on in another `VkCommandBuffer`, the
/// last's recorded. The texels left are a private texture's, and the bytes between the rows keep
/// theirs: the run's blends change some bytes back to the value the first pass's keep holds, so
/// a run that went on over that keep past the merge in place would leave the first pass's value.
#[test]
fn a_pass_alone_merges_in_place() {
    let host = Host::new(ROW * H);
    let cache = Scratch::new("shadow-runs-alone");
    let device = device(&cache.0);
    let queue = device.new_command_queue().unwrap();
    let blending = blending(&device);
    let reference = private_texture(&queue);
    let Some(buffer) = over_pages(&device, &host) else {
        return;
    };
    fill(&host);
    let texture = texture_over(&buffer);
    let mut counted = Vec::new();
    for (first, passes) in [(0, 1), (1, ALONE_RUN)] {
        for over in [&reference, &texture] {
            let before = device.work_counts();
            let command_buffer = queue.command_buffer().unwrap();
            for i in first..first + passes {
                pass(&command_buffer, over, &blending, (i, Corners::Drawn), None);
            }
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None);
            let after = device.work_counts();
            let begun = |c: &WorkCounts| c.command_buffers_begun_continuations;
            let copies = after.shadow_copies - before.shadow_copies;
            counted.push((passes, begun(&after) - begun(&before), copies));
        }
    }
    println!("SHADOW-RUNS alone: (passes, continuations, shadow copies) {counted:?}");
    let (alone, run) = (counted[1], counted[3]);
    assert_eq!(
        alone,
        (1, 0, 3),
        "a pass alone copies in, keeps and merges in place"
    );
    assert_eq!(
        run,
        (ALONE_RUN, ALONE_RUN as u64 - 1, 5),
        "a copy in, two keeps and two merges, the run's passes leaving their merges"
    );
    drop(texture);
    let bytes = host_bytes(&host, ROW * H);
    same(
        "the bytes left",
        &(texels(&bytes), between_changed(&bytes)),
        &texels_of(&queue, &reference),
    );
}

/// The passes of [`rebound_passes_leave_their_runs`], and those of them whose rectangle is read
/// from the buffer the rebind moves.
const REBOUND_PASSES: usize = 12;
const NAMING: [usize; 4] = [0, 3, 4, 8];

/// The rectangle the moved buffer's pages hold before the rebind, and the one its new pages hold.
const BEFORE: [f32; 4] = [-1.0, -1.0, 1.0, 1.0];
const MOVED: [f32; 4] = [-0.5, -0.625, 0.25, 0.75];

/// Write `corners` at the start of `host`, which no command uses now.
fn write_corners(host: &Host, corners: [f32; 4]) {
    let bytes = bytes_of(&corners, f32::to_le_bytes);
    // SAFETY: inside the allocation, which no command uses now.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), host.0.as_ptr(), bytes.len()) };
}

/// Pass `i`'s rectangle in [`rebound_passes_leave_their_runs`]: read from `from` for those of
/// [`NAMING`], [`drawn`]'s for the others.
fn corners(i: usize, from: &Buffer) -> (usize, Corners<'_>) {
    match NAMING.contains(&i) {
        true => (i, Corners::From(from)),
        false => (i, Corners::Drawn),
    }
}

/// Passes over the texture behind a wait for a shared event only the host signals, four of them
/// drawing the rectangle a second no-copy buffer holds (the first among them, which merged in
/// place), which is rebound over pages holding another before the signal: the rebind encodes
/// those four again before the first submission (`command_buffer/reencode.rs`), so they draw the
/// new pages' rectangle, each copying in, keeping and merging for itself; each of the three a
/// later pass went on from leaves its run, keeping the run's scratch again after itself, and
/// the passes before and after them form runs of their own. The texels left are those of a
/// private texture whose four passes drew the new rectangle, and the bytes between the rows keep
/// theirs.
#[test]
fn rebound_passes_leave_their_runs() {
    let (host, moved, moved_to) = (Host::new(ROW * H), Host::new(1 << 16), Host::new(1 << 16));
    write_corners(&moved, BEFORE);
    write_corners(&moved_to, MOVED);
    let cache = Scratch::new("shadow-runs-rebound");
    let device = device(&cache.0);
    let queue = device.new_command_queue().unwrap();
    let blending = blending(&device);
    let reference = private_texture(&queue);
    let moved_rect = staged(&device, &bytes_of(&MOVED, f32::to_le_bytes));

    let command_buffer = queue.command_buffer().unwrap();
    for i in 0..REBOUND_PASSES {
        pass(
            &command_buffer,
            &reference,
            &blending,
            corners(i, &moved_rect),
            None,
        );
    }
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let left = texels_of(&queue, &reference);
    let (Some(buffer), Some(named)) = (over_pages(&device, &host), over_pages(&device, &moved))
    else {
        return;
    };
    fill(&host);
    let texture = texture_over(&buffer);
    let gate = device.new_shared_event().unwrap();
    let before = device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    (command_buffer.encode_wait_for_event(EventRef::Shared(&gate), 1)).unwrap();
    for i in 0..REBOUND_PASSES {
        pass(
            &command_buffer,
            &texture,
            &blending,
            corners(i, &named),
            None,
        );
    }
    command_buffer.commit().unwrap();
    assert_eq!(named.detach_backing(), Ok(true), "the detach");
    // SAFETY: the pages outlive every buffer over them, which alone uses them while it lives.
    unsafe { named.rebind_bytes_no_copy(moved_to.memory(), None, None) }.unwrap();
    let encoded = device.work_counts();
    gate.set_signaled_value(1);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let after = device.work_counts();
    let count = Counted::between(&before, &after);
    println!(
        "SHADOW-RUNS rebound: {:?} {:?}; encoders encoded again {}, refused {}; shadow copies {}, \
         texels kept {}, merge dispatches planned {}",
        command_buffer.status(),
        command_buffer.error(),
        encoded.encoders_reencoded - before.encoders_reencoded,
        encoded.reencodes_refused - before.reencodes_refused,
        count.copies,
        count.kept,
        count.planned,
    );
    assert_eq!(command_buffer.error(), None);
    assert_eq!(
        encoded.encoders_reencoded - before.encoders_reencoded,
        NAMING.len() as u64,
        "the rebind encodes the passes that bound the buffer again"
    );
    // The first pass's keep and the run's, then each pass encoded again: its own keep, and,
    // where a later pass went on from it, one after itself.
    let again: u64 = NAMING.iter().map(|&i| if i == 0 { 1 } else { 2 }).sum();
    assert_eq!(
        count.kept,
        (2 + again) * W * H,
        "each pass encoded again keeps for itself, and for the pass that went on from it"
    );
    drop(texture);
    let bytes = host_bytes(&host, ROW * H);
    same(
        "the bytes left",
        &(texels(&bytes), between_changed(&bytes)),
        &left,
    );
}

/// The scissors of [`scissored_runs_keep_again_once_and_merge_in_blocks`]'s passes: the first
/// pass's, then one outside it, then others anywhere.
const SCISSORS: [[u64; 4]; 10] = [
    [64, 64, 256, 256],
    [512, 512, 256, 256],
    [0, 0, 1024, 128],
    [700, 100, 300, 600],
    [96, 600, 128, 400],
    [400, 400, 64, 64],
    [0, 960, 1024, 64],
    [256, 0, 512, 1024],
    [900, 900, 124, 124],
    [128, 300, 640, 200],
];

/// The counts the scissored passes are told by, between two readings.
struct Scissored {
    kept: u64,
    planned: u64,
    in_blocks: u64,
}

/// [`SCISSORS`]'s passes over `texture`, each blending its index's color over the whole texture
/// under its scissor, in one command buffer, committed and completed; a blit between every two
/// where `apart`. The passes that kept, and the counts.
fn scissored(
    queue: &CommandQueue,
    texture: &Texture,
    state: &RenderPipelineState,
    apart: bool,
) -> (Vec<usize>, Scissored) {
    let device = queue.device();
    let (from, into) = (staged(device, &[1; 4]), staged(device, &[0; 4]));
    let before = device.work_counts();
    let mut keeps = Vec::new();
    let mut kept = before.shadow_texels_kept;
    let command_buffer = queue.command_buffer().unwrap();
    for (i, [x, y, width, height]) in SCISSORS.into_iter().enumerate() {
        if apart && i > 0 {
            let mut encoder = command_buffer.blit_command_encoder().unwrap();
            (encoder.copy_from_buffer_to_buffer(&from, 0, &into, 0, 4)).unwrap();
            encoder.end_encoding().unwrap();
        }
        let mut descriptor = RenderPassDescriptor::new();
        let color = &mut descriptor.color_attachments[0];
        color.texture = Some(texture.clone());
        color.load_action = LoadAction::Load;
        color.store_action = StoreAction::Store;
        let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
        encoder.set_render_pipeline_state(state).unwrap();
        (encoder.set_scissor_rect(ScissorRect {
            x,
            y,
            width,
            height,
        }))
        .unwrap();
        let whole = [-1.0f32, -1.0, 1.0, 1.0];
        (encoder.set_vertex_bytes(&bytes_of(&whole, f32::to_le_bytes), 0)).unwrap();
        let rgba = drawn(i).1;
        (encoder.set_fragment_bytes(&bytes_of(&rgba, f32::to_le_bytes), 0)).unwrap();
        (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
        encoder.end_encoding().unwrap();
        let now = device.work_counts().shadow_texels_kept;
        if now > kept {
            keeps.push(i);
        }
        kept = now;
    }
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let after = device.work_counts();
    let counts = Scissored {
        kept: after.shadow_texels_kept - before.shadow_texels_kept,
        planned: after.shadow_merge_dispatches_planned - before.shadow_merge_dispatches_planned,
        in_blocks: after.shadow_merge_dispatches_in_blocks
            - before.shadow_merge_dispatches_in_blocks,
    };
    (keeps, counts)
}

/// [`SCISSORS`]'s passes over the texture, as a run and with every pass apart, against a private
/// texture's: the same texels left and the bytes between the rows kept. As a run the first pass
/// keeps its scissor's texels and merges in place, the second, which opens the run, its own
/// scissor's, and the third, which writes outside them, every texel, and no later pass keeps;
/// where the passes apart merge in whole blocks, so do the run's merges.
#[test]
fn scissored_runs_keep_again_once_and_merge_in_blocks() {
    let host = Host::new(ROW * H);
    let cache = Scratch::new("shadow-runs-scissored");
    let device = device(&cache.0);
    let queue = device.new_command_queue().unwrap();
    let blending = blending(&device);
    let reference = private_texture(&queue);
    scissored(&queue, &reference, &blending, false);
    let left = texels_of(&queue, &reference);
    let Some(buffer) = over_pages(&device, &host) else {
        return;
    };
    let mut counted = Vec::new();
    for apart in [false, true] {
        fill(&host);
        let texture = texture_over(&buffer);
        let (keeps, count) = scissored(&queue, &texture, &blending, apart);
        println!(
            "SHADOW-RUNS scissored, apart {apart}: kept at {keeps:?}, texels kept {}, merge \
             dispatches planned {}, in blocks {}",
            count.kept, count.planned, count.in_blocks
        );
        drop(texture);
        let bytes = host_bytes(&host, ROW * H);
        same(
            &format!("scissored, apart {apart}, the bytes left"),
            &(texels(&bytes), between_changed(&bytes)),
            &left,
        );
        let firsts: Vec<usize> = match apart {
            true => (0..SCISSORS.len()).collect(),
            false => vec![0, 1, 2],
        };
        assert_eq!(
            keeps, firsts,
            "scissored, apart {apart}: the passes that keep"
        );
        counted.push(count);
    }
    let (run, apart) = (&counted[0], &counted[1]);
    let scissor = |i: usize| SCISSORS[i][2] * SCISSORS[i][3];
    assert_eq!(
        run.kept,
        scissor(0) + scissor(1) + W * H,
        "the first two scissors' texels, then every texel"
    );
    if apart.in_blocks == apart.planned {
        assert_eq!(
            run.in_blocks, run.planned,
            "the run's merges store whole blocks too"
        );
    }
}
