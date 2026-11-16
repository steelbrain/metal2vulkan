//! Image uses after writes through a buffer over their memory on a device (`alias_split.rs`),
//! around render pass instances: a render encoder whose pass could continue the instance the
//! render encoder before it left open (`encoder_fold.rs`'s **Continued passes**) continues it on a
//! device that splits at aliases only where no image its pass uses meets the bytes the group's
//! encoders wrote, else begins its own in a `VkCommandBuffer` of its own, counted; and every byte
//! read through an image after such a write is the write's: after a continuation, in a later
//! instance of the same pass, and through a texture over another buffer of the same host pages.
//!
//! A device whose driver has `DriverFact::ImageUsesMissBufferWritesInOneCommandBuffer` (NVIDIA's
//! 620.70.06 on the RTX 3090) splits by its own facts; any other runs each case as its own and
//! with the split forced (`DeviceShared::aliases_split_forced`), so a CPU device counts the
//! splits the RTX 3090's driver takes.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`). The functions: `smw_full` and `smw_early` of the owned fixture
//! `validation/fixtures/public/engine/smw_writers.ll` (a triangle over the whole target whose every
//! fragment adds 1 to the word at `buffer(1)` and returns `buffer(0)`'s color), `cover_vertex` and
//! `read_fragment` of `examples/heap_draws.ll` (the texel of texture 0 under each fragment), and
//! `colored_vertex` and `colored_fragment` of `examples/runtime_examples.ll`.

use super::blit_holds::{options, Scratch, WAIT};
use super::encoder_holds::{render_state, words_of};
use super::rebind_behind_a_wait::{Page, BYTES};
use crate::{
    BlitOption, Buffer, CommandBuffer, CommandBufferStatus, CpuCacheMode, Device, Fence,
    HazardTrackingMode, Heap, HeapDescriptor, HeapType, LoadAction, Origin, RenderCommandEncoder,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size, StorageMode, StoreAction,
    Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::device::RenderLoweringPolicy;
use metal2vulkan::metal::{BarrierScope, PixelFormat, PrimitiveType, RenderStages, TextureUsage};
use std::sync::atomic::Ordering;
use std::time::Duration;

/// The target's side, in texels.
const SIDE: u64 = 16;
/// The side of a texture over a buffer's bytes, its rows' bytes, and its bytes: the measured
/// probe's 64x64 `RGBA8Unorm` (`driver_facts.rs`), one host page of the tests' (`Page`).
const WIDE: u64 = 64;
const ROW: u64 = WIDE * 4;
const TEXTURE_BYTES: u64 = ROW * WIDE;
/// The word the writer adds 1 to for each fragment, before its draw.
const RUNS_BEFORE: u32 = 0x0102_0300;
/// The longest the device's check of images in place may take.
const CHECK_WAIT: Duration = Duration::from_secs(120);

/// The functions and buffers every case draws with, on a device with a buffer over host pages, so
/// it logs its render encoders' calls (`encoder_fold.rs`: a render encoder takes a group only so).
struct Lane {
    device: Device,
    /// `smw_full` and `smw_early`: every fragment adds 1 to fragment buffer 1's first word.
    writer: RenderPipelineState,
    /// `cover_vertex` and `read_fragment`: the texel of fragment texture 0 under each fragment.
    reader: RenderPipelineState,
    /// `colored_vertex` and `colored_fragment`: the quad over the target, blue.
    plain: RenderPipelineState,
    /// `smw_full`'s depth, 0.
    depth: Buffer,
    /// `smw_early`'s color, a `half4`.
    tint: Buffer,
    /// The quad's positions and its vertices' colors.
    positions: Buffer,
    colors: Buffer,
    /// The buffer over host pages, and the pages, dropped after it and the device.
    _host: Buffer,
    _page: Page,
}

impl Lane {
    /// A lane on a device that lowers render calls at the call, with a buffer over host pages;
    /// `None`, printed, where the device imports no host memory.
    fn new(scratch: &Scratch) -> Option<Lane> {
        let mut options = options(&scratch.0);
        options.policy.render_lowering = RenderLoweringPolicy::AtCall;
        let device = Device::system_default_with(&options).unwrap();
        let page = Page::new();
        let host = no_copy(&device, &page)?;
        let corners: [[f32; 2]; 6] = [
            [-1.0, 1.0],
            [1.0, 1.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            [1.0, -1.0],
            [-1.0, -1.0],
        ];
        let tint: Vec<u8> = [0x0000u16, 0x0000, 0x3c00, 0x3c00]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect();
        Some(Lane {
            writer: render_state(&device, "engine/smw_writers.ll", "smw_full", "smw_early"),
            reader: render_state(
                &device,
                "examples/heap_draws.ll",
                "cover_vertex",
                "read_fragment",
            ),
            plain: render_state(
                &device,
                "examples/runtime_examples.ll",
                "colored_vertex",
                "colored_fragment",
            ),
            depth: bytes(&device, &0.0f32.to_le_bytes()),
            tint: bytes(&device, &tint),
            positions: floats(&device, corners.into_iter().flatten()),
            colors: floats(&device, (0..6).flat_map(|_| [0.0, 0.0, 1.0, 1.0])),
            _host: host,
            _page: page,
            device,
        })
    }

    /// Split at aliases as the RTX 3090's driver does where `forced`, else by the device's own
    /// facts.
    fn force(&self, forced: bool) {
        (self.device.shared().aliases_split_forced).store(forced, Ordering::Relaxed);
    }

    /// A `Private` `RGBA8Unorm` target of [`SIDE`] texels a side.
    fn target(&self) -> Texture {
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
        descriptor.usage = TextureUsage::RENDER_TARGET;
        descriptor.resource_options = ResourceOptions::private();
        self.device.new_texture(&descriptor).unwrap()
    }

    /// A `Shared` buffer of [`pattern`]'s bytes, and the `RGBA8Unorm` texture of [`WIDE`] texels
    /// a side over them, as `over` says.
    fn over_pattern(&self, over: Over) -> Written {
        match over {
            Over::Buffer => {
                let runs = bytes(&self.device, &pattern());
                let texture = self::over(&runs);
                Written {
                    runs,
                    texture,
                    _heap: None,
                }
            }
            Over::Heap => {
                let options = ResourceOptions::new(
                    StorageMode::Shared,
                    CpuCacheMode::DefaultCache,
                    HazardTrackingMode::Tracked,
                );
                let heap = (self.device)
                    .new_heap(&HeapDescriptor::new(HeapType::Placement, 1 << 20, options))
                    .unwrap();
                let mut descriptor =
                    TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, WIDE, WIDE, false);
                descriptor.usage = TextureUsage::SHADER_READ;
                descriptor.resource_options = options;
                let texture = heap.new_texture_at_offset(&descriptor, 0).unwrap();
                let size = (self.device.heap_texture_size_and_align(&descriptor))
                    .unwrap()
                    .size;
                let runs =
                    (heap.new_buffer_at_offset(size.max(TEXTURE_BYTES), options, 0)).unwrap();
                let pattern = pattern();
                let contents = runs.contents().unwrap();
                // SAFETY: the buffer's first `TEXTURE_BYTES` bytes, which no command uses yet.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        pattern.as_ptr(),
                        contents.as_ptr(),
                        pattern.len(),
                    )
                };
                Written {
                    runs,
                    texture,
                    _heap: Some(heap),
                }
            }
        }
    }

    /// The writer's draw: every fragment over the target adds 1 to `runs`' first word.
    fn write(&self, encoder: &mut RenderCommandEncoder, runs: &Buffer) {
        encoder.set_render_pipeline_state(&self.writer).unwrap();
        encoder.set_vertex_buffer(Some(&self.depth), 0, 0).unwrap();
        encoder.set_fragment_buffer(Some(&self.tint), 0, 0).unwrap();
        encoder.set_fragment_buffer(Some(runs), 0, 1).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
    }

    /// The reader's draw: every texel of the target takes `texture`'s under it.
    fn read(&self, encoder: &mut RenderCommandEncoder, texture: &Texture) {
        encoder.set_render_pipeline_state(&self.reader).unwrap();
        encoder.set_fragment_texture(Some(texture), 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
    }

    /// The plain quad's draw, blue over the whole target.
    fn quad(&self, encoder: &mut RenderCommandEncoder) {
        encoder.set_render_pipeline_state(&self.plain).unwrap();
        encoder
            .set_vertex_buffer(Some(&self.positions), 0, 0)
            .unwrap();
        encoder.set_vertex_buffer(Some(&self.colors), 0, 1).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
    }

    /// The leader: a render encoder over `target`, loaded, whose writer's draw adds to `runs`,
    /// then a memory barrier over buffers and textures from the fragment stage to the fragment
    /// stage, which splits its pass, then the plain quad, an instance of its attachments alone,
    /// which stays open; it updates `fence` after the fragment stage.
    fn lead(&self, command_buffer: &CommandBuffer, target: &Texture, runs: &Buffer, fence: &Fence) {
        let mut encoder = command_buffer
            .render_command_encoder(&pass(target))
            .unwrap();
        self.write(&mut encoder, runs);
        let (scope, stages) = (
            BarrierScope::BUFFERS | BarrierScope::TEXTURES,
            RenderStages::FRAGMENT,
        );
        encoder.memory_barrier_with_scope(scope, stages, stages);
        self.quad(&mut encoder);
        encoder.update_fence(fence, RenderStages::FRAGMENT).unwrap();
        encoder.end_encoding().unwrap();
    }

    /// A blit encoder that copies `texture`'s first `side` texels a side into `back`, rows
    /// `side * 4` bytes apart, after waiting for `fence`, ended.
    fn copy_back(
        &self,
        command_buffer: &CommandBuffer,
        (texture, side): (&Texture, u64),
        back: &Buffer,
        fence: &Fence,
    ) {
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.wait_for_fence(fence).unwrap();
        blit.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(side, side, 1),
            back,
            0,
            side * 4,
            side * side * 4,
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
    }

    /// A `Shared` buffer of `bytes` zero bytes.
    fn zeros(&self, bytes: u64) -> Buffer {
        self::bytes(&self.device, &vec![0; bytes as usize])
    }
}

/// A no-copy buffer of [`BYTES`] over `page`; `None`, printed, where the device imports no host
/// memory.
fn no_copy(device: &Device, page: &Page) -> Option<Buffer> {
    // SAFETY: the page outlives the buffer (dropped after it and the device), and only the
    // buffer's commands use it while it lives.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(
            page.memory(),
            BYTES as u64,
            ResourceOptions::shared(),
            None,
        )
    };
    match made {
        Ok(buffer) => Some(buffer),
        Err(crate::Error::Unsupported(gap)) => {
            eprintln!("ALIAS-SPLIT: the device imports no host memory: {gap:?}");
            None
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    }
}

/// The `RGBA8Unorm` texture of [`WIDE`] texels a side over `buffer`'s first [`TEXTURE_BYTES`].
fn over(buffer: &Buffer) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, WIDE, WIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    buffer.new_texture(&descriptor, 0, ROW).unwrap()
}

/// [`TEXTURE_BYTES`] bytes: [`RUNS_BEFORE`], then a byte pattern none of whose words is zero.
fn pattern() -> Vec<u8> {
    let mut bytes: Vec<u8> = (0..TEXTURE_BYTES)
        .map(|at| (at * 7 % 251 + 1) as u8)
        .collect();
    bytes[..4].copy_from_slice(&RUNS_BEFORE.to_le_bytes());
    bytes
}

/// A pass over `target` that loads and stores it.
fn pass(target: &Texture) -> RenderPassDescriptor {
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Load;
    color.store_action = StoreAction::Store;
    pass
}

/// A `Shared` buffer of `bytes`.
fn bytes(device: &Device, bytes: &[u8]) -> Buffer {
    device
        .new_buffer_with_bytes(bytes, ResourceOptions::shared())
        .unwrap()
}

/// A `Shared` buffer of `values`' bytes.
fn floats(device: &Device, values: impl IntoIterator<Item = f32>) -> Buffer {
    let bytes: Vec<u8> = values.into_iter().flat_map(f32::to_le_bytes).collect();
    self::bytes(device, &bytes)
}

/// Commit `command_buffer` and wait for it: it completes with no error.
fn run(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        command_buffer.status(),
        CommandBufferStatus::Completed,
        "{:?}",
        command_buffer.error()
    );
}

/// How many of the target's texels copied into `back`, [`SIDE`] a row, are not the texel of the
/// texture over `bytes` under them, [`WIDE`] a row.
fn texels_not_under(back: &Buffer, bytes: &Buffer) -> usize {
    let (back, under) = (words_of(back), words_of(bytes));
    (0..SIDE * SIDE)
        .filter(|&at| back[at as usize] != under[(at / SIDE * WIDE + at % SIDE) as usize])
        .count()
}

/// The words of `back` that differ from those of `expected`, at most four, with their places.
fn differing(back: &Buffer, expected: &Buffer) -> Vec<(usize, u32, u32)> {
    let (back, expected) = (words_of(back), words_of(expected));
    (back.iter().zip(&expected).enumerate())
        .filter(|(_, (got, want))| got != want)
        .map(|(at, (got, want))| (at, *got, *want))
        .take(4)
        .collect()
}

/// What moved of the counts the cases name.
#[derive(Debug, PartialEq, Eq)]
struct Counted {
    /// `render_passes_begun_inline`.
    begun: u64,
    /// `render_passes_begun_inline_splits`.
    splits: u64,
    /// `render_passes_continued`.
    continued: u64,
    /// `commands_recorded.cmd_begin_render_pass`.
    begins: u64,
    /// `command_buffers_split_at_aliases`.
    aliases: u64,
}

impl Counted {
    fn of(before: &WorkCounts, after: &WorkCounts) -> Counted {
        Counted {
            begun: after.render_passes_begun_inline - before.render_passes_begun_inline,
            splits: after.render_passes_begun_inline_splits
                - before.render_passes_begun_inline_splits,
            continued: after.render_passes_continued - before.render_passes_continued,
            begins: after.commands_recorded.cmd_begin_render_pass
                - before.commands_recorded.cmd_begin_render_pass,
            aliases: after.command_buffers_split_at_aliases
                - before.command_buffers_split_at_aliases,
        }
    }
}

/// Where a texture over the bytes the writer adds to lies.
#[derive(Clone, Copy, Debug)]
enum Over {
    /// Over a `Shared` buffer (`Buffer::new_texture`): where its image is linear, one aliasing
    /// the buffer's bytes, whose uses name them as the buffer's (`Texture::image_use`).
    Buffer,
    /// In a `Shared` placement heap at the offset of the heap's buffer, the measured probe's form
    /// (`driver_facts.rs`): its uses name its image's place in the heap's bytes.
    Heap,
}

/// The bytes the writer adds to, and a texture over them.
struct Written {
    runs: Buffer,
    texture: Texture,
    _heap: Option<Heap>,
}

/// What the second render encoder of [`after_leader`] reads.
#[derive(Clone, Copy, Debug)]
enum Reads {
    /// A texture over a buffer nothing writes.
    Quiet,
    /// The texture over the bytes the leader's writer adds to.
    Written,
}

/// One command buffer of the leader ([`Lane::lead`]) over a new target, a render encoder that
/// loads the target and reads, behind the leader's fence, a texture as `reads` says over all of
/// the target, and a copy of the target back: what moved, how many of the target's texels are not
/// the texel of the texture under them, and the writer's word after the frame.
fn after_leader(lane: &Lane, reads: Reads) -> (Counted, usize, u32) {
    let queue = lane.device.new_command_queue().unwrap();
    let target = lane.target();
    let written = lane.over_pattern(Over::Buffer);
    let quiet = lane.over_pattern(Over::Buffer);
    let runs = &written.runs;
    let (read, texture) = match reads {
        Reads::Quiet => (&quiet.runs, &quiet.texture),
        Reads::Written => (runs, &written.texture),
    };
    let back = lane.zeros(SIDE * SIDE * 4);
    let fence = lane.device.new_fence();
    let before = lane.device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    lane.lead(&command_buffer, &target, runs, &fence);
    let mut encoder = command_buffer
        .render_command_encoder(&pass(&target))
        .unwrap();
    encoder
        .wait_for_fence(&fence, RenderStages::FRAGMENT)
        .unwrap();
    lane.read(&mut encoder, texture);
    encoder.end_encoding().unwrap();
    lane.copy_back(&command_buffer, (&target, SIDE), &back, &fence);
    run(&command_buffer);
    let counted = Counted::of(&before, &lane.device.work_counts());
    (counted, texels_not_under(&back, read), words_of(runs)[0])
}

/// A render encoder after the leader, whose pass the leader's open instance may take: where it
/// reads a texture over a buffer nothing writes, it continues that instance on every device, so
/// two render pass instances begin for the two encoders, and nothing splits at aliases. Where it
/// reads the texture over the bytes the leader's first instance wrote through a buffer, which
/// the group's `VkCommandBuffer` holds before the open instance, it continues it only on a
/// device that does not split at aliases; on one that does (the RTX 3090's, or this device
/// forced) its pass begins its own instance in a `VkCommandBuffer` of its own, counted as a split
/// at aliases. Every texel it reads, and so writes, is the texture's under it, after the
/// leader's write (its word `RUNS_BEFORE` plus one for each of the target's texels). (Not in a
/// heap: a leader that reaches a heap's memory leaves no instance open, `encoder_fold.rs`'s
/// **Continued passes**.)
#[test]
fn a_pass_continues_the_open_instance_only_where_no_buffer_write_meets_its_images() {
    let scratch = Scratch::new("alias-split-continued");
    let Some(lane) = Lane::new(&scratch) else {
        return;
    };
    let own = lane.device.shared().splits_at_aliases();
    println!("ALIAS-SPLIT-CONTINUED: the device splits at aliases by its own facts: {own}");
    let frames = [Reads::Quiet, Reads::Written];
    // Each pass shape once, so every draw's pipeline is made before the counted frames.
    for reads in frames {
        after_leader(&lane, reads);
    }
    let written = RUNS_BEFORE + (SIDE * SIDE) as u32;
    for forced in [true, false] {
        lane.force(forced);
        let splits = forced || own;
        for reads in frames {
            let (counted, wrong, word) = after_leader(&lane, reads);
            println!(
                "ALIAS-SPLIT-CONTINUED forced {forced} reads {reads:?}: {counted:?}, wrong texels \
                 {wrong}, the writer's word {word:#010x}"
            );
            assert_eq!(word, written, "every fragment of the writer added 1");
            assert_eq!(
                wrong, 0,
                "every texel is the texture's under it, after the write"
            );
            let expected = match (reads, splits) {
                (Reads::Written, true) => (2, 1, 0, 3, 1),
                (Reads::Quiet, _) | (Reads::Written, false) => (1, 1, 1, 2, 0),
            };
            let Counted {
                begun,
                splits: inline_splits,
                continued,
                begins,
                aliases,
            } = counted;
            assert_eq!(
                (begun, inline_splits, continued, begins, aliases),
                expected,
                "(inline begun, inline splits, continued, render pass begins, splits at aliases) \
                 forced {forced} reads {reads:?}"
            );
        }
    }
    lane.force(false);
}

/// The leader, then a render encoder of the plain quad alone that continues its open instance,
/// then a blit encoder that copies the texture over the bytes the leader's first instance wrote
/// through a buffer out, behind the leader's fence: it joins their group, whose `VkCommandBuffer`
/// holds that write, and every byte it copies is the buffer's after the write. Whether the copy
/// goes on in a new `VkCommandBuffer` where the device splits at aliases is printed: the
/// texture's uses name the buffer's bytes (`Texture::image_use`), which the split does not take
/// for an image's. (Not in a heap: a leader that reaches a heap's memory leaves no instance open,
/// `encoder_fold.rs`'s **Continued passes**.)
#[test]
fn a_copy_out_of_a_texture_after_a_continuation_reads_the_leader_s_buffer_write() {
    let scratch = Scratch::new("alias-split-copy-after");
    let Some(lane) = Lane::new(&scratch) else {
        return;
    };
    let own = lane.device.shared().splits_at_aliases();
    let frame = || {
        let queue = lane.device.new_command_queue().unwrap();
        let target = lane.target();
        let written = lane.over_pattern(Over::Buffer);
        let back = lane.zeros(TEXTURE_BYTES);
        let fence = lane.device.new_fence();
        let before = lane.device.work_counts();
        let command_buffer = queue.command_buffer().unwrap();
        lane.lead(&command_buffer, &target, &written.runs, &fence);
        let mut encoder = command_buffer
            .render_command_encoder(&pass(&target))
            .unwrap();
        lane.quad(&mut encoder);
        encoder.end_encoding().unwrap();
        lane.copy_back(&command_buffer, (&written.texture, WIDE), &back, &fence);
        run(&command_buffer);
        let counted = Counted::of(&before, &lane.device.work_counts());
        let linear = written.texture.linear_image().unwrap();
        let runs = &written.runs;
        (counted, differing(&back, runs), words_of(runs)[0], linear)
    };
    frame();
    for forced in [false, true] {
        lane.force(forced);
        let (counted, wrong_words, word, linear) = frame();
        println!(
            "ALIAS-SPLIT-COPY-AFTER forced {forced} (own {own}): {counted:?}, differing words \
             (at, copied, written) {wrong_words:?}, the writer's word {word:#010x}, the texture \
             a linear image {linear}"
        );
        assert_eq!(word, RUNS_BEFORE + (SIDE * SIDE) as u32);
        assert_eq!(
            wrong_words,
            Vec::new(),
            "every byte copied out of the texture is the buffer's after the write"
        );
        assert_eq!(
            counted.continued, 1,
            "the plain encoder continued the leader's instance"
        );
    }
    lane.force(false);
}

/// One render encoder whose writer's draw adds to a buffer, then, past a memory barrier over
/// buffers and textures from the fragment stage to the fragment stage, which splits its pass
/// into a second instance recorded inline into the same `VkCommandBuffer`, the reader's draw of
/// a texture over that buffer's bytes (over a buffer, or in a heap: `Over`): where the
/// texture's image is linear, every texel it reads, and so writes, is the texture's after the
/// write. A buffer write and an image use of its memory in two instances of one pass;
/// `alias_split.rs` takes no split between them (printed). (Where a heap's texture is not
/// linear, its texels lie in the image's own layout over the bytes, which Vulkan does not define
/// and Metal does not either for a heap's aliasing resources: printed alone.)
#[test]
fn a_later_instance_of_one_pass_reads_the_buffer_write_of_an_earlier_one() {
    let scratch = Scratch::new("alias-split-one-pass");
    let Some(lane) = Lane::new(&scratch) else {
        return;
    };
    let own = lane.device.shared().splits_at_aliases();
    let frame = |over: Over| {
        let queue = lane.device.new_command_queue().unwrap();
        let target = lane.target();
        let written = lane.over_pattern(over);
        let back = lane.zeros(SIDE * SIDE * 4);
        let fence = lane.device.new_fence();
        let before = lane.device.work_counts();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer
            .render_command_encoder(&pass(&target))
            .unwrap();
        lane.write(&mut encoder, &written.runs);
        let (scope, stages) = (
            BarrierScope::BUFFERS | BarrierScope::TEXTURES,
            RenderStages::FRAGMENT,
        );
        encoder.memory_barrier_with_scope(scope, stages, stages);
        lane.read(&mut encoder, &written.texture);
        encoder
            .update_fence(&fence, RenderStages::FRAGMENT)
            .unwrap();
        encoder.end_encoding().unwrap();
        lane.copy_back(&command_buffer, (&target, SIDE), &back, &fence);
        run(&command_buffer);
        let counted = Counted::of(&before, &lane.device.work_counts());
        let linear = written.texture.linear_image().unwrap();
        let runs = &written.runs;
        (
            counted,
            texels_not_under(&back, runs),
            words_of(runs)[0],
            linear,
        )
    };
    for over in [Over::Buffer, Over::Heap] {
        frame(over);
        for forced in [false, true] {
            lane.force(forced);
            let (counted, wrong, word, linear) = frame(over);
            println!(
                "ALIAS-SPLIT-ONE-PASS {over:?} forced {forced} (own {own}): {counted:?}, wrong \
                 texels {wrong}, the writer's word {word:#010x}, the texture a linear image {linear}"
            );
            assert_eq!(word, RUNS_BEFORE + (SIDE * SIDE) as u32);
            assert_eq!(
                counted.splits, 1,
                "the barrier split the pass into a second inline instance"
            );
            if linear {
                assert_eq!(
                    wrong, 0,
                    "every texel is the texture's under it, after the write"
                );
            }
        }
    }
    lane.force(false);
}

/// Two no-copy buffers over the same host page, a blit encoder that copies new bytes into the
/// first and updates a fence, and one that waits for it and copies the texture over the second
/// out, both in one `VkCommandBuffer` (`encoder_fold.rs`): every byte copied out is the new
/// one's. The write and the image use name two keys of one memory (`host_aliases.rs`); whether
/// the second goes in a new `VkCommandBuffer` where the device splits at aliases is printed.
#[test]
fn a_copy_out_of_a_texture_over_another_buffer_of_the_same_pages_reads_the_write() {
    let scratch = Scratch::new("alias-split-host-aliases");
    let Some(lane) = Lane::new(&scratch) else {
        return;
    };
    assert!(
        lane.device.settle_in_place_checks(CHECK_WAIT),
        "the device's check of images in place"
    );
    let own = lane.device.shared().splits_at_aliases();
    let page = Page::new();
    let frame = || {
        page.write(&pattern());
        let (Some(first), Some(second)) =
            (no_copy(&lane.device, &page), no_copy(&lane.device, &page))
        else {
            return None;
        };
        let texture = over(&second);
        let new: Vec<u8> = pattern().iter().map(|byte| !byte).collect();
        let source = bytes(&lane.device, &new);
        let back = lane.zeros(TEXTURE_BYTES);
        let fence = lane.device.new_fence();
        let queue = lane.device.new_command_queue().unwrap();
        let before = lane.device.work_counts();
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(&source, 0, &first, 0, TEXTURE_BYTES)
            .unwrap();
        blit.update_fence(&fence).unwrap();
        blit.end_encoding().unwrap();
        lane.copy_back(&command_buffer, (&texture, WIDE), &back, &fence);
        run(&command_buffer);
        let counted = Counted::of(&before, &lane.device.work_counts());
        let linear = texture.linear_image().unwrap();
        Some((counted, differing(&back, &source), linear))
    };
    if frame().is_none() {
        return;
    }
    for forced in [false, true] {
        lane.force(forced);
        let (counted, wrong_words, linear) = frame().unwrap();
        println!(
            "ALIAS-SPLIT-HOST-ALIASES forced {forced} (own {own}): {counted:?}, differing words \
             (at, copied, written) {wrong_words:?}, the texture a linear image over the pages \
             {linear}"
        );
        assert_eq!(
            wrong_words,
            Vec::new(),
            "every byte copied out of the texture is the one copied into the other buffer"
        );
    }
    lane.force(false);
}
