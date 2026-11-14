//! Readers after a barrier of a texture whose linear image lacks feedback loop usage, as NVIDIA's
//! drivers make none with it (`render/snapshot.rs`, leading snapshots): each reader's snapshot
//! draws run in a render pass instance of their own that leads the reader's, and the pass takes
//! no copy, where the device reads through snapshots and has `VK_ATTACHMENT_STORE_OP_NONE`
//! (Vulkan 1.3, or `VK_KHR_load_store_op_none` or `VK_EXT_load_store_op_none` below it, which the
//! runtime enables where listed: its instance is Vulkan 1.2 by default). On a device that makes
//! such an image with the usage, the test withholds it (`DeviceShared::linear_feedback_withheld`).
//!
//! The plane is large enough that the area rule takes the snapshot through its whole scissor,
//! as it weighs a leading reader even over a buffer's bytes (`RenderLowering::weigh_snapshot`).
//! On a device that measures readers (`occlusionQueryPrecise`, queries reset on the host), a
//! reader whose identity holds a count from a pass that measured it (`Texture::learned_coverage`;
//! a reader recorded once the pool made its pipeline is not measured) is weighed by what it
//! covered there (`render/coverage.rs`): the reader of the whole plane then splits with the copy,
//! and the reader of a small square, through the same scissor, keeps its leading snapshot.
//!
//! The same passes run again with the leading snapshots withheld too
//! (`DeviceShared::leading_snapshots_withheld`): each reader then splits the pass and reads the
//! split's copy, the form before them, which every device takes where the first run's do not lead
//! (lavapipe, whose passes fetch colors as input attachments, splits in both). Both runs leave the
//! same bytes, Metal's (each reader after a barrier over the render targets reads what the draws
//! before it wrote, `metal2vulkan-conformance`'s `pass-start-barrier-*` rows), and the counts
//! name the form each took: `WorkCounts::reading_draws_snapshotted_leading`, with no texel copied
//! in and no split's copy, against `WorkCounts::attachment_read_splits`.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are `tob_rect`,
//! `tob_solid` and `tob_read` of the owned fixture `validation/fixtures/public/engine/tob_renders.ll`.

use super::blit_holds::{device, Scratch, WAIT};
use super::encoder_holds::{bytes_of, library};
use crate::framebuffer_fetch::through_storage;
use crate::pipeline::RenderStateKind;
use crate::render::coverage::ReaderKey;
use crate::{
    Buffer, CommandQueue, Device, IoSurfacePlane, LoadAction, PipelineOptions,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, ScissorRect, StoreAction, Texture,
    TextureDescriptor, WorkCounts,
};
use metal2vulkan::device::VkFeature;
use metal2vulkan::metal::{BarrierScope, PixelFormat, PrimitiveType, RenderStages, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan::vk::ApiVersion;
use std::sync::atomic::Ordering;
use std::time::Duration;

/// The plane: `WIDTH` texels a row of 4 bytes, `HEIGHT` rows, rows of `ROW` bytes (4096, a row
/// pitch every device's linear image takes). Through its whole scissor, a reader with no count
/// takes the snapshot by the area rule.
const WIDTH: u64 = 1024;
const HEIGHT: u64 = 1024;
const ROW: u64 = WIDTH * 4;
const LENGTH: u64 = ROW * HEIGHT;

/// The byte every channel of the plane holds before each pass.
const BEFORE: u8 = 0x11;

/// The draws of each pass before its readers: a value over rows, each then behind a barrier over
/// the render targets and a reader through the whole plane's scissor of the corners given, in
/// normalized device coordinates: the whole plane, then a square of 64 pixels a side.
const DRAWS: [(u8, std::ops::Range<u64>, [f32; 4]); 2] = [
    (0x99, 0..512, [-1.0, -1.0, 1.0, 1.0]),
    (0x55, 256..768, [-0.5, -0.5, -0.375, -0.375]),
];

/// The pixels each reader of [`DRAWS`] covers: the whole plane, and the square.
const COVERED: [u64; 2] = [WIDTH * HEIGHT, 64 * 64];

/// The rows `rows` of the plane, every column, as a scissor.
const fn rows(rows: std::ops::Range<u64>) -> ScissorRect {
    ScissorRect {
        x: 0,
        y: rows.start,
        width: WIDTH,
        height: rows.end - rows.start,
    }
}

/// The pipelines a pass draws with: a solid color over the plane, and the read of the plane's
/// texel under each pixel, which writes it back.
struct Tools {
    queue: CommandQueue,
    solid: RenderPipelineState,
    read: RenderPipelineState,
    back: Buffer,
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
        Tools {
            queue: device.new_command_queue().unwrap(),
            solid: pipeline("tob_solid"),
            read: pipeline("tob_read"),
            back: device
                .new_buffer(LENGTH, ResourceOptions::shared())
                .unwrap(),
        }
    }

    /// A fill of `buffer`'s bytes with `value` in a command buffer of its own.
    fn fill(&self, buffer: &Buffer, value: u8) {
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(buffer, 0..LENGTH, value).unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
    }

    /// A pass loading and storing `texture`, in a command buffer of its own: for each of
    /// [`DRAWS`], a draw of its value (every channel) through its rows, a barrier over the render
    /// targets, and a draw of its corners through the whole texture's scissor that writes the
    /// texture's texel under each pixel, as it reads it. The work counts before and after it.
    fn pass(&self, texture: &Texture) -> (WorkCounts, WorkCounts) {
        let device = self.queue.device();
        let before = device.work_counts();
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(texture.clone());
        color.load_action = LoadAction::Load;
        color.store_action = StoreAction::Store;
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        let whole = bytes_of(&[-1.0f32, -1.0, 1.0, 1.0], f32::to_le_bytes);
        let fragment = RenderStages::FRAGMENT;
        for (value, drawn, read) in DRAWS {
            encoder.set_render_pipeline_state(&self.solid).unwrap();
            encoder.set_scissor_rect(rows(drawn)).unwrap();
            encoder.set_vertex_bytes(&whole, 0).unwrap();
            let unorm = f32::from(value) / 255.0;
            let rgba = bytes_of(&[unorm; 4], f32::to_le_bytes);
            encoder.set_fragment_bytes(&rgba, 0).unwrap();
            (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
            encoder.memory_barrier_with_scope(BarrierScope::RENDER_TARGETS, fragment, fragment);
            encoder.set_render_pipeline_state(&self.read).unwrap();
            encoder.set_scissor_rect(rows(0..HEIGHT)).unwrap();
            let corners = bytes_of(&read, f32::to_le_bytes);
            encoder.set_vertex_bytes(&corners, 0).unwrap();
            encoder.set_fragment_texture(Some(texture), 0).unwrap();
            (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
        }
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        (before, device.work_counts())
    }

    /// `buffer`'s bytes, as a blit copy into the read-back buffer moves them.
    fn bytes(&self, buffer: &Buffer) -> Vec<u8> {
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        (blit.copy_from_buffer_to_buffer(buffer, 0, &self.back, 0, LENGTH)).unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        let bytes = self.back.contents().unwrap().as_ptr();
        // SAFETY: the read-back buffer's bytes, which no command writes now.
        unsafe { std::slice::from_raw_parts(bytes, LENGTH as usize) }.to_vec()
    }
}

/// What one pass's readers took, between its counts `before` and `after`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Took {
    /// `WorkCounts::reading_draws_snapshotted_leading`.
    leading: u64,
    /// `WorkCounts::attachment_read_splits`.
    splits: u64,
    /// `WorkCounts::shadow_texels_copied_in`.
    copied_in: u64,
    /// `WorkCounts::reading_draws_by_coverage`.
    by_coverage: u64,
    /// `WorkCounts::reading_draws_leading_covered_within`, `_to_half` and `_over_half`.
    shares: [u64; 3],
}

fn took((before, after): (WorkCounts, WorkCounts)) -> Took {
    let delta = |count: fn(&WorkCounts) -> u64| count(&after) - count(&before);
    Took {
        leading: delta(|c| c.reading_draws_snapshotted_leading),
        splits: delta(|c| c.attachment_read_splits),
        copied_in: delta(|c| c.shadow_texels_copied_in),
        by_coverage: delta(|c| c.reading_draws_by_coverage),
        shares: [
            delta(|c| c.reading_draws_leading_covered_within),
            delta(|c| c.reading_draws_leading_covered_to_half),
            delta(|c| c.reading_draws_leading_covered_over_half),
        ],
    }
}

/// The bytes a pass leaves: each row the last of [`DRAWS`] over it wrote, [`BEFORE`] elsewhere,
/// every byte of a texel alike.
fn expected() -> Vec<u8> {
    (0..HEIGHT)
        .flat_map(|y| {
            let value = (DRAWS.iter().rev())
                .find(|(_, drawn, _)| drawn.contains(&y))
                .map_or(BEFORE, |&(value, _, _)| value);
            std::iter::repeat_n(value, ROW as usize)
        })
        .collect()
}

/// Whether `device` reads a color through snapshots (`dynamic_state::ColorWriteMasks`, the pixel
/// interlock) with `VK_ATTACHMENT_STORE_OP_NONE`, where a reader of a color whose image lacks
/// feedback loop usage takes leading snapshots (`RenderLowering::leading_snapshots`).
fn leads(device: &Device) -> bool {
    let profile = device.compile_device().profile();
    let stores_none = device.properties().api_version >= ApiVersion::V1_3
        || (crate::device::create::STORE_NONE_NAMES.iter())
            .any(|name| device.enables_extension(name));
    device.shared().color_write_masks.is_some() && through_storage(profile) && stores_none
}

/// The identities of the readers of [`DRAWS`], in their order, all of the state `read`: its first
/// and second readers of the plane in a pass (`render/coverage.rs`'s `ReaderKey`).
fn reader_keys(read: &RenderPipelineState) -> [ReaderKey; 2] {
    let RenderStateKind::Graphics(state) = read.kind() else {
        panic!("a tile state of a render descriptor");
    };
    [0, 1].map(|ordinal| ReaderKey::of(state.address(), ordinal))
}

/// Whether `device` measures readers (`render/coverage.rs`: `occlusionQueryPrecise`, and queries
/// reset on the host).
fn measures(device: &Device) -> bool {
    let precise = (device.properties().description).has(VkFeature::OcclusionQueryPrecise);
    precise && device.shared().timing.resets_on_host()
}

/// Three passes of two readers after barriers over a plane of a `Private` buffer, its image a
/// linear alias of the bytes without feedback loop usage, each pass after a fill of the bytes:
/// with leading snapshots where the device takes them, then with them withheld. The passes leave
/// Metal's bytes under both; the first run's readers take leading snapshots (no split, no texel
/// copied in) where the device leads, but for the whole plane's reader once a pass measured it,
/// which splits by what it covered; every other reader splits the pass. Each reader is weighed by
/// the count its identity holds as its pass records, where it holds one, the pixels it covers;
/// the others by the area rule. A device that leads and measures readers weighs a pass's readers
/// so at least once.
#[test]
fn readers_through_leading_snapshots_leave_the_split_s_bytes() {
    let scratch = Scratch::new("leading-snapshots");
    let device = device(&scratch.0);
    let settled = device.settle_in_place_checks(Duration::from_secs(120));
    (device.shared().linear_feedback_withheld).store(true, Ordering::Relaxed);
    let tools = Tools::new(&device);
    let buffer = device
        .new_buffer(LENGTH, ResourceOptions::private())
        .unwrap();
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, WIDTH, HEIGHT, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    let plane = IoSurfacePlane::new(0, ROW, true);
    let texture = (buffer.new_texture_with_iosurface_plane(&descriptor, plane)).unwrap();
    assert!(
        texture.shadow().is_none(),
        "the plane's image aliases its bytes (in-place checks settled: {settled})"
    );
    let (leads, measures) = (leads(&device), measures(&device));
    let keys = reader_keys(&tools.read);
    for withheld in [false, true] {
        (device.shared().leading_snapshots_withheld).store(withheld, Ordering::Relaxed);
        // Three times: a reader of the first pass may record once the pool made its pipeline,
        // unmeasured (`render/coverage.rs`), the later ones at once; a reader whose identity holds
        // a count, from the pass that last measured it, is weighed by it. The counts of a pass are
        // kept as its command buffer completes, before the later command buffers of its queue
        // (`command_buffer/mod.rs`): the read-back's, which the bytes waited for.
        let mut weighed = false;
        for round in 0..3 {
            let counted = keys.map(|key| texture.learned_coverage(key));
            for (reader, count) in counted.iter().enumerate() {
                let want = count.map(|_| COVERED[reader]);
                assert_eq!(*count, want, "reader {reader}'s count (round {round})");
            }
            assert!(
                measures || counted == [None; 2],
                "a device that measures no reader holds a count (round {round})"
            );
            tools.fill(&buffer, BEFORE);
            let counts = tools.pass(&texture);
            let pooled = counts.1.pipelines_pooled - counts.0.pipelines_pooled;
            let queries = counts.1.coverage_queries - counts.0.coverage_queries;
            let took = took(counts);
            eprintln!(
                "LEADING-SNAPSHOTS {}: leads {leads}, measures {measures}, withheld {withheld}, \
                 round {round}: the readers' counts held {counted:?}, {took:?}, pipelines pooled \
                 {pooled}, coverage queries {queries}",
                device.name()
            );
            let left = tools.bytes(&buffer);
            let differs = (left.iter().zip(expected())).position(|(left, want)| *left != want);
            assert_eq!(
                differs, None,
                "the first byte the readers left otherwise (withheld {withheld}, round {round})"
            );
            // The readers that split: none, or the whole plane's where its identity held a count
            // (all of its pixels, beyond the area rule's bound: the small square's are within it),
            // or both.
            let (leading, by_coverage, shares) = match leads && !withheld {
                true => {
                    let [plane, square] = counted.map(|count| u64::from(count.is_some()));
                    (2 - plane, plane + square, [square, 0, plane])
                }
                false => (0, 0, [0; 3]),
            };
            let form = Took {
                leading,
                splits: 2 - leading,
                copied_in: took.copied_in,
                by_coverage,
                shares,
            };
            assert_eq!(
                took, form,
                "the readers' form (withheld {withheld}, round {round})"
            );
            assert_eq!(
                took.copied_in > 0,
                leading < 2,
                "the splits' copies moved texels, and only theirs (round {round})"
            );
            weighed |= by_coverage > 0;
            if queries == 2 {
                assert!(
                    keys.iter()
                        .all(|&key| texture.learned_coverage(key).is_some()),
                    "a pass measured both readers, yet an identity holds no count (round {round})"
                );
            }
        }
        assert_eq!(
            weighed,
            leads && measures && !withheld,
            "a pass weighed its leading readers by what they covered (withheld {withheld})"
        );
    }
}
