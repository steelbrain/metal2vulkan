//! Icons rendered as a two-color pass whose second color a later draw fetches: each icon a new
//! `Private` `RGBA16Float` target of 32x32, loaded `DontCare` and stored, beside a `RG16Float`
//! second color of its size, cleared to the icon's own value and stored `DontCare`. The pass
//! draws a mark into both colors over the left half (`dcs_mark`), then a draw over the whole
//! target that fetches the second color into the first (`dcs_read`), then a flag into the first
//! color over the top rows (`dcs_flag`). A device that fetches through the pixel interlock leaves
//! the second color out of the fetching draw's subpass while the others need it in theirs, so
//! the pass runs as three render pass instances; where the second color's image lacks `STORAGE`
//! (the M3), the fetching instance renders it through a scratch, and the texture takes its
//! storage form at a commit (`texture/storage_form.rs`). Each icon's first color holds the flag
//! over the top rows, the mark's second color below them on the left, and its own clear on the
//! right.
//!
//! Four forms of the render: every icon in one command buffer; each in its own, every one made
//! before the first commits; each in its own, encoded and committed on several threads at once
//! on one queue; and every icon in one command buffer with one second color kept across icons
//! and rounds, so that its first pass renders through the scratch and its later ones in place.
//! And three homes of the icon: a `Private` texture, a texture over a `Shared` buffer's bytes, and
//! one over host pages a no-copy buffer imports, as a surface's texture lies.
//!
//! The functions are `dcs_rect`, `dcs_mark`, `dcs_read` and `dcs_flag` of the owned fixture
//! `validation/fixtures/public/engine/dcs_dont_care_split.ll`.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, over_host, Host, Scratch, WAIT};
use super::encoder_holds::{bytes_of, library};
use crate::{
    BlitOption, Buffer, ClearColor, CommandBuffer, CommandQueue, Device, LoadAction, Origin,
    PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size, StoreAction,
    Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

/// The icon's side, and an `RGBA16Float` texel's bytes.
const SIDE: u64 = 32;
const TEXEL: u64 = 8;

/// The icons each round renders, the rounds, and the threads that render them in the threaded
/// form.
const ICONS: usize = 24;
const ROUNDS: usize = 3;
const THREADS: usize = 4;

/// The draws' rectangles, in pixels (left, top, right, bottom): the mark, the fetch, the flag.
const MARK: [u64; 4] = [0, 0, SIDE / 2, SIDE];
const WHOLE: [u64; 4] = [0, 0, SIDE, SIDE];
const FLAG: [u64; 4] = [0, 0, SIDE, SIDE / 4];

/// A half's 1.0.
const ONE: u16 = 0x3c00;

/// How a run renders its icons.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Shape {
    OneBuffer,
    OwnBuffers,
    Threads,
    KeptCoverage,
}

/// Where an icon's texels lie.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Home {
    Private,
    /// A `Shared` buffer's bytes.
    Buffer,
    /// Host pages a no-copy buffer imports.
    HostPages,
}

/// The queue, the states, the target's size as `dcs_rect` takes it, and a read-back buffer.
struct Tools {
    queue: CommandQueue,
    mark: RenderPipelineState,
    read: RenderPipelineState,
    flag: RenderPipelineState,
    size: Vec<u8>,
    back: Buffer,
}

impl Tools {
    /// The tools, or `None` where the device makes no state that fetches.
    fn new(device: &Device) -> Option<Tools> {
        let library = library(device, "engine/dcs_dont_care_split.ll");
        let function = |name: &str| {
            let function = library.new_function_with_name(name).unwrap();
            StageSource::Function(function.function_descriptor())
        };
        let state = |fragment: &str| {
            let mut descriptor = RenderPipelineDescriptor::new(function("dcs_rect"));
            descriptor.fragment = Some(function(fragment));
            descriptor.color_attachments[0] =
                Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float));
            descriptor.color_attachments[1] =
                Some(ColorAttachmentDescriptor::new(PixelFormat::RG16Float));
            device.new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        };
        Some(Tools {
            queue: device.new_command_queue().unwrap(),
            mark: state("dcs_mark").unwrap(),
            read: state("dcs_read").ok()?,
            flag: state("dcs_flag").unwrap(),
            size: bytes_of(&[SIDE as f32, SIDE as f32], f32::to_le_bytes),
            back: device
                .new_buffer(
                    SIDE * SIDE * TEXEL * ICONS as u64,
                    ResourceOptions::shared(),
                )
                .unwrap(),
        })
    }

    /// A new texture of `format` at the icon's side.
    fn target(&self, format: PixelFormat) -> Texture {
        let mut d = TextureDescriptor::texture_2d(format, SIDE, SIDE, false);
        d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        d.resource_options = ResourceOptions::private();
        self.queue.device().new_texture(&d).unwrap()
    }

    /// Icon `index`'s new texture: `Private`, or over `bytes` at the icon's place in them.
    fn icon(&self, bytes: Option<&Buffer>, index: usize) -> Texture {
        let Some(bytes) = bytes else {
            return self.target(PixelFormat::RGBA16Float);
        };
        let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, SIDE, SIDE, false);
        d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        d.resource_options = bytes.resource_options();
        let at = index as u64 * SIDE * SIDE * TEXEL;
        bytes.new_texture(&d, at, SIDE * TEXEL).unwrap()
    }

    /// Icon `index`'s pass into `icon`, beside `coverage` as its second color, encoded in
    /// `command_buffer`.
    fn pass(
        &self,
        command_buffer: &CommandBuffer,
        (icon, coverage): (&Texture, &Texture),
        index: usize,
    ) {
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(icon.clone());
        color.load_action = LoadAction::DontCare;
        color.store_action = StoreAction::Store;
        let second = &mut pass.color_attachments[1];
        second.texture = Some(coverage.clone());
        second.load_action = LoadAction::Clear;
        let [x, y] = clear(index);
        second.clear_color = ClearColor::new(f64::from(x), f64::from(y), 0.0, 0.0);
        second.store_action = StoreAction::DontCare;
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        for (state, rect) in [(&self.mark, MARK), (&self.read, WHOLE), (&self.flag, FLAG)] {
            encoder.set_render_pipeline_state(state).unwrap();
            let rect = bytes_of(&rect.map(|p| p as f32), f32::to_le_bytes);
            encoder.set_vertex_bytes(&rect, 0).unwrap();
            encoder.set_vertex_bytes(&self.size, 1).unwrap();
            (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
        }
        encoder.end_encoding().unwrap();
    }

    /// Every icon's first color copied into the read-back buffer, in a command buffer of its own.
    fn read_back(&self, icons: &[Texture]) {
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        for (index, icon) in icons.iter().enumerate() {
            blit.copy_from_texture_to_buffer(
                icon,
                0,
                0,
                Origin::new(0, 0, 0),
                Size::new(SIDE, SIDE, 1),
                &self.back,
                index as u64 * SIDE * SIDE * TEXEL,
                SIDE * TEXEL,
                SIDE * SIDE * TEXEL,
                BlitOption::None,
            )
            .unwrap();
        }
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
    }

    /// Icon `index`'s wrong texels in the read-back buffer, and the first: its pixel and texel.
    fn wrong(&self, index: usize) -> (usize, Option<((u64, u64), [u16; 4])>) {
        let contents = self.back.contents().unwrap();
        let count = (SIDE * SIDE * TEXEL) as usize;
        // SAFETY: bytes within the read-back buffer, which no command buffer writes now.
        let bytes =
            unsafe { std::slice::from_raw_parts(contents.as_ptr().add(index * count), count) };
        let wrong: Vec<((u64, u64), [u16; 4])> = (bytes.as_chunks::<8>().0.iter())
            .enumerate()
            .map(|(at, t)| {
                let h = |i: usize| u16::from_le_bytes([t[i], t[i + 1]]);
                let pixel = (at as u64 % SIDE, at as u64 / SIDE);
                (pixel, [h(0), h(2), h(4), h(6)])
            })
            .filter(|&(pixel, texel)| texel != expected(index, pixel))
            .collect();
        (wrong.len(), wrong.first().copied())
    }
}

/// Icon `index`'s second color's clear, exact in a half.
fn clear(index: usize) -> [f32; 2] {
    [(index + 1) as f32 / 64.0, (63 - index) as f32 / 64.0]
}

/// The bits of a half that holds `value` exactly, `value` zero or normal.
fn half(value: f32) -> u16 {
    if value == 0.0 {
        return 0;
    }
    let bits = value.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    ((exponent as u16) << 10) | ((bits & 0x7f_ffff) >> 13) as u16
}

/// Icon `index`'s first color at `(x, y)`.
fn expected(index: usize, (x, y): (u64, u64)) -> [u16; 4] {
    if y < FLAG[3] {
        [ONE, 0, ONE, ONE]
    } else if x < MARK[2] {
        [half(0.25), half(0.5), ONE, ONE]
    } else {
        let [r, g] = clear(index);
        [half(r), half(g), ONE, ONE]
    }
}

/// The counts that say how the passes ran: render pass instances, splits for the interlocked
/// writes, storage forms taken.
fn moved(before: &WorkCounts, after: &WorkCounts) -> [u64; 3] {
    [
        after.render_passes_begun_instances - before.render_passes_begun_instances,
        after.instance_splits_interlocked_writes - before.instance_splits_interlocked_writes,
        after.storage_forms_taken - before.storage_forms_taken,
    ]
}

/// Commit `command_buffers` in order and wait for each.
fn commit_all(command_buffers: Vec<CommandBuffer>) {
    for command_buffer in &command_buffers {
        command_buffer.commit().unwrap();
    }
    for command_buffer in command_buffers {
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
    }
}

fn run(case: &str, shape: Shape, home: Home) {
    // Declared before the device, so dropped after it.
    let host = Host::new();
    let scratch = Scratch::new(&format!("icon-coverage-{case}"));
    let device = device(&scratch.0);
    let Some(tools) = Tools::new(&device) else {
        eprintln!(
            "ICON-COVERAGE {case} on {}: no framebuffer fetch: nothing to check",
            device.name()
        );
        return;
    };
    let length = SIDE * SIDE * TEXEL * ICONS as u64;
    let bytes = match home {
        Home::Private => None,
        Home::Buffer => Some(
            device
                .new_buffer(length, ResourceOptions::shared())
                .unwrap(),
        ),
        Home::HostPages => {
            let Some(bytes) = over_host(&device, &host, &Arc::new(AtomicUsize::new(0))) else {
                return;
            };
            Some(bytes)
        }
    };
    let kept = tools.target(PixelFormat::RG16Float);
    let mut failures = Vec::new();
    let mut counts = Vec::new();
    for round in 0..ROUNDS {
        let before = device.work_counts();
        let icons: Vec<Texture> = (0..ICONS)
            .map(|index| tools.icon(bytes.as_ref(), index))
            .collect();
        let coverages: Vec<Texture> = (0..ICONS)
            .map(|_| match shape {
                Shape::KeptCoverage => kept.clone(),
                _ => tools.target(PixelFormat::RG16Float),
            })
            .collect();
        let pair = |index: usize| (&icons[index], &coverages[index]);
        match shape {
            Shape::OneBuffer | Shape::KeptCoverage => {
                let command_buffer = tools.queue.command_buffer().unwrap();
                for index in 0..ICONS {
                    tools.pass(&command_buffer, pair(index), index);
                }
                commit_all(vec![command_buffer]);
            }
            Shape::OwnBuffers => {
                let command_buffers: Vec<CommandBuffer> = (0..ICONS)
                    .map(|_| tools.queue.command_buffer().unwrap())
                    .collect();
                for (index, command_buffer) in command_buffers.iter().enumerate() {
                    tools.pass(command_buffer, pair(index), index);
                }
                commit_all(command_buffers);
            }
            Shape::Threads => std::thread::scope(|scope| {
                for thread in 0..THREADS {
                    let (tools, pair) = (&tools, &pair);
                    scope.spawn(move || {
                        let command_buffers = (thread..ICONS).step_by(THREADS).map(|index| {
                            let command_buffer = tools.queue.command_buffer().unwrap();
                            tools.pass(&command_buffer, pair(index), index);
                            command_buffer.commit().unwrap();
                            command_buffer
                        });
                        for command_buffer in command_buffers.collect::<Vec<_>>() {
                            command_buffer.wait_until_completed_for(WAIT).unwrap();
                            assert_eq!(command_buffer.error(), None);
                        }
                    });
                }
            }),
        }
        counts.push(moved(&before, &device.work_counts()));
        tools.read_back(&icons);
        for index in 0..ICONS {
            if let (n @ 1.., first) = tools.wrong(index) {
                failures.push(format!(
                    "round {round} icon {index}: {n} wrong texels, first {first:x?}"
                ));
            }
        }
    }
    eprintln!(
        "ICON-COVERAGE {case} on {}: {} of {} icons wrong; per round (instances, interlocked \
         splits, storage forms) {counts:?}; first {:?}",
        device.name(),
        failures.len(),
        ROUNDS * ICONS,
        failures.first()
    );
    assert!(failures.is_empty(), "{case}: {failures:?}");
}

#[test]
fn icons_rendered_in_one_command_buffer_keep_their_fetched_second_colors_value() {
    run("one-buffer", Shape::OneBuffer, Home::Private);
}

#[test]
fn icons_rendered_in_command_buffers_of_their_own_keep_their_fetched_second_colors_value() {
    run("own-buffers", Shape::OwnBuffers, Home::Private);
}

#[test]
fn icons_rendered_on_several_threads_keep_their_fetched_second_colors_value() {
    run("threads", Shape::Threads, Home::Private);
}

#[test]
fn icons_sharing_one_kept_second_color_keep_its_fetched_value() {
    run("kept-coverage", Shape::KeptCoverage, Home::Private);
}

#[test]
fn icons_over_a_buffers_bytes_keep_their_fetched_second_colors_value() {
    run("over-buffer", Shape::OneBuffer, Home::Buffer);
}

#[test]
fn icons_over_host_pages_keep_their_fetched_second_colors_value() {
    run("over-host", Shape::OneBuffer, Home::HostPages);
}

#[test]
fn icons_over_host_pages_rendered_on_several_threads_keep_their_fetched_second_colors_value() {
    run("over-host-threads", Shape::Threads, Home::HostPages);
}
