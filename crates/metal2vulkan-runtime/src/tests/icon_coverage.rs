//! Icons rendered as a window server renders them after its process starts: each icon's pass the
//! first of targets made moments before. Its first color is the icon, an `RGBA16Float` texture
//! of 128x128 cleared and stored, which a program can write (`ShaderRead`, `ShaderWrite`,
//! `RenderTarget`), over a buffer's bytes that the CPU reads after the pass (or `Private`, read
//! back by a blit); its second color a new `RG16Float` texture of its size, `RenderTarget` usage
//! alone, cleared to the icon's own value (or loaded `DontCare`) and stored `DontCare`; and a
//! `Stencil8` texture of its size, loaded and stored `DontCare`. The pass draws a mark into both
//! colors over the left half (`dcs_mark`), then a draw over the middle rows that fetches the second
//! color into the first (`dcs_read`), then a flag into the first color over the top rows
//! (`dcs_flag`).
//!
//! A device that fetches through the pixel interlock leaves the second color out of the fetching
//! draw's subpass while the others need it in theirs, so the pass runs as three render pass
//! instances; where the second color's image lacks `STORAGE` (the M3), the fetching instance
//! renders it through a scratch, and the texture takes its storage form at a commit
//! (`texture/storage_form.rs`). Each icon's first color holds the flag over the top rows; over the
//! middle rows the mark's second color on the left and, on the right, the second color's clear
//! (zero where it loads `DontCare`: Metal's answer for a `DontCare` load a pass does not store,
//! `clear_reach.rs`); below them the mark on the left and the icon's clear on the right.
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

/// The icon's side, an `RGBA16Float` texel's bytes, and an icon's bytes.
const SIDE: u64 = 128;
const TEXEL: u64 = 8;
const ICON_BYTES: u64 = SIDE * SIDE * TEXEL;

/// The rows of an icon over bytes: one after another, and 64 bytes apart past that, as a
/// surface's rows lie, at a pitch a device's linear image may not take.
const TIGHT: u64 = SIDE * TEXEL;
const PADDED: u64 = SIDE * TEXEL + 64;

/// The icons each round renders, the rounds, and the threads that render them in the threaded
/// form.
const ICONS: usize = 24;
const ROUNDS: usize = 3;
const THREADS: usize = 4;

/// The draws' rectangles, in pixels (left, top, right, bottom): the mark, the fetch, the flag.
const MARK: [u64; 4] = [0, 0, SIDE / 2, SIDE];
const MIDDLE: [u64; 4] = [0, SIDE / 4, SIDE, SIDE * 3 / 4];
const FLAG: [u64; 4] = [0, 0, SIDE, SIDE / 4];

/// A half's 1.0.
const ONE: u16 = 0x3c00;

/// How a run renders its icons.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Form {
    OneBuffer,
    OwnBuffers,
    Threads,
    KeptCoverage,
}

/// Where an icon's texels lie.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Home {
    Private,
    /// A `Shared` buffer's bytes, its rows this far apart.
    Buffer(u64),
    /// Host pages a no-copy buffer imports, its rows this far apart.
    HostPages(u64),
}

impl Home {
    /// How far apart an icon's rows lie where the CPU reads them.
    fn pitch(self) -> u64 {
        match self {
            Home::Private => TIGHT,
            Home::Buffer(pitch) | Home::HostPages(pitch) => pitch,
        }
    }
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
            descriptor.stencil_format = Some(PixelFormat::Stencil8);
            device.new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        };
        Some(Tools {
            queue: device.new_command_queue().unwrap(),
            mark: state("dcs_mark").unwrap(),
            read: state("dcs_read").ok()?,
            flag: state("dcs_flag").unwrap(),
            size: bytes_of(&[SIDE as f32, SIDE as f32], f32::to_le_bytes),
            back: device
                .new_buffer(ICON_BYTES * ICONS as u64, ResourceOptions::shared())
                .unwrap(),
        })
    }

    /// A new `Private` render target of `format` at the icon's side, of `usage`.
    fn target(&self, format: PixelFormat, usage: TextureUsage) -> Texture {
        let mut d = TextureDescriptor::texture_2d(format, SIDE, SIDE, false);
        d.usage = usage;
        d.resource_options = ResourceOptions::private();
        self.queue.device().new_texture(&d).unwrap()
    }

    /// Icon `index`'s new texture: `Private`, or over `bytes` at the icon's place in them, its
    /// rows `pitch` apart.
    fn icon(&self, (bytes, pitch): (Option<&Buffer>, u64), index: usize) -> Texture {
        let usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        let usage = usage | TextureUsage::RENDER_TARGET;
        let Some(bytes) = bytes else {
            return self.target(PixelFormat::RGBA16Float, usage);
        };
        let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, SIDE, SIDE, false);
        d.usage = usage;
        d.resource_options = bytes.resource_options();
        (bytes.new_texture(&d, index as u64 * pitch * SIDE, pitch)).unwrap()
    }

    /// Icon `index`'s pass into `icon`, beside `coverage` as its second color and `stencil`,
    /// encoded in `command_buffer`.
    fn pass(
        &self,
        command_buffer: &CommandBuffer,
        (icon, coverage, stencil): (&Texture, &Texture, &Texture),
        index: usize,
    ) {
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(icon.clone());
        color.load_action = LoadAction::Clear;
        color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        color.store_action = StoreAction::Store;
        let second = &mut pass.color_attachments[1];
        second.texture = Some(coverage.clone());
        second.load_action = match clear(index) {
            Some([x, y]) => {
                second.clear_color = ClearColor::new(f64::from(x), f64::from(y), 0.0, 0.0);
                LoadAction::Clear
            }
            None => LoadAction::DontCare,
        };
        second.store_action = StoreAction::DontCare;
        let stencil_attachment = &mut pass.stencil_attachment;
        stencil_attachment.texture = Some(stencil.clone());
        stencil_attachment.load_action = LoadAction::DontCare;
        stencil_attachment.store_action = StoreAction::DontCare;
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        for (state, rect) in [(&self.mark, MARK), (&self.read, MIDDLE), (&self.flag, FLAG)] {
            encoder.set_render_pipeline_state(state).unwrap();
            let rect = bytes_of(&rect.map(|p| p as f32), f32::to_le_bytes);
            encoder.set_vertex_bytes(&rect, 0).unwrap();
            encoder.set_vertex_bytes(&self.size, 1).unwrap();
            (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
        }
        encoder.end_encoding().unwrap();
    }

    /// Every `Private` icon's first color copied into the read-back buffer, in a command buffer of
    /// its own.
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
                index as u64 * ICON_BYTES,
                SIDE * TEXEL,
                ICON_BYTES,
                BlitOption::None,
            )
            .unwrap();
        }
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
    }
}

/// Icon `index`'s wrong texels in `bytes` (the icons one after another, their rows `pitch`
/// apart), and the first: its pixel and texel.
fn wrong(bytes: &Buffer, pitch: u64, index: usize) -> (usize, Option<((u64, u64), [u16; 4])>) {
    let contents = bytes.contents().unwrap();
    let span = (pitch * SIDE) as usize;
    // SAFETY: bytes within the buffer, which no command buffer writes now.
    let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr().add(index * span), span) };
    let wrong: Vec<((u64, u64), [u16; 4])> = (0..SIDE)
        .flat_map(|y| (0..SIDE).map(move |x| (x, y)))
        .map(|(x, y)| {
            let at = (y * pitch + x * TEXEL) as usize;
            let h = |i: usize| u16::from_le_bytes([bytes[at + i], bytes[at + i + 1]]);
            ((x, y), [h(0), h(2), h(4), h(6)])
        })
        .filter(|&(pixel, texel)| texel != expected(index, pixel))
        .collect();
    (wrong.len(), wrong.first().copied())
}

/// Icon `index`'s second color's clear, exact in a half; `None` for the odd icons, whose second
/// color loads `DontCare`.
fn clear(index: usize) -> Option<[f32; 2]> {
    (index % 2 == 0).then(|| [(index + 1) as f32 / 64.0, (63 - index) as f32 / 64.0])
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
    let left = x < MARK[2];
    if y < FLAG[3] {
        [ONE, 0, ONE, ONE]
    } else if y < MIDDLE[3] && left {
        [half(0.25), half(0.5), ONE, ONE]
    } else if y < MIDDLE[3] {
        let [r, g] = clear(index).unwrap_or([0.0; 2]);
        [half(r), half(g), ONE, ONE]
    } else if left {
        [0, half(0.5), 0, ONE]
    } else {
        [0; 4]
    }
}

/// The counts that say how the passes ran: render pass instances, splits for the interlocked
/// writes, storage forms taken, textures over bytes shadowed, linear merges recorded, shadow
/// copies.
fn moved(before: &WorkCounts, after: &WorkCounts) -> [u64; 6] {
    [
        after.render_passes_begun_instances - before.render_passes_begun_instances,
        after.instance_splits_interlocked_writes - before.instance_splits_interlocked_writes,
        after.storage_forms_taken - before.storage_forms_taken,
        after.shadowed_textures - before.shadowed_textures,
        after.linear_merges_recorded - before.linear_merges_recorded,
        after.shadow_copies - before.shadow_copies,
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

fn run(case: &str, form: Form, home: Home) {
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
    let pitch = home.pitch();
    let length = pitch * SIDE * ICONS as u64;
    let bytes = match home {
        Home::Private => None,
        Home::Buffer(_) => Some(
            device
                .new_buffer(length, ResourceOptions::shared())
                .unwrap(),
        ),
        Home::HostPages(_) => {
            let Some(bytes) = over_host(&device, &host, &Arc::new(AtomicUsize::new(0))) else {
                return;
            };
            Some(bytes)
        }
    };
    let coverage = || tools.target(PixelFormat::RG16Float, TextureUsage::RENDER_TARGET);
    let kept = coverage();
    let mut failures = Vec::new();
    let mut counts = Vec::new();
    for round in 0..ROUNDS {
        let before = device.work_counts();
        let icons: Vec<Texture> = (0..ICONS)
            .map(|index| tools.icon((bytes.as_ref(), pitch), index))
            .collect();
        let coverages: Vec<Texture> = (0..ICONS)
            .map(|_| match form {
                Form::KeptCoverage => kept.clone(),
                _ => coverage(),
            })
            .collect();
        let stencils: Vec<Texture> = (0..ICONS)
            .map(|_| tools.target(PixelFormat::Stencil8, TextureUsage::RENDER_TARGET))
            .collect();
        let targets = |index: usize| (&icons[index], &coverages[index], &stencils[index]);
        match form {
            Form::OneBuffer | Form::KeptCoverage => {
                let command_buffer = tools.queue.command_buffer().unwrap();
                for index in 0..ICONS {
                    tools.pass(&command_buffer, targets(index), index);
                }
                commit_all(vec![command_buffer]);
            }
            Form::OwnBuffers => {
                let command_buffers: Vec<CommandBuffer> = (0..ICONS)
                    .map(|_| tools.queue.command_buffer().unwrap())
                    .collect();
                for (index, command_buffer) in command_buffers.iter().enumerate() {
                    tools.pass(command_buffer, targets(index), index);
                }
                commit_all(command_buffers);
            }
            Form::Threads => std::thread::scope(|scope| {
                for thread in 0..THREADS {
                    let (tools, targets) = (&tools, &targets);
                    scope.spawn(move || {
                        let command_buffers = (thread..ICONS).step_by(THREADS).map(|index| {
                            let command_buffer = tools.queue.command_buffer().unwrap();
                            tools.pass(&command_buffer, targets(index), index);
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
        // The CPU reads an icon over bytes in them; a `Private` one is copied out first.
        let read = match &bytes {
            Some(bytes) => bytes,
            None => {
                tools.read_back(&icons);
                &tools.back
            }
        };
        for index in 0..ICONS {
            if let (n @ 1.., first) = wrong(read, pitch, index) {
                failures.push(format!(
                    "round {round} icon {index}: {n} wrong texels, first {first:x?}"
                ));
            }
        }
    }
    eprintln!(
        "ICON-COVERAGE {case} on {}: {} of {} icons wrong; per round (instances, interlocked \
         splits, storage forms, shadowed textures, linear merges, shadow copies) {counts:?}; \
         first {:?}",
        device.name(),
        failures.len(),
        ROUNDS * ICONS,
        failures.first()
    );
    assert!(failures.is_empty(), "{case}: {failures:?}");
}

#[test]
fn icons_rendered_in_one_command_buffer_keep_their_fetched_second_colors_value() {
    run("one-buffer", Form::OneBuffer, Home::Private);
}

#[test]
fn icons_rendered_in_command_buffers_of_their_own_keep_their_fetched_second_colors_value() {
    run("own-buffers", Form::OwnBuffers, Home::Private);
}

#[test]
fn icons_rendered_on_several_threads_keep_their_fetched_second_colors_value() {
    run("threads", Form::Threads, Home::Private);
}

#[test]
fn icons_sharing_one_kept_second_color_keep_its_fetched_value() {
    run("kept-coverage", Form::KeptCoverage, Home::Private);
}

#[test]
fn icons_over_a_buffers_bytes_keep_their_fetched_second_colors_value() {
    run("over-buffer", Form::OneBuffer, Home::Buffer(TIGHT));
}

#[test]
fn icons_over_a_buffers_padded_rows_keep_their_fetched_second_colors_value() {
    run("over-buffer-padded", Form::OneBuffer, Home::Buffer(PADDED));
}

#[test]
fn icons_over_host_pages_keep_their_fetched_second_colors_value() {
    run("over-host", Form::OneBuffer, Home::HostPages(TIGHT));
}

#[test]
fn icons_over_host_pages_padded_rows_keep_their_fetched_second_colors_value() {
    run("over-host-padded", Form::OneBuffer, Home::HostPages(PADDED));
}

#[test]
fn icons_over_host_pages_rendered_on_several_threads_keep_their_fetched_second_colors_value() {
    run("over-host-threads", Form::Threads, Home::HostPages(PADDED));
}

#[test]
fn icons_over_host_pages_in_command_buffers_of_their_own_keep_their_fetched_second_colors_value() {
    run(
        "over-host-own-buffers",
        Form::OwnBuffers,
        Home::HostPages(PADDED),
    );
}
