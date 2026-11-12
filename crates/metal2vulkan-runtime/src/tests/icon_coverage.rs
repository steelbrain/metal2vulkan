//! Icons rendered as a window server renders them after its process starts, every pass the first
//! of targets made moments before: each icon's content in a pass of several draws, then its
//! bitmap copied out of the content by a pass of one draw.
//!
//! - **The content pass.** Its first color a new `Private` `RGBA16Float` texture of 128x128,
//!   loaded and stored; its second color a new `RG16Float` texture of its size, `RenderTarget`
//!   usage alone, cleared to the icon's own value (or loaded `DontCare`) and stored `DontCare`;
//!   and a `Stencil8` texture of its size, loaded and stored `DontCare`. It draws a mark into
//!   both colors over the left half (`dcs_mark`); over the middle rows a draw that fetches the
//!   second color and writes both from it (`con_fetch_coverage`: `w = c1.x / 2 + c1.y` into every
//!   channel of the first color, `(w, c1.y / 4)` into the second); a flag into the first color
//!   over the top rows (`dcs_flag`); a draw over the whole target that fetches the first color and
//!   halves it (`fetch_ordered`); the coverage draw again; and over the bottom rows a draw that
//!   fetches the second color into the first (`dcs_read`).
//! - **The bitmap pass.** Its first color the bitmap, an `RGBA16Float` texture a program can
//!   write (`ShaderRead`, `ShaderWrite`, `RenderTarget`) over a buffer's bytes that the CPU reads
//!   after the command buffer completed (or `Private`, read back by a blit), cleared and stored;
//!   a new `RG16Float` second color and a `Stencil8` of its size, loaded and stored `DontCare`.
//!   One draw samples the content into it, texel for texel (`alv_implicit`).
//!
//! A device that fetches through the pixel interlock leaves a color out of the subpass of the
//! draws that write it so, while the others need it in theirs, so the content pass runs as
//! several render pass instances; where a fetched color's image lacks `STORAGE` (the M3), the pass
//! renders it through a scratch, and the texture takes its storage form at a commit
//! (`texture/storage_form.rs`). Each bitmap holds, over the top rows, the flag halved; over the
//! middle rows, in every channel, the second coverage draw's `w` of the mark's second color on the
//! left and of the second color's clear on the right (zero where it loads `DontCare`: Metal's
//! answer for a `DontCare` load a pass does not store, `clear_reach.rs`); over the bottom rows the
//! mark's second color on the left and the clear on the right, each `(c1.x, c1.y, 1, 1)`. The
//! device clears such a `DontCare` load (`UnstoredDontCarePolicy::Cleared`): the default does
//! only on a driver measured to clear it no slower (`DriverFact::UnstoredClearNoSlower`, the M3
//! among them, so the M3 renders as under the default), and elsewhere leaves the driver's
//! `DONT_CARE`, whose texels on the right are a texture's earlier ones (lavapipe: from the second
//! round on, as textures are made again in freed memory).
//!
//! Four forms of the render: every icon in one command buffer; each in its own, every one made
//! before the first commits; each in its own, encoded and committed on several threads at once on
//! one queue; and every icon in one command buffer with one second color kept across the content
//! passes of every round, so that its first pass renders through the scratch and its later ones
//! in place. And three homes of the bitmap: a `Private` texture, a texture over a `Shared`
//! buffer's bytes, and one over host pages a no-copy buffer imports, as a surface's texture lies,
//! its rows one after another or 64 bytes apart past that (a pitch a device's linear image may not
//! take, which shadows the texture).
//!
//! The functions are `dcs_rect`, `dcs_mark`, `dcs_read` and `dcs_flag` of the owned fixture
//! `validation/fixtures/public/engine/dcs_dont_care_split.ll`, `con_fetch_coverage` of
//! `con_fetch_coverage.ll`, `fetch_ordered` of `fragment_fetch.ll`, and `alv_rect` and
//! `alv_implicit` of `alv_levels.ll` there.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{options, over_host, Host, Scratch, WAIT};
use super::encoder_holds::{bytes_of, library};
use crate::{
    BlitOption, Buffer, ClearColor, CommandBuffer, CommandQueue, DepthStencilState, Device,
    Library, LoadAction, Origin, PipelineOptions, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, SamplerState, Size, StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::device::UnstoredDontCarePolicy;
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{
    CompareFunction, PixelFormat, PrimitiveType, StencilOperation, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

/// The icon's side, an `RGBA16Float` texel's bytes, and an icon's bytes.
const SIDE: u64 = 128;
const TEXEL: u64 = 8;
const ICON_BYTES: u64 = SIDE * SIDE * TEXEL;

/// The rows of a bitmap over bytes: one after another, and 64 bytes apart past that, as a
/// surface's rows lie, at a pitch a device's linear image may not take.
const TIGHT: u64 = SIDE * TEXEL;
const PADDED: u64 = SIDE * TEXEL + 64;

/// The icons each round renders, the rounds, and the threads that render them in the threaded
/// form.
const ICONS: usize = 24;
const ROUNDS: usize = 3;
const THREADS: usize = 4;

/// The draws' rectangles, in pixels (left, top, right, bottom).
const MARK: [u64; 4] = [0, 0, SIDE / 2, SIDE];
const MIDDLE: [u64; 4] = [0, SIDE / 4, SIDE, SIDE * 3 / 4];
const FLAG: [u64; 4] = [0, 0, SIDE, SIDE / 4];
const WHOLE: [u64; 4] = [0, 0, SIDE, SIDE];
const BOTTOM: [u64; 4] = [0, SIDE * 3 / 4, SIDE, SIDE];

/// A half's 1.0.
const ONE: u16 = 0x3c00;

/// How a run renders its icons.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Form {
    OneBuffer,
    OwnBuffers,
    Threads,
    KeptCoverage,
    /// One command buffer a round, every round's states, samplers and queue made new as it
    /// starts, as a process started again makes its own: each state's first draw is in the
    /// round's first pass.
    FreshStates,
    /// One command buffer, the content pass's stencil cleared to 0 and its draws testing it:
    /// the mark writes 1 over the left half, the first coverage draw passes where it is 1 and
    /// the second where it is not, across the instances the fetching draws split the pass into.
    StencilTests,
}

/// Where a bitmap's texels lie.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Home {
    Private,
    /// A `Shared` buffer's bytes, its rows this far apart.
    Buffer(u64),
    /// Host pages a no-copy buffer imports, its rows this far apart.
    HostPages(u64),
}

impl Home {
    /// How far apart a bitmap's rows lie where the CPU reads them.
    fn pitch(self) -> u64 {
        match self {
            Home::Private => TIGHT,
            Home::Buffer(pitch) | Home::HostPages(pitch) => pitch,
        }
    }
}

/// An icon's textures: its content and the content pass's second color, its clear (`None`:
/// loaded `DontCare`) and stencil, whether its draws test the stencil; its bitmap and the bitmap
/// pass's second color and stencil.
struct Icon {
    content: Texture,
    clear: Option<[f32; 2]>,
    stencil_tests: bool,
    coverage: Texture,
    stencil: Texture,
    bitmap: Texture,
    bitmap_coverage: Texture,
    bitmap_stencil: Texture,
}

/// The queue, the states, the stencil states (write the reference; pass where it equals it,
/// where it does not, always), the sampler, the target's size as `dcs_rect` takes it, the
/// coverage's tint, and a read-back buffer.
struct Tools {
    queue: CommandQueue,
    mark: RenderPipelineState,
    coverage: RenderPipelineState,
    flag: RenderPipelineState,
    halve: RenderPipelineState,
    read: RenderPipelineState,
    copy: RenderPipelineState,
    stencil_states: [DepthStencilState; 4],
    sampler: SamplerState,
    size: Vec<u8>,
    tint: Vec<u8>,
    back: Buffer,
}

impl Tools {
    /// The tools, or `None` where the device makes no state that fetches.
    fn new(device: &Device) -> Option<Tools> {
        let [dcs, con, fetch, alv] = [
            "engine/dcs_dont_care_split.ll",
            "engine/con_fetch_coverage.ll",
            "engine/fragment_fetch.ll",
            "engine/alv_levels.ll",
        ]
        .map(|path| library(device, path));
        let function = |library: &Library, name: &str| {
            let function = library.new_function_with_name(name).unwrap();
            StageSource::Function(function.function_descriptor())
        };
        let state = |vertex: StageSource, fragment: StageSource| {
            let mut descriptor = RenderPipelineDescriptor::new(vertex);
            descriptor.fragment = Some(fragment);
            descriptor.color_attachments[0] =
                Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float));
            descriptor.color_attachments[1] =
                Some(ColorAttachmentDescriptor::new(PixelFormat::RG16Float));
            descriptor.stencil_format = Some(PixelFormat::Stencil8);
            device.new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        };
        let rect = |fragment: StageSource| state(function(&dcs, "dcs_rect"), fragment);
        let stencil = |compare: CompareFunction, pass: StencilOperation| {
            let mut descriptor = DepthStencilDescriptor::default();
            for face in [
                &mut descriptor.front_face_stencil,
                &mut descriptor.back_face_stencil,
            ] {
                face.stencil_compare_function = compare;
                face.depth_stencil_pass_operation = pass;
            }
            device.new_depth_stencil_state(&descriptor).unwrap()
        };
        Some(Tools {
            stencil_states: [
                stencil(CompareFunction::Always, StencilOperation::Replace),
                stencil(CompareFunction::Equal, StencilOperation::Keep),
                stencil(CompareFunction::NotEqual, StencilOperation::Keep),
                stencil(CompareFunction::Always, StencilOperation::Keep),
            ],
            queue: device.new_command_queue().unwrap(),
            mark: rect(function(&dcs, "dcs_mark")).unwrap(),
            coverage: rect(function(&con, "con_fetch_coverage")).ok()?,
            flag: rect(function(&dcs, "dcs_flag")).unwrap(),
            halve: rect(function(&fetch, "fetch_ordered")).ok()?,
            read: rect(function(&dcs, "dcs_read")).ok()?,
            copy: state(function(&alv, "alv_rect"), function(&alv, "alv_implicit")).unwrap(),
            sampler: device.new_sampler_state(&SamplerDescriptor::new()).unwrap(),
            size: bytes_of(&[SIDE as f32, SIDE as f32], f32::to_le_bytes),
            tint: bytes_of(&[ONE; 4], u16::to_le_bytes),
            back: device
                .new_buffer(ICON_BYTES * ICONS as u64, ResourceOptions::shared())
                .unwrap(),
        })
    }

    /// A new `Private` texture of `format` at the icon's side, of `usage`.
    fn target(&self, format: PixelFormat, usage: TextureUsage) -> Texture {
        let mut d = TextureDescriptor::texture_2d(format, SIDE, SIDE, false);
        d.usage = usage;
        d.resource_options = ResourceOptions::private();
        self.queue.device().new_texture(&d).unwrap()
    }

    /// A new `RG16Float` second color, `RenderTarget` usage alone.
    fn second(&self) -> Texture {
        self.target(PixelFormat::RG16Float, TextureUsage::RENDER_TARGET)
    }

    /// A new `Stencil8` texture.
    fn stencil(&self) -> Texture {
        self.target(PixelFormat::Stencil8, TextureUsage::RENDER_TARGET)
    }

    /// Icon `index`'s new bitmap: `Private`, or over `bytes` at the icon's place in them, its
    /// rows `pitch` apart.
    fn bitmap(&self, (bytes, pitch): (Option<&Buffer>, u64), index: usize) -> Texture {
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

    /// A pass over `color`, beside `second` and `stencil`, both loaded and stored `DontCare` but
    /// that `second` is cleared to `clear` where it is given, and `stencil` to 0 where `tested`.
    fn descriptor(
        (color, load): (&Texture, LoadAction),
        (second, clear): (&Texture, Option<[f32; 2]>),
        (stencil, tested): (&Texture, bool),
    ) -> RenderPassDescriptor {
        let mut pass = RenderPassDescriptor::new();
        let first = &mut pass.color_attachments[0];
        first.texture = Some(color.clone());
        first.load_action = load;
        first.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        first.store_action = StoreAction::Store;
        let coverage = &mut pass.color_attachments[1];
        coverage.texture = Some(second.clone());
        coverage.load_action = match clear {
            Some([x, y]) => {
                coverage.clear_color = ClearColor::new(f64::from(x), f64::from(y), 0.0, 0.0);
                LoadAction::Clear
            }
            None => LoadAction::DontCare,
        };
        coverage.store_action = StoreAction::DontCare;
        let depth_stencil = &mut pass.stencil_attachment;
        depth_stencil.texture = Some(stencil.clone());
        depth_stencil.load_action = match tested {
            true => LoadAction::Clear,
            false => LoadAction::DontCare,
        };
        depth_stencil.store_action = StoreAction::DontCare;
        pass
    }

    /// An icon's content pass and bitmap pass, encoded in `command_buffer`.
    fn render(&self, command_buffer: &CommandBuffer, icon: &Icon) {
        let pass = Tools::descriptor(
            (&icon.content, LoadAction::Load),
            (&icon.coverage, icon.clear),
            (&icon.stencil, icon.stencil_tests),
        );
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_fragment_bytes(&self.tint, 0).unwrap();
        encoder.set_stencil_reference_value(1);
        // Each draw's stencil state where the draws test it: the mark writes 1, the first
        // coverage draw passes where it is 1 and the second where it is not.
        let [write, equal, unequal, always] = &self.stencil_states;
        let draws = [
            (&self.mark, MARK, write),
            (&self.coverage, MIDDLE, equal),
            (&self.flag, FLAG, always),
            (&self.halve, WHOLE, always),
            (&self.coverage, MIDDLE, unequal),
            (&self.read, BOTTOM, always),
        ];
        for (state, rect, stencil) in draws {
            if icon.stencil_tests {
                encoder.set_depth_stencil_state(stencil).unwrap();
            }
            encoder.set_render_pipeline_state(state).unwrap();
            let rect = bytes_of(&rect.map(|p| p as f32), f32::to_le_bytes);
            encoder.set_vertex_bytes(&rect, 0).unwrap();
            encoder.set_vertex_bytes(&self.size, 1).unwrap();
            (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
        }
        encoder.end_encoding().unwrap();
        let pass = Tools::descriptor(
            (&icon.bitmap, LoadAction::Clear),
            (&icon.bitmap_coverage, None),
            (&icon.bitmap_stencil, false),
        );
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&self.copy).unwrap();
        encoder
            .set_fragment_texture(Some(&icon.content), 0)
            .unwrap();
        (encoder.set_fragment_sampler_state(Some(&self.sampler), 0)).unwrap();
        (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
        encoder.end_encoding().unwrap();
    }

    /// Every `Private` bitmap copied into the read-back buffer, in a command buffer of its own.
    fn read_back(&self, icons: &[Icon]) {
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        for (index, icon) in icons.iter().enumerate() {
            blit.copy_from_texture_to_buffer(
                &icon.bitmap,
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
/// apart), its second color cleared to `facts`' clear and its draws testing the stencil or not,
/// and the first: its pixel and texel.
fn wrong(
    bytes: &Buffer,
    pitch: u64,
    (index, facts): (usize, (Option<[f32; 2]>, bool)),
) -> (usize, Option<((u64, u64), [u16; 4])>) {
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
        .filter(|&(pixel, texel)| texel != expected(facts, pixel))
        .collect();
    (wrong.len(), wrong.first().copied())
}

/// Icon `index`'s second color's clear in `form`, exact in a half; `None` for the odd icons,
/// whose second color loads `DontCare`, but where it is kept (a driver's `DontCare` may leave
/// the last icon's there).
fn clear(index: usize, form: Form) -> Option<[f32; 2]> {
    (index % 2 == 0 || form == Form::KeptCoverage)
        .then(|| [(index + 1) as f32 / 64.0, (63 - index) as f32 / 64.0])
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

/// A bitmap at `(x, y)` whose second color was cleared to `clear` (`None`: loaded `DontCare`).
fn expected((clear, stencil_tests): (Option<[f32; 2]>, bool), (x, y): (u64, u64)) -> [u16; 4] {
    // The second color there before the coverage draws: the mark's, or the clear (zero where it
    // loads `DontCare`).
    let left = x < MARK[2];
    let [r, g] = match left {
        true => [0.25, 0.5],
        false => clear.unwrap_or([0.0; 2]),
    };
    // A coverage draw's `w = c1.x / 2 + c1.y`, after which `c1 = (w, c1.y / 4)`.
    let first = r / 2.0 + g;
    if y < FLAG[3] {
        // The flag, halved.
        [half(0.5), 0, half(0.5), half(0.5)]
    } else if y < MIDDLE[3] && stencil_tests && left {
        // The first coverage draw passes where the mark wrote the stencil, then the halving;
        // the second fails there.
        [half(first / 2.0); 4]
    } else if y < MIDDLE[3] && stencil_tests {
        // Only the second coverage draw passes there.
        [half(first); 4]
    } else if y < MIDDLE[3] {
        // Both coverage draws; the halving between them is drawn over.
        [half(first / 2.0 + g / 4.0); 4]
    } else {
        [half(r), half(g), ONE, ONE]
    }
}

/// The counts that say how the passes ran: render pass instances, splits for the interlocked
/// writes, storage forms taken, textures over bytes shadowed, linear merges recorded, shadow
/// copies; variant pipelines made, host steps recorded, splits for a pipeline still being made,
/// draws dropped while their pipeline was being made.
fn moved(before: &WorkCounts, after: &WorkCounts) -> [u64; 10] {
    [
        after.render_passes_begun_instances - before.render_passes_begun_instances,
        after.instance_splits_interlocked_writes - before.instance_splits_interlocked_writes,
        after.storage_forms_taken - before.storage_forms_taken,
        after.shadowed_textures - before.shadowed_textures,
        after.linear_merges_recorded - before.linear_merges_recorded,
        after.shadow_copies - before.shadow_copies,
        after.variant_pipelines_made - before.variant_pipelines_made,
        after.host_steps_recorded - before.host_steps_recorded,
        after.instance_splits_pending_pipeline - before.instance_splits_pending_pipeline,
        after.draws_dropped_still_making - before.draws_dropped_still_making,
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
    let mut options = options(&scratch.0);
    options.policy.unstored_dont_care_loads = UnstoredDontCarePolicy::Cleared;
    let device = Device::system_default_with(&options).unwrap();
    let Some(mut tools) = Tools::new(&device) else {
        eprintln!(
            "ICON-COVERAGE {case} on {}: no framebuffer fetch: nothing to check",
            device.name()
        );
        return;
    };
    let pitch = home.pitch();
    let bytes = match home {
        Home::Private => None,
        Home::Buffer(_) => {
            let length = pitch * SIDE * ICONS as u64;
            Some(
                device
                    .new_buffer(length, ResourceOptions::shared())
                    .unwrap(),
            )
        }
        Home::HostPages(_) => {
            let Some(bytes) = over_host(&device, &host, &Arc::new(AtomicUsize::new(0))) else {
                return;
            };
            Some(bytes)
        }
    };
    let kept = tools.second();
    let mut failures = Vec::new();
    let mut counts = Vec::new();
    for round in 0..ROUNDS {
        let before = device.work_counts();
        if form == Form::FreshStates && round > 0 {
            tools = Tools::new(&device).unwrap();
        }
        let icons: Vec<Icon> = (0..ICONS)
            .map(|index| Icon {
                content: tools.target(
                    PixelFormat::RGBA16Float,
                    TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
                ),
                clear: clear(index, form),
                stencil_tests: form == Form::StencilTests,
                coverage: match form {
                    Form::KeptCoverage => kept.clone(),
                    _ => tools.second(),
                },
                stencil: tools.stencil(),
                bitmap: tools.bitmap((bytes.as_ref(), pitch), index),
                bitmap_coverage: tools.second(),
                bitmap_stencil: tools.stencil(),
            })
            .collect();
        match form {
            Form::OneBuffer | Form::KeptCoverage | Form::FreshStates | Form::StencilTests => {
                let command_buffer = tools.queue.command_buffer().unwrap();
                for icon in &icons {
                    tools.render(&command_buffer, icon);
                }
                commit_all(vec![command_buffer]);
            }
            Form::OwnBuffers => {
                let command_buffers: Vec<CommandBuffer> = (0..ICONS)
                    .map(|_| tools.queue.command_buffer().unwrap())
                    .collect();
                for (command_buffer, icon) in command_buffers.iter().zip(&icons) {
                    tools.render(command_buffer, icon);
                }
                commit_all(command_buffers);
            }
            Form::Threads => std::thread::scope(|scope| {
                for thread in 0..THREADS {
                    let (tools, icons) = (&tools, &icons);
                    scope.spawn(move || {
                        let command_buffers = (thread..ICONS).step_by(THREADS).map(|index| {
                            let command_buffer = tools.queue.command_buffer().unwrap();
                            tools.render(&command_buffer, &icons[index]);
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
        // The CPU reads a bitmap over bytes in them; a `Private` one is copied out first.
        let read = match &bytes {
            Some(bytes) => bytes,
            None => {
                tools.read_back(&icons);
                &tools.back
            }
        };
        for (index, icon) in icons.iter().enumerate() {
            let facts = (icon.clear, icon.stencil_tests);
            if let (n @ 1.., first) = wrong(read, pitch, (index, facts)) {
                failures.push(format!(
                    "round {round} icon {index}: {n} wrong texels, first {first:x?}"
                ));
            }
        }
    }
    eprintln!(
        "ICON-COVERAGE {case} on {}: {} of {} icons wrong; per round (instances, interlocked \
         splits, storage forms, shadowed textures, linear merges, shadow copies, variants made, \
         host steps, pending-pipeline splits, draws dropped) {counts:?}; first {:?}",
        device.name(),
        failures.len(),
        ROUNDS * ICONS,
        failures.first()
    );
    assert!(failures.is_empty(), "{case}: {failures:?}");
}

#[test]
fn icons_rendered_in_one_command_buffer_keep_their_fetched_colors_values() {
    run("one-buffer", Form::OneBuffer, Home::Private);
}

#[test]
fn icons_rendered_in_command_buffers_of_their_own_keep_their_fetched_colors_values() {
    run("own-buffers", Form::OwnBuffers, Home::Private);
}

#[test]
fn icons_rendered_on_several_threads_keep_their_fetched_colors_values() {
    run("threads", Form::Threads, Home::Private);
}

#[test]
fn icons_sharing_one_kept_second_color_keep_their_fetched_colors_values() {
    run("kept-coverage", Form::KeptCoverage, Home::Private);
}

#[test]
fn icon_bitmaps_over_a_buffers_bytes_keep_their_fetched_colors_values() {
    run("over-buffer", Form::OneBuffer, Home::Buffer(TIGHT));
}

#[test]
fn icon_bitmaps_over_a_buffers_padded_rows_keep_their_fetched_colors_values() {
    run("over-buffer-padded", Form::OneBuffer, Home::Buffer(PADDED));
}

#[test]
fn icon_bitmaps_over_host_pages_keep_their_fetched_colors_values() {
    run("over-host", Form::OneBuffer, Home::HostPages(TIGHT));
}

#[test]
fn icon_bitmaps_over_host_pages_padded_rows_keep_their_fetched_colors_values() {
    run("over-host-padded", Form::OneBuffer, Home::HostPages(PADDED));
}

#[test]
fn icon_bitmaps_over_host_pages_rendered_on_several_threads_keep_their_fetched_colors_values() {
    run("over-host-threads", Form::Threads, Home::HostPages(PADDED));
}

#[test]
fn icon_bitmaps_over_host_pages_in_command_buffers_of_their_own_keep_their_fetched_colors_values() {
    run(
        "over-host-own-buffers",
        Form::OwnBuffers,
        Home::HostPages(PADDED),
    );
}

#[test]
fn icons_whose_states_are_made_new_each_round_keep_their_fetched_colors_values() {
    run("fresh-states", Form::FreshStates, Home::Private);
}

#[test]
fn icon_bitmaps_over_host_pages_whose_states_are_made_new_each_round_keep_their_fetched_colors_values(
) {
    run(
        "over-host-fresh-states",
        Form::FreshStates,
        Home::HostPages(PADDED),
    );
}

#[test]
fn icons_whose_draws_test_the_stencil_across_the_split_keep_their_fetched_colors_values() {
    run("stencil-tests", Form::StencilTests, Home::Private);
}

#[test]
fn icon_bitmaps_over_host_pages_whose_draws_test_the_stencil_keep_their_fetched_colors_values() {
    run(
        "over-host-stencil-tests",
        Form::StencilTests,
        Home::HostPages(PADDED),
    );
}
