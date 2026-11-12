//! Copies in of a shadow of a tracked `Private` texture's own image across command buffers
//! (`shadow_standing.rs`). A draw that samples the texture in the pass that renders it reads a
//! copy (`texture/attached_reads.rs`); a later command buffer's copy for such a draw moves only the
//! texels inside the rectangle bounding the scissors the pass before drew through, where no
//! other command wrote the texture since, and every texel where one did.
//!
//! And a chain of passes over a mipmapped texture, the dockicons guest's pattern: the first
//! clears and stores level 0, each later one renders the next level while sampling the whole
//! texture, and a later command buffer samples the levels; each pass reads the levels the passes
//! before it wrote, with the texture's usage declaring its reads or not (a usage shadow,
//! `texture/usage_shadow.rs`), the passes in command buffers of their own or in one.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are `tob_rect`,
//! `tob_solid` and `tob_sample` of the owned fixture
//! `validation/fixtures/public/engine/tob_renders.ll`, and `alv_rect`, `alv_level1`, `alv_level2`
//! and `alv_implicit` of `engine/alv_levels.ll`.

use super::blit_holds::{device, Scratch, WAIT};
use super::encoder_holds::{bytes_of, library};
use crate::{
    BlitOption, Buffer, ClearColor, CommandBuffer, CommandQueue, Device, LoadAction, Origin,
    PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions, SamplerState,
    ScissorRect, Size, StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, SamplerMipFilter, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};

const FORMAT: PixelFormat = PixelFormat::RGBA8Unorm;

/// The side of the texture an attached read reads.
const SIDE: u64 = 256;

/// The reds: the texture's clear, a solid draw's in the attached read's pass, and one's in a
/// command buffer between.
const CLEARED: u8 = 40;
const SOLID: u8 = 200;
const BETWEEN: u8 = 120;

/// The scissors of the attached read's pass: of its read, and of its solid draw.
const READ_AT: ScissorRect = ScissorRect {
    x: 96,
    y: 64,
    width: 32,
    height: 32,
};
const SOLID_AT: ScissorRect = ScissorRect {
    x: 16,
    y: 16,
    width: 32,
    height: 32,
};
/// The scissor of the solid draw of a command buffer between, apart from both.
const BETWEEN_AT: ScissorRect = ScissorRect {
    x: 160,
    y: 160,
    width: 48,
    height: 48,
};

/// The whole texture, as a scissor.
const WHOLE: ScissorRect = ScissorRect {
    x: 0,
    y: 0,
    width: SIDE,
    height: SIDE,
};

/// Whether `(x, y)` lies inside `rect`.
fn inside(rect: ScissorRect, (x, y): (u64, u64)) -> bool {
    (rect.x..rect.x + rect.width).contains(&x) && (rect.y..rect.y + rect.height).contains(&y)
}

/// The texels of the rectangle bounding `a` and `b`.
fn bound_texels(a: ScissorRect, b: ScissorRect) -> u64 {
    let width = (a.x + a.width).max(b.x + b.width) - a.x.min(b.x);
    let height = (a.y + a.height).max(b.y + b.height) - a.y.min(b.y);
    width * height
}

/// A draw over the whole target through a scissor.
#[derive(Clone, Copy)]
enum Draw {
    /// `tob_solid` of the red.
    Solid(ScissorRect, u8),
    /// `tob_sample` of the texture the pass renders: an attached read.
    Sample(ScissorRect),
}

struct Tools {
    queue: CommandQueue,
    solid: RenderPipelineState,
    sample: RenderPipelineState,
    sampler: SamplerState,
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
            descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(FORMAT));
            device
                .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                .unwrap()
        };
        Tools {
            queue: device.new_command_queue().unwrap(),
            solid: pipeline("tob_solid"),
            sample: pipeline("tob_sample"),
            sampler: device.new_sampler_state(&SamplerDescriptor::new()).unwrap(),
            back: device
                .new_buffer(SIDE * SIDE * 4, ResourceOptions::shared())
                .unwrap(),
        }
    }

    /// A private texture of one level that draws sample, cleared to [`CLEARED`].
    fn texture(&self) -> Texture {
        let device = self.queue.device();
        let mut descriptor = TextureDescriptor::texture_2d(FORMAT, SIDE, SIDE, false);
        descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        descriptor.resource_options = ResourceOptions::private();
        let texture = device.new_texture(&descriptor).unwrap();
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(texture.clone());
        color.load_action = LoadAction::Clear;
        color.clear_color = ClearColor::new(f64::from(CLEARED) / 255.0, 0.0, 0.0, 1.0);
        color.store_action = StoreAction::Store;
        let command_buffer = self.queue.command_buffer().unwrap();
        let encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.end_encoding().unwrap();
        completed(&command_buffer);
        texture
    }

    /// A pass loading and storing `texture`, of `draws`, in a command buffer of its own: the
    /// work counts before and after it.
    fn pass(&self, texture: &Texture, draws: &[Draw]) -> (WorkCounts, WorkCounts) {
        let device = self.queue.device();
        let before = device.work_counts();
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(texture.clone());
        color.load_action = LoadAction::Load;
        color.store_action = StoreAction::Store;
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        let corners = bytes_of(&[-1.0f32, -1.0, 1.0, 1.0], f32::to_le_bytes);
        for draw in draws {
            match *draw {
                Draw::Solid(scissor, red) => {
                    encoder.set_render_pipeline_state(&self.solid).unwrap();
                    let color = [f32::from(red) / 255.0, 0.0, 0.0, 1.0];
                    let color = bytes_of(&color, f32::to_le_bytes);
                    encoder.set_fragment_bytes(&color, 0).unwrap();
                    encoder.set_scissor_rect(scissor).unwrap();
                }
                Draw::Sample(scissor) => {
                    encoder.set_render_pipeline_state(&self.sample).unwrap();
                    encoder.set_fragment_texture(Some(texture), 0).unwrap();
                    (encoder.set_fragment_sampler_state(Some(&self.sampler), 0)).unwrap();
                    encoder.set_scissor_rect(scissor).unwrap();
                }
            }
            encoder.set_vertex_bytes(&corners, 0).unwrap();
            (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
        }
        encoder.end_encoding().unwrap();
        completed(&command_buffer);
        (before, device.work_counts())
    }

    /// The red of each of `texture`'s texels, row by row.
    fn reds(&self, texture: &Texture) -> Vec<u8> {
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(SIDE, SIDE, 1),
            &self.back,
            0,
            SIDE * 4,
            SIDE * SIDE * 4,
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        completed(&command_buffer);
        let bytes = self.back.contents().unwrap().as_ptr();
        // SAFETY: the read-back buffer's bytes, which no command writes now.
        let bytes = unsafe { std::slice::from_raw_parts(bytes, (SIDE * SIDE * 4) as usize) };
        // `RGBA8Unorm`'s red is its first byte.
        bytes.chunks_exact(4).map(|texel| texel[0]).collect()
    }
}

/// Commit `command_buffer` and wait for it, which must end without an error.
fn completed(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// What a command buffer between the attached read's and the next did.
#[derive(Clone, Copy, Debug)]
enum Between {
    /// No command buffer came between.
    Nothing,
    /// A pass drew [`BETWEEN`] through [`BETWEEN_AT`] alone: a write of the texture that no
    /// shadow's copy saw, which lets its standing go.
    Write,
    /// A pass read the texture through [`READ_AT`] and drew [`BETWEEN`] through [`BETWEEN_AT`]:
    /// another attached read, which moved what the first pass drew and leaves what it drew.
    AttachedRead,
}

/// A texture cleared, then a pass that samples it through [`READ_AT`] and draws [`SOLID`] through
/// [`SOLID_AT`], then `between`, then a pass that samples it over every pixel. That last pass's
/// copy in moves what changed since the last copy alone, all a standing knows of: the texels
/// inside the rectangle bounding the scissors the pass before drew through, where that pass was
/// an attached read's; every texel after another write. Every texel holds what Metal's draws
/// leave: [`SOLID`] inside [`SOLID_AT`], [`BETWEEN`] inside [`BETWEEN_AT`] where a pass drew it,
/// [`CLEARED`] elsewhere (the reads sample what is there).
#[test]
fn a_copy_in_after_an_attached_read_moves_what_its_pass_drew_alone() {
    let scratch = Scratch::new("image-standing");
    let device = device(&scratch.0);
    let tools = Tools::new(&device);
    for between in [Between::Nothing, Between::Write, Between::AttachedRead] {
        let texture = tools.texture();
        let first = [Draw::Sample(READ_AT), Draw::Solid(SOLID_AT, SOLID)];
        tools.pass(&texture, &first);
        match between {
            Between::Nothing => {}
            Between::Write => {
                tools.pass(&texture, &[Draw::Solid(BETWEEN_AT, BETWEEN)]);
            }
            Between::AttachedRead => {
                let draws = [Draw::Sample(READ_AT), Draw::Solid(BETWEEN_AT, BETWEEN)];
                tools.pass(&texture, &draws);
            }
        }
        let (before, after) = tools.pass(&texture, &[Draw::Sample(WHOLE)]);
        let delta = |count: fn(&WorkCounts) -> u64| count(&after) - count(&before);
        let texels = delta(|c| c.shadow_texels_copied_in_images);
        let skipped = delta(|c| c.shadow_texels_skipped_by_standing);
        let stale_within = delta(|c| c.shadow_texels_unskipped_stale_within);
        eprintln!(
            "IMAGE-STANDING {} {between:?}: the last pass's copy {texels} texels, {skipped} \
             skipped by standing, {stale_within} moved inside a stale rectangle",
            device.name()
        );
        let moved = match between {
            Between::Nothing => bound_texels(READ_AT, SOLID_AT),
            Between::Write => SIDE * SIDE,
            Between::AttachedRead => bound_texels(READ_AT, BETWEEN_AT),
        };
        let within = match between {
            Between::Write => 0,
            Between::Nothing | Between::AttachedRead => moved,
        };
        assert_eq!(
            (texels, texels - skipped, stale_within),
            (SIDE * SIDE, moved, within),
            "{between:?}: the last pass's copy's texels, those it moved, and those it moved \
             inside a stale rectangle"
        );
        let reds = tools.reds(&texture);
        let drew_between = !matches!(between, Between::Nothing);
        let wrong = (0..SIDE * SIDE).find_map(|at| {
            let texel = (at % SIDE, at / SIDE);
            let expected = if inside(SOLID_AT, texel) {
                SOLID
            } else if drew_between && inside(BETWEEN_AT, texel) {
                BETWEEN
            } else {
                CLEARED
            };
            let red = reds[at as usize];
            (red != expected).then_some((texel, red, expected))
        });
        assert_eq!(
            wrong, None,
            "{between:?}: the first texel (at, red, Metal's red) wrong"
        );
    }
}

/// The mipmapped texture's level-0 side, and its levels.
const MIP_SIDE: u64 = 16;
const MIP_LEVELS: usize = 5;

/// Every byte of level `l`'s texels as made, and of those the chain's first pass clears.
const MADE: [u8; MIP_LEVELS] = [0x10, 0x40, 0x80, 0xc0, 0xf0];
const CHAINED: u8 = 0x20;

/// The side of level `level`.
fn mip_side(level: u64) -> u64 {
    (MIP_SIDE >> level).max(1)
}

/// The mip chain's tools: the pipelines of `alv_level1`, `alv_level2` and `alv_implicit`, samplers
/// of nearest filters and mip filter with their levels of detail clamped at 0 and unclamped, a
/// target of the side of level 0 the later reads draw into, and a read-back buffer of a level 0.
struct Chain {
    queue: CommandQueue,
    pipelines: Vec<(&'static str, RenderPipelineState)>,
    at_zero: SamplerState,
    unclamped: SamplerState,
    target: Texture,
    back: Buffer,
}

/// A command of the chain: a pass clearing level 0 to [`CHAINED`], or one rendering `level` by
/// `function` through `sampler` while sampling the whole texture.
enum Step<'a> {
    Clear,
    Render(u64, &'static str, &'a SamplerState),
}

impl Chain {
    fn new(device: &Device) -> Chain {
        let levels = library(device, "engine/alv_levels.ll");
        let vertex = levels.new_function_with_name("alv_rect").unwrap();
        let pipelines = ["alv_level1", "alv_level2", "alv_implicit"]
            .into_iter()
            .map(|name| {
                let fragment = levels.new_function_with_name(name).unwrap();
                let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(
                    vertex.function_descriptor(),
                ));
                descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
                descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(FORMAT));
                let options = PipelineOptions::default();
                let state = device.new_render_pipeline_state(&descriptor, &options);
                (name, state.unwrap())
            })
            .collect();
        let sampler = |max_lod: f32| {
            let mut descriptor = SamplerDescriptor::new();
            descriptor.mip_filter = SamplerMipFilter::Nearest;
            descriptor.lod_max_clamp = max_lod;
            device.new_sampler_state(&descriptor).unwrap()
        };
        let mut descriptor = TextureDescriptor::texture_2d(FORMAT, MIP_SIDE, MIP_SIDE, false);
        descriptor.usage = TextureUsage::RENDER_TARGET;
        descriptor.resource_options = ResourceOptions::private();
        Chain {
            queue: device.new_command_queue().unwrap(),
            pipelines,
            at_zero: sampler(0.0),
            unclamped: sampler(f32::MAX),
            target: device.new_texture(&descriptor).unwrap(),
            back: device
                .new_buffer(MIP_SIDE * MIP_SIDE * 4, ResourceOptions::shared())
                .unwrap(),
        }
    }

    /// A private mipmapped texture of `usage`, each level holding its [`MADE`] byte.
    fn texture(&self, usage: TextureUsage) -> Texture {
        let device = self.queue.device();
        let mut descriptor = TextureDescriptor::texture_2d(FORMAT, MIP_SIDE, MIP_SIDE, true);
        descriptor.usage = usage;
        descriptor.resource_options = ResourceOptions::private();
        let texture = device.new_texture(&descriptor).unwrap();
        assert_eq!(texture.mipmap_level_count(), MIP_LEVELS as u64);
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        let mut sources = Vec::new();
        for (level, byte) in MADE.iter().enumerate() {
            let side = mip_side(level as u64);
            let bytes = vec![*byte; (side * side * 4) as usize];
            let source = device
                .new_buffer_with_bytes(&bytes, ResourceOptions::shared())
                .unwrap();
            blit.copy_from_buffer_to_texture(
                &source,
                0,
                side * 4,
                side * side * 4,
                Size::new(side, side, 1),
                &texture,
                0,
                level as u64,
                Origin::new(0, 0, 0),
                BlitOption::None,
            )
            .unwrap();
            sources.push(source);
        }
        blit.end_encoding().unwrap();
        completed(&command_buffer);
        texture
    }

    fn pipeline(&self, function: &str) -> &RenderPipelineState {
        let state = (self.pipelines.iter()).find(|(name, _)| *name == function);
        &state.unwrap().1
    }

    /// Encode `step` over `texture` into `command_buffer`.
    fn encode(&self, command_buffer: &CommandBuffer, texture: &Texture, step: &Step<'_>) {
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(texture.clone());
        color.store_action = StoreAction::Store;
        match *step {
            Step::Clear => {
                color.load_action = LoadAction::Clear;
                let clear = f64::from(CHAINED) / 255.0;
                color.clear_color = ClearColor::new(clear, clear, clear, clear);
                let encoder = command_buffer.render_command_encoder(&pass).unwrap();
                encoder.end_encoding().unwrap();
            }
            Step::Render(level, function, sampler) => {
                color.level = level;
                color.load_action = LoadAction::Load;
                let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
                (encoder.set_render_pipeline_state(self.pipeline(function))).unwrap();
                encoder.set_fragment_texture(Some(texture), 0).unwrap();
                (encoder.set_fragment_sampler_state(Some(sampler), 0)).unwrap();
                (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
                encoder.end_encoding().unwrap();
            }
        }
    }

    /// The bytes of `texture`'s `level`.
    fn level(&self, texture: &Texture, level: u64) -> Vec<u8> {
        let side = mip_side(level);
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            texture,
            0,
            level,
            Origin::new(0, 0, 0),
            Size::new(side, side, 1),
            &self.back,
            0,
            side * 4,
            side * side * 4,
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        completed(&command_buffer);
        let bytes = self.back.contents().unwrap().as_ptr();
        // SAFETY: the read-back buffer's bytes, which no command writes now.
        unsafe { std::slice::from_raw_parts(bytes, (side * side * 4) as usize) }.to_vec()
    }

    /// The bytes a pass into the chain's target writes drawing `function` through `sampler`
    /// with `texture` bound, in a command buffer of its own.
    fn sampled(
        &self,
        texture: &Texture,
        function: &'static str,
        sampler: &SamplerState,
    ) -> Vec<u8> {
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(self.target.clone());
        color.load_action = LoadAction::Clear;
        color.store_action = StoreAction::Store;
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        (encoder.set_render_pipeline_state(self.pipeline(function))).unwrap();
        encoder.set_fragment_texture(Some(texture), 0).unwrap();
        (encoder.set_fragment_sampler_state(Some(sampler), 0)).unwrap();
        (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
        encoder.end_encoding().unwrap();
        completed(&command_buffer);
        self.level(&self.target, 0)
    }
}

/// Whether every byte of `bytes` is `byte`: the first that is not, where one is not.
fn all(bytes: &[u8], byte: u8) -> Option<(usize, u8)> {
    (bytes.iter().position(|&b| b != byte)).map(|at| (at, bytes[at]))
}

/// A mipmapped texture's chain, with its usage declaring its reads and without (a usage shadow),
/// its passes in command buffers of their own and in one: a later read of level 1 through a pass
/// of its own takes the texture's shadow, where it has one, and its copy as it was made; the
/// first pass clears level 0 to [`CHAINED`]; the second renders level 1 sampling level 0 (an
/// implicit level of detail clamped at 0), the third level 2 sampling level 1 (an explicit
/// one). Each level the chain rendered then holds [`CHAINED`] in every byte, read by a later
/// command buffer's draws and copied out by a blit, and the levels after them their own bytes.
#[test]
fn a_chain_of_passes_over_a_texture_s_levels_reads_what_the_passes_before_wrote() {
    let scratch = Scratch::new("image-standing-chain");
    let device = device(&scratch.0);
    let chain = Chain::new(&device);
    let usages = [
        TextureUsage::RENDER_TARGET,
        TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
    ];
    for usage in usages {
        for apart in [true, false] {
            let case = format!("usage {usage:?}, passes apart {apart}");
            let texture = chain.texture(usage);
            let read = chain.sampled(&texture, "alv_level1", &chain.unclamped);
            assert_eq!(all(&read, MADE[1]), None, "{case}: level 1 as made");
            let steps = [
                Step::Clear,
                Step::Render(1, "alv_implicit", &chain.at_zero),
                Step::Render(2, "alv_level1", &chain.unclamped),
            ];
            match apart {
                true => {
                    for step in &steps {
                        let command_buffer = chain.queue.command_buffer().unwrap();
                        chain.encode(&command_buffer, &texture, step);
                        completed(&command_buffer);
                    }
                }
                false => {
                    let command_buffer = chain.queue.command_buffer().unwrap();
                    for step in &steps {
                        chain.encode(&command_buffer, &texture, step);
                    }
                    completed(&command_buffer);
                }
            }
            let reads = [
                (0, "alv_implicit", &chain.at_zero),
                (1, "alv_level1", &chain.unclamped),
                (2, "alv_level2", &chain.unclamped),
            ];
            for (level, function, sampler) in reads {
                let read = chain.sampled(&texture, function, sampler);
                assert_eq!(all(&read, CHAINED), None, "{case}: level {level} sampled");
            }
            for level in 0..MIP_LEVELS {
                let expected = match level {
                    0..=2 => CHAINED,
                    _ => MADE[level],
                };
                let bytes = chain.level(&texture, level as u64);
                assert_eq!(
                    all(&bytes, expected),
                    None,
                    "{case}: level {level} copied out"
                );
            }
            eprintln!(
                "IMAGE-STANDING-CHAIN {} {case}: every level right",
                device.name()
            );
        }
    }
}
