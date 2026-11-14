//! A render encoder's kept draws past a split (`render/draw_again.rs`, re-homed past a split): a
//! draw that keeps nothing only because it recorded into an instance after a split, here one that
//! reads its own attachment after a memory barrier over the render targets, re-homes the kept
//! draws set aside into the instance (`WorkCounts::kept_draws_rehomed`), and the draws after it of
//! their pipeline states are rebound (`WorkCounts::draws_rebound`) into the instance, which records
//! inline after the split (`render/inline_splits.rs`), and so are the draws again there with no
//! call between (never a Vulkan draw of their own ahead of the instance's room), with the same
//! commands of every kind, state commands and pixels as the same draws each taking every step. In
//! builds with debug assertions, every second draw again takes every step instead and checks what
//! it worked out (`render/draw_again.rs`'s checks), so the test runs those checks too.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are `tob_rect`,
//! `tob_sample`, `tob_solid` and `tob_read` of the owned fixture
//! `validation/fixtures/public/engine/tob_renders.ll` (`llvm-as`).

use super::blit_holds::{until, Scratch};
use super::draw_again::{checks_ended, device, drawn_alone, run_into, Heard, Ran};
use super::encoder_holds::library;
use crate::{
    CommandQueue, Device, PipelineOptions, Region, RenderCommandEncoder, RenderPipelineState,
    ResourceOptions, SamplerState, ScissorRect, Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{
    BarrierScope, BlendFactor, PixelFormat, PrimitiveType, RenderStages, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};

/// The target's side, in pixels.
const SIDE: u64 = 16;

/// The texels of the two textures draw A samples, the first before the split, the second after.
const TEXELS: [[u8; 4]; 2] = [[200, 0, 0, 255], [0, 0, 200, 255]];

/// The colors draw B adds, the first before the split, the second after.
const ADDED: [[u8; 4]; 2] = [[0, 51, 0, 0], [51, 51, 0, 255]];

/// The boxes `[x0, y0, x1, y1)` of the encoder's draws, in its order: A, B, the reader, A, B (and
/// B's draws again, [`AGAIN`]).
const BOXES: [[u64; 4]; 5] = [
    [0, 0, 8, 8],
    [4, 4, 12, 12],
    [2, 2, 10, 10],
    [6, 6, 14, 14],
    [8, 0, 16, 8],
];

/// The draws of the last B again after it, with no call between.
const AGAIN: u64 = 2;

/// The states and resources of the encoder's draws.
struct Draws {
    /// `tob_sample`, opaque.
    sampling: RenderPipelineState,
    /// `tob_solid`, adding its color to the target's.
    adding: RenderPipelineState,
    /// `tob_read`, opaque: the target's texel at the fragment's pixel.
    reading: RenderPipelineState,
    textures: [Texture; 2],
    sampler: SamplerState,
}

impl Draws {
    fn new(device: &Device) -> Draws {
        let state = |fragment: &str, adding: bool| {
            let library = library(device, "engine/tob_renders.ll");
            let function = |name| {
                let function = library.new_function_with_name(name).unwrap();
                StageSource::Function(function.function_descriptor())
            };
            let mut descriptor = RenderPipelineDescriptor::new(function("tob_rect"));
            descriptor.fragment = Some(function(fragment));
            let mut color = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
            if adding {
                color.blending_enabled = true;
                color.destination_rgb_blend_factor = BlendFactor::One;
                color.destination_alpha_blend_factor = BlendFactor::One;
            }
            descriptor.color_attachments[0] = Some(color);
            device
                .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                .unwrap()
        };
        let texture = |texel: [u8; 4]| {
            let mut descriptor =
                TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 1, 1, false);
            descriptor.usage = TextureUsage::SHADER_READ;
            descriptor.resource_options = ResourceOptions::shared();
            let texture = device.new_texture(&descriptor).unwrap();
            let whole = Region::new_2d(0, 0, 1, 1);
            texture.replace_region(whole, 0, 0, &texel, 4, 0).unwrap();
            texture
        };
        Draws {
            sampling: state("tob_sample", false),
            adding: state("tob_solid", true),
            reading: state("tob_read", false),
            textures: TEXELS.map(texture),
            sampler: device.new_sampler_state(&SamplerDescriptor::new()).unwrap(),
        }
    }

    /// The encoder's draws into `target` (the module docs), after the sampler and fragment buffer 0
    /// bound before them (as the walk binds them, so that every state's bind state holds the slot
    /// B binds), each setting its state, its scissor rectangle (its box) and its corners, with
    /// `between` called after each: A sampling the first texture, B adding the first color, a
    /// memory barrier over the render targets from the fragment stage to the fragment stage, the
    /// reader, A sampling the second texture, B adding the second color, and B's draw again
    /// [`AGAIN`] times.
    fn encode(
        &self,
        encoder: &mut RenderCommandEncoder,
        target: &Texture,
        between: &dyn Fn(&mut RenderCommandEncoder),
    ) {
        encoder
            .set_fragment_sampler_state(Some(&self.sampler), 0)
            .unwrap();
        let added = ADDED[0].map(|byte| f32::from(byte) / 255.0);
        encoder.set_fragment_bytes(&floats(added), 0).unwrap();
        for (at, area) in BOXES.into_iter().enumerate() {
            match at {
                0 | 3 => {
                    let texture = &self.textures[at / 3];
                    encoder.set_render_pipeline_state(&self.sampling).unwrap();
                    encoder.set_fragment_texture(Some(texture), 0).unwrap();
                }
                1 | 4 => {
                    encoder.set_render_pipeline_state(&self.adding).unwrap();
                    let added = ADDED[at / 4].map(|byte| f32::from(byte) / 255.0);
                    encoder.set_fragment_bytes(&floats(added), 0).unwrap();
                }
                _ => {
                    let fragment = RenderStages::FRAGMENT;
                    let scope = BarrierScope::RENDER_TARGETS;
                    encoder.memory_barrier_with_scope(scope, fragment, fragment);
                    encoder.set_render_pipeline_state(&self.reading).unwrap();
                    encoder.set_fragment_texture(Some(target), 0).unwrap();
                }
            }
            let [x0, y0, x1, y1] = area;
            let rect = ScissorRect {
                x: x0,
                y: y0,
                width: x1 - x0,
                height: y1 - y0,
            };
            encoder.set_scissor_rect(rect).unwrap();
            encoder.set_vertex_bytes(&corners(area), 0).unwrap();
            let drawn = encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0);
            assert_eq!(drawn, Ok(()));
            between(encoder);
        }
        for _ in 0..AGAIN {
            let drawn = encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0);
            assert_eq!(drawn, Ok(()));
            between(encoder);
        }
    }

    /// [`Draws::encode`] into a new cleared private target, with `between` after each draw, on
    /// `queue`: what it recorded and drew.
    fn run(
        &self,
        device: &Device,
        queue: &CommandQueue,
        between: &dyn Fn(&mut RenderCommandEncoder),
    ) -> Ran {
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
        descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        descriptor.resource_options = ResourceOptions::private();
        let target = device.new_texture(&descriptor).unwrap();
        let encode = |encoder: &mut RenderCommandEncoder| self.encode(encoder, &target, between);
        run_into(device, queue, &target, encode, true)
    }
}

fn floats(v: [f32; 4]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// The corners, in normalized device coordinates, of the target's pixels `[x0, x1) x [y0, y1)`.
fn corners([x0, y0, x1, y1]: [u64; 4]) -> Vec<u8> {
    let x = |p: u64| p as f32 / SIDE as f32 * 2.0 - 1.0;
    let y = |p: u64| 1.0 - p as f32 / SIDE as f32 * 2.0;
    floats([x(x0), y(y1), x(x1), y(y0)])
}

/// The two kept draws set aside when the reader splits the pass are re-homed into the instance
/// after the split, and the draws of their states after it are rebound there, as are B's draws
/// again with no call between (a draw again's every other form records a Vulkan draw of its own,
/// which an instance after a split records only from its room), with the same
/// commands, state commands and pixels as every draw taking every step after a call between each
/// that changes nothing for them (`set_tessellation_factor_scale`, which no draw of their states
/// reads), which forgets the kept draws. The reader splits the pass once in both runs, and no
/// draw keeps nothing for another reason than its own (`WorkCounts::draws_unkept_other`): a draw
/// into the instance after the split keeps itself.
#[test]
fn kept_draws_set_aside_are_rebound_past_a_split() {
    let scratch = Scratch::new("split-keeps");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    let queue = device.new_command_queue().unwrap();
    let draws = Draws::new(&device);
    until("every key's own pipeline made", || {
        let residual = device.work_counts().residual_form_draws;
        draws.run(&device, &queue, &|_| {});
        device.work_counts().residual_form_draws == residual
    });
    until("the pipeline pool idle", || {
        device.work_counts().pipeline_pool_busy == 0
    });
    checks_ended(&device);
    let forgets = |encoder: &mut RenderCommandEncoder| encoder.set_tessellation_factor_scale(1.0);
    let kept = draws.run(&device, &queue, &|_| {});
    let every = draws.run(&device, &queue, &forgets);
    eprintln!("kept: {}", kept.moved());
    eprintln!("every step: {}", every.moved());
    let counted = |of: fn(&WorkCounts) -> u64| (kept.counted(of), every.counted(of));
    assert_eq!(
        counted(|c| c.inline_split_instances),
        (1, 1),
        "the instances after the split"
    );
    assert_eq!(
        counted(|c| c.instance_splits_attached_read),
        (1, 1),
        "the splits by the reader's read of its attachment"
    );
    assert_eq!(
        counted(|c| c.kept_draws_rehomed),
        (2, 0),
        "the kept draws re-homed"
    );
    assert_eq!(
        counted(|c| c.draws_unlike_kept + c.draws_rebind_stopped),
        (0, 0),
        "the draws unlike their kept draws, or whose rebind stopped"
    );
    assert_eq!(
        counted(|c| c.draws_rebound),
        (drawn_alone(2 + AGAIN), 0),
        "the draws rebound"
    );
    assert_eq!(
        counted(|c| c.draws_unkept_other),
        (0, 0),
        "the draws that kept nothing for another reason"
    );
    assert_eq!(kept.commands, every.commands);
    assert_eq!(kept.commands["cmd_draw"], BOXES.len() as u64 + AGAIN);
    assert_eq!(
        kept.counted(|c| c.draw_states_recorded),
        every.counted(|c| c.draw_states_recorded)
    );
    assert_eq!(kept.words, every.words, "the pixels of the draws");
    // The draws after the split drew what they bound: the second texture, the second color (added
    // by B and its draws again).
    let word = |x: u64, y: u64| kept.words[(y * SIDE + x) as usize];
    assert_eq!(
        word(13, 13),
        u32::from_le_bytes(TEXELS[1]),
        "A after the split"
    );
    assert_eq!(
        word(15, 1),
        u32::from_le_bytes(ADDED[1].map(|byte| byte.saturating_mul(1 + AGAIN as u8))),
        "B after the split"
    );
}
