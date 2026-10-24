//! Device tests of a draw's encoding work on the device the cfg selects, under the validation
//! layer (the consumer's per-draw encode): each holds a piece of work the runtime once repeated at
//! every draw, whatever changed since the draw before, to what changed, through the device's work
//! counts (`Device::work_counts`, not part of the API). Deterministic: they count, and time
//! nothing; and every draw's pixels are checked, so what a draw binds stays exact.
//!
//! - **Settles.** A draw whose bind state is its encoder's last draw's makes no bind shape and
//!   settles no variant (`WorkCounts::draw_shapes_made`, `draws_settled`); nor does one that
//!   binds another texture of the same type, format and usage, whose slot keeps its shape; one of
//!   another shape (a texture of another format, a slot's binding of another class) finds its
//!   shape among those the state settled lately, or makes it and settles it: draws alternating
//!   between two shapes make and settle each once, and a pass that draws as the one before it
//!   makes and settles none. A draw that changes only values no shape reads (the depth/stencil
//!   state, the stencil references, the blend color) keeps its encoder's last draw's shape, and
//!   looks for none among those settled (`WorkCounts::draw_shapes_found`); nor do draws that
//!   switch between pipeline states whose bind states differ, each starting from its own pipeline
//!   state's last draw's state, which the encoder set aside.
//! - **Zeroed colors.** Draws into a pass of a color their pipeline state lacks, which Metal
//!   zeroes where they cover it, read the colors they zero where their encoder keeps them, compare
//!   them with their bind state's in place, and copy none while those hold (allocations counted on
//!   the drawing thread).
//! - **Argument samplers.** A draw lists the device's live sampler states as its encoder's bind
//!   state keeps them, without taking them again, until a state is made or goes
//!   (`WorkCounts::argument_samplers_taken`).
//! - **Render passes.** A pass that draws as the one before it makes no render pass: its device
//!   keeps the compatible pass and the pass its instance begins (`WorkCounts::render_passes_made`).
//! - **Views.** A pass that binds the textures the pass before it bound works out none of their
//!   views (`WorkCounts::bind_views_worked_out`); a texture made after one of them went works out
//!   its own.
//! - **Stage tables.** A draw reads into its encoder's bind state only the stage tables a call
//!   changed since a draw read them, through the same pipeline state's interface
//!   (`WorkCounts::stage_tables_read`), of each only the slots its calls changed where it keeps
//!   them (`WorkCounts::stage_tables_read_in_part`), and looks at where a no-copy buffer a table
//!   binds lies once for its binding and once for its descriptor (`WorkCounts::no_copy_lies_read`).
//! - **Pipelines.** A draw of a pipeline state one of whose last draws found its `VkPipeline`
//!   made, of the same variant and an equal key, finds that entry again without looking the key
//!   up (`WorkCounts::graphics_pipelines_looked_up`), states and keys switched between included.
//! - **Shapes.** A draw takes what it derives from its encoder's state alone (its settings, its
//!   draw state and its `VkPipeline`) from an earlier draw of its encoder that derived it from
//!   the same inputs (`WorkCounts::draw_shapes_reused`): a draw of other bytes, and one switched
//!   back to a pipeline state the encoder drew; one after any change of the encoder's state
//!   derives its own.
//! - **Recorded states.** A draw records into its secondary command buffer only the state
//!   commands that set what changed since the draws before it there
//!   (`WorkCounts::draw_states_recorded`): of an unchanged state, none (its set 0 the last draw's,
//!   or pushed alike where the device pushes set 0, `WorkCounts::draws_pushing_set_zero`);
//!   switching pipelines, not the dynamic states both make dynamic.
//! - **Bound sets.** Where the device binds set 0, a draw that writes the descriptors its
//!   encoder's last draw wrote into a set of the same layout binds that set again
//!   (`WorkCounts::set_zero_sets_reused`), and one that writes those an earlier draw of its
//!   command buffer wrote, in any of its encoders, binds that one
//!   (`WorkCounts::set_zero_sets_found`); a draw that writes descriptors no kept set holds writes
//!   a set of its own. Inline bytes are bound at their dynamic offset where the variant makes
//!   their binding dynamic, so other bytes change no descriptor.
//! - **Pushed sets.** Where the device pushes set 0, a draw that changes some of its set's
//!   descriptors pushes only those (`WorkCounts::set_zero_writes_pushed`). Each test of set 0 runs
//!   with `VK_KHR_push_descriptor` withheld too, so every device runs the path of one without the
//!   push command, which binds set 0.
//! - **Retains.** An argument table whose every object its encoder holds already, from its last
//!   retain or an earlier one of any table, is not retained again
//!   (`WorkCounts::argument_tables_retained`); one that binds an object the encoder does not hold
//!   is, taking hold of those alone (`WorkCounts::argument_objects_retained`), and an object the
//!   program let go of after its draw lives until the command buffer completes. Its encoder lists
//!   every object a retain took or held already (`CommandBuffer::ended_encoders`), a slot rebound
//!   to one that looks like its last included.
//! - **Uploads.** Draws that set their bytes write them, and their tables, into no device-local
//!   memory on a device with host-coherent, host-cached system memory
//!   (`WorkCounts::upload_chunks_in_device_memory`).
//! - **Resident set.** A draw takes its encoder's last draw's used resources while what the
//!   encoder made resident is unchanged (`Residency::generation`), and states those resident as it
//!   was encoded: after a resource made resident, a usage or render stage widened, or another made
//!   resident through `useResources`, the new set.
//!
//! The pipeline is the owned fixture `tob_renders`'s `tob_rect` (a rectangle of the corners at
//! vertex buffer 0) and `tob_sample` (texture 0 through sampler 0, where its pixel lies).

use super::degrade_notices::{on_device, on_device_with};
use super::{buffer_with, pipelines_settled, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::bind::{BindState, SamplerDescriptor, UsedClass};
use metal2vulkan::device::VkExtension;
use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{
    BlendFactor, CompareFunction, DataType, DepthClipMode, PixelFormat, PrimitiveType,
    RenderStages, ResourceUsage, SamplerMinMagFilter, SamplerMipFilter, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBuffer, CommandBufferStatus, Device, LoadAction, PipelineOptions,
    Region, RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState, ResourceOptions,
    ResourceRef, SamplerState, ScissorRect, StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use std::collections::BTreeMap;

/// The target's side, in pixels: one column a draw.
const SIDE: u64 = 8;
/// The sampled textures' side.
const TEXTURE_SIDE: u64 = 4;
/// Red and green, and the target's clear, blue.
const RED: [u8; 4] = [255, 0, 0, 255];
const GREEN: [u8; 4] = [0, 255, 0, 255];
const CLEAR: [u8; 4] = [0, 0, 255, 255];

/// What a lane test draws with: the pipeline, a nearest sampler, and the target.
struct Scene {
    pipeline: RenderPipelineState,
    sampler: SamplerState,
    target: Texture,
}

/// A pipeline state of `tob_rect` and `tob_sample` into an `RGBA8Unorm` color attachment, which
/// `blends` (Metal's default factors, which store the fragment's color) or not.
fn pipeline(device: &Device, blends: bool) -> RenderPipelineState {
    pipeline_of(device, "tob_sample", blends)
}

/// A pipeline state of `tob_rect` and `tob_renders`'s fragment function `fragment` into an
/// `RGBA8Unorm` color attachment, which `blends` or not ([`pipeline`]).
fn pipeline_of(device: &Device, fragment: &str, blends: bool) -> RenderPipelineState {
    let ll = engine_dir().join("tob_renders.ll");
    let library = device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap();
    let function = |name| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("tob_rect"));
    descriptor.fragment = Some(function(fragment));
    let mut color = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
    color.blending_enabled = blends;
    descriptor.color_attachments[0] = Some(color);
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

impl Scene {
    fn new(device: &Device) -> Scene {
        let pipeline = pipeline(device, false);
        let sampler = device.new_sampler_state(&SamplerDescriptor::new()).unwrap();
        let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
        target.usage = TextureUsage::RENDER_TARGET;
        target.resource_options = ResourceOptions::shared();
        Scene {
            pipeline,
            sampler,
            target: device.new_texture(&target).unwrap(),
        }
    }

    /// A pass into the target, which `load`s it (a clear to [`CLEAR`]) and stores it.
    fn pass(&self, load: LoadAction) -> RenderPassDescriptor {
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(self.target.clone());
        color.load_action = load;
        color.store_action = StoreAction::Store;
        color.clear_color = ClearColor::new(0.0, 0.0, 1.0, 1.0);
        pass
    }

    /// A render encoder of a new command buffer of `device`'s that clears the target to
    /// [`CLEAR`], its pipeline and sampler set.
    fn encoder(&self, device: &Device) -> (CommandBuffer, RenderCommandEncoder) {
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let pass = self.pass(LoadAction::Clear);
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&self.pipeline).unwrap();
        encoder
            .set_fragment_sampler_state(Some(&self.sampler), 0)
            .unwrap();
        (command_buffer, encoder)
    }

    /// Each pixel of the target, by row, once `command_buffer` completed.
    fn pixels(&self, command_buffer: &CommandBuffer) -> Vec<[u8; 4]> {
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert_eq!(command_buffer.error(), None);
        let mut bytes = vec![0u8; (4 * SIDE * SIDE) as usize];
        self.target
            .get_bytes(
                &mut bytes,
                4 * SIDE,
                0,
                Region::new_2d(0, 0, SIDE, SIDE),
                0,
                0,
            )
            .unwrap();
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| [p[0], p[1], p[2], p[3]])
            .collect()
    }
}

/// A `TEXTURE_SIDE` square texture of `format`, every texel `bytes`.
fn solid(device: &Device, format: PixelFormat, bytes: [u8; 4]) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(format, TEXTURE_SIDE, TEXTURE_SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&descriptor).unwrap();
    let texels: Vec<u8> = (0..TEXTURE_SIDE * TEXTURE_SIDE)
        .flat_map(|_| bytes)
        .collect();
    let region = Region::new_2d(0, 0, TEXTURE_SIDE, TEXTURE_SIDE);
    texture
        .replace_region(region, 0, 0, &texels, 4 * TEXTURE_SIDE, 0)
        .unwrap();
    texture
}

/// The corners of the target's column `column`, as `tob_rect` reads them.
fn column(column: u64) -> Vec<u8> {
    let x = |c: u64| c as f32 * 2.0 / SIDE as f32 - 1.0;
    [x(column), -1.0, x(column + 1), 1.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect()
}

/// Draw the rectangle of the vertex buffer or bytes bound at index 0.
fn draw(encoder: &mut RenderCommandEncoder) {
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
}

/// `after`'s count of `count` less `before`'s.
fn grew(before: &WorkCounts, after: &WorkCounts, count: fn(&WorkCounts) -> u64) -> u64 {
    count(after) - count(before)
}

/// Draws of an unchanged state after the first make no bind shape, settle no variant and retain
/// no table: the first draw's work is the whole draw's. Each draw is recorded, its execution of
/// the first's shape, and the target is the texture's.
#[test]
fn draws_of_an_unchanged_state_settle_and_retain_it_once() {
    const DRAWS: usize = 16;
    on_device(|device, _| {
        let scene = Scene::new(device);
        let red = solid(device, PixelFormat::RGBA8Unorm, RED);
        let corners = buffer_with(
            device,
            &[-1.0f32, -1.0, 1.0, 1.0].map(f32::to_le_bytes).concat(),
        )
        .unwrap();
        let (command_buffer, mut encoder) = scene.encoder(device);
        encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
        encoder.set_fragment_texture(Some(&red), 0).unwrap();
        draw(&mut encoder);
        let first = device.work_counts();
        for _ in 1..DRAWS {
            draw(&mut encoder);
        }
        let rest = device.work_counts();
        encoder.end_encoding().unwrap();
        assert_eq!(grew(&first, &rest, |c| c.draw_shapes_made), 0);
        assert_eq!(grew(&first, &rest, |c| c.draws_settled), 0);
        assert_eq!(grew(&first, &rest, |c| c.argument_tables_retained), 0);
        let pixels = scene.pixels(&command_buffer);
        assert!(pixels.iter().all(|&p| p == RED), "{pixels:?}");
        let executions = command_buffer.executions();
        assert_eq!(executions.len(), DRAWS);
        assert!(executions
            .iter()
            .all(|e| e.shape() == executions[0].shape()));
    });
}

/// Draws that change between them only values no bind shape reads, one column each: the stencil
/// references, the blend color, and the depth/stencil state (of a depth test every fragment
/// passes, with and without depth writes). None past the first looks for a bind shape, makes one
/// or settles: each keeps its encoder's last draw's. Each column is the texture's.
#[test]
fn draws_that_change_only_values_no_shape_reads_keep_their_shape() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let red = solid(device, PixelFormat::RGBA8Unorm, RED);
        let ll = engine_dir().join("tob_renders.ll");
        let library = device
            .new_library_with_air(&llvm_assemble(&ll).unwrap())
            .unwrap();
        let function = |name| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let mut descriptor = RenderPipelineDescriptor::new(function("tob_rect"));
        descriptor.fragment = Some(function("tob_sample"));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        descriptor.depth_format = Some(PixelFormat::Depth32Float);
        let pipeline = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        let mut depth = TextureDescriptor::texture_2d(PixelFormat::Depth32Float, SIDE, SIDE, false);
        depth.usage = TextureUsage::RENDER_TARGET;
        depth.resource_options = ResourceOptions::private();
        let depth = device.new_texture(&depth).unwrap();
        let states = [false, true].map(|writes| {
            let mut state = DepthStencilDescriptor::default();
            state.depth_compare_function = CompareFunction::LessEqual;
            state.depth_write_enabled = writes;
            device.new_depth_stencil_state(&state).unwrap()
        });
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(scene.target.clone());
        color.load_action = LoadAction::Clear;
        color.store_action = StoreAction::Store;
        color.clear_color = ClearColor::new(0.0, 0.0, 1.0, 1.0);
        pass.depth_attachment.texture = Some(depth);
        pass.depth_attachment.load_action = LoadAction::Clear;
        pass.depth_attachment.store_action = StoreAction::DontCare;
        pass.depth_attachment.clear_depth = 1.0;
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&pipeline).unwrap();
        encoder
            .set_fragment_sampler_state(Some(&scene.sampler), 0)
            .unwrap();
        encoder.set_fragment_texture(Some(&red), 0).unwrap();
        let mut first = None;
        for c in 0..SIDE {
            encoder.set_vertex_bytes(&column(c), 0).unwrap();
            encoder.set_stencil_reference_value(c as u32);
            let level = c as f32 / SIDE as f32;
            encoder.set_blend_color(level, 1.0 - level, level, 1.0);
            encoder
                .set_depth_stencil_state(&states[(c % 2) as usize])
                .unwrap();
            draw(&mut encoder);
            first.get_or_insert_with(|| device.work_counts());
        }
        let (first, rest) = (first.unwrap(), device.work_counts());
        encoder.end_encoding().unwrap();
        assert_eq!(grew(&first, &rest, |c| c.draw_shapes_found), 0);
        assert_eq!(grew(&first, &rest, |c| c.draw_shapes_made), 0);
        assert_eq!(grew(&first, &rest, |c| c.draws_settled), 0);
        let pixels = scene.pixels(&command_buffer);
        assert!(pixels.iter().all(|&p| p == RED), "{pixels:?}");
    });
}

/// Draws of one encoder take the device's live sampler states to list only where they changed
/// since the listing their encoder's bind state keeps (`WorkCounts::argument_samplers_taken`):
/// none while no sampler state is made or goes; a state made between two draws, and that state
/// gone, each make the next draw take them once. Each column is the texture's.
#[test]
fn draws_take_the_device_s_sampler_states_only_where_they_changed() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let red = solid(device, PixelFormat::RGBA8Unorm, RED);
        let (command_buffer, mut encoder) = scene.encoder(device);
        encoder.set_fragment_texture(Some(&red), 0).unwrap();
        let taken = || device.work_counts().argument_samplers_taken;
        let mut counts = Vec::new();
        let mut made = None;
        for c in 0..SIDE {
            // A sampler state made before column 3's draw, gone before column 6's.
            if c == 3 {
                let mut descriptor = SamplerDescriptor::new();
                descriptor.lod_max_clamp = 7.0;
                made = Some(device.new_sampler_state(&descriptor).unwrap());
            }
            if c == 6 {
                made = None;
            }
            let before = taken();
            encoder.set_vertex_bytes(&column(c), 0).unwrap();
            draw(&mut encoder);
            counts.push(taken() - before);
        }
        assert!(made.is_none());
        encoder.end_encoding().unwrap();
        // The first draws settle the encoder's state from another's, and may take them.
        assert_eq!(counts[2..], [0, 1, 0, 0, 1, 0], "{counts:?}");
        let pixels = scene.pixels(&command_buffer);
        assert!(pixels.iter().all(|&p| p == RED), "{pixels:?}");
    });
}

/// Passes that each draw as the one before them, one column a draw, make no render pass past the
/// first's (`WorkCounts::render_passes_made`): their device keeps the compatible pass their draws
/// record against and the pass their instance begins, each by what it is made of. Each pass's
/// target is the texture's.
#[test]
fn a_pass_that_draws_as_the_one_before_makes_no_render_pass() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let red = solid(device, PixelFormat::RGBA8Unorm, RED);
        let made = || device.work_counts().render_passes_made;
        let mut counts = Vec::new();
        for _ in 0..3 {
            let before = made();
            let (command_buffer, mut encoder) = scene.encoder(device);
            encoder.set_fragment_texture(Some(&red), 0).unwrap();
            for c in 0..SIDE {
                encoder.set_vertex_bytes(&column(c), 0).unwrap();
                draw(&mut encoder);
            }
            encoder.end_encoding().unwrap();
            let pixels = scene.pixels(&command_buffer);
            assert!(pixels.iter().all(|&p| p == RED), "{pixels:?}");
            counts.push(made() - before);
        }
        assert_eq!(counts[1..], [0, 0], "{counts:?}");
    });
}

/// Draws each after a call of another setter form, its column's corners and its texel's color
/// set by the calls alone: a draw that reads its tables in part (`WorkCounts::
/// stage_tables_read_in_part`) brings in every slot such a call changed. The vertex slot's
/// corners come from a buffer of every column's at the column's offset (`set_vertex_buffer`,
/// `set_vertex_buffer_offset`, `set_vertex_buffers`) or from bytes (`set_vertex_bytes`); the color
/// from the texture bound (`set_fragment_texture`, `set_fragment_textures`) and from the level a
/// sampler's LOD clamps pick of a two-level texture, red then green (`set_fragment_sampler_state`,
/// `set_fragment_sampler_states` with clamps and without, the same state again). A slot a call
/// changed that the read left as it was would draw a column elsewhere, or in the other color.
///
/// The corners lie 16 bytes apart, and the vertex function reads the slot through a uniform view:
/// on a device whose view base alignment (`BindState::view_base_alignment_of`: NVIDIA's 64) does
/// not divide 16, the second column's bind leaves the slots that tell the view base apart
/// (`StageTable::at_bind_views`), which every buffer's shape reads, so that one draw reads its
/// vertex table whole instead of in part. The mask stays empty from then on, so no later draw does.
#[test]
fn a_draw_reading_its_tables_in_part_brings_in_each_setters_slot() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let corners: Vec<u8> = (0..SIDE).flat_map(column).collect();
        let corners = buffer_with(device, &corners).unwrap();
        let at = |c: u64| c * 16;
        let limits = device.compile_device().profile().limits();
        let alignment = u64::from(BindState::view_base_alignment_of(limits));
        let leaves_the_views = alignment != 0 && !at(1).is_multiple_of(alignment);
        eprintln!("view base alignment {alignment}: leaves the views {leaves_the_views}");
        // Two levels: red, then green.
        let mut levels = TextureDescriptor::texture_2d(
            PixelFormat::RGBA8Unorm,
            TEXTURE_SIDE,
            TEXTURE_SIDE,
            true,
        );
        levels.usage = TextureUsage::SHADER_READ;
        levels.resource_options = ResourceOptions::shared();
        let levels = device.new_texture(&levels).unwrap();
        for (level, color) in [(0, RED), (1, GREEN), (2, GREEN)] {
            let side = (TEXTURE_SIDE >> level).max(1);
            let texels: Vec<u8> = (0..side * side).flat_map(|_| color).collect();
            levels
                .replace_region(
                    Region::new_2d(0, 0, side, side),
                    level,
                    0,
                    &texels,
                    4 * side,
                    0,
                )
                .unwrap();
        }
        let green = solid(device, PixelFormat::RGBA8Unorm, GREEN);
        // Each samples one level: the target is twice the texture's side, so the draws' LOD is
        // below 0 and the clamps decide it.
        let sampler = |lod: f32| {
            let mut descriptor = SamplerDescriptor::new();
            descriptor.mip_filter = SamplerMipFilter::Nearest;
            descriptor.lod_min_clamp = lod;
            descriptor.lod_max_clamp = lod;
            device.new_sampler_state(&descriptor).unwrap()
        };
        let (first, second) = (sampler(0.0), sampler(1.0));
        let (command_buffer, mut encoder) = scene.encoder(device);
        encoder.set_vertex_buffer(Some(&corners), at(0), 0).unwrap();
        encoder.set_fragment_texture(Some(&levels), 0).unwrap();
        encoder.set_fragment_sampler_state(Some(&first), 0).unwrap();
        draw(&mut encoder);
        let before = device.work_counts();
        for c in 1..SIDE {
            match c {
                1 => {
                    encoder.set_vertex_buffer_offset(at(1), 0).unwrap();
                    encoder
                        .set_fragment_sampler_state(Some(&second), 0)
                        .unwrap();
                }
                2 => {
                    encoder.set_vertex_bytes(&column(2), 0).unwrap();
                    encoder.set_fragment_sampler_state(Some(&first), 0).unwrap();
                }
                3 => {
                    encoder
                        .set_vertex_buffers(&[Some(&corners)], &[at(3)], 0)
                        .unwrap();
                    encoder.set_fragment_textures(&[Some(&green)], 0).unwrap();
                }
                4 => {
                    encoder.set_vertex_buffer(Some(&corners), at(4), 0).unwrap();
                    encoder.set_fragment_texture(Some(&levels), 0).unwrap();
                }
                5 => {
                    encoder.set_vertex_buffer_offset(at(5), 0).unwrap();
                    let clamps: (&[f32], &[f32]) = (&[1.0], &[1.0]);
                    (encoder.set_fragment_sampler_states(&[Some(&first)], Some(clamps), 0))
                        .unwrap();
                }
                6 => {
                    encoder.set_vertex_buffer_offset(at(6), 0).unwrap();
                    (encoder.set_fragment_sampler_states(&[Some(&first)], None, 0)).unwrap();
                }
                _ => encoder.set_vertex_buffer_offset(at(c), 0).unwrap(),
            }
            draw(&mut encoder);
        }
        let after = device.work_counts();
        encoder.end_encoding().unwrap();
        let colors = [RED, GREEN, RED, GREEN, RED, GREEN, RED, RED];
        let pixels = scene.pixels(&command_buffer);
        for (i, &pixel) in pixels.iter().enumerate() {
            let want = colors[i % SIDE as usize];
            assert_eq!(pixel, want, "pixel {i}: {pixels:?}");
        }
        // Every later draw reads the vertex table in part; those that changed a fragment slot, the
        // fragment table too. The draw that leaves the slots telling the view base apart reads its
        // vertex table whole.
        let fragment_reads = 6;
        assert_eq!(
            grew(&before, &after, |c| c.stage_tables_read_in_part),
            (SIDE - 1) + fragment_reads - u64::from(leaves_the_views)
        );
    });
}

/// Draws read into their encoder's kept bind state only the stage tables a call changed since a
/// draw read them (`WorkCounts::stage_tables_read`): each column's first draw the vertex table,
/// whose bytes it sets, and the fragment table too where it binds another texture; its second
/// draw, with no call before it, none; and every table at the draw after another pipeline state
/// is set, whose interface they are read through. A table read through the same interface reads
/// only the slot its call changed (`WorkCounts::stage_tables_read_in_part`); the encoder's first
/// draw and the draw through another interface read theirs whole. Each second draw's uses are its first's, which
/// its encoder noted already (`WorkCounts::commands_noted_as_before`); so are each first draw's
/// the draw's before it, inline bytes being no use (they lie in the command buffer's upload
/// arena, which hazard tracking leaves alone), another pipeline state of the same stages too, but
/// at the encoder's first draw, which no draw of its noted, and where another texture is bound.
/// Each column is its texture's.
#[test]
fn draws_read_only_the_stage_tables_a_call_changed() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let blending = pipeline(device, true);
        let red = solid(device, PixelFormat::RGBA8Unorm, RED);
        let green = solid(device, PixelFormat::RGBA8Unorm, GREEN);
        let (command_buffer, mut encoder) = scene.encoder(device);
        encoder.set_fragment_texture(Some(&red), 0).unwrap();
        let read = || device.work_counts().stage_tables_read;
        let noted = || device.work_counts().commands_noted_as_before;
        let parted = || device.work_counts().stage_tables_read_in_part;
        let mut counts = Vec::new();
        let mut notes = Vec::new();
        let mut parts = Vec::new();
        for c in 0..SIDE {
            match c {
                3 => encoder.set_fragment_texture(Some(&green), 0).unwrap(),
                5 => encoder.set_render_pipeline_state(&blending).unwrap(),
                _ => {}
            }
            encoder.set_vertex_bytes(&column(c), 0).unwrap();
            let (before, noted_before, parted_before) = (read(), noted(), parted());
            draw(&mut encoder);
            let (first, noted_first) = (read() - before, noted() - noted_before);
            let parted_first = parted() - parted_before;
            draw(&mut encoder);
            counts.push((first, read() - before - first));
            notes.push((noted_first, noted() - noted_before - noted_first));
            parts.push((parted_first, parted() - parted_before - parted_first));
        }
        encoder.end_encoding().unwrap();
        let as_before = [
            (0, 1),
            (1, 1),
            (1, 1),
            (0, 1),
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 1),
        ];
        assert_eq!(notes, as_before, "{notes:?}");
        // The first draw reads the tables from another encoder's state: all four.
        let reads = [
            (4, 0),
            (1, 0),
            (1, 0),
            (2, 0),
            (1, 0),
            (4, 0),
            (1, 0),
            (1, 0),
        ];
        assert_eq!(counts, reads, "{counts:?}");
        let in_part = [
            (0, 0),
            (1, 0),
            (1, 0),
            (2, 0),
            (1, 0),
            (0, 0),
            (1, 0),
            (1, 0),
        ];
        assert_eq!(parts, in_part, "{parts:?}");
        let pixels = scene.pixels(&command_buffer);
        for (i, &pixel) in pixels.iter().enumerate() {
            let want = match i as u64 % SIDE >= 3 {
                true => GREEN,
                false => RED,
            };
            assert_eq!(pixel, want, "pixel {i}: {pixels:?}");
        }
    });
}

/// Draws that each bind another texture of the same type, format and usage, and other inline
/// bytes of the same length, one column each: none past the first makes a bind shape (its
/// texture slot keeps its shape) or settles, and only the second retains the fragment table,
/// taking hold of its texture: each later draw's texture is held from an earlier one. Each column
/// is its own draw's texture's.
#[test]
fn draws_of_other_objects_of_one_shape_make_and_settle_nothing() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let textures = [
            solid(device, PixelFormat::RGBA8Unorm, RED),
            solid(device, PixelFormat::RGBA8Unorm, GREEN),
        ];
        let (command_buffer, mut encoder) = scene.encoder(device);
        let mut first = None;
        for c in 0..SIDE {
            encoder.set_vertex_bytes(&column(c), 0).unwrap();
            encoder
                .set_fragment_texture(Some(&textures[(c % 2) as usize]), 0)
                .unwrap();
            draw(&mut encoder);
            first.get_or_insert_with(|| device.work_counts());
        }
        let (first, rest) = (first.unwrap(), device.work_counts());
        encoder.end_encoding().unwrap();
        assert_eq!(grew(&first, &rest, |c| c.draw_shapes_made), 0);
        assert_eq!(grew(&first, &rest, |c| c.draws_settled), 0);
        assert_eq!(grew(&first, &rest, |c| c.argument_tables_retained), 1);
        assert_eq!(grew(&first, &rest, |c| c.argument_objects_retained), 1);
        let pixels = scene.pixels(&command_buffer);
        for (at, &pixel) in pixels.iter().enumerate() {
            let want = [RED, GREEN][at % 2];
            assert_eq!(pixel, want, "pixel {at}: {pixels:?}");
        }
    });
}

/// Draws that bind textures of two formats in turn make and settle each bind shape once: the state
/// keeps the settles of its latest shapes, among which a draw finds its shape. Before, each draw
/// made its shape, and before that settled it again. Each column is its own draw's texture's.
#[test]
fn draws_of_another_shape_settle_again() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        // Green either way: `BGRA8Unorm`'s bytes are blue, green, red, alpha.
        let textures = [
            solid(device, PixelFormat::RGBA8Unorm, GREEN),
            solid(device, PixelFormat::BGRA8Unorm, GREEN),
        ];
        let (command_buffer, mut encoder) = scene.encoder(device);
        let before = device.work_counts();
        for c in 0..SIDE {
            encoder.set_vertex_bytes(&column(c), 0).unwrap();
            encoder
                .set_fragment_texture(Some(&textures[(c % 2) as usize]), 0)
                .unwrap();
            draw(&mut encoder);
        }
        let after = device.work_counts();
        encoder.end_encoding().unwrap();
        assert_eq!(grew(&before, &after, |c| c.draw_shapes_made), 2);
        assert_eq!(grew(&before, &after, |c| c.draws_settled), 2);
        let pixels = scene.pixels(&command_buffer);
        assert!(pixels.iter().all(|&p| p == GREEN), "{pixels:?}");
        let executions = command_buffer.executions();
        assert_eq!(executions.len(), SIDE as usize);
        assert_ne!(executions[0].shape(), executions[1].shape());
        assert_eq!(executions[0].shape(), executions[2].shape());
    });
}

/// A pass that draws as the one before it, each of its encoder's draws of a bind state a draw of
/// the pass before had, makes no bind shape and settles nothing: its first draw, whose encoder
/// keeps no shape, and its draw of the other shape find theirs among those the state settled.
/// Before, each pass's encoder made its shapes again. Every column is drawn.
#[test]
fn a_pass_that_draws_as_the_one_before_makes_no_shape() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let textures = [
            solid(device, PixelFormat::RGBA8Unorm, GREEN),
            solid(device, PixelFormat::BGRA8Unorm, GREEN),
        ];
        let pass = || {
            let (command_buffer, mut encoder) = scene.encoder(device);
            for c in 0..SIDE {
                encoder.set_vertex_bytes(&column(c), 0).unwrap();
                encoder
                    .set_fragment_texture(Some(&textures[(c % 2) as usize]), 0)
                    .unwrap();
                draw(&mut encoder);
            }
            encoder.end_encoding().unwrap();
            command_buffer
        };
        let first = pass();
        let pixels = scene.pixels(&first);
        assert!(pixels.iter().all(|&p| p == GREEN), "{pixels:?}");
        let before = device.work_counts();
        let second = pass();
        let after = device.work_counts();
        assert_eq!(grew(&before, &after, |c| c.draw_shapes_made), 0);
        assert_eq!(grew(&before, &after, |c| c.draws_settled), 0);
        let pixels = scene.pixels(&second);
        assert!(pixels.iter().all(|&p| p == GREEN), "{pixels:?}");
        let (first, second) = (first.executions(), second.executions());
        assert_eq!(second.len(), SIDE as usize);
        assert_eq!(first[0].shape(), second[0].shape());
        assert_eq!(first[1].shape(), second[1].shape());
    });
}

/// A pass that binds the two textures the pass before it bound works out neither's view: its
/// encoder takes the views the last one's binds found (`WorkCounts::bind_views_worked_out`).
/// One of them let go of and a texture of another color made, a third pass works out the new
/// texture's view alone, and draws its color. Every column is drawn. The textures are
/// `RGBA8Unorm`, as the target, so each texel's bytes are the pixel its column stores.
#[test]
fn a_pass_that_binds_as_the_one_before_works_out_no_view() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let pass = |textures: &[&Texture; 2]| {
            let before = device.work_counts();
            let (command_buffer, mut encoder) = scene.encoder(device);
            for c in 0..SIDE {
                encoder.set_vertex_bytes(&column(c), 0).unwrap();
                encoder
                    .set_fragment_texture(Some(textures[(c % 2) as usize]), 0)
                    .unwrap();
                draw(&mut encoder);
            }
            encoder.end_encoding().unwrap();
            let after = device.work_counts();
            let worked_out = grew(&before, &after, |c| c.bind_views_worked_out);
            (worked_out, scene.pixels(&command_buffer))
        };
        let green = solid(device, PixelFormat::RGBA8Unorm, GREEN);
        let other = solid(device, PixelFormat::RGBA8Unorm, GREEN);
        let (first, pixels) = pass(&[&green, &other]);
        assert_eq!(first, 2);
        assert!(pixels.iter().all(|&p| p == GREEN), "{pixels:?}");
        let (second, pixels) = pass(&[&green, &other]);
        assert_eq!(second, 0);
        assert!(pixels.iter().all(|&p| p == GREEN), "{pixels:?}");
        drop(other);
        let red = solid(device, PixelFormat::RGBA8Unorm, RED);
        let (third, pixels) = pass(&[&green, &red]);
        assert_eq!(third, 1);
        for (at, &pixel) in pixels.iter().enumerate() {
            let want = if at.is_multiple_of(2) { GREEN } else { RED };
            assert_eq!(pixel, want, "pixel {at}: {pixels:?}");
        }
    });
}

/// Draws whose vertex buffer slot 0 binds a buffer and inline bytes in turn, a binding of another
/// class at the slot each draw, a slot's class being of its shape, make and settle each shape
/// once (the state keeps the settles of its latest shapes, among which a draw finds its shape).
/// The buffer is bound at 0, 32, 64 and 96, where `tob_rect` reads it through a uniform view: on
/// a device whose view base alignment (`BindState::view_base_alignment_of`: ANV's 64) does not
/// divide 32, the slot leaves the slots that tell it apart at the second, a shape of its own,
/// and reads at its residual from then on, so its draws make and settle three shapes there.
/// Every column is drawn, from the buffer's corners or the bytes'.
#[test]
fn draws_that_change_a_slot_s_class_make_their_shapes_again() {
    on_device(|device, _| {
        let limits = device.compile_device().profile().limits();
        let alignment = BindState::view_base_alignment_of(limits);
        let shapes = if alignment != 0 && 32 % alignment != 0 {
            3
        } else {
            2
        };
        eprintln!("view base alignment {alignment}: {shapes} shapes");
        let scene = Scene::new(device);
        let red = solid(device, PixelFormat::RGBA8Unorm, RED);
        let corners: Vec<u8> = (0..SIDE).flat_map(column).collect();
        let corners = buffer_with(device, &corners).unwrap();
        let (command_buffer, mut encoder) = scene.encoder(device);
        encoder.set_fragment_texture(Some(&red), 0).unwrap();
        let before = device.work_counts();
        for c in 0..SIDE {
            match c % 2 {
                0 => encoder
                    .set_vertex_buffer(Some(&corners), 16 * c, 0)
                    .unwrap(),
                _ => encoder.set_vertex_bytes(&column(c), 0).unwrap(),
            }
            draw(&mut encoder);
        }
        let after = device.work_counts();
        encoder.end_encoding().unwrap();
        assert_eq!(grew(&before, &after, |c| c.draw_shapes_made), shapes);
        assert_eq!(grew(&before, &after, |c| c.draws_settled), shapes);
        let pixels = scene.pixels(&command_buffer);
        assert!(pixels.iter().all(|&p| p == RED), "{pixels:?}");
        let executions = command_buffer.executions();
        assert_eq!(executions.len(), SIDE as usize);
        assert_ne!(executions[0].shape(), executions[1].shape());
    });
}

/// Draws that switch between two pipeline states at every draw, of the same functions, look
/// neither's `VkPipeline` up by its key once each state's draw found it made: each state's later
/// draws find that entry again (`WorkCounts::graphics_pipelines_looked_up`), and neither makes a
/// shape or settles. A first command buffer, completed, and the pipeline makings its draws began
/// on the device's pool, settled, have both pipelines made. Every column is drawn.
#[test]
fn draws_switching_pipeline_states_find_their_pipelines_again() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let pipelines = [scene.pipeline.clone(), pipeline(device, true)];
        let red = solid(device, PixelFormat::RGBA8Unorm, RED);
        let (made, mut encoder) = scene.encoder(device);
        encoder.set_fragment_texture(Some(&red), 0).unwrap();
        for pipeline in &pipelines {
            encoder.set_render_pipeline_state(pipeline).unwrap();
            encoder.set_vertex_bytes(&column(0), 0).unwrap();
            draw(&mut encoder);
        }
        encoder.end_encoding().unwrap();
        scene.pixels(&made);
        pipelines_settled(device);
        let (command_buffer, mut encoder) = scene.encoder(device);
        encoder.set_fragment_texture(Some(&red), 0).unwrap();
        let mut first = None;
        for c in 0..SIDE {
            encoder
                .set_render_pipeline_state(&pipelines[(c % 2) as usize])
                .unwrap();
            encoder.set_vertex_bytes(&column(c), 0).unwrap();
            draw(&mut encoder);
            if c == 1 {
                first = Some(device.work_counts());
            }
        }
        let (first, rest) = (first.unwrap(), device.work_counts());
        encoder.end_encoding().unwrap();
        assert_eq!(grew(&first, &rest, |c| c.graphics_pipelines_looked_up), 0);
        assert_eq!(grew(&first, &rest, |c| c.draw_shapes_made), 0);
        assert_eq!(grew(&first, &rest, |c| c.draws_settled), 0);
        let pixels = scene.pixels(&command_buffer);
        assert!(pixels.iter().all(|&p| p == RED), "{pixels:?}");
    });
}

/// Draws that switch at every draw between pipeline states whose bind states differ (`tob_sample`,
/// and `tob_solid`, which reads its color through a uniform view of fragment buffer 0), each
/// binding what it bound before but its column's bytes, find each one's shape kept past the first
/// draw of each: the encoder sets aside the state of the pipeline state it drew before
/// (`LiveBind::select`), so none looks its shape up among those settled
/// (`WorkCounts::draw_shapes_found`), makes one or settles. Before, each draw past the first two
/// looked its shape up, the state its encoder kept having last been read through the other
/// pipeline state's interface. Each column is its own pipeline state's color.
#[test]
fn draws_switching_between_pipeline_states_of_other_states_keep_each_shape() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let pipelines = [
            scene.pipeline.clone(),
            pipeline_of(device, "tob_solid", false),
        ];
        let red = solid(device, PixelFormat::RGBA8Unorm, RED);
        let green: Vec<u8> = [0.0f32, 1.0, 0.0, 1.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let (command_buffer, mut encoder) = scene.encoder(device);
        encoder.set_fragment_texture(Some(&red), 0).unwrap();
        encoder.set_fragment_bytes(&green, 0).unwrap();
        let mut first = None;
        for c in 0..SIDE {
            encoder
                .set_render_pipeline_state(&pipelines[(c % 2) as usize])
                .unwrap();
            encoder.set_vertex_bytes(&column(c), 0).unwrap();
            draw(&mut encoder);
            if c == 1 {
                first = Some(device.work_counts());
            }
        }
        let (first, rest) = (first.unwrap(), device.work_counts());
        encoder.end_encoding().unwrap();
        assert_eq!(grew(&first, &rest, |c| c.draw_shapes_found), 0);
        assert_eq!(grew(&first, &rest, |c| c.draw_shapes_made), 0);
        assert_eq!(grew(&first, &rest, |c| c.draws_settled), 0);
        let pixels = scene.pixels(&command_buffer);
        for (at, &pixel) in pixels.iter().enumerate() {
            let want = [RED, GREEN][at % 2];
            assert_eq!(pixel, want, "pixel {at}: {pixels:?}");
        }
    });
}

/// Draws of one pipeline state that switch between two keys of its `VkPipeline` at every draw
/// (its depth clip mode, which no pipeline makes dynamic) look neither up by its key once each
/// key's draw found it made: the state keeps the pipelines its last draws found
/// (`WorkCounts::graphics_pipelines_looked_up`). A first command buffer, completed, and the
/// pipeline makings its draws began on the device's pool, settled, have both made. Every column is
/// drawn. Not run on a device without `depthClamp`, which refuses the clamp.
#[test]
fn draws_of_one_state_switching_keys_find_their_pipelines_again() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let red = solid(device, PixelFormat::RGBA8Unorm, RED);
        let modes = [DepthClipMode::Clip, DepthClipMode::Clamp];
        let (made, mut encoder) = scene.encoder(device);
        if encoder.set_depth_clip_mode(DepthClipMode::Clamp).is_err() {
            return;
        }
        encoder.set_fragment_texture(Some(&red), 0).unwrap();
        for mode in modes {
            encoder.set_depth_clip_mode(mode).unwrap();
            encoder.set_vertex_bytes(&column(0), 0).unwrap();
            draw(&mut encoder);
        }
        encoder.end_encoding().unwrap();
        scene.pixels(&made);
        pipelines_settled(device);
        let (command_buffer, mut encoder) = scene.encoder(device);
        encoder.set_fragment_texture(Some(&red), 0).unwrap();
        let mut first = None;
        for c in 0..SIDE {
            encoder
                .set_depth_clip_mode(modes[(c % 2) as usize])
                .unwrap();
            encoder.set_vertex_bytes(&column(c), 0).unwrap();
            draw(&mut encoder);
            if c == 1 {
                first = Some(device.work_counts());
            }
        }
        let (first, rest) = (first.unwrap(), device.work_counts());
        encoder.end_encoding().unwrap();
        assert_eq!(grew(&first, &rest, |c| c.graphics_pipelines_looked_up), 0);
        let pixels = scene.pixels(&command_buffer);
        assert!(pixels.iter().all(|&p| p == RED), "{pixels:?}");
    });
}

/// The most allocations a draw into a pass of a color its pipeline state lacks makes beyond a
/// draw of the same state into a pass of its own colors alone, both of a state unchanged but for
/// its column's bytes ([`draws_into_a_pass_color_their_state_lacks_copy_no_color_set`]), on
/// average: none. The colors the draw zeroes are those the encoder kept as the pipeline state was
/// set, which its settings read in place (`RenderCommandEncoder::draw_colors`), and its bind
/// state compares in place (`Reshape::field_from`), copying them only where they changed.
/// Measured: -0.20 on ANV (UHD 770); 0.80 where the settings copied them, 1.80 where the bind
/// state copied them too.
#[cfg(not(m2v_system_alloc))]
const ZEROED_COLORS_EXTRA: f64 = 0.3;
/// [`ZEROED_COLORS_EXTRA`] in a build with debug assertions, where each draw's checks build its
/// state's whole bind shape (`LiveBind::shape`), the colors it zeroes included. Measured: 0.80
/// on lavapipe; 1.80 where the settings copied them, on lavapipe and on ANV (UHD 770).
#[cfg(not(m2v_system_alloc))]
const ZEROED_COLORS_EXTRA_DEBUG: f64 = 1.3;

/// Draws into a pass with a color their pipeline state lacks, which plain Metal zeroes where a
/// draw covers it (`BindState::zeroed_colors`), allocate at most [`ZEROED_COLORS_EXTRA`] each
/// beyond the same draws into a pass of the state's colors alone, past a first command buffer that
/// made the pipelines: an unchanged set of colors to zero is read in place, and copied neither
/// into the draw's settings nor into its bind state again. Each draw binds what the draw before it bound but its column's
/// bytes; the target is the texture's, and the lacked color zero where the draws covered it.
#[cfg(not(m2v_system_alloc))]
#[test]
fn draws_into_a_pass_color_their_state_lacks_copy_no_color_set() {
    use crate::worker::alloc::thread_allocations;
    // Enough draws that what grows by doubling (the command buffer's lists) adds little to each.
    const DRAWS: u64 = 64;
    on_device(|device, _| {
        let scene = Scene::new(device);
        let red = solid(device, PixelFormat::RGBA8Unorm, RED);
        let mut lacked = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
        lacked.usage = TextureUsage::RENDER_TARGET;
        lacked.resource_options = ResourceOptions::shared();
        let lacked = device.new_texture(&lacked).unwrap();
        // The allocations each draw past the first two makes, of a pass of the target and, where
        // given, a second color cleared to `CLEAR`; and the command buffer.
        let draws = |second: Option<&Texture>| {
            let queue = device.new_command_queue().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut pass = RenderPassDescriptor::new();
            for (at, texture) in [Some(&scene.target), second].into_iter().enumerate() {
                let color = &mut pass.color_attachments[at];
                color.texture = texture.cloned();
                color.load_action = LoadAction::Clear;
                color.store_action = StoreAction::Store;
                color.clear_color = ClearColor::new(0.0, 0.0, 1.0, 1.0);
            }
            let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
            encoder.set_render_pipeline_state(&scene.pipeline).unwrap();
            encoder
                .set_fragment_sampler_state(Some(&scene.sampler), 0)
                .unwrap();
            encoder.set_fragment_texture(Some(&red), 0).unwrap();
            let mut before = None;
            for at in 0..DRAWS {
                encoder.set_vertex_bytes(&column(at % SIDE), 0).unwrap();
                draw(&mut encoder);
                if at == 1 {
                    before = thread_allocations();
                }
            }
            let each = thread_allocations()
                .zip(before)
                .map(|(after, before)| (after - before) as f64 / (DRAWS - 2) as f64);
            encoder.end_encoding().unwrap();
            (each, command_buffer)
        };
        // Every pipeline made first, in a command buffer of each pass.
        for second in [None, Some(&lacked)] {
            let (_, command_buffer) = draws(second);
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
        }
        pipelines_settled(device);
        let (own, command_buffer) = draws(None);
        assert!(scene.pixels(&command_buffer).iter().all(|&p| p == RED));
        let (beside, command_buffer) = draws(Some(&lacked));
        let pixels = scene.pixels(&command_buffer);
        assert!(pixels.iter().all(|&p| p == RED), "{pixels:?}");
        let mut bytes = vec![0u8; (4 * SIDE * SIDE) as usize];
        let region = Region::new_2d(0, 0, SIDE, SIDE);
        lacked
            .get_bytes(&mut bytes, 4 * SIDE, 0, region, 0, 0)
            .unwrap();
        assert!(bytes.iter().all(|&b| b == 0), "{bytes:?}");
        let (Some(own), Some(beside)) = (own, beside) else {
            eprintln!("DRAW-WORK {}: allocations not counted", device.name());
            return;
        };
        eprintln!(
            "DRAW-WORK {}: {beside:.2} allocations a draw beside a lacked color, {own:.2} without",
            device.name()
        );
        let bound = match cfg!(debug_assertions) {
            true => ZEROED_COLORS_EXTRA_DEBUG,
            false => ZEROED_COLORS_EXTRA,
        };
        assert!(
            beside - own <= bound,
            "{beside:.2} allocations a draw beside a lacked color, {own:.2} without"
        );
    });
}

/// Draws whose encoder reads a table binding a no-copy buffer (the corners, over host pages, at
/// vertex buffer 0) again at each draw, for the bytes each sets at vertex buffer 1, look at where
/// the buffer lies at most twice each (`WorkCounts::no_copy_lies_read`): once for its binding,
/// its base and device address read together (`Buffer::base_and_device_address`), and once for a
/// descriptor written for it (`Buffer::space`), where each read took two looks. A build with
/// debug assertions looks once more a draw, as its check of the table brought in place reads the
/// slot whole again (`ArgumentTable::check_reading`). The target is the texture's.
#[test]
fn draws_reading_a_no_copy_buffer_s_binding_look_where_it_lies_once() {
    use super::host_memory::{host_page, no_copy, Mapping};
    on_device(|device, _| {
        let scene = Scene::new(device);
        let red = solid(device, PixelFormat::RGBA8Unorm, RED);
        let page = host_page();
        let pages = Mapping::anonymous(page);
        pages.write(0, &[-1.0f32, -1.0, 1.0, 1.0].map(f32::to_le_bytes).concat());
        // SAFETY: the pages are this process's, mapped read and write, and outlive the buffer,
        // which goes first.
        let corners = unsafe {
            no_copy(
                device,
                pages.pointer(),
                page,
                ResourceOptions::shared(),
                None,
            )
        }
        .unwrap();
        let (command_buffer, mut encoder) = scene.encoder(device);
        encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
        encoder.set_fragment_texture(Some(&red), 0).unwrap();
        let mut first = None;
        for at in 0..SIDE {
            encoder.set_vertex_bytes(&at.to_le_bytes(), 1).unwrap();
            draw(&mut encoder);
            if at == 0 {
                first = Some(device.work_counts());
            }
        }
        let (first, rest) = (first.unwrap(), device.work_counts());
        encoder.end_encoding().unwrap();
        let later = SIDE - 1;
        eprintln!(
            "DRAW-WORK {}: {} looks at where the buffer lies, {} tables read",
            device.name(),
            grew(&first, &rest, |c| c.no_copy_lies_read),
            grew(&first, &rest, |c| c.stage_tables_read),
        );
        assert_eq!(grew(&first, &rest, |c| c.stage_tables_read), later);
        let looks = 2 + u64::from(cfg!(debug_assertions));
        assert!(grew(&first, &rest, |c| c.no_copy_lies_read) <= looks * later);
        let pixels = scene.pixels(&command_buffer);
        assert!(pixels.iter().all(|&p| p == RED), "{pixels:?}");
    });
}

/// Whether `device` has unified memory, or host-coherent, host-cached memory none of whose types
/// is device-local: its upload arenas then lie outside device-local memory.
fn uploads_outside_device_memory(device: &Device) -> bool {
    use ash::vk::MemoryPropertyFlags as P;
    if device.properties().has_unified_memory {
        return true;
    }
    // SAFETY: the handles are used only while the device lives.
    let raw = unsafe { device.raw() };
    // SAFETY: a physical device of the instance.
    let properties =
        unsafe { (raw.instance).get_physical_device_memory_properties(raw.physical_device) };
    let types = &properties.memory_types[..properties.memory_type_count as usize];
    let cached = || {
        (types.iter())
            .map(|ty| ty.property_flags)
            .filter(|flags| flags.contains(P::HOST_VISIBLE | P::HOST_COHERENT | P::HOST_CACHED))
    };
    cached().next().is_some() && cached().all(|flags| !flags.contains(P::DEVICE_LOCAL))
}

/// Draws that each set their bytes, one column each, write them into the command buffer's upload
/// arena in no device-local memory where the device has host-coherent, host-cached system memory
/// ([`uploads_outside_device_memory`]): a write into device-local memory waits for a GPU powered
/// down to wake. Every column is drawn.
#[test]
fn draws_setting_bytes_write_no_device_local_memory() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let red = solid(device, PixelFormat::RGBA8Unorm, RED);
        let before = device.work_counts();
        let (command_buffer, mut encoder) = scene.encoder(device);
        encoder.set_fragment_texture(Some(&red), 0).unwrap();
        for c in 0..SIDE {
            encoder.set_vertex_bytes(&column(c), 0).unwrap();
            draw(&mut encoder);
        }
        encoder.end_encoding().unwrap();
        let pixels = scene.pixels(&command_buffer);
        let opened = grew(&before, &device.work_counts(), |c| {
            c.upload_chunks_in_device_memory
        });
        println!("runtime_lane: upload chunks in device-local memory: {opened}");
        if uploads_outside_device_memory(device) {
            assert_eq!(opened, 0);
        }
        assert!(pixels.iter().all(|&p| p == RED), "{pixels:?}");
    });
}

/// Draw through `encoder` (`draw`), returning the state commands it recorded
/// (`WorkCounts::draw_states_recorded`).
fn recorded_by(device: &Device, encoder: &mut RenderCommandEncoder) -> u64 {
    let before = device.work_counts();
    draw(encoder);
    grew(&before, &device.work_counts(), |c| c.draw_states_recorded)
}

/// Draw once through each of `pipelines` in a command buffer of its own, completed, and wait
/// until the device's pool settled ([`pipelines_settled`]), so each draw's own key is made: a
/// draw that took its key's residual form began that key's making on the pool, which its command
/// buffer does not wait for. A later draw then takes its key's pipeline from its first draw on,
/// and records as it draws, never switching pipelines once a making lands mid-test.
fn make(device: &Device, scene: &Scene, pipelines: &[RenderPipelineState], texture: &Texture) {
    let (command_buffer, mut encoder) = scene.encoder(device);
    encoder.set_fragment_texture(Some(texture), 0).unwrap();
    for pipeline in pipelines {
        encoder.set_render_pipeline_state(pipeline).unwrap();
        encoder.set_vertex_bytes(&column(0), 0).unwrap();
        draw(&mut encoder);
    }
    encoder.end_encoding().unwrap();
    scene.pixels(&command_buffer);
    pipelines_settled(device);
}

/// Whether `device` pushes set 0: Vulkan 1.4 or `VK_KHR_push_descriptor` (the instance's Vulkan
/// 1.2 leaves `VK_KHR_push_descriptor` alone), whose `maxPushDescriptors` the lanes' set 0 fits,
/// on a driver whose pushed sets keep their buffers' sizes
/// (`RuntimeDriverDefect::PushedBufferSizesUnset`, MoltenVK's: set 0 is bound there).
pub(super) fn pushes(device: &Device) -> bool {
    let unset = metal2vulkan::error::RuntimeDriverDefect::PushedBufferSizesUnset;
    let profile = device.compile_device().profile();
    profile.descriptors().push_descriptor && !device.properties().driver_defects.contains(&unset)
}

/// Run `test` on the device twice: as it is, and with `VK_KHR_push_descriptor` withheld, so a
/// device of the instance's Vulkan 1.2 has no push command and binds set 0 as a device without
/// one does.
pub(super) fn on_device_both_ways(test: impl Fn(&Device)) {
    on_device(|device, _| test(device));
    on_device_with(
        |options| {
            options
                .withheld_extensions
                .insert(VkExtension::KhrPushDescriptor);
        },
        |device, _| {
            assert!(!pushes(device), "withheld, the device pushes");
            test(device);
        },
    );
}

/// Draws of an unchanged state into one render pass instance record its state commands once:
/// past the first, a draw records nothing, so no pipeline, descriptor set, push constant, buffer
/// or dynamic state is set again: where the device binds set 0, each draw past the first binds
/// the set its last draw wrote, the same descriptors (`WorkCounts::set_zero_sets_reused`), and
/// where it pushes set 0 ([`pushes`]), every draw pushes it, its writes the same. The first draw records them all, and the target is the
/// texture's. Every draw takes its key's own pipeline, made beforehand ([`make`]), none the
/// residual form, so no pipeline is bound again past the first draw. Both ways
/// ([`on_device_both_ways`]).
#[test]
fn draws_of_an_unchanged_state_record_its_commands_once() {
    const DRAWS: usize = 16;
    on_device_both_ways(|device| {
        let scene = Scene::new(device);
        let red = solid(device, PixelFormat::RGBA8Unorm, RED);
        make(device, &scene, std::slice::from_ref(&scene.pipeline), &red);
        let corners = [-1.0f32, -1.0, 1.0, 1.0].map(f32::to_le_bytes).concat();
        let (command_buffer, mut encoder) = scene.encoder(device);
        encoder.set_vertex_bytes(&corners, 0).unwrap();
        encoder.set_fragment_texture(Some(&red), 0).unwrap();
        let before = device.work_counts();
        let first = recorded_by(device, &mut encoder);
        let later: Vec<u64> = (1..DRAWS)
            .map(|_| recorded_by(device, &mut encoder))
            .collect();
        let after = device.work_counts();
        let pushed = grew(&before, &after, |c| c.draws_pushing_set_zero);
        let reused = grew(&before, &after, |c| c.set_zero_sets_reused);
        encoder.end_encoding().unwrap();
        let residual = grew(&before, &after, |c| c.residual_form_draws);
        assert_eq!(
            residual, 0,
            "a draw took its key's residual form: its key was not made"
        );
        let pushes = pushes(device);
        assert_eq!(pushed, if pushes { DRAWS as u64 } else { 0 }, "{pushes}");
        let sets = if pushes { 0 } else { DRAWS as u64 - 1 };
        assert_eq!(reused, sets, "{pushes}");
        assert!(first > 1, "the first draw recorded {first}");
        assert!(later.iter().all(|&n| n == 0), "{first}, then {later:?}");
        let pixels = scene.pixels(&command_buffer);
        assert!(pixels.iter().all(|&p| p == RED), "{pixels:?}");
    });
}

/// A draw binds the set 0 its encoder's last draw allocated and wrote only where it writes the
/// same descriptors into a set of the same layout (`WorkCounts::set_zero_sets_reused`): a draw
/// again of an unchanged state takes it; a draw after a change of an object set 0 names does not:
/// another texture, the texture an earlier draw but not the last bound, a sampler state of
/// another descriptor, a buffer in place of bytes; and a second encoder's first draw of the first
/// encoder's last state does not, the memo being its encoder's (such a draw may bind a set its
/// command buffer keeps, `WorkCounts::set_zero_sets_found`). Whether other inline bytes
/// or another offset of a buffer change the set's writes is the variant's (either may be push
/// data); each column holds what its last draw drew all the same, so a set taken again held
/// what its draw reads. Where the device pushes set 0, none binds one. Both ways
/// ([`on_device_both_ways`]).
#[test]
fn a_draw_binds_the_last_set_zero_only_where_it_writes_the_same() {
    on_device_both_ways(|device| {
        let scene = Scene::new(device);
        let red = solid(device, PixelFormat::RGBA8Unorm, RED);
        let green = solid(device, PixelFormat::RGBA8Unorm, GREEN);
        let mut linear = SamplerDescriptor::new();
        linear.min_filter = SamplerMinMagFilter::Linear;
        linear.mag_filter = SamplerMinMagFilter::Linear;
        let linear = device.new_sampler_state(&linear).unwrap();
        // Each column's corners at 256 bytes of the next, a constant buffer's offset alignment.
        let columns: Vec<u8> = (0..SIDE)
            .flat_map(|c| {
                let mut corners = column(c);
                corners.resize(256, 0);
                corners
            })
            .collect();
        let columns = buffer_with(device, &columns).unwrap();
        let (command_buffer, mut encoder) = scene.encoder(device);
        let mut reused = Vec::new();
        let mut drawn = |encoder: &mut RenderCommandEncoder| {
            let before = device.work_counts();
            draw(encoder);
            reused.push(grew(&before, &device.work_counts(), |c| {
                c.set_zero_sets_reused
            }));
        };
        encoder.set_fragment_texture(Some(&red), 0).unwrap();
        encoder.set_vertex_bytes(&column(0), 0).unwrap();
        drawn(&mut encoder);
        // The same again.
        drawn(&mut encoder);
        encoder.set_vertex_bytes(&column(1), 0).unwrap();
        drawn(&mut encoder);
        encoder.set_fragment_texture(Some(&green), 0).unwrap();
        drawn(&mut encoder);
        encoder.set_vertex_bytes(&column(2), 0).unwrap();
        drawn(&mut encoder);
        encoder.set_fragment_texture(Some(&red), 0).unwrap();
        drawn(&mut encoder);
        encoder
            .set_fragment_sampler_state(Some(&linear), 0)
            .unwrap();
        drawn(&mut encoder);
        encoder
            .set_vertex_buffer(Some(&columns), 256 * 3, 0)
            .unwrap();
        drawn(&mut encoder);
        encoder.set_vertex_buffer_offset(256 * 4, 0).unwrap();
        drawn(&mut encoder);
        encoder.end_encoding().unwrap();
        // The last state again, in another encoder of the command buffer.
        let mut encoder = command_buffer
            .render_command_encoder(&scene.pass(LoadAction::Load))
            .unwrap();
        encoder.set_render_pipeline_state(&scene.pipeline).unwrap();
        encoder
            .set_fragment_sampler_state(Some(&linear), 0)
            .unwrap();
        encoder.set_fragment_texture(Some(&red), 0).unwrap();
        encoder
            .set_vertex_buffer(Some(&columns), 256 * 4, 0)
            .unwrap();
        drawn(&mut encoder);
        encoder.set_vertex_buffer_offset(256 * 5, 0).unwrap();
        drawn(&mut encoder);
        encoder.end_encoding().unwrap();
        // `None` where the variant decides (the bytes and offsets alone changed).
        let binds = !pushes(device);
        let want = [
            Some(0),
            Some(u64::from(binds)),
            None,
            Some(0),
            None,
            Some(0),
            Some(0),
            Some(0),
            None,
            Some(0),
            None,
        ];
        assert_eq!(reused.len(), want.len());
        for (at, (&reused, want)) in reused.iter().zip(want).enumerate() {
            match want {
                Some(want) => assert_eq!(reused, want, "draw {at}"),
                None => assert!(reused <= u64::from(binds), "draw {at}"),
            }
        }
        let pixels = scene.pixels(&command_buffer);
        for (at, &pixel) in pixels.iter().enumerate() {
            let want = match at as u64 % SIDE {
                1 => GREEN,
                0 | 2..=5 => RED,
                _ => CLEAR,
            };
            assert_eq!(pixel, want, "pixel {at}: {pixels:?}");
        }
    });
}

/// Draws of other inline bytes, one column each, alternating between two textures: a draw binds
/// the set 0 an earlier draw of its command buffer allocated and wrote where it writes the same
/// descriptors into a set of the same layout (`WorkCounts::set_zero_sets_found`), its bytes bound
/// at their dynamic offset, so the set holds the same descriptors whatever bytes the draw reads:
/// past the first two, each draw binds the set its texture's last draw wrote, its encoder's last
/// draw having bound the other texture, and so does a second encoder's first draw, whose encoder
/// wrote none. Each column holds its own bytes' rectangle in its own texture's color, so each
/// set found binds the bytes its draw set. Where the device pushes set 0, none binds one. Both
/// ways ([`on_device_both_ways`]).
#[test]
fn draws_bind_a_set_zero_their_command_buffer_wrote_holding_the_same() {
    on_device_both_ways(|device| {
        let scene = Scene::new(device);
        let textures = [
            solid(device, PixelFormat::RGBA8Unorm, RED),
            solid(device, PixelFormat::RGBA8Unorm, GREEN),
        ];
        let texture = |c: u64| &textures[(c % 2) as usize];
        let (command_buffer, mut encoder) = scene.encoder(device);
        let mut found = Vec::new();
        for c in 0..SIDE - 1 {
            encoder.set_vertex_bytes(&column(c), 0).unwrap();
            encoder.set_fragment_texture(Some(texture(c)), 0).unwrap();
            let before = device.work_counts();
            draw(&mut encoder);
            found.push(grew(&before, &device.work_counts(), |c| {
                c.set_zero_sets_found
            }));
        }
        encoder.end_encoding().unwrap();
        // The last column in a second encoder of the command buffer.
        let mut encoder = command_buffer
            .render_command_encoder(&scene.pass(LoadAction::Load))
            .unwrap();
        encoder.set_render_pipeline_state(&scene.pipeline).unwrap();
        encoder
            .set_fragment_sampler_state(Some(&scene.sampler), 0)
            .unwrap();
        encoder.set_vertex_bytes(&column(SIDE - 1), 0).unwrap();
        encoder
            .set_fragment_texture(Some(texture(SIDE - 1)), 0)
            .unwrap();
        let before = device.work_counts();
        draw(&mut encoder);
        found.push(grew(&before, &device.work_counts(), |c| {
            c.set_zero_sets_found
        }));
        encoder.end_encoding().unwrap();
        let binds = u64::from(!pushes(device));
        let want: Vec<u64> = (0..SIDE).map(|c| if c < 2 { 0 } else { binds }).collect();
        assert_eq!(found, want);
        let pixels = scene.pixels(&command_buffer);
        for (at, &pixel) in pixels.iter().enumerate() {
            let want = [RED, GREEN][(at as u64 % SIDE % 2) as usize];
            assert_eq!(pixel, want, "pixel {at}: {pixels:?}");
        }
    });
}

/// Draws of other inline bytes, one column each, the first half of the columns with one texture
/// and the second half with the other: where the device pushes set 0, each draw past the first
/// but the one that switches the texture pushes fewer writes than the first, only those its bytes
/// changed, and that one at most as many (`WorkCounts::set_zero_writes_pushed`); where it binds
/// set 0, none pushes. Each column is drawn, in its half's texture, both ways
/// ([`on_device_both_ways`]).
#[test]
fn draws_of_other_bytes_push_only_what_changed() {
    on_device_both_ways(|device| {
        let scene = Scene::new(device);
        let textures = [
            solid(device, PixelFormat::RGBA8Unorm, RED),
            solid(device, PixelFormat::RGBA8Unorm, GREEN),
        ];
        make(
            device,
            &scene,
            std::slice::from_ref(&scene.pipeline),
            &textures[0],
        );
        let half = SIDE / 2;
        let (command_buffer, mut encoder) = scene.encoder(device);
        let pushed: Vec<u64> = (0..SIDE)
            .map(|c| {
                encoder.set_vertex_bytes(&column(c), 0).unwrap();
                if c % half == 0 {
                    encoder
                        .set_fragment_texture(Some(&textures[(c / half) as usize]), 0)
                        .unwrap();
                }
                let before = device.work_counts();
                draw(&mut encoder);
                grew(&before, &device.work_counts(), |c| c.set_zero_writes_pushed)
            })
            .collect();
        encoder.end_encoding().unwrap();
        match pushes(device) {
            true => {
                let first = pushed[0];
                assert!(first > 0, "{pushed:?}");
                for (c, &n) in (1..).zip(&pushed[1..]) {
                    match c == half {
                        true => assert!(0 < n && n <= first, "draw {c}: {pushed:?}"),
                        false => assert!(n < first, "draw {c}: {pushed:?}"),
                    }
                }
            }
            false => assert!(pushed.iter().all(|&n| n == 0), "{pushed:?}"),
        }
        let pixels = scene.pixels(&command_buffer);
        for (at, &pixel) in pixels.iter().enumerate() {
            let want = if at as u64 % SIDE < half { RED } else { GREEN };
            assert_eq!(pixel, want, "pixel {at}: {pixels:?}");
        }
    });
}

/// Draws that switch between two pipeline states of the same layout and dynamic states at every
/// draw, each with other inline bytes, record their pipeline and what changed, never the dynamic
/// states the first draw set: each records fewer state commands than the first. Every column
/// is drawn, through either pipeline.
#[test]
fn draws_switching_pipelines_keep_the_dynamic_states_set() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let pipelines = [scene.pipeline.clone(), pipeline(device, true)];
        let red = solid(device, PixelFormat::RGBA8Unorm, RED);
        make(device, &scene, &pipelines, &red);
        let (command_buffer, mut encoder) = scene.encoder(device);
        encoder.set_fragment_texture(Some(&red), 0).unwrap();
        let recorded: Vec<u64> = (0..SIDE)
            .map(|c| {
                encoder
                    .set_render_pipeline_state(&pipelines[(c % 2) as usize])
                    .unwrap();
                encoder.set_vertex_bytes(&column(c), 0).unwrap();
                recorded_by(device, &mut encoder)
            })
            .collect();
        encoder.end_encoding().unwrap();
        let (first, later) = (recorded[0], &recorded[1..]);
        assert!(later.iter().all(|&n| 1 <= n && n < first), "{recorded:?}");
        let pixels = scene.pixels(&command_buffer);
        assert!(pixels.iter().all(|&p| p == RED), "{pixels:?}");
    });
}

/// Draws take their shape, what they derive from their encoder's state alone (settings, draw
/// state, `VkPipeline`), from an earlier draw of their encoder that derived it from the same
/// inputs (`WorkCounts::draw_shapes_reused`): a draw of other bytes takes it, and so does one
/// switched back to a pipeline state the encoder drew before another's; the first draw, a draw of
/// another pipeline state, and a draw after a change of the encoder's state (a scissor rectangle,
/// the blend color) derive their own. Each column is drawn, through either state (the second
/// blends, storing the fragment's color), those under the half-height scissor rectangle in that
/// half alone.
#[test]
fn draws_take_their_shape_from_their_encoder_s_draws_of_the_same_inputs() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let pipelines = [scene.pipeline.clone(), pipeline(device, true)];
        let red = solid(device, PixelFormat::RGBA8Unorm, RED);
        make(device, &scene, &pipelines, &red);
        let (command_buffer, mut encoder) = scene.encoder(device);
        encoder.set_fragment_texture(Some(&red), 0).unwrap();
        let mut reused = Vec::new();
        for c in 0..SIDE {
            match c {
                2 => encoder.set_render_pipeline_state(&pipelines[1]).unwrap(),
                3 => encoder.set_render_pipeline_state(&pipelines[0]).unwrap(),
                4 => {
                    let half = ScissorRect::new(0, 0, SIDE, SIDE / 2);
                    encoder.set_scissor_rect(half).unwrap();
                }
                6 => {
                    let whole = ScissorRect::new(0, 0, SIDE, SIDE);
                    encoder.set_scissor_rect(whole).unwrap();
                    encoder.set_blend_color(0.5, 0.5, 0.5, 1.0);
                }
                _ => {}
            }
            encoder.set_vertex_bytes(&column(c), 0).unwrap();
            let before = device.work_counts();
            draw(&mut encoder);
            reused.push(grew(&before, &device.work_counts(), |c| {
                c.draw_shapes_reused
            }));
        }
        encoder.end_encoding().unwrap();
        assert_eq!(reused, [0, 1, 0, 1, 0, 1, 0, 1]);
        let pixels = scene.pixels(&command_buffer);
        for (at, &pixel) in pixels.iter().enumerate() {
            let (row, c) = (at as u64 / SIDE, at as u64 % SIDE);
            let want = match (4..6).contains(&c) && row >= SIDE / 2 {
                true => CLEAR,
                false => RED,
            };
            assert_eq!(pixel, want, "pixel {at}: {pixels:?}");
        }
    });
}

/// Draws alternating between two pipeline states of the same functions, one blending (its own
/// pipeline key, so its own variant), whose layouts are defined alike and so are one
/// (`pipeline/shared_layouts.rs`, `WorkCounts::pipeline_layouts_shared`), each draw the target's
/// whole rectangle, the texture switched halfway: where the device pushes set 0, each draw pushes
/// what the same draws of the first state alone push, the writes that changed, and at least one
/// past the first where the driver's pushed descriptors go stale across a pipeline bind
/// (`RuntimeDriverDefect::PushedDescriptorsStaleAfterPipelineBinds`). No push counts as a switch
/// between equal shapes (`WorkCounts::set_zero_equal_shape_switches` and kin), either way. The
/// target is the second texture's; both ways ([`on_device_both_ways`]).
#[test]
fn draws_alternating_states_of_one_layout_push_only_what_changed() {
    const DRAWS: usize = 8;
    on_device_both_ways(|device| {
        let scene = Scene::new(device);
        let textures = [
            solid(device, PixelFormat::RGBA8Unorm, RED),
            solid(device, PixelFormat::RGBA8Unorm, GREEN),
        ];
        make(
            device,
            &scene,
            std::slice::from_ref(&scene.pipeline),
            &textures[0],
        );
        let before = device.work_counts();
        let pipelines = [scene.pipeline.clone(), pipeline(device, true)];
        make(device, &scene, &pipelines, &textures[0]);
        let shared = grew(&before, &device.work_counts(), |c| {
            c.pipeline_layouts_shared
        });
        assert!(shared >= 1, "the second state's layout shared: {shared}");
        let corners = [-1.0f32, -1.0, 1.0, 1.0].map(f32::to_le_bytes).concat();
        // Each draw's pushed writes, and the three counts' growth, through the first state alone
        // or both in turn.
        let run = |alternate: bool| {
            let (command_buffer, mut encoder) = scene.encoder(device);
            encoder.set_vertex_bytes(&corners, 0).unwrap();
            let before = device.work_counts();
            let pushed: Vec<u64> = (0..DRAWS)
                .map(|d| {
                    let state = if alternate { d % 2 } else { 0 };
                    encoder
                        .set_render_pipeline_state(&pipelines[state])
                        .unwrap();
                    if d % (DRAWS / 2) == 0 {
                        let texture = &textures[d / (DRAWS / 2)];
                        encoder.set_fragment_texture(Some(texture), 0).unwrap();
                    }
                    let before = device.work_counts();
                    draw(&mut encoder);
                    grew(&before, &device.work_counts(), |c| c.set_zero_writes_pushed)
                })
                .collect();
            let after = device.work_counts();
            encoder.end_encoding().unwrap();
            let counts = [
                grew(&before, &after, |c| c.set_zero_equal_shape_switches),
                grew(&before, &after, |c| {
                    c.set_zero_writes_on_equal_shape_switches
                }),
                grew(&before, &after, |c| c.set_zero_writes_if_shapes_shared),
            ];
            let pixels = scene.pixels(&command_buffer);
            assert!(pixels.iter().all(|&p| p == GREEN), "{pixels:?}");
            (pushed, counts)
        };
        let (alone, alone_counts) = run(false);
        let (alternating, counts) = run(true);
        println!(
            "runtime_lane: set 0 pushed alone {alone:?}, alternating {alternating:?}; equal-shape \
             switches, writes carried and writes if shared {counts:?}"
        );
        assert_eq!(alone_counts, [0; 3], "{alone:?}");
        assert_eq!(counts, [0; 3], "{alternating:?}");
        if !pushes(device) {
            return;
        }
        let stale =
            metal2vulkan::error::RuntimeDriverDefect::PushedDescriptorsStaleAfterPipelineBinds;
        let at_least = u64::from(device.properties().driver_defects.contains(&stale));
        assert!(alone[0] > 0, "{alone:?}");
        let expected: Vec<u64> = (alone.iter().enumerate())
            .map(|(d, &n)| if d == 0 { n } else { n.max(at_least) })
            .collect();
        assert_eq!(alternating, expected, "alone {alone:?}");
    });
}

/// A pipeline state of the owned fixture `zwd_shapes`' `zwd_quad` (a shape's box, from its bytes
/// at buffer 0) and its fragment function `fragment` (the shape's fill times its coverage, from the
/// same bytes at fragment buffer 0), its constant 0 (`zwd_variant`) 0, into an `RGBA8Unorm` color
/// attachment, blending premultiplied colors.
fn shape_pipeline(device: &Device, fragment: &str) -> RenderPipelineState {
    let ll = engine_dir().join("zwd_shapes.ll");
    let library = device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap();
    let constants = FunctionConstantValues::defaults().with_index(
        0,
        FunctionConstantValue::new(DataType::UInt, 0u32.to_le_bytes().to_vec()),
    );
    let function = |name| {
        StageSource::Function(
            library
                .new_function_with_name_constant_values(name, &constants)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("zwd_quad"));
    descriptor.fragment = Some(function(fragment));
    let mut color = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
    color.blending_enabled = true;
    color.source_rgb_blend_factor = BlendFactor::One;
    color.source_alpha_blend_factor = BlendFactor::One;
    color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
    color.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
    descriptor.color_attachments[0] = Some(color);
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// The bytes of [`SHAPES`] `ZwdShape`s, one every [`SHAPE_STRIDE`] bytes: shape `p` a box over the
/// target's columns `2p` and `2p + 1`, of half a pixel's radius, a premultiplied fill of its own.
fn shape_bytes() -> Vec<u8> {
    let side = SIDE as f32;
    (0..SHAPES)
        .flat_map(|p| {
            let x = 2.0 * p as f32;
            let fill = [0.125 * x, 0.5 - 0.0625 * x, 0.25, 0.75];
            let words = [x, 1.0, x + 2.0, side - 1.0]
                .into_iter()
                .chain(fill)
                .chain([0.0; 4])
                .chain([0.5, side, side, 0.0]);
            let mut bytes: Vec<u8> = words.flat_map(f32::to_le_bytes).collect();
            bytes.resize(SHAPE_STRIDE as usize, 0);
            bytes
        })
        .collect()
}

/// The shapes [`shape_bytes`] holds, and the bytes from one to the next: past every device's
/// uniform buffer offset alignment (Vulkan's largest allowed is 256).
const SHAPES: u64 = SIDE / 2;
const SHAPE_STRIDE: u64 = 256;

/// Draws alternating between two pipeline states of other fragment functions of one interface
/// (`zwd_panel`, `zwd_disc`), whose layouts are defined alike and so are one
/// (`pipeline/shared_layouts.rs`), each pair of draws one shape of a buffer ([`shape_bytes`]) at
/// vertex and fragment buffer 0, the second of each pair binding what the first bound: in one
/// pass, where a pipeline bind between them keeps set 0, they draw what the same draws draw each
/// in a command buffer of its own, which pushes or binds set 0 whole. Where the device pushes set
/// 0, the second of each pair pushes nothing, as set 0 holds its descriptors; where the driver's
/// pushed descriptors go stale across a pipeline bind
/// (`RuntimeDriverDefect::PushedDescriptorsStaleAfterPipelineBinds`), it pushes one write, and
/// each draw past the first at least one. No push counts as a switch between equal shapes. Both
/// ways ([`on_device_both_ways`]).
#[test]
fn draws_alternating_functions_of_one_layout_read_their_own_bytes() {
    on_device_both_ways(|device| {
        let scene = Scene::new(device);
        let panel = shape_pipeline(device, "zwd_panel");
        let made = device.work_counts();
        let states = [panel, shape_pipeline(device, "zwd_disc")];
        let shapes = buffer_with(device, &shape_bytes()).unwrap();
        let record = |encoder: &mut RenderCommandEncoder, d: u64| {
            let offset = d / 2 * SHAPE_STRIDE;
            encoder
                .set_render_pipeline_state(&states[(d % 2) as usize])
                .unwrap();
            encoder.set_vertex_buffer(Some(&shapes), offset, 0).unwrap();
            encoder
                .set_fragment_buffer(Some(&shapes), offset, 0)
                .unwrap();
            draw(encoder);
        };
        // Each draw in a command buffer of its own, the first clearing the target.
        let queue = device.new_command_queue().unwrap();
        let mut apart = Vec::new();
        for d in 0..2 * SHAPES {
            let command_buffer = queue.command_buffer().unwrap();
            let load = if d == 0 {
                LoadAction::Clear
            } else {
                LoadAction::Load
            };
            let mut encoder = command_buffer
                .render_command_encoder(&scene.pass(load))
                .unwrap();
            record(&mut encoder, d);
            encoder.end_encoding().unwrap();
            match d + 1 < 2 * SHAPES {
                true => command_buffer.commit().unwrap(),
                false => apart = scene.pixels(&command_buffer),
            }
        }
        // The disc's variant, made by now, took the panel's layout.
        let shared = grew(&made, &device.work_counts(), |c| c.pipeline_layouts_shared);
        assert!(shared >= 1, "the disc's layout shared: {shared}");
        // Every draw in one pass.
        let (command_buffer, mut encoder) = scene.encoder(device);
        let before = device.work_counts();
        let pushed: Vec<u64> = (0..2 * SHAPES)
            .map(|d| {
                let before = device.work_counts();
                record(&mut encoder, d);
                grew(&before, &device.work_counts(), |c| c.set_zero_writes_pushed)
            })
            .collect();
        let switches = grew(&before, &device.work_counts(), |c| {
            c.set_zero_equal_shape_switches
        });
        encoder.end_encoding().unwrap();
        let together = scene.pixels(&command_buffer);
        println!("runtime_lane: alternating functions of one layout pushed {pushed:?}");
        assert!(together.iter().any(|&p| p != CLEAR), "{together:?}");
        assert_eq!(together, apart, "pushed {pushed:?}");
        assert_eq!(switches, 0, "{pushed:?}");
        if !pushes(device) {
            return;
        }
        let stale =
            metal2vulkan::error::RuntimeDriverDefect::PushedDescriptorsStaleAfterPipelineBinds;
        let at_least = u64::from(device.properties().driver_defects.contains(&stale));
        let seconds: Vec<u64> = pushed.iter().copied().skip(1).step_by(2).collect();
        assert!(seconds.iter().all(|&n| n == at_least), "{pushed:?}");
        assert!(pushed[1..].iter().all(|&n| n >= at_least), "{pushed:?}");
    });
}

/// A pipeline state that rasterizes nothing: the owned fixture `rvc_draw_bases`'s vertex
/// function, which returns nothing, as Metal requires there, and writes 16 bytes at element
/// `(iid * 8 & 24) | (vid & 7)` of buffer 0, into an `RGBA8Unorm` color attachment.
fn unrasterized(device: &Device) -> RenderPipelineState {
    let ll = engine_dir().join("rvc_draw_bases.ll");
    let library = device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap();
    let vertex = library
        .new_function_with_name("draw_bases_in_a_vertex_function")
        .unwrap()
        .function_descriptor();
    let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(vertex));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    descriptor.rasterization_enabled = false;
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// Draw once through `unrasterized`, binding `written` at vertex buffer 0.
fn draw_unrasterized(
    encoder: &mut RenderCommandEncoder,
    unrasterized: &RenderPipelineState,
    written: &Buffer,
) {
    encoder.set_render_pipeline_state(unrasterized).unwrap();
    encoder.set_vertex_buffer(Some(written), 0, 0).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .unwrap();
}

/// A draw after one through a pipeline that rasterizes nothing, in one render pass instance,
/// sets its depth, stencil and blend-constant states again: a pipeline without fragment state
/// holds none of them, so its bind leaves them unknown, and the record of the secondary command
/// buffer (`render/bound.rs`) must not keep them as set. Both pipelines are made first, so every
/// draw records as it draws, into one secondary. Under the validation layer, a draw that left
/// them out reports them unset; every column is drawn.
#[test]
fn a_draw_after_an_unrasterized_draw_sets_its_fragment_states_again() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let red = solid(device, PixelFormat::RGBA8Unorm, RED);
        let unrasterized = unrasterized(device);
        let written = buffer_with(device, &[0; 512]).unwrap();
        make(device, &scene, std::slice::from_ref(&scene.pipeline), &red);
        let (command_buffer, mut encoder) = scene.encoder(device);
        draw_unrasterized(&mut encoder, &unrasterized, &written);
        encoder.end_encoding().unwrap();
        scene.pixels(&command_buffer);
        let (command_buffer, mut encoder) = scene.encoder(device);
        encoder.set_fragment_texture(Some(&red), 0).unwrap();
        for c in 0..SIDE {
            encoder.set_render_pipeline_state(&scene.pipeline).unwrap();
            encoder.set_vertex_bytes(&column(c), 0).unwrap();
            draw(&mut encoder);
            if c == 0 {
                draw_unrasterized(&mut encoder, &unrasterized, &written);
            }
        }
        encoder.end_encoding().unwrap();
        let pixels = scene.pixels(&command_buffer);
        assert!(pixels.iter().all(|&p| p == RED), "{pixels:?}");
    });
}

/// A texture the program let go of after its draw, the fragment table then binding another, is
/// held until the command buffer completes: its column is its texture's, and the validation
/// layer reports no use of a destroyed object. The columns left of the first draw's keep the
/// clear.
#[test]
fn a_texture_let_go_after_its_draw_lives_until_the_command_buffer_completes() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let green = solid(device, PixelFormat::RGBA8Unorm, GREEN);
        let (command_buffer, mut encoder) = scene.encoder(device);
        for c in 1..SIDE {
            encoder.set_vertex_bytes(&column(c), 0).unwrap();
            if c % 2 == 1 {
                let red = solid(device, PixelFormat::RGBA8Unorm, RED);
                encoder.set_fragment_texture(Some(&red), 0).unwrap();
                draw(&mut encoder);
                // The program's last handle goes: the encoder alone holds it now.
                drop(red);
                encoder.set_fragment_texture(Some(&green), 0).unwrap();
            } else {
                draw(&mut encoder);
            }
        }
        encoder.end_encoding().unwrap();
        let pixels = scene.pixels(&command_buffer);
        for (at, &pixel) in pixels.iter().enumerate() {
            let want = match at as u64 % SIDE {
                0 => CLEAR,
                c if c % 2 == 1 => RED,
                _ => GREEN,
            };
            assert_eq!(pixel, want, "pixel {at}: {pixels:?}");
        }
    });
}

/// A fragment table that binds two textures, one slot rebound mid-encoder to a texture the program
/// lets go of after its draw: each draw that binds a texture its encoder does not hold yet retains
/// the table, and one whose objects the encoder holds does not; the texture lives until the
/// command buffer completes (its column is its own, and the validation layer reports no use of a
/// destroyed object), as the other slot's, held from the first retain, does. The columns left of
/// the first draw's keep the clear.
#[test]
fn a_slot_rebound_to_a_texture_let_go_after_its_draw_holds_it_until_completion() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let green = solid(device, PixelFormat::RGBA8Unorm, GREEN);
        // At slot 1, which the shader does not read: bound, and so held, throughout.
        let unread = solid(device, PixelFormat::RGBA8Unorm, CLEAR);
        let (command_buffer, mut encoder) = scene.encoder(device);
        encoder.set_fragment_texture(Some(&unread), 1).unwrap();
        let mut first = None;
        for c in 1..SIDE {
            encoder.set_vertex_bytes(&column(c), 0).unwrap();
            if c % 2 == 1 {
                let red = solid(device, PixelFormat::RGBA8Unorm, RED);
                encoder.set_fragment_texture(Some(&red), 0).unwrap();
                draw(&mut encoder);
                first.get_or_insert_with(|| device.work_counts());
                // The program's last handle goes, and the slot is rebound: the encoder alone
                // holds it now.
                drop(red);
                encoder.set_fragment_texture(Some(&green), 0).unwrap();
            } else {
                // The rebound slot's draw, then one that changed nothing.
                draw(&mut encoder);
                draw(&mut encoder);
            }
        }
        let (first, rest) = (first.unwrap(), device.work_counts());
        encoder.end_encoding().unwrap();
        // After the first draw: the green texture's first draw, and each later odd column's of
        // its own red texture.
        let taking = 1 + (SIDE - 2) / 2;
        assert_eq!(grew(&first, &rest, |c| c.argument_tables_retained), taking);
        assert_eq!(grew(&first, &rest, |c| c.argument_objects_retained), taking);
        let pixels = scene.pixels(&command_buffer);
        for (at, &pixel) in pixels.iter().enumerate() {
            let want = match at as u64 % SIDE {
                0 => CLEAR,
                c if c % 2 == 1 => RED,
                _ => GREEN,
            };
            assert_eq!(pixel, want, "pixel {at}: {pixels:?}");
        }
    });
}

/// Draws that each set other vertex bytes and rebind the fragment table's texture slot 0 in turn
/// to one of two textures that look alike (one format, size, usage and texels), the table also
/// binding a buffer and, at slot 1, a texture the shader does not read: the second draw retains
/// the table, taking the second texture, and each later one holds every object already. The ended
/// encoder lists the target, the buffer and the three textures (`CommandBuffer::ended_encoders`),
/// whether a draw took one or held it already; each column is the textures' color.
#[test]
fn draws_rebinding_a_slot_to_a_texture_that_looks_alike_list_each_object() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let alike = [
            solid(device, PixelFormat::RGBA8Unorm, RED),
            solid(device, PixelFormat::RGBA8Unorm, RED),
        ];
        // Bound throughout, and read by no draw: held from the first retain on.
        let unread = solid(device, PixelFormat::RGBA8Unorm, CLEAR);
        let unread_buffer = buffer_with(device, &[0; 16]).unwrap();
        let (command_buffer, mut encoder) = scene.encoder(device);
        encoder.set_fragment_texture(Some(&unread), 1).unwrap();
        encoder
            .set_fragment_buffer(Some(&unread_buffer), 0, 0)
            .unwrap();
        let mut first = None;
        for c in 0..SIDE {
            encoder.set_vertex_bytes(&column(c), 0).unwrap();
            encoder
                .set_fragment_texture(Some(&alike[(c % 2) as usize]), 0)
                .unwrap();
            draw(&mut encoder);
            first.get_or_insert_with(|| device.work_counts());
        }
        let (first, rest) = (first.unwrap(), device.work_counts());
        encoder.end_encoding().unwrap();
        assert_eq!(grew(&first, &rest, |c| c.argument_tables_retained), 1);
        assert_eq!(grew(&first, &rest, |c| c.argument_objects_retained), 1);
        let ended = command_buffer.ended_encoders();
        assert_eq!(ended.len(), 1);
        let used = &ended[0].used_resources;
        let want = [&scene.target, &alike[0], &alike[1], &unread];
        assert_eq!(used.textures.len(), want.len(), "{:?}", used.textures);
        for (at, texture) in want.into_iter().enumerate() {
            assert!(used.textures.contains(texture), "texture {at} not listed");
        }
        assert_eq!(used.buffers, std::slice::from_ref(&unread_buffer));
        assert!(used.heaps.is_empty(), "{:?}", used.heaps);
        let pixels = scene.pixels(&command_buffer);
        assert!(pixels.iter().all(|&p| p == RED), "{pixels:?}");
    });
}

/// What one draw of [`draws_state_the_resources_resident_as_they_were_encoded`] changed before it,
/// and the usages and render stages resident as it was encoded, by token.
struct Resident {
    change: &'static str,
    resident: &'static [(ResourceUsage, RenderStages)],
}

/// A draw states the resources its encoder made resident as they were when it was encoded, though
/// the encoder's draws take its last draw's used resources without making them again while the
/// residency is unchanged (`Residency::generation`): a draw after a resource made resident, its
/// usage widened, its render stages widened, or another made resident through `useResources`
/// states the new set; one after the same resource made resident again as it was states the set
/// it had. Each draw is its own column, the texture's.
#[test]
fn draws_state_the_resources_resident_as_they_were_encoded() {
    const READ: ResourceUsage = ResourceUsage::READ;
    const BOTH: ResourceUsage = ResourceUsage::READ.union(ResourceUsage::WRITE);
    const VERTEX: RenderStages = RenderStages::VERTEX;
    const FRAGMENT: RenderStages = RenderStages::FRAGMENT;
    const DRAWN: RenderStages = RenderStages::VERTEX.union(RenderStages::FRAGMENT);
    let steps = [
        Resident {
            change: "the first",
            resident: &[],
        },
        Resident {
            change: "a buffer made resident",
            resident: &[(READ, VERTEX)],
        },
        Resident {
            change: "nothing, it resident",
            resident: &[(READ, VERTEX)],
        },
        Resident {
            change: "it made resident again as it was",
            resident: &[(READ, VERTEX)],
        },
        Resident {
            change: "its usage widened",
            resident: &[(BOTH, VERTEX)],
        },
        Resident {
            change: "its stages widened",
            resident: &[(BOTH, DRAWN)],
        },
        Resident {
            change: "another made resident",
            resident: &[(BOTH, DRAWN), (READ, FRAGMENT)],
        },
        Resident {
            change: "nothing, both resident",
            resident: &[(BOTH, DRAWN), (READ, FRAGMENT)],
        },
    ];
    assert_eq!(steps.len() as u64, SIDE, "a column a draw");
    on_device(|device, _| {
        let scene = Scene::new(device);
        let red = solid(device, PixelFormat::RGBA8Unorm, RED);
        let first = buffer_with(device, &[0; 64]).unwrap();
        let second = buffer_with(device, &[0; 64]).unwrap();
        let (command_buffer, mut encoder) = scene.encoder(device);
        encoder.set_fragment_texture(Some(&red), 0).unwrap();
        for (c, step) in (0..).zip(&steps) {
            let first = ResourceRef::Buffer(&first);
            match step.change {
                "a buffer made resident" | "it made resident again as it was" => {
                    encoder.use_resource(first, READ, VERTEX).unwrap();
                }
                "its usage widened" => {
                    encoder
                        .use_resource(first, ResourceUsage::WRITE, VERTEX)
                        .unwrap();
                }
                "its stages widened" => encoder.use_resource(first, READ, FRAGMENT).unwrap(),
                "another made resident" => encoder
                    .use_resources(&[ResourceRef::Buffer(&second)], READ, FRAGMENT)
                    .unwrap(),
                _ => {}
            }
            encoder.set_vertex_bytes(&column(c), 0).unwrap();
            draw(&mut encoder);
        }
        encoder.end_encoding().unwrap();
        let pixels = scene.pixels(&command_buffer);
        assert!(pixels.iter().all(|&p| p == RED), "{pixels:?}");
        let executions = command_buffer.executions();
        assert_eq!(executions.len(), steps.len());
        for (step, execution) in steps.iter().zip(&executions) {
            let want: BTreeMap<u32, (UsedClass, ResourceUsage, RenderStages)> = (0u32..)
                .zip(step.resident)
                .map(|(token, &(usage, stages))| (token, (UsedClass::Buffer, usage, stages)))
                .collect();
            let used: BTreeMap<u32, (UsedClass, ResourceUsage, RenderStages)> =
                (execution.shape().used.iter())
                    .map(|(&token, used)| (token, (used.class, used.usage, used.stages)))
                    .collect();
            assert_eq!(used, want, "{}", step.change);
        }
    });
}
