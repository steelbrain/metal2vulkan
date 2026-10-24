//! Device tests of a render encoder's argument buffers on the device the cfg selects, under the
//! validation layer: a draw's fragment function reaches a texture, a sampler state and a buffer
//! through what its argument buffer holds, once the encoder made them resident in the fragment
//! stage (`use_resource(s)`), and its variant's set 1 holds the texture and the sampler at their
//! resource IDs. Render pipeline states give handles and make visible-function tables per stage,
//! which the encoder binds per stage.
//!
//! Each test draws a committed public fragment case whose argument buffer holds those members,
//! encoding the members itself as the Metal oracle's argument encoder does (each buffer's
//! `gpu_address` or texture's or sampler state's `gpu_resource_id` at its field), and compares
//! what the draw wrote with the case's Metal observation.

use super::renders::{public_render, refusal, PublicRender};
use super::{
    buffer_with, read_buffer, texture_with, validated_device, Messages, ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::candidate::engine::graphics::{Assembly, RenderFunction, RenderOutput};
use crate::candidate::engine::resources::{argument_sampler_descriptor, pixel_format};
use crate::candidate::runtime_case::{render_pipeline_descriptor, render_pipeline_state};
use crate::case::indirect_commands::metal_primitive_type;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::DrawCall;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, RenderStages, ResourceUsage, TextureUsage};
use metal2vulkan::pipeline::{LinkedFunctions, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, CommandBufferStatus, Device, Error, FunctionTableProblem, Invalid, LoadAction, Nil,
    PipelineOptions, Problem, Region, RenderPassDescriptor, ResourceOptions, ResourceRef,
    SamplerState, SamplerStateOptions, StoreAction, Texture, TextureDescriptor,
    VisibleFunctionTableDescriptor,
};
use std::collections::BTreeMap;

/// A fragment samples a texture through a sampler state, both held in its argument buffer (also
/// `render_executions.rs`'s host-step draw of a state made after the commit).
pub(super) const SAMPLED: &str =
    "sample-in-a-fragment-through-a-sampler-held-beside-its-texture-in-an-argument-buffer";

/// Sixteen fragments of each pixel update a word of a buffer their argument buffer points to, in
/// a raster order group.
pub(super) const ORDERED: &str =
    "sixteen-overlapping-triangles-update-a-raster-ordered-argument-buffer-member";

/// Run `test` on a validated device of the lane, then fail on any validation message.
fn on_device(test: impl FnOnce(&Device)) {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        test(&device);
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}

/// What a draw of a case reaches through its fragment function's argument buffers: the owners
/// (bound buffers, each holding its members' words) and the members, by `(owner, field)`.
pub(super) struct Members {
    pub(super) owners: BTreeMap<u32, Buffer>,
    buffers: BTreeMap<(u32, u32), Buffer>,
    textures: BTreeMap<(u32, u32), Texture>,
    samplers: BTreeMap<(u32, u32), SamplerState>,
}

impl Members {
    /// The members of `function`'s argument buffers on `device`, each owner holding at each
    /// field its member's `gpu_address` or `gpu_resource_id`.
    pub(super) fn of(device: &Device, function: &RenderFunction) -> Members {
        let resources = &function.resources;
        let buffers: BTreeMap<_, _> = resources
            .argument_buffer_buffers
            .iter()
            .map(|(&key, bytes)| (key, buffer_with(device, bytes).unwrap()))
            .collect();
        let textures: BTreeMap<_, _> = resources
            .argument_buffer_textures
            .iter()
            .map(|(&key, literal)| {
                let texture = texture_with(device, literal, TextureUsage::SHADER_READ).unwrap();
                (key, texture)
            })
            .collect();
        let mut options = SamplerStateOptions::default();
        options.support_argument_buffers = true;
        let samplers: BTreeMap<_, _> = resources
            .argument_buffer_samplers
            .iter()
            .map(|(&key, sampler)| {
                let descriptor = argument_sampler_descriptor(sampler);
                let state = device
                    .new_sampler_state_with_options(&descriptor, options)
                    .unwrap();
                (key, state)
            })
            .collect();
        let mut owned = resources.buffers.clone();
        let words = buffers
            .iter()
            .map(|(&key, buffer)| (key, buffer.gpu_address().0))
            .chain(textures.iter().map(|(&k, t)| (k, t.gpu_resource_id().0)))
            .chain(samplers.iter().map(|(&k, s)| (k, s.gpu_resource_id().0)));
        for ((owner, field), word) in words {
            let bytes = owned.get_mut(&owner).expect("a member's owner is bound");
            let at = field as usize;
            bytes[at..at + 8].copy_from_slice(&word.to_le_bytes());
        }
        let owners = owned
            .iter()
            .map(|(&index, bytes)| (index, buffer_with(device, bytes).unwrap()))
            .collect();
        Members {
            owners,
            buffers,
            textures,
            samplers,
        }
    }

    /// Every member, as a resource to make resident.
    pub(super) fn resources(&self) -> Vec<ResourceRef<'_>> {
        let buffers = self.buffers.values().map(ResourceRef::Buffer);
        let textures = self.textures.values().map(ResourceRef::Texture);
        buffers.chain(textures).collect()
    }
}

/// Draw `case` on `device` `times` times in one encoder, with its fragment function's argument
/// buffers' members made resident in `stages`: its color target 0 and the members, once the
/// command buffer completed.
fn draw(
    device: &Device,
    case: &PublicRender,
    stages: RenderStages,
    times: usize,
) -> (Texture, Members) {
    let render = &case.render;
    let fragment = render.fragment.as_ref().expect("a fragment case");
    let members = Members::of(device, fragment);
    let state = render_pipeline_state(device, &case.inputs()).unwrap();
    let target = &render.color_targets[&0];
    let [width, height] = target.dimensions;
    let mut descriptor = TextureDescriptor::texture_2d(
        pixel_format(target.format),
        width.into(),
        height.into(),
        false,
    );
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&descriptor).unwrap();
    let row = u64::from(width) * target.format.bytes_per_pixel() as u64;
    let region = Region::new_2d(0, 0, width.into(), height.into());
    texture
        .replace_region(region, 0, 0, &target.bytes, row, row * u64::from(height))
        .unwrap();
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(texture.clone());
    attachment.load_action = LoadAction::Load;
    attachment.store_action = StoreAction::Store;
    let command_buffer = device
        .new_command_queue()
        .unwrap()
        .command_buffer()
        .unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(&state).unwrap();
    for (&index, owner) in &members.owners {
        encoder.set_fragment_buffer(Some(owner), 0, index).unwrap();
    }
    encoder
        .use_resources(
            &members.resources(),
            ResourceUsage::READ | ResourceUsage::WRITE,
            stages,
        )
        .unwrap();
    let DrawCall::Primitives {
        vertex_start,
        vertex_count,
        instance_count,
        base_instance,
    } = render.draw
    else {
        panic!("{}: a draw of primitives", case.case.name);
    };
    let &Assembly::Primitives(primitive) = &render.assembly else {
        panic!("{}: a draw of primitives", case.case.name);
    };
    for _ in 0..times {
        encoder
            .draw_primitives(
                metal_primitive_type(primitive),
                vertex_start,
                vertex_count,
                instance_count,
                base_instance,
            )
            .unwrap();
    }
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        command_buffer.status(),
        CommandBufferStatus::Completed,
        "{:?}",
        command_buffer.error()
    );
    (texture, members)
}

/// The bytes `case`'s output selects of a draw of it: its color target 0's texels, or an
/// argument-buffer buffer's range.
pub(super) fn output(case: &PublicRender, target: &Texture, members: &Members) -> Vec<u8> {
    match case.render.output {
        RenderOutput::ArgumentBufferBuffer {
            owner,
            field_offset,
            offset,
            length,
            ..
        } => read_buffer(&members.buffers[&(owner, field_offset)], offset, length).unwrap(),
        _ => {
            let color = &case.render.color_targets[&0];
            let [width, height] = color.dimensions;
            let row = u64::from(width) * color.format.bytes_per_pixel() as u64;
            let mut bytes = vec![0u8; (row * u64::from(height)) as usize];
            let region = Region::new_2d(0, 0, width.into(), height.into());
            target
                .get_bytes(&mut bytes, row, row * u64::from(height), region, 0, 0)
                .unwrap();
            bytes
        }
    }
}

/// A fragment function samples a texture through a sampler state its argument buffer holds by
/// resource ID, the texture resident in the fragment stage: Metal's texels. The draw's variant
/// has set 1, which holds both at their IDs.
#[test]
fn a_fragment_samples_what_its_argument_buffer_names() {
    let case = public_render(SAMPLED);
    let metal = case.metal_output.clone().unwrap();
    on_device(|device| {
        let (target, members) = draw(device, &case, RenderStages::FRAGMENT, 1);
        assert_eq!(members.samplers.len(), 1);
        assert_eq!(output(&case, &target, &members), metal);
    });
}

/// A fragment function updates a buffer its argument buffer points to, the buffer resident in
/// the vertex and fragment stages: sixteen overlapping fragments of each pixel each add their
/// update in a raster order group, Metal's words. A pipeline the device cannot make is an honest
/// refusal (`renders::refusal`), and is printed.
#[test]
fn a_fragment_updates_the_buffer_its_argument_buffer_points_to() {
    let case = public_render(ORDERED);
    let metal = case.metal_output.clone().unwrap();
    on_device(|device| {
        if let Err(error) = render_pipeline_state(device, &case.inputs()) {
            let why = error.to_string();
            eprintln!("REFUSED {ORDERED}: {why}");
            assert!(refusal(&why), "{why}");
            return;
        }
        let stages = RenderStages::VERTEX | RenderStages::FRAGMENT;
        let (target, members) = draw(device, &case, stages, 1);
        assert_eq!(output(&case, &target, &members), metal);
    });
}

/// Four draws of [`ORDERED`] in one encoder take the same uses, the buffer they update reached
/// through the resident set alone: each after the first is noted already, the encoder having
/// noted the same uses, reaching the same keys so, for the one before
/// (`WorkCounts::commands_noted_as_before`). Each draw still updates as it would alone: a word
/// Metal's one draw takes `k` rounds of `v -> 3v + 1` from its first value takes `4k`. A pipeline
/// the device cannot make is an honest refusal, and is printed.
#[test]
fn draws_reaching_a_resident_buffer_as_the_one_before_are_noted_once() {
    let case = public_render(ORDERED);
    let metal = case.metal_output.clone().unwrap();
    on_device(|device| {
        if let Err(error) = render_pipeline_state(device, &case.inputs()) {
            let why = error.to_string();
            eprintln!("REFUSED {ORDERED}: {why}");
            assert!(refusal(&why), "{why}");
            return;
        }
        let stages = RenderStages::VERTEX | RenderStages::FRAGMENT;
        let noted = || device.work_counts().commands_noted_as_before;
        let before = noted();
        let (target, members) = draw(device, &case, stages, 4);
        assert_eq!(noted() - before, 3);
        let RenderOutput::ArgumentBufferBuffer {
            owner,
            field_offset,
            offset,
            length,
            ..
        } = case.render.output
        else {
            panic!("{ORDERED}: an argument-buffer buffer's output");
        };
        let fragment = case.render.fragment.as_ref().unwrap();
        let first = &fragment.resources.argument_buffer_buffers[&(owner, field_offset)];
        let (offset, length) = (offset as usize, length as usize);
        let words = |bytes: &[u8]| -> Vec<u32> {
            (bytes.as_chunks::<4>().0.iter())
                .map(|word| u32::from_le_bytes(*word))
                .collect()
        };
        let round = |v: u32| v.wrapping_mul(3).wrapping_add(1);
        let rounds = |mut v: u32, n: usize| {
            for _ in 0..n {
                v = round(v);
            }
            v
        };
        let drawn = words(&output(&case, &target, &members));
        let once = words(&metal);
        let firsts = words(&first[offset..offset + length]);
        assert_eq!((drawn.len(), once.len()), (firsts.len(), firsts.len()));
        for (word, ((&first, &once), &drawn)) in firsts.iter().zip(&once).zip(&drawn).enumerate() {
            let k = (0..=64)
                .find(|&k| rounds(first, k) == once)
                .unwrap_or_else(|| {
                    panic!("word {word}: Metal's {once:#x} is no rounds of {first:#x}")
                });
            assert_eq!(
                drawn,
                rounds(first, 4 * k),
                "word {word}: {k} rounds a draw"
            );
        }
    });
}

/// `use_resource(s)` and `use_heap(s)` refuse another device's resource or heap, as a compute
/// encoder's do, and a refusal makes nothing resident.
#[test]
fn another_devices_resources_are_refused() {
    on_device(|device| {
        let messages = Messages::default();
        let cache = ScratchCache::new().unwrap();
        let other = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let foreign = buffer_with(&other, &[0; 16]).unwrap();
        let own = buffer_with(device, &[0; 16]).unwrap();
        let target = device
            .new_texture(&{
                let mut descriptor =
                    TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 4, 4, false);
                descriptor.usage = TextureUsage::RENDER_TARGET;
                descriptor
            })
            .unwrap();
        let mut pass = RenderPassDescriptor::new();
        pass.color_attachments[0].texture = Some(target);
        let command_buffer = device
            .new_command_queue()
            .unwrap()
            .command_buffer()
            .unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        let refused =
            |call: &'static str| Error::Invalid(Invalid::new(call, Problem::ForeignObject));
        let resources = [ResourceRef::Buffer(&own), ResourceRef::Buffer(&foreign)];
        assert_eq!(
            encoder.use_resources(&resources, ResourceUsage::READ, RenderStages::FRAGMENT),
            Err(refused("use_resources"))
        );
        encoder.end_encoding().unwrap();
        drop(command_buffer);
        drop(foreign);
        drop(other);
        assert!(messages.take().is_empty());
    });
}

/// A render pipeline state's handles and tables are per stage (`functionHandleWithFunction:stage:`,
/// `newVisibleFunctionTableWithDescriptor:stage:`): a handle of a function the stage's function
/// links, none of one it does not or of a stage the state has no function of, whose table is
/// made, as plain Metal's, and a refusal of a value that is not one stage. A table takes the
/// handles of its own stage alone. A stage's table binds at its buffer index, where a draw whose
/// function reads a buffer reads it as an unbound buffer, as plain Metal's, and a table at an index
/// the function does not read draws Metal's texels.
///
/// The state is [`SAMPLED`]'s, its fragment function linking the owned `visible_table_functions`
/// fixture's `table_add` (its vertex function, the generated companion, links nothing).
#[test]
fn render_states_give_handles_and_tables_per_stage() {
    let case = public_render(SAMPLED);
    let metal = case.metal_output.clone().unwrap();
    on_device(|device| {
        let bitcode = llvm_assemble(&engine_dir().join("visible_table_functions.ll")).unwrap();
        let functions = device.new_library_with_air(&bitcode).unwrap();
        let add = functions.new_function_with_name("table_add").unwrap();
        let mut descriptor = render_pipeline_descriptor(device, &case.inputs()).unwrap();
        let Some(StageSource::Function(fragment)) = descriptor.fragment.take() else {
            panic!("the case's fragment function");
        };
        let linked = LinkedFunctions::new().with_function(add.linked_function());
        descriptor.fragment = Some(StageSource::Function(fragment.with_linked(linked)));
        let state = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();

        let fragment = RenderStages::FRAGMENT;
        let handle = state.function_handle(&add, fragment).unwrap().unwrap();
        assert_eq!(handle.name(), "table_add");
        assert_eq!(
            state.function_handle(&add, RenderStages::VERTEX).unwrap(),
            None
        );
        for stage in [RenderStages::TILE, RenderStages::OBJECT] {
            assert_eq!(
                state.function_handle(&add, stage).unwrap(),
                None,
                "{stage:?}"
            );
            let stageless = state
                .new_visible_function_table(&VisibleFunctionTableDescriptor::new(2), stage)
                .unwrap();
            stageless.set_function(None, 1).unwrap();
            assert_eq!(
                stageless.set_function(Some(&handle), 0).unwrap_err(),
                Error::Invalid(Invalid::new(
                    "set_function",
                    Problem::FunctionTable(FunctionTableProblem::HandleOfAnotherState)
                )),
                "{stage:?}"
            );
        }
        // A stage that is not one stage: plain Metal answers nil, but faults on a table of no
        // stage.
        for stage in [
            RenderStages::empty(),
            RenderStages::VERTEX | RenderStages::FRAGMENT,
            RenderStages::FRAGMENT | RenderStages::TILE,
        ] {
            let problem =
                Problem::FunctionTable(FunctionTableProblem::StageWithoutFunction { stage });
            assert_eq!(
                state.function_handle(&add, stage).unwrap_err(),
                Error::Nil(Nil::new("function_handle", problem.clone())),
                "{stage:?}"
            );
            let call = "new_visible_function_table";
            let table = match stage.is_empty() {
                true => Error::Invalid(Invalid::new(call, problem)),
                false => Error::Nil(Nil::new(call, problem)),
            };
            assert_eq!(
                state
                    .new_visible_function_table(&VisibleFunctionTableDescriptor::new(2), stage)
                    .map(|_| ())
                    .unwrap_err(),
                table,
                "{stage:?}"
            );
        }
        let descriptor = VisibleFunctionTableDescriptor::new(2);
        let table = state
            .new_visible_function_table(&descriptor, fragment)
            .unwrap();
        table.set_function(Some(&handle), 0).unwrap();
        let vertex_table = state
            .new_visible_function_table(&descriptor, RenderStages::VERTEX)
            .unwrap();
        assert_eq!(
            vertex_table.set_function(Some(&handle), 0).unwrap_err(),
            Error::Invalid(Invalid::new(
                "set_function",
                Problem::FunctionTable(FunctionTableProblem::HandleOfAnotherState)
            ))
        );

        // The function reads its argument buffer at fragment buffer 0: a table there reads as an
        // unbound buffer and draws, and one at an index it does not read draws Metal's texels.
        let draw_with = |index: u32| {
            let render = &case.render;
            let members = Members::of(device, render.fragment.as_ref().unwrap());
            let target = &render.color_targets[&0];
            let [width, height] = target.dimensions;
            let mut texture_descriptor = TextureDescriptor::texture_2d(
                pixel_format(target.format),
                width.into(),
                height.into(),
                false,
            );
            texture_descriptor.usage = TextureUsage::RENDER_TARGET;
            texture_descriptor.resource_options = ResourceOptions::shared();
            let texture = device.new_texture(&texture_descriptor).unwrap();
            let mut pass = RenderPassDescriptor::new();
            pass.color_attachments[0].texture = Some(texture.clone());
            pass.color_attachments[0].load_action = LoadAction::Clear;
            pass.color_attachments[0].store_action = StoreAction::Store;
            let command_buffer = device
                .new_command_queue()
                .unwrap()
                .command_buffer()
                .unwrap();
            let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
            encoder.set_render_pipeline_state(&state).unwrap();
            for (&owner, buffer) in &members.owners {
                encoder.set_fragment_buffer(Some(buffer), 0, owner).unwrap();
            }
            encoder
                .use_resources(&members.resources(), ResourceUsage::READ, fragment)
                .unwrap();
            encoder
                .set_fragment_visible_function_table(Some(&table), index)
                .unwrap();
            let drawn = encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0);
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            drawn.map(|()| output(&case, &texture, &members))
        };
        draw_with(0).unwrap();
        assert_eq!(draw_with(5).unwrap(), metal);
    });
}
