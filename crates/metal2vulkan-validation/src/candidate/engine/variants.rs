//! The lowering inputs a case's bind shape decides.
//!
//! A storage texture's format is a pipeline input on a profile that cannot access the texture
//! without one: the compile asks for it (`NeedsInput(StorageTextureFormat{slot})`), and the plan
//! states the variant a bind shape requires (`BindingPlan::variant`). The consumer answers both
//! from what it binds, as a translation layer does when the guest binds a texture: the slot's
//! `StorageImageChoice` is `metal::format::storage_image_choice` of the bound pixel format, and
//! a plan whose variant differs from the key compiled is compiled again with the plan's key.
//! Nothing is guessed: an unbound slot keeps the product's `NeedsInput`, and a variant that does
//! not settle after one recompile is an internal disagreement between compile and plan.
//!
//! A kernel's dispatch can ask for a variant too: a threadgroup with a partial simdgroup, which the
//! kernel's flattened simd tier cannot run, runs on its emulated-simdgroups variant, and the plan's
//! `dispatch_variant` of the case's dispatch names it (`VariantKey::emulated_simdgroups`), so the
//! compile settles on the key the dispatch runs on.
//!
//! One loop ([`compile_for_bind_state`]) serves both pipeline kinds, in terms of the pipeline's
//! `VariantKey`: a kernel compiles its function under the key's kernel inputs
//! ([`compile_compute_for_shape`]; a tile function the same way, under its attachments' formats
//! too, into a tile pipeline, [`compile_tile_for_shape`]), a render pipeline both stages under the
//! key (`Device::compile_render_variant`, [`compile_render_for_shape`]), so a storage texture of
//! either the vertex or the fragment stage gets its format, and a render plan's pulled vertex
//! buffers are compiled for too. These are what the compile worker runs for the engine
//! candidate (`acceptance::PipelineRequest::BoundCompute`, and `VertexDraw` or `Draw` with a
//! shape): the whole loop runs under the worker's one deadline and memory limit, and the device
//! process only plans the returned pipeline for the same shape. Everything here is device-free.

use metal2vulkan::bind::{BindShape, Dispatch, TextureShape};
use metal2vulkan::cache::VariantKey;
use metal2vulkan::device::{Device, DeviceProfile};
use metal2vulkan::error::{
    DescriptorError, Error, InternalError, InternalKind, MetalAnswer, RuntimeInput,
};
use metal2vulkan::metal::format::storage_image_choice;
use metal2vulkan::metal::PixelFormat;
use metal2vulkan::pipeline::{
    ComputePipeline, ComputePipelineDescriptor, FunctionDescriptor, MeshPipelineDescriptor,
    RenderPipeline, RenderPipelineDescriptor, StageInputDescriptor, StageSource,
    TileRenderPipelineDescriptor,
};
use metal2vulkan::pipeline::{SlotClass, SlotRef, Stage};
use metal2vulkan::Budget;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Whether a pipeline compiled under `compiled` is the one `variant` asks for.
pub fn compiled_for(compiled: &VariantKey, variant: &VariantKey) -> bool {
    compiled == variant
}

/// The pixel format of the texture or texture buffer `shape` binds at texture slot `slot`.
pub fn bound_format(shape: &BindShape, slot: SlotRef) -> Option<PixelFormat> {
    if slot.class != SlotClass::Texture {
        return None;
    }
    match shape.stages.get(&slot.stage)?.textures.get(&slot.index)? {
        TextureShape::Texture { pixel_format, .. } | TextureShape::Buffer { pixel_format, .. } => {
            Some(*pixel_format)
        }
        _ => None,
    }
}

/// Compile with the variant a bind decides: `compile` under the empty key first; each
/// `NeedsInput(StorageTextureFormat{slot})` adds the storage-image choice of the format
/// `bound_format` reports at that slot (of whichever stage) and compiles again; then `variant`
/// plans the pipeline (`None` when the plan refuses the bind, which the caller's own plan then
/// reports), and a plan asking for a different key compiles once more under that key. Returns
/// the pipeline and the key it was compiled under.
///
/// Every retry adds a slot the key did not have, or is the one recompile for the plan's key, so
/// the loop ends.
pub fn compile_for_bind_state<P>(
    bound_format: impl Fn(SlotRef) -> Option<PixelFormat>,
    profile: &DeviceProfile,
    mut compile: impl FnMut(&VariantKey) -> Result<P, Error>,
    mut variant: impl FnMut(&P) -> Option<VariantKey>,
) -> Result<(P, VariantKey), Error> {
    let mut key = VariantKey::new();
    let mut recompiled = false;
    loop {
        let pipeline = match compile(&key) {
            Ok(pipeline) => pipeline,
            Err(Error::NeedsInput(RuntimeInput::StorageTextureFormat { slot }))
                if !key.storage_images.contains_key(&slot) =>
            {
                let Some(format) = bound_format(slot) else {
                    return Err(Error::NeedsInput(RuntimeInput::StorageTextureFormat {
                        slot,
                    }));
                };
                key.storage_images
                    .insert(slot, storage_image_choice(format, profile)?);
                continue;
            }
            Err(error) => return Err(error),
        };
        let Some(wanted) = variant(&pipeline) else {
            return Ok((pipeline, key));
        };
        if compiled_for(&key, &wanted) {
            return Ok((pipeline, key));
        }
        if recompiled {
            return Err(Error::Internal(InternalError::new(
                InternalKind::Invariant,
                format!("the plan asks for variant {wanted:?} of the pipeline compiled for it"),
            )));
        }
        recompiled = true;
        key = wanted;
    }
}

/// Compile `function` into a compute pipeline for `device` under `budget` with the inputs the
/// bind shape `shape` decides, and those `dispatch` runs on when given (its plan's
/// `dispatch_variant`) ([`compile_for_bind_state`], the key's kernel inputs), in a pipeline of
/// stage-input descriptor `stage_input`, which the kernel compiles under too; returns the pipeline
/// and that key.
pub fn compile_compute_for_shape(
    device: &Device,
    function: &FunctionDescriptor,
    shape: &BindShape,
    dispatch: Option<&Dispatch>,
    stage_input: Option<&StageInputDescriptor>,
    budget: &Budget,
) -> Result<(ComputePipeline, VariantKey), Error> {
    compile_kernel_for_shape(device, shape, dispatch, |key| {
        let mut inputs = key.lowering_inputs(Stage::Kernel);
        inputs.stage_input = stage_input.cloned();
        let compiled = device.compile_function(function, &inputs, budget)?;
        let mut descriptor = ComputePipelineDescriptor::new(StageSource::Compiled(compiled));
        descriptor.stage_input = stage_input.cloned();
        device.compile_compute_pipeline(&descriptor, budget)
    })
}

/// Compile `function` into the device-replay variant of a template for `device` under `budget`, for
/// the bind shape `shape` (which binds no kernel buffer) and the variant's indirect `dispatch`, as
/// [`compile_compute_for_shape`] does: under the key's kernel inputs with
/// `LoweringInputs::replay_bindings` set. Returns the pipeline and that key.
pub fn compile_replay_for_shape(
    device: &Device,
    function: &FunctionDescriptor,
    shape: &BindShape,
    dispatch: Option<&Dispatch>,
    budget: &Budget,
) -> Result<(ComputePipeline, VariantKey), Error> {
    compile_kernel_for_shape(device, shape, dispatch, |key| {
        let mut inputs = key.lowering_inputs(Stage::Kernel);
        inputs.replay_bindings = true;
        let compiled = device.compile_function(function, &inputs, budget)?;
        let descriptor = ComputePipelineDescriptor::new(StageSource::Compiled(compiled));
        device.compile_compute_pipeline(&descriptor, budget)
    })
}

/// Compile the tile function `function` into a tile pipeline (`Device::compile_tile_pipeline`)
/// for `device` under `budget`, its color attachments of `formats` by color index, with the
/// inputs the bind shape `shape` and `dispatch` decide as for [`compile_compute_for_shape`]: the
/// function compiles under the key's kernel inputs and the attachments' formats
/// (`LoweringInputs::tile_attachments`), and the pipeline links it under the same formats. Returns
/// the pipeline and that key.
pub fn compile_tile_for_shape(
    device: &Device,
    function: &FunctionDescriptor,
    shape: &BindShape,
    dispatch: Option<&Dispatch>,
    formats: &BTreeMap<u32, PixelFormat>,
    budget: &Budget,
) -> Result<(ComputePipeline, VariantKey), Error> {
    compile_kernel_for_shape(device, shape, dispatch, |key| {
        let mut inputs = key.lowering_inputs(Stage::Kernel);
        inputs.tile_attachments = Some(formats.clone());
        let compiled = device.compile_function(function, &inputs, budget)?;
        let mut descriptor = TileRenderPipelineDescriptor::new(StageSource::Compiled(compiled));
        for (&index, &format) in formats {
            let attachment = usize::try_from(index)
                .ok()
                .and_then(|index| descriptor.color_attachment_formats.get_mut(index))
                .ok_or(Error::InvalidDescriptor(DescriptorError::ColorAttachment {
                    index,
                    metal: MetalAnswer::EndsProcess,
                }))?;
            *attachment = Some(format);
        }
        device.compile_tile_pipeline(&descriptor, budget)
    })
}

/// [`compile_for_bind_state`] of a kernel pipeline `compile` builds under a key, settling on the
/// variant the plan of `shape` asks for, or the one `dispatch` runs on when given (its plan's
/// `dispatch_variant`).
fn compile_kernel_for_shape(
    device: &Device,
    shape: &BindShape,
    dispatch: Option<&Dispatch>,
    compile: impl FnMut(&VariantKey) -> Result<ComputePipeline, Error>,
) -> Result<(ComputePipeline, VariantKey), Error> {
    compile_for_bind_state(
        |slot| bound_format(shape, slot),
        device.profile(),
        compile,
        // A dispatch the plan refuses leaves the plan's own key: the caller's instantiation of
        // the same dispatch reports the refusal.
        |pipeline| {
            let plan = pipeline.plan(device, shape).ok()?;
            Some(match dispatch {
                Some(dispatch) => plan.dispatch_variant(dispatch).unwrap_or(plan.variant),
                None => plan.variant,
            })
        },
    )
}

/// The form of the variant a render pipeline compiles in for what draws it
/// ([`compile_render_for_shape`]), as the runtime's draws take them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DrawForm {
    /// The variant the plan of the bind shape asks for: a case's draw.
    #[default]
    Bound,
    /// Its large-buffer form (`VariantKey::large_buffer_form`): an executed render command's draw,
    /// which the runtime's host step draws in that form.
    Large,
    /// Its draw replay variant (`VariantKey::replay_bindings`): an execution's commands drawn on
    /// the device, one draw per command, every buffer from the command's slot table, as the
    /// runtime's GPU form of an execution of GPU-written render commands draws them.
    Replay,
    /// Its merged draw replay variant (`VariantKey::replay_bindings` and `merged_draws`): an
    /// execution's commands drawn on the device as one merged draw.
    Merged,
}

impl DrawForm {
    /// Whether this is the plain form, which a request leaves unstated.
    pub fn is_bound(&self) -> bool {
        *self == DrawForm::Bound
    }
}

/// Compile `descriptor` into a render pipeline for `device` under `budget` with the variant the
/// bind shape `shape` decides ([`compile_for_bind_state`] over `Device::compile_render_variant`:
/// storage-texture formats of both stages, the plan's pulled vertex buffers) in `form`; returns
/// the pipeline and that key. In the large-buffer form, as the runtime's host step compiles an
/// executed render command's draw (`VariantKey::large_buffer_form` of the first pipeline's plan,
/// from its dependencies), the plan of the pipeline compiled under it settles on it where that key
/// covers the plan's. In a draw replay's forms the key is the plan's with `replay_bindings` (and
/// `merged_draws` for the merged one) set, as the runtime's state settles a replayed draw.
pub fn compile_render_for_shape(
    device: &Device,
    descriptor: &RenderPipelineDescriptor,
    shape: &BindShape,
    form: DrawForm,
    budget: &Budget,
) -> Result<(RenderPipeline, VariantKey), Error> {
    let mut large_key: Option<VariantKey> = None;
    compile_for_bind_state(
        |slot| bound_format(shape, slot),
        device.profile(),
        |key| device.compile_render_variant(descriptor, key, budget),
        |pipeline| {
            let plan = pipeline.plan(device, shape).ok()?;
            match form {
                DrawForm::Bound => Some(plan.variant),
                DrawForm::Large => {
                    if let Some(key) = large_key.as_ref().filter(|key| key.covers(&plan.variant)) {
                        return Some(key.clone());
                    }
                    let key = plan
                        .variant
                        .large_buffer_form(&pipeline.dependencies, shape);
                    large_key = Some(key.clone());
                    Some(key)
                }
                DrawForm::Replay | DrawForm::Merged => {
                    let mut key = plan.variant;
                    key.replay_bindings = true;
                    key.merged_draws = form == DrawForm::Merged;
                    Some(key)
                }
            }
        },
    )
}

/// The mesh render pipeline `descriptor` compiled for the bind shape `shape` as the runtime compiles
/// a mesh draw's: [`compile_for_bind_state`] of `Device::compile_mesh_variant`, each storage
/// texture's representation from its bound format, until the variant its plan for `shape` asks
/// for is the one compiled.
pub fn compile_mesh_for_shape(
    device: &Device,
    descriptor: &MeshPipelineDescriptor,
    shape: &BindShape,
    budget: &Budget,
) -> Result<(RenderPipeline, VariantKey), Error> {
    compile_for_bind_state(
        |slot| bound_format(shape, slot),
        device.profile(),
        |key| device.compile_mesh_variant(descriptor, key, budget),
        |pipeline| pipeline.plan(device, shape).ok().map(|plan| plan.variant),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate::engine::public_cases::device;
    use std::cell::Cell;

    fn texture(index: u32) -> SlotRef {
        SlotRef {
            stage: Stage::Kernel,
            class: SlotClass::Texture,
            index,
        }
    }

    fn buffer(index: u32) -> SlotRef {
        SlotRef {
            stage: Stage::Kernel,
            class: SlotClass::Buffer,
            index,
        }
    }

    /// A compile that needs a format for `needed` and otherwise returns the key it ran under.
    fn needing(
        needed: SlotRef,
        calls: &Cell<u32>,
    ) -> impl FnMut(&VariantKey) -> Result<VariantKey, Error> + '_ {
        move |key| {
            calls.set(calls.get() + 1);
            if key.storage_images.contains_key(&needed) {
                Ok(key.clone())
            } else {
                Err(Error::NeedsInput(RuntimeInput::StorageTextureFormat {
                    slot: needed,
                }))
            }
        }
    }

    /// The plan of a pipeline compiled under `key` asks for exactly that key.
    fn settled(key: &VariantKey) -> Option<VariantKey> {
        Some(key.clone())
    }

    #[test]
    fn a_needed_storage_format_is_the_bound_textures_choice() {
        let device = device("lavapipe");
        let calls = Cell::new(0);
        let (_, inputs) = compile_for_bind_state(
            |slot| (slot == texture(2)).then_some(PixelFormat::R32Uint),
            device.profile(),
            needing(texture(2), &calls),
            settled,
        )
        .unwrap();
        let choice = storage_image_choice(PixelFormat::R32Uint, device.profile()).unwrap();
        assert_eq!(inputs.storage_images.get(&texture(2)), Some(&choice));
        assert_eq!(calls.get(), 2);
    }

    #[test]
    fn an_unbound_storage_slot_keeps_the_products_needs_input() {
        let device = device("lavapipe");
        let calls = Cell::new(0);
        let error = compile_for_bind_state(
            |_| None,
            device.profile(),
            needing(texture(0), &calls),
            settled,
        )
        .unwrap_err();
        assert_eq!(
            error,
            Error::NeedsInput(RuntimeInput::StorageTextureFormat { slot: texture(0) })
        );
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn a_plan_variant_recompiles_once_under_its_key() {
        let device = device("lavapipe");
        let mut wanted = VariantKey::new();
        wanted.psb_slots.insert(buffer(1));
        let calls = Cell::new(0);
        let (pipeline, inputs) = compile_for_bind_state(
            |_| None,
            device.profile(),
            |inputs| {
                calls.set(calls.get() + 1);
                Ok(inputs.clone())
            },
            |_| Some(wanted.clone()),
        )
        .unwrap();
        assert!(compiled_for(&inputs, &wanted));
        assert_eq!(pipeline, inputs);
        assert_eq!(calls.get(), 2);
    }

    #[test]
    fn a_variant_that_does_not_settle_is_internal() {
        let device = device("lavapipe");
        let calls = Cell::new(0);
        let error = compile_for_bind_state(
            |_| None,
            device.profile(),
            |inputs| {
                calls.set(calls.get() + 1);
                Ok(inputs.clone())
            },
            |compiled: &VariantKey| {
                let mut key = VariantKey::new();
                key.psb_slots
                    .insert(buffer(compiled.psb_slots.len() as u32));
                Some(key)
            },
        )
        .unwrap_err();
        assert!(matches!(error, Error::Internal(_)), "{error:?}");
        assert_eq!(calls.get(), 2);
    }

    /// A plan that refuses the bind ends the loop with the pipeline compiled so far; the caller's
    /// own plan of the same shape reports the refusal.
    #[test]
    fn a_refused_plan_returns_the_pipeline_for_the_callers_plan() {
        let device = device("lavapipe");
        let calls = Cell::new(0);
        let (_, inputs) = compile_for_bind_state(
            |_| None,
            device.profile(),
            |inputs| {
                calls.set(calls.get() + 1);
                Ok(inputs.clone())
            },
            |_| None,
        )
        .unwrap();
        assert_eq!(inputs, VariantKey::new());
        assert_eq!(calls.get(), 1);
    }

    /// A dispatch whose threadgroup has a partial simdgroup compiles the kernel's
    /// emulated-simdgroups variant on a device whose subgroups run whole simdgroups
    /// (moltenvk-m3), whose plan states that key and instantiates the dispatch; a full
    /// threadgroup, and a compile for no dispatch, keep the shape's key.
    #[test]
    fn a_partial_simdgroup_dispatch_compiles_the_emulated_variant() {
        use metal2vulkan::bind::{BindState, BoundBuffer, StageTable};
        use metal2vulkan::library::Library;
        use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};

        let bitcode = match llvm_assemble(&engine_dir().join("simd_ops.ll")) {
            Ok(bitcode) => bitcode,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected simd_ops.ll: {why}"),
        };
        let device = device("moltenvk-m3");
        let library = Library::from_air(&bitcode).unwrap();
        let function = FunctionDescriptor::new(library.function("simd_sum_k").unwrap());
        let mut table = StageTable::default();
        table.buffers.insert(
            0,
            BoundBuffer::Buffer {
                length: 4096,
                offset: 0,
                gpu_address: 0x1_0000,
                device_address: None,
                attribute_stride: None,
            },
        );
        let mut state = BindState::default();
        state.stages.insert(Stage::Kernel, table);
        let shape = state.shape();
        let dispatch = |n| Dispatch::Threadgroups {
            groups: [2, 1, 1],
            threads_per_group: [n, 1, 1],
        };
        let budget = Budget::standard();
        for n in [1, 4, 16] {
            let (pipeline, key) = compile_compute_for_shape(
                &device,
                &function,
                &shape,
                Some(&dispatch(n)),
                None,
                &budget,
            )
            .unwrap();
            assert!(key.emulated_simdgroups, "{n}");
            let plan = pipeline.plan(&device, &shape).unwrap();
            assert!(compiled_for(&key, &plan.variant), "{n}");
            assert!(
                plan.instantiate_dispatch(&state, &dispatch(n)).is_ok(),
                "{n}"
            );
        }
        for d in [Some(dispatch(32)), None] {
            let (pipeline, key) =
                compile_compute_for_shape(&device, &function, &shape, d.as_ref(), None, &budget)
                    .unwrap();
            assert!(key.is_empty(), "{d:?}");
            let plan = pipeline.plan(&device, &shape).unwrap();
            assert!(plan.instantiate_dispatch(&state, &dispatch(32)).is_ok());
            assert!(matches!(
                plan.instantiate_dispatch(&state, &dispatch(16)),
                Err(metal2vulkan::error::BindError::NeedsVariant { .. })
            ));
        }
    }

    /// A fragment function's storage texture gets the format the bind shape binds at its slot, on a
    /// device that cannot access a storage image without one (lavapipe with the formatless-storage
    /// features off), exactly as a kernel's does. The pipeline is the L4 fragment row's (its
    /// generated passthrough companion drawing the function).
    #[test]
    fn a_fragment_storage_texture_is_compiled_for_the_bound_format() {
        use metal2vulkan::bind::{BindState, BoundTexture, TextureFacts};
        use metal2vulkan::device::{DeviceDescription, DevicePolicy};
        use metal2vulkan::library::Library;
        use metal2vulkan::metal::{TextureType, TextureUsage};
        use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};

        let bitcode = match llvm_assemble(&engine_dir().join("fragment_rog.ll")) {
            Ok(bitcode) => bitcode,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected fragment_rog.ll: {why}"),
        };
        let text = std::fs::read_to_string(crate::engine::profiles_dir().join("lavapipe.json"))
            .unwrap()
            .replace(
                "\"shaderStorageImageReadWithoutFormat\": true",
                "\"shaderStorageImageReadWithoutFormat\": false",
            )
            .replace(
                "\"shaderStorageImageWriteWithoutFormat\": true",
                "\"shaderStorageImageWriteWithoutFormat\": false",
            );
        let description = DeviceDescription::from_khronos_json(&text, None).unwrap();
        let device = Device::new(description, DevicePolicy::default()).unwrap();
        assert!(!device.profile().images().storage_write_without_format);

        let row = crate::acceptance::row::fragment_row(
            &bitcode,
            "rog_texture",
            &metal2vulkan::library::FunctionConstantValues::defaults(),
        )
        .unwrap();
        let function = |bitcode: &[u8], entry: &str| {
            let library = Library::from_air(bitcode).unwrap();
            StageSource::Function(FunctionDescriptor::new(library.function(entry).unwrap()))
        };
        let descriptor = row
            .template
            .descriptor(
                function(&row.companion.opaque_bitcode().unwrap(), row.vertex_entry),
                Some(function(&bitcode, "rog_texture")),
            )
            .unwrap();
        let slot = SlotRef {
            stage: Stage::Fragment,
            class: SlotClass::Texture,
            index: 0,
        };
        let budget = Budget::standard();
        assert_eq!(
            device
                .compile_render_variant(&descriptor, &VariantKey::new(), &budget)
                .err(),
            Some(Error::NeedsInput(RuntimeInput::StorageTextureFormat {
                slot
            }))
        );

        let mut facts = TextureFacts::new(TextureType::Type2D, PixelFormat::RGBA32Float);
        facts.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        let mut state = BindState::default();
        state
            .stages
            .entry(Stage::Fragment)
            .or_default()
            .textures
            .insert(
                0,
                BoundTexture::Texture {
                    facts,
                    resource_id: None,
                },
            );
        let (_, key) = compile_render_for_shape(
            &device,
            &descriptor,
            &state.shape(),
            DrawForm::Bound,
            &budget,
        )
        .unwrap();
        let choice = storage_image_choice(PixelFormat::RGBA32Float, device.profile()).unwrap();
        assert_eq!(key.storage_images.get(&slot), Some(&choice));
    }
}
