//! The kernel a case's dispatch runs, compiled as the runtime compiles it.
//!
//! A threadgroup with a partial simdgroup, which the kernel's flattened simd tier cannot run, runs
//! on its emulated-simdgroups variant ([`compile_kernel`]), as the runtime's encoder chooses the
//! module a dispatch runs. The authored compile census compiles a kernel so
//! (`acceptance::PipelineRequest::Kernel`), and a render or mesh pipeline as
//! `Device::compile_render_pipeline` and `Device::compile_mesh_pipeline` compile it. Everything here
//! is device-free.

use metal2vulkan::device::Device;
use metal2vulkan::error::Error;
use metal2vulkan::pipeline::{
    ComputePipeline, ComputePipelineDescriptor, FunctionDescriptor, KernelVariant, LoweringInputs,
    RuntimeFact, StageSource,
};
use metal2vulkan::Budget;

/// Compile `function` into a compute pipeline for `device` under `budget` as the runtime compiles
/// it for a dispatch of threadgroups of `threads_per_threadgroup`: its plain module
/// (`LoweringInputs::new`), or its emulated-simdgroups variant
/// (`KernelVariant::EmulatedSimdgroups`) where the threadgroup's threads are not a multiple of a
/// `RuntimeFact::LocalSizeConstraint` the plain module states (a partial simdgroup its
/// flattened simd tier cannot run), as the runtime's encoder chooses the module a dispatch runs.
pub fn compile_kernel(
    device: &Device,
    function: &FunctionDescriptor,
    threads_per_threadgroup: [u32; 3],
    budget: &Budget,
) -> Result<ComputePipeline, Error> {
    let compile = |inputs: &LoweringInputs| {
        let compiled = device.compile_function(function, inputs, budget)?;
        let descriptor = ComputePipelineDescriptor::new(StageSource::Compiled(compiled));
        device.compile_compute_pipeline(&descriptor, budget)
    };
    let plain = LoweringInputs::new();
    let pipeline = compile(&plain)?;
    let threads = threads_per_threadgroup
        .iter()
        .map(|&axis| u64::from(axis))
        .product::<u64>();
    let partial = pipeline.dependencies.iter().any(|dependency| {
        matches!(dependency.fact, RuntimeFact::LocalSizeConstraint { multiple }
            if !threads.is_multiple_of(u64::from(multiple.max(1))))
    });
    if partial {
        compile(&plain.kernel_variant(KernelVariant::EmulatedSimdgroups))
    } else {
        Ok(pipeline)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate::engine::public_cases::device;

    /// A dispatch whose threadgroup has a partial simdgroup runs the kernel's emulated-simdgroups
    /// variant on a device whose subgroups run whole simdgroups (moltenvk-m3); a full threadgroup
    /// runs the plain module.
    #[test]
    fn a_partial_simdgroup_dispatch_compiles_the_emulated_variant() {
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
        let budget = Budget::standard();
        let under = |inputs: &LoweringInputs| {
            let compiled = device.compile_function(&function, inputs, &budget).unwrap();
            let descriptor = ComputePipelineDescriptor::new(StageSource::Compiled(compiled));
            device
                .compile_compute_pipeline(&descriptor, &budget)
                .unwrap()
        };
        let plain = under(&LoweringInputs::new());
        let emulated =
            under(&LoweringInputs::new().kernel_variant(KernelVariant::EmulatedSimdgroups));
        assert_ne!(plain, emulated);
        for n in [1, 4, 16] {
            let kernel = compile_kernel(&device, &function, [n, 1, 1], &budget).unwrap();
            assert_eq!(kernel, emulated, "{n}");
        }
        let kernel = compile_kernel(&device, &function, [32, 1, 1], &budget).unwrap();
        assert_eq!(kernel, plain);
    }
}

/// The binder's loop of a kernel's compile and its plan for a bind shape: what the binder's own
/// tests compile a case with until the binder goes.
#[cfg(test)]
pub(crate) mod bound {
    use metal2vulkan::bind::{BindShape, Dispatch, TextureShape};
    use metal2vulkan::cache::VariantKey;
    use metal2vulkan::device::{Device, DeviceProfile};
    use metal2vulkan::error::{Error, InternalError, InternalKind, RuntimeInput};
    use metal2vulkan::metal::format::storage_image_choice;
    use metal2vulkan::metal::PixelFormat;
    use metal2vulkan::pipeline::{
        ComputePipeline, ComputePipelineDescriptor, FunctionDescriptor, SlotClass, SlotRef, Stage,
        StageSource,
    };
    use metal2vulkan::Budget;

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
            TextureShape::Texture { pixel_format, .. }
            | TextureShape::Buffer { pixel_format, .. } => Some(*pixel_format),
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
    /// `dispatch_variant`) ([`compile_for_bind_state`], the key's kernel inputs); returns the pipeline
    /// and that key.
    pub fn compile_compute_for_shape(
        device: &Device,
        function: &FunctionDescriptor,
        shape: &BindShape,
        dispatch: Option<&Dispatch>,
        budget: &Budget,
    ) -> Result<(ComputePipeline, VariantKey), Error> {
        compile_kernel_for_shape(device, shape, dispatch, |key| {
            let inputs = key.lowering_inputs(Stage::Kernel);
            let compiled = device.compile_function(function, &inputs, budget)?;
            let descriptor = ComputePipelineDescriptor::new(StageSource::Compiled(compiled));
            device.compile_compute_pipeline(&descriptor, budget)
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
    }
}
