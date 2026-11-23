//! A kernel that reads what the compute encoder does not supply is refused at its state's make
//! as `Gap::KernelInput` of the kind the runtime found, and the refusal's text names that kind,
//! but for an explicit imageblock's extent, whose state is made and whose dispatch is refused:
//! one test per kind an engine fixture's kernel meets on lavapipe. No engine fixture's kernel
//! meets the other kinds there.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::each_device;
    use crate::{ComputePipelineDescriptor, Error, FunctionDescriptor, Gap, KernelInputKind};
    use crate::{Dispatch, DispatchType, NumberedSampler, StageSource};
    use metal2vulkan::library::Library;
    use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};

    /// The kernel `entry` of the engine fixture `fixture.ll`, or None where `llvm-as` is missing.
    fn descriptor_of(fixture: &str, entry: &str) -> Option<ComputePipelineDescriptor> {
        let bitcode = match llvm_assemble(&engine_dir().join(format!("{fixture}.ll"))) {
            Ok(bitcode) => bitcode,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return None;
            }
            Err(error) => panic!("{fixture}: {error:?}"),
        };
        let library = Library::from_air(&bitcode).expect("the library");
        let function = library.function(entry).expect("the kernel");
        Some(ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        )))
    }

    /// An index for each sampler the kernel numbers.
    fn sampler_ids(samplers: &[NumberedSampler]) -> Result<Vec<u32>, Error> {
        Ok((0..samplers.len() as u32).collect())
    }

    /// `descriptor`'s state is refused on each device as `Gap::KernelInput` of `kind`, whose text
    /// ends with the kind's name.
    fn refused_as(test: &str, descriptor: &ComputePipelineDescriptor, kind: KernelInputKind) {
        each_device(test, |name, _, device| {
            match device.new_compute_pipeline_state(descriptor, sampler_ids) {
                Err(Error::Unsupported(Gap::KernelInput { kind: found })) => {
                    assert_eq!(found, kind, "{name}: the kind the runtime found");
                    let text = Error::Unsupported(Gap::KernelInput { kind }).to_string();
                    assert!(
                        text.ends_with(&format!("{kind:?}")),
                        "{name}: the refusal's text names its kind: {text}"
                    );
                }
                Ok(_) => panic!("{name}: a state, where the encoder supplies no {kind:?}"),
                Err(error) => panic!("{name}: {error}, not Gap::KernelInput of {kind:?}"),
            }
        });
    }

    /// `a4_imageblock_extent`, a kernel over an explicit imageblock: its module states the
    /// imageblock's size as a runtime fact, which the encoder does not supply (it has no
    /// `setImageblockWidth:height:`). Its state is made, its imageblock length the typed gap; a
    /// pre-pass gets no key of its dispatch, and the dispatch, one threadgroup of the kernel's 4 by
    /// 2, is refused naming that fact.
    #[test]
    fn an_explicit_imageblock_kernel_is_made_and_its_dispatch_refused_as_its_named_runtime_fact() {
        let Some(descriptor) = descriptor_of("a4_imageblock_extent", "a4_imageblock_extent") else {
            return;
        };
        let kind = KernelInputKind::RuntimeFact {
            fact: "ImageblockSize",
        };
        let (groups, threads) = ([1, 1, 1], [4, 2, 1]);
        each_device("kernel_input_imageblock", |name, _, device| {
            let state = (device.new_compute_pipeline_state(&descriptor, sampler_ids))
                .unwrap_or_else(|error| panic!("{name}: no state: {error}"));
            let length = state.imageblock_memory_length_for_dimensions([8, 4, 3]);
            assert!(
                matches!(length, Err(Error::Unsupported(Gap::ImageblockLength))),
                "{name}: the imageblock length {length:?}"
            );
            let dispatch = Dispatch::Threadgroups(groups);
            let key = state.dispatch_variant(dispatch, threads, &[]);
            assert!(key.is_none(), "{name}: a pre-pass key {key:?}");
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder =
                (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
            encoder
                .set_compute_pipeline_state(&state)
                .expect("the state");
            match encoder.dispatch_threadgroups(groups, threads) {
                Err(Error::Unsupported(Gap::KernelInput { kind: found })) => {
                    assert_eq!(found, kind, "{name}: the kind the dispatch found")
                }
                Ok(()) => panic!("{name}: a dispatch, where the encoder sets no imageblock extent"),
                Err(error) => panic!("{name}: {error}, not Gap::KernelInput of {kind:?}"),
            }
            encoder.end_encoding();
            println!(
                "{name}: the imageblock kernel's state made, its dispatch refused as {kind:?}"
            );
        });
    }

    /// `visible_table_dispatch`, a kernel that takes a visible function table at buffer 1: a slot
    /// the encoder does not bind (it has no `setVisibleFunctionTable:atBufferIndex:`).
    #[test]
    fn a_visible_function_table_kernel_is_refused_as_a_slot() {
        let Some(descriptor) = descriptor_of("visible_table_caller", "visible_table_dispatch")
        else {
            return;
        };
        refused_as(
            "kernel_input_function_table",
            &descriptor,
            KernelInputKind::Slot,
        );
    }
}
