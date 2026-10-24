//! Device tests of the runtime on the device the cfg selects. Each test fails on any validation
//! message; a lane run with the layer's synchronization validation on
//! (`METAL2VULKAN_LANE_VALIDATION=sync`, `LaneChecks`) fails a test on a missing barrier too.
//!
//! Each claim of what the validation layer reported is stated through
//! [`layer_reported_nothing`](super::layer_reported_nothing), against what the device's instance
//! enabled of the layers it asked for: so a device the loader could not give the layer to says in
//! its test's output that the claim was not exercised there, rather than passing by reporting
//! nothing at all.

use super::{
    bound_buffers, buffer_with, buffer_with_options, encode_case, layer_reported_nothing,
    pipeline_state, public_case, read_buffer, run_case, validated_device, CaseResources, Messages,
    ScratchCache,
};
use crate::air_library::AirLibrary;
use crate::gpu::{DeviceType, TEST_DEVICE};
use metal2vulkan::bind::StatusReport;
use metal2vulkan::metal::{BarrierScope, DispatchType};
use metal2vulkan_runtime::{
    Buffer, CommandBufferStatus, CpuCacheMode, Device, DeviceKind, Error, HazardTrackingMode,
    Invalid, Nil, PipelineOptions, Problem, ResourceOptions, Size, StorageMode,
};
use std::collections::BTreeMap;
use std::time::Duration;

/// The longest a command buffer of these tests may take.
const WAIT: Duration = Duration::from_secs(60);

/// `f` of `buffer`'s contents, its `length` bytes.
fn with_bytes<R>(buffer: &metal2vulkan_runtime::Buffer, f: impl FnOnce(&mut [u8]) -> R) -> R {
    let contents = buffer.contents().expect("a Shared buffer has contents");
    // SAFETY: the buffer holds `length` bytes at `contents`, alive with the buffer, and no
    // command buffer of these tests uses them while `f` runs.
    f(unsafe { std::slice::from_raw_parts_mut(contents.as_ptr(), buffer.length() as usize) })
}

/// Whether every byte of `buffer` is zero.
fn all_zero(buffer: &metal2vulkan_runtime::Buffer) -> bool {
    with_bytes(buffer, |bytes| bytes.iter().all(|&byte| byte == 0))
}

fn invalid(call: &'static str, problem: Problem) -> Error {
    Error::Invalid(Invalid::new(call, problem))
}

/// The runtime creates a device of the selected kind under the validation layer; a new Shared
/// buffer reads zero, also in memory a dropped buffer wrote; the CPU fills one and reads it back
/// after a committed command buffer (an encoder with a scope barrier) completed and ran its
/// completed handler; a Private buffer is made and cleared on the device; encoder misuse is
/// refused typed; and once everything is dropped (the device last), the layer reported nothing,
/// so no Vulkan object leaked. The test prints the features the device enabled.
#[test]
fn a_runtime_device_makes_fills_and_reads_buffers() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let kind = device.properties().kind;
        match TEST_DEVICE {
            DeviceType::Cpu => assert_eq!(kind, DeviceKind::Cpu),
            DeviceType::Gpu => assert_ne!(kind, DeviceKind::Cpu),
        }
        assert_ne!(device.registry_id(), 0);
        eprintln!("runtime device: {} ({kind:?})", device.name());
        // The lanes read what a device of each venue enables at `vkCreateDevice` here.
        eprintln!("runtime device enables: {:?}", device.enabled().features);

        // A buffer's memory, written and freed, reads zero in the next buffer placed there.
        let first = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
        assert!(all_zero(&first));
        with_bytes(&first, |bytes| bytes.fill(0xa5));
        let reused = first.gpu_address();
        drop(first);
        let buffer = device.new_buffer(1000, ResourceOptions::shared()).unwrap();
        assert_eq!(
            buffer.gpu_address(),
            reused,
            "the freed address range is reused"
        );
        assert!(all_zero(&buffer));
        assert_eq!(buffer.length(), 1000);
        assert!(buffer.gpu_address().0 >= 1 << 32);
        with_bytes(&buffer, |bytes| {
            for (index, byte) in bytes.iter_mut().enumerate() {
                *byte = index as u8;
            }
        });

        let private = device
            .new_buffer(3 << 20, ResourceOptions::private())
            .unwrap();
        assert!(private.contents().is_none());
        assert_ne!(private.gpu_address(), buffer.gpu_address());
        assert_ne!(private, buffer);
        assert_eq!(private.clone(), private);
        assert_eq!(
            device.new_buffer(0, ResourceOptions::shared()).err(),
            Some(Error::Nil(Nil::new(
                "new_buffer",
                Problem::BufferLength { length: 0 }
            )))
        );

        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let (sender, receiver) = std::sync::mpsc::channel();
        command_buffer
            .add_completed_handler(Box::new(move |done| {
                let _ = sender.send(done.status());
            }))
            .unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        assert_eq!(
            command_buffer.compute_command_encoder().err(),
            Some(invalid("compute_command_encoder", Problem::EncoderOpen))
        );
        assert_eq!(
            command_buffer.commit(),
            Err(invalid("commit", Problem::EncoderOpen))
        );
        encoder.memory_barrier_with_scope(BarrierScope::BUFFERS);
        encoder.memory_barrier_with_scope(BarrierScope::RENDER_TARGETS);
        encoder.end_encoding().unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::NotEnqueued);
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert_eq!(command_buffer.error(), None);
        assert_eq!(
            receiver.recv_timeout(Duration::ZERO),
            Ok(CommandBufferStatus::Completed),
            "the completed handler ran before the wait returned"
        );
        assert_eq!(
            command_buffer.commit(),
            Err(invalid("commit", Problem::AlreadyCommitted))
        );
        assert_eq!(
            command_buffer.add_completed_handler(Box::new(|_| {})).err(),
            Some(invalid(
                "add_completed_handler",
                Problem::HandlerAfterCommit
            ))
        );
        with_bytes(&buffer, |bytes| {
            for (index, &byte) in bytes.iter().enumerate() {
                assert_eq!(byte, index as u8);
            }
        });

        // An encoder dropped without `end_encoding` loses its commands: the command buffer
        // cannot be committed.
        let abandoned = queue.command_buffer().unwrap();
        drop(abandoned.compute_command_encoder().unwrap());
        assert_eq!(
            abandoned.commit(),
            Err(invalid("commit", Problem::EncoderAbandoned))
        );

        // An empty command buffer commits and completes.
        let empty = queue.command_buffer().unwrap();
        empty.commit().unwrap();
        empty.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(empty.status(), CommandBufferStatus::Completed);
    }
    layer_reported_nothing(
        &messages,
        "no Vulkan object leaked, and every command was valid",
    );
}

/// More device lifetimes, one after another in one process, than NVIDIA's Linux driver can be
/// loaded for without the runtime's keeper (its 27th load fails).
const LIFETIMES: usize = 40;

/// A program makes and drops devices one after another as often as it likes, as on Metal: once
/// it made a device, the runtime keeps the Vulkan drivers loaded, so the loader never loads one
/// again. Each device runs an empty command buffer and is gone when its drop returns: its
/// instance, the only other holder of the lifetime's message list, was destroyed (the instance
/// the runtime keeps for the drivers is bare and holds none).
#[test]
fn forty_devices_one_after_another_all_open_and_go() {
    let cache = ScratchCache::new().unwrap();
    for lifetime in 0..LIFETIMES {
        let messages = Messages::default();
        {
            let device = validated_device(TEST_DEVICE, &messages, cache.path())
                .unwrap_or_else(|error| panic!("device {lifetime}: {error}"));
            let queue = device.new_command_queue().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        }
        assert_eq!(
            messages.holders(),
            1,
            "device {lifetime} was not destroyed when its drop returned"
        );
        layer_reported_nothing(
            &messages,
            &format!("device {lifetime} leaked no Vulkan object"),
        );
    }
}

/// Run `test` on a validated device of the lane, then fail on any validation message (a leaked
/// object is reported when the device is destroyed, after `test` dropped everything), and say what
/// that holds: nothing where the device's instance could not enable the validation layer
/// ([`layer_reported_nothing`]). `test` may take the messages reported so far itself, to fail at
/// the step that caused them.
fn on_device(test: impl FnOnce(&Device, &Messages)) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        test(&device, &messages);
    }
    layer_reported_nothing(
        &messages,
        "no Vulkan object leaked, and every command was valid",
    );
}

/// The committed public case `name` runs through the runtime as a Metal program would run it,
/// and its output is the output its committed Metal observation records, with a clean execution
/// report.
fn matches_metal(name: &str) {
    on_device(|device, _| {
        let case = public_case(name).unwrap();
        let run = run_case(device, &case).unwrap();
        assert_eq!(run.report, StatusReport::default(), "{name}");
        assert!(
            run.output == case.metal_output,
            "{name}: the output differs from Metal's:\n runtime {:02x?}\n metal   {:02x?}",
            run.output,
            case.metal_output
        );
    });
}

/// Two buffers, `dispatchThreads` of a grid of one threadgroup, integer intrinsics.
#[test]
fn integers_at_their_edges_match_metal() {
    matches_metal("saturating-halving-and-wide-product-integers-at-their-edges");
}

/// Dynamic threadgroup memory (`set_threadgroup_memory_length`).
#[test]
fn threadgroup_memory_matches_metal() {
    matches_metal("threadgroup-word");
}

/// A `dispatchThreads` grid with ragged edges: several Vulkan regions of one Metal dispatch.
#[test]
fn a_ragged_thread_grid_matches_metal() {
    matches_metal("a-ragged-thread-grid-observes-each-threadgroups-size-and-position");
}

/// Kernels specialized with function constant values (`newFunctionWithName:constantValues:`): a
/// gate folded at zero, the declaration a constant selects (float and half), a threadgroup width
/// the constant declares, and two matrix multiplies whose shapes and modes constants fix.
#[test]
fn kernels_specialized_by_function_constants_match_metal() {
    for name in [
        "fold-a-zero-function-constant-gate",
        "read-one-slot-through-its-float-declaration",
        "read-one-slot-through-its-half-declaration",
        "read-a-threadgroup-width-the-pipeline-declares",
        "matmul2d-edge-float-partial-13x11",
        "matmul2d-int8-into-int32-nn-device-device-whole-tile",
    ] {
        matches_metal(name);
    }
}

/// The owned `stage_in_regions` cases, whose dispatch sets a stage-in region smaller than the grid
/// or of size 0: each thread past the region reads the region's last element, and the runtime
/// computes what Metal does (`runtime_lane::stage_in_regions` runs the measured probes' other
/// steps).
#[test]
fn stage_in_region_cases_match_metal() {
    for name in [
        "stage-in-threads-past-the-region-read-its-last-element",
        "stage-in-region-origin-adds-to-the-clamped-position",
        "stage-in-region-of-size-zero-clamps-nothing",
        "stage-in-region-clamps-x-whatever-the-row",
    ] {
        matches_metal(name);
    }
}

/// A kernel reading its `[[stage_in]]` attributes, which its pipeline state's stage-input
/// descriptor fetches (`stageInputDescriptor`) at the stage-in region its encoder sets
/// (`setStageInRegion:`, the oracle's grid from the origin), computes what Metal does. With no
/// region set, its dispatch is refused typed, as the binder refuses one (`MissingStageInRegion`),
/// and records nothing.
#[test]
fn a_kernel_reads_its_stage_inputs_as_metal() {
    const NAME: &str = "kernel-stage-input-word";
    matches_metal(NAME);
    on_device(|device, _| {
        let case = public_case(NAME).unwrap();
        let pipeline = pipeline_state(device, &case).unwrap();
        let resources = CaseResources::new(device, &case).unwrap();
        let buffers = bound_buffers(device, &case.compute, &resources).unwrap();
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&pipeline).unwrap();
        for (&index, buffer) in &buffers {
            encoder.set_buffer(Some(buffer), 0, index).unwrap();
        }
        assert_eq!(
            encoder
                .dispatch_threads(Size::new(1, 1, 1), Size::new(1, 1, 1))
                .err(),
            Some(Error::Bind(
                metal2vulkan::error::BindError::MissingStageInRegion
            ))
        );
        encoder.end_encoding().unwrap();
        assert!(command_buffer.executions().is_empty());
    });
}

/// Attribute strides on a compute encoder (`setBuffer:offset:attributeStride:atIndex:`,
/// `setBuffers:offsets:attributeStrides:withRange:`, `setBufferOffset:attributeStride:atIndex:`).
/// `MTLAttributeStrideStatic` binds as the plain setters do. The stage-in kernel's layout has a
/// fixed stride, so it dispatches whatever stride its buffers are bound with and reads its own,
/// as Metal does (`stage_in_strides` reads dynamic layouts). A stride past 32 bits binds its low 32
/// bits; slices of different lengths are refused `Invalid`.
#[test]
fn compute_attribute_strides_bind_as_metal_reads_them() {
    use metal2vulkan_runtime::Region;
    const STATIC: u64 = u64::MAX;
    on_device(|device, _| {
        let case = public_case("kernel-stage-input-word").unwrap();
        let pipeline = pipeline_state(device, &case).unwrap();
        let resources = CaseResources::new(device, &case).unwrap();
        let buffers = bound_buffers(device, &case.compute, &resources).unwrap();
        let (first, last) = (
            *buffers.keys().next().unwrap(),
            *buffers.keys().last().unwrap(),
        );
        let slots: Vec<Option<&Buffer>> = (first..=last).map(|i| buffers.get(&i)).collect();
        let offsets = vec![0; slots.len()];
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&pipeline).unwrap();
        encoder
            .set_stage_in_region(Region::new_3d(0, 0, 0, 1, 1, 1))
            .unwrap();
        let one = Size::new(1, 1, 1);
        // Static strides: the plain binding.
        encoder
            .set_buffers_with_strides(&slots, &offsets, &vec![STATIC; slots.len()], first)
            .unwrap();
        for (&index, buffer) in &buffers {
            encoder
                .set_buffer_with_stride(Some(buffer), 0, STATIC, index)
                .unwrap();
        }
        encoder.dispatch_threads(one, one).unwrap();
        // A per-bind stride on every buffer, the stage-in source among them.
        encoder
            .set_buffers_with_strides(&slots, &offsets, &vec![16; slots.len()], first)
            .unwrap();
        encoder.dispatch_threads(one, one).unwrap();
        // The plain offset form clears the stride, and so does the plain range form.
        for &index in buffers.keys() {
            encoder.set_buffer_offset(0, index).unwrap();
        }
        encoder.dispatch_threads(one, one).unwrap();
        encoder
            .set_buffers_with_strides(&slots, &offsets, &vec![16; slots.len()], first)
            .unwrap();
        encoder.dispatch_threads(one, one).unwrap();
        encoder.set_buffers(&slots, &offsets, first).unwrap();
        encoder.dispatch_threads(one, one).unwrap();
        // The offset form sets a stride, and `MTLAttributeStrideStatic` there clears it.
        for &index in buffers.keys() {
            encoder.set_buffer_offset_with_stride(0, 32, index).unwrap();
        }
        encoder.dispatch_threads(one, one).unwrap();
        for &index in buffers.keys() {
            encoder
                .set_buffer_offset_with_stride(0, STATIC, index)
                .unwrap();
        }
        encoder.dispatch_threads(one, one).unwrap();
        // Strides past 32 bits bind their low 32 bits.
        encoder
            .set_buffer_with_stride(Some(&buffers[&first]), 0, (1 << 32) + 16, first)
            .unwrap();
        encoder
            .set_buffer_offset_with_stride(0, 1 << 32, first)
            .unwrap();
        encoder.dispatch_threads(one, one).unwrap();
        // Refusals.
        assert_eq!(
            encoder.set_buffers_with_strides(&slots, &offsets, &[], first),
            Err(invalid("set_buffers_with_strides", Problem::LengthMismatch))
        );
        encoder.end_encoding().unwrap();
        // Every dispatch ran.
        assert_eq!(command_buffer.executions().len(), 8);
    });
}

/// `setStageInRegionWithIndirectBuffer:indirectBufferOffset:`: the stage-in kernel's dispatch
/// reads its region from a buffer as it runs, after a blit earlier in the command buffer wrote
/// it there (the buffer is zero when encoded), and writes what the same dispatch with the region
/// set directly writes, which is Metal's (`a_kernel_reads_its_stage_inputs_as_metal`). A
/// `set_stage_in_region` after it replaces it. An offset that is not a multiple of 4 reads at the
/// multiple below, as plain Metal reads it; a region past the buffer from there is refused
/// `Invalid`.
#[test]
fn an_indirect_stage_in_region_reads_as_the_direct_one() {
    use metal2vulkan::bind::Dispatch;
    use metal2vulkan_runtime::Region;
    const NAME: &str = "kernel-stage-input-word";
    let size = |axes: [u32; 3]| Size::new(axes[0].into(), axes[1].into(), axes[2].into());
    on_device(|device, _| {
        let case = public_case(NAME).unwrap();
        let Dispatch::Threads {
            threads,
            threads_per_group,
        } = case.compute.dispatch
        else {
            panic!("{NAME} dispatches a grid of threads");
        };
        let pipeline = pipeline_state(device, &case).unwrap();
        // The region's words at offset 4: origin 0, then the grid.
        let words: Vec<u8> = [0, 0, 0, 0, threads[0], threads[1], threads[2]]
            .iter()
            .flat_map(|w: &u32| w.to_le_bytes())
            .collect();
        let run = |indirect: Option<u64>| -> Vec<Vec<u8>> {
            let resources = CaseResources::new(device, &case).unwrap();
            let buffers = bound_buffers(device, &case.compute, &resources).unwrap();
            let source = buffer_with(device, &words).unwrap();
            let region = buffer_with(device, &vec![0; words.len()]).unwrap();
            let queue = device.new_command_queue().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_from_buffer_to_buffer(&source, 0, &region, 0, words.len() as u64)
                .unwrap();
            blit.end_encoding().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder.set_compute_pipeline_state(&pipeline).unwrap();
            for (&index, buffer) in &buffers {
                encoder.set_buffer(Some(buffer), 0, index).unwrap();
            }
            if let Some(offset) = indirect {
                // Replaced by the indirect region, which a direct one would replace in turn.
                encoder
                    .set_stage_in_region(Region::new_3d(0, 0, 0, 1, 1, 1))
                    .unwrap();
                encoder
                    .set_stage_in_region_indirect(&region, offset)
                    .unwrap();
            } else {
                let [width, height, depth] = threads.map(u64::from);
                encoder
                    .set_stage_in_region(Region::new_3d(0, 0, 0, width, height, depth))
                    .unwrap();
            }
            encoder
                .dispatch_threads(size(threads), size(threads_per_group))
                .unwrap();
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            assert_eq!(command_buffer.error(), None);
            buffers
                .values()
                .map(|buffer| read_buffer(buffer, 0, buffer.length()).unwrap())
                .collect()
        };
        let direct = run(None);
        assert_eq!(run(Some(4)), direct);
        // At the multiple of 4 below, as plain Metal reads it: past the buffer from 6, not from 4.
        assert_eq!(run(Some(6)), direct);
        let region = buffer_with(device, &words).unwrap();
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        assert!(matches!(
            encoder.set_stage_in_region_indirect(&region, 8),
            Err(Error::Invalid(_))
        ));
        encoder.end_encoding().unwrap();
    });
}

/// A kernel and the functions its descriptor links (`newComputePipelineStateWithDescriptor:`,
/// `MTLComputePipelineDescriptor.linkedFunctions`). The owned `linked_visible_caller` calls
/// `linked_scale` and `linked_blend`, which `linked_visible_functions` defines, and `linked_scale`
/// calls `linked_offset`, which the caller's own module defines. Each linked function is made with
/// a value for a constant it does not declare, which Metal ignores, and so does the runtime.
/// Linked, the kernel computes what Metal does (`tests/engine_linking.rs`): `words[i] = 3i + (i
/// & 7)` and `floats[i] = i * 0.5 + 1`. With nothing linked, the state is refused for the first
/// call no linked function supplies; a name the linked `functions` repeat, and a kernel the compile
/// layer already compiled, are refused typed, and one function in `functions` and
/// `binary_functions` is linked once, as plain Metal takes it.
#[test]
fn a_kernel_runs_the_functions_its_descriptor_links() {
    use metal2vulkan::error::{Error as CompileError, RuntimeInput};
    use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
    use metal2vulkan::metal::DataType;
    use metal2vulkan::pipeline::{
        ComputePipelineDescriptor, FunctionDescriptor, LinkedFunctions, StageSource,
    };
    use metal2vulkan_devtools::{engine_dir, llvm_assemble};
    const N: u32 = 32;
    let fixture = |stem: &str| llvm_assemble(&engine_dir().join(format!("{stem}.ll"))).unwrap();
    let (caller, functions) = (
        fixture("linked_visible_caller"),
        fixture("linked_visible_functions"),
    );
    on_device(|device, _| {
        let caller = device.new_library_with_air(&caller).unwrap();
        let functions = device.new_library_with_air(&functions).unwrap();
        let undeclared = FunctionConstantValues::defaults().with_index(
            5,
            FunctionConstantValue::new(DataType::UInt, 1u32.to_le_bytes()),
        );
        let linked = |library: &metal2vulkan_runtime::Library, name: &str| {
            library
                .new_function_with_name_constant_values(name, &undeclared)
                .unwrap()
                .linked_function()
        };
        let kernel = caller
            .new_function_with_name("linked_visible_caller")
            .unwrap();
        let descriptor = |linked: LinkedFunctions| {
            ComputePipelineDescriptor::new(StageSource::Function(
                kernel.function_descriptor().with_linked(linked),
            ))
        };
        let defaults = PipelineOptions::default();
        match device.new_compute_pipeline_state(&descriptor(LinkedFunctions::new()), &defaults) {
            Err(Error::Compile(CompileError::NeedsInput(RuntimeInput::LinkedFunction {
                symbol,
            }))) => assert_eq!(symbol.to_string(), "linked_scale"),
            other => panic!("nothing linked: {other:?}"),
        }
        let zeros = vec![0u8; N as usize * 4];
        let queue = device.new_command_queue().unwrap();
        // The dispatch of `pipeline` computes what Metal's does.
        let runs = |pipeline: &metal2vulkan_runtime::ComputePipelineState| {
            let (words, floats) = (
                buffer_with(device, &zeros).unwrap(),
                buffer_with(device, &zeros).unwrap(),
            );
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder.set_compute_pipeline_state(pipeline).unwrap();
            encoder.set_buffer(Some(&words), 0, 0).unwrap();
            encoder.set_buffer(Some(&floats), 0, 1).unwrap();
            encoder
                .dispatch_threads(Size::new(N.into(), 1, 1), Size::new(N.into(), 1, 1))
                .unwrap();
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            let values = |buffer: &Buffer| {
                read_buffer(buffer, 0, u64::from(N) * 4)
                    .unwrap()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                values(&words),
                (0..N).map(|i| 3 * i + (i & 7)).collect::<Vec<_>>()
            );
            assert_eq!(
                values(&floats),
                (0..N)
                    .map(|i| (i as f32 * 0.5 + 1.0).to_bits())
                    .collect::<Vec<_>>()
            );
        };
        let pipeline = device
            .new_compute_pipeline_state(
                &descriptor(
                    LinkedFunctions::new()
                        .with_function(linked(&functions, "linked_scale"))
                        .with_function(linked(&functions, "linked_blend"))
                        .with_private_function(linked(&caller, "linked_offset")),
                ),
                &defaults,
            )
            .unwrap();
        runs(&pipeline);
        // One function in `functions` and `binary_functions`: plain Metal takes it, with one
        // handle, where its API validation stops the descriptor; the state runs it as linked once.
        let mut across = LinkedFunctions::new()
            .with_function(linked(&functions, "linked_scale"))
            .with_function(linked(&functions, "linked_blend"))
            .with_private_function(linked(&caller, "linked_offset"));
        across
            .binary_functions
            .push(linked(&functions, "linked_scale").with_precompiled(true));
        runs(
            &device
                .new_compute_pipeline_state(&descriptor(across), &defaults)
                .unwrap(),
        );
        // A name `functions` repeats: Metal's creation fails ("duplicated symbol").
        let within = LinkedFunctions::new()
            .with_function(linked(&functions, "linked_scale"))
            .with_function(linked(&functions, "linked_blend"))
            .with_function(linked(&functions, "linked_scale"))
            .with_private_function(linked(&caller, "linked_offset"));
        assert_eq!(
            device
                .new_compute_pipeline_state(&descriptor(within), &defaults)
                .err(),
            Some(invalid(
                "new_compute_pipeline_state",
                Problem::DuplicateLinkedFunction {
                    name: "linked_scale".to_owned()
                }
            ))
        );
        // Any compiled kernel: the state refuses it before it reads it.
        let profile = crate::engine::profiles_dir().join("moltenvk-m3.json");
        let description = metal2vulkan::device::DeviceDescription::from_khronos_json(
            &std::fs::read_to_string(profile).unwrap(),
            None,
        )
        .unwrap();
        let compiled = metal2vulkan::device::Device::new(description, Default::default())
            .unwrap()
            .compile_function(
                &FunctionDescriptor::new(
                    metal2vulkan::library::Library::from_air(&fixture("linked_visible_functions"))
                        .unwrap()
                        .function("linked_visible_functions")
                        .unwrap(),
                ),
                &Default::default(),
                &metal2vulkan::Budget::standard(),
            )
            .unwrap();
        assert_eq!(
            device
                .new_compute_pipeline_state(
                    &ComputePipelineDescriptor::new(StageSource::Compiled(compiled)),
                    &defaults,
                )
                .err(),
            Some(invalid(
                "new_compute_pipeline_state",
                Problem::CompiledKernel
            ))
        );
    });
}

/// `case`'s buffers on `device`, made with `options`, holding the case's bytes.
fn case_buffers(
    device: &Device,
    case: &super::PublicCase,
    options: ResourceOptions,
) -> BTreeMap<u32, Buffer> {
    case.compute
        .buffers
        .iter()
        .map(|(&index, bytes)| (index, buffer_with_options(device, bytes, options).unwrap()))
        .collect()
}

/// Chains of two dispatches each leg of the chain test encodes in one command buffer.
const CHAINS: usize = 8;

/// A serial encoder runs its dispatches in order: a dispatch that reads what an earlier one wrote,
/// through tracked buffers and through untracked ones (one class: they may alias), reads the
/// written values, which a run of the second dispatch alone, over the first one's output,
/// reproduces. A concurrent encoder orders the same chains only by the program's scope barriers.
/// Each leg encodes [`CHAINS`] chains over their own buffers, so a missing barrier lets later
/// dispatches overlap earlier ones. The lanes run with synchronization validation, and a leg fails
/// on the messages its own command buffer caused; that validation cannot see accesses through
/// device addresses (a kernel's buffers on a device with `bufferDeviceAddress`), so each chain's
/// values are compared too.
#[test]
fn serial_dispatches_read_what_earlier_dispatches_wrote() {
    on_device(|device, messages| {
        let case =
            public_case("saturating-halving-and-wide-product-integers-at-their-edges").unwrap();
        let pipeline = pipeline_state(device, &case).unwrap();
        let queue = device.new_command_queue().unwrap();
        let output = case.compute.buffers[&1].len() as u64;
        // The second dispatch alone, over the first one's output (Metal's).
        let mut alone = case_buffers(device, &case, ResourceOptions::shared());
        alone.insert(0, buffer_with(device, &case.metal_output).unwrap());
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encode_case(
            &mut encoder,
            &pipeline,
            &case.compute,
            &alone,
            &CaseResources::default(),
        )
        .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let expected = read_buffer(&alone[&1], 0, output).unwrap();
        let legs = [
            (DispatchType::Serial, HazardTrackingMode::Tracked),
            (DispatchType::Serial, HazardTrackingMode::Untracked),
            (DispatchType::Concurrent, HazardTrackingMode::Tracked),
        ];
        for (dispatch_type, hazard) in legs {
            let options =
                ResourceOptions::new(StorageMode::Shared, CpuCacheMode::DefaultCache, hazard);
            // Each chain: input -> b, then b -> c.
            let chains: Vec<_> = (0..CHAINS)
                .map(|_| {
                    let first = case_buffers(device, &case, options);
                    let mut second = case_buffers(device, &case, options);
                    second.insert(0, first[&1].clone());
                    (first, second)
                })
                .collect();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer
                .compute_command_encoder_with_dispatch_type(dispatch_type)
                .unwrap();
            assert_eq!(encoder.dispatch_type(), dispatch_type);
            for (first, second) in &chains {
                encode_case(
                    &mut encoder,
                    &pipeline,
                    &case.compute,
                    first,
                    &CaseResources::default(),
                )
                .unwrap();
                if dispatch_type == DispatchType::Concurrent {
                    encoder.memory_barrier_with_scope(BarrierScope::BUFFERS);
                }
                encode_case(
                    &mut encoder,
                    &pipeline,
                    &case.compute,
                    second,
                    &CaseResources::default(),
                )
                .unwrap();
            }
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            layer_reported_nothing(
                messages,
                &format!("{dispatch_type:?} {hazard:?}: every command was valid"),
            );
            for (chain, (first, second)) in chains.iter().enumerate() {
                let written = read_buffer(&first[&1], 0, output).unwrap();
                assert_eq!(
                    written, case.metal_output,
                    "{dispatch_type:?} {hazard:?} chain {chain}: first"
                );
                let chained = read_buffer(&second[&1], 0, output).unwrap();
                assert_eq!(
                    chained, expected,
                    "{dispatch_type:?} {hazard:?} chain {chain}: second"
                );
            }
        }
    });
}

/// What plain Metal's compute encoder faults at or reads past a buffer for is refused typed and
/// records nothing: a dispatch with no pipeline state, an index past the table, and a dispatch
/// whose kernel reads a slot bound past its buffer's length (the bind itself is held, as Metal's).
/// The command buffer still commits and completes, with a clean execution report and its buffer
/// untouched. The pipeline state answers Metal's queries.
#[test]
fn refused_compute_commands_record_nothing() {
    on_device(|device, _| {
        let case = public_case("threadgroup-word").unwrap();
        let pipeline = pipeline_state(device, &case).unwrap();
        assert_eq!(pipeline.thread_execution_width(), 32);
        let limit = pipeline.max_total_threads_per_threadgroup();
        assert!(limit >= 1);
        assert!(!pipeline.support_indirect_command_buffers());
        let buffer = buffer_with(device, &[0; 64]).unwrap();
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        assert_eq!(encoder.dispatch_type(), DispatchType::Serial);
        let one = Size::new(1, 1, 1);
        assert_eq!(
            encoder.dispatch_threads(one, one),
            Err(invalid("dispatch_threads", Problem::NoPipelineState))
        );
        assert_eq!(
            encoder.set_buffer(Some(&buffer), 0, 31),
            Err(invalid(
                "set_buffer",
                Problem::IndexOutOfRange {
                    index: 31,
                    limit: 31
                }
            ))
        );
        assert_eq!(
            encoder.set_buffers(&[Some(&buffer)], &[], 0),
            Err(invalid("set_buffers", Problem::LengthMismatch))
        );
        encoder.set_compute_pipeline_state(&pipeline).unwrap();
        // The kernel writes its word through slot 1.
        encoder.set_buffer(Some(&buffer), 65, 1).unwrap();
        encoder.set_threadgroup_memory_length(16, 0).unwrap();
        assert_eq!(
            encoder.dispatch_threads(one, one),
            Err(invalid(
                "dispatch_threads",
                Problem::OffsetOutOfRange {
                    offset: 65,
                    length: 64
                }
            ))
        );
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert_eq!(
            command_buffer.execution_report(),
            Some(StatusReport::default())
        );
        assert!(all_zero(&buffer));
    });
}

/// Where Metal's API validation asserts and plain Metal's compute encoder gives a measured
/// answer, the encoder gives it: an offset set at an index with nothing bound changes nothing; a
/// buffer bound past its length at a slot the kernel does not read is held; a threadgroup memory
/// length of 4 is taken; a threadgroup of no threads, of more than the state's maximum or of
/// `2^32 + 1` threads runs nothing, and so does a grid or a threadgroup count of `2^32` on an axis;
/// a grid of `2^32 + 1` threads runs one. The kernel (`threadgroup-word`) writes 42 through its
/// threadgroup memory.
#[test]
fn compute_misuses_answer_as_plain_metal() {
    on_device(|device, _| {
        let case = public_case("threadgroup-word").unwrap();
        let pipeline = pipeline_state(device, &case).unwrap();
        let limit = u64::from(pipeline.max_total_threads_per_threadgroup());
        let poison = [0xab; 4];
        let out = buffer_with(device, &poison).unwrap();
        let spare = buffer_with(device, &[0; 64]).unwrap();
        let queue = device.new_command_queue().unwrap();
        let one = Size::new(1, 1, 1);
        let past = 1u64 << 32;
        for runs in [false, true] {
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder.set_compute_pipeline_state(&pipeline).unwrap();
            encoder.set_buffer_offset(16, 5).unwrap();
            encoder.set_buffer(Some(&spare), 65, 2).unwrap();
            encoder.set_buffer(Some(&out), 0, 1).unwrap();
            encoder.set_threadgroup_memory_length(4, 0).unwrap();
            if runs {
                encoder
                    .dispatch_threads(Size::new(past + 1, 1, 1), one)
                    .unwrap();
            } else {
                encoder.dispatch_threads(one, Size::new(0, 1, 1)).unwrap();
                encoder
                    .dispatch_threadgroups(one, Size::new(limit + 1, 1, 1))
                    .unwrap();
                encoder
                    .dispatch_threadgroups(one, Size::new(past + 1, 1, 1))
                    .unwrap();
                encoder
                    .dispatch_threads(Size::new(past, 1, 1), one)
                    .unwrap();
                encoder
                    .dispatch_threadgroups(Size::new(1, past, 1), one)
                    .unwrap();
            }
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            assert_eq!(
                command_buffer.execution_report(),
                Some(StatusReport::default())
            );
            let expected = if runs { 42u32.to_le_bytes() } else { poison };
            assert_eq!(read_buffer(&out, 0, 4).unwrap(), expected, "runs {runs}");
        }
    });
}
