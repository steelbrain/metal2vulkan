//! Device tests of argument buffers on the device the cfg selects, under the validation layer:
//! buffers a kernel reaches through addresses held in memory, and textures and sampler states it
//! reaches through resource IDs held in memory, the buffers and textures made resident with
//! `use_resource(s)`.
//!
//! Every committed public kernel case whose argument buffers hold buffers, textures or sampler
//! states runs through the runtime ([`run_case`]: each member's `gpu_address` or
//! `gpu_resource_id` written into its owner, and each buffer and texture resident for reading and
//! writing) against its committed Metal observation, or the engine candidate run afresh where
//! the two differ, as `kernel_textures` judges texture cases. Addresses and resource IDs a kernel
//! copies into an argument buffer reach their resources as the CPU's do. Resource IDs are dense
//! per device and kind of resource, from 1, and a freed one is reused. A command buffer's
//! dispatches share one heap set per pipeline variant, so 2048 dispatches over textures of their
//! own each read their own and the process stays small.

use super::kernel_textures::{absent_read_only, fresh_candidate, refusal};
use super::{
    at_plan, bound_buffers, buffer_with, encode_case, layer_reported_nothing,
    layer_reported_nothing_at_all, pipeline_state, public_case, public_cases, read_buffer,
    run_case, validated_device_with, CaseResources, Messages, ScratchCache,
};
use crate::candidate::engine::ComputeOutput;
use crate::case::{AuthoredCase, Stage};
use crate::gpu::{DeviceType, TEST_DEVICE};
use metal2vulkan::bind::{SamplerDescriptor, StatusReport};
use metal2vulkan::metal::{PixelFormat, ResourceUsage};
use metal2vulkan_runtime::DeviceOptions;
use metal2vulkan_runtime::{
    CommandBuffer, Device, ResourceId, SamplerStateOptions, Size, TextureDescriptor,
};
use std::time::Duration;

/// The longest a command buffer of these tests may take.
const WAIT: Duration = Duration::from_secs(60);

/// One thread copies a 32-bit word from its buffer 0 into its buffer 1.
const WORD_COPIER: &str = "single-thread-copy-word";

/// Walks a pointer chain from the address its argument buffer (buffer 0) holds at offset 0.
const POINTER_WALKER: &str = "walk-a-device-pointer-chain-through-a-loop-phi";

/// Samples an unsigned texture through a nearest sampler, both held in its argument buffer.
const HANDLE_SAMPLER: &str =
    "sample-an-unsigned-texture-through-a-sampler-held-beside-it-in-an-argument-buffer";

/// Run `test` on a validated device of the lane, then fail on any validation message.
fn on_device(test: impl FnOnce(&Device)) {
    on_device_with(|_| {}, test)
}

/// [`on_device`], `adjust` applied to the options of each device it makes (`at_plan` gives them
/// their queue plan).
fn on_device_with(adjust: fn(&mut DeviceOptions), test: impl FnOnce(&Device)) {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), adjust).unwrap();
        test(&device);
    }
    layer_reported_nothing_at_all(&messages);
}

/// Every committed public kernel case whose argument buffers hold buffers runs through the
/// runtime with a clean execution report and Metal's output, or the output of the candidate
/// executor run afresh on the case, or is refused honestly; the validation layer reports nothing.
#[test]
fn kernels_reach_buffers_through_addresses_in_argument_buffers() {
    let census = census_with(at_plan, "buffers", |case| {
        !case.argument_buffer_buffers.is_empty()
    });
    assert!(census.matched > 0, "no case matched Metal");
}

/// Every committed public kernel case whose argument buffers hold textures or sampler states
/// runs as [`kernels_reach_buffers_through_addresses_in_argument_buffers`] judges its cases, its
/// members made, their IDs written into their owners, and its textures made resident.
#[test]
fn kernels_reach_textures_and_samplers_through_ids_in_argument_buffers() {
    let census = census_with(at_plan, "textures and samplers", |case| {
        !case.argument_buffer_textures.is_empty() || !case.argument_buffer_samplers.is_empty()
    });
    assert!(census.matched > 0, "no case matched Metal");
}

/// What [`census_with`] saw: how many cases matched Metal.
struct Census {
    matched: usize,
}

/// Run every committed public kernel case `select` takes, its argument buffers holding `what`:
/// each has a clean execution report and Metal's output, or the output of the candidate executor
/// run afresh on the case, or is refused honestly; the validation layer reports nothing.
///
/// `adjust` is applied to the options of each device it makes (`at_plan` gives them their
/// queue plan).
fn census_with(
    adjust: fn(&mut DeviceOptions),
    what: &str,
    select: impl Fn(&AuthoredCase) -> bool,
) -> Census {
    let cases = public_cases(|case| {
        case.stage == Stage::Kernel && case.public_source.is_some() && select(case)
    })
    .unwrap();
    assert!(!cases.is_empty(), "no committed case holds {what}");
    let messages = Messages::default();
    let (mut matched, mut as_candidate, mut refused, mut failed) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let device_name;
    let mut differing = Vec::new();
    {
        let cache = ScratchCache::new().unwrap();
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), adjust).unwrap();
        device_name = device.name().to_owned();
        for (name, case) in &cases {
            let run = case
                .as_ref()
                .map_err(Clone::clone)
                .and_then(|case| Ok((case, run_case(&device, case)?)));
            match run {
                Ok((_, run)) if absent_read_only(&run.report) => refused.push(format!(
                    "{name}: it read an argument its function constants removed \
                     (FunctionConstantArgument, not representable)"
                )),
                Ok((_, run)) if run.report != StatusReport::default() => {
                    failed.push(format!("{name}: the execution report is {:?}", run.report))
                }
                Ok((case, run)) if run.output == case.metal_output => matched.push(name),
                Ok((case, run)) => differing.push((name, case, run.output)),
                Err(why) if refusal(&why) => refused.push(format!("{name}: {why}")),
                Err(why) => failed.push(format!("{name}: {why}")),
            }
            let reported = messages.take();
            if !reported.is_empty() {
                failed.push(format!(
                    "{name}: the validation layer reported {reported:#?}"
                ));
            }
        }
    }
    let leaked = messages.take();
    for (name, case, output) in differing {
        match fresh_candidate(case) {
            Ok((candidate_device, _, _)) if candidate_device != device_name => {
                failed.push(format!(
                    "{name}: the candidate ran on {candidate_device}, the runtime on {device_name}"
                ))
            }
            Ok((_, candidate, _)) if candidate == output => as_candidate.push(name),
            Ok((_, candidate, _)) => failed.push(format!(
                "{name}: differs from Metal and from the fresh candidate:\n runtime   \
                 {output:02x?}\n candidate {candidate:02x?}\n metal     {:02x?}",
                case.metal_output
            )),
            Err(why) => failed.push(format!("{name}: the fresh candidate gave no output: {why}")),
        }
    }
    eprintln!(
        "runtime argument-buffer {what} on {device_name}: {} match Metal ({matched:?}), {} the \
         fresh candidate ({as_candidate:?}), {} refused, {} failed, of {}",
        matched.len(),
        as_candidate.len(),
        refused.len(),
        failed.len(),
        cases.len()
    );
    for why in &refused {
        eprintln!("refused: {why}");
    }
    executor_census(what, &device_name, &cases);
    assert!(
        failed.is_empty() && leaked.is_empty(),
        "failed:\n{}\nat device destruction: {leaked:#?}",
        failed.join("\n")
    );
    layer_reported_nothing(
        &messages,
        "every case's commands were valid and left nothing behind",
    );
    Census {
        matched: matched.len(),
    }
}

/// Run every case of `cases` through validation's Vulkan executor too, which plans as the runtime
/// does but places heap members as the plan packs them (`HeapPlacement::Planned`), and print how
/// many match Metal: a census line for the venue, which the runtime's ID placement leaves alone.
fn executor_census(what: &str, device_name: &str, cases: &[super::NamedCase]) {
    let (mut matched, mut differ, mut none) = (Vec::new(), Vec::new(), Vec::new());
    for (name, case) in cases {
        let Ok(case) = case else {
            continue;
        };
        match fresh_candidate(case) {
            Ok((_, output, _)) if output == case.metal_output => matched.push(name),
            Ok(_) => differ.push(name),
            Err(why) => none.push(format!("{name}: {why}")),
        }
    }
    eprintln!(
        "runtime argument-buffer {what} through validation's executor on {device_name}: {} match \
         Metal, {} differ ({differ:?}), {} gave no output ({none:?}), of {}",
        matched.len(),
        differ.len(),
        none.len(),
        cases.len()
    );
}

/// Commit `command_buffer` and wait for it to complete.
fn run(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// A kernel copies, one word per dispatch, the address of the pointer walker's buffer from a
/// buffer into the walker's argument buffer, which did not hold it; the walker then reads
/// through it,
/// in the same encoder and in the next one of the same command buffer, its buffer made resident
/// for reading only (`use_resource`). Its output is its run over the CPU's address, Metal's.
#[test]
fn addresses_a_kernel_writes_reach_buffers_as_the_cpus_do() {
    on_device_with(at_plan, |device| {
        let queue = device.new_command_queue().unwrap();
        let copier = public_case(WORD_COPIER).unwrap();
        let walker = public_case(POINTER_WALKER).unwrap();
        let expected = run_case(device, &walker).unwrap().output;
        assert_eq!(
            expected, walker.metal_output,
            "{POINTER_WALKER} over the CPU's address"
        );
        let ComputeOutput::Buffer {
            binding,
            offset,
            length,
        } = walker.compute.output
        else {
            panic!("{POINTER_WALKER}: a buffer output");
        };
        let copy = pipeline_state(device, &copier).unwrap();
        let walk = pipeline_state(device, &walker).unwrap();
        for one_encoder in [true, false] {
            let how = if one_encoder {
                "one encoder"
            } else {
                "two encoders"
            };
            let resources = CaseResources::new(device, &walker).unwrap();
            let [(&field, member)] = resources.argument_buffers.iter().collect::<Vec<_>>()[..]
            else {
                panic!("{POINTER_WALKER}: one argument-buffer buffer");
            };
            let word = member.gpu_address().0.to_le_bytes();
            let address = buffer_with(device, &word).unwrap();
            // The walker's buffers with nothing written into its argument buffer.
            let unwritten = CaseResources {
                argument_buffers: Default::default(),
                ..CaseResources::new(device, &walker).unwrap()
            };
            let buffers = bound_buffers(device, &walker.compute, &unwritten).unwrap();
            let owner = &buffers[&field.0];
            assert_ne!(
                read_buffer(owner, u64::from(field.1), 8).unwrap(),
                word,
                "{how}: the argument buffer starts without the address"
            );
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder.set_compute_pipeline_state(&copy).unwrap();
            encoder.set_buffer(Some(&address), 0, 0).unwrap();
            for word in [0, 4] {
                encoder.set_buffer_offset(word, 0).unwrap();
                encoder
                    .set_buffer(Some(owner), u64::from(field.1) + word, 1)
                    .unwrap();
                let one = Size::new(1, 1, 1);
                encoder.dispatch_threads(one, one).unwrap();
            }
            if !one_encoder {
                encoder.end_encoding().unwrap();
                encoder = command_buffer.compute_command_encoder().unwrap();
            }
            encoder
                .use_resource(member.as_resource(), ResourceUsage::READ)
                .unwrap();
            encode_case(&mut encoder, &walk, &walker.compute, &buffers, &unwritten).unwrap();
            encoder.end_encoding().unwrap();
            run(&command_buffer);
            let output = read_buffer(&buffers[&binding], offset, length).unwrap();
            assert_eq!(output, expected, "{how}");
        }
    });
}

/// A device's textures, views included, and its sampler states have resource IDs, dense from 1
/// in a space per kind, and an ID is reused once its resource is gone; a sampler state made
/// without argument-buffer support has one too, as Metal's has (probe 7). The device's first
/// texture took sampler ID 1 for the device, as Metal's does, so the states start at 2. Each
/// sampler state here has a descriptor of its own: an equal one gets the live state.
#[test]
fn resource_ids_are_dense_per_kind_and_reused() {
    on_device(|device| {
        let descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 4, 4, false);
        let first = device.new_texture(&descriptor).unwrap();
        let second = device.new_texture(&descriptor).unwrap();
        let view = first.new_texture_view(PixelFormat::RGBA8Unorm).unwrap();
        let ids = [&first, &second, &view].map(|texture| texture.gpu_resource_id());
        assert_eq!(ids, [1, 2, 3].map(ResourceId));
        drop(second);
        let third = device.new_texture(&descriptor).unwrap();
        assert_eq!(third.gpu_resource_id(), ResourceId(2), "the lowest free ID");

        let plain = device.new_sampler_state(&SamplerDescriptor::new()).unwrap();
        assert_eq!(plain.gpu_resource_id(), ResourceId(2));
        let mut options = SamplerStateOptions::default();
        options.support_argument_buffers = true;
        let made = |clamp: f32| {
            let mut descriptor = SamplerDescriptor::new();
            descriptor.lod_min_clamp = clamp;
            device
                .new_sampler_state_with_options(&descriptor, options)
                .unwrap()
        };
        let (one, two) = (made(0.0), made(1.0));
        assert_eq!(one.gpu_resource_id(), ResourceId(3));
        assert_eq!(two.gpu_resource_id(), ResourceId(4));
        drop(one);
        assert_eq!(made(2.0).gpu_resource_id(), ResourceId(3));
        drop(plain);
    });
}

/// A kernel copies, one word per dispatch, the resource IDs of a texture and a sampler state
/// from a buffer into an argument buffer, which did not hold them; a kernel then samples the
/// texture through the sampler, both named by the IDs it loads, in the same encoder and in the
/// next one of the same command buffer, the texture made resident for reading only
/// (`use_resource`). Its output is its run over the CPU's IDs, Metal's.
#[test]
fn resource_ids_a_kernel_writes_name_resources_as_the_cpus_do() {
    on_device_with(at_plan, |device| {
        let queue = device.new_command_queue().unwrap();
        let copier = public_case(WORD_COPIER).unwrap();
        let sampler = public_case(HANDLE_SAMPLER).unwrap();
        let expected = run_case(device, &sampler).unwrap().output;
        assert_eq!(
            expected, sampler.metal_output,
            "{HANDLE_SAMPLER} over the CPU's IDs"
        );
        let ComputeOutput::Buffer {
            binding,
            offset,
            length,
        } = sampler.compute.output
        else {
            panic!("{HANDLE_SAMPLER}: a buffer output");
        };
        let copy = pipeline_state(device, &copier).unwrap();
        let sample = pipeline_state(device, &sampler).unwrap();
        for one_encoder in [true, false] {
            let how = if one_encoder {
                "one encoder"
            } else {
                "two encoders"
            };
            let resources = CaseResources::new(device, &sampler).unwrap();
            let [(&texture_field, texture)] =
                resources.argument_textures.iter().collect::<Vec<_>>()[..]
            else {
                panic!("{HANDLE_SAMPLER}: one argument-buffer texture");
            };
            let [(&sampler_field, state)] =
                resources.argument_samplers.iter().collect::<Vec<_>>()[..]
            else {
                panic!("{HANDLE_SAMPLER}: one argument-buffer sampler");
            };
            let sampler_id = state.gpu_resource_id();
            let ids = [
                (texture_field, texture.gpu_resource_id()),
                (sampler_field, sampler_id),
            ];
            let words: Vec<u8> = ids.iter().flat_map(|(_, id)| id.0.to_le_bytes()).collect();
            let source = buffer_with(device, &words).unwrap();
            // The case's buffers with nothing written into its argument buffer; `resources` keeps
            // its texture and sampler state alive.
            let unwritten = CaseResources {
                textures: resources.textures.clone(),
                samplers: resources.samplers.clone(),
                ..CaseResources::default()
            };
            let buffers = bound_buffers(device, &sampler.compute, &unwritten).unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder.set_compute_pipeline_state(&copy).unwrap();
            encoder.set_buffer(Some(&source), 0, 0).unwrap();
            for ((field, id), at) in ids.iter().zip((0u64..).step_by(8)) {
                let owner = &buffers[&field.0];
                assert_ne!(
                    read_buffer(owner, u64::from(field.1), 8).unwrap(),
                    id.0.to_le_bytes(),
                    "{how}: the argument buffer starts without the ID"
                );
                for word in [0, 4] {
                    encoder.set_buffer_offset(at + word, 0).unwrap();
                    encoder
                        .set_buffer(Some(owner), u64::from(field.1) + word, 1)
                        .unwrap();
                    let one = Size::new(1, 1, 1);
                    encoder.dispatch_threads(one, one).unwrap();
                }
            }
            if !one_encoder {
                encoder.end_encoding().unwrap();
                encoder = command_buffer.compute_command_encoder().unwrap();
            }
            encoder
                .use_resource(texture.as_resource(), ResourceUsage::READ)
                .unwrap();
            encode_case(
                &mut encoder,
                &sample,
                &sampler.compute,
                &buffers,
                &unwritten,
            )
            .unwrap();
            encoder.end_encoding().unwrap();
            run(&command_buffer);
            let output = read_buffer(&buffers[&binding], offset, length).unwrap();
            assert_eq!(output, expected, "{how}");
        }
    });
}

/// How many dispatches [`dispatches_of_their_own_textures_share_one_heap_set`] records.
const OWN_TEXTURES: usize = 2048;

/// The most the peak resident set may grow over [`OWN_TEXTURES`] dispatches on a CPU device.
const PEAK_GROWTH: u64 = 256 << 20;

/// The most the runtime's own allocations may grow over [`OWN_TEXTURES`] dispatches: the peak of
/// the process's heap (`worker::alloc`), and the `VkDeviceMemory` the runtime allocated.
const OWN_GROWTH: u64 = 256 << 20;

/// The process's peak resident set (`VmHWM`), where `/proc/self/status` tells it.
fn peak_resident() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib << 10)
}

/// `case`'s output after its dispatch over `resources`, encoded by an encoder of its own on
/// `command_buffer`; read after it completes, by the closure returned.
fn encode_own(
    command_buffer: &CommandBuffer,
    pipeline: &metal2vulkan_runtime::ComputePipelineState,
    case: &super::PublicCase,
    resources: &CaseResources,
) -> impl FnOnce() -> Vec<u8> {
    let device = command_buffer.device();
    let buffers = bound_buffers(device, &case.compute, resources).unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encode_case(&mut encoder, pipeline, &case.compute, &buffers, resources).unwrap();
    encoder.end_encoding().unwrap();
    let ComputeOutput::Buffer {
        binding,
        offset,
        length,
    } = case.compute.output
    else {
        panic!("{}: the output is not a buffer", case.case.name);
    };
    move || read_buffer(&buffers[&binding], offset, length).unwrap()
}

/// One command buffer of [`OWN_TEXTURES`] dispatches, each in an encoder of its own, each
/// sampling a texture of its own through the ID its own argument buffer holds (the sampler held
/// beside it, [`HANDLE_SAMPLER`]): the even dispatches' textures hold the case's texels, the odd
/// ones' zero. The dispatches share the command buffer's heap set, and each reads its own texture:
/// the even ones give a lone dispatch's output over the case's texels, the odd ones a lone
/// dispatch's over zero texels, which differ. With every texture made first, the runtime's own
/// allocations over the dispatches (the process's heap at its peak, and the `VkDeviceMemory` it
/// allocated) grow by less than [`OWN_GROWTH`], and on a CPU device, whose memory is all the
/// process's own, the process's peak resident set grows by less than [`PEAK_GROWTH`] where
/// `/proc` tells it (a set per dispatch held lavapipe's whole heap capacity, about 10 MiB, each).
/// A GPU's resident set is printed, not judged: it holds the driver's own memory, which the host's
/// load moves while the runtime's work stays the same (an RTX 3090's grew 128 to 130 MiB alone, and
/// 395 or 396 MiB in 3 of 5 runs beside 48 CPU-bound processes, with every work count of the
/// dispatches equal and the heap's peak 17 MiB in each).
#[test]
fn dispatches_of_their_own_textures_share_one_heap_set() {
    on_device(|device| {
        let case = public_case(HANDLE_SAMPLER).unwrap();
        let pipeline = pipeline_state(device, &case).unwrap();
        let queue = device.new_command_queue().unwrap();
        let base = CaseResources::new(device, &case).unwrap();
        let (&key, literal) = case
            .compute
            .argument_buffer_textures
            .iter()
            .next()
            .expect("an argument-buffer texture");
        let usage = case.argument_texture_usage[&key];
        let mut zero = literal.clone();
        zero.bytes.fill(0);
        let own = |even: bool| {
            let texels = if even { literal } else { &zero };
            let texture = super::texture_with(device, texels, usage).unwrap();
            CaseResources {
                textures: base.textures.clone(),
                samplers: base.samplers.clone(),
                argument_buffers: base.argument_buffers.clone(),
                argument_textures: [(key, texture)].into(),
                argument_samplers: base.argument_samplers.clone(),
                // The case binds no visible-function table, no multisampled texture, no texture
                // buffer an out-of-bounds target guards, and no indirect command buffer, so
                // nothing is filled.
                tables: Default::default(),
                fills: Vec::new(),
                guarded_texture_buffers: Default::default(),
                commands: Default::default(),
            }
        };
        let lone = |even: bool| {
            let command_buffer = queue.command_buffer().unwrap();
            let output = encode_own(&command_buffer, &pipeline, &case, &own(even));
            run(&command_buffer);
            output()
        };
        let (texels, zeros) = (lone(true), lone(false));
        assert_ne!(texels, zeros, "the texels do not tell the textures apart");
        // The dispatches' own textures exist before the first reading, so the growth is the
        // dispatches' alone: made inside the window, a texture per dispatch grew the RTX 3090's
        // peak by 266 MiB.
        let owned: Vec<CaseResources> = (0..OWN_TEXTURES).map(|i| own(i % 2 == 0)).collect();
        let before = peak_resident();
        let memory = device.work_counts().device_memory_allocated;
        let (outputs, heap) = crate::worker::alloc::measure(|| {
            let command_buffer = queue.command_buffer().unwrap();
            let outputs: Vec<_> = owned
                .iter()
                .map(|resources| encode_own(&command_buffer, &pipeline, &case, resources))
                .collect();
            run(&command_buffer);
            outputs
        });
        let memory = device.work_counts().device_memory_allocated - memory;
        let after = peak_resident();
        for (i, output) in outputs.into_iter().enumerate() {
            let expected = if i % 2 == 0 { &texels } else { &zeros };
            assert_eq!(&output(), expected, "dispatch {i}");
        }
        let resident = before
            .zip(after)
            .map(|(before, after)| after.saturating_sub(before));
        let mib =
            |bytes: Option<u64>| bytes.map_or("-".to_owned(), |bytes| (bytes >> 20).to_string());
        eprintln!(
            "runtime heap set: {OWN_TEXTURES} dispatches of their own textures grew the heap's \
             peak by {} MiB, the runtime's device memory by {} MiB, and the peak resident set by \
             {} MiB",
            mib(heap),
            memory >> 20,
            mib(resident),
        );
        if let Some(heap) = heap {
            assert!(heap < OWN_GROWTH, "the heap's peak grew {} MiB", heap >> 20);
        }
        assert!(
            memory < OWN_GROWTH,
            "the runtime allocated {} MiB of device memory",
            memory >> 20
        );
        if let (DeviceType::Cpu, Some(resident)) = (TEST_DEVICE, resident) {
            assert!(
                resident < PEAK_GROWTH,
                "the peak resident set grew {} MiB",
                resident >> 20
            );
        }
    });
}
