//! A probe, not a check: where a command buffer's completed handler waits while another queue's
//! first command buffer makes that queue's index pool block on the program's thread, as the
//! validation lane's `index_pools::a_completed_handler_waits_for_no_pool_block_made_meanwhile`
//! does. Ignored; run by name. Prints `POOL-PROBE` lines.
//!
//! - `runtime`: the lane's draw (`ob3_index_buffer`, on a device without the robustness
//!   features, exact answers, executions recorded), with and without the validation layer; each
//!   round's trace (`command_buffer::completion::trace`) from the making's start to the handler.
//! - `driver`: the making's driver calls made raw on the program's thread (a pool block's
//!   buffer, its buffer of addresses, the memory, the binds, then their release) while another
//!   thread calls `vkGetSemaphoreCounterValue`, `vkWaitSemaphores` of no time and
//!   `vkGetFenceStatus` on objects of its own in a loop: how long each of its calls took while
//!   each driver call of the making ran.

use crate::command_buffer::completion::trace::{self, Traced};
use crate::gpu_pools::PoolKind;
use crate::gpu_time::host_nanos;
use crate::{
    Buffer, CommandBuffer, CommandQueue, Device, DeviceKind, DeviceOptions, PipelineOptions,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, ShaderCacheOptions,
};
use ash::vk;
use metal2vulkan::device::VkFeature;
use metal2vulkan::metal::{IndexType, PrimitiveType};
use metal2vulkan::pipeline::{RenderPipelineDescriptor, StageSource};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::ThreadId;
use std::time::Duration;

const ROUNDS: usize = 20;
const WAIT: Duration = Duration::from_secs(10);

fn device(cache: &Path, layered: bool) -> Device {
    let mut options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        record_executions: true,
        withheld_features: [
            VkFeature::RobustBufferAccess,
            VkFeature::RobustBufferAccess2,
            VkFeature::RobustImageAccess,
            VkFeature::RobustImageAccess2,
        ]
        .into_iter()
        .collect(),
        ..DeviceOptions::default()
    };
    options.policy = options.policy.exact_answers();
    if layered {
        options
            .layers
            .push(c"VK_LAYER_KHRONOS_validation".to_owned());
    }
    Device::system_default_with(&options).unwrap()
}

struct Lane {
    device: Device,
    pipeline: RenderPipelineState,
    seen: Buffer,
    indices: Buffer,
    arguments: Buffer,
}

impl Lane {
    fn new(device: Device) -> Lane {
        let ll = metal2vulkan_devtools::engine_dir().join("ob3_index_buffer.ll");
        let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
        let library = device.new_library_with_compile_library(
            metal2vulkan::library::Library::parse(&bitcode).unwrap(),
        );
        let function = library.new_function_with_name("ob3_index_buffer").unwrap();
        let mut descriptor =
            RenderPipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
        descriptor.rasterization_enabled = false;
        let pipeline = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        let words = |words: &[u32]| -> Vec<u8> {
            words.iter().flat_map(|word| word.to_le_bytes()).collect()
        };
        let shared = ResourceOptions::shared();
        let indices = device
            .new_buffer_with_bytes(&words(&[1, 2]), shared)
            .unwrap();
        let arguments = device
            .new_buffer_with_bytes(&words(&[2, 1, 0, 0, 0]), shared)
            .unwrap();
        let seen = device.new_buffer_with_bytes(&[0; 32], shared).unwrap();
        Lane {
            device,
            pipeline,
            seen,
            indices,
            arguments,
        }
    }

    fn encoded(&self, queue: &CommandQueue) -> CommandBuffer {
        let mut pass = RenderPassDescriptor::new();
        pass.render_target_width = 1;
        pass.render_target_height = 1;
        pass.default_raster_sample_count = 1;
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&self.pipeline).unwrap();
        encoder.set_vertex_buffer(Some(&self.seen), 0, 0).unwrap();
        encoder
            .draw_indexed_primitives_indirect(
                PrimitiveType::Point,
                IndexType::UInt32,
                &self.indices,
                0,
                &self.arguments,
                0,
            )
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer
    }
}

fn ms(at: u64, from: u64) -> f64 {
    (at as f64 - from as f64) / 1e6
}

#[test]
#[ignore = "a probe: run by name"]
fn pool_making_probe() {
    let cache = std::env::temp_dir().join(format!("m2v-pool-probe-{}", std::process::id()));
    std::fs::create_dir_all(&cache).unwrap();
    for layered in [true, false] {
        let lane = Lane::new(device(&cache, layered));
        driver(&lane.device, layered);
        runtime(&lane, layered);
        driver(&lane.device, layered);
    }
    std::fs::remove_dir_all(&cache).unwrap();
}

fn runtime(lane: &Lane, layered: bool) {
    let program = std::thread::current().id();
    let queue = lane.device.new_command_queue().unwrap();
    let warm = lane.encoded(&queue);
    warm.commit().unwrap();
    warm.wait_until_completed_for(WAIT).unwrap();
    for round in 0..ROUNDS {
        let other = lane.device.new_command_queue().unwrap();
        let first = lane.encoded(&queue);
        let ran: Arc<Mutex<Option<(u64, ThreadId)>>> = Arc::default();
        let log = Arc::clone(&ran);
        first
            .add_completed_handler(Box::new(move |_| {
                *log.lock().unwrap() = Some((host_nanos(), std::thread::current().id()));
            }))
            .unwrap();
        trace::start();
        let commit = host_nanos();
        first.commit().unwrap();
        let start = host_nanos();
        let second = lane.encoded(&other);
        let encoded = host_nanos();
        second.commit().unwrap();
        first.wait_until_completed_for(WAIT).unwrap();
        second.wait_until_completed_for(WAIT).unwrap();
        let steps = trace::stop();
        let (handled, handler) = ran.lock().unwrap().expect("the completed handler ran");
        let gpu_end = (first.gpu_end_time().unwrap() * 1e9) as u64;
        let mut threads: BTreeMap<ThreadId, String> = BTreeMap::new();
        threads.insert(program, "P".into());
        threads.insert(handler, "H".into());
        let mut name = |thread: ThreadId| {
            let next = threads.len();
            threads
                .entry(thread)
                .or_insert_with(|| format!("T{next}"))
                .clone()
        };
        let mut events: Vec<(u64, String)> = Vec::new();
        for step in &steps {
            match *step {
                Traced::Watched { queue, value, at } => {
                    events.push((at, format!("watched q{queue} v{value}")));
                }
                Traced::Waited {
                    queue,
                    value,
                    entered,
                    returned,
                } => {
                    events.push((
                        returned,
                        format!("waited q{queue} v{value} from {:+.3}", ms(entered, start)),
                    ));
                }
                Traced::Finished { queue, value, seen } => {
                    events.push((seen, format!("finished q{queue} v{value}")));
                }
                Traced::Step { what, at } => events.push((at, format!("step {what}"))),
                Traced::Submitting { what, at } => {
                    events.push((at, format!("submitting {what}")));
                }
                Traced::Mark { what, at, thread } => {
                    events.push((at, format!("{} {what}", name(thread))));
                }
            }
        }
        events.push((gpu_end, "GPU END (first)".into()));
        events.push((handled, "HANDLER (first)".into()));
        events.push((encoded, "P encoded (second)".into()));
        events.push((commit, "P commit (first)".into()));
        events.sort_by_key(|(at, _)| *at);
        let lines: Vec<String> = (events.iter())
            .map(|(at, what)| format!("{:+.3} {what}", ms(*at, start)))
            .collect();
        eprintln!(
            "POOL-PROBE runtime layered={layered} round={round} making={:.3} gpu_end={:+.3} \
             handler={:+.3} lag={:.3} :: {}",
            ms(encoded, start),
            ms(gpu_end, start),
            ms(handled, start),
            ms(handled, gpu_end),
            lines.join(" | ")
        );
        drop((first, second, other));
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// One call of the sampling thread: which, entered and returned.
type Sampled = (&'static str, u64, u64);

fn driver(device: &Device, layered: bool) {
    let shared = device.shared();
    let raw = shared.vk().clone();
    let properties =
        unsafe { (shared.instance.raw).get_physical_device_memory_properties(shared.physical) };
    let addressed = (device.compile_device().profile().enabled().features)
        .contains(&VkFeature::BufferDeviceAddress);
    let bytes = shared.gpu_pools.block_bytes(PoolKind::Index);
    let mut kind = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    let semaphore = unsafe {
        raw.create_semaphore(
            &vk::SemaphoreCreateInfo::default().push_next(&mut kind),
            None,
        )
    }
    .unwrap();
    let fence = unsafe { raw.create_fence(&vk::FenceCreateInfo::default(), None) }.unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let sampler = {
        let (raw, stop) = (raw.clone(), Arc::clone(&stop));
        std::thread::spawn(move || {
            let mut sampled: Vec<Sampled> = Vec::new();
            while !stop.load(Ordering::Relaxed) {
                let entered = host_nanos();
                let _ = unsafe { raw.get_semaphore_counter_value(semaphore) };
                sampled.push(("counter", entered, host_nanos()));
                let semaphores = [semaphore];
                let values = [1];
                let info = vk::SemaphoreWaitInfo::default()
                    .semaphores(&semaphores)
                    .values(&values);
                let entered = host_nanos();
                let _ = unsafe { raw.wait_semaphores(&info, 0) };
                sampled.push(("wait0", entered, host_nanos()));
                let entered = host_nanos();
                let _ = unsafe { raw.get_fence_status(fence) };
                sampled.push(("fence", entered, host_nanos()));
                std::thread::sleep(Duration::from_micros(50));
            }
            sampled
        })
    };
    std::thread::sleep(Duration::from_millis(50));
    let mut phases: Vec<(&'static str, u64, u64)> = Vec::new();
    let mut timed = |what: &'static str, call: &mut dyn FnMut()| {
        let entered = host_nanos();
        call();
        phases.push((what, entered, host_nanos()));
    };
    for _ in 0..10 {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER;
        let info = vk::BufferCreateInfo::default().size(bytes).usage(usage);
        let mut buffer = vk::Buffer::null();
        timed("create buffer", &mut || {
            buffer = unsafe { raw.create_buffer(&info, None) }.unwrap();
        });
        let requirements = unsafe { raw.get_buffer_memory_requirements(buffer) };
        let mut addresses = vk::Buffer::null();
        if addressed {
            let info = vk::BufferCreateInfo::default().size(bytes).usage(
                usage
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            );
            timed("create addresses", &mut || {
                addresses = unsafe { raw.create_buffer(&info, None) }.unwrap();
            });
        }
        let count = properties.memory_type_count as usize;
        let local = |index: usize| {
            let flags = properties.memory_types[index].property_flags;
            requirements.memory_type_bits & (1 << index) != 0
                && flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                && !flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        };
        let type_index = (0..count)
            .find(|&index| local(index))
            .or_else(|| (0..count).find(|&index| requirements.memory_type_bits & (1 << index) != 0))
            .unwrap() as u32;
        let mut flags = vk::MemoryAllocateFlagsInfo::default().flags(match addressed {
            true => vk::MemoryAllocateFlags::DEVICE_ADDRESS,
            false => vk::MemoryAllocateFlags::empty(),
        });
        let info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(type_index)
            .push_next(&mut flags);
        let mut memory = vk::DeviceMemory::null();
        timed("allocate", &mut || {
            memory = unsafe { raw.allocate_memory(&info, None) }.unwrap();
        });
        if addressed {
            timed("bind addresses", &mut || {
                unsafe { raw.bind_buffer_memory(addresses, memory, 0) }.unwrap();
            });
            timed("address", &mut || {
                let info = vk::BufferDeviceAddressInfo::default().buffer(addresses);
                let _ = unsafe { raw.get_buffer_device_address(&info) };
            });
        }
        timed("bind buffer", &mut || {
            unsafe { raw.bind_buffer_memory(buffer, memory, 0) }.unwrap();
        });
        std::thread::sleep(Duration::from_millis(5));
        timed("destroy buffer", &mut || unsafe {
            raw.destroy_buffer(buffer, None);
        });
        if addressed {
            timed("destroy addresses", &mut || unsafe {
                raw.destroy_buffer(addresses, None);
            });
        }
        timed("free", &mut || unsafe { raw.free_memory(memory, None) });
        std::thread::sleep(Duration::from_millis(20));
    }
    std::thread::sleep(Duration::from_millis(50));
    stop.store(true, Ordering::Relaxed);
    let sampled = sampler.join().unwrap();
    unsafe {
        raw.destroy_semaphore(semaphore, None);
        raw.destroy_fence(fence, None);
    }
    // Calls outside every phase: the baseline.
    let inside = |(_, entered, returned): &Sampled| {
        (phases.iter()).any(|&(_, from, to)| *entered < to && *returned > from)
    };
    let mut outside: Vec<u64> = (sampled.iter())
        .filter(|call| !inside(call))
        .map(|&(_, entered, returned)| returned - entered)
        .collect();
    outside.sort_unstable();
    let quantile = |sorted: &[u64], q: f64| -> f64 {
        sorted
            .get(((sorted.len() as f64 - 1.0) * q) as usize)
            .map_or(0.0, |&nanos| nanos as f64 / 1e6)
    };
    eprintln!(
        "POOL-PROBE driver layered={layered} bytes={bytes} addressed={addressed} samples={} \
         outside: n={} p50={:.3} p99={:.3} max={:.3}",
        sampled.len(),
        outside.len(),
        quantile(&outside, 0.5),
        quantile(&outside, 0.99),
        quantile(&outside, 1.0)
    );
    let mut by_phase: BTreeMap<&'static str, Vec<String>> = BTreeMap::new();
    for &(what, from, to) in &phases {
        // The longest sampled call that overlapped this phase, and how far past the phase's
        // start it returned.
        let longest = (sampled.iter())
            .filter(|&&(_, entered, returned)| entered < to && returned > from)
            .max_by_key(|&&(_, entered, returned)| returned - entered);
        let text = match longest {
            Some(&(call, entered, returned)) => format!(
                "{:.3}ms[{call} {:.3}ms, returned {:+.3} of end]",
                ms(to, from),
                ms(returned, entered),
                ms(returned, to)
            ),
            None => format!("{:.3}ms[no sample]", ms(to, from)),
        };
        by_phase.entry(what).or_default().push(text);
    }
    for (what, texts) in by_phase {
        eprintln!(
            "POOL-PROBE driver layered={layered} {what}: {}",
            texts.join(" ")
        );
    }
}
